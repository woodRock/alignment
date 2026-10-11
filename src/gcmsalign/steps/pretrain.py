"""Step 02 — pretrain the WarpTransformer on synthetic two-batch datasets with known warps.
No real fish-oil data and no TIC correlation are used.

Each iteration builds a mini-dataset of 4–16 chromatograms:
  1. Draw compounds from the training split of the step-01 library. Stored spectra are
     sqrt + L2; they are squared back to intensities, so the chromatogram's sqrt + L2 per bin
     is applied exactly once, as for the real data.
  2. Place them at random consensus positions, each with its own peak width
     σ ~ U(sigma_range) bins (after apex-scan binning real peaks span about one bin).
  3. Batch 0: per-compound jitter. Batch 1: the same jitter plus a global offset of
     min_offset … max_offset bins in a random direction. Elution order is kept.
  4. Target warp of each sample (consensus bin → source bin): linear through the peak pairs
     and continued with slope 1 beyond the outer peaks. Not clipped: the model's warps may point
     past the run's edge (apply_warp clamps where it reads), and clipping would make the target
     flat near the ends again.
  5. Loss = MSE of predicted vs target warp (in run lengths) + λ · smoothness.
torch and numpy are seeded, so the checkpoint is reproducible on a given device.

Output: --out (default <root>/checkpoints/warp_transformer_v2.pt) and a .txt report beside it
with the settings, the library used and the loss history.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..calibration import pick_device, set_seed
from ..config import MZ_BINS, N_BINS, Paths
from ..warp_transformer import ChromaWarpTransformer, warp_smoothness_loss
from ._common import parser, setup_logging, write_report

log = logging.getLogger('gcmsalign')


@dataclass(frozen=True)
class SyntheticConfig:
    n_compounds: int = 30
    sigma_range: tuple[float, float] = (0.5, 1.5)  # bins, per compound
    kernel_half_width: int = 8  # bins either side of the apex
    min_spacing: int = 6  # bins between consensus positions
    margin: int = 10  # bins kept free at both ends
    within_jitter: int = 4  # ±bins per compound and sample
    min_offset: int = 5  # bins, between batches
    max_offset: int = 25  # bins (±5.6 min)
    noise_scale: float = 0.005  # exponential noise on intensities
    n_samples: tuple[int, int] = (4, 16)


@dataclass(frozen=True)
class TrainConfig:
    iterations: int = 10_000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lambda_smooth: float = 0.01
    log_every: int = 200
    seed: int = 42


DEFAULT_SYNTH = SyntheticConfig()
DEFAULT_TRAIN = TrainConfig()


# ── Spectral library ──────────────────────────────────────────────────────────


def load_library(path: Path) -> np.ndarray:
    """Training-split spectra as L2-normalised intensities [n, M] (the stored sqrt is undone)."""
    with h5py.File(path, 'r') as f:
        train = np.array(f['split']).astype(str) == 'train'
        specs = np.asarray(f['spectra'][train], dtype=np.float64) ** 2
    norms = np.linalg.norm(specs, axis=1, keepdims=True)
    return (specs / np.where(norms > 0, norms, 1.0)).astype(np.float32)


# ── Synthetic data ────────────────────────────────────────────────────────────


def random_positions(rng: np.random.Generator, cfg: SyntheticConfig) -> np.ndarray:
    """Up to n_compounds sorted positions in [margin, N_BINS − margin), min_spacing apart."""
    pool = np.arange(cfg.margin, N_BINS - cfg.margin)
    rng.shuffle(pool)
    chosen: list[int] = []
    for c in pool.tolist():
        if all(abs(c - p) >= cfg.min_spacing for p in chosen):
            chosen.append(c)
            if len(chosen) == cfg.n_compounds:
                break
    return np.array(sorted(chosen), dtype=np.int64)


def build_chromatogram(
    compounds: np.ndarray,
    positions: np.ndarray,
    heights: np.ndarray,
    sigmas: np.ndarray,
    rng: np.random.Generator,
    cfg: SyntheticConfig,
) -> np.ndarray:
    """[T, M]: Gaussian elution profiles of intensity spectra + exponential noise, then sqrt
    and L2 per bin (as step 03 does for real data)."""
    offsets = np.arange(-cfg.kernel_half_width, cfg.kernel_half_width + 1)
    bins = positions[:, None] + offsets[None, :]
    weights = heights[:, None] * np.exp(-0.5 * (offsets[None, :] / sigmas[:, None]) ** 2)
    profile = np.zeros((N_BINS, len(compounds)), dtype=np.float32)
    valid = (bins >= 0) & (bins < N_BINS)
    comp = np.broadcast_to(np.arange(len(compounds))[:, None], bins.shape)
    np.add.at(profile, (bins[valid], comp[valid]), weights[valid])
    chroma = profile @ compounds
    if cfg.noise_scale > 0:
        chroma += rng.exponential(cfg.noise_scale, size=chroma.shape).astype(np.float32)
    chroma = np.sqrt(np.clip(chroma, 0.0, None))
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    return (chroma / np.where(norms > 1e-8, norms, 1.0)).astype(np.float32)


def drifted_positions(
    pks_ref: np.ndarray, offset: int, jitter: int, rng: np.random.Generator
) -> np.ndarray:
    """Consensus positions + offset + per-compound jitter; strictly increasing, inside the run."""
    pks = np.clip(
        pks_ref + offset + rng.integers(-jitter, jitter + 1, size=len(pks_ref)), 2, N_BINS - 3
    )
    for i in range(1, len(pks)):
        pks[i] = max(pks[i], pks[i - 1] + 1)
    if pks[-1] > N_BINS - 3:  # pushed off the end: compress from the back
        pks[-1] = N_BINS - 3
        for i in range(len(pks) - 2, -1, -1):
            pks[i] = min(pks[i], pks[i + 1] - 1)
    return pks


def target_warp(pks_ref: np.ndarray, pks_samp: np.ndarray) -> np.ndarray:
    """Consensus bin → source bin: linear through the peak pairs, slope 1 beyond the outer
    peaks. Not clipped."""
    t = np.arange(N_BINS, dtype=np.float64)
    xp = np.concatenate([[pks_ref[0] - N_BINS], pks_ref, [pks_ref[-1] + N_BINS]])
    fp = np.concatenate([[pks_samp[0] - N_BINS], pks_samp, [pks_samp[-1] + N_BINS]])
    return np.interp(t, xp, fp).astype(np.float32)


def mini_dataset(
    library: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    cfg: SyntheticConfig = DEFAULT_SYNTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(chromas [N, T, M], target warps [N, T], study ids [N]), or None if < 5 compounds fit."""
    pks_ref = random_positions(rng, cfg)
    if len(pks_ref) < 5:
        return None
    compounds = library[rng.choice(len(library), len(pks_ref), replace=False)]
    sigmas = rng.uniform(*cfg.sigma_range, size=len(pks_ref))
    sign = 1 if rng.random() > 0.5 else -1
    offset = sign * int(rng.integers(cfg.min_offset, cfg.max_offset + 1))
    n_a = max(1, n_samples // 2)
    chromas, warps, sids = [], [], []
    for study, n, off in ((0, n_a, 0), (1, n_samples - n_a, offset)):
        for _ in range(n):
            pks = drifted_positions(pks_ref, off, cfg.within_jitter, rng)
            heights = rng.uniform(0.3, 2.5, len(pks_ref))
            chromas.append(build_chromatogram(compounds, pks, heights, sigmas, rng, cfg))
            warps.append(target_warp(pks_ref, pks))
            sids.append(study)
    return np.stack(chromas), np.stack(warps), np.array(sids, dtype=np.int64)


# ── Training ──────────────────────────────────────────────────────────────────


def train(
    library: np.ndarray,
    device: torch.device,
    cfg: TrainConfig = DEFAULT_TRAIN,
    data_cfg: SyntheticConfig = DEFAULT_SYNTH,
) -> tuple[ChromaWarpTransformer, list[float]]:
    """The trained model and the mean loss of every log_every-iteration window."""
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = ChromaWarpTransformer(mz_bins=MZ_BINS, n_bins=N_BINS).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.iterations)
    window: list[float] = []
    history: list[float] = []
    model.train()
    for it in range(1, cfg.iterations + 1):
        lo, hi = data_cfg.n_samples
        data = mini_dataset(library, int(rng.integers(lo, hi + 1)), rng, data_cfg)
        if data is not None:
            x, target, sid = (torch.from_numpy(v).to(device) for v in data)
            _, pred = model(x, sid)
            loss = F.mse_loss(
                pred / (N_BINS - 1), target / (N_BINS - 1)
            ) + cfg.lambda_smooth * warp_smoothness_loss(pred)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            window.append(float(loss))
        sched.step()
        if (it % cfg.log_every == 0 or it == cfg.iterations) and window:
            history.append(float(np.mean(window)))
            window = []
            log.info('  iter %6d/%d  loss %.5f', it, cfg.iterations, history[-1])
    return model, history


def format_report(
    out: Path,
    h5: Path,
    n_library: int,
    cfg: TrainConfig,
    data_cfg: SyntheticConfig,
    device: torch.device,
    seconds: float,
    history: list[float],
) -> list[str]:
    return [
        f'checkpoint  {out}',
        f'library     {h5} ({n_library} training spectra)',
        f'device      {device}   {seconds:.0f} s',
        f'training    {asdict(cfg)}',
        f'synthetic   {asdict(data_cfg)}',
        f'final loss  {history[-1]:.5f} (mean over the last logging window)',
        'history     ' + ' '.join(f'{v:.5f}' for v in history),
    ]


def run(
    h5: Path,
    out: Path,
    device_name: str = 'auto',
    cfg: TrainConfig = DEFAULT_TRAIN,
    data_cfg: SyntheticConfig = DEFAULT_SYNTH,
) -> list[float]:
    device = pick_device(device_name)
    library = load_library(h5)
    log.info('Device %s; library %s: %d training spectra', device, h5, len(library))
    t0 = time.perf_counter()
    model, history = train(library, device, cfg, data_cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    write_report(
        out.with_suffix('.txt'),
        format_report(
            out, h5, len(library), cfg, data_cfg, device, time.perf_counter() - t0, history
        ),
    )
    return history


def cli(argv: list[str] | None = None) -> None:
    ap = parser(__doc__)
    ap.add_argument('--h5', type=Path, default=None, help='library (default: step 01 output)')
    ap.add_argument(
        '--out',
        type=Path,
        default=None,
        help='checkpoint (default <root>/checkpoints/warp_transformer_v2.pt)',
    )
    ap.add_argument('--iterations', type=int, default=TrainConfig.iterations)
    ap.add_argument('--seed', type=int, default=TrainConfig.seed)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args(argv)
    setup_logging()
    paths = Paths(args.root)
    run(
        args.h5 or paths.pretrain_h5,
        args.out or paths.default_ckpt,
        args.device,
        TrainConfig(iterations=args.iterations, seed=args.seed),
    )
