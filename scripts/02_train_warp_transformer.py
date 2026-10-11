"""
02_train_warp_transformer.py — Pretrain ChromaWarpTransformer on synthetic two-batch datasets
with known warps. No real fish-oil data and no TIC correlation are used.

Each iteration builds a mini-dataset of 4–16 chromatograms:
  1. Draw N_COMP compounds from the MoNA training split of data/pretraining/spectra.h5
     (01_download_pretrain_data.py). Stored spectra are sqrt + L2; they are squared back to
     intensities so the chromatogram's sqrt + L2 per bin is applied exactly once, as for the
     real data.
  2. Place them at random consensus positions; draw a per-compound peak width
     σ ~ U(PEAK_SIGMA_RANGE) bins (after apex-scan binning real peaks span about one bin).
  3. Batch 0: per-compound jitter (±WITHIN_DRIFT bins). Batch 1: the same jitter plus a global
     offset of MIN_DRIFT … MAX_DRIFT_BINS bins in a random direction. Elution order is kept.
  4. Target warp of each sample (consensus bin t → source bin): linear through the
     (consensus, sample) peak positions and continued with slope 1 beyond the outer peaks,
     i.e. the outer peaks' offsets carry on to the ends of the run. It is not clipped: the
     model's warps may point past the run's edge (apply_warp clamps where it reads), and
     clipping would make the target flat near the ends again.
  5. Loss = MSE of predicted vs target warp (in units of the run length) + λ · smoothness.

torch and numpy are seeded, so the checkpoint is reproducible on a given device.

Output: --out (default checkpoints/warp_transformer_v2.pt) and <out>.txt with the settings,
the spectral library used and the final loss.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from calibration import pick_device, set_seed                               # noqa: E402
from config import DEFAULT_CKPT, MZ_BINS, N_BINS, PRETRAIN_H5               # noqa: E402
from warp_transformer import ChromaWarpTransformer, warp_smoothness_loss    # noqa: E402

N_COMP           = 30            # compounds per synthetic chromatogram
PEAK_SIGMA_RANGE = (0.5, 1.5)    # bins, per compound
PEAK_HALF_WIDTH  = 8             # bins of the Gaussian kernel either side of the apex
MIN_SPACING      = 6             # bins between consensus peak positions
WITHIN_DRIFT     = 4             # ±bins (±0.9 min), per compound and sample
MIN_DRIFT        = 5             # bins, minimum between-batch offset
MAX_DRIFT_BINS   = 25            # bins (±5.6 min), maximum between-batch offset
NOISE_SCALE      = 0.005         # exponential baseline noise (on intensities)
N_ITERATIONS     = 10_000
LOG_EVERY        = 200
LAMBDA_SMOOTH    = 0.01


# ── Spectral library ──────────────────────────────────────────────────────────

def load_library(path: Path) -> np.ndarray:
    """Training-split spectra as L2-normalised intensities [n, M]. The file stores sqrt + L2
    spectra; squaring undoes the sqrt."""
    with h5py.File(path, 'r') as f:
        train = np.array(f['split']).astype(str) == 'train'
        specs = np.asarray(f['spectra'][train], dtype=np.float64)
    specs = specs ** 2
    norms = np.linalg.norm(specs, axis=1, keepdims=True)
    return (specs / np.where(norms > 0, norms, 1.0)).astype(np.float32)


# ── Synthetic data ────────────────────────────────────────────────────────────

def random_positions(n: int, rng: np.random.Generator, margin: int = 10) -> np.ndarray:
    """Up to n sorted positions in [margin, N_BINS − margin), at least MIN_SPACING apart."""
    pool = np.arange(margin, N_BINS - margin)
    rng.shuffle(pool)
    chosen: list[int] = []
    for c in pool.tolist():
        if all(abs(c - p) >= MIN_SPACING for p in chosen):
            chosen.append(c)
            if len(chosen) == n:
                break
    return np.array(sorted(chosen), dtype=np.int64)


def build_chromatogram(compounds: np.ndarray, positions: np.ndarray, heights: np.ndarray,
                       sigmas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """[T, M] chromatogram: Gaussian elution profiles of the compounds' intensity spectra plus
    exponential noise, then sqrt and L2 per bin (as 03_preprocess does for real data)."""
    offsets = np.arange(-PEAK_HALF_WIDTH, PEAK_HALF_WIDTH + 1)
    bins = positions[:, None] + offsets[None, :]
    weights = heights[:, None] * np.exp(-0.5 * (offsets[None, :] / sigmas[:, None]) ** 2)
    profile = np.zeros((N_BINS, len(compounds)), dtype=np.float32)
    valid = (bins >= 0) & (bins < N_BINS)
    comp = np.broadcast_to(np.arange(len(compounds))[:, None], bins.shape)
    np.add.at(profile, (bins[valid], comp[valid]), weights[valid])
    chroma = profile @ compounds
    chroma += rng.exponential(NOISE_SCALE, size=chroma.shape).astype(np.float32)
    chroma = np.sqrt(np.clip(chroma, 0.0, None))
    norms = np.linalg.norm(chroma, axis=1, keepdims=True)
    return (chroma / np.where(norms > 1e-8, norms, 1.0)).astype(np.float32)


def drifted_positions(pks_ref: np.ndarray, offset: int, within: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Consensus positions + offset + per-compound jitter, kept strictly increasing and inside
    the run."""
    pks = pks_ref + offset + rng.integers(-within, within + 1, size=len(pks_ref))
    pks = np.clip(pks, 2, N_BINS - 3)
    for i in range(1, len(pks)):
        pks[i] = max(pks[i], pks[i - 1] + 1)
    if pks[-1] > N_BINS - 3:                       # pushed off the end: compress from the back
        pks[-1] = N_BINS - 3
        for i in range(len(pks) - 2, -1, -1):
            pks[i] = min(pks[i], pks[i + 1] - 1)
    return pks


def target_warp(pks_ref: np.ndarray, pks_samp: np.ndarray) -> np.ndarray:
    """Consensus bin t → source bin: linear through the peak pairs, slope 1 beyond the outer
    peaks. Not clipped."""
    t  = np.arange(N_BINS, dtype=np.float64)
    xp = np.concatenate([[pks_ref[0] - N_BINS], pks_ref, [pks_ref[-1] + N_BINS]])
    fp = np.concatenate([[pks_samp[0] - N_BINS], pks_samp, [pks_samp[-1] + N_BINS]])
    return np.interp(t, xp, fp).astype(np.float32)


def mini_dataset(library: np.ndarray, n_samples: int, rng: np.random.Generator):
    """(chromas [N, T, M], target warps [N, T], study ids [N]) or None if too few compounds fit."""
    pks_ref = random_positions(N_COMP, rng)
    if len(pks_ref) < 5:
        return None
    compounds = library[rng.choice(len(library), len(pks_ref), replace=False)]
    sigmas = rng.uniform(*PEAK_SIGMA_RANGE, size=len(pks_ref))
    offset = int(rng.integers(MIN_DRIFT, MAX_DRIFT_BINS + 1)) * (1 if rng.random() > 0.5 else -1)
    n_a = max(1, n_samples // 2)
    chromas, warps, sids = [], [], []
    for study, n, off in ((0, n_a, 0), (1, n_samples - n_a, offset)):
        for _ in range(n):
            pks = drifted_positions(pks_ref, off, WITHIN_DRIFT, rng)
            heights = rng.uniform(0.3, 2.5, len(pks_ref))
            chromas.append(build_chromatogram(compounds, pks, heights, sigmas, rng))
            warps.append(target_warp(pks_ref, pks))
            sids.append(study)
    return np.stack(chromas), np.stack(warps), np.array(sids, dtype=np.int64)


# ── Training ──────────────────────────────────────────────────────────────────

def train(library: np.ndarray, device: torch.device, n_iterations: int, seed: int,
          lr: float = 3e-4) -> tuple[ChromaWarpTransformer, list[float]]:
    """Returns the trained model and the mean loss of every LOG_EVERY-iteration window."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = ChromaWarpTransformer(mz_bins=MZ_BINS, n_bins=N_BINS).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iterations)
    window, history = [], []
    model.train()
    for it in range(1, n_iterations + 1):
        data = mini_dataset(library, int(rng.integers(4, 17)), rng)
        if data is None:
            continue
        x, target, sid = (torch.from_numpy(v).to(device) for v in data)
        _, pred = model(x, sid)
        loss = (F.mse_loss(pred / (N_BINS - 1), target / (N_BINS - 1))
                + LAMBDA_SMOOTH * warp_smoothness_loss(pred))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        window.append(float(loss))
        if it % LOG_EVERY == 0 or it == n_iterations:
            history.append(float(np.mean(window)))
            window = []
            print(f'  iter {it:>6d}/{n_iterations}  loss {history[-1]:.5f}', flush=True)
    return model, history


def main(h5: Path, out: Path, n_iterations: int, seed: int, device_name: str) -> list[float]:
    device = pick_device(device_name)
    library = load_library(h5)
    print(f'Device {device}; library {h5}: {len(library)} training spectra', flush=True)
    t0 = time.perf_counter()
    model, history = train(library, device, n_iterations, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    report = [f'checkpoint      {out}', f'library         {h5} ({len(library)} training spectra)',
              f'iterations      {n_iterations}   seed {seed}   device {device}   '
              f'{time.perf_counter() - t0:.0f} s',
              f'compounds {N_COMP}, peak σ {PEAK_SIGMA_RANGE} bins, within ±{WITHIN_DRIFT}, '
              f'offset {MIN_DRIFT}–{MAX_DRIFT_BINS} bins, λ_smooth {LAMBDA_SMOOTH}',
              f'final loss      {history[-1]:.5f} (mean over the last logging window)',
              'loss per window ' + ' '.join(f'{v:.5f}' for v in history)]
    out.with_suffix('.txt').write_text('\n'.join(report) + '\n')
    print('\n'.join(report))
    return history


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h5', type=Path, default=PRETRAIN_H5, help='spectral library (from 01)')
    ap.add_argument('--out', type=Path, default=DEFAULT_CKPT)
    ap.add_argument('--iterations', type=int, default=N_ITERATIONS)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    main(args.h5, args.out, args.iterations, args.seed, args.device)
