"""
06_align_testtime_calib.py — Anchor-supervised test-time calibration.

Motivation
----------
The cross-batch fine-tuned model (archive/15_finetune_cross_batch.py) applies the wrong warp magnitude
because the training pair (jan2016 vs apr2016) may have had different lab
conditions from the evaluation pair (sep2015 vs jul2016).  The model cannot
know the correct magnitude without seeing the actual evaluation data.

Calibration strategy
--------------------
The same spectral cosine matching used by PCHIP already locates ~4-5 anchor
pairs (ref_bin, query_bin) where the same compound is confidently identified
in both batches.  These anchors tell us:
  - The correct warp DIRECTION for this specific experiment
  - The correct warp MAGNITUDE at those ~4-5 RT positions

We use these as SPARSE supervision — 4-5 points out of 200 — to run ~200
gradient steps that calibrate the transformer's warp without using TIC
correlation at all (no leakage of the evaluation metric).

The transformer then INTERPOLATES between anchors using the full 200×1000
chromatographic context.  PCHIP is limited to a cubic spline; the transformer
can use every m/z peak in every sample to decide how to fill in the gaps.

No leakage
----------
The supervision signal is spectral cosine similarity (peak-matching), which is
the same signal PCHIP uses.  TIC r is never used during calibration — only
during the final evaluation.

Flow
----
  warp_transformer_crossbatch.pt
        ↓ 200 calibration steps (anchor MSE + smoothness)
  calibrated model
        ↓ evaluate on sep2015 vs jul2016
  TIC r, study silhouette
"""

from __future__ import annotations

import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from warp_transformer import (ChromaWarpTransformer, apply_warp, warp_smoothness_loss,
                              enable_cuda_determinism)
from alignment import (load_chromas, load_tic_raw, align_pair, warp_chroma_2d, warp_tic,
                       pooled_anchor_pchip, pearson_rows, TIME_AXIS)

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

N_BINS        = 200
BIN_MIN       = 45.0 / N_BINS
N_CALIB_STEPS = 500
LR_CALIB      = 5e-4
LAMBDA_ANCHOR = 1.0
LAMBDA_SMOOTH = 0.05
N_REF_PAIRS   = 3    # number of study-A samples to use for anchor extraction
OUTLIER_IQR   = 1.5  # filter anchors with drift > OUTLIER_IQR * IQR from median


# ── Anchor extraction ─────────────────────────────────────────────────────────

def extract_anchors(chromas_a: list, chromas_b: list,
                    n_ref: int = N_REF_PAIRS) -> list[tuple[int, int]]:
    """
    Run PCHIP peak-matching on n_ref pairs of (study-A sample, study-B[0])
    and return RANSAC-filtered (ref_bin, query_bin) anchor pairs.

    Aggregates across pairs by taking the median query_bin when multiple
    pairs agree on the same reference bin position.  This suppresses
    within-batch noise in individual sample anchor estimates.
    """
    ref_query_per_rbin: dict[int, list[int]] = defaultdict(list)

    query_sample = chromas_b[0]
    ref_indices  = list(range(min(n_ref, len(chromas_a))))

    n_found = 0
    for i in ref_indices:
        _, n_a, _, _, _, anchors = align_pair(
            query_sample, chromas_a[i], encode_fn=None, return_anchors=True
        )
        n_found += n_a
        for r_bin, q_bin in anchors:
            ref_query_per_rbin[r_bin].append(q_bin)

    if not ref_query_per_rbin:
        return []

    # Aggregate: median query_bin per reference bin position
    aggregated = []
    for r_bin in sorted(ref_query_per_rbin):
        q_bins = ref_query_per_rbin[r_bin]
        q_med  = int(np.round(np.median(q_bins)))
        aggregated.append((r_bin, q_med))

    # Filter outlier anchors whose drift deviates from the consensus direction.
    # IQR filtering is too loose when drift variance is high (e.g. -22 to +7 gives
    # IQR=15 bins → threshold=22.5 keeps the +7 outlier).  σ-based filtering is
    # stricter and correctly removes direction-reversed misidentifications.
    drifts = np.array([q - r for r, q in aggregated], dtype=float)
    mean   = float(np.mean(drifts))
    std    = float(np.std(drifts)) if len(drifts) > 1 else 1.0
    std    = max(std, 1.0)   # avoid collapsing to 0 when all equal
    keep   = np.abs(drifts - mean) <= OUTLIER_IQR * std
    aggregated = [a for a, k in zip(aggregated, keep) if k]

    return aggregated


# ── Calibration loss ──────────────────────────────────────────────────────────

def anchor_loss(warps: torch.Tensor, study_ids: torch.Tensor,
                anchor_pairs: list[tuple[int, int]],
                device: torch.device) -> torch.Tensor:
    """
    Relative warp supervision: the inter-batch warp offset at each anchor
    position must match the measured spectral drift.

    Instead of anchoring study-1 to absolute bin positions (which fights the
    joint consensus-frame alignment), we supervise the DIFFERENCE:
        warp_study1[r_bin] - mean(warp_study0)[r_bin]  ≈  q_bin - r_bin

    This lets the model put the consensus frame wherever it wants, while
    ensuring the cross-batch offset at each anchor position is correct.

    warps       : [N, T]
    study_ids   : [N] int64 — 0=reference, 1=query
    anchor_pairs: [(r_bin, q_bin), ...]  — spectral-cosine RANSAC anchors
    """
    if not anchor_pairs:
        return torch.tensor(0.0, device=device)

    ref_mean    = warps[study_ids == 0].mean(dim=0)   # [T] — mean reference warp
    query_warps = warps[study_ids == 1]               # [n_b, T]
    loss = torch.tensor(0.0, device=device)
    for r_bin, q_bin in anchor_pairs:
        drift    = float(q_bin - r_bin)
        rel_warp = query_warps[:, r_bin] - ref_mean[r_bin]   # [n_b]
        loss     = loss + (rel_warp - drift).pow(2).mean()
    return loss / len(anchor_pairs)


# ── Metrics ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    enable_cuda_determinism()


def tic_r_per_pair(chromas_a: list, chromas_b: list) -> np.ndarray:
    """Pearson TIC r for every (A_i, B_j), i-major: index i * n_b + j (same order as script 05)."""
    return np.array([float(pearsonr(a.sum(1), b.sum(1))[0])
                     for a in chromas_a for b in chromas_b])


def warp_tics_bins(tics: np.ndarray, warps: np.ndarray) -> np.ndarray:
    """Apply [N, T] WarpTransformer warps (source bins) to 1-D TICs exactly as apply_warp does."""
    out = apply_warp(torch.from_numpy(np.asarray(tics, dtype=np.float32)[:, :, None]),
                     torch.from_numpy(np.asarray(warps, dtype=np.float32)))
    return out[:, :, 0].numpy()


def tic_correlations(chromas_a: list, chromas_b: list) -> float:
    return float(np.mean(tic_r_per_pair(chromas_a, chromas_b)))


def study_silhouette(chromas_a: list, chromas_b: list) -> float:
    feats  = np.stack([c.max(0) for c in chromas_a + chromas_b])
    labels = np.array([0] * len(chromas_a) + [1] * len(chromas_b))
    n_comp = min(50, feats.shape[0] - 1)
    pca    = PCA(n_components=n_comp).fit_transform(feats)
    return float(silhouette_score(pca, labels))


def pchip_warp_magnitude(warp_fn) -> float:
    """Mean |w(t) − t| in minutes for a PCHIP warp (reference time → query time)."""
    if warp_fn is None:
        return 0.0
    return float(np.mean(np.abs(warp_fn(TIME_AXIS) - TIME_AXIS)))


def align_pooled_pchip(chromas_a: list, anchors: list) -> tuple[list, float, list]:
    """PCHIP through the pooled calibration anchors, one warp applied to every A_i (→ B frame).
    Returns aligned A, warp magnitude (min) and the anchors the spline used."""
    wfn, used = pooled_anchor_pchip(anchors)
    return [warp_chroma_2d(c, wfn) for c in chromas_a], pchip_warp_magnitude(wfn), used


# ── Main ──────────────────────────────────────────────────────────────────────

def calibrate(model: ChromaWarpTransformer, chromas_t: torch.Tensor, study_ids: torch.Tensor,
              anchors: list[tuple[int, int]], device: torch.device, seed: int | None = None,
              log_labels: tuple[str, str] | None = None) -> float:
    """
    Anchor-supervised calibration in place (N_CALIB_STEPS of Adam, best-loss state kept).
    Seeded immediately before the loop, so any caller using the same seed and device
    consumes an identical dropout RNG stream. Logs every 50 steps if log_labels is given.
    Returns wall-clock seconds.
    """
    if seed is not None:
        set_seed(seed)
    t_start = time.perf_counter()
    n_a   = int((study_ids == 0).sum())
    model.train()
    opt   = optim.Adam(model.parameters(), lr=LR_CALIB)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_CALIB_STEPS)

    id_t       = torch.arange(N_BINS, dtype=torch.float32, device=device)
    best_loss  = float('inf')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for step in range(1, N_CALIB_STEPS + 1):
        aligned, warps = model(chromas_t, study_ids)
        a_loss  = anchor_loss(warps, study_ids, anchors, device)
        s_loss  = warp_smoothness_loss(warps)
        loss    = LAMBDA_ANCHOR * a_loss + LAMBDA_SMOOTH * s_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if float(loss) < best_loss:
            best_loss  = float(loss)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if log_labels and (step % 50 == 0 or step == N_CALIB_STEPS):
            with torch.no_grad():
                al_np = aligned.detach().cpu().numpy()
                tic_r = tic_correlations(list(al_np[:n_a]), list(al_np[n_a:]))
                wa = float((warps[:n_a] - id_t).abs().mean()) * BIN_MIN
                wb = float((warps[n_a:] - id_t).abs().mean()) * BIN_MIN
            print(f"  step {step:>4d}/{N_CALIB_STEPS}  "
                  f"loss={float(loss):.4f}  anchor={float(a_loss):.4f}  "
                  f"TIC_r={tic_r:.4f}  "
                  f"warp {log_labels[0]}={wa:.3f}m  {log_labels[1]}={wb:.3f}m")

    model.load_state_dict(best_state)
    model.eval()
    if device.type == 'mps':
        torch.mps.synchronize()
    return time.perf_counter() - t_start


def pick_device(name: str = 'auto') -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main(study_a_dir: Path, study_b_dir: Path, label_a: str, label_b: str,
         seed: int | None = None, ckpt: Path = CKPT_DIR / 'warp_transformer.pt',
         device_name: str = 'auto'):
    device = pick_device(device_name)
    print(f"Device: {device}\n")

    # Load chromatograms
    chromas_a   = load_chromas(study_a_dir / 'chroma')
    chromas_b   = load_chromas(study_b_dir / 'chroma')
    n_a, n_b    = len(chromas_a), len(chromas_b)
    raw_a       = load_tic_raw(study_a_dir / 'chroma')
    raw_b       = load_tic_raw(study_b_dir / 'chroma')
    all_chromas = chromas_a + chromas_b
    study_ids   = torch.tensor([0]*n_a + [1]*n_b, dtype=torch.long, device=device)
    chromas_t   = torch.from_numpy(np.stack(all_chromas)).to(device)
    print(f"{label_a}: {n_a} samples   {label_b}: {n_b} samples")

    # Extract RANSAC anchors (spectral cosine — no TIC r used)
    print(f"\nExtracting RANSAC anchors from {N_REF_PAIRS} {label_a} samples vs {label_b}[0] …")
    anchors = extract_anchors(chromas_a, chromas_b, n_ref=N_REF_PAIRS)
    if anchors:
        drift_bins = [q - r for r, q in anchors]
        print(f"  {len(anchors)} anchors after outlier filtering (IQR×{OUTLIER_IQR})")
        print(f"  ref bins : {[r for r,_ in anchors]}")
        print(f"  drift    : {drift_bins} bins "
              f"(median {int(np.median(drift_bins)):+d}, "
              f"range [{min(drift_bins)}, {max(drift_bins)}])")
    else:
        print("  WARNING: no anchors found — calibration skipped, using base model")

    # Load base synthetic model as calibration starting point.
    # The crossbatch fine-tuned model can produce large degenerate warps (8-9m)
    # that the sparse anchor supervision cannot correct in a small number of steps.
    # The base model starts near-identity (~0.25m) and the anchor signal is
    # sufficient to move it to the correct magnitude and direction.
    model = ChromaWarpTransformer()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    print(f"\nLoaded {ckpt.name}")

    # Pre-calibration metrics
    model.eval()
    with torch.no_grad():
        al0, w0 = model(chromas_t, study_ids)
    al0_a = [al0[i].cpu().numpy() for i in range(n_a)]
    al0_b = [al0[i].cpu().numpy() for i in range(n_a, n_a+n_b)]
    tic_pre = tic_correlations(al0_a, al0_b)
    warp_a_pre = float((w0[:n_a] - torch.arange(N_BINS, device=device).float()).abs().mean()) * BIN_MIN
    warp_b_pre = float((w0[n_a:] - torch.arange(N_BINS, device=device).float()).abs().mean()) * BIN_MIN
    print(f"Pre-calibration : TIC r={tic_pre:.4f}  "
          f"warp {label_a}={warp_a_pre:.3f}m  {label_b}={warp_b_pre:.3f}m")

    calib_seconds = 0.0
    if anchors:
        print(f"\nCalibrating: {N_CALIB_STEPS} steps  "
              f"lr={LR_CALIB}  λ_anchor={LAMBDA_ANCHOR}  λ_smooth={LAMBDA_SMOOTH}  seed={seed}\n")
        calib_seconds = calibrate(model, chromas_t, study_ids, anchors, device,
                                  seed=seed, log_labels=(label_a, label_b))

    # Final evaluation
    model.eval()
    with torch.no_grad():
        aligned_t, warps_t = model(chromas_t, study_ids)

    aligned_np = aligned_t.cpu().numpy()
    warps_np   = warps_t.cpu().numpy()
    al_a = [aligned_np[i] for i in range(n_a)]
    al_b = [aligned_np[i] for i in range(n_a, n_a+n_b)]
    id_np = np.arange(N_BINS, dtype=np.float32)[None, :]

    r_unaligned   = tic_r_per_pair(chromas_a, chromas_b)
    tic_unaligned = float(r_unaligned.mean())
    sil_unaligned = study_silhouette(chromas_a, chromas_b)

    print("\nAligning with PCHIP through the pooled calibration anchors (A → B frame) …")
    al_a_pc, mag_pc, used_pc = align_pooled_pchip(chromas_a, anchors)
    r_pc   = tic_r_per_pair(al_a_pc, chromas_b)
    sil_pc = study_silhouette(al_a_pc, chromas_b)

    # Raw TIC: the same warps applied to the raw (pre-sqrt, pre-normalisation) TIC
    wfn_pc, _ = pooled_anchor_pchip(anchors)
    raw_r_unaligned = pearson_rows(raw_a, raw_b)
    raw_r_pc = pearson_rows(np.stack([warp_tic(t, wfn_pc) for t in raw_a]), raw_b)
    raw_wt   = warp_tics_bins(np.concatenate([raw_a, raw_b]), warps_np)
    raw_r_wt = pearson_rows(raw_wt[:n_a], raw_wt[n_a:])

    r_wt     = tic_r_per_pair(al_a, al_b)
    tic_wt   = float(r_wt.mean())
    sil_wt   = study_silhouette(al_a, al_b)
    warp_all = float(np.abs(warps_np - id_np).mean()) * BIN_MIN
    warp_a   = float(np.abs(warps_np[:n_a] - id_np).mean()) * BIN_MIN
    warp_b   = float(np.abs(warps_np[n_a:] - id_np).mean()) * BIN_MIN

    def row(name, r, sil, mag):
        mag_s = f"{mag:>8.3f}m" if mag is not None else f"{'—':>9}"
        return (f"{name:<40} {r:>7.3f}  {r - tic_unaligned:>+12.3f}  "
                f"{sil:>10.3f}  {mag_s}")

    w = 86
    n_anchors = len(anchors)
    lines = [
        f"\n{'=' * w}",
        f"{'Method':<40} {'TIC r':>7}  {'Δ unaligned':>12}  {'Study sil':>10}  {'Warp mag':>9}",
        f"{'-' * w}",
        f"{'Unaligned':<40} {tic_unaligned:>7.3f}  {'—':>12}  {sil_unaligned:>10.3f}  {'—':>9}",
        row('PCHIP (pooled anchors)', float(r_pc.mean()), sil_pc, mag_pc),
        row('WarpTransformer + anchor calib', tic_wt, sil_wt, warp_b),
        f"{'=' * w}",
        f"{'TIC r (raw)':<40} {'TIC r':>7}  {'Δ unaligned':>12}",
        f"{'-' * w}",
        f"{'Unaligned':<40} {raw_r_unaligned.mean():>7.3f}  {'—':>12}",
        f"{'PCHIP (pooled anchors)':<40} {raw_r_pc.mean():>7.3f}  "
        f"{raw_r_pc.mean() - raw_r_unaligned.mean():>+12.3f}",
        f"{'WarpTransformer + anchor calib':<40} {raw_r_wt.mean():>7.3f}  "
        f"{raw_r_wt.mean() - raw_r_unaligned.mean():>+12.3f}",
        f"{'=' * w}",
        f"TIC r: mean Pearson r over all {n_a}×{n_b} = {n_a * n_b} (A_i, B_j) pairs.",
        f"PCHIP: one spline through the WarpTransformer's pooled anchors ({len(used_pc)} used), applied to "
        f"every A_i; B unwarped.",
        f"Warp mag: PCHIP = mean |w(t) − t| of that spline; WarpTransformer = batch-B samples only.",
        f"TIC r (raw): the same warps applied to the raw TIC (summed raw intensity of each binned "
        f"apex scan, no sqrt or normalisation).",
        f"",
        f"Seed                         : {seed}",
        f"Checkpoint                   : {ckpt}",
        f"Anchors used for calibration : {n_anchors} RANSAC pairs (raw cosine — no TIC leakage)",
        f"Anchors (ref_bin, query_bin) : {anchors}",
        f"Calibration steps            : {N_CALIB_STEPS} (λ_anchor={LAMBDA_ANCHOR}, λ_smooth={LAMBDA_SMOOTH})",
        f"Calibration wall time        : {calib_seconds:.1f} s on {device}",
        f"",
        f"WarpTransformer warp magnitudes (mean |warp(t)−t| in minutes):",
        f"  All samples : {warp_all:.3f} min",
        f"  {label_a:<22}: {warp_a:.3f} min",
        f"  {label_b:<22}: {warp_b:.3f} min",
    ]
    report = '\n'.join(lines)
    print(report)

    suffix = f'_seed{seed}' if seed is not None else ''
    out = RESULTS_DIR / f'warp_transformer_calib_{label_a}_{label_b}{suffix}.txt'
    out.write_text(report + '\n')

    per_pair_dir = RESULTS_DIR / 'per_pair'
    per_pair_dir.mkdir(exist_ok=True)
    np.savez(per_pair_dir / f'tic_r_calib_{label_a}_{label_b}{suffix}.npz',
             unaligned=r_unaligned, pchip_pooled=r_pc, warptransformer=r_wt,
             unaligned_raw=raw_r_unaligned, pchip_pooled_raw=raw_r_pc, warptransformer_raw=raw_r_wt,
             pchip_warp_mag=mag_pc, anchors=np.array(anchors),
             n_a=n_a, n_b=n_b, seed=-1 if seed is None else seed,
             sil=np.array([sil_unaligned, sil_pc, sil_wt]),         # unaligned, PCHIP, WT
             wt_warp_mag=np.array([warp_all, warp_a, warp_b]),
             calib_seconds=calib_seconds)
    print(f"\n  results → {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-a', default='fish_oil/batch_sep2015')
    parser.add_argument('--study-b', default='fish_oil/batch_jul2016')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed torch, numpy and random before calibration')
    parser.add_argument('--ckpt', type=Path, default=CKPT_DIR / 'warp_transformer.pt')
    parser.add_argument('--device', default='auto',
                        help="auto | cpu | mps | cuda. MPS is not bit-reproducible under a fixed "
                             "seed; use cpu for exactly repeatable runs.")
    args = parser.parse_args()
    label_a = Path(args.study_a).name
    label_b = Path(args.study_b).name
    main(DATA_DIR / args.study_a, DATA_DIR / args.study_b, label_a, label_b,
         seed=args.seed, ckpt=args.ckpt, device_name=args.device)
