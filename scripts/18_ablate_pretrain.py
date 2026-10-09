"""
18_ablate_pretrain.py — Does pretraining matter, or is calibration doing all the work?

Runs the same anchor-supervised calibration loop from script 16 twice:
  (a) starting from the pretrained warp_transformer.pt checkpoint
  (b) starting from a freshly initialised (random weights) ChromaWarpTransformer

Zero-initialised warp_head means both starts produce a near-identity warp at step 0,
so any difference in the final TIC r reflects what the pretrained representations
contribute beyond the anchor supervision signal itself.
"""

from __future__ import annotations

import sys
import random
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict
from pathlib import Path
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from warp_transformer import ChromaWarpTransformer, warp_smoothness_loss
from alignment import load_chromas, align_pair

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

N_BINS        = 200
BIN_MIN       = 45.0 / N_BINS
N_CALIB_STEPS = 500
LR_CALIB      = 5e-4
LAMBDA_ANCHOR = 1.0
LAMBDA_SMOOTH = 0.05
N_REF_PAIRS   = 3
OUTLIER_IQR   = 1.5

STUDY_A_DEFAULT = 'fish_oil/batch_sep2015'
STUDY_B_DEFAULT = 'fish_oil/batch_jul2016'


def extract_anchors(chromas_a, chromas_b, n_ref=N_REF_PAIRS):
    ref_query_per_rbin = defaultdict(list)
    query_sample = chromas_b[0]
    for i in range(min(n_ref, len(chromas_a))):
        _, _, _, _, _, anchors = align_pair(
            query_sample, chromas_a[i], encode_fn=None, return_anchors=True
        )
        for r_bin, q_bin in anchors:
            ref_query_per_rbin[r_bin].append(q_bin)

    if not ref_query_per_rbin:
        return []
    aggregated = [(r, int(np.round(np.median(q))))
                  for r, q in sorted(ref_query_per_rbin.items())]
    drifts = np.array([q - r for r, q in aggregated], dtype=float)
    mean, std = float(np.mean(drifts)), max(float(np.std(drifts)), 1.0)
    keep = np.abs(drifts - mean) <= OUTLIER_IQR * std
    return [a for a, k in zip(aggregated, keep) if k]


def anchor_loss(warps, study_ids, anchor_pairs, device):
    if not anchor_pairs:
        return torch.tensor(0.0, device=device)
    ref_mean    = warps[study_ids == 0].mean(dim=0)
    query_warps = warps[study_ids == 1]
    loss = torch.tensor(0.0, device=device)
    for r_bin, q_bin in anchor_pairs:
        drift    = float(q_bin - r_bin)
        rel_warp = query_warps[:, r_bin] - ref_mean[r_bin]
        loss     = loss + (rel_warp - drift).pow(2).mean()
    return loss / len(anchor_pairs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tic_r_per_pair(chromas_a, chromas_b) -> np.ndarray:
    """Pearson TIC r for every (A_i, B_j), i-major (same order as scripts 16 and 17)."""
    return np.array([float(pearsonr(a.sum(1), b.sum(1))[0])
                     for a in chromas_a for b in chromas_b])


def tic_r(chromas_a, chromas_b):
    return float(tic_r_per_pair(chromas_a, chromas_b).mean())


def run_calibration(model, chromas_t, study_ids, anchors, device, label, seed=None):
    """Returns (mean TIC r, warp mag all samples, warp mag batch B, per-pair TIC r)."""
    model.eval()
    with torch.no_grad():
        _, w0 = model(chromas_t, study_ids)
    warp_init = float((w0 - torch.arange(N_BINS, device=device).float()).abs().mean()) * BIN_MIN
    print(f"  [{label}] initial warp mag = {warp_init:.3f} min")

    if anchors:
        # Seeded immediately before the loop, exactly as in script 16, so the pretrained
        # arm reproduces script 16's calibration for the same seed and device.
        if seed is not None:
            set_seed(seed)
        model.train()
        opt   = optim.Adam(model.parameters(), lr=LR_CALIB)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_CALIB_STEPS)
        best_loss, best_state = float('inf'), {k: v.clone() for k, v in model.state_dict().items()}

        for step in range(1, N_CALIB_STEPS + 1):
            aligned, warps = model(chromas_t, study_ids)
            a_loss = anchor_loss(warps, study_ids, anchors, device)
            s_loss = warp_smoothness_loss(warps)
            loss   = LAMBDA_ANCHOR * a_loss + LAMBDA_SMOOTH * s_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if float(loss) < best_loss:
                best_loss  = float(loss)
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        aligned_t, warps_t = model(chromas_t, study_ids)

    n_a = (study_ids == 0).sum().item()
    al_np = aligned_t.cpu().numpy()
    al_a  = [al_np[i] for i in range(n_a)]
    al_b  = [al_np[i] for i in range(n_a, len(al_np))]

    warp_np  = warps_t.cpu().numpy()
    id_np    = np.arange(N_BINS, dtype=np.float32)[None, :]
    warp_mag = float(np.abs(warp_np - id_np).mean()) * BIN_MIN
    warp_b   = float(np.abs(warp_np[n_a:] - id_np).mean()) * BIN_MIN
    r_pairs  = tic_r_per_pair(al_a, al_b)
    r        = float(r_pairs.mean())
    print(f"  [{label}] TIC r = {r:.4f}  warp_mag = {warp_mag:.3f} min  (batch B {warp_b:.3f} min)")
    return r, warp_mag, warp_b, r_pairs


def pick_device(name: str = 'auto') -> torch.device:
    if name != 'auto':
        return torch.device(name)
    return (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('mps') if torch.backends.mps.is_available()
            else torch.device('cpu'))


def main(study_a: str = STUDY_A_DEFAULT, study_b: str = STUDY_B_DEFAULT,
         seed: int | None = None, ckpt: Path = CKPT_DIR / 'warp_transformer.pt',
         device_name: str = 'auto'):
    device = pick_device(device_name)
    print(f"Device: {device}")
    print(f"Dataset: {study_a}  vs  {study_b}   seed={seed}\n")

    chromas_a = load_chromas(DATA_DIR / study_a / 'chroma')
    chromas_b = load_chromas(DATA_DIR / study_b / 'chroma')
    n_a, n_b  = len(chromas_a), len(chromas_b)
    print(f"  {Path(study_a).name}: {n_a} samples")
    print(f"  {Path(study_b).name}: {n_b} samples\n")

    all_chromas = chromas_a + chromas_b
    study_ids   = torch.tensor([0]*n_a + [1]*n_b, dtype=torch.long, device=device)
    chromas_t   = torch.from_numpy(np.stack(all_chromas)).to(device)

    anchors = extract_anchors(chromas_a, chromas_b)
    drift_bins = [q - r for r, q in anchors]
    print(f"Anchors: {len(anchors)} RANSAC pairs")
    if anchors:
        print(f"  ref bins : {[r for r, _ in anchors]}")
        print(f"  drift    : {drift_bins} bins  "
              f"(median {int(np.median(drift_bins)):+d}, "
              f"range [{min(drift_bins)}, {max(drift_bins)}])\n")

    # Unaligned baseline
    r_unaligned = tic_r(chromas_a, chromas_b)
    print(f"Unaligned TIC r = {r_unaligned:.4f}\n")

    # (a) Pretrained model
    model_pre = ChromaWarpTransformer()
    model_pre.load_state_dict(torch.load(ckpt, map_location=device))
    model_pre.to(device)
    print(f"Running calibration from pretrained checkpoint ({ckpt}) …")
    r_pre, mag_pre, mag_pre_b, pairs_pre = run_calibration(
        model_pre, chromas_t, study_ids, anchors, device, 'pretrained', seed)

    # (b) Random-init model (same architecture, no checkpoint). The init itself is
    # seeded so that the random-init arm is repeatable too.
    if seed is not None:
        set_seed(seed)
    model_rand = ChromaWarpTransformer()
    model_rand.to(device)
    print(f"\nRunning calibration from random initialisation …")
    r_rand, mag_rand, mag_rand_b, pairs_rand = run_calibration(
        model_rand, chromas_t, study_ids, anchors, device, 'random-init', seed)

    # Summary
    lines = [
        f"{'=' * 74}",
        f"{'Method':<30}  {'TIC r':>7}  {'Δ unaligned':>12}  {'Warp mag':>9}  {'Warp B':>9}",
        f"{'-' * 74}",
        f"{'Unaligned':<30}  {r_unaligned:>7.4f}  {'—':>12}  {'—':>9}  {'—':>9}",
        f"{'WarpTransformer (pretrained)':<30}  {r_pre:>7.4f}  "
        f"{r_pre - r_unaligned:>+12.4f}  {mag_pre:>8.3f}m  {mag_pre_b:>8.3f}m",
        f"{'WarpTransformer (random init)':<30}  {r_rand:>7.4f}  "
        f"{r_rand - r_unaligned:>+12.4f}  {mag_rand:>8.3f}m  {mag_rand_b:>8.3f}m",
        f"{'=' * 74}",
        f"Warp mag: mean |w(t) − t| over all samples of both batches. Warp B: batch-B samples only.",
        f"Seed {seed}   device {device}   checkpoint {ckpt}",
        f"Anchors ({len(anchors)}): {anchors}",
    ]
    print('\n' + '\n'.join(lines))
    gap = r_pre - r_rand
    print(f"\nPretraining advantage: {gap:+.4f} TIC r")

    a_lbl, b_lbl = Path(study_a).name, Path(study_b).name
    suffix = f'_seed{seed}' if seed is not None else ''
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f'ablate_pretrain_{a_lbl}_{b_lbl}{suffix}.txt').write_text(
        '\n'.join(lines) + f"\nPretraining advantage: {gap:+.4f} TIC r\n")
    (RESULTS_DIR / 'per_pair').mkdir(exist_ok=True)
    np.savez(RESULTS_DIR / 'per_pair' / f'tic_r_ablate_{a_lbl}_{b_lbl}{suffix}.npz',
             pretrained=pairs_pre, random_init=pairs_rand,
             warp_mag=np.array([mag_pre, mag_rand]), warp_mag_b=np.array([mag_pre_b, mag_rand_b]),
             anchors=np.array(anchors), seed=-1 if seed is None else seed)
    if abs(gap) < 0.01:
        print("→ calibration alone accounts for nearly all the improvement; "
              "pretraining adds negligible benefit on this task.")
    elif gap > 0:
        print("→ pretrained representations help the model interpolate between "
              f"sparse anchors ({len(anchors)} pairs), giving a meaningful edge.")
    else:
        print("→ random init outperforms pretrained — calibration overshoots with "
              "pretrained representations on this pair.")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--study-a', default=STUDY_A_DEFAULT)
    ap.add_argument('--study-b', default=STUDY_B_DEFAULT)
    ap.add_argument('--seed', type=int, default=None,
                    help='seed torch, numpy and random (random init and calibration)')
    ap.add_argument('--ckpt', type=Path, default=CKPT_DIR / 'warp_transformer.pt')
    ap.add_argument('--device', default='auto',
                    help='auto | cpu | mps | cuda. Use cpu for exactly repeatable runs.')
    args = ap.parse_args()
    main(args.study_a, args.study_b, args.seed, args.ckpt, args.device)
