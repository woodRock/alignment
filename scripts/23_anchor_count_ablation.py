"""
23_anchor_count_ablation.py — How many calibration anchors are needed, with and without pretraining?

For each evaluated batch pair, the anchor set from script 16 is subsampled to k = 2 … n
anchors. For each k < n, up to --subsets distinct random subsets are drawn (all subsets if
there are fewer); k = n uses the full set once. For every subset, the pretrained checkpoint and
a random-init model of the same architecture are each calibrated (500 steps, script 16
settings) and the mean TIC r over all A×B pairs is recorded.

Each subset s at size k uses calibration seed = seed + 100·k + s for both models, so the two
arms see the same dropout stream; the random-init model's weights are drawn from the same seed.

Output: results/anchor_count_ablation.txt and results/per_pair/anchor_count_ablation.npz
"""

from __future__ import annotations

import argparse
import importlib.util
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).parent
ROOT    = SCRIPTS.parent

_spec   = importlib.util.spec_from_file_location('calib16', SCRIPTS / '16_align_testtime_calib.py')
calib16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calib16)
from warp_transformer import ChromaWarpTransformer  # noqa: E402  (src on path via calib16)
from alignment import load_chromas                  # noqa: E402

PAIRS = [('batch_sep2015', 'batch_jul2016'), ('batch_sep2015', 'batch_apr2016'),
         ('batch_jan2016', 'batch_apr2016'), ('batch_jan2016', 'batch_jul2016')]


def subsets_for(n: int, k: int, n_subsets: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    if comb(n, k) <= n_subsets:
        return list(combinations(range(n), k))
    chosen: set[tuple[int, ...]] = set()
    while len(chosen) < n_subsets:
        chosen.add(tuple(sorted(rng.choice(n, size=k, replace=False).tolist())))
    return sorted(chosen)


def tic_r_after(model, chromas_t, study_ids) -> float:
    with torch.no_grad():
        aligned, _ = model(chromas_t, study_ids)
    al = aligned.cpu().numpy()
    n_a = int((study_ids == 0).sum())
    return calib16.tic_correlations(list(al[:n_a]), list(al[n_a:]))


def main(ckpt: Path, seed: int, n_subsets: int, device_name: str) -> None:
    device = calib16.pick_device(device_name)
    rng    = np.random.default_rng(seed)
    lines  = [__doc__.strip().split('\n')[0], '',
              f"Checkpoint {ckpt}   base seed {seed}   subsets per k ≤ {n_subsets}   device {device}"]
    if device.type == 'mps':
        lines.append("MPS kernels are not bit-reproducible under a fixed seed (same-seed reruns differ "
                     "by ~0.002 mean TIC r); values are averages over subsets, so this is below the "
                     "between-subset spread reported here.")
    lines.append('')
    saved = {}

    for a, b in PAIRS:
        chromas_a = load_chromas(ROOT / 'data' / 'fish_oil' / a / 'chroma')
        chromas_b = load_chromas(ROOT / 'data' / 'fish_oil' / b / 'chroma')
        n_a, n_b  = len(chromas_a), len(chromas_b)
        study_ids = torch.tensor([0] * n_a + [1] * n_b, dtype=torch.long, device=device)
        chromas_t = torch.from_numpy(np.stack(chromas_a + chromas_b)).to(device)
        anchors   = calib16.extract_anchors(chromas_a, chromas_b)
        n         = len(anchors)
        r_unal    = calib16.tic_correlations(chromas_a, chromas_b)
        label     = f"{a.replace('batch_', '')} × {b.replace('batch_', '')}"
        print(f"\n{label}: {n} anchors, unaligned TIC r {r_unal:.3f}", flush=True)

        rows, records = [], []
        for k in range(2, n + 1):
            subs = subsets_for(n, k, 1 if k == n else n_subsets, rng)
            r_pre, r_rand = [], []
            for s, idx in enumerate(subs):
                sub   = [anchors[i] for i in idx]
                s_cal = seed + 100 * k + s

                pre = ChromaWarpTransformer()
                pre.load_state_dict(torch.load(ckpt, map_location=device))
                pre.to(device)
                calib16.calibrate(pre, chromas_t, study_ids, sub, device, seed=s_cal)
                r_pre.append(tic_r_after(pre, chromas_t, study_ids))

                calib16.set_seed(s_cal)
                rand = ChromaWarpTransformer().to(device)
                calib16.calibrate(rand, chromas_t, study_ids, sub, device, seed=s_cal)
                r_rand.append(tic_r_after(rand, chromas_t, study_ids))

                records.append((k, s, r_pre[-1], r_rand[-1]))
                print(f"  k={k} subset {idx}: pretrained {r_pre[-1]:.3f}  random-init {r_rand[-1]:.3f}",
                      flush=True)
            rows.append((k, len(subs), np.mean(r_pre), np.std(r_pre), np.mean(r_rand), np.std(r_rand)))

        lines += [f"{'=' * 84}", f"{label}   {n} anchors {anchors}   unaligned TIC r {r_unal:.3f}", '',
                  f"  {'k':>3} {'subsets':>8} {'pretrained':>17} {'random init':>17} {'pretraining Δ':>15}"]
        for k, ns, mp, sp, mr, sr in rows:
            lines.append(f"  {k:>3} {ns:>8} {mp:>9.3f} ± {sp:<5.3f} {mr:>9.3f} ± {sr:<5.3f} {mp - mr:>+15.3f}")
        lines.append("  (mean ± s.d. over subsets; s.d. is 0 where only one subset exists)")
        lines.append('')
        saved[f"{a}_{b}"] = np.array(records)
        saved[f"{a}_{b}_anchors"] = np.array(anchors)

    report = '\n'.join(lines)
    print('\n' + report)
    (ROOT / 'results' / 'anchor_count_ablation.txt').write_text(report + '\n')
    (ROOT / 'results' / 'per_pair').mkdir(parents=True, exist_ok=True)
    np.savez(ROOT / 'results' / 'per_pair' / 'anchor_count_ablation.npz', **saved)
    print(f"\n  results → {ROOT / 'results' / 'anchor_count_ablation.txt'}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, default=ROOT / 'checkpoints' / 'warp_transformer.pt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--subsets', type=int, default=3)
    ap.add_argument('--device', default='mps')
    args = ap.parse_args()
    main(args.ckpt, args.seed, args.subsets, args.device)
