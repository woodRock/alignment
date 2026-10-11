"""
09_mechanism_ablation.py — Where does the WarpTransformer's gain over a spline come from?

On the four evaluated pairs, from one seeded calibration (script 06's settings):

  Unaligned
  WT original          the calibrated WarpTransformer: one warp per sample, both batches
  WT batch-shared      its warps averaged within each batch (a mean of monotone warps is
                       monotone), so every sample of a batch gets the same warp
  WT residual only     identity + (w_i − batch mean): the within-batch part alone
  Free warp            no network: one monotone warp per batch with the WT head's
                       parameterisation, fitted with the same loss and loop
                       (src/calibration.free_warp), at the WT learning rate and 100× it
  Pooled PCHIP         script 07: the spline through the same pooled anchors, A only
  Pooled PCHIP + COW   pooled PCHIP, then a per-sample COW refinement (8 segments, ±4 bins) of
                       every A_i and B_j onto the mean batch-B TIC (COW optimises TIC r itself)
  Smoothing controls   a pure half-bin shift (maximal interpolation smoothing, no alignment)
                       applied to both batches of Unaligned and to batch B of Pooled PCHIP: the
                       WT resamples both batches, the baselines only A

Reports between-batch TIC r (the paper's metric) and within-batch TIC r (do samples of the
same batch align to each other?), the WT's within-batch warp spread and the final losses.
Warping is linear interpolation along RT, which commutes with the sum over m/z, so every
variant is evaluated on warped TICs.

Output: results/mechanism_ablation.txt, results/per_pair/mechanism_ablation_<A>_<B>.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from anchors import extract_anchors                                             # noqa: E402
from baselines import cow_tic_align                                             # noqa: E402
from calibration import (batch_tensors, calibrate, final_losses, free_warp,    # noqa: E402
                         load_model, pick_device, predict)
from config import (BIN_MIN, DEFAULT_CKPT, EVAL_PAIRS, FREE_WARP_LR, LR_CALIB,  # noqa: E402
                    N_BINS, PER_PAIR_DIR, RESULTS_DIR, pair_label)
from data import load_batch                                                     # noqa: E402
from metrics import holm, pearson_rows, per_a_mean, within_batch_r, paired_wilcoxon  # noqa: E402
from warps import pchip_warp, src_from_fn, warp_rows                            # noqa: E402

REFINE_SEGS, REFINE_SLACK = 8, 4
FREE_LRS = (LR_CALIB, FREE_WARP_LR)


def cow_refine(tics: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.stack([cow_tic_align(t, ref, n_segs=REFINE_SEGS, slack=REFINE_SLACK).warped
                     for t in tics])


def half_bin_shift(n: int) -> np.ndarray:
    return np.tile(np.clip(np.arange(N_BINS) + 0.5, 0, N_BINS - 1), (n, 1))


def variants(ta, tb, w_wt, anchors, seed) -> tuple[dict, dict]:
    """name → (A TICs, B TICs), and the free warps' final losses."""
    n_a, n_b = len(ta), len(tb)
    both = np.concatenate([ta, tb])

    def split(x):
        return x[:n_a], x[n_a:]
    w_shared = np.concatenate([np.tile(w_wt[:n_a].mean(0), (n_a, 1)),
                               np.tile(w_wt[n_a:].mean(0), (n_b, 1))])
    resid = np.concatenate([w_wt[:n_a] - w_wt[:n_a].mean(0), w_wt[n_a:] - w_wt[n_a:].mean(0)])
    resid = np.clip(resid + np.arange(N_BINS), 0, N_BINS - 1)
    pa = warp_rows(ta, src_from_fn(pchip_warp(anchors)))
    ref_b = tb.mean(0)

    V = {'Unaligned': (ta, tb),
         'WT original': split(warp_rows(both, w_wt)),
         'WT batch-shared': split(warp_rows(both, w_shared)),
         'WT residual only': split(warp_rows(both, resid))}
    losses = {}
    for lr in FREE_LRS:
        w_free, losses[lr] = free_warp(n_a, n_b, anchors, lr, seed)
        V[f'Free warp (lr {lr:g})'] = split(warp_rows(both, w_free))
    V['Pooled PCHIP'] = (pa, tb)
    V['Pooled PCHIP + COW refine'] = (cow_refine(pa, ref_b), cow_refine(tb, ref_b))
    V['Control: unaligned, both ½-bin shifted'] = split(warp_rows(both, half_bin_shift(n_a + n_b)))
    V['Control: pooled PCHIP, B ½-bin shifted'] = (pa, warp_rows(tb, half_bin_shift(n_b)))
    return V, losses


def main(seed: int, device_name: str, ckpt: Path) -> list[str]:
    device = pick_device(device_name)
    lines = [__doc__.strip().splitlines()[0], '',
             f'Seed {seed}   device {device}   checkpoint {ckpt}   COW refinement {REFINE_SEGS} '
             f'segments ±{REFINE_SLACK} bins', '']
    summary, per_a = [], []
    for name_a, name_b in EVAL_PAIRS:
        a, b = load_batch(name_a), load_batch(name_b)
        n_a, n_b = len(a), len(b)
        anchors = extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw).anchors
        print(f'{pair_label(name_a, name_b)}: {len(anchors)} anchors — calibrating …', flush=True)
        model = load_model(ckpt, device)
        x, sid = batch_tensors(a.chromas, b.chromas, device)
        calibrate(model, x, sid, anchors, seed=seed)
        aligned, w_wt = predict(model, x, sid)
        loss_wt = final_losses(w_wt, sid, anchors)

        V, free_losses = variants(a.tic, b.tic, w_wt, anchors, seed)
        r = {k: pearson_rows(xa, xb) for k, (xa, xb) in V.items()}
        model_tic = aligned.sum(-1)
        if not np.allclose(r['WT original'], pearson_rows(model_tic[:n_a], model_tic[n_a:]),
                           atol=1e-4):
            raise SystemExit('re-applying the WT warps does not reproduce the model output')
        u = float(r['Unaligned'].mean())

        def spread(w):
            return float(np.abs(w - w.mean(0)).mean()) * BIN_MIN
        lines += ['=' * 96, f'{pair_label(name_a, name_b)}   n = {n_a}×{n_b}   anchors {len(anchors)}',
                  f'  WT within-batch warp spread (mean |w_i − batch mean|): '
                  f'A {spread(w_wt[:n_a]):.3f} min, B {spread(w_wt[n_a:]):.3f} min',
                  f'  Final anchor / smoothness loss: WT {loss_wt[0]:.4f} / {loss_wt[1]:.5f}; '
                  + '; '.join(f'free lr {lr:g} {l[0]:.4f} / {l[1]:.5f}' for lr, l in free_losses.items()),
                  f'  {"Variant":<42} {"TIC r A×B":>9} {"Δ unal":>8} {"within A":>9} {"within B":>9}']
        for k, (xa, xb) in V.items():
            lines.append(f'  {k:<42} {r[k].mean():>9.3f} {r[k].mean() - u:>+8.3f} '
                         f'{within_batch_r(xa)[0]:>9.3f} {within_batch_r(xb)[0]:>9.3f}')
        summary.append((name_a, name_b, {k: float(v.mean()) for k, v in r.items()}))
        per_a.append((name_a, name_b, n_a, n_b, r))
        np.savez(PER_PAIR_DIR / f'mechanism_ablation_{name_a}_{name_b}.npz',
                 **{k.replace(' ', '_'): v for k, v in r.items()},
                 w_wt=w_wt, anchors=np.array(anchors).reshape(-1, 2), n_a=n_a, n_b=n_b)

    names = list(summary[0][2])
    lines += ['', '=' * 96, 'Summary: mean between-batch TIC r',
              f'  {"Variant":<42} ' + ' '.join(f'{pair_label(a, b):>12}' for a, b, _ in summary)]
    for k in names:
        lines.append(f'  {k:<42} ' + ' '.join(f'{s[k]:>12.3f}' for *_, s in summary))

    best_free = f'Free warp (lr {FREE_LRS[-1]:g})'
    comps = [('WT original', 'WT batch-shared'), ('WT batch-shared', best_free),
             (best_free, 'Pooled PCHIP'), ('WT original', 'Pooled PCHIP + COW refine'),
             ('Control: unaligned, both ½-bin shifted', 'Unaligned')]
    lines += ['', 'Paired two-sided Wilcoxon per A sample (r averaged over B_j); Holm over the 4 '
              'evaluated pairs, separately per comparison.',
              'Diagnostic tests; not part of the 10-comparison family of significance_tests.txt.']
    for x, y in comps:
        res = []
        for na, nb, n_a, n_b, r in per_a:
            dx, dy = per_a_mean(r[x], n_a, n_b), per_a_mean(r[y], n_a, n_b)
            res.append((na, nb, paired_wilcoxon(dx, dy)[1], float(np.median(dx - dy))))
        lines.append(f'  {x}  vs  {y}:')
        for (na, nb, _, md), pa in zip(res, holm([t[2] for t in res])):
            lines.append(f'    {pair_label(na, nb):<12} Holm p={pa:.2e}  median Δr {md:+.4f}')

    out = RESULTS_DIR / 'mechanism_ablation.txt'
    out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines) + f'\n\n  results → {out}')
    return lines


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', help='cpu (default, reproducible) | mps | cuda')
    ap.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()
    main(args.seed, args.device, args.ckpt)
