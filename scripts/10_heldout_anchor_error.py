"""
10_heldout_anchor_error.py — Leave-one-anchor-out retention-time error at known compounds.

For each evaluated pair, every calibration anchor (r in batch A, q in batch B) is held out in
turn and each method predicts q from r without it:

  WarpTransformer          recalibrated from the checkpoint on the remaining anchors (script
                           06's settings and seed); prediction = mean over B samples of
                           w_B(w̄_A⁻¹(r)) (warp_transformer.predict_query_bins)
  Free warp (no network)   calibration.free_warp on the remaining anchors (100× the WT learning
                           rate), same prediction; same anchors and loss as the WT, so a gap is
                           what the network adds between anchors
  PCHIP (pooled anchors)   warps.predict_query_bin through the remaining anchors (PCHIP between
                           them, slope 1 beyond)
  COW-TIC                  uses no anchors, so nothing is held out: COW aligns each of A[0..2]
                           (the samples the anchors come from) onto B[0]; its warp is inverted at
                           r and the median taken, as the anchors themselves are pooled
  References               no alignment (q̂ = r) and a constant shift (median drift of the rest)

Also reported: the WT's in-sample residual (calibrated on all anchors). Errors in minutes.

Note: the anchor set is the pooled, outlier- and slope-filtered set of script 06; those filters
ran with the held-out anchor included.

Output: results/heldout_anchor_error.txt, results/per_pair/heldout_anchor_error.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from anchors import extract_anchors                                             # noqa: E402
from baselines import cow_tic_align                                             # noqa: E402
from calibration import (batch_tensors, calibrate, free_warp, load_model,      # noqa: E402
                         pick_device, predict)
from config import (BIN_MIN, DEFAULT_CKPT, EVAL_PAIRS, FREE_WARP_LR,            # noqa: E402
                    N_REF_PAIRS, PER_PAIR_DIR, RESULTS_DIR, pair_label)
from data import load_batch                                                     # noqa: E402
from warp_transformer import predict_query_bins                                 # noqa: E402
from warps import invert_src, predict_query_bin                                 # noqa: E402

FREE = 'Free warp (no network)'
METHODS = ['No alignment', 'Constant shift', 'PCHIP (pooled anchors)', 'COW-TIC', FREE,
           'WarpTransformer']


def warps_predict(warps: np.ndarray, n_a: int, r_bins) -> np.ndarray:
    """Predicted B bins at A bins r_bins from warps [N, T] (A first): mean over B samples."""
    w = torch.as_tensor(warps, dtype=torch.float64)
    sid = torch.tensor([0] * n_a + [1] * (len(w) - n_a))
    return predict_query_bins(w, sid, list(r_bins)).mean(dim=0).numpy()


def cow_predict(a, b, r_bins) -> np.ndarray:
    """Median over A[0..N_REF_PAIRS−1] of COW's inverse warp (A bin → B[0] bin) at r_bins."""
    preds = [invert_src(cow_tic_align(a.tic[i], b.tic[0]).src, np.asarray(r_bins, float))
             for i in range(min(N_REF_PAIRS, len(a)))]
    return np.median(preds, axis=0)


def main(ckpt: Path, seed: int, device_name: str) -> list[str]:
    device = pick_device(device_name)
    lines = [__doc__.strip().splitlines()[0], '',
             'Note: anchors are script 06\'s pooled set, outlier- and slope-filtered with the '
             'held-out anchor included.',
             f'Checkpoint {ckpt}   seed {seed}   device {device}', '']
    saved, summary = {}, []
    for name_a, name_b in EVAL_PAIRS:
        a, b = load_batch(name_a), load_batch(name_b)
        n_a, n_b = len(a), len(b)
        anchors = extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw).anchors
        r_bins = [r for r, _ in anchors]
        q_obs = np.array([q for _, q in anchors], dtype=float)
        x, sid = batch_tensors(a.chromas, b.chromas, device)
        label = pair_label(name_a, name_b)
        print(f'\n{label}: {len(anchors)} anchors {anchors}', flush=True)
        if len(anchors) < 2:
            lines += ['=' * 96, f'{label}: {len(anchors)} anchor(s); at least 2 are needed to hold '
                      'one out. Skipped.', '']
            continue

        def calibrated_warps(anc):
            model = load_model(ckpt, device)
            calibrate(model, x, sid, anc, seed=seed)
            return predict(model, x, sid)[1]

        in_sample = np.abs(warps_predict(calibrated_warps(anchors), n_a, r_bins) - q_obs) * BIN_MIN
        cow_q = cow_predict(a, b, r_bins)
        err = {m: [] for m in METHODS}
        for k, (r, q) in enumerate(anchors):
            rest = anchors[:k] + anchors[k + 1:]
            pred = {'No alignment': r,
                    'Constant shift': r + np.median([qq - rr for rr, qq in rest]),
                    'PCHIP (pooled anchors)': predict_query_bin(rest, r),
                    'COW-TIC': cow_q[k],
                    FREE: warps_predict(free_warp(n_a, n_b, rest, FREE_WARP_LR, seed)[0], n_a, [r])[0],
                    'WarpTransformer': warps_predict(calibrated_warps(rest), n_a, [r])[0]}
            for m in METHODS:
                err[m].append(abs(float(pred[m]) - q) * BIN_MIN)
            print(f'  held out ({r}, {q}) d={q - r:+d}: ' +
                  '  '.join(f'{m}={err[m][-1]:.3f}' for m in METHODS), flush=True)

        lines += ['=' * 96, f'{label}   {len(anchors)} anchors (A bin, B bin): {anchors}', '',
                  f'  {"Method":<24} {"mean (min)":>11} {"max (min)":>10}   per anchor (min)']
        for m in METHODS:
            e = np.array(err[m])
            lines.append(f'  {m:<24} {e.mean():>11.3f} {e.max():>10.3f}   ' +
                         ' '.join(f'{v:.2f}' for v in e))
        lines.append(f'  {"WT in-sample":<24} {in_sample.mean():>11.3f} {in_sample.max():>10.3f}   ' +
                     ' '.join(f'{v:.2f}' for v in in_sample) + '   (calibrated on all anchors)')
        lines.append('')
        summary.append((label, {m: (np.mean(err[m]), np.max(err[m])) for m in METHODS}))
        for m in METHODS:
            saved[f'{name_a}_{name_b}_{m.replace(" ", "_")}'] = np.array(err[m])
        saved[f'{name_a}_{name_b}_anchors'] = np.array(anchors).reshape(-1, 2)
        saved[f'{name_a}_{name_b}_wt_in_sample'] = in_sample

    lines += ['=' * 96, 'Summary: mean / max held-out error (min)',
              f'  {"Pair":<12} ' + ' '.join(f'{m[:22]:>22}' for m in METHODS)]
    for label, s in summary:
        lines.append(f'  {label:<12} ' + ' '.join(f'{s[m][0]:>12.3f} / {s[m][1]:<7.3f}' for m in METHODS))
    lines += ['', 'End anchors are extrapolated by every method once held out; read the per-anchor '
              'values, not only the means (4–8 anchors per pair).']

    out = RESULTS_DIR / 'heldout_anchor_error.txt'
    out.write_text('\n'.join(lines) + '\n')
    PER_PAIR_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PER_PAIR_DIR / 'heldout_anchor_error.npz', **saved)
    print('\n' + '\n'.join(lines) + f'\n\n  results → {out}')
    return lines


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    main(args.ckpt, args.seed, args.device)
