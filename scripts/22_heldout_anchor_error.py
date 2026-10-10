"""
22_heldout_anchor_error.py — Leave-one-anchor-out RT error at known compounds.

For each evaluated batch pair, the calibration anchor set (ref_bin r in batch A, query_bin q
in batch B[0]; observed drift d = q − r) is taken from script 16. Each anchor is held out in
turn and every method predicts its drift without it:

  WarpTransformer  recalibrated from the pretrained checkpoint on the remaining anchors
                   (500 steps, same settings and seed as script 16), then
                   error = |(mean_j w_Bj(r) − mean_i w_Ai(r)) − d|
  PCHIP (pooled anchors) PCHIP spline through the remaining pooled anchors with the same end points as
                   align_pair ((0, 0) and (45, 45) min), evaluated at r: error = |q̂ − q|
  COW-TIC          uses no anchors, so nothing is held out. COW aligns each of the A samples
                   used for anchor extraction (A_0..A_2) to B[0]; its inverse warp at r is
                   taken per sample and the median compared with q, mirroring how the anchors
                   themselves were aggregated.
  Reference rows   No alignment (error = |d|) and Constant shift (median drift of the
                   remaining anchors).

The WarpTransformer is also calibrated once on all anchors, giving the in-sample residual at
each anchor for comparison with the held-out error.

All errors are in minutes. Output: results/heldout_anchor_error.txt and
results/per_pair/heldout_anchor_error.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

SCRIPTS = Path(__file__).parent
ROOT    = SCRIPTS.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


calib16 = _load('calib16', '16_align_testtime_calib.py')
base17  = _load('base17', '17_literature_baselines.py')
from warp_transformer import ChromaWarpTransformer  # noqa: E402  (src on path via calib16)
from alignment import load_chromas                  # noqa: E402

BIN_MIN = calib16.BIN_MIN
RUN_MIN = 45.0
PAIRS = [('batch_sep2015', 'batch_jul2016'), ('batch_sep2015', 'batch_apr2016'),
         ('batch_jan2016', 'batch_apr2016'), ('batch_jan2016', 'batch_jul2016')]
METHODS = ['No alignment', 'Constant shift', 'PCHIP (pooled anchors)', 'COW-TIC', 'WarpTransformer']


def wt_offset_at(model, chromas_t, study_ids, r_bins: list[int]) -> np.ndarray:
    """Predicted drift (bins) at reference bins: mean_j w_Bj(r) − mean_i w_Ai(r)."""
    with torch.no_grad():
        _, warps = model(chromas_t, study_ids)
    w = warps.cpu().numpy()
    sid = study_ids.cpu().numpy()
    return np.array([w[sid == 1, r].mean() - w[sid == 0, r].mean() for r in r_bins])


def calibrated_model(anchors, chromas_t, study_ids, device, ckpt, seed):
    model = ChromaWarpTransformer()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    calib16.calibrate(model, chromas_t, study_ids, anchors, device, seed=seed)
    return model


def pchip_predict(rest: list[tuple[int, int]], r_bin: int) -> float:
    """Predicted query bin at r_bin from a PCHIP through the remaining anchors."""
    rf = np.concatenate([[0.0], [r * BIN_MIN for r, _ in rest], [RUN_MIN]])
    qf = np.concatenate([[0.0], [q * BIN_MIN for _, q in rest], [RUN_MIN]])
    return float(PchipInterpolator(rf, qf)(r_bin * BIN_MIN)) / BIN_MIN


def cow_predict(chromas_a, chromas_b, r_bins: list[int], n_ref: int) -> np.ndarray:
    """Median over A_0..A_{n_ref-1} of COW's inverse warp (A time → B[0] time) at each r, in bins."""
    ref_tic = chromas_b[0].sum(axis=1)
    preds = []
    for i in range(min(n_ref, len(chromas_a))):
        _, _, warp_fn = base17.cow_tic_align(chromas_a[i].sum(axis=1), ref_tic)
        b_times = np.linspace(0.0, RUN_MIN, 2001)          # B[0] time grid
        a_times = warp_fn(b_times)                          # COW maps B[0] time → A_i time
        preds.append(np.interp(np.array(r_bins) * BIN_MIN, a_times, b_times) / BIN_MIN)
    return np.median(np.array(preds), axis=0)


def main(ckpt: Path, seed: int, device_name: str) -> None:
    device = calib16.pick_device(device_name)
    lines = [__doc__.strip().split('\n')[0], '',
             f"Checkpoint {ckpt}   seed {seed}   device {device}", '']
    saved = {}
    summary = []

    for a, b in PAIRS:
        chromas_a = load_chromas(ROOT / 'data' / 'fish_oil' / a / 'chroma')
        chromas_b = load_chromas(ROOT / 'data' / 'fish_oil' / b / 'chroma')
        n_a, n_b  = len(chromas_a), len(chromas_b)
        study_ids = torch.tensor([0] * n_a + [1] * n_b, dtype=torch.long, device=device)
        chromas_t = torch.from_numpy(np.stack(chromas_a + chromas_b)).to(device)
        anchors   = calib16.extract_anchors(chromas_a, chromas_b)
        r_bins    = [r for r, _ in anchors]
        drifts    = np.array([q - r for r, q in anchors], dtype=float)
        label     = f"{a.replace('batch_', '')} × {b.replace('batch_', '')}"
        print(f"\n{label}: {len(anchors)} anchors {anchors}", flush=True)

        full = calibrated_model(anchors, chromas_t, study_ids, device, ckpt, seed)
        in_sample = np.abs(wt_offset_at(full, chromas_t, study_ids, r_bins) - drifts) * BIN_MIN
        cow_q = cow_predict(chromas_a, chromas_b, r_bins, calib16.N_REF_PAIRS)

        err = {m: [] for m in METHODS}
        for k, (r, q) in enumerate(anchors):
            rest = [x for j, x in enumerate(anchors) if j != k]
            d = q - r
            err['No alignment'].append(abs(d) * BIN_MIN)
            err['Constant shift'].append(abs(np.median([qq - rr for rr, qq in rest]) - d) * BIN_MIN)
            err['PCHIP (pooled anchors)'].append(abs(pchip_predict(rest, r) - q) * BIN_MIN)
            err['COW-TIC'].append(abs(cow_q[k] - q) * BIN_MIN)
            model = calibrated_model(rest, chromas_t, study_ids, device, ckpt, seed)
            pred = wt_offset_at(model, chromas_t, study_ids, [r])[0]
            err['WarpTransformer'].append(abs(pred - d) * BIN_MIN)
            print(f"  held out ({r}, {q}) d={d:+d}: " +
                  '  '.join(f"{m}={err[m][-1]:.3f}" for m in METHODS), flush=True)

        lines += [f"{'=' * 92}", f"{label}   {len(anchors)} anchors (ref_bin, query_bin): {anchors}", '',
                  f"  {'Method':<20} {'mean (min)':>11} {'max (min)':>10}   per anchor (min, in anchor order)"]
        for m in METHODS:
            e = np.array(err[m])
            lines.append(f"  {m:<20} {e.mean():>11.3f} {e.max():>10.3f}   " +
                         ' '.join(f"{x:.2f}" for x in e))
        lines.append(f"  {'WT in-sample':<20} {in_sample.mean():>11.3f} {in_sample.max():>10.3f}   " +
                     ' '.join(f"{x:.2f}" for x in in_sample) +
                     "   (calibrated on all anchors; not held out)")
        lines.append('')
        summary.append((label, {m: (np.mean(err[m]), np.max(err[m])) for m in METHODS}))
        for m in METHODS:
            saved[f"{a}_{b}_{m.replace(' ', '_')}"] = np.array(err[m])
        saved[f"{a}_{b}_anchors"] = np.array(anchors)
        saved[f"{a}_{b}_wt_in_sample"] = in_sample

    lines += [f"{'=' * 92}", "Summary: mean / max held-out error (min)",
              f"  {'Pair':<20} " + ' '.join(f"{m:>18}" for m in METHODS)]
    for label, s in summary:
        lines.append(f"  {label:<20} " + ' '.join(f"{s[m][0]:>8.3f} / {s[m][1]:<7.3f}" for m in METHODS))
    lines += ['', "With 4–8 anchors per pair, each mean is over 4–8 values; end anchors are",
              "extrapolated by every method once held out, so per-anchor values are listed above."]

    report = '\n'.join(lines)
    print('\n' + report)
    (ROOT / 'results' / 'heldout_anchor_error.txt').write_text(report + '\n')
    (ROOT / 'results' / 'per_pair').mkdir(parents=True, exist_ok=True)
    np.savez(ROOT / 'results' / 'per_pair' / 'heldout_anchor_error.npz', **saved)
    print(f"\n  results → {ROOT / 'results' / 'heldout_anchor_error.txt'}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, default=ROOT / 'checkpoints' / 'warp_transformer.pt')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    main(args.ckpt, args.seed, args.device)
