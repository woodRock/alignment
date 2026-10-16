"""Step 10 — leave-one-anchor-out retention-time error at known compounds.

For each evaluated pair, every calibration anchor (r in batch A, q in batch B) is held out in
turn and each method predicts q from r without it:
  WarpTransformer          recalibrated from the checkpoint on the remaining anchors (step 06's
                           settings and seed); prediction = mean over B samples of
                           w_B(w̄_A⁻¹(r)) (warp_transformer.predict_query_bins)
  Free warp (no network)   calibration.free_warp on the remaining anchors (100× the WT learning
                           rate), same prediction; same anchors and loss as the WT, so a gap is
                           what the network adds between anchors
  PCHIP (pooled anchors)   warps.predict_query_bin through the remaining anchors
  COW-TIC                  uses no anchors, so nothing is held out: COW aligns each of A[0..2]
                           (the samples the anchors come from) onto B[0]; its warp is inverted
                           at r and the median taken, as the anchors themselves are pooled
  Linear warp              the least-squares line q = a·r + b through the remaining anchors
                           (warps.anchor_line), as step 07's linear-warp baseline
  References               no alignment (q̂ = r) and a constant shift (median drift of the rest)
Also reported: the WT's in-sample residual (calibrated on all anchors). Errors in minutes.

Note: the anchor set is step 06's pooled, outlier- and slope-filtered set; those filters ran
with the held-out anchor included. Pairs with fewer than two anchors are skipped.

Output: results/heldout_anchor_error.txt, results/per_pair/heldout_anchor_error.npz
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .._compat import (
    Batch,
    anchor_drift,
    anchor_line,
    cow_tic_align,
    extract_anchors,
    invert_src,
    load_batch,
    predict_query_bin,
)
from ..calibration import batch_tensors, calibrate, free_warp, load_model, pick_device, predict
from ..config import (
    BIN_MIN,
    EVAL_PAIRS,
    FREE_WARP_LR,
    N_CALIB_STEPS,
    N_REF_PAIRS,
    Paths,
    pair_label,
)
from ..warp_transformer import predict_query_bins
from ._common import (
    add_model_args,
    ckpt_or_default,
    parser,
    save_npz,
    setup_logging,
    table,
    title,
    write_report,
)

log = logging.getLogger('warpalign')
FREE = 'Free warp (no network)'
METHODS = (
    'No alignment',
    'Constant shift',
    'Linear warp',
    'PCHIP (pooled anchors)',
    'COW-TIC',
    FREE,
    'WarpTransformer',
)
Anchors = list[tuple[int, int]]


def warps_predict(warps: np.ndarray, n_a: int, r_bins: list[int]) -> np.ndarray:
    """Predicted B bins at A bins r_bins from warps [N, T] (A first): mean over B samples."""
    w = torch.as_tensor(warps, dtype=torch.float64)
    sid = torch.tensor([0] * n_a + [1] * (len(w) - n_a))
    return predict_query_bins(w, sid, r_bins).mean(dim=0).numpy()


def cow_predict(a: Batch, b: Batch, r_bins: list[int]) -> np.ndarray:
    """Median over A[0..N_REF_PAIRS−1] of COW's inverse warp (A bin → B[0] bin) at r_bins."""
    preds = [
        invert_src(cow_tic_align(a.tic[i], b.tic[0]).src, np.asarray(r_bins, float))
        for i in range(min(N_REF_PAIRS, len(a)))
    ]
    return np.median(preds, axis=0)


@dataclass
class PairResult:
    name_a: str
    name_b: str
    anchors: Anchors
    errors: dict[str, np.ndarray] = field(default_factory=dict)  # method → per anchor (min)
    in_sample: np.ndarray | None = None

    @property
    def skipped(self) -> bool:
        return len(self.anchors) < 2


def compute_pair(
    a: Batch, b: Batch, ckpt: Path, device: torch.device, seed: int, calib_steps: int
) -> PairResult:
    anchors = extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw).anchors
    res = PairResult(a.name, b.name, anchors)
    log.info('%s: %d anchors %s', pair_label(a.name, b.name), len(anchors), anchors)
    if res.skipped:
        return res
    n_a, n_b = len(a), len(b)
    x, sid = batch_tensors(a.chromas, b.chromas, device)

    def calibrated_warps(anc: Anchors) -> np.ndarray:
        model = load_model(ckpt, device)
        calibrate(model, x, sid, anc, seed=seed, steps=calib_steps)
        return predict(model, x, sid)[1]

    r_bins = [r for r, _ in anchors]
    q_obs = np.array([q for _, q in anchors], dtype=float)
    res.in_sample = np.abs(warps_predict(calibrated_warps(anchors), n_a, r_bins) - q_obs) * BIN_MIN
    cow_q = cow_predict(a, b, r_bins)
    err: dict[str, list[float]] = {m: [] for m in METHODS}
    for k, (r, q) in enumerate(anchors):
        rest = anchors[:k] + anchors[k + 1 :]
        w_free = free_warp(n_a, n_b, rest, FREE_WARP_LR, seed, calib_steps)[0]
        pred = {
            'No alignment': float(r),
            'Constant shift': r + anchor_drift(rest),
            'Linear warp': float(np.polyval(anchor_line(rest), r)),
            'PCHIP (pooled anchors)': predict_query_bin(rest, r),
            'COW-TIC': float(cow_q[k]),
            FREE: float(warps_predict(w_free, n_a, [r])[0]),
            'WarpTransformer': float(warps_predict(calibrated_warps(rest), n_a, [r])[0]),
        }
        for m in METHODS:
            err[m].append(abs(pred[m] - q) * BIN_MIN)
        log.info(
            '  held out (%d, %d): %s', r, q, '  '.join(f'{m}={err[m][-1]:.3f}' for m in METHODS)
        )
    res.errors = {m: np.array(v) for m, v in err.items()}
    return res


def format_report(results: list[PairResult], seed: int, device: str, ckpt: Path) -> list[str]:
    lines = [
        title(__doc__),
        '',
        "Note: anchors are step 06's pooled set, outlier- and slope-filtered with the "
        'held-out anchor included.',
        f'Checkpoint {ckpt}   seed {seed}   device {device}',
    ]
    for p in results:
        label = pair_label(p.name_a, p.name_b)
        if p.skipped:
            lines += [
                '',
                f'{label}: {len(p.anchors)} anchor(s); at least 2 are needed to hold '
                'one out. Skipped.',
            ]
            continue
        assert p.in_sample is not None
        rows = [
            [m, e.mean(), e.max(), ' '.join(f'{v:.2f}' for v in e)] for m, e in p.errors.items()
        ]
        rows.append(
            [
                'WT in-sample',
                p.in_sample.mean(),
                p.in_sample.max(),
                ' '.join(f'{v:.2f}' for v in p.in_sample) + '  (calibrated on all anchors)',
            ]
        )
        lines += [
            '',
            f'{label}   {len(p.anchors)} anchors (A bin, B bin): {p.anchors}',
            *table(['Method', 'mean (min)', 'max (min)', 'per anchor (min)'], rows),
        ]
    done = [p for p in results if not p.skipped]
    if done:
        lines += [
            '',
            'Summary: mean / max held-out error (min)',
            *table(
                ['Pair', *METHODS],
                [
                    [
                        pair_label(p.name_a, p.name_b),
                        *(f'{p.errors[m].mean():.3f} / {p.errors[m].max():.3f}' for m in METHODS),
                    ]
                    for p in done
                ],
            ),
        ]
    lines += [
        '',
        'End anchors are extrapolated by every method once held out; read the per-anchor '
        'values, not only the means.',
    ]
    return lines


def save(results: list[PairResult], lines: list[str], paths: Paths) -> None:
    saved: dict[str, np.ndarray] = {}
    for p in results:
        key = f'{p.name_a}_{p.name_b}'
        saved[f'{key}_anchors'] = np.array(p.anchors).reshape(-1, 2)
        for m, e in p.errors.items():
            saved[f'{key}_{m.replace(" ", "_")}'] = e
        if p.in_sample is not None:
            saved[f'{key}_wt_in_sample'] = p.in_sample
    paths.ensure_results()
    save_npz(paths.per_pair / 'heldout_anchor_error.npz', saved)
    write_report(paths.results / 'heldout_anchor_error.txt', lines)


def run(
    paths: Paths,
    seed: int = 0,
    ckpt: Path | None = None,
    device: str = 'cpu',
    calib_steps: int = N_CALIB_STEPS,
    pairs: tuple[tuple[str, str], ...] = EVAL_PAIRS,
) -> list[PairResult]:
    dev, ck = pick_device(device), ckpt_or_default(ckpt, paths)
    results = [
        compute_pair(
            load_batch(a, paths.fish_oil), load_batch(b, paths.fish_oil), ck, dev, seed, calib_steps
        )
        for a, b in pairs
    ]
    save(results, format_report(results, seed, str(dev), ck), paths)
    return results


def cli(argv: list[str] | None = None) -> None:
    ap = parser(__doc__)
    add_model_args(ap)
    args = ap.parse_args(argv)
    setup_logging()
    run(Paths(args.root), args.seed, args.ckpt, args.device, args.calib_steps)
