"""Step 09 — where does the WarpTransformer's gain over a spline come from?

On the four evaluated pairs, from one seeded calibration (step 06's settings):
  Unaligned
  WT original          the calibrated WarpTransformer: one warp per sample, both batches
  WT batch-shared      its warps averaged within each batch, so every sample of a batch gets
                       the same warp (a mean of monotone warps is monotone)
  WT residual only     identity + (w_i − batch mean): the within-batch part alone
  Free warp            no network: one monotone warp per batch with the WT head's
                       parameterisation, same loss and loop (calibration.free_warp), at the
                       WT learning rate and 100× it
  Pooled PCHIP         step 07: the spline through the same pooled anchors, A only
  Pooled PCHIP + COW   pooled PCHIP, then a per-sample COW refinement (8 segments, ±4 bins) of
                       every A_i and B_j onto the mean batch-B TIC (COW optimises TIC r itself)
  Smoothing controls   a pure half-bin shift (maximal interpolation smoothing, no alignment) of
                       both batches of Unaligned, and of batch B of Pooled PCHIP: the WT
                       resamples both batches, the baselines only A

Reports between-batch TIC r (the paper's metric), within-batch TIC r, the WT's within-batch
warp spread and the final losses. Warping is linear interpolation along RT, which commutes with
the sum over m/z, so every variant is evaluated on warped TICs.

Output: results/mechanism_ablation.txt, results/per_pair/mechanism_ablation_<A>_<B>.npz
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..anchors import extract_anchors
from ..baselines import cow_tic_align
from ..calibration import (
    batch_tensors,
    calibrate,
    final_losses,
    free_warp,
    load_model,
    pick_device,
    predict,
)
from ..config import (
    BIN_MIN,
    EVAL_PAIRS,
    FREE_WARP_LR,
    LR_CALIB,
    N_BINS,
    N_CALIB_STEPS,
    Paths,
    pair_label,
)
from ..data import load_batch
from ..metrics import holm, paired_wilcoxon, pearson_rows, per_a_mean, within_batch_r
from ..warps import pchip_warp, src_from_fn, warp_rows
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

log = logging.getLogger('gcmsalign')
REFINE_SEGS, REFINE_SLACK = 8, 4
FREE_LRS = (LR_CALIB, FREE_WARP_LR)
BEST_FREE = f'Free warp (lr {FREE_WARP_LR:g})'
COMPARISONS = (
    ('WT original', 'WT batch-shared'),
    ('WT batch-shared', BEST_FREE),
    (BEST_FREE, 'Pooled PCHIP'),
    ('WT original', 'Pooled PCHIP + COW refine'),
    ('Control: unaligned, both ½-bin shifted', 'Unaligned'),
)
Tics = tuple[np.ndarray, np.ndarray]


def cow_refine(tics: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.stack(
        [cow_tic_align(t, ref, n_segs=REFINE_SEGS, slack=REFINE_SLACK).warped for t in tics]
    )


def half_bin_shift(n: int) -> np.ndarray:
    return np.tile(np.clip(np.arange(N_BINS) + 0.5, 0, N_BINS - 1), (n, 1))


def batch_shared(w: np.ndarray, n_a: int) -> np.ndarray:
    """Each sample's warp replaced by its batch's mean warp."""
    return np.concatenate(
        [np.tile(w[:n_a].mean(0), (n_a, 1)), np.tile(w[n_a:].mean(0), (len(w) - n_a, 1))]
    )


def residual_only(w: np.ndarray, n_a: int) -> np.ndarray:
    """Identity + each sample's deviation from its batch's mean warp (clipped to the run)."""
    dev = w - batch_shared(w, n_a)
    return np.clip(dev + np.arange(N_BINS), 0, N_BINS - 1)


def variants(
    ta: np.ndarray,
    tb: np.ndarray,
    w_wt: np.ndarray,
    anchors: list[tuple[int, int]],
    seed: int,
    calib_steps: int,
) -> tuple[dict[str, Tics], dict[float, tuple[float, float]]]:
    """name → (A TICs, B TICs), and each free warp's final (anchor, smoothness) loss."""
    n_a, n_b = len(ta), len(tb)
    both = np.concatenate([ta, tb])

    def split(x: np.ndarray) -> Tics:
        return x[:n_a], x[n_a:]

    pa = warp_rows(ta, src_from_fn(pchip_warp(anchors)))
    out: dict[str, Tics] = {
        'Unaligned': (ta, tb),
        'WT original': split(warp_rows(both, w_wt)),
        'WT batch-shared': split(warp_rows(both, batch_shared(w_wt, n_a))),
        'WT residual only': split(warp_rows(both, residual_only(w_wt, n_a))),
    }
    losses = {}
    for lr in FREE_LRS:
        w_free, losses[lr] = free_warp(n_a, n_b, anchors, lr, seed, calib_steps)
        out[f'Free warp (lr {lr:g})'] = split(warp_rows(both, w_free))
    ref_b = tb.mean(0)
    out['Pooled PCHIP'] = (pa, tb)
    out['Pooled PCHIP + COW refine'] = (cow_refine(pa, ref_b), cow_refine(tb, ref_b))
    out['Control: unaligned, both ½-bin shifted'] = split(
        warp_rows(both, half_bin_shift(n_a + n_b))
    )
    out['Control: pooled PCHIP, B ½-bin shifted'] = (pa, warp_rows(tb, half_bin_shift(n_b)))
    return out, losses


@dataclass
class PairResult:
    name_a: str
    name_b: str
    n_a: int
    n_b: int
    anchors: list[tuple[int, int]]
    w_wt: np.ndarray
    tics: dict[str, Tics]
    r: dict[str, np.ndarray]
    loss_wt: tuple[float, float]
    loss_free: dict[float, tuple[float, float]]


def compute_pair(
    paths: Paths,
    name_a: str,
    name_b: str,
    ckpt: Path,
    device: torch.device,
    seed: int,
    calib_steps: int,
) -> PairResult:
    a, b = load_batch(name_a, paths.fish_oil), load_batch(name_b, paths.fish_oil)
    anchors = extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw).anchors
    log.info('%s: %d anchors — calibrating …', pair_label(name_a, name_b), len(anchors))
    model = load_model(ckpt, device)
    x, sid = batch_tensors(a.chromas, b.chromas, device)
    calibrate(model, x, sid, anchors, seed=seed, steps=calib_steps)
    aligned, w_wt = predict(model, x, sid)
    tics, loss_free = variants(a.tic, b.tic, w_wt, anchors, seed, calib_steps)
    r = {k: pearson_rows(xa, xb) for k, (xa, xb) in tics.items()}
    model_tic = aligned.sum(-1)
    if not np.allclose(
        r['WT original'], pearson_rows(model_tic[: len(a)], model_tic[len(a) :]), atol=1e-4
    ):
        raise RuntimeError('re-applying the WT warps does not reproduce the model output')
    return PairResult(
        name_a,
        name_b,
        len(a),
        len(b),
        anchors,
        w_wt,
        tics,
        r,
        final_losses(w_wt, sid, anchors),
        loss_free,
    )


def comparison_rows(results: list[PairResult], x: str, y: str) -> list[list[object]]:
    diffs = [(per_a_mean(p.r[x], p.n_a, p.n_b), per_a_mean(p.r[y], p.n_a, p.n_b)) for p in results]
    ps = [paired_wilcoxon(dx, dy)[1] for dx, dy in diffs]
    return [
        [pair_label(p.name_a, p.name_b), f'{pa:.2e}', f'{float(np.median(dx - dy)):+.4f}']
        for p, pa, (dx, dy) in zip(results, holm(ps), diffs, strict=True)
    ]


def format_report(results: list[PairResult], seed: int, device: str, ckpt: Path) -> list[str]:
    lines = [
        title(__doc__),
        '',
        f'Seed {seed}   device {device}   checkpoint {ckpt}   COW '
        f'refinement {REFINE_SEGS} segments ±{REFINE_SLACK} bins',
    ]
    for p in results:

        def spread(w: np.ndarray) -> float:
            return float(np.abs(w - w.mean(0)).mean()) * BIN_MIN

        u = float(p.r['Unaligned'].mean())
        free = '; '.join(
            f'free lr {lr:g} {lo[0]:.4f} / {lo[1]:.5f}' for lr, lo in p.loss_free.items()
        )
        lines += [
            '',
            f'{pair_label(p.name_a, p.name_b)}   n = {p.n_a}×{p.n_b}   anchors {len(p.anchors)}',
            f'  WT within-batch warp spread (mean |w_i − batch mean|): A '
            f'{spread(p.w_wt[: p.n_a]):.3f} min, B {spread(p.w_wt[p.n_a :]):.3f} min',
            f'  Final anchor / smoothness loss: WT {p.loss_wt[0]:.4f} / {p.loss_wt[1]:.5f}; {free}',
            *table(
                ['Variant', 'TIC r A×B', 'Δ unal', 'within A', 'within B'],
                [
                    [
                        k,
                        p.r[k].mean(),
                        p.r[k].mean() - u,
                        within_batch_r(xa)[0],
                        within_batch_r(xb)[0],
                    ]
                    for k, (xa, xb) in p.tics.items()
                ],
            ),
        ]
    lines += [
        '',
        'Summary: mean between-batch TIC r',
        *table(
            ['Variant', *(pair_label(p.name_a, p.name_b) for p in results)],
            [[k, *(p.r[k].mean() for p in results)] for k in results[0].r],
        ),
    ]
    lines += [
        '',
        'Paired two-sided Wilcoxon per A sample (r averaged over B_j); Holm over the 4 '
        'evaluated pairs, separately per comparison.',
        'Diagnostic tests; not part of the 10-comparison family of significance_tests.txt.',
    ]
    for x, y in COMPARISONS:
        lines += [
            f'  {x}  vs  {y}:',
            *table(['Pair', 'Holm p', 'median Δr'], comparison_rows(results, x, y), indent='    '),
        ]
    return lines


def save(results: list[PairResult], lines: list[str], paths: Paths) -> None:
    paths.ensure_results()
    for p in results:
        arrays: dict[str, object] = {k.replace(' ', '_'): v for k, v in p.r.items()}
        arrays.update(w_wt=p.w_wt, anchors=np.array(p.anchors).reshape(-1, 2), n_a=p.n_a, n_b=p.n_b)
        save_npz(paths.per_pair / f'mechanism_ablation_{p.name_a}_{p.name_b}.npz', arrays)
    write_report(paths.results / 'mechanism_ablation.txt', lines)


def run(
    paths: Paths,
    seed: int = 0,
    ckpt: Path | None = None,
    device: str = 'cpu',
    calib_steps: int = N_CALIB_STEPS,
) -> list[PairResult]:
    dev, ck = pick_device(device), ckpt_or_default(ckpt, paths)
    results = [compute_pair(paths, a, b, ck, dev, seed, calib_steps) for a, b in EVAL_PAIRS]
    save(results, format_report(results, seed, str(dev), ck), paths)
    return results


def cli(argv: list[str] | None = None) -> None:
    ap = parser(__doc__)
    add_model_args(ap)
    args = ap.parse_args(argv)
    setup_logging()
    run(Paths(args.root), args.seed, args.ckpt, args.device, args.calib_steps)
