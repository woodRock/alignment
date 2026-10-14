"""Step 05 — COW-TIC and icoshift on one batch pair, and the resampling control.

Every batch-A sample is aligned onto every batch-B sample using the TIC only
(gcmsalign.baselines). TIC r is Pearson r between the warped A TIC and the B TIC on the common
acquisition window. The same warp, optimised on the normalised TIC, is applied to the raw TIC
for the raw-TIC rows. Smoothed r: both TICs smoothed with the same Gaussian before correlating
(config.SMOOTH_SIGMAS), as for every other method. Study silhouette: every A sample aligned to
B[0] (full chromatogram), B unwarped.

Resampling control (no alignment): batch A read 0.25, 0.5 and 0.75 bin later, and both batches
0.5 bin later, by linear interpolation. Shows how much TIC r falls from resampling alone.

Output: results/literature_baselines_<A>_<B>.txt, results/per_pair/tic_r_baselines_<A>_<B>.npz
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .. import evaluation
from ..baselines import COW_N_SEGS, COW_SLACK, ICO_MAX_SHIFT, ICO_N_INTERVALS, METHODS, TicAlignment
from ..config import BIN_MIN, Paths, pair_label
from ..data import Batch, load_batch
from ..evaluation import METRICS, PairEval, identity_src, in_range, tic_metrics
from ..metrics import paired_wilcoxon, study_silhouette
from ..warps import warp_rows, warp_signal
from ._common import (
    add_pair_args,
    batch_name_of,
    parser,
    save_npz,
    setup_logging,
    table,
    title,
    write_report,
)

log = logging.getLogger('gcmsalign')
Aligner = Callable[[np.ndarray, np.ndarray], TicAlignment]
KEYS = {'Unaligned': 'unaligned', 'COW-TIC': 'cow_tic', 'icoshift': 'icoshift'}
SHIFTS = {  # control name → (shift of A, shift of B) in bins, and its saved-array key
    'A shifted ¼ bin': (0.25, 0.0, 'shift_a_q1'),
    'A shifted ½ bin': (0.5, 0.0, 'shift_a_q2'),
    'A shifted ¾ bin': (0.75, 0.0, 'shift_a_q3'),
    'both shifted ½ bin': (0.5, 0.5, 'shift_both_q2'),
}


@dataclass
class BaselineResult:
    name_a: str
    name_b: str
    n_a: int
    n_b: int
    methods: dict[str, PairEval]  # 'Unaligned', 'COW-TIC', 'icoshift'
    shifts: dict[str, PairEval]  # resampling controls (SHIFTS)


def evaluate(method: Aligner, a: Batch, b: Batch) -> PairEval:
    """Align every A_i onto every B_j (i-major). Coverage per pair: A_i's warped bins must read
    inside A_i's acquired range; B_j is not warped."""
    n_b = len(b)
    als = [method(qa, rb) for qa in a.tic for rb in b.tic]
    src = np.stack([al.src for al in als])
    idx_a = np.repeat(np.arange(len(a)), n_b)
    idx_b = np.tile(np.arange(n_b), len(a))
    met, outside = tic_metrics(
        np.stack([al.warped for al in als]),
        b.tic[idx_b],
        np.stack([warp_signal(a.tic_raw[i], s) for i, s in zip(idx_a, src, strict=True)]),
        b.tic_raw[idx_b],
        in_range(src, a.acquisition_start[idx_a]),
        in_range(identity_src(), b.acquisition_start[idx_b]),
        paired=True,
    )
    to_b0 = np.stack([method(qa, b.tic[0]).src for qa in a.tic])
    return PairEval(
        met,
        study_silhouette(warp_rows(a.chromas, to_b0), b.chromas),
        float(np.mean([al.warp_mag for al in als])),
        outside,
    )


def compute(a: Batch, b: Batch) -> BaselineResult:
    methods = {'Unaligned': evaluation.unaligned(a, b)}
    for name, method in METHODS.items():
        log.info('Evaluating %s …', name)
        methods[name] = evaluate(method, a, b)
    shifts = {k: evaluation.shifted(a, b, sa, sb) for k, (sa, sb, _) in SHIFTS.items()}
    return BaselineResult(a.name, b.name, len(a), len(b), methods, shifts)


def metric_table(evals: dict[str, PairEval], first: str = 'Method') -> list[str]:
    """Every method × every metric of METRICS (means over sample pairs)."""
    return table(
        [first, *METRICS], [[k, *(e.m(sfx) for sfx in METRICS.values())] for k, e in evals.items()]
    )


def format_report(res: BaselineResult) -> list[str]:
    u = res.methods['Unaligned']
    rows = [
        [
            k,
            m.mean,
            m.mean - u.mean,
            m.mean_raw,
            m.mean_raw - u.mean_raw,
            m.sil,
            f'{m.warp_mag:.3f}m',
            f'{100 * m.outside:.1f}%',
            m.mean_valid,
        ]
        for k, m in res.methods.items()
    ]
    names = list(res.methods)
    tests = [
        [f'{y} vs {x}', f'{paired_wilcoxon(res.methods[y].r, res.methods[x].r)[1]:.2e}']
        for i, x in enumerate(names)
        for y in names[i + 1 :]
    ]
    return [
        title(__doc__),
        '',
        f'{pair_label(res.name_a, res.name_b)}: {res.name_a} ({res.n_a}) × {res.name_b} '
        f'({res.n_b}) = {res.n_a * res.n_b} pairs',
        f'COW-TIC: {COW_N_SEGS} segments, slack ±{COW_SLACK} bins '
        f'(±{COW_SLACK * BIN_MIN:.1f} min); icoshift: {ICO_N_INTERVALS} intervals, max shift '
        f'±{ICO_MAX_SHIFT} bins',
        '',
        *table(
            [
                'Method',
                'TIC r',
                'Δ unal',
                'raw r',
                'Δ unal',
                'Study sil',
                'Warp mag',
                'outside',
                'r in range',
            ],
            rows,
        ),
        'outside: share of evaluated (bin, sample) values whose warped source lies outside the '
        "run's acquired range; r in range: TIC r with those values masked out.",
        '',
        'Every TIC metric (σ: both batches smoothed with the same Gaussian, in bins; non-empty: '
        'bins with raw TIC > 0 in both samples):',
        *metric_table(res.methods),
        '',
        'Resampling control (no alignment; linear interpolation at a fractional shift):',
        *metric_table({'Unaligned': u, **res.shifts}, 'Control'),
        '',
        'Paired two-sided Wilcoxon over sample pairs (not independent; descriptive):',
        *table(['Comparison', 'p'], tests),
    ]


def save(res: BaselineResult, paths: Paths) -> None:
    paths.ensure_results()
    m = res.methods
    arrays: dict[str, object] = {
        'n_a': res.n_a,
        'n_b': res.n_b,
        'sil': np.array([m[k].sil for k in KEYS]),
        'warp_mag': np.array([m[k].warp_mag for k in ('COW-TIC', 'icoshift')]),
        'outside': np.array([m[k].outside for k in KEYS]),
    }
    for name, key in KEYS.items():
        arrays.update(m[name].arrays(key))
    for name, (_, _, key) in SHIFTS.items():
        arrays.update(res.shifts[name].arrays(key))
    save_npz(paths.per_pair / f'tic_r_baselines_{res.name_a}_{res.name_b}.npz', arrays)
    write_report(
        paths.results / f'literature_baselines_{res.name_a}_{res.name_b}.txt', format_report(res)
    )


def run(paths: Paths, name_a: str, name_b: str) -> BaselineResult:
    res = compute(load_batch(name_a, paths.fish_oil), load_batch(name_b, paths.fish_oil))
    save(res, paths)
    return res


def cli(argv: list[str] | None = None) -> None:
    ap = parser(__doc__)
    add_pair_args(ap)
    args = ap.parse_args(argv)
    setup_logging()
    run(Paths(args.root), batch_name_of(args.study_a), batch_name_of(args.study_b))
