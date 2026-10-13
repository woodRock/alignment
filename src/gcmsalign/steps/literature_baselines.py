"""Step 05 — COW-TIC and icoshift on one batch pair.

Every batch-A sample is aligned onto every batch-B sample using the TIC only
(gcmsalign.baselines). TIC r is Pearson r between the warped A TIC and the B TIC on the common
acquisition window. The same warp, optimised on the normalised TIC, is applied to the raw TIC
for the raw-TIC rows. Smoothed r: both TICs smoothed with the same Gaussian before correlating
(config.SMOOTH_SIGMAS), as for every other method. Study silhouette: every A sample aligned to
B[0] (full chromatogram), B unwarped.

Output: results/literature_baselines_<A>_<B>.txt, results/per_pair/tic_r_baselines_<A>_<B>.npz
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .. import evaluation
from ..baselines import COW_N_SEGS, COW_SLACK, ICO_MAX_SHIFT, ICO_N_INTERVALS, METHODS, TicAlignment
from ..config import BIN_MIN, SMOOTH_SIGMAS, Paths, pair_label
from ..data import Batch, load_batch
from ..evaluation import coverage, identity_src, in_range, smoothed_arrays
from ..metrics import paired_wilcoxon, pearson, smooth_rows, study_silhouette
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


@dataclass
class MethodResult:
    r: np.ndarray  # per (A_i, B_j), i-major
    r_raw: np.ndarray
    warp_mag: float  # minutes, mean over pairs
    sil: float
    outside: float  # fraction of evaluated (bin, sample) values read outside the acquired range
    r_valid: np.ndarray  # per pair, TIC r on in-range bins only
    r_smooth: dict[float, np.ndarray] = field(default_factory=dict)  # σ → per pair, smoothed
    r_raw_smooth: dict[float, np.ndarray] = field(default_factory=dict)


@dataclass
class BaselineResult:
    name_a: str
    name_b: str
    n_a: int
    n_b: int
    methods: dict[str, MethodResult]  # 'Unaligned', 'COW-TIC', 'icoshift'


def evaluate(method: Aligner, a: Batch, b: Batch) -> MethodResult:
    """Align every A_i onto every B_j. Coverage per pair: A_i's warped bins must read inside
    A_i's acquired range; B_j is not warped."""
    r, r_raw, r_valid, mags, outside = [], [], [], [], []
    r_s: dict[float, list[float]] = {s: [] for s in SMOOTH_SIGMAS}
    r_raw_s: dict[float, list[float]] = {s: [] for s in SMOOTH_SIGMAS}
    b_s = {s: smooth_rows(b.tic, s) for s in SMOOTH_SIGMAS}
    b_raw_s = {s: smooth_rows(b.tic_raw, s) for s in SMOOTH_SIGMAS}
    start_a, start_b = a.acquisition_start, b.acquisition_start
    for i, (qa, qa_raw) in enumerate(zip(a.tic, a.tic_raw, strict=True)):
        for j, (rb, rb_raw) in enumerate(zip(b.tic, b.tic_raw, strict=True)):
            al = method(qa, rb)
            r.append(pearson(al.warped, rb))
            al_raw = warp_signal(qa_raw, al.src)
            r_raw.append(pearson(al_raw, rb_raw))
            for s in SMOOTH_SIGMAS:
                r_s[s].append(pearson(smooth_rows(al.warped, s), b_s[s][j]))
                r_raw_s[s].append(pearson(smooth_rows(al_raw, s), b_raw_s[s][j]))
            mags.append(al.warp_mag)
            va = in_range(al.src, start_a[i : i + 1])
            vb = in_range(identity_src(), start_b[j : j + 1])
            out, rv = coverage(al.warped[None], rb[None], va, vb)
            outside.append(out)
            r_valid.append(rv[0])
    to_b0 = np.stack([method(qa, b.tic[0]).src for qa in a.tic])
    return MethodResult(
        np.array(r),
        np.array(r_raw),
        float(np.mean(mags)),
        study_silhouette(warp_rows(a.chromas, to_b0), b.chromas),
        float(np.mean(outside)),
        np.array(r_valid),
        {s: np.array(v) for s, v in r_s.items()},
        {s: np.array(v) for s, v in r_raw_s.items()},
    )


def compute(a: Batch, b: Batch) -> BaselineResult:
    un = evaluation.unaligned(a, b)
    methods = {
        'Unaligned': MethodResult(
            un.r, un.r_raw, 0.0, un.sil, un.outside, un.r_valid, un.r_smooth, un.r_raw_smooth
        )
    }
    for name, method in METHODS.items():
        log.info('Evaluating %s …', name)
        methods[name] = evaluate(method, a, b)
    return BaselineResult(a.name, b.name, len(a), len(b), methods)


def format_report(res: BaselineResult) -> list[str]:
    u = res.methods['Unaligned']
    rows = [
        [
            k,
            m.r.mean(),
            m.r.mean() - u.r.mean(),
            m.r_raw.mean(),
            m.r_raw.mean() - u.r_raw.mean(),
            m.sil,
            f'{m.warp_mag:.3f}m',
            f'{100 * m.outside:.1f}%',
            float(np.nanmean(m.r_valid)),
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
        'TIC r after smoothing both batches with the same Gaussian (σ in bins):',
        *table(
            [
                'Method',
                *(f'TIC r σ{s:g}' for s in SMOOTH_SIGMAS),
                'raw r',
                *(f'raw r σ{s:g}' for s in SMOOTH_SIGMAS),
            ],
            [
                [
                    k,
                    *(m.r_smooth[s].mean() for s in SMOOTH_SIGMAS),
                    m.r_raw.mean(),
                    *(m.r_raw_smooth[s].mean() for s in SMOOTH_SIGMAS),
                ]
                for k, m in res.methods.items()
            ],
        ),
        '',
        'Paired two-sided Wilcoxon over sample pairs (not independent; descriptive):',
        *table(['Comparison', 'p'], tests),
    ]


def save(res: BaselineResult, paths: Paths) -> None:
    paths.ensure_results()
    m = res.methods
    save_npz(
        paths.per_pair / f'tic_r_baselines_{res.name_a}_{res.name_b}.npz',
        dict(
            unaligned=m['Unaligned'].r,
            cow_tic=m['COW-TIC'].r,
            icoshift=m['icoshift'].r,
            unaligned_raw=m['Unaligned'].r_raw,
            cow_tic_raw=m['COW-TIC'].r_raw,
            icoshift_raw=m['icoshift'].r_raw,
            n_a=res.n_a,
            n_b=res.n_b,
            sil=np.array([m[k].sil for k in ('Unaligned', 'COW-TIC', 'icoshift')]),
            warp_mag=np.array([m[k].warp_mag for k in ('COW-TIC', 'icoshift')]),
            outside=np.array([m[k].outside for k in ('Unaligned', 'COW-TIC', 'icoshift')]),
            unaligned_valid=m['Unaligned'].r_valid,
            cow_tic_valid=m['COW-TIC'].r_valid,
            icoshift_valid=m['icoshift'].r_valid,
            **{
                k: v
                for key, name in (
                    ('unaligned', 'Unaligned'),
                    ('cow_tic', 'COW-TIC'),
                    ('icoshift', 'icoshift'),
                )
                for k, v in smoothed_arrays(key, m[name].r_smooth, m[name].r_raw_smooth).items()
            },
        ),
    )
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
