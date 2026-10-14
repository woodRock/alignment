"""Step 07 — the anchor baselines: constant shift, linear warp and PCHIP through the pooled
anchors, on every pair.

They get exactly the WarpTransformer's calibration anchors (checked against the saved step-06
run) and one warp each (warpalign.warps), applied to every batch-A sample (A → B frame); B is
unwarped. Deterministic, CPU.
  Constant shift   A shifted by the median anchor drift q − r (fractional bins)
  Linear warp      the least-squares line q = a·r + b through the anchors
  PCHIP            PCHIP between the outer anchors and slope 1 beyond them, so neither end of the
                   run is pinned (the WarpTransformer's warps are not pinned either)

Writes the per-pair arrays that steps 12 (significance), 13 (seed summary) and 14 (figures)
read, and compares with the saved arrays of steps 05 and 06 (--seed) on every metric.

Output: results/pchip_pooled_anchors.txt, results/per_pair/tic_r_pchip_pooled_<A>_<B>.npz
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import evaluation
from ..anchors import extract_anchors
from ..config import EVAL_PAIRS, EXCLUDED_PAIRS, Paths, pair_label
from ..data import load_batch
from ..evaluation import ANCHOR_BASELINES, METRICS
from ..metrics import holm, paired_wilcoxon
from ._common import parser, save_npz, setup_logging, table, title, write_report

PCHIP = 'PCHIP (pooled anchors)'
WT = 'WarpTransformer'
METHODS = ('Unaligned', *ANCHOR_BASELINES, 'COW-TIC', WT)
SAVED = {
    'Unaligned': ('calib', 'unaligned'),
    'COW-TIC': ('base', 'cow_tic'),
    WT: ('calib', 'warptransformer'),
}
COMPARISONS = (
    (WT, PCHIP, ''),
    (WT, PCHIP, '_valid'),
    (WT, 'Constant shift', ''),
    (WT, 'Linear warp', ''),
    ('COW-TIC', PCHIP, ''),
)


class SavedRunMismatch(RuntimeError):
    """The saved step-05/06 arrays do not belong to the current data or anchors."""


@dataclass
class PairResult:
    name_a: str
    name_b: str
    n_a: int
    n_b: int
    anchors: list[tuple[int, int]]
    baselines: dict[str, evaluation.PairEval]  # ANCHOR_BASELINES
    r: dict[str, dict[str, np.ndarray]]  # method → metric suffix → per-pair array

    @property
    def evaluated(self) -> bool:
        return (self.name_a, self.name_b) in EVAL_PAIRS

    def mean(self, method: str, sfx: str = '') -> float:
        return float(np.nanmean(self.r[method][sfx]))


def compute_pair(paths: Paths, name_a: str, name_b: str, seed: int) -> PairResult:
    a, b = load_batch(name_a, paths.fish_oil), load_batch(name_b, paths.fish_oil)
    saved = {
        'calib': np.load(paths.per_pair / f'tic_r_calib_{name_a}_{name_b}_seed{seed}.npz'),
        'base': np.load(paths.per_pair / f'tic_r_baselines_{name_a}_{name_b}.npz'),
    }
    if (int(saved['calib']['n_a']), int(saved['calib']['n_b'])) != (len(a), len(b)):
        raise SavedRunMismatch(
            f'{name_a} × {name_b}: saved arrays do not match the data; rerun steps 05 and 06'
        )
    anchors = extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw).anchors
    step06 = [(int(r), int(q)) for r, q in saved['calib']['anchors'].reshape(-1, 2)]
    if anchors != step06:
        raise SavedRunMismatch(
            f"{name_a} × {name_b}: anchors {anchors} differ from step 06's {step06}"
        )
    baselines = {k: fn(a, b, anchors) for k, (_, fn) in ANCHOR_BASELINES.items()}
    r = {k: e.metrics for k, e in baselines.items()}
    for m, (src, key) in SAVED.items():
        missing = [key + sfx for sfx in METRICS.values() if key + sfx not in saved[src].files]
        if missing:
            raise SavedRunMismatch(f'{name_a} × {name_b}: {missing} not saved; rerun 05 and 06')
        r[m] = {sfx: saved[src][key + sfx] for sfx in METRICS.values()}
    return PairResult(name_a, name_b, len(a), len(b), anchors, baselines, r)


def compute(paths: Paths, seed: int) -> list[PairResult]:
    return [compute_pair(paths, a, b, seed) for a, b in EVAL_PAIRS + EXCLUDED_PAIRS]


def holm_tests(results: list[PairResult], x: str, y: str, sfx: str = '') -> list[list[object]]:
    """Per-A-sample Wilcoxon of x vs y on metric sfx on each evaluated pair, Holm over the
    pairs. Masked metrics: a pair's NaNs are ignored in the per-A mean."""
    ev = [p for p in results if p.evaluated]

    def per_a(p: PairResult, m: str) -> np.ndarray:
        return np.nanmean(p.r[m][sfx].reshape(p.n_a, p.n_b), axis=1)

    diffs = [(per_a(p, x), per_a(p, y)) for p in ev]
    ps = [paired_wilcoxon(dx, dy)[1] for dx, dy in diffs]
    return [
        [
            pair_label(p.name_a, p.name_b),
            f'{pv:.2e}',
            f'{pa:.2e}',
            f'{float(np.median(dx - dy)):+.4f}',
        ]
        for p, pv, pa, (dx, dy) in zip(ev, ps, holm(ps), diffs, strict=True)
    ]


def format_report(results: list[PairResult], seed: int) -> list[str]:
    lines = [
        title(__doc__),
        '',
        f'Comparison arrays: tic_r_calib_*_seed{seed}.npz (06), tic_r_baselines_*.npz (05). '
        'σ: both batches smoothed with the same Gaussian (bins); non-empty: bins with raw '
        'TIC > 0 in both samples.',
    ]
    for p in results:
        mags = '   '.join(f'{k} {e.warp_mag:.3f}' for k, e in p.baselines.items())
        lines += [
            '',
            f'{pair_label(p.name_a, p.name_b)}   n = {p.n_a}×{p.n_b}   anchors '
            f'{len(p.anchors)}: {p.anchors}' + ('' if p.evaluated else '   (excluded pair)'),
            f'  warp magnitude (min): {mags}',
            *table(
                ['Method', *METRICS],
                [[m, *(p.mean(m, s) for s in METRICS.values())] for m in METHODS],
            ),
        ]
    for label, sfx in (('TIC r', ''), ('raw r σ1', '_raw_s1'), ('r in range', '_valid')):
        lines += [
            '',
            f'Summary: mean {label}',
            *table(
                ['Pair', *METHODS, 'WT − PCHIP', 'WT − shift'],
                [
                    [
                        pair_label(p.name_a, p.name_b),
                        *(p.mean(m, sfx) for m in METHODS),
                        f'{p.mean(WT, sfx) - p.mean(PCHIP, sfx):+.3f}',
                        f"{p.mean(WT, sfx) - p.mean('Constant shift', sfx):+.3f}",
                    ]
                    for p in results
                ],
            ),
        ]
    lines += [
        '',
        'Paired two-sided Wilcoxon per A sample (r averaged over B_j); Holm over the 4 '
        'evaluated pairs, separately per comparison. The full table is '
        'significance_tests.txt (step 12).',
    ]
    for x, y, sfx in COMPARISONS:
        metric = next(k for k, v in METRICS.items() if v == sfx)
        lines += [
            f'  {x} vs {y} ({metric}):',
            *table(
                ['Pair', 'p', 'Holm p', 'median Δr'], holm_tests(results, x, y, sfx), indent='    '
            ),
        ]
    return lines


def save(results: list[PairResult], seed: int, paths: Paths) -> None:
    paths.ensure_results()
    for p in results:
        arrays: dict[str, object] = dict(
            anchors=np.array(p.anchors).reshape(-1, 2),
            n_a=p.n_a,
            n_b=p.n_b,
            sil=p.baselines[PCHIP].sil,  # PCHIP's, as read by step 14
            warp_mag=p.baselines[PCHIP].warp_mag,
            **{f'unaligned{sfx}': v for sfx, v in p.r['Unaligned'].items()},
        )
        for name, (key, _) in ANCHOR_BASELINES.items():
            e = p.baselines[name]
            arrays.update(e.arrays(key))
            arrays.update(
                {f'{key}_sil': e.sil, f'{key}_warp_mag': e.warp_mag, f'{key}_outside': e.outside}
            )
        save_npz(paths.per_pair / f'tic_r_pchip_pooled_{p.name_a}_{p.name_b}.npz', arrays)
    write_report(paths.results / 'pchip_pooled_anchors.txt', format_report(results, seed))


def run(paths: Paths, seed: int = 0) -> list[PairResult]:
    results = compute(paths, seed)
    save(results, seed, paths)
    return results


def cli(argv: list[str] | None = None) -> None:
    ap = parser(__doc__)
    ap.add_argument(
        '--seed',
        type=int,
        default=0,
        help='which saved step-06 run (tic_r_calib_*_seed<N>.npz) to compare with',
    )
    args = ap.parse_args(argv)
    setup_logging()
    run(Paths(args.root), args.seed)
