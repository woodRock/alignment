"""Step 12 — paired Wilcoxon signed-rank tests between all methods.

Reads the per-sample-pair TIC r arrays of steps 05 (Unaligned, COW-TIC, icoshift), 06
(WarpTransformer, one file per seed) and 07 (constant shift, linear warp, PCHIP). All are
indexed by the same (A_i, B_j) pair, i-major, so the tests are paired; the unaligned arrays of
steps 05 and 06 are checked to be identical.

Two levels, both two-sided:
  per sample pair  n = n_A × n_B. Not independent (each A_i is in n_B pairs), so optimistic.
  per A sample     r averaged over B_j, n = n_A. Conservative; recommended for the paper.
Holm correction over the 21 comparisons of the 7 methods within each batch pair and level.
The WarpTransformer is tested at --seed; for the other seeds, the number at which each
WarpTransformer comparison stays significant is reported.

Also reported, per A sample: the same 21-comparison family on the raw TIC smoothed with σ = 1
bin (both batches; the form in which raw TIC r is robust to resampling), and WarpTransformer vs
PCHIP on TIC r with values read outside the acquired range masked ("r in range"; Holm over the
4 pairs).

Output: results/significance_tests.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ..config import EVAL_PAIRS, Paths, pair_label
from ..metrics import holm, paired_wilcoxon
from ._common import parser, setup_logging, table, title, write_report

PCHIP = 'PCHIP (pooled anchors)'
WT = 'WarpTransformer'
METHODS = ('Unaligned', 'icoshift', 'Constant shift', 'Linear warp', PCHIP, 'COW-TIC', WT)
SOURCES = {  # method → (step whose arrays hold it, array key)
    'Unaligned': ('base', 'unaligned'),
    'icoshift': ('base', 'icoshift'),
    'Constant shift': ('pool', 'constant_shift'),
    'Linear warp': ('pool', 'linear'),
    PCHIP: ('pool', 'pchip_pooled'),
    'COW-TIC': ('base', 'cow_tic'),
    WT: ('calib', 'warptransformer'),
}
ALPHA = 0.05
FAMILY = len(METHODS) * (len(METHODS) - 1) // 2
RAW_S1 = '_raw_s1'
MASKED = '_valid'


class PairingError(ValueError):
    """Saved arrays that should describe the same sample pairs do not."""


@dataclass
class Test:
    m1: str
    m2: str
    stat: float
    p: float
    p_holm: float
    median_diff: float  # median of m2 − m1

    @property
    def significant(self) -> bool:
        return self.p_holm < ALPHA

    def involves(self, method: str) -> bool:
        return method in (self.m1, self.m2)

    def other(self, method: str) -> str:
        return self.m1 if self.m2 == method else self.m2


def load_methods(
    paths: Paths, a: str, b: str, seed: int, sfx: str = ''
) -> tuple[dict[str, np.ndarray], int, int]:
    """Per-sample-pair arrays of every method on one metric (evaluation.METRICS suffix)."""
    pooled = paths.per_pair / f'tic_r_pchip_pooled_{a}_{b}.npz'
    if not pooled.exists():
        raise FileNotFoundError(f'{pooled.name} missing: run step 07 first')
    z = {
        'base': np.load(paths.per_pair / f'tic_r_baselines_{a}_{b}.npz'),
        'calib': np.load(paths.per_pair / f'tic_r_calib_{a}_{b}_seed{seed}.npz'),
        'pool': np.load(pooled),
    }
    n_a, n_b = int(z['base']['n_a']), int(z['base']['n_b'])
    if not np.allclose(z['base']['unaligned'], z['calib']['unaligned'], atol=1e-6):
        raise PairingError(f'{a} × {b}: unaligned arrays of steps 05 and 06 differ')
    if (int(z['pool']['n_a']), int(z['pool']['n_b'])) != (n_a, n_b):
        raise PairingError(f'{a} × {b}: step 07 arrays are not {n_a}×{n_b}; rerun step 07')
    return {m: z[src][key + sfx] for m, (src, key) in SOURCES.items()}, n_a, n_b


def run_tests(data: dict[str, np.ndarray], methods: tuple[str, ...] = METHODS) -> list[Test]:
    raw = [
        (m1, m2, *paired_wilcoxon(data[m2], data[m1]), float(np.median(data[m2] - data[m1])))
        for m1, m2 in combinations(methods, 2)
    ]
    adj = holm([r[3] for r in raw])
    return [
        Test(m1, m2, stat, p, float(pa), md)
        for (m1, m2, stat, p, md), pa in zip(raw, adj, strict=True)
    ]


def per_sample(data: dict[str, np.ndarray], n_a: int, n_b: int) -> dict[str, np.ndarray]:
    """r averaged over B_j per A sample (NaN-ignoring: masked metrics can be undefined)."""
    return {m: np.nanmean(np.asarray(v).reshape(n_a, n_b), axis=1) for m, v in data.items()}


def seed_robustness(
    paths: Paths, a: str, b: str, seeds: list[int], sfx: str = ''
) -> tuple[list[int], dict[str, int]]:
    """Seeds found, and per other method the number at which WT vs it is significant (per A)."""
    counts = dict.fromkeys(METHODS[:-1], 0)
    found = []
    for s in seeds:
        if not (paths.per_pair / f'tic_r_calib_{a}_{b}_seed{s}.npz').exists():
            continue
        found.append(s)
        data, n_a, n_b = load_methods(paths, a, b, s, sfx)
        for t in run_tests(per_sample(data, n_a, n_b)):
            if t.involves(WT):
                counts[t.other(WT)] += int(t.significant)
    return found, counts


def format_tests(tests: list[Test], heading: str) -> list[str]:
    return [
        heading,
        *table(
            ['Comparison (B vs A)', 'W', 'p', 'p (Holm)', '', 'median Δr'],
            [
                [
                    f'{t.m2} vs {t.m1}',
                    f'{t.stat:.1f}',
                    f'{t.p:.3e}',
                    f'{t.p_holm:.3e}',
                    '*' if t.significant else '',
                    f'{t.median_diff:+.4f}',
                ]
                for t in tests
            ],
        ),
    ]


def masked_wt_vs_pchip(paths: Paths, seed: int, seeds: list[int]) -> list[str]:
    """WT vs PCHIP per A sample on r in range, Holm over the evaluated pairs, and the number of
    seeds at which it stays significant."""
    rows, ps = [], []
    for a, b in EVAL_PAIRS:
        data, n_a, n_b = load_methods(paths, a, b, seed, MASKED)
        pa = per_sample(data, n_a, n_b)
        ps.append(paired_wilcoxon(pa[WT], pa[PCHIP])[1])
        found = [s for s in seeds if (paths.per_pair / f'tic_r_calib_{a}_{b}_seed{s}.npz').exists()]
        margins = []
        for s in found:
            d = per_sample(*load_methods(paths, a, b, s, MASKED))
            margins.append(float(d[WT].mean() - d[PCHIP].mean()))
        rows.append(
            [
                pair_label(a, b),
                float(np.nanmean(data[WT])),
                float(np.nanmean(data[PCHIP])),
                f'{float(np.median(pa[WT] - pa[PCHIP])):+.4f}',
                f'{sum(m > 0 for m in margins)}/{len(found)}',
            ]
        )
    return [
        '',
        f'WarpTransformer vs PCHIP on r in range (values read outside the acquired range '
        f'masked), per A sample, seed {seed}; Holm over the {len(EVAL_PAIRS)} pairs:',
        *table(
            ['Pair', 'WT', 'PCHIP', 'median Δr', 'seeds WT > PCHIP', 'p', 'p (Holm)'],
            [
                [*r, f'{p:.3e}', f'{ph:.3e}' + (' *' if ph < ALPHA else '')]
                for r, p, ph in zip(rows, ps, holm(ps), strict=True)
            ],
        ),
    ]


def report(paths: Paths, seed: int, seeds: list[int]) -> list[str]:
    lines = [
        title(__doc__),
        '',
        f'Methods ({len(METHODS)}): {", ".join(METHODS)}.',
        f'Holm family: all {FAMILY} pairwise comparisons, within each batch pair, level and '
        f'metric. WarpTransformer seed tested: {seed}. * = significant at α = {ALPHA} after Holm.',
    ]
    summary, summary_raw = [], []
    for a, b in EVAL_PAIRS:
        data, n_a, n_b = load_methods(paths, a, b, seed)
        per_a = run_tests(per_sample(data, n_a, n_b))
        raw = load_methods(paths, a, b, seed, RAW_S1)[0]
        per_a_raw = run_tests(per_sample(raw, n_a, n_b))
        found, counts = seed_robustness(paths, a, b, seeds)
        _, counts_raw = seed_robustness(paths, a, b, seeds, RAW_S1)
        lines += [
            '',
            f'{pair_label(a, b)}   n_A = {n_a}  n_B = {n_b}   mean TIC r: '
            + '  '.join(f'{m} {data[m].mean():.3f}' for m in METHODS),
            *format_tests(
                run_tests(data), f'  Per sample pair (n = {n_a * n_b}; not independent):'
            ),
            *format_tests(per_a, f'  Per A sample (n = {n_a}; recommended):'),
            f'  Seeds available: {found}. WarpTransformer comparisons significant (per A '
            f'sample, Holm) at: '
            + '  '.join(f'vs {m} {k}/{len(found)}' for m, k in counts.items()),
            '  Raw TIC smoothed σ = 1 bin, mean r: '
            + '  '.join(f'{m} {raw[m].mean():.3f}' for m in METHODS),
            *format_tests(per_a_raw, f'  Per A sample, raw TIC σ1 (n = {n_a}):'),
            '  WarpTransformer comparisons significant on raw TIC σ1 (per A sample, Holm) at: '
            + '  '.join(f'vs {m} {k}/{len(found)}' for m, k in counts_raw.items()),
        ]
        summary.append([pair_label(a, b), *(f'{t.p_holm:.2e}' for t in per_a if t.involves(WT))])
        summary_raw.append(
            [pair_label(a, b), *(f'{t.p_holm:.2e}' for t in per_a_raw if t.involves(WT))]
        )
    lines += [
        '',
        'Summary: per-A-sample Holm p, WarpTransformer vs each method (normalised TIC r)',
        *table(['Pair', *METHODS[:-1]], summary),
        '',
        'Summary: per-A-sample Holm p, WarpTransformer vs each method (raw TIC r, σ1)',
        *table(['Pair', *METHODS[:-1]], summary_raw),
        *masked_wt_vs_pchip(paths, seed, seeds),
    ]
    return lines


def run(paths: Paths, seed: int = 0, seeds: list[int] | None = None) -> list[str]:
    lines = report(paths, seed, seeds if seeds is not None else [0, 1, 2, 3, 4])
    write_report(paths.results / 'significance_tests.txt', lines)
    return lines


def cli(argv: list[str] | None = None) -> None:
    ap = parser(__doc__)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = ap.parse_args(argv)
    setup_logging()
    run(Paths(args.root), args.seed, args.seeds)
