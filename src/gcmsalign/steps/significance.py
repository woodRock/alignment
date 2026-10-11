"""Step 12 — paired Wilcoxon signed-rank tests between all methods.

Reads the per-sample-pair TIC r arrays of steps 05 (Unaligned, COW-TIC, icoshift), 06
(WarpTransformer, one file per seed) and 07 (PCHIP). All are indexed by the same (A_i, B_j)
pair, i-major, so the tests are paired; the unaligned arrays of steps 05 and 06 are checked to
be identical.

Two levels, both two-sided:
  per sample pair  n = n_A × n_B. Not independent (each A_i is in n_B pairs), so optimistic.
  per A sample     r averaged over B_j, n = n_A. Conservative; recommended for the paper.
Holm correction over the 10 comparisons of the 5 methods within each batch pair and level.
The WarpTransformer is tested at --seed; for the other seeds, the number at which each
WarpTransformer comparison stays significant is reported.

Output: results/significance_tests.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ..config import EVAL_PAIRS, Paths, pair_label
from ..metrics import holm, paired_wilcoxon, per_a_mean
from ._common import parser, setup_logging, table, title, write_report

PCHIP = 'PCHIP (pooled anchors)'
METHODS = ('Unaligned', 'icoshift', PCHIP, 'COW-TIC', 'WarpTransformer')
ALPHA = 0.05
FAMILY = len(METHODS) * (len(METHODS) - 1) // 2


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


def load_methods(paths: Paths, a: str, b: str, seed: int) -> tuple[dict[str, np.ndarray], int, int]:
    base = np.load(paths.per_pair / f'tic_r_baselines_{a}_{b}.npz')
    calib = np.load(paths.per_pair / f'tic_r_calib_{a}_{b}_seed{seed}.npz')
    pooled = paths.per_pair / f'tic_r_pchip_pooled_{a}_{b}.npz'
    if not pooled.exists():
        raise FileNotFoundError(f'{pooled.name} missing: run step 07 first')
    pool = np.load(pooled)
    n_a, n_b = int(base['n_a']), int(base['n_b'])
    if not np.allclose(base['unaligned'], calib['unaligned'], atol=1e-6):
        raise PairingError(f'{a} × {b}: unaligned arrays of steps 05 and 06 differ')
    if (int(pool['n_a']), int(pool['n_b'])) != (n_a, n_b):
        raise PairingError(f'{a} × {b}: step 07 arrays are not {n_a}×{n_b}; rerun step 07')
    return (
        {
            'Unaligned': base['unaligned'],
            'icoshift': base['icoshift'],
            PCHIP: pool['pchip_pooled'],
            'COW-TIC': base['cow_tic'],
            'WarpTransformer': calib['warptransformer'],
        },
        n_a,
        n_b,
    )


def run_tests(data: dict[str, np.ndarray]) -> list[Test]:
    raw = [
        (m1, m2, *paired_wilcoxon(data[m2], data[m1]), float(np.median(data[m2] - data[m1])))
        for m1, m2 in combinations(METHODS, 2)
    ]
    adj = holm([r[3] for r in raw])
    return [
        Test(m1, m2, stat, p, float(pa), md)
        for (m1, m2, stat, p, md), pa in zip(raw, adj, strict=True)
    ]


def per_sample(data: dict[str, np.ndarray], n_a: int, n_b: int) -> dict[str, np.ndarray]:
    return {m: per_a_mean(v, n_a, n_b) for m, v in data.items()}


def seed_robustness(
    paths: Paths, a: str, b: str, seeds: list[int]
) -> tuple[list[int], dict[str, int]]:
    """Seeds found, and per other method the number at which WT vs it is significant (per A)."""
    counts = dict.fromkeys(METHODS[:-1], 0)
    found = []
    for s in seeds:
        if not (paths.per_pair / f'tic_r_calib_{a}_{b}_seed{s}.npz').exists():
            continue
        found.append(s)
        data, n_a, n_b = load_methods(paths, a, b, s)
        for t in run_tests(per_sample(data, n_a, n_b)):
            if t.involves('WarpTransformer'):
                counts[t.other('WarpTransformer')] += int(t.significant)
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


def report(paths: Paths, seed: int, seeds: list[int]) -> list[str]:
    lines = [
        title(__doc__),
        '',
        f'Methods ({len(METHODS)}): {", ".join(METHODS)}.',
        f'Holm family: all {FAMILY} pairwise comparisons, within each batch pair and level. '
        f'WarpTransformer seed tested: {seed}. * = significant at α = {ALPHA} after Holm.',
    ]
    summary = []
    for a, b in EVAL_PAIRS:
        data, n_a, n_b = load_methods(paths, a, b, seed)
        per_a = run_tests(per_sample(data, n_a, n_b))
        found, counts = seed_robustness(paths, a, b, seeds)
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
        ]
        summary.append(
            [pair_label(a, b), *(f'{t.p_holm:.2e}' for t in per_a if t.involves('WarpTransformer'))]
        )
    lines += [
        '',
        'Summary: per-A-sample Holm p, WarpTransformer vs each method',
        *table(['Pair', *METHODS[:-1]], summary),
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
