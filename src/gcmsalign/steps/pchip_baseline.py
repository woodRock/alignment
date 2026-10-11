"""Step 07 — the PCHIP baseline: one spline through the pooled anchors, on every pair.

PCHIP gets exactly the WarpTransformer's calibration anchors (checked against the saved step-06
run) and gcmsalign.warps.pchip_warp: PCHIP between the outer anchors and slope 1 beyond them,
so neither end of the run is pinned (the WarpTransformer's warps are not pinned either). The one
warp is applied to every batch-A sample (A → B frame); B is unwarped. Deterministic, CPU.

Writes the per-pair arrays that steps 12 (significance) and 14 (figures) read, and compares
with the saved arrays of steps 05 and 06 (--seed).

Output: results/pchip_pooled_anchors.txt, results/per_pair/tic_r_pchip_pooled_<A>_<B>.npz
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import evaluation
from ..anchors import extract_anchors
from ..config import EVAL_PAIRS, EXCLUDED_PAIRS, Paths, pair_label
from ..data import load_batch
from ..metrics import holm, paired_wilcoxon, per_a_mean
from ._common import parser, setup_logging, table, title, write_report

PCHIP = 'PCHIP (pooled anchors)'
METHODS = ('Unaligned', PCHIP, 'COW-TIC', 'WarpTransformer')


class SavedRunMismatch(RuntimeError):
    """The saved step-05/06 arrays do not belong to the current data or anchors."""


@dataclass
class PairResult:
    name_a: str
    name_b: str
    n_a: int
    n_b: int
    anchors: list[tuple[int, int]]
    pchip: evaluation.PairEval
    r: dict[str, np.ndarray]  # method → per-pair normalised TIC r
    r_raw: dict[str, np.ndarray]

    @property
    def evaluated(self) -> bool:
        return (self.name_a, self.name_b) in EVAL_PAIRS


def compute_pair(paths: Paths, name_a: str, name_b: str, seed: int) -> PairResult:
    a, b = load_batch(name_a, paths.fish_oil), load_batch(name_b, paths.fish_oil)
    calib = np.load(paths.per_pair / f'tic_r_calib_{name_a}_{name_b}_seed{seed}.npz')
    base = np.load(paths.per_pair / f'tic_r_baselines_{name_a}_{name_b}.npz')
    if (int(calib['n_a']), int(calib['n_b'])) != (len(a), len(b)):
        raise SavedRunMismatch(
            f'{name_a} × {name_b}: saved arrays do not match the data; rerun steps 05 and 06'
        )
    anchors = extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw).anchors
    saved = [(int(r), int(q)) for r, q in calib['anchors'].reshape(-1, 2)]
    if anchors != saved:
        raise SavedRunMismatch(
            f'{name_a} × {name_b}: anchors {anchors} differ from step 06\'s {saved}'
        )
    pc = evaluation.pchip(a, b, anchors)
    return PairResult(
        name_a,
        name_b,
        len(a),
        len(b),
        anchors,
        pc,
        {
            'Unaligned': calib['unaligned'],
            PCHIP: pc.r,
            'COW-TIC': base['cow_tic'],
            'WarpTransformer': calib['warptransformer'],
        },
        {
            'Unaligned': calib['unaligned_raw'],
            PCHIP: pc.r_raw,
            'COW-TIC': base['cow_tic_raw'],
            'WarpTransformer': calib['warptransformer_raw'],
        },
    )


def compute(paths: Paths, seed: int) -> list[PairResult]:
    return [compute_pair(paths, a, b, seed) for a, b in EVAL_PAIRS + EXCLUDED_PAIRS]


def holm_tests(results: list[PairResult], x: str, y: str) -> list[list[object]]:
    """Per-A-sample Wilcoxon of x vs y on each evaluated pair, Holm over the pairs."""
    ev = [p for p in results if p.evaluated]
    diffs = [(per_a_mean(p.r[x], p.n_a, p.n_b), per_a_mean(p.r[y], p.n_a, p.n_b)) for p in ev]
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
        f'Comparison arrays: tic_r_calib_*_seed{seed}.npz (06), tic_r_baselines_*.npz (05).',
    ]
    for p in results:
        u, u_raw = p.r['Unaligned'].mean(), p.r_raw['Unaligned'].mean()
        lines += [
            '',
            f'{pair_label(p.name_a, p.name_b)}   n = {p.n_a}×{p.n_b}   anchors '
            f'{len(p.anchors)}: {p.anchors}   PCHIP warp {p.pchip.warp_mag:.3f} min'
            + ('' if p.evaluated else '   (excluded pair)'),
            *table(
                ['Method', 'TIC r', 'Δ unal', 'raw r', 'Δ unal'],
                [
                    [
                        m,
                        p.r[m].mean(),
                        p.r[m].mean() - u,
                        p.r_raw[m].mean(),
                        p.r_raw[m].mean() - u_raw,
                    ]
                    for m in METHODS
                ],
            ),
        ]
    lines += [
        '',
        'Summary: mean TIC r (WT − PCHIP: what the network adds beyond the same anchors)',
        *table(
            ['Pair', *METHODS, 'WT − PCHIP'],
            [
                [
                    pair_label(p.name_a, p.name_b),
                    *(p.r[m].mean() for m in METHODS),
                    f"{p.r['WarpTransformer'].mean() - p.r[PCHIP].mean():+.3f}",
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
    for x, y in (('WarpTransformer', PCHIP), ('COW-TIC', PCHIP)):
        lines += [
            f'  {x} vs {y}:',
            *table(['Pair', 'p', 'Holm p', 'median Δr'], holm_tests(results, x, y), indent='    '),
        ]
    return lines


def save(results: list[PairResult], seed: int, paths: Paths) -> None:
    paths.ensure_results()
    for p in results:
        np.savez(
            paths.per_pair / f'tic_r_pchip_pooled_{p.name_a}_{p.name_b}.npz',
            pchip_pooled=p.pchip.r,
            pchip_pooled_raw=p.pchip.r_raw,
            unaligned_raw=p.r_raw['Unaligned'],
            anchors=np.array(p.anchors).reshape(-1, 2),
            sil=p.pchip.sil,
            warp_mag=p.pchip.warp_mag,
            n_a=p.n_a,
            n_b=p.n_b,
        )
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
