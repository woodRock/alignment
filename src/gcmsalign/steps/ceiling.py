"""Step 11 — an upper reference for between-batch TIC r: within-batch TIC r.

Samples of one batch were run together and share the retention-time axis, so their mean
pairwise TIC r, unaligned, is roughly what a perfect between-batch alignment could reach while
composition still differs between samples (tissues, species). Computed on the normalised and
the raw TIC, on the common acquisition window.

Per pair the ceiling is the mean of its two batches' within-batch values, beside the unaligned
between-batch r, so for any method: share of gap closed = (r − unaligned) / (ceiling − unaligned).
Within-batch r is unaligned, so residual within-batch drift makes the ceiling conservative.

The same on both TICs after Gaussian smoothing (config.SMOOTH_SIGMAS), the form in which every
method is also compared. Step 13 computes each method's share of the gap from the npz.

Output: results/within_batch_ceiling.txt, results/per_pair/within_batch_ceiling.npz
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import BATCHES, EVAL_PAIRS, EXCLUDED_PAIRS, SMOOTH_SIGMAS, Paths, pair_label
from ..data import Batch, load_batch
from ..metrics import pearson_rows, smooth_rows, within_batch_r
from ._common import parser, save_npz, setup_logging, table, title, write_report

# metric suffix (as evaluation.METRICS) → (raw TIC?, smoothing σ in bins; 0 = none)
SIGNALS = {
    '': (False, 0.0),
    **{f'_s{s:g}': (False, s) for s in SMOOTH_SIGMAS},
    '_raw': (True, 0.0),
    **{f'_raw_s{s:g}': (True, s) for s in SMOOTH_SIGMAS},
}


def signal(b: Batch, sfx: str) -> np.ndarray:
    raw, sigma = SIGNALS[sfx]
    return smooth_rows(b.tic_raw if raw else b.tic, sigma)


def label(sfx: str) -> str:
    raw, sigma = SIGNALS[sfx]
    return ('raw' if raw else 'normalised') + (f' σ{sigma:g}' if sigma else '')


@dataclass
class PairCeiling:
    a: str
    b: str
    evaluated: bool
    ceiling: dict[str, float]  # metric suffix → mean of the two within-batch values
    unaligned: dict[str, float]  # metric suffix → between-batch r, unaligned

    @property
    def label(self) -> str:
        return pair_label(self.a, self.b)


def compute(
    batches: dict[str, Batch],
) -> tuple[dict[str, dict[str, tuple[float, float, int]]], list[PairCeiling]]:
    """(batch → metric suffix → (mean, s.d., pairs) within the batch, per-pair ceilings)."""
    within = {
        k: {sfx: within_batch_r(signal(b, sfx)) for sfx in SIGNALS} for k, b in batches.items()
    }
    pairs = [
        PairCeiling(
            a,
            b,
            (a, b) in EVAL_PAIRS,
            {sfx: (within[a][sfx][0] + within[b][sfx][0]) / 2 for sfx in SIGNALS},
            {
                sfx: float(pearson_rows(signal(batches[a], sfx), signal(batches[b], sfx)).mean())
                for sfx in SIGNALS
            },
        )
        for a, b in EVAL_PAIRS + EXCLUDED_PAIRS
    ]
    return within, pairs


def format_report(
    within: dict[str, dict[str, tuple[float, float, int]]], pairs: list[PairCeiling]
) -> list[str]:
    lines = [
        title(__doc__),
        '',
        'Per batch: mean (s.d.) Pearson r over all pairs of distinct samples, unaligned.',
        *table(
            ['Batch', 'pairs', *(label(sfx) for sfx in SIGNALS)],
            [
                [k, w[''][2], *(f'{w[sfx][0]:.3f} ({w[sfx][1]:.3f})' for sfx in SIGNALS)]
                for k, w in within.items()
            ],
        ),
    ]
    for sfx in SIGNALS:
        lines += [
            '',
            f'Per pair, {label(sfx)} TIC: ceiling = mean of the two within-batch values; gap = '
            'ceiling − unaligned (the most an alignment could add).',
            *table(
                ['Pair', 'ceiling', 'unaligned', 'gap', ''],
                [
                    [
                        p.label,
                        p.ceiling[sfx],
                        p.unaligned[sfx],
                        p.ceiling[sfx] - p.unaligned[sfx],
                        '' if p.evaluated else '(excluded pair)',
                    ]
                    for p in pairs
                ],
            ),
        ]
    return lines


def save(pairs: list[PairCeiling], lines: list[str], paths: Paths) -> None:
    arrays: dict[str, object] = {}
    for p in pairs:
        for sfx in SIGNALS:
            arrays[f'{p.a}_{p.b}_ceiling{sfx}'] = p.ceiling[sfx]
            arrays[f'{p.a}_{p.b}_unaligned{sfx}'] = p.unaligned[sfx]
    paths.ensure_results()
    save_npz(paths.per_pair / 'within_batch_ceiling.npz', arrays)
    write_report(paths.results / 'within_batch_ceiling.txt', lines)


def run(paths: Paths) -> list[PairCeiling]:
    within, pairs = compute({name: load_batch(name, paths.fish_oil) for name in BATCHES})
    save(pairs, format_report(within, pairs), paths)
    return pairs


def cli(argv: list[str] | None = None) -> None:
    args = parser(__doc__).parse_args(argv)
    setup_logging()
    run(Paths(args.root))
