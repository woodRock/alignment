"""Step 11 — an upper reference for between-batch TIC r: within-batch TIC r.

Samples of one batch were run together and share the retention-time axis, so their mean
pairwise TIC r, unaligned, is roughly what a perfect between-batch alignment could reach while
composition still differs between samples (tissues, species). Computed on the normalised and
the raw TIC, on the common acquisition window.

Per pair the ceiling is the mean of its two batches' within-batch values, beside the unaligned
between-batch r, so for any method: share of gap closed = (r − unaligned) / (ceiling − unaligned).
Within-batch r is unaligned, so residual within-batch drift makes the ceiling conservative.

Output: results/within_batch_ceiling.txt
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import BATCHES, EVAL_PAIRS, EXCLUDED_PAIRS, Paths, pair_label
from ..data import Batch, load_batch
from ..metrics import pearson_rows, within_batch_r
from ._common import parser, setup_logging, table, title, write_report


@dataclass
class BatchWithin:
    n: int
    norm: tuple[float, float, int]  # mean, s.d., pairs
    raw: tuple[float, float, int]


@dataclass
class PairCeiling:
    label: str
    evaluated: bool
    ceiling: float
    unaligned: float
    ceiling_raw: float
    unaligned_raw: float


def compute(batches: dict[str, Batch]) -> tuple[dict[str, BatchWithin], list[PairCeiling]]:
    within = {
        k: BatchWithin(len(b), within_batch_r(b.tic), within_batch_r(b.tic_raw))
        for k, b in batches.items()
    }
    pairs = []
    for a, b in EVAL_PAIRS + EXCLUDED_PAIRS:
        pairs.append(
            PairCeiling(
                pair_label(a, b),
                (a, b) in EVAL_PAIRS,
                (within[a].norm[0] + within[b].norm[0]) / 2,
                float(pearson_rows(batches[a].tic, batches[b].tic).mean()),
                (within[a].raw[0] + within[b].raw[0]) / 2,
                float(pearson_rows(batches[a].tic_raw, batches[b].tic_raw).mean()),
            )
        )
    return within, pairs


def format_report(within: dict[str, BatchWithin], pairs: list[PairCeiling]) -> list[str]:
    return [
        title(__doc__),
        '',
        'Per batch: mean (s.d.) Pearson r over all pairs of distinct samples, unaligned.',
        *table(
            ['Batch', 'n', 'pairs', 'normalised TIC', 'raw TIC'],
            [
                [
                    k,
                    w.n,
                    w.norm[2],
                    f'{w.norm[0]:.3f} ({w.norm[1]:.3f})',
                    f'{w.raw[0]:.3f} ({w.raw[1]:.3f})',
                ]
                for k, w in within.items()
            ],
        ),
        '',
        'Per pair: ceiling = mean of the two within-batch values; gap = ceiling − '
        'unaligned (the most an alignment could add).',
        *table(
            ['Pair', 'ceiling', 'unaligned', 'gap', 'raw ceiling', 'raw unaligned', 'raw gap', ''],
            [
                [
                    p.label,
                    p.ceiling,
                    p.unaligned,
                    p.ceiling - p.unaligned,
                    p.ceiling_raw,
                    p.unaligned_raw,
                    p.ceiling_raw - p.unaligned_raw,
                    '' if p.evaluated else '(excluded pair)',
                ]
                for p in pairs
            ],
        ),
    ]


def run(paths: Paths) -> list[PairCeiling]:
    within, pairs = compute({name: load_batch(name, paths.fish_oil) for name in BATCHES})
    write_report(paths.results / 'within_batch_ceiling.txt', format_report(within, pairs))
    return pairs


def cli(argv: list[str] | None = None) -> None:
    args = parser(__doc__).parse_args(argv)
    setup_logging()
    run(Paths(args.root))
