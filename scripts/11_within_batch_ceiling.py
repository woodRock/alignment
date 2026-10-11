"""
11_within_batch_ceiling.py — An upper reference for between-batch TIC r: within-batch TIC r.

Samples of the same batch were run together, so they share the retention-time axis. Their mean
pairwise TIC r, unaligned, is what a perfect between-batch alignment could at best reach while
composition still differs between samples (tissues, species). It is computed on the normalised
TIC (sum over m/z of the sqrt + L2 chromatogram, the paper's metric) and on the raw TIC
(tic_raw: summed raw intensity of each binned apex scan).

For each evaluated batch pair the reference is the mean of its two batches' within-batch values,
listed next to the unaligned between-batch TIC r, so

    share of gap closed = (method r − unaligned r) / (ceiling − unaligned r)

can be read off for any method. No model, no randomness; seconds.

Output: results/within_batch_ceiling.txt
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from alignment import load_chromas, load_tic_raw, pearson_rows   # noqa: E402

BATCHES  = ['batch_sep2015', 'batch_jan2016', 'batch_apr2016', 'batch_jul2016']
EVAL     = [('batch_sep2015', 'batch_jul2016'), ('batch_sep2015', 'batch_apr2016'),
            ('batch_jan2016', 'batch_apr2016'), ('batch_jan2016', 'batch_jul2016')]
EXCLUDED = [('batch_sep2015', 'batch_jan2016'), ('batch_apr2016', 'batch_jul2016')]


def within(x: np.ndarray) -> tuple[float, float, int]:
    """Mean and s.d. of Pearson r over all pairs of distinct rows, and the number of pairs."""
    r = pearson_rows(x, x).reshape(len(x), len(x))
    v = np.array([r[i, j] for i, j in combinations(range(len(x)), 2)])
    return float(v.mean()), float(v.std()), len(v)


def main() -> None:
    tics = {}
    for b in BATCHES:
        d = ROOT / 'data' / 'fish_oil' / b / 'chroma'
        tics[b] = {'norm': np.stack([c.sum(axis=1) for c in load_chromas(d)]),
                   'raw':  load_tic_raw(d)}

    lines = [__doc__.strip().splitlines()[0], '',
             'Per batch: mean (s.d.) Pearson r over all pairs of distinct samples, unaligned.',
             f'  {"Batch":<16} {"n":>3} {"pairs":>6}   {"normalised TIC":>16}   {"raw TIC":>16}']
    w = {}
    for b in BATCHES:
        n = len(tics[b]['norm'])
        mn, sn, k = within(tics[b]['norm'])
        mr, sr, _ = within(tics[b]['raw'])
        w[b] = (mn, mr)
        lines.append(f'  {b:<16} {n:>3} {k:>6}   {mn:>7.3f} ({sn:.3f})   {mr:>7.3f} ({sr:.3f})')

    lines += ['', 'Per batch pair: ceiling = mean of the two within-batch values; unaligned = mean '
              'between-batch r over all A×B pairs.',
              f'  {"":<30} {"normalised TIC":^30}   {"raw TIC":^28}',
              f'  {"Pair":<30} {"ceiling":>10} {"unaligned":>10} {"gap":>8}   '
              f'{"ceiling":>8} {"unaligned":>10} {"gap":>8}']
    for a, b in EVAL + EXCLUDED:
        cn, cr = (w[a][0] + w[b][0]) / 2, (w[a][1] + w[b][1]) / 2
        un = float(pearson_rows(tics[a]['norm'], tics[b]['norm']).mean())
        ur = float(pearson_rows(tics[a]['raw'], tics[b]['raw']).mean())
        tag = '' if (a, b) in EVAL else '  (excluded pair)'
        lines.append(f'  {a + " × " + b:<30} {cn:>10.3f} {un:>10.3f} {cn - un:>8.3f}   '
                     f'{cr:>8.3f} {ur:>10.3f} {cr - ur:>8.3f}{tag}')
    lines += ['', 'gap = ceiling − unaligned: the improvement a between-batch alignment could at best '
              'deliver on this metric.',
              'Within-batch r is unaligned: residual within-batch drift, if any, makes it a '
              'conservative (low) ceiling.']

    out = ROOT / 'results' / 'within_batch_ceiling.txt'
    out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n  results → {out}')


if __name__ == '__main__':
    main()
