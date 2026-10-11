"""
11_within_batch_ceiling.py — An upper reference for between-batch TIC r: within-batch TIC r.

Samples of one batch were run together and share the retention-time axis, so their mean
pairwise TIC r, unaligned, is roughly what a perfect between-batch alignment could reach while
composition still differs between samples (tissues, species). Computed on the normalised TIC
(the paper's metric) and the raw TIC, on the common acquisition window (config.EVAL_BINS),
like every TIC r.

Per evaluated pair the reference is the mean of its two batches' within-batch values, next to
the unaligned between-batch TIC r, so for any method
    share of gap closed = (method r − unaligned r) / (ceiling − unaligned r).
No model, no randomness; seconds.

Output: results/within_batch_ceiling.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from config import BATCHES, EVAL_PAIRS, EXCLUDED_PAIRS, RESULTS_DIR, pair_label  # noqa: E402
from data import load_batch                                                    # noqa: E402
from metrics import pearson_rows, within_batch_r                               # noqa: E402


def main() -> list[str]:
    batches = {name: load_batch(name) for name in BATCHES}
    lines = [__doc__.strip().splitlines()[0], '',
             'Per batch: mean (s.d.) Pearson r over all pairs of distinct samples, unaligned.',
             f'  {"Batch":<16} {"n":>3} {"pairs":>6}   {"normalised TIC":>16}   {"raw TIC":>16}']
    within = {}
    for name, bt in batches.items():
        mn, sn, k = within_batch_r(bt.tic)
        mr, sr, _ = within_batch_r(bt.tic_raw)
        within[name] = (mn, mr)
        lines.append(f'  {name:<16} {len(bt):>3} {k:>6}   {mn:>7.3f} ({sn:.3f})   {mr:>7.3f} ({sr:.3f})')

    lines += ['', 'Per batch pair: ceiling = mean of the two within-batch values; unaligned = mean '
              'between-batch r over all A×B pairs.',
              f'  {"":<14} {"normalised TIC":^30}   {"raw TIC":^28}',
              f'  {"Pair":<14} {"ceiling":>10} {"unaligned":>10} {"gap":>8}   '
              f'{"ceiling":>8} {"unaligned":>10} {"gap":>8}']
    for a, b in EVAL_PAIRS + EXCLUDED_PAIRS:
        cn = (within[a][0] + within[b][0]) / 2
        cr = (within[a][1] + within[b][1]) / 2
        un = float(pearson_rows(batches[a].tic, batches[b].tic).mean())
        ur = float(pearson_rows(batches[a].tic_raw, batches[b].tic_raw).mean())
        tag = '' if (a, b) in EVAL_PAIRS else '  (excluded pair)'
        lines.append(f'  {pair_label(a, b):<14} {cn:>10.3f} {un:>10.3f} {cn - un:>8.3f}   '
                     f'{cr:>8.3f} {ur:>10.3f} {cr - ur:>8.3f}{tag}')
    lines += ['', 'gap = ceiling − unaligned: the most a between-batch alignment could add on this '
              'metric. Within-batch r is unaligned, so residual within-batch drift makes the '
              'ceiling conservative (low).']

    out = RESULTS_DIR / 'within_batch_ceiling.txt'
    out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines) + f'\n\n  results → {out}')
    return lines


if __name__ == '__main__':
    main()
