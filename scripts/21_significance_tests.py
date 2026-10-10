"""
21_significance_tests.py — Paired Wilcoxon signed-rank tests between all methods.

Reads the per-sample-pair TIC r arrays written by
  17_literature_baselines.py  → results/per_pair/tic_r_baselines_{A}_{B}.npz
                                (Unaligned, COW-TIC, icoshift)
  16_align_testtime_calib.py  → results/per_pair/tic_r_calib_{A}_{B}_seed{S}.npz
                                (WarpTransformer)
  24_pchip_pooled_anchors.py  → results/per_pair/tic_r_pchip_pooled_{A}_{B}.npz
                                (PCHIP through the WarpTransformer's pooled anchors)
All arrays are indexed by the same (A_i, B_j) pair, i-major (index i * n_B + j), so the tests
are paired. The Unaligned arrays from both scripts are checked to be identical.

Two levels of test, both two-sided:
  per-pair    n = n_A × n_B (sample pairs). These observations are not independent: each A_i
              appears in n_B pairs and each B_j in n_A pairs, so the p-values are optimistic.
  per-sample  r averaged over B_j for each A_i, n = n_A. Conservative; recommended for the paper.
Holm correction is applied across the 10 comparisons of the 5 methods within each batch pair
and level. The family size is stated in the output.

The WarpTransformer is tested at --seed (default 0). The other seeds are summarised as the
number of seeds at which each WarpTransformer comparison stays significant.

Output: results/significance_tests.txt
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RESULTS_DIR = Path(__file__).parent.parent / 'results'
PER_PAIR    = RESULTS_DIR / 'per_pair'

PAIRS = [('batch_sep2015', 'batch_jul2016'), ('batch_sep2015', 'batch_apr2016'),
         ('batch_jan2016', 'batch_apr2016'), ('batch_jan2016', 'batch_jul2016')]
PCHIP   = 'PCHIP (pooled anchors)'
METHODS = ['Unaligned', 'icoshift', PCHIP, 'COW-TIC', 'WarpTransformer']
ALPHA   = 0.05


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m, adj, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        adj[idx] = running
    return adj


def load_methods(a: str, b: str, seed: int) -> tuple[dict[str, np.ndarray], int, int]:
    base  = np.load(PER_PAIR / f'tic_r_baselines_{a}_{b}.npz')
    calib = np.load(PER_PAIR / f'tic_r_calib_{a}_{b}_seed{seed}.npz')
    if not np.allclose(base['unaligned'], calib['unaligned'], atol=1e-6):
        raise ValueError(f"{a}×{b}: Unaligned arrays from scripts 16 and 17 differ — pairing is broken")
    data = {
        'Unaligned':        base['unaligned'],
        'icoshift':         base['icoshift'],
        'COW-TIC':          base['cow_tic'],
        'WarpTransformer':  calib['warptransformer'],
    }
    n_a, n_b = int(base['n_a']), int(base['n_b'])
    if not pooled_file(a, b).exists():
        raise SystemExit(f"{a}×{b}: {pooled_file(a, b).name} missing — run script 24 first")
    pool = np.load(pooled_file(a, b))
    if (int(pool['n_a']), int(pool['n_b'])) != (n_a, n_b):
        raise ValueError(f"{a}×{b}: PCHIP arrays are {int(pool['n_a'])}×{int(pool['n_b'])}, "
                         f"others {n_a}×{n_b} — rerun script 24")
    data[PCHIP] = pool['pchip_pooled']
    return data, n_a, n_b


def pooled_file(a: str, b: str) -> Path:
    return PER_PAIR / f'tic_r_pchip_pooled_{a}_{b}.npz'


def per_sample(x: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
    return x.reshape(n_a, n_b).mean(axis=1)


def run_tests(data: dict[str, np.ndarray], methods: list[str]) -> list[dict]:
    rows = []
    for m1, m2 in combinations(methods, 2):
        d = data[m2] - data[m1]
        stat, p = wilcoxon(data[m2], data[m1], alternative='two-sided')
        rows.append(dict(m1=m1, m2=m2, stat=float(stat), p=float(p), n=len(d),
                         n_nonzero=int(np.sum(d != 0)), med=float(np.median(d))))
    for row, p_adj in zip(rows, holm([r['p'] for r in rows])):
        row['p_holm'] = p_adj
    return rows


def fmt(rows: list[dict], title: str) -> list[str]:
    out = [title,
           f"  {'Comparison (B vs A)':<42} {'n':>4} {'W':>9} {'p':>11} {'p (Holm)':>11} {'median Δr':>10}"]
    for r in rows:
        sig = '*' if r['p_holm'] < ALPHA else ' '
        out.append(f"  {r['m2'] + ' vs ' + r['m1']:<42} {r['n']:>4} {r['stat']:>9.1f} "
                   f"{r['p']:>11.3e} {r['p_holm']:>11.3e}{sig} {r['med']:>+10.4f}")
    return out


def main(seed: int, seeds: list[int]) -> None:
    methods = METHODS
    family  = len(methods) * (len(methods) - 1) // 2
    lines = [__doc__.strip().split('\n')[0], '',
             f"Methods ({len(methods)}): {', '.join(methods)}.",
             f"Holm family: all {family} pairwise method comparisons, corrected separately within "
             f"each batch pair and test level.",
             f"WarpTransformer seed tested: {seed}.  '*' = significant at α={ALPHA} after Holm.",
             "W = Wilcoxon signed-rank statistic (smaller of the two rank sums); median Δr = median of B − A.",
             ""]
    summary = []
    for a, b in PAIRS:
        data, n_a, n_b = load_methods(a, b, seed)
        label = f"{a.replace('batch_', '')} × {b.replace('batch_', '')}"
        means = '  '.join(f"{m}={data[m].mean():.3f}" for m in methods)
        lines += [f"{'=' * 96}", f"{label}   n_A={n_a}  n_B={n_b}", f"  mean TIC r: {means}", ""]
        pp = run_tests(data, methods)
        ps = run_tests({m: per_sample(v, n_a, n_b) for m, v in data.items()}, methods)
        lines += fmt(pp, f"  Per sample pair (n = {n_a}×{n_b} = {n_a * n_b}; not independent; "
                         f"Holm over {family}):")
        lines += ['']
        lines += fmt(ps, f"  Per A sample (r averaged over B_j, n = {n_a}; recommended; "
                         f"Holm over {family}):")
        lines += ['']

        # Robustness of the WarpTransformer comparisons across calibration seeds
        robust = {m: 0 for m in methods if m != 'WarpTransformer'}
        available = []
        for s in seeds:
            f = PER_PAIR / f'tic_r_calib_{a}_{b}_seed{s}.npz'
            if not f.exists():
                continue
            available.append(s)
            d_s, _, _ = load_methods(a, b, s)
            rows = run_tests({m: per_sample(v, n_a, n_b) for m, v in d_s.items()}, methods)
            for r in rows:
                if 'WarpTransformer' in (r['m1'], r['m2']):
                    other = r['m1'] if r['m2'] == 'WarpTransformer' else r['m2']
                    robust[other] += int(r['p_holm'] < ALPHA)
        lines.append(f"  Seeds available: {available}. Per-A-sample WarpTransformer comparisons "
                     f"significant after Holm, out of {len(available)} seeds:")
        lines.append('    ' + '  '.join(f"vs {m}: {k}/{len(available)}" for m, k in robust.items()))
        lines.append('')
        summary.append((label, ps))

    lines += [f"{'=' * 96}", f"Summary (per A sample, Holm-adjusted p over {family}, WarpTransformer vs each method):"]
    for label, ps in summary:
        cells = [f"{(r['m1'] if r['m2'] == 'WarpTransformer' else r['m2'])}: {r['p_holm']:.2e}"
                 for r in ps if 'WarpTransformer' in (r['m1'], r['m2'])]
        lines.append(f"  {label:<20} " + '   '.join(cells))

    report = '\n'.join(lines)
    print(report)
    out = RESULTS_DIR / 'significance_tests.txt'
    out.write_text(report + '\n')
    print(f"\n  results → {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    main(args.seed, args.seeds)
