"""
07_pchip_pooled_anchors.py — The paper's PCHIP baseline: one spline through the pooled anchors.

PCHIP is fitted through exactly the WarpTransformer's calibration anchors: pooled over three
sample pairs (A[0..2] vs B[0]), median-aggregated and outlier-filtered (calib06.extract_anchors).
The same monotone spline construction as alignment.align_pair is used (alignment.
pooled_anchor_pchip), and the one warp is applied to every batch-A sample (A_i → B frame,
compared with every unwarped B_j). So PCHIP and the WarpTransformer receive identical anchors
and differ only in how they use them. No network, no calibration, no randomness: CPU,
deterministic, seconds.

Script 06 also reports this baseline in its per-pair table; this script writes the per-pair
arrays that scripts 12 (significance tests) and 14 (Figs. 4–5) read, and compares against the
saved arrays of the latest run (scripts 05 and 06, --seed): Unaligned, COW-TIC, WarpTransformer.

Output: results/pchip_pooled_anchors.txt
        results/per_pair/tic_r_pchip_pooled_<A>_<B>.npz
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SCRIPTS = Path(__file__).parent
ROOT    = SCRIPTS.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


calib06 = _load('calib06', '06_align_testtime_calib.py')
from alignment import (load_chromas, load_tic_raw, warp_chroma_2d, warp_tic,   # noqa: E402
                       pooled_anchor_pchip, pearson_rows)

pooled_pchip_warp = pooled_anchor_pchip      # name used by script 09

EVAL     = [('batch_sep2015', 'batch_jul2016'), ('batch_sep2015', 'batch_apr2016'),
            ('batch_jan2016', 'batch_apr2016'), ('batch_jan2016', 'batch_jul2016')]
EXCLUDED = [('batch_sep2015', 'batch_jan2016'), ('batch_apr2016', 'batch_jul2016')]
PCHIP    = 'PCHIP (pooled anchors)'


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = running
    return adj.tolist()


def per_a(r: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
    return r.reshape(n_a, n_b).mean(axis=1)


def main(seed: int) -> None:
    per_pair = ROOT / 'results' / 'per_pair'
    lines = [__doc__.strip().splitlines()[0], '',
             'Anchors: calib06.extract_anchors (A[0..2] vs B[0]), the WarpTransformer\'s calibration '
             'anchors.',
             f'Comparison arrays: tic_r_calib_*_seed{seed}.npz (script 06) and tic_r_baselines_*.npz '
             f'(script 05).', '']
    rows, rows_raw, tests = [], [], []

    for a, b in EVAL + EXCLUDED:
        ca = load_chromas(ROOT / 'data' / 'fish_oil' / a / 'chroma')
        cb = load_chromas(ROOT / 'data' / 'fish_oil' / b / 'chroma')
        n_a, n_b = len(ca), len(cb)
        calib = np.load(per_pair / f'tic_r_calib_{a}_{b}_seed{seed}.npz')
        base  = np.load(per_pair / f'tic_r_baselines_{a}_{b}.npz')
        if (int(calib['n_a']), int(calib['n_b'])) != (n_a, n_b):
            raise SystemExit(f'{a} × {b}: saved arrays are {int(calib["n_a"])}×{int(calib["n_b"])}, '
                             f'data are {n_a}×{n_b}; rerun scripts 05/06 first')

        anchors = calib06.extract_anchors(ca, cb)
        saved   = [tuple(x) for x in calib['anchors'].tolist()]
        if [tuple(x) for x in anchors] != saved:
            raise SystemExit(f'{a} × {b}: recomputed anchors {anchors} differ from saved {saved}')

        warp, used = pooled_anchor_pchip(anchors)
        al_a   = [warp_chroma_2d(c, warp) for c in ca]
        r_pool = calib06.tic_r_per_pair(al_a, cb)
        sil    = calib06.study_silhouette(al_a, cb)
        mag    = calib06.pchip_warp_magnitude(warp)
        raw_a  = load_tic_raw(ROOT / 'data' / 'fish_oil' / a / 'chroma')
        raw_b  = load_tic_raw(ROOT / 'data' / 'fish_oil' / b / 'chroma')
        r_raw  = {'Unaligned': pearson_rows(raw_a, raw_b),
                  PCHIP: pearson_rows(np.stack([warp_tic(t, warp) for t in raw_a]), raw_b)}
        if 'cow_tic_raw' in base.files:
            r_raw['COW-TIC'] = base['cow_tic_raw']
        if 'warptransformer_raw' in calib.files:
            r_raw['WarpTransformer'] = calib['warptransformer_raw']

        m = lambda x: float(np.mean(x))
        r = {'Unaligned': calib['unaligned'], PCHIP: r_pool, 'COW-TIC': base['cow_tic'],
             'WarpTransformer': calib['warptransformer']}
        sils = {'Unaligned': calib['sil'][0], PCHIP: sil, 'COW-TIC': base['sil'][1],
                'WarpTransformer': calib['sil'][2]}
        mags = {PCHIP: mag, 'COW-TIC': float(base['warp_mag'][0]),
                'WarpTransformer': float(calib['wt_warp_mag'][2])}
        u = m(r['Unaligned'])
        lines.append('=' * 88)
        lines.append(f'{a} × {b}   n = {n_a}×{n_b} = {n_a * n_b}   pooled anchors {len(anchors)} '
                     f'(used in spline {len(used)}): {used}')
        lines.append(f'  {"Method":<32} {"TIC r":>7} {"Δ unal":>8} {"Study sil":>10} {"Warp mag":>10}')
        for k, v in r.items():
            mg = f'{mags[k]:.3f} min' if k in mags else '—'
            lines.append(f'  {k:<32} {m(v):>7.3f} {m(v) - u:>+8.3f} {float(sils[k]):>10.3f} {mg:>10}')
        u_raw = m(r_raw['Unaligned'])
        lines.append(f'  {"TIC r (raw)":<32} {"TIC r":>7} {"Δ unal":>8}')
        for k, v in r_raw.items():
            lines.append(f'  {k:<32} {m(v):>7.3f} {m(v) - u_raw:>+8.3f}')
        missing = [k for k in r if k not in r_raw]
        if missing:
            lines.append(f'  (no raw arrays saved yet for: {", ".join(missing)})')
        rows.append((a, b, len(used), u, m(r_pool), m(r['COW-TIC']), m(r['WarpTransformer'])))
        rows_raw.append((a, b, *(m(r_raw[k]) if k in r_raw else np.nan for k in r)))
        if (a, b) in EVAL:
            tests.append((a, b, n_a, n_b, r))

        np.savez(per_pair / f'tic_r_pchip_pooled_{a}_{b}.npz', pchip_pooled=r_pool,
                 pchip_pooled_raw=r_raw[PCHIP], unaligned_raw=r_raw['Unaligned'],
                 anchors=np.array(anchors), anchors_used=np.array(used), sil=sil,
                 warp_mag=mag, n_a=n_a, n_b=n_b)

    lines += ['', '=' * 88, 'Summary (mean TIC r; WT − PCHIP is what the network adds beyond the '
              'same anchors)',
              f'  {"Pair":<28} {"anch":>4} {"Unal":>6} {"PCHIP":>6} {"COW":>6} {"WT":>6} '
              f'{"WT−PCHIP":>9}']
    for a, b, k, u, pool, cow, wt in rows:
        tag = '' if (a, b) in EVAL else '  (excluded pair)'
        lines.append(f'  {a + " × " + b:<28} {k:>4} {u:>6.3f} {pool:>6.3f} {cow:>6.3f} {wt:>6.3f} '
                     f'{wt - pool:>+9.3f}{tag}')

    lines += ['', 'Summary, raw TIC (same warps applied to the raw TIC; nan = not saved yet)',
              f'  {"Pair":<28} {"Unal":>6} {"PCHIP":>6} {"COW":>6} {"WT":>6} {"WT−PCHIP":>9}']
    for a, b, u, pool, cow, wt in rows_raw:
        tag = '' if (a, b) in EVAL else '  (excluded pair)'
        lines.append(f'  {a + " × " + b:<28} {u:>6.3f} {pool:>6.3f} {cow:>6.3f} {wt:>6.3f} '
                     f'{wt - pool:>+9.3f}{tag}')

    lines += ['', 'Paired two-sided Wilcoxon, per A sample (r averaged over B_j). Holm family: the 4 '
              'evaluated batch pairs, corrected separately for each comparison below (family size 4).',
              'The paper\'s significance table (all 10 method pairs within a batch pair) is '
              'significance_tests.txt (script 12).']
    for x, y in [('WarpTransformer', PCHIP), ('COW-TIC', PCHIP)]:
        res = []
        for a, b, n_a, n_b, r in tests:
            dx, dy = per_a(r[x], n_a, n_b), per_a(r[y], n_a, n_b)
            res.append((a, b, n_a, wilcoxon(dx, dy).pvalue, float(np.median(dx - dy))))
        adj = holm([p for *_, p, _ in res])
        lines.append(f'  {x} vs {y}:')
        for (a, b, n, p, md), pa in zip(res, adj):
            lines.append(f'    {a + " × " + b:<28} n={n:<3} p={p:.2e}  Holm p={pa:.2e}  '
                         f'median Δr {md:+.4f}')

    out = ROOT / 'results' / 'pchip_pooled_anchors.txt'
    out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n  results → {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0,
                    help='which saved script-16 run (tic_r_calib_*_seed{N}.npz) to compare against')
    main(ap.parse_args().seed)
