"""
24_pchip_pooled_anchors.py — Is the network needed, or are the pooled anchors enough?

Raw cosine PCHIP (script 16, table protocol) fits a spline through each A_i's own anchors
against B[0]. The WarpTransformer instead calibrates on anchors pooled over three sample
pairs (A[0..2] vs B[0]), median-aggregated and outlier-filtered (calib16.extract_anchors).
That difference in anchor handling, not the network, could explain part of its gain.

This baseline removes the confound: it fits the same monotone PCHIP construction used by
alignment.align_pair through exactly the WarpTransformer's pooled anchors, and applies that
one warp to every batch-A sample (A_i → B frame, compared with every unwarped B_j: the
table protocol). No network, no calibration, no randomness: CPU, deterministic, seconds.

Anchors are (r, q): r = RT bin in batch A, q = matching RT bin in B[0]. The warp maps
B time → A time, so the warped A_i at B time t is A_i(w(t)), with end points fixed at
0 and 45 min as in align_pair. Anchors that would break monotonicity are dropped the same
way align_pair drops them (sorted by r, q must increase).

Compares against the saved arrays of the latest run (scripts 16 and 17, seed 0):
Unaligned, Raw cosine PCHIP (A_i → B[0]), COW-TIC and the WarpTransformer.

Output: results/pchip_pooled_anchors.txt
        results/per_pair/tic_r_pchip_pooled_<A>_<B>.npz
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import wilcoxon

SCRIPTS = Path(__file__).parent
ROOT    = SCRIPTS.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


calib16 = _load('calib16', '16_align_testtime_calib.py')
from alignment import load_chromas, warp_chroma_2d   # noqa: E402  (src on path via calib16)

BIN_MIN  = calib16.BIN_MIN
RUN_MIN  = 45.0
EVAL     = [('batch_sep2015', 'batch_jul2016'), ('batch_sep2015', 'batch_apr2016'),
            ('batch_jan2016', 'batch_apr2016'), ('batch_jan2016', 'batch_jul2016')]
EXCLUDED = [('batch_sep2015', 'batch_jan2016'), ('batch_apr2016', 'batch_jul2016')]


def pooled_pchip_warp(anchors: list[tuple[int, int]]):
    """PCHIP B time → A time through the pooled anchors, as align_pair builds it."""
    anchors = sorted(anchors)                                  # by r (batch-A bin)
    mono = [anchors[0]] if anchors else []
    for r, q in anchors[1:]:
        if q > mono[-1][1]:
            mono.append((r, q))
    if len(mono) < 2:
        return None, mono
    ra = np.array([r for r, _ in mono], dtype=float) * BIN_MIN
    qa = np.array([q for _, q in mono], dtype=float) * BIN_MIN
    xf = np.concatenate([[0.0], qa, [RUN_MIN]])               # B (reference) time
    yf = np.concatenate([[0.0], ra, [RUN_MIN]])               # A (query) time
    return PchipInterpolator(xf, yf), mono


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
             f'Pooled anchors: calib16.extract_anchors (A[0..2] vs B[0]), the WarpTransformer\'s '
             f'calibration anchors.',
             f'Comparison arrays: tic_r_calib_*_seed{seed}.npz (script 16) and '
             f'tic_r_baselines_*.npz (script 17).', '']
    rows, tests = [], []

    for a, b in EVAL + EXCLUDED:
        ca = load_chromas(ROOT / 'data' / 'fish_oil' / a / 'chroma')
        cb = load_chromas(ROOT / 'data' / 'fish_oil' / b / 'chroma')
        n_a, n_b = len(ca), len(cb)
        calib = np.load(per_pair / f'tic_r_calib_{a}_{b}_seed{seed}.npz')
        base  = np.load(per_pair / f'tic_r_baselines_{a}_{b}.npz')
        if (int(calib['n_a']), int(calib['n_b'])) != (n_a, n_b):
            raise SystemExit(f'{a} × {b}: saved arrays are {int(calib["n_a"])}×{int(calib["n_b"])}, '
                             f'data are {n_a}×{n_b}; rerun scripts 16/17 first')

        anchors = calib16.extract_anchors(ca, cb)
        saved   = [tuple(x) for x in calib['anchors'].tolist()]
        if [tuple(x) for x in anchors] != saved:
            raise SystemExit(f'{a} × {b}: recomputed anchors {anchors} differ from saved {saved}')

        warp, used = pooled_pchip_warp(anchors)
        al_a   = [warp_chroma_2d(c, warp) for c in ca]
        r_pool = calib16.tic_r_per_pair(al_a, cb)
        sil    = calib16.study_silhouette(al_a, cb)
        mag    = calib16.pchip_warp_magnitude(warp)

        m = lambda x: float(np.mean(x))
        r = {'Unaligned': calib['unaligned'], 'Raw cosine PCHIP (A_i → B[0])': calib['pchip_b0'],
             'Pooled-anchor PCHIP': r_pool, 'COW-TIC': base['cow_tic'],
             'WarpTransformer': calib['warptransformer']}
        u = m(r['Unaligned'])
        lines.append('=' * 88)
        lines.append(f'{a} × {b}   n = {n_a}×{n_b} = {n_a * n_b}   pooled anchors {len(anchors)} '
                     f'(used in spline {len(used)}): {used}')
        lines.append(f'  {"Method":<32} {"TIC r":>7} {"Δ unal":>8} {"Study sil":>10} {"Warp mag":>10}')
        sils = {'Unaligned': calib['sil'][0], 'Raw cosine PCHIP (A_i → B[0])': calib['sil'][1],
                'Pooled-anchor PCHIP': sil, 'COW-TIC': base['sil'][1], 'WarpTransformer': calib['sil'][2]}
        mags = {'Raw cosine PCHIP (A_i → B[0])': m(calib['pchip_b0_warp_mag']),
                'Pooled-anchor PCHIP': mag, 'COW-TIC': float(base['warp_mag'][0]),
                'WarpTransformer': float(calib['wt_warp_mag'][2])}
        for k, v in r.items():
            mg = f'{mags[k]:.3f} min' if k in mags else '—'
            lines.append(f'  {k:<32} {m(v):>7.3f} {m(v) - u:>+8.3f} {float(sils[k]):>10.3f} {mg:>10}')
        rows.append((a, b, len(used), u, m(r['Raw cosine PCHIP (A_i → B[0])']), m(r_pool),
                     m(r['COW-TIC']), m(r['WarpTransformer'])))
        if (a, b) in EVAL:
            tests.append((a, b, n_a, n_b, r))

        np.savez(per_pair / f'tic_r_pchip_pooled_{a}_{b}.npz', pchip_pooled=r_pool,
                 anchors=np.array(anchors), anchors_used=np.array(used), sil=sil,
                 warp_mag=mag, n_a=n_a, n_b=n_b)

    lines += ['', '=' * 88, 'Summary (mean TIC r; WT − pooled is what the network adds beyond the '
              'same anchors)',
              f'  {"Pair":<28} {"anch":>4} {"Unal":>6} {"PCHIP":>6} {"Pooled":>7} {"COW":>6} '
              f'{"WT":>6} {"WT−Pooled":>10} {"Pooled−PCHIP":>13}']
    for a, b, k, u, raw, pool, cow, wt in rows:
        tag = '' if (a, b) in EVAL else '  (excluded pair)'
        lines.append(f'  {a + " × " + b:<28} {k:>4} {u:>6.3f} {raw:>6.3f} {pool:>7.3f} {cow:>6.3f} '
                     f'{wt:>6.3f} {wt - pool:>+10.3f} {pool - raw:>+13.3f}{tag}')

    lines += ['', 'Paired two-sided Wilcoxon, per A sample (r averaged over B_j), Holm-adjusted '
              'across the 4 evaluated pairs per comparison:']
    comps = [('WarpTransformer', 'Pooled-anchor PCHIP'),
             ('Pooled-anchor PCHIP', 'Raw cosine PCHIP (A_i → B[0])'),
             ('Pooled-anchor PCHIP', 'COW-TIC')]
    for x, y in comps:
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
