"""
07_pchip_pooled_anchors.py — The PCHIP baseline: one spline through the pooled anchors.

PCHIP gets exactly the WarpTransformer's calibration anchors (src/anchors.py, checked against
the saved script-06 run) and the warp of src/warps.pchip_warp: PCHIP between the outer anchors
and slope 1 beyond them, so neither end of the run is pinned (the WarpTransformer's warps are
not pinned either). The one warp is applied to every batch-A sample (A → B frame); B is
unwarped. Deterministic, CPU, seconds.

Writes the per-pair arrays that scripts 12 (significance) and 14 (figures) read, and compares
with the saved arrays of scripts 05 and 06 (--seed).

Output: results/pchip_pooled_anchors.txt, results/per_pair/tic_r_pchip_pooled_<A>_<B>.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import evaluation                                                               # noqa: E402
from anchors import extract_anchors                                             # noqa: E402
from config import EVAL_PAIRS, EXCLUDED_PAIRS, PER_PAIR_DIR, RESULTS_DIR, pair_label  # noqa: E402
from data import load_batch                                                     # noqa: E402
from metrics import holm, per_a_mean, paired_wilcoxon  # noqa: E402

PCHIP = 'PCHIP (pooled anchors)'


def main(seed: int) -> list[str]:
    lines = [__doc__.strip().splitlines()[0], '',
             f'Comparison arrays: tic_r_calib_*_seed{seed}.npz (06), tic_r_baselines_*.npz (05).', '']
    summary, tests = [], []
    for name_a, name_b in EVAL_PAIRS + EXCLUDED_PAIRS:
        a, b = load_batch(name_a), load_batch(name_b)
        calib = np.load(PER_PAIR_DIR / f'tic_r_calib_{name_a}_{name_b}_seed{seed}.npz')
        base  = np.load(PER_PAIR_DIR / f'tic_r_baselines_{name_a}_{name_b}.npz')
        if (int(calib['n_a']), int(calib['n_b'])) != (len(a), len(b)):
            raise SystemExit(f'{name_a} × {name_b}: saved arrays do not match the data; rerun 05/06')
        anchors = extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw).anchors
        saved = [tuple(map(int, x)) for x in calib['anchors'].reshape(-1, 2)]
        if anchors != saved:
            raise SystemExit(f'{name_a} × {name_b}: anchors {anchors} differ from 06\'s {saved}')

        pc = evaluation.pchip(a, b, anchors)
        r = {'Unaligned': (calib['unaligned'], calib['unaligned_raw']),
             PCHIP: (pc.r, pc.r_raw),
             'COW-TIC': (base['cow_tic'], base['cow_tic_raw']),
             'WarpTransformer': (calib['warptransformer'], calib['warptransformer_raw'])}
        u, u_raw = r['Unaligned'][0].mean(), r['Unaligned'][1].mean()
        lines += ['=' * 80, f'{pair_label(name_a, name_b)}   n = {len(a)}×{len(b)}   '
                  f'anchors {len(anchors)}: {anchors}   PCHIP warp {pc.warp_mag:.3f} min',
                  f'  {"Method":<26} {"TIC r":>7} {"Δ unal":>8} {"raw r":>8} {"Δ unal":>8}']
        for k, (x, xr) in r.items():
            lines.append(f'  {k:<26} {x.mean():>7.3f} {x.mean() - u:>+8.3f} '
                         f'{xr.mean():>8.3f} {xr.mean() - u_raw:>+8.3f}')
        summary.append((name_a, name_b, {k: (x.mean(), xr.mean()) for k, (x, xr) in r.items()}))
        if (name_a, name_b) in EVAL_PAIRS:
            tests.append((name_a, name_b, len(a), len(b), {k: x for k, (x, _) in r.items()}))
        np.savez(PER_PAIR_DIR / f'tic_r_pchip_pooled_{name_a}_{name_b}.npz',
                 pchip_pooled=pc.r, pchip_pooled_raw=pc.r_raw, unaligned_raw=r['Unaligned'][1],
                 anchors=np.array(anchors).reshape(-1, 2), sil=pc.sil, warp_mag=pc.warp_mag,
                 n_a=len(a), n_b=len(b))

    lines += ['', '=' * 80, 'Summary: mean TIC r normalised / raw (WT − PCHIP: what the network adds '
              'beyond the same anchors)']
    for name_a, name_b, s in summary:
        tag = '' if (name_a, name_b) in EVAL_PAIRS else '  (excluded pair)'
        cells = '  '.join(f'{k.split(" ")[0]} {v[0]:.3f}/{v[1]:.3f}' for k, v in s.items())
        d = s['WarpTransformer'][0] - s[PCHIP][0]
        lines.append(f'  {pair_label(name_a, name_b):<12} {cells}  WT−PCHIP {d:+.3f}{tag}')

    lines += ['', 'Paired two-sided Wilcoxon per A sample (r averaged over B_j); Holm over the 4 '
              'evaluated pairs, separately per comparison. The full 10-comparison table is '
              'significance_tests.txt (script 12).']
    for x, y in [('WarpTransformer', PCHIP), ('COW-TIC', PCHIP)]:
        res = [(na, nb, paired_wilcoxon(per_a_mean(r[x], n_a, n_b), per_a_mean(r[y], n_a, n_b))[1],
                float(np.median(per_a_mean(r[x], n_a, n_b) - per_a_mean(r[y], n_a, n_b))))
               for na, nb, n_a, n_b, r in tests]
        lines.append(f'  {x} vs {y}:')
        for (na, nb, p, md), pa in zip(res, holm([t[2] for t in res])):
            lines.append(f'    {pair_label(na, nb):<12} p {p:.2e}  Holm p {pa:.2e}  median Δr {md:+.4f}')

    out = RESULTS_DIR / 'pchip_pooled_anchors.txt'
    out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines) + f'\n\n  results → {out}')
    return lines


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0,
                    help='which saved script-06 run (tic_r_calib_*_seed<N>.npz) to compare with')
    main(ap.parse_args().seed)
