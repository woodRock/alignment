"""
05_literature_baselines.py — COW-TIC and icoshift on one batch pair.

Every batch-A sample is aligned onto every batch-B sample (TIC only; implementations in
src/baselines.py). TIC r is Pearson r between the warped A TIC and the B TIC on the common
acquisition window (config.EVAL_BINS). The same warp, optimised on the normalised TIC, is
applied to the raw TIC for the raw-TIC rows. Study silhouette: every A sample aligned to B[0]
(full chromatogram), B unwarped.

Output: results/literature_baselines_<A>_<B>.txt, results/per_pair/tic_r_baselines_<A>_<B>.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from baselines import METHODS, COW_N_SEGS, COW_SLACK, ICO_MAX_SHIFT, ICO_N_INTERVALS  # noqa: E402
from config import BIN_MIN, PER_PAIR_DIR, RESULTS_DIR, pair_label                    # noqa: E402
from data import load_batch                                                          # noqa: E402
from metrics import pearson, pearson_rows, study_silhouette, paired_wilcoxon  # noqa: E402
from warps import warp_rows, warp_signal                                             # noqa: E402


def evaluate(method, a, b) -> dict:
    """Per-pair TIC r (normalised and raw), warp magnitudes, and study silhouette for one method."""
    tic_a, tic_b = a.tic, b.tic
    r, r_raw, mags = [], [], []
    for qa, qa_raw in zip(tic_a, a.tic_raw):
        for rb, rb_raw in zip(tic_b, b.tic_raw):
            al = method(qa, rb)
            r.append(pearson(al.warped, rb))
            r_raw.append(pearson(warp_signal(qa_raw, al.src), rb_raw))
            mags.append(al.warp_mag)
    src_to_b0 = np.stack([method(qa, tic_b[0]).src for qa in tic_a])
    return {'r': np.array(r), 'r_raw': np.array(r_raw), 'warp_mag': float(np.mean(mags)),
            'sil': study_silhouette(warp_rows(a.chromas, src_to_b0), b.chromas)}


def main(name_a: str, name_b: str) -> list[str]:
    a, b = load_batch(name_a), load_batch(name_b)
    n_a, n_b = len(a), len(b)
    res = {'Unaligned': {'r': pearson_rows(a.tic, b.tic), 'r_raw': pearson_rows(a.tic_raw, b.tic_raw),
                         'warp_mag': 0.0, 'sil': study_silhouette(a.chromas, b.chromas)}}
    for name, method in METHODS.items():
        print(f'Evaluating {name} …', flush=True)
        res[name] = evaluate(method, a, b)

    u, u_raw = res['Unaligned']['r'].mean(), res['Unaligned']['r_raw'].mean()
    lines = [__doc__.strip().splitlines()[0], '',
             f'{pair_label(name_a, name_b)}: {name_a} ({n_a}) × {name_b} ({n_b}) = {n_a * n_b} pairs',
             f'COW-TIC: {COW_N_SEGS} segments, slack ±{COW_SLACK} bins (±{COW_SLACK * BIN_MIN:.1f} '
             f'min); icoshift: {ICO_N_INTERVALS} intervals, max shift ±{ICO_MAX_SHIFT} bins', '',
             f'  {"Method":<12} {"TIC r":>7} {"Δ unal":>8} {"TIC r raw":>10} {"Δ unal":>8} '
             f'{"Study sil":>10} {"Warp mag":>9}']
    for k, v in res.items():
        lines.append(f'  {k:<12} {v["r"].mean():>7.3f} {v["r"].mean() - u:>+8.3f} '
                     f'{v["r_raw"].mean():>10.3f} {v["r_raw"].mean() - u_raw:>+8.3f} '
                     f'{v["sil"]:>10.3f} {v["warp_mag"]:>8.3f}m')
    lines += ['', 'Paired two-sided Wilcoxon over sample pairs (not independent; descriptive):']
    names = list(res)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            _, p = paired_wilcoxon(res[y]['r'], res[x]['r'])
            lines.append(f'  {y:<10} vs {x:<10} p = {p:.2e}')

    PER_PAIR_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PER_PAIR_DIR / f'tic_r_baselines_{name_a}_{name_b}.npz',
             unaligned=res['Unaligned']['r'], cow_tic=res['COW-TIC']['r'],
             icoshift=res['icoshift']['r'], unaligned_raw=res['Unaligned']['r_raw'],
             cow_tic_raw=res['COW-TIC']['r_raw'], icoshift_raw=res['icoshift']['r_raw'],
             n_a=n_a, n_b=n_b,
             sil=np.array([res[k]['sil'] for k in ('Unaligned', 'COW-TIC', 'icoshift')]),
             warp_mag=np.array([res[k]['warp_mag'] for k in ('COW-TIC', 'icoshift')]))
    out = RESULTS_DIR / f'literature_baselines_{name_a}_{name_b}.txt'
    out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines) + f'\n\n  results → {out}')
    return lines


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study-a', default='fish_oil/batch_sep2015')
    ap.add_argument('--study-b', default='fish_oil/batch_jul2016')
    args = ap.parse_args()
    main(Path(args.study_a).name, Path(args.study_b).name)
