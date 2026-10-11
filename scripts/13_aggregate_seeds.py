"""
13_aggregate_seeds.py — One summary over calibration seeds for the paper's tables.

Reads, for the four evaluated batch pairs,
  tic_r_baselines_<A>_<B>.npz          script 05   Unaligned, icoshift, COW-TIC (deterministic)
  tic_r_pchip_pooled_<A>_<B>.npz       script 07   PCHIP through the pooled anchors (deterministic)
  tic_r_calib_<A>_<B>_seed{s}.npz      script 06   WarpTransformer, one file per seed
  tic_r_ablate_<A>_<B>_seed{s}.npz     script 08   pretrained vs random-init, one file per seed
and reports, per pair:
  - WarpTransformer TIC r (normalised and raw): mean ± s.d. over seeds, min and max; warp
    magnitude (batch B, batch A, all samples) mean ± s.d.
  - the deterministic rows, normalised and raw, with a check that the arrays every seed's file
    carries (unaligned, pooled PCHIP, anchors) are identical across seeds and match scripts 05/07
  - WarpTransformer − COW-TIC and WarpTransformer − PCHIP: mean ± s.d. over seeds and the number
    of seeds at which each is positive
  - per batch-A sample (r averaged over B_j): how many A samples the WarpTransformer beats
    COW-TIC for, at each seed
  - ablation: TIC r and warp magnitude (all / B) of both arms, mean ± s.d.; the pretraining gain
    per seed, then mean ± s.d.; and the share of the improvement over unaligned that random init
    recovers ((random − unaligned) / (pretrained − unaligned))

s.d. is the sample standard deviation (ddof = 1) over seeds.

Output: results/seed_summary.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from config import EVAL_PAIRS as PAIRS, PER_PAIR_DIR as PER_PAIR, RESULTS_DIR as RESULTS  # noqa: E402


def ms(x) -> str:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1) if len(x) > 1 else float('nan')
    return f'{x.mean():.3f} ± {sd:.3f}'


def ms4(x) -> str:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1) if len(x) > 1 else float('nan')
    return f'{x.mean():+.4f} ± {sd:.4f}'


def per_a(r: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
    return np.asarray(r).reshape(n_a, n_b).mean(axis=1)


def same(x, y) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    return x.shape == y.shape and np.allclose(x, y, atol=1e-6)


def ablate_device(abl) -> str:
    """Device recorded by script 08 ('?' for files written before it was recorded)."""
    return str(abl['device']) if 'device' in abl.files else '?'


def main(seeds: list[int]) -> None:
    lines = [__doc__.strip().splitlines()[0], '',
             f'Seeds requested: {seeds}. s.d. = sample s.d. over the seeds found (ddof = 1).', '']
    problems = []
    overview = []

    for a, b in PAIRS:
        label = f"{a.replace('batch_', '')} × {b.replace('batch_', '')}"
        base  = np.load(PER_PAIR / f'tic_r_baselines_{a}_{b}.npz')
        pool  = np.load(PER_PAIR / f'tic_r_pchip_pooled_{a}_{b}.npz')
        n_a, n_b = int(base['n_a']), int(base['n_b'])

        cal = {s: np.load(f) for s in seeds
               if (f := PER_PAIR / f'tic_r_calib_{a}_{b}_seed{s}.npz').exists()}
        abl = {s: np.load(f) for s in seeds
               if (f := PER_PAIR / f'tic_r_ablate_{a}_{b}_seed{s}.npz').exists()}
        cs, as_ = sorted(cal), sorted(abl)
        lines += ['=' * 96, f'{label}   n = {n_a}×{n_b} = {n_a * n_b} sample pairs   '
                  f'calibration seeds {cs}   ablation seeds {as_}', '']
        if not cs:
            problems.append(f'{label}: no tic_r_calib files')
            continue

        # ── Consistency checks on the deterministic arrays ────────────────────
        s0 = cal[cs[0]]
        checks = {
            'unaligned (06) identical across seeds':
                all(same(cal[s]['unaligned'], s0['unaligned']) for s in cs),
            'unaligned (06) = unaligned (05)': same(s0['unaligned'], base['unaligned']),
            'unaligned raw (06) identical across seeds':
                all(same(cal[s]['unaligned_raw'], s0['unaligned_raw']) for s in cs),
            'unaligned raw (06) = (05) = (07)': same(s0['unaligned_raw'], base['unaligned_raw'])
                and same(s0['unaligned_raw'], pool['unaligned_raw']),
            'pooled PCHIP (06) identical across seeds and = (07)':
                all(same(cal[s]['pchip_pooled'], pool['pchip_pooled']) for s in cs),
            'pooled PCHIP raw (06) identical across seeds and = (07)':
                all(same(cal[s]['pchip_pooled_raw'], pool['pchip_pooled_raw']) for s in cs),
            'anchors identical across seeds (06, 08) and = (07)':
                all(same(cal[s]['anchors'], pool['anchors']) for s in cs)
                and all(same(abl[s]['anchors'], pool['anchors']) for s in as_),
        }
        for k, ok in checks.items():
            lines.append(f'  check: {k:<58} {"OK" if ok else "FAILED"}')
            if not ok:
                problems.append(f'{label}: {k} FAILED')
        lines.append('')

        # ── Main table ────────────────────────────────────────────────────────
        wt     = np.array([cal[s]['warptransformer'].mean() for s in cs])
        wt_raw = np.array([cal[s]['warptransformer_raw'].mean() for s in cs])
        mags   = np.array([cal[s]['wt_warp_mag'] for s in cs])        # [seeds, (all, A, B)]
        det = [('Unaligned', base['unaligned'], base['unaligned_raw']),
               ('icoshift', base['icoshift'], base['icoshift_raw']),
               ('PCHIP (pooled anchors)', pool['pchip_pooled'], pool['pchip_pooled_raw']),
               ('COW-TIC', base['cow_tic'], base['cow_tic_raw'])]
        u, u_raw = float(base['unaligned'].mean()), float(base['unaligned_raw'].mean())
        lines += [f'  {"Method":<26} {"TIC r":>15} {"Δ unal":>8} {"min":>7} {"max":>7}   '
                  f'{"TIC r (raw)":>15} {"Δ unal":>8} {"min":>7} {"max":>7}']
        for name, r, rr in det:
            lines.append(f'  {name:<26} {r.mean():>15.3f} {r.mean() - u:>+8.3f} {"":>7} {"":>7}   '
                         f'{rr.mean():>15.3f} {rr.mean() - u_raw:>+8.3f}')
        lines.append(f'  {"WarpTransformer":<26} {ms(wt):>15} {wt.mean() - u:>+8.3f} '
                     f'{wt.min():>7.3f} {wt.max():>7.3f}   {ms(wt_raw):>15} '
                     f'{wt_raw.mean() - u_raw:>+8.3f} {wt_raw.min():>7.3f} {wt_raw.max():>7.3f}')
        lines += ['', '  WarpTransformer per seed: ' +
                  '  '.join(f's{s} {x:.3f}/{y:.3f}' for s, x, y in zip(cs, wt, wt_raw)) +
                  '   (normalised/raw)',
                  f'  WarpTransformer warp magnitude (min): batch B {ms(mags[:, 2])}   '
                  f'batch A {ms(mags[:, 1])}   all {ms(mags[:, 0])}']
        if all('wt_warp_mag_pre' in cal[s].files for s in cs):
            pre_m = np.array([cal[s]['wt_warp_mag_pre'] for s in cs])
            lines.append(f'  … before calibration (pretrained model): batch B {ms(pre_m[:, 2])}   '
                         f'batch A {ms(pre_m[:, 1])}   all {ms(pre_m[:, 0])}')
        lines.append('')

        # ── Margins ───────────────────────────────────────────────────────────
        cow, pc = float(base['cow_tic'].mean()), float(pool['pchip_pooled'].mean())
        cow_raw, pc_raw = float(base['cow_tic_raw'].mean()), float(pool['pchip_pooled_raw'].mean())
        lines.append(f'  {"Margin":<26} {"normalised":>17} {"seeds > 0":>10}   '
                     f'{"raw":>17} {"seeds > 0":>10}')
        for name, ref, ref_raw in [('WT − COW-TIC', cow, cow_raw), ('WT − PCHIP', pc, pc_raw)]:
            d, dr = wt - ref, wt_raw - ref_raw
            lines.append(f'  {name:<26} {ms4(d):>17} {f"{(d > 0).sum()}/{len(d)}":>10}   '
                         f'{ms4(dr):>17} {f"{(dr > 0).sum()}/{len(dr)}":>10}')
        lines.append('')

        # ── Per batch-A sample ───────────────────────────────────────────────
        cow_a, cow_a_raw = per_a(base['cow_tic'], n_a, n_b), per_a(base['cow_tic_raw'], n_a, n_b)
        wins = [(per_a(cal[s]['warptransformer'], n_a, n_b) > cow_a).sum() for s in cs]
        wins_raw = [(per_a(cal[s]['warptransformer_raw'], n_a, n_b) > cow_a_raw).sum() for s in cs]
        lines += [f'  Batch-A samples (of {n_a}) where WT beats COW-TIC (r averaged over B_j):',
                  '    normalised: ' + '  '.join(f's{s} {w}' for s, w in zip(cs, wins)) +
                  f'   mean {np.mean(wins):.1f}',
                  '    raw       : ' + '  '.join(f's{s} {w}' for s, w in zip(cs, wins_raw)) +
                  f'   mean {np.mean(wins_raw):.1f}', '']

        # ── Pretraining ablation ─────────────────────────────────────────────
        if as_:
            devs = sorted({ablate_device(abl[s]) for s in as_})
            pre  = np.array([abl[s]['pretrained'].mean() for s in as_])
            rnd  = np.array([abl[s]['random_init'].mean() for s in as_])
            pre_r = np.array([abl[s]['pretrained_raw'].mean() for s in as_])
            rnd_r = np.array([abl[s]['random_init_raw'].mean() for s in as_])
            wm   = np.array([abl[s]['warp_mag'] for s in as_])       # [seeds, (pre, rand)]
            wmb  = np.array([abl[s]['warp_mag_b'] for s in as_])
            gain, gain_r = pre - rnd, pre_r - rnd_r
            share, share_r = (rnd - u) / (pre - u), (rnd_r - u_raw) / (pre_r - u_raw)
            lines += [f'  Pretraining ablation (script 08, device {", ".join(devs)}):',
                      f'    {"Arm":<14} {"TIC r":>15} {"TIC r (raw)":>15} {"warp all (min)":>16} '
                      f'{"warp B (min)":>16}',
                      f'    {"pretrained":<14} {ms(pre):>15} {ms(pre_r):>15} {ms(wm[:, 0]):>16} '
                      f'{ms(wmb[:, 0]):>16}',
                      f'    {"random init":<14} {ms(rnd):>15} {ms(rnd_r):>15} {ms(wm[:, 1]):>16} '
                      f'{ms(wmb[:, 1]):>16}',
                      '    gain per seed (normalised): ' +
                      '  '.join(f's{s} {g:+.4f}' for s, g in zip(as_, gain)) +
                      f'   mean {ms4(gain)}   seeds > 0: {(gain > 0).sum()}/{len(gain)}',
                      '    gain per seed (raw)       : ' +
                      '  '.join(f's{s} {g:+.4f}' for s, g in zip(as_, gain_r)) +
                      f'   mean {ms4(gain_r)}   seeds > 0: {(gain_r > 0).sum()}/{len(gain_r)}',
                      f'    share of the improvement random init recovers: normalised {ms(share)}   '
                      f'raw {ms(share_r)}']
            if all('warp_mag_pre' in abl[s].files for s in as_):
                wp = np.array([abl[s]['warp_mag_pre'] for s in as_])   # [seeds, arm, (all, A, B)]
                for k, arm in enumerate(('pretrained', 'random init')):
                    lines.append(f'    starting warp, {arm:<11} (min): all {ms(wp[:, k, 0])}   '
                                 f'A {ms(wp[:, k, 1])}   B {ms(wp[:, k, 2])}')
            lines.append('')
            overview.append((label, wt, cow, pc, gain))
        else:
            overview.append((label, wt, cow, pc, None))

    lines += ['=' * 96, 'Overview (normalised TIC r; mean ± s.d. over seeds)',
              f'  {"Pair":<20} {"WarpTransformer":>16} {"WT − COW-TIC":>17} {"WT − PCHIP":>17} '
              f'{"pretraining gain":>18}']
    for label, wt, cow, pc, gain in overview:
        g = ms4(gain) if gain is not None else '—'
        lines.append(f'  {label:<20} {ms(wt):>16} {ms4(wt - cow):>17} {ms4(wt - pc):>17} {g:>18}')
    lines += ['', 'Checks: ' + ('all passed' if not problems else '; '.join(problems))]

    report = '\n'.join(lines)
    print(report)
    out = RESULTS / 'seed_summary.txt'
    out.write_text(report + '\n')
    print(f'\n  results → {out}')
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    main(ap.parse_args().seeds)
