"""
08_ablate_pretrain.py — Does pretraining matter, or does calibration do all the work?

Runs script 06's calibration twice on the same anchors and seed: (a) from the pretrained
checkpoint, (b) from a random-init model (initialisation seeded too). The zero-initialised
warp and start heads make the random-init model start exactly at the identity warp; the
pretrained model starts from whatever warp its trained heads predict. Both starting warp
magnitudes are measured and reported, so the comparison's starting points are on record.

Output: results/ablate_pretrain_<A>_<B>_seed<S>.txt, results/per_pair/tic_r_ablate_<A>_<B>_seed<S>.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import evaluation                                                               # noqa: E402
from anchors import extract_anchors                                             # noqa: E402
from calibration import batch_tensors, calibrate, load_model, pick_device, predict  # noqa: E402
from config import DEFAULT_CKPT, PER_PAIR_DIR, RESULTS_DIR, pair_label          # noqa: E402
from data import load_batch                                                     # noqa: E402


def run_arm(ckpt, a, b, anchors, device, seed):
    """Calibrate one arm. Returns (PairEval, warp mags before, warp mags after)."""
    model = load_model(ckpt, device, seed=seed)
    x, sid = batch_tensors(a.chromas, b.chromas, device)
    _, w_pre = predict(model, x, sid)
    calibrate(model, x, sid, anchors, seed=seed)
    aligned, warps = predict(model, x, sid)
    n_a = len(a)
    return (evaluation.warped_pair(a, b, warps, aligned),
            evaluation.warp_mags(w_pre, n_a), evaluation.warp_mags(warps, n_a))


def main(name_a: str, name_b: str, seed: int | None, ckpt: Path, device_name: str) -> list[str]:
    device = pick_device(device_name)
    a, b = load_batch(name_a), load_batch(name_b)
    anchors = extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw).anchors
    un = evaluation.unaligned(a, b)
    arms = {'pretrained': run_arm(ckpt, a, b, anchors, device, seed),
            'random init': run_arm(None, a, b, anchors, device, seed)}

    lines = [__doc__.strip().splitlines()[0], '',
             f'{pair_label(name_a, name_b)}   seed {seed}   device {device}   checkpoint {ckpt}',
             f'Anchors ({len(anchors)}): {anchors}', '',
             f'  {"Arm":<14} {"TIC r":>7} {"Δ unal":>8} {"raw r":>8} {"Δ unal":>8}   '
             f'{"warp before (all/A/B, min)":>27}   {"warp after (all/A/B, min)":>26}',
             f'  {"Unaligned":<14} {un.mean:>7.4f} {"—":>8} {un.mean_raw:>8.4f} {"—":>8}']
    for k, (e, pre, post) in arms.items():
        lines.append(f'  {k:<14} {e.mean:>7.4f} {e.mean - un.mean:>+8.4f} {e.mean_raw:>8.4f} '
                     f'{e.mean_raw - un.mean_raw:>+8.4f}   '
                     f'{pre[0]:>7.3f} {pre[1]:>7.3f} {pre[2]:>7.3f}     '
                     f'{post[0]:>7.3f} {post[1]:>7.3f} {post[2]:>7.3f}')
    (pe, ppre, ppost), (re, rpre, rpost) = arms['pretrained'], arms['random init']
    lines += ['', f'Pretraining advantage: {pe.mean - re.mean:+.4f} TIC r '
              f'({pe.mean_raw - re.mean_raw:+.4f} raw)']

    suffix = '' if seed is None else f'_seed{seed}'
    out = RESULTS_DIR / f'ablate_pretrain_{name_a}_{name_b}{suffix}.txt'
    out.write_text('\n'.join(lines) + '\n')
    PER_PAIR_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PER_PAIR_DIR / f'tic_r_ablate_{name_a}_{name_b}{suffix}.npz',
             pretrained=pe.r, random_init=re.r, pretrained_raw=pe.r_raw, random_init_raw=re.r_raw,
             warp_mag=np.array([ppost[0], rpost[0]]), warp_mag_b=np.array([ppost[2], rpost[2]]),
             warp_mag_pre=np.array([ppre, rpre]),          # [arm, (all, A, B)]
             anchors=np.array(anchors).reshape(-1, 2), seed=-1 if seed is None else seed,
             device=str(device))
    print('\n'.join(lines) + f'\n\n  results → {out}')
    return lines


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study-a', default='fish_oil/batch_sep2015')
    ap.add_argument('--study-b', default='fish_oil/batch_jul2016')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    ap.add_argument('--device', default='cpu', help='cpu (default, reproducible) | mps | cuda | auto')
    args = ap.parse_args()
    main(Path(args.study_a).name, Path(args.study_b).name, args.seed, args.ckpt, args.device)
