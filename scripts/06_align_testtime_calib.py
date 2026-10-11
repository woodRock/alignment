"""
06_align_testtime_calib.py — WarpTransformer with anchor-supervised test-time calibration.

Anchors (src/anchors.py): peaks on the raw TIC of A[0..2] and B[0], spectral-cosine matching,
RANSAC, longest increasing subsequence, pooled over the three A samples (median B bin per A
bin), drift outlier filter and local-slope filter. The anchor count after every stage is
reported.

The pretrained model (checkpoints/warp_transformer_v2.pt) is calibrated for N_CALIB_STEPS Adam
steps on the exact anchor loss (each anchor's A and B positions must map to the same consensus
bin) plus a smoothness penalty. TIC r is never used for calibration, only for evaluation.

Reported per pair: Unaligned, PCHIP through the same pooled anchors, and the WarpTransformer —
TIC r (normalised and raw, on the common acquisition window), study silhouette, warp magnitude
— and the pre-calibration warp magnitudes.

Output: results/warp_transformer_calib_<A>_<B>_seed<S>.txt,
        results/per_pair/tic_r_calib_<A>_<B>_seed<S>.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import evaluation                                                               # noqa: E402
from anchors import extract_anchors, format_stages                              # noqa: E402
from calibration import (batch_tensors, calibrate, load_model, pick_device,    # noqa: E402
                         predict)
from config import (DEFAULT_CKPT, LAMBDA_ANCHOR, LAMBDA_SMOOTH, LR_CALIB,      # noqa: E402
                    N_CALIB_STEPS, PER_PAIR_DIR, RESULTS_DIR, pair_label)
from data import load_batch                                                     # noqa: E402


def main(name_a: str, name_b: str, seed: int | None, ckpt: Path, device_name: str) -> list[str]:
    device = pick_device(device_name)
    a, b = load_batch(name_a), load_batch(name_b)
    n_a, n_b = len(a), len(b)
    anc = extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw)
    print(f'{pair_label(name_a, name_b)}: {n_a}×{n_b}; anchors {anc.anchors}\n  '
          f'{format_stages(anc.stages)}', flush=True)

    model = load_model(ckpt, device)
    x, sid = batch_tensors(a.chromas, b.chromas, device)
    _, w_pre = predict(model, x, sid)
    log = calibrate(model, x, sid, anc.anchors, seed=seed, log_every=50,
                    on_log=lambda s, rec, _: print(f"  step {s:>4}  loss {rec['loss']:.4f}  "
                                                   f"anchor {rec['anchor']:.4f}", flush=True))
    aligned, warps = predict(model, x, sid)

    un = evaluation.unaligned(a, b)
    pc = evaluation.pchip(a, b, anc.anchors)
    wt = evaluation.warped_pair(a, b, warps, aligned)
    mags, pre = evaluation.warp_mags(warps, n_a), evaluation.warp_mags(w_pre, n_a)

    def row(name, e, mag):
        return (f'  {name:<32} {e.mean:>7.3f} {e.mean - un.mean:>+8.3f} {e.mean_raw:>9.3f} '
                f'{e.mean_raw - un.mean_raw:>+8.3f} {e.sil:>10.3f} {mag:>9}')
    lines = [__doc__.strip().splitlines()[0], '',
             f'{pair_label(name_a, name_b)}: {name_a} ({n_a}) × {name_b} ({n_b}) = {n_a * n_b} pairs',
             f'Seed {seed}   device {device}   checkpoint {ckpt}', '',
             f'  {"Method":<32} {"TIC r":>7} {"Δ unal":>8} {"raw r":>9} {"Δ unal":>8} '
             f'{"Study sil":>10} {"Warp mag":>9}',
             row('Unaligned', un, '—'),
             row('PCHIP (pooled anchors)', pc, f'{pc.warp_mag:.3f}m'),
             row('WarpTransformer + anchor calib', wt, f'{wt.warp_mag:.3f}m'),
             '', 'TIC r: mean Pearson r over all A×B pairs on the common window; raw r: the same '
             'warps applied to the raw TIC.',
             'Warp mag: PCHIP = its one warp; WarpTransformer = batch-B samples.', '',
             f'Anchors ({len(anc.anchors)}, (A bin, B bin)): {anc.anchors}',
             f'Anchor stages (match stages summed over the A samples): {format_stages(anc.stages)}',
             f'Removed by the slope filter: {anc.removed_by_slope}',
             f'Calibration: {N_CALIB_STEPS} steps, lr {LR_CALIB}, λ_anchor {LAMBDA_ANCHOR}, '
             f'λ_smooth {LAMBDA_SMOOTH}, best loss {log.best_loss:.4f}, {log.seconds:.1f} s', '',
             'WarpTransformer warp magnitude (min)   all      A        B',
             f'  before calibration                  {pre[0]:.3f}    {pre[1]:.3f}    {pre[2]:.3f}',
             f'  after calibration                   {mags[0]:.3f}    {mags[1]:.3f}    {mags[2]:.3f}']

    suffix = '' if seed is None else f'_seed{seed}'
    PER_PAIR_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(PER_PAIR_DIR / f'tic_r_calib_{name_a}_{name_b}{suffix}.npz',
             unaligned=un.r, pchip_pooled=pc.r, warptransformer=wt.r,
             unaligned_raw=un.r_raw, pchip_pooled_raw=pc.r_raw, warptransformer_raw=wt.r_raw,
             pchip_warp_mag=pc.warp_mag, anchors=np.array(anc.anchors).reshape(-1, 2),
             anchor_stages=np.array(list(anc.stages.values())),
             anchor_stage_names=np.array(list(anc.stages)),
             n_a=n_a, n_b=n_b, seed=-1 if seed is None else seed,
             sil=np.array([un.sil, pc.sil, wt.sil]), wt_warp_mag=np.array(mags),
             wt_warp_mag_pre=np.array(pre), warps=warps, calib_seconds=log.seconds)
    out = RESULTS_DIR / f'warp_transformer_calib_{name_a}_{name_b}{suffix}.txt'
    out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines) + f'\n\n  results → {out}')
    return lines


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study-a', default='fish_oil/batch_sep2015')
    ap.add_argument('--study-b', default='fish_oil/batch_jul2016')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    ap.add_argument('--device', default='cpu',
                    help='cpu (default; bit-reproducible under a seed) | mps | cuda | auto')
    args = ap.parse_args()
    main(Path(args.study_a).name, Path(args.study_b).name, args.seed, args.ckpt, args.device)
