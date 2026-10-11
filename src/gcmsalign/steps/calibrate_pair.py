"""Step 06 — WarpTransformer with anchor-supervised test-time calibration, on one batch pair.

Anchors (gcmsalign.anchors): peaks on the raw TIC of A[0..2] and B[0], spectral-cosine matching,
RANSAC, longest increasing subsequence, pooled over the three A samples (median B bin per A
bin), drift outlier filter (around the drift trend) and local-slope filter; the count after
every stage is reported.

The pretrained model is calibrated for --calib-steps Adam steps on the exact anchor loss (each
anchor's A and B positions must map to the same consensus bin) plus a smoothness penalty. TIC r
is never used for calibration, only for evaluation.

Reported: Unaligned, PCHIP through the same pooled anchors, and the WarpTransformer — TIC r
(normalised and raw, common window), study silhouette, warp magnitude — and the warp
magnitudes before calibration.

Output: results/warp_transformer_calib_<A>_<B>_seed<S>.txt,
        results/per_pair/tic_r_calib_<A>_<B>_seed<S>.npz
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .. import evaluation
from ..anchors import AnchorResult, extract_anchors, format_stages
from ..calibration import CalibrationLog, batch_tensors, calibrate, load_model, pick_device, predict
from ..config import LAMBDA_ANCHOR, LAMBDA_SMOOTH, LR_CALIB, N_CALIB_STEPS, Paths, pair_label
from ..data import Batch, load_batch
from ._common import (
    add_model_args,
    add_pair_args,
    batch_name_of,
    ckpt_or_default,
    parser,
    setup_logging,
    table,
    title,
    write_report,
)

log = logging.getLogger('gcmsalign')


@dataclass
class CalibResult:
    name_a: str
    name_b: str
    n_a: int
    n_b: int
    seed: int | None
    device: str
    ckpt: Path
    calib_steps: int
    anchors: AnchorResult
    unaligned: evaluation.PairEval
    pchip: evaluation.PairEval
    wt: evaluation.PairEval
    warps: np.ndarray
    mags: tuple[float, float, float]  # after calibration: all, A, B (min)
    mags_pre: tuple[float, float, float]  # before calibration
    log: CalibrationLog


def compute(
    a: Batch,
    b: Batch,
    ckpt: Path,
    device: torch.device,
    seed: int | None,
    calib_steps: int = N_CALIB_STEPS,
) -> CalibResult:
    anc = extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw)
    log.info(
        '%s: anchors %s\n  %s', pair_label(a.name, b.name), anc.anchors, format_stages(anc.stages)
    )
    model = load_model(ckpt, device)
    x, sid = batch_tensors(a.chromas, b.chromas, device)
    _, w_pre = predict(model, x, sid)
    clog = calibrate(
        model,
        x,
        sid,
        anc.anchors,
        seed=seed,
        steps=calib_steps,
        log_every=50,
        on_log=lambda s, rec, _: log.info(
            '  step %4d  loss %.4f  anchor %.4f', s, rec['loss'], rec['anchor']
        ),
    )
    aligned, warps = predict(model, x, sid)
    return CalibResult(
        a.name,
        b.name,
        len(a),
        len(b),
        seed,
        str(device),
        ckpt,
        calib_steps,
        anc,
        evaluation.unaligned(a, b),
        evaluation.pchip(a, b, anc.anchors),
        evaluation.warped_pair(a, b, warps, aligned),
        warps,
        evaluation.warp_mags(warps, len(a)),
        evaluation.warp_mags(w_pre, len(a)),
        clog,
    )


def format_report(res: CalibResult) -> list[str]:
    un = res.unaligned

    def row(name: str, e: evaluation.PairEval, mag: str | None) -> list[object]:
        return [
            name,
            e.mean,
            e.mean - un.mean,
            e.mean_raw,
            e.mean_raw - un.mean_raw,
            e.sil,
            mag,
            f'{100 * e.outside:.1f}%',
            e.mean_valid,
        ]

    return [
        title(__doc__),
        '',
        f'{pair_label(res.name_a, res.name_b)}: {res.name_a} ({res.n_a}) × {res.name_b} '
        f'({res.n_b}) = {res.n_a * res.n_b} pairs',
        f'Seed {res.seed}   device {res.device}   checkpoint {res.ckpt}',
        '',
        *table(
            [
                'Method',
                'TIC r',
                'Δ unal',
                'raw r',
                'Δ unal',
                'Study sil',
                'Warp mag',
                'outside',
                'r in range',
            ],
            [
                row('Unaligned', un, None),
                row('PCHIP (pooled anchors)', res.pchip, f'{res.pchip.warp_mag:.3f}m'),
                row('WarpTransformer + anchor calib', res.wt, f'{res.wt.warp_mag:.3f}m'),
            ],
        ),
        '',
        'TIC r: mean Pearson r over all A×B pairs on the common window; raw r: the same '
        'warps applied to the raw TIC.',
        'Warp mag: PCHIP = its one warp; WarpTransformer = batch-B samples.',
        "outside: share of evaluated (bin, sample) values read outside a run's acquired range; "
        'r in range: TIC r with those values masked out.',
        '',
        f'Anchors ({len(res.anchors.anchors)}, (A bin, B bin)): {res.anchors.anchors}',
        f'Anchor stages (match stages summed over the A samples): '
        f'{format_stages(res.anchors.stages)}',
        f'Removed by the drift outlier filter: {res.anchors.removed_by_outlier}',
        f'Removed by the slope filter: {res.anchors.removed_by_slope}',
        f'Calibration: {res.calib_steps} steps, lr {LR_CALIB}, λ_anchor {LAMBDA_ANCHOR}, '
        f'λ_smooth {LAMBDA_SMOOTH}, best loss {res.log.best_loss:.4f}, '
        f'{res.log.seconds:.1f} s',
        '',
        *table(
            ['WarpTransformer warp magnitude (min)', 'all', 'A', 'B'],
            [['before calibration', *res.mags_pre], ['after calibration', *res.mags]],
        ),
    ]


def save(res: CalibResult, paths: Paths) -> None:
    suffix = '' if res.seed is None else f'_seed{res.seed}'
    paths.ensure_results()
    np.savez(
        paths.per_pair / f'tic_r_calib_{res.name_a}_{res.name_b}{suffix}.npz',
        unaligned=res.unaligned.r,
        pchip_pooled=res.pchip.r,
        warptransformer=res.wt.r,
        unaligned_raw=res.unaligned.r_raw,
        pchip_pooled_raw=res.pchip.r_raw,
        warptransformer_raw=res.wt.r_raw,
        pchip_warp_mag=res.pchip.warp_mag,
        anchors=np.array(res.anchors.anchors).reshape(-1, 2),
        anchor_stages=np.array(list(res.anchors.stages.values())),
        anchor_stage_names=np.array(list(res.anchors.stages)),
        n_a=res.n_a,
        n_b=res.n_b,
        seed=-1 if res.seed is None else res.seed,
        sil=np.array([res.unaligned.sil, res.pchip.sil, res.wt.sil]),
        wt_warp_mag=np.array(res.mags),
        wt_warp_mag_pre=np.array(res.mags_pre),
        warps=res.warps,
        outside=np.array([res.unaligned.outside, res.pchip.outside, res.wt.outside]),
        unaligned_valid=res.unaligned.r_valid,
        pchip_pooled_valid=res.pchip.r_valid,
        warptransformer_valid=res.wt.r_valid,
        calib_seconds=res.log.seconds,
        calib_steps=res.calib_steps,
    )
    write_report(
        paths.results / f'warp_transformer_calib_{res.name_a}_{res.name_b}{suffix}.txt',
        format_report(res),
    )


def run(
    paths: Paths,
    name_a: str,
    name_b: str,
    seed: int | None,
    ckpt: Path | None = None,
    device: str = 'cpu',
    calib_steps: int = N_CALIB_STEPS,
) -> CalibResult:
    res = compute(
        load_batch(name_a, paths.fish_oil),
        load_batch(name_b, paths.fish_oil),
        ckpt_or_default(ckpt, paths),
        pick_device(device),
        seed,
        calib_steps,
    )
    save(res, paths)
    return res


def cli(argv: list[str] | None = None) -> None:
    ap = parser(__doc__)
    add_pair_args(ap)
    add_model_args(ap, seed_default=None)
    args = ap.parse_args(argv)
    setup_logging()
    run(
        Paths(args.root),
        batch_name_of(args.study_a),
        batch_name_of(args.study_b),
        args.seed,
        args.ckpt,
        args.device,
        args.calib_steps,
    )
