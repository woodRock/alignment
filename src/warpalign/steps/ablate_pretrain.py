"""Step 08 — does pretraining matter, or does calibration do all the work?

Runs step 06's calibration twice on the same anchors and seed: from the pretrained checkpoint,
and from a random-init model (initialisation seeded too). The zero-initialised warp and start
heads make the random-init model start exactly at the identity warp; the pretrained model
starts from whatever warp its trained heads predict. Both starting warp magnitudes are
measured and reported.

Output: results/ablate_pretrain_<A>_<B>_seed<S>.txt,
        results/per_pair/tic_r_ablate_<A>_<B>_seed<S>.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .. import evaluation
from .._compat import Batch, extract_anchors, load_batch
from ..calibration import batch_tensors, calibrate, load_model, pick_device, predict
from ..config import N_CALIB_STEPS, Paths, pair_label
from ._common import (
    add_model_args,
    add_pair_args,
    batch_name_of,
    ckpt_or_default,
    parser,
    save_npz,
    setup_logging,
    table,
    title,
    write_report,
)

ARMS = ('pretrained', 'random init')


@dataclass
class Arm:
    eval: evaluation.PairEval
    mags_pre: tuple[float, float, float]  # all, A, B (min) before calibration
    mags: tuple[float, float, float]  # after


@dataclass
class AblationResult:
    name_a: str
    name_b: str
    seed: int | None
    device: str
    ckpt: Path
    anchors: list[tuple[int, int]]
    unaligned: evaluation.PairEval
    arms: dict[str, Arm]

    @property
    def gain(self) -> float:
        return self.arms['pretrained'].eval.mean - self.arms['random init'].eval.mean

    @property
    def gain_raw(self) -> float:
        return self.arms['pretrained'].eval.mean_raw - self.arms['random init'].eval.mean_raw

    def gain_of(self, sfx: str) -> float:
        """Pretraining gain on one metric (evaluation.METRICS suffix)."""
        return self.arms['pretrained'].eval.m(sfx) - self.arms['random init'].eval.m(sfx)


def run_arm(
    ckpt: Path | None,
    a: Batch,
    b: Batch,
    anchors: list[tuple[int, int]],
    device: torch.device,
    seed: int | None,
    calib_steps: int,
) -> Arm:
    model = load_model(ckpt, device, seed=seed)
    x, sid = batch_tensors(a.chromas, b.chromas, device)
    _, w_pre = predict(model, x, sid)
    calibrate(model, x, sid, anchors, seed=seed, steps=calib_steps)
    aligned, warps = predict(model, x, sid)
    return Arm(
        evaluation.warped_pair(a, b, warps, aligned),
        evaluation.warp_mags(w_pre, len(a)),
        evaluation.warp_mags(warps, len(a)),
    )


def compute(
    a: Batch,
    b: Batch,
    ckpt: Path,
    device: torch.device,
    seed: int | None,
    calib_steps: int = N_CALIB_STEPS,
) -> AblationResult:
    anchors = extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw).anchors
    arms = {
        'pretrained': run_arm(ckpt, a, b, anchors, device, seed, calib_steps),
        'random init': run_arm(None, a, b, anchors, device, seed, calib_steps),
    }
    return AblationResult(
        a.name, b.name, seed, str(device), ckpt, anchors, evaluation.unaligned(a, b), arms
    )


def format_report(res: AblationResult) -> list[str]:
    un = res.unaligned
    rows: list[list[object]] = [
        ['Unaligned', un.mean, None, un.mean_raw, None, None, None, None, None, None, None]
    ]
    for k, arm in res.arms.items():
        e = arm.eval
        rows.append(
            [
                k,
                e.mean,
                e.mean - un.mean,
                e.mean_raw,
                e.mean_raw - un.mean_raw,
                *arm.mags_pre,
                *arm.mags,
            ]
        )
    return [
        title(__doc__),
        '',
        f'{pair_label(res.name_a, res.name_b)}   seed {res.seed}   device {res.device}   '
        f'checkpoint {res.ckpt}',
        f'Anchors ({len(res.anchors)}): {res.anchors}',
        '',
        *table(
            [
                'Arm',
                'TIC r',
                'Δ unal',
                'raw r',
                'Δ unal',
                'pre all',
                'pre A',
                'pre B',
                'post all',
                'post A',
                'post B',
            ],
            rows,
            floatfmt='.4f',
        ),
        '',
        'pre / post: warp magnitude (min) before and after calibration.',
        f'Pretraining advantage: {res.gain:+.4f} TIC r ({res.gain_raw:+.4f} raw)',
        '',
        'Every TIC metric (σ: both batches smoothed with the same Gaussian, in bins):',
        *table(
            ['Arm', *evaluation.METRICS],
            [
                [k, *(e.m(sfx) for sfx in evaluation.METRICS.values())]
                for k, e in (('Unaligned', un), *((k, a.eval) for k, a in res.arms.items()))
            ],
            floatfmt='.4f',
        ),
        '  gain: '
        + '  '.join(f'{k} {res.gain_of(sfx):+.4f}' for k, sfx in evaluation.METRICS.items()),
    ]


def save(res: AblationResult, paths: Paths) -> None:
    suffix = '' if res.seed is None else f'_seed{res.seed}'
    pre, rnd = res.arms['pretrained'], res.arms['random init']
    paths.ensure_results()
    save_npz(
        paths.per_pair / f'tic_r_ablate_{res.name_a}_{res.name_b}{suffix}.npz',
        dict(
            **pre.eval.arrays('pretrained'),
            **rnd.eval.arrays('random_init'),
            warp_mag=np.array([pre.mags[0], rnd.mags[0]]),
            warp_mag_b=np.array([pre.mags[2], rnd.mags[2]]),
            warp_mag_pre=np.array([pre.mags_pre, rnd.mags_pre]),  # [arm, (all, A, B)]
            anchors=np.array(res.anchors).reshape(-1, 2),
            seed=-1 if res.seed is None else res.seed,
            device=res.device,
        ),
    )
    write_report(
        paths.results / f'ablate_pretrain_{res.name_a}_{res.name_b}{suffix}.txt', format_report(res)
    )


def run(
    paths: Paths,
    name_a: str,
    name_b: str,
    seed: int | None,
    ckpt: Path | None = None,
    device: str = 'cpu',
    calib_steps: int = N_CALIB_STEPS,
) -> AblationResult:
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
