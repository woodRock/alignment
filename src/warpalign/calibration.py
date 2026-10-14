"""Seeding, devices, model loading, anchor-supervised test-time calibration, and the
network-free warp used as a control."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from .config import LAMBDA_ANCHOR, LAMBDA_FRAME, LAMBDA_SMOOTH, LR_CALIB, N_BINS, N_CALIB_STEPS
from .warp_transformer import (
    ChromaWarpTransformer,
    anchor_loss_exact,
    enable_cuda_determinism,
    frame_loss,
    make_monotone_warp,
    warp_smoothness_loss,
)

LossFn = Callable[[torch.Tensor, torch.Tensor, list], torch.Tensor]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    enable_cuda_determinism()


def pick_device(name: str = 'auto') -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(
    ckpt: Path | None, device: torch.device, seed: int | None = None
) -> ChromaWarpTransformer:
    """The pretrained model from ckpt, or (ckpt=None) a random-init model seeded with seed."""
    if ckpt is None and seed is not None:
        set_seed(seed)
    model = ChromaWarpTransformer()
    if ckpt is not None:
        state = torch.load(ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f'{ckpt} does not match ChromaWarpTransformer (missing {missing}, '
                f'unexpected {unexpected}); v1 checkpoints pin both warp ends and '
                'are not compatible: retrain with 02_train_warp_transformer.py'
            )
    return model.to(device)


def batch_tensors(
    chromas_a: np.ndarray, chromas_b: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stacked [N, T, M] chromatograms (A first) and study ids (0 = A, 1 = B)."""
    x = torch.from_numpy(np.concatenate([chromas_a, chromas_b]).astype(np.float32)).to(device)
    sid = torch.tensor([0] * len(chromas_a) + [1] * len(chromas_b), dtype=torch.long, device=device)
    return x, sid


@torch.no_grad()
def predict(
    model: ChromaWarpTransformer, x: torch.Tensor, study_ids: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """(aligned [N, T, M], warps [N, T]) in eval mode, as numpy."""
    was_training = model.training
    model.eval()
    aligned, warps = model(x, study_ids)
    model.train(was_training)
    return aligned.cpu().numpy(), warps.cpu().numpy()


@dataclass
class CalibrationLog:
    seconds: float = 0.0
    best_loss: float = float('inf')
    history: list[dict] = field(default_factory=list)  # every log_every steps


def calibrate(
    model: ChromaWarpTransformer,
    x: torch.Tensor,
    study_ids: torch.Tensor,
    anchors: list,
    seed: int | None = None,
    steps: int = N_CALIB_STEPS,
    lr: float = LR_CALIB,
    loss_fn: LossFn = anchor_loss_exact,
    log_every: int = 0,
    on_log: Callable[[int, dict, torch.Tensor], None] | None = None,
) -> CalibrationLog:
    """Anchor-supervised calibration in place: Adam, cosine schedule, grad-norm clip 1, the
    parameters with the lowest training loss kept (the state the loss was measured on, before
    that step's update; the loss includes dropout, as in training). Seeded immediately before
    the loop, so every caller with the same seed and device consumes the same dropout RNG
    stream. Without anchors the model is left unchanged. on_log(step, record, aligned) is
    called every log_every steps."""
    log = CalibrationLog()
    if not anchors:
        model.eval()
        return log
    if seed is not None:
        set_seed(seed)
    t0 = time.perf_counter()
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for step in range(1, steps + 1):
        aligned, warps = model(x, study_ids)
        loss, parts = calibration_objective(warps, study_ids, anchors, loss_fn)
        if float(loss) < log.best_loss:  # snapshot the parameters this loss was measured on
            log.best_loss = float(loss)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if log_every and (step % log_every == 0 or step == steps):
            rec = {
                'step': step,
                'loss': float(loss),
                **parts,
            }
            if on_log is not None:
                on_log(step, rec, aligned.detach())
            log.history.append(rec)

    model.load_state_dict(best_state)
    model.eval()
    if x.device.type == 'mps':
        torch.mps.synchronize()
    log.seconds = time.perf_counter() - t0
    return log


def calibration_objective(
    warps: torch.Tensor, study_ids: torch.Tensor, anchors: list, loss_fn: LossFn = anchor_loss_exact
) -> tuple[torch.Tensor, dict[str, float]]:
    """The calibration loss: λ_anchor · anchor + λ_smooth · smoothness + λ_frame · frame, and its
    unweighted parts. Shared by calibrate and free_warp."""
    a_loss = loss_fn(warps, study_ids, anchors)
    s_loss = warp_smoothness_loss(warps)
    f_loss = frame_loss(warps, study_ids)
    total = LAMBDA_ANCHOR * a_loss + LAMBDA_SMOOTH * s_loss + LAMBDA_FRAME * f_loss
    return total, {'anchor': float(a_loss), 'smooth': float(s_loss), 'frame': float(f_loss)}


def final_losses(
    warps: np.ndarray | torch.Tensor, study_ids: torch.Tensor, anchors: list
) -> tuple[float, float, float]:
    """(anchor, smoothness, frame) losses of given warps."""
    w = torch.as_tensor(warps, dtype=torch.float32, device=study_ids.device)
    _, parts = calibration_objective(w, study_ids, anchors)
    return parts['anchor'], parts['smooth'], parts['frame']


def free_warp(
    n_a: int, n_b: int, anchors: list, lr: float, seed: int, steps: int = N_CALIB_STEPS
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Network-free control: one monotone warp per batch with the WarpTransformer head's
    parameterisation (shape logits, start offset, end shift), fitted with the same loss and
    optimisation loop as calibrate. Per-sample warps would get identical gradients within a
    batch, so per-batch parameters are equivalent. CPU.

    Initialisation: batch A at the identity and batch B at the best constant shift, the anchors'
    median drift (start offset = median(q − r), end shift 0, so w_B(t) = t + drift). The loop then
    only learns deviations from that shift. Started at the identity instead, Adam stopped at
    warps that fit the anchors by bending between them although a pure shift scores lower on
    the same loss, which would handicap the control and overstate what the network adds.
    Returns (warps [n_a + n_b, T], (anchor, smoothness, frame) losses)."""
    set_seed(seed)
    study_ids = torch.tensor([0] * n_a + [1] * n_b, dtype=torch.long)
    drift = float(np.median([q - r for r, q in anchors])) if anchors else 0.0
    params = [
        torch.zeros(2, N_BINS, requires_grad=True),  # shape logits
        torch.tensor([0.0, drift], requires_grad=True),  # start offsets: A identity, B shifted
        torch.zeros(2, requires_grad=True),  # end shifts (bins)
    ]
    opt = optim.Adam(params, lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    def warps_of(ps: list[torch.Tensor]) -> torch.Tensor:
        return make_monotone_warp(*ps)[study_ids]

    best, best_params = float('inf'), [p.detach().clone() for p in params]
    for _ in range(steps):
        w = warps_of(params)
        loss, _ = calibration_objective(w, study_ids, anchors)
        if float(loss) < best:
            best, best_params = float(loss), [p.detach().clone() for p in params]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
    with torch.no_grad():
        w = warps_of(best_params)
    return w.numpy(), final_losses(w, study_ids, anchors)
