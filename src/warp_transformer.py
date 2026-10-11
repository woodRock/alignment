"""
ChromaWarpTransformer — dataset-level GC-MS alignment, and the warp primitives it uses.

Input : chromas [N, T, M]   N samples, T = 200 RT bins, M = 1000 m/z channels
Output: aligned [N, T, M], warps [N, T]

A warp gives source positions: output bin t (the shared consensus frame) reads the sample at
fractional bin warps[t]. Warps are monotone and neither end is pinned (see make_monotone_warp).

Architecture: per-bin projection + RT positional encoding + study embedding; n_layers × (local
attention over T within a sample, global attention over N at each bin); a dense warp head
(increments) and a start head (offset of the first bin), both zero-initialised so that an
untrained model predicts exactly the identity warp.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# softplus(u + _UNIT) = 1 at u = 0, so zero logits give unit increments (the identity warp)
_UNIT = math.log(math.e - 1.0)


# ── Reproducibility ───────────────────────────────────────────────────────────

def enable_cuda_determinism() -> None:
    """Make seeded CUDA runs bit-reproducible. CPU runs already are; a no-op without CUDA."""
    if not torch.cuda.is_available():
        return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


# ── Warp primitives ───────────────────────────────────────────────────────────

def apply_warp(chromas: Tensor, warps: Tensor) -> Tensor:
    """Differentiable pull-back warp by linear interpolation.

    chromas [N, T, M] (or [N, T, 1] for TICs), warps [N, T] fractional source positions.
    Positions are clamped to [0, T−1]; the last bin is read exactly (alpha is measured from the
    clamped lower index, so w = T−1 reads bin T−1, not T−2)."""
    N, T, _ = chromas.shape
    idx_lo = warps.detach().floor().long().clamp(0, T - 2)
    alpha  = (warps - idx_lo.to(warps.dtype)).clamp(0.0, 1.0).unsqueeze(-1)
    n_idx  = torch.arange(N, device=chromas.device).unsqueeze(1)
    lo     = chromas[n_idx, idx_lo, :]
    hi     = chromas[n_idx, idx_lo + 1, :]
    return (1.0 - alpha) * lo + alpha * hi          # exact at alpha = 0 and alpha = 1


def make_monotone_warp(raw: Tensor, start: Tensor) -> Tensor:
    """Unconstrained increments raw [N, T] and start offsets start [N] → monotone warps [N, T].

    δ(t) = softplus(raw(t) + c) with c = log(e − 1), so δ = 1 at raw = 0; w(0) = start and
    w(t) = start + Σ_{τ=1..t} δ(τ). Strictly increasing, and neither end is pinned, so drift at
    the start or end of the run can be represented. raw = 0, start = 0 is exactly the identity.
    (raw[:, 0] does not affect the warp.)"""
    deltas = F.softplus(raw[:, 1:] + _UNIT)
    zero   = torch.zeros_like(raw[:, :1])
    return start.unsqueeze(-1) + torch.cat([zero, deltas.cumsum(dim=-1)], dim=-1)


def warp_smoothness_loss(warps: Tensor) -> Tensor:
    """Mean squared deviation of the warp velocity from 1 (the identity's)."""
    velocity = warps[:, 1:] - warps[:, :-1]
    return ((velocity - 1.0) ** 2).mean()


def sample_warp(warp: Tensor, t: Tensor) -> Tensor:
    """Value of a warp warp [T] at fractional output positions t [...] by linear interpolation,
    continued linearly with the end segments' slope outside [0, T−1]. Differentiable in warp
    and t."""
    T  = warp.shape[-1]
    i0 = t.detach().floor().long().clamp(0, T - 2)
    a  = t - i0.to(t.dtype)
    return warp[i0] + a * (warp[i0 + 1] - warp[i0])


def invert_warp(warp: Tensor, positions: Tensor) -> Tensor:
    """Output positions t with warp(t) = position, for a strictly increasing warp [T]; linear
    within each segment and continued with the end segments' slope outside the warp's range.
    Differentiable in warp (and positions). The segment is chosen without gradient."""
    T   = warp.shape[-1]
    i0  = torch.searchsorted(warp.detach().contiguous(), positions.detach().contiguous(),
                             right=True) - 1
    i0  = i0.clamp(0, T - 2)
    w0, w1 = warp[i0], warp[i0 + 1]
    return i0.to(warp.dtype) + (positions - w0) / (w1 - w0).clamp(min=1e-6)


def _anchor_tensors(anchors, device, dtype):
    r = torch.tensor([float(a) for a, _ in anchors], device=device, dtype=dtype)
    q = torch.tensor([float(b) for _, b in anchors], device=device, dtype=dtype)
    return r, q


def predict_query_bins(warps: Tensor, study_ids: Tensor, r_bins) -> Tensor:
    """Predicted batch-B bins of compounds at batch-A bins r_bins, from the warps.

    The consensus position of r is t_c = w̄_A⁻¹(r) (w̄_A: mean batch-A warp); each B sample
    reads w_B(t_c) there. Returns [n_B, K] (one row per batch-B sample)."""
    r = torch.as_tensor(r_bins, device=warps.device, dtype=warps.dtype)
    t_c = invert_warp(warps[study_ids == 0].mean(dim=0), r)               # [K]
    return torch.stack([sample_warp(w, t_c) for w in warps[study_ids == 1]])


def anchor_loss_exact(warps: Tensor, study_ids: Tensor, anchors) -> Tensor:
    """Mean over anchors and batch-B samples of (w_B(w̄_A⁻¹(r)) − q)²: a compound at A bin r and
    B bin q must land on the same consensus bin. Zero (with gradient) without anchors."""
    if not anchors:
        return warps.sum() * 0.0
    _, q = _anchor_tensors(anchors, warps.device, warps.dtype)
    pred = predict_query_bins(warps, study_ids, [a for a, _ in anchors])
    return ((pred - q) ** 2).mean()


def anchor_loss_approx(warps: Tensor, study_ids: Tensor, anchors) -> Tensor:
    """The v1 loss, kept for the before/after comparison: (w_B(r) − w̄_A(r) − (q − r))², reading
    both warps at consensus bin r. Exact only when w̄_A is the identity near r."""
    if not anchors:
        return warps.sum() * 0.0
    r, q = _anchor_tensors(anchors, warps.device, warps.dtype)
    idx = r.long()
    ref = warps[study_ids == 0].mean(dim=0)[idx]
    qry = warps[study_ids == 1][:, idx]
    return ((qry - ref - (q - r)) ** 2).mean()


# ── Transformer building blocks ───────────────────────────────────────────────

class _LocalLayer(nn.Module):
    """Pre-LN transformer layer attending over T (RT) within each sample."""

    def __init__(self, d: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(d * 4, d))

    def forward(self, x: Tensor) -> Tensor:            # [N, T, d]
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class _GlobalLayer(_LocalLayer):
    """Pre-LN transformer layer attending over N (samples) at each RT bin. No positional
    encoding over N: sample order must not matter."""

    def forward(self, x: Tensor) -> Tensor:            # [N, T, d]
        return super().forward(x.transpose(0, 1)).transpose(0, 1)


# ── Model ─────────────────────────────────────────────────────────────────────

class ChromaWarpTransformer(nn.Module):
    """study_ids ([N] int64: 0 = batch A, 1 = batch B) are required at train and inference
    time. N is free (trained on 4–16, used on up to 52)."""

    def __init__(self, mz_bins: int = 1000, n_bins: int = 200, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 3, n_studies: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_bins = n_bins
        self.proj = nn.Sequential(nn.Linear(mz_bins, d_model), nn.LayerNorm(d_model),
                                  nn.Dropout(dropout))
        self.register_buffer('pe', self._sinusoidal_pe(n_bins, d_model))
        self.study_embed   = nn.Embedding(n_studies, d_model)
        self.local_layers  = nn.ModuleList([_LocalLayer(d_model, n_heads, dropout)
                                            for _ in range(n_layers)])
        self.global_layers = nn.ModuleList([_GlobalLayer(d_model, n_heads, dropout)
                                            for _ in range(n_layers)])
        self.warp_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
                                       nn.GELU(), nn.Linear(d_model // 2, 1))
        self.start_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        for head in (self.warp_head, self.start_head):         # identity warp at init
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _sinusoidal_pe(n: int, d: int) -> Tensor:
        pos = torch.arange(n, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe = torch.zeros(n, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def encode(self, chromas: Tensor, study_ids: Tensor) -> Tensor:
        """[N, T, M], [N] → [N, T, d]."""
        x = self.proj(chromas) + self.pe.unsqueeze(0) + self.study_embed(study_ids).unsqueeze(1)
        for local, glob in zip(self.local_layers, self.global_layers):
            x = glob(local(x))
        return x

    def warps(self, chromas: Tensor, study_ids: Tensor) -> Tensor:
        enc   = self.encode(chromas, study_ids)
        raw   = self.warp_head(enc).squeeze(-1)                  # [N, T]
        start = self.start_head(enc.mean(dim=1)).squeeze(-1)     # [N]
        return make_monotone_warp(raw, start)

    def forward(self, chromas: Tensor, study_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (aligned [N, T, M], warps [N, T])."""
        w = self.warps(chromas, study_ids)
        return apply_warp(chromas, w), w
