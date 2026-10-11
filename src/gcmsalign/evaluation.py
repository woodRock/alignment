"""Evaluating the anchor-based methods on a batch pair (normalised and raw TIC r, silhouette,
warp magnitude), shared by the calibration, PCHIP, ablation and figure scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Batch
from .metrics import pearson_rows, study_silhouette, warp_magnitude
from .warps import pchip_warp, src_from_fn, warp_rows


@dataclass
class PairEval:
    r: np.ndarray  # per (A_i, B_j) normalised TIC r, i-major
    r_raw: np.ndarray  # per (A_i, B_j) raw TIC r, i-major
    sil: float
    warp_mag: float  # minutes (method-specific; see the producing function)

    @property
    def mean(self) -> float:
        return float(self.r.mean())

    @property
    def mean_raw(self) -> float:
        return float(self.r_raw.mean())


def unaligned(a: Batch, b: Batch) -> PairEval:
    return PairEval(
        pearson_rows(a.tic, b.tic),
        pearson_rows(a.tic_raw, b.tic_raw),
        study_silhouette(a.chromas, b.chromas),
        0.0,
    )


def pchip(a: Batch, b: Batch, anchors) -> PairEval:
    """PCHIP through the pooled anchors: one warp applied to every batch-A sample (A → B frame),
    B unwarped. warp_mag = mean |w(t) − t| of that warp."""
    src = src_from_fn(pchip_warp(anchors))
    al_tic = warp_rows(a.tic, src)
    al_raw = warp_rows(a.tic_raw, src)
    return PairEval(
        pearson_rows(al_tic, b.tic),
        pearson_rows(al_raw, b.tic_raw),
        study_silhouette(warp_rows(a.chromas, src), b.chromas),
        warp_magnitude(src),
    )


def warped_pair(
    a: Batch, b: Batch, warps: np.ndarray, aligned: np.ndarray | None = None
) -> PairEval:
    """Both batches warped by per-sample warps [n_a + n_b, T] (A first), e.g. the
    WarpTransformer's. aligned: the model's own aligned chromatograms, if available (otherwise
    computed). warp_mag = batch-B samples."""
    n_a = len(a)
    chromas = np.concatenate([a.chromas, b.chromas])
    al = warp_rows(chromas, warps) if aligned is None else aligned
    raw = warp_rows(np.concatenate([a.tic_raw, b.tic_raw]), warps)
    tic = al.sum(-1)
    return PairEval(
        pearson_rows(tic[:n_a], tic[n_a:]),
        pearson_rows(raw[:n_a], raw[n_a:]),
        study_silhouette(al[:n_a], al[n_a:]),
        warp_magnitude(warps[n_a:]),
    )


def warp_mags(warps: np.ndarray, n_a: int) -> tuple[float, float, float]:
    """(all samples, batch A, batch B) mean |w(t) − t| in minutes."""
    return warp_magnitude(warps), warp_magnitude(warps[:n_a]), warp_magnitude(warps[n_a:])
