"""Evaluating the anchor-based methods on a batch pair (normalised and raw TIC r, silhouette,
warp magnitude, coverage), shared by the calibration, PCHIP, ablation and figure scripts.

Coverage: an output bin of a warped sample is *outside* the acquired range if its source
position lies before that sample's acquisition start (it reads zeros) or past the last bin (it
reads the repeated edge value). Every PairEval reports the fraction of evaluated (output bin,
sample) values that are outside, and TIC r with them masked out (r_valid)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EVAL_BINS, N_BINS
from .data import Batch
from .metrics import pearson_rows, pearson_rows_masked, study_silhouette, warp_magnitude
from .warps import pchip_warp, src_from_fn, warp_rows


@dataclass
class PairEval:
    r: np.ndarray  # per (A_i, B_j) normalised TIC r, i-major
    r_raw: np.ndarray  # per (A_i, B_j) raw TIC r, i-major
    sil: float
    warp_mag: float  # minutes (method-specific; see the producing function)
    outside: float  # fraction of evaluated (bin, sample) values read outside the run
    r_valid: np.ndarray  # per-pair normalised TIC r on in-range bins only

    @property
    def mean(self) -> float:
        return float(self.r.mean())

    @property
    def mean_raw(self) -> float:
        return float(self.r_raw.mean())

    @property
    def mean_valid(self) -> float:
        return float(np.nanmean(self.r_valid))


def in_range(src: np.ndarray, start: np.ndarray) -> np.ndarray:
    """[N, T] True where source positions src [N, T] (or [T], shared) lie inside each sample's
    acquired range [start_i, T − 1]."""
    src = np.broadcast_to(np.asarray(src, dtype=np.float64), (len(start), N_BINS))
    return (src >= np.asarray(start)[:, None]) & (src <= N_BINS - 1)


def coverage(
    tic_a: np.ndarray, tic_b: np.ndarray, valid_a: np.ndarray, valid_b: np.ndarray
) -> tuple[float, np.ndarray]:
    """(fraction of evaluated values outside the acquired range, per-pair masked TIC r)."""
    window = np.concatenate([valid_a, valid_b])[:, EVAL_BINS]
    return float(1.0 - window.mean()), pearson_rows_masked(tic_a, tic_b, valid_a, valid_b)


def identity_src() -> np.ndarray:
    return np.arange(N_BINS, dtype=np.float64)


def unaligned(a: Batch, b: Batch) -> PairEval:
    outside, r_valid = coverage(
        a.tic,
        b.tic,
        in_range(identity_src(), a.acquisition_start),
        in_range(identity_src(), b.acquisition_start),
    )
    return PairEval(
        pearson_rows(a.tic, b.tic),
        pearson_rows(a.tic_raw, b.tic_raw),
        study_silhouette(a.chromas, b.chromas),
        0.0,
        outside,
        r_valid,
    )


def pchip(a: Batch, b: Batch, anchors) -> PairEval:
    """PCHIP through the pooled anchors: one warp applied to every batch-A sample (A → B frame),
    B unwarped. warp_mag = mean |w(t) − t| of that warp."""
    src = src_from_fn(pchip_warp(anchors))
    al_tic = warp_rows(a.tic, src)
    al_raw = warp_rows(a.tic_raw, src)
    outside, r_valid = coverage(
        al_tic,
        b.tic,
        in_range(src, a.acquisition_start),
        in_range(identity_src(), b.acquisition_start),
    )
    return PairEval(
        pearson_rows(al_tic, b.tic),
        pearson_rows(al_raw, b.tic_raw),
        study_silhouette(warp_rows(a.chromas, src), b.chromas),
        warp_magnitude(src),
        outside,
        r_valid,
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
    outside, r_valid = coverage(
        tic[:n_a],
        tic[n_a:],
        in_range(warps[:n_a], a.acquisition_start),
        in_range(warps[n_a:], b.acquisition_start),
    )
    return PairEval(
        pearson_rows(tic[:n_a], tic[n_a:]),
        pearson_rows(raw[:n_a], raw[n_a:]),
        study_silhouette(al[:n_a], al[n_a:]),
        warp_magnitude(warps[n_a:]),
        outside,
        r_valid,
    )


def warp_mags(warps: np.ndarray, n_a: int) -> tuple[float, float, float]:
    """(all samples, batch A, batch B) mean |w(t) − t| in minutes."""
    return warp_magnitude(warps), warp_magnitude(warps[:n_a]), warp_magnitude(warps[n_a:])
