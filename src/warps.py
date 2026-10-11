"""Applying time warps to signals, and the PCHIP warp through anchors (numpy).

Convention used throughout: a warp is given as source positions. For output bin t, the warped
signal reads the input at fractional bin position src[t] (linear interpolation, clamped to the
signal). This is the same pull-back convention as warp_transformer.apply_warp, so a warp has
one meaning everywhere.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator

from config import BIN_MIN, TIME_AXIS

WarpFn = Callable[[np.ndarray], np.ndarray]     # minutes → minutes


def warp_signal(x: np.ndarray, src_bins: np.ndarray) -> np.ndarray:
    """Read x [T] or [T, M] at fractional positions src_bins [T] (bins), linear interpolation.
    Positions are clamped to [0, T−1], so the end bins are read exactly (not their neighbours)."""
    x   = np.asarray(x)
    T   = x.shape[0]
    src = np.clip(np.asarray(src_bins, dtype=np.float64), 0.0, T - 1)
    near = np.round(src)
    src = np.where(np.abs(src - near) < 1e-9, near, src)   # i·BIN_MIN / BIN_MIN is not exactly i
    lo  = np.clip(np.floor(src).astype(np.int64), 0, T - 2)
    a   = np.clip(src - lo, 0.0, 1.0)
    if x.ndim > 1:
        a = a.reshape(-1, *([1] * (x.ndim - 1)))
    return ((1.0 - a) * x[lo] + a * x[lo + 1]).astype(x.dtype, copy=False)


def warp_rows(x: np.ndarray, src_bins: np.ndarray) -> np.ndarray:
    """Warp each row x[i] ([N, T] or [N, T, M]) by src_bins[i] ([N, T]) or one shared src [T]."""
    src = np.asarray(src_bins)
    if src.ndim == 1:
        return np.stack([warp_signal(xi, src) for xi in x])
    return np.stack([warp_signal(xi, si) for xi, si in zip(x, src)])


def src_from_fn(warp_fn: WarpFn | None) -> np.ndarray:
    """Source positions (bins) of a minutes → minutes warp function, evaluated at every bin.
    None means the identity."""
    if warp_fn is None:
        return np.arange(len(TIME_AXIS), dtype=np.float64)
    src = np.asarray(warp_fn(TIME_AXIS), dtype=np.float64) / BIN_MIN
    near = np.round(src)
    return np.where(np.abs(src - near) < 1e-9, near, src)


def monotone_interp(x: np.ndarray, y: np.ndarray) -> WarpFn:
    """Monotone map through the points (x_k, y_k): PCHIP between the outer points and slope 1
    beyond them (the outer points' offset y − x continues), so no end is pinned and nothing is
    extrapolated by a cubic. x and y must be strictly increasing. One point gives a constant
    shift."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(x) != len(y):
        raise ValueError('monotone_interp needs equal-length, non-empty x and y')
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError('monotone_interp needs strictly increasing x and y')
    if len(x) == 1:
        shift = y[0] - x[0]
        return lambda t: np.asarray(t, dtype=np.float64) + shift
    inner = PchipInterpolator(x, y, extrapolate=False)
    lo_shift, hi_shift = y[0] - x[0], y[-1] - x[-1]

    def fn(t):
        t = np.asarray(t, dtype=np.float64)
        out = inner(t)
        out = np.where(t < x[0], t + lo_shift, out)
        return np.where(t > x[-1], t + hi_shift, out)
    return fn


def pchip_warp(anchors: list[tuple[int, int]]) -> WarpFn | None:
    """The PCHIP baseline's warp through (r, q) anchors (r: bin in batch A, q: bin in batch B).

    Returns a minutes → minutes function mapping batch-B time to batch-A time, so
    warp_signal(a_sample, src_from_fn(fn)) puts a batch-A sample on batch B's time axis.
    None (identity) without anchors. Anchors must be strictly increasing in both r and q
    (anchors.extract_anchors guarantees this)."""
    if not anchors:
        return None
    anchors = sorted(anchors)
    r = np.array([a for a, _ in anchors], dtype=np.float64) * BIN_MIN
    q = np.array([b for _, b in anchors], dtype=np.float64) * BIN_MIN
    return monotone_interp(q, r)


def predict_query_bin(anchors: list[tuple[int, int]], r_bin: float) -> float:
    """PCHIP prediction of the batch-B bin of a compound at batch-A bin r_bin, through the
    (r, q) anchors (same construction as pchip_warp, in the A → B direction)."""
    anchors = sorted(anchors)
    r = np.array([a for a, _ in anchors], dtype=np.float64)
    q = np.array([b for _, b in anchors], dtype=np.float64)
    return float(monotone_interp(r, q)(np.array([r_bin], dtype=np.float64))[0])


def invert_src(src_bins: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """For a non-decreasing warp src [T], the output bins t with src(t) = position (linear
    interpolation; slope-1 continuation beyond the ends). Used to map a source position back
    to the output frame."""
    src = np.asarray(src_bins, dtype=np.float64)
    t   = np.arange(len(src), dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    out = np.interp(pos, src, t)
    out = np.where(pos < src[0], pos - src[0], out)
    return np.where(pos > src[-1], t[-1] + pos - src[-1], out)
