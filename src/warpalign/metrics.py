"""Evaluation metrics: TIC correlation, silhouette, warp magnitude, multiple testing."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .config import BIN_MIN, EVAL_BINS


def pearson_rows(a: np.ndarray, b: np.ndarray, bins: slice | None = EVAL_BINS) -> np.ndarray:
    """Pearson r for every (a_i, b_j) row pair of a [n_a, T] and b [n_b, T], i-major flat
    (index i · n_b + j). Computed on the time bins `bins` (default: the common acquisition
    window; None = all bins). A row that is constant over those bins gives NaN."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    if bins is not None:
        a, b = a[:, bins], b[:, bins]
    az = a - a.mean(axis=1, keepdims=True)
    bz = b - b.mean(axis=1, keepdims=True)
    na = np.linalg.norm(az, axis=1, keepdims=True)
    nb = np.linalg.norm(bz, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (az / na) @ (bz / nb).T
    return np.clip(r, -1.0, 1.0).ravel()


def smooth_rows(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along time (the last axis) with σ = sigma bins; sigma 0 = unchanged.
    Edges are padded with the end value, so a zero-filled run start stays zero."""
    x = np.asarray(x, dtype=np.float64)
    return gaussian_filter1d(x, sigma, axis=-1, mode='nearest') if sigma > 0 else x


def pearson_rows_smoothed(
    a: np.ndarray, b: np.ndarray, sigmas: tuple[float, ...]
) -> dict[float, np.ndarray]:
    """pearson_rows after smoothing both a and b with the same Gaussian, for each σ in sigmas."""
    return {s: pearson_rows(smooth_rows(a, s), smooth_rows(b, s)) for s in sigmas}


def all_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows of a [n_a, ...] and b [n_b, ...] expanded to every (a_i, b_j), i-major: row
    i · n_b + j of the results is (a_i, b_j)."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    return np.repeat(a, len(b), axis=0), np.tile(b, (len(a),) + (1,) * (b.ndim - 1))


def pearson_paired(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None, bins: slice | None = EVAL_BINS
) -> np.ndarray:
    """Pearson r of each row pair (a_k, b_k) of a, b [K, T], on the time bins `bins` and, if
    given, only where mask [K, T] is True. NaN for fewer than 3 bins or a constant row."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    m = np.ones(a.shape, dtype=bool) if mask is None else np.atleast_2d(np.asarray(mask, bool))
    if bins is not None:
        a, b, m = a[:, bins], b[:, bins], m[:, bins]
    w = m.astype(np.float64)
    n = w.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        az = (a - (a * w).sum(axis=1, keepdims=True) / n) * w
        bz = (b - (b * w).sum(axis=1, keepdims=True) / n) * w
        r = (az * bz).sum(axis=1) / np.sqrt((az**2).sum(axis=1) * (bz**2).sum(axis=1))
    r = np.where(n[:, 0] >= 3, r, np.nan)
    return np.clip(r, -1.0, 1.0)


def pearson_rows_masked(
    a: np.ndarray, b: np.ndarray, valid_a: np.ndarray, valid_b: np.ndarray, bins: slice = EVAL_BINS
) -> np.ndarray:
    """Like pearson_rows, but each (a_i, b_j) uses only the bins in `bins` where both
    valid_a[i] and valid_b[j] are True (NaN if fewer than 3 such bins or a constant row)."""
    xa, xb = all_pairs(np.asarray(a), np.asarray(b))
    va, vb = all_pairs(np.asarray(valid_a), np.asarray(valid_b))
    return pearson_paired(xa, xb, va & vb, bins)


def pearson(a: np.ndarray, b: np.ndarray, bins: slice | None = EVAL_BINS) -> float:
    """Pearson r of two 1-D signals on `bins`."""
    return float(pearson_rows(a, b, bins)[0])


def tic_r_pairs(chromas_a: np.ndarray, chromas_b: np.ndarray) -> np.ndarray:
    """Normalised-TIC Pearson r for every (A_i, B_j), i-major, on the evaluation window."""
    return pearson_rows(np.asarray(chromas_a).sum(-1), np.asarray(chromas_b).sum(-1))


def within_batch_r(x: np.ndarray, bins: slice | None = EVAL_BINS) -> tuple[float, float, int]:
    """Mean and s.d. of Pearson r over all pairs of distinct rows of x [n, T], and the number of
    pairs."""
    n = len(x)
    if n < 2:
        raise ValueError('within_batch_r needs at least two rows')
    r = pearson_rows(x, x, bins).reshape(n, n)
    v = np.array([r[i, j] for i, j in combinations(range(n), 2)])
    return float(v.mean()), float(v.std()), len(v)


def study_silhouette(chromas_a: np.ndarray, chromas_b: np.ndarray) -> float:
    """Study-label silhouette of per-m/z max projections in PCA space (≤ 50 components).
    Lower = less batch separation. The max projection is largely RT-invariant, so this is a
    secondary diagnostic, not a measure of alignment."""
    feats = np.concatenate([np.asarray(chromas_a).max(axis=1), np.asarray(chromas_b).max(axis=1)])
    labels = np.array([0] * len(chromas_a) + [1] * len(chromas_b))
    n_comp = min(50, feats.shape[0] - 1, feats.shape[1])
    return float(silhouette_score(PCA(n_components=n_comp).fit_transform(feats), labels))


def warp_magnitude(src_bins: np.ndarray) -> float:
    """Mean |w(t) − t| in minutes for source positions src_bins [..., T] (bins)."""
    src = np.asarray(src_bins, dtype=np.float64)
    return float(np.abs(src - np.arange(src.shape[-1])).mean()) * BIN_MIN


def per_a_mean(r: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
    """Per batch-A sample mean of an i-major per-pair array."""
    return np.asarray(r).reshape(n_a, n_b).mean(axis=1)


def holm(pvals) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (monotone, capped at 1), in the input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p, kind='stable')
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def mean_sd(x) -> tuple[float, float]:
    """Mean and sample s.d. (ddof 1; NaN for a single value)."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float('nan')


def paired_wilcoxon(x, y) -> tuple[float, float]:
    """Two-sided paired Wilcoxon signed-rank test of x vs y: (statistic, p). Identical inputs
    (every difference zero) give (0, 1) instead of SciPy's NaN."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.all(x == y):
        return 0.0, 1.0
    res = wilcoxon(x, y, alternative='two-sided')
    return float(res.statistic), float(res.pvalue)
