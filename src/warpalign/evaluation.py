"""Evaluating the anchor-based methods on a batch pair (normalised and raw TIC r, silhouette,
warp magnitude, coverage), shared by the calibration, PCHIP, ablation and figure scripts.

Smoothed TIC r: every PairEval also reports normalised and raw TIC r after smoothing both
batches' TICs with the same Gaussian (config.SMOOTH_SIGMAS), so methods that resample
differently are compared on equal terms.

Empty bins: some runs record scans with no ions (num_readings = 0) in parts of the run, which
bin to exact zeros. Every PairEval also reports TIC r on bins non-empty in both samples.

Every metric is kept per sample pair in PairEval.metrics, keyed by its saved-array suffix
(METRICS), so the steps save and report the same set. metric_suffixes, METRICS, tic_metrics,
in_range and identity_src are alignbench.metrics.tic's / alignbench.warps' since warpalign 3.0.

Coverage: an output bin of a warped sample is *outside* the acquired range if its source
position lies before that sample's acquisition start (it reads zeros) or past the last bin (it
reads the repeated edge value). Every PairEval reports the fraction of evaluated (output bin,
sample) values that are outside, and TIC r with them masked out (r_valid)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from alignbench.metrics.tic import METRICS, in_range, metric_suffixes, tic_metrics  # noqa: F401
from alignbench.warps import identity_src

from ._compat import (
    Batch,
    WarpFn,
    constant_shift_warp,
    linear_warp,
    pchip_warp,
    src_from_fn,
    study_silhouette,
    warp_magnitude,
    warp_rows,
)
from .config import BIN_MIN, N_BINS, SMOOTH_SIGMAS


@dataclass
class PairEval:
    """One method on one batch pair: per-pair metric arrays (i-major over (A_i, B_j)) keyed by
    suffix (METRICS: '' = normalised TIC r, '_raw', '_raw_s1', '_valid', …), and scalars."""

    metrics: dict[str, np.ndarray]
    sil: float
    warp_mag: float  # minutes (method-specific; see the producing function)
    outside: float  # fraction of evaluated (bin, sample) values read outside the run

    @property
    def r(self) -> np.ndarray:
        return self.metrics['']

    @property
    def r_raw(self) -> np.ndarray:
        return self.metrics['_raw']

    @property
    def r_valid(self) -> np.ndarray:
        return self.metrics['_valid']

    @property
    def r_smooth(self) -> dict[float, np.ndarray]:
        return {s: self.metrics[f'_s{s:g}'] for s in SMOOTH_SIGMAS}

    @property
    def r_raw_smooth(self) -> dict[float, np.ndarray]:
        return {s: self.metrics[f'_raw_s{s:g}'] for s in SMOOTH_SIGMAS}

    @property
    def mean(self) -> float:
        return float(self.r.mean())

    @property
    def mean_raw(self) -> float:
        return float(self.r_raw.mean())

    @property
    def mean_valid(self) -> float:
        return float(np.nanmean(self.r_valid))

    def m(self, suffix: str) -> float:
        """Mean of one metric (NaN-ignoring: masked metrics can be undefined for a pair)."""
        return float(np.nanmean(self.metrics[suffix]))

    def arrays(self, key: str) -> dict[str, np.ndarray]:
        """Every per-pair metric as saved arrays <key><suffix>."""
        return {key + sfx: v for sfx, v in self.metrics.items()}


def unaligned(a: Batch, b: Batch) -> PairEval:
    met, outside = tic_metrics(
        a.tic,
        b.tic,
        a.tic_raw,
        b.tic_raw,
        in_range(identity_src(), a.acquisition_start),
        in_range(identity_src(), b.acquisition_start),
    )
    return PairEval(met, study_silhouette(a.chromas, b.chromas), 0.0, outside)


def anchor_warp(a: Batch, b: Batch, warp_fn: WarpFn | None) -> PairEval:
    """One anchor-derived warp (batch-B minutes → batch-A minutes) applied to every batch-A
    sample (A → B frame), B unwarped. warp_mag = mean |w(t) − t| of that warp."""
    src = src_from_fn(warp_fn)
    met, outside = tic_metrics(
        warp_rows(a.tic, src),
        b.tic,
        warp_rows(a.tic_raw, src),
        b.tic_raw,
        in_range(src, a.acquisition_start),
        in_range(identity_src(), b.acquisition_start),
    )
    sil = study_silhouette(warp_rows(a.chromas, src), b.chromas)
    return PairEval(met, sil, warp_magnitude(src), outside)


def pchip(a: Batch, b: Batch, anchors) -> PairEval:
    """PCHIP through the pooled anchors (warps.pchip_warp), applied as anchor_warp."""
    return anchor_warp(a, b, pchip_warp(anchors))


def constant_shift(a: Batch, b: Batch, anchors) -> PairEval:
    """Batch A shifted by the median anchor drift (warps.constant_shift_warp)."""
    return anchor_warp(a, b, constant_shift_warp(anchors))


def linear(a: Batch, b: Batch, anchors) -> PairEval:
    """The least-squares line through the anchors (warps.linear_warp)."""
    return anchor_warp(a, b, linear_warp(anchors))


ANCHOR_BASELINES = {
    'Constant shift': ('constant_shift', constant_shift),
    'Linear warp': ('linear', linear),
    'PCHIP (pooled anchors)': ('pchip_pooled', pchip),
}  # report name → (saved-array key, evaluation through the pooled anchors)


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
    met, outside = tic_metrics(
        tic[:n_a],
        tic[n_a:],
        raw[:n_a],
        raw[n_a:],
        in_range(warps[:n_a], a.acquisition_start),
        in_range(warps[n_a:], b.acquisition_start),
    )
    return PairEval(met, study_silhouette(al[:n_a], al[n_a:]), warp_magnitude(warps[n_a:]), outside)


def shifted(a: Batch, b: Batch, shift_a: float, shift_b: float) -> PairEval:
    """Resampling control, no alignment: every sample of A read shift_a bins later and every
    sample of B shift_b bins later (linear interpolation). A fractional shift averages each bin
    with its neighbour, which is what any warp does; warp_mag = shift_a."""
    src_a = np.clip(identity_src() + shift_a, 0, N_BINS - 1)
    src_b = np.clip(identity_src() + shift_b, 0, N_BINS - 1)
    w = np.concatenate([np.tile(src_a, (len(a), 1)), np.tile(src_b, (len(b), 1))])
    ev = warped_pair(a, b, w)
    ev.warp_mag = abs(shift_a) * BIN_MIN
    return ev


def warp_mags(warps: np.ndarray, n_a: int) -> tuple[float, float, float]:
    """(all samples, batch A, batch B) mean |w(t) − t| in minutes."""
    return warp_magnitude(warps), warp_magnitude(warps[:n_a]), warp_magnitude(warps[n_a:])
