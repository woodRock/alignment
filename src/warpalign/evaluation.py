"""Evaluating the anchor-based methods on a batch pair (normalised and raw TIC r, silhouette,
warp magnitude, coverage), shared by the calibration, PCHIP, ablation and figure scripts.

Smoothed TIC r: every PairEval also reports normalised and raw TIC r after smoothing both
batches' TICs with the same Gaussian (config.SMOOTH_SIGMAS), so methods that resample
differently are compared on equal terms.

Empty bins: some runs record scans with no ions (num_readings = 0) in parts of the run, which
bin to exact zeros. Every PairEval also reports TIC r on bins non-empty in both samples.

Every metric is kept per sample pair in PairEval.metrics, keyed by its saved-array suffix
(METRICS), so the steps save and report the same set.

Coverage: an output bin of a warped sample is *outside* the acquired range if its source
position lies before that sample's acquisition start (it reads zeros) or past the last bin (it
reads the repeated edge value). Every PairEval reports the fraction of evaluated (output bin,
sample) values that are outside, and TIC r with them masked out (r_valid)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BIN_MIN, EVAL_BINS, N_BINS, SMOOTH_SIGMAS
from .data import Batch
from .metrics import all_pairs, pearson_paired, smooth_rows, study_silhouette, warp_magnitude
from .warps import WarpFn, constant_shift_warp, linear_warp, pchip_warp, src_from_fn, warp_rows


def metric_suffixes(sigmas: tuple[float, ...] = SMOOTH_SIGMAS) -> dict[str, str]:
    """Report label → saved-array suffix of every per-pair TIC metric: an array saved as <key>
    holds normalised TIC r, <key><suffix> the others (e.g. 'pchip_pooled_raw_s1')."""
    return {
        'TIC r': '',
        **{f'TIC r σ{s:g}': f'_s{s:g}' for s in sigmas},
        'raw r': '_raw',
        **{f'raw r σ{s:g}': f'_raw_s{s:g}' for s in sigmas},
        'r in range': '_valid',
        'r non-empty': '_nonempty',
        'raw r non-empty': '_raw_nonempty',
    }


METRICS = metric_suffixes()


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


def tic_metrics(
    tic_a: np.ndarray,
    tic_b: np.ndarray,
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    paired: bool = False,
) -> tuple[dict[str, np.ndarray], float]:
    """(per-pair metrics keyed by METRICS suffix, outside fraction) of warped TICs.

    paired=False: every (A_i, B_j) of tic_a [n_a, T] and tic_b [n_b, T], i-major. paired=True:
    row k of tic_a against row k of tic_b (each pair warped separately, as COW-TIC).
    valid_*: in-range masks (in_range). Non-empty bins: raw TIC > 0 in both warped samples."""
    xs = [np.asarray(x, dtype=np.float64) for x in (tic_a, tic_b, raw_a, raw_b)]
    va, vb = np.asarray(valid_a, bool), np.asarray(valid_b, bool)
    outside = float(1.0 - np.concatenate([va, vb])[:, EVAL_BINS].mean())
    if not paired:
        (ta, tb), (ra, rb), (va, vb) = all_pairs(*xs[:2]), all_pairs(*xs[2:]), all_pairs(va, vb)
    else:
        ta, tb, ra, rb = xs
    nonempty = (ra > 0) & (rb > 0)
    out = {'': pearson_paired(ta, tb), '_raw': pearson_paired(ra, rb)}
    for s in SMOOTH_SIGMAS:
        out[f'_s{s:g}'] = pearson_paired(smooth_rows(ta, s), smooth_rows(tb, s))
        out[f'_raw_s{s:g}'] = pearson_paired(smooth_rows(ra, s), smooth_rows(rb, s))
    out['_valid'] = pearson_paired(ta, tb, va & vb)
    out['_nonempty'] = pearson_paired(ta, tb, nonempty)
    out['_raw_nonempty'] = pearson_paired(ra, rb, nonempty)
    return out, outside


def in_range(src: np.ndarray, start: np.ndarray) -> np.ndarray:
    """[N, T] True where source positions src [N, T] (or [T], shared) lie inside each sample's
    acquired range [start_i, T − 1]."""
    src = np.broadcast_to(np.asarray(src, dtype=np.float64), (len(start), N_BINS))
    return (src >= np.asarray(start)[:, None]) & (src <= N_BINS - 1)


def identity_src() -> np.ndarray:
    return np.arange(N_BINS, dtype=np.float64)


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
