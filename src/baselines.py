"""Classical TIC-only alignment baselines: COW-TIC and icoshift.

Both align a query TIC onto a reference TIC and return the warp as source positions (bins,
see warps.py), so the same warp can be applied to the raw TIC or the full chromatogram with
warps.warp_signal, and plotted, from one implementation.

COW-TIC   Nielsen, Carstensen & Smedsgaard 1998, J. Chromatogr. A 805, 17–35.
          Correlation optimised warping: n_segs equal reference segments; interior query
          boundaries free within ±slack bins; dynamic programming maximises the summed
          segment-wise Pearson r; each segment is resampled linearly.
icoshift  Savorani, Tomasi & Engelsen 2010, J. Magn. Reson. 202, 190–202.
          n_intervals equal intervals, each shifted by the integer lag (|lag| ≤ max_shift) that
          maximises its cross-correlation with the reference interval.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import correlate

from metrics import warp_magnitude
from warps import warp_signal

COW_N_SEGS      = 4
COW_SLACK       = 22     # ±bins per interior node (~5 min)
ICO_N_INTERVALS = 10
ICO_MAX_SHIFT   = 20     # ±bins (~4.5 min)


@dataclass
class TicAlignment:
    warped: np.ndarray          # [T] query TIC on the reference axis
    src: np.ndarray             # [T] source positions in the query (bins)
    details: dict = field(default_factory=dict)

    @property
    def warp_mag(self) -> float:
        return warp_magnitude(self.src)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    az, bz = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(az), np.linalg.norm(bz)
    return float(az @ bz / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def _resample(seg: np.ndarray, n: int) -> np.ndarray:
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(seg)), seg)


def cow_tic_align(query_tic: np.ndarray, ref_tic: np.ndarray, n_segs: int = COW_N_SEGS,
                  slack: int = COW_SLACK) -> TicAlignment:
    """COW of a query TIC onto a reference TIC (same length T). Falls back to the identity if
    no valid node path exists. details: r_nodes, q_nodes (bins)."""
    query_tic = np.asarray(query_tic, dtype=np.float64)
    ref_tic   = np.asarray(ref_tic, dtype=np.float64)
    T = len(ref_tic)
    seg_len = T // n_segs
    r_nodes = [i * seg_len for i in range(n_segs)] + [T - 1]
    cands = [[r_nodes[0]]] + [list(range(max(1, rn - slack), min(T - 2, rn + slack) + 1))
                              for rn in r_nodes[1:-1]] + [[r_nodes[-1]]]

    score: list[dict[int, float]] = [{} for _ in r_nodes]
    back:  list[dict[int, int]]   = [{} for _ in r_nodes]
    score[0][r_nodes[0]] = 0.0
    for s in range(n_segs):
        r_seg = ref_tic[r_nodes[s]:r_nodes[s + 1] + 1]
        for q_s, cum in score[s].items():
            for q_e in cands[s + 1]:
                if q_e - q_s < 1:
                    continue
                total = cum + _pearson(r_seg, _resample(query_tic[q_s:q_e + 1], len(r_seg)))
                if total > score[s + 1].get(q_e, -np.inf):
                    score[s + 1][q_e], back[s + 1][q_e] = total, q_s

    if r_nodes[-1] not in back[-1]:
        ident = np.arange(T, dtype=np.float64)
        return TicAlignment(query_tic.copy(), ident, {'r_nodes': r_nodes, 'q_nodes': r_nodes})
    q_nodes = [r_nodes[-1]]
    for s in range(n_segs, 0, -1):
        q_nodes.append(back[s][q_nodes[-1]])
    q_nodes = q_nodes[::-1]

    src = np.interp(np.arange(T), r_nodes, q_nodes)    # linear within each segment
    return TicAlignment(warp_signal(query_tic, src), src,
                        {'r_nodes': list(r_nodes), 'q_nodes': list(map(int, q_nodes))})


def icoshift_align(query_tic: np.ndarray, ref_tic: np.ndarray,
                   n_intervals: int = ICO_N_INTERVALS, max_shift: int = ICO_MAX_SHIFT
                   ) -> TicAlignment:
    """icoshift of a query TIC onto a reference TIC. Shifted-in values are the edge values
    (clamping). details: bounds (bins), lags (bins; output k reads query k − lag)."""
    query_tic = np.asarray(query_tic, dtype=np.float64)
    ref_tic   = np.asarray(ref_tic, dtype=np.float64)
    T = len(query_tic)
    bounds = np.round(np.linspace(0, T, n_intervals + 1)).astype(int)
    src  = np.arange(T, dtype=np.float64)
    lags = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        n = hi - lo
        if n < 4:
            lags.append(0)
            continue
        cc = correlate(ref_tic[lo:hi], query_tic[lo:hi], mode='full')   # index lag + n − 1
        allowed = np.arange(-(n - 1), n)
        ok = np.abs(allowed) <= max_shift
        if np.ptp(cc[ok]) == 0:          # no signal in the interval (e.g. before the run starts)
            lags.append(0)
            continue
        lag = int(allowed[ok][np.argmax(cc[ok])])
        src[lo:hi] = np.clip(np.arange(lo, hi) - lag, 0, T - 1)
        lags.append(lag)
    return TicAlignment(warp_signal(query_tic, src), src,
                        {'bounds': bounds.tolist(), 'lags': lags})


METHODS = {'COW-TIC': cow_tic_align, 'icoshift': icoshift_align}
