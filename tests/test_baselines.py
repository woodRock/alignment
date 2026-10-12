"""baselines.py: COW-TIC and icoshift."""

import itertools

import numpy as np
import pytest

from gcmsalign.baselines import cow_tic_align, icoshift_align
from gcmsalign.metrics import pearson
from gcmsalign.warps import warp_signal

T = 200


def peaks(positions, width=1.5, heights=None):
    t = np.arange(T)
    heights = heights if heights is not None else np.linspace(1, 3, len(positions))
    return (
        sum(
            h * np.exp(-0.5 * ((t - p) / width) ** 2)
            for p, h in zip(positions, heights, strict=False)
        )
        + 0.01
    )


POS = [35, 60, 90, 120, 150, 175]


@pytest.mark.parametrize('method', [cow_tic_align, icoshift_align])
def test_identical_signals_are_left_alone(method):
    x = peaks(POS)
    al = method(x, x)
    assert np.allclose(al.src, np.arange(T))
    assert np.allclose(al.warped, x) and al.warp_mag == 0


@pytest.mark.parametrize('method', [cow_tic_align, icoshift_align])
def test_warped_is_the_query_read_at_src(method):
    """One warp, three uses: warped TIC = warp_signal(query, src), so the raw TIC and the
    chromatogram can be warped identically."""
    q, r = peaks([p - 6 for p in POS]), peaks(POS)
    al = method(q, r)
    assert np.allclose(al.warped, warp_signal(q, al.src))


CENTRES = [30, 50, 70, 90, 110, 130, 150, 170]  # one peak per icoshift interval


@pytest.mark.parametrize('shift', [-7, 5])
def test_icoshift_recovers_a_shift_per_interval(shift):
    q, r = peaks([p + shift for p in CENTRES], width=1.0), peaks(CENTRES, width=1.0)
    al = icoshift_align(q, r)
    lags = al.details['lags'][1:-1]  # intervals 2–9 hold one peak each
    assert all(lag == -shift for lag in lags)
    assert pearson(al.warped, r) > 0.95 > pearson(q, r)


@pytest.mark.parametrize('shift', [-7, 5])
def test_cow_recovers_a_shift_at_interior_nodes(shift):
    """COW pins its first and last nodes, so only the interior nodes can follow a shift."""
    q, r = peaks([p + shift for p in POS]), peaks(POS)
    al = cow_tic_align(q, r)
    rn, qn = al.details['r_nodes'], al.details['q_nodes']
    assert all(
        abs((b - a) - shift) <= 2 for a, b in zip(rn[1:-1], qn[1:-1], strict=False)
    )  # nodes sit between peaks
    assert pearson(al.warped, r) > pearson(q, r) + 0.3


def test_icoshift_sign_convention():
    """Output bin k reads query k − lag; the reference→query map is t − lag (v1's figure code
    used t + lag)."""
    q, r = peaks([p + 4 for p in CENTRES], width=1.0), peaks(CENTRES, width=1.0)
    al = icoshift_align(q, r)
    for lo, hi, lag in zip(
        al.details['bounds'][:-1], al.details['bounds'][1:], al.details['lags'], strict=False
    ):
        k = np.arange(lo, hi)
        assert np.allclose(al.src[lo:hi], np.clip(k - lag, 0, T - 1))
    assert al.details['lags'][3] == -4
    assert np.argmax(al.warped[60:80]) + 60 == 70  # query peak at 74 lands on 70


def test_icoshift_respects_max_shift():
    q, r = peaks([p + 15 for p in POS]), peaks(POS)
    assert max(abs(l) for l in icoshift_align(q, r, max_shift=3).details['lags']) <= 3


def test_cow_nodes_are_monotone_and_within_slack():
    q, r = peaks([p - 9 for p in POS]), peaks(POS)
    al = cow_tic_align(q, r, n_segs=4, slack=10)
    rn, qn = al.details['r_nodes'], al.details['q_nodes']
    assert rn[0] == qn[0] == 0 and rn[-1] == qn[-1] == T - 1
    assert all(b > a for a, b in itertools.pairwise(qn))
    assert all(abs(a - b) <= 10 for a, b in zip(rn[1:-1], qn[1:-1], strict=False))
    assert np.allclose(al.src, np.interp(np.arange(T), rn, qn))


def test_cow_matches_v1_segment_resampling():
    """v1 built each warped segment by resampling the query segment; reading the query at src
    is the same operation."""
    q, r = peaks([p - 5 for p in POS]), peaks(POS)
    al = cow_tic_align(q, r)
    rn, qn = al.details['r_nodes'], al.details['q_nodes']
    v1 = np.empty(T)
    for s in range(len(rn) - 1):
        seg = q[qn[s] : qn[s + 1] + 1]
        n = rn[s + 1] - rn[s] + 1
        v1[rn[s] : rn[s + 1] + 1] = np.interp(
            np.linspace(0, 1, n), np.linspace(0, 1, len(seg)), seg
        )
    assert np.allclose(al.warped, v1)


def test_warp_magnitude_in_minutes():
    q, r = peaks([p + 4 for p in POS]), peaks(POS)
    al = icoshift_align(q, r)
    assert al.warp_mag == pytest.approx(np.abs(al.src - np.arange(T)).mean() * 0.225)


def test_icoshift_leaves_intervals_without_signal_unshifted():
    """Time-binned runs are zero before acquisition starts; a flat cross-correlation must give
    lag 0, not the first allowed lag (−max_shift)."""
    q, r = peaks(CENTRES, width=1.0), peaks(CENTRES, width=1.0)
    q[:20] = r[:20] = 0.0
    al = icoshift_align(q, r)
    assert al.details['lags'][0] == 0 and np.allclose(al.src[:20], np.arange(20))
