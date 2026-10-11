"""warps.py: applying warps, PCHIP warp, prediction and inversion (numpy)."""
import numpy as np
import pytest

from config import BIN_MIN, N_BINS, TIME_AXIS
from warps import (invert_src, monotone_interp, pchip_warp, predict_query_bin, src_from_fn,
                   warp_rows, warp_signal)


def test_time_axis_has_one_spacing():
    assert np.allclose(np.diff(TIME_AXIS), BIN_MIN)
    assert TIME_AXIS[0] == 0 and np.isclose(TIME_AXIS[-1], (N_BINS - 1) * BIN_MIN)


def test_identity_warp_returns_input_exactly_1d_and_2d(rng):
    x1 = rng.normal(size=N_BINS)
    x2 = rng.normal(size=(N_BINS, 7)).astype(np.float32)
    ident = np.arange(N_BINS, dtype=float)
    assert np.array_equal(warp_signal(x1, ident), x1)
    assert np.array_equal(warp_signal(x2, ident), x2)


def test_identity_warp_function_returns_input_exactly(rng):
    """Section C, test 2: warping with t ↦ t must not stretch the signal (v1 read bin i from
    1.005·i because two time axes were mixed)."""
    x = rng.normal(size=(N_BINS, 5))
    assert np.array_equal(src_from_fn(lambda t: t), np.arange(N_BINS))
    assert np.array_equal(warp_signal(x, src_from_fn(lambda t: t)), x)
    assert np.array_equal(warp_signal(x, src_from_fn(None)), x)


def test_last_bin_is_read_exactly():
    x = np.arange(N_BINS, dtype=float) ** 2
    assert warp_signal(x, np.full(N_BINS, N_BINS - 1.0))[0] == x[-1]


def test_integer_shift_and_clamping():
    x = np.arange(N_BINS, dtype=float)
    out = warp_signal(x, np.arange(N_BINS) - 5.0)
    assert np.array_equal(out[5:], x[:-5]) and np.all(out[:5] == 0)
    out = warp_signal(x, np.arange(N_BINS) + 5.0)
    assert np.array_equal(out[:-5], x[5:]) and np.all(out[-5:] == x[-1])


def test_fractional_positions_match_np_interp(rng):
    x = rng.normal(size=N_BINS)
    src = np.sort(rng.uniform(0, N_BINS - 1, N_BINS))
    assert np.allclose(warp_signal(x, src), np.interp(src, np.arange(N_BINS), x))


def test_warp_rows_shared_and_per_row(rng):
    x = rng.normal(size=(3, N_BINS))
    src = np.arange(N_BINS) + 1.5
    assert np.array_equal(warp_rows(x, src), np.stack([warp_signal(r, src) for r in x]))
    srcs = np.stack([np.arange(N_BINS) + k for k in range(3)], dtype=float)
    assert np.array_equal(warp_rows(x, srcs)[2], warp_signal(x[2], srcs[2]))


def test_warp_is_linear_so_it_commutes_with_summing_mz(rng):
    x = rng.uniform(size=(N_BINS, 6))
    src = np.sort(rng.uniform(-3, N_BINS + 3, N_BINS))
    assert np.allclose(warp_signal(x, src).sum(1), warp_signal(x.sum(1), src))


class TestMonotoneInterp:
    def test_passes_through_points_and_is_monotone(self):
        x, y = np.array([2.0, 5.0, 9.0, 14.0]), np.array([1.0, 3.0, 9.5, 12.0])
        f = monotone_interp(x, y)
        assert np.allclose(f(x), y)
        t = np.linspace(-10, 30, 500)
        assert np.all(np.diff(f(t)) > 0)

    def test_slope_one_beyond_the_outer_points(self):
        f = monotone_interp(np.array([10.0, 20.0]), np.array([7.0, 18.0]))
        assert np.allclose(f(np.array([0.0, 5.0])), [-3.0, 2.0])      # offset −3 continues
        assert np.allclose(f(np.array([25.0, 40.0])), [23.0, 38.0])   # offset −2 continues

    def test_single_point_is_a_shift(self):
        assert np.allclose(monotone_interp(np.array([4.0]), np.array([1.0]))(np.arange(3.0)),
                           [-3, -2, -1])

    @pytest.mark.parametrize('x,y', [([], []), ([1, 2], [3]), ([1, 1], [2, 3]), ([1, 2], [3, 2])])
    def test_rejects_bad_input(self, x, y):
        with pytest.raises(ValueError):
            monotone_interp(np.array(x, float), np.array(y, float))


class TestPchipWarp:
    def test_no_anchors_is_identity(self):
        assert pchip_warp([]) is None

    def test_constant_drift_is_a_pure_shift_everywhere(self):
        """Ends are not pinned: a constant drift stays constant up to the run's edges."""
        anchors = [(40, 32), (90, 82), (150, 142)]               # q = r − 8
        src = src_from_fn(pchip_warp(anchors))
        assert np.allclose(src, np.arange(N_BINS) + 8)

    def test_maps_b_time_to_a_time_at_anchors(self):
        anchors = [(30, 20), (80, 66), (160, 150)]
        fn = pchip_warp(anchors)
        for r, q in anchors:
            assert np.isclose(fn(np.array([q * BIN_MIN]))[0], r * BIN_MIN)

    def test_puts_a_sample_on_the_b_axis(self):
        """A peak at A bin r lands at B bin q after warping A with the PCHIP warp."""
        anchors = [(30, 20), (80, 66), (160, 150)]
        a = np.zeros(N_BINS)
        a[80] = 1.0
        warped = warp_signal(a, src_from_fn(pchip_warp(anchors)))
        assert np.argmax(warped) == 66

    def test_unsorted_input_is_accepted(self):
        anchors = [(80, 66), (30, 20), (160, 150)]
        assert np.allclose(src_from_fn(pchip_warp(anchors)), src_from_fn(pchip_warp(sorted(anchors))))


def test_predict_query_bin_inverts_pchip_warp_direction():
    anchors = [(30, 20), (80, 66), (160, 150)]
    for r, q in anchors:
        assert np.isclose(predict_query_bin(anchors, r), q)
    assert np.isclose(predict_query_bin(anchors, 10), 0)        # slope 1 beyond: 10 − 10
    assert np.isclose(predict_query_bin(anchors, 190), 180)


def test_invert_src_round_trip(rng):
    src = np.cumsum(rng.uniform(0.5, 1.5, N_BINS)) - 3.0
    t = rng.uniform(0, N_BINS - 1, 50)
    pos = np.interp(t, np.arange(N_BINS), src)
    assert np.allclose(invert_src(src, pos), t)
    # beyond the ends: slope-1 continuation
    assert np.isclose(invert_src(src, np.array([src[0] - 4.0]))[0], -4.0)
    assert np.isclose(invert_src(src, np.array([src[-1] + 2.0]))[0], N_BINS - 1 + 2.0)
