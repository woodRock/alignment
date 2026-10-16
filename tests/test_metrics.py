"""metrics.py and config.py."""

import numpy as np
import pytest

from warpalign.config import (
    BIN_MIN,
    EVAL_BINS,
    EVAL_PAIRS,
    EVAL_START_BIN,
    EXCLUDED_PAIRS,
    N_BINS,
    batch_name,
    pair_label,
)
from warpalign.metrics import (
    holm,
    mean_sd,
    pearson,
    pearson_rows,
    per_a_mean,
    study_silhouette,
    tic_r_pairs,
    warp_magnitude,
    within_batch_r,
)


def test_evaluation_window_starts_at_5_min():
    assert EVAL_START_BIN == 22
    assert EVAL_START_BIN * BIN_MIN <= 5.0 < (EVAL_START_BIN + 1) * BIN_MIN
    assert EVAL_BINS.stop == N_BINS


def test_pairs_and_labels():
    assert len(EVAL_PAIRS) == 4 and len(EXCLUDED_PAIRS) == 2
    assert pair_label('batch_sep2015', 'batch_jul2016') == 'Sep15×Jul16'
    assert pair_label('x', 'y') == 'x × y'
    assert batch_name('batch_jul2016') == 'Jul 2016' and batch_name('other') == 'other'


def test_pearson_rows_matches_corrcoef_on_window_i_major(rng):
    a, b = rng.normal(size=(3, N_BINS)), rng.normal(size=(4, N_BINS))
    r = pearson_rows(a, b)
    for i in range(3):
        for j in range(4):
            ref = np.corrcoef(a[i, EVAL_BINS], b[j, EVAL_BINS])[0, 1]
            assert r[i * 4 + j] == pytest.approx(ref)
    assert pearson_rows(a, b, bins=None)[5] == pytest.approx(np.corrcoef(a[1], b[1])[0, 1])


def test_window_excludes_shared_empty_region(rng):
    """Shared zeros before the window would inflate r; inside the window they are excluded."""
    a, b = rng.normal(size=N_BINS), rng.normal(size=N_BINS)
    a[:EVAL_START_BIN] = b[:EVAL_START_BIN] = -50
    assert abs(pearson(a, b)) < 0.3 < pearson(a, b, bins=None)


def test_pearson_constant_row_is_nan(rng):
    r = pearson_rows(np.ones((1, N_BINS)), rng.normal(size=(2, N_BINS)))
    assert np.all(np.isnan(r))


def test_pearson_is_bounded(rng):
    a = rng.normal(size=(2, N_BINS))
    r = pearson_rows(a, a)
    # r(a, a) is 1 up to rounding: exactly 1.0 with Accelerate (macOS), 1 − 2⁻⁵² with OpenBLAS (CI)
    assert np.all(np.abs(r) <= 1) and r[0] == pytest.approx(1.0, abs=1e-12)


def test_tic_r_pairs_sums_mz(rng):
    a, b = rng.uniform(size=(2, N_BINS, 5)), rng.uniform(size=(3, N_BINS, 5))
    assert np.allclose(tic_r_pairs(a, b), pearson_rows(a.sum(-1), b.sum(-1)))


def test_within_batch_r(rng):
    x = rng.normal(size=(4, N_BINS))
    m, s, k = within_batch_r(x)
    full = pearson_rows(x, x).reshape(4, 4)
    off = full[np.triu_indices(4, 1)]
    assert k == 6 and m == pytest.approx(off.mean()) and s == pytest.approx(off.std())
    with pytest.raises(ValueError):
        within_batch_r(x[:1])


def test_warp_magnitude_and_per_a_mean():
    assert warp_magnitude(np.arange(N_BINS) + 2.0) == pytest.approx(2 * BIN_MIN)
    assert warp_magnitude(np.tile(np.arange(N_BINS) - 1.0, (3, 1))) == pytest.approx(BIN_MIN)
    assert per_a_mean(np.arange(6.0), 2, 3).tolist() == [1.0, 4.0]


def test_holm_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.all(holm([0.5, 0.9]) <= 1.0)


@pytest.mark.parametrize('seed', range(10))
def test_holm_properties(seed):
    p = np.random.default_rng(seed).uniform(size=8)
    adj = holm(p)
    order = np.argsort(p)
    assert np.all(adj >= p) and np.all(adj <= 1)
    assert np.all(np.diff(adj[order]) >= -1e-15)  # monotone in p


def test_mean_sd():
    assert mean_sd([1.0, 3.0]) == (2.0, pytest.approx(np.sqrt(2)))
    m, s = mean_sd([4.0])
    assert m == 4.0 and np.isnan(s)


def test_study_silhouette_separates_distinct_batches(rng):
    a = rng.uniform(size=(6, 20, 10))
    b = rng.uniform(size=(6, 20, 10)) + 2.0
    assert study_silhouette(a, b) > 0.5
    assert abs(study_silhouette(a[:3], a[3:])) < 0.3


def test_paired_wilcoxon_identical_and_different(rng):
    from warpalign.metrics import paired_wilcoxon

    x = rng.normal(size=20)
    assert paired_wilcoxon(x, x) == (0.0, 1.0)
    stat, p = paired_wilcoxon(x + 1.0, x)
    assert p < 1e-4 and stat == 0.0


def test_smooth_rows_sigma_zero_is_identity_and_smoothing_keeps_the_sum(rng):
    from warpalign.metrics import smooth_rows

    x = rng.random((3, 200))
    assert np.array_equal(smooth_rows(x, 0), x)
    x[:, :20] = 0.0
    y = smooth_rows(x, 2.0)
    assert np.allclose(y[:, :10], 0.0)  # an empty run start stays empty
    assert np.allclose(y[:, 30:170].sum(1), x[:, 30:170].sum(1), rtol=0.05)


def test_smoothing_both_signals_removes_sub_bin_misregistration():
    """One-bin peaks offset by half a bin correlate poorly unsmoothed, well after smoothing."""
    from warpalign.metrics import pearson_rows_smoothed
    from warpalign.warps import warp_rows

    x = np.zeros((1, 200))
    x[0, 40:190:15] = 1.0
    shifted = warp_rows(x, np.arange(200) - 0.5)
    raw = pearson_rows(shifted, x)[0]
    sm = pearson_rows_smoothed(shifted, x, (1.0, 2.0))
    assert raw < 0.75 and sm[1.0][0] > raw + 0.2 and sm[2.0][0] > sm[1.0][0]


def test_pearson_paired_matches_pearson_rows_and_masks(rng):
    from warpalign.metrics import all_pairs, pearson_paired

    a, b = rng.normal(size=(3, 200)), rng.normal(size=(4, 200))
    xa, xb = all_pairs(a, b)
    assert np.allclose(pearson_paired(xa, xb), pearson_rows(a, b))
    m = np.ones(xa.shape, bool)
    m[:, 150:] = False
    assert np.allclose(pearson_paired(xa, xb, m), pearson_rows(a[:, :150], b[:, :150]))
    m[0, 25:] = False  # 3 bins (22–24) left: defined; 2 bins: NaN
    assert np.isfinite(pearson_paired(xa, xb, m)[0])
    m[0, 24] = False
    assert np.isnan(pearson_paired(xa, xb, m)[0])


def test_tic_metrics_non_empty_bins_drop_shared_zeros(rng):
    """Bins empty (raw TIC 0) in a sample are left out of the non-empty metrics only."""
    from warpalign.evaluation import METRICS, tic_metrics

    a, b = rng.random((2, 200)) + 0.1, rng.random((3, 200)) + 0.1
    a[0, 100:140] = 0.0
    ones_a, ones_b = np.ones(a.shape, bool), np.ones(b.shape, bool)
    met, outside = tic_metrics(a, b, a, b, ones_a, ones_b)
    assert set(met) == set(METRICS.values()) and outside == 0.0
    keep = np.r_[0:100, 140:200]
    assert np.allclose(met['_nonempty'][:3], pearson_rows(a[:1, keep], b[:, keep]))
    assert np.allclose(met['_nonempty'][3:], met[''][3:])
    assert np.allclose(met['_valid'], met[''])
