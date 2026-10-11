"""anchors.py: peaks, order filters, matching and pooling."""

import itertools

import numpy as np
import pytest
import synth
from gcmsalign.anchors import (
    MATCH_STAGES,
    POOL_STAGES,
    detect_peaks,
    drift_outlier_filter,
    merge_close,
    extract_anchors,
    fingerprint,
    longest_increasing_subsequence,
    match_anchors,
    monotone_subset,
    pool_anchors,
    slope_filter,
)

# ── detect_peaks ──────────────────────────────────────────────────────────────


def test_detect_peaks_finds_gaussian_apexes():
    t = np.arange(200)
    tic = sum(h * np.exp(-0.5 * ((t - p) / 1.0) ** 2) for p, h in [(30, 5), (80, 8), (150, 6)])
    assert detect_peaks(tic, height_pct=50).tolist() == [30, 80, 150]


def test_detect_peaks_keeps_the_tallest_n_and_respects_distance():
    t = np.arange(200)
    pos = np.arange(10, 190, 6)
    tic = sum((i + 1) * np.exp(-0.5 * ((t - p) / 0.8) ** 2) for i, p in enumerate(pos))
    pks = detect_peaks(tic, height_pct=0, min_dist=5, n_peaks=4)
    assert pks.tolist() == sorted(pos[-4:].tolist())
    assert np.all(np.diff(detect_peaks(tic, height_pct=0, min_dist=5)) >= 5)


def test_detect_peaks_flat_signal_has_none():
    assert len(detect_peaks(np.ones(200))) == 0


def test_fingerprint_is_unit_norm_and_clipped_at_edges(rng):
    c = rng.uniform(size=(200, 20))
    for pk in (0, 100, 199):
        assert np.isclose(np.linalg.norm(fingerprint(c, pk)), 1.0)
    assert np.allclose(fingerprint(c, 0), c[0:2].mean(0) / np.linalg.norm(c[0:2].mean(0)))
    assert np.all(fingerprint(np.zeros((200, 4)), 5) == 0)


# ── longest increasing subsequence and order filters ──────────────────────────


def _brute_lis_len(v):
    best = 0
    for k in range(len(v), 0, -1):
        for idx in itertools.combinations(range(len(v)), k):
            if all(v[idx[i]] < v[idx[i + 1]] for i in range(k - 1)):
                return k
    return best


@pytest.mark.parametrize('seed', range(40))
def test_lis_is_increasing_and_maximal(seed):
    r = np.random.default_rng(seed)
    v = r.integers(0, 12, size=r.integers(0, 10)).tolist()
    idx = longest_increasing_subsequence(v)
    assert idx == sorted(idx)
    assert all(v[a] < v[b] for a, b in zip(idx, idx[1:]))
    assert len(idx) == _brute_lis_len(v)


def test_monotone_subset_survives_an_early_mismatch():
    """8e: v1's greedy filter kept (10, 150) and then dropped every later anchor."""
    anchors = [(10, 150), (40, 32), (80, 71), (120, 111), (160, 152)]
    assert monotone_subset(anchors) == anchors[1:]


def test_monotone_subset_one_q_per_r():
    out = monotone_subset([(50, 40), (50, 45), (60, 52)])
    assert [r for r, _ in out] == [50, 60]


def test_slope_filter_removes_the_implausible_anchor():
    """8f: the v1 Sep15×Jul16 pattern, a drift jump from −20 to −6 bins over 7 bins."""
    anchors = [(53, 45), (108, 90), (162, 140), (177, 157), (184, 178), (194, 188)]
    kept, removed = slope_filter(anchors)
    assert len(removed) >= 1
    r, q = np.array(kept).T
    s = np.diff(q) / np.diff(r)
    assert np.all((s >= 0.5) & (s <= 2.0))


def test_slope_filter_keeps_smooth_anchors():
    anchors = [(40, 32), (80, 70), (120, 109), (160, 148)]
    assert slope_filter(anchors) == (anchors, [])


def test_slope_filter_handles_tiny_sets():
    assert slope_filter([]) == ([], [])
    assert slope_filter([(5, 3)]) == ([(5, 3)], [])


def test_pool_anchors_median_outlier_and_stages():
    per_sample = [
        [(40, 32), (80, 72), (120, 112)],
        [(40, 33), (80, 72), (120, 111), (150, 60)],  # (150, 60): drift −90
        [(40, 31), (80, 73)],
    ]
    anchors, counts, removed_outlier, removed_slope = pool_anchors(per_sample)
    assert anchors == [(40, 32), (80, 72), (120, 112)]           # median q per r
    assert removed_outlier == [(150, 60)] and removed_slope == []
    assert list(counts) == list(POOL_STAGES)
    assert counts == {'pooled': 4, 'after merging': 4, 'after outlier filter': 3,
                      'after slope filter': 3}


def test_pool_anchors_identical_drifts_are_kept():
    """The s.d. floor of 1 bin keeps a set with zero spread."""
    anchors, *_ = pool_anchors([[(40, 30), (80, 70), (120, 110)]])
    assert anchors == [(40, 30), (80, 70), (120, 110)]


def test_pool_anchors_empty():
    assert pool_anchors([[], []]) == (
        [],
        {'pooled': 0, 'after merging': 0, 'after outlier filter': 0, 'after slope filter': 0},
        [],
        [],
    )


# ── merge_close (round-3 review, issue 4) ─────────────────────────────────────

def test_merge_close_combines_neighbours_one_or_two_bins_apart():
    anchors = [(40, 30), (63, 46), (64, 46), (100, 80), (161, 139), (162, 140), (164, 141)]
    # (164, 141) is 3 bins from its group's first anchor (161, 139): kept separate
    assert merge_close(anchors) == [(40, 30), (64, 46), (100, 80), (162, 140), (164, 141)]


def test_merge_close_leaves_separated_anchors_and_sorts():
    anchors = [(100, 80), (40, 30), (70, 55)]
    assert merge_close(anchors) == [(40, 30), (70, 55), (100, 80)]
    assert merge_close([]) == []
    assert merge_close([(10, 5), (13, 8)], max_gap=2) == [(10, 5), (13, 8)]
    assert merge_close([(10, 5), (13, 8)], max_gap=3) == [(12, 7)]   # 11.5, 6.5 rounded up


def test_merge_close_does_not_average_a_wrong_match_into_a_right_one():
    """Round-4 review: the same compound matched correctly, (161, 139), and wrongly,
    (162, 150), must stay two anchors (drift −22 vs −12) for the filters to judge; averaging
    gave an invented (162, 144) 5 bins from both."""
    assert merge_close([(161, 139), (162, 150)]) == [(161, 139), (162, 150)]
    kept, *_ = pool_anchors([[(40, 32), (80, 72), (120, 112), (161, 153)],
                             [(40, 32), (80, 72), (120, 112), (162, 120)]])
    assert (162, 120) not in kept and all(q - r == -8 for r, q in kept)


def test_merge_close_does_not_chain_neighbouring_compounds():
    """Round-4 review: anchors 2, 4, 6 and 8 bins apart used to collapse into one; groups are
    now measured from their first anchor."""
    run = [(100, 90), (102, 92), (104, 94), (106, 96), (108, 98)]
    merged = merge_close(run)
    assert len(merged) == 3
    assert all(abs(b[0] - a[0]) > 2 for a, b in zip(merged, merged[1:], strict=False))


def test_merge_close_rounds_half_up():
    assert merge_close([(100, 90), (101, 91)]) == [(101, 91)]     # 100.5 → 101 (not 100)
    assert merge_close([(101, 91), (102, 92)]) == [(102, 92)]     # 101.5 → 102


def test_slope_filter_no_longer_sees_rounding_slopes():
    """(63, 46) and (64, 46) are one compound; unmerged, their slope of 0 trips the filter."""
    pair = [(48, 31), (63, 46), (64, 46), (86, 67)]
    assert slope_filter(pair)[1] != []
    assert slope_filter(merge_close(pair)) == (merge_close(pair), [])


# ── drift_outlier_filter ──────────────────────────────────────────────────────

# Real Sep15×Jul16 pattern (bins): drift trends from −15 at r = 48 to −22 at r = 165
TRENDING = [(48, 33), (64, 46), (69, 51), (75, 57), (86, 67), (118, 98), (165, 143)]


def test_trending_drift_keeps_both_end_anchors():
    """A smooth drift trend is not an outlier: the end anchors stay (a constant-mean rule
    removed (48, 33) and (165, 143) here)."""
    kept, removed = drift_outlier_filter(TRENDING)
    assert kept == TRENDING and removed == []
    d = np.array([q - r for r, q in TRENDING], float)
    sd = max(d.std(), 1.0)
    assert any(abs(x - d.mean()) > 1.5 * sd for x in d)       # the old rule would drop some


def test_anchor_far_from_the_trend_is_removed():
    bad = (100, 60)                                           # drift −40 among −15 … −22
    kept, removed = drift_outlier_filter([*TRENDING, bad])
    assert removed == [bad] and kept == TRENDING


def test_robust_to_a_gross_outlier_at_the_end():
    """A high-leverage outlier at the end of the range would pull a fitted line onto itself;
    judging each anchor against a line fitted to the others prevents that."""
    anchors = [(40, 32), (60, 52), (80, 72), (100, 92), (120, 112), (190, 130)]
    kept, removed = drift_outlier_filter(anchors)
    assert removed == [(190, 130)] and kept == anchors[:5]


def test_drift_outlier_filter_small_and_unsorted_inputs():
    assert drift_outlier_filter([]) == ([], [])
    assert drift_outlier_filter([(80, 10), (40, 30)]) == ([(40, 30), (80, 10)], [])
    three = [(40, 32), (80, 72), (120, 10)]                  # < 4: nothing to judge against
    assert drift_outlier_filter(three) == (three, [])
    kept, _ = drift_outlier_filter(list(reversed(TRENDING)))
    assert kept == TRENDING


def test_scatter_within_the_bin_precision_is_kept():
    """Scatter of ±1 bin around the trend (0.225-min binning) stays under the 3-bin floor."""
    jitter = [(40, 30), (60, 51), (80, 69), (100, 90), (120, 111), (140, 129)]
    assert drift_outlier_filter(jitter) == (jitter, [])
    assert len(drift_outlier_filter(jitter, n_sd=0.1, floor=0.5)[1]) > 0


def test_removes_several_outliers_one_at_a_time():
    bad = [(70, 20), (130, 160)]
    kept, removed = drift_outlier_filter([*TRENDING, *bad])
    assert removed == sorted(bad) and kept == TRENDING


# ── match_anchors / extract_anchors ───────────────────────────────────────────


def test_sample_against_itself_gives_r_equal_q(synthetic_pair):
    """Section C, test 2: matching a sample with itself must return anchors with r == q (v1
    shifted every anchor bin ≥ 100 by one bin)."""
    (ca, ra), _ = synthetic_pair
    anchors, counts = match_anchors(ca[0], ca[0], ra[0], ra[0])
    assert len(anchors) >= 5
    assert all(r == q for r, q in anchors)
    assert list(counts) == list(MATCH_STAGES)


def test_recovers_a_known_shift(synthetic_pair):
    (ca, ra), (cb, rb) = synthetic_pair
    anchors, counts = match_anchors(ca[0], cb[0], ra[0], rb[0])
    assert len(anchors) >= 5
    assert all(abs((q - r) - synth.SHIFT) <= 1 for r, q in anchors)
    vals = list(counts.values())
    assert vals == sorted(vals, reverse=True)  # each stage only removes


def test_peaks_come_from_the_raw_tic(synthetic_pair):
    """Fix 1: with a raw TIC whose only peaks are at 60 and 140, anchors can only sit there."""
    (ca, _), (cb, _) = synthetic_pair
    t = np.arange(200)
    tic = 10 * np.exp(-0.5 * ((t - 60) / 1.0) ** 2) + 10 * np.exp(-0.5 * ((t - 140) / 1.0) ** 2)
    anchors, _ = match_anchors(ca[0], ca[0], tic, tic)
    assert {r for r, _ in anchors} <= {59, 60, 61, 139, 140, 141}


def test_extract_anchors_pools_and_is_monotone(synthetic_pair):
    (ca, ra), (cb, rb) = synthetic_pair
    res = extract_anchors(ca, cb, ra, rb)
    assert len(res.anchors) >= 5
    r, q = np.array(res.anchors).T
    assert np.all(np.diff(r) > 0) and np.all(np.diff(q) > 0)
    assert np.all(np.abs((q - r) - synth.SHIFT) <= 1)
    assert list(res.stages) == list(MATCH_STAGES) + list(POOL_STAGES)


def test_extract_anchors_too_few_peaks():
    flat = np.zeros((2, 200, 30), dtype=np.float32)
    res = extract_anchors(flat, flat, np.ones((2, 200)), np.ones((2, 200)))
    assert res.anchors == []


def test_pooled_median_rounds_half_up_like_merging():
    """A median that falls on .5 (two samples, q = 40 and 41) rounds up, as merged anchors do."""
    kept, *_ = pool_anchors([[(50, 40), (90, 80), (130, 120), (170, 160)],
                             [(50, 41), (90, 80), (130, 120), (170, 160)]])
    assert (50, 41) in kept
