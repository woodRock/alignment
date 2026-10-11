"""Checks on the real (time-binned) fish-oil data. Skipped when data/fish_oil is absent."""

import numpy as np
import pytest
from gcmsalign.anchors import detect_peaks, extract_anchors
from gcmsalign.config import BIN_MIN, EVAL_PAIRS, EVAL_START_BIN, EXCLUDED_PAIRS, Paths

PATHS = Paths()
FISH_OIL_DIR, RESULTS_DIR = PATHS.fish_oil, PATHS.results

pytestmark = [
    pytest.mark.real,
    pytest.mark.skipif(
        not (FISH_OIL_DIR / 'batch_sep2015' / 'chroma').exists(), reason='real data not available'
    ),
]


@pytest.fixture(scope='module')
def batches():
    from gcmsalign.data import load_batch

    return {
        n: load_batch(n, FISH_OIL_DIR)
        for n in ('batch_sep2015', 'batch_jan2016', 'batch_apr2016', 'batch_jul2016')
    }


def test_batch_sizes(batches):
    assert {k: len(v) for k, v in batches.items()} == {
        'batch_sep2015': 36,
        'batch_jan2016': 26,
        'batch_apr2016': 10,
        'batch_jul2016': 16,
    }


def test_time_binned_runs_are_empty_before_their_start_only(batches):
    for name, start in (
        ('batch_sep2015', 22),
        ('batch_jan2016', 22),
        ('batch_apr2016', 13),
        ('batch_jul2016', 13),
    ):
        raw = batches[name].tic_raw
        assert np.all(raw[:, :start] == 0), name
        assert np.all(raw[:, start] > 0), name
        # The 5.0-min (Sep 2015, Jan 2016) exports contain windows in which every scan recorded
        # zero ions (16 % and 9 % of windows; the same in the v1 index-binned data, so a
        # property of the raw files). The 3.0-min exports have none.
        zero = np.mean(raw[:, start:] == 0)
        assert zero < 0.2 if start == 22 else zero == 0, (name, zero)
    assert EVAL_START_BIN == 22


def test_peaks_sit_on_raw_tic_maxima_and_plot(batches):
    """Section C, test 5: the chosen peaks are local maxima of the raw TIC; a PNG per sample is
    saved to results/checks/ for inspection."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = RESULTS_DIR / 'checks'
    out.mkdir(parents=True, exist_ok=True)
    for name, i in (('batch_sep2015', 0), ('batch_jan2016', 0), ('batch_jul2016', 0)):
        raw = batches[name].tic_raw[i].astype(float)
        pks = detect_peaks(raw)
        assert len(pks) >= 10
        assert np.all(raw[pks] > raw[pks - 1]) and np.all(
            raw[pks] >= raw[np.minimum(pks + 1, len(raw) - 1)]
        )
        assert np.all(raw[pks] >= np.percentile(raw, 80))
        fig, ax = plt.subplots(figsize=(10, 3))
        t = np.arange(len(raw)) * BIN_MIN
        ax.plot(t, raw, lw=0.8, color='#333333')
        ax.plot(t[pks], raw[pks], 'o', color='#CC3311', ms=4, label=f'{len(pks)} detected peaks')
        ax.set_xlabel('RT (min)')
        ax.set_ylabel('raw TIC')
        ax.set_title(f'{name} sample {i} ({batches[name].sample_ids[i]})')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'peaks_{name}_{i}.png', dpi=120)
        plt.close(fig)


@pytest.mark.parametrize('pair', EVAL_PAIRS + EXCLUDED_PAIRS)
def test_anchors_are_monotone_plausible_and_consistent(batches, pair):
    a, b = batches[pair[0]], batches[pair[1]]
    res = extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw)
    assert len(res.anchors) >= 4
    r, q = np.array(res.anchors, dtype=float).T
    slope = np.diff(q) / np.diff(r)
    assert np.all((slope >= 0.5) & (slope <= 2.0))
    drift = (q - r) * BIN_MIN
    if pair in EVAL_PAIRS:  # true drift across Jan–Apr 2016: about −4 min
        assert np.all((drift < -3.0) & (drift > -5.5))
    else:
        assert np.all(np.abs(drift) <= 0.5)
    stages = list(res.stages.values())
    assert stages[:4] == sorted(stages[:4], reverse=True) and stages[4:] == sorted(
        stages[4:], reverse=True
    )
