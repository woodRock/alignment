"""warpalign.config keeps its 2.x names and values (derived from alignbench since 3.0)."""

from __future__ import annotations

from warpalign import config


def test_time_axis_and_window():
    assert (config.N_BINS, config.RUN_MIN, config.BIN_MIN) == (200, 45.0, 0.225)
    assert config.LAST_MIN == 199 * 0.225 and config.TIME_AXIS[-1] == config.LAST_MIN
    assert config.EVAL_START_BIN == 22 and slice(22, 200) == config.EVAL_BINS
    assert config.SMOOTH_SIGMAS == (1.0, 2.0)


def test_batches_pairs_and_anchor_settings():
    assert config.BATCHES == ('batch_sep2015', 'batch_jan2016', 'batch_apr2016', 'batch_jul2016')
    assert len(config.EVAL_PAIRS) == 4 and len(config.EXCLUDED_PAIRS) == 2
    assert config.pair_label('batch_sep2015', 'batch_jul2016') == 'Sep15×Jul16'
    assert config.batch_name('batch_jul2016') == 'Jul 2016'
    assert (config.N_REF_PAIRS, config.MERGE_GAP, config.SLOPE_RANGE) == (3, 2, (0.5, 2.0))
    assert (config.SIM_THRESHOLD, config.RANSAC_RESID, config.MAX_DRIFT_MIN) == (0.30, 3.0, 20.0)
