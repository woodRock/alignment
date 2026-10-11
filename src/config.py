"""Constants shared by every script: time axis, evaluation window, batch pairs, paths, settings.

One definition of each, imported everywhere, so a change lands once.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ── Time axis ─────────────────────────────────────────────────────────────────
# 03_preprocess_fish_oil_batches.py bins every run on a fixed 0–45 min grid of 0.225-min windows:
# bin i covers [i·BIN_MIN, (i+1)·BIN_MIN) min. Bin i is placed at i·BIN_MIN minutes everywhere.
N_BINS    = 200
MZ_BINS   = 1000
RUN_MIN   = 45.0
BIN_MIN   = RUN_MIN / N_BINS                       # 0.225 min per bin
TIME_AXIS = np.arange(N_BINS) * BIN_MIN            # minutes of bin i; spacing exactly BIN_MIN
LAST_MIN  = (N_BINS - 1) * BIN_MIN                 # 44.775 min, the last bin

# ── Evaluation window ─────────────────────────────────────────────────────────
# Sep 2015 and Jan 2016 runs start at 5.0 min, Apr and Jul 2016 runs at 3.0 min
# (results/scan_time_audit.txt). Bins before 5.0 min are empty in some or all runs, and shared
# zeros inflate Pearson r, so every TIC metric is computed on the common acquisition window.
COMMON_START_MIN = 5.0
EVAL_START_BIN   = int(COMMON_START_MIN // BIN_MIN)    # 22: first bin with data in every run
EVAL_BINS        = slice(EVAL_START_BIN, N_BINS)

# ── Batches and pairs ─────────────────────────────────────────────────────────
BATCHES        = ('batch_sep2015', 'batch_jan2016', 'batch_apr2016', 'batch_jul2016')
EVAL_PAIRS     = (('batch_sep2015', 'batch_jul2016'), ('batch_sep2015', 'batch_apr2016'),
                  ('batch_jan2016', 'batch_apr2016'), ('batch_jan2016', 'batch_jul2016'))
EXCLUDED_PAIRS = (('batch_sep2015', 'batch_jan2016'), ('batch_apr2016', 'batch_jul2016'))
PAIR_LABELS    = {('batch_sep2015', 'batch_jul2016'): 'Sep15×Jul16',
                  ('batch_sep2015', 'batch_apr2016'): 'Sep15×Apr16',
                  ('batch_jan2016', 'batch_apr2016'): 'Jan16×Apr16',
                  ('batch_jan2016', 'batch_jul2016'): 'Jan16×Jul16',
                  ('batch_sep2015', 'batch_jan2016'): 'Sep15×Jan16',
                  ('batch_apr2016', 'batch_jul2016'): 'Apr16×Jul16'}

# ── Paths ─────────────────────────────────────────────────────────────────────
# ALIGNMENT_ROOT relocates data/, results/ and checkpoints/ (the tests run the scripts on a
# synthetic dataset in a temporary directory this way).
ROOT         = Path(os.environ.get('ALIGNMENT_ROOT', Path(__file__).resolve().parent.parent))
DATA_DIR     = ROOT / 'data'
FISH_OIL_DIR = DATA_DIR / 'fish_oil'
RESULTS_DIR  = ROOT / 'results'
PER_PAIR_DIR = RESULTS_DIR / 'per_pair'
CKPT_DIR     = ROOT / 'checkpoints'
DEFAULT_CKPT = CKPT_DIR / 'warp_transformer_v2.pt'
PRETRAIN_H5  = DATA_DIR / 'pretraining' / 'spectra.h5'

# ── Anchor extraction ─────────────────────────────────────────────────────────
N_REF_PAIRS      = 3            # batch-A samples matched against B[0]
OUTLIER_SD       = 1.5          # pooled anchors with |drift − mean| > OUTLIER_SD · s.d. are dropped
SLOPE_RANGE      = (0.5, 2.0)   # plausible local Δq/Δr between consecutive pooled anchors
MAX_DRIFT_MIN    = 20.0         # candidate peak matches further apart than this are not considered
SIM_THRESHOLD    = 0.30         # minimum spectral cosine for a match
RANSAC_RESID     = 3.0          # bins
REFINE_RADIUS    = 3            # bins searched around a query peak for the best-matching scan
PEAK_HEIGHT_PCT  = 80
PEAK_MIN_DIST    = 5
PEAK_MAX_N       = 30

# ── Test-time calibration ─────────────────────────────────────────────────────
N_CALIB_STEPS = int(os.environ.get('ALIGNMENT_CALIB_STEPS', 500))   # override only for fast tests
LR_CALIB      = 5e-4
LAMBDA_ANCHOR = 1.0
LAMBDA_SMOOTH = 0.05
FREE_WARP_LR  = LR_CALIB * 100


def pair_label(a: str, b: str) -> str:
    """'batch_sep2015', 'batch_jul2016' → 'Sep15×Jul16' (falls back to the raw names)."""
    return PAIR_LABELS.get((a, b), f'{a} × {b}')


def batch_name(label: str) -> str:
    """'batch_jul2016' → 'Jul 2016'."""
    s = label.replace('batch_', '')
    return f'{s[:3].capitalize()} {s[3:]}' if len(s) == 7 and s[3:].isdigit() else label
