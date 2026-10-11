"""Constants shared by every step: time axis, evaluation window, batch pairs, settings; and the
project's file layout (Paths).

One definition of each, imported everywhere, so a change lands once. Nothing here reads the
environment: the file layout is passed explicitly as a Paths object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── Time axis ─────────────────────────────────────────────────────────────────
# 03_preprocess_fish_oil_batches.py bins every run on a fixed 0–45 min grid of 0.225-min windows:
# bin i covers [i·BIN_MIN, (i+1)·BIN_MIN) min. Bin i is placed at i·BIN_MIN minutes everywhere.
N_BINS = 200
MZ_BINS = 1000
RUN_MIN = 45.0
BIN_MIN = RUN_MIN / N_BINS  # 0.225 min per bin
TIME_AXIS = np.arange(N_BINS) * BIN_MIN  # minutes of bin i; spacing exactly BIN_MIN
LAST_MIN = (N_BINS - 1) * BIN_MIN  # 44.775 min, the last bin

# ── Evaluation window ─────────────────────────────────────────────────────────
# Sep 2015 and Jan 2016 runs start at 5.0 min, Apr and Jul 2016 runs at 3.0 min
# (results/scan_time_audit.txt). Bins before 5.0 min are empty in some or all runs, and shared
# zeros inflate Pearson r, so every TIC metric is computed on the common acquisition window.
COMMON_START_MIN = 5.0
EVAL_START_BIN = int(COMMON_START_MIN // BIN_MIN)  # 22: first bin with data in every run
EVAL_BINS = slice(EVAL_START_BIN, N_BINS)

# ── Batches and pairs ─────────────────────────────────────────────────────────
BATCHES = ('batch_sep2015', 'batch_jan2016', 'batch_apr2016', 'batch_jul2016')
EVAL_PAIRS = (
    ('batch_sep2015', 'batch_jul2016'),
    ('batch_sep2015', 'batch_apr2016'),
    ('batch_jan2016', 'batch_apr2016'),
    ('batch_jan2016', 'batch_jul2016'),
)
EXCLUDED_PAIRS = (('batch_sep2015', 'batch_jan2016'), ('batch_apr2016', 'batch_jul2016'))
PAIR_LABELS = {
    ('batch_sep2015', 'batch_jul2016'): 'Sep15×Jul16',
    ('batch_sep2015', 'batch_apr2016'): 'Sep15×Apr16',
    ('batch_jan2016', 'batch_apr2016'): 'Jan16×Apr16',
    ('batch_jan2016', 'batch_jul2016'): 'Jan16×Jul16',
    ('batch_sep2015', 'batch_jan2016'): 'Sep15×Jan16',
    ('batch_apr2016', 'batch_jul2016'): 'Apr16×Jul16',
}

# ── File layout ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Where a run reads data and writes results. Defaults to the repository; the tests point
    it at a temporary directory with a synthetic dataset."""

    root: Path = REPO_ROOT

    @property
    def data(self) -> Path:
        return self.root / 'data'

    @property
    def fish_oil(self) -> Path:
        return self.data / 'fish_oil'

    @property
    def pretrain_h5(self) -> Path:
        return self.data / 'pretraining' / 'spectra.h5'

    @property
    def results(self) -> Path:
        return self.root / 'results'

    @property
    def per_pair(self) -> Path:
        return self.results / 'per_pair'

    @property
    def checkpoints(self) -> Path:
        return self.root / 'checkpoints'

    @property
    def default_ckpt(self) -> Path:
        return self.checkpoints / 'warp_transformer_v2.pt'

    def ensure_results(self) -> None:
        self.per_pair.mkdir(parents=True, exist_ok=True)


# ── Anchor extraction ─────────────────────────────────────────────────────────
N_REF_PAIRS = 3  # batch-A samples matched against B[0]
MERGE_GAP = 2  # pooled anchors ≤ this many bins apart are one compound: merged
OUTLIER_SD = 1.5  # drop pooled anchors > OUTLIER_SD · s.d. (≥ RANSAC_RESID) off the trend
SLOPE_RANGE = (0.5, 2.0)  # plausible local Δq/Δr between consecutive pooled anchors
MAX_DRIFT_MIN = 20.0  # candidate peak matches further apart than this are not considered
SIM_THRESHOLD = 0.30  # minimum spectral cosine for a match
RANSAC_RESID = 3.0  # bins
REFINE_RADIUS = 3  # bins searched around a query peak for the best-matching scan
PEAK_HEIGHT_PCT = 80
PEAK_MIN_DIST = 5
PEAK_MAX_N = 30

# ── Test-time calibration ─────────────────────────────────────────────────────
N_CALIB_STEPS = 500
LR_CALIB = 5e-4
LAMBDA_ANCHOR = 1.0
LAMBDA_SMOOTH = 0.05
LAMBDA_FRAME = 0.1  # fixes the consensus frame's free translation (warp_transformer.frame_loss)
FREE_WARP_LR = LR_CALIB * 100


def pair_label(a: str, b: str) -> str:
    """'batch_sep2015', 'batch_jul2016' → 'Sep15×Jul16' (falls back to the raw names)."""
    return PAIR_LABELS.get((a, b), f'{a} × {b}')


def batch_name(label: str) -> str:
    """'batch_jul2016' → 'Jul 2016'."""
    s = label.replace('batch_', '')
    return f'{s[:3].capitalize()} {s[3:]}' if len(s) == 7 and s[3:].isdigit() else label
