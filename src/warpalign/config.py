"""Constants shared by every step: time axis, evaluation window, batch pairs, settings; and the
project's file layout (Paths).

One definition of each, imported everywhere, so a change lands once. Nothing here reads the
environment: the file layout is passed explicitly as a Paths object.

Since warpalign 3.0 the time axis, evaluation window, batch pairs and anchor settings come from
alignbench (FISH_OIL_AXIS, alignbench.datasets.fish_oil, MatchConfig); the names below keep their
2.x values exactly. What stays warpalign's own: the file layout (Paths) and the calibration
settings of the WarpTransformer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alignbench.anchors import DEFAULT_MATCH as _M
from alignbench.axis import FISH_OIL_AXIS
from alignbench.datasets.fish_oil import (  # noqa: F401  (re-exported)
    BATCHES,
    EVAL_PAIRS,
    EXCLUDED_PAIRS,
    PAIR_LABELS,
    batch_name,
    pair_label,
)
from alignbench.metrics.tic import SMOOTH_SIGMAS  # noqa: F401  (re-exported)

# ── Time axis ─────────────────────────────────────────────────────────────────
# 03_preprocess_fish_oil_batches.py bins every run on a fixed 0–45 min grid of 0.225-min windows:
# bin i covers [i·BIN_MIN, (i+1)·BIN_MIN) min. Bin i is placed at i·BIN_MIN minutes everywhere.
N_BINS = FISH_OIL_AXIS.n_bins  # 200
MZ_BINS = 1000
RUN_MIN = FISH_OIL_AXIS.run_min  # 45.0
BIN_MIN = FISH_OIL_AXIS.bin_min  # 0.225 min per bin
TIME_AXIS = FISH_OIL_AXIS.times  # minutes of bin i; spacing exactly BIN_MIN (read-only)
LAST_MIN = FISH_OIL_AXIS.last_min  # 44.775 min, the last bin

# ── Evaluation window ─────────────────────────────────────────────────────────
# Sep 2015 and Jan 2016 runs start at 5.0 min, Apr and Jul 2016 runs at 3.0 min
# (results/scan_time_audit.txt). Bins before 5.0 min are empty in some or all runs, and shared
# zeros inflate Pearson r, so every TIC metric is computed on the common acquisition window.
COMMON_START_MIN = FISH_OIL_AXIS.eval_start_min  # 5.0
EVAL_START_BIN = FISH_OIL_AXIS.eval_start_bin  # 22: first bin with data in every run
EVAL_BINS = FISH_OIL_AXIS.eval_bins
# Raw TICs of apex-scan bins are dominated by one-bin-wide peaks, so resampling a signal at
# fractional positions (any warp) lowers its raw TIC r even without misalignment. Every method's
# TICs are therefore also compared after the same Gaussian smoothing of both batches (σ in bins).

# ── Batches and pairs ─────────────────────────────────────────────────────────
# alignbench.datasets.fish_oil: BATCHES, EVAL_PAIRS, EXCLUDED_PAIRS, PAIR_LABELS (re-exported above)

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
# alignbench.anchors.MatchConfig's defaults (the matcher reads those, not these names)
N_REF_PAIRS = _M.n_ref  # batch-A samples matched against B[0]
MERGE_GAP = _M.merge_gap  # pooled anchors ≤ this many bins apart are one compound: merged
OUTLIER_SD = _M.outlier_sd  # drop anchors > OUTLIER_SD · s.d. (≥ RANSAC_RESID) off the trend
SLOPE_RANGE = _M.slope_range  # plausible local Δq/Δr between consecutive pooled anchors
MAX_DRIFT_MIN = _M.max_drift_min  # candidate peak matches further apart are not considered
SIM_THRESHOLD = _M.sim_threshold  # minimum spectral cosine for a match
RANSAC_RESID = _M.ransac_resid  # bins
REFINE_RADIUS = _M.refine_radius  # bins searched around a query peak for the best-matching scan
PEAK_HEIGHT_PCT = _M.peak_height_pct
PEAK_MIN_DIST = _M.peak_min_dist
PEAK_MAX_N = _M.peak_max_n

# ── Test-time calibration ─────────────────────────────────────────────────────
N_CALIB_STEPS = 500
LR_CALIB = 5e-4
LAMBDA_ANCHOR = 1.0
LAMBDA_SMOOTH = 0.05
LAMBDA_FRAME = 0.1  # fixes the consensus frame's free translation (warp_transformer.frame_loss)
FREE_WARP_LR = LR_CALIB * 100
