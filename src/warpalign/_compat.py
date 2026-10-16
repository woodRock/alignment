"""warpalign's names for the code that moved to alignbench in warpalign 3.0 (private, no warnings).

The paper pipeline (calibration, the WarpTransformer, the steps) imports from here. The public
modules warpalign.anchors, .warps, .metrics, .baselines and .data re-export the same names with a
DeprecationWarning, for code written against warpalign 2.x. Bit-identity with warpalign 2.0.0 is
alignbench's golden tests (tests/golden/warpalign_ea404e75.*.npz there).
"""

from __future__ import annotations

# ruff: noqa: F401  (this module is the re-export surface)
from alignbench.anchors import (
    DEFAULT_MATCH,
    MATCH_STAGES,
    POOL_STAGES,
    Anchor,
    AnchorResult,
    MatchConfig,
    detect_peaks,
    drift_outlier_filter,
    fingerprint,
    format_stages,
    longest_increasing_subsequence,
    match_pooled_arrays,
    merge_close,
    monotone_subset,
    pool_anchors,
    slope_filter,
)
from alignbench.anchors import match_sample_pair as match_anchors
from alignbench.data import Batch, acquisition_start
from alignbench.datasets.fish_oil import chroma_files, load_batch, load_chromas, load_tic_raw
from alignbench.methods.tic import (
    COW_N_SEGS,
    COW_SLACK,
    ICO_MAX_SHIFT,
    ICO_N_INTERVALS,
    TicAlignment,
    cow_tic_align,
    icoshift_align,
)
from alignbench.methods.tic import TIC_ALIGNERS as METHODS
from alignbench.metrics.diagnostics import study_silhouette, warp_magnitude
from alignbench.metrics.stats import holm, mean_sd, paired_wilcoxon
from alignbench.metrics.tic import (
    all_pairs,
    pearson,
    pearson_paired,
    pearson_rows,
    pearson_rows_masked,
    pearson_rows_smoothed,
    per_a_mean,
    smooth_rows,
    tic_r_pairs,
    within_batch_r,
)
from alignbench.warps import (
    WarpFn,
    anchor_drift,
    anchor_line,
    constant_shift_warp,
    invert_src,
    linear_warp,
    monotone_interp,
    pchip_warp,
    predict_query_bin,
    src_from_fn,
    warp_rows,
    warp_signal,
)


def extract_anchors(
    chromas_a, chromas_b, tic_raw_a, tic_raw_b, n_ref: int = DEFAULT_MATCH.n_ref
) -> AnchorResult:
    """Calibration anchors between two batches: A[0 … n_ref−1] each matched against B[0], then
    pooled (alignbench.anchors.match_pooled_arrays). Inputs are [N, T, M] chromatograms and [N, T]
    raw TICs."""
    cfg = DEFAULT_MATCH if n_ref == DEFAULT_MATCH.n_ref else MatchConfig(n_ref=n_ref)
    return match_pooled_arrays(chromas_a, chromas_b, tic_raw_a, tic_raw_b, cfg)


ANCHORS = [
    'MATCH_STAGES', 'POOL_STAGES', 'Anchor', 'AnchorResult', 'detect_peaks',
    'drift_outlier_filter', 'extract_anchors', 'fingerprint', 'format_stages',
    'longest_increasing_subsequence', 'match_anchors', 'merge_close', 'monotone_subset',
    'pool_anchors', 'slope_filter',
]  # fmt: skip
WARPS = [
    'WarpFn', 'anchor_drift', 'anchor_line', 'constant_shift_warp', 'invert_src', 'linear_warp',
    'monotone_interp', 'pchip_warp', 'predict_query_bin', 'src_from_fn', 'warp_rows',
    'warp_signal',
]  # fmt: skip
METRICS_ = [
    'all_pairs', 'holm', 'mean_sd', 'paired_wilcoxon', 'pearson', 'pearson_paired',
    'pearson_rows', 'pearson_rows_masked', 'pearson_rows_smoothed', 'per_a_mean', 'smooth_rows',
    'study_silhouette', 'tic_r_pairs', 'warp_magnitude', 'within_batch_r',
]  # fmt: skip
BASELINES = [
    'COW_N_SEGS', 'COW_SLACK', 'ICO_MAX_SHIFT', 'ICO_N_INTERVALS', 'METHODS', 'TicAlignment',
    'cow_tic_align', 'icoshift_align',
]  # fmt: skip
DATA = ['Batch', 'acquisition_start', 'chroma_files', 'load_batch', 'load_chromas', 'load_tic_raw']


def deprecated(module: str) -> None:
    """The DeprecationWarning of the public re-export modules."""
    import warnings

    warnings.warn(
        f'warpalign.{module} moved to alignbench in warpalign 3.0 '
        f'(github.com/woodRock/align-bench); import it from there. These re-exports will be '
        f'removed in warpalign 4.',
        DeprecationWarning,
        stacklevel=3,
    )
