# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`warpalign`: cross-batch GC-MS retention-time alignment. A transformer (`ChromaWarpTransformer`),
pretrained on synthetic chromatograms, predicts one monotone warp per sample and is calibrated at
test time on spectrally matched anchors. The repo holds the method, the baselines (COW-TIC, icoshift,
and constant shift, linear warp and PCHIP through the same anchors) and every experiment and
figure of a paper in preparation.

Since warpalign 3.0 the generic core lives in alignbench (github.com/woodRock/align-bench, pinned
in pyproject.toml): `src/warpalign/_compat.py` re-exports it under warpalign's names for the
pipeline (no warnings); `anchors.py`, `warps.py`, `metrics.py`, `baselines.py` and `data.py` are
deprecated re-exports for outside code. Fix shared code in alignbench, not here.
`alignbench_plugin.py` registers the WarpTransformer as the alignbench method `warp-transformer`.

## Commands

Always call `python3.11` explicitly (`python3` is a zsh alias that is not expanded under `nohup`).

```bash
python3.11 -m pip install -e ".[dev]"            # editable install + pytest, coverage, ruff, mypy
python3.11 -m pytest -m "not slow"               # fast suite, ~10 s, synthetic data
python3.11 -m pytest -m slow                     # end-to-end runs of every step (minutes)
python3.11 -m pytest tests/test_anchors.py::test_name   # one test
python3.11 -m coverage run -m pytest && python3.11 -m coverage report
ruff check . && ruff format .                    # line length 100, quote style preserved
mypy                                             # checks src/warpalign
```

Markers: `slow` (end-to-end, minutes) and `real` (needs the private data under `data/fish_oil`;
skipped otherwise).

## Architecture

Three layers:
- `src/warpalign/*.py`: the library (model, calibration, evaluation, config; anchors, warps,
  baselines, metrics and data are alignbench's, via `_compat`). No I/O beyond loading batches;
  nothing here writes reports.
- `src/warpalign/steps/*.py`: one module per pipeline step, each with `compute` → `format_report` →
  `save` → `run`, plus `cli()`. Shared table/report/argument helpers are in `steps/_common.py`.
- `scripts/NN_*.py`: three-line entry points that call a step's `cli()`, numbered 01–14 in run
  order with no gaps; each step depends only on lower-numbered ones. `scripts/archive/` is unused
  legacy code (excluded from ruff).

All constants (time axis, evaluation window, batch pairs, anchor and calibration settings) and the
`Paths` dataclass live in `config.py`; there are no environment variables. Key facts that span files:
- **Time axis.** Step 03 bins each run by retention time onto a fixed 0–45 min grid of 200 bins
  (0.225 min). Sep 2015/Jan 2016 runs start acquiring at 5.0 min, Apr/Jul 2016 at 3.0 min, so
  metrics are computed on `EVAL_BINS` (bins 22–199, 5–45 min). Each sample's
  `acquisition_start` drives the coverage metric ("outside" fraction, "r in range").
- **Two TIC metrics.** `tic` (summed normalised spectra) is the primary metric; `tic_raw` is a
  robustness check, not an ablation. Raw TIC r of apex-scan bins drops under any sub-bin
  resampling, so every method is also scored after smoothing both batches (σ = 1, 2 bins). The
  full per-pair metric set is `evaluation.METRICS`; saved arrays are `<method key><suffix>`.
- **Empty windows.** Sep 2015 (15%) and Jan 2016 (9%) runs have scans with zero readings
  (mostly 22–40 min), binned to exact zeros; `r non-empty` excludes them.
- **Warp frames.** The WarpTransformer warps both batches into a shared frame; PCHIP, COW-TIC and
  icoshift warp batch A onto B. Baselines return source positions (`TicAlignment.src`).
- **Warp parameterisation** (`warp_transformer.make_monotone_warp`): normalised cumulative softplus
  shape + start offset + end shift (in bins) from a zero-initialised `ends_head`. A fully
  free-end parameterisation did not learn in pretraining; don't revert to it.
- **Calibration objective** (`calibration.calibration_objective`, shared by `calibrate` and the
  `free_warp` control): exact anchor loss + 0.05·smoothness + 0.1·`frame_loss`. The frame term
  pins the shared frame's free translation.
- **Anchors** (`anchors.extract_anchors`): raw-TIC peaks → spectral cosine + Hungarian → ±3-bin
  refinement → RANSAC → longest increasing subsequence, pooled over 3 batch-A samples, then
  `merge_close` → `drift_outlier_filter` (leave-one-out trend line) → `slope_filter`. Peaks after
  ~37 min are column bleed (siloxane) and are correctly rejected.
- Checkpoints are `checkpoints/warp_transformer_v2.pt`; `load_model` rejects v1 checkpoints
  (state-dict keys differ).

## Rules for this repo

- **`data/` is private and must never be committed or pushed** (the GitHub repo is public). The
  code is MIT.
- **Do not commit, tag or push results**; the user commits them.
- **Paper runs use CPU** (`--device cpu`, the default): CPU is bit-reproducible under a seed, MPS is
  not. MPS is fine only for throwaway runs.
- **Never edit `src/warpalign/` while a pipeline run is in progress.** The editable install means
  later steps would import the changed code. Check `logs/run_v2/*_progress.log` first.
- Raw fish-oil exports (for steps 03/04, `--raw-dir`): use
  `~/Desktop/fish-data-analysis/engr489/src/data/raw-subsetted-CVSs/`; the other copy lacks TAR3
  files. Batches: Sep 2015 (36), Jan 2016 (26), Apr 2016 (10), Jul 2016 (16).
- The full paper run (four pairs × five seeds, ablations, figures) is in `run.md`; logs and stage
  scripts go in `logs/run_v2/`. v1 results for comparison are in `results/v1_before_fixes/`.
- **Figures.** All pair-specific figures (fig1–3) use Jan16×Jul16, seed 0, median sample
  (`14_generate_figures.py --study-a fish_oil/batch_jan2016 --study-b fish_oil/batch_jul2016
  --select median --seed 0`); the step's default pair is still Sep15×Jul16, so pass the pair.
  Copy them from `results/` to `figs/` under the manuscript names (`fig1_warp_functions.pdf`, `fig2_tic_overlays.pdf`,
  `fig3_calibration_curve.pdf`, `fig4_per_pair_distributions.pdf`, `fig5_silhouette_scatter.pdf`).
  Repo→paper numbering: fig4 = Fig 2, fig2 = Fig 3, fig1 = Fig 4, fig3 = Fig 5.
- The paper is written separately (Claude web client) from a handover message; it must state what
  changed, the numbers, and which `results/` file each number comes from.
- The README must not name the target venue; it keeps only the software citation (`CITATION.cff`).
