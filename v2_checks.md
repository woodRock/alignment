# Checks before rewriting the paper on the v2 results

Repo `woodRock/alignment` at 5b75419. Use the existing package and step scripts; add to them rather
than writing one-off code. CPU, seeds 0–4 where a WarpTransformer is involved. Back up `results/`
to `results/v2_before_checks/` first. Leave git to the user; do not commit `data/`.

The paper will be rewritten evaluation-first ("TIC correlation rewards wrong alignments;
anchor-calibrated warps fix it"). Three things in the v2 handover need resolving before any
number goes into the text.

---

## 1. Raw TIC r is very sensitive to sub-bin resampling — HIGH

Evidence in the v2 files:
- Apr16×Jul16 (no real drift): raw r falls from 0.858 to 0.356 (PCHIP, mean warp **0.08 min**,
  about a third of a bin) and 0.368 (WT, 0.13 min). Normalised r barely moves (0.898 → 0.84).
- Same pair, COW-TIC: a warp of **0.009 min** lowers raw r from 0.858 to 0.798.
- icoshift (integer shifts, no interpolation) leaves it at 0.858.

Likely cause: after apex-scan binning the raw TIC is dominated by a few very tall peaks only one
bin wide. Linear interpolation at a fractional position averages a tall bin with its neighbour,
cutting the peak and spreading it, so raw r collapses even when nothing is misaligned. The
methods resample differently (PCHIP resamples A only, WT both batches, COW per pair, icoshift
not at all), so raw-TIC comparisons between methods are confounded by resampling, not just by
alignment. The handover's raw-TIC claims (WT beats COW-TIC by +0.09 to +0.38; COW's "large drop"
shows it optimises the normalised TIC) cannot be used until this is controlled.

Do:
1. Quantify it: on every pair, shift the **unaligned** batch A by 0.25, 0.5 and 0.75 bin (pure
   linear interpolation, no alignment) and report raw and normalised TIC r. Also shift both
   batches by 0.5 bin.
2. Make the raw-TIC metric robust: after each method's warp, smooth **both** batches' raw TICs
   with the same kernel before correlating (Gaussian, σ = 1 bin; also report σ = 2 bins), so
   every method is compared after the same smoothing. Keep the current unsmoothed raw r in the
   files, but report the smoothed version alongside in `seed_summary.txt` and the per-pair
   files, for all methods including the controls.
3. Repeat the check for the normalised TIC (smaller effect expected) and say how much it moves.
4. Report: do the raw-TIC rankings (WT vs PCHIP vs COW-TIC) survive smoothing? Does the
   Apr16×Jul16 collapse disappear?

## 2. Simple baselines the paper is missing — HIGH

The held-out test shows a **constant shift** (median anchor drift) places held-out compounds as
well as PCHIP, the free warp and the WarpTransformer (0.23–0.41 min mean). A chemist's first
question will be: how does a constant shift do on TIC r? Add two anchor-based baselines through
the **same pooled anchors**, applied like PCHIP (one warp to every A sample, A → B frame):
- **Constant shift:** A shifted by the median anchor drift (fractional bins, linear interpolation).
- **Linear warp:** least-squares straight line q = a·r + b through the anchors.

Report them everywhere PCHIP appears: normalised and raw (and smoothed raw) TIC r, coverage
(outside, r in range), silhouette, warp magnitude, the per-sample Wilcoxon tests (add to step
12's family, so it becomes 7 methods and 21 comparisons per pair; state the family), the
ceiling share, and the two control pairs. If the constant shift is within ~0.02 of the
WarpTransformer on TIC r, say so plainly in the handover.

## 3. Empty time windows — MEDIUM (check the data before trusting the TICs)

The handover says 16% of Sep 2015 and 9% of Jan 2016 windows contain zero ions. With 0.5 s
scans each 0.225-min window holds ~27 scans, so a window with no ions at all is unexpected
unless the export dropped or thresholded scans (the raw folder is `raw-subsetted-CVSs`).
Empty windows become exact zeros in both TICs and zero vectors in the normalised chromatogram,
which affects TIC r, peak detection and the coverage metric.

Do:
1. For every run, count windows inside its acquired range (start → 44.99 min) whose raw TIC is
   zero; report per batch: fraction, and where they fall (early / mid / late; contiguous runs or
   scattered). Add this to `scan_time_audit.txt`.
2. Open one Sep 2015 raw file and look at the scans inside one empty window: are the scans
   present with no peaks (thresholded export), or missing altogether? Quote what you find.
3. `data.acquisition_start` takes the first non-zero bin as the run's start. Compare it with
   the audit's start bin (13 for Apr/Jul, 22 for Sep/Jan) for every sample and report any
   mismatch. If any differ, use the audit's start (from the scan times) for the coverage metric.
4. If empty windows are common inside the evaluation window, also report TIC r with bins that
   are empty in either sample excluded, for every method, so the paper can say whether they
   matter.

## 4. Small

- PCHIP reads 8–10% of values outside the acquired range and the WarpTransformer 0.7–1.6%; with
  those masked, the two are close on three pairs. In the per-sample tests and the main margins,
  also report the masked ("r in range") comparison for WT vs PCHIP, so the paper can state
  the margin both ways.
- In the handover's mechanism table, the batch-shared WT is slightly better than the
  per-sample WT on every pair. Confirm this is not just the batch averaging smoothing the
  warps (report each variant's smoothness loss alongside its TIC r).

---

## Report back (a short handover)

1. Section 1: the shift table, and the main raw-TIC table with smoothing (σ = 1 and 2), with
   rankings and the seeds > 0 for WT − COW-TIC and WT − PCHIP. Your conclusion on whether raw
   TIC r should stay in the paper and in which form.
2. Section 2: the main table with the two new baselines, their significance against WT, and
   their held-out errors (the held-out step already has the constant shift; add the linear
   warp).
3. Section 3: the empty-window audit, what the raw file shows, and any change to coverage or
   TIC r.
4. Anything that changes the claims in section 3 of the v2 handover.
