# Runs for a solid WarpTransformer paper (one day on an Apple M2)

The current draft rests on a single calibration run per configuration. This plan fills
the four gaps a reviewer would raise first, using only the existing data and checkpoint:

| Gap in the current draft | What fixes it | Steps |
|---|---|---|
| Single run; repeats differ by up to 0.020 TIC r, the same size as the Sep15×Jul16 margin over COW-TIC | 5 calibration seeds per pair; tables as mean ± s.d.; significance checked at every seed | 1, 3, 5, 6 |
| No compound-level metric (TIC r is whole-chromatogram; silhouette is RT-invariant) | Leave-one-anchor-out RT error at known compounds | 2 |
| The WarpTransformer's anchors are pooled and filtered, PCHIP's are not, so the network's contribution is confounded | New baseline: PCHIP through the same pooled calibration anchors | A1, 1 |
| "Few anchors suffice" is asserted, not tested | Anchor-count ablation, pretrained vs random init | 4 |

Figures are regenerated from the final run at the end (step 7), with Fig. 2 showing a
typical rather than best-case sample and Fig. 5's legends moved off the data.

Out of scope for one day, and stays in Limitations: a second dataset, and drift events
independent of the Jan–Apr 2016 interval.

## 0. Prerequisites

- **Python.** Call `python3.11` explicitly (`python3` is a zsh alias, not expanded under `nohup`).
- **Data.** `data/fish_oil/batch_{sep2015,jan2016,apr2016,jul2016}/chroma/*.npz`, including the
  14 TAR3 samples (Sep 2015 = 36, Jan 2016 = 26, Apr 2016 = 10, Jul 2016 = 16).
- **Checkpoint.** Every command passes `--ckpt checkpoints/warp_transformer_baseline.pt`.
- **Devices.** Two lanes run in parallel all day:
  - **CPU lane** (`--device cpu`): the main seeded runs and the held-out anchor error. CPU
    calibration is bit-reproducible under a fixed seed, so these numbers can be regenerated
    exactly. About 5–8 min per 500-step calibration with the current batch sizes.
  - **GPU lane** (`--device mps`): the pretraining ablation and the anchor-count sweep. MPS is
    about 3× faster but not bit-reproducible; both are reported as means over seeds or
    subsets, so that is acceptable.
- **Overwrites.** Step 1 rewrites `results/per_pair/tic_r_calib_*_seed0.npz`, which Figs. 4–5
  and the significance tests read. Everything downstream must be rerun after step 1, which
  this plan does.

## A. Code changes (morning, before starting the lanes)

**A1. `scripts/16_align_testtime_calib.py`: add a pooled-anchor PCHIP baseline.**
This isolates what the network adds beyond its anchors. Using the calibration anchors
returned by `extract_anchors` (pairs `(r, q)`: `r` = bin in batch A, `q` = bin in batch B),
after the same outlier filtering:
1. Sort by `r`; drop any anchor whose `q` would break monotonicity (as `align_pair` does).
2. Fit `PchipInterpolator(rf, qf)` in minutes, with end points `(0, 0)` and `(45, 45)`, giving
   a warp from batch-A time to batch-B time.
3. Warp **every batch-B sample** into the batch-A frame: `warp_chroma_2d(b, wfn)`.
4. Report, alongside the existing rows: mean TIC r over all A×B pairs (unwarped A vs warped B),
   Δ unaligned, study silhouette (`study_silhouette(chromas_a, warped_b)`), and warp magnitude
   (mean `|f(t) − t|` in minutes).
5. Save the per-pair array to the `.npz` as `pchip_pooled`, plus `pchip_pooled_sil` and
   `pchip_pooled_warp_mag`.

It needs no training and adds seconds per run. Its result does not depend on the seed.

**A2. `scripts/21_significance_tests.py`: include the new baseline.**
Add `'PCHIP (pooled anchors)'` to `METHODS`, loaded from `calib['pchip_pooled']`. Holm
correction is then across 15 comparisons per pair.

**A3. New `scripts/24_aggregate_seeds.py`: one summary file for the paper.**
Reads `results/per_pair/tic_r_baselines_*.npz`, `tic_r_calib_*_seed{0..4}.npz` and
`tic_r_ablate_*_seed{0..4}.npz` for the four evaluated pairs, and writes
`results/seed_summary.txt` with, per pair:
- Mean ± s.d. over seeds of the WarpTransformer's TIC r, Δ unaligned, study silhouette, and
  warp magnitude (batch B, batch A, all samples), plus the min and max over seeds.
- The deterministic rows (unaligned, icoshift, COW-TIC, PCHIP to B[0], PCHIP pairwise, PCHIP
  pooled) with TIC r, Δ, silhouette and warp magnitude, and a check that they are identical
  across seeds.
- Per batch-A sample (r averaged over batch B): the number of samples where the
  WarpTransformer beats COW-TIC and beats pooled PCHIP, at each seed and pooled over seeds.
- Ablation: mean ± s.d. over seeds of TIC r and warp magnitude (all / B) for both arms, the
  pretraining gain Δ (per seed, then mean ± s.d.), and the share of the improvement that
  random init recovers.

**A4. `scripts/20_generate_figures.py`, Fig. 5: move the legends off the data.**
Both legends currently cover points: the method legend covers icoshift's Sep15×Apr16 point,
and the batch-pair legend covers the WarpTransformer's Jan16×Jul16 point. Place both outside
the axes, e.g. `bbox_to_anchor=(1.02, 1), loc='upper left'` and
`bbox_to_anchor=(1.02, 0), loc='lower left'`, and save with `bbox_inches='tight'`.

**Smoke test (about 10 min).** Run A1 on one pair and seed before starting the lanes:
```bash
python3.11 scripts/16_align_testtime_calib.py --study-a fish_oil/batch_jan2016 --study-b fish_oil/batch_apr2016 \
  --seed 0 --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
```
Check that the pooled-PCHIP row appears and is plausible (TIC r should be well above
unaligned), then start the lanes. This run is also step 1's first job, so nothing is wasted.

## 1. CPU lane: seeded calibration, 4 pairs × 5 seeds, plus excluded pairs (~2.5–3 h)

```bash
for s in 0 1 2 3 4; do
  for p in "sep2015 jul2016" "sep2015 apr2016" "jan2016 apr2016" "jan2016 jul2016"; do set -- $p
    python3.11 scripts/16_align_testtime_calib.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2 \
      --seed $s --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
  done
done
for p in "sep2015 jan2016" "apr2016 jul2016"; do set -- $p      # excluded pairs, seed 0 only
  python3.11 scripts/16_align_testtime_calib.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2 \
    --seed 0 --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
done
```
→ `results/warp_transformer_calib_*_seed{s}.txt`, `results/per_pair/tic_r_calib_*_seed{s}.npz`.
The excluded pairs are rerun on CPU so the excluded-pairs table comes from the same setup as
the main tables.

## 2. CPU lane, after step 1: held-out anchor RT error (~2.5 h)

```bash
python3.11 scripts/22_heldout_anchor_error.py --seed 0 --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
```
→ `results/heldout_anchor_error.txt`, `results/per_pair/heldout_anchor_error.npz`. For each of the
22 calibration anchors (7 + 5 + 4 + 6), held out in turn: the RT error (minutes) of the
WarpTransformer recalibrated without it, PCHIP refit through the remaining anchors (this is the
pooled-anchor PCHIP of A1, evaluated at a compound it has not seen), and COW-TIC; plus the
no-alignment and constant-shift reference rows and the WarpTransformer's in-sample residual.
This is the paper's only compound-level measure of alignment accuracy.

## 3. GPU lane: pretraining ablation, 4 pairs × 5 seeds (~1–1.5 h)

```bash
for s in 0 1 2 3 4; do
  for p in "sep2015 jul2016" "sep2015 apr2016" "jan2016 apr2016" "jan2016 jul2016"; do set -- $p
    python3.11 scripts/18_ablate_pretrain.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2 \
      --seed $s --device mps --ckpt checkpoints/warp_transformer_baseline.pt
  done
done
```
→ `results/ablate_pretrain_*_seed{s}.txt`, `results/per_pair/tic_r_ablate_*_seed{s}.npz`.
Each seed gives a paired comparison (pretrained vs random init, same seed), so the
pretraining gain gets a mean ± s.d. and can be judged against seed variation, which the
current draft cannot do.

## 4. GPU lane, after step 3: anchor-count ablation (~2–3 h)

```bash
python3.11 scripts/23_anchor_count_ablation.py --seed 0 --subsets 3 --device mps --ckpt checkpoints/warp_transformer_baseline.pt
```
→ `results/anchor_count_ablation.txt`, `results/per_pair/anchor_count_ablation.npz`. TIC r
against the number of calibration anchors k (2 up to 7, 5, 4 and 6 for the four pairs; mean ±
s.d. over up to 3 random subsets per k), for pretrained and random-init models. About 46
anchor subsets, two calibrations each. It answers how few anchors are needed, and whether
pretraining matters more when anchors are scarce.

## 5. After step 1: significance tests across seeds (seconds)

```bash
python3.11 scripts/21_significance_tests.py --seed 0 --seeds 0 1 2 3 4
```
→ `results/significance_tests.txt`. Per-A-sample Wilcoxon tests with Holm correction at
seed 0, now including pooled-anchor PCHIP (A2), and the number of seeds (out of 5) at which
each WarpTransformer comparison stays significant.

## 6. After steps 1 and 3: aggregate (seconds)

```bash
python3.11 scripts/24_aggregate_seeds.py
```
→ `results/seed_summary.txt`. Every number for Tables IV–VIII and the per-sample table.

## 7. After steps 1 and 5: figures (~20 min on CPU)

```bash
# Figs. 1 and 3: same sample pair as now (batch A[0] vs B[0]); Fig. 3 calibrates both arms
python3.11 scripts/20_generate_figures.py --figs 1 3 --seed 0 --device cpu \
  --ckpt checkpoints/warp_transformer_baseline.pt
# Fig. 2: the median-improvement sample instead of the best one
python3.11 scripts/20_generate_figures.py --figs 2 --select median --seed 0 --device cpu \
  --ckpt checkpoints/warp_transformer_baseline.pt
# Figs. 4 and 5: from the seed-0 arrays of step 1 (A4 applied to Fig. 5)
python3.11 scripts/20_generate_figures.py --figs 4 5 --fig-seed 0
```
→ `results/fig1_*`, `fig2_*_median.pdf`, `fig3_*`, `fig4_*`, `fig5_*`, and
`results/fig_selection_*_median.txt` (sample ID and Δ TIC r for the Fig. 2 caption).

## Suggested schedule

| Time | CPU lane | GPU lane |
|---|---|---|
| Morning, first hour | A1–A4, smoke test | — |
| Then | Step 1 (~2.5–3 h) | Step 3 (~1–1.5 h), then step 4 (~2–3 h) |
| After step 1 | Step 2 (~2.5 h); steps 5 and 7 can run alongside it | |
| End of day | Step 6 | |

Wall-clock time: about 6–7 h after the code changes.

## Paper section ← output

| Paper | Source |
|---|---|
| Tables IV–VII (mean ± s.d. over 5 seeds; new pooled-PCHIP row) | `seed_summary.txt` (step 6) |
| Per-sample table (WT vs COW-TIC; add WT vs pooled PCHIP) and significance text | `significance_tests.txt` (step 5), `seed_summary.txt` |
| Ablation table (mean ± s.d.; gain judged against seed s.d.) | `seed_summary.txt` |
| Excluded-pairs table | step 1 (excluded pairs) |
| New held-out anchor RT error table; Discussion "What the Silhouette Can and Cannot Show" | `heldout_anchor_error.txt` (step 2) |
| Discussion "What the Network Adds" (pooled-PCHIP comparison replaces the confound caveat) | step 1, step 2 |
| Discussion "Few Anchors Suffice"; pretraining vs anchor count | `anchor_count_ablation.txt` (step 4) |
| "Runs" paragraph and Limitations (single-run caveat replaced by seed s.d.) | `seed_summary.txt` |
| Figs. 1–5 and captions | step 7 |