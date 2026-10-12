# Paper run v2 (2026-10-12): every result and figure, after the code audit

All steps on **CPU** (bit-reproducible under a seed), checkpoint `checkpoints/warp_transformer_v2.pt`
(the default), data time-binned by step 03 (`results/scan_time_audit.txt`). The v1 results are
backed up in `results/v1_before_fixes/`. Logs: `logs/run_v2/`.

**Stage 1** (`logs/run_v2/stage1.sh`, ~2 h), then a check before stage 2: WarpTransformer TIC r
above PCHIP on every pair, batch-A warp magnitude small, "outside" fraction near 0 %, calibration
loss falling.

```bash
python3.11 scripts/04_audit_fish_oil_samples.py --raw-dir <raw-subsetted-CVSs>
python3.11 scripts/02_train_warp_transformer.py --device cpu          # → checkpoints/warp_transformer_v2.pt
for p in "sep2015 jul2016" "sep2015 apr2016" "jan2016 apr2016" "jan2016 jul2016" \
         "sep2015 jan2016" "apr2016 jul2016"; do set -- $p
  python3.11 scripts/05_literature_baselines.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2
done
for p in "sep2015 jul2016" "sep2015 apr2016" "jan2016 apr2016" "jan2016 jul2016"; do set -- $p
  python3.11 scripts/06_align_testtime_calib.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2 --seed 0
done
```

**Stage 2** (~6–8 h, two parallel lanes; parallelism does not change CPU results)

```bash
P4='"sep2015 jul2016" "sep2015 apr2016" "jan2016 apr2016" "jan2016 jul2016"'
# Lane 1: calibration seeds 1–4 and the excluded pairs (seed 0)
for s in 1 2 3 4; do for p in <P4>; do set -- $p
  python3.11 scripts/06_align_testtime_calib.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2 --seed $s
done; done
for p in "sep2015 jan2016" "apr2016 jul2016"; do set -- $p
  python3.11 scripts/06_align_testtime_calib.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2 --seed 0
done
# Lane 2: pretraining ablation, 4 pairs × 5 seeds (CPU)
for s in 0 1 2 3 4; do for p in <P4>; do set -- $p
  python3.11 scripts/08_ablate_pretrain.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2 --seed $s
done; done
# After both lanes
python3.11 scripts/07_pchip_pooled_anchors.py --seed 0
python3.11 scripts/09_mechanism_ablation.py --seed 0
python3.11 scripts/10_heldout_anchor_error.py --seed 0
python3.11 scripts/11_within_batch_ceiling.py
python3.11 scripts/12_significance_tests.py --seed 0 --seeds 0 1 2 3 4
python3.11 scripts/13_aggregate_seeds.py --seeds 0 1 2 3 4
python3.11 scripts/14_generate_figures.py --figs 1 2 3 4 5 --select median --seed 0
# Figure pair changed to Jan16×Jul16 after the run (2026-10-12): Sep15×Jul16 is the weakest pair
python3.11 scripts/14_generate_figures.py --study-a fish_oil/batch_jan2016 --study-b fish_oil/batch_jul2016 --figs 1 2 3 --select median --seed 0
# then copy results/fig*.pdf to figs/ under the manuscript names
```

---

# Earlier plans (history; superseded by the run above)

# Three experiments for the WarpTransformer paper (one day on an Apple M2)

The draft rests on one GPU run, a whole-chromatogram metric computed on a normalised TIC,
and no compound-level check. This plan adds the three experiments that fix those, in
priority order. If the day runs short, protect them in this order.

| # | Experiment | What it answers | Paper gap it closes | Time |
|---|---|---|---|---|
| 1 | Held-out anchor RT error | Does each method put a compound it never saw in the right place? | No compound-level evidence; whether COW-TIC's gains are real alignment; whether the network beats a network-free warp between anchors | ~2.5–3.5 h CPU |
| 2 | Five calibration seeds on CPU | Are the differences larger than run-to-run variation? | Single run; the +0.023 margin over COW-TIC and the pretraining gain on Sep15×Jul16; GPU vs CPU mismatch | ~2.5–3 h CPU, plus ~1–1.5 h GPU |
| 3 | Raw TIC and a TIC r ceiling | Do the results hold on the real TIC, and what share of the achievable gap does each method close? | The metric is a normalised TIC that does not look like a chromatogram; no upper bound for TIC r | ~1–2 h, mostly code |

Out of scope, and stays in Limitations: the anchor-count sweep (script 23), a second
dataset, and drift events independent of the Jan–Apr 2016 interval.

## 0. Prerequisites

- **Python.** Call `python3.11` explicitly (`python3` is a zsh alias, not expanded under `nohup`).
- **Checkpoint.** Every command passes `--ckpt checkpoints/warp_transformer_baseline.pt`. Several
  scripts default to `warp_transformer.pt`, which no longer exists.
- **Data.** `data/fish_oil/batch_{sep2015,jan2016,apr2016,jul2016}/chroma/*.npz` with TAR3
  (Sep 2015 = 36, Jan 2016 = 26, Apr 2016 = 10, Jul 2016 = 16).
- **Devices.** Everything that feeds the main tables runs on **CPU** (`--device cpu`), which is
  exactly reproducible under a seed. Only the pretraining ablation (2c) uses the GPU, as a
  parallel lane, because it is reported as a mean over seeds anyway.
- **Script numbers.** 24 = pooled-anchor PCHIP and 25 = mechanism ablation already exist.
  New scripts in this plan are 26 (aggregate seeds) and 27 (within-batch ceiling).
- **Overwrites.** Step 2a rewrites `results/per_pair/tic_r_calib_*_seed0.npz` (now CPU instead of
  MPS). Every downstream step is rerun after it, so nothing is left inconsistent.

## A. Code changes (morning, before the long runs; ~1–1.5 h)

**A1. Raw TIC in preprocessing (for experiment 3).**
`scripts/preprocess_fish_oil_batches.py`: after `bin_chromatogram`, also save `tic_raw`, the
raw (pre-sqrt, pre-normalisation) summed intensity of each selected apex scan, shape `[200]`,
in each sample's `.npz`. `chroma` is unchanged. Rerun the preprocessing.

**A2. Raw-TIC evaluation in every method (for experiment 3).**
Wherever a method produces a warp, apply the same warp to `tic_raw` (1-D linear
interpolation at the warped positions) and compute Pearson r on it too:
- `scripts/17_literature_baselines.py`: icoshift and COW-TIC. COW-TIC is still *optimised* on
  the normalised TIC, as now; only the evaluation changes.
- `scripts/24_pchip_pooled_anchors.py`: PCHIP.
- `scripts/16_align_testtime_calib.py` and `scripts/18_ablate_pretrain.py`: the WarpTransformer.

Save per-pair arrays with a `_raw` suffix next to the existing ones and print a "TIC r (raw)"
row in each results file. Keep the normalised numbers.

**A3. Network-free warp in the held-out test (for experiment 1).**
`scripts/22_heldout_anchor_error.py`: add a method `Free warp (no network)` to `METHODS`. For
each held-out anchor, call `free_warp(n_a, n_b, rest, lr=calib16.LR_CALIB * 100, seed)` from
script 25 (load it the same way script 25 loads its helpers), and predict the drift at `r` as
`w_B(r) − w_A(r)` from its two batch warps, as `wt_offset_at` does for the WarpTransformer.
This takes seconds per anchor. It makes the held-out test answer the mechanism question
directly: both fit the same anchors with the same loss, so a lower held-out error for the
WarpTransformer means the network shapes the warp better between anchors.

**A4. Seed aggregation (for experiment 2). New `scripts/26_aggregate_seeds.py`.**
Reads `tic_r_baselines_*.npz` (script 17), `tic_r_pchip_pooled_*.npz` (script 24),
`tic_r_calib_*_seed{0..4}.npz` (script 16) and `tic_r_ablate_*_seed{0..4}.npz` (script 18) for
the four evaluated pairs, and writes `results/seed_summary.txt` with, per pair:
- WarpTransformer TIC r (normalised and raw): mean ± s.d. over seeds, min and max; warp
  magnitude (batch B, batch A, all) mean ± s.d.
- The deterministic rows (unaligned, icoshift, PCHIP, COW-TIC), normalised and raw, with a
  check that they are identical across seeds.
- WarpTransformer − COW-TIC and WarpTransformer − PCHIP: mean ± s.d. over seeds, and the
  number of seeds at which each is positive.
- Per batch-A sample: how many samples the WarpTransformer beats COW-TIC for, at each seed.
- Ablation: TIC r and warp magnitude (all / B) for both arms, mean ± s.d.; the pretraining
  gain per seed, then mean ± s.d.; and the share of the improvement random init recovers.

**A5. Within-batch ceiling (for experiment 3). New `scripts/27_within_batch_ceiling.py`.**
For each batch: mean Pearson r over all pairs of distinct samples within the batch, unaligned,
on both the normalised TIC and `tic_raw`. For each evaluated pair, also the mean of its two
batches' values. Write `results/within_batch_ceiling.txt`. No model; seconds.

**A6. Figure fixes (for experiment 3). `scripts/20_generate_figures.py`.**
- Repo fig2 (TIC overlays, paper Fig. 3): plot `tic_raw` (max-normalised for display; y-axis
  "TIC (scaled)") instead of the normalised TIC. Legend labels `Jul 2016 sample 0` and
  `Sep 2015 sample {i} (aligned)` instead of `batch_…[i]`. Drop the suptitle (the caption says
  the same).
- Repo fig1 (warp functions, paper Fig. 4): same plain labels.
- Repo fig4 (distributions, paper Fig. 2): tick labels 9 pt, panel titles and y-label 10 pt.

**Smoke test (~10 min).** After A1 and A2, rerun the preprocessing, then:
```bash
python3.11 scripts/16_align_testtime_calib.py --study-a fish_oil/batch_jan2016 --study-b fish_oil/batch_apr2016 \
  --seed 0 --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
```
Check that the "TIC r (raw)" row appears and that the WarpTransformer's raw value is well
above the unaligned raw value. This is also the first job of step 2a, so nothing is wasted.

## Experiment 3 first: raw TIC and ceiling (~20 min of runs, after A1, A2, A5)

It is quick, and steps 2a and 1 depend on the new preprocessing.

```bash
python3.11 scripts/preprocess_fish_oil_batches.py
for p in "sep2015 jul2016" "sep2015 apr2016" "jan2016 apr2016" "jan2016 jul2016" \
         "sep2015 jan2016" "apr2016 jul2016"; do set -- $p
  python3.11 scripts/17_literature_baselines.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2
done
python3.11 scripts/24_pchip_pooled_anchors.py --seed 0
python3.11 scripts/27_within_batch_ceiling.py
```
→ `literature_baselines_*.txt` and `tic_r_baselines_*.npz` with raw rows;
`pchip_pooled_anchors.txt` with raw rows; `within_batch_ceiling.txt`.
The WarpTransformer's raw-TIC numbers come from step 2a.

## Experiment 2: five seeds (CPU lane, with the GPU lane in parallel)

**2a. CPU: seeded calibration, 4 pairs × 5 seeds, plus excluded pairs (~2.5–3 h).**
```bash
for s in 0 1 2 3 4; do
  for p in "sep2015 jul2016" "sep2015 apr2016" "jan2016 apr2016" "jan2016 jul2016"; do set -- $p
    python3.11 scripts/16_align_testtime_calib.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2 \
      --seed $s --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
  done
done
for p in "sep2015 jan2016" "apr2016 jul2016"; do set -- $p      # excluded pairs, seed 0
  python3.11 scripts/16_align_testtime_calib.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2 \
    --seed 0 --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
done
```
→ `warp_transformer_calib_*_seed{s}.txt`, `tic_r_calib_*_seed{s}.npz`. Ignore the raw-PCHIP rows
these files still print; PCHIP comes from script 24.

**2b. After 2a: significance across seeds (seconds).**
```bash
python3.11 scripts/21_significance_tests.py --seed 0 --seeds 0 1 2 3 4
```
→ `significance_tests.txt`: per-A-sample Wilcoxon, Holm over the 10 comparisons within each
pair, at seed 0, plus the number of seeds (of 5) at which each WarpTransformer comparison stays
significant.

**2c. GPU lane, in parallel with 2a: pretraining ablation, 4 pairs × 5 seeds (~1–1.5 h).**
```bash
for s in 0 1 2 3 4; do
  for p in "sep2015 jul2016" "sep2015 apr2016" "jan2016 apr2016" "jan2016 jul2016"; do set -- $p
    python3.11 scripts/18_ablate_pretrain.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2 \
      --seed $s --device mps --ckpt checkpoints/warp_transformer_baseline.pt
  done
done
```
→ `ablate_pretrain_*_seed{s}.txt`, `tic_r_ablate_*_seed{s}.npz`. Each seed is a paired comparison
(pretrained vs random init), so the pretraining gain gets a mean ± s.d. It runs on MPS to save
time; that is fine because it is reported only as a mean over seeds. If the GPU is busy, run
it on CPU after experiment 1 instead (~2 h more).

**2d. After 2a and 2c: aggregate (seconds).**
```bash
python3.11 scripts/26_aggregate_seeds.py
```
→ `seed_summary.txt`.

## Experiment 1: held-out anchor RT error (CPU lane, after 2a; ~2.5–3.5 h)

```bash
python3.11 scripts/22_heldout_anchor_error.py --seed 0 --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
```
→ `heldout_anchor_error.txt`, `per_pair/heldout_anchor_error.npz`. For each of the 22
calibration anchors (7 + 5 + 4 + 6), held out in turn, the RT error in minutes of:
- the WarpTransformer recalibrated without it;
- the network-free warp fitted without it (A3);
- PCHIP refit through the remaining pooled anchors;
- COW-TIC (which never uses anchors, so nothing is held out);
- no alignment and a constant shift, as references;
- plus the WarpTransformer's in-sample residual.

About 26 calibrations, which is why it takes this long. Seed 0 only.

**How to read it.** Low WarpTransformer error relative to PCHIP and the free warp supports the
claim that the network shapes the warp correctly between anchors, at the compound level.
COW-TIC's error tests whether its TIC gains are real alignment. End anchors are extrapolated by
every method once held out, so read the per-anchor values, not just the means.

## Figures (after 2a; ~10 min)

```bash
# Paper Figs. 3 and 4 (repo fig2, fig1): the median-improvement sample, raw TIC (A6)
python3.11 scripts/20_generate_figures.py --figs 1 2 --select median --seed 0 --device cpu \
  --ckpt checkpoints/warp_transformer_baseline.pt
# Paper Fig. 2 (repo fig4): from the seed-0 arrays of 2a
python3.11 scripts/20_generate_figures.py --figs 4 --fig-seed 0
```
Paper Fig. 5 (repo fig3, calibration curve) is unchanged. Repo fig5 (silhouette) is not in the
paper. Note the sample ID, Δ TIC r and per-panel r values the script prints, for the captions.

## Suggested schedule

| Time | CPU lane | GPU lane |
|---|---|---|
| Morning, first 1–1.5 h | A1–A6, smoke test | — |
| Then ~20 min | Experiment 3 runs (preprocess, 17, 24, 27) | — |
| Then ~2.5–3 h | 2a (seeds) | 2c (pretraining ablation) |
| After 2a | Experiment 1 (~2.5–3.5 h); 2b, 2d and figures alongside | — |

Wall-clock time: about 7–8 h including the code changes.

## Paper section ← output

| Paper | Source |
|---|---|
| Tables IV, V, VII: mean ± s.d. over 5 seeds (CPU); raw-TIC rows | `seed_summary.txt`, experiment 3 outputs |
| Per-sample table and significance text; seeds at which each comparison holds | `significance_tests.txt`, `seed_summary.txt` |
| Pretraining ablation table, with gain ± s.d. | `seed_summary.txt` (from 2c) |
| New held-out anchor RT error table; Discussion; Limitations sentence on compound-level accuracy | `heldout_anchor_error.txt` |
| TIC r ceiling (sentence or column in Results; Limitations sentence updated) | `within_batch_ceiling.txt` |
| Dataset paragraph on the normalised TIC; abstract if the raw-TIC results differ | experiment 3 outputs, `seed_summary.txt` |
| "Runs" paragraph and Limitations: single-run caveat replaced by seed s.d.; GPU/CPU note removed | `seed_summary.txt` |
| Figs. 2–4 and captions | Figures step |

The mechanism ablation (Section VI-D and its table) stays as a single CPU seed-0 diagnostic.
After 2a its WarpTransformer row matches the seed-0 CPU run, so the "up to 0.017 lower than
the GPU" note can go.

## Question for the data holders (no code)

Were n-alkane standards run alongside any of the four batches? If so, retention indices would
give an independent, chemistry-based check of the alignment (Limitations currently says this
is unknown).