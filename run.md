# Runs needed to finish the WarpTransformer paper

Every script below writes its output to `results/` (git-ignored). Run them in the order given;
the dependencies are noted at each step. Times were measured on an Apple M2 (8 cores, MPS).

Scripts 16, 18, 20 (`--select median`), 21, 22 and 23 were smoke-tested end to end on
2026-10-09 with calibration cut to 2 steps; those outputs were deleted. On CPU with the same
seed, script 18's pretrained arm reproduced script 16's per-pair r exactly (max |Δr| = 0).

## 0. Prerequisites

- **Python.** Call `python3.11` explicitly. In this shell `python3` is a zsh alias, and aliases are
  not expanded under `nohup` or in scripts, so `nohup python3 …` silently picks up a different
  interpreter without `h5py`.
- **Data.** `data/fish_oil/batch_{sep2015,jan2016,apr2016,jul2016}/chroma/*.npz` must exist.
- **Checkpoint.** All paper numbers use `checkpoints/warp_transformer_baseline.pt`. It is the
  paper's architecture (1,326,849 parameters), retrained on 2026-10-09 with the current
  (v − 1)² smoothness loss, because the original June checkpoint no longer exists. Every command
  below passes it via `--ckpt`, so nothing depends on what `warp_transformer.pt` currently holds.
  If it is missing, rebuild it (about 13–16 min on MPS):
  ```bash
  python3.11 scripts/13_train_warp_transformer.py
  cp checkpoints/warp_transformer.pt checkpoints/warp_transformer_baseline.pt
  ```
  Script 13 always writes to `checkpoints/warp_transformer.pt` and overwrites it.
- **Device.** Calibration on MPS is not bit-reproducible under a fixed seed: same-seed reruns
  differ by up to 0.009 TIC r per sample pair, or about 0.002 in the mean. On CPU they are
  identical. So every number that goes into a table runs with `--device cpu`. That costs about
  4.3 min per 500-step calibration, against about 1.4 min on MPS. Only the anchor-count sweep
  (step 8), which is reported as averages over anchor subsets, runs on MPS.
- **Absolute paths.** Scripts 06 and 13 read MoNA from
  `/Users/woodj/Desktop/chroma-dcnn/data/pretraining/spectra.h5`, and the audit script reads
  `/Users/woodj/Desktop/chroma-dcnn/data/fish_oil/sample_ids.txt`. Edit these on another machine.

## 1. Sample audit (task 4). Independent; needs the raw data

```bash
python3.11 scripts/audit_fish_oil_samples.py --raw-dir <path>/raw-subsetted-CVSs
```
→ `results/fish_oil_sample_audit.txt`. Gives a reason for each of the 27 unassigned samples.
The raw `.txt` files were not on this machine. `results/fish_oil_missing_samples.txt` already
lists which 29 samples are missing (2 Aug-2015 + 27 unassigned, including all 17 TAR3).

## 2. Pretraining wall-clock time (task 5). Run alone on an idle machine

```bash
cp checkpoints/warp_transformer.pt checkpoints/warp_transformer.pt.bak   # script 13 overwrites it
/usr/bin/time -p python3.11 scripts/13_train_warp_transformer.py
mv checkpoints/warp_transformer.pt.bak checkpoints/warp_transformer.pt
```
**Already measured**, so this step is optional: 853.5 s (14 min 14 s) on an Apple M2 with MPS,
idle machine. See `results/pretraining_timing.txt`. Rerun only if the paper should report a
different machine. The 500-step calibration time is printed by every run of step 4
("Calibration wall time").

## 3. Literature baselines with per-pair arrays (task 1). About minutes per pair

```bash
for p in "sep2015 jul2016" "sep2015 apr2016" "jan2016 apr2016" "jan2016 jul2016"; do set -- $p
  python3.11 scripts/17_literature_baselines.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2
done
```
→ `results/literature_baselines_*.txt`, `results/per_pair/tic_r_baselines_*.npz`
(Unaligned, COW-TIC, icoshift, indexed by (A_i, B_j)). Deterministic.

## 4. Seeded calibration, 4 pairs × 5 seeds (tasks 1, 6, 8). About 20 × 4–5 min on CPU

```bash
for s in 0 1 2 3 4; do
  for p in "sep2015 jul2016" "sep2015 apr2016" "jan2016 apr2016" "jan2016 jul2016"; do set -- $p
    python3.11 scripts/16_align_testtime_calib.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2 \
      --seed $s --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
  done
done
```
→ `results/warp_transformer_calib_*_seed{s}.txt` and `results/per_pair/tic_r_calib_*_seed{s}.npz`.
Each run reports, per pair:
- TIC r, Δ unaligned and study silhouette for Unaligned, Raw cosine PCHIP (A_i → B[0], the
  table protocol), Raw cosine PCHIP (A_i → each B_j, task 8) and the WarpTransformer.
- Warp magnitude for both PCHIP protocols (task 8), and for the WarpTransformer over all
  samples, batch A and batch B (task 6).
- The anchors and the calibration wall time (task 5).

For Tables IV–VII, take mean ± s.d. over the five seeds.

## 5. Pretraining ablation, 4 pairs × 5 seeds (tasks 2, 6). About 20 × 9 min on CPU

```bash
for s in 0 1 2 3 4; do
  for p in "sep2015 jul2016" "sep2015 apr2016" "jan2016 apr2016" "jan2016 jul2016"; do set -- $p
    python3.11 scripts/18_ablate_pretrain.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2 \
      --seed $s --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
  done
done
```
→ `results/ablate_pretrain_*_seed{s}.txt`, `results/per_pair/tic_r_ablate_*_seed{s}.npz`.
Gives TIC r and warp magnitude (all samples of both batches, plus batch B) for the pretrained
and random-init arms. These fill the "---" cells of Table VIII (Sep15×Jul16 and Jan16×Jul16),
and give mean ± s.d. for the whole table.

**Consistency check:** for the same seed on CPU, the pretrained arm's per-pair r must equal
step 4's WarpTransformer array exactly. This is what removes the run-to-run caveat.
```bash
python3.11 -c "import numpy as np; a=np.load('results/per_pair/tic_r_calib_batch_sep2015_batch_jul2016_seed0.npz')['warptransformer']; b=np.load('results/per_pair/tic_r_ablate_batch_sep2015_batch_jul2016_seed0.npz')['pretrained']; print(abs(a-b).max())"
```
Steps 4 and 5 can run at the same time in two terminals.

## 6. Significance tests (task 1). Needs steps 3 and 4. Seconds

```bash
python3.11 scripts/21_significance_tests.py            # tests seed 0; robustness over seeds 0-4
```
→ `results/significance_tests.txt`. Two-sided paired Wilcoxon tests for all 10 method pairs
per batch pair, with statistic, p, Holm-adjusted p, n and median Δr, at two levels:
- per sample pair (n = n_A × n_B). These observations are not independent, so the p-values
  are optimistic.
- per A sample (r averaged over B_j, n = n_A). Conservative; use this one in the paper.

It also checks that the Unaligned arrays from scripts 16 and 17 are identical, which confirms
the pairing.

## 7. Held-out anchor RT error (task 7). About 26 calibrations, around 2 h on CPU

```bash
python3.11 scripts/22_heldout_anchor_error.py --seed 0 --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
```
→ `results/heldout_anchor_error.txt`. Mean and max leave-one-anchor-out error (minutes) per
pair for the WarpTransformer, Raw cosine PCHIP (refit without the anchor) and COW-TIC (its warp
at r). It also gives two reference rows (no alignment; constant shift by the median remaining
drift) and the WarpTransformer's in-sample residual.

## 8. Anchor-count ablation (task 9). About 92 calibrations, around 2–2.5 h on MPS

```bash
python3.11 scripts/23_anchor_count_ablation.py --seed 0 --subsets 3 --device mps --ckpt checkpoints/warp_transformer_baseline.pt
```
→ `results/anchor_count_ablation.txt`. TIC r against k (mean ± s.d. over up to 3 anchor subsets
per k) for the pretrained and random-init models. Use `--device cpu` for exact repeatability,
at about 3× the time. It can run alongside steps 4–5 (GPU vs CPU).

## 9. Median-improvement figure (task 10). One calibration, about 5 min on CPU

```bash
python3.11 scripts/20_generate_figures.py --figs 1 2 --select median --seed 0 --device cpu \
  --ckpt checkpoints/warp_transformer_baseline.pt
```
→ `results/fig1_warp_functions_*_median.pdf`, `results/fig2_tic_overlays_*_median.pdf`, and
`results/fig_selection_*_median.txt` (the chosen sample's ID, its Δ TIC r, and the median and
max Δ over all A samples, for the caption). `--select best` reproduces the current figures.

## Already done (no run needed)

| Task | Output |
|---|---|
| 3 Smoothness loss provenance | `results/smoothness_loss_provenance.txt` |
| 4 Missing samples (identified; reasons need step 1) | `results/fish_oil_missing_samples.txt` |
| 5 Parameter count (1,326,849, matches the paper), pretraining time, calibration time | `results/pretraining_timing.txt` |
| 11 Literature check and reference verification | `results/literature_check.txt` |

## Paper section ← output

| Paper | Source |
|---|---|
| Significance table; §V Statistical Testing; §VI Summary Across Pairs | step 6 |
| Table VIII (incl. the four "---" cells) | step 5 |
| §IV-C3 smoothness equation | `smoothness_loss_provenance.txt` (no change needed if Tables IV–VIII use steps 4–5) |
| §III Dataset sentence | `fish_oil_missing_samples.txt`, step 1 |
| §IV-C2 Synthetic Pretraining (device, time, parameters) | step 2, step 4 wall times, this file |
| Tables IV–VII mean ± s.d.; PCHIP pairwise rows and warp magnitudes; §VII-B | step 4 |
| New held-out anchor table; §VII-A | step 7 |
| §VII-C, §VII-E, §VI ablation (anchor count) | step 8 |
| Fig. 2 (and Fig. 1) caption | step 9 |
| Related work, reference fixes | `literature_check.txt` |
