# Runs needed to finish the WarpTransformer paper

**Status (2026-10-10, evening):** the code changes (A1–A3) and the data rebuild (step B) are done.
Steps 3–11 must be **rerun from scratch** because the batches now include the 14 tarakihi (TAR3)
samples that were missing before. A full run on the old data finished earlier today; it was
backed up and then partly overwritten when the rerun was stopped (see "State of `results/`").

Every script below writes its output to `results/` (tracked in git; see Hand-back). Run steps
3–11 in the order given; the dependencies are noted at each step. Times are measured on an Apple
M2 (8 cores) and are for the old batch sizes. Sep15 is now 50% larger (24 → 36) and appears in
three of the four evaluated pairs, so expect runs on those pairs to take roughly 1.5× longer.

**Goal:** every number and figure in the paper comes from `warp_transformer_baseline.pt`, the
complete fish-oil data (TAR3 included), and the same seeded CPU runs. When this is done, the
paper's "Checkpoints and Runs" paragraph, which explains the current mix of old and new
checkpoints, can be removed.

## 0. Prerequisites

- **Python.** Call `python3.11` explicitly. In this shell `python3` is a zsh alias, and aliases are
  not expanded under `nohup` or in scripts, so `nohup python3 …` silently picks up a different
  interpreter without `h5py`.
- **Data.** `data/fish_oil/batch_{sep2015,jan2016,apr2016,jul2016}/chroma/*.npz` must exist and
  hold **36 / 26 / 10 / 16** samples (step B). Check before starting:
  ```bash
  for b in sep2015 jan2016 apr2016 jul2016; do echo $b $(ls data/fish_oil/batch_$b/chroma/*.npz | wc -l); done
  ```
- **Checkpoint.** All paper numbers use `checkpoints/warp_transformer_baseline.pt`. It is the
  paper's architecture (1,326,849 parameters), retrained on 2026-10-09 with the current
  (v − 1)² smoothness loss, because the original June checkpoint no longer exists. It is
  pretrained on MoNA only, so it does **not** need retraining for the TAR3 change. Every command
  below passes it via `--ckpt`, so nothing depends on what `warp_transformer.pt` currently holds.
  If it is missing, rebuild it (about 13–16 min on MPS):
  ```bash
  python3.11 scripts/13_train_warp_transformer.py
  cp checkpoints/warp_transformer.pt checkpoints/warp_transformer_baseline.pt
  ```
  Script 13 always writes to `checkpoints/warp_transformer.pt` and overwrites it.
- **Device: CPU for everything.** Calibration on MPS is not bit-reproducible under a fixed seed.
  On CPU, same-seed reruns are identical, and this held in every check of the 2026-10-10 run
  (20/20 ablation runs matched step 4 exactly; a seed-0 rerun matched the committed file
  exactly). MPS is worse than previously assumed: the MPS anchor-count run gave full-anchor
  TIC r about 0.02 above the CPU tables on two pairs, not the ~0.002 noted before. So step 8
  also runs on CPU (see step 8 for the MPS option if time matters more).
- **Absolute paths.** Scripts 06 and 13 read MoNA from
  `/Users/woodj/Desktop/chroma-dcnn/data/pretraining/spectra.h5`, and the audit script reads
  `/Users/woodj/Desktop/chroma-dcnn/data/fish_oil/sample_ids.txt`. Edit these on another machine.

## A. Code changes. Done 2026-10-10

- **A1** `scripts/17_literature_baselines.py` also saves `sil` (unaligned, COW-TIC, icoshift)
  and `warp_mag` (COW-TIC, icoshift) in `results/per_pair/tic_r_baselines_*.npz`.
- **A2** `scripts/20_generate_figures.py` Fig. 4 loads the saved per-pair arrays
  (`tic_r_baselines_*.npz`; `pchip_b0` and `warptransformer` from `tic_r_calib_*_seed{N}.npz`)
  instead of recalibrating. New argument `--fig-seed` (default 0).
- **A3** Fig. 5 reads TIC r and silhouette from the same files instead of the hard-coded list.
  Both figures check that the Unaligned arrays from scripts 16 and 17 agree (`np.allclose`,
  atol 1e-6, as in script 21; they differ by ~4e-8 from float rounding).

## B. Rebuild the fish-oil batches with TAR3. Done 2026-10-10

The previous `data/fish_oil` was built from
`~/Desktop/fish-data-analysis/src/data/raw-subsetted-CVSs/`, which holds only 1 of the 17 TAR3
files. The complete export (103 files, otherwise identical) is
`~/Desktop/fish-data-analysis/engr489/src/data/raw-subsetted-CVSs/`. The batches were rebuilt
from it, and `scripts/preprocess_fish_oil_batches.py` now uses it as the default `--raw-dir`:
```bash
python3.11 scripts/preprocess_fish_oil_batches.py --raw-dir ~/Desktop/fish-data-analysis/engr489/src/data/raw-subsetted-CVSs
```
Result: Sep15 24 → 36 (+12 TAR3), Jul16 14 → 16 (+2 TAR3 Skin), Jan16 26 and Apr16 10 unchanged.
The 74 existing sample files are byte-identical to before; only TAR3 files were added. The old
batches are kept in `data/fish_oil_pre_tar3/` (git-ignored).

## 1. Sample audit (task 4). Done 2026-10-10

```bash
python3.11 scripts/audit_fish_oil_samples.py --raw-dir ~/Desktop/fish-data-analysis/engr489/src/data/raw-subsetted-CVSs
```
→ `results/fish_oil_sample_audit.txt` (already up to date). All 103 samples have a raw file and
there is **no fifth batch**: every acquisition date is in `DATE_TO_BATCH`.

| Outcome | n |
|---|---|
| Sep 2015 / Jan 2016 / Apr 2016 / Jul 2016 | 36 / 26 / 10 / 16 |
| Instrument blanks (Frame A–C of SNA1, TAR3, BCO4; BCO5 Gonad Male A, Rest of Guts A–C) | 13 |
| Aug 2015, not evaluated (BCO4 Head A/B) | 2 |

TAR3 (17 files): 12 tissues acquired 2015-09-16/17 → Sep15; Skin A/B 2016-07-11 → Jul16;
Frame A–C are blanks (`160112 BLANK009–011`). `results/fish_oil_missing_samples.txt` (29
missing) is now out of date; the §III Dataset sentence should use the audit instead.

## 2. Pretraining wall-clock time (task 5). Already measured; optional

```bash
cp checkpoints/warp_transformer.pt checkpoints/warp_transformer.pt.bak   # script 13 overwrites it
/usr/bin/time -p python3.11 scripts/13_train_warp_transformer.py
mv checkpoints/warp_transformer.pt.bak checkpoints/warp_transformer.pt
```
**Already measured**: 853.5 s (14 min 14 s) on an Apple M2 with MPS, idle machine. See
`results/pretraining_timing.txt`. Independent of the fish data, so TAR3 does not affect it.

**Calibration wall time:** do not take it from step 4 if steps run in parallel. The 2026-10-10
run printed 600–690 ± 230 s per calibration because two or three jobs shared the CPU. For the
paper, time one calibration on an idle machine, e.g. one step 4 command run on its own.

## 3. Literature baselines, all six pairs (task 1). Deterministic. ~5 min

The four evaluated pairs and the two excluded pairs (for the excluded-pairs table). COW-TIC on
Sep15×Jan16 is the slowest.
```bash
for p in "sep2015 jul2016" "sep2015 apr2016" "jan2016 apr2016" "jan2016 jul2016" \
         "sep2015 jan2016" "apr2016 jul2016"; do set -- $p
  python3.11 scripts/17_literature_baselines.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2
done
```
→ `results/literature_baselines_*.txt`, `results/per_pair/tic_r_baselines_*.npz`
(Unaligned, COW-TIC and icoshift TIC r indexed by (A_i, B_j), silhouette and warp magnitude).

## 4. Seeded calibration, 4 evaluated pairs × 5 seeds (tasks 1, 6, 8)

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
- The anchors and the calibration wall time (task 5; see step 2).

For Tables IV–VII, take mean ± s.d. over the five seeds.

Measured per run (old data): ~4.3 min alone, ~7 min with one other CPU job, ~13 min with two.

## 5. Seeded calibration, 2 excluded pairs (excluded-pairs table)

```bash
for p in "sep2015 jan2016" "apr2016 jul2016"; do set -- $p
  python3.11 scripts/16_align_testtime_calib.py --study-a fish_oil/batch_$1 --study-b fish_oil/batch_$2 \
    --seed 0 --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
done
```
→ `results/warp_transformer_calib_batch_sep2015_batch_jan2016_seed0.txt` and
`..._batch_apr2016_batch_jul2016_seed0.txt` (plus `.npz`). Seed 0 is enough for this table;
loop over seeds 0–4 as in step 4 if the table should also show mean ± s.d.

## 6. Pretraining ablation, 4 pairs × 5 seeds (tasks 2, 6)

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
and random-init arms. These fill the "---" cells of the ablation table (Sep15×Jul16 and
Jan16×Jul16), and give mean ± s.d. for the whole table.

Measured per run (old data): ~12–15 min with one other CPU job, ~26 min with two.

**Consistency check:** for the same seed on CPU, the pretrained arm's per-pair r must equal
step 4's WarpTransformer array exactly. Check all 20 at once:
```bash
python3.11 -c "
import numpy as np, glob
fs = sorted(glob.glob('results/per_pair/tic_r_ablate_*.npz'))
print(len(fs), 'runs, max |dr| =', max(abs(np.load(f)['pretrained'] - np.load(f.replace('ablate', 'calib'))['warptransformer']).max() for f in fs))"
```

## 7. Significance tests (task 1). Needs steps 3 and 4 (all five seeds). Seconds

```bash
python3.11 scripts/21_significance_tests.py            # tests seed 0; robustness over seeds 0-4
```
→ `results/significance_tests.txt`. Two-sided paired Wilcoxon tests for all 10 method pairs
per batch pair, with statistic, p, Holm-adjusted p, n and median Δr, at two levels:
- per sample pair (n = n_A × n_B). These observations are not independent, so the p-values
  are optimistic.
- per A sample (r averaged over B_j, n = n_A). Conservative; use this one in the paper.

It also checks that the Unaligned arrays from scripts 16 and 17 agree, which confirms the pairing.

## 8. Anchor-count ablation (task 9). About 92 calibrations

```bash
python3.11 scripts/23_anchor_count_ablation.py --seed 0 --subsets 3 --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
```
→ `results/anchor_count_ablation.txt`. TIC r against k (mean ± s.d. over up to 3 anchor subsets
per k) for the pretrained and random-init models.

**Run it last, on its own.** On CPU it takes roughly 7–9 h alone (about 3× the MPS time), and
running it next to steps 4–6 slows them down too. The MPS alternative (`--device mps`) took
2 h 45 min on the old data while sharing the machine, but it is not bit-reproducible, its
full-anchor values were ~0.02 above the CPU tables on Sep15×Jul16 and Sep15×Apr16, and it still
used enough CPU to slow the CPU jobs about 2×.

## 9. Held-out anchor RT error (task 7). Needs step 3 only. About 26 calibrations

```bash
python3.11 scripts/22_heldout_anchor_error.py --seed 0 --device cpu --ckpt checkpoints/warp_transformer_baseline.pt
```
→ `results/heldout_anchor_error.txt`. Mean and max leave-one-anchor-out error (minutes) per
pair for the WarpTransformer, Raw cosine PCHIP (refit without the anchor) and COW-TIC (its warp
at r). It also gives two reference rows (no alignment; constant shift by the median remaining
drift) and the WarpTransformer's in-sample residual.

Measured (old data): ~5 min per held-out anchor alone, ~9 min while sharing the CPU.

## 10. Figs. 1–3: warp functions, TIC overlays, calibration curve (task 10). ~15 min on CPU

```bash
python3.11 scripts/20_generate_figures.py --figs 1 2 3 --select median --seed 0 --device cpu \
  --ckpt checkpoints/warp_transformer_baseline.pt
```
→ `results/fig1_warp_functions_*_median.pdf`, `results/fig2_tic_overlays_*_median.pdf`,
`results/fig3_calibration_curve_batch_sep2015_batch_jul2016.pdf`, and
`results/fig_selection_*_median.txt` (the chosen sample's ID, its Δ TIC r, and the median and
max Δ over all A samples, for the captions). Fig. 3 calibrates both the pretrained and the
random-init model, which is why this step takes about twice as long as one calibration.
`--select best` reproduces the old best-case figures.

With an even number of A samples (Sep15 has 36), "median" picks the lower of the two middle
samples, so its Δ is slightly below the reported median Δ. Caption it as "the median-ranked
sample".

## 11. Figs. 4–5: per-pair distributions, silhouette scatter. Needs steps 3 and 4. Seconds

```bash
python3.11 scripts/20_generate_figures.py --figs 4 5 --fig-seed 0
```
→ `results/fig4_per_pair_distributions.pdf`, `results/fig5_silhouette_scatter.pdf`.
Built entirely from the saved arrays, so they match the tables and significance tests exactly.

## Suggested schedule

Ready-made runner scripts from 2026-10-10 are in `logs/run2/` (git-ignored logs; the `.sh`
files are untracked). Each is self-contained; start them with
`nohup logs/run2/chain_X.sh > logs/run2/chain_X.log 2>&1 &`.

| Chain | Steps | Estimate on the new data |
|---|---|---|
| A (CPU) | 3 → 4 → 7, 11, 10 → 9 | ~6–7 h |
| B (CPU) | 5 → 6 | ~6–7 h |
| C (CPU) | 8, waits until A and B are done | +7–9 h |

So about 6–7 h for everything except step 8, and about 14–16 h including it. These estimates
are scaled up from the old data and are rough. Running A and B in parallel gives about 1.2× the
throughput of running them back to back; adding a third job makes every job about 3× slower.

**Optional shortcut:** Jan16×Apr16 uses neither Sep15 nor Jul16, so its inputs did not change.
On CPU it would reproduce the old results exactly, and those are in `logs/results_pre_tar3/`.
Dropping that pair from steps 4, 6, 8 and 9 and copying its files back saves about a fifth of
the time, at the cost of not re-confirming reproducibility.

## Alternative: ECS GPU servers via the `task` spooler

`scripts/run_gpu_spooler.sh` queues steps 3–11 as 1-GPU tasks: 51 tasks, then one more for
steps 7 and 11. Run it from a checkout that the GPU servers see at the same path (ECS NFS
home), with `data/fish_oil/` and `checkpoints/warp_transformer_baseline.pt` copied in. Neither
is tracked by git.
```bash
PY=~/venvs/warp/bin/python scripts/run_gpu_spooler.sh preflight
PY=~/venvs/warp/bin/python scripts/run_gpu_spooler.sh check   # wait for it: must end "done"
PY=~/venvs/warp/bin/python nohup scripts/run_gpu_spooler.sh all > logs/gpu_run.log 2>&1 &
```
Seeded CUDA runs are deterministic only on one GPU model and software stack
(`enable_cuda_determinism()` in `src/warp_transformer.py`), and the spooler picks any free
card. `post` checks that step 6 matches step 4 exactly and reports any mismatch. GPU results
will not be bit-identical to CPU results, so do not mix the two in one set of tables. Step 1
and step 2 are not part of the script.

## State of `results/` (when the runs were stopped, 19:35 on 2026-10-10)

`results/` currently **mixes old and new data, so do not commit it as it is.**
- Old data (24 Sep15 / 14 Jul16, complete run finished 18:59): everything from steps 4–11,
  except the files below. A full copy is in `logs/results_pre_tar3/` (git-ignored).
- New data (with TAR3): step 1 (`fish_oil_sample_audit.txt`), step 3 for all six pairs, step 4
  seed 0 for Sep15×Jul16, and step 5 for Sep15×Jan16.

Rerunning steps 3–11 overwrites every affected file.

## Hand-back

`results/` is tracked in git. Commit it once **every** step has finished, not partway through:
reruns overwrite files, so a partial commit can mix runs. Tag the commit so the paper can point
to exactly these results.
```bash
git add results/
git commit -m "Results for preliminary draft: baseline checkpoint, TAR3 included, seeds 0-4"
git tag paper-draft-1
git push && git push --tags
```
Before committing, check that nothing from `data/` or `checkpoints/` is staged
(`git status`). Both stay git-ignored: the fish data are not public, and the checkpoint will be
published as a release asset rather than committed. Also leave out `logs/` (the runner `.sh`
files there are untracked). The old non-median Figs. 1–2 (`fig1_warp_functions_batch_sep2015_batch_jul2016.pdf`,
`fig2_…`) are still in `results/` from earlier runs.

## Already done (no run needed)

| Task | Output |
|---|---|
| 3 Smoothness loss provenance | `results/smoothness_loss_provenance.txt` |
| 4 Sample audit, reasons for every excluded sample (step 1) | `results/fish_oil_sample_audit.txt` |
| 5 Parameter count (1,326,849, matches the paper), pretraining time | `results/pretraining_timing.txt` |
| 11 Literature check and reference verification | `results/literature_check.txt` |

## Findings from the old-data run to re-check on the new data

These held on the old data (no TAR3) and may shift:
- The WarpTransformer was best on all four pairs (0.691, 0.667, 0.640, 0.607 ± ≤ 0.009) and
  significant against every method in all 5 seeds (Holm p ≤ 6.6e-6). Narrowest margin:
  Sep15×Apr16 vs COW-TIC, +0.038.
- Its study silhouette was above Unaligned on every pair (COW-TIC and icoshift too, on most),
  so the paper should not claim better batch integration.
- Pretraining helped in 20/20 ablation runs (+0.054 to +0.099 mean) and cut seed s.d. from up
  to 0.055 to ≤ 0.009.
- Held-out anchor error had no single winner. Anchor (177, 176) on Sep15×Jul16 (drift −1 bin,
  against −18 to −22 for its neighbours) was missed by about 4 min by every method; it is
  probably a mismatched anchor.

## Paper section ← output

| Paper | Source |
|---|---|
| Tables IV–VII (mean ± s.d.; WarpTransformer silhouette and warp magnitude; PCHIP pairwise rows and warp magnitudes); §VII-B | step 4 (COW-TIC and icoshift rows: step 3) |
| Excluded-pairs table | steps 3 and 5 |
| Significance table; §V Statistical Testing; §VI Summary Across Pairs | step 7 |
| Ablation table (incl. the four "---" cells) | step 6 |
| §VII-C, §VII-E, §VI ablation (anchor count) | step 8 |
| New held-out anchor table; §VII-A | step 9 |
| Figs. 1–3 and captions | step 10 |
| Figs. 4–5 and captions | step 11 |
| §V "Checkpoints and Runs" paragraph and the matching Limitations sentence | removed once steps 3–6 and 10–11 are done |
| §IV-C3 smoothness equation | `smoothness_loss_provenance.txt` (no change needed if all tables use the baseline checkpoint) |
| §III Dataset sentence (sample counts per batch; TAR3 included; 13 blanks and 2 Aug-2015 samples excluded) | step 1, `fish_oil_sample_audit.txt` |
| §IV-C2 Synthetic Pretraining (device, time, parameters) | `pretraining_timing.txt`; calibration time from an idle-machine run (step 2) |
| Related work, reference fixes | `literature_check.txt` |
