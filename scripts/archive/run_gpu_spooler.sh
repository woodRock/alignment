#!/usr/bin/env bash
# run_gpu_spooler.sh — run the paper experiments (run.md steps 3–11) on the ECS GPU servers
# through the `task` GPU spooler.
#
# Each calibration run is its own 1-GPU task, so the spooler spreads them across free GPUs.
# The spooler has no job dependencies, so this script records the task IDs, waits on the
# ones that later steps need, and then submits those steps itself.
#
# Where to run it
#   On the machine you normally submit `task` jobs from, inside a checkout of this repo that
#   the GPU servers can also see at the same path (e.g. your ECS NFS home). The spooler
#   `cd`s into that path on whichever server it picks, and every task writes its results
#   straight into <repo>/results/ there.
#
# What the checkout needs that git does not carry
#   data/fish_oil/batch_{sep2015,jan2016,apr2016,jul2016}/   (36 / 26 / 10 / 16 samples, TAR3 included)
#   checkpoints/warp_transformer_baseline.pt
#   A Python on the servers with torch (CUDA), numpy, scipy, scikit-learn, matplotlib.
#
# Usage
#   scripts/run_gpu_spooler.sh preflight   # local checks: data counts, checkpoint, task binary
#   scripts/run_gpu_spooler.sh check       # 1 GPU task: same-seed calibration twice; must match exactly
#   scripts/run_gpu_spooler.sh submit      # queue steps 3, 4, 5, 6, 8, 9, 10
#   scripts/run_gpu_spooler.sh post        # wait for steps 3–4, then run 7 and 11; wait for all; consistency check
#   scripts/run_gpu_spooler.sh all         # preflight, submit, post (keep it alive: tmux/nohup)
#   scripts/run_gpu_spooler.sh status      # task IDs submitted by this script, with their status
#
# Settings (environment variables)
#   PY=python3            Python on the GPU servers (e.g. a venv path: ~/venvs/warp/bin/python)
#   DEVICE=cuda           cuda | cpu   (cpu still runs through the spooler, one GPU slot per task)
#   SEEDS="0 1 2 3 4"     calibration seeds for steps 4 and 6
#   MIN_MEM=              minimum GPU memory in GB, passed to `task -m` (see Reproducibility)
#   TASK_BIN=task         spooler binary
#   DRY_RUN=1             print the `task` commands instead of submitting them
#
# Reproducibility
#   Seeded CUDA runs are bit-reproducible only on the same GPU model and software stack
#   (enable_cuda_determinism() in src/warp_transformer.py). The spooler picks any free GPU,
#   and the ECS servers mix RTX A4000/A5000/6000, A40, A6000, L4, L40S and 6000 Ada cards.
#   Run `check` first. Each task's log header records the server it ran on, and `post`
#   checks that step 6's pretrained arm equals step 4 exactly; a mismatch usually means the
#   two ran on different GPU models. MIN_MEM=48 narrows the pool to 48 GB cards, but it
#   still mixes models.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MIN_MEM="${MIN_MEM:-}"
TASK_BIN="${TASK_BIN:-task}"
DRY_RUN="${DRY_RUN:-0}"

CKPT="checkpoints/warp_transformer_baseline.pt"
STATE_DIR="$REPO/logs/gpu_run"
IDS="$STATE_DIR/task_ids.tsv"              # id <TAB> step <TAB> label

EVAL_PAIRS=("sep2015 jul2016" "sep2015 apr2016" "jan2016 apr2016" "jan2016 jul2016")
EXCL_PAIRS=("sep2015 jan2016" "apr2016 jul2016")

# Exported into every task. PYTHONUNBUFFERED keeps `task -f` output live.
ENV=(env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1)

die() { echo "ERROR: $*" >&2; exit 1; }

# submit <step> <label> <command…>  → queues a 1-GPU task, records and prints its ID
submit() {
    local step="$1" label="$2"; shift 2
    local args=(-G 1 -n "$label" -d "$REPO")
    [[ -n "$MIN_MEM" ]] && args+=(-m "$MIN_MEM")
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '%s ' "$TASK_BIN" "${args[@]}" "${ENV[@]}" "$@"; echo
        return
    fi
    local out id
    out="$("$TASK_BIN" "${args[@]}" "${ENV[@]}" "$@")"
    id="$(sed -n 's/^Queued task \([0-9][0-9]*\).*/\1/p' <<<"$out" | head -1)"
    [[ -n "$id" ]] || die "could not read a task ID from: $out"
    mkdir -p "$STATE_DIR"
    printf '%s\t%s\t%s\n' "$id" "$step" "$label" >> "$IDS"
    echo "  task $id  $label"
}

# ids_for <step…>  → task IDs recorded for those steps (latest submission only)
ids_for() {
    [[ -f "$IDS" ]] || return 0
    local want=" $* "
    awk -F'\t' -v want="$want" 'index(want, " " $2 " ") { print $1 }' "$IDS"
}

# wait_ok <id…>  → blocks until each task finishes; fails unless every one ends as "done"
wait_ok() {
    local id res bad=0
    for id in "$@"; do
        res="$("$TASK_BIN" -w "$id" | tail -1)"
        if [[ "$res" != *"finished: done"* ]]; then
            echo "  task $id: ${res:-no status}  (see: $TASK_BIN -o $id)" >&2
            bad=1
        fi
    done
    return "$bad"
}

cmd_preflight() {
    echo "Repo: $REPO"
    command -v "$TASK_BIN" >/dev/null || die "'$TASK_BIN' not found on PATH"
    [[ -f "$REPO/$CKPT" ]] || die "missing $CKPT"
    local b n expect=(sep2015:36 jan2016:26 apr2016:10 jul2016:16) e
    for e in "${expect[@]}"; do
        b="${e%%:*}"
        n=$(find "$REPO/data/fish_oil/batch_$b/chroma" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
        [[ "$n" == "${e##*:}" ]] || die "batch_$b has $n samples, expected ${e##*:} (TAR3 included; see run.md step B)"
        echo "  batch_$b: $n samples"
    done
    echo "  checkpoint: $CKPT"
    echo "  python on servers: $PY   device: $DEVICE   seeds: $SEEDS   min mem: ${MIN_MEM:-any}"
}

# One task that calibrates the same pair with the same seed twice and compares the arrays.
cmd_check() {
    local f="results/per_pair/tic_r_calib_batch_sep2015_batch_jul2016_seed0.npz"
    local run="$PY scripts/16_align_testtime_calib.py --study-a fish_oil/batch_sep2015 \
--study-b fish_oil/batch_jul2016 --seed 0 --device $DEVICE --ckpt $CKPT"
    local script="set -e; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
$PY -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, torch.cuda.is_available())'
$run > /dev/null; cp $f /tmp/warp_check_\$\$.npz
$run > /dev/null
$PY -c 'import sys, numpy as np; a = np.load(sys.argv[1])[\"warptransformer\"]; b = np.load(sys.argv[2])[\"warptransformer\"]
d = float(abs(a - b).max()); print(\"same-seed rerun max |dr| =\", d, \"mean r =\", round(float(b.mean()), 4)); sys.exit(d != 0)' /tmp/warp_check_\$\$.npz $f"
    echo "Submitting determinism check (about 2 calibrations):"
    submit check "warp check: same-seed rerun" bash -c "$script"
    [[ "$DRY_RUN" == 1 ]] || echo "Follow it with: $TASK_BIN -f $(ids_for check | tail -1). It ends 'done' only if both runs match exactly."
}

cmd_submit() {
    cmd_preflight
    [[ -f "$IDS" && "$DRY_RUN" != 1 ]] && mv "$IDS" "$IDS.$(date +%Y%m%d-%H%M%S)"
    local p s a b

    echo "Step 3: literature baselines (6 pairs)"
    for p in "${EVAL_PAIRS[@]}" "${EXCL_PAIRS[@]}"; do set -- $p
        submit s3 "s3 baselines $1x$2" "$PY" scripts/17_literature_baselines.py \
            --study-a "fish_oil/batch_$1" --study-b "fish_oil/batch_$2"
    done

    echo "Step 4: seeded calibration (4 pairs x seeds $SEEDS)"
    for s in $SEEDS; do for p in "${EVAL_PAIRS[@]}"; do set -- $p
        submit s4 "s4 calib $1x$2 seed$s" "$PY" scripts/16_align_testtime_calib.py \
            --study-a "fish_oil/batch_$1" --study-b "fish_oil/batch_$2" \
            --seed "$s" --device "$DEVICE" --ckpt "$CKPT"
    done; done

    echo "Step 5: excluded pairs (seed 0)"
    for p in "${EXCL_PAIRS[@]}"; do set -- $p
        submit s5 "s5 calib $1x$2 seed0" "$PY" scripts/16_align_testtime_calib.py \
            --study-a "fish_oil/batch_$1" --study-b "fish_oil/batch_$2" \
            --seed 0 --device "$DEVICE" --ckpt "$CKPT"
    done

    echo "Step 6: pretraining ablation (4 pairs x seeds $SEEDS)"
    for s in $SEEDS; do for p in "${EVAL_PAIRS[@]}"; do set -- $p
        submit s6 "s6 ablate $1x$2 seed$s" "$PY" scripts/18_ablate_pretrain.py \
            --study-a "fish_oil/batch_$1" --study-b "fish_oil/batch_$2" \
            --seed "$s" --device "$DEVICE" --ckpt "$CKPT"
    done; done

    echo "Step 8: anchor-count ablation (all pairs, one task)"
    submit s8 "s8 anchor count" "$PY" scripts/23_anchor_count_ablation.py \
        --seed 0 --subsets 3 --device "$DEVICE" --ckpt "$CKPT"

    echo "Step 9: held-out anchor error (all pairs, one task)"
    submit s9 "s9 heldout anchors" "$PY" scripts/22_heldout_anchor_error.py \
        --seed 0 --device "$DEVICE" --ckpt "$CKPT"

    echo "Step 10: Figs. 1-3"
    submit s10 "s10 figs 1-3" "$PY" scripts/20_generate_figures.py --figs 1 2 3 \
        --select median --seed 0 --device "$DEVICE" --ckpt "$CKPT"

    [[ "$DRY_RUN" == 1 ]] || echo "Recorded in $IDS. Next: $0 post"
}

cmd_post() {
    [[ "$DRY_RUN" == 1 ]] && { echo "(dry run: post waits on real task IDs; nothing to do)"; return; }
    [[ -f "$IDS" ]] || die "no $IDS; run '$0 submit' first"

    echo "Waiting for steps 3 and 4 …"
    # shellcheck disable=SC2046
    wait_ok $(ids_for s3 s4) || die "a step 3/4 task failed; steps 7 and 11 not submitted"

    local post="set -e
$PY scripts/21_significance_tests.py --seeds $SEEDS
$PY scripts/20_generate_figures.py --figs 4 5 --fig-seed 0"
    echo "Steps 7 and 11: significance tests, Figs. 4-5"
    submit s7 "s7+s11 significance, figs 4-5" bash -c "$post"

    echo "Waiting for every remaining task …"
    local failed=0
    # shellcheck disable=SC2046
    wait_ok $(ids_for s5 s6 s7 s8 s9 s10) || failed=1

    echo "Consistency check: step 6 pretrained arm vs step 4 (must be 0) …"
    (cd "$REPO" && "$PY" - <<'EOF') || failed=1
import glob, sys
import numpy as np
fs = sorted(glob.glob('results/per_pair/tic_r_ablate_*.npz'))
if not fs:
    sys.exit('no ablation files found')
worst = 0.0
for f in fs:
    a, b = np.load(f)['pretrained'], np.load(f.replace('ablate', 'calib'))['warptransformer']
    if a.shape != b.shape:
        print(f'  MISMATCH {f}: {a.shape[0]} vs {b.shape[0]} sample pairs (files from different data)')
        worst = float('inf')
        continue
    d = float(abs(a - b).max())
    worst = max(worst, d)
    if d:
        print(f'  MISMATCH {f}: max |dr| = {d:.3g}')
print(f'  {len(fs)} runs, max |dr| = {worst}')
sys.exit(worst != 0)
EOF
    [[ "$failed" == 0 ]] || die "finished with failures or mismatches (above)"
    echo "All steps finished. Review results/ before committing (run.md, Hand-back)."
}

cmd_status() {
    [[ -f "$IDS" ]] || die "no $IDS"
    local id step label
    while IFS=$'\t' read -r id step label; do
        printf '%-6s %-5s %s\n' "$id" "$step" "$label"
    done < "$IDS"
    echo; "$TASK_BIN" -l
}

case "${1:-}" in
    preflight) cmd_preflight ;;
    check)     cmd_check ;;
    submit)    cmd_submit ;;
    post)      cmd_post ;;
    all)       cmd_submit; cmd_post ;;
    status)    cmd_status ;;
    *) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 1 ;;
esac
