"""
14_generate_figures.py — generate all five paper figures.

Figures
-------
  fig1_warp_functions.pdf        — warp function shapes for all four methods
  fig2_tic_overlays.pdf          — 5-panel raw-TIC alignment comparison with anchor ticks
  fig3_calibration_curve.pdf     — loss and TIC r vs calibration step (pretrained vs random)
  fig4_per_pair_distributions.pdf— per-pair TIC r violin plots across all batch pairs
  fig5_silhouette_scatter.pdf    — study silhouette vs TIC r scatter

Usage:
    python scripts/14_generate_figures.py               # all figures
    python scripts/14_generate_figures.py --figs 1 3 5  # specific figures
    python scripts/14_generate_figures.py \\
        --study-a fish_oil/batch_sep2015 \\
        --study-b fish_oil/batch_jul2016
"""

from __future__ import annotations

import argparse
import random
import sys
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path
from scipy.signal import correlate
from scipy.interpolate import PchipInterpolator
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from warp_transformer import (ChromaWarpTransformer, apply_warp, warp_smoothness_loss,
                              enable_cuda_determinism)
from alignment import (
    load_chromas, load_tic_raw, align_pair, detect_peaks, fingerprints, pooled_anchor_pchip,
    warp_chroma_2d, warp_tic, N_BINS, RUN_MIN, BIN_MIN, TIME_AXIS, PRECISION_CUTOFF,
)
import importlib.util as _ilu
_spec  = _ilu.spec_from_file_location('base05', Path(__file__).parent / '05_literature_baselines.py')
base05 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(base05)

DATA_DIR    = Path(__file__).parent.parent / 'data'
CKPT_DIR    = Path(__file__).parent.parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

COW_N_SEGS = 4;  COW_SLACK  = 22
ICO_N      = 10; ICO_SHIFT  = 20

N_CALIB_STEPS = 500;  LR_CALIB = 5e-4
LAMBDA_ANCHOR = 1.0;  LAMBDA_SMOOTH = 0.05;  N_REF_PAIRS = 3;  OUTLIER_IQR = 1.5

COLORS = {
    'Unaligned':         '#888888',
    'icoshift':          '#4477AA',
    'COW-TIC':           '#EE7733',
    'PCHIP (pooled anchors)': '#228833',
    'WarpTransformer':   '#CC3311',
}
BATCH_PAIRS = [
    ('fish_oil/batch_sep2015', 'fish_oil/batch_jul2016'),
    ('fish_oil/batch_sep2015', 'fish_oil/batch_apr2016'),
    ('fish_oil/batch_jan2016', 'fish_oil/batch_apr2016'),
    ('fish_oil/batch_jan2016', 'fish_oil/batch_jul2016'),
]
PAIR_LABELS = ['Sep15×Jul16', 'Sep15×Apr16', 'Jan16×Apr16', 'Jan16×Jul16']


def batch_name(label: str) -> str:
    """'batch_jul2016' → 'Jul 2016'."""
    s = label.replace('batch_', '')
    return f'{s[:3].capitalize()} {s[3:]}' if len(s) == 7 and s[3:].isdigit() else label


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    enable_cuda_determinism()

# ── Classical alignment ───────────────────────────────────────────────────────

def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def icoshift_align_detail(query_tic, ref_tic, n=ICO_N, ms=ICO_SHIFT):
    """Returns (warped, warp_fn, interval_mids, lags_min)."""
    N      = len(query_tic)
    warped = query_tic.copy().astype(np.float32)
    bounds = np.round(np.linspace(0, N, n + 1)).astype(int)
    best_lags, mids = [], []
    for i in range(n):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi - lo < 4:
            continue
        cc   = correlate(ref_tic[lo:hi].astype(np.float64),
                         query_tic[lo:hi].astype(np.float64), mode='full')
        lags = np.arange(-(hi - lo - 1), hi - lo)
        valid = np.abs(lags) <= ms
        lag  = int(lags[valid][np.argmax(cc[valid])])
        if lag != 0:
            src = np.arange(lo, hi) - lag
            warped[lo:hi] = query_tic[src.clip(0, N - 1)]
        best_lags.append(lag)
        mids.append((lo + hi) / 2 * BIN_MIN)
    lags_min = np.array(best_lags, dtype=float) * BIN_MIN
    _b = bounds.astype(float) * BIN_MIN
    def warp_fn(t, _b=_b, _l=lags_min):
        t   = np.asarray(t, dtype=float)
        idx = np.clip(np.searchsorted(_b[1:], t), 0, len(_l) - 1)
        return t + _l[idx]
    return warped, warp_fn, np.array(mids), lags_min


def cow_tic_align_detail(query_tic, ref_tic, n_segs=COW_N_SEGS, slack=COW_SLACK):
    """Returns (warped, warp_fn, r_nodes_min, q_nodes_min)."""
    N = len(ref_tic); T = N // n_segs
    r_nodes = [i * T for i in range(n_segs)] + [N - 1]
    cands = []
    for i, rn in enumerate(r_nodes):
        if i == 0 or i == n_segs:
            cands.append([int(rn)])
        else:
            cands.append(list(range(max(1, rn - slack), min(N - 2, rn + slack) + 1)))
    dp   = [dict() for _ in range(n_segs + 1)]
    back = [dict() for _ in range(n_segs + 1)]
    dp[0][0] = 0.0
    for seg in range(n_segs):
        r_s, r_e = r_nodes[seg], r_nodes[seg + 1]
        r_seg = ref_tic[r_s:r_e + 1]; r_len = len(r_seg)
        for q_s, cum in dp[seg].items():
            for q_e in cands[seg + 1]:
                if q_e <= q_s:
                    continue
                q_raw = query_tic[q_s:q_e + 1]
                if len(q_raw) < 2:
                    continue
                q_seg = np.interp(np.linspace(0, 1, r_len),
                                  np.linspace(0, 1, len(q_raw)), q_raw)
                nc = cum + _pearson(r_seg, q_seg)
                if nc > dp[seg + 1].get(q_e, -np.inf):
                    dp[seg + 1][q_e] = nc; back[seg + 1][q_e] = q_s
    q_last = r_nodes[-1]
    if q_last not in back[n_segs]:
        return query_tic.astype(np.float32), lambda t: t, np.array(r_nodes)*BIN_MIN, np.array(r_nodes)*BIN_MIN
    q_rev = [q_last]
    for seg in range(n_segs, 0, -1):
        p = back[seg].get(q_rev[-1])
        if p is None:
            return query_tic.astype(np.float32), lambda t: t, np.array(r_nodes)*BIN_MIN, np.array(r_nodes)*BIN_MIN
        q_rev.append(p)
    q_nodes = list(reversed(q_rev))
    warped = np.empty(N, dtype=np.float32)
    for seg in range(n_segs):
        r_s, r_e = r_nodes[seg], r_nodes[seg + 1]; r_len = r_e - r_s + 1
        q_raw = query_tic[q_nodes[seg]:q_nodes[seg + 1] + 1]
        warped[r_s:r_e + 1] = (
            np.interp(np.linspace(0, 1, r_len), np.linspace(0, 1, len(q_raw)), q_raw)
            if len(q_raw) >= 2 else np.full(r_len, float(q_raw[0]) if q_raw.size else 0.0)
        )
    r_times = np.array(r_nodes, dtype=float) * BIN_MIN
    q_times = np.array(q_nodes, dtype=float) * BIN_MIN
    warp_fn = lambda t, _r=r_times, _q=q_times: np.interp(t, _r, _q)
    return warped, warp_fn, r_times, q_times


# ── WarpTransformer calibration ───────────────────────────────────────────────

def extract_anchors(chromas_a, chromas_b):
    rq = defaultdict(list)
    for i in range(min(N_REF_PAIRS, len(chromas_a))):
        _, _, _, _, _, ancs = align_pair(chromas_b[0], chromas_a[i],
                                         encode_fn=None, return_anchors=True)
        for r, q in ancs:
            rq[r].append(q)
    if not rq:
        return []
    agg    = [(r, int(np.round(np.median(q)))) for r, q in sorted(rq.items())]
    drifts = np.array([q - r for r, q in agg], dtype=float)
    mean, std = float(np.mean(drifts)), max(float(np.std(drifts)), 1.0)
    return [a for a, k in zip(agg, np.abs(drifts - mean) <= OUTLIER_IQR * std) if k]


def anchor_loss_fn(warps, study_ids, anchors, device):
    if not anchors:
        return torch.tensor(0.0, device=device)
    ref_mean = warps[study_ids == 0].mean(dim=0)
    qw = warps[study_ids == 1]
    loss = torch.tensor(0.0, device=device)
    for r, q in anchors:
        loss = loss + (qw[:, r] - ref_mean[r] - float(q - r)).pow(2).mean()
    return loss / len(anchors)


def calibrate(chromas_a, chromas_b, device, pretrained=True, log_every=5, seed=None,
              ckpt=None):
    """
    Returns (aligned_a, aligned_b, warp_np, anchors, log) where log is a list
    of dicts {step, loss, tic_r} recorded every log_every steps. With a seed, the
    random init (if any) and the calibration loop are seeded as in scripts 06 and 08.
    """
    n_a, n_b = len(chromas_a), len(chromas_b)
    study_ids = torch.tensor([0]*n_a + [1]*n_b, dtype=torch.long, device=device)
    chromas_t = torch.from_numpy(np.stack(chromas_a + chromas_b)).to(device)

    anchors = extract_anchors(chromas_a, chromas_b)

    if seed is not None and not pretrained:
        _set_seed(seed)
    model = ChromaWarpTransformer()
    if pretrained:
        model.load_state_dict(torch.load(ckpt or CKPT_DIR / 'warp_transformer.pt',
                                          map_location=device))
    model.to(device)

    log = []
    if anchors:
        if seed is not None:
            _set_seed(seed)
        model.train()
        opt   = optim.Adam(model.parameters(), lr=LR_CALIB)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_CALIB_STEPS)
        best_loss, best_state = float('inf'), {k: v.clone() for k, v in model.state_dict().items()}

        for step in range(1, N_CALIB_STEPS + 1):
            aligned, warps = model(chromas_t, study_ids)
            a_loss = anchor_loss_fn(warps, study_ids, anchors, device)
            s_loss = warp_smoothness_loss(warps)
            loss   = LAMBDA_ANCHOR * a_loss + LAMBDA_SMOOTH * s_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if float(loss) < best_loss:
                best_loss, best_state = float(loss), {k: v.clone() for k, v in model.state_dict().items()}

            if step % log_every == 0 or step == N_CALIB_STEPS:
                with torch.no_grad():
                    al_d = aligned.detach()
                    al_a = [al_d[i].cpu().numpy() for i in range(n_a)]
                    al_b = [al_d[i].cpu().numpy() for i in range(n_a, n_a + n_b)]
                    r_vals = [float(pearsonr(a.sum(1), b.sum(1))[0])
                              for a in al_a for b in al_b]
                    log.append({'step': step, 'loss': float(loss),
                                'tic_r': float(np.mean(r_vals))})

        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        aligned_t, warps_t = model(chromas_t, study_ids)
    np_al = aligned_t.cpu().numpy()
    return (np_al[:n_a], np_al[n_a:], warps_t.cpu().numpy(), anchors, log)


# ── Pairwise TIC r helpers ────────────────────────────────────────────────────

def pairwise_r(chromas_a, chromas_b, align_fn=None):
    """Return per-pair TIC r list. align_fn(q_tic, r_tic) → warped_tic or None for unaligned."""
    cors = []
    for qa in chromas_a:
        q_tic = qa.sum(axis=1)
        for rb in chromas_b:
            r_tic = rb.sum(axis=1)
            if align_fn is None:
                w = q_tic
            else:
                w = align_fn(q_tic, r_tic)
            cors.append(float(pearsonr(r_tic.astype(np.float64),
                                       w.astype(np.float64))[0]))
    return cors


def ico_fn(q, r):
    warped, _, _, _ = icoshift_align_detail(q, r)
    return warped


def cow_fn(q, r):
    warped, _, _, _ = cow_tic_align_detail(q, r)
    return warped


# ── Figure 1: Warp function comparison ────────────────────────────────────────

def fig1_warp_functions(chromas_a, chromas_b, warp_np, n_a, anchors, out, ai=0,
                        name_a='Sep 2015', name_b='Jul 2016'):
    """Single-panel plot of all four warp functions on one representative pair (A[ai], B[0]).
    The WarpTransformer curve is B[0]'s warp into the shared frame, independent of ai."""
    ref_chroma, qry_chroma = chromas_a[ai], chromas_b[0]
    ref_tic = ref_chroma.sum(axis=1)
    qry_tic = qry_chroma.sum(axis=1)

    # --- icoshift ---
    _, ico_fn_w, ico_mids, ico_lags = icoshift_align_detail(qry_tic, ref_tic)
    ico_curve = ico_fn_w(TIME_AXIS)
    ico_dots_x = ico_mids
    ico_dots_y = ico_fn_w(ico_mids)

    # --- COW-TIC ---
    _, cow_fn_w, cow_r, cow_q = cow_tic_align_detail(qry_tic, ref_tic)
    cow_curve = cow_fn_w(TIME_AXIS)

    # --- PCHIP through the pooled calibration anchors (the WarpTransformer's anchors) ---
    # pooled_anchor_pchip maps B time → A time; this plot is A (reference) → B (query), so
    # the same monotone spline is fitted through the same anchors in that direction.
    pchip_fn_ba, used = pooled_anchor_pchip(anchors)
    if pchip_fn_ba is not None:
        r_t = np.array([r for r, _ in used], dtype=float) * BIN_MIN
        q_t = np.array([q for _, q in used], dtype=float) * BIN_MIN
        pchip_curve = PchipInterpolator(np.concatenate([[0.0], r_t, [RUN_MIN]]),
                                        np.concatenate([[0.0], q_t, [RUN_MIN]]))(TIME_AXIS)
    else:
        pchip_curve = TIME_AXIS.copy()

    # --- WarpTransformer: use warp for B[0] (study_ids=1, index n_a) ---
    wt_curve = warp_np[n_a, :] * BIN_MIN   # B[0]'s warp → consensus ≈ A frame
    anc_r = np.array([r * BIN_MIN for r, _ in anchors])
    anc_q = np.array([q * BIN_MIN for _, q in anchors])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(TIME_AXIS, TIME_AXIS, '--', color='#bbbbbb', lw=1.2,
            label='Identity (no drift)', zorder=1)

    # icoshift staircase
    ax.step(TIME_AXIS, ico_curve, where='post',
            color=COLORS['icoshift'], lw=1.6, label='icoshift', zorder=3)
    ax.scatter(ico_dots_x, ico_dots_y, color=COLORS['icoshift'],
               s=35, zorder=5, marker='D', linewidths=0)

    # COW-TIC piecewise linear
    ax.plot(TIME_AXIS, cow_curve,
            color=COLORS['COW-TIC'], lw=1.6, label='COW-TIC', zorder=3)
    ax.scatter(cow_r, cow_q, color=COLORS['COW-TIC'],
               s=50, zorder=5, linewidths=0)

    # PCHIP smooth cubic through the pooled anchors (drawn below with the WarpTransformer's)
    ax.plot(TIME_AXIS, pchip_curve,
            color=COLORS['PCHIP (pooled anchors)'], lw=1.6, label='PCHIP (pooled anchors)', zorder=3)

    # WarpTransformer smooth monotone
    ax.plot(TIME_AXIS, wt_curve,
            color=COLORS['WarpTransformer'], lw=2.0,
            label='WarpTransformer', zorder=4)
    if len(anc_r):
        ax.scatter(anc_r, anc_q, color='#222222', s=50, zorder=6, linewidths=0,
                   label='Pooled anchors (PCHIP and WarpTransformer)')

    ax.set_xlabel(f'Retention time, {name_a} sample {ai} (reference) (min)', fontsize=10)
    ax.set_ylabel(f'Source retention time, {name_b} sample 0 (query) (min)', fontsize=10)
    ax.set_title(f'Warp functions — {name_a} vs {name_b}', fontsize=11)
    ax.set_xlim(0, RUN_MIN); ax.set_ylim(0, RUN_MIN)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.85)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig1 → {out}")


# ── Figure 2: TIC alignment overlays with anchor ticks ───────────────────────

def fig2_tic_overlays(raw_a, raw_b, chromas_a, chromas_b, warp_np, n_a, anchors, qi, out,
                      name_a='Sep 2015', name_b='Jul 2016'):
    """5-panel overlay of the raw TIC (max-normalised for display) of A[qi] against B[0].

    Every warp is the one each method computes as usual (icoshift and COW-TIC optimised on the
    normalised TIC, PCHIP through the pooled anchors, the calibrated WarpTransformer's per-sample
    warps); it is then applied to the raw TIC, and r is Pearson r on the raw TIC.
    warp_np: [N, T] WarpTransformer warps (source bins), A samples first."""
    qry_raw, ref_raw = raw_a[qi], raw_b[0]
    qry_tic, ref_tic = chromas_a[qi].sum(axis=1), chromas_b[0].sum(axis=1)

    def norm(t): mx = t.max(); return t / mx if mx > 0 else t

    ico_warped = base05.icoshift_align(qry_tic, ref_tic, raw_query=qry_raw)[3]
    cow_warped = base05.cow_tic_align(qry_tic, ref_tic, raw_query=qry_raw)[3]
    pchip_fn_ba, _ = pooled_anchor_pchip(anchors)            # B time → A time
    pchip_warped   = warp_tic(qry_raw, pchip_fn_ba)
    wt_w = apply_warp(torch.from_numpy(np.stack([qry_raw, ref_raw])[:, :, None]),
                      torch.from_numpy(warp_np[[qi, n_a]]))[:, :, 0].numpy()
    wt_qry, wt_ref = wt_w[0], wt_w[1]

    def r(x, y): return float(pearsonr(x.astype(np.float64), y.astype(np.float64))[0])

    # Anchor ticks: the same pooled anchors in both panels, at their position on each panel's
    # time axis (PCHIP panel = batch-B frame → q; WarpTransformer panel = consensus ≈ A frame → r)
    pchip_ticks_r = [q * BIN_MIN for _, q in anchors]
    calib_ticks_r = [r_ * BIN_MIN for r_, _ in anchors]

    panels = [
        ('Unaligned',              qry_raw,      ref_raw, []),
        ('icoshift',               ico_warped,   ref_raw, []),
        ('COW-TIC',                cow_warped,   ref_raw, []),
        ('PCHIP (pooled anchors)', pchip_warped, ref_raw, pchip_ticks_r),
        ('WarpTransformer',        wt_qry,       wt_ref,  calib_ticks_r),
    ]
    panels = [(m, norm(q), norm(rf), r(q, rf), t) for m, q, rf, t in panels]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.0), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.06, left=0.05, right=0.99, top=0.86, bottom=0.16)

    for ax, (method, qry, ref, rv, ticks_r) in zip(axes, panels):
        color = COLORS.get(method, '#333333')
        ax.fill_between(TIME_AXIS, ref, alpha=0.18, color='#444444', linewidth=0)
        ax.plot(TIME_AXIS, ref, color='#555555', lw=0.9)
        ax.plot(TIME_AXIS, qry, color=color, lw=1.1, alpha=0.9)
        for t in ticks_r:
            ax.axvline(t, ymin=0, ymax=0.08, color=color, lw=1.4, alpha=0.85)

        ax.set_title(f'{method}\n$r={rv:.3f}$', fontsize=8.5, pad=4)
        ax.set_xlim(0, RUN_MIN); ax.set_ylim(-0.04, 1.2)
        ax.set_xlabel('RT (min)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('TIC (scaled)', fontsize=8)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0],[0], color='#555555', lw=1.4, label=f'{name_b} sample 0'),
        Line2D([0],[0], color='#333333', lw=1.4, label=f'{name_a} sample {qi} (aligned)'),
        Line2D([0],[0], color='#333333', lw=1.4, label='Ticks: pooled calibration anchors'),
    ], fontsize=7.5, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), framealpha=0.8)

    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig2 per-panel r on the raw TIC ({name_a} sample {qi} vs {name_b} sample 0): " +
          ', '.join(f"{m} {rv:.3f}" for m, _, _, rv, _ in panels))
    print(f"  fig2 → {out}")


# ── Figure 3: Calibration learning curve ─────────────────────────────────────

def fig3_calibration_curve(log_pre, log_rand, unaligned_r, out):
    """Two-panel: loss (left) and TIC r (right), pretrained vs random-init."""
    steps_p = [d['step'] for d in log_pre]
    loss_p  = [d['loss'] for d in log_pre]
    r_p     = [d['tic_r'] for d in log_pre]
    steps_r = [d['step'] for d in log_rand]
    loss_r  = [d['loss'] for d in log_rand]
    r_r     = [d['tic_r'] for d in log_rand]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(wspace=0.28, left=0.09, right=0.97, top=0.88, bottom=0.14)

    col_pre  = COLORS['WarpTransformer']
    col_rand = '#888888'

    ax1.plot(steps_p, loss_p,  color=col_pre,  lw=1.8, label='Pretrained init')
    ax1.plot(steps_r, loss_r,  color=col_rand, lw=1.8, linestyle='--', label='Random init')
    ax1.set_xlabel('Calibration step', fontsize=9)
    ax1.set_ylabel('Loss (anchor MSE + smoothness)', fontsize=9)
    ax1.set_title('Calibration loss', fontsize=10)
    ax1.legend(fontsize=8.5)
    ax1.tick_params(labelsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.axhline(unaligned_r, color='#bbbbbb', lw=1.2, linestyle=':', label='Unaligned')
    ax2.plot(steps_p, r_p,  color=col_pre,  lw=1.8, label='Pretrained init')
    ax2.plot(steps_r, r_r,  color=col_rand, lw=1.8, linestyle='--', label='Random init')
    ax2.set_xlabel('Calibration step', fontsize=9)
    ax2.set_ylabel('Mean pairwise TIC r', fontsize=9)
    ax2.set_title('Alignment quality (TIC r)', fontsize=10)
    ax2.legend(fontsize=8.5)
    ax2.tick_params(labelsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Anchor calibration convergence — Sep 2015 × Jul 2016', fontsize=10, y=0.97)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig3 → {out}")


# ── Figure 4: Per-pair TIC r violin plots ────────────────────────────────────

def fig4_violin(all_data, out):
    """
    all_data: dict of method_name → dict of pair_label → list of TIC r values
    """
    methods = ['Unaligned', 'icoshift', 'COW-TIC', 'PCHIP (pooled anchors)', 'WarpTransformer']
    n_pairs = len(PAIR_LABELS)

    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.08, right=0.99, top=0.88, bottom=0.18)

    positions = np.arange(1, len(methods) + 1)

    for col, (ax, pair_label) in enumerate(zip(axes, PAIR_LABELS)):
        data  = [all_data[m][pair_label] for m in methods]
        parts = ax.violinplot(data, positions=positions, showmedians=True,
                              showextrema=True)
        for body, method in zip(parts['bodies'], methods):
            body.set_facecolor(COLORS[method])
            body.set_alpha(0.65)
            body.set_linewidth(0)
        parts['cmedians'].set_color('#222222'); parts['cmedians'].set_linewidth(1.5)
        parts['cmins'].set_color('#555555');    parts['cmaxes'].set_color('#555555')
        parts['cbars'].set_color('#555555')

        ax.set_xticks(positions)
        ax.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=9)
        ax.set_title(pair_label, fontsize=10)
        ax.tick_params(axis='y', labelsize=9)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('Pairwise TIC r', fontsize=10)
    fig.suptitle('Per-pair TIC r distributions across batch pairs', fontsize=10, y=0.97)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  fig4 → {out}")


# ── Figure 5: Silhouette vs TIC r scatter ────────────────────────────────────

def _per_pair_files(sa, sb, fig_seed):
    """Saved per-pair arrays for one batch pair: baselines (05), calibration (06), PCHIP (07)."""
    la, lb = Path(sa).name, Path(sb).name
    d = RESULTS_DIR / 'per_pair'
    base  = np.load(d / f'tic_r_baselines_{la}_{lb}.npz')
    calib = np.load(d / f'tic_r_calib_{la}_{lb}_seed{fig_seed}.npz')
    pchip = np.load(d / f'tic_r_pchip_pooled_{la}_{lb}.npz')
    if not np.allclose(base['unaligned'], calib['unaligned'], atol=1e-6):
        raise ValueError(f'Unaligned arrays differ between scripts 05 and 06 for {la} × {lb}')
    if pchip['pchip_pooled'].shape != calib['warptransformer'].shape:
        raise ValueError(f'PCHIP arrays (script 07) do not match the calibration run for {la} × {lb}')
    return base, calib, pchip


def load_per_pair_r(fig_seed):
    """method → pair_label → per-pair TIC r array, from saved results (scripts 05, 06)."""
    all_data = {m: {} for m in
                ['Unaligned', 'icoshift', 'COW-TIC', 'PCHIP (pooled anchors)', 'WarpTransformer']}
    for (sa, sb), pl in zip(BATCH_PAIRS, PAIR_LABELS):
        base, calib, pchip = _per_pair_files(sa, sb, fig_seed)
        all_data['Unaligned'][pl]        = base['unaligned']
        all_data['icoshift'][pl]         = base['icoshift']
        all_data['COW-TIC'][pl]          = base['cow_tic']
        all_data['PCHIP (pooled anchors)'][pl] = pchip['pchip_pooled']
        all_data['WarpTransformer'][pl]  = calib['warptransformer']
    return all_data


def fig5_scatter(out, fig_seed=0):
    """Read from saved results. x = Δ TIC r, y = Δ Study sil (both vs unaligned)."""
    # (method, pair_label, tic_r, study_sil, unaligned_r, unaligned_sil)
    data = []
    for (sa, sb), pl in zip(BATCH_PAIRS, PAIR_LABELS):
        base, calib, pchip = _per_pair_files(sa, sb, fig_seed)
        r0 = float(np.mean(calib['unaligned']))
        sil0, _, sil_wt = (float(v) for v in calib['sil'])
        sil_pchip       = float(pchip['sil'])
        _, sil_cow, sil_ico     = (float(v) for v in base['sil'])
        data += [
            ('icoshift',         pl, float(np.mean(base['icoshift'])),           sil_ico,   r0, sil0),
            ('PCHIP (pooled anchors)', pl, float(np.mean(pchip['pchip_pooled'])), sil_pchip, r0, sil0),
            ('COW-TIC',          pl, float(np.mean(base['cow_tic'])),            sil_cow,   r0, sil0),
            ('WarpTransformer',  pl, float(np.mean(calib['warptransformer'])), sil_wt,    r0, sil0),
        ]
        print(f"    {pl}: unaligned r={r0:.3f} sil={sil0:.3f}")
    markers = {'Sep15×Jul16': 'o', 'Sep15×Apr16': 'D', 'Jan16×Apr16': 's', 'Jan16×Jul16': '^'}

    fig, ax = plt.subplots(figsize=(7.6, 5))
    from matplotlib.lines import Line2D

    plotted_methods = set(); plotted_pairs = set()
    for method, pair, r, sil, r0, sil0 in data:
        delta_r   = r - r0
        delta_sil = sil - sil0
        color  = COLORS[method]
        marker = markers[pair]
        ax.scatter(delta_r, delta_sil, color=color, marker=marker,
                   s=80, zorder=4, edgecolors='white', linewidths=0.5)
        plotted_methods.add(method); plotted_pairs.add(pair)

    ax.axhline(0, color='#aaaaaa', lw=0.9, linestyle='--', zorder=1)
    ax.axvline(0, color='#aaaaaa', lw=0.9, linestyle='--', zorder=1)

    method_handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=COLORS[m],
                             markersize=9, label=m) for m in
                      ['icoshift','PCHIP (pooled anchors)','COW-TIC','WarpTransformer']]
    pair_handles   = [Line2D([0],[0], marker=markers[p], color='w', markerfacecolor='#555555',
                             markersize=9, label=p) for p in PAIR_LABELS]
    leg1 = ax.legend(handles=method_handles, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                     title='Method', title_fontsize=8, framealpha=0.85)
    ax.add_artist(leg1)
    ax.legend(handles=pair_handles, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 0.55),
              title='Batch pair', title_fontsize=8, framealpha=0.85)

    ax.set_xlabel('Δ TIC r  (vs unaligned)', fontsize=10)
    ax.set_ylabel('Δ Study silhouette  (vs unaligned)\n← better integration      worse →', fontsize=9)
    ax.set_title('Alignment quality vs batch integration', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight', bbox_extra_artists=[leg1])
    plt.close(fig)
    print(f"  fig5 → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--study-a', default='fish_oil/batch_sep2015')
    ap.add_argument('--study-b', default='fish_oil/batch_jul2016')
    ap.add_argument('--figs', nargs='+', type=int, default=[1,2,3,4,5],
                    metavar='N', help='Which figures to generate (default: all)')
    ap.add_argument('--select', choices=['best', 'median'], default='best',
                    help="Batch-A sample shown in Figs 1-2: largest ('best') or median "
                         "WarpTransformer TIC r improvement over unaligned")
    ap.add_argument('--seed', type=int, default=None,
                    help='seed calibration as in scripts 06/08 (with --device cpu, repeatable)')
    ap.add_argument('--device', default='auto', help='auto | cpu | mps | cuda')
    ap.add_argument('--ckpt', type=Path, default=CKPT_DIR / 'warp_transformer.pt')
    ap.add_argument('--fig-seed', type=int, default=0,
                    help='calibration seed whose saved per-pair arrays Figs 4-5 use')
    args = ap.parse_args()

    want = set(args.figs)
    label_a, label_b = Path(args.study_a).name, Path(args.study_b).name
    seed, ckpt = args.seed, args.ckpt

    if args.device != 'auto':
        device = torch.device(args.device)
    else:
        device = (torch.device('cuda') if torch.cuda.is_available()
                  else torch.device('mps') if torch.backends.mps.is_available()
                  else torch.device('cpu'))
    print(f"Device: {device}   seed: {seed}   checkpoint: {ckpt}")

    # Load primary dataset
    print(f"\nLoading {label_a} × {label_b} …")
    chromas_a = load_chromas(DATA_DIR / args.study_a / 'chroma')
    chromas_b = load_chromas(DATA_DIR / args.study_b / 'chroma')
    n_a, n_b  = len(chromas_a), len(chromas_b)
    print(f"  {label_a}: {n_a}  |  {label_b}: {n_b}")

    # Calibrate pretrained model (needed for fig 1, 2, 3)
    need_calib = want & {1, 2, 3}
    aligned_a = aligned_b = warp_full = anchors = log_pre = None
    if need_calib:
        print("\nCalibrating (pretrained) …")
        aligned_a, aligned_b, warp_full, anchors, log_pre = calibrate(
            chromas_a, chromas_b, device, pretrained=True, log_every=5, seed=seed, ckpt=ckpt)
        r_unaligned = float(np.mean([
            float(pearsonr(a.sum(1), b.sum(1))[0])
            for a in chromas_a for b in chromas_b
        ]))
        print(f"  Unaligned TIC r = {r_unaligned:.4f}")
        print(f"  Post-calib TIC r = {log_pre[-1]['tic_r']:.4f}")

    # Calibrate random-init model (needed for fig 3)
    log_rand = None
    if 3 in want:
        print("\nCalibrating (random init) …")
        _, _, _, _, log_rand = calibrate(
            chromas_a, chromas_b, device, pretrained=False, log_every=5, seed=seed)

    # Representative query sample: largest or median Δ TIC r (WT vs unaligned, against B[0])
    qi = 0
    tag = '' if args.select == 'best' else '_median'
    if need_calib:
        ref_orig   = chromas_b[0].sum(axis=1).astype(np.float64)
        ref_warped = aligned_b[0].sum(axis=1).astype(np.float64)
        deltas = np.array([
            float(pearsonr(aligned_a[i].sum(1).astype(np.float64), ref_warped)[0])
            - float(pearsonr(chromas_a[i].sum(1).astype(np.float64), ref_orig)[0])
            for i in range(n_a)])
        order = np.argsort(deltas)
        qi = int(order[-1]) if args.select == 'best' else int(order[(n_a - 1) // 2])
        ids = [p.stem for p in sorted((DATA_DIR / args.study_a / 'chroma').glob('*.npz'))]
        print(f"\nRepresentative query ({args.select}): {label_a}[{qi}] = {ids[qi]}  "
              f"Δ TIC r = {deltas[qi]:+.3f}  (rank {int(np.where(order == qi)[0][0]) + 1} of {n_a} "
              f"from lowest; median Δ = {np.median(deltas):+.3f}, max Δ = {deltas.max():+.3f})")
        sel_lines = [f"Figs 1-2 sample selection ({args.select}) for {label_a} × {label_b}, "
                     f"seed {seed}, device {device}",
                     f"chosen: index {qi}  id {ids[qi]}  Δ TIC r {deltas[qi]:+.4f}",
                     f"median Δ {np.median(deltas):+.4f}   max Δ {deltas.max():+.4f}   "
                     f"min Δ {deltas.min():+.4f}", "",
                     "all samples (index, id, Δ TIC r = r(WT A_i, WT B0) − r(A_i, B0)):"]
        sel_lines += [f"  {i:>3}  {ids[i]:<36} {deltas[i]:+.4f}" for i in order[::-1]]
        (RESULTS_DIR / f'fig_selection_{label_a}_{label_b}_{args.select}.txt').write_text(
            '\n'.join(sel_lines) + '\n')

    # ── Figure 1
    if 1 in want:
        print("\nGenerating Figure 1 (warp functions) …")
        fig1_warp_functions(
            chromas_a, chromas_b, warp_full, n_a, anchors,
            out=RESULTS_DIR / f'fig1_warp_functions_{label_a}_{label_b}{tag}.pdf',
            ai=qi if args.select == 'median' else 0,
            name_a=batch_name(label_a), name_b=batch_name(label_b),
        )

    # ── Figure 2
    if 2 in want:
        print("\nGenerating Figure 2 (TIC overlays) …")

        fig2_tic_overlays(
            load_tic_raw(DATA_DIR / args.study_a / 'chroma'),
            load_tic_raw(DATA_DIR / args.study_b / 'chroma'),
            chromas_a, chromas_b, warp_full, n_a, anchors, qi,
            out=RESULTS_DIR / f'fig2_tic_overlays_{label_a}_{label_b}{tag}.pdf',
            name_a=batch_name(label_a), name_b=batch_name(label_b),
        )

    # ── Figure 3
    if 3 in want:
        print("\nGenerating Figure 3 (calibration curve) …")
        fig3_calibration_curve(
            log_pre, log_rand, r_unaligned,
            out=RESULTS_DIR / f'fig3_calibration_curve_{label_a}_{label_b}.pdf'
        )

    # ── Figure 4: pairwise TIC r distributions across all 3 batch pairs
    if 4 in want:
        print(f"\nGenerating Figure 4 (violin plots, seed {args.fig_seed} arrays) …")
        all_data = load_per_pair_r(args.fig_seed)

        fig4_violin(all_data,
                    out=RESULTS_DIR / 'fig4_per_pair_distributions.pdf')

    # ── Figure 5
    if 5 in want:
        print("\nGenerating Figure 5 (silhouette scatter) …")
        fig5_scatter(out=RESULTS_DIR / 'fig5_silhouette_scatter.pdf', fig_seed=args.fig_seed)

    print("\nDone.")


if __name__ == '__main__':
    main()
