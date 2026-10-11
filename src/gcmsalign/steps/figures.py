"""Step 14 — the paper's figures.

  fig1_warp_functions_<A>_<B>[_median].pdf  warp functions of all methods for one sample pair
  fig2_tic_overlays_<A>_<B>[_median].pdf    raw-TIC overlays, one panel per method
  fig3_calibration_curve_<A>_<B>.pdf        loss and TIC r during calibration, both init arms
  fig4_per_pair_distributions.pdf           per-pair TIC r violins, all evaluated pairs
  fig5_silhouette_scatter.pdf               Δ study silhouette vs Δ TIC r
Figs 1–3 calibrate the WarpTransformer themselves (step 06's settings); Figs 4–5 read the saved
arrays of steps 05–07 (--fig-seed).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from ..anchors import extract_anchors
from ..baselines import cow_tic_align, icoshift_align
from ..calibration import batch_tensors, calibrate, load_model, pick_device, predict
from ..config import (
    BIN_MIN,
    EVAL_PAIRS,
    LAST_MIN,
    N_CALIB_STEPS,
    TIME_AXIS,
    Paths,
    batch_name,
    pair_label,
)
from ..data import Batch, load_batch
from ..metrics import pearson, pearson_rows, tic_r_pairs
from ..warps import monotone_interp, pchip_warp, src_from_fn, warp_rows, warp_signal
from ._common import (
    add_model_args,
    add_pair_args,
    batch_name_of,
    ckpt_or_default,
    parser,
    setup_logging,
)

log = logging.getLogger('gcmsalign')

COLORS = {
    'Unaligned': '#888888',
    'icoshift': '#4477AA',
    'COW-TIC': '#EE7733',
    'PCHIP (pooled anchors)': '#228833',
    'WarpTransformer': '#CC3311',
}
METHODS = list(COLORS)


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Calibration for Figs 1–3 ──────────────────────────────────────────────────


def calibrate_pair(
    a: Batch,
    b: Batch,
    anchors,
    ckpt: Path | None,
    device: torch.device,
    seed: int,
    steps: int = N_CALIB_STEPS,
    log_every: int = 0,
):
    """Calibrated (aligned, warps, history) for the pretrained (ckpt) or random-init (None) model;
    history records the normalised TIC r every log_every steps."""
    model = load_model(ckpt, device, seed=seed)
    x, sid = batch_tensors(a.chromas, b.chromas, device)
    n_a = len(a)

    def on_log(step, rec, aligned):
        tic = aligned.sum(-1).cpu().numpy()
        rec['tic_r'] = float(pearson_rows(tic[:n_a], tic[n_a:]).mean())

    clog = calibrate(
        model, x, sid, anchors, seed=seed, steps=steps, log_every=log_every, on_log=on_log
    )
    aligned, warps = predict(model, x, sid)
    return aligned, warps, clog.history


def select_sample(a, b, aligned, select: str) -> tuple[int, np.ndarray]:
    """Index of the batch-A sample with the largest ('best') or median Δ TIC r of the WT over
    unaligned, against B[0]; and all the Δ values."""
    n_a = len(a)
    al_tic = aligned.sum(-1)
    deltas = np.array(
        [pearson(al_tic[i], al_tic[n_a]) - pearson(a.tic[i], b.tic[0]) for i in range(n_a)]
    )
    order = np.argsort(deltas)
    return int(order[-1] if select == 'best' else order[(n_a - 1) // 2]), deltas


# ── Figure 1: warp functions ──────────────────────────────────────────────────


def fig1_warp_functions(a, b, warps, anchors, ai, out):
    """Reference = A[ai] (x axis), query = B[0] (y axis); every curve maps A time → B time."""
    n_a = len(a)
    ico = icoshift_align(b.tic[0], a.tic[ai])
    cow = cow_tic_align(b.tic[0], a.tic[ai])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(
        TIME_AXIS, TIME_AXIS, '--', color='#bbbbbb', lw=1.2, label='Identity (no drift)', zorder=1
    )
    ax.step(
        TIME_AXIS,
        ico.src * BIN_MIN,
        where='mid',
        color=COLORS['icoshift'],
        lw=1.6,
        label='icoshift',
        zorder=3,
    )
    ax.plot(
        TIME_AXIS, cow.src * BIN_MIN, color=COLORS['COW-TIC'], lw=1.6, label='COW-TIC', zorder=3
    )
    ax.scatter(
        np.array(cow.details['r_nodes']) * BIN_MIN,
        np.array(cow.details['q_nodes']) * BIN_MIN,
        color=COLORS['COW-TIC'],
        s=50,
        zorder=5,
        linewidths=0,
    )
    if anchors:
        r = np.array([x for x, _ in anchors], float) * BIN_MIN
        q = np.array([y for _, y in anchors], float) * BIN_MIN
        ax.plot(
            TIME_AXIS,
            monotone_interp(r, q)(TIME_AXIS),
            color=COLORS['PCHIP (pooled anchors)'],
            lw=1.6,
            label='PCHIP (pooled anchors)',
            zorder=3,
        )
    # WarpTransformer: consensus bin t ↦ (mean batch-A source position, B[0] source position)
    ax.plot(
        warps[:n_a].mean(0) * BIN_MIN,
        warps[n_a] * BIN_MIN,
        color=COLORS['WarpTransformer'],
        lw=2.0,
        label='WarpTransformer',
        zorder=4,
    )
    if anchors:
        ax.scatter(
            r,
            q,
            color='#222222',
            s=50,
            zorder=6,
            linewidths=0,
            label='Pooled anchors (PCHIP and WarpTransformer)',
        )
    na, nb = batch_name(a.name), batch_name(b.name)
    ax.set_xlabel(f'Retention time, {na} sample {ai} (reference) (min)', fontsize=10)
    ax.set_ylabel(f'Retention time, {nb} sample 0 (query) (min)', fontsize=10)
    ax.set_title(f'Warp functions — {na} vs {nb}', fontsize=11)
    ax.set_xlim(0, LAST_MIN)
    ax.set_ylim(0, LAST_MIN)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.85)
    ax.tick_params(labelsize=8)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info('  fig1 → %s', out)


# ── Figure 2: raw-TIC overlays ────────────────────────────────────────────────


def fig2_tic_overlays(a, b, warps, anchors, qi, out) -> dict[str, float]:
    """Raw TIC (max-normalised for display) of A[qi] against B[0], one panel per method. Each
    method's own warp, applied to the raw TIC; r = raw-TIC Pearson r on the evaluation window."""
    n_a = len(a)
    q_raw, r_raw = a.tic_raw[qi], b.tic_raw[0]
    q_tic, r_tic = a.tic[qi], b.tic[0]
    wt = warp_rows(np.stack([q_raw, r_raw]), warps[[qi, n_a]])
    panels = {
        'Unaligned': (q_raw, r_raw, []),
        'icoshift': (warp_signal(q_raw, icoshift_align(q_tic, r_tic).src), r_raw, []),
        'COW-TIC': (warp_signal(q_raw, cow_tic_align(q_tic, r_tic).src), r_raw, []),
        'PCHIP (pooled anchors)': (
            warp_signal(q_raw, src_from_fn(pchip_warp(anchors))),
            r_raw,
            [q * BIN_MIN for _, q in anchors],
        ),
        'WarpTransformer': (wt[0], wt[1], []),
    }
    r_vals = {m: pearson(q, r) for m, (q, r, _) in panels.items()}

    def norm(t):
        return t / t.max() if t.max() > 0 else t

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.0), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.06, left=0.05, right=0.99, top=0.86, bottom=0.16)
    for ax, (m, (q, r, ticks)) in zip(axes, panels.items(), strict=True):
        ax.fill_between(TIME_AXIS, norm(r), alpha=0.18, color='#444444', linewidth=0)
        ax.plot(TIME_AXIS, norm(r), color='#555555', lw=0.9)
        ax.plot(TIME_AXIS, norm(q), color=COLORS[m], lw=1.1, alpha=0.9)
        for t in ticks:
            ax.axvline(t, ymin=0, ymax=0.08, color=COLORS[m], lw=1.4, alpha=0.85)
        ax.set_title(f'{m}\n$r={r_vals[m]:.3f}$', fontsize=8.5, pad=4)
        ax.set_xlim(0, LAST_MIN)
        ax.set_ylim(-0.04, 1.2)
        ax.set_xlabel('RT (min)', fontsize=8)
        ax.tick_params(labelsize=7)
        _despine(ax)
    axes[0].set_ylabel('TIC (scaled)', fontsize=8)
    na, nb = batch_name(a.name), batch_name(b.name)
    fig.legend(
        handles=[
            Line2D([0], [0], color='#555555', lw=1.4, label=f'{nb} sample 0'),
            Line2D([0], [0], color='#333333', lw=1.4, label=f'{na} sample {qi} (aligned)'),
            Line2D([0], [0], color='#333333', lw=1.4, label='Ticks: pooled anchors'),
        ],
        fontsize=7.5,
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        framealpha=0.8,
    )
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info(
        '  fig2 raw-TIC r (%s sample %d vs %s sample 0): %s → %s',
        na,
        qi,
        nb,
        ', '.join(f'{m} {v:.3f}' for m, v in r_vals.items()),
        out,
    )
    return r_vals


# ── Figure 3: calibration curve ───────────────────────────────────────────────


def fig3_calibration_curve(hist_pre, hist_rand, unaligned_r, title, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for hist, color, ls, label in (
        (hist_pre, COLORS['WarpTransformer'], '-', 'Pretrained init'),
        (hist_rand, '#888888', '--', 'Random init'),
    ):
        steps = [h['step'] for h in hist]
        ax1.plot(steps, [h['loss'] for h in hist], color=color, lw=1.8, ls=ls, label=label)
        ax2.plot(steps, [h['tic_r'] for h in hist], color=color, lw=1.8, ls=ls, label=label)
    ax2.axhline(unaligned_r, color='#bbbbbb', lw=1.2, ls=':', label='Unaligned')
    for ax, ylabel, t in (
        (ax1, 'Loss (anchor MSE + smoothness)', 'Calibration loss'),
        (ax2, 'Mean pairwise TIC r', 'Alignment quality (TIC r)'),
    ):
        ax.set_xlabel('Calibration step', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=8.5)
        ax.tick_params(labelsize=8)
        _despine(ax)
    fig.suptitle(f'Anchor calibration convergence — {title}', fontsize=10, y=0.97)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info('  fig3 → %s', out)


# ── Figures 4–5: saved arrays ─────────────────────────────────────────────────


def load_saved(paths: Paths, fig_seed: int) -> dict:
    """pair → method → (per-pair r, silhouette), from scripts 05, 06 (seed) and 07."""
    out = {}
    for a, b in EVAL_PAIRS:
        base = np.load(paths.per_pair / f'tic_r_baselines_{a}_{b}.npz')
        calib = np.load(paths.per_pair / f'tic_r_calib_{a}_{b}_seed{fig_seed}.npz')
        pchip = np.load(paths.per_pair / f'tic_r_pchip_pooled_{a}_{b}.npz')
        if not np.allclose(base['unaligned'], calib['unaligned'], atol=1e-6):
            raise ValueError(f'{a} × {b}: unaligned arrays of scripts 05 and 06 differ')
        if pchip['pchip_pooled'].shape != calib['warptransformer'].shape:
            raise ValueError(f'{a} × {b}: script 07 arrays do not match script 06')
        sil_u, sil_cow, sil_ico = (float(v) for v in base['sil'])
        out[pair_label(a, b)] = {
            'Unaligned': (base['unaligned'], sil_u),
            'icoshift': (base['icoshift'], sil_ico),
            'COW-TIC': (base['cow_tic'], sil_cow),
            'PCHIP (pooled anchors)': (pchip['pchip_pooled'], float(pchip['sil'])),
            'WarpTransformer': (calib['warptransformer'], float(calib['sil'][2])),
        }
    return out


def fig4_violin(saved, out):
    pairs = list(saved)
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.08, right=0.99, top=0.88, bottom=0.18)
    pos = np.arange(1, len(METHODS) + 1)
    for ax, p in zip(np.atleast_1d(axes), pairs, strict=True):
        parts = ax.violinplot(
            [saved[p][m][0] for m in METHODS], positions=pos, showmedians=True, showextrema=True
        )
        for body, m in zip(parts['bodies'], METHODS, strict=True):
            body.set_facecolor(COLORS[m])
            body.set_alpha(0.65)
            body.set_linewidth(0)
        parts['cmedians'].set_color('#222222')
        parts['cmedians'].set_linewidth(1.5)
        for k in ('cmins', 'cmaxes', 'cbars'):
            parts[k].set_color('#555555')
        ax.set_xticks(pos)
        ax.set_xticklabels([m.replace(' ', '\n') for m in METHODS], fontsize=9)
        ax.set_title(p, fontsize=10)
        ax.tick_params(axis='y', labelsize=9)
        _despine(ax)
    np.atleast_1d(axes)[0].set_ylabel('Pairwise TIC r', fontsize=10)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info('  fig4 → %s', out)


def fig5_scatter(saved, out):
    markers = dict(zip(saved, 'oDs^v<>', strict=False))  # more markers than pairs
    fig, ax = plt.subplots(figsize=(7.6, 5))
    for p, d in saved.items():
        r0, sil0 = float(d['Unaligned'][0].mean()), d['Unaligned'][1]
        for m in METHODS[1:]:
            ax.scatter(
                float(d[m][0].mean()) - r0,
                d[m][1] - sil0,
                color=COLORS[m],
                marker=markers[p],
                s=80,
                zorder=4,
                edgecolors='white',
                linewidths=0.5,
            )
    ax.axhline(0, color='#aaaaaa', lw=0.9, ls='--', zorder=1)
    ax.axvline(0, color='#aaaaaa', lw=0.9, ls='--', zorder=1)
    leg1 = ax.legend(
        handles=[
            Line2D(
                [0], [0], marker='o', color='w', markerfacecolor=COLORS[m], markersize=9, label=m
            )
            for m in METHODS[1:]
        ],
        fontsize=8,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        title='Method',
        title_fontsize=8,
        framealpha=0.85,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=markers[p],
                color='w',
                markerfacecolor='#555555',
                markersize=9,
                label=p,
            )
            for p in saved
        ],
        fontsize=8,
        loc='upper left',
        bbox_to_anchor=(1.02, 0.55),
        title='Batch pair',
        title_fontsize=8,
        framealpha=0.85,
    )
    ax.set_xlabel('Δ TIC r  (vs unaligned)', fontsize=10)
    ax.set_ylabel(
        'Δ Study silhouette  (vs unaligned)\n← better integration      worse →', fontsize=9
    )
    ax.set_title('Alignment quality vs batch integration', fontsize=10)
    ax.tick_params(labelsize=8)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight', bbox_extra_artists=[leg1])
    plt.close(fig)
    log.info('  fig5 → %s', out)


# ── Running ───────────────────────────────────────────────────────────────────


def selection_report(
    a: Batch, b: Batch, qi: int, deltas: np.ndarray, select: str, seed: int, device: torch.device
) -> list[str]:
    head = (
        f'Figs 1–2 sample ({select}) for {pair_label(a.name, b.name)}, seed {seed}, device '
        f'{device}: index {qi} ({a.sample_ids[qi]}), Δ TIC r {deltas[qi]:+.4f} (median '
        f'{np.median(deltas):+.4f}, max {deltas.max():+.4f}, min {deltas.min():+.4f})'
    )
    return [
        head,
        '',
        'index  id  Δ TIC r = r(WT A_i, WT B0) − r(A_i, B0):',
        *(f'  {i:>3}  {a.sample_ids[i]:<36} {deltas[i]:+.4f}' for i in np.argsort(deltas)[::-1]),
    ]


def run(
    paths: Paths,
    figs: set[int],
    name_a: str = 'batch_sep2015',
    name_b: str = 'batch_jul2016',
    select: str = 'median',
    seed: int = 0,
    ckpt: Path | None = None,
    device: str = 'cpu',
    calib_steps: int = N_CALIB_STEPS,
    fig_seed: int = 0,
) -> list[Path]:
    """Make the requested figures; returns the files written."""
    out_dir, tag, written = paths.results, '' if select == 'best' else '_median', []
    out_dir.mkdir(parents=True, exist_ok=True)
    if figs & {1, 2, 3}:
        dev, ck = pick_device(device), ckpt_or_default(ckpt, paths)
        a, b = load_batch(name_a, paths.fish_oil), load_batch(name_b, paths.fish_oil)
        anchors = extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw).anchors
        aligned, warps, hist_pre = calibrate_pair(
            a, b, anchors, ck, dev, seed, calib_steps, log_every=5 if 3 in figs else 0
        )
        qi, deltas = select_sample(a, b, aligned, select)
        sel = out_dir / f'fig_selection_{name_a}_{name_b}_{select}.txt'
        sel.write_text('\n'.join(selection_report(a, b, qi, deltas, select, seed, dev)) + '\n')
        written.append(sel)
        if 1 in figs:
            written.append(out_dir / f'fig1_warp_functions_{name_a}_{name_b}{tag}.pdf')
            fig1_warp_functions(a, b, warps, anchors, qi if select == 'median' else 0, written[-1])
        if 2 in figs:
            written.append(out_dir / f'fig2_tic_overlays_{name_a}_{name_b}{tag}.pdf')
            fig2_tic_overlays(a, b, warps, anchors, qi, written[-1])
        if 3 in figs:
            _, _, hist_rand = calibrate_pair(
                a, b, anchors, None, dev, seed, calib_steps, log_every=5
            )
            written.append(out_dir / f'fig3_calibration_curve_{name_a}_{name_b}.pdf')
            fig3_calibration_curve(
                hist_pre,
                hist_rand,
                float(tic_r_pairs(a.chromas, b.chromas).mean()),
                pair_label(name_a, name_b),
                written[-1],
            )
    if figs & {4, 5}:
        saved = load_saved(paths, fig_seed)
        if 4 in figs:
            written.append(out_dir / 'fig4_per_pair_distributions.pdf')
            fig4_violin(saved, written[-1])
        if 5 in figs:
            written.append(out_dir / 'fig5_silhouette_scatter.pdf')
            fig5_scatter(saved, written[-1])
    return written


def cli(argv: list[str] | None = None) -> None:
    ap = parser(__doc__)
    add_pair_args(ap)
    add_model_args(ap)
    ap.add_argument('--figs', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument(
        '--select',
        choices=['best', 'median'],
        default='median',
        help='batch-A sample for Figs 1–2: largest or median WT Δ TIC r',
    )
    ap.add_argument(
        '--fig-seed', type=int, default=0, help='calibration seed whose saved arrays Figs 4–5 use'
    )
    args = ap.parse_args(argv)
    setup_logging()
    run(
        Paths(args.root),
        set(args.figs),
        batch_name_of(args.study_a),
        batch_name_of(args.study_b),
        args.select,
        args.seed,
        args.ckpt,
        args.device,
        args.calib_steps,
        args.fig_seed,
    )
