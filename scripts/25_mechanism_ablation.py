"""
25_mechanism_ablation.py — Where does the WarpTransformer's gain over a spline come from?

Script 24 showed that a PCHIP spline through the WarpTransformer's own pooled anchors stays
~0.15 TIC r below it, so anchor handling is not the explanation. This script separates the
remaining candidates on the four evaluated pairs, all from one seeded calibration:

  Unaligned
  WT original          the calibrated WarpTransformer: one warp per sample, both batches
  WT batch-shared      its warps averaged within each batch (mean of monotone warps is
                       monotone), so every sample of a batch gets the same warp
  WT residual only     identity + (w_i − batch mean): the within-batch part alone, with no
                       between-batch correction
  Free warp            no network: one monotone warp per batch (softplus-cumsum, the WT head's
                       parameterisation, zero-initialised), optimised with the same anchor and
                       smoothness loss and the same Adam/cosine/clip/500-step/best-loss loop.
                       Per-sample free warps would receive identical gradients within a batch,
                       so per-batch parameters are equivalent. Run at the WT learning rate and
                       at a 100× larger one, so slow convergence cannot explain a gap.
  Pooled PCHIP         script 24: spline through the same pooled anchors, A only
  Pooled PCHIP + COW   pooled PCHIP, then a per-sample COW refinement (8 segments, ±4 bins) of
                       every A_i and B_j onto the mean batch-B TIC: classical per-sample
                       correction. Note COW optimises TIC correlation itself.
  Smoothing controls   a pure half-bin shift (maximal linear-interpolation smoothing, no
                       alignment) applied to both batches of Unaligned and to batch B of Pooled
                       PCHIP. The WT resamples both batches, the baselines only batch A, so
                       resampling alone could inflate its TIC r.

Besides between-batch TIC r (the paper's metric) it reports within-batch TIC r (mean over
pairs inside A and inside B), which shows directly whether a variant aligns samples of the
same batch to each other, and the within-batch warp spread of the WT.

Everything is evaluated on TICs: warping is linear interpolation along RT, which commutes with
summing over m/z, so the warped TIC equals the TIC of the warped chromatogram.

Output: results/mechanism_ablation.txt, results/per_pair/mechanism_ablation_<A>_<B>.npz
"""

from __future__ import annotations

import argparse
import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import wilcoxon

SCRIPTS = Path(__file__).parent
ROOT    = SCRIPTS.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


calib16 = _load('calib16', '16_align_testtime_calib.py')
base17  = _load('base17', '17_literature_baselines.py')
pool24  = _load('pool24', '24_pchip_pooled_anchors.py')
from warp_transformer import (ChromaWarpTransformer, apply_warp,           # noqa: E402
                              make_monotone_warp, warp_smoothness_loss)
from alignment import load_chromas, warp_chroma_2d                          # noqa: E402

N_BINS  = calib16.N_BINS
BIN_MIN = calib16.BIN_MIN
PAIRS   = [('batch_sep2015', 'batch_jul2016'), ('batch_sep2015', 'batch_apr2016'),
           ('batch_jan2016', 'batch_apr2016'), ('batch_jan2016', 'batch_jul2016')]
REFINE_SEGS, REFINE_SLACK = 8, 4
FREE_LRS = (calib16.LR_CALIB, calib16.LR_CALIB * 100)


# ── TIC helpers ───────────────────────────────────────────────────────────────

def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """All-pairs Pearson r between rows of a [n_a, T] and b [n_b, T], i-major flat."""
    az = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    bz = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    return (az @ bz.T / a.shape[1]).ravel()


def within_r(x: np.ndarray) -> float:
    r = pearson_rows(x, x).reshape(len(x), len(x))
    return float(np.mean([r[i, j] for i, j in combinations(range(len(x)), 2)]))


def warp_tics(tics: np.ndarray, warps: np.ndarray) -> np.ndarray:
    """Apply [N, T] pull-back warps (bins) exactly as the WarpTransformer does (apply_warp)."""
    out = apply_warp(torch.from_numpy(tics[:, :, None].astype(np.float32)),
                     torch.from_numpy(warps.astype(np.float32)))
    return out[:, :, 0].numpy()


def warp_tics_fn(tics: np.ndarray, warp_fn) -> np.ndarray:
    """Apply a minutes→minutes warp function exactly as the baselines do (warp_chroma_2d)."""
    return np.stack([warp_chroma_2d(t[:, None], warp_fn)[:, 0] for t in tics])


# ── Variants ──────────────────────────────────────────────────────────────────

def calibrate_wt(chromas_a, chromas_b, anchors, ckpt, device, seed):
    n_a, n_b  = len(chromas_a), len(chromas_b)
    study_ids = torch.tensor([0] * n_a + [1] * n_b, dtype=torch.long, device=device)
    x         = torch.from_numpy(np.stack(chromas_a + chromas_b)).to(device)
    model     = ChromaWarpTransformer()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    calib16.calibrate(model, x, study_ids, anchors, device, seed=seed)
    with torch.no_grad():
        aligned, warps = model(x, study_ids)
        loss = final_loss(warps, study_ids, anchors, device)
    return warps.cpu().numpy(), aligned.sum(-1).cpu().numpy(), loss


def final_loss(warps, study_ids, anchors, device) -> tuple[float, float]:
    a = float(calib16.anchor_loss(warps, study_ids, anchors, device))
    s = float(warp_smoothness_loss(warps))
    return a, s


def free_warp(n_a, n_b, anchors, lr, seed):
    """One monotone warp per batch, same loss and optimisation loop as calib16.calibrate."""
    calib16.set_seed(seed)
    dev       = torch.device('cpu')
    study_ids = torch.tensor([0] * n_a + [1] * n_b, dtype=torch.long)
    logits    = torch.zeros(2, N_BINS, requires_grad=True)
    opt       = optim.Adam([logits], lr=lr)
    sched     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=calib16.N_CALIB_STEPS)
    best, best_logits = float('inf'), logits.detach().clone()

    def expand(lg):
        return make_monotone_warp(lg)[study_ids]                # [N, T]

    for _ in range(calib16.N_CALIB_STEPS):
        warps = expand(logits)
        loss  = (calib16.LAMBDA_ANCHOR * calib16.anchor_loss(warps, study_ids, anchors, dev)
                 + calib16.LAMBDA_SMOOTH * warp_smoothness_loss(warps))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([logits], 1.0)
        opt.step()
        sched.step()
        if float(loss) < best:
            best, best_logits = float(loss), logits.detach().clone()
    with torch.no_grad():
        warps = expand(best_logits)
        return warps.numpy(), final_loss(warps, study_ids, anchors, dev)


def cow_refine(tics: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.stack([base17.cow_tic_align(t, ref, n_segs=REFINE_SEGS, slack=REFINE_SLACK)[0]
                     for t in tics])


def half_bin_shift(n: int) -> np.ndarray:
    return np.tile(np.clip(np.arange(N_BINS) + 0.5, 0, N_BINS - 1), (n, 1))


# ── Main ──────────────────────────────────────────────────────────────────────

def holm(ps):
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def main(seed: int, device_name: str, ckpt: Path) -> None:
    device = calib16.pick_device(device_name)
    per_pair = ROOT / 'results' / 'per_pair'
    lines = [__doc__.strip().splitlines()[0], '',
             f'Seed {seed}   device {device}   checkpoint {ckpt.relative_to(ROOT)}   '
             f'COW refinement {REFINE_SEGS} segments ±{REFINE_SLACK} bins', '']
    summary, per_a_store = [], []

    for a, b in PAIRS:
        ca = load_chromas(ROOT / 'data' / 'fish_oil' / a / 'chroma')
        cb = load_chromas(ROOT / 'data' / 'fish_oil' / b / 'chroma')
        n_a, n_b = len(ca), len(cb)
        ta = np.stack([c.sum(1) for c in ca])
        tb = np.stack([c.sum(1) for c in cb])
        anchors = calib16.extract_anchors(ca, cb)
        print(f'\n{a} × {b}: {n_a}×{n_b}, {len(anchors)} anchors — calibrating WarpTransformer …',
              flush=True)

        w_wt, wt_tics, loss_wt = calibrate_wt(ca, cb, anchors, ckpt, device, seed)
        w_shared = np.concatenate([np.tile(w_wt[:n_a].mean(0), (n_a, 1)),
                                   np.tile(w_wt[n_a:].mean(0), (n_b, 1))])
        ident    = np.arange(N_BINS, dtype=np.float64)
        resid    = np.concatenate([w_wt[:n_a] - w_wt[:n_a].mean(0),
                                   w_wt[n_a:] - w_wt[n_a:].mean(0)]) + ident
        resid    = np.clip(resid, 0, N_BINS - 1)
        free     = {lr: free_warp(n_a, n_b, anchors, lr, seed) for lr in FREE_LRS}

        pchip_fn, _ = pool24.pooled_pchip_warp(anchors)
        pa          = warp_tics_fn(ta, pchip_fn)
        ref_b       = tb.mean(0)

        both = np.concatenate([ta, tb])
        V = {}                                                   # name → (A tics, B tics)
        V['Unaligned']                 = (ta, tb)
        wt_al = warp_tics(both, w_wt)
        V['WT original']               = (wt_al[:n_a], wt_al[n_a:])
        sh_al = warp_tics(both, w_shared)
        V['WT batch-shared']           = (sh_al[:n_a], sh_al[n_a:])
        rs_al = warp_tics(both, resid)
        V['WT residual only']          = (rs_al[:n_a], rs_al[n_a:])
        for lr, (w_free, _) in free.items():
            fr_al = warp_tics(both, w_free)
            V[f'Free warp (lr {lr:g})'] = (fr_al[:n_a], fr_al[n_a:])
        V['Pooled PCHIP']              = (pa, tb)
        V['Pooled PCHIP + COW refine'] = (cow_refine(pa, ref_b), cow_refine(tb, ref_b))
        hs = warp_tics(both, half_bin_shift(n_a + n_b))
        V['Control: unaligned, both ½-bin shifted'] = (hs[:n_a], hs[n_a:])
        V['Control: pooled PCHIP, B ½-bin shifted'] = (pa, warp_tics(tb, half_bin_shift(n_b)))

        r = {k: pearson_rows(xa, xb) for k, (xa, xb) in V.items()}
        # Sanity: re-applying the WT warps to the TICs must reproduce the model's own output
        r_model = pearson_rows(wt_tics[:n_a], wt_tics[n_a:])
        if not np.allclose(r['WT original'], r_model, atol=1e-4):
            raise SystemExit(f'WT warp re-application mismatch: max |dr| '
                             f'{np.abs(r["WT original"] - r_model).max():.2e}')
        u = float(r['Unaligned'].mean())

        spread = lambda w: float(np.abs(w - w.mean(0)).mean()) * BIN_MIN
        lines += ['=' * 96,
                  f'{a} × {b}   n = {n_a}×{n_b} = {n_a * n_b}   anchors {len(anchors)}',
                  f'  WT within-batch warp spread (mean |w_i − batch mean|): '
                  f'A {spread(w_wt[:n_a]):.3f} min, B {spread(w_wt[n_a:]):.3f} min',
                  f'  Final anchor / smoothness loss: WT {loss_wt[0]:.4f} / {loss_wt[1]:.5f}; '
                  + '; '.join(f'free lr {lr:g} {l[0]:.4f} / {l[1]:.5f}' for lr, (_, l) in free.items()),
                  f'  {"Variant":<42} {"TIC r A×B":>9} {"Δ unal":>8} {"within A":>9} {"within B":>9}']
        for k, (xa, xb) in V.items():
            lines.append(f'  {k:<42} {r[k].mean():>9.3f} {r[k].mean() - u:>+8.3f} '
                         f'{within_r(xa):>9.3f} {within_r(xb):>9.3f}')
        summary.append((a, b, {k: float(v.mean()) for k, v in r.items()}))
        per_a_store.append((a, b, n_a, n_b, r))
        np.savez(per_pair / f'mechanism_ablation_{a}_{b}.npz',
                 **{k.replace(' ', '_'): v for k, v in r.items()},
                 w_wt=w_wt, anchors=np.array(anchors), n_a=n_a, n_b=n_b)

    names = list(summary[0][2])
    lines += ['', '=' * 96, 'Summary: mean between-batch TIC r',
              f'  {"Variant":<42} ' + ' '.join(f'{a[6:13] + "×" + b[6:13]:>16}' for a, b, _ in summary)]
    for k in names:
        lines.append(f'  {k:<42} ' + ' '.join(f'{s[k]:>16.3f}' for *_, s in summary))

    best_free = f'Free warp (lr {FREE_LRS[-1]:g})'
    comps = [('WT original', 'WT batch-shared'), ('WT batch-shared', best_free),
             (best_free, 'Pooled PCHIP'), ('WT original', 'Pooled PCHIP + COW refine'),
             ('Control: unaligned, both ½-bin shifted', 'Unaligned')]
    lines += ['', 'Paired two-sided Wilcoxon per A sample (r averaged over B_j). Holm family: the 4 '
              'evaluated batch pairs, corrected separately for each comparison below (family size 4).',
              'Diagnostic tests; not part of the 15-comparison family of significance_tests.txt.']
    for x, y in comps:
        res = []
        for a, b, n_a, n_b, r in per_a_store:
            dx = r[x].reshape(n_a, n_b).mean(1)
            dy = r[y].reshape(n_a, n_b).mean(1)
            res.append((a, b, wilcoxon(dx, dy).pvalue, float(np.median(dx - dy))))
        adj = holm([p for _, _, p, _ in res])
        lines.append(f'  {x}  vs  {y}:')
        for (a, b, p, md), pa in zip(res, adj):
            lines.append(f'    {a + " × " + b:<28} Holm p={pa:.2e}  median Δr {md:+.4f}')

    out = ROOT / 'results' / 'mechanism_ablation.txt'
    out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\n  results → {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', help='cpu (reproducible) | mps | cuda')
    ap.add_argument('--ckpt', type=Path, default=ROOT / 'checkpoints' / 'warp_transformer_baseline.pt')
    a = ap.parse_args()
    main(a.seed, a.device, a.ckpt.resolve())
