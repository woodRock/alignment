"""The WarpTransformer as an alignbench method (entry point group `alignbench.methods`).

For a batch pair (A, B) the pretrained model is calibrated at test time on that pair's pooled
anchors (steps.calibrate_pair.compute, the paper's step 06: seed 0, CPU by default) and predicts
one monotone warp per sample into a shared consensus frame. The correspondence map B_j → A_i is
A_i's warp read at the frame position of B_j's times (joint-align scripts/32_fair_benchmark.py's
reduction), so it is per sample pair. Pairs are calibrated lazily, once each, when first scored.

    alignbench run --methods warp-transformer,pooled-pchip --dataset fish_oil \\
        --data-root data/fish_oil --out results/bench/ \\
        --param warp-transformer.ckpt='"checkpoints/warp_transformer_v2.pt"'

The checkpoint path is an explicit parameter (nothing is found relative to this repository).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from alignbench.anchors import Anchors
from alignbench.data import Study
from alignbench.methods import Fit
from alignbench.warps import invert_src

from .config import N_CALIB_STEPS


def _strict(w: np.ndarray) -> np.ndarray:
    """Non-decreasing, then strictly increasing (script 32's guard on the v2 warps)."""
    return np.maximum.accumulate(w) + 1e-6 * np.arange(len(w))


def _read_warp(w: np.ndarray, t: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Warp w [T] (output bin → source bin) at fractional output positions t, continued linearly
    with the end segments' slope (v2 warps are not pinned to [0, T − 1]); script 32's read_warp."""
    n = len(grid)
    out = np.interp(t, grid, w)
    out = np.where(t < 0, w[0] + t * (w[1] - w[0]), out)
    return np.where(t > n - 1, w[-1] + (t - n + 1) * (w[-1] - w[-2]), out)


class WarpTransformer:
    name = 'warp-transformer'
    needs_anchors = False  # calibrates on its own pooled anchors per pair

    def fit(self, study: Study, anchors: Anchors | None = None, ckpt: str | Path | None = None,
            seed: int | None = 0, device: str = 'cpu', calib_steps: int = N_CALIB_STEPS,
            **_: Any) -> Fit:
        if ckpt is None:
            raise ValueError('warp-transformer needs ckpt= (the pretrained checkpoint, e.g. '
                             'checkpoints/warp_transformer_v2.pt)')
        from .calibration import pick_device
        from .steps.calibrate_pair import compute

        ckpt, dev = Path(ckpt), pick_device(device)
        grid = np.arange(study.axis.n_bins, dtype=float)
        warps: dict[tuple[int, int], np.ndarray] = {}  # (A, B) → strict warps [n_A + n_B, T]
        results: dict[tuple[int, int], object] = {}

        def pair_warps(a: int, b: int) -> np.ndarray:
            if (a, b) not in warps:
                res = compute(study[a], study[b], ckpt, dev, seed, calib_steps)
                results[(a, b)] = res
                warps[(a, b)] = np.stack([_strict(w) for w in res.warps])
            return warps[(a, b)]

        def g(k: int, l: int, t: np.ndarray, *, i: int | None = None,
              j: int | None = None) -> np.ndarray:
            t = np.asarray(t, dtype=float)
            if k == l and i == j:
                return t
            if i is None or j is None:
                raise ValueError(
                    'warp-transformer is per sample pair: pass i (in batch k) and j (in batch l)'
                )
            W = pair_warps(l, k)  # A = l (target), B = k (source)
            n_a = len(study[l])
            return _read_warp(W[j], invert_src(W[n_a + i], t), grid)

        params = {'ckpt': str(ckpt), 'seed': seed, 'device': str(dev), 'calib_steps': calib_steps}
        return Fit(g, self.name, params, extras={'calibrations': results}, per_sample=True)
