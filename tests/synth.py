"""Synthetic GC-MS batches with known compound positions and drift, for the tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

T, M = 200, 1000


def compound_spectra(n: int, rng: np.random.Generator, n_peaks: int = 12) -> np.ndarray:
    """n distinct sparse intensity spectra [n, M] (L2 = 1), clearly separable by cosine."""
    specs = np.zeros((n, M))
    for i in range(n):
        mz = rng.choice(np.arange(40, 600), n_peaks, replace=False)
        specs[i, mz] = rng.uniform(0.2, 1.0, n_peaks)
    return specs / np.linalg.norm(specs, axis=1, keepdims=True)


def sample_intensity(positions, heights, specs, rng, sigma=0.8, noise=1e-4, start_bin=0):
    """Intensity chromatogram [T, M] with Gaussian elution profiles; zero before start_bin
    (as runs that start acquiring later)."""
    t = np.arange(T)[:, None]
    prof = heights[None, :] * np.exp(-0.5 * ((t - positions[None, :]) / sigma) ** 2)
    x = prof @ specs + rng.exponential(noise, size=(T, M))
    x[:start_bin] = 0.0
    return x


def normalise(x: np.ndarray) -> np.ndarray:
    """sqrt + L2 per bin, as 03_preprocess does."""
    s = np.sqrt(np.clip(x, 0, None))
    n = np.linalg.norm(s, axis=-1, keepdims=True)
    return (s / np.where(n > 1e-8, n, 1.0)).astype(np.float32)


def make_batch(
    n_samples: int,
    positions: np.ndarray,
    shift: float,
    specs: np.ndarray,
    rng: np.random.Generator,
    jitter: float = 0.0,
    start_bin: int = 0,
):
    """(chromas [N, T, M], tic_raw [N, T]) for a batch whose compounds sit at positions + shift."""
    chromas, raws = [], []
    heights = rng.uniform(1.0, 5.0, len(positions))
    for _ in range(n_samples):
        pos = positions + shift + rng.uniform(-jitter, jitter, len(positions))
        x = sample_intensity(
            pos, heights * rng.uniform(0.8, 1.2, len(positions)), specs, rng, start_bin=start_bin
        )
        chromas.append(normalise(x))
        raws.append(x.sum(axis=1).astype(np.float32))
    return np.stack(chromas), np.stack(raws)


# Four batches mirroring the real layout: Sep/Jan unshifted, Apr/Jul shifted earlier by SHIFT bins
SHIFT = -8
BATCH_SHIFTS = {
    'batch_sep2015': 0,
    'batch_jan2016': 0,
    'batch_apr2016': SHIFT,
    'batch_jul2016': SHIFT,
}
POSITIONS = np.array([40, 52, 63, 75, 88, 101, 115, 128, 141, 155, 168, 181], dtype=float)


def write_dataset(root: Path, n_per_batch: int = 4, seed: int = 0) -> dict:
    """Write data/fish_oil/<batch>/chroma/*.npz under root, as 03 does. Returns the arrays."""
    rng = np.random.default_rng(seed)
    specs = compound_spectra(len(POSITIONS), rng)
    out = {}
    for name, shift in BATCH_SHIFTS.items():
        chromas, raws = make_batch(
            n_per_batch,
            POSITIONS,
            shift,
            specs,
            rng,
            jitter=0.3,
            start_bin=22 if shift == 0 else 13,
        )
        d = root / 'data' / 'fish_oil' / name / 'chroma'
        d.mkdir(parents=True, exist_ok=True)
        for i, (c, r) in enumerate(zip(chromas, raws)):
            np.savez_compressed(d / f'sample_{i:02d}.npz', chroma=c, tic_raw=r)
        out[name] = (chromas, raws)
    return out
