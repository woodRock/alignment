"""Loading the preprocessed fish-oil batches written by 03_preprocess_fish_oil_batches.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Batch:
    """One acquisition batch. Samples are in sorted filename order everywhere."""

    name: str
    chromas: np.ndarray  # [N, T, M] float32, sqrt + L2 per bin
    tic_raw: np.ndarray  # [N, T]    float32, summed raw intensity of each binned apex scan
    sample_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.chromas)

    @property
    def tic(self) -> np.ndarray:
        """Normalised TIC [N, T]: the sum over m/z of the normalised chromatogram."""
        return self.chromas.sum(axis=-1)


def chroma_files(chroma_dir: Path) -> list[Path]:
    files = sorted(Path(chroma_dir).glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'no .npz chromatograms in {chroma_dir}')
    return files


def load_chromas(chroma_dir: Path) -> np.ndarray:
    """[N, T, M] float32 chromatograms in sorted filename order."""
    return np.stack([np.load(p)['chroma'].astype(np.float32) for p in chroma_files(chroma_dir)])


def load_tic_raw(chroma_dir: Path) -> np.ndarray:
    """[N, T] raw TICs, in the same order as load_chromas."""
    out = []
    for p in chroma_files(chroma_dir):
        with np.load(p) as z:
            if 'tic_raw' not in z.files:
                raise ValueError(
                    f'{p} has no tic_raw; rerun scripts/03_preprocess_fish_oil_batches.py'
                )
            out.append(z['tic_raw'].astype(np.float32))
    return np.stack(out)


def load_batch(name: str, fish_oil_dir: Path) -> Batch:
    """One batch from <fish_oil_dir>/<name>/chroma (Paths.fish_oil)."""
    d = Path(fish_oil_dir) / name / 'chroma'
    return Batch(
        name=name,
        chromas=load_chromas(d),
        tic_raw=load_tic_raw(d),
        sample_ids=tuple(p.stem for p in chroma_files(d)),
    )
