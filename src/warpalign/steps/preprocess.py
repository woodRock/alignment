"""Step 03 — raw fish-oil GC-MS exports (.txt, one per run) → per-batch binned chromatograms.

Runs are grouped into batches by acquisition date (comment1 = YYMMDD); instrument blanks
are skipped. Each run is binned by retention time on a fixed 0–45 min grid of 0.225-min windows
(bin i covers [0.225 i, 0.225 (i+1)) min), keeping the highest-TIC scan per window; windows
outside the run's acquisition range stay zero. (Batches start acquiring at 3.0 or 5.0 min, so
binning by scan index would give them different time axes: see step 04's scan-time audit.)
--binning index reproduces the v1 data.

Output, per batch under <root>/data/fish_oil/<batch>/:
  chroma/<sample>.npz  chroma [200, 1000] float32 (sqrt + L2 per bin) and tic_raw [200]
                       (summed raw intensity of each bin's apex scan)
  X.npy                [N, 1000] sqrt + L2 sum spectrum of the whole run
  y.npy                [N] tissue label (tissue_labels.json)
  groups.txt           fish individual per sample; sample_ids.txt: file stem per sample
and data/fish_oil/tissue_labels.json, batch_manifest.json.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..config import BIN_MIN, MZ_BINS, N_BINS, REPO_ROOT, Paths
from ._common import parser, setup_logging

log = logging.getLogger('warpalign')

DEFAULT_RAW_DIR = (
    REPO_ROOT.parent / 'fish-data-analysis' / 'engr489' / 'src' / 'data' / 'raw-subsetted-CVSs'
)

DATE_TO_BATCH: dict[str, str] = {
    '150807': 'batch_aug2015',  # two head samples only
    '150916': 'batch_sep2015',
    '150917': 'batch_sep2015',
    '160112': 'batch_jan2016',
    '160429': 'batch_apr2016',
    '160711': 'batch_jul2016',
}
TISSUE_LABELS: dict[str, int] = {
    'Frame': 0,
    'Gonad': 1,
    'Head': 2,
    'Liver': 3,
    'Rest of Guts': 4,
    'Skin': 5,
}
# Longest first, so multi-word patterns match before their substrings
_TISSUE_PATTERNS = [
    ('Rest of Guts', 'Rest of Guts'),
    ('Gonad Female', 'Gonad'),
    ('Gonad Male', 'Gonad'),
    ('Liver Female', 'Liver'),
    ('Liver Male', 'Liver'),
    ('Gonad', 'Gonad'),
    ('Liver', 'Liver'),
    ('Frame', 'Frame'),
    ('Head', 'Head'),
    ('Skin', 'Skin'),
    ('ROG', 'Rest of Guts'),
]

_FNAME = re.compile(r'^\d+\s+-\s+(?P<species>[A-Z]+\d+)\s+(?P<rest>.+?)$', re.IGNORECASE)
ACQ_DATE = re.compile(r'comment1\s*=\s*(\d{6})')
BLANK = re.compile(r'comment1\s*=\s*\d{6}\s+BLANK', re.IGNORECASE)
START_TIME = re.compile(r'start_time\s*=\s*([\d.]+)')
_INTEG = re.compile(r'integ_intens\s*=\s*([\d.]+)')
_PACKET = re.compile(r'intensity\s*=\s*([\d.]+).*?mass/position\s*=\s*([\d.]+)')


class ScanData(NamedTuple):
    spectra: np.ndarray  # [n_scans, M] float32, max intensity per integer m/z
    tic: np.ndarray  # [n_scans] float32, instrument TIC (integ_intens)
    acq_date: str  # YYMMDD
    times: np.ndarray  # [n_scans] float64, scan start time (min)


def parse_filename(stem: str) -> tuple[str, str] | None:
    """(species, tissue) from e.g. '4 - BCO4 Liver Female A', or None."""
    m = _FNAME.match(stem.strip())
    if not m:
        return None
    rest = m.group('rest').lower()
    tissue = next((t for pattern, t in _TISSUE_PATTERNS if pattern.lower() in rest), None)
    return (m.group('species').upper(), tissue) if tissue else None


def parse_txt(path: Path, mz_max: int = MZ_BINS) -> ScanData | None:
    """Parse one raw export; None for instrument blanks or files without scans or a date."""
    acq_date: str | None = None
    scans: list[tuple[float, float, dict[int, float]]] = []
    in_scan = in_peaks = False
    rt, tic = 0.0, 0.0
    peaks: dict[int, float] = {}
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if not in_scan:
                if acq_date is None and (m := ACQ_DATE.match(line)):
                    acq_date = m.group(1)
                if BLANK.match(line):
                    return None
                in_scan = line.startswith('ScanHeader #')
            elif line.startswith('ScanHeader #'):
                scans.append((rt, tic, peaks))
                rt, tic, peaks, in_peaks = 0.0, 0.0, {}, False
            elif line.startswith('start_time =') and 'end_time' in line:
                rt = float(START_TIME.search(line).group(1))  # type: ignore[union-attr]
            elif line.startswith('num_readings') and 'integ_intens' in line:
                tic = float(_INTEG.search(line).group(1))  # type: ignore[union-attr]
            elif line == 'DataPeaks':
                in_peaks = True
            elif in_peaks and line.startswith('Packet #') and (m := _PACKET.search(line)):
                intensity, mz = float(m.group(1)), round(float(m.group(2)))
                if 0 <= mz < mz_max and intensity > peaks.get(mz, 0.0):
                    peaks[mz] = intensity
    if not in_scan or acq_date is None:
        return None
    scans.append((rt, tic, peaks))
    spectra = np.zeros((len(scans), mz_max), dtype=np.float32)
    for i, (_, _, p) in enumerate(scans):
        for mz, v in p.items():
            spectra[i, mz] = v
    return ScanData(
        spectra,
        np.array([s[1] for s in scans], dtype=np.float32),
        acq_date,
        np.array([s[0] for s in scans], dtype=np.float64),
    )


def bin_by_time(
    spectra: np.ndarray,
    tic: np.ndarray,
    times: np.ndarray,
    n_bins: int = N_BINS,
    bin_min: float = BIN_MIN,
) -> np.ndarray:
    """[n_bins, M]: window i covers [i·bin_min, (i+1)·bin_min) min from 0; the highest-TIC scan
    of each window; windows without scans stay zero."""
    window = np.floor(np.asarray(times) / bin_min).astype(np.int64)
    out = np.zeros((n_bins, spectra.shape[1]), dtype=np.float32)
    for i in np.unique(window[(window >= 0) & (window < n_bins)]):
        scans = np.flatnonzero(window == i)
        out[i] = spectra[scans[np.argmax(tic[scans])]]
    return out


def bin_by_index(spectra: np.ndarray, tic: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """v1 binning (reproduces the 2026-10-11 data): n_bins equal windows of scan index. Equals
    bin_by_time only when every run has the same start time and scan count."""
    n = len(tic)
    out = np.zeros((n_bins, spectra.shape[1]), dtype=np.float32)
    for i in range(n_bins):
        lo, hi = int(i * n / n_bins), min(int((i + 1) * n / n_bins), n)
        if lo < hi:
            out[i] = spectra[lo + int(np.argmax(tic[lo:hi]))]
    return out


def sqrt_l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """sqrt, then L2 = 1 along the last axis (all-zero rows stay zero)."""
    s = np.sqrt(np.maximum(x, 0.0))
    return (s / (np.linalg.norm(s, axis=-1, keepdims=True) + eps)).astype(np.float32)


@dataclass
class Sample:
    stem: str
    species: str
    tissue: str
    batch: str
    chroma: np.ndarray  # [T, M]
    tic_raw: np.ndarray  # [T]
    sum_spectrum: np.ndarray  # [M]


def process_run(path: Path, binning: str = 'time') -> Sample | str:
    """A binned sample, or the reason the file was skipped."""
    meta = parse_filename(path.stem)
    if meta is None:
        return 'unparseable filename'
    scan = parse_txt(path)
    if scan is None:
        return 'blank or no scan data'
    batch = DATE_TO_BATCH.get(scan.acq_date)
    if batch is None:
        return f'unknown date {scan.acq_date}'
    binned = (
        bin_by_time(scan.spectra, scan.tic, scan.times)
        if binning == 'time'
        else bin_by_index(scan.spectra, scan.tic)
    )
    return Sample(
        stem=path.stem,
        species=meta[0],
        tissue=meta[1],
        batch=batch,
        chroma=sqrt_l2(binned),
        tic_raw=binned.sum(axis=1, dtype=np.float64).astype(np.float32),
        sum_spectrum=sqrt_l2(scan.spectra.sum(axis=0)),
    )


def write_batch(samples: list[Sample], batch_dir: Path) -> None:
    chroma_dir = batch_dir / 'chroma'
    chroma_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        safe = re.sub(r'[^\w]', '_', s.stem)
        np.savez_compressed(chroma_dir / f'{safe}.npz', chroma=s.chroma, tic_raw=s.tic_raw)
    np.save(batch_dir / 'X.npy', np.stack([s.sum_spectrum for s in samples]))
    np.save(batch_dir / 'y.npy', np.array([TISSUE_LABELS[s.tissue] for s in samples], np.int64))
    (batch_dir / 'groups.txt').write_text('\n'.join(s.species for s in samples))
    (batch_dir / 'sample_ids.txt').write_text('\n'.join(s.stem for s in samples))


def run(raw_dir: Path, out_dir: Path, binning: str = 'time') -> dict[str, list[Sample]]:
    """Process every .txt under raw_dir and write the batches. Returns samples per batch."""
    batches: dict[str, list[Sample]] = defaultdict(list)
    for path in sorted(
        p for p in Path(raw_dir).rglob('*.txt') if '.ipynb_checkpoints' not in str(p)
    ):
        res = process_run(path, binning)
        if isinstance(res, str):
            log.info('  skip %s: %s', path.name, res)
        else:
            batches[res.batch].append(res)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, samples in sorted(batches.items()):
        write_batch(samples, out_dir / name)
        log.info('%s: %d samples %s', name, len(samples), dict(Counter(s.tissue for s in samples)))
    (out_dir / 'tissue_labels.json').write_text(json.dumps(TISSUE_LABELS, indent=2))
    manifest = {
        name: {
            'n_samples': len(s),
            'tissues': dict(Counter(x.tissue for x in s)),
            'species': sorted({x.species for x in s}),
        }
        for name, s in sorted(batches.items())
    }
    (out_dir / 'batch_manifest.json').write_text(json.dumps(manifest, indent=2))
    return dict(batches)


def cli(argv: list[str] | None = None) -> None:
    ap = parser(__doc__)
    ap.add_argument('--raw-dir', type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument('--output-dir', type=Path, default=None, help='default <root>/data/fish_oil')
    ap.add_argument('--binning', choices=['time', 'index'], default='time')
    args = ap.parse_args(argv)
    setup_logging()
    run(args.raw_dir, args.output_dir or Paths(args.root).fish_oil, args.binning)
