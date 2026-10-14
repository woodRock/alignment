"""Step 01 — download and build the EI-MS spectral library for pretraining (step 02).

Sources: the MoNA GC-MS export (JSON) and MassBank EU records (plain text, GitHub release).
Records are filtered to EI single-stage spectra with an InChIKey, deduplicated by InChIKey
(MoNA preferred, then most peaks), binned to 1 Da (0–999 m/z), sqrt-compressed and
L2-normalised, and split train/val per source.

Output: <root>/data/pretraining/spectra.h5 with /spectra [N, 1000] float32, /inchikeys,
/smiles, /sources ("mona" | "massbank"), /split ("train" | "val").
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import h5py
import numpy as np
import requests

from ..config import MZ_BINS, Paths
from ._common import parser, setup_logging

log = logging.getLogger('warpalign')

MONA_URL = 'https://mona.fiehnlab.ucdavis.edu/downloads/retrieve/MoNA-export-GC-MS_Spectra.json.zip'
MASSBANK_RELEASE_API = 'https://api.github.com/repos/MassBank/MassBank-data/releases/latest'
_HEADERS = {
    'User-Agent': 'warpalign/2 (spectral library download)',
    'Accept': 'application/json, */*',
}
# MoNA's instrument field. Note: 'GC-MS' does not by itself rule out chemical ionisation.
_MONA_EI = re.compile(r'GC-EI|GC/EI|EI-B|GC-MS|gas chromatograph', re.IGNORECASE)
_MB_EI_INSTRUMENT = re.compile(r'\bEI\b|GC-EI|EI-B', re.IGNORECASE)
_PEAK = re.compile(r'(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)')

Record = dict
Downloader = Callable[[str, Path], None]


# ── Downloads ─────────────────────────────────────────────────────────────────


def download(url: str, dest: Path) -> None:
    """Stream url to dest."""
    resp = requests.get(url, stream=True, timeout=300, headers=_HEADERS)
    resp.raise_for_status()
    with open(dest, 'wb') as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)


def fetch_mona(raw_dir: Path, force: bool = False, get: Downloader = download) -> Path:
    """The MoNA GC-MS JSON (downloaded and extracted unless already present)."""
    json_path = raw_dir / 'mona_gcms.json'
    if json_path.exists() and not force:
        return json_path
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / 'mona_gcms.zip'
    log.info('Downloading the MoNA GC-MS export …')
    get(MONA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith('.json')]
        if not names:
            raise RuntimeError(f'no JSON file in {zip_path}')
        json_path.write_bytes(zf.read(names[0]))
    zip_path.unlink()
    return json_path


def fetch_massbank(
    raw_dir: Path,
    force: bool = False,
    get: Downloader = download,
    release_api: Callable[[], dict] | None = None,
) -> Path:
    """MassBank record files (latest GitHub release), flattened into raw_dir/massbank_records."""
    records_dir = raw_dir / 'massbank_records'
    if records_dir.exists() and any(records_dir.rglob('*.txt')) and not force:
        return records_dir
    release = release_api() if release_api else _latest_release()
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"massbank_{release['tag_name']}.zip"
    log.info('Downloading MassBank release %s …', release['tag_name'])
    get(release['zipball_url'], zip_path)
    records_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith('.txt'):
                (records_dir / Path(member).name).write_bytes(zf.read(member))
    zip_path.unlink()
    return records_dir


def _latest_release() -> dict:
    resp = requests.get(MASSBANK_RELEASE_API, timeout=30, headers=_HEADERS)
    resp.raise_for_status()
    return resp.json()


# ── Parsing ───────────────────────────────────────────────────────────────────


def parse_mona(json_path: Path) -> Iterator[Record]:
    """EI GC-MS records with an InChIKey and at least one peak ≤ 1000 m/z."""
    records = json.loads(Path(json_path).read_text(encoding='utf-8'))
    for rec in records:
        meta = {m['name'].lower(): m['value'] for m in rec.get('metaData', [])}
        if not _MONA_EI.search(meta.get('instrument type', '') or meta.get('instrument', '')):
            continue
        compound = rec.get('compound', [{}])
        if isinstance(compound, list):
            compound = compound[0] if compound else {}
        inchikey = compound.get('inchiKey', '')
        peaks = [
            (float(a), float(b))
            for a, b in _PEAK.findall(rec.get('spectrum', ''))
            if float(a) <= 1000
        ]
        if len(inchikey) < 14 or not peaks:
            continue
        smiles = next(
            (
                m.get('value', '')
                for m in compound.get('metaData', [])
                if m.get('name', '').lower() == 'smiles'
            ),
            '',
        )
        yield {'inchikey': inchikey, 'smiles': smiles, 'peaks': peaks, 'source': 'mona'}


def is_ei_ms(text: str) -> bool:
    """A MassBank record is EI-MS if AC$INSTRUMENT_TYPE names an EI instrument and it is a
    single-stage spectrum (AC$MASS_SPECTROMETRY: MS_TYPE MS). (v1 tested for the substring "EI"
    anywhere in the record, which also passed compound and author names such as caffeine,
    cysteine or Leibniz, and so let LC-ESI MS/MS spectra in.)"""
    instrument, ms_type = '', ''
    for line in text.splitlines():
        if line.startswith('AC$INSTRUMENT_TYPE:'):
            instrument = line.partition(':')[2].strip()
        elif line.startswith('AC$MASS_SPECTROMETRY: MS_TYPE'):
            ms_type = line.split()[-1]
    return bool(_MB_EI_INSTRUMENT.search(instrument)) and ms_type.upper() == 'MS'


def parse_massbank_record(text: str) -> Record | None:
    """One MassBank record → a library record, or None if it is not EI-MS or lacks an InChIKey
    or peaks."""
    if not is_ei_ms(text):
        return None
    fields: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    in_peaks = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith('PK$PEAK:'):
            in_peaks = True
        elif in_peaks and line.startswith('//'):
            in_peaks = False
        elif in_peaks:
            parts = line.split()
            try:
                mz, intensity = float(parts[0]), float(parts[1])
            except (IndexError, ValueError):
                continue
            if mz <= 1000:
                peaks.append((mz, intensity))
        elif ':' in line:
            key, _, value = line.partition(':')
            fields[key.strip()] = value.strip()
    inchikey = fields.get('CH$INCHIKEY', '') or fields.get('CH$INCHI_KEY', '')
    if len(inchikey) < 14 or not peaks:
        return None
    return {
        'inchikey': inchikey,
        'smiles': fields.get('CH$SMILES', ''),
        'peaks': peaks,
        'source': 'massbank',
    }


def parse_massbank(records_dir: Path) -> Iterator[Record]:
    for path in sorted(Path(records_dir).rglob('*.txt')):
        rec = parse_massbank_record(path.read_text(encoding='utf-8', errors='replace'))
        if rec is not None:
            yield rec


def count_massbank_filters(records_dir: Path) -> tuple[int, int, int]:
    """(records, kept by the v1 substring filter, kept by is_ei_ms)."""
    n = old = new = 0
    for path in Path(records_dir).rglob('*.txt'):
        text = path.read_text(encoding='utf-8', errors='replace')
        n += 1
        old += 'EI' in text.upper()
        new += is_ei_ms(text)
    return n, old, new


# ── Library ───────────────────────────────────────────────────────────────────


def deduplicate(records: Iterable[Record]) -> list[Record]:
    """One record per InChIKey: MoNA first, then the most peaks."""
    by_key: dict[str, list[Record]] = defaultdict(list)
    for rec in records:
        by_key[rec['inchikey']].append(rec)
    return [
        min(group, key=lambda r: (r['source'] != 'mona', -len(r['peaks'])))
        for group in by_key.values()
    ]


def bin_and_normalise(peaks: Iterable[tuple[float, float]]) -> np.ndarray:
    """1-Da bins (max intensity per bin), sqrt, L2 = 1 (zeros if empty)."""
    vec = np.zeros(MZ_BINS, dtype=np.float64)
    for mz, intensity in peaks:
        idx = round(mz)
        if 0 <= idx < MZ_BINS and intensity > 0:
            vec[idx] = max(vec[idx], float(intensity))
    vec = np.sqrt(vec)
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 1e-8 else vec).astype(np.float32)


def train_val_split(sources: list[str], val_fraction: float, seed: int) -> np.ndarray:
    """'train'/'val' per record; at least one 'val' per source, chosen at random."""
    rng = np.random.default_rng(seed)
    src = np.array(sources)
    split = np.array(['train'] * len(src), dtype=object)
    for s in np.unique(src):
        idx = np.flatnonzero(src == s)
        rng.shuffle(idx)
        split[idx[: max(1, int(len(idx) * val_fraction))]] = 'val'
    return split


def write_library(
    records: list[Record], out_path: Path, val_fraction: float = 0.1, seed: int = 42
) -> dict[str, int]:
    """Write the HDF5 library. Returns {'total', 'train', 'val'}."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spectra = (
        np.stack([bin_and_normalise(r['peaks']) for r in records])
        if records
        else np.zeros((0, MZ_BINS), np.float32)
    )
    split = train_val_split([r['source'] for r in records], val_fraction, seed)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('spectra', data=spectra, compression='gzip', compression_opts=4)
        for key in ('inchikeys', 'smiles', 'sources'):
            field = key[:-1] if key != 'smiles' else 'smiles'
            f.create_dataset(key, data=np.array([r.get(field, '').encode() for r in records]))
        f.create_dataset('split', data=np.array([s.encode() for s in split]))
        f.attrs['mz_max'] = MZ_BINS
    counts = {
        'total': len(records),
        'train': int((split == 'train').sum()),
        'val': int((split == 'val').sum()),
    }
    f_attrs = ', '.join(f'{k} {v}' for k, v in counts.items())
    log.info('Library → %s (%s)', out_path, f_attrs)
    return counts


def build(
    paths: Paths, force: bool = False, mona_only: bool = False, get: Downloader = download
) -> dict[str, int]:
    raw = paths.data / 'pretraining' / 'raw'
    records = list(parse_mona(fetch_mona(raw, force, get)))
    log.info('MoNA: %d EI records', len(records))
    if not mona_only:
        mb = list(parse_massbank(fetch_massbank(raw, force, get)))
        log.info('MassBank: %d EI records', len(mb))
        records += mb
    unique = deduplicate(records)
    log.info('Deduplicated: %d → %d', len(records), len(unique))
    return write_library(unique, paths.pretrain_h5)


def cli(argv: list[str] | None = None) -> None:
    ap = parser(__doc__)
    ap.add_argument('--force', action='store_true', help='re-download even if cached')
    ap.add_argument('--mona-only', action='store_true', help='skip MassBank')
    args = ap.parse_args(argv)
    setup_logging()
    build(Paths(args.root), args.force, args.mona_only)
