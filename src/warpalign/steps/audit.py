"""Step 04 — account for every fish-oil sample, and audit the time axis of every run.

Sample audit: each raw export is classified with the same rules as step 03 (filename, blank
detection, acquisition date → batch, tissue), and the sample list used by ChromaDCNN is
cross-referenced, so samples without a .txt export are listed too.

Scan-time audit: per run, the scan count, first and last scan start time and scan interval;
per batch, their ranges. Step 03 bins on a fixed retention-time grid; binning by scan index
would only be equivalent if every run had the same start time and scan count.

Empty-window audit: per run, the 0.225-min windows inside its acquired range (the window of
its first scan → 44.99 min) with no ions, and whether their scans are missing from the export
or present with zero readings (num_readings = 0); where they fall (early < 15 min, mid
15–30 min, late ≥ 30 min) and how contiguous they are. If step 03's data exist, each run's
empty windows and acquisition start (data.acquisition_start, the first non-zero bin) are
compared with the scan headers.

Output: <root>/results/fish_oil_sample_audit.txt, <root>/results/scan_time_audit.txt
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .._compat import acquisition_start
from ..config import BIN_MIN, N_BINS, Paths
from ._common import parser, setup_logging, table, write_report
from .preprocess import (
    ACQ_DATE,
    BLANK,
    DATE_TO_BATCH,
    DEFAULT_RAW_DIR,
    START_TIME,
    TISSUE_LABELS,
    parse_filename,
)

EVALUATED = frozenset({'batch_sep2015', 'batch_jan2016', 'batch_apr2016', 'batch_jul2016'})
READINGS = re.compile(r'num_readings\s*=\s*(\d+)')
REGIONS = (('early', 0.0, 15.0), ('mid', 15.0, 30.0), ('late', 30.0, 45.0))  # minutes
CHROMA_DCNN_IDS = Path('/Users/woodj/Desktop/chroma-dcnn/data/fish_oil/sample_ids.txt')


@dataclass
class RunHeader:
    date: str | None
    blank: bool
    times: list[float]
    readings: list[int] = field(default_factory=list)  # num_readings of each scan


def read_header(path: Path) -> RunHeader:
    """Acquisition date, blank flag and scan start times, without parsing the spectra."""
    date, times, readings = None, [], []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            line = raw.strip()
            if date is None and (m := ACQ_DATE.match(line)):
                date = m.group(1)
            if BLANK.match(line):
                return RunHeader(date, True, [])
            if line.startswith('start_time =') and 'end_time' in line:
                times.append(float(START_TIME.search(line).group(1)))  # type: ignore[union-attr]
            elif line.startswith('num_readings') and (m := READINGS.match(line)):
                readings.append(int(m.group(1)))
    return RunHeader(date, False, times, readings)


def classify(path: Path) -> str:
    """Why a file does or does not reach an evaluated batch ('OK -> <batch>' if it does)."""
    meta = parse_filename(path.stem)
    if meta is None:
        return 'unparseable filename'
    head = read_header(path)
    if head.blank:
        return 'instrument blank'
    if not head.times or head.date is None:
        return 'no scan data or no acquisition date'
    batch = DATE_TO_BATCH.get(head.date)
    if batch is None:
        return f'acquisition date {head.date} not in DATE_TO_BATCH'
    if meta[1] not in TISSUE_LABELS:
        return f"unknown tissue '{meta[1]}'"
    if batch not in EVALUATED:
        return f'assigned to {batch} (not an evaluated batch)'
    return f'OK -> {batch}'


def _norm(stem: str) -> str:
    return re.sub(r'[^\w]', '_', stem.strip())


def sample_audit(files: list[Path], reference_ids: Path | None = CHROMA_DCNN_IDS) -> list[str]:
    rows = [(p.stem, classify(p)) for p in files]
    seen = {_norm(stem) for stem, _ in rows}
    if reference_ids is not None and reference_ids.exists():
        rows += [
            (sid.strip(), "not in raw subset (in ChromaDCNN's list, no .txt export)")
            for sid in reference_ids.read_text().splitlines()
            if sid.strip() and _norm(sid) not in seen
        ]
    excluded = sorted((s, r) for s, r in rows if not r.startswith('OK'))
    return [
        f'Raw .txt files scanned : {len(files)}',
        f'Samples considered     : {len(rows)}',
        '',
        'Counts by outcome:',
        *(f'  {n:>4}  {r}' for r, n in Counter(r for _, r in rows).most_common()),
        '',
        f'Excluded samples ({len(excluded)}):',
        *(f'  {s:<40} {r}' for s, r in excluded),
    ]


@dataclass
class RunTimes:
    batch: str
    stem: str
    n_scans: int
    first: float
    last: float
    interval: float  # min
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))  # scan start times (min)
    readings: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # per scan


def run_times(files: list[Path]) -> list[RunTimes]:
    out = []
    for p in files:
        head = read_header(p)
        if head.blank or len(head.times) < 2 or parse_filename(p.stem) is None:
            continue
        t = np.asarray(head.times)
        out.append(
            RunTimes(
                DATE_TO_BATCH.get(head.date or '', f'date {head.date}'),
                p.stem,
                len(t),
                float(t[0]),
                float(t[-1]),
                float(np.median(np.diff(t))),
                t,
                np.asarray(head.readings if len(head.readings) == len(t) else [1] * len(t)),
            )
        )
    return sorted(out, key=lambda r: (r.batch, r.stem))


def times_agree(runs: list[RunTimes]) -> bool:
    """Scan count, first/last time and interval equal across runs within one scan interval."""
    if not runs:
        return True
    dt = float(np.median([r.interval for r in runs]))
    spread = [np.ptp([getattr(r, k) for r in runs]) for k in ('first', 'last', 'interval')]
    return all(s <= dt + 1e-9 for s in spread) and np.ptp([r.n_scans for r in runs]) <= 1


def scan_time_report(files: list[Path]) -> list[str]:
    runs = run_times(files)
    lines = [
        'Scan-time audit: one row per non-blank run, then per-batch summaries.',
        '',
        *table(
            ['Batch', 'Sample', 'n_scans', 'first (min)', 'last (min)', 'interval (s)'],
            [[r.batch, r.stem, r.n_scans, r.first, r.last, r.interval * 60] for r in runs],
        ),
        '',
    ]
    summary = []
    for batch in sorted({r.batch for r in runs}):
        b = [r for r in runs if r.batch == batch]
        summary.append(
            [
                batch,
                len(b),
                f'{min(r.n_scans for r in b)}–{max(r.n_scans for r in b)}',
                f'{min(r.first for r in b):.3f}–{max(r.first for r in b):.3f}',
                f'{min(r.last for r in b):.3f}–{max(r.last for r in b):.3f}',
                f'{min(r.interval for r in b) * 60:.3f}–{max(r.interval for r in b) * 60:.3f}',
            ]
        )
    lines += table(
        ['Batch', 'runs', 'n_scans', 'first (min)', 'last (min)', 'interval (s)'], summary
    )
    lines += ['', 'Decision:']
    if times_agree(runs):
        lines.append(
            '  Scan counts, first/last scan times and scan interval agree across runs to '
            'within one scan interval, so binning by scan index is equivalent to binning '
            'by time.'
        )
    else:
        lines.append(
            '  Runs DIFFER in scan count and/or acquisition time range (by more than one '
            'scan interval). Binning by scan index would stretch runs relative to each '
            'other, so step 03 bins by retention time on a fixed 0–45 min grid.'
        )
    return lines


@dataclass
class EmptyWindows:
    run: RunTimes
    start_bin: int  # window of the first scan
    empty: np.ndarray  # [N_BINS] bool: in the acquired range, no scan with readings
    no_scans: np.ndarray  # [N_BINS] bool: in the acquired range, no scan at all

    @property
    def n_range(self) -> int:
        return N_BINS - self.start_bin

    def segments(self) -> list[tuple[int, int]]:
        """Contiguous runs of empty windows as (first bin, length)."""
        out, i = [], 0
        while i < N_BINS:
            if self.empty[i]:
                j = i
                while j < N_BINS and self.empty[j]:
                    j += 1
                out.append((i, j - i))
                i = j
            else:
                i += 1
        return out


def empty_windows(r: RunTimes) -> EmptyWindows:
    """Windows of r's acquired range whose scans all have zero readings, or that have none."""
    window = np.floor(r.times / BIN_MIN).astype(np.int64)
    start = int(window.min())
    scans = np.bincount(window[window < N_BINS], minlength=N_BINS)
    ions = np.bincount(
        window[window < N_BINS], weights=r.readings[window < N_BINS] > 0, minlength=N_BINS
    )
    in_range = np.arange(N_BINS) >= start
    return EmptyWindows(r, start, in_range & (ions == 0), in_range & (scans == 0))


def region_of(bin_: int) -> str:
    t = bin_ * BIN_MIN
    return next(name for name, lo, hi in REGIONS if lo <= t < hi)


def data_tic_raw(paths: Paths | None, r: RunTimes) -> np.ndarray | None:
    """Step 03's raw TIC of run r, if the binned data exist."""
    if paths is None:
        return None
    f = paths.fish_oil / r.batch / 'chroma' / (re.sub(r'[^\w]', '_', r.stem) + '.npz')
    if not f.exists():
        return None
    with np.load(f) as z:
        return z['tic_raw'] if 'tic_raw' in z.files else None


def region_share(empty: np.ndarray, in_range: np.ndarray, lo: float, hi: float) -> str:
    """Share of the acquired-range windows in [lo, hi) min that are empty, over runs [N, T]."""
    t = np.arange(N_BINS) * BIN_MIN
    m = in_range & (t >= lo) & (t < hi)
    return f'{100 * empty[m].mean():.1f}%' if m.any() else '—'


def empty_window_report(files: list[Path], paths: Paths | None = None) -> list[str]:
    runs = [e for e in map(empty_windows, run_times(files)) if e.run.batch in EVALUATED]
    lines = [
        '',
        'Empty-window audit: windows inside each run\'s acquired range (first scan → '
        f'{N_BINS * BIN_MIN - 0.01:.2f} min) with no ions.',
        'scans 0 readings: the export has the scans, every one with num_readings = 0; '
        'no scans: the window has no scan at all.',
        '',
    ]
    rows, mismatch = [], []
    for e in runs:
        segs = e.segments()
        n_empty = int(e.empty.sum())
        tic = data_tic_raw(paths, e.run)
        if tic is not None:
            d_start = int(acquisition_start(tic[None])[0])
            d_empty = int((tic[e.start_bin :] == 0).sum())
            if d_start != e.start_bin or d_empty != n_empty:
                mismatch.append(
                    f'  {e.run.batch} {e.run.stem}: start bin {e.start_bin} (headers) vs '
                    f'{d_start} (data); empty {n_empty} vs {d_empty}'
                )
        rows.append(
            [
                e.run.batch,
                e.run.stem,
                e.start_bin,
                n_empty,
                f'{100 * n_empty / e.n_range:.1f}%',
                int(e.no_scans.sum()),
                *(
                    int(sum(e.empty[i] for i in range(e.start_bin, N_BINS) if region_of(i) == name))
                    for name, _, _ in REGIONS
                ),
                len(segs),
                max((n for _, n in segs), default=0),
            ]
        )
    region_cols = [f'{name}' for name, _, _ in REGIONS]
    lines += table(
        [
            'Batch',
            'Sample',
            'start bin',
            'empty',
            'share',
            'no scans',
            *region_cols,
            'runs',
            'longest',
        ],
        rows,
    )
    lines += [
        '',
        'Per batch: share of acquired-range windows that are empty; by region, the share of '
        'that region\'s windows; contiguous runs per sample (mean) and the longest (windows).',
    ]
    summary = []
    for batch in sorted({e.run.batch for e in runs}):
        b = [e for e in runs if e.run.batch == batch]
        empty = np.stack([e.empty for e in b])
        in_range = np.stack([np.arange(N_BINS) >= e.start_bin for e in b])
        per_bin = empty.sum(0)
        common = [i for i in range(N_BINS) if per_bin[i] >= len(b) / 2]

        summary.append(
            [
                batch,
                len(b),
                f'{100 * empty.sum() / in_range.sum():.1f}%',
                int(sum(int(e.no_scans.sum()) for e in b)),
                *(region_share(empty, in_range, lo, hi) for _, lo, hi in REGIONS),
                f'{np.mean([len(e.segments()) for e in b]):.1f}',
                max(max((n for _, n in e.segments()), default=0) for e in b),
                f'{common[0] * BIN_MIN:.1f}–{(common[-1] + 1) * BIN_MIN:.1f} min ({len(common)})'
                if common
                else '—',
            ]
        )
    lines += table(
        [
            'Batch',
            'runs',
            'empty',
            'no-scan windows',
            *region_cols,
            'runs/sample',
            'longest',
            'empty in ≥ half the runs: span (windows)',
        ],
        summary,
    )
    lines += ['', 'Comparison with step 03\'s data (acquisition start = first non-zero bin):']
    if paths is None or all(data_tic_raw(paths, e.run) is None for e in runs):
        lines.append('  no binned data found; not compared.')
    elif mismatch:
        lines += [f'  {len(mismatch)} run(s) differ:', *mismatch]
    else:
        lines.append(
            f'  all {len(runs)} runs agree: data.acquisition_start = the window of the first '
            'scan, and the empty windows are exactly the zero raw-TIC bins.'
        )
    return lines


def run(raw_dir: Path, paths: Paths, reference_ids: Path | None = CHROMA_DCNN_IDS) -> None:
    files = sorted(p for p in Path(raw_dir).rglob('*.txt') if '.ipynb_checkpoints' not in str(p))
    write_report(paths.results / 'fish_oil_sample_audit.txt', sample_audit(files, reference_ids))
    write_report(
        paths.results / 'scan_time_audit.txt',
        scan_time_report(files) + empty_window_report(files, paths),
    )


def cli(argv: list[str] | None = None) -> None:
    ap = parser(__doc__)
    ap.add_argument('--raw-dir', type=Path, default=DEFAULT_RAW_DIR)
    args = ap.parse_args(argv)
    setup_logging()
    run(args.raw_dir, Paths(args.root))
