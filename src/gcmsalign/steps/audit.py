"""Step 04 — account for every fish-oil sample, and audit the time axis of every run.

Sample audit: each raw export is classified with the same rules as step 03 (filename, blank
detection, acquisition date → batch, tissue), and the sample list used by ChromaDCNN is
cross-referenced, so samples without a .txt export are listed too.

Scan-time audit: per run, the scan count, first and last scan start time and scan interval;
per batch, their ranges. Step 03 bins on a fixed retention-time grid; binning by scan index
would only be equivalent if every run had the same start time and scan count.

Output: <root>/results/fish_oil_sample_audit.txt, <root>/results/scan_time_audit.txt
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Paths
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
CHROMA_DCNN_IDS = Path('/Users/woodj/Desktop/chroma-dcnn/data/fish_oil/sample_ids.txt')


@dataclass
class RunHeader:
    date: str | None
    blank: bool
    times: list[float]


def read_header(path: Path) -> RunHeader:
    """Acquisition date, blank flag and scan start times, without parsing the spectra."""
    date, times = None, []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            line = raw.strip()
            if date is None and (m := ACQ_DATE.match(line)):
                date = m.group(1)
            if BLANK.match(line):
                return RunHeader(date, True, [])
            if line.startswith('start_time =') and 'end_time' in line:
                times.append(float(START_TIME.search(line).group(1)))  # type: ignore[union-attr]
    return RunHeader(date, False, times)


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


def run(raw_dir: Path, paths: Paths, reference_ids: Path | None = CHROMA_DCNN_IDS) -> None:
    files = sorted(p for p in Path(raw_dir).rglob('*.txt') if '.ipynb_checkpoints' not in str(p))
    write_report(paths.results / 'fish_oil_sample_audit.txt', sample_audit(files, reference_ids))
    write_report(paths.results / 'scan_time_audit.txt', scan_time_report(files))


def cli(argv: list[str] | None = None) -> None:
    ap = parser(__doc__)
    ap.add_argument('--raw-dir', type=Path, default=DEFAULT_RAW_DIR)
    args = ap.parse_args(argv)
    setup_logging()
    run(args.raw_dir, Paths(args.root))
