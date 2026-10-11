"""
Account for every fish oil sample that does not reach one of the four evaluated batches.

Mirrors the skip logic in 03_preprocess_fish_oil_batches.py exactly (same filename regex,
blank detection, DATE_TO_BATCH table and tissue patterns) and cross-references the
103-sample set used by ChromaDCNN, so that samples whose .txt export is missing from
raw-subsetted-CVSs are also listed ("not in raw subset").

It also audits the time axis of every run (scan count, first and last scan start time, scan
interval), because 03 bins on a fixed retention-time grid: runs with different start times or
scan counts would be stretched relative to each other by index binning.

Usage:
  python scripts/04_audit_fish_oil_samples.py --raw-dir /path/to/raw-subsetted-CVSs
Output: results/fish_oil_sample_audit.txt, results/scan_time_audit.txt
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

import numpy as np
from pathlib import Path

import importlib.util
_spec = importlib.util.spec_from_file_location(
    'preprocess', Path(__file__).parent / '03_preprocess_fish_oil_batches.py')
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)

ROOT        = Path(__file__).resolve().parents[1]
EVALUATED   = {"batch_sep2015", "batch_jan2016", "batch_apr2016", "batch_jul2016"}
CHROMA_DCNN = Path('/Users/woodj/Desktop/chroma-dcnn/data/fish_oil/sample_ids.txt')


def _norm(stem: str) -> str:
    return re.sub(r"[^\w]", "_", stem.strip())


def _header(path: Path) -> tuple[str | None, bool, bool]:
    """(acquisition date, is_blank, has_scans) from the header, as parse_txt reads it."""
    date, blank = None, False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if date is None and (m := pp._ACQ_DATE_RE.match(line)):
                date = m.group(1)
            if pp._BLANK_RE.match(line):
                blank = True
                break
            if line.startswith("ScanHeader #"):
                return date, blank, True
    return date, blank, False


def classify(path: Path) -> str:
    meta = pp._parse_filename(path.stem)
    if meta is None:
        return "unparseable filename"
    _, tissue = meta
    date, blank, has_scans = _header(path)
    if blank:
        return "instrument blank"
    if not has_scans or date is None:
        return "no scan data or no acquisition date"
    batch = pp.DATE_TO_BATCH.get(date)
    if batch is None:
        return f"acquisition date {date} not in DATE_TO_BATCH"
    if pp.TISSUE_LABELS.get(tissue) is None:
        return f"unknown tissue '{tissue}'"
    if batch not in EVALUATED:
        return f"assigned to {batch} (not an evaluated batch)"
    return f"OK -> {batch}"


def scan_times(path: Path) -> tuple[str | None, list[float]] | None:
    """(acquisition date, scan start times in min), or None for blanks and files with no scans."""
    date, times = None, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if date is None and (m := pp._ACQ_DATE_RE.match(line)):
                date = m.group(1)
            if pp._BLANK_RE.match(line):
                return None
            if line.startswith("start_time =") and "end_time" in line:
                times.append(float(pp._START_TIME_RE.search(line).group(1)))
    return (date, times) if times else None


def scan_time_report(files: list[Path]) -> list[str]:
    """Per-file and per-batch scan counts, time ranges and intervals, plus the binning verdict."""
    per_file = []
    for path in files:
        res = scan_times(path)
        if res is None or pp._parse_filename(path.stem) is None:
            continue
        date, t = res
        t = np.asarray(t)
        per_file.append((pp.DATE_TO_BATCH.get(date, f"date {date}"), path.stem, len(t),
                         t[0], t[-1], float(np.median(np.diff(t)))))
    lines = ["Scan-time audit: one row per non-blank run, then per-batch summaries.", "",
             f"  {'Batch':<16} {'Sample':<40} {'n_scans':>8} {'first (min)':>12} "
             f"{'last (min)':>11} {'interval (s)':>13} {'scans/min':>10}"]
    for b, stem, n, t0, t1, dt in sorted(per_file):
        lines.append(f"  {b:<16} {stem:<40} {n:>8} {t0:>12.3f} {t1:>11.3f} "
                     f"{dt * 60:>13.3f} {1 / dt:>10.1f}")
    lines += ["", f"  {'Batch':<16} {'runs':>4}   {'n_scans min/mean/max':>22}   "
              f"{'first min/max':>14}   {'last min/max':>15}   {'interval (s) min/max':>21}"]
    batches = sorted({r[0] for r in per_file})
    stats = {}
    for b in batches:
        v = np.array([r[2:] for r in per_file if r[0] == b], dtype=float)
        stats[b] = v
        lines.append(f"  {b:<16} {len(v):>4}   {v[:, 0].min():>6.0f} {v[:, 0].mean():>7.1f} "
                     f"{v[:, 0].max():>7.0f}   {v[:, 1].min():>6.3f} {v[:, 1].max():>7.3f}   "
                     f"{v[:, 2].min():>7.3f} {v[:, 2].max():>7.3f}   "
                     f"{v[:, 3].min() * 60:>10.3f} {v[:, 3].max() * 60:>10.3f}")
    allv = np.concatenate(list(stats.values()))
    dt = float(np.median(allv[:, 3]))
    agree = all(np.ptp(allv[:, k]) <= dt + 1e-9 for k in (1, 2, 3)) and np.ptp(allv[:, 0]) <= 1
    lines += ["", "Decision:"]
    if agree:
        lines.append("  Scan counts, first/last scan times and scan interval agree across runs to "
                     "within one scan interval, so binning by scan index is equivalent to binning "
                     "by time.")
    else:
        lines.append("  Runs DIFFER in scan count and/or acquisition time range (by more than one "
                     f"scan interval, {dt * 60:.2f} s). Binning by scan index would stretch runs "
                     "relative to each other, so 03 bins by retention time on a fixed "
                     "0–45 min grid (0.225-min windows; windows outside a run's range are zero).")
    return lines


def main(raw_dir: Path) -> None:
    files = sorted(p for p in raw_dir.rglob("*.txt") if ".ipynb_checkpoints" not in str(p))
    rows  = [(p.stem, classify(p)) for p in files]
    seen  = {_norm(stem) for stem, _ in rows}

    if CHROMA_DCNN.exists():
        for sid in CHROMA_DCNN.read_text().splitlines():
            if sid.strip() and _norm(sid) not in seen:
                rows.append((sid.strip(), "not in raw subset (in ChromaDCNN's 103, no .txt export)"))

    excluded = [(s, r) for s, r in rows if not r.startswith("OK")]
    counts   = Counter(r for _, r in rows)
    lines = [f"Raw .txt files scanned : {len(files)}  ({raw_dir})",
             f"Samples considered     : {len(rows)}",
             "", "Counts by outcome:"]
    lines += [f"  {n:>4}  {r}" for r, n in counts.most_common()]
    lines += ["", f"Excluded samples ({len(excluded)}):"]
    lines += [f"  {s:<40} {r}" for s, r in sorted(excluded)]
    report = "\n".join(lines)
    print(report)
    out = ROOT / "results" / "fish_oil_sample_audit.txt"
    out.write_text(report + "\n")
    print(f"\n  results → {out}")

    scan = "\n".join(scan_time_report(files))
    out = ROOT / "results" / "scan_time_audit.txt"
    out.write_text(scan + "\n")
    print("\n" + scan + f"\n\n  results → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    main(ap.parse_args().raw_dir)
