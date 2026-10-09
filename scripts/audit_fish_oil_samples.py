"""
Account for every fish oil sample that does not reach one of the four evaluated batches.

Mirrors the skip logic in preprocess_fish_oil_batches.py exactly (same filename regex,
blank detection, DATE_TO_BATCH table and tissue patterns) and cross-references the
103-sample set used by ChromaDCNN, so that samples whose .txt export is missing from
raw-subsetted-CVSs are also listed ("not in raw subset").

Usage:
  python scripts/audit_fish_oil_samples.py --raw-dir /path/to/raw-subsetted-CVSs
Output: results/fish_oil_sample_audit.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import preprocess_fish_oil_batches as pp

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    main(ap.parse_args().raw_dir)
