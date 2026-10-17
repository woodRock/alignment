"""Shared plumbing for the pipeline steps: command-line options, logging, text tables, reports."""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..config import N_CALIB_STEPS, Paths

log = logging.getLogger('warpalign')


def setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format='%(message)s', force=True
    )


def parser(doc: str | None) -> argparse.ArgumentParser:
    """An ArgumentParser with --root (the project directory: data/, results/, checkpoints/)."""
    ap = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        '--root',
        type=Path,
        default=Paths().root,
        help='project directory holding data/, results/ and checkpoints/',
    )
    return ap


def add_pair_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument('--study-a', default='fish_oil/batch_sep2015')
    ap.add_argument('--study-b', default='fish_oil/batch_jul2016')


def add_model_args(ap: argparse.ArgumentParser, seed_default: int | None = 0) -> None:
    ap.add_argument('--seed', type=int, default=seed_default)
    ap.add_argument(
        '--device',
        default='cpu',
        help='cpu (default; bit-reproducible under a seed) | mps | cuda | auto',
    )
    ap.add_argument(
        '--ckpt',
        type=Path,
        default=None,
        help='checkpoint (default: <root>/checkpoints/warp_transformer_v2.pt)',
    )
    ap.add_argument(
        '--calib-steps',
        type=int,
        default=N_CALIB_STEPS,
        help=f'calibration steps (default {N_CALIB_STEPS}; fewer only for tests)',
    )


def ckpt_or_default(ckpt: Path | None, paths: Paths) -> Path:
    return ckpt if ckpt is not None else paths.default_ckpt


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file's contents (identifies a checkpoint across machines and paths)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def batch_name_of(study: str) -> str:
    """'fish_oil/batch_sep2015' → 'batch_sep2015'."""
    return Path(study).name


def title(doc: str | None) -> str:
    """First non-empty line of a module docstring."""
    return next((line for line in (doc or '').strip().splitlines() if line.strip()), '')


def table(
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
    indent: str = '  ',
    floatfmt: str = '.3f',
) -> list[str]:
    """Plain-text table: first column left-aligned, the rest right-aligned, floats formatted
    with floatfmt (strings and None are shown as given; None as '—')."""

    def cell(v: object) -> str:
        if v is None:
            return '—'
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    cells = [[cell(v) for v in row] for row in rows]
    widths = [max(len(str(h)), *(len(r[k]) for r in cells)) for k, h in enumerate(header)]

    def line(vals: Sequence[str]) -> str:
        first = f'{vals[0]:<{widths[0]}}'
        rest = '  '.join(f'{v:>{w}}' for v, w in zip(vals[1:], widths[1:], strict=True))
        return (indent + first + ('  ' + rest if rest else '')).rstrip()

    return [line([str(h) for h in header]), *(line(r) for r in cells)]


def save_npz(path: Path, arrays: Mapping[str, object]) -> None:
    """np.savez of named arrays (one place for the numpy-stub workaround: the stubs map
    **kwargs onto savez's allow_pickle parameter)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)  # type: ignore[arg-type]


def write_report(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(lines) + '\n'
    path.write_text(text)
    log.info('%s\n  results → %s', text, path)
