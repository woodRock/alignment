"""Shared fixtures: import paths, script loading, synthetic data."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS   = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
ROOT    = SCRIPTS.parent
for p in (ROOT / 'src', TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import synth  # noqa: E402


def load_script(filename: str):
    """Import a numbered script (e.g. '05_literature_baselines.py') as a module."""
    spec = importlib.util.spec_from_file_location(Path(filename).stem.replace('-', '_'),
                                                  SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(scope='session')
def synthetic_pair():
    """Batch A at the reference positions, batch B shifted by synth.SHIFT bins."""
    r = np.random.default_rng(7)
    specs = synth.compound_spectra(len(synth.POSITIONS), r)
    a = synth.make_batch(4, synth.POSITIONS, 0, specs, r, jitter=0.2)
    b = synth.make_batch(4, synth.POSITIONS, synth.SHIFT, specs, r, jitter=0.2)
    return a, b


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: end-to-end script runs (minutes)')
    config.addinivalue_line('markers', 'real: needs the real fish-oil data under data/fish_oil')
