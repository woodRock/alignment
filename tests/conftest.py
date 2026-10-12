"""Shared fixtures: synthetic data and a tiny model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import synth

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / 'scripts'


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(scope='session')
def synthetic_pair():
    """Batch A at the reference positions, batch B shifted by synth.SHIFT bins:
    ((chromas_a, raw_a), (chromas_b, raw_b))."""
    r = np.random.default_rng(7)
    specs = synth.compound_spectra(len(synth.POSITIONS), r)
    a = synth.make_batch(4, synth.POSITIONS, 0, specs, r, jitter=0.2)
    b = synth.make_batch(4, synth.POSITIONS, synth.SHIFT, specs, r, jitter=0.2)
    return a, b


@pytest.fixture(scope='session')
def batches(synthetic_pair):
    """The synthetic pair as gcmsalign Batch objects."""
    from gcmsalign.data import Batch

    (ca, ra), (cb, rb) = synthetic_pair
    return (
        Batch('a', ca, ra, tuple(f'a{i}' for i in range(len(ca)))),
        Batch('b', cb, rb, tuple(f'b{i}' for i in range(len(cb)))),
    )
