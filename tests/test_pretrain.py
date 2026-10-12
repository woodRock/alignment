"""02_train_warp_transformer.py and the EI filter of 01_download_pretrain_data.py."""

import h5py
import numpy as np
import pytest
import torch

from gcmsalign.steps import download_library as dl
from gcmsalign.steps import pretrain as tr
from gcmsalign.steps.pretrain import SyntheticConfig, TrainConfig

CFG = SyntheticConfig()
T = 200


def test_target_warp_continues_the_offset_at_both_ends():
    """Section C, test 4 (fix 4): with offset +10, the first and last bins follow t + 10 (v1
    held them flat at the outer peaks' positions). Not clipped (see the docstring)."""
    ref = np.array([30, 80, 140, 170])
    w = tr.target_warp(ref, ref + 10)
    t = np.arange(T)
    assert np.allclose(w[:5], t[:5] + 10) and np.allclose(w[-5:], t[-5:] + 10)
    assert np.allclose(w, t + 10)


def test_target_warp_interpolates_between_peaks():
    w = tr.target_warp(np.array([50, 150]), np.array([45, 160]))
    assert w[50] == 45 and w[150] == 160 and w[100] == pytest.approx(102.5)
    assert w[0] == pytest.approx(-5) and w[199] == pytest.approx(209)


@pytest.mark.parametrize('seed', range(20))
def test_drifted_positions_stay_ordered_and_inside(seed):
    rng = np.random.default_rng(seed)
    ref = tr.random_positions(rng, CFG)
    off = int(rng.choice([-25, 25]))
    pks = tr.drifted_positions(ref, off, 4, rng)
    assert np.all(np.diff(pks) >= 1) and pks.min() >= 2 and pks.max() <= T - 3


def test_random_positions_spacing(rng):
    pos = tr.random_positions(rng, CFG)
    assert np.all(np.diff(pos) >= CFG.min_spacing) and pos.min() >= 10 and pos.max() < T - 10


def test_build_chromatogram_applies_sqrt_once():
    """8b: a single compound's bins are proportional to sqrt(its intensity spectrum)."""
    spec = np.zeros((1, 1000), np.float32)
    spec[0, [100, 200]] = [0.8, 0.6]  # intensities (L2 = 1)
    c = tr.build_chromatogram(
        spec,
        np.array([100]),
        np.array([1.0]),
        np.array([1.0]),
        np.random.default_rng(0),
        SyntheticConfig(noise_scale=0.0),
    )
    expect = np.sqrt([0.8, 0.6]) / np.linalg.norm(np.sqrt([0.8, 0.6]))
    assert np.allclose(c[100, [100, 200]], expect, atol=1e-6)


def test_peak_width_range_is_about_one_bin():
    """8c: after apex-scan binning real peaks span about one bin (v1 used σ = 2.5 bins)."""
    lo, hi = CFG.sigma_range
    assert 0.4 <= lo < hi <= 1.6


def test_load_library_squares_the_stored_sqrt_spectra(tmp_path):
    inten = np.array([[0.0, 0.64, 0.36] + [0.0] * 997, [0.25] * 4 + [0.0] * 996])
    stored = np.sqrt(inten)
    stored /= np.linalg.norm(stored, axis=1, keepdims=True)
    with h5py.File(tmp_path / 'lib.h5', 'w') as f:
        f['spectra'] = stored.astype(np.float32)
        f['split'] = np.array([b'train', b'val'])
    lib = tr.load_library(tmp_path / 'lib.h5')
    assert lib.shape == (1, 1000)
    assert np.allclose(lib[0, :3], np.array([0, 0.64, 0.36]) / np.linalg.norm([0.64, 0.36]))


def _library(n=40, seed=0):
    rng = np.random.default_rng(seed)
    lib = np.zeros((n, 1000), np.float32)
    for i in range(n):
        lib[i, rng.choice(np.arange(40, 500), 10, replace=False)] = rng.uniform(0.1, 1, 10)
    return lib / np.linalg.norm(lib, axis=1, keepdims=True)


def test_mini_dataset_shapes_and_targets():
    x, w, sid = tr.mini_dataset(_library(), 6, np.random.default_rng(1))
    assert x.shape == (6, T, 1000) and w.shape == (6, T) and sid.tolist() == [0, 0, 0, 1, 1, 1]
    assert np.all(np.diff(w, axis=1) > 0)
    norms = np.linalg.norm(x, axis=2)
    assert np.isfinite(x).all()
    assert np.all(np.isclose(norms, 1.0, atol=1e-4) | (norms == 0))  # unit bins or empty start
    assert np.all(np.diff((norms > 0).astype(int), axis=1) >= 0)  # zeros only at the start


def test_training_runs_and_is_reproducible():
    """8h: torch is seeded, so two short runs give identical weights."""
    lib = _library()
    cfg = TrainConfig(iterations=4, seed=3, log_every=2)
    m1, h1 = tr.train(lib, torch.device('cpu'), cfg)
    m2, h2 = tr.train(lib, torch.device('cpu'), cfg)
    assert h1 == h2 and len(h1) == 2 and all(np.isfinite(h1))
    assert all(
        torch.equal(p, q)
        for p, q in zip(m1.state_dict().values(), m2.state_dict().values(), strict=False)
    )


def test_run_writes_checkpoint_and_report(tmp_path):
    with h5py.File(tmp_path / 'lib.h5', 'w') as f:
        f['spectra'] = np.sqrt(_library())
        f['split'] = np.array([b'train'] * 40)
    hist = tr.run(
        tmp_path / 'lib.h5',
        tmp_path / 'ck.pt',
        'cpu',
        TrainConfig(iterations=2, seed=0, log_every=2),
    )
    report = (tmp_path / 'ck.txt').read_text()
    assert (
        (tmp_path / 'ck.pt').exists() and 'final loss' in report and '40 training spectra' in report
    )
    assert len(hist) == 1


# ── 01: MassBank EI filter (8d) ───────────────────────────────────────────────

EI = 'CH$NAME: Caffeine\nAC$INSTRUMENT_TYPE: EI-B\nAC$MASS_SPECTROMETRY: MS_TYPE MS\n'
ESI = (
    'CH$NAME: Cysteine\nAUTHORS: Leibniz Institute\nAC$INSTRUMENT_TYPE: LC-ESI-QTOF\n'
    'AC$MASS_SPECTROMETRY: MS_TYPE MS2\n'
)


@pytest.mark.parametrize(
    'text,expected',
    [
        (EI, True),
        (ESI, False),  # 'EI' in names only
        ('AC$INSTRUMENT_TYPE: GC-EI-TOF\nAC$MASS_SPECTROMETRY: MS_TYPE MS\n', True),
        ('AC$INSTRUMENT_TYPE: EI-B\nAC$MASS_SPECTROMETRY: MS_TYPE MS2\n', False),
        ('AC$INSTRUMENT_TYPE: EI-B\n', False),  # MS type missing
    ],
)
def test_is_ei_ms(text, expected):
    assert dl.is_ei_ms(text) is expected


def test_old_substring_filter_would_have_passed_esi():
    assert 'EI' in ESI.upper() and not dl.is_ei_ms(ESI)


def test_mini_dataset_has_an_empty_start_per_batch():
    """Round-3 review, issue 2: each batch's runs are all-zero before one acquisition start,
    as step 03 leaves the bins before a real run's start; batches can differ."""
    starts = set()
    for seed in range(20):
        x, _, sid = tr.mini_dataset(_library(), 6, np.random.default_rng(seed))
        for study in (0, 1):
            nz = np.abs(x[sid == study]).sum(axis=2) > 0  # [n, T]
            first = nz.argmax(axis=1)
            assert len(set(first.tolist())) == 1  # one start per batch
            s = int(first[0])
            assert s <= CFG.max_acquisition_start
            assert np.all(x[sid == study][:, :s] == 0) and np.all(nz[:, s:])
            starts.add(s)
    assert len(starts) > 5


def test_acquisition_start_can_be_switched_off():
    x, _, _ = tr.mini_dataset(
        _library(), 4, np.random.default_rng(0), SyntheticConfig(max_acquisition_start=0)
    )
    assert np.all(np.abs(x).sum(axis=2) > 0)


@pytest.mark.slow
def test_pretraining_learns_well_below_the_no_warp_loss():
    """Regression test for the free-end warp that stopped pretraining learning: on v1-like
    synthetic data (wide peaks, no empty starts), 400 iterations must take the warp loss well
    below what predicting no warp scores (v1's model and the current one reach ~0.3 of it).
    Uses the real spectral library when present, otherwise the synthetic one."""
    from gcmsalign.config import Paths

    h5 = Paths().pretrain_h5
    lib = tr.load_library(h5) if h5.exists() else _library(400)
    cfg = SyntheticConfig(sigma_range=(2.5, 2.5), max_acquisition_start=0)
    rng = np.random.default_rng(1)
    no_warp = []
    for _ in range(60):
        d = tr.mini_dataset(lib, int(rng.integers(4, 17)), rng, cfg)
        if d is not None:
            no_warp.append(np.mean(((d[1] - np.arange(T)) / (T - 1)) ** 2))
    torch.set_num_threads(4)
    _, hist = tr.train(lib, torch.device('cpu'), TrainConfig(iterations=400, log_every=100), cfg)
    assert hist[-1] < 0.6 * float(np.mean(no_warp)), (hist, float(np.mean(no_warp)))
