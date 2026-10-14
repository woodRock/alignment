"""01 (library parsing, no network), 04 (sample accounting), 03 (index mode, CLI), data loading,
and edge cases of the baselines."""

import json
import subprocess
import sys

import h5py
import numpy as np
import pytest
from conftest import SCRIPTS
from test_preprocess import run, write_raw

from warpalign.config import Paths
from warpalign.steps import audit
from warpalign.steps import download_library as dl
from warpalign.steps import preprocess as pp

# ── 01 ────────────────────────────────────────────────────────────────────────

MB_RECORD = '''ACCESSION: TEST000001
CH$NAME: Hexadecanoic acid
CH$INCHIKEY: IPCSVZSSVZVIGE-UHFFFAOYSA-N
AC$INSTRUMENT_TYPE: EI-B
AC$MASS_SPECTROMETRY: MS_TYPE MS
PK$PEAK: m/z int. rel.int.
  43.0 999 999
  60.0 500 500
  1200.0 10 10
//
'''


def test_parse_massbank_record(tmp_path):
    f = tmp_path / 'rec.txt'
    f.write_text(MB_RECORD)
    rec = dl.parse_massbank_record(f.read_text())
    assert rec['inchikey'] == 'IPCSVZSSVZVIGE-UHFFFAOYSA-N' and rec['source'] == 'massbank'
    assert rec['peaks'] == [(43.0, 999.0), (60.0, 500.0)]  # m/z > 1000 dropped


def test_parse_massbank_rejects_non_ei_and_missing_inchikey(tmp_path):
    f = tmp_path / 'rec.txt'
    f.write_text(MB_RECORD.replace('EI-B', 'LC-ESI-QTOF'))
    assert dl.parse_massbank_record(f.read_text()) is None
    f.write_text(MB_RECORD.replace('CH$INCHIKEY: IPCSVZSSVZVIGE-UHFFFAOYSA-N\n', ''))
    assert dl.parse_massbank_record(f.read_text()) is None


def test_count_massbank_filters(tmp_path):
    (tmp_path / 'a.txt').write_text(MB_RECORD)
    (tmp_path / 'b.txt').write_text(
        MB_RECORD.replace('EI-B', 'LC-ESI-QTOF').replace('Hexadecanoic acid', 'Caffeine')
    )  # 'EI' in the name
    (tmp_path / 'c.txt').write_text('CH$NAME: water\n')
    assert dl.count_massbank_filters(tmp_path) == (3, 2, 1)


def test_parse_mona(tmp_path):
    recs = [
        {
            'metaData': [{'name': 'instrument type', 'value': 'GC-EI-TOF'}],
            'compound': [
                {
                    'inchiKey': 'IPCSVZSSVZVIGE-UHFFFAOYSA-N',
                    'metaData': [{'name': 'SMILES', 'value': 'CCCC'}],
                }
            ],
            'spectrum': '43:100 60.1:50 1500:3',
        },
        {
            'metaData': [{'name': 'instrument type', 'value': 'LC-ESI-QTOF'}],
            'compound': [{'inchiKey': 'IPCSVZSSVZVIGE-UHFFFAOYSA-N'}],
            'spectrum': '43:1',
        },
        {
            'metaData': [{'name': 'instrument type', 'value': 'GC-EI-TOF'}],
            'compound': [{'inchiKey': 'short'}],
            'spectrum': '43:1',
        },
    ]
    f = tmp_path / 'mona.json'
    f.write_text(json.dumps(recs))
    out = list(dl.parse_mona(f))
    assert len(out) == 1 and out[0]['smiles'] == 'CCCC'
    assert out[0]['peaks'] == [(43.0, 100.0), (60.1, 50.0)]


def test_bin_and_normalise():
    v = dl.bin_and_normalise([(43.2, 4.0), (42.8, 9.0), (60.0, 0.0), (999.4, 1.0), (-1, 5)])
    assert v.shape == (1000,) and np.isclose(np.linalg.norm(v), 1.0)
    assert v[43] == pytest.approx(3 / np.sqrt(10)) and v[999] == pytest.approx(1 / np.sqrt(10))
    assert v[60] == 0 and np.all(dl.bin_and_normalise([]) == 0)


def test_deduplicate_prefers_mona_then_more_peaks():
    recs = [
        {'inchikey': 'K', 'source': 'massbank', 'peaks': [(1, 1)] * 9},
        {'inchikey': 'K', 'source': 'mona', 'peaks': [(1, 1)] * 2},
        {'inchikey': 'K', 'source': 'mona', 'peaks': [(1, 1)] * 5},
        {'inchikey': 'J', 'source': 'massbank', 'peaks': [(1, 1)]},
    ]
    out = {r['inchikey']: r for r in dl.deduplicate(recs)}
    assert out['K']['source'] == 'mona' and len(out['K']['peaks']) == 5 and 'J' in out


def test_write_hdf5_round_trip(tmp_path):
    recs = [
        {
            'inchikey': f'KEY{i:011d}',
            'smiles': '',
            'source': 'mona' if i % 2 else 'massbank',
            'peaks': [(40.0 + i, 1.0)],
        }
        for i in range(20)
    ]
    counts = dl.write_library(recs, tmp_path / 's.h5', val_fraction=0.2)
    assert counts == {'total': 20, 'train': 16, 'val': 4}
    with h5py.File(tmp_path / 's.h5') as f:
        assert f['spectra'].shape == (20, 1000)
        split = np.array(f['split']).astype(str)
        assert set(split) == {'train', 'val'} and (split == 'val').sum() == 4


# ── 04 ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    'name,date,blank,expected',
    [
        ('4 - BCO4 Liver A', '150916', False, 'OK -> batch_sep2015'),
        ('4 - BCO4 Liver A', '150807', False, 'assigned to batch_aug2015 (not an evaluated batch)'),
        ('4 - BCO4 Liver A', '170101', False, 'acquisition date 170101 not in DATE_TO_BATCH'),
        ('4 - BCO4 Liver A', '160112', True, 'instrument blank'),
        ('not a sample name', '150916', False, 'unparseable filename'),
    ],
)
def test_audit_classify(tmp_path, name, date, blank, expected):
    f = tmp_path / f'{name}.txt'
    write_raw(f, date, [5.0, 5.01], [{100.0: 1.0}, {100.0: 2.0}], blank=blank)
    assert audit.classify(f) == expected


def test_audit_run_writes_both_reports(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    write_raw(raw / '4 - BCO4 Liver A.txt', '150916', *run(5.0, end=5.2))
    write_raw(raw / '4 - BCO4 Liver B.txt', '160711', *run(3.0, end=3.2))
    audit.run(raw, Paths(tmp_path), reference_ids=None)
    assert (
        'Samples considered     : 2'
        in (tmp_path / 'results' / 'fish_oil_sample_audit.txt').read_text()
    )
    assert 'DIFFER' in (tmp_path / 'results' / 'scan_time_audit.txt').read_text()


# ── 03 ────────────────────────────────────────────────────────────────────────


def test_cli_index_mode_reproduces_index_binning(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    raw.mkdir()
    times, peaks = run(3.0)
    write_raw(raw / '4 - BCO4 Liver A.txt', '160711', times, peaks)
    res = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / '03_preprocess_fish_oil_batches.py'),
            '--raw-dir',
            str(raw),
            '--output-dir',
            str(out),
            '--binning',
            'index',
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    z = np.load(next((out / 'batch_jul2016' / 'chroma').glob('*.npz')))
    spectra, tic, _, _ = pp.parse_txt(raw / '4 - BCO4 Liver A.txt')
    expect = pp.sqrt_l2(pp.bin_by_index(spectra, tic, 200))
    assert np.array_equal(z['chroma'], expect)


def test_sqrt_l2_rows():
    x = np.array([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
    out = pp.sqrt_l2(x)
    assert np.allclose(np.linalg.norm(out[[0, 2]], axis=1), 1.0) and np.all(out[1] == 0)


# ── data.py and baselines edge cases ──────────────────────────────────────────


def test_data_loading_errors(tmp_path):
    from warpalign.data import load_batch, load_tic_raw

    with pytest.raises(FileNotFoundError):
        load_batch('nothing', tmp_path)
    d = tmp_path / 'b' / 'chroma'
    d.mkdir(parents=True)
    np.savez(d / 's.npz', chroma=np.zeros((200, 5), np.float32))
    with pytest.raises(ValueError, match='tic_raw'):
        load_tic_raw(d)


def test_data_batch_properties(tmp_path):
    from warpalign.data import load_batch

    d = tmp_path / 'b' / 'chroma'
    d.mkdir(parents=True)
    for name in ('s2', 's1'):
        np.savez(
            d / f'{name}.npz',
            chroma=np.ones((200, 5), np.float32),
            tic_raw=np.ones(200, np.float32),
        )
    b = load_batch('b', tmp_path)
    assert len(b) == 2 and b.sample_ids == ('s1', 's2') and np.allclose(b.tic, 5.0)


def test_cow_falls_back_to_identity_when_no_path_exists():
    from warpalign.baselines import cow_tic_align

    x = np.random.default_rng(0).normal(size=20)
    al = cow_tic_align(x, x[::-1].copy(), n_segs=4, slack=0)  # nodes fixed; still a path
    assert np.allclose(al.src, np.arange(20))
    al = cow_tic_align(x[:6], x[:6], n_segs=4, slack=0)  # segments too short: no path
    assert np.allclose(al.src, np.arange(6)) and np.allclose(al.warped, x[:6])


def test_icoshift_tiny_intervals_are_left_alone():
    from warpalign.baselines import icoshift_align

    x = np.arange(12, dtype=float)
    al = icoshift_align(x, x[::-1].copy(), n_intervals=4)  # 3-bin intervals < 4
    assert al.details['lags'] == [0, 0, 0, 0] and np.allclose(al.src, np.arange(12))


def test_audit_empty_windows_tells_zero_reading_scans_from_missing_scans(tmp_path):
    """Window 23 (5.175–5.4 min) has scans with no readings; window 25 has no scans at all."""
    times = [t for t in np.arange(5.0, 6.0, 0.01) if not 5.625 <= t < 5.85]
    peaks = [{} if 5.175 <= t < 5.4 else {100.0: 1.0} for t in times]
    f = tmp_path / '4 - BCO4 Liver A.txt'
    write_raw(f, '150916', times, peaks)
    (e,) = map(audit.empty_windows, audit.run_times([f]))
    assert e.start_bin == 22
    assert np.flatnonzero(e.empty[:27]).tolist() == [23, 25]
    assert np.flatnonzero(e.no_scans[:27]).tolist() == [25]
    assert e.segments()[:2] == [(23, 1), (25, 1)]
    assert e.empty[27:].all()  # past the last scan: nothing acquired, every window empty
    text = '\n'.join(audit.empty_window_report([f]))
    assert 'batch_sep2015' in text and 'not compared' in text
