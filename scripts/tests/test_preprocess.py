"""03_preprocess_fish_oil_batches.py and 04_audit_fish_oil_samples.py on synthetic raw exports."""
from pathlib import Path

import numpy as np
import pytest

from conftest import load_script

pp = load_script('03_preprocess_fish_oil_batches.py')
audit = load_script('04_audit_fish_oil_samples.py')


def write_raw(path: Path, date: str, times, peaks_per_scan, blank=False):
    """Minimal export in the instrument's text format. peaks_per_scan: list of {mz: intensity}."""
    lines = ['', 'RunHeaderInfo',
             f'first_scan = 1, last_scan = {len(times)}, start_time = {times[0]:.6f}, end_time = {times[-1]:.6f}',
             f'comment1 = {date}{" BLANK012" if blank else ""}', '']
    for i, (t, peaks) in enumerate(zip(times, peaks_per_scan), start=1):
        lines += [f'ScanHeader # {i}', f'position = {i}, start_mass= 50.000000, end_mass = 550.000000',
                  f'start_time = {t:.6f}, end_time = 0.000000, packet_type = 2',
                  f'num_readings = {len(peaks)}, integ_intens = {sum(peaks.values()):.6f}, data packet pos = 0',
                  'DataPeaks', '']
        for k, (mz, inten) in enumerate(peaks.items()):
            lines += [f'Packet # {k}, intensity = {inten:.6f}, mass/position = {mz:.6f}',
                      'saturated = 0, fragmented = 0, merged = 0', '']
    path.write_text('\n'.join(lines) + '\n')


def run(start: float, end: float = 44.991667, dt: float = 0.5 / 60, peak_at: float = 20.0):
    """Scan times and per-scan peaks: one compound (m/z 100 and 150) eluting at peak_at min."""
    times = np.arange(start, end + 1e-9, dt)
    peaks = [{100.0: 10.0 + 5000 * np.exp(-0.5 * ((t - peak_at) / 0.02) ** 2),
              150.0: 5.0 + 2000 * np.exp(-0.5 * ((t - peak_at) / 0.02) ** 2)} for t in times]
    return times, peaks


def test_parse_txt_reads_spectra_tic_and_times(tmp_path):
    f = tmp_path / '4 - BCO4 Liver A.txt'
    write_raw(f, '150916', [5.0, 5.5, 6.0], [{57.1: 2.0, 100.4: 3.0}, {100.0: 7.0}, {}])
    spectra, tic, date, times = pp.parse_txt(f)
    assert date == '150916'
    assert np.allclose(times, [5.0, 5.5, 6.0])
    assert np.allclose(tic, [5.0, 7.0, 0.0])
    assert spectra.shape == (3, 1000)
    assert spectra[0, 57] == 2.0 and spectra[0, 100] == 3.0 and spectra[1, 100] == 7.0


def test_parse_txt_skips_blanks(tmp_path):
    f = tmp_path / '4 - BCO4 Liver A.txt'
    write_raw(f, '160112', [5.0], [{100.0: 1.0}], blank=True)
    assert pp.parse_txt(f) is None


def test_bin_by_time_puts_a_compound_in_the_same_bin_whatever_the_start_time():
    """The step-0 finding: runs starting at 3.0 and 5.0 min must share one time axis."""
    out = {}
    for start in (3.0, 5.0):
        times, peaks = run(start)
        spectra = np.zeros((len(times), 1000), np.float32)
        for i, p in enumerate(peaks):
            for mz, v in p.items():
                spectra[i, int(mz)] = v
        tic = spectra.sum(1)
        out[start] = (pp.bin_chromatogram(spectra, tic, times, 200),
                      pp.bin_chromatogram_by_index(spectra, tic, 200))
    t_bins = [np.argmax(out[s][0].sum(1)) for s in (3.0, 5.0)]
    assert t_bins[0] == t_bins[1] == int(20.0 // 0.225)
    i_bins = [np.argmax(out[s][1].sum(1)) for s in (3.0, 5.0)]
    assert i_bins[0] != i_bins[1]                       # index binning disagrees (the v1 bug)


def test_bin_by_time_leaves_windows_outside_the_run_empty_and_keeps_the_max_tic_scan():
    times = np.array([5.01, 5.1, 5.2, 44.9])
    spectra = np.eye(4, 1000, dtype=np.float32) * np.array([[1], [3], [2], [1]], dtype=np.float32)
    out = pp.bin_chromatogram(spectra, spectra.sum(1), times, 200)
    assert np.all(out[:22] == 0)
    assert out[22, 1] == 3 and out[22].sum() == 3        # 5.01, 5.1 → window 22; max TIC wins
    assert out[23, 2] == 2 and out[199, 3] == 1
    assert np.all(out[24:199] == 0)


def test_main_writes_time_binned_chroma_and_raw_tic(tmp_path):
    raw, out = tmp_path / 'raw', tmp_path / 'out'
    raw.mkdir()
    for name, date, start in [('4 - BCO4 Liver A', '150916', 5.0), ('4 - BCO4 Liver B', '160711', 3.0),
                              ('4 - BCO4 Liver C', '160711', 3.0)]:
        write_raw(raw / f'{name}.txt', date, *run(start))
    pp.main(raw, out, 200, 1000, binning='time')
    sep = np.load(next((out / 'batch_sep2015' / 'chroma').glob('*.npz')))
    jul = np.load(sorted((out / 'batch_jul2016' / 'chroma').glob('*.npz'))[0])
    assert sep['chroma'].shape == (200, 1000) and sep['tic_raw'].shape == (200,)
    assert np.argmax(sep['tic_raw']) == np.argmax(jul['tic_raw']) == int(20.0 // 0.225)
    assert np.all(sep['tic_raw'][:22] == 0) and np.all(jul['tic_raw'][:13] == 0)
    assert jul['tic_raw'][14] > 0
    norms = np.linalg.norm(sep['chroma'][22:], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)            # sqrt + L2 per non-empty bin
    assert (out / 'batch_jul2016' / 'sample_ids.txt').read_text().count('\n') == 1


def test_scan_time_audit_detects_different_start_times(tmp_path):
    for name, date, start in [('4 - BCO4 Liver A', '150916', 5.0), ('4 - BCO4 Liver B', '160711', 3.0)]:
        times, peaks = run(start, end=6.0)
        write_raw(tmp_path / f'{name}.txt', date, times, peaks)
    lines = audit.scan_time_report(sorted(tmp_path.glob('*.txt')))
    text = '\n'.join(lines)
    assert 'batch_sep2015' in text and 'batch_jul2016' in text and 'DIFFER' in text


def test_scan_time_audit_agreeing_runs(tmp_path):
    for name in ('4 - BCO4 Liver A', '4 - BCO4 Liver B'):
        times, peaks = run(5.0, end=6.0)
        write_raw(tmp_path / f'{name}.txt', '150916', times, peaks)
    assert 'equivalent to binning by time' in '\n'.join(audit.scan_time_report(sorted(tmp_path.glob('*.txt'))))


@pytest.mark.parametrize('stem,expected', [('4 - BCO4 Liver Female A', ('BCO4', 'Liver')),
                                           ('12 - TAR3 Rest of Guts B', ('TAR3', 'Rest of Guts')),
                                           ('5 - SNA1 ROG A', ('SNA1', 'Rest of Guts')),
                                           ('BCO4 Liver', None), ('4 - BCO4 Fin A', None)])
def test_parse_filename(stem, expected):
    assert pp._parse_filename(stem) == expected
