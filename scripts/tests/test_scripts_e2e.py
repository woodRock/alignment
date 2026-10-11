"""End-to-end: every analysis script (05–14) run as a subprocess on a synthetic dataset in a
temporary ALIGNMENT_ROOT, as section D runs them on the real data. Short calibrations
(ALIGNMENT_CALIB_STEPS=20), an untrained checkpoint: this tests plumbing and file contracts,
not alignment quality."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import synth

SCRIPTS = Path(__file__).resolve().parent.parent
PAIRS = [('sep2015', 'jul2016'), ('sep2015', 'apr2016'), ('jan2016', 'apr2016'), ('jan2016', 'jul2016')]
EXCL  = [('sep2015', 'jan2016'), ('apr2016', 'jul2016')]
pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def run_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('root')
    synth.write_dataset(root, n_per_batch=3)
    sys.path.insert(0, str(SCRIPTS.parent / 'src'))
    from warp_transformer import ChromaWarpTransformer
    torch.manual_seed(0)
    (root / 'checkpoints').mkdir()
    torch.save(ChromaWarpTransformer().state_dict(), root / 'checkpoints' / 'ck.pt')
    (root / 'results' / 'per_pair').mkdir(parents=True)
    env = dict(os.environ, ALIGNMENT_ROOT=str(root), ALIGNMENT_CALIB_STEPS='20', MPLBACKEND='Agg')
    ck = str(root / 'checkpoints' / 'ck.pt')

    cov = ['-m', 'coverage', 'run', '--parallel-mode',
           f'--rcfile={SCRIPTS.parent / ".coveragerc"}'] if os.environ.get('E2E_COVERAGE') else []

    def run(script, *args):
        res = subprocess.run([sys.executable, *cov, str(SCRIPTS / script), *args], env=env,
                             capture_output=True, text=True, timeout=900, cwd=SCRIPTS.parent)
        assert res.returncode == 0, f'{script} {args}\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}'
        return res.stdout

    def pair(a, b):
        return ['--study-a', f'fish_oil/batch_{a}', '--study-b', f'fish_oil/batch_{b}']
    for a, b in PAIRS + EXCL:
        run('05_literature_baselines.py', *pair(a, b))
        run('06_align_testtime_calib.py', *pair(a, b), '--seed', '0', '--device', 'cpu', '--ckpt', ck)
    for a, b in PAIRS:
        run('06_align_testtime_calib.py', *pair(a, b), '--seed', '1', '--device', 'cpu', '--ckpt', ck)
        for s in ('0', '1'):
            run('08_ablate_pretrain.py', *pair(a, b), '--seed', s, '--device', 'cpu', '--ckpt', ck)
    run('07_pchip_pooled_anchors.py', '--seed', '0')
    run('09_mechanism_ablation.py', '--seed', '0', '--device', 'cpu', '--ckpt', ck)
    run('10_heldout_anchor_error.py', '--seed', '0', '--device', 'cpu', '--ckpt', ck)
    run('11_within_batch_ceiling.py')
    run('12_significance_tests.py', '--seed', '0', '--seeds', '0', '1')
    run('13_aggregate_seeds.py', '--seeds', '0', '1')
    run('14_generate_figures.py', '--figs', '1', '2', '3', '4', '5', '--select', 'median',
        '--seed', '0', '--device', 'cpu', '--ckpt', ck)
    return root


def res(root, name):
    return (root / 'results' / name).read_text()


def npz(root, name):
    return np.load(root / 'results' / 'per_pair' / name)


def test_every_expected_output_exists(run_root):
    names = [f'literature_baselines_batch_{a}_batch_{b}.txt' for a, b in PAIRS + EXCL]
    names += [f'warp_transformer_calib_batch_{a}_batch_{b}_seed0.txt' for a, b in PAIRS + EXCL]
    names += ['pchip_pooled_anchors.txt', 'mechanism_ablation.txt', 'heldout_anchor_error.txt',
              'within_batch_ceiling.txt', 'significance_tests.txt', 'seed_summary.txt',
              'fig1_warp_functions_batch_sep2015_batch_jul2016_median.pdf',
              'fig2_tic_overlays_batch_sep2015_batch_jul2016_median.pdf',
              'fig3_calibration_curve_batch_sep2015_batch_jul2016.pdf',
              'fig4_per_pair_distributions.pdf', 'fig5_silhouette_scatter.pdf',
              'fig_selection_batch_sep2015_batch_jul2016_median.txt']
    missing = [n for n in names if not (run_root / 'results' / n).exists()]
    assert not missing


def test_anchors_recover_the_synthetic_drift(run_root):
    for a, b in PAIRS:
        anchors = npz(run_root, f'tic_r_calib_batch_{a}_batch_{b}_seed0.npz')['anchors']
        assert len(anchors) >= 4
        assert np.all(np.abs((anchors[:, 1] - anchors[:, 0]) - synth.SHIFT) <= 1)
    for a, b in EXCL:
        anchors = npz(run_root, f'tic_r_calib_batch_{a}_batch_{b}_seed0.npz')['anchors']
        assert np.all(np.abs(anchors[:, 1] - anchors[:, 0]) <= 1)


def test_arrays_are_paired_and_consistent_across_scripts(run_root):
    for a, b in PAIRS:
        tag = f'batch_{a}_batch_{b}'
        base, cal = npz(run_root, f'tic_r_baselines_{tag}.npz'), npz(run_root, f'tic_r_calib_{tag}_seed0.npz')
        pool, cal1 = npz(run_root, f'tic_r_pchip_pooled_{tag}.npz'), npz(run_root, f'tic_r_calib_{tag}_seed1.npz')
        n = int(base['n_a']) * int(base['n_b'])
        for arr in (base['unaligned'], base['cow_tic'], base['icoshift_raw'], cal['warptransformer'],
                    cal['warptransformer_raw'], pool['pchip_pooled']):
            assert arr.shape == (n,) and np.isfinite(arr).all()
        assert np.allclose(base['unaligned'], cal['unaligned'])
        assert np.allclose(base['unaligned_raw'], cal['unaligned_raw'])
        assert np.allclose(pool['pchip_pooled'], cal['pchip_pooled'])
        assert np.array_equal(cal['anchors'], cal1['anchors'])
        assert cal['wt_warp_mag_pre'].shape == (3,) and cal['warps'].shape[1] == 200


def test_pchip_through_the_true_drift_aligns(run_root):
    for a, b in PAIRS:
        cal = npz(run_root, f'tic_r_calib_batch_{a}_batch_{b}_seed0.npz')
        assert cal['pchip_pooled'].mean() > cal['unaligned'].mean() + 0.3


def test_seed_summary_checks_pass_and_reports_both_arms(run_root):
    text = res(run_root, 'seed_summary.txt')
    assert 'Checks: all passed' in text
    assert 'starting warp, pretrained' in text and 'starting warp, random init' in text
    assert 'before calibration (pretrained model)' in text


def test_ablation_records_starting_warps_and_device(run_root):
    z = npz(run_root, 'tic_r_ablate_batch_sep2015_batch_jul2016_seed0.npz')
    assert z['warp_mag_pre'].shape == (2, 3) and str(z['device']) == 'cpu'
    assert np.allclose(z['warp_mag_pre'][1], 0.0, atol=1e-6)       # random init starts at identity


def test_heldout_report_lists_every_method_and_the_filter_note(run_root):
    text = res(run_root, 'heldout_anchor_error.txt')
    for m in ('No alignment', 'Constant shift', 'PCHIP (pooled anchors)', 'COW-TIC',
              'Free warp (no network)', 'WarpTransformer', 'WT in-sample', 'held-out anchor included'):
        assert m in text
    z = np.load(run_root / 'results' / 'per_pair' / 'heldout_anchor_error.npz')
    err = z['batch_sep2015_batch_jul2016_PCHIP_(pooled_anchors)']
    assert np.isfinite(err).all() and err.max() < 1.0              # constant drift is easy for PCHIP


def test_mechanism_ablation_text(run_root):
    text = res(run_root, 'mechanism_ablation.txt')
    assert '10-comparison family' in text and '15-comparison' not in text
    assert 'Control: unaligned, both ½-bin shifted' in text


def test_ceiling_and_significance(run_root):
    assert 'Sep15×Jul16' in res(run_root, 'within_batch_ceiling.txt')
    sig = res(run_root, 'significance_tests.txt')
    assert 'Seeds available: [0, 1]' in sig and 'Holm' in sig


def test_figure_selection_file(run_root):
    first = res(run_root, 'fig_selection_batch_sep2015_batch_jul2016_median.txt').splitlines()[0]
    assert 'median' in first and 'index' in first
