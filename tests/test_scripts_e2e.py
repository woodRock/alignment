"""End-to-end: every analysis step (05–14) run through its command-line entry point on a
synthetic dataset in a temporary --root, as section D runs them on the real data, with short
calibrations (--calib-steps 20) and an untrained checkpoint. Tests plumbing and file contracts,
not alignment quality. The numbered scripts themselves are checked as subprocesses (--help)."""

import subprocess
import sys

import numpy as np
import pytest
import synth
import torch
from conftest import SCRIPTS

from gcmsalign.steps import (
    ablate_pretrain,
    aggregate_seeds,
    calibrate_pair,
    ceiling,
    figures,
    heldout_anchors,
    literature_baselines,
    mechanism_ablation,
    pchip_baseline,
    significance,
)
from gcmsalign.warp_transformer import ChromaWarpTransformer

PAIRS = [
    ('sep2015', 'jul2016'),
    ('sep2015', 'apr2016'),
    ('jan2016', 'apr2016'),
    ('jan2016', 'jul2016'),
]
EXCL = [('sep2015', 'jan2016'), ('apr2016', 'jul2016')]
pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def run_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('root')
    synth.write_dataset(root, n_per_batch=3)
    torch.manual_seed(0)
    (root / 'checkpoints').mkdir()
    ck = str(root / 'checkpoints' / 'ck.pt')
    torch.save(ChromaWarpTransformer().state_dict(), ck)
    common = ['--root', str(root)]
    model = [*common, '--device', 'cpu', '--ckpt', ck, '--calib-steps', '20']

    def pair(a, b):
        return ['--study-a', f'fish_oil/batch_{a}', '--study-b', f'fish_oil/batch_{b}']

    for a, b in PAIRS + EXCL:
        literature_baselines.cli([*common, *pair(a, b)])
        calibrate_pair.cli([*model, *pair(a, b), '--seed', '0'])
    for a, b in PAIRS:
        calibrate_pair.cli([*model, *pair(a, b), '--seed', '1'])
        for s in ('0', '1'):
            ablate_pretrain.cli([*model, *pair(a, b), '--seed', s])
    pchip_baseline.cli([*common, '--seed', '0'])
    mechanism_ablation.cli([*model, '--seed', '0'])
    heldout_anchors.cli([*model, '--seed', '0'])
    ceiling.cli(common)
    significance.cli([*common, '--seed', '0', '--seeds', '0', '1'])
    aggregate_seeds.cli([*common, '--seeds', '0', '1'])
    figures.cli([*model, '--figs', '1', '2', '3', '4', '5', '--select', 'median', '--seed', '0'])
    return root


@pytest.mark.parametrize('script', sorted(p.name for p in SCRIPTS.glob('[01]*.py')))
def test_numbered_scripts_run(script):
    res = subprocess.run(
        [sys.executable, str(SCRIPTS / script), '--help'],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr
    assert '--root' in res.stdout


def res(root, name):
    return (root / 'results' / name).read_text()


def npz(root, name):
    return np.load(root / 'results' / 'per_pair' / name)


def test_every_expected_output_exists(run_root):
    names = [f'literature_baselines_batch_{a}_batch_{b}.txt' for a, b in PAIRS + EXCL]
    names += [f'warp_transformer_calib_batch_{a}_batch_{b}_seed0.txt' for a, b in PAIRS + EXCL]
    names += [
        'pchip_pooled_anchors.txt',
        'mechanism_ablation.txt',
        'heldout_anchor_error.txt',
        'within_batch_ceiling.txt',
        'significance_tests.txt',
        'seed_summary.txt',
        'fig1_warp_functions_batch_sep2015_batch_jul2016_median.pdf',
        'fig2_tic_overlays_batch_sep2015_batch_jul2016_median.pdf',
        'fig3_calibration_curve_batch_sep2015_batch_jul2016.pdf',
        'fig4_per_pair_distributions.pdf',
        'fig5_silhouette_scatter.pdf',
        'fig_selection_batch_sep2015_batch_jul2016_median.txt',
    ]
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
        base, cal = (
            npz(run_root, f'tic_r_baselines_{tag}.npz'),
            npz(run_root, f'tic_r_calib_{tag}_seed0.npz'),
        )
        pool, cal1 = (
            npz(run_root, f'tic_r_pchip_pooled_{tag}.npz'),
            npz(run_root, f'tic_r_calib_{tag}_seed1.npz'),
        )
        n = int(base['n_a']) * int(base['n_b'])
        for arr in (
            base['unaligned'],
            base['cow_tic'],
            base['icoshift_raw'],
            cal['warptransformer'],
            cal['warptransformer_raw'],
            pool['pchip_pooled'],
        ):
            assert arr.shape == (n,) and np.isfinite(arr).all()
        assert np.allclose(base['unaligned'], cal['unaligned'])
        assert np.allclose(base['unaligned_raw'], cal['unaligned_raw'])
        assert np.allclose(pool['pchip_pooled'], cal['pchip_pooled'])
        assert np.array_equal(cal['anchors'], cal1['anchors'])
        assert cal['wt_warp_mag_pre'].shape == (3,) and cal['warps'].shape[1] == 200
        assert cal['outside'].shape == (3,) and np.all(
            (cal['outside'] >= 0) & (cal['outside'] <= 1)
        )
        assert cal['warptransformer_valid'].shape == (n,) and base['cow_tic_valid'].shape == (n,)
        assert base['outside'][0] == 0.0  # unaligned reads nothing outside


def test_pchip_through_the_true_drift_aligns(run_root):
    for a, b in PAIRS:
        cal = npz(run_root, f'tic_r_calib_batch_{a}_batch_{b}_seed0.npz')
        assert cal['pchip_pooled'].mean() > cal['unaligned'].mean() + 0.3


def test_seed_summary_checks_pass_and_reports_both_arms(run_root):
    text = res(run_root, 'seed_summary.txt')
    assert 'Checks: all passed' in text
    assert 'warp before (all)' in text and 'warp after (all)' in text
    ablation = text[text.index('Pretraining ablation') :]
    assert '\n    pretrained ' in ablation and '\n    random init ' in ablation
    assert 'before calibration' in text and 'after calibration' in text


def test_ablation_records_starting_warps_and_device(run_root):
    z = npz(run_root, 'tic_r_ablate_batch_sep2015_batch_jul2016_seed0.npz')
    assert z['warp_mag_pre'].shape == (2, 3) and str(z['device']) == 'cpu'
    assert np.allclose(z['warp_mag_pre'][1], 0.0, atol=1e-6)  # random init starts at identity


def test_heldout_report_lists_every_method_and_the_filter_note(run_root):
    text = res(run_root, 'heldout_anchor_error.txt')
    for m in (
        'No alignment',
        'Constant shift',
        'PCHIP (pooled anchors)',
        'COW-TIC',
        'Free warp (no network)',
        'WarpTransformer',
        'WT in-sample',
        'held-out anchor included',
    ):
        assert m in text
    z = np.load(run_root / 'results' / 'per_pair' / 'heldout_anchor_error.npz')
    err = z['batch_sep2015_batch_jul2016_PCHIP_(pooled_anchors)']
    assert np.isfinite(err).all() and err.max() < 1.0  # constant drift is easy for PCHIP


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
