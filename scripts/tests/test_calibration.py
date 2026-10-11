"""calibration.py and evaluation.py."""
import numpy as np
import pytest
import torch

import evaluation
from calibration import (batch_tensors, calibrate, final_losses, free_warp, load_model,
                         pick_device, predict, set_seed)
from data import Batch
from warp_transformer import ChromaWarpTransformer, anchor_loss_exact, warp_smoothness_loss

T = 200
ANCHORS = [(40, 32), (75, 67), (115, 107), (155, 147)]     # constant drift −8 bins


def tiny_model(dropout=0.0, seed=0):
    torch.manual_seed(seed)
    return ChromaWarpTransformer(mz_bins=1000, n_bins=T, d_model=32, n_heads=4, n_layers=1,
                                 dropout=dropout)


@pytest.fixture(scope='module')
def tensors(synthetic_pair):
    (ca, _), (cb, _) = synthetic_pair
    return batch_tensors(ca, cb, torch.device('cpu'))


def test_set_seed_makes_torch_and_numpy_repeatable():
    set_seed(3)
    a = (torch.rand(3), np.random.rand(3))
    set_seed(3)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_pick_device():
    assert pick_device('cpu').type == 'cpu'
    assert pick_device('auto').type in {'cpu', 'cuda', 'mps'}


def test_batch_tensors_layout(synthetic_pair):
    (ca, _), (cb, _) = synthetic_pair
    x, sid = batch_tensors(ca, cb, torch.device('cpu'))
    assert x.shape == (len(ca) + len(cb), T, 1000) and x.dtype == torch.float32
    assert sid.tolist() == [0] * len(ca) + [1] * len(cb)


def test_load_model_random_init_is_seeded_and_identity():
    m1, m2 = load_model(None, torch.device('cpu'), seed=1), load_model(None, torch.device('cpu'), seed=1)
    assert all(torch.equal(p, q) for p, q in zip(m1.state_dict().values(), m2.state_dict().values()))


def test_load_model_rejects_v1_checkpoints(tmp_path):
    """v1 checkpoints have no start head (both warp ends were pinned); loading one must fail."""
    state = {k: v for k, v in ChromaWarpTransformer().state_dict().items()
             if not k.startswith('start_head')}
    torch.save(state, tmp_path / 'v1.pt')
    with pytest.raises(ValueError, match='not compatible'):
        load_model(tmp_path / 'v1.pt', torch.device('cpu'))


def test_load_model_round_trip(tmp_path):
    m = ChromaWarpTransformer()
    torch.save(m.state_dict(), tmp_path / 'm.pt')
    loaded = load_model(tmp_path / 'm.pt', torch.device('cpu'))
    assert all(torch.equal(p, q) for p, q in zip(m.state_dict().values(), loaded.state_dict().values()))


def test_calibrate_without_anchors_leaves_model_unchanged(tensors):
    m = tiny_model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    log = calibrate(m, *tensors, [], seed=0, steps=5)
    assert log.history == [] and log.seconds == 0.0
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


def test_calibration_fits_the_anchors(tensors):
    """Section C, test 5: a short calibration on the exact loss moves batch B by the anchors'
    drift (−8 bins)."""
    x, sid = tensors
    m = tiny_model()
    log = calibrate(m, x, sid, ANCHORS, seed=0, steps=150, lr=5e-3, log_every=50)
    _, warps = predict(m, x, sid)
    a_loss, _ = final_losses(warps, sid, ANCHORS)
    assert a_loss < 1.0
    assert [h['step'] for h in log.history] == [50, 100, 150]
    assert log.history[-1]['anchor'] < log.history[0]['anchor'] or log.history[0]['anchor'] < 1.0


def test_best_state_is_the_one_the_best_loss_was_measured_on(tensors):
    """v1 saved the parameters after the step that followed the best loss. With dropout off,
    the restored model's loss must equal the recorded best loss."""
    x, sid = tensors
    m = tiny_model(dropout=0.0)
    log = calibrate(m, x, sid, ANCHORS, seed=0, steps=40, lr=2e-2)
    _, warps = predict(m, x, sid)
    w = torch.as_tensor(warps)
    loss = float(anchor_loss_exact(w, sid, ANCHORS) + 0.05 * warp_smoothness_loss(w))
    assert loss == pytest.approx(log.best_loss, rel=1e-4, abs=1e-6)


def test_calibration_is_reproducible_on_cpu(tensors):
    x, sid = tensors
    out = []
    for _ in range(2):
        m = tiny_model(dropout=0.1)
        calibrate(m, x, sid, ANCHORS, seed=7, steps=10)
        out.append(predict(m, x, sid)[1])
    assert np.array_equal(*out)


def test_on_log_callback_receives_records(tensors):
    x, sid = tensors
    seen = []
    calibrate(tiny_model(), x, sid, ANCHORS, seed=0, steps=4, log_every=2,
              on_log=lambda step, rec, aligned: seen.append((step, aligned.shape)))
    assert [s for s, _ in seen] == [2, 4] and seen[0][1] == x.shape


def test_predict_restores_training_mode(tensors):
    m = tiny_model().train()
    predict(m, *tensors)
    assert m.training


def test_free_warp_fits_constant_drift():
    warps, (a_loss, s_loss) = free_warp(3, 2, ANCHORS, lr=0.05, seed=0, steps=300)
    assert warps.shape == (5, T)
    assert np.allclose(warps[0], warps[1]) and np.allclose(warps[3], warps[4])   # per batch
    assert a_loss < 0.5
    t = 100
    assert (warps[3, t] - warps[0, t]) == pytest.approx(-8, abs=1.0)


def test_free_warp_without_anchors_stays_identity():
    warps, (a_loss, s_loss) = free_warp(2, 2, [], lr=0.05, seed=0, steps=20)
    assert np.allclose(warps, np.arange(T)) and a_loss == 0.0 and s_loss == 0.0


# ── evaluation.py ─────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def batches(synthetic_pair):
    (ca, ra), (cb, rb) = synthetic_pair
    return (Batch('a', ca, ra, tuple(f'a{i}' for i in range(len(ca)))),
            Batch('b', cb, rb, tuple(f'b{i}' for i in range(len(cb)))))


def test_identity_warps_reproduce_unaligned(batches):
    a, b = batches
    un = evaluation.unaligned(a, b)
    ident = np.tile(np.arange(T, dtype=float), (len(a) + len(b), 1))
    wp = evaluation.warped_pair(a, b, ident)
    assert np.allclose(wp.r, un.r) and np.allclose(wp.r_raw, un.r_raw)
    assert wp.warp_mag == 0.0 and wp.sil == pytest.approx(un.sil)


def test_pchip_through_true_anchors_aligns(batches):
    a, b = batches
    un, pc = evaluation.unaligned(a, b), evaluation.pchip(a, b, ANCHORS)
    assert pc.mean > un.mean + 0.3 and pc.mean_raw > un.mean_raw + 0.3
    assert pc.warp_mag == pytest.approx(8 * 0.225)


def test_true_shift_warps_align_both_batches(batches):
    a, b = batches
    w = np.concatenate([np.tile(np.arange(T, dtype=float), (len(a), 1)),
                        np.tile(np.arange(T) - 8.0, (len(b), 1))])
    assert evaluation.warped_pair(a, b, w).mean > evaluation.unaligned(a, b).mean + 0.3


def test_warp_mags_order():
    w = np.concatenate([np.tile(np.arange(T) + 1.0, (2, 1)), np.tile(np.arange(T) + 3.0, (2, 1))])
    assert evaluation.warp_mags(w, 2) == pytest.approx((2 * 0.225, 0.225, 3 * 0.225))


def test_heldout_script_skips_pairs_with_too_few_anchors(tmp_path, monkeypatch, batches):
    """10 needs ≥ 2 anchors to hold one out; a pair with one anchor is reported and skipped."""
    from anchors import AnchorResult
    from conftest import load_script
    h = load_script('10_heldout_anchor_error.py')
    a, b = batches
    monkeypatch.setattr(h, 'EVAL_PAIRS', (('a', 'b'),))
    monkeypatch.setattr(h, 'load_batch', lambda name: a if name == 'a' else b)
    monkeypatch.setattr(h, 'extract_anchors', lambda *args: AnchorResult([(60, 52)]))
    monkeypatch.setattr(h, 'RESULTS_DIR', tmp_path)
    monkeypatch.setattr(h, 'PER_PAIR_DIR', tmp_path)
    lines = h.main(None, 0, 'cpu')
    assert any('Skipped' in line for line in lines)
    assert (tmp_path / 'heldout_anchor_error.txt').exists()
