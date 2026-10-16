"""calibration.py and evaluation.py."""

import numpy as np
import pytest
import torch

from warpalign import evaluation
from warpalign.calibration import (
    batch_tensors,
    calibrate,
    calibration_objective,
    final_losses,
    free_warp,
    load_model,
    pick_device,
    predict,
    set_seed,
)
from warpalign.warp_transformer import (
    ChromaWarpTransformer,
    anchor_loss_exact,
    warp_smoothness_loss,
)

T = 200
ANCHORS = [(40, 32), (75, 67), (115, 107), (155, 147)]  # constant drift −8 bins


def tiny_model(dropout=0.0, seed=0):
    torch.manual_seed(seed)
    return ChromaWarpTransformer(
        mz_bins=1000, n_bins=T, d_model=32, n_heads=4, n_layers=1, dropout=dropout
    )


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
    m1, m2 = (
        load_model(None, torch.device('cpu'), seed=1),
        load_model(None, torch.device('cpu'), seed=1),
    )
    assert all(
        torch.equal(p, q)
        for p, q in zip(m1.state_dict().values(), m2.state_dict().values(), strict=False)
    )


def test_load_model_rejects_v1_checkpoints(tmp_path):
    """v1 checkpoints have no ends head (both warp ends were pinned); loading one must fail."""
    state = {
        k: v
        for k, v in ChromaWarpTransformer().state_dict().items()
        if not k.startswith('ends_head')
    }
    torch.save(state, tmp_path / 'v1.pt')
    with pytest.raises(ValueError, match='not compatible'):
        load_model(tmp_path / 'v1.pt', torch.device('cpu'))


def test_load_model_round_trip(tmp_path):
    m = ChromaWarpTransformer()
    torch.save(m.state_dict(), tmp_path / 'm.pt')
    loaded = load_model(tmp_path / 'm.pt', torch.device('cpu'))
    assert all(
        torch.equal(p, q)
        for p, q in zip(m.state_dict().values(), loaded.state_dict().values(), strict=False)
    )


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
    a_loss, _, f_loss = final_losses(warps, sid, ANCHORS)
    assert f_loss < 1.0  # frame stays put
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
    loss = float(calibration_objective(w, sid, ANCHORS)[0])
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
    calibrate(
        tiny_model(),
        x,
        sid,
        ANCHORS,
        seed=0,
        steps=4,
        log_every=2,
        on_log=lambda step, rec, aligned: seen.append((step, aligned.shape)),
    )
    assert [s for s, _ in seen] == [2, 4] and seen[0][1] == x.shape


def test_predict_restores_training_mode(tensors):
    m = tiny_model().train()
    predict(m, *tensors)
    assert m.training


def test_free_warp_fits_constant_drift():
    warps, (a_loss, _, _f_loss) = free_warp(3, 2, ANCHORS, lr=0.05, seed=0, steps=300)
    assert warps.shape == (5, T)
    assert np.allclose(warps[0], warps[1]) and np.allclose(warps[3], warps[4])  # per batch
    assert a_loss < 0.5
    t = 100
    assert (warps[3, t] - warps[0, t]) == pytest.approx(-8, abs=1.0)


def test_free_warp_without_anchors_stays_identity():
    warps, (a_loss, s_loss, f_loss) = free_warp(2, 2, [], lr=0.05, seed=0, steps=20)
    assert np.allclose(warps, np.arange(T), atol=1e-4)  # identity to float32 precision
    assert a_loss == 0.0 and s_loss < 1e-8 and f_loss < 1e-8


# ── evaluation.py ─────────────────────────────────────────────────────────────


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
    w = np.concatenate(
        [np.tile(np.arange(T, dtype=float), (len(a), 1)), np.tile(np.arange(T) - 8.0, (len(b), 1))]
    )
    assert evaluation.warped_pair(a, b, w).mean > evaluation.unaligned(a, b).mean + 0.3


def test_warp_mags_order():
    w = np.concatenate([np.tile(np.arange(T) + 1.0, (2, 1)), np.tile(np.arange(T) + 3.0, (2, 1))])
    assert evaluation.warp_mags(w, 2) == pytest.approx((2 * 0.225, 0.225, 3 * 0.225))


def test_heldout_skips_pairs_with_too_few_anchors(monkeypatch, batches):
    """Step 10 needs ≥ 2 anchors to hold one out; a pair with one anchor is reported, skipped."""
    from alignbench.anchors import AnchorResult

    from warpalign.steps import heldout_anchors as h

    a, b = batches
    monkeypatch.setattr(h, 'extract_anchors', lambda *args: AnchorResult([(60, 52)]))
    res = h.compute_pair(a, b, None, torch.device('cpu'), 0, 5)
    assert res.skipped and res.errors == {}
    lines = h.format_report([res], 0, 'cpu', None)
    assert any('Skipped' in line for line in lines) and not any('Summary' in l for l in lines)


# ── The consensus frame (round-3 review, issue 1) ─────────────────────────────


def test_a_common_translation_is_invisible_to_anchor_and_smoothness_losses():
    """Why the frame term is needed: for shift-like warps (slope 1, as the drift here nearly
    is), adding one constant c to every warp changes neither loss, so only the frame term can
    stop the consensus frame translating. (For curved warps the invariance is approximate.)"""
    from warpalign.warp_transformer import frame_loss

    t = torch.arange(T, dtype=torch.float64)
    w = torch.stack([t + 1.0, t - 1.0, t - 7.0, t - 9.0])
    sid = torch.tensor([0, 0, 1, 1])
    for c in (-7.0, 3.5):
        assert torch.isclose(
            anchor_loss_exact(w + c, sid, ANCHORS), anchor_loss_exact(w, sid, ANCHORS)
        )
        assert torch.isclose(warp_smoothness_loss(w + c), warp_smoothness_loss(w))
        assert float(frame_loss(w + c, sid)) == pytest.approx(c**2)


def test_frame_loss_ignores_shape_and_batch_b():
    """The frame term measures only the batch-A mean offset: zero for any shape with no net
    offset, and independent of batch B."""
    from warpalign.warp_transformer import frame_loss

    t = torch.arange(T, dtype=torch.float64)
    wiggle = 3 * torch.sin(t / 10)
    wiggle = wiggle - wiggle.mean()  # shape change, no net offset
    w = torch.stack([t + wiggle, t - wiggle, t + 40.0, t - 9.0])
    sid = torch.tensor([0, 0, 1, 1])
    assert float(frame_loss(w, sid)) == pytest.approx(0.0, abs=1e-20)
    assert float(frame_loss(torch.stack([t + 2.0, t + 4.0, t, t]), sid)) == pytest.approx(9.0)


def test_calibration_keeps_batch_a_near_the_identity(tensors):
    """With the frame term, a calibration that moves batch B by −8 bins leaves batch A's mean
    offset near zero (the free translation is not used)."""
    x, sid = tensors
    m = tiny_model()
    calibrate(m, x, sid, ANCHORS, seed=0, steps=150, lr=5e-3)
    _, warps = predict(m, x, sid)
    n_a = int((sid == 0).sum())
    offset_a = float((warps[:n_a].mean(0) - np.arange(T)).mean())
    assert abs(offset_a) < 1.0
    assert float((warps[n_a:].mean(0) - warps[:n_a].mean(0)).mean()) == pytest.approx(-8, abs=1.5)


# ── Coverage of the acquired range (round-3 review, issue 3) ──────────────────


def test_acquisition_start():
    from alignbench.data import acquisition_start

    raw = np.zeros((3, T))
    raw[0, 22:] = 1.0
    raw[1, 13:] = 1.0
    assert acquisition_start(raw).tolist() == [22, 13, T]


def test_in_range_flags_reads_before_start_and_past_the_end():
    from warpalign.evaluation import in_range

    src = np.arange(T, dtype=float) - 5.0  # reads 5 bins earlier
    v = in_range(np.stack([src, src + 10.0]), np.array([10, 0]))
    assert not v[0, 14] and v[0, 15]  # 15 − 5 = 10: first acquired bin
    assert v[1, T - 6] and not v[1, T - 5]  # +5 past the end


def test_identity_has_no_outside_and_masked_r_equals_r(batches):
    a, b = batches
    un = evaluation.unaligned(a, b)
    assert un.outside == 0.0 and np.allclose(un.r_valid, un.r)


def test_large_shift_is_reported_outside_and_masked(batches):
    """Warps reading 60 bins past the end replicate the last bin; coverage reports it."""
    a, b = batches
    n = len(a) + len(b)
    w = np.tile(np.arange(T, dtype=float) + 60.0, (n, 1))
    ev = evaluation.warped_pair(a, b, w)
    window = T - 22
    assert ev.outside == pytest.approx(60 / window, abs=1e-9)
    assert np.all(np.isfinite(ev.r_valid)) and not np.allclose(ev.r_valid, ev.r)


def test_pearson_rows_masked_matches_manual():
    from alignbench.metrics.tic import pearson_rows, pearson_rows_masked

    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, T)), rng.normal(size=(1, T))
    va = np.ones((2, T), bool)
    vb = np.ones((1, T), bool)
    va[0, 100:] = False
    r = pearson_rows_masked(a, b, va, vb)
    assert r[0] == pytest.approx(np.corrcoef(a[0, 22:100], b[0, 22:100])[0, 1])
    assert r[1] == pytest.approx(pearson_rows(a[1:], b)[0])
    vb[0, 23:] = False  # only 1 bin left: undefined
    assert np.all(np.isnan(pearson_rows_masked(a, b, va, vb)))


def test_fig1_aligns_a_onto_b_like_the_evaluation(batches, monkeypatch, tmp_path):
    """Round-3 review, item 5: Fig. 4's icoshift and COW-TIC curves must be the warps that
    were scored, i.e. A aligned onto B (query = A sample, reference = B[0])."""
    from warpalign.steps import figures

    calls = []

    def spy(fn):
        def wrapped(query, ref, *args, **kw):
            calls.append((fn.__name__, query, ref))
            return fn(query, ref, *args, **kw)

        return wrapped

    monkeypatch.setattr(figures, 'icoshift_align', spy(figures.icoshift_align))
    monkeypatch.setattr(figures, 'cow_tic_align', spy(figures.cow_tic_align))
    a, b = batches
    w = np.tile(np.arange(T, dtype=float), (len(a) + len(b), 1))
    figures.fig1_warp_functions(a, b, w, ANCHORS, 1, tmp_path / 'f.pdf')
    assert (tmp_path / 'f.pdf').exists() and len(calls) == 2
    for _, query, ref in calls:
        assert np.array_equal(query, a.tic[1]) and np.array_equal(ref, b.tic[0])


def test_frame_term_stops_the_free_warp_translating_batch_a():
    """With shift-like anchors the anchor loss pulls batch A's and batch B's offsets equally in
    opposite directions; without the frame term batch A drifts by about +3 bins (measured).
    With it, batch A stays on the identity and only batch B moves."""
    warps, _ = free_warp(3, 2, ANCHORS, lr=0.05, seed=0, steps=300)
    t = np.arange(T)
    assert abs((warps[0] - t).mean()) < 0.5
    assert (warps[3] - t).mean() < -4


def test_free_warp_starts_at_the_anchors_median_shift():
    """Constant-drift anchors: the free warp is a pure −8 shift everywhere, with every loss term
    at zero (started at the identity, Adam stopped at a bent solution with a higher loss)."""
    warps, losses = free_warp(3, 2, ANCHORS, lr=0.05, seed=0, steps=200)
    t = np.arange(T)
    assert np.allclose(warps[:3], t, atol=1e-3)  # batch A: identity
    assert np.allclose(warps[3:], t - 8.0, atol=1e-3)  # batch B: −8 everywhere
    assert max(losses) < 1e-6


def test_free_warp_still_fits_non_constant_drift():
    """Drift trending from −4 to −12: the warp moves away from the initial shift as needed."""
    from warpalign.warp_transformer import predict_query_bins

    trending = [(40, 36), (75, 69), (115, 106), (155, 143)]
    warps, (a_loss, _, _) = free_warp(3, 2, trending, lr=0.05, seed=0, steps=500)
    pred = predict_query_bins(
        torch.as_tensor(warps, dtype=torch.float64),
        torch.tensor([0, 0, 0, 1, 1]),
        [r for r, _ in trending],
    ).mean(0)
    assert a_loss < 0.05
    assert np.allclose(pred.numpy(), [q for _, q in trending], atol=0.5)


def test_anchor_baselines_report_smoothed_r_and_recover_the_shift(batches):
    """Constant shift and linear warp through the true anchors align the synthetic pair (a pure
    shift) about as well as PCHIP, and every PairEval carries the smoothed metrics."""
    from warpalign import evaluation
    from warpalign.config import SMOOTH_SIGMAS

    a, b = batches
    un = evaluation.unaligned(a, b)
    evs = {
        'pchip': evaluation.pchip(a, b, ANCHORS),
        'shift': evaluation.constant_shift(a, b, ANCHORS),
        'linear': evaluation.linear(a, b, ANCHORS),
    }
    for ev in (un, *evs.values()):
        assert set(ev.r_smooth) == set(ev.r_raw_smooth) == set(SMOOTH_SIGMAS)
        assert all(r.shape == ev.r.shape for r in ev.r_raw_smooth.values())
    for ev in evs.values():
        assert ev.mean > un.mean + 0.2
        assert abs(ev.mean - evs['pchip'].mean) < 0.05
