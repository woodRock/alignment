"""warp_transformer.py: warp primitives, exact anchor loss, model behaviour."""
import numpy as np
import pytest
import torch

from warp_transformer import (ChromaWarpTransformer, anchor_loss_approx, anchor_loss_exact,
                              apply_warp, invert_warp, make_monotone_warp, predict_query_bins,
                              sample_warp, warp_smoothness_loss)
from warps import warp_rows

T = 200


def ident(n=1, dtype=torch.float64):
    return torch.arange(T, dtype=dtype).repeat(n, 1)


# ── apply_warp ────────────────────────────────────────────────────────────────

def test_apply_warp_identity_is_exact_including_last_bin():
    """Section C, test 3 (fix 5: v1 read bin T−2 for the last output bin)."""
    x = torch.randn(3, T, 5, dtype=torch.float64)
    assert torch.equal(apply_warp(x, ident(3)), x)


def test_apply_warp_matches_numpy_warp(rng):
    x = rng.normal(size=(4, T, 3))
    w = np.sort(rng.uniform(-5, T + 5, size=(4, T)), axis=1)
    out = apply_warp(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    assert np.allclose(out, warp_rows(x, w))


def test_apply_warp_gradient_flows_through_positions():
    x = torch.randn(1, T, 2, dtype=torch.float64)
    w = (ident() + 0.3).requires_grad_()
    apply_warp(x, w).sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0


# ── make_monotone_warp ────────────────────────────────────────────────────────

def test_zero_logits_give_exact_identity():
    w = make_monotone_warp(torch.zeros(3, T, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(w, ident(3), atol=1e-12)


def test_warp_is_strictly_increasing_and_ends_are_free():
    g = torch.Generator().manual_seed(0)
    raw = torch.randn(5, T, generator=g) * 3
    start = torch.tensor([-7.0, 0.0, 3.0, 12.0, -1.0])
    w = make_monotone_warp(raw, start)
    assert (w[:, 1:] > w[:, :-1]).all()
    assert torch.allclose(w[:, 0], start)
    assert not torch.allclose(w[:, -1], torch.full((5,), T - 1.0))   # last bin not pinned


def test_constant_shift_is_representable():
    w = make_monotone_warp(torch.zeros(1, T), torch.tensor([6.0]))
    assert torch.allclose(w, ident(1, torch.float32) + 6.0)


def test_smoothness_loss_zero_for_any_shift_positive_otherwise():
    assert float(warp_smoothness_loss(ident(2) + 4.0)) == 0.0
    assert float(warp_smoothness_loss(ident(1) * 1.1)) > 0


# ── sample_warp / invert_warp (section C, test 1) ─────────────────────────────

def _random_warp(seed=0):
    g = torch.Generator().manual_seed(seed)
    return make_monotone_warp(torch.randn(1, T, generator=g, dtype=torch.float64),
                              torch.tensor([2.5], dtype=torch.float64))[0]


def test_invert_then_sample_round_trip():
    w = _random_warp()
    pos = torch.linspace(float(w[0]) - 10, float(w[-1]) + 10, 300, dtype=torch.float64)
    assert torch.allclose(sample_warp(w, invert_warp(w, pos)), pos, atol=1e-9)


def test_sample_then_invert_round_trip():
    w = _random_warp(1)
    t = torch.linspace(-5, T + 5, 301, dtype=torch.float64)
    assert torch.allclose(invert_warp(w, sample_warp(w, t)), t, atol=1e-9)


def test_sample_warp_at_integer_bins_is_the_warp():
    w = _random_warp(2)
    assert torch.allclose(sample_warp(w, torch.arange(T, dtype=torch.float64)), w)


# ── Exact vs approximate anchor loss (section C, tests 2–4) ───────────────────

def test_exact_loss_equals_approx_when_batch_a_is_identity():
    """Test 2: with w̄_A = identity, w̄_A⁻¹(r) = r and the two losses coincide."""
    g = torch.Generator().manual_seed(3)
    wb = make_monotone_warp(torch.randn(3, T, generator=g, dtype=torch.float64) * 0.3,
                            torch.tensor([-4.0, -5.0, -3.0], dtype=torch.float64))
    warps = torch.cat([ident(2), wb])
    sid = torch.tensor([0, 0, 1, 1, 1])
    anchors = [(40, 33), (90, 80), (150, 141)]
    assert torch.isclose(anchor_loss_exact(warps, sid, anchors),
                         anchor_loss_approx(warps, sid, anchors))


def test_exact_loss_is_zero_where_approx_is_wrong():
    """Test 3: A shifted by +5, B identity. A compound at A bin r sits at consensus r − 5, where
    B reads r − 5, so q = r − 5 is satisfied exactly; the v1 loss reads both at r and wants a
    drift of −5 in B − A where the warps differ by −5 … but with a nonlinear A warp the v1 loss
    is not zero."""
    wa = (ident(1) + 5.0)
    wb = ident(1)
    sid = torch.tensor([0, 1])
    anchors = [(60, 55), (120, 115)]
    assert float(anchor_loss_exact(torch.cat([wa, wb]), sid, anchors)) == pytest.approx(0, abs=1e-12)
    # nonlinear A warp: w_A(t) = t + 0.002 t² (stretch); B identity
    t = torch.arange(T, dtype=torch.float64)
    wa = (t + 0.002 * t ** 2).unsqueeze(0)
    tc = invert_warp(wa[0], torch.tensor([60.0, 120.0], dtype=torch.float64))
    anchors = [(60, float(tc[0])), (120, float(tc[1]))]          # truth: q = w_B(w_A⁻¹(r))
    warps = torch.cat([wa, wb])
    assert float(anchor_loss_exact(warps, sid, anchors)) == pytest.approx(0, abs=1e-12)
    assert float(anchor_loss_approx(warps, sid, anchors)) > 1.0


def test_predict_query_bins_shapes_and_values():
    warps = torch.cat([ident(2) + 3.0, ident(3) - 4.0])
    sid = torch.tensor([0, 0, 1, 1, 1])
    pred = predict_query_bins(warps, sid, [50, 100])
    assert pred.shape == (3, 2)
    assert torch.allclose(pred, torch.tensor([[43.0, 93.0]] * 3, dtype=torch.float64))


def test_exact_loss_gradcheck():
    """Test 4: analytic gradients of the exact loss match finite differences."""
    g = torch.Generator().manual_seed(5)
    raw = (torch.randn(4, T, generator=g, dtype=torch.float64) * 0.2).requires_grad_()
    start = torch.tensor([1.0, -1.0, -5.0, -6.0], dtype=torch.float64, requires_grad=True)
    sid = torch.tensor([0, 0, 1, 1])
    anchors = [(50, 44), (100, 93), (160, 150)]

    def f(r, s):
        return anchor_loss_exact(make_monotone_warp(r, s), sid, anchors)
    assert torch.autograd.gradcheck(f, (raw, start), eps=1e-6, atol=1e-5)


def test_losses_without_anchors_are_zero_with_gradient():
    w = (ident(2) + 1.0).requires_grad_()
    sid = torch.tensor([0, 1])
    for fn in (anchor_loss_exact, anchor_loss_approx):
        loss = fn(w, sid, [])
        assert float(loss) == 0.0 and loss.requires_grad


# ── Model ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def small_model():
    torch.manual_seed(0)
    return ChromaWarpTransformer(mz_bins=16, n_bins=T, d_model=32, n_heads=4, n_layers=1).eval()


def test_untrained_model_predicts_identity(small_model):
    x = torch.rand(5, T, 16)
    sid = torch.tensor([0, 0, 0, 1, 1])
    aligned, warps = small_model(x, sid)
    assert torch.allclose(warps, ident(5, torch.float32))
    assert torch.allclose(aligned, x)


def test_model_is_equivariant_to_sample_order(small_model):
    """No positional encoding over samples: permuting samples permutes the warps."""
    torch.manual_seed(1)
    for p in small_model.warp_head[-1].parameters():          # make the head non-trivial
        torch.nn.init.normal_(p, std=0.1)
    x = torch.rand(4, T, 16)
    sid = torch.tensor([0, 1, 0, 1])
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        w = small_model.warps(x, sid)
        wp = small_model.warps(x[perm], sid[perm])
    assert torch.allclose(w[perm], wp, atol=1e-5)
    for p in small_model.warp_head[-1].parameters():
        torch.nn.init.zeros_(p)


def test_default_model_state_dict_round_trip(tmp_path):
    m = ChromaWarpTransformer()
    torch.save(m.state_dict(), tmp_path / 'm.pt')
    m2 = ChromaWarpTransformer()
    m2.load_state_dict(torch.load(tmp_path / 'm.pt'))
    assert {k for k in m2.state_dict()} >= {'start_head.1.weight', 'warp_head.3.weight'}
