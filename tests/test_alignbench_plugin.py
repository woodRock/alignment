"""The WarpTransformer as an alignbench plugin (warpalign 3.0) and the deprecated re-exports."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from alignbench import methods
from alignbench.data import Batch, Study
from alignbench.warps import invert_src

from warpalign import alignbench_plugin as plugin
from warpalign.warp_transformer import ChromaWarpTransformer

GRID = np.arange(200, dtype=float)


@pytest.fixture(scope='module')
def study(synthetic_pair):
    (ca, ra), (cb, rb) = synthetic_pair
    return Study(
        [Batch('a', ca, ra, ('a0', 'a1', 'a2', 'a3')), Batch('b', cb, rb, ('b0', 'b1', 'b2', 'b3'))]
    )


@pytest.fixture(scope='module')
def ckpt(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp('ckpt') / 'random.pt'
    torch.save(ChromaWarpTransformer().state_dict(), path)
    return path


def test_registered_and_needs_a_checkpoint(study):
    assert isinstance(methods.get('warp-transformer'), plugin.WarpTransformer)
    with pytest.raises(ValueError, match='ckpt'):
        methods.get('warp-transformer').fit(study, None)


def test_map_is_script32_reduction_of_the_calibrated_warps(study, ckpt):
    fit = methods.get('warp-transformer').fit(study, None, ckpt=ckpt, calib_steps=2)
    assert fit.per_sample and fit.params['device'] == 'cpu' and fit.params['calib_steps'] == 2
    m = fit.map(1, 0, GRID, i=2, j=1)  # B_2 → A_1
    res = fit.extras['calibrations'][(0, 1)]
    W = np.stack([plugin._strict(w) for w in res.warps])
    assert np.array_equal(m, plugin._read_warp(W[1], invert_src(W[4 + 2], GRID), GRID))
    assert np.all(np.diff(m) >= 0)
    assert len(fit.extras['calibrations']) == 1  # cached: the same pair is not recalibrated
    fit.map(1, 0, GRID, i=0, j=0)
    assert len(fit.extras['calibrations']) == 1
    assert np.array_equal(fit.map(0, 0, GRID, i=1, j=1), GRID)
    with pytest.raises(ValueError, match='per sample'):
        fit.map(1, 0, GRID)


def test_read_warp_continues_linearly():
    w = np.arange(200, dtype=float) * 2.0
    assert np.allclose(plugin._read_warp(w, np.array([-1.0, 0.5, 200.0]), GRID), [-2.0, 1.0, 400.0])


def test_old_module_names_warn_and_reexport():
    import importlib
    import sys

    for mod in ('anchors', 'warps', 'metrics', 'baselines', 'data'):
        sys.modules.pop(f'warpalign.{mod}', None)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            m = importlib.import_module(f'warpalign.{mod}')
        assert any(
            issubclass(x.category, DeprecationWarning) and 'alignbench' in str(x.message) for x in w
        )
        assert m.__all__ and all(hasattr(m, name) for name in m.__all__)
    from alignbench.anchors import match_pooled_arrays

    from warpalign import anchors

    assert anchors.extract_anchors.__module__ == 'warpalign._compat'
    assert anchors.match_anchors.__name__ == 'match_sample_pair' and match_pooled_arrays
