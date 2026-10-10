"""Streamed factors agree with explicitly materialized small references."""
import numpy as np
import pytest
from pyqed.qchem import basis


@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("solver", ["cholesky", "eigh"])
@pytest.mark.parametrize("block_size", [1, 80, 10000])
@pytest.mark.parametrize("screen", [0., 1e6])
def test_streamed_factors(spherical, solver, block_size, screen):
    shells = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
              (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    signatures = tuple((shell, center, (.8, .3), (.7, -.2))
                       for center in [(0., 0., 0.), (.3, .5, 1.4)] for shell in shells)
    auxiliary = tuple((shell, (.1, -.2, .4), (.9, .2), (.6, .3)) for shell in shells)
    bounds = np.ones((len(signatures), len(signatures)))
    transforms = (None, None)
    if spherical:
        transforms = tuple(basis._global_cartesian_to_spherical_transform(s)[0]
                           for s in (signatures, auxiliary))
    metric, raw, computed, skipped = basis._compute_native_ri_pair_tensors_cpp(
        signatures, auxiliary, bounds, screen, *transforms)
    expected, info = basis._metric_factorize_ri(metric, raw, solver=solver)
    actual = basis._compute_native_ri_pair_tensors_cpp(
        signatures, auxiliary, bounds, screen, *transforms,
        factor_options={"solver": solver, "block_size": block_size})
    np.testing.assert_allclose(actual[0], metric, atol=2e-12)
    np.testing.assert_allclose(actual[1], expected, atol=2e-12, rtol=2e-12)
    assert actual[2:] == (computed, skipped)
    profile = basis._NATIVE_RI_LAST_KERNEL_INFO
    assert profile["streamed"]
    assert profile["factor_info"]["metric_rank"] == info["metric_rank"]
    assert profile["triplet_seconds"] >= 0
    if block_size == 1:
        assert profile["block_count"] == 6
        assert profile["peak_block_pairs"] < raw.shape[1]


def test_metric_rank_fallback():
    metric = np.diag([1., 2., 0.])
    operator, info = basis._ri_metric_operator(metric, 1e-10, "auto")
    assert info == {"metric_solver": "eigh", "metric_rank": 2}
    np.testing.assert_allclose(operator @ metric @ operator.T, np.eye(2))


def test_stream_callback_error_is_propagated(monkeypatch):
    def fail(*args):
        raise RuntimeError("whitening failure")
    monkeypatch.setattr(basis, "_ri_metric_operator", fail)
    signatures = (((0, 0, 0), (0., 0., 0.), (.8,), (1.,)),)
    with pytest.raises(RuntimeError, match="whitening failure"):
        basis._compute_native_ri_pair_tensors_cpp(
            signatures, signatures, np.ones((1, 1)), 0., factor_options={})


def test_stream_returns_no_global_raw_tensor():
    signatures = tuple(((0, 0, 0), center, (.8,), (1.,))
                       for center in [(0., 0., 0.), (.3, .5, 1.4)])
    arrays = basis._pack_signatures_for_numba(signatures)
    blocks = []
    def collect(array, start, stop):
        if start is not None:
            blocks.append((start, stop, array.copy()))
    metric, empty, *_ = basis._integrals_cpp.compute_ri_tensors_packed(
        *arrays, *arrays, np.ones((2, 2)), 0., None, None, collect, 1)
    assert empty.shape == (2, 0)
    assert [(start, stop) for start, stop, _ in blocks] == [(0, 1), (1, 3)]
    reference = basis._compute_native_ri_pair_tensors_cpp(
        signatures, signatures, np.ones((2, 2)), 0.)
    np.testing.assert_allclose(metric, reference[0])
    np.testing.assert_allclose(np.concatenate([b[2] for b in blocks], axis=1), reference[1])


@pytest.mark.parametrize("location", [0, 1, 2])
@pytest.mark.parametrize("screen", [0., 1e-2])
def test_single_p_triplets(location, screen):
    centers = [(0., 0., 0.), (.3, -.4, .7), (-.2, .5, 1.1)]
    groups = []
    for i, center in enumerate(centers):
        angular = np.eye(3, dtype=int) if location == i else [(0, 0, 0)]
        groups.append(tuple((tuple(shell), center, (.8, .3), (.7+j*.03, -.2))
                            for j, shell in enumerate(angular)))
    primary, aux = groups[0]+groups[1], groups[2]
    bounds = np.arange(1, len(primary)**2+1).reshape(len(primary), -1)*1e-3
    actual = basis._compute_native_ri_pair_tensors_cpp(primary, aux, bounds, screen)
    reference = basis._compute_native_ri_pair_tensors_cython(primary, aux, bounds, screen)
    for a, b in zip(actual[:2], reference[:2]):
        np.testing.assert_allclose(a, b, atol=3e-12, rtol=3e-12)
