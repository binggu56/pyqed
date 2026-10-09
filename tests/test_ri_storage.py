"""RI storage changes preserve the metric and Cartesian/spherical contractions."""

import numpy as np
import pytest
from pyqed.qchem import basis


@pytest.mark.parametrize("solver", ["cholesky", "eigh"])
def test_metric_whitening_blocks(solver):
    rng = np.random.default_rng(18)
    a = rng.normal(size=(9, 9))
    metric = a @ a.T + np.eye(9)
    pairs = rng.normal(size=(9, 31))
    saved = pairs.copy()
    factors, info = basis._metric_factorize_ri(
        metric, pairs, solver=solver, block_size=3, disk_backed=True
    )
    np.testing.assert_allclose(
        factors.T @ factors, pairs.T @ np.linalg.solve(metric, pairs), atol=1e-12
    )
    np.testing.assert_array_equal(pairs, saved)
    assert info["metric_rank"] == 9


def test_ri_large_storage_retains_mapping():
    array = basis._ri_storage((17, 724 * 725 // 2))
    assert isinstance(array, np.memmap)
    array[0, :3] = [1, 2, 3]
    packed = basis.PackedRIFactors(array, 724)
    assert np.shares_memory(packed.pair_factors, array)
    view = np.asarray(array)
    assert np.shares_memory(view, array)
    np.testing.assert_array_equal(view[0, :3], [1, 2, 3])


def test_spherical_ri_blocks():
    rng = np.random.default_rng(31)
    primary = rng.normal(size=(5, 3))
    auxiliary = rng.normal(size=(7, 4))
    metric = np.eye(7)
    dense = rng.normal(size=(7, 5, 5))
    dense = dense + dense.transpose(0, 2, 1)
    rows, cols = np.tril_indices(5)
    actual_metric, actual = basis._transform_ri_tensors_to_spherical(
        metric, dense[:, rows, cols], primary, auxiliary
    )
    expected = np.einsum("PA,Pmn,mi,nj->Aij", auxiliary, dense, primary, primary)
    rows, cols = np.tril_indices(3)
    np.testing.assert_allclose(actual, expected[:, rows, cols], atol=1e-12)
    np.testing.assert_allclose(actual_metric, auxiliary.T @ auxiliary, atol=1e-12)
