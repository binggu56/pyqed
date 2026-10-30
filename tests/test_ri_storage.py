"""RI storage changes preserve the metric and Cartesian/spherical contractions."""

import numpy as np
import pytest
from pyqed.qchem import basis


@pytest.mark.parametrize("solver", ["cholesky", "eigh"])
@pytest.mark.parametrize("block_size", [1, 3, 31])
@pytest.mark.parametrize("order", ["C", "F"])
def test_metric_whitening_blocks(solver, block_size, order):
    rng = np.random.default_rng(18)
    a = rng.normal(size=(9, 9))
    metric = a @ a.T + np.eye(9)
    pairs = np.array(rng.normal(size=(9, 31)), order=order)
    saved = pairs.copy()
    factors, info = basis._metric_factorize_ri(
        metric, pairs, solver=solver, block_size=block_size, disk_backed=True
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


def test_disk_storage_uses_explicit_local_scratch(tmp_path, monkeypatch):
    scratch = tmp_path / 'ssd'
    monkeypatch.setenv('PYQED_SCRATCH_DIR', str(scratch))
    original = basis.tempfile.TemporaryFile
    destinations = []
    def create(*args, **kwargs):
        destinations.append(kwargs['dir'])
        return original(*args, **kwargs)
    monkeypatch.setattr(basis.tempfile, 'TemporaryFile', create)
    array = basis._ri_storage((1025, 4096), dtype=np.complex128)
    assert isinstance(array, np.memmap)
    assert destinations == [str(scratch)]
    array[0, 0] = 2+3j
    assert array[0, 0] == 2+3j


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
