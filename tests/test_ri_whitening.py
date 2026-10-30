"""Parallel packed whitening against independent triangular/eigen references."""
import numpy as np
import pytest
from scipy.linalg import solve_triangular
from pyqed.qchem import basis


@pytest.mark.parametrize('workers', [1, 16])
@pytest.mark.parametrize('width', [1, 511, 512, 1025, 2051])
@pytest.mark.parametrize('solver', ['cholesky', 'eigh'])
def test_whitening_strided_output(workers, width, solver):
    rng = np.random.default_rng(41)
    n = 23
    if solver == 'cholesky':
        operator = np.tril(rng.normal(scale=.02, size=(n, n))) + np.eye(n)
    else:
        operator = rng.normal(size=(n-5, n))
    block = np.array(rng.normal(size=(n, width)), order='F')
    saved = block.copy()
    storage = np.full((len(operator), width+11), np.nan)
    out = storage[:, 3:3+width]
    actual = basis._whiten_ri_block(operator, block, solver, out=out, workers=workers)
    expected = (solve_triangular(operator, block, lower=True)
                if solver == 'cholesky' else operator @ block)
    assert actual is out
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_array_equal(block, saved)
    assert np.isnan(storage[:, :3]).all() and np.isnan(storage[:, 3+width:]).all()


@pytest.mark.parametrize('workers', [1, 16])
def test_whitening_ill_conditioned_metric(workers):
    rng = np.random.default_rng(42)
    q, _ = np.linalg.qr(rng.normal(size=(24, 24)))
    metric = (q * np.geomspace(1e-10, 1., 24)) @ q.T
    lower = np.linalg.cholesky(metric)
    block = rng.normal(size=(24, 2051))
    actual = basis._whiten_ri_block(lower, block, 'cholesky', workers=workers)
    expected = solve_triangular(lower, block, lower=True)
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-10)
    np.testing.assert_allclose(lower @ actual, block, atol=2e-11, rtol=2e-11)


def test_whitening_rejects_invalid_destination():
    if not hasattr(basis._integrals_cpp, 'whiten_ri_block'):
        pytest.skip('Compiled whitening unavailable')
    lower, block = np.eye(3), np.ones((3, 5))
    for out in [np.empty((3, 5), dtype=np.float32), np.empty((3, 6)),
                np.empty((3, 10))[:, ::2]]:
        with pytest.raises(ValueError):
            basis._integrals_cpp.whiten_ri_block(lower, block, out, 4, True, basis._ri_blas.__pyx_capi__['dtrsm'])


@pytest.mark.parametrize('width', [129, 326, 513])
@pytest.mark.parametrize('solver', ['cholesky', 'eigh'])
def test_large_metric_narrow_panels(width, solver):
    rng = np.random.default_rng(47)
    n = 513
    operator = rng.normal(scale=.002, size=(n, n))
    if solver == 'cholesky':
        operator = np.tril(operator)+np.eye(n)
    block = rng.normal(size=(n, width))
    storage = np.full((n, width+7), np.nan)
    out = storage[:, 3:width+3]
    basis._whiten_ri_block(operator, block, solver, out=out, workers=4)
    reference = (solve_triangular(operator, block, lower=True)
                 if solver == 'cholesky' else operator@block)
    np.testing.assert_allclose(out, reference, atol=2e-12, rtol=2e-12)
    assert np.isnan(storage[:, :3]).all() and np.isnan(storage[:, width+3:]).all()


@pytest.mark.parametrize('workers', [1, 4])
def test_whitening_row_offset_beyond_int32(workers):
    import mmap

    if np.dtype(np.intp).itemsize < 8:
        pytest.skip('Requires a 64-bit address space')
    if not hasattr(basis._integrals_cpp, 'whiten_ri_block'):
        pytest.skip('Compiled whitening unavailable')
    # Reserve virtual address space; only three short row panels are touched.
    # Each BLAS dimension fits int, while the final row offset does not.
    stride, width = 2**30, 2051
    assert stride <= np.iinfo(np.int32).max < 2 * stride
    lower = np.array([[2., 0., 0.], [.3, 1.5, 0.], [-.2, .4, 1.2]], order='F')
    block = np.random.default_rng(43).normal(size=(3, width))
    expected = solve_triangular(lower, block, lower=True)
    with mmap.mmap(-1, (2 * stride + width) * 8) as storage:
        out = np.ndarray((3, width), dtype=np.float64, buffer=storage,
                         strides=(stride * 8, 8))
        basis._integrals_cpp.whiten_ri_block(
            lower, block, out, workers, True, basis._ri_blas.__pyx_capi__['dtrsm'])
        np.testing.assert_allclose(out, expected, atol=2e-13, rtol=2e-13)
        del out
