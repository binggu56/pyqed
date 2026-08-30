import numpy as np
import pytest

from pyqed.tn import PackedBlockEffectiveOperator, resolve_workers


def test_packed_block_operator_matches_dense_for_scalar_batch_and_diagonal():
    rng = np.random.default_rng(7)
    indices = (np.array([0, 2]), np.array([1, 3]))
    blocks = {
        (0, 0): rng.normal(size=(2, 2)),
        (0, 1): rng.normal(size=(2, 2)),
        (1, 0): rng.normal(size=(2, 2)),
        (1, 1): rng.normal(size=(2, 2)),
    }
    operator = PackedBlockEffectiveOperator(indices, blocks)
    dense = np.zeros((4, 4))
    for (row, column), block in blocks.items():
        dense[np.ix_(indices[row], indices[column])] = block

    vector = rng.normal(size=4) + 1.0j * rng.normal(size=4)
    vectors = np.stack((vector, 2.0 * vector.conj()))
    np.testing.assert_allclose(operator.matvec(vector), dense @ vector)
    np.testing.assert_allclose(operator.matvecs(vectors), vectors @ dense.T)
    np.testing.assert_allclose(operator.diagonal(), np.diag(dense))


def test_packed_block_operator_supports_reduced_precision_and_cuda_is_optional():
    operator = PackedBlockEffectiveOperator(
        (np.array([0, 1]),),
        {(0, 0): np.eye(2)},
        compute_dtype=np.float32,
    )
    assert operator.matvec(np.ones(2, dtype=np.float64)).dtype == np.float32
    try:
        import cupy  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="CuPy"):
            PackedBlockEffectiveOperator(
                (np.array([0, 1]),),
                {(0, 0): np.eye(2)},
                device="cuda",
            )


def test_worker_autotuning_avoids_nested_blas_threads(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    assert resolve_workers("auto") == 1
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    assert 1 <= resolve_workers("auto", maximum=3) <= 3


@pytest.mark.parametrize("complex_values", [False, True])
def test_packed_block_operator_selects_compiled_grouped_gemm_when_available(
    complex_values,
):
    from pyqed.mps import cpp_davidson

    if not cpp_davidson.CPP_DAVIDSON_AVAILABLE:
        pytest.skip(cpp_davidson.CPP_DAVIDSON_BUILD_ERROR or "C++ backend unavailable")
    rng = np.random.default_rng(23)
    indices = (np.array([0, 2]), np.array([1, 3]))
    blocks = {
        (row, column): rng.normal(size=(2, 2))
        + (1.0j * rng.normal(size=(2, 2)) if complex_values else 0.0)
        for row in range(2)
        for column in range(2)
    }
    operator = PackedBlockEffectiveOperator(indices, blocks)
    vectors = rng.normal(size=(5, 4)) + (
        1.0j * rng.normal(size=(5, 4)) if complex_values else 0.0
    )
    dense = np.zeros((4, 4), dtype=np.complex128 if complex_values else float)
    for (row, column), block in blocks.items():
        dense[np.ix_(indices[row], indices[column])] = block

    assert operator.backend == "cpp-grouped-gemm"
    np.testing.assert_allclose(operator.matvecs(vectors), vectors @ dense.T)
    np.testing.assert_allclose(operator.diagonal(), np.diag(dense))
