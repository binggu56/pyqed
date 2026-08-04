import numpy as np
import pytest

from pyqed.mps import cpp_davidson


def _array_pool(arrays):
    arrays = tuple(np.ascontiguousarray(array, dtype=float) for array in arrays)
    return (
        np.asarray([0, *np.cumsum([array.size for array in arrays])], dtype=np.int64),
        np.asarray([0, *np.cumsum([array.ndim for array in arrays])], dtype=np.int64),
        np.asarray([dim for array in arrays for dim in array.shape], dtype=np.int64),
        np.concatenate([array.reshape(-1) for array in arrays]),
    )


def _raw_side(factor_indices, boundary_ids, w_ids, boundaries, w_arrays):
    boundary = _array_pool(boundaries)
    w_pool = _array_pool(w_arrays)
    return (
        np.asarray(factor_indices, dtype=np.int64),
        np.asarray(boundary_ids, dtype=np.int64),
        np.asarray(w_ids, dtype=np.int64),
        *boundary,
        *w_pool,
    )


def test_packed_su2_block_davidson_batches_dense_pair_kernels():
    table_cls = cpp_davidson.SU2PackedFactorizedFamilyTable
    matrix = np.asarray(
        [
            [-1.0, 0.2, 0.0],
            [0.2, 0.4, -0.15],
            [0.0, -0.15, 1.2],
        ]
    )
    route_count = 4
    packed_arrays = (
        np.zeros(route_count, dtype=np.int64),
        np.zeros(route_count, dtype=np.int64),
        np.arange(route_count, dtype=np.int64),
        np.zeros(route_count, dtype=np.int64),
        np.asarray([0, 3], dtype=np.int64),
        np.asarray([[3, 1, 1, 1]], dtype=np.int64),
        *_raw_side(
            range(route_count),
            range(route_count),
            [0] * route_count,
            [matrix.reshape(1, 3, 3) / route_count] * route_count,
            [np.ones((1, 1, 1, 1))],
        ),
        *_raw_side(
            [0],
            [0],
            [0],
            [np.ones((1, 1, 1))],
            [np.ones((1, 1, 1, 1))],
        ),
    )
    table = table_cls(
        (("dense", 0, np.eye(3, dtype=complex)),),
        (np.arange(3, dtype=np.int64),),
        packed_arrays,
        3,
        3,
    )
    assert table.stats["dense_pair_kernels"] == 1
    np.testing.assert_allclose(
        table.diagonal(),
        np.diag(matrix),
        atol=1.0e-13,
    )
    transform = np.asarray(
        [
            [1.0 / np.sqrt(2.0), 0.0],
            [1.0 / np.sqrt(2.0), 0.0],
            [0.0, 1.0],
        ],
        dtype=complex,
    )
    transformed_table = table_cls(
        (("dense", 0, transform),),
        (np.arange(3, dtype=np.int64),),
        packed_arrays,
        3,
        2,
    )
    transformed_matrix = transform.conj().T @ matrix @ transform
    np.testing.assert_allclose(
        transformed_table.diagonal(),
        np.diag(transformed_matrix),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        transformed_table.matmat(np.eye(2, dtype=complex)),
        transformed_matrix,
        atol=1.0e-13,
    )

    rng = np.random.default_rng(20260724)
    vectors = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    expected = matrix @ vectors
    np.testing.assert_allclose(table.matmat(vectors), expected, atol=1.0e-13)
    np.testing.assert_allclose(
        table.matmat(vectors),
        np.column_stack([table.matvec(vector) for vector in vectors.T]),
        atol=1.0e-13,
    )

    result = table.davidson_block(
        np.diag(matrix),
        np.ones(3, dtype=complex),
        1.0e-12,
        30,
        12,
        False,
        2,
    )
    assert result["accepted"]
    assert result["converged"]
    assert result["block_size"] == 2
    assert result["energy"] == pytest.approx(
        np.linalg.eigvalsh(matrix)[0],
        abs=1.0e-12,
    )
    assert table.stats["block_davidson_calls"] == 1
    assert table.stats["matmat_vectors"] > table.stats["matmat_calls"]
