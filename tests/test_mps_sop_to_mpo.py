import numpy as np
import pytest

from pyqed.mps.mpo import sop_to_mpo
from pyqed.mps.mps import _mpo_to_dense_operator


def _kron_all(operators):
    result = np.asarray(operators[0])
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def test_sop_to_mpo_matches_dense_kron_sum():
    rng = np.random.default_rng(1234)
    dims = (2, 3, 2)
    terms = []
    expected = np.zeros((12, 12), dtype=complex)
    for coefficient in (0.7, -0.2j, 1.3 + 0.4j):
        operators = tuple(
            rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
            for dim in dims
        )
        terms.append((coefficient, operators))
        expected += coefficient * _kron_all(operators)

    mpo = sop_to_mpo(dims, terms)

    assert mpo.dims == list(dims)
    assert mpo.bond_orders() == [3, 3, 1]
    np.testing.assert_allclose(
        _mpo_to_dense_operator(mpo),
        expected,
        atol=1.0e-12,
    )


def test_sop_to_mpo_accepts_labeled_flat_terms_and_identity_factors():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    n = np.diag([0.0, 1.0, 2.0])
    dims = (2, 3, 2)

    mpo = sop_to_mpo(
        dims,
        [
            ("stretch_bend", 1.5, x, None, z),
            (-0.25, z, n, x),
        ],
    )

    expected = (
        1.5 * _kron_all((x, np.eye(3), z))
        - 0.25 * _kron_all((z, n, x))
    )
    np.testing.assert_allclose(
        _mpo_to_dense_operator(mpo),
        expected,
        atol=1.0e-12,
    )


def test_sop_to_mpo_one_site_and_empty_sum():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])

    one_site = sop_to_mpo((2,), [(2.0, x), (-0.5j, z)])
    np.testing.assert_allclose(
        _mpo_to_dense_operator(one_site),
        2.0 * x - 0.5j * z,
        atol=1.0e-12,
    )

    zero = sop_to_mpo((2, 3), [])
    np.testing.assert_allclose(
        _mpo_to_dense_operator(zero),
        np.zeros((6, 6), dtype=complex),
        atol=1.0e-12,
    )


def test_sop_to_mpo_compresses_redundant_channels():
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])

    mpo = sop_to_mpo(
        (2, 2),
        [(1.0, (x, z)), (2.0, (x, z))],
        max_rank=1,
    )

    assert max(mpo.bond_orders()) == 1
    np.testing.assert_allclose(
        _mpo_to_dense_operator(mpo),
        3.0 * np.kron(x, z),
        atol=1.0e-12,
    )


def test_sop_to_mpo_rejects_bad_local_shapes():
    with pytest.raises(ValueError, match="operator at site 1"):
        sop_to_mpo((2, 3), [(1.0, (np.eye(2), np.eye(2)))])
