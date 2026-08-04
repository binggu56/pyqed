import numpy as np
import pytest
from opt_einsum import contract

from pyqed.letta.copy_einsum import (
    contract_class_einsum,
    contract_copy_einsum,
    native_available,
)


pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="optional LETTA copy-einsum extension is unavailable",
)


def _copy_tensor(dimension):
    result = np.zeros((dimension, dimension, dimension))
    diagonal = np.arange(dimension)
    result[diagonal, diagonal, diagonal] = 1.0
    return result


@pytest.mark.parametrize("complex_values", [False, True])
def test_copy_einsum_preserves_duplicate_output_axes(complex_values):
    rng = np.random.default_rng(9)
    left = rng.normal(size=(3, 2, 2))
    right = rng.normal(size=(3, 2, 2))
    operator = rng.normal(size=(3, 2, 2))
    if complex_values:
        left = left + 1.0j * rng.normal(size=left.shape)
        right = right + 1.0j * rng.normal(size=right.shape)
        operator = operator + 1.0j * rng.normal(size=operator.shape)
    copy = _copy_tensor(2)
    expected = contract(
        left,
        (0, 1, 2),
        right,
        (0, 3, 4),
        operator,
        (0, 5, 6),
        copy,
        (2, 5, 6),
        copy,
        (4, 7, 8),
        (1, 3, 5, 6, 7, 8),
    )
    actual = contract_copy_einsum(
        left,
        right,
        operator,
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5, 6, 7, 8),
        ((2, 5, 6), (4, 7, 8)),
        (2, 2),
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)
    assert np.count_nonzero(actual) == 16
    for bra in range(2):
        for ket in range(2):
            if bra != ket:
                assert not np.any(actual[:, :, bra, ket])


def test_copy_einsum_rejects_inconsistent_copy_dimensions():
    with pytest.raises(ValueError, match="inconsistent"):
        contract_copy_einsum(
            np.ones((2, 3)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            ((1, 2, 3),),
            (2,),
        )


@pytest.mark.parametrize("complex_values", [False, True])
def test_class_einsum_two_operands_preserves_duplicate_output_class(
    complex_values,
):
    rng = np.random.default_rng(11)
    left = rng.normal(size=(2, 3))
    right = rng.normal(size=(3, 2))
    if complex_values:
        left = left + 1.0j * rng.normal(size=left.shape)
        right = right + 1.0j * rng.normal(size=right.shape)
    expected = contract(
        left,
        (0, 1),
        right,
        (1, 2),
        _copy_tensor(2),
        (2, 3, 4),
        (0, 3, 4),
    )
    actual = contract_class_einsum(
        (left, right),
        ((0, 1), (1, 2)),
        (0, 2, 2),
        (2, 3, 2),
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=2.0e-15)
    for row in range(2):
        for column in range(2):
            if row != column:
                np.testing.assert_array_equal(actual[:, row, column], 0.0)


@pytest.mark.parametrize("complex_values", [False, True])
def test_class_einsum_four_operands_matches_einsum(complex_values):
    rng = np.random.default_rng(12)
    operands = (
        rng.normal(size=(3, 2, 2)),
        rng.normal(size=(2, 2, 2)),
        rng.normal(size=(2, 2, 2)),
        rng.normal(size=(3, 2, 2)),
    )
    if complex_values:
        operands = tuple(
            value + 1.0j * rng.normal(size=value.shape)
            for value in operands
        )
    classes = (
        (0, 1, 2),
        (1, 3, 4),
        (2, 3, 5),
        (0, 4, 5),
    )
    expected = contract(
        operands[0],
        classes[0],
        operands[1],
        classes[1],
        operands[2],
        classes[2],
        operands[3],
        classes[3],
        (3,),
    )
    actual = contract_class_einsum(
        operands,
        classes,
        (3,),
        (3, 2, 2, 2, 2, 2),
    )

    np.testing.assert_allclose(actual, expected, rtol=3.0e-15, atol=2.0e-14)


def test_class_einsum_preserves_scalar_operand_rank():
    vector = np.arange(3.0)
    actual = contract_class_einsum(
        (np.asarray(2.0), vector),
        ((), (0,)),
        (0,),
        (3,),
    )

    np.testing.assert_array_equal(actual, 2.0 * vector)
