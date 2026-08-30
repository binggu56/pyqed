import itertools

import numpy as np
import pytest

from pyqed.mps import MPS, MPO
from pyqed.mps.abelian_direct import AbelianRenormalizedActionDataTable
from pyqed.mps.decompose import decompose, tt_to_tensor
from pyqed.mps.mps import LeftCanonical, PauliSite, RightCanonical, Site, apply_mpo


STANDARD_LABELS = ("lv", "p", "rv")


def _random_state(shape=(2, 3, 2), seed=17):
    rng = np.random.default_rng(seed)
    tensor = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return tensor, decompose(tensor, rank=max(shape) ** len(shape))


def _dense(state):
    return tt_to_tensor([state._get_std_B(i) for i in range(state.L)])


@pytest.mark.parametrize("labels", itertools.permutations(STANDARD_LABELS))
def test_mps_accepts_every_declared_tensor_layout(labels):
    tensor, factors = _random_state()
    axes = [STANDARD_LABELS.index(label) for label in labels]
    state = MPS([factor.transpose(axes) for factor in factors], labels=labels)

    assert state.check_sanity()
    np.testing.assert_allclose(_dense(state), tensor, atol=1.0e-12)
    assert state.bond_orders() == [factor.shape[2] for factor in factors]


def test_mps_uses_left_physical_right_as_its_default_order():
    _, factors = _random_state()
    state = MPS(factors)

    assert state.labels == ["lv", "p", "rv"]
    assert state.lv_idx == 0
    assert state.p_idx == 1
    assert state.rv_idx == 2


def test_mps_relabeling_transposes_data_without_changing_the_state():
    tensor, factors = _random_state()
    state = MPS(factors)

    state.set_labels(["p", "rv", "lv"])
    assert state.labels == ["p", "rv", "lv"]
    np.testing.assert_allclose(_dense(state), tensor, atol=1.0e-12)

    reordered = state.to_order(["lv", "p", "rv"])
    assert reordered.labels == ["lv", "p", "rv"]
    np.testing.assert_allclose(_dense(reordered), tensor, atol=1.0e-12)


@pytest.mark.parametrize(
    ("alias", "canonical", "center"),
    [
        ("left", "left_canonical", 2),
        ("left_canonical", "left_canonical", 2),
        ("right", "right_canonical", 0),
        ("right_canonical", "right_canonical", 0),
    ],
)
def test_mps_gauge_aliases_are_canonicalized(alias, canonical, center):
    _, factors = _random_state()
    state = MPS(factors, gauge=alias)

    assert state.gauge == canonical
    assert state.center == center
    assert state.norm() == pytest.approx(state.norm_squared())


def test_mps_mixed_gauge_requires_a_center():
    _, factors = _random_state()

    with pytest.raises(ValueError, match="requires an explicit center"):
        MPS(factors, gauge="mixed")

    state = MPS(factors, gauge="mixed", center=1)
    assert state.center == 1
    assert state.gauge == "mixed"


def test_mps_canonicalization_preserves_state_and_exposes_schmidt_values():
    tensor, factors = _random_state()
    tensor = tensor / np.linalg.norm(tensor)
    state = MPS(decompose(tensor, rank=12))

    state.right_canonicalize()
    assert state.gauge == "right_canonical"
    assert state.center == 0
    assert state.check_sanity()
    np.testing.assert_allclose(_dense(state), tensor, atol=1.0e-12)
    assert state.norm() == pytest.approx(1.0)
    for i, expected_size in enumerate(state.get_bond_dimensions()):
        assert state.get_singular_values(i).shape == (expected_size,)

    state.left_to_right()
    assert state.gauge == "right_canonical"


def test_mps_vidal_factors_reconstruct_the_state():
    _, factors = _random_state()
    state = MPS(factors).normalize()
    gammas, lambdas = state.left_to_vidal()

    reconstructed = gammas[0]
    for i, values in enumerate(lambdas):
        reconstructed = np.tensordot(
            reconstructed, np.diag(values), axes=([-1], [0])
        )
        reconstructed = np.tensordot(
            reconstructed, gammas[i + 1], axes=([-1], [0])
        )
    reconstructed = np.squeeze(reconstructed, axis=(0, -1))
    np.testing.assert_allclose(reconstructed, _dense(state), atol=1.0e-12)


def test_mps_compression_preserves_scale_and_is_scale_invariant():
    tensor, factors = _random_state(seed=23)
    state = MPS(factors)
    compressed = state.compress(1)

    scaled_factors = [factor.copy() for factor in factors]
    scaled_factors[0] *= 1.0e-12
    scaled = MPS(scaled_factors).compress(1)

    assert compressed.norm() != pytest.approx(1.0)
    np.testing.assert_allclose(
        _dense(scaled), 1.0e-12 * _dense(compressed), rtol=1.0e-12, atol=1.0e-24
    )


def test_two_site_expectation_does_not_apply_center_schmidt_values_twice():
    rng = np.random.default_rng(31)
    tensor, factors = _random_state(shape=(2, 2), seed=29)
    state = MPS(factors).normalize().right_canonicalize()
    operator = rng.normal(size=(2, 2, 2, 2))

    dense = _dense(state)
    applied = np.tensordot(operator, dense, axes=([2, 3], [0, 1]))
    expected = np.vdot(dense, applied)

    np.testing.assert_allclose(
        state.bond_expectation_value([operator]), [expected], atol=1.0e-12
    )


def test_mps_sanity_checks_boundaries_bonds_and_zero_normalization():
    with pytest.raises(ValueError, match="unit left and right boundary bonds"):
        MPS([np.ones((2, 2, 1))]).check_sanity()

    with pytest.raises(ValueError, match="incompatible dimensions"):
        MPS([np.ones((1, 2, 3)), np.ones((2, 2, 1))]).check_sanity()

    with pytest.raises(ValueError, match="zero MPS"):
        MPS([np.zeros((1, 2, 1))]).normalize()


def test_get_singular_values_requires_canonicalization():
    _, factors = _random_state()
    state = MPS(factors)

    with pytest.raises(ValueError, match="canonicalize"):
        state.get_singular_values(0)
    with pytest.raises(IndexError, match="out of range"):
        state.get_singular_values(state.nbonds)


def test_dense_helpers_use_the_canonical_lattice_sites():
    tensor, factors = _random_state()

    left = LeftCanonical(factors)
    right = RightCanonical(factors)
    np.testing.assert_allclose(tt_to_tensor(left), tensor / np.linalg.norm(tensor))
    np.testing.assert_allclose(tt_to_tensor(right), tensor / np.linalg.norm(tensor))

    site = Site(2)
    spin = SpinHalfSite()
    assert site.operators["I"].shape == (2, 2)
    np.testing.assert_allclose(spin.operator("Sz"), np.diag([0.5, -0.5]))


def test_mpo_validates_its_fixed_tensor_order():
    identity = np.eye(2).reshape(1, 1, 2, 2)
    mpo = MPO([identity])

    assert mpo.labels == ("left", "right", "out", "in")
    with pytest.raises(ValueError, match="must use"):
        MPO([identity], labels=("left", "out", "in", "right"))


@pytest.mark.parametrize("labels", itertools.permutations(STANDARD_LABELS))
def test_mpo_application_preserves_scale_and_declared_mps_layout(labels):
    tensor, factors = _random_state()
    axes = [STANDARD_LABELS.index(label) for label in labels]
    state = MPS([factor.transpose(axes) for factor in factors], labels=labels)
    local_ops = [
        np.diag([2.0, -0.5]),
        np.array([[1.0, 0.2, 0.0], [0.1, 1.5, 0.3], [0.0, -0.4, 0.7]]),
        np.array([[0.8, 0.1], [-0.2, 1.2]]),
    ]
    mpo = MPO([op.reshape(1, 1, *op.shape) for op in local_ops])
    expected = np.einsum(
        "ai,bj,ck,ijk->abc", *local_ops, tensor, optimize=True
    )

    compressed = MPS(apply_mpo(mpo, state, chi_max=32))
    np.testing.assert_allclose(_dense(compressed), expected, atol=1.0e-12)
    np.testing.assert_allclose(_dense(mpo.dot(state, D=32)), expected, atol=1.0e-12)
    np.testing.assert_allclose(
        _dense(mpo.matmul(state, chi_max=32)), expected, atol=1.0e-12
    )
    np.testing.assert_allclose(_dense(mpo @ state), expected, atol=1.0e-12)


def test_apply_mpo_validates_lengths_and_physical_dimensions():
    identity = np.eye(2).reshape(1, 1, 2, 2)
    state = MPS([np.ones((1, 2, 1)), np.ones((1, 2, 1))])

    with pytest.raises(ValueError, match="lengths must match"):
        apply_mpo([identity], state, chi_max=2)
    with pytest.raises(ValueError, match="Physical input dimension mismatch"):
        apply_mpo(
            [np.eye(3).reshape(1, 1, 3, 3), identity],
            state,
            chi_max=2,
        )


def test_apply_mpo_handles_the_zero_operator_without_nan_bonds():
    _, factors = _random_state(shape=(2, 2), seed=41)
    state = MPS(factors)
    zero = np.zeros((1, 1, 2, 2))

    result = MPS(apply_mpo([zero, zero], state, chi_max=4))

    assert result.check_sanity()
    assert all(np.all(np.isfinite(factor)) for factor in result.factors)
    np.testing.assert_allclose(_dense(result), 0.0)


def test_apply_mpo_preserves_bonded_mpo_virtual_index_order():
    tensor, factors = _random_state(shape=(2, 2), seed=43)
    state = MPS(factors)
    identity = np.eye(2)
    x_op = np.array([[0.0, 1.0], [1.0, 0.0]])
    z_op = np.diag([1.0, -1.0])
    first = np.zeros((1, 2, 2, 2))
    last = np.zeros((2, 1, 2, 2))
    first[0, 0] = identity
    first[0, 1] = x_op
    last[0, 0] = identity
    last[1, 0] = z_op
    mpo = MPO([first, last])
    expected = tensor + np.einsum("ai,bj,ij->ab", x_op, z_op, tensor)

    np.testing.assert_allclose(
        _dense(MPS(apply_mpo(mpo, state, chi_max=4))),
        expected,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(_dense(mpo @ state), expected, atol=1.0e-12)
    np.testing.assert_allclose(_dense(mpo.matmul(state)), expected, atol=1.0e-12)


def test_compressed_mpo_product_preserves_operator_scale():
    identity = np.eye(2).reshape(1, 1, 2, 2)
    left = MPO([2.0 * identity, identity])
    right = MPO([3.0 * identity, identity])

    product = left.matmul(right, chi_max=1)

    state = MPS([np.ones((1, 2, 1)), np.ones((1, 2, 1))])
    np.testing.assert_allclose(_dense(product @ state), 6.0 * _dense(state))


@pytest.mark.parametrize("labels", itertools.permutations(STANDARD_LABELS))
def test_mpo_application_preserves_scale_and_declared_mps_layout(labels):
    tensor, factors = _random_state()
    axes = [STANDARD_LABELS.index(label) for label in labels]
    state = MPS([factor.transpose(axes) for factor in factors], labels=labels)
    local_ops = [
        np.diag([2.0, -0.5]),
        np.array([[1.0, 0.2, 0.0], [0.1, 1.5, 0.3], [0.0, -0.4, 0.7]]),
        np.array([[0.8, 0.1], [-0.2, 1.2]]),
    ]
    mpo = MPO([op.reshape(1, 1, *op.shape) for op in local_ops])
    expected = np.einsum(
        "ai,bj,ck,ijk->abc", *local_ops, tensor, optimize=True
    )

    compressed = MPS(apply_mpo(mpo, state, chi_max=32))
    np.testing.assert_allclose(_dense(compressed), expected, atol=1.0e-12)
    np.testing.assert_allclose(_dense(mpo.dot(state, D=32)), expected, atol=1.0e-12)
    np.testing.assert_allclose(
        _dense(mpo.matmul(state, chi_max=32)), expected, atol=1.0e-12
    )
    np.testing.assert_allclose(_dense(mpo @ state), expected, atol=1.0e-12)


def test_apply_mpo_validates_lengths_and_physical_dimensions():
    identity = np.eye(2).reshape(1, 1, 2, 2)
    state = MPS([np.ones((1, 2, 1)), np.ones((1, 2, 1))])

    with pytest.raises(ValueError, match="lengths must match"):
        apply_mpo([identity], state, chi_max=2)
    with pytest.raises(ValueError, match="Physical input dimension mismatch"):
        apply_mpo(
            [np.eye(3).reshape(1, 1, 3, 3), identity],
            state,
            chi_max=2,
        )


def test_apply_mpo_handles_the_zero_operator_without_nan_bonds():
    _, factors = _random_state(shape=(2, 2), seed=41)
    state = MPS(factors)
    zero = np.zeros((1, 1, 2, 2))

    result = MPS(apply_mpo([zero, zero], state, chi_max=4))

    assert result.check_sanity()
    assert all(np.all(np.isfinite(factor)) for factor in result.factors)
    np.testing.assert_allclose(_dense(result), 0.0)


def test_apply_mpo_preserves_bonded_mpo_virtual_index_order():
    tensor, factors = _random_state(shape=(2, 2), seed=43)
    state = MPS(factors)
    identity = np.eye(2)
    x_op = np.array([[0.0, 1.0], [1.0, 0.0]])
    z_op = np.diag([1.0, -1.0])
    first = np.zeros((1, 2, 2, 2))
    last = np.zeros((2, 1, 2, 2))
    first[0, 0] = identity
    first[0, 1] = x_op
    last[0, 0] = identity
    last[1, 0] = z_op
    mpo = MPO([first, last])
    expected = tensor + np.einsum("ai,bj,ij->ab", x_op, z_op, tensor)

    np.testing.assert_allclose(
        _dense(MPS(apply_mpo(mpo, state, chi_max=4))),
        expected,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(_dense(mpo @ state), expected, atol=1.0e-12)
    np.testing.assert_allclose(_dense(mpo.matmul(state)), expected, atol=1.0e-12)


def test_compressed_mpo_product_preserves_operator_scale():
    identity = np.eye(2).reshape(1, 1, 2, 2)
    left = MPO([2.0 * identity, identity])
    right = MPO([3.0 * identity, identity])

    product = left.matmul(right, chi_max=1)

    state = MPS([np.ones((1, 2, 1)), np.ones((1, 2, 1))])
    np.testing.assert_allclose(_dense(product @ state), 6.0 * _dense(state))


def test_renormalized_table_python_fallback_applies_raw_entries():
    collected = {
        "left": (np.asarray([[[1.0, 2.0], [3.0, 4.0]]]),),
        "right": (np.asarray([[[2.0]]]),),
        "dims_array": np.asarray([[2, 1, 1, 1, 2, 1, 1, 1]], dtype=np.int64),
        "in_starts_array": np.asarray([0], dtype=np.int64),
        "out_starts_array": np.asarray([0], dtype=np.int64),
        "scales_array": np.asarray([0.5], dtype=np.complex128),
        "matvec_groups": None,
    }
    table = AbelianRenormalizedActionDataTable(
        collected,
        dim=2,
        layout=(((0, 0, 0, 0), (2, 1, 1, 1)),),
        qns=((0,), (0,), (0,), (0,)),
        dirs=(1, 1, -1, -1),
        kernel_backend=None,
    )

    np.testing.assert_allclose(table.matvec([5.0, 7.0]), [19.0, 43.0])
