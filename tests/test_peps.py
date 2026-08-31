import numpy as np

from pyqed.lattice import Site
from pyqed.peps import (
    AbelianPEPS,
    AbelianPEPSTensor,
    PEPS,
    contract,
    simple_update_bond,
)


def _random_peps(shape, bond_dim=2, physical_dim=2, seed=7):
    rng = np.random.default_rng(seed)
    nrows, ncols = shape
    tensors = []
    for row in range(nrows):
        values = []
        for col in range(ncols):
            dimensions = (
                1 if col == 0 else bond_dim,
                1 if col + 1 == ncols else bond_dim,
                1 if row == 0 else bond_dim,
                1 if row + 1 == nrows else bond_dim,
                physical_dim,
            )
            values.append(rng.normal(size=dimensions) + 1j * rng.normal(size=dimensions))
        tensors.append(values)
    return PEPS(tensors)


def test_product_state_norm_and_local_expectation():
    zero = np.array([1.0, 0.0])
    one = np.array([0.0, 1.0])
    state = PEPS.product_state([zero, one, zero, one], (2, 2))
    z = np.diag([1.0, -1.0])

    assert np.allclose(state.norm_squared(method="exact"), 1.0)
    assert np.allclose(state.norm_squared(method="boundary"), 1.0)
    assert np.allclose(state.local_expectation((0, 0), z, method="exact"), 1.0)
    assert np.allclose(state.local_expectation((0, 1), z, method="boundary"), -1.0)


def test_boundary_mps_matches_exact_small_random_peps():
    state = _random_peps((2, 3), bond_dim=2)
    exact = contract(state, method="exact")
    boundary = contract(state, method="boundary", max_bond=None)
    assert np.allclose(boundary, exact, rtol=1.0e-11, atol=1.0e-11)


def test_boundary_truncation_converges_to_exact_result():
    state = _random_peps((3, 2), bond_dim=2)
    exact = contract(state, method="exact")
    untruncated = contract(state, method="boundary", max_bond=64)
    assert np.allclose(untruncated, exact, rtol=1.0e-11, atol=1.0e-11)


def test_simple_update_creates_entanglement_and_preserves_bonds():
    zero = np.array([1.0, 0.0])
    state = PEPS.product_state([zero, zero], (1, 2))
    # Maps |00> to (|00> + |11>) / sqrt(2); completing columns are irrelevant here.
    gate = np.eye(4, dtype=complex)
    gate[:, 0] = np.array([np.sqrt(0.8), 0.0, 0.0, np.sqrt(0.2)])
    gate = gate.reshape(2, 2, 2, 2)

    discarded = simple_update_bond(
        state, (0, 0), (0, 1), gate, max_bond=2, normalize=True
    )

    assert discarded < 1.0e-14
    assert state.tensors[0][0].shape[1] == 2
    assert state.tensors[0][1].shape[0] == 2
    assert np.allclose(state.bond_singular_values[state.bond_key((0, 0), (0, 1))], np.sqrt([0.8, 0.2]))
    z = np.diag([1.0, -1.0])
    assert np.allclose(state.local_expectation((0, 0), z, method="exact"), 0.6, atol=1.0e-12)


def test_abelian_block_tensor_roundtrip_and_contraction():
    dense = np.zeros((1, 1, 1, 1, 2), dtype=complex)
    dense[..., 1] = 1.0
    charges = [[(0,)], [(0,)], [(0,)], [(0,)], [(0,), (1,)]]
    tensor = AbelianPEPSTensor.from_dense(dense, charges)
    site = Site.spinless_fermion()
    state = AbelianPEPS([[tensor]], sites=[site])

    assert len(tensor.blocks) == 1
    assert np.array_equal(tensor.to_dense(), dense)
    assert np.allclose(state.norm_squared(method="exact"), 1.0)
    assert np.allclose(state.to_dense().local_expectation((0, 0), np.diag([0, 1])), 1.0)


def test_abelian_flux_validation():
    dense = np.ones((1, 1, 1, 1, 1))
    charges = [[(0,)]] * 4 + [[(1,)]]
    tensor = AbelianPEPSTensor.from_dense(dense, charges, total_charge=(1,))
    assert tensor.check_flux()
