import numpy as np
from scipy.linalg import expm

from pyqed.letta import (
    NNNLETTATDVPEngine,
    nnn_product_state,
    nnn_system_reduced_density_matrix,
    one_site_nnn_tdvp_step,
)
from pyqed.mps.mpo import sop_to_mpo


def _dense_product_state(factors):
    out = np.asarray(factors[0], dtype=complex)
    for factor in factors[1:]:
        out = np.kron(out, factor)
    return out / np.linalg.norm(out)


def _random_three_site_mpo(dims, seed=7):
    rng = np.random.default_rng(seed)
    terms = []
    for _ in range(8):
        operators = []
        for dim in dims:
            matrix = rng.normal(size=(dim, dim)) + 1.0j * rng.normal(
                size=(dim, dim)
            )
            operators.append(matrix + matrix.conj().T)
        terms.append((rng.normal(), operators))
    return sop_to_mpo(dims, terms)


def _phase_error(actual, reference):
    overlap = np.vdot(reference, actual)
    phase = 1.0 if abs(overlap) == 0.0 else overlap / abs(overlap)
    return np.linalg.norm(actual - phase * reference)


def test_nnn_product_state_is_exact_with_padded_ranks():
    factors = ([1.0, 2.0j], [0.0, 1.0], [2.0, -1.0], [1.0j, 3.0])
    state = nnn_product_state(factors, max_bond=3)
    np.testing.assert_allclose(
        state.state_vector(), _dense_product_state(factors), atol=2.0e-14
    )
    assert tuple(tensor.shape[-1] for tensor in state.tensors[:-1]) == (2,)


def test_nnn_system_reduced_density_matrix_matches_dense_state():
    state = nnn_product_state(
        ([1.0, 1.0j], [1.0, 2.0], [0.5, -1.0], [1.0, -0.2j]),
        max_bond=2,
    )
    state = one_site_nnn_tdvp_step(
        state, _random_three_site_mpo(state.dims, seed=31), 0.01,
        krylov_dim=48, krylov_tol=1.0e-13,
    )
    vector = state.state_vector().reshape(state.dims[0], -1)
    expected = vector @ vector.conj().T
    expected /= np.trace(expected)
    np.testing.assert_allclose(
        nnn_system_reduced_density_matrix(state), expected, atol=3.0e-12
    )


def test_nnn_tdvp_is_exact_when_one_tensor_spans_the_system():
    dims = (2, 2, 2)
    factors = ([1.0, 0.0], [0.0, 1.0], [1.0, 1.0j])
    state = nnn_product_state(factors, max_bond=2)
    mpo = _random_three_site_mpo(dims)
    actual, info = one_site_nnn_tdvp_step(
        state, mpo, 0.013, krylov_dim=64, krylov_tol=1.0e-14,
        return_info=True,
    )
    expected = expm(-0.013j * mpo.to_dense()) @ state.state_vector()
    np.testing.assert_allclose(actual.state_vector(), expected, atol=3.0e-12)
    assert info["integrator"] == "nnn-tdvp1"


def test_nnn_tdvp_is_time_reversible_at_fixed_rank():
    dims = (2, 2, 2, 2)
    factors = ([1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0])
    initial = nnn_product_state(factors, max_bond=2)
    mpo = _random_three_site_mpo(dims, seed=13)
    engine = NNNLETTATDVPEngine(mpo, krylov_dim=64, krylov_tol=1.0e-14)
    forward, _ = engine.step(initial, 0.004)
    backward, _ = engine.step(forward, -0.004)
    assert _phase_error(backward.state_vector(), initial.state_vector()) < 2.0e-10


def test_nnn_tdvp_preserves_norm_and_energy_for_small_steps():
    dims = (2, 2, 2, 2)
    factors = ([0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0])
    state = nnn_product_state(factors, max_bond=2)
    mpo = _random_three_site_mpo(dims, seed=19)
    engine = NNNLETTATDVPEngine(mpo, krylov_dim=48, krylov_tol=1.0e-13)
    norm0 = state.norm()
    energy0 = state.expectation_mpo(mpo)
    for _ in range(5):
        state, _ = engine.step(state, 0.001)
    assert abs(state.norm() - norm0) < 2.0e-9
    assert abs(state.expectation_mpo(mpo) - energy0) < 2.0e-7
