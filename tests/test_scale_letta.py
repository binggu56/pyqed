import numpy as np

from pyqed.narg import transverse_field_ising_hamiltonian
from pyqed.tn import (
    EightSiteScaleLETTA,
    contract_operator_schmidt,
    ising_tie_gate,
    operator_schmidt_factors,
    polar_isometry,
)


def test_operator_schmidt_tie_reconstructs_q1_and_q2_gates():
    identity = np.eye(4)
    left, right = operator_schmidt_factors(identity)
    assert left.shape[0] == 1
    np.testing.assert_allclose(
        contract_operator_schmidt(left, right), identity, atol=1.0e-12
    )

    gate = ising_tie_gate(0.23)
    left, right = operator_schmidt_factors(gate)
    assert left.shape[0] == 2
    np.testing.assert_allclose(
        contract_operator_schmidt(left, right), gate, atol=1.0e-12
    )


def test_q1_dense_layers_match_independent_ttn_contraction():
    state = EightSiteScaleLETTA(q=1)

    assert state.tie_dimension == 1
    np.testing.assert_allclose(
        state.state_vector(), state.ttn_state_vector(), atol=1.0e-12
    )
    np.testing.assert_allclose(state.norm(), 1.0, atol=1.0e-12)


def test_complete_two_cell_layer_is_canonical_after_polar_projection():
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(16, 4)) + 1j * rng.normal(size=(16, 4))
    canonical, residual = polar_isometry(raw, return_residual=True)

    assert residual < 1.0e-12
    np.testing.assert_allclose(
        canonical.conj().T @ canonical, np.eye(4), atol=1.0e-12
    )

    state = EightSiteScaleLETTA(q=2)
    layer = state.two_cell_layer()
    np.testing.assert_allclose(
        layer.conj().T @ layer, np.eye(4), atol=1.0e-12
    )


def test_critical_ising_q2_tie_improves_energy_and_matches_dense_checks():
    q1 = EightSiteScaleLETTA(q=1).fit_critical_ising(maxiter=100)
    q2 = EightSiteScaleLETTA(q=2).fit_critical_ising(maxiter=100)
    hamiltonian = transverse_field_ising_hamiltonian(
        8, periodic=True, sparse=False
    )

    assert q1.success
    assert q2.success
    assert q2.tie_dimension == 2
    assert q2.energy < q1.energy - 5.0e-2
    assert q2.energy >= q2.exact_energy - 1.0e-10
    np.testing.assert_allclose(q2.energy, q2.expectation(hamiltonian), atol=1.0e-12)
    np.testing.assert_allclose(q2.norm(), 1.0, atol=1.0e-12)


def test_shared_scaling_channel_contains_ising_sigma_and_energy_dimensions():
    state = EightSiteScaleLETTA(q=2).fit_critical_ising(maxiter=100)
    odd = state.scaling_dimensions(sector="odd")["dimensions"]
    even = state.scaling_dimensions(sector="even")["dimensions"]

    np.testing.assert_allclose(odd[0], 1.0 / 8.0, atol=1.0e-2)
    assert np.min(np.abs(even[1:] - 1.0)) < 8.0e-2
    full = state.scaling_dimensions()
    np.testing.assert_allclose(full["dimensions"][0], 0.0, atol=1.0e-10)
