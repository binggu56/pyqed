import numpy as np

from pyqed.tn import LocalHamiltonian, LocalTerm
from pyqed.letta.mpo_frontier import MPOFrontier
from pyqed.tn import MPO
from tests.test_letta_frontier_tying import _states


def _identity_mpo(dims):
    return MPO(
        [np.eye(dim)[None, None, :, :] for dim in dims],
    )


def _engine(state, mpo):
    return MPOFrontier(
        state.dims,
        state.physical_groups,
        [tensor.shape for tensor in state.tensors],
        mpo.tensors,
    )


def test_local_term_mpo_and_cached_frontiers_match_explicit_projectors():
    state, dense = _states(seed=17)
    norm_engine = _engine(state, _identity_mpo(state.dims))
    hamiltonian_engine = _engine(state, state.hamiltonian.to_mpo())
    vector = dense.state_vector()

    np.testing.assert_allclose(
        norm_engine.scalar(state.tensors),
        np.vdot(vector, vector),
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        hamiltonian_engine.scalar(state.tensors),
        np.vdot(vector, dense.hamiltonian @ vector),
        atol=3.0e-13,
    )

    norm_left = norm_engine.build_left(state.tensors)
    norm_right = norm_engine.build_right(state.tensors)
    hamiltonian_left = hamiltonian_engine.build_left(state.tensors)
    hamiltonian_right = hamiltonian_engine.build_right(state.tensors)
    for site, tensor in enumerate(state.tensors):
        projector = dense.local_projector(site)
        reference_metric = projector.T.conj() @ projector
        reference_effective = projector.T.conj() @ dense.hamiltonian @ projector
        metric = norm_engine.hole_matrix(site, norm_left[site], norm_right[site + 1])
        effective = hamiltonian_engine.hole_matrix(
            site, hamiltonian_left[site], hamiltonian_right[site + 1]
        )
        np.testing.assert_allclose(metric, reference_metric, atol=5.0e-13)
        np.testing.assert_allclose(effective, reference_effective, atol=5.0e-13)

        probe = np.linspace(-0.3, 0.8, tensor.size).astype(complex)
        np.testing.assert_allclose(
            norm_engine.hole_action(site, norm_left[site], norm_right[site + 1], probe),
            reference_metric @ probe,
            atol=5.0e-13,
        )
        np.testing.assert_allclose(
            hamiltonian_engine.hole_action(
                site,
                hamiltonian_left[site],
                hamiltonian_right[site + 1],
                probe,
            ),
            reference_effective @ probe,
            atol=5.0e-13,
        )


def test_directional_messages_remain_valid_during_a_forward_sweep():
    state, _dense = _states(seed=21)
    engine = _engine(state, state.hamiltonian.to_mpo())
    fixed_right = engine.build_right(state.tensors)
    moving_left = engine.left_boundary()
    rng = np.random.default_rng(8)

    for site in range(len(state.dims)):
        state.tensors[site] += 1.0e-3 * rng.normal(size=state.tensors[site].shape)
        moving_left = engine.advance_left(moving_left, state.tensors, site)
        fresh_left = engine.build_left(state.tensors)[site + 1]
        np.testing.assert_allclose(moving_left, fresh_left, atol=2.0e-13)
        if site + 1 < len(state.dims):
            fresh_right = engine.build_right(state.tensors)[site + 1]
            np.testing.assert_allclose(fixed_right[site + 1], fresh_right, atol=2.0e-13)

def test_local_hamiltonian_mpo_matches_dense_for_mixed_supports():
    state, _dense = _states(seed=2)
    mpo = state.hamiltonian.to_mpo()
    np.testing.assert_allclose(
        mpo.to_dense(),
        state.hamiltonian.to_dense(),
        atol=3.0e-14,
    )
    assert max(mpo.bond_dims) < 1 + sum(
        term.operator.size for term in state.hamiltonian.terms
    )


def test_local_mpo_handles_complex_heterogeneous_and_zero_hamiltonians():
    rng = np.random.default_rng(31)
    dims = (2, 3, 2, 2)
    terms = []
    for sites in ((0,), (0, 2), (1, 2, 3), (0, 3)):
        size = int(np.prod([dims[site] for site in sites]))
        matrix = rng.normal(size=(size, size)) + 1.0j * rng.normal(size=(size, size))
        matrix = 0.5 * (matrix + matrix.T.conj())
        terms.append(LocalTerm(sites, matrix))
    hamiltonian = LocalHamiltonian(dims, terms, constant=-0.23)

    np.testing.assert_allclose(
        hamiltonian.to_mpo().to_dense(),
        hamiltonian.to_dense(),
        atol=3.0e-13,
    )
    zero = LocalHamiltonian(dims)
    np.testing.assert_allclose(zero.to_mpo().to_dense(), 0.0, atol=0.0)
