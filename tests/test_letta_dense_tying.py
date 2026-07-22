import numpy as np
from scipy.sparse import csr_matrix, identity

from pyqed.letta import CPTiedLETTA, DenseTiedLETTA


def test_dense_tied_letta_expands_cp_state_exactly():
    diagonal = np.linspace(-1.0, 2.0, 16)
    hamiltonian = csr_matrix(np.diag(diagonal))
    cp_state = CPTiedLETTA(
        hamiltonian,
        (2, 2, 2, 2),
        ((1, 3), (2,), (3,), ()),
        bond_dim=2,
        tie_ranks=(2, 3, 2, 1),
        seed=4,
    )

    dense_state = DenseTiedLETTA.from_cp(cp_state)

    np.testing.assert_allclose(
        dense_state.state_vector(normalize=True),
        cp_state.state_vector(normalize=True),
        atol=2.0e-14,
    )
    np.testing.assert_allclose(dense_state.energy, cp_state.energy, atol=2.0e-14)
    assert dense_state.parent_sets == cp_state.parent_sets
    assert dense_state.bond_dim == cp_state.bond_dim


def test_dense_local_projectors_reconstruct_the_state_and_updates_are_variational():
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(16, 16))
    hamiltonian = matrix + matrix.T
    state = DenseTiedLETTA(
        hamiltonian,
        (2, 2, 2, 2),
        ((1, 3), (2,), (3,), ()),
        bond_dim=2,
        seed=5,
    )

    vector = state.state_vector()
    for site, tensor in enumerate(state.tensors):
        np.testing.assert_allclose(
            state.local_projector(site) @ tensor.reshape(-1),
            vector,
            atol=2.0e-14,
        )

    energies = [state.energy]
    for site in range(len(state.dims)):
        update = state.optimize_site(site)
        energies.append(update.energy)
        assert update.energy <= update.energy_before + 1.0e-12
    assert np.all(np.diff(energies) <= 1.0e-12)


def test_real_dense_sweeps_match_uncached_updates_and_sparse_projectors():
    diagonal = np.linspace(-2.0, 1.0, 16)
    state = DenseTiedLETTA(
        csr_matrix(np.diag(diagonal)),
        (2, 2, 2, 2),
        ((1, 3), (2,), (3,), ()),
        bond_dim=2,
        seed=9,
    )
    assert all(np.issubdtype(tensor.dtype, np.floating) for tensor in state.tensors)
    np.testing.assert_allclose(
        state.local_projector(0, sparse=True).toarray(),
        state.local_projector(0),
        atol=0.0,
    )

    uncached = state.copy()
    for site in range(len(uncached.dims)):
        uncached.optimize_site(site)
    cached = state.copy().run(nsweeps=1, tol=0.0)

    np.testing.assert_allclose(cached.energy, uncached.energy, atol=2.0e-12)
    assert cached.fidelity(uncached.state_vector()) > 1.0 - 2.0e-12


def test_complex_bidirectional_cached_sweeps_match_uncached_updates():
    hamiltonian = np.diag(np.linspace(-1.0, 1.0, 8)).astype(complex)
    hamiltonian[1, 2] = 0.2j
    hamiltonian[2, 1] = -0.2j
    state = DenseTiedLETTA(
        hamiltonian,
        (2, 2, 2),
        ((1, 2), (2,), ()),
        bond_dim=2,
        seed=11,
    )
    assert all(np.issubdtype(tensor.dtype, np.complexfloating) for tensor in state.tensors)

    uncached = state.copy()
    for sites in (range(3), range(2, -1, -1)):
        for site in sites:
            uncached.optimize_site(site)
        uncached.balance_gauges()
    uncached.energy = uncached.expectation()
    cached = state.copy().run(nsweeps=2, tol=0.0)

    np.testing.assert_allclose(cached.energy, uncached.energy, atol=3.0e-12)
    assert cached.fidelity(uncached.state_vector()) > 1.0 - 3.0e-12


def test_dense_final_square_graph_has_expected_parameter_count():
    parents = (
        (1,),
        (2,),
        (3, 6),
        (4,),
        (5, 10),
        (6, 11),
        (7,),
        (8,),
        (9,),
        (10,),
        (11,),
        (),
    )
    state = DenseTiedLETTA(
        identity(1 << 12, format="csr"),
        (2,) * 12,
        parents,
        bond_dim=4,
        seed=2,
    )

    assert state.nparameters == 856
    assert max(tensor.size for tensor in state.tensors) == 128
    tensor_norms = np.asarray([np.linalg.norm(tensor) for tensor in state.tensors])
    assert tensor_norms.max() / tensor_norms.min() < 1.0001
