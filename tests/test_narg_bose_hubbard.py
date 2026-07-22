import numpy as np

from pyqed.narg import (
    BoseHubbardNARG,
    bose_hubbard_observables,
    exact_bose_hubbard,
    fixed_number_basis,
)
from pyqed.narg.bose_hubbard import bose_hubbard_hamiltonian, boson_annihilation
from pyqed.mps.mps import _mpo_to_dense_operator

from examples.mps.bose_hubbard_1d_mps_vs_letta import (
    bose_hubbard_mpo,
    fixed_basis_observables_from_product_state,
    fixed_number_weight_from_product_state,
    local_basis_transform,
    number_penalty_mpo,
    run_point,
    transform_mpo_local_basis,
    transform_product_state_local_basis,
)


def test_boson_annihilation_matrix_elements():
    b = boson_annihilation(4)

    expected = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, np.sqrt(2.0), 0.0],
            [0.0, 0.0, 0.0, np.sqrt(3.0)],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(b, expected)


def test_fixed_number_basis_respects_cutoff_and_ordering():
    basis = fixed_number_basis(nsites=3, nbosons=2, nmax=1)

    assert basis == [(0, 1, 1), (1, 0, 1), (1, 1, 0)]


def test_exact_two_site_single_boson_has_bonding_antibonding_roots():
    hamiltonian, basis = bose_hubbard_hamiltonian(
        nsites=2,
        nbosons=1,
        t=0.7,
        U=0.0,
        nmax=1,
    )

    assert basis == [(0, 1), (1, 0)]
    np.testing.assert_allclose(np.linalg.eigvalsh(hamiltonian.toarray()), [-0.7, 0.7])


def test_analytical_bose_hubbard_mpo_matches_fixed_number_hamiltonian():
    nsites = 4
    nbosons = 3
    nmax = 3
    mpo = bose_hubbard_mpo(nsites, nmax, hopping=0.8, onsite_u=1.7, mu=0.2)
    mpo_obj = type("MPOList", (), {"factors": mpo, "dims": (nmax + 1,) * nsites})()
    dense = _mpo_to_dense_operator(mpo_obj)
    basis = fixed_number_basis(nsites, nbosons, nmax)
    full_indices = np.asarray(
        [np.ravel_multi_index(state, (nmax + 1,) * nsites) for state in basis],
        dtype=int,
    )
    projected = dense[np.ix_(full_indices, full_indices)]
    expected, expected_basis = bose_hubbard_hamiltonian(
        nsites,
        nbosons,
        t=0.8,
        U=1.7,
        nmax=nmax,
        mu=0.2,
    )

    assert basis == expected_basis
    np.testing.assert_allclose(projected, expected.toarray(), atol=1.0e-12)


def test_bose_hubbard_gh_dvr_basis_rotation_matches_similarity_transform():
    nsites = 3
    nmax = 3
    mpo = bose_hubbard_mpo(nsites, nmax, hopping=0.8, onsite_u=1.7, mu=0.2)
    _, transform = local_basis_transform(nmax, "gh-dvr")
    dvr_mpo = transform_mpo_local_basis(mpo, transform)

    fock_dense = _mpo_to_dense_operator(type("MPOList", (), {"factors": mpo, "dims": (nmax + 1,) * nsites})())
    dvr_dense = _mpo_to_dense_operator(type("MPOList", (), {"factors": dvr_mpo, "dims": (nmax + 1,) * nsites})())
    product_transform = transform
    for _ in range(nsites - 1):
        product_transform = np.kron(product_transform, transform)

    np.testing.assert_allclose(
        dvr_dense,
        product_transform.conj().T @ fock_dense @ product_transform,
        atol=1.0e-12,
    )

    rng = np.random.default_rng(3)
    vector = rng.normal(size=(nmax + 1) ** nsites)
    dvr_vector = transform_product_state_local_basis(vector, nsites, transform, direction="fock-to-local")
    round_trip = transform_product_state_local_basis(dvr_vector, nsites, transform, direction="local-to-fock")
    np.testing.assert_allclose(round_trip, vector, atol=1.0e-12)


def test_bose_hubbard_number_penalty_mpo_matches_total_number_square():
    nsites = 3
    nbosons = 3
    nmax = 2
    penalty = number_penalty_mpo(nsites, nmax, nbosons)
    dense = _mpo_to_dense_operator(type("MPOList", (), {"factors": penalty, "dims": (nmax + 1,) * nsites})())
    expected_diag = [
        (sum(state) - nbosons) ** 2
        for state in np.ndindex((nmax + 1,) * nsites)
    ]

    np.testing.assert_allclose(dense, np.diag(expected_diag), atol=1.0e-12)

    basis = fixed_number_basis(nsites, nbosons, nmax)
    full_indices = np.asarray(
        [np.ravel_multi_index(state, (nmax + 1,) * nsites) for state in basis],
        dtype=int,
    )
    np.testing.assert_allclose(dense[np.ix_(full_indices, full_indices)], 0.0, atol=1.0e-12)


def test_bose_hubbard_narg_matches_exact_without_truncation():
    exact, exact_vectors, basis = exact_bose_hubbard(
        nsites=4,
        nbosons=4,
        t=1.0,
        U=1.0,
        nmax=4,
        nroots=3,
    )
    result = BoseHubbardNARG(
        nsites=4,
        nbosons=4,
        t=1.0,
        U=1.0,
        nmax=4,
        D=128,
    ).run(nroots=3)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    exact_obs = bose_hubbard_observables(exact_vectors[:, 0], basis)
    narg_obs = result.observables[0]
    np.testing.assert_allclose(
        narg_obs.one_body_density_matrix,
        exact_obs.one_body_density_matrix,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        narg_obs.average_number_variance,
        exact_obs.average_number_variance,
        atol=1e-10,
    )
    assert len(basis) == 35
    assert result.steps[-1].product_dim == 175
    assert result.steps[-1].kept == 3


def test_bose_hubbard_mps_vs_letta_smoke_matches_exact():
    result = run_point(
        nsites=4,
        nbosons=4,
        nmax=4,
        hopping=1.0,
        onsite_u=1.0,
        mu=0.0,
        bond_dim=8,
        sweeps=4,
        letta_sweeps=1,
        letta_expand_noise=1.0e-6,
        letta_seed=2,
        skip_ed=False,
        verbose=0,
        davidson_tol=1.0e-9,
        sweep_tol=1.0e-9,
    )

    np.testing.assert_allclose(result.dmrg_energy, result.ed_energy, atol=1.0e-10)
    assert result.letta_energy <= result.letta_initial + 1.0e-9
    np.testing.assert_allclose(result.letta_energy, result.ed_energy, atol=1.0e-8)
    np.testing.assert_allclose(result.letta_number_weight, 1.0, atol=1.0e-10)
    assert result.dmrg_converged
    assert 0.0 <= result.letta_observables.condensate_fraction <= 1.0


def test_bose_hubbard_gh_dvr_letta_seed_matches_dmrg_energy():
    result = run_point(
        nsites=4,
        nbosons=4,
        nmax=4,
        hopping=1.0,
        onsite_u=1.0,
        mu=0.0,
        bond_dim=8,
        sweeps=4,
        letta_sweeps=0,
        letta_expand_noise=0.0,
        letta_seed=2,
        skip_ed=False,
        verbose=0,
        davidson_tol=1.0e-9,
        sweep_tol=1.0e-9,
        letta_basis="gh-dvr",
    )

    np.testing.assert_allclose(result.letta_initial, result.dmrg_energy, atol=1.0e-10)
    np.testing.assert_allclose(result.letta_energy, result.dmrg_energy, atol=1.0e-10)
    np.testing.assert_allclose(result.letta_number_weight, 1.0, atol=1.0e-10)


def test_bose_hubbard_letta_observables_project_to_fixed_number_basis():
    basis = fixed_number_basis(nsites=3, nbosons=2, nmax=2)
    full = np.zeros((3, 3, 3))
    full[0, 1, 1] = 0.4
    full[1, 0, 1] = 0.5
    full[1, 1, 0] = 0.6

    direct = bose_hubbard_observables([full[state] for state in basis], basis)
    projected = fixed_basis_observables_from_product_state(full.reshape(-1), 3, 2, 2)

    np.testing.assert_allclose(projected.one_body_density_matrix, direct.one_body_density_matrix)
    np.testing.assert_allclose(projected.average_number_variance, direct.average_number_variance)
    np.testing.assert_allclose(fixed_number_weight_from_product_state(full.reshape(-1), 3, 2, 2), 1.0)


def test_bose_hubbard_narg_truncated_result_is_variational():
    exact, _, _ = exact_bose_hubbard(
        nsites=6,
        nbosons=6,
        t=1.0,
        U=1.0,
        nmax=4,
        nroots=1,
    )
    result = BoseHubbardNARG(
        nsites=6,
        nbosons=6,
        t=1.0,
        U=1.0,
        nmax=4,
        D=16,
    ).run(nroots=1)

    assert result.energies[0] >= exact[0] - 1e-10
    assert 0.0 <= result.observables[0].condensate_fraction <= 1.0
    assert result.steps[-2].kept <= 16
    assert result.steps[-1].kept == 1
