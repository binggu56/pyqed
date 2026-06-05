import numpy as np

from pyqed.narg.letta import LETTA
from pyqed.narg.holstein import (
    HolsteinChainAdiabaticNARG,
    HolsteinChainCouplingNARG,
    HolsteinChainNARG,
    HolsteinDimer,
    HolsteinDimerCoordinateNARG,
    HolsteinElectronicFirstNARG,
    SpinfulHolsteinAdiabaticElectronicNARG,
    SpinfulHolsteinElectronicFirstNARG,
    SpinfulHolsteinHubbardCouplingNARG,
    SpinfulHolsteinHubbardNARG,
    SpinfulHolsteinHubbardTwoSiteNARG,
    conditional_rank1_factor,
    holstein_chain_exact_energies,
    reconstruct_conditional_factor,
    sine_dvr_grid,
    sine_dvr_kinetic,
    spinful_holstein_hubbard_exact_energies,
    spinful_holstein_hubbard_exact_hamiltonian,
    spinful_hh_bipolaron_diagnostics,
    spinful_hh_pair_binding_energy,
)


def test_holstein_dimer_hamiltonian_and_lang_firsov_unitary():
    model = HolsteinDimer(t=0.2, omega=1.0, g=1.2, nphonon=5)

    hamiltonian = model.hamiltonian()
    unitary = model.lang_firsov_unitary()

    assert hamiltonian.shape == (model.dim, model.dim)
    np.testing.assert_allclose(hamiltonian, hamiltonian.T.conj(), atol=1e-12)
    np.testing.assert_allclose(unitary.T.conj() @ unitary, np.eye(model.dim), atol=1e-12)

    transformed = model.transformed_hamiltonian()
    np.testing.assert_allclose(
        np.linalg.eigvalsh(transformed),
        np.linalg.eigvalsh(hamiltonian),
        atol=1e-10,
    )


def test_lang_firsov_frame_reduces_rank_one_entanglement_and_energy_error():
    model = HolsteinDimer(t=0.2, omega=1.0, g=1.2, nphonon=6)
    report = model.report(ranks=(1, 2), nstates=3)

    exact_energy = report.exact_energies[0]
    bare_error = report.bare.energies[1] - exact_energy
    lf_error = report.lang_firsov.energies[1] - exact_energy

    assert report.lang_firsov.discarded_weights[1] < 0.02
    assert report.bare.discarded_weights[1] > 0.15
    assert report.lang_firsov.discarded_weights[1] < report.bare.discarded_weights[1]
    assert lf_error < bare_error


def test_conditional_rank_one_factor_reconstructs_one_target_state():
    model = HolsteinDimer(t=0.2, omega=1.0, g=1.2, nphonon=5)
    _, vectors = model.eigensystem(nstates=1)
    state = vectors[:, 0]

    a_tensor, b_tensor = conditional_rank1_factor(
        state, model.electron_dim, model.phonon_dim
    )
    reconstructed = reconstruct_conditional_factor(a_tensor, b_tensor).reshape(-1)

    assert a_tensor.shape == (model.electron_dim, model.phonon_dim, 1)
    assert b_tensor.shape == (1, model.phonon_dim)
    np.testing.assert_allclose(reconstructed, state, atol=1e-12)


def test_coordinate_narg_full_conditional_basis_matches_exact_grid():
    model = HolsteinDimerCoordinateNARG(
        t=0.2, omega=1.0, g=1.2, ngrid=7, xmax=4.0
    )

    exact_energies, _ = model.exact(nroots=4)
    result = model.run(nstates_per_point=2, nroots=4)

    np.testing.assert_allclose(result.energies, exact_energies, atol=1e-10)
    assert result.conditional_vectors.shape == (model.phonon_dim, 2, 2)
    assert result.vectors.shape == (model.phonon_dim * 2, 4)


def test_coordinate_narg_truncated_basis_is_variational_and_conditional():
    model = HolsteinDimerCoordinateNARG(
        t=0.2, omega=1.0, g=1.2, ngrid=7, xmax=4.0
    )

    exact_energies, _ = model.exact(nroots=1)
    result = model.run(nstates_per_point=1, nroots=1)
    wavefunction = model.reconstruct_wavefunction(
        result.vectors[:, 0], result.conditional_vectors
    )

    assert result.energies[0] >= exact_energies[0] - 1e-10
    assert result.conditional_vectors.shape == (model.phonon_dim, 2, 1)
    np.testing.assert_allclose(np.linalg.norm(wavefunction), 1.0, atol=1e-12)


def test_sine_dvr_helpers_use_dimensionless_box_range():
    grid = sine_dvr_grid(5, 6.0)
    kinetic = sine_dvr_kinetic(5, 6.0)

    np.testing.assert_allclose(grid, np.array([-4.0, -2.0, 0.0, 2.0, 4.0]))
    assert grid[0] > -6.0
    assert grid[-1] < 6.0
    np.testing.assert_allclose(kinetic, kinetic.T, atol=1e-12)

    expected = 0.5 * (np.pi * np.arange(1, 6) / 12.0) ** 2
    np.testing.assert_allclose(np.linalg.eigvalsh(kinetic), expected, atol=1e-12)


def test_chain_narg_matches_exact_when_no_states_are_truncated():
    exact = holstein_chain_exact_energies(
        3, t=0.2, omega=1.0, g=1.2, nphonon=3, nroots=5
    )
    result = HolsteinChainNARG(
        nsites=3, t=0.2, omega=1.0, g=1.2, nphonon=3, bond_dim=128
    ).run(nroots=5)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    assert result.sector_dims[-1] == (27, 81)


def test_chain_narg_truncates_to_bond_dimension_and_is_variational():
    exact = holstein_chain_exact_energies(
        4, t=0.2, omega=1.0, g=1.2, nphonon=3, nroots=1
    )
    result = HolsteinChainNARG(
        nsites=4, t=0.2, omega=1.0, g=1.2, nphonon=3, bond_dim=8
    ).run(nroots=1)

    assert result.energies[0] >= exact[0] - 1e-10
    assert all(dim0 <= 8 and dim1 <= 8 for dim0, dim1 in result.sector_dims)


def test_chain_narg_supports_local_dressed_site_truncation():
    exact = holstein_chain_exact_energies(
        3, t=0.2, omega=1.0, g=1.2, nphonon=4, nroots=1
    )
    result = HolsteinChainNARG(
        nsites=3,
        t=0.2,
        omega=1.0,
        g=1.2,
        nphonon=4,
        local_dim=2,
        bond_dim=16,
    ).run(nroots=1)

    assert result.energies[0] >= exact[0] - 1e-10
    assert result.sector_dims[0] == (2, 2)
    assert all(dim0 <= 16 and dim1 <= 16 for dim0, dim1 in result.sector_dims)


def test_adiabatic_chain_narg_exposes_conditional_basis_flow():
    params = dict(nsites=4, t=0.2, omega=1.0, g=1.2, nphonon=4, local_dim=3, bond_dim=16)
    nrg_result = HolsteinChainNARG(**params).run(nroots=3)
    narg_result = HolsteinChainAdiabaticNARG(**params).run(nroots=3)

    np.testing.assert_allclose(narg_result.energies, nrg_result.energies, atol=1e-12)
    assert len(narg_result.steps) == params["nsites"] - 1
    assert narg_result.steps[-1].conditional_dim <= narg_result.steps[-1].raw_dim
    assert narg_result.steps[-1].states_per_branch[0] <= params["bond_dim"]
    assert narg_result.steps[-1].states_per_branch[1] <= params["bond_dim"]


def test_coupling_conditioned_chain_narg_matches_exact_without_truncation():
    exact = holstein_chain_exact_energies(
        3, t=0.2, omega=1.0, g=1.2, nphonon=3, nroots=5
    )
    result = HolsteinChainCouplingNARG(
        nsites=3,
        t=0.2,
        omega=1.0,
        g=1.2,
        nphonon=3,
        bond_dim=128,
        states_per_branch=128,
    ).run(nroots=5)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    assert result.steps[-1].orthonormal_dim == result.steps[-1].raw_dim


def test_coupling_conditioned_chain_narg_uses_site_coupling_eigenvalues():
    model = HolsteinChainCouplingNARG(
        nsites=2, t=0.2, omega=1.0, g=1.2, nphonon=4, local_dim=3, bond_dim=8
    )
    result = model.run(nroots=1)
    singular_values = np.linalg.svd(model.dressed_site().c, compute_uv=False)
    expected = np.sort(np.concatenate((-singular_values, singular_values)))

    np.testing.assert_allclose(result.steps[0].site_eigenvalues, expected, atol=1e-12)
    np.testing.assert_allclose(
        2.0 * result.steps[0].site_annihilation_expectations.real,
        result.steps[0].site_eigenvalues,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.steps[0].site_annihilation_expectations.imag,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.steps[0].site_p_expectations,
        0.0,
        atol=1e-12,
    )
    assert result.steps[0].overlap_eigenvalues is not None
    assert np.sum(result.steps[0].overlap_eigenvalues > 1e-12) == result.steps[0].orthonormal_dim
    assert result.steps[0].orthonormal_dim <= result.steps[0].conditional_dim
    assert result.steps[0].orthonormal_dim <= result.steps[0].raw_dim


def test_electronic_first_holstein_narg_matches_exact_without_truncation():
    exact = holstein_chain_exact_energies(
        3, t=0.7, omega=1.0, g=0.8, nphonon=3, nroots=3
    )
    result = HolsteinElectronicFirstNARG(
        nsites=3,
        t=0.7,
        omega=1.0,
        g=0.8,
        nphonon=3,
        bond_dim=128,
    ).run(nroots=3)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    assert [step.mode for step in result.steps] == [0, 1, 2]
    assert result.steps[-1].product_dim == 81


def test_electronic_first_holstein_narg_is_variational_when_truncated():
    exact = holstein_chain_exact_energies(
        4, t=1.0, omega=1.0, g=1.0, nphonon=4, nroots=1
    )
    result = HolsteinElectronicFirstNARG(
        nsites=4,
        t=1.0,
        omega=1.0,
        g=1.0,
        nphonon=4,
        bond_dim=8,
    ).run(nroots=1)

    assert result.energies[0] >= exact[0] - 1e-10
    assert result.steps[-1].kept <= 8


def test_spinful_half_filled_electronic_first_holstein_matches_exact_without_truncation():
    exact = spinful_holstein_hubbard_exact_energies(
        2,
        t=0.7,
        omega=1.0,
        g=0.8,
        hubbard_u=0.0,
        nphonon=3,
        nroots=3,
    )
    result = SpinfulHolsteinElectronicFirstNARG(
        nsites=2,
        t=0.7,
        omega=1.0,
        g=0.8,
        hubbard_u=0.0,
        nphonon=3,
        bond_dim=128,
    ).run(nroots=3)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    assert result.target == (1, 1)
    assert result.electronic_dim == 4
    assert [step.mode for step in result.steps] == [0, 1]


def test_spinful_half_filled_electronic_first_holstein_is_variational_when_truncated():
    exact = spinful_holstein_hubbard_exact_energies(
        4,
        t=1.0,
        omega=1.0,
        g=1.0,
        hubbard_u=0.0,
        nphonon=2,
        nroots=1,
    )
    result = SpinfulHolsteinElectronicFirstNARG(
        nsites=4,
        t=1.0,
        omega=1.0,
        g=1.0,
        hubbard_u=0.0,
        nphonon=2,
        bond_dim=16,
    ).run(nroots=1)

    assert result.energies[0] >= exact[0] - 1e-10
    assert result.target == (2, 2)
    assert result.electronic_dim == 36
    assert result.steps[-1].kept <= 16


def test_spinful_adiabatic_electronic_holstein_matches_coordinate_exact_full_basis():
    model = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=2,
        t=0.7,
        omega=1.0,
        g=0.8,
        hubbard_u=0.0,
        ngrid=3,
        xmax=3.0,
        active_modes=(0, 1),
    )

    exact, _ = model.exact(nroots=3)
    result = model.run(nstates_per_point=model.electronic_dim, nroots=3)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    assert result.target == (1, 1)
    assert result.active_modes == (0, 1)
    assert result.conditional_vectors.shape == (9, 4, 4)


def test_spinful_adiabatic_electronic_letta_mpo_matches_dense_action():
    model = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=4,
        t=0.7,
        omega=1.0,
        g=0.8,
        hubbard_u=0.0,
        ngrid=3,
        xmax=3.0,
        active_modes=(0, 2),
    )

    dims = model.letta_product_dims()
    mpo = model.letta_mpo()
    hamiltonian = model.full_hamiltonian()
    letta = LETTA(None, dims, bond_dim=2, seed=23)
    rng = np.random.default_rng(24)
    vector = rng.normal(size=hamiltonian.shape[0])

    assert dims == (3, 3, model.electronic_dim)
    np.testing.assert_allclose(
        letta.apply_mpo(mpo, vector),
        hamiltonian @ vector,
        atol=1e-10,
    )

    electronic_first_dims = model.letta_product_dims(order="electronic-first")
    electronic_first_mpo = model.letta_mpo(order="electronic-first")
    electronic_first_letta = LETTA(None, electronic_first_dims, bond_dim=2, seed=25)
    electronic_first_vector = rng.normal(size=hamiltonian.shape[0])
    mode_first_vector = np.transpose(
        electronic_first_vector.reshape(electronic_first_dims),
        (1, 2, 0),
    ).reshape(-1)
    expected = hamiltonian @ mode_first_vector
    expected = np.transpose(
        expected.reshape(dims),
        (2, 0, 1),
    ).reshape(-1)

    assert electronic_first_dims == (model.electronic_dim, 3, 3)
    np.testing.assert_allclose(
        electronic_first_letta.apply_mpo(electronic_first_mpo, electronic_first_vector),
        expected,
        atol=1e-10,
    )


def test_spinful_adiabatic_electronic_sequential_exports_direct_letta_seed():
    model = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=2,
        t=0.7,
        omega=1.0,
        g=0.8,
        hubbard_u=0.0,
        ngrid=3,
        xmax=3.0,
        active_modes=(0, 1),
    )

    result = model.run_sequential(
        nstates_per_point=model.electronic_dim,
        bond_dim=model.electronic_dim * model.ngrid**2,
        initial_electronic_states=model.electronic_dim,
        nroots=2,
        store_narg_state=True,
    )
    state = result.narg_state(root=0)
    psi = state.state_vector()
    mpo = model.letta_mpo(order="electronic-first")
    letta_helper = LETTA(None, result.narg_dims, bond_dim=2, seed=26)
    applied = letta_helper.apply_mpo(mpo, psi)
    energy = np.vdot(psi, applied) / np.vdot(psi, psi)
    letta = result.to_letta(root=0)
    letta_state = letta.state_vector()

    assert result.narg_dims == (model.electronic_dim, 3, 3)
    assert len(result.narg_tensors) == 2
    np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-10)
    np.testing.assert_allclose(energy.real, result.energies[0], atol=1e-10)
    np.testing.assert_allclose(abs(np.vdot(psi, letta_state)), 1.0, atol=1e-10)
    np.testing.assert_allclose(
        letta.expectation_mpo(mpo),
        result.energies[0],
        atol=1e-10,
    )


def test_spinful_adiabatic_electronic_sequential_exports_compressed_letta_seed():
    model = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=4,
        t=1.0,
        omega=1.0,
        g=0.9,
        hubbard_u=0.0,
        ngrid=3,
        xmax=3.0,
        active_modes=(1, 2),
    )

    result = model.run_sequential(
        nstates_per_point=4,
        bond_dim=8,
        initial_electronic_states=8,
        nroots=1,
        store_narg_state=True,
        narg_electronic_basis="initial",
    )
    mpo = model.letta_mpo(
        order="electronic-first",
        electronic_hamiltonian=result.narg_electronic_hamiltonian,
        density_operators=result.narg_density_operators,
    )
    letta = result.to_letta(root=0)

    assert result.narg_dims == (8, 3, 3)
    assert result.narg_electronic_basis.shape == (model.electronic_dim, 8)
    np.testing.assert_allclose(
        letta.expectation_mpo(mpo),
        result.energies[0],
        atol=1e-10,
    )


def test_spinful_adiabatic_electronic_holstein_truncated_basis_is_variational():
    model = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=2,
        t=0.7,
        omega=1.0,
        g=0.8,
        hubbard_u=0.0,
        ngrid=5,
        xmax=3.0,
        active_modes=(0, 1),
    )

    exact, _ = model.exact(nroots=1)
    result = model.run(nstates_per_point=2, nroots=1)
    wavefunction = model.reconstruct_wavefunction(
        result.vectors[:, 0],
        result.conditional_vectors,
    )

    assert result.energies[0] >= exact[0] - 1e-10
    np.testing.assert_allclose(np.linalg.norm(wavefunction), 1.0, atol=1e-10)


def test_spinful_adiabatic_electronic_uses_sine_dvr_dimensionless_q():
    model = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=2,
        t=0.7,
        omega=1.5,
        g=0.8,
        hubbard_u=0.0,
        ngrid=5,
        xmax=6.0,
        active_modes=(0,),
    )

    q = model.grid()
    np.testing.assert_allclose(q, np.array([-4.0, -2.0, 0.0, 2.0, 4.0]))
    np.testing.assert_allclose(
        model.phonon_potential(),
        1.5 * (0.5 * q * q - 0.5),
        atol=1e-12,
    )
    shifted = model.electronic_hamiltonian_at(np.array([1.0])) - model.electronic_hamiltonian()
    np.testing.assert_allclose(
        shifted,
        np.sqrt(2.0) * 0.8 * model.electronic_density_operators()[0],
        atol=1e-12,
    )


def test_spinful_adiabatic_electronic_identity_mode_transform_matches_local_modes():
    local = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=2,
        t=0.7,
        omega=1.0,
        g=0.8,
        hubbard_u=0.0,
        ngrid=3,
        xmax=3.0,
        active_modes=(0, 1),
    )
    transformed = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=2,
        t=0.7,
        omega=1.0,
        g=0.8,
        hubbard_u=0.0,
        ngrid=3,
        xmax=3.0,
        mode_transform=np.eye(2),
    )

    local_result = local.run(nstates_per_point=2, nroots=2)
    transformed_result = transformed.run(nstates_per_point=2, nroots=2)

    np.testing.assert_allclose(transformed_result.energies, local_result.energies, atol=1e-10)
    np.testing.assert_allclose(transformed_result.mode_transform, np.eye(2), atol=1e-12)


def test_spinful_adiabatic_electronic_density_response_modes_are_ordered():
    model = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=4,
        t=1.0,
        omega=1.0,
        g=0.9,
        hubbard_u=0.0,
        ngrid=3,
        xmax=2.0,
    )

    modes = model.density_response_mode_transform(nlow=8)

    assert modes.transform.shape == (4, 4)
    np.testing.assert_allclose(modes.transform @ modes.transform.T, np.eye(4), atol=1e-12)
    assert np.all(modes.strengths[:-1] >= modes.strengths[1:] - 1e-12)
    assert modes.strengths[-1] < 1e-10


def test_spinful_adiabatic_electronic_sequential_one_mode_matches_joint_full_basis():
    model = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=2,
        t=0.7,
        omega=1.0,
        g=0.8,
        hubbard_u=0.0,
        ngrid=3,
        xmax=3.0,
        active_modes=(0,),
    )

    joint = model.run(nstates_per_point=model.electronic_dim, nroots=3)
    sequential = model.run_sequential(
        nstates_per_point=model.electronic_dim,
        bond_dim=model.electronic_dim * model.ngrid,
        initial_electronic_states=model.electronic_dim,
        nroots=3,
    )

    np.testing.assert_allclose(sequential.energies, joint.energies, atol=1e-10)
    assert sequential.steps[0].grid_dim == 3
    assert sequential.steps[0].hamiltonian_dim == model.electronic_dim * model.ngrid


def test_spinful_adiabatic_electronic_sequential_adds_modes_one_at_a_time():
    model = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=4,
        t=1.0,
        omega=1.0,
        g=0.9,
        hubbard_u=0.0,
        ngrid=3,
        xmax=3.0,
        active_modes=(1, 2),
    )

    result = model.run_sequential(
        nstates_per_point=4,
        bond_dim=16,
        initial_electronic_states=16,
        nroots=1,
    )

    assert [step.mode for step in result.steps] == [1, 2]
    assert [step.grid_dim for step in result.steps] == [3, 3]
    assert result.steps[0].hamiltonian_dim == 3 * 4
    assert result.steps[1].hamiltonian_dim == 3 * 4
    assert result.steps[-1].kept <= 16


def test_spinful_holstein_hubbard_exact_half_filled_hamiltonian_is_hermitian():
    hamiltonian = spinful_holstein_hubbard_exact_hamiltonian(
        2, t=0.2, omega=1.0, g=1.2, hubbard_u=3.0, nphonon=2
    )

    assert hamiltonian.shape == (16, 16)
    np.testing.assert_allclose(hamiltonian, hamiltonian.T.conj(), atol=1e-12)


def test_spinful_holstein_hubbard_bipolaron_diagnostics_obey_sum_rules():
    diagnostics = spinful_hh_bipolaron_diagnostics(
        2,
        t=0.2,
        omega=1.0,
        g=1.2,
        hubbard_u=3.0,
        nphonon=2,
        include_pair_binding=True,
    )
    pair_binding = spinful_hh_pair_binding_energy(
        2, t=0.2, omega=1.0, g=1.2, hubbard_u=3.0, nphonon=2
    )

    np.testing.assert_allclose(np.sum(diagnostics.density), 2.0, atol=1e-12)
    assert diagnostics.double_occupancy.shape == (2,)
    assert diagnostics.density_correlation.shape == (2, 2)
    np.testing.assert_allclose(
        diagnostics.density_correlation,
        diagnostics.density_correlation.T,
        atol=1e-12,
    )
    np.testing.assert_allclose(diagnostics.charge_structure_factor[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        diagnostics.staggered_charge_structure,
        diagnostics.charge_structure_factor[1],
        atol=1e-12,
    )
    np.testing.assert_allclose(diagnostics.pair_binding_energy, pair_binding, atol=1e-12)


def test_spinful_holstein_hubbard_narg_pair_binding_matches_exact_without_truncation():
    params = dict(
        nsites=2,
        t=0.2,
        omega=1.0,
        g=1.2,
        hubbard_u=3.0,
        nphonon=2,
        nup=0,
        ndown=0,
    )
    exact = spinful_hh_pair_binding_energy(**params)
    result = SpinfulHolsteinHubbardNARG(
        **params,
        bond_dim=128,
    ).pair_binding_energy()

    np.testing.assert_allclose(result, exact, atol=1e-10)


def test_spinful_holstein_hubbard_narg_matches_exact_without_truncation():
    exact = spinful_holstein_hubbard_exact_energies(
        2, t=0.2, omega=1.0, g=1.2, hubbard_u=3.0, nphonon=3, nroots=6
    )
    result = SpinfulHolsteinHubbardNARG(
        nsites=2,
        t=0.2,
        omega=1.0,
        g=1.2,
        hubbard_u=3.0,
        nphonon=3,
        bond_dim=256,
    ).run(nroots=6)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    assert result.target == (1, 1)
    assert result.sector_dims[-1][(1, 1)] == 36


def test_spinful_holstein_hubbard_two_site_narg_matches_exact_without_truncation():
    exact = spinful_holstein_hubbard_exact_energies(
        4, t=0.2, omega=1.0, g=1.2, hubbard_u=3.0, nphonon=2, nroots=4
    )
    result = SpinfulHolsteinHubbardTwoSiteNARG(
        nsites=4,
        t=0.2,
        omega=1.0,
        g=1.2,
        hubbard_u=3.0,
        nphonon=2,
        bond_dim=512,
        pair_dim=512,
    ).run(nroots=4)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    assert result.target == (2, 2)


def test_spinful_holstein_hubbard_dvr_phonon_basis_matches_exact_without_truncation():
    exact = spinful_holstein_hubbard_exact_energies(
        2, t=0.2, omega=1.0, g=1.2, hubbard_u=3.0, nphonon=3, nroots=6
    )
    result = SpinfulHolsteinHubbardNARG(
        nsites=2,
        t=0.2,
        omega=1.0,
        g=1.2,
        hubbard_u=3.0,
        nphonon=3,
        phonon_basis="dvr",
        bond_dim=256,
    ).run(nroots=6)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)


def test_spinful_holstein_hubbard_sine_dvr_grid_uses_requested_range():
    model = SpinfulHolsteinHubbardNARG(
        nsites=2,
        nphonon=5,
        phonon_basis="sine_dvr",
        dvr_xmin=-6.0,
        dvr_xmax=6.0,
    )
    grid, kinetic = model._sine_dvr_grid_and_kinetic()

    np.testing.assert_allclose(grid, np.array([-4.0, -2.0, 0.0, 2.0, 4.0]))
    np.testing.assert_allclose(kinetic, kinetic.T, atol=1e-12)


def test_spinful_holstein_hubbard_half_filled_truncation_is_variational():
    exact = spinful_holstein_hubbard_exact_energies(
        4, t=0.2, omega=1.0, g=1.2, hubbard_u=3.0, nphonon=2, nroots=1
    )
    result = SpinfulHolsteinHubbardNARG(
        nsites=4,
        t=0.2,
        omega=1.0,
        g=1.2,
        hubbard_u=3.0,
        nphonon=2,
        local_dim=2,
        bond_dim=24,
    ).run(nroots=1)

    assert result.target == (2, 2)
    assert result.energies[0] >= exact[0] - 1e-10
    assert result.sector_dims[-1][(2, 2)] <= 24


def test_spinful_holstein_hubbard_coupling_narg_matches_exact_without_truncation():
    exact = spinful_holstein_hubbard_exact_energies(
        2, t=0.2, omega=1.0, g=1.2, hubbard_u=3.0, nphonon=2, nroots=4
    )
    for mode in ("x_up", "x_down", "p_up", "p_down", "x_charge", "x_spin", "bilinear"):
        result = SpinfulHolsteinHubbardCouplingNARG(
            nsites=2,
            t=0.2,
            omega=1.0,
            g=1.2,
            hubbard_u=3.0,
            nphonon=2,
            bond_dim=128,
            states_per_branch=128,
            mode=mode,
        ).run(nroots=4)

        np.testing.assert_allclose(result.energies, exact, atol=1e-10)
        assert result.steps
        assert all(step.orthonormal_dim <= step.conditional_dim for step in result.steps)
        assert result.steps[0].site_branch_count == result.steps[0].local_site_dim == 8


def test_spinful_holstein_hubbard_electronic_branch_polaron_matches_exact_without_truncation():
    exact = spinful_holstein_hubbard_exact_energies(
        2, t=0.2, omega=1.0, g=1.2, hubbard_u=3.0, nphonon=2, nroots=4
    )
    result = SpinfulHolsteinHubbardCouplingNARG(
        nsites=2,
        t=0.2,
        omega=1.0,
        g=1.2,
        hubbard_u=3.0,
        nphonon=2,
        phonon_basis="polaron",
        bond_dim=128,
        states_per_branch=128,
        branch_rule="electronic",
    ).run(nroots=4)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    assert result.steps[0].site_branch_count == 4
    assert result.steps[0].local_site_dim == 8
    assert result.steps[0].conditional_dim == result.steps[0].raw_dim
    assert result.steps[0].orthonormal_dim == result.steps[0].raw_dim


def test_spinful_holstein_hubbard_electronic_coupling_branch_matches_exact_without_truncation():
    exact = spinful_holstein_hubbard_exact_energies(
        2, t=0.2, omega=1.0, g=1.2, hubbard_u=3.0, nphonon=2, nroots=4
    )
    result = SpinfulHolsteinHubbardCouplingNARG(
        nsites=2,
        t=0.2,
        omega=1.0,
        g=1.2,
        hubbard_u=3.0,
        nphonon=2,
        phonon_basis="polaron",
        bond_dim=128,
        states_per_branch=128,
        branch_rule="electronic_coupling",
        mode="x_charge",
    ).run(nroots=4)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    assert result.steps[0].site_branch_count == 4
    assert result.steps[0].local_site_dim == 8
    assert result.steps[0].orthonormal_dim == result.steps[0].raw_dim


def test_spinful_holstein_hubbard_electronic_virtual_branch_matches_exact_without_truncation():
    exact = spinful_holstein_hubbard_exact_energies(
        2, t=0.2, omega=1.0, g=1.2, hubbard_u=3.0, nphonon=2, nroots=4
    )
    result = SpinfulHolsteinHubbardCouplingNARG(
        nsites=2,
        t=0.2,
        omega=1.0,
        g=1.2,
        hubbard_u=3.0,
        nphonon=2,
        phonon_basis="polaron",
        bond_dim=128,
        states_per_branch=128,
        branch_rule="electronic_virtual",
    ).run(nroots=4)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    assert result.steps[0].site_branch_count == 4
    assert result.steps[0].local_site_dim == 8
    assert result.steps[0].conditional_dim == result.steps[0].raw_dim
    assert result.steps[0].orthonormal_dim == result.steps[0].raw_dim


def test_spinful_holstein_hubbard_electronic_resolvent_branch_matches_exact_without_truncation():
    exact = spinful_holstein_hubbard_exact_energies(
        2, t=0.2, omega=1.0, g=1.2, hubbard_u=3.0, nphonon=2, nroots=4
    )
    result = SpinfulHolsteinHubbardCouplingNARG(
        nsites=2,
        t=0.2,
        omega=1.0,
        g=1.2,
        hubbard_u=3.0,
        nphonon=2,
        phonon_basis="polaron",
        bond_dim=128,
        states_per_branch=128,
        branch_rule="electronic_resolvent",
    ).run(nroots=4)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    assert result.steps[0].site_branch_count == 4
    assert result.steps[0].local_site_dim == 8
    assert result.steps[0].conditional_dim == result.steps[0].raw_dim
    assert result.steps[0].orthonormal_dim == result.steps[0].raw_dim


def test_spinful_holstein_hubbard_coupling_modes_are_variational():
    exact = spinful_holstein_hubbard_exact_energies(
        4, t=0.2, omega=1.0, g=1.2, hubbard_u=3.0, nphonon=2, nroots=1
    )
    for mode in ("x_up", "x_down", "p_up", "p_down", "x_charge", "x_spin", "bilinear"):
        result = SpinfulHolsteinHubbardCouplingNARG(
            nsites=4,
            t=0.2,
            omega=1.0,
            g=1.2,
            hubbard_u=3.0,
            nphonon=2,
            local_dim=2,
            bond_dim=16,
            states_per_branch=4,
            mode=mode,
        ).run(nroots=1)

        assert result.energies[0] >= exact[0] - 1e-10
        assert result.target == (2, 2)


def test_spinful_holstein_hubbard_coupling_iterative_solver_matches_dense_solver():
    params = dict(
        nsites=4,
        t=0.2,
        omega=1.0,
        g=1.2,
        hubbard_u=3.0,
        nphonon=2,
        local_dim=2,
        bond_dim=16,
        states_per_branch=4,
        mode="x_charge",
    )
    dense = SpinfulHolsteinHubbardCouplingNARG(
        **params,
        conditional_solver="dense",
    ).run(nroots=2)
    iterative = SpinfulHolsteinHubbardCouplingNARG(
        **params,
        conditional_solver="iterative",
        conditional_solver_tol=1e-12,
    ).run(nroots=2)

    np.testing.assert_allclose(iterative.energies, dense.energies, atol=1e-8)


def test_spinful_holstein_hubbard_coupling_dvr_phonon_basis_matches_exact_without_truncation():
    exact = spinful_holstein_hubbard_exact_energies(
        2, t=0.2, omega=1.0, g=1.2, hubbard_u=3.0, nphonon=2, nroots=4
    )
    result = SpinfulHolsteinHubbardCouplingNARG(
        nsites=2,
        t=0.2,
        omega=1.0,
        g=1.2,
        hubbard_u=3.0,
        nphonon=2,
        phonon_basis="dvr",
        bond_dim=128,
        states_per_branch=128,
        mode="x_charge",
    ).run(nroots=4)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    assert result.steps[0].site_branch_count == result.steps[0].local_site_dim == 8


def test_spinful_holstein_hubbard_coupling_sine_dvr_matches_nrg_without_truncation():
    params = dict(
        nsites=2,
        t=0.2,
        omega=1.0,
        g=1.2,
        hubbard_u=3.0,
        nphonon=5,
        phonon_basis="sine_dvr",
        dvr_xmin=-6.0,
        dvr_xmax=6.0,
        bond_dim=512,
    )
    nrg = SpinfulHolsteinHubbardNARG(**params).run(nroots=4)
    narg = SpinfulHolsteinHubbardCouplingNARG(
        **params,
        states_per_branch=512,
        mode="x_charge",
    ).run(nroots=4)

    np.testing.assert_allclose(narg.energies, nrg.energies, atol=1e-10)
    assert narg.steps[0].site_branch_count == narg.steps[0].local_site_dim == 20
