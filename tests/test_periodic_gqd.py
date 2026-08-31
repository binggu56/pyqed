import numpy as np
import scipy.sparse as sp

from pyqed.ldr import (
    DiagonalElectronicContinuum,
    FeshbachEmbedding,
    MatrixElectronicContinuum,
    PeriodicSSHHolsteinHalfFilledScan,
    PeriodicSSHHolsteinGQD,
    PeriodicSSHHolsteinMomentumGQD,
)
from pyqed.ldr.periodic_scan import real_normal_modes


def test_periodic_ssh_holstein_bloch_hamiltonian_is_hermitian_and_periodic():
    model = PeriodicSSHHolsteinGQD()
    coordinates = np.linspace(-1.0, 1.0, 7)
    reference = model.electronic_hamiltonian(coordinates)
    translated = model.electronic_hamiltonian(
        coordinates,
        kpoint=model.kpoint + 2.0 * np.pi,
    )

    np.testing.assert_allclose(
        reference,
        reference.swapaxes(-1, -2).conj(),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(reference, translated, atol=2.0e-15)


def test_periodic_ssh_holstein_gqd_is_exactly_diabatization_equivalent():
    model = PeriodicSSHHolsteinGQD().build(domain=(-5.0, 5.0), npts=41)

    assert model.hamiltonian_error < 2.0e-12
    assert model.link_unitarity_error < 2.0e-15
    assert model.minimum_gap > 0.0


def test_periodic_ssh_holstein_gqd_dynamics_matches_exact_reference():
    model = PeriodicSSHHolsteinGQD().build(
        domain=(-7.0, 7.0),
        npts=61,
    ).run(
        dt=0.04,
        nsteps=200,
        nout=4,
    )

    assert model.success
    assert model.max_excited_population > 0.2
    assert model.max_state_error < 2.0e-10
    assert model.max_population_error < 2.0e-11
    assert model.max_norm_drift < 2.0e-11
    assert model.max_energy_drift < 2.0e-11


def test_periodic_ssh_holstein_gqd_is_phase_gauge_covariant():
    npts = 41
    reference = PeriodicSSHHolsteinGQD().build(
        domain=(-6.0, 6.0),
        npts=npts,
    )
    rng = np.random.default_rng(17)
    gauge = np.exp(1.0j * rng.uniform(-np.pi, np.pi, size=(npts, 2)))
    transformed = PeriodicSSHHolsteinGQD().build(
        domain=(-6.0, 6.0),
        npts=npts,
        gauge=gauge,
    )
    gauge_matrix = sp.block_diag(
        tuple(sp.diags(gauge[index]) for index in range(npts)),
        format="csr",
    )
    expected = (
        gauge_matrix.conj().T
        @ reference.solver.hamiltonian(sparse=True)
        @ gauge_matrix
    )
    delta = transformed.solver.hamiltonian(sparse=True) - expected
    assert np.max(np.abs(delta.data), initial=0.0) < 3.0e-12

    reference.run(dt=0.05, nsteps=80, nout=4)
    transformed.run(dt=0.05, nsteps=80, nout=4)
    np.testing.assert_allclose(
        transformed.adiabatic_populations,
        reference.adiabatic_populations,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        transformed.nuclear_density,
        reference.nuclear_density,
        atol=2.0e-11,
    )


def test_diagonal_electronic_continuum_matches_explicit_self_energy():
    energies = np.array([-0.7, 0.2, 1.1])
    couplings = np.array(
        [
            [0.12 + 0.03j, -0.08j, 0.17],
            [0.05, -0.11 + 0.02j, 0.09j],
        ]
    )
    weights = np.array([0.4, 0.7, 0.2])
    continuum = DiagonalElectronicContinuum(
        energies,
        couplings,
        weights=weights,
    )
    energy = 0.35
    eta = 0.04
    weighted = couplings * np.sqrt(weights)[None, :]
    expected = (
        weighted / (energy + 1.0j * eta - energies)[None, :]
    ) @ weighted.conj().T

    np.testing.assert_allclose(
        continuum.self_energy(energy, eta=eta),
        expected,
        atol=2.0e-16,
    )
    gamma = continuum.hybridization(energy, eta=eta)
    np.testing.assert_allclose(gamma, gamma.conj().T, atol=2.0e-16)
    assert np.min(np.linalg.eigvalsh(gamma)) > -2.0e-16
    np.testing.assert_allclose(
        continuum.memory_kernel([0.0])[0],
        weighted @ weighted.conj().T,
        atol=2.0e-16,
    )


def test_matrix_and_diagonal_electronic_continua_are_equivalent():
    continuum_hamiltonian = np.array(
        [
            [-0.4, 0.12 - 0.03j, 0.0],
            [0.12 + 0.03j, 0.3, -0.07],
            [0.0, -0.07, 0.9],
        ]
    )
    coupling = np.array(
        [
            [0.11, 0.03j, -0.04],
            [0.02 - 0.01j, 0.08, 0.06j],
        ]
    )
    matrix = MatrixElectronicContinuum(
        sp.csr_matrix(continuum_hamiltonian),
        sp.csr_matrix(coupling),
    )
    diagonal = matrix.diagonalize()

    for energy in (-0.2, 0.5, 1.2):
        np.testing.assert_allclose(
            matrix.self_energy(energy, eta=0.03),
            diagonal.self_energy(energy, eta=0.03),
            atol=2.0e-15,
        )


def test_feshbach_embedding_is_exact_active_block_of_complete_resolvent():
    model = PeriodicSSHHolsteinGQD().build(
        domain=(-4.0, 4.0),
        npts=17,
    )
    embedded = FeshbachEmbedding.from_ldr(model.solver, active_states=1)
    energy = 0.37
    eta = 0.025
    full = model.solver.hamiltonian(sparse=True).toarray()
    full_green = np.linalg.inv(
        (energy + 1.0j * eta) * np.eye(full.shape[0]) - full
    )
    projected = full_green[np.ix_(
        embedded.active_indices,
        embedded.active_indices,
    )]

    np.testing.assert_allclose(
        embedded.green_function(energy, eta=eta),
        projected,
        atol=2.0e-12,
    )
    assert embedded.nactive == model.dvr.size
    assert embedded.ncontinuum == model.dvr.size
    assert embedded.continuum_coupling_norm > 0.0
    assert 0.0 < embedded.minimum_projector_overlap <= 1.0
    assert embedded.maximum_projector_leakage > 0.0


def test_feshbach_embedding_spectrum_is_phase_gauge_invariant():
    npts = 15
    reference = PeriodicSSHHolsteinGQD().build(
        domain=(-4.0, 4.0),
        npts=npts,
    )
    rng = np.random.default_rng(81)
    gauge = np.exp(1.0j * rng.uniform(-np.pi, np.pi, size=(npts, 2)))
    transformed = PeriodicSSHHolsteinGQD().build(
        domain=(-4.0, 4.0),
        npts=npts,
        gauge=gauge,
    )
    energies = np.linspace(-0.8, 1.4, 37)
    embedded_reference = FeshbachEmbedding.from_ldr(
        reference.solver,
        active_states=1,
    ).run_spectrum(energies, eta=0.04)
    embedded_transformed = FeshbachEmbedding.from_ldr(
        transformed.solver,
        active_states=1,
    ).run_spectrum(energies, eta=0.04)

    np.testing.assert_allclose(
        embedded_transformed.spectral_density,
        embedded_reference.spectral_density,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        embedded_transformed.hybridization_trace,
        embedded_reference.hybridization_trace,
        atol=2.0e-12,
    )


def test_finite_q_hamiltonian_recovers_bloch_blocks_and_selection_rule():
    model = PeriodicSSHHolsteinMomentumGQD(ncells=4, q_index=1)
    hamiltonian = model.electronic_hamiltonian(0.0)
    reference = PeriodicSSHHolsteinGQD(
        hopping=model.hopping,
        dimerization=model.dimerization,
        ssh_coupling=model.ssh_coupling,
        sublattice_bias=model.sublattice_bias,
        holstein_coupling=model.holstein_coupling,
        phonon_frequency=model.phonon_frequency,
    )

    np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1.0e-14)
    for k_index, kpoint in enumerate(model.kpoints):
        section = slice(2 * k_index, 2 * k_index + 2)
        np.testing.assert_allclose(
            hamiltonian[section, section],
            reference.electronic_hamiltonian(0.0, kpoint=kpoint),
            atol=2.0e-15,
        )
        outside = hamiltonian[section].copy()
        outside[:, section] = 0.0
        np.testing.assert_allclose(outside, 0.0, atol=2.0e-15)

    model.build(domain=(-5.0, 5.0), npts=31)
    assert model.selection_rule_error < 1.0e-14
    assert np.max(model.coupling_block_norms) > 0.2


def test_finite_q_gqd_scattering_matches_exact_coupled_k_reference():
    model = PeriodicSSHHolsteinMomentumGQD(ncells=4, q_index=1).build(
        domain=(-5.0, 5.0),
        npts=41,
    ).run(
        dt=0.04,
        nsteps=100,
        nout=4,
    )

    assert model.success
    assert model.max_scattered_population > 0.5
    assert model.hamiltonian_error < 2.0e-12
    assert model.max_state_error < 2.0e-10
    assert model.max_momentum_population_error < 2.0e-11
    np.testing.assert_allclose(
        np.sum(model.momentum_populations, axis=1),
        1.0,
        atol=2.0e-11,
    )


def test_finite_q_gqd_is_multistate_phase_gauge_covariant():
    npts = 25
    reference = PeriodicSSHHolsteinMomentumGQD().build(
        domain=(-4.0, 4.0),
        npts=npts,
    )
    rng = np.random.default_rng(29)
    gauge = np.exp(
        1.0j * rng.uniform(-np.pi, np.pi, size=(npts, reference.nstates))
    )
    transformed = PeriodicSSHHolsteinMomentumGQD().build(
        domain=(-4.0, 4.0),
        npts=npts,
        gauge=gauge,
    )
    gauge_matrix = sp.block_diag(
        tuple(sp.diags(gauge[index]) for index in range(npts)),
        format="csr",
    )
    expected = (
        gauge_matrix.conj().T
        @ reference.solver.hamiltonian(sparse=True)
        @ gauge_matrix
    )
    delta = transformed.solver.hamiltonian(sparse=True) - expected
    assert np.max(np.abs(delta.data), initial=0.0) < 3.0e-12


def test_real_normal_modes_form_complete_orthonormal_cell_basis():
    modes = real_normal_modes(4)
    profiles = np.asarray([mode["profile"] for mode in modes])

    assert tuple(mode["name"] for mode in modes) == (
        "q0",
        "q1_cos",
        "q1_sin",
        "qpi",
    )
    np.testing.assert_allclose(
        profiles @ profiles.T / 4.0,
        np.eye(4),
        atol=3.0e-16,
    )


def test_half_filled_scan_resolves_all_determinants_and_mode_spectra():
    scan = PeriodicSSHHolsteinHalfFilledScan().scan(
        np.linspace(-2.0, 2.0, 17)
    )

    assert scan.success
    assert scan.nelectrons == 4
    assert scan.ndeterminants == 70
    assert scan.one_particle_energies.shape == (4, 17, 8)
    assert scan.many_body_energies.shape == (4, 17, 70)
    assert scan.single_excitation_energies.shape == (4, 17, 4, 4)
    np.testing.assert_allclose(
        scan.electronic_ground_energies,
        np.sum(scan.one_particle_energies[:, :, :4], axis=2),
        atol=3.0e-15,
    )
    np.testing.assert_allclose(
        scan.excitation_energies[:, :, 1],
        scan.fundamental_gaps,
        atol=2.0e-15,
    )
    assert scan.mode_orthogonality_error < 3.0e-16
    assert scan.cosine_sine_spectrum_error < 3.0e-15
