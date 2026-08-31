import numpy as np
import pytest

from pyqed.qchem import (
    DMRGDensityProvider,
    DenseCIDensityProvider,
    MCTDHF,
    Molecule,
    RDM12DensityProvider,
    TDCASCI,
)
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.casci import transform_spatial_eri_to_mo
from pyqed.qchem.mcscf.direct_ci import CASCI


def _h2_mf(basis="sto-3g"):
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis=basis)
    mol.build()
    return RHF(mol).run()


def _lih_mf(basis="sto-3g"):
    mol = Molecule(atom="Li 0 0 0; H 0 0 3.0", unit="bohr", basis=basis)
    mol.build()
    return RHF(mol).run()


def _h2_casci(nstates=2):
    return CASCI(_h2_mf(), ncas=2, nelecas=2, verbose=0).run(
        nstates=nstates,
        method="direct_ci",
    )


def test_mctdhf_is_exported():
    from pyqed.qchem.mctdhf import DenseCIDensityProvider as DirectDensityProvider
    from pyqed.qchem.mctdhf import DMRGDensityProvider as DirectDMRGDensityProvider
    from pyqed.qchem.mctdhf import MCTDHF as DirectMCTDHF
    from pyqed.qchem.mctdhf import RDM12DensityProvider as DirectRDM12DensityProvider

    assert MCTDHF is DirectMCTDHF
    assert DenseCIDensityProvider is DirectDensityProvider
    assert DMRGDensityProvider is DirectDMRGDensityProvider
    assert RDM12DensityProvider is DirectRDM12DensityProvider


def test_mctdhf_validates_active_orbital_count():
    mf = _h2_mf()

    with pytest.raises(ValueError, match="AO basis dimension"):
        MCTDHF(mf, norb=mf.nmo + 1, nelec=2)

    with pytest.raises(ValueError, match="max"):
        MCTDHF(mf, norb=0, nelec=2)

    with pytest.raises(ValueError, match="orbital_integrator"):
        MCTDHF(mf, norb=2, nelec=2, orbital_integrator="euler")


def test_mctdhf_selects_explicit_initial_active_orbitals():
    mf = _h2_mf(basis="6-31g")
    td = MCTDHF(mf, norb=2, nelec=2, active_orbitals=(1, 3))

    assert td.active_orbitals == (1, 3)
    np.testing.assert_allclose(td.orbitals0, mf.mo_coeff[:, (1, 3)], atol=1.0e-10)


def test_mctdhf_selects_integer_active_orbital_window():
    mf = _h2_mf(basis="6-31g")
    td = MCTDHF(mf, norb=2, nelec=2, active_orbitals=1)

    assert td.active_orbitals == (1, 2)
    np.testing.assert_allclose(td.orbitals0, mf.mo_coeff[:, 1:3], atol=1.0e-10)


def test_mctdhf_validates_active_orbital_selection():
    mf = _h2_mf(basis="6-31g")

    with pytest.raises(ValueError, match="exactly"):
        MCTDHF(mf, norb=2, nelec=2, active_orbitals=(1,))

    with pytest.raises(ValueError, match="duplicate"):
        MCTDHF(mf, norb=2, nelec=2, active_orbitals=(1, 1))

    with pytest.raises(ValueError, match="out-of-range"):
        MCTDHF(mf, norb=2, nelec=2, active_orbitals=(1, mf.nmo))


def test_mctdhf_frozen_mode_preserves_norm_energy_and_phase():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2, orbital_mode="frozen")
    h0 = td.hamiltonian_matrix()
    evals, vecs = np.linalg.eigh(h0)

    traj = td.run(dt=0.05, nsteps=6, ci0=0)

    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(traj.electronic_energies, evals[0], atol=1.0e-10)
    np.testing.assert_allclose(traj.energies, evals[0] + mf.energy_nuc(), atol=1.0e-10)
    phase = np.exp(-1j * evals[0] * traj.times)
    expected = phase[:, None] * vecs[:, [0]].T
    np.testing.assert_allclose(np.abs(np.sum(traj.ci.conj() * expected, axis=1)), 1.0, atol=1.0e-10)


def test_mctdhf_from_casci_uses_active_orbitals_and_initial_ci():
    mc = _h2_casci(nstates=2)
    td = MCTDHF.from_casci(mc, state_id=1, orbital_mode="frozen")

    np.testing.assert_allclose(td.orbitals0, mc.mo_cas, atol=1.0e-10)
    np.testing.assert_allclose(td.initial_ci, mc.ci[1], atol=1.0e-12)
    traj = td.run(dt=0.05, nsteps=0)

    np.testing.assert_allclose(traj.ci[0], mc.ci[1], atol=1.0e-12)
    np.testing.assert_allclose(traj.energies[0], td.energy(mc.ci[1], td.orbitals0).real, atol=1.0e-10)


def test_mctdhf_frozen_from_casci_matches_tdcasci_no_core():
    mc = _h2_casci(nstates=2)
    rt_ref = TDCASCI(mc).run(dt=0.04, nsteps=4, ci0=1)
    td = MCTDHF.from_casci(mc, state_id=1, orbital_mode="frozen")

    traj = td.run(dt=0.04, nsteps=4)

    np.testing.assert_allclose(traj.ci, rt_ref.ci, atol=1.0e-10)
    np.testing.assert_allclose(traj.energies, rt_ref.energies, atol=1.0e-10)
    np.testing.assert_allclose(traj.electronic_energies, rt_ref.active_energies, atol=1.0e-10)
    np.testing.assert_allclose(traj.autocorrelation, rt_ref.autocorrelation, atol=1.0e-10)


def test_mctdhf_from_casci_supports_frozen_core_energy():
    mc = CASCI(_lih_mf(), ncas=2, nelecas=2, verbose=0).run(
        nstates=1,
        method="direct_ci",
    )

    td = MCTDHF.from_casci(mc, orbital_mode="frozen")
    traj = td.run(dt=0.04, nsteps=2)

    assert td.core_orbitals is not None
    assert td.core_density_ao is not None
    np.testing.assert_allclose(td.energy(mc.ci[0], td.orbitals0).real, mc.e_tot[0], atol=1.0e-10)
    np.testing.assert_allclose(traj.energies, mc.e_tot[0], atol=1.0e-10)
    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)


def test_mctdhf_frozen_from_casci_matches_tdcasci_with_frozen_core():
    mc = CASCI(_lih_mf(), ncas=2, nelecas=2, verbose=0).run(
        nstates=1,
        method="direct_ci",
    )
    rt_ref = TDCASCI(mc).run(dt=0.03, nsteps=3, ci0=0)
    td = MCTDHF.from_casci(mc, orbital_mode="frozen")

    traj = td.run(dt=0.03, nsteps=3)

    np.testing.assert_allclose(traj.ci, rt_ref.ci, atol=1.0e-10)
    np.testing.assert_allclose(traj.energies, rt_ref.energies, atol=1.0e-10)
    np.testing.assert_allclose(traj.electronic_energies, rt_ref.active_energies, atol=1.0e-10)
    np.testing.assert_allclose(traj.autocorrelation, rt_ref.autocorrelation, atol=1.0e-10)


def test_mctdhf_frozen_core_moving_orbitals_track_core_overlap():
    mc = CASCI(_lih_mf(), ncas=2, nelecas=2, verbose=0).run(
        nstates=1,
        method="direct_ci",
    )
    td = MCTDHF.from_casci(mc, orbital_mode="mctdhf")

    traj = td.run(dt=0.01, nsteps=2)

    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)
    assert np.max(traj.orbital_errors) < 1.0e-10
    assert np.max(traj.core_overlap_errors) < 1.0e-10


def test_mctdhf_frozen_core_orbital_rhs_is_core_orthogonal():
    mc = CASCI(_lih_mf(), ncas=2, nelecas=2, verbose=0).run(
        nstates=1,
        method="direct_ci",
    )
    td = MCTDHF.from_casci(mc)

    rhs = td.orbital_rhs(td.orbitals0, td.initial_ci)
    core_overlap = td.core_orbitals.conj().T @ td.overlap @ rhs

    np.testing.assert_allclose(core_overlap, 0.0, atol=1.0e-10)


def test_mctdhf_rejects_overlapping_core_and_active_orbitals():
    mf = _h2_mf()

    with pytest.raises(ValueError, match="S-orthogonal"):
        MCTDHF(
            mf,
            norb=1,
            nelec=1,
            spin=1,
            mo_coeff=mf.mo_coeff[:, [0]],
            core_orbitals=mf.mo_coeff[:, [0]],
        )


def test_mctdhf_project_out_subspace_handles_nonorthogonal_basis():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2)
    basis = np.array(
        [
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=complex,
    )
    vectors = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=complex,
    )

    projected = td.project_out_subspace(vectors, basis)

    np.testing.assert_allclose(basis.conj().T @ projected, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(projected[2], vectors[2], atol=1.0e-12)


def test_mctdhf_run_validates_time_grid():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2)

    with pytest.raises(ValueError, match="dt"):
        td.run(dt=np.nan, nsteps=1)

    with pytest.raises(ValueError, match="t0"):
        td.run(dt=0.1, nsteps=1, t0=np.inf)

    with pytest.raises(ValueError, match="nsteps"):
        td.run(dt=0.1, nsteps=-1)

    with pytest.raises(ValueError, match="nsteps"):
        td.run(dt=0.1, nsteps=1.5)

    with pytest.raises(ValueError, match="save_every"):
        td.run(dt=0.1, nsteps=1, save_every=0)

    with pytest.raises(ValueError, match="save_every"):
        td.run(dt=0.1, nsteps=1, save_every=1.5)


def test_mctdhf_zero_step_trajectory_stores_initial_state_only(monkeypatch):
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2)

    def fail_step(*args, **kwargs):
        raise AssertionError("step should not be called for nsteps=0")

    monkeypatch.setattr(td, "step", fail_step)
    traj = td.run(dt=0.1, nsteps=0, store_orbitals=False)

    np.testing.assert_allclose(traj.times, [0.0])
    assert traj.orbitals is None
    assert traj.ci.shape == (1, td.ndet)
    np.testing.assert_allclose(traj.norms, [1.0], atol=1.0e-12)
    assert traj.dipoles.shape == (1, 3)
    assert traj.fields.shape == (1, 3)


def test_mctdhf_trajectory_final_state_restarts_dense_run():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2, orbital_mode="frozen")

    first = td.run(dt=0.05, nsteps=2)
    restarted = td.run(
        dt=0.05,
        nsteps=2,
        ci0=first.final_ci,
        orbitals0=first.final_orbitals,
        t0=first.final_time,
    )
    full = td.run(dt=0.05, nsteps=4)

    np.testing.assert_allclose(restarted.final_time, full.final_time)
    np.testing.assert_allclose(restarted.final_ci, full.final_ci, atol=1.0e-10)
    np.testing.assert_allclose(restarted.final_orbitals, full.final_orbitals, atol=1.0e-12)


def test_mctdhf_run_save_every_downsamples_and_keeps_final_state():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2, orbital_mode="frozen")

    sparse = td.run(dt=0.05, nsteps=5, save_every=2)
    full = td.run(dt=0.05, nsteps=5)

    np.testing.assert_allclose(sparse.times, [0.0, 0.1, 0.2, 0.25])
    assert sparse.ci.shape == (4, td.ndet)
    assert sparse.orbitals.shape == (4, mf.nao, 2)
    np.testing.assert_allclose(sparse.final_time, full.final_time)
    np.testing.assert_allclose(sparse.final_ci, full.final_ci, atol=1.0e-10)
    np.testing.assert_allclose(sparse.final_orbitals, full.final_orbitals, atol=1.0e-12)


def test_mctdhf_trajectory_final_orbitals_requires_storage():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2, orbital_mode="frozen")
    traj = td.run(dt=0.05, nsteps=0, store_orbitals=False)

    with pytest.raises(ValueError, match="not stored"):
        _ = traj.final_orbitals


def test_mctdhf_run_can_skip_ci_storage_but_keep_diagnostics():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2, orbital_mode="frozen")
    traj = td.run(dt=0.05, nsteps=3, store_ci=False)

    assert traj.ci is None
    assert traj.orbitals.shape == (4, mf.nao, 2)
    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)
    assert traj.energies.shape == (4,)
    assert traj.natural_occupations.shape == (4, 2)
    with pytest.raises(ValueError, match="CI states were not stored"):
        _ = traj.final_ci


def test_mctdhf_direct_sigma_matches_dense_hamiltonian():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2)
    h_dense = td.hamiltonian_matrix()
    ci = np.array([0.2 + 0.1j, -0.3j, 0.4, 0.5 - 0.2j])
    ci = ci / np.linalg.norm(ci)

    sigma_dense = h_dense @ ci
    sigma_direct = td.sigma_vector(ci, backend="krylov")

    np.testing.assert_allclose(sigma_direct, sigma_dense, atol=1.0e-10)


def test_mctdhf_krylov_propagation_matches_dense_step():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2)
    ci = td.ci_vector(0)

    dense = td.propagate_ci(ci, dt=0.07, backend="dense")
    krylov = td.propagate_ci(ci, dt=0.07, backend="krylov")

    np.testing.assert_allclose(krylov, dense, atol=1.0e-10)


def test_mctdhf_impulsive_kick_changes_ci_and_preserves_norm():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2, orbital_mode="frozen")
    ci = td.ci_vector(0)

    kicked = td.kick_ci(ci, strength=1.0e-2, axis="z")

    np.testing.assert_allclose(np.linalg.norm(kicked), 1.0, atol=1.0e-12)
    assert abs(np.vdot(ci, kicked)) < 1.0 - 1.0e-8


def test_mctdhf_kick_run_and_dipole_spectrum_helper():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2, orbital_mode="frozen")

    traj = td.run(
        dt=0.05,
        nsteps=6,
        kick={"strength": 1.0e-3, "axis": "z"},
    )

    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)
    omega, power = traj.dipole_spectrum(axis="z")
    assert omega.shape == power.shape
    assert np.all(power >= 0.0)


def test_mctdhf_autocorrelation_and_spectrum_helper():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2, orbital_mode="frozen")
    ci = td.ci_vector(0)
    h = td.hamiltonian_matrix()
    energy = np.vdot(ci, h @ ci).real

    traj = td.run(dt=0.05, nsteps=5, ci0=ci)

    expected = np.exp(-1j * energy * traj.times)
    np.testing.assert_allclose(traj.autocorrelation, expected, atol=1.0e-10)
    omega, power = traj.autocorrelation_spectrum()
    assert omega.shape == power.shape
    assert np.all(power >= 0.0)


def test_mctdhf_natural_occupations_track_active_rdm1():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2, orbital_mode="frozen")
    traj = td.run(dt=0.05, nsteps=2)

    assert traj.natural_occupations.shape == (3, 2)
    np.testing.assert_allclose(np.sum(traj.natural_occupations, axis=1), 2.0, atol=1.0e-12)
    np.testing.assert_allclose(traj.natural_occupation_trace_errors, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        traj.natural_occupations[0],
        td.natural_occupations(traj.ci[0]),
        atol=1.0e-12,
    )


def test_mctdhf_natural_occupation_trace_error_flags_bad_provider_rdm1():
    class BadTraceProvider:
        def bind(self, driver):
            self.delegate = DenseCIDensityProvider(driver)

        def make_rdm1(self, ci):
            return np.diag([1.2, 1.0])

        def contract_rdm2_eri_full(self, ci, eri_full):
            return self.delegate.contract_rdm2_eri_full(ci, eri_full)

    mf = _h2_mf()
    provider = BadTraceProvider()
    td = MCTDHF(mf, norb=2, nelec=2, orbital_mode="frozen", density_provider=provider)

    traj = td.run(dt=0.05, nsteps=0)

    np.testing.assert_allclose(traj.natural_occupations[0], [1.2, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(traj.natural_occupation_trace_errors[0], 0.2, atol=1.0e-12)


def test_mctdhf_krylov_eigenstate_matches_dense():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2)

    e_dense, v_dense = td.ci_eigenstates(1, backend="dense")
    e_krylov, v_krylov = td.ci_eigenstates(1, backend="krylov")

    np.testing.assert_allclose(e_krylov, e_dense, atol=1.0e-10)
    np.testing.assert_allclose(abs(np.vdot(v_dense[:, 0], v_krylov[:, 0])), 1.0, atol=1.0e-10)


def test_mctdhf_krylov_ci_vector_does_not_build_dense_hamiltonian(monkeypatch):
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2, ci_backend="krylov")

    def fail_dense(*args, **kwargs):
        raise AssertionError("dense Hamiltonian should not be built")

    monkeypatch.setattr(td, "hamiltonian_matrix", fail_dense)
    ci = td.ci_vector(0)

    np.testing.assert_allclose(np.linalg.norm(ci), 1.0, atol=1.0e-12)
    sigma = td.sigma_vector(ci, backend="krylov")
    energy = np.vdot(ci, sigma)
    residual = np.linalg.norm(sigma - energy * ci)
    assert residual < 1.0e-8


def test_mctdhf_active_rotation_preserves_represented_energy():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2)
    ci = td.ci_vector(0)
    energy0 = td.energy(ci, td.orbitals0)
    theta = 0.23
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    orbitals_rot = td.orbitals0 @ rot
    ci_rot = td.rotate_ci_for_orbital_rotation(ci, rot)
    energy_rot = td.energy(ci_rot, orbitals_rot)

    np.testing.assert_allclose(energy_rot, energy0, atol=1.0e-10)


def test_mctdhf_state_overlap_uses_active_orbital_frames():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2)
    ci = td.ci_vector(0)
    theta = 0.23
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    orbitals_rot = td.orbitals0 @ rot
    ci_rot = td.rotate_ci_for_orbital_rotation(ci, rot)

    assert abs(np.vdot(ci, ci_rot)) < 0.99
    np.testing.assert_allclose(
        td.state_overlap(
            ci,
            ci_rot,
            bra_orbitals=td.orbitals0,
            ket_orbitals=orbitals_rot,
        ),
        1.0,
        atol=1.0e-10,
    )


def test_mctdhf_run_autocorrelation_uses_moving_orbital_overlap(monkeypatch):
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2)
    ci = td.ci_vector(0)
    theta = 0.23
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    ci_rot = td.rotate_ci_for_orbital_rotation(ci, rot)
    orbitals_rot = td.orbitals0 @ rot

    def rotate_once(ci, orbitals, time=0.0, dt=0.0, field=None, h1_ao=None):
        return ci_rot, orbitals_rot

    monkeypatch.setattr(td, "step", rotate_once)
    traj = td.run(dt=0.05, nsteps=1, ci0=ci)

    assert abs(np.vdot(ci, ci_rot)) < 0.99
    np.testing.assert_allclose(traj.autocorrelation, [1.0, 1.0], atol=1.0e-10)


def test_mctdhf_gauge_alignment_removes_active_rotation():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2)
    ci = td.ci_vector(0)
    theta = -0.17
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    orbitals_rot = td.orbitals0 @ rot
    ci_rot = td.rotate_ci_for_orbital_rotation(ci, rot)

    aligned_ci, aligned_orbitals, _ = td.align_orbital_gauge(td.orbitals0, orbitals_rot, ci_rot)

    np.testing.assert_allclose(aligned_orbitals, td.orbitals0, atol=1.0e-10)
    np.testing.assert_allclose(abs(np.vdot(aligned_ci, ci)), 1.0, atol=1.0e-10)
    assert td.active_gauge_error(td.orbitals0, aligned_orbitals) < 1.0e-12


def test_mctdhf_ao_rdm1_contracts_one_body_operators():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2)
    ci = td.ci_vector(0)
    h_ao = mf.get_hcore()
    h_mo = td.orbitals0.conj().T @ h_ao @ td.orbitals0
    dm1 = td.make_rdm1(ci)

    e_mo = np.einsum("pq,pq->", h_mo, dm1, optimize=True)
    e_ao = td.one_body_expectation_ao(ci, h_ao)

    np.testing.assert_allclose(e_ao, e_mo, atol=1.0e-10)


def test_mctdhf_contracted_rdm2_matches_full_reference():
    mf = _h2_mf(basis="6-31g")
    td = MCTDHF(mf, norb=2, nelec=2)
    ci = td.ci_vector(0)
    eri_full = transform_spatial_eri_to_mo(
        mf,
        td.s_inv_half,
        td.orbitals0,
        td.orbitals0,
        td.orbitals0,
    )

    contracted = td.contract_rdm2_eri_full(ci, eri_full)
    reference = td.contract_rdm2_eri_full_reference(ci, eri_full)

    np.testing.assert_allclose(contracted, reference, atol=1.0e-10)


def test_mctdhf_contracted_rdm2_matches_reference_for_larger_active_space():
    mf = _lih_mf()
    td = MCTDHF(mf, norb=3, nelec=2)
    ci = np.linspace(1.0, td.ndet, td.ndet) + 0.1j * np.linspace(td.ndet, 1.0, td.ndet)
    ci = ci / np.linalg.norm(ci)
    eri_full = transform_spatial_eri_to_mo(
        mf,
        td.s_inv_half,
        td.orbitals0,
        td.orbitals0,
        td.orbitals0,
    )

    contracted = td.contract_rdm2_eri_full(ci, eri_full)
    reference = td.contract_rdm2_eri_full_reference(ci, eri_full)

    np.testing.assert_allclose(contracted, reference, atol=1.0e-10)


def test_mctdhf_orbital_rhs_uses_contracted_rdm2_reference():
    mf = _h2_mf(basis="6-31g")
    td = MCTDHF(mf, norb=2, nelec=2)
    ci = td.ci_vector(0)
    orbitals = td.orbitals0
    h_ao = td.one_body_ao(0.0)
    h_full = td.s_inv_half.conj().T @ h_ao @ orbitals
    eri_full = transform_spatial_eri_to_mo(
        mf,
        td.s_inv_half,
        orbitals,
        orbitals,
        orbitals,
    )
    dm1 = td.make_rdm1(ci)
    rho_inv = np.linalg.pinv(dm1, rcond=td.rdm_rcond)
    two_body_ref = td.contract_rdm2_eri_full_reference(ci, eri_full)
    mean_ref = h_full + two_body_ref @ rho_inv.T
    y = td.s_half @ orbitals
    rhs_ref = -1j * (td.s_inv_half @ (mean_ref - y @ (y.conj().T @ mean_ref)))

    np.testing.assert_allclose(td.orbital_rhs(orbitals, ci), rhs_ref, atol=1.0e-10)


def test_mctdhf_density_provider_routes_orbital_rhs_and_observables():
    class CountingProvider:
        def bind(self, driver):
            self.delegate = DenseCIDensityProvider(driver)
            self.n_rdm1 = 0
            self.n_contract = 0

        def make_rdm1(self, ci):
            self.n_rdm1 += 1
            return self.delegate.make_rdm1(ci)

        def make_rdm2(self, ci):
            return self.delegate.make_rdm2(ci)

        def contract_rdm2_eri_full(self, ci, eri_full):
            self.n_contract += 1
            return self.delegate.contract_rdm2_eri_full(ci, eri_full)

    provider = CountingProvider()
    mf = _h2_mf(basis="6-31g")
    td = MCTDHF(mf, norb=2, nelec=2, density_provider=provider)
    ci = td.ci_vector(0)

    _rhs = td.orbital_rhs(td.orbitals0, ci)
    _dipole = td.dipole_moment(ci, td.orbitals0)

    assert provider.n_contract == 1
    assert provider.n_rdm1 >= 2


def test_mctdhf_dmrg_density_provider_uses_spatial_rdm_api():
    class FakeDMRG:
        def __init__(self, dm1, dm2):
            self.dm1 = dm1
            self.dm2 = dm2
            self.calls = []

        def make_rdm1(self, state_id=0, spatial=False, with_core=True):
            self.calls.append(("rdm1", state_id, spatial, with_core))
            return self.dm1

        def make_rdm2(self, state_id=0, spatial=False, with_core=True):
            self.calls.append(("rdm2", state_id, spatial, with_core))
            return self.dm2

    mf = _h2_mf()
    dm1 = np.eye(2)
    dm2 = np.zeros((2, 2, 2, 2))
    dm2[0, 0, 0, 0] = 1.0
    dm2[1, 1, 1, 1] = 1.0
    backend = FakeDMRG(dm1, dm2)
    provider = DMRGDensityProvider(backend, state_id=1)
    td = MCTDHF(mf, norb=2, nelec=2, density_provider=provider)

    np.testing.assert_allclose(td.make_rdm1(None), dm1)
    np.testing.assert_allclose(td.make_rdm2(None), dm2)

    assert ("rdm1", 1, True, False) in backend.calls
    assert ("rdm2", 1, True, False) in backend.calls


def test_mctdhf_dmrg_density_provider_contracts_like_rdm2_reference():
    class FakeDMRG:
        def __init__(self, dm1, dm2):
            self.dm1 = dm1
            self.dm2 = dm2

        def make_rdm1(self, state_id=0, spatial=False, with_core=False):
            return self.dm1

        def make_rdm2(self, state_id=0, spatial=False, with_core=False):
            return self.dm2

    mf = _lih_mf()
    td_ref = MCTDHF(mf, norb=3, nelec=2)
    ci = np.linspace(1.0, td_ref.ndet, td_ref.ndet)
    ci = ci / np.linalg.norm(ci)
    dm1 = td_ref.make_rdm1(ci)
    dm2 = td_ref.make_rdm2(ci)
    provider = DMRGDensityProvider(FakeDMRG(dm1, dm2))
    td = MCTDHF(mf, norb=3, nelec=2, density_provider=provider)
    eri_full = transform_spatial_eri_to_mo(
        mf,
        td.s_inv_half,
        td.orbitals0,
        td.orbitals0,
        td.orbitals0,
    )

    contracted = td.contract_rdm2_eri_full(None, eri_full)
    reference = np.einsum("oqrs,xqrs->xo", dm2, eri_full, optimize=True)

    np.testing.assert_allclose(contracted, reference, atol=1.0e-10)


def test_mctdhf_rdm_provider_forwards_current_state_to_density_methods():
    class DynamicRDMBackend:
        def __init__(self):
            self.rdm1_state = None
            self.rdm2_state = None
            self.driver_seen = None

        def make_rdm1(self, ci=None, state_id=0, spatial=False, with_core=False, driver=None):
            self.rdm1_state = ci
            self.driver_seen = driver
            assert state_id == 4
            assert spatial is True
            assert with_core is False
            return ci["dm1"]

        def make_rdm2(self, state=None, state_id=0, spatial=False, with_core=False):
            self.rdm2_state = state
            assert state_id == 4
            assert spatial is True
            assert with_core is False
            return state["dm2"]

    mf = _h2_mf()
    backend = DynamicRDMBackend()
    provider = DMRGDensityProvider(backend, state_id=4)
    td = MCTDHF(mf, norb=2, nelec=2, density_provider=provider)
    dm1 = np.diag([1.5, 0.5])
    dm2 = np.zeros((2, 2, 2, 2))
    dm2[0, 0, 0, 0] = 1.2
    dm2[1, 1, 1, 1] = 0.8
    state = {"dm1": dm1, "dm2": dm2}
    eri_full = np.arange(td.nao * td.norb**3, dtype=float).reshape(td.nao, td.norb, td.norb, td.norb)

    np.testing.assert_allclose(td.make_rdm1(state), dm1)
    contracted = td.contract_rdm2_eri_full(state, eri_full)
    reference = np.einsum("oqrs,xqrs->xo", dm2, eri_full, optimize=True)

    assert backend.rdm1_state is state
    assert backend.rdm2_state is state
    assert backend.driver_seen is td
    np.testing.assert_allclose(contracted, reference, atol=1.0e-12)


def test_mctdhf_external_state_provider_runs_opaque_ci():
    class OpaqueProvider:
        def __init__(self):
            self.n_propagate = 0

        def bind(self, driver):
            self.driver = driver

        def ci_vector(self, ci0=None, h1=None, eri=None):
            assert ci0 is None
            assert h1.shape == (2, 2)
            assert eri.shape == (2, 2, 2, 2)
            return {"step": 0}

        def propagate_ci(self, ci, dt=0.0, h1=None, eri=None):
            self.n_propagate += 1
            assert h1.shape == (2, 2)
            assert eri.shape == (2, 2, 2, 2)
            return {"step": ci["step"] + 1, "dt": dt}

        def electronic_energy(self, ci, **kwargs):
            return 0.25 + 0.1 * ci["step"]

        def norm(self, ci):
            return 1.0

        def overlap(self, bra, ket):
            return 1.0 + 0.1j * (ket["step"] - bra["step"])

        def make_rdm1(self, ci):
            return np.eye(2)

        def make_rdm2(self, ci):
            return np.zeros((2, 2, 2, 2))

        def contract_rdm2_eri_full(self, ci, eri_full):
            return np.zeros((self.driver.nao, self.driver.norb), dtype=complex)

    mf = _h2_mf()
    provider = OpaqueProvider()
    td = MCTDHF(mf, norb=2, nelec=2, orbital_mode="frozen", density_provider=provider)

    traj = td.run(dt=0.05, nsteps=2, ci0=None)

    assert traj.ci.shape == (3,)
    assert traj.ci.dtype == object
    assert [state["step"] for state in traj.ci] == [0, 1, 2]
    assert traj.final_ci["step"] == 2
    np.testing.assert_allclose(traj.final_time, 0.1)
    assert provider.n_propagate == 2
    np.testing.assert_allclose(traj.norms, 1.0)
    np.testing.assert_allclose(traj.electronic_energies, [0.25, 0.35, 0.45])
    np.testing.assert_allclose(traj.autocorrelation, [1.0, 1.0 + 0.1j, 1.0 + 0.2j])
    np.testing.assert_allclose(traj.natural_occupations, np.ones((3, 2)))
    np.testing.assert_allclose(traj.natural_occupation_trace_errors, 0.0)

    sparse = td.run(dt=0.05, nsteps=3, ci0=None, save_every=2)
    assert [state["step"] for state in sparse.ci] == [0, 2, 3]
    np.testing.assert_allclose(sparse.times, [0.0, 0.1, 0.15])

    no_ci = td.run(dt=0.05, nsteps=2, ci0=None, store_ci=False)
    assert no_ci.ci is None
    np.testing.assert_allclose(no_ci.electronic_energies, [0.25, 0.35, 0.45])


def test_mctdhf_dmrg_provider_forwards_optional_backend_hooks():
    class FakeTDVPDMRG:
        def __init__(self):
            self.calls = []

        def ci_vector(self, ci0=None, h1=None, eri=None, state_id=0, spatial=False):
            self.calls.append(("ci_vector", ci0, h1.shape, eri.shape, state_id, spatial))
            return {"step": 0}

        def propagate_ci(self, ci, dt=0.0, h1=None, eri=None, state_id=0, driver=None):
            self.calls.append(("propagate_ci", ci["step"], dt, h1.shape, eri.shape, state_id, driver.norb))
            return {"step": ci["step"] + 1}

        def electronic_energy(self, ci, state_id=0, **kwargs):
            self.calls.append(("electronic_energy", ci["step"], state_id))
            return -0.5 + 0.01 * ci["step"]

        def norm(self, ci, state_id=0):
            self.calls.append(("norm", ci["step"], state_id))
            return 1.0

        def make_rdm1(self, state_id=0, spatial=False, with_core=True):
            return np.eye(2)

        def make_rdm2(self, state_id=0, spatial=False, with_core=True):
            return np.zeros((2, 2, 2, 2))

    mf = _h2_mf()
    backend = FakeTDVPDMRG()
    provider = DMRGDensityProvider(backend, state_id=3)
    td = MCTDHF(mf, norb=2, nelec=2, orbital_mode="frozen", density_provider=provider)

    traj = td.run(dt=0.02, nsteps=1, ci0=None)

    assert traj.ci.dtype == object
    assert traj.ci[0]["step"] == 0
    assert traj.ci[1]["step"] == 1
    assert np.all(np.isnan(traj.autocorrelation))
    assert ("ci_vector", None, (2, 2), (2, 2, 2, 2), 3, True) in backend.calls
    assert ("propagate_ci", 0, 0.02, (2, 2), (2, 2, 2, 2), 3, 2) in backend.calls
    assert ("electronic_energy", 0, 3) in backend.calls
    assert ("norm", 0, 3) in backend.calls


def test_mctdhf_external_state_provider_requires_gauge_rotation_hook():
    class OpaqueProvider:
        def bind(self, driver):
            self.driver = driver

        def make_rdm1(self, ci):
            return np.eye(2)

        def contract_rdm2_eri_full(self, ci, eri_full):
            return np.zeros((self.driver.nao, self.driver.norb), dtype=complex)

    mf = _h2_mf()
    provider = OpaqueProvider()
    td = MCTDHF(mf, norb=2, nelec=2, density_provider=provider)

    with pytest.raises(NotImplementedError, match="rotate_ci_for_orbital_rotation"):
        td.align_orbital_gauge(td.orbitals0, td.orbitals0, {"opaque": True})


def test_mctdhf_rk4_orbital_integrator_uses_four_rhs_stages(monkeypatch):
    mf = _h2_mf()
    td = MCTDHF(mf, norb=2, nelec=2, orbital_integrator="rk4")
    ci = td.ci_vector(0)
    calls = {"rhs": 0, "propagate": 0}

    def fake_rhs(orbitals, ci, time=0.0, field=None, h1_ao=None):
        calls["rhs"] += 1
        return np.zeros_like(orbitals, dtype=complex)

    def fake_propagate(ci, orbitals=None, time=0.0, dt=0.0, field=None, h1_ao=None, backend=None):
        calls["propagate"] += 1
        return np.asarray(ci, dtype=complex)

    monkeypatch.setattr(td, "orbital_rhs", fake_rhs)
    monkeypatch.setattr(td, "propagate_ci", fake_propagate)

    ci1, orbitals1 = td.step(ci, td.orbitals0, time=0.0, dt=0.01)

    assert calls == {"rhs": 4, "propagate": 3}
    np.testing.assert_allclose(abs(np.vdot(ci1, ci)), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(orbitals1, td.orbitals0, atol=1.0e-12)


def test_mctdhf_moving_orbitals_preserve_norm_and_metric():
    mf = _h2_mf(basis="6-31g")
    td = MCTDHF(
        mf,
        norb=2,
        nelec=2,
        orbital_mode="mctdhf",
        ci_backend="krylov",
        field=lambda t: np.array([0.02 * np.sin(t), 0.0, 0.0]),
    )

    traj = td.run(dt=0.02, nsteps=4)

    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)
    assert np.max(traj.orbital_errors) < 1.0e-10
    assert np.max(traj.active_gauge_errors) < 1.0e-10
    np.testing.assert_allclose(traj.core_overlap_errors, 0.0, atol=1.0e-12)
    assert traj.orbitals.shape == (5, mf.nao, 2)
    assert traj.ci.shape == (5, td.ndet)
    assert traj.dipoles.shape == (5, 3)
    np.testing.assert_allclose(traj.dipoles[0], td.dipole_moment(traj.ci[0], traj.orbitals[0]))


def test_mctdhf_field_free_kicked_truncated_active_energy_is_stable():
    mf = _h2_mf(basis="6-31g")
    td = MCTDHF(
        mf,
        norb=2,
        nelec=2,
        orbital_mode="mctdhf",
        ci_backend="krylov",
    )

    traj = td.run(
        dt=0.005,
        nsteps=16,
        kick={"strength": 1.0e-3, "axis": "z"},
    )

    assert np.max(np.abs(traj.energies - traj.energies[0])) < 5.0e-8
    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)
    assert np.max(traj.orbital_errors) < 1.0e-10
    assert np.max(traj.active_gauge_errors) < 1.0e-10


def test_mctdhf_rk4_moving_orbitals_preserve_norm_and_metric():
    mf = _h2_mf(basis="6-31g")
    td = MCTDHF(
        mf,
        norb=2,
        nelec=2,
        orbital_mode="mctdhf",
        orbital_integrator="rk4",
        ci_backend="krylov",
        field=lambda t: np.array([0.02 * np.sin(t), 0.0, 0.0]),
    )

    traj = td.run(dt=0.02, nsteps=3)

    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)
    assert np.max(traj.orbital_errors) < 1.0e-10
    assert np.max(traj.active_gauge_errors) < 1.0e-10
    assert traj.orbitals.shape == (4, mf.nao, 2)


def test_mctdhf_full_active_limit_has_no_orbital_rhs_and_conserves_energy():
    mf = _h2_mf()
    td = MCTDHF(mf, norb=mf.nao, nelec=2, orbital_mode="mctdhf")
    ci0 = np.linspace(1.0, td.ndet, td.ndet) + 0.2j * np.linspace(td.ndet, 1.0, td.ndet)
    ci0 = ci0 / np.linalg.norm(ci0)

    rhs = td.orbital_rhs(td.orbitals0, ci0)
    traj = td.run(dt=0.05, nsteps=4, ci0=ci0)

    assert np.linalg.norm(rhs) < 1.0e-12
    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(traj.energies, traj.energies[0], atol=1.0e-10)
    assert np.max(traj.orbital_errors) < 1.0e-10
    assert np.max(traj.active_gauge_errors) < 1.0e-10


def test_mctdhf_full_active_limit_matches_frozen_with_field():
    mf = _h2_mf(basis="6-31g")
    field = lambda t: np.array([0.01 * np.sin(0.7 * t), 0.0, 0.0])
    moving = MCTDHF(
        mf,
        norb=mf.nao,
        nelec=2,
        orbital_mode="mctdhf",
        ci_backend="krylov",
        field=field,
    )
    frozen = MCTDHF(
        mf,
        norb=mf.nao,
        nelec=2,
        orbital_mode="frozen",
        ci_backend="krylov",
        field=field,
    )
    ci0 = moving.ci_vector(0)

    traj_moving = moving.run(dt=0.02, nsteps=5, ci0=ci0)
    traj_frozen = frozen.run(dt=0.02, nsteps=5, ci0=ci0)

    np.testing.assert_allclose(traj_moving.energies, traj_frozen.energies, atol=1.0e-10)
    np.testing.assert_allclose(traj_moving.autocorrelation, traj_frozen.autocorrelation, atol=1.0e-10)
    np.testing.assert_allclose(traj_moving.dipoles, traj_frozen.dipoles, atol=1.0e-10)
    assert np.max(traj_moving.orbital_errors) < 1.0e-10
    assert np.max(traj_moving.active_gauge_errors) < 1.0e-10
