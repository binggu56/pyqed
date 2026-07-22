import numpy as np
from scipy.linalg import expm

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.dmrg import TDDMRG, gaussian_pulse
from pyqed.qchem.dmrg.tddmrg import _DenseStateTransformOperator, _mpo_to_dense_matrix
from pyqed.mps import MPS
from pyqed.mps.decompose import tt_to_tensor
from pyqed.mps.mps import expect_mps


def test_tddmrg_runs_from_converged_ground_state():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    td = TDDMRG(mf, ncas=2, nelecas=2, init_guess="cid").build()
    td.optimize_ground_state(
        D=8,
        nstates=1,
        nsweeps=4,
        symmetry_list=["charge", "sz"],
        compute_s2=False,
    )
    td.run(dt=0.01, steps=2, e_ops=["H"], interval=1, D=8)

    np.testing.assert_allclose(td.times, np.array([0.01, 0.02]))
    assert td.observables.shape == (2, 1)
    assert td.final_state is not None
    np.testing.assert_allclose(td.pre_normalization_norms, np.ones(2), atol=1.0e-12)
    assert td.static_energies.shape == (3,)
    np.testing.assert_allclose(td.energy_drift, np.zeros(3), atol=1.0e-10)

    reversal = td.time_reversal_error(dt=0.01, steps=2, D=8)
    assert reversal["state_error"] < 1.0e-10


def test_tddmrg_supports_gaussian_pulse_and_dipole_observable():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    pulse = gaussian_pulse(
        amplitude=1e-3,
        center=0.02,
        width=0.01,
        omega=0.4,
        polarization=(1.0, 0.0, 0.0),
    )

    td = TDDMRG(mf, ncas=2, nelecas=2, init_guess="cid").build()
    td.optimize_ground_state(
        D=8,
        nstates=1,
        nsweeps=4,
        symmetry_list=["charge", "sz"],
        compute_s2=False,
    )
    td.run(dt=0.01, steps=3, e_ops=["H", "mu_x"], interval=1, field=pulse, D=8)

    assert td.observables.shape == (3, 2)
    assert td.fields.shape == (3, 3)
    assert np.any(np.abs(td.fields[:, 0]) > 0.0)


def test_tddmrg_builds_exact_one_body_field_propagator():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    pulse = gaussian_pulse(
        amplitude=1e-3,
        center=0.02,
        width=0.01,
        omega=0.3,
        polarization=(0.0, 0.0, 1.0),
    )

    td = TDDMRG(mf, ncas=2, nelecas=2, init_guess="cid").build()
    mpo = td.build_interaction_unitary_mpo(dt=0.01, time=0.02, field=pulse)

    assert mpo is not None
    assert isinstance(mpo, _DenseStateTransformOperator)


def test_dense_state_transform_operator_respects_td_bond_cap():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    pulse = gaussian_pulse(
        amplitude=1e-3,
        center=0.02,
        width=0.01,
        omega=0.3,
        polarization=(0.0, 0.0, 1.0),
    )

    td = TDDMRG(mf, ncas=2, nelecas=2, init_guess="cid").build()
    mpo = td.build_interaction_unitary_mpo(dt=0.01, time=0.02, field=pulse, D=2)
    psi0 = MPS(td.get_initial_guess_dense(noise=0.0), labels=["lv", "p", "rv"]).normalize()
    psi1 = mpo @ psi0

    max_bond = max(max(f.shape[0], f.shape[-1]) for f in psi1.factors)
    assert isinstance(mpo, _DenseStateTransformOperator)
    assert max_bond <= td.bond_dim


def test_rhf_dipole_operator_defaults_to_center_of_mass():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    ao_op = mf.dipole()
    ref = np.asarray(mol.moment_integral(center=mol.center_of_mass()), dtype=float)
    if ref.shape[0] != 3:
        ref = np.moveaxis(ref, -1, 0)
    origin_ref = np.asarray(mol.moment_integral(), dtype=float)
    if origin_ref.shape[0] != 3:
        origin_ref = np.moveaxis(origin_ref, -1, 0)

    np.testing.assert_allclose(ao_op, -ref)
    assert mf.dipole(basis="mo").shape == (3, mf.nmo, mf.nmo)
    assert not np.allclose(ao_op, -origin_ref)


def test_tddmrg_uses_center_of_mass_dipole_origin():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    td = TDDMRG(mf, ncas=2, nelecas=2, init_guess="cid").build()
    np.testing.assert_allclose(td.get_interaction_ao(), mf.dipole(basis="ao"))


def test_tddmrg_run_with_mo_coeff_rebuilds_interaction_caches():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    td = TDDMRG(mf, ncas=2, nelecas=2, init_guess="cid").build()
    old_spatial = td.get_interaction_spatial(axis=2)

    theta = 0.37
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )
    mo_new = np.array(mf.mo_coeff, copy=True)
    mo_new[:, :2] = mo_new[:, :2] @ rotation

    pulse = gaussian_pulse(
        amplitude=1e-3,
        center=0.02,
        width=0.01,
        omega=0.3,
        polarization=(0.0, 0.0, 1.0),
    )
    psi0 = MPS(td.get_initial_guess_dense(noise=0.0), labels=["lv", "p", "rv"]).normalize()
    td.run(
        psi0=psi0,
        dt=0.01,
        steps=0,
        e_ops=["mu_z"],
        field=pulse,
        mo_coeff=mo_new,
        D=8,
    )

    expected = td.mo_cas.conj().T @ td.get_interaction_ao()[2] @ td.mo_cas
    new_spatial = td.get_interaction_spatial(axis=2)
    assert not np.allclose(old_spatial, expected)
    np.testing.assert_allclose(new_spatial, expected)


def test_tddmrg_h4_uses_exact_dense_td_path_and_matches_dense_oracle():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    td = TDDMRG(mf, ncas=4, nelecas=4, init_guess="hf").build()
    psi0 = MPS(td.get_initial_guess_dense(noise=0.0), labels=["lv", "p", "rv"]).normalize()
    pulse = gaussian_pulse(
        amplitude=2e-3,
        center=0.5,
        width=0.2,
        omega=0.0,
        phase=0.0,
        polarization=(0.0, 0.0, 1.0),
    )

    dt = 0.05
    steps = 2
    mu_mpo = td.get_interaction_mpo(axis=2)
    h_dense = _mpo_to_dense_matrix(td._get_td_hamiltonian())
    mu_dense = _mpo_to_dense_matrix(mu_mpo)
    u_half = expm(-0.5j * dt * h_dense)

    vec = np.asarray(tt_to_tensor(psi0.factors), dtype=complex).reshape(-1)
    mu_exact = [float(np.real(np.vdot(vec, mu_dense @ vec)))]
    time = 0.0
    for _ in range(steps):
        field_vec = td._field_vector(time + 0.5 * dt, pulse)
        h_int = -field_vec[2] * mu_dense
        vec = u_half @ vec
        vec = expm(-1j * dt * h_int) @ vec
        vec = u_half @ vec
        vec = vec / np.linalg.norm(vec)
        mu_exact.append(float(np.real(np.vdot(vec, mu_dense @ vec))))
        time += dt

    builder = td.build_interaction_unitary_mpo(dt, time=0.025, field=pulse)
    assert isinstance(builder, _DenseStateTransformOperator)

    td.run(psi0=psi0, dt=dt, steps=steps, interval=1, field=pulse, e_ops=["mu_z"], D=8)
    mu0 = float(np.real(expect_mps(psi0.factors, mu_mpo.factors)))
    mu_td = np.concatenate(([mu0], np.real(td.observables[:, 0])))

    np.testing.assert_allclose(mu_td, mu_exact, atol=1e-9, rtol=1e-7)


def test_gaussian_pulse_omega_alias_matches_frequency():
    pulse_omega = gaussian_pulse(
        amplitude=1.0,
        center=0.2,
        width=0.4,
        omega=0.7,
        phase=0.3,
        polarization=(0.0, 0.0, 1.0),
    )
    pulse_frequency = gaussian_pulse(
        amplitude=1.0,
        center=0.2,
        width=0.4,
        frequency=0.7,
        phase=0.3,
        polarization=(0.0, 0.0, 1.0),
    )

    np.testing.assert_allclose(pulse_omega(0.5), pulse_frequency(0.5))
    assert pulse_omega.omega == pulse_omega.frequency == 0.7
