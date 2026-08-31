import numpy as np
import pytest
from types import SimpleNamespace
from scipy.linalg import expm

from pyqed.mps import MPS
from pyqed.mps.decompose import decompose, tt_to_tensor
from pyqed.mps.mps import dense_to_symmetric
from pyqed.mps.symmetry import AbelianSector, BlockTensor
from pyqed.qchem.dmrg.dmrg import (
    _build_spatial_active_hamiltonian_matrix,
    _build_spin_orbital_dense_hamiltonian_tensor_mpo,
)
from pyqed.qchem.dmrg import TDDMRG as QChemTDDMRG
from pyqed.qchem.dmrg.tddmrg import _mpo_to_dense_matrix
from pyqed.qchem.gdvr import TDDMRG
from pyqed.qchem.gdvr.tddmrg import (
    acceleration_mpo,
    active_eri_from_gdvr_collocation,
    apply_gdvr_spatial_density_phase,
    apply_gdvr_spatial_one_body_rotation,
    build_gdvr_spatial_factorized_density_phase_mpo,
    build_gdvr_spatial_exponential_density_hamiltonian_mpo,
    build_gdvr_dipole_mpo,
    build_gdvr_hamiltonian_mpo,
    build_gdvr_spatial_density_phase_mpo,
    build_gdvr_spatial_hamiltonian_mpo,
    build_gdvr_spatial_one_body_rotation_mpo,
    build_gdvr_spatial_prony_density_hamiltonian_mpo,
    build_gdvr_spatial_svd_density_hamiltonian_mpo,
    cap_mpo,
    cap_operator,
    cap_profile,
    dipole_mpo,
    force_mpo,
    GDVRSpatialLocalPhase,
    GDVRSpatialHybridDensityPhase,
    GDVRSpatialFactorizedDensityPhase,
    GDVRSpatialTaylorDensityPhase,
    prony_exponential_fit,
    rhf_determinant_mps,
)


class _ToyGDVRMolecule:
    def __init__(self):
        self.z = np.array([-0.5, 0.75])
        self.shapes = {"Nz": 2, "M": 1, "size": 2}
        self.hcore = np.array([[-0.8, 0.13], [0.13, -0.25]])
        self.e_slices = np.array([[-0.8], [-0.25]])
        self.eri_j = [
            [np.array([[0.70]]), np.array([[0.31]])],
            [np.array([[0.31]]), np.array([[0.55]])],
        ]
        self.eri_k = [
            [np.array([[0.70]]), np.array([[0.31]])],
            [np.array([[0.31]]), np.array([[0.55]])],
        ]
        self.nelec = 2
        self.spin = 0

    def nuclear_repulsion_energy(self):
        return 0.0


class _ToyGDVRRHF:
    def __init__(self):
        self.mol = _ToyGDVRMolecule()
        self.mo_coeff = np.eye(2)
        self.mo_energy = np.array([-0.8, -0.25])
        self.mo_occ = np.array([2.0, 0.0])
        self.dm = np.diag(self.mo_occ)
        self.e_tot = -1.0

    def to_gto(self, orbitals=None):
        from pyqed.qchem.gdvr.tddmrg import GDVRMeanFieldAdapter

        mo_coeff = None
        if orbitals is not None:
            orbitals = tuple(int(i) for i in orbitals)
            mo_coeff = self.mo_coeff[:, orbitals]
        return GDVRMeanFieldAdapter(self, mo_coeff=mo_coeff)

    def energy_nuc(self):
        return self.mol.nuclear_repulsion_energy()

    def get_hcore(self):
        return self.mol.hcore

    def dipole(self, basis="ao"):
        z = np.diag(self.mol.z)
        return np.array([np.zeros_like(z), np.zeros_like(z), -z])


class _ThreeSiteToyGDVRMolecule:
    def __init__(self):
        self.z = np.array([-0.6, 0.1, 0.9])
        self.shapes = {"Nz": 3, "M": 1, "size": 3}
        self.hcore = np.array(
            [
                [-0.7, 0.12, -0.03],
                [0.12, -0.4, 0.08],
                [-0.03, 0.08, -0.2],
            ]
        )
        eri = np.array(
            [
                [0.70, 0.31, 0.18],
                [0.31, 0.55, 0.26],
                [0.18, 0.26, 0.48],
            ]
        )
        self.eri_j = [[np.array([[eri[i, j]]]) for j in range(3)] for i in range(3)]
        self.eri_k = [[np.array([[eri[i, j]]]) for j in range(3)] for i in range(3)]


def _random_spatial_mps(nsites, seed=8):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=4**nsites) + 1j * rng.normal(size=4**nsites)
    vec = vec / np.linalg.norm(vec)
    tensor = vec.reshape((4,) * nsites)
    return MPS(decompose(tensor, rank=4**nsites), labels=["lv", "p", "rv"]), vec


def _one_site_spatial_symmetric_state():
    q_empty = AbelianSector(("charge", "sz"), (0, 0))
    q_up = AbelianSector(("charge", "sz"), (1, 1))
    q_down = AbelianSector(("charge", "sz"), (1, -1))
    q_double = AbelianSector(("charge", "sz"), (2, 0))
    data = {
        (q_empty, q_empty, q_up): np.array([[[3.0]]], dtype=complex),
        (q_empty, q_empty, q_down): np.array([[[7.0]]], dtype=complex),
    }
    tensor = BlockTensor(
        data,
        [[q_empty], [q_empty], [q_empty, q_down, q_up, q_double]],
        [-1, 1, 1],
    )
    site_qn_maps = [{0: q_empty, 1: q_up, 2: q_down, 3: q_double}]
    return MPS([tensor], labels=["lv", "rv", "p"]), site_qn_maps


def _spatial_to_spin_permutation(nspatial):
    local_to_bits = {
        0: (0, 0),
        1: (1, 0),
        2: (0, 1),
        3: (1, 1),
    }
    perm = []
    for spatial_index in range(4**nspatial):
        local_states = np.unravel_index(spatial_index, (4,) * nspatial)
        bits = []
        for local_state in local_states:
            bits.extend(local_to_bits[int(local_state)])
        perm.append(np.ravel_multi_index(tuple(bits), (2,) * (2 * nspatial)))
    return np.asarray(perm, dtype=int)


def test_direct_gdvr_hamiltonian_mpo_matches_dense_eri_oracle():
    mol = _ToyGDVRMolecule()

    direct_mpo, info = build_gdvr_hamiltonian_mpo(mol)
    direct = _mpo_to_dense_matrix(direct_mpo)

    eri = active_eri_from_gdvr_collocation(mol.eri_j, np.eye(2), nz=2, m=1)
    h2 = np.stack(((eri, eri.copy()), (eri.copy(), eri.copy())))
    dense_mpo, _, _ = _build_spin_orbital_dense_hamiltonian_tensor_mpo(
        [mol.hcore, mol.hcore.copy()],
        h2,
        ncas=2,
    )
    oracle = _mpo_to_dense_matrix(dense_mpo)

    np.testing.assert_allclose(direct, oracle, atol=1.0e-12)
    assert info["representation"] == "gdvr_direct_spin_orbital_mpo"


def test_direct_spatial_gdvr_hamiltonian_mpo_matches_spin_orbital_oracle():
    mol = _ToyGDVRMolecule()

    spatial_mpo, info = build_gdvr_spatial_hamiltonian_mpo(mol)
    spatial = _mpo_to_dense_matrix(spatial_mpo)

    spin_mpo, _ = build_gdvr_hamiltonian_mpo(mol)
    spin = _mpo_to_dense_matrix(spin_mpo)
    perm = _spatial_to_spin_permutation(nspatial=2)
    spin_in_spatial_order = spin[np.ix_(perm, perm)]

    np.testing.assert_allclose(spatial, spin_in_spatial_order, atol=1.0e-12)
    assert info["representation"] == "gdvr_direct_spatial_mpo"
    assert info["native_abelian_mpo"] is True
    assert hasattr(spatial_mpo.factors[0], "qns")


def test_to_gto_feeds_generic_active_space_tddmrg():
    mf = _ToyGDVRRHF()
    td = QChemTDDMRG(mf.to_gto(), ncas=2, nelecas=(1, 1)).build()

    expected_eri = active_eri_from_gdvr_collocation(mf.mol.eri_j, np.eye(2), nz=2, m=1)
    np.testing.assert_allclose(td.h1e[0], mf.mol.hcore, atol=1.0e-12)
    np.testing.assert_allclose(td.h2e[0, 0], expected_eri, atol=1.0e-12)
    assert td.build_info["mode"] == "gdvr_collocated_dense_active"


def test_gdvr_tddmrg_root_api_has_no_active_space_aliases():
    import pyqed.qchem.gdvr as gdvr

    assert gdvr.TDDMRG is TDDMRG
    assert not hasattr(gdvr, "qchem_mf")
    assert not hasattr(gdvr, "ActiveSpaceTDDMRG")
    assert not hasattr(gdvr, "RealTimeDMRG")
    assert not hasattr(gdvr, "RTTDDMRG")


def test_direct_gdvr_dipole_mpo_is_diagonal_number_operator():
    mol = _ToyGDVRMolecule()
    mu = _mpo_to_dense_matrix(build_gdvr_dipole_mpo(mol))

    expected = np.zeros_like(mu)
    z = -np.repeat(mol.z, 2)
    for state in range(mu.shape[0]):
        occ = np.asarray(np.unravel_index(state, (2, 2, 2, 2)))
        expected[state, state] = float(np.dot(z, occ))

    np.testing.assert_allclose(mu, expected, atol=1.0e-12)


def test_rhf_determinant_mps_can_preserve_abelian_sectors():
    mf = _ToyGDVRRHF()
    psi = rhf_determinant_mps(mf, preserve_quantum_numbers=True)
    q_empty = AbelianSector(("charge", "sz"), (0, 0))
    q_up = AbelianSector(("charge", "sz"), (1, 1))
    q_down = AbelianSector(("charge", "sz"), (1, -1))
    q_double = AbelianSector(("charge", "sz"), (2, 0))

    sym_factors = dense_to_symmetric(
        psi.to_order(["lv", "p", "rv"]).factors,
        phys_qns=[q_empty, q_up, q_down, q_double],
    )

    assert all(hasattr(factor, "qns") for factor in sym_factors)


def test_rhf_single_occupied_orbital_fast_path_has_closed_shell_amplitudes():
    orbital = np.array([0.4, -0.2 + 0.3j, 0.7], dtype=complex)
    orbital = orbital / np.linalg.norm(orbital)
    mo_coeff = np.eye(3, dtype=complex)
    mo_coeff[:, 0] = orbital
    mf = SimpleNamespace(
        mol=SimpleNamespace(spin=0),
        mo_coeff=mo_coeff,
        mo_occ=np.array([2.0, 0.0, 0.0]),
    )

    psi = rhf_determinant_mps(mf)
    actual = np.asarray(tt_to_tensor(psi.factors), dtype=complex).reshape((4, 4, 4))
    expected = np.zeros((4, 4, 4), dtype=complex)
    for alpha_site, alpha_coeff in enumerate(orbital):
        for beta_site, beta_coeff in enumerate(orbital):
            states = [0, 0, 0]
            if alpha_site == beta_site:
                states[alpha_site] = 3
            else:
                states[alpha_site] = 1
                states[beta_site] = 2
            sign = -1.0 if beta_site < alpha_site else 1.0
            expected[tuple(states)] = sign * alpha_coeff * beta_coeff

    np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_direct_spatial_gdvr_dipole_mpo_is_site_number_operator():
    mol = _ToyGDVRMolecule()
    mu = _mpo_to_dense_matrix(dipole_mpo(mol))

    expected = np.zeros_like(mu)
    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    z = -mol.z
    for state in range(mu.shape[0]):
        local_states = np.asarray(np.unravel_index(state, (4, 4)))
        expected[state, state] = float(np.dot(z, occupation[local_states]))

    np.testing.assert_allclose(mu, expected, atol=1.0e-12)


def test_gdvr_cap_mpo_is_negative_imaginary_number_operator():
    mol = _ThreeSiteToyGDVRMolecule()
    cap = _mpo_to_dense_matrix(cap_mpo(mol, width=0.4, strength=0.2, order=2))

    expected = np.zeros_like(cap)
    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    one_body_diag = np.diag(cap_operator(mol, width=0.4, strength=0.2, order=2))
    for state in np.ndindex(4, 4, 4):
        idx = np.ravel_multi_index(state, (4, 4, 4))
        expected[idx, idx] = np.dot(one_body_diag, occupation[np.asarray(state)])

    np.testing.assert_allclose(cap, expected, atol=1.0e-12)
    assert np.all(np.real(np.diag(cap)) == pytest.approx(0.0))
    assert np.min(np.imag(np.diag(cap))) < 0.0


def test_gdvr_local_phase_matches_dense_diagonal_action():
    mol = _ThreeSiteToyGDVRMolecule()
    psi, vec = _random_spatial_mps(3)
    dt = 0.037
    field_z = 0.02
    cap_values = cap_profile(mol, width=0.4, strength=0.2, order=2)

    actual = GDVRSpatialLocalPhase.from_mol(
        mol,
        dt,
        field_z=field_z,
        cap_values=cap_values,
    ) @ psi

    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    phases = []
    for state in np.ndindex(4, 4, 4):
        phase = 1.0 + 0.0j
        for site, local_state in enumerate(state):
            occ = occupation[int(local_state)]
            phase *= np.exp(-1j * dt * field_z * mol.z[site] * occ)
            phase *= np.exp(-dt * cap_values[site] * occ)
        phases.append(phase)
    expected = np.asarray(phases) * vec

    np.testing.assert_allclose(tt_to_tensor(actual.factors).reshape(-1), expected, atol=1.0e-12)


def test_field_free_dipole_acceleration_mpo_matches_dense_commutator():
    mol = _ToyGDVRMolecule()
    h_mpo, _ = build_gdvr_spatial_hamiltonian_mpo(mol)
    mu_mpo = dipole_mpo(mol)
    acc_mpo = acceleration_mpo(h_mpo, mu_mpo)

    h = _mpo_to_dense_matrix(h_mpo)
    mu = _mpo_to_dense_matrix(mu_mpo)
    expected = -(mu @ h @ h - 2.0 * h @ mu @ h + h @ h @ mu)

    np.testing.assert_allclose(_mpo_to_dense_matrix(acc_mpo), expected, atol=1.0e-10)


def test_gdvr_spatial_force_acceleration_mpo_is_diagonal_number_operator():
    mol = _ToyGDVRMolecule()
    acc = _mpo_to_dense_matrix(force_mpo(mol))

    force = np.gradient(mol.e_slices[:, 0], mol.z, edge_order=1)
    expected = np.zeros_like(acc)
    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    for state in range(acc.shape[0]):
        local_states = np.asarray(np.unravel_index(state, (4, 4)))
        expected[state, state] = float(np.dot(force, occupation[local_states]))

    np.testing.assert_allclose(acc, expected, atol=1.0e-12)


def test_gdvr_spatial_one_body_rotation_mpo_matches_dense_oracle():
    mol = _ToyGDVRMolecule()
    dt = 0.037

    actual = _mpo_to_dense_matrix(build_gdvr_spatial_one_body_rotation_mpo(mol.hcore, dt))
    h_dense, _ = _build_spatial_active_hamiltonian_matrix(
        [mol.hcore, mol.hcore.copy()],
        np.zeros((2, 2, 2, 2, 2, 2)),
    )
    expected = expm(-1j * dt * h_dense)

    np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_gdvr_spatial_density_phase_mpo_matches_dense_oracle():
    mol = _ToyGDVRMolecule()
    dt = 0.041
    field_z = 0.02

    actual = _mpo_to_dense_matrix(build_gdvr_spatial_density_phase_mpo(mol, dt, field_z=field_z))

    eri = np.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            eri[i, i, j, j] = mol.eri_j[i][j][0, 0]
    h2 = np.stack(((eri, eri.copy()), (eri.copy(), eri.copy())))
    h_dense, _ = _build_spatial_active_hamiltonian_matrix(
        [np.zeros_like(mol.hcore), np.zeros_like(mol.hcore)],
        h2,
    )

    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    field_diag = []
    for state in np.ndindex(4, 4):
        field_diag.append(sum(field_z * mol.z[i] * occupation[s] for i, s in enumerate(state)))
    expected = expm(-1j * dt * (h_dense + np.diag(field_diag)))

    np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_gdvr_spatial_factorized_density_phase_mpo_matches_pair_phase():
    mol = _ThreeSiteToyGDVRMolecule()
    dt = 0.041
    field_z = 0.02

    actual, info = build_gdvr_spatial_factorized_density_phase_mpo(
        mol,
        dt,
        field_z=field_z,
        rank=3,
    )
    expected = build_gdvr_spatial_density_phase_mpo(mol, dt, field_z=field_z)

    np.testing.assert_allclose(
        _mpo_to_dense_matrix(actual),
        _mpo_to_dense_matrix(expected),
        atol=1.0e-12,
    )
    assert info["full_kernel_rel_error"] < 1.0e-12


def test_apply_gdvr_spatial_one_body_rotation_matches_dense_vector():
    mol = _ThreeSiteToyGDVRMolecule()
    psi, vec = _random_spatial_mps(3)
    dt = 0.037

    actual = apply_gdvr_spatial_one_body_rotation(psi, mol.hcore, dt, max_bond=64)
    h_dense, _ = _build_spatial_active_hamiltonian_matrix(
        [mol.hcore, mol.hcore.copy()],
        np.zeros((2, 2, 3, 3, 3, 3), dtype=complex),
    )
    expected = expm(-1j * dt * h_dense) @ vec

    np.testing.assert_allclose(tt_to_tensor(actual.factors).reshape(-1), expected, atol=1.0e-12)


def test_apply_gdvr_spatial_density_phase_matches_dense_vector():
    mol = _ThreeSiteToyGDVRMolecule()
    psi, vec = _random_spatial_mps(3)
    dt = 0.041
    field_z = 0.02
    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    double_occupation = np.array([0.0, 0.0, 0.0, 1.0])

    actual = apply_gdvr_spatial_density_phase(
        psi,
        mol,
        dt,
        field_z=field_z,
        max_bond=64,
    )

    energy = []
    for state in np.ndindex(4, 4, 4):
        value = 0.0
        for i, local_state in enumerate(state):
            value += mol.eri_j[i][i][0, 0] * double_occupation[local_state]
            value += field_z * mol.z[i] * occupation[local_state]
        for i in range(3):
            for j in range(i + 1, 3):
                value += mol.eri_j[i][j][0, 0] * occupation[state[i]] * occupation[state[j]]
        energy.append(value)
    expected = np.exp(-1j * dt * np.asarray(energy)) * vec

    np.testing.assert_allclose(tt_to_tensor(actual.factors).reshape(-1), expected, atol=1.0e-12)


def test_factorized_density_phase_apply_matches_dense_vector():
    mol = _ThreeSiteToyGDVRMolecule()
    psi, vec = _random_spatial_mps(3)
    dt = 0.041
    field_z = 0.02
    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    double_occupation = np.array([0.0, 0.0, 0.0, 1.0])

    actual = GDVRSpatialFactorizedDensityPhase(mol, dt, rank=3).apply(
        psi,
        field_z=field_z,
        max_bond=64,
    )

    energy = []
    for state in np.ndindex(4, 4, 4):
        value = 0.0
        for i, local_state in enumerate(state):
            value += mol.eri_j[i][i][0, 0] * double_occupation[local_state]
            value += field_z * mol.z[i] * occupation[local_state]
        for i in range(3):
            for j in range(i + 1, 3):
                value += mol.eri_j[i][j][0, 0] * occupation[state[i]] * occupation[state[j]]
        energy.append(value)
    expected = np.exp(-1j * dt * np.asarray(energy)) * vec

    np.testing.assert_allclose(tt_to_tensor(actual.factors).reshape(-1), expected, atol=1.0e-12)


def test_taylor_density_phase_apply_converges_to_dense_vector():
    mol = _ThreeSiteToyGDVRMolecule()
    psi, vec = _random_spatial_mps(3)
    dt = 0.01
    field_z = 0.02
    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    double_occupation = np.array([0.0, 0.0, 0.0, 1.0])

    actual = GDVRSpatialTaylorDensityPhase(mol, dt, order=10, rank=3).apply(
        psi,
        field_z=field_z,
        max_bond=128,
    )

    energy = []
    for state in np.ndindex(4, 4, 4):
        value = 0.0
        for i, local_state in enumerate(state):
            value += mol.eri_j[i][i][0, 0] * double_occupation[local_state]
            value += field_z * mol.z[i] * occupation[local_state]
        for i in range(3):
            for j in range(i + 1, 3):
                value += mol.eri_j[i][j][0, 0] * occupation[state[i]] * occupation[state[j]]
        energy.append(value)
    expected = np.exp(-1j * dt * np.asarray(energy)) * vec

    error = np.max(np.abs(tt_to_tensor(actual.factors).reshape(-1) - expected))
    assert error < 5.0e-4


def test_prony_exponential_fit_recovers_synthetic_density_kernel():
    offsets = np.arange(1, 8, dtype=float)
    coeffs = np.array([0.7, -0.2])
    lambdas = np.array([0.82, 0.35])
    values = sum(c * lambdas[i] ** offsets for i, c in enumerate(coeffs))

    fit = prony_exponential_fit(values, rank=2, offsets=offsets)

    np.testing.assert_allclose(fit["fitted"], values, atol=1.0e-12)
    assert fit["rel_error"] < 1.0e-12


def test_exponential_density_hamiltonian_mpo_matches_dense_diagonal():
    nsites = 4
    coeffs = np.array([0.7, -0.2])
    lambdas = np.array([0.82, 0.35])

    actual = _mpo_to_dense_matrix(
        build_gdvr_spatial_exponential_density_hamiltonian_mpo(nsites, coeffs, lambdas)
    )

    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    expected = np.zeros_like(actual)
    for state in np.ndindex(*(4,) * nsites):
        energy = 0.0
        for i in range(nsites):
            for j in range(i + 1, nsites):
                distance = j - i
                value = sum(c * lambdas[a] ** distance for a, c in enumerate(coeffs))
                energy += value * occupation[state[i]] * occupation[state[j]]
        idx = np.ravel_multi_index(state, (4,) * nsites)
        expected[idx, idx] = energy

    np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_prony_residual_density_hamiltonian_mpo_can_recover_exact_kernel():
    mol = _ThreeSiteToyGDVRMolecule()

    mpo, info = build_gdvr_spatial_prony_density_hamiltonian_mpo(
        mol,
        rank=1,
        residual_rank=3,
    )
    actual = _mpo_to_dense_matrix(mpo)

    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    expected = np.zeros_like(actual)
    for state in np.ndindex(4, 4, 4):
        energy = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                energy += mol.eri_j[i][j][0, 0] * occupation[state[i]] * occupation[state[j]]
        idx = np.ravel_multi_index(state, (4, 4, 4))
        expected[idx, idx] = energy

    np.testing.assert_allclose(actual, expected, atol=1.0e-12)
    assert info["full_kernel_rel_error"] < 1.0e-12


def test_svd_density_hamiltonian_mpo_can_recover_exact_kernel():
    mol = _ThreeSiteToyGDVRMolecule()

    mpo, info = build_gdvr_spatial_svd_density_hamiltonian_mpo(mol, rank=3)
    actual = _mpo_to_dense_matrix(mpo)

    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    expected = np.zeros_like(actual)
    for state in np.ndindex(4, 4, 4):
        energy = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                energy += mol.eri_j[i][j][0, 0] * occupation[state[i]] * occupation[state[j]]
        idx = np.ravel_multi_index(state, (4, 4, 4))
        expected[idx, idx] = energy

    np.testing.assert_allclose(actual, expected, atol=1.0e-12)
    assert info["full_kernel_rel_error"] < 1.0e-12


def test_hybrid_density_phase_exposes_residual_fit():
    mol = _ThreeSiteToyGDVRMolecule()

    density = GDVRSpatialHybridDensityPhase(
        mol,
        dt=0.01,
        prony_rank=1,
        residual_rank=3,
    )

    assert density.fit_info["residual_retained_rank"] > 0
    assert density.fit_info["residual_retained_rank"] <= 3
    assert density.fit_info["full_kernel_rel_error"] < 1.0e-12


def test_gdvr_tddmrg_runs_against_direct_mpo():
    td = TDDMRG(_ToyGDVRRHF()).build()
    td.run(dt=0.01, steps=2, e_ops=["mu_z"], D=8)

    np.testing.assert_allclose(td.times, [0.01, 0.02])
    assert td.observables.shape == (2, 1)
    assert td.site == "spatial"
    assert td.build_info["representation"] == "gdvr_direct_spatial_mpo"
    np.testing.assert_allclose(td.pre_normalization_norms, np.ones(2), atol=1.0e-12)
    assert td.static_energies.shape == (3,)

    reversal = td.time_reversal_error(dt=0.01, steps=2, D=8)
    assert reversal["state_error"] < 1.0e-10


def test_gdvr_tddmrg_compressed_taylor_path_densifies_block_mpo():
    mf = _ToyGDVRRHF()
    td = TDDMRG(mf).build()
    td._use_exact_dense_td = lambda: False
    psi0 = rhf_determinant_mps(mf, max_bond=16)
    h_dense = _mpo_to_dense_matrix(td._get_td_hamiltonian())
    vec0 = np.asarray(tt_to_tensor(psi0.factors), dtype=complex).reshape(-1)

    td.run(
        psi0=psi0,
        dt=0.01,
        steps=2,
        e_ops=[],
        integrator="taylor",
        order=4,
        scale=2,
        D=16,
        progress=False,
    )

    expected = expm(-0.02j * h_dense) @ vec0
    actual = np.asarray(tt_to_tensor(td.final_state.factors), dtype=complex).reshape(-1)
    np.testing.assert_allclose(actual, expected, atol=1.0e-10, rtol=1.0e-10)
    np.testing.assert_allclose(td.pre_normalization_norms, np.ones(2), atol=1.0e-10)


def test_gdvr_tddmrg_accepts_direct_force_mpo_observable():
    mf = _ToyGDVRRHF()
    td = TDDMRG(mf).build()
    force = force_mpo(mf.mol)
    foreign_mpo = SimpleNamespace(factors=force.factors)

    td.run(
        dt=0.01,
        steps=1,
        e_ops=[force, foreign_mpo],
        track_energy=False,
        progress=False,
        D=8,
    )

    assert td.observables.shape == (1, 2)
    np.testing.assert_allclose(td.observables[0, 0], td.observables[0, 1], atol=1.0e-12)


def test_gdvr_tddmrg_cap_is_real_time_only_and_temporary():
    mf = _ToyGDVRRHF()
    td = TDDMRG(mf).build()
    h0 = _mpo_to_dense_matrix(td._get_td_hamiltonian())

    td.set_cap(width=0.5, strength=0.4, order=2)
    h_cap = _mpo_to_dense_matrix(td._get_td_hamiltonian())
    assert np.min(np.imag(np.diag(h_cap - h0))) < 0.0

    td.clear_cap()
    np.testing.assert_allclose(_mpo_to_dense_matrix(td._get_td_hamiltonian()), h0, atol=1.0e-12)

    td.run(
        dt=0.1,
        steps=1,
        e_ops=[],
        cap={"width": 0.5, "strength": 0.4, "order": 2},
        track_energy=False,
        progress=False,
        D=8,
    )
    assert td.cap_settings is None
    assert td._cap_mpo is None
    assert not np.allclose(td.pre_normalization_norms, np.ones_like(td.pre_normalization_norms))
    np.testing.assert_allclose(_mpo_to_dense_matrix(td._get_td_hamiltonian()), h0, atol=1.0e-12)


def test_gdvr_tddmrg_omitted_psi0_is_dmrg_ground_state_and_init_guess_is_not_public():
    mf = _ToyGDVRRHF()
    angle = 0.37
    c = np.cos(angle)
    s = np.sin(angle)
    mf.mo_coeff = np.array([[c, -s], [s, c]])
    mf.dm = 2.0 * mf.mo_coeff[:, :1] @ mf.mo_coeff[:, :1].T

    with pytest.raises(TypeError):
        TDDMRG(mf, init_guess="random")

    td = TDDMRG(mf).build()
    actual = td.default_initial_condition(D=8)
    assert td._has_ground_state()
    expected = td.export_ground_state(dense=True)
    rhf = rhf_determinant_mps(mf, max_bond=8)
    product = MPS(td.get_initial_guess_dense(noise=0.0), labels=["lv", "p", "rv"]).normalize()

    actual_vec = np.asarray(tt_to_tensor(actual.factors), dtype=complex).reshape(-1)
    expected_vec = np.asarray(tt_to_tensor(expected.factors), dtype=complex).reshape(-1)
    rhf_vec = np.asarray(tt_to_tensor(rhf.factors), dtype=complex).reshape(-1)
    product_vec = np.asarray(tt_to_tensor(product.factors), dtype=complex).reshape(-1)

    overlap = abs(np.vdot(expected_vec, actual_vec))
    rhf_overlap = abs(np.vdot(rhf_vec, actual_vec))
    product_overlap = abs(np.vdot(product_vec, actual_vec))
    np.testing.assert_allclose(overlap, 1.0, atol=1.0e-12)
    assert rhf_overlap < 0.99
    assert product_overlap < 0.99


def test_gdvr_tddmrg_constructor_does_not_eagerly_build_rhf_mps(monkeypatch):
    import pyqed.qchem.gdvr.tddmrg as gdvr_tddmrg_module

    def _fail_rhf_mps(*args, **kwargs):
        raise AssertionError("constructor should keep a lightweight DMRG initial guess")

    monkeypatch.setattr(gdvr_tddmrg_module, "rhf_determinant_mps", _fail_rhf_mps)

    td = TDDMRG(_ToyGDVRRHF()).build()

    assert td.init_guess == "hf"
    assert not td._has_ground_state()


def test_gdvr_tddmrg_optimize_ground_state_uses_fast_defaults(monkeypatch):
    import pyqed.qchem.gdvr.tddmrg as gdvr_tddmrg_module

    captured = {}

    def _capture_optimize(self, *args, **kwargs):
        captured.update(kwargs)
        return self

    monkeypatch.setattr(
        gdvr_tddmrg_module.BaseTDDMRG,
        "optimize_ground_state",
        _capture_optimize,
    )

    td = TDDMRG(_ToyGDVRRHF()).build()

    assert td.optimize_ground_state(D=6) is td
    assert captured["D"] == 6
    assert captured["nsweeps"] == 4
    assert captured["initial_guess"] == "hf"
    assert captured["symmetry_list"] == ["charge", "sz"]
    assert captured["compute_s2"] is False
    assert captured["abelian_matvec_options"]["native_site_storage"] is True

    captured.clear()
    assert td.optimize_ground_state(D=6, symmetry=False) is td
    assert captured["symmetry"] is False
    assert "symmetry_list" not in captured
    assert "abelian_matvec_options" not in captured


def test_direct_gdvr_tddmrg_build_reuses_native_mpo_for_dmrg():
    td = TDDMRG(_ToyGDVRRHF()).build()

    assert td._symmetric_mpo_cache[(("charge", "sz"), "native")] is td.H
    assert td.build_info["native_abelian_mpo"] is True


def test_direct_gdvr_tddmrg_optimize_ground_state_tiny_native_setup():
    td = TDDMRG(_ToyGDVRRHF()).build()

    td.optimize_ground_state(
        D=2,
        nsweeps=1,
        symmetry_list=["charge", "sz"],
        compute_s2=False,
        davidson_tol=1.0e-4,
        not_conv_err=False,
    )

    assert td._has_ground_state()
    assert np.isfinite(td.e_tot)
    assert hasattr(td.dmrg.ground_state.factors[0], "qns")


def test_gdvr_tddmrg_dense_export_preserves_spatial_local_order():
    td = TDDMRG(_ToyGDVRRHF()).build()
    state, site_qn_maps = _one_site_spatial_symmetric_state()
    td.dmrg = SimpleNamespace(states=[state], site_qn_maps=site_qn_maps)

    dense = td.export_ground_state(dense=True)

    np.testing.assert_allclose(dense.factors[0][0, :, 0], [0.0, 3.0, 7.0, 0.0])


def test_gdvr_tddmrg_ensure_dense_preserves_spatial_local_order():
    td = TDDMRG(_ToyGDVRRHF()).build()
    state, site_qn_maps = _one_site_spatial_symmetric_state()
    td.dmrg = SimpleNamespace(site_qn_maps=site_qn_maps)

    dense = td._ensure_dense_mps(state)

    np.testing.assert_allclose(dense.factors[0][0, :, 0], [0.0, 3.0, 7.0, 0.0])


def test_direct_gdvr_tddmrg_defaults_to_block_sparse_tdvp(monkeypatch):
    import pyqed.qchem.dmrg.tddmrg as qchem_tddmrg_module
    import pyqed.mps.tdvp as tdvp_module

    def _fail_densify(*args, **kwargs):
        raise AssertionError("block-sparse GDVR-TDDMRG should keep the QN MPS initial state")

    def _fail_dense_mpo_conversion(*args, **kwargs):
        raise AssertionError("direct GDVR Hamiltonian should already be a native Abelian MPO")

    monkeypatch.setattr(qchem_tddmrg_module, "symmetric_to_dense", _fail_densify)
    monkeypatch.setattr(tdvp_module, "dense_to_symmetric_mpo", _fail_dense_mpo_conversion)
    td = TDDMRG(_ToyGDVRRHF()).build()
    td._use_exact_dense_td = lambda: False

    td.run(
        dt=0.01,
        steps=1,
        e_ops=[],
        integrator="tdvp",
        measure_observables=False,
        track_energy=False,
        progress=False,
        D=4,
    )

    assert td.tdmps.projection == "block-sparse"
    assert hasattr(td.final_state.factors[0], "qns")
    np.testing.assert_allclose(td.pre_normalization_norms, np.ones(1), atol=1.0e-12)


def test_direct_gdvr_tddmrg_block_sparse_field_uses_local_phase():
    mf = _ToyGDVRRHF()
    td = TDDMRG(mf).build()
    td._use_exact_dense_td = lambda: False
    field = lambda t: np.array([0.0, 0.0, 0.02 * np.sin(t)])

    td.run(
        dt=0.01,
        steps=1,
        field=field,
        e_ops=[],
        cap={"width": 0.5, "strength": 0.4, "order": 2},
        cap_mode="local-phase",
        measure_observables=False,
        track_energy=False,
        progress=False,
        D=4,
    )

    assert td.tdmps.projection == "block-sparse"
    assert hasattr(td.final_state.factors[0], "qns")
    assert td.tdmps.tdvp_split_dynamic_block_sparse is True
    assert td.cap_settings is None
    assert td._local_cap_values is None
    assert td.pre_normalization_norms[0] < 1.0
