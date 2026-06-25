import numpy as np
import pytest
from types import SimpleNamespace
from scipy.linalg import expm

from pyqed.mps import MPS
from pyqed.mps.decompose import decompose, tt_to_tensor
from pyqed.mps.symmetry import AbelianSector, BlockTensor
from pyqed.qchem.dmrg.dmrg import (
    _build_spatial_active_hamiltonian_matrix,
    _build_spin_orbital_dense_hamiltonian_tensor_mpo,
)
from pyqed.qchem.dmrg.tddmrg import _mpo_to_dense_matrix
from pyqed.qchem.gdvr import (
    TDDMRG,
    active_eri_from_gdvr_collocation,
    apply_gdvr_spatial_density_phase,
    apply_gdvr_spatial_one_body_rotation,
    build_gdvr_spatial_factorized_density_phase_mpo,
    build_gdvr_spatial_exponential_density_hamiltonian_mpo,
    build_gdvr_dipole_mpo,
    build_gdvr_hamiltonian_mpo,
    build_gdvr_spatial_density_phase_mpo,
    build_gdvr_spatial_dipole_mpo,
    build_gdvr_spatial_hamiltonian_mpo,
    build_gdvr_spatial_one_body_rotation_mpo,
    build_gdvr_spatial_prony_density_hamiltonian_mpo,
    build_gdvr_spatial_svd_density_hamiltonian_mpo,
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


def test_direct_gdvr_dipole_mpo_is_diagonal_number_operator():
    mol = _ToyGDVRMolecule()
    mu = _mpo_to_dense_matrix(build_gdvr_dipole_mpo(mol))

    expected = np.zeros_like(mu)
    z = -np.repeat(mol.z, 2)
    for state in range(mu.shape[0]):
        occ = np.asarray(np.unravel_index(state, (2, 2, 2, 2)))
        expected[state, state] = float(np.dot(z, occ))

    np.testing.assert_allclose(mu, expected, atol=1.0e-12)


def test_direct_spatial_gdvr_dipole_mpo_is_site_number_operator():
    mol = _ToyGDVRMolecule()
    mu = _mpo_to_dense_matrix(build_gdvr_spatial_dipole_mpo(mol))

    expected = np.zeros_like(mu)
    occupation = np.array([0.0, 1.0, 1.0, 2.0])
    z = -mol.z
    for state in range(mu.shape[0]):
        local_states = np.asarray(np.unravel_index(state, (4, 4)))
        expected[state, state] = float(np.dot(z, occupation[local_states]))

    np.testing.assert_allclose(mu, expected, atol=1.0e-12)


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
    td = TDDMRG(_ToyGDVRRHF(), D=8).build()
    td.run(dt=0.01, steps=2, e_ops=["mu_z"])

    np.testing.assert_allclose(td.times, [0.01, 0.02])
    assert td.observables.shape == (2, 1)
    assert td.site == "spatial"
    assert td._active_integral_build_info["representation"] == "gdvr_direct_spatial_mpo"
    np.testing.assert_allclose(td.pre_normalization_norms, np.ones(2), atol=1.0e-12)
    assert td.static_energies.shape == (3,)

    reversal = td.time_reversal_error(dt=0.01, steps=2)
    assert reversal["state_error"] < 1.0e-10


def test_gdvr_tddmrg_omitted_psi0_is_rhf_determinant_and_init_guess_is_not_public():
    mf = _ToyGDVRRHF()
    angle = 0.37
    c = np.cos(angle)
    s = np.sin(angle)
    mf.mo_coeff = np.array([[c, -s], [s, c]])
    mf.dm = 2.0 * mf.mo_coeff[:, :1] @ mf.mo_coeff[:, :1].T

    with pytest.raises(TypeError):
        TDDMRG(mf, D=8, init_guess="random")

    td = TDDMRG(mf, D=8).build()
    actual = td._default_initial_state()
    expected = rhf_determinant_mps(mf, max_bond=8)
    product = MPS(td.get_initial_guess_dense(noise=0.0), labels=["lv", "p", "rv"]).normalize()

    actual_vec = np.asarray(tt_to_tensor(actual.factors), dtype=complex).reshape(-1)
    expected_vec = np.asarray(tt_to_tensor(expected.factors), dtype=complex).reshape(-1)
    product_vec = np.asarray(tt_to_tensor(product.factors), dtype=complex).reshape(-1)

    overlap = abs(np.vdot(expected_vec, actual_vec))
    product_overlap = abs(np.vdot(product_vec, actual_vec))
    np.testing.assert_allclose(overlap, 1.0, atol=1.0e-12)
    assert product_overlap < 0.99


def test_gdvr_tddmrg_dense_export_preserves_spatial_local_order():
    td = TDDMRG(_ToyGDVRRHF(), D=8).build()
    state, site_qn_maps = _one_site_spatial_symmetric_state()
    td.dmrg = SimpleNamespace(states=[state], site_qn_maps=site_qn_maps)

    dense = td.export_initial_guess(dense=True)

    np.testing.assert_allclose(dense.factors[0][0, :, 0], [0.0, 3.0, 7.0, 0.0])


def test_gdvr_tddmrg_ensure_dense_preserves_spatial_local_order():
    td = TDDMRG(_ToyGDVRRHF(), D=8).build()
    state, site_qn_maps = _one_site_spatial_symmetric_state()
    td.dmrg = SimpleNamespace(site_qn_maps=site_qn_maps)

    dense = td._ensure_dense_mps(state)

    np.testing.assert_allclose(dense.factors[0][0, :, 0], [0.0, 3.0, 7.0, 0.0])
