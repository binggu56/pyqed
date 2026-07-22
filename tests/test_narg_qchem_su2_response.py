import numpy as np
import pytest
from scipy.linalg import eigh
from types import SimpleNamespace

from pyqed import Molecule
from pyqed.narg.irrep_tensor import Irrep
from pyqed.narg.qchem import NARGSCF
from pyqed.narg.qchem.su2 import NARG
from pyqed.narg.qchem.su2_two_site import build_two_site_su2_narg, truncate_to_D
from pyqed.narg.qchem.su2_response import (
    active_pair_kappa,
    active_symmetric_integral_coefficients,
    active_symmetric_adjoint_arrays_from_values,
    active_symmetric_pair_response_matrix,
    cas_integral_response_from_full,
    cas_integral_response_from_pair,
    cas_integral_response_from_pairs,
    _component_v1_packages_tangent,
    _component_v1_packages_tangent_adjoint,
    coupled_reduced_product_adjoint,
    direct_reduced_full_hamiltonian_tangent_adjoint,
    _grown_coupling_operators_tangent,
    _grown_coupling_operators_tangent_adjoint,
    _grown_hamiltonian_tangent,
    _grown_hamiltonian_tangent_adjoint,
    _grown_reduced_v1_packages_tangent,
    _grown_reduced_v1_packages_tangent_adjoint,
    _grown_reduced_v1_packages_bilinear,
    _grown_reduced_v1_packages_bilinear_adjoint_x,
    _new_site_weighted_packages_bilinear,
    _new_site_weighted_packages_bilinear_adjoint_x,
    _new_site_weighted_packages_tangent,
    _new_site_weighted_packages_tangent_adjoint,
    _density_tangent,
    _density_tangent_adjoint,
    _grown_hamiltonian_bilinear,
    _pair_annihilate_tangent,
    _pair_annihilate_tangent_adjoint,
    _pre_rotation_tensors_and_tangents_adjoint,
    _pre_rotation_tensors_and_bilinears,
    _pre_rotation_tensors_and_bilinears_adjoint_x,
    _recursive_growth_step_tangent_adjoint,
    _rotate_reduced_tensors_bilinear,
    _seed_two_site_bilinear_block,
    _seed_two_site_tangent_adjoint,
    _spin_density_tangent,
    _spin_density_tangent_adjoint,
    _terminal_block_from_recursive_tangent_path,
    _weighted_packages_tangent,
    _weighted_packages_tangent_adjoint,
    hamiltonian_block_from_density,
    density_operator_blocks,
    recursive_active_integral_response_basis,
    recursive_active_integral_adjoint_from_path,
    recursive_active_integral_adjoint_arrays,
    recursive_active_integral_adjoint_coefficients,
    recursive_bilinear_active_integral_adjoint_arrays_x,
    recursive_bilinear_active_integral_adjoint_coefficients,
    recursive_perturbation_for_active_integrals,
    recursive_tangent_path_for_active_integrals,
    recursive_response_block_from_active_basis,
    recursive_response_pair_components_from_active_basis,
    reduced_product_tensor_block_adjoint,
    rotate_irrep_tensor_tangent_adjoint,
    rotate_irrep_tensor_bilinear,
    rotate_irrep_tensor_bilinear_adjoint_x,
    rotate_irrep_tensor_tangent,
    rotate_reduced_tensors_bilinear_adjoint_x,
    rotate_reduced_tensors_tangent_adjoint,
    solve_terminal_response,
    truncation_bilinear_tangent,
    truncation_bilinear_tangent_adjoint_x,
    truncation_tangent_adjoint,
    truncation_tangent,
)
from pyqed.qchem.mcscf.casci import h1e_for_cas
from pyqed.qchem.mcscf.orbopt import rotate_orbitals


class DummyMol:
    def __init__(self, nelec, spin):
        self.nelec = (nelec // 2, nelec // 2)
        self.spin = int(spin)

    def energy_nuc(self):
        return 0.0


def _physical_random_integrals(nsites: int, *, seed: int):
    rng = np.random.default_rng(seed)
    h1e = rng.normal(scale=0.3, size=(nsites, nsites))
    h1e = 0.5 * (h1e + h1e.T)
    eri = rng.normal(scale=0.04, size=(nsites, nsites, nsites, nsites))
    eri = 0.25 * (
        eri
        + eri.swapaxes(0, 1)
        + eri.swapaxes(2, 3)
        + eri.swapaxes(0, 1).swapaxes(2, 3)
    )
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return h1e, eri


def _antihermitian_pair(nmo, pair, value=1.0):
    kappa = np.zeros((int(nmo), int(nmo)))
    p, q = pair
    kappa[p, q] = float(value)
    kappa[q, p] = -float(value)
    return kappa


def _effective_cas_integrals_from_full(h1_mo, eri_mo, *, ncore: int, ncas: int):
    active = slice(int(ncore), int(ncore) + int(ncas))
    h1e = np.array(h1_mo[active, active], copy=True)
    if ncore:
        core = slice(0, int(ncore))
        h1e += 2.0 * np.einsum(
            "abii->ab",
            eri_mo[active, active, core, core],
            optimize=True,
        )
        h1e -= np.einsum(
            "aiib->ab",
            eri_mo[active, core, core, active],
            optimize=True,
        )
    return h1e, np.array(eri_mo[active, active, active, active], copy=True)


class ToyRestrictedMF:
    def __init__(self, h1e, eri, *, nelec, spin):
        self.h1e = np.asarray(h1e)
        self.eri = np.asarray(eri)
        self.mo_coeff = np.eye(self.h1e.shape[0])
        self.nmo = self.h1e.shape[0]
        self.nelec = nelec
        self.mol = DummyMol(int(np.sum(np.asarray(nelec))), spin)

    def get_hcore(self):
        return self.h1e

    def energy_nuc(self):
        return 0.0

    def get_hcore_mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        mo_coeff = np.asarray(mo_coeff)
        return mo_coeff.conj().T @ self.h1e @ mo_coeff

    def get_eri_mo(self, mo_coeff=None, notation="chem"):
        del notation
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        mo_coeff = np.asarray(mo_coeff)
        return np.einsum(
            "ip,jq,ijkl,kr,ls->pqrs",
            mo_coeff.conj(),
            mo_coeff,
            self.eri,
            mo_coeff.conj(),
            mo_coeff,
            optimize=True,
        )


def _align_columns(reference, trial):
    trial = np.array(trial, copy=True)
    for col in range(trial.shape[1]):
        overlap = np.vdot(reference[:, col], trial[:, col])
        if overlap.real < 0.0:
            trial[:, col] *= -1.0
    return trial


def _irrep_tensor_pairing(left, right) -> float:
    total = 0.0
    for key in set(left.blocks) | set(right.blocks):
        lblock = left.block(*key)
        rblock = right.block(*key)
        total += float(np.real(np.vdot(lblock, rblock)))
    return total


def _transform_pairing(left_blocks, right_blocks) -> float:
    total = 0.0
    for key in set(left_blocks) | set(right_blocks):
        left = left_blocks.get(key)
        right = right_blocks.get(key)
        if left is None:
            left = np.zeros_like(right)
        if right is None:
            right = np.zeros_like(left)
        total += float(np.real(np.vdot(left, right)))
    return total


def _reduced_operator_pairing(left_ops, right_ops) -> float:
    total = 0.0
    for key in set(left_ops) | set(right_ops):
        left = left_ops.get(key)
        right = right_ops.get(key)
        if left is None or right is None:
            continue
        total += _irrep_tensor_pairing(left.tensor, right.tensor)
    return total


def test_terminal_response_solves_projected_tangent_equation():
    hamiltonian = np.diag([0.0, 1.0, 3.0])
    perturbation = np.array(
        [
            [0.25, 2.0, -3.0],
            [2.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
        ]
    )
    psi = np.array([1.0, 0.0, 0.0])

    response = solve_terminal_response(hamiltonian, psi, perturbation)

    np.testing.assert_allclose(response.vector, [0.0, -2.0, 1.0], atol=1.0e-12)
    np.testing.assert_allclose(np.vdot(psi, response.vector), 0.0, atol=1.0e-12)
    assert response.residual_norm < 1.0e-12
    assert response.first_order_energy == 0.25


def test_truncation_tangent_matches_two_site_finite_difference():
    h1e, eri = _physical_random_integrals(2, seed=41)
    dh1, deri = _physical_random_integrals(2, seed=43)
    irrep = Irrep((2, 0))
    narg = build_two_site_su2_narg(h1e, eri)
    dnarg = build_two_site_su2_narg(dh1, deri)
    truncated = truncate_to_D(narg, D=1, allowed_irreps={irrep}, backend="python")

    tangent = truncation_tangent(narg, truncated, dnarg.hamiltonian)

    eps = 1.0e-6
    plus = truncate_to_D(
        build_two_site_su2_narg(h1e + eps * dh1, eri + eps * deri),
        D=1,
        allowed_irreps={irrep},
        backend="python",
    )
    minus = truncate_to_D(
        build_two_site_su2_narg(h1e - eps * dh1, eri - eps * deri),
        D=1,
        allowed_irreps={irrep},
        backend="python",
    )
    base_u = truncated.transform.block(irrep, irrep)
    plus_u = _align_columns(base_u, plus.transform.block(irrep, irrep))
    minus_u = _align_columns(base_u, minus.transform.block(irrep, irrep))

    np.testing.assert_allclose(
        tangent.d_transform_blocks[(irrep, irrep)],
        (plus_u - minus_u) / (2.0 * eps),
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        tangent.d_hamiltonian.block(irrep, irrep),
        (
            plus.hamiltonian.block(irrep, irrep)
            - minus.hamiltonian.block(irrep, irrep)
        )
        / (2.0 * eps),
        atol=1.0e-9,
    )


def test_truncation_tangent_adjoint_matches_forward_pairing():
    h1e, eri = _physical_random_integrals(2, seed=44)
    dh1, deri = _physical_random_integrals(2, seed=45)
    irrep = Irrep((2, 0))
    narg = build_two_site_su2_narg(h1e, eri)
    dnarg = build_two_site_su2_narg(dh1, deri)
    truncated = truncate_to_D(narg, D=2, allowed_irreps={irrep}, backend="python")
    tangent = truncation_tangent(narg, truncated, dnarg.hamiltonian)

    rng = np.random.default_rng(46)
    transform_adjoint = {}
    for key, block in tangent.d_transform_blocks.items():
        transform_adjoint[key] = rng.normal(size=block.shape) + 1j * rng.normal(
            size=block.shape
        )
    h_adj_blocks = {}
    for key, block in tangent.d_hamiltonian.blocks.items():
        raw = rng.normal(size=block.shape)
        h_adj_blocks[key] = 0.5 * (raw + raw.T)
    hamiltonian_adjoint = type(tangent.d_hamiltonian)(
        tangent.d_hamiltonian.bra,
        tangent.d_hamiltonian.ket,
        tangent.d_hamiltonian.op,
        h_adj_blocks,
    )

    adjoint = truncation_tangent_adjoint(
        narg,
        truncated,
        transform_adjoint_blocks=transform_adjoint,
        hamiltonian_adjoint=hamiltonian_adjoint,
    )

    lhs = _transform_pairing(transform_adjoint, tangent.d_transform_blocks)
    lhs += _irrep_tensor_pairing(hamiltonian_adjoint, tangent.d_hamiltonian)
    rhs = _irrep_tensor_pairing(adjoint, dnarg.hamiltonian)

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_projected_hamiltonian_tangent_matches_truncation_energy_response():
    h1e, eri = _physical_random_integrals(2, seed=51)
    dh1, deri = _physical_random_integrals(2, seed=53)
    irrep = Irrep((2, 0))
    narg = build_two_site_su2_narg(h1e, eri)
    dnarg = build_two_site_su2_narg(dh1, deri)
    truncated = truncate_to_D(narg, D=1, allowed_irreps={irrep}, backend="python")
    tangent = truncation_tangent(narg, truncated, dnarg.hamiltonian)

    projected = rotate_irrep_tensor_tangent(
        truncated,
        narg.hamiltonian,
        dnarg.hamiltonian,
        tangent,
    )

    np.testing.assert_allclose(
        projected.block(irrep, irrep),
        tangent.d_hamiltonian.block(irrep, irrep),
        atol=1.0e-10,
    )


def test_rotate_irrep_tensor_tangent_adjoint_matches_forward_pairing():
    h1e, eri = _physical_random_integrals(2, seed=56)
    dh1, deri = _physical_random_integrals(2, seed=57)
    irrep = Irrep((2, 0))
    narg = build_two_site_su2_narg(h1e, eri)
    dnarg = build_two_site_su2_narg(dh1, deri)
    truncated = truncate_to_D(narg, D=2, allowed_irreps={irrep}, backend="python")
    tangent = truncation_tangent(narg, truncated, dnarg.hamiltonian)
    rotated = rotate_irrep_tensor_tangent(
        truncated,
        narg.hamiltonian,
        dnarg.hamiltonian,
        tangent,
    )

    rng = np.random.default_rng(58)
    adj_blocks = {}
    for key, block in rotated.blocks.items():
        raw = rng.normal(size=block.shape)
        adj_blocks[key] = 0.5 * (raw + raw.T)
    rotated_adjoint = type(rotated)(
        rotated.bra,
        rotated.ket,
        rotated.op,
        adj_blocks,
    )

    operator_adjoint, transform_adjoint = rotate_irrep_tensor_tangent_adjoint(
        truncated,
        narg.hamiltonian,
        rotated_adjoint,
    )

    lhs = _irrep_tensor_pairing(rotated_adjoint, rotated)
    rhs = _irrep_tensor_pairing(operator_adjoint, dnarg.hamiltonian)
    rhs += _transform_pairing(transform_adjoint, tangent.d_transform_blocks)

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_rotate_irrep_tensor_bilinear_adjoint_x_matches_forward_pairing():
    h1e, eri = _physical_random_integrals(2, seed=59)
    xh1, xeri = _physical_random_integrals(2, seed=60)
    yh1, yeri = _physical_random_integrals(2, seed=61)
    xyh1, xyeri = _physical_random_integrals(2, seed=62)
    irrep = Irrep((2, 0))
    narg = build_two_site_su2_narg(h1e, eri)
    xnarg = build_two_site_su2_narg(xh1, xeri)
    ynarg = build_two_site_su2_narg(yh1, yeri)
    xynarg = build_two_site_su2_narg(xyh1, xyeri)
    truncated = truncate_to_D(narg, D=2, allowed_irreps={irrep}, backend="python")
    tangent_x = truncation_tangent(narg, truncated, xnarg.hamiltonian)
    tangent_y = truncation_tangent(narg, truncated, ynarg.hamiltonian)
    bilinear = truncation_bilinear_tangent(
        narg,
        truncated,
        xnarg.hamiltonian,
        ynarg.hamiltonian,
        xynarg.hamiltonian,
        tangent_x=tangent_x,
        tangent_y=tangent_y,
    )
    rotated = rotate_irrep_tensor_bilinear(
        truncated,
        narg.hamiltonian,
        xnarg.hamiltonian,
        ynarg.hamiltonian,
        xynarg.hamiltonian,
        bilinear,
    )

    rng = np.random.default_rng(63)
    adj_blocks = {}
    for key, block in rotated.blocks.items():
        raw = rng.normal(size=block.shape)
        adj_blocks[key] = 0.5 * (raw + raw.T)
    rotated_adjoint = type(rotated)(
        rotated.bra,
        rotated.ket,
        rotated.op,
        adj_blocks,
    )

    (
        operator_x_adjoint,
        operator_xy_adjoint,
        transform_x_adjoint,
        transform_xy_adjoint,
    ) = rotate_irrep_tensor_bilinear_adjoint_x(
        truncated,
        narg.hamiltonian,
        ynarg.hamiltonian,
        rotated_adjoint,
        bilinear,
    )

    lhs = _irrep_tensor_pairing(rotated_adjoint, rotated)
    rhs = _irrep_tensor_pairing(operator_x_adjoint, xnarg.hamiltonian)
    rhs += _irrep_tensor_pairing(operator_xy_adjoint, xynarg.hamiltonian)
    rhs += _transform_pairing(transform_x_adjoint, tangent_x.d_transform_blocks)
    rhs += _transform_pairing(transform_xy_adjoint, bilinear.dxy_transform_blocks)

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_truncation_bilinear_tangent_adjoint_x_matches_forward_pairing():
    h1e, eri = _physical_random_integrals(2, seed=64)
    xh1, xeri = _physical_random_integrals(2, seed=65)
    yh1, yeri = _physical_random_integrals(2, seed=66)
    xyh1, xyeri = _physical_random_integrals(2, seed=67)
    irrep = Irrep((2, 0))
    narg = build_two_site_su2_narg(h1e, eri)
    xnarg = build_two_site_su2_narg(xh1, xeri)
    ynarg = build_two_site_su2_narg(yh1, yeri)
    xynarg = build_two_site_su2_narg(xyh1, xyeri)
    truncated = truncate_to_D(narg, D=2, allowed_irreps={irrep}, backend="python")
    tangent_x = truncation_tangent(narg, truncated, xnarg.hamiltonian)
    tangent_y = truncation_tangent(narg, truncated, ynarg.hamiltonian)
    bilinear = truncation_bilinear_tangent(
        narg,
        truncated,
        xnarg.hamiltonian,
        ynarg.hamiltonian,
        xynarg.hamiltonian,
        tangent_x=tangent_x,
        tangent_y=tangent_y,
    )

    rng = np.random.default_rng(68)
    transform_x_adjoint = {}
    for key, block in tangent_x.d_transform_blocks.items():
        transform_x_adjoint[key] = rng.normal(size=block.shape) + 1j * rng.normal(
            size=block.shape
        )
    transform_xy_adjoint = {}
    for key, block in bilinear.dxy_transform_blocks.items():
        transform_xy_adjoint[key] = rng.normal(size=block.shape) + 1j * rng.normal(
            size=block.shape
        )
    hx_adj_blocks = {}
    for key, block in tangent_x.d_hamiltonian.blocks.items():
        raw = rng.normal(size=block.shape)
        hx_adj_blocks[key] = 0.5 * (raw + raw.T)
    hxy_adj_blocks = {}
    for key, block in bilinear.dxy_hamiltonian.blocks.items():
        raw = rng.normal(size=block.shape)
        hxy_adj_blocks[key] = 0.5 * (raw + raw.T)
    hamiltonian_x_adjoint = type(tangent_x.d_hamiltonian)(
        tangent_x.d_hamiltonian.bra,
        tangent_x.d_hamiltonian.ket,
        tangent_x.d_hamiltonian.op,
        hx_adj_blocks,
    )
    hamiltonian_xy_adjoint = type(bilinear.dxy_hamiltonian)(
        bilinear.dxy_hamiltonian.bra,
        bilinear.dxy_hamiltonian.ket,
        bilinear.dxy_hamiltonian.op,
        hxy_adj_blocks,
    )

    perturbation_x_adjoint, perturbation_xy_adjoint = (
        truncation_bilinear_tangent_adjoint_x(
            narg,
            truncated,
            bilinear,
            ynarg.hamiltonian,
            transform_x_adjoint_blocks=transform_x_adjoint,
            transform_xy_adjoint_blocks=transform_xy_adjoint,
            hamiltonian_x_adjoint=hamiltonian_x_adjoint,
            hamiltonian_xy_adjoint=hamiltonian_xy_adjoint,
        )
    )

    lhs = _transform_pairing(transform_x_adjoint, tangent_x.d_transform_blocks)
    lhs += _irrep_tensor_pairing(hamiltonian_x_adjoint, tangent_x.d_hamiltonian)
    lhs += _transform_pairing(transform_xy_adjoint, bilinear.dxy_transform_blocks)
    lhs += _irrep_tensor_pairing(hamiltonian_xy_adjoint, bilinear.dxy_hamiltonian)
    rhs = _irrep_tensor_pairing(perturbation_x_adjoint, xnarg.hamiltonian)
    rhs += _irrep_tensor_pairing(perturbation_xy_adjoint, xynarg.hamiltonian)

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_density_hamiltonian_block_matches_complete_su2_sector():
    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=31)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=80,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
    ).run()
    vector = solver.root_vectors[:, 0]
    density = density_operator_blocks(
        solver.chain.final,
        vector=vector,
        nelec=nelec,
        j2=j2,
        site_count=nsites,
    )

    rebuilt = hamiltonian_block_from_density(density, h1e, eri)

    np.testing.assert_allclose(rebuilt, solver.block, atol=1.0e-12)


@pytest.mark.parametrize("bond_dim", [4, 8])
@pytest.mark.parametrize("project_v1_packages", [False, True])
def test_recursive_perturbation_matches_retained_energy_derivative(
    bond_dim, project_v1_packages
):
    nsites = 3
    nelec = 3
    j2 = 1
    h1e, eri = _physical_random_integrals(nsites, seed=71)
    dh1, deri = _physical_random_integrals(nsites, seed=73)
    mol = DummyMol(nelec, j2)

    def run(h, v):
        return NARG(
            SimpleNamespace(mol=mol),
            mol=mol,
            h1e=h,
            eri=v,
            D=bond_dim,
            nstates=1,
            final_size=nsites,
            target_nelec=nelec,
            target_j2=j2,
            su2_backend="python",
            project_v1_packages=project_v1_packages,
            carry_rdm_operators=True,
        ).run()

    solver = run(h1e, eri)
    perturbation = recursive_perturbation_for_active_integrals(solver, dh1, deri)
    vector = solver.root_vectors[:, 0]
    analytic = np.vdot(vector, perturbation.block @ vector) / np.vdot(vector, vector)

    eps = 1.0e-5
    finite_difference = (
        run(h1e + eps * dh1, eri + eps * deri).e_tot[0]
        - run(h1e - eps * dh1, eri - eps * deri).e_tot[0]
    ) / (2.0 * eps)

    np.testing.assert_allclose(analytic, finite_difference, atol=1.0e-10)


def test_recursive_two_site_perturbation_keeps_final_offdiagonal_couplings():
    nsites = 2
    nelec = 2
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=76)
    dh1, deri = _physical_random_integrals(nsites, seed=77)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=8,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
    ).run()

    perturbation = recursive_perturbation_for_active_integrals(solver, dh1, deri)
    psi = solver.root_vectors[:, 0]
    density = density_operator_blocks(
        solver.chain.final,
        vector=psi,
        nelec=nelec,
        j2=j2,
        site_count=nsites,
    )
    fixed_basis = hamiltonian_block_from_density(density, dh1, deri)

    np.testing.assert_allclose(perturbation.block, fixed_basis, atol=1.0e-12)
    offdiag = fixed_basis @ psi - psi * np.vdot(psi, fixed_basis @ psi)
    assert np.linalg.norm(offdiag) > 1.0e-10


def test_direct_reduced_full_hamiltonian_tangent_adjoint_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor, OpIrrep
    from pyqed.narg.qchem.su2_three_site import direct_reduced_full_hamiltonian_tensor

    nsites = 3
    nelec = 3
    j2 = 1
    h1e, eri = _physical_random_integrals(nsites, seed=778)
    dh1, deri = _physical_random_integrals(nsites, seed=779)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=4,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    source = recursive_tangent_path_for_active_integrals(solver, dh1, deri).sources[2]
    zeros_h = np.zeros((nsites, nsites))
    zeros_g = np.zeros((nsites, nsites, nsites, nsites))
    forward = direct_reduced_full_hamiltonian_tensor(
        source,
        zeros_h,
        zeros_g,
        site_index=2,
    )

    rng = np.random.default_rng(780)
    adj_blocks = {}
    for key, block in forward.blocks.items():
        raw = rng.normal(size=block.shape)
        adj_blocks[key] = 0.5 * (raw + raw.T)
    scalar_adjoint = IrrepTensor(
        forward.bra,
        forward.ket,
        OpIrrep((0, 0)),
        adj_blocks,
    )
    h_adj, op_adj = direct_reduced_full_hamiltonian_tangent_adjoint(
        source,
        scalar_adjoint,
        site_index=2,
    )

    lhs = _irrep_tensor_pairing(scalar_adjoint, forward)
    rhs = _irrep_tensor_pairing(h_adj, source.hamiltonian)
    rhs += _reduced_operator_pairing(op_adj, source.reduced_operators)

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_grown_hamiltonian_tangent_adjoint_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=806)
    dh1, deri = _physical_random_integrals(nsites, seed=807)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=4,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    path = recursive_tangent_path_for_active_integrals(solver, dh1, deri)
    source_block = solver.chain.blocks[2]
    source_tangent = path.sources[2]
    h1e_n = h1e[:3, :3]
    eri_n = eri[:3, :3, :3, :3]
    dh1_n = dh1[:3, :3]
    deri_n = deri[:3, :3, :3, :3]
    forward = _grown_hamiltonian_tangent(
        source_block,
        source_tangent,
        h1e_n,
        dh1_n,
        eri_n,
        deri_n,
        site_index=2,
    )

    rng = np.random.default_rng(808)
    adj_blocks = {}
    for key, block in forward.blocks.items():
        raw = rng.normal(size=block.shape)
        adj_blocks[key] = 0.5 * (raw + raw.T)
    scalar_adjoint = IrrepTensor(
        forward.bra,
        forward.ket,
        forward.op,
        adj_blocks,
    )
    h_adj, op_adj, dh1_adj, deri_adj = _grown_hamiltonian_tangent_adjoint(
        source_block,
        source_tangent,
        scalar_adjoint,
        h1e_n,
        eri_n,
        site_index=2,
    )

    lhs = _irrep_tensor_pairing(scalar_adjoint, forward)
    rhs = _irrep_tensor_pairing(h_adj, source_tangent.hamiltonian)
    rhs += _reduced_operator_pairing(op_adj, source_tangent.reduced_operators)
    rhs += float(np.real(np.vdot(dh1_adj, dh1_n)))
    rhs += float(np.real(np.vdot(deri_adj, deri_n)))

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_reduced_product_tensor_block_adjoint_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor
    from pyqed.narg.qchem.su2_three_site import (
        local_reduced_operator,
        reduced_product_tensor_irrep,
    )

    nsites = 3
    nelec = 3
    j2 = 1
    h1e, eri = _physical_random_integrals(nsites, seed=781)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=4,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    block = solver.chain.blocks[2]
    block_tensor = block.reduced_operators[("Cdag", 0)]
    local_tensor = local_reduced_operator("JW")
    product = reduced_product_tensor_irrep(
        block,
        block_tensor,
        local_tensor,
        total_rank2=1,
    )

    rng = np.random.default_rng(782)
    adj_blocks = {}
    for key, value in product.blocks.items():
        adj_blocks[key] = rng.normal(size=value.shape) + 1j * rng.normal(
            size=value.shape
        )
    product_adjoint = ReducedSU2Tensor(
        IrrepTensor(product.site, product.site, product.op, adj_blocks)
    )

    got = reduced_product_tensor_block_adjoint(
        block,
        block_tensor,
        local_tensor,
        product_adjoint,
        total_rank2=1,
    )

    lhs = _irrep_tensor_pairing(product_adjoint.tensor, product.tensor)
    rhs = _irrep_tensor_pairing(got.tensor, block_tensor.tensor)

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_coupled_reduced_product_adjoint_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor, coupled_reduced_product

    nsites = 3
    nelec = 3
    j2 = 1
    h1e, eri = _physical_random_integrals(nsites, seed=789)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=4,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    block = solver.chain.blocks[2]
    left = block.reduced_operators[("Cdag", 0)]
    right = block.reduced_operators[("Ctilde", 1)]
    product = coupled_reduced_product(left, right, rank2=0, scale=np.sqrt(2.0))

    rng = np.random.default_rng(790)
    adj_blocks = {}
    for key, value in product.blocks.items():
        adj_blocks[key] = rng.normal(size=value.shape) + 1j * rng.normal(
            size=value.shape
        )
    product_adjoint = ReducedSU2Tensor(
        IrrepTensor(product.site, product.site, product.op, adj_blocks)
    )

    left_adj, right_adj = coupled_reduced_product_adjoint(
        left,
        right,
        product_adjoint,
        rank2=0,
        scale=np.sqrt(2.0),
    )

    lhs = _irrep_tensor_pairing(product_adjoint.tensor, product.tensor)
    left_rhs = _irrep_tensor_pairing(left_adj.tensor, left.tensor)
    right_rhs = _irrep_tensor_pairing(right_adj.tensor, right.tensor)

    np.testing.assert_allclose(left_rhs, lhs, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(right_rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_grown_coupling_operators_tangent_adjoint_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=786)
    dh1, deri = _physical_random_integrals(nsites, seed=787)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    path = recursive_tangent_path_for_active_integrals(solver, dh1, deri)
    narg = solver.chain.blocks[3].truncated.source
    source_tangent = path.sources[2]
    grown = _grown_coupling_operators_tangent(
        narg,
        source_tangent,
        include_even_composites=True,
    )

    rng = np.random.default_rng(788)
    grown_adjoint = {}
    for key, tensor in grown.items():
        blocks = {}
        for block_key, block in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=block.shape) + 1j * rng.normal(
                size=block.shape
            )
        grown_adjoint[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )

    got = _grown_coupling_operators_tangent_adjoint(
        narg,
        source_tangent,
        grown_adjoint,
        include_even_composites=True,
    )

    lhs = _reduced_operator_pairing(grown_adjoint, grown)
    rhs = _reduced_operator_pairing(got, source_tangent.reduced_operators)

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_new_site_weighted_packages_tangent_adjoint_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=794)
    dh1, deri = _physical_random_integrals(nsites, seed=795)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    path = recursive_tangent_path_for_active_integrals(solver, dh1, deri)
    parts = path.pre_rotation_parts[3]
    grown = dict(parts["grown"])
    dgrown = dict(parts["dgrown"])
    site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
    future_sites = tuple(range(site_count, nsites))

    actual, dweighted = _new_site_weighted_packages_tangent(
        grown,
        dgrown,
        h1e,
        dh1,
        eri,
        deri,
        site_count,
        future_sites,
        build_v1=False,
    )

    rng = np.random.default_rng(796)
    package_adjoint = {}
    for key, tensor in actual.items():
        blocks = {}
        for block_key, block in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=block.shape) + 1j * rng.normal(
                size=block.shape
            )
        package_adjoint[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )

    op_adj, dh1_adj, deri_adj = _new_site_weighted_packages_tangent_adjoint(
        grown,
        dgrown,
        package_adjoint,
        h1e,
        eri,
        site_count,
        future_sites,
    )

    lhs = _reduced_operator_pairing(package_adjoint, dweighted)
    rhs = _reduced_operator_pairing(op_adj, dgrown)
    rhs += float(np.real(np.vdot(dh1_adj, dh1)))
    rhs += float(np.real(np.vdot(deri_adj, deri)))

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_new_site_weighted_packages_bilinear_adjoint_x_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=838)
    xh1, xeri = _physical_random_integrals(nsites, seed=839)
    yh1, yeri = _physical_random_integrals(nsites, seed=840)
    xyh1, xyeri = _physical_random_integrals(nsites, seed=841)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    x_path = recursive_tangent_path_for_active_integrals(solver, xh1, xeri)
    y_path = recursive_tangent_path_for_active_integrals(solver, yh1, yeri)
    xy_path = recursive_tangent_path_for_active_integrals(solver, xyh1, xyeri)
    parts = x_path.pre_rotation_parts[3]
    yparts = y_path.pre_rotation_parts[3]
    xyparts = xy_path.pre_rotation_parts[3]
    grown = parts["grown"]
    xgrown = parts["dgrown"]
    ygrown = yparts["dgrown"]
    xygrown = xyparts["dgrown"]
    site_count = max(key[1] for key in grown if key[0] == "Cdag") + 1
    future_sites = (3,)
    _actual, forward = _new_site_weighted_packages_bilinear(
        grown,
        xgrown,
        ygrown,
        xygrown,
        h1e,
        xh1,
        yh1,
        xyh1,
        eri,
        xeri,
        yeri,
        xyeri,
        site_count,
        future_sites,
        build_v1=False,
        actual=parts["weighted"],
    )

    rng = np.random.default_rng(842)
    package_adjoint = {}
    for key, tensor in forward.items():
        blocks = {}
        for block_key, value in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=value.shape) + 1j * rng.normal(
                size=value.shape
            )
        package_adjoint[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )

    (
        xop_adj,
        xyop_adj,
        xh_adj,
        xg_adj,
        xyh_adj,
        xyg_adj,
    ) = _new_site_weighted_packages_bilinear_adjoint_x(
        grown,
        xgrown,
        ygrown,
        xygrown,
        package_adjoint,
        h1e,
        eri,
        yeri,
        site_count,
        future_sites,
    )

    lhs = _reduced_operator_pairing(package_adjoint, forward)
    rhs = _reduced_operator_pairing(xop_adj, xgrown)
    rhs += _reduced_operator_pairing(xyop_adj, xygrown)
    rhs += float(np.real(np.vdot(xh_adj, xh1)))
    rhs += float(np.real(np.vdot(xg_adj, xeri)))
    rhs += float(np.real(np.vdot(xyh_adj, xyh1)))
    rhs += float(np.real(np.vdot(xyg_adj, xyeri)))

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_grown_reduced_v1_packages_tangent_adjoint_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=797)
    dh1, deri = _physical_random_integrals(nsites, seed=798)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    path = recursive_tangent_path_for_active_integrals(solver, dh1, deri)
    narg = solver.chain.blocks[3].truncated.source
    source_block = narg._su2_source_renormalized_block
    source_tangent = path.sources[2]
    grown = dict(path.pre_rotation_parts[3]["grown"])
    future_sites = (3,)
    actual, dv1 = _grown_reduced_v1_packages_tangent(
        source_block,
        source_tangent,
        grown,
        h1e,
        dh1,
        eri,
        deri,
        future_sites,
    )

    rng = np.random.default_rng(799)
    package_adjoint = {}
    for key, tensor in actual.items():
        blocks = {}
        for block_key, block in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=block.shape) + 1j * rng.normal(
                size=block.shape
            )
        package_adjoint[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )

    op_adj, dh1_adj, deri_adj = _grown_reduced_v1_packages_tangent_adjoint(
        source_block,
        source_tangent,
        grown,
        package_adjoint,
        h1e,
        eri,
        future_sites,
    )

    lhs = _reduced_operator_pairing(package_adjoint, dv1)
    rhs = _reduced_operator_pairing(op_adj, source_tangent.reduced_operators)
    rhs += float(np.real(np.vdot(dh1_adj, dh1)))
    rhs += float(np.real(np.vdot(deri_adj, deri)))

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_grown_reduced_v1_packages_bilinear_adjoint_x_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=843)
    xh1, xeri = _physical_random_integrals(nsites, seed=844)
    yh1, yeri = _physical_random_integrals(nsites, seed=845)
    xyh1, xyeri = _physical_random_integrals(nsites, seed=846)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    x_path = recursive_tangent_path_for_active_integrals(solver, xh1, xeri)
    y_path = recursive_tangent_path_for_active_integrals(solver, yh1, yeri)
    xy_path = recursive_tangent_path_for_active_integrals(solver, xyh1, xyeri)
    narg = solver.chain.blocks[3].truncated.source
    source_block = narg._su2_source_renormalized_block
    source_x = x_path.sources[2]
    source_y = y_path.sources[2]
    source_xy = xy_path.sources[2]
    grown = dict(x_path.pre_rotation_parts[3]["grown"])
    future_sites = (3,)
    _actual, forward = _grown_reduced_v1_packages_bilinear(
        source_block,
        source_x,
        source_y,
        source_xy,
        grown,
        h1e,
        xh1,
        yh1,
        xyh1,
        eri,
        xeri,
        yeri,
        xyeri,
        future_sites,
    )

    rng = np.random.default_rng(847)
    package_adjoint = {}
    for key, tensor in forward.items():
        blocks = {}
        for block_key, value in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=value.shape) + 1j * rng.normal(
                size=value.shape
            )
        package_adjoint[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )

    (
        xop_adj,
        xyop_adj,
        xh_adj,
        xg_adj,
        xyh_adj,
        xyg_adj,
    ) = _grown_reduced_v1_packages_bilinear_adjoint_x(
        source_block,
        source_x,
        source_y,
        source_xy,
        grown,
        package_adjoint,
        h1e,
        eri,
        yh1,
        yeri,
        future_sites,
    )

    lhs = _reduced_operator_pairing(package_adjoint, forward)
    rhs = _reduced_operator_pairing(xop_adj, source_x.reduced_operators)
    rhs += _reduced_operator_pairing(xyop_adj, source_xy.reduced_operators)
    rhs += float(np.real(np.vdot(xh_adj, xh1)))
    rhs += float(np.real(np.vdot(xg_adj, xeri)))
    rhs += float(np.real(np.vdot(xyh_adj, xyh1)))
    rhs += float(np.real(np.vdot(xyg_adj, xyeri)))

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_component_v1_packages_tangent_adjoint_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=800)
    dh1, deri = _physical_random_integrals(nsites, seed=801)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=4,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    path = recursive_tangent_path_for_active_integrals(solver, dh1, deri)
    block = solver.chain.blocks[2]
    dops = path.sources[2].reduced_operators
    future_sites = (2, 3)
    dv1 = _component_v1_packages_tangent(
        block,
        dops,
        h1e,
        dh1,
        eri,
        deri,
        2,
        future_sites,
    )

    rng = np.random.default_rng(802)
    package_adjoint = {}
    for key, tensor in dv1.items():
        blocks = {}
        for block_key, block_value in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=block_value.shape) + 1j * rng.normal(
                size=block_value.shape
            )
        package_adjoint[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )

    op_adj, dh1_adj, deri_adj = _component_v1_packages_tangent_adjoint(
        block,
        dops,
        package_adjoint,
        h1e,
        eri,
        2,
        future_sites,
    )

    lhs = _reduced_operator_pairing(package_adjoint, dv1)
    rhs = _reduced_operator_pairing(op_adj, dops)
    rhs += float(np.real(np.vdot(dh1_adj, dh1)))
    rhs += float(np.real(np.vdot(deri_adj, deri)))

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_weighted_packages_tangent_adjoint_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=812)
    dh1, deri = _physical_random_integrals(nsites, seed=813)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=4,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    path = recursive_tangent_path_for_active_integrals(solver, dh1, deri)
    block = solver.chain.blocks[2]
    dops = path.sources[2].reduced_operators
    future_sites = (2, 3)
    dweighted = _weighted_packages_tangent(
        block.reduced_operators,
        dops,
        h1e,
        dh1,
        eri,
        deri,
        2,
        future_sites,
        build_v1=False,
    )

    rng = np.random.default_rng(814)
    package_adjoint = {}
    for key, tensor in dweighted.items():
        if key[0] == "NextV1Spinor":
            continue
        blocks = {}
        for block_key, block_value in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=block_value.shape) + 1j * rng.normal(
                size=block_value.shape
            )
        package_adjoint[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )

    op_adj, dh1_adj, deri_adj = _weighted_packages_tangent_adjoint(
        block.reduced_operators,
        dops,
        package_adjoint,
        h1e,
        eri,
        2,
        future_sites,
        build_v1=False,
    )

    lhs = _reduced_operator_pairing(package_adjoint, dweighted)
    rhs = _reduced_operator_pairing(op_adj, dops)
    rhs += float(np.real(np.vdot(dh1_adj, dh1)))
    rhs += float(np.real(np.vdot(deri_adj, deri)))

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.parametrize(
    "builder,adjoint_builder,args",
    [
        (_density_tangent, _density_tangent_adjoint, (0, 1)),
        (_spin_density_tangent, _spin_density_tangent_adjoint, (0, 1)),
        (_pair_annihilate_tangent, _pair_annihilate_tangent_adjoint, (0, 1)),
    ],
)
def test_composite_tangent_adjoint_matches_forward_pairing(
    builder,
    adjoint_builder,
    args,
):
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 3
    nelec = 3
    j2 = 1
    h1e, eri = _physical_random_integrals(nsites, seed=791)
    dh1, deri = _physical_random_integrals(nsites, seed=792)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=4,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    path = recursive_tangent_path_for_active_integrals(solver, dh1, deri)
    operators = solver.chain.blocks[2].reduced_operators
    doperators = path.sources[2].reduced_operators
    tangent = builder(operators, doperators, {}, {}, *args)

    rng = np.random.default_rng(793)
    adj_blocks = {}
    for key, value in tangent.blocks.items():
        adj_blocks[key] = rng.normal(size=value.shape) + 1j * rng.normal(
            size=value.shape
        )
    tangent_adjoint = ReducedSU2Tensor(
        IrrepTensor(tangent.site, tangent.site, tangent.op, adj_blocks)
    )
    got = adjoint_builder(operators, doperators, tangent_adjoint, *args)

    lhs = _irrep_tensor_pairing(tangent_adjoint.tensor, tangent.tensor)
    rhs = _reduced_operator_pairing(got, doperators)

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_rotate_reduced_tensors_tangent_adjoint_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=783)
    dh1, deri = _physical_random_integrals(nsites, seed=784)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    path = recursive_tangent_path_for_active_integrals(solver, dh1, deri)
    parts = path.pre_rotation_parts[3]
    tensors = {**parts["grown"], **parts["weighted"]}
    dtensors = {**parts["dgrown"], **parts["dweighted"]}
    rotated = path.sources[3].reduced_operators

    rng = np.random.default_rng(785)
    rotated_adjoint = {}
    for key, tensor in rotated.items():
        blocks = {}
        for block_key, block in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=block.shape) + 1j * rng.normal(
                size=block.shape
            )
        rotated_adjoint[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )

    tensor_adjoint, transform_adjoint = rotate_reduced_tensors_tangent_adjoint(
        solver.chain.blocks[3].truncated,
        tensors,
        rotated_adjoint,
    )

    lhs = _reduced_operator_pairing(rotated_adjoint, rotated)
    rhs = _reduced_operator_pairing(tensor_adjoint, dtensors)
    rhs += _transform_pairing(transform_adjoint, path.responses[3].d_transform_blocks)

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_rotate_reduced_tensors_bilinear_adjoint_x_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=833)
    xh1, xeri = _physical_random_integrals(nsites, seed=834)
    yh1, yeri = _physical_random_integrals(nsites, seed=835)
    xyh1, xyeri = _physical_random_integrals(nsites, seed=836)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    x_path = recursive_tangent_path_for_active_integrals(solver, xh1, xeri)
    y_path = recursive_tangent_path_for_active_integrals(solver, yh1, yeri)
    source_x, source_y, source_xy, _min_gap = _seed_two_site_bilinear_block(
        solver.chain.blocks[2],
        xh1[:2, :2],
        xeri[:2, :2, :2, :2],
        yh1[:2, :2],
        yeri[:2, :2, :2, :2],
        xyh1[:2, :2],
        xyeri[:2, :2, :2, :2],
        h1e_full=h1e,
        eri_full=eri,
        xdh1e_full=xh1,
        xderi_full=xeri,
        ydh1e_full=yh1,
        yderi_full=yeri,
        xydh1e_full=xyh1,
        xyderi_full=xyeri,
        final_size=nsites,
        project_v1_packages=True,
        include_retained_mixing=True,
        x_path=x_path,
        y_path=y_path,
    )

    block = solver.chain.blocks[3]
    source_block = solver.chain.blocks[2]
    grown = block.truncated.source
    x_grown_h = x_path.grown_hamiltonians[3]
    y_grown_h = y_path.grown_hamiltonians[3]
    xy_grown_h = _grown_hamiltonian_bilinear(
        source_block,
        source_xy,
        h1e[:3, :3],
        xyh1[:3, :3],
        eri[:3, :3, :3, :3],
        xyeri[:3, :3, :3, :3],
        site_index=2,
    )
    response = truncation_bilinear_tangent(
        grown,
        block.truncated,
        x_grown_h,
        y_grown_h,
        xy_grown_h,
        tangent_x=x_path.responses[3],
        tangent_y=y_path.responses[3],
    )
    tensors, xtensors, ytensors, xytensors = _pre_rotation_tensors_and_bilinears(
        grown,
        source_x,
        source_y,
        source_xy,
        h1e,
        xh1,
        yh1,
        xyh1,
        eri,
        xeri,
        yeri,
        xyeri,
        tuple(range(3, nsites)),
        project_v1_packages=True,
        carry_rdm_operators=True,
        x_pre_rotation_parts=x_path.pre_rotation_parts[3],
        y_pre_rotation_parts=y_path.pre_rotation_parts[3],
    )
    rotated = _rotate_reduced_tensors_bilinear(
        block.truncated,
        tensors,
        xtensors,
        ytensors,
        xytensors,
        response,
    )

    rng = np.random.default_rng(837)
    rotated_adjoint = {}
    for key, tensor in rotated.items():
        blocks = {}
        for block_key, value in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=value.shape) + 1j * rng.normal(
                size=value.shape
            )
        rotated_adjoint[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )

    (
        tensor_x_adjoint,
        tensor_xy_adjoint,
        transform_x_adjoint,
        transform_xy_adjoint,
    ) = rotate_reduced_tensors_bilinear_adjoint_x(
        block.truncated,
        tensors,
        ytensors,
        rotated_adjoint,
        response,
    )

    lhs = _reduced_operator_pairing(rotated_adjoint, rotated)
    rhs = _reduced_operator_pairing(tensor_x_adjoint, xtensors)
    rhs += _reduced_operator_pairing(tensor_xy_adjoint, xytensors)
    rhs += _transform_pairing(transform_x_adjoint, response.x.d_transform_blocks)
    rhs += _transform_pairing(transform_xy_adjoint, response.dxy_transform_blocks)

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_pre_rotation_tensors_and_tangents_adjoint_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=803)
    dh1, deri = _physical_random_integrals(nsites, seed=804)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    path = recursive_tangent_path_for_active_integrals(solver, dh1, deri)
    parts = path.pre_rotation_parts[3]
    dtensors = {**parts["dgrown"], **parts["dweighted"]}

    rng = np.random.default_rng(805)
    tensor_adjoint = {}
    for key, tensor in dtensors.items():
        blocks = {}
        for block_key, block in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=block.shape) + 1j * rng.normal(
                size=block.shape
            )
        tensor_adjoint[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )

    narg = solver.chain.blocks[3].truncated.source
    op_adj, dh1_adj, deri_adj = _pre_rotation_tensors_and_tangents_adjoint(
        narg,
        path.sources[2],
        tensor_adjoint,
        h1e,
        eri,
        (3,),
        project_v1_packages=True,
        carry_rdm_operators=True,
        parts=parts,
    )

    lhs = _reduced_operator_pairing(tensor_adjoint, dtensors)
    rhs = _reduced_operator_pairing(op_adj, path.sources[2].reduced_operators)
    rhs += float(np.real(np.vdot(dh1_adj, dh1)))
    rhs += float(np.real(np.vdot(deri_adj, deri)))

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_pre_rotation_tensors_and_bilinears_adjoint_x_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=848)
    xh1, xeri = _physical_random_integrals(nsites, seed=849)
    yh1, yeri = _physical_random_integrals(nsites, seed=850)
    xyh1, xyeri = _physical_random_integrals(nsites, seed=851)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    x_path = recursive_tangent_path_for_active_integrals(solver, xh1, xeri)
    y_path = recursive_tangent_path_for_active_integrals(solver, yh1, yeri)
    source_x, source_y, source_xy, _min_gap = _seed_two_site_bilinear_block(
        solver.chain.blocks[2],
        xh1[:2, :2],
        xeri[:2, :2, :2, :2],
        yh1[:2, :2],
        yeri[:2, :2, :2, :2],
        xyh1[:2, :2],
        xyeri[:2, :2, :2, :2],
        h1e_full=h1e,
        eri_full=eri,
        xdh1e_full=xh1,
        xderi_full=xeri,
        ydh1e_full=yh1,
        yderi_full=yeri,
        xydh1e_full=xyh1,
        xyderi_full=xyeri,
        final_size=nsites,
        project_v1_packages=True,
        include_retained_mixing=True,
        x_path=x_path,
        y_path=y_path,
    )
    grown = solver.chain.blocks[3].truncated.source
    tensors, xtensors, _ytensors, xytensors = _pre_rotation_tensors_and_bilinears(
        grown,
        source_x,
        source_y,
        source_xy,
        h1e,
        xh1,
        yh1,
        xyh1,
        eri,
        xeri,
        yeri,
        xyeri,
        (3,),
        project_v1_packages=True,
        carry_rdm_operators=True,
        x_pre_rotation_parts=x_path.pre_rotation_parts[3],
        y_pre_rotation_parts=y_path.pre_rotation_parts[3],
    )
    del tensors

    rng = np.random.default_rng(852)
    tensor_x_adjoint = {}
    for key, tensor in xtensors.items():
        blocks = {}
        for block_key, value in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=value.shape) + 1j * rng.normal(
                size=value.shape
            )
        tensor_x_adjoint[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )
    tensor_xy_adjoint = {}
    for key, tensor in xytensors.items():
        blocks = {}
        for block_key, value in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=value.shape) + 1j * rng.normal(
                size=value.shape
            )
        tensor_xy_adjoint[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )

    (
        source_x_adj,
        source_xy_adj,
        xh_adj,
        xg_adj,
        xyh_adj,
        xyg_adj,
    ) = _pre_rotation_tensors_and_bilinears_adjoint_x(
        grown,
        source_x,
        source_y,
        source_xy,
        tensor_x_adjoint,
        tensor_xy_adjoint,
        h1e,
        eri,
        yh1,
        yeri,
        (3,),
        project_v1_packages=True,
        carry_rdm_operators=True,
        x_pre_rotation_parts=x_path.pre_rotation_parts[3],
        y_pre_rotation_parts=y_path.pre_rotation_parts[3],
    )

    lhs = _reduced_operator_pairing(tensor_x_adjoint, xtensors)
    lhs += _reduced_operator_pairing(tensor_xy_adjoint, xytensors)
    rhs = _reduced_operator_pairing(source_x_adj, source_x.reduced_operators)
    rhs += _reduced_operator_pairing(source_xy_adj, source_xy.reduced_operators)
    rhs += float(np.real(np.vdot(xh_adj, xh1)))
    rhs += float(np.real(np.vdot(xg_adj, xeri)))
    rhs += float(np.real(np.vdot(xyh_adj, xyh1)))
    rhs += float(np.real(np.vdot(xyg_adj, xyeri)))

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_recursive_growth_step_tangent_adjoint_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=809)
    dh1, deri = _physical_random_integrals(nsites, seed=810)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    path = recursive_tangent_path_for_active_integrals(solver, dh1, deri)
    source_tangent = path.sources[2]
    next_tangent = path.sources[3]

    rng = np.random.default_rng(811)
    h_adj_blocks = {}
    for key, block in next_tangent.hamiltonian.blocks.items():
        raw = rng.normal(size=block.shape)
        h_adj_blocks[key] = 0.5 * (raw + raw.T)
    next_h_adj = IrrepTensor(
        next_tangent.hamiltonian.bra,
        next_tangent.hamiltonian.ket,
        next_tangent.hamiltonian.op,
        h_adj_blocks,
    )
    next_op_adj = {}
    for key, tensor in next_tangent.reduced_operators.items():
        blocks = {}
        for block_key, block in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=block.shape) + 1j * rng.normal(
                size=block.shape
            )
        next_op_adj[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )

    h_adj, op_adj, dh1_adj, deri_adj = _recursive_growth_step_tangent_adjoint(
        solver.chain.blocks[2],
        solver.chain.blocks[3].truncated.source,
        solver.chain.blocks[3],
        source_tangent,
        next_h_adj,
        next_op_adj,
        h1e,
        eri,
        (3,),
        path.responses[3],
        path.pre_rotation_parts[3],
        project_v1_packages=True,
        carry_rdm_operators=True,
    )

    lhs = _irrep_tensor_pairing(next_h_adj, next_tangent.hamiltonian)
    lhs += _reduced_operator_pairing(next_op_adj, next_tangent.reduced_operators)
    rhs = _irrep_tensor_pairing(h_adj, source_tangent.hamiltonian)
    rhs += _reduced_operator_pairing(op_adj, source_tangent.reduced_operators)
    rhs += float(np.real(np.vdot(dh1_adj, dh1)))
    rhs += float(np.real(np.vdot(deri_adj, deri)))

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_seed_two_site_tangent_adjoint_matches_forward_pairing():
    from pyqed.narg.irrep_tensor import IrrepTensor
    from pyqed.narg.qchem.su2_reduced_tensor import ReducedSU2Tensor

    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=815)
    dh1, deri = _physical_random_integrals(nsites, seed=816)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=4,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    path = recursive_tangent_path_for_active_integrals(solver, dh1, deri)
    source_tangent = path.sources[2]

    rng = np.random.default_rng(817)
    h_adj_blocks = {}
    for key, block in source_tangent.hamiltonian.blocks.items():
        raw = rng.normal(size=block.shape)
        h_adj_blocks[key] = 0.5 * (raw + raw.T)
    h_adj = IrrepTensor(
        source_tangent.hamiltonian.bra,
        source_tangent.hamiltonian.ket,
        source_tangent.hamiltonian.op,
        h_adj_blocks,
    )
    op_adj = {}
    for key, tensor in source_tangent.reduced_operators.items():
        blocks = {}
        for block_key, block in tensor.blocks.items():
            blocks[block_key] = rng.normal(size=block.shape) + 1j * rng.normal(
                size=block.shape
            )
        op_adj[key] = ReducedSU2Tensor(
            IrrepTensor(tensor.site, tensor.site, tensor.op, blocks)
        )

    dh1_adj, deri_adj = _seed_two_site_tangent_adjoint(
        solver.chain.blocks[2],
        source_tangent,
        h_adj,
        op_adj,
        h1e,
        eri,
        final_size=nsites,
        project_v1_packages=True,
        include_retained_mixing=True,
    )

    lhs = _irrep_tensor_pairing(h_adj, source_tangent.hamiltonian)
    lhs += _reduced_operator_pairing(op_adj, source_tangent.reduced_operators)
    rhs = float(np.real(np.vdot(dh1_adj, dh1)))
    rhs += float(np.real(np.vdot(deri_adj, deri)))

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_recursive_active_integral_adjoint_from_path_matches_terminal_pairing():
    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=818)
    dh1, deri = _physical_random_integrals(nsites, seed=819)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    path = recursive_tangent_path_for_active_integrals(solver, dh1, deri)
    terminal = _terminal_block_from_recursive_tangent_path(solver, path)

    rng = np.random.default_rng(820)
    raw = rng.normal(size=terminal.shape)
    terminal_adjoint = 0.5 * (raw + raw.T)
    dh1_adj, deri_adj = recursive_active_integral_adjoint_from_path(
        solver,
        path,
        terminal_adjoint,
    )

    lhs = float(np.real(np.vdot(terminal_adjoint, terminal)))
    rhs = float(np.real(np.vdot(dh1_adj, dh1)))
    rhs += float(np.real(np.vdot(deri_adj, deri)))

    np.testing.assert_allclose(rhs, lhs, rtol=1.0e-10, atol=1.0e-10)


def test_recursive_active_integral_adjoint_from_path_is_path_independent():
    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=821)
    dh1_a, deri_a = _physical_random_integrals(nsites, seed=822)
    dh1_b, deri_b = _physical_random_integrals(nsites, seed=823)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    path_a = recursive_tangent_path_for_active_integrals(solver, dh1_a, deri_a)
    path_b = recursive_tangent_path_for_active_integrals(solver, dh1_b, deri_b)
    terminal = _terminal_block_from_recursive_tangent_path(solver, path_a)
    rng = np.random.default_rng(824)
    raw = rng.normal(size=terminal.shape)
    terminal_adjoint = 0.5 * (raw + raw.T)

    dh1_adj_a, deri_adj_a = recursive_active_integral_adjoint_from_path(
        solver,
        path_a,
        terminal_adjoint,
    )
    dh1_adj_b, deri_adj_b = recursive_active_integral_adjoint_from_path(
        solver,
        path_b,
        terminal_adjoint,
    )

    np.testing.assert_allclose(dh1_adj_a, dh1_adj_b, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(deri_adj_a, deri_adj_b, rtol=1.0e-10, atol=1.0e-10)


def test_recursive_active_integral_adjoint_arrays_match_response_basis():
    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=825)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    psi = np.asarray(solver.root_vectors[:, 0], dtype=complex)
    dh1_adj, deri_adj, _ = recursive_active_integral_adjoint_arrays(
        solver,
        psi,
        psi,
    )
    h_values, g_values, basis = recursive_active_integral_adjoint_coefficients(
        solver,
        psi,
        psi,
    )
    got = []
    for h_comp, g_comp in zip(basis.h1_components, basis.eri_components):
        value = float(np.real(np.vdot(dh1_adj, h_comp)))
        value += float(np.real(np.vdot(deri_adj, g_comp)))
        got.append(value)
    ref = np.real(np.concatenate([h_values, g_values]))

    np.testing.assert_allclose(got, ref, rtol=1.0e-10, atol=1.0e-10)


def test_recursive_projected_v1_matches_four_site_retained_energy_derivative():
    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=81)
    dh1, deri = _physical_random_integrals(nsites, seed=83)
    mol = DummyMol(nelec, j2)

    def run(h, v):
        return NARG(
            SimpleNamespace(mol=mol),
            mol=mol,
            h1e=h,
            eri=v,
            D=4,
            nstates=1,
            final_size=nsites,
            target_nelec=nelec,
            target_j2=j2,
            su2_backend="python",
            project_v1_packages=True,
            carry_rdm_operators=True,
        ).run()

    solver = run(h1e, eri)
    perturbation = recursive_perturbation_for_active_integrals(solver, dh1, deri)
    vector = solver.root_vectors[:, 0]
    analytic = np.vdot(vector, perturbation.block @ vector) / np.vdot(vector, vector)

    eps = 1.0e-5
    finite_difference = (
        run(h1e + eps * dh1, eri + eps * deri).e_tot[0]
        - run(h1e - eps * dh1, eri - eps * deri).e_tot[0]
    ) / (2.0 * eps)

    np.testing.assert_allclose(analytic, finite_difference, atol=1.0e-10)


def test_recursive_active_response_basis_reconstructs_symmetric_perturbation():
    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=84)
    dh1, deri = _physical_random_integrals(nsites, seed=85)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=4,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()

    direct = recursive_perturbation_for_active_integrals(solver, dh1, deri)
    basis = recursive_active_integral_response_basis(solver)
    reconstructed = recursive_response_block_from_active_basis(
        solver,
        dh1,
        deri,
        basis=basis,
    )

    np.testing.assert_allclose(reconstructed, direct.block, atol=1.0e-11)


def test_recursive_active_response_basis_parallel_matches_serial():
    nsites = 3
    nelec = 3
    j2 = 1
    h1e, eri = _physical_random_integrals(nsites, seed=858)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
        recursive_response_workers=1,
    ).run()

    serial = recursive_active_integral_response_basis(solver)
    delattr(solver, "_su2_recursive_active_response_basis_cache")
    block_only = recursive_active_integral_response_basis(solver, include_paths=False)
    delattr(solver, "_su2_recursive_active_response_basis_cache")
    solver.recursive_response_workers = 2
    solver.timings["recursive_response_workers"] = 2
    parallel = recursive_active_integral_response_basis(solver)

    assert serial.worker_count == 1
    assert block_only.paths is None
    assert parallel.worker_count == 2
    np.testing.assert_allclose(block_only.blocks, serial.blocks, atol=1.0e-12)
    np.testing.assert_allclose(block_only.h1_components, serial.h1_components)
    np.testing.assert_allclose(block_only.eri_components, serial.eri_components)
    np.testing.assert_allclose(parallel.blocks, serial.blocks, atol=1.0e-12)
    np.testing.assert_allclose(parallel.h1_components, serial.h1_components)
    np.testing.assert_allclose(parallel.eri_components, serial.eri_components)


def test_recursive_active_response_basis_projects_all_pair_components():
    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=86)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=4,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    rng = np.random.default_rng(87)
    psi = solver.root_vectors[:, 0]
    bra = rng.normal(size=psi.size) + 1j * rng.normal(size=psi.size)
    pairs = [(p, q) for p in range(nsites) for q in range(p + 1, nsites)]

    got = recursive_response_pair_components_from_active_basis(
        solver,
        h1e,
        eri,
        pairs,
        bra,
        psi,
        ncore=0,
        ncas=nsites,
    )
    ref = []
    for pair in pairs:
        dh1, deri = cas_integral_response_from_pair(
            h1e,
            eri,
            pair,
            ncore=0,
            ncas=nsites,
        )
        perturbation = recursive_perturbation_for_active_integrals(solver, dh1, deri)
        ref.append(2.0 * np.real(np.vdot(bra, perturbation.block @ psi)))

    np.testing.assert_allclose(got, ref, atol=1.0e-11)


def test_active_symmetric_pair_response_matrix_matches_explicit_coefficients():
    nmo = 7
    ncore = 1
    ncas = 4
    nelec = 4
    j2 = 0
    h1_mo, eri_mo = _physical_random_integrals(nmo, seed=884)
    h1e, eri = _effective_cas_integrals_from_full(
        h1_mo,
        eri_mo,
        ncore=ncore,
        ncas=ncas,
    )
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=4,
        nstates=1,
        final_size=ncas,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    basis = recursive_active_integral_response_basis(solver)
    pairs = [(0, 2), (1, 5), (3, 6), (4, 6)]

    got = active_symmetric_pair_response_matrix(
        h1_mo,
        eri_mo,
        pairs,
        ncore=ncore,
        ncas=ncas,
        basis=basis,
    )
    ref = []
    for pair in pairs:
        dh1, deri = cas_integral_response_from_pair(
            h1_mo,
            eri_mo,
            pair,
            ncore=ncore,
            ncas=ncas,
        )
        h_coeff, eri_coeff = active_symmetric_integral_coefficients(
            dh1,
            deri,
            basis,
        )
        ref.append(np.concatenate([h_coeff, eri_coeff]))

    np.testing.assert_allclose(got, np.vstack(ref), atol=1.0e-13)


def test_recursive_active_response_basis_projects_full_mo_pair_components():
    nmo = 6
    ncore = 1
    ncas = 4
    nelec = 4
    j2 = 0
    h1_mo, eri_mo = _physical_random_integrals(nmo, seed=88)
    h1e, eri = _effective_cas_integrals_from_full(h1_mo, eri_mo, ncore=ncore, ncas=ncas)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=4,
        nstates=1,
        final_size=ncas,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    rng = np.random.default_rng(89)
    psi = solver.root_vectors[:, 0]
    bra = rng.normal(size=psi.size) + 1j * rng.normal(size=psi.size)
    pairs = [(p, q) for p in range(nmo) for q in range(p + 1, nmo)]

    got = recursive_response_pair_components_from_active_basis(
        solver,
        h1_mo,
        eri_mo,
        pairs,
        bra,
        psi,
        ncore=ncore,
        ncas=ncas,
    )
    ref = []
    for pair in pairs:
        dh1, deri = cas_integral_response_from_pair(
            h1_mo,
            eri_mo,
            pair,
            ncore=ncore,
            ncas=ncas,
        )
        perturbation = recursive_perturbation_for_active_integrals(solver, dh1, deri)
        ref.append(2.0 * np.real(np.vdot(bra, perturbation.block @ psi)))

    np.testing.assert_allclose(got, ref, atol=1.0e-11)


def test_active_symmetric_adjoint_arrays_from_complex_values_match_coefficients():
    nsites = 3
    h1_keys = tuple((p, q) for p in range(nsites) for q in range(p, nsites))
    eri_keys = ((0, 0, 0, 0), (0, 1, 1, 2), (1, 2, 1, 2))
    rng = np.random.default_rng(907)
    h_values = rng.normal(size=len(h1_keys)) + 1j * rng.normal(size=len(h1_keys))
    g_values = rng.normal(size=len(eri_keys)) + 1j * rng.normal(size=len(eri_keys))
    dh1, deri = _physical_random_integrals(nsites, seed=908)
    basis = SimpleNamespace(h1_keys=h1_keys, eri_keys=eri_keys)
    h_coeff, g_coeff = active_symmetric_integral_coefficients(dh1, deri, basis)

    h_adj, g_adj = active_symmetric_adjoint_arrays_from_values(
        nsites,
        h1_keys,
        eri_keys,
        h_values,
        g_values,
    )

    lhs = np.vdot(h_adj, dh1) + np.vdot(g_adj, deri)
    rhs = np.dot(h_coeff, h_values) + np.dot(g_coeff, g_values)
    np.testing.assert_allclose(lhs, rhs, rtol=1.0e-12, atol=1.0e-12)


def test_full_cas_integral_response_matches_nonredundant_finite_difference():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = mol.RHF().run()
    mo = np.asarray(mf.mo_coeff)
    h1_mo = mf.get_hcore_mo(mo)
    eri_mo = mf.get_eri_mo(mo, notation="chem")
    ncore = 1
    ncas = 2
    active = slice(ncore, ncore + ncas)
    eps = 1.0e-5

    for pair in [(0, 1), (1, 3), (0, 3)]:
        kappa = np.zeros((mo.shape[1], mo.shape[1]))
        p, q = pair
        kappa[p, q] = 1.0
        kappa[q, p] = -1.0
        dh1, deri = cas_integral_response_from_full(
            h1_mo,
            eri_mo,
            kappa,
            ncore=ncore,
            ncas=ncas,
        )

        mo_plus = rotate_orbitals(mo, eps * kappa)
        mo_minus = rotate_orbitals(mo, -eps * kappa)
        h1_plus, _ = h1e_for_cas(mf, ncas=ncas, ncore=ncore, mo_coeff=mo_plus)
        h1_minus, _ = h1e_for_cas(mf, ncas=ncas, ncore=ncore, mo_coeff=mo_minus)
        eri_plus = mf.get_eri_mo(mo_plus, notation="chem")[active, active, active, active]
        eri_minus = mf.get_eri_mo(mo_minus, notation="chem")[active, active, active, active]

        np.testing.assert_allclose(dh1, (h1_plus - h1_minus) / (2.0 * eps), atol=1.0e-9)
        np.testing.assert_allclose(deri, (eri_plus - eri_minus) / (2.0 * eps), atol=1.0e-9)


def test_pair_cas_integral_response_matches_full_sparse_generator():
    nmo = 7
    ncore = 2
    ncas = 3
    h1e, eri = _physical_random_integrals(nmo, seed=91)

    for pair, value in [
        ((0, 2), 1.0),
        ((0, 6), -0.5),
        ((3, 6), 0.25),
        ((2, 4), -0.75),
    ]:
        kappa = _antihermitian_pair(nmo, pair, value=value)
        ref_dh1, ref_deri = cas_integral_response_from_full(
            h1e,
            eri,
            kappa,
            ncore=ncore,
            ncas=ncas,
        )
        got_dh1, got_deri = cas_integral_response_from_pair(
            h1e,
            eri,
            pair,
            ncore=ncore,
            ncas=ncas,
            value=value,
        )

        np.testing.assert_allclose(got_dh1, ref_dh1, atol=1.0e-13)
        np.testing.assert_allclose(got_deri, ref_deri, atol=1.0e-13)


def test_batched_pair_cas_integral_response_matches_full_generator():
    nmo = 8
    ncore = 2
    ncas = 4
    h1e, eri = _physical_random_integrals(nmo, seed=97)
    pairs = [(0, 2), (1, 7), (3, 6), (2, 5), (4, 7)]
    coeffs = np.array([0.5, -0.25, 0.125, -0.375, 0.75])
    kappa = sum(
        (_antihermitian_pair(nmo, pair, value=coeff) for pair, coeff in zip(pairs, coeffs)),
        np.zeros((nmo, nmo)),
    )

    ref_dh1, ref_deri = cas_integral_response_from_full(
        h1e,
        eri,
        kappa,
        ncore=ncore,
        ncas=ncas,
    )
    got_dh1, got_deri = cas_integral_response_from_pairs(
        h1e,
        eri,
        pairs,
        coeffs,
        ncore=ncore,
        ncas=ncas,
    )

    np.testing.assert_allclose(got_dh1, ref_dh1, atol=1.0e-13)
    np.testing.assert_allclose(got_deri, ref_deri, atol=1.0e-13)


def test_active_kappa_terminal_response_matches_finite_difference_vector():
    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=37)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=80,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
    ).run()

    kappa = active_pair_kappa(nsites, 1, 3)
    perturbation = solver.active_perturbation_block(kappa)
    response = solver.terminal_response(perturbation)

    eps = 1.0e-5
    evals_p, evecs_p = eigh(solver.block + eps * perturbation, check_finite=False)
    evals_m, evecs_m = eigh(solver.block - eps * perturbation, check_finite=False)
    del evals_p, evals_m
    psi = solver.root_vectors[:, 0]
    plus = evecs_p[:, 0]
    minus = evecs_m[:, 0]
    if np.vdot(psi, plus).real < 0.0:
        plus = -plus
    if np.vdot(psi, minus).real < 0.0:
        minus = -minus

    finite_difference = (plus - minus) / (2.0 * eps)

    np.testing.assert_allclose(response.vector, finite_difference, atol=1.0e-8)
    assert response.residual_norm < 1.0e-10


def test_nargscf_accepts_terminal_response_ah_for_active_pairs():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = mol.RHF().run()

    mc = NARGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        nstates=1,
        target_j2=0,
        su2_backend="python",
        rotation_space="casscf",
        max_pairs_per_cycle=3,
        optimizer="AH",
        ah_hessian="terminal_response",
        max_cycle=1,
        max_step=0.02,
    ).run()

    assert mc.history[0]["ah_hessian"] == "terminal_response"
    assert mc.history[0]["ah_terminal_response_solves"] >= 1
    assert mc.history[0]["ah_terminal_response_residual_norm"] < 1.0e-8
    assert mc.history[0]["ah_response_pair_cache_enabled"]
    assert mc.history[0]["ah_response_pair_cache_blocks"] == 3


def test_nargscf_accepts_recursive_response_ah():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = mol.RHF().run()

    mc = NARGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        nstates=1,
        target_j2=0,
        su2_backend="python",
        rotation_space="casscf",
        max_pairs_per_cycle=2,
        optimizer="AH",
        ah_hessian="recursive_response",
        max_cycle=1,
        max_step=0.02,
    ).run()

    assert mc.history[0]["ah_hessian"] == "recursive_response"
    assert mc.history[0]["ah_terminal_response_solves"] >= 1
    assert mc.history[0]["ah_recursive_fd_evaluations"] == 0
    assert mc.history[0]["ah_recursive_gradient_evaluations"] == 0
    assert mc.history[0]["ah_recursive_bilinear_evaluations"] >= 1
    assert mc.history[0]["ah_gradient_kind"] == "recursive"
    assert mc.history[0]["ah_recursive_response_block_count"] >= 1
    assert not mc.history[0]["ah_response_pair_cache_enabled"]
    assert mc.history[0]["ah_response_pair_cache_blocks"] == 0
    assert mc.narg.timings["project_v1_packages"]


def test_nargscf_recursive_response_hessian_matches_relaxed_fd_for_two_site_cas():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = mol.RHF().run()

    mc = NARGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        nstates=1,
        target_j2=0,
        su2_backend="python",
        rotation_space="casscf",
        optimizer="AH",
        ah_hessian="recursive_response",
        max_cycle=0,
    )
    pairs = mc._ordered_pairs()
    energy, solver = mc._evaluate(mf.mo_coeff)
    energy, solver, _fock, _grad, _grad_vec = mc._evaluate_with_gradient(
        mf.mo_coeff,
        pairs=pairs,
        energy=energy,
        solver=solver,
    )
    context = dict(mc._last_gradient_context)
    rng = np.random.default_rng(619)
    vec = rng.normal(size=len(pairs))
    vec /= np.linalg.norm(vec)

    mc.ah_fd_step = 1.0e-4
    mc.ah_hessian = "recursive_response"
    recursive = mc._pair_hessian_action(dict(context), pairs, vec)
    mc.ah_hessian = "relaxed_fd"
    relaxed_fd = mc._pair_hessian_action(dict(context), pairs, vec)

    np.testing.assert_allclose(recursive, relaxed_fd, rtol=1.0e-7, atol=1.0e-7)


def test_nargscf_recursive_response_hessian_batches_many_pairs_with_active_basis():
    nmo = 12
    h1e, eri = _physical_random_integrals(nmo, seed=751)
    mf = ToyRestrictedMF(h1e, eri, nelec=(1, 1), spin=0)
    mc = NARGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        nstates=1,
        target_j2=0,
        su2_backend="python",
        rotation_space="casscf",
        optimizer="AH",
        ah_hessian="recursive_response",
        ah_fd_step=1.0e-4,
        max_cycle=0,
    )
    pairs = mc._ordered_pairs()
    assert len(pairs) > 2 * 9
    energy, solver = mc._evaluate(mf.mo_coeff)
    mc._evaluate_with_gradient(
        mf.mo_coeff,
        pairs=pairs,
        energy=energy,
        solver=solver,
    )
    context = dict(mc._last_gradient_context)
    rng = np.random.default_rng(753)
    vec = rng.normal(size=len(pairs))
    vec /= np.linalg.norm(vec)

    action = mc._pair_hessian_action(context, pairs, vec)
    mc.ah_hessian = "relaxed_fd"
    relaxed_fd = mc._pair_hessian_action(dict(mc._last_gradient_context), pairs, vec)

    np.testing.assert_allclose(action, relaxed_fd, rtol=1.0e-7, atol=1.0e-7)
    assert context.get("_recursive_bilinear_active_basis_blocks", 0) == 0
    assert context["_recursive_bilinear_evaluations"] == len(pairs)
    assert context.get("_recursive_response_active_basis_blocks", 0) == 0
    assert context.get("_recursive_response_pair_coefficients_bytes", 0) == 0
    assert context.get("_recursive_response_xy_pair_coefficients_bytes", 0) == 0
    assert context["_recursive_gradient_kind"] == "recursive_analytic"


def test_nargscf_recursive_response_hessian_uses_bilinear_array_adjoint(monkeypatch):
    import pyqed.narg.qchem.su2_response as su2_response

    nmo = 6
    ncas = 3
    h1e, eri = _physical_random_integrals(nmo, seed=858)
    mf = ToyRestrictedMF(h1e, eri, nelec=(2, 1), spin=1)
    mc = NARGSCF(
        mf,
        ncas=ncas,
        nelecas=(2, 1),
        D=4,
        nstates=1,
        target_j2=1,
        su2_backend="python",
        rotation_space="casscf",
        optimizer="AH",
        ah_hessian="recursive_response",
        ah_fd_step=1.0e-4,
        max_cycle=0,
    )
    pairs = mc._ordered_pairs()
    energy, solver = mc._evaluate(mf.mo_coeff)
    mc._evaluate_with_gradient(
        mf.mo_coeff,
        pairs=pairs,
        energy=energy,
        solver=solver,
    )
    context = dict(mc._last_gradient_context)
    rng = np.random.default_rng(859)
    vec = rng.normal(size=len(pairs))
    vec /= np.linalg.norm(vec)

    reference = mc._pair_hessian_action(dict(context), pairs, vec)
    monkeypatch.setattr(
        su2_response,
        "symmetric_active_integral_basis_size",
        lambda _ncas: 1,
    )
    action = mc._pair_hessian_action(context, pairs, vec)

    np.testing.assert_allclose(action, reference, rtol=1.0e-10, atol=1.0e-10)
    assert context["_recursive_bilinear_evaluations"] == 1
    assert context["_recursive_bilinear_active_basis_blocks"] == 0
    assert context["_recursive_response_active_basis_blocks"] == 0
    assert context["_recursive_gradient_kind"] == "recursive_analytic_adjoint"


def test_recursive_bilinear_active_adjoint_parallel_matches_serial():
    nmo = 8
    h1e, eri = _physical_random_integrals(nmo, seed=761)
    mf = ToyRestrictedMF(h1e, eri, nelec=(1, 1), spin=0)
    mc = NARGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        nstates=1,
        target_j2=0,
        su2_backend="python",
        rotation_space="casscf",
        optimizer="AH",
        ah_hessian="recursive_response",
        max_cycle=0,
    )
    pairs = mc._ordered_pairs()
    energy, solver = mc._evaluate(mf.mo_coeff)
    mc._evaluate_with_gradient(
        mf.mo_coeff,
        pairs=pairs,
        energy=energy,
        solver=solver,
    )
    context = dict(mc._last_gradient_context)
    rng = np.random.default_rng(763)
    vec = rng.normal(size=len(pairs))
    vec /= np.linalg.norm(vec)
    ydh1, yderi = cas_integral_response_from_pairs(
        context["h1_mo"],
        context["eri_mo"],
        pairs,
        vec,
        ncore=mc.ncore,
        ncas=mc.ncas,
    )
    basis = recursive_active_integral_response_basis(solver, state_id=mc.state_id)
    psi = solver.root_vectors[:, mc.state_id]

    serial = recursive_bilinear_active_integral_adjoint_coefficients(
        solver,
        ydh1,
        yderi,
        psi,
        psi,
        state_id=mc.state_id,
        basis=basis,
        workers=1,
    )
    parallel = recursive_bilinear_active_integral_adjoint_coefficients(
        solver,
        ydh1,
        yderi,
        psi,
        psi,
        state_id=mc.state_id,
        basis=basis,
        workers=2,
    )

    np.testing.assert_allclose(parallel.x_values, serial.x_values, atol=1.0e-12)
    np.testing.assert_allclose(parallel.xy_values, serial.xy_values, atol=1.0e-12)
    assert parallel.evaluation_count == serial.evaluation_count == 9
    assert parallel.worker_count == 2


def test_recursive_bilinear_active_adjoint_arrays_match_component_loop():
    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=856)
    ydh1, yderi = _physical_random_integrals(nsites, seed=857)
    mol = DummyMol(nelec, j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=3,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
        project_v1_packages=True,
        carry_rdm_operators=True,
    ).run()
    psi = np.asarray(solver.root_vectors[:, 0], dtype=complex)
    basis = recursive_active_integral_response_basis(solver, state_id=0)
    y_path = recursive_tangent_path_for_active_integrals(
        solver,
        ydh1,
        yderi,
        state_id=0,
    )

    loop = recursive_bilinear_active_integral_adjoint_coefficients(
        solver,
        ydh1,
        yderi,
        psi,
        psi,
        state_id=0,
        basis=basis,
        workers=1,
        y_path=y_path,
    )
    xh_adj, xg_adj, xyh_adj, xyg_adj, info = (
        recursive_bilinear_active_integral_adjoint_arrays_x(
            solver,
            ydh1,
            yderi,
            psi,
            psi,
            state_id=0,
            y_path=y_path,
        )
    )

    got_x = []
    got_xy = []
    for h_comp, g_comp in zip(basis.h1_components, basis.eri_components):
        got_x.append(
            np.vdot(xh_adj, h_comp) + np.vdot(xg_adj, g_comp)
        )
        got_xy.append(
            np.vdot(xyh_adj, h_comp) + np.vdot(xyg_adj, g_comp)
        )

    np.testing.assert_allclose(got_x, loop.x_values, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(got_xy, loop.xy_values, rtol=1.0e-10, atol=1.0e-10)
    assert info["evaluation_count"] == 1


def test_nargscf_recursive_gradient_matches_energy_fd_three_site():
    nsites = 3
    h1e, eri = _physical_random_integrals(nsites, seed=621)
    mf = ToyRestrictedMF(h1e, eri, nelec=(2, 1), spin=1)
    mc = NARGSCF(
        mf,
        ncas=nsites,
        nelecas=(2, 1),
        D=2,
        nstates=1,
        target_j2=1,
        su2_backend="python",
        rotation_space="active_active",
        optimizer="AH",
        ah_hessian="recursive_response",
        max_cycle=0,
    )
    pairs = mc._ordered_pairs()
    energy, solver = mc._evaluate(mf.mo_coeff)
    _energy, _solver, _fock, _grad, grad_vec = mc._evaluate_with_gradient(
        mf.mo_coeff,
        pairs=pairs,
        energy=energy,
        solver=solver,
    )

    eps = 1.0e-5
    fd = []
    for pair in pairs:
        kappa = _antihermitian_pair(nsites, pair)
        e_plus, _ = mc._evaluate(rotate_orbitals(mf.mo_coeff, eps * kappa))
        e_minus, _ = mc._evaluate(rotate_orbitals(mf.mo_coeff, -eps * kappa))
        fd.append((e_plus - e_minus) / (2.0 * eps))

    np.testing.assert_allclose(grad_vec, fd, rtol=1.0e-6, atol=1.0e-7)
    assert mc._last_gradient_context["gradient_kind"] == "recursive"
    assert mc._last_gradient_context["_recursive_gradient_evaluations"] == 1


def test_nargscf_recursive_gradient_can_skip_rdm_preconditioner():
    nsites = 3
    h1e, eri = _physical_random_integrals(nsites, seed=627)
    mf = ToyRestrictedMF(h1e, eri, nelec=(2, 1), spin=1)
    mc = NARGSCF(
        mf,
        ncas=nsites,
        nelecas=(2, 1),
        D=2,
        nstates=1,
        target_j2=1,
        su2_backend="python",
        rotation_space="active_active",
        optimizer="AH",
        ah_hessian="recursive_response",
        recursive_preconditioner="hcore",
        carry_rdm_operators=False,
        max_cycle=0,
    )
    pairs = mc._ordered_pairs()
    energy, solver = mc._evaluate(mf.mo_coeff)
    _energy, _solver, _fock, _grad, grad_vec = mc._evaluate_with_gradient(
        mf.mo_coeff,
        pairs=pairs,
        energy=energy,
        solver=solver,
    )

    eps = 1.0e-5
    fd = []
    for pair in pairs:
        kappa = _antihermitian_pair(nsites, pair)
        e_plus, _ = mc._evaluate(rotate_orbitals(mf.mo_coeff, eps * kappa))
        e_minus, _ = mc._evaluate(rotate_orbitals(mf.mo_coeff, -eps * kappa))
        fd.append((e_plus - e_minus) / (2.0 * eps))

    context = mc._last_gradient_context
    np.testing.assert_allclose(grad_vec, fd, rtol=1.0e-6, atol=1.0e-7)
    assert context["gradient_kind"] == "recursive"
    assert context["preconditioner_kind"] == "hcore"
    assert context["dm1"] is None
    assert context["dm2"] is None
    assert not solver.timings["carry_rdm_operators"]


def test_nargscf_recursive_response_hessian_differentiates_recursive_gradient():
    nsites = 3
    h1e, eri = _physical_random_integrals(nsites, seed=631)
    mf = ToyRestrictedMF(h1e, eri, nelec=(2, 1), spin=1)
    mc = NARGSCF(
        mf,
        ncas=nsites,
        nelecas=(2, 1),
        D=2,
        nstates=1,
        target_j2=1,
        su2_backend="python",
        rotation_space="active_active",
        optimizer="AH",
        ah_hessian="recursive_response",
        ah_fd_step=1.0e-4,
        max_cycle=0,
    )
    pairs = mc._ordered_pairs()
    energy, solver = mc._evaluate(mf.mo_coeff)
    mc._evaluate_with_gradient(
        mf.mo_coeff,
        pairs=pairs,
        energy=energy,
        solver=solver,
    )
    context = dict(mc._last_gradient_context)
    context["_recursive_fd_evaluations"] = 0
    context["_recursive_gradient_evaluations"] = 0
    rng = np.random.default_rng(633)
    vec = rng.normal(size=len(pairs))
    vec /= np.linalg.norm(vec)

    action = mc._pair_hessian_action(context, pairs, vec)

    eps = mc.ah_fd_step
    kappa = sum(
        (
            _antihermitian_pair(nsites, pair, value=eps * coeff)
            for pair, coeff in zip(pairs, vec)
        ),
        np.zeros((nsites, nsites)),
    )
    mo_plus = rotate_orbitals(mf.mo_coeff, kappa)
    e_plus, solver_plus = mc._evaluate(mo_plus)
    h1_plus, eri_plus = mc._get_integrals(mo_plus)
    grad_plus = mc._recursive_energy_gradient_vec(
        solver_plus,
        h1_plus,
        eri_plus,
        pairs,
    )
    mo_minus = rotate_orbitals(mf.mo_coeff, -kappa)
    e_minus, solver_minus = mc._evaluate(mo_minus)
    h1_minus, eri_minus = mc._get_integrals(mo_minus)
    grad_minus = mc._recursive_energy_gradient_vec(
        solver_minus,
        h1_minus,
        eri_minus,
        pairs,
    )
    del e_plus, e_minus
    fd = (grad_plus - grad_minus) / (2.0 * eps)

    np.testing.assert_allclose(action, fd, rtol=1.0e-7, atol=1.0e-7)
    assert context["_recursive_fd_evaluations"] == 0
    assert context["_recursive_gradient_evaluations"] == 0
    assert context["_recursive_bilinear_evaluations"] == len(pairs)
    assert context["_terminal_response_solves"] == 1


def test_nargscf_recursive_response_hessian_matches_fd_for_four_site_cas():
    nsites = 4
    h1e, eri = _physical_random_integrals(nsites, seed=641)
    mf = ToyRestrictedMF(h1e, eri, nelec=(2, 2), spin=0)
    mc = NARGSCF(
        mf,
        ncas=nsites,
        nelecas=(2, 2),
        D=2,
        nstates=1,
        target_j2=0,
        su2_backend="python",
        rotation_space=[(0, 1), (2, 3)],
        optimizer="AH",
        ah_hessian="recursive_response",
        ah_fd_step=1.0e-4,
        max_cycle=0,
    )
    pairs = mc._ordered_pairs()
    energy, solver = mc._evaluate(mf.mo_coeff)
    mc._evaluate_with_gradient(
        mf.mo_coeff,
        pairs=pairs,
        energy=energy,
        solver=solver,
    )
    context = dict(mc._last_gradient_context)
    context["_recursive_fd_evaluations"] = 0
    context["_recursive_gradient_evaluations"] = 0
    context["_recursive_bilinear_evaluations"] = 0

    vec = np.array([0.6, -0.8])
    action = mc._pair_hessian_action(context, pairs, vec)

    eps = mc.ah_fd_step
    kappa = sum(
        (
            _antihermitian_pair(nsites, pair, value=eps * coeff)
            for pair, coeff in zip(pairs, vec)
        ),
        np.zeros((nsites, nsites)),
    )
    mo_plus = rotate_orbitals(mf.mo_coeff, kappa)
    e_plus, solver_plus = mc._evaluate(mo_plus)
    h1_plus, eri_plus = mc._get_integrals(mo_plus)
    grad_plus = mc._recursive_energy_gradient_vec(
        solver_plus,
        h1_plus,
        eri_plus,
        pairs,
    )
    mo_minus = rotate_orbitals(mf.mo_coeff, -kappa)
    e_minus, solver_minus = mc._evaluate(mo_minus)
    h1_minus, eri_minus = mc._get_integrals(mo_minus)
    grad_minus = mc._recursive_energy_gradient_vec(
        solver_minus,
        h1_minus,
        eri_minus,
        pairs,
    )
    del e_plus, e_minus
    fd = (grad_plus - grad_minus) / (2.0 * eps)

    np.testing.assert_allclose(action, fd, rtol=1.0e-7, atol=1.0e-7)
    assert context["_recursive_fd_evaluations"] == 0
    assert context["_recursive_gradient_evaluations"] == 0
    assert context["_recursive_bilinear_evaluations"] == len(pairs)
