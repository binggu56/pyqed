import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.casci import CASCI, CI_H
from pyqed.units import amu_to_au


def _h2_casci(r_bohr, nstates=2):
    mol = Molecule(
        atom=f"H 0 0 {-0.5 * r_bohr}; H 0 0 {0.5 * r_bohr}",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=nstates)
    return mol, mc


def _mass_weighted_h2_stretch(mol):
    masses = np.asarray(mol.atom_mass_list(), dtype=float) * amu_to_au
    mode = np.zeros((1, mol.natom, 3))
    mode[0, 0, 2] = -1.0 / np.sqrt(2.0 * masses[0])
    mode[0, 1, 2] = 1.0 / np.sqrt(2.0 * masses[1])
    return mode


def _displaced_energy(coords, symbols, mode, q):
    displaced = coords + q * mode[0]
    atom = [[symbol, *xyz] for symbol, xyz in zip(symbols, displaced)]
    mol = Molecule(atom=atom, unit="bohr", basis="sto-3g")
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)
    return np.asarray(mc.e_tot, dtype=float)


def _lih_casci(r_bohr, nstates=2):
    mol = Molecule(
        atom=f"Li 0 0 0; H 0 0 {r_bohr}",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(tol=1.0e-11)
    return CASCI(mf, ncas=2, nelecas=2).run(nstates=nstates)


def _tracked_lih_casci(r_bohr, reference, parallel):
    gto = pytest.importorskip("pyscf.gto")
    mol = Molecule(
        atom=f"Li 0 0 0; H 0 0 {r_bohr}",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(tol=1.0e-11)
    cross = gto.intor_cross("int1e_ovlp", reference.mol.topyscf(), mol.topyscf())
    mo_overlap = reference.mo_coeff.T @ cross @ mf.mo_coeff
    reference_indices, current_indices = linear_sum_assignment(-abs(mo_overlap))
    order = current_indices[np.argsort(reference_indices)]
    mo_coeff = np.asarray(mf.mo_coeff[:, order], dtype=complex)
    diagonal = mo_overlap[np.arange(len(order)), order]
    mo_coeff *= np.exp(-1j * np.angle(diagonal))[None, :]
    if parallel:
        active = slice(reference.ncore, reference.ncore + reference.ncas)
        active_overlap = reference.mo_coeff[:, active].T @ cross @ mo_coeff[:, active]
        left, _singular, right = np.linalg.svd(active_overlap)
        mo_coeff[:, active] = mo_coeff[:, active] @ right.T @ left.T
    if np.allclose(mo_coeff.imag, 0.0, atol=1.0e-14):
        mo_coeff = mo_coeff.real
    return CASCI(mf, ncas=2, nelecas=2).run(
        nstates=4,
        mo_coeff=mo_coeff,
    )


def _casci_hamiltonian_in_reference_frame(point, reference):
    determinant = CI_H(
        point.binary,
        point.hcore,
        point.eri_so,
        point.SC1,
        point.SC2,
    )
    determinant += point.e_core * np.eye(len(determinant))
    ci = np.asarray(reference.ci)
    return ci.conj() @ determinant @ ci.T


@pytest.mark.parametrize("moving_basis", ["rhf-relaxed", "rhf-relaxed-pt"])
def test_casci_rhf_relaxed_gradients_match_reoptimized_energy_finite_difference(
    moving_basis,
):
    r0 = 3.0
    mc = _lih_casci(r0)
    mode = np.zeros((1, mc.mol.natom, 3))
    mode[0, 1, 2] = 1.0

    relaxed = mc.vibronic_gradients(
        state_ids=[0, 1],
        modes=mode,
        moving_basis=moving_basis,
        backend="native",
    )
    step = 2.0e-4
    finite_difference = (
        np.asarray(_lih_casci(r0 + step).e_tot)
        - np.asarray(_lih_casci(r0 - step).e_tot)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        np.diagonal(relaxed[:, :, 0]).real,
        finite_difference,
        atol=2.0e-7,
        rtol=2.0e-7,
    )


@pytest.mark.parametrize("moving_basis", ["rhf-relaxed", "rhf-relaxed-pt"])
def test_casci_rhf_relaxed_hessian_recovers_reoptimized_energy_curvature(
    moving_basis,
):
    r0 = 3.0
    nstates = 4
    mc = _lih_casci(r0, nstates=nstates)
    mode = np.zeros((1, mc.mol.natom, 3))
    mode[0, 1, 2] = 1.0

    first, second = mc.vibronic_couplings(
        state_ids=range(nstates),
        modes=mode,
        moving_basis=moving_basis,
        backend="native",
    )
    curvature = np.diagonal(second[:, :, 0, 0]).real.copy()
    energies = np.asarray(mc.e_tot)
    for state in range(nstates):
        for other in range(nstates):
            if state != other:
                curvature[state] += (
                    2.0 * abs(first[other, state, 0]) ** 2
                    / (energies[state] - energies[other])
                )

    step = 2.0e-3
    finite_difference = (
        np.asarray(_lih_casci(r0 + step, nstates=nstates).e_tot)
        - 2.0 * energies
        + np.asarray(_lih_casci(r0 - step, nstates=nstates).e_tot)
    ) / step**2
    np.testing.assert_allclose(
        curvature,
        finite_difference,
        atol=2.0e-7,
        rtol=2.0e-6,
    )


@pytest.mark.parametrize(
    ("moving_basis", "parallel"),
    [("rhf-relaxed", False), ("rhf-relaxed-pt", True)],
)
def test_casci_relaxed_fg_matches_tracked_hamiltonian_finite_difference(
    moving_basis,
    parallel,
):
    r0 = 3.0
    reference = _lih_casci(r0, nstates=4)
    mode = np.zeros((1, reference.mol.natom, 3))
    mode[0, 1, 2] = 1.0
    first, second = reference.vibronic_couplings(
        state_ids=range(4),
        modes=mode,
        moving_basis=moving_basis,
        backend="native",
    )
    step = 2.0e-3
    plus = _tracked_lih_casci(r0 + step, reference, parallel)
    minus = _tracked_lih_casci(r0 - step, reference, parallel)
    h_plus = _casci_hamiltonian_in_reference_frame(plus, reference)
    h_zero = np.diag(reference.e_tot)
    h_minus = _casci_hamiltonian_in_reference_frame(minus, reference)

    np.testing.assert_allclose(
        first[:, :, 0],
        (h_plus - h_minus) / (2.0 * step),
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        second[:, :, 0, 0],
        (h_plus - 2.0 * h_zero + h_minus) / step**2,
        atol=2.0e-7,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "mc.vibronic_couplings() currently returns clamped BO-Hamiltonian "
        "derivatives, not relaxed finite-AO CASCI energy derivatives."
    ),
)
def test_casci_vibronic_couplings_match_relaxed_energy_finite_difference():
    r_equilibrium = 1.3886941021119301
    mol, mc = _h2_casci(r_equilibrium, nstates=2)
    mode = _mass_weighted_h2_stretch(mol)

    f, g = mc.vibronic_couplings(state_ids=[0, 1], modes=mode)

    coords = np.asarray(mol.atom_coords(), dtype=float)
    symbols = [mol.atom_symbol(atom_id) for atom_id in range(mol.natom)]
    step = 2.0e-3
    e_plus = _displaced_energy(coords, symbols, mode, step)
    e_0 = _displaced_energy(coords, symbols, mode, 0.0)
    e_minus = _displaced_energy(coords, symbols, mode, -step)

    fd_first = (e_plus - e_minus) / (2.0 * step)
    fd_second = (e_plus - 2.0 * e_0 + e_minus) / step**2

    np.testing.assert_allclose(np.diagonal(f[:, :, 0]), fd_first, atol=1.0e-7, rtol=1.0e-5)
    np.testing.assert_allclose(np.diagonal(g[:, :, 0, 0]), fd_second, atol=1.0e-6, rtol=1.0e-4)
