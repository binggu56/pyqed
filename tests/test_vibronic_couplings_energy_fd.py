import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.casci import CASCI
from pyqed.units import amu_to_au


def _h2_casci(r_bohr, nstates=2):
    mol = Molecule(
        atom=f"H 0 0 {-0.5 * r_bohr}; H 0 0 {0.5 * r_bohr}",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
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
    mol.build(driver="gbasis")
    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)
    return np.asarray(mc.e_tot, dtype=float)


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
