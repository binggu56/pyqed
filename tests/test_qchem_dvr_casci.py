import numpy as np
import pytest

from pyqed.qchem.dvr import CASCI, Molecule, RHF1D


class _ToyDVRMolecule:
    nelec = 2
    nelectron = 2

    def energy_nuc(self):
        return 0.0


class _ToyDVRMeanField:
    def __init__(self):
        self.mol = _ToyDVRMolecule()
        self.nelec = 2
        self.nmo = 2
        self.mo_coeff = np.eye(2)
        self.mo_occ = np.array([[2, 0], [2, 0]])
        self.eri = np.array([[0.7, 0.1], [0.1, 0.5]])

    def get_hcore(self):
        return np.diag([-1.0, 0.4])


class _ThreeOrbitalDVRMeanField:
    def __init__(self):
        self.mol = _ToyDVRMolecule()
        self.nelec = 2
        self.nmo = 3
        self.mo_coeff = np.eye(3)
        self.mo_occ = np.array([[2, 0, 0], [2, 0, 0]])
        self.eri = np.diag([0.7, 0.5, 0.4])

    def get_hcore(self):
        return np.diag([-1.0, -0.2, 0.1])


def test_dvr_package_imports_public_api():
    assert CASCI is not None
    assert RHF1D is not None
    assert Molecule is not None


def test_dvr_casci_runs_minimal_two_orbital_problem():
    mc = CASCI(_ToyDVRMeanField(), ncas=2, nelecas=2).run(nstates=1)

    assert len(mc.ci) == 1
    assert mc.e_tot[0] == pytest.approx(-1.3)
    assert mc.spin_square(0) == pytest.approx(0.0)


def test_dvr_casci_uses_parent_fix_spin():
    mc = CASCI(_ToyDVRMeanField(), ncas=2, nelecas=2)

    assert mc.fix_spin(ss=0, shift=0.3) is mc
    assert mc.spin_purification is True
    assert mc.ss == 0
    assert mc.shift == pytest.approx(0.3)


def test_dvr_casci_fix_spin_applies_singlet_penalty():
    plain = CASCI(_ThreeOrbitalDVRMeanField(), ncas=3, nelecas=2).run(nstates=4)
    fixed = (
        CASCI(_ThreeOrbitalDVRMeanField(), ncas=3, nelecas=2)
        .fix_spin(ss=0, shift=1.0)
        .run(nstates=4)
    )

    assert max(plain.spin_square(i) for i in range(4)) > 0.5
    assert all(fixed.spin_square(i) == pytest.approx(0.0) for i in range(4))
