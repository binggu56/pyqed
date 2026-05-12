import sys
from pathlib import Path

import numpy as np


def _prefer_source_package():
    root = Path(__file__).resolve().parents[1]
    outer_init = (root / "__init__.py").resolve()
    loaded = sys.modules.get("pyqed")
    loaded_file_raw = getattr(loaded, "__file__", "") or ""
    loaded_file = Path(loaded_file_raw).resolve() if loaded_file_raw else None
    if loaded_file == outer_init:
        del sys.modules["pyqed"]
    sys.path.insert(0, str(root))


def test_rovibronic_modules_import():
    _prefer_source_package()
    from pyqed.namd.triatom import Triatom as LegacyTriatom
    from pyqed.namd.triatomic import Triatom

    assert Triatom is not None
    assert LegacyTriatom is not None


def test_curvilinear_ldr_identity_overlap_propagates():
    _prefer_source_package()
    from pyqed.ldr.curvilinear_2d import LDR2_Curvilinear

    mol = LDR2_Curvilinear([1.008, 1.008, 1.008], theta=1.8, nstates=2)
    mol.set_dvr([[1.0, 2.0], [1.0, 2.0]], [3, 3])
    mol.apes = np.zeros((*mol.nx, mol.nstates))

    psi0 = np.zeros((*mol.nx, mol.nstates), dtype=complex)
    psi0[1, 1, 0] = 1.0 / np.sqrt(mol.dv)

    result = mol.run(psi0, dt=0.01, nt=1, nout=1)

    assert len(result["psilist"]) == 2
    np.testing.assert_allclose(mol.norm(result["psilist"][-1]), 1.0, atol=1e-12)


def test_triatomic_fixed_j_rovibronic_propagates():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=1, charge=1, spin=0, unit="bohr", J=1)
    mol.set_dvr(domains=[[1.0, 1.4], [1.0, 1.4], [1.1, 1.5]], npts=[2, 2, 2])
    mol.apes = np.zeros((*mol.nx, mol.nstates))

    psi0 = np.zeros((*mol.nx, mol.nrot, mol.nstates), dtype=complex)
    psi0[0, 0, 0, 0, 0] = 1.0 / np.sqrt(mol.dv)

    result = mol.run(psi0, dt=1e-4, nt=1, nout=1)

    assert mol.nrot == 9
    assert result["psilist"][-1].shape == (*mol.nx, mol.nrot, mol.nstates)
    np.testing.assert_allclose(mol.H, mol.H.conj().T, atol=1e-10)
    norm = np.sqrt(np.sum(np.abs(result["psilist"][-1]) ** 2) * mol.dv)
    np.testing.assert_allclose(norm, 1.0, atol=1e-10)


def test_triatomic_fixed_jz_reduces_rotational_dimension():
    _prefer_source_package()
    from pyqed.namd.triatomic import Triatom

    atom = [
        ["H", (1.0, 0.0, 0.0)],
        ["H", (0.0, 0.0, 0.0)],
        ["H", (0.0, 1.0, 0.0)],
    ]
    mol = Triatom(atom, nstates=1, charge=1, spin=0, unit="bohr", J=1, Jz=0)
    mol.set_dvr(domains=[[1.0, 1.4], [1.0, 1.4], [1.1, 1.5]], npts=[2, 2, 2])
    mol.apes = np.zeros((*mol.nx, mol.nstates))

    psi0 = np.zeros((*mol.nx, mol.nrot, mol.nstates), dtype=complex)
    psi0[0, 0, 0, 1, 0] = 1.0 / np.sqrt(mol.dv)

    result = mol.run(psi0, dt=1e-4, nt=1, nout=1)

    assert mol.nrot == 3
    assert result["psilist"][-1].shape == (*mol.nx, mol.nrot, mol.nstates)
    assert mol.H.shape == (np.prod(mol.nx) * mol.nrot, np.prod(mol.nx) * mol.nrot)
    np.testing.assert_allclose(mol.H, mol.H.conj().T, atol=1e-10)
    norm = np.sqrt(np.sum(np.abs(result["psilist"][-1]) ** 2) * mol.dv)
    np.testing.assert_allclose(norm, 1.0, atol=1e-10)
