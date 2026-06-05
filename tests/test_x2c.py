import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.relativistic import x2c1e_hcore


def test_x2c1e_hcore_matches_pyscf_no_uncontract():
    pyscf = pytest.importorskip("pyscf")
    from pyscf.x2c import sfx2c1e

    mol = Molecule(atom="H 0 0 0; F 0 0 1.7", unit="bohr", basis="sto-3g")
    mol.build(driver="builtin", eri="s8")

    hcore = x2c1e_hcore(mol)
    helper = sfx2c1e.SpinFreeX2CHelper(mol.topyscf())
    helper.xuncontract = False
    hcore_ref = helper.get_hcore()

    np.testing.assert_allclose(hcore, hcore_ref, atol=1e-10)


def test_rhf_run_x2c_uses_scalar_x2c_hcore_and_restores_molecule():
    mol = Molecule(atom="H 0 0 0; F 0 0 1.7", unit="bohr", basis="sto-3g")
    mol.build(driver="builtin", eri="s8")
    nonrel_hcore = mol.hcore.copy()
    x2c_hcore = x2c1e_hcore(mol)

    mf = RHF(mol).run(x2c=True, tol=1e-10, conv_tol_dm=1e-8)

    assert mf.x2c is True
    assert mf.relativistic == "x2c"
    np.testing.assert_allclose(mf.get_hcore(), x2c_hcore, atol=1e-12)
    np.testing.assert_allclose(mol.hcore, nonrel_hcore, atol=1e-12)

def test_builtin_x2c_aligns_segmented_spherical_ordering():
    pyscf = pytest.importorskip("pyscf")
    from pyscf.x2c import sfx2c1e

    mol = Molecule(atom="H 0 0 0; Cl 0 0 2.4", unit="bohr", basis="sto-3g")
    mol.build(driver="builtin", options={"coord_type": "spherical", "eri_representation": "s8"})

    helper = sfx2c1e.SpinFreeX2CHelper(mol.topyscf())
    helper.xuncontract = False
    ref = helper.get_hcore()
    pyscf_labels = [" ".join(label.split()) for label in mol.topyscf().ao_labels()]
    pyqed_labels = [" ".join(label.split()) for label in mol.ao_labels()]
    perm = [pyscf_labels.index(label) for label in pyqed_labels]

    np.testing.assert_allclose(x2c1e_hcore(mol), ref[np.ix_(perm, perm)], atol=1e-10)
