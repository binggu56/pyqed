import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF, UHF


def test_uhf_open_shell_li_smoke():
    mol = Molecule(atom='Li 0 0 0', unit='bohr', basis='sto-3g', spin=1)
    mol.build(driver='gbasis')

    mf = UHF(mol).run()

    assert mf.converged
    assert mf.na == 2
    assert mf.nb == 1
    assert np.isfinite(mf.e_tot)
    np.testing.assert_allclose(
        np.einsum('ij,ji->', mol.overlap, mf.dm[0]).real,
        mf.na,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.einsum('ij,ji->', mol.overlap, mf.dm[1]).real,
        mf.nb,
        atol=1e-8,
    )


def test_rhf_to_uhf_preserves_closed_shell_reference():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    rhf = RHF(mol).run()
    uhf = rhf.to_uhf()

    assert uhf.converged
    assert uhf.nocc == (1, 1)
    np.testing.assert_allclose(uhf.e_tot, rhf.e_tot)
    np.testing.assert_allclose(uhf.dm[0] + uhf.dm[1], rhf.dm)


def test_uhf_matches_pyscf_for_open_shell_li():
    pyscf = pytest.importorskip('pyscf')

    mol = Molecule(atom='Li 0 0 0', unit='bohr', basis='sto-3g', spin=1)
    mol.build(driver='gbasis')

    mf = UHF(mol).run()

    pmf = pyscf.scf.UHF(mol.topyscf())
    pmf.conv_tol = 1e-10
    pmf.kernel()

    np.testing.assert_allclose(mf.e_tot, pmf.e_tot, atol=1e-6)
