import numpy as np

from pyqed.qchem import Molecule


def test_native_build_is_default_and_produces_ao_tensors():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    assert mol._build_driver == 'native'
    assert mol.nao == 2
    assert mol.overlap.shape == (2, 2)
    assert mol.hcore.shape == (2, 2)
    assert mol.eri.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(np.diag(mol.overlap), np.ones(2), atol=1e-12)


def test_native_build_runs_rhf_without_external_integral_backends():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='native')

    mf = mol.RHF().run(max_cycle=60)
    assert np.isfinite(mf.e_tot)
