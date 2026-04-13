import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.dft import RKS, TDA, TDDFT


def test_lrtddft_lda_smoke():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RKS(mol, xc='lda').run()

    tda = TDA(mf).run(nstates=1)
    tddft = TDDFT(mf).run(nstates=1)

    assert tda.a.shape == (1, 1, 1, 1)
    assert tddft.a.shape == (1, 1, 1, 1)
    assert np.isfinite(tda.e[0])
    assert np.isfinite(tddft.e[0])
    assert tda.e[0] > 0.0
    assert tddft.e[0] > 0.0
