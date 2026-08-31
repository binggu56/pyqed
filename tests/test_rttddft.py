import numpy as np

from pyqed.qchem import Molecule, RTTDDFT
from pyqed.qchem.dft import RKS


def test_rttddft_ground_state_is_stationary_without_field():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RKS(mol, xc='lda').run()
    rt = RTTDDFT(mf).run(dt=0.05, nsteps=5, store_dm=True)

    np.testing.assert_allclose(rt.dms[-1], rt.dms[0], atol=1e-8)
    np.testing.assert_allclose(rt.energies[-1], rt.energies[0], atol=1e-8)
    np.testing.assert_allclose(rt.electron_count(), mol.nelec, atol=1e-8)
