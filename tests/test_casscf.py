import numpy as np

from pyqed.qchem import Molecule, CASSCF
from pyqed.qchem.mcscf.direct_ci import CASCI


def test_casscf_lih_lowers_the_initial_casci_energy():
    """Exercise the native U-matrix orbital optimizer on a nontrivial case."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()

    mc0 = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method='ci')
    mc = CASSCF(mf, ncas=2, nelecas=2, max_cycles=20).run()

    assert mc.e_tot[0] < mc0.e_tot[0] - 1e-6
    assert np.isfinite(mc.e_tot[0])
