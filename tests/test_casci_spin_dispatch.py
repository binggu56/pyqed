import numpy as np
from pyqed.qchem import Molecule, RHF, CASCI
from pyqed.qchem.mcscf.casci import spin_square


def test_public_spin_square_reuses_direct_action(monkeypatch):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()
    mc = CASCI(RHF(mol).run(), ncas=2, nelecas=2, ms2=0).run(nstates=3)
    reference = [spin_square(*mc.make_rdm12(i)) for i in range(3)]
    def reject(*args, **kwargs):
        raise AssertionError('Direct-CI spin diagnostics must not construct RDMs')
    monkeypatch.setattr(mc, 'make_rdm12', reject)
    np.testing.assert_allclose([mc.spin_square(i) for i in range(3)], reference, atol=1e-10)


def test_replaced_ci_does_not_use_stale_solver(monkeypatch):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()
    mc = CASCI(RHF(mol).run(), ncas=2, nelecas=2, ms2=0).run(nstates=3)
    mc.ci = np.asarray(mc.ci).copy()[::-1]
    reference = spin_square(*mc.make_rdm12(0))
    def reject(*args, **kwargs):
        raise AssertionError('Stale direct result was used')
    monkeypatch.setattr(mc._direct_solver, 'spin_square', reject)
    np.testing.assert_allclose(mc.spin_square(0), reference, atol=1e-10)
