"""Small molecular fixtures for reduced DMRG validation."""
import numpy as np
import pytest
from pyqed.qchem import Molecule, RHF, CASSCF


@pytest.fixture(scope='module')
def h3plus():
    mol = Molecule(atom='H 1 0 0; H -0.5 0.8660254037844386 0; H -0.5 -0.8660254037844386 0',
                   unit='bohr', basis='sto-3g', charge=1, symmetry='D3h')
    mol.build()
    return RHF(mol).run(verbose=0)


@pytest.fixture(scope='module')
def benzene():
    from examples.qchem.validate_molecular_symmetry import geometry
    atom, unit, charge, *_ = geometry('benzene')
    mol = Molecule(atom=atom, unit=unit, basis='sto-3g', symmetry=True).build(eri='s8')
    assert mol.groupname == 'D6h'
    mf = RHF(mol); mf.chkfile = None
    mf.run(verbose=0, tol=1e-12, conv_tol_grad=1e-8)
    pz = [i for i, label in enumerate(mol.ao_labels()) if '2pz' in label]
    active = np.flatnonzero(np.sum(mf.mo_coeff[pz]**2, axis=0) > .1)
    assert len(active) == 6
    coeff = CASSCF(mf, 6, 6).reorder_mo_for_active_orbitals(mf.mo_coeff, active)
    return mf, coeff


@pytest.fixture(scope='module')
def independent_integrals(benzene):
    pytest.importorskip('pyscf')
    from pyscf import scf, mcscf, ao2mo
    mf, _ = benzene
    pmol = mf.mol.topyscf(); pmol.verbose = 0
    pmf = scf.RHF(pmol).run(conv_tol=1e-12, conv_tol_grad=1e-8)
    np.testing.assert_allclose(mf.e_tot, pmf.e_tot, atol=1e-7, rtol=0)
    permutation = mf.mol.pyscf_ao_permutation(pmol)
    def transform(coeff):
        mapped = np.empty_like(coeff); mapped[permutation] = coeff
        pc = mcscf.CASCI(pmf, 6, 6)
        h, core = pc.get_h1eff(mapped)
        return h, ao2mo.restore(1, pc.get_h2eff(mapped), 6), core
    return transform
