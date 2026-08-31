import numpy as np
from pyscf import gto as pyscf_gto
from pyscf import mp as pyscf_mp
from pyscf import scf as pyscf_scf

from pyqed.qchem import COMP2, MP2, UMP2, Molecule
from pyqed.qchem.hf import RHF, UHF


def test_rmp2_matches_pyscf_on_h2o_sto3g():
    atom = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'

    mol = Molecule(atom=atom, basis='sto3g', unit='angstrom')
    mol.build()
    mf = RHF(mol).run()
    mymp = MP2(mf).run()

    pmol = pyscf_gto.M(atom=atom, basis='sto3g', unit='angstrom', verbose=0)
    pmf = pyscf_scf.RHF(pmol).run(verbose=0)
    pmp = pyscf_mp.MP2(pmf).run(verbose=0)

    np.testing.assert_allclose(mymp.e_corr, pmp.e_corr, atol=1e-7)
    np.testing.assert_allclose(mymp.e_tot, pmp.e_tot, atol=1e-7)

    dm1 = mymp.make_rdm1(ao_repr=True)
    dm2 = mymp.make_rdm2(ao_repr=True)
    ref_dm1 = pmp.make_rdm1(ao_repr=True)
    ref_dm2 = pmp.make_rdm2(ao_repr=True)
    s = pmol.intor('int1e_ovlp')

    np.testing.assert_allclose(dm1, ref_dm1, atol=1e-5)
    np.testing.assert_allclose(dm2, ref_dm2, atol=1e-5)
    np.testing.assert_allclose(np.einsum('pq,qp->', s, dm1), mol.nelec, atol=1e-8)


def test_ump2_matches_pyscf_on_li_sto3g():
    atom = 'Li 0 0 0'

    mol = Molecule(atom=atom, basis='sto3g', unit='angstrom', spin=1)
    mol.build()
    mf = UHF(mol).run()
    mymp = UMP2(mf).run()

    pmol = pyscf_gto.M(atom=atom, basis='sto3g', unit='angstrom', spin=1, verbose=0)
    pmf = pyscf_scf.UHF(pmol).run(verbose=0)
    pmp = pyscf_mp.UMP2(pmf).run(verbose=0)

    np.testing.assert_allclose(mymp.e_corr, pmp.e_corr, atol=1e-7)
    np.testing.assert_allclose(mymp.e_tot, pmp.e_tot, atol=1e-7)

    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = mymp.make_rdm12(ao_repr=True)
    ref_dm1a, ref_dm1b = pmp.make_rdm1(ao_repr=True)
    ref_dm2aa, ref_dm2ab, ref_dm2bb = pmp.make_rdm2(ao_repr=True)
    s = pmol.intor('int1e_ovlp')

    np.testing.assert_allclose(dm1a, ref_dm1a, atol=1e-5)
    np.testing.assert_allclose(dm1b, ref_dm1b, atol=1e-5)
    np.testing.assert_allclose(dm2aa, ref_dm2aa, atol=1e-5)
    np.testing.assert_allclose(dm2ab, ref_dm2ab, atol=1e-5)
    np.testing.assert_allclose(dm2bb, ref_dm2bb, atol=1e-5)
    nelec = np.einsum('pq,qp->', s, dm1a + dm1b)
    np.testing.assert_allclose(nelec, mol.nelec, atol=1e-8)


def test_comp2_runs_and_preserves_rhf_constraints():
    atom = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'

    mol = Molecule(atom=atom, basis='sto3g', unit='angstrom')
    mol.build()
    mf = RHF(mol).run()
    comp2 = COMP2(mf, max_cycle=4, optimizer_max_steps=20).run()

    assert np.isfinite(comp2.e_corr)
    assert np.isfinite(comp2.e_tot)
    assert len(comp2.energy_history) >= 1
    if len(comp2.energy_history) > 1:
        assert np.all(np.diff(comp2.energy_history) <= 1.0e-8)

    s = mol.overlap
    np.testing.assert_allclose(comp2.mo_coeff.conj().T @ s @ comp2.mo_coeff, np.eye(comp2.nmo), atol=1e-8)
    fock_oo = comp2.fock_mo[:comp2.nocc, :comp2.nocc].copy()
    fock_vv = comp2.fock_mo[comp2.nocc:, comp2.nocc:].copy()
    fock_oo[np.diag_indices_from(fock_oo)] = 0.0
    fock_vv[np.diag_indices_from(fock_vv)] = 0.0
    np.testing.assert_allclose(fock_oo, 0.0, atol=1e-8)
    np.testing.assert_allclose(fock_vv, 0.0, atol=1e-8)

    dm1 = comp2.make_rdm1(ao_repr=True)
    nelec = np.einsum('pq,qp->', s, dm1)
    np.testing.assert_allclose(nelec, mol.nelec, atol=1e-8)


def test_rmp2_factor_backend_matches_dense_builtin():
    atom = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'

    dense_mol = Molecule(atom=atom, basis='sto3g', unit='angstrom')
    dense_mol.build(options={'coord_type': 'spherical', 'eri_representation': 'dense', 'aosym': 's1'})
    dense_mf = RHF(dense_mol).run()
    dense_mp2 = MP2(dense_mf).run()

    factor_mol = Molecule(atom=atom, basis='sto3g', unit='angstrom')
    factor_mol.build(options={'coord_type': 'spherical', 'eri_representation': 'factors', 'low_rank_tol': 1e-8},
    )
    factor_mf = RHF(factor_mol).run()
    factor_mp2 = MP2(factor_mf).run()

    assert dense_mp2.eri_backend == 'dense'
    assert factor_mp2.eri_backend == 'factors'
    np.testing.assert_allclose(factor_mp2.e_corr, dense_mp2.e_corr, atol=1e-8)
    np.testing.assert_allclose(factor_mp2.e_tot, dense_mp2.e_tot, atol=1e-8)
    np.testing.assert_allclose(
        factor_mp2.make_rdm1(ao_repr=True),
        dense_mp2.make_rdm1(ao_repr=True),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        factor_mp2.make_rdm2(ao_repr=True),
        dense_mp2.make_rdm2(ao_repr=True),
        atol=1e-6,
    )


def test_ump2_factor_backend_matches_dense_builtin():
    atom = 'Li 0 0 0'

    dense_mol = Molecule(atom=atom, basis='sto3g', unit='angstrom', spin=1)
    dense_mol.build(options={'coord_type': 'spherical', 'eri_representation': 'dense', 'aosym': 's1'})
    dense_mf = UHF(dense_mol).run()
    dense_mp2 = UMP2(dense_mf).run()

    factor_mol = Molecule(atom=atom, basis='sto3g', unit='angstrom', spin=1)
    factor_mol.build(options={'coord_type': 'spherical', 'eri_representation': 'dense+factors', 'aosym': 's1', 'low_rank_tol': 1e-8},
    )
    factor_mf = UHF(factor_mol).run()
    factor_mp2 = UMP2(factor_mf).run()

    assert dense_mp2.eri_backend == 'dense'
    assert factor_mp2.eri_backend == 'factors'
    np.testing.assert_allclose(factor_mp2.e_corr, dense_mp2.e_corr, atol=1e-8)
    np.testing.assert_allclose(factor_mp2.e_tot, dense_mp2.e_tot, atol=1e-8)

    dense_dm1 = dense_mp2.make_rdm1(ao_repr=True)
    factor_dm1 = factor_mp2.make_rdm1(ao_repr=True)
    dense_dm2 = dense_mp2.make_rdm2(ao_repr=True)
    factor_dm2 = factor_mp2.make_rdm2(ao_repr=True)

    np.testing.assert_allclose(factor_dm1[0], dense_dm1[0], atol=1e-7)
    np.testing.assert_allclose(factor_dm1[1], dense_dm1[1], atol=1e-7)
    np.testing.assert_allclose(factor_dm2[0], dense_dm2[0], atol=1e-6)
    np.testing.assert_allclose(factor_dm2[1], dense_dm2[1], atol=1e-6)
    np.testing.assert_allclose(factor_dm2[2], dense_dm2[2], atol=1e-6)


def test_comp2_factor_backend_matches_dense_builtin():
    atom = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'

    dense_mol = Molecule(atom=atom, basis='sto3g', unit='angstrom')
    dense_mol.build(options={'coord_type': 'spherical', 'eri_representation': 'dense', 'aosym': 's1'})
    dense_mf = RHF(dense_mol).run()
    dense_comp2 = COMP2(dense_mf, max_cycle=4, optimizer_max_steps=20).run()

    factor_mol = Molecule(atom=atom, basis='sto3g', unit='angstrom')
    factor_mol.build(options={'coord_type': 'spherical', 'eri_representation': 'factors', 'low_rank_tol': 1e-8},
    )
    factor_mf = RHF(factor_mol).run()
    factor_comp2 = COMP2(factor_mf, max_cycle=4, optimizer_max_steps=20).run()

    assert dense_comp2.eri_backend == 'dense'
    assert factor_comp2.eri_backend == 'factors'
    np.testing.assert_allclose(factor_comp2.e_tot, dense_comp2.e_tot, atol=1e-8)
    np.testing.assert_allclose(factor_comp2.e_corr, dense_comp2.e_corr, atol=1e-8)
    np.testing.assert_allclose(
        factor_comp2.make_rdm1(ao_repr=True),
        dense_comp2.make_rdm1(ao_repr=True),
        atol=1e-6,
    )
