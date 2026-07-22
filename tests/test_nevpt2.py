import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf import direct_ci
from pyqed.qchem.mcscf import nevpt2 as nevpt2_module
from pyqed.qchem.mcscf.nevpt2 import NEVPT2


def _lih_cas22():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')
    mf = RHF(mol).run()
    return direct_ci.CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method='direct_ci')


def _lih_cas44():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')
    mf = RHF(mol).run()
    return direct_ci.CASCI(mf, ncas=4, nelecas=4).run(nstates=1, method='direct_ci')


def test_nevpt2_sijrs_component_matches_manual_core_virtual_formula():
    mc = _lih_cas22()
    pt = NEVPT2(mc, classes=('Sijrs',))

    e_corr = pt.run()
    component = pt.components['Sijrs']

    mo_core, _mo_cas, mo_virt = pt._orbital_spaces()
    eri = pt._eri_mo(mo_core, mo_virt, mo_core, mo_virt)
    eps = pt._mo_energy()
    ncore = mc.ncore
    nocc = mc.ncore + mc.ncas
    theta = 2.0 * eri - eri.swapaxes(1, 3)
    denom = (
        eps[:ncore, None, None, None]
        + eps[None, None, :ncore, None]
        - eps[None, nocc:, None, None]
        - eps[None, None, None, nocc:]
    )
    expected_norm = np.einsum('iajb,iajb->', eri, theta, optimize=True)
    expected_energy = np.einsum('iajb,iajb->', eri * theta, 1.0 / denom, optimize=True)

    np.testing.assert_allclose(component.norm, expected_norm, atol=1e-14)
    np.testing.assert_allclose(component.energy, expected_energy, atol=1e-14)
    np.testing.assert_allclose(e_corr, expected_energy, atol=1e-14)
    assert e_corr < 0.0


def test_nevpt2_full_run_accumulates_all_sc_classes():
    mc = _lih_cas44()

    pt = NEVPT2(mc)
    e_corr = pt.run()

    assert tuple(pt.components) == NEVPT2.all_classes
    np.testing.assert_allclose(
        e_corr,
        sum(component.energy for component in pt.components.values()),
        atol=1e-14,
    )
    assert e_corr < 0.0


def test_nevpt2_4rdm_cap_can_skip_singly_external_classes():
    mc = _lih_cas22()
    old_cpp = nevpt2_module._casscf_cpp
    nevpt2_module._casscf_cpp = None

    try:
        with pytest.raises(NotImplementedError, match='4-RDM'):
            NEVPT2(mc, max_4rdm_ncas=1).run()

        partial = NEVPT2(
            mc,
            classes=('Sijrs', 'Sijr', 'Srsi', 'Sij', 'Srs', 'Sir'),
            max_4rdm_ncas=1,
        ).run()
    finally:
        nevpt2_module._casscf_cpp = old_cpp
    assert partial < 0.0


def test_nevpt2_cpp_spin_free_rdms123_match_python_reference():
    cpp = nevpt2_module._casscf_cpp
    if cpp is None or not hasattr(cpp, 'nevpt_spin_free_rdms123'):
        pytest.skip('spin-free NEVPT2 C++ RDM builder is not built')

    mc = _lih_cas44()
    ci = mc.ci[0]
    ref = nevpt2_module._spin_free_rdms123_python(ci, mc.binary)
    got = cpp.nevpt_spin_free_rdms123(ci, mc.binary)
    for actual, expected in zip(got, ref):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_nevpt2_cpp_contracted_a16_a22_match_full_4rdm_terms():
    cpp = nevpt2_module._casscf_cpp
    if (
        cpp is None
        or not hasattr(cpp, 'nevpt_a16_a22_4rdm_terms')
    ):
        pytest.skip('contracted NEVPT2 C++ kernels are not built')

    mc = _lih_cas44()
    pt = NEVPT2(mc)
    blocks = pt._integral_blocks()
    dms = pt._dms(include_rdm4=True)
    ci = pt._ci_vector()
    binary = mc.binary
    h2e = blocks['h2e']
    dm4 = dms['4']

    a16_terms, a22_terms = cpp.nevpt_a16_a22_4rdm_terms(h2e, ci, binary)
    ca1, ac, ca2 = a16_terms
    np.testing.assert_allclose(
        ca1,
        np.einsum('kbij,rpqjkiac->pqrabc', h2e, dm4, optimize=True),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ac,
        np.einsum('ijka,rpqbjcik->pqrabc', h2e, dm4, optimize=True),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ca2,
        np.einsum('kcij,rpqbajki->pqrabc', h2e, dm4, optimize=True),
        atol=1e-12,
    )

    ac1, ac2, ca = a22_terms
    np.testing.assert_allclose(
        ac1,
        np.einsum('pqrb,kiqjprac->ijkabc', h2e, dm4, optimize=True),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ac2,
        np.einsum('pqra,kibjqcpr->ijkabc', h2e, dm4, optimize=True),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ca,
        np.einsum('rcpq,kibjaqrp->ijkabc', h2e, dm4, optimize=True),
        atol=1e-12,
    )


def test_nevpt2_cpp_a22_4rdm_energy_matches_full_tensor_difference():
    cpp = nevpt2_module._casscf_cpp
    if cpp is None or not hasattr(cpp, 'nevpt_a22_4rdm_energy'):
        pytest.skip('direct NEVPT2 A22 energy kernel is not built')

    mc = _lih_cas44()
    pt = NEVPT2(mc)
    blocks = pt._integral_blocks()
    dms = pt._dms(include_rdm4=True)
    h2e_v = blocks['ppaa'][pt._ncore:pt._nocc, :pt._ncore].transpose(0, 2, 1, 3)
    a22_full = nevpt2_module._make_a22(blocks['h1e'], blocks['h2e'], dms)
    a22_dense = nevpt2_module._make_a22(blocks['h1e'], blocks['h2e'], dms, include_4rdm=False)

    expected = np.einsum(
        'qpir,pqrabc,baic->i',
        h2e_v,
        a22_full - a22_dense,
        h2e_v,
        optimize=True,
    )
    got = cpp.nevpt_a22_4rdm_energy(
        blocks['h2e'],
        h2e_v,
        pt._ci_vector(),
        mc.binary,
    )

    np.testing.assert_allclose(got, expected, atol=1e-11)


def test_nevpt2_theory_note_describes_sc_dyall_setup():
    theory = NEVPT2.theory()

    assert 'Strongly contracted' in theory
    assert "Dyall's Hamiltonian" in theory
    assert '|Phi_mu> = P_mu V |Psi0>' in theory


def test_nevpt2_lithium_hydride_tracks_pyscf_reference():
    pyscf_gto = pytest.importorskip('pyscf.gto')
    pyscf_scf = pytest.importorskip('pyscf.scf')
    pyscf_mcscf = pytest.importorskip('pyscf.mcscf')
    pyscf_mrpt = pytest.importorskip('pyscf.mrpt')

    atom = 'Li 0 0 0; H 0 0 1.6'
    mc = _lih_cas22()
    e_native = NEVPT2(mc).run()

    mol = pyscf_gto.M(atom=atom, unit='angstrom', basis='sto-3g', verbose=0)
    mf = pyscf_scf.RHF(mol).run(verbose=0)
    pmc = pyscf_mcscf.CASCI(mf, 2, 2).run(verbose=0)
    e_ref = pyscf_mrpt.NEVPT(pmc).kernel()

    np.testing.assert_allclose(e_native, e_ref, atol=1.0e-5)
