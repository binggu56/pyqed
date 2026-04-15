import pytest
import numpy as np

from pyqed.qchem import CASSCF, COCASCI, Molecule, soc_state_interaction
from pyqed.qchem.mcscf import casci as casci_module
from pyqed.qchem.mcscf import direct_ci as direct_ci_module
from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed.qchem.soc import (
    get_soc_1e_ao,
    get_soc_2e_somf_ao,
    get_soc_somf_spin_orbital,
    soc_1e_prefactor,
)


def _grouped_spin_orbital_occ(binary):
    return np.concatenate((binary[0], binary[1])).astype(np.int8)


def _bruteforce_tdm1_spin_orbital(mc_bra, bra_id, mc_ket, ket_id):
    bra_occ = [_grouped_spin_orbital_occ(det) for det in mc_bra.binary]
    ket_occ = [_grouped_spin_orbital_occ(det) for det in mc_ket.binary]
    bra_lookup = {occ.tobytes(): idx for idx, occ in enumerate(bra_occ)}

    cibra = np.asarray(mc_bra.ci[bra_id])
    ciket = np.asarray(mc_ket.ci[ket_id])
    nso = 2 * mc_bra.ncas
    tdm = np.zeros((nso, nso), dtype=complex)

    for j, occ in enumerate(ket_occ):
        occupied = np.flatnonzero(occ)
        for v in occupied:
            sign_ann = -1 if int(np.sum(occ[:v])) % 2 else 1
            occ_after_ann = occ.copy()
            occ_after_ann[v] = 0
            unoccupied = np.flatnonzero(1 - occ_after_ann)
            for u in unoccupied:
                sign_cre = -1 if int(np.sum(occ_after_ann[:u])) % 2 else 1
                occ_final = occ_after_ann.copy()
                occ_final[u] = 1
                i = bra_lookup.get(occ_final.tobytes())
                if i is None:
                    continue
                tdm[u, v] += cibra[i].conj() * ciket[j] * (sign_ann * sign_cre)

    return tdm


def test_casscf_lih_lowers_the_initial_casci_energy():
    """Exercise the native U-matrix orbital optimizer on a nontrivial case."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()

    mc0 = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method='ci')
    mc = COCASCI(mf, ncas=2, nelecas=2, max_cycles=20).run()

    assert mc.e_tot[0] < mc0.e_tot[0] - 1e-6
    assert np.isfinite(mc.e_tot[0])


def test_casscf_lih_lbfgs_matches_rcg_energy():
    """The alternative orbital optimizer should reach the same LiH minimum."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()

    mc_rcg = COCASCI(mf, ncas=2, nelecas=2, max_cycles=20, optimizer='RCG').run()
    mc_lbfgs = COCASCI(mf, ncas=2, nelecas=2, max_cycles=20, optimizer='LBFGS').run()

    np.testing.assert_allclose(mc_lbfgs.e_tot, mc_rcg.e_tot, atol=1e-6)


def test_casscf_lih_diis_matches_non_diis_energy():
    """DIIS should accelerate the U updates without changing the LiH minimum."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()

    mc_plain = COCASCI(
        mf, ncas=2, nelecas=2, max_cycles=20, optimizer='RCG', diis=False,
    ).run()
    mc_diis = COCASCI(
        mf, ncas=2, nelecas=2, max_cycles=20, optimizer='RCG', diis=True,
    ).run()

    np.testing.assert_allclose(mc_diis.e_tot, mc_plain.e_tot, atol=1e-6)


def test_first_order_casscf_lih_lowers_initial_casci_energy():
    """The public CASSCF class should improve on the initial CASCI reference."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()

    mc0 = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method='direct_ci')
    mc = CASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=40,
        ci_method='direct_ci',
    ).run()

    assert mc.e_tot[0] < mc0.e_tot[0] - 1e-6
    assert mc.converged
    assert np.isfinite(mc.e_tot[0])


def test_first_order_casscf_repeated_run_resets_history():
    """Reusing the same CASSCF object should start from a clean history."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc = CASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=40,
        ci_method='direct_ci',
    )

    mc.run()
    history_len = len(mc.history)
    assert history_len > 0
    assert "step_norm" in mc.history[-1]

    mc.run()
    assert len(mc.history) == history_len
    assert mc.converged


def test_cocasci_lih_4e4o_matches_first_order_casscf():
    """Regression test for the LiH (4e,4o) COCASCI energy consistency bug."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='pyscf')

    mf = mol.RHF().run()

    mc_ref = CASSCF(
        mf,
        ncas=4,
        nelecas=4,
        max_cycle=60,
        ci_method='direct_ci',
    ).run()
    mc_opt = COCASCI(
        mf,
        ncas=4,
        nelecas=4,
        max_cycles=20,
        ci_method='direct_ci',
    ).run()

    np.testing.assert_allclose(mc_opt.e_tot, mc_ref.e_tot, atol=1e-6)


def test_first_order_casscf_state_average_two_roots():
    """State-averaged first-order CASSCF should support 2-root optimization."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='pyscf')

    mf = mol.RHF().run()

    mc = CASSCF(
        mf,
        ncas=4,
        nelecas=4,
        max_cycle=60,
        ci_method='direct_ci',
    )
    mc.state_average([0.5, 0.5]).run(nstates=2)

    assert mc.converged
    assert len(mc.e_tot) == 2
    assert np.all(np.isfinite(mc.e_tot))
    assert mc.e_tot[0] < mc.e_tot[1]


def test_casci_overlap_self_is_identity():
    """The determinant-overlap contraction should preserve orthonormal roots."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='pyscf')

    mf = mol.RHF().run()
    mc = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='direct_ci')

    smo = mf.mo_coeff.T @ mf.get_ovlp() @ mf.mo_coeff
    ref = np.eye(2)

    np.testing.assert_allclose(direct_ci_module.overlap(mc, mc, s=smo), ref, atol=1e-10)
    np.testing.assert_allclose(casci_module.overlap(mc, mc, s=smo), ref, atol=1e-10)
    np.testing.assert_allclose(direct_ci_module.overlap(mc, mc), ref, atol=1e-10)
    np.testing.assert_allclose(casci_module.overlap(mc, mc), ref, atol=1e-10)


def test_casci_spin_orbital_tdm_matches_spin_blocks():
    """Spin-orbital TDMs should reduce to the alpha/beta RDM blocks on the diagonal."""
    mol = Molecule(atom='H 0 0 0; H 0 0 0.74', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc = CASCI(mf, ncas=2, nelecas=2, spin=0).run(nstates=1, method='direct_ci')

    dm1a, dm1b = mc.make_rdm1s(0)
    dm1so = mc.make_tdm1_spin_orbital(0, order='grouped')
    ncas = mc.ncas

    np.testing.assert_allclose(dm1so[:ncas, :ncas], dm1a, atol=1e-10)
    np.testing.assert_allclose(dm1so[ncas:, ncas:], dm1b, atol=1e-10)
    np.testing.assert_allclose(dm1so[:ncas, ncas:], 0.0, atol=1e-10)
    np.testing.assert_allclose(dm1so[ncas:, :ncas], 0.0, atol=1e-10)


def test_casci_spin_orbital_tdm_matches_bruteforce_between_spin_sectors():
    """Spin-flip blocks should agree with an explicit determinant-space contraction."""
    mol = Molecule(atom='H 0 0 0; H 0 0 0.74', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc_singlet = CASCI(mf, ncas=2, nelecas=2, spin=0).run(nstates=1, method='direct_ci')
    mc_triplet = CASCI(mf, ncas=2, nelecas=2, spin=2).run(nstates=1, method='direct_ci')

    tdm_ref = _bruteforce_tdm1_spin_orbital(mc_singlet, 0, mc_triplet, 0)
    tdm = mc_singlet.make_tdm1_spin_orbital(0, other=mc_triplet, order='grouped')
    np.testing.assert_allclose(tdm, tdm_ref, atol=1e-10)

    hso = np.zeros((2 * mc_singlet.ncas, 2 * mc_singlet.ncas), dtype=complex)
    hso[0, 2] = 0.37 + 0.11j
    hso[1, 3] = -0.19 + 0.23j
    value = mc_singlet.contract_with_tdm1_spin_orbital(0, other=mc_triplet, h1e=hso)
    value_ref = np.einsum('uv,uv->', hso, tdm_ref)
    np.testing.assert_allclose(value, value_ref, atol=1e-10)


def test_soc_state_interaction_builds_hermitian_matrix():
    """The SOC SI helper should build a Hermitian total Hamiltonian."""
    mol = Molecule(atom='H 0 0 0; H 0 0 0.74', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc_singlet = CASCI(mf, ncas=2, nelecas=2, spin=0).run(nstates=1, method='direct_ci')
    mc_triplet = CASCI(mf, ncas=2, nelecas=2, spin=2).run(nstates=1, method='direct_ci')

    hso = np.zeros((2 * mc_singlet.ncas, 2 * mc_singlet.ncas), dtype=complex)
    hso[0, 2] = 0.37 + 0.11j
    hso[1, 3] = -0.19 + 0.23j
    hso[2, 0] = hso[0, 2].conjugate()
    hso[3, 1] = hso[1, 3].conjugate()

    result = soc_state_interaction([(mc_singlet, 0), (mc_triplet, 0)], hso=hso)

    np.testing.assert_allclose(result.h_soc, result.h_soc.conjugate().T, atol=1e-10)
    np.testing.assert_allclose(result.h_total, result.h_total.conjugate().T, atol=1e-10)
    np.testing.assert_allclose(
        result.h_soc[0, 1],
        mc_singlet.soc_matrix_element(0, other=mc_triplet, hso=hso),
        atol=1e-10,
    )
    assert result.eigenvalues.shape == (2,)
    assert result.eigenvectors.shape == (2, 2)


@pytest.mark.parametrize(
    ("atom", "basis"),
    [
        (
            '''
            Li 0 0 0
            H  0 0 1.6
            ''',
            'sto-3g',
        ),
        (
            '''
            S  0.000000  0.000000  0.000000
            H  0.000000  1.229000  0.958000
            H  0.000000 -1.229000  0.958000
            ''',
            'sto-3g',
        ),
    ],
)
def test_gbasis_pyscf_soc_ao_matches_direct_pyscf_operator(atom, basis):
    """The gbasis-pyscf SOC AO build should reproduce the PySCF one-center operator."""
    from pyscf import gto

    mol = Molecule(atom=atom, unit='angstrom', basis=basis)
    mol.build(driver='gbasis-pyscf')
    hso = get_soc_1e_ao(mol, one_center=True)

    pmol = gto.M(atom=atom, basis=basis, unit='angstrom', verbose=0)
    hso_ref = np.zeros_like(hso)
    aoslices = pmol.aoslice_by_atom()
    for ia in range(pmol.natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        with pmol.with_rinv_as_nucleus(ia):
            w = pmol.intor('int1e_prinvxp', comp=3)
        hso_ref[:, p0:p1, p0:p1] += (-pmol.atom_charge(ia)) * w[:, p0:p1, p0:p1]
    hso_ref *= soc_1e_prefactor()

    np.testing.assert_allclose(hso, hso_ref, atol=1e-11)


def test_gbasis_pyscf_somf_ao_matches_manual_contraction():
    """The SOMF AO builder should match a direct PySCF int2e_p1vxp1 contraction."""
    from pyscf import gto

    atom = '''
        S  0.000000  0.000000  0.000000
        H  0.000000  1.229000  0.958000
        H  0.000000 -1.229000  0.958000
    '''
    basis = 'sto-3g'

    mol = Molecule(atom=atom, unit='angstrom', basis=basis)
    mol.build(driver='gbasis-pyscf')
    mf = mol.RHF().run()
    hso = get_soc_2e_somf_ao(mf)

    pmol = gto.M(atom=atom, basis=basis, unit='angstrom', verbose=0)
    dm = mf.make_rdm1()
    g = pmol.intor('int2e_p1vxp1', comp=3)
    hso_ref = (
        np.einsum('xpqrs,rs->xpq', g, dm, optimize=True)
        - 1.5 * np.einsum('xprsq,rs->xpq', g, dm, optimize=True)
        - 1.5 * np.einsum('xsqpr,rs->xpq', g, dm, optimize=True)
    ) * soc_1e_prefactor()

    np.testing.assert_allclose(hso, hso_ref, atol=1e-11)


def test_soc_state_interaction_somf_matches_explicit_operator():
    """The SOMF SI helper should use the same state-averaged operator as the explicit path."""
    mol = Molecule(
        atom='''
            S  0.000000  0.000000  0.000000
            H  0.000000  1.229000  0.958000
            H  0.000000 -1.229000  0.958000
        ''',
        unit='angstrom',
        basis='sto-3g',
    )
    mol.build(driver='gbasis-pyscf')

    mf = mol.RHF().run()
    mc_singlet = CASCI(mf, ncas=2, nelecas=2, spin=0).run(nstates=1, method='direct_ci')
    mc_triplet = CASCI(mf, ncas=2, nelecas=2, spin=2).run(nstates=1, method='direct_ci')
    states = [(mc_singlet, 0), (mc_triplet, 0)]

    hso = get_soc_somf_spin_orbital(
        mf,
        representation='mo',
        mo_coeff=mc_singlet.mo_cas,
        states=states,
        order='grouped',
    )
    result = soc_state_interaction(states, soc_model='somf')

    np.testing.assert_allclose(result.h_soc, result.h_soc.conjugate().T, atol=1e-10)
    np.testing.assert_allclose(
        result.h_soc[0, 1],
        mc_singlet.soc_matrix_element(0, other=mc_triplet, hso=hso),
        atol=1e-10,
    )


def test_first_order_casscf_line_search_failure_raises(monkeypatch):
    """A rejected orbital step should surface as a RuntimeError."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc = CASSCF(mf, ncas=2, nelecas=2, max_cycle=4, ci_method='direct_ci')

    def fail_line_search(*args, **kwargs):
        return False, args[0], float("inf"), 0.0, None

    monkeypatch.setattr(mc, "_line_search", fail_line_search)

    with pytest.raises(RuntimeError, match="line search failed"):
        mc.run()
