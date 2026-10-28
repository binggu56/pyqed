"""Fragment-spin limits and independent PySCF comparisons."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyqed.qchem.mcscf.local_spin import local_spin, local_spin_from_rdms


def test_one_electron_and_closed_core_limits():
    assert local_spin_from_rdms(np.ones((1,1)), np.zeros((1,)*4), np.eye(1)) == .75
    empty1, empty2 = np.empty((0,0)), np.empty((0,)*4)
    assert local_spin_from_rdms(empty1, empty2, np.eye(1), ncore=1) == 0
    assert local_spin_from_rdms(empty1, empty2, np.array([[.5]]), ncore=1) == .375


@pytest.mark.parametrize('nelec', [(2,2),(2,1),(3,0)])
def test_restricted_contraction_matches_pyscf(nelec):
    pytest.importorskip('pyscf')
    from pyscf.fci import cistring, direct_spin1, spin_op
    rng = np.random.default_rng(174)
    n = 4
    ci = rng.normal(size=tuple(cistring.num_strings(n,k) for k in nelec))
    ci /= np.linalg.norm(ci)
    dm1, dm2 = direct_spin1.make_rdm12(ci,n,nelec)
    x = rng.normal(size=(6,6)); overlap = x.T@x+np.eye(6)
    vals, vecs = np.linalg.eigh(overlap)
    coefficients = ((vecs/np.sqrt(vals))@vecs.T)@np.linalg.qr(rng.normal(size=(6,6)))[0][:,:n]
    aos = [0,2,5]
    c = coefficients[aos]
    projected = c.T@overlap[np.ix_(aos,aos)]@c
    expected = spin_op.local_spin(ci,n,nelec,coefficients,overlap,aos)[0]
    assert local_spin_from_rdms(dm1,dm2,projected) == pytest.approx(expected,abs=2e-12)


def test_core_terms_match_explicit_fci_embedding():
    pytest.importorskip('pyscf')
    from pyscf.fci import cistring, direct_spin1, spin_op
    rng = np.random.default_rng(295)
    core, active, norb = 2, 3, 6
    nelec = (2,1)
    strings = [cistring.make_strings(range(active),n) for n in nelec]
    ci = rng.normal(size=tuple(map(len,strings))); ci /= np.linalg.norm(ci)
    d1,d2 = direct_spin1.make_rdm12(ci,active,nelec)
    full_strings = [cistring.make_strings(range(norb),n+core) for n in nelec]
    positions = [{int(b):i for i,b in enumerate(bits)} for bits in full_strings]
    full = np.zeros(tuple(map(len,full_strings)))
    for i,a in enumerate(strings[0]):
        for j,b in enumerate(strings[1]):
            full[positions[0][(int(a)<<core)|3],positions[1][(int(b)<<core)|3]] = ci[i,j]
    coefficients = np.linalg.qr(rng.normal(size=(norb,norb)))[0]
    aos = [0,2,3]
    p = coefficients[aos].T@coefficients[aos]
    expected = spin_op.local_spin(full,norb,tuple(n+core for n in nelec),coefficients,np.eye(norb),aos)[0]
    assert local_spin_from_rdms(d1,d2,p[:core+active,:core+active],ncore=core) == pytest.approx(expected,abs=2e-12)


@pytest.fixture(scope='module')
def molecular_solvers():
    from pyqed.qchem import Molecule
    from pyqed.qchem.mcscf.direct_ci import CASCI
    from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
    mol = Molecule(atom='H 0 0 0; H 0 0 .8; H .2 .1 2; H .1 0 2.9',
                   basis='sto-3g',unit='angstrom').build(eri='ri',auxbasis='def2-universal-jkfit')
    mf = mol.RHF().run()
    ci = CASCI(mf,2,2,multiplicity=1).run(nstates=2)
    mc = SecondOrderCASSCF(mf,2,2,multiplicity=1,ci_method='ci',
                          conv_tol_grad=1e-4,conv_tol_grad_relaxed=1e-4,verbose=0)
    mc.state_average([.5,.5]).run(nstates=2)
    assert ci.converged and mc.converged
    return ci,mc


@pytest.mark.parametrize('solver_id',[0,1])
def test_fragment_api_and_total_spin(molecular_solvers,solver_id,monkeypatch):
    mc = molecular_solvers[solver_id]
    for root in range(2):
        assert mc.local_spin(state_id=root) == pytest.approx(mc.spin_square(root),abs=2e-10)
        assert mc.local_spin([0,1],state_id=root) == pytest.approx(
            mc.local_spin(aolst=[0,1],state_id=root),abs=2e-12)
    assert mc.local_spin([]) == 0
    assert mc.local_spin(aolst=[]) == 0
    # The driver owns the optimized AO coefficients even for frozen-frame CI.
    if solver_id:
        expected = mc.local_spin([0,1])
        with monkeypatch.context() as patch:
            patch.setattr(mc.casci,'mo_coeff',np.eye(len(mc.mo_coeff)))
            assert mc.local_spin([0,1]) == pytest.approx(expected,abs=2e-12)
    # Guard against accidentally building a core-inclusive RDM.
    from pyqed.qchem.mcscf import casci
    original = casci.make_state_average_rdms
    def checked(roots,weights,binary):
        d1,d2 = original(roots,weights,binary)
        assert d1.shape == (mc.ncas,)*2 and d2.shape == (mc.ncas,)*4
        return d1,d2
    monkeypatch.setattr(casci,'make_state_average_rdms',checked)
    assert np.isfinite(mc.local_spin([0]))


@pytest.mark.parametrize('solver_id',[0,1])
@pytest.mark.parametrize('with_core',[False,True])
def test_molecular_api_matches_pyscf(molecular_solvers,solver_id,with_core):
    pytest.importorskip('pyscf')
    from pyscf.fci import cistring,spin_op
    from pyqed.qchem.mcscf.casci import _ci_to_spin_string_matrix
    mc = molecular_solvers[solver_id]
    target = getattr(mc,'casci',mc)
    core = target.ncore if with_core else 0
    n = core+target.ncas
    for root in range(2):
        alpha,beta,matrix = _ci_to_spin_string_matrix(target.ci[root],target.binary)
        counts = tuple(int(occ[0].sum())+core for occ in (alpha,beta))
        lists = [cistring.make_strings(range(n),k) for k in counts]
        positions = [{int(b):i for i,b in enumerate(bits)} for bits in lists]
        ci = np.zeros(tuple(map(len,lists)))
        for i,a in enumerate(alpha):
            for j,b in enumerate(beta):
                abits = sum(int(v)<<k for k,v in enumerate(a))
                bbits = sum(int(v)<<k for k,v in enumerate(b))
                ci[positions[0][(abits<<core)|((1<<core)-1)],
                   positions[1][(bbits<<core)|((1<<core)-1)]] = matrix[i,j]
        c = mc.mo_coeff[:,:n] if with_core else mc.mo_coeff[:,target.ncore:target.ncore+target.ncas]
        expected = spin_op.local_spin(ci,n,counts,c,mc.mf.get_ovlp(),[0,1])[0]
        assert mc.local_spin([0,1],state_id=root,with_core=with_core) == pytest.approx(expected,abs=2e-10)


@pytest.mark.parametrize('kwargs',[{'atoms':[-1]}, {'atoms':[4]}, {'atoms':[0,0]},
    {'atoms':[.5]}, {'atoms':[True]}, {'atoms':[0],'aolst':[0]}, {'aolst':[4]},
    {'state_id':-1}, {'state_id':2}, {'state_id':.5}])
def test_invalid_selections(molecular_solvers,kwargs):
    with pytest.raises(ValueError):
        molecular_solvers[0].local_spin(**kwargs)


def test_requires_converged_result():
    with pytest.raises(ValueError,match='converged'):
        local_spin(SimpleNamespace(converged=False))


def test_direct_frozen_frame_casci(molecular_solvers):
    from copy import copy
    mc = molecular_solvers[1]
    frozen = copy(mc.casci)
    frozen.mf = SimpleNamespace(_h1_mo=np.eye(len(mc.mo_coeff)),mo_coeff=mc.mo_coeff)
    frozen.mo_coeff = np.eye(len(mc.mo_coeff))
    assert frozen.local_spin([0,1]) == pytest.approx(mc.local_spin([0,1]),abs=2e-12)


def test_complex_orbitals_are_explicitly_unsupported(molecular_solvers):
    from copy import copy
    mc = copy(molecular_solvers[0])
    mc.mo_coeff = mc.mo_coeff.astype(complex)
    with pytest.raises(NotImplementedError,match='real restricted'):
        mc.local_spin([0])


def test_complex_ci_with_real_orbitals_matches_real_components():
    pytest.importorskip('pyscf')
    from pyscf.fci import cistring,spin_op
    from pyqed.qchem.ci.fci import FCIStringBasis
    from pyqed.qchem.mcscf.casci import make_state_average_rdms
    rng = np.random.default_rng(612)
    n = 3
    bits = cistring.make_strings(range(n),1)
    occupations = ((bits[:,None] >> np.arange(n)) & 1).astype(np.int8)
    basis = FCIStringBasis(occupations,occupations)
    ci = rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)); ci /= np.linalg.norm(ci)
    c = np.linalg.qr(rng.normal(size=(n,n)))[0]
    p = c[:2].T@c[:2]
    d1,d2 = make_state_average_rdms([ci.reshape(-1)],[1.],basis)
    expected = sum(spin_op.local_spin(part,n,(1,1),c,np.eye(n),[0,1])[0]
                   for part in (ci.real.copy(),ci.imag.copy()))
    assert local_spin_from_rdms(d1,d2,p) == pytest.approx(expected,abs=2e-12)
