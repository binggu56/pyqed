"""Degenerate orbital/pole response: invariance, displacements and Dyson blocks."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.optimize import newton

from pyqed.gw import GW, BSE, TDA
from pyqed.gw.response import _Jet, _sqrt, _contract, _detached
from pyqed.gw.subspace_response import (
    screening, self_energy, quasiparticles, orbital_groups, inverse)
from pyqed.qchem import Molecule


def _evaluate(jet, q):
    return jet.v+np.einsum('x,x...->...',q,jet.d)+.5*np.einsum('x,y,xy...->...',q,q,jet.h)


def _constant_at(jet,q):
    return _Jet.constant(_evaluate(jet,q),jet.n)


def _toy():
    rng = np.random.default_rng(62)
    nmo,n = 4,2
    e = np.diag([-1.,-1.,.5,.5])
    ed = np.zeros((n,nmo,nmo))
    eh = np.zeros((n,n,nmo,nmo))
    for start in (0,2):
        a = rng.normal(size=(n,2,2))*.1
        ed[:,start:start+2,start:start+2] = (a+a.swapaxes(-1,-2))/2
        a = rng.normal(size=(n,n,2,2))*.03
        a = (a+a.swapaxes(0,1))/2
        eh[:,:,start:start+2,start:start+2] = (a+a.swapaxes(-1,-2))/2
    # Orthonormal symmetric matrix basis gives rotationally invariant ERIs
    # and exactly repeated charge-RPA poles at the reference geometry.
    factors = []
    for i in range(nmo):
        for j in range(i+1):
            a = np.zeros((nmo,nmo))
            a[i,j] = a[j,i] = .15 if i == j else .15/np.sqrt(2)
            factors.append(a)
    factors = np.asarray(factors)
    fd = rng.normal(size=(n,)+factors.shape)*.015
    fd = (fd+fd.swapaxes(-1,-2))/2
    fh = rng.normal(size=(n,n)+factors.shape)*.01
    fh = (fh+fh.swapaxes(0,1))/2
    fh = (fh+fh.swapaxes(-1,-2))/2
    return _Jet(e,ed,eh),_Jet(factors,fd,fh)


def test_matrix_square_root_and_static_screening_at_repeated_poles():
    e,f = _toy()
    w,m = screening(e,f,2,True)
    assert np.ptp(np.diag(w.v)) < 1e-12
    static = _contract('pqL,LM,rsM->pqrs',m,inverse(w),m)
    step = 2e-4
    def value(q):
        wi,mi = screening(_constant_at(e,q),_constant_at(f,q),2,True)
        return np.einsum('pqL,LM,rsM->pqrs',mi.v,np.linalg.inv(wi.v),mi.v)
    for x in range(2):
        q = np.eye(2)[x]*step
        np.testing.assert_allclose(static.d[x],(value(q)-value(-q))/(2*step),atol=2e-9)
        np.testing.assert_allclose(static.h[x,x],(value(q)-2*value([0,0])+value(-q))/step**2,atol=2e-8)
    mixed = sum(s*t*value(step*np.array([s,t])) for s in (-1,1) for t in (-1,1))/(4*step**2)
    np.testing.assert_allclose(static.h[0,1],mixed,atol=2e-8)


def test_degenerate_dyson_block_against_full_arrowhead():
    e,f = _toy()
    w,m = screening(e,f,2,True)
    groups = orbital_groups(np.diag(e.v),2,1e-7)
    roots = np.empty(4)
    for group in groups:
        def residual(omega):
            sigma,_,_ = self_energy(omega,e,w,m,2,group)
            return omega-e.v[group[0],group[0]]-sigma.v[0,0]
        roots[group] = newton(residual,e.v[group[0],group[0]],tol=1e-13)
    qp,residuals = quasiparticles(SimpleNamespace(nocc=2,e_qp=roots),e,w,m,groups,1e-7)
    assert max(residuals) < 1e-12
    def exact(q,group):
        ei,fi = _constant_at(e,q),_constant_at(f,q)
        wi,mi = screening(ei,fi,2,True)
        centers = block_diag(np.kron(ei.v[:2,:2],np.eye(4))-np.kron(np.eye(2),wi.v),
                             np.kron(ei.v[2:,2:],np.eye(4))+np.kron(np.eye(2),wi.v))
        coupling = np.sqrt(2)*mi.v[:,group,:].transpose(1,0,2).reshape(len(group),-1)
        arrow = np.block([[ei.v[np.ix_(group,group)],coupling],[coupling.T,centers]])
        values = np.linalg.eigvalsh(arrow)
        nearest = np.argsort(abs(values-roots[group[0]]))[:len(group)]
        return np.sort(values[nearest])
    for group in groups:
        for direction in ([1.,0.],[0.,1.],[.6,-.8]):
            errors = []
            for step in (2e-3,1e-3):
                q = step*np.asarray(direction)
                prediction = np.linalg.eigvalsh(_evaluate(qp,q)[np.ix_(group,group)])
                errors.append(np.max(abs(prediction-exact(q,group))))
            assert errors[1] < max(errors[0]/5,2e-13)
            assert errors[1] < 2e-10


def test_degenerate_pole_rotations_do_not_change_self_energy_response():
    e,f = _toy()
    w,m = screening(e,f,2,True)
    rotation,_ = np.linalg.qr(np.random.default_rng(31).normal(size=(4,4)))
    wr = _contract('pi,pq,qj->ij',rotation,w,rotation)
    mr = _contract('pqt,tL->pqL',m,rotation)
    for group in (np.array([0,1]),np.array([2,3])):
        expected = self_energy(-.8,e,w,m,2,group)
        actual = self_energy(-.8,e,wr,mr,2,group)
        for a,b in zip(actual[:2],expected[:2]):
            for value,reference in zip((a.v,a.d,a.h),(b.v,b.d,b.h)):
                if value is None or reference is None:
                    assert value is reference
                else:
                    np.testing.assert_allclose(value,reference,atol=1e-11)
        np.testing.assert_allclose(actual[2],expected[2],atol=1e-11)


def _h3(cls=TDA,screening_energies='hf',rotation=None,shift=None):
    coords = np.array([[1.,0.,0.],[-.5,np.sqrt(3)/2,0.],[-.5,-np.sqrt(3)/2,0.]])
    if shift is not None:
        coords = coords+shift
    mol = Molecule(atom=[('H',tuple(r)) for r in coords],charge=1,
                   basis='sto-3g',unit='bohr').build(eri='dense')
    mf = mol.RHF().run(tol=1e-13)
    if rotation is not None:
        mf.mo_coeff = mf.mo_coeff@rotation
    gw = GW(mf,freq_int='exact').run()
    return cls(gw,screening_energies=screening_energies).run(nroots=2,low_rank=False)


@pytest.mark.parametrize('cls',[TDA,BSE])
@pytest.mark.parametrize('screening_energies',['hf','qp'])
def test_D3h_degenerate_molecule_and_orbital_rotation(cls,screening_energies):
    ref = _h3(cls,screening_energies)
    theta = .731
    rotation = np.eye(3)
    rotation[1:,1:] = [[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
    # Rotate the SAME reference orbitals, so arbitrary AO eigensolver signs
    # cannot masquerade as a failure of covariance.
    mf = _detached(ref._scf)
    mf.mo_coeff = mf.mo_coeff@rotation
    rotated = cls(GW(mf,freq_int='exact').run(),screening_energies=screening_energies).run(nroots=2,low_rank=False)
    modes = np.zeros((2,3,3))
    modes[0,0,0],modes[1,0,1] = 1,1
    f,g = ref.vibronic_couplings(modes=modes)
    fr,gr = rotated.vibronic_couplings(modes=modes)
    def vectors(obj):
        if cls is TDA:
            return obj.x
        from pyqed.gw.bse import bse_AB_matrices
        a,b = bse_AB_matrices(obj)
        minus = _sqrt(_Jet.constant(a-b,1)).v
        return np.linalg.solve(minus,obj.x+obj.y)*np.sqrt(obj.e)[None,:]
    transform = vectors(ref).T@rotation[1:,1:]@vectors(rotated)
    np.testing.assert_allclose(f,np.einsum('ip,pqx,jq->ijx',transform,fr,transform),atol=2e-9)
    np.testing.assert_allclose(g,np.einsum('ip,pqxy,jq->ijxy',transform,gr,transform),atol=2e-8)
    assert ref.derivative_info['orbital_subspaces'] == [[0],[1,2]]
    assert not any(ref.derivative_info['surface_hessian_defined'])
    assert np.all(np.isnan(ref.energy_hessian))
    assert np.all(np.isnan(ref.energy_gradient))
    assert np.all(np.isfinite(ref.subspace_hessian))
    np.testing.assert_allclose(ref.subspace_hessian,np.trace(g,axis1=0,axis2=1),atol=1e-10)
    ref.vibronic_couplings(state_ids=[0],modes=modes)
    assert np.all(np.isnan(ref.subspace_hessian))  # Cut electronic doublet.
    # Jahn-Teller splitting is given by eigenvalues of the complete F block,
    # not by independently differentiating arbitrarily labeled roots.
    step = 2e-4
    for direction in ([1.,0.],[0.,1.],[.6,-.8]):
        q = step*np.asarray(direction)
        shifted = _h3(cls,screening_energies,shift=np.einsum('x,xai->ai',q,modes))
        local = np.diag(ref.e_tot)+np.einsum('ijx,x->ij',f,q)+.5*np.einsum('ijxy,x,y->ij',g,q,q)
        np.testing.assert_allclose(np.linalg.eigvalsh(local),shifted.e_tot,atol=2e-8)


def test_occupied_and_virtual_multiplets():
    atoms = [('C',(0.,0.,0.))]+[('H',tuple(1.18*np.asarray(v)))
        for v in ((1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1))]
    mol = Molecule(atom=atoms,unit='bohr',basis='sto-3g').build(eri='dense')
    mf = mol.RHF().run(tol=1e-13)
    ref = TDA(GW(mf,freq_int='exact').run(),screening_energies='hf').run(nroots=20,low_rank=False)
    modes = np.zeros((1,5,3));modes[0,0,0] = 1
    f,g = ref.vibronic_couplings(modes=modes)
    groups = ref.derivative_info['orbital_subspaces']
    assert any(len(group)>1 and group[-1]<ref.nocc for group in groups)
    assert any(len(group)>1 and group[0]>=ref.nocc for group in groups)
    rng = np.random.default_rng(83)
    rotation = np.eye(len(mf.mo_energy))
    for group in groups:
        r,_ = np.linalg.qr(rng.normal(size=(len(group),len(group))))
        rotation[np.ix_(group,group)] = r
    rotated_mf = _detached(mf)
    rotated_mf.mo_coeff = mf.mo_coeff@rotation
    rotated = TDA(GW(rotated_mf,freq_int='exact').run(),screening_energies='hf').run(nroots=20,low_rank=False)
    fr,gr = rotated.vibronic_couplings(modes=modes)
    excitation_rotation = np.kron(rotation[:ref.nocc,:ref.nocc],rotation[ref.nocc:,ref.nocc:])
    transform = ref.x.T@excitation_rotation@rotated.x
    np.testing.assert_allclose(f,np.einsum('ip,pqx,jq->ijx',transform,fr,transform),atol=2e-8)
    np.testing.assert_allclose(g,np.einsum('ip,pqxy,jq->ijxy',transform,gr,transform),atol=2e-7)


@pytest.mark.parametrize('order', [1, 2])
def test_contraction_skips_fixed_operand_responses(monkeypatch, order):
    from pyqed.gw.response import _FirstJet, _Jet, _contract
    rng = np.random.default_rng(41)
    a, b, c, da, dc, ha, hc = rng.normal(size=(7, 3, 3))
    left = _FirstJet(a, da[None]) if order == 1 else _Jet(a, da[None], ha[None,None])
    right = _FirstJet(c, dc[None]) if order == 1 else _Jet(c, dc[None], hc[None,None])
    original = np.einsum
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(np, 'einsum', counted)
    result = _contract('ij,jk,kl->il', left, b, right)
    np.testing.assert_allclose(result.v, a@b@c)
    np.testing.assert_allclose(result.d[0], da@b@c+a@b@dc)
    if order == 2:
        np.testing.assert_allclose(result.h[0,0], ha@b@c+2*da@b@dc+a@b@hc)
    assert len(calls) == (3 if order == 1 else 6)


@pytest.mark.parametrize('function', [_sqrt, inverse])
def test_diagonal_reference_preserves_offdiagonal_response(monkeypatch, function):
    rng = np.random.default_rng(87)
    value = np.diag([1.3, 1.3, 3.2])
    d = rng.normal(size=(2,3,3))
    d = d + d.transpose(0,2,1)
    h = rng.normal(size=(2,2,3,3))
    h = h + h.transpose(1,0,2,3)
    h = h + h.transpose(0,1,3,2)
    rotation, _ = np.linalg.qr(rng.normal(size=(3,3)))
    dense = function(_Jet(rotation@value@rotation.T,
                         rotation@d@rotation.T, rotation@h@rotation.T))
    def forbidden(*args, **kwargs):
        raise AssertionError('Diagonal reference must not use dense factorization')
    monkeypatch.setattr(np.linalg, 'eigh', forbidden)
    monkeypatch.setattr(np.linalg, 'inv', forbidden)
    result = function(_Jet(value,d,h))
    for attr in ('v','d','h'):
        np.testing.assert_allclose(rotation@getattr(result,attr)@rotation.T,
                                   getattr(dense,attr), atol=1e-12)


@pytest.mark.parametrize('mixed', [False, True])
def test_staged_factor_contractions_match_product_rule(mixed):
    from contextlib import nullcontext
    from pyqed.gw.response import _veff, mixed_response
    rng = np.random.default_rng(18)
    def tensor(shape):
        h = rng.normal(size=(2,2)+shape)
        h = (h+h.swapaxes(0,1))/2
        if mixed:
            h = h[:1,1:2]
        return _Jet(rng.normal(size=shape), rng.normal(size=(2,)+shape), h)
    with mixed_response() if mixed else nullcontext():
        factors, density, coefficients = tensor((5,3,3)), tensor((3,3)), tensor((3,3))
        expected = (_contract('Ppq,Prs,rs->pq',factors,factors,density)
                    -.5*_contract('Ppr,Pqs,rs->pq',factors,factors,density))
        actual = _veff(factors,density,True)
        half = _contract('Ppq,pi->Piq',factors,coefficients)
        rotated = _contract('Piq,qj->Pij',half,coefficients)
        reference = _contract('Ppq,pi,qj->Pij',factors,coefficients,coefficients)
        for attr in ('v','d','h'):
            np.testing.assert_allclose(getattr(actual,attr),getattr(expected,attr),atol=2e-12)
            np.testing.assert_allclose(getattr(rotated,attr),getattr(reference,attr),atol=2e-12)


def test_contraction_diagonal_values_keep_dense_derivatives(monkeypatch):
    rng = np.random.default_rng(92)
    a = rng.normal(size=(2,3,3))
    b = np.diag([1.,2.,3.])
    da, ha = rng.normal(size=(2,2,3,3))
    db, hb = rng.normal(size=(2,3,3))
    original = np.einsum
    expressions = []
    def recorded(expression, *args, **kwargs):
        expressions.append(expression)
        return original(expression,*args,**kwargs)
    monkeypatch.setattr(np, 'einsum', recorded)
    result = _contract('pqt,tL->pqL', _Jet(a,da[None],ha[None,None]),
                       _Jet(b,db[None],hb[None,None]))
    np.testing.assert_allclose(result.v,a@b)
    np.testing.assert_allclose(result.d[0],da@b+a@db)
    np.testing.assert_allclose(result.h[0,0],ha@b+2*da@db+a@hb)
    assert 'pqL,L->pqL' in expressions
    assert 'pqt,tL->pqL' in expressions
    # A diagonal matrix with two free output axes cannot be collapsed.
    outer = _contract('ij,ab->iajb', b, _Jet(b,db[None],hb[None,None]))
    np.testing.assert_allclose(outer.v, original('ij,ab->iajb',b,b))
