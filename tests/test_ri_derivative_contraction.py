"""Direct shell-contracted RI response against the directional reference."""
import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.eri_response import CoulombIntegrals


@pytest.mark.parametrize('cartesian,solver,screen', [
    (False, 'cholesky', 0.), (True, 'cholesky', 0.),
    (False, 'eigh', 0.), (False, 'cholesky', .1)])
def test_shell_contracted_ri_matches_directional(cartesian, solver, screen, monkeypatch):
    mol = Molecule(atom='O .1 .2 0; H .2 1.4 1.1; H -.1 -1.4 1.2',
                   unit='bohr', basis='def2-svp').build(eri='ri', options={
        'auxbasis': 'def2-universal-jfit', 'ri_cache': False,
        'coord_type': 'cartesian' if cartesian else 'spherical',
        'ri_metric_solver': solver, 'ri_metric_tol': .01 if solver == 'eigh' else 1e-10,
        'ri_screen_tol': screen})
    response = CoulombIntegrals(mol)
    rng = np.random.default_rng(77)
    a, b = rng.normal(size=(2, mol.nao, mol.nao))
    a, b = (a+a.T)/mol.nao, (b+b.T)/mol.nao
    direction = rng.normal(size=(mol.natom, 3))
    derivative = response.derivative(direction)
    terms = [[(1.3, a, b)], [(.5, b, b), (-.2, b, a)]]
    def forbidden(*args, **kwargs):
        raise AssertionError('Contracted RI must not rebuild coordinate-wise derivative tensors')
    monkeypatch.setattr(CoulombIntegrals, '_ri_primitive_derivative', forbidden)
    def exchange_matrix(density):
        return sum(np.einsum('Ppr,rs,Pqs->pq', left, density, right, optimize=True)
                   for left, right in derivative.terms)
    for exchange in (0., .2):
        expected = [sum(scale*np.sum(left*(derivative.j(right)-exchange*exchange_matrix(right)))
                        for scale, left, right in row) for row in terms]
        gradient = response.contract_derivatives(terms, exchange_fraction=exchange)
        np.testing.assert_allclose(np.einsum('oax,ax->o', gradient, direction),
                                   expected, atol=2e-10, rtol=2e-9)
        np.testing.assert_allclose(gradient.sum(axis=1), 0., atol=2e-11)


def test_ri_derivative_owner_validation():
    from pyqed.qchem import basis
    signatures = (((0, 0, 0), (0., 0., 0.), (1.,), (1.,)),)
    packed = tuple(basis._pack_signatures_for_numba(signatures))
    with pytest.raises(ValueError, match='index an atom'):
        basis._integrals_cpp.contract_ri_derivatives(
            packed, packed, np.array([1]), np.array([0]),
            np.ones((1, 1, 1)), np.ones((1, 1, 1)), 1)


def test_derivative_profile_reset_and_numerical_identity():
    from pyqed.qchem import basis
    kernel=basis._integrals_cpp
    sigs=(((0,0,0),(0.,0.,0.),(1.,),(1.,)),
          ((0,0,0),(0.,0.,1.4),(1.,),(1.,)))
    packed=tuple(basis._pack_signatures_for_numba(sigs))
    owners=np.arange(2)
    args=(packed,packed,owners,owners,np.ones((1,2,3)),np.ones((1,2,2)),2,2)
    kernel.derivative_profile(False)
    reference=kernel.contract_ri_derivatives(*args)
    assert not any(kernel.derivative_profile(True).values())
    try:
        actual=kernel.contract_ri_derivatives(*args)
    finally:
        timings=kernel.derivative_profile(False)
    np.testing.assert_array_equal(actual,reference)
    assert timings['ri_second_total']>0
    assert timings['ri_second_recurrence']>0
    assert timings['ri_second_metric']>0
    assert not any(kernel.derivative_profile(False).values())


@pytest.mark.parametrize('angular',[0,1,2])
@pytest.mark.parametrize('aux_owner',[0,2])
def test_occupied_shell_columns_match_packed_reference(angular,aux_owner):
    from pyqed.qchem import basis
    rng = np.random.default_rng(94)
    def shell(l,center):
        return [(power,center,(.4,1.2),tuple(rng.normal(size=2))) for power in basis._shell(l)]
    p = shell(angular,(0.,.1,.2))+shell(0,(1.2,.3,-.1))
    a = shell(1,(-.4,.7,.2))
    po = np.r_[np.zeros(len(p)-1,dtype=int),1]
    ao = np.full(len(a),aux_owner)
    args = (tuple(basis._pack_signatures_for_numba(p)),
            tuple(basis._pack_signatures_for_numba(a)),po,ao,3)
    c,rho = rng.normal(size=(len(p),2)),rng.normal(size=len(a))
    for atom in range(3):
        packed,metric = basis._integrals_cpp.ri_derivative_columns(*args,3*atom,3)
        projected,mx,jx = basis._integrals_cpp.ri_derivative_columns(*args,3*atom,3,c,rho)
        dense = np.array([basis._pair_factors_to_full(block,len(p)) for block in packed])
        np.testing.assert_allclose(projected,dense@c,atol=2e-11,rtol=2e-11)
        np.testing.assert_allclose(jx,np.einsum('xPij,P->xij',dense,rho),atol=2e-11,rtol=2e-11)
        np.testing.assert_array_equal(mx,metric)
        t = rng.normal(size=(len(p),max(1,len(p)-1)))
        u = rng.normal(size=(len(a),2))
        sph,ms,js = basis._integrals_cpp.ri_derivative_columns(*args,3*atom,3,c,rho,t,u)
        np.testing.assert_allclose(sph,np.einsum('AP,xAmi,mp->xPpi',u,projected,t),atol=3e-10,rtol=3e-11)
        np.testing.assert_allclose(ms,u.T@metric@u,atol=3e-10,rtol=3e-11)
        np.testing.assert_allclose(js,t.T@jx@t,atol=3e-10,rtol=3e-11)


def test_cholesky_response_reuses_fitted_factors(monkeypatch):
    from pyqed.qchem import eri_response
    mol = Molecule(atom='H 0 0 0; H .2 .1 1.5', unit='bohr', basis='sto-3g').build(
        eri='ri', options={'auxbasis': 'def2-universal-jfit', 'ri_cache': False,
                           'ri_metric_solver': 'cholesky', 'ri_storage': 'packed'})
    tensors = eri_response._ri_tensors
    calls = []
    def metric_only(primary, auxiliary):
        assert len(primary) == 0
        calls.append(True)
        return tensors(primary, auxiliary)
    monkeypatch.setattr(eri_response, '_ri_tensors', metric_only)
    response = CoulombIntegrals(mol)
    response._prepare_ri()
    assert calls == [True]
    assert response._ri[4] is None
    np.testing.assert_array_equal(response._ri[6], np.asarray(mol.eri_factors))


@pytest.mark.parametrize('proportional', [False, True])
@pytest.mark.parametrize('sparse', [False, True])
def test_f_shell_triplet_and_metric_finite_difference(proportional, sparse):
    from pyqed.qchem import basis
    from pyqed.qchem.eri_response import _ri_tensors
    rng = np.random.default_rng(63)
    centers = np.array([[.2, -.1, .3], [1.2, .7, -.3], [-.4, .5, 1.3]])
    def shell(l, center):
        radial = rng.normal(size=2)
        return [(power, tuple(centers[center]), (.4, 1.2),
                 tuple(rng.normal()*radial if proportional else rng.normal(size=2)))
                for power in basis._shell(l)]
    primary = shell(3, 0)+shell(3, 1)
    auxiliary = shell(2, 1)+shell(3, 2)
    po = np.repeat([0, 1], 10)
    ao = np.r_[np.full(6, 1), np.full(10, 2)]
    p, q = np.tril_indices(len(primary))
    jw = rng.normal(size=(2, len(auxiliary), len(p)))
    mw = rng.normal(size=(2, len(auxiliary), len(auxiliary)))
    if sparse:
        jw[:,::2] = 0
        mw[:,::2,:] = 0
        mw[:,:,::2] = 0
        # Antisymmetric metric adjoints also have exactly zero contribution.
        mw[:,0,1], mw[:,1,0] = 1., -1.
    gradient = basis._integrals_cpp.contract_ri_derivatives(
        tuple(basis._pack_signatures_for_numba(primary)),
        tuple(basis._pack_signatures_for_numba(auxiliary)), po, ao, jw, mw, 3)
    direction = rng.normal(size=(3, 3))
    def displaced(signatures, owners, scale):
        return [(s, tuple(np.asarray(c)+scale*direction[a]), e, w)
                for (s, c, e, w), a in zip(signatures, owners)]
    def energy(scale):
        metric, three = _ri_tensors(displaced(primary, po, scale), displaced(auxiliary, ao, scale))
        return np.einsum('oAp,Ap->o', jw, three[:, p, q])+np.einsum('oAB,AB->o', mw, metric)
    step = 1e-5
    expected = (energy(step)-energy(-step))/(2*step)
    np.testing.assert_allclose(np.einsum('oax,ax->o', gradient, direction), expected,
                               atol=2e-7, rtol=2e-8)
    np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=2e-10)


@pytest.mark.parametrize('angular', [0,1,2])
@pytest.mark.parametrize('proportional',[True,False])
def test_shell_curvature_matches_gradient_difference(angular,proportional):
    from pyqed.qchem import basis
    rng = np.random.default_rng(418)
    def shell(l,center):
        return [(power,center,(.5,1.1),(.7,.3) if proportional else tuple(rng.normal(size=2)))
                for power in basis._shell(l)]
    p = shell(angular,(0.,.1,.2))+shell(0,(1.2,.3,-.1))
    a = shell(1,(-.4,.7,.2))+shell(angular,(.6,-.2,1.1))
    po = np.r_[np.zeros(len(p)-1,dtype=int),1]
    ao = np.r_[np.full(3,1),np.full(len(a)-3,2)]
    jw = rng.normal(size=(2,len(a),len(p)*(len(p)+1)//2))
    mw = rng.normal(size=(2,len(a),len(a)))
    direction = rng.normal(size=(3,3))
    def evaluate(scale,order):
        def pack(sigs,owners):
            shifted = [(l,tuple(np.array(c)+scale*direction[o]),e,w)
                       for (l,c,e,w),o in zip(sigs,owners)]
            return tuple(basis._pack_signatures_for_numba(shifted))
        return basis._integrals_cpp.contract_ri_derivatives(pack(p,po),pack(a,ao),po,ao,jw,mw,3,order)
    h = evaluate(0.,2)
    step = 1e-5
    expected = ((evaluate(step,1)-evaluate(-step,1))/(2*step)).reshape(2,-1)
    np.testing.assert_allclose(h@direction.ravel(),expected,atol=2e-7,rtol=2e-8)
    np.testing.assert_allclose(h,h.transpose(0,2,1),atol=2e-11)
    np.testing.assert_allclose(h@np.tile(np.eye(3),(3,1)),0.,atol=2e-10)
