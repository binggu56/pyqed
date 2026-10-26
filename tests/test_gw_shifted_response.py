import numpy as np
import pytest
from pyqed.gw.experimental.shifted_response import ShiftedGW
from test_gw_contour import model
from benchmarks.benchmark_gw_shifted_response import dense_reference


def data():
    gw=model();energy=gw.e_mf[::2];f=gw._pair_factors
    rng=np.random.default_rng(14)
    de=rng.normal(size=energy.shape)*.1
    df=rng.normal(size=f.shape)*.03;df=(df+df.transpose(0,2,1))/2
    return energy,f,de,df


@pytest.mark.parametrize('orbital',[0,2])
@pytest.mark.parametrize('frequency',[-.71,.13])
def test_value_and_first_response(orbital,frequency):
    e,f,de,df=data();r=ShiftedGW(e,f,2,tolerance=1e-11)
    actual=r.evaluate(orbital,frequency,energy_tangent=de,factor_tangent=df,frequency_tangent=.17)
    expected=dense_reference(e,f,2,orbital,frequency,de,df,.17)
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
    h=1e-5
    plus=dense_reference(e+h*de,f+h*df,2,orbital,frequency+h*.17,de,df)[0]
    minus=dense_reference(e-h*de,f-h*df,2,orbital,frequency-h*.17,de,df)[0]
    np.testing.assert_allclose(actual[1],(plus-minus)/(2*h),atol=1e-9,rtol=0)
    assert r.info['converged'] and r.info['max_relative_residual']<=r.solve_tol


def test_action_response_and_no_dense_solver(monkeypatch):
    e,f,de,df=data();r=ShiftedGW(e,f,2)
    v=np.random.default_rng(2).normal(size=(6,2));h=1e-5
    fd=(ShiftedGW(e+h*de,f+h*df,2).action(v)-ShiftedGW(e-h*de,f-h*df,2).action(v))/(2*h)
    np.testing.assert_allclose(r.action_response(v,de,df),fd,atol=1e-9,rtol=0)
    import scipy.linalg as la
    for module in [np.linalg,la]:
        for name in ['eigh','eigvalsh','inv','solve']:
            monkeypatch.setattr(module,name,lambda *a,**k:pytest.fail('dense solver called'))
    r.evaluate(2,.13,energy_tangent=de,factor_tangent=df)


def test_failure_and_controls():
    e,f,de,df=data()
    with pytest.raises(np.linalg.LinAlgError,match='solve failed'):
        ShiftedGW(e,f,2,maxiter=1,restart=1).evaluate(2,.13)
    with pytest.raises(np.linalg.LinAlgError,match='quadrature failed'):
        ShiftedGW(e,f,2,initial_order=1,max_order=2,tolerance=1e-15).evaluate(2,.13)
    with pytest.raises(ValueError):ShiftedGW(e,f,2,solve_tol=0)
    with pytest.raises(ValueError):ShiftedGW(e,f,2,restart=True)
    with pytest.raises(ValueError):ShiftedGW(e,f,2).evaluate(0,.1,energy_tangent=np.zeros(2))


def test_zero_coupling_and_repeated_poles():
    e=np.array([-.5,.5,.5]);f=np.zeros((2,3,3))
    r=ShiftedGW(e,f,1)
    assert r.evaluate(0,-.3)==(0.,0.)
    # Equal gaps with orthogonal equal-norm transition factors give repeated poles.
    f[0,0,1]=f[0,1,0]=.1;f[1,0,2]=f[1,2,0]=.1
    rng=np.random.default_rng(3);df=rng.normal(size=f.shape)*.02;de=np.array([.01,-.03,.04])
    r=ShiftedGW(e,f,1)
    np.testing.assert_allclose(r.evaluate(0,-.3,energy_tangent=de,factor_tangent=df),
        dense_reference(e,f,1,0,-.3,de,df),atol=1e-10,rtol=0)


def test_existing_gw_spectral_sum_and_near_pole():
    import importlib
    g=importlib.import_module('pyqed.gw.gw')
    obj=model();obj.eta=0.
    e=obj.e_mf[::2];f=obj._pair_factors
    poles,vectors=g.rpa(obj)
    g._charge_spatial_couplings(obj,poles,vectors)
    for w in [.13, e[2], e[0]+poles[0], e[2]+poles[0]+.01]:
        r=ShiftedGW(e,f,2,tolerance=1e-10)
        value,slope=r.evaluate(2,float(w),frequency_tangent=1.)
        expected=g.sigma(obj,4,4,float(w),poles,vectors)[0].real
        np.testing.assert_allclose(value,expected,atol=1e-10,rtol=0)
        ref=dense_reference(e,f,2,2,w,np.zeros_like(e),np.zeros_like(f),1.)
        np.testing.assert_allclose(slope,ref[1],atol=2e-8,rtol=0)


def test_shared_frequencies_and_scalar_adjoint():
    e,f,de,df=data()
    frequencies=np.array([-.71,.13,e[2]])
    r=ShiftedGW(e,f,2,tolerance=1e-10)
    actual=r.evaluate_many(2,frequencies,energy_tangent=de,factor_tangent=df)
    shared_solves=r.info['solves']
    independent_solves=0
    for j,w in enumerate(frequencies):
        old=r.evaluate(2,w,energy_tangent=de,factor_tangent=df,reuse=False)
        independent_solves+=r.info['solves']
        slope=r.evaluate(2,w,frequency_tangent=1.,reuse=False)
        independent_solves+=r.info['solves']
        np.testing.assert_allclose(actual[j],[old[0],old[1],slope[1]],atol=1e-10,rtol=0)
        expected=dense_reference(e,f,2,2,w,de,df)
        np.testing.assert_allclose(actual[j,:2],expected,atol=1e-10,rtol=0)
    assert shared_solves < independent_solves
    np.testing.assert_allclose(r.evaluate_many(2,frequencies[::-1],energy_tangent=de,factor_tangent=df),actual[::-1],atol=1e-10,rtol=0)


def test_shared_no_stale_cache_and_zero_response():
    e,f,de,df=data();r=ShiftedGW(e,f,2)
    frequencies=[-.71,.13,.13]
    first=r.evaluate_many(2,frequencies)
    np.testing.assert_array_equal(first[:,1],0.)
    np.testing.assert_array_equal(first[1],first[2])
    f*=1.02
    # Construct a new operator for changed reference data; no cross-call response cache.
    changed=ShiftedGW(e,f,2).evaluate_many(2,frequencies)
    assert np.max(abs(changed-first))>1e-8
    with pytest.raises(ValueError):r.evaluate_many(2,[])
    with pytest.raises(ValueError):r.evaluate_many(2,[1j])


def test_batched_dense_benchmark_reference():
    from benchmarks.benchmark_gw_shifted_reuse import dense_many
    e,f,de,df=data();orbitals=[1,2];frequencies=[[-.71,-.61],[.13,.2]]
    batched=dense_many(e,f,2,orbitals,frequencies,de,df)
    for i,p in enumerate(orbitals):
        for j,w in enumerate(frequencies[i]):
            expected=dense_reference(e,f,2,p,w,de,df)
            slope=dense_reference(e,f,2,p,w,np.zeros_like(de),np.zeros_like(df),1.)[1]
            np.testing.assert_allclose(batched[i,j],list(expected)+[slope],atol=1e-13,rtol=0)
