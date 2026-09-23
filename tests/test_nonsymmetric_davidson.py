import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator
from pyqed.linalg import davidson_nonsymmetric


@pytest.mark.parametrize('complex_values',[False,True])
@pytest.mark.parametrize('selection',['smallest_real','smallest_magnitude','target'])
def test_dense_general_spectrum(complex_values,selection):
    rng=np.random.default_rng(438)
    a=np.diag(np.linspace(-3,5,36))+.02*rng.normal(size=(36,36))
    if complex_values:a=a+1j*(np.diag(np.linspace(-.4,.6,36))+.01*rng.normal(size=(36,36)))
    target=.8+.1j
    expected=np.linalg.eigvals(a)
    key=expected.real if selection=='smallest_real' else abs(expected if selection=='smallest_magnitude' else expected-target)
    expected=expected[np.argsort(key)[:3]]
    w,v,info=davidson_nonsymmetric(a,3,selection=selection,target=target,tolerance=1e-10,space=22,iterations=300)
    assert info['converged']
    np.testing.assert_allclose(w,expected,atol=1e-9,rtol=0)
    assert max(np.linalg.norm(a@v-v*w,axis=0))<1e-10


def test_complex_pairs_and_positive_filter():
    a=np.array([[1,-2,0,0],[2,1,0,0],[0,0,-1,.1],[0,0,0,3.]])
    w,_,_=davidson_nonsymmetric(a,2,selection='smallest_magnitude',space=4)
    assert np.min(abs(w+1))<1e-10
    assert np.min(abs(abs(w.imag)-2))<1e-10
    w,_,_=davidson_nonsymmetric(a,1,selection='positive_real',space=4)
    np.testing.assert_allclose(w,[3.],atol=1e-10)
    with pytest.raises(RuntimeError):davidson_nonsymmetric(a,2,selection='positive_real',space=4)


def test_matrix_free_positive_bse_and_batched_callbacks():
    rng=np.random.default_rng(914)
    a=np.diag(np.linspace(.5,4,28)); q=rng.normal(size=a.shape)*.004
    a+=(q+q.T); b=rng.normal(size=a.shape)*.005; b=(b+b.T)/2
    full=np.block([[a,b],[-b,-a]])
    expected=np.sort(np.linalg.eigvals(full).real)[28:31]
    op=LinearOperator(full.shape,matvec=full.dot,matmat=full.dot,dtype=float)
    w,v,info=davidson_nonsymmetric(op,3,diag=full.diagonal(),selection='positive_real',space=22,tolerance=1e-10)
    np.testing.assert_allclose(w,expected,atol=1e-9)
    assert info['matmat_callback_calls']>0 and info['matvec_callback_calls']==0
    assert np.max(np.linalg.norm(full@v-v*w,axis=0))<1e-10
    w2,_,info2=davidson_nonsymmetric(full.dot,3,diag=full.diagonal(),selection='positive_real',space=22,tolerance=1e-10)
    np.testing.assert_allclose(w2,w,atol=1e-9)
    assert info2['matvec_callback_calls']>0


def test_failures_and_initial_rank():
    rng=np.random.default_rng(128)
    a=np.diag(np.arange(30.))+.03*rng.normal(size=(30,30))
    with pytest.raises(RuntimeError,match='did not converge'):
        davidson_nonsymmetric(a,3,iterations=1)
    _,_,info=davidson_nonsymmetric(a,3,iterations=1,return_partial=True)
    assert not info['converged']
    w,v,_=davidson_nonsymmetric(a,2,guess=np.ones((30,3)),space=20)
    assert max(np.linalg.norm(a@v-v*w,axis=0))<1e-9
    with pytest.raises(MemoryError):davidson_nonsymmetric(a,memory_limit=1)
    with pytest.raises(ValueError):davidson_nonsymmetric(a,space=1)
    with pytest.raises(ValueError):davidson_nonsymmetric(lambda x:x)
    with pytest.raises(ValueError):davidson_nonsymmetric(lambda x:np.zeros(2),diag=np.arange(6.))
    with pytest.raises(ValueError):davidson_nonsymmetric(a,guess=np.full((30,2),np.nan))


@pytest.mark.parametrize('tolerance',[1e-8,1e-12])
def test_restart_and_tight_residual(tolerance):
    rng=np.random.default_rng(582)
    a=np.diag(np.linspace(1,12,60))+.03*rng.normal(size=(60,60))
    w,v,info=davidson_nonsymmetric(a,2,space=10,tolerance=tolerance,iterations=300)
    assert info['restarts']>0
    np.testing.assert_allclose(w,np.sort_complex(np.linalg.eigvals(a))[:2],atol=1e-8)
    assert np.max(np.linalg.norm(a@v-v*w,axis=0))<=tolerance


def test_zero_root_filter_and_nonnormal_near_degeneracy():
    a=np.diag([-2.,0.,1.,2.])
    w,_,_=davidson_nonsymmetric(a,1,selection='positive_real')
    np.testing.assert_allclose(w,[1.],atol=1e-12)
    rng=np.random.default_rng(726)
    transform=np.eye(12)+.1*rng.normal(size=(12,12))
    exact=np.array([1.,1.+1e-8,*range(2,12)])
    a=transform@np.diag(exact)@np.linalg.inv(transform)
    w,v,_=davidson_nonsymmetric(a,2,space=12,tolerance=1e-11)
    np.testing.assert_allclose(w,exact[:2],atol=1e-9)
    assert np.max(np.linalg.norm(a@v-v*w,axis=0))<1e-11


def test_fresh_residual_check_rejects_changed_action():
    calls=0
    diagonal=np.arange(1.,9.)
    def action(x):
        nonlocal calls
        calls+=1
        return (diagonal+(calls>1))[:,None]*x
    with pytest.raises(RuntimeError,match='did not converge'):
        davidson_nonsymmetric(lambda x:x,1,diag=diagonal,matmat=action)


@pytest.mark.parametrize('seed', [0, 5])
@pytest.mark.parametrize('complex_values', [False, True])
def test_harmonic_restart_with_rotated_degenerate_bse(seed, complex_values):
    # Doubled symmetry sectors mixed by an orthogonal basis change expose the
    # generalized-QZ failure seen in molecular BSE, without molecular fixtures.
    rng = np.random.default_rng(seed)
    t = rng.normal(size=(15, 15))
    a = np.diag(np.linspace(.3, 3, 15))+.04*(t+t.T)
    t = rng.normal(size=(15, 15))
    b = .025*(t+t.T)
    a = np.kron(np.eye(2), a)
    b = np.kron(np.eye(2), b)
    q, _ = np.linalg.qr(rng.normal(size=a.shape))
    if complex_values:
        q = np.exp(1j*np.linspace(0, 1, len(q)))[:, None]*q
    a = q@a@q.conj().T
    b = q@b@q.conj().T
    h = np.block([[a, b], [-b, -a]])
    reference = np.sort(np.linalg.eigvals(h).real)[30:35]
    w, v, info = davidson_nonsymmetric(h, 5, selection='positive_real',
        space=24, tolerance=1e-9, iterations=300)
    assert info['restarts'] > 0
    assert info['converged']
    np.testing.assert_allclose(w.real, reference, atol=1e-8, rtol=0)
    assert max(np.linalg.norm(h@v-v*w, axis=0)) <= 1e-9
    assert np.linalg.svd(v, compute_uv=False)[-1] > 1e-4
