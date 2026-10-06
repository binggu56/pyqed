import numpy as np
import pytest
from pyqed.mps.periodic_tools import circular_sections, metric_whitener, norm_factors, matrix_roots


def test_complex_separable_norm_chart_and_restored_support():
    rng=np.random.default_rng(31)
    u,_=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
    v,_=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
    l=(u*np.array([1e-4,2.]))@u.conj().T
    r=(v*np.array([.002,7.]))@v.conj().T
    metric=np.kron(l,np.kron(np.eye(3),r))
    a,b,info=norm_factors(metric,(2,3,2))
    _,ai=matrix_roots(a); _,bi=matrix_roots(b)
    chart=np.kron(ai,np.kron(np.eye(3),bi))
    transformed=chart.conj().T@metric@chart
    np.testing.assert_allclose(transformed,np.eye(12)*np.trace(transformed).real/12,atol=1e-7)
    assert info['relative_factorization_tail']<1e-12
    assert not info['partial_trace_fallback']
    # A bounded invertible chart restores tiny positive directions without
    # changing the metric or discarding tensor coefficients.
    tiny=np.kron(np.diag([1e-14,1.]),np.eye(2))
    a,b,_=norm_factors(tiny,(2,1,2))
    _,ai=matrix_roots(a); _,bi=matrix_roots(b)
    chart=np.kron(ai,bi)
    assert metric_whitener(tiny)[1]['rank']==2
    assert metric_whitener(chart.conj().T@tiny@chart)[1]['rank']==4


def test_circular_sections_cover_unequal_ring_once():
    assert circular_sections(2)==[[0],[1]]
    assert circular_sections(7)==[[0,1,2],[3,4],[5,6]]


@pytest.mark.parametrize('shape',[(8,13),(13,8)])
def test_rectangular_transfer_compression_and_factored_core(shape):
    from pyqed.mps.periodic_tools import compress_transfer,compress_factor_product
    rng=np.random.default_rng(95)
    left=rng.normal(size=(shape[0],3))+1j*rng.normal(size=(shape[0],3))
    right=rng.normal(size=(3,shape[1]))+1j*rng.normal(size=(3,shape[1]))
    matrix=left@right
    for l,r,info in [compress_transfer(lambda x:matrix@x,lambda x:matrix.conj().T@x,shape,tolerance=1e-12),
                     compress_factor_product(left,right,tolerance=1e-12)]:
        np.testing.assert_allclose(l@r,matrix,atol=1e-10)
        assert info['tolerance_met']
        assert info['rank']<=4
