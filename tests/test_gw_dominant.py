"""Dominant Dyson poles versus independently diagonalized arrowhead matrices."""
import numpy as np
import pytest

solver=pytest.importorskip('pyqed.gw._dyson')


@pytest.mark.parametrize('seed',range(12))
def test_global_weight_selection_against_dense(seed):
    rng=np.random.default_rng(seed)
    poles=np.r_[rng.normal(size=35),.25,.25]
    residues=10**rng.uniform(-10,0,(3,len(poles)))
    residues[:,::9]=0
    energies=rng.normal(size=3)
    shifts=rng.normal(size=3)
    result=solver.dominant(poles,residues,energies,shifts,1,1e-9)
    roots,weights,errors,_,bounds,_,_=result
    for i,r in enumerate(residues):
        arrow=np.diag(np.r_[energies[i]+shifts[i],poles])
        arrow[0,1:]=arrow[1:,0]=np.sqrt(r)
        e,v=np.linalg.eigh(arrow)
        j=np.argmax(v[0]**2)
        np.testing.assert_allclose(roots[i],e[j],atol=1e-8,rtol=0)
        np.testing.assert_allclose(weights[i],v[0,j]**2,atol=1e-8,rtol=0)
    assert max(errors)<=1e-9
    assert np.all(bounds<=weights)


def test_static_shift_does_not_trap_selection_on_satellite():
    # The largest-weight root is on the other side of the pole from the MF seed.
    roots,z,*_=solver.dominant(np.array([0.]),np.array([[.01]]),
                               np.array([.1]),np.array([-1.]),1,1e-10)
    assert roots[0]<-.9 and z[0]>.98


def test_symmetry_noise_and_worker_invariance():
    if not solver.openmp_available():pytest.skip('OpenMP unavailable')
    p=np.array([-1.,-.1,0.,.2,1.])
    r=np.tile([.03,1e-35,.1,1e-35,.02],(4,1))
    args=p,r,np.zeros(4),np.full(4,-.3)
    one=solver.dominant(*args,1,1e-9)
    many=solver.dominant(*args,4,1e-9)
    for i in (0,1,2,4,5,6):np.testing.assert_array_equal(one[i],many[i])
    assert many[3]==4


def test_zero_coupling():
    result=solver.dominant(np.array([-.2,.2]),np.zeros((1,2)),np.array([.1]),np.array([.3]))
    assert result[0][0]==pytest.approx(.4)
    assert result[1][0]==1.


def test_production_pole_contour_matches_direct():
    from test_gw_contour import model
    from pyqed.gw.contour import ContourDeformation
    from pyqed.gw.quasiparticle import PoleContour,spectrum,refine_contour
    obj=model();obj.threads=1;obj.eta=1e-8;obj.v_mf=np.zeros((10,10))
    poles=spectrum(obj);exact=poles.solve(obj,range(5),tol=1e-10)
    direct=ContourDeformation(obj,obj.e_mf[::2],range(5),nw=96)
    fast=PoleContour(obj,obj.e_mf[::2],range(5),nw=96)
    np.testing.assert_allclose(fast.w0,direct.w0,atol=1e-12)
    np.testing.assert_allclose(fast.w_imag,direct.w_imag,atol=1e-12)
    from pyqed.gw.gw import _sigma_x_diagonal
    static=_sigma_x_diagonal(obj,range(5))-np.diag(obj.v_mf)[::2]
    for p in range(5):
        np.testing.assert_allclose(fast.residue(.3,p,1),direct.residue(.3,p,1),atol=1e-10)
        np.testing.assert_allclose(fast.evaluate(p,exact[p]),direct.evaluate(p,exact[p]),atol=1e-9)
        a=refine_contour(direct,p,exact[p],poles.brackets[p],poles.reference[p],static[p],1e-9)
        assert a[0]==pytest.approx(exact[p],abs=1e-8)
