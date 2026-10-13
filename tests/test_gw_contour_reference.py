"""The benchmark accelerator must solve PySCF's equation, not trust its cache."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import importlib
import numpy as np
import pytest
from pyscf.gw import gw_cd

spec = importlib.util.spec_from_file_location('gw_contour_reference',
    Path(__file__).parents[1]/'benchmarks/gw_contour_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
g = importlib.import_module('pyqed.gw.gw')


@pytest.mark.parametrize('corrupt', [False, True])
def test_reference_residues_against_unmodified_pyscf(corrupt):
    rng = np.random.default_rng(38)
    factors = rng.normal(scale=.08, size=(7,5,5))
    factors = (factors+factors.transpose(0,2,1))/2
    energy = np.array([-1.,-.5,.1,.4,.9])
    obj = SimpleNamespace(nocc=4, e_mf=np.repeat(energy,2),
        _pair_factors=factors, max_memory=10., mo_coeff=np.eye(5))
    poles, vectors = g._charge_rpa(obj)
    couplings = g._charge_spatial_couplings(obj,poles,vectors)
    projection = couplings.projection.copy()
    if corrupt:
        projection *= .93
    solver = reference.ContourReference(factors,energy,2,poles,projection)
    pairs = np.ascontiguousarray(factors[:,:2,2:])
    for p in [0,4]:
        for w in [-1.3,-.7,.3,1.1]:
            expected = gw_cd.get_sigmaR_diag(energy,w,p,-.2,factors,pairs,0.)
            actual = solver.residues(energy,w,p,-.2,factors,pairs,0.)
            assert actual == pytest.approx(expected.real,abs=1e-10)
    if corrupt:
        assert solver.info['refinements'] > 0
    assert solver.info['max_relative_residual'] <= solver.tol


def test_reference_does_not_accept_unchecked_frequency_settings():
    solver = reference.ContourReference(np.ones((1,2,2)),np.array([-1.,1.]),1,
                                        np.array([2.]),np.ones((1,1)))
    with pytest.raises(ValueError,match='eta=0'):
        solver.residues(solver.energy,1.,1,0.,None,None,.001)


def test_static_subtraction_resolves_near_orbital_quadrature():
    import io
    from pyscf.gw.utils.ac_grid import _get_scaled_legendre_roots
    rng = np.random.default_rng(73)
    factors = rng.normal(scale=.1,size=(7,5,5))
    factors = (factors+factors.transpose(0,2,1))/2
    energy = np.array([-1.,-.5,.1,.4,.9])
    obj = SimpleNamespace(nocc=4,e_mf=np.repeat(energy,2),_pair_factors=factors,
                          max_memory=10.,mo_coeff=np.eye(5))
    poles,vectors = g._charge_rpa(obj)
    coupling = g._charge_spatial_couplings(obj,poles,vectors)
    solver = reference.ContourReference(factors,energy,2,poles,coupling.projection)
    p = 4
    pairs = np.ascontiguousarray(factors[:,:2,2:])
    log = SimpleNamespace(verbose=0,stdout=io.StringIO())
    freqs,weights = _get_scaled_legendre_roots(64)
    W = gw_cd.get_WmnI_diag(log,[p],factors,pairs,freqs,energy)[:,0,:]
    W0 = gw_cd.get_WmnI_diag(log,[p],factors,pairs,np.array([0.]),energy)[0,0,:]
    centers = np.concatenate((energy[:2,None]-poles,energy[2:,None]+poles))
    residue = 2*coupling[:,p,:]**2
    for w in [energy[p],energy[p]+1e-6,1.23]:
        expected = np.sum(residue/(w-centers))
        actual = solver.imaginary_integral(w,W,freqs,weights,W0)
        actual += solver.residues(energy,w,p,-.2,factors,pairs,0.)
        assert actual == pytest.approx(expected,abs=1e-10)
    w = energy[p]+1e-6
    raw = gw_cd.get_sigma_diag(-.2,w,p,energy,factors,pairs,W,freqs,weights,0.)
    assert abs(raw.real-np.sum(residue/(w-centers))) > 1e-5


def test_dense_backward_gate_does_not_hide_large_relative_error():
    rng = np.random.default_rng(73)
    factors = rng.normal(scale=.1,size=(7,5,5))
    factors = (factors+factors.transpose(0,2,1))/2
    energy = np.array([-1.,-.5,.1,.4,.9])
    obj = SimpleNamespace(nocc=4,e_mf=np.repeat(energy,2),_pair_factors=factors,
                          max_memory=10.,mo_coeff=np.eye(5))
    poles,vectors = g._charge_rpa(obj)
    coupling = g._charge_spatial_couplings(obj,poles,vectors)
    solver = reference.ContourReference(factors,energy,2,poles,coupling.projection)
    # At a bare transition pole, a tiny scaled error alone can hide a large
    # RHS-relative error. Such a calculation must still be rejected.
    with pytest.raises(np.linalg.LinAlgError,match='dielectric residual'):
        solver.residues(energy,1.2,4,-.2,factors,None,0.)
    assert solver.info['dense_fallbacks'] > 0
    assert solver.info['max_relative_residual'] > 10*solver.tol
