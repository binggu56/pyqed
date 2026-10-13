"""Inverse-response reuse against complete dielectric and molecular references."""
import importlib

import numpy as np
import pytest
import scipy.linalg as la

from pyqed.gw.contour import ContourDeformation
from pyqed.gw.lanczos_response import LanczosResponse
from test_gw_contour import model, water


def test_inverse_action_equals_full_casida_inverse():
    gw = model()
    c = ContourDeformation(gw, gw.e_mf[::2], [0], nw=8, residue_solver='lanczos')
    response = LanczosResponse(c)
    root = np.sqrt(c.gaps)
    casida = np.diag(c.gaps**2)+4*root[:,None]*(c.pairs.T@c.pairs)*root[None,:]
    v = np.random.default_rng(2).normal(size=len(c.gaps))
    np.testing.assert_allclose(response.inverse_action(v), la.solve(casida,v), atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize('frequency', [0., .03, .1, .3, 1.13, 2.4])
def test_residue_value_and_derivative_equal_direct(frequency):
    gw = model()
    direct = ContourDeformation(gw, gw.e_mf[::2], [0], nw=8)
    reused = ContourDeformation(gw, gw.e_mf[::2], [0], nw=8, residue_solver='lanczos',screening_tol=1e-12)
    for f in [frequency, frequency+1e-5]:
        np.testing.assert_allclose(reused.residue(f,0,1),direct.residue(f,0,1),atol=2e-11,rtol=2e-11)
    assert reused.screening_info['reused_responses'] >= 1


def test_unconverged_chain_raises():
    gw = model()
    c = ContourDeformation(gw,gw.e_mf[::2],[0],nw=8,residue_solver='lanczos',screening_maxiter=1)
    with pytest.raises(np.linalg.LinAlgError,match='Screening Lanczos failed'):
        c.residue(.3,0,1)


def test_zero_coupling():
    gw = model();gw._pair_factors[:]=0
    c = ContourDeformation(gw,gw.e_mf[::2],[0],nw=8,residue_solver='lanczos')
    assert c.residue(.3,0,1)==(0j,0j)


def test_derivative_and_near_pole_against_direct():
    gw = model();gw.eta=1e-4
    g = importlib.import_module('pyqed.gw.gw')
    pole = float(g.rpa(gw)[0][0])
    c=ContourDeformation(gw,gw.e_mf[::2],[0],nw=8,residue_solver='lanczos',screening_tol=1e-11)
    direct=ContourDeformation(gw,gw.e_mf[::2],[0],nw=8)
    np.testing.assert_allclose(c.residue(pole+1e-4,0,1),direct.residue(pole+1e-4,0,1),atol=1e-7,rtol=1e-7)
    h=1e-5
    slope=(c.residue(.1+h,0,1)[0]-c.residue(.1-h,0,1)[0])/(2*h)
    np.testing.assert_allclose(c.residue(.1,0,1)[1],slope,atol=1e-9,rtol=1e-9)


def test_molecular_qp_and_bse_equal_current_solver(water):
    from pyqed.gw.bse import BSE
    g = importlib.import_module('pyqed.gw.gw')
    results=[]
    for solver in ['gmres','lanczos']:
        gw = g.GW(water,freq_int='contour',eta=1e-6).run(nw=8,residue_solver=solver,screening_tol=1e-12)
        results.append((gw.e_qp.copy(),gw.qp_weights.copy(),BSE(gw).run(nroots=3,low_rank=False).e))
    for a,b in zip(*results):np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-9)
