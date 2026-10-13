"""Default molecular GW/BSE physics against installed PySCF 2.14."""
import inspect
import numpy as np
import pytest

pyscf = pytest.importorskip('pyscf')
from pyscf import gto, scf
from pyscf.gw import gw_ac
from pyqed.gw.gw import GW
from pyqed.gw.bse import BSE, _ensure_bse_screening


@pytest.mark.parametrize('density_fit', [False, True])
def test_default_gw_matches_pyscf_with_stability_checks(density_fit):
    mol=gto.M(atom='H 0 0 0; H 0 0 .74',basis='sto-3g',verbose=0)
    mf=scf.RHF(mol);mf.chkfile=None;mf.conv_tol=1e-12;mf.kernel()
    if density_fit:
        mf=mf.density_fit();mf.kernel()
    q=GW(mf).run()
    p=gw_ac.GWAC(mf);p.kernel()
    assert q.freq_int=='analytic_continuation'
    assert q.eta==p.eta and not q.vhf_df
    assert q.ac_info['stability']['checked'] and q.ac_info['stability']['passed']
    assert len(q.ac_info['frequencies'])==p.nw==100
    np.testing.assert_allclose(q.e_qp,p.mo_energy,atol=1e-6,rtol=0)
    np.testing.assert_allclose(q.ac_static,0,atol=1e-12)


@pytest.fixture
def reference():
    mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis='sto-3g',verbose=0)
    mf=scf.RHF(mol);mf.chkfile=None;mf.conv_tol=1e-12;mf.kernel()
    q=GW(mf)
    # Deliberate QP gap change makes accidental HF-screening reuse observable.
    q.e_qp=mf.mo_energy+np.r_[np.full(5,-.1),np.full(2,.15)]
    return q


def test_default_bse_matches_unmodified_pyscf_screening_and_roots(reference):
    pbse=pytest.importorskip('pyscf.gw.bse')
    q=reference;p=gw_ac.GWAC(q._scf);p.mo_energy=q.e_qp.copy();p.Lpq=q._pair_factors
    b=BSE(q).run(return_vectors=True)
    other=pbse.BSE(p);other.kernel('s')
    assert len(b.e)==other.nroot==10
    assert inspect.signature(BSE.run).parameters['max_cycle'].default==other.max_iter==100
    assert inspect.signature(BSE.run).parameters['tol'].default**2==other.residue_thresh
    ii,ia=pbse._get_lpq_bar_by_block([5],q.e_qp[None,:],[p.Lpq[:,:5,:5]],[p.Lpq[:,:5,5:]])
    np.testing.assert_allclose((p.Lpq[:,:5,:5]-4*b._static_screening.static_occupied(5)).transpose(1,0,2),ii[0],atol=1e-12)
    np.testing.assert_allclose((p.Lpq[:,:5,5:]-4*b._static_screening.static_ov(5)).transpose(1,0,2),ia[0],atol=1e-12)
    np.testing.assert_allclose(b.e,other.exci,atol=1e-9)
    old=BSE(q,screening_energies='hf').run(nroots=10,return_vectors=True)
    assert np.max(abs(old.e-b.e))>1e-5


def test_qp_screening_cache_refresh_and_full_space_window(reference):
    q=reference;b=BSE(q,orbital_window=(2,1));full=BSE(q)
    np.testing.assert_allclose(b._static_screening.cholesky[0],full._static_screening.cholesky[0])
    before=b._static_screening
    b.e_qp[-1]+=.02
    _ensure_bse_screening(b)
    assert b._static_screening is not before
    q.e_qp[-1]=np.nan
    with pytest.raises(ValueError,match='all GW energies'):
        BSE(q,orbital_window=(2,1))
    # Explicit HF mode can reuse an independently prepared full HF response.
    from pyqed.gw.contour import ContourDeformation
    q._contour=ContourDeformation(q,q._scf.mo_energy,[0],nw=8)
    BSE(q,orbital_window=(2,1),screening_energies='hf')


def test_screening_option_rejects_typo(reference):
    with pytest.raises(ValueError,match='screening_energies'):
        BSE(reference,screening_energies='qpp')


def test_window_use_mean_field_preserves_full_screening_space(reference):
    b=BSE(reference,orbital_window=(2,1)).run(nroots=1,use_qp=False)
    full=BSE(reference).run(nroots=1,use_qp=False)
    np.testing.assert_allclose(b._static_screening.cholesky[0],full._static_screening.cholesky[0],atol=1e-12)
