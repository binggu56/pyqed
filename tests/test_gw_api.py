"""The GW driver owns quasiparticles; BSE owns neutral excitations."""
import numpy as np
import pytest
from pyqed.gw import GW, BSE, TDA
from pyqed.qchem import Molecule


@pytest.fixture
def mf():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(eri='cd', options={'low_rank_tol': 1e-12, 'eri_screen_tol': 0.})
    return mol.RHF().run(tol=1e-12)


@pytest.mark.parametrize('cls', [BSE, TDA])
def test_requires_completed_gw_but_accepts_direct_scf(mf, cls):
    gw = GW(mf, freq_int='exact', eta=0.)
    with pytest.raises(ValueError, match='Run GW'):
        cls(gw)
    direct = cls(mf).run(nroots=1, low_rank=False)
    assert np.isfinite(direct.e).all()
    np.testing.assert_array_equal(direct.e_qp, mf.mo_energy)
    assert all(not hasattr(direct, name) for name in ('kernel', 'sigma', 'g0'))


def test_gw_bse_derivatives_chain(mf):
    gw = GW(mf, freq_int='exact', eta=0.).run()
    qp = gw.e_qp.copy()
    bse = BSE(gw, screening_energies='hf').run(nroots=1, low_rank=False, tol=1e-10)
    mode = np.array([[0., 0., -.5, 0., 0., .5]])
    gradient, hessian = bse.derivatives(order=2, state_ids=[0], modes=mode)
    assert np.isfinite(gradient).all() and np.isfinite(hessian).all()
    np.testing.assert_array_equal(gw.e_qp, qp)
    assert bse._adjoint_workspace is None and bse._first_response_cache is None
