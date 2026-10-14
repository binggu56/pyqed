"""AC restart reuses numerical state, not the original SCF or screening job."""
import numpy as np
import pytest
from test_gw_contour import model
from pyqed.gw import GW, BSE
from pyqed.gw.ac import kernel


@pytest.mark.parametrize('ac', ['pade', 'twopole'])
def test_ac_checkpoint_roundtrip(tmp_path, monkeypatch, ac):
    gw = GW.__new__(GW)
    gw.__dict__.update(vars(model()))
    gw.v_mf = np.zeros((10, 10))
    gw.freq_int = 'analytic_continuation'
    gw.vhf_df = True
    gw.threads = 1
    gw.method = 'g0w0'
    gw.info = None
    kernel(gw, gw.e_mf[::2], gw.mo_coeff, nw=32, ac=ac)
    gw.save(tmp_path/'gw.npz')
    with np.load(tmp_path/'gw.npz', allow_pickle=False) as data:
        assert data['energy_unit'] == 'hartree'
        np.testing.assert_array_equal(data['ac_sigma'], gw.ac_sigma)
    gw.checkpoint(tmp_path/'checkpoint')
    restored = GW.from_checkpoint(tmp_path/'checkpoint')
    assert isinstance(restored._pair_factors, np.memmap)
    monkeypatch.setattr('pyqed.gw.contour.ContourDeformation', lambda *a, **kw: pytest.fail('screening recomputed'))
    restored.refit(ac=ac)
    np.testing.assert_allclose(restored.e_qp, gw.e_qp, atol=1e-10)
    np.testing.assert_allclose(restored.ac_coeff, gw.ac_coeff)
    bse = BSE(restored, screening_energies='hf')
    assert bse._static_screening is restored._ac_screening.static_screening
    bse.run(nroots=2, return_vectors=True)
    assert len(bse.e) == 2
    with pytest.raises(RuntimeError, match='refit'):
        restored.run()
    with pytest.raises(FileExistsError):
        restored.checkpoint(tmp_path/'checkpoint')


def test_reject_unsupported_checkpoint(tmp_path):
    gw = GW.__new__(GW)
    gw.freq_int = 'exact'
    with pytest.raises(NotImplementedError, match='AC'):
        gw.checkpoint(tmp_path/'unsupported')
    assert not (tmp_path/'unsupported').exists()
