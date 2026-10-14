"""Static operator restarts must preserve actions and excitation energies."""
import numpy as np
import pytest
from pyqed.gw.bse import BSE, TDA, _bse_full_matmat
from pyqed.gw.screening import StaticScreening


def model(cls):
    obj = cls.__new__(cls)
    rng = np.random.default_rng(15)
    factors = rng.normal(size=(5, 4, 4)) * .01
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    obj.nocc = obj.full_nocc = 2
    obj.nso = 4
    obj.e_mf = np.array([-1., -.6, .3, .8])
    obj.e_qp = obj.e_mf + np.array([-.02, -.01, .01, .02])
    obj.mo_coeff = np.eye(4)
    obj.v_mf = np.zeros((4, 4))
    obj.orbital_indices = np.arange(4)
    obj.orbital_window = None
    obj._pair_factors = obj._screening_factors = factors
    obj._screening_mf_energy = obj.e_mf.copy()
    obj._screening_reference_energy = obj._screening_qp = obj.e_qp.copy()
    obj.screening = 'TDH'
    obj.screening_energies = 'qp'
    obj.max_memory = 100
    obj.threads = 1
    obj.eta = .01
    obj.verbose = 0
    obj.threading_info = obj.info = obj.bse_metric = None
    obj._xy = obj._x = obj._y = obj.e = None
    obj.eri = obj._M = obj.e_rpa = None
    obj._static_screening = StaticScreening(factors, (np.eye(5)*1.1, True), 100)
    return obj


@pytest.mark.parametrize('cls', [BSE, TDA])
def test_checkpoint_restart(tmp_path, monkeypatch, cls):
    obj = model(cls)
    obj.run(nroots=2, return_vectors=True, tol=1e-9)
    obj.save(tmp_path/'results.npz')
    with np.load(tmp_path/'results.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['excitation_energy'], obj.e)
        assert data['energy_unit'] == 'hartree'
        assert ('y' in data) == (cls is BSE)
    obj.checkpoint(tmp_path/'checkpoint')
    restored = BSE.from_checkpoint(tmp_path/'checkpoint')
    assert type(restored) is cls
    assert isinstance(restored._pair_factors, np.memmap)
    assert restored._pair_factors is restored._screening_factors
    # Any attempted rescreening fails, including on the subsequent solve.
    monkeypatch.setattr('pyqed.gw.contour.ContourDeformation', lambda *a, **k: pytest.fail('rescreening'))
    trial = np.arange(24.).reshape(8, 3)
    np.testing.assert_allclose(_bse_full_matmat(restored, trial, 2),
                               _bse_full_matmat(obj, trial, 2))
    restored.run(nroots=2, return_vectors=True, tol=1e-9)
    np.testing.assert_allclose(restored.e, obj.e, atol=1e-10)
    with pytest.raises(FileExistsError):
        obj.checkpoint(tmp_path/'checkpoint')


def test_save_before_run(tmp_path):
    with pytest.raises(ValueError, match='Run BSE'):
        model(BSE).save(tmp_path/'results.npz')
