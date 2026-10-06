from types import SimpleNamespace

import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casci import CASCI
from pyqed.qchem.mcscf import casscf
from pyqed.qchem.mcscf.casscf import FirstOrderCASSCF, SecondOrderCASSCF


@pytest.fixture(scope='module')
def mf():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', unit='angstrom').build()
    return mol.RHF().run()


@pytest.mark.parametrize('driver', [FirstOrderCASSCF, SecondOrderCASSCF])
@pytest.mark.parametrize('failure', ['unconverged', 'missing_status', 'energy_nan',
                                     'coefficient_nan', 'missing_root', 'empty_root'])
def test_inner_ci_rejects_invalid_result(driver, failure):
    solver = object.__new__(driver)
    solver.converged = True
    mc = SimpleNamespace(converged=True, e_tot=np.array([-1., -.5]),
                         ci=[np.array([1., 0.]), np.array([0., 1.])])
    if failure == 'unconverged':
        mc.converged = False
    elif failure == 'missing_status':
        del mc.converged
    elif failure == 'energy_nan':
        mc.e_tot[0] = np.nan
    elif failure == 'coefficient_nan':
        mc.ci[0][0] = np.nan
    elif failure == 'missing_root':
        mc.ci.pop()
    else:
        mc.ci[0] = np.array([])
    with pytest.raises(RuntimeError, match='converged inner CI'):
        solver._require_converged_ci(mc, 2)
    assert not solver.converged


def test_dense_casci_status_and_failed_rerun(monkeypatch, mf):
    mc = CASCI(mf, 2, 2, verbose=0).run(method='ci')
    assert mc.converged
    def fail(*args, **kwargs):
        raise RuntimeError('injected integral failure')
    monkeypatch.setattr(mc, 'get_SO_matrix', fail)
    with pytest.raises(RuntimeError, match='injected'):
        mc.run(method='ci')
    assert not mc.converged


@pytest.mark.parametrize('path', ['molecular', 'integral', 'factor'])
def test_ci_entry_paths_reject_before_caching(monkeypatch, mf, path):
    solver = SecondOrderCASSCF(mf, 2, 2, verbose=0)
    def failed_ci(mc, **kwargs):
        mc.converged = False
        mc.e_tot = np.array([-1.])
        mc.ci = [np.array([1.])]
        return mc
    def unexpected_cache(*args):
        pytest.fail('Invalid CI reached the cache')
    monkeypatch.setattr(casscf.CASCI, 'run', failed_ci)
    monkeypatch.setattr(solver, '_update_casci_cache', unexpected_cache)
    n = mf.mo_coeff.shape[1]
    with pytest.raises(RuntimeError, match='converged inner CI'):
        if path == 'molecular':
            solver._make_casci(mf.mo_coeff, 1)
        elif path == 'integral':
            solver._make_integral_casci(np.zeros((n,n)), np.zeros((n,)*4), mf.mo_coeff, 1)
        else:
            solver._make_factor_integral_casci(np.zeros((n,n)), np.zeros((1,n,n)), mf.mo_coeff, 1)
    assert not solver.converged


@pytest.mark.parametrize('invalid', [False, True])
def test_inner_ci_validates_reduced_mps_blocks(invalid):
    from pyqed.mps.nonabelian import MPS, build_random_reduced_spatial_mps, spatial_target_sector
    root = MPS(build_random_reduced_spatial_mps(2, target_sector=spatial_target_sector(2, 0), bond_multiplicity=2), target_sector=spatial_target_sector(2, 0))
    if invalid:
        next(iter(root.tensors[0].data.values())).flat[0] = np.nan
    solver = object.__new__(FirstOrderCASSCF)
    solver.converged = True
    mc = SimpleNamespace(converged=True, e_tot=np.array([-1.]), ci=[root])
    if invalid:
        with pytest.raises(RuntimeError, match='converged inner CI'):
            solver._require_converged_ci(mc, 1)
    else:
        solver._require_converged_ci(mc, 1)
        assert solver.converged
