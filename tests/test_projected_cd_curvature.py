"""Early shell contraction versus late contraction and Cartesian projection."""
import numpy as np
import pytest
from pyqed.qchem import Molecule
from pyqed.qchem.eri_response import CoulombIntegrals


@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('basis_name',['sto-3g','cc-pvdz'])
def test_projected_curvature(workers,basis_name):
    mol=Molecule(atom='O 0 0 0; H .1 .8 .5; H -.2 -.7 .6',basis=basis_name).build(
        eri='cd',workers=workers,options={'eri_screen_tol':0.,'low_rank_tol':1e-10})
    response=CoulombIntegrals(mol)
    rng=np.random.default_rng(83)
    modes=rng.normal(size=(5,9))*.1
    npair=mol.nao*(mol.nao+1)//2
    weights=rng.normal(size=(2,len(mol._cd_factor_data['pivots']),npair))*.01
    early=response._cd_columns(modes,order=2,sensitivities=weights,workers=workers)
    assert response.derivative_info['recurrence_workspace_bytes'] < workers*34*1024**2
    late=response._cd_columns(modes,order=2,sensitivities=weights,workers=workers,early_contraction=False)
    np.testing.assert_allclose(early,late,atol=2e-11,rtol=0)
    cart=response._cd_columns(np.eye(9),order=2,sensitivities=weights,workers=workers)
    np.testing.assert_allclose(early,np.einsum('ai,ijk,bj->abk',modes,cart,modes),atol=2e-11,rtol=0)
    pairs=np.array([[4,1],[0,0],[1,4],[3,2]])
    selected=response._cd_columns(modes,order=2,sensitivities=weights,workers=workers,pairs=pairs)
    np.testing.assert_allclose(selected[:,0],early[pairs[:,0],pairs[:,1]],atol=2e-11,rtol=0)
    single=response._cd_columns(modes,order=2,sensitivities=weights,workers=workers,pair=21)
    np.testing.assert_allclose(single[0,0],early[4,1],atol=2e-11,rtol=0)
