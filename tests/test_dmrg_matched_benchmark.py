"""Shared reduced MPS interchange and physical endpoint convergence checks."""
import json
from pathlib import Path
import subprocess
import sys
import os
import numpy as np
import pytest
from examples.qchem.benchmark_matched_dmrg import converged


def test_endpoint_convergence_requires_adjacent_complete_sweeps():
    previous=dict(energy=-1.,half_sweeps=2)
    assert converged(previous,dict(energy=-1.+1e-8,half_sweeps=4),1e-6)
    assert not converged(previous,dict(energy=-1.+1e-4,half_sweeps=4),1e-6)
    assert not converged(previous,dict(energy=-1.,half_sweeps=6),1e-6)


@pytest.mark.parametrize('spatial',[False,True])
def test_block2_shared_singlet_initial_energy(tmp_path, spatial):
    pytest.importorskip('pyblock2')
    pytest.importorskip('pyscf')
    from examples.qchem.benchmark_abelian_dmrg_scaling import model, transform_active_space_spatial_integrals
    from examples.qchem._abelian_dmrg_worker import load_shared_state
    from pyqed.qchem.dmrg.dmrg import DMRG
    from pyqed.qchem.dmrg.backends.nonabelian import _expectation_from_nonabelian_mps
    from pyqed.qchem.symmetry import resolve_mo_irreps
    from pyscf import symm
    mf,c=model(4)
    h,g,e=transform_active_space_spatial_integrals(mf,c,4,0,use_cholesky=False)
    labels,_,_=resolve_mo_irreps(mf,c)
    orbsym=[symm.irrep_name2id('D2h',s) for s in labels] if spatial else [0]*4
    integral=tmp_path/'integrals.npz';state=tmp_path/'state.npz';result=tmp_path/'result.json'
    np.savez(integral,ncas=4,n_elec=4,spin=0,ecore=e,h1e=h,g2e=g,orb_sym=orbsym)
    worker=Path(__file__).resolve().parents[1]/'examples/qchem/_block2_su2_worker.py'
    subprocess.run([sys.executable,str(worker),'--input',str(integral),'--output',str(result),
        '--shared-state',str(state),'--bond-dim','8','--half-sweeps','2','--seed','7',
        '--davidson-tol','1e-7','--nstates','1','--mpo-algorithm','conventional','--fixed-sweeps'],
        env=os.environ.copy(),check=True,capture_output=True)
    initial=load_shared_state(state,mf.mol.symmetry_info.group if spatial else None)
    driver=DMRG(mf,4,4,D=8,symmetry='su2',verbose=0)
    solver=driver.run(mo_coeff=c,initial_guess=initial,nsweeps=1,conv_tol=-1,
        require_convergence=False,verify_returned_mps_energy=True,**({'wfnsym':'Ag'} if spatial else {}))
    reference=json.loads(result.read_text())
    active=driver._active_hamiltonian
    observed=_expectation_from_nonabelian_mps(initial,active.mpo,moving_environment=active.moving_environment)
    assert observed == pytest.approx(reference['initial_energy'],abs=1e-12)
    assert driver.e_tot == pytest.approx(reference['energy'],abs=1e-10)
    if hasattr(solver, 'close'):
        solver.close()
