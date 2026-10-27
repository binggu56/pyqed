"""Stopping must not confuse a small energy change with a solved local problem."""
import numpy as np
import pytest


def test_zero_tolerance_forces_requested_sweeps():
    control = SweepControl(2,tolerance=0)
    assert not control.finish({},step=1,energy_delta=0.,metric=0.)
from pyqed.mps.nonabelian.sweep_control import SweepControl


def test_full_cycle_boundary_minimum_and_failed_local_solve():
    control = SweepControl(6, tolerance=1e-8, min_steps=2)
    for options in [dict(step=1), dict(step=2,boundary=False),
                    dict(step=2,local_converged=False)]:
        assert not control.finish({},energy_delta=1e-10,**options)
    assert not control.finish({},step=2,energy_delta=1e-10,metric=1e-4)
    assert control.finish({},step=2,energy_delta=1e-10,metric=1e-10)


@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_nonfinite_objective_never_converges(value):
    control = SweepControl(2,tolerance=1e-8)
    assert not control.finish({},step=2,energy_delta=value)
    assert not control.finish({},step=2,metric=value)


def test_state_average_and_metric_only_policies():
    control = SweepControl(2,tolerance=1e-8)
    assert not control.finish({},step=2)
    record={}
    assert control.finish(record,step=2,energy_delta=1e-10)
    assert record['convergence_metric']=='energy_delta'
    assert control.finish(record,step=2,metric=1e-10)
    assert record['convergence_metric']=='metric'
    assert not control.finish(record,step=2,metric=1.)
    assert 'converged' not in record


@pytest.mark.parametrize('options',[dict(steps=0),dict(steps=True),dict(steps=1.5),
    dict(steps=2,min_steps=3),dict(steps=2,tolerance=np.nan)])
def test_invalid_controls_fail_before_sweeping(options):
    with pytest.raises(ValueError): SweepControl(**options)


def test_su2_shared_control_keeps_compiled_execution_and_exact_energy():
    from pyqed.qchem import Molecule
    from pyqed.qchem.dmrg import ED
    from pyqed.qchem.dmrg.dmrg import QCDMRG
    mol=Molecule(atom='H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8',
                 unit='bohr',basis='sto-3g').build(eri='s8')
    mf=mol.RHF();mf.chkfile=None;mf.run(tol=1e-11,verbose=0)
    reference=ED(mf,4,4,symmetry='su2',verbose=0).run()
    solver=QCDMRG(mf,4,4,D=32,symmetry='su2',init_guess='hf',verbose=0)
    solver.run(nsweeps=12,conv_tol=1e-9,max_bond_mode='per_sector',
               su2_kernel_backend='cpp',mixer_zero_block_noise_scale=0.)
    assert solver.dmrg.success
    assert solver.dmrg.diagnostics['sweep_control']=='shared'
    assert solver.dmrg.diagnostics['max_bond_mode']=='per_sector'
    assert all(r['cpp_owned_half_sweep'] for r in solver.dmrg.history)
    np.testing.assert_allclose(solver.e_tot,reference.e_tot,atol=1e-9,rtol=0)
