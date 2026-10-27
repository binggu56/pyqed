"""Reduced NC restart and actual threaded pair execution."""
import json
import numpy as np
import pytest
from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG
from examples.qchem.validate_finite_nc import model
from examples.qchem.benchmark_nc_stability import guards


@pytest.mark.parametrize('group,orbital,target',[
    ('D3h',"E'","E'"),('D6h','E2u','E2g')])
def test_restart_matches_uninterrupted_and_retains_complex_partners(tmp_path,group,orbital,target):
    h,v,options=model(group,orbital,target,shells=4)
    with guards():
        full=MultipletDMRG.from_integrals(h,v,**options)
        split=MultipletDMRG.from_integrals(h,v,**options)
        for solver in (full,split):solver.vectors[0]=solver.vectors[0]*(1+.3j)
        full.run(nsweeps=6,min_sweeps=6)
        split.run(nsweeps=3,min_sweeps=3)
        path=split.save_checkpoint(tmp_path/'state.npz')
        resumed=MultipletDMRG.from_integrals(h,v,**dict(options,D=1,n_threads=2)).load_checkpoint(path)
        assert resumed.completed_sweeps==3
        assert not resumed.success
        np.testing.assert_allclose(resumed.e_tot,split.e_tot,atol=1e-12,rtol=0)
        resumed.run(nsweeps=3,min_sweeps=3)
    assert resumed.completed_sweeps==6
    assert resumed.success and full.success
    np.testing.assert_allclose(resumed.e_tot,full.e_tot,atol=1e-12,rtol=0)
    for a,b in zip(resumed.states,full.states):
        assert abs(abs(a.overlap(b))-1)<1e-10
    assert resumed.symmetry_diagnostics()['covariance_error']<1e-10
    records=[s for cycle in resumed.history for s in cycle['sites']]
    assert max(s['pair_threads_used'] for s in records)==2
    assert sum(s['pair_parallel_calls'] for s in records)>0


def test_rejects_incompatible_and_corrupt_checkpoint_without_changing_state(tmp_path):
    h,v,options=model('D3h',"E'","E'",shells=3)
    solver=MultipletDMRG.from_integrals(h,v,**options).run(nsweeps=4)
    path=solver.save_checkpoint(tmp_path/'state.npz')
    changed=MultipletDMRG.from_integrals(h+.01*np.eye(len(h)),v,**options)
    with pytest.raises(ValueError,match='different Hamiltonian'):
        changed.load_checkpoint(path)
    energy=solver.energy;vectors=[v.copy() for v in solver.vectors]
    with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    metadata=json.loads(str(arrays['metadata']));metadata['energy']+=1
    arrays['metadata']=np.array(json.dumps(metadata))
    np.savez(tmp_path/'bad.npz',**arrays)
    with pytest.raises(ValueError,match='energy'):
        solver.load_checkpoint(tmp_path/'bad.npz')
    assert solver.energy==energy
    for a,b in zip(solver.vectors,vectors):np.testing.assert_array_equal(a,b)


def test_sparse_pair_halves_do_not_allocate_dense_operator_matrices():
    from scipy.sparse import eye
    from pyqed.mps.nonabelian import _su2_kernel as kernels
    size=8000  # A dense half would require about 1 GiB.
    action=kernels.NCPairHamiltonian({0:np.zeros((size,1))},
        {(0,0,0):(eye(size,format='csr'),np.ones((1,1)))},2)
    vector=np.arange(size,dtype=complex)*(1+.1j)
    np.testing.assert_array_equal(action(vector),vector)
    np.testing.assert_array_equal(action.diagonal(),np.ones(size))


@pytest.mark.parametrize('group,orbital,target',[
    ('D3h',"E'","E'"),('D6h','E2u','E2g')])
def test_sweep_boundary_reuse_matches_rebuilding_after_each_cycle(group,orbital,target,monkeypatch):
    h,v,options=model(group,orbital,target,shells=4)
    options['D']=1
    cached=MultipletDMRG.from_integrals(h,v,**options)
    rebuilt=MultipletDMRG.from_integrals(h,v,**options)
    for solver in (cached,rebuilt):
        solver.D=4
        solver.vectors[0]=solver.vectors[0]*(1+.2j)
    calls=[];original=cached._boundaries
    def boundaries(direction):
        calls.append(direction)
        return original(direction)
    monkeypatch.setattr(cached,'_boundaries',boundaries)
    cached.run(nsweeps=4,min_sweeps=4)
    for _ in range(4):rebuilt.run(nsweeps=1,min_sweeps=1)
    assert calls==['lr']
    np.testing.assert_allclose(cached.e_tot,rebuilt.e_tot,atol=1e-11,rtol=0)
    for a,b in zip(cached.states,rebuilt.states):assert abs(abs(a.overlap(b))-1)<1e-10
