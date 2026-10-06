"""Small explicit validation references for the constrained reduced path."""
from types import SimpleNamespace
import numpy as np
import pytest

from pyqed.qchem.dmrg.backends.nonabelian import SU2DMRG
from pyqed.mps.nonabelian import spatial_target_sector
from pyqed.mps.nonabelian.periodic import PeriodicReducedMPS, local_problem
from pyqed.mps.nonabelian.environment import BlockSparseEnvironmentChain, contract_chain_transition
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo


def context(n=2, interaction=1.3):
    h=np.diag(np.linspace(-.2,.3,n))
    for i in range(n-1): h[i,i+1]=h[i+1,i]=-.7
    eri=np.zeros((n,)*4)
    for i in range(n): eri[i,i,i,i]=interaction
    return SimpleNamespace(site='spatial',H=[None]*n,spin_purification=False,
               spatial_site_basis='fully_reduced',ncas=n,nelecas=n,spin=0,D=2,h1e=h,h2e=eri)


@pytest.mark.parametrize('interaction',[0.,1.3])
@pytest.mark.parametrize('environment',['exact','compressed'])
def test_periodic_backend_converges_hubbard_singlet_without_letta(monkeypatch,interaction,environment):
    from pyqed.letta import SU2LETTA
    monkeypatch.setattr(SU2LETTA,'from_integrals',lambda *a,**k:pytest.fail('LETTA solver used'))
    q=context(interaction=interaction)
    solver=SU2DMRG().run(q,virtual_boundary='periodic',nsweeps=6,seed=11,residual_tol=1e-8,environment=environment,
        sweep_schedule='circular' if environment=='compressed' else 'back_and_forth')
    reference_matrix=np.diag([2*q.h1e[0,0]+interaction,np.trace(q.h1e),2*q.h1e[1,1]+interaction])
    reference_matrix[0,1]=reference_matrix[1,0]=reference_matrix[1,2]=reference_matrix[2,1]=np.sqrt(2)*q.h1e[0,1]
    reference=np.linalg.eigvalsh(reference_matrix)[0]
    assert solver.converged
    assert solver.diagnostics['algorithm']=='reduced_periodic_one_site'
    np.testing.assert_allclose(solver.energy,reference,atol=1e-10)
    state=solver.ground_state
    assert state.bc=='periodic'
    assert all(b.shape[0]==b.shape[2]==2 for t in state.tensors for b in t.data.values())
    np.testing.assert_allclose(state.copy().norm_squared(),state.norm_squared(),atol=1e-12)
    from pyqed.mps.nonabelian.periodic import tangent_audit
    audit=tangent_audit(state,solver.mpo)
    assert audit['joint_tangent_residual']<1e-8
    assert audit['joint_metric_rank']==3


def test_trace_actions_metric_and_exact_internal_gauge():
    q=context(4)
    solver=SU2DMRG().run(q,virtual_boundary='periodic',nsweeps=1,seed=11)
    state=PeriodicReducedMPS.random(4,target_sector=spatial_target_sector(4,0),dimension=2,seed=17)
    rng=np.random.default_rng(8)
    for i in range(4):
        state.tensors[i]=state.tensor_from_vector(i,state.pack(i)+.2j*rng.normal(size=len(state.pack(i))))
    sites=state.lifted_tensors()
    identity=_identity_mpo_factors_for_sites_and_mpo(sites,solver.mpo)
    hc=BlockSparseEnvironmentChain.build(sites,solver.mpo)
    nc=BlockSparseEnvironmentChain.build(sites,identity)
    c,h,n,w,norm,energy,_=local_problem(state,1,hc,nc)
    variation=rng.normal(size=len(c))+1j*rng.normal(size=len(c))
    ket=sites.copy()
    from pyqed.mps.nonabelian.closure import lift_closure
    ket[1]=lift_closure(state.tensor_from_vector(1,variation),1,4,2)
    np.testing.assert_allclose(np.vdot(c,h(variation)),contract_chain_transition(sites,solver.mpo,ket),atol=1e-10)
    np.testing.assert_allclose(np.vdot(c,n(variation)),contract_chain_transition(sites,identity,ket),atol=1e-10)
    np.testing.assert_allclose(w.conj().T@np.column_stack([n(v) for v in w.T]),np.eye(w.shape[1]),atol=1e-8)
    original=state.pack(1).copy(); info={}
    conditioned=local_problem(state,1,hc,nc,norm_gauge=True,gauge_info=info)
    wc=conditioned[3]
    np.testing.assert_allclose(wc.conj().T@np.column_stack([n(v) for v in wc.T]),np.eye(wc.shape[1]),atol=1e-8)
    np.testing.assert_array_equal(state.pack(1),original)
    np.testing.assert_allclose(conditioned[5],energy,atol=1e-12)
    before=(state.norm_squared(),state.expectation(solver.mpo))
    state.balance()
    np.testing.assert_allclose((state.norm_squared(),state.expectation(solver.mpo)),before,atol=1e-9)


@pytest.mark.parametrize('norm_gauge',[False,True])
def test_moving_boundaries_match_fresh_rebuilds(norm_gauge):
    from pyqed.mps.nonabelian.periodic import run_periodic_sweeps
    q=context(4)
    solver=SU2DMRG().run(q,virtual_boundary='periodic',nsweeps=1)
    state=PeriodicReducedMPS.random(4,target_sector=spatial_target_sector(4,0),dimension=2,seed=19)
    moving=run_periodic_sweeps(state,solver.mpo,nsweeps=2,reuse_boundaries=True,norm_gauge=norm_gauge,sweep_schedule="circular")
    rebuilt=run_periodic_sweeps(state,solver.mpo,nsweeps=2,reuse_boundaries=False,norm_gauge=norm_gauge,sweep_schedule="circular")
    np.testing.assert_allclose([h['energy'] for h in moving['history']],
                               [h['energy'] for h in rebuilt['history']],atol=1e-10)
    np.testing.assert_allclose(moving['diagnostics']['max_one_site_residual'],
                               rebuilt['diagnostics']['max_one_site_residual'],atol=1e-8)


def test_cyclic_sector_gauges_preserve_complex_trace_and_condition_norm():
    from pyqed.mps.nonabelian.periodic import condition_site, _local_metric
    from pyqed.mps.periodic_tools import metric_whitener
    q=context(4)
    solver=SU2DMRG().run(q,virtual_boundary='periodic',nsweeps=1)
    state=PeriodicReducedMPS.random(4,target_sector=spatial_target_sector(4,0),dimension=2,seed=23)
    rng=np.random.default_rng(71)
    for i in range(state.L):
        state.tensors[i]=state.tensor_from_vector(i,state.pack(i)+.1j*rng.normal(size=state.pack(i).size))
    # Deliberately distort both sides of the charge-twisted seam with an exact gauge.
    sectors=set(state.tensors[0].qns[0])
    factor=np.array([[1e4,0],[0,1e-4]])
    state.apply_norm_gauge(0,{s:factor for s in sectors},{})
    before=(state.norm_squared(),state.expectation(solver.mpo))
    old_condition=None; new_condition=None
    for i in range(state.L):
        sites=state.lifted_tensors(); identity=_identity_mpo_factors_for_sites_and_mpo(sites,solver.mpo)
        nc=BlockSparseEnvironmentChain.build(sites,identity)
        if i==0: old_condition=metric_whitener(_local_metric(state,i,nc,4096),rtol=1e-15)[1]['retained_condition']
        condition_site(state,i,nc)
        if i==0:
            sites=state.lifted_tensors(); nc=BlockSparseEnvironmentChain.build(sites,identity)
            new_condition=metric_whitener(_local_metric(state,i,nc,4096),rtol=1e-15)[1]['retained_condition']
        np.testing.assert_allclose((state.norm_squared(),state.expectation(solver.mpo)),before,atol=1e-8,rtol=1e-9)
        state.regauge(i)
        np.testing.assert_allclose((state.norm_squared(),state.expectation(solver.mpo)),before,atol=1e-8,rtol=1e-9)
    assert new_condition<old_condition/100


def test_periodic_rejects_unsupported_spin_and_dimension_mode():
    q=context(); q.spin=2
    with pytest.raises(NotImplementedError,match='singlet'):
        SU2DMRG().run(q,virtual_boundary='periodic',nsweeps=1)
    q.spin=0
    with pytest.raises(ValueError,match='per sector'):
        SU2DMRG().run(q,virtual_boundary='periodic',max_bond_mode='global_reduced',nsweeps=1)


def test_qchem_public_periodic_su2_path(monkeypatch):
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf import RHF
    from pyqed.qchem.dmrg.dmrg import QCDMRG
    mol=Molecule(atom='H 0 0 0; H 0 0 1.4',unit='bohr',basis='sto-3g').build(eri='dense',aosym='s8')
    q=QCDMRG(RHF(mol).run(),ncas=2,nelecas=2,D=2,verbose=0)
    solver=q.run(symmetry='su2',virtual_boundary='periodic',nsweeps=6)
    assert isinstance(solver,SU2DMRG)
    assert solver.converged and q.converged
    assert solver.diagnostics['environment']=='compressed'
    assert solver.diagnostics['sweep_schedule']=='circular'
    import pyqed.qchem.dmrg.dmrg as dmrg_module
    monkeypatch.setattr(dmrg_module, '_nonabelian_mps_to_dense_vector', lambda *a, **k: pytest.fail('spin diagnostic expanded SU2 state'))
    assert q.spin_square() == 0.
    assert q.calc_spin_square() == 0.
    np.testing.assert_allclose(solver.energy,-1.137275943783,atol=1e-8)
    dm1,dm2=q.make_rdm12(spatial=True)
    np.testing.assert_allclose(np.trace(dm1),2.,atol=1e-9)
    electronic=np.einsum('pq,pq',q.h1e[0],dm1)+.5*np.einsum('pqrs,pqrs',q.h2e[0,0],dm2)
    np.testing.assert_allclose(electronic,solver.e_active,atol=1e-9)
