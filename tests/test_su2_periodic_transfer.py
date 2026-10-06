"""Exact reduced references for circular transfer compression."""
from types import SimpleNamespace
import numpy as np
import pytest
from pyqed.mps.nonabelian import spatial_target_sector
from pyqed.mps.nonabelian.periodic import PeriodicReducedMPS, parameter_action, run_periodic_sweeps
from pyqed.mps.nonabelian.periodic_transfer import ReducedTransferChain, ReducedCircularEnvironment
from pyqed.mps.nonabelian.environment import BlockSparseEnvironmentChain
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo
from pyqed.qchem.dmrg.backends.nonabelian import SU2DMRG


@pytest.fixture(scope='module')
def mpo():
    h=np.diag(np.linspace(-.2,.3,4))
    for i in range(3): h[i,i+1]=h[i+1,i]=-.7
    eri=np.zeros((4,)*4)
    for i in range(4): eri[i,i,i,i]=1.3
    q=SimpleNamespace(site='spatial',H=[None]*4,spin_purification=False,
        spatial_site_basis='fully_reduced',ncas=4,nelecas=4,spin=0,D=2,h1e=h,h2e=eri)
    return SU2DMRG().run(q,virtual_boundary='periodic',nsweeps=1).mpo


def state(complex_values=False):
    s=PeriodicReducedMPS.random(4,target_sector=spatial_target_sector(4,0),dimension=2,seed=17)
    if complex_values:
        rng=np.random.default_rng(71)
        for i in range(s.L): s.tensors[i]=s.tensor_from_vector(i,s.pack(i)+.2j*rng.normal(size=len(s.pack(i))))
    return s


@pytest.mark.parametrize('complex_values',[False,True])
def test_reduced_transfer_trace_adjoint_and_local_actions(mpo,complex_values):
    s=state(complex_values); rng=np.random.default_rng(9)
    identity=_identity_mpo_factors_for_sites_and_mpo(s.lifted_tensors(),mpo)
    for operator in [identity,mpo]:
        chain=ReducedTransferChain(s,operator); product=chain.product(range(4))
        full=product.apply(np.eye(4))
        np.testing.assert_allclose(np.trace(full),s.expectation(operator),rtol=1e-12,atol=1e-10)
        x=rng.normal(size=(4,3))+1j*rng.normal(size=(4,3))
        np.testing.assert_allclose(product.apply(x,True),full.conj().T@x,atol=1e-10)
        exact=BlockSparseEnvironmentChain.build(s.lifted_tensors(),operator)
        for site in range(4):
            env=ReducedCircularEnvironment(chain,[site],tolerance=0.).prepare(site)
            vector=rng.normal(size=len(s.pack(site)))+1j*rng.normal(size=len(s.pack(site)))
            np.testing.assert_allclose(env.parameter_action(site,vector),parameter_action(s,site,vector,exact),rtol=1e-11,atol=1e-10)
            assert env.hermiticity_error<1e-12
            assert env.info['rank']<=s.closure_dimension**2


@pytest.mark.parametrize('section',[[0,1],[2,3]])
def test_cached_factor_advance_survives_cyclic_qr(mpo,section):
    s=state(True); chain=ReducedTransferChain(s,mpo)
    cached=ReducedCircularEnvironment(chain,section,tolerance=0.)
    site=section[0]
    s.tensors[site]=s.tensor_from_vector(site,s.pack(site)*1.02)
    s.regauge(site); cached.advance(site)
    cached.prepare(site+1)
    fresh=ReducedCircularEnvironment(chain,[site+1],tolerance=0.).prepare(site+1)
    np.testing.assert_allclose(cached.matrix,fresh.matrix,atol=1e-10,rtol=1e-11)


def test_factor_svd_tail_and_rank_cap(mpo):
    s=state(True)
    identity=_identity_mpo_factors_for_sites_and_mpo(s.lifted_tensors(),mpo)
    chain=ReducedTransferChain(s,identity); p=chain.product([2,3])
    full=p.apply(np.eye(p.shape[1]))
    left,right,info=p.compress(tolerance=.99)
    assert info['rank']==1 and info['tolerance_met']
    error=np.linalg.norm(full-left@right)/np.linalg.norm(full)
    np.testing.assert_allclose(error,info['relative_frobenius_tail'],atol=1e-12)
    assert error<.99
    saved=[s.pack(i).copy() for i in range(s.L)]
    with pytest.raises(ValueError,match='rank cap'):
        run_periodic_sweeps(s,mpo,nsweeps=1,sweep_schedule='circular',environment='compressed',compression_tol=1e-14,compression_max_rank=1)
    for i,before in enumerate(saved): np.testing.assert_array_equal(s.pack(i),before)


def test_compressed_cycles_match_exact_reduced_circular_sweeps(mpo):
    s=state()
    exact=run_periodic_sweeps(s,mpo,nsweeps=2,sweep_schedule='circular')
    compressed=run_periodic_sweeps(s,mpo,nsweeps=2,sweep_schedule='circular',environment='compressed',compression_tol=0.)
    np.testing.assert_allclose([h['energy'] for h in exact['history']],
                               [h['energy'] for h in compressed['history']],atol=1e-10)
    np.testing.assert_allclose(exact['diagnostics']['max_one_site_residual'],
                               compressed['diagnostics']['max_one_site_residual'],atol=1e-8)
    assert compressed['diagnostics']['environment']=='compressed'
    assert all(r['hamiltonian']['rank']<=4 for h in compressed['history'] for r in h['compression'])


def test_untruncated_transfer_audit_matches_lifted_reference(mpo):
    from pyqed.mps.nonabelian.periodic import _audit_transfer_cycle,local_problem
    s=state(True); sites=s.lifted_tensors()
    identity=_identity_mpo_factors_for_sites_and_mpo(sites,mpo)
    chains=[ReducedTransferChain(s,m) for m in [mpo,identity]]
    energy,residual=_audit_transfer_cycle(s,chains,1e-11,4096)
    hc=BlockSparseEnvironmentChain.build(sites,mpo); nc=BlockSparseEnvironmentChain.build(sites,identity)
    problems=[local_problem(s,i,hc,nc) for i in range(s.L)]
    np.testing.assert_allclose(energy,problems[0][5],atol=1e-12)
    np.testing.assert_allclose(residual,max(p[6] for p in problems),atol=1e-10)


def test_exact_transfer_rejects_hermiticity_discrepancy(mpo,monkeypatch):
    chain=ReducedTransferChain(state(),mpo)
    environment=ReducedCircularEnvironment(chain,[0],tolerance=0.)
    def bad_matrix(*args):
        return np.array([[1.,1.],[0.,1.]])
    monkeypatch.setattr(chain,'local_matrix',bad_matrix)
    with pytest.raises(ArithmeticError,match='not Hermitian'):
        environment.prepare(0)


def test_compressed_local_size_guard_precedes_operator_allocation(mpo):
    with pytest.raises(MemoryError,match='max_parameters'):
        run_periodic_sweeps(state(),mpo,nsweeps=1,sweep_schedule='circular',environment='compressed',max_parameters=1)
