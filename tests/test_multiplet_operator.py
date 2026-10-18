"""Reduced operator recursion against explicit local-component references."""
import numpy as np
import pytest
from pyqed.qchem.symmetry import irrep_representation_matrices
from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG
from pyqed.mps.nonabelian.multiplet_operator import OperatorBasis,MultipletMPO,advance,boundary


def mpo_overlap(a,b):
    env=np.ones((1,1),complex)
    for x,y in zip(a,b):
        env=np.einsum('ab,acpq,bdpq->cd',env,x.conj(),y,optimize=True)
    return env.item()


@pytest.mark.parametrize('group,e,target', [('D3h',"E'","A1'"),('D6h','E1u','E2g')])
def test_reduced_operator_mpo_and_environments(group,e,target):
    reps={g:np.kron(np.eye(2),u) for g,u in irrep_representation_matrices(group,e).items()}
    h=np.kron([[.1,.3],[.3,.7]],np.eye(2))
    v=.2*np.einsum('pq,rs->pqrs',np.eye(4),np.eye(4))
    solver=MultipletDMRG.from_integrals(h,v,nelec=2,D=2,point_group=group,wfnsym=target,orbital_representations=reps,contractions="components").run(nsweeps=4)
    legs=[s.leg for s in solver.sites]
    for leg in legs:
        basis=OperatorBasis(leg).basis.reshape(leg.full_dim**2,-1)
        np.testing.assert_allclose(basis.conj().T@basis,np.eye(len(basis)),atol=1e-12)
    reduced=MultipletMPO.from_components(solver.mpo,legs)
    expanded=reduced.component_cores()
    a,b,c=mpo_overlap(solver.mpo,solver.mpo),mpo_overlap(expanded,expanded),mpo_overlap(solver.mpo,expanded)
    np.testing.assert_allclose(a+b-2*c.real,0,atol=1e-10)
    for state in solver.states:
        np.testing.assert_allclose(reduced.expectation(state),state.expectation(solver.mpo),atol=1e-10)
    tensors=solver.states[0].tensors
    left=boundary(tensors[0].legs[0],reduced.legs[0],False)
    right=boundary(tensors[-1].legs[-1],reduced.legs[-1],True)
    for i in range(len(tensors)):
        left=advance(reduced.symmetry,left,reduced[i],tensors[i].data,tensors[i].data,'lr')
    for i in reversed(range(len(tensors))):
        right=advance(reduced.symmetry,right,reduced[i],tensors[i].data,tensors[i].data,'rl')
    np.testing.assert_allclose(next(iter(left.values())),next(iter(right.values())),atol=1e-10)


@pytest.mark.parametrize('group,e', [('D3h',"E'"),('D3h',"E''"),
    ('D6h','E1g'),('D6h','E2g'),('D6h','E1u'),('D6h','E2u')])
@pytest.mark.parametrize('compiled',[True,False])
def test_fully_reduced_sweep_and_matrix_free_action(group,e,compiled,monkeypatch):
    if not compiled:
        from pyqed.mps.nonabelian import multiplet_sweep
        monkeypatch.setattr(multiplet_sweep,'_su2_kernel_module',lambda:None)
    from pyqed.mps.nonabelian.fusion import TensorFusionMap
    from pyqed.mps.nonabelian.multiplet_dmrg import _contract
    reps={g:np.kron(np.eye(2),u) for g,u in irrep_representation_matrices(group,e).items()}
    h=np.kron([[.1,.3],[.3,.7]],np.eye(2))
    def forbidden(*args,**kwargs):
        pytest.fail('Component expansion entered a reduced sweep.')
    with monkeypatch.context() as patch:
        patch.setattr(TensorFusionMap,'expand',forbidden)
        patch.setattr(TensorFusionMap,'matrix',property(forbidden))
        patch.setattr(MultipletMPO,'component_cores',forbidden)
        solver=MultipletDMRG.from_integrals(h,np.zeros((4,)*4),nelec=1,spin=1,D=2,
            point_group=group,wfnsym=e,orbital_representations=reps).run(nstates=2,nsweeps=4)
    assert not hasattr(solver,'components')
    np.testing.assert_allclose(solver.e_tot,np.linalg.eigvalsh(h),atol=1e-10)
    assert solver.success and solver.symmetry_diagnostics()['covariance_error']<1e-12
    cores=solver.mpo.component_cores()
    tensors=[f.expand(v) for f,v in zip(solver.fusion_maps,solver.vectors)]
    lh,_=solver._boundaries('rl');rh,_=solver._boundaries('lr')
    for site,fusion in enumerate(solver.fusion_maps):
        left=np.ones((1,1,1),complex)
        dim=tensors[-1].shape[-1]
        right=np.eye(dim,dtype=complex)[:,None,:]/dim
        for i in range(site):
            left=_contract('axb,apc,xypq,bqd->cyd',left,tensors[i].conj(),cores[i],tensors[i])
        for i in range(len(tensors)-1,site,-1):
            right=_contract('apc,xypq,bqd,cyd->axb',tensors[i].conj(),cores[i],tensors[i],right)
        rng=np.random.default_rng(61)
        vector=rng.normal(size=fusion.matrix.shape[1])+1j*rng.normal(size=fusion.matrix.shape[1])
        expanded=fusion.expand(vector)
        expected=fusion.matrix.conj().T@_contract('axb,xypq,bqd,cyd->apc',left,cores[site],expanded,right).ravel()
        action=solver._local_action(site,solver.mpo[site],lh[site],rh[site+1])
        np.testing.assert_allclose(action(vector),expected,atol=1e-11)
        matrix=solver._local_action(site,solver.mpo[site],lh[site],rh[site+1],assemble=True)
        np.testing.assert_allclose(matrix@vector,expected,atol=1e-11)
        np.testing.assert_allclose(action.diagonal(),matrix.diagonal(),atol=1e-11)
        np.testing.assert_allclose(matrix,matrix.conj().T,atol=1e-11)


def test_zero_operator_keeps_scalar_boundaries():
    reps=irrep_representation_matrices('D3h',"E'")
    solver=MultipletDMRG.from_integrals(np.zeros((2,2)),np.zeros((2,)*4),
        nelec=1,spin=1,D=1,point_group='D3h',wfnsym="E'",orbital_representations=reps).run()
    np.testing.assert_allclose(solver.e_tot,[0,0],atol=1e-12)
    assert solver.success


def test_small_operator_scale_is_not_discarded():
    reps=irrep_representation_matrices('D3h',"E'")
    solver=MultipletDMRG.from_integrals(1e-14*np.eye(2),np.zeros((2,)*4),cutoff=0,
        nelec=1,spin=1,D=1,point_group='D3h',wfnsym="E'",orbital_representations=reps).run()
    np.testing.assert_allclose(solver.e_tot,[1e-14,1e-14],atol=1e-27,rtol=1e-12)


@pytest.mark.parametrize('group,e,target', [('D3h',"E'","A1'"),('D6h','E1u','E2g')])
@pytest.mark.parametrize('compiled',[False,True])
def test_batched_environment_complex_values_and_plan_reuse(group,e,target,compiled,monkeypatch):
    if not compiled:
        from pyqed.mps.nonabelian import environment
        monkeypatch.setattr(environment,'_su2_kernel_module',lambda:None)
    from pyqed.mps.nonabelian.multiplet_operator import routes,dimension,_advance_plan
    reps={g:np.kron(np.eye(2),u) for g,u in irrep_representation_matrices(group,e).items()}
    solver=MultipletDMRG.from_integrals(np.kron([[.1,.3],[.3,.7]],np.eye(2)),
        np.zeros((4,)*4),nelec=2,D=2,point_group=group,wfnsym=target,orbital_representations=reps)
    rng=np.random.default_rng(91)
    def complex_blocks(blocks):
        return {k:rng.normal(size=v.shape)+1j*rng.normal(size=v.shape) for k,v in blocks.items()}
    sym=solver.mpo.symmetry
    for direction in ('lr','rl'):
        env=solver._edge(direction=='rl')[0]
        for i in (range(solver.nsites) if direction=='lr' else reversed(range(solver.nsites))):
            bra=complex_blocks(solver._tensor(i).data)
            ket=complex_blocks(solver._tensor(i).data)
            core=complex_blocks(solver.mpo[i])
            for repeat in range(2):
                env=complex_blocks(env)
                expected={}
                for ak,bk,wk,lk,rk,weight in routes(sym,tuple(bra),tuple(ket),tuple(core)):
                    source,target=(lk,rk) if direction=='lr' else (rk,lk)
                    if source not in env:continue
                    weight/=dimension(sym,ak[2] if direction=='lr' else ak[0])
                    if direction=='lr':
                        value=np.einsum('axb,apc,xypq,bqd->cyd',env[source],bra[ak].conj(),core[wk],ket[bk],optimize=True)
                    else:
                        value=np.einsum('apc,xypq,bqd,cyd->axb',bra[ak].conj(),core[wk],ket[bk],env[source],optimize=True)
                    expected[target]=expected.get(target,0)+weight*value
                hits=_advance_plan.cache_info().hits
                actual=advance(sym,env,core,bra,ket,direction)
                if repeat:assert _advance_plan.cache_info().hits>hits
                assert actual.keys()==expected.keys()
                for key in actual:np.testing.assert_allclose(actual[key],expected[key],atol=1e-10)
                for value in core.values():value*=.7+.2j
            env=actual


def test_reverse_sweep_reuses_environments(monkeypatch):
    reps={g:np.kron(np.eye(2),u) for g,u in irrep_representation_matrices('D3h',"E'").items()}
    solver=MultipletDMRG.from_integrals(np.kron([[.1,.3],[.3,.7]],np.eye(2)),
        np.zeros((4,)*4),nelec=2,D=2,point_group='D3h',wfnsym="A1'",orbital_representations=reps)
    calls=[];original=solver._boundaries
    def record(direction):
        calls.append(direction)
        return original(direction)
    monkeypatch.setattr(solver,'_boundaries',record)
    solver.run(nsweeps=4)
    assert solver.success and len(solver.history)>=2
    # One initial suffix build; later suffixes/prefixes come from moving sweeps.
    assert calls == ['lr']
    for state in solver.states:
        np.testing.assert_allclose(state.expectation(solver.mpo.component_cores()).real,
            solver.energy,atol=1e-11)


@pytest.mark.parametrize('complex_values',[False,True])
def test_compiled_multiplet_local_routes(complex_values):
    kernel=pytest.importorskip('pyqed.mps.nonabelian._su2_kernel')
    rng=np.random.default_rng(19)
    def random(shape):
        a=rng.normal(size=shape)
        return a+1j*rng.normal(size=shape) if complex_values else a
    # Unequal multiplicities, overlapping routes, offsets and strided inputs.
    l=random((3,2,4))[:,::-1,:];w=random((2,3,2,3));r=random((2,3,3))
    routes=[(2,5,l,w,r,.7+.2j),(2,5,l,w,r,-.3),(0,1,l,w,r,.1)]
    action=kernel.MultipletLocalAction(45,routes)
    expected=np.zeros((45,45),complex)
    for a,b,left,op,right,weight in routes:
        expected[a:a+12,b:b+36]+=weight*np.einsum('axb,xypq,cyd->apcbqd',left,op,right,optimize=True).reshape(12,36)
    vector=random((45,))
    np.testing.assert_allclose(action.assemble(),expected,atol=1e-12)
    np.testing.assert_allclose(action.diagonal(),expected.diagonal(),atol=1e-12)
    np.testing.assert_allclose(action(vector),expected@vector,atol=1e-11)
    listed=kernel.MultipletLocalAction(45,[(a,b,(left*(i+1)).tolist(),op.tolist(),right.tolist(),weight/(i+1))
        for i,(a,b,left,op,right,weight) in enumerate(routes)])
    np.testing.assert_allclose(listed(vector),expected@vector,atol=1e-11)
    # Plans own factors: mutating caller inputs cannot change an active solve.
    l[:]=0;w[:]=0;r[:]=0
    np.testing.assert_allclose(action(vector),expected@vector,atol=1e-11)
    with pytest.raises(ValueError):action(np.ones(44))
    with pytest.raises(ValueError):kernel.MultipletLocalAction(5,routes)
    with pytest.raises(ValueError):kernel.MultipletLocalAction(45,[(0,0,l,w[:1],r,1)])
    block=random((2,3,2,2,5))[:,::-1]
    cg=random((3,2,2));cg[0]=0
    np.testing.assert_allclose(kernel.reduce_operator_channel(block,cg),
        np.einsum('ixjky,xkr->ijry',block,cg.conj()),atol=1e-12)
    with pytest.raises(ValueError):kernel.reduce_operator_channel(block,cg[:1])


def test_multiplet_basis_cache_owns_arrays():
    from pyqed.mps.nonabelian.multiplet_sites import OrbitalMultipletSite
    a=OrbitalMultipletSite('D6h','E1u')
    expected=a.basis.copy()
    a.basis[:]=0;a.number[:]=0
    for value in a.representations.values():value[:]=0
    b=OrbitalMultipletSite('D6h','E1u')
    np.testing.assert_allclose(b.basis,expected,atol=1e-14)
    assert np.linalg.norm(b.number)>0
    assert all(np.linalg.norm(value)>0 for value in b.representations.values())


def test_local_reduced_contraction_fallback(monkeypatch):
    from pyqed.mps.nonabelian import multiplet_sweep
    reps={g:np.kron(np.eye(2),u) for g,u in irrep_representation_matrices('D3h',"E'").items()}
    h=np.kron([[.1,.3],[.3,.7]],np.eye(2));v=np.zeros((4,)*4)
    solver=MultipletDMRG.from_integrals(h,v,nelec=2,D=2,point_group='D3h',wfnsym="A1'",orbital_representations=reps)
    lh,_=solver._boundaries('rl');rh,_=solver._boundaries('lr')
    expected=solver._local_action(0,solver.mpo[0],lh[0],rh[1],assemble=True)
    monkeypatch.setattr(multiplet_sweep,'_su2_kernel_module',lambda:None)
    np.testing.assert_allclose(solver._local_action(0,solver.mpo[0],lh[0],rh[1],assemble=True),expected,atol=1e-12)
    vector=np.arange(len(expected))+1j
    np.testing.assert_allclose(solver._local_action(0,solver.mpo[0],lh[0],rh[1])(vector),expected@vector,atol=1e-12)
    from types import SimpleNamespace
    def exhausted(*args):raise MemoryError('factor workspace')
    monkeypatch.setattr(multiplet_sweep,'_su2_kernel_module',lambda:SimpleNamespace(MultipletLocalAction=exhausted))
    np.testing.assert_allclose(solver._local_action(0,solver.mpo[0],lh[0],rh[1])(vector),expected@vector,atol=1e-12)
    from pyqed.mps.nonabelian import environment
    monkeypatch.setattr(environment,'_su2_kernel_module',lambda:None)
    fallback=MultipletMPO.from_components(solver.mpo.component_cores(),[s.leg for s in solver.sites])
    a,b=solver.mpo.component_cores(),fallback.component_cores()
    np.testing.assert_allclose(mpo_overlap(a,a)+mpo_overlap(b,b)-2*mpo_overlap(a,b).real,0,atol=1e-10)


def test_compiled_multiplet_workspace_guards():
    kernel=pytest.importorskip('pyqed.mps.nonabelian._su2_kernel')
    # Small inputs whose factorized workspaces would exceed the cap.
    with pytest.raises(MemoryError,match='factor workspace'):
        kernel.MultipletLocalAction(128,[(0,0,np.ones((128,1,128)),
            np.ones((1,513,1,1)),np.ones((1,513,1)),1)])
    with pytest.raises(MemoryError,match='action workspace'):
        kernel.MultipletLocalAction(16384,[(0,0,np.ones((1,1,128)),
            np.ones((1,513,1,128)),np.ones((2,513,1)),1)])


@pytest.mark.parametrize('group,e,scalar',[('D3h',"E'","A1'"),('D6h','E1u','A1g'),('D6h','E2g','A1g')])
@pytest.mark.parametrize('complex_h',[False,True])
def test_direct_operator_construction_matches_whole_symbolic_mpo(group,e,scalar,complex_h,monkeypatch):
    from pyqed.mps.nonabelian.multiplet_sites import OrbitalMultipletSite
    from pyqed.mps.nonabelian.multiplet_dmrg import _group_mpo
    from importlib import import_module
    sites=[OrbitalMultipletSite(group,label) for label in (e,scalar,e)]
    blocks=((0,1),(2,),(3,4))
    h=np.diag([.1,.1,.4,.7,.7]).astype(complex)
    h[:2,3:]=np.eye(2)*(.3j if complex_h else .3)
    h[3:,:2]=h[:2,3:].conj().T
    eye=np.eye(5)
    v=.2*np.einsum('pq,rs->pqrs',eye,eye)+.07*(np.einsum('pr,qs->pqrs',eye,eye)+np.einsum('ps,qr->pqrs',eye,eye))
    reference=_group_mpo(h,v,sites,blocks,1e-12)
    def forbidden(*args,**kwargs):pytest.fail('Symbolic/component MPO construction entered the direct path.')
    monkeypatch.setattr(import_module('pyqed.qchem.dmrg.dmrg'),'_build_spatial_hamiltonian_tensor_mpo',forbidden)
    monkeypatch.setattr(MultipletMPO,'from_components',forbidden)
    direct=MultipletMPO.from_integrals(h,v,sites,blocks).component_cores()
    # Full operator-space comparison, including all particle/spin sectors.
    error=mpo_overlap(reference,reference)+mpo_overlap(direct,direct)-2*mpo_overlap(reference,direct).real
    np.testing.assert_allclose(error,0,atol=1e-9)


def test_direct_operator_workspace_guard(monkeypatch):
    from pyqed.mps.nonabelian import multiplet_construction
    from pyqed.mps.nonabelian.multiplet_sites import OrbitalMultipletSite
    sites=[OrbitalMultipletSite('D3h',"E'")]*2
    monkeypatch.setattr(multiplet_construction,'_MAX_OPERATOR_ELEMENTS',1)
    with pytest.raises(MemoryError,match='Operator construction workspace'):
        MultipletMPO.from_integrals(np.eye(4),np.zeros((4,)*4),sites,((0,1),(2,3)))


@pytest.mark.parametrize('blocks',[((0,1),(1,2)),((2,3),(0,1)),((0,),(1,2,3))])
def test_direct_operator_rejects_incomplete_site_layout(blocks):
    from pyqed.mps.nonabelian.multiplet_sites import OrbitalMultipletSite
    sites=[OrbitalMultipletSite('D3h',"E'")]*2
    with pytest.raises(ValueError,match='partition'):
        MultipletMPO.from_integrals(np.eye(4),np.zeros((4,)*4),sites,blocks)


@pytest.mark.parametrize('group,e,target', [('D3h',"E'","A1'"),('D6h','E1u','E2g'),('D2h','B1u','Ag')])
def test_diagonal_norm_matches_general_metric(group,e,target,monkeypatch):
    from pyqed.mps.nonabelian.multiplet_sweep import _diagonal_norm
    reps={g:np.kron(np.eye(4),u) for g,u in irrep_representation_matrices(group,e).items()}
    n=len(next(iter(reps.values())))
    solver=MultipletDMRG.from_integrals(np.eye(n),np.zeros((n,)*4),nelec=2,D=3,
        point_group=group,wfnsym=target,orbital_representations=reps)
    _,left=solver._boundaries('rl');_,right=solver._boundaries('lr')
    rng=np.random.default_rng(71)
    checked=False
    for i,fusion in enumerate(solver.fusion_maps):
        def diagonalize(env):
            return {key:np.diag(rng.uniform(.3,2.,size=value.shape[0]))[:,None,:].astype(complex)
                    for key,value in env.items()}
        l,r=diagonalize(left[i]),diagonalize(right[i+1])
        diag=_diagonal_norm(fusion,l,r)
        matrix=solver._local_action(i,solver.identity[i],l,r,assemble=True)
        np.testing.assert_allclose(np.diag(diag),matrix,atol=1e-11)
        for block in l.values():
            if block.shape[0]>1:
                block[0,0,1]=.01j;block[1,0,0]=-.01j
                assert _diagonal_norm(fusion,l,r) is None
                checked=True
                break
    assert checked
    # Force the general whitening route and compare its physical solution.
    fast=MultipletDMRG.from_integrals(np.eye(n),np.zeros((n,)*4),nelec=2,D=3,
        point_group=group,wfnsym=target,orbital_representations=reps).run(nsweeps=4)
    from pyqed.mps.nonabelian import multiplet_sweep
    monkeypatch.setattr(multiplet_sweep,'_diagonal_norm',lambda *args:None)
    solver.run(nsweeps=4)
    np.testing.assert_allclose(solver.e_tot,fast.e_tot,atol=1e-11)


@pytest.mark.parametrize('dimensions',[(8,7,6,5),(16,16,1,8)])
def test_large_local_matmul_routes_own_inputs(dimensions):
    kernel=pytest.importorskip('pyqed.mps.nonabelian._su2_kernel')
    a,b,c,d=dimensions
    rng=np.random.default_rng(38)
    def random(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    l=random((a,3,b));w=random((3,9,2,3));r=random((c,9,d))
    na,nb=a*2*c,b*3*d
    count=max(na+2,nb+5)
    action=kernel.MultipletLocalAction(count,[(2,5,l,w,r,.7)])
    matrix=np.zeros((count,count),complex)
    matrix[2:2+na,5:5+nb]=.7*np.einsum('axb,xypq,cyd->apcbqd',l,w,r,optimize=True).reshape(na,nb)
    v=random((count,))
    np.testing.assert_allclose(action(v),matrix@v,atol=1e-10)
    np.testing.assert_allclose(action.diagonal(),matrix.diagonal(),atol=1e-11)
    l[:]=0;w[:]=0;r[:]=0
    np.testing.assert_allclose(action(v),matrix@v,atol=1e-10)


@pytest.mark.parametrize('group,e,target',[('D3h',"E'","E'"),('D6h','E1u','E2g')])
def test_large_reduced_ground_preserves_E_partners(group,e,target):
    copy_h=np.diag([-.8,-.2,.3,.8])+np.diag(np.full(3,.1),1)+np.diag(np.full(3,.1),-1)
    h=np.kron(copy_h,np.eye(2));v=.13*np.einsum('pq,rs->pqrs',np.eye(8),np.eye(8))
    reps={op:np.kron(np.eye(4),matrix) for op,matrix in irrep_representation_matrices(group,e).items()}
    solver=MultipletDMRG.from_integrals(h,v,nelec=4,D=16,point_group=group,wfnsym=target,
        orbital_representations=reps).run(nsweeps=6)
    levels=np.linalg.eigvalsh(copy_h)
    expected=3*levels[0]+levels[1]+6*.13
    np.testing.assert_allclose(solver.e_tot,[expected,expected],atol=1e-9,rtol=0)
    assert solver.success and len(solver.states)==2
    assert solver.symmetry_diagnostics()['covariance_error']<1e-10
