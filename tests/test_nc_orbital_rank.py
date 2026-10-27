"""Compiled whole-orbital rank algebra against independent Python fusion."""
import pytest
from pyqed.symmetry import OrbitalRotationSymmetry
from pyqed.mps.nonabelian import _su2_kernel as kernel


@pytest.mark.parametrize('code,group,labels',[
    (1,'Coov',('A1','A2','E1','E2','E9')),
    (2,'Dooh',('A1g','A2g','A1u','A2u','E1g','E1u','E8u')),
    (3,'SO3',('L0','L1','L2','L4','L12'))])
def test_compiled_orbital_fusion_keeps_complete_partners(code,group,labels):
    sym = OrbitalRotationSymmetry(group)
    for a in labels:
        aq = sym.quantum_numbers(a)
        assert kernel.orbital_rank_dimension(code,aq) == sym.dimension(a)
        for b in labels:
            actual = kernel.orbital_rank_fusion(code,aq,sym.quantum_numbers(b))
            expected = [sym.quantum_numbers(c) for c in sym.fuse(a,b)]
            assert actual == expected
            assert sum(kernel.orbital_rank_dimension(code,c) for c in actual) == sym.dimension(a)*sym.dimension(b)


@pytest.mark.parametrize('group,rank',[(1,(-1,1,1)),(1,(1,-1,1)),
    (1,(1,1,-1)),(3,(0,-1,1)),(2,(0,0,1)),(4,(0,1,1))])
def test_invalid_orbital_ranks_raise(group,rank):
    with pytest.raises(ValueError):
        kernel.orbital_rank_dimension(group,rank)


def test_orbital_rank_has_no_magnetic_or_angular_cutoff():
    assert kernel.orbital_rank_dimension(3,(100000,1,1)) == 200001
    assert kernel.orbital_rank_fusion(2,(100000,1,-1),(1,1,1)) == [
        (99999,1,-1),(100001,1,-1)]
    with pytest.raises(OverflowError):
        kernel.orbital_rank_fusion(3,(2147483647,1,1),(1,1,1))


@pytest.mark.parametrize('offsets',[(0,1,2),(0,2),(0,1,2,3),(0,1,3),(0,3)])
def test_compiled_nc_shell_against_fermionic_reference(offsets):
    import numpy as np
    from pyqed.qchem.dmrg.spatial_terms import (dense_from_spatial_term_map,
        merge_term_maps,spatial_one_body_term_map,spatial_two_body_spinfree_term_map)
    norb = offsets[-1]
    rng = np.random.default_rng(922+norb)
    h = rng.normal(size=(norb,norb))
    h = (h+h.T)/2
    factors = rng.normal(size=(3,norb,norb))
    factors = (factors+factors.transpose(0,2,1))/2
    v = np.einsum('apq,ars->pqrs',factors,factors)
    core = .29
    owner = kernel.SU2MovingEnvironment(h,v,2,ecore=core,cutoff=1e-14)
    owner.configure_spatial_shells(offsets)
    schedule = owner.spatial_normal_complementary_schedule()
    current = {1:np.ones((1,1))}
    for site in schedule['sites']:
        following = {}
        for source,target,primitive,coefficient,_ in site['transitions']:
            if source in current:
                block = coefficient*np.kron(current[source],site['primitives'][primitive])
                following[target] = following.get(target,0)+block
        current = following
    terms = merge_term_maps(spatial_one_body_term_map(h),spatial_two_body_spinfree_term_map(v))
    expected = dense_from_spatial_term_map(terms,norb)+core*np.eye(4**norb)
    np.testing.assert_allclose(current[0],expected,atol=1e-11,rtol=0)


def test_compiled_nc_shell_rejects_invalid_intervals_and_stale_integrals():
    import numpy as np
    owner = kernel.SU2MovingEnvironment(np.eye(2),np.zeros((2,)*4),2)
    for offsets in [(-1,2),(0,0,2),(0,3),(1,2)]:
        with pytest.raises(ValueError):
            owner.configure_spatial_shells(offsets)
    owner.configure_spatial_shells([0,1,2])
    owner.update_integrals(np.eye(2)*2,np.zeros((2,)*4),0.)
    with pytest.raises(RuntimeError,match='stale'):
        owner.spatial_normal_complementary_schedule()


@pytest.mark.parametrize('group,label',[('Coov','E1'),('Dooh','E1u'),('SO3','L1'),
    ('D3h',"E'"),('D3h',"E''"),('D6h','E1u'),('D6h','E2g'),('D6h','E2u')])
def test_compiled_nc_rank_reduction_reconstructs_all_local_partners(group,label,monkeypatch):
    import numpy as np
    from pyqed.mps.nonabelian.multiplet_sites import OrbitalMultipletSite
    from pyqed.mps.nonabelian.multiplet_operator import MultipletMPO,dimension
    from pyqed.mps.nonabelian import multiplet_construction
    from pyqed.mps.nonabelian.nc_spatial import channel_basis,prepare_rank_schedule
    from pyqed.mps.nonabelian.fusion import full_slices,irrep_coupling
    def forbidden(*args,**kwargs):
        pytest.fail('An MPO compiler entered direct NC rank preparation.')
    for name in ('from_integrals','from_components','_from_operator_train'):
        monkeypatch.setattr(MultipletMPO,name,forbidden)
    monkeypatch.setattr(multiplet_construction,'hamiltonian_operator_train',forbidden)
    sites = [OrbitalMultipletSite(group,label) for _ in range(2)]
    symmetry = sites[0].symmetry
    d = sites[0].norb
    n = 2*d
    offsets = [0,d,n]
    h = np.kron([[.1,.2],[.2,.5]],np.eye(d))
    v = (.2*np.einsum('pq,rs->pqrs',np.eye(n),np.eye(n))
         +.1*np.einsum('ps,qr->pqrs',np.eye(n),np.eye(n)))
    owner = kernel.SU2MovingEnvironment(h,v,2)
    owner.configure_spatial_shells(offsets)
    raw = owner.spatial_normal_complementary_schedule()
    ranks,legs = prepare_rank_schedule(owner,sites,offsets)
    transforms = []
    for descriptors,leg in zip(raw['channels'],legs):
        _,expansion = channel_basis(symmetry,descriptors,sites,offsets)
        spans = full_slices(leg)
        U = np.zeros((len(descriptors),leg.full_dim))
        for row,entries in enumerate(expansion):
            for q,copy,partner,value in entries:
                U[row,spans[q].start+copy*dimension(symmetry,q)+partner] = value
        transforms.append(U)
    for i,site in enumerate(sites):
        plan = raw['sites'][i]
        reference = np.zeros((len(transforms[i]),len(transforms[i+1]),site.leg.full_dim,site.leg.full_dim))
        for a,b,primitive,coefficient,_ in plan['transitions']:
            reference[a,b] += coefficient*plan['primitives'][primitive]
        reference = np.einsum('xa,xyij,yb,ip,jq->abpq',transforms[i],reference,
            transforms[i+1],site.basis.conj(),site.basis,optimize=True)
        reconstructed = np.zeros_like(reference)
        ls,rs,ps = map(full_slices,(legs[i],legs[i+1],site.leg))
        labels,values,_ = owner.spatial_nc_rank_plan(i)
        for row,value in zip(labels,values):
            a,b,p,q,k = [ranks[index] for index in row[:5]]
            ca,cb,cp,cq = row[5:]
            dims = [dimension(symmetry,x) for x in (a,b,p,q)]
            spans = [slice(table[x].start+copy*dim,table[x].start+(copy+1)*dim)
                     for table,x,copy,dim in zip((ls,rs,ps,ps),(a,b,p,q),(ca,cb,cp,cq),dims)]
            block = value*np.einsum('lkr,qkp->lrpq',
                irrep_coupling(symmetry,a.charge,k.charge,b.charge),
                irrep_coupling(symmetry,q.charge,k.charge,p.charge))
            reconstructed[tuple(spans)] += block
        np.testing.assert_allclose(reconstructed,reference,atol=1e-11,rtol=0)


@pytest.mark.parametrize('group,orbital,target',[('Coov','E1','E1'),
    ('Dooh','E1u','E1u'),('SO3','L1','L1')])
@pytest.mark.parametrize("shells,nelec", [(2,2),(3,3)])
def test_direct_compiled_nc_environments_and_local_actions_match_mpo(group,orbital,target,shells,nelec,monkeypatch):
    import numpy as np
    from scipy.linalg import block_diag
    from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG
    from pyqed.mps.nonabelian.multiplet_operator import MultipletMPO
    from pyqed.mps.nonabelian.nc_spatial_environment import SpatialNCEnvironment
    sym = OrbitalRotationSymmetry(group)
    identity = sym.identity
    reps = {name:block_diag(value,sym.generators(orbital)[name],*([value] if shells==3 else []))
            for name,value in sym.generators(identity).items()}
    n = shells-1+sym.dimension(orbital)
    h = np.diag([.1]+[.3]*sym.dimension(orbital)+([.7] if shells==3 else []))
    if shells==3:
        h[0,-1] = h[-1,0] = .13
    v = (.2*np.einsum('pq,rs->pqrs',np.eye(n),np.eye(n))
         +.1*np.einsum('ps,qr->pqrs',np.eye(n),np.eye(n)))
    ref = MultipletDMRG.from_integrals(h,v,nelec=nelec,spin=nelec%2,D=4,point_group=group,
        orbital_representations=reps,wfnsym=target,seed=41,contractions='reduced')
    rng = np.random.default_rng(173)
    for i,vector in enumerate(ref.vectors):
        ref.vectors[i] = vector+.2j*rng.normal(size=len(vector))
    ref._refresh()
    def forbidden(*args,**kwargs):
        pytest.fail('An MPO compiler entered direct NC sweep execution.')
    for name in ('from_integrals','from_components','_from_operator_train'):
        monkeypatch.setattr(MultipletMPO,name,forbidden)
    u = ref.orbital_transform
    ht = u.conj().T@h@u
    vt = np.einsum('pa,qb,rc,sd,pqrs->abcd',u.conj(),u,u.conj(),u,v,optimize=True)
    env = SpatialNCEnvironment(ht,vt,ref.sites,ref.orbital_blocks,
        [leg for fusion in ref.fusion_maps for leg in fusion.legs],nelec=nelec,spin=nelec%2,cutoff=1e-12,n_threads=1)
    data = [fusion.unpack(vector) for fusion,vector in zip(ref.fusion_maps,ref.vectors)]
    left = [env.edge(ref.fusion_maps[0].legs[0])]
    for i in range(ref.nsites):
        left.append(env.advance(i,left[-1],data[i],data[i],'lr'))
    right = [None]*ref.nsites+[env.edge(ref.fusion_maps[-1].legs[-1],right=True)]
    for i in reversed(range(ref.nsites)):
        right[i] = env.advance(i,right[i+1],data[i],data[i],'rl')
    np.testing.assert_allclose(next(iter(left[-1].values()))[0,0,0],ref.energy,atol=1e-11)
    np.testing.assert_allclose(next(iter(right[0].values()))[0,1,0],ref.energy,atol=1e-11)
    mpo_left,_ = ref._boundaries('rl')
    mpo_right,_ = ref._boundaries('lr')
    for i in range(ref.nsites):
        action = env.local_action(i,ref.fusion_maps[i],left[i],right[i+1])
        expected = ref._local_action(i,ref.mpo[i],mpo_left[i],mpo_right[i+1],assemble=True)
        actual = action.assemble()
        np.testing.assert_allclose(actual,expected,atol=1e-11,rtol=0)
        np.testing.assert_allclose(action.diagonal(),expected.diagonal(),atol=1e-11,rtol=0)
        probe = rng.normal(size=len(expected))+1j*rng.normal(size=len(expected))
        np.testing.assert_allclose(action(probe),expected@probe,atol=1e-11,rtol=0)
