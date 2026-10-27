import numpy as np
import pytest
from pyqed.symmetry import OrbitalRotationSymmetry, Irrep
from pyqed.mps.nonabelian.fusion import irrep_coupling
from pyqed.mps.nonabelian.spatial_complementary import grow_operator


@pytest.mark.parametrize('group,orbital,operator',[('Coov','E1','E2'),
    ('Dooh','E1u','E2g'),('SO3','L1','L1')])
def test_growth_against_explicit_local_cg_contraction(group,orbital,operator):
    sym=OrbitalRotationSymmetry(group)
    a=Irrep(orbital)
    op=irrep_coupling(sym,orbital,operator,orbital)
    tested=0
    for c in sym.fuse(orbital,orbital):
        bra=irrep_coupling(sym,orbital,orbital,c)
        for d in sym.fuse(orbital,orbital):
            ket=irrep_coupling(sym,orbital,orbital,d)
            for r in sym.fuse(operator,operator):
                if c not in sym.fuse(d,r): continue
                coupling=irrep_coupling(sym,operator,operator,r)
                component=np.einsum('apc,bla,qkp,bqd,lkr->drc',bra.conj(),op,op,ket,coupling)
                expected=np.vdot(irrep_coupling(sym,d,r,c),component)/sym.dimension(c)
                actual=grow_operator(sym,{(a,a):np.ones((1,1))},{(a,a):np.ones((1,1))},
                    {(a,a,Irrep(c)):np.ones((1,1,1))},
                    {(a,a,Irrep(d)):np.ones((1,1,1))},
                    left_rank=operator,local_rank=operator,rank=r)
                value=actual.get((Irrep(c),Irrep(d)),np.zeros((1,1))).item()
                np.testing.assert_allclose(value,expected,atol=1e-12)
                tested+=1
    assert tested>1


@pytest.mark.parametrize('group,orbital,operator',[('Coov','E1','E2'),
    ('Dooh','E1u','E2g'),('SO3','L1','L1')])
@pytest.mark.parametrize("product", [False, True])
def test_family_environments_and_local_action_match_mpo(group,orbital,operator,product):
    from pyqed.mps.nonabelian.multiplet_operator import advance, routes
    from pyqed.mps.nonabelian.spatial_complementary import update_families, local_action
    sym = OrbitalRotationSymmetry(group)
    if product:
        from pyqed.symmetry import ProductSymmetry, U1Symmetry, SU2Symmetry
        sym = ProductSymmetry((U1Symmetry(),SU2Symmetry(),sym))
        orbital,operator = (1,1,orbital),(0,0,operator)
    rng = np.random.default_rng(451)
    def random(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    a,k = Irrep(orbital),Irrep(operator)
    data = {(a,a,Irrep(c)):random((2,3,2)) for c in sym.fuse(orbital,orbital)}
    for rank in sym.fuse(operator,operator):
        r = Irrep(rank)
        core = {(k,r,a,a,k):random((1,1,3,3))}
        local = {'B':{(a,a):core[k,r,a,a,k][0,0]}}
        left = {'A':{(a,a):random((2,2))}}
        right = {'Q':{(Irrep(c),Irrep(d)):random((2,2))
                 for c in sym.fuse(orbital,orbital)
                 for d in sym.fuse(orbital,orbital) if c in sym.fuse(d,rank)}}
        transitions = [('A','B','Q',operator,operator,rank,.7),
                       ('A','B','Q',operator,operator,rank,.3)]
        le = {(x,k,y):v[:,None,:] for (x,y),v in left['A'].items()}
        re = {(x,r,y):v[:,None,:] for (x,y),v in right['Q'].items()}
        for direction,env,families,target in [('lr',le,left,'Q'),('rl',re,right,'A')]:
            expected = advance(sym,env,core,data,data,direction)
            actual = update_families(sym,families,local,data,data,transitions,direction=direction)[target]
            assert set(actual) == {(x,y) for x,_,y in expected}
            for (x,_,y),value in expected.items():
                np.testing.assert_allclose(actual[x,y],value[:,0,:],atol=1e-11)
        actual = local_action(sym,left,local,right,data,tuple(data),transitions)
        expected = {}
        for ak,bk,wk,lk,rk,weight in routes(sym,tuple(data),tuple(data),tuple(core)):
            if lk not in le or rk not in re: continue
            value = weight*np.einsum('axb,xypq,cyd,bqd->apc',le[lk],core[wk],re[rk],data[bk])
            expected[ak] = expected.get(ak,0)+value
        assert set(actual) == set(expected)
        for key,value in expected.items():
            np.testing.assert_allclose(actual[key],value,atol=1e-11)
