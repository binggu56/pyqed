"""CBE uses reduced residuals, whole multiplets, and one-site optimization."""
import numpy as np
import pytest
from pyqed.mps.nonabelian.cbe import expand_subspace


@pytest.mark.parametrize('direction',['lr','rl'])
def test_dd_selection_preserves_state_and_ignores_one_site_residual(direction):
    m=np.diag([1.,0.,0.]).astype(complex)
    r=np.array([[0,3j,0],[4,0,0],[0,0,2]],complex)
    bases,added=expand_subspace({'E':m},{'E':r},direction=direction,expansion_size=1)
    b=bases['E'];assert added==1
    np.testing.assert_allclose(b.conj().T@b,np.eye(2),atol=1e-13)
    np.testing.assert_allclose(b@b.conj().T,np.diag([1,0,1]),atol=1e-13)
    np.testing.assert_allclose(m@b@b.conj().T if direction=='lr' else b@b.conj().T@m,m,atol=1e-13)


def test_cbe_hubbard_matches_independent_small_sector_reference(monkeypatch):
    from pyqed.mps.nonabelian import update
    def forbidden(*args,**kwargs):pytest.fail("Two-site eigensolve entered CBE")
    monkeypatch.setattr(update,'solve_local_two_site',forbidden)
    from pyqed.mps.nonabelian import cbe
    monkeypatch.setattr(cbe,'expand_subspace',forbidden)
    from examples.mps.benchmark_cbe_hubbard import calculate
    from examples.mps.hubbard_2d_ed import hubbard_2d_hamiltonian
    from scipy.sparse.linalg import eigsh
    h,_=hubbard_2d_hamiltonian(4,1,nup=2,ndown=2,t=1,u=4)
    exact=eigsh(h,k=1,which='SA',return_eigenvectors=False)[0]
    r=calculate(4,'one_site_cbe',D=16,passes=6,expansion=4)
    assert r['energy']==pytest.approx(exact,abs=1e-9)
    assert r['norm']==pytest.approx(1,abs=1e-10)
    assert all(h['local_converged'] for h in r['history'])
    assert sum(h['added'] for h in r['history'])>0
    assert all(h['one_site_action'] and h['compiled_update'] for h in r['history'])


@pytest.mark.parametrize('group,e,target', [('D3h',"E'","E'"),('D6h','E1u','E2g')])
def test_spatial_cbe_interacting_E_partners(group,e,target,monkeypatch):
    from pyqed.qchem.symmetry import irrep_representation_matrices
    from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG
    from pyqed.mps.nonabelian.fusion import TensorFusionMap
    from pyqed.mps.nonabelian.multiplet_operator import MultipletMPO
    reps={g:np.kron(np.eye(3),u) for g,u in irrep_representation_matrices(group,e).items()}
    h=np.kron([[.1,.3,0],[.3,.7,.2],[0,.2,.5]],np.eye(2))
    # Repulsion depends on complete E-site occupations, preserving the group
    # while producing a correlated state. Only the small test uses FCI.
    eri=.2*np.einsum('pq,rs->pqrs',np.eye(6),np.eye(6))
    for i in range(3):
        number=np.diag([float(j//2==i) for j in range(6)])
        eri+=.8*np.einsum('pq,rs->pqrs',number,number)
    fci=pytest.importorskip('pyscf.fci')
    expected=fci.direct_spin1.kernel(h,eri,6,(1,1),tol=1e-12)[0]
    solver=MultipletDMRG.from_integrals(h,eri,nelec=2,spin=0,D=1,
        orbital_representations=reps,wfnsym=target,point_group=group)
    solver.D=3
    def forbidden(*args,**kwargs):pytest.fail('Partner components entered the active solver')
    with monkeypatch.context() as patch:
        patch.setattr(TensorFusionMap,'expand',forbidden)
        patch.setattr(TensorFusionMap,'matrix',property(forbidden))
        patch.setattr(MultipletMPO,'component_cores',forbidden)
        from pyqed.mps.nonabelian import cbe
        patch.setattr(cbe,'multiplet_pair_action',forbidden)
        patch.setattr(cbe,'expand_subspace',forbidden)
        solver.run(algorithm='one_site_cbe',cbe_expansion=3,nsweeps=6)
    np.testing.assert_allclose(solver.e_tot,expected,atol=2e-9)
    assert solver.diagnostics['cbe_compiled_update']
    assert len(solver.states)==2
    diagnostics=solver.symmetry_diagnostics()
    assert diagnostics['covariance_error']<1e-10
    assert diagnostics['orthogonality_error']<1e-10
    np.testing.assert_allclose(diagnostics['partner_energies_hartree'],solver.e_tot,atol=1e-10)
    assert max(max(leg.dims.values()) for leg in solver.bond_legs[1:-1])<=solver.D


@pytest.mark.parametrize('group,e',[('D3h',"E'"),('D6h','E1u')])
def test_spatial_pair_action_and_expansion_preserve_complex_state(group,e):
    from pyqed.qchem.symmetry import irrep_representation_matrices
    from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG
    from pyqed.mps.nonabelian.cbe import multiplet_pair,multiplet_pair_action,expand_multiplet_bond
    reps={g:np.kron(np.eye(2),u) for g,u in irrep_representation_matrices(group,e).items()}
    h=np.kron([[.1,.3],[.3,.7]],np.eye(2))
    solver=MultipletDMRG.from_integrals(h,np.zeros((4,)*4),nelec=1,spin=1,D=2,
        orbital_representations=reps,wfnsym=e,point_group=group)
    rng=np.random.default_rng(3)
    solver.vectors[0]=solver.vectors[0]+1j*rng.normal(size=solver.vectors[0].size)
    solver._refresh()
    left,nl=solver._edge();right,nr=solver._edge(True)
    matrices,layout,lm,rm=multiplet_pair(solver,0)
    applied=multiplet_pair_action(solver,0,matrices,layout,lm,rm,left,right)
    from pyqed.mps.nonabelian.cbe_selection import multiplet_halves
    halves=multiplet_halves(solver,0,matrices,layout,lm,rm,left,right)
    factorized={q:np.zeros_like(m) for q,m in matrices.items()}
    for (a,b,_),(l,r) in halves.items():
        factorized[a]+=l@matrices[b]@r.T
    for q in matrices:
        np.testing.assert_allclose(factorized[q],applied[q],atol=1e-11)
    energy=sum(np.vdot(m,applied[q]) for q,m in matrices.items())
    assert energy==pytest.approx(solver.energy,abs=1e-11)
    expand_multiplet_bond(solver,0,'lr',left,right,nl,nr,2)
    after,*_=multiplet_pair(solver,0)
    for q in matrices:np.testing.assert_allclose(after[q],matrices[q],atol=1e-12)


def test_cbe_preserves_abelian_spatial_sector():
    from pyqed.mps.su2 import SpinChargeSector,SU2Irrep,fuse_charge_spin_sectors
    from pyqed.mps.nonabelian.states import build_random_reduced_spatial_mps
    from pyqed.mps.nonabelian._su2_kernel import SU2MovingEnvironment
    from pyqed.qchem.dmrg.backends.reduced import build_su2_normal_complementary_mpo
    from pyqed.mps.nonabelian.sweep import run_sweeps
    from examples.mps.hubbard_2d_dmrg import normalized_energy
    from examples.mps.hubbard_2d_ed import hubbard_2d_hamiltonian
    from scipy.sparse.linalg import eigsh
    from pyqed import num_workers
    target=SpinChargeSector(3,SU2Irrep(1),point_group=1)
    sites=build_random_reduced_spatial_mps(4,target_sector=target,orbsym=[1]*4,
        seed=7,bond_multiplicity=1,cap_multiplicity=True)
    h=-np.diag(np.ones(3),1)-np.diag(np.ones(3),-1)
    v=np.zeros((4,)*4);i=np.arange(4);v[i,i,i,i]=4
    owner=SU2MovingEnvironment(h,v,3,two_s=1,orb_sym=[1]*4,include_half=True)
    owner.set_num_threads(num_workers())
    mpo=build_su2_normal_complementary_mpo(owner,fully_reduced=True,
        physical_legs=[site.legs[1] for site in sites])
    for core in mpo:object.__setattr__(core,'normal_complementary_right_dual',True)
    result=run_sweeps(sites,mpo_factors=mpo,algorithm='one_site_cbe',cbe_expansion=4,
        max_bond=24,max_bond_mode='reduced',nsweeps=6,su2_moving_environment=owner,
        local_solver_kwargs={'tol':1e-11,'tol_residual':1e-9})
    reference,_=hubbard_2d_hamiltonian(4,1,nup=2,ndown=1,t=1,u=4)
    exact=eigsh(reference,k=1,which='SA',return_eigenvectors=False)[0]
    assert normalized_energy(result['sites'],mpo)[0]==pytest.approx(exact,abs=1e-9)
    assert result['sites'][-1].qns[-1]==[target]
    for tensor in result['sites']:
        for a,p,b in tensor.data:assert b in fuse_charge_spin_sectors(a,p)


@pytest.mark.parametrize("verify", [False, True])
def test_qchem_driver_exposes_cbe_without_noise_mixer(verify):
    from pyqed.qchem import Molecule,RHF
    from pyqed.qchem.dmrg.dmrg import DMRG
    mol=Molecule(atom='H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2',
                 unit='bohr',basis='sto-3g').build(eri='s8')
    mf=RHF(mol);mf.chkfile=None;mf.run(verbose=0,tol=1e-11)
    energies=[]
    for algorithm in ('two_site','one_site_cbe'):
        driver=DMRG(mf,4,4,D=32,spin=0,symmetry='su2',verbose=0)
        driver.run(algorithm=algorithm,nsweeps=8,cbe_expansion=4,
                   verify_returned_mps_energy=verify)
        energies.append(float(np.asarray(driver.e_tot).ravel()[0]))
        assert driver.success
    assert energies[0]==pytest.approx(energies[1],abs=1e-8)


def test_empty_reduced_sector_uses_leg_multiplicity_not_spin_degeneracy():
    from pyqed.symmetry import Leg, IrrepTensor
    from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
    from pyqed.mps.nonabelian.update import _sector_dim_from_data
    q = SpinChargeSector(10, SU2Irrep(4))
    empty = SpinChargeSector(0, SU2Irrep(0))
    tensor = IrrepTensor({}, [Leg({q: 2}), Leg({empty: 1}), Leg({q: 2})],
                         (1, 1, -1), metadata={'physical_basis': 'fully_reduced_su2'})
    assert _sector_dim_from_data(tensor, 0) == {q: 2}
    assert _sector_dim_from_data(tensor, 2) == {q: 2}


@pytest.mark.parametrize('options,match', [
    ({'cbe_expansion': 0}, 'positive'),
    ({'local_solver_kwargs': {'nstates': 2}}, 'single-root'),
    ({'mixer_zero_block_noise_scale': 1e-5}, 'noise mixer'),
])
def test_cbe_rejects_unsupported_updates_before_sweeping(options, match):
    from pyqed.mps.nonabelian.sweep import sweep_once
    with pytest.raises(ValueError, match=match):
        sweep_once([], algorithm='one_site_cbe', **options)


def test_cbe_workspace_guard_precedes_pair_allocation():
    from types import SimpleNamespace
    from pyqed.mps.nonabelian.cbe import solve_cbe
    operator = SimpleNamespace(basis=SimpleNamespace(size=3_000_000))
    with pytest.raises(MemoryError, match='workspace'):
        solve_cbe(None, operator, direction='lr', expansion_size=8)


@pytest.mark.parametrize('direction', ['lr', 'rl'])
def test_shrewd_untruncated_preselection_matches_exact_DD(direction):
    from pyqed.mps.nonabelian.cbe_selection import shrewd_subspace
    rng = np.random.default_rng(104)
    def random(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    factors = {q: (random((n, 2)), random((2, m)))
               for q, n, m in [('A', 7, 6), ('E', 5, 8)]}
    matrices = {q: l@r for q,(l,r) in factors.items()}
    halves = {(a, b, w): (random((ma.shape[0], mb.shape[0])),
                           random((ma.shape[1], mb.shape[1])))
              for a, ma in matrices.items() for b, mb in matrices.items() for w in range(3)}
    residual = {q: np.zeros_like(m) for q, m in matrices.items()}
    for (a, b, _), (l, r) in halves.items():
        residual[a] += l@matrices[b]@r.T
    exact, _ = expand_subspace(matrices, residual, direction=direction, expansion_size=3)
    selected, added, info = shrewd_subspace(matrices, halves, direction=direction,
        expansion_size=3, preselection_size=100, state_factors=factors)
    assert added == 3
    for q, b in selected.items():
        np.testing.assert_allclose(b@b.conj().T, exact[q]@exact[q].conj().T, atol=1e-10)
    selected, added, info = shrewd_subspace(matrices, halves, direction=direction,
        expansion_size=3, preselection_size=1, state_factors=factors)
    assert info['cbe_preselection_budget'] == 1
    for q, b in selected.items():
        np.testing.assert_allclose(b.conj().T@b, np.eye(b.shape[1]), atol=1e-12)
        m = matrices[q]
        np.testing.assert_allclose(b@b.conj().T@m if direction=='rl' else m@b@b.conj().T,
                                   m, atol=1e-11)


def test_shrewd_packed_halves_reproduce_reduced_operator():
    from examples.mps.hubbard_2d_dmrg import build_2d_hubbard_mpo
    from pyqed.mps.nonabelian.states import build_random_reduced_spatial_mps
    from pyqed.mps.nonabelian.canonical import mixed_canonicalize_sites
    from pyqed.mps.nonabelian.contraction import merge_mps_sites
    from pyqed.mps.nonabelian.update import _expand_two_site_support
    from pyqed.mps.nonabelian.environment import BlockSparseEnvironmentChain
    from pyqed.mps.nonabelian.cbe import _sector_indices
    from pyqed.mps.nonabelian.cbe_selection import packed_halves
    sites = build_random_reduced_spatial_mps(6, seed=7, bond_multiplicity=2, cap_multiplicity=True)
    mpo = build_2d_hubbard_mpo(sites, 6, 1, hopping_t=1, onsite_u=4)
    sites = mixed_canonicalize_sites(sites, 2, cutoff=0)
    merged = _expand_two_site_support(sites[2], sites[3], merge_mps_sites(sites[2], sites[3]))
    op = BlockSparseEnvironmentChain.build(sites, mpo).bond_operator(2, merged)
    indices = _sector_indices(op.basis)
    halves = packed_halves(op, indices)
    rng = np.random.default_rng(56)
    x = rng.normal(size=op.basis.size)+1j*rng.normal(size=op.basis.size)
    result = np.zeros_like(x)
    for (a, b, _), (l, r) in halves.items():
        result[indices[a]] += l@x[indices[b]]@r.T
    np.testing.assert_allclose(result, (op.packed_matvec or op.aux_packed_matvec)(x), atol=1e-10)


def test_shrewd_canonicalization_uses_only_bond_svd(monkeypatch):
    from pyqed.mps.nonabelian import cbe_selection
    rng=np.random.default_rng(67)
    left=rng.normal(size=(41,4))+1j*rng.normal(size=(41,4))
    right=rng.normal(size=(4,37))+1j*rng.normal(size=(4,37))
    original=cbe_selection._svd
    def small(matrix,cutoff):
        assert matrix.shape==(4,4)
        return original(matrix,cutoff)
    monkeypatch.setattr(cbe_selection,'_svd',small)
    u,s,vh=cbe_selection.canonical_factors(left,right)
    np.testing.assert_allclose((u*s)@vh,left@right,atol=1e-12)
    np.testing.assert_allclose(u.conj().T@u,np.eye(4),atol=1e-12)
    np.testing.assert_allclose(vh@vh.conj().T,np.eye(4),atol=1e-12)


@pytest.mark.parametrize('direction', ['lr', 'rl'])
def test_one_site_action_matches_projected_pair_reference(direction):
    from pyqed.mps.nonabelian.cbe_selection import one_site_action
    rng = np.random.default_rng(314)
    def random(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    shapes = {0: (7, 6), 1: (5, 8)}
    bases, sections, offset = {}, {}, 0
    for q, (m, n) in shapes.items():
        bases[q] = np.linalg.qr(random((n if direction == 'lr' else m, 3)))[0]
        shape = (m, 3) if direction == 'lr' else (3, n)
        sections[q] = slice(offset, offset+np.prod(shape)), shape
        offset += np.prod(shape)
    halves = {(oq, iq, c): (random((om, im)), random((on, inn)))
              for oq, (om, on) in shapes.items()
              for iq, (im, inn) in shapes.items() for c in range(2)}
    action, diagonal = one_site_action(halves, bases, sections, direction)
    def reference(v):
        pairs, out = {}, {}
        for q, (span, shape) in sections.items():
            site = v[span].reshape(shape)
            pairs[q] = site @ bases[q].conj().T if direction == 'lr' else bases[q] @ site
            out[q] = np.zeros(shapes[q], complex)
        for (oq, iq, _), (left, right) in halves.items():
            out[oq] += left @ pairs[iq] @ right.T
        return np.concatenate([(out[q] @ bases[q] if direction == 'lr'
                                else bases[q].conj().T @ out[q]).ravel() for q in sections])
    v = random((offset,))
    np.testing.assert_allclose(action.matvec(v), reference(v), atol=1e-11)
    exact_diagonal = np.array([reference(np.eye(offset)[j])[j] for j in range(offset)])
    np.testing.assert_allclose(diagonal, exact_diagonal.real, atol=1e-11)


@pytest.mark.parametrize('mode', ['reduced', 'states', 'per_sector'])
def test_compiled_factor_update_matches_pair_svd(mode):
    from pyqed.mps.nonabelian._su2_kernel import SU2MovingEnvironment
    owner = SU2MovingEnvironment(np.zeros((2, 2)), np.zeros((2, 2, 2, 2)), 2)
    rng = np.random.default_rng(172)
    def random(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)
    left = [random((9, 3)), random((4, 7)), random((6, 2))]
    right = [random((3, 8)), random((7, 5)), random((2, 4))]
    left[2][:, 1] = 2*left[2][:, 0]  # rank-deficient site
    weights = [1, 2, 3]
    args = dict(cutoff=1e-12, max_bond=4, max_bond_mode=mode, retain_sector_topology=True)
    actual = owner.blockwise_svd(left, weights, right_factors=right, **args)
    reference = owner.blockwise_svd([a@b for a, b in zip(left, right)], weights, **args)
    for i in range(3):
        def retained(result):
            keep = result['kept_indices'][i]
            return ((result['left'][i][:, keep]*result['singular_values'][i][keep])
                    @ result['right'][i][keep])
        np.testing.assert_allclose(retained(actual), retained(reference), atol=2e-12)
    assert actual['truncation_error'] == pytest.approx(reference['truncation_error'], abs=1e-13)
    assert actual['full_squared_norm'] == pytest.approx(reference['full_squared_norm'], rel=1e-13)
    with pytest.raises(ValueError, match='dimensions'):
        owner.blockwise_svd(left, weights, right_factors=[right[0][:-1], *right[1:]])


def test_cbe_compiled_update_handles_empty_selected_sector():
    from examples.mps.benchmark_cbe_hubbard import calculate
    result = calculate(8, 'one_site_cbe', D=32, passes=4, expansion=8)
    assert result['norm'] == pytest.approx(1., abs=1e-10)
    assert result['energy'] < -4.23
    assert all(h['one_site_action'] and h['compiled_update'] for h in result['history'])


@pytest.mark.parametrize('left', [True, False])
@pytest.mark.parametrize('dual', [True, False])
def test_environment_recoupling_matches_python_reference(left, dual):
    from pyqed.mps.nonabelian._su2_kernel import SU2MovingEnvironment
    from pyqed.qchem.dmrg.backends.reduced import build_su2_normal_complementary_mpo
    from pyqed.mps.nonabelian import environment as env
    from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
    rng = np.random.default_rng(321)
    h = rng.normal(size=(4, 4)); h += h.T.copy()
    factors = rng.normal(size=(4, 4, 3))
    factors += factors.swapaxes(0, 1).copy()
    eri = np.einsum('pqL,rsL->pqrs', factors, factors)
    owner = SU2MovingEnvironment(h, eri, 4, two_s=0, cutoff=1e-12, include_half=True)
    mpo = build_su2_normal_complementary_mpo(owner, fully_reduced=True)
    builder = env._left_reduced_rank_coupled_block if left else env._right_reduced_rank_coupled_block
    nonzero = 0
    for _ in range(32):
        core = mpo[int(rng.integers(4))]
        object.__setattr__(core, 'normal_complementary_right_dual', dual)
        physical = tuple(core.phys_in_leg.sectors)
        pb, pk = (physical[int(rng.integers(3))] for _ in range(2))
        lb, lk = (SpinChargeSector(0, SU2Irrep(int(rng.integers(3)))) for _ in range(2))
        rb = SpinChargeSector(0, SU2Irrep(abs(lb.irrep.two_j-pb.irrep.two_j)))
        rk = SpinChargeSector(0, SU2Irrep(abs(lk.irrep.two_j-pk.irrep.two_j)))
        sectors = lb, lk, pb, pk, rb, rk
        core._environment_reduced_block_cache.clear()
        object.__setattr__(core, 'normal_complementary_contextual_python_reference', False)
        actual = builder(core, *sectors)
        core._environment_reduced_block_cache.clear()
        object.__setattr__(core, 'normal_complementary_contextual_python_reference', True)
        expected = builder(core, *sectors)
        for key in actual.keys() | expected.keys():
            np.testing.assert_allclose(actual.get(key, 0), expected.get(key, 0), atol=1e-11)
        nonzero += bool(actual)
    assert nonzero > 0
