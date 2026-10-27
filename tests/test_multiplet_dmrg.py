"""Spatial Legs, CG fusion, complete sites, and genuine reduced MPS sweeps."""
import numpy as np
import pytest
from scipy.linalg import block_diag
from pyqed.symmetry import Irrep, Leg, IrrepTensor, PointGroupSymmetry
from pyqed.qchem.symmetry import irrep_representation_matrices
from pyqed.mps.nonabelian.fusion import irrep_coupling, fusion_recoupling, full_slices
from pyqed.mps.nonabelian.multiplet_sites import OrbitalMultipletSite, molecular_symmetry
from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG


@pytest.mark.parametrize('group', ['D3h','D6h'])
def test_spatial_leg_fusion_and_all_CG_channels(group):
    symmetry = PointGroupSymmetry(group)
    representations = {q:irrep_representation_matrices(group,q) for q in symmetry.group.irrep_names}
    for a in representations:
        for b in representations:
            left = Leg({Irrep(a):2},symmetry=symmetry)
            right = Leg({Irrep(b):3},symmetry=symmetry)
            fused = left.fuse(right)
            assert fused.full_dim == left.full_dim*right.full_dim
            channels = []
            for c in symmetry.fuse(a,b):
                cg = irrep_coupling(symmetry,a,b,c)
                flat = cg.reshape(-1,cg.shape[-1])
                channels.append(flat)
                for op in symmetry.group.operation_names:
                    np.testing.assert_allclose(np.kron(representations[a][op],representations[b][op]) @ flat,
                        flat @ representations[c][op],atol=1e-12)
            unitary = np.column_stack(channels)
            np.testing.assert_allclose(unitary.conj().T @ unitary,np.eye(len(unitary)),atol=1e-12)
    e = "E'" if group=='D3h' else 'E1u'
    leg = Leg({Irrep(e):2},symmetry=symmetry)
    tensor = IrrepTensor({(Irrep(e),):np.ones(2)},[leg],[1])
    assert tensor.has_nonabelian_symmetry
    assert leg.dim==2 and leg.full_dim==4


@pytest.mark.parametrize('group,e', [('D3h',"E'"),('D6h','E1u')])
def test_product_recoupling_preserves_all_channels(group,e):
    symmetry = molecular_symmetry(group)
    a = (1,1,e)
    left,right,f = fusion_recoupling(symmetry,a,a,a,(3,1,e))
    assert len(left)==len(right)>1
    np.testing.assert_allclose(f.conj().T @ f,np.eye(len(left)),atol=1e-12)


@pytest.mark.parametrize('group,e', [('D3h',"E'"),('D3h',"E''"),
    ('D6h','E1g'),('D6h','E2g'),('D6h','E1u'),('D6h','E2u')])
def test_complete_E_sites_and_repeated_MPS_multiplets(group,e,monkeypatch):
    site = OrbitalMultipletSite(group,e)
    assert site.leg.dim==7 and site.leg.full_dim==16
    slices = full_slices(site.leg)
    for op in site.representations:
        expected = np.zeros((16,16),complex)
        for sector,span in slices.items():
            n,j,label = sector.charge
            matrix = irrep_representation_matrices(group,label)[op]
            expected[span,span] = np.kron(np.eye(site.leg.dims[sector]*(j+1)),matrix)
        np.testing.assert_allclose(site.basis.conj().T @ site.representations[op] @ site.basis,expected,atol=1e-12)
    from pyqed.qchem.dmrg import dmrg
    from pyqed.letta.su2_qchem import SU2LETTA
    def forbidden(*args,**kwargs):
        pytest.fail('Global determinant projection or LETTA entered the MPS solver.')
    monkeypatch.setattr(dmrg,'_build_spatial_active_hamiltonian_matrix',forbidden)
    monkeypatch.setattr(SU2LETTA,'from_integrals',forbidden)
    reps = {g:np.kron(np.eye(2),u) for g,u in irrep_representation_matrices(group,e).items()}
    h = np.kron([[.1,.3],[.3,.7]],np.eye(2))
    solver = MultipletDMRG.from_integrals(h,np.zeros((4,)*4),nelec=1,spin=1,D=2,
        orbital_representations=reps,wfnsym=e,point_group=group,contractions='reduced').run(nstates=2,nsweeps=4)
    assert solver.success and solver.spatial_multiplet_count==2
    assert solver.diagnostics["sweep_control"]=="shared"
    assert solver.diagnostics["max_bond_mode"]=="per_sector"
    np.testing.assert_allclose(solver.e_tot,np.linalg.eigvalsh(h),atol=1e-10)
    diagnostics = solver.symmetry_diagnostics()
    assert diagnostics['orthogonality_error']<1e-10 and diagnostics['covariance_error']<1e-10
    np.testing.assert_allclose(diagnostics['partner_energies_hartree'],solver.e_tot,atol=1e-10)
    for state in solver.states:
        for center in range(len(state.tensors)):
            before = state.copy()
            state.canonicalize(center)
            np.testing.assert_allclose(state.overlap(before),1,atol=1e-10)
            assert all(t.legs[1].symmetry==molecular_symmetry(group) for t in state.tensors)
    with pytest.raises(ValueError,match='Only'):
        solver.run(nstates=3)


@pytest.mark.parametrize('group,e,target', [('D3h',"E'","E'"),('D6h','E1u','E2g')])
@pytest.mark.parametrize('spin', [0,2])
def test_two_electron_product_sectors_and_orbital_rotation(group,e,target,spin):
    reps = {g:np.kron(np.eye(2),u) for g,u in irrep_representation_matrices(group,e).items()}
    h = np.kron([[.1,.3],[.3,.7]],np.eye(2))
    eri = .2*np.einsum('pq,rs->pqrs',np.eye(4),np.eye(4))
    rotation = np.linalg.qr(np.random.default_rng(72).normal(size=(4,4)))[0]
    reps = {op:rotation.T @ matrix @ rotation for op,matrix in reps.items()}
    if spin:
        target = "A2'" if group=='D3h' else 'A2g'
    solver = MultipletDMRG.from_integrals(rotation.T @ h @ rotation,eri,nelec=2,spin=spin,D=4,
        orbital_representations=reps,wfnsym=target,point_group=group).run(nsweeps=6)
    assert solver.success
    np.testing.assert_allclose(solver.e_tot,2*np.linalg.eigvalsh(h)[0]+.2,atol=1e-10)
    assert solver.symmetry_diagnostics()['covariance_error']<1e-10


from dmrg_molecule_fixtures import benzene, independent_integrals


@pytest.mark.parametrize('target', ['A1g','E1u','E2g'])
def test_benzene_multiplet_dmrg_independent_reference(benzene,independent_integrals,target):
    from pyqed.qchem.dmrg.dmrg import DMRG
    from examples.qchem.qualify_casci_symmetry import reference
    mf,coeff = benzene
    driver = DMRG(mf,6,6,D=32,spin=0,symmetry='su2',verbose=0)
    solver = driver.run(mo_coeff=coeff,wfnsym=target,nsweeps=8)
    assert isinstance(solver,MultipletDMRG) and solver.success
    expected = reference(mf,coeff,6,6,0,target,integrals=independent_integrals(coeff))[0]
    np.testing.assert_allclose(solver.e_tot,expected[:solver.partner_dimension],atol=1e-7,rtol=0)
    diagnostics = solver.symmetry_diagnostics()
    assert max(diagnostics['covariance_error'],diagnostics['orthogonality_error'])<1e-10
    np.testing.assert_allclose(diagnostics['partner_energies_hartree'],solver.e_tot,atol=1e-9,rtol=0)
    assert driver.spatial_method=='complementary'
    assert solver.nsites==driver.nsites==4
    assert driver.d is None and sorted(driver.physical_site_dimensions)==[4,4,16,16]
    if target == 'A1g':
        from pyqed.qchem.mcscf.direct_ci import transform_active_space_spatial_integrals
        one,two = driver.make_rdm12()
        h,v,core = transform_active_space_spatial_integrals(mf,coeff,6,driver.ncore,use_cholesky=False)
        energy = np.einsum('pq,qp',h,one)+.5*np.einsum('pqrs,pqrs',v,two)+core
        np.testing.assert_allclose(energy,solver.energy,atol=1e-9,rtol=0)
        np.testing.assert_allclose(np.einsum('pqrr->pq',two),5*one.T,atol=1e-9)
        np.testing.assert_allclose(np.trace(driver.make_rdm1(spatial=True,with_core=True)),42,atol=1e-9)



from dmrg_molecule_fixtures import h3plus


@pytest.mark.parametrize('target', ["A1'","E'"])
def test_multiplet_dmrg_rdms_match_ci(h3plus,target,monkeypatch):
    from pyqed.qchem.dmrg.dmrg import DMRG
    from pyqed.qchem.mcscf.casci import CASCI
    from pyqed.qchem.dmrg import dmrg
    reference = CASCI(h3plus,3,2,multiplicity=1).run(wfnsym=target)
    driver = DMRG(h3plus,3,2,D=4,spin=0,symmetry='su2',verbose=0)
    solver = driver.run(wfnsym=target,nsweeps=6)
    def forbidden(*args,**kwargs):
        pytest.fail('RDM calculation materialized the molecular determinant vector.')
    monkeypatch.setattr(dmrg,'_spatial_mps_to_spin_string_ci',forbidden)
    monkeypatch.setattr(dmrg,'_nonabelian_mps_to_dense_vector',forbidden)
    actual = [driver.make_rdm12(i) for i in range(solver.partner_dimension)]
    expected = [reference.make_rdm12(i) for i in range(solver.partner_dimension)]
    for rank in (0,1):
        np.testing.assert_allclose(np.mean([r[rank] for r in actual],axis=0),
            np.mean([r[rank] for r in expected],axis=0),atol=1e-9)
    for one,two in actual:
        np.testing.assert_allclose(np.trace(one),2,atol=1e-10)
        np.testing.assert_allclose(np.einsum('pqrr->pq',two),one.T,atol=1e-9)


@pytest.mark.parametrize('group', ['D3h','D6h'])
def test_one_dimensional_irreps_and_parameter_guards(group):
    symmetry = PointGroupSymmetry(group)
    for label in symmetry.group.irrep_names:
        if symmetry.dimension(label) != 1:
            continue
        solver = MultipletDMRG.from_integrals(np.array([[.3]]),np.zeros((1,)*4),
            nelec=1,spin=1,D=1,point_group=group,wfnsym=label,
            orbital_representations=irrep_representation_matrices(group,label))
        solver.run(nsweeps=2)
        np.testing.assert_allclose(solver.e_tot,[.3],atol=1e-12)
        assert solver.success and solver.symmetry_diagnostics()['covariance_error']<1e-12
    for kwargs in ({'nsweeps':True},{'tol':float('nan')},{'metric_rtol':1}):
        with pytest.raises(ValueError):
            solver.run(**kwargs)
    with pytest.raises(IndexError):
        solver.states[0].component_tensors(boundary_row=100)
    with pytest.raises(ValueError,match='lengths'):
        solver.states[0].expectation([])


def test_local_workspace_guard_precedes_dense_allocation(monkeypatch):
    from types import SimpleNamespace
    reps = irrep_representation_matrices('D3h',"E'")
    solver = MultipletDMRG.from_integrals(np.eye(2),np.zeros((2,)*4),nelec=1,
        spin=1,D=1,point_group='D3h',wfnsym="E'",orbital_representations=reps,contractions='reduced')
    from pyqed.mps.nonabelian import multiplet_sweep
    monkeypatch.setattr(multiplet_sweep,'_diagonal_norm',lambda *args:None)
    monkeypatch.setattr(solver,'fusion_maps',(SimpleNamespace(parameter_count=10**4),))
    def forbidden(*args,**kwargs):
        pytest.fail('Workspace guard must precede dense allocation.')
    monkeypatch.setattr(solver,'_local_action',forbidden)
    with pytest.raises(MemoryError,match='workspace'):
        solver._optimize_site(0,None,None,None,None,metric_rtol=1e-10,local_tol=1e-11)
