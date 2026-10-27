import numpy as np
import pytest
from pyqed.symmetry import OrbitalRotationSymmetry, Leg, Irrep, ProductSymmetry, U1Symmetry, SU2Symmetry
from pyqed.symmetry.continuous import orbital_subspaces
from pyqed.mps.nonabelian.fusion import irrep_coupling


@pytest.mark.parametrize('group,labels', [('Coov',['A1','A2','E1','E3']),
    ('Dooh',['A1g','A2u','E1u','E3g']), ('SO3',['L0','L1','L2','L3'])])
def test_couplings_are_complete_intertwiners(group, labels):
    sym = OrbitalRotationSymmetry(group)
    for a in labels:
        for b in labels:
            ga,gb = sym.generators(a),sym.generators(b)
            da,db = sym.dimension(a),sym.dimension(b)
            blocks = []
            for c in sym.fuse(a,b):
                cg = irrep_coupling(sym,a,b,c).reshape(da*db,-1)
                for key, target in sym.generators(c).items():
                    total = (np.kron(ga[key],np.eye(db))+np.kron(np.eye(da),gb[key])
                             if key.startswith('L') else np.kron(ga[key],gb[key]))
                    np.testing.assert_allclose(total@cg,cg@target,atol=2e-12)
                blocks.append(cg)
            unitary = np.column_stack(blocks)
            np.testing.assert_allclose(unitary@unitary.conj().T,np.eye(da*db),atol=2e-12)


@pytest.mark.parametrize('group,label',[('SO3','L2'),('Coov','E3'),('Dooh','E2u')])
def test_rotated_multiplet_decomposition_and_product_legs(group,label):
    from scipy.linalg import block_diag
    sym = OrbitalRotationSymmetry(group)
    n = sym.dimension(label)
    rng = np.random.default_rng(8)
    u,_ = np.linalg.qr(rng.normal(size=(2*n,2*n)))
    generators = {k:u@block_diag(v,v)@u.T for k,v in sym.generators(label).items()}
    spaces = orbital_subspaces(sym,generators)
    assert spaces[label].shape == (2*n,2,n)
    for key,matrix in generators.items():
        np.testing.assert_allclose(np.einsum('ij,jcr->icr',matrix,spaces[label]),
                                   spaces[label]@sym.generators(label)[key],atol=1e-11)
    product = ProductSymmetry((U1Symmetry(),SU2Symmetry(),sym))
    leg = Leg({Irrep((2,2,label)):4},symmetry=product)
    assert leg.dim == 4 and leg.full_dim == 4*3*n


def test_no_angular_cutoff_and_no_partner_labels():
    sym = OrbitalRotationSymmetry('SO3')
    assert sym.dimension('L100') == 201
    assert sym.fuse('L100','L1') == ('L99','L100','L101')
    with pytest.raises(ValueError):
        OrbitalRotationSymmetry('Coov').dimension('E1x')


def test_atomic_p_shell_has_pauli_allowed_ls_terms():
    from pyqed.mps.nonabelian.multiplet_sites import OrbitalMultipletSite
    site = OrbitalMultipletSite('SO3','L1')
    terms = {q.charge:copies for q,copies in site.leg.dims.items() if q.charge[0] == 2}
    assert terms == {(2,0,'L0'):1, (2,0,'L2'):1, (2,2,'L1'):1}
    assert site.leg.full_dim == 64
    np.testing.assert_allclose(site.basis.conj().T@site.basis,np.eye(64),atol=1e-11)


@pytest.mark.parametrize('group,label',[('Coov','E1'),('Dooh','E1u')])
def test_cylindrical_sites_preserve_complete_doublets(group,label):
    from pyqed.mps.nonabelian.multiplet_sites import OrbitalMultipletSite
    site = OrbitalMultipletSite(group,label)
    assert site.leg.dims[Irrep((1,1,label))] == 1
    assert site.leg.full_dim == 16


@pytest.mark.parametrize('group,label',[('Coov','E1'),('Dooh','E1u'),('SO3','L1')])
@pytest.mark.parametrize('spin',[0,2])
def test_reduced_dmrg_continuous_target_and_partners(group,label,spin):
    from scipy.linalg import block_diag
    from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG
    sym = OrbitalRotationSymmetry(group)
    g0,g1 = sym.generators(sym.identity),sym.generators(label)
    generators = {k:block_diag(g0[k],g1[k]) for k in g0}
    n = 1+sym.dimension(label)
    h = np.diag([0.]+[1.]*(n-1))
    eri = .2*np.einsum('pq,rs->pqrs',np.eye(n),np.eye(n))
    solver = MultipletDMRG.from_integrals(h,eri,nelec=2,spin=spin,D=4,
        orbital_representations=generators,point_group=group,wfnsym=label,n_threads=1).run(nsweeps=3)
    assert solver.success
    assert len(solver.states) == sym.dimension(label)
    np.testing.assert_allclose(solver.e_tot,1.2,atol=1e-10)
    assert solver.symmetry_diagnostics()['covariance_error'] < 1e-9


@pytest.mark.parametrize('atom,group,charge',[('Ne 0 0 0','SO3',0),
    ('H 0 0 0; F 0 0 1.7','Coov',0),('N 0 0 -1; N 0 0 1','Dooh',0)])
def test_qchem_builds_full_continuous_multiplets(atom,group,charge):
    from pyqed.qchem import Molecule
    from pyqed.qchem.symmetry import mo_symmetry_representations, SymmetryEigensolver
    mol = Molecule(atom=atom,unit='bohr',basis='sto-3g',charge=charge,symmetry=group).build()
    info = mol.symmetry_info
    assert info.group.continuous and info.group.name == group
    basis = np.column_stack(list(info.symm_orb.values()))
    np.testing.assert_allclose(basis.conj().T@mol.overlap@basis,np.eye(mol.nao),atol=1e-10)
    generators = mo_symmetry_representations(mol,basis,generators=True)
    orbital_subspaces(group,generators)
    solver = SymmetryEigensolver(mol)
    values,coeff = solver.eigh(mol.hcore)
    assert len(values) == mol.nao








def test_continuous_displacement_labels():
    from pyqed.qchem import Molecule
    from pyqed.qchem.symmetry import assign_mode_irreps
    mol = Molecule(atom='H 0 0 -1; H 0 0 1',unit='bohr',basis='sto-3g',symmetry=True).build()
    modes = np.array([[[0,0,-1],[0,0,1]], [[1,0,0],[1,0,0]], [[0,1,0],[0,1,0]]])
    assert assign_mode_irreps(mol,modes)[0] == ('A1g','E1u','E1u')




def test_continuous_generators_reject_broken_hamiltonian():
    from pyqed.symmetry.continuous import OrbitalGeneratorAction
    sym = OrbitalRotationSymmetry('SO3')
    action = OrbitalGeneratorAction(sym.generators('L1'),'L1',point_group='SO3')
    with pytest.raises(ValueError,match='breaks continuous'):
        action.validate_hamiltonian(np.diag([0.,0.,1.]))


@pytest.mark.parametrize('group,orbital,target',[('Coov','E1','E2'),
    ('Dooh','E1u','E2g'),('SO3','L1','L2')])
def test_manybody_target_absent_from_orbital_catalogue(group,orbital,target):
    from scipy.linalg import block_diag
    from pyqed.mps.nonabelian.multiplet_dmrg import MultipletDMRG
    sym = OrbitalRotationSymmetry(group)
    scalar,shell = sym.generators(sym.identity),sym.generators(orbital)
    generators = {key:block_diag(scalar[key],shell[key]) for key in scalar}
    n = sym.dimension(orbital)+1
    h = np.diag([0.]+[1.]*(n-1))
    eri = .2*np.einsum('pq,rs->pqrs',np.eye(n),np.eye(n))
    solver = MultipletDMRG.from_integrals(h,eri,nelec=2,spin=0,D=8,
        orbital_representations=generators,point_group=group,wfnsym=target,n_threads=1).run(nsweeps=3)
    np.testing.assert_allclose(solver.e_tot,2.2,atol=1e-10)
    assert len(solver.states) == sym.dimension(target)
    assert solver.symmetry_diagnostics()['covariance_error']<1e-9


def test_large_complete_shell_rejects_before_fock_allocation():
    from pyqed.mps.nonabelian.multiplet_sites import OrbitalMultipletSite
    with pytest.raises(MemoryError,match='512 MiB'):
        OrbitalMultipletSite('SO3','L3')
