"""Continuous orbital symmetry from analytic AO rotation generators.

Uses SO(3) angular-momentum ladders (NIST DLMF 14.30 and 34.1) and Schur
multiplicity spaces. AO generators differentiate normalized Cartesian shell
rotations analytically and reuse PyQED's spherical transformation. This is an
adaptation to contracted Gaussian bases, not a finite-subgroup approximation.
Only rotationally invariant, nonrelativistic atomic Hamiltonians admit SO3;
Coov/Dooh require a linear geometry. Spin-orbit coupling is outside this model.
"""
from dataclasses import replace
import numpy as np
from scipy.linalg import eigh, expm
from pyqed.symmetry.continuous import OrbitalRotationSymmetry, orbital_subspaces


def continuous_point_group(name):
    from .symmetry import PointGroup
    symmetry = OrbitalRotationSymmetry(name)
    axes = np.eye(3) if name == 'SO3' else np.eye(3)[2:]
    names = ['Lx','Ly','Lz'] if name == 'SO3' else ['Lz']
    operations = [expm(.37*_cross(axis)) for axis in axes]
    if name != 'SO3':
        names.append('reflection'); operations.append(np.diag([1.,-1.,1.]))
    if name == 'Dooh':
        names.append('inversion'); operations.append(-np.eye(3))
    return PointGroup(name,tuple(names),tuple(operations),(symmetry.identity,),{},continuous=True)


def _cross(axis):
    x,y,z = axis
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def build_continuous_molecular_symmetry(mol, group, origin, axes, *, tol=1e-8):
    from .symmetry import (MolecularSymmetry, atom_permutation, ao_representation_matrix,
                           _rotating_ao_representation)
    if group.name == 'SO3' and len(mol.atom_coords()) != 1:
        raise ValueError('Orbital SO3 requires a single atomic center.')
    group = replace(group,operation_matrices=tuple(axes@a@axes.T for a in group.operation_matrices))
    reps,permutations,rotations = {},{},{}
    for name,operation in zip(group.operation_names,group.operation_matrices):
        permutations[name] = atom_permutation(mol,operation,tol=tol,origin=origin)
        rotations[name] = ao_representation_matrix(mol,operation,tol=tol,origin=origin)
        if name.startswith('L'):
            axis = axes[:,{'Lx':0,'Ly':1,'Lz':2}[name]]
            reps[name] = 1j*_rotating_ao_representation(mol,_cross(axis),tol,
                origin=origin,atom_perm=np.arange(len(mol.atom_coords())),generator=True)
        else:
            reps[name] = ao_representation_matrix(mol,operation,tol=tol,origin=origin)
    s = np.asarray(mol.overlap)
    values,vectors = eigh(s)
    if np.min(values)<=0:
        raise ValueError('Continuous orbital symmetry requires a positive AO metric.')
    x = (vectors/np.sqrt(values))@vectors.conj().T
    orthogonal = {name:x.conj().T@s@a@x for name,a in reps.items()}
    spaces = orbital_subspaces(group.name,orthogonal,tol=tol)
    multiplets = {label:np.einsum('ij,jcr->icr',x,space) for label,space in spaces.items()}
    bases = {label:space.reshape(mol.nao,-1) for label,space in multiplets.items()}
    labels = tuple(dict.fromkeys((OrbitalRotationSymmetry(group.name).identity,*spaces)))
    group = replace(group,irrep_names=labels)
    projectors = {label:basis@basis.conj().T@s for label,basis in bases.items()}
    ao_labels,ao_ids = [],[]
    for i in range(mol.nao):
        weights = {label:float(np.sum(abs(basis.conj().T@s[:,i])**2)/s[i,i].real)
                   for label,basis in bases.items()}
        label = max(weights,key=weights.get)
        pure = weights[label]>1-tol
        ao_labels.append(label if pure else None)
        ao_ids.append(group.irrep_id(label) if pure else None)
    return MolecularSymmetry(group,permutations,rotations,projectors,bases,
        tuple(ao_labels),tuple(ao_ids),origin=origin,axes=axes,multiplets=multiplets,ao_generators=reps)


class CIOrbitalAction:
    """Exact continuous matrix units in a bounded CI determinant space.

    Second-quantized angular generators are assembled from existing spin-string
    links, then decomposed by Casimir/ladder operations. This CI-only reference
    setup has a 256 MiB dense workspace guard; it is never used by MPS DMRG.
    Hamiltonian contractions and spin restriction reuse the direct-CI solver.
    """
    def __init__(self, model, wfnsym):
        from scipy.sparse import coo_matrix, eye, kron
        from pyqed.symmetry.continuous import OrbitalGeneratorAction
        from .symmetry import mo_symmetry_representations, _symmetry_ao_coeff
        from .mcscf.direct_ci import build_spin_string_connectivity
        from .mcscf.casci import _basis_spin_strings, _cached_spin_string_basis, _string_transform_matrix
        self.active_symmetry = model._update_symmetry_metadata(model.binary)
        group = model.mol.symmetry_info.group
        coeff = _symmetry_ao_coeff(model.mf,model.mo_coeff)[:,model.ncore:model.ncore+model.ncas]
        generators = mo_symmetry_representations(model.mol,coeff,generators=True)
        self.orbital_action = OrbitalGeneratorAction(generators,wfnsym,point_group=group.name)
        self.group,self.wfnsym = group,self.orbital_action.wfnsym
        self.norb,self.dimension = model.ncas,self.orbital_action.dimension
        self.ndet = len(model.binary)
        if self.ndet**2*16*(len(generators)+8)>256*1024**2:
            raise MemoryError('Continuous CI symmetry setup exceeds its 256 MiB workspace budget; use reduced MPS DMRG.')
        conn = build_spin_string_connectivity(model.binary)
        self.alpha,self.beta = _basis_spin_strings(model.binary)
        if hasattr(model.binary,'alpha_occ'):
            ia,ib = np.indices((len(self.alpha),len(self.beta)))
            ia,ib = ia.ravel(),ib.ravel()
        else:
            binary = np.ascontiguousarray(model.binary,dtype=np.int8)
            _,_,ia,ib = _cached_spin_string_basis(binary.shape,binary.tobytes())
        order = ia*len(self.beta)+ib
        if len(np.unique(order)) != self.ndet or self.ndet != len(self.alpha)*len(self.beta):
            raise ValueError('Continuous CI requires complete spin-string products.')
        many = {}
        for name,g in generators.items():
            if name.startswith('L'):
                matrices = []
                for suffix,occupations in (('A',self.alpha),('B',self.beta)):
                    matrices.append(coo_matrix((-getattr(conn,'phase_'+suffix)*g[
                        getattr(conn,'p_'+suffix),getattr(conn,'q_'+suffix)],
                        (getattr(conn,'I_'+suffix),getattr(conn,'J_'+suffix))),
                        shape=(len(occupations),len(occupations))).tocsr())
                    matrices[-1].setdiag(occupations@np.diag(g))
                full = kron(matrices[0],eye(len(self.beta)))+kron(eye(len(self.alpha)),matrices[1])
                many[name] = full.toarray()[np.ix_(order,order)]
            else:
                full = np.kron(_string_transform_matrix(g,self.alpha,complex),_string_transform_matrix(g,self.beta,complex))
                many[name] = full[np.ix_(order,order)]
        spaces = orbital_subspaces(group.name,many)
        self.generators = many
        self.partner_basis = spaces.get(self.wfnsym,np.empty((self.ndet,0,self.dimension)))
        self.multiplicity = self.partner_basis.shape[1]

    def validate_hamiltonian(self,h1e,eri=None):
        self.orbital_action.validate_hamiltonian(h1e,eri)

    def apply(self,vector,row=0,column=0):
        return self.partner_basis[:,:,row]@(self.partner_basis[:,:,column].conj().T@vector)
