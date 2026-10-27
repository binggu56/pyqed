"""Complete orbital-multiplet sites for charge x spin x point-group MPS.

Local Fock spaces have dimension 4 (A/B) or 16 (E). Only these bounded local
spaces are decomposed; no determinant basis for the molecular active space
is formed. Spatial matrix units fix partner phases, and spin lowering fixes
the Condon--Shortley convention shared with the MPS fusion maps.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from pyqed.symmetry import Irrep, Leg, ProductSymmetry, SU2Symmetry, U1Symmetry, PointGroupSymmetry
from pyqed.symmetry.point_group import irrep_subspaces
from pyqed.qchem.symmetry import irrep_representation_matrices
from pyqed.qchem.dmrg.spatial_terms import spatial_local_ops
from pyqed.symmetry.continuous import OrbitalRotationSymmetry, orbital_subspaces


def molecular_symmetry(group):
    spatial = (OrbitalRotationSymmetry(group) if group in ('Coov','Dooh','SO3')
               else PointGroupSymmetry(group))
    return ProductSymmetry((U1Symmetry(), SU2Symmetry(), spatial))


def orbital_multiplets(group, representations, *, subspaces=None):
    """Orthonormal orbital change of basis with one complete multiplet per site."""
    if group in ('Coov','Dooh','SO3'):
        spaces = orbital_subspaces(group, representations) if subspaces is None else subspaces
        return _multiplet_columns(spaces)
    definition = PointGroupSymmetry(group).group if isinstance(group,str) else group
    if all(d==1 for d in definition.irrep_dimensions.values()):
        matrices = np.stack([representations[op] for op in definition.operation_names])
        diagonal = np.diagonal(matrices,axis1=1,axis2=2)
        eye = np.eye(matrices.shape[1])
        if np.max(abs(matrices-diagonal[:,:,None]*eye),initial=0)<1e-12:
            labels = [None]*len(eye)
            for label in definition.irrep_names:
                selected = np.flatnonzero(np.max(abs(diagonal-np.asarray(
                    definition.character_table[label])[:,None]),axis=0)<1e-10)
                for i in selected:
                    labels[i] = label
            if all(label is not None for label in labels):
                return eye,tuple(labels),tuple((i,) for i in range(len(eye)))
    spaces = irrep_subspaces(group, representations)
    return _multiplet_columns(spaces)


def _multiplet_columns(spaces):
    columns, labels, blocks = [], [], []
    offset = 0
    for label, space in spaces.items():
        for copy in range(space.shape[1]):
            columns.append(space[:,copy,:])
            labels.append(label)
            blocks.append(tuple(range(offset, offset+space.shape[2])))
            offset += space.shape[2]
    transform = np.column_stack(columns)
    if transform.shape[0] != transform.shape[1] or not np.allclose(
            transform.conj().T @ transform, np.eye(len(transform)), atol=1e-9):
        raise ValueError('The orbital space does not contain complete molecular irreps.')
    return np.real_if_close(transform), tuple(labels), tuple(blocks)


def _local_fock_representation(matrix):
    """Exterior powers for one bounded (one/two orbital) physical site."""
    norb = len(matrix)
    if norb not in (1,2):
        raise ValueError('A multiplet site must contain one or two spatial orbitals.')
    spin_orbital = np.kron(matrix, np.eye(2))
    occupations = []
    for index in range(4**norb):
        digits = np.unravel_index(index, (4,)*norb)
        occupations.append([2*p+s for p,digit in enumerate(digits) for s in range(2)
                            if digit & (1 << s)])
    out = np.zeros((len(occupations),)*2, dtype=complex)
    for i, bra in enumerate(occupations):
        for j, ket in enumerate(occupations):
            if len(bra) == len(ket):
                out[i,j] = np.linalg.det(spin_orbital[np.ix_(bra,ket)])
    return out


class OrbitalMultipletSite:
    """A bounded local Fock space reduced under U(1) x SU(2) x point group."""
    def __init__(self, group, orbital_irrep):
        template = self._template(group,orbital_irrep)
        self.group,self.orbital_irrep = group,orbital_irrep
        self.norb,self.symmetry = template.norb,template.symmetry
        self.leg = Leg(dict(template.leg.dims),symmetry=self.symmetry,name=orbital_irrep)
        self.representations = {key:value.copy() for key,value in template.representations.items()}
        self.basis,self.spin_squared,self.number = (value.copy() for value in
            (template.basis,template.spin_squared,template.number))

    @classmethod
    @lru_cache(maxsize=16)
    def _template(cls, group, orbital_irrep):
        site = object.__new__(cls)
        site._build(group,orbital_irrep)
        return site

    def _build(self, group, orbital_irrep):
        self.group, self.orbital_irrep = group, orbital_irrep
        continuous = group in ('Coov','Dooh','SO3')
        matrices = (OrbitalRotationSymmetry(group).generators(orbital_irrep) if continuous
                    else irrep_representation_matrices(group, orbital_irrep))
        self.norb = next(iter(matrices.values())).shape[0]
        self.representations = (_local_fock_generators(matrices) if continuous else
                               {op: _local_fock_representation(matrix) for op,matrix in matrices.items()})
        self.symmetry = molecular_symmetry(group)
        ops = spatial_local_ops()
        n = np.zeros((4**self.norb,)*2, complex)
        sz, sm = np.zeros_like(n), np.zeros_like(n)
        for orbital in range(self.norb):
            def local(operator):
                out = np.ones((1,1))
                for site in range(self.norb):
                    out = np.kron(out, operator if site == orbital else ops['I'])
                return out
            n += local(ops['n'])
            sz += local((ops['nu']-ops['nd'])/2)
            sm += local(ops['cdd'] @ ops['cu'])
        s2 = sz @ sz + (sm @ sm.conj().T + sm.conj().T @ sm)/2
        sectors, columns = {}, []
        for charge in range(2*self.norb+1):
            for j2 in range(charge%2, min(charge,2*self.norb-charge)+1,2):
                selection = np.flatnonzero((abs(n.diagonal()-charge)<1e-10) &
                                           (abs(2*sz.diagonal()-j2)<1e-10))
                values, vectors = eigh(s2[np.ix_(selection,selection)])
                vectors = vectors[:, abs(values-j2*(j2+2)/4)<1e-9]
                highest = np.zeros((len(n), vectors.shape[1]), complex)
                highest[selection] = vectors
                if not highest.shape[1]:
                    continue
                reps = {op: highest.conj().T @ matrix @ highest
                        for op,matrix in self.representations.items()}
                spaces_by_label = orbital_subspaces(group,reps) if continuous else irrep_subspaces(group,reps)
                for label, spaces in spaces_by_label.items():
                    sector = Irrep((charge,j2,label))
                    sectors[sector] = spaces.shape[1]
                    for copy in range(spaces.shape[1]):
                        partners = highest @ spaces[:,copy,:]
                        for m2 in range(j2,-j2-1,-2):
                            columns.append(partners)
                            if m2 > -j2:
                                partners = sm @ partners / np.sqrt((j2+m2)*(j2-m2+2)/4)
        self.leg = Leg(sectors, symmetry=self.symmetry, name=orbital_irrep)
        self.basis = np.column_stack(columns)
        if not np.allclose(self.basis.conj().T @ self.basis, np.eye(len(n)), atol=1e-10):
            raise ArithmeticError('Incomplete orbital-multiplet local basis.')
        self.spin_squared = self.basis.conj().T @ s2 @ self.basis
        self.number = self.basis.conj().T @ n @ self.basis


def _local_fock_generators(generators):
    """Second-quantize generators within one complete orbital shell only."""
    norb = len(generators['Lz'])
    size = 4**norb
    if size*size*16*(2*norb+8) > 512*1024**2:
        raise MemoryError('Complete orbital-shell construction exceeds its 512 MiB workspace budget.')
    ops = spatial_local_ops()
    annihilators = []
    for p in range(norb):
        for spin in ('cu','cd'):
            a = np.ones((1,1))
            for q in range(norb):
                a = np.kron(a,ops['JW'] if q<p else ops[spin] if q==p else ops['I'])
            annihilators.append(a)
    result = {}
    for name,matrix in generators.items():
        if not name.startswith('L'):
            result[name] = _local_fock_representation(matrix)
            continue
        out = np.zeros((size,size),complex)
        for p,q in zip(*np.nonzero(matrix)):
            for spin in range(2):
                out += matrix[p,q]*(annihilators[2*p+spin].conj().T@annihilators[2*q+spin])
        result[name] = out
    return result
