"""Finite-group irreps for shared Legs, using PyQED's molecular conventions.

Matrix-unit reduction is the standard finite-group projector construction
(Cotton, Chemical Applications of Group Theory, 3rd ed., Wiley, 1990).
The real finite-group conventions are those of ``qchem.symmetry``; no subgroup
reduction is made. These groups have multiplicity-free binary irrep products.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh


@lru_cache(maxsize=16)
def _group(name):
    from pyqed.qchem.symmetry import get_point_group
    return get_point_group(name)


@dataclass(frozen=True)
class PointGroupSymmetry:
    """Real finite molecular irreps, dimensions, products and CG maps.

    Uses finite-group matrix units following Cotton, Chemical Applications
    of Group Theory, 3rd ed., Wiley (1990), without subgroup reduction.
    """
    name: str

    def __post_init__(self):
        group = _group(self.name)
        if group.name not in ('C1', 'Cs', 'Ci', 'C2', 'C2v', 'C2h', 'D2', 'D2h', 'D3h', 'D6h'):
            raise ValueError('Point-group Legs require a supported finite real molecular group.')
        object.__setattr__(self, 'name', group.name)

    @property
    @lru_cache(maxsize=16)
    def group(self):
        return _group(self.name)

    @property
    def identity(self):
        return next(q for q in self.group.irrep_names
                    if all(abs(x-1) < 1e-12 for x in self.group.character_table[q]))

    @lru_cache(maxsize=32)
    def dual(self, irrep):
        return self.group.irrep_names[self.group.irrep_id(irrep)]

    @lru_cache(maxsize=32)
    def dimension(self, irrep):
        return self.group.irrep_dimensions[self.dual(irrep)]

    @lru_cache(maxsize=256)
    def fuse(self, left, right):
        return tuple(label for label, count in
                     self.group.irrep_product_decomposition(left, right).items()
                     for _ in range(count))

    def allows(self, bra, op, ket):
        return bra in self.fuse(ket, op)

    def coupling(self, left, right, fused):
        return _point_group_coupling(self.name, left, right, fused)


def irrep_subspaces(group, representations, *, tol=1e-10, irreps=None):
    """Return orthonormal (basis, copy, row) intertwiners for a representation.

    Only diagonalize the first-row matrix unit, then generate every partner
    with the off-diagonal units, keeping their relative phases fixed.
    """
    from pyqed.qchem.symmetry import symmetry_partner_projectors
    group = _group(group) if isinstance(group,str) else group
    result = {}
    for label in group.irrep_names if irreps is None else irreps:
        units = symmetry_partner_projectors(group, representations, label)
        first = units[0, 0]
        values, vectors = eigh((first + first.conj().T)/2)
        vectors = vectors[:, values > 1-tol]
        if not vectors.shape[1]:
            continue
        for column in range(vectors.shape[1]):
            pivot = np.argmax(abs(vectors[:, column]))
            vectors[:, column] *= np.conj(vectors[pivot, column])/abs(vectors[pivot, column])
        result[label] = np.stack([units[row, 0] @ vectors
                                 for row in range(group.irrep_dimensions[label])], axis=-1)
    return result


@lru_cache(maxsize=256)
def _product_spaces(group, left, right):
    from pyqed.qchem.symmetry import irrep_representation_matrices
    group = _group(group)
    a = irrep_representation_matrices(group, left)
    b = irrep_representation_matrices(group, right)
    product = {op: np.kron(matrix, b[op]) for op, matrix in a.items()}
    labels = group.irrep_product_decomposition(left,right)
    spaces = irrep_subspaces(group,product,irreps=labels)
    return spaces,next(iter(a.values())).shape[0],next(iter(b.values())).shape[0]


@lru_cache(maxsize=512)
def _point_group_coupling(group, left, right, fused):
    definition = _group(group)
    if all(d == 1 for d in definition.irrep_dimensions.values()):
        if fused not in definition.irrep_product_decomposition(left, right):
            raise ValueError(f'{fused} is absent from {left} x {right} in {group}.')
        out = np.ones((1, 1, 1))
        out.setflags(write=False)
        return out
    spaces,da,db = _product_spaces(group,left,right)
    if fused not in spaces:
        raise ValueError(f'{fused} is absent from {left} x {right} in {group}.')
    space = spaces[fused]
    if space.shape[1] != 1:
        raise NotImplementedError('Binary outer multiplicity requires an explicit fusion slot.')
    out = space[:,0].reshape(da,db,-1)
    out.setflags(write=False)
    return out
