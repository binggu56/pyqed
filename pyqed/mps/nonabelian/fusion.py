"""Structural Clebsch--Gordan maps for shared symmetry Legs.

Product intertwiners follow Singh, Pfeifer and Vidal, PRA 82, 050301 (2010),
doi:10.1103/PhysRevA.82.050301. This implementation explicitly contracts
small irrep-row tensors; it does not implement compressed F-symbol kernels.
Free tensor entries are multiplicities, never independent irrep partners.
"""
from functools import lru_cache, cached_property
import numpy as np
from scipy.sparse import coo_matrix
from pyqed.symmetry import Irrep, Leg, ProductSymmetry, SU2Symmetry, U1Symmetry
from pyqed.symmetry.point_group import PointGroupSymmetry
from pyqed.symmetry.continuous import OrbitalRotationSymmetry
from pyqed.mps.su2 import SU2Irrep
from .coupling import clebsch_gordan, ordered_two_m_values


@lru_cache(maxsize=4096)
def irrep_coupling(symmetry, left, right, fused):
    """Isometric map from a fused irrep row into two ordered child rows."""
    if fused not in symmetry.fuse(left, right):
        raise ValueError('Forbidden fusion channel.')
    if isinstance(symmetry, U1Symmetry):
        out = np.ones((1, 1, 1))
    elif isinstance(symmetry, SU2Symmetry):
        a, b, c = (SU2Irrep(j) for j in (left, right, fused))
        out = np.array([[[clebsch_gordan(a,b,c,ma,mb,mc)
                         for mc in ordered_two_m_values(c)]
                        for mb in ordered_two_m_values(b)] for ma in ordered_two_m_values(a)])
    elif isinstance(symmetry, (PointGroupSymmetry, OrbitalRotationSymmetry)):
        out = symmetry.coupling(left, right, fused)
    elif isinstance(symmetry, ProductSymmetry):
        out = np.ones((1, 1, 1))
        for factor, a, b, c in zip(symmetry.factors, left, right, fused):
            term = irrep_coupling(factor, a, b, c)
            shape = tuple(x*y for x,y in zip(out.shape, term.shape))
            out = np.einsum('abc,def->adbecf', out, term).reshape(shape)
    else:
        raise TypeError(f'No coupling maps for {symmetry}.')
    out.setflags(write=False)
    return out


def full_slices(leg):
    start, slices = 0, {}
    for sector in leg.sectors:
        stop = start + leg.sector_full_dim(sector)
        slices[sector] = slice(start, stop)
        start = stop
    return slices


class TensorFusionMap:
    """Sparse structural map from reduced MPS blocks to component tensors.

    The local physical leg and both bond Legs retain complete irreps. Storage
    scales with local tensor dimensions, not the Hilbert space of the chain.
    Reduced sweeps use the block layout; the sparse component map is built
    lazily for export/reference work. This adaptation follows Singh, Pfeifer and Vidal,
    Phys. Rev. A 82, 050301 (2010), doi:10.1103/PhysRevA.82.050301.
    """
    def __init__(self, left, physical, right):
        if left.symmetry != physical.symmetry or right.symmetry != left.symmetry:
            raise ValueError('Fusion Legs must have the same product symmetry.')
        self.legs = (left, physical, right)
        self.shape = tuple(leg.full_dim for leg in self.legs)
        self.blocks = {}
        offset = 0
        for a in left.sectors:
            for b in physical.sectors:
                for ccharge in left.symmetry.fuse(a.charge, b.charge):
                    c = Irrep(ccharge)
                    if c not in right.dims:
                        continue
                    multiplicities = (left.dims[a], physical.dims[b], right.dims[c])
                    size = int(np.prod(multiplicities))
                    self.blocks[a,b,c] = (slice(offset, offset+size), multiplicities)
                    offset += size
        self.parameter_count = offset

    @cached_property
    def matrix(self):
        # Reduced sweeps only need block layouts. Build the component map on
        # demand for export and explicit component-reference contractions.
        slices = [full_slices(leg) for leg in self.legs]
        rows, columns, values = [], [], []
        for key,(span,multiplicities) in self.blocks.items():
            cg = irrep_coupling(self.legs[0].symmetry, *(q.charge for q in key))
            nz = np.argwhere(abs(cg) > 1e-14)
            indices = np.indices(multiplicities).reshape(3,-1).T
            full = tuple(slices[axis][sector].start + indices[:,axis,None]*cg.shape[axis]+nz[None,:,axis]
                         for axis,sector in enumerate(key))
            rows.append(np.ravel_multi_index(full,self.shape).ravel())
            columns.append(np.repeat(np.arange(span.start,span.stop),len(nz)))
            values.append(np.tile(cg[tuple(nz.T)],len(indices)))
        if not rows:
            return coo_matrix((int(np.prod(self.shape)),self.parameter_count)).tocsr()
        return coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),
                          shape=(int(np.prod(self.shape)),self.parameter_count)).tocsr()

    def unpack(self, vector):
        """Block views of a reduced vector, without component expansion."""
        return {key: np.asarray(vector[span]).reshape(shape)
                for key,(span,shape) in self.blocks.items()}

    def expand(self, vector):
        return (self.matrix @ vector).reshape(self.shape)

    def tensor(self, vector):
        from pyqed.symmetry import IrrepTensor
        data = {key: np.asarray(vector[span]).reshape(shape).copy()
                for key,(span,shape) in self.blocks.items()}
        return IrrepTensor(data, self.legs, [-1,1,1], metadata={'fusion_map': self})

    def pack(self, tensor):
        return np.concatenate([tensor.data[key].ravel() for key in self.blocks])


@lru_cache(maxsize=2048)
def fusion_recoupling(symmetry, a, b, c, total):
    """Unitary change of reduced channels from ((a b) c) to (a (b c))."""
    left = tuple(q for q in symmetry.fuse(a,b) if total in symmetry.fuse(q,c))
    right = tuple(q for q in symmetry.fuse(b,c) if total in symmetry.fuse(a,q))
    lmaps = [np.einsum('abx,xct->abct',irrep_coupling(symmetry,a,b,q),
                      irrep_coupling(symmetry,q,c,total)) for q in left]
    rmaps = [np.einsum('bcx,axt->abct',irrep_coupling(symmetry,b,c,q),
                      irrep_coupling(symmetry,a,q,total)) for q in right]
    matrix = np.array([[np.vdot(r,l)/l.shape[-1] for l in lmaps] for r in rmaps])
    matrix.setflags(write=False)
    return left,right,matrix


def move_multiplet_center(tensors, site, direction):
    """Exact sector-wise QR/RQ gauge move; never split an irrep's rows.

    Right normalization weights a reduced block by sqrt(dim(right)/dim(left)),
    the partial trace of its CG isometry. Only multiplicities change.
    """
    from pyqed.symmetry import IrrepTensor
    neighbor = site+1 if direction=='lr' else site-1
    source, adjacent = tensors[site],tensors[neighbor]
    axis = 2 if direction=='lr' else 0
    old_leg = source.legs[axis]
    updated, transfer, dims = {},{},{}
    for sector in old_leg.sectors:
        keys = [key for key in source.data if key[axis]==sector]
        pieces, weights = [],[]
        for key in keys:
            block = source.data[key]
            if direction=='lr':
                pieces.append(block.reshape(-1,block.shape[2])); weights.append(1.)
            else:
                weight = np.sqrt(Leg._irrep_dimension(old_leg.symmetry,key[2].charge)/
                                 Leg._irrep_dimension(old_leg.symmetry,key[0].charge))
                pieces.append(weight*block.reshape(block.shape[0],-1)); weights.append(weight)
        matrix = np.concatenate(pieces,axis=0 if direction=='lr' else 1)
        q,r = np.linalg.qr(matrix if direction=='lr' else matrix.T,mode='reduced')
        if direction=='rl':
            q,r = q.T,r.T
        dims[sector] = q.shape[1] if direction=='lr' else q.shape[0]
        transfer[sector] = r
        offset = 0
        for key,piece,weight in zip(keys,pieces,weights):
            shape = list(source.data[key].shape)
            shape[axis] = dims[sector]
            length = piece.shape[0] if direction=='lr' else piece.shape[1]
            values = q[offset:offset+length] if direction=='lr' else q[:,offset:offset+length]/weight
            updated[key] = values.reshape(shape)
            offset += length
    new_leg = Leg(dims,symmetry=old_leg.symmetry,direction=old_leg.direction)
    legs = list(source.legs); legs[axis] = new_leg
    adjacent_legs = list(adjacent.legs); adjacent_legs[0 if direction=='lr' else 2] = new_leg.dual()
    adjacent_data = {key: (np.einsum('ab,bpc->apc',transfer[key[0]],block) if direction=='lr'
                          else np.einsum('apb,bc->apc',block,transfer[key[2]]))
                     for key,block in adjacent.data.items()}
    for index,data,new_legs in ((site,updated,legs),(neighbor,adjacent_data,adjacent_legs)):
        metadata = tensors[index].metadata.copy()
        old_map = metadata['fusion_map']
        if not all(old.same_blocks(new) for old,new in zip(old_map.legs,new_legs)):
            metadata['fusion_map'] = TensorFusionMap(*new_legs)
        tensors[index] = IrrepTensor(data,new_legs,[-1,1,1],metadata=metadata)
    return tensors
