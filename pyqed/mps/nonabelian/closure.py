"""Exact spin-zero multiplicity-loop lifting for reduced SU(2) tensors.

Adapts the periodic trace construction of Verstraete, Porras and Cirac,
PRL 93, 227205 (2004), doi:10.1103/PhysRevLett.93.227205, by carrying the
cut multiplicity index as a spectator through open reduced environments.
Only a spin-zero seam with fixed total particle charge is supported. There
is no magnetic expansion, variational projection, or truncation in this lift.
"""
import numpy as np
from pyqed.symmetry import IrrepTensor


def closure_scatter(shape, site, count, dimension, *, indices=None):
    """Map selected raw entries to the exact open contraction of a closed loop.

    Supplying structural route indices avoids allocating a map for every
    entry of a mostly zero frontier block.
    """
    left, physical, right = shape
    indices = (np.arange(left*physical*right) if indices is None
               else np.asarray(indices, dtype=np.intp))
    l,p,r = np.unravel_index(indices,shape)
    if site==0:
        if left!=dimension: raise ValueError('closing left multiplicity differs from D')
        target_shape = (1,physical,dimension*right)
        targets = np.ravel_multi_index((np.zeros_like(l),p,l*right+r),target_shape)
        return indices,targets,target_shape
    if site==count-1:
        if right!=dimension: raise ValueError('closing right multiplicity differs from D')
        target_shape = (dimension*left,physical,1)
        targets = np.ravel_multi_index((r*left+l,p,np.zeros_like(r)),target_shape)
        return indices,targets,target_shape
    target_shape = (dimension*left,physical,dimension*right)
    anchors = np.arange(dimension)[:,None]
    targets = ((anchors*left+l)*physical+p)*(dimension*right)+anchors*right+r
    return np.tile(indices,dimension),targets.ravel(),target_shape


def lift_closure(tensor, site, count, dimension):
    """Unfold a spin-zero virtual trace into reduced spectator-index blocks."""
    data = {}
    for key,block in tensor.data.items():
        source,target,shape = closure_scatter(block.shape,site,count,dimension)
        lifted = np.zeros(shape,dtype=block.dtype)
        lifted.ravel()[target] = block.ravel()[source]
        data[key] = lifted
    qns = [list(q) for q in tensor.qns]
    qns[0] = qns[0][:1] if site==0 else qns[0]*dimension
    qns[2] = qns[2][:1] if site==count-1 else qns[2]*dimension
    return IrrepTensor(data=data,qns=qns,dirs=list(tensor.dirs),
                       metadata={**tensor.metadata,'virtual_boundary':'periodic',
                                 'closure_spin':0,'closure_dimension':dimension})
