"""Numerical product-operator construction in local irreducible channels.

Adapts tensor-train orthogonalization (Schollwoeck, Ann. Phys. 326,
96--192, 2011, doi:10.1016/j.aop.2010.09.012) to shared operator suffixes.
Temporary suffix coordinates carry local operator rows; the final MPO
is reduced by product-group CG maps. No molecular Hilbert basis is built.
"""
from functools import lru_cache
import numpy as np
from scipy.linalg import qr
from pyqed.qchem.dmrg.spatial_terms import (spatial_local_ops,
    spatial_one_body_term_map,spatial_two_body_spinfree_term_map,merge_term_maps)
from .multiplet_sites import OrbitalMultipletSite


_MAX_OPERATOR_ELEMENTS = 128*1024**2//16


def _check_workspace(shape):
    if np.prod(shape,dtype=object)>_MAX_OPERATOR_ELEMENTS:
        raise MemoryError('Operator construction workspace exceeds 128 MiB.')


def _row_basis(matrix):
    """Orthonormal row span and coefficients, dropping numerical null rows."""
    q,r,p = qr(matrix.T,mode='economic',pivoting=True,check_finite=False)
    diagonal = abs(r.diagonal())
    rank = int(np.count_nonzero(diagonal>diagonal.max(initial=0)*1e-14))
    return q[:,:rank].T,r[:rank,np.argsort(p)].T


@lru_cache(maxsize=1024)
def _operator_vector(group,label,symbols):
    from .multiplet_operator import _operator_basis
    site = OrbitalMultipletSite._template(group,label)
    local = _operator_basis(site.symmetry,tuple(site.leg.dims.items()))
    matrix = np.ones((1,1))
    for symbol in symbols:
        matrix = np.kron(matrix,spatial_local_ops()[symbol])
    vector = np.real_if_close(local.transform @ (site.basis.conj().T@matrix@site.basis).ravel())
    vector.setflags(write=False)
    return vector


def hamiltonian_operator_train(h,v,sites,blocks,cutoff):
    """Build an operator train directly from weighted fermionic terms.

    Adapts tensor-train QR orthogonalization from Schollwoeck, Ann. Phys.
    326, 96--192 (2011), doi:10.1016/j.aop.2010.09.012, to weighted
    prefixes with shared suffixes. QR drops relative null directions at 1e-14.
    Shared suffixes and local operator spans avoid symbolic MPO construction
    and physical component cores. A 128 MiB per-workspace guard bounds dense
    suffix compression; this is a finite-term builder, not a large-space
    complementary-operator algorithm.
    """
    h,v = np.asarray(h),np.asarray(v)
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or v.shape!=(len(h),)*4:
        raise ValueError('Expected square h and a matching four-index active-space ERI.')
    if (not sites or len(sites)!=len(blocks) or
            tuple(p for block in blocks for p in block)!=tuple(range(len(h))) or
            any(len(block)!=site.norb for site,block in zip(sites,blocks))):
        raise ValueError('Multiplet blocks must partition the complete orbital order.')
    if any(site.symmetry!=sites[0].symmetry for site in sites):
        raise ValueError('Multiplet sites must share a product symmetry.')
    terms = merge_term_maps(spatial_one_body_term_map(h,cutoff=cutoff),
        spatial_two_body_spinfree_term_map(v,cutoff=cutoff),cutoff=cutoff)
    if not terms:
        return None
    table,weights = [],[]
    lookup = [{} for _ in sites]
    for (symbols,dofs),weight in terms.items():
        tokens = ['I']*len(h)
        for symbol,dof in zip(symbols.split(),dofs):
            tokens[dof] = symbol
        row = []
        for i,block in enumerate(blocks):
            key = tuple(tokens[j] for j in block)
            if key not in lookup[i]:lookup[i][key] = len(lookup[i])
            row.append(lookup[i][key])
        table.append(tuple(row));weights.append(weight)
    spans,coefficients = [],[]
    for site,operators in zip(sites,lookup):
        vectors = np.stack([_operator_vector(site.group,site.orbital_irrep,key) for key in operators])
        span,coeff = _row_basis(vectors)
        spans.append(span);coefficients.append(coeff)
    suffixes = {key:j for j,key in enumerate(dict.fromkeys(table))}
    transfer = np.zeros((1,len(suffixes)),complex)
    for row,weight in zip(table,weights):transfer[0,suffixes[row]] += weight
    transfer = np.real_if_close(transfer)
    arrays = []
    for i in range(len(sites)-1):
        following = {key:j for j,key in enumerate(dict.fromkeys(key[1:] for key in suffixes))}
        shape = (transfer.shape[0],len(spans[i]),len(following))
        _check_workspace(shape)
        matrix = np.zeros(shape,dtype=transfer.dtype)
        for key,j in suffixes.items():
            matrix[:,:,following[key[1:]]] += transfer[:,j,None]*coefficients[i][key[0]][None,:]
        basis,next_transfer = _row_basis(matrix.reshape(-1,len(following)).T)
        if not len(basis):return None
        core = basis.T.reshape(transfer.shape[0],len(spans[i]),-1)
        _check_workspace((core.shape[0],spans[i].shape[1],core.shape[2]))
        arrays.append(np.einsum('aqb,qp->apb',core,spans[i],optimize=True))
        transfer,suffixes = next_transfer.T,following
    _check_workspace((transfer.shape[0],spans[-1].shape[1]))
    last = transfer @ coefficients[-1][[key[0] for key in suffixes]] @ spans[-1]
    arrays.append(last[:,:,None])
    return arrays
