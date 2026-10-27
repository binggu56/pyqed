"""Numerical product-operator construction in local irreducible channels.

Adapts tensor-train orthogonalization (Schollwoeck, Ann. Phys. 326,
96--192, 2011, doi:10.1016/j.aop.2010.09.012) to shared operator suffixes.
Temporary suffix coordinates carry local operator rows; the final MPO
is reduced by product-group CG maps. No molecular Hilbert basis is built.
"""
from functools import lru_cache
from tempfile import TemporaryFile
import numpy as np
from scipy.linalg import qr
from pyqed.qchem.dmrg.spatial_terms import (spatial_local_ops,
    spatial_one_body_term_map,spatial_two_body_spinfree_term_map,merge_term_maps)
from .multiplet_sites import OrbitalMultipletSite


_MAX_OPERATOR_ELEMENTS = 128*1024**2//16


def _check_workspace(shape):
    if np.prod(shape,dtype=object)>_MAX_OPERATOR_ELEMENTS:
        raise MemoryError('Operator construction workspace exceeds 128 MiB.')


def _suffix_storage(shape, dtype):
    """Anonymous disk storage for large range bases and coordinates; contraction blocks stay bounded."""
    elements=int(np.prod(shape,dtype=object))
    if elements<=_MAX_OPERATOR_ELEMENTS:
        return np.empty(shape,dtype=dtype)
    size=elements*np.dtype(dtype).itemsize
    if size>2*1024**3:
        raise MemoryError('Operator suffix storage exceeds the 2 GiB disk budget.')
    # Same anonymous mmap lifetime pattern as qchem.basis._ri_storage;
    # suffix coordinates also require complex dtype and column-contiguous reads.
    with TemporaryFile() as file:
        file.truncate(size)
        return np.memmap(file,mode='r+',dtype=dtype,shape=shape,order='F')


def _row_basis(matrix):
    """Orthonormal row span and coefficients, dropping numerical null rows."""
    q,r,p = qr(matrix.T,mode='economic',pivoting=True,check_finite=False)
    diagonal = abs(r.diagonal())
    rank = int(np.count_nonzero(diagonal>diagonal.max(initial=0)*1e-14))
    return q[:,:rank].T,r[:rank,np.argsort(p)].T


def _suffix_factor(transfer, coefficients, suffixes, following, *, batch_size=64):
    """Deterministic blocked range construction without a dense suffix unfolding.

    Twice-projected block QR finds the column range; a second pass forms its
    coordinates and verifies the residual. This adapts reorthogonalized block
    Gram--Schmidt (Bjorck, Numerical Methods for Least Squares Problems, SIAM,
    1996, doi:10.1137/1.9781611971484). It is not randomized compression.
    """
    rows = transfer.shape[0]*coefficients.shape[1]
    columns = len(following)
    width = min(batch_size,max(1,_MAX_OPERATOR_ELEMENTS//max(rows,1)))
    _check_workspace((rows,width))
    dtype = np.result_type(transfer,coefficients)
    groups = {}
    for key,source in suffixes.items():
        groups.setdefault(key[0],[]).append((following[key[1:]],source))
    groups = [(op,np.asarray(entries,dtype=int)) for op,entries in groups.items()]
    def block(start):
        stop = min(start+width,columns)
        value = np.zeros((transfer.shape[0],coefficients.shape[1],stop-start),dtype=dtype)
        for op,entries in groups:
            chosen = entries[(entries[:,0]>=start)&(entries[:,0]<stop)]
            if len(chosen):
                value[:,:,chosen[:,0]-start] += (transfer[:,chosen[:,1]][:,None,:]
                                               *coefficients[op][None,:,None])
        return value.reshape(rows,-1)
    basis = np.empty((rows,0),dtype=dtype)
    def project(value):
        if not isinstance(basis,np.memmap):
            return basis.conj().T@value
        result=np.empty((basis.shape[1],value.shape[1]),dtype=dtype)
        for first in range(0,basis.shape[1],width):
            part=basis[:,first:first+width]
            result[first:first+width]=part.conj().T@value
        return result
    def remove(value,coordinates=None):
        coordinates=project(value) if coordinates is None else coordinates
        if not isinstance(basis,np.memmap):
            value-=basis@coordinates
        else:
            for first in range(0,basis.shape[1],width):
                value-=basis[:,first:first+width]@coordinates[first:first+width]
    scale = 0.
    for start in range(0,columns,width):
        residual = block(start)
        scale = max(scale,float(np.linalg.norm(residual,axis=0).max(initial=0)))
        for _ in range(2):
            remove(residual)
        q,r,_ = qr(residual,mode='economic',pivoting=True,check_finite=False)
        rank = int(np.count_nonzero(abs(r.diagonal())>scale*1e-14))
        if rank:
            candidate = q[:,:rank]
            # QR normalizes small residual columns; remove amplified roundoff
            # along the existing range before admitting the new directions.
            for _ in range(2):
                remove(candidate)
            q,r,_ = qr(candidate,mode='economic',pivoting=True,check_finite=False)
            rank = int(np.count_nonzero(abs(r.diagonal())>1e-10))
            expanded=_suffix_storage((rows,basis.shape[1]+rank),dtype)
            expanded[:,:basis.shape[1]]=basis
            expanded[:,basis.shape[1]:]=q[:,:rank]
            basis=expanded
    coordinates = _suffix_storage((basis.shape[1],columns),dtype)
    for start in range(0,columns,width):
        value = block(start)
        projected = project(value)
        residual=value.copy();remove(residual,projected)
        if np.linalg.norm(residual)>1e-11*max(scale,np.linalg.norm(value)):
            raise ArithmeticError('Blocked suffix QR failed its reconstruction check.')
        coordinates[:,start:start+value.shape[1]] = projected
    return basis,coordinates


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


def hamiltonian_operator_train(h,v,sites,blocks,cutoff, *, factored=False):
    """Build an operator train directly from weighted fermionic terms.

    Adapts tensor-train QR orthogonalization from Schollwoeck, Ann. Phys.
    326, 96--192 (2011), doi:10.1016/j.aop.2010.09.012, to weighted
    prefixes with shared suffixes. Small unfoldings use pivoted QR; larger
    unfoldings use deterministic reorthogonalized block QR and a second-pass
    reconstruction check. QR drops relative null directions at 1e-14.
    Shared suffixes and local operator spans avoid symbolic MPO construction
    and physical component cores. A 128 MiB per-workspace guard bounds numerical blocks.
    Larger range bases and suffix coordinates use anonymous local disk (up to
    2 GiB each), projected and read in bounded column blocks; this is a finite-term builder, not a large-space
    complementary-operator algorithm. With ``factored=True``, orthonormal
    local spans stay separate through canonicalization and only individual
    symmetry channels are expanded during CG reduction.
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
        if np.prod(shape,dtype=object)>131072:
            basis,next_transfer = _suffix_factor(transfer,coefficients[i],suffixes,following)
        else:
            matrix = np.zeros(shape,dtype=transfer.dtype)
            for key,j in suffixes.items():
                matrix[:,:,following[key[1:]]] += transfer[:,j,None]*coefficients[i][key[0]][None,:]
            row_basis,coordinates = _row_basis(matrix.reshape(-1,len(following)).T)
            basis,next_transfer = row_basis.T,coordinates.T
        if not basis.shape[1]:return None
        core = basis.reshape(transfer.shape[0],len(spans[i]),-1)
        if factored:
            arrays.append(core)
        else:
            _check_workspace((core.shape[0],spans[i].shape[1],core.shape[2]))
            arrays.append(np.einsum('aqb,qp->apb',core,spans[i],optimize=True))
        transfer,suffixes = next_transfer,following
    _check_workspace((transfer.shape[0],len(spans[-1])))
    last = transfer @ coefficients[-1][[key[0] for key in suffixes]]
    if not factored:
        _check_workspace((transfer.shape[0],spans[-1].shape[1]))
        last = last @ spans[-1]
    arrays.append(last[:,:,None])
    return (arrays,spans) if factored else arrays
