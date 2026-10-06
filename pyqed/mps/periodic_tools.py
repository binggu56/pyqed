"""Shared circular schedules and periodic norm conditioning.

Circular sections follow Pippan, White and Evertz, PRB 81, 081103(R) (2010),
https://doi.org/10.1103/PhysRevB.81.081103. Separable norm gauges adapt
Rossini, Giovannetti and Fazio, J. Stat. Mech. P05021 (2011),
https://doi.org/10.1088/1742-5468/2011/05/P05021, Sec. 2.4.
The norm remains nontrivial. Positive factor floors make gauges invertible;
they do not truncate states or add a ridge to the physical norm/Hamiltonian.
"""
from operator import index
import numpy as np
from scipy.linalg import eigh


def circular_sections(count):
    """Three nonempty contiguous sections in circular site order."""
    count=index(count)
    if count<2: raise ValueError('a circular sweep requires at least two sites')
    return [[int(i) for i in s] for s in np.array_split(np.arange(count),min(3,count))]


def metric_whitener(metric, *, rtol=1e-11, negative_rtol=1e-9):
    """Whiten retained positive norm support, reporting its condition number."""
    metric=np.asarray(metric)
    values,vectors=eigh((metric+metric.conj().T)/2)
    scale=max(float(values[-1]),np.finfo(float).tiny)
    if values[0]<-negative_rtol*scale:
        raise FloatingPointError('periodic norm metric is indefinite')
    keep=values>rtol*scale
    if not np.any(keep): raise FloatingPointError('empty periodic metric support')
    return vectors[:,keep]/np.sqrt(values[keep]),dict(rank=int(np.sum(keep)),
        retained_condition=float(values[-1]/values[keep][0]),largest_eigenvalue=float(values[-1]))


def norm_factors(metric, shape):
    """Positive factors of the leading virtual operator-Schmidt norm term.

    Adaptation of Rossini et al. (2011), Sec. 2.4, reference above. Trace out
    physical multiplicity, reshape the virtual pairs and take the leading SVD
    term. If degeneracy gives nonpositive factors, use positive partial traces
    instead. These factors condition coordinates; the full metric is retained.
    """
    left,physical,right=shape
    virtual=np.einsum('apbcpd->abcd',np.asarray(metric).reshape(shape+shape))/physical
    reshuffled=virtual.transpose(0,2,1,3).reshape(left*left,right*right)
    u,s,vh=np.linalg.svd(reshuffled,full_matrices=False)
    if s[0]<=0: raise FloatingPointError('null virtual norm')
    l=u[:,0].reshape(left,left)*np.sqrt(s[0])
    r=vh[0].reshape(right,right)*np.sqrt(s[0])
    trace=np.trace(l)
    phase=trace/abs(trace) if abs(trace)>np.finfo(float).eps*np.linalg.norm(l) else 1.
    l=l/phase; r=r*phase
    l=(l+l.conj().T)/2; r=(r+r.conj().T)/2
    fallback=any(np.linalg.eigvalsh(a)[0]<-1e-10*max(np.linalg.norm(a),np.finfo(float).tiny)
                 or np.trace(a).real<=0 for a in (l,r))
    if fallback:
        l=np.einsum('abcb->ac',virtual); r=np.einsum('abad->bd',virtual)
    l=l/(np.trace(l).real/left); r=r/(np.trace(r).real/right)
    return l,r,dict(relative_factorization_tail=float(np.linalg.norm(s[1:])/np.linalg.norm(s)),
                    partial_trace_fallback=bool(fallback))


def matrix_roots(matrix, *, floor=1e-8):
    """Bounded positive square root and inverse; no directions are removed."""
    if not 0<floor<1: raise ValueError('gauge floor must lie between zero and one')
    values,vectors=eigh((matrix+matrix.conj().T)/2)
    if values[-1]<=0: raise FloatingPointError('null positive gauge factor')
    values=np.maximum(values,values[-1]*floor)
    root=(vectors*np.sqrt(values))@vectors.conj().T
    inverse=(vectors/np.sqrt(values))@vectors.conj().T
    return root,inverse


def compress_transfer(apply, adjoint, shape, *, dtype=complex,
                      tolerance=1e-10, max_rank=None, seed=0):
    """Adaptive matrix-free SVD of a possibly rectangular transfer product.

    Adapted from Pippan, White and Evertz, PRB 81, 081103(R) (2010),
    https://doi.org/10.1103/PhysRevB.81.081103. Randomized range finding,
    one subspace iteration and independent probe checks are additions. The
    residual is an estimate, not an error bound. A missed rank cap is reported.
    No physical-state basis is constructed. Zero tolerance uses numerical
    full accuracy; rectangular reduced-channel products are supported.
    """
    rows,columns=shape; limit=min(rows,columns)
    if limit<1 or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('transfer dimensions must be positive and tolerance nonnegative')
    if max_rank is not None and (index(max_rank)<1):
        raise ValueError('max_rank must be a positive integer')
    cap=limit if max_rank is None else min(limit,index(max_rank))
    rng=np.random.default_rng(seed)
    def random(size,width):
        value=rng.normal(size=(size,width))
        if np.issubdtype(np.dtype(dtype),np.complexfloating):
            value=(value+1j*rng.normal(size=value.shape))/np.sqrt(2)
        return value
    probes=random(columns,min(8,columns)); reference=apply(probes)
    scale=max(np.linalg.norm(reference),np.finfo(float).tiny)
    rank=min(4,cap)
    while True:
        width=min(limit,rank+4)
        if width==limit:
            if rows<=columns:
                q=np.eye(rows,dtype=dtype)
                matrix=adjoint(q).conj().T
            else:
                matrix=apply(np.eye(columns,dtype=dtype))
                q=None
        else:
            q,_=np.linalg.qr(apply(random(columns,width)),mode='reduced')
            z,_=np.linalg.qr(adjoint(q),mode='reduced')
            q,_=np.linalg.qr(apply(z),mode='reduced')
            matrix=adjoint(q).conj().T
        u,s,vh=np.linalg.svd(matrix,full_matrices=False)
        if q is not None: u=q@u
        left,right=u[:,:rank]*s[:rank],vh[:rank]
        error=float(np.linalg.norm(reference-left@(right@probes))/scale)
        met=error<=max(tolerance,100*np.finfo(float).eps)
        if met or rank==cap:
            return left,right,dict(rank=rank,relative_error_estimate=error,
                tolerance_met=bool(met),full_dimension=limit,
                input_dimension=columns,output_dimension=rows,
                factor_entries=int(left.size+right.size),dense_entries=int(rows*columns))
        rank=min(cap,2*rank)


def compress_factor_product(left, right, *, tolerance=1e-10, max_rank=None):
    """Compress a factored transfer through QR and a small SVD core.

    Adaptation of Pippan et al. (2010), DOI in :func:`compress_transfer`,
    exploiting an already known transfer bottleneck instead of randomized
    range finding. The reported tail is the relative Frobenius norm of the
    discarded singular values, excluding floating-point contraction errors.
    """
    if not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('compression tolerance must be finite and nonnegative')
    if max_rank is not None and index(max_rank)<1:
        raise ValueError('max_rank must be a positive integer')
    ql,rl=np.linalg.qr(left,mode='reduced')
    qr,rr=np.linalg.qr(right.conj().T,mode='reduced')
    u,s,vh=np.linalg.svd(rl@rr.conj().T,full_matrices=False)
    tails=np.sqrt(np.r_[np.cumsum(s[::-1]**2)[::-1],0.])
    scale=max(float(tails[0]),np.finfo(float).tiny)
    eligible=np.flatnonzero(tails[1:]<=tolerance*scale)
    requested=int(eligible[0]+1) if len(eligible) else len(s)
    rank=requested if max_rank is None else min(requested,index(max_rank))
    error=float(tails[rank]/scale)
    rows,columns=left.shape[0],right.shape[1]
    return (ql@u[:,:rank])*s[:rank],vh[:rank]@qr.conj().T,dict(
        rank=rank,relative_error_estimate=error,relative_frobenius_tail=error,
        tolerance_met=bool(error<=max(tolerance,100*np.finfo(float).eps)),
        full_dimension=min(rows,columns),input_dimension=columns,output_dimension=rows,
        bottleneck_dimension=int(left.shape[1]),factor_entries=int(rank*(rows+columns)),
        dense_entries=int(rows*columns),factorization='seam_qr_svd')
