"""AO-to-MO transformations of molecular electron-repulsion integrals."""

from pyqed.parallel import resolve_workers

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
import warnings

try:
    from . import _ao2mo
except ImportError:
    _ao2mo = None


def _compiled_blas_kernel(name):
    kernel = None if _ao2mo is None else getattr(_ao2mo, name, None)
    if _ao2mo is not None and getattr(_ao2mo, 'has_blas', False) and callable(kernel):
        return kernel
    reason = ('extension could not be loaded' if _ao2mo is None else
              'build has no compiled BLAS support' if not getattr(_ao2mo, 'has_blas', False) else
              f'build is missing {name}')
    warnings.warn(
        f'AO-to-MO compiled BLAS kernel {name} is unavailable ({reason}); '
        'using the NumPy fallback. Install or rebuild PyQED with '
        'compiled BLAS support to enable acceleration.',
        RuntimeWarning, stacklevel=3,
    )
    return None


def _transform_dense_general(eri, coefficients):
    """Four one-index products for complex or unequal orbital spaces."""
    left, right, left2, right2 = coefficients
    nao = left.shape[0]
    tmp = eri @ right2
    tmp = left2.conj().T @ tmp
    tmp = left.conj().T @ tmp.reshape(nao, -1)
    return (right.T @ tmp.reshape(left.shape[1], nao, left2.shape[1]*right2.shape[1])).reshape(
        left.shape[1], right.shape[1], left2.shape[1], right2.shape[1])


def ao2mo_dense(eri, mo_coeff):
    """Return dense chemists' MO ERIs for one coefficient matrix.

    Real AO ERIs must have eightfold permutation symmetry. The real compiled
    path transforms independent AO and MO pairs in two half transformations,
    each using two matrix products. This is an exact change of basis without
    screening or truncation. Complex inputs use four full one-index products,
    conjugating the first and third orbital coefficients.
    Real inputs use the shared pair-half fallback with a ``RuntimeWarning``
    if the compiled BLAS kernel is unavailable.
    """
    eri, c = np.asarray(eri), np.asarray(mo_coeff)
    if c.ndim != 2 or eri.shape != (c.shape[0],) * 4:
        raise ValueError("Expected dense (nao,nao,nao,nao) ERIs and (nao,nmo) coefficients.")
    nao, nmo = c.shape
    if nmo == 0:
        return np.empty((0,) * 4, dtype=np.result_type(eri, c))
    if not np.iscomplexobj(eri) and not np.iscomplexobj(c):
        kernel = _compiled_blas_kernel('dense_pair_transform')
        if kernel is not None:
            return kernel(eri, c)
        return _transform_pairs(eri, c, kernel=None, workers=1)
    return _transform_dense_general(eri, (c,c,c,c))


def ao2mo_packed(eri_s8, mo_coeff, *, workers=None):
    """Transform real s8 AO integrals using two independent-pair halves.

    Reuses the compiled packed half-transform when available; otherwise uses
    bounded batches of one-index matrix products. This is an exact change of
    basis, without fitting or screening. AO integrals remain packed, while the
    pair intermediates and returned MO tensor are in memory. This dense-output
    API is for existing CASSCF/general callers, not production molecular CC.
    With compiled BLAS, workers partition independent rows in sufficiently
    large spaces; keep BLAS at one thread per worker.
    Missing compiled BLAS support emits a ``RuntimeWarning`` before fallback.
    """
    workers = resolve_workers(workers)
    source, c = np.asarray(eri_s8), np.asarray(mo_coeff)
    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError('workers must be a positive integer.')
    if c.ndim != 2 or np.iscomplexobj(c) or np.iscomplexobj(source):
        raise ValueError('Expected real s8 integrals and real (nao,nmo) coefficients.')
    nao,nmo = c.shape
    ap,mp = nao*(nao+1)//2,nmo*(nmo+1)//2
    if nao < 1 or source.shape != (ap*(ap+1)//2,):
        raise ValueError('s8 integral shape is inconsistent with the coefficients.')
    if nmo == 0:
        return np.empty((0,)*4,dtype=np.float64)
    kernel = _compiled_blas_kernel('packed_half_transform')
    return _transform_pairs(source,c,kernel=kernel,workers=workers)


def _transform_pairs(source, c, *, kernel, workers):
    """Shared pair-half engine, including the real dense/packed fallback."""
    nao,nmo = c.shape
    ap,mp = nao*(nao+1)//2,nmo*(nmo+1)//2
    source,c = np.ascontiguousarray(source,dtype=float),np.ascontiguousarray(c,dtype=float)
    from .basis import _ao_pair_index_matrix, _packed_pair_pair_indices
    ao_pairs = _ao_pair_index_matrix(nao)
    ai,aj = np.tril_indices(nao)
    mi,mj = np.tril_indices(nmo)
    def transform_rows(data,start,stop):
        if kernel is not None:
            return kernel(data,c,start,stop)
        if data.ndim == 1:
            rows = np.arange(start,stop)[:,None,None]
            matrices = data[_packed_pair_pair_indices(rows,ao_pairs)]
        elif data.ndim == 2:
            matrices = data[start:stop,ao_pairs]
        else:
            matrices = data[ai[start:stop],aj[start:stop]]
        transformed = c.T @ (matrices @ c)
        return transformed[:,mi,mj]
    count = 1 if nao < 40 or kernel is None else min(int(workers),max(1,min(ap,mp)//64))
    pool = ThreadPoolExecutor(max_workers=count) if count > 1 else None
    try:
        def half(data,nrows):
            result = np.empty((nrows,mp))
            batch = max(16,(nrows+count*4-1)//(count*4)) if kernel is not None else 32
            if pool is None:
                for start in range(0,nrows,batch):
                    stop = min(nrows,start+batch)
                    result[start:stop] = transform_rows(data,start,stop)
            else:
                intervals = [(start,min(nrows,start+batch)) for start in range(0,nrows,batch)]
                futures = [pool.submit(transform_rows,data,start,stop) for start,stop in intervals]
                for (start,stop),future in zip(intervals,futures):
                    result[start:stop] = future.result()
            return result
        first = half(source,ap)
        rows = np.ascontiguousarray(first.T)
        del first
        pairs = half(rows,mp)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    ids = _ao_pair_index_matrix(nmo)
    return pairs[ids[:,:,None,None],ids[None,None,:,:]]
