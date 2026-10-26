"""Disk-backed first responses and direct mixed GW/BSE Hessian entries.

Two cached first directions and one mixed second block are propagated at a
time. Polarization remains an explicit validation/benchmark reference. This
changes evaluation/storage order only; the method approximations and primary
references are those in response.py.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from tempfile import TemporaryDirectory

import logging
import mmap

import numpy as np

_store = ContextVar('response_store', default=None)
_reference = ContextVar('response_reference', default=None)
_pair_columns = ContextVar('response_pair_columns', default=None)
_first_columns = ContextVar('response_first_columns', default=None)
_ARRAY_MEMORY_BYTES = 65536


class _DiskArray:
    def __init__(self, path, offset, array):
        self.path, self.offset = path, offset
        self.shape, self.dtype = array.shape, array.dtype

    def read(self):
        return np.memmap(self.path, mode='r', offset=self.offset,
                         shape=self.shape, dtype=self.dtype)


class _LinearArray:
    def __init__(self, arrays):
        self.arrays = arrays

    def read(self):
        value = np.array(read(self.arrays[0]), copy=True)
        for array in self.arrays[1:]:
            value += read(array)
        return value


class _StackArray:
    def __init__(self, arrays):
        self.arrays = arrays

    def read(self):
        return np.concatenate([read(array) for array in self.arrays], axis=0)


class ResponseStore:
    """Own temporary first-response data outside the repository."""
    def __init__(self):
        self.directory = TemporaryDirectory(prefix='pyqed-response-')
        self.path = str(Path(self.directory.name)/'arrays.bin')

    def __deepcopy__(self, memo):
        # Arrays are immutable; a detached scanner may share their lifetime.
        return self

    def put(self, value):
        # Full mappings returned by read() already belong to a live store.
        if isinstance(value, np.memmap) and isinstance(value.base, mmap.mmap):
            return _DiskArray(str(value.filename), value.offset, value)
        if isinstance(value, np.ndarray) and value.nbytes > _ARRAY_MEMORY_BYTES:
            with open(self.path, 'ab') as stream:
                offset = stream.tell()
                np.asarray(value).tofile(stream)
            return _DiskArray(self.path, offset, value)
        if isinstance(value, dict):
            return {k: self.put(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return (list if isinstance(value, list) else tuple)(self.put(v) for v in value)
        return value


@contextmanager
def storage(store):
    token = _store.set(store)
    try:
        yield
    finally:
        _store.reset(token)


@contextmanager
def first_columns(value):
    token = _first_columns.set(value)
    try:
        yield
    finally:
        _first_columns.reset(token)


def keep(value):
    store = _store.get()
    return value if store is None else store.put(value)


def read(value):
    if isinstance(value, (_DiskArray, _LinearArray, _StackArray)):
        return value.read()
    if isinstance(value, dict):
        return {k: read(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return (list if isinstance(value, list) else tuple)(read(v) for v in value)
    return value


@contextmanager
def reference_scope(cache, store):
    token = _reference.set((cache, {}, store))
    try:
        yield
    finally:
        _reference.reset(token)


@contextmanager
def reference_stage(name):
    context = _reference.get()
    if context is None:
        yield
        return
    cache, counts, store = context
    index = counts.get(name, 0)
    counts[name] = index + 1
    with reference_scope(cache.setdefault((name, index), {}), store):
        yield


def reference_value(name, compute):
    """Share only zeroth-order data within a fingerprinted response request."""
    context = _reference.get()
    if context is None:
        return compute()
    cache, _, store = context
    if name not in cache:
        with storage(store):
            cache[name] = keep(compute())
    return read(cache[name])


def combine(caches, *, stack=False):
    """Stack or superpose first derivatives, sharing reference-only arrays."""
    if len(caches) == 1:
        # New Hessian-only entries must not accumulate in the gradient cache.
        return _copy_tree(caches[0])
    merge = _StackArray if stack else _LinearArray
    def walk(nodes, path=()):
        output = {}
        for key, value in nodes[0].items():
            values = [node[key] for node in nodes]
            if key == 'key':
                continue
            if key in ('hcore', 'overlap', 'eri', 'coefficients') and not path:
                output[key] = merge(values)
            elif key == 'factor_first' and not path:
                output[key] = (tuple([item for v in values for item in v[i]] for i in range(2))
                               if stack else tuple([_LinearArray([v[i][0] for v in values])]
                                                   for i in range(2)))
            elif key == 'orbital_first' and not path:
                dynamic = {'amplitudes', 'h_vo', 'overlap_mo', 'full_response',
                           'density_response', 'fock_total_ao', 'fock_total_mo',
                           'orbital_energy_response', 'occupied_energy_response'}
                first = {k: merge([v[0][k] for v in values]) if k in dynamic else item
                         for k,item in value[0].items()}
                output[key] = (first, value[1], merge([v[2] for v in values]))
            elif key == 'data':
                name = path[-1][0]
                if name in ('pole_basis', 'frequency_slope'):
                    output[key] = value
                elif name == 'qp_first':
                    output[key] = merge(values)
                else:
                    output[key] = (value[0], merge([v[1] for v in values]), *value[2:])
            elif isinstance(value, dict):
                output[key] = walk(values, path+(key,))
            else:
                raise ValueError(f'Unknown first-response cache entry: {path+(key,)}')
        return output
    return walk(caches)


def _copy_tree(value):
    if isinstance(value, dict):
        return {k: _copy_tree(v) for k,v in value.items()}
    return value


def _retain_new_first(target, source):
    """Move Hessian-only first response into the persistent direction cache."""
    def materialize(value):
        if isinstance(value, dict):
            return {k: materialize(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(materialize(v) for v in value)
        value = read(value)
        return np.array(value, copy=True) if isinstance(value, np.ndarray) else value
    for key, value in source.items():
        if key not in target:
            target[key] = keep(materialize(value))
        elif isinstance(value, dict):
            _retain_new_first(target[key], value)


def directional_derivatives(bse, ids, modes, backend, groups, budget, tolerance, order, *, method="polarization"):
    """Evaluate individual gradients and selected Hessian pairs with disk caches.

    ``mixed`` propagates one cross derivative and retains Hessian-only first
    frequency responses after diagonal passes. ``polarization`` preserves the
    former summed-direction algorithm for numerical and timing comparisons.
    """
    from .response import _derivatives, _response_key, mixed_response
    if method not in ('mixed', 'polarization'):
        raise ValueError('Unknown directional Hessian method.')
    modes = np.asarray(modes).reshape(len(modes), -1)
    key = _response_key(bse, modes, backend, groups)
    previous = getattr(bse, '_first_response_cache', None)
    if previous is not None and previous.get('key') == key and 'directions' in previous:
        cache = previous
    else:
        cache = dict(key=key, store=ResponseStore(), directions={}, reference={})
    n, ns = len(modes), len(ids)
    gradient = np.empty((ns,n))
    arguments = dict(state_ids=ids, backend=backend, max_memory=budget,
                     degeneracy_tol=tolerance, _bounded=True)
    try:
        for i in range(n):
            logging.getLogger(__name__).info('First response direction %d/%d', i+1, n)
            bse._first_response_cache = cache['directions'].get(i)
            with storage(cache['store']), reference_scope(cache['reference'], cache['store']):
                _derivatives(bse, modes=modes[i:i+1], order=1, **arguments)
            cache['directions'][i] = bse._first_response_cache
            gradient[:,i] = bse.energy_gradient[:,0]
        if order == 1:
            bse.energy_gradient, bse.energy_hessian = gradient, None
            bse.derivative_info = dict(order=1, state_ids=ids.tolist(), modes=modes.copy(),
                                       storage='disk-backed directional response')
            return gradient
        hessian = np.empty((ns,n,n))
        f, g = np.empty((ns,ns,n)), np.empty((ns,ns,n,n))
        trace = np.empty((n,n))
        pairs = ([(i,i) for i in range(n)] + [(i,j) for i in range(n) for j in range(i+1,n)]
                 if method == 'mixed' else [(i,j) for i in range(n) for j in range(i,n)])
        for (i,j), columns in _batched_columns(bse, modes, pairs, method, budget):
            logging.getLogger(__name__).info('%s second response pair (%d,%d) / %d modes', method, i+1, j+1, n)
            indices = [i] if i == j else [i,j]
            mixed = method == 'mixed' and i != j
            direction = modes[indices] if mixed else np.sum(modes[indices], axis=0, keepdims=True)
            local = combine([cache['directions'][k] for k in indices], stack=mixed)
            local['key'] = _response_key(bse, direction, backend, groups)
            bse._first_response_cache = local
            scratch = ResponseStore()
            token = _pair_columns.set(columns)
            try:
                with storage(scratch), reference_scope(cache['reference'], cache['store']):
                    with mixed_response() if mixed else nullcontext():
                        _derivatives(bse, modes=direction, order=2, **arguments)
            finally:
                _pair_columns.reset(token)
            if i == j:
                f[:,:,i] = bse.F[:,:,0]
                if method == 'mixed':
                    with storage(cache['store']):
                        _retain_new_first(cache['directions'][i], local)
            hessian[:,i,j] = hessian[:,j,i] = bse.energy_hessian[:,0,0]
            g[:,:,i,j] = g[:,:,j,i] = bse.G[:,:,0,0]
            trace[i,j] = trace[j,i] = bse.subspace_hessian[0,0]
            bse._first_response_cache = None
            del local, scratch
        if method == 'polarization':
            for i in range(n):
                for j in range(i+1,n):
                    hessian[:,i,j] = hessian[:,j,i] = (hessian[:,i,j]-hessian[:,i,i]-hessian[:,j,j])/2
                    g[:,:,i,j] = g[:,:,j,i] = (g[:,:,i,j]-g[:,:,i,i]-g[:,:,j,j])/2
                    trace[i,j] = trace[j,i] = (trace[i,j]-trace[i,i]-trace[j,j])/2
        bse.energy_gradient, bse.energy_hessian = gradient, hessian
        bse.F, bse.G = f, g
        bse.subspace_gradient = np.trace(f,axis1=0,axis2=1)
        bse.subspace_hessian = trace
        bse.derivative_info.update(state_ids=ids.tolist(), modes=modes.copy(), hessian_method=method,
                                   storage='disk-backed directional response',
                                   retained_subspace_trace_hessian=trace.tolist())
        return f,g
    finally:
        bse._first_response_cache = cache


def _batched_columns(bse, modes, pairs, method, budget):
    """Stream bounded groups of first or mixed CD columns through temporary storage."""
    from pyqed.qchem.eri_response import CoulombIntegrals
    from pyqed.qchem import basis as gaussian
    data = getattr(bse.mol, '_cd_factor_data', None)
    if data is None:
        for pair in pairs:
            yield pair, None
        return
    workers = gaussian._builtin_worker_count(bse.mol, bse.mol.nao)
    size = len(data['pivots'])*data['nao']*(data['nao']+1)//2*8
    # First columns share one output across workers; second columns retain
    # private reductions. Leave the rest of the budget to electronic tensors.
    limit = min(1024*1024**2, int(budget*1e6/4))
    copies = 1 if method == 'first' else workers+1
    count = min(2 if method == 'first' else 8, limit//max(1,copies*size))
    if count < 2:
        for pair in pairs:
            yield pair, None
        return
    integrals = CoulombIntegrals(bse.mol)
    logging.getLogger(__name__).info('CD %s batching: %d directions/pairs, %.1f MiB output-buffer allowance',
                                     method, count, limit/1024**2)
    for start in range(0,len(pairs),count):
        batch = pairs[start:start+count]
        if method == 'first':
            directions = modes[[i for i, _ in batch]]
        elif method == 'mixed':
            indices = sorted({i for pair in batch for i in pair})
            directions = modes[indices]
            selected = np.array([[indices.index(i),indices.index(j)] for i,j in batch])
        else:
            directions = np.array([modes[i] if i == j else modes[i]+modes[j] for i,j in batch])
            selected = np.array([[i,i] for i in range(len(batch))])
        store = ResponseStore()
        values = (integrals._cd_columns(directions, order=1, workers=workers) if method == 'first' else
                  integrals._cd_columns(directions, order=2, pairs=selected, workers=workers)[:, 0])
        saved = [store.put(values[i]) for i in range(len(batch))]
        del values
        for pair,value in zip(batch,saved):
            yield pair,value
