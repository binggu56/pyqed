"""Analytic second-order response of restricted molecular G0W0/BSE.

Forward differentiation of the spectral equations, not finite differences or
the Z-vector implementation of Tölle, Kitsaras and Loos, JPCL 16, 11134
(2025), doi:10.1021/acs.jpclett.5c02219. See ``BSE.vibronic_couplings`` for
the effective-Hamiltonian frame and supported approximations.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import numpy as np

from .response_storage import keep, read, reference_stage, reference_value


# Each stage owns an ordered set of first-order intermediates. Nested stages
# have independent cursors, so Hessian-only branches cannot shift other stages.
_response_context = ContextVar('gw_bse_response', default=None)


@contextmanager
def _response_scope(cache, name=None):
    token = _response_context.set((cache, {}))
    try:
        with reference_stage(name):
            yield
    finally:
        _response_context.reset(token)


def _response_slot(name):
    context = _response_context.get()
    if context is None:
        return None
    cache, counts = context
    index = counts.get(name, 0)
    counts[name] = index + 1
    return cache.setdefault((name, index), {})


def _cached_first(name, compute):
    slot = _response_slot(name)
    with reference_stage(name):
        if slot is None:
            return compute()
        if 'data' not in slot:
            slot['data'] = keep(compute())
    return read(slot['data'])


def _response_stage(function):
    @wraps(function)
    def evaluate(*args, **kwargs):
        slot = _response_slot(function.__name__)
        if slot is None:
            return function(*args, **kwargs)
        with _response_scope(slot, function.__name__):
            return function(*args, **kwargs)
    return evaluate


def _detached(obj):
    from copy import deepcopy
    import sys
    streams = [sys.stdout,sys.stderr]
    for item in (obj,getattr(obj,'_scf',None),getattr(obj,'mol',None),getattr(obj,'gw',None)):
        stream = getattr(item,'stdout',None)
        if stream is not None:
            streams.append(stream)
    return deepcopy(obj,{id(stream):stream for stream in streams})


_mixed_response = ContextVar('mixed_response', default=False)


@contextmanager
def mixed_response():
    token = _mixed_response.set(True)
    try:
        yield
    finally:
        _mixed_response.reset(token)


def _second_shape(n):
    return (1, 1) if _mixed_response.get() else (n, n)


def _second_entries(n):
    if _mixed_response.get():
        if n != 2:
            raise ValueError('A mixed response requires exactly two first directions.')
        return [(0, 0, 0, 1)]
    return [(x, y, x, y) for x in range(n) for y in range(n)]


def _second_array(n, function):
    values = np.asarray([function(a,b,x,y) for a,b,x,y in _second_entries(n)])
    return values.reshape(*_second_shape(n), *values.shape[1:])


class _Jet:
    """Value and ordinary (not factorial-scaled) first/mixed derivatives."""

    __array_priority__ = 1000

    def __init__(self, value, first, second):
        self.v, self.d, self.h = map(np.asarray, (value, first, second))
        self.n = len(self.d)

    @classmethod
    def constant(cls, value, n):
        value = np.asarray(value)
        return cls(value, np.zeros((n,) + value.shape),
                   np.zeros(_second_shape(n) + value.shape))

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        return _Jet(self.v[key], self.d[(slice(None),) + key],
                    self.h[(slice(None), slice(None)) + key])

    def reshape(self, *shape):
        return _Jet(self.v.reshape(*shape), self.d.reshape(self.n, *shape),
                    self.h.reshape(*_second_shape(self.n), *shape))

    def __add__(self, other):
        other = other if isinstance(other, _Jet) else _Jet.constant(other, self.n)
        value, first = _cached_first('add', lambda: (self.v+other.v, np.array([self.d[x]+other.d[x] for x in range(self.n)])))
        return _Jet(value, first,
                    self.h + other.h)

    __radd__ = __add__

    def __neg__(self):
        value, first = _cached_first('neg', lambda: (-self.v, -self.d))
        return _Jet(value, first, -self.h)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, _Jet) else _Jet.constant(other, self.n)
        value, first = _cached_first('mul',
            lambda: (self.v*other.v, np.array([self.d[x]*other.v+self.v*other.d[x] for x in range(self.n)])))
        return _Jet(value, first,
            _second_array(self.n, lambda a,b,x,y: self.h[a,b]*other.v + self.v*other.h[a,b]
                          + self.d[x]*other.d[y] + self.d[y]*other.d[x]))

    __rmul__ = __mul__

    def __pow__(self, power):
        if power != int(power) and np.any(self.v <= 0):
            raise np.linalg.LinAlgError('Response requires positive spectral gaps.')
        v, first = _cached_first('power',
            lambda: (self.v**power, power*self.v**(power-1)*self.d))
        a, b = power*self.v**(power-1), power*(power-1)*self.v**(power-2)
        return _Jet(v, first,
                    _second_array(self.n, lambda i,j,x,y: a*self.h[i,j] + b*self.d[x]*self.d[y]))

    def __truediv__(self, other):
        if isinstance(other, _Jet):
            return self * other**-1
        return self * (1/other)


class _FirstJet(_Jet):
    """First-order response with no second-derivative allocation."""

    def __init__(self, value, first):
        self.v, self.d = np.asarray(value), np.asarray(first)
        self.n, self.h = len(self.d), None

    @classmethod
    def constant(cls, value, n):
        value = np.asarray(value)
        return cls(value, np.zeros((n,) + value.shape))

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        return _FirstJet(self.v[key], self.d[(slice(None),) + key])

    def reshape(self, *shape):
        return _FirstJet(self.v.reshape(*shape), self.d.reshape(self.n, *shape))

    def __add__(self, other):
        other = other if isinstance(other, _FirstJet) else self.constant(other, self.n)
        return _FirstJet(*_cached_first('add', lambda: (self.v+other.v, np.array([self.d[x]+other.d[x] for x in range(self.n)]))))

    __radd__ = __add__

    def __neg__(self):
        return _FirstJet(*_cached_first('neg', lambda: (-self.v, -self.d)))

    def __mul__(self, other):
        other = other if isinstance(other, _FirstJet) else self.constant(other, self.n)
        return _FirstJet(*_cached_first('mul',
            lambda: (self.v*other.v, np.array([self.d[x]*other.v+self.v*other.d[x] for x in range(self.n)]))))

    __rmul__ = __mul__

    def __pow__(self, power):
        if power != int(power) and np.any(self.v <= 0):
            raise np.linalg.LinAlgError('Response requires positive spectral gaps.')
        return _FirstJet(*_cached_first('power',
            lambda: (self.v**power, power*self.v**(power-1)*self.d)))


def _contract(expression, *args):
    jet = next(a for a in args if isinstance(a, _Jet))
    n = jet.n
    # Fixed operands have no response: do not allocate or contract zero jets.
    active = [(i, a) for i, a in enumerate(args) if isinstance(a, _Jet)]
    values = [a.v if isinstance(a, _Jet) else np.asarray(a) for a in args]
    diagonal = {}
    def evaluate(replacements):
        operands = values.copy()
        changed = {i for i, _ in replacements}
        for i, value in replacements:
            operands[i] = value
        inputs, output = expression.split('->')
        terms = inputs.split(',')
        for i in range(len(terms)):
            term = terms[i]
            if i in changed or len(term) != 2 or term[0] == term[1]:
                continue
            if all(label in output for label in term):
                continue  # Both free axes must remain in the output.
            value = operands[i]
            if value.ndim != 2 or value.shape[0] != value.shape[1]:
                continue
            if i not in diagonal:
                diagonal[i] = np.count_nonzero(value) == np.count_nonzero(np.diag(value))
            if not diagonal[i]:
                continue
            keep, remove = term if term[1] not in output else term[::-1]
            operands[i] = np.diag(value)
            terms[i] = keep
            terms = [labels.replace(remove, keep) for labels in terms]
            output = output.replace(remove, keep)
        return np.einsum(','.join(terms)+'->'+output, *operands, optimize=True)
    def first_response():
        value = reference_value('value', lambda: evaluate([]))
        first = np.zeros((n,) + value.shape)
        for i, a in active:
            for x in range(n):
                first[x] += evaluate([(i,a.d[x])])
        return value, first
    value, first = _cached_first('contract:' + expression, first_response)
    second = None if jet.h is None else np.zeros(_second_shape(n) + value.shape)
    if second is not None:
        for position, (i, a) in enumerate(active):
            for row,col,x,y in _second_entries(n):
                second[row,col] += evaluate([(i,a.h[row,col])])
                for j, b in active[:position]:
                    cross = evaluate([(i,a.d[x]), (j,b.d[y])])
                    second[row,col] += 2*cross if x == y else cross
                    if x != y:
                        second[row,col] += evaluate([(i,a.d[y]), (j,b.d[x])])
    return _FirstJet(value,first) if second is None else _Jet(value,first,second)


def _sqrt(matrix):
    """Positive matrix square root via Sylvester equations; degeneracy safe."""
    def first_response():
        def prepare():
            diagonal = np.diag(matrix.v)
            if np.count_nonzero(matrix.v) == np.count_nonzero(diagonal):
                e, u = diagonal, None
            else:
                e, u = np.linalg.eigh(matrix.v)
            if np.min(e) <= 1e-12:
                raise np.linalg.LinAlgError('Unstable BSE: positive A-B and Casida matrix required.')
            root = np.diag(np.sqrt(e)) if u is None else (u*np.sqrt(e))@u.T
            denominator = np.sqrt(e)[:,None]+np.sqrt(e)[None,:]
            return root, u, denominator
        root, u, denominator = reference_value('spectral', prepare)
        first = (matrix.d/denominator if u is None else
                 np.array([u@((u.T@d@u)/denominator)@u.T for d in matrix.d]))
        return root, first, u, denominator
    root, first, u, denominator = _cached_first('sqrt', first_response)
    def solve(rhs):
        return rhs/denominator if u is None else u@((u.T@rhs@u)/denominator)@u.T
    if matrix.h is None:
        return _FirstJet(root, first)
    second = _second_array(matrix.n, lambda a,b,x,y: solve(
        matrix.h[a,b] - first[x]@first[y] - first[y]@first[x]))
    return _Jet(root,first,second)


def _veff(integrals, density, factorized):
    if factorized:
        charge = _contract('Prs,rs->P',integrals,density)
        j = _contract('Ppq,P->pq',integrals,charge)
        contracted = _contract('Ppr,rs->Pps',integrals,density)
        k = _contract('Pps,Pqs->pq',contracted,integrals)
    else:
        j = _contract('pqrs,rs->pq',integrals,density)
        k = _contract('prqs,rs->pq',integrals,density)
    return j - .5*k


def _factor_response(mol, modes, order=2, cache=None):
    from pyqed.qchem.hf.factor_curvature import FactorCurvature
    fc = reference_value('factor_curvature', lambda: FactorCurvature(mol, prepare_first=False))
    n = len(modes)
    supports = [np.flatnonzero(mode) for mode in modes]
    def integrals(x, y=None, packed=False):
        if fc.cd:
            from pyqed.qchem.basis import _pair_factors_to_full
            directions = modes[[x]] if y is None or x == y else modes[[x,y]]
            from .response_storage import _pair_columns, _first_columns
            saved = _pair_columns.get() if y is not None else (_first_columns.get() if n == 1 else None)
            if saved is None:
                columns = fc.integral_response._cd_columns(
                    directions, order=1 if y is None else 2, pair=1 if y is not None and x != y else -1)
                columns = columns[0] if y is None else columns[0, 0]
            else:
                columns = read(saved)
            metric = columns[:,fc.pivots]
            return (columns if packed else _pair_factors_to_full(columns,fc.source.shape[1]).reshape(len(metric),-1)), metric
        v = np.zeros((len(fc.root),fc.factors[0].size))
        m = np.zeros_like(fc.root)
        for a in supports[x]:
            for b in ([None] if y is None else supports[y]):
                scale = modes[x,a] if y is None else modes[x,a]*modes[y,b]
                vi, mi = fc.integrals((int(a),) if b is None else (int(a),int(b)))
                v += scale*vi
                m += scale*mi
        return v,m
    cached = read(cache.get('factor_first')) if cache is not None else None
    l1, b1 = ([], []) if cached is None else cached
    for x in (range(n) if cached is None else ()):
        v,m = integrals(x, packed=fc.cd)
        if fc.cd:
            from pyqed.qchem.basis import _as_ri_pair_factors, _pair_factors_to_full
            from pyqed.qchem.eri_response import _cholesky_factor_response
            from scipy.linalg.blas import get_blas_funcs
            data = mol._cd_factor_data
            base = _as_ri_pair_factors(data['factors'], data['nao'])
            connection, b = _cholesky_factor_response(fc.root, base, v, m)
            l = get_blas_funcs('trmm', (fc.root, connection))(
                1., fc.root, connection, lower=True)
            del connection
            b1.append(_pair_factors_to_full(b, data['nao']).reshape(len(l), -1))
        else:
            l = fc.root_derivative(m)
            b1.append(fc.solve(v-l@fc.factors.reshape(len(l),-1)))
        l1.append(l)
    if cache is not None:
        if 'factor_first' not in cache:
            cache['factor_first'] = keep((l1, b1))
    if order == 1:
        return _FirstJet(fc.molecular(fc.factors),
                         np.array([fc.molecular(b.reshape(fc.factors.shape)) for b in b1]))
    b2 = np.empty(_second_shape(n)+fc.molecular(fc.factors).shape)
    for row,col,x,y in _second_entries(n):
        if not _mixed_response.get() and y > x:
            continue
        v,m = integrals(x,y)
        l = fc.root_derivative(m-l1[x]@l1[y].T-l1[y]@l1[x].T)
        b = fc.solve(v-l@fc.factors.reshape(len(l),-1)-l1[x]@b1[y]-l1[y]@b1[x])
        b2[row,col] = fc.molecular(b.reshape(fc.factors.shape))
        if not _mixed_response.get():
            b2[col,row] = b2[row,col]
    return _Jet(fc.molecular(fc.factors),
                np.array([fc.molecular(b.reshape(fc.factors.shape)) for b in b1]),b2)


@_response_stage
def _nuclear(mol, modes, order=2):
    from pyqed.qchem.hf.hessian import _nuclear_hessian
    def first_response():
        coords, charges = mol.atom_coords(), mol.atom_charges()
        gradient = np.zeros_like(coords)
        for a in range(len(coords)):
            for b in range(a):
                r = coords[a]-coords[b]
                g = -charges[a]*charges[b]*r/np.linalg.norm(r)**3
                gradient[a] += g
                gradient[b] -= g
        return mol.energy_nuc(), modes@gradient.ravel()
    value, first = _cached_first('nuclear', first_response)
    if order == 1:
        return _FirstJet(value, first)
    nuclear = _nuclear_hessian(mol).reshape(modes.shape[1],modes.shape[1])
    return _Jet(value, first, _second_array(len(modes), lambda a,b,x,y: modes[x]@nuclear@modes[y]))


def _response_key(bse, modes, backend, groups):
    """Content fingerprint: detect in-place changes as well as replacements."""
    import hashlib
    digest = hashlib.sha256()
    mf, mol = bse._scf, bse.mol
    for value in (modes, mol.atom_coords(), mol.atom_charges(), mf.mo_coeff,
                  mf.mo_energy, mf.mo_occ, mf.make_rdm1(), mf.get_hcore(),
                  mf.get_ovlp(), getattr(mol, 'eri_factors', None),
                  getattr(mf, '_eri', None), getattr(mol, 'eri', None)):
        if value is None:
            digest.update(b'None')
        else:
            array = np.ascontiguousarray(value)
            digest.update(str((array.dtype.str, array.shape)).encode())
            if array.dtype.hasobject:
                digest.update(repr(value).encode())
            else:
                digest.update(memoryview(array).cast('B'))
    def metadata(value):
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(memoryview(array).cast('B'))
        elif isinstance(value, dict):
            for k in sorted(value, key=str):
                metadata(k)
                metadata(value[k])
        elif isinstance(value, (list, tuple)):
            for item in value:
                metadata(item)
        else:
            digest.update(repr(value).encode())
    metadata((id(mol), backend, groups, getattr(mol, 'basis', None),
              getattr(mol, '_cd_factor_data', None),
              getattr(mol, '_builtin_build_info', None), bse.e_qp,
              bse.screening_energies, bse.bse_metric, bse.screening,
              bse.gw.freq_int, bse.gw.eta, bse.e, bse.x, bse.y))
    return digest.digest()


def _reference_response(bse, modes, backend, groups, order=2, *, orbital_solver=None):
    from pyqed.qchem.basis_derivatives import (
        directional_one_electron_derivatives as one,
        directional_eri_derivatives as two)
    from pyqed.qchem.hf.hessian import RHFHessian
    from .subspace_response import orbital_first, orbital_second
    mf, mol, n = bse._scf, bse.mol, len(modes)
    # General contractions can span the same basis with different AO functions
    # in PySCF. Differentiate the actual builtin AO frame by default.
    integral_backend = 'native' if backend == 'auto' else backend
    key = _response_key(bse, modes, backend, groups)
    previous = getattr(bse, '_first_response_cache', None)
    reused = previous is not None and previous['key'] == key
    cache = previous if reused else {'key': key}
    bse._first_response_cache = None
    def first_integral(name, compute):
        if name not in cache:
            cache[name] = keep(compute())
        return read(cache[name])
    jet = _FirstJet if order == 1 else _Jet
    def make(value, first, second):
        return jet(value, first) if order == 1 else jet(value, first, second())
    def curvature(compute):
        value = compute()
        return value[0:1,1:2] if _mixed_response.get() else value
    h = make(mf.get_hcore(), first_integral('hcore', lambda: one(mol,modes,'hcore',order=1,backend=integral_backend)),
             lambda: curvature(lambda: one(mol,modes,'hcore',order=2,backend=integral_backend)))
    s1 = first_integral('overlap', lambda: one(mol,modes,'overlap',order=1,backend=integral_backend))
    s2 = curvature(lambda: one(mol,modes,'overlap',order=2,backend=integral_backend)) if order == 2 else None
    factorized = getattr(mol,'eri_factors',None) is not None
    if factorized:
        integrals = _factor_response(mol,modes,order,cache)
    else:
        from pyqed.qchem.tddft import _dense_eri
        integrals = make(_dense_eri(mol), first_integral('eri', lambda: two(mol,modes,order=1,backend=integral_backend)),
                         lambda: curvature(lambda: two(mol,modes,order=2,backend=integral_backend)))
    driver = RHFHessian(mf)
    driver._require_scf()
    density = jet.constant(mf.make_rdm1(),n)
    if 'orbital_first' in cache:
        first, mask, e1 = read(cache['orbital_first'])
    else:
        explicit = h + _veff(integrals,density,factorized)
        first = driver._first_order_mo_response_data(
            explicit.d,s1,orbital_solver=orbital_solver)
        mask,e1 = orbital_first(first,groups)
        cache['orbital_first'] = keep((first, mask, e1))
    density.d = first['density_response']
    with _response_scope(cache.setdefault('explicit_fock', {}), 'explicit_fock'):
        explicit = h + _veff(integrals,density,factorized)
    if order == 2:
        from scipy.linalg import lu_factor, lu_solve
        def prepare_solver():
            return lu_factor(driver._build_cphf_matrix(
                first['mo_coeff'], first['cocc'], first['occidx'], first['viridx'],
                first['e_occ'], first['mo_energy'][first['viridx']]))
        solver = None
        if len(first['viridx']):
            factors = reference_value('orbital_response_lu', prepare_solver)
            solver = lambda rhs: lu_solve(factors, rhs)
        second = driver._second_order_mo_response_data(first,explicit.h,s2,
                        pair=(0,1) if _mixed_response.get() else None,
                        orbital_solver=solver)
        density.h = second['density_response']
    c = np.asarray(mf.mo_coeff)
    coefficients = make(c, first_integral('coefficients',
                        lambda: np.einsum('pi,xij->xpj',c,first['full_response'])),
                        lambda: np.einsum('pi,xyij->xypj',c,orbital_second(driver,first,second,mask,e1)))
    with _response_scope(cache.setdefault('reference_transform', {}), 'reference_transform'):
        fock = h + _veff(integrals,density,factorized)
        energy = _contract('pi,pq,qj->ij',coefficients,fock,coefficients)
        energy.v = np.diag(mf.mo_energy)
        energy.d = energy.d * mask
        if order == 2:
            energy.h *= mask
        ground = .5*_contract('pq,pq->',density,h+fock) + _nuclear(mol,modes,order)
        if factorized:
            half = _contract('Ppq,pi->Piq',integrals,coefficients)
            mo_integrals = _contract('Piq,qj->Pij',half,coefficients)
        else:
            mo_integrals = _contract('pqrs,pi,qj,rk,sl->ijkl',integrals,
                                     coefficients,coefficients,coefficients,coefficients)
    bse._first_response_cache = cache
    return energy, mo_integrals, ground, factorized


def _derivatives(bse, state_ids=None, modes=None, frame='relaxed',
                       backend='auto', max_memory=None, degeneracy_tol=1e-7, order=2, _bounded=False, hessian_method='polarization'):
    """Implementation of the public BSE derivative interface."""
    from pyqed.qchem.geometric import _as_cartesian_mode_matrix
    if hessian_method not in ('mixed', 'polarization'):
        raise ValueError('Hessian method must be mixed or polarization.')
    from .subspace_response import orbital_groups, screening, quasiparticles, blocks
    mf, mol, gw = bse._scf,bse.mol,bse.gw
    if frame != 'relaxed':
        raise ValueError("Molecular GW/BSE requires frame='relaxed'; fixed real-space states are unsupported.")
    if not getattr(mf,'converged',False) or bse.e is None:
        raise ValueError('Converged RHF and completed BSE/TDA excitations are required.')
    if bse.info is not None and not bse.info.get('converged',True):
        raise ValueError('BSE/TDA did not converge; converge excitations before requesting analytic response.')
    if hasattr(mf,'xc') or gw is None or getattr(gw,'method',None) != 'g0w0' or bse.screening != 'TDH':
        raise NotImplementedError('Analytic molecular response requires RHF G0W0 with TDH screening.')
    if gw.freq_int != 'exact' or bse.orbital_window is not None:
        raise NotImplementedError('Analytic response requires exact spectral GW and the full orbital space.')
    if not hasattr(mol,'natom') or getattr(mf,'density_fit',False):
        raise NotImplementedError('Use a builtin molecular RHF reference, with native RI/CD or exact integrals.')
    if np.iscomplexobj(mf.mo_coeff) or not np.all(np.isin(mf.mo_occ,[0,2])):
        raise NotImplementedError('Analytic response requires real closed-shell RHF orbitals.')
    if not np.allclose(mf.mo_coeff,bse.mo_coeff,atol=1e-12,rtol=0):
        raise ValueError('BSE and RHF orbital frames differ.')
    energies = np.asarray(mf.mo_energy)
    if not np.isfinite(degeneracy_tol) or degeneracy_tol <= 0:
        raise ValueError('degeneracy_tol must be positive Hartree.')
    if energies[bse.nocc]-energies[bse.nocc-1] < degeneracy_tol:
        raise ValueError('Occupied and virtual subspaces must be separated by a gap.')
    groups = orbital_groups(energies,bse.nocc,degeneracy_tol)
    if not np.allclose(gw.e_mf[::2],energies,atol=1e-12,rtol=0):
        raise ValueError('GW and RHF starting energies differ.')
    if not np.allclose(bse.e_qp,gw.e_qp,atol=1e-12,rtol=0):
        raise ValueError('BSE must use the converged GW quasiparticle energies.')
    cartesian = modes is None
    modes = np.eye(3*mol.natom) if cartesian else _as_cartesian_mode_matrix(modes,mol.natom)
    if not len(modes) or not np.all(np.isfinite(modes)):
        raise ValueError('Require nonempty finite displacement modes in bohr per coordinate.')
    ids = np.arange(len(bse.e)) if state_ids is None else np.asarray(state_ids)
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu'
            or len(np.unique(ids)) != len(ids) or np.any(ids<0) or np.any(ids>=len(bse.e))):
        raise ValueError('state_ids must be unique valid zero-based excitation indices.')
    # This dense response algorithm is intentionally bounded; it does not
    # silently turn a matrix-free production BSE into an unbounded allocation.
    n,nao,nmo = len(modes),mol.nao,len(energies)
    dim = bse.nocc*(nmo-bse.nocc)
    factors = getattr(mol,'eri_factors',None)
    rank = 0 if factors is None else len(factors)
    size = nao**4 if factors is None else rank*nao**2 + rank**2
    estimate = 8*(1+n+(n*n if order == 2 else 0))*(8*size+20*dim**2+8*nmo*nmo*max(1,dim))
    budget = bse.max_memory if max_memory is None else max_memory
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('max_memory must be positive MB.')
    if _bounded:
        # One second block and one/two first directions are live. Retained
        # intermediates use disk mappings; this is a working-memory estimate.
        estimate = 8*((12 if _mixed_response.get() else 10)*size+20*dim**2+8*nmo*nmo*max(1,dim))
        if estimate > budget*1e6:
            raise MemoryError(f'One directional BSE response needs approximately {estimate/1e6:.1f} MB; '
                              f'max_memory is {budget:g} MB.')
    elif estimate > budget*1e6 or 'directions' in (getattr(bse, '_first_response_cache', None) or {}):
        from .response_storage import directional_derivatives
        result = directional_derivatives(bse, ids, modes, backend, groups, budget, degeneracy_tol, order, method=hessian_method)
        if cartesian and order == 2:
            f,g = result
            return f.reshape(len(ids),len(ids),mol.natom,3),g.reshape(len(ids),len(ids),mol.natom,3,mol.natom,3)
        return result
    energy,integrals,ground,factorized = _reference_response(bse,modes,backend,groups,order)
    with _response_scope(bse._first_response_cache.setdefault('gw_bse', {}), 'gw_bse'):
        poles,couplings = screening(energy,integrals,bse.nocc,factorized)
        qp,qp_residuals = quasiparticles(bse,energy,poles,couplings,groups,degeneracy_tol)
        if bse.screening_energies == 'qp':
            poles,couplings = screening(qp,integrals,bse.nocc,factorized)
        a,b = blocks(qp,integrals,poles,couplings,bse.nocc,factorized)
        if bse.bse_metric == 'tda':
            ham = a
            vectors = bse.x
        elif bse.bse_metric == 'full':
            minus = _sqrt(a-b)
            ham = _sqrt(_contract('ij,jk,kl->il',minus,a+b,minus))
            vectors = reference_value('exciton_vectors', lambda: np.linalg.solve(minus.v,bse.x+bse.y)*np.sqrt(bse.e)[None,:])
        else:
            raise ValueError('Unknown BSE excitation metric.')
    residual = np.max(abs(ham.v@vectors-vectors*bse.e[None,:]))
    orthogonality = np.max(abs(vectors.T@vectors-np.eye(len(bse.e))))
    if residual > 2e-6 or orthogonality > 2e-6:
        raise ValueError('Stored BSE states do not match the differentiated Hamiltonian: '
                         f'residual={residual:g}, orthogonality_defect={orthogonality:g}.')
    # Freeze the reference exciton amplitudes, while differentiating the
    # underlying canonical RHF/GW/kernel data through second order.
    selected = vectors[:,ids]
    f = np.einsum('pi,xpq,qj->ijx',selected,ham.d,selected)
    if order == 1:
        all_e, _ = reference_value('exciton_spectrum', lambda: np.linalg.eigh(ham.v))
        defined = np.array([np.count_nonzero(abs(all_e-bse.e[i]) < 1e-8) == 1 for i in ids])
        gradient = np.array([f[k,k]+ground.d if defined[k] else np.full(n,np.nan)
                             for k in range(len(ids))])
        bse.energy_gradient = gradient
        bse.energy_hessian = None
        bse.derivative_info = dict(order=1, state_ids=ids.tolist(), modes=modes.copy(),
                                   estimate_MB=estimate/1e6, surface_gradient_defined=defined.tolist())
        return gradient
    g = np.einsum('pi,xypq,qj->ijxy',selected,ham.h,selected)
    eye = np.eye(len(ids))
    f += eye[:,:,None]*ground.d
    g += eye[:,:,None,None]*ground.h
    # Complete excitation-space response, including states not requested by
    # the caller, is needed for an actual adiabatic surface Hessian.
    all_e,all_u = reference_value('exciton_spectrum', lambda: np.linalg.eigh(ham.v))
    trace_hessian, surface = _surface_curvature(ham.d, selected, all_e, all_u, bse.e[ids], g)
    bse.subspace_gradient = np.trace(f,axis1=0,axis2=1)
    bse.subspace_hessian = trace_hessian
    defined = np.all(np.isfinite(surface),axis=(1,2))
    bse.energy_gradient = np.array([f[k,k] if defined[k] else np.full(n,np.nan)
                             for k in range(len(ids))])
    bse.energy_hessian = surface
    bse.F,bse.G = f,g
    bse.derivative_info = dict(frame=frame, relaxation='RHF+G0W0+screening',
        representation='orbital-block excitation frame' if bse.bse_metric=='tda' else 'positive-frequency Casida frame',
        state_ids=ids.tolist(), modes=modes.copy(), estimate_MB=estimate/1e6,
        orbital_subspaces=[g.tolist() for g in groups], degeneracy_tol=degeneracy_tol,
        quasiparticle_model='Hermitian Dyson response within reference orbital subspaces',
        quasiparticle_subspace_residuals=qp_residuals,
        screening_response='matrix square roots and resolvents; no pole eigenvector derivatives',
        retained_subspace_trace_hessian=trace_hessian.tolist(),
        surface_gradient_defined=defined.tolist(),surface_hessian_defined=defined.tolist(),
        transverse_kinetic_connection_included=False)
    if cartesian:
        return f.reshape(len(ids),len(ids),mol.natom,3),g.reshape(len(ids),len(ids),mol.natom,3,mol.natom,3)
    return f,g


def _surface_curvature(first, selected, all_e, all_u, energies, g):
    """Full-spectrum state response with one state's linear-size workspace.

    Apply each derivative matrix to the selected vector before rotating into
    the full eigenbasis. Reuse that coupling for the surface and trace sums.
    """
    n = len(first)
    complement = 1-np.sum(abs(all_u.T@selected)**2,axis=1)
    trace = np.trace(g,axis1=0,axis2=1).copy()
    surface = np.empty((len(energies),*_second_shape(n)))
    for k, energy in enumerate(energies):
        vector = selected[:,k]
        coupling = np.asarray([all_u.T@(derivative@vector) for derivative in first])
        gaps = energy-all_e
        near = abs(gaps)<1e-8
        if np.sum(complement[near]) > 1e-5:
            trace[:] = np.nan
        elif np.all(np.isfinite(trace)):
            weight = np.zeros_like(gaps)
            weight[~near] = complement[~near]/gaps[~near]
            correction = (coupling*weight)@coupling.T
            trace += _second_array(n, lambda a,b,x,y: correction[x,y]+correction[y,x])
        target = np.argmax(abs(all_u.T@vector))
        gaps[target] = np.inf
        if np.any(abs(gaps)<1e-8):
            surface[k] = np.nan
        else:
            correction = (coupling/gaps)@coupling.T
            surface[k] = g[k,k]+_second_array(n, lambda a,b,x,y: correction[x,y]+correction[y,x])
    return trace, surface


def vibronic_couplings(bse, state_ids=None, modes=None, frame='relaxed', **kwargs):
    """Local electronic-Hamiltonian derivatives; see BSE.vibronic_couplings."""
    return _derivatives(bse, state_ids, modes, frame, **kwargs)


def energy_derivatives(bse, state_ids=None, modes=None, *, order=2, **kwargs):
    """Relaxed adiabatic energy derivatives; see BSE.gradient and BSE.hessian."""
    _derivatives(bse, state_ids, modes, 'relaxed', order=order, **kwargs)
    return bse.energy_gradient, bse.energy_hessian
