"""Bounded local operators in tied reduced tensor coordinates.

The Wigner--Eckart contractions adapt Weichselbaum, Phys. Rev. B 86, 245124
(2012), https://doi.org/10.1103/PhysRevB.86.245124. Projection is into local
tensor parameters, never magnetic or determinant states. This experimental
assembly does not change the optimizer or provide convergence guarantees.
"""
import tempfile
import time

import numpy as np

from pyqed.mps.nonabelian import environment as env


def relax_parameter_candidate(operator, metric, current, candidate, *, relaxation,
                              energy_before, root_energy):
    """Try a normalized, phase-aligned overrelaxation with energy backtracking.

    Inspired by Oseledets, Rakhuba and Uschmajew, 'Local convergence of
    alternating low-rank optimization methods with overrelaxation',
    arXiv:2111.14758, doi:10.1002/nla.2459. This Rayleigh-quotient/closed-LETTA
    variant adds normalization and an energy guard; it does not inherit their
    convergence or optimal-parameter guarantees. The caller must first validate
    the local root and independently check any accepted extrapolated update.
    """
    if not np.isfinite(relaxation) or not 1 <= relaxation < 2:
        raise ValueError('relaxation must lie in [1, 2)')
    if relaxation == 1:
        return candidate, 1.
    norm_current = float(np.vdot(current, metric.matvec(current)).real)
    norm_candidate = float(np.vdot(candidate, metric.matvec(candidate)).real)
    if not all(np.isfinite(x) and x > 0 for x in (norm_current, norm_candidate)):
        raise ArithmeticError('Invalid norm for parameter relaxation')
    base = current/np.sqrt(norm_current)
    root = candidate/np.sqrt(norm_candidate)
    overlap = np.vdot(root, metric.matvec(base))
    if abs(overlap):
        root = root*(overlap/abs(overlap))
    limit = energy_before-.05*max(0., energy_before-root_energy)
    limit += 1e-12*max(1., abs(energy_before))
    weight = float(relaxation)
    for _ in range(8):
        trial = (1-weight)*base+weight*root
        norm = float(np.vdot(trial, metric.matvec(trial)).real)
        if np.isfinite(norm) and norm > 0:
            energy = float(np.vdot(trial, operator.matvec(trial)).real/norm)
            if np.isfinite(energy) and energy <= limit:
                return trial/np.sqrt(norm), weight
        weight = 1+.5*(weight-1)
    return candidate, 1.


def preserve_metric_reference(metric, transform, current):
    """Retain the incumbent using one extra sparse metric-orthogonal direction.

    Two reorthogonalization passes preserve the variational reference in the
    restricted tensor-parameter search, not every excluded metric direction.
    """
    from scipy.sparse import csr_matrix, hstack
    norm = float(np.vdot(current, metric.matvec(current)).real)
    if not np.isfinite(norm) or norm <= 0:
        raise ArithmeticError('Invalid incumbent norm')
    remainder = np.array(current, dtype=complex, copy=True)
    for _ in range(2):
        remainder -= transform@(transform.conj().T@metric.matvec(remainder))
    missing = float(np.vdot(remainder, metric.matvec(remainder)).real)
    if not np.isfinite(missing) or missing < -1e-12*norm:
        raise ArithmeticError('Invalid incumbent remainder norm')
    cancellation_bound = 0.
    for indices, block in metric.components:
        amplitude = np.abs(remainder[indices])
        cancellation_bound += (8*np.finfo(float).eps*len(indices)
                               *float(amplitude@(np.abs(block)@amplitude)))
    if missing > max(1e-20*norm, cancellation_bound):
        transform = hstack((transform, csr_matrix((remainder/np.sqrt(missing))[:, None])), format='csr')
    return transform, max(0., missing/norm)


def solve_parameter_operator(operator, metric, current, *, metric_rtol=1e-11,
                             tolerance=1e-10, maxiter=300, max_space=96,
                             preconditioner='diagonal', norm_matvec=None):
    """Return a candidate, sparse coordinates and local convergence diagnostics.

    Adapts Davidson, J. Comput. Phys. 17, 87--94 (1975),
    doi:10.1016/0021-9991(75)90065-0, to retained reduced metric coordinates
    augmented by the incumbent. The residual certifies only that restricted
    space; no state is mutated and global convergence is not guaranteed.
    The actual transformed norm is applied during generalized Davidson;
    approximate numerical whitening is not treated as an exact identity.
    Independent contraction checks remain the responsibility of the caller.
    Optional ``norm_matvec`` supplies a factored norm action for the solve and
    reported residual. Component blocks still select the retained coordinates
    and incumbent direction; the supplied action must represent the same norm.
    """
    from pyqed.mps.nonabelian.solver import _solve_packed_generalized_davidson
    if not 0 < metric_rtol < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Invalid metric cutoff or residual tolerance')
    maxiter = _positive_integer(maxiter, 'maxiter')
    max_space = _positive_integer(max_space, 'max_space')
    if preconditioner not in {'constant', 'diagonal'}:
        raise ValueError('Unknown parameter preconditioner')
    current = np.asarray(current)
    if current.shape != (operator.dimension,) or metric.dimension != operator.dimension:
        raise ValueError('Inconsistent parameter dimensions')
    transform, *_ = metric.whitener(rtol=metric_rtol, sparse=True)
    cutoff_rank = transform.shape[1]
    transform, missing_norm = preserve_metric_reference(metric, transform, current)
    apply_norm = metric.matvec if norm_matvec is None else norm_matvec
    initial = transform.conj().T@apply_norm(current)
    def action(vector):
        return transform.conj().T@operator.matvec(transform@vector)
    def metric_action(vector):
        return transform.conj().T@apply_norm(transform@vector)
    current_norm = float(np.vdot(current, apply_norm(current)).real)
    energy_before = float(np.vdot(current, operator.matvec(current)).real/current_norm)
    started = time.perf_counter()
    diagonal = (operator.transformed_diagonal(transform).real if preconditioner == 'diagonal'
                else np.zeros(transform.shape[1]))
    preconditioner_seconds = time.perf_counter()-started
    started = time.perf_counter()
    _, vector, info = _solve_packed_generalized_davidson(
        initial, action, h_diag=diagonal, N=metric_action, n_diag=None,
        tol=tolerance, tol_residual=tolerance, itermax=maxiter,
        max_space=min(max_space, transform.shape[1]), minimum_explored_dimension=2,
        convergence_confirmations=2)
    davidson_seconds = time.perf_counter()-started
    candidate = transform@vector
    norm_action, h_action = apply_norm(candidate), operator.matvec(candidate)
    norm = float(np.vdot(candidate, norm_action).real)
    if not np.isfinite(norm) or norm <= 0:
        raise ArithmeticError('Invalid candidate norm')
    energy = float(np.vdot(candidate, h_action).real/norm)
    residual = float(np.linalg.norm(transform.conj().T@(h_action-energy*norm_action))/np.sqrt(norm))
    return candidate, transform, dict(
        energy_before=energy_before, operator_energy=energy, operator_residual=residual,
        norm=norm, tolerance=tolerance, residual_space='retained_metric_and_incumbent',
        generalized_norm=True, factored_norm=norm_matvec is not None,
        davidson_converged=bool(info['davidson_converged']),
        davidson_residual=float(info['residual']),
        davidson_restarts=int(info['restarts']),
        davidson_subspace_dim=int(info['subspace_dim']),
        davidson_correction_fallbacks=int(info['correction_fallbacks']),
        iterations=int(info['davidson_iterations']), retained_rank=transform.shape[1],
        cutoff_rank=cutoff_rank, incumbent_missing_norm_squared=missing_norm,
        incumbent_direction_added=transform.shape[1] > cutoff_rank,
        preconditioner=preconditioner, preconditioner_seconds=preconditioner_seconds,
        davidson_seconds=davidson_seconds,
        whitener_bytes=transform.data.nbytes+transform.indices.nbytes+transform.indptr.nbytes)


def _positive_integer(value, name):
    if not np.isfinite(value) or value < 1 or int(value) != value:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def iter_parameter_blocks(plan, core, left, right, *, block_bytes=2**20, route_tile=128,
                          compiled=True, compensated=False):
    """Yield exact local operator blocks with bounded output and route tiles.

    ``block_bytes`` limits each complex output block, not boundary decoding,
    route metadata or total process memory. NumPy route-pair temporaries have
    at most ``route_tile**2`` entries each. The compiled kernel streams route
    pairs without such temporaries and receives one route table per block.
    No numerical entries are truncated.
    Real scalar-physical terms use compiled route accumulation when available;
    ``compiled=False`` selects the NumPy reference. Nonzero imaginary data
    always retain the complex NumPy path, without a tolerance-based projection.
    Experimental ``compensated=True`` requires the real scalar-physical
    compiled path and adds one correction block of at most ``block_bytes``.
    Unsupported terms raise rather than silently dropping compensation.
    See the module reference for the reduced formulation and fidelity limits.
    """
    block_bytes = _positive_integer(block_bytes, 'block_bytes')
    route_tile = _positive_integer(route_tile, 'route_tile')
    if block_bytes < 16:
        raise ValueError('block_bytes must hold at least one complex value')
    width = int(np.sqrt(block_bytes//16))
    kernel = getattr(env._su2_kernel_module(), 'accumulate_parameter_routes', None) if compiled else None
    records = []
    for key, (parameters, targets, shape) in plan._group_by_key.items():
        unique, local = np.unique(parameters, return_inverse=True)
        records.append((key, unique, local, *np.unravel_index(targets, shape)))
    for (lb, pb, rb), oi, ob, ol, op, ore in records:
        for (lk, pk, rk), ii, ib, il, ip, ir in records:
            e, f = left.get((lb, lk)), right.get((rb, rk))
            if e is None or f is None:
                continue
            terms = env._left_reduced_rank_coupled_block(core, lb, lk, pb, pk, rb, rk)
            if not terms:
                continue
            for os in range(0, len(oi), width):
                outs = np.flatnonzero((ob >= os) & (ob < os+width))
                for ins in range(0, len(ii), width):
                    inputs = np.flatnonzero((ib >= ins) & (ib < ins+width))
                    block = np.zeros((len(oi[os:os+width]), len(ii[ins:ins+width])), dtype=complex)
                    correction = np.zeros_like(block) if compensated else None
                    bra_routes = ket_routes = None
                    previous_left = previous_right = None
                    for (lc, rc), w in (terms or {}).items():
                        if lc not in e or rc not in f:
                            continue
                        # Keep the active channel even when it exceeds the
                        # boundary's decode-cache budget. Preserve term order.
                        if lc != previous_left:
                            a = e[lc]
                            previous_left = lc
                        if rc != previous_right:
                            b = f[rc]
                            previous_right = rc
                        weights = [env._component_basis_norm(rb, rk, core.right_channel_irreps[rc], m)
                                   for m in env.ordered_two_m_values(core.right_channel_irreps[rc])]
                        real = None
                        if kernel is not None and w.shape[2:] == (1, 1):
                            arrays = (a, b, w[:, :, 0, 0])
                            if all(not np.iscomplexobj(t) or not np.any(t.imag) for t in arrays):
                                real = tuple(np.ascontiguousarray(t.real, dtype=float) for t in arrays)
                                metric_weights = np.asarray(weights, dtype=float)
                        if real is not None:
                            if bra_routes is None:
                                bra_routes = np.column_stack((ob[outs]-os, ol[outs], ore[outs])).astype(np.int64, copy=False)
                                ket_routes = np.column_stack((ib[inputs]-ins, il[inputs], ir[inputs])).astype(np.int64, copy=False)
                            kernel(block, *real, metric_weights, bra_routes, ket_routes, correction)
                            continue
                        if compensated:
                            raise NotImplementedError('Compensated assembly requires compiled real scalar-physical terms')
                        for start in range(0, len(outs), route_tile):
                            u = outs[start:start+route_tile]
                            for end in range(0, len(inputs), route_tile):
                                v = inputs[end:end+route_tile]
                                value = np.zeros((len(u), len(v)), dtype=complex)
                                for x in range(w.shape[0]):
                                    for y in range(w.shape[1]):
                                        value += (a[x][np.ix_(ol[u], il[v])]
                                                  * b[y][np.ix_(ore[u], ir[v])]
                                                  * w[x, y][np.ix_(op[u], ip[v])] * weights[y])
                                np.add.at(block, (ob[u, None]-os, ib[None, v]-ins), value)
                    if np.any(block):
                        yield oi[os:os+width], ii[ins:ins+width], block


class TiedParameterOperator:
    """Retained or disk-backed blocks for repeated local parameter actions.

    Numerical resident arrays, including block indices, respect ``storage_bytes``.
    Disk blocks respect ``disk_bytes``. Python metadata, input/output vectors,
    and one decoded block are outside these budgets. Assembly block size is
    controlled by :func:`iter_parameter_blocks`. Close after the local solve;
    no site, core, or boundary objects are retained. See the module reference.
    """

    def __init__(self, dimension, blocks, *, storage_bytes=32*2**20,
                 spill_directory=None, disk_bytes=8*2**30):
        self.dimension = _positive_integer(dimension, 'dimension')
        storage_bytes = _positive_integer(storage_bytes, 'storage_bytes')
        disk_bytes = _positive_integer(disk_bytes, 'disk_bytes')
        self.file = None
        self.blocks = []
        self.resident_bytes = self.disk_bytes = 0
        self.closed = False
        try:
            if spill_directory is not None:
                self.file = tempfile.TemporaryFile(dir=spill_directory)
            for rows, columns, values in blocks:
                rows, columns = np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)
                values = np.asarray(values, dtype=complex)
                if (rows.ndim != 1 or columns.ndim != 1 or values.shape != (len(rows), len(columns))
                        or np.any(rows < 0) or np.any(rows >= self.dimension)
                        or np.any(columns < 0) or np.any(columns >= self.dimension)):
                    raise ValueError('Invalid parameter operator block')
                size = rows.nbytes+columns.nbytes+(values.nbytes if self.file is None else 0)
                if self.resident_bytes+size > storage_bytes:
                    raise MemoryError('Parameter operator resident storage exceeds budget')
                rows, columns = rows.copy(), columns.copy()
                rows.flags.writeable = columns.flags.writeable = False
                if self.file is None:
                    payload = values.copy()
                    payload.flags.writeable = False
                else:
                    if self.disk_bytes+values.nbytes > disk_bytes:
                        raise MemoryError('Parameter operator disk storage exceeds budget')
                    payload = self.disk_bytes
                    values.tofile(self.file)
                    self.disk_bytes += values.nbytes
                self.blocks.append((rows, columns, payload))
                self.resident_bytes += size
        except BaseException:
            self.close()
            raise

    def _iter_blocks(self):
        if self.closed:
            raise RuntimeError('Parameter operator is closed')
        for rows, columns, payload in self.blocks:
            if self.file is None:
                values = payload
            else:
                self.file.seek(payload)
                count = len(rows)*len(columns)
                raw = self.file.read(count*16)
                if len(raw) != count*16:
                    raise OSError('Incomplete parameter operator payload')
                values = np.frombuffer(raw, dtype=complex).reshape(len(rows), len(columns))
            yield rows, columns, values

    def matvec(self, vector):
        vector = np.asarray(vector)
        if vector.shape != (self.dimension,):
            raise ValueError('Parameter vector has the wrong shape')
        result = np.zeros(self.dimension, dtype=np.result_type(vector, complex))
        for rows, columns, values in self._iter_blocks():
            np.add.at(result, rows, values@vector[columns])
        return result

    def diagonal(self):
        """Read the exact diagonal, including overlap between stored blocks.

        Uses one decoded block and its index-match arrays at a time; the
        returned vector is not retained or included in the storage budget.
        """
        result = np.zeros(self.dimension, dtype=complex)
        for rows, columns, values in self._iter_blocks():
            i, j = np.nonzero(rows[:, None] == columns[None, :])
            np.add.at(result, rows[i], values[i, j])
        return result

    def transformed_diagonal(self, transform):
        """Stream diag(T† H T) for sparse coordinates, retaining all couplings.

        Each direction uses only its support within the current block. No full
        transformed matrix or dense coordinate array is constructed.
        """
        from scipy.sparse import issparse
        if not issparse(transform) or transform.shape[0] != self.dimension:
            raise ValueError('Expected sparse coordinates with one row per parameter')
        transform = transform.tocsr()
        result = np.zeros(transform.shape[1], dtype=complex)
        for rows, columns, values in self._iter_blocks():
            bra, ket = transform[rows].tocsc(), transform[columns].tocsc()
            shared = np.flatnonzero((np.diff(bra.indptr) != 0) & (np.diff(ket.indptr) != 0))
            for j in shared:
                bs = slice(bra.indptr[j], bra.indptr[j+1])
                ks = slice(ket.indptr[j], ket.indptr[j+1])
                result[j] += np.vdot(bra.data[bs],
                    values[np.ix_(bra.indices[bs], ket.indices[ks])]@ket.data[ks])
        return result

    def close(self):
        if self.file is not None:
            self.file.close()
        self.blocks.clear()
        self.resident_bytes = 0
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
