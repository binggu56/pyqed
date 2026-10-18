"""Exact sparse frontier contractions in reduced SU(2) coordinates.

Uses the Wigner--Eckart representation of Weichselbaum, Annals of Physics
327, 2972--3047 (2012), doi:10.1016/j.aop.2012.07.009, adapted to graph-tied
spatial-orbital tensors. Sparse storage removes structural zeros, not small
singular values. This is not a new truncation or a convergence guarantee.
The current implementation supports open chains with physical multiplicity
one; it retains the existing scalar rank-coupled Hamiltonian conventions.
"""
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix, eye, vstack
from scipy.sparse.linalg import LinearOperator

from pyqed.mps.nonabelian.environment import (
    _left_reduced_rank_coupled_block, _reduced_operator_rank, ordered_two_m_values,
)
from pyqed.mps.nonabelian.mpo import prepare_hamiltonian_factors, iter_virtual_routes
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo


def _bytes(matrix):
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


class SU2FrontierContractions:
    """Sparse reduced site, boundary and tied-parameter operator contractions.

    Reuses the solver's structural scatter maps and spin-recoupling helpers.
    No expanded site array, dense norm boundary, pair tensor or unit-direction
    basis is built. ``memory_bytes`` bounds each sparse boundary and matrix;
    input/output objects, route tables and the MPO additionally remain resident.
    Products are row-chunked to bound their dense worst-case output size.
    Exact sparse boundaries can still fill in exponentially with graph width:
    budgets raise MemoryError rather than silently truncating the state.

    See the module reference for formulation and restrictions. The object
    reads current state parameters on every call; local operators freeze only
    the other sites and must be rebuilt after those sites change.
    """

    def __init__(self, state, *, memory_bytes=256 * 2**20, chunk_bytes=8 * 2**20):
        if state.virtual_boundary != 'open':
            raise NotImplementedError('Sparse SU(2) frontiers currently require open virtual boundaries')
        if any(b.shape[1] != 1 for s in state._base_sites for b in s.data.values()):
            raise NotImplementedError('Sparse SU(2) frontiers require physical multiplicity one')
        if not 0 < chunk_bytes <= memory_bytes:
            raise ValueError('Require 0 < chunk_bytes <= memory_bytes')
        self.state = state
        self.memory_bytes, self.chunk_bytes = int(memory_bytes), int(chunk_bytes)
        self.stats = dict(peak_matrix_bytes=0, peak_boundary_bytes=0,
                          peak_site_bytes=0, dense_site_allocations=0)
        self._hamiltonian = None
        self._supports = {}
        self.identity = _identity_mpo_factors_for_sites_and_mpo(state._base_sites, state.mpo)

    @property
    def hamiltonian(self):
        if self.state._state_only:
            raise ValueError('This state has no Hamiltonian')
        if self._hamiltonian is None:
            # Materialize reduced operator terms only, never tied site tensors.
            self._hamiltonian = prepare_hamiltonian_factors(self.state.mpo, materialize=True)
        return self._hamiltonian

    def _check(self, matrix):
        matrix = matrix.tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        size = _bytes(matrix)
        self.stats['peak_matrix_bytes'] = max(self.stats['peak_matrix_bytes'], size)
        if size > self.memory_bytes:
            raise MemoryError(f'Sparse frontier matrix needs {size} bytes; budget {self.memory_bytes}')
        return matrix

    def _product(self, left, right, *, support=None, dims=None):
        # CSR accumulator workspace scales with columns; each returned chunk
        # fits the budget even if all of its entries become nonzero.
        item = np.dtype(np.result_type(left.dtype, right.dtype)).itemsize + 8
        row_bytes = max(1, right.shape[1]) * item + 8
        if row_bytes > self.chunk_bytes:
            raise MemoryError('One sparse product row exceeds chunk_bytes')
        rows = max(1, self.chunk_bytes // row_bytes)
        parts, size = [], 0
        for start in range(0, left.shape[0], rows):
            block = left if rows >= left.shape[0] else left[start:start + rows]
            part = self._check(block @ right)
            if support is not None and part.nnz:
                coo = part.tocoo()
                keep = np.asarray(support[(coo.row + start) // dims[0],
                                          coo.col // dims[1]]).ravel().astype(bool)
                part = csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=part.shape)
            size += _bytes(part)
            if size > self.memory_bytes:
                raise MemoryError('Sparse product exceeds memory_bytes; no truncation was applied')
            parts.append(part)
        return parts[0] if len(parts) == 1 else self._check(vstack(parts, format='csr'))

    def _suffix_support(self, factors):
        """Conservative exact physical-label support of the remaining MPO.

        Spin components and virtual MPS sectors are unioned, never selected.
        Therefore pruning can remove only structurally impossible suffixes.
        """
        key = id(factors)
        if key in self._supports:
            return self._supports[key]
        if self.state.tie != 'physical':
            # Fusion labels depend on virtual sectors, not physical operators.
            return None
        if factors is self.identity:
            supports = [{0: eye(n, dtype=bool, format='csr')}
                        for n in self.state.frontier_states]
            self._supports[key] = supports
            return supports
        supports = [None] * (self.state.nsites + 1)
        supports[-1] = {0: eye(1, dtype=bool, format='csr')}
        for i in reversed(range(self.state.nsites)):
            old, new = self.state.frontiers[i:i+2]
            old_index = {tuple(a): j for j, a in enumerate(self.state._assignments[i])}
            maps = {}
            for p, label in self.state._tie_index_by_site[i].items():
                maps[p] = np.array([old_index[tuple(label if s == i else a[new.index(s)]
                                                       for s in old)]
                                    for a in self.state._assignments[i+1]])
            transitions = self._physical_transitions(i, factors[i])
            result = {}
            n = len(self.state._assignments[i])
            for pb, pk, lc, rc in transitions:
                tail = supports[i+1].get(rc)
                if tail is None:
                    continue
                tail = tail.tocoo()
                matrix = csr_matrix((np.ones(tail.nnz, dtype=bool),
                    (maps[pb][tail.row], maps[pk][tail.col])), shape=(n, n))
                result[lc] = matrix if lc not in result else result[lc] + matrix
            supports[i] = result
        self._supports[key] = supports
        return supports

    def _physical_transitions(self, index, core):
        """Union raw reduced-operator support before any spin recoupling.

        Ignoring interference and spin selection only enlarges this mask.
        In particular, a cancellation in a magnetic component cannot remove
        a route that contributes after reduced recoupling.
        """
        result = set()
        sectors = self.state.physical_sectors_by_site[index]
        for pb in sectors:
            for pk in sectors:
                dense = core.dense_blocks.get((pb, pk))
                if dense is not None:
                    result.update((pb, pk, lc, rc) for lc, rc, v in iter_virtual_routes(dense)
                                  if np.any(v))
                for term in core.reduced_terms:
                    operator = term.reduced_operator
                    supported = False
                    for m in ordered_two_m_values(_reduced_operator_rank(operator)):
                        block = operator.component_block(m, pb, pk)
                        if block is not None and np.any(block):
                            supported = True
                            break
                    if supported:
                        result.update((pb, pk, lc, rc)
                                      for lc, rc, v in iter_virtual_routes(term.visible_virtual_block)
                                      if np.any(v))
        return result

    def site(self, index, vector=None):
        """Scatter tied parameters directly to CSR virtual-sector matrices."""
        plan = self.state._wigner_eckart_route_plan(index)
        vector = self.state._pack_site(index) if vector is None else np.asarray(vector)
        if vector.shape != (plan.nparameters,):
            raise ValueError('Incorrect tied-parameter vector shape')
        result = {}
        size = 0
        for key, parameters, targets, shape in plan.groups:
            if shape[1] != 1:
                raise NotImplementedError('Physical multiplicity must be one')
            rows, cols = np.divmod(targets, shape[2])
            matrix = self._check(csr_matrix((vector[parameters], (rows, cols)),
                                           shape=(shape[0], shape[2])))
            size += _bytes(matrix)
            if size > self.memory_bytes:
                raise MemoryError('Sparse site exceeds memory_bytes')
            result[key] = matrix
        self.stats['peak_site_bytes'] = max(self.stats['peak_site_bytes'], size)
        return result

    def _endpoint(self, factors, side):
        index, axis = (0, 0) if side == 'left' else (-1, 2)
        channel = 1 if side == 'left' and getattr(factors[0], 'normal_complementary_plan', None) is not None else 0
        return {(q, q, channel, 0): eye(d, format='csr')
                for q, d in self.state._base_sites[index].legs[axis].dims.items()}

    def _routes(self, index, core):
        keys = self.state._base_sites[index].data
        for bra in keys:
            for ket in keys:
                reduced = _left_reduced_rank_coupled_block(core, bra[0], ket[0],
                    bra[1], ket[1], bra[2], ket[2])
                for (lc, rc), weights in reduced.items():
                    for x, y in zip(*np.nonzero(weights[:, :, 0, 0])):
                        yield bra, ket, (bra[0], ket[0], lc, int(x)), \
                            (bra[2], ket[2], rc, int(y)), weights[x, y, 0, 0]

    def advance(self, boundary, index, factors, *, direction='lr'):
        """Advance one sparse boundary; reverse transport is the exact adjoint sum."""
        if direction not in ('lr', 'rl'):
            raise ValueError('Expected lr or rl')
        sites, out, size = self.site(index), {}, 0
        supports = self._suffix_support(factors) if direction == 'lr' else None
        for bra, ket, left, right, weight in self._routes(index, factors[index]):
            incoming, outgoing = (left, right) if direction == 'lr' else (right, left)
            parent = boundary.get(incoming)
            if parent is None:
                continue
            a, b = sites[bra], sites[ket]
            if direction == 'lr':
                support = None if supports is None else supports[index+1].get(outgoing[2])
                if supports is not None and support is None:
                    continue
                dims = (self.state._base_sites[index].legs[2].dims[bra[2]],
                        self.state._base_sites[index].legs[2].dims[ket[2]])
                value = self._product(self._product(a.conj().T, parent), b,
                                      support=support, dims=dims)
            else:
                value = self._product(self._product(a.conj(), parent), b.T)
            if value.nnz == 0:
                continue
            value = weight * value
            old = out.get(outgoing)
            if old is not None:
                size -= _bytes(old)
                value = self._check(old + value)
            out[outgoing] = value
            size += _bytes(value)
            if size > self.memory_bytes:
                raise MemoryError(f'Sparse frontier boundary exceeds {self.memory_bytes} bytes at site {index}')
        self.stats['peak_boundary_bytes'] = max(self.stats['peak_boundary_bytes'], size)
        return out

    def contract(self, *, identity=False):
        """Unnormalized exact contraction, retaining only the moving frontier."""
        factors = self.identity if identity else self.hamiltonian
        boundary = self._endpoint(factors, 'left')
        for i in range(self.state.nsites):
            boundary = self.advance(boundary, i, factors)
        endpoint = self._endpoint(factors, 'right')
        return sum((boundary[key].multiply(value).sum()
                    for key, value in endpoint.items() if key in boundary), 0j)

    def norm(self):
        value = self.contract(identity=True)
        if abs(value.imag) > 1e-10 * max(abs(value.real), 1e-30) or value.real <= 0:
            raise ArithmeticError(f'Invalid sparse frontier norm: {value}')
        return float(value.real)

    def expectation(self):
        value = self.contract() / self.norm()
        if abs(value.imag) > 1e-9:
            raise ArithmeticError(f'Nonreal sparse frontier energy: {value}')
        return float(value.real) + self.state.ecore

    def local_operator(self, index, *, identity=False):
        """Return an exact matrix-free action on one site's tied parameters.

        Environments are frozen, so changing the active tensor is allowed.
        Changing any other tensor requires constructing a new operator.
        No dense local metric, parameter basis, or pair space is allocated.
        """
        if not 0 <= index < self.state.nsites:
            raise IndexError(index)
        factors = self.identity if identity else self.hamiltonian
        left, right = self._endpoint(factors, 'left'), self._endpoint(factors, 'right')
        for i in range(index):
            left = self.advance(left, i, factors)
        for i in range(self.state.nsites - 1, index, -1):
            right = self.advance(right, i, factors, direction='rl')
        return self._local_operator(index, factors, left, right)

    def _local_operator(self, index, factors, left, right):
        plan = self.state._wigner_eckart_route_plan(index)
        groups = {key: (parameters, *np.divmod(targets, shape[2]))
                  for key, parameters, targets, shape in plan.groups}
        routes = {}
        for bra, ket, lkey, rkey, weight in self._routes(index, factors[index]):
            if lkey not in left or rkey not in right:
                continue
            parameters, rows, cols = groups[bra]
            routes.setdefault((lkey, ket), []).append(
                (rkey, weight, parameters, rows, cols))
        # Freeze boundary transposes once. Only one shared intermediate is
        # retained at a time, and it is recomputed for each trial vector.
        right_transposes = {key: value.T for key, value in right.items()}
        def action(vector):
            sites = self.site(index, np.asarray(vector).reshape(-1))
            result = np.zeros(plan.nparameters, dtype=np.result_type(vector, complex))
            for (lkey, ket), terms in routes.items():
                intermediate = self._product(left[lkey], sites[ket])
                for rkey, weight, parameters, rows, cols in terms:
                    value = self._product(intermediate, right_transposes[rkey])
                    np.add.at(result, parameters,
                              weight * np.asarray(value[rows, cols]).ravel())
            return result

        return LinearOperator((plan.nparameters, plan.nparameters), matvec=action, dtype=complex)

    def parameter_actions(self, *, identity=False, directory=None):
        """Yield exact current-state actions with disk-backed right boundaries.

        At most one right checkpoint and the advancing left boundary are
        resident. Temporary files are removed on generator close or failure.
        """
        factors = self.identity if identity else self.hamiltonian
        with TemporaryDirectory(dir=directory) as folder:
            path = Path(folder)
            right = self._endpoint(factors, 'right')
            for i in range(self.state.nsites - 1, -1, -1):
                with (path / str(i)).open('wb') as handle:
                    pickle.dump(right, handle, protocol=pickle.HIGHEST_PROTOCOL)
                right = self.advance(right, i, factors, direction='rl')
            del right
            left = self._endpoint(factors, 'left')
            for i in range(self.state.nsites):
                with (path / str(i)).open('rb') as handle:
                    right = pickle.load(handle)
                (path / str(i)).unlink()
                operator = self._local_operator(i, factors, left, right)
                yield operator @ self.state._pack_site(i)
                del operator, right
                left = self.advance(left, i, factors)

    def sweep(self, *, direction='lr', directory=None, tolerance=1e-8,
              maxiter=200, max_space=48, callback=None):
        """One fixed-layout variational pass using sparse H and N actions.

        Uses the existing generalized Davidson solver (Davidson, J. Comput.
        Phys. 17, 87--94, 1975, doi:10.1016/0021-9991(75)90065-0), with
        an unpreconditioned residual expansion and bounded Krylov workspace.
        No pair split, dense norm whitening or sector growth is performed.
        Each root must pass its true residual and local energy-descent checks
        before commitment. A pass is not a full-state convergence certificate.
        ``callback(update)`` runs after each accepted site for progress or saving.
        """
        from pyqed.mps.nonabelian.solver import _solve_packed_generalized_davidson
        if direction not in ('lr', 'rl'):
            raise ValueError('Expected lr or rl')
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('tolerance must be positive and finite')
        state = self.state
        state.converged = False
        state.success = None
        traversal = list(range(state.nsites))
        if direction == 'rl':
            traversal.reverse()
        opposite = 'rl' if direction == 'lr' else 'lr'
        incoming = 'left' if direction == 'lr' else 'right'
        far = 'right' if direction == 'lr' else 'left'
        operators = (self.hamiltonian, self.identity)
        updates = []
        with TemporaryDirectory(dir=directory) as folder:
            path = Path(folder)
            for j, factors in enumerate(operators):
                boundary = self._endpoint(factors, far)
                for i in reversed(traversal):
                    with (path / f'{j}-{i}').open('wb') as handle:
                        pickle.dump(boundary, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    boundary = self.advance(boundary, i, factors, direction=opposite)
                del boundary
            moving = [self._endpoint(f, incoming) for f in operators]
            for i in traversal:
                started = perf_counter()
                local = []
                for j, factors in enumerate(operators):
                    file = path / f'{j}-{i}'
                    with file.open('rb') as handle:
                        boundary = pickle.load(handle)
                    file.unlink()
                    left, right = (moving[j], boundary) if direction == 'lr' else (boundary, moving[j])
                    local.append(self._local_operator(i, factors, left, right))
                h, n = local
                current = state._pack_site(i)
                norm = float(np.vdot(current, n @ current).real)
                before = float(np.vdot(current, h @ current).real / norm)
                energy, vector, info = _solve_packed_generalized_davidson(
                    current, h.matvec, h_diag=np.zeros(current.size),
                    N=n.matvec, n_diag=np.ones(current.size), tol=max(tolerance**2, 1e-12),
                    tol_residual=tolerance, itermax=maxiter, max_space=max_space,
                    precond=lambda residual, theta, v: residual,
                    use_block_preconditioner=False,
                    workspace_bytes=self.memory_bytes)
                nv, hv = n @ vector, h @ vector
                weight = np.vdot(vector, nv)
                energy = np.vdot(vector, hv) / weight
                residual = float(np.linalg.norm(hv - energy * nv))
                if (not info['davidson_converged'] or not np.isfinite(energy)
                        or abs(energy.imag) > 1e-9 or weight.real <= 0
                        or residual > tolerance or energy.real > before + 1e-10):
                    raise ArithmeticError(f'Sparse frontier local solve failed at site {i}: '
                                          f'energy={energy}, residual={residual}, before={before}')
                state._set_site_vector(i, vector)
                state.energy = float(energy.real) + state.ecore
                state.converged = False
                state.success = None
                updates.append(dict(site=i, direction=direction, energy=state.energy,
                                    energy_before=before+state.ecore, residual=residual,
                                    elapsed_s=perf_counter()-started))
                if callback is not None:
                    callback(updates[-1].copy())
                del local, h, n, boundary, left, right
                for j, factors in enumerate(operators):
                    moving[j] = self.advance(moving[j], i, factors, direction=direction)
        state.message = 'Completed sparse-frontier pass; full-cycle convergence not assessed'
        return updates
