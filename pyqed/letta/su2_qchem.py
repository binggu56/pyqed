"""Reduced SU(2) LETTA for spatial-orbital quantum chemistry.

This module implements the first non-Abelian LETTA variant.  A tie conditions
on the invariant local multiplet label ``(N, S)``; it never copies a magnetic
projection.  Open tie assignments are embedded in virtual multiplicity space,
so the materialized state remains a conventional reduced SU(2) MPS and can use
the existing rank-coupled quantum-chemistry MPO contractions.

The one-site optimizer contracts native Wigner--Eckart routes throughout.
Reduced structural routes are cached and packed by compatible tensor block;
large local problems are metric-orthonormalized and solved by a projected
matrix-free Davidson iteration.  The two-site optimizer acts in the
channel-resolved reduced pair space and retracts the result to the tied
manifold.  The original polarization construction remains available as a
small-system reference path.  A bounded worker pool can be shared by every
exact construction, with threaded BLAS held to one thread inside parallel
regions.
"""

from __future__ import annotations

from pyqed.parallel import resolve_workers as resolve_default_workers

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass, replace
from itertools import product
from numbers import Integral
from pathlib import Path
import os
import pickle
import time

import numpy as np
from threadpoolctl import threadpool_limits

from pyqed.mps.nonabelian.contraction import merge_mps_sites
from pyqed.mps.nonabelian.environment import (
    AdjacentPairTransitionPlan,
    BlockSparseEnvironmentChain,
    iter_one_site_reduced_adjoint,
    LocalTransitionPlan,
    _apply_two_site_dense_factorized,
    _left_reduced_rank_coupled_block,
    _component_basis_norm,
    _right_reduced_rank_coupled_block,
    _factorize_left_two_site_dense_term,
    _factorize_right_two_site_dense_term,
    contract_chain_expectation,
    contract_chain_transition,
)
from pyqed.mps.nonabelian.coupling import ordered_two_m_values
from pyqed.mps.nonabelian.decompose import svd_two_site
from pyqed.mps.nonabelian.solver import (
    _resolve_davidson_operator,
    _solve_packed_generalized_davidson,
    pack_two_site_state,
    unpack_two_site_state,
)
from pyqed.mps.nonabelian.mpo import prepare_hamiltonian_factors
from pyqed.mps.nonabelian.states import (
    FullyReducedSpatialOrbitalSite,
    build_random_reduced_spatial_mps,
    spatial_target_sector,
)
from pyqed.mps.nonabelian.sweep import (
    MovingEnvironment,
    _identity_mpo_factors_for_sites_and_mpo,
)
from pyqed.symmetry import IrrepTensor


def resolve_workers(workers, *, maximum=4):
    """Choose one parallel layer without oversubscribing threaded BLAS."""
    if workers is None:
        return resolve_default_workers()
    if isinstance(workers, str):
        if workers.strip().lower() != "auto":
            raise ValueError("workers must be a positive integer or 'auto'.")
        blas_threads = max(
            (
                int(os.environ.get(name, "1") or 1)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            ),
            default=1,
        )
        return 1 if blas_threads > 1 else max(
            1, min(int(maximum), resolve_default_workers())
        )
    if isinstance(workers, (bool, np.bool_)):
        raise TypeError("workers must be a positive integer or 'auto'.")
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be a positive integer or 'auto'.")
    return workers


def _unique(values):
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class _PairChannelEntry:
    key: tuple
    shape: tuple
    offset: int
    size: int


class _ChannelResolvedPairSpace:
    """Packed adjacent-pair space retaining the intermediate reduced sector."""

    def __init__(self, left, right):
        self.left_template = left
        self.right_template = right
        descriptors = {}
        for (q_l, q_p1, q_mid), a_block in left.data.items():
            for (q_mid_b, q_p2, q_r), b_block in right.data.items():
                if q_mid_b != q_mid:
                    continue
                shape = (
                    int(a_block.shape[0]),
                    int(a_block.shape[1]),
                    int(b_block.shape[1]),
                    int(b_block.shape[2]),
                )
                descriptors[(q_l, q_p1, q_p2, q_r, q_mid)] = shape
        offset = 0
        entries = []
        for key in sorted(descriptors):
            shape = descriptors[key]
            size = int(np.prod(shape, dtype=int))
            entries.append(_PairChannelEntry(key, shape, offset, size))
            offset += size
        if not entries:
            raise ValueError("Adjacent reduced pair has no common intermediate sector.")
        self.entries = tuple(entries)
        self.size = int(offset)

    def __iter__(self):
        return iter(self.entries)

    def blocks(self, vector):
        vector = np.asarray(vector)
        if vector.size != self.size:
            raise ValueError("Channel-resolved pair vector has the wrong size.")
        return tuple(
            vector[entry.offset : entry.offset + entry.size].reshape(entry.shape)
            for entry in self.entries
        )

    def pack_blocks(self, blocks, *, dtype=None):
        if dtype is None:
            dtype = np.result_type(*(np.asarray(block).dtype for block in blocks))
        vector = np.zeros(self.size, dtype=dtype)
        for entry, block in zip(self.entries, blocks):
            vector[entry.offset : entry.offset + entry.size] = np.asarray(block).reshape(-1)
        return vector

    def pack_sites(self, left, right):
        dtype = np.result_type(
            *(np.asarray(block).dtype for block in left.data.values()),
            *(np.asarray(block).dtype for block in right.data.values()),
        )
        vector = np.zeros(self.size, dtype=dtype)
        for entry in self.entries:
            q_l, q_p1, q_p2, q_r, q_mid = entry.key
            a_block = left.data.get((q_l, q_p1, q_mid))
            b_block = right.data.get((q_mid, q_p2, q_r))
            if a_block is None or b_block is None:
                continue
            block = np.tensordot(a_block, b_block, axes=([2], [0]))
            vector[entry.offset : entry.offset + entry.size] = block.reshape(-1)
        return vector

    def basis_pair(self, index):
        """Return the two rank-3 tensors for one packed rank-one basis state."""
        index = int(index)
        entry = next(
            (
                candidate
                for candidate in self.entries
                if candidate.offset <= index < candidate.offset + candidate.size
            ),
            None,
        )
        if entry is None:
            raise IndexError(f"Pair basis index {index} is out of range.")
        local = index - entry.offset
        i_l, i_p1, i_p2, i_r = np.unravel_index(local, entry.shape)
        q_l, q_p1, q_p2, q_r, q_mid = entry.key
        left_block = np.zeros((entry.shape[0], entry.shape[1], 1), dtype=complex)
        right_block = np.zeros((1, entry.shape[2], entry.shape[3]), dtype=complex)
        left_block[i_l, i_p1, 0] = 1.0
        right_block[0, i_p2, i_r] = 1.0
        left = IrrepTensor(
            {(q_l, q_p1, q_mid): left_block},
            [
                list(self.left_template.qns[0]),
                list(self.left_template.qns[1]),
                [q_mid],
            ],
            list(self.left_template.dirs),
            fusion_legs=list(self.left_template.fusion_legs),
            metadata=self._clean_metadata(self.left_template.metadata),
        )
        right = IrrepTensor(
            {(q_mid, q_p2, q_r): right_block},
            [
                [q_mid],
                list(self.right_template.qns[1]),
                list(self.right_template.qns[2]),
            ],
            list(self.right_template.dirs),
            fusion_legs=list(self.right_template.fusion_legs),
            metadata=self._clean_metadata(self.right_template.metadata),
        )
        return left, right

    @staticmethod
    def _clean_metadata(metadata):
        cleaned = dict(metadata or {})
        for key in tuple(cleaned):
            if key.startswith("_rank_coupled_site_entries_by_"):
                cleaned.pop(key, None)
        return cleaned

    def split(self, vector, *, cutoff=1.0e-13):
        vector = np.asarray(vector, dtype=complex).reshape(-1)
        if vector.size != self.size:
            raise ValueError("Channel-resolved pair vector has the wrong size.")
        entries_by_mid = {}
        for entry in self.entries:
            entries_by_mid.setdefault(entry.key[-1], []).append(entry)

        left_data = {}
        right_data = {}
        bond_sectors = []
        for q_mid, entries in sorted(entries_by_mid.items()):
            row_specs = sorted(
                {
                    (entry.key[0], entry.key[1], entry.shape[0], entry.shape[1])
                    for entry in entries
                }
            )
            col_specs = sorted(
                {
                    (entry.key[2], entry.key[3], entry.shape[2], entry.shape[3])
                    for entry in entries
                }
            )
            row_slices = {}
            col_slices = {}
            offset = 0
            for spec in row_specs:
                width = int(spec[2] * spec[3])
                row_slices[spec[:2]] = slice(offset, offset + width)
                offset += width
            nrow = offset
            offset = 0
            for spec in col_specs:
                width = int(spec[2] * spec[3])
                col_slices[spec[:2]] = slice(offset, offset + width)
                offset += width
            matrix = np.zeros((nrow, offset), dtype=complex)
            for entry in entries:
                q_l, q_p1, q_p2, q_r, _ = entry.key
                block = vector[
                    entry.offset : entry.offset + entry.size
                ].reshape(entry.shape)
                matrix[row_slices[(q_l, q_p1)], col_slices[(q_p2, q_r)]] = (
                    block.reshape(entry.shape[0] * entry.shape[1], -1)
                )

            u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
            scale = max(float(np.max(singular, initial=0.0)), 1.0)
            rank = int(np.count_nonzero(singular > float(cutoff) * scale))
            if rank == 0:
                continue
            u = u[:, :rank]
            right_factor = singular[:rank, None] * vh[:rank]
            bond_sectors.extend([q_mid] * rank)
            for q_l, q_p1, d_l, d_p1 in row_specs:
                left_data[(q_l, q_p1, q_mid)] = u[
                    row_slices[(q_l, q_p1)], :
                ].reshape(d_l, d_p1, rank)
            for q_p2, q_r, d_p2, d_r in col_specs:
                right_data[(q_mid, q_p2, q_r)] = right_factor[
                    :, col_slices[(q_p2, q_r)]
                ].reshape(rank, d_p2, d_r)

        if not bond_sectors:
            raise ValueError("Cannot split a null channel-resolved pair vector.")
        left = IrrepTensor(
            left_data,
            [
                list(self.left_template.qns[0]),
                list(self.left_template.qns[1]),
                list(bond_sectors),
            ],
            list(self.left_template.dirs),
            fusion_legs=list(self.left_template.fusion_legs),
            metadata=self._clean_metadata(self.left_template.metadata),
        )
        right = IrrepTensor(
            right_data,
            [
                list(bond_sectors),
                list(self.right_template.qns[1]),
                list(self.right_template.qns[2]),
            ],
            list(self.right_template.dirs),
            fusion_legs=list(self.right_template.fusion_legs),
            metadata=self._clean_metadata(self.right_template.metadata),
        )
        return left, right


class _ReducedMetricComponents:
    """Connected projected-norm blocks in tied reduced coordinates."""

    def __init__(self, dimension, components):
        self.dimension = int(dimension)
        self.components = tuple(
            (
                np.asarray(indices, dtype=np.int64),
                np.asarray(block, dtype=complex),
            )
            for indices, block in components
        )

    def dense(self):
        metric = np.zeros((self.dimension, self.dimension), dtype=complex)
        for indices, block in self.components:
            metric[np.ix_(indices, indices)] += block
        return metric

    def __array__(self, dtype=None, copy=None):
        metric = self.dense()
        if dtype is not None:
            metric = metric.astype(dtype, copy=False)
        return np.array(metric, copy=True) if copy else metric

    @property
    def diagonal(self):
        diagonal = np.zeros(self.dimension, dtype=float)
        for indices, block in self.components:
            diagonal[indices] += np.real(np.diag(block))
        return diagonal

    @property
    def nbytes(self):
        return int(sum(indices.nbytes + block.nbytes for indices, block in self.components))

    def matvec(self, vector):
        vector = np.asarray(vector)
        output = np.zeros_like(vector, dtype=np.result_type(vector, complex))
        for indices, block in self.components:
            output[indices] += block @ vector[indices]
        return output

    def whitened_norm(self, vector, *, rtol):
        """Apply the existing component whitening without its global zero panels."""
        vector = np.asarray(vector).reshape(self.dimension)
        scale = max((float(np.max(np.abs(block), initial=0.0))
                     for _, block in self.components), default=0.0)
        norm_squared = 0.0
        retained = False
        for indices, block in self.components:
            result = self._whiten_component(indices, block, threshold=float(rtol)*scale)
            if result is not None:
                projected = result[0].conj().T @ vector[indices]
                norm_squared += float(np.real(np.vdot(projected, projected)))
                retained = True
        if not retained:
            raise np.linalg.LinAlgError("the local SU2 metric support is null")
        return np.sqrt(norm_squared)

    def restrict(self, support):
        support = np.asarray(support, dtype=np.int64)
        inverse = {int(old): new for new, old in enumerate(support)}
        components = []
        for indices, block in self.components:
            keep = np.asarray(
                [position for position, index in enumerate(indices) if int(index) in inverse],
                dtype=np.int64,
            )
            if keep.size == 0:
                continue
            new_indices = np.asarray(
                [inverse[int(indices[position])] for position in keep],
                dtype=np.int64,
            )
            components.append((new_indices, block[np.ix_(keep, keep)]))
        return type(self)(support.size, components)

    @staticmethod
    def _whiten_component(indices, raw_block, *, threshold):
        block = 0.5 * (raw_block + raw_block.conj().T)
        diagonal = np.real(np.diag(block))
        off_diagonal = block - np.diag(diagonal)
        if not np.any(off_diagonal):
            keep = diagonal > threshold
            if not np.any(keep):
                return None
            local = np.zeros(
                (indices.size, np.count_nonzero(keep)), dtype=complex
            )
            local[np.flatnonzero(keep), np.arange(np.count_nonzero(keep))] = (
                1.0 / np.sqrt(diagonal[keep])
            )
            return local, diagonal[keep], "diagonal"
        values, vectors = np.linalg.eigh(block)
        keep = values > threshold
        if not np.any(keep):
            return None
        return vectors[:, keep] / np.sqrt(values[keep]), values[keep], "eigh"

    def whitener(self, *, rtol, component_cache=None, revision=None, sparse=False):
        """Whiten independent components using a relative metric cutoff.

        ``sparse=True`` returns CSR coordinates assembled directly from local
        blocks, avoiding full-height zero panels. The default remains dense.
        """
        scale = max(
            (
                float(np.max(np.abs(block), initial=0.0))
                for _indices, block in self.components
            ),
            default=0.0,
        )
        threshold = float(rtol) * scale
        columns = []
        retained = []
        backends = []
        reused = 0
        for component_index, (indices, raw_block) in enumerate(self.components):
            cache_key = None
            result = None
            if component_cache is not None and revision is not None:
                component_revision = int(revision(indices, raw_block))
                cache_key = (component_index, float(threshold), component_revision)
                result = component_cache.get(cache_key)
                reused += result is not None
            if result is None:
                result = self._whiten_component(
                    indices, raw_block, threshold=threshold
                )
                if component_cache is not None and cache_key is not None:
                    component_cache[cache_key] = result
            if result is None:
                continue
            local, values, component_backend = result
            backends.append(component_backend)
            if sparse:
                rows, cols = np.nonzero(local)
                columns.append((indices[rows], len(retained)+cols, local[rows, cols]))
            else:
                lifted = np.zeros((self.dimension, local.shape[1]), dtype=complex)
                lifted[indices] = local
                columns.append(lifted)
            retained.extend(map(float, values))
        if not columns:
            raise np.linalg.LinAlgError(
                "the projected SU2LETTA parameter metric is singular."
            )
        backend = (
            "conditional_identity"
            if backends and all(value == "diagonal" for value in backends)
            else "block_eigh"
        )
        if sparse:
            from scipy.sparse import coo_matrix
            rows, cols, values = (np.concatenate([part[i] for part in columns]) for i in range(3))
            transform = coo_matrix((values, (rows, cols)),
                                   shape=(self.dimension, len(retained))).tocsr()
        else:
            transform = np.column_stack(columns)
        return transform, np.asarray(retained), backend, int(reused)

    def orthonormal_components(self, whitener):
        """Return the retained coordinate columns belonging to each block."""
        from scipy.sparse import issparse
        sparse = issparse(whitener)
        if not sparse:
            whitener = np.asarray(whitener)
        if whitener.shape[0] != self.dimension:
            raise ValueError("Metric whitener has the wrong row dimension.")
        groups = []
        for indices, _block in self.components:
            columns = np.flatnonzero((whitener[indices] != 0).getnnz(axis=0) if sparse else
                                     np.any(whitener[indices] != 0.0, axis=0))
            if columns.size:
                groups.append(columns)
        return tuple(groups)


class _FactorizedPairWhitener:
    """Spectral pair-metric support using small left/right boundary factors."""

    def __init__(self, dimension, blocks, rtol):
        decomposed = []
        largest = 0.0
        for entry, left, right in blocks:
            lv, lu = np.linalg.eigh((left + left.conj().T) * .5)
            rv, ru = np.linalg.eigh((right + right.conj().T) * .5)
            positive_l, positive_r = lv > 0, rv > 0
            lv, lu = lv[positive_l], lu[:, positive_l]
            rv, ru = rv[positive_r], ru[:, positive_r]
            values = lv[:, None] * rv[None, :]
            largest = max(largest, float(values.max(initial=0.0)))
            decomposed.append((entry, lv, lu, rv, ru, values))
        self.dimension = int(dimension)
        self.blocks = []
        self.rank = 0
        for entry, lv, lu, rv, ru, values in decomposed:
            keep = values > float(rtol) * largest
            count = int(keep.sum())
            if count:
                self.blocks.append((entry, lu/np.sqrt(lv), ru/np.sqrt(rv),
                                    keep, self.rank, count))
                self.rank += count
        if not self.rank:
            raise np.linalg.LinAlgError('Pair metric has no retained positive support')

    def matvec(self, vector):
        out = np.zeros(self.dimension, dtype=complex)
        for entry, left, right, keep, offset, count in self.blocks:
            core = np.zeros(keep.shape, dtype=complex)
            core[keep] = vector[offset:offset+count]
            out[entry.offset:entry.offset+entry.size] = (left @ core @ right.T).ravel()
        return out

    def rmatvec(self, vector):
        out = np.empty(self.rank, dtype=complex)
        for entry, left, right, keep, offset, count in self.blocks:
            block = vector[entry.offset:entry.offset+entry.size].reshape(left.shape[0], right.shape[0])
            out[offset:offset+count] = (left.conj().T @ block @ right.conj())[keep]
        return out


class _MatrixFreeProjectedMetric:
    """Exact projected norm action without a dense tied-coordinate metric."""

    def __init__(self, dimension, diagonal, action):
        self.dimension = int(dimension)
        self.diagonal = np.asarray(diagonal, dtype=float).reshape(self.dimension)
        self._action = action

    def matvec(self, vector):
        return np.asarray(self._action(vector), dtype=complex).reshape(
            self.dimension
        )


class _MetricPreconditioner:
    """Bounded CG inverse of a diagonal-scaled regularized local metric.

    Metric-informed adaptation of Davidson preconditioning, not an exact
    inverse of H-E*S. Davidson, J. Comput. Phys. 17, 87 (1975),
    doi:10.1016/0021-9991(75)90065-0. Never forms a dense metric.
    """

    def __init__(self, metric, diagonal, *, iterations=6, regularization=1e-3):
        diagonal = np.asarray(diagonal, dtype=float)
        floor = max(float(np.max(np.abs(diagonal))) * 1e-12, 1e-30)
        self.scale = 1 / np.sqrt(np.maximum(diagonal, floor))
        self.metric = metric
        self.iterations = int(iterations)
        self.regularization = float(regularization)
        self.calls = self.matvecs = 0

    def __call__(self, residual, theta, vector):
        del theta, vector
        rhs = self.scale * residual
        r = rhs.copy()
        p = r.copy()
        x = np.zeros_like(rhs)
        rr = float(np.vdot(r, r).real)
        target = rr * 1e-4
        self.calls += 1
        for _ in range(self.iterations):
            if rr <= max(target, 1e-30):
                break
            ap = self.scale * self.metric(self.scale * p) + self.regularization * p
            self.matvecs += 1
            pap = float(np.vdot(p, ap).real)
            if not np.isfinite(pap) or pap <= 0:
                break
            alpha = rr / pap
            x += alpha * p
            r -= alpha * ap
            next_rr = float(np.vdot(r, r).real)
            p = r + (next_rr / rr) * p
            rr = next_rr
        return self.scale * x


def _canonical_graph(nsites, graph):
    if graph is None:
        return ()
    if isinstance(graph, str):
        key = graph.lower().replace("-", "_")
        if key in {"nn", "nearest_neighbor"}:
            return _nearest_neighbor_graph(nsites)
        ranges = {"nnn": 2, "next_nearest_neighbor": 2,
                  "nnnn": 3, "third_neighbor": 3}
        if key in ranges:
            return tuple((i, j) for i in range(nsites)
                         for j in range(i + 1, min(nsites, i + ranges[key] + 1)))
        raise ValueError("SU2LETTA graph strings must be 'nn', 'nnn', or 'nnnn' (or their long forms).")
    if hasattr(graph, "is_directed") and graph.is_directed():
        raise ValueError("SU2LETTA graph must be undirected.")
    if hasattr(graph, "edges"):
        edges = graph.edges() if callable(graph.edges) else graph.edges
    elif isinstance(graph, dict):
        edges = (
            (site, neighbor)
            for site, neighbors in graph.items()
            for neighbor in neighbors
        )
    else:
        edges = graph
    out = set()
    for edge in edges:
        edge = tuple(edge)
        if len(edge) != 2:
            raise ValueError("each SU2LETTA graph edge must contain two sites.")
        left, right = (int(value) for value in edge)
        if left == right:
            raise ValueError("an SU2LETTA tie cannot connect a site to itself.")
        if min(left, right) < 0 or max(left, right) >= int(nsites):
            raise ValueError("SU2LETTA graph edges must reference valid sites.")
        out.add(tuple(sorted((left, right))))
    return tuple(sorted(out))


def _nearest_neighbor_graph(nsites):
    """Return the bounded-width default tie graph for an orbital chain."""
    return tuple((site, site + 1) for site in range(int(nsites) - 1))


def _frontiers(nsites, parents):
    frontiers = []
    for cut in range(int(nsites) + 1):
        active = []
        for future in range(cut, int(nsites)):
            if any(future in parents[site] for site in range(cut)):
                active.append(future)
        frontiers.append(tuple(active))
    return tuple(frontiers)


def _real_scalar(value, *, atol=1.0e-9):
    value = complex(value)
    if abs(value.imag) > atol * max(1.0, abs(value.real)):
        raise FloatingPointError(f"expected a real SU(2) expectation value, got {value!r}.")
    return float(value.real)


def _lowest_generalized_pair(hamiltonian, metric, *, rtol=1.0e-11):
    metric = 0.5 * (np.asarray(metric) + np.asarray(metric).conj().T)
    hamiltonian = 0.5 * (
        np.asarray(hamiltonian) + np.asarray(hamiltonian).conj().T
    )
    values, vectors = np.linalg.eigh(metric)
    scale = max(float(np.max(np.abs(values))), 1.0)
    keep = values > float(rtol) * scale
    if not np.any(keep):
        raise np.linalg.LinAlgError("the local SU(2)-LETTA metric is null.")
    basis = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    reduced = basis.conj().T @ hamiltonian @ basis
    energies, coefficients = np.linalg.eigh(0.5 * (reduced + reduced.conj().T))
    vector = basis @ coefficients[:, 0]
    norm = np.vdot(vector, metric @ vector)
    vector /= np.sqrt(np.real(norm))
    return float(np.real(energies[0])), vector


class _WignerEckartRoutePlan:
    """Linear map from tied parameters to grouped reduced tensor blocks."""

    def __init__(self, *, template, groups, nparameters, nbytes):
        self.template = template
        self.groups = tuple(groups)
        self.nparameters = int(nparameters)
        self.nbytes = int(nbytes)
        self.route_count = int(sum(indices.size for _, indices, _, _ in self.groups))
        self.backend = "block-grouped-gemm"
        self._group_by_key = {
            key: (indices, matrix, shape)
            for key, indices, matrix, shape in self.groups
        }
        self._basis = None

    def block_basis(self, key, *, dtype=complex):
        """Return all parameter directions for one reduced tensor block."""
        record = self._group_by_key.get(key)
        if record is None:
            shape = np.asarray(self.template.data[key]).shape
            return np.zeros((self.nparameters,) + shape, dtype=dtype)
        indices, matrix, shape = record
        blocks = np.zeros(
            (self.nparameters,) + tuple(shape),
            dtype=np.result_type(dtype, matrix.dtype),
        )
        blocks[indices] = np.asarray(matrix).reshape((-1,) + tuple(shape))
        return blocks

    def active_block_basis(self, key, *, dtype=complex):
        """Return only nonzero parameter directions and their global indices."""
        record = self._group_by_key.get(key)
        if record is None:
            shape = np.asarray(self.template.data[key]).shape
            return np.empty(0, dtype=np.int64), np.empty((0,) + shape, dtype=dtype)
        indices, matrix, shape = record
        order = np.argsort(indices)
        rows = np.asarray(matrix)[order]
        keep = np.any(rows != 0.0, axis=1)
        return np.asarray(indices)[order][keep], np.asarray(
            rows[keep], dtype=np.result_type(dtype, matrix.dtype)
        ).reshape((-1,) + tuple(shape))

    @property
    def basis(self):
        """Materialize unit-direction tensors only for reference solvers."""
        if self._basis is None:
            basis_data = [dict() for _ in range(self.nparameters)]
            for key, indices, matrix, shape in self.groups:
                rows = np.asarray(matrix).reshape((-1,) + tuple(shape))
                for row, parameter in enumerate(indices):
                    basis_data[int(parameter)][key] = rows[row]
            self._basis = tuple(
                IrrepTensor(
                    data=data,
                    qns=[list(values) for values in self.template.qns],
                    dirs=list(self.template.dirs),
                    fusion_legs=list(self.template.fusion_legs),
                    metadata=dict(self.template.metadata),
                )
                for data in basis_data
            )
        return self._basis

    def tensor(self, coefficients):
        coefficients = np.asarray(coefficients).reshape(-1)
        if coefficients.size != self.nparameters:
            raise ValueError("Wigner--Eckart route coefficients have the wrong size.")
        data = {}
        for key, indices, matrix, shape in self.groups:
            # All parameter routes contributing to a compatible reduced block
            # are packed into one BLAS matrix-vector product.
            data[key] = (coefficients[indices] @ matrix).reshape(shape)
        return IrrepTensor(
            data=data,
            qns=[list(values) for values in self.template.qns],
            dirs=list(self.template.dirs),
            fusion_legs=list(self.template.fusion_legs),
            metadata=dict(self.template.metadata),
        )

    def apply_delta(self, tensor, coefficients):
        """Apply a tied parameter delta directly to materialized blocks."""
        coefficients = np.asarray(coefficients).reshape(-1)
        if coefficients.size != self.nparameters:
            raise ValueError("Wigner--Eckart route coefficients have the wrong size.")
        for key, indices, matrix, shape in self.groups:
            delta = (coefficients[indices] @ matrix).reshape(shape)
            tensor.data[key] = np.asarray(tensor.data[key]) + delta
        return tensor


class _FrontierTieRoutePlan(_WignerEckartRoutePlan):
    """Reduced scatter plan for arbitrary frontier ties."""

    def __init__(self, *, template, groups, nparameters, nbytes):
        self.template = template
        self.groups = tuple(groups)
        self.nparameters = int(nparameters)
        self.nbytes = int(nbytes)
        self.route_count = int(sum(parameters.size for _, parameters, _, _ in groups))
        self.backend = "frontier-block-scatter"
        self._group_by_key = {
            key: (parameters, targets, shape)
            for key, parameters, targets, shape in groups
        }
        self._basis = None

    def block_basis(self, key, *, dtype=complex):
        record = self._group_by_key.get(key)
        shape = np.asarray(self.template.data[key]).shape
        blocks = np.zeros((self.nparameters,) + shape, dtype=dtype)
        if record is not None:
            parameters, targets, _shape = record
            np.add.at(blocks.reshape(self.nparameters, -1), (parameters, targets), 1.0)
        return blocks

    def active_block_basis(self, key, *, dtype=complex):
        record = self._group_by_key.get(key)
        shape = np.asarray(self.template.data[key]).shape
        if record is None:
            return np.empty(0, dtype=np.int64), np.empty((0,) + shape, dtype=dtype)
        parameters, targets, _shape = record
        active, rows = np.unique(parameters, return_inverse=True)
        blocks = np.zeros((active.size,) + shape, dtype=dtype)
        np.add.at(blocks.reshape(active.size, -1), (rows, targets), 1.0)
        return active, blocks

    @property
    def basis(self):
        """Materialize unit directions only for small reference problems."""
        if self._basis is None:
            basis_data = [dict() for _ in range(self.nparameters)]
            for key, parameters, targets, shape in self.groups:
                for parameter, target in zip(parameters, targets):
                    block = basis_data[int(parameter)].get(key)
                    if block is None:
                        block = np.zeros(shape, dtype=complex)
                        basis_data[int(parameter)][key] = block
                    block.reshape(-1)[int(target)] += 1.0
            self._basis = tuple(
                IrrepTensor(
                    data=data,
                    qns=[list(values) for values in self.template.qns],
                    dirs=list(self.template.dirs),
                    fusion_legs=list(self.template.fusion_legs),
                    metadata=dict(self.template.metadata),
                )
                for data in basis_data
            )
        return self._basis

    def tensor(self, coefficients):
        coefficients = np.asarray(coefficients).reshape(self.nparameters)
        data = {}
        for key, parameters, targets, shape in self.groups:
            block = np.zeros(int(np.prod(shape)), dtype=coefficients.dtype)
            np.add.at(block, targets, coefficients[parameters])
            data[key] = block.reshape(shape)
        return IrrepTensor(
            data=data,
            qns=[list(values) for values in self.template.qns],
            dirs=list(self.template.dirs),
            fusion_legs=list(self.template.fusion_legs),
            metadata=dict(self.template.metadata),
        )

    def adjoint(self, blocks):
        result = np.zeros(self.nparameters, dtype=complex)
        items = blocks.items() if hasattr(blocks, 'items') else blocks
        for key, block in items:
            group = self._group_by_key.get(key)
            if group is not None:
                parameters, targets, _shape = group
                np.add.at(result, parameters, np.asarray(block).reshape(-1)[targets])
        return result

    def apply_delta(self, tensor, coefficients):
        delta = self.tensor(coefficients)
        for key, block in delta.data.items():
            tensor.data[key] = np.asarray(tensor.data[key]) + block
        return tensor


class _MatrixFreePairEmbedding:
    """One-site tangent contraction backed by compact frontier scatter routes."""

    def __init__(self, layout, routes, fixed_site, *, varied_left):
        self.layout = layout
        self.routes = routes
        self.fixed_site = fixed_site
        self.varied_left = bool(varied_left)
        self.pair_dimension = int(layout.size)
        self.parameter_dimension = int(routes.nparameters)
        self._contractions = self._build_contractions()
        self._contraction_batches = self._batch_contractions()

    def _build_contractions(self):
        """Compile fixed block reshapes used by every Krylov application."""
        contractions = []
        for entry in self.layout.entries:
            q_l, q_p1, q_p2, q_r, q_mid = entry.key
            varied_key = (
                (q_l, q_p1, q_mid)
                if self.varied_left
                else (q_mid, q_p2, q_r)
            )
            route = self.routes._group_by_key.get(varied_key)
            if route is None:
                continue
            parameters, targets, varied_shape = route
            fixed_key = (
                (q_mid, q_p2, q_r)
                if self.varied_left
                else (q_l, q_p1, q_mid)
            )
            fixed = self.fixed_site.data.get(fixed_key)
            if fixed is None:
                continue
            varied_shape = tuple(map(int, varied_shape))
            fixed = np.ascontiguousarray(fixed)
            if self.varied_left:
                rows = varied_shape[0] * varied_shape[1]
                shared = varied_shape[2]
                fixed_matrix = fixed.reshape(shared, -1)
            else:
                shared = varied_shape[0]
                rows = int(fixed.shape[0] * fixed.shape[1])
                fixed_matrix = fixed.reshape(rows, shared)
            contractions.append(
                (
                    int(entry.offset),
                    int(entry.size),
                    np.asarray(parameters, dtype=np.int64),
                    np.asarray(targets, dtype=np.int64),
                    varied_shape,
                    rows,
                    shared,
                    fixed_matrix,
                    np.ascontiguousarray(fixed_matrix.conj().T),
                )
            )
        return tuple(contractions)

    def _batch_contractions(self):
        """Group equal matrix shapes into batched BLAS operations."""
        grouped = {}
        for record in self._contractions:
            fixed_matrix = record[-2]
            key = (
                int(record[5]),
                int(record[6]),
                int(fixed_matrix.shape[1]) if self.varied_left
                else int(record[4][1] * record[4][2]),
            )
            grouped.setdefault(key, []).append(record)
        batches = []
        for (rows, shared, output_columns), records in grouped.items():
            batches.append(
                (
                    rows,
                    shared,
                    output_columns,
                    tuple(record[0] for record in records),
                    tuple(record[1] for record in records),
                    tuple(record[2] for record in records),
                    tuple(record[3] for record in records),
                    np.ascontiguousarray(np.stack([record[-2] for record in records])),
                    np.ascontiguousarray(np.stack([record[-1] for record in records])),
                )
            )
        return tuple(batches)

    @property
    def shape(self):
        return self.pair_dimension, self.parameter_dimension

    @property
    def nbytes(self):
        return int(
            self.routes.nbytes
            + sum(
                fixed_adjoint.nbytes
                for (
                    _offset,
                    _size,
                    _parameters,
                    _targets,
                    _shape,
                    _rows,
                    _shared,
                    _fixed_matrix,
                    fixed_adjoint,
                ) in self._contractions
            )
            + sum(
                fixed_matrices.nbytes + fixed_adjoints.nbytes
                for (
                    _rows,
                    _shared,
                    _output_columns,
                    _offsets,
                    _sizes,
                    _parameter_groups,
                    _target_groups,
                    fixed_matrices,
                    fixed_adjoints,
                ) in self._contraction_batches
            )
        )

    def matvec(self, vector):
        vector = np.asarray(vector).reshape(-1)
        if vector.size != self.parameter_dimension:
            raise ValueError("Tied coordinate vector has the wrong size.")
        result = np.zeros(
            self.pair_dimension,
            dtype=np.result_type(vector.dtype, complex),
        )
        for (
            rows,
            shared,
            output_columns,
            offsets,
            sizes,
            parameter_groups,
            target_groups,
            fixed_matrices,
            _fixed_adjoints,
        ) in self._contraction_batches:
            count = len(offsets)
            varied_shape = (
                (count, rows, shared)
                if self.varied_left
                else (count, shared, output_columns)
            )
            varied = np.zeros(varied_shape, dtype=result.dtype)
            for block, parameters, targets in zip(
                varied, parameter_groups, target_groups
            ):
                np.add.at(block.reshape(-1), targets, vector[parameters])
            pairs = (
                varied @ fixed_matrices
                if self.varied_left
                else fixed_matrices @ varied
            )
            for offset, size, pair in zip(offsets, sizes, pairs):
                result[offset : offset + size] = pair.reshape(-1)
        return result

    def rmatvec(self, vector):
        vector = np.asarray(vector).reshape(-1)
        if vector.size != self.pair_dimension:
            raise ValueError("Reduced pair vector has the wrong size.")
        result = np.zeros(
            self.parameter_dimension,
            dtype=np.result_type(vector.dtype, complex),
        )
        for (
            rows,
            _shared,
            output_columns,
            offsets,
            sizes,
            parameter_groups,
            target_groups,
            _fixed_matrices,
            fixed_adjoints,
        ) in self._contraction_batches:
            pairs = np.empty(
                (len(offsets), rows, output_columns), dtype=result.dtype
            )
            for pair, offset, size in zip(pairs, offsets, sizes):
                pair[:] = vector[offset : offset + size].reshape(
                    rows, output_columns
                )
            gradients = (
                pairs @ fixed_adjoints
                if self.varied_left
                else fixed_adjoints @ pairs
            )
            for gradient, parameters, targets in zip(
                gradients, parameter_groups, target_groups
            ):
                np.add.at(result, parameters, gradient.reshape(-1)[targets])
        return result

    def restrict(self, support):
        support = np.asarray(support, dtype=np.int64)
        if np.array_equal(support, np.arange(self.parameter_dimension)):
            return self
        return _RestrictedMatrixFreePairEmbedding(self, support)

    def projected_diagonal(self, parent_diagonal):
        """Project a pair-space diagonal without forming the embedding."""
        gradients = {}
        for entry, pair_diagonal in zip(
            self.layout.entries, self.layout.blocks(parent_diagonal)
        ):
            q_l, q_p1, q_p2, q_r, q_mid = entry.key
            if self.varied_left:
                fixed = self.fixed_site.data.get((q_mid, q_p2, q_r))
                if fixed is None:
                    continue
                gradient = np.einsum(
                    "abcr,mcr->abm",
                    pair_diagonal,
                    np.abs(np.asarray(fixed)) ** 2,
                    optimize=True,
                )
                key = (q_l, q_p1, q_mid)
            else:
                fixed = self.fixed_site.data.get((q_l, q_p1, q_mid))
                if fixed is None:
                    continue
                gradient = np.einsum(
                    "abcr,abm->mcr",
                    pair_diagonal,
                    np.abs(np.asarray(fixed)) ** 2,
                    optimize=True,
                )
                key = (q_mid, q_p2, q_r)
            gradients[key] = gradients.get(key, 0.0) + gradient
        return np.real(self.routes.adjoint(gradients))


class _RestrictedMatrixFreePairEmbedding:
    def __init__(self, parent, support):
        self.parent = parent
        self.support = np.asarray(support, dtype=np.int64)
        self.pair_dimension = parent.pair_dimension
        self.parameter_dimension = int(self.support.size)

    @property
    def shape(self):
        return self.pair_dimension, self.parameter_dimension

    @property
    def nbytes(self):
        return int(self.parent.nbytes + self.support.nbytes)

    def matvec(self, vector):
        lifted = np.zeros(self.parent.parameter_dimension, dtype=complex)
        lifted[self.support] = vector
        return self.parent.matvec(lifted)

    def rmatvec(self, vector):
        return self.parent.rmatvec(vector)[self.support]

    def projected_diagonal(self, parent_diagonal):
        return self.parent.projected_diagonal(parent_diagonal)[self.support]


class _StructuredPairEmbedding:
    """Block-scatter tied coordinates without a dense pair-by-parameter map."""

    def __init__(self, pair_dimension, parameter_dimension, records=()):
        self.pair_dimension = int(pair_dimension)
        self.parameter_dimension = int(parameter_dimension)
        self._records = {
            (int(offset), int(size)): (
                np.asarray(columns, dtype=np.int64),
                np.asarray(design, dtype=complex),
            )
            for offset, size, columns, design in records
        }

    @property
    def shape(self):
        return self.pair_dimension, self.parameter_dimension

    @property
    def nbytes(self):
        return int(sum(
            columns.nbytes + design.nbytes
            for columns, design in self._records.values()
        ))

    def block_design(self, offset, size):
        record = self._records.get((int(offset), int(size)))
        if record is None:
            return np.zeros(0, dtype=np.int64), np.zeros(
                (int(size), 0), dtype=complex
            )
        return record

    def add_block(self, offset, size, columns, contribution):
        key = (int(offset), int(size))
        columns = np.asarray(columns, dtype=np.int64)
        contribution = np.asarray(contribution, dtype=complex)
        cached = self._records.get(key)
        if cached is None:
            self._records[key] = (columns, np.array(contribution, copy=True))
            return
        cached_columns, design = cached
        if not np.array_equal(cached_columns, columns):
            raise RuntimeError("Tied pair-embedding topology changed unexpectedly.")
        design += contribution

    def dense(self):
        result = np.zeros(self.shape, dtype=complex)
        for (offset, size), (columns, design) in self._records.items():
            result[offset : offset + size, columns] = design
        return result

    def __array__(self, dtype=None, copy=None):
        result = self.dense()
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        return np.array(result, copy=True) if copy else result

    def __getitem__(self, key):
        return self.dense()[key]

    def matvec(self, vector):
        vector = np.asarray(vector)
        if vector.size != self.parameter_dimension:
            raise ValueError("Tied coordinate vector has the wrong size.")
        result = np.zeros(
            self.pair_dimension, dtype=np.result_type(vector.dtype, complex)
        )
        for (offset, size), (columns, design) in self._records.items():
            result[offset : offset + size] = design @ vector[columns]
        return result

    def rmatvec(self, vector):
        vector = np.asarray(vector)
        if vector.size != self.pair_dimension:
            raise ValueError("Reduced pair vector has the wrong size.")
        result = np.zeros(
            self.parameter_dimension, dtype=np.result_type(vector.dtype, complex)
        )
        for (offset, size), (columns, design) in self._records.items():
            result[columns] += design.conj().T @ vector[offset : offset + size]
        return result

    def restrict(self, support):
        support = np.asarray(support, dtype=np.int64)
        inverse = np.full(self.parameter_dimension, -1, dtype=np.int64)
        inverse[support] = np.arange(support.size, dtype=np.int64)
        records = []
        for (offset, size), (columns, design) in self._records.items():
            keep = inverse[columns] >= 0
            if np.any(keep):
                records.append((offset, size, inverse[columns[keep]], design[:, keep]))
        return type(self)(self.pair_dimension, support.size, records)

    def compose(self, transform):
        transform = np.asarray(transform)
        if transform.shape[0] != self.parameter_dimension:
            raise ValueError("Tied-coordinate transform has the wrong row dimension.")
        records = []
        output_dimension = int(transform.shape[1])
        for (offset, size), (columns, design) in self._records.items():
            local = design @ transform[columns]
            active = np.flatnonzero(np.any(local != 0.0, axis=0))
            if active.size:
                records.append((offset, size, active, local[:, active]))
        return type(self)(self.pair_dimension, output_dimension, records)

    def projected_diagonal(self, parent_diagonal):
        parent_diagonal = np.asarray(parent_diagonal)
        diagonal = np.zeros(self.parameter_dimension, dtype=float)
        for (offset, size), (columns, design) in self._records.items():
            diagonal[columns] += np.real(np.sum(
                design.conj()
                * (parent_diagonal[offset : offset + size, None] * design),
                axis=0,
            ))
        return diagonal

    def projected_diagonal_blocks(self, parent_diagonal, groups):
        """Project a parent diagonal into selected coordinate blocks."""
        parent_diagonal = np.asarray(parent_diagonal)
        groups = tuple(np.asarray(group, dtype=np.int64) for group in groups)
        blocks = [
            np.zeros((group.size, group.size), dtype=complex)
            for group in groups
        ]
        for (offset, size), (columns, design) in self._records.items():
            weights = parent_diagonal[offset : offset + size]
            for block, group in zip(blocks, groups):
                common, record_positions, group_positions = np.intersect1d(
                    columns,
                    group,
                    assume_unique=True,
                    return_indices=True,
                )
                if common.size == 0:
                    continue
                local = design[:, record_positions]
                block[np.ix_(group_positions, group_positions)] += (
                    local.conj().T @ (weights[:, None] * local)
                )
        return tuple(0.5 * (block + block.conj().T) for block in blocks)


class NonAbelianFrontierLETTA:
    r"""Graph-tied reduced SU(2) frontier LETTA.

    A graph tie conditions an earlier tensor on either a future invariant
    physical-sector label or a future reduced fusion sector. Magnetic
    projections are supplied only by the structural Clebsch--Gordan coupling
    in the non-Abelian MPS contraction and are never copied by a tie.

    ``D`` counts reduced multiplets per reachable virtual sector, not the
    magnetic-state dimension.

    ``n_threads=None`` uses ``pyqed.num_workers()`` for the compiled OpenMP
    team. Independent Python setup defaults to ``workers=1`` to keep one
    parallel layer active. Explicit counts override these defaults.

    ``virtual_boundary='periodic'`` closes a D-dimensional multiplicity index
    with a spin-zero seam and a fixed total-particle-charge twist. An exact
    spectator-index lift retains the raw closed-loop parameter constraints
    while using reduced open environments (interior multiplicities grow by D).
    This adapts the periodic trace ansatz of F. Verstraete, D. Porras and
    J. I. Cirac, Phys. Rev. Lett. 93, 227205 (2004),
    https://doi.org/10.1103/PhysRevLett.93.227205. It is not a reproduction of
    their DMRG algorithm: only singlet targets, spin-zero seam sectors, reduced
    MPO Hamiltonians and fixed-D ``algorithm='projected'`` one-site updates
    are supported. General spin-carrying seams and periodic canonical gauges
    are unsupported; no global convergence guarantee is provided.

    ``compact_frontier=True`` removes closed-loop contraction coordinates
    absent from every complete structural parameter path. This exact layout
    reduction uses no numerical cutoff or Schmidt truncation and preserves
    zero-valued parameter directions. ``False`` retains the unfolded reference
    layout. Route compilation uses shape-only templates and selected seam
    indices, avoiding dense intermediate site tensors.

    The reduced-tensor layer follows the Wigner--Eckart/non-Abelian tensor
    network formulation of A. Weichselbaum, Phys. Rev. B 86, 245124 (2012),
    https://doi.org/10.1103/PhysRevB.86.245124.  This is a PyQED adaptation to
    the project-specific LETTA graph-tied ansatz, not a reproduction of that
    paper's MPS/NRG algorithms.  The implementation currently supports fully
    reduced spatial-orbital SU(2) states and scalar rank-coupled Hamiltonians;
    arbitrary tensor ranks are not provided. ``from_integrals`` additionally
    supports finite spatial groups through SpatialSymmetryLETTA's structural
    regular-representation leg; spatial rows are explicit, not CG-compressed.
    """

    def __init__(
        self,
        mpo,
        *,
        target_sector=None,
        nelec=None,
        spin=0,
        graph=None,
        D=1,
        ecore=0.0,
        seed=None,
        init="mps",
        base_sites=None,
        tie="auto",
        max_frontier_states=4096,
        workers=1,
        n_threads=None,
        we_route_memory=256.0,
        virtual_boundary="open",
        compact_frontier=True,
        _evaluate_initial_state=True,
    ):
        if virtual_boundary not in {'open','periodic'}:
            raise ValueError("virtual_boundary must be 'open' or 'periodic'")
        self.virtual_boundary = virtual_boundary
        self.compact_frontier = bool(compact_frontier)
        self._frontier_compaction = None
        state_only = mpo is None
        self._state_only = state_only
        if state_only:
            if base_sites is None:
                raise ValueError("State-only LETTA construction requires base_sites.")
            base_sites = tuple(base_sites)
            mpo = _identity_mpo_factors_for_sites_and_mpo(
                base_sites, [None] * len(base_sites),
            )
        factors = getattr(mpo, "factors", mpo)
        self._su2_moving_environment = getattr(mpo, "moving_environment", None)
        self._complementary_operators = getattr(
            mpo, "complementary_operators", None
        )
        self.mpo = prepare_hamiltonian_factors(factors)
        self.hamiltonian_representation = (
            "complementary"
            if self.mpo
            and getattr(self.mpo[0], "normal_complementary_owner", None) is not None
            and not tuple(getattr(self.mpo[0], "reduced_terms", ()))
            else "mpo"
        )
        self._hamiltonian_spec = None
        self._pair_hamiltonian_cache = None
        self._hamiltonian_representation_policy = self.hamiltonian_representation
        self._representation_decision = {
            "policy": self.hamiltonian_representation,
            "selected": self.hamiltonian_representation,
            "reason": "explicit construction request",
        }
        self.nsites = len(self.mpo)
        if self.nsites < 2:
            raise ValueError("NonAbelianFrontierLETTA requires at least two sites.")
        if not isinstance(D, Integral) or isinstance(D, (bool, np.bool_)) or int(D) < 1:
            raise ValueError("D must be a positive reduced multiplet dimension.")
        self.D = int(D)
        self.workers = resolve_workers(1 if workers is None else workers)
        if n_threads is None:
            n_threads = resolve_default_workers()
        if isinstance(n_threads, (bool, np.bool_)) or not isinstance(
            n_threads, Integral
        ):
            raise TypeError("n_threads must be a positive integer.")
        if int(n_threads) < 1:
            raise ValueError("n_threads must be positive.")
        self.n_threads = int(n_threads)
        if self._su2_moving_environment is not None and hasattr(
            self._su2_moving_environment, "set_num_threads"
        ):
            self._su2_moving_environment.set_num_threads(self.n_threads)
        self.we_route_memory = float(we_route_memory)
        if self.we_route_memory <= 0.0:
            raise ValueError("we_route_memory must be positive in MiB.")
        self._we_route_limit_bytes = int(self.we_route_memory * 1024**2)
        self._we_route_cache = {}
        self._we_route_cache_hits = 0
        self._we_route_cache_misses = 0
        self._projected_route_cache = {}
        self._projected_route_cache_hits = 0
        self._projected_route_cache_misses = 0
        self._embedding_basis_cache = {}
        self._embedding_basis_cache_hits = 0
        self._embedding_basis_cache_misses = 0
        self._pair_coordinate_cache = {}
        self._pair_coordinate_cache_hits = 0
        self._pair_coordinate_cache_misses = 0
        self._embedding_plan_cache = {}
        self._embedding_numeric_cache = {}
        self._embedding_numeric_cache_hits = 0
        self._embedding_incremental_updates = 0
        self._embedding_reused_blocks = 0
        self._embedding_updated_blocks = 0
        self._embedding_active_columns_cache = {}
        self._projected_krylov_cache = {}
        self._metric_whitener_cache = {}
        self._metric_whitener_component_cache = {}
        self._metric_whitener_cache_hits = 0
        self._metric_whitener_cache_misses = 0
        self._metric_whitener_component_hits = 0
        self._metric_component_topology_cache = {}
        self._metric_component_cache_hits = 0
        self._metric_component_cache_misses = 0
        self._state_revision = 0
        self._site_revisions = [0] * self.nsites
        self._global_guard_bonds = set()
        self._stationary_certificate_cache = {}
        self._stationary_certificate_cache_hits = 0
        self._materialized_site_cache = {}
        self._materialized_site_cache_hits = 0
        self._materialized_site_cache_misses = 0
        self._incremental_materialization_enabled = False
        self._solver_executor = (
            ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix="su2-letta-local",
            )
            if self.workers > 1
            else None
        )
        self.nelec = None if nelec is None else int(nelec)
        self.spin = None if spin is None else int(spin)
        if target_sector is None:
            if self.nelec is not None:
                target_sector = spatial_target_sector(self.nelec, self.spin or 0)
            elif base_sites is not None:
                boundary = tuple(_unique(tuple(base_sites)[-1].qns[2]))
                if len(boundary) == 1:
                    target_sector = boundary[0]
        if target_sector is None:
            raise ValueError(
                "target_sector is required unless it can be inferred from nelec/spin "
                "or the right boundary of base_sites."
            )
        self.target_sector = target_sector
        if virtual_boundary=='periodic' and int(target_sector.irrep.two_j)!=0:
            raise ValueError("periodic SU2 multiplicity closure requires a singlet target")
        self.core_energy = float(getattr(mpo, "ecore", ecore))
        mpo_info = getattr(mpo, "info", {})
        self.mpo_includes_core_energy = bool(
            isinstance(mpo_info, dict)
            and mpo_info.get("includes_core_energy", False)
        )
        self.ecore = (
            0.0 if self.mpo_includes_core_energy else self.core_energy
        )
        self.graph = _canonical_graph(self.nsites, graph)
        parents = [set() for _ in range(self.nsites)]
        for left, right in self.graph:
            parents[left].add(right)
        self.parent_sets = tuple(tuple(sorted(values)) for values in parents)
        self.frontiers = _frontiers(self.nsites, self.parent_sets)

        if base_sites is None:
            if self.nelec is None:
                raise ValueError(
                    "generic non-Abelian frontier construction requires base_sites."
                )
            base_sites = build_random_reduced_spatial_mps(
                self.nsites,
                target_sector=self.target_sector,
                bond_multiplicity=self.D,
                seed=seed,
            )
            if virtual_boundary=='periodic':
                rng = np.random.default_rng(seed)
                for index, tensor in enumerate(base_sites):
                    qns = [list(q) for q in tensor.qns]
                    # A cut loop has D incoming channels: open-chain path
                    # counts must not cap multiplicities near either end.
                    for axis in (0, 2):
                        qns[axis] = [q for q in _unique(qns[axis]) for _ in range(self.D)]
                    data = {key:rng.normal(size=(self.D, block.shape[1], self.D))
                            for key, block in tensor.data.items()}
                    base_sites[index] = IrrepTensor(data=data, qns=qns,
                                                    dirs=tensor.dirs, metadata=tensor.metadata)
        self._base_sites = self._validate_base_sites(base_sites)
        self.physical_sectors_by_site = tuple(
            tuple(_unique(site.qns[1])) for site in self._base_sites
        )
        self._physical_index_by_site = tuple(
            {sector: index for index, sector in enumerate(sectors)}
            for sectors in self.physical_sectors_by_site
        )
        tie = str(tie).lower().replace("-", "_")
        if tie == "auto":
            tie = (
                "physical"
                if any(len(sectors) > 1 for sectors in self.physical_sectors_by_site)
                else "fusion"
            )
        if tie not in {"physical", "fusion"}:
            raise ValueError("tie must be 'auto', 'physical', or 'fusion'.")
        self.tie = tie
        self.tie_domains = tuple(
            (
                self.physical_sectors_by_site[site]
                if tie == "physical"
                else tuple(_unique(base.qns[0]))
            )
            for site, base in enumerate(self._base_sites)
        )
        self._tie_index_by_site = tuple(
            {sector: index for index, sector in enumerate(domain)}
            for domain in self.tie_domains
        )
        self._assignments = tuple(
            tuple(
                product(
                    *(range(len(self.tie_domains[future])) for future in frontier)
                )
            )
            for frontier in self.frontiers
        )
        self.max_frontier_states = int(max_frontier_states)
        self.tie_history = []
        largest = max(len(values) for values in self._assignments)
        if largest > int(max_frontier_states):
            raise MemoryError(
                "non-Abelian invariant frontier has "
                f"{largest} assignments; increase max_frontier_states or improve ordering/graph."
            )
        self.tensors, self._tensor_keys = self._initialize_tensors(
            init=init,
            seed=seed,
        )
        self.history = []
        self.energy = (self.expectation()
                       if not state_only and _evaluate_initial_state else None)
        self.converged = False
        self.success = None
        self.message = "initialized reduced non-Abelian FrontierLETTA"
        self.is_native_su2 = True
        self.local_environment_backend = "wigner_eckart_reduced"
        self.has_fully_reduced_local_operator = True

    @classmethod
    def from_integrals(
        cls,
        h1e,
        eri=None,
        *,
        nelec,
        spin=0,
        graph=None,
        D=1,
        ecore=0.0,
        cutoff=1.0e-10,
        hamiltonian_representation="auto",
        point_group=None,
        orbital_representations=None,
        wfnsym=None,
        **kwargs,
    ):
        """Build a reduced qchem Hamiltonian and its SU(2)-LETTA state.

        ``orbital_representations`` together with ``wfnsym`` selects a complete spatial multiplet and returns a
        SpatialSymmetryLETTA driver. Its finite-group structural leg is kept
        in every reduced local objective, including both E rows. See that
        class for Singh/Malmqvist/Knecht references, fidelity, bond-dimension
        conventions, and the fixed-D one-site sweep restrictions.

        The default LETTA tie graph is the nearest-neighbor orbital chain;
        ``graph="nn"`` selects it explicitly. Hamiltonian couplings remain
        complete in either representation; ``graph`` controls only variational
        tensor ties. ``"nnn"`` and ``"nnnn"`` include all ties through the
        second and third neighbors, respectively, using the general frontier
        optimizer rather than the NN conditional canonical gauge.
        The default direct complementary representation carries
        the reduced $S/R/A/P/B/Q$ operators without storing a Hamiltonian MPO;
        ``"mpo"`` remains an explicit reference option.

        With ``virtual_boundary='periodic'``, auto selects the reduced MPO.
        The spin-zero multiplicity seam and its restrictions are described
        in :class:`NonAbelianFrontierLETTA`; optimize with ``algorithm='projected'``.
        """
        if "target_irrep" in kwargs:
            raise TypeError("Use wfnsym; target_irrep has been removed.")
        if any(value is not None for value in (point_group, orbital_representations, wfnsym)):
            from pyqed.qchem.symmetry import get_point_group
            from pyqed.qchem.spatial_symmetry import resolve_spatial_target
            from .spatial_symmetry import SpatialSymmetryLETTA
            group = get_point_group(point_group or "D3h") if point_group is None or isinstance(point_group, str) else point_group
            target = resolve_spatial_target(group, wfnsym)
            if orbital_representations is None or target is None:
                raise ValueError("Spatial symmetry requires orbital_representations and wfnsym.")
            if hamiltonian_representation not in {"auto", "mpo"}:
                raise ValueError("Spatial structural legs currently use reduced MPO Hamiltonians.")
            from pyqed.qchem.spatial_symmetry import SpatialGroupAction
            SpatialGroupAction(orbital_representations, target, point_group=group).validate_hamiltonian(h1e, eri)
            options = dict(kwargs)
            seed = cls.from_integrals(h1e, eri, nelec=nelec, spin=spin, graph=graph,
                D=D, ecore=ecore, cutoff=cutoff, hamiltonian_representation="mpo", **options)
            solver = SpatialSymmetryLETTA(seed, orbital_representations, target,
                point_group=group)
            def seed_factory(root, previous):
                root_options = dict(options)
                root_options["seed"] = int(options.get("seed") or 0) + root
                next_seed = cls.from_integrals(h1e, eri, nelec=nelec, spin=spin, graph=graph,
                    D=D, ecore=ecore, cutoff=cutoff, hamiltonian_representation="mpo", **root_options)
                return SpatialSymmetryLETTA(next_seed, orbital_representations, target,
                    point_group=group, orthogonal_to=previous)
            solver._seed_factory = seed_factory
            return solver

        from pyqed.qchem.dmrg.backends.reduced import (
            build_spatial_reduced_hamiltonian_mpo,
        )

        eri_for_builder = eri
        if eri is not None and np.asarray(eri).ndim == 4:
            eri_for_builder = np.asarray(eri)[None, None, ...]
        hamiltonian = build_spatial_reduced_hamiltonian_mpo(
            h1e,
            eri=eri_for_builder,
            cutoff=cutoff,
            fully_reduced=True,
            n_elec=nelec,
            spin=spin,
            ecore=ecore,
        )
        representation_policy = str(hamiltonian_representation).lower().replace(
            "-", "_"
        )
        representation = representation_policy
        if representation in {"direct", "normal_complementary", "nc"}:
            representation = "complementary"
        if representation == "auto":
            representation = "complementary"
        if representation not in {"complementary", "mpo"}:
            raise ValueError(
                "hamiltonian_representation must be 'auto', 'complementary', "
                "or 'mpo'."
            )
        if kwargs.get('virtual_boundary','open')=='periodic':
            if representation_policy in {'complementary','direct','normal_complementary','nc'}:
                raise ValueError("closed SU2 multiplicity loops currently require a reduced MPO Hamiltonian")
            representation = 'mpo'
        if representation == "complementary" and np.asarray(h1e).shape[-1] <= 2:
            representation = "mpo"
        if representation == "mpo":
            factors = prepare_hamiltonian_factors(
                hamiltonian.factors,
                materialize=True,
            )
            hamiltonian = replace(
                hamiltonian,
                factors=list(factors),
            )
        if graph is None:
            graph = _nearest_neighbor_graph(np.asarray(h1e).shape[-1])
        state = cls(
            hamiltonian,
            nelec=nelec,
            spin=spin,
            graph=graph,
            D=D,
            ecore=ecore,
            **kwargs,
        )
        state.hamiltonian = replace(
            hamiltonian,
            factors=list(state.mpo),
            complementary_operators=(
                state._complementary_operators
                if representation == "complementary"
                else None
            ),
            moving_environment=(
                state._su2_moving_environment
                if representation == "complementary"
                else None
            ),
        )
        state._hamiltonian_spec = {
            "h1e": np.array(h1e, copy=True),
            "eri": None if eri is None else np.array(eri, copy=True),
            "nelec": int(nelec),
            "spin": int(spin),
            "ecore": float(ecore),
            "cutoff": float(cutoff),
            "hamiltonian_representation": representation,
        }
        state._hamiltonian_representation_policy = representation_policy
        state._representation_decision = {
            "policy": representation_policy,
            "selected": representation,
            "reason": "direct complementary has the lower cold-start cost",
        }
        return state

    @classmethod
    def from_mps(
        cls,
        sites,
        mpo,
        *,
        target_sector=None,
        nelec=None,
        spin=0,
        graph=None,
        tie="auto",
        **kwargs,
    ):
        """Embed a fully reduced SU(2) MPS as a neutral-control frontier state."""
        sites = tuple(sites)
        multiplicities = [
            max(
                (
                    site.legs[2].sector_dim(sector)
                    for sector in site.legs[2].sectors
                ),
                default=1,
            )
            for site in sites[:-1]
        ]
        return cls(
            mpo,
            target_sector=target_sector,
            nelec=nelec,
            spin=spin,
            graph=graph,
            tie=tie,
            D=max(multiplicities, default=1),
            base_sites=sites,
            init="mps",
            **kwargs,
        )

    def _validate_base_sites(self, sites):
        sites = tuple(sites)
        if len(sites) != self.nsites:
            raise ValueError("base SU(2) MPS and MPO lengths differ.")
        for site_index, site in enumerate(sites):
            if not isinstance(site, IrrepTensor) or site.rank != 3:
                raise TypeError("base_sites must contain rank-3 IrrepTensor objects.")
            if (site.metadata or {}).get("physical_basis") != "fully_reduced_su2":
                raise ValueError(
                    "NonAbelianFrontierLETTA requires fully reduced SU(2) site tensors."
                )
            physical_sectors = tuple(_unique(site.qns[1]))
            core = self.mpo[site_index]
            if set(physical_sectors) != set(core.phys_in_leg.sectors):
                raise ValueError(
                    f"base site {site_index} physical sectors do not match its MPO core."
                )
            for sector in physical_sectors:
                multiplicity = site.legs[1].sector_dim(sector)
                if multiplicity != core.phys_in_leg.sector_dim(sector):
                    raise ValueError(
                        f"base site {site_index} physical multiplicity for {sector!r} "
                        "does not match its MPO core."
                    )
        if tuple(_unique(sites[-1].qns[2])) != (self.target_sector,):
            raise ValueError(
                "base MPS right boundary does not match the requested target_sector."
            )
        for left, right in zip(sites, sites[1:]):
            if (
                left.legs[2].sectors != right.legs[0].sectors
                or dict(left.legs[2].dims) != dict(right.legs[0].dims)
            ):
                raise ValueError("neighboring base MPS bond-sector layouts do not match.")
        if self.virtual_boundary=='periodic':
            for endpoint,axis in [(sites[0],0),(sites[-1],2)]:
                if len(endpoint.legs[axis].sectors)!=1 or next(iter(endpoint.legs[axis].dims.values()))!=self.D:
                    raise ValueError("periodic seam endpoints must each have D multiplicity copies")
            left_sector = sites[0].legs[0].sectors[0]
            if left_sector.charge != 0 or left_sector.irrep.two_j != 0:
                raise ValueError("periodic spin-zero charge-twisted seam starts in the vacuum sector")
        return tuple(site.copy() for site in sites)

    def with_tie_graph(self, graph, *, max_frontier_states=None, evaluate_energy=True):
        """Return an independent reduced state with additional invariant ties.

        Existing conditioned blocks are copied to every new label slice;
        magnetic projections and virtual-sector layouts are unchanged.
        Removing ties is unsupported because it requires state compression.
        Set ``evaluate_energy=False`` to prepare tensors without a frontier
        contraction; the returned energy is then unset.
        """
        from .su2_adaptive_graph import with_tie_graph
        return with_tie_graph(self, graph, max_frontier_states=max_frontier_states,
                             evaluate_energy=evaluate_energy)

    def adapt_ties(self, **options):
        """Try cost-bounded additions and adopt a variationally improved graph.

        Invariant-sector information shortlists candidates; relaxed energy
        gains decide acceptance. This is a greedy adaptation inspired by
        Kovyrshin and Reiher, JCP 147, 214111 (2017),
        https://doi.org/10.1063/1.5004693, not their SATNS algorithm. Sector
        information omits intra-sector quantum coherences; there is no global
        graph-optimality or convergence guarantee. See su2_adaptive_graph.
        """
        from .su2_adaptive_graph import adapt_ties
        return adapt_ties(self, **options)

    def _initialize_tensors(self, *, init, seed):
        init = str(init).lower().replace("-", "_")
        if init not in {"mps", "random"}:
            raise ValueError("SU2LETTA init must be 'mps' or 'random'.")
        rng = np.random.default_rng(seed)
        tensors = []
        key_lists = []
        for site, base in enumerate(self._base_sites):
            local = {}
            conditions = tuple(
                product(
                    *(
                        range(len(self.tie_domains[parent]))
                        for parent in self.parent_sets[site]
                    )
                )
            )
            for block_key, block in base.data.items():
                for condition in conditions:
                    key = tuple(block_key) + (tuple(condition),)
                    if init == "mps":
                        value = np.array(block, copy=True)
                    else:
                        value = rng.normal(scale=1.0, size=np.asarray(block).shape)
                    local[key] = value
            tensors.append(local)
            key_lists.append(tuple(local))
        return tensors, tuple(key_lists)

    @property
    def nparameters(self):
        return int(sum(block.size for tensor in self.tensors for block in tensor.values()))

    @property
    def max_frontier_width(self):
        return max(len(frontier) for frontier in self.frontiers)

    @property
    def supports_conditional_canonical_gauge(self):
        """Whether every internal frontier is one nearest-neighbor condition."""
        return bool(
            self.virtual_boundary == 'open' and self.tie == "physical"
            and all(
                self.parent_sets[bond] == (bond + 1,)
                and self.frontiers[bond + 1] == (bond + 1,)
                for bond in range(self.nsites - 1)
            )
        )

    @staticmethod
    def _conditional_right_weight(q_left, q_right, physical_dim):
        if int(physical_dim) != 1:
            return 1.0
        left_dim = max(int(getattr(q_left.irrep, "dim", 1)), 1)
        right_dim = max(int(getattr(q_right.irrep, "dim", 1)), 1)
        return float(right_dim / left_dim)

    def canonicalize_conditional_bond(
        self,
        bond,
        *,
        direction="lr",
        rcond=1.0e-12,
    ):
        """Move one exact reduced gauge conditioned on the crossing sector.

        This is the SU(2) analogue of the physical-conditioned LETTA gauge.
        It is representable without enlarging the ansatz when the frontier at
        the bond contains only the next physical-sector label, as in an NN
        chain tie graph. Rank-deficient blocks are left unchanged.
        """
        bond = int(bond)
        if bond < 0 or bond >= self.nsites - 1:
            raise IndexError(f"bond {bond} is outside a chain of length {self.nsites}.")
        direction = str(direction).lower().replace("-", "_")
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        if not self.supports_conditional_canonical_gauge:
            raise ValueError(
                "conditional canonical gauge requires a physical nearest-neighbor "
                "tie frontier at every internal bond."
            )
        rcond = float(rcond)
        if not np.isfinite(rcond) or rcond < 0.0:
            raise ValueError("rcond must be finite and nonnegative.")

        left_tensor = self.tensors[bond]
        right_tensor = self.tensors[bond + 1]
        updates = []
        for condition_index, shared_sector in enumerate(
            self.tie_domains[bond + 1]
        ):
            middle_sectors = sorted(
                {
                    key[2]
                    for key in left_tensor
                    if key[3] == (condition_index,)
                }.intersection(
                    {
                        key[0]
                        for key in right_tensor
                        if key[1] == shared_sector
                    }
                )
            )
            for middle_sector in middle_sectors:
                left_keys = [
                    key
                    for key in self._tensor_keys[bond]
                    if key[2] == middle_sector
                    and key[3] == (condition_index,)
                ]
                right_keys = [
                    key
                    for key in self._tensor_keys[bond + 1]
                    if key[0] == middle_sector and key[1] == shared_sector
                ]
                if not left_keys or not right_keys:
                    continue
                dimension = int(np.asarray(left_tensor[left_keys[0]]).shape[2])
                left_matrix = np.concatenate(
                    [
                        np.asarray(left_tensor[key]).reshape(-1, dimension)
                        for key in left_keys
                    ],
                    axis=0,
                )
                right_records = []
                right_parts = []
                for key in right_keys:
                    block = np.asarray(right_tensor[key])
                    weight = self._conditional_right_weight(
                        middle_sector,
                        key[2],
                        block.shape[1],
                    )
                    scale = np.sqrt(weight)
                    matrix = block.reshape(dimension, -1)
                    right_records.append((key, matrix.shape[1], scale))
                    right_parts.append(scale * matrix)
                right_matrix = np.concatenate(right_parts, axis=1)

                source = left_matrix if direction == "lr" else right_matrix.T
                if min(source.shape) < dimension:
                    updates.append(
                        {
                            "bond": bond,
                            "condition": condition_index,
                            "sector": middle_sector,
                            "direction": direction,
                            "applied": False,
                            "message": "insufficient conditional support",
                        }
                    )
                    continue
                singular = np.linalg.svd(source, compute_uv=False)
                scale = max(float(np.max(singular, initial=0.0)), 1.0)
                rank = int(np.count_nonzero(singular > rcond * scale))
                if rank < dimension:
                    updates.append(
                        {
                            "bond": bond,
                            "condition": condition_index,
                            "sector": middle_sector,
                            "direction": direction,
                            "applied": False,
                            "message": "rank-deficient conditional support",
                        }
                    )
                    continue

                orthogonal, transfer = np.linalg.qr(source, mode="reduced")
                if direction == "lr":
                    offset = 0
                    for key in left_keys:
                        block = np.asarray(left_tensor[key])
                        rows = block.shape[0] * block.shape[1]
                        left_tensor[key] = orthogonal[offset : offset + rows].reshape(
                            block.shape
                        )
                        offset += rows
                    for key in right_keys:
                        block = np.asarray(right_tensor[key])
                        right_tensor[key] = np.einsum(
                            "ab,bpr->apr", transfer, block, optimize=True
                        )
                    error = float(
                        np.linalg.norm(
                            orthogonal.conj().T @ orthogonal
                            - np.eye(dimension, dtype=orthogonal.dtype)
                        )
                    )
                else:
                    left_transfer = transfer.T
                    for key in left_keys:
                        block = np.asarray(left_tensor[key])
                        left_tensor[key] = np.einsum(
                            "lpa,ab->lpb", block, left_transfer, optimize=True
                        )
                    canonical_right = orthogonal.T
                    offset = 0
                    for key, width, right_scale in right_records:
                        block = np.asarray(right_tensor[key])
                        right_tensor[key] = (
                            canonical_right[:, offset : offset + width] / right_scale
                        ).reshape(block.shape)
                        offset += width
                    error = float(
                        np.linalg.norm(
                            canonical_right @ canonical_right.conj().T
                            - np.eye(dimension, dtype=canonical_right.dtype)
                        )
                    )
                updates.append(
                    {
                        "bond": bond,
                        "condition": condition_index,
                        "sector": middle_sector,
                        "direction": direction,
                        "applied": True,
                        "rank": rank,
                        "dimension": dimension,
                        "canonical_error": error,
                        "message": "canonicalized",
                    }
                )
        if any(bool(update.get("applied", False)) for update in updates):
            self._state_revision += 1
            self._site_revisions[bond] += 1
            self._site_revisions[bond + 1] += 1
            self._invalidate_materialized_sites(bond, bond + 1)
        return tuple(updates)

    def balance_coefficients(self, *, conditional=True, rcond=1.0e-12):
        """Balance internal reduced virtual bonds with sector-wise QR transfers.

        Adaptation of QR gauge absorption in Schollwoeck, Ann. Phys. 326,
        96--192 (2011), https://doi.org/10.1016/j.aop.2010.09.012. A transfer
        may depend on the next physical label when both neighboring tensors
        share it; other tie conditions are stacked. This preserves tensor
        shapes and the represented state in exact arithmetic. It is coefficient
        balancing, not a periodic canonical norm gauge or Schmidt truncation.
        The closing bond is not transformed. Rank-deficient or relatively
        ill-conditioned transfers are skipped at ``rcond`` without truncation.

        No Hamiltonian or norm environments are built. The caller must audit
        numerical preservation for ill-conditioned states before optimization.
        Returns per-transfer diagnostics and invalidates changed site caches.
        """
        if not np.isfinite(rcond) or rcond < 0 or rcond >= 1:
            raise ValueError('rcond must be finite and in [0, 1)')
        updates = []
        for bond in range(self.nsites-1):
            left, right = dict(self.tensors[bond]), dict(self.tensors[bond+1])
            conditioned = conditional and self.tie == 'physical' and bond+1 in self.parent_sets[bond]
            slot = self.parent_sets[bond].index(bond+1) if conditioned else None
            groups = enumerate(self.tie_domains[bond+1]) if conditioned else [(None, None)]
            changed = False
            for index, physical in groups:
                for sector in self._base_sites[bond].legs[2].sectors:
                    keys = [key for key in self._tensor_keys[bond] if key[2] == sector
                            and (not conditioned or key[3][slot] == index)]
                    targets = [key for key in self._tensor_keys[bond+1] if key[0] == sector
                               and (not conditioned or key[1] == physical)]
                    if not keys or not targets:
                        continue
                    rows = [left[key].reshape(-1, left[key].shape[-1]) for key in keys]
                    matrix = np.concatenate(rows)
                    singular = np.linalg.svd(matrix, compute_uv=False)
                    applied = (matrix.shape[0] >= matrix.shape[1] and singular[0] > 0
                               and singular[-1] > rcond*singular[0])
                    updates.append(dict(bond=bond, condition=index, sector=sector,
                                        applied=bool(applied),
                                        relative_minimum_singular_value=float(singular[-1]/singular[0])
                                        if singular[0] else 0.0))
                    if not applied:
                        continue
                    q, transfer = np.linalg.qr(matrix, mode='reduced')
                    offset = 0
                    for key, row in zip(keys, rows):
                        left[key] = q[offset:offset+row.shape[0]].reshape(left[key].shape)
                        offset += row.shape[0]
                    for key in targets:
                        right[key] = np.einsum('ab,bpc->apc', transfer, right[key])
                    changed = True
            if changed:
                for site, blocks in ((bond, left), (bond+1, right)):
                    vector = np.concatenate([blocks[key].ravel() for key in self._tensor_keys[site]])
                    self._set_site_vector(site, vector)
        return tuple(updates)

    def canonicalize_conditional_center(self, center, *, rcond=1.0e-12):
        """Build a conditioned mixed-canonical gauge around one site."""
        center = int(center)
        if center < 0 or center >= self.nsites:
            raise IndexError(f"center {center} is outside a chain of length {self.nsites}.")
        updates = []
        for bond in range(center):
            updates.extend(
                self.canonicalize_conditional_bond(
                    bond, direction="lr", rcond=rcond
                )
            )
        for bond in reversed(range(center, self.nsites - 1)):
            updates.extend(
                self.canonicalize_conditional_bond(
                    bond, direction="rl", rcond=rcond
                )
            )
        return tuple(updates)

    @property
    def frontier_states(self):
        return tuple(len(values) for values in self._assignments)

    def _bond_sector_dims(self, site, axis):
        base = self._base_sites[site]
        return {
            sector: base.legs[axis].sector_dim(sector)
            for sector in base.legs[axis].sectors
        }

    def materialize_site(self, site):
        """Materialize one tied tensor as a reduced SU(2) MPS site."""
        site = int(site)
        cached = self._materialized_site_cache.get(site)
        if cached is not None:
            self._materialized_site_cache_hits += 1
            return cached
        self._materialized_site_cache_misses += 1
        if self.virtual_boundary == 'periodic' and self.compact_frontier:
            materialized = self._wigner_eckart_route_plan(site).tensor(self._pack_site(site))
        else:
            materialized = self._materialize_unfolded_site(site)
        self._materialized_site_cache[site] = materialized
        return materialized

    def _materialize_unfolded_site(self, site):
        """Explicit unfolded reference for validating the structural layout."""
        base = self._base_sites[site]
        left_frontier = self.frontiers[site]
        right_frontier = self.frontiers[site + 1]
        left_assignments = self._assignments[site]
        right_assignments = self._assignments[site + 1]
        left_maps = [dict(zip(left_frontier, values)) for values in left_assignments]
        right_maps = [dict(zip(right_frontier, values)) for values in right_assignments]
        left_dims = self._bond_sector_dims(site, 0)
        right_dims = self._bond_sector_dims(site, 2)
        physical_sectors = self.physical_sectors_by_site[site]

        data = {}
        for q_left, q_phys, q_right in base.data:
            d_left = left_dims[q_left]
            d_phys = np.asarray(base.data[(q_left, q_phys, q_right)]).shape[1]
            d_right = right_dims[q_right]
            block = np.zeros(
                (
                    d_left * len(left_assignments),
                    d_phys,
                    d_right * len(right_assignments),
                ),
                dtype=np.result_type(
                    *[
                        value.dtype
                        for key, value in self.tensors[site].items()
                        if key[:3] == (q_left, q_phys, q_right)
                    ]
                ),
            )
            endpoint_value = self._tie_index_by_site[site][
                q_phys if self.tie == "physical" else q_left
            ]
            for left_index, left_values in enumerate(left_maps):
                if site in left_values and left_values[site] != endpoint_value:
                    continue
                for right_index, right_values in enumerate(right_maps):
                    common = set(left_values).intersection(right_values)
                    if any(left_values[value] != right_values[value] for value in common):
                        continue
                    condition = tuple(
                        right_values[parent] for parent in self.parent_sets[site]
                    )
                    source = self.tensors[site][
                        (q_left, q_phys, q_right, condition)
                    ]
                    left_slice = slice(left_index * d_left, (left_index + 1) * d_left)
                    right_slice = slice(right_index * d_right, (right_index + 1) * d_right)
                    block[left_slice, :, right_slice] = source
            # Retain structural zero blocks. Their fixed layout provides one
            # exact linear embedding for all tied parameter directions,
            # including directions that vanish in the current iterate.
            data[(q_left, q_phys, q_right)] = block

        left_qns = [
            sector
            for sector, dimension in left_dims.items()
            for _ in range(dimension * len(left_assignments))
        ]
        right_qns = [
            sector
            for sector, dimension in right_dims.items()
            for _ in range(dimension * len(right_assignments))
        ]
        materialized = IrrepTensor(
            data=data,
            qns=[left_qns, list(physical_sectors), right_qns],
            dirs=[-1, 1, 1],
            metadata={
                "physical_basis": "fully_reduced_su2",
                "letta_frontier": right_frontier,
                "letta_invariant_ties": True,
                "letta_tie": self.tie,
            },
        )
        if self.virtual_boundary=='periodic':
            from .su2_closure import lift_closure
            materialized = lift_closure(materialized,site,self.nsites,self.D)
        return materialized

    def materialize(self):
        """Return the exact unfolded reduced SU(2) MPS."""
        return [self.materialize_site(site) for site in range(self.nsites)]

    @property
    def state(self):
        from pyqed.mps.nonabelian.mps import MPS

        return MPS.from_tensors(self.materialize(), target_sector=self.target_sector)

    def close(self):
        """Release the bounded local-contraction worker pool."""
        self._pair_hamiltonian_cache = None
        executor = getattr(self, "_solver_executor", None)
        self._solver_executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def __deepcopy__(self, memo):
        """Copy variational state while dropping transient compiled owners."""
        if self._hamiltonian_spec is not None:
            options = copy.deepcopy(self._hamiltonian_spec, memo)
            clone = type(self).from_integrals(
                **options,
                graph=self.graph,
                D=self.D,
                virtual_boundary=self.virtual_boundary,
                compact_frontier=self.compact_frontier,
                base_sites=copy.deepcopy(self._base_sites, memo),
                tie=self.tie,
                init="mps",
                workers=self.workers,
                n_threads=self.n_threads,
                we_route_memory=self.we_route_memory,
                max_frontier_states=self.max_frontier_states,
            )
            memo[id(self)] = clone
            runtime_names = {
                "mpo",
                "hamiltonian",
                "_hamiltonian_spec",
                "_su2_moving_environment",
                "_complementary_operators",
                "_solver_executor",
                "_pair_hamiltonian_cache",
            }
            for name, value in self.__dict__.items():
                if name not in runtime_names:
                    setattr(clone, name, copy.deepcopy(value, memo))
            clone._hamiltonian_spec = options
            return clone
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            if name in {"_su2_moving_environment", "_complementary_operators", "_pair_hamiltonian_cache"}:
                setattr(clone, name, None)
            elif name != "_solver_executor":
                setattr(clone, name, copy.deepcopy(value, memo))
        clone._solver_executor = (
            ThreadPoolExecutor(
                max_workers=clone.workers,
                thread_name_prefix="su2-letta-local",
            )
            if clone.workers > 1
            else None
        )
        return clone

    def __del__(self):
        executor = getattr(self, "_solver_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)

    def frontier_contractions(self, **options):
        """Sparse reduced frontier energies and tied-parameter actions.

        Avoids dense frontier site and pair tensors. See
        :class:`pyqed.letta.su2_frontier.SU2FrontierContractions` for the
        Weichselbaum (2012) adaptation, exactness, budgets and restrictions.
        """
        from .su2_frontier import SU2FrontierContractions
        return SU2FrontierContractions(self, **options)

    def norm(self):
        sites = self.materialize()
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, self.mpo)
        return _real_scalar(contract_chain_expectation(sites, identity))

    def transform_orbitals(self, matrix, *, max_bond=128, cutoff=1e-12,
                           memory_limit=2**24, return_info=False):
        """Counter-transform reduced chain or NN LETTA coefficients by Γ(matrix).

        See :func:`pyqed.letta.biorthogonal.transform_orbitals` for conventions,
        Malmqvist/Knecht references, compression bounds, and restrictions.
        """
        if self.virtual_boundary == 'periodic':
            raise NotImplementedError("orbital transformations of closed multiplicity loops are unsupported")
        from .biorthogonal import transform_orbitals
        return transform_orbitals(self, matrix, max_bond=max_bond, cutoff=cutoff,
                                  memory_limit=memory_limit, return_info=return_info)

    def biorthogonalize(self, other, s, *, max_bond=None,
                        tol=1e-8, memory_limit=2**24, return_info=False):
        """Return two reduced LETTA states biorthogonalized for orbital metric s.

        See :func:`pyqed.letta.biorthogonal.biorthogonalize` for the explicit
        basis maps, Malmqvist/Knecht adaptation, and current restrictions.
        """
        if self.virtual_boundary == 'periodic' or getattr(other, 'virtual_boundary', 'open') == 'periodic':
            raise NotImplementedError("biorthogonalization of closed multiplicity loops is unsupported")
        from .biorthogonal import biorthogonalize
        return biorthogonalize(self, other, s, max_bond=max_bond,
                               tol=tol, memory_limit=memory_limit,
                               return_info=return_info)

    def overlap(self, other, *, s=None,
                      memory_limit=2**24, max_bond=None,
                      tol=1e-8, return_info=False):
        """Contract LETTA conditional tensors without materializing an MPS.

        See :func:`pyqed.letta.overlap.overlap` for the spatial-orbital
        metric, explicit biorthogonalization, state compression, and restrictions.
        """
        if self.virtual_boundary == 'periodic' or getattr(other, 'virtual_boundary', 'open') == 'periodic':
            raise NotImplementedError("use reduced materialized-chain contractions for closed-loop overlaps")
        from .overlap import overlap

        return overlap(
            self, other, s=s,
            memory_limit=memory_limit, max_bond=max_bond,
            tol=tol, return_info=return_info,
        )

    def _complementary_rayleigh_quotient(self, sites):
        """Evaluate the state through one exact direct-complementary bond action."""
        # Conditional gauges do not flatten the complete norm. Almost-null
        # tied directions can leave huge cancelling reduced coefficients;
        # using those directly also corrupts the global acceptance guard.
        # Gauge a temporary unfolded reduced MPS, without truncation or any
        # change to the variational LETTA tensors.
        from pyqed.mps.nonabelian.canonical import left_qr_sites

        sites = left_qr_sites(sites)
        for site in sites:
            metadata = getattr(site, "metadata", None)
            if isinstance(metadata, dict):
                metadata.pop("_cpp_split_site", None)
        local_environment = MovingEnvironment(
            sites,
            mpo_factors=self.mpo,
            complementary_operator_families=self._complementary_operators,
            materialize_complementary_family_operator_tables=False,
            su2_moving_environment=None,
            su2_boundary_environment=(
                self._su2_moving_environment if self.nsites >= 6 else None
            ),
        )
        local_environment.norm_stack.su2_boundary_environment = None
        local_environment.hamiltonian_stack.su2_moving_environment = (
            self._su2_moving_environment
        )
        self._su2_moving_environment.clear_boundaries()
        self._su2_moving_environment.clear_factor_routes()
        h_chain = BlockSparseEnvironmentChain.build(
            sites,
            self.mpo,
            renormalized_blocks=local_environment.hamiltonian_stack,
            sweep_direction="lr",
        )
        n_chain = BlockSparseEnvironmentChain.build(
            sites,
            local_environment.identity_mpo_factors,
            renormalized_blocks=local_environment.norm_stack,
            sweep_direction="lr",
        )
        merged = merge_mps_sites(sites[0], sites[1])
        layout = _ChannelResolvedPairSpace(sites[0], sites[1])
        h_action, n_action, _diagnostics = self._reduced_pair_transition_actions(
            0,
            sites,
            merged,
            layout,
            environment_sweeps=(
                h_chain.start_sweep("lr"),
                n_chain.start_sweep("lr"),
            ),
            materialize_metric_blocks=False,
        )
        vector = layout.pack_sites(sites[0], sites[1])
        metric_vector = np.asarray(n_action(vector))
        denominator = float(np.real(np.vdot(vector, metric_vector)))
        if denominator <= 0.0:
            raise FloatingPointError("SU2LETTA norm is non-positive.")
        return float(np.real(np.vdot(vector, h_action(vector))) / denominator)

    def expectation(self, mpo=None):
        if mpo is None and self._state_only:
            raise ValueError("This LETTA has no Hamiltonian; supply an explicit MPO.")
        factors = self.mpo if mpo is None else tuple(getattr(mpo, "factors", mpo))
        if mpo is None and self.hamiltonian_representation == "complementary":
            self._materialized_site_cache.clear()
        sites = self.materialize()
        if mpo is None and self.hamiltonian_representation == "complementary":
            return self._complementary_rayleigh_quotient(sites) + self.ecore
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, factors)
        denominator = _real_scalar(contract_chain_expectation(sites, identity))
        if denominator <= 0.0:
            raise FloatingPointError("SU2LETTA norm is non-positive.")
        numerator = _real_scalar(
            contract_chain_expectation(
                sites,
                factors,
                moving_environment=getattr(mpo, "moving_environment", None),
            )
        )
        shift = self.ecore if mpo is None else float(getattr(mpo, "ecore", 0.0))
        return numerator / denominator + shift

    expect = expectation

    @property
    def storage_nbytes(self):
        """Bytes owned by variational tensors and reusable reduced routes."""
        tensor_bytes = sum(
            np.asarray(block).nbytes
            for tensor in self.tensors
            for block in tensor.values()
        )
        base_bytes = sum(
            np.asarray(block).nbytes
            for site in self._base_sites
            for block in site.data.values()
        )
        route_bytes = sum(plan.nbytes for plan in self._we_route_cache.values())
        embedding_bytes = sum(
            indices.nbytes + blocks.nbytes
            for indices, blocks in self._embedding_basis_cache.values()
        )
        embedding_bytes += sum(
            int(record["embedding"].nbytes)
            + sum(np.asarray(block).nbytes for block in record["fixed"].values())
            for record in self._embedding_numeric_cache.values()
        )
        krylov_bytes = sum(
            np.asarray(vectors).nbytes
            for vectors in self._projected_krylov_cache.values()
        )
        whitener_bytes = sum(
            np.asarray(whitener).nbytes + np.asarray(spectrum).nbytes
            for _revision, whitener, spectrum, _backend
            in self._metric_whitener_cache.values()
        )
        materialized_bytes = sum(
            np.asarray(block).nbytes
            for tensor in self._materialized_site_cache.values()
            for block in tensor.data.values()
        )
        metric_topology_bytes = sum(
            indices.nbytes
            for groups in self._metric_component_topology_cache.values()
            for indices in groups
        )
        reduced_block_bytes = 0
        for core in self.mpo:
            for blocks in getattr(core, "_environment_reduced_block_cache", {}).values():
                reduced_block_bytes += sum(
                    np.asarray(block).nbytes for block in blocks.values()
                )
            plan = getattr(core, "normal_complementary_plan", None)
            if plan is not None:
                reduced_block_bytes += sum(
                    record["routes"].nbytes
                    for record in plan.get("_boundary_action_cache", {}).values()
                    if record.get("routes") is not None
                )
        return int(
            tensor_bytes
            + base_bytes
            + route_bytes
            + embedding_bytes
            + krylov_bytes
            + whitener_bytes
            + materialized_bytes
            + metric_topology_bytes
            + reduced_block_bytes
        )

    def _one_site_parameter_adjoint(self,site,bra,ket,core,left,right):
        """Pull a channel-resolved reduced one-site contraction back to parameters."""
        blocks=iter_one_site_reduced_adjoint(bra,ket,core,left,right)
        return self._wigner_eckart_route_plan(site).adjoint(blocks)

    def _parameter_chain_actions(self,sites,chain):
        return [self._one_site_parameter_adjoint(site,tensor,tensor,chain.mpo_factors[site],
                chain.left_envs[site].data,chain.right_envs[site].data)
                for site,tensor in enumerate(sites)]

    def _one_site_norm_components(self, site, core, left, right):
        """Contract the reduced identity directly into tied parameter blocks.

        Distinct physical/fusion paths and distinct future tie labels are
        orthogonal in the norm. Each raw conditional tensor block therefore
        supplies an independent metric component. Its virtual multiplicities
        remain fully coupled, including the closed seam. Structural routes,
        rather than current nonzero coefficients, define every direction.
        """
        plan = self._wigner_eckart_route_plan(site)
        components = []
        offset = 0
        for source_key in self._tensor_keys[site]:
            size = self.tensors[site][source_key].size
            start, offset = offset, offset + size
            key = source_key[:3]
            record = plan._group_by_key.get(key)
            if record is None:
                continue
            parameters, targets, shape = record
            selected = (parameters >= start) & (parameters < offset)
            if not np.any(selected):
                continue
            local = parameters[selected] - start
            l, p, r = np.unravel_index(targets[selected], shape)
            ql, qp, qr = key
            el, fr = left.get((ql, ql)), right.get((qr, qr))
            if el is None or fr is None:
                continue
            terms = _left_reduced_rank_coupled_block(core, ql, ql, qp, qp, qr, qr)
            block = np.zeros((size, size), dtype=complex)
            for (lc, rc), w in (terms or {}).items():
                e, f = el.get(int(lc)), fr.get(int(rc))
                if e is None or f is None:
                    continue
                irrep = core.right_channel_irreps[int(rc)]
                weights = [_component_basis_norm(qr, qr, irrep, m)
                           for m in ordered_two_m_values(irrep)]
                # Tile route pairs: no full lifted-site Gram matrix or unit tensors.
                for begin in range(0, local.size, 256):
                    a = slice(begin, begin + 256)
                    for end in range(0, local.size, 256):
                        b = slice(end, end + 256)
                        values = np.zeros((local[a].size, local[b].size), dtype=complex)
                        for x in range(w.shape[0]):
                            for y in range(w.shape[1]):
                                values += (e[x][np.ix_(l[a], l[b])]
                                           * f[y][np.ix_(r[a], r[b])]
                                           * w[x, y][np.ix_(p[a], p[b])] * weights[y])
                        bins = (local[a, None]*size + local[None, b]).ravel()
                        block += (np.bincount(bins, weights=values.real.ravel(), minlength=size*size)
                                  + 1j*np.bincount(bins, weights=values.imag.ravel(),
                                                  minlength=size*size)).reshape(size, size)
            components.append((np.arange(start, offset), .5*(block + block.conj().T)))
        return _ReducedMetricComponents(plan.nparameters, components)

    def _stream_parameter_actions(self, sites, factors, *, norm_components=False,
                                  sparse_boundaries=False, spill_directory=None,
                                  checkpoint_stride=None, right_checkpoints=None, norm_consumer=None):
        """Recompute right segments while carrying the current left boundary."""
        if norm_consumer is not None and (not norm_components or sparse_boundaries):
            raise ValueError('norm_consumer requires dense norm components')
        chain = BlockSparseEnvironmentChain.build(
            sites, factors, sweep_direction='lr', boundaries_only=True)
        stride = int(np.ceil(np.sqrt(len(sites)))) if len(sites) >= 8 else 1
        if checkpoint_stride is not None:
            if not np.isfinite(checkpoint_stride) or checkpoint_stride < 1 or int(checkpoint_stride) != checkpoint_stride:
                raise ValueError('checkpoint_stride must be a positive integer')
            stride = int(checkpoint_stride)
        if sparse_boundaries:
            if norm_components:
                raise NotImplementedError('Sparse identity-metric transport is unavailable')
            from pyqed.mps.nonabelian.sparse_boundary import SparseBoundary, iter_right_boundaries
            left = SparseBoundary.from_reduced(chain.left_envs[0], side='left', bond=0)
            endpoint = SparseBoundary.from_reduced(
                chain.right_envs[-1], side='right', bond=len(sites)-1)
            boundaries = iter_right_boundaries(
                sites, chain.mpo_factors, endpoint,
                moving_environment=self._su2_moving_environment, stride=stride,
                spill_directory=spill_directory, release_recoupling=True)
            if right_checkpoints is not None:
                boundaries = ((i, right_checkpoints[i][0]) for i in range(len(sites)))
            actions = []
            try:
                for site, right in boundaries:
                    tensor, core = sites[site], chain.mpo_factors[site]
                    actions.append(self._one_site_parameter_adjoint(
                        site, tensor, tensor, core, left, right))
                    right.release_decoded()
                    del right
                    if site+1 < len(sites):
                        left = left.advance(core, tensor,
                            moving_environment=self._su2_moving_environment,
                            spill_directory=spill_directory, release_recoupling=True)
                    core._environment_reduced_block_cache.clear()
            finally:
                boundaries.close()
                if right_checkpoints is not None:
                    for boundary, _ in right_checkpoints.values():
                        boundary.release_decoded()
                for core in chain.mpo_factors:
                    core._environment_reduced_block_cache.clear()
            return actions, []
        boundaries = chain.iter_right_boundaries(stride)
        sweep = chain.start_sweep('lr')
        actions, metrics = [], []
        try:
            for site, right in boundaries:
                tensor = sites[site]
                core = chain.mpo_factors[site]
                action = self._one_site_parameter_adjoint(
                    site, tensor, tensor, core, sweep.current_env.data, right.data)
                metric = (self._one_site_norm_components(
                    site, core, sweep.current_env.data, right.data) if norm_components else None)
                if norm_consumer is not None:
                    norm_consumer(site, action, metric)
                else:
                    actions.append(action)
                    if norm_components:
                        metrics.append(metric)
                del action, metric
                chain.right_envs[site] = None
                del right
                owner = getattr(core, 'normal_complementary_owner', None)
                if owner is not None:
                    owner.release_boundary('right', site)
                if site + 1 < len(sites):
                    sweep.advance_after_update(site, tensor, sites[site + 1])
        finally:
            boundaries.close()
            chain.right_envs.clear()
            for site, core in enumerate(chain.mpo_factors):
                owner = getattr(core, 'normal_complementary_owner', None)
                if owner is not None:
                    owner.release_boundary('right', site)
        return actions, metrics

    def reduced_parameter_actions(self, *, sparse_hamiltonian=False, spill_directory=None,
                                  checkpoint_stride=None):
        """Return exact reduced ``H*c`` and ``N*c`` for every current site.

        Direct Wigner--Eckart adjoints follow the reduced formulation of
        Weichselbaum, PRB 86, 245124 (2012), doi:10.1103/PhysRevB.86.245124.
        This is a LETTA adaptation with tied route pullbacks, not a separate
        optimization algorithm. Only explicit reduced MPOs are supported;
        no magnetic-state expansion or determinant projection is performed.
        Avoids adjacent-pair embeddings and their extra cancellation near
        stationarity. The returned actions omit the external core constant.
        Singleton physical multiplicities use virtual matrix products before
        summing operator components; other blocks retain tensor contractions.
        Ket absorption is shared by terms with the same incoming channel.
        One left boundary advances while right segments are reconstructed from
        exact checkpoints. For eight or more sites this retains O(sqrt(n))
        right blocks, with at most one extra right pass. This bounds the block
        count, not bytes or the size of an individual contraction.
        ``sparse_hamiltonian=True`` uses experimental real sparse Hamiltonian
        boundaries throughout checkpoint construction and local actions. Norm
        transport remains dense. Per-boundary storage is limited to 256 MiB,
        gathered arenas to 16 MiB; exceeding either raises MemoryError.
        With ``spill_directory`` each sparse Hamiltonian boundary instead uses
        a temporary file (8 GiB limit), deleted when that boundary is released.
        The directory must exist. This option requires ``sparse_hamiltonian``.
        ``checkpoint_stride`` optionally controls sparse Hamiltonian checkpoint
        spacing; 1 retains every right boundary and avoids recomputation.
        Disk payloads stay on disk, but O(n) metadata remains in RAM. Norm
        checkpoints retain their default spacing.
        """
        if self.hamiltonian_representation!='mpo':
            raise NotImplementedError('direct parameter actions require an explicit reduced MPO')
        if spill_directory is not None and not sparse_hamiltonian:
            raise ValueError('spill_directory requires sparse_hamiltonian=True')
        if checkpoint_stride is not None and not sparse_hamiltonian:
            raise ValueError('checkpoint_stride requires sparse_hamiltonian=True')
        sites=self.materialize()
        identity=_identity_mpo_factors_for_sites_and_mpo(sites,self.mpo)
        h, _ = self._stream_parameter_actions(
            sites, self.mpo, sparse_boundaries=sparse_hamiltonian, spill_directory=spill_directory,
            checkpoint_stride=checkpoint_stride)
        n, _ = self._stream_parameter_actions(sites, identity)
        return h, n

    def local_tangent_residual(self, *, metric_rtol=1e-11, support_rtol=1e-13,
                               max_parameters=4096, sparse_hamiltonian=False,
                               spill_directory=None, checkpoint_stride=None, _checkpoint_cache=None):
        """Audit one-site stationarity at the current reduced SU(2) state.

        Unlike local eigensolver residuals collected during a sweep, this
        computes every supported tangent residual with all other tensors fixed
        at the final state. Uses the existing Wigner--Eckart reduced actions
        and metric whitening, without determinant or magnetic-state projection.
        The maximum certifies retained-metric one-site stationarity, not an
        exact eigenstate or a global optimum. Materialized reduced metrics
        are limited by ``max_parameters``; no matrix-free metric audit is
        implemented here. Updates diagnostic fields, but does not change tensors.
        Closed loops use direct one-site adjoints with a common total Rayleigh
        quotient and conditional reduced norm blocks. Their audit consumes
        boundary stacks sequentially and never assembles an adjacent-pair
        Hamiltonian or a global whitening matrix. Norm blocks retain the
        existing support and eigenvalue thresholds; this remains an adaptation
        of the reduced Wigner--Eckart formulation cited in the class docstring.
        For real closed states with an explicit reduced MPO,
        ``sparse_hamiltonian=True`` uses sparse Hamiltonian checkpoints and
        actions. ``spill_directory`` optionally keeps their payload on disk,
        with the budgets documented in ``reduced_parameter_actions``. Norm
        transport and conditional metric blocks retain their dense path.
        Each site's metric is whitened and released before constructing the
        next, so the audit does not retain the full set of local metrics.
        ``checkpoint_stride`` controls only sparse Hamiltonian checkpoints,
        as in ``reduced_parameter_actions``.
        """
        if not np.isfinite(metric_rtol) or metric_rtol<=0:
            raise ValueError("metric_rtol must be finite and positive")
        if spill_directory is not None and not sparse_hamiltonian:
            raise ValueError('spill_directory requires sparse_hamiltonian=True')
        if checkpoint_stride is not None and not sparse_hamiltonian:
            raise ValueError('checkpoint_stride requires sparse_hamiltonian=True')
        if sparse_hamiltonian and (self.virtual_boundary != 'periodic'
                                   or self.hamiltonian_representation != 'mpo'):
            raise NotImplementedError('Sparse stationarity requires a closed state and explicit reduced MPO')
        right_checkpoints = None
        if sparse_hamiltonian and _checkpoint_cache is not None:
            signature = (id(self), self._state_revision, tuple(map(id, self.mpo)), 'lr')
            if (_checkpoint_cache.get('signature') == signature
                    and set(_checkpoint_cache.get('blocks', {})) == set(range(self.nsites))):
                right_checkpoints = _checkpoint_cache['blocks']
        sites = self.materialize()
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, self.mpo)
        if self.virtual_boundary == 'periodic':
            if any(self._pack_site(i).size > int(max_parameters) for i in range(self.nsites)):
                raise MemoryError('stationarity audit exceeds max_parameters')
            h, _ = self._stream_parameter_actions(
                sites, self.mpo, sparse_boundaries=sparse_hamiltonian,
                spill_directory=spill_directory, checkpoint_stride=checkpoint_stride,
                right_checkpoints=right_checkpoints)
            residuals = []
            norm = energy = None
            def consume_norm(site, nv, metric):
                nonlocal norm, energy
                if site == 0:
                    anchor = self._pack_site(0)
                    norm = float(np.real(np.vdot(anchor, nv)))
                    if norm <= 0:
                        raise np.linalg.LinAlgError('the local SU2 metric norm is null')
                    energy = float(np.real(np.vdot(anchor, h[0])) / norm)
                diagonal = metric.diagonal
                scale = max(float(np.max(np.abs(diagonal), initial=0.0)), 1.0)
                support = np.flatnonzero(diagonal > float(support_rtol)*scale)
                if not support.size:
                    raise np.linalg.LinAlgError('local SU2LETTA parameter support is null')
                residuals.append(metric.restrict(support).whitened_norm(
                    (h[site]-energy*nv)[support], rtol=metric_rtol) / np.sqrt(norm))
                h[site] = None
            self._stream_parameter_actions(sites, identity, norm_components=True,
                                           norm_consumer=consume_norm)
            self.tangent_residuals = tuple(map(float, residuals))
            self.max_tangent_residual = max(self.tangent_residuals, default=0.0)
            return self.max_tangent_residual
        h_chain = BlockSparseEnvironmentChain.build(sites, self.mpo)
        n_chain = BlockSparseEnvironmentChain.build(sites, identity)
        sweeps = (h_chain, n_chain)
        residuals = []
        for site in range(self.nsites):
            bond = min(site,self.nsites-2)
            current,support,h_action,n_action,_,_,info = self._frontier_wigner_eckart_matrix_free_problem(
                site,bond,support_rtol=support_rtol,max_parameters=max_parameters,
                max_dense_metric_parameters=max_parameters,sites=sites,
                environment_sweeps=sweeps)
            metric = info['_projected_metric']
            if isinstance(metric,_MatrixFreeProjectedMetric):
                raise MemoryError("stationarity audit requires a materialized reduced metric")
            if isinstance(metric,_ReducedMetricComponents):
                whitener,_,_,_ = metric.whitener(rtol=metric_rtol)
            else:
                whitener,_,_ = self._metric_whitener(metric,rtol=metric_rtol)
            vector = current[support]
            nv,hv=n_action(vector),h_action(vector)
            norm=float(np.real(np.vdot(vector,nv)))
            if norm<=0:
                raise np.linalg.LinAlgError("the local SU2 metric norm is null")
            energy=float(np.real(np.vdot(vector,hv))/norm)
            residuals.append(float(np.linalg.norm(whitener.conj().T@(hv-energy*nv))/np.sqrt(norm)))
        self.tangent_residuals = tuple(residuals)
        self.max_tangent_residual = max(residuals,default=0.)
        return self.max_tangent_residual

    @property
    def convergence_summary(self):
        """Return the latest complete-cycle convergence diagnostics."""
        latest = self.history[-1] if self.history else {}
        return {
            "converged": bool(self.converged),
            "cycles": int(len(self.history)),
            "energy": None if self.energy is None else float(self.energy),
            "energy_delta": latest.get("energy_delta"),
            "max_local_residual": latest.get("max_local_residual"),
            "max_truncation_error": latest.get("max_truncation_error"),
            "rejected_updates": latest.get("rejected_updates", 0),
            "consecutive_cycles": int(latest.get("consecutive_cycles", 0)),
            "storage_nbytes": int(self.storage_nbytes),
            "message": self.message,
        }

    def save_checkpoint(self, path):
        """Atomically save a restartable SU(2)-LETTA checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_mpo = None
        if self._hamiltonian_spec is None:
            memo = {}
            for core in self.mpo:
                for name in (
                    "_reduced_block_cache",
                    "_environment_reduced_block_cache",
                    "_block_cache",
                ):
                    cache = getattr(core, name, None)
                    if cache is not None:
                        memo[id(cache)] = {}
            checkpoint_mpo = copy.deepcopy(self.mpo, memo)
        payload = {
            "format": "pyqed-nonabelian-frontier-letta",
            "version": 3,
            "mpo": checkpoint_mpo,
            "hamiltonian_spec": self._hamiltonian_spec,
            "nelec": self.nelec,
            "spin": self.spin,
            "target_sector": self.target_sector,
            "tie": self.tie,
            "graph": self.graph,
            "max_frontier_states": self.max_frontier_states,
            "tie_history": self.tie_history,
            "D": self.D,
            "virtual_boundary": self.virtual_boundary,
            "compact_frontier": self.compact_frontier,
            "ecore": self.ecore,
            "core_energy": self.core_energy,
            "mpo_includes_core_energy": self.mpo_includes_core_energy,
            "base_sites": self._base_sites,
            "tensors": self.tensors,
            "history": self.history,
            "energy": self.energy,
            "converged": self.converged,
            "success": self.success,
            "message": self.message,
            "we_route_memory": self.we_route_memory,
            "n_threads": self.n_threads,
            "hamiltonian_representation_policy": (
                self._hamiltonian_representation_policy
            ),
            "hamiltonian_representation": self.hamiltonian_representation,
            "representation_decision": self._representation_decision,
            "global_guard_bonds": tuple(sorted(self._global_guard_bonds)),
            "projected_krylov": self._projected_krylov_cache,
        }
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return path

    @classmethod
    def load_checkpoint(cls, path, *, workers=1, n_threads=None):
        """Restore saved tensors and diagnostics without evaluating a throwaway state.

        The saved energy is restored as a diagnostic; call ``expectation()``
        to recompute it. A checkpoint without energy is evaluated after its
        tensors and Hamiltonian representation have been restored.
        An explicitly unset saved energy stays unset, allowing sparse-frontier
        evaluation without an implicit dense contraction on load.
        Thread defaults resolve from the current runtime rather than the
        checkpoint; Python setup defaults to one worker.
        """
        path = Path(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        legacy = (
            payload.get("format") == "pyqed-su2-letta"
            and payload.get("version") == 1
        )
        current = (
            payload.get("format") == "pyqed-nonabelian-frontier-letta"
            and payload.get("version") in {2, 3}
        )
        if not (legacy or current):
            raise ValueError("unsupported SU(2)-LETTA checkpoint format.")
        hamiltonian_spec = payload.get("hamiltonian_spec")
        common = {
            "graph": payload["graph"],
            "max_frontier_states": payload.get("max_frontier_states", 4096),
            "D": payload["D"],
            "virtual_boundary": payload.get("virtual_boundary","open"),
            "compact_frontier": payload.get("compact_frontier", True),
            "base_sites": payload["base_sites"],
            "tie": payload.get("tie", "physical"),
            "init": "mps",
            "workers": workers,
            "n_threads": n_threads,
            "we_route_memory": payload.get("we_route_memory", 256.0),
            "_evaluate_initial_state": False,
        }
        if hamiltonian_spec is not None:
            state = cls.from_integrals(**hamiltonian_spec, **common)
        else:
            state = cls(
                payload["mpo"],
                nelec=payload["nelec"],
                spin=payload["spin"],
                target_sector=payload.get("target_sector"),
                ecore=payload["ecore"],
                **common,
            )
        state.core_energy = float(payload.get("core_energy", state.ecore))
        state.ecore = float(payload["ecore"])
        state.mpo_includes_core_energy = bool(
            payload.get("mpo_includes_core_energy", False)
        )
        state.tensors = [
            {key: np.array(value, copy=True) for key, value in tensor.items()}
            for tensor in payload["tensors"]
        ]
        state._materialized_site_cache.clear()
        state._hamiltonian_representation_policy = payload.get(
            "hamiltonian_representation_policy",
            state._hamiltonian_representation_policy,
        )
        if (
            payload.get("hamiltonian_representation") == "mpo"
            and state.hamiltonian_representation == "complementary"
        ):
            state._materialize_explicit_hamiltonian()
        state._representation_decision = dict(
            payload.get("representation_decision", state._representation_decision)
        )
        state._global_guard_bonds = {
            (str(direction), int(bond))
            for direction, bond in payload.get("global_guard_bonds", ())
        }
        state.tie_history = list(payload.get("tie_history", ()))
        state.history = list(payload.get("history", ()))
        for key, vectors in payload.get("projected_krylov", {}).items():
            site, bond = key
            vectors = np.asarray(vectors)
            if (vectors.ndim != 2 or vectors.shape[0] != state._pack_site(site).size
                    or not np.all(np.isfinite(vectors))):
                raise ValueError("Invalid checkpoint Krylov vectors")
            state._projected_krylov_cache[(int(site), int(bond))] = vectors.copy()
        saved_energy = payload["energy"] if "energy" in payload else state.expectation()
        state.energy = None if saved_energy is None else float(saved_energy)
        state.converged = bool(payload.get("converged", False))
        state.success = payload.get("success")
        state.message = str(payload.get("message", "restored SU(2)-LETTA checkpoint"))
        return state

    def _pack_site(self, site):
        arrays = [self.tensors[site][key].reshape(-1) for key in self._tensor_keys[site]]
        if not arrays:
            return np.zeros(0)
        return np.concatenate(arrays)

    def _invalidate_materialized_sites(self, *sites):
        for site in sites:
            self._materialized_site_cache.pop(int(site), None)

    def _set_site_vector(self, site, vector):
        vector = np.asarray(vector)
        previous = self._pack_site(site)
        offset = 0
        updated = {}
        for key in self._tensor_keys[site]:
            shape = self.tensors[site][key].shape
            size = int(np.prod(shape))
            updated[key] = np.array(vector[offset : offset + size].reshape(shape), copy=True)
            offset += size
        if offset != vector.size:
            raise ValueError("local SU2LETTA parameter vector has the wrong size.")
        self.tensors[site] = updated
        if not np.array_equal(vector, previous):
            self._state_revision += 1
            self._site_revisions[int(site)] += 1
        cached = self._materialized_site_cache.get(int(site))
        route = self._we_route_cache.get(int(site))
        if (
            self._incremental_materialization_enabled
            and cached is not None
            and route is not None
        ):
            # Preserve references held by sweep environments, but overwrite the
            # entries. Adding a large trial delta and then its inverse loses
            # the incumbent to cancellation when an ill-conditioned solve fails.
            cached.data = route.tensor(vector).data
            cached.metadata.pop("_cpp_split_site", None)
        else:
            self._invalidate_materialized_sites(site)

    def _stationary_dependency_signature(self, site):
        """Return the exact tensor dependencies of a local effective problem."""
        site = int(site)
        return (
            int(self._site_revisions[site]),
            tuple(self._site_revisions[:site]),
            tuple(self._site_revisions[site + 1 :]),
        )

    def _quadratic_value(self, sites, site, vector, factors):
        self._set_site_vector(site, vector)
        varied = list(sites)
        varied[site] = self.materialize_site(site)
        owner = (
            self._su2_moving_environment
            if factors is self.mpo
            and self.hamiltonian_representation == "complementary"
            else None
        )
        return _real_scalar(
            contract_chain_expectation(
                varied,
                factors,
                moving_environment=owner,
            )
        )

    def _local_quadratic_matrices(
        self,
        site,
        *,
        support_rtol=1.0e-13,
        max_local_parameters=128,
    ):
        current = self._pack_site(site)
        sites = self.materialize()
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, self.mpo)
        dtype = np.result_type(current.dtype, complex)
        diagonal_n = np.zeros(current.size, dtype=float)
        diagonal_h = np.zeros(current.size, dtype=float)
        unit = np.zeros(current.size, dtype=dtype)
        try:
            for index in range(current.size):
                unit[index] = 1.0
                diagonal_n[index] = self._quadratic_value(sites, site, unit, identity)
                diagonal_h[index] = self._quadratic_value(sites, site, unit, self.mpo)
                unit[index] = 0.0
            scale = max(float(np.max(diagonal_n)), 1.0)
            support = np.flatnonzero(diagonal_n > float(support_rtol) * scale)
            if support.size == 0:
                raise np.linalg.LinAlgError("local SU2LETTA parameter support is null.")
            if support.size > int(max_local_parameters):
                raise MemoryError(
                    f"local SU2LETTA support has {support.size} parameters; "
                    "the exact polarization solver is limited to "
                    f"{int(max_local_parameters)}."
                )
            dimension = support.size
            metric = np.diag(diagonal_n[support]).astype(complex)
            hamiltonian = np.diag(diagonal_h[support]).astype(complex)
            for row in range(dimension):
                a = int(support[row])
                for col in range(row + 1, dimension):
                    b = int(support[col])
                    pair = np.zeros(current.size, dtype=dtype)
                    pair[a] = 1.0
                    pair[b] = 1.0
                    n_plus = self._quadratic_value(sites, site, pair, identity)
                    h_plus = self._quadratic_value(sites, site, pair, self.mpo)
                    pair[b] = 1.0j
                    n_i = self._quadratic_value(sites, site, pair, identity)
                    h_i = self._quadratic_value(sites, site, pair, self.mpo)
                    n_real = 0.5 * (n_plus - diagonal_n[a] - diagonal_n[b])
                    h_real = 0.5 * (h_plus - diagonal_h[a] - diagonal_h[b])
                    n_imag = 0.5 * (diagonal_n[a] + diagonal_n[b] - n_i)
                    h_imag = 0.5 * (diagonal_h[a] + diagonal_h[b] - h_i)
                    metric[row, col] = n_real + 1.0j * n_imag
                    hamiltonian[row, col] = h_real + 1.0j * h_imag
                    metric[col, row] = metric[row, col].conj()
                    hamiltonian[col, row] = hamiltonian[row, col].conj()
        finally:
            self._set_site_vector(site, current)
        return current, support, hamiltonian, metric

    @staticmethod
    def _apply_packed_action(action, vector):
        if isinstance(action, np.ndarray):
            return action @ vector
        return np.asarray(action(vector))

    @staticmethod
    def _resolve_packed_action(operator, template, layout):
        auxiliary = getattr(operator, "aux_packed_matvec", None)
        if auxiliary is not None:
            return auxiliary
        action, _ = _resolve_davidson_operator(operator, template, layout)
        return action

    @staticmethod
    def _map_tasks(executor, function, tasks):
        if executor is None:
            return [function(task) for task in tasks]
        # All NumPy work inside the shared pool uses one BLAS thread.  This
        # prevents nested BLAS pools from multiplying the requested workers.
        with threadpool_limits(limits=1):
            return list(executor.map(function, tasks))

    def _frontier_route_template(self, site):
        """Build reduced block shapes with scalar-backed, read-only placeholders."""
        base = self._base_sites[site]
        left_dims = self._bond_sector_dims(site, 0)
        right_dims = self._bond_sector_dims(site, 2)
        raw_shapes = {
            key: (left_dims[key[0]] * len(self._assignments[site]), block.shape[1],
                  right_dims[key[2]] * len(self._assignments[site + 1]))
            for key, block in base.data.items()
        }
        qns = [[q for q, d in left_dims.items()
                for _ in range(d * len(self._assignments[site]))],
               list(self.physical_sectors_by_site[site]),
               [q for q, d in right_dims.items()
                for _ in range(d * len(self._assignments[site + 1]))]]
        shapes = raw_shapes
        metadata = dict(physical_basis='fully_reduced_su2',
                        letta_frontier=self.frontiers[site + 1],
                        letta_invariant_ties=True, letta_tie=self.tie)
        if self.virtual_boundary == 'periodic':
            shapes = {
                key: (1 if site == 0 else self.D * left, physical,
                      1 if site == self.nsites - 1 else self.D * right)
                for key, (left, physical, right) in raw_shapes.items()
            }
            qns[0] = qns[0][:1] if site == 0 else qns[0] * self.D
            qns[2] = qns[2][:1] if site == self.nsites - 1 else qns[2] * self.D
            metadata.update(virtual_boundary='periodic', closure_spin=0,
                            closure_dimension=self.D)
        zero = np.zeros(())
        template = IrrepTensor(
            data={key: np.broadcast_to(zero, shape) for key, shape in shapes.items()},
            qns=qns, dirs=[-1, 1, 1], metadata=metadata)
        return template, raw_shapes

    def _compile_wigner_eckart_routes(self, site):
        """Compile the structural tied-parameter scatter map once."""
        site = int(site)
        current = self._pack_site(site)
        template, raw_shapes = self._frontier_route_template(site)
        if self.virtual_boundary=='periodic':
            # Compile ordinary frontier routes first, then carry the seam
            # spectator through their target indices without adding parameters.
            from .su2_closure import closure_scatter
        left_frontier = self.frontiers[site]
        right_frontier = self.frontiers[site + 1]
        left_maps = [
            dict(zip(left_frontier, values))
            for values in self._assignments[site]
        ]
        right_maps = [
            dict(zip(right_frontier, values))
            for values in self._assignments[site + 1]
        ]
        left_dims = self._bond_sector_dims(site, 0)
        right_dims = self._bond_sector_dims(site, 2)
        parameter_offsets = {}
        offset = 0
        for key in self._tensor_keys[site]:
            parameter_offsets[key] = offset
            offset += int(np.asarray(self.tensors[site][key]).size)
        if offset != current.size:
            raise RuntimeError("Tied parameter offsets do not cover the site.")
        groups = []
        nbytes = 0
        # Assignment compatibility depends on tie labels, not sector degeneracies.
        # Compile it once and reuse it across every reduced parameter block.
        assignment_pairs = {}
        for left_index, left_values in enumerate(left_maps):
            for right_index, right_values in enumerate(right_maps):
                if any(left_values[value] != right_values[value]
                       for value in left_frontier if value in right_values):
                    continue
                condition = tuple(right_values[parent]
                                  for parent in self.parent_sets[site])
                endpoint = left_values.get(site)
                assignment_pairs.setdefault((endpoint, condition), []).append(
                    (left_index, right_index)
                )
        for block_key, target_block in template.data.items():
            q_left, q_phys, q_right = block_key
            d_left = left_dims[q_left]
            d_right = right_dims[q_right]
            endpoint_value = self._tie_index_by_site[site][
                q_phys if self.tie == "physical" else q_left
            ] if site in left_frontier else None
            indices, targets = [], []
            target_shape = raw_shapes[block_key]
            for source_key in self._tensor_keys[site]:
                if source_key[:3] != block_key:
                    continue
                pairs = assignment_pairs.get((endpoint_value, source_key[3]))
                if not pairs:
                    continue
                left_indices, right_indices = np.asarray(pairs, dtype=np.intp).T
                source_shape = np.asarray(self.tensors[site][source_key]).shape
                local = np.arange(int(np.prod(source_shape)), dtype=np.intp)
                i_left, i_phys, i_right = np.unravel_index(local, source_shape)
                target = (
                    ((left_indices[None, :] * d_left + i_left[:, None])
                     * target_shape[1] + i_phys[:, None]) * target_shape[2]
                    + right_indices[None, :] * d_right + i_right[:, None]
                )
                indices.append(np.repeat(parameter_offsets[source_key] + local,
                                         len(pairs)))
                targets.append(target.reshape(-1))
            if not indices:
                continue
            indices = np.concatenate(indices)
            targets = np.concatenate(targets)
            if self.virtual_boundary=='periodic':
                _,closed_targets,closed_shape = closure_scatter(
                    target_shape,site,self.nsites,self.D,indices=targets)
                # Each raw entry appears once at endpoints and D times inside.
                copies = 1 if site in (0,self.nsites-1) else self.D
                targets = closed_targets
                indices = np.tile(indices,copies)
                target_shape = closed_shape
            nbytes += int(indices.nbytes + targets.nbytes)
            if nbytes > self._we_route_limit_bytes:
                raise MemoryError(
                    "packed Wigner--Eckart routes require more than "
                    f"{self.we_route_memory:g} MiB; increase we_route_memory."
                )
            groups.append((block_key, indices, targets, target_shape))
        if not groups:
            raise np.linalg.LinAlgError("the local Wigner--Eckart route is empty.")
        return _FrontierTieRoutePlan(
            template=template,
            groups=groups,
            nparameters=current.size,
            nbytes=nbytes,
        )

    def _compact_frontier_routes(self):
        """Keep coordinates on complete structural paths, regardless of values."""
        plans = [self._compile_wigner_eckart_routes(i) for i in range(self.nsites)]
        dimensions = [{q: p.template.legs[0].sector_dim(q)
                       for q in p.template.legs[0].sectors} for p in plans]
        last = plans[-1].template.legs[2]
        dimensions.append({q: last.sector_dim(q) for q in last.sectors})

        def edges(plan):
            for key, _, targets, shape in plan.groups:
                left, _, right = np.unravel_index(targets, shape)
                yield key[0], key[2], left, right

        forward = [{q: np.ones(d, dtype=bool) for q, d in dimensions[0].items()}]
        for i, plan in enumerate(plans):
            mask = {q: np.zeros(d, dtype=bool) for q, d in dimensions[i + 1].items()}
            for lq, rq, left, right in edges(plan):
                mask[rq][right[forward[i][lq][left]]] = True
            forward.append(mask)
        backward = [None] * (self.nsites + 1)
        backward[-1] = {q: np.ones(d, dtype=bool) for q, d in dimensions[-1].items()}
        for i in range(self.nsites - 1, -1, -1):
            mask = {q: np.zeros(d, dtype=bool) for q, d in dimensions[i].items()}
            for lq, rq, left, right in edges(plans[i]):
                mask[lq][left[backward[i + 1][rq][right]]] = True
            backward[i] = mask
        retained, remaps, statistics = [], [], []
        for bond, dims in enumerate(dimensions):
            keep = {q: np.flatnonzero(forward[bond][q] & backward[bond][q]) for q in dims}
            maps = {q: np.full(d, -1, dtype=np.intp) for q, d in dims.items()}
            for q, indices in keep.items():
                maps[q][indices] = np.arange(indices.size)
            retained.append(keep)
            remaps.append(maps)
            statistics.append(dict(bond=bond, original=sum(dims.values()),
                                   retained=sum(v.size for v in keep.values())))
        compact_plans = {}
        zero = np.zeros(())
        for site, plan in enumerate(plans):
            groups, data = [], {}
            for key, parameters, targets, shape in plan.groups:
                left, physical, right = np.unravel_index(targets, shape)
                left = remaps[site][key[0]][left]
                right = remaps[site + 1][key[2]][right]
                keep = (left >= 0) & (right >= 0)
                if not np.any(keep):
                    continue
                compact_shape = (retained[site][key[0]].size, shape[1],
                                 retained[site + 1][key[2]].size)
                compact_targets = np.ravel_multi_index(
                    (left[keep], physical[keep], right[keep]), compact_shape)
                groups.append((key, parameters[keep], compact_targets, compact_shape))
                data[key] = np.broadcast_to(zero, compact_shape)
            qns = [[q for q, indices in retained[site].items() for _ in indices],
                   list(plan.template.qns[1]),
                   [q for q, indices in retained[site + 1].items() for _ in indices]]
            template = IrrepTensor(data=data, qns=qns, dirs=list(plan.template.dirs),
                metadata={**plan.template.metadata, 'frontier_compacted': True})
            compact_plans[site] = _FrontierTieRoutePlan(
                template=template, groups=groups, nparameters=plan.nparameters,
                nbytes=sum(p.nbytes + t.nbytes for _, p, t, _ in groups))
        self._we_route_cache.update(compact_plans)
        self._frontier_compaction = tuple(statistics)

    @property
    def frontier_compaction(self):
        """Original and retained reduced multiplicities at each closed-loop cut."""
        if self.virtual_boundary == 'periodic' and self.compact_frontier:
            self._wigner_eckart_route_plan(0)
        return self._frontier_compaction

    def _wigner_eckart_route_plan(self, site):
        site = int(site)
        plan = self._we_route_cache.get(site)
        if plan is not None:
            self._we_route_cache_hits += 1
            return plan
        self._we_route_cache_misses += 1
        if self.virtual_boundary == 'periodic' and self.compact_frontier:
            self._compact_frontier_routes()
            return self._we_route_cache[site]
        plan = self._compile_wigner_eckart_routes(site)
        self._we_route_cache[site] = plan
        return plan

    @property
    def wigner_eckart_cache_stats(self):
        return {
            "plans": int(len(self._we_route_cache)),
            "hits": int(self._we_route_cache_hits),
            "misses": int(self._we_route_cache_misses),
            "bytes": int(sum(plan.nbytes for plan in self._we_route_cache.values())),
            "limit_bytes": int(self._we_route_limit_bytes),
            "routes": int(
                sum(plan.route_count for plan in self._we_route_cache.values())
            ),
            "backend": "frontier-block-scatter",
        }

    def _magnetic_expansion_factor(self, bond):
        factors = []
        for site in (int(bond), int(bond) + 1):
            core = self.mpo[site]
            for irreps in (
                getattr(core, "left_channel_irreps", ()),
                getattr(core, "right_channel_irreps", ()),
            ):
                if irreps:
                    factors.append(
                        sum(irrep.dim for irrep in irreps) / float(len(irreps))
                    )
        return max(factors, default=1.0)

    @staticmethod
    def _select_local_solver(requested):
        requested = str(requested).lower().replace("-", "_")
        if requested == "auto":
            return "wigner_eckart"
        if requested not in {"wigner_eckart", "polarization"}:
            raise ValueError(
                "SU2LETTA solver must be 'auto', 'wigner_eckart', "
                "or 'polarization'."
            )
        return requested

    def _apply_packed_columns(self, action, columns, *, executor=None):
        columns = np.asarray(columns)
        dense = getattr(action, "dense_matrix", None)
        if isinstance(action, np.ndarray):
            dense = action
        if dense is not None:
            return np.asarray(dense) @ columns
        if columns.shape[1] == 0:
            return np.zeros_like(columns)
        if executor is None or columns.shape[1] == 1:
            return np.column_stack(
                [self._apply_packed_action(action, column) for column in columns.T]
            )
        workers = min(int(getattr(executor, "_max_workers", 1)), columns.shape[1])
        chunks = tuple(
            chunk
            for chunk in np.array_split(np.arange(columns.shape[1]), workers)
            if chunk.size
        )

        def apply_chunk(indices):
            return np.column_stack(
                [
                    self._apply_packed_action(action, columns[:, int(index)])
                    for index in indices
                ]
            )

        return np.concatenate(
            self._map_tasks(executor, apply_chunk, chunks), axis=1
        )

    @staticmethod
    def _site_topology_signature(site):
        return tuple(
            (key, tuple(map(int, np.asarray(block).shape)))
            for key, block in sorted(site.data.items())
        )

    def _pair_coordinate_space(self, bond, sites):
        """Return the fixed-D pair topology shared by successive local solves."""
        bond = int(bond)
        signature = (
            self._site_topology_signature(sites[bond]),
            self._site_topology_signature(sites[bond + 1]),
        )
        cached = self._pair_coordinate_cache.get(bond)
        if cached is not None and cached[0] == signature:
            self._pair_coordinate_cache_hits += 1
            return cached[1], cached[2]
        merged = merge_mps_sites(sites[bond], sites[bond + 1])
        layout = _ChannelResolvedPairSpace(sites[bond], sites[bond + 1])
        self._pair_coordinate_cache[bond] = (signature, merged, layout)
        self._pair_coordinate_cache_misses += 1
        return merged, layout

    @staticmethod
    def _embedding_fixed_block(site, key, shape, dtype):
        block = site.data.get(key)
        if block is None:
            return np.zeros(shape, dtype=dtype)
        return np.asarray(block, dtype=dtype)

    def _local_pair_embedding(
        self, site, sites, bond, layout, *, matrix_free=False
    ):
        """Embed tied one-site parameters into a packed two-site tensor."""
        current = self._pack_site(site)
        dtype = np.result_type(current.dtype, complex)
        routes = self._wigner_eckart_route_plan(site)
        if routes.nparameters != current.size:
            raise RuntimeError(
                "Cached Wigner--Eckart basis does not match the local parameter space."
            )
        if isinstance(layout, _ChannelResolvedPairSpace):
            if (
                isinstance(routes, _FrontierTieRoutePlan)
                and matrix_free
            ):
                fixed_site = sites[bond + 1] if site == bond else sites[bond]
                return current, _MatrixFreePairEmbedding(
                    layout,
                    routes,
                    fixed_site,
                    varied_left=site == bond,
                )
            layout_signature = tuple(
                (entry.key, entry.shape, int(entry.offset), int(entry.size))
                for entry in layout.entries
            )
            plan_key = (int(site), int(bond), layout_signature, np.dtype(dtype).str)
            plan = self._embedding_plan_cache.get(plan_key)
            if plan is None:
                records = []
                for entry in layout.entries:
                    q_l, q_p1, q_p2, q_r, q_mid = entry.key
                    varied_key = (
                        (q_l, q_p1, q_mid)
                        if site == bond
                        else (q_mid, q_p2, q_r)
                    )
                    fixed_key = (
                        (q_mid, q_p2, q_r)
                        if site == bond
                        else (q_l, q_p1, q_mid)
                    )
                    cache_key = (int(site), varied_key)
                    cached_basis = self._embedding_basis_cache.get(cache_key)
                    if cached_basis is None:
                        cached_basis = routes.active_block_basis(varied_key, dtype=dtype)
                        self._embedding_basis_cache[cache_key] = cached_basis
                        self._embedding_basis_cache_misses += 1
                    else:
                        self._embedding_basis_cache_hits += 1
                    active, varied_blocks = cached_basis
                    active_key = (
                        int(site), int(bond), int(entry.offset), int(entry.size)
                    )
                    self._embedding_active_columns_cache[active_key] = np.asarray(
                        active, dtype=np.int64
                    )
                    template = (
                        sites[bond + 1].data.get(fixed_key)
                        if site == bond
                        else sites[bond].data.get(fixed_key)
                    )
                    if template is None:
                        continue
                    records.append(
                        (
                            entry,
                            fixed_key,
                            tuple(np.asarray(template).shape),
                            varied_blocks,
                            np.asarray(active, dtype=np.int64),
                        )
                    )
                plan = tuple(records)
                self._embedding_plan_cache[plan_key] = plan
            fixed_site = sites[bond + 1] if site == bond else sites[bond]
            cached = self._embedding_numeric_cache.get(plan_key)
            if cached is None:
                embedding = _StructuredPairEmbedding(layout.size, current.size)
                fixed_blocks = {}
                changed = None
            else:
                embedding = cached["embedding"]
                fixed_blocks = cached["fixed"]
                changed = set()
                for _entry, fixed_key, fixed_shape, _varied_blocks, _active in plan:
                    if fixed_key in changed:
                        continue
                    fixed = self._embedding_fixed_block(
                        fixed_site, fixed_key, fixed_shape, dtype
                    )
                    previous = fixed_blocks.get(fixed_key)
                    if previous is None or not np.array_equal(fixed, previous):
                        changed.add(fixed_key)
                if not changed:
                    self._embedding_numeric_cache_hits += 1
                    self._embedding_reused_blocks += len(plan)
                    return current, embedding
                self._embedding_incremental_updates += 1
            fixed_values = {}
            deltas = {}
            for _entry, fixed_key, fixed_shape, _varied_blocks, _active in plan:
                if fixed_key in fixed_values:
                    continue
                fixed = self._embedding_fixed_block(
                    fixed_site, fixed_key, fixed_shape, dtype
                )
                previous = fixed_blocks.get(fixed_key)
                fixed_values[fixed_key] = fixed
                deltas[fixed_key] = fixed if previous is None else fixed - previous
            for entry, fixed_key, fixed_shape, varied_blocks, active in plan:
                if changed is not None and fixed_key not in changed:
                    self._embedding_reused_blocks += 1
                    continue
                if active.size == 0:
                    continue
                delta = deltas[fixed_key]
                q_l, q_p1, q_p2, q_r, q_mid = entry.key
                if site == bond:
                    pair_blocks = np.tensordot(
                        varied_blocks, delta, axes=([3], [0])
                    )
                else:
                    pair_blocks = np.moveaxis(
                        np.tensordot(
                            delta, varied_blocks, axes=([2], [1])
                        ),
                        2,
                        0,
                    )
                contribution = pair_blocks.reshape(active.size, -1).T
                embedding.add_block(
                    entry.offset,
                    entry.size,
                    active,
                    contribution,
                )
                self._embedding_updated_blocks += 1
            for fixed_key, fixed in fixed_values.items():
                fixed_blocks[fixed_key] = np.array(fixed, copy=True)
            self._embedding_numeric_cache[plan_key] = {
                "embedding": embedding,
                "fixed": fixed_blocks,
            }
            return current, embedding
        columns = []
        for varied in routes.basis:
            if site == bond:
                left, right = varied, sites[bond + 1]
            else:
                left, right = sites[bond], varied
            if isinstance(layout, _ChannelResolvedPairSpace):
                packed = layout.pack_sites(left, right)
            else:
                pair = merge_mps_sites(left, right)
                packed, _ = pack_two_site_state(pair, layout=layout)
            columns.append(np.asarray(packed, dtype=dtype))
        if not columns:
            return current, np.zeros((0, 0), dtype=dtype)
        return current, np.column_stack(columns)

    @staticmethod
    def _project_metric_blocks(embedding, metric_blocks):
        """Contract ``E† N E`` directly from reduced pair-metric blocks."""
        embedding = np.asarray(embedding)
        projected = np.zeros(
            (embedding.shape[1], embedding.shape[1]),
            dtype=np.result_type(embedding.dtype, complex),
        )
        active_designs = {}

        def active_design(entry):
            key = (int(entry.offset), int(entry.size))
            cached = active_designs.get(key)
            if cached is not None:
                return cached
            rows = embedding[key[0] : key[0] + key[1]]
            columns = np.flatnonzero(np.any(rows != 0.0, axis=0))
            cached = (columns, np.ascontiguousarray(rows[:, columns]))
            active_designs[key] = cached
            return cached

        for out_entry, in_entry, block in metric_blocks:
            out_columns, out_design = active_design(out_entry)
            in_columns, in_design = active_design(in_entry)
            if out_columns.size == 0 or in_columns.size == 0:
                continue
            contribution = out_design.conj().T @ (
                np.asarray(block) @ in_design
            )
            projected[np.ix_(out_columns, in_columns)] += contribution
        return 0.5 * (projected + projected.conj().T)

    def _project_metric_components(
        self,
        embedding,
        metric_blocks,
        *,
        site,
        bond,
        factorized=False,
    ):
        """Project and assemble only connected tied-metric components."""
        structured = isinstance(embedding, _StructuredPairEmbedding)
        if not structured:
            embedding = np.asarray(embedding)
        dimension = int(embedding.shape[1])
        active_designs = {}

        def active_design(entry):
            key = (int(entry.offset), int(entry.size))
            cached = active_designs.get(key)
            if cached is None:
                if structured:
                    columns, rows = embedding.block_design(*key)
                    cached = (columns, np.ascontiguousarray(rows))
                else:
                    rows = embedding[key[0] : key[0] + key[1]]
                    columns = self._embedding_active_columns_cache.get(
                        (int(site), int(bond), key[0], key[1])
                    )
                    if columns is None:
                        columns = np.flatnonzero(np.any(rows != 0.0, axis=0))
                    cached = (columns, np.ascontiguousarray(rows[:, columns]))
                active_designs[key] = cached
            return cached

        records = []
        topology_values = []
        for out_entry, in_entry, block in metric_blocks:
            out_columns, out_design = active_design(out_entry)
            in_columns, in_design = active_design(in_entry)
            if out_columns.size == 0 or in_columns.size == 0:
                continue
            if factorized:
                panel = in_design.reshape(
                    in_entry.shape[0] * in_entry.shape[1],
                    in_entry.shape[2] * in_entry.shape[3],
                    in_columns.size,
                )
                applied = sum(
                    np.einsum("ab,bcn,dc->adn", left, panel, right, optimize=True)
                    for left, right in block
                )
                contribution = out_design.conj().T @ applied.reshape(
                    out_entry.size, in_columns.size
                )
            else:
                contribution = out_design.conj().T @ (
                    np.asarray(block) @ in_design
                )
            records.append((out_columns, in_columns, contribution))
            topology_values.append(
                (
                    int(out_entry.offset),
                    int(in_entry.offset),
                    tuple(map(int, out_columns)),
                    tuple(map(int, in_columns)),
                )
            )
        topology_key = (
            int(site),
            int(bond),
            dimension,
            tuple(topology_values),
        )
        groups = self._metric_component_topology_cache.get(topology_key)
        if groups is None:
            parent = np.arange(dimension, dtype=np.int64)

            def find(index):
                index = int(index)
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = int(parent[index])
                return index

            def union(left, right):
                left_root = find(left)
                right_root = find(right)
                if left_root != right_root:
                    parent[right_root] = left_root

            for out_columns, in_columns, _contribution in records:
                if out_columns.size == 0 or in_columns.size == 0:
                    continue
                anchor = int(out_columns[0])
                for index in out_columns[1:]:
                    union(anchor, int(index))
                for index in in_columns:
                    union(anchor, int(index))
            grouped = {}
            for index in range(dimension):
                grouped.setdefault(find(index), []).append(index)
            groups = tuple(
                np.asarray(values, dtype=np.int64)
                for values in grouped.values()
            )
            self._metric_component_topology_cache[topology_key] = groups
            self._metric_component_cache_misses += 1
        else:
            self._metric_component_cache_hits += 1

        labels = np.empty(dimension, dtype=np.int64)
        positions = np.empty(dimension, dtype=np.int64)
        blocks = []
        for label, indices in enumerate(groups):
            labels[indices] = label
            positions[indices] = np.arange(indices.size)
            blocks.append(np.zeros((indices.size, indices.size), dtype=complex))
        for out_columns, in_columns, contribution in records:
            # Every record was joined into one component when grouping its columns.
            label = int(labels[out_columns[0]])
            blocks[label][np.ix_(positions[out_columns], positions[in_columns])] += contribution
        components = []
        for indices, block in zip(groups, blocks):
            block = 0.5 * (block + block.conj().T)
            if np.max(np.abs(block), initial=0.0) > 1.0e-15:
                components.append((indices, block))
        return _ReducedMetricComponents(dimension, components)

    @staticmethod
    def _metric_whitener(metric, *, rtol):
        """Return an exact block-aware whitener for a projected norm."""
        metric = 0.5 * (np.asarray(metric) + np.asarray(metric).conj().T)
        dimension = int(metric.shape[0])
        scale = max(float(np.max(np.abs(metric), initial=0.0)), 1.0)
        threshold = float(rtol) * scale
        diagonal = np.real(np.diag(metric))
        off_diagonal = metric - np.diag(diagonal)
        off_error = float(np.linalg.norm(off_diagonal))
        if off_error <= threshold * max(np.sqrt(dimension), 1.0):
            keep = diagonal > threshold
            if not np.any(keep):
                raise np.linalg.LinAlgError(
                    "the projected SU2LETTA parameter metric is singular."
                )
            indices = np.flatnonzero(keep)
            whitener = np.zeros((dimension, indices.size), dtype=metric.dtype)
            whitener[indices, np.arange(indices.size)] = 1.0 / np.sqrt(
                diagonal[indices]
            )
            identity_error = float(
                np.max(np.abs(diagonal[indices] - 1.0), initial=0.0)
            )
            backend = "conditional_identity" if identity_error <= threshold else "diagonal"
            return whitener, diagonal[indices], backend

        adjacency = np.abs(off_diagonal) > threshold
        unseen = set(range(dimension))
        components = []
        while unseen:
            seed = unseen.pop()
            component = {seed}
            frontier = [seed]
            while frontier:
                row = frontier.pop()
                neighbors = set(np.flatnonzero(adjacency[row])) & unseen
                unseen.difference_update(neighbors)
                component.update(neighbors)
                frontier.extend(neighbors)
            components.append(np.asarray(sorted(component), dtype=np.int64))

        columns = []
        retained = []
        for indices in components:
            block = metric[np.ix_(indices, indices)]
            values, vectors = np.linalg.eigh(block)
            keep = values > threshold
            if not np.any(keep):
                continue
            local = vectors[:, keep] / np.sqrt(values[keep])
            lifted = np.zeros((dimension, local.shape[1]), dtype=metric.dtype)
            lifted[indices] = local
            columns.append(lifted)
            retained.extend(map(float, values[keep]))
        if not columns:
            raise np.linalg.LinAlgError(
                "the projected SU2LETTA parameter metric is singular."
            )
        return (
            np.column_stack(columns),
            np.asarray(retained),
            "block_eigh" if len(components) > 1 else "dense_eigh",
        )


    def _install_projected_factor_route(
        self,
        *,
        site,
        bond,
        parent_action,
        orthonormal_design,
    ):
        """Install ``E† H_eff E`` in the persistent reduced C++ owner."""
        owner = getattr(parent_action, "su2_moving_environment", None)
        factor_route_key = getattr(parent_action, "factor_route_key", None)
        projection_builder = getattr(
            parent_action, "operator_projection_blocks", None
        )
        parent_dimension, dimension = map(int, orthonormal_design.shape)
        if (
            owner is None
            or factor_route_key is None
            or not callable(projection_builder)
            or not owner.factor_route_installed(
                factor_route_key, parent_dimension
            )
        ):
            return None

        from pyqed.mps.nonabelian import _su2_kernel

        projection_blocks = tuple(
            (rows, indices, np.ascontiguousarray(transform.real)
             if np.iscomplexobj(transform) and not np.any(transform.imag)
             else transform)
            for rows, indices, transform in projection_builder(orthonormal_design)
        )
        if not projection_blocks:
            return None
        topology_signature = tuple(
            (
                int(row_slice.start),
                int(row_slice.stop),
                tuple(map(int, indices)),
            )
            for row_slice, indices, _transform in projection_blocks
        )
        cache_key = (
            int(site),
            int(bond),
            parent_dimension,
            dimension,
            topology_signature,
        )
        cached = self._projected_route_cache.get(cache_key)
        if cached is None:
            topology_revision = _su2_kernel._cpp_array_revision(
                np.asarray(
                    [site, bond, parent_dimension, dimension], dtype=np.int64
                ),
                *(
                    value
                    for row_slice, indices, _transform in projection_blocks
                    for value in (
                        np.asarray(
                            [int(row_slice.start), int(row_slice.stop)],
                            dtype=np.int64,
                        ),
                        indices,
                    )
                ),
            )
            cached = int(topology_revision)
            self._projected_route_cache[cache_key] = cached
            self._projected_route_cache_misses += 1
        else:
            self._projected_route_cache_hits += 1
        topology_revision = int(cached)
        transforms = tuple(block[2] for block in projection_blocks)
        numeric_revision = int(_su2_kernel._cpp_array_revision(*transforms))
        projection_key = (
            f"letta:{int(site)}:{int(bond)}:{factor_route_key}:"
            f"{topology_revision}"
        )
        owner.install_indexed_factor_route_projection(
            projection_key,
            factor_route_key,
            projection_blocks,
            parent_dimension,
            dimension,
            topology_revision,
            numeric_revision,
        )
        return {
            "owner": owner,
            "projection_key": projection_key,
            "projection_blocks": projection_blocks,
            "real_projection": all(not np.iscomplexobj(block[2]) for block in projection_blocks),
            "topology_revision": topology_revision,
            "numeric_revision": numeric_revision,
        }

    def _prepare_wigner_eckart_local_problem(
        self,
        site,
        *,
        support_rtol=1.0e-13,
        max_parameters=4096,
        executor=None,
        sites=None,
    ):
        """Prepare cached native reduced transition routes for one local solve."""
        current = self._pack_site(site)
        sites = self.materialize() if sites is None else list(sites)
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, self.mpo)
        routes = self._wigner_eckart_route_plan(site)
        h_plan = LocalTransitionPlan.build(sites, self.mpo, site)
        n_plan = LocalTransitionPlan.build(sites, identity, site)

        def pair_value(pair):
            bra_index, ket_index = pair
            bra_site = routes.basis[bra_index]
            ket_site = routes.basis[ket_index]
            metric_value = n_plan.contract(bra_site, ket_site)
            hamiltonian_value = h_plan.contract(bra_site, ket_site)
            return bra_index, ket_index, metric_value, hamiltonian_value

        diagonal_records = self._map_tasks(
            executor,
            pair_value,
            tuple((index, index) for index in range(current.size)),
        )
        diagonal_n = np.zeros(current.size, dtype=float)
        diagonal_h = np.zeros(current.size, dtype=float)
        for row, _col, n_value, h_value in diagonal_records:
            diagonal_n[row] = _real_scalar(n_value)
            diagonal_h[row] = _real_scalar(h_value)
        scale = max(float(np.max(diagonal_n, initial=0.0)), 1.0)
        support = np.flatnonzero(diagonal_n > float(support_rtol) * scale)
        if support.size == 0:
            raise np.linalg.LinAlgError("local SU2LETTA parameter support is null.")
        if support.size > int(max_parameters):
            raise MemoryError(
                f"local SU2LETTA support has {support.size} parameters; "
                "the matrix-free Wigner--Eckart solver is limited to "
                f"{int(max_parameters)}."
            )

        diagnostics = {
            "route_backend": routes.backend,
            "route_bytes": int(routes.nbytes),
            "route_count": int(routes.route_count),
            "hamiltonian_cached_sites": int(h_plan.cached_sites),
            "hamiltonian_traversed_sites": int(h_plan.traversed_sites),
            "norm_cached_sites": int(n_plan.cached_sites),
            "norm_traversed_sites": int(n_plan.traversed_sites),
        }
        return (
            current,
            support,
            diagonal_h[support],
            diagonal_n[support],
            routes,
            h_plan,
            n_plan,
            pair_value,
            diagnostics,
        )

    def _wigner_eckart_local_matrices(
        self,
        site,
        *,
        support_rtol=1.0e-13,
        max_local_parameters=128,
        executor=None,
        sites=None,
    ):
        """Build the exact small dense problem from cached reduced routes."""
        (
            current,
            support,
            diagonal_h,
            diagonal_n,
            _routes,
            _h_plan,
            _n_plan,
            pair_value,
            diagnostics,
        ) = self._prepare_wigner_eckart_local_problem(
            site,
            support_rtol=support_rtol,
            max_parameters=max_local_parameters,
            executor=executor,
            sites=sites,
        )

        dimension = support.size
        metric = np.diag(diagonal_n).astype(complex)
        hamiltonian = np.diag(diagonal_h).astype(complex)
        off_diagonal = tuple(
            (int(support[row]), int(support[col]))
            for row in range(dimension)
            for col in range(row + 1, dimension)
        )
        records = self._map_tasks(executor, pair_value, off_diagonal)
        positions = {int(value): index for index, value in enumerate(support)}
        for bra_index, ket_index, n_value, h_value in records:
            row = positions[bra_index]
            col = positions[ket_index]
            metric[row, col] = n_value
            metric[col, row] = np.conj(n_value)
            hamiltonian[row, col] = h_value
            hamiltonian[col, row] = np.conj(h_value)
        return current, support, hamiltonian, metric, diagnostics

    def _wigner_eckart_matrix_free_problem(
        self,
        site,
        *,
        support_rtol=1.0e-13,
        max_parameters=4096,
        executor=None,
        sites=None,
    ):
        """Return exact reduced ``H x`` and ``N x`` actions without dense matrices."""
        (
            current,
            support,
            h_diag,
            n_diag,
            routes,
            h_plan,
            n_plan,
            _pair_value,
            diagnostics,
        ) = self._prepare_wigner_eckart_local_problem(
            site,
            support_rtol=support_rtol,
            max_parameters=max_parameters,
            executor=executor,
            sites=sites,
        )
        cache = {"vector": None, "hamiltonian": None, "metric": None}
        matvec_counts = {"combined": 0, "cache_hits": 0}

        def apply_both(vector):
            vector = np.asarray(vector, dtype=complex).reshape(-1)
            if vector.size != support.size:
                raise ValueError("matrix-free Wigner--Eckart vector has the wrong size.")
            cached = cache["vector"]
            if cached is not None and np.array_equal(vector, cached):
                matvec_counts["cache_hits"] += 1
                return cache["hamiltonian"], cache["metric"]
            coefficients = np.zeros(current.size, dtype=complex)
            coefficients[support] = vector
            ket_site = routes.tensor(coefficients)

            def row_value(bra_index):
                bra_site = routes.basis[int(bra_index)]
                return (
                    h_plan.contract(bra_site, ket_site),
                    n_plan.contract(bra_site, ket_site),
                )

            records = self._map_tasks(executor, row_value, tuple(map(int, support)))
            h_out = np.asarray([record[0] for record in records], dtype=complex)
            n_out = np.asarray([record[1] for record in records], dtype=complex)
            cache["vector"] = np.array(vector, copy=True)
            cache["hamiltonian"] = h_out
            cache["metric"] = n_out
            matvec_counts["combined"] += 1
            return h_out, n_out

        def h_action(vector):
            return apply_both(vector)[0]

        def n_action(vector):
            return apply_both(vector)[1]

        diagnostics["matvec_counts"] = matvec_counts
        return (
            current,
            support,
            h_action,
            n_action,
            np.asarray(h_diag, dtype=float),
            np.asarray(n_diag, dtype=float),
            diagnostics,
        )

    def _frontier_wigner_eckart_matrix_free_problem(
        self,
        site,
        bond,
        *,
        support_rtol=1.0e-13,
        max_parameters=4096,
        max_dense_metric_parameters=4096,
        executor=None,
        sites=None,
        environment_sweeps=None,
    ):
        """Project a reusable reduced bond action onto one-site parameters.

        The one-site tangent map is embedded in the adjacent channel-resolved
        pair space. Contracting the environments once gives the exact local
        actions ``E† H_eff E`` and ``E† N_eff E`` without traversing the MPS for
        every bra parameter in every Davidson iteration.
        """
        site = int(site)
        bond = int(bond)
        if bond < 0 or bond >= self.nsites - 1 or site not in (bond, bond + 1):
            raise ValueError("one-site frontier bond must contain the optimized site.")
        current = self._pack_site(site)
        if current.size > int(max_parameters):
            raise MemoryError(
                f"local SU2LETTA support has {current.size} parameters; "
                "the matrix-free Wigner--Eckart solver is limited to "
                f"{int(max_parameters)}."
            )
        sites = self.materialize() if sites is None else list(sites)
        merged, layout = self._pair_coordinate_space(bond, sites)
        matrix_free_metric = bool(
            current.size > int(max_dense_metric_parameters)
        )
        _embedded_current, embedding = self._local_pair_embedding(
            site, sites, bond, layout, matrix_free=matrix_free_metric
        )
        if embedding.shape[1] != current.size:
            raise RuntimeError("one-site frontier embedding has the wrong size.")
        h_pair, n_pair, pair_diagnostics = self._reduced_pair_transition_actions(
            bond,
            sites,
            merged,
            layout,
            environment_sweeps=environment_sweeps,
            materialize_metric_blocks=not matrix_free_metric,
            compact_metric_blocks=True,
        )

        metric_blocks = getattr(n_pair, "metric_blocks", None)
        if matrix_free_metric:
            parent_n_diagonal = np.asarray(
                getattr(n_pair, "diag", np.ones(layout.size)), dtype=float
            )
            n_diagonal = (
                embedding.projected_diagonal(parent_n_diagonal)
                if hasattr(embedding, "projected_diagonal")
                else np.real(np.sum(
                    embedding.conj()
                    * (parent_n_diagonal[:, None] * embedding),
                    axis=0,
                ))
            )
            projected_metric_full = None
            metric_backend = "matrix_free_factorized_routes"
        elif metric_blocks is None:
            dense_embedding = np.asarray(embedding)
            n_columns = self._apply_packed_columns(
                n_pair, dense_embedding, executor=executor
            )
            dense_metric = dense_embedding.conj().T @ n_columns
            dense_metric = 0.5 * (dense_metric + dense_metric.conj().T)
            projected_metric_full = _ReducedMetricComponents(
                current.size,
                ((np.arange(current.size, dtype=np.int64), dense_metric),),
            )
            metric_backend = "operator_columns"
        else:
            projected_metric_full = self._project_metric_components(
                embedding,
                metric_blocks,
                site=site,
                bond=bond,
                factorized=bool(getattr(n_pair, "compact_metric_blocks", False)),
            )
            metric_backend = "connected_reduced_blocks"
        if projected_metric_full is not None:
            n_diagonal = projected_metric_full.diagonal
        scale = max(float(np.max(np.abs(n_diagonal), initial=0.0)), 1.0)
        support = np.flatnonzero(n_diagonal > float(support_rtol) * scale)
        if support.size == 0:
            raise np.linalg.LinAlgError("local SU2LETTA parameter support is null.")
        design = (
            embedding.restrict(support)
            if hasattr(embedding, "restrict")
            else np.asarray(embedding[:, support])
        )
        projected_metric = (
            None
            if projected_metric_full is None
            else projected_metric_full.restrict(support)
        )
        parent_h_diagonal = np.asarray(
            getattr(h_pair, "diag", np.zeros(layout.size)), dtype=float
        )
        h_diagonal = (
            design.projected_diagonal(parent_h_diagonal)
            if hasattr(design, "projected_diagonal")
            else np.real(
                np.sum(
                    design.conj()
                    * (parent_h_diagonal[:, None] * design),
                    axis=0,
                )
            )
        )
        n_diagonal = n_diagonal[support]
        matvec_counts = {"hamiltonian": 0, "metric": 0}

        def h_action(vector):
            vector = np.asarray(vector, dtype=complex).reshape(-1)
            if vector.size != support.size:
                raise ValueError("frontier Wigner--Eckart vector has the wrong size.")
            matvec_counts["hamiltonian"] += 1
            if hasattr(design, "matvec") and hasattr(design, "rmatvec"):
                return design.rmatvec(h_pair(design.matvec(vector)))
            return design.conj().T @ h_pair(design @ vector)

        def n_action(vector):
            vector = np.asarray(vector, dtype=complex).reshape(-1)
            if vector.size != support.size:
                raise ValueError("frontier Wigner--Eckart vector has the wrong size.")
            matvec_counts["metric"] += 1
            if hasattr(design, "matvec") and hasattr(design, "rmatvec"):
                return design.rmatvec(n_pair(design.matvec(vector)))
            return design.conj().T @ n_pair(design @ vector)

        if projected_metric is None:
            projected_metric = _MatrixFreeProjectedMetric(
                support.size,
                n_diagonal,
                n_action,
            )

        routes = self._wigner_eckart_route_plan(site)
        diagnostics = dict(pair_diagnostics)
        diagnostics.update(
            {
                "route_backend": routes.backend,
                "route_bytes": int(routes.nbytes),
                "route_count": int(routes.route_count),
                "local_action_backend": "frontier_projected_pair",
                "frontier_dimension": int(layout.size),
                "embedding_bytes": int(embedding.nbytes),
                "embedding_representation": (
                    "matrix_free_frontier_scatter"
                    if isinstance(embedding, _MatrixFreePairEmbedding)
                    else "block_scatter"
                    if isinstance(embedding, _StructuredPairEmbedding)
                    else "dense"
                ),
                "matvec_counts": matvec_counts,
                "projected_metric_backend": metric_backend,
                "_projected_metric": projected_metric,
                "_projected_design": design,
                "_parent_h_diagonal": parent_h_diagonal,
                "_parent_h_action": h_pair,
                "parent_action_type": type(
                    getattr(h_pair, "compiled_factorized_terms", None)
                ).__name__,
                "parent_factor_route_key": getattr(
                    h_pair, "factor_route_key", None
                ),
            }
        )
        return (
            current,
            support,
            h_action,
            n_action,
            np.asarray(h_diagonal, dtype=float),
            np.asarray(n_diagonal, dtype=float),
            diagnostics,
        )

    def _retract_reduced_pair(
        self,
        bond,
        target,
        layout,
        metric_action,
        *,
        cutoff=1.0e-10,
        maxiter=8,
        max_parameters=4096,
    ):
        """Retract a reduced two-site target into the fixed-D tied factors."""
        bond = int(bond)
        target = np.asarray(target, dtype=complex).reshape(-1)
        target_metric = np.asarray(metric_action(target))
        target_norm_squared = float(np.real(np.vdot(target, target_metric)))
        if target_norm_squared <= 0.0:
            raise np.linalg.LinAlgError(
                "Cannot retract a reduced pair with null metric norm."
            )

        def metric_error(vector):
            residual = np.asarray(vector) - target
            squared = float(
                np.real(np.vdot(residual, np.asarray(metric_action(residual))))
            )
            if squared < -float(cutoff) * max(target_norm_squared, 1.0):
                raise np.linalg.LinAlgError(
                    "Reduced pair metric produced a negative retraction norm."
                )
            return float(np.sqrt(max(squared, 0.0) / target_norm_squared))

        original = (
            np.array(self._pack_site(bond), copy=True),
            np.array(self._pack_site(bond + 1), copy=True),
        )
        best = tuple(np.array(vector, copy=True) for vector in original)
        initial_sites = self.materialize()
        if isinstance(layout, _ChannelResolvedPairSpace):
            initial_retracted = layout.pack_sites(
                initial_sites[bond], initial_sites[bond + 1]
            )
        else:
            initial_pair = merge_mps_sites(
                initial_sites[bond], initial_sites[bond + 1]
            )
            initial_retracted, _ = pack_two_site_state(initial_pair, layout=layout)
        best_error = metric_error(initial_retracted)
        best_coefficient_error = float(
            np.linalg.norm(initial_retracted - target)
            / max(float(np.linalg.norm(target)), np.finfo(float).tiny)
        )
        iterations = 0
        support_sizes = [0, 0]

        try:
            for iteration in range(max(1, int(maxiter))):
                for offset, site in enumerate((bond, bond + 1)):
                    sites = self.materialize()
                    current, embedding = self._local_pair_embedding(
                        site, sites, bond, layout, matrix_free=True
                    )
                    if current.size > int(max_parameters):
                        raise MemoryError(
                            "two-site SU2LETTA retraction has "
                            f"{current.size} parameters on site {site}; limit is "
                            f"{int(max_parameters)}."
                        )
                    column_norms = embedding.projected_diagonal(np.ones(layout.size))
                    scale = max(float(np.max(column_norms, initial=0.0)), 1.0)
                    support = np.flatnonzero(column_norms > float(cutoff) * scale)
                    if support.size == 0:
                        continue
                    from scipy.sparse.linalg import LinearOperator, lsmr
                    design = embedding.restrict(support)
                    operator = LinearOperator(design.shape, matvec=design.matvec,
                                              rmatvec=design.rmatvec, dtype=complex)
                    coefficients = lsmr(
                        operator, target, atol=float(cutoff), btol=float(cutoff),
                        maxiter=min(200, support.size),
                        x0=np.asarray(current[support], dtype=complex),
                    )[0]
                    candidate = np.zeros(
                        current.size,
                        dtype=np.result_type(current.dtype, coefficients.dtype),
                    )
                    candidate[support] = coefficients
                    if np.linalg.norm(candidate) <= float(cutoff):
                        continue
                    self._set_site_vector(site, candidate)
                    support_sizes[offset] = int(support.size)

                sites = self.materialize()
                if isinstance(layout, _ChannelResolvedPairSpace):
                    retracted = layout.pack_sites(sites[bond], sites[bond + 1])
                else:
                    merged = merge_mps_sites(sites[bond], sites[bond + 1])
                    retracted, _ = pack_two_site_state(merged, layout=layout)
                error = metric_error(retracted)
                coefficient_error = float(
                    np.linalg.norm(retracted - target)
                    / max(float(np.linalg.norm(target)), np.finfo(float).tiny)
                )
                iterations = iteration + 1
                if error < best_error:
                    best_error = error
                    best_coefficient_error = coefficient_error
                    best = (
                        np.array(self._pack_site(bond), copy=True),
                        np.array(self._pack_site(bond + 1), copy=True),
                    )
                if error <= float(cutoff):
                    break
                if iteration > 0 and abs(previous_error - error) <= float(cutoff) * max(
                    1.0, error
                ):
                    break
                previous_error = error
        except Exception:
            self._set_site_vector(bond, original[0])
            self._set_site_vector(bond + 1, original[1])
            raise

        self._set_site_vector(bond, best[0])
        self._set_site_vector(bond + 1, best[1])
        return {
            "truncation_error": float(best_error),
            "discarded_weight": float(best_error**2),
            "coefficient_retraction_error": float(best_coefficient_error),
            "iterations": int(iterations),
            "support": tuple(support_sizes),
            "fixed_reduced_bond_dim": int(self.D),
        }

    def _channel_resolved_pair_action(
        self,
        bond,
        sites,
        factors,
        space,
        *,
        transition_weights=None,
    ):
        """Compile the exact reduced two-site action with an explicit mid sector."""
        chain = BlockSparseEnvironmentChain.build(sites, factors)
        left_env = chain.left_envs[bond]
        right_env = chain.right_envs[bond + 1]
        w1 = chain.mpo_factors[bond]
        w2 = chain.mpo_factors[bond + 1]
        transitions = [[] for _ in space.entries]
        left_cache = {}
        right_cache = {}

        for in_index, in_entry in enumerate(space.entries):
            q_lk, q_p1k, q_p2k, q_rk, q_mk = in_entry.key
            for out_index, out_entry in enumerate(space.entries):
                q_lb, q_p1b, q_p2b, q_rb, q_mb = out_entry.key
                e_blocks = left_env.get((q_lb, q_lk))
                f_blocks = right_env.get((q_rb, q_rk))
                if e_blocks is None or f_blocks is None:
                    continue
                if not chain.rank_coupled:
                    left_block = w1.block(q_p1b, q_p1k)
                    right_block = w2.block(q_p2b, q_p2k)
                    if left_block is None or right_block is None:
                        continue
                    transitions[in_index].append(
                        (
                            out_index,
                            _factorize_left_two_site_dense_term(
                                np.asarray(e_blocks), np.asarray(left_block)
                            ),
                            _factorize_right_two_site_dense_term(
                                np.asarray(right_block), np.asarray(f_blocks)
                            ),
                        )
                    )
                    continue
                left_key = (q_lb, q_lk, q_p1b, q_p1k, q_mb, q_mk)
                left_reduced = left_cache.get(left_key)
                if left_reduced is None:
                    left_reduced = _left_reduced_rank_coupled_block(
                        w1, *left_key
                    ) or {}
                    left_cache[left_key] = left_reduced
                if not left_reduced:
                    continue
                right_key = (q_mb, q_mk, q_p2b, q_p2k, q_rb, q_rk)
                right_reduced = right_cache.get(right_key)
                if right_reduced is None:
                    right_reduced = _right_reduced_rank_coupled_block(
                        w2, *right_key
                    ) or {}
                    right_cache[right_key] = right_reduced
                if not right_reduced:
                    continue

                right_by_middle = {}
                for (middle_index, right_index), block in right_reduced.items():
                    right_env_block = (
                        f_blocks.get(right_index)
                        if hasattr(f_blocks, "get")
                        else (
                            f_blocks[right_index]
                            if right_index < len(f_blocks)
                            else None
                        )
                    )
                    if right_env_block is not None:
                        right_by_middle.setdefault(int(middle_index), []).append(
                            (np.asarray(right_env_block), np.asarray(block))
                        )
                for (left_index, middle_index), left_block in left_reduced.items():
                    left_env_block = (
                        e_blocks.get(left_index)
                        if hasattr(e_blocks, "get")
                        else (
                            e_blocks[left_index]
                            if left_index < len(e_blocks)
                            else None
                        )
                    )
                    if left_env_block is None:
                        continue
                    left_factor = _factorize_left_two_site_dense_term(
                        np.asarray(left_env_block),
                        np.asarray(left_block),
                    )
                    for right_env_block, right_block in right_by_middle.get(
                        int(middle_index), ()
                    ):
                        right_factor = _factorize_right_two_site_dense_term(
                            right_block,
                            right_env_block,
                        )
                        transitions[in_index].append(
                            (out_index, left_factor, right_factor)
                        )

        dtype = np.result_type(
            *(np.asarray(block).dtype for core in factors for block in getattr(core, "data", {}).values()),
            *(np.asarray(block).dtype for site in sites for block in site.data.values()),
        )

        def action(vector):
            in_blocks = space.blocks(vector)
            weight_dtype = (
                float
                if not transition_weights
                else np.result_type(
                    *(np.asarray(value).dtype for value in transition_weights.values())
                )
            )
            out_blocks = [
                np.zeros(
                    entry.shape,
                    dtype=np.result_type(dtype, np.asarray(vector).dtype, weight_dtype),
                )
                for entry in space.entries
            ]
            for in_index, block in enumerate(in_blocks):
                if not np.any(block):
                    continue
                for out_index, left_factor, right_factor in transitions[in_index]:
                    contribution = _apply_two_site_dense_factorized(
                        left_factor,
                        right_factor,
                        block,
                    )
                    if transition_weights is not None:
                        contribution = (
                            transition_weights.get((in_index, out_index), 0.0)
                            * contribution
                        )
                    out_blocks[out_index] += contribution
            return space.pack_blocks(out_blocks)

        action.backend = (
            "channel-resolved-rank-coupled-factorized"
            if chain.rank_coupled
            else "channel-resolved-block-factorized"
        )
        action.transition_count = int(sum(len(items) for items in transitions))
        action.entry_transitions = tuple(
            sorted(
                {
                    (in_index, int(item[0]))
                    for in_index, items in enumerate(transitions)
                    for item in items
                }
            )
        )
        return action

    def _exact_reduced_pair_matrices(
        self,
        bond,
        sites,
        merged,
        layout,
        *,
        environment_sweeps=None,
    ):
        """Build an exact pair pencil from channel-resolved pair actions."""
        h_action, n_action, _diagnostics = self._reduced_pair_transition_actions(
            bond,
            sites,
            merged,
            layout,
            environment_sweeps=environment_sweeps,
        )
        dimension = int(getattr(layout, "size", sum(entry.size for entry in layout)))

        def column(index):
            vector = np.zeros(dimension, dtype=complex)
            vector[int(index)] = 1.0
            return int(index), h_action(vector), n_action(vector)

        # Each column action already parallelizes over output rows.  Keeping
        # the outer loop serial avoids nested submission to the shared pool.
        records = [column(index) for index in range(dimension)]
        hamiltonian = np.zeros((dimension, dimension), dtype=complex)
        metric = np.zeros((dimension, dimension), dtype=complex)
        for index, h_column, n_column in records:
            hamiltonian[:, index] = h_column
            metric[:, index] = n_column
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
        metric = 0.5 * (metric + metric.conj().T)
        return hamiltonian, metric

    def _pair_hamiltonian_factors(self):
        """Reuse immutable reduced terms; environments remain state-dependent."""
        owner = getattr(self.mpo[0], "normal_complementary_owner", None)
        if owner is None:
            return self.mpo
        revision = owner.system_stats["revision"]
        cached = self._pair_hamiltonian_cache
        if cached is None or cached[0] is not self.mpo or cached[1] != revision:
            from pyqed.qchem.dmrg.backends.reduced import build_su2_normal_complementary_mpo
            factors = tuple(build_su2_normal_complementary_mpo(
                owner,
                fully_reduced=bool(getattr(self.mpo[0], "normal_complementary_fully_reduced", False)),
                materialize_reduced_terms=True))
            for factor in factors:
                object.__setattr__(factor, "normal_complementary_right_dual",
                                   bool(getattr(self.mpo[0], "normal_complementary_right_dual", False)))
            cached = (self.mpo, revision, factors)
            self._pair_hamiltonian_cache = cached
        return cached[2]

    def _reduced_pair_transition_actions(
        self,
        bond,
        sites,
        merged,
        layout,
        *,
        environment_sweeps=None,
        materialize_metric_blocks=True,
        compact_metric_blocks=False,
    ):
        """Return exact matrix-free actions with explicit intermediate channels.

        A merged rank-4 reduced tensor does not itself label the intermediate
        MPS fusion channel.  Splitting each vector into adjacent rank-3 tensors
        before contraction retains that channel and makes the action exact for
        reduced bond dimensions greater than one.
        """
        dimension = int(getattr(layout, "size", sum(entry.size for entry in layout)))
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, self.mpo)
        h_plan = n_plan = None
        if not isinstance(layout, _ChannelResolvedPairSpace):
            h_plan = AdjacentPairTransitionPlan.build(sites, self.mpo, bond)
            n_plan = AdjacentPairTransitionPlan.build(sites, identity, bond)
        if isinstance(layout, _ChannelResolvedPairSpace):
            def compiled_action(factors, sweep=None):
                if sweep is None:
                    if factors is self.mpo:
                        factors = self._pair_hamiltonian_factors()
                    chain = BlockSparseEnvironmentChain.build(
                        sites, factors, target_bond=bond)
                    operator = chain.bond_operator(bond, merged)
                else:
                    operator = sweep.bond_operator(bond, merged)
                packed_apply = (
                    operator.aux_packed_matvec or operator.packed_matvec
                )
                if packed_apply is None:
                    raise RuntimeError(
                        "Current non-Abelian bond operator has no packed action."
                    )
                operator_entries = {
                    entry.key: (index, entry)
                    for index, entry in enumerate(operator.basis.entries)
                }
                entry_map = []
                for own_index, own_entry in enumerate(layout.entries):
                    q_l, q_p1, q_p2, q_r, q_mid = own_entry.key
                    operator_index, operator_entry = operator_entries[
                        (q_l, q_p1, q_mid, q_p2, q_r)
                    ]
                    if tuple(operator_entry.shape) != tuple(own_entry.shape):
                        raise ValueError(
                            "Compiled SU(2) pair basis shape does not match LETTA."
                        )
                    entry_map.append(
                        (own_index, own_entry, operator_index, operator_entry)
                    )

                def action(vector):
                    vector = np.asarray(vector)
                    packed = np.zeros(
                        dimension,
                        dtype=np.result_type(vector.dtype, complex),
                    )
                    for _own_index, own_entry, _op_index, op_entry in entry_map:
                        packed[op_entry.offset : op_entry.offset + op_entry.size] = (
                            vector[own_entry.offset : own_entry.offset + own_entry.size]
                        )
                    applied = np.asarray(packed_apply(packed))
                    out = np.zeros(
                        dimension,
                        dtype=np.result_type(applied.dtype, vector.dtype),
                    )
                    for _own_index, own_entry, _op_index, op_entry in entry_map:
                        out[own_entry.offset : own_entry.offset + own_entry.size] = (
                            applied[op_entry.offset : op_entry.offset + op_entry.size]
                        )
                    return out

                compiled = getattr(packed_apply, "block_matrices", None)
                diagonal = operator.diag
                if diagonal is None and callable(
                    getattr(compiled, "diagonal", None)
                ):
                    diagonal = compiled.diagonal()
                if diagonal is None:
                    diagonal = np.empty(dimension, dtype=complex)
                    unit = np.zeros(dimension, dtype=complex)
                    for index in range(dimension):
                        unit[index] = 1.0
                        diagonal[index] = packed_apply(unit)[index]
                        unit[index] = 0.0
                diagonal = np.asarray(diagonal)
                action.diag = np.empty(dimension, dtype=float)
                for _own_index, own_entry, _op_index, op_entry in entry_map:
                    action.diag[own_entry.offset : own_entry.offset + own_entry.size] = (
                        diagonal[op_entry.offset : op_entry.offset + op_entry.size]
                    )
                compiled_indices = getattr(compiled, "in_indices", None)
                action.transition_count = int(
                    len(compiled_indices)
                    if compiled_indices is not None
                    else getattr(
                        compiled,
                        "su2_qchem_factor_match_count",
                        getattr(compiled, "route_count", 0),
                    )
                )
                action.backend = str(
                    getattr(packed_apply, "backend", "compiled-packed")
                )
                action.compiled_factorized_terms = compiled
                action.factor_route_key = getattr(
                    compiled,
                    "_cpp_factor_route_key",
                    getattr(compiled, "factor_route_key", None),
                )
                action.su2_moving_environment = getattr(
                    compiled,
                    "su2_moving_environment",
                    getattr(compiled, "owner", None),
                )

                def operator_embedding(columns):
                    columns = np.asarray(columns)
                    out = np.zeros_like(columns)
                    for _own_index, own_entry, _op_index, op_entry in entry_map:
                        out[
                            op_entry.offset : op_entry.offset + op_entry.size
                        ] = columns[
                            own_entry.offset : own_entry.offset + own_entry.size
                        ]
                    return np.ascontiguousarray(out)

                action.operator_embedding = operator_embedding

                def operator_projection_blocks(columns):
                    blocks = []
                    for _own_index, own_entry, _op_index, op_entry in entry_map:
                        if isinstance(columns, _StructuredPairEmbedding):
                            active, local = columns.block_design(
                                own_entry.offset, own_entry.size
                            )
                        else:
                            dense_columns = np.asarray(columns)
                            local = dense_columns[
                                own_entry.offset : own_entry.offset + own_entry.size
                            ]
                            active = np.flatnonzero(
                                np.any(local != 0.0, axis=0)
                            )
                        if active.size == 0:
                            continue
                        blocks.append(
                            (
                                slice(
                                    int(op_entry.offset),
                                    int(op_entry.offset + op_entry.size),
                                ),
                                np.asarray(active, dtype=np.int64),
                                np.ascontiguousarray(
                                    local
                                    if isinstance(columns, _StructuredPairEmbedding)
                                    else local[:, active]
                                ),
                            )
                        )
                    return tuple(blocks)

                action.operator_projection_blocks = operator_projection_blocks
                return action, operator, tuple(entry_map), compiled

            h_sweep = n_sweep = None
            if environment_sweeps is not None:
                h_sweep, n_sweep = environment_sweeps
            h_action, h_operator, h_entry_map, h_compiled = compiled_action(
                self.mpo, h_sweep
            )
            n_action, n_operator, n_entry_map, n_compiled = compiled_action(
                identity, n_sweep
            )

            metric_routes = (
                None
                if n_compiled is None
                else n_compiled.factorized_metric_routes()
            )
            if metric_routes is None:
                raise RuntimeError(
                    "Compiled identity pair action does not expose reduced metric routes."
                )
            n_action.factorized_metric_routes = metric_routes
            n_action.metric_route_owner = getattr(
                n_compiled, "su2_moving_environment", None
            )
            own_by_operator_index = {
                operator_index: own_entry
                for _own_index, own_entry, operator_index, _operator_entry
                in n_entry_map
            }
            if not materialize_metric_blocks:
                packed_metric_diagonal = np.asarray(n_action.diag, dtype=float)

                def factorized_metric_action(vector):
                    vector = np.asarray(vector)
                    out = np.zeros(
                        dimension,
                        dtype=np.result_type(vector.dtype, complex),
                    )
                    for (
                        in_index,
                        out_index,
                        _in_entry,
                        _out_entry,
                        left,
                        right,
                    ) in metric_routes:
                        in_entry = own_by_operator_index[int(in_index)]
                        out_entry = own_by_operator_index[int(out_index)]
                        in_rows = int(in_entry.shape[0] * in_entry.shape[1])
                        in_cols = int(in_entry.shape[2] * in_entry.shape[3])
                        block = vector[
                            in_entry.offset : in_entry.offset + in_entry.size
                        ].reshape(in_rows, in_cols)
                        applied = np.asarray(left) @ block @ np.asarray(right).T
                        out[
                            out_entry.offset : out_entry.offset + out_entry.size
                        ] += applied.reshape(-1)
                    return out

                factorized_metric_action.diag = packed_metric_diagonal
                factorized_metric_action.transition_count = len(metric_routes)
                factorized_metric_action.backend = "factorized-metric-routes"
                factorized_metric_action.factorized_metric_routes = metric_routes
                factorized_metric_action.metric_route_owner = n_action.metric_route_owner
                factorized_metric_action.diagonal_factor_blocks = (
                    tuple((own_by_operator_index[int(in_index)], left, right)
                          for in_index, out_index, _, _, left, right in metric_routes)
                    if all(in_index == out_index for in_index, out_index, *_ in metric_routes)
                    and len({int(r[0]) for r in metric_routes}) == len(metric_routes)
                    else None
                )
                n_action = factorized_metric_action
            metric_blocks = ()
            if materialize_metric_blocks:
                metric_by_pair = {}
                for (
                    in_index,
                    out_index,
                    _in_entry,
                    _out_entry,
                    left,
                    right,
                ) in metric_routes:
                    in_entry = own_by_operator_index[int(in_index)]
                    out_entry = own_by_operator_index[int(out_index)]
                    key = (out_entry.key, in_entry.key)
                    if compact_metric_blocks:
                        metric_by_pair.setdefault(key, []).append(
                            (np.asarray(left), np.asarray(right))
                        )
                        continue
                    block = metric_by_pair.get(key)
                    if block is None:
                        block = np.zeros(
                            (out_entry.size, in_entry.size),
                            dtype=np.result_type(left, right, complex),
                        )
                        metric_by_pair[key] = block
                    block += np.kron(np.asarray(left), np.asarray(right))
                entry_by_key = {entry.key: entry for entry in layout.entries}
                metric_blocks = tuple(
                    (
                        entry_by_key[out_key],
                        entry_by_key[in_key],
                        block,
                    )
                    for (out_key, in_key), block in metric_by_pair.items()
                    if compact_metric_blocks or np.linalg.norm(block) > 1.0e-14
                )
                n_action.metric_blocks = metric_blocks
                n_action.compact_metric_blocks = bool(compact_metric_blocks)
                n_action.transition_count = len(metric_blocks)

            diagnostics = {
                "backend": "compiled_contextual_channel_resolved",
                "dimension": dimension,
                "hamiltonian_direction": (
                    "full"
                    if h_sweep is None
                    else getattr(h_sweep, "direction", "full")
                ),
                "metric_direction": (
                    "full"
                    if n_sweep is None
                    else getattr(n_sweep, "direction", "full")
                ),
                "hamiltonian_cached_sites": (
                    self.nsites - 2
                ),
                "metric_cached_sites": (
                    self.nsites - 2
                ),
                "hamiltonian_transitions": h_action.transition_count,
                "metric_transitions": n_action.transition_count,
                "metric_blocks": len(metric_blocks),
                "metric_nbytes": int(sum(
                    sum(left.nbytes + right.nbytes for left, right in block)
                    if compact_metric_blocks else block.nbytes
                    for _, _, block in metric_blocks
                )),
                "metric_blocks_materialized": bool(
                    materialize_metric_blocks and not compact_metric_blocks
                ),
                "metric_blocks_factorized": bool(
                    materialize_metric_blocks and compact_metric_blocks
                ),
                "environment_reused": environment_sweeps is not None,
                "hamiltonian_backend": h_action.backend,
                "metric_backend": n_action.backend,
                "metric_route_owner": (
                    "compiled_reduced_owner"
                    if n_action.metric_route_owner is not None
                    else "python_factorized_routes"
                ),
            }
            return h_action, n_action, diagnostics

        def split_vector(vector):
            tensor = unpack_two_site_state(vector, merged, layout=layout)
            left, right, *_ = svd_two_site(
                tensor,
                max_bond=None,
                cutoff=0.0,
                absorb="right",
                max_bond_mode="reduced",
            )
            return left, right

        def split_basis(index):
            vector = np.zeros(dimension, dtype=complex)
            vector[int(index)] = 1.0
            return split_vector(vector)

        basis_pairs = tuple(self._map_tasks(
            self._solver_executor,
            split_basis,
            tuple(range(dimension)),
        ))
        cache = {"vector": None, "pair": None}

        def ket_pair(vector):
            vector = np.asarray(vector, dtype=complex).reshape(-1)
            if vector.size != dimension:
                raise ValueError(
                    "Reduced pair action vector does not match the packed pair basis."
                )
            cached = cache["vector"]
            if cached is None or not np.array_equal(vector, cached):
                cache["vector"] = np.array(vector, copy=True)
                cache["pair"] = split_vector(vector)
            return cache["pair"]

        def build_action(plan):
            def action(vector):
                ket_left, ket_right = ket_pair(vector)

                def row_value(bra_pair):
                    bra_left, bra_right = bra_pair
                    return plan.contract(
                        bra_left,
                        bra_right,
                        ket_left,
                        ket_right,
                    )

                return np.asarray(
                    self._map_tasks(
                        self._solver_executor,
                        row_value,
                        basis_pairs,
                    ),
                    dtype=complex,
                )

            return action

        diagnostics = {
            "backend": "channel_resolved_pair_transitions",
            "dimension": dimension,
            "hamiltonian_direction": h_plan.direction,
            "metric_direction": n_plan.direction,
            "hamiltonian_cached_sites": h_plan.cached_sites,
            "metric_cached_sites": n_plan.cached_sites,
        }
        return build_action(h_plan), build_action(n_plan), diagnostics

    def _variationally_relax_reduced_pair(
        self,
        bond,
        layout,
        h_action,
        n_action,
        *,
        cutoff=1.0e-10,
        maxiter=2,
        max_parameters=4096,
        davidson_tol=1.0e-10,
        davidson_maxiter=80,
        davidson_max_space=32,
    ):
        """Minimize the cached pair Rayleigh quotient over both tied factors."""
        bond = int(bond)

        def pair_energy():
            sites = self.materialize()
            if isinstance(layout, _ChannelResolvedPairSpace):
                vector = layout.pack_sites(sites[bond], sites[bond + 1])
            else:
                merged = merge_mps_sites(sites[bond], sites[bond + 1])
                vector, _ = pack_two_site_state(merged, layout=layout)
            metric_vector = np.asarray(n_action(vector))
            norm = float(np.real(np.vdot(vector, metric_vector)))
            if norm <= 0.0:
                return np.inf
            return float(np.real(np.vdot(vector, h_action(vector))) / norm)

        energy = pair_energy()
        iterations = 0
        max_residual = 0.0
        for iteration in range(max(0, int(maxiter))):
            cycle_before = energy
            for site in (bond, bond + 1):
                sites = self.materialize()
                current, embedding = self._local_pair_embedding(
                    site, sites, bond, layout, matrix_free=True
                )
                if current.size > int(max_parameters):
                    raise MemoryError(
                        "variational pair retraction has "
                        f"{current.size} parameters on site {site}; limit is "
                        f"{int(max_parameters)}."
                    )
                column_norms = embedding.projected_diagonal(np.ones(layout.size))
                scale = max(float(np.max(column_norms, initial=0.0)), 1.0)
                support = np.flatnonzero(column_norms > float(cutoff) * scale)
                if support.size == 0:
                    continue
                design = embedding.restrict(support)
                # Diagonal-only projections are preconditioners, not claims
                # about the full factor metric; actions below retain all terms.
                n_diagonal = embedding.projected_diagonal(n_action.diag)[support]
                metric_scale = max(
                    float(np.max(np.abs(n_diagonal), initial=0.0)), 1.0
                )
                active = np.flatnonzero(
                    n_diagonal > float(cutoff) * metric_scale
                )
                if active.size == 0:
                    continue
                support = support[active]
                design = embedding.restrict(support)
                n_diagonal = n_diagonal[active]
                h_diagonal = embedding.projected_diagonal(h_action.diag)[support]
                initial = np.asarray(current[support], dtype=complex)

                def local_h(vector):
                    return design.rmatvec(h_action(design.matvec(vector)))

                def local_n(vector):
                    return design.rmatvec(n_action(design.matvec(vector)))

                initial_metric_norm = float(
                    np.real(np.vdot(initial, local_n(initial)))
                )
                if initial_metric_norm <= float(cutoff) * metric_scale:
                    initial = np.zeros(support.size, dtype=complex)
                    initial[int(np.argmin(h_diagonal / n_diagonal))] = 1.0

                local_energy, reduced, info = _solve_packed_generalized_davidson(
                    initial,
                    local_h,
                    h_diag=h_diagonal,
                    N=local_n,
                    n_diag=n_diagonal,
                    tol=float(davidson_tol),
                    tol_residual=float(davidson_tol),
                    itermax=int(davidson_maxiter),
                    max_space=min(int(davidson_max_space), int(support.size)),
                )
                max_residual = max(max_residual, float(info.get("residual", 0.0)))
                if not bool(info.get("davidson_converged", False)):
                    continue
                candidate = np.zeros(
                    current.size,
                    dtype=np.result_type(current.dtype, reduced.dtype, complex),
                )
                candidate[support] = reduced
                overlap = np.vdot(initial, local_n(reduced))
                if abs(overlap) > 0.0:
                    candidate *= np.conj(overlap) / abs(overlap)
                self._set_site_vector(site, candidate)
                proposed = pair_energy()
                if proposed <= energy + 1.0e-10:
                    energy = float(proposed)
                else:
                    self._set_site_vector(site, current)
            iterations = iteration + 1
            if abs(cycle_before - energy) <= float(cutoff) * max(1.0, abs(energy)):
                break
        return {
            "energy": float(energy),
            "iterations": int(iterations),
            "max_residual": float(max_residual),
        }

    def optimize_two_sites(
        self,
        bond,
        *,
        cutoff=1.0e-10,
        retraction_maxiter=8,
        retraction_relax_sweeps=2,
        max_retraction_parameters=4096,
        accept_tol=1.0e-10,
        davidson_tol=1.0e-10,
        davidson_maxiter=80,
        davidson_max_space=32,
        dense_dim=0,
        sites=None,
        environment_sweeps=None,
    ):
        """Optimize one adjacent pair in the native reduced SU(2) space.

        The exact rank-coupled pair eigenproblem is solved without magnetic
        expansion.  Its root is then variationally retracted to the current
        tied fixed-``D`` manifold; ``truncation_error`` reports the relative
        error in the reduced pair metric, excluding gauge and null directions.

        Factorized spectral whitening uses small boundary eigensystems when
        the metric consists of diagonal channel products; other connectivity
        uses generalized Davidson. Both paths have a bounded 128 MiB Krylov
        workspace and never construct the full norm whitener. Fixed-D
        retraction uses bounded LSMR coefficient fits, selected by pair-metric
        error, followed by variational factor relaxation. This is an adaptation
        of Davidson (1975), doi:10.1016/0021-9991(75)90065-0, and Fong & Saunders
        (2011), doi:10.1137/10079687X; it is not an exact metric-optimal split.
        Iteration limits and directional metric filtering can leave incomplete
        local convergence; returned residuals and global energy checks matter.
        """
        bond = int(bond)
        if bond < 0 or bond >= self.nsites - 1:
            raise IndexError(f"bond {bond} is outside a chain of length {self.nsites}.")
        started = time.perf_counter()
        global_before = float(self.expectation())
        sites = self.materialize() if sites is None else list(sites)
        original = (
            np.array(self._pack_site(bond), copy=True),
            np.array(self._pack_site(bond + 1), copy=True),
        )
        merged = merge_mps_sites(sites[bond], sites[bond + 1])
        layout = _ChannelResolvedPairSpace(sites[bond], sites[bond + 1])
        initial = layout.pack_sites(sites[bond], sites[bond + 1])
        pair_matrix_free = initial.size > int(dense_dim)
        if pair_matrix_free:
            h_action, n_action, pair_diagnostics = (
                self._reduced_pair_transition_actions(
                    bond,
                    sites,
                    merged,
                    layout,
                    environment_sweeps=environment_sweeps,
                    materialize_metric_blocks=False,
                )
            )
            hamiltonian = metric = None
        else:
            hamiltonian, metric = self._exact_reduced_pair_matrices(
                bond,
                sites,
                merged,
                layout,
                environment_sweeps=environment_sweeps,
            )

            def h_action(vector):
                return hamiltonian @ vector

            def n_action(vector):
                return metric @ vector

            h_action.diag = np.diag(hamiltonian).real
            n_action.diag = np.diag(metric).real
            pair_diagnostics = {
                "backend": "channel_resolved_pair_transitions",
                "dimension": int(initial.size),
                "materialized": True,
            }

        initial_h = np.asarray(h_action(initial))
        initial_n = np.asarray(n_action(initial))
        initial_norm = float(np.real(np.vdot(initial, initial_n)))
        if initial_norm <= 0.0:
            before = self.expectation()
            return {
                "bond": bond,
                "sites": (bond, bond + 1),
                "energy_before": float(before),
                "energy_after": float(before),
                "local_energy": float(before),
                "accepted": False,
                "native_su2": True,
                "fully_wigner_eckart_reduced": True,
                "environment_backend": "channel_resolved_pair_transitions",
                "solver": "two_site_wigner_eckart",
                "matrix_free": bool(pair_matrix_free),
                "local_linear_algebra": (
                    "matrix_free_generalized_davidson"
                    if pair_matrix_free
                    else "dense_metric_filtered_eigh"
                ),
                "local_residual": float("inf"),
                "truncation_error": 0.0,
                "discarded_weight": 0.0,
                "coefficient_retraction_error": 0.0,
                "retraction_iterations": 0,
                "retraction_relax_sweeps": 0,
                "retraction_residual": float("inf"),
                "retraction_support": (0, 0),
                "fixed_reduced_bond_dim": int(self.D),
                "solver_info": {},
                "elapsed": float(time.perf_counter() - started),
                "message": "rejected null reduced pair metric",
            }
        before = float(np.real(np.vdot(initial, initial_h)) / initial_norm + self.ecore)
        if pair_matrix_free:
            factor_blocks = getattr(n_action, 'diagonal_factor_blocks', None)
            whitening = (_FactorizedPairWhitener(initial.size, factor_blocks, cutoff)
                         if factor_blocks is not None else None)
            shift = before - self.ecore
            if whitening is not None:
                def solve_h(vector):
                    physical = whitening.matvec(vector)
                    return whitening.rmatvec(h_action(physical)) - shift * vector
                solve_initial = whitening.rmatvec(initial_n)
                solve_n, solve_n_diag = None, None
                solve_h_diag = np.zeros(whitening.rank)
            else:
                solve_h = lambda vector: h_action(vector) - shift*n_action(vector)
                solve_initial = initial / np.sqrt(initial_norm)
                solve_n, solve_n_diag = n_action, n_action.diag
                solve_h_diag = h_action.diag - shift*n_action.diag
            local_energy, solved, solver_info = _solve_packed_generalized_davidson(
                solve_initial,
                solve_h,
                h_diag=solve_h_diag,
                N=solve_n,
                n_diag=solve_n_diag,
                tol=float(davidson_tol),
                tol_residual=float(davidson_tol),
                itermax=int(davidson_maxiter),
                max_space=min(int(davidson_max_space), int(initial.size)),
                workspace_bytes=128 * 1024**2,
                metric_rtol=float(cutoff),
            )
            target = whitening.matvec(solved) if whitening is not None else solved
            local_energy += shift
            solver_info["metric_whitening"] = 'factorized_spectral' if whitening is not None else None
            solver_info["energy_shift"] = float(shift)
            solver_info["solver"] = "channel_resolved_generalized_pair_davidson"
            solver_info["metric_rank"] = whitening.rank if whitening is not None else None
            solver_info["parent_dimension"] = int(initial.size)
        else:
            local_energy, target = _lowest_generalized_pair(
                hamiltonian,
                metric,
                rtol=cutoff,
            )
            solver_info = {
                "metric": 0.0,
                "residual": 0.0,
                "davidson_iterations": 0,
                "davidson_converged": True,
                "subspace_dim": int(initial.size),
                "solver": "channel_resolved_pair_eigh",
            }
        solver_info["pair_transition_plan"] = pair_diagnostics
        target_h = np.asarray(h_action(target))
        target_n = np.asarray(n_action(target))
        residual = float(
            np.linalg.norm(target_h - local_energy * target_n)
            / max(np.linalg.norm(target_h), abs(local_energy) * np.linalg.norm(target_n), 1.0)
        )
        verified_residual = float(np.linalg.norm(
            whitening.rmatvec(target_h - local_energy*target_n)
            if pair_matrix_free and whitening is not None
            else target_h - local_energy*target_n
        ))
        solver_info['verified_residual'] = verified_residual
        root_valid = bool(np.isfinite(verified_residual)
                          and verified_residual <= max(10*float(davidson_tol), 1e-10))
        if not root_valid:
            return dict(bond=bond, sites=(bond,bond+1), energy_before=global_before,
                energy_after=global_before, local_energy=float(local_energy+self.ecore),
                accepted=False, stationary=False, native_su2=True,
                fully_wigner_eckart_reduced=True, matrix_free=pair_matrix_free,
                solver='two_site_wigner_eckart', local_residual=verified_residual,
                truncation_error=0.0, discarded_weight=0.0,
                coefficient_retraction_error=0.0, retraction_iterations=0,
                retraction_relax_sweeps=0, retraction_residual=verified_residual,
                retraction_support=(0,0), fixed_reduced_bond_dim=int(self.D),
                solver_info={'davidson':solver_info}, elapsed=time.perf_counter()-started,
                message='Rejected unresolved pair root before retraction')
        retraction = self._retract_reduced_pair(
            bond,
            target,
            layout,
            n_action,
            cutoff=cutoff,
            maxiter=retraction_maxiter,
            max_parameters=max_retraction_parameters,
        )
        relaxation = self._variationally_relax_reduced_pair(
            bond,
            layout,
            h_action,
            n_action,
            cutoff=cutoff,
            maxiter=retraction_relax_sweeps,
            max_parameters=max_retraction_parameters,
            davidson_tol=davidson_tol,
            davidson_maxiter=davidson_maxiter,
            davidson_max_space=davidson_max_space,
        )
        retracted_sites = self.materialize()
        retracted = layout.pack_sites(
            retracted_sites[bond], retracted_sites[bond + 1]
        )
        retracted_n = np.asarray(n_action(retracted))
        retracted_norm = float(np.real(np.vdot(retracted, retracted_n)))
        if retracted_norm <= float(cutoff) * max(initial_norm, 1.0):
            after = np.inf
        else:
            after = float(
                np.real(np.vdot(retracted, h_action(retracted))) / retracted_norm
                + self.ecore
            )
        locally_accepted = bool(np.isfinite(after) and after <= before + float(accept_tol))
        global_after = float(self.expectation()) if locally_accepted else np.inf
        accepted = bool(locally_accepted and np.isfinite(global_after)
                        and global_after <= global_before + float(accept_tol))
        if not accepted:
            self._set_site_vector(bond, original[0])
            self._set_site_vector(bond + 1, original[1])
            global_after = global_before
        after = global_after
        return {
            "bond": bond,
            "sites": (bond, bond + 1),
            "energy_before": global_before,
            "energy_after": float(after),
            "local_energy_before": float(before),
            "global_variational_guard": True,
            "local_energy": float(local_energy + self.ecore),
            "accepted": bool(accepted),
            "native_su2": True,
            "fully_wigner_eckart_reduced": True,
            "environment_backend": "channel_resolved_pair_transitions",
            "solver": "two_site_wigner_eckart",
            "matrix_free": bool(pair_matrix_free),
            "local_linear_algebra": (
                "matrix_free_generalized_davidson"
                if pair_matrix_free
                else "dense_metric_filtered_eigh"
            ),
            "local_residual": residual,
            "truncation_error": retraction["truncation_error"],
            "discarded_weight": retraction["discarded_weight"],
            "coefficient_retraction_error": retraction[
                "coefficient_retraction_error"
            ],
            "retraction_iterations": retraction["iterations"],
            "retraction_relax_sweeps": relaxation["iterations"],
            "retraction_residual": relaxation["max_residual"],
            "retraction_support": retraction["support"],
            "fixed_reduced_bond_dim": retraction["fixed_reduced_bond_dim"],
            "solver_info": {"davidson": solver_info},
            "elapsed": float(time.perf_counter() - started),
        }

    def _optimize_site_matrix_free_metric(
        self,
        site,
        bond,
        *,
        current,
        support,
        current_reduced,
        h_action,
        projected_metric,
        h_diag,
        n_diag,
        solver_info,
        denominator,
        before,
        accept_tol,
        davidson_tol,
        davidson_maxiter,
        davidson_max_space,
        davidson_minimum_explored_dimension,
        davidson_convergence_confirmations,
        conditional_preconditioner,
        davidson_workspace_mib,
        metric_rtol,
        solver,
        requested_solver,
        started,
    ):
        """Solve a large tied problem without materializing its norm matrix."""
        guess = current_reduced / np.sqrt(float(denominator))
        krylov_key = (int(site), int(bond))
        cached = self._projected_krylov_cache.get(krylov_key)
        recycled = None
        if cached is not None and cached.shape[0] == current.size:
            recycled = cached[support]
        preconditioner = (_MetricPreconditioner(projected_metric.matvec, n_diag)
                          if conditional_preconditioner else None)
        local_energy, reduced, davidson = _solve_packed_generalized_davidson(
            guess,
            h_action,
            h_diag=h_diag,
            N=projected_metric.matvec,
            n_diag=n_diag,
            tol=float(davidson_tol),
            tol_residual=float(davidson_tol),
            itermax=int(davidson_maxiter),
            max_space=min(int(davidson_max_space), int(support.size)),
            initial_vectors=recycled,
            return_recycle_space=True,
            minimum_explored_dimension=davidson_minimum_explored_dimension,
            convergence_confirmations=davidson_convergence_confirmations,
            profile=True,
            precond=preconditioner,
            workspace_bytes=int(davidson_workspace_mib * 1024**2),
            metric_rtol=metric_rtol,
        )
        recycle_space = davidson.pop("_recycle_space")
        tied_recycle = np.zeros((current.size, recycle_space.shape[1]), dtype=complex)
        tied_recycle[support] = recycle_space
        self._projected_krylov_cache[krylov_key] = tied_recycle
        candidate_h = h_action(reduced)
        candidate_n = projected_metric.matvec(reduced)
        candidate_norm = float(np.real(np.vdot(reduced, candidate_n)))
        local_energy = float(np.real(np.vdot(reduced, candidate_h)) / candidate_norm)
        true_residual = float(np.linalg.norm(candidate_h - local_energy * candidate_n))
        davidson["residual"] = true_residual
        davidson["metric"] = true_residual
        davidson["davidson_converged"] = bool(
            davidson.get("davidson_converged", False)
            and true_residual <= float(davidson_tol)
        )
        overlap = np.vdot(current_reduced, candidate_n)
        candidate = np.zeros(current.size, dtype=complex)
        candidate[support] = reduced
        if abs(overlap) > 0.0:
            candidate *= np.conj(overlap) / abs(overlap)
        step = candidate[support] - current_reduced
        stationary_step = float(np.sqrt(max(
            np.real(np.vdot(step, projected_metric.matvec(step)))
            / max(float(denominator), np.finfo(float).tiny),
            0.0,
        )))
        stationary = stationary_step <= max(1.0e-12, 0.01 * float(davidson_tol))
        after = float(local_energy + self.ecore)
        metric_direction_norm = candidate_norm / max(
            float(np.vdot(reduced, reduced).real), np.finfo(float).tiny
        )
        metric_supported = bool(
            metric_direction_norm > float(metric_rtol) * float(np.max(np.abs(n_diag)))
        )
        accepted = bool(
            candidate_norm > 0.0
            and np.isfinite(after)
            and np.isfinite(true_residual)
            and metric_supported
            and after <= before + float(accept_tol)
        )
        accepted_inexact = bool(
            accepted and not davidson.get("davidson_converged", False)
        )
        if accepted and not stationary:
            self._set_site_vector(site, candidate)
        if not accepted or stationary:
            after = before
        davidson.update({
            "native_davidson": False,
            "metric_rank": None,
            "metric_nullity": None,
            "metric_whitening_backend": "matrix_free_generalized",
            "recycled_vectors": 0 if recycled is None else int(recycled.shape[1]),
            "transported_ritz_vectors": 0 if recycled is None else int(recycled.shape[1]),
            "preconditioner_mode": ("regularized_metric_cg" if preconditioner
                                    else "generalized_diagonal"),
            "preconditioner_metric_actions": (preconditioner.matvecs if preconditioner else 0),
            "candidate_metric_supported": metric_supported,
            "candidate_metric_direction_norm": metric_direction_norm,
            "accepted_inexact_variational_step": accepted_inexact,
        })
        solver_info["projected_action_backend"] = "matrix_free_generalized"
        solver_info["davidson"] = davidson
        return {
            "site": int(site),
            "energy_before": float(before),
            "energy_after": float(after),
            "local_energy": float(local_energy + self.ecore),
            "support": int(support.size),
            "parameters": int(current.size),
            "accepted": accepted,
            "stationary": bool(stationary),
            "stationary_step": stationary_step,
            "norm_before": float(denominator),
            "norm_after": float(candidate_norm if accepted and not stationary else denominator),
            "native_su2": True,
            "solver": solver,
            "requested_solver": requested_solver,
            "auto_selected": requested_solver == "auto",
            "workers": int(self.workers),
            "fully_wigner_eckart_reduced": True,
            "matrix_free": True,
            "local_linear_algebra": "matrix_free_generalized_davidson",
            "solver_info": solver_info,
            "magnetic_expansion_factor": float(self._magnetic_expansion_factor(bond)),
            "environment_backend": self.local_environment_backend,
            "elapsed": float(time.perf_counter() - started),
        }

    def optimize_site(
        self,
        site,
        *,
        metric_rtol=1.0e-11,
        support_rtol=1.0e-13,
        max_local_parameters=128,
        max_matrix_free_parameters=4096,
        max_dense_metric_parameters=4096,
        accept_tol=1.0e-10,
        solver="auto",
        we_dense_dim=24,
        davidson_tol=1.0e-10,
        davidson_maxiter=80,
        davidson_max_space=None,
        conditional_preconditioner=False,
        davidson_workspace_mib=32,
        conditional_preconditioner_max_block=64,
        conditional_preconditioner_weight=0.1,
        compiled_davidson="auto",
        davidson_minimum_explored_dimension=2,
        davidson_convergence_confirmations=2,
        projected_diagonal_weight=0.0,
        local_eigensolver="davidson",
        sites=None,
        bond=None,
        executor=None,
        environment_sweeps=None,
    ):
        """Perform one reduced local Rayleigh--Ritz update.

        Large matrix-free problems accept ``davidson_workspace_mib`` and
        ``conditional_preconditioner`` (a bounded regularized metric-CG
        correction). This is a Davidson adaptation, not an exact inverse
        shifted Hamiltonian: Davidson, JCP 17, 87 (1975),
        doi:10.1016/0021-9991(75)90065-0. Check the returned true residual;
        a variationally accepted inexact step need not be converged.
        ``metric_rtol`` also filters near-null matrix-free Krylov directions
        by their norm Rayleigh quotient relative to the norm diagonal scale;
        this is directional stabilization, not exact spectral whitening.
        ``davidson_max_space=None`` allows 96 vectors for wider frontiers at
        ``D >= 3`` and 32 otherwise, bounded by the supported coordinate dimension. This is
        a restart-capacity heuristic; explicit values override it, and the
        requested residual tolerance is unchanged.
        ``local_eigensolver="dense"`` instead diagonalizes the small supported
        reduced tensor-coordinate Hamiltonian, assembled with the same batched
        actions. This is an exact local root within the retained metric support,
        not a determinant-space projection or a guarantee of a global minimum.
        """
        if davidson_max_space is None:
            davidson_max_space = 96 if self.max_frontier_width > 1 and self.D >= 3 else 32
        local_eigensolver = str(local_eigensolver).lower()
        if local_eigensolver not in {"davidson", "dense"}:
            raise ValueError("local_eigensolver must be 'davidson' or 'dense'.")
        started = time.perf_counter()
        site = int(site)
        if executor is None:
            executor = self._solver_executor
        requested_solver = str(solver).lower().replace("-", "_")
        if bond is None:
            bond = site if site < self.nsites - 1 else site - 1
        cached_certificate = self._stationary_certificate_cache.get(site)
        dependency_signature = self._stationary_dependency_signature(site)
        if (
            environment_sweeps is not None
            and local_eigensolver == "davidson"
            and requested_solver in {"auto", "wigner_eckart"}
            and cached_certificate is not None
            and cached_certificate[0] == dependency_signature
            and float(cached_certificate[1]) <= float(davidson_tol)
        ):
            update = copy.deepcopy(cached_certificate[2])
            update["requested_solver"] = requested_solver
            update["auto_selected"] = requested_solver == "auto"
            update["magnetic_expansion_factor"] = float(
                self._magnetic_expansion_factor(bond)
            )
            davidson = update["solver_info"]["davidson"]
            davidson["stationary_certificate_cache_hit"] = True
            davidson["certificate_source_bond"] = int(
                cached_certificate[3]
            )
            update["solver_info"]["environment_reused"] = True
            update["elapsed"] = float(time.perf_counter() - started)
            self._stationary_certificate_cache_hits += 1
            return update
        solver = self._select_local_solver(requested_solver)
        matrix_free = False
        solver_info = {}
        if solver == "polarization":
            current, support, hamiltonian, metric = self._local_quadratic_matrices(
                site,
                support_rtol=support_rtol,
                max_local_parameters=max_local_parameters,
            )
        else:
            matrix_free = self._pack_site(site).size > min(
                int(we_dense_dim), int(max_local_parameters)
            )
            if matrix_free:
                (
                    current,
                    support,
                    h_action,
                    n_action,
                    h_diag,
                    n_diag,
                    solver_info,
                ) = self._frontier_wigner_eckart_matrix_free_problem(
                    site,
                    bond,
                    support_rtol=support_rtol,
                    max_parameters=max_matrix_free_parameters,
                    max_dense_metric_parameters=max_dense_metric_parameters,
                    executor=executor,
                    sites=sites,
                    environment_sweeps=environment_sweeps,
                )
            else:
                (
                    current,
                    support,
                    hamiltonian,
                    metric,
                    solver_info,
                ) = self._wigner_eckart_local_matrices(
                    site,
                    support_rtol=support_rtol,
                    max_local_parameters=max_local_parameters,
                    executor=executor,
                    sites=sites,
                )
        current_reduced = current[support]
        if matrix_free:
            projected_metric = solver_info.pop("_projected_metric")
            matrix_free_projected_metric = isinstance(
                projected_metric, _MatrixFreeProjectedMetric
            )
            if not matrix_free_projected_metric and not isinstance(
                projected_metric, _ReducedMetricComponents
            ):
                dense_metric = np.asarray(projected_metric, dtype=complex)
                projected_metric = _ReducedMetricComponents(
                    dense_metric.shape[0],
                    ((np.arange(dense_metric.shape[0]), dense_metric),),
                )
            projected_design = solver_info.pop("_projected_design")
            if not (
                hasattr(projected_design, "matvec")
                and hasattr(projected_design, "rmatvec")
            ):
                projected_design = np.asarray(projected_design, dtype=complex)
            parent_h_diagonal = np.asarray(
                solver_info.pop("_parent_h_diagonal"), dtype=float
            )
            parent_h_action = solver_info.pop("_parent_h_action")
            current_h = h_action(current_reduced)
            denominator = np.real(
                np.vdot(current_reduced, projected_metric.matvec(current_reduced))
            )
        else:
            denominator = np.real(np.vdot(current_reduced, metric @ current_reduced))
        if denominator <= 0.0:
            raise np.linalg.LinAlgError("the current local SU2LETTA metric norm is null.")
        numerator = np.vdot(
            current_reduced,
            current_h if matrix_free else hamiltonian @ current_reduced,
        )
        before = float(np.real(numerator) / denominator) + self.ecore
        if matrix_free and matrix_free_projected_metric:
            if local_eigensolver == "dense":
                raise MemoryError("Dense local roots require a materialized reduced metric.")
            return self._optimize_site_matrix_free_metric(
                site,
                bond,
                current=current,
                support=support,
                current_reduced=current_reduced,
                h_action=h_action,
                projected_metric=projected_metric,
                h_diag=h_diag,
                n_diag=n_diag,
                solver_info=solver_info,
                denominator=denominator,
                before=before,
                accept_tol=accept_tol,
                davidson_tol=davidson_tol,
                davidson_maxiter=davidson_maxiter,
                davidson_max_space=davidson_max_space,
                davidson_minimum_explored_dimension=(
                    davidson_minimum_explored_dimension
                ),
                davidson_convergence_confirmations=(
                    davidson_convergence_confirmations
                ),
                conditional_preconditioner=conditional_preconditioner,
                davidson_workspace_mib=davidson_workspace_mib,
                metric_rtol=metric_rtol,
                solver=solver,
                requested_solver=requested_solver,
                started=started,
            )
        if matrix_free:
            from pyqed.mps.nonabelian import _su2_kernel

            metric_revision = int(
                _su2_kernel._cpp_array_revision(
                    *(
                        value
                        for indices, block in projected_metric.components
                        for value in (indices, block)
                    )
                )
            )
            metric_cache_key = (
                int(site),
                int(bond),
                tuple(map(int, support)),
            )
            cached_whitener = self._metric_whitener_cache.get(metric_cache_key)
            if (
                cached_whitener is not None
                and cached_whitener[0] == metric_revision
            ):
                (
                    _revision,
                    whitener,
                    retained_metric,
                    whitening_backend,
                ) = cached_whitener
                self._metric_whitener_cache_hits += 1
            else:
                component_cache = self._metric_whitener_component_cache.setdefault(
                    metric_cache_key, {}
                )
                (
                    whitener,
                    retained_metric,
                    whitening_backend,
                    reused_components,
                ) = projected_metric.whitener(
                    rtol=metric_rtol,
                    component_cache=component_cache,
                    revision=lambda indices, block: _su2_kernel._cpp_array_revision(
                        indices, block
                    ),
                )
                self._metric_whitener_component_hits += int(reused_components)
                cache_limit = max(4, 4 * len(projected_metric.components))
                while len(component_cache) > cache_limit:
                    component_cache.pop(next(iter(component_cache)))
                self._metric_whitener_cache[metric_cache_key] = (
                    metric_revision,
                    whitener,
                    retained_metric,
                    whitening_backend,
                )
                self._metric_whitener_cache_misses += 1
            whitened_current = (
                whitener.conj().T @ projected_metric.matvec(current_reduced)
            )
            current_coordinate_norm = float(np.linalg.norm(whitened_current))
            if current_coordinate_norm <= np.finfo(float).tiny:
                raise np.linalg.LinAlgError(
                    "the current local SU2LETTA vector is outside the retained "
                    "metric range."
                )
            normalized_current = whitened_current / current_coordinate_norm
            current_projected_h = (
                whitener.conj().T @ current_h / current_coordinate_norm
            )
            current_local_energy = float(
                np.real(np.vdot(normalized_current, current_projected_h))
            )
            current_residual = float(
                np.linalg.norm(
                    current_projected_h
                    - current_local_energy * normalized_current
                )
            )
            reconstructed_current = whitener @ whitened_current
            reconstruction_delta = reconstructed_current - current_reduced
            reconstruction_error = float(
                np.sqrt(max(
                    np.real(np.vdot(
                        reconstruction_delta,
                        projected_metric.matvec(reconstruction_delta),
                    )) / max(float(denominator), np.finfo(float).tiny),
                    0.0,
                ))
            )
            stationary_certificate = bool(
                current_residual <= float(davidson_tol)
                and reconstruction_error
                <= max(1.0e-11, 0.1 * float(davidson_tol))
            )
            if stationary_certificate and local_eigensolver == "davidson":
                cached_krylov = self._projected_krylov_cache.get(
                    (int(site), int(bond))
                )
                recycled_vectors = int(
                    0
                    if cached_krylov is None
                    else cached_krylov.shape[1]
                )
                orthonormality_error = float(
                    np.linalg.norm(
                        whitener.conj().T
                        @ projected_metric.matvec(whitener)
                        - np.eye(whitener.shape[1])
                    )
                )
                solver_info["projected_action_backend"] = (
                    "certified_current_residual"
                )
                solver_info["davidson"] = {
                    "metric": current_residual,
                    "residual": current_residual,
                    "davidson_iterations": 0,
                    "davidson_converged": True,
                    "subspace_dim": 1,
                    "restarts": 0,
                    "packed_dimension": int(whitener.shape[1]),
                    "native_davidson": False,
                    "stationary_residual_certificate": True,
                    "reconstruction_error": reconstruction_error,
                    "recycled_vectors": recycled_vectors,
                    "transported_ritz_vectors": recycled_vectors,
                    "fused_cpp_action": False,
                    "preconditioner_mode": "stationary_residual_certificate",
                    "metric_rank": int(whitener.shape[1]),
                    "metric_nullity": int(support.size - whitener.shape[1]),
                    "metric_condition": float(
                        np.max(retained_metric) / np.min(retained_metric)
                    ),
                    "orthonormality_error": orthonormality_error,
                    "metric_whitening_backend": whitening_backend,
                }
                update = {
                    "site": site,
                    "energy_before": float(before),
                    "energy_after": float(before),
                    "local_energy": float(current_local_energy + self.ecore),
                    "support": int(support.size),
                    "parameters": int(current.size),
                    "accepted": True,
                    "stationary": True,
                    "stationary_step": 0.0,
                    "norm_before": float(denominator),
                    "norm_after": float(denominator),
                    "native_su2": True,
                    "solver": solver,
                    "requested_solver": requested_solver,
                    "auto_selected": requested_solver == "auto",
                    "workers": int(self.workers),
                    "fully_wigner_eckart_reduced": True,
                    "matrix_free": True,
                    "local_linear_algebra": (
                        "metric_orthonormal_projected_davidson"
                    ),
                    "solver_info": solver_info,
                    "magnetic_expansion_factor": float(
                        self._magnetic_expansion_factor(bond)
                    ),
                    "environment_backend": self.local_environment_backend,
                    "elapsed": float(time.perf_counter() - started),
                }
                self._stationary_certificate_cache[site] = (
                    dependency_signature,
                    current_residual,
                    copy.deepcopy(update),
                    int(bond),
                )
                return update
            if isinstance(projected_design, _StructuredPairEmbedding):
                orthonormal_design = projected_design.compose(whitener)
                projected_h_diagonal = orthonormal_design.projected_diagonal(
                    parent_h_diagonal
                )
            else:
                orthonormal_design = np.ascontiguousarray(
                    projected_design @ whitener
                )
                projected_h_diagonal = np.real(
                    np.sum(
                        orthonormal_design.conj()
                        * (parent_h_diagonal[:, None] * orthonormal_design),
                        axis=0,
                    )
                )

            conditional_groups = projected_metric.orthonormal_components(
                whitener
            )
            maximum_conditional_block = max(
                (int(group.size) for group in conditional_groups),
                default=0,
            )
            use_conditional_preconditioner = bool(
                conditional_preconditioner
                and conditional_groups
                and whitener.shape[1] > 16
                and maximum_conditional_block
                <= int(conditional_preconditioner_max_block)
            )
            conditional_eigensystems = ()
            if use_conditional_preconditioner:
                if isinstance(orthonormal_design, _StructuredPairEmbedding):
                    conditional_blocks = (
                        orthonormal_design.projected_diagonal_blocks(
                            parent_h_diagonal,
                            conditional_groups,
                        )
                    )
                else:
                    conditional_blocks = []
                    for group in conditional_groups:
                        local = orthonormal_design[:, group]
                        block = local.conj().T @ (
                            parent_h_diagonal[:, None] * local
                        )
                        conditional_blocks.append(
                            0.5 * (block + block.conj().T)
                        )
                    conditional_blocks = tuple(conditional_blocks)
                conditional_eigensystems = tuple(
                    np.linalg.eigh(block) for block in conditional_blocks
                )

            def orthonormal_h_action(vector):
                return whitener.conj().T @ h_action(whitener @ vector)

            projected_route = self._install_projected_factor_route(
                site=site,
                bond=bond,
                parent_action=parent_h_action,
                orthonormal_design=orthonormal_design,
            )
            if projected_route is None:
                projected_action = orthonormal_h_action
                solver_info["projected_action_backend"] = "python_composition"
            else:
                owner = projected_route["owner"]
                projection_key = projected_route["projection_key"]
                real_projection = projected_route["real_projection"]

                def projected_action(vector):
                    vector = np.asarray(vector)
                    if not real_projection:
                        return owner.factor_route_projected_matvec(projection_key, vector)
                    result = owner.factor_route_projected_real_matvec(projection_key, vector.real)
                    if np.iscomplexobj(vector) and np.any(vector.imag):
                        result = result + 1j * owner.factor_route_projected_real_matvec(
                            projection_key, vector.imag
                        )
                    return result

                def projected_columns(vectors):
                    vectors = np.asarray(vectors)
                    if not real_projection:
                        return owner.factor_route_projected_matmat(projection_key, vectors)
                    if vectors.shape[1] == 0:
                        return np.empty_like(vectors)
                    return np.column_stack([projected_action(column) for column in vectors.T])

                projected_action.matmat = projected_columns
                solver_info["projected_action_real_arithmetic"] = real_projection
                if os.environ.get("PYQED_VALIDATE_LETTA_PROJECTED_ROUTE"):
                    reference_probe = orthonormal_h_action(whitened_current)
                    compiled_probe = projected_action(whitened_current)
                    solver_info["projected_route_validation_error"] = float(
                        np.linalg.norm(compiled_probe - reference_probe)
                        / max(np.linalg.norm(reference_probe), 1.0)
                    )
                solver_info["projected_action_backend"] = "cpp_fused_factor_routes"
                solver_info["projected_route_topology_revision"] = int(
                    projected_route["topology_revision"]
                )
            # The projected parent diagonal is retained as a structured
            # preconditioner diagnostic.  Its off-diagonal transform terms are
            # not available cheaply, so a constant shift is the robust
            # preconditioner until the exact projected diagonal is installed.
            diagonal_center = float(np.median(projected_h_diagonal))
            davidson_diagonal = projected_h_diagonal
            diagonal_weight = min(
                1.0, max(0.0, float(projected_diagonal_weight))
            )
            diagonal_scale = max(
                float(np.median(np.abs(projected_h_diagonal - diagonal_center))),
                1.0e-10,
            )
            trusted_diagonal = np.clip(
                projected_h_diagonal,
                diagonal_center - 8.0 * diagonal_scale,
                diagonal_center + 8.0 * diagonal_scale,
            )
            preconditioner_diagonal = (
                (1.0 - diagonal_weight) * diagonal_center
                + diagonal_weight * trusted_diagonal
            )

            def projected_preconditioner(residual, theta, _vector):
                denominator = theta - preconditioner_diagonal
                floor = np.maximum(
                    1.0e-10,
                    1.0e-6 * np.maximum(np.abs(preconditioner_diagonal), abs(theta)),
                )
                safe = np.where(
                    np.abs(denominator) > floor,
                    denominator,
                    np.where(denominator >= 0.0, floor, -floor),
                )
                shifted = residual / safe
                if conditional_eigensystems:
                    corrected = np.zeros_like(residual, dtype=complex)
                    for group, (values, vectors) in zip(
                        conditional_groups,
                        conditional_eigensystems,
                    ):
                        denominators = theta - values
                        floor = np.maximum(
                            1.0e-10,
                            1.0e-6 * np.maximum(np.abs(values), abs(theta)),
                        )
                        safe = np.where(
                            np.abs(denominators) > floor,
                            denominators,
                            np.where(denominators >= 0.0, floor, -floor),
                        )
                        corrected[group] = vectors @ (
                            (vectors.conj().T @ residual[group]) / safe
                        )
                    if np.all(np.isfinite(corrected)) and np.linalg.norm(
                        corrected
                    ) > 1.0e-15:
                        weight = min(
                            1.0,
                            max(0.0, float(conditional_preconditioner_weight)),
                        )
                        corrected *= np.linalg.norm(shifted) / max(
                            np.linalg.norm(corrected),
                            np.finfo(float).tiny,
                        )
                        return (1.0 - weight) * shifted + weight * corrected
                return shifted

            krylov_key = (int(site), int(bond))
            cached_krylov = self._projected_krylov_cache.get(krylov_key)
            recycled_vectors = None
            if (
                cached_krylov is not None
                and cached_krylov.shape[0] == current.size
            ):
                recycled_reduced = cached_krylov[support]
                recycled_vectors = (
                    whitener.conj().T
                    @ projected_metric.matvec(recycled_reduced)
                )
            native_result = None
            use_compiled_davidson = bool(
                projected_route is not None
                and local_eigensolver == "davidson"
                and (
                    compiled_davidson is True
                    or (
                        str(compiled_davidson).lower() == "auto"
                        and whitener.shape[1] <= 16
                    )
                )
            )
            if use_compiled_davidson:
                trial = owner.factor_route_projected_davidson(
                    projection_key,
                    np.full(whitener.shape[1], diagonal_center, dtype=complex),
                    whitened_current,
                    float(davidson_tol),
                    int(davidson_maxiter),
                    min(int(davidson_max_space), int(whitener.shape[1])),
                    True,
                )
                if (
                    bool(trial.get("converged", False))
                    and float(trial.get("residual_norm", np.inf))
                    <= float(davidson_tol)
                ):
                    native_result = trial
            if local_eigensolver == "dense":
                columns = np.eye(whitener.shape[1], dtype=complex)
                matmat = getattr(projected_action, "matmat", None)
                matrix = (matmat(columns) if matmat is not None else
                          self._apply_packed_columns(projected_action, columns, executor=executor))
                values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
                local_energy, whitened_reduced = float(values[0]), vectors[:, 0]
                residual = float(np.linalg.norm(
                    projected_action(whitened_reduced) - local_energy * whitened_reduced))
                recycle_space = whitened_reduced[:, None]
                davidson_info = dict(metric=residual, residual=residual,
                    davidson_iterations=0, davidson_converged=residual <= float(davidson_tol),
                    subspace_dim=int(whitener.shape[1]), restarts=0,
                    packed_dimension=int(whitener.shape[1]), dense_reduced_local=True)
            elif native_result is None:
                local_energy, whitened_reduced, davidson_info = (
                    _solve_packed_generalized_davidson(
                        whitened_current,
                        projected_action,
                        h_diag=davidson_diagonal,
                        N=None,
                        n_diag=None,
                        tol=float(davidson_tol),
                        tol_residual=float(davidson_tol),
                        itermax=int(davidson_maxiter),
                        max_space=min(
                            int(davidson_max_space), int(whitener.shape[1])
                        ),
                        precond=projected_preconditioner,
                        initial_vectors=recycled_vectors,
                        return_recycle_space=True,
                        minimum_explored_dimension=(
                            davidson_minimum_explored_dimension
                        ),
                        convergence_confirmations=(
                            davidson_convergence_confirmations
                        ),
                    )
                )
                recycle_space = davidson_info.pop("_recycle_space")
                davidson_info["native_davidson"] = False
            else:
                local_energy = float(native_result["energy"])
                whitened_reduced = np.asarray(native_result["vector"])
                recycle_space = whitened_reduced[:, None]
                davidson_info = {
                    "metric": float(native_result["residual_norm"]),
                    "residual": float(native_result["residual_norm"]),
                    "davidson_iterations": int(native_result["iterations"]),
                    "davidson_converged": True,
                    "subspace_dim": int(native_result["basis_size"]),
                    "restarts": int(native_result["restarts"]),
                    "packed_dimension": int(whitener.shape[1]),
                    "native_davidson": True,
                    "cpp_davidson_kind": str(native_result.get("kind")),
                }
            tied_recycle = np.zeros(
                (current.size, recycle_space.shape[1]), dtype=complex
            )
            tied_recycle[support] = whitener @ recycle_space
            self._projected_krylov_cache[krylov_key] = tied_recycle
            davidson_info["recycled_vectors"] = int(
                0 if recycled_vectors is None else recycled_vectors.shape[1]
            )
            davidson_info["transported_ritz_vectors"] = int(
                0 if recycled_vectors is None else recycled_vectors.shape[1]
            )
            davidson_info["fused_cpp_action"] = projected_route is not None
            davidson_info["preconditioner_mode"] = (
                "none" if local_eigensolver == "dense" else "compiled_constant_shift"
                if native_result is not None or local_eigensolver == "dense"
                else (
                    "conditional_block_jacobi_hybrid"
                    if conditional_eigensystems
                    else "projected_diagonal_seed_constant_shift"
                )
            )
            davidson_info["conditional_preconditioner_blocks"] = int(
                len(conditional_eigensystems)
            )
            davidson_info["conditional_preconditioner_max_block"] = int(
                maximum_conditional_block
            )
            davidson_info["conditional_preconditioner_weight"] = float(
                conditional_preconditioner_weight
            )
            davidson_info["projected_diagonal_weight"] = float(diagonal_weight)
            davidson_info["compiled_davidson_policy"] = str(compiled_davidson)
            reduced = whitener @ whitened_reduced
            candidate_h = projected_action(whitened_reduced)
            candidate_norm = float(np.real(np.vdot(
                reduced, projected_metric.matvec(reduced)
            )))
            if candidate_norm <= 0.0 or not np.all(np.isfinite(candidate_h)):
                local_energy = np.inf
            else:
                local_energy = float(
                    np.real(np.vdot(whitened_reduced, candidate_h))
                    / candidate_norm
                )
            davidson_info["metric_rank"] = int(whitener.shape[1])
            davidson_info["metric_nullity"] = int(
                support.size - whitener.shape[1]
            )
            davidson_info["metric_condition"] = float(
                np.max(retained_metric) / np.min(retained_metric)
            )
            davidson_info["orthonormality_error"] = float(
                np.linalg.norm(
                    whitener.conj().T
                    @ projected_metric.matvec(whitener)
                    - np.eye(whitener.shape[1])
                )
            )
            davidson_info["metric_whitening_backend"] = whitening_backend
            davidson_info["projected_diagonal_min"] = float(
                np.min(projected_h_diagonal, initial=0.0)
            )
            davidson_info["projected_diagonal_max"] = float(
                np.max(projected_h_diagonal, initial=0.0)
            )
            if float(davidson_info.get("residual", np.inf)) <= float(davidson_tol):
                davidson_info["davidson_converged"] = True
            solver_info["davidson"] = davidson_info
            overlap = np.vdot(
                current_reduced, projected_metric.matvec(reduced)
            )
        else:
            local_energy, reduced = _lowest_generalized_pair(
                hamiltonian,
                metric,
                rtol=metric_rtol,
            )
            overlap = np.vdot(current_reduced, metric @ reduced)
        candidate = np.zeros(
            current.size,
            dtype=np.result_type(current.dtype, reduced.dtype, complex),
        )
        candidate[support] = reduced
        if abs(overlap) > 0.0:
            candidate *= np.conj(overlap) / abs(overlap)
        if matrix_free:
            step = candidate[support] - current_reduced
            stationary_step = float(
                np.sqrt(
                    max(
                        np.real(np.vdot(step, projected_metric.matvec(step)))
                        / max(float(denominator), np.finfo(float).tiny),
                        0.0,
                    )
                )
            )
        else:
            stationary_step = float(
                np.linalg.norm(candidate - current)
                / max(float(np.linalg.norm(current)), np.finfo(float).tiny)
            )
        stationary_tolerance = max(1.0e-12, 0.01 * float(davidson_tol))
        stationary = stationary_step <= stationary_tolerance
        if not stationary:
            self._set_site_vector(site, candidate)
        after = float(local_energy + self.ecore)
        accepted = after <= before + float(accept_tol)
        if matrix_free:
            accepted = accepted and bool(
                davidson_info.get("davidson_converged", False)
            )
        if not accepted:
            if not stationary:
                self._set_site_vector(site, current)
            after = before
        elif stationary:
            after = before
        norm_after = float(
            candidate_norm
            if accepted and matrix_free and not stationary
            else denominator
        )
        return {
            "site": site,
            "energy_before": float(before),
            "energy_after": float(after),
            "local_energy": float(local_energy + self.ecore),
            "support": int(support.size),
            "parameters": int(current.size),
            "accepted": bool(accepted),
            "stationary": bool(stationary),
            "stationary_step": stationary_step,
            "norm_before": float(denominator),
            "norm_after": norm_after,
            "native_su2": True,
            "solver": solver,
            "requested_solver": requested_solver,
            "auto_selected": requested_solver == "auto",
            "workers": int(self.workers),
            "fully_wigner_eckart_reduced": solver == "wigner_eckart",
            "matrix_free": bool(matrix_free),
            "local_linear_algebra": (
                "metric_orthonormal_dense" if local_eigensolver == "dense" else
                "metric_orthonormal_projected_davidson"
                if matrix_free
                else "dense_generalized_eigh"
            ),
            "solver_info": solver_info,
            "magnetic_expansion_factor": float(
                self._magnetic_expansion_factor(bond)
            ),
            "environment_backend": (
                self.local_environment_backend
                if solver == "wigner_eckart"
                else "full_chain_polarization"
            ),
            "elapsed": float(time.perf_counter() - started),
        }

    def _materialize_explicit_hamiltonian(self):
        """Switch a direct complementary state to the exact reduced MPO."""
        if self.hamiltonian_representation != "complementary":
            return False
        factors = prepare_hamiltonian_factors(self.mpo, materialize=True)
        self.mpo = list(factors)
        self._complementary_operators = None
        self.hamiltonian_representation = "mpo"
        if hasattr(self, "hamiltonian"):
            self.hamiltonian = replace(
                self.hamiltonian,
                factors=list(self.mpo),
                complementary_operators=None,
                moving_environment=None,
            )
        self._projected_route_cache.clear()
        self._projected_krylov_cache.clear()
        self._metric_whitener_cache.clear()
        self._metric_whitener_component_cache.clear()
        self._stationary_certificate_cache.clear()
        return True

    def sweep_parameter_blocks(self, *, spill_directory, **kwargs):
        """Run one experimental disk-backed reduced-parameter pass.

        ``direction='lr'`` or ``'rl'`` selects traversal. Updates the state
        using moving reduced environments; returns per-site
        diagnostics. See :func:`pyqed.letta.parameter_sweep.sweep_parameter_blocks`
        for the Weichselbaum (2012) adaptation, budgets, and restricted-space
        convergence limitations. This does not certify global convergence.
        """
        from .parameter_sweep import sweep_parameter_blocks
        return sweep_parameter_blocks(self, spill_directory=spill_directory, **kwargs)

    def run_parameter_blocks(self, *, spill_directory, **kwargs):
        """Run LR/RL cycles with sparse local coordinates and boundary reuse.

        Populates energy, history and success on this solver. Convergence
        requires an independent final-state retained-metric stationarity audit,
        not merely local roots during a pass. See
        :func:`pyqed.letta.parameter_sweep.run_parameter_blocks` for the reduced
        Weichselbaum (2012) adaptation, memory limits and fidelity restrictions.
        """
        from .parameter_sweep import run_parameter_blocks
        return run_parameter_blocks(self, spill_directory=spill_directory, **kwargs)

    def run(
        self,
        *,
        nsweeps=2,
        tol=1.0e-9,
        residual_tol=None,
        truncation_tol=None,
        consecutive_cycles=1,
        max_local_parameters=128,
        max_matrix_free_parameters=4096,
        verbose=0,
        solver="auto",
        algorithm="one_site",
        we_dense_dim=24,
        pair_cutoff=1.0e-10,
        retraction_maxiter=8,
        max_retraction_parameters=4096,
        checkpoint=None,
        checkpoint_every=1,
        reset_history=True,
        gauge="auto",
        gauge_rcond=1.0e-12,
        reuse_environments=None,
        global_variational_guard="adaptive",
        global_accept_tol=1.0e-10,
        global_guard_trigger_tol=1.0e-8,
        adaptive_davidson=False,
        adaptive_ties=False,
        tie_options=None,
        representation_break_even_cycles=40,
        **kwargs,
    ):
        """Optimize by complete left/right reduced SU(2) cycles.

        Projected sweeps reuse moving reduced environments by default.
        ``reuse_environments=False`` selects explicit rebuilding for reference
        checks on MPO states; complementary states always require reuse.

        ``adaptive_ties=True`` alternates fixed-graph cycles with cost-bounded
        graph additions; ``tie_options`` controls the interval and trial budget.
        See :meth:`adapt_ties` for the selection heuristic and limitations.

        ``adaptive_davidson`` loosens the requested Davidson tolerance using
        the previous cycle energy change, capped at 1e-4. This is a heuristic
        inexact-solve schedule, not a convergence guarantee; stopping criteria
        and global variational checks are unchanged.
        """
        if adaptive_ties:
            options = dict(locals())
            from .su2_adaptive_graph import run_adaptive
            options.pop("self")
            options.pop("adaptive_ties")
            options.pop("tie_options")
            options.pop("nsweeps")
            options.update(options.pop("kwargs"))
            return run_adaptive(
                self, nsweeps=nsweeps, tie_options=tie_options, run_options=options,
            )
        if tie_options is not None:
            raise ValueError("tie_options requires adaptive_ties=True")
        target_davidson_tol = float(kwargs.get("davidson_tol", 1.0e-10))
        if not np.isfinite(target_davidson_tol) or target_davidson_tol <= 0:
            raise ValueError("davidson_tol must be finite and positive")
        nsweeps = int(nsweeps)
        if nsweeps < 1:
            raise ValueError("nsweeps must be positive.")
        representation_break_even_cycles = max(
            1, int(representation_break_even_cycles)
        )
        if (
            self._hamiltonian_representation_policy == "auto"
            and self.hamiltonian_representation == "complementary"
            and nsweeps >= representation_break_even_cycles
        ):
            switched = self._materialize_explicit_hamiltonian()
            self._representation_decision = {
                "policy": "auto",
                "selected": self.hamiltonian_representation,
                "reason": (
                    "requested cycles exceed the measured direct/MPO "
                    "cold-start break-even"
                ),
                "break_even_cycles": representation_break_even_cycles,
                "switched": bool(switched),
            }
        elif self._hamiltonian_representation_policy == "auto":
            self._representation_decision = {
                "policy": "auto",
                "selected": self.hamiltonian_representation,
                "reason": "direct complementary wins below the cold-start break-even",
                "break_even_cycles": representation_break_even_cycles,
                "switched": False,
            }
        algorithm = str(algorithm).lower().replace("-", "_")
        if algorithm in {"1site", "single_site"}:
            algorithm = "one_site"
        if algorithm in {"2site", "pair"}:
            algorithm = "two_site"
        if algorithm in {"projected_one_site", "tied_space"}:
            algorithm = "projected"
        if algorithm not in {"one_site", "two_site", "projected"}:
            raise ValueError(
                "SU2LETTA algorithm must be 'one_site', 'two_site', or "
                "'projected'."
            )
        if self.virtual_boundary=='periodic' and algorithm!='projected':
            raise ValueError("closed multiplicity loops require algorithm='projected' one-site updates")
        consecutive_cycles = int(consecutive_cycles)
        if consecutive_cycles < 1:
            raise ValueError("consecutive_cycles must be positive.")
        final_checkpoint_only = (
            checkpoint_every is None
            or str(checkpoint_every).lower().replace("-", "_")
            in {"final", "at_end"}
        )
        checkpoint_interval = None
        if not final_checkpoint_only:
            checkpoint_interval = int(checkpoint_every)
            if checkpoint_interval < 1:
                raise ValueError(
                    "checkpoint_every must be positive, None, or 'final'."
                )
        requested_solver = str(solver).lower().replace("-", "_")
        local_solver = self._select_local_solver(requested_solver)
        if gauge is not None:
            gauge = str(gauge).lower().replace("-", "_")
        if gauge == "auto":
            gauge = (
                "conditional"
                if algorithm == "projected"
                and self.hamiltonian_representation == "complementary"
                else None
            )
        elif gauge in {"none", "off"}:
            gauge = None
        if gauge not in {None, "conditional"}:
            raise ValueError(
                "SU2LETTA gauge must be 'auto', None, or 'conditional'."
            )
        conditional_gauge = bool(
            algorithm in {"two_site", "projected"}
            and gauge == "conditional"
            and self.supports_conditional_canonical_gauge
        )
        if isinstance(global_variational_guard, (bool, np.bool_)):
            global_guard_mode = (
                "adaptive" if bool(global_variational_guard) else "off"
            )
        else:
            global_guard_mode = str(global_variational_guard).lower().replace(
                "-", "_"
            )
            if global_guard_mode in {"none", "false", "disabled"}:
                global_guard_mode = "off"
        if global_guard_mode not in {"off", "adaptive", "all"}:
            raise ValueError(
                "global_variational_guard must be 'adaptive', 'all', or False."
            )
        global_accept_tol = float(global_accept_tol)
        global_guard_trigger_tol = float(global_guard_trigger_tol)
        if global_accept_tol < 0.0 or global_guard_trigger_tol < 0.0:
            raise ValueError("global guard tolerances must be nonnegative.")
        use_global_guard = bool(
            global_guard_mode != "off"
            and algorithm == "projected"
            and self.hamiltonian_representation == "complementary"
        )
        reuse_environments = (algorithm == "projected" if reuse_environments is None
                              else bool(reuse_environments))
        if self.hamiltonian_representation == "complementary":
            reuse_environments = True
        moving_environment = None
        # The C++ reduced-boundary route batch has a fixed packing/install
        # cost.  Four- and five-site active spaces have too little interior
        # work to amortize it; keep their small Hamiltonian boundaries in the
        # exact Python recursion while still using C++ contextual pair routes.
        cpp_boundary_environment = (
            self._su2_moving_environment if self.nsites >= 6 else None
        )

        def new_moving_environment(sites):
            environment = MovingEnvironment(
                sites,
                mpo_factors=self.mpo,
                complementary_operator_families=self._complementary_operators,
                materialize_complementary_family_operator_tables=False,
                su2_moving_environment=None,
                su2_boundary_environment=cpp_boundary_environment,
            )
            # Keep explicit factorized norm boundary components available for
            # connected whitening. Their local pair action is still owned by
            # the compiled reduced engine.
            environment.norm_stack.su2_boundary_environment = None
            return environment

        if algorithm in {"two_site", "projected"} and reuse_environments:
            moving_environment = new_moving_environment(self.materialize())
        previous = float(self.energy)
        if reset_history:
            self.history = []
        self.converged = False
        streak = 0
        canonical_center = None
        h_chain = n_chain = None
        for sweep in range(nsweeps):
            cycle_kwargs = dict(kwargs)
            cycle_davidson_tol = target_davidson_tol
            if adaptive_davidson:
                last_delta = (
                    float(self.history[-1]["energy_delta"])
                    if self.history else np.inf
                )
                if last_delta > float(tol):
                    cycle_davidson_tol = max(
                        target_davidson_tol, min(1.0e-4, 0.1 * last_delta)
                    )
                cycle_kwargs["davidson_tol"] = cycle_davidson_tol
            if (
                sweep
                and moving_environment is not None
                and self.hamiltonian_representation == "complementary"
            ):
                moving_environment = new_moving_environment(self.materialize())
                h_chain = n_chain = None
            cycle_started = time.perf_counter()
            cycle_vectors = (
                tuple(self._pack_site(site).copy() for site in range(self.nsites))
                if use_global_guard
                else None
            )
            updates = []
            gauge_updates = []
            environment_build_s = 0.0
            hamiltonian_environment_build_s = 0.0
            norm_environment_build_s = 0.0
            environment_chain_reuses = 0
            carried_energy = float(previous)
            global_guard_evaluations = 0
            global_guard_rejections = 0
            global_guard_estimates = 0
            global_guard_s = 0.0
            for direction, bonds in (
                ("lr", range(self.nsites - 1)),
                ("rl", range(self.nsites - 2, -1, -1)),
            ):
                bonds = tuple(bonds)
                recentered = False
                if conditional_gauge:
                    center = 0 if direction == "lr" else self.nsites - 1
                    needed_side = (
                        None
                        if moving_environment is None
                        else moving_environment.needed_prebuilt_side(direction)
                    )
                    preserve_moving_boundary = bool(
                        algorithm == "projected"
                        and needed_side is not None
                        and moving_environment.hamiltonian_valid_boundary_side
                        == needed_side
                        and moving_environment.norm_valid_boundary_side
                        == needed_side
                    )
                    if canonical_center != center and not preserve_moving_boundary:
                        center_updates = self.canonicalize_conditional_center(
                            center, rcond=gauge_rcond
                        )
                        gauge_updates.extend(center_updates)
                        recentered = any(
                            bool(item.get("applied", False))
                            for item in center_updates
                        )
                        canonical_center = center
                materialized = self.materialize()
                direct_half_sweep = bool(
                    algorithm == "projected"
                    and self.hamiltonian_representation == "complementary"
                )
                if direct_half_sweep:
                    self._su2_moving_environment.install_mps(materialized)
                    self._su2_moving_environment.begin_half_sweep(
                        direction,
                        self.nsites,
                    )
                environment_sweeps = None
                if (
                    algorithm in {"two_site", "projected"}
                    and reuse_environments
                ):
                    environment_started = time.perf_counter()
                    if (
                        algorithm == "projected"
                        and self._su2_moving_environment is not None
                    ):
                        if self.hamiltonian_representation == "complementary":
                            self._su2_moving_environment.clear_boundaries()
                        self._su2_moving_environment.clear_factor_routes()
                        moving_environment.hamiltonian_stack.su2_moving_environment = (
                            self._su2_moving_environment
                        )
                    if recentered:
                        h_reuse = n_reuse = None
                    else:
                        h_reuse, n_reuse = moving_environment.reuse_sides_for(
                            direction
                        )
                        if self.hamiltonian_representation == "complementary":
                            h_reuse = None
                    if h_chain is not None and h_reuse is not None:
                        h_chain.sites = list(materialized)
                        environment_chain_reuses += 1
                    else:
                        hamiltonian_environment_started = time.perf_counter()
                        h_chain = BlockSparseEnvironmentChain.build(
                            materialized,
                            self.mpo,
                            renormalized_blocks=(
                                moving_environment.hamiltonian_stack
                            ),
                            sweep_direction=direction,
                            reuse_prebuilt_boundary_side=h_reuse,
                        )
                        hamiltonian_environment_build_s += (
                            time.perf_counter() - hamiltonian_environment_started
                        )
                    if n_chain is not None and n_reuse is not None:
                        n_chain.sites = list(materialized)
                        environment_chain_reuses += 1
                    else:
                        norm_environment_started = time.perf_counter()
                        n_chain = BlockSparseEnvironmentChain.build(
                            materialized,
                            moving_environment.identity_mpo_factors,
                            renormalized_blocks=moving_environment.norm_stack,
                            sweep_direction=direction,
                            reuse_prebuilt_boundary_side=n_reuse,
                        )
                        norm_environment_build_s += (
                            time.perf_counter() - norm_environment_started
                        )
                    environment_sweeps = (
                        h_chain.start_sweep(direction),
                        n_chain.start_sweep(direction),
                    )
                    environment_build_s += time.perf_counter() - environment_started
                for bond in bonds:
                    if direct_half_sweep:
                        self._su2_moving_environment.begin_bond(bond)
                        self._su2_moving_environment.merge_active_bond()
                    local_environment_sweeps = environment_sweeps
                    if algorithm == "two_site":
                        h_stack = (
                            None
                            if moving_environment is None or algorithm == "projected"
                            else moving_environment.hamiltonian_stack
                        )
                        if h_stack is not None:
                            if self._su2_moving_environment is not None:
                                self._su2_moving_environment.clear_boundaries()
                                self._su2_moving_environment.clear_factor_routes()
                            h_stack.su2_moving_environment = (
                                self._su2_moving_environment
                            )
                        try:
                            update = self.optimize_two_sites(
                                bond,
                                cutoff=pair_cutoff,
                                retraction_maxiter=retraction_maxiter,
                                max_retraction_parameters=max_retraction_parameters,
                                sites=materialized,
                                environment_sweeps=local_environment_sweeps,
                                **cycle_kwargs,
                            )
                        finally:
                            if h_stack is not None:
                                h_stack.su2_moving_environment = None
                        if (
                            conditional_gauge
                            and update.get("accepted", False)
                            and not update.get("stationary", False)
                        ):
                            local_gauge_updates = self.canonicalize_conditional_bond(
                                bond,
                                direction=direction,
                                rcond=gauge_rcond,
                            )
                            gauge_updates.extend(local_gauge_updates)
                            update["conditional_gauge"] = {
                                "applied": int(
                                    sum(
                                        bool(item.get("applied", False))
                                        for item in local_gauge_updates
                                    )
                                ),
                                "skipped": int(
                                    sum(
                                        not bool(item.get("applied", False))
                                        for item in local_gauge_updates
                                    )
                                ),
                            }
                            canonical_center = (
                                bond + 1 if direction == "lr" else bond
                            )
                        elif conditional_gauge:
                            update["conditional_gauge"] = {
                                "applied": 0,
                                "skipped": 0,
                                "stationary": True,
                            }
                        materialized[bond] = self.materialize_site(bond)
                        materialized[bond + 1] = self.materialize_site(bond + 1)
                        if environment_sweeps is not None:
                            for environment_sweep in environment_sweeps:
                                environment_sweep.advance_after_update(
                                    bond,
                                    materialized[bond],
                                    materialized[bond + 1],
                                )
                    else:
                        site = bond if direction == "lr" else bond + 1
                        site_vector_before = self._pack_site(site).copy()
                        h_stack = (
                            None
                            if moving_environment is None or algorithm == "projected"
                            else moving_environment.hamiltonian_stack
                        )
                        if h_stack is not None:
                            if self._su2_moving_environment is not None:
                                self._su2_moving_environment.clear_boundaries()
                                self._su2_moving_environment.clear_factor_routes()
                            h_stack.su2_moving_environment = (
                                self._su2_moving_environment
                            )
                        try:
                            self._incremental_materialization_enabled = (
                                algorithm == "projected"
                            )
                            update = self.optimize_site(
                                site,
                                max_local_parameters=max_local_parameters,
                                max_matrix_free_parameters=max_matrix_free_parameters,
                                solver=local_solver,
                                we_dense_dim=(
                                    0 if algorithm == "projected" else we_dense_dim
                                ),
                                sites=materialized,
                                bond=bond,
                                executor=self._solver_executor,
                                environment_sweeps=(
                                    local_environment_sweeps
                                    if algorithm == "projected"
                                    else None
                                ),
                                **cycle_kwargs,
                            )
                        finally:
                            self._incremental_materialization_enabled = False
                            if h_stack is not None:
                                h_stack.su2_moving_environment = None
                        update["requested_solver"] = requested_solver
                        update["auto_selected"] = requested_solver == "auto"
                        update["bond"] = int(bond)
                        local_before = float(update["energy_before"])
                        local_after = float(update["energy_after"])
                        local_delta = local_after - local_before
                        guard_key = (direction, int(bond))
                        coordinate_offset = abs(local_before - carried_energy)
                        update["local_energy_before"] = local_before
                        update["local_energy_after"] = local_after
                        update["local_energy_delta"] = float(local_delta)
                        update["global_coordinate_offset"] = float(
                            coordinate_offset
                        )
                        if (
                            use_global_guard
                            and update.get("accepted", False)
                            and not update.get("stationary", False)
                        ):
                            check_global = bool(
                                global_guard_mode == "all"
                                or guard_key in self._global_guard_bonds
                                or coordinate_offset > global_guard_trigger_tol
                            )
                            if check_global:
                                guard_started = time.perf_counter()
                                global_after = float(self.expectation())
                                global_guard_s += (
                                    time.perf_counter() - guard_started
                                )
                                global_guard_evaluations += 1
                                global_delta = global_after - carried_energy
                                if (
                                    abs(global_delta - local_delta)
                                    > global_guard_trigger_tol
                                ):
                                    self._global_guard_bonds.add(guard_key)
                            else:
                                global_after = carried_energy + local_delta
                                global_guard_estimates += 1
                            globally_accepted = bool(
                                global_after <= carried_energy + global_accept_tol
                            )
                            if not globally_accepted:
                                self._set_site_vector(site, site_vector_before)
                                self._stationary_certificate_cache.pop(site, None)
                                for key in tuple(self._projected_krylov_cache):
                                    if key[0] == site:
                                        del self._projected_krylov_cache[key]
                                update["rejected_krylov_cleared"] = True
                                global_after = carried_energy
                                update["accepted"] = False
                                update["stationary"] = False
                                global_guard_rejections += 1
                                self._global_guard_bonds.add(guard_key)
                            update["globally_accepted"] = globally_accepted
                            update["global_variational_guard"] = True
                            update["global_guard_checked"] = check_global
                            update["global_energy_estimated"] = not check_global
                            update["energy_before"] = float(carried_energy)
                            update["energy_after"] = float(global_after)
                        elif use_global_guard:
                            update["globally_accepted"] = bool(
                                update.get("accepted", False)
                            )
                            update["global_variational_guard"] = True
                            update["global_guard_checked"] = False
                            update["global_energy_estimated"] = False
                            update["energy_before"] = float(carried_energy)
                            update["energy_after"] = float(carried_energy)
                        if not update.get("accepted", False):
                            for key in tuple(self._projected_krylov_cache):
                                if key[0] == site:
                                    del self._projected_krylov_cache[key]
                            self._stationary_certificate_cache.pop(site, None)
                            update["rejected_krylov_cleared"] = True
                        if (
                            conditional_gauge
                            and update.get("accepted", False)
                            and not update.get("stationary", False)
                        ):
                            local_gauge_updates = self.canonicalize_conditional_bond(
                                bond,
                                direction=direction,
                                rcond=gauge_rcond,
                            )
                            gauge_updates.extend(local_gauge_updates)
                            update["conditional_gauge"] = {
                                "applied": int(sum(
                                    bool(item.get("applied", False))
                                    for item in local_gauge_updates
                                )),
                                "skipped": int(sum(
                                    not bool(item.get("applied", False))
                                    for item in local_gauge_updates
                                )),
                            }
                            canonical_center = (
                                bond + 1 if direction == "lr" else bond
                            )
                        elif conditional_gauge:
                            update["conditional_gauge"] = {
                                "applied": 0,
                                "skipped": 0,
                                "stationary": True,
                            }
                        materialized[site] = self.materialize_site(site)
                        if algorithm == "projected":
                            gauge_changed = bool(
                                update.get("conditional_gauge", {}).get(
                                    "applied", 0
                                )
                            )
                            refresh_boundary_sites = bool(
                                not update.get("stationary", False)
                                or gauge_changed
                            )
                            if refresh_boundary_sites:
                                self._invalidate_materialized_sites(
                                    bond, bond + 1
                                )
                                materialized[bond] = self.materialize_site(bond)
                                materialized[bond + 1] = self.materialize_site(
                                    bond + 1
                                )
                            update["boundary_refresh_skipped"] = bool(
                                not refresh_boundary_sites
                            )
                            if environment_sweeps is not None:
                                for environment_sweep in environment_sweeps:
                                    environment_sweep.advance_after_update(
                                        bond,
                                        materialized[bond],
                                        materialized[bond + 1],
                                    )
                    update["direction"] = direction
                    carried_energy = float(update.get("energy_after", carried_energy))
                    updates.append(update)
                    if direct_half_sweep:
                        self._su2_moving_environment.stage_bond_update(
                            materialized[bond],
                            materialized[bond + 1],
                        )
                        self._su2_moving_environment.commit_bond_update(
                            energy=update.get("energy_after")
                        )
                if environment_sweeps is not None:
                    moving_environment.finish_sweep(direction)
                if direct_half_sweep:
                    self._su2_moving_environment.finish_half_sweep()
                if algorithm == "projected" and moving_environment is not None:
                    moving_environment.hamiltonian_stack.su2_moving_environment = None
            if self.hamiltonian_representation == "complementary":
                energy = self.expectation()
            elif algorithm == "projected" and updates:
                energy = float(updates[-1]["energy_after"])
            else:
                energy = self.expectation()
            candidate_energy = float(energy)
            global_energy_closure = float(candidate_energy - carried_energy)
            global_energy_closure_error = abs(global_energy_closure)
            if use_global_guard and (
                global_energy_closure_error > global_guard_trigger_tol
            ):
                inconsistent = max(
                    updates,
                    key=lambda item: item.get("global_coordinate_offset", 0.0),
                    default=None,
                )
                if inconsistent is not None:
                    self._global_guard_bonds.add(
                        (
                            str(inconsistent["direction"]),
                            int(inconsistent["bond"]),
                        )
                    )
            cycle_rolled_back = bool(
                use_global_guard
                and candidate_energy > previous + global_accept_tol
            )
            if cycle_rolled_back:
                for site, vector in enumerate(cycle_vectors):
                    self._set_site_vector(site, vector)
                self._stationary_certificate_cache.clear()
                self._projected_krylov_cache.clear()
                canonical_center = None
                h_chain = n_chain = None
                energy = float(previous)
                global_guard_rejections += 1
            if self.hamiltonian_representation == "complementary":
                cycle_norm = float(self.norm())
            elif algorithm == "projected" and updates and not cycle_rolled_back:
                cycle_norm = float(updates[-1]["norm_after"])
            else:
                cycle_norm = float(self.norm())
            delta = abs(energy - previous)
            residuals = []
            truncation_errors = []
            for update in updates:
                if update.get("local_residual") is not None:
                    residuals.append(float(update["local_residual"]))
                if update.get("retraction_residual") is not None:
                    residuals.append(float(update["retraction_residual"]))
                davidson = update.get("solver_info", {}).get("davidson", {})
                if davidson.get("residual") is not None:
                    residuals.append(float(davidson["residual"]))
                if update.get("truncation_error") is not None:
                    truncation_errors.append(float(update["truncation_error"]))
            max_residual = max(residuals, default=0.0)
            max_truncation = max(truncation_errors, default=0.0)
            rejected = sum(not bool(update.get("accepted", True)) for update in updates)
            cycle_ok = delta <= float(tol)
            if residual_tol is not None:
                cycle_ok = cycle_ok and max_residual <= float(residual_tol)
            if truncation_tol is not None:
                cycle_ok = cycle_ok and max_truncation <= float(truncation_tol)
            if cycle_rolled_back:
                cycle_ok = False
            stationary_fixed_point = bool(
                algorithm == "projected"
                and updates
                and all(
                    update.get("stationary", False)
                    and update.get("accepted", False)
                    for update in updates
                )
            )
            if cycle_ok and stationary_fixed_point:
                streak = max(streak + 1, consecutive_cycles)
            else:
                streak = streak + 1 if cycle_ok else 0
            cpp_local_stats = None
            if (
                moving_environment is not None
                and self._su2_moving_environment is not None
            ):
                owner_stats = self._su2_moving_environment.stats
                cpp_local_stats = {
                    key: owner_stats.get(key)
                    for key in (
                        "backend",
                        "factor_route_count",
                        "factor_route_matvec_calls",
                        "factor_route_diagonal_calls",
                        "contextual_route_plan_builds",
                        "contextual_route_plan_hits",
                        "reduced_contextual_execution_count",
                        "reduced_contextual_matrix_elements",
                        "memory_bytes",
                    )
                }
                cpp_local_stats["threading"] = dict(
                    self._su2_moving_environment.threading_info
                )
            record = {
                "sweep": len(self.history) + 1,
                "energy": float(energy),
                "energy_delta": float(delta),
                "max_local_residual": float(max_residual),
                "max_truncation_error": float(max_truncation),
                "rejected_updates": int(rejected),
                "accepted_updates": int(len(updates) - rejected),
                "stationary_updates": int(
                    sum(bool(update.get("stationary", False)) for update in updates)
                ),
                "boundary_refreshes_skipped": int(
                    sum(
                        bool(update.get("boundary_refresh_skipped", False))
                        for update in updates
                    )
                ),
                "stationary_fixed_point": stationary_fixed_point,
                "consecutive_cycles": int(streak),
                "updates": updates,
                "complete_cycle": True,
                "symmetry": "SU2",
                "algorithm": algorithm,
                "hamiltonian_representation": self.hamiltonian_representation,
                "representation_decision": dict(self._representation_decision),
                "norm": cycle_norm,
                "storage_nbytes": int(self.storage_nbytes),
                "gauge": "conditional" if conditional_gauge else None,
                "conditional_gauge_supported": bool(
                    self.supports_conditional_canonical_gauge
                ),
                "conditional_gauge_applied": int(
                    sum(
                        bool(item.get("applied", False))
                        for item in gauge_updates
                    )
                ),
                "conditional_gauge_skipped": int(
                    sum(
                        not bool(item.get("applied", False))
                        for item in gauge_updates
                    )
                ),
                "environment_reuse": bool(
                    algorithm in {"two_site", "projected"} and reuse_environments
                ),
                "environment_build_s": float(environment_build_s),
                "hamiltonian_environment_build_s": float(
                    hamiltonian_environment_build_s
                ),
                "norm_environment_build_s": float(norm_environment_build_s),
                "environment_chain_reuses": int(environment_chain_reuses),
                "global_variational_guard": use_global_guard,
                "global_guard_mode": global_guard_mode,
                "davidson_tolerance": cycle_davidson_tol,
                "global_guard_evaluations": int(global_guard_evaluations),
                "global_guard_estimates": int(global_guard_estimates),
                "global_guard_rejections": int(global_guard_rejections),
                "global_guard_learned_bonds": tuple(
                    sorted(self._global_guard_bonds)
                ),
                "global_guard_s": float(global_guard_s),
                "candidate_energy": candidate_energy,
                "cycle_rolled_back": cycle_rolled_back,
                "global_energy_closure": global_energy_closure,
                "global_energy_closure_error": float(
                    global_energy_closure_error
                ),
                "moving_environment_backend": (
                    (
                        "cpp_su2_local_boundaries_python_metric"
                        if cpp_boundary_environment is not None
                        else "cpp_su2_local_routes_python_boundaries_metric"
                    )
                    if moving_environment is not None
                    and self._su2_moving_environment is not None
                    else "python"
                    if moving_environment is not None
                    else None
                ),
                "moving_environment_stats": (
                    None
                    if moving_environment is None
                    else moving_environment.stats
                ),
                "cpp_local_operator_stats": cpp_local_stats,
                "constraint_cache_stats": {
                    "pair_coordinate_hits": int(
                        self._pair_coordinate_cache_hits
                    ),
                    "pair_coordinate_misses": int(
                        self._pair_coordinate_cache_misses
                    ),
                    "embedding_hits": int(self._embedding_basis_cache_hits),
                    "embedding_numeric_hits": int(
                        self._embedding_numeric_cache_hits
                    ),
                    "embedding_incremental_updates": int(
                        self._embedding_incremental_updates
                    ),
                    "embedding_reused_blocks": int(
                        self._embedding_reused_blocks
                    ),
                    "embedding_updated_blocks": int(
                        self._embedding_updated_blocks
                    ),
                    "metric_component_hits": int(
                        self._metric_component_cache_hits
                    ),
                    "metric_component_misses": int(
                        self._metric_component_cache_misses
                    ),
                    "metric_whitener_hits": int(
                        self._metric_whitener_cache_hits
                    ),
                    "metric_whitener_component_hits": int(
                        self._metric_whitener_component_hits
                    ),
                    "projected_route_hits": int(
                        self._projected_route_cache_hits
                    ),
                    "stationary_certificate_hits": int(
                        self._stationary_certificate_cache_hits
                    ),
                    "materialized_site_hits": int(
                        self._materialized_site_cache_hits
                    ),
                },
                "elapsed_s": float(time.perf_counter() - cycle_started),
            }
            self.history.append(record)
            if verbose:
                print(
                    f"SU2-LETTA cycle {record['sweep']}: E={energy:.12f} "
                    f"dE={delta:.3e} residual={max_residual:.3e} "
                    f"trunc={max_truncation:.3e}"
                )
            self.energy = float(energy)
            if (
                checkpoint is not None
                and checkpoint_interval is not None
                and record["sweep"] % checkpoint_interval == 0
            ):
                self.save_checkpoint(checkpoint)
            if streak >= consecutive_cycles:
                self.converged = True
                break
            previous = energy
        self.energy = float(energy)
        self.success = bool(self.converged)
        self.message = (
            "converged by a complete LR/RL SU(2) cycle"
            if self.converged
            else "completed requested reduced SU(2)-LETTA cycles"
        )
        if checkpoint is not None:
            self.save_checkpoint(checkpoint)
        return self


class SU2LETTA(NonAbelianFrontierLETTA):
    """Spatial-orbital qchem adapter for :class:`NonAbelianFrontierLETTA`.

    Reduced algebra adapts Weichselbaum, PRB 86, 245124 (2012),
    https://doi.org/10.1103/PhysRevB.86.245124. Periodic multiplicity traces
    adapt Verstraete, Porras and Cirac, PRL 93, 227205 (2004),
    https://doi.org/10.1103/PhysRevLett.93.227205. These are LETTA adaptations,
    not reproductions of their solvers. Periodic support is restricted to a
    spin-zero, fixed-charge seam and singlet targets, using exact spectator
    lifting and reduced MPO one-site projected updates. General spin-carrying
    seams, periodic canonical gauges and global convergence guarantees are
    not provided. See the base class for the contraction and fidelity details.
    """

    @classmethod
    def from_narg(cls, solver, state_ids=None):
        """Read direct-reduced NARG truncation tensors as a LETTA root batch.

        This exact conversion supports the sequential reduced chain. It does
        not reconstruct determinants, fit a state vector, or construct an MPS.
        Detached-frame chains require their own conditional-tensor exporter.
        """
        from pyqed.narg.qchem.su2_overlap import _root_tensors

        sites, root_ids, target = _root_tensors(solver, state_ids)
        dimension = max(int(dim) for site in sites[:-1] for dim in site.legs[2].dims.values())
        state = cls(None, base_sites=sites, target_sector=target, graph=None,
                    nelec=solver.target_irrep[0], spin=solver.target_irrep[1], D=dimension)
        state.root_ids = tuple(root_ids)
        return state


__all__ = ["NonAbelianFrontierLETTA", "SU2LETTA"]
