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
    LocalTransitionPlan,
    _apply_two_site_dense_factorized,
    _left_reduced_rank_coupled_block,
    _right_reduced_rank_coupled_block,
    _factorize_left_two_site_dense_term,
    _factorize_right_two_site_dense_term,
    contract_chain_expectation,
    contract_chain_transition,
)
from pyqed.mps.nonabelian.decompose import svd_two_site
from pyqed.mps.nonabelian.solver import (
    _resolve_davidson_operator,
    _solve_packed_generalized_davidson,
    pack_two_site_state,
    unpack_two_site_state,
)
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
            1, min(int(maximum), int(os.cpu_count() or 1))
        )
    if isinstance(workers, (bool, np.bool_)):
        raise TypeError("workers must be a positive integer or 'auto'.")
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be a positive integer or 'auto'.")
    return workers


def _materialized_rank_coupled_factors(factors):
    """Return reduced-term MPO cores when given a lightweight NC carrier."""
    factors = tuple(factors)
    if not factors:
        return factors
    owner = getattr(factors[0], "normal_complementary_owner", None)
    if owner is None or not all(
        getattr(factor, "normal_complementary_plan", None) is not None
        and not tuple(getattr(factor, "reduced_terms", ()))
        for factor in factors
    ):
        return factors
    from pyqed.qchem.dmrg.backends.reduced import (
        build_su2_normal_complementary_mpo,
    )

    materialized = tuple(
        build_su2_normal_complementary_mpo(
            owner,
            fully_reduced=bool(
                getattr(factors[0], "normal_complementary_fully_reduced", False)
            ),
            materialize_reduced_terms=True,
        )
    )
    for factor in materialized:
        # Right-to-left cached contractions use the dual reduced carrier.
        # Do not retain the C++ owner on every materialized core.  The LETTA
        # state keeps one transient runtime reference, while checkpoints and
        # deep copies remain free of the non-pickleable object.
        object.__setattr__(factor, "normal_complementary_right_dual", True)
        object.__setattr__(factor, "normal_complementary_owner", None)
        object.__setattr__(
            factor,
            "normal_complementary_force_contextual_routes",
            True,
        )
    return materialized


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

    def whitener(self, *, rtol):
        scale = max(
            (
                float(np.max(np.abs(block), initial=0.0))
                for _indices, block in self.components
            ),
            default=1.0,
        )
        scale = max(scale, 1.0)
        threshold = float(rtol) * scale
        columns = []
        retained = []
        backends = []
        for indices, raw_block in self.components:
            block = 0.5 * (raw_block + raw_block.conj().T)
            diagonal = np.real(np.diag(block))
            off_diagonal = block - np.diag(diagonal)
            if np.linalg.norm(off_diagonal) <= threshold * max(
                np.sqrt(indices.size), 1.0
            ):
                keep = diagonal > threshold
                if not np.any(keep):
                    continue
                local = np.zeros((indices.size, np.count_nonzero(keep)), dtype=complex)
                local[np.flatnonzero(keep), np.arange(np.count_nonzero(keep))] = (
                    1.0 / np.sqrt(diagonal[keep])
                )
                values = diagonal[keep]
                backends.append("diagonal")
            else:
                values, vectors = np.linalg.eigh(block)
                keep = values > threshold
                if not np.any(keep):
                    continue
                local = vectors[:, keep] / np.sqrt(values[keep])
                values = values[keep]
                backends.append("eigh")
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
        return np.column_stack(columns), np.asarray(retained), backend

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


def _canonical_graph(nsites, graph):
    if graph is None:
        return ()
    if isinstance(graph, str):
        key = graph.lower().replace("-", "_")
        if key in {"nn", "nearest_neighbor"}:
            return _nearest_neighbor_graph(nsites)
        raise ValueError(
            "SU2LETTA graph strings must be 'nn' or 'nearest_neighbor'."
        )
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


class _NearestNeighborTieRoutePlan(_WignerEckartRoutePlan):
    """Bounded-frontier reduced scatter plan for physical NN ties."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backend = "nearest-neighbor-block-scatter"
        self.conditioned_frontier_width = 1


class NonAbelianFrontierLETTA:
    r"""Graph-tied reduced SU(2) frontier LETTA.

    A graph tie conditions an earlier tensor on either a future invariant
    physical-sector label or a future reduced fusion sector. Magnetic
    projections are supplied only by the structural Clebsch--Gordan coupling
    in the non-Abelian MPS contraction and are never copied by a tie.

    ``D`` counts reduced multiplets per reachable virtual sector, not the
    magnetic-state dimension.

    The reduced-tensor layer follows the Wigner--Eckart/non-Abelian tensor
    network formulation of A. Weichselbaum, Phys. Rev. B 86, 245124 (2012),
    https://doi.org/10.1103/PhysRevB.86.245124.  This is a PyQED adaptation to
    the project-specific LETTA graph-tied ansatz, not a reproduction of that
    paper's MPS/NRG algorithms.  The implementation currently supports fully
    reduced spatial-orbital SU(2) states and scalar rank-coupled Hamiltonians;
    it does not provide general non-Abelian groups or arbitrary tensor ranks.
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
        n_threads=1,
        we_route_memory=256.0,
    ):
        factors = getattr(mpo, "factors", mpo)
        self._su2_moving_environment = getattr(mpo, "moving_environment", None)
        self._complementary_operators = getattr(
            mpo, "complementary_operators", None
        )
        self.mpo = _materialized_rank_coupled_factors(factors)
        self.nsites = len(self.mpo)
        if self.nsites < 2:
            raise ValueError("NonAbelianFrontierLETTA requires at least two sites.")
        if not isinstance(D, Integral) or isinstance(D, (bool, np.bool_)) or int(D) < 1:
            raise ValueError("D must be a positive reduced multiplet dimension.")
        self.D = int(D)
        self.workers = resolve_workers(workers)
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
        self._projected_krylov_cache = {}
        self._metric_whitener_cache = {}
        self._metric_whitener_cache_hits = 0
        self._metric_whitener_cache_misses = 0
        self._metric_component_topology_cache = {}
        self._metric_component_cache_hits = 0
        self._metric_component_cache_misses = 0
        self._state_revision = 0
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
        self.energy = self.expectation()
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
        **kwargs,
    ):
        """Build the fully reduced qchem MPO and its SU(2)-LETTA state.

        The default LETTA tie graph is the nearest-neighbor orbital chain;
        ``graph="nn"`` selects it explicitly. Hamiltonian couplings remain
        complete in the reduced MPO; ``graph`` controls only variational
        tensor ties and may be supplied explicitly.
        """
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
            complementary_operators=None,
            moving_environment=None,
        )
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
        return tuple(site.copy() for site in sites)

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
            self.tie == "physical"
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
            self._invalidate_materialized_sites(bond, bond + 1)
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
        self._materialized_site_cache[site] = materialized
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
        executor = getattr(self, "_solver_executor", None)
        self._solver_executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def __deepcopy__(self, memo):
        """Copy variational state while dropping transient compiled owners."""
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            if name in {"_su2_moving_environment", "_complementary_operators"}:
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

    def norm(self):
        sites = self.materialize()
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, self.mpo)
        return _real_scalar(contract_chain_expectation(sites, identity))

    def expectation(self, mpo=None):
        factors = self.mpo if mpo is None else tuple(getattr(mpo, "factors", mpo))
        sites = self.materialize()
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, factors)
        denominator = _real_scalar(contract_chain_expectation(sites, identity))
        if denominator <= 0.0:
            raise FloatingPointError("SU2LETTA norm is non-positive.")
        numerator = _real_scalar(contract_chain_expectation(sites, factors))
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
            np.asarray(blocks).nbytes
            for blocks in self._embedding_basis_cache.values()
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

    @property
    def convergence_summary(self):
        """Return the latest complete-cycle convergence diagnostics."""
        latest = self.history[-1] if self.history else {}
        return {
            "converged": bool(self.converged),
            "cycles": int(len(self.history)),
            "energy": float(self.energy),
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
        checkpoint_mpo = copy.deepcopy(self.mpo)
        for core in checkpoint_mpo:
            for name in (
                "_reduced_block_cache",
                "_environment_reduced_block_cache",
                "_block_cache",
            ):
                cache = getattr(core, name, None)
                if cache is not None:
                    cache.clear()
        payload = {
            "format": "pyqed-nonabelian-frontier-letta",
            "version": 2,
            "mpo": checkpoint_mpo,
            "nelec": self.nelec,
            "spin": self.spin,
            "target_sector": self.target_sector,
            "tie": self.tie,
            "graph": self.graph,
            "D": self.D,
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
        }
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return path

    @classmethod
    def load_checkpoint(cls, path, *, workers=1):
        """Restore a checkpoint created by :meth:`save_checkpoint`."""
        path = Path(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        legacy = (
            payload.get("format") == "pyqed-su2-letta"
            and payload.get("version") == 1
        )
        current = (
            payload.get("format") == "pyqed-nonabelian-frontier-letta"
            and payload.get("version") == 2
        )
        if not (legacy or current):
            raise ValueError("unsupported SU(2)-LETTA checkpoint format.")
        state = cls(
            payload["mpo"],
            nelec=payload["nelec"],
            spin=payload["spin"],
            target_sector=payload.get("target_sector"),
            graph=payload["graph"],
            D=payload["D"],
            ecore=payload["ecore"],
            base_sites=payload["base_sites"],
            tie=payload.get("tie", "physical"),
            init="mps",
            workers=workers,
            n_threads=payload.get("n_threads", 1),
            we_route_memory=payload.get("we_route_memory", 256.0),
        )
        state.core_energy = float(payload.get("core_energy", state.ecore))
        state.mpo_includes_core_energy = bool(
            payload.get("mpo_includes_core_energy", False)
        )
        state.tensors = [
            {key: np.array(value, copy=True) for key, value in tensor.items()}
            for tensor in payload["tensors"]
        ]
        state._materialized_site_cache.clear()
        state.history = list(payload.get("history", ()))
        state.energy = float(payload.get("energy", state.expectation()))
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
        cached = self._materialized_site_cache.get(int(site))
        route = self._we_route_cache.get(int(site))
        if (
            self._incremental_materialization_enabled
            and cached is not None
            and route is not None
        ):
            route.apply_delta(cached, vector - previous)
        else:
            self._invalidate_materialized_sites(site)

    def _quadratic_value(self, sites, site, vector, factors):
        self._set_site_vector(site, vector)
        varied = list(sites)
        varied[site] = self.materialize_site(site)
        return _real_scalar(contract_chain_expectation(varied, factors))

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

    def _compile_wigner_eckart_routes(self, site):
        """Compile the structural tied-parameter scatter map once."""
        site = int(site)
        current = self._pack_site(site)
        template = self.materialize_site(site)
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
        for block_key, target_block in template.data.items():
            q_left, q_phys, q_right = block_key
            source_keys = [
                key
                for key in self._tensor_keys[site]
                if key[:3] == block_key
            ]
            if not source_keys:
                continue
            d_left = left_dims[q_left]
            d_right = right_dims[q_right]
            endpoint_value = self._tie_index_by_site[site][
                q_phys if self.tie == "physical" else q_left
            ]
            indices = []
            matrix_rows = []
            for source_key in source_keys:
                condition = source_key[3]
                source_shape = np.asarray(self.tensors[site][source_key]).shape
                source_offset = parameter_offsets[source_key]
                for local_index in range(int(np.prod(source_shape))):
                    i_left, i_phys, i_right = np.unravel_index(
                        local_index, source_shape
                    )
                    row = np.zeros(np.asarray(target_block).size, dtype=float)
                    for left_index, left_values in enumerate(left_maps):
                        if (
                            site in left_values
                            and left_values[site] != endpoint_value
                        ):
                            continue
                        for right_index, right_values in enumerate(right_maps):
                            common = set(left_values).intersection(right_values)
                            if any(
                                left_values[value] != right_values[value]
                                for value in common
                            ):
                                continue
                            if tuple(
                                right_values[parent]
                                for parent in self.parent_sets[site]
                            ) != condition:
                                continue
                            target_index = np.ravel_multi_index(
                                (
                                    left_index * d_left + i_left,
                                    i_phys,
                                    right_index * d_right + i_right,
                                ),
                                np.asarray(target_block).shape,
                            )
                            row[target_index] = 1.0
                    if np.any(row):
                        indices.append(source_offset + local_index)
                        matrix_rows.append(row)
            if not indices:
                continue
            indices = np.asarray(indices, dtype=np.intp)
            matrix = np.ascontiguousarray(np.stack(matrix_rows))
            nbytes += int(indices.nbytes + matrix.nbytes)
            if nbytes > self._we_route_limit_bytes:
                raise MemoryError(
                    "packed Wigner--Eckart routes require more than "
                    f"{self.we_route_memory:g} MiB; increase we_route_memory."
                )
            groups.append(
                (block_key, indices, matrix, np.asarray(target_block).shape)
            )
        if not groups:
            raise np.linalg.LinAlgError("the local Wigner--Eckart route is empty.")
        plan_type = (
            _NearestNeighborTieRoutePlan
            if self.supports_conditional_canonical_gauge
            else _WignerEckartRoutePlan
        )
        return plan_type(
            template=template,
            groups=groups,
            nparameters=current.size,
            nbytes=nbytes,
        )

    def _wigner_eckart_route_plan(self, site):
        site = int(site)
        plan = self._we_route_cache.get(site)
        if plan is not None:
            self._we_route_cache_hits += 1
            return plan
        self._we_route_cache_misses += 1
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
            "backend": "block-grouped-gemm",
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

    def _local_pair_embedding(self, site, sites, bond, layout):
        """Embed tied one-site parameters into a packed two-site tensor."""
        current = self._pack_site(site)
        dtype = np.result_type(current.dtype, complex)
        routes = self._wigner_eckart_route_plan(site)
        if routes.nparameters != current.size:
            raise RuntimeError(
                "Cached Wigner--Eckart basis does not match the local parameter space."
            )
        if isinstance(layout, _ChannelResolvedPairSpace):
            embedding = np.zeros(
                (layout.size, current.size), dtype=dtype
            )
            for entry in layout.entries:
                q_l, q_p1, q_p2, q_r, q_mid = entry.key
                varied_key = (
                    (q_l, q_p1, q_mid)
                    if site == bond
                    else (q_mid, q_p2, q_r)
                )
                cache_key = (int(site), varied_key)
                varied_blocks = self._embedding_basis_cache.get(cache_key)
                if varied_blocks is None:
                    template = (
                        sites[bond].data.get((q_l, q_p1, q_mid))
                        if site == bond
                        else sites[bond + 1].data.get((q_mid, q_p2, q_r))
                    )
                    if template is None:
                        continue
                    varied_blocks = np.asarray(
                        routes.block_basis(varied_key, dtype=dtype),
                        dtype=dtype,
                    )
                    self._embedding_basis_cache[cache_key] = varied_blocks
                    self._embedding_basis_cache_misses += 1
                else:
                    self._embedding_basis_cache_hits += 1
                if site == bond:
                    fixed = sites[bond + 1].data.get((q_mid, q_p2, q_r))
                    if fixed is None:
                        continue
                    pair_blocks = np.tensordot(
                        varied_blocks, fixed, axes=([3], [0])
                    )
                else:
                    fixed = sites[bond].data.get((q_l, q_p1, q_mid))
                    if fixed is None:
                        continue
                    pair_blocks = np.moveaxis(
                        np.tensordot(
                            fixed, varied_blocks, axes=([2], [1])
                        ),
                        2,
                        0,
                    )
                embedding[
                    entry.offset : entry.offset + entry.size
                ] = pair_blocks.reshape(current.size, -1).T
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
    ):
        """Project and assemble only connected tied-metric components."""
        embedding = np.asarray(embedding)
        dimension = int(embedding.shape[1])
        active_designs = {}
        from pyqed.mps.nonabelian import _su2_kernel

        def active_design(entry):
            key = (int(entry.offset), int(entry.size))
            cached = active_designs.get(key)
            if cached is None:
                rows = embedding[key[0] : key[0] + key[1]]
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
            contribution = out_design.conj().T @ (
                np.asarray(block) @ in_design
            )
            threshold = 1.0e-15 * max(
                float(np.max(np.abs(contribution), initial=0.0)), 1.0
            )
            pattern = np.ascontiguousarray(
                np.abs(contribution) > threshold, dtype=np.uint8
            )
            records.append((out_columns, in_columns, contribution, pattern))
            topology_values.append(
                (
                    int(out_entry.offset),
                    int(in_entry.offset),
                    tuple(map(int, out_columns)),
                    tuple(map(int, in_columns)),
                    int(_su2_kernel._cpp_array_revision(pattern)),
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

            for out_columns, in_columns, _contribution, pattern in records:
                for out_position, in_position in np.argwhere(pattern):
                    union(
                        int(out_columns[out_position]),
                        int(in_columns[in_position]),
                    )
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
        for out_columns, in_columns, contribution, _pattern in records:
            for label in np.intersect1d(
                labels[out_columns], labels[in_columns], assume_unique=False
            ):
                out_keep = np.flatnonzero(labels[out_columns] == label)
                in_keep = np.flatnonzero(labels[in_columns] == label)
                blocks[int(label)][
                    np.ix_(positions[out_columns[out_keep]], positions[in_columns[in_keep]])
                ] += contribution[np.ix_(out_keep, in_keep)]
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

        projection_blocks = projection_builder(orthonormal_design)
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
        merged = merge_mps_sites(sites[bond], sites[bond + 1])
        layout = _ChannelResolvedPairSpace(sites[bond], sites[bond + 1])
        _embedded_current, embedding = self._local_pair_embedding(
            site, sites, bond, layout
        )
        if embedding.shape[1] != current.size:
            raise RuntimeError("one-site frontier embedding has the wrong size.")
        h_pair, n_pair, pair_diagnostics = self._reduced_pair_transition_actions(
            bond,
            sites,
            merged,
            layout,
            environment_sweeps=environment_sweeps,
        )

        metric_blocks = getattr(n_pair, "metric_blocks", None)
        if metric_blocks is None:
            n_columns = self._apply_packed_columns(
                n_pair, embedding, executor=executor
            )
            dense_metric = embedding.conj().T @ n_columns
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
            )
            metric_backend = "connected_reduced_blocks"
        n_diagonal = projected_metric_full.diagonal
        scale = max(float(np.max(np.abs(n_diagonal), initial=0.0)), 1.0)
        support = np.flatnonzero(n_diagonal > float(support_rtol) * scale)
        if support.size == 0:
            raise np.linalg.LinAlgError("local SU2LETTA parameter support is null.")
        design = np.asarray(embedding[:, support])
        projected_metric = projected_metric_full.restrict(support)
        parent_h_diagonal = np.asarray(
            getattr(h_pair, "diag", np.zeros(layout.size)), dtype=float
        )
        h_diagonal = np.real(
            np.sum(design.conj() * (parent_h_diagonal[:, None] * design), axis=0)
        )
        n_diagonal = n_diagonal[support]
        matvec_counts = {"hamiltonian": 0, "metric": 0}

        def h_action(vector):
            vector = np.asarray(vector, dtype=complex).reshape(-1)
            if vector.size != support.size:
                raise ValueError("frontier Wigner--Eckart vector has the wrong size.")
            matvec_counts["hamiltonian"] += 1
            return design.conj().T @ h_pair(design @ vector)

        def n_action(vector):
            vector = np.asarray(vector, dtype=complex).reshape(-1)
            if vector.size != support.size:
                raise ValueError("frontier Wigner--Eckart vector has the wrong size.")
            matvec_counts["metric"] += 1
            return design.conj().T @ n_pair(design @ vector)

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
                        site, sites, bond, layout
                    )
                    if current.size > int(max_parameters):
                        raise MemoryError(
                            "two-site SU2LETTA retraction has "
                            f"{current.size} parameters on site {site}; limit is "
                            f"{int(max_parameters)}."
                        )
                    column_norms = np.sum(np.abs(embedding) ** 2, axis=0).real
                    scale = max(float(np.max(column_norms, initial=0.0)), 1.0)
                    support = np.flatnonzero(column_norms > float(cutoff) * scale)
                    if support.size == 0:
                        continue
                    coefficients, *_ = np.linalg.lstsq(
                        embedding[:, support],
                        target,
                        rcond=float(cutoff),
                    )
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

    def _reduced_pair_transition_actions(
        self,
        bond,
        sites,
        merged,
        layout,
        *,
        environment_sweeps=None,
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
        if environment_sweeps is None or not isinstance(
            layout, _ChannelResolvedPairSpace
        ):
            h_plan = AdjacentPairTransitionPlan.build(sites, self.mpo, bond)
            n_plan = AdjacentPairTransitionPlan.build(sites, identity, bond)
        if isinstance(layout, _ChannelResolvedPairSpace):
            def compiled_action(factors, sweep=None):
                if sweep is None:
                    chain = BlockSparseEnvironmentChain.build(sites, factors)
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
                    diagonal = np.asarray(
                        [
                            packed_apply(np.eye(dimension)[:, index])[index]
                            for index in range(dimension)
                        ]
                    )
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
                    columns = np.asarray(columns)
                    blocks = []
                    for _own_index, own_entry, _op_index, op_entry in entry_map:
                        local = columns[
                            own_entry.offset : own_entry.offset + own_entry.size
                        ]
                        active = np.flatnonzero(np.any(local != 0.0, axis=0))
                        if active.size == 0:
                            continue
                        blocks.append(
                            (
                                slice(
                                    int(op_entry.offset),
                                    int(op_entry.offset + op_entry.size),
                                ),
                                np.asarray(active, dtype=np.int64),
                                np.ascontiguousarray(local[:, active]),
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
                if np.linalg.norm(block) > 1.0e-14
            )
            n_action.metric_blocks = metric_blocks
            n_action.transition_count = len(metric_blocks)

            diagnostics = {
                "backend": "compiled_contextual_channel_resolved",
                "dimension": dimension,
                "hamiltonian_direction": (
                    h_plan.direction if h_sweep is None else h_sweep.direction
                ),
                "metric_direction": (
                    n_plan.direction if n_sweep is None else n_sweep.direction
                ),
                "hamiltonian_cached_sites": (
                    h_plan.cached_sites if h_sweep is None else self.nsites - 2
                ),
                "metric_cached_sites": (
                    n_plan.cached_sites if n_sweep is None else self.nsites - 2
                ),
                "hamiltonian_transitions": h_action.transition_count,
                "metric_transitions": n_action.transition_count,
                "metric_blocks": len(metric_blocks),
                "metric_nbytes": int(sum(block.nbytes for _, _, block in metric_blocks)),
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
                    site, sites, bond, layout
                )
                if current.size > int(max_parameters):
                    raise MemoryError(
                        "variational pair retraction has "
                        f"{current.size} parameters on site {site}; limit is "
                        f"{int(max_parameters)}."
                    )
                column_norms = np.sum(np.abs(embedding) ** 2, axis=0).real
                scale = max(float(np.max(column_norms, initial=0.0)), 1.0)
                support = np.flatnonzero(column_norms > float(cutoff) * scale)
                if support.size == 0:
                    continue
                design = np.asarray(embedding[:, support])
                n_columns = self._apply_packed_columns(
                    n_action, design, executor=self._solver_executor
                )
                n_diagonal = np.real(np.sum(design.conj() * n_columns, axis=0))
                metric_scale = max(
                    float(np.max(np.abs(n_diagonal), initial=0.0)), 1.0
                )
                active = np.flatnonzero(
                    n_diagonal > float(cutoff) * metric_scale
                )
                if active.size == 0:
                    continue
                support = support[active]
                design = design[:, active]
                n_diagonal = n_diagonal[active]
                h_columns = self._apply_packed_columns(
                    h_action, design, executor=self._solver_executor
                )
                h_diagonal = np.real(np.sum(design.conj() * h_columns, axis=0))
                initial = np.asarray(current[support], dtype=complex)

                def local_h(vector):
                    return design.conj().T @ h_action(design @ vector)

                def local_n(vector):
                    return design.conj().T @ n_action(design @ vector)

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
        """
        bond = int(bond)
        if bond < 0 or bond >= self.nsites - 1:
            raise IndexError(f"bond {bond} is outside a chain of length {self.nsites}.")
        started = time.perf_counter()
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
            metric_matrix = np.zeros(
                (initial.size, initial.size), dtype=complex
            )
            for out_entry, in_entry, block in n_action.metric_blocks:
                metric_matrix[
                    out_entry.offset : out_entry.offset + out_entry.size,
                    in_entry.offset : in_entry.offset + in_entry.size,
                ] += block
            metric_matrix = 0.5 * (metric_matrix + metric_matrix.conj().T)
            metric_values, metric_vectors = np.linalg.eigh(metric_matrix)
            metric_scale = max(
                float(np.max(np.abs(metric_values), initial=0.0)), 1.0
            )
            metric_support = metric_values > float(cutoff) * metric_scale
            if not np.any(metric_support):
                raise np.linalg.LinAlgError(
                    "Channel-resolved pair metric has no positive support."
                )
            whitener = metric_vectors[:, metric_support] / np.sqrt(
                metric_values[metric_support]
            )[None, :]
            whitened_initial = whitener.conj().T @ initial_n

            def whitened_h(vector):
                return whitener.conj().T @ h_action(whitener @ vector)

            local_energy, whitened_target, solver_info = _solve_packed_generalized_davidson(
                whitened_initial,
                whitened_h,
                h_diag=np.zeros(whitener.shape[1], dtype=float),
                tol=float(davidson_tol),
                tol_residual=float(davidson_tol),
                itermax=int(davidson_maxiter),
                max_space=min(int(davidson_max_space), int(whitener.shape[1])),
            )
            target = whitener @ whitened_target
            solver_info["solver"] = "channel_resolved_whitened_pair_davidson"
            solver_info["metric_rank"] = int(whitener.shape[1])
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
        if pair_matrix_free:
            whitened_residual = whitener.conj().T @ (
                target_h - local_energy * target_n
            )
            residual = float(
                np.linalg.norm(whitened_residual)
                / max(
                    np.linalg.norm(whitener.conj().T @ target_h),
                    abs(local_energy)
                    * np.linalg.norm(whitener.conj().T @ target_n),
                    1.0,
                )
            )
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
        accepted = after <= before + float(accept_tol)
        if not accepted:
            self._set_site_vector(bond, original[0])
            self._set_site_vector(bond + 1, original[1])
            after = before
        return {
            "bond": bond,
            "sites": (bond, bond + 1),
            "energy_before": float(before),
            "energy_after": float(after),
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

    def optimize_site(
        self,
        site,
        *,
        metric_rtol=1.0e-11,
        support_rtol=1.0e-13,
        max_local_parameters=128,
        max_matrix_free_parameters=4096,
        accept_tol=1.0e-10,
        solver="auto",
        we_dense_dim=24,
        davidson_tol=1.0e-10,
        davidson_maxiter=80,
        davidson_max_space=32,
        sites=None,
        bond=None,
        executor=None,
        environment_sweeps=None,
    ):
        """Perform one exact reduced local Rayleigh--Ritz update."""
        site = int(site)
        if executor is None:
            executor = self._solver_executor
        requested_solver = str(solver).lower().replace("-", "_")
        if bond is None:
            bond = site if site < self.nsites - 1 else site - 1
        cached_certificate = self._stationary_certificate_cache.get(site)
        if (
            environment_sweeps is not None
            and requested_solver in {"auto", "wigner_eckart"}
            and cached_certificate is not None
            and int(cached_certificate[0]) == int(self._state_revision)
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
            if not isinstance(projected_metric, _ReducedMetricComponents):
                dense_metric = np.asarray(projected_metric, dtype=complex)
                projected_metric = _ReducedMetricComponents(
                    dense_metric.shape[0],
                    ((np.arange(dense_metric.shape[0]), dense_metric),),
                )
            projected_design = np.asarray(
                solver_info.pop("_projected_design"), dtype=complex
            )
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
                whitener, retained_metric, whitening_backend = (
                    projected_metric.whitener(rtol=metric_rtol)
                )
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
            if stationary_certificate:
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
                }
                self._stationary_certificate_cache[site] = (
                    int(self._state_revision),
                    current_residual,
                    copy.deepcopy(update),
                    int(bond),
                )
                return update
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
                projected_action = lambda vector: owner.factor_route_projected_matvec(
                    projection_key, np.asarray(vector)
                )
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

            def projected_preconditioner(residual, theta, _vector):
                denominator = theta - diagonal_center
                floor = max(
                    1.0e-10,
                    1.0e-6 * abs(float(denominator)),
                )
                safe = (
                    denominator
                    if abs(denominator) > floor
                    else (floor if denominator >= 0.0 else -floor)
                )
                return residual / safe

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
            if projected_route is not None and whitener.shape[1] <= 16:
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
            if native_result is None:
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
                "native_constant_shift"
                if native_result is not None
                else "projected_diagonal_seed_constant_shift"
            )
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
        }

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
        gauge=None,
        gauge_rcond=1.0e-12,
        reuse_environments=False,
        **kwargs,
    ):
        """Optimize by complete left/right reduced SU(2) cycles."""
        nsweeps = int(nsweeps)
        if nsweeps < 1:
            raise ValueError("nsweeps must be positive.")
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
        if gauge not in {None, "conditional"}:
            raise ValueError("SU2LETTA gauge must be None or 'conditional'.")
        conditional_gauge = bool(
            algorithm in {"two_site", "projected"}
            and gauge == "conditional"
            and self.supports_conditional_canonical_gauge
        )
        reuse_environments = bool(reuse_environments)
        moving_environment = None
        # The C++ reduced-boundary route batch has a fixed packing/install
        # cost.  Four- and five-site active spaces have too little interior
        # work to amortize it; keep their small Hamiltonian boundaries in the
        # exact Python recursion while still using C++ contextual pair routes.
        cpp_boundary_environment = (
            self._su2_moving_environment if self.nsites >= 6 else None
        )
        if algorithm in {"two_site", "projected"} and reuse_environments:
            moving_environment = MovingEnvironment(
                self.materialize(),
                mpo_factors=self.mpo,
                complementary_operator_families=self._complementary_operators,
                materialize_complementary_family_operator_tables=False,
                su2_moving_environment=None,
                su2_boundary_environment=cpp_boundary_environment,
            )
            # Keep explicit factorized norm boundary components available for
            # connected whitening.  Their local pair action is still owned by
            # the compiled reduced engine.
            moving_environment.norm_stack.su2_boundary_environment = None
        previous = float(self.energy)
        if reset_history:
            self.history = []
        self.converged = False
        streak = 0
        canonical_center = None
        h_chain = n_chain = None
        for sweep in range(nsweeps):
            cycle_started = time.perf_counter()
            updates = []
            gauge_updates = []
            environment_build_s = 0.0
            hamiltonian_environment_build_s = 0.0
            norm_environment_build_s = 0.0
            environment_chain_reuses = 0
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
                environment_sweeps = None
                if algorithm in {"two_site", "projected"} and reuse_environments:
                    environment_started = time.perf_counter()
                    if (
                        algorithm == "projected"
                        and self._su2_moving_environment is not None
                    ):
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
                                environment_sweeps=environment_sweeps,
                                **kwargs,
                            )
                        finally:
                            if h_stack is not None:
                                h_stack.su2_moving_environment = None
                        if conditional_gauge and not update.get("stationary", False):
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
                                    environment_sweeps
                                    if algorithm == "projected"
                                    else None
                                ),
                                **kwargs,
                            )
                        finally:
                            self._incremental_materialization_enabled = False
                            if h_stack is not None:
                                h_stack.su2_moving_environment = None
                        update["requested_solver"] = requested_solver
                        update["auto_selected"] = requested_solver == "auto"
                        if conditional_gauge and not update.get("stationary", False):
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
                            materialized[bond] = self.materialize_site(bond)
                            materialized[bond + 1] = self.materialize_site(bond + 1)
                            if environment_sweeps is not None:
                                for environment_sweep in environment_sweeps:
                                    environment_sweep.advance_after_update(
                                        bond,
                                        materialized[bond],
                                        materialized[bond + 1],
                                    )
                    update["direction"] = direction
                    updates.append(update)
                if environment_sweeps is not None:
                    moving_environment.finish_sweep(direction)
                if algorithm == "projected" and moving_environment is not None:
                    moving_environment.hamiltonian_stack.su2_moving_environment = None
            if algorithm == "projected" and updates:
                energy = float(updates[-1]["energy_after"])
                cycle_norm = float(updates[-1]["norm_after"])
            else:
                energy = self.expectation()
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
                "stationary_fixed_point": stationary_fixed_point,
                "consecutive_cycles": int(streak),
                "updates": updates,
                "complete_cycle": True,
                "symmetry": "SU2",
                "algorithm": algorithm,
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
                    "embedding_hits": int(self._embedding_basis_cache_hits),
                    "metric_component_hits": int(
                        self._metric_component_cache_hits
                    ),
                    "metric_component_misses": int(
                        self._metric_component_cache_misses
                    ),
                    "metric_whitener_hits": int(
                        self._metric_whitener_cache_hits
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
    """Spatial-orbital qchem adapter for :class:`NonAbelianFrontierLETTA`."""


__all__ = ["NonAbelianFrontierLETTA", "SU2LETTA"]
