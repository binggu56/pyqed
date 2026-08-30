"""Reduced SU(2) LETTA for spatial-orbital quantum chemistry.

This module implements the first non-Abelian LETTA variant.  A tie conditions
on the invariant local multiplet label ``(N, S)``; it never copies a magnetic
projection.  Open tie assignments are embedded in virtual multiplicity space,
so the materialized state remains a conventional reduced SU(2) MPS and can use
the existing rank-coupled quantum-chemistry MPO contractions.

The automatic local optimizer uses the projected two-site action for small
problems and switches to native Wigner--Eckart contractions when magnetic
expansion or tied support becomes large.  Reduced structural routes are cached
and packed by compatible tensor block; large local problems are solved by a
matrix-free generalized Davidson iteration.  The original polarization
construction remains available as a small-system reference path.  A bounded
worker pool can be shared by every exact construction, with threaded BLAS held
to one thread inside parallel regions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations, product
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
    _left_reduced_rank_coupled_block,
    _right_reduced_rank_coupled_block,
    _factorize_left_two_site_dense_term,
    _factorize_right_two_site_dense_term,
    contract_chain_expectation,
    contract_chain_transition,
)
from pyqed.mps.nonabelian.decompose import svd_two_site
from pyqed.mps.nonabelian.mpo import expand_rank_coupled_mpo
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
    _identity_mpo_factors_for_sites_and_mpo,
)
from pyqed.mps.nonabelian.tensor import NonabelianTensor
from pyqed.tn.effective_operator import resolve_workers


def _unique(values):
    return tuple(dict.fromkeys(values))


def _strip_reconstructible_caches(root):
    """Temporarily detach nested MPO caches before checkpoint serialization.

    Rank-coupled molecular MPOs share reduced operators across many terms.  A
    sweep can populate component and environment caches several orders of
    magnitude larger than the static automaton.  They are reproducible from
    the MPO metadata and must not become part of a restart file.
    """
    saved = []
    pending = [root]
    seen = set()
    while pending:
        value = pending.pop()
        if value is None or isinstance(
            value, (str, bytes, int, float, complex, bool, np.ndarray, np.generic)
        ):
            continue
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(value, dict):
            pending.extend(value.values())
            continue
        if isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)
            continue
        attributes = getattr(value, "__dict__", None)
        if attributes is None:
            continue
        for name, child in tuple(attributes.items()):
            if name.endswith("_cache"):
                replacement = {} if isinstance(child, dict) else None
                saved.append((value, name, child))
                object.__setattr__(value, name, replacement)
            else:
                pending.append(child)
    return saved


def _restore_reconstructible_caches(saved):
    for owner, name, value in reversed(saved):
        object.__setattr__(owner, name, value)


def _sector_multiplicity(qns, sector):
    return sum(value == sector for value in qns)


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
        left = NonabelianTensor(
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
        right = NonabelianTensor(
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
        left = NonabelianTensor(
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
        right = NonabelianTensor(
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


def _interaction_graph(h1e, eri, cutoff):
    """Infer orbital pairs that share a screened Hamiltonian term."""
    h1e = np.asarray(h1e)
    nsites = int(h1e.shape[-1])
    edges = set()
    spatial_h1e = h1e if h1e.ndim == 2 else h1e[0]
    for left, right in combinations(range(nsites), 2):
        if abs(spatial_h1e[left, right]) > cutoff or abs(spatial_h1e[right, left]) > cutoff:
            edges.add((left, right))
    if eri is not None:
        eri = np.asarray(eri)
        spatial_eri = eri if eri.ndim == 4 else eri[0, 0]
        for indices in zip(*np.nonzero(np.abs(spatial_eri) > cutoff)):
            support = tuple(sorted(set(int(index) for index in indices)))
            edges.update(combinations(support, 2))
    return tuple(sorted(edges))


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

    def __init__(self, *, template, groups, basis, nparameters, nbytes):
        self.template = template
        self.groups = tuple(groups)
        self.basis = tuple(basis)
        self.nparameters = int(nparameters)
        self.nbytes = int(nbytes)
        self.route_count = int(sum(indices.size for _, indices, _, _ in self.groups))
        self.backend = "block-grouped-gemm"

    def tensor(self, coefficients):
        coefficients = np.asarray(coefficients).reshape(-1)
        if coefficients.size != self.nparameters:
            raise ValueError("Wigner--Eckart route coefficients have the wrong size.")
        data = {}
        for key, indices, matrix, shape in self.groups:
            # All parameter routes contributing to a compatible reduced block
            # are packed into one BLAS matrix-vector product.
            data[key] = (coefficients[indices] @ matrix).reshape(shape)
        return NonabelianTensor(
            data=data,
            qns=[list(values) for values in self.template.qns],
            dirs=list(self.template.dirs),
            fusion_legs=list(self.template.fusion_legs),
            metadata=dict(self.template.metadata),
        )


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
    https://doi.org/10.1103/PhysRevB.86.245124. This is a PyQED adaptation to
    the project-specific LETTA graph-tied ansatz, not a reproduction of that
    paper's MPS/NRG algorithms. It currently supports fully reduced
    spatial-orbital SU(2) states and scalar rank-coupled Hamiltonians.
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
        adaptive_bond=False,
        initial_D=None,
        ecore=0.0,
        seed=None,
        init="mps",
        base_sites=None,
        tie="auto",
        max_frontier_states=4096,
        workers=1,
        we_route_memory=256.0,
    ):
        factors = getattr(mpo, "factors", mpo)
        factors = tuple(factors)
        native_owner = (
            getattr(factors[0], "normal_complementary_owner", None)
            if factors
            else None
        )
        materialized_transition_view = False
        if (
            native_owner is not None
            and all(
                getattr(factor, "normal_complementary_plan", None) is not None
                and not factor.reduced_terms
                for factor in factors
            )
        ):
            # Production SU(2)-DMRG keeps only the compact C++ NC owner. LETTA
            # also needs arbitrary bra/ket transition elements, so request the
            # exact integral-backed reduced carrier without changing DMRG.
            materializer = getattr(mpo, "materialize_transition_factors", None)
            if materializer is None:
                raise TypeError(
                    "A lightweight native SU(2) MPO needs an integral-backed "
                    "materialize_transition_factors() method for LETTA."
                )
            factors = tuple(materializer())
            for factor in factors:
                object.__setattr__(
                    factor,
                    "normal_complementary_force_contextual_routes",
                    True,
                )
            materialized_transition_view = True
        self._native_hamiltonian_owner = native_owner
        self._su2_moving_environment = (
            getattr(mpo, "moving_environment", None) or native_owner
        )
        self._complementary_operators = getattr(
            mpo, "complementary_operators", None
        )
        self._materialized_transition_view = materialized_transition_view
        self.mpo = tuple(factors)
        self._component_mpo = None
        self.nsites = len(self.mpo)
        if self.nsites < 2:
            raise ValueError("NonAbelianFrontierLETTA requires at least two sites.")
        if not isinstance(D, Integral) or isinstance(D, (bool, np.bool_)) or int(D) < 1:
            raise ValueError("D must be a positive reduced multiplet dimension.")
        self.D = int(D)
        self.adaptive_bond = bool(adaptive_bond)
        if initial_D is None:
            initial_D = min(self.D, 2) if self.adaptive_bond else self.D
        if (
            not isinstance(initial_D, Integral)
            or isinstance(initial_D, (bool, np.bool_))
            or int(initial_D) < 1
            or int(initial_D) > self.D
        ):
            raise ValueError("initial_D must be between one and D.")
        self.initial_D = int(initial_D)
        self._growth_rng = np.random.default_rng(seed)
        self.workers = resolve_workers(workers)
        self.we_route_memory = float(we_route_memory)
        if self.we_route_memory <= 0.0:
            raise ValueError("we_route_memory must be positive in MiB.")
        self._we_route_limit_bytes = int(self.we_route_memory * 1024**2)
        self._we_route_cache = {}
        self._we_route_cache_hits = 0
        self._we_route_cache_misses = 0
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
        self.ecore = float(getattr(mpo, "ecore", ecore))
        self._mpo_includes_core_energy = bool(
            getattr(mpo, "info", {}).get("includes_core_energy", False)
        ) and not materialized_transition_view
        self._energy_shift = (
            0.0 if self._mpo_includes_core_energy else self.ecore
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
                bond_multiplicity=self.initial_D,
                seed=seed,
            )
        self._base_sites = self._validate_base_sites(base_sites)
        largest_initial_multiplicity = max(
            (
                _sector_multiplicity(site.qns[2], sector)
                for site in self._base_sites[:-1]
                for sector in _unique(site.qns[2])
            ),
            default=1,
        )
        if largest_initial_multiplicity > self.D:
            raise ValueError(
                "base_sites contain a reduced bond multiplicity larger than D."
            )
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
        self.warmup_history = []
        self._widest_pair_warmup_done = False
        self.energy = self.expectation()
        self.converged = False
        self.success = None
        self.message = "initialized reduced non-Abelian FrontierLETTA"
        self.is_native_su2 = True
        self.local_environment_backend = "block_sparse_components"
        self.has_fully_reduced_local_operator = True

    def _expanded_component_mpo(self):
        """Materialize the magnetic-component fallback only when requested."""
        if self._component_mpo is None:
            self._component_mpo = tuple(
                expand_rank_coupled_mpo(core) for core in self.mpo
            )
        return self._component_mpo

    def __deepcopy__(self, memo):
        """Copy variational state while sharing the immutable native owner."""
        import copy

        duplicate = type(self).__new__(type(self))
        memo[id(self)] = duplicate
        shared = {
            "hamiltonian",
            "_native_hamiltonian_owner",
            "_su2_moving_environment",
            "_complementary_operators",
        }
        for name, value in self.__dict__.items():
            if name == "_solver_executor":
                continue
            object.__setattr__(
                duplicate,
                name,
                value if name in shared else copy.deepcopy(value, memo),
            )
        object.__setattr__(
            duplicate,
            "_solver_executor",
            (
                ThreadPoolExecutor(
                    max_workers=duplicate.workers,
                    thread_name_prefix="su2-letta-local",
                )
                if duplicate.workers > 1
                else None
            ),
        )
        return duplicate

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
        """Build the fully reduced qchem MPO and its SU(2)-LETTA state."""
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
            nelec=nelec,
            spin=spin,
            ecore=ecore,
        )
        if graph is None:
            graph = _interaction_graph(h1e, eri, cutoff)
        state = cls(
            hamiltonian,
            nelec=nelec,
            spin=spin,
            graph=graph,
            D=D,
            ecore=ecore,
            **kwargs,
        )
        state.hamiltonian = hamiltonian
        state._hamiltonian_recipe = {
            "kind": "spatial_integrals",
            "h1e": np.array(h1e, copy=True),
            "eri": None if eri is None else np.array(eri, copy=True),
            "cutoff": float(cutoff),
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
                (_sector_multiplicity(site.qns[2], sector) for sector in _unique(site.qns[2])),
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
            if not isinstance(site, NonabelianTensor) or site.rank != 3:
                raise TypeError("base_sites must contain rank-3 NonabelianTensor objects.")
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
                multiplicity = _sector_multiplicity(site.qns[1], sector)
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
            if tuple(left.qns[2]) != tuple(right.qns[0]):
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

        The gauge remains inside the LETTA ansatz when the bond frontier
        contains only the next physical-sector label, as in an NN tie graph.
        Rank-deficient conditional blocks are left unchanged.
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
            sector: _sector_multiplicity(base.qns[axis], sector)
            for sector in _unique(base.qns[axis])
        }

    def reduced_bond_multiplicities(self, bond):
        """Return the multiplet count of every sector on an internal bond."""
        bond = int(bond)
        if bond < 0 or bond >= self.nsites - 1:
            raise IndexError(f"bond {bond} is outside a chain of length {self.nsites}.")
        return self._bond_sector_dims(bond, 2)

    @staticmethod
    def _pad_reduced_site_axis(site, axis, additions):
        qns = [list(values) for values in site.qns]
        for sector in _unique(site.qns[axis]):
            qns[axis].extend([sector] * int(additions.get(sector, 0)))
        data = {}
        for key, block in site.data.items():
            amount = int(additions.get(key[axis], 0))
            padding = [(0, 0)] * np.asarray(block).ndim
            padding[axis] = (0, amount)
            data[key] = np.pad(np.asarray(block), padding)
        metadata = dict(site.metadata or {})
        for key in tuple(metadata):
            if key.startswith("_rank_coupled_site_entries_by_"):
                metadata.pop(key, None)
        return NonabelianTensor(
            data=data,
            qns=qns,
            dirs=list(site.dirs),
            fusion_legs=list(site.fusion_legs),
            metadata=metadata,
        )

    def _grow_reduced_bond(
        self,
        bond,
        *,
        growth=1,
        required=None,
        seed_scale=1.0e-3,
    ):
        """Expand one reduced multiplet bond while preserving the wavefunction."""
        bond = int(bond)
        growth = int(growth)
        if growth < 1:
            raise ValueError("bond growth must be positive.")
        current = self.reduced_bond_multiplicities(bond)
        required = {} if required is None else dict(required)
        additions = {
            sector: min(
                growth,
                self.D - int(dimension),
                max(
                    int(required.get(sector, self.D)) - int(dimension),
                    0,
                ),
            )
            for sector, dimension in current.items()
            if int(dimension) < self.D
        }
        additions = {sector: value for sector, value in additions.items() if value > 0}
        if not additions:
            return {
                "bond": bond,
                "grown": False,
                "before": dict(current),
                "after": dict(current),
                "added_multiplets": 0,
            }

        base_sites = list(self._base_sites)
        base_sites[bond] = self._pad_reduced_site_axis(
            base_sites[bond], 2, additions
        )
        base_sites[bond + 1] = self._pad_reduced_site_axis(
            base_sites[bond + 1], 0, additions
        )

        left = {}
        for key, block in self.tensors[bond].items():
            amount = int(additions.get(key[2], 0))
            arr = np.asarray(block)
            expanded = np.zeros(
                arr.shape[:2] + (arr.shape[2] + amount,),
                dtype=arr.dtype,
            )
            expanded[..., : arr.shape[2]] = arr
            if amount:
                scale = float(seed_scale) * max(
                    float(np.linalg.norm(arr)) / max(np.sqrt(arr.size), 1.0),
                    1.0,
                )
                random = self._growth_rng.normal(
                    scale=scale,
                    size=arr.shape[:2] + (amount,),
                )
                expanded[..., arr.shape[2] :] = np.asarray(
                    random,
                    dtype=arr.dtype,
                )
            left[key] = expanded

        right = {}
        for key, block in self.tensors[bond + 1].items():
            amount = int(additions.get(key[0], 0))
            arr = np.asarray(block)
            expanded = np.zeros(
                (arr.shape[0] + amount,) + arr.shape[1:],
                dtype=arr.dtype,
            )
            expanded[: arr.shape[0], ...] = arr
            right[key] = expanded

        self._base_sites = tuple(base_sites)
        self.tensors[bond] = left
        self.tensors[bond + 1] = right
        after = self.reduced_bond_multiplicities(bond)
        return {
            "bond": bond,
            "grown": True,
            "before": dict(current),
            "after": dict(after),
            "added_multiplets": int(sum(additions.values())),
        }

    def materialize_site(self, site):
        """Materialize one tied tensor as a reduced SU(2) MPS site."""
        site = int(site)
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
            # Retain structural zero blocks.  Their fixed layout lets the
            # projected optimizer build one exact linear embedding for all
            # tied parameter directions, including directions that are zero
            # in the current iterate.
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
        return NonabelianTensor(
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

    def materialize(self):
        """Return the exact unfolded reduced SU(2) MPS."""
        return [self.materialize_site(site) for site in range(self.nsites)]

    @property
    def state(self):
        from pyqed.mps.nonabelian.mps import MPS

        return MPS.from_sites(self.materialize(), target_sector=self.target_sector)

    def close(self):
        """Release the bounded local-contraction worker pool."""
        executor = getattr(self, "_solver_executor", None)
        self._solver_executor = None
        if executor is not None:
            executor.shutdown(wait=True)

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
        if mpo is None:
            shift = self._energy_shift
        else:
            shift = (
                0.0
                if bool(
                    getattr(mpo, "info", {}).get(
                        "includes_core_energy",
                        False,
                    )
                )
                else float(getattr(mpo, "ecore", 0.0))
            )
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
        reduced_block_bytes = 0
        for core in self.mpo:
            for blocks in getattr(core, "_environment_reduced_block_cache", {}).values():
                reduced_block_bytes += sum(
                    np.asarray(block).nbytes for block in blocks.values()
                )
        return int(tensor_bytes + base_bytes + route_bytes + reduced_block_bytes)

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
        recipe = getattr(self, "_hamiltonian_recipe", None)
        saved_caches = (
            [] if recipe is not None else _strip_reconstructible_caches(self.mpo)
        )
        payload = {
            "format": "pyqed-nonabelian-frontier-letta",
            "version": 3,
            # Integral-built molecular MPOs are deterministic but can contain
            # many repeated operator objects.  Store their compact recipe and
            # rebuild on restart instead of pickling the expanded automaton.
            "mpo": None if recipe is not None else self.mpo,
            "hamiltonian_recipe": recipe,
            "nelec": self.nelec,
            "spin": self.spin,
            "target_sector": self.target_sector,
            "tie": self.tie,
            "graph": self.graph,
            "D": self.D,
            "adaptive_bond": self.adaptive_bond,
            "initial_D": self.initial_D,
            "growth_rng_state": self._growth_rng.bit_generator.state,
            "ecore": self.ecore,
            "base_sites": self._base_sites,
            "tensors": self.tensors,
            "history": self.history,
            "warmup_history": self.warmup_history,
            "widest_pair_warmup_done": self._widest_pair_warmup_done,
            "energy": self.energy,
            "converged": self.converged,
            "success": self.success,
            "message": self.message,
            "we_route_memory": self.we_route_memory,
        }
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            _restore_reconstructible_caches(saved_caches)
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
            and payload.get("version") in {2, 3}
        )
        if not (legacy or current):
            raise ValueError("unsupported SU(2)-LETTA checkpoint format.")
        recipe = payload.get("hamiltonian_recipe")
        hamiltonian = None
        mpo = payload["mpo"]
        if recipe is not None:
            if recipe.get("kind") != "spatial_integrals":
                raise ValueError("unsupported SU(2)-LETTA Hamiltonian recipe.")
            from pyqed.qchem.dmrg.backends.reduced import (
                build_spatial_reduced_hamiltonian_mpo,
            )

            eri = recipe.get("eri")
            eri_for_builder = eri
            if eri is not None and np.asarray(eri).ndim == 4:
                eri_for_builder = np.asarray(eri)[None, None, ...]
            hamiltonian = build_spatial_reduced_hamiltonian_mpo(
                recipe["h1e"],
                eri=eri_for_builder,
                cutoff=float(recipe.get("cutoff", 1.0e-10)),
                fully_reduced=True,
                nelec=payload["nelec"],
                spin=payload["spin"],
                ecore=payload["ecore"],
            )
            mpo = hamiltonian
        state = cls(
            mpo,
            nelec=payload["nelec"],
            spin=payload["spin"],
            target_sector=payload.get("target_sector"),
            graph=payload["graph"],
            D=payload["D"],
            adaptive_bond=payload.get("adaptive_bond", False),
            initial_D=payload.get("initial_D"),
            ecore=payload["ecore"],
            base_sites=payload["base_sites"],
            tie=payload.get("tie", "physical"),
            init="mps",
            workers=workers,
            we_route_memory=payload.get("we_route_memory", 256.0),
        )
        if recipe is not None:
            state.hamiltonian = hamiltonian
            state._hamiltonian_recipe = recipe
        state.tensors = [
            {key: np.array(value, copy=True) for key, value in tensor.items()}
            for tensor in payload["tensors"]
        ]
        state.history = list(payload.get("history", ()))
        state.warmup_history = list(payload.get("warmup_history", ()))
        state._widest_pair_warmup_done = bool(
            payload.get("widest_pair_warmup_done", False)
        )
        if "energy" in payload:
            state.energy = float(payload["energy"])
        else:
            state.energy = float(state.expectation())
        if payload.get("growth_rng_state") is not None:
            state._growth_rng.bit_generator.state = payload["growth_rng_state"]
        state.converged = bool(payload.get("converged", False))
        state.success = payload.get("success")
        state.message = str(payload.get("message", "restored SU(2)-LETTA checkpoint"))
        return state

    def _pack_site(self, site):
        arrays = [self.tensors[site][key].reshape(-1) for key in self._tensor_keys[site]]
        if not arrays:
            return np.zeros(0)
        return np.concatenate(arrays)

    def _set_site_vector(self, site, vector):
        vector = np.asarray(vector)
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
        """Compile the structural tied-parameter map once for one site."""
        site = int(site)
        current = self._pack_site(site)
        unit = np.zeros(current.size, dtype=np.result_type(current.dtype, float))
        rows = {}
        template = None
        try:
            for parameter in range(current.size):
                unit[parameter] = 1.0
                self._set_site_vector(site, unit)
                local = self.materialize_site(site)
                if template is None:
                    template = local
                for key, block in local.data.items():
                    array = np.asarray(block)
                    if np.any(array):
                        rows.setdefault(key, []).append(
                            (int(parameter), np.array(array, copy=True))
                        )
                unit[parameter] = 0.0
        finally:
            self._set_site_vector(site, current)
        if template is None:
            raise np.linalg.LinAlgError("the local Wigner--Eckart route is empty.")

        groups = []
        basis_data = [dict() for _ in range(current.size)]
        nbytes = 0
        for key in template.data:
            records = rows.get(key, ())
            if not records:
                continue
            indices = np.asarray([record[0] for record in records], dtype=np.intp)
            shape = np.asarray(records[0][1]).shape
            matrix = np.ascontiguousarray(
                np.stack([np.asarray(record[1]).reshape(-1) for record in records])
            )
            nbytes += int(indices.nbytes + matrix.nbytes)
            if nbytes > self._we_route_limit_bytes:
                raise MemoryError(
                    "packed Wigner--Eckart routes require more than "
                    f"{self.we_route_memory:g} MiB; increase we_route_memory."
                )
            groups.append((key, indices, matrix, shape))
            for row, parameter in enumerate(indices):
                basis_data[int(parameter)][key] = matrix[row].reshape(shape)

        basis = tuple(
            NonabelianTensor(
                data=data,
                qns=[list(values) for values in template.qns],
                dirs=list(template.dirs),
                fusion_legs=list(template.fusion_legs),
                metadata=dict(template.metadata),
            )
            for data in basis_data
        )
        return _WignerEckartRoutePlan(
            template=template,
            groups=groups,
            basis=basis,
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

    def _select_local_solver(
        self,
        site,
        bond,
        *,
        requested,
        max_local_parameters,
        auto_we_min_parameters,
        auto_we_min_expansion,
    ):
        requested = str(requested).lower().replace("-", "_")
        if requested in {"we", "wigner", "fully_reduced"}:
            requested = "wigner_eckart"
        if requested == "projected":
            requested = "wigner_eckart"
        if requested != "auto":
            return requested
        return "wigner_eckart"

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
        if len(routes.basis) != current.size:
            raise RuntimeError(
                "Cached Wigner--Eckart basis does not match the local parameter space."
            )
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

    def _projected_local_matrices(
        self,
        site,
        *,
        sites,
        h_sweep,
        n_sweep,
        bond,
        support_rtol=1.0e-13,
        max_local_parameters=128,
        executor=None,
    ):
        """Build ``P† H_eff P`` and ``P† N_eff P`` for one tied site."""
        template = merge_mps_sites(sites[bond], sites[bond + 1])
        _, layout = pack_two_site_state(template)
        current, embedding = self._local_pair_embedding(site, sites, bond, layout)

        column_norms = np.sum(np.abs(embedding) ** 2, axis=0).real
        scale = max(float(np.max(column_norms, initial=0.0)), 1.0)
        structural = np.flatnonzero(column_norms > float(support_rtol) * scale)
        if structural.size == 0:
            raise np.linalg.LinAlgError("local SU2LETTA parameter embedding is null.")

        h_operator = h_sweep.bond_operator(bond, template)
        n_operator = n_sweep.bond_operator(bond, template)
        h_action = self._resolve_packed_action(h_operator, template, layout)
        n_action = self._resolve_packed_action(n_operator, template, layout)
        embedded = embedding[:, structural]
        n_columns = self._apply_packed_columns(
            n_action,
            embedded,
            executor=executor,
        )
        metric = embedded.conj().T @ n_columns
        metric = 0.5 * (metric + metric.conj().T)
        diagonal = np.real(np.diag(metric))
        scale = max(float(np.max(np.abs(diagonal), initial=0.0)), 1.0)
        active = np.flatnonzero(diagonal > float(support_rtol) * scale)
        support = structural[active]
        if support.size == 0:
            raise np.linalg.LinAlgError("local SU2LETTA parameter support is null.")
        if support.size > int(max_local_parameters):
            raise MemoryError(
                f"local SU2LETTA support has {support.size} parameters; "
                "the projected solver is limited to "
                f"{int(max_local_parameters)}."
            )

        embedded = embedded[:, active]
        metric = metric[np.ix_(active, active)]
        h_columns = self._apply_packed_columns(
            h_action,
            embedded,
            executor=executor,
        )
        hamiltonian = embedded.conj().T @ h_columns
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
        return current, support, hamiltonian, metric

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

    def _retract_reduced_pair(
        self,
        bond,
        target,
        layout,
        *,
        cutoff=1.0e-10,
        maxiter=8,
        max_parameters=4096,
        metric_action=None,
        right_first=False,
    ):
        """Retract a reduced two-site target into the tied factors."""
        bond = int(bond)
        target = np.asarray(target, dtype=complex).reshape(-1)
        target_norm = max(float(np.linalg.norm(target)), np.finfo(float).tiny)
        if metric_action is None:
            physical_target_norm_squared = target_norm**2
        else:
            physical_metric_target = np.asarray(metric_action(target))
            physical_target_norm_squared = max(
                float(np.real(np.vdot(target, physical_metric_target))),
                np.finfo(float).tiny,
            )

        def errors(retracted):
            residual = np.asarray(retracted) - target
            parameter_error = float(np.linalg.norm(residual) / target_norm)
            if metric_action is None:
                return parameter_error, parameter_error
            metric_retracted = np.asarray(metric_action(retracted))
            retracted_norm_squared = max(
                float(np.real(np.vdot(retracted, metric_retracted))),
                0.0,
            )
            if retracted_norm_squared <= np.finfo(float).tiny:
                return 1.0, parameter_error
            overlap = np.vdot(retracted, physical_metric_target)
            fidelity = min(
                max(
                    float(abs(overlap) ** 2)
                    / (physical_target_norm_squared * retracted_norm_squared),
                    0.0,
                ),
                1.0,
            )
            return (
                float(np.sqrt(max(1.0 - fidelity, 0.0))),
                parameter_error,
            )

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
        best_error, best_parameter_error = errors(initial_retracted)
        iterations = 0
        support_sizes = [0, 0]

        try:
            for iteration in range(max(1, int(maxiter))):
                # State-preserving bond growth seeds the new columns on the
                # left tensor and leaves the matching right rows at zero.  A
                # left-first ALS step would therefore classify the seeds as
                # unsupported and erase them before the right tensor can use
                # them.  Start on the zero-padded side after growth so the new
                # multiplets become live physical directions.
                site_order = (
                    (bond + 1, bond) if right_first else (bond, bond + 1)
                )
                for site in site_order:
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
                    if metric_action is None:
                        metric_embedding = None
                        column_norms = np.sum(np.abs(embedding) ** 2, axis=0).real
                    else:
                        metric_embedding = self._apply_packed_columns(
                            metric_action,
                            embedding,
                            executor=self._solver_executor,
                        )
                        column_norms = np.real(
                            np.sum(embedding.conj() * metric_embedding, axis=0)
                        )
                    scale = max(float(np.max(column_norms, initial=0.0)), 1.0)
                    support = np.flatnonzero(column_norms > float(cutoff) * scale)
                    if support.size == 0:
                        continue
                    if metric_embedding is None:
                        coefficients, *_ = np.linalg.lstsq(
                            embedding[:, support],
                            target,
                            rcond=float(cutoff),
                        )
                    else:
                        design = embedding[:, support]
                        metric_design = metric_embedding[:, support]
                        gram = design.conj().T @ metric_design
                        gram = 0.5 * (gram + gram.conj().T)
                        values, vectors = np.linalg.eigh(gram)
                        gram_scale = max(
                            float(np.max(np.abs(values), initial=0.0)),
                            1.0,
                        )
                        keep = values > float(cutoff) * gram_scale
                        if not np.any(keep):
                            continue
                        rhs = design.conj().T @ physical_metric_target
                        coefficients = vectors[:, keep] @ (
                            (vectors[:, keep].conj().T @ rhs) / values[keep]
                        )
                    candidate = np.zeros(
                        current.size,
                        dtype=np.result_type(current.dtype, coefficients.dtype),
                    )
                    candidate[support] = coefficients
                    if np.linalg.norm(candidate) <= float(cutoff):
                        continue
                    self._set_site_vector(site, candidate)
                    support_sizes[site - bond] = int(support.size)

                sites = self.materialize()
                if isinstance(layout, _ChannelResolvedPairSpace):
                    retracted = layout.pack_sites(sites[bond], sites[bond + 1])
                else:
                    merged = merge_mps_sites(sites[bond], sites[bond + 1])
                    retracted, _ = pack_two_site_state(merged, layout=layout)
                error, parameter_error = errors(retracted)
                iterations = iteration + 1
                if error < best_error:
                    best_error = error
                    best_parameter_error = parameter_error
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
            "parameter_retraction_error": float(best_parameter_error),
            "iterations": int(iterations),
            "support": tuple(support_sizes),
            "fixed_reduced_bond_dim": int(
                max(self.reduced_bond_multiplicities(bond).values())
            ),
        }

    def _channel_resolved_pair_action(
        self,
        bond,
        sites,
        factors,
        space,
        *,
        transition_weights=None,
        chain=None,
        left_env=None,
        right_env=None,
    ):
        """Compile the exact reduced two-site action with an explicit mid sector."""
        if chain is None:
            chain = BlockSparseEnvironmentChain.build(sites, factors)
        if left_env is None:
            left_env = chain.left_envs[bond]
        if right_env is None:
            right_env = chain.right_envs[bond + 1]
        w1 = chain.mpo_factors[bond]
        w2 = chain.mpo_factors[bond + 1]
        transitions = [[] for _ in space.entries]
        left_cache = {}
        right_cache = {}
        left_factor_cache = {}
        right_factor_cache = {}
        left_factor_uses = 0
        right_factor_uses = 0

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
                    if right_index in f_blocks:
                        right_by_middle.setdefault(int(middle_index), []).append(
                            (int(right_index), np.asarray(block))
                        )
                for (left_index, middle_index), left_block in left_reduced.items():
                    if left_index not in e_blocks:
                        continue
                    left_factor_key = (
                        left_key,
                        int(left_index),
                        int(middle_index),
                    )
                    left_factor = left_factor_cache.get(left_factor_key)
                    if left_factor is None:
                        left_factor = _factorize_left_two_site_dense_term(
                            np.asarray(e_blocks[left_index]),
                            np.asarray(left_block),
                        )
                        left_factor_cache[left_factor_key] = left_factor
                    left_factor_uses += 1
                    for right_index, right_block in right_by_middle.get(
                        int(middle_index), ()
                    ):
                        right_factor_key = (
                            right_key,
                            int(middle_index),
                            int(right_index),
                        )
                        right_factor = right_factor_cache.get(right_factor_key)
                        if right_factor is None:
                            right_factor = _factorize_right_two_site_dense_term(
                                right_block,
                                np.asarray(f_blocks[right_index]),
                            )
                            right_factor_cache[right_factor_key] = right_factor
                        right_factor_uses += 1
                        transitions[in_index].append(
                            (out_index, left_factor, right_factor)
                        )

        dtype = np.result_type(
            *(np.asarray(block).dtype for core in factors for block in getattr(core, "data", {}).values()),
            *(np.asarray(block).dtype for site in sites for block in site.data.values()),
        )

        # Sum all MPO-channel paths connecting the same pair of reduced state
        # blocks.  Molecular automata can contribute hundreds of thousands of
        # paths to only a modest number of block pairs.  Applying every path as
        # a separate pair of tensordots inside each Davidson matvec is vastly
        # more expensive than compiling their exact block matrix once.
        path_transition_count = int(sum(len(items) for items in transitions))
        compiled = [[] for _ in space.entries]
        compiled_nbytes = 0
        compiled_count = 0
        for in_index, items in enumerate(transitions):
            by_output = {}
            for out_index, left_factor, right_factor in items:
                weight = (
                    1.0
                    if transition_weights is None
                    else transition_weights.get((in_index, out_index), 0.0)
                )
                if weight == 0:
                    continue
                kernel = np.einsum(
                    "abmui,mcdvj->auvcbijd",
                    np.asarray(left_factor),
                    np.asarray(right_factor),
                    optimize=False,
                ).reshape(space.entries[out_index].size, space.entries[in_index].size)
                if weight != 1.0:
                    kernel = np.asarray(weight) * kernel
                current = by_output.get(int(out_index))
                if current is None:
                    by_output[int(out_index)] = np.array(kernel, copy=True)
                else:
                    current += kernel
            for out_index, kernel in by_output.items():
                if np.any(kernel):
                    compiled[in_index].append((out_index, kernel))
                    compiled_nbytes += int(kernel.nbytes)
                    compiled_count += 1
        # Release the path-level factor references before Davidson starts.
        left_factor_count = int(len(left_factor_cache))
        right_factor_count = int(len(right_factor_cache))
        transitions = None
        left_factor_cache.clear()
        right_factor_cache.clear()

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
                packed_block = np.asarray(block).reshape(-1)
                for out_index, kernel in compiled[in_index]:
                    out_blocks[out_index] += (kernel @ packed_block).reshape(
                        out_blocks[out_index].shape
                    )
            return space.pack_blocks(out_blocks)

        action.backend = (
            "channel-resolved-rank-coupled-factorized"
            if chain.rank_coupled
            else "channel-resolved-block-factorized"
        )
        action.transition_count = path_transition_count
        action.compiled_transition_count = int(compiled_count)
        action.compiled_transition_nbytes = int(compiled_nbytes)
        action.left_factor_count = left_factor_count
        action.right_factor_count = right_factor_count
        action.left_factor_uses = int(left_factor_uses)
        action.right_factor_uses = int(right_factor_uses)
        action.entry_transitions = tuple(
            sorted(
                {
                    (in_index, int(out_index))
                    for in_index, items in enumerate(compiled)
                    for out_index, _kernel in items
                }
            )
        )
        return action

    @staticmethod
    def _pair_sweep_environments(sweep, bond):
        """Return the live two-sided boundary environments for one sweep bond."""
        if sweep is None:
            return None, None, None
        if sweep.direction == "lr":
            return (
                sweep.chain,
                sweep.current_env,
                sweep.chain.right_envs[bond + 1],
            )
        return (
            sweep.chain,
            sweep.chain.left_envs[bond],
            sweep.current_env,
        )

    @staticmethod
    def _channel_resolved_identity_metric_blocks(space, left_env, right_env):
        """Build the exact pair norm metric from two boundary Gram maps.

        The identity MPO preserves both physical sectors and the explicit
        intermediate fusion sector.  A surviving block therefore factorizes
        into the left boundary Gram matrix, two physical identities, and the
        right boundary Gram matrix.  Constructing that Kronecker product once
        per sector block replaces a scalar contraction for every row/column.
        """
        metric_blocks = []
        for in_entry in space.entries:
            q_lk, q_p1k, q_p2k, q_rk, q_mk = in_entry.key
            for out_entry in space.entries:
                q_lb, q_p1b, q_p2b, q_rb, q_mb = out_entry.key
                if (
                    q_p1b != q_p1k
                    or q_p2b != q_p2k
                    or q_mb != q_mk
                ):
                    continue
                e_blocks = left_env.get((q_lb, q_lk))
                f_blocks = right_env.get((q_rb, q_rk))
                if e_blocks is None or f_blocks is None:
                    continue
                e_block = e_blocks.get(0)
                f_block = f_blocks.get(0)
                if e_block is None or f_block is None:
                    raise ValueError(
                        "The channel-resolved norm metric requires a scalar "
                        "identity-MPO channel."
                    )
                e_block = np.asarray(e_block)
                f_block = np.asarray(f_block)
                physical_1 = np.eye(
                    out_entry.shape[1],
                    in_entry.shape[1],
                    dtype=np.result_type(e_block, f_block),
                )
                physical_2 = np.eye(
                    out_entry.shape[2],
                    in_entry.shape[2],
                    dtype=np.result_type(e_block, f_block),
                )
                block = np.einsum(
                    "ab,pq,uv,cd->apucbqvd",
                    e_block[0],
                    physical_1,
                    physical_2,
                    f_block[0],
                    optimize=True,
                ).reshape(out_entry.size, in_entry.size)
                if np.linalg.norm(block) > 1.0e-14:
                    metric_blocks.append((out_entry, in_entry, block))
        return tuple(metric_blocks)

    @staticmethod
    def _channel_metric_whitener(layout, metric_blocks, cutoff):
        """Whiten a block-sparse pair metric without forming its dense parent."""
        entry_index = {entry.key: index for index, entry in enumerate(layout.entries)}
        adjacency = [set((index,)) for index in range(len(layout.entries))]
        for out_entry, in_entry, _block in metric_blocks:
            out_index = entry_index[out_entry.key]
            in_index = entry_index[in_entry.key]
            adjacency[out_index].add(in_index)
            adjacency[in_index].add(out_index)

        components = []
        unseen = set(range(len(layout.entries)))
        while unseen:
            seed = unseen.pop()
            component = {seed}
            pending = [seed]
            while pending:
                current = pending.pop()
                additions = adjacency[current] & unseen
                unseen.difference_update(additions)
                component.update(additions)
                pending.extend(additions)
            components.append(tuple(sorted(component)))

        block_lookup = {
            (entry_index[out_entry.key], entry_index[in_entry.key]): block
            for out_entry, in_entry, block in metric_blocks
        }
        columns = []
        component_dims = []
        component_ranks = []
        for component in components:
            entries = [layout.entries[index] for index in component]
            offsets = np.cumsum([0] + [entry.size for entry in entries])
            metric = np.zeros((offsets[-1], offsets[-1]), dtype=complex)
            for out_local, out_index in enumerate(component):
                for in_local, in_index in enumerate(component):
                    block = block_lookup.get((out_index, in_index))
                    if block is not None:
                        metric[
                            offsets[out_local] : offsets[out_local + 1],
                            offsets[in_local] : offsets[in_local + 1],
                        ] += block
            metric = 0.5 * (metric + metric.conj().T)
            values, vectors = np.linalg.eigh(metric)
            scale = max(float(np.max(np.abs(values), initial=0.0)), 1.0)
            support = values > float(cutoff) * scale
            component_dims.append(int(metric.shape[0]))
            component_ranks.append(int(np.count_nonzero(support)))
            if not np.any(support):
                continue
            local = vectors[:, support] / np.sqrt(values[support])[None, :]
            embedded = np.zeros((layout.size, local.shape[1]), dtype=complex)
            for local_index, entry in enumerate(entries):
                embedded[
                    entry.offset : entry.offset + entry.size,
                    :,
                ] = local[offsets[local_index] : offsets[local_index + 1], :]
            columns.append(embedded)
        if not columns:
            raise np.linalg.LinAlgError(
                "Channel-resolved pair metric has no positive support."
            )
        return np.concatenate(columns, axis=1), {
            "metric_components": int(len(components)),
            "metric_max_component_dim": int(max(component_dims, default=0)),
            "metric_component_ranks": tuple(component_ranks),
        }

    def _exact_reduced_pair_matrices(
        self,
        bond,
        sites,
        merged,
        layout,
        *,
        h_sweep=None,
        n_sweep=None,
    ):
        """Build an exact pair pencil from channel-resolved pair actions."""
        h_action, n_action, _diagnostics = self._reduced_pair_transition_actions(
            bond,
            sites,
            merged,
            layout,
            h_sweep=h_sweep,
            n_sweep=n_sweep,
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
        h_sweep=None,
        n_sweep=None,
    ):
        """Return exact matrix-free actions with explicit intermediate channels.

        A merged rank-4 reduced tensor does not itself label the intermediate
        MPS fusion channel.  Splitting each vector into adjacent rank-3 tensors
        before contraction retains that channel and makes the action exact for
        reduced bond dimensions greater than one.
        """
        dimension = int(getattr(layout, "size", sum(entry.size for entry in layout)))
        identity = _identity_mpo_factors_for_sites_and_mpo(sites, self.mpo)
        if isinstance(layout, _ChannelResolvedPairSpace):
            h_chain, h_left, h_right = self._pair_sweep_environments(
                h_sweep, bond
            )
            if h_chain is None:
                h_chain = BlockSparseEnvironmentChain.build(sites, self.mpo)
                h_left = h_chain.left_envs[bond]
                h_right = h_chain.right_envs[bond + 1]
            h_action = self._channel_resolved_pair_action(
                bond,
                sites,
                self.mpo,
                layout,
                chain=h_chain,
                left_env=h_left,
                right_env=h_right,
            )
            n_chain, n_left, n_right = self._pair_sweep_environments(
                n_sweep, bond
            )
            if n_chain is None:
                n_chain = BlockSparseEnvironmentChain.build(sites, identity)
                n_left = n_chain.left_envs[bond]
                n_right = n_chain.right_envs[bond + 1]
            metric_blocks = self._channel_resolved_identity_metric_blocks(
                layout,
                n_left,
                n_right,
            )

            def n_action(vector):
                vector = np.asarray(vector)
                out = np.zeros(dimension, dtype=np.result_type(vector.dtype, complex))
                for out_entry, in_entry, block in metric_blocks:
                    out[out_entry.offset : out_entry.offset + out_entry.size] += (
                        block
                        @ vector[in_entry.offset : in_entry.offset + in_entry.size]
                    )
                return out

            n_action.backend = "channel-resolved-exact-block-metric"
            n_action.transition_count = len(metric_blocks)
            n_action.metric_blocks = metric_blocks

            diagnostics = {
                "backend": "channel_resolved_rank_coupled_factorized",
                "dimension": dimension,
                "hamiltonian_direction": (
                    "two_sided" if h_sweep is None else h_sweep.direction
                ),
                "metric_direction": (
                    "two_sided" if n_sweep is None else n_sweep.direction
                ),
                "hamiltonian_cached_sites": int(self.nsites - 2),
                "metric_cached_sites": int(self.nsites - 2),
                "hamiltonian_transitions": h_action.transition_count,
                "hamiltonian_compiled_transitions": (
                    h_action.compiled_transition_count
                ),
                "hamiltonian_compiled_transition_nbytes": (
                    h_action.compiled_transition_nbytes
                ),
                "hamiltonian_left_factors": h_action.left_factor_count,
                "hamiltonian_right_factors": h_action.right_factor_count,
                "hamiltonian_left_factor_uses": h_action.left_factor_uses,
                "hamiltonian_right_factor_uses": h_action.right_factor_uses,
                "metric_transitions": n_action.transition_count,
                "metric_blocks": len(metric_blocks),
                "metric_nbytes": int(sum(block.nbytes for _, _, block in metric_blocks)),
            }
            return h_action, n_action, diagnostics

        h_plan = AdjacentPairTransitionPlan.build(sites, self.mpo, bond)
        n_plan = AdjacentPairTransitionPlan.build(sites, identity, bond)

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
        retraction_relax_sweeps=0,
        max_retraction_parameters=4096,
        accept_tol=1.0e-10,
        davidson_tol=1.0e-10,
        davidson_maxiter=80,
        davidson_max_space=32,
        dense_dim=0,
        growth_truncation_tol=0.05,
        bond_growth=1,
        sites=None,
        h_sweep=None,
        n_sweep=None,
    ):
        """Optimize one adjacent pair in the native reduced SU(2) space.

        The exact rank-coupled pair eigenproblem is solved without magnetic
        expansion.  Its root is then variationally retracted to the current
        tied manifold; ``D`` is only a per-sector cap when adaptive growth is
        enabled.  ``truncation_error`` reports the physical-metric fidelity
        loss of the retraction.
        """
        bond = int(bond)
        if bond < 0 or bond >= self.nsites - 1:
            raise IndexError(f"bond {bond} is outside a chain of length {self.nsites}.")
        started = time.perf_counter()
        sites = self.materialize() if sites is None else list(sites)
        original_base_sites = (
            self._base_sites[bond],
            self._base_sites[bond + 1],
        )
        original_tensors = (
            {
                key: np.array(block, copy=True)
                for key, block in self.tensors[bond].items()
            },
            {
                key: np.array(block, copy=True)
                for key, block in self.tensors[bond + 1].items()
            },
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
                    h_sweep=h_sweep,
                    n_sweep=n_sweep,
                )
            )
            hamiltonian = metric = None
        else:
            hamiltonian, metric = self._exact_reduced_pair_matrices(
                bond,
                sites,
                merged,
                layout,
                h_sweep=h_sweep,
                n_sweep=n_sweep,
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
                "parameter_retraction_error": 0.0,
                "retraction_iterations": 0,
                "retraction_relax_sweeps": 0,
                "retraction_residual": float("inf"),
                "retraction_support": (0, 0),
                "fixed_reduced_bond_dim": int(
                    max(self.reduced_bond_multiplicities(bond).values())
                ),
                "adaptive_bond": bool(self.adaptive_bond),
                "bond_growth": (),
                "reduced_bond_multiplicities": dict(
                    self.reduced_bond_multiplicities(bond)
                ),
                "target_bond_multiplicities": {},
                "solver_info": {},
                "elapsed": float(time.perf_counter() - started),
                "message": "rejected null reduced pair metric",
            }
        before = float(
            np.real(np.vdot(initial, initial_h)) / initial_norm
            + self._energy_shift
        )
        if pair_matrix_free:
            whitener, whitening_diagnostics = self._channel_metric_whitener(
                layout,
                n_action.metric_blocks,
                cutoff,
            )
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
            solver_info.update(whitening_diagnostics)
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
            projected_residual = whitener.conj().T @ (
                target_h - local_energy * target_n
            )
            residual = float(
                np.linalg.norm(projected_residual)
                / max(
                    np.linalg.norm(whitener.conj().T @ target_h),
                    abs(local_energy)
                    * np.linalg.norm(whitener.conj().T @ target_n),
                    1.0,
                )
            )
        target_bond_multiplicities = {}
        if isinstance(layout, _ChannelResolvedPairSpace):
            target_left, _target_right = layout.split(target, cutoff=cutoff)
            target_bond_multiplicities = {
                sector: _sector_multiplicity(target_left.qns[2], sector)
                for sector in _unique(target_left.qns[2])
            }
        retraction = self._retract_reduced_pair(
            bond,
            target,
            layout,
            cutoff=cutoff,
            maxiter=retraction_maxiter,
            max_parameters=max_retraction_parameters,
            metric_action=n_action,
        )
        growth_records = []
        while self.adaptive_bond and retraction[
            "truncation_error"
        ] > float(growth_truncation_tol):
            current_multiplicities = self.reduced_bond_multiplicities(bond)
            can_grow = any(
                int(current) < min(
                    self.D,
                    int(target_bond_multiplicities.get(sector, self.D)),
                )
                for sector, current in current_multiplicities.items()
            )
            if not can_grow:
                break
            # Enlarge the best lower-rank retraction in place.  Padding is
            # exactly state preserving, so this nests the old variational
            # manifold inside the larger one and supplies a proper warm start.
            growth_record = self._grow_reduced_bond(
                bond,
                growth=bond_growth,
                required=target_bond_multiplicities,
            )
            if not growth_record["grown"]:
                break
            growth_records.append(growth_record)
            retraction = self._retract_reduced_pair(
                bond,
                target,
                layout,
                cutoff=cutoff,
                maxiter=retraction_maxiter,
                max_parameters=max_retraction_parameters,
                metric_action=n_action,
                right_first=True,
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
                + self._energy_shift
            )
        accepted = after <= before + float(accept_tol)
        if not accepted:
            base_sites = list(self._base_sites)
            base_sites[bond], base_sites[bond + 1] = original_base_sites
            self._base_sites = tuple(base_sites)
            self.tensors[bond] = original_tensors[0]
            self.tensors[bond + 1] = original_tensors[1]
            after = before
        return {
            "bond": bond,
            "sites": (bond, bond + 1),
            "energy_before": float(before),
            "energy_after": float(after),
            "local_energy": float(local_energy + self._energy_shift),
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
            "parameter_retraction_error": retraction[
                "parameter_retraction_error"
            ],
            "retraction_iterations": retraction["iterations"],
            "retraction_relax_sweeps": relaxation["iterations"],
            "retraction_residual": relaxation["max_residual"],
            "retraction_support": retraction["support"],
            "fixed_reduced_bond_dim": retraction["fixed_reduced_bond_dim"],
            "adaptive_bond": bool(self.adaptive_bond),
            "bond_growth": tuple(growth_records),
            "reduced_bond_multiplicities": dict(
                self.reduced_bond_multiplicities(bond)
            ),
            "target_bond_multiplicities": dict(target_bond_multiplicities),
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
        auto_we_min_parameters=64,
        auto_we_min_expansion=1.25,
        sites=None,
        h_sweep=None,
        n_sweep=None,
        bond=None,
        executor=None,
    ):
        """Perform one exact reduced local Rayleigh--Ritz update."""
        site = int(site)
        if executor is None:
            executor = self._solver_executor
        requested_solver = str(solver).lower().replace("-", "_")
        if bond is None:
            bond = site if site < self.nsites - 1 else site - 1
        solver = self._select_local_solver(
            site,
            bond,
            requested=requested_solver,
            max_local_parameters=max_local_parameters,
            auto_we_min_parameters=auto_we_min_parameters,
            auto_we_min_expansion=auto_we_min_expansion,
        )
        if solver not in {"projected", "wigner_eckart", "polarization"}:
            raise ValueError(
                "SU2LETTA solver must be 'auto', 'projected', "
                "'wigner_eckart', or 'polarization'."
            )
        matrix_free = False
        solver_info = {}
        if solver == "polarization":
            current, support, hamiltonian, metric = self._local_quadratic_matrices(
                site,
                support_rtol=support_rtol,
                max_local_parameters=max_local_parameters,
            )
        elif solver == "wigner_eckart":
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
                ) = self._wigner_eckart_matrix_free_problem(
                    site,
                    support_rtol=support_rtol,
                    max_parameters=max_matrix_free_parameters,
                    executor=executor,
                    sites=sites,
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
        else:
            if sites is None:
                sites = self.materialize()
            if h_sweep is None or n_sweep is None:
                identity = _identity_mpo_factors_for_sites_and_mpo(sites, self.mpo)
                direction = "lr" if site == bond else "rl"
                h_sweep = BlockSparseEnvironmentChain.build(
                    sites,
                    self._expanded_component_mpo(),
                    sweep_direction=direction,
                ).start_sweep(direction)
                n_sweep = BlockSparseEnvironmentChain.build(
                    sites, identity, sweep_direction=direction
                ).start_sweep(direction)
            current, support, hamiltonian, metric = self._projected_local_matrices(
                site,
                sites=sites,
                h_sweep=h_sweep,
                n_sweep=n_sweep,
                bond=int(bond),
                support_rtol=support_rtol,
                max_local_parameters=max_local_parameters,
                executor=executor,
            )
        current_reduced = current[support]
        if matrix_free:
            current_h = h_action(current_reduced)
            current_n = n_action(current_reduced)
            denominator = np.real(np.vdot(current_reduced, current_n))
        else:
            denominator = np.real(np.vdot(current_reduced, metric @ current_reduced))
        if denominator <= 0.0:
            raise np.linalg.LinAlgError("the current local SU2LETTA metric norm is null.")
        numerator = np.vdot(
            current_reduced,
            current_h if matrix_free else hamiltonian @ current_reduced,
        )
        before = float(np.real(numerator) / denominator) + self._energy_shift
        if matrix_free:
            local_energy, reduced, davidson_info = _solve_packed_generalized_davidson(
                current_reduced,
                h_action,
                h_diag=h_diag,
                N=n_action,
                n_diag=n_diag,
                tol=float(davidson_tol),
                tol_residual=float(davidson_tol),
                itermax=int(davidson_maxiter),
                max_space=min(int(davidson_max_space), int(support.size)),
            )
            solver_info["davidson"] = davidson_info
            overlap = np.vdot(current_reduced, n_action(reduced))
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
        self._set_site_vector(site, candidate)
        after = float(local_energy + self._energy_shift)
        accepted = after <= before + float(accept_tol)
        if not accepted:
            self._set_site_vector(site, current)
            after = before
        return {
            "site": site,
            "energy_before": float(before),
            "energy_after": float(after),
            "local_energy": float(local_energy + self._energy_shift),
            "support": int(support.size),
            "parameters": int(current.size),
            "accepted": bool(accepted),
            "native_su2": True,
            "solver": solver,
            "requested_solver": requested_solver,
            "auto_selected": requested_solver == "auto",
            "workers": int(self.workers),
            "fully_wigner_eckart_reduced": solver == "wigner_eckart",
            "matrix_free": bool(matrix_free),
            "local_linear_algebra": (
                "matrix_free_generalized_davidson"
                if matrix_free
                else "dense_generalized_eigh"
            ),
            "solver_info": solver_info,
            "magnetic_expansion_factor": float(
                self._magnetic_expansion_factor(bond)
            ),
            "environment_backend": (
                self.local_environment_backend
                if solver == "projected"
                else (
                    "wigner_eckart_reduced"
                    if solver == "wigner_eckart"
                    else "full_chain_polarization"
                )
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
        auto_we_min_parameters=64,
        auto_we_min_expansion=1.25,
        pair_cutoff=1.0e-10,
        retraction_maxiter=8,
        max_retraction_parameters=4096,
        growth_truncation_tol=0.05,
        bond_growth=1,
        widest_pair_warmup=True,
        checkpoint=None,
        checkpoint_every=1,
        reset_history=True,
        gauge=None,
        gauge_rcond=1.0e-12,
        reuse_environments=True,
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
        if algorithm not in {"one_site", "two_site"}:
            raise ValueError("SU2LETTA algorithm must be 'one_site' or 'two_site'.")
        consecutive_cycles = int(consecutive_cycles)
        if consecutive_cycles < 1:
            raise ValueError("consecutive_cycles must be positive.")
        checkpoint_every = int(checkpoint_every)
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive.")
        solver = str(solver).lower().replace("-", "_")
        if solver in {"we", "wigner", "fully_reduced"}:
            solver = "wigner_eckart"
        if solver not in {"auto", "projected", "wigner_eckart", "polarization"}:
            raise ValueError(
                "SU2LETTA solver must be 'auto', 'projected', "
                "'wigner_eckart', or 'polarization'."
            )
        if gauge is not None:
            gauge = str(gauge).lower().replace("-", "_")
        if gauge not in {None, "conditional"}:
            raise ValueError("SU2LETTA gauge must be None or 'conditional'.")
        conditional_gauge = bool(
            algorithm == "two_site"
            and gauge == "conditional"
            and self.supports_conditional_canonical_gauge
        )
        reuse_environments = bool(reuse_environments)
        # Construction and checkpoint restore leave ``self.energy`` consistent
        # with the current tensors.  Recomputing it here is especially costly
        # for a cold rank-coupled qchem MPO and merely repeats the same full
        # contraction before the first moving environment is built.
        previous = float(self.energy)
        if reset_history:
            self.history = []
        if (
            algorithm == "two_site"
            and bool(widest_pair_warmup)
            and not self._widest_pair_warmup_done
        ):
            materialized = self.materialize()
            warmup_bond = max(
                range(self.nsites - 1),
                key=lambda candidate: (
                    _ChannelResolvedPairSpace(
                        materialized[candidate],
                        materialized[candidate + 1],
                    ).size,
                    -abs(2 * candidate - (self.nsites - 2)),
                ),
            )
            warmup = self.optimize_two_sites(
                warmup_bond,
                cutoff=pair_cutoff,
                retraction_maxiter=retraction_maxiter,
                max_retraction_parameters=max_retraction_parameters,
                growth_truncation_tol=growth_truncation_tol,
                bond_growth=bond_growth,
                sites=materialized,
                **kwargs,
            )
            warmup["direction"] = "widest_pair_warmup"
            self.warmup_history.append(warmup)
            self._widest_pair_warmup_done = True
            self.energy = float(warmup["energy_after"])
            if verbose:
                print(
                    "SU2-LETTA widest-pair warmup: "
                    f"bond={warmup_bond} E={self.energy:.12f} "
                    f"trunc={warmup['truncation_error']:.3e}"
                )
        self.converged = False
        streak = 0
        for sweep in range(nsweeps):
            cycle_started = time.perf_counter()
            updates = []
            gauge_updates = []
            for direction, bonds in (
                ("lr", range(self.nsites - 1)),
                ("rl", range(self.nsites - 2, -1, -1)),
            ):
                bonds = tuple(bonds)
                if conditional_gauge:
                    center = 0 if direction == "lr" else self.nsites - 1
                    gauge_updates.extend(
                        self.canonicalize_conditional_center(
                            center, rcond=gauge_rcond
                        )
                    )
                materialized = self.materialize()
                identity = _identity_mpo_factors_for_sites_and_mpo(
                    materialized, self.mpo
                )
                if algorithm == "two_site":
                    local_solvers = {}
                    if reuse_environments:
                        h_sweep = BlockSparseEnvironmentChain.build(
                            materialized,
                            self.mpo,
                            sweep_direction=direction,
                        ).start_sweep(direction)
                        n_sweep = BlockSparseEnvironmentChain.build(
                            materialized,
                            identity,
                            sweep_direction=direction,
                        ).start_sweep(direction)
                        uses_moving_environments = True
                    else:
                        h_sweep = None
                        n_sweep = None
                        uses_moving_environments = False
                else:
                    local_solvers = {}
                    for bond in bonds:
                        site = bond if direction == "lr" else bond + 1
                        local_solvers[site] = self._select_local_solver(
                            site,
                            bond,
                            requested=solver,
                            max_local_parameters=max_local_parameters,
                            auto_we_min_parameters=auto_we_min_parameters,
                            auto_we_min_expansion=auto_we_min_expansion,
                        )
                    uses_moving_environments = any(
                        selected == "projected"
                        for selected in local_solvers.values()
                    )
                    if uses_moving_environments:
                        # The projected fallback expands only the small MPO
                        # operator-channel irreps; state legs stay reduced.
                        h_sweep = BlockSparseEnvironmentChain.build(
                            materialized,
                            self._expanded_component_mpo(),
                            sweep_direction=direction,
                        ).start_sweep(direction)
                        n_sweep = BlockSparseEnvironmentChain.build(
                            materialized,
                            identity,
                            sweep_direction=direction,
                        ).start_sweep(direction)
                    else:
                        h_sweep = None
                        n_sweep = None
                for bond in bonds:
                    if algorithm == "two_site":
                        update = self.optimize_two_sites(
                            bond,
                            cutoff=pair_cutoff,
                            retraction_maxiter=retraction_maxiter,
                            max_retraction_parameters=max_retraction_parameters,
                            growth_truncation_tol=growth_truncation_tol,
                            bond_growth=bond_growth,
                            sites=materialized,
                            h_sweep=h_sweep,
                            n_sweep=n_sweep,
                            **kwargs,
                        )
                        if conditional_gauge:
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
                        materialized[bond] = self.materialize_site(bond)
                        materialized[bond + 1] = self.materialize_site(bond + 1)
                    else:
                        site = bond if direction == "lr" else bond + 1
                        update = self.optimize_site(
                            site,
                            max_local_parameters=max_local_parameters,
                            max_matrix_free_parameters=max_matrix_free_parameters,
                            solver=local_solvers[site],
                            we_dense_dim=we_dense_dim,
                            auto_we_min_parameters=auto_we_min_parameters,
                            auto_we_min_expansion=auto_we_min_expansion,
                            sites=materialized,
                            h_sweep=h_sweep,
                            n_sweep=n_sweep,
                            bond=bond,
                            executor=self._solver_executor,
                            **kwargs,
                        )
                        update["requested_solver"] = solver
                        update["auto_selected"] = solver == "auto"
                        materialized[site] = self.materialize_site(site)
                    update["direction"] = direction
                    updates.append(update)
                    if uses_moving_environments:
                        h_sweep.advance_after_update(
                            bond, materialized[bond], materialized[bond + 1]
                        )
                        n_sweep.advance_after_update(
                            bond, materialized[bond], materialized[bond + 1]
                        )
            energy = self.expectation()
            delta = abs(energy - previous)
            residuals = []
            truncation_errors = []
            for update in updates:
                if update.get("local_residual") is not None:
                    residuals.append(float(update["local_residual"]))
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
            streak = streak + 1 if cycle_ok else 0
            record = {
                "sweep": len(self.history) + 1,
                "energy": float(energy),
                "energy_delta": float(delta),
                "max_local_residual": float(max_residual),
                "max_truncation_error": float(max_truncation),
                "rejected_updates": int(rejected),
                "accepted_updates": int(len(updates) - rejected),
                "consecutive_cycles": int(streak),
                "updates": updates,
                "complete_cycle": True,
                "symmetry": "SU2",
                "algorithm": algorithm,
                "norm": float(self.norm()),
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
                    algorithm == "two_site" and reuse_environments
                ),
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
            if checkpoint is not None and record["sweep"] % checkpoint_every == 0:
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
