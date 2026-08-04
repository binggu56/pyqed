"""NARG-like complementary-operator frontiers for graph-tied LETTA."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from opt_einsum import contract_expression
from scipy.sparse import coo_matrix, csr_matrix

from .block_mpo_frontier import BlockMPOFrontier
from .local_terms import LocalHamiltonian, LocalMPO
from .physical_blocks import hamiltonian_physical_connectivity


class _PreparedSupportPairAction:
    """Environment-bound sparse action in packed merged-pair support."""

    def __init__(
        self,
        operator,
        *,
        assembly_seconds,
        connected_blocks,
        raw_block_elements,
    ):
        self.operator = csr_matrix(operator)
        self.shape = self.operator.shape
        self.dtype = self.operator.dtype
        self.assembly_seconds = float(assembly_seconds)
        self.connected_blocks = int(connected_blocks)
        self.nonzero_elements = int(self.operator.nnz)
        stored_bytes = int(
            self.operator.data.nbytes
            + self.operator.indices.nbytes
            + self.operator.indptr.nbytes
        )
        self.stored_elements = int(
            (stored_bytes + self.dtype.itemsize - 1) // self.dtype.itemsize
        )
        self.raw_block_elements = int(raw_block_elements)
        self.peak_elements = int(self.stored_elements + self.raw_block_elements)

    def __call__(self, vectors):
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[0] != self.shape[1]:
            raise ValueError(
                "vectors must have shape (support_size, batch_size)."
            )
        return np.asarray(self.operator @ vectors)


def _right_transfer_columns(dim, local_qns):
    if local_qns is None:
        return ((None, np.arange(dim * dim, dtype=np.intp)),)
    groups = {}
    for bra, q_bra in enumerate(local_qns):
        for ket, q_ket in enumerate(local_qns):
            transfer = tuple(a - b for a, b in zip(q_bra, q_ket))
            groups.setdefault(transfer, []).append(bra * dim + ket)
    return tuple(
        (transfer, np.asarray(columns, dtype=np.intp))
        for transfer, columns in sorted(groups.items())
    )


def _two_site_matrix(term, dims):
    left_site, right_site = term.sites
    left_dim = dims[left_site]
    right_dim = dims[right_site]
    return (
        np.asarray(term.operator)
        .reshape(left_dim, right_dim, left_dim, right_dim)
        .transpose(0, 2, 1, 3)
        .reshape(left_dim * left_dim, right_dim * right_dim)
    )


def _endpoint_components(hamiltonian, local_qns):
    """Factor all two-site terms into shared right-endpoint operator bases."""
    grouped = {}
    for term in hamiltonian.terms:
        if len(term.sites) == 2:
            grouped.setdefault(term.sites[1], []).append(
                (term.sites[0], _two_site_matrix(term, hamiltonian.dims))
            )
        elif len(term.sites) > 2:
            raise NotImplementedError(
                "the renormalized frontier currently supports only one- and "
                "two-site LocalTerm objects"
            )

    result = {}
    for right_site, items in grouped.items():
        right_dim = hamiltonian.dims[right_site]
        endpoint = []
        qns = None if local_qns is None else local_qns[right_site]
        for transfer, columns in _right_transfer_columns(right_dim, qns):
            stacked = np.vstack([matrix[:, columns] for _site, matrix in items])
            if not np.any(stacked):
                continue
            _left, singular_values, right = np.linalg.svd(
                stacked,
                full_matrices=False,
            )
            scale = float(np.max(singular_values, initial=0.0))
            threshold = np.finfo(singular_values.dtype).eps * max(stacked.shape) * scale
            rank = int(np.count_nonzero(singular_values > threshold))
            for basis_index in range(rank):
                basis = right[basis_index]
                right_vector = np.zeros(
                    right_dim * right_dim,
                    dtype=right.dtype,
                )
                right_vector[columns] = basis
                starts = {}
                for left_site, matrix in items:
                    coefficient = matrix[:, columns] @ basis.conj()
                    left_dim = hamiltonian.dims[left_site]
                    left_operator = coefficient.reshape(left_dim, left_dim)
                    if float(np.max(np.abs(left_operator), initial=0.0)) > threshold:
                        starts[left_site] = left_operator
                if starts:
                    endpoint.append(
                        {
                            "transfer": transfer,
                            "basis": basis_index,
                            "right_operator": right_vector.reshape(
                                right_dim,
                                right_dim,
                            ),
                            "starts": starts,
                        }
                    )
        result[right_site] = tuple(endpoint)
    return result


def _renormalized_operator_tensors(
    hamiltonian,
    *,
    physical_sites=None,
    local_qns=None,
):
    """Build the sparse complementary-operator recurrence as local tensors."""
    if not isinstance(hamiltonian, LocalHamiltonian):
        raise TypeError("hamiltonian must be a LocalHamiltonian.")
    dims = hamiltonian.dims
    nsites = len(dims)
    if physical_sites is not None and len(tuple(physical_sites)) != nsites:
        raise ValueError("physical_sites must contain one entry per site.")
    if local_qns is not None:
        local_qns = tuple(
            tuple(tuple(int(value) for value in charge) for charge in labels)
            for labels in local_qns
        )
        if len(local_qns) != nsites or any(
            len(labels) != dim for labels, dim in zip(local_qns, dims)
        ):
            raise ValueError("local_qns are inconsistent with the Hamiltonian.")
    one_site = {
        term.sites[0]: np.asarray(term.operator)
        for term in hamiltonian.terms
        if len(term.sites) == 1 and np.any(term.operator != 0)
    }
    components = _endpoint_components(hamiltonian, local_qns)
    if hamiltonian.constant == 0.0 and not one_site and not any(components.values()):
        if nsites == 1:
            tensors = (np.zeros((1, 1, dims[0], dims[0]), dtype=hamiltonian.dtype),)
            transition_count = 0
            bond_dims = (1, 1)
        else:
            tensors = []
            for site, dim in enumerate(dims):
                identity = np.eye(dim, dtype=hamiltonian.dtype)
                if site == 0:
                    tensor = np.stack((identity, -identity), axis=0)[None, :]
                elif site == nsites - 1:
                    tensor = np.stack((identity, identity), axis=0)[:, None]
                else:
                    tensor = np.zeros((2, 2, dim, dim), dtype=hamiltonian.dtype)
                    tensor[0, 0] = identity
                    tensor[1, 1] = identity
                tensors.append(tensor)
            tensors = tuple(tensors)
            transition_count = 2 * nsites
            bond_dims = (1,) + (2,) * (nsites - 1) + (1,)
        return tensors, {
            "bond_dims": bond_dims,
            "max_bond_dim": max(bond_dims),
            "transition_count": transition_count,
            "endpoint_components": 0,
            "representation": "renormalized_complementary_operators",
        }

    idle = ("idle",)
    done = ("done",)

    def complement(right_site, component):
        return (
            "complement",
            right_site,
            component["transfer"],
            component["basis"],
        )

    states = [(idle,)]
    for cut in range(1, nsites):
        cut_states = [idle, done]
        for right_site, endpoint in sorted(components.items()):
            for component in endpoint:
                starts = tuple(sorted(component["starts"]))
                if any(left_site < cut for left_site in starts) and cut <= right_site:
                    cut_states.append(complement(right_site, component))
        states.append(tuple(cut_states))
    states.append((done,))
    state_maps = tuple(
        {state: index for index, state in enumerate(cut_states)}
        for cut_states in states
    )

    dtype = np.dtype(
        np.result_type(
            hamiltonian.dtype,
            *[
                component["right_operator"].dtype
                for endpoint in components.values()
                for component in endpoint
            ],
        )
    )
    tensors = []
    transition_count = 0
    for site, dim in enumerate(dims):
        left_states = state_maps[site]
        right_states = state_maps[site + 1]
        tensor = np.zeros(
            (len(left_states), len(right_states), dim, dim),
            dtype=dtype,
        )
        identity = np.eye(dim, dtype=dtype)

        def add(left_state, right_state, operator):
            nonlocal transition_count
            if left_state not in left_states or right_state not in right_states:
                return
            pair = (left_states[left_state], right_states[right_state])
            if not np.any(tensor[pair]):
                transition_count += 1
            tensor[pair] += operator

        if idle in left_states and idle in right_states:
            add(idle, idle, identity)
        if done in left_states and done in right_states:
            add(done, done, identity)
        if site == 0 and hamiltonian.constant != 0.0:
            add(idle, done, hamiltonian.constant * identity)
        if site in one_site:
            add(idle, done, one_site[site])

        for right_site, endpoint in components.items():
            for component in endpoint:
                complement_state = complement(right_site, component)
                if site in component["starts"]:
                    add(
                        idle,
                        complement_state,
                        component["starts"][site],
                    )
                if complement_state in left_states and complement_state in right_states:
                    add(complement_state, complement_state, identity)
                if site == right_site:
                    add(
                        complement_state,
                        done,
                        component["right_operator"],
                    )
        tensors.append(tensor)

    diagnostics = {
        "bond_dims": tuple(len(cut_states) for cut_states in states),
        "max_bond_dim": max(len(cut_states) for cut_states in states),
        "transition_count": transition_count,
        "endpoint_components": int(
            sum(len(endpoint) for endpoint in components.values())
        ),
        "representation": "renormalized_complementary_operators",
    }
    return tuple(tensors), diagnostics


def renormalized_operator_mpo(
    hamiltonian,
    *,
    local_qns=None,
):
    """Return an exact complementary-operator MPO for one-/two-site terms.

    ``local_qns`` separates operator-transfer sectors when the Hamiltonian
    conserves an Abelian charge.  It changes only the fixed Hamiltonian MPO;
    it does not assign charges to LETTA bonds or restrict any variational
    tensor entry.
    """
    tensors, diagnostics = _renormalized_operator_tensors(
        hamiltonian,
        local_qns=local_qns,
    )
    return LocalMPO(hamiltonian.dims, tensors), diagnostics


class _TermRenormalizedPairBinding:
    """Direct two-site hole contractions over one frontier transition pair."""

    uses_outer_messages = True

    def __init__(self, frontier, site, union_sites, merged_shape):
        self.frontier = frontier
        self.site = int(site)
        if self.site < 0 or self.site + 1 >= frontier.nsites:
            raise ValueError("site must be the left member of an adjacent pair.")
        following = self.site + 1
        expected_union = (self.site,) + tuple(
            sorted(
                (
                    set(frontier.physical_sites[self.site])
                    | set(frontier.physical_sites[following])
                )
                - {self.site}
            )
        )
        self.union_sites = tuple(int(value) for value in union_sites)
        if self.union_sites != expected_union:
            raise ValueError("union_sites are inconsistent with the adjacent pair.")
        self.merged_shape = tuple(int(value) for value in merged_shape)
        expected_shape = (
            frontier.virtual_bonds[self.site],
            frontier.virtual_bonds[self.site + 2],
            *(frontier.dims[index] for index in self.union_sites),
        )
        if self.merged_shape != expected_shape:
            raise ValueError(f"merged_shape must be {expected_shape}.")
        self.charge_resolved = frontier.charge_resolved
        self.dtype = frontier.dtype
        self._operators = self._build_pair_operators()
        self._transition_groups = self._group_transitions()
        self._operator_batches = {}
        self._expressions = {}
        self._support_action_plans = {}
        self._fused_support_action_plans = {}
        self._fused_support_action_expressions = {}

    def _build_pair_operators(self):
        """Collapse all two-step paths with common outer channels."""
        if self.frontier._zero_hamiltonian:
            return {}
        site = self.site
        following_by_left = {}
        for middle, right in self.frontier._transitions[site + 1]:
            following_by_left.setdefault(middle, []).append(right)
        operators = {}
        for left, middle in self.frontier._transitions[site]:
            for right in following_by_left.get(middle, ()):
                if (
                    left not in self.frontier._active_channel_sets[site]
                    or middle
                    not in self.frontier._active_channel_sets[site + 1]
                    or right
                    not in self.frontier._active_channel_sets[site + 2]
                ):
                    continue
                left_operator = self.frontier.mpo_tensors[site][left, middle]
                right_operator = self.frontier.mpo_tensors[site + 1][middle, right]
                operator = (
                    left_operator[:, None, :, None]
                    * right_operator[None, :, None, :]
                )
                key = (left, right)
                if key in operators:
                    operators[key] = operators[key] + operator
                else:
                    operators[key] = np.array(operator, copy=True)
        return {
            transition: operator
            for transition, operator in operators.items()
            if np.any(operator != 0)
        }

    def _group_transitions(self):
        groups = {}
        site = self.site
        for left, right in self._operators:
            key = (
                self.frontier.paired_sites(site, left),
                self.frontier.paired_sites(site + 2, right),
                (
                    self.frontier._bond_pairs[site][left]
                    if self.charge_resolved
                    else None
                ),
                (
                    self.frontier._bond_pairs[site + 2][right]
                    if self.charge_resolved
                    else None
                ),
            )
            groups.setdefault(key, []).append((left, right))
        return tuple(tuple(transitions) for transitions in groups.values())

    def _operators_for_transitions(self, transitions):
        if len(transitions) == 1:
            return self._operators[transitions[0]]
        cached = self._operator_batches.get(transitions)
        if cached is None:
            cached = np.stack(
                [self._operators[transition] for transition in transitions]
            )
            self._operator_batches[transitions] = cached
        return cached

    def _merged_labels(self, *, bra):
        bonds = self.frontier.bra_bonds if bra else self.frontier.ket_bonds
        physical = (
            self.frontier.bra_physical
            if bra
            else self.frontier.ket_physical
        )
        return (
            bonds[self.site],
            bonds[self.site + 2],
            *(physical[index] for index in self.union_sites),
        )

    def _expression(
        self,
        mode,
        left_channel,
        right_channel,
        *,
        batch_size=1,
        request_batch_size=1,
    ):
        batch_size = int(batch_size)
        request_batch_size = int(request_batch_size)
        if request_batch_size < 1:
            raise ValueError("request_batch_size must be positive.")
        key = (
            mode,
            int(left_channel),
            int(right_channel),
            batch_size,
            request_batch_size,
        )
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        if mode not in {
            "matrix",
            "action",
            "action_batch",
            "support_action_batch",
            "physical_block",
            "physical_blocks",
        }:
            raise ValueError("invalid pair-hole contraction mode.")

        site = self.site
        left_labels, left_copies = self.frontier._block_labels_and_copies(
            site,
            int(left_channel),
        )
        right_labels, right_copies = self.frontier._block_labels_and_copies(
            site + 2,
            int(right_channel),
        )
        left_shape = self.frontier.block_shape(site, left_channel)
        right_shape = self.frontier.block_shape(site + 2, right_channel)
        operator_shape = (
            self.frontier.dims[site],
            self.frontier.dims[site + 1],
            self.frontier.dims[site],
            self.frontier.dims[site + 1],
        )
        operator_labels = (
            self.frontier.bra_physical[site],
            self.frontier.bra_physical[site + 1],
            self.frontier.ket_physical[site],
            self.frontier.ket_physical[site + 1],
        )
        if batch_size > 1:
            batch_label = self.frontier._new_label()
            left_shape = (batch_size, *left_shape)
            right_shape = (batch_size, *right_shape)
            operator_shape = (batch_size, *operator_shape)
            left_labels = (batch_label, *left_labels)
            right_labels = (batch_label, *right_labels)
            operator_labels = (batch_label, *operator_labels)

        arguments = [
            left_shape,
            left_labels,
            right_shape,
            right_labels,
            operator_shape,
            operator_labels,
        ]
        bra_labels = self._merged_labels(bra=True)
        ket_labels = self._merged_labels(bra=False)
        if mode == "support_action_batch":
            raise ValueError(
                "support-action expressions require packed-support metadata."
            )
        if mode == "action":
            arguments.extend((self.merged_shape, ket_labels))
            output_labels = bra_labels
        elif mode == "action_batch":
            vector_batch_label = self.frontier._new_label()
            arguments.extend(
                (
                    (request_batch_size, *self.merged_shape),
                    (vector_batch_label, *ket_labels),
                )
            )
            output_labels = (vector_batch_label, *bra_labels)
        elif mode in {"physical_block", "physical_blocks"}:
            request_label = (
                self.frontier._new_label()
                if mode == "physical_blocks"
                else None
            )
            for physical_site in self.union_sites:
                selector_shape = (self.frontier.dims[physical_site],)
                bra_selector_labels = (
                    self.frontier.bra_physical[physical_site],
                )
                ket_selector_labels = (
                    self.frontier.ket_physical[physical_site],
                )
                if request_label is not None:
                    selector_shape = (request_batch_size, *selector_shape)
                    bra_selector_labels = (
                        request_label,
                        *bra_selector_labels,
                    )
                    ket_selector_labels = (
                        request_label,
                        *ket_selector_labels,
                    )
                arguments.extend(
                    (
                        selector_shape,
                        bra_selector_labels,
                        selector_shape,
                        ket_selector_labels,
                    )
                )
            output_labels = (
                self.frontier.bra_bonds[site],
                self.frontier.bra_bonds[site + 2],
                self.frontier.ket_bonds[site],
                self.frontier.ket_bonds[site + 2],
            )
            if request_label is not None:
                output_labels = (request_label, *output_labels)
        else:
            output_labels = bra_labels + ket_labels
        copies = left_copies + right_copies
        for value, labels in copies:
            arguments.extend((value.shape, labels))
        expression = contract_expression(
            *arguments,
            output_labels,
            optimize=self.frontier.optimize,
        )
        cached = (expression, tuple(value for value, _labels in copies))
        self._expressions[key] = cached
        return cached

    def _support_action_plan(self, support_indices):
        support_indices = np.asarray(support_indices, dtype=np.intp).reshape(-1)
        key = tuple(int(index) for index in support_indices)
        cached = self._support_action_plans.get(key)
        if cached is not None:
            return cached

        left_dimension, right_dimension = self.merged_shape[:2]
        physical_shape = self.merged_shape[2:]
        coordinates = np.unravel_index(support_indices, self.merged_shape)
        virtual_entries = (
            np.asarray(coordinates[0], dtype=np.intp) * right_dimension
            + np.asarray(coordinates[1], dtype=np.intp)
        )
        physical_blocks = np.ravel_multi_index(
            tuple(np.asarray(axis, dtype=np.intp) for axis in coordinates[2:]),
            physical_shape,
        )
        block_support = {}
        for block in np.unique(physical_blocks):
            positions = np.flatnonzero(physical_blocks == block)
            block_support[int(block)] = (
                positions,
                virtual_entries[positions],
            )

        connected_pairs = tuple(
            (int(row), int(column))
            for row, column in hamiltonian_physical_connectivity(
                self.frontier.hamiltonian,
                self.union_sites,
            )
            if int(row) in block_support and int(column) in block_support
        )
        cached = {
            "connected_blocks": len(connected_pairs),
            "block_support": block_support,
            "connected_pairs": connected_pairs,
            "physical_shape": physical_shape,
        }
        self._support_action_plans[key] = cached
        return cached

    def _fused_support_action_plan(self, support_indices):
        """Plan structurally pruned physical requests for packed actions."""
        support_indices = np.asarray(support_indices, dtype=np.intp).reshape(-1)
        key = tuple(int(index) for index in support_indices)
        cached = self._fused_support_action_plans.get(key)
        if cached is not None:
            return cached

        right_dimension = self.merged_shape[1]
        physical_shape = self.merged_shape[2:]
        coordinates = np.unravel_index(support_indices, self.merged_shape)
        virtual_entries = (
            np.asarray(coordinates[0], dtype=np.intp) * right_dimension
            + np.asarray(coordinates[1], dtype=np.intp)
        )
        physical_blocks = np.ravel_multi_index(
            tuple(np.asarray(axis, dtype=np.intp) for axis in coordinates[2:]),
            physical_shape,
        )
        block_support = {}
        for block in np.unique(physical_blocks):
            packed = np.flatnonzero(physical_blocks == block)
            block_support[int(block)] = (
                packed,
                virtual_entries[packed],
            )

        configurations = tuple(
            tuple(
                int(value)
                for value in np.unravel_index(block, physical_shape)
            )
            for block in range(int(np.prod(physical_shape)))
        )
        connected_pairs = tuple(
            (int(row), int(column))
            for row, column in hamiltonian_physical_connectivity(
                self.frontier.hamiltonian,
                self.union_sites,
            )
            if int(row) in block_support and int(column) in block_support
        )
        physical_axes = {
            physical_site: axis
            for axis, physical_site in enumerate(self.union_sites)
        }
        local_axes = (
            physical_axes[self.site],
            physical_axes[self.site + 1],
        )
        union_set = set(self.union_sites)
        left_frontier = set(self.frontier.frontier_sites[self.site])
        right_frontier = set(self.frontier.frontier_sites[self.site + 2])
        identity = {
            physical_site: self.frontier._identities[
                self.frontier.dims[physical_site]
            ]
            for physical_site in self.union_sites
        }
        groups = []
        request_count = 0
        for transitions in self._transition_groups:
            left_channel, right_channel = transitions[0]
            shared_sites = (
                (
                    left_frontier
                    - set(self.frontier.paired_sites(self.site, left_channel))
                )
                | (
                    right_frontier
                    - set(
                        self.frontier.paired_sites(
                            self.site + 2,
                            right_channel,
                        )
                    )
                )
            ) & union_set
            operator = np.asarray(
                self._operators_for_transitions(transitions)
            )
            operator_nonzero = (
                np.any(operator != 0, axis=0)
                if operator.ndim == 5
                else operator != 0
            )
            requests = []
            for row, column in connected_pairs:
                bra = configurations[row]
                ket = configurations[column]
                if any(
                    bra[physical_axes[physical_site]]
                    != ket[physical_axes[physical_site]]
                    for physical_site in shared_sites
                ):
                    continue
                operator_index = (
                    bra[local_axes[0]],
                    bra[local_axes[1]],
                    ket[local_axes[0]],
                    ket[local_axes[1]],
                )
                if not operator_nonzero[operator_index]:
                    continue
                requests.append((row, column))
            for start in range(0, len(requests), 256):
                chunk = tuple(requests[start : start + 256])
                rows = np.asarray([row for row, _column in chunk], dtype=np.intp)
                columns = np.asarray(
                    [column for _row, column in chunk],
                    dtype=np.intp,
                )
                selectors = []
                for axis, physical_site in enumerate(self.union_sites):
                    selectors.extend(
                        (
                            identity[physical_site][
                                np.asarray(
                                    [
                                        configurations[row][axis]
                                        for row in rows
                                    ],
                                    dtype=np.intp,
                                )
                            ],
                            identity[physical_site][
                                np.asarray(
                                    [
                                        configurations[column][axis]
                                        for column in columns
                                    ],
                                    dtype=np.intp,
                                )
                            ],
                        )
                    )
                groups.append(
                    {
                        "transitions": transitions,
                        "rows": rows,
                        "columns": columns,
                        "selectors": tuple(selectors),
                    }
                )
                request_count += len(chunk)
        cached = {
            "block_support": block_support,
            "groups": tuple(groups),
            "request_count": request_count,
        }
        self._fused_support_action_plans[key] = cached
        return cached

    def fused_support_action_workspace_elements(
        self,
        support_indices,
        batch_size,
    ):
        """Estimate cached selectors and peak explicit fused work arrays."""
        support_indices = np.asarray(support_indices)
        if support_indices.ndim != 1:
            raise ValueError("support_indices must be one-dimensional.")
        if not np.issubdtype(support_indices.dtype, np.integer):
            raise TypeError("support_indices must contain integers.")
        support_indices = support_indices.astype(np.intp, copy=False)
        size = int(np.prod(self.merged_shape))
        if np.any((support_indices < 0) | (support_indices >= size)):
            raise ValueError("support_indices contains an out-of-range index.")
        if np.unique(support_indices).size != support_indices.size:
            raise ValueError("support_indices must not contain duplicates.")
        batch_size = int(batch_size)
        if batch_size < 0:
            raise ValueError("batch_size must be nonnegative.")

        plan = self._fused_support_action_plan(support_indices)
        selector_elements = int(
            sum(
                selector.size
                for group in plan["groups"]
                for selector in group["selectors"]
            )
        )
        peak_requests = max(
            (group["rows"].size for group in plan["groups"]),
            default=0,
        )
        action_elements = int(
            peak_requests
            * batch_size
            * self.merged_shape[0]
            * self.merged_shape[1]
        )
        return {
            "cached_selector_elements": selector_elements,
            "peak_input_elements": action_elements,
            "peak_output_elements": action_elements,
            "upper_bound_elements": selector_elements + 2 * action_elements,
            "groups": len(plan["groups"]),
            "requests": int(plan["request_count"]),
            "peak_requests": int(peak_requests),
        }

    def _fused_support_action_expression(
        self,
        transitions,
        *,
        request_count,
        vector_count,
    ):
        """Compile one request-batched packed-action contraction topology."""
        left_channel, right_channel = transitions[0]
        transition_count = len(transitions)
        key = (
            int(left_channel),
            int(right_channel),
            transition_count,
            int(request_count),
            int(vector_count),
        )
        cached = self._fused_support_action_expressions.get(key)
        if cached is not None:
            return cached

        site = self.site
        left_labels, left_copies = self.frontier._block_labels_and_copies(
            site,
            int(left_channel),
        )
        right_labels, right_copies = self.frontier._block_labels_and_copies(
            site + 2,
            int(right_channel),
        )
        left_shape = self.frontier.block_shape(site, left_channel)
        right_shape = self.frontier.block_shape(site + 2, right_channel)
        operator_shape = (
            self.frontier.dims[site],
            self.frontier.dims[site + 1],
            self.frontier.dims[site],
            self.frontier.dims[site + 1],
        )
        operator_labels = (
            self.frontier.bra_physical[site],
            self.frontier.bra_physical[site + 1],
            self.frontier.ket_physical[site],
            self.frontier.ket_physical[site + 1],
        )
        if transition_count > 1:
            transition_label = self.frontier._new_label()
            left_shape = (transition_count, *left_shape)
            right_shape = (transition_count, *right_shape)
            operator_shape = (transition_count, *operator_shape)
            left_labels = (transition_label, *left_labels)
            right_labels = (transition_label, *right_labels)
            operator_labels = (transition_label, *operator_labels)

        arguments = [
            left_shape,
            left_labels,
            right_shape,
            right_labels,
            operator_shape,
            operator_labels,
        ]
        request_label = self.frontier._new_label()
        vector_label = self.frontier._new_label()
        for physical_site in self.union_sites:
            selector_shape = (
                int(request_count),
                self.frontier.dims[physical_site],
            )
            arguments.extend(
                (
                    selector_shape,
                    (
                        request_label,
                        self.frontier.bra_physical[physical_site],
                    ),
                    selector_shape,
                    (
                        request_label,
                        self.frontier.ket_physical[physical_site],
                    ),
                )
            )
        request_vector_shape = (
            int(request_count),
            int(vector_count),
            self.merged_shape[0],
            self.merged_shape[1],
        )
        arguments.extend(
            (
                request_vector_shape,
                (
                    request_label,
                    vector_label,
                    self.frontier.ket_bonds[site],
                    self.frontier.ket_bonds[site + 2],
                ),
            )
        )
        output_labels = (
            request_label,
            vector_label,
            self.frontier.bra_bonds[site],
            self.frontier.bra_bonds[site + 2],
        )
        copies = left_copies + right_copies
        for value, labels in copies:
            arguments.extend((value.shape, labels))
        cached = (
            contract_expression(
                *arguments,
                output_labels,
                optimize=self.frontier.optimize,
            ),
            tuple(value for value, _labels in copies),
        )
        self._fused_support_action_expressions[key] = cached
        return cached

    def prepare_hole_action_support(
        self,
        site,
        left,
        right,
        support_indices,
    ):
        """Bind outer messages and assemble only support-compatible blocks."""
        left, right = self._validated_messages(site, left, right)
        support_indices = np.asarray(support_indices)
        if support_indices.ndim != 1:
            raise ValueError("support_indices must be one-dimensional.")
        if not np.issubdtype(support_indices.dtype, np.integer):
            raise TypeError("support_indices must contain integers.")
        support_indices = support_indices.astype(np.intp, copy=False)
        size = int(np.prod(self.merged_shape))
        if np.any((support_indices < 0) | (support_indices >= size)):
            raise ValueError("support_indices contains an out-of-range index.")
        if np.unique(support_indices).size != support_indices.size:
            raise ValueError("support_indices must not contain duplicates.")
        plan = self._support_action_plan(support_indices)
        fused_plan = self._fused_support_action_plan(support_indices)
        start = perf_counter()
        rows = []
        columns = []
        values = []
        virtual_size = self.merged_shape[0] * self.merged_shape[1]
        virtual_basis = np.eye(virtual_size, dtype=float).reshape(
            virtual_size,
            self.merged_shape[0],
            self.merged_shape[1],
        )
        peak_raw_elements = 0
        for group in fused_plan["groups"]:
            transitions = group["transitions"]
            left_channel, right_channel = transitions[0]
            transition_count = len(transitions)
            request_count = group["rows"].size
            expression, copy_factors = self._fused_support_action_expression(
                transitions,
                request_count=request_count,
                vector_count=virtual_size,
            )
            left_blocks = left.blocks[left_channel]
            right_blocks = right.blocks[right_channel]
            if transition_count > 1:
                left_blocks = np.stack(
                    [left.blocks[channel] for channel, _right in transitions]
                )
                right_blocks = np.stack(
                    [right.blocks[channel] for _left, channel in transitions]
                )
            raw = np.asarray(
                expression(
                    left_blocks,
                    right_blocks,
                    self._operators_for_transitions(transitions),
                    *group["selectors"],
                    np.broadcast_to(
                        virtual_basis,
                        (request_count, *virtual_basis.shape),
                    ),
                    *copy_factors,
                )
            ).reshape(request_count, virtual_size, virtual_size)
            raw = raw.transpose(0, 2, 1)
            peak_raw_elements = max(
                peak_raw_elements,
                int(2 * raw.size),
            )
            for request, (row, column) in enumerate(
                zip(group["rows"], group["columns"])
            ):
                output_packed, output_virtual = plan["block_support"][
                    int(row)
                ]
                input_packed, input_virtual = plan["block_support"][
                    int(column)
                ]
                block = raw[request][np.ix_(output_virtual, input_virtual)]
                rows.append(np.repeat(output_packed, input_packed.size))
                columns.append(np.tile(input_packed, output_packed.size))
                values.append(block.reshape(-1))
        dtype = np.result_type(
            self.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        if values:
            operator = coo_matrix(
                (
                    np.concatenate(values),
                    (np.concatenate(rows), np.concatenate(columns)),
                ),
                shape=(support_indices.size, support_indices.size),
                dtype=dtype,
            ).tocsr()
            operator.sum_duplicates()
            operator.eliminate_zeros()
        else:
            operator = csr_matrix(
                (support_indices.size, support_indices.size),
                dtype=dtype,
            )
        assembly_seconds = perf_counter() - start
        return _PreparedSupportPairAction(
            operator,
            assembly_seconds=assembly_seconds,
            connected_blocks=plan["connected_blocks"],
            raw_block_elements=peak_raw_elements,
        )

    def _validated_messages(self, site, left, right):
        if int(site) != self.site:
            raise ValueError("pair binding belongs to a different adjacent pair.")
        return (
            self.frontier._validated_message(left, self.site),
            self.frontier._validated_message(right, self.site + 2),
        )

    def _validated_configuration(self, configuration, *, name):
        configuration = tuple(int(value) for value in configuration)
        if len(configuration) != len(self.union_sites):
            raise ValueError(
                f"{name} must contain {len(self.union_sites)} physical values."
            )
        if any(
            value < 0 or value >= self.frontier.dims[physical_site]
            for value, physical_site in zip(configuration, self.union_sites)
        ):
            raise ValueError(f"{name} contains an out-of-range physical value.")
        return configuration

    def hole_matrix(self, site, left, right):
        left, right = self._validated_messages(site, left, right)
        size = int(np.prod(self.merged_shape))
        dtype = np.result_type(
            self.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros((size, size), dtype=dtype)
        for transitions in self._transition_groups:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._expression(
                "matrix",
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = left.blocks[left_channel]
            right_blocks = right.blocks[right_channel]
            if batch_size > 1:
                left_blocks = np.stack(
                    [left.blocks[channel] for channel, _right in transitions]
                )
                right_blocks = np.stack(
                    [right.blocks[channel] for _left, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                self._operators_for_transitions(transitions),
                *copy_factors,
            )
            result += np.asarray(value).reshape(size, size)
        return result

    def hole_action(self, site, left, right, vector):
        left, right = self._validated_messages(site, left, right)
        vector = np.asarray(vector).reshape(self.merged_shape)
        dtype = np.result_type(
            self.dtype,
            vector.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros(vector.size, dtype=dtype)
        for transitions in self._transition_groups:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._expression(
                "action",
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = left.blocks[left_channel]
            right_blocks = right.blocks[right_channel]
            if batch_size > 1:
                left_blocks = np.stack(
                    [left.blocks[channel] for channel, _right in transitions]
                )
                right_blocks = np.stack(
                    [right.blocks[channel] for _left, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                self._operators_for_transitions(transitions),
                vector,
                *copy_factors,
            )
            result += np.asarray(value).reshape(-1)
        return result

    def hole_action_batch(self, site, left, right, vectors):
        """Apply the pair Hamiltonian to a column batch in shared contractions.

        ``vectors`` must have shape ``(merged_size, batch_size)`` or
        ``(*merged_shape, batch_size)``. The returned array always has shape
        ``(merged_size, batch_size)``.
        """
        left, right = self._validated_messages(site, left, right)
        vectors = np.asarray(vectors)
        size = int(np.prod(self.merged_shape))
        if vectors.ndim == 2 and vectors.shape[0] == size:
            columns = vectors
        elif (
            vectors.ndim == len(self.merged_shape) + 1
            and tuple(vectors.shape[:-1]) == self.merged_shape
        ):
            columns = vectors.reshape(size, vectors.shape[-1])
        else:
            raise ValueError(
                "vectors must have shape (merged_size, batch_size) or "
                "(*merged_shape, batch_size)."
            )
        batch_size = columns.shape[1]
        dtype = np.result_type(
            self.dtype,
            columns.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros((size, batch_size), dtype=dtype)
        if batch_size == 0:
            return result
        vector_batch = columns.T.reshape(batch_size, *self.merged_shape)
        for transitions in self._transition_groups:
            left_channel, right_channel = transitions[0]
            transition_batch_size = len(transitions)
            expression, copy_factors = self._expression(
                "action_batch",
                left_channel,
                right_channel,
                batch_size=transition_batch_size,
                request_batch_size=batch_size,
            )
            left_blocks = left.blocks[left_channel]
            right_blocks = right.blocks[right_channel]
            if transition_batch_size > 1:
                left_blocks = np.stack(
                    [left.blocks[channel] for channel, _right in transitions]
                )
                right_blocks = np.stack(
                    [right.blocks[channel] for _left, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                self._operators_for_transitions(transitions),
                vector_batch,
                *copy_factors,
            )
            result += np.asarray(value).reshape(batch_size, size).T
        return result

    def hole_action_support_batch(
        self,
        site,
        left,
        right,
        support_indices,
        vectors,
    ):
        """Apply a column batch directly in flattened merged-pair support."""
        left, right = self._validated_messages(site, left, right)
        support_indices = np.asarray(support_indices)
        if support_indices.ndim != 1:
            raise ValueError("support_indices must be one-dimensional.")
        if not np.issubdtype(support_indices.dtype, np.integer):
            raise TypeError("support_indices must contain integers.")
        support_indices = support_indices.astype(np.intp, copy=False)
        size = int(np.prod(self.merged_shape))
        if np.any((support_indices < 0) | (support_indices >= size)):
            raise ValueError("support_indices contains an out-of-range index.")
        if np.unique(support_indices).size != support_indices.size:
            raise ValueError("support_indices must not contain duplicates.")
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[0] != support_indices.size:
            raise ValueError(
                "vectors must have shape (support_size, batch_size)."
            )
        dtype = np.result_type(
            self.dtype,
            vectors.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros((support_indices.size, vectors.shape[1]), dtype=dtype)
        if vectors.shape[1] == 0 or support_indices.size == 0:
            return result
        prepared = self.prepare_hole_action_support(
            site,
            left,
            right,
            support_indices,
        )
        return np.asarray(prepared(vectors), dtype=dtype)

    def hole_action_support_fused_batch(
        self,
        site,
        left,
        right,
        support_indices,
        vectors,
    ):
        """Apply packed columns through pruned request-batched contractions.

        Unlike :meth:`hole_action_support_batch`, this path neither assembles
        support-projected Hamiltonian blocks nor lifts a column into the full
        merged physical tensor.  It batches only physical configuration pairs
        compatible with each renormalized transition topology.
        """
        left, right = self._validated_messages(site, left, right)
        support_indices = np.asarray(support_indices)
        if support_indices.ndim != 1:
            raise ValueError("support_indices must be one-dimensional.")
        if not np.issubdtype(support_indices.dtype, np.integer):
            raise TypeError("support_indices must contain integers.")
        support_indices = support_indices.astype(np.intp, copy=False)
        size = int(np.prod(self.merged_shape))
        if np.any((support_indices < 0) | (support_indices >= size)):
            raise ValueError("support_indices contains an out-of-range index.")
        if np.unique(support_indices).size != support_indices.size:
            raise ValueError("support_indices must not contain duplicates.")
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[0] != support_indices.size:
            raise ValueError(
                "vectors must have shape (support_size, batch_size)."
            )
        dtype = np.result_type(
            self.dtype,
            vectors.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros((support_indices.size, vectors.shape[1]), dtype=dtype)
        if vectors.shape[1] == 0 or support_indices.size == 0:
            return result

        plan = self._fused_support_action_plan(support_indices)
        block_support = plan["block_support"]
        vector_count = vectors.shape[1]
        virtual_size = self.merged_shape[0] * self.merged_shape[1]
        for group in plan["groups"]:
            transitions = group["transitions"]
            rows = group["rows"]
            columns = group["columns"]
            request_count = rows.size
            request_vectors = np.zeros(
                (request_count, vector_count, virtual_size),
                dtype=vectors.dtype,
            )
            for request, column in enumerate(columns):
                packed, virtual = block_support[int(column)]
                request_vectors[request][:, virtual] = vectors[packed].T
            request_vectors = request_vectors.reshape(
                request_count,
                vector_count,
                self.merged_shape[0],
                self.merged_shape[1],
            )

            left_channel, right_channel = transitions[0]
            left_blocks = left.blocks[left_channel]
            right_blocks = right.blocks[right_channel]
            if len(transitions) > 1:
                left_blocks = np.stack(
                    [left.blocks[channel] for channel, _right in transitions]
                )
                right_blocks = np.stack(
                    [right.blocks[channel] for _left, channel in transitions]
                )
            expression, copy_factors = self._fused_support_action_expression(
                transitions,
                request_count=request_count,
                vector_count=vector_count,
            )
            value = expression(
                left_blocks,
                right_blocks,
                self._operators_for_transitions(transitions),
                *group["selectors"],
                request_vectors,
                *copy_factors,
            )
            contributions = np.asarray(value).reshape(
                request_count,
                vector_count,
                virtual_size,
            )
            for request, row in enumerate(rows):
                packed, virtual = block_support[int(row)]
                result[packed] += contributions[request][:, virtual].T
        return result

    def hole_blocks(
        self,
        site,
        left,
        right,
        configuration_pairs,
        *,
        request_batch_size=64,
    ):
        left, right = self._validated_messages(site, left, right)
        request_batch_size = int(request_batch_size)
        if request_batch_size < 1:
            raise ValueError("request_batch_size must be positive.")
        virtual_size = (
            self.frontier.virtual_bonds[self.site]
            * self.frontier.virtual_bonds[self.site + 2]
        )
        dtype = np.result_type(
            self.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        prepared = []
        for row, column, bra_configuration, ket_configuration in configuration_pairs:
            prepared.append(
                (
                    int(row),
                    int(column),
                    self._validated_configuration(
                        bra_configuration,
                        name="bra_configuration",
                    ),
                    self._validated_configuration(
                        ket_configuration,
                        name="ket_configuration",
                    ),
                )
            )
        result = {
            (row, column): np.zeros((virtual_size, virtual_size), dtype=dtype)
            for row, column, _bra, _ket in prepared
        }
        if not prepared:
            return result
        selector_batches = []
        for axis, physical_site in enumerate(self.union_sites):
            identity = self.frontier._identities[
                self.frontier.dims[physical_site]
            ]
            selector_batches.extend(
                (
                    identity[
                        np.asarray(
                            [bra[axis] for _row, _column, bra, _ket in prepared],
                            dtype=np.intp,
                        )
                    ],
                    identity[
                        np.asarray(
                            [ket[axis] for _row, _column, _bra, ket in prepared],
                            dtype=np.intp,
                        )
                    ],
                )
            )
        values = np.zeros(
            (len(prepared), virtual_size, virtual_size),
            dtype=dtype,
        )
        for transitions in self._transition_groups:
            left_channel, right_channel = transitions[0]
            transition_batch_size = len(transitions)
            left_blocks = left.blocks[left_channel]
            right_blocks = right.blocks[right_channel]
            if transition_batch_size > 1:
                left_blocks = np.stack(
                    [left.blocks[channel] for channel, _right in transitions]
                )
                right_blocks = np.stack(
                    [right.blocks[channel] for _left, channel in transitions]
                )
            for start in range(0, len(prepared), request_batch_size):
                stop = min(start + request_batch_size, len(prepared))
                chunk_size = stop - start
                expression, copy_factors = self._expression(
                    "physical_blocks",
                    left_channel,
                    right_channel,
                    batch_size=transition_batch_size,
                    request_batch_size=chunk_size,
                )
                value = expression(
                    left_blocks,
                    right_blocks,
                    self._operators_for_transitions(transitions),
                    *(selectors[start:stop] for selectors in selector_batches),
                    *copy_factors,
                )
                values[start:stop] += np.asarray(value).reshape(
                    chunk_size,
                    virtual_size,
                    virtual_size,
                )
        for index, (row, column, _bra, _ket) in enumerate(prepared):
            result[(row, column)] += values[index]
        return result

    def hole_block(
        self,
        site,
        left,
        right,
        bra_configuration,
        ket_configuration,
    ):
        key = (0, 0)
        return self.hole_blocks(
            site,
            left,
            right,
            ((key[0], key[1], bra_configuration, ket_configuration),),
            request_batch_size=1,
        )[key]


class TermRenormalizedFrontier(BlockMPOFrontier):
    """Exact NARG-like frontier built from complementary local operators."""

    def __init__(
        self,
        hamiltonian,
        physical_sites,
        tensor_shapes,
        *,
        optimize="greedy",
        local_qns=None,
        bond_qns=None,
        local_backend="dense",
        local_rank=None,
        local_rtol=0.0,
        local_atol=0.0,
    ):
        tensors, diagnostics = _renormalized_operator_tensors(
            hamiltonian,
            physical_sites=physical_sites,
            local_qns=local_qns,
        )
        self._zero_hamiltonian = hamiltonian.constant == 0.0 and not any(
            np.any(term.operator != 0) for term in hamiltonian.terms
        )
        if self._zero_hamiltonian and len(hamiltonian.dims) == 1:
            dim = hamiltonian.dims[0]
            tensors = (np.eye(dim, dtype=hamiltonian.dtype)[None, None],)
        self.hamiltonian = hamiltonian
        self.renormalized_diagnostics = diagnostics
        self._pair_bindings = {}
        super().__init__(
            hamiltonian.dims,
            physical_sites,
            tensor_shapes,
            tensors,
            optimize=optimize,
            local_qns=local_qns,
            bond_qns=bond_qns,
            local_backend=local_backend,
            local_rank=local_rank,
            local_rtol=local_rtol,
            local_atol=local_atol,
        )

    def bind_pair(self, site, union_sites, merged_shape):
        """Return a cached direct contraction over two local operator steps."""
        key = (
            int(site),
            tuple(int(value) for value in union_sites),
            tuple(int(value) for value in merged_shape),
        )
        binding = self._pair_bindings.get(key)
        if binding is None:
            binding = _TermRenormalizedPairBinding(self, *key)
            self._pair_bindings[key] = binding
        return binding

    def scalar(self, tensors):
        if self._zero_hamiltonian:
            super().scalar(tensors)
            dtype = np.result_type(self.dtype, np.asarray(tensors[0]).dtype)
            return np.asarray(0, dtype=dtype).item()
        return super().scalar(tensors)

    def boundary_scalar(self, message, cut):
        if self._zero_hamiltonian:
            cut = int(cut)
            if cut not in {0, self.nsites}:
                raise ValueError("a scalar can only be extracted at a boundary cut.")
            self._validated_message(message, cut)
            return np.asarray(0, dtype=self.dtype).item()
        return super().boundary_scalar(message, cut)

    def hole_matrix(self, site, left, right):
        if not self._zero_hamiltonian:
            return super().hole_matrix(site, left, right)
        site = int(site)
        self._validated_message(left, site)
        self._validated_message(right, site + 1)
        size = int(np.prod(self.tensor_shapes[site]))
        return np.zeros((size, size), dtype=self.dtype)

    def hole_block(self, site, left, right, bra_configuration, ket_configuration):
        if not self._zero_hamiltonian:
            return super().hole_block(
                site,
                left,
                right,
                bra_configuration,
                ket_configuration,
            )
        site = int(site)
        self._validated_message(left, site)
        self._validated_message(right, site + 1)
        self._validated_physical_configuration(
            site,
            bra_configuration,
            name="bra_configuration",
        )
        self._validated_physical_configuration(
            site,
            ket_configuration,
            name="ket_configuration",
        )
        virtual_size = self.virtual_bonds[site] * self.virtual_bonds[site + 1]
        return np.zeros((virtual_size, virtual_size), dtype=self.dtype)

    def hole_blocks(
        self,
        site,
        left,
        right,
        configuration_pairs,
        *,
        request_batch_size=64,
    ):
        if not self._zero_hamiltonian:
            return super().hole_blocks(
                site,
                left,
                right,
                configuration_pairs,
                request_batch_size=request_batch_size,
            )
        request_batch_size = int(request_batch_size)
        if request_batch_size < 1:
            raise ValueError("request_batch_size must be positive.")
        return {
            (int(row), int(column)): self.hole_block(
                site,
                left,
                right,
                bra_configuration,
                ket_configuration,
            )
            for row, column, bra_configuration, ket_configuration in configuration_pairs
        }

    def hole_action(self, site, left, right, vector):
        if not self._zero_hamiltonian:
            return super().hole_action(site, left, right, vector)
        site = int(site)
        self._validated_message(left, site)
        self._validated_message(right, site + 1)
        vector = np.asarray(vector).reshape(self.tensor_shapes[site])
        return np.zeros(vector.size, dtype=np.result_type(self.dtype, vector.dtype))


__all__ = ["TermRenormalizedFrontier", "renormalized_operator_mpo"]
