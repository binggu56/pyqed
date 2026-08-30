"""Identity-aware block frontiers for graph-tied tensors and an MPO.

This is an exact prototype of :mod:`pyqed.letta.mpo_frontier` in which the
MPO bond is represented by separate blocks.  Whenever the operator suffix
from one MPO channel is diagonal on a tied physical variable, the
corresponding bra/ket frontier pair is stored as one shared index.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import importlib

import numpy as np
from opt_einsum import contract_expression
from scipy.sparse import coo_matrix

from pyqed.tn.effective_operator import _array_module


@dataclass(frozen=True)
class BlockFrontierMessage:
    """Channel-blocked numerical message at one site cut."""

    cut: int
    blocks: tuple[np.ndarray, ...]


class BlockMPOFrontier:
    r"""Cache exact identity-aware frontier messages for a graph-tied state.

    For MPO channel ``a`` at cut ``c``, let ``Q[c, a]`` contain every future
    physical site on which at least one completed MPO suffix path can be
    off-diagonal.  A frontier site in ``Q[c, a]`` carries separate bra and ket
    values; every other site carries one value shared by bra and ket.  The
    latter reduction is exact because the corresponding suffix operator is
    diagonal on that site.

    The implementation deliberately consumes the uncompressed MPO tensors.
    A generic MPO compression can mix diagonal and off-diagonal automaton
    channels and therefore hide the block structure used here.
    """

    def __init__(
        self,
        dims,
        physical_sites,
        tensor_shapes,
        mpo_tensors,
        *,
        optimize="greedy",
        local_qns=None,
        bond_qns=None,
        compute_dtype=None,
        device="cpu",
        route_memory=32,
        action_memory=32,
    ):
        self.dims = tuple(int(dim) for dim in dims)
        self.physical_groups = tuple(
            tuple(int(site) for site in sites) for sites in physical_sites
        )
        self.tensor_shapes = tuple(
            tuple(int(dim) for dim in shape) for shape in tensor_shapes
        )
        self.mpo_tensors = tuple(np.asarray(tensor) for tensor in mpo_tensors)
        self.optimize = optimize
        self._xp, self.device = _array_module(device)
        self.local_qns = local_qns
        self.bond_qns = bond_qns
        self.charge_resolved = local_qns is not None or bond_qns is not None
        if self.charge_resolved:
            if local_qns is None or bond_qns is None:
                raise ValueError("local_qns and bond_qns must be supplied together.")
            self.local_qns = tuple(
                tuple(tuple(int(x) for x in charge) for charge in site)
                for site in local_qns
            )
            self.bond_qns = tuple(
                tuple(tuple(int(x) for x in charge) for charge in bond)
                for bond in bond_qns
            )
        self.nsites = len(self.dims)
        if not self.dims or any(dim < 1 for dim in self.dims):
            raise ValueError("dims must contain positive local dimensions.")
        if not (
            len(self.physical_groups)
            == len(self.tensor_shapes)
            == len(self.mpo_tensors)
            == self.nsites
        ):
            raise ValueError("frontier inputs must contain one entry per site.")

        for site, (sites, shape) in enumerate(
            zip(self.physical_groups, self.tensor_shapes)
        ):
            if not sites or sites[0] != site:
                raise ValueError("physical_sites must begin with the tensor site.")
            if len(set(sites)) != len(sites) or any(
                physical_site < site or physical_site >= self.nsites
                for physical_site in sites
            ):
                raise ValueError("physical_sites must contain unique forward sites.")
            expected_physical = tuple(self.dims[index] for index in sites)
            if len(shape) != 2 + len(sites) or shape[2:] != expected_physical:
                raise ValueError(
                    f"tensor shape {site} is inconsistent with physical_sites."
                )
        self.virtual_bonds = (self.tensor_shapes[0][0],) + tuple(
            shape[1] for shape in self.tensor_shapes
        )
        self.mpo_bonds = (self.mpo_tensors[0].shape[0],) + tuple(
            tensor.shape[1] for tensor in self.mpo_tensors
        )
        for site, (shape, mpo) in enumerate(zip(self.tensor_shapes, self.mpo_tensors)):
            if shape[0] != self.virtual_bonds[site]:
                raise ValueError(f"virtual bond mismatch before site {site}.")
            if mpo.shape != (
                self.mpo_bonds[site],
                self.mpo_bonds[site + 1],
                self.dims[site],
                self.dims[site],
            ):
                raise ValueError(f"MPO tensor {site} has an invalid shape.")
        if self.virtual_bonds[0] != 1 or self.virtual_bonds[-1] != 1:
            raise ValueError("frontier virtual boundary dimensions must be one.")
        if self.mpo_bonds[0] != 1 or self.mpo_bonds[-1] != 1:
            raise ValueError("frontier MPO boundary dimensions must be one.")

        first_reference = list(range(self.nsites))
        for left_site, group in enumerate(self.physical_groups):
            for physical_site in group[1:]:
                first_reference[physical_site] = min(
                    first_reference[physical_site], left_site
                )
        self.frontier_sites = tuple(
            tuple(
                physical_site
                for physical_site in range(cut, self.nsites)
                if first_reference[physical_site] < cut
            )
            for cut in range(self.nsites + 1)
        )

        start = 0
        self.bra_bonds = tuple(range(start, start + self.nsites + 1))
        start += self.nsites + 1
        self.ket_bonds = tuple(range(start, start + self.nsites + 1))
        start += self.nsites + 1
        self.bra_physical = tuple(range(start, start + self.nsites))
        start += self.nsites
        self.ket_physical = tuple(range(start, start + self.nsites))
        self._next_auxiliary_label = start + self.nsites

        self._transitions = tuple(
            tuple(
                (left, right)
                for left in range(tensor.shape[0])
                for right in range(tensor.shape[1])
                if np.any(tensor[left, right] != 0)
            )
            for tensor in self.mpo_tensors
        )
        self._prefix_reachable = self._build_prefix_reachable()
        self._suffix_masks = self._build_suffix_masks()
        self._active_channels = tuple(
            tuple(
                channel
                for channel, (from_left, to_right) in enumerate(
                    zip(self._prefix_reachable[cut], self._suffix_reachable[cut])
                )
                if from_left and to_right
            )
            for cut in range(self.nsites + 1)
        )
        self._active_channel_sets = tuple(map(frozenset, self._active_channels))
        self._bond_pairs = self._build_bond_pairs()
        self._left_transition_groups = tuple(
            self._group_transitions(site, "left") for site in range(self.nsites)
        )
        self._right_transition_groups = tuple(
            self._group_transitions(site, "right") for site in range(self.nsites)
        )
        self._hole_transition_groups = tuple(
            self._group_transitions(site, "hole") for site in range(self.nsites)
        )
        self._operator_batches = {}
        self._block_paths = {}
        self._copies = {dim: self._copy_tensor(dim) for dim in set(self.dims)}
        self._identities = {dim: np.eye(dim) for dim in set(self.dims)}
        self._pair_selectors = {}
        self._expressions = {}
        self.dtype = np.dtype(
            np.result_type(*[tensor.dtype for tensor in self.mpo_tensors])
        )
        self._compute_dtype_requested = not (
            compute_dtype is None
            or str(compute_dtype).lower() in {"same", "native"}
        )
        if not self._compute_dtype_requested:
            compute_dtype = self.dtype
        compute_dtype = np.dtype(compute_dtype)
        if self.dtype.kind == "c" and compute_dtype.kind != "c":
            compute_dtype = np.dtype(
                np.complex64 if compute_dtype.itemsize <= 4 else np.complex128
            )
        if compute_dtype.kind not in "fc":
            raise TypeError("compute_dtype must be a real or complex floating dtype.")
        self.compute_dtype = compute_dtype
        if route_memory is None:
            route_memory = 0
        self.route_memory = float(route_memory)
        if not np.isfinite(self.route_memory) or self.route_memory < 0.0:
            raise ValueError("route_memory must be finite and nonnegative.")
        self._packed_route_limit_bytes = int(self.route_memory * 1024**2)
        self.action_memory = float(action_memory)
        if not np.isfinite(self.action_memory) or self.action_memory < 0.0:
            raise ValueError("action_memory must be finite and nonnegative.")
        self._packed_dense_action_limit_bytes = int(self.action_memory * 1024**2)
        self._packed_route_cache_bytes = 0
        self._packed_route_lru = OrderedDict()
        self._packed_route_cache_hits = 0
        self._packed_route_cache_misses = 0
        self._packed_route_cache_evictions = 0
        self._charge_sum_support_cache = {}
        self._physical_support_cache = {}
        self._storage_coordinate_cache = {}
        self._storage_inverse_cache = {}
        self._bond_pair_array_cache = {}
        self._tensor_support_cache = {}
        self._tensor_support_array_cache = {}
        self._packed_advance_plan_cache = OrderedDict()
        self._packed_hole_plan_cache = OrderedDict()
        self.last_packed_advance_peak_bytes = 0
        self.last_packed_hole_peak_bytes = 0
        self._compiled_route_builders_cache = None
        self.packed_route_backend = "python"

    @staticmethod
    def _copy_tensor(dim):
        value = np.zeros((dim, dim, dim))
        diagonal = np.arange(dim)
        value[diagonal, diagonal, diagonal] = 1.0
        return value

    @staticmethod
    def _has_offdiagonal(matrix):
        matrix = np.asarray(matrix)
        rows, columns = np.nonzero(matrix)
        return bool(np.any(rows != columns))

    def _build_prefix_reachable(self):
        reachable = [np.zeros(bond, dtype=bool) for bond in self.mpo_bonds]
        reachable[0][0] = True
        for site in range(self.nsites):
            for left, right in self._transitions[site]:
                if reachable[site][left]:
                    reachable[site + 1][right] = True
        return tuple(reachable)

    @staticmethod
    def _charge_difference(left, right):
        return tuple(a - b for a, b in zip(left, right))

    def _build_bond_pairs(self):
        if not self.charge_resolved:
            return None
        transfers = [[set() for _ in range(bond)] for bond in self.mpo_bonds]
        zero = tuple(0 for _ in self.local_qns[0][0])
        transfers[0][0].add(zero)
        for site in range(self.nsites):
            for left, right in self._transitions[site]:
                local = {
                    self._charge_difference(
                        self.local_qns[site][bra], self.local_qns[site][ket]
                    )
                    for bra, ket in zip(
                        *np.nonzero(self.mpo_tensors[site][left, right])
                    )
                }
                for prefix in transfers[site][left]:
                    for delta in local:
                        transfers[site + 1][right].add(
                            tuple(a + b for a, b in zip(prefix, delta))
                        )
        result = []
        for cut, labels in enumerate(self.bond_qns):
            cut_pairs = []
            for channel in range(self.mpo_bonds[cut]):
                allowed = transfers[cut][channel]
                cut_pairs.append(
                    tuple(
                        (bra, ket)
                        for bra, q_bra in enumerate(labels)
                        for ket, q_ket in enumerate(labels)
                        if self._charge_difference(q_bra, q_ket) in allowed
                    )
                )
            result.append(tuple(cut_pairs))
        return tuple(result)

    def _pair_selector(self, cut, channel):
        key = (int(cut), int(channel))
        value = self._pair_selectors.get(key)
        if value is None:
            pairs = self._bond_pairs[key[0]][key[1]]
            dimension = self.virtual_bonds[key[0]]
            value = np.zeros((len(pairs), dimension, dimension))
            for pair, (bra, ket) in enumerate(pairs):
                value[pair, bra, ket] = 1.0
            self._pair_selectors[key] = value
        return value

    def _build_suffix_masks(self):
        can_finish = [None] * (self.nsites + 1)
        masks = [None] * (self.nsites + 1)
        can_finish[-1] = np.ones(self.mpo_bonds[-1], dtype=bool)
        masks[-1] = (0,) * self.mpo_bonds[-1]
        for site in range(self.nsites - 1, -1, -1):
            reachable = np.zeros(self.mpo_bonds[site], dtype=bool)
            site_masks = [0] * self.mpo_bonds[site]
            for left, right in self._transitions[site]:
                if not can_finish[site + 1][right]:
                    continue
                reachable[left] = True
                site_masks[left] |= masks[site + 1][right]
                if self._has_offdiagonal(self.mpo_tensors[site][left, right]):
                    site_masks[left] |= 1 << site
            can_finish[site] = reachable
            masks[site] = tuple(site_masks)
        self._suffix_reachable = tuple(can_finish)
        return tuple(masks)

    @property
    def plan_count(self):
        return len(self._expressions)

    def clear_contraction_plans(self):
        """Release value-independent contraction workspaces and selectors."""

        self._operator_batches.clear()
        self._block_paths.clear()
        self._pair_selectors.clear()
        self._expressions.clear()
        self._packed_advance_plan_cache.clear()
        self._packed_hole_plan_cache.clear()
        self._packed_route_lru.clear()
        self._packed_route_cache_bytes = 0

    @staticmethod
    def _packed_plan_bytes(plan):
        return int(sum(np.asarray(value).nbytes for value in plan))

    def _cached_packed_plan(self, kind, key):
        cache = (
            self._packed_advance_plan_cache
            if kind == "advance"
            else self._packed_hole_plan_cache
        )
        plan = cache.get(key)
        if plan is None:
            self._packed_route_cache_misses += 1
            return None
        self._packed_route_cache_hits += 1
        self._packed_route_lru.move_to_end((kind, key))
        cache.move_to_end(key)
        return plan

    def _cache_packed_plan(self, kind, key, plan):
        size = self._packed_plan_bytes(plan)
        if size > self._packed_route_limit_bytes:
            return plan
        cache = (
            self._packed_advance_plan_cache
            if kind == "advance"
            else self._packed_hole_plan_cache
        )
        old = cache.pop(key, None)
        if old is not None:
            old_size = self._packed_route_lru.pop((kind, key))
            self._packed_route_cache_bytes -= old_size
        while (
            self._packed_route_lru
            and self._packed_route_cache_bytes + size
            > self._packed_route_limit_bytes
        ):
            (old_kind, old_key), old_size = self._packed_route_lru.popitem(
                last=False
            )
            old_cache = (
                self._packed_advance_plan_cache
                if old_kind == "advance"
                else self._packed_hole_plan_cache
            )
            old_cache.pop(old_key, None)
            self._packed_route_cache_bytes -= old_size
            self._packed_route_cache_evictions += 1
        cache[key] = plan
        self._packed_route_lru[(kind, key)] = size
        self._packed_route_cache_bytes += size
        return plan

    @property
    def packed_route_cache_stats(self):
        return {
            "bytes": self._packed_route_cache_bytes,
            "limit_bytes": self._packed_route_limit_bytes,
            "plans": len(self._packed_route_lru),
            "hits": self._packed_route_cache_hits,
            "misses": self._packed_route_cache_misses,
            "evictions": self._packed_route_cache_evictions,
        }

    def paired_sites(self, cut, channel):
        cut = int(cut)
        channel = int(channel)
        mask = self._suffix_masks[cut][channel]
        return tuple(site for site in self.frontier_sites[cut] if mask & (1 << site))

    def _group_transitions(self, site, mode):
        """Group MPO transitions with one common contraction topology."""
        groups = {}
        for left, right in self._transitions[int(site)]:
            if (
                left not in self._active_channel_sets[int(site)]
                or right not in self._active_channel_sets[int(site) + 1]
            ):
                continue
            left_pairs = (
                self._bond_pairs[site][left]
                if self.charge_resolved
                else None
            )
            right_pairs = (
                self._bond_pairs[site + 1][right]
                if self.charge_resolved
                else None
            )
            if mode == "left":
                key = (
                    right,
                    self.paired_sites(site, left),
                    self.block_shape(site, left),
                    left_pairs,
                    right_pairs,
                )
            elif mode == "right":
                key = (
                    left,
                    self.paired_sites(site + 1, right),
                    self.block_shape(site + 1, right),
                    left_pairs,
                    right_pairs,
                )
            elif mode == "hole":
                key = (
                    self.paired_sites(site, left),
                    self.paired_sites(site + 1, right),
                    self.block_shape(site, left),
                    self.block_shape(site + 1, right),
                    left_pairs,
                    right_pairs,
                )
            else:
                raise ValueError("transition group mode is invalid.")
            groups.setdefault(key, []).append((left, right))
        return tuple(tuple(transitions) for transitions in groups.values())

    def _operators_for_transitions(self, site, transitions):
        if len(transitions) == 1:
            left, right = transitions[0]
            return self.mpo_tensors[int(site)][left, right]
        key = (int(site), transitions)
        cached = self._operator_batches.get(key)
        if cached is None:
            cached = np.stack(
                [
                    self.mpo_tensors[int(site)][left, right]
                    for left, right in transitions
                ]
            )
            self._operator_batches[key] = cached
        return cached

    def block_shape(self, cut, channel):
        cut = int(cut)
        paired = set(self.paired_sites(cut, channel))
        shape = (
            [len(self._bond_pairs[cut][channel])]
            if self.charge_resolved
            else [self.virtual_bonds[cut], self.virtual_bonds[cut]]
        )
        for site in self.frontier_sites[cut]:
            shape.append(self.dims[site])
            if site in paired:
                shape.append(self.dims[site])
        return tuple(shape)

    @staticmethod
    def _add_charge(left, right):
        return tuple(a + b for a, b in zip(left, right))

    @staticmethod
    def _subtract_charge(left, right):
        return tuple(a - b for a, b in zip(left, right))

    def _charge_sum_support(self, sites):
        sites = tuple(int(site) for site in sites)
        cached = self._charge_sum_support_cache.get(sites)
        if cached is not None:
            return cached
        rank = len(self.local_qns[0][0])
        support = {tuple(0 for _ in range(rank))}
        for site in sites:
            support = {
                self._add_charge(prefix, charge)
                for prefix in support
                for charge in self.local_qns[site]
            }
        result = frozenset(support)
        self._charge_sum_support_cache[sites] = result
        return result

    def _physical_support_indices(self, cut, channel):
        if not self.charge_resolved:
            return None
        key = (int(cut), int(channel))
        if key in self._physical_support_cache:
            return self._physical_support_cache[key]
        cut, channel = key
        shape = self.block_shape(cut, channel)
        size = int(np.prod(shape, dtype=np.int64))
        frontier = self.frontier_sites[cut]
        free_sites = tuple(site for site in range(cut, self.nsites) if site not in frontier)
        free_support = self._charge_sum_support(free_sites)
        target = self.bond_qns[-1][0]
        paired = set(self.paired_sites(cut, channel))
        valid = np.zeros(size, dtype=bool)
        pairs = self._bond_pairs[cut][channel]
        for flat, configuration in enumerate(np.ndindex(*shape)):
            bra_bond, ket_bond = pairs[configuration[0]]
            bra_charge = self.bond_qns[cut][bra_bond]
            ket_charge = self.bond_qns[cut][ket_bond]
            offset = 1
            for site in frontier:
                bra_value = configuration[offset]
                offset += 1
                if site in paired:
                    ket_value = configuration[offset]
                    offset += 1
                else:
                    ket_value = bra_value
                bra_charge = self._add_charge(
                    bra_charge,
                    self.local_qns[site][bra_value],
                )
                ket_charge = self._add_charge(
                    ket_charge,
                    self.local_qns[site][ket_value],
                )
            valid[flat] = (
                self._subtract_charge(target, bra_charge) in free_support
                and self._subtract_charge(target, ket_charge) in free_support
            )
        indices = np.asarray(np.flatnonzero(valid), dtype=np.int32)
        result = None if indices.size == size else indices
        self._physical_support_cache[key] = result
        return result

    def storage_shape(self, cut, channel):
        support = self._physical_support_indices(cut, channel)
        return self.block_shape(cut, channel) if support is None else (int(support.size),)

    def _store_block(self, cut, channel, block):
        block = np.asarray(block)
        logical_shape = self.block_shape(cut, channel)
        if block.shape != logical_shape:
            block = block.reshape(logical_shape)
        support = self._physical_support_indices(cut, channel)
        return block if support is None else np.asarray(block).reshape(-1)[support].copy()

    def _logical_block(self, cut, channel, block):
        block = np.asarray(block)
        logical_shape = self.block_shape(cut, channel)
        support = self._physical_support_indices(cut, channel)
        if support is None:
            return block.reshape(logical_shape)
        result = np.zeros(logical_shape, dtype=block.dtype)
        result.reshape(-1)[support] = block.reshape(-1)
        return result

    def _message_block(self, message, channel):
        return self._logical_block(message.cut, int(channel), message.blocks[int(channel)])

    def _stack_message_blocks(self, message, channels):
        return np.stack([self._message_block(message, channel) for channel in channels])

    def _stored_message(self, cut, blocks):
        result = list(blocks)
        for channel in self._active_channels[int(cut)]:
            result[channel] = self._store_block(cut, channel, result[channel])
        return BlockFrontierMessage(int(cut), tuple(result))

    def _storage_coordinates(self, cut, channel):
        """Logical coordinates represented by one packed message block."""

        key = (int(cut), int(channel))
        cached = self._storage_coordinate_cache.get(key)
        if cached is not None:
            return cached
        shape = self.block_shape(*key)
        support = self._physical_support_indices(*key)
        flat = (
            np.arange(int(np.prod(shape, dtype=np.int64)), dtype=np.int32)
            if support is None
            else np.asarray(support, dtype=np.int32)
        )
        coordinates = np.empty((flat.size, len(shape)), dtype=np.int32)
        if flat.size == 0:
            self._storage_coordinate_cache[key] = coordinates
            return coordinates
        stride = int(np.prod(shape, dtype=np.int64))
        for axis, dimension in enumerate(shape):
            stride //= int(dimension)
            coordinates[:, axis] = (flat // stride) % int(dimension)
        self._storage_coordinate_cache[key] = coordinates
        return coordinates

    def _storage_inverse(self, cut, channel):
        """Map a logical flat index to its packed storage position."""

        key = (int(cut), int(channel))
        cached = self._storage_inverse_cache.get(key)
        if cached is not None:
            return cached
        shape = self.block_shape(*key)
        size = int(np.prod(shape, dtype=np.int64))
        support = self._physical_support_indices(*key)
        if support is None:
            inverse = np.arange(size, dtype=np.int32)
        else:
            inverse = np.full(size, -1, dtype=np.int32)
            inverse[np.asarray(support, dtype=np.int32)] = np.arange(
                len(support), dtype=np.int32
            )
        self._storage_inverse_cache[key] = inverse
        return inverse

    def _expanded_storage_coordinates(self, cut, channel):
        """Decode packed coordinates into bonds and bra/ket frontier values."""

        coordinates = self._storage_coordinates(cut, channel)
        key = (int(cut), int(channel))
        pairs = self._bond_pair_array_cache.get(key)
        if pairs is None:
            pairs = np.asarray(self._bond_pairs[key[0]][key[1]], dtype=np.int32)
            pairs = pairs.reshape(-1, 2)
            self._bond_pair_array_cache[key] = pairs
        pair_values = pairs[coordinates[:, 0]]
        bra = {}
        ket = {}
        paired = set(self.paired_sites(cut, channel))
        offset = 1
        for physical_site in self.frontier_sites[int(cut)]:
            bra[physical_site] = coordinates[:, offset]
            offset += 1
            if physical_site in paired:
                ket[physical_site] = coordinates[:, offset]
                offset += 1
            else:
                ket[physical_site] = bra[physical_site]
        return pair_values[:, 0], pair_values[:, 1], bra, ket

    def _tensor_support(self, site):
        """Return every structurally allowed U(1) entry of one graph tensor."""

        site = int(site)
        cached = self._tensor_support_cache.get(site)
        if cached is not None:
            return cached
        shape = self.tensor_shapes[site]
        physical_group = self.physical_groups[site]
        entries = []
        spectator_shape = tuple(self.dims[index] for index in physical_group[1:])
        for left, q_left in enumerate(self.bond_qns[site]):
            for owned, q_owned in enumerate(self.local_qns[site]):
                needed = self._add_charge(q_left, q_owned)
                for right, q_right in enumerate(self.bond_qns[site + 1]):
                    if q_right != needed:
                        continue
                    for spectators in np.ndindex(*spectator_shape):
                        physical = (owned, *spectators)
                        coordinate = (left, right, *physical)
                        entries.append(
                            (
                                int(np.ravel_multi_index(coordinate, shape)),
                                left,
                                right,
                                physical,
                            )
                        )
        result = tuple(entries)
        self._tensor_support_cache[site] = result
        return result

    def _compiled_route_builders(self):
        cached = self._compiled_route_builders_cache
        if cached is not None:
            return cached
        try:
            cpp = importlib.import_module("pyqed.mps.cpp_davidson")
            cached = (
                getattr(cpp, "build_packed_advance_routes", None),
                getattr(cpp, "build_packed_hole_routes", None),
            )
        except Exception:
            cached = (None, None)
        self._compiled_route_builders_cache = cached
        return cached

    @staticmethod
    def _physical_coordinate_matrix(values, sites, size):
        sites = tuple(sites)
        if not sites:
            return np.empty((int(size), 0), dtype=np.int32)
        return np.ascontiguousarray(
            np.stack([values[site] for site in sites], axis=1),
            dtype=np.int32,
        )

    def _tensor_support_arrays(self, site):
        site = int(site)
        cached = self._tensor_support_array_cache.get(site)
        if cached is not None:
            return cached
        entries = self._tensor_support(site)
        cached = (
            np.asarray([entry[0] for entry in entries], dtype=np.int32),
            np.asarray([entry[1] for entry in entries], dtype=np.int32),
            np.asarray([entry[2] for entry in entries], dtype=np.int32),
            np.asarray([entry[3] for entry in entries], dtype=np.int32),
        )
        self._tensor_support_array_cache[site] = cached
        return cached

    def _packed_advance_plan(
        self,
        direction,
        site,
        left_channel,
        right_channel,
        *,
        source_values=None,
        tensor_values=None,
        target_size=None,
        real_output=False,
    ):
        """Compile exact sparse routes between two packed U(1) messages."""

        key = (str(direction), int(site), int(left_channel), int(right_channel))
        contract = source_values is not None or tensor_values is not None
        if contract and (source_values is None or tensor_values is None):
            raise TypeError("source_values and tensor_values must be supplied together.")
        if not contract:
            cached = self._cached_packed_plan("advance", key)
            if cached is not None:
                return cached
        direction, site, left_channel, right_channel = key
        if direction == "left":
            source_cut, source_channel = site, left_channel
            target_cut, target_channel = site + 1, right_channel
            source_bond_axis = 0
            target_bond_axis = 1
        elif direction == "right":
            source_cut, source_channel = site + 1, right_channel
            target_cut, target_channel = site, left_channel
            source_bond_axis = 1
            target_bond_axis = 0
        else:
            raise ValueError("direction must be 'left' or 'right'.")

        source_bra_bond, source_ket_bond, source_bra, source_ket = (
            self._expanded_storage_coordinates(source_cut, source_channel)
        )
        target_pairs = {
            pair: index
            for index, pair in enumerate(self._bond_pairs[target_cut][target_channel])
        }
        target_inverse = self._storage_inverse(target_cut, target_channel)
        target_shape = self.block_shape(target_cut, target_channel)
        target_paired = set(self.paired_sites(target_cut, target_channel))
        local_sites = self.physical_groups[site]
        local_position = {physical_site: axis for axis, physical_site in enumerate(local_sites)}
        source_frontier = set(self.frontier_sites[source_cut])
        tensor_entries = self._tensor_support(site)

        entry_groups = {}
        overlap = tuple(index for index in local_sites if index in source_frontier)
        advance_builder, _hole_builder = self._compiled_route_builders()
        if advance_builder is not None:
            source_sites = self.frontier_sites[source_cut]
            target_sites = self.frontier_sites[target_cut]
            tensor_flat, tensor_left, tensor_right, tensor_physical = (
                self._tensor_support_arrays(site)
            )
            pair_lookup = np.full(
                (
                    self.virtual_bonds[target_cut],
                    self.virtual_bonds[target_cut],
                ),
                -1,
                dtype=np.int32,
            )
            for pair_index, (bra_bond, ket_bond) in enumerate(
                self._bond_pairs[target_cut][target_channel]
            ):
                pair_lookup[bra_bond, ket_bond] = pair_index
            compiled = advance_builder(
                direction == "left",
                np.asarray(source_bra_bond, dtype=np.int32),
                np.asarray(source_ket_bond, dtype=np.int32),
                np.asarray(source_sites, dtype=np.int64),
                self._physical_coordinate_matrix(
                    source_bra, source_sites, len(source_bra_bond)
                ),
                self._physical_coordinate_matrix(
                    source_ket, source_sites, len(source_bra_bond)
                ),
                tensor_flat,
                tensor_left,
                tensor_right,
                np.asarray(local_sites, dtype=np.int64),
                np.ascontiguousarray(tensor_physical, dtype=np.int32),
                np.asarray(overlap, dtype=np.int64),
                np.asarray(target_sites, dtype=np.int64),
                np.asarray(
                    [site in target_paired for site in target_sites],
                    dtype=np.uint8,
                ),
                np.asarray(self.dims, dtype=np.int64),
                pair_lookup,
                np.asarray(target_shape, dtype=np.int64),
                np.asarray(target_inverse, dtype=np.int32),
                np.asarray(
                    self.mpo_tensors[site][left_channel, right_channel],
                    dtype=np.complex128,
                ),
                source_values,
                tensor_values,
                0 if target_size is None else int(target_size),
                bool(real_output),
            )
            if contract:
                self.packed_route_backend = "cpp-fused"
                return np.asarray(compiled)
            result = tuple(
                np.asarray(value, dtype=np.int32)
                for value in compiled[:4]
            ) + (
                np.asarray(
                    np.asarray(compiled[4]).real
                    if self.dtype.kind != "c"
                    else compiled[4],
                    dtype=self.dtype,
                ),
            )
            self.packed_route_backend = "cpp"
            return self._cache_packed_plan("advance", key, result)
        for flat, left_bond, right_bond, physical in tensor_entries:
            bond = left_bond if source_bond_axis == 0 else right_bond
            values = tuple(physical[local_position[index]] for index in overlap)
            entry_groups.setdefault((bond, values), []).append(
                (flat, left_bond, right_bond, physical)
            )

        source_indices = []
        target_indices = []
        bra_indices = []
        ket_indices = []
        operator_values = []
        operator = self.mpo_tensors[site][left_channel, right_channel]
        for source_index in range(len(source_bra_bond)):
            bra_key = (
                int(source_bra_bond[source_index]),
                tuple(int(source_bra[index][source_index]) for index in overlap),
            )
            ket_key = (
                int(source_ket_bond[source_index]),
                tuple(int(source_ket[index][source_index]) for index in overlap),
            )
            for bra_entry in entry_groups.get(bra_key, ()):
                bra_flat, bra_left, bra_right, bra_physical = bra_entry
                for ket_entry in entry_groups.get(ket_key, ()):
                    ket_flat, ket_left, ket_right, ket_physical = ket_entry
                    coefficient = operator[
                        bra_physical[0], ket_physical[0]
                    ]
                    if coefficient == 0:
                        continue
                    target_pair = (
                        (bra_left, ket_left)
                        if target_bond_axis == 0
                        else (bra_right, ket_right)
                    )
                    pair_index = target_pairs.get(target_pair)
                    if pair_index is None:
                        continue
                    target_coordinate = [pair_index]
                    compatible = True
                    for physical_site in self.frontier_sites[target_cut]:
                        if physical_site in local_position:
                            bra_value = bra_physical[local_position[physical_site]]
                            ket_value = ket_physical[local_position[physical_site]]
                        else:
                            bra_value = int(source_bra[physical_site][source_index])
                            ket_value = int(source_ket[physical_site][source_index])
                        target_coordinate.append(bra_value)
                        if physical_site in target_paired:
                            target_coordinate.append(ket_value)
                        elif bra_value != ket_value:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    logical_target = int(
                        np.ravel_multi_index(tuple(target_coordinate), target_shape)
                    )
                    packed_target = int(target_inverse[logical_target])
                    if packed_target < 0:
                        continue
                    source_indices.append(source_index)
                    target_indices.append(packed_target)
                    bra_indices.append(bra_flat)
                    ket_indices.append(ket_flat)
                    operator_values.append(coefficient)

        result = tuple(
            np.asarray(values, dtype=dtype)
            for values, dtype in (
                (source_indices, np.int32),
                (target_indices, np.int32),
                (bra_indices, np.int32),
                (ket_indices, np.int32),
                (operator_values, self.dtype),
            )
        )
        return self._cache_packed_plan("advance", key, result)

    def _packed_advance(self, direction, message, tensors, site):
        """Advance a charge-resolved message without logical dense tensors."""

        site = int(site)
        source_cut = site if direction == "left" else site + 1
        target_cut = site + 1 if direction == "left" else site
        message = self._validated_message(message, source_cut)
        tensor = np.asarray(tensors[site])
        dtype = np.result_type(
            self.dtype,
            tensor.dtype,
            *[block.dtype for block in message.blocks if block is not None],
        )
        result = [np.zeros(0, dtype=dtype) for _ in range(self.mpo_bonds[target_cut])]
        for channel in self._active_channels[target_cut]:
            result[channel] = np.zeros(self.storage_shape(target_cut, channel), dtype=dtype)
        groups = (
            self._left_transition_groups[site]
            if direction == "left"
            else self._right_transition_groups[site]
        )
        tensor_flat = tensor.reshape(-1)
        fused_advance = (
            self._packed_route_limit_bytes == 0
            and self._compiled_route_builders()[0] is not None
        )
        self.last_packed_advance_peak_bytes = 0
        for transitions in groups:
            for left_channel, right_channel in transitions:
                source_channel = left_channel if direction == "left" else right_channel
                target_channel = right_channel if direction == "left" else left_channel
                if fused_advance:
                    contribution = self._packed_advance_plan(
                        direction,
                        site,
                        left_channel,
                        right_channel,
                        source_values=np.asarray(
                            message.blocks[source_channel]
                        ).reshape(-1),
                        tensor_values=tensor_flat,
                        target_size=result[target_channel].size,
                        real_output=np.dtype(dtype).kind != "c",
                    )
                    result[target_channel].reshape(-1)[:] += contribution
                    continue
                (
                    source_index,
                    target_index,
                    bra_index,
                    ket_index,
                    coefficient,
                ) = self._packed_advance_plan(
                    direction, site, left_channel, right_channel
                )
                self.last_packed_advance_peak_bytes = max(
                    self.last_packed_advance_peak_bytes,
                    int(
                        sum(
                            value.nbytes
                            for value in (
                                source_index,
                                target_index,
                                bra_index,
                                ket_index,
                                coefficient,
                            )
                        )
                    ),
                )
                if source_index.size == 0:
                    continue
                values = (
                    np.asarray(message.blocks[source_channel]).reshape(-1)[source_index]
                    * np.conj(tensor_flat[bra_index])
                    * tensor_flat[ket_index]
                    * coefficient
                )
                np.add.at(result[target_channel].reshape(-1), target_index, values)
        return BlockFrontierMessage(target_cut, tuple(result))

    def _packed_hole_plan(
        self,
        site,
        left_channel,
        right_channel,
        *,
        left_values=None,
        right_values=None,
        real_output=False,
        dense_output=None,
    ):
        """Compile sparse routes for one charge-resolved local operator channel."""

        key = (int(site), int(left_channel), int(right_channel))
        bind = left_values is not None or right_values is not None
        if bind and (left_values is None or right_values is None):
            raise TypeError("left_values and right_values must be supplied together.")
        if not bind:
            cached = self._cached_packed_plan("hole", key)
            if cached is not None:
                return cached
        site, left_channel, right_channel = key
        left_bra_bond, left_ket_bond, left_bra, left_ket = (
            self._expanded_storage_coordinates(site, left_channel)
        )
        right_bra_bond, right_ket_bond, right_bra, right_ket = (
            self._expanded_storage_coordinates(site + 1, right_channel)
        )
        left_frontier = set(self.frontier_sites[site])
        right_frontier = set(self.frontier_sites[site + 1])
        common = tuple(sorted(left_frontier & right_frontier))
        local_sites = self.physical_groups[site]
        local_position = {physical_site: axis for axis, physical_site in enumerate(local_sites)}
        constrained = tuple(
            physical_site
            for physical_site in local_sites
            if physical_site in left_frontier or physical_site in right_frontier
        )
        _advance_builder, hole_builder = self._compiled_route_builders()
        if hole_builder is not None:
            left_sites = self.frontier_sites[site]
            right_sites = self.frontier_sites[site + 1]
            tensor_flat, tensor_left, tensor_right, tensor_physical = (
                self._tensor_support_arrays(site)
            )
            compiled = hole_builder(
                np.asarray(left_bra_bond, dtype=np.int32),
                np.asarray(left_ket_bond, dtype=np.int32),
                np.asarray(left_sites, dtype=np.int64),
                self._physical_coordinate_matrix(
                    left_bra, left_sites, len(left_bra_bond)
                ),
                self._physical_coordinate_matrix(
                    left_ket, left_sites, len(left_bra_bond)
                ),
                np.asarray(right_bra_bond, dtype=np.int32),
                np.asarray(right_ket_bond, dtype=np.int32),
                np.asarray(right_sites, dtype=np.int64),
                self._physical_coordinate_matrix(
                    right_bra, right_sites, len(right_bra_bond)
                ),
                self._physical_coordinate_matrix(
                    right_ket, right_sites, len(right_bra_bond)
                ),
                tensor_flat,
                tensor_left,
                tensor_right,
                np.asarray(local_sites, dtype=np.int64),
                np.ascontiguousarray(tensor_physical, dtype=np.int32),
                np.asarray(common, dtype=np.int64),
                np.asarray(constrained, dtype=np.int64),
                np.asarray(self.dims, dtype=np.int64),
                int(self.virtual_bonds[site + 1]),
                np.asarray(
                    self.mpo_tensors[site][left_channel, right_channel],
                    dtype=np.complex128,
                ),
                left_values,
                right_values,
                bool(real_output),
                dense_output,
            )
            if bind:
                self.packed_route_backend = "cpp-fused"
                if dense_output is not None:
                    return np.asarray(compiled)
                return tuple(np.asarray(value) for value in compiled)
            result = tuple(
                np.asarray(value, dtype=np.int32)
                for value in compiled[:4]
            ) + (
                np.asarray(
                    np.asarray(compiled[4]).real
                    if self.dtype.kind != "c"
                    else compiled[4],
                    dtype=self.dtype,
                ),
            )
            self.packed_route_backend = "cpp"
            return self._cache_packed_plan("hole", key, result)
        right_groups = {}
        for right_index in range(len(right_bra_bond)):
            common_key = tuple(
                (int(right_bra[index][right_index]), int(right_ket[index][right_index]))
                for index in common
            )
            right_groups.setdefault(common_key, []).append(right_index)
        tensor_groups = {}
        for flat, left_bond, right_bond, physical in self._tensor_support(site):
            values = tuple(physical[local_position[index]] for index in constrained)
            tensor_groups.setdefault((left_bond, right_bond, values), []).append(
                (flat, physical)
            )

        left_indices = []
        right_indices = []
        bra_indices = []
        ket_indices = []
        operator_values = []
        operator = self.mpo_tensors[site][left_channel, right_channel]
        for left_index in range(len(left_bra_bond)):
            common_key = tuple(
                (int(left_bra[index][left_index]), int(left_ket[index][left_index]))
                for index in common
            )
            for right_index in right_groups.get(common_key, ()):
                bra_values = []
                ket_values = []
                compatible = True
                for physical_site in constrained:
                    bra_value = ket_value = None
                    if physical_site in left_frontier:
                        bra_value = int(left_bra[physical_site][left_index])
                        ket_value = int(left_ket[physical_site][left_index])
                    if physical_site in right_frontier:
                        right_bra_value = int(right_bra[physical_site][right_index])
                        right_ket_value = int(right_ket[physical_site][right_index])
                        if bra_value is not None and (
                            bra_value != right_bra_value or ket_value != right_ket_value
                        ):
                            compatible = False
                            break
                        bra_value = right_bra_value
                        ket_value = right_ket_value
                    bra_values.append(bra_value)
                    ket_values.append(ket_value)
                if not compatible:
                    continue
                bra_key = (
                    int(left_bra_bond[left_index]),
                    int(right_bra_bond[right_index]),
                    tuple(bra_values),
                )
                ket_key = (
                    int(left_ket_bond[left_index]),
                    int(right_ket_bond[right_index]),
                    tuple(ket_values),
                )
                for bra_flat, bra_physical in tensor_groups.get(bra_key, ()):
                    for ket_flat, ket_physical in tensor_groups.get(ket_key, ()):
                        coefficient = operator[bra_physical[0], ket_physical[0]]
                        if coefficient == 0:
                            continue
                        left_indices.append(left_index)
                        right_indices.append(right_index)
                        bra_indices.append(bra_flat)
                        ket_indices.append(ket_flat)
                        operator_values.append(coefficient)

        result = tuple(
            np.asarray(values, dtype=dtype)
            for values, dtype in (
                (left_indices, np.int32),
                (right_indices, np.int32),
                (bra_indices, np.int32),
                (ket_indices, np.int32),
                (operator_values, self.dtype),
            )
        )
        return self._cache_packed_plan("hole", key, result)

    def _packed_hole_action(self, site, left, right, vector):
        """Apply a one-site hole directly from packed U(1) messages."""

        site = int(site)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        vector = np.asarray(vector).reshape(-1)
        size = int(np.prod(self.tensor_shapes[site], dtype=np.int64))
        if vector.size != size:
            raise ValueError(f"vector must contain {size} entries.")
        dtype = np.result_type(
            self.dtype,
            vector.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros(size, dtype=dtype)
        self.last_packed_hole_peak_bytes = 0
        for transitions in self._hole_transition_groups[site]:
            for left_channel, right_channel in transitions:
                left_index, right_index, bra_index, ket_index, coefficient = (
                    self._packed_hole_plan(site, left_channel, right_channel)
                )
                self.last_packed_hole_peak_bytes = max(
                    self.last_packed_hole_peak_bytes,
                    int(
                        sum(
                            value.nbytes
                            for value in (
                                left_index,
                                right_index,
                                bra_index,
                                ket_index,
                                coefficient,
                            )
                        )
                    ),
                )
                if left_index.size == 0:
                    continue
                values = (
                    np.asarray(left.blocks[left_channel]).reshape(-1)[left_index]
                    * np.asarray(right.blocks[right_channel]).reshape(-1)[right_index]
                    * coefficient
                    * vector[ket_index]
                )
                np.add.at(result, bra_index, values)
        return result

    def _packed_fixed_hole_routes(self, site, left, right, transitions=None):
        """Bind packed messages and return sparse ``(bra, ket, weight)`` routes."""

        site = int(site)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        transition_groups = (
            self._hole_transition_groups[site]
            if transitions is None
            else (tuple(transitions),)
        )
        routes = []
        fused_hole = (
            self._packed_route_limit_bytes == 0
            and self._compiled_route_builders()[1] is not None
        )
        self.last_packed_hole_peak_bytes = 0
        for group in transition_groups:
            for left_channel, right_channel in group:
                if fused_hole:
                    left_values = np.asarray(
                        left.blocks[left_channel]
                    ).reshape(-1)
                    right_values = np.asarray(
                        right.blocks[right_channel]
                    ).reshape(-1)
                    bra_index, ket_index, weight = self._packed_hole_plan(
                        site,
                        left_channel,
                        right_channel,
                        left_values=left_values,
                        right_values=right_values,
                        real_output=np.result_type(
                            self.dtype,
                            left_values.dtype,
                            right_values.dtype,
                        ).kind
                        != "c",
                    )
                    if bra_index.size:
                        routes.append((bra_index, ket_index, weight))
                    continue
                left_index, right_index, bra_index, ket_index, coefficient = (
                    self._packed_hole_plan(site, left_channel, right_channel)
                )
                self.last_packed_hole_peak_bytes = max(
                    self.last_packed_hole_peak_bytes,
                    int(
                        sum(
                            value.nbytes
                            for value in (
                                left_index,
                                right_index,
                                bra_index,
                                ket_index,
                                coefficient,
                            )
                        )
                    ),
                )
                if left_index.size == 0:
                    continue
                weight = (
                    np.asarray(left.blocks[left_channel]).reshape(-1)[left_index]
                    * np.asarray(right.blocks[right_channel]).reshape(-1)[right_index]
                    * coefficient
                )
                routes.append((bra_index, ket_index, weight))
        return tuple(routes)

    def _prepare_packed_hole_action(self, site, left, right):
        """Bind a sector-native U(1) hole without dense frontier workspaces."""

        site = int(site)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        shape = self.tensor_shapes[site]
        size = int(np.prod(shape, dtype=np.int64))
        native_dtype = np.dtype(
            np.result_type(
                self.dtype,
                *[block.dtype for block in left.blocks if block is not None],
                *[block.dtype for block in right.blocks if block is not None],
            )
        )
        build_dtype = np.dtype(
            np.complex128 if native_dtype.kind == "c" else np.float64
        )
        dense_bytes = size * size * build_dtype.itemsize
        fused_dense = (
            self._packed_route_limit_bytes == 0
            and self._compiled_route_builders()[1] is not None
            and dense_bytes <= self._packed_dense_action_limit_bytes
        )
        if fused_dense:
            matrix = np.zeros((size, size), dtype=build_dtype)
            for group in self._hole_transition_groups[site]:
                for left_channel, right_channel in group:
                    self._packed_hole_plan(
                        site,
                        left_channel,
                        right_channel,
                        left_values=np.asarray(
                            left.blocks[left_channel]
                        ).reshape(-1),
                        right_values=np.asarray(
                            right.blocks[right_channel]
                        ).reshape(-1),
                        real_output=build_dtype.kind != "c",
                        dense_output=matrix,
                    )
            dtype = self.compute_dtype
            if native_dtype.kind == "c" and dtype.kind != "c":
                dtype = np.dtype(
                    np.complex64 if dtype.itemsize <= 4 else np.complex128
                )
            matrix = np.asarray(matrix, dtype=dtype)

            def dense_action(vector):
                vector = np.asarray(vector)
                if vector.size != size:
                    raise ValueError(f"vector must contain {size} entries.")
                return matrix @ vector.reshape(-1)

            def dense_actions(vectors):
                vectors = np.asarray(vectors)
                if vectors.ndim != 2 or vectors.shape[1] != size:
                    raise ValueError(f"vectors must have shape (batch, {size}).")
                return vectors @ matrix.T

            dense_action.many = dense_actions
            dense_action.backend = "packed-u1-dense-fused"
            dense_action.packed_route_count = 0
            dense_action.packed_route_bytes = int(matrix.nbytes)
            dense_action.dense_action_bytes = int(matrix.nbytes)
            if self._compute_dtype_requested and dtype != native_dtype:
                dense_action.verify = lambda vector: self._packed_hole_action(
                    site, left, right, vector
                )
            return dense_action

        routes = self._packed_fixed_hole_routes(site, left, right)
        native_dtype = np.dtype(
            np.result_type(
                self.dtype,
                *[route[2].dtype for route in routes],
            )
        )
        dtype = self.compute_dtype
        if native_dtype.kind == "c" and dtype.kind != "c":
            dtype = np.dtype(np.complex64 if dtype.itemsize <= 4 else np.complex128)
        bound = tuple(
            (bra, ket, np.asarray(weight, dtype=dtype))
            for bra, ket, weight in routes
        )
        route_count = int(sum(len(route[0]) for route in bound))
        if route_count:
            bra = np.concatenate([route[0] for route in bound])
            ket = np.concatenate([route[1] for route in bound])
            weight = np.concatenate([route[2] for route in bound])
            sparse_action = coo_matrix(
                (weight, (bra, ket)),
                shape=(size, size),
                dtype=dtype,
            ).tocsr()
            sparse_action.sum_duplicates()
            sparse_action.sort_indices()
        else:
            sparse_action = coo_matrix((size, size), dtype=dtype).tocsr()
        sparse_bytes = int(
            sparse_action.data.nbytes
            + sparse_action.indices.nbytes
            + sparse_action.indptr.nbytes
        )

        def action_dtype_for(value):
            value_dtype = np.asarray(value).dtype
            if value_dtype.kind == "c" and dtype.kind != "c":
                return np.dtype(np.complex64 if dtype.itemsize <= 4 else np.complex128)
            return dtype

        def action(vector):
            vector = np.asarray(vector)
            if vector.size != size:
                raise ValueError(f"vector must contain {size} entries.")
            action_dtype = action_dtype_for(vector)
            flat = np.asarray(vector, dtype=action_dtype).reshape(-1)
            return np.asarray(sparse_action @ flat).reshape(-1)

        def actions(vectors):
            vectors = np.asarray(vectors)
            if vectors.ndim != 2 or vectors.shape[1] != size:
                raise ValueError(f"vectors must have shape (batch, {size}).")
            action_dtype = action_dtype_for(vectors)
            flat = np.asarray(vectors, dtype=action_dtype)
            return np.asarray(sparse_action @ flat.T).T

        action.many = actions
        action.backend = "packed-u1-csr"
        action.packed_route_count = route_count
        action.packed_route_bytes = sparse_bytes
        if self._compute_dtype_requested and dtype != native_dtype:
            action.verify = lambda vector: self._packed_hole_action(
                site, left, right, vector
            )
        return action

    def message_elements(self, cut):
        return sum(
            int(np.prod(self.storage_shape(cut, channel)))
            for channel in self._active_channels[int(cut)]
        )

    def logical_message_elements(self, cut):
        return sum(
            int(np.prod(self.block_shape(cut, channel)))
            for channel in self._active_channels[int(cut)]
        )

    @property
    def physical_packing_ratio(self):
        logical = sum(self.logical_message_elements(cut) for cut in range(self.nsites + 1))
        return self.total_message_elements / max(logical, 1)

    def dense_message_elements(self, cut):
        cut = int(cut)
        return int(
            self.virtual_bonds[cut] ** 2
            * self.mpo_bonds[cut]
            * np.prod(
                [self.dims[site] ** 2 for site in self.frontier_sites[cut]],
                dtype=np.int64,
            )
        )

    @property
    def peak_message_elements(self):
        return max(self.message_elements(cut) for cut in range(self.nsites + 1))

    @property
    def dense_peak_message_elements(self):
        return max(self.dense_message_elements(cut) for cut in range(self.nsites + 1))

    @property
    def total_message_elements(self):
        return sum(self.message_elements(cut) for cut in range(self.nsites + 1))

    @property
    def dense_total_message_elements(self):
        return sum(self.dense_message_elements(cut) for cut in range(self.nsites + 1))

    def _new_label(self):
        label = self._next_auxiliary_label
        self._next_auxiliary_label += 1
        return label

    def _block_labels_and_copies(self, cut, channel):
        copies = []
        if self.charge_resolved:
            pair_label = self._new_label()
            labels = [pair_label]
            copies.append(
                (
                    self._pair_selector(cut, channel),
                    (pair_label, self.bra_bonds[cut], self.ket_bonds[cut]),
                )
            )
        else:
            labels = [self.bra_bonds[cut], self.ket_bonds[cut]]
        paired = set(self.paired_sites(cut, channel))
        for site in self.frontier_sites[cut]:
            if site in paired:
                labels.extend((self.bra_physical[site], self.ket_physical[site]))
            else:
                shared = self._new_label()
                labels.append(shared)
                copies.append(
                    (
                        self._copies[self.dims[site]],
                        (
                            shared,
                            self.bra_physical[site],
                            self.ket_physical[site],
                        ),
                    )
                )
        return tuple(labels), tuple(copies)

    def _tensor_labels(self, site, *, bra):
        bonds = self.bra_bonds if bra else self.ket_bonds
        physical = self.bra_physical if bra else self.ket_physical
        return (
            bonds[site],
            bonds[site + 1],
            *(physical[index] for index in self.physical_groups[site]),
        )

    def _advance_expression(
        self,
        direction,
        site,
        left_channel,
        right_channel,
        *,
        batch_size=1,
    ):
        batch_size = int(batch_size)
        key = (
            "advance",
            direction,
            int(site),
            int(left_channel),
            int(right_channel),
            batch_size,
        )
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        site = int(site)
        if direction == "left":
            source_cut, source_channel = site, int(left_channel)
            target_cut, target_channel = site + 1, int(right_channel)
        elif direction == "right":
            source_cut, source_channel = site + 1, int(right_channel)
            target_cut, target_channel = site, int(left_channel)
        else:
            raise ValueError("direction must be 'left' or 'right'.")
        source_labels, source_copies = self._block_labels_and_copies(
            source_cut, source_channel
        )
        target_labels, target_copies = self._block_labels_and_copies(
            target_cut, target_channel
        )
        copies = source_copies + target_copies
        source_shape = self.block_shape(source_cut, source_channel)
        operator_shape = (self.dims[site], self.dims[site])
        operator_labels = (self.bra_physical[site], self.ket_physical[site])
        if batch_size > 1:
            batch_label = self._new_label()
            source_shape = (batch_size,) + source_shape
            source_labels = (batch_label,) + source_labels
            operator_shape = (batch_size,) + operator_shape
            operator_labels = (batch_label,) + operator_labels
        arguments = [
            source_shape,
            source_labels,
            self.tensor_shapes[site],
            self._tensor_labels(site, bra=True),
            self.tensor_shapes[site],
            self._tensor_labels(site, bra=False),
            operator_shape,
            operator_labels,
        ]
        for value, labels in copies:
            arguments.extend((value.shape, labels))
        expression = contract_expression(
            *arguments,
            target_labels,
            optimize=self.optimize,
        )
        cached = (expression, tuple(value for value, _labels in copies))
        self._expressions[key] = cached
        return cached

    def _validated_message(self, message, cut):
        if not isinstance(message, BlockFrontierMessage):
            raise TypeError("message must be a BlockFrontierMessage.")
        if message.cut != cut:
            raise ValueError(f"message belongs to cut {message.cut}, not cut {cut}.")
        if len(message.blocks) != self.mpo_bonds[cut]:
            raise ValueError("message has the wrong number of MPO-channel blocks.")
        for channel, block in enumerate(message.blocks):
            if channel not in self._active_channel_sets[cut]:
                if np.size(block) != 0:
                    raise ValueError(
                        f"inactive message block {channel} must be empty."
                    )
                continue
            if np.shape(block) != self.storage_shape(cut, channel):
                raise ValueError(f"message block {channel} has the wrong shape.")
        return message

    def left_boundary(self):
        blocks = [np.zeros(0, dtype=self.dtype) for _ in range(self.mpo_bonds[0])]
        blocks[0] = self._store_block(
            0,
            0,
            np.ones(self.block_shape(0, 0), dtype=self.dtype),
        )
        return BlockFrontierMessage(
            0,
            tuple(blocks),
        )

    def right_boundary(self):
        blocks = [np.zeros(0, dtype=self.dtype) for _ in range(self.mpo_bonds[-1])]
        blocks[0] = self._store_block(
            self.nsites,
            0,
            np.ones(self.block_shape(self.nsites, 0), dtype=self.dtype),
        )
        return BlockFrontierMessage(
            self.nsites,
            tuple(blocks),
        )

    def _empty_message(self, cut, dtype):
        result = [np.zeros(0, dtype=dtype) for _ in range(self.mpo_bonds[cut])]
        for channel in self._active_channels[cut]:
            result[channel] = np.zeros(self.block_shape(cut, channel), dtype=dtype)
        return result

    def advance_left(self, message, tensors, site):
        site = int(site)
        if self.charge_resolved and self.device == "cpu":
            return self._packed_advance("left", message, tensors, site)
        message = self._validated_message(message, site)
        tensor = np.asarray(tensors[site])
        dtype = np.result_type(
            self.dtype,
            tensor.dtype,
            *[block.dtype for block in message.blocks if block is not None],
        )
        result = self._empty_message(site + 1, dtype)
        for transitions in self._left_transition_groups[site]:
            left, right = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._advance_expression(
                "left",
                site,
                left,
                right,
                batch_size=batch_size,
            )
            source = self._message_block(message, left)
            operator = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                source = self._stack_message_blocks(
                    message,
                    [channel for channel, _ in transitions],
                )
            contribution = expression(
                source,
                tensor.conj(),
                tensor,
                operator,
                *copy_factors,
            )
            result[right] += contribution
        return self._stored_message(site + 1, result)

    def advance_right(self, message, tensors, site):
        site = int(site)
        if self.charge_resolved and self.device == "cpu":
            return self._packed_advance("right", message, tensors, site)
        message = self._validated_message(message, site + 1)
        tensor = np.asarray(tensors[site])
        dtype = np.result_type(
            self.dtype,
            tensor.dtype,
            *[block.dtype for block in message.blocks if block is not None],
        )
        result = self._empty_message(site, dtype)
        for transitions in self._right_transition_groups[site]:
            left, right = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._advance_expression(
                "right",
                site,
                left,
                right,
                batch_size=batch_size,
            )
            source = self._message_block(message, right)
            operator = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                source = self._stack_message_blocks(
                    message,
                    [channel for _, channel in transitions],
                )
            contribution = expression(
                source,
                tensor.conj(),
                tensor,
                operator,
                *copy_factors,
            )
            result[left] += contribution
        return self._stored_message(site, result)

    def build_left(self, tensors):
        messages = [None] * (self.nsites + 1)
        messages[0] = self.left_boundary()
        for site in range(self.nsites):
            messages[site + 1] = self.advance_left(messages[site], tensors, site)
        return messages

    def build_right(self, tensors):
        messages = [None] * (self.nsites + 1)
        messages[-1] = self.right_boundary()
        for site in range(self.nsites - 1, -1, -1):
            messages[site] = self.advance_right(messages[site + 1], tensors, site)
        return messages

    def scalar(self, tensors):
        message = self.left_boundary()
        for site in range(self.nsites):
            message = self.advance_left(message, tensors, site)
        return self.boundary_scalar(message, self.nsites)

    def boundary_scalar(self, message, cut):
        """Extract a scalar from a completed left or right block message."""
        cut = int(cut)
        if cut not in {0, self.nsites}:
            raise ValueError("a scalar can only be extracted at a boundary cut.")
        message = self._validated_message(message, cut)
        value = message.blocks[0]
        return np.asarray(value).reshape(()).item()

    def dense_message(self, message):
        """Expand an identity/charge-block message to dense MPO-frontier form."""
        message = self._validated_message(message, message.cut)
        cut = message.cut
        shape = [
            self.virtual_bonds[cut],
            self.virtual_bonds[cut],
            self.mpo_bonds[cut],
        ]
        for site in self.frontier_sites[cut]:
            shape.extend((self.dims[site], self.dims[site]))
        dense = np.zeros(shape, dtype=self.dtype)
        for channel in self._active_channels[cut]:
            block = self._message_block(message, channel)
            paired = set(self.paired_sites(cut, channel))
            for compressed_index in np.ndindex(*block.shape):
                offset = 0
                if self.charge_resolved:
                    bra, ket = self._bond_pairs[cut][channel][compressed_index[0]]
                    offset = 1
                else:
                    bra, ket = compressed_index[:2]
                    offset = 2
                physical = []
                for site in self.frontier_sites[cut]:
                    if site in paired:
                        physical.extend(compressed_index[offset : offset + 2])
                        offset += 2
                    else:
                        value = compressed_index[offset]
                        physical.extend((value, value))
                        offset += 1
                dense[(bra, ket, channel, *physical)] = block[compressed_index]
        return dense

    def uncharge_message(self, message):
        """Expand only the virtual charge-pair selector of a block message."""
        if not self.charge_resolved:
            return self._validated_message(message, message.cut)
        message = self._validated_message(message, message.cut)
        cut = message.cut
        result = [
            np.zeros(0, dtype=self.dtype)
            for _ in range(self.mpo_bonds[cut])
        ]
        for channel in self._active_channels[cut]:
            source = self._message_block(message, channel)
            shape = (
                self.virtual_bonds[cut],
                self.virtual_bonds[cut],
                *source.shape[1:],
            )
            target = np.zeros(shape, dtype=source.dtype)
            for pair, (bra, ket) in enumerate(
                self._bond_pairs[cut][channel]
            ):
                target[(bra, ket)] = source[pair]
            result[channel] = target
        return BlockFrontierMessage(cut, tuple(result))

    def _hole_expression(
        self,
        mode,
        site,
        left_channel,
        right_channel,
        *,
        batch_size=1,
        vector_batch_size=1,
    ):
        batch_size = int(batch_size)
        vector_batch_size = int(vector_batch_size)
        key = (
            "hole",
            mode,
            int(site),
            int(left_channel),
            int(right_channel),
            batch_size,
            vector_batch_size,
        )
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        if mode not in {"matrix", "action", "actions", "physical_block"}:
            raise ValueError(
                "hole mode must be 'matrix', 'action', 'actions', or "
                "'physical_block'."
            )
        site = int(site)
        left_labels, left_copies = self._block_labels_and_copies(
            site, int(left_channel)
        )
        right_labels, right_copies = self._block_labels_and_copies(
            site + 1, int(right_channel)
        )
        copies = left_copies + right_copies
        bra_labels = self._tensor_labels(site, bra=True)
        ket_labels = self._tensor_labels(site, bra=False)
        left_shape = self.block_shape(site, left_channel)
        right_shape = self.block_shape(site + 1, right_channel)
        operator_shape = (self.dims[site], self.dims[site])
        operator_labels = (self.bra_physical[site], self.ket_physical[site])
        if batch_size > 1:
            batch_label = self._new_label()
            left_shape = (batch_size,) + left_shape
            right_shape = (batch_size,) + right_shape
            left_labels = (batch_label,) + left_labels
            right_labels = (batch_label,) + right_labels
            operator_shape = (batch_size,) + operator_shape
            operator_labels = (batch_label,) + operator_labels
        arguments = [
            left_shape,
            left_labels,
            right_shape,
            right_labels,
            operator_shape,
            operator_labels,
        ]
        if mode in {"action", "actions"}:
            if mode == "actions":
                vector_label = self._new_label()
                arguments.extend(
                    (
                        (vector_batch_size, *self.tensor_shapes[site]),
                        (vector_label, *ket_labels),
                    )
                )
                output_labels = (vector_label, *bra_labels)
            else:
                arguments.extend((self.tensor_shapes[site], ket_labels))
                output_labels = bra_labels
        elif mode == "physical_block":
            for physical_site in self.physical_groups[site]:
                arguments.extend(
                    (
                        (self.dims[physical_site],),
                        (self.bra_physical[physical_site],),
                        (self.dims[physical_site],),
                        (self.ket_physical[physical_site],),
                    )
                )
            output_labels = (
                self.bra_bonds[site],
                self.bra_bonds[site + 1],
                self.ket_bonds[site],
                self.ket_bonds[site + 1],
            )
        else:
            output_labels = bra_labels + ket_labels
        for value, labels in copies:
            arguments.extend((value.shape, labels))
        expression = contract_expression(
            *arguments,
            output_labels,
            optimize=self.optimize,
        )
        cached = (expression, tuple(value for value, _labels in copies))
        self._expressions[key] = cached
        return cached

    def hole_matrix(self, site, left, right):
        site = int(site)
        if self.charge_resolved and self.device == "cpu":
            routes = self._packed_fixed_hole_routes(site, left, right)
            size = int(np.prod(self.tensor_shapes[site], dtype=np.int64))
            dtype = np.result_type(
                self.dtype,
                *[route[2].dtype for route in routes],
            )
            result = np.zeros((size, size), dtype=dtype)
            for bra, ket, weight in routes:
                np.add.at(result, (bra, ket), weight)
            return result
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        size = int(np.prod(self.tensor_shapes[site]))
        dtype = np.result_type(
            self.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros((size, size), dtype=dtype)
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._hole_expression(
                "matrix",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = self._message_block(left, left_channel)
            right_blocks = self._message_block(right, right_channel)
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = self._stack_message_blocks(
                    left, [channel for channel, _ in transitions]
                )
                right_blocks = self._stack_message_blocks(
                    right, [channel for _, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                operators,
                *copy_factors,
            )
            result += np.asarray(value).reshape(size, size)
        return result

    def _validated_physical_configuration(self, site, configuration, *, name):
        configuration = tuple(int(value) for value in configuration)
        physical_sites = self.physical_groups[int(site)]
        if len(configuration) != len(physical_sites):
            raise ValueError(
                f"{name} must contain {len(physical_sites)} physical values."
            )
        if any(
            value < 0 or value >= self.dims[physical_site]
            for value, physical_site in zip(configuration, physical_sites)
        ):
            raise ValueError(f"{name} contains an out-of-range physical value.")
        return configuration

    def hole_block(self, site, left, right, bra_configuration, ket_configuration):
        """Return one fixed-physical virtual block without a dense hole matrix."""
        site = int(site)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        bra_configuration = self._validated_physical_configuration(
            site, bra_configuration, name="bra_configuration"
        )
        ket_configuration = self._validated_physical_configuration(
            site, ket_configuration, name="ket_configuration"
        )
        if self.charge_resolved and self.device == "cpu":
            routes = self._packed_fixed_hole_routes(site, left, right)
            virtual_size = self.virtual_bonds[site] * self.virtual_bonds[site + 1]
            physical_shape = self.tensor_shapes[site][2:]
            physical_size = int(np.prod(physical_shape, dtype=np.int64))
            bra_physical = int(np.ravel_multi_index(bra_configuration, physical_shape))
            ket_physical = int(np.ravel_multi_index(ket_configuration, physical_shape))
            dtype = np.result_type(
                self.dtype,
                *[route[2].dtype for route in routes],
            )
            result = np.zeros((virtual_size, virtual_size), dtype=dtype)
            for bra, ket, weight in routes:
                selected = (
                    (bra % physical_size == bra_physical)
                    & (ket % physical_size == ket_physical)
                )
                if np.any(selected):
                    np.add.at(
                        result,
                        (
                            bra[selected] // physical_size,
                            ket[selected] // physical_size,
                        ),
                        weight[selected],
                    )
            return result
        virtual_size = self.virtual_bonds[site] * self.virtual_bonds[site + 1]
        dtype = np.result_type(
            self.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros((virtual_size, virtual_size), dtype=dtype)
        selectors = []
        for physical_site, bra_value, ket_value in zip(
            self.physical_groups[site], bra_configuration, ket_configuration
        ):
            identity = self._identities[self.dims[physical_site]]
            selectors.extend((identity[bra_value], identity[ket_value]))
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._hole_expression(
                "physical_block",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = self._message_block(left, left_channel)
            right_blocks = self._message_block(right, right_channel)
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = self._stack_message_blocks(
                    left, [channel for channel, _ in transitions]
                )
                right_blocks = self._stack_message_blocks(
                    right, [channel for _, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                operators,
                *selectors,
                *copy_factors,
            )
            result += np.asarray(value).reshape(virtual_size, virtual_size)
        return result

    def hole_blocks(
        self,
        site,
        left,
        right,
        bra_configurations,
        ket_configurations,
    ):
        """Return multiple fixed-physical blocks without a dense hole matrix."""
        bra_configurations = tuple(bra_configurations)
        ket_configurations = tuple(ket_configurations)
        if len(bra_configurations) != len(ket_configurations):
            raise ValueError(
                "bra_configurations and ket_configurations must have equal length."
            )
        if not bra_configurations:
            site = int(site)
            virtual_size = (
                self.virtual_bonds[site] * self.virtual_bonds[site + 1]
            )
            return np.empty(
                (0, virtual_size, virtual_size),
                dtype=self.dtype,
            )
        return np.stack(
            [
                self.hole_block(site, left, right, bra, ket)
                for bra, ket in zip(
                    bra_configurations,
                    ket_configurations,
                )
            ]
        )

    def hole_action(self, site, left, right, vector):
        site = int(site)
        if self.charge_resolved and self.device == "cpu":
            return self._packed_hole_action(site, left, right, vector)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        vector = np.asarray(vector).reshape(self.tensor_shapes[site])
        size = int(np.prod(self.tensor_shapes[site]))
        dtype = np.result_type(
            self.dtype,
            vector.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros(size, dtype=dtype)
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._hole_expression(
                "action",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = self._message_block(left, left_channel)
            right_blocks = self._message_block(right, right_channel)
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = self._stack_message_blocks(
                    left, [channel for channel, _ in transitions]
                )
                right_blocks = self._stack_message_blocks(
                    right, [channel for _, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                operators,
                vector,
                *copy_factors,
            )
            result += np.asarray(value).reshape(-1)
        return result

    def prepare_hole_action(self, site, left, right):
        """Bind fixed messages and grouped transition operands for a local solve."""

        site = int(site)
        if self.charge_resolved and self.device == "cpu":
            return self._prepare_packed_hole_action(site, left, right)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        routes = []
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._hole_expression(
                "action",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = self._message_block(left, left_channel)
            right_blocks = self._message_block(right, right_channel)
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = self._stack_message_blocks(
                    left, [channel for channel, _ in transitions]
                )
                right_blocks = self._stack_message_blocks(
                    right, [channel for _, channel in transitions]
                )
            xp = self._xp
            route_dtype = self.compute_dtype
            if route_dtype.kind != "c" and any(
                np.asarray(value).dtype.kind == "c"
                for value in (left_blocks, right_blocks, operators)
            ):
                route_dtype = np.dtype(
                    np.complex64
                    if route_dtype.itemsize <= 4
                    else np.complex128
                )
            routes.append(
                (
                    expression,
                    left_channel,
                    right_channel,
                    batch_size,
                    xp.asarray(left_blocks, dtype=route_dtype),
                    xp.asarray(right_blocks, dtype=route_dtype),
                    xp.asarray(operators, dtype=route_dtype),
                    tuple(
                        xp.asarray(factor, dtype=route_dtype)
                        for factor in copy_factors
                    ),
                )
            )

        shape = self.tensor_shapes[site]
        size = int(np.prod(shape))
        dtype = np.dtype(
            np.result_type(
                self.compute_dtype,
                *[
                    value.dtype
                    for route in routes
                    for value in route[4:7]
                ],
            )
        )
        native_dtype = np.dtype(
            np.result_type(
                self.dtype,
                *[block.dtype for block in left.blocks if block is not None],
                *[block.dtype for block in right.blocks if block is not None],
            )
        )

        def action_dtype_for(value):
            value_dtype = np.asarray(value).dtype
            if value_dtype.kind == "c" and dtype.kind != "c":
                return np.dtype(
                    np.complex64 if dtype.itemsize <= 4 else np.complex128
                )
            return dtype

        def action(vector):
            vector = np.asarray(vector)
            action_dtype = action_dtype_for(vector)
            xp = self._xp
            result = xp.zeros(size, dtype=action_dtype)
            tensor = xp.asarray(vector.reshape(shape), dtype=action_dtype)
            for (
                expression,
                _left_channel,
                _right_channel,
                _batch_size,
                left_blocks,
                right_blocks,
                operators,
                copies,
            ) in routes:
                operands = (
                    left_blocks,
                    right_blocks,
                    operators,
                    tensor,
                    *copies,
                )
                value = (
                    expression(*operands)
                    if self.device == "cpu"
                    else expression(*operands, backend="cupy")
                )
                result += value.reshape(-1)
            return xp.asnumpy(result) if self.device == "cuda" else np.asarray(result)

        def actions(vectors):
            vectors = np.asarray(vectors)
            if vectors.ndim != 2 or vectors.shape[1] != size:
                raise ValueError(f"vectors must have shape (batch, {size}).")
            vector_batch = vectors.shape[0]
            xp = self._xp
            action_dtype = action_dtype_for(vectors)
            result = xp.zeros(
                (vector_batch, size),
                dtype=action_dtype,
            )
            tensors = xp.asarray(
                vectors.reshape(vector_batch, *shape), dtype=action_dtype
            )
            for (
                _expression,
                left_channel,
                right_channel,
                transition_batch,
                left_blocks,
                right_blocks,
                operators,
                copies,
            ) in routes:
                expression, _copy_factors = self._hole_expression(
                    "actions",
                    site,
                    left_channel,
                    right_channel,
                    batch_size=transition_batch,
                    vector_batch_size=vector_batch,
                )
                operands = (
                    left_blocks,
                    right_blocks,
                    operators,
                    tensors,
                    *copies,
                )
                value = (
                    expression(*operands)
                    if self.device == "cpu"
                    else expression(*operands, backend="cupy")
                )
                result += value.reshape(vector_batch, size)
            return xp.asnumpy(result) if self.device == "cuda" else np.asarray(result)

        action.many = actions
        if self._compute_dtype_requested and dtype != native_dtype:
            action.verify = lambda vector: self.hole_action(site, left, right, vector)
        return action

    def hole_action_components(self, site, left, right, vector):
        """Yield exact sparse-MPO transition-group contributions.

        Keeping these contributions separate exposes the Hamiltonian channel
        range needed by AMEn-style subspace enrichment without assembling a
        dense effective Hamiltonian or retaining every contribution at once.
        Their sum is exactly :meth:`hole_action`.
        """
        site = int(site)
        if self.charge_resolved and self.device == "cpu":
            vector = np.asarray(vector).reshape(-1)
            size = int(np.prod(self.tensor_shapes[site], dtype=np.int64))
            if vector.size != size:
                raise ValueError(f"vector must contain {size} entries.")
            for transitions in self._hole_transition_groups[site]:
                routes = self._packed_fixed_hole_routes(
                    site, left, right, transitions
                )
                dtype = np.result_type(
                    self.dtype,
                    vector.dtype,
                    *[route[2].dtype for route in routes],
                )
                result = np.zeros(size, dtype=dtype)
                for bra, ket, weight in routes:
                    np.add.at(result, bra, weight * vector[ket])
                yield result
            return
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        vector = np.asarray(vector).reshape(self.tensor_shapes[site])
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._hole_expression(
                "action",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = self._message_block(left, left_channel)
            right_blocks = self._message_block(right, right_channel)
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = self._stack_message_blocks(
                    left, [channel for channel, _ in transitions]
                )
                right_blocks = self._stack_message_blocks(
                    right, [channel for _, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                operators,
                vector,
                *copy_factors,
            )
            yield np.asarray(value).reshape(-1)

    def hole_action_component_count(self, site):
        return len(self._hole_transition_groups[int(site)])

    def _enrichment_expression(self, direction, site, left_channel, right_channel):
        """Plan one single-layer ``L W A`` or ``A W R`` contribution."""
        key = (
            "enrichment",
            str(direction),
            int(site),
            int(left_channel),
            int(right_channel),
        )
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        site = int(site)
        if direction == "left":
            cut, channel = site, int(left_channel)
            row_labels = (
                self.bra_bonds[site],
                *(self.bra_physical[index] for index in self.physical_groups[site]),
            )
            column_labels = [
                self.ket_bonds[site + 1],
            ]
        elif direction == "right":
            cut, channel = site + 1, int(right_channel)
            row_labels = (
                self.bra_bonds[site + 1],
                *(self.bra_physical[index] for index in self.physical_groups[site]),
            )
            column_labels = [
                self.ket_bonds[site],
            ]
        else:
            raise ValueError("direction must be 'left' or 'right'.")

        source_labels, source_copies = self._block_labels_and_copies(cut, channel)
        local = set(self.physical_groups[site])
        paired = set(self.paired_sites(cut, channel))
        for physical_site in self.frontier_sites[cut]:
            if physical_site in local:
                continue
            column_labels.append(self.bra_physical[physical_site])
            if physical_site in paired:
                column_labels.append(self.ket_physical[physical_site])

        arguments = [
            self.block_shape(cut, channel),
            source_labels,
            self.tensor_shapes[site],
            self._tensor_labels(site, bra=False),
            (self.dims[site], self.dims[site]),
            (self.bra_physical[site], self.ket_physical[site]),
        ]
        identities = []
        for physical_site in self.physical_groups[site][1:]:
            identity = self._identities[self.dims[physical_site]]
            identities.append(identity)
            arguments.extend(
                (
                    identity.shape,
                    (
                        self.bra_physical[physical_site],
                        self.ket_physical[physical_site],
                    ),
                )
            )
        for value, labels in source_copies:
            arguments.extend((value.shape, labels))
        expression = contract_expression(
            *arguments,
            row_labels + tuple(column_labels),
            optimize=self.optimize,
        )
        cached = (
            expression,
            tuple(identities),
            tuple(value for value, _labels in source_copies),
            int(
                self.virtual_bonds[site if direction == "left" else site + 1]
                * np.prod(
                    [self.dims[index] for index in self.physical_groups[site]],
                    dtype=np.int64,
                )
            ),
        )
        self._expressions[key] = cached
        return cached

    def _packed_enrichment_component(
        self,
        direction,
        site,
        message,
        vector,
        left_channel,
        right_channel,
    ):
        """Build one open AMEn component directly from packed U(1) storage."""

        site = int(site)
        if direction == "left":
            cut, channel, source_bond_axis = site, int(left_channel), 0
            opposite_bond_axis = 2
        elif direction == "right":
            cut, channel, source_bond_axis = site + 1, int(right_channel), 1
            opposite_bond_axis = 1
        else:
            raise ValueError("direction must be 'left' or 'right'.")
        message = self._validated_message(message, cut)
        source_bra_bond, source_ket_bond, source_bra, source_ket = (
            self._expanded_storage_coordinates(cut, channel)
        )
        source = np.asarray(message.blocks[channel]).reshape(-1)
        tensor = np.asarray(vector).reshape(-1)
        local_sites = self.physical_groups[site]
        local_position = {physical_site: axis for axis, physical_site in enumerate(local_sites)}
        source_frontier = set(self.frontier_sites[cut])
        overlap = tuple(index for index in local_sites if index in source_frontier)
        outside = tuple(index for index in self.frontier_sites[cut] if index not in local_position)
        paired = set(self.paired_sites(cut, channel))

        tensor_groups = {}
        for flat, left_bond, right_bond, physical in self._tensor_support(site):
            bond = left_bond if source_bond_axis == 0 else right_bond
            values = tuple(physical[local_position[index]] for index in overlap)
            tensor_groups.setdefault((bond, values), []).append(
                (flat, left_bond, right_bond, physical)
            )

        row_shape = (
            self.virtual_bonds[cut],
            *(self.dims[index] for index in local_sites),
        )
        column_shape = [
            self.virtual_bonds[site + 1]
            if direction == "left"
            else self.virtual_bonds[site]
        ]
        for physical_site in outside:
            column_shape.append(self.dims[physical_site])
            if physical_site in paired:
                column_shape.append(self.dims[physical_site])
        row_size = int(np.prod(row_shape, dtype=np.int64))
        column_size = int(np.prod(column_shape, dtype=np.int64))
        operator = self.mpo_tensors[site][left_channel, right_channel]
        dtype = np.result_type(self.dtype, source.dtype, tensor.dtype)
        result = np.zeros((row_size, column_size), dtype=dtype)
        for source_index in range(len(source)):
            ket_key = (
                int(source_ket_bond[source_index]),
                tuple(int(source_ket[index][source_index]) for index in overlap),
            )
            column_tail = []
            for physical_site in outside:
                column_tail.append(int(source_bra[physical_site][source_index]))
                if physical_site in paired:
                    column_tail.append(int(source_ket[physical_site][source_index]))
            for ket_flat, ket_left, ket_right, ket_physical in tensor_groups.get(
                ket_key, ()
            ):
                bra_physical = list(ket_physical)
                compatible = True
                for physical_site in local_sites[1:]:
                    if physical_site in source_frontier and (
                        int(source_bra[physical_site][source_index])
                        != bra_physical[local_position[physical_site]]
                    ):
                        compatible = False
                        break
                if not compatible:
                    continue
                owned_bra_values = (
                    (int(source_bra[site][source_index]),)
                    if site in source_frontier
                    else range(self.dims[site])
                )
                for owned_bra in owned_bra_values:
                    coefficient = operator[owned_bra, ket_physical[0]]
                    if coefficient == 0:
                        continue
                    bra_physical[0] = owned_bra
                    row = int(
                        np.ravel_multi_index(
                            (
                                int(source_bra_bond[source_index]),
                                *bra_physical,
                            ),
                            row_shape,
                        )
                    )
                    opposite_bond = (
                        ket_right if opposite_bond_axis == 2 else ket_left
                    )
                    column = int(
                        np.ravel_multi_index(
                            (opposite_bond, *column_tail),
                            tuple(column_shape),
                        )
                    )
                    result[row, column] += (
                        source[source_index] * tensor[ket_flat] * coefficient
                    )
        return result

    def left_enrichment_components(self, site, left, vector):
        """Yield open ``L W A`` matrices for right-going AMEn enrichment."""
        site = int(site)
        left = self._validated_message(left, site)
        vector = np.asarray(vector).reshape(self.tensor_shapes[site])
        for left_channel, right_channel in self._transitions[site]:
            if (
                left_channel not in self._active_channel_sets[site]
                or right_channel not in self._active_channel_sets[site + 1]
            ):
                continue
            if self.charge_resolved and self.device == "cpu":
                yield self._packed_enrichment_component(
                    "left",
                    site,
                    left,
                    vector,
                    left_channel,
                    right_channel,
                )
                continue
            expression, identities, copies, row_size = self._enrichment_expression(
                "left", site, left_channel, right_channel
            )
            value = expression(
                self._message_block(left, left_channel),
                vector,
                self.mpo_tensors[site][left_channel, right_channel],
                *identities,
                *copies,
            )
            yield np.asarray(value).reshape(row_size, -1)

    def right_enrichment_components(self, site, right, vector):
        """Yield open ``A W R`` matrices for left-going AMEn enrichment."""
        site = int(site)
        right = self._validated_message(right, site + 1)
        vector = np.asarray(vector).reshape(self.tensor_shapes[site])
        for left_channel, right_channel in self._transitions[site]:
            if (
                left_channel not in self._active_channel_sets[site]
                or right_channel not in self._active_channel_sets[site + 1]
            ):
                continue
            if self.charge_resolved and self.device == "cpu":
                yield self._packed_enrichment_component(
                    "right",
                    site,
                    right,
                    vector,
                    left_channel,
                    right_channel,
                )
                continue
            expression, identities, copies, row_size = self._enrichment_expression(
                "right", site, left_channel, right_channel
            )
            value = expression(
                self._message_block(right, right_channel),
                vector,
                self.mpo_tensors[site][left_channel, right_channel],
                *identities,
                *copies,
            )
            yield np.asarray(value).reshape(row_size, -1)

    def enrichment_component_count(self, site):
        site = int(site)
        return sum(
            left in self._active_channel_sets[site]
            and right in self._active_channel_sets[site + 1]
            for left, right in self._transitions[site]
        )

    def hole_actions(self, site, left, right, vectors):
        """Apply one block-frontier hole to a batch of local vectors."""
        site = int(site)
        if self.charge_resolved and self.device == "cpu":
            return self._prepare_packed_hole_action(site, left, right).many(vectors)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        size = int(np.prod(self.tensor_shapes[site]))
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != size:
            raise ValueError(f"vectors must have shape (batch, {size}).")
        vector_batch_size = vectors.shape[0]
        reshaped = vectors.reshape(vector_batch_size, *self.tensor_shapes[site])
        dtype = np.result_type(
            self.dtype,
            vectors.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros((vector_batch_size, size), dtype=dtype)
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_factors = self._hole_expression(
                "actions",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
                vector_batch_size=vector_batch_size,
            )
            left_blocks = self._message_block(left, left_channel)
            right_blocks = self._message_block(right, right_channel)
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = self._stack_message_blocks(
                    left, [channel for channel, _ in transitions]
                )
                right_blocks = self._stack_message_blocks(
                    right, [channel for _, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                operators,
                reshaped,
                *copy_factors,
            )
            result += np.asarray(value).reshape(vector_batch_size, size)
        return result

    def _block_transition_paths(self, start, stop):
        """Enumerate sparse MPO-channel paths through a short site block."""
        start = int(start)
        stop = int(stop)
        key = (start, stop)
        cached = self._block_paths.get(key)
        if cached is not None:
            return cached
        if start < 0 or stop > self.nsites or start >= stop:
            raise ValueError("block bounds are invalid.")
        outgoing = []
        for site in range(start, stop):
            by_left = {}
            for left, right in self._transitions[site]:
                if (
                    left in self._active_channel_sets[site]
                    and right in self._active_channel_sets[site + 1]
                ):
                    by_left.setdefault(left, []).append(right)
            outgoing.append(by_left)

        paths = []

        def extend(site, channels):
            if site == stop:
                paths.append(tuple(channels))
                return
            for right in outgoing[site - start].get(channels[-1], ()):
                extend(site + 1, (*channels, right))

        for channel in self._active_channels[start]:
            extend(start, (channel,))
        cached = tuple(paths)
        self._block_paths[key] = cached
        return cached

    def _block_hole_action_expression(
        self,
        start,
        stop,
        union_sites,
        merged_shape,
        left_channel,
        right_channel,
    ):
        key = (
            "block_hole_action",
            int(start),
            int(stop),
            tuple(union_sites),
            tuple(merged_shape),
            int(left_channel),
            int(right_channel),
        )
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        start = int(start)
        stop = int(stop)
        union_sites = tuple(int(site) for site in union_sites)
        merged_shape = tuple(int(dim) for dim in merged_shape)
        left_labels, left_copies = self._block_labels_and_copies(
            start,
            int(left_channel),
        )
        right_labels, right_copies = self._block_labels_and_copies(
            stop,
            int(right_channel),
        )
        copies = left_copies + right_copies
        ket_labels = (
            self.ket_bonds[start],
            self.ket_bonds[stop],
            *(self.ket_physical[site] for site in union_sites),
        )
        bra_labels = (
            self.bra_bonds[start],
            self.bra_bonds[stop],
            *(self.bra_physical[site] for site in union_sites),
        )
        arguments = [
            self.block_shape(start, left_channel),
            left_labels,
            self.block_shape(stop, right_channel),
            right_labels,
        ]
        for site in range(start, stop):
            arguments.extend(
                (
                    (self.dims[site], self.dims[site]),
                    (
                        self.bra_physical[site],
                        self.ket_physical[site],
                    ),
                )
            )
        arguments.extend((merged_shape, ket_labels))
        for value, labels in copies:
            arguments.extend((value.shape, labels))
        expression = contract_expression(
            *arguments,
            bra_labels,
            optimize=self.optimize,
        )
        cached = (expression, tuple(value for value, _labels in copies))
        self._expressions[key] = cached
        return cached

    def block_hole_action(
        self,
        start,
        stop,
        left,
        right,
        union_sites,
        merged_shape,
        vector,
    ):
        """Apply a short merged-block Hamiltonian termwise.

        Unlike an artificial supersite construction, this preserves the
        original frontier at every physical site and never forms the enlarged
        intermediate message immediately to the right of the merged tensor.
        """
        start = int(start)
        stop = int(stop)
        left = self._validated_message(left, start)
        right = self._validated_message(right, stop)
        merged_shape = tuple(int(dim) for dim in merged_shape)
        vector = np.asarray(vector).reshape(merged_shape)
        dtype = np.result_type(
            self.dtype,
            vector.dtype,
            *[block.dtype for block in left.blocks if block is not None],
            *[block.dtype for block in right.blocks if block is not None],
        )
        result = np.zeros(merged_shape, dtype=dtype)
        for path in self._block_transition_paths(start, stop):
            left_channel = path[0]
            right_channel = path[-1]
            left_block = self._message_block(left, left_channel)
            right_block = self._message_block(right, right_channel)
            if left_block is None or right_block is None:
                continue
            expression, copies = self._block_hole_action_expression(
                start,
                stop,
                union_sites,
                merged_shape,
                left_channel,
                right_channel,
            )
            operators = tuple(
                self.mpo_tensors[site][path[site - start], path[site - start + 1]]
                for site in range(start, stop)
            )
            result += expression(
                left_block,
                right_block,
                *operators,
                vector,
                *copies,
            )
        return result.reshape(-1)


__all__ = ["BlockFrontierMessage", "BlockMPOFrontier"]
