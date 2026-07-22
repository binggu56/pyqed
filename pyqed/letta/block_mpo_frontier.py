"""Identity-aware block frontiers for graph-tied tensors and an MPO.

This is an exact prototype of :mod:`pyqed.letta.mpo_frontier` in which the
MPO bond is represented by separate blocks.  Whenever the operator suffix
from one MPO channel is diagonal on a tied physical variable, the
corresponding bra/ket frontier pair is stored as one shared index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from opt_einsum import contract_expression


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
    ):
        self.dims = tuple(int(dim) for dim in dims)
        self.physical_sites = tuple(
            tuple(int(site) for site in sites) for sites in physical_sites
        )
        self.tensor_shapes = tuple(
            tuple(int(dim) for dim in shape) for shape in tensor_shapes
        )
        self.mpo_tensors = tuple(np.asarray(tensor) for tensor in mpo_tensors)
        self.optimize = optimize
        self.nsites = len(self.dims)
        if not self.dims or any(dim < 1 for dim in self.dims):
            raise ValueError("dims must contain positive local dimensions.")
        if not (
            len(self.physical_sites)
            == len(self.tensor_shapes)
            == len(self.mpo_tensors)
            == self.nsites
        ):
            raise ValueError("frontier inputs must contain one entry per site.")

        for site, (sites, shape) in enumerate(
            zip(self.physical_sites, self.tensor_shapes)
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
        for left_site, sites in enumerate(self.physical_sites):
            for physical_site in sites[1:]:
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
        self._suffix_masks = self._build_suffix_masks()
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
        self._copies = {dim: self._copy_tensor(dim) for dim in set(self.dims)}
        self._identities = {dim: np.eye(dim) for dim in set(self.dims)}
        self._expressions = {}
        self.dtype = np.dtype(
            np.result_type(*[tensor.dtype for tensor in self.mpo_tensors])
        )

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
        return tuple(masks)

    @property
    def plan_count(self):
        return len(self._expressions)

    def paired_sites(self, cut, channel):
        cut = int(cut)
        channel = int(channel)
        mask = self._suffix_masks[cut][channel]
        return tuple(site for site in self.frontier_sites[cut] if mask & (1 << site))

    def _group_transitions(self, site, mode):
        """Group MPO transitions with one common contraction topology."""
        groups = {}
        for left, right in self._transitions[int(site)]:
            if mode == "left":
                key = (right, self.paired_sites(site, left))
            elif mode == "right":
                key = (left, self.paired_sites(site + 1, right))
            elif mode == "hole":
                key = (
                    self.paired_sites(site, left),
                    self.paired_sites(site + 1, right),
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
        shape = [self.virtual_bonds[cut], self.virtual_bonds[cut]]
        for site in self.frontier_sites[cut]:
            shape.append(self.dims[site])
            if site in paired:
                shape.append(self.dims[site])
        return tuple(shape)

    def message_elements(self, cut):
        return sum(
            int(np.prod(self.block_shape(cut, channel)))
            for channel in range(self.mpo_bonds[int(cut)])
        )

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
        labels = [self.bra_bonds[cut], self.ket_bonds[cut]]
        copies = []
        paired = set(self.paired_sites(cut, channel))
        for site in self.frontier_sites[cut]:
            if site in paired:
                labels.extend((self.bra_physical[site], self.ket_physical[site]))
            else:
                shared = self._new_label()
                labels.append(shared)
                copies.append(
                    (
                        self.dims[site],
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
            *(physical[index] for index in self.physical_sites[site]),
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
        for dim, labels in copies:
            arguments.extend(((dim, dim, dim), labels))
        expression = contract_expression(
            *arguments,
            target_labels,
            optimize=self.optimize,
        )
        cached = (expression, tuple(dim for dim, _labels in copies))
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
            if np.shape(block) != self.block_shape(cut, channel):
                raise ValueError(f"message block {channel} has the wrong shape.")
        return message

    def left_boundary(self):
        return BlockFrontierMessage(
            0,
            (np.ones(self.block_shape(0, 0), dtype=self.dtype),),
        )

    def right_boundary(self):
        return BlockFrontierMessage(
            self.nsites,
            (np.ones(self.block_shape(self.nsites, 0), dtype=self.dtype),),
        )

    def _empty_message(self, cut, dtype):
        return [
            np.zeros(self.block_shape(cut, channel), dtype=dtype)
            for channel in range(self.mpo_bonds[cut])
        ]

    def advance_left(self, message, tensors, site):
        site = int(site)
        message = self._validated_message(message, site)
        tensor = np.asarray(tensors[site])
        dtype = np.result_type(
            self.dtype,
            tensor.dtype,
            *[block.dtype for block in message.blocks],
        )
        result = self._empty_message(site + 1, dtype)
        for transitions in self._left_transition_groups[site]:
            left, right = transitions[0]
            batch_size = len(transitions)
            expression, copy_dims = self._advance_expression(
                "left",
                site,
                left,
                right,
                batch_size=batch_size,
            )
            source = message.blocks[left]
            operator = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                source = np.stack(
                    [message.blocks[channel] for channel, _ in transitions]
                )
            contribution = expression(
                source,
                tensor.conj(),
                tensor,
                operator,
                *(self._copies[dim] for dim in copy_dims),
            )
            result[right] += contribution
        return BlockFrontierMessage(site + 1, tuple(result))

    def advance_right(self, message, tensors, site):
        site = int(site)
        message = self._validated_message(message, site + 1)
        tensor = np.asarray(tensors[site])
        dtype = np.result_type(
            self.dtype,
            tensor.dtype,
            *[block.dtype for block in message.blocks],
        )
        result = self._empty_message(site, dtype)
        for transitions in self._right_transition_groups[site]:
            left, right = transitions[0]
            batch_size = len(transitions)
            expression, copy_dims = self._advance_expression(
                "right",
                site,
                left,
                right,
                batch_size=batch_size,
            )
            source = message.blocks[right]
            operator = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                source = np.stack(
                    [message.blocks[channel] for _, channel in transitions]
                )
            contribution = expression(
                source,
                tensor.conj(),
                tensor,
                operator,
                *(self._copies[dim] for dim in copy_dims),
            )
            result[left] += contribution
        return BlockFrontierMessage(site, tuple(result))

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
        return self.boundary_scalar(self.build_left(tensors)[-1], self.nsites)

    def boundary_scalar(self, message, cut):
        """Extract a scalar from a completed left or right block message."""
        cut = int(cut)
        if cut not in {0, self.nsites}:
            raise ValueError("a scalar can only be extracted at a boundary cut.")
        message = self._validated_message(message, cut)
        if len(message.blocks) != 1:
            raise ValueError("a boundary message must contain one MPO channel.")
        value = message.blocks[0]
        return np.asarray(value).reshape(()).item()

    def _hole_expression(
        self,
        mode,
        site,
        left_channel,
        right_channel,
        *,
        batch_size=1,
    ):
        batch_size = int(batch_size)
        key = (
            "hole",
            mode,
            int(site),
            int(left_channel),
            int(right_channel),
            batch_size,
        )
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        if mode not in {"matrix", "action", "physical_block"}:
            raise ValueError(
                "hole mode must be 'matrix', 'action', or 'physical_block'."
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
        if mode == "action":
            arguments.extend((self.tensor_shapes[site], ket_labels))
            output_labels = bra_labels
        elif mode == "physical_block":
            for physical_site in self.physical_sites[site]:
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
        for dim, labels in copies:
            arguments.extend(((dim, dim, dim), labels))
        expression = contract_expression(
            *arguments,
            output_labels,
            optimize=self.optimize,
        )
        cached = (expression, tuple(dim for dim, _labels in copies))
        self._expressions[key] = cached
        return cached

    def hole_matrix(self, site, left, right):
        site = int(site)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        size = int(np.prod(self.tensor_shapes[site]))
        dtype = np.result_type(
            self.dtype,
            *[block.dtype for block in left.blocks],
            *[block.dtype for block in right.blocks],
        )
        result = np.zeros((size, size), dtype=dtype)
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_dims = self._hole_expression(
                "matrix",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = left.blocks[left_channel]
            right_blocks = right.blocks[right_channel]
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = np.stack(
                    [left.blocks[channel] for channel, _ in transitions]
                )
                right_blocks = np.stack(
                    [right.blocks[channel] for _, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                operators,
                *(self._copies[dim] for dim in copy_dims),
            )
            result += np.asarray(value).reshape(size, size)
        return result

    def _validated_physical_configuration(self, site, configuration, *, name):
        configuration = tuple(int(value) for value in configuration)
        physical_sites = self.physical_sites[int(site)]
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
        virtual_size = self.virtual_bonds[site] * self.virtual_bonds[site + 1]
        dtype = np.result_type(
            self.dtype,
            *[block.dtype for block in left.blocks],
            *[block.dtype for block in right.blocks],
        )
        result = np.zeros((virtual_size, virtual_size), dtype=dtype)
        selectors = []
        for physical_site, bra_value, ket_value in zip(
            self.physical_sites[site], bra_configuration, ket_configuration
        ):
            identity = self._identities[self.dims[physical_site]]
            selectors.extend((identity[bra_value], identity[ket_value]))
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_dims = self._hole_expression(
                "physical_block",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = left.blocks[left_channel]
            right_blocks = right.blocks[right_channel]
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = np.stack(
                    [left.blocks[channel] for channel, _ in transitions]
                )
                right_blocks = np.stack(
                    [right.blocks[channel] for _, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                operators,
                *selectors,
                *(self._copies[dim] for dim in copy_dims),
            )
            result += np.asarray(value).reshape(virtual_size, virtual_size)
        return result

    def hole_action(self, site, left, right, vector):
        site = int(site)
        left = self._validated_message(left, site)
        right = self._validated_message(right, site + 1)
        vector = np.asarray(vector).reshape(self.tensor_shapes[site])
        size = int(np.prod(self.tensor_shapes[site]))
        dtype = np.result_type(
            self.dtype,
            vector.dtype,
            *[block.dtype for block in left.blocks],
            *[block.dtype for block in right.blocks],
        )
        result = np.zeros(size, dtype=dtype)
        for transitions in self._hole_transition_groups[site]:
            left_channel, right_channel = transitions[0]
            batch_size = len(transitions)
            expression, copy_dims = self._hole_expression(
                "action",
                site,
                left_channel,
                right_channel,
                batch_size=batch_size,
            )
            left_blocks = left.blocks[left_channel]
            right_blocks = right.blocks[right_channel]
            operators = self._operators_for_transitions(site, transitions)
            if batch_size > 1:
                left_blocks = np.stack(
                    [left.blocks[channel] for channel, _ in transitions]
                )
                right_blocks = np.stack(
                    [right.blocks[channel] for _, channel in transitions]
                )
            value = expression(
                left_blocks,
                right_blocks,
                operators,
                vector,
                *(self._copies[dim] for dim in copy_dims),
            )
            result += np.asarray(value).reshape(-1)
        return result


__all__ = ["BlockFrontierMessage", "BlockMPOFrontier"]
