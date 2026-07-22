"""Directional frontier messages for graph-tied tensors and an MPO."""

from __future__ import annotations

import numpy as np
from opt_einsum import contract_expression


class MPOFrontier:
    r"""Cache exact left/right cut messages for a graph-tied double layer.

    A message at cut ``c`` carries the bra/ket virtual bond, the MPO bond, and
    both bra and ket values of every future physical variable referenced by a
    tensor to the left of the cut.  Its size is therefore controlled by the
    weighted physical-tie frontier, not by the full Hilbert-space dimension.
    """

    def __init__(
        self,
        dims,
        physical_sites,
        tensor_shapes,
        mpo_tensors,
        *,
        paired_sites=None,
        optimize="greedy",
    ):
        self.dims = tuple(int(dim) for dim in dims)
        self.physical_sites = tuple(tuple(sites) for sites in physical_sites)
        self.tensor_shapes = tuple(tuple(shape) for shape in tensor_shapes)
        self.mpo_tensors = tuple(np.asarray(tensor) for tensor in mpo_tensors)
        if paired_sites is None:
            paired_sites = range(len(self.dims))
        self.paired_sites = frozenset(int(site) for site in paired_sites)
        if any(site < 0 or site >= len(self.dims) for site in self.paired_sites):
            raise ValueError("paired_sites contains an invalid site.")
        self.optimize = optimize
        self.nsites = len(self.dims)
        if not (
            len(self.physical_sites)
            == len(self.tensor_shapes)
            == len(self.mpo_tensors)
            == self.nsites
        ):
            raise ValueError("frontier inputs must contain one entry per site.")

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
            if site not in self.paired_sites:
                identity = np.eye(self.dims[site], dtype=mpo.dtype)
                channel = np.trace(mpo, axis1=2, axis2=3) / self.dims[site]
                reconstructed = channel[:, :, None, None] * identity
                scale = max(float(np.linalg.norm(mpo)), 1.0)
                if not np.allclose(
                    mpo,
                    reconstructed,
                    rtol=0.0,
                    atol=256.0 * np.finfo(float).eps * scale,
                ):
                    raise ValueError(
                        f"unpaired MPO site {site} is not proportional to identity."
                    )
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
        self.operator_bonds = tuple(range(start, start + self.nsites + 1))
        start += self.nsites + 1
        self.bra_physical = tuple(range(start, start + self.nsites))
        start += self.nsites
        self.ket_physical = tuple(range(start, start + self.nsites))
        start += self.nsites
        self.local_ket_physical = tuple(range(start, start + self.nsites))
        self._identities = tuple(np.eye(dim) for dim in self.dims)
        self._expressions = {}

    @property
    def plan_count(self):
        return len(self._expressions)

    @property
    def peak_message_elements(self):
        return max(self.message_elements(cut) for cut in range(self.nsites + 1))

    def message_elements(self, cut):
        return int(np.prod(self.message_shape(cut)))

    @property
    def total_message_elements(self):
        return sum(self.message_elements(cut) for cut in range(self.nsites + 1))

    def message_shape(self, cut):
        cut = int(cut)
        shape = [
            self.virtual_bonds[cut],
            self.virtual_bonds[cut],
            self.mpo_bonds[cut],
        ]
        for site in self.frontier_sites[cut]:
            shape.append(self.dims[site])
            if site in self.paired_sites:
                shape.append(self.dims[site])
        return tuple(shape)

    def _message_labels(self, cut):
        labels = [
            self.bra_bonds[cut],
            self.ket_bonds[cut],
            self.operator_bonds[cut],
        ]
        for site in self.frontier_sites[cut]:
            labels.append(self.bra_physical[site])
            if site in self.paired_sites:
                labels.append(self.ket_physical[site])
        return tuple(labels)

    def _ket_label(self, site):
        return (
            self.ket_physical[site]
            if site in self.paired_sites
            else self.bra_physical[site]
        )

    def _tensor_labels(self, site, *, bra):
        bonds = self.bra_bonds if bra else self.ket_bonds
        physical = (
            self.bra_physical
            if bra
            else tuple(self._ket_label(index) for index in range(self.nsites))
        )
        return (
            bonds[site],
            bonds[site + 1],
            *(physical[index] for index in self.physical_sites[site]),
        )

    def left_boundary(self):
        return np.ones(self.message_shape(0))

    def right_boundary(self):
        return np.ones(self.message_shape(self.nsites))

    def _advance_expression(self, direction, site):
        key = ("advance", direction, int(site))
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        site = int(site)
        source_cut, target_cut = (
            (site, site + 1) if direction == "left" else (site + 1, site)
        )
        expression = contract_expression(
            self.message_shape(source_cut),
            self._message_labels(source_cut),
            self.tensor_shapes[site],
            self._tensor_labels(site, bra=True),
            self.tensor_shapes[site],
            self._tensor_labels(site, bra=False),
            self.mpo_tensors[site].shape,
            (
                self.operator_bonds[site],
                self.operator_bonds[site + 1],
                self.bra_physical[site],
                self._ket_label(site),
            ),
            self._message_labels(target_cut),
            optimize=self.optimize,
        )
        self._expressions[key] = expression
        return expression

    def advance_left(self, message, tensors, site):
        expression = self._advance_expression("left", site)
        tensor = tensors[int(site)]
        return expression(message, tensor.conj(), tensor, self.mpo_tensors[int(site)])

    def advance_right(self, message, tensors, site):
        expression = self._advance_expression("right", site)
        tensor = tensors[int(site)]
        return expression(message, tensor.conj(), tensor, self.mpo_tensors[int(site)])

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
        value = self.build_left(tensors)[-1]
        return np.asarray(value).reshape(()).item()

    def _hole_expression(self, mode, site):
        key = ("hole", mode, int(site))
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        if mode not in {"matrix", "action"}:
            raise ValueError("hole mode must be 'matrix' or 'action'.")
        site = int(site)
        bra_labels = (
            self.bra_bonds[site],
            self.bra_bonds[site + 1],
            *(self.bra_physical[index] for index in self.physical_sites[site]),
        )
        ket_labels = (
            self.ket_bonds[site],
            self.ket_bonds[site + 1],
            *(
                (
                    self.ket_physical[index]
                    if index in self.paired_sites
                    else self.local_ket_physical[index]
                )
                for index in self.physical_sites[site]
            ),
        )
        own_ket_label = (
            self.ket_physical[site]
            if site in self.paired_sites
            else self.local_ket_physical[site]
        )
        arguments = [
            self.message_shape(site),
            self._message_labels(site),
            self.message_shape(site + 1),
            self._message_labels(site + 1),
            self.mpo_tensors[site].shape,
            (
                self.operator_bonds[site],
                self.operator_bonds[site + 1],
                self.bra_physical[site],
                own_ket_label,
            ),
        ]
        for physical_site in self.physical_sites[site][1:]:
            if physical_site not in self.paired_sites:
                arguments.extend(
                    (
                        (self.dims[physical_site], self.dims[physical_site]),
                        (
                            self.bra_physical[physical_site],
                            self.local_ket_physical[physical_site],
                        ),
                    )
                )
        if mode == "action":
            arguments.extend((self.tensor_shapes[site], ket_labels))
            output_labels = bra_labels
        else:
            output_labels = bra_labels + ket_labels
        expression = contract_expression(
            *arguments,
            output_labels,
            optimize=self.optimize,
        )
        self._expressions[key] = expression
        return expression

    def hole_matrix(self, site, left, right):
        site = int(site)
        expression = self._hole_expression("matrix", site)
        identities = [
            self._identities[physical_site]
            for physical_site in self.physical_sites[site][1:]
            if physical_site not in self.paired_sites
        ]
        value = expression(left, right, self.mpo_tensors[site], *identities)
        size = int(np.prod(self.tensor_shapes[site]))
        return np.asarray(value).reshape(size, size)

    def _hole_block_expression(self, site):
        """Contract one fixed pair of local physical configurations."""
        key = ("hole", "physical_block", int(site))
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        site = int(site)
        arguments = [
            self.message_shape(site),
            self._message_labels(site),
            self.message_shape(site + 1),
            self._message_labels(site + 1),
            self.mpo_tensors[site].shape,
            (
                self.operator_bonds[site],
                self.operator_bonds[site + 1],
                self.bra_physical[site],
                (
                    self.ket_physical[site]
                    if site in self.paired_sites
                    else self.local_ket_physical[site]
                ),
            ),
        ]
        for physical_site in self.physical_sites[site][1:]:
            if physical_site not in self.paired_sites:
                arguments.extend(
                    (
                        (self.dims[physical_site], self.dims[physical_site]),
                        (
                            self.bra_physical[physical_site],
                            self.local_ket_physical[physical_site],
                        ),
                    )
                )
        for physical_site in self.physical_sites[site]:
            ket_label = (
                self.ket_physical[physical_site]
                if physical_site in self.paired_sites
                else self.local_ket_physical[physical_site]
            )
            arguments.extend(
                (
                    (self.dims[physical_site],),
                    (self.bra_physical[physical_site],),
                    (self.dims[physical_site],),
                    (ket_label,),
                )
            )
        output_labels = (
            self.bra_bonds[site],
            self.bra_bonds[site + 1],
            self.ket_bonds[site],
            self.ket_bonds[site + 1],
        )
        expression = contract_expression(
            *arguments,
            output_labels,
            optimize=self.optimize,
        )
        self._expressions[key] = expression
        return expression

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
        r"""Return one virtual block of the local effective operator.

        The local tensor is viewed as a matrix with combined virtual index
        ``(left, right)`` and physical-slice index ``(s_k, s_{P_k})``.  This
        method fixes the bra and ket physical slices before contraction, so it
        never constructs the full local matrix.
        """
        site = int(site)
        bra_configuration = self._validated_physical_configuration(
            site, bra_configuration, name="bra_configuration"
        )
        ket_configuration = self._validated_physical_configuration(
            site, ket_configuration, name="ket_configuration"
        )
        expression = self._hole_block_expression(site)
        identities = [
            self._identities[physical_site]
            for physical_site in self.physical_sites[site][1:]
            if physical_site not in self.paired_sites
        ]
        selectors = []
        for physical_site, bra_value, ket_value in zip(
            self.physical_sites[site], bra_configuration, ket_configuration
        ):
            selectors.extend(
                (
                    self._identities[physical_site][bra_value],
                    self._identities[physical_site][ket_value],
                )
            )
        value = expression(
            left,
            right,
            self.mpo_tensors[site],
            *identities,
            *selectors,
        )
        virtual_size = self.virtual_bonds[site] * self.virtual_bonds[site + 1]
        return np.asarray(value).reshape(virtual_size, virtual_size)

    def hole_action(self, site, left, right, vector):
        site = int(site)
        expression = self._hole_expression("action", site)
        identities = [
            self._identities[physical_site]
            for physical_site in self.physical_sites[site][1:]
            if physical_site not in self.paired_sites
        ]
        value = expression(
            left,
            right,
            self.mpo_tensors[site],
            *identities,
            np.asarray(vector).reshape(self.tensor_shapes[site]),
        )
        return np.asarray(value).reshape(-1)


__all__ = ["MPOFrontier"]
