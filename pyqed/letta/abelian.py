"""Abelian sector layouts for LETTA tensors.

The dense LETTA engine stores pair tensors as ``(left, site_i, site_j, right)``.
For Abelian calculations the allowed entries are not arbitrary sparsity: they
are block matrices between prefix-charge sectors.  This module records that
structure with :class:`pyqed.symmetry.IrrepTensor` so symmetry support
can be shared by masks, diagnostics, and later block-native local solves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.symmetry import (
    Irrep,
    Leg,
    IrrepTensor,
    OpIrrep,
    ProductSymmetry,
    U1Symmetry,
)


def _as_charge(value) -> tuple[int, ...]:
    if isinstance(value, Irrep):
        value = value.charge
    if isinstance(value, tuple):
        return tuple(int(x) for x in value)
    if isinstance(value, list):
        return tuple(int(x) for x in value)
    return (int(value),)


def _add_charges(*charges) -> tuple[int, ...]:
    charges = [_as_charge(charge) for charge in charges]
    if not charges:
        return ()
    rank = len(charges[0])
    if any(len(charge) != rank for charge in charges):
        raise ValueError("all Abelian charges must have the same rank.")
    return tuple(sum(charge[i] for charge in charges) for i in range(rank))


def _neg_charge(charge) -> tuple[int, ...]:
    charge = _as_charge(charge)
    return tuple(-x for x in charge)


def _sub_charges(left, right) -> tuple[int, ...]:
    return _add_charges(left, _neg_charge(right))


def _unique_sorted_charges(charges) -> list[tuple[int, ...]]:
    return sorted({_as_charge(charge) for charge in charges})


def _possible_sums(local_qns, *, rank: int | None = None) -> set[tuple[int, ...]]:
    if not local_qns:
        if rank is None:
            return {()}
        return {tuple(0 for _ in range(int(rank)))}
    rank = len(_as_charge(local_qns[0][0]))
    sums = {tuple(0 for _ in range(rank))}
    for site in local_qns:
        sums = {_add_charges(prefix, qn) for prefix in sums for qn in site}
    return sums


def _product_u1_symmetry(rank: int) -> ProductSymmetry:
    return ProductSymmetry(tuple(U1Symmetry(f"q{i}") for i in range(int(rank))), name="x".join(["U1"] * int(rank)))


def _sector_dims(labels: list[tuple[int, ...]]) -> dict[Irrep, int]:
    dims: dict[Irrep, int] = {}
    for label in labels:
        irrep = Irrep(_as_charge(label))
        dims[irrep] = dims.get(irrep, 0) + 1
    return dims


def _sector_positions(labels: list[tuple[int, ...]]) -> list[int]:
    counts: dict[tuple[int, ...], int] = {}
    positions = []
    for label in labels:
        label = _as_charge(label)
        positions.append(counts.get(label, 0))
        counts[label] = counts.get(label, 0) + 1
    return positions


def _sector_indices(labels: list[tuple[int, ...]]) -> dict[Irrep, list[int]]:
    indices: dict[Irrep, list[int]] = {}
    for index, label in enumerate(labels):
        indices.setdefault(Irrep(_as_charge(label)), []).append(index)
    return indices


@dataclass(frozen=True)
class AbelianLETTALocalBlock:
    """One IrrepTensor block embedded in a dense LETTA tensor."""

    physical: tuple[int, int]
    bra: Irrep
    ket: Irrep
    shape: tuple[int, int]
    flat_indices: np.ndarray
    coords: np.ndarray


@dataclass(frozen=True)
class AbelianXLETTALocalBlock:
    """One Abelian block embedded in an XLETTA local variable."""

    kind: str
    index: int
    physical: tuple[int, ...]
    bra: Irrep
    ket: Irrep
    shape: tuple[int, int]
    flat_indices: np.ndarray
    coords: np.ndarray


@dataclass(frozen=True)
class Layout:
    """Charge-sector layout for one Abelian LETTA chain.

    ``bond_qns[i]`` labels the LETTA bond immediately to the left of pair
    tensor ``i``.  For non-final pair tensors, the right bond is
    ``bond_qns[i + 1]`` and allowed entries satisfy
    ``q_right = q_left + q(site_i)``.  For the final pair tensor, allowed
    entries satisfy ``target = q_left + q(site_i) + q(site_j)``.
    """

    local_qns: list[list[tuple[int, ...]]]
    bond_qns: list[list[tuple[int, ...]]]
    target: tuple[int, ...]

    def __post_init__(self):
        local_qns = [[_as_charge(qn) for qn in site] for site in self.local_qns]
        bond_qns = [[_as_charge(qn) for qn in bond] for bond in self.bond_qns]
        target = _as_charge(self.target)
        if len(local_qns) < 2:
            raise ValueError("Abelian LETTA layout needs at least two physical sites.")
        if len(bond_qns) != len(local_qns) - 1:
            raise ValueError("bond_qns must have len(local_qns)-1 entries.")
        rank = len(target)
        all_charges = [target]
        all_charges.extend(qn for site in local_qns for qn in site)
        all_charges.extend(qn for bond in bond_qns for qn in bond)
        if any(len(qn) != rank for qn in all_charges):
            raise ValueError("all Abelian charges must have the same rank as target.")
        object.__setattr__(self, "local_qns", local_qns)
        object.__setattr__(self, "bond_qns", bond_qns)
        object.__setattr__(self, "target", target)

    @property
    def nsites(self) -> int:
        return len(self.local_qns)

    @property
    def nlocal_tensors(self) -> int:
        return self.nsites - 1

    @property
    def symmetry(self) -> ProductSymmetry:
        return _product_u1_symmetry(len(self.target))

    @property
    def bond_legs(self) -> list[Leg]:
        sym = self.symmetry
        return [Leg(_sector_dims(labels), symmetry=sym) for labels in self.bond_qns]

    @property
    def target_leg(self) -> Leg:
        return Leg({Irrep(self.target): 1}, symmetry=self.symmetry)

    def local_masks(self) -> list[np.ndarray]:
        masks = []
        for tensor_index in range(self.nsites - 2):
            left_labels = self.bond_qns[tensor_index]
            right_labels = self.bond_qns[tensor_index + 1]
            mask = np.zeros(
                (
                    len(left_labels),
                    len(self.local_qns[tensor_index]),
                    len(self.local_qns[tensor_index + 1]),
                    len(right_labels),
                ),
                dtype=bool,
            )
            for left, q_left in enumerate(left_labels):
                for si, q_site in enumerate(self.local_qns[tensor_index]):
                    needed = _add_charges(q_left, q_site)
                    for right, q_right in enumerate(right_labels):
                        if q_right == needed:
                            mask[left, si, :, right] = True
            masks.append(mask)

        left_labels = self.bond_qns[-1]
        final = np.zeros(
            (
                len(left_labels),
                len(self.local_qns[-2]),
                len(self.local_qns[-1]),
                1,
            ),
            dtype=bool,
        )
        for left, q_left in enumerate(left_labels):
            for si, q_i in enumerate(self.local_qns[-2]):
                for sj, q_j in enumerate(self.local_qns[-1]):
                    if _add_charges(q_left, q_i, q_j) == self.target:
                        final[left, si, sj, 0] = True
        masks.append(final)
        return masks

    def tensor_operator_grid(self, tensor_index: int, tensor: np.ndarray) -> dict[tuple[int, int], IrrepTensor]:
        """Return ``IrrepTensor`` operator blocks for every physical pair."""
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nlocal_tensors:
            raise IndexError("tensor_index out of range.")
        tensor = np.asarray(tensor)
        expected = self.local_masks()[tensor_index].shape
        if tensor.shape != expected:
            raise ValueError(f"tensor shape {tensor.shape} does not match Abelian layout shape {expected}.")
        if tensor_index == self.nsites - 2:
            return self._final_operator_grid(tensor_index, tensor)
        return self._internal_operator_grid(tensor_index, tensor)

    def local_tensor_blocks(self, tensor_index: int) -> tuple[AbelianLETTALocalBlock, ...]:
        """Return dense-coordinate maps for every local ``IrrepTensor`` block."""
        tensor_index = int(tensor_index)
        mask = self.local_masks()[tensor_index]
        dummy = np.zeros(mask.shape)
        grid = self.tensor_operator_grid(tensor_index, dummy)
        if tensor_index == self.nsites - 2:
            left_labels = self.bond_qns[tensor_index]
            right_indices = {Irrep(self.target): [0]}
        else:
            left_labels = self.bond_qns[tensor_index]
            right_indices = _sector_indices(self.bond_qns[tensor_index + 1])
        left_indices = _sector_indices(left_labels)

        blocks = []
        for physical, operator in grid.items():
            si, sj = physical
            for (bra, ket), block in operator.blocks.items():
                left = left_indices.get(ket, [])
                right = right_indices.get(bra, [])
                if not left or not right:
                    continue
                coords = []
                flat = []
                for right_index in right:
                    for left_index in left:
                        coord = (left_index, si, sj, right_index)
                        coords.append(coord)
                        flat.append(np.ravel_multi_index(coord, mask.shape))
                blocks.append(
                    AbelianLETTALocalBlock(
                        physical=(si, sj),
                        bra=bra,
                        ket=ket,
                        shape=tuple(int(x) for x in block.shape),
                        flat_indices=np.asarray(flat, dtype=np.int64),
                        coords=np.asarray(coords, dtype=np.int64),
                    )
                )
        return tuple(blocks)

    def _internal_operator_grid(self, tensor_index: int, tensor: np.ndarray) -> dict[tuple[int, int], IrrepTensor]:
        left_labels = self.bond_qns[tensor_index]
        right_labels = self.bond_qns[tensor_index + 1]
        left_leg = self.bond_legs[tensor_index]
        right_leg = self.bond_legs[tensor_index + 1]
        left_pos = _sector_positions(left_labels)
        right_pos = _sector_positions(right_labels)
        out = {}
        for si, q_i in enumerate(self.local_qns[tensor_index]):
            op = OpIrrep(q_i)
            for sj in range(len(self.local_qns[tensor_index + 1])):
                blocks = {}
                for left, q_left in enumerate(left_labels):
                    needed = _add_charges(q_left, q_i)
                    for right, q_right in enumerate(right_labels):
                        if q_right != needed:
                            continue
                        bra = Irrep(q_right)
                        ket = Irrep(q_left)
                        block = blocks.setdefault(
                            (bra, ket),
                            np.zeros((right_leg.sector_dim(bra), left_leg.sector_dim(ket)), dtype=tensor.dtype),
                        )
                        block[right_pos[right], left_pos[left]] = tensor[left, si, sj, right]
                out[(si, sj)] = IrrepTensor(right_leg, left_leg, op, blocks)
        return out

    def _final_operator_grid(self, tensor_index: int, tensor: np.ndarray) -> dict[tuple[int, int], IrrepTensor]:
        left_labels = self.bond_qns[tensor_index]
        left_leg = self.bond_legs[tensor_index]
        right_leg = self.target_leg
        left_pos = _sector_positions(left_labels)
        out = {}
        for si, q_i in enumerate(self.local_qns[-2]):
            for sj, q_j in enumerate(self.local_qns[-1]):
                op = OpIrrep(_add_charges(q_i, q_j))
                blocks = {}
                for left, q_left in enumerate(left_labels):
                    if _add_charges(q_left, q_i, q_j) != self.target:
                        continue
                    bra = Irrep(self.target)
                    ket = Irrep(q_left)
                    block = blocks.setdefault(
                        (bra, ket),
                        np.zeros((1, left_leg.sector_dim(ket)), dtype=tensor.dtype),
                    )
                    block[0, left_pos[left]] = tensor[left, si, sj, 0]
                out[(si, sj)] = IrrepTensor(right_leg, left_leg, op, blocks)
        return out

    def structural_support_size(self, tensor_index: int) -> int:
        return int(np.count_nonzero(self.local_masks()[int(tensor_index)]))

    def to_xlayout(self, *, view_qns=None, view_dim=None) -> "XLayout":
        """Return the corresponding Abelian layout for XLETTA tensors."""
        return XLayout.from_letta_layout(self, view_qns=view_qns, view_dim=view_dim)


@dataclass(frozen=True)
class TiedFrontierLayout:
    """Conditional charge sectors for an overlapping-pair LETTA chain.

    ``frontier_qns[i][s][a]`` is the cumulative charge carried by channel
    ``a`` after pair tensor ``i``, conditional on the shared physical state
    ``s`` at site ``i + 1``.  Conditioning avoids charging that shared site
    again when the next overlapping pair tensor is applied.
    """

    local_qns: list[list[tuple[int, ...]]]
    frontier_qns: list[list[list[tuple[int, ...]]]]
    target: tuple[int, ...]
    left_boundary: tuple[int, ...] | None = None

    def __post_init__(self):
        local_qns = [[_as_charge(qn) for qn in site] for site in self.local_qns]
        frontier_qns = [
            [[_as_charge(qn) for qn in channels] for channels in frontier]
            for frontier in self.frontier_qns
        ]
        target = _as_charge(self.target)
        left_boundary = self.left_boundary
        if left_boundary is None:
            left_boundary = tuple(0 for _ in target)
        left_boundary = _as_charge(left_boundary)

        if len(local_qns) < 2:
            raise ValueError("tied-frontier LETTA layout needs at least two physical sites.")
        if any(not site for site in local_qns):
            raise ValueError("every physical site must contain at least one charge label.")
        if len(frontier_qns) != len(local_qns) - 2:
            raise ValueError("frontier_qns must have len(local_qns)-2 entries.")
        for bond, frontier in enumerate(frontier_qns):
            if len(frontier) != len(local_qns[bond + 1]):
                raise ValueError(
                    f"frontier_qns[{bond}] must have one conditional entry "
                    f"for each state of shared site {bond + 1}."
                )
            dims = {len(channels) for channels in frontier}
            if len(dims) != 1 or not dims or next(iter(dims)) < 1:
                raise ValueError("all shared states on a frontier must have the same positive channel dimension.")

        rank = len(target)
        all_charges = [target, left_boundary]
        all_charges.extend(qn for site in local_qns for qn in site)
        all_charges.extend(qn for frontier in frontier_qns for channels in frontier for qn in channels)
        if any(len(qn) != rank for qn in all_charges):
            raise ValueError("all Abelian charges must have the same rank as target.")

        object.__setattr__(self, "local_qns", local_qns)
        object.__setattr__(self, "frontier_qns", frontier_qns)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "left_boundary", left_boundary)

    @property
    def nsites(self) -> int:
        return len(self.local_qns)

    @property
    def nlocal_tensors(self) -> int:
        return self.nsites - 1

    @property
    def bond_dims(self) -> tuple[int, ...]:
        return tuple(len(frontier[0]) for frontier in self.frontier_qns)

    def frontier_labels(
        self,
        bond: int,
        shared_state: int | None = None,
    ) -> tuple[tuple[int, ...], ...] | tuple[tuple[tuple[int, ...], ...], ...]:
        """Return labels at fixed shared state, or each channel's signature."""
        bond = int(bond)
        if bond < 0 or bond >= len(self.frontier_qns):
            raise IndexError("bond out of range.")
        frontier = self.frontier_qns[bond]
        if shared_state is not None:
            shared_state = int(shared_state)
            if shared_state < 0 or shared_state >= len(frontier):
                raise IndexError("shared_state out of range.")
            return tuple(frontier[shared_state])
        return tuple(tuple(channels[channel] for channels in frontier) for channel in range(len(frontier[0])))

    def local_masks(self) -> list[np.ndarray]:
        """Return dense masks obeying conditional cumulative-charge flow."""
        if self.nsites == 2:
            mask = np.zeros((1, len(self.local_qns[0]), len(self.local_qns[1]), 1), dtype=bool)
            for s0, q0 in enumerate(self.local_qns[0]):
                for s1, q1 in enumerate(self.local_qns[1]):
                    mask[0, s0, s1, 0] = _add_charges(self.left_boundary, q0, q1) == self.target
            return [mask]

        masks = []
        first = np.zeros(
            (1, len(self.local_qns[0]), len(self.local_qns[1]), self.bond_dims[0]),
            dtype=bool,
        )
        for s0, q0 in enumerate(self.local_qns[0]):
            for s1, q1 in enumerate(self.local_qns[1]):
                needed = _add_charges(self.left_boundary, q0, q1)
                for right, q_right in enumerate(self.frontier_qns[0][s1]):
                    first[0, s0, s1, right] = q_right == needed
        masks.append(first)

        for tensor in range(1, self.nlocal_tensors - 1):
            mask = np.zeros(
                (
                    self.bond_dims[tensor - 1],
                    len(self.local_qns[tensor]),
                    len(self.local_qns[tensor + 1]),
                    self.bond_dims[tensor],
                ),
                dtype=bool,
            )
            for left in range(self.bond_dims[tensor - 1]):
                for old_shared in range(len(self.local_qns[tensor])):
                    q_left = self.frontier_qns[tensor - 1][old_shared][left]
                    for new_shared, q_new in enumerate(self.local_qns[tensor + 1]):
                        needed = _add_charges(q_left, q_new)
                        for right, q_right in enumerate(self.frontier_qns[tensor][new_shared]):
                            mask[left, old_shared, new_shared, right] = q_right == needed
            masks.append(mask)

        final = np.zeros(
            (self.bond_dims[-1], len(self.local_qns[-2]), len(self.local_qns[-1]), 1),
            dtype=bool,
        )
        for left in range(self.bond_dims[-1]):
            for shared in range(len(self.local_qns[-2])):
                q_left = self.frontier_qns[-1][shared][left]
                for last, q_last in enumerate(self.local_qns[-1]):
                    final[left, shared, last, 0] = _add_charges(q_left, q_last) == self.target
        masks.append(final)
        return masks

    def structural_support_sizes(self) -> tuple[int, ...]:
        return tuple(int(np.count_nonzero(mask)) for mask in self.local_masks())

    @classmethod
    def from_state_vector(
        cls,
        state,
        local_qns,
        target,
        *,
        bond_dims=None,
        left_boundary=None,
        project: bool = False,
        sector_rtol: float = 1e-10,
        sector_atol: float = 0.0,
        truncate: bool = False,
    ) -> "TiedFrontierLayout":
        """Infer conditional frontier sectors from charge-block Schmidt ranks.

        At frontier ``i`` and fixed shared state ``s_{i+1}``, the state is
        reshaped between sites ``0..i`` and ``i+2..L-1``.  Each nonzero
        charge-block singular value contributes one channel carrying the
        cumulative charge through the fixed shared state.
        """
        local_qns = [[_as_charge(qn) for qn in site] for site in local_qns]
        target = _as_charge(target)
        if left_boundary is None:
            left_boundary = tuple(0 for _ in target)
        left_boundary = _as_charge(left_boundary)
        if len(local_qns) < 2 or any(not site for site in local_qns):
            raise ValueError("local_qns must describe at least two nonempty physical sites.")
        rank = len(target)
        if len(left_boundary) != rank or any(len(qn) != rank for site in local_qns for qn in site):
            raise ValueError("all Abelian charges must have the same rank as target.")
        if sector_rtol < 0 or sector_atol < 0:
            raise ValueError("sector tolerances must be nonnegative.")

        dims = tuple(len(site) for site in local_qns)
        state = np.asarray(state)
        if state.size != int(np.prod(dims, dtype=np.int64)):
            raise ValueError(f"state has size {state.size}, expected {int(np.prod(dims))}.")
        state = np.array(state.reshape(dims), copy=True)
        state_norm = float(np.linalg.norm(state.ravel()))
        if state_norm == 0.0:
            raise ValueError("state vector must be nonzero.")

        sector_mask = np.zeros(dims, dtype=bool)
        for config in np.ndindex(*dims):
            total = _add_charges(left_boundary, *(local_qns[site][s] for site, s in enumerate(config)))
            sector_mask[config] = total == target
        leakage = float(np.linalg.norm(state[~sector_mask]))
        tolerance = max(float(sector_atol), float(sector_rtol) * state_norm)
        if leakage > tolerance and not project:
            raise ValueError(
                f"state has target-sector leakage {leakage:.3e}; "
                "pass project=True to discard incompatible amplitudes."
            )
        state[~sector_mask] = 0
        if float(np.linalg.norm(state.ravel())) <= tolerance:
            raise ValueError("the target-sector projection of state is zero.")

        nfrontiers = len(local_qns) - 2
        if bond_dims is None:
            requested_dims = None
        elif np.isscalar(bond_dims):
            requested_dims = (int(bond_dims),) * nfrontiers
        else:
            requested_dims = tuple(int(dim) for dim in bond_dims)
            if len(requested_dims) != nfrontiers:
                raise ValueError("bond_dims must have one entry for each tied frontier.")
        if requested_dims is not None and any(dim < 1 for dim in requested_dims):
            raise ValueError("bond dimensions must be positive.")

        inferred = []
        for cut in range(nfrontiers):
            by_shared = []
            for shared, q_shared in enumerate(local_qns[cut + 1]):
                matrix = np.take(state, shared, axis=cut + 1).reshape(
                    int(np.prod(dims[: cut + 1], dtype=np.int64)),
                    int(np.prod(dims[cut + 2 :], dtype=np.int64)),
                )
                row_labels = []
                for config in np.ndindex(*dims[: cut + 1]):
                    row_labels.append(
                        _add_charges(
                            left_boundary,
                            *(local_qns[site][s] for site, s in enumerate(config)),
                            q_shared,
                        )
                    )
                column_labels = []
                for config in np.ndindex(*dims[cut + 2 :]):
                    column_labels.append(
                        _add_charges(
                            *(local_qns[cut + 2 + site][s] for site, s in enumerate(config))
                        )
                    )

                row_sectors = {charge: [] for charge in row_labels}
                column_sectors = {charge: [] for charge in column_labels}
                for index, charge in enumerate(row_labels):
                    row_sectors[charge].append(index)
                for index, charge in enumerate(column_labels):
                    column_sectors[charge].append(index)
                feasible = [
                    charge
                    for charge in _unique_sorted_charges(row_labels)
                    if _sub_charges(target, charge) in column_sectors
                ]

                spectra = []
                for charge in feasible:
                    rows = row_sectors[charge]
                    columns = column_sectors[_sub_charges(target, charge)]
                    values = np.linalg.svd(matrix[np.ix_(rows, columns)], compute_uv=False)
                    spectra.extend((float(value), charge) for value in values)
                scale = max((value for value, _ in spectra), default=0.0)
                cutoff = max(float(sector_atol), float(sector_rtol) * scale)
                active = [(value, charge) for value, charge in spectra if value > cutoff]
                active.sort(key=lambda item: (-item[0], item[1]))
                by_shared.append((active, feasible))
            inferred.append(by_shared)

        if requested_dims is None:
            requested_dims = tuple(max(len(active) for active, _ in frontier) for frontier in inferred)

        frontier_qns = []
        for cut, (frontier, dim) in enumerate(zip(inferred, requested_dims)):
            conditional = []
            for shared, (active, feasible) in enumerate(frontier):
                if len(active) > dim and not truncate:
                    raise ValueError(
                        f"bond dimension {dim} truncates rank {len(active)} "
                        f"at frontier {cut}, shared state {shared}; pass truncate=True to allow it."
                    )
                labels = [charge for _, charge in active[:dim]]
                if len(labels) < dim:
                    if not feasible:
                        raise ValueError(
                            f"frontier {cut}, shared state {shared} has no charge compatible with target."
                        )
                    extra = 0
                    while len(labels) < dim:
                        labels.append(feasible[extra % len(feasible)])
                        extra += 1
                conditional.append(labels)
            frontier_qns.append(conditional)

        return cls(
            local_qns=local_qns,
            frontier_qns=frontier_qns,
            target=target,
            left_boundary=left_boundary,
        )


@dataclass(frozen=True)
class XLayout:
    """Charge-sector layout for dense masked Abelian XLETTA.

    ``prefix_qns[i]`` labels the charge before site ``i`` for ``i=0`` and
    after site ``i-1`` for ``i>0``.  Therefore ``len(prefix_qns) == nsites``:
    XLETTA needs an explicit prefix sector before the terminal physical site.

    ``view_qns[i-1]`` labels the lifted ``u_i``/``v_i`` legs tied to physical
    site ``i``.
    """

    local_qns: list[list[tuple[int, ...]]]
    prefix_qns: list[list[tuple[int, ...]]]
    target: tuple[int, ...]
    view_qns: list[list[tuple[int, ...]]] | None = None

    def __post_init__(self):
        local_qns = [[_as_charge(qn) for qn in site] for site in self.local_qns]
        prefix_qns = [[_as_charge(qn) for qn in bond] for bond in self.prefix_qns]
        target = _as_charge(self.target)
        view_qns = self.view_qns
        if view_qns is None:
            view_qns = [list(site) for site in local_qns[1:]]
        else:
            view_qns = [[_as_charge(qn) for qn in site] for site in view_qns]

        if len(local_qns) < 2:
            raise ValueError("Abelian XLETTA layout needs at least two physical sites.")
        if len(prefix_qns) != len(local_qns):
            raise ValueError("prefix_qns must have one entry before each physical site.")
        if len(prefix_qns[0]) != 1:
            raise ValueError("prefix_qns[0] must contain exactly one left-boundary charge.")
        if len(view_qns) != len(local_qns) - 1:
            raise ValueError("view_qns must have one entry for each shared physical site.")

        rank = len(target)
        all_charges = [target]
        all_charges.extend(qn for site in local_qns for qn in site)
        all_charges.extend(qn for bond in prefix_qns for qn in bond)
        all_charges.extend(qn for site in view_qns for qn in site)
        if any(len(qn) != rank for qn in all_charges):
            raise ValueError("all Abelian charges must have the same rank as target.")

        object.__setattr__(self, "local_qns", local_qns)
        object.__setattr__(self, "prefix_qns", prefix_qns)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "view_qns", view_qns)

    @property
    def nsites(self) -> int:
        return len(self.local_qns)

    @property
    def dims(self) -> tuple[int, ...]:
        return tuple(len(site) for site in self.local_qns)

    @property
    def view_dims(self) -> tuple[int, ...]:
        return tuple(len(site) for site in self.view_qns)

    @property
    def symmetry(self) -> ProductSymmetry:
        return _product_u1_symmetry(len(self.target))

    @property
    def prefix_legs(self) -> list[Leg]:
        sym = self.symmetry
        return [Leg(_sector_dims(labels), symmetry=sym) for labels in self.prefix_qns]

    @property
    def view_legs(self) -> list[Leg]:
        sym = self.symmetry
        return [Leg(_sector_dims(labels), symmetry=sym) for labels in self.view_qns]

    @property
    def target_leg(self) -> Leg:
        return Leg({Irrep(self.target): 1}, symmetry=self.symmetry)

    @classmethod
    def from_local_charges(
        cls,
        local_qns,
        *,
        target,
        left_boundary=None,
        prefix_qns=None,
        view_qns=None,
        view_dim=None,
    ) -> "XLayout":
        """Build a compact fixed-target XLETTA layout from site charges."""
        local_qns = [[_as_charge(qn) for qn in site] for site in local_qns]
        target = _as_charge(target)
        if left_boundary is None:
            left_boundary = tuple(0 for _ in target)
        left_boundary = _as_charge(left_boundary)

        if prefix_qns is None:
            prefix_qns = [[left_boundary]]
            current = {left_boundary}
            for site in range(len(local_qns) - 1):
                remaining_sums = _possible_sums(local_qns[site + 1 :], rank=len(target))
                next_charges = []
                for prefix in current:
                    for q_site in local_qns[site]:
                        charge = _add_charges(prefix, q_site)
                        if any(_add_charges(charge, suffix) == target for suffix in remaining_sums):
                            next_charges.append(charge)
                current = set(next_charges)
                prefix_qns.append(_unique_sorted_charges(current))
        else:
            prefix_qns = [[_as_charge(qn) for qn in bond] for bond in prefix_qns]

        if view_qns is None:
            if view_dim is None:
                view_qns = [list(site) for site in local_qns[1:]]
            else:
                if np.isscalar(view_dim):
                    view_dims = (int(view_dim),) * (len(local_qns) - 1)
                else:
                    view_dims = tuple(int(dim) for dim in view_dim)
                    if len(view_dims) != len(local_qns) - 1:
                        raise ValueError("view_dim must have one entry for each shared physical site.")
                view_qns = []
                for site_qns, dim in zip(local_qns[1:], view_dims):
                    if dim < len(site_qns):
                        raise ValueError("view_dim must be at least the physical local dimension.")
                    labels = list(site_qns)
                    while len(labels) < dim:
                        labels.append(site_qns[(len(labels) - len(site_qns)) % len(site_qns)])
                    view_qns.append(labels)
        return cls(local_qns=local_qns, prefix_qns=prefix_qns, target=target, view_qns=view_qns)

    @classmethod
    def from_letta_layout(cls, layout: Layout, *, view_qns=None, view_dim=None) -> "XLayout":
        """Build an XLETTA layout from an ordinary Abelian LETTA layout."""
        if not isinstance(layout, Layout):
            raise TypeError("from_letta_layout expects a pyqed.letta.Layout instance.")
        return cls.from_local_charges(
            layout.local_qns,
            target=layout.target,
            left_boundary=layout.bond_qns[0][0],
            view_qns=view_qns,
            view_dim=view_dim,
        )

    def tensor_shapes(self) -> list[tuple[int, ...]]:
        shapes = [(len(self.local_qns[0]), len(self.prefix_qns[1]), len(self.view_qns[0]))]
        for site in range(1, self.nsites - 1):
            shapes.append(
                (
                    len(self.prefix_qns[site]),
                    len(self.view_qns[site - 1]),
                    len(self.view_qns[site]),
                    len(self.prefix_qns[site + 1]),
                )
            )
        shapes.append((len(self.prefix_qns[-1]), len(self.view_qns[-1])))
        return shapes

    def w_shapes(self) -> list[tuple[int, ...]]:
        return [
            (len(view_leg), len(view_leg), len(local_site))
            for view_leg, local_site in zip(self.view_qns, self.local_qns[1:])
        ]

    def tensor_masks(self) -> list[np.ndarray]:
        masks = []
        left_boundary = self.prefix_qns[0][0]
        first = np.zeros(self.tensor_shapes()[0], dtype=bool)
        for s0, q_s0 in enumerate(self.local_qns[0]):
            needed = _add_charges(left_boundary, q_s0)
            for a1, q_a1 in enumerate(self.prefix_qns[1]):
                if q_a1 == needed:
                    first[s0, a1, :] = True
        masks.append(first)

        for site in range(1, self.nsites - 1):
            mask = np.zeros(self.tensor_shapes()[site], dtype=bool)
            for left, q_left in enumerate(self.prefix_qns[site]):
                for v, q_v in enumerate(self.view_qns[site - 1]):
                    needed = _add_charges(q_left, q_v)
                    for right, q_right in enumerate(self.prefix_qns[site + 1]):
                        if q_right == needed:
                            mask[left, v, :, right] = True
            masks.append(mask)

        last = np.zeros(self.tensor_shapes()[-1], dtype=bool)
        for left, q_left in enumerate(self.prefix_qns[-1]):
            for v, q_v in enumerate(self.view_qns[-1]):
                if _add_charges(q_left, q_v) == self.target:
                    last[left, v] = True
        masks.append(last)
        return masks

    def w_masks(self) -> list[np.ndarray]:
        masks = []
        for view_leg, local_site in zip(self.view_qns, self.local_qns[1:]):
            mask = np.zeros((len(view_leg), len(view_leg), len(local_site)), dtype=bool)
            for u, q_u in enumerate(view_leg):
                for v, q_v in enumerate(view_leg):
                    if q_u != q_v:
                        continue
                    for s, q_s in enumerate(local_site):
                        if q_v == q_s:
                            mask[u, v, s] = True
            masks.append(mask)
        return masks

    def masks(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return self.tensor_masks(), self.w_masks()

    def structural_support_sizes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        tensor_sizes = tuple(int(np.count_nonzero(mask)) for mask in self.tensor_masks())
        w_sizes = tuple(int(np.count_nonzero(mask)) for mask in self.w_masks())
        return tensor_sizes, w_sizes

    def tensor_operator_grid(self, tensor_index: int, tensor: np.ndarray) -> dict[tuple[int, ...], IrrepTensor]:
        """Return IrrepTensor blocks for one XLETTA ``A`` tensor.

        Spectator lifted-view legs are included in the dictionary key.  The
        IrrepTensor itself carries only the charge-changing prefix-sector map,
        so dense work is confined to legal sector blocks.
        """
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nsites:
            raise IndexError("tensor_index out of range.")
        tensor = np.asarray(tensor)
        expected = self.tensor_shapes()[tensor_index]
        if tensor.shape != expected:
            raise ValueError(f"tensor shape {tensor.shape} does not match Abelian XLETTA shape {expected}.")
        if tensor_index == 0:
            return self._first_tensor_operator_grid(tensor)
        if tensor_index == self.nsites - 1:
            return self._final_tensor_operator_grid(tensor)
        return self._internal_tensor_operator_grid(tensor_index, tensor)

    def w_operator_grid(self, shared_index: int, tensor: np.ndarray) -> dict[tuple[int, ...], IrrepTensor]:
        """Return scalar IrrepTensor blocks for one XLETTA lifted ``W`` tensor."""
        shared_index = int(shared_index)
        if shared_index < 0 or shared_index >= self.nsites - 1:
            raise IndexError("shared_index out of range.")
        tensor = np.asarray(tensor)
        expected = self.w_shapes()[shared_index]
        if tensor.shape != expected:
            raise ValueError(f"W shape {tensor.shape} does not match Abelian XLETTA shape {expected}.")
        view_leg = self.view_legs[shared_index]
        labels = self.view_qns[shared_index]
        positions = _sector_positions(labels)
        zero = tuple(0 for _ in self.target)
        out = {}
        for s, q_s in enumerate(self.local_qns[shared_index + 1]):
            blocks = {}
            for u, q_u in enumerate(labels):
                if q_u != q_s:
                    continue
                for v, q_v in enumerate(labels):
                    if q_v != q_s:
                        continue
                    irrep = Irrep(q_s)
                    block = blocks.setdefault(
                        (irrep, irrep),
                        np.zeros((view_leg.sector_dim(irrep), view_leg.sector_dim(irrep)), dtype=tensor.dtype),
                    )
                    block[positions[u], positions[v]] = tensor[u, v, s]
            out[(s,)] = IrrepTensor(view_leg, view_leg, OpIrrep(zero), blocks)
        return out

    def local_tensor_blocks(self, tensor_index: int) -> tuple[AbelianXLETTALocalBlock, ...]:
        tensor_index = int(tensor_index)
        mask = self.tensor_masks()[tensor_index]
        dummy = np.zeros(mask.shape)
        grid = self.tensor_operator_grid(tensor_index, dummy)
        blocks = []
        for physical, operator in grid.items():
            for (bra, ket), block in operator.blocks.items():
                coords = self._tensor_block_coords(tensor_index, physical, bra, ket)
                if coords.size == 0:
                    continue
                flat = np.ravel_multi_index(coords.T, mask.shape)
                blocks.append(
                    AbelianXLETTALocalBlock(
                        kind="tensor",
                        index=tensor_index,
                        physical=tuple(int(x) for x in physical),
                        bra=bra,
                        ket=ket,
                        shape=tuple(int(x) for x in block.shape),
                        flat_indices=np.asarray(flat, dtype=np.int64),
                        coords=np.asarray(coords, dtype=np.int64),
                    )
                )
        return tuple(blocks)

    def local_w_blocks(self, shared_index: int) -> tuple[AbelianXLETTALocalBlock, ...]:
        shared_index = int(shared_index)
        mask = self.w_masks()[shared_index]
        dummy = np.zeros(mask.shape)
        grid = self.w_operator_grid(shared_index, dummy)
        blocks = []
        labels = self.view_qns[shared_index]
        for physical, operator in grid.items():
            s = physical[0]
            for (bra, ket), block in operator.blocks.items():
                coords = []
                for u, q_u in enumerate(labels):
                    if Irrep(q_u) != bra:
                        continue
                    for v, q_v in enumerate(labels):
                        if Irrep(q_v) == ket and mask[u, v, s]:
                            coords.append((u, v, s))
                if not coords:
                    continue
                coords = np.asarray(coords, dtype=np.int64)
                flat = np.ravel_multi_index(coords.T, mask.shape)
                blocks.append(
                    AbelianXLETTALocalBlock(
                        kind="w",
                        index=shared_index,
                        physical=tuple(int(x) for x in physical),
                        bra=bra,
                        ket=ket,
                        shape=tuple(int(x) for x in block.shape),
                        flat_indices=np.asarray(flat, dtype=np.int64),
                        coords=coords,
                    )
                )
        return tuple(blocks)

    def local_variable_blocks(self, kind: str, index: int) -> tuple[AbelianXLETTALocalBlock, ...]:
        if kind == "tensor":
            return self.local_tensor_blocks(index)
        if kind == "w":
            return self.local_w_blocks(index)
        raise ValueError("kind must be 'tensor' or 'w'.")

    def _first_tensor_operator_grid(self, tensor: np.ndarray) -> dict[tuple[int, ...], IrrepTensor]:
        left_leg = self.prefix_legs[0]
        right_leg = self.prefix_legs[1]
        right_labels = self.prefix_qns[1]
        right_pos = _sector_positions(right_labels)
        out = {}
        for s0, q_s0 in enumerate(self.local_qns[0]):
            op = OpIrrep(q_s0)
            for u in range(len(self.view_qns[0])):
                blocks = {}
                for left, q_left in enumerate(self.prefix_qns[0]):
                    needed = _add_charges(q_left, q_s0)
                    for right, q_right in enumerate(right_labels):
                        if q_right != needed:
                            continue
                        bra = Irrep(q_right)
                        ket = Irrep(q_left)
                        block = blocks.setdefault(
                            (bra, ket),
                            np.zeros((right_leg.sector_dim(bra), left_leg.sector_dim(ket)), dtype=tensor.dtype),
                        )
                        block[right_pos[right], 0] = tensor[s0, right, u]
                out[(s0, u)] = IrrepTensor(right_leg, left_leg, op, blocks)
        return out

    def _internal_tensor_operator_grid(self, tensor_index: int, tensor: np.ndarray) -> dict[tuple[int, ...], IrrepTensor]:
        left_labels = self.prefix_qns[tensor_index]
        right_labels = self.prefix_qns[tensor_index + 1]
        left_leg = self.prefix_legs[tensor_index]
        right_leg = self.prefix_legs[tensor_index + 1]
        left_pos = _sector_positions(left_labels)
        right_pos = _sector_positions(right_labels)
        out = {}
        for v, q_v in enumerate(self.view_qns[tensor_index - 1]):
            op = OpIrrep(q_v)
            for u in range(len(self.view_qns[tensor_index])):
                blocks = {}
                for left, q_left in enumerate(left_labels):
                    needed = _add_charges(q_left, q_v)
                    for right, q_right in enumerate(right_labels):
                        if q_right != needed:
                            continue
                        bra = Irrep(q_right)
                        ket = Irrep(q_left)
                        block = blocks.setdefault(
                            (bra, ket),
                            np.zeros((right_leg.sector_dim(bra), left_leg.sector_dim(ket)), dtype=tensor.dtype),
                        )
                        block[right_pos[right], left_pos[left]] = tensor[left, v, u, right]
                out[(v, u)] = IrrepTensor(right_leg, left_leg, op, blocks)
        return out

    def _final_tensor_operator_grid(self, tensor: np.ndarray) -> dict[tuple[int, ...], IrrepTensor]:
        left_labels = self.prefix_qns[-1]
        left_leg = self.prefix_legs[-1]
        right_leg = self.target_leg
        left_pos = _sector_positions(left_labels)
        out = {}
        for v, q_v in enumerate(self.view_qns[-1]):
            blocks = {}
            for left, q_left in enumerate(left_labels):
                if _add_charges(q_left, q_v) != self.target:
                    continue
                bra = Irrep(self.target)
                ket = Irrep(q_left)
                block = blocks.setdefault(
                    (bra, ket),
                    np.zeros((1, left_leg.sector_dim(ket)), dtype=tensor.dtype),
                )
                block[0, left_pos[left]] = tensor[left, v]
            out[(v,)] = IrrepTensor(right_leg, left_leg, OpIrrep(q_v), blocks)
        return out

    def _tensor_block_coords(self, tensor_index: int, physical: tuple[int, ...], bra: Irrep, ket: Irrep) -> np.ndarray:
        mask = self.tensor_masks()[tensor_index]
        coords = []
        if tensor_index == 0:
            s0, u = physical
            for right, q_right in enumerate(self.prefix_qns[1]):
                if Irrep(q_right) == bra and Irrep(self.prefix_qns[0][0]) == ket and mask[s0, right, u]:
                    coords.append((s0, right, u))
        elif tensor_index == self.nsites - 1:
            (v,) = physical
            for left, q_left in enumerate(self.prefix_qns[-1]):
                if Irrep(self.target) == bra and Irrep(q_left) == ket and mask[left, v]:
                    coords.append((left, v))
        else:
            v, u = physical
            for left, q_left in enumerate(self.prefix_qns[tensor_index]):
                if Irrep(q_left) != ket:
                    continue
                for right, q_right in enumerate(self.prefix_qns[tensor_index + 1]):
                    if Irrep(q_right) == bra and mask[left, v, u, right]:
                        coords.append((left, v, u, right))
        if not coords:
            return np.empty((0, mask.ndim), dtype=np.int64)
        return np.asarray(coords, dtype=np.int64)
