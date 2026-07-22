"""Neighbor-list utilities for short-range MD interactions."""

import numpy as np
try:
    from scipy.spatial import cKDTree as _cKDTree
except Exception:  # pragma: no cover - SciPy is available in the normal test env.
    _cKDTree = None


def minimum_image(vector, cell, pbc):
    """Return the minimum-image displacement vector."""
    pbc = np.asarray(pbc, dtype=bool)
    if not np.any(pbc):
        return np.asarray(vector, dtype=float)

    cell = np.asarray(cell, dtype=float)
    lengths = orthorhombic_lengths(cell)
    if lengths is not None:
        return minimum_image_orthorhombic(vector, lengths, pbc)

    if np.linalg.matrix_rank(cell) < 3:
        raise ValueError("Periodic calculations require a full 3D cell.")

    scaled = np.linalg.solve(cell.T, np.asarray(vector, dtype=float))
    scaled[pbc] -= np.round(scaled[pbc])
    return scaled @ cell


def minimum_image_orthorhombic(vector, lengths, pbc):
    """Return the minimum-image displacement for an orthorhombic cell."""
    displacement = np.asarray(vector, dtype=float).copy()
    pbc = np.asarray(pbc, dtype=bool)
    lengths = np.asarray(lengths, dtype=float)
    displacement[pbc] -= lengths[pbc] * np.round(displacement[pbc] / lengths[pbc])
    return displacement


def orthorhombic_lengths(cell):
    """Return cell lengths for an orthorhombic cell, otherwise ``None``."""
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        return None
    off_diagonal = cell - np.diag(np.diag(cell))
    if np.max(np.abs(off_diagonal)) > 1e-12:
        return None
    lengths = np.diag(cell)
    if np.any(lengths <= 0.0):
        return None
    return lengths


class NeighborList:
    """Build pair lists for cutoff-based nonbonded interactions.

    The implementation uses a cell list for orthorhombic boxes and falls back
    to an all-pairs scan for nonperiodic or non-orthorhombic inputs.  Pair
    distances are checked with the same minimum-image convention used by the
    calculators.
    """

    def __init__(self, cutoff=None, skin=0.0, exclusions=None):
        self.cutoff = None if cutoff is None else float(cutoff)
        self.skin = float(skin)
        if self.cutoff is not None and self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive.")
        if self.skin < 0.0:
            raise ValueError("skin must be non-negative.")
        self.exclusions = None if exclusions is None else {tuple(sorted(pair)) for pair in exclusions}
        self._pairs = []

    def build(self, positions, cell=None, pbc=None):
        positions = np.asarray(positions, dtype=float)
        cell = np.zeros((3, 3)) if cell is None else np.asarray(cell, dtype=float)
        pbc = np.zeros(3, dtype=bool) if pbc is None else np.asarray(pbc, dtype=bool)

        if self.cutoff is None:
            self._pairs = list(self._all_pairs(len(positions)))
            return self

        search_cutoff = self.cutoff + self.skin
        lengths = orthorhombic_lengths(cell)
        if lengths is None:
            self._pairs = list(self._cutoff_pairs_bruteforce(positions, cell, pbc, search_cutoff))
        else:
            self._pairs = list(self._cutoff_pairs_cell_list(positions, cell, pbc, lengths, search_cutoff))
        return self

    @property
    def pairs(self):
        return list(self._pairs)

    def __iter__(self):
        return iter(self._pairs)

    def __len__(self):
        return len(self._pairs)

    def _all_pairs(self, natoms):
        for i in range(natoms - 1):
            for j in range(i + 1, natoms):
                pair = (i, j)
                if self.exclusions is None or pair not in self.exclusions:
                    yield pair

    def _cutoff_pairs_bruteforce(self, positions, cell, pbc, cutoff):
        cutoff2 = cutoff * cutoff
        for i, j, _rij in _cutoff_pair_displacements_bruteforce(
            positions, cell, pbc, cutoff, self.exclusions
        ):
            yield i, j

    def _cutoff_pairs_cell_list(self, positions, cell, pbc, lengths, cutoff):
        for i, j, _rij in _cutoff_pair_displacements_cell_list(
            positions, pbc, lengths, cutoff, self.exclusions
        ):
            yield i, j


def candidate_pairs(positions, cell, pbc, cutoff=None, exclusions=None):
    """Yield candidate atom pairs, optionally using a cutoff neighbor list."""
    yield from NeighborList(cutoff=cutoff, exclusions=exclusions).build(positions, cell, pbc)


def candidate_pair_displacements(positions, cell, pbc, cutoff=None, exclusions=None):
    """Yield ``(i, j, rij)`` pairs with minimum-image displacements."""
    pair_i, pair_j, displacements = candidate_pair_displacement_arrays(
        positions,
        cell,
        pbc,
        cutoff,
        exclusions,
    )
    yield from zip(pair_i, pair_j, displacements)


def candidate_pair_displacement_arrays(positions, cell, pbc, cutoff=None, exclusions=None):
    """Return candidate pair indices and minimum-image displacements as arrays."""
    positions = np.asarray(positions, dtype=float)
    cell = np.zeros((3, 3)) if cell is None else np.asarray(cell, dtype=float)
    pbc = np.zeros(3, dtype=bool) if pbc is None else np.asarray(pbc, dtype=bool)
    exclusions = None if exclusions is None else {tuple(sorted(pair)) for pair in exclusions}

    if cutoff is None:
        pair_i, pair_j = np.triu_indices(len(positions), k=1)
        if exclusions is not None:
            mask = _nonexcluded_pair_mask(pair_i, pair_j, exclusions, len(positions))
            pair_i = pair_i[mask]
            pair_j = pair_j[mask]
        displacements = positions[pair_i] - positions[pair_j]
        lengths = orthorhombic_lengths(cell)
        if lengths is not None:
            axes = np.nonzero(pbc)[0]
            if len(axes) > 0:
                displacements[:, axes] -= lengths[axes] * np.round(
                    displacements[:, axes] / lengths[axes]
                )
        else:
            displacements = np.asarray(
                [minimum_image(vector, cell, pbc) for vector in displacements],
                dtype=float,
            )
        return pair_i, pair_j, displacements

    lengths = orthorhombic_lengths(cell)
    if lengths is None:
        pairs = list(
            _cutoff_pair_displacements_bruteforce(
                positions, cell, pbc, float(cutoff), exclusions
            )
        )
        return _tuple_pairs_to_arrays(pairs)
    return _cutoff_pair_displacements_cell_list_arrays(
        positions, pbc, lengths, float(cutoff), exclusions
    )


def _cutoff_pair_displacements_bruteforce(positions, cell, pbc, cutoff, exclusions):
    cutoff2 = cutoff * cutoff
    for i in range(len(positions) - 1):
        for j in range(i + 1, len(positions)):
            pair = (i, j)
            if exclusions is not None and pair in exclusions:
                continue
            rij = minimum_image(positions[i] - positions[j], cell, pbc)
            if np.dot(rij, rij) <= cutoff2:
                yield i, j, rij


def _cutoff_pair_displacements_cell_list(positions, pbc, lengths, cutoff, exclusions):
    pair_i, pair_j, displacements = _cutoff_pair_displacements_cell_list_arrays(
        positions,
        pbc,
        lengths,
        cutoff,
        exclusions,
    )
    yield from zip(pair_i, pair_j, displacements)


def _cutoff_pair_displacements_cell_list_arrays(positions, pbc, lengths, cutoff, exclusions):
    pbc = np.asarray(pbc, dtype=bool)
    if _cKDTree is not None and np.all(pbc):
        return _cutoff_pair_displacements_periodic_kdtree_arrays(
            positions,
            lengths,
            cutoff,
            exclusions,
        )
    cutoff2 = cutoff * cutoff
    scaled = positions.copy()
    scaled[:, pbc] = np.mod(scaled[:, pbc], lengths[pbc])
    periodic_axes = np.nonzero(pbc)[0]
    all_periodic = bool(np.all(pbc))

    bin_counts = np.maximum(np.floor(lengths / cutoff).astype(int), 1)
    bins = {}
    for atom_index, xyz in enumerate(scaled):
        key = tuple(np.minimum((xyz / lengths * bin_counts).astype(int), bin_counts - 1))
        bins.setdefault(key, []).append(atom_index)
    bins = {key: np.asarray(atom_indices, dtype=int) for key, atom_indices in bins.items()}

    offsets = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3)
    seen_cell_pairs = set()
    pair_i_chunks = []
    pair_j_chunks = []
    displacement_chunks = []
    for key, atom_indices in bins.items():
        key_array = np.array(key)
        for offset in offsets:
            neighbor = key_array + offset
            valid = np.ones(3, dtype=bool)
            for axis in range(3):
                if pbc[axis]:
                    neighbor[axis] %= bin_counts[axis]
                elif neighbor[axis] < 0 or neighbor[axis] >= bin_counts[axis]:
                    valid[axis] = False
            if not np.all(valid):
                continue
            neighbor_key = tuple(neighbor)
            neighbor_indices = bins.get(neighbor_key)
            if neighbor_indices is None:
                continue
            cell_pair = tuple(sorted((key, neighbor_key)))
            if cell_pair in seen_cell_pairs:
                continue
            seen_cell_pairs.add(cell_pair)
            if key == neighbor_key:
                if len(atom_indices) < 2:
                    continue
                first, second = np.triu_indices(len(atom_indices), k=1)
                pair_i = atom_indices[first]
                pair_j = atom_indices[second]
            else:
                pair_i = np.repeat(atom_indices, len(neighbor_indices))
                pair_j = np.tile(neighbor_indices, len(atom_indices))
                lower = np.minimum(pair_i, pair_j)
                pair_j = np.maximum(pair_i, pair_j)
                pair_i = lower
            if exclusions is not None:
                active = _nonexcluded_pair_mask(pair_i, pair_j, exclusions, len(positions))
                if not np.any(active):
                    continue
                pair_i = pair_i[active]
                pair_j = pair_j[active]
            rij = positions[pair_i] - positions[pair_j]
            if all_periodic:
                rij -= lengths * np.round(rij / lengths)
            elif len(periodic_axes) > 0:
                axes = periodic_axes
                rij[:, axes] -= lengths[axes] * np.round(rij[:, axes] / lengths[axes])
            active = np.einsum("ij,ij->i", rij, rij) <= cutoff2
            if not np.any(active):
                continue
            pair_i_chunks.append(pair_i[active])
            pair_j_chunks.append(pair_j[active])
            displacement_chunks.append(rij[active])
    if not pair_i_chunks:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.zeros((0, 3), dtype=float),
        )
    return (
        np.concatenate(pair_i_chunks),
        np.concatenate(pair_j_chunks),
        np.vstack(displacement_chunks),
    )


def _cutoff_pair_displacements_periodic_kdtree_arrays(positions, lengths, cutoff, exclusions):
    positions = np.asarray(positions, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    wrapped = np.mod(positions, lengths)
    pairs = _cKDTree(wrapped, boxsize=lengths).query_pairs(float(cutoff), output_type="ndarray")
    if pairs.size == 0:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.zeros((0, 3), dtype=float),
        )
    pair_i = pairs[:, 0].astype(int, copy=False)
    pair_j = pairs[:, 1].astype(int, copy=False)
    if exclusions is not None:
        active = _nonexcluded_pair_mask(pair_i, pair_j, exclusions, len(positions))
        if not np.any(active):
            return (
                np.asarray([], dtype=int),
                np.asarray([], dtype=int),
                np.zeros((0, 3), dtype=float),
            )
        pair_i = pair_i[active]
        pair_j = pair_j[active]
    displacements = positions[pair_i] - positions[pair_j]
    displacements -= lengths * np.round(displacements / lengths)
    return pair_i, pair_j, displacements


def _tuple_pairs_to_arrays(pairs):
    if not pairs:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.zeros((0, 3), dtype=float),
        )
    pair_i = np.asarray([pair[0] for pair in pairs], dtype=int)
    pair_j = np.asarray([pair[1] for pair in pairs], dtype=int)
    displacements = np.asarray([pair[2] for pair in pairs], dtype=float)
    return pair_i, pair_j, displacements


def _nonexcluded_pair_mask(pair_i, pair_j, exclusions, natoms):
    if not exclusions:
        return np.ones(len(pair_i), dtype=bool)
    excluded_keys = np.fromiter(
        (_pair_key(i, j, natoms) for i, j in exclusions),
        dtype=np.int64,
        count=len(exclusions),
    )
    pair_keys = _pair_keys(pair_i, pair_j, natoms)
    return ~np.isin(pair_keys, excluded_keys)


def _pair_keys(pair_i, pair_j, natoms):
    lower = np.minimum(pair_i, pair_j).astype(np.int64, copy=False)
    upper = np.maximum(pair_i, pair_j).astype(np.int64, copy=False)
    return lower * int(natoms) + upper


def _pair_key(i, j, natoms):
    i = int(i)
    j = int(j)
    return min(i, j) * int(natoms) + max(i, j)
