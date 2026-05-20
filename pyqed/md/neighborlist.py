"""Neighbor-list utilities for short-range MD interactions."""

import numpy as np


def minimum_image(vector, cell, pbc):
    """Return the minimum-image displacement vector."""
    pbc = np.asarray(pbc, dtype=bool)
    if not np.any(pbc):
        return np.asarray(vector, dtype=float)

    cell = np.asarray(cell, dtype=float)
    if np.linalg.matrix_rank(cell) < 3:
        raise ValueError("Periodic calculations require a full 3D cell.")

    scaled = np.linalg.solve(cell.T, np.asarray(vector, dtype=float))
    scaled[pbc] -= np.round(scaled[pbc])
    return scaled @ cell


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
        for i, j in self._all_pairs(len(positions)):
            rij = minimum_image(positions[i] - positions[j], cell, pbc)
            if np.dot(rij, rij) <= cutoff2:
                yield i, j

    def _cutoff_pairs_cell_list(self, positions, cell, pbc, lengths, cutoff):
        pbc = np.asarray(pbc, dtype=bool)
        cutoff2 = cutoff * cutoff
        scaled = positions.copy()
        scaled[:, pbc] = np.mod(scaled[:, pbc], lengths[pbc])

        bin_counts = np.maximum(np.floor(lengths / cutoff).astype(int), 1)
        bins = {}
        for atom_index, xyz in enumerate(scaled):
            key = tuple(np.minimum((xyz / lengths * bin_counts).astype(int), bin_counts - 1))
            bins.setdefault(key, []).append(atom_index)

        offsets = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3)
        seen = set()
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
                for i in atom_indices:
                    for j in bins.get(tuple(neighbor), []):
                        if i >= j:
                            continue
                        pair = (i, j)
                        if pair in seen:
                            continue
                        seen.add(pair)
                        if self.exclusions is not None and pair in self.exclusions:
                            continue
                        rij = minimum_image(positions[i] - positions[j], cell, pbc)
                        if np.dot(rij, rij) <= cutoff2:
                            yield pair


def candidate_pairs(positions, cell, pbc, cutoff=None, exclusions=None):
    """Yield candidate atom pairs, optionally using a cutoff neighbor list."""
    yield from NeighborList(cutoff=cutoff, exclusions=exclusions).build(positions, cell, pbc)
