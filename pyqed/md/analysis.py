"""Trajectory-independent solvent analysis helpers."""

import numpy as np

from .neighborlist import minimum_image


def radial_distribution(atoms, group_a, group_b, r_max, bins=100):
    """Compute a simple pair-distance histogram/RDF-like curve."""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    group_a = np.asarray(group_a, dtype=int)
    group_b = np.asarray(group_b, dtype=int)
    distances = []
    for i in group_a:
        for j in group_b:
            if i == j:
                continue
            rij = minimum_image(positions[i] - positions[j], cell, pbc)
            r = float(np.linalg.norm(rij))
            if r <= r_max:
                distances.append(r)
    hist, edges = np.histogram(distances, bins=bins, range=(0.0, r_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def solvent_shell_count(atoms, solute_indices, solvent_indices, cutoff):
    """Count solvent atoms within ``cutoff`` of any solute atom."""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    count = 0
    for j in solvent_indices:
        for i in solute_indices:
            rij = minimum_image(positions[i] - positions[j], cell, pbc)
            if np.dot(rij, rij) <= cutoff * cutoff:
                count += 1
                break
    return count


def water_oxygen_indices(atoms, start=0):
    """Return oxygen indices for waters stored as O-H-H triplets."""
    return np.arange(int(start), len(atoms), 3, dtype=int)


def hydrogen_bonds(atoms, donor_hydrogen_pairs, acceptors, distance_cutoff, angle_cutoff_deg=30.0):
    """Return hydrogen bonds matching D-H...A distance and angle cutoffs."""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    hbonds = []
    cos_cutoff = np.cos(np.deg2rad(180.0 - angle_cutoff_deg))
    for donor, hydrogen in donor_hydrogen_pairs:
        dh = minimum_image(positions[hydrogen] - positions[donor], cell, pbc)
        dh_norm = np.linalg.norm(dh)
        if dh_norm == 0.0:
            continue
        hd_unit = -dh / dh_norm
        for acceptor in acceptors:
            if acceptor in (donor, hydrogen):
                continue
            ha = minimum_image(positions[acceptor] - positions[hydrogen], cell, pbc)
            ha_norm = np.linalg.norm(ha)
            if ha_norm == 0.0 or ha_norm > distance_cutoff:
                continue
            cos_angle = float(np.dot(hd_unit, ha / ha_norm))
            if cos_angle <= cos_cutoff:
                hbonds.append((donor, hydrogen, acceptor))
    return hbonds


def dipole_moment(atoms, charges=None, indices=None):
    """Return the point-charge dipole moment in atomic units."""
    if indices is None:
        indices = np.arange(len(atoms))
    indices = np.asarray(indices, dtype=int)
    if charges is None:
        charges = atoms.get_array("charges")
    charges = np.asarray(charges, dtype=float)[indices]
    positions = atoms.get_positions()[indices]
    return np.sum(charges[:, None] * positions, axis=0)


def autocorrelation(values):
    """Return a normalized vector autocorrelation for a time series."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n = len(values)
    corr = np.zeros(n)
    for lag in range(n):
        dots = np.sum(values[: n - lag] * values[lag:], axis=1)
        corr[lag] = np.mean(dots)
    if corr[0] != 0.0:
        corr /= corr[0]
    return corr
