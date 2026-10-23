"""Integer closed-shell occupations, independent of orbital ordering."""
import numpy as np


def rhf_occupations(energies, nelec, multiplet_ids=None):
    """Minimize the orbital-energy sum with zero/two occupations.

    With partner IDs, a dynamic program selects complete multiplets at fixed
    electron count. This is constrained integer filling, not fractional or
    thermal filling, and does not establish determinant stability.
    """
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 1 or not np.all(np.isfinite(energies)):
        raise ValueError('Orbital energies must be a finite vector.')
    if int(nelec) != nelec or nelec % 2 or not 0 <= nelec <= 2 * len(energies):
        raise ValueError('RHF requires an even electron count within orbital capacity.')
    nocc = int(nelec) // 2
    occ = np.zeros(len(energies))
    if multiplet_ids is None:
        occ[np.argsort(energies, kind='stable')[:nocc]] = 2
        return occ
    ids = np.asarray(multiplet_ids)
    if ids.shape != energies.shape:
        raise ValueError('Multiplet IDs must match orbital energies.')
    # Each entry stores the minimum cost and chosen IDs at this capacity.
    states = {0: (0.0, ())}
    for label in dict.fromkeys(ids.tolist()):
        indices = np.flatnonzero(ids == label)
        size, cost = len(indices), float(2 * energies[indices].sum())
        updated = dict(states)
        for count, (value, chosen) in states.items():
            target = count + size
            if target <= nocc and (target not in updated or value + cost < updated[target][0]):
                updated[target] = (value + cost, chosen + (label,))
        states = updated
    if nocc not in states:
        raise ValueError('Electron count cannot fill complete symmetry multiplets.')
    occ[np.isin(ids, states[nocc][1])] = 2
    return occ
