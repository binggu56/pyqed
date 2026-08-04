"""Standard finite tree tensor-network topologies."""

from __future__ import annotations

from operator import index

from .tree import TTN


def balanced_ttn(
    nleaves,
    *,
    physical_dim=2,
    bond_dim=2,
    seed=None,
) -> TTN:
    """Return a balanced binary TTN with physical degrees at its leaves.

    Internal nodes have physical dimension one.  ``nleaves`` must be a power
    of two.  Edge dimensions respect the exact Hilbert-space capacity below
    each edge and are capped by ``bond_dim``.
    """
    try:
        nleaves = index(nleaves)
        physical_dim = index(physical_dim)
        bond_dim = index(bond_dim)
    except TypeError as error:
        raise ValueError("nleaves, physical_dim, and bond_dim must be integers.") from error
    if nleaves < 2 or nleaves & (nleaves - 1):
        raise ValueError("nleaves must be a power of two greater than one.")
    if physical_dim < 1 or bond_dim < 1:
        raise ValueError("physical_dim and bond_dim must be positive.")

    parents = [None] * nleaves
    dims = [physical_dim] * nleaves
    subtree_leaves = [1] * nleaves
    level = list(range(nleaves))
    while len(level) > 1:
        next_level = []
        for left, right in zip(level[::2], level[1::2]):
            parent = len(parents)
            parents.extend([None])
            dims.extend([1])
            subtree_leaves.extend([subtree_leaves[left] + subtree_leaves[right]])
            parents[left] = parent
            parents[right] = parent
            next_level.append(parent)
        level = next_level

    root = level[0]
    edge_dims = {
        child: min(bond_dim, physical_dim ** subtree_leaves[child])
        for child, parent in enumerate(parents)
        if parent is not None
    }
    return TTN(dims, parents, bond_dim=edge_dims, seed=seed, root=root)
