"""Exact physical blocking utilities for plaquette LETTA trials.

The blocking performed here is only a change of physical basis ordering.  It
does not introduce ties inside a block and it does not discard Hamiltonian
terms.  A block can therefore be represented by one LETTA/MPS tensor with a
product physical dimension while retaining a view with one axis per original
physical site.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pyqed.tn import Hamiltonian, LocalTerm


def _validated_blocks(dims, blocks) -> tuple[tuple[int, ...], ...]:
    dims = tuple(int(dim) for dim in dims)
    if not dims or any(dim < 1 for dim in dims):
        raise ValueError("dims must contain positive dimensions.")
    normalized = tuple(tuple(int(site) for site in block) for block in blocks)
    if not normalized or any(not block for block in normalized):
        raise ValueError("blocks must contain nonempty site groups.")
    sites = tuple(site for block in normalized for site in block)
    if sorted(sites) != list(range(len(dims))):
        raise ValueError("blocks must partition all physical sites exactly once.")
    return normalized


def square_plaquette_blocks(
    nrows: int,
    ncols: int,
    *,
    block_rows: int = 2,
    block_cols: int = 2,
) -> tuple[tuple[int, ...], ...]:
    """Return rectangular blocks in a row-wise snake block order.

    Physical sites are assumed to use the same row-wise snake ordering as the
    square-lattice LETTA examples.  Within a block, sites use geometric
    row-major order, making the unfused physical axes independent of the
    direction of the global snake.
    """
    nrows = int(nrows)
    ncols = int(ncols)
    block_rows = int(block_rows)
    block_cols = int(block_cols)
    if min(nrows, ncols, block_rows, block_cols) < 1:
        raise ValueError("lattice and block dimensions must be positive.")
    if nrows % block_rows or ncols % block_cols:
        raise ValueError("the block shape must tile the lattice exactly.")

    coordinates = []
    for row in range(nrows):
        columns = range(ncols) if row % 2 == 0 else range(ncols - 1, -1, -1)
        coordinates.extend((row, column) for column in columns)
    site_at = {coordinate: site for site, coordinate in enumerate(coordinates)}

    blocks = []
    nblock_rows = nrows // block_rows
    nblock_cols = ncols // block_cols
    for block_row in range(nblock_rows):
        block_columns = (
            range(nblock_cols)
            if block_row % 2 == 0
            else range(nblock_cols - 1, -1, -1)
        )
        for block_col in block_columns:
            sites = tuple(
                site_at[
                    (
                        block_row * block_rows + local_row,
                        block_col * block_cols + local_col,
                    )
                ]
                for local_row in range(block_rows)
                for local_col in range(block_cols)
            )
            blocks.append(sites)
    return tuple(blocks)


def plaquette_site_order(blocks) -> tuple[int, ...]:
    """Return the microscopic site order obtained by flattening ``blocks``."""
    normalized = tuple(tuple(int(site) for site in block) for block in blocks)
    if not normalized or any(not block for block in normalized):
        raise ValueError("blocks must contain nonempty site groups.")
    order = tuple(site for block in normalized for site in block)
    if sorted(order) != list(range(len(order))):
        raise ValueError("blocks must partition consecutive site indices.")
    return order


def interplaquette_edges(
    blocks,
    edges,
) -> tuple[tuple[int, int], ...]:
    """Keep microscopic edges whose endpoints belong to different blocks.

    The returned edges remain in the input site numbering.  They can therefore
    be filtered before the Hamiltonian and tie graph are remapped into a
    plaquette-contiguous ordering.
    """
    order = plaquette_site_order(blocks)
    block_of = {
        site: block_index
        for block_index, block in enumerate(blocks)
        for site in block
    }
    result = set()
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("edges must contain pairs of microscopic sites.")
        left, right = sorted((int(edge[0]), int(edge[1])))
        if left == right or left < 0 or right >= len(order):
            raise ValueError("edges must contain distinct valid site indices.")
        if block_of[left] != block_of[right]:
            result.add((left, right))
    return tuple(sorted(result))


def remap_site_edges(
    edges,
    site_order,
) -> tuple[tuple[int, int], ...]:
    """Map microscopic edges into positions of ``site_order``."""
    site_order = tuple(int(site) for site in site_order)
    if sorted(site_order) != list(range(len(site_order))):
        raise ValueError("site_order must be a permutation of all site indices.")
    position = {site: new_site for new_site, site in enumerate(site_order)}
    result = {
        tuple(sorted((position[int(left)], position[int(right)])))
        for left, right in edges
    }
    return tuple(sorted(result))


def blocked_dims(dims, blocks) -> tuple[int, ...]:
    """Return the product physical dimension of every block."""
    dims = tuple(int(dim) for dim in dims)
    blocks = _validated_blocks(dims, blocks)
    return tuple(int(np.prod([dims[site] for site in block])) for block in blocks)


def block_state_vector(vector, dims, blocks) -> np.ndarray:
    """Permute an original-site state into consecutive block-site ordering."""
    dims = tuple(int(dim) for dim in dims)
    blocks = _validated_blocks(dims, blocks)
    vector = np.asarray(vector)
    if vector.size != int(np.prod(dims)):
        raise ValueError("vector size is inconsistent with dims.")
    permutation = tuple(site for block in blocks for site in block)
    return vector.reshape(dims).transpose(permutation).reshape(-1)


def unblock_state_vector(vector, dims, blocks) -> np.ndarray:
    """Undo :func:`block_state_vector`."""
    dims = tuple(int(dim) for dim in dims)
    blocks = _validated_blocks(dims, blocks)
    permutation = tuple(site for block in blocks for site in block)
    blocked_shape = tuple(dims[site] for site in permutation)
    vector = np.asarray(vector)
    if vector.size != int(np.prod(dims)):
        raise ValueError("vector size is inconsistent with dims.")
    inverse = tuple(int(axis) for axis in np.argsort(permutation))
    return vector.reshape(blocked_shape).transpose(inverse).reshape(-1)


def _permuted_term_operator(term, dims, full_sites):
    position = {site: axis for axis, site in enumerate(full_sites)}
    ordered_sites = tuple(sorted(term.sites, key=position.__getitem__))
    permutation = tuple(term.sites.index(site) for site in ordered_sites)
    term_dims = tuple(dims[site] for site in term.sites)
    rank = len(term.sites)
    operator = np.asarray(term.operator).reshape(term_dims + term_dims)
    axes = permutation + tuple(rank + axis for axis in permutation)
    ordered_dims = tuple(term_dims[axis] for axis in permutation)
    return (
        ordered_sites,
        operator.transpose(axes).reshape(
            int(np.prod(ordered_dims)),
            int(np.prod(ordered_dims)),
        ),
    )


def _embedded_operator(operator, local_positions, full_dims):
    """Embed a small operator into a complete block-support product space."""
    local_positions = tuple(int(position) for position in local_positions)
    full_dims = tuple(int(dim) for dim in full_dims)
    local_dims = tuple(full_dims[position] for position in local_positions)
    operator = np.asarray(operator)
    dimension = int(np.prod(full_dims))
    result = np.zeros((dimension, dimension), dtype=operator.dtype)
    configurations = np.stack(
        np.unravel_index(np.arange(dimension), full_dims),
        axis=1,
    )
    rows, columns = np.nonzero(operator)
    if not rows.size:
        return result
    bra_values = np.stack(np.unravel_index(rows, local_dims), axis=1)
    ket_values = np.stack(np.unravel_index(columns, local_dims), axis=1)
    positions = np.asarray(local_positions, dtype=int)
    for row, column, bra, ket in zip(
        rows,
        columns,
        bra_values,
        ket_values,
    ):
        sources = np.flatnonzero(
            np.all(configurations[:, positions] == ket[None, :], axis=1)
        )
        targets = np.array(configurations[sources], copy=True)
        targets[:, positions] = bra[None, :]
        target_indices = np.ravel_multi_index(targets.T, full_dims)
        result[target_indices, sources] += operator[row, column]
    return result


def block_local_hamiltonian(
    hamiltonian: Hamiltonian,
    blocks,
) -> Hamiltonian:
    """Exactly rewrite a local Hamiltonian in product block bases.

    Terms spanning sites in one block become one-block operators.  Terms
    crossing blocks become operators on the complete product spaces of the
    touched blocks.  Multiple microscopic terms on the same block support are
    combined by :class:`Hamiltonian`.
    """
    if not isinstance(hamiltonian, Hamiltonian):
        raise TypeError("hamiltonian must be a Hamiltonian.")
    dims = hamiltonian.dims
    blocks = _validated_blocks(dims, blocks)
    block_dimensions = blocked_dims(dims, blocks)
    site_to_block = {
        site: block for block, sites in enumerate(blocks) for site in sites
    }
    blocked_terms = []
    for term in hamiltonian.terms:
        support_blocks = tuple(sorted({site_to_block[site] for site in term.sites}))
        full_sites = tuple(site for block in support_blocks for site in blocks[block])
        support_dims = tuple(dims[site] for site in full_sites)
        ordered_sites, operator = _permuted_term_operator(term, dims, full_sites)
        local_positions = tuple(full_sites.index(site) for site in ordered_sites)
        embedded = _embedded_operator(operator, local_positions, support_dims)
        blocked_terms.append(LocalTerm(support_blocks, embedded))
    result = Hamiltonian(
        block_dimensions,
        blocked_terms,
        constant=hamiltonian.constant,
    )
    result.block_physical_dims = tuple(
        tuple(dims[site] for site in block) for block in blocks
    )
    return result


def blocked_local_charges(
    local_qns: Sequence[Sequence[Sequence[int]]],
    blocks,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Add original-site charges in each product-basis block state."""
    normalized_qns = tuple(
        tuple(tuple(int(value) for value in charge) for charge in site)
        for site in local_qns
    )
    if not normalized_qns or any(not site for site in normalized_qns):
        raise ValueError("local_qns must contain nonempty charge lists.")
    rank = len(normalized_qns[0][0])
    if any(
        len(charge) != rank for site in normalized_qns for charge in site
    ):
        raise ValueError("all local charges must have the same rank.")
    dims = tuple(len(site) for site in normalized_qns)
    blocks = _validated_blocks(dims, blocks)

    result = []
    for block in blocks:
        block_dims = tuple(dims[site] for site in block)
        charges = []
        for configuration in np.ndindex(*block_dims):
            total = [0] * rank
            for site, state in zip(block, configuration):
                for axis, value in enumerate(normalized_qns[site][state]):
                    total[axis] += value
            charges.append(tuple(total))
        result.append(tuple(charges))
    return tuple(result)


def unfused_block_tensor(tensor, physical_dims) -> np.ndarray:
    """View ``(D_left, D_right, prod(d_i))`` as separate physical legs."""
    tensor = np.asarray(tensor)
    physical_dims = tuple(int(dim) for dim in physical_dims)
    if tensor.ndim != 3:
        raise ValueError("tensor must have shape (D_left, D_right, block_dim).")
    if tensor.shape[2] != int(np.prod(physical_dims)):
        raise ValueError("physical_dims do not multiply to the block dimension.")
    return tensor.reshape(tensor.shape[:2] + physical_dims)
