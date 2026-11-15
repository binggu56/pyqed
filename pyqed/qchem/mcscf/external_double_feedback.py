"""Matrix-free Schur action for the inactive-to-virtual doubles class."""

import numpy as np
from numba import njit


def disjoint_hole_pair_rounds(nholes):
    """Round-robin matchings where no two pairs write the same parent hole row."""
    indices = np.arange(nholes)
    yield np.column_stack((indices, indices))
    # The last slot for an odd number of holes is an idle participant.
    order = list(range(nholes + nholes % 2))
    for _ in range(len(order) - 1):
        pairs = [
            tuple(sorted((i, j)))
            for i, j in zip(order[: len(order) // 2], reversed(order))
            if i < nholes and j < nholes
        ]
        yield np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        order[1:] = [order[-1], *order[1:-1]]


@njit(inline="always")
def _inverse_denominator(
    denominator,
    eta_squared,
    screen_tol,
    first,
    end,
    mode,
    scales,
    edges,
    block_offsets,
    blocks,
):
    if eta_squared:
        if abs(denominator) < screen_tol:
            denominator = np.copysign(screen_tol, denominator)
        return denominator / (denominator * denominator + eta_squared)
    if abs(denominator) >= screen_tol:
        return 1.0 / denominator
    for edge in range(first, end):
        if scales[edge] != 0.0:
            parent_rank = edges[edge, 3]
            block = block_offsets[edge] + mode * parent_rank
            for k in range(parent_rank):
                if blocks[block + k] != 0.0:
                    raise ZeroDivisionError(
                        "Singular coupled external-doubles block; use an imaginary shift."
                    )
    return 0.0


@njit(cache=True, nogil=True)
def apply_external_double_feedback(
    vector,
    result,
    hole_pairs,
    hole_energies,
    virtual_energies,
    cross_fock,
    pattern_offsets,
    ranks,
    energies,
    edges,
    blocks,
    eta_squared,
    screen_tol,
):
    """Accumulate -C D^-1 C.T @ vector using implicit orbital-pair indices."""
    nh, nv = len(hole_energies), len(virtual_energies)
    hole_rows = np.empty(len(edges), dtype=np.int64)
    hole_blocks = np.empty(len(edges), dtype=np.int64)
    dropped_holes = np.empty(len(edges), dtype=np.int64)
    row_offsets = np.empty(len(edges), dtype=np.int64)
    block_offsets = np.empty(len(edges), dtype=np.int64)
    scales = np.empty(len(edges))
    for i, j in hole_pairs:
        h_pair = i if i == j else i * (2 * nh - i - 1) // 2 + j - i - 1
        hole_energy = hole_energies[i] + hole_energies[j]
        hole_pattern = 0 if i == j else 2
        for pattern in range(hole_pattern, hole_pattern + 2):
            for edge in range(pattern_offsets[pattern], pattern_offsets[pattern + 1]):
                drop_h, drop_v = edges[edge, 0], edges[edge, 1]
                hole = i if drop_h == 0 else j
                h_row = h_pair if drop_h < 0 else (j if drop_h == 0 else i)
                parent_rank = edges[edge, 3]
                hole_rows[edge] = edges[edge, 2] + h_row * edges[edge, 4] * parent_rank
                hole_blocks[edge] = edges[edge, 5]
                if drop_v < 0:
                    hole_blocks[edge] += hole * ranks[pattern] * parent_rank
                dropped_holes[edge] = hole
        for a in range(nv):
            start_v = a * (2 * nv - a - 1) // 2 - a - 1
            pair_energy = hole_energy - virtual_energies[a]
            for b in range(a, nv):
                pattern = hole_pattern + (a != b)
                first, end = pattern_offsets[pattern], pattern_offsets[pattern + 1]
                if first == end:
                    continue
                rank = ranks[pattern]
                v_pair = a if a == b else start_v + b
                # At most two contracted modes for the two pairings of
                # the inactive and virtual orbitals.
                applied0, applied1 = 0.0, 0.0
                for edge in range(first, end):
                    drop_h, drop_v = edges[edge, 0], edges[edge, 1]
                    parent_rank, block = edges[edge, 3], hole_blocks[edge]
                    v_row = v_pair if drop_v < 0 else (b if drop_v == 0 else a)
                    virtual = a if drop_v == 0 else b
                    scale = 1.0
                    if drop_h < 0:
                        block += virtual * rank * parent_rank
                    elif drop_v >= 0:
                        scale = cross_fock[virtual, dropped_holes[edge]]
                    row = hole_rows[edge] + v_row * parent_rank
                    row_offsets[edge] = row
                    block_offsets[edge] = block
                    scales[edge] = scale
                    if scale == 0.0:
                        continue
                    value0, value1 = 0.0, 0.0
                    if rank == 2:
                        for k in range(parent_rank):
                            value = vector[row + k]
                            value0 += blocks[block + k] * value
                            value1 += blocks[block + parent_rank + k] * value
                    else:
                        for k in range(parent_rank):
                            value0 += blocks[block + k] * vector[row + k]
                    applied0 += scale * value0
                    applied1 += scale * value1
                shift = pair_energy - virtual_energies[b]
                amplitude0 = -applied0 * _inverse_denominator(
                    energies[pattern, 0] + shift,
                    eta_squared,
                    screen_tol,
                    first,
                    end,
                    0,
                    scales,
                    edges,
                    block_offsets,
                    blocks,
                )
                amplitude1 = 0.0
                if rank == 2:
                    amplitude1 = -applied1 * _inverse_denominator(
                        energies[pattern, 1] + shift,
                        eta_squared,
                        screen_tol,
                        first,
                        end,
                        1,
                        scales,
                        edges,
                        block_offsets,
                        blocks,
                    )
                for edge in range(first, end):
                    scale = scales[edge]
                    if scale == 0.0:
                        continue
                    parent_rank = edges[edge, 3]
                    row, block = row_offsets[edge], block_offsets[edge]
                    value0 = scale * amplitude0
                    if rank == 2:
                        value1 = scale * amplitude1
                        for k in range(parent_rank):
                            result[row + k] += (
                                value0 * blocks[block + k]
                                + value1 * blocks[block + parent_rank + k]
                            )
                    else:
                        for k in range(parent_rank):
                            result[row + k] += value0 * blocks[block + k]
