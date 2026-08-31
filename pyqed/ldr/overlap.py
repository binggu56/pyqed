"""Electronic overlaps and nearest-neighbor transport on product grids."""

from __future__ import annotations

import itertools
import heapq

import numpy as np

from . import _kernels as kernels


def layout(shape):
    """Return product-grid indices, flat lookup, and forward edges."""

    shape = tuple(int(n) for n in shape)
    if not shape or any(n <= 0 for n in shape):
        raise ValueError("shape entries must be positive")

    indices = tuple(np.ndindex(shape))
    flat = {idx: i for i, idx in enumerate(indices)}
    edges = []
    for idx in indices:
        for axis in range(len(shape)):
            if idx[axis] + 1 >= shape[axis]:
                continue
            nxt = list(idx)
            nxt[axis] += 1
            nxt = tuple(nxt)
            edges.append((axis, idx, flat[idx], flat[nxt]))
    return np.asarray(indices, dtype=int), flat, tuple(edges)


def snake(shape):
    """Return a nearest-neighbor serpentine traversal of a product grid."""

    shape = tuple(int(n) for n in shape)
    if not shape or any(n <= 0 for n in shape):
        raise ValueError("shape entries must be positive")

    def build(axis):
        if axis == len(shape) - 1:
            return [(value,) for value in range(shape[axis])]

        tails = build(axis + 1)
        out = []
        for value in range(shape[axis]):
            traversal = tails if value % 2 == 0 else reversed(tails)
            out.extend((value,) + tail for tail in traversal)
        return out

    return tuple(build(0))


def as_block(value, nstates):
    """Validate one scalar or matrix electronic overlap block."""

    value = np.asarray(value, dtype=complex)
    if nstates is None:
        if value.ndim != 0:
            raise ValueError(f"scalar overlap required; got shape {value.shape}")
        return complex(value)

    nstates = int(nstates)
    if value.ndim == 0 and nstates == 1:
        value = value.reshape(1, 1)
    if value.shape != (nstates, nstates):
        raise ValueError(
            f"overlap shape {value.shape} != {(nstates, nstates)}"
        )
    return value


def unitary(value, *, threshold=1.0e-14):
    """Return the phase or unitary polar factor of an overlap block."""

    value = np.asarray(value, dtype=complex)
    if value.ndim == 0:
        magnitude = abs(complex(value))
        return 0.0j if magnitude < threshold else complex(value) / magnitude
    if value.ndim != 2:
        raise ValueError("overlap block must be scalar or matrix")
    return procrustes(value)[0]


def procrustes(value):
    r"""Return the unitary Procrustes factor, positive residual, and spectrum.

    For a square overlap matrix ``S``, the returned matrices satisfy
    ``S = U @ P``. ``U`` minimizes ``||S - U||_F`` and ``P`` is positive
    semidefinite. Leading batch dimensions are supported.
    """

    value = np.asarray(value, dtype=complex)
    if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
        raise ValueError("Procrustes alignment requires square matrices")
    left, singular_values, right = np.linalg.svd(value, full_matrices=False)
    rotation = left @ right
    positive = np.einsum(
        "...ai,...a,...aj->...ij",
        right.conj(),
        singular_values,
        right,
        optimize=True,
    )
    positive = 0.5 * (positive + positive.swapaxes(-1, -2).conj())
    return rotation, positive, singular_values


def nearest(shape, overlap, *, unitarize=False):
    """Evaluate ``overlap(index, neighbor)`` on all forward grid edges."""

    indices, _, edges = layout(shape)
    links = {}
    for axis, idx, _, neighbor_flat in edges:
        neighbor = tuple(indices[neighbor_flat])
        value = np.asarray(overlap(idx, neighbor), dtype=complex)
        links[(axis, idx)] = unitary(value) if unitarize else value
    return links


def follow(bra, ket, links, *, nstates=None, axes=None):
    """Compose links along one axis-ordered path between two grid points."""

    bra = tuple(int(i) for i in bra)
    ket = tuple(int(i) for i in ket)
    if len(bra) != len(ket):
        raise ValueError("bra and ket indices must have the same dimension")
    if axes is None:
        axes = range(len(bra))
    current = list(bra)
    value = 1.0 + 0.0j if nstates is None else np.eye(int(nstates), dtype=complex)

    for axis in axes:
        axis = int(axis)
        while current[axis] < ket[axis]:
            link = as_block(links[(axis, tuple(current))], nstates)
            value = value * link if nstates is None else value @ link
            current[axis] += 1
        while current[axis] > ket[axis]:
            current[axis] -= 1
            link = as_block(links[(axis, tuple(current))], nstates)
            link = link.conjugate() if nstates is None else link.conj().T
            value = value * link if nstates is None else value @ link

    if tuple(current) != ket:
        raise ValueError("axes do not connect bra to ket")
    return value


def between(bra, ket, links, *, nstates=None, average_paths=False):
    """Return linked overlap, optionally averaged over active-axis orderings."""

    bra = tuple(int(i) for i in bra)
    ket = tuple(int(i) for i in ket)
    active = tuple(i for i, (left, right) in enumerate(zip(bra, ket)) if left != right)
    if not average_paths or len(active) <= 1:
        return follow(bra, ket, links, nstates=nstates)

    paths = tuple(itertools.permutations(active))
    if nstates is None:
        return sum(follow(bra, ket, links, axes=path) for path in paths) / len(paths)
    value = np.zeros((int(nstates), int(nstates)), dtype=complex)
    for path in paths:
        value += follow(bra, ket, links, nstates=nstates, axes=path)
    return value / len(paths)


def phase_gauge(
    shape,
    links,
    *,
    state=0,
    anchor=None,
    threshold=1.0e-10,
    support=None,
):
    """Return coefficients that phase-align one adiabatic state to an anchor."""

    shape = tuple(int(n) for n in shape)
    state = np.asarray(state, dtype=int)
    if state.ndim == 0:
        scalar_state = int(state)
        state_labels = None
    elif state.shape == shape:
        scalar_state = None
        state_labels = state
        if np.any(state_labels < 0):
            raise ValueError("state labels must be nonnegative")
    else:
        raise ValueError(f"state-label shape {state.shape} != {shape}")
    if anchor is None:
        anchor = tuple(n // 2 for n in shape)
    anchor = tuple(int(i) for i in anchor)
    if len(anchor) != len(shape) or any(
        index < 0 or index >= size for index, size in zip(anchor, shape)
    ):
        raise ValueError("anchor is outside the product grid")

    if support is not None or state_labels is not None:
        if support is None:
            support = np.ones(shape, dtype=bool)
        support = np.asarray(support, dtype=bool)
        if support.shape != shape:
            raise ValueError(f"support shape {support.shape} != {shape}")
        if not support[anchor]:
            raise ValueError("anchor must lie inside the phase-gauge support")
        values = np.ones(shape, dtype=complex)
        visited = np.zeros(shape, dtype=bool)
        visited[anchor] = True
        frontier = [anchor]
        while frontier:
            index = frontier.pop()
            for axis, size in enumerate(shape):
                if index[axis] + 1 < size:
                    neighbor = list(index)
                    neighbor[axis] += 1
                    neighbor = tuple(neighbor)
                    key = (axis, index)
                    direction = 1
                    candidates = ((neighbor, key, direction),)
                else:
                    candidates = ()
                if index[axis] > 0:
                    neighbor = list(index)
                    neighbor[axis] -= 1
                    neighbor = tuple(neighbor)
                    candidates += ((neighbor, (axis, neighbor), -1),)
                for neighbor, key, direction in candidates:
                    if visited[neighbor] or not support[neighbor]:
                        continue
                    block = np.asarray(links[key], dtype=complex)
                    left = index if direction > 0 else neighbor
                    right = neighbor if direction > 0 else index
                    left_state = (
                        scalar_state
                        if state_labels is None
                        else int(state_labels[left])
                    )
                    right_state = (
                        scalar_state
                        if state_labels is None
                        else int(state_labels[right])
                    )
                    if (
                        block.ndim != 2
                        or left_state < 0
                        or right_state < 0
                        or left_state >= block.shape[0]
                        or right_state >= block.shape[1]
                    ):
                        raise ValueError(
                            "state is incompatible with the overlap-link blocks"
                        )
                    link = complex(block[left_state, right_state])
                    magnitude = abs(link)
                    if magnitude < threshold:
                        continue
                    phase = link / magnitude
                    values[neighbor] = values[index] * (
                        phase.conjugate() if direction > 0 else phase
                    )
                    visited[neighbor] = True
                    frontier.append(neighbor)
        missing = support & ~visited
        if np.any(missing):
            first = tuple(int(value) for value in np.argwhere(missing)[0])
            raise ValueError(
                f"phase-gauge support is disconnected at grid index {first}"
            )
        return values

    scalar_links = {}
    for key, block in links.items():
        block = np.asarray(block, dtype=complex)
        if (
            block.ndim != 2
            or scalar_state < 0
            or scalar_state >= min(block.shape)
        ):
            raise ValueError("state is incompatible with the overlap-link blocks")
        value = complex(block[scalar_state, scalar_state])
        magnitude = abs(value)
        if magnitude < threshold:
            raise ValueError(
                f"adiabatic overlap on link {key} is too small for phase transport"
            )
        scalar_links[key] = value / magnitude

    values = np.empty(shape, dtype=complex)
    for index in np.ndindex(shape):
        values[index] = follow(anchor, index, scalar_links).conjugate()
    return values


def dense(shape, links, *, nstates=None, average_paths=False):
    """Materialize the global linked overlap matrix or block tensor."""

    indices, _, _ = layout(shape)
    ngrid = len(indices)
    if nstates is None:
        result = np.empty((ngrid, ngrid), dtype=complex)
        for i, bra in enumerate(map(tuple, indices)):
            result[i, i] = 1.0
            for j in range(i + 1, ngrid):
                value = between(
                    bra,
                    tuple(indices[j]),
                    links,
                    average_paths=average_paths,
                )
                result[i, j] = value
                result[j, i] = value.conjugate()
        return result

    nstates = int(nstates)
    result = np.zeros((ngrid, nstates, ngrid, nstates), dtype=complex)
    if nstates is not None and not average_paths:
        fast = kernels.linked_overlap_dense(shape, links, nstates=nstates, average_paths=False)
        if fast is not None:
            return fast
    for i, bra in enumerate(map(tuple, indices)):
        result[i, :, i, :] = np.eye(nstates)
        for j in range(i + 1, ngrid):
            value = between(
                bra,
                tuple(indices[j]),
                links,
                nstates=nstates,
                average_paths=average_paths,
            )
            result[i, :, j, :] = value
            result[j, :, i, :] = value.conj().T
    return result


def full(objects, overlap, *, nstates):
    """Evaluate every pairwise overlap between electronic-frame objects."""

    flat = np.asarray(objects, dtype=object).reshape(-1)
    nstates = int(nstates)
    result = np.zeros((len(flat), nstates, len(flat), nstates), dtype=complex)
    identity = np.eye(nstates, dtype=complex)
    for i, left in enumerate(flat):
        result[i, :, i, :] = identity
        for j in range(i + 1, len(flat)):
            value = as_block(overlap(left, flat[j]), nstates)
            result[i, :, j, :] = value
            result[j, :, i, :] = value.conj().T
    return result


def from_frames(frames):
    """Return all pair overlaps for column-oriented local frame matrices."""

    frames = np.asarray(frames, dtype=complex)
    if frames.ndim < 3 or frames.shape[-2] < frames.shape[-1]:
        raise ValueError("frames must have shape (..., basis, states)")
    basis, nstates = frames.shape[-2:]
    flat = frames.reshape(-1, basis, nstates)
    return np.einsum(
        "gxa,hxb->gahb",
        flat.conj(),
        flat,
        optimize=True,
    )


def track_states(links, anchor, states):
    """Track anchor-selected channels through a chain of complete root overlaps.

    Each step uses a rectangular maximum-overlap assignment, so the returned
    channel order remains tied to ``states`` even when energy-ordered root
    indices exchange.
    """

    from scipy.optimize import linear_sum_assignment

    links = np.asarray(links, dtype=complex)
    if links.ndim != 3 or links.shape[1] != links.shape[2]:
        raise ValueError("links must have shape (npoints - 1, nroots, nroots)")
    npoints = len(links) + 1
    nroots = links.shape[1]
    anchor = int(anchor)
    states = np.asarray(states, dtype=int)
    if not 0 <= anchor < npoints:
        raise ValueError("anchor is outside the overlap chain")
    if (
        states.ndim != 1
        or not len(states)
        or len(np.unique(states)) != len(states)
        or np.any(states < 0)
        or np.any(states >= nroots)
    ):
        raise ValueError("states must be distinct root indices inside the overlap blocks")

    indices = np.full((npoints, len(states)), -1, dtype=int)
    indices[anchor] = states

    def assignment(score):
        channels, roots = linear_sum_assignment(-np.asarray(score, dtype=float))
        order = np.empty(len(states), dtype=int)
        order[channels] = roots
        return order

    for edge in range(anchor, npoints - 1):
        indices[edge + 1] = assignment(np.abs(links[edge][indices[edge], :]))
    for edge in range(anchor - 1, -1, -1):
        indices[edge] = assignment(
            np.abs(links[edge][:, indices[edge + 1]]).T
        )

    selected = np.asarray(
        [
            links[edge][np.ix_(indices[edge], indices[edge + 1])]
            for edge in range(npoints - 1)
        ]
    )
    return indices, selected


def positive_link_gauge(links, anchor):
    r"""Return gauges $G_i$ for which $G_i^\dagger S_iG_{i+1}$ is positive."""

    links = np.asarray(links, dtype=complex)
    if links.ndim != 3 or links.shape[1] != links.shape[2]:
        raise ValueError("links must have shape (npoints - 1, nstates, nstates)")
    npoints = len(links) + 1
    nstates = links.shape[1]
    anchor = int(anchor)
    if not 0 <= anchor < npoints:
        raise ValueError("anchor is outside the overlap chain")
    gauges = np.empty((npoints, nstates, nstates), dtype=complex)
    gauges[anchor] = np.eye(nstates)
    for edge in range(anchor, npoints - 1):
        rotation = procrustes(gauges[edge].conj().T @ links[edge])[0]
        gauges[edge + 1] = rotation.conj().T
    for edge in range(anchor - 1, -1, -1):
        rotation = procrustes(links[edge] @ gauges[edge + 1])[0]
        gauges[edge] = rotation
    aligned = np.asarray(
        [
            gauges[edge].conj().T @ links[edge] @ gauges[edge + 1]
            for edge in range(npoints - 1)
        ]
    )
    return gauges, aligned


def _checked_graph(points, pairs, links):
    points = tuple(dict.fromkeys(tuple(map(int, point)) for point in points))
    if not points:
        raise ValueError("overlap graph requires at least one point")
    point_ids = {point: index for index, point in enumerate(points)}
    pairs = tuple(
        (tuple(map(int, left)), tuple(map(int, right)))
        for left, right in pairs
    )
    links = np.asarray(links, dtype=complex)
    if links.ndim != 3 or links.shape[1] != links.shape[2]:
        raise ValueError("graph links must have shape (nedges, nroots, nroots)")
    if len(pairs) != len(links):
        raise ValueError("graph pairs and links must have the same length")
    if any(left not in point_ids or right not in point_ids for left, right in pairs):
        raise ValueError("graph-pair endpoints must belong to points")
    if any(left == right for left, right in pairs):
        raise ValueError("overlap graph cannot contain self edges")
    undirected = [tuple(sorted((point_ids[left], point_ids[right]))) for left, right in pairs]
    if len(set(undirected)) != len(undirected):
        raise ValueError("overlap graph cannot contain duplicate undirected edges")
    adjacency = [[] for _ in points]
    for edge, (left, right) in enumerate(pairs):
        left_id = point_ids[left]
        right_id = point_ids[right]
        adjacency[left_id].append((right_id, edge, False))
        adjacency[right_id].append((left_id, edge, True))
    reached = {0}
    frontier = [0]
    while frontier:
        left = frontier.pop()
        for right, _edge, _reverse in adjacency[left]:
            if right not in reached:
                reached.add(right)
                frontier.append(right)
    if len(reached) != len(points):
        raise ValueError("overlap graph must be connected")
    return points, point_ids, pairs, links, adjacency


def track_states_graph(
    points,
    pairs,
    links,
    anchor,
    states,
    *,
    fixed=None,
    sweeps=20,
):
    """Track anchor-selected adiabatic roots over a connected overlap graph.

    A maximum-reliability spanning tree supplies the initial assignment.  Local
    discrete refinement then maximizes the channel-resolved overlap over every
    graph edge, so the result uses one root ordering per vertex rather than one
    independently chosen ordering per path.
    """

    from scipy.optimize import linear_sum_assignment

    points, point_ids, pairs, links, adjacency = _checked_graph(points, pairs, links)
    nroots = links.shape[-1]
    anchor = tuple(map(int, anchor))
    if anchor not in point_ids:
        raise ValueError("tracking anchor must belong to points")
    anchor_id = point_ids[anchor]
    states = np.asarray(states, dtype=int)
    if (
        states.ndim != 1
        or not len(states)
        or len(np.unique(states)) != len(states)
        or np.any(states < 0)
        or np.any(states >= nroots)
    ):
        raise ValueError("states must be distinct root indices inside the overlap blocks")
    nstates = len(states)
    reliability = np.linalg.svd(links, compute_uv=False)[:, nstates - 1]
    labels = np.full((len(points), nstates), -1, dtype=int)
    fixed = {} if fixed is None else {
        tuple(map(int, point)): np.asarray(value, dtype=int)
        for point, value in fixed.items()
    }
    fixed[anchor] = states
    for point, value in fixed.items():
        if point not in point_ids:
            raise ValueError(f"fixed tracking point {point} does not belong to points")
        if (
            value.shape != (nstates,)
            or len(np.unique(value)) != nstates
            or np.any(value < 0)
            or np.any(value >= nroots)
        ):
            raise ValueError("fixed root assignments must match states and lie inside links")
        labels[point_ids[point]] = value
    fixed_ids = {point_ids[point] for point in fixed}
    assigned = set(fixed_ids)
    queue = []

    def add_edges(left):
        for right, edge, reverse in adjacency[left]:
            if right not in assigned:
                heapq.heappush(
                    queue,
                    (-float(reliability[edge]), left, right, edge, reverse),
                )

    for point in assigned:
        add_edges(point)
    while len(assigned) < len(points):
        if not queue:
            raise ValueError("overlap graph must be connected")
        _weight, left, right, edge, reverse = heapq.heappop(queue)
        if right in assigned or left not in assigned:
            continue
        block = links[edge].conj().T if reverse else links[edge]
        channels, roots = linear_sum_assignment(-np.abs(block[labels[left], :]))
        order = np.empty(nstates, dtype=int)
        order[channels] = roots
        labels[right] = order
        assigned.add(right)
        add_edges(right)

    candidates = np.asarray(tuple(itertools.permutations(range(nroots), nstates)))

    def local_score(point, candidate):
        score = 0.0
        for neighbor, edge, reverse in adjacency[point]:
            block = links[edge].conj().T if reverse else links[edge]
            values = block[candidate, labels[neighbor]]
            score += float(np.vdot(values, values).real)
        return score

    for _ in range(int(sweeps)):
        changed = False
        for point in range(len(points)):
            if point in fixed_ids:
                continue
            scores = np.asarray([local_score(point, candidate) for candidate in candidates])
            best = candidates[int(np.argmax(scores))]
            if not np.array_equal(best, labels[point]):
                labels[point] = best
                changed = True
        if not changed:
            break
    selected = np.asarray(
        [
            block[np.ix_(labels[point_ids[left]], labels[point_ids[right]])]
            for (left, right), block in zip(pairs, links)
        ]
    )
    return labels, selected


def synchronize_link_gauge(
    points,
    pairs,
    links,
    anchor,
    *,
    weights=None,
    sweeps=200,
    tolerance=1.0e-12,
):
    r"""Synchronize one anchored unitary gauge over an overlap graph.

    The gauges maximize the weighted agreement of all polar link rotations at
    once.  Unlike independent path transport, every vertex receives exactly one
    gauge even when the graph contains loops.
    """

    points, point_ids, pairs, links, adjacency = _checked_graph(points, pairs, links)
    nstates = links.shape[-1]
    anchor = tuple(map(int, anchor))
    if anchor not in point_ids:
        raise ValueError("gauge anchor must belong to points")
    anchor_id = point_ids[anchor]
    rotations = procrustes(links)[0]
    if weights is None:
        weights = np.ones(len(links))
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(links),) or np.any(weights <= 0.0):
            raise ValueError("gauge weights must be one positive value per edge")
    gauges = np.empty((len(points), nstates, nstates), dtype=complex)
    gauges[anchor_id] = np.eye(nstates)
    assigned = {anchor_id}
    queue = []

    def add_edges(left):
        for right, edge, reverse in adjacency[left]:
            if right not in assigned:
                heapq.heappush(
                    queue,
                    (-float(weights[edge]), left, right, edge, reverse),
                )

    add_edges(anchor_id)
    while len(assigned) < len(points):
        _weight, left, right, edge, reverse = heapq.heappop(queue)
        if right in assigned or left not in assigned:
            continue
        block = links[edge].conj().T if reverse else links[edge]
        rotation = procrustes(gauges[left].conj().T @ block)[0]
        gauges[right] = rotation.conj().T
        assigned.add(right)
        add_edges(right)

    for _ in range(int(sweeps)):
        maximum_change = 0.0
        for point in range(len(points)):
            if point == anchor_id:
                continue
            mean = np.zeros((nstates, nstates), dtype=complex)
            for neighbor, edge, reverse in adjacency[point]:
                rotation = rotations[edge].conj().T if reverse else rotations[edge]
                mean += weights[edge] * rotation @ gauges[neighbor]
            updated = procrustes(mean)[0]
            maximum_change = max(
                maximum_change, float(np.linalg.norm(updated - gauges[point]))
            )
            gauges[point] = updated
        if maximum_change <= float(tolerance):
            break
    aligned = np.asarray(
        [
            gauges[point_ids[left]].conj().T
            @ block
            @ gauges[point_ids[right]]
            for (left, right), block in zip(pairs, links)
        ]
    )
    return gauges, aligned


def pack(links, *, ndim, nstates):
    """Convert a link dictionary to arrays suitable for ``numpy.savez``."""

    items = sorted(links.items(), key=lambda item: (item[0][0], item[0][1]))
    if not items:
        return (
            np.empty(0, dtype=int),
            np.empty((0, int(ndim)), dtype=int),
            np.empty((0, int(nstates), int(nstates)), dtype=complex),
        )
    axes = np.asarray([axis for (axis, _), _ in items], dtype=int)
    indices = np.asarray([idx for (_, idx), _ in items], dtype=int)
    data = np.asarray([value for _, value in items], dtype=complex)
    return axes, indices, data


def unpack(axes, indices, data):
    """Rebuild a link dictionary from packed arrays."""

    return {
        (int(axis), tuple(int(i) for i in idx)): np.asarray(value, dtype=complex)
        for axis, idx, value in zip(axes, indices, data)
    }
