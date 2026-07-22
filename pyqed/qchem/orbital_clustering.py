"""Orbital graph clustering utilities for active-space and NARG workflows."""

from __future__ import annotations

import itertools
import math

import numpy as np


def two_body_cumulant(dm1, dm2):
    """Return ``lambda[p,q,r,s]`` from spin-orbital 1- and 2-RDMs.

    The convention is ``dm1[p,r] = <p^+ r>`` and
    ``dm2[p,q,r,s] = <p^+ q^+ s r>``.
    """
    gamma = np.asarray(dm1)
    gamma2 = np.asarray(dm2)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
        raise ValueError("dm1 must be a square spin-orbital matrix.")
    expected = (gamma.shape[0],) * 4
    if gamma2.shape != expected:
        raise ValueError(f"dm2 must have shape {expected}, got {gamma2.shape}.")
    disconnected = np.einsum("pr,qs->pqrs", gamma, gamma, optimize=True)
    exchange = np.einsum("ps,qr->pqrs", gamma, gamma, optimize=True)
    return gamma2 - disconnected + exchange


def orbital_mutual_correlation_graph(dm1, dm2, *, spin_order="blocked"):
    """Compute Evangelista's orbital mutual correlation from spin RDMs.

    Each graph vertex is one spatial orbital and represents the two-spin
    fragment ``{P alpha, P beta}``.  Edge weights are the pair contribution
    to the squared Frobenius norm of the two-body cumulant (JCTC 2025,
    Eq. 21).
    """
    cumulant = two_body_cumulant(dm1, dm2)
    nspin = cumulant.shape[0]
    if nspin % 2:
        raise ValueError("spin-orbital RDM dimension must be even.")
    norb = nspin // 2
    order = str(spin_order).lower().replace("-", "_")
    if order == "blocked":
        fragments = [(p, norb + p) for p in range(norb)]
    elif order in {"interleaved", "alternating"}:
        fragments = [(2 * p, 2 * p + 1) for p in range(norb)]
    else:
        raise ValueError("spin_order must be 'blocked' or 'interleaved'.")

    graph = np.zeros((norb, norb), dtype=float)
    for p in range(norb):
        a = np.asarray(fragments[p], dtype=int)
        for q in range(p + 1, norb):
            b = np.asarray(fragments[q], dtype=int)
            one_three = cumulant[np.ix_(a, b, b, b)]
            two_two_particle = cumulant[np.ix_(a, a, b, b)]
            two_two_particle_hole = cumulant[np.ix_(a, b, a, b)]
            three_one = cumulant[np.ix_(a, a, a, b)]
            value = (
                np.vdot(one_three, one_three).real
                + 0.5 * np.vdot(two_two_particle, two_two_particle).real
                + np.vdot(two_two_particle_hole, two_two_particle_hole).real
                + np.vdot(three_one, three_one).real
            )
            graph[p, q] = graph[q, p] = max(0.0, float(value))
    return graph


def maximum_weight_pair_clusters(graph):
    """Return disjoint pairs maximizing the total graph weight.

    For an odd number of orbitals, exactly one singleton is returned.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    graph = np.asarray(graph, dtype=float)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be a square matrix.")
    n = graph.shape[0]
    if n < 2:
        return [] if n == 0 else [(0,)]
    graph = np.maximum(0.0, 0.5 * (graph + graph.T))
    np.fill_diagonal(graph, 0.0)
    if not np.any(graph):
        pairs = [(i, i + 1) for i in range(0, n - 1, 2)]
        if n % 2:
            pairs.append((n - 1,))
        return pairs

    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    incidence = np.zeros((n, len(edges)), dtype=float)
    for edge, (i, j) in enumerate(edges):
        incidence[i, edge] = 1.0
        incidence[j, edge] = 1.0
    if n % 2 == 0:
        constraint = LinearConstraint(incidence, np.ones(n), np.ones(n))
    else:
        matrix = np.vstack((incidence, np.ones((1, len(edges)))))
        lower = np.concatenate((np.zeros(n), [n // 2]))
        upper = np.concatenate((np.ones(n), [n // 2]))
        constraint = LinearConstraint(matrix, lower, upper)
    objective = -np.asarray([graph[i, j] for i, j in edges])
    result = milp(
        objective,
        integrality=np.ones(len(edges)),
        bounds=Bounds(0.0, 1.0),
        constraints=constraint,
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"maximum-weight orbital pairing failed: {result.message}")
    selected = [edges[k] for k, value in enumerate(result.x) if value > 0.5]
    used = {i for pair in selected for i in pair}
    selected.extend((i,) for i in range(n) if i not in used)
    return sorted(selected, key=lambda block: block[0])


def orbital_interaction_graph(
    h1e=None,
    eri=None,
    dm=None,
    *,
    weights="integral+rdm",
    h1_weight=1.0,
    eri_weight=1.0,
    rdm_weight=1.0,
    symmetrize=True,
):
    """Build a non-negative orbital interaction graph.

    Parameters are expected in the same orthonormal orbital basis.  The
    returned matrix has zero diagonal and larger entries for orbitals that
    should prefer the same cluster.
    """
    tokens = _parse_weight_tokens(weights)
    n = _infer_norb(h1e=h1e, eri=eri, dm=dm)
    graph = np.zeros((n, n), dtype=float)

    if "h1" in tokens:
        if h1e is None:
            raise ValueError("weights request h1/integral terms, but h1e is missing.")
        h = np.asarray(h1e)
        if h.shape != (n, n):
            raise ValueError(f"h1e must have shape {(n, n)}, got {h.shape}.")
        graph += float(h1_weight) * np.abs(h)

    if "eri" in tokens:
        if eri is None:
            raise ValueError("weights request eri/integral terms, but eri is missing.")
        g = _reduce_eri_to_spatial(np.asarray(eri))
        if g.shape != (n, n, n, n):
            raise ValueError(f"eri must reduce to shape {(n, n, n, n)}, got {g.shape}.")
        coulomb_like = np.abs(np.einsum("ikjk->ij", g, optimize=True))
        exchange_like = np.abs(np.einsum("ijji->ij", g, optimize=True))
        graph += float(eri_weight) * (coulomb_like + exchange_like)

    if "rdm" in tokens:
        if dm is None:
            raise ValueError("weights request rdm/density terms, but dm is missing.")
        gamma = _reduce_dm_to_spatial(np.asarray(dm))
        if gamma.shape != (n, n):
            raise ValueError(f"dm must reduce to shape {(n, n)}, got {gamma.shape}.")
        graph += float(rdm_weight) * np.abs(gamma)

    if symmetrize:
        graph = 0.5 * (graph + graph.T)
    np.fill_diagonal(graph, 0.0)
    return np.real_if_close(graph, tol=1000).astype(float, copy=False)


def spectral_orbital_clusters(
    graph,
    *,
    n_clusters=None,
    max_size=4,
    random_state=0,
    return_labels=False,
):
    """Cluster orbitals by spectral clustering of an interaction graph."""
    graph = np.asarray(graph, dtype=float)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be a square matrix.")
    n = graph.shape[0]
    if n == 0:
        clusters = []
        labels = np.zeros(0, dtype=int)
        return (clusters, labels) if return_labels else clusters

    graph = np.maximum(0.0, 0.5 * (graph + graph.T))
    np.fill_diagonal(graph, 0.0)

    if n_clusters is None:
        if max_size is None:
            n_clusters = 1
        else:
            max_size = int(max_size)
            if max_size < 1:
                raise ValueError("max_size must be positive.")
            n_clusters = int(math.ceil(n / max_size))
    n_clusters = int(n_clusters)
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive.")
    n_clusters = min(n_clusters, n)

    if n_clusters == 1:
        clusters = [tuple(range(n))]
        labels = np.zeros(n, dtype=int)
        return (clusters, labels) if return_labels else clusters

    if not np.any(graph):
        clusters = _contiguous_clusters(n, n_clusters=n_clusters, max_size=max_size)
        labels = _labels_from_clusters(clusters, n)
        return (clusters, labels) if return_labels else clusters

    embedding = _spectral_embedding(graph, n_clusters)
    labels = _deterministic_kmeans(embedding, n_clusters, random_state=random_state)
    clusters = _clusters_from_labels(labels)
    if max_size is not None:
        clusters = _split_oversized_clusters(graph, clusters, int(max_size), random_state)
        labels = _labels_from_clusters(clusters, n)
    return (clusters, labels) if return_labels else clusters


def orbital_boundary_cut_cost(graph, clusters, *, boundary_weights=None):
    """Return the weighted correlation cut across successive cluster prefixes.

    For an ordered partition ``C[0], ..., C[m-1]``, boundary ``k`` separates
    the first ``k`` clusters from the rest.  This is the graph quantity most
    directly exposed to a left-to-right NARG truncation.
    """
    graph, clusters = _validate_ordered_clusters(graph, clusters)
    nboundaries = max(0, len(clusters) - 1)
    weights = _boundary_weights(boundary_weights, nboundaries)
    if nboundaries == 0:
        return 0.0

    prefix = np.zeros(graph.shape[0], dtype=bool)
    cost = 0.0
    for boundary, cluster in enumerate(clusters[:-1]):
        prefix[np.asarray(cluster, dtype=int)] = True
        cost += weights[boundary] * float(np.sum(graph[np.ix_(prefix, ~prefix)]))
    return float(cost)


def order_orbital_clusters(
    graph,
    clusters,
    *,
    boundary_weights=None,
    exact_limit=18,
):
    """Order fixed orbital clusters to minimize successive NARG boundary cuts.

    The exact path uses subset dynamic programming with ``O(m 2**m)`` work for
    ``m`` clusters.  A deterministic insertion and 2-opt search is used above
    ``exact_limit``.
    """
    graph, clusters = _validate_ordered_clusters(graph, clusters)
    m = len(clusters)
    if m < 2:
        return list(clusters)
    weights = _boundary_weights(boundary_weights, m - 1)
    cluster_graph = _cluster_interaction_graph(graph, clusters)
    if m <= int(exact_limit):
        order = _exact_boundary_order(cluster_graph, weights)
    else:
        order = _local_boundary_order(cluster_graph, weights)
    return [clusters[index] for index in order]


def orbital_cluster_order_candidates(
    graph,
    clusters,
    *,
    max_candidates=12,
    boundary_weights=None,
    permutation_limit=8,
):
    """Return graph-ranked candidate orders for inexpensive NARG trials."""
    graph, clusters = _validate_ordered_clusters(graph, clusters)
    m = len(clusters)
    max_candidates = int(max_candidates)
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive.")
    weights = _boundary_weights(boundary_weights, max(0, m - 1))
    cluster_graph = _cluster_interaction_graph(graph, clusters)

    if m <= int(permutation_limit):
        orders = itertools.permutations(range(m))
    else:
        seed = tuple(_local_boundary_order(cluster_graph, weights))
        pool = {seed, tuple(reversed(seed))}
        for i in range(m - 1):
            for j in range(i + 1, m):
                swapped = list(seed)
                swapped[i], swapped[j] = swapped[j], swapped[i]
                pool.add(tuple(swapped))
                pool.add(seed[:i] + tuple(reversed(seed[i : j + 1])) + seed[j + 1 :])
        orders = pool

    ranked = sorted(
        orders,
        key=lambda order: (
            _cluster_boundary_cost(cluster_graph, order, weights),
            tuple(order),
        ),
    )
    selected = []
    seen = set()
    for order in ranked:
        for candidate in (tuple(order), tuple(reversed(order))):
            if candidate in seen:
                continue
            selected.append(candidate)
            seen.add(candidate)
            if len(selected) == max_candidates:
                return [[clusters[index] for index in item] for item in selected]
    return [[clusters[index] for index in item] for item in selected]


def _validate_ordered_clusters(graph, clusters):
    graph = np.asarray(graph, dtype=float)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be a square matrix.")
    graph = np.maximum(0.0, 0.5 * (graph + graph.T))
    np.fill_diagonal(graph, 0.0)
    clusters = [tuple(int(i) for i in cluster) for cluster in clusters]
    flat = [i for cluster in clusters for i in cluster]
    if sorted(flat) != list(range(graph.shape[0])):
        raise ValueError("clusters must partition every graph vertex exactly once.")
    if any(not cluster for cluster in clusters):
        raise ValueError("clusters cannot contain empty blocks.")
    return graph, clusters


def _boundary_weights(boundary_weights, count):
    if boundary_weights is None:
        return np.ones(count, dtype=float)
    weights = np.asarray(boundary_weights, dtype=float)
    if weights.shape != (count,):
        raise ValueError(f"boundary_weights must have shape {(count,)}, got {weights.shape}.")
    if np.any(weights < 0.0):
        raise ValueError("boundary_weights must be non-negative.")
    return weights


def _cluster_interaction_graph(graph, clusters):
    m = len(clusters)
    out = np.zeros((m, m), dtype=float)
    for i in range(m):
        left = np.asarray(clusters[i], dtype=int)
        for j in range(i + 1, m):
            right = np.asarray(clusters[j], dtype=int)
            out[i, j] = out[j, i] = float(np.sum(graph[np.ix_(left, right)]))
    return out


def _cluster_boundary_cost(cluster_graph, order, weights):
    prefix = set()
    value = 0.0
    for boundary, node in enumerate(order[:-1]):
        prefix.add(node)
        outside = [i for i in order if i not in prefix]
        value += weights[boundary] * float(
            np.sum(cluster_graph[np.ix_(sorted(prefix), outside)])
        )
    return float(value)


def _exact_boundary_order(cluster_graph, weights):
    m = cluster_graph.shape[0]
    full = (1 << m) - 1
    cuts = np.zeros(1 << m, dtype=float)
    for mask in range(1, full):
        inside = [i for i in range(m) if mask & (1 << i)]
        outside = [i for i in range(m) if not mask & (1 << i)]
        cuts[mask] = float(np.sum(cluster_graph[np.ix_(inside, outside)]))

    costs = np.full(1 << m, np.inf)
    paths = [None] * (1 << m)
    costs[0] = 0.0
    paths[0] = ()
    for mask in range(1, full + 1):
        size = mask.bit_count()
        boundary_cost = 0.0 if mask == full else weights[size - 1] * cuts[mask]
        for last in range(m):
            bit = 1 << last
            if not mask & bit:
                continue
            previous = mask ^ bit
            candidate_cost = costs[previous] + boundary_cost
            candidate_path = paths[previous] + (last,)
            if candidate_cost < costs[mask] - 1.0e-14 or (
                abs(candidate_cost - costs[mask]) <= 1.0e-14
                and (paths[mask] is None or candidate_path < paths[mask])
            ):
                costs[mask] = candidate_cost
                paths[mask] = candidate_path
    return paths[full]


def _local_boundary_order(cluster_graph, weights):
    m = cluster_graph.shape[0]

    def cost(order):
        return _cluster_boundary_cost(cluster_graph, order, weights)

    order = []
    for node in sorted(range(m), key=lambda i: (-float(np.sum(cluster_graph[i])), i)):
        candidates = [order[:at] + [node] + order[at:] for at in range(len(order) + 1)]
        order = min(candidates, key=lambda candidate: (cost(candidate), tuple(candidate)))

    improved = True
    while improved:
        improved = False
        best = order
        best_key = (cost(order), tuple(order))
        for start in range(m - 1):
            for stop in range(start + 2, m + 1):
                candidate = order[:start] + list(reversed(order[start:stop])) + order[stop:]
                key = (cost(candidate), tuple(candidate))
                if key < best_key:
                    best = candidate
                    best_key = key
                    improved = True
        order = best
    return tuple(order)


def cluster_orbitals(
    h1e=None,
    eri=None,
    dm=None,
    *,
    method="spectral",
    n_clusters=None,
    max_size=4,
    weights="integral+rdm",
    active=None,
    return_info=False,
    **kwargs,
):
    """Cluster orbital indices from integrals and/or density information."""
    h1e, eri, dm, index_map = _slice_orbital_inputs(h1e, eri, dm, active)
    graph = orbital_interaction_graph(h1e, eri, dm, weights=weights, **kwargs)

    method_key = str(method).lower().replace("-", "_")
    if method_key != "spectral":
        raise ValueError("Only method='spectral' is currently implemented.")
    clusters, labels = spectral_orbital_clusters(
        graph,
        n_clusters=n_clusters,
        max_size=max_size,
        return_labels=True,
    )
    mapped = [tuple(int(index_map[i]) for i in cluster) for cluster in clusters]
    if not return_info:
        return mapped
    info = {
        "method": method_key,
        "weights": weights,
        "graph": graph,
        "labels": labels,
        "active": tuple(int(i) for i in index_map),
        "cut_ratio": graph_cut_ratio(graph, clusters),
    }
    return mapped, info


def cluster_mf_orbitals(
    mf,
    *,
    method="spectral",
    n_clusters=None,
    max_size=4,
    weights="integral+rdm",
    orbitals="canonical",
    localization="pm",
    mo_coeff=None,
    dm=None,
    active=None,
    space=None,
    localize_kwargs=None,
    return_info=False,
    return_orbitals=False,
    **kwargs,
):
    """Cluster orbitals from a mean-field object.

    The density term uses ``dm`` when supplied; otherwise it tries
    ``mf.make_rdm1()``.  Passing ``weights='integral'`` avoids density access.
    """
    (
        cluster_coeff,
        cluster_active,
        cluster_labels,
        output_coeff,
        basis_info,
    ) = _resolve_mf_cluster_orbitals(
        mf,
        orbitals=orbitals,
        localization=localization,
        mo_coeff=mo_coeff,
        active=active,
        space=space,
        localize_kwargs=localize_kwargs,
    )

    tokens = _parse_weight_tokens(weights)
    h1e = _mf_hcore_mo(mf, cluster_coeff) if "h1" in tokens else None
    eri = _mf_eri_mo(mf, cluster_coeff) if "eri" in tokens else None
    dm_mo = None
    if "rdm" in tokens:
        dm_mo = _mf_dm_mo(mf, cluster_coeff, dm)

    clustered = cluster_orbitals(
        h1e=h1e,
        eri=eri,
        dm=dm_mo,
        method=method,
        n_clusters=n_clusters,
        max_size=max_size,
        weights=weights,
        active=cluster_active,
        return_info=True,
        **kwargs,
    )
    clusters, info = clustered
    if cluster_labels is not None:
        labels = np.asarray(cluster_labels, dtype=int)
        clusters = [tuple(int(labels[i]) for i in cluster) for cluster in clusters]
        info["orbital_labels"] = tuple(int(i) for i in labels)
        info["cut_ratio"] = graph_cut_ratio(info["graph"], _unmap_clusters(clusters, labels))
    info.update(basis_info)
    info["mo_coeff"] = output_coeff

    if return_info and return_orbitals:
        return clusters, output_coeff, info
    if return_info:
        return clusters, info
    if return_orbitals:
        return clusters, output_coeff
    return clusters


def graph_cut_ratio(graph, clusters):
    graph = np.asarray(graph, dtype=float)
    total = float(np.sum(graph))
    if total <= 0.0:
        return 0.0
    mask = np.zeros_like(graph, dtype=bool)
    for cluster in clusters:
        idx = np.asarray(cluster, dtype=int)
        mask[np.ix_(idx, idx)] = True
    return float(np.sum(graph[~mask]) / total)


def _parse_weight_tokens(weights):
    if isinstance(weights, str):
        raw = weights.lower().replace("-", "_").replace(",", "+").split("+")
        tokens = {item.strip() for item in raw if item.strip()}
    else:
        tokens = {str(item).lower().replace("-", "_") for item in weights}
    expanded = set()
    for token in tokens:
        if token in {"integral", "integrals", "eri_h1", "h1_eri"}:
            expanded.update(("h1", "eri"))
        elif token in {"hcore", "one_body", "onebody"}:
            expanded.add("h1")
        elif token in {"two_body", "twobody", "coulomb"}:
            expanded.add("eri")
        elif token in {"density", "gamma", "rdm1"}:
            expanded.add("rdm")
        elif token in {"h1", "eri", "rdm"}:
            expanded.add(token)
        elif token:
            raise ValueError(f"Unknown orbital graph weight term '{token}'.")
    if not expanded:
        raise ValueError("At least one graph weight term is required.")
    return expanded


def _infer_norb(h1e=None, eri=None, dm=None):
    for obj in (h1e, dm):
        if obj is None:
            continue
        arr = _reduce_dm_to_spatial(np.asarray(obj)) if np.asarray(obj).ndim == 3 else np.asarray(obj)
        if arr.ndim >= 2:
            return int(arr.shape[-1])
    if eri is not None:
        arr = _reduce_eri_to_spatial(np.asarray(eri))
        return int(arr.shape[0])
    raise ValueError("Cannot infer orbital count without h1e, eri, or dm.")


def _reduce_dm_to_spatial(dm):
    if dm.ndim == 2:
        return dm
    if dm.ndim == 3 and dm.shape[0] == 2:
        return dm[0] + dm[1]
    raise ValueError("dm must have shape (n,n) or spin-resolved shape (2,n,n).")


def _reduce_eri_to_spatial(eri):
    if eri.ndim == 4:
        return eri
    if eri.ndim == 6 and eri.shape[:2] == (2, 2):
        return 0.25 * np.sum(eri, axis=(0, 1))
    raise ValueError("eri must have shape (n,n,n,n) or spin-block shape (2,2,n,n,n,n).")


def _slice_orbital_inputs(h1e, eri, dm, active):
    n = _infer_norb(h1e=h1e, eri=eri, dm=dm)
    if active is None:
        index_map = np.arange(n, dtype=int)
        return h1e, eri, dm, index_map
    index_map = np.asarray(active, dtype=int)
    if index_map.ndim != 1:
        raise ValueError("active must be a one-dimensional sequence of orbital indices.")
    if np.any(index_map < 0) or np.any(index_map >= n):
        raise ValueError("active contains orbital indices outside the available range.")

    idx2 = np.ix_(index_map, index_map)
    h1e_s = None if h1e is None else np.asarray(h1e)[idx2]
    dm_s = None
    if dm is not None:
        dm_arr = np.asarray(dm)
        if dm_arr.ndim == 3 and dm_arr.shape[0] == 2:
            dm_s = dm_arr[:, :, :][:, index_map][:, :, index_map]
        else:
            dm_s = dm_arr[idx2]
    eri_s = None
    if eri is not None:
        eri_arr = np.asarray(eri)
        if eri_arr.ndim == 6 and eri_arr.shape[:2] == (2, 2):
            eri_s = eri_arr[:, :, :, :, :, :][
                :, :, index_map
            ][:, :, :, index_map][:, :, :, :, index_map][:, :, :, :, :, index_map]
        else:
            eri_s = eri_arr[np.ix_(index_map, index_map, index_map, index_map)]
    return h1e_s, eri_s, dm_s, index_map


def _spectral_embedding(graph, n_clusters):
    degree = np.sum(graph, axis=1)
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 1.0e-14
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    laplacian = np.eye(graph.shape[0]) - (inv_sqrt[:, None] * graph * inv_sqrt[None, :])
    _, eigvecs = np.linalg.eigh(laplacian)
    embedding = eigvecs[:, :n_clusters]
    norms = np.linalg.norm(embedding, axis=1)
    norms[norms == 0.0] = 1.0
    return embedding / norms[:, None]


def _deterministic_kmeans(points, n_clusters, *, random_state=0, max_iter=100):
    del random_state
    points = np.asarray(points, dtype=float)
    centers = _farthest_point_centers(points, n_clusters)
    labels = -np.ones(points.shape[0], dtype=int)
    for _ in range(max_iter):
        dist2 = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            members = points[labels == k]
            if members.size:
                centers[k] = np.mean(members, axis=0)
            else:
                farthest = int(np.argmax(np.min(dist2, axis=1)))
                centers[k] = points[farthest]
    return labels


def _farthest_point_centers(points, n_clusters):
    norms = np.linalg.norm(points, axis=1)
    first = int(np.argmax(norms))
    centers = [points[first]]
    while len(centers) < n_clusters:
        dist2 = np.min(
            np.sum((points[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        centers.append(points[int(np.argmax(dist2))])
    return np.asarray(centers, dtype=float)


def _clusters_from_labels(labels):
    clusters = []
    for label in sorted(set(int(x) for x in labels)):
        idx = tuple(int(i) for i in np.flatnonzero(labels == label))
        if idx:
            clusters.append(idx)
    clusters.sort(key=lambda block: (block[0], len(block)))
    return clusters


def _labels_from_clusters(clusters, n):
    labels = np.empty(n, dtype=int)
    for label, cluster in enumerate(clusters):
        labels[np.asarray(cluster, dtype=int)] = label
    return labels


def _contiguous_clusters(n, *, n_clusters=None, max_size=None):
    if n_clusters is None:
        if max_size is None:
            return [tuple(range(n))]
        max_size = int(max_size)
        return [tuple(range(i, min(i + max_size, n))) for i in range(0, n, max_size)]
    sizes = [n // n_clusters + (1 if i < n % n_clusters else 0) for i in range(n_clusters)]
    clusters = []
    start = 0
    for size in sizes:
        clusters.append(tuple(range(start, start + size)))
        start += size
    return clusters


def _split_oversized_clusters(graph, clusters, max_size, random_state):
    split = []
    for cluster in clusters:
        if len(cluster) <= max_size:
            split.append(cluster)
            continue
        local = np.asarray(cluster, dtype=int)
        subgraph = graph[np.ix_(local, local)]
        n_sub = int(math.ceil(len(cluster) / max_size))
        subclusters = spectral_orbital_clusters(
            subgraph,
            n_clusters=n_sub,
            max_size=max_size,
            random_state=random_state,
        )
        split.extend(tuple(int(local[i]) for i in sub) for sub in subclusters)
    split.sort(key=lambda block: (block[0], len(block)))
    return split


def _default_mo_coeff(mf, mo_coeff):
    if mo_coeff is not None:
        return mo_coeff
    mo_coeff = getattr(mf, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("mf.cluster() needs mo_coeff; run SCF or pass mo_coeff explicitly.")
    return mo_coeff


def _resolve_mf_cluster_orbitals(
    mf,
    *,
    orbitals,
    localization,
    mo_coeff,
    active,
    space,
    localize_kwargs,
):
    mode = _normalize_orbital_mode(orbitals)
    coeff = _default_mo_coeff(mf, mo_coeff)
    if mode == "canonical":
        return (
            coeff,
            active,
            None,
            coeff,
            {
                "orbitals": "canonical",
                "localization": None,
                "source_orbitals": None if active is None else tuple(int(i) for i in active),
            },
        )

    if isinstance(coeff, tuple):
        raise NotImplementedError(
            "orbitals='localized' currently requires a shared spatial MO coefficient matrix; "
            "pass a localized mo_coeff explicitly or use an RHF-like reference."
        )
    if not hasattr(mf, "localize_orbitals"):
        raise NotImplementedError(
            "orbitals='localized' requires mf.localize_orbitals(). "
            "Pass mo_coeff explicitly if the orbitals are already localized."
        )

    coeff = np.asarray(coeff)
    if coeff.ndim != 2:
        raise ValueError("mo_coeff must be a 2D array for localized clustering.")
    indices, selected = _select_orbital_block(mf, coeff, active=active, space=space)
    loc_kwargs = {} if localize_kwargs is None else dict(localize_kwargs)
    localized = mf.localize_orbitals(
        method=localization,
        mo_coeff=selected,
        **loc_kwargs,
    )
    localized = np.asarray(localized)
    if localized.shape != selected.shape:
        raise ValueError(
            f"localized orbital block has shape {localized.shape}, expected {selected.shape}."
        )

    output_coeff = coeff.copy()
    if np.max(indices, initial=-1) < output_coeff.shape[1]:
        output_coeff[:, indices] = localized
    else:
        output_coeff = localized

    return (
        localized,
        None,
        indices,
        output_coeff,
        {
            "orbitals": "localized",
            "localization": str(localization).lower().replace("-", "_"),
            "source_orbitals": tuple(int(i) for i in indices),
        },
    )


def _normalize_orbital_mode(orbitals):
    key = str(orbitals).lower().replace("-", "_")
    if key in {"canonical", "mo", "scf", "original"}:
        return "canonical"
    if key in {"localized", "localised", "local"}:
        return "localized"
    raise ValueError("orbitals must be 'canonical' or 'localized'.")


def _select_orbital_block(mf, coeff, *, active, space):
    nmo = coeff.shape[1]
    if active is not None:
        active = np.asarray(active, dtype=int)
        if active.ndim != 1:
            raise ValueError("active must be a one-dimensional sequence of orbital indices.")
        if active.size == 0:
            raise ValueError("active must contain at least one orbital.")
        if np.all((active >= 0) & (active < nmo)):
            return active, coeff[:, active]
        if active.size == nmo:
            return active, coeff
        raise ValueError("active contains orbital indices outside the available MO range.")

    space_key = "all" if space is None else str(space).lower().replace("-", "_")
    if space_key in {"all", "full"}:
        indices = np.arange(nmo, dtype=int)
    elif space_key in {"occ", "occupied"}:
        occ = getattr(mf, "mo_occ", None)
        if occ is None:
            raise ValueError("space='occ' requires mf.mo_occ.")
        indices = np.flatnonzero(np.asarray(occ) > 0.5)
    elif space_key in {"vir", "virtual"}:
        occ = getattr(mf, "mo_occ", None)
        if occ is None:
            raise ValueError("space='vir' requires mf.mo_occ.")
        indices = np.flatnonzero(np.asarray(occ) <= 0.5)
    else:
        raise ValueError("space must be 'all', 'occ', or 'vir'.")
    if indices.size == 0:
        raise ValueError(f"space='{space_key}' selected no orbitals.")
    return indices, coeff[:, indices]


def _unmap_clusters(clusters, labels):
    lookup = {int(label): i for i, label in enumerate(np.asarray(labels, dtype=int))}
    return [tuple(lookup[int(label)] for label in cluster) for cluster in clusters]


def _mf_hcore_mo(mf, mo_coeff):
    if hasattr(mf, "get_hcore_mo"):
        h1e = mf.get_hcore_mo(mo_coeff=mo_coeff)
    else:
        hcore = mf.get_hcore() if hasattr(mf, "get_hcore") else mf.mol.hcore
        if isinstance(mo_coeff, tuple):
            h1e = tuple(c.conj().T @ hcore @ c for c in mo_coeff)
        else:
            h1e = mo_coeff.conj().T @ hcore @ mo_coeff
    if isinstance(h1e, tuple) or (isinstance(h1e, np.ndarray) and h1e.ndim == 3):
        arr = np.asarray(h1e)
        return np.mean(arr, axis=0)
    return h1e


def _mf_eri_mo(mf, mo_coeff):
    if hasattr(mf, "get_eri_mo"):
        return mf.get_eri_mo(mo_coeff=mo_coeff)
    eri = getattr(mf, "eri", None)
    if eri is None:
        eri = getattr(mf.mol, "eri", None)
    if eri is None:
        raise ValueError("mf.cluster() needs ERIs for integral graph weights.")
    c = mo_coeff[0] if isinstance(mo_coeff, tuple) else mo_coeff
    return np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, c.conj(), c, c.conj(), c, optimize=True)


def _mf_dm_mo(mf, mo_coeff, dm):
    if dm is None:
        if not hasattr(mf, "make_rdm1"):
            raise ValueError("weights request RDM terms, but mf has no make_rdm1().")
        dm = mf.make_rdm1()
    overlap = mf.get_ovlp() if hasattr(mf, "get_ovlp") else getattr(mf.mol, "overlap", None)
    if overlap is None:
        overlap = np.eye(np.asarray(mo_coeff[0] if isinstance(mo_coeff, tuple) else mo_coeff).shape[0])
    s = np.asarray(overlap)
    dm_arr = np.asarray(dm)
    if isinstance(mo_coeff, tuple):
        coeffs = tuple(np.asarray(c) for c in mo_coeff)
        if dm_arr.ndim == 2:
            return sum(c.conj().T @ s @ dm_arr @ s @ c for c in coeffs)
        if dm_arr.ndim == 3 and dm_arr.shape[0] == 2:
            return np.asarray(
                [c.conj().T @ s @ dm_arr[i] @ s @ c for i, c in enumerate(coeffs)]
            )
        raise ValueError("UHF density must have shape (n,n) or (2,n,n).")
    c = np.asarray(mo_coeff)
    if dm_arr.ndim == 3 and dm_arr.shape[0] == 2:
        dm_arr = dm_arr[0] + dm_arr[1]
    return c.conj().T @ s @ dm_arr @ s @ c
