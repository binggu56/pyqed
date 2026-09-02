"""SO2 spin-vibronic fields for symmetry-aware LDR fitting."""

from __future__ import annotations

import heapq
from itertools import product

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casci import overlap as casci_overlap

from .overlap import procrustes


SO2_SPECIES = ("O", "S", "O")
SO2_MS = (-1, 0, 1)
SO2_POINT_GROUP = {
    "E": (1.0, 1.0, 1.0),
    "C2(x)": (1.0, -1.0, -1.0),
    "sigma_xy": (1.0, 1.0, -1.0),
    "sigma_xz": (1.0, -1.0, 1.0),
}


def geometry(coordinate):
    """Return planar SO2 Cartesian coordinates in the bond-bisector frame."""

    r1, r2, theta = np.asarray(coordinate, dtype=float)
    half = 0.5 * theta
    return np.asarray(
        (
            (r1 * np.cos(half), r1 * np.sin(half), 0.0),
            (0.0, 0.0, 0.0),
            (r2 * np.cos(half), -r2 * np.sin(half), 0.0),
        )
    )


def molecule(coordinate, basis):
    value = Molecule(
        atom=list(zip(SO2_SPECIES, geometry(coordinate))),
        charge=0,
        spin=0,
        unit="bohr",
        basis=basis,
    )
    return value.build(eri="dense")


def invariant_coordinates(coordinates, bounds):
    """Scale symmetric stretch, unsigned asymmetric stretch, and bend."""

    coordinates = np.asarray(coordinates, dtype=float)
    rs_min, rs_max, ra_max, theta_min, theta_max = map(float, bounds)
    values = np.column_stack(
        (
            0.5 * (coordinates[:, 0] + coordinates[:, 1]),
            0.5 * np.abs(coordinates[:, 0] - coordinates[:, 1]),
            coordinates[:, 2],
        )
    )
    lower = np.asarray((rs_min, 0.0, theta_min))
    upper = np.asarray((rs_max, ra_max, theta_max))
    return (values - lower) / (upper - lower)


def sparse_overlap_graph(coordinates, bounds, neighbors=4):
    """Return a connected local k-nearest-neighbor graph."""

    scaled = invariant_coordinates(coordinates, bounds)
    distances = cdist(scaled, scaled)
    np.fill_diagonal(distances, np.inf)
    neighbors = min(int(neighbors), len(coordinates) - 1)
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    pairs = set()
    for left in range(len(coordinates)):
        nearest = np.argpartition(distances[left], neighbors - 1)[:neighbors]
        pairs.update(tuple(sorted((left, int(right)))) for right in nearest)
    finite = distances.copy()
    np.fill_diagonal(finite, 0.0)
    tree = minimum_spanning_tree(finite).tocoo()
    pairs.update(
        tuple(sorted((int(left), int(right))))
        for left, right in zip(tree.row, tree.col)
    )
    pairs = np.asarray(sorted(pairs), dtype=int)
    lengths = np.linalg.norm(
        scaled[pairs[:, 0]] - scaled[pairs[:, 1]], axis=1
    )
    return pairs, lengths


def adaptive_points(
    feature,
    coordinates,
    bounds,
    count,
    *,
    candidate_pool=2048,
    seed=0,
    batch_size=256,
    max_distance=0.30,
):
    """Acquire continuous canonical SO2 points from coverage and frame defect."""

    coordinates = np.asarray(coordinates, dtype=float)
    count = int(count)
    candidate_pool = max(int(candidate_pool), count)
    if count < 1:
        return np.empty((0, 3)), {"candidate_pool": 0, "selected": []}
    rs_min, rs_max, ra_max, theta_min, theta_max = map(float, bounds)
    exponent = int(np.ceil(np.log2(candidate_pool)))
    unit = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(exponent)
    unit = unit[:candidate_pool]
    symmetric = rs_min + unit[:, 0] * (rs_max - rs_min)
    asymmetric = unit[:, 1] * ra_max
    candidates = np.column_stack(
        (
            symmetric + asymmetric,
            symmetric - asymmetric,
            theta_min + unit[:, 2] * (theta_max - theta_min),
        )
    )
    values = np.concatenate(
        [
            np.asarray(feature.predict(candidates[start : start + batch_size]))
            for start in range(0, len(candidates), int(batch_size))
        ]
    )
    nstates = values.shape[-1]
    gram = np.einsum("nra,nrb->nab", values.conj(), values, optimize=True)
    defects = np.linalg.norm(gram - np.eye(nstates), axis=(1, 2))
    positive = defects[defects > 64.0 * np.finfo(float).eps]
    defect_scale = float(np.median(positive)) if len(positive) else 1.0
    candidate_scaled = invariant_coordinates(candidates, bounds)
    sampled_scaled = invariant_coordinates(coordinates, bounds)
    distance = np.min(cdist(candidate_scaled, sampled_scaled), axis=1)
    max_distance = float(max_distance)
    if max_distance <= 0.0:
        raise ValueError("max_distance must be positive")
    active = distance <= max_distance
    if np.count_nonzero(active) < count:
        raise RuntimeError(
            "continuous candidate pool contains too few points inside the "
            f"adaptive trust radius {max_distance:.3f}"
        )
    base = 1.0 + defects / max(defect_scale, np.finfo(float).tiny)
    selected = []
    scores = []
    selected_defects = []
    for _ in range(min(count, len(candidates))):
        acquisition = np.where(active, distance * base, -np.inf)
        index = int(np.argmax(acquisition))
        selected.append(index)
        scores.append(float(acquisition[index]))
        selected_defects.append(float(defects[index]))
        active[index] = False
        distance = np.minimum(
            distance,
            np.linalg.norm(candidate_scaled - candidate_scaled[index], axis=1),
        )
    chosen = candidates[selected]
    return chosen, {
        "backend": "continuous-frame-defect-and-coverage",
        "candidate_pool": len(candidates),
        "trust_radius": max_distance,
        "selected": chosen.tolist(),
        "selected_scores": scores,
        "selected_self_overlap_defects": selected_defects,
        "candidate_maximum_self_overlap_defect": float(np.max(defects)),
        "candidate_median_self_overlap_defect": float(np.median(defects)),
    }


def full_spin_overlap(left, right):
    """Return the raw 3-singlet/3-triplet-Ms overlap without unitarization."""

    singlet = np.asarray(
        casci_overlap(left["singlet_frame"], right["singlet_frame"]),
        dtype=complex,
    )
    n_singlets = singlet.shape[0]
    n_triplets = len(left["triplet_frames"][0].ci)
    size = n_singlets + len(SO2_MS) * n_triplets
    value = np.zeros((size, size), dtype=complex)
    value[:n_singlets, :n_singlets] = singlet
    for ms_index, ms in enumerate(SO2_MS):
        block = np.asarray(
            casci_overlap(
                left["triplet_frames"][ms], right["triplet_frames"][ms]
            ),
            dtype=complex,
        )
        for left_root in range(n_triplets):
            for right_root in range(n_triplets):
                row = n_singlets + len(SO2_MS) * left_root + ms_index
                column = n_singlets + len(SO2_MS) * right_root + ms_index
                value[row, column] = block[left_root, right_root]
    return value


def _ao_diagonal_operator(mol, signs, tolerance=1.0e-10):
    def same_array(left, right):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        return left.shape == right.shape and np.allclose(
            left, right, atol=tolerance, rtol=0.0
        )

    signs = np.asarray(signs, dtype=float)
    transform = getattr(mol, "_ao_cart2sph", None)
    basis = list(mol._bas_cart if transform is not None else mol._bas)
    operator = np.zeros((len(basis), len(basis)))
    for source, function in enumerate(basis):
        shell = tuple(int(value) for value in function.shell)
        origin = signs * np.asarray(function.origin, dtype=float)
        matches = [
            target
            for target, candidate in enumerate(basis)
            if tuple(int(value) for value in candidate.shell) == shell
            and np.allclose(candidate.origin, origin, atol=tolerance, rtol=0.0)
            and same_array(candidate.exps, function.exps)
            and same_array(candidate.coefs, function.coefs)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"could not map AO {source} under SO2 symmetry")
        operator[matches[0], source] = float(
            np.prod(signs ** np.asarray(shell))
        )
    if transform is not None:
        transform = np.asarray(transform, dtype=float)
        operator, _residual, rank, _singular = np.linalg.lstsq(
            transform, operator @ transform, rcond=None
        )
        if rank != transform.shape[1]:
            raise RuntimeError("Cartesian-to-spherical AO map lost rank")
        operator[np.abs(operator) < 100.0 * np.finfo(float).eps] = 0.0
    return operator


def _spatial_representation(frame, mol, signs, tolerance=2.0e-5):
    ao = _ao_diagonal_operator(mol, signs)
    metric = np.asarray(mol.overlap)
    orbitals = np.asarray(frame.mo_coeff)
    mo = orbitals.conj().T @ metric @ ao @ orbitals
    raw = np.asarray(casci_overlap(frame, frame, s=mo), dtype=complex)
    raw = 0.5 * (raw + raw.conj().T)
    diagonal = np.real(np.diag(raw))
    representation = np.diag(np.where(diagonal >= 0.0, 1.0, -1.0)).astype(complex)
    defect = float(np.max(np.abs(np.abs(diagonal) - 1.0)))
    off_diagonal = float(np.max(np.abs(raw - np.diag(np.diag(raw)))))
    if max(defect, off_diagonal) > float(tolerance):
        raise RuntimeError(
            "SO2 electronic roots are not symmetry resolved: "
            f"diagonal defect={defect:.3e}, off-diagonal={off_diagonal:.3e}"
        )
    return representation, {
        "diagonal_defect": defect,
        "off_diagonal_max": off_diagonal,
        "involution_defect": float(np.linalg.norm(raw @ raw - np.eye(len(raw)))),
    }


def spin_one_representation(signs):
    r"""Return the axial-vector spin-one action in $M_S=(-1,0,+1)$ order."""

    polar = np.diag(np.asarray(signs, dtype=float))
    axial = np.linalg.det(polar) * polar
    root_two = np.sqrt(2.0)
    spherical = np.column_stack(
        (
            np.asarray((1.0, -1.0j, 0.0)) / root_two,
            np.asarray((0.0, 0.0, 1.0)),
            -np.asarray((1.0, 1.0j, 0.0)) / root_two,
        )
    )
    value = spherical.conj().T @ axial @ spherical
    value[np.abs(value) < 1.0e-14] = 0.0
    return value


def point_group_representations(record, basis):
    """Lift C2v into the stored 12-state spin-vibronic manifold."""

    coordinate = np.asarray(record["coordinate"], dtype=float)
    if not np.isclose(coordinate[0], coordinate[1], atol=1.0e-12):
        raise ValueError("point-group calibration record must have r1 = r2")
    mol = molecule(coordinate, basis)
    output = {}
    diagnostics = {}
    for name, signs in SO2_POINT_GROUP.items():
        singlet, singlet_info = _spatial_representation(
            record["singlet_frame"], mol, signs
        )
        triplet, triplet_info = _spatial_representation(
            record["triplet_frames"][0], mol, signs
        )
        spin = spin_one_representation(signs)
        triplet_spin = np.kron(triplet, spin)
        representation = np.zeros(
            (len(singlet) + len(triplet_spin),) * 2, dtype=complex
        )
        representation[: len(singlet), : len(singlet)] = singlet
        representation[len(singlet) :, len(singlet) :] = triplet_spin
        output[name] = representation
        diagnostics[name] = {
            "singlet": singlet_info,
            "triplet": triplet_info,
            "unitarity_defect": float(
                np.linalg.norm(representation.conj().T @ representation - np.eye(len(representation)))
            ),
            "involution_defect": float(
                np.linalg.norm(representation @ representation - np.eye(len(representation)))
            ),
        }
    diagnostics["closure_defect"] = float(
        np.linalg.norm(output["C2(x)"] @ output["sigma_xy"] - output["sigma_xz"])
    )
    return output, diagnostics


def plane_parities(record, basis):
    """Return singlet and triplet-root reflection parities for planar SO2."""

    coordinate = np.asarray(record["coordinate"], dtype=float)
    mol = molecule(coordinate, basis)
    signs = SO2_POINT_GROUP["sigma_xy"]
    singlet, singlet_info = _spatial_representation(
        record["singlet_frame"], mol, signs
    )
    triplet, triplet_info = _spatial_representation(
        record["triplet_frames"][0], mol, signs
    )
    return (
        np.diag(singlet).real.astype(int),
        np.diag(triplet).real.astype(int),
        {"singlet": singlet_info, "triplet": triplet_info},
    )


def frame_parities(frame, mol, operation="sigma_xy"):
    r"""Return resolved $\pm1$ characters for one CASCI root frame."""

    try:
        signs = SO2_POINT_GROUP[str(operation)]
    except KeyError as error:
        raise ValueError(f"unknown SO2 symmetry operation {operation!r}") from error
    representation, diagnostics = _spatial_representation(frame, mol, signs)
    return np.diag(representation).real.astype(int), diagnostics


def canonical_spin_vibronic_permutation(
    singlet_parities,
    triplet_parities,
    target_singlet,
    target_triplet,
):
    """Reorder roots to one fixed plane-reflection sector sequence."""

    def root_permutation(values, target):
        values = np.asarray(values, dtype=int)
        target = np.asarray(target, dtype=int)
        if sorted(values.tolist()) != sorted(target.tolist()):
            raise ValueError("selected roots do not span the target symmetry sectors")
        available = {
            sign: list(np.flatnonzero(values == sign)) for sign in (-1, 1)
        }
        return np.asarray([available[int(sign)].pop(0) for sign in target], dtype=int)

    singlet = root_permutation(singlet_parities, target_singlet)
    triplet = root_permutation(triplet_parities, target_triplet)
    nsinglet = len(singlet)
    indices = list(map(int, singlet))
    for root in triplet:
        indices.extend(nsinglet + 3 * int(root) + ms for ms in range(3))
    return np.asarray(indices, dtype=int)


def select_root_sectors(parities, target):
    """Select the lowest-energy roots spanning a fixed parity sequence.

    ``parities`` must follow the energy ordering of a larger candidate-root
    calculation.  Each candidate is used at most once, so the returned indices
    define a fixed symmetry-sector state window without fitting energy-ordered
    roots that change character across geometry.
    """

    parities = np.asarray(parities, dtype=int)
    target = np.asarray(target, dtype=int)
    if np.any(np.abs(parities) != 1) or np.any(np.abs(target) != 1):
        raise ValueError("root parities and target sectors must be +1 or -1")
    available = {
        sign: list(map(int, np.flatnonzero(parities == sign)))
        for sign in (-1, 1)
    }
    selected = []
    for sign in target:
        pool = available[int(sign)]
        if not pool:
            counts = {key: len(value) for key, value in available.items()}
            raise ValueError(
                "candidate roots do not span the target symmetry sectors: "
                f"remaining counts={counts}, target={target.tolist()}"
            )
        selected.append(pool.pop(0))
    return np.asarray(selected, dtype=int)


def symmetry_block_procrustes(value, representations):
    """Take independent Procrustes factors in joint involution sectors."""

    value = np.asarray(value, dtype=complex)
    representations = tuple(np.asarray(item, dtype=complex) for item in representations)
    identity = np.eye(len(value), dtype=complex)
    rotation = np.zeros_like(value)
    for characters in product((-1.0, 1.0), repeat=len(representations)):
        projector = identity.copy()
        for character, representation in zip(characters, representations):
            projector = projector @ (identity + character * representation) / 2.0
        projector = 0.5 * (projector + projector.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(projector)
        basis = eigenvectors[:, eigenvalues > 0.5]
        if basis.shape[1]:
            block = basis.conj().T @ value @ basis
            rotation += basis @ procrustes(block)[0] @ basis.conj().T
    defect = np.linalg.norm(rotation.conj().T @ rotation - identity)
    if defect > 1.0e-8:
        raise RuntimeError(f"symmetry-block Procrustes defect {defect:.3e}")
    return rotation


def procrustes_fields(records, pairs, overlaps, representations, anchor):
    """Graph-transport Hamiltonians while retaining nonunitary link residuals."""

    coordinates = np.asarray([record["coordinate"] for record in records])
    n_singlets = sum(label.startswith("S") for label in records[anchor]["labels"])
    spin_tag = np.diag(
        np.r_[np.ones(n_singlets), -np.ones(len(records[anchor]["labels"]) - n_singlets)]
    ).astype(complex)
    fixed = representations["sigma_xy"]
    exchange = representations["C2(x)"]
    symmetric = np.isclose(coordinates[:, 0], coordinates[:, 1], atol=1.0e-12)
    pairs = np.asarray(pairs, dtype=int)
    overlaps = np.asarray(overlaps, dtype=complex)
    npoints = len(records)
    nstates = overlaps.shape[-1]
    gauges = np.empty((npoints, nstates, nstates), dtype=complex)
    gauges[anchor] = np.eye(nstates)
    singular = np.linalg.svd(overlaps, compute_uv=False)
    reliability = np.maximum(np.min(singular, axis=1), np.finfo(float).tiny)
    adjacency = [[] for _ in range(npoints)]
    for edge, (left, right) in enumerate(pairs):
        adjacency[left].append((right, edge, False))
        adjacency[right].append((left, edge, True))
    assigned = {int(anchor)}
    queue = []

    def add_edges(point):
        for neighbor, edge, reverse in adjacency[point]:
            if neighbor not in assigned:
                heapq.heappush(
                    queue,
                    (-float(reliability[edge]), point, neighbor, edge, reverse),
                )

    add_edges(int(anchor))
    tree_edges = []
    while len(assigned) < npoints:
        if not queue:
            raise ValueError("SO2 overlap graph is disconnected")
        _weight, parent, point, edge, reverse = heapq.heappop(queue)
        if point in assigned:
            continue
        sectors = (spin_tag, fixed, exchange) if symmetric[point] else (spin_tag, fixed)
        if reverse:
            effective = overlaps[edge] @ gauges[parent]
            gauges[point] = symmetry_block_procrustes(effective, sectors)
        else:
            effective = gauges[parent].conj().T @ overlaps[edge]
            gauges[point] = symmetry_block_procrustes(effective, sectors).conj().T
        assigned.add(point)
        tree_edges.append(edge)
        add_edges(point)
    raw_hamiltonians = np.asarray([record["h_total"] for record in records])
    shift = float(np.trace(raw_hamiltonians[anchor]).real / raw_hamiltonians.shape[-1])
    shifted = raw_hamiltonians - shift * np.eye(raw_hamiltonians.shape[-1])
    hamiltonians = gauges.conj().swapaxes(-1, -2) @ shifted @ gauges
    links = np.asarray(
        [
            gauges[left].conj().T @ overlap @ gauges[right]
            for (left, right), overlap in zip(pairs, overlaps)
        ]
    )
    return hamiltonians, links, gauges, shift, {
        "method": "maximum-reliability graph Procrustes transport",
        "tree_edges": list(map(int, tree_edges)),
        "minimum_tree_singular_value": float(np.min(reliability[tree_edges])),
        "minimum_graph_singular_value": float(np.min(reliability)),
    }
