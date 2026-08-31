#!/usr/bin/env python3
"""Assemble a gauge-consistent scattered phenol data set for MACE-Y."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyqed.ldr.overlap import procrustes


MID_THETA = np.deg2rad(108.8)
REFLECTION = np.diag((1.0, 1.0, -1.0)).astype(complex)


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _key(coordinate):
    return tuple(np.round(np.asarray(coordinate, dtype=float), 10))


def _forward_gauge(left_gauge, overlap):
    rotation = procrustes(left_gauge.conj().T @ overlap)[0]
    return rotation.conj().T


def _backward_gauge(overlap, right_gauge):
    return procrustes(overlap @ right_gauge)[0]


def regauge_inner_cross(data, planar_radii, planar_gauges):
    """Anchor each old 3D angular star to the resolved planar P gauge."""

    points = tuple(map(tuple, data["points"]))
    point_ids = {point: index for index, point in enumerate(points)}
    overlaps = {
        (tuple(left), tuple(right)): value
        for (left, right), value in zip(data["pairs"], data["selected_overlaps"])
    }
    gauges = np.zeros((len(points), 3, 3), dtype=complex)
    for radial, radius in enumerate(data["r_oh"]):
        match = np.flatnonzero(np.isclose(planar_radii, radius, atol=1.0e-9))
        if len(match) != 1:
            raise RuntimeError(f"resolved planar gauge does not contain R={radius}")
        anchor = (radial, 2, 1)
        gauges[point_ids[anchor]] = planar_gauges[match[0]]

        for left_phi, right_phi in ((2, 3), (3, 4)):
            left = (radial, left_phi, 1)
            right = (radial, right_phi, 1)
            gauges[point_ids[right]] = _forward_gauge(
                gauges[point_ids[left]], overlaps[(left, right)]
            )
        for left_phi, right_phi in ((1, 2), (0, 1)):
            left = (radial, left_phi, 1)
            right = (radial, right_phi, 1)
            gauges[point_ids[left]] = _backward_gauge(
                overlaps[(left, right)], gauges[point_ids[right]]
            )
        for torsion in range(len(data["phi"])):
            center = (radial, torsion, 1)
            upper = (radial, torsion, 2)
            lower = (radial, torsion, 0)
            gauges[point_ids[upper]] = _forward_gauge(
                gauges[point_ids[center]], overlaps[(center, upper)]
            )
            gauges[point_ids[lower]] = _backward_gauge(
                overlaps[(lower, center)], gauges[point_ids[center]]
            )

    hamiltonians = []
    for point, roots, gauge in zip(points, data["root_indices"], gauges):
        diagonal = np.diag(data["energies"][point][roots])
        value = gauge.conj().T @ diagonal @ gauge
        hamiltonians.append(0.5 * (value + value.conj().T))
    links = np.asarray(
        [
            gauges[point_ids[tuple(left)]].conj().T
            @ overlap
            @ gauges[point_ids[tuple(right)]]
            for (left, right), overlap in zip(
                data["pairs"], data["selected_overlaps"]
            )
        ]
    )
    return gauges, np.asarray(hamiltonians), links


class Dataset:
    def __init__(self):
        self.coordinates = []
        self.hamiltonians = []
        self.energy_sources = []
        self.point_ids = {}
        self.pairs = []
        self.links = []
        self.link_sources = []
        self.edge_ids = {}
        self.maximum_duplicate_energy_defect = 0.0
        self.maximum_duplicate_link_defect = 0.0

    def add_energy(self, coordinate, hamiltonian, source):
        coordinate = np.asarray(coordinate, dtype=float)
        hamiltonian = np.asarray(hamiltonian, dtype=complex)
        key = _key(coordinate)
        if key in self.point_ids:
            index = self.point_ids[key]
            defect = float(np.linalg.norm(self.hamiltonians[index] - hamiltonian))
            self.maximum_duplicate_energy_defect = max(
                self.maximum_duplicate_energy_defect, defect
            )
            if defect > 1.0e-7:
                raise RuntimeError(
                    f"inconsistent P-gauge Hamiltonian at {coordinate}: {defect:.3e}"
                )
            self.energy_sources[index] += "+" + str(source)
            return index
        index = len(self.coordinates)
        self.point_ids[key] = index
        self.coordinates.append(coordinate)
        self.hamiltonians.append(hamiltonian)
        self.energy_sources.append(str(source))
        return index

    def add_link(self, left, right, link, source):
        left_id = self.point_ids[_key(left)]
        right_id = self.point_ids[_key(right)]
        link = np.asarray(link, dtype=complex)
        if left_id == right_id:
            raise ValueError("a link must connect distinct coordinates")
        if left_id > right_id:
            left_id, right_id = right_id, left_id
            link = link.conj().T
        key = (left_id, right_id)
        if key in self.edge_ids:
            index = self.edge_ids[key]
            defect = float(np.linalg.norm(self.links[index] - link))
            self.maximum_duplicate_link_defect = max(
                self.maximum_duplicate_link_defect, defect
            )
            if defect > 2.0e-7:
                raise RuntimeError(
                    f"inconsistent P-gauge link for edge {key}: {defect:.3e}"
                )
            self.link_sources[index] += "+" + str(source)
            return index
        index = len(self.pairs)
        self.edge_ids[key] = index
        self.pairs.append(key)
        self.links.append(link)
        self.link_sources.append(str(source))
        return index


def _spanning_tree_mask(size, pairs, order=None):
    parent = np.arange(size)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    mask = np.zeros(len(pairs), dtype=bool)
    order = np.arange(len(pairs)) if order is None else np.asarray(order, dtype=int)
    for edge in order:
        left, right = pairs[edge]
        root_left, root_right = find(int(left)), find(int(right))
        if root_left == root_right:
            continue
        parent[root_right] = root_left
        mask[edge] = True
    if len({find(index) for index in range(size)}) != 1:
        raise RuntimeError("assembled electronic-overlap graph is disconnected")
    return mask


def _holdouts(dataset, seed=73):
    rng = np.random.default_rng(seed)
    sources = np.asarray(dataset.energy_sources)
    energy = np.zeros(len(sources), dtype=bool)
    for label in ("inner-3d", "resolved-planar", "outer-angular"):
        candidates = np.flatnonzero(np.char.find(sources, label) >= 0)
        if len(candidates):
            count = max(1, int(round(0.15 * len(candidates))))
            energy[rng.choice(candidates, size=min(count, len(candidates)), replace=False)] = True
    anchor = dataset.point_ids[_key((0.95, 0.0, MID_THETA))]
    energy[anchor] = False

    pairs = np.asarray(dataset.pairs, dtype=int)
    tree = _spanning_tree_mask(len(sources), pairs, rng.permutation(len(pairs)))
    coordinates = np.asarray(dataset.coordinates)
    deltas = coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]]
    axes = np.argmax(np.abs(deltas), axis=1)
    link = np.zeros(len(pairs), dtype=bool)
    for axis in range(coordinates.shape[1]):
        candidates = np.flatnonzero((~tree) & (axes == axis))
        if len(candidates):
            count = max(1, int(round(0.15 * len(candidates))))
            link[rng.choice(candidates, size=count, replace=False)] = True
    return energy, link, anchor, tree


def radial_backbone(planar, bridge, minimum_radius):
    """Join the continuous planar component to the resolved outer bridge."""
    bridge_radii = np.asarray(bridge["combined_radii"])
    bridge_hamiltonian = np.asarray(bridge["combined_p_hamiltonian"])
    keep = bridge_radii >= float(minimum_radius) - 1.0e-10
    bridge_radii = bridge_radii[keep]
    bridge_hamiltonian = bridge_hamiltonian[keep]
    if not bridge_radii.size:
        raise ValueError("the requested minimum radius excludes the bridge backbone")

    planar_radii = np.asarray(planar["radii"])
    if "tracked_singular_values" in planar:
        singular = np.min(np.asarray(planar["tracked_singular_values"]), axis=1)
        failures = np.flatnonzero(singular < 0.90)
        planar_stop = len(planar_radii) - 1 if not len(failures) else int(failures[0])
        use_planar = (
            (planar_radii >= float(minimum_radius) - 1.0e-10)
            & (np.arange(len(planar_radii)) <= planar_stop)
        )
    else:
        use_planar = (
            (planar_radii >= float(minimum_radius) - 1.0e-10)
            & (planar_radii < bridge_radii[0] - 1.0e-10)
        )
    selected_radii = planar_radii[use_planar]
    if selected_radii.size:
        after_planar = bridge_radii > selected_radii[-1] + 1.0e-10
        bridge_radii = bridge_radii[after_planar]
        bridge_hamiltonian = bridge_hamiltonian[after_planar]
    radii = np.concatenate((selected_radii, bridge_radii))
    hamiltonian = np.concatenate(
        (np.asarray(planar["p_hamiltonian"])[use_planar], bridge_hamiltonian), axis=0
    )
    return radii, hamiltonian


def assemble(
    inner, planar, bridge, angular, *, bridge_history=(), minimum_radius=0.95
):
    dataset = Dataset()
    gauges, inner_hamiltonians, inner_links = regauge_inner_cross(
        inner, planar["radii"], planar["p_gauge"]
    )
    inner_coordinates = np.asarray(
        [
            (inner["r_oh"][r], inner["phi"][p], inner["theta"][t])
            for r, p, t in inner["points"]
        ]
    )
    for coordinate, value in zip(inner_coordinates, inner_hamiltonians):
        dataset.add_energy(coordinate, value, "inner-3d")

    radii, radial_hamiltonian = radial_backbone(planar, bridge, minimum_radius)
    radial_coordinates = np.column_stack(
        (
            radii,
            np.zeros(len(radii)),
            np.full(len(radii), MID_THETA),
        )
    )
    for coordinate, value in zip(
        radial_coordinates, radial_hamiltonian
    ):
        dataset.add_energy(coordinate, value, "resolved-planar")

    old_edges = {
        (round(float(left), 10), round(float(right), 10)): value
        for left, right, value in zip(
            planar["radii"][:-1], planar["radii"][1:], planar["p_links"]
        )
    }
    bridge_edges = {}
    for bridge_data in (*tuple(bridge_history), bridge):
        bridge_edges.update(
            {
                (round(float(left), 10), round(float(right), 10)): value
                for left, right, value in zip(
                    bridge_data["radii"][:-1],
                    bridge_data["radii"][1:],
                    bridge_data["p_links"],
                )
            }
        )
    for left, right in zip(radii[:-1], radii[1:]):
        key = (round(float(left), 10), round(float(right), 10))
        candidates = [
            value
            for value in (old_edges.get(key), bridge_edges.get(key))
            if value is not None
        ]
        if not candidates:
            raise RuntimeError(f"missing resolved planar link {key}")
        value = max(
            candidates,
            key=lambda candidate: np.min(np.linalg.svd(candidate, compute_uv=False)),
        )
        dataset.add_link(
            (left, 0.0, MID_THETA),
            (right, 0.0, MID_THETA),
            value,
            "resolved-planar",
        )

    point_coordinates = {
        tuple(point): coordinate
        for point, coordinate in zip(map(tuple, inner["points"]), inner_coordinates)
    }
    for (left, right), value in zip(inner["pairs"], inner_links):
        dataset.add_link(
            point_coordinates[tuple(left)],
            point_coordinates[tuple(right)],
            value,
            "inner-3d",
        )

    angular_sets = angular if isinstance(angular, (tuple, list)) else (angular,)
    for angular_data in angular_sets:
        for radial, radius in enumerate(angular_data["radii"]):
            for torsion, value in zip(
                angular_data["torsions"],
                angular_data["phi_p_hamiltonian"][radial],
            ):
                coordinate = (float(radius), float(torsion), MID_THETA)
                dataset.add_energy(coordinate, value, "outer-angular")
                if torsion > 1.0e-12:
                    reflected = REFLECTION @ value @ REFLECTION
                    dataset.add_energy(
                        (float(radius), -float(torsion), MID_THETA),
                        reflected,
                        "outer-angular-reflection",
                    )
            for bend, value in zip(
                angular_data["bends"],
                angular_data["bend_p_hamiltonian"][radial],
            ):
                dataset.add_energy(
                    (float(radius), 0.0, float(bend)), value, "outer-angular"
                )

            phi = angular_data["torsions"]
            positive_links = angular_data["phi_selected_links"][radial]
            positive_gauges = angular_data["phi_p_gauge"][radial]
            positive_p_links = np.asarray(
                [
                    positive_gauges[edge].conj().T
                    @ positive_links[edge]
                    @ positive_gauges[edge + 1]
                    for edge in range(len(positive_links))
                ]
            )
            for edge, value in enumerate(positive_p_links):
                dataset.add_link(
                    (float(radius), float(phi[edge]), MID_THETA),
                    (float(radius), float(phi[edge + 1]), MID_THETA),
                    value,
                    "outer-angular",
                )
                dataset.add_link(
                    (float(radius), -float(phi[edge + 1]), MID_THETA),
                    (float(radius), -float(phi[edge]), MID_THETA),
                    REFLECTION @ value.conj().T @ REFLECTION,
                    "outer-angular-reflection",
                )
            for edge, value in enumerate(angular_data["bend_p_links"][radial]):
                dataset.add_link(
                    (float(radius), 0.0, float(angular_data["bends"][edge])),
                    (float(radius), 0.0, float(angular_data["bends"][edge + 1])),
                    value,
                    "outer-angular",
                )

    energy_holdout, link_holdout, anchor, tree = _holdouts(dataset)
    coordinates = np.asarray(dataset.coordinates)
    pairs = np.asarray(dataset.pairs, dtype=int)
    deltas = coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]]
    axes = np.argmax(np.abs(deltas), axis=1)
    if np.max(np.count_nonzero(np.abs(deltas) > 1.0e-9, axis=1)) != 1:
        raise RuntimeError("all overlap samples must be coordinate-axis edges")
    return dataset, gauges, energy_holdout, link_holdout, anchor, tree, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inner", type=Path,
        default=Path("/private/tmp/phenol_sa6_3d_p_gauge_20260820/phenol_sa6_3d_p_gauge_data.npz"),
    )
    parser.add_argument(
        "--planar", type=Path,
        default=Path("/private/tmp/phenol_sa6_p_gauge_20260820/phenol_sa6_tracked3_p_gauge.npz"),
    )
    parser.add_argument(
        "--bridge", type=Path,
        default=Path("/private/tmp/phenol_sa6_bridge_20260820/phenol_sa6_bridge_p_gauge.npz"),
    )
    parser.add_argument(
        "--bridge-history", type=Path, nargs="*", default=[],
        help="earlier local bridge artifacts needed to cover the combined backbone",
    )
    parser.add_argument(
        "--angular", type=Path, nargs="+",
        default=[Path("/private/tmp/phenol_sa6_outer_angular_20260820/phenol_sa6_outer_angular_p_gauge.npz")],
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/phenol_sa6_mace_dataset_20260820"),
    )
    parser.add_argument(
        "--minimum-radius", type=float, default=0.95,
        help="retain already-computed planar backbone points down to this R_OH",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {name: _load(getattr(args, name)) for name in ("inner", "planar", "bridge")}
    inputs["bridge_history"] = tuple(_load(path) for path in args.bridge_history)
    inputs["angular"] = tuple(_load(path) for path in args.angular)
    dataset, gauges, energy_holdout, link_holdout, anchor, tree, axes = assemble(
        **inputs, minimum_radius=args.minimum_radius
    )
    coordinates = np.asarray(dataset.coordinates)
    hamiltonians = np.asarray(dataset.hamiltonians)
    pairs = np.asarray(dataset.pairs, dtype=int)
    links = np.asarray(dataset.links)
    data_path = args.output / "phenol_sa6_3d_mace_y.npz"
    np.savez_compressed(
        data_path,
        coordinates=coordinates,
        p_hamiltonian=hamiltonians,
        energy_sources=np.asarray(dataset.energy_sources),
        pairs=pairs,
        p_links=links,
        link_sources=np.asarray(dataset.link_sources),
        pair_axes=axes,
        energy_holdout=energy_holdout,
        link_holdout=link_holdout,
        spanning_tree=tree,
        anchor=np.asarray(anchor),
        reflection=REFLECTION,
        inner_regauged_gauges=gauges,
    )
    singular = np.linalg.svd(links, compute_uv=False)
    summary = {
        "passed": bool(
            dataset.maximum_duplicate_energy_defect <= 1.0e-7
            and dataset.maximum_duplicate_link_defect <= 2.0e-7
            and np.min(singular) >= 0.65
        ),
        "points": len(coordinates),
        "links": len(pairs),
        "energy_holdouts": int(np.count_nonzero(energy_holdout)),
        "link_holdouts": int(np.count_nonzero(link_holdout)),
        "radial_range_angstrom": [float(coordinates[:, 0].min()), float(coordinates[:, 0].max())],
        "torsion_range_radian": [float(coordinates[:, 1].min()), float(coordinates[:, 1].max())],
        "bend_range_degree": np.rad2deg([coordinates[:, 2].min(), coordinates[:, 2].max()]).tolist(),
        "minimum_link_singular_value": float(np.min(singular)),
        "maximum_link_singular_value": float(np.max(singular)),
        "maximum_duplicate_energy_defect_hartree": dataset.maximum_duplicate_energy_defect,
        "maximum_duplicate_link_defect": dataset.maximum_duplicate_link_defect,
        "reflection_representation": REFLECTION.real.tolist(),
        "data": str(data_path),
        "sources": {
            **{name: str(getattr(args, name)) for name in ("inner", "planar", "bridge")},
            "bridge_history": [str(path) for path in args.bridge_history],
            "angular": [str(path) for path in args.angular],
        },
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
