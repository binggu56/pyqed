import numpy as np

from examples.namd.phenol_sa_casscf_3d_mace_y import reflection_group
from examples.namd.phenol_sa_casscf_mace_dataset import (
    MID_THETA,
    REFLECTION,
    Dataset,
    _backward_gauge,
    _forward_gauge,
    _holdouts,
    radial_backbone,
)


def test_forward_and_backward_gauges_make_transport_positive():
    rng = np.random.default_rng(7)
    left, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    right, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    singular = np.diag((0.98, 0.91, 0.73))
    overlap = left @ singular @ right.T
    anchor, _ = np.linalg.qr(rng.normal(size=(3, 3)))

    transported_right = _forward_gauge(anchor, overlap)
    forward = anchor.conj().T @ overlap @ transported_right
    assert np.allclose(forward, forward.conj().T, atol=1.0e-12)
    assert np.min(np.linalg.eigvalsh(forward)) > 0.0

    transported_left = _backward_gauge(overlap, anchor)
    backward = transported_left.conj().T @ overlap @ anchor
    assert np.allclose(backward, backward.conj().T, atol=1.0e-12)
    assert np.min(np.linalg.eigvalsh(backward)) > 0.0


def test_reflection_group_intertwines_the_canonical_anchor():
    group = reflection_group(12, REFLECTION)
    coordinate = group["coordinate_representations"]
    electronic = group["electronic_representations"]
    ambient = group["ambient_representations"]
    frame = np.zeros((12, 3), dtype=complex)
    frame[:3] = np.eye(3)

    assert np.allclose(coordinate[1] @ coordinate[1], np.eye(3))
    assert np.allclose(electronic[1] @ electronic[1], np.eye(3))
    assert np.allclose(ambient[1] @ ambient[1], np.eye(12))
    assert np.allclose(ambient[1] @ frame @ electronic[1], frame)


def test_stratified_holdouts_keep_training_graph_connected():
    dataset = Dataset()
    radial = (0.95, 1.0, 1.05)
    torsion = (-0.2, 0.0, 0.2)
    bend = (MID_THETA - 0.1, MID_THETA, MID_THETA + 0.1)
    for r in radial:
        for p in torsion:
            for t in bend:
                dataset.add_energy((r, p, t), np.eye(3), "inner-3d")
    for ir, r in enumerate(radial):
        for ip, p in enumerate(torsion):
            for it, t in enumerate(bend):
                if ir + 1 < len(radial):
                    dataset.add_link((r, p, t), (radial[ir + 1], p, t), np.eye(3), "inner-3d")
                if ip + 1 < len(torsion):
                    dataset.add_link((r, p, t), (r, torsion[ip + 1], t), np.eye(3), "inner-3d")
                if it + 1 < len(bend):
                    dataset.add_link((r, p, t), (r, p, bend[it + 1]), np.eye(3), "inner-3d")

    _energy, link_holdout, _anchor, _tree = _holdouts(dataset, seed=19)
    coordinates = np.asarray(dataset.coordinates)
    pairs = np.asarray(dataset.pairs)
    deltas = coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]]
    axes = np.argmax(np.abs(deltas), axis=1)
    assert np.all(np.bincount(axes[link_holdout], minlength=3) > 0)

    adjacency = [[] for _ in coordinates]
    for left, right in pairs[~link_holdout]:
        adjacency[left].append(right)
        adjacency[right].append(left)
    reached = {0}
    frontier = [0]
    while frontier:
        for neighbor in adjacency[frontier.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    assert len(reached) == len(coordinates)


def test_radial_backbone_prepends_only_new_inward_planar_points():
    identity = np.eye(3)
    planar = {
        "radii": np.asarray((0.75, 0.80, 0.85, 0.90, 0.95)),
        "p_hamiltonian": np.asarray([value * identity for value in range(5)]),
    }
    bridge = {
        "combined_radii": np.asarray((0.90, 0.95, 1.00)),
        "combined_p_hamiltonian": np.asarray(
            [(10 + value) * identity for value in range(3)]
        ),
    }

    radii, hamiltonian = radial_backbone(planar, bridge, 0.75)

    np.testing.assert_allclose(radii, (0.75, 0.80, 0.85, 0.90, 0.95, 1.00))
    np.testing.assert_allclose(
        np.trace(hamiltonian, axis1=1, axis2=2) / 3.0,
        (0.0, 1.0, 2.0, 10.0, 11.0, 12.0),
    )


def test_radial_backbone_uses_dense_planar_points_up_to_continuity_break():
    identity = np.eye(3)
    planar = {
        "radii": np.asarray((1.70, 1.80, 1.825, 1.85, 1.875, 1.90)),
        "p_hamiltonian": np.asarray([value * identity for value in range(6)]),
        "tracked_singular_values": np.asarray(
            [
                (0.99, 0.98, 0.97),
                (0.99, 0.98, 0.97),
                (0.99, 0.98, 0.97),
                (0.99, 0.98, 0.97),
                (0.80, 0.70, 0.60),
            ]
        ),
    }
    bridge = {
        "combined_radii": np.asarray((1.70, 1.85, 1.875, 1.90, 2.00)),
        "combined_p_hamiltonian": np.asarray(
            [(10 + value) * identity for value in range(5)]
        ),
    }

    radii, hamiltonian = radial_backbone(planar, bridge, 1.70)

    np.testing.assert_allclose(
        radii, (1.70, 1.80, 1.825, 1.85, 1.875, 1.90, 2.00)
    )
    np.testing.assert_allclose(
        np.trace(hamiltonian, axis1=1, axis2=2) / 3.0,
        (0.0, 1.0, 2.0, 3.0, 4.0, 13.0, 14.0),
    )
