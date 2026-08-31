import numpy as np

from examples.namd.h3plus_casci_positive_link_fit import (
    anchor_procrustes_fields,
    interior_holdout,
    plaquette_defects,
)


def test_anchor_procrustes_makes_reference_links_positive():
    theta = np.asarray([[0.2, -0.4], [0.7, 0.1]])
    rotations = np.empty((2, 2, 2, 2), dtype=complex)
    for index in np.ndindex(theta.shape):
        angle = theta[index]
        rotations[index] = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    positive = np.broadcast_to(np.diag([0.96, 0.91]), rotations.shape)
    reference = rotations @ positive
    identity = np.broadcast_to(np.eye(6), (2, 2, 6, 6)).copy().astype(complex)
    identity[..., 1:3, 1:3] = reference
    raw_x = np.einsum(
        "...ai,...bi->...ab", rotations[:-1], rotations[1:].conj(),
    )
    raw_y = np.einsum(
        "...ai,...bi->...ab", rotations[:, :-1], rotations[:, 1:].conj(),
    )
    links_x = np.broadcast_to(np.eye(6), (1, 2, 6, 6)).copy().astype(complex)
    links_y = np.broadcast_to(np.eye(6), (2, 1, 6, 6)).copy().astype(complex)
    links_x[..., 1:3, 1:3] = raw_x
    links_y[..., 1:3, 1:3] = raw_y
    energies = np.broadcast_to(np.arange(6.0), (2, 2, 6)).copy()
    data = {
        "energies": energies,
        "reference_links": identity,
        "links_x": links_x,
        "links_y": links_y,
        "anchor": np.asarray((0, 0)),
    }
    fields = anchor_procrustes_fields(data)
    reconstructed = np.einsum(
        "...ia,...ij->...aj",
        fields["gauges"].conj(),
        reference,
    )
    np.testing.assert_allclose(reconstructed, positive, atol=1.0e-13)


def test_interior_holdout_excludes_boundaries():
    mask = interior_holdout((7, 6))
    assert np.any(mask)
    assert not np.any(mask[[0, -1]])
    assert not np.any(mask[:, [0, -1]])


def test_identity_links_have_zero_plaquette_defect():
    links_x = np.broadcast_to(np.eye(2), (3, 4, 2, 2)).copy()
    links_y = np.broadcast_to(np.eye(2), (4, 3, 2, 2)).copy()
    defects, phases = plaquette_defects(links_x, links_y)
    np.testing.assert_allclose(defects, 0.0)
    np.testing.assert_allclose(phases, 0.0)
