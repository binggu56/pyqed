import numpy as np

from examples.ldr.so2_procrustes_gauge import (
    gauged_hamiltonian,
    rotate_kernel,
    rotate_local,
    stitch,
)


def test_procrustes_gauge_hamiltonian_is_exact_basis_rotation():
    kinetic = np.asarray([[0.7, -0.2], [-0.2, 0.9]])
    overlap = np.zeros((2, 2, 2, 2), dtype=complex)
    overlap[0, :, 0, :] = overlap[1, :, 1, :] = np.eye(2)
    link = np.asarray([[0.91, 0.12], [-0.08, 0.82]])
    overlap[0, :, 1, :] = link
    overlap[1, :, 0, :] = link.conj().T
    energies = np.asarray([[0.1, 0.4], [0.2, 0.55]])
    angles = (0.31, -0.27)
    gauge = np.asarray(
        [
            [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
            for a in angles
        ],
        dtype=complex,
    )

    original = kinetic[:, None, :, None] * overlap
    shifted = energies - np.min(energies)
    for point in range(2):
        original[point, :, point, :] += np.diag(shifted[point])
    original = original.reshape(4, 4)
    original = 0.5 * (original + original.conj().T)
    aligned, aligned_overlap, local = gauged_hamiltonian(
        kinetic,
        overlap,
        energies,
        gauge,
    )
    expected = rotate_kernel(
        original.reshape(2, 2, 2, 2),
        gauge,
    ).reshape(4, 4)

    np.testing.assert_allclose(aligned, expected, atol=1.0e-13)
    np.testing.assert_allclose(
        aligned_overlap[0, :, 1, :],
        gauge[0].conj().T @ link @ gauge[1],
    )
    np.testing.assert_allclose(
        local,
        rotate_local(
            np.asarray([np.diag(row - np.min(energies)) for row in energies]),
            gauge,
        ),
    )


def test_stitch_removes_boundary_link_rotation_on_each_transverse_line():
    shape = (2, 2)
    primary = np.broadcast_to(np.eye(2), (*shape, 2, 2)).copy()
    secondary = np.broadcast_to(np.eye(2), (*shape, 2, 2)).copy()
    angles = (0.2, -0.35)
    links = {}
    for transverse, angle in enumerate(angles):
        links[(1, (transverse, 0))] = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )

    combined, transition = stitch(
        shape,
        links,
        primary,
        secondary,
        axis=1,
        boundary=0,
    )

    for transverse in range(2):
        aligned = (
            combined[transverse, 0].conj().T
            @ links[(1, (transverse, 0))]
            @ combined[transverse, 1]
        )
        np.testing.assert_allclose(aligned, np.eye(2), atol=1.0e-13)
        np.testing.assert_allclose(
            combined[transverse, 0],
            transition[transverse],
        )
