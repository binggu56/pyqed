import numpy as np

from pyqed.ldr import OverlapBasis, project_basis, sync_gauge


def test_overlap_basis_matches_dense_projected_kinetic_action():
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(4, 2, 5)) + 1j * rng.normal(size=(4, 2, 5))
    for point in range(4):
        q, _ = np.linalg.qr(raw[point].T)
        raw[point] = q.T
    blocks = np.einsum('pia,qja->piqj', raw.conj(), raw)
    basis = OverlapBasis.fit(blocks)
    kinetic = rng.normal(size=(4, 4))
    kinetic = 0.5 * (kinetic + kinetic.T)
    coefficients = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))

    dense = (
        blocks * kinetic[:, None, :, None]
    ).reshape(8, 8) @ coefficients.reshape(-1)

    np.testing.assert_allclose(basis.blocks(), blocks, atol=1e-12)
    np.testing.assert_allclose(
        basis.apply_kinetic(kinetic, coefficients).reshape(-1),
        dense,
        atol=1e-12,
    )


def test_sync_gauge_recovers_nonabelian_frames_from_redundant_links():
    rng = np.random.default_rng(11)
    frames = []
    for _ in range(7):
        q, r = np.linalg.qr(
            rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        )
        frames.append(q @ np.diag(np.exp(-1j * np.angle(np.diag(r)))))
    frames = np.asarray(frames)
    edges = [(i, i + 1) for i in range(6)] + [(i, i + 2) for i in range(5)]
    links = [
        (i, j, frames[i] @ frames[j].conj().T, 1.0)
        for i, j in edges
    ]

    synchronized = sync_gauge(links, len(frames), root=0)
    expected = frames @ frames[0].conj().T

    np.testing.assert_allclose(synchronized, expected, atol=1e-12)
    for i, j, overlap, _weight in links:
        np.testing.assert_allclose(
            synchronized[i].conj().T @ overlap @ synchronized[j],
            np.eye(3),
            atol=1e-12,
        )


def test_sync_gauge_robustly_downweights_inconsistent_shortcut():
    angles = np.linspace(0.0, 0.7, 6)
    frames = np.asarray([
        np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=complex,
        )
        for angle in angles
    ])
    links = [
        (i, i + 1, frames[i] @ frames[i + 1].conj().T)
        for i in range(5)
    ]
    links.extend(
        (i, i + 2, frames[i] @ frames[i + 2].conj().T)
        for i in range(4)
    )
    links.append((0, 5, -frames[0] @ frames[5].conj().T))

    synchronized = sync_gauge(
        links,
        len(frames),
        root=0,
        robust_scale=0.1,
        max_cycle=20,
    )

    expected = frames @ frames[0].conj().T
    np.testing.assert_allclose(synchronized, expected, atol=3e-2)


def test_project_basis_is_covariant_to_independent_sample_gauges():
    rng = np.random.default_rng(21)
    coordinates = np.linspace(-1.0, 1.0, 5)[:, None]
    raw = rng.normal(size=(5, 2, 7)) + 1j * rng.normal(size=(5, 2, 7))
    for point in range(5):
        q, _ = np.linalg.qr(raw[point].T)
        raw[point] = q.T
    blocks = np.einsum("pia,qja->piqj", raw.conj(), raw)
    basis = OverlapBasis.fit(blocks)
    energies = np.column_stack(
        (0.2 * coordinates[:, 0] ** 2, 0.5 + 0.1 * coordinates[:, 0])
    )
    query = np.array([[-0.7], [-0.1], [0.45]])

    reference, reference_energies = project_basis(
        basis, coordinates, energies, query, neighbors=4
    )
    rotations = []
    rotated_vectors = np.empty_like(basis.vectors)
    rotated_hamiltonians = np.empty((5, 2, 2), dtype=complex)
    for point in range(5):
        q, r = np.linalg.qr(
            rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        )
        rotation = q @ np.diag(np.exp(-1j * np.angle(np.diag(r))))
        rotations.append(rotation)
        rotated_vectors[point] = rotation.T @ basis.vectors[point]
        rotated_hamiltonians[point] = (
            rotation.conj().T @ np.diag(energies[point]) @ rotation
        )
    rotated_basis = OverlapBasis(
        rotated_vectors, basis.eigenvalues, basis.residual
    )
    transformed, transformed_energies = project_basis(
        rotated_basis,
        coordinates,
        rotated_hamiltonians,
        query,
        neighbors=4,
    )

    np.testing.assert_allclose(transformed_energies, reference_energies, atol=1e-11)
    for point in range(len(query)):
        reference_projector = (
            reference.vectors[point].T @ reference.vectors[point].conj()
        )
        transformed_projector = (
            transformed.vectors[point].T @ transformed.vectors[point].conj()
        )
        np.testing.assert_allclose(
            transformed_projector, reference_projector, atol=1e-11
        )
