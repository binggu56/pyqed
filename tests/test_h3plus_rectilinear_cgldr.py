import numpy as np
from types import SimpleNamespace

from examples.ldr.h3plus_full_rectilinear_ldr import (
    build_sparse_hamiltonian,
    initial_wavepacket as full_initial_wavepacket,
    linked_line_overlaps,
    secondary_electronic_frames,
)
from examples.ldr.h3plus_rectilinear_cgldr import (
    h3plus_geometry,
    h3plus_rectilinear_modes,
    initial_wavepacket,
)
from pyqed.dvr import DVR


def test_h3plus_modes_are_rectilinear_orthonormal_and_vibrational():
    modes = h3plus_rectilinear_modes()

    np.testing.assert_allclose(
        np.einsum("mAx,nAx->mn", modes, modes),
        np.eye(3),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(modes.sum(axis=1), 0.0, atol=1.0e-14)

    reference = h3plus_geometry({"Qs": 0.0, "Qx": 0.0, "Qy": 0.0})
    step = 1.0e-5
    for coordinate, expected in zip(("Qs", "Qx", "Qy"), modes):
        values = {"Qs": 0.0, "Qx": 0.0, "Qy": 0.0}
        values[coordinate] = step
        displaced = h3plus_geometry(values)
        np.testing.assert_allclose(
            (displaced - reference) / step,
            expected,
            atol=1.0e-11,
        )


def test_initial_wavepacket_is_normalized_on_s2_and_centered():
    grids = (
        np.linspace(-0.4, 0.8, 64),
        np.linspace(-0.2, 0.2, 9),
        np.linspace(-0.2, 0.2, 9),
    )
    dynamics = SimpleNamespace(
        x=grids,
        state_ids=(1, 2),
        nstates=2,
    )

    packet = initial_wavepacket(dynamics)

    np.testing.assert_allclose(packet.norm_squared(), 1.0, atol=1.0e-14)
    np.testing.assert_allclose(packet.factors[0][0, :, 0], [0.0, 1.0])
    expected_centers = (-0.20, -0.015, 0.0)
    for grid, factor, expected in zip(
        grids,
        packet.factors[1:],
        expected_centers,
    ):
        probability = np.abs(factor[0, :, 0]) ** 2
        mean = np.sum(grid * probability)
        assert abs(mean - expected) < 0.015


def test_linked_line_overlaps_compose_unitary_neighbor_links():
    angles = (0.2, -0.35)
    links = np.asarray(
        [
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
            for angle in angles
        ],
        dtype=complex,
    )

    transports = linked_line_overlaps(links)

    np.testing.assert_allclose(transports[0, 2], links[0] @ links[1])
    np.testing.assert_allclose(
        transports[2, 0],
        transports[0, 2].conj().T,
    )
    for i in range(3):
        np.testing.assert_allclose(transports[i, i], np.eye(2))


def test_full_reference_hamiltonian_is_hermitian_and_has_expected_limit():
    dvr = DVR(
        domains=((-0.4, 0.8), (-0.2, 0.2), (-0.2, 0.2)),
        npts=(3, 2, 2),
        mass=(1836.0, 1836.0, 1836.0),
    )
    energies = np.zeros((*dvr.shape, 2))
    energies[..., 0] = -1.0
    energies[..., 1] = 0.5
    identity = np.eye(2, dtype=complex)
    links = (
        np.broadcast_to(identity, (2, 2, 2, 2, 2)).copy(),
        np.broadcast_to(identity, (3, 1, 2, 2, 2)).copy(),
        np.broadcast_to(identity, (3, 2, 1, 2, 2)).copy(),
    )

    hamiltonian = build_sparse_hamiltonian(dvr, energies, links)
    expected = (
        np.kron(dvr.kinetic().toarray(), np.eye(2))
        + np.diag(energies.reshape(-1))
    )

    np.testing.assert_allclose(hamiltonian.toarray(), expected)
    np.testing.assert_allclose(
        hamiltonian.toarray(),
        hamiltonian.toarray().conj().T,
    )
    np.testing.assert_allclose(
        np.linalg.norm(full_initial_wavepacket(dvr)),
        1.0,
    )

    frames = secondary_electronic_frames(dvr, links)
    np.testing.assert_allclose(
        np.einsum("...ji,...jk->...ik", frames.conj(), frames),
        np.broadcast_to(np.eye(2), frames.shape),
        atol=1.0e-14,
    )
    transported = full_initial_wavepacket(
        dvr,
        electronic_frames=frames,
    )
    center_coefficients = np.einsum(
        "...ab,...a->...b",
        frames.conj(),
        transported,
    )
    np.testing.assert_allclose(center_coefficients[..., 0], 0.0)
