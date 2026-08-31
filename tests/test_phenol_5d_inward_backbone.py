import numpy as np

from examples.namd.phenol_sa_casscf_5d_inward_backbone import (
    augment_inward_backbone,
)


def test_augment_inward_backbone_reuses_short_range_chain():
    theta = np.deg2rad(108.8)
    reflection = np.diag((1.0, 1.0, -1.0))
    coordinates = np.asarray(
        ((0.95, 0.0, theta, 0.0, 0.0), (1.15, 0.0, theta, 0.0, 0.0))
    )
    base_h = np.asarray(
        (np.diag((-1.0, 0.2, 0.5)), np.diag((-0.9, 0.25, 0.55))),
        dtype=complex,
    )
    base = {
        "coordinates": coordinates,
        "p_hamiltonian": base_h,
        "gauges": np.asarray((np.eye(3), np.eye(3))),
        "pairs": np.asarray(((0, 1),)),
        "p_links": np.asarray((0.98 * np.eye(3),)),
        "pair_axes": np.asarray((0,)),
        "energy_holdout": np.asarray((False, True)),
        "link_holdout": np.asarray((True,)),
        "reflection": reflection,
        "coordinate_parities": np.asarray((1.0, -1.0, 1.0, -1.0, 1.0)),
        "coordinate_scales": np.ones(5),
        "modes": np.zeros((2, 4, 3)),
    }
    radii = np.asarray((0.75, 0.95, 1.00, 1.15))
    sign = np.diag((1.0, -1.0, 1.0))
    source_h = np.asarray(
        (
            np.diag((-0.7, 0.35, 0.65)),
            sign @ base_h[0] @ sign,
            np.diag((-0.97, 0.22, 0.52)),
            sign @ base_h[1] @ sign,
        )
    )
    inward = {
        "radii": radii,
        "p_hamiltonian": source_h,
        "p_links": np.asarray((0.99 * np.eye(3),) * 3),
        "p_gauge": np.asarray((sign,) * len(radii)),
    }

    artifact, summary = augment_inward_backbone(base, inward)

    assert summary["passed"]
    np.testing.assert_allclose(summary["appended_radii_angstrom"], (0.75, 1.0))
    assert artifact["coordinates"].shape == (4, 5)
    assert artifact["pairs"].shape == (4, 2)
    assert artifact["p_links"].shape == (4, 3, 3)
    assert artifact["pair_axes"].tolist() == [0, 0, 0, 0]
    assert artifact["energy_holdout"].tolist() == [False, True, False, False]
    assert artifact["link_holdout"].tolist() == [True, False, False, False]
    np.testing.assert_allclose(
        np.linalg.eigvalsh(artifact["p_hamiltonian"][2]),
        np.linalg.eigvalsh(source_h[0]),
    )
