import numpy as np

from pyqed.ml.nn import EquivariantMLP, H3PES, MLP, MPNN, grid_to_samples


def test_grid_to_samples_flattens_multistate_pes():
    x_axis = np.array([-1.0, 0.0, 1.0])
    y_axis = np.array([0.0, 2.0])
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
    values = np.stack((xx**2 + yy, xx - yy), axis=-1)

    coords, energies = grid_to_samples((x_axis, y_axis), values)

    assert coords.shape == (6, 2)
    assert energies.shape == (6, 2)
    np.testing.assert_allclose(coords[0], [-1.0, 0.0])
    np.testing.assert_allclose(energies[0], [1.0, -1.0])


def test_mlp_fits_smooth_surface():
    x_axis = np.linspace(-1.0, 1.0, 9)
    y_axis = np.linspace(-1.0, 1.0, 9)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
    values = np.sin(xx) + 0.5 * yy**2
    coords, energies = grid_to_samples((x_axis, y_axis), values)

    model = MLP(
        hidden_layers=(24, 24),
        learning_rate=0.01,
        batch_size=None,
        max_iter=1200,
        patience=300,
        random_state=7,
    ).fit(coords, energies)

    probe = np.array([[-0.35, 0.25], [0.4, -0.5]])
    expected = np.sin(probe[:, 0]) + 0.5 * probe[:, 1] ** 2
    predicted = model.predict(probe)

    np.testing.assert_allclose(predicted, expected, atol=4e-2)


def test_mlp_save_load_roundtrip(tmp_path):
    coords = np.linspace(-1.0, 1.0, 15)[:, None]
    energies = coords[:, 0] ** 2
    model = MLP(
        hidden_layers=(8,),
        learning_rate=0.01,
        batch_size=None,
        max_iter=400,
        random_state=4,
    ).fit(coords, energies)

    filename = tmp_path / "pes_ann.npz"
    model.save(filename)
    loaded = MLP.load(filename)

    probe = np.array([[-0.2], [0.3]])
    np.testing.assert_allclose(loaded.predict(probe), model.predict(probe))


def test_equivariant_mlp_descriptors_are_invariant():
    geom = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.2, 0.8, 0.1],
        ]
    )
    theta = 0.4
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    shifted_rotated = geom @ rotation.T + np.array([2.0, -0.3, 0.4])
    permuted = geom[[2, 1, 0]]

    model = EquivariantMLP(
        species=("H", "H", "H"),
        radial_centers=np.linspace(0.5, 1.5, 4),
        angle_order=2,
    )

    ref = model.describe(geom)
    np.testing.assert_allclose(model.describe(shifted_rotated), ref, atol=1e-12)
    np.testing.assert_allclose(model.describe(permuted), ref, atol=1e-12)


def test_equivariant_mlp_energy_and_forces_respect_symmetry():
    geoms = np.array(
        [
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.2, 0.8, 0.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 0.7, 0.2]],
            [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.3, 0.9, 0.1]],
            [[0.0, 0.0, 0.0], [1.0, -0.1, 0.2], [0.2, 0.75, -0.1]],
        ]
    )

    def symmetric_energy(xyz):
        d01 = np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1)
        d02 = np.linalg.norm(xyz[:, 0] - xyz[:, 2], axis=1)
        d12 = np.linalg.norm(xyz[:, 1] - xyz[:, 2], axis=1)
        return d01**2 + d02**2 + d12**2

    model = EquivariantMLP(
        species=("H", "H", "H"),
        n_radial=4,
        angle_order=1,
        hidden_layers=(8,),
        learning_rate=0.01,
        batch_size=None,
        max_iter=20,
        random_state=3,
    ).fit(geoms, symmetric_energy(geoms))

    geom = geoms[0]
    theta = 0.3
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = geom @ rotation.T + np.array([0.2, -0.4, 0.7])
    permuted = geom[[2, 1, 0]]

    np.testing.assert_allclose(model.energy(transformed), model.energy(geom), atol=1e-10)
    np.testing.assert_allclose(model.energy(permuted), model.energy(geom), atol=1e-10)

    forces = model.forces(geom)
    transformed_forces = model.forces(transformed)
    np.testing.assert_allclose(transformed_forces, forces @ rotation.T, atol=1e-5)


def test_equivariant_mlp_uses_autodiff_for_equivariant_forces():
    geoms = np.array(
        [
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.2, 0.8, 0.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 0.7, 0.2]],
            [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.3, 0.9, 0.1]],
            [[0.0, 0.0, 0.0], [1.0, -0.1, 0.2], [0.2, 0.75, -0.1]],
        ]
    )

    def symmetric_energy(xyz):
        d01 = np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1)
        d02 = np.linalg.norm(xyz[:, 0] - xyz[:, 2], axis=1)
        d12 = np.linalg.norm(xyz[:, 1] - xyz[:, 2], axis=1)
        return d01**2 + d02**2 + d12**2

    model = EquivariantMLP(
        species=("H", "H", "H"),
        n_radial=4,
        angle_order=1,
        hidden_layers=(8,),
        learning_rate=0.01,
        batch_size=None,
        max_iter=4,
        random_state=3,
    ).fit(geoms, symmetric_energy(geoms))

    geom = geoms[0]
    theta = 0.3
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = geom @ rotation.T + np.array([0.2, -0.4, 0.7])
    permuted = geom[[2, 1, 0]]

    np.testing.assert_allclose(model.energy(transformed), model.energy(geom), atol=1e-5)
    np.testing.assert_allclose(model.energy(permuted), model.energy(geom), atol=1e-5)

    forces = model.forces(geom)
    transformed_forces = model.forces(transformed)
    np.testing.assert_allclose(transformed_forces, forces @ rotation.T, atol=2e-4)


def test_equivariant_mpnn_message_passing_symmetry():
    geoms = np.array(
        [
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.2, 0.8, 0.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 0.7, 0.2]],
            [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.3, 0.9, 0.1]],
        ]
    )

    def symmetric_energy(xyz):
        d01 = np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1)
        d02 = np.linalg.norm(xyz[:, 0] - xyz[:, 2], axis=1)
        d12 = np.linalg.norm(xyz[:, 1] - xyz[:, 2], axis=1)
        return d01**2 + d02**2 + d12**2

    model = MPNN(
        species=("H", "H", "H"),
        features=4,
        n_layers=2,
        n_radial=3,
        readout_hidden=6,
        learning_rate=0.005,
        batch_size=None,
        max_iter=3,
        random_state=5,
    ).fit(geoms, symmetric_energy(geoms))

    geom = geoms[0]
    theta = 0.25
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = geom @ rotation.T + np.array([0.2, -0.4, 0.7])
    permuted = geom[[2, 1, 0]]

    np.testing.assert_allclose(model.energy(transformed), model.energy(geom), atol=1e-8)
    np.testing.assert_allclose(model.energy(permuted), model.energy(geom), atol=1e-8)

    forces = model.forces(geom)
    transformed_forces = model.forces(transformed)
    np.testing.assert_allclose(transformed_forces, forces @ rotation.T, atol=1e-6)


def test_jax_geometry_models_save_load_roundtrip(tmp_path):
    geoms = np.array(
        [
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.2, 0.8, 0.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 0.7, 0.2]],
            [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.3, 0.9, 0.1]],
        ]
    )

    def symmetric_energy(xyz):
        d01 = np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1)
        d02 = np.linalg.norm(xyz[:, 0] - xyz[:, 2], axis=1)
        d12 = np.linalg.norm(xyz[:, 1] - xyz[:, 2], axis=1)
        return d01**2 + d02**2 + d12**2

    energies = symmetric_energy(geoms)
    probe = geoms[0]

    equivariant_mlp = EquivariantMLP(
        species=("H", "H", "H"),
        n_radial=3,
        angle_order=1,
        hidden_layers=(6,),
        batch_size=None,
        max_iter=2,
        random_state=6,
    ).fit(geoms, energies)
    mlp_path = tmp_path / "equivariant_mlp.npz"
    equivariant_mlp.save(mlp_path)
    loaded_mlp = EquivariantMLP.load(mlp_path)
    np.testing.assert_allclose(loaded_mlp.energy(probe), equivariant_mlp.energy(probe))
    np.testing.assert_allclose(loaded_mlp.forces(probe), equivariant_mlp.forces(probe))

    mpnn = MPNN(
        species=("H", "H", "H"),
        features=4,
        n_layers=1,
        n_radial=3,
        readout_hidden=5,
        batch_size=None,
        max_iter=2,
        random_state=7,
    ).fit(geoms, energies)
    mpnn_path = tmp_path / "mpnn.npz"
    mpnn.save(mpnn_path)
    loaded_mpnn = MPNN.load(mpnn_path)
    np.testing.assert_allclose(loaded_mpnn.energy(probe), mpnn.energy(probe))
    np.testing.assert_allclose(loaded_mpnn.forces(probe), mpnn.forces(probe))


def test_h3pes_symmetry_and_save_load(tmp_path):
    geoms = np.array(
        [
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.2, 0.8, 0.0]],
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 0.7, 0.2]],
            [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.3, 0.9, 0.1]],
            [[0.0, 0.0, 0.0], [1.0, -0.1, 0.2], [0.2, 0.75, -0.1]],
        ]
    )

    def symmetric_energy(xyz):
        d01 = np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1)
        d02 = np.linalg.norm(xyz[:, 0] - xyz[:, 2], axis=1)
        d12 = np.linalg.norm(xyz[:, 1] - xyz[:, 2], axis=1)
        return d01**2 + d02**2 + d12**2

    model = H3PES(
        hidden_layers=(8,),
        learning_rate=0.01,
        batch_size=None,
        max_iter=300,
        random_state=8,
    ).fit(geoms, symmetric_energy(geoms))

    geom = geoms[0]
    theta = 0.2
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = geom @ rotation.T + np.array([0.2, -0.3, 0.5])
    permuted = geom[[2, 1, 0]]
    np.testing.assert_allclose(model.energy(transformed), model.energy(geom), atol=1e-10)
    np.testing.assert_allclose(model.energy(permuted), model.energy(geom), atol=1e-10)
    np.testing.assert_allclose(model.forces(transformed), model.forces(geom) @ rotation.T, atol=1e-5)

    path = tmp_path / "h3pes.npz"
    model.save(path)
    loaded = H3PES.load(path)
    np.testing.assert_allclose(loaded.energy(geom), model.energy(geom))
    np.testing.assert_allclose(loaded.forces(geom), model.forces(geom))
