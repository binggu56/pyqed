import numpy as np
from scipy.linalg import expm

from examples.namd.phenol_sa_casscf_5d_quasibound import (
    add_constant_to_tt,
    discrete_tt_values,
    fit_discrete_tt,
    load_interpolated_state,
    mps_to_dense,
    mpo_diagonal_values,
    product_gaussian,
    project_electronic_component,
    reflection_pair,
    sample_mps_indices,
    support_plateau_windows,
    symmetrize_state,
)
from pyqed.mps.cross import tt_value
from pyqed.mps import MPS, MPO
from pyqed.mps.tdvp import one_site_tdvp_sum_step
from pyqed.dvr import ExponentialDVR
from pyqed.namd.phenol import _reflection_indices


def test_project_electronic_component_matches_dense_diagonal_block():
    rng = np.random.default_rng(91)
    left = rng.normal(size=(1, 3, 2, 2))
    middle = rng.normal(size=(3, 4, 3, 3))
    electronic = rng.normal(size=(4, 1, 2, 2))
    vibronic = MPO([left, middle, electronic])

    projected = project_electronic_component(vibronic.factors, state=1)
    dense = vibronic.to_dense().reshape(2 * 3, 2, 2 * 3, 2)

    np.testing.assert_allclose(projected.to_dense(), dense[:, 1, :, 1])


def test_product_gaussian_dense_shape_and_norm():
    axes = (np.linspace(-1.0, 1.0, 5), np.linspace(-0.4, 0.4, 3))
    state = product_gaussian(axes, center=(0.1, -0.1), widths=(0.3, 0.2))
    dense = mps_to_dense(state)

    assert dense.shape == (5, 3)
    np.testing.assert_allclose(np.linalg.norm(dense), 1.0, atol=1.0e-14)


def test_add_constant_to_tt_is_exact():
    rng = np.random.default_rng(43)
    cores = [
        rng.normal(size=(1, 4, 2)),
        rng.normal(size=(2, 3, 3)),
        rng.normal(size=(3, 5, 1)),
    ]
    shifted = add_constant_to_tt(cores, -7.25)

    for index in np.ndindex(4, 3, 5):
        np.testing.assert_allclose(
            tt_value(shifted, index), tt_value(cores, index) - 7.25
        )


def test_reflection_pair_is_exactly_even_on_the_diagonal():
    rng = np.random.default_rng(17)
    potential = MPO(
        [
            rng.normal(size=(1, 3, 4, 4)),
            rng.normal(size=(3, 2, 5, 5)),
            rng.normal(size=(2, 1, 3, 3)),
        ]
    )
    components = reflection_pair(potential, reflection_sites=(1,))
    indices = np.asarray(list(np.ndindex(4, 5, 3)))
    reflected = indices.copy()
    reflected[:, 1] = 4 - reflected[:, 1]

    np.testing.assert_allclose(
        mpo_diagonal_values(components, indices),
        mpo_diagonal_values(components, reflected),
        atol=1.0e-14,
    )


def test_periodic_reflection_pair_closes_across_the_torsional_seam():
    rng = np.random.default_rng(18)
    dvr = ExponentialDVR(npts=7, L=2.0 * np.pi)
    mapping = _reflection_indices(dvr)
    potential = MPO([rng.normal(size=(1, 1, 7, 7))])
    components = reflection_pair(
        potential, reflection_sites=(0,), reflection_maps={0: mapping}
    )
    indices = np.arange(7)[:, None]

    np.testing.assert_array_equal(mapping[mapping], np.arange(7))
    np.testing.assert_allclose(
        mpo_diagonal_values(components, indices),
        mpo_diagonal_values(components, mapping[:, None]),
        atol=1.0e-14,
    )


def test_mps_born_sampler_and_even_projection():
    first = np.sqrt(np.asarray((0.15, 0.35, 0.50)))
    second = np.sqrt(np.asarray((0.8, 0.2)))
    product = MPS([first[None, :, None], second[None, :, None]])
    samples = sample_mps_indices(product, 30000, seed=8)

    np.testing.assert_allclose(
        np.bincount(samples[:, 0], minlength=3) / len(samples),
        first**2,
        atol=1.0e-2,
    )
    projected = symmetrize_state(product, reflection_sites=(0,))
    dense = mps_to_dense(projected)
    np.testing.assert_allclose(dense, dense[::-1], atol=1.0e-14)


def test_support_plateau_windows_are_even_and_zero_at_remote_boundaries():
    indices = np.asarray(
        [[3, 4, 2], [4, 5, 3], [5, 6, 4], [4, 4, 3]] * 32
    )
    windows, plateaus = support_plateau_windows(
        indices, (11, 11, 9), quantile=0.99, reflection_sites=(1,)
    )

    assert windows[0][0] == 0.0
    assert windows[0][-1] == 0.0
    assert windows[1][0] == 0.0
    assert windows[1][-1] == 0.0
    np.testing.assert_allclose(windows[1], windows[1][::-1])
    for window, (lower, upper) in zip(windows, plateaus):
        np.testing.assert_allclose(window[lower : upper + 1], 1.0)


def test_support_plateau_window_does_not_taper_a_periodic_axis():
    indices = np.asarray([[0, 3], [4, 4], [8, 5]] * 8)
    windows, plateaus = support_plateau_windows(
        indices, (9, 9), periodic_sites=(0,)
    )

    np.testing.assert_array_equal(windows[0], np.ones(9))
    assert plateaus[0] == (0, 8)


def test_discrete_tt_als_recovers_sampled_rank_two_tensor():
    rng = np.random.default_rng(72)
    exact = [
        rng.normal(size=(1, 4, 2)),
        rng.normal(size=(2, 5, 2)),
        rng.normal(size=(2, 3, 1)),
    ]
    indices = np.asarray(list(np.ndindex(4, 5, 3)))
    values = discrete_tt_values(exact, indices)
    fitted, info = fit_discrete_tt(
        indices,
        values,
        (4, 5, 3),
        rank=2,
        sweeps=8,
        regularization=1.0e-10,
        seed=4,
        validation=(indices, values),
    )

    np.testing.assert_allclose(discrete_tt_values(fitted, indices), values, atol=1e-7)
    assert info["best_validation_rms_hartree"] < 1.0e-8


def test_compiled_sum_tdvp_supports_imaginary_time_projection():
    first_matrix = np.asarray([[0.4, -0.12], [-0.12, 0.9]])
    second_matrix = np.asarray([[0.1, 0.03], [0.03, -0.2]])
    operators = (
        MPO([first_matrix[None, None, :, :]]),
        MPO([second_matrix[None, None, :, :]]),
    )
    vector = np.asarray([0.8, 0.6j], dtype=complex)
    vector /= np.linalg.norm(vector)
    state = MPS([vector[None, :, None]])
    tau = 0.37

    projected, info = one_site_tdvp_sum_step(
        state,
        operators,
        tau,
        imaginary_time=True,
        return_info=True,
    )
    expected = expm(-tau * (first_matrix + second_matrix)) @ vector
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(mps_to_dense(projected), expected, atol=1.0e-12)
    assert info["imaginary_time"]


def test_saved_mps_interpolation_preserves_shape_and_normalization(tmp_path):
    old_axes = (np.linspace(-1.0, 1.0, 5), np.linspace(-0.5, 0.5, 4))
    source = product_gaussian(old_axes, center=(0.1, -0.05), widths=(0.3, 0.2))
    path = tmp_path / "state.npz"
    np.savez(
        path,
        axis_0=old_axes[0],
        axis_1=old_axes[1],
        mps_factor_0=source._get_std_B(0),
        mps_factor_1=source._get_std_B(1),
    )
    new_axes = (np.linspace(-1.5, 1.5, 9), np.linspace(-0.8, 0.8, 7))

    interpolated = load_interpolated_state(path, new_axes, None)

    assert tuple(interpolated.dims) == tuple(map(len, new_axes))
    np.testing.assert_allclose(np.linalg.norm(mps_to_dense(interpolated)), 1.0)
    assert np.allclose(mps_to_dense(interpolated)[[0, -1]], 0.0)


def test_axis_free_checkpoint_loads_on_its_original_grid(tmp_path):
    axes = (np.linspace(-1.0, 1.0, 5), np.linspace(-0.5, 0.5, 4))
    source = product_gaussian(axes, center=(0.1, -0.05), widths=(0.3, 0.2))
    path = tmp_path / "checkpoint.npz"
    np.savez(
        path,
        factor_count=np.asarray(2),
        factor_0=source._get_std_B(0),
        factor_1=source._get_std_B(1),
    )

    restored = load_interpolated_state(path, axes, None)

    np.testing.assert_allclose(
        abs(np.vdot(mps_to_dense(source), mps_to_dense(restored))), 1.0
    )
