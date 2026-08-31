import numpy as np

from pyqed.ml import (
    CorrectedMatrixField,
    RadialMatrixCorrection,
    ReflectionScalarMLP,
)


class _ConstantField:
    output_shape_ = (2, 2)

    def predict(self, coordinates):
        return np.broadcast_to(np.eye(2), (len(coordinates), 2, 2)).copy()


def test_radial_matrix_correction_fits_group_means_and_round_trips(tmp_path):
    coordinates = np.asarray(((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (2.0, 0.0)))
    baseline = np.broadcast_to(np.eye(2), (4, 2, 2)).copy()
    residual = np.asarray(
        (
            [[1.0, 0.3], [0.3, 2.0]],
            [[3.0, -0.3], [-0.3, 4.0]],
            [[2.0, 0.4], [0.4, 3.0]],
            [[4.0, 0.8], [0.8, 5.0]],
        )
    )
    target = baseline + residual
    reflection = np.diag((1.0, -1.0))

    correction = RadialMatrixCorrection.fit(
        coordinates, target, baseline, representation=reflection
    )
    expected = np.asarray(
        ([[2.0, 0.0], [0.0, 3.0]], [[2.0, 0.0], [0.0, 3.0]], [[4.0, 0.0], [0.0, 5.0]])
    )
    np.testing.assert_allclose(
        correction.predict(np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))),
        expected,
    )
    restored = RadialMatrixCorrection.load(correction.save(tmp_path / "delta.npz"))
    np.testing.assert_allclose(restored.predict(coordinates), correction.predict(coordinates))

    corrected = CorrectedMatrixField(_ConstantField(), restored)
    np.testing.assert_allclose(
        corrected.predict(coordinates), baseline + restored.predict(coordinates)
    )


def test_reflection_scalar_mlp_is_exactly_symmetric_and_round_trips(tmp_path):
    field = ReflectionScalarMLP(
        (-2.0, -1.0),
        (2.0, 1.0),
        (1.0, -1.0),
        (
            np.asarray(((0.7, -0.4), (-0.2, 0.9))),
            np.asarray(((1.3, -0.8),)),
        ),
        (np.asarray((0.1, -0.3)), np.asarray((0.2,))),
        output_shift=-3.0,
        output_scale=0.25,
    )
    coordinates = np.asarray(((0.4, 0.7), (-1.1, -0.2)))
    reflected = coordinates * np.asarray((1.0, -1.0))
    np.testing.assert_array_equal(field.predict(coordinates), field.predict(reflected))

    restored = ReflectionScalarMLP.load(field.save(tmp_path / "scalar.npz"))
    np.testing.assert_allclose(restored.predict(coordinates), field.predict(coordinates))


def test_reflection_scalar_mlp_periodic_axis_is_seam_continuous_and_round_trips(tmp_path):
    field = ReflectionScalarMLP(
        (-2.0, -np.pi),
        (2.0, np.pi),
        (1.0, -1.0),
        (np.asarray(((0.4, 0.7, -0.3),)),),
        (np.asarray((0.2,)),),
        periodic_axes=(1,),
    )
    left = np.asarray(((0.3, -np.pi), (0.3, -np.pi + 1.0e-8)))
    right = np.asarray(((0.3, np.pi), (0.3, np.pi + 1.0e-8)))
    np.testing.assert_allclose(field.predict(left), field.predict(right), atol=1.0e-14)
    np.testing.assert_array_equal(
        field.predict(left), field.predict(left * np.asarray((1.0, -1.0)))
    )

    restored = ReflectionScalarMLP.load(field.save(tmp_path / "periodic.npz"))
    assert restored.periodic_axes == (1,)
    np.testing.assert_allclose(restored.predict(left), field.predict(left))


def test_reflection_scalar_mlp_supports_multiple_periodic_harmonics(tmp_path):
    field = ReflectionScalarMLP(
        (-np.pi,),
        (np.pi,),
        (-1.0,),
        (np.asarray(((0.2, -0.3, 0.4, 0.1, -0.2, 0.5),)),),
        (np.asarray((0.1,)),),
        periodic_axes=(0,),
        periodic_harmonics=3,
    )
    coordinates = np.linspace(-np.pi, np.pi, 9)[:, None]
    shifted = coordinates + 2.0 * np.pi
    np.testing.assert_allclose(
        field.predict(coordinates), field.predict(shifted), atol=1.0e-14
    )

    restored = ReflectionScalarMLP.load(field.save(tmp_path / "harmonics.npz"))
    assert restored.periodic_harmonics == 3
    np.testing.assert_allclose(restored.predict(coordinates), field.predict(coordinates))
