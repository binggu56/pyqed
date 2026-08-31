import numpy as np
import pytest

from examples.ldr.so2_casci_cgldr import (
    REFERENCE_BOND,
    REFERENCE_BOND_WIDTH,
    REFERENCE_THETA_DEG,
    REFERENCE_THETA_WIDTH_DEG,
    SO2LinkedScan,
    SQRT2,
    active_space_gaps,
    centered_grid_window,
    casci_overlap_active,
    default_initial_packet_spec,
    initial_state,
    parse_triplet,
    qa_axis_from_scan,
    require_smooth_active_space,
    sampled_product_gaussian_support,
    so2_qa_mode,
    so2_qs_theta_body_frame,
    so2_theta_qa_modes,
    so2_theta_modes,
    symmetric_stretch_nodes,
    theta_qa_vibronic_couplings,
    theta_vibronic_couplings,
    theta_center_hamiltonian,
    theta_quadratic_derivatives,
    transformed_stretch_nodes,
)
from examples.ldr.so2_casci_cgldr_dense import (
    harmonic_matrix_extension,
    overlap_quantum_metric,
    single_anchor_quadratic,
)
from pyqed.dvr.dvr import DVR
from pyqed.dvr.dvr_1d import LegendreDVR, SineDVR
from pyqed.ldr import CGLDRElectronicData
from pyqed.mps.decompose import tt_to_tensor


class _NonunitaryOverlapSolver:
    nx = (1, 1, 3)
    nstates = 1
    overlap_links = {}

    def _linked_overlap_between(self, bra_idx, ket_idx, links, nstates):
        if tuple(bra_idx) == tuple(ket_idx):
            return np.eye(nstates, dtype=complex)
        return np.array([[0.2 * np.exp(0.3j)]], dtype=complex)


def _one_state_theta_scan():
    return SO2LinkedScan(
        solver=_NonunitaryOverlapSolver(),
        apes=np.array([[[[1.0], [0.0], [1.0]]]]),
        r1=np.array([2.7]),
        r2=np.array([2.7]),
        theta=np.array([-1.0, 0.0, 1.0]),
        meta={},
    )


def test_theta_center_hamiltonian_uses_unitary_transport():
    scan = _one_state_theta_scan()

    hamiltonian = theta_center_hamiltonian(scan, 1)

    np.testing.assert_allclose(
        hamiltonian.operators[0, 0, :, 0, 0],
        [1.0, 0.0, 1.0],
        atol=1.0e-12,
    )


def test_theta_quadratic_derivatives_do_not_shrink_pes_curvature():
    scan = _one_state_theta_scan()

    _energies, _gradients, hessians, anchors = theta_quadratic_derivatives(
        scan,
        1,
        anchor_count=3,
    )

    np.testing.assert_array_equal(anchors, [0, 1, 2])
    np.testing.assert_allclose(hessians[0, 0, 0, 0, 0, 0], 2.0, atol=1.0e-12)


def test_so2_qa_mode_is_valence_antisymmetric_stretch_derivative():
    qs = 3.8
    theta = np.deg2rad(104.0)
    step = 1.0e-5
    plus = np.asarray([coord for _symbol, coord in so2_qs_theta_body_frame(qs, theta, step)])
    minus = np.asarray([coord for _symbol, coord in so2_qs_theta_body_frame(qs, theta, -step)])

    numerical = (plus - minus) / (2.0 * step)

    np.testing.assert_allclose(numerical, so2_qa_mode(theta), atol=1.0e-10)


def test_so2_theta_modes_are_curvilinear_geometry_derivatives():
    qs = 3.8
    qa = 0.1
    theta = np.deg2rad(104.0)
    step = 1.0e-4

    def geometry(value):
        return np.asarray(
            [
                coord
                for _symbol, coord in so2_qs_theta_body_frame(
                    qs, value, qa
                )
            ]
        )

    center = geometry(theta)
    plus = geometry(theta + step)
    minus = geometry(theta - step)
    numerical_first = (plus - minus) / (2.0 * step)
    numerical_second = (plus - 2.0 * center + minus) / step**2
    tangent, curvature = so2_theta_modes(qs, qa, theta)

    np.testing.assert_allclose(numerical_first, tangent, atol=1.0e-8)
    np.testing.assert_allclose(numerical_second, curvature, atol=2.0e-8)


def test_so2_theta_qa_modes_include_mixed_geometry_curvature():
    qs = 3.8
    qa = 0.1
    theta = np.deg2rad(104.0)
    step = 1.0e-4

    def geometry(angle, antisymmetric):
        return np.asarray([
            coord
            for _symbol, coord in so2_qs_theta_body_frame(
                qs, angle, antisymmetric
            )
        ])

    first, curvature = so2_theta_qa_modes(qs, qa, theta)
    mixed = (
        geometry(theta + step, qa + step)
        - geometry(theta + step, qa - step)
        - geometry(theta - step, qa + step)
        + geometry(theta - step, qa - step)
    ) / (4.0 * step**2)

    np.testing.assert_allclose(first[0], so2_theta_modes(qs, qa, theta)[0])
    np.testing.assert_allclose(first[1], so2_qa_mode(theta))
    np.testing.assert_allclose(curvature[0, 1], mixed, atol=1.0e-8)
    np.testing.assert_allclose(curvature[1, 0], mixed, atol=1.0e-8)
    np.testing.assert_allclose(curvature[1, 1], 0.0)


def test_theta_vibronic_hessian_includes_coordinate_curvature_term():
    class Point:
        def vibronic_couplings(self, **kwargs):
            assert kwargs["modes"].shape == (2, 3, 3)
            assert kwargs["moving_basis"] == "rhf-relaxed-pt"
            assert kwargs["backend"] == "native"
            first = np.array([[[2.0, 3.0]]])
            second = np.zeros((1, 1, 2, 2))
            second[..., 0, 0] = 5.0
            return first, second

    first, second = theta_vibronic_couplings(
        Point(), (0,), 3.8, 0.1, np.deg2rad(104.0)
    )

    np.testing.assert_allclose(first, [[2.0]])
    np.testing.assert_allclose(second, [[8.0]])


def test_theta_qa_vibronic_hessian_includes_both_curvature_terms():
    class Point:
        def vibronic_couplings(self, **kwargs):
            assert kwargs["modes"].shape == (4, 3, 3)
            first = np.array([[[2.0, 3.0, 5.0, 7.0]]])
            second = np.zeros((1, 1, 4, 4))
            second[..., 0, 0] = 11.0
            second[..., 0, 1] = second[..., 1, 0] = 13.0
            second[..., 1, 1] = 17.0
            return first, second

    first, second = theta_qa_vibronic_couplings(
        Point(), (0,), 3.8, 0.1, np.deg2rad(104.0)
    )

    np.testing.assert_allclose(first, [[[2.0, 3.0]]])
    np.testing.assert_allclose(second[..., 0, 0], [[16.0]])
    np.testing.assert_allclose(second[..., 0, 1], [[20.0]])
    np.testing.assert_allclose(second[..., 1, 0], [[20.0]])
    np.testing.assert_allclose(second[..., 1, 1], [[17.0]])


def test_casci_sampled_overlap_preserves_nonunitary_part_by_default():
    class Point:
        def overlap(self, other):
            return np.diag([0.5, 0.25])

    raw = casci_overlap_active(Point(), Point(), (0, 1))
    polar = casci_overlap_active(Point(), Point(), (0, 1), polar=True)

    np.testing.assert_allclose(raw, np.diag([0.5, 0.25]))
    np.testing.assert_allclose(polar, np.eye(2))


def test_relaxed_fg_rejects_active_space_boundary_crossing():
    class MeanField:
        mo_energy = np.array([-1.0, -1.0 + 5.0e-5, -0.2, 0.3])

    class Point:
        mf = MeanField()
        ncore = 1
        ncas = 2

    np.testing.assert_allclose(active_space_gaps(Point()), (5.0e-5, 0.5))
    with pytest.raises(ValueError, match="core-active orbital gap"):
        require_smooth_active_space(Point())


def test_qa_default_initial_packet_matches_valence_reference_widths():
    center_text, width_text, angle_indices = default_initial_packet_spec("qa")

    center = parse_triplet(center_text, degree_indices=angle_indices)
    width = parse_triplet(width_text, degree_indices=angle_indices)

    np.testing.assert_allclose(
        center,
        (SQRT2 * REFERENCE_BOND, np.deg2rad(REFERENCE_THETA_DEG), 0.0),
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        width,
        (
            REFERENCE_BOND_WIDTH,
            np.deg2rad(REFERENCE_THETA_WIDTH_DEG),
            REFERENCE_BOND_WIDTH,
        ),
        atol=1.0e-15,
    )


def test_theta_qa_default_packet_uses_qs_theta_qa_order():
    center_text, width_text, angle_indices = default_initial_packet_spec(
        "theta-qa"
    )

    center = parse_triplet(center_text, degree_indices=angle_indices)
    width = parse_triplet(width_text, degree_indices=angle_indices)

    np.testing.assert_allclose(
        center,
        (SQRT2 * REFERENCE_BOND, np.deg2rad(REFERENCE_THETA_DEG), 0.0),
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        width,
        (
            REFERENCE_BOND_WIDTH,
            np.deg2rad(REFERENCE_THETA_WIDTH_DEG),
            REFERENCE_BOND_WIDTH,
        ),
        atol=1.0e-15,
    )


def test_theta_packet_uses_qs_qa_center_for_transformed_scan():
    center_text, width_text, angle_indices = default_initial_packet_spec(
        "theta",
        coordinates="qs-qa-theta",
    )

    center = parse_triplet(center_text, degree_indices=angle_indices)
    width = parse_triplet(width_text, degree_indices=angle_indices)

    np.testing.assert_allclose(
        center,
        (SQRT2 * REFERENCE_BOND, 0.0, np.deg2rad(REFERENCE_THETA_DEG)),
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        width,
        (
            REFERENCE_BOND_WIDTH,
            REFERENCE_BOND_WIDTH,
            np.deg2rad(REFERENCE_THETA_WIDTH_DEG),
        ),
        atol=1.0e-15,
    )


def test_cgldr_initial_packet_matches_quadrature_weighted_physical_gaussian():
    axes = (
        SineDVR(3.1, 4.5, 9),
        LegendreDVR(np.deg2rad(90.0), np.deg2rad(150.0), 9),
        SineDVR(-0.7, 0.7, 9),
    )
    dvr = DVR.from_axes(axes, names=("qs", "theta", "qa"))

    class _Dynamics:
        x = dvr.x
        axes = dvr.axes
        nstates = 3

    center = (
        SQRT2 * REFERENCE_BOND,
        np.deg2rad(REFERENCE_THETA_DEG),
        0.0,
    )
    width = (
        REFERENCE_BOND_WIDTH,
        np.deg2rad(REFERENCE_THETA_WIDTH_DEG),
        REFERENCE_BOND_WIDTH,
    )
    packet = initial_state(
        _Dynamics(),
        state=2,
        center=center,
        width=width,
    )

    coefficients = np.asarray(tt_to_tensor(packet.factors))
    qs, theta, qa = np.meshgrid(*dvr.x, indexing="ij")
    raw = np.exp(
        -0.5
        * (
            ((qs - center[0]) / width[0]) ** 2
            + ((theta - center[1]) / width[1]) ** 2
            + ((qa - center[2]) / width[2]) ** 2
        )
    )
    quadrature_weights = (
        np.full(axes[0].npts, axes[0].dx)[:, None, None]
        * axes[1].w[None, :, None]
        * np.full(axes[2].npts, axes[2].dx)[None, None, :]
    )
    expected = np.sqrt(quadrature_weights) * raw
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(coefficients[:2], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(coefficients[2], expected, atol=1.0e-14)


def test_qa_packet_matches_full_ldr_bond_moments_after_coordinate_transform():
    r = np.linspace(2.32, 3.28, 9)
    scan = SO2LinkedScan(
        solver=None,
        apes=np.empty((0,)),
        r1=r,
        r2=r,
        theta=np.empty((0,)),
        meta={},
    )
    qs, qa = transformed_stretch_nodes(scan)

    bond_probability = np.exp(
        -((r - REFERENCE_BOND) / REFERENCE_BOND_WIDTH) ** 2
    )
    bond_probability /= bond_probability.sum()
    full_probability = bond_probability[:, None] * bond_probability[None, :]

    qs_probability = np.exp(
        -(
            (qs - SQRT2 * REFERENCE_BOND)
            / REFERENCE_BOND_WIDTH
        ) ** 2
    )
    qa_probability = np.exp(
        -(qa / REFERENCE_BOND_WIDTH) ** 2
    )
    qs_probability /= qs_probability.sum()
    qa_probability /= qa_probability.sum()
    cg_probability = qs_probability[:, None] * qa_probability[None, :]

    r1_full, _r2_full = np.meshgrid(r, r, indexing="ij")
    qs_grid, qa_grid = np.meshgrid(qs, qa, indexing="ij")
    r1_cg = (qs_grid + qa_grid) / SQRT2

    full_mean = np.sum(full_probability * r1_full)
    cg_mean = np.sum(cg_probability * r1_cg)
    full_variance = np.sum(full_probability * (r1_full - full_mean) ** 2)
    cg_variance = np.sum(cg_probability * (r1_cg - cg_mean) ** 2)

    np.testing.assert_allclose(cg_mean, full_mean, atol=2.0e-5)
    np.testing.assert_allclose(
        np.sqrt(cg_variance),
        np.sqrt(full_variance),
        atol=3.0e-5,
    )


def test_qa_axis_uses_matched_full_difference_grid_when_refined():
    scan = SO2LinkedScan(
        solver=None,
        apes=np.empty((0,)),
        r1=np.array([2.32, 2.44, 2.56, 2.68, 2.80, 2.92, 3.04, 3.16, 3.28]),
        r2=np.array([2.32, 2.44, 2.56, 2.68, 2.80, 2.92, 3.04, 3.16, 3.28]),
        theta=np.empty((0,)),
        meta={},
    )

    coarse = qa_axis_from_scan(scan, npts=9)
    refined = qa_axis_from_scan(scan, npts=17)
    _qs_full, qa_full = transformed_stretch_nodes(scan)
    qa_coarse = centered_grid_window(qa_full, 9, center=0.0)
    expected_spacing = np.mean(np.diff(scan.r1)) / SQRT2

    np.testing.assert_allclose(coarse.x, qa_coarse, atol=1.0e-14)
    np.testing.assert_allclose(refined.x, qa_full, atol=1.0e-14)
    np.testing.assert_allclose(np.diff(coarse.x), expected_spacing, atol=1.0e-14)
    np.testing.assert_allclose(np.diff(refined.x), expected_spacing, atol=1.0e-14)


def test_transformed_stretch_patch_matches_full_tensor_grid_spacing():
    scan = SO2LinkedScan(
        solver=None,
        apes=np.empty((0,)),
        r1=np.array([2.32, 2.44, 2.56, 2.68, 2.80, 2.92, 3.04, 3.16, 3.28]),
        r2=np.array([2.32, 2.44, 2.56, 2.68, 2.80, 2.92, 3.04, 3.16, 3.28]),
        theta=np.empty((0,)),
        meta={},
    )

    qs_full, qa_full = transformed_stretch_nodes(scan)
    qs_axis, qs_indices = symmetric_stretch_nodes(scan)
    qs_patch = centered_grid_window(qs_full, 7, center=SQRT2 * REFERENCE_BOND)
    qa_patch = centered_grid_window(qa_full, 7, center=0.0)

    expected_spacing = np.mean(np.diff(scan.r1)) / SQRT2
    diagonal_qs = SQRT2 * scan.r1
    np.testing.assert_allclose(qs_axis, qs_full, atol=1.0e-14)
    np.testing.assert_array_equal(qs_indices, np.arange(qs_full.size))
    np.testing.assert_allclose(np.diff(qs_full), expected_spacing, atol=1.0e-14)
    np.testing.assert_allclose(np.diff(qa_full), expected_spacing, atol=1.0e-14)
    np.testing.assert_allclose(np.diff(qs_patch), expected_spacing, atol=1.0e-14)
    np.testing.assert_allclose(np.diff(qa_patch), expected_spacing, atol=1.0e-14)
    np.testing.assert_allclose(np.diff(diagonal_qs), 2.0 * expected_spacing)

    qs_grid, qa_grid = np.meshgrid(qs_patch, qa_patch, indexing="ij")
    r1_grid = (qs_grid + qa_grid) / SQRT2
    r2_grid = (qs_grid - qa_grid) / SQRT2
    assert r1_grid.min() >= scan.r1.min() - 1.0e-12
    assert r1_grid.max() <= scan.r1.max() + 1.0e-12
    assert r2_grid.min() >= scan.r2.min() - 1.0e-12
    assert r2_grid.max() <= scan.r2.max() + 1.0e-12


def test_sampled_product_gaussian_support_is_writable_after_broadcast():
    mask = sampled_product_gaussian_support(
        (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
        (0.0, 0.0),
        (0.5, 0.5),
    )

    assert mask.shape == (2, 2)
    assert mask[0, 0]


def test_cached_single_anchor_model_is_quadratic_analytical_fg():
    qa = np.array([-0.2, 0.0, 0.2])
    energies = np.array([[1.0, 2.0]])
    gradient = np.array([[0.3, 0.1], [0.1, -0.2]])
    hessian = np.array([[0.4, -0.05], [-0.05, 0.6]])
    data = CGLDRElectronicData(
        energies=energies,
        overlaps=np.eye(2).reshape(1, 2, 1, 2),
        hamiltonian_gradients=gradient.reshape(1, 1, 2, 2),
        hamiltonian_hessians=hessian.reshape(1, 1, 1, 2, 2),
        reactive_grids=(np.array([0.0]),),
        expanded_grids=(qa,),
        metadata={"qa_model": "3-anchor-relaxed-pt-quintic"},
    )

    quadratic = single_anchor_quadratic(data)
    expected = np.asarray(
        [np.diag(energies[0]) + q * gradient + 0.5 * q**2 * hessian for q in qa]
    )

    np.testing.assert_allclose(quadratic.separable_hamiltonian.evaluate()[0], expected)
    assert quadratic.metadata["qa_model"] == "single-reference-quadratic"
    assert quadratic.metadata["electronic_structure_recomputed"] is False


def test_overlap_quantum_metric_recovers_matrix_exponential():
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    expected = rotation @ np.diag([0.8, 3.0]) @ rotation.T
    displacement = 0.4
    values, vectors = np.linalg.eigh(expected)
    positive = (vectors * np.exp(-0.5 * values * displacement**2)) @ vectors.T
    unitary = np.array([[0.0, 1.0], [-1.0, 0.0]])

    metric, singular_values, ratio = overlap_quantum_metric(
        positive @ unitary,
        displacement,
    )

    np.testing.assert_allclose(metric, expected, atol=1.0e-13)
    np.testing.assert_allclose(
        singular_values,
        np.sort(np.exp(-0.5 * values * displacement**2)),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(ratio, singular_values[0] / singular_values[-1])


def test_harmonic_matrix_extension_preserves_data_and_fills_center():
    values = np.zeros((3, 3, 2, 2), dtype=complex)
    valid = np.ones((3, 3), dtype=bool)
    valid[1, 1] = False
    values[0, 1] = np.diag([1.0, 2.0])
    values[2, 1] = np.diag([3.0, 4.0])
    values[1, 0] = np.diag([5.0, 6.0])
    values[1, 2] = np.diag([7.0, 8.0])

    continued = harmonic_matrix_extension(values, valid)

    np.testing.assert_allclose(continued[valid], values[valid])
    np.testing.assert_allclose(continued[1, 1], np.diag([4.0, 5.0]))


def test_pyscf_fallback_electron_nuclear_derivatives_for_pyqed_molecule():
    pytest.importorskip("pyscf")

    from pyqed import Molecule
    from pyqed.qchem.geometric import _electron_nuclear_operator_derivatives

    mol = Molecule(
        atom=[["H", (0.0, 0.0, 0.0)], ["H", (0.0, 0.0, 1.4)]],
        basis="sto-3g",
        unit="bohr",
    )
    mol.build()

    first, second = _electron_nuclear_operator_derivatives(mol)

    assert first.shape == (6, 2, 2)
    assert second.shape == (6, 6, 2, 2)
    np.testing.assert_allclose(first, first.transpose(0, 2, 1).conj())
    np.testing.assert_allclose(second, second.transpose(0, 1, 3, 2).conj())
