from types import SimpleNamespace

import numpy as np

from examples.namd.phenol_dissociation_rate import fit_exponential_rate
from examples.namd.phenol_sa_casscf_3d_ftt_ttldr import (
    cap_operators,
    cap_profile,
    condon_packet,
    cumulative_cap_yield,
    directional_link_ftts,
    dvr_validation_design,
    gaussian_nuclear_packet,
    initial_packet,
    vibrational_ground_state,
    _update_rank_convergence,
)
from examples.namd.phenol_sa_casscf_3d_gp_control import (
    maximum_spanning_tree_gauge,
    rectangular_loop_phase,
)
from examples.namd.phenol_sa_casscf_5d_gp_control import (
    DiscreteLinkTT,
    ProjectedS1Oracle,
    _branch_segment,
    _encoded_loop_phases,
    _inner_conical_phase_links,
    _phase_link_cores,
    _selected_modes,
    _tt_hadamard,
)
from pyqed.models.phenol_coordinates import PhenolReactiveChart
from pyqed.mps.cross import tt_value
from pyqed.mps.functional import FunctionalTT


def test_hybrid_branch_uses_two_site_warmup_then_one_site_tdvp():
    args = SimpleNamespace(
        steps=40,
        checkpoint_steps=8,
        integrator="hybrid",
        tdvp2_warmup_steps=5,
    )
    assert _branch_segment(args, 0) == (5, "tdvp2")
    assert _branch_segment(args, 5) == (8, "tdvp")
    assert _branch_segment(args, 37) == (3, "tdvp")


def test_gp_ngp_branches_can_be_scheduled_independently():
    assert _selected_modes("both") == ("gp", "ngp")
    assert _selected_modes("gp") == ("gp",)
    assert _selected_modes("ngp") == ("ngp",)


def test_dvr_validation_design_enumerates_nodes_and_edges():
    axes = (
        np.linspace(0.9, 1.2, 4),
        np.linspace(-0.2, 0.2, 3),
        np.linspace(1.8, 2.0, 2),
    )
    coordinates, edges = dvr_validation_design(axes)
    assert coordinates.shape == (24, 3)
    assert [len(left) for left, _right in edges] == [18, 16, 12]
    for axis, (left, right) in enumerate(edges):
        changed = right - left
        inactive = [index for index in range(3) if index != axis]
        np.testing.assert_allclose(changed[:, inactive], 0.0)


def test_initial_packet_is_normalized_in_bright_channel():
    center = PhenolReactiveChart().equilibrium[:3]
    axes = tuple(
        np.linspace(value - width, value + width, count)
        for value, width, count in zip(center, (0.2, 0.3, 0.1), (7, 5, 5))
    )
    packet = initial_packet(axes, state=1)
    np.testing.assert_allclose(np.linalg.norm(packet), 1.0)
    np.testing.assert_allclose(packet[..., 0], 0.0)
    np.testing.assert_allclose(packet[..., 2], 0.0)


def test_condon_packet_places_normalized_nuclear_state_in_selected_channel():
    nuclear = np.arange(1.0, 13.0).reshape(3, 4)
    packet = condon_packet(nuclear, state=2)

    np.testing.assert_allclose(np.linalg.norm(packet), 1.0)
    np.testing.assert_allclose(packet[..., :2], 0.0)
    np.testing.assert_allclose(
        packet[..., 2], nuclear / np.linalg.norm(nuclear)
    )


def test_vibrational_ground_state_matches_dense_product_hamiltonian():
    kinetic = (
        np.asarray([[0.8, -0.2, 0.0], [-0.2, 0.5, -0.1], [0.0, -0.1, 0.9]]),
        np.asarray([[0.3, -0.08], [-0.08, 0.6]]),
    )
    potential = np.asarray([[0.2, 0.5], [0.0, 0.4], [0.3, 0.8]])
    dense = (
        np.kron(kinetic[0], np.eye(2))
        + np.kron(np.eye(3), kinetic[1])
        + np.diag(potential.reshape(-1))
    )
    reference_energy, reference_vectors = np.linalg.eigh(dense)

    energy, state, residual = vibrational_ground_state(
        kinetic, potential, guess=np.ones_like(potential)
    )

    np.testing.assert_allclose(energy, reference_energy[0], atol=1.0e-11)
    np.testing.assert_allclose(
        abs(np.vdot(reference_vectors[:, 0], state.reshape(-1))), 1.0, atol=1.0e-11
    )
    assert residual < 1.0e-10


def test_cap_profile_and_channel_operators_partition_absorber():
    axes = (
        np.linspace(0.9, 3.0, 8),
        np.linspace(-0.4, 0.4, 3),
        np.linspace(1.8, 2.0, 2),
    )
    profile = cap_profile(axes[0], 2.4, 0.02, order=4)
    total, channels = cap_operators(axes, profile, nstates=3)

    assert np.all(profile[axes[0] <= 2.4] == 0.0)
    np.testing.assert_allclose(profile[-1], 0.02)
    np.testing.assert_allclose(
        sum(channel.to_dense() for channel in channels), total.to_dense()
    )


def test_cumulative_cap_yield_integrates_norm_loss_rate():
    times = np.linspace(0.0, 2.0, 5)
    expectations = np.column_stack((np.full(5, 0.1), np.full(5, 0.2)))
    yields = cumulative_cap_yield(times, expectations)

    np.testing.assert_allclose(yields[-1], (0.4, 0.8))


def test_exponential_rate_recovers_known_first_order_decay():
    times = np.linspace(0.0, 200.0, 401)
    expected_rate_per_fs = 2.5e-4
    cumulative_yield = 1.0 - np.exp(-expected_rate_per_fs * times)

    fit = fit_exponential_rate(times, cumulative_yield, 50.0, 200.0)

    np.testing.assert_allclose(fit["rate_per_fs"], expected_rate_per_fs)
    np.testing.assert_allclose(fit["lifetime_ps"], 4.0)
    np.testing.assert_allclose(fit["r_squared"], 1.0)


def test_rank_convergence_uses_highest_rank_reference():
    records = [
        {"rank": 24, "populations": np.array([[0.2, 0.8]]), "radial": np.array([0.3, 0.7])},
        {"rank": 16, "populations": np.array([[0.1, 0.9]]), "radial": np.array([0.4, 0.6])},
    ]
    _update_rank_convergence(records)

    assert [record["rank"] for record in records] == [16, 24]
    np.testing.assert_allclose(records[0]["population_difference_from_highest"], 0.1)
    np.testing.assert_allclose(records[0]["radial_l1_difference_from_highest"], 0.2)
    assert records[1]["population_difference_from_highest"] == 0.0


def test_directional_link_ftts_avoid_feature_rank_squaring(tmp_path):
    axes = (
        np.linspace(-0.5, 0.5, 4),
        np.linspace(-0.3, 0.3, 3),
        np.linspace(-0.2, 0.2, 3),
    )
    coordinates, _ = dvr_validation_design(axes)
    angles = coordinates[:, 0] + 0.4 * coordinates[:, 1] * coordinates[:, 2]
    values = np.stack(
        (
            np.stack((np.cos(angles), -np.sin(angles)), axis=-1),
            np.stack((np.sin(angles), np.cos(angles)), axis=-1),
        ),
        axis=-2,
    ).reshape(4, 3, 3, 2, 2)
    feature = FunctionalTT(
        degrees=(3, 2, 2), rank=12, hermitian=False
    ).fit_grid(axes, values)

    models, records = directional_link_ftts(feature, axes, 12, tmp_path)

    assert len(models) == len(records) == 3
    assert max(record["relative_max"] for record in records) < 1.0e-10
    assert all(max(record["ranks"]) <= 12 for record in records)
    assert all((tmp_path / f"link_axis{axis}_rank12.npz").exists() for axis in range(3))


def test_gp_control_preserves_signed_loop_and_strips_only_its_phase():
    links = (
        np.ones((1, 2, 1), dtype=complex),
        np.asarray([[[1.0]], [[-1.0]]], dtype=complex),
        np.ones((2, 2, 0), dtype=complex),
    )
    _gauge, signed, tree_minimum = maximum_spanning_tree_gauge(
        links, (2, 2, 1), anchor=(0, 0, 0)
    )
    positive = tuple(np.abs(value).astype(complex) for value in signed)

    signed_phase, signed_minimum = rectangular_loop_phase(
        signed, (0, 0, 0), (1, 1, 0)
    )
    positive_phase, positive_minimum = rectangular_loop_phase(
        positive, (0, 0, 0), (1, 1, 0)
    )

    np.testing.assert_allclose(abs(signed_phase), np.pi)
    np.testing.assert_allclose(positive_phase, 0.0)
    np.testing.assert_allclose(signed_minimum, positive_minimum)
    np.testing.assert_allclose(tree_minimum, 1.0)
    for physical, control in zip(signed, positive):
        np.testing.assert_allclose(np.abs(physical), np.abs(control))


def test_5d_projected_s1_oracle_retains_gp_and_strips_only_link_phase():
    class ConicalEnergy:
        def predict(self, coordinates):
            coordinates = np.asarray(coordinates)
            x = coordinates[:, 0] - 1.105
            y = coordinates[:, 1] - 0.025
            values = np.zeros((len(coordinates), 3, 3))
            values[:, 0, 0] = -3.0
            values[:, 1, 1] = x
            values[:, 2, 2] = -x
            values[:, 1, 2] = values[:, 2, 1] = y
            return values

    class IdentityFeature:
        def predict(self, coordinates):
            return np.broadcast_to(np.eye(3), (len(coordinates), 3, 3)).copy()

    equilibrium = PhenolReactiveChart().equilibrium
    axes = (
        np.asarray((1.0, 1.1, 1.2)),
        np.asarray((-0.2, 0.0, 0.2)),
        np.asarray((equilibrium[2] - 0.1, equilibrium[2] + 0.1)),
        np.asarray((-0.1, 0.1)),
        np.asarray((-0.1, 0.1)),
    )
    oracle = ProjectedS1Oracle(axes, ConicalEnergy(), IdentityFeature())
    phase, minimum = rectangular_loop_phase(
        oracle.reference_links, (0, 0, 0), (2, 2, 0)
    )
    np.testing.assert_allclose(abs(phase), np.pi)
    assert minimum > 0.0

    indices = np.asarray(((0, 0, 0, 0, 0), (1, 1, 1, 1, 1)))
    gp = oracle.link(0, indices)
    ngp = oracle.link(0, indices, strip_phase=True)
    np.testing.assert_allclose(np.abs(gp), ngp)


def test_discrete_scalar_link_tt_exposes_edge_grid_cores():
    shape = (3, 2, 4)
    cores = tuple(np.ones((1, size, 1), dtype=complex) for size in shape)
    model = DiscreteLinkTT(cores, shape)

    exposed = model.tensor_cores(tuple(np.arange(size) for size in shape))

    assert model.output_shape_ == (1, 1)
    assert len(exposed) == len(shape) + 1
    assert exposed[-1].shape == (1, 1, 1)
    np.testing.assert_allclose(model.values(((0, 0, 0), (2, 1, 3))), 1.0)


def test_z2_branch_cut_keeps_only_the_inner_ci_flux():
    class ConicalEnergy:
        def predict(self, coordinates):
            coordinates = np.asarray(coordinates)
            x = coordinates[:, 0] - 1.14
            y = coordinates[:, 1]
            radius = np.sqrt(x**2 + y**2)
            values = np.zeros((len(coordinates), 3, 3))
            values[:, 0, 0] = -3.0
            values[:, 1, 1] = -radius
            values[:, 2, 2] = radius
            return values

    equilibrium = PhenolReactiveChart().equilibrium
    axes = (
        np.linspace(0.95, 2.05, 23),
        np.linspace(-0.3, 0.3, 7),
        np.asarray((equilibrium[2] - 0.1, equilibrium[2] + 0.1)),
        np.asarray((-0.1, 0.1)),
        np.asarray((-0.1, 0.1)),
    )
    physical, info = _inner_conical_phase_links(axes, ConicalEnergy())
    cores, _records = _phase_link_cores(physical, tuple(map(len, axes)))
    loops = _encoded_loop_phases(axes, cores)

    np.testing.assert_allclose(abs(loops["inner"]["phase_radian"]), np.pi)
    np.testing.assert_allclose(loops["outer"]["phase_radian"], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(loops["inner"]["minimum_link_magnitude"], 1.0)
    assert 0.95 <= info["phase_flux_center"][0] <= 1.50
    assert info["construction"] == "real Z2 branch-cut connection"


def test_full_magnitude_gp_control_removes_only_link_phase():
    shape = (4, 3, 2, 2, 2)
    axes = tuple(np.arange(size, dtype=float) for size in shape)
    phase_links = (
        np.ones((shape[0] - 1, shape[1], 1), dtype=complex),
        np.ones((shape[0], shape[1] - 1, 1), dtype=complex),
        np.ones((shape[0], shape[1], 0), dtype=complex),
    )
    phase_links[1][2:, 0, 0] = -1.0
    phases, _records = _phase_link_cores(phase_links, shape)

    for axis in range(len(shape)):
        edge_shape = list(shape)
        edge_shape[axis] -= 1
        magnitude = tuple(
            np.linspace(0.91 + 0.01 * site, 0.99, size).reshape(1, size, 1)
            for site, size in enumerate(edge_shape)
        )
        gp = _tt_hadamard(magnitude, phases[axis])
        for index in np.ndindex(*edge_shape):
            gp_value = tt_value(gp, index)
            ngp_value = tt_value(magnitude, index)
            np.testing.assert_allclose(abs(gp_value), abs(ngp_value), atol=1.0e-14)
