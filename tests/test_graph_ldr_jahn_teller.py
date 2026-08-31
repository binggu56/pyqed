import importlib.util
from pathlib import Path

import numpy as np

from pyqed.ldr import GraphLDR, GraphMesh


def _load_example():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "ldr"
        / "graph_ldr_jahn_teller.py"
    )
    spec = importlib.util.spec_from_file_location("graph_ldr_jahn_teller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_polar_mesh_collapses_the_coordinate_singularity_to_one_node():
    radii = np.linspace(0.0, 4.0, 9)
    mesh = GraphMesh.polar(radii, 16)

    assert mesh.size == 1 + (len(radii) - 1) * 16
    assert np.count_nonzero(np.linalg.norm(mesh.nodes, axis=1) == 0.0) == 1
    np.testing.assert_allclose(
        mesh.kinetic() @ np.sqrt(mesh.volumes),
        0.0,
        atol=1.0e-13,
    )


def test_full_jahn_teller_polar_and_cartesian_graphs_agree():
    model = _load_example()
    result = model.full_two_state_comparison()

    assert result["cartesian_mesh"].size == 441
    assert result["polar_mesh"].size == 289
    assert result["max_energy_difference"] < 0.03
    cartesian = result["cartesian_energies"]
    polar = result["polar_energies"]
    np.testing.assert_allclose(cartesian[::2], cartesian[1::2], atol=1.0e-10)
    np.testing.assert_allclose(polar[::2], polar[1::2], atol=1.0e-10)


def test_full_polar_adiabatic_ldr_is_exactly_diabatic_at_the_center():
    model = _load_example()
    mesh = GraphMesh.polar(np.linspace(0.0, 4.0, 9), 16)
    potential = model.diabatic_potential(mesh.nodes)
    solver = GraphLDR(mesh, 2).set_diabatic(potential)
    global_diabatic = np.kron(mesh.kinetic().toarray(), np.eye(2))
    for node in range(mesh.size):
        section = slice(2 * node, 2 * node + 2)
        global_diabatic[section, section] += potential[node]
    transform = np.zeros_like(global_diabatic, dtype=complex)
    for node, frame in enumerate(solver.frames):
        section = slice(2 * node, 2 * node + 2)
        transform[section, section] = frame

    expected = transform.conj().T @ global_diabatic @ transform
    np.testing.assert_allclose(solver.hamiltonian(), expected, atol=1.0e-12)


def test_single_surface_annulus_has_berry_phase_and_qgt_limit():
    model = _load_example()
    coarse = model.lower_surface_annulus(ntheta=24)
    fine = model.lower_surface_annulus(ntheta=48)

    np.testing.assert_allclose(coarse["wilson_loop"], -1.0, atol=1.0e-12)
    np.testing.assert_allclose(fine["wilson_loop"], -1.0, atol=1.0e-12)
    assert fine["max_energy_difference"] < coarse["max_energy_difference"]
    assert fine["max_energy_difference"] < 1.0e-3


def test_jahn_teller_wavepacket_crossing_preserves_norm_and_transfers_population():
    model = _load_example()
    result = model.wavepacket_dynamics(
        nr=13,
        ntheta=24,
        dt=0.04,
        nsteps=100,
        nout=5,
    )

    np.testing.assert_allclose(result["solver"].norm, 1.0, atol=1.0e-11)
    np.testing.assert_allclose(
        result["adiabatic_populations"].sum(axis=1),
        1.0,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        result["diabatic_populations"].sum(axis=1),
        1.0,
        atol=1.0e-11,
    )
    assert result["diabatic_populations"][0, 0] > 1.0 - 1.0e-12
    assert np.max(result["adiabatic_populations"][:, 1]) > 0.4
    assert np.max(result["coordinate_means"][:, 0]) > 1.0


def test_full_ldr_reference_uses_full_overlaps_and_preserves_norm():
    model = _load_example()
    result = model.full_ldr_dynamics(
        ncart=9,
        dt=0.04,
        nsteps=10,
        nout=2,
    )

    assert result["solver"].overlaps is not None
    assert result["solver"].links is None
    np.testing.assert_allclose(result["solver"].norm, 1.0, atol=1.0e-11)
    np.testing.assert_allclose(
        result["diabatic_populations"][0],
        (1.0, 0.0),
        atol=1.0e-12,
    )


def test_fem_graph_wavepacket_dynamics_preserves_norm():
    model = _load_example()
    result = model.wavepacket_dynamics(
        nr=9,
        ntheta=16,
        dt=0.04,
        nsteps=10,
        nout=2,
        mesh_method="fem",
    )

    assert result["mesh"]._stiffness is not None
    np.testing.assert_allclose(result["solver"].norm, 1.0, atol=1.0e-11)


def test_fourth_order_graph_dynamics_preserves_norm():
    model = _load_example()
    result = model.fourth_order_graph_dynamics(
        ncart=9,
        dt=0.04,
        nsteps=10,
        nout=2,
    )

    assert result["mesh"]._stiffness is not None
    np.testing.assert_allclose(result["solver"].norm, 1.0, atol=1.0e-11)


def test_adaptive_p2_dynamics_refines_a_pilot_mesh_and_preserves_norm():
    model = _load_example()
    result = model.adaptive_fem_ldr_dynamics(
        nr=4,
        ntheta=8,
        dt=0.04,
        nsteps=4,
        nout=2,
    )
    pilot = result["adaptation_history"][0]

    assert len(pilot["marked"]) > 0
    assert result["mesh"].size > pilot["mesh"].size
    np.testing.assert_allclose(result["solver"].norm, 1.0, atol=1.0e-11)


def test_iterative_adaptation_stages_a_fixed_node_budget():
    model = _load_example()
    result = model.adaptive_fem_ldr_dynamics(
        nr=4,
        ntheta=8,
        cycles=2,
        target_nodes=180,
        dt=0.04,
        nsteps=4,
        nout=2,
    )
    history = result["adaptation_history"]

    assert len(history) == 2
    assert history[0]["stage_target_nodes"] < history[1]["stage_target_nodes"]
    assert history[0]["mesh"].size < history[1]["mesh"].size
    assert abs(result["mesh"].size - 180) < 10
