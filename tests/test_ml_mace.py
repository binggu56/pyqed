import importlib.util
import runpy

import numpy as np
import pytest

from pyqed.ml import (
    MACE,
    MACEEncoder,
    MACEStateModel,
    canonicalize_coordinate_exchange,
    conserve_atomic_charges,
    frame_projector,
    infer_exchange_ambient_representation,
    positions_to_angstrom,
    qcschema_training_records,
    transform_electronic_gauge,
)


MACE_AVAILABLE = importlib.util.find_spec("mace") is not None


@pytest.mark.skipif(not MACE_AVAILABLE, reason="mace-torch is not installed")
def test_mace_chart_bounds_can_remain_fixed_when_the_grid_expands():
    grid = np.linspace(-0.4, 0.4, 3)

    def geometry(q):
        return np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.8 + q[0])))

    fit = MACE(
        (grid,), ("H", "H"), geometry, 1,
        chart_features=True, chart_bounds=((-0.2, 0.2),),
        channels=2, max_ell=1, interactions=1, radial_basis=2,
        radial_mlp=(4,), cutoff=3.0,
    )

    assert fit.chart_bounds == ((-0.2, 0.2),)
    np.testing.assert_allclose(fit._chart_center, (0.0,))
    np.testing.assert_allclose(fit._chart_scale, (0.2,))


def test_so2_casci_generator_requests_pure_singlet_roots(monkeypatch):
    namespace = runpy.run_path("examples/namd/generate_so2_casci_singlets.py")
    captured = {}

    class FakeMolecule:
        def __init__(self, **options):
            captured["molecule"] = options

        def build(self, **options):
            captured["build"] = options

    class FakeReference:
        converged = True

        def run(self, **options):
            captured["scf"] = options
            return self

    class FakeCASCI:
        def __init__(self, _reference, **options):
            captured["casci"] = options

        def run(self, **options):
            captured["run"] = options
            return self

    monkeypatch.setitem(namespace["electronic_structure"].__globals__, "Molecule", FakeMolecule)
    monkeypatch.setitem(
        namespace["electronic_structure"].__globals__,
        "RHF",
        lambda _molecule: FakeReference(),
    )
    monkeypatch.setitem(namespace["electronic_structure"].__globals__, "CASCI", FakeCASCI)
    options = type(
        "Options",
        (),
        {
            "basis": "sto-3g",
            "scf_tol": 1.0e-10,
            "max_cycle": 100,
            "ncas": 6,
            "nelecas": 6,
            "spin_root_cushion": 8,
            "nstates": 3,
        },
    )()
    namespace["electronic_structure"](2.8, 2.8, np.deg2rad(120.0), options)
    assert captured["casci"] == {
        "ncas": 6,
        "nelecas": 6,
        "ms2": 0,
        "multiplicity": 1,
    }
    assert captured["run"] == {"nstates": 3, "method": "direct_ci"}


def test_so2_spin_pure_cache_metadata_is_checked(tmp_path):
    namespace = runpy.run_path("examples/namd/generate_so2_casci_singlets.py")
    options = type(
        "Options",
        (),
        {
            "basis": "sto-3g",
            "nstates": 3,
            "ncas": 6,
            "nelecas": 6,
            "spin_root_cushion": 8,
        },
    )()
    path = tmp_path / "electronic.npz"
    np.savez(path, **namespace["electronic_metadata"](options))
    with np.load(path, allow_pickle=False) as archive:
        namespace["validate_electronic_metadata"](archive, options)
    options.ncas = 4
    with np.load(path, allow_pickle=False) as archive:
        with pytest.raises(ValueError, match="ncas"):
            namespace["validate_electronic_metadata"](archive, options)


def test_so2_spin_pure_cache_rejects_contaminated_root():
    namespace = runpy.run_path("examples/namd/generate_so2_casci_singlets.py")
    namespace["require_spin_pure_singlets"](np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="Spin-pure singlet selection failed"):
        namespace["require_spin_pure_singlets"]([[0.0, 2.0, 0.0]])


def test_so2_sobol_design_is_nested_exchange_reduced_and_sparse():
    namespace = runpy.run_path("examples/namd/so2_casci_sobol_mace.py")
    bounds = (2.68, 2.92, np.deg2rad(110.0), np.deg2rad(130.0))
    small = namespace["sobol_coordinates"](17, bounds, 19)
    large = namespace["sobol_coordinates"](33, bounds, 19)
    np.testing.assert_allclose(small, large[:17])
    assert np.all(large[:, 0] >= large[:, 1])
    pairs, lengths = namespace["sparse_overlap_graph"](large, bounds, 4)
    assert len(pairs) < 4 * len(large)
    assert np.all(lengths > 0.0)
    reached = {0}
    while True:
        expanded = reached | {
            right if left in reached else left
            for left, right in pairs
            if left in reached or right in reached
        }
        if expanded == reached:
            break
        reached = expanded
    assert len(reached) == len(large)


def test_so2_sobol_cached_overlap_graph_scales_linearly():
    namespace = runpy.run_path("examples/namd/so2_casci_sobol_mace.py")
    bounds = (2.68, 2.92, np.deg2rad(110.0), np.deg2rad(130.0))
    coordinates = namespace["sobol_coordinates"](65, bounds, 19)
    pairs = set()
    for count in (17, 33, 65):
        graph, _lengths = namespace["sparse_overlap_graph"](
            coordinates[:count], bounds, 6
        )
        pairs.update(map(tuple, graph))
    assert len(pairs) < 9 * len(coordinates)


def test_so2_sobol_overlap_pruning_preserves_connectivity():
    namespace = runpy.run_path("examples/namd/so2_casci_sobol_mace.py")
    pairs = np.asarray([[0, 1], [1, 2], [0, 2], [2, 3]])
    lengths = np.asarray([0.1, 0.2, 0.4, 0.3])
    values = np.asarray([[[0.9]], [[0.8]], [[0.1]], [[0.2]]])
    kept, _lengths, _values, singular_values = namespace["prune_overlap_graph"](
        pairs, lengths, values, 4, 0.5
    )
    assert {tuple(pair) for pair in kept} == {(0, 1), (1, 2), (2, 3)}
    np.testing.assert_allclose(singular_values, [0.9, 0.8, 0.2])


def test_so2_sobol_tangent_probe_design_has_one_probe_per_axis():
    namespace = runpy.run_path("examples/namd/so2_casci_sobol_probes.py")
    bounds = (2.68, 2.92, np.deg2rad(110.0), np.deg2rad(130.0))
    centers = np.asarray([[2.8, 2.8, np.deg2rad(120.0)], [2.91, 2.7, np.deg2rad(129.0)]])
    coordinates, pairs, axes = namespace["probe_design"](
        centers, bounds, (0.06, 0.06, np.deg2rad(5.0))
    )
    assert coordinates.shape == (8, 3)
    np.testing.assert_array_equal(axes, [0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(pairs[:, 0], [0, 0, 0, 1, 1, 1])
    assert np.all(coordinates >= np.asarray((bounds[0], bounds[0], bounds[2])))
    assert np.all(coordinates <= np.asarray((bounds[1], bounds[1], bounds[3])))


def test_so2_sobol_anchor_gauge_preserves_spectra_and_links():
    namespace = runpy.run_path("examples/namd/so2_casci_sobol_probes.py")
    angles = (0.0, 0.2, -0.3)
    gauges = np.asarray([
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        for angle in angles
    ], dtype=complex)
    energies = np.asarray([[0.0, 1.0], [0.2, 1.3], [0.4, 1.5]])
    pairs = np.asarray([[0, 1], [1, 2]])
    raw_links = np.asarray([
        gauges[left] @ np.diag([0.9, 0.7]) @ gauges[right].conj().T
        for left, right in pairs
    ])
    hamiltonians, links, recovered, shift = namespace["align_to_anchor"](
        energies, pairs, raw_links, gauges
    )
    np.testing.assert_allclose(np.linalg.eigvalsh(hamiltonians), energies - shift)
    np.testing.assert_allclose(
        np.linalg.svd(links, compute_uv=False),
        np.linalg.svd(raw_links, compute_uv=False),
    )
    np.testing.assert_allclose(
        recovered.conj().swapaxes(-1, -2) @ recovered,
        np.broadcast_to(np.eye(2), recovered.shape),
        atol=1.0e-12,
    )


def test_so2_sobol_probe_design_can_enrich_one_axis_on_both_sides():
    namespace = runpy.run_path("examples/namd/so2_casci_sobol_probes.py")
    bounds = (2.68, 2.92, np.deg2rad(110.0), np.deg2rad(130.0))
    centers = np.asarray([[2.8, 2.8, np.deg2rad(120.0)]])
    coordinates, pairs, axes = namespace["probe_design"](
        centers, bounds, (0.06, 0.06, np.deg2rad(5.0)),
        two_sided_axes=(2,),
    )
    assert coordinates.shape == (5, 3)
    np.testing.assert_array_equal(axes, [0, 1, 2, 2])
    np.testing.assert_allclose(
        np.sort(np.rad2deg(coordinates[pairs[axes == 2, 1], 2])),
        [115.0, 125.0],
    )


def test_so2_nystrom_landmark_features_reconstruct_exact_gram():
    namespace = runpy.run_path("examples/namd/so2_casci_sobol_probes.py")
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(6, 6, 2))
    frames = np.linalg.qr(raw)[0][:, :, :2]
    landmark_indices = np.asarray([0, 2, 4])
    blocks = np.asarray([
        [frames[left].T @ frames[right] for right in range(len(frames))]
        for left in landmark_indices
    ])
    features, spectrum = namespace["nystrom_features"](
        blocks, landmark_indices, np.broadcast_to(np.eye(2), (6, 2, 2)), 6
    )
    reconstructed = np.asarray([
        features[left].conj().T @ features[right]
        for left in range(6) for right in range(6)
    ])
    reference = np.asarray([
        frames[left].T @ frames[right]
        for left in range(6) for right in range(6)
    ])
    np.testing.assert_allclose(reconstructed, reference, atol=1.0e-10)
    assert len(spectrum) == 6


def test_so2_anisotropic_design_resolves_bend_without_tensor_bond_grid():
    namespace = runpy.run_path("examples/namd/so2_casci_anisotropic_mace.py")
    bounds = (2.68, 2.92, np.deg2rad(110.0), np.deg2rad(130.0))
    coordinates, bonds, theta, anchor = namespace["anisotropic_design"](
        9, 5, bounds, 19
    )
    pairs, axes = namespace["anisotropic_links"](
        coordinates, bonds, theta, bounds, 4
    )
    assert coordinates.shape == (45, 3)
    assert anchor == 2
    assert np.count_nonzero(axes == 2) == 9 * 4
    assert np.count_nonzero(axes == -1) < 9 * 5 * 4
    assert len(pairs) < 5 * len(coordinates)


def test_so2_bisector_frame_turns_oxygen_exchange_into_fixed_c2x():
    namespace = runpy.run_path("examples/namd/generate_so2_casci_singlets.py")
    geometry = namespace["geometry"]
    coordinate = (2.9, 2.7, np.deg2rad(124.0))
    original = geometry(*coordinate)
    exchanged = geometry(coordinate[1], coordinate[0], coordinate[2])
    c2x = np.diag([1.0, -1.0, -1.0])
    np.testing.assert_allclose(exchanged, (c2x @ original[[2, 1, 0]].T).T)


def test_so2_631gstar_ao_symmetry_maps_split_contractions_by_shape():
    namespace = runpy.run_path("examples/namd/generate_so2_casci_singlets.py")
    xyz = namespace["geometry"](2.8, 2.8, np.deg2rad(120.0))
    molecule = namespace["Molecule"](
        atom=[[symbol, tuple(position)] for symbol, position in zip(
            ("O", "S", "O"), xyz
        )],
        charge=0,
        spin=0,
        unit="bohr",
        basis="6-31g*",
    )
    molecule.build(eri="dense")
    operator = namespace["ao_diagonal_symmetry_operator"](
        molecule, (1.0, -1.0, -1.0)
    )
    np.testing.assert_allclose(operator.T @ operator, np.eye(molecule.nao))


def test_so2_symmetry_cut_selects_irreps_independently_of_root_order():
    namespace = runpy.run_path("examples/namd/generate_so2_casci_symmetry_cut.py")
    energies = np.asarray([[0.0, 0.3, 0.2, 0.1], [0.0, 0.1, 0.2, 0.3]])
    labels = np.asarray([
        ["A1", "A2", "B2", "B1"],
        ["A1", "B2", "A2", "B1"],
    ])
    selected, roots = namespace["select_lowest_irreps"](energies, labels)
    np.testing.assert_allclose(selected, [[0.0, 0.2, 0.3], [0.0, 0.1, 0.2]])
    np.testing.assert_array_equal(roots, [[0, 2, 1], [0, 1, 2]])


def test_so2_positive_link_gauge_removes_neighbor_rotations():
    namespace = runpy.run_path("examples/namd/generate_so2_casci_symmetry_cut.py")

    def rotation(angle):
        return np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])

    links = np.asarray([
        rotation(0.31) @ np.diag([0.92, 0.88, 0.81]),
        rotation(-0.27) @ np.diag([0.94, 0.85, 0.79]),
    ])
    gauges, aligned = namespace["positive_link_gauge"](links, anchor=1)
    for gauge in gauges:
        np.testing.assert_allclose(gauge.conj().T @ gauge, np.eye(3), atol=1e-13)
    for link in aligned:
        np.testing.assert_allclose(link, link.conj().T, atol=1e-13)
        assert np.min(np.linalg.eigvalsh(link)) > 0.0
        np.testing.assert_allclose(
            namespace["procrustes"](link)[0], np.eye(3), atol=1e-13
        )
    energies = np.asarray([[0.0, 0.2, 0.4], [0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    hamiltonian = namespace["rotate_selected_energies"](energies, gauges)
    np.testing.assert_allclose(np.linalg.eigvalsh(hamiltonian), energies, atol=1e-13)


def test_so2_three_state_transport_uses_full_raw_root_links():
    namespace = runpy.run_path(
        "examples/namd/fit_so2_casci_3state_from_6roots.py"
    )

    def rotation(angle):
        value = np.eye(6)
        value[0, 0] = value[3, 3] = np.cos(angle)
        value[0, 3] = -np.sin(angle)
        value[3, 0] = np.sin(angle)
        return value

    raw_frames = np.asarray([rotation(angle) for angle in (-0.3, 0.1, 0.6)])
    full_links = np.asarray([
        0.8 * raw_frames[left].T @ raw_frames[left + 1]
        for left in range(2)
    ])
    frames, links = namespace["transport_subspace"](
        full_links, anchor=1, anchor_states=(0, 1, 2)
    )
    np.testing.assert_allclose(
        frames.conj().swapaxes(-1, -2) @ frames,
        np.broadcast_to(np.eye(3), (3, 3, 3)),
        atol=1e-13,
    )
    np.testing.assert_allclose(links, np.broadcast_to(0.8 * np.eye(3), links.shape))


def test_so2_casci_lift_resolves_lowest_singlets_into_exchange_sectors():
    namespace = runpy.run_path("examples/namd/generate_so2_casci_singlets.py")
    options = type(
        "Options",
        (),
        {
            "basis": "sto-3g",
            "scf_tol": 1.0e-10,
            "max_cycle": 100,
            "ncas": 6,
            "nelecas": 6,
            "spin_root_cushion": 8,
            "nstates": 3,
        },
    )()
    representation, raw, diagnostics = namespace["so2_exchange_representation"](
        2.8, np.deg2rad(120.0), options
    )
    np.testing.assert_allclose(np.diag(representation), [1.0, -1.0, 1.0])
    np.testing.assert_allclose(raw @ raw, np.eye(3), atol=1.0e-10)
    assert diagnostics["ao_metric_defect"] < 1.0e-10
    names, point_group, _raw, group_diagnostics = namespace[
        "so2_point_group_representations"
    ](2.8, np.deg2rad(120.0), options)
    assert names == ("E", "C2(x)", "sigma_xy", "sigma_xz")
    np.testing.assert_allclose(
        np.real(np.diagonal(point_group, axis1=1, axis2=2)),
        [[1, 1, 1], [1, -1, 1], [1, -1, -1], [1, 1, -1]],
        atol=1.0e-6,
    )
    assert group_diagnostics["generator_product_defect"] < 1.0e-10


def test_so2_symmetry_block_procrustes_does_not_mix_exchange_sectors():
    namespace = runpy.run_path("examples/namd/so2_casci_anisotropic_mace.py")
    representation = np.diag([1.0, -1.0, 1.0])
    value = np.asarray(
        [[0.8, 0.03, -0.2], [0.04, -0.9, 0.02], [0.2, -0.01, 0.8]],
        dtype=complex,
    )
    rotation = namespace["symmetry_block_procrustes"](value, representation)
    np.testing.assert_allclose(rotation.conj().T @ rotation, np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(
        rotation @ representation - representation @ rotation, 0.0, atol=1.0e-12
    )


def test_exchange_ambient_involution_intertwines_fixed_endpoint_frames():
    representation = np.diag([1.0, -1.0])
    frames = np.zeros((2, 4, 2), dtype=complex)
    frames[0, 0, 0] = frames[0, 1, 1] = 1.0
    frames[1, 2, 0] = frames[1, 3, 1] = 1.0
    ambient, diagnostics = infer_exchange_ambient_representation(
        frames, representation
    )
    np.testing.assert_allclose(ambient.conj().T @ ambient, np.eye(4), atol=1.0e-12)
    np.testing.assert_allclose(ambient @ ambient, np.eye(4), atol=1.0e-12)
    np.testing.assert_allclose(
        np.einsum("ab,nbi,ij->naj", ambient, frames, representation),
        frames,
        atol=1.0e-12,
    )
    assert diagnostics["ambient_odd_dimension"] == 2


def test_coordinate_exchange_canonicalization_reports_orbit_branch_and_fixed_set():
    coordinates = np.asarray([[2.9, 2.7, 2.0], [2.7, 2.9, 2.0], [2.8, 2.8, 2.0]])
    canonical, swapped, fixed = canonicalize_coordinate_exchange(coordinates)
    np.testing.assert_allclose(canonical[0], canonical[1])
    np.testing.assert_array_equal(swapped, [False, True, False])
    np.testing.assert_array_equal(fixed, [False, False, True])


def test_mace_length_units_convert_to_ase_angstroms():
    np.testing.assert_allclose(
        positions_to_angstrom([[1.0, 0.0, 0.0]], "bohr"),
        [[0.529177210544, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        positions_to_angstrom([[1.0, 0.0, 0.0]], "angstrom"),
        [[1.0, 0.0, 0.0]],
    )
    with pytest.raises(ValueError, match="unknown geometry units"):
        positions_to_angstrom([[0.0, 0.0, 0.0]], "nanometer")


def test_state_charges_are_exactly_conserved_with_padding_mask():
    raw = np.arange(24.0).reshape(2, 3, 4)
    mask = np.asarray([[True, True, False, False], [True, True, True, False]])
    charges = conserve_atomic_charges(raw, [0.0, 1.0], mask)
    np.testing.assert_allclose(charges.sum(axis=-1), [[0.0] * 3, [1.0] * 3])
    padded = np.broadcast_to(~mask[:, None, :], charges.shape)
    np.testing.assert_allclose(charges[padded], 0.0)


def test_frame_projector_is_invariant_under_independent_local_electronic_gauges():
    rng = np.random.default_rng(11)
    frames = rng.normal(size=(5, 7, 3)) + 1j * rng.normal(size=(5, 7, 3))
    gauges = []
    for _ in range(len(frames)):
        q, r = np.linalg.qr(
            rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        )
        phases = np.diag(r)
        gauges.append(q * (phases / np.abs(phases)).conj()[None, :])
    transformed = frames @ np.asarray(gauges)
    np.testing.assert_allclose(
        frame_projector(transformed), frame_projector(frames), atol=1.0e-12
    )


def test_frame_hamiltonian_and_links_transform_in_the_same_local_gauge():
    rng = np.random.default_rng(29)
    frames = rng.normal(size=(4, 6, 2)) + 1j * rng.normal(size=(4, 6, 2))
    raw_h = rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2))
    hamiltonians = 0.5 * (raw_h + raw_h.conj().swapaxes(-1, -2))
    gauges = np.asarray(
        [
            np.linalg.qr(
                rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            )[0]
            for _ in range(len(frames))
        ]
    )
    transformed_y, transformed_h = transform_electronic_gauge(
        frames, hamiltonians, gauges
    )
    expected_h = gauges.conj().swapaxes(-1, -2) @ hamiltonians @ gauges
    np.testing.assert_allclose(transformed_h, expected_h, atol=1.0e-12)
    links = frames[:-1].conj().swapaxes(-1, -2) @ frames[1:]
    transformed_links = (
        transformed_y[:-1].conj().swapaxes(-1, -2) @ transformed_y[1:]
    )
    expected_links = (
        gauges[:-1].conj().swapaxes(-1, -2) @ links @ gauges[1:]
    )
    np.testing.assert_allclose(transformed_links, expected_links, atol=1.0e-12)


def test_qcschema_records_feed_variable_molecule_training_view():
    records = [
        {
            "molecule": {
                "symbols": ["H", "H"],
                "geometry": [0.0, 0.0, -0.7, 0.0, 0.0, 0.7],
                "molecular_charge": 0,
                "molecular_multiplicity": 1,
            },
            "model": {"method": "SA-CASSCF", "basis": "sto-3g"},
            "extras": {
                "pyqed_hamiltonian": [[0.0, 0.01], [0.01, 0.2]],
                "pyqed_state_charges": [[0.1, -0.1], [0.2, -0.2]],
                "pyqed_manifold": "singlet-S0S1",
            },
        },
        {
            "molecule": {
                "symbols": ["H", "O", "H"],
                "geometry": [0.0] * 9,
                "molecular_charge": 1,
                "molecular_multiplicity": 2,
            },
            "model": {"method": "SA-CASSCF", "basis": "sto-3g"},
            "extras": {
                "pyqed_hamiltonian": [[0.1, 0.0], [0.0, 0.3]],
                "pyqed_state_charges": [[0.2, 0.6, 0.2], [0.3, 0.4, 0.3]],
                "pyqed_manifold": "singlet-S0S1",
            },
        },
    ]
    values = qcschema_training_records(records)
    assert [len(item) for item in values["atomic_numbers"]] == [2, 3]
    assert values["units"] == "bohr"
    assert values["fidelities"] == ["SA-CASSCF/sto-3g"] * 2
    assert values["manifolds"] == ["singlet-S0S1"] * 2
    np.testing.assert_allclose(values["molecular_charges"], [0.0, 1.0])


def test_mace_dependency_error_is_actionable_when_backend_missing():
    if MACE_AVAILABLE:
        pytest.skip("mace-torch is installed")
    with pytest.raises(ModuleNotFoundError, match=r"Install pyqed\[mace\]"):
        MACEEncoder(("H", "H"))


def test_qcschema_ingestion_rejects_partial_charge_coverage():
    record = {
        "molecule": {"symbols": ["H"], "geometry": [0.0, 0.0, 0.0]},
        "model": {"method": "HF", "basis": "sto-3g"},
        "extras": {"pyqed_hamiltonian": [[0.0]]},
    }
    values = qcschema_training_records([record])
    assert values["atomic_charges"] is None


@pytest.mark.skipif(not MACE_AVAILABLE, reason="mace-torch is not installed")
def test_native_mace_encoder_is_invariant_and_differentiable():
    encoder = MACEEncoder(
        ("H", "H"),
        channels=2,
        max_ell=1,
        interactions=1,
        radial_basis=2,
        radial_mlp=(4,),
        cutoff=3.0,
    )
    geometry = np.asarray([[0.0, 0.0, -0.4], [0.0, 0.0, 0.4]])
    rotated = geometry @ np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    ).T

    values = encoder.encode([geometry, rotated])

    assert values.shape == (2, encoder.output_size)
    np.testing.assert_allclose(values[0], values[1], atol=2.0e-6)
    batch = encoder.batch([geometry])
    loss = encoder.forward(batch).square().sum()
    loss.backward()
    assert any(parameter.grad is not None for parameter in encoder.parameters())


@pytest.mark.skipif(not MACE_AVAILABLE, reason="mace-torch is not installed")
def test_mace_ldr_fit_distills_fields_for_ttldr():
    from pyqed.namd.ttldr import TTLDR

    grid = np.linspace(-0.2, 0.2, 3)

    def geometry(q):
        distance = 0.8 + q[0]
        return np.asarray([[0.0, 0.0, -distance / 2], [0.0, 0.0, distance / 2]])

    energy = (0.1 * grid**2)[:, None, None]
    links = (np.full((2, 1, 1), 0.98),)
    fit = MACE(
        (grid,),
        ("H", "H"),
        geometry,
        1,
        chart_features=True,
        channels=2,
        max_ell=1,
        interactions=1,
        radial_basis=2,
        radial_mlp=(4,),
        cutoff=3.0,
    ).fit_grid(
        energy,
        links,
        hidden=(4,),
        epochs=2,
        learning_rate=1.0e-2,
        tt_rank=3,
        tt_degree=2,
    )

    assert fit.success
    assert fit.info["backend"] == "mace-ldr"
    assert fit.info["chart_features"] is True
    assert fit.energy.output_shape_ == (1, 1)
    assert fit.links[0].output_shape_ == (1, 1)
    identity = np.eye(3)
    driver = TTLDR.from_fit(
        fit,
        keo=((1.0, (identity,)),),
        overlap_rank=3,
        operator_rank=None,
    )
    assert driver.dims == (3, 1)
    dense = driver.hamiltonian.to_dense()
    np.testing.assert_allclose(dense, dense.conj().T, atol=1.0e-10)


@pytest.mark.skipif(not MACE_AVAILABLE, reason="mace-torch is not installed")
def test_mace_fits_one_hermitian_matrix_field():
    grid = np.linspace(-0.2, 0.2, 3)

    def geometry(q):
        distance = 0.8 + q[0]
        return np.asarray([[0.0, 0.0, -distance / 2], [0.0, 0.0, distance / 2]])

    coordinates = grid[:, None]
    values = np.zeros((3, 2, 2), dtype=complex)
    values[:, 0, 0] = grid
    values[:, 1, 1] = -grid
    values[:, 0, 1] = 0.1j * grid
    values[:, 1, 0] = -0.1j * grid
    fit = MACE(
        (grid,),
        ("H", "H"),
        geometry,
        2,
        chart_features=True,
        channels=2,
        max_ell=1,
        interactions=1,
        radial_basis=2,
        radial_mlp=(4,),
        cutoff=3.0,
    ).fit_h(coordinates, values, hidden=(4,), epochs=2, seed=2)

    predicted = fit.neural_energy.predict(coordinates)
    assert predicted.shape == values.shape
    np.testing.assert_allclose(
        predicted, predicted.conj().swapaxes(-1, -2), atol=1.0e-7
    )
    assert fit.info["backend"] == "mace-h"


@pytest.mark.skipif(not MACE_AVAILABLE, reason="mace-torch is not installed")
def test_mace_hermitian_basis_head_has_no_forbidden_channels(tmp_path):
    grid = np.linspace(-0.2, 0.2, 3)

    def geometry(q):
        distance = 0.8 + q[0]
        return np.asarray([[0.0, 0.0, -distance / 2], [0.0, 0.0, distance / 2]])

    coordinates = grid[:, None]
    coefficients = np.column_stack((grid, -grid, 0.1 + grid**2))
    basis = np.asarray([
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
    ])
    fit = MACE(
        (grid,),
        ("H", "H"),
        geometry,
        2,
        chart_features=True,
        channels=2,
        max_ell=1,
        interactions=1,
        radial_basis=2,
        radial_mlp=(4,),
        cutoff=3.0,
    ).fit_basis_h(
        coordinates,
        coefficients,
        basis,
        hidden=(4,),
        epochs=2,
        learning_rate=1.0e-2,
        seed=2,
    )
    predicted = fit.neural_energy.predict(coordinates)
    np.testing.assert_allclose(predicted, predicted.swapaxes(-1, -2).conj())
    np.testing.assert_allclose(predicted.imag, 0.0)
    checkpoint = fit.save(tmp_path / "basis-head.pt")
    loaded = MACE.load(checkpoint, geometry, distill=False)
    np.testing.assert_allclose(loaded.neural_energy.predict(coordinates), predicted)
    assert fit.info["coefficients"] == 3
    fit.distill_energy(
        rank=3,
        degree=2,
        method="cross",
        points=3,
        sweeps=2,
        cross_validation=4,
        validation_points=5,
        seed=9,
    )
    distilled = fit.energy.predict(coordinates)
    np.testing.assert_allclose(distilled, distilled.swapaxes(-1, -2).conj())
    assert fit.info["distillation"]["method"] == "cross"
    assert fit.info["distillation"]["geometry_queries"] <= 3


@pytest.mark.skipif(not MACE_AVAILABLE, reason="mace-torch is not installed")
def test_mace_spectral_fit_uses_selected_eigenspace():
    grid = np.linspace(-0.2, 0.2, 3)

    def geometry(q):
        distance = 0.8 + q[0]
        return np.asarray([[0.0, 0.0, -distance / 2], [0.0, 0.0, distance / 2]])

    coordinates = grid[:, None]
    angles = 0.4 * grid
    frames = np.zeros((3, 3, 1))
    frames[:, 0, 0] = np.cos(angles)
    frames[:, 1, 0] = np.sin(angles)
    pairs = np.asarray([[0, 1], [1, 2]])
    links = frames[pairs[:, 0]].conj().swapaxes(-1, -2) @ frames[pairs[:, 1]]
    projectors = frames @ frames.conj().swapaxes(-1, -2)
    latent_h = -projectors + (np.eye(3) - projectors)
    fit = MACE(
        (grid,),
        ("H", "H"),
        geometry,
        3,
        chart_features=True,
        channels=2,
        max_ell=1,
        interactions=1,
        radial_basis=2,
        radial_mlp=(4,),
        cutoff=3.0,
    ).fit_spectral(
        coordinates,
        pairs,
        links,
        frames,
        pretrain_values=latent_h,
        selected_states=1,
        hidden=(4,),
        epochs=2,
        pretrain_epochs=1,
        seed=2,
    )

    assert fit.success
    assert fit.info["backend"] == "mace-spectral"
    assert len(fit.losses[-1]) == 4
    predicted = fit.neural_energy.predict(coordinates)
    np.testing.assert_allclose(
        predicted, predicted.conj().swapaxes(-1, -2), atol=1.0e-7
    )


@pytest.mark.skipif(not MACE_AVAILABLE, reason="mace-torch is not installed")
def test_mace_y_fit_builds_ttldr_feature_backend():
    from pyqed.dvr import DVR, SineDVR
    from pyqed.ldr import Coord
    from pyqed.namd.ttldr import TTLDR

    grid = np.linspace(-0.2, 0.2, 3)

    def geometry(q):
        distance = 0.8 + q[0]
        return np.asarray([[0.0, 0.0, -distance / 2], [0.0, 0.0, distance / 2]])

    coordinates = grid[:, None]
    energy = (0.1 * grid**2)[:, None, None]
    pairs = np.asarray([[0, 1], [1, 2]])
    links = np.full((2, 1, 1), 0.98)
    fit = MACE(
        (grid,),
        ("H", "H"),
        geometry,
        1,
        chart_features=True,
        channels=2,
        max_ell=1,
        interactions=1,
        radial_basis=2,
        radial_mlp=(4,),
        cutoff=3.0,
    ).fit_y(
        (coordinates, energy),
        coordinates,
        pairs,
        links,
        feature_rank=2,
        hidden=(4,),
        epochs=2,
        learning_rate=1.0e-2,
        tt_rank=3,
        tt_degree=2,
    )

    assert fit.success
    assert fit.info["backend"] == "mace-y"
    assert fit.info["feature_objective"] == "links-only"
    assert fit.info["energy_representation"] == "Y.H @ A @ Y"
    assert fit.info["structured_feature_samples"] is True
    assert fit.links is None
    assert fit.feature.output_shape_ == (2, 1)
    original_predict = fit._predict
    prediction_batches = []

    def recorded_predict(kind, values, *, axis=None):
        prediction_batches.append(len(np.atleast_2d(values)))
        return original_predict(kind, values, axis=axis)

    fit._predict = recorded_predict
    fit.distill_y(rank=2, degree=2, prediction_batch_size=2)
    fit._predict = original_predict
    assert max(prediction_batches) <= 2
    assert fit.info["distillation"]["prediction_batch_size"] == 2
    dynamics_grid = DVR.from_axes((SineDVR(-0.2, 0.2, 3),))
    dynamics = TTLDR(
        fit,
        grid=dynamics_grid,
        coord=Coord(to_cartesian=geometry, bounds=((-0.2, 0.2),)),
        keo=[(1.0, (np.eye(3),))],
    )
    assert dynamics.overlap_info["electronic_sampling"]["representation"] == (
        "mace-y"
    )
    anchored = fit.neural_feature.predict(coordinates[[fit.feature_anchor_]])[0]
    np.testing.assert_allclose(anchored, [[1.0], [0.0]], atol=1.0e-7)
    targets = fit.feature_targets_
    np.testing.assert_allclose(
        targets.conj().swapaxes(-1, -2) @ targets,
        np.ones((len(targets), 1, 1)),
        atol=2.0e-14,
    )
    assert fit.info["synchronization"]["isometry"] == "exact-polar-retraction"
    canonical = fit.predict_covariant(coordinates[:2])
    gauges = np.exp(1j * np.asarray([0.2, -0.4]))[:, None, None]
    rotated = fit.predict_covariant(coordinates[:2], gauges)
    np.testing.assert_allclose(
        rotated["feature"], canonical["feature"] @ gauges, atol=1.0e-7
    )
    np.testing.assert_allclose(
        rotated["energy"],
        gauges.conj().swapaxes(-1, -2) @ canonical["energy"] @ gauges,
        atol=1.0e-7,
    )
    warm = MACE(
        (grid,),
        ("H", "H"),
        geometry,
        1,
        chart_features=True,
        channels=2,
        max_ell=1,
        interactions=1,
        radial_basis=2,
        radial_mlp=(4,),
        cutoff=3.0,
    ).fit_y(
        (coordinates, energy),
        coordinates,
        pairs,
        links,
        feature_rank=2,
        hidden=(4,),
        epochs=1,
        initial_fit=fit,
        loss_scales={"energy": 0.25, "link": 0.5},
        distill=False,
    )
    assert warm.info["warm_started"] is True
    assert warm.info["loss_scales"]["energy"] == pytest.approx(0.25)
    assert warm.info["loss_scales"]["link"] == pytest.approx(0.5)
    warm.distill_y(
        rank=2,
        degree=2,
        method="cross",
        cross_points=3,
        cross_sweeps=2,
        cross_validation=4,
        validation_points=8,
        seed=3,
    )
    assert warm.info["distillation"]["method"] == "cross"
    assert warm.info["distillation"]["cross"]["energy"]["geometry_queries"] <= 3
    assert warm.info["distillation"]["cross"]["feature"]["geometry_queries"] <= 3
    identity = np.eye(3)
    driver = TTLDR.from_fit(
        fit,
        keo=((1.0, (identity,)),),
        overlap_rank=4,
        operator_rank=None,
    )
    assert driver.dims == (3, 1)
    dense = driver.hamiltonian.to_dense()
    np.testing.assert_allclose(dense, dense.conj().T, atol=1.0e-10)


@pytest.mark.skipif(not MACE_AVAILABLE, reason="mace-torch is not installed")
def test_mace_y_exchange_symmetry_is_exact_and_survives_checkpoint(tmp_path):
    grid = np.linspace(0.7, 0.9, 3)

    def geometry(q):
        return np.asarray(
            [[q[0], 0.0, 0.0], [0.0, 0.0, 0.0], [-q[1], 0.0, 0.0]]
        )

    mesh = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    coordinates = mesh[mesh[:, 0] >= mesh[:, 1]]
    pairs = np.column_stack((np.arange(len(coordinates) - 1), np.arange(1, len(coordinates))))
    links = np.broadcast_to(0.98 * np.eye(2), (len(pairs), 2, 2)).copy()
    hamiltonians = np.zeros((len(coordinates), 2, 2), dtype=complex)
    hamiltonians[:, 0, 0] = coordinates[:, 0] + coordinates[:, 1]
    hamiltonians[:, 1, 1] = hamiltonians[:, 0, 0] + 0.2
    hamiltonians[:, 0, 1] = hamiltonians[:, 1, 0] = (
        0.1 * (coordinates[:, 0] - coordinates[:, 1])
    )
    representation = np.diag([1.0, -1.0])
    fixed_representation = np.diag([1.0, -1.0])
    fit = MACE(
        (grid, grid),
        ("H", "O", "H"),
        geometry,
        2,
        channels=2,
        max_ell=1,
        interactions=1,
        radial_basis=2,
        radial_mlp=(4,),
        cutoff=3.0,
    ).fit_y(
        (coordinates, hamiltonians),
        coordinates,
        pairs,
        links,
        feature_rank=4,
        hidden=(4,),
        epochs=2,
        sync_steps=20,
        coordinate_exchange=representation,
        fixed_symmetry_representations=(fixed_representation,),
        energy_representation="direct",
        distill=False,
        seed=3,
    )
    point = np.asarray([[0.9, 0.7]])
    exchanged = point[:, ::-1]
    ambient = fit.coordinate_exchange_["ambient_representation"]
    np.testing.assert_allclose(
        fit.neural_feature.predict(exchanged),
        ambient @ fit.neural_feature.predict(point) @ representation,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        fit.neural_energy.predict(exchanged),
        representation @ fit.neural_energy.predict(point) @ representation,
        atol=2.0e-6,
    )
    fixed_ambient = fit.coordinate_exchange_["fixed_ambient_representations"][0]
    fixed_feature = fit.neural_feature.predict(point)
    fixed_energy = fit.neural_energy.predict(point)
    np.testing.assert_allclose(
        fixed_ambient @ fixed_feature @ fixed_representation,
        fixed_feature,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        fixed_energy @ fixed_representation,
        fixed_representation @ fixed_energy,
        atol=2.0e-6,
    )

    checkpoint = tmp_path / "exchange.pt"
    fit.save(checkpoint)
    restored = MACE.load(checkpoint, geometry, distill=False)
    np.testing.assert_allclose(
        restored.neural_feature.predict(exchanged),
        restored.coordinate_exchange_["ambient_representation"]
        @ restored.neural_feature.predict(point)
        @ representation,
        atol=2.0e-6,
    )
    assert len(restored.coordinate_exchange_["fixed_ambient_representations"]) == 1


@pytest.mark.skipif(not MACE_AVAILABLE, reason="mace-torch is not installed")
def test_mace_y_accepts_scattered_feature_samples():
    grid = np.linspace(-0.2, 0.2, 3)

    def geometry(q):
        distance = 0.8 + q[0]
        return np.asarray([[0.0, 0.0, -distance / 2], [0.0, 0.0, distance / 2]])

    coordinates = np.asarray([[-0.2], [-0.03], [0.2]])
    energy = (0.1 * coordinates[:, 0] ** 2)[:, None, None]
    fit = MACE(
        (grid,),
        ("H", "H"),
        geometry,
        1,
        chart_features=True,
        channels=2,
        max_ell=1,
        interactions=1,
        radial_basis=2,
        radial_mlp=(4,),
        cutoff=3.0,
    ).fit_y(
        (coordinates, energy),
        coordinates,
        np.asarray([[0, 1], [1, 2]]),
        np.full((2, 1, 1), 0.98),
        feature_rank=2,
        hidden=(4,),
        epochs=2,
        learning_rate=1.0e-2,
        distill=False,
    )
    assert fit.info["structured_feature_samples"] is False
    features = fit.neural_feature.predict(coordinates)
    assert features.shape == (3, 2, 1)
    np.testing.assert_allclose(
        features.conj().swapaxes(-1, -2) @ features,
        np.ones((3, 1, 1)),
        atol=2.0e-14,
    )


@pytest.mark.skipif(not MACE_AVAILABLE, reason="mace-torch is not installed")
def test_transferable_mace_handles_variable_atom_counts_and_conserves_charge(tmp_path):
    geometries = [
        np.asarray([[0.0, 0.0, -0.4], [0.0, 0.0, 0.4]]),
        np.asarray([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [-0.7, 0.0, 0.0]]),
    ]
    numbers = [(1, 1), (1, 8, 1)]
    hamiltonians = np.asarray([[[0.0]], [[0.1]]])
    charges = [np.asarray([[0.1, -0.1]]), np.asarray([[0.2, 0.6, 0.2]])]
    model = MACEStateModel(
        ("H", "O"),
        1,
        fidelities=("test/minimal",),
        manifolds=("ground",),
        hidden=(4,),
        channels=2,
        max_ell=1,
        interactions=1,
        radial_basis=2,
        radial_mlp=(4,),
        cutoff=3.0,
    ).fit(
        geometries,
        numbers,
        hamiltonians,
        atomic_charges=charges,
        molecular_charges=[0.0, 1.0],
        multiplicities=[1, 2],
        fidelities=["test/minimal"] * 2,
        manifolds=["ground"] * 2,
        epochs=2,
    )
    prediction = model.predict(
        geometries,
        numbers,
        molecular_charges=[0.0, 1.0],
        multiplicities=[1, 2],
        fidelities=["test/minimal"] * 2,
        manifolds=["ground"] * 2,
    )
    np.testing.assert_allclose(prediction["atomic_charges"][0].sum(axis=1), 0.0, atol=1e-6)
    np.testing.assert_allclose(prediction["atomic_charges"][1].sum(axis=1), 1.0, atol=1e-6)
    checkpoint = model.save(tmp_path / "state_model.pt")
    restored = MACEStateModel.load(checkpoint)
    restored_prediction = restored.predict(
        geometries,
        numbers,
        molecular_charges=[0.0, 1.0],
        multiplicities=[1, 2],
        fidelities=["test/minimal"] * 2,
        manifolds=["ground"] * 2,
    )
    np.testing.assert_allclose(
        restored_prediction["hamiltonian"], prediction["hamiltonian"], atol=1e-6
    )


@pytest.mark.skipif(not MACE_AVAILABLE, reason="mace-torch is not installed")
def test_mace_y_finite_noncommuting_group_is_exact_and_checkpointed(tmp_path):
    grid = np.linspace(-0.05, 0.05, 3)
    axes = (grid, grid, grid)
    coordinates = np.stack(
        np.meshgrid(*axes, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    angle = 2.0 * np.pi / 3.0
    rotation2 = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rotation3 = np.eye(3)
    rotation3[1:, 1:] = rotation2
    reflection3 = np.diag([1.0, 1.0, -1.0])
    reflection2 = np.diag([1.0, -1.0])
    coordinate_group = []
    electronic_group = []
    for power in range(3):
        coordinate_group.append(np.linalg.matrix_power(rotation3, power))
        electronic_group.append(np.linalg.matrix_power(rotation2, power))
    for power in range(3):
        coordinate_group.append(
            reflection3 @ np.linalg.matrix_power(rotation3, power)
        )
        electronic_group.append(
            reflection2 @ np.linalg.matrix_power(rotation2, power)
        )
    coordinate_group = np.asarray(coordinate_group)
    electronic_group = np.asarray(electronic_group, dtype=complex)
    ambient_group = np.asarray(
        [np.kron(np.eye(2), value) for value in electronic_group]
    )
    group = {
        "coordinate_representations": coordinate_group,
        "electronic_representations": electronic_group,
        "ambient_representations": ambient_group,
    }

    sigma_z = np.diag([1.0, -1.0])
    sigma_x = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    raw_energy = np.asarray(
        [q[0] * np.eye(2) + q[1] * sigma_z + q[2] * sigma_x for q in coordinates]
    )
    energy = np.zeros_like(raw_energy, dtype=complex)
    for coordinate_action, representation in zip(
        coordinate_group, electronic_group
    ):
        transformed = coordinates @ coordinate_action.T
        raw = np.asarray(
            [q[0] * np.eye(2) + q[1] * sigma_z + q[2] * sigma_x for q in transformed]
        )
        energy += representation.conj().T @ raw @ representation
    energy /= len(coordinate_group)

    shape = (3, 3, 3)
    pairs = []
    for index in np.ndindex(shape):
        for axis in range(3):
            if index[axis] + 1 >= shape[axis]:
                continue
            right = list(index)
            right[axis] += 1
            pairs.append(
                (np.ravel_multi_index(index, shape), np.ravel_multi_index(right, shape))
            )
    pairs = np.asarray(pairs)
    links = np.broadcast_to(0.995 * np.eye(2), (len(pairs), 2, 2)).copy()

    triangle = np.asarray(
        [[-0.5, -0.3, 0.0], [0.5, -0.3, 0.0], [0.0, 0.6, 0.0]]
    )

    def geometry(q):
        return triangle * (1.0 + q[0]) + np.asarray(
            [[q[1], q[2], 0.0], [-q[1], q[2], 0.0], [0.0, -2.0 * q[2], 0.0]]
        )

    fit = MACE(
        axes,
        ("H", "H", "H"),
        geometry,
        2,
        chart_features=True,
        channels=2,
        max_ell=1,
        interactions=1,
        correlation=1,
        radial_basis=2,
        radial_mlp=(4,),
        cutoff=3.0,
    ).fit_y(
        (coordinates, energy),
        coordinates,
        pairs,
        links,
        feature_rank=4,
        finite_group=group,
        hidden=(4,),
        epochs=2,
        sync_steps=5,
        distill=False,
        seed=4,
    )
    probe = np.asarray([[0.01, 0.02, -0.015], [-0.02, -0.01, 0.025]])
    base = fit.predict_covariant(probe)
    for coordinate_action, electronic, ambient in zip(
        coordinate_group, electronic_group, ambient_group
    ):
        transformed = fit.predict_covariant(probe @ coordinate_action.T)
        np.testing.assert_allclose(
            transformed["feature"],
            ambient @ base["feature"] @ electronic.conj().T,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            transformed["energy"],
            electronic @ base["energy"] @ electronic.conj().T,
            atol=2.0e-6,
        )
    gram = base["feature"].conj().swapaxes(-1, -2) @ base["feature"]
    np.testing.assert_allclose(
        gram, np.broadcast_to(np.eye(2), gram.shape), atol=2.0e-7
    )

    checkpoint = fit.save(tmp_path / "d3_y.pt")
    restored = MACE.load(checkpoint, geometry, distill=False)
    np.testing.assert_allclose(
        restored.predict_covariant(probe)["feature"], base["feature"], atol=2.0e-6
    )
    assert len(restored.finite_group_["coordinate_representations"]) == 6
