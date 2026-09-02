import numpy as np
import pickle
import importlib.util
import pytest

from pyqed.ldr import (
    AbInitioFit,
    Coord,
    ElectronicDatabase,
    FiniteGroupSamplingSymmetry,
    PhenolCASSCFOverlap,
    PhenolReflectionSymmetry,
    PhenolSACASSCFProvider,
    SamplingSymmetryImage,
)
from pyqed.ldr.sampling_symmetry import (
    detect_symmetry,
    infer_state_repr,
)
from pyqed.models.phenol_coordinates import (
    PHENOL_SPECIES,
    PhenolReactiveChart,
)


def toy_geometry(coordinates):
    radius, torsion = coordinates
    return np.asarray([[radius, np.cos(torsion), np.sin(torsion)]])


class _ToyMolecule:
    natom = 2
    charge = 0
    spin = 0
    basis = "toy"

    def __init__(self, distance=1.4):
        self.distance = float(distance)

    def atom_symbol(self, _index):
        return "H"

    def atom_charges(self):
        return np.ones(2, dtype=int)

    def atom_coords(self):
        return np.asarray(((0.0, 0.0, 0.0), (self.distance, 0.0, 0.0)))

    def set_geom(self, geometry):
        geometry = np.asarray(geometry, dtype=float)
        self.distance = float(np.linalg.norm(geometry[1] - geometry[0]))

    def build(self):
        return self


class _ToyFrame:
    def __init__(self, distance):
        angle = 0.15 * float(distance)
        self.vectors = np.asarray(((np.cos(angle),), (np.sin(angle),)))

    def overlap(self, other):
        return self.vectors.T @ other.vectors


class _ToyResult:
    def __init__(self, distance):
        self.e_tot = np.asarray((0.08 * (distance - 1.4) ** 2,))
        self._frame = _ToyFrame(distance)

    def frame(self):
        return self._frame


class _ToyScanner:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, molecule):
        self.calls.append(molecule.distance)
        return _ToyResult(molecule.distance)


class _ToyElectronic(_ToyResult):
    nstates = 1

    def __init__(self):
        super().__init__(1.4)
        self.mol = _ToyMolecule()
        self.calls = []

    def as_scanner(self, nstates=None):
        assert nstates == 1
        return _ToyScanner(self.calls)


def s3_coordinate_group():
    angle = 2.0 * np.pi / 3.0
    rotation = np.eye(3)
    rotation[1:, 1:] = (
        (np.cos(angle), -np.sin(angle)),
        (np.sin(angle), np.cos(angle)),
    )
    reflection = np.diag((1.0, 1.0, -1.0))
    return np.asarray(
        [np.linalg.matrix_power(rotation, power) for power in range(3)]
        + [
            reflection @ np.linalg.matrix_power(rotation, power)
            for power in range(3)
        ]
    )


def test_finite_group_sampling_keeps_one_representative_per_s3_orbit():
    group = s3_coordinate_group()
    symmetry = FiniteGroupSamplingSymmetry(
        group,
        name="H3+-S3",
        operations=("identity", "C3", "C3^2", "sigma", "sigma-C3", "sigma-C3^2"),
    )
    coordinate = np.asarray((0.1, 0.03, -0.04))
    orbit = np.einsum("gij,j->gi", group, coordinate)
    representatives, inverse, operations = symmetry.canonicalize_many(
        orbit, unique=True
    )
    assert representatives.shape == (1, 3)
    np.testing.assert_array_equal(inverse, np.zeros(6, dtype=int))
    assert len(operations) == 6
    assert np.arctan2(representatives[0, 2], representatives[0, 1]) >= 0.0
    assert np.arctan2(representatives[0, 2], representatives[0, 1]) <= np.pi / 3.0
    assert symmetry.representative_count(384) == 64

    grid = tuple(np.linspace(-1.0, 1.0, 3) for _ in range(3))
    with AbInitioFit(grid, 1, lambda index: index, symmetry=symmetry) as fit:
        assert fit.group == "H3+-S3"
        np.testing.assert_allclose(fit.coord_repr, group)


def test_loaded_fit_restores_finite_group_operations(tmp_path):
    group = s3_coordinate_group()
    symmetry = FiniteGroupSamplingSymmetry(
        group,
        name="H3+-S3",
        operations=("identity", "C3", "C3^2", "sigma", "sigma-C3", "sigma-C3^2"),
    )
    grid = tuple(np.linspace(-0.2, 0.2, 3) for _ in range(3))

    def builder(index):
        coordinates = np.asarray([axis[value] for axis, value in zip(grid, index)])
        return np.ones((1, 1)), np.asarray([coordinates @ coordinates])

    output = tmp_path / "fit"
    with AbInitioFit(
        grid,
        1,
        builder,
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=lambda left, right: left.T @ right,
        symmetry=symmetry,
    ) as fit:
        fit.run(rank=2, degrees=2, sweeps=2, validation=8, seed=3)
        fit.save(output)

    restored = AbInitioFit.load(output)
    assert restored.group == "H3+-S3"
    assert restored._symmetry is not None
    assert len(restored.orbit((0.1, 0.03, -0.04))) == 6
    assert restored.reduced_size(13) == 3


def test_detects_s3_from_molecule_and_coordinate_chart():
    root3 = np.sqrt(3.0)
    triangle = np.asarray(
        ((-0.5, -0.5 / root3, 0.0),
         (0.5, -0.5 / root3, 0.0),
         (0.0, 1.0 / root3, 0.0))
    )

    def geometry(q):
        breathing, x, y = np.asarray(q, dtype=float)
        strain = np.asarray(((x, y), (y, -x)))
        value = np.array(triangle, copy=True)
        value[:, :2] = triangle[:, :2] @ (
            (1.5 + breathing) * np.eye(2) + strain
        )
        return value.astype(np.float32)

    class Molecule:
        def atom_coords(self):
            return geometry((0.0, 0.0, 0.0))

        def atom_charges(self):
            return np.ones(3, dtype=int)

    coord = Coord(
        to_cartesian=geometry,
        bounds=((-0.4, 0.4), (-0.5, 0.5), (-0.5, 0.5)),
    )
    symmetry, validation = detect_symmetry(Molecule(), coord)

    assert validation["detected"]
    assert validation["group"] == "S3"
    assert validation["order"] == 6
    assert symmetry.order == 6
    np.testing.assert_allclose(
        symmetry.coordinate_representations[:, 0, 0], 1.0, atol=1.0e-8
    )


def test_native_fit_owns_automatic_symmetry(tmp_path):
    root3 = np.sqrt(3.0)
    triangle = np.asarray(
        ((-0.5, -0.5 / root3, 0.0),
         (0.5, -0.5 / root3, 0.0),
         (0.0, 1.0 / root3, 0.0))
    )

    def geometry(q):
        breathing, x, y = np.asarray(q, dtype=float)
        strain = np.asarray(((x, y), (y, -x)))
        value = np.array(triangle, copy=True)
        value[:, :2] = triangle[:, :2] @ (
            (1.5 + breathing) * np.eye(2) + strain
        )
        return value

    class Molecule:
        natom = 3
        charge = 1
        spin = 0

        def atom_coords(self):
            return geometry((0.0, 0.0, 0.0))

        def atom_charges(self):
            return np.ones(3, dtype=int)

        def atom_symbol(self, _index):
            return "H"

    class Electronic:
        e_tot = np.asarray((0.0,))
        nstates = 1
        mol = Molecule()

    coord = Coord(
        to_cartesian=geometry,
        bounds=((-0.4, 0.4), (-0.5, 0.5), (-0.5, 0.5)),
    )
    with AbInitioFit(
        Electronic(),
        coord=coord,
        states=(0,),
        database=tmp_path / "electronic.sqlite",
    ) as fit:
        assert fit.group == "S3"
        assert fit.coord_repr.shape == (6, 3, 3)
        assert fit.coord_irreps == ("A1", "E")
        assert fit.coord_blocks == ((0,), (1, 2))
        finite_group = fit.mace_group(feature_rank=2)
        assert fit.state_repr.shape == (6, 1, 1)
        assert finite_group["ambient_representations"].shape == (6, 2, 2)


@pytest.mark.skipif(
    importlib.util.find_spec("mace") is None, reason="mace-torch is not installed"
)
def test_native_build_uses_adaptive_mace_and_distills_to_ftt(tmp_path):
    coord = Coord(
        to_cartesian=lambda q: np.asarray(
            ((0.0, 0.0, 0.0), (float(q[0]), 0.0, 0.0))
        ),
        bounds=((1.1, 1.7),),
        periodic_axes=(0,),
    )
    fit = AbInitioFit(
        _ToyElectronic(),
        coord=coord,
        states=(0,),
        symmetry=False,
        database=tmp_path / "electronic.sqlite",
        fit_options={
            "model": "mace",
            "initial": 6,
            "batch": 2,
            "maximum": 6,
            "calibration": 4,
            "validation": 4,
            "ensemble": 2,
            "epochs": 2,
            "sync_steps": 5,
            "feature_rank": 2,
            "hidden": (4,),
            "encoder": {
                "channels": 2,
                "max_ell": 1,
                "interactions": 1,
                "correlation": 1,
                "radial_basis": 2,
                "radial_mlp": (4,),
                "cutoff": 3.0,
            },
            "rank": 4,
            "degrees": 2,
            "hamiltonian_atol": 1.0,
            "hamiltonian_rms": 1.0,
            "link_rtol": 1.0,
            "distill_rtol": 1.0,
            "coverage": 0.0,
        },
    ).build()

    assert fit.success
    assert fit.model == "mace-ftt"
    assert len(fit.ensemble) == 2
    assert fit.energy.output_shape_ == (1, 1)
    assert fit.feature.output_shape_ == (2, 1)
    assert fit.mace.periodic_axes == (0,)
    assert fit.acceptance["accepted"]
    assert "selected_member" in fit.validation
    assert "final" in fit.validation
    saved = fit.save(tmp_path / "fit")
    assert saved.paths["mace"].is_file()
    restored = AbInitioFit.load(
        tmp_path / "fit", geometry=coord.cartesian
    )
    assert restored.acceptance["accepted"]
    assert restored.mace.success
    assert restored.mace.periodic_axes == (0,)

    repeated_electronic = _ToyElectronic()
    repeated = AbInitioFit(
        repeated_electronic,
        coord=coord,
        states=(0,),
        symmetry=False,
        database=tmp_path / "electronic.sqlite",
    )
    reusable = repeated._database_coordinates()
    assert len(reusable) >= 2
    with repeated:
        repeated.continuous_fields(reusable[:2])
    assert repeated_electronic.calls == []


def test_infers_selected_state_representation_from_gauged_hamiltonians():
    coordinate = s3_coordinate_group()
    state = coordinate[:, 1:, 1:]
    random = np.random.default_rng(17)
    orbits = []
    for _ in range(3):
        value = random.normal(size=(2, 2))
        value = value + value.T
        orbits.append([operation @ value @ operation.T for operation in state])

    inferred, validation = infer_state_repr(coordinate, np.asarray(orbits))

    assert validation["maximum_covariance_error"] < 1.0e-12
    assert validation["closure_error"] < 1.0e-12
    for orbit in orbits:
        for operation, expected in zip(inferred, orbit):
            np.testing.assert_allclose(
                operation @ orbit[0] @ operation.T, expected, atol=1.0e-12
            )


def test_joint_pair_reduction_preserves_raw_nonunitary_link():
    group = s3_coordinate_group()
    operations = ("identity", "C3", "C3^2", "sigma", "sigma-C3", "sigma-C3^2")
    symmetry = FiniteGroupSamplingSymmetry(group, operations=operations)
    coordinates = np.asarray(
        ((0.1, 0.03, -0.04), (0.11, 0.035, -0.042))
    )
    reduced, pairs, pair_operations = symmetry.canonicalize_pairs(
        coordinates, ((0, 1),)
    )
    np.testing.assert_array_equal(pairs, ((0, 1),))
    np.testing.assert_allclose(
        np.linalg.norm(reduced[1] - reduced[0]),
        np.linalg.norm(coordinates[1] - coordinates[0]),
    )

    electronic = group[:, 1:, 1:]
    raw_link = np.asarray(((0.91, 0.12), (-0.08, 0.73)))
    transported = symmetry.transform_link(
        raw_link, pair_operations[0], electronic
    )
    np.testing.assert_allclose(
        np.linalg.svd(transported, compute_uv=False),
        np.linalg.svd(raw_link, compute_uv=False),
    )
    assert not np.allclose(transported.conj().T @ transported, np.eye(2))


def test_coordinate_only_group_never_aliases_untransformed_records():
    symmetry = FiniteGroupSamplingSymmetry(s3_coordinate_group())
    grid = tuple(np.asarray((-0.1, 0.0, 0.1)) for _ in range(3))
    built = []

    def builder(index):
        built.append(index)
        return index

    with AbInitioFit(grid, 1, builder, symmetry=symmetry) as fit:
        fit.frames.get_many(((1, 2, 0), (1, 0, 2)))
        assert fit._record_symmetry is None
        assert len(built) == 2


def test_sampling_symmetry_builds_and_stores_only_one_representative(tmp_path):
    grid = (np.asarray([0.9, 1.0, 1.1]), np.asarray([-0.2, 0.0, 0.2]))
    built = []

    def electronic(sample):
        built.append(sample)
        return {
            "geometry": np.asarray(sample["geometry"]),
            "energies": np.asarray([sample["coordinates"][0]]),
        }

    with AbInitioFit(
        grid,
        1,
        electronic=electronic,
        geometry=toy_geometry,
        symmetry=PhenolReflectionSymmetry(),
        database=tmp_path / "electronic.sqlite",
        protocol={"system": "toy", "basis": "sto-3g"},
        run_id="reflection",
    ) as fit:
        negative, positive = fit.frames.get_many(((1, 0), (1, 2)))
        assert len(built) == 1
        assert built[0]["index"] == (1, 2)
        assert built[0]["coordinates"] == (1.0, 0.2)
        np.testing.assert_allclose(negative["geometry"], toy_geometry((1.0, -0.2)))
        np.testing.assert_allclose(positive["geometry"], toy_geometry((1.0, 0.2)))
        assert fit.database.stats["records"] == 1
        assert fit.frames.stats["built"] == 1
        assert fit.frames.record_id((1, 0)) == fit.frames.record_id((1, 2))

        run = fit.database.run("reflection")
        assert len(run["records"]) == 2
        assert {item["record_id"] for item in run["records"]} == {
            fit.frames.record_id((1, 2))
        }
        by_index = {tuple(item["grid_index"]): item for item in run["records"]}
        assert by_index[(1, 0)]["sample"]["sampling_symmetry"]["operation"] == "sigma_xy"
        assert by_index[(1, 2)]["sample"]["sampling_symmetry"]["operation"] == "identity"


def test_sampling_symmetry_overlap_cache_distinguishes_views_of_one_record(tmp_path):
    grid = (np.asarray([0.9, 1.0, 1.1]), np.asarray([-0.2, 0.0, 0.2]))
    database = tmp_path / "electronic.sqlite"
    protocol = {"system": "toy", "basis": "sto-3g"}
    overlap_protocol = {"algorithm": "frame-dot", "version": 1}

    def builder(index):
        radius = grid[0][index[0]]
        torsion = grid[1][index[1]]
        return {
            "geometry": toy_geometry((radius, torsion)),
            "energies": np.asarray([0.0, 0.5]),
            "dipoles": np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        }

    def frame(record):
        return np.asarray(record["dipoles"]).T

    def overlap(left, right):
        return left.T @ right

    with AbInitioFit(
        grid,
        2,
        builder,
        frame=frame,
        energies=lambda record: record["energies"],
        overlap=overlap,
        overlap_protocol=overlap_protocol,
        geometry=toy_geometry,
        symmetry=PhenolReflectionSymmetry(),
        database=database,
        protocol=protocol,
    ) as fit:
        positive = (1, 2)
        negative = (1, 0)
        records = dict(
            zip((positive, negative), fit.frames.get_many((positive, negative)))
        )
        same = fit.oracle._raw_overlap(positive, positive, records)
        reflected = fit.oracle._raw_overlap(positive, negative, records)
        np.testing.assert_allclose(same, np.eye(2))
        np.testing.assert_allclose(reflected, np.diag((1.0, -1.0)))
        assert fit.database.stats["records"] == 1
        assert fit.database.stats["overlaps"] == 2

    def forbidden_overlap(_left, _right):
        raise AssertionError("operation-aware persistent overlap was not reused")

    with AbInitioFit(
        grid,
        2,
        lambda _index: (_ for _ in ()).throw(
            AssertionError("canonical electronic record was not reused")
        ),
        frame=frame,
        energies=lambda record: record["energies"],
        overlap=forbidden_overlap,
        overlap_protocol=overlap_protocol,
        geometry=toy_geometry,
        symmetry=PhenolReflectionSymmetry(),
        database=database,
        protocol=protocol,
    ) as fit:
        positive = (1, 2)
        negative = (1, 0)
        records = dict(
            zip((positive, negative), fit.frames.get_many((positive, negative)))
        )
        np.testing.assert_allclose(
            fit.oracle._raw_overlap(positive, negative, records),
            np.diag((1.0, -1.0)),
        )
        assert fit.oracle.persistent_overlap_hits == 1


def test_explicit_points_and_pairs_expand_to_complete_reflection_orbits():
    grid = (
        np.asarray([0.9, 1.0, 1.1]),
        np.asarray([-0.4, -0.2, 0.0, 0.2, 0.4]),
    )
    with AbInitioFit(
        grid,
        1,
        lambda index: index,
        symmetry=PhenolReflectionSymmetry(),
    ) as fit:
        assert fit.expand_points(((0, 2), (1, 3), (2, 4))) == (
            (0, 2),
            (1, 3),
            (1, 1),
            (2, 4),
            (2, 0),
        )
        assert fit.expand_pairs((((0, 2), (1, 3)),)) == (
            ((0, 2), (1, 3)),
            ((0, 2), (1, 1)),
        )


def test_phenol_reflection_flips_torsion_and_16a_as_one_operation():
    symmetry = PhenolReflectionSymmetry(torsion_axis=1, odd_axes=(1, 3))
    positive = (1.1, 0.2, 1.9, -0.3, 0.1)
    negative = (1.1, -0.2, 1.9, 0.3, 0.1)

    image = symmetry.resolve(negative)
    np.testing.assert_allclose(image.representative_coordinates, positive)
    assert image.operation == "sigma_xy"
    assert symmetry.images(positive) == (positive, negative)

    planar_negative = (1.1, 0.0, 1.9, -0.3, 0.1)
    planar_image = symmetry.resolve(planar_negative)
    np.testing.assert_allclose(
        planar_image.representative_coordinates,
        (1.1, 0.0, 1.9, 0.3, 0.1),
    )


def test_phenol_reflection_transports_molecular_orbitals_in_the_ao_basis():
    from pyscf import gto

    chart = PhenolReactiveChart()
    coordinate = np.array(chart.equilibrium, copy=True)
    coordinate[1] = 0.23
    representative = chart.geometry(coordinate)
    requested_coordinate = coordinate.copy()
    requested_coordinate[1] *= -1.0
    requested = chart.geometry(requested_coordinate)
    basis = "sto-3g"

    positive = gto.M(
        atom=list(zip(PHENOL_SPECIES, representative)),
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=0,
        verbose=0,
    )
    negative = gto.M(
        atom=list(zip(PHENOL_SPECIES, requested)),
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=0,
        verbose=0,
    )
    rng = np.random.default_rng(7)
    coefficients = np.eye(positive.nao)
    ci = rng.normal(size=(2, 4, 4))
    symmetry = PhenolReflectionSymmetry()
    image = SamplingSymmetryImage(tuple(coordinate), "sigma_xy")
    transformed = symmetry.transform_record(
        {
            "geometry": representative,
            "mo_coeff": coefficients,
            "ci": ci,
            "energies": np.asarray([-1.0, -0.8]),
        },
        image,
        representative_geometry=representative,
        requested_geometry=requested,
        protocol={"basis": basis},
    )

    positive_metric = coefficients.T @ positive.intor_symmetric("int1e_ovlp") @ coefficients
    reflected_coefficients = transformed["mo_coeff"]
    negative_metric = (
        reflected_coefficients.T
        @ negative.intor_symmetric("int1e_ovlp")
        @ reflected_coefficients
    )
    np.testing.assert_allclose(negative_metric, positive_metric, atol=1.0e-11)
    positive_hcore = positive.intor_symmetric("int1e_kin") + positive.intor_symmetric(
        "int1e_nuc"
    )
    negative_hcore = negative.intor_symmetric("int1e_kin") + negative.intor_symmetric(
        "int1e_nuc"
    )
    np.testing.assert_allclose(
        reflected_coefficients.T @ negative_hcore @ reflected_coefficients,
        positive_hcore,
        atol=1.0e-10,
    )
    np.testing.assert_array_equal(transformed["ci"], ci)
    np.testing.assert_array_equal(transformed["energies"], (-1.0, -0.8))
    np.testing.assert_allclose(transformed["geometry"], requested)


def test_phenol_record_overlap_is_signed_and_includes_the_ao_metric():
    from pyscf import gto

    chart = PhenolReactiveChart()
    geometry = chart.geometry(chart.equilibrium)
    molecule = gto.M(
        atom=list(zip(PHENOL_SPECIES, geometry)),
        unit="Angstrom",
        basis="sto-3g",
        charge=0,
        spin=0,
        verbose=0,
    )
    metric = molecule.intor_symmetric("int1e_ovlp")
    values, vectors = np.linalg.eigh(metric)
    orbitals = (vectors * values**-0.5) @ vectors.T
    left_ci = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    right_ci = left_ci.copy()
    right_ci[0] *= -1.0
    common = {"geometry": geometry, "mo_coeff": orbitals}
    overlap = PhenolCASSCFOverlap(
        basis="sto-3g", ncore=1, ncas=2, nelecas=2
    )
    block = overlap(
        {**common, "ci": left_ci},
        {**common, "ci": right_ci},
    )
    np.testing.assert_allclose(block, np.diag((-1.0, 1.0)), atol=1.0e-10)


def test_phenol_provider_selects_nearest_qualified_protocol_record(tmp_path):
    chart = PhenolReactiveChart()
    protocol = {
        "basis": "sto-3g",
        "active_space": {"electrons": 2, "orbitals": 2},
        "state_average": {"roots": 1, "weights": [1.0]},
    }
    database = ElectronicDatabase(tmp_path / "electronic.sqlite")
    for radius in (0.9, 1.2):
        coordinate = chart.equilibrium.copy()
        coordinate[0] = radius
        geometry = chart.geometry(coordinate)
        database.put(
            {"geometry": geometry, "protocol": protocol},
            {
                "geometry": geometry,
                "mo_coeff": np.eye(2),
                "ci": np.ones((1, 1)),
                "scf_converged": np.asarray(True),
                "orbital_relaxed": np.asarray(True),
            },
        )
    target = chart.equilibrium.copy()
    target[0] = 1.15
    provider = PhenolSACASSCFProvider(database, protocol)
    nearest = provider.nearest(
        {"geometry": chart.geometry(target), "coordinates": target[:2]}
    )
    assert nearest is not None
    assert np.isclose(nearest[2][0], 1.2)
    restored = pickle.loads(pickle.dumps(provider))
    restored_nearest = restored.nearest(
        {"geometry": chart.geometry(target), "coordinates": target[:3]}
    )
    assert restored_nearest is not None
    assert np.isclose(restored_nearest[2][0], 1.2)
    restored.close()
    database.close()
