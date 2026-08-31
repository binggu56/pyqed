import numpy as np
import pickle

from pyqed.ldr import (
    AbInitioFit,
    ElectronicDatabase,
    PhenolCASSCFOverlap,
    PhenolReflectionSymmetry,
    PhenolSACASSCFProvider,
    SamplingSymmetryImage,
)
from pyqed.models.phenol_coordinates import (
    PHENOL_SPECIES,
    PhenolReactiveChart,
)


def toy_geometry(coordinates):
    radius, torsion = coordinates
    return np.asarray([[radius, np.cos(torsion), np.sin(torsion)]])


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
