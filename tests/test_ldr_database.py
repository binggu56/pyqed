import hashlib
import pickle
import sqlite3
import zlib

import numpy as np
import pytest

from pyqed.ldr import AbInitioFit, ElectronicDatabase
from pyqed.ldr.oracle import Frames


def test_electronic_database_round_trip_and_protocol_identity(tmp_path):
    path = tmp_path / "electronic.sqlite"
    first = {
        "geometry": [["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 1.0]]],
        "protocol": {"method": "SA-CASSCF", "basis": "sto-3g", "roots": 2},
    }
    changed = {
        **first,
        "protocol": {**first["protocol"], "basis": "6-31g"},
    }
    record = {"energies": np.asarray([-1.0, -0.8]), "converged": True}

    with ElectronicDatabase(path) as database:
        key, inserted = database.put(first, record, metadata={"source": "test"})
        assert inserted
        assert key == database.identifier(first)
        restored = database.get(first)
        np.testing.assert_array_equal(restored["energies"], record["energies"])
        assert database.get(changed) is None
        assert database.stats["records"] == 1
        assert database.entries()[0]["metadata"] == {"source": "test"}


def test_frames_reuse_database_record_across_different_grid_indices(tmp_path):
    path = tmp_path / "electronic.sqlite"
    protocol = {"system": "toy", "method": "CASCI"}
    first_grid = np.asarray([-1.0, 0.0, 1.0])
    second_grid = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])

    with ElectronicDatabase(path) as database:
        first = Frames(
            (3,),
            lambda index: {"coordinate": first_grid[index[0]]},
            database=database,
            database_key=lambda index: {
                "geometry": [float(first_grid[index[0]])],
                "protocol": protocol,
            },
        )
        assert first.get((1,))["coordinate"] == 0.0
        assert first.stats["built"] == 1

        def must_not_build(_index):
            raise AssertionError("a reusable geometry was recalculated")

        second = Frames(
            (5,),
            must_not_build,
            database=database,
            database_key=lambda index: {
                "geometry": [float(second_grid[index[0]])],
                "protocol": protocol,
            },
        )
        assert second.get((2,))["coordinate"] == 0.0
        assert second.stats["database_hits"] == 1
        assert second.stats["built"] == 0


def test_abinitio_fit_exposes_geometry_and_reuses_it_across_grids(tmp_path):
    path = tmp_path / "phenol.sqlite"
    protocol = {
        "system": "phenol",
        "method": "SA-CASSCF(6,6)",
        "basis": "sto-3g",
        "roots": 6,
        "spin_constraint": "fix_spin",
    }

    def geometry(coordinates):
        return [["O", [0.0, 0.0, 0.0]], ["H", [coordinates[0], 0.0, 0.0]]]

    first_grid = np.asarray([0.9, 1.0, 1.1])
    with AbInitioFit(
        (first_grid,),
        6,
        lambda index: {"distance": first_grid[index[0]], "ci": np.eye(6)},
        database=path,
        protocol=protocol,
        geometry=geometry,
    ) as fit:
        record = fit.frames.get((1,))
        assert record["distance"] == 1.0
        assert fit.sample((1,))["coordinates"] == (1.0,)
        assert fit.sample_geometry((1,))[1][1][0] == 1.0

    second_grid = np.asarray([0.8, 0.9, 1.0, 1.1, 1.2])

    def must_not_build(_index):
        raise AssertionError("the stored phenol calculation was not reused")

    with AbInitioFit(
        (second_grid,),
        6,
        must_not_build,
        database=path,
        protocol=protocol,
        geometry=geometry,
    ) as fit:
        assert fit.frames.get((2,))["distance"] == 1.0
        assert fit.frames.stats["database_hits"] == 1
    assert fit.stats["database"]["records"] == 1


def test_existing_point_cache_is_migrated_into_database(tmp_path):
    cache = tmp_path / "points"
    database_path = tmp_path / "electronic.sqlite"
    original = Frames((3,), lambda index: (index, 2.0), cache_dir=cache)
    assert original.get((1,)) == ((1,), 2.0)

    with ElectronicDatabase(database_path) as database:
        migrated = Frames(
            (3,),
            cache_dir=cache,
            database=database,
            database_key=lambda index: {
                "geometry": [float(index[0])],
                "protocol": {"method": "toy"},
            },
        )
        assert migrated.get((1,)) == ((1,), 2.0)
        assert migrated.stats["database_migrations"] == 1
        assert database.stats["records"] == 1


def test_abinitio_electronic_builder_receives_complete_sample():
    seen = []

    def geometry(coordinates):
        return [["H", [coordinates[0], 0.0, 0.0]]]

    def electronic(sample):
        seen.append(sample)
        return {"energy": sample["coordinates"][0] ** 2}

    with AbInitioFit(
        (np.asarray([-1.0, 0.0, 1.0]),),
        1,
        electronic=electronic,
        geometry=geometry,
    ) as fit:
        assert fit.frames.get((2,))["energy"] == 1.0

    assert seen == [
        {
            "index": (2,),
            "coordinates": (1.0,),
            "geometry": [["H", [1.0, 0.0, 0.0]]],
        }
    ]


def test_database_uses_sharded_external_npz_objects(tmp_path):
    path = tmp_path / "electronic.sqlite"
    specification = {"geometry": [1.2], "protocol": {"method": "CASCI"}}
    record = {
        "energies": np.asarray([-1.0, -0.7]),
        "ci": np.arange(24.0).reshape(2, 12),
        "converged": np.asarray(True),
    }
    with ElectronicDatabase(path) as database:
        database.put(specification, record)
        entry = database.entries()[0]
        object_path = tmp_path / "objects" / entry["object_hash"][:2] / (
            entry["object_hash"][2:4]
        ) / f"{entry['object_hash']}.npz"
        assert entry["object_path"] == str(object_path)
        assert object_path.is_file()
        assert database.stats["catalog_bytes"] < database.stats["stored_bytes"] * 5
        with np.load(object_path, allow_pickle=False) as archive:
            assert "__manifest__" in archive
            assert any(
                np.asarray(archive[name]).shape == (2, 12)
                for name in archive.files
            )


def test_calculation_claim_prevents_duplicate_work(tmp_path):
    path = tmp_path / "electronic.sqlite"
    specification = {"geometry": [0.0], "protocol": {"method": "CASSCF"}}
    with ElectronicDatabase(path) as first, ElectronicDatabase(path) as second:
        assert first.claim(specification, "worker-a") == "acquired"
        assert second.claim(specification, "worker-b") == "busy"
        assert second.active_claims()[0]["owner"] == "worker-a"
        first.put(specification, {"energy": -1.0})
        assert second.claim(specification, "worker-b") == "complete"
        assert second.active_claims() == []


def test_successful_points_survive_a_later_builder_failure(tmp_path):
    path = tmp_path / "electronic.sqlite"
    protocol = {"method": "SA-CASSCF"}

    def builder(index):
        if index == (1,):
            raise RuntimeError("failed point")
        return {"index": index}

    with ElectronicDatabase(path) as database:
        frames = Frames(
            (3,),
            builder,
            database=database,
            database_key=lambda index: {
                "geometry": [float(index[0])],
                "protocol": protocol,
            },
        )
        with pytest.raises(RuntimeError, match="failed point"):
            frames.get_many([(0,), (1,), (2,)])
        assert database.stats["records"] == 1
        assert database.get(
            {"geometry": [0.0], "protocol": protocol}
        ) == {"index": (0,)}
        assert database.active_claims() == []


def test_abinitio_runs_record_built_and_reused_samples(tmp_path):
    path = tmp_path / "electronic.sqlite"
    grid = np.asarray([-1.0, 0.0, 1.0])
    protocol = {"system": "toy", "method": "CASCI"}
    with AbInitioFit(
        (grid,),
        1,
        lambda index: {"coordinate": grid[index[0]]},
        database=path,
        protocol=protocol,
        run_id="first-fit",
    ) as first:
        first.frames.get((1,))
        assert first.database.run("first-fit")["records"][0]["source"] == "built"

    with AbInitioFit(
        (grid,),
        1,
        lambda _index: pytest.fail("database record was recalculated"),
        database=path,
        protocol=protocol,
        run_id="second-fit",
    ) as second:
        second.frames.get((1,))
        run = second.database.run("second-fit")
        assert run["records"][0]["source"] == "database"
        assert run["records"][0]["sample"]["coordinates"] == [0.0]


def test_overlap_blocks_are_persistent_and_reverse_by_adjoint(tmp_path):
    path = tmp_path / "electronic.sqlite"
    first = {"geometry": [0.0], "protocol": {"method": "CASCI"}}
    second = {"geometry": [1.0], "protocol": {"method": "CASCI"}}
    overlap_protocol = {"algorithm": "active-space-overlap", "version": 1}
    block = np.asarray([[0.9, 0.1j], [-0.2j, 0.8]])
    with ElectronicDatabase(path) as database:
        left, _ = database.put(first, {"energy": 0.0})
        right, _ = database.put(second, {"energy": 1.0})
        database.put_overlap(left, right, overlap_protocol, block)
        np.testing.assert_array_equal(
            database.get_overlap(left, right, overlap_protocol), block
        )
        np.testing.assert_array_equal(
            database.get_overlap(right, left, overlap_protocol), block.conj().T
        )
        assert database.get_overlap(left, right, {"version": 2}) is None
        assert database.stats["overlaps"] == 1


def test_abinitio_reuses_persistent_raw_overlaps(tmp_path):
    path = tmp_path / "electronic.sqlite"
    grid = np.asarray([-1.0, 0.0, 1.0])
    protocol = {"system": "toy", "method": "CASCI"}
    overlap_protocol = {"algorithm": "frame-dot", "version": 1}
    overlap_calls = []

    def builder(index):
        angle = 0.1 * index[0]
        frame = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return frame, np.asarray([index[0], index[0] + 0.5])

    def overlap(left, right):
        overlap_calls.append(1)
        return left.T @ right

    with AbInitioFit(
        (grid,),
        2,
        builder,
        database=path,
        protocol=protocol,
        overlap_protocol=overlap_protocol,
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=overlap,
    ) as first:
        expected = first.oracle.hamiltonian_many([(0,), (1,)])
        first.oracle.overlap_many([((0,), (1,))])
    calls = len(overlap_calls)
    assert calls > 0

    def forbidden_overlap(_left, _right):
        raise AssertionError("persistent overlap was recomputed")

    with AbInitioFit(
        (grid,),
        2,
        lambda _index: pytest.fail("database record was recalculated"),
        database=path,
        protocol=protocol,
        overlap_protocol=overlap_protocol,
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=forbidden_overlap,
    ) as second:
        actual = second.oracle.hamiltonian_many([(0,), (1,)])
        second.oracle.overlap_many([((0,), (1,))])
        assert second.oracle.persistent_overlap_hits >= calls
    np.testing.assert_allclose(actual, expected)


def test_schema_one_database_migrates_payloads_to_objects(tmp_path):
    path = tmp_path / "legacy.sqlite"
    specification = {"geometry": [0.5], "protocol": {"method": "CASCI"}}
    record = {"energies": np.asarray([-1.0, -0.5]), "ci": np.eye(2)}
    serialized = pickle.dumps(record, pickle.HIGHEST_PROTOCOL)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO metadata VALUES ('schema_version', '1');
        CREATE TABLE records (
            id TEXT PRIMARY KEY,
            specification TEXT NOT NULL UNIQUE,
            payload BLOB NOT NULL,
            checksum TEXT NOT NULL,
            codec TEXT NOT NULL,
            metadata TEXT NOT NULL,
            created_at TEXT NOT NULL,
            accessed_at TEXT NOT NULL,
            accesses INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    connection.execute(
        "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            ElectronicDatabase.identifier(specification),
            json_for_test(specification),
            zlib.compress(serialized),
            hashlib.sha256(serialized).hexdigest(),
            "pickle+zlib",
            "{}",
            "created",
            "accessed",
            2,
        ),
    )
    connection.commit()
    connection.close()

    with ElectronicDatabase(path) as database:
        restored = database.get(specification)
        np.testing.assert_array_equal(restored["ci"], record["ci"])
        entry = database.entries()[0]
        assert entry["object_path"].endswith(".npz")
        assert database.connection.execute(
            "SELECT value FROM metadata WHERE name = 'schema_version'"
        ).fetchone()[0] == "2"


def json_for_test(value):
    from pyqed.ldr.database import canonical_json

    return canonical_json(value)
