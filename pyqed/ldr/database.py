"""Persistent electronic-record catalog and content-addressed object store."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import pickle
import sqlite3
import tempfile
import time
import zlib

import numpy as np


def _canonical(value):
    if isinstance(value, dict):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("database keys and metadata must contain finite floats")
        return 0.0 if value == 0.0 else value
    if isinstance(value, complex):
        return {"real": _canonical(value.real), "imag": _canonical(value.imag)}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"cannot encode {type(value).__name__} in an electronic-record key")


def canonical_json(value):
    """Return a deterministic JSON representation for keys and metadata."""

    return json.dumps(
        _canonical(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pickle_array(value):
    return np.frombuffer(pickle.dumps(value, pickle.HIGHEST_PROTOCOL), dtype=np.uint8)


def _record_arrays(record):
    arrays = {}
    if not isinstance(record, dict):
        arrays["payload"] = _pickle_array(record)
        manifest = {"kind": "pickle", "field": "payload"}
    else:
        fields = []
        for number, (name, value) in enumerate(record.items()):
            field = f"field_{number:06d}"
            if isinstance(value, np.ndarray) and value.dtype != object:
                arrays[field] = value
                storage = "array"
            elif isinstance(value, np.generic):
                arrays[field] = np.asarray(value)
                storage = "numpy-scalar"
            else:
                arrays[field] = _pickle_array(value)
                storage = "pickle"
            fields.append({"name": str(name), "field": field, "storage": storage})
        manifest = {"kind": "mapping", "fields": fields}
    arrays["__manifest__"] = np.asarray(canonical_json(manifest))
    return arrays


def _load_record(path):
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(np.asarray(archive["__manifest__"]).item()))
        if manifest["kind"] == "pickle":
            return pickle.loads(np.asarray(archive[manifest["field"]]).tobytes())
        record = {}
        for item in manifest["fields"]:
            value = np.asarray(archive[item["field"]])
            if item["storage"] == "array":
                value = np.array(value, copy=True)
            elif item["storage"] == "numpy-scalar":
                value = value[()]
            elif item["storage"] == "pickle":
                value = pickle.loads(value.tobytes())
            else:
                raise RuntimeError(
                    f"unsupported electronic-record storage {item['storage']!r}"
                )
            record[item["name"]] = value
        return record


class ElectronicDatabase:
    """SQLite catalog with content-addressed ``.npz`` electronic records.

    Records are addressed by a JSON-compatible specification, normally
    ``{"geometry": ..., "protocol": ...}``. The protocol must include every
    setting that changes the electronic result, including the selected
    multiconfigurational solution branch.
    """

    schema_version = 2

    def __init__(self, path, *, object_dir=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.object_dir = (
            self.path.parent / "objects" if object_dir is None else Path(object_dir)
        )
        self.object_dir.mkdir(parents=True, exist_ok=True)
        (self.object_dir / ".tmp").mkdir(exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=60.0)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(name TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        version = self.connection.execute(
            "SELECT value FROM metadata WHERE name = 'schema_version'"
        ).fetchone()
        if version is None:
            self._create_schema()
            self.connection.execute(
                "INSERT INTO metadata(name, value) VALUES('schema_version', ?)",
                (str(self.schema_version),),
            )
            self.connection.commit()
        elif int(version[0]) == 1:
            self._migrate_v1()
        elif int(version[0]) == self.schema_version:
            self._create_schema()
            self.connection.commit()
        else:
            raise RuntimeError(
                f"electronic database schema {version[0]} is not supported"
            )
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.overlap_hits = 0
        self.overlap_misses = 0
        self.overlap_writes = 0
        self._closed_stats = None

    def _create_schema(self):
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                specification TEXT NOT NULL UNIQUE,
                object_hash TEXT NOT NULL,
                object_path TEXT NOT NULL,
                object_format TEXT NOT NULL,
                metadata TEXT NOT NULL,
                payload_bytes INTEGER NOT NULL,
                stored_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                accesses INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS records_object_hash ON records(object_hash);
            CREATE INDEX IF NOT EXISTS records_accessed_at ON records(accessed_at);
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS run_records (
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                grid_index TEXT NOT NULL,
                record_id TEXT NOT NULL REFERENCES records(id),
                sample TEXT NOT NULL,
                source TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                PRIMARY KEY(run_id, grid_index)
            );
            CREATE INDEX IF NOT EXISTS run_records_record ON run_records(record_id);
            CREATE TABLE IF NOT EXISTS claims (
                record_id TEXT PRIMARY KEY,
                specification TEXT NOT NULL,
                owner TEXT NOT NULL,
                claimed_at TEXT NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS claims_expires_at ON claims(expires_at);
            CREATE TABLE IF NOT EXISTS overlaps (
                id TEXT PRIMARY KEY,
                left_record_id TEXT NOT NULL REFERENCES records(id),
                right_record_id TEXT NOT NULL REFERENCES records(id),
                protocol TEXT NOT NULL,
                shape TEXT NOT NULL,
                dtype TEXT NOT NULL,
                payload BLOB NOT NULL,
                checksum TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                accesses INTEGER NOT NULL DEFAULT 0,
                UNIQUE(left_record_id, right_record_id, protocol)
            );
            CREATE INDEX IF NOT EXISTS overlaps_pair
                ON overlaps(left_record_id, right_record_id);
            """
        )

    def _migrate_v1(self):
        rows = self.connection.execute(
            """
            SELECT id, specification, payload, checksum, codec, metadata,
                   created_at, accessed_at, accesses
            FROM records
            """
        ).fetchall()
        self.connection.execute("ALTER TABLE records RENAME TO records_v1")
        self._create_schema()
        for row in rows:
            key, specification, payload, checksum, codec, metadata = row[:6]
            created, accessed, accesses = row[6:]
            if codec != "pickle+zlib":
                raise RuntimeError(f"cannot migrate record codec {codec!r}")
            serialized = zlib.decompress(payload)
            if hashlib.sha256(serialized).hexdigest() != checksum:
                raise IOError(f"electronic record {key} failed its migration checksum")
            record = pickle.loads(serialized)
            object_info = self._write_object(record)
            self.connection.execute(
                """
                INSERT INTO records(
                    id, specification, object_hash, object_path, object_format,
                    metadata, payload_bytes, stored_bytes, created_at,
                    accessed_at, accesses
                ) VALUES (?, ?, ?, ?, 'npz-v1', ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    specification,
                    object_info["hash"],
                    object_info["path"],
                    metadata,
                    len(serialized),
                    object_info["bytes"],
                    created,
                    accessed,
                    accesses,
                ),
            )
        self.connection.execute("DROP TABLE records_v1")
        self._create_schema()
        self.connection.execute(
            "UPDATE metadata SET value = ? WHERE name = 'schema_version'",
            (str(self.schema_version),),
        )
        self.connection.commit()
        self.connection.execute("VACUUM")
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    @staticmethod
    def identifier(specification):
        encoded = canonical_json(specification).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _stored_path(self, path):
        try:
            return str(path.relative_to(self.path.parent))
        except ValueError:
            return str(path)

    def _resolved_path(self, stored):
        path = Path(stored)
        return path if path.is_absolute() else self.path.parent / path

    def _write_object(self, record):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="record_", suffix=".npz", dir=self.object_dir / ".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, **_record_arrays(record))
                stream.flush()
                os.fsync(stream.fileno())
            checksum = _digest_file(temporary)
            destination = (
                self.object_dir
                / checksum[:2]
                / checksum[2:4]
                / f"{checksum}.npz"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                temporary.unlink()
            else:
                os.replace(temporary, destination)
            return {
                "hash": checksum,
                "path": self._stored_path(destination),
                "bytes": destination.stat().st_size,
            }
        finally:
            if temporary.exists():
                temporary.unlink()

    def get(self, specification):
        """Return a stored record, or ``None`` when the key is absent."""

        key = self.identifier(specification)
        row = self.connection.execute(
            "SELECT object_hash, object_path, object_format FROM records WHERE id = ?",
            (key,),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        checksum, stored_path, object_format = row
        if object_format != "npz-v1":
            raise RuntimeError(f"unsupported electronic object format {object_format!r}")
        path = self._resolved_path(stored_path)
        if not path.is_file():
            raise FileNotFoundError(f"electronic object {path} is missing")
        if _digest_file(path) != checksum:
            raise IOError(f"electronic object {path} failed its checksum")
        self.connection.execute(
            "UPDATE records SET accessed_at = ?, accesses = accesses + 1 WHERE id = ?",
            (_now(), key),
        )
        self.connection.commit()
        self.hits += 1
        return _load_record(path)

    def put(self, specification, record, *, metadata=None):
        """Store one complete record without replacing an existing calculation."""

        key_json = canonical_json(specification)
        key = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
        if self.connection.execute(
            "SELECT 1 FROM records WHERE id = ?", (key,)
        ).fetchone():
            return key, False
        object_info = self._write_object(record)
        payload_bytes = len(pickle.dumps(record, pickle.HIGHEST_PROTOCOL))
        now = _now()
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO records(
                id, specification, object_hash, object_path, object_format,
                metadata, payload_bytes, stored_bytes, created_at,
                accessed_at, accesses
            ) VALUES (?, ?, ?, ?, 'npz-v1', ?, ?, ?, ?, ?, 0)
            """,
            (
                key,
                key_json,
                object_info["hash"],
                object_info["path"],
                canonical_json({} if metadata is None else metadata),
                payload_bytes,
                object_info["bytes"],
                now,
                now,
            ),
        )
        self.connection.execute("DELETE FROM claims WHERE record_id = ?", (key,))
        self.connection.commit()
        inserted = cursor.rowcount == 1
        self.writes += int(inserted)
        return key, inserted

    def entries(self):
        """Return catalog metadata without loading wavefunction objects."""

        rows = self.connection.execute(
            """
            SELECT id, specification, object_hash, object_path, object_format,
                   metadata, payload_bytes, stored_bytes, created_at,
                   accessed_at, accesses
            FROM records ORDER BY created_at
            """
        ).fetchall()
        return [
            {
                "id": row[0],
                "specification": json.loads(row[1]),
                "object_hash": row[2],
                "object_path": str(self._resolved_path(row[3])),
                "object_format": row[4],
                "metadata": json.loads(row[5]),
                "payload_bytes": int(row[6]),
                "stored_bytes": int(row[7]),
                "created_at": row[8],
                "accessed_at": row[9],
                "accesses": int(row[10]),
            }
            for row in rows
        ]

    def start_run(self, run_id, *, metadata=None, status="initialized"):
        now = _now()
        self.connection.execute(
            """
            INSERT INTO runs(id, status, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at
            """,
            (
                str(run_id),
                str(status),
                canonical_json({} if metadata is None else metadata),
                now,
                now,
            ),
        )
        self.connection.commit()

    def update_run(self, run_id, status, *, metadata=None):
        now = _now()
        if metadata is None:
            cursor = self.connection.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
                (str(status), now, str(run_id)),
            )
        else:
            cursor = self.connection.execute(
                """
                UPDATE runs SET status = ?, metadata = ?, updated_at = ? WHERE id = ?
                """,
                (str(status), canonical_json(metadata), now, str(run_id)),
            )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown electronic run {run_id!r}")
        self.connection.commit()

    def note_run_record(self, run_id, record_id, sample, source):
        grid_index = canonical_json(sample.get("index", ()))
        self.connection.execute(
            """
            INSERT INTO run_records(
                run_id, grid_index, record_id, sample, source, requested_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, grid_index) DO UPDATE SET
                record_id = excluded.record_id,
                sample = excluded.sample,
                source = excluded.source,
                requested_at = excluded.requested_at
            """,
            (
                str(run_id),
                grid_index,
                str(record_id),
                canonical_json(sample),
                str(source),
                _now(),
            ),
        )
        self.connection.commit()

    def run(self, run_id):
        row = self.connection.execute(
            "SELECT status, metadata, created_at, updated_at FROM runs WHERE id = ?",
            (str(run_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown electronic run {run_id!r}")
        samples = self.connection.execute(
            """
            SELECT grid_index, record_id, sample, source, requested_at
            FROM run_records WHERE run_id = ? ORDER BY grid_index
            """,
            (str(run_id),),
        ).fetchall()
        return {
            "id": str(run_id),
            "status": row[0],
            "metadata": json.loads(row[1]),
            "created_at": row[2],
            "updated_at": row[3],
            "records": [
                {
                    "grid_index": json.loads(item[0]),
                    "record_id": item[1],
                    "sample": json.loads(item[2]),
                    "source": item[3],
                    "requested_at": item[4],
                }
                for item in samples
            ],
        }

    def claim(self, specification, owner, *, ttl=7 * 24 * 60 * 60):
        """Claim a missing calculation and return complete, acquired, or busy."""

        key_json = canonical_json(specification)
        key = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
        now = time.time()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            if self.connection.execute(
                "SELECT 1 FROM records WHERE id = ?", (key,)
            ).fetchone():
                status = "complete"
            else:
                self.connection.execute(
                    "DELETE FROM claims WHERE expires_at <= ?", (now,)
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO claims(
                        record_id, specification, owner, claimed_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (key, key_json, str(owner), _now(), now + float(ttl)),
                )
                claimed = self.connection.execute(
                    "SELECT owner FROM claims WHERE record_id = ?", (key,)
                ).fetchone()
                status = "acquired" if claimed[0] == str(owner) else "busy"
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return status

    def release_claim(self, specification, owner):
        key = self.identifier(specification)
        cursor = self.connection.execute(
            "DELETE FROM claims WHERE record_id = ? AND owner = ?",
            (key, str(owner)),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def release_claims(self, owner):
        """Release every unfinished calculation claim owned by one run."""

        cursor = self.connection.execute(
            "DELETE FROM claims WHERE owner = ?", (str(owner),)
        )
        self.connection.commit()
        return int(cursor.rowcount)

    def active_claims(self):
        self.connection.execute("DELETE FROM claims WHERE expires_at <= ?", (time.time(),))
        self.connection.commit()
        rows = self.connection.execute(
            """
            SELECT record_id, specification, owner, claimed_at, expires_at
            FROM claims ORDER BY claimed_at
            """
        ).fetchall()
        return [
            {
                "record_id": row[0],
                "specification": json.loads(row[1]),
                "owner": row[2],
                "claimed_at": row[3],
                "expires_at": float(row[4]),
            }
            for row in rows
        ]

    @staticmethod
    def overlap_identifier(left_record_id, right_record_id, protocol):
        return hashlib.sha256(
            canonical_json(
                {
                    "left": str(left_record_id),
                    "right": str(right_record_id),
                    "protocol": protocol,
                }
            ).encode("utf-8")
        ).hexdigest()

    def put_overlap(
        self,
        left_record_id,
        right_record_id,
        protocol,
        block,
        *,
        metadata=None,
    ):
        block = np.asarray(block)
        if block.ndim != 2:
            raise ValueError("an electronic overlap block must be a matrix")
        stream = BytesIO()
        np.save(stream, block, allow_pickle=False)
        serialized = stream.getvalue()
        protocol_json = canonical_json(protocol)
        key = self.overlap_identifier(left_record_id, right_record_id, protocol)
        now = _now()
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO overlaps(
                id, left_record_id, right_record_id, protocol, shape, dtype,
                payload, checksum, metadata, created_at, accessed_at, accesses
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                key,
                str(left_record_id),
                str(right_record_id),
                protocol_json,
                canonical_json(block.shape),
                str(block.dtype),
                zlib.compress(serialized, level=3),
                hashlib.sha256(serialized).hexdigest(),
                canonical_json({} if metadata is None else metadata),
                now,
                now,
            ),
        )
        self.connection.commit()
        inserted = cursor.rowcount == 1
        self.overlap_writes += int(inserted)
        return key, inserted

    def get_overlap(self, left_record_id, right_record_id, protocol):
        protocol_json = canonical_json(protocol)
        row = self.connection.execute(
            """
            SELECT id, payload, checksum, 0 FROM overlaps
            WHERE left_record_id = ? AND right_record_id = ? AND protocol = ?
            """,
            (str(left_record_id), str(right_record_id), protocol_json),
        ).fetchone()
        if row is None:
            row = self.connection.execute(
                """
                SELECT id, payload, checksum, 1 FROM overlaps
                WHERE left_record_id = ? AND right_record_id = ? AND protocol = ?
                """,
                (str(right_record_id), str(left_record_id), protocol_json),
            ).fetchone()
        if row is None:
            self.overlap_misses += 1
            return None
        key, payload, checksum, reverse = row
        serialized = zlib.decompress(payload)
        if hashlib.sha256(serialized).hexdigest() != checksum:
            raise IOError(f"electronic overlap {key} failed its checksum")
        block = np.load(BytesIO(serialized), allow_pickle=False)
        self.connection.execute(
            "UPDATE overlaps SET accessed_at = ?, accesses = accesses + 1 WHERE id = ?",
            (_now(), key),
        )
        self.connection.commit()
        self.overlap_hits += 1
        return block.conj().T if reverse else block

    @property
    def stats(self):
        if self.connection is None:
            return dict(self._closed_stats)
        records, stored_bytes, payload_bytes, objects = self.connection.execute(
            """
            SELECT count(*), coalesce(sum(stored_bytes), 0),
                   coalesce(sum(payload_bytes), 0), count(DISTINCT object_hash)
            FROM records
            """
        ).fetchone()
        runs = self.connection.execute("SELECT count(*) FROM runs").fetchone()[0]
        claims = self.connection.execute("SELECT count(*) FROM claims").fetchone()[0]
        overlaps = self.connection.execute("SELECT count(*) FROM overlaps").fetchone()[0]
        return {
            "path": str(self.path),
            "object_dir": str(self.object_dir),
            "records": int(records),
            "objects": int(objects),
            "stored_bytes": int(stored_bytes),
            "payload_bytes": int(payload_bytes),
            "catalog_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "runs": int(runs),
            "claims": int(claims),
            "overlaps": int(overlaps),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "writes": int(self.writes),
            "overlap_hits": int(self.overlap_hits),
            "overlap_misses": int(self.overlap_misses),
            "overlap_writes": int(self.overlap_writes),
        }

    def close(self):
        if self.connection is not None:
            self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._closed_stats = self.stats
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = ["ElectronicDatabase", "canonical_json"]
