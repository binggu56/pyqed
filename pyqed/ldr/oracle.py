"""On-demand electronic frames and Procrustes-aligned matrix fields."""

from __future__ import annotations

from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
import pickle
import uuid

import numpy as np

from .database import canonical_json
from .overlap import procrustes


def isometric_frames(values):
    r"""Return the closest frames satisfying $Y^\dagger Y=I$ pointwise."""

    values = np.asarray(values)
    if values.ndim < 2 or values.shape[-2] < values.shape[-1]:
        raise ValueError("frames must end in (feature_rank, nstates) with rank >= nstates")
    if not np.all(np.isfinite(values)):
        raise ValueError("frames must be finite")
    dtype = np.complex128 if np.iscomplexobj(values) else np.float64
    left, _singular, right = np.linalg.svd(
        values.astype(dtype, copy=False), full_matrices=False
    )
    return left @ right


def _build(builder, index):
    return tuple(index), builder(tuple(index))


class Frames:
    """Lazily build, cache, and batch electronic records on a product grid."""

    def __init__(
        self,
        shape,
        builder=None,
        *,
        cache_dir=None,
        database=None,
        database_key=None,
        database_metadata=None,
        database_run=None,
        claim_ttl=7 * 24 * 60 * 60,
        workers=1,
        progress=None,
        representative=None,
        transform=None,
        view_key=None,
    ):
        self.shape = tuple(int(size) for size in shape)
        if not self.shape or any(size < 1 for size in self.shape):
            raise ValueError("shape must contain positive dimensions")
        self.builder = builder
        self.representative_index = representative
        self.transform = transform
        self._view_key = view_key
        if representative is None and (transform is not None or view_key is not None):
            raise ValueError("transform and view_key require a representative callback")
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if (database is None) != (database_key is None):
            raise ValueError("database and database_key must be provided together")
        self.database = database
        self.database_key = database_key
        self.database_metadata = database_metadata
        self.database_run = None if database_run is None else str(database_run)
        self.database_owner = self.database_run or f"frames-{uuid.uuid4().hex}"
        self.claim_ttl = float(claim_ttl)
        if self.claim_ttl <= 0.0:
            raise ValueError("claim_ttl must be positive")
        self.workers = int(workers)
        if self.workers < 1:
            raise ValueError("workers must be positive")
        self.cache = {}
        self._representative_cache = {}
        self._representative_sources = {}
        self.sources = {}
        self.points = set()
        self.requested = 0
        self.memory_hits = 0
        self.restored = 0
        self.database_hits = 0
        self.database_writes = 0
        self.database_migrations = 0
        self.built = 0
        self.batches = 0
        self.symmetry_hits = 0
        self._executor = None
        if isinstance(progress, (bool, np.bool_)):
            if progress:
                def progress(_index, stats):
                    print(f"electronic point {stats['built']}", flush=True)
            else:
                progress = None
        if progress is not None and not callable(progress):
            raise TypeError("progress must be a boolean or callback")
        self.progress = progress

    def _index(self, index):
        index = tuple(int(value) for value in index)
        if len(index) != len(self.shape) or any(
            value < 0 or value >= size
            for value, size in zip(index, self.shape)
        ):
            raise IndexError(f"grid index {index} is outside {self.shape}")
        return index

    def path(self, index):
        if self.cache_dir is None:
            return None
        return self.cache_dir / ("point_" + "_".join(map(str, index)) + ".pkl")

    @staticmethod
    def _read(path):
        with path.open("rb") as stream:
            return pickle.load(stream)

    @staticmethod
    def _write(path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    def _pool(self):
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.workers)
        return self._executor

    def representative(self, index):
        index = self._index(index)
        if self.representative_index is None:
            return index
        return self._index(self.representative_index(index))

    def view_key(self, index):
        index = self._index(index)
        return None if self._view_key is None else self._view_key(index)

    def record_id(self, index):
        if self.database is None:
            return None
        representative = self.representative(index)
        return self.database.identifier(self.database_key(representative))

    def _note_run_record(self, index):
        if self.database is None or self.database_run is None:
            return
        sample = (
            {"index": index}
            if self.database_metadata is None
            else self.database_metadata(index)
        )
        self.database.note_run_record(
            self.database_run,
            self.record_id(index),
            sample,
            self.sources[index],
        )

    def _release_claim(self, index):
        if self.database is not None:
            index = self.representative(index)
            self.database.release_claim(
                self.database_key(index), self.database_owner
            )

    def _materialize(self, value, representative, index):
        if self.transform is None or representative == index:
            return value
        return self.transform(value, representative, index)

    def _accept_symmetry_built(self, representative, value):
        representative = self.representative(representative)
        self._representative_cache[representative] = value
        self._representative_sources[representative] = "built"
        path = self.path(representative)
        if path is not None:
            self._write(path, value)
        if self.database is not None:
            specification = self.database_key(representative)
            metadata = (
                None
                if self.database_metadata is None
                else self.database_metadata(representative)
            )
            try:
                _key, inserted = self.database.put(
                    specification, value, metadata=metadata
                )
                self.database_writes += int(inserted)
            finally:
                self._release_claim(representative)
        self.built += 1
        if self.progress is not None:
            self.progress(representative, self.stats)

    def _get_many_symmetry(self, indices):
        indices = list(dict.fromkeys(self._index(index) for index in indices))
        self.requested += len(indices)
        self.points.update(indices)
        missing = []
        for index in indices:
            if index in self.cache:
                self.memory_hits += 1
                self._note_run_record(index)
            else:
                missing.append(index)

        groups = {}
        for index in missing:
            representative = self.representative(index)
            groups.setdefault(representative, []).append(index)

        unresolved = []
        for representative in groups:
            if representative in self._representative_cache:
                self.symmetry_hits += 1
                continue
            specification = (
                None
                if self.database is None
                else self.database_key(representative)
            )
            if specification is not None:
                value = self.database.get(specification)
                if value is not None:
                    self._representative_cache[representative] = value
                    self._representative_sources[representative] = "database"
                    self.database_hits += 1
                    continue
            path = self.path(representative)
            if path is not None and path.is_file():
                value = self._read(path)
                self._representative_cache[representative] = value
                self._representative_sources[representative] = "point-cache"
                self.restored += 1
                if specification is not None:
                    metadata = (
                        None
                        if self.database_metadata is None
                        else self.database_metadata(representative)
                    )
                    _key, inserted = self.database.put(
                        specification, value, metadata=metadata
                    )
                    self.database_writes += int(inserted)
                    self.database_migrations += int(inserted)
            else:
                unresolved.append(representative)

        if unresolved:
            if self.builder is None:
                raise FileNotFoundError(
                    f"No cached electronic record for {unresolved[0]} and no builder"
                )
            claimed = []
            build = []
            try:
                for representative in unresolved:
                    if self.database is None:
                        build.append(representative)
                        continue
                    specification = self.database_key(representative)
                    status = self.database.claim(
                        specification,
                        self.database_owner,
                        ttl=self.claim_ttl,
                    )
                    if status == "acquired":
                        claimed.append(representative)
                        build.append(representative)
                    elif status == "complete":
                        value = self.database.get(specification)
                        self._representative_cache[representative] = value
                        self._representative_sources[representative] = "database"
                        self.database_hits += 1
                    else:
                        record_id = self.database.identifier(specification)
                        owner = next(
                            claim["owner"]
                            for claim in self.database.active_claims()
                            if claim["record_id"] == record_id
                        )
                        raise RuntimeError(
                            f"electronic record {representative} is already being "
                            f"calculated by {owner}"
                        )
            except Exception:
                for representative in claimed:
                    self._release_claim(representative)
                raise

            if build:
                self.batches += 1
            if self.workers == 1:
                for number, representative in enumerate(build):
                    try:
                        self._accept_symmetry_built(
                            representative, self.builder(representative)
                        )
                    except Exception:
                        for pending in build[number:]:
                            self._release_claim(pending)
                        raise
            elif build:
                futures = {
                    self._pool().submit(
                        _build, self.builder, representative
                    ): representative
                    for representative in build
                }
                first_error = None
                for future in as_completed(futures):
                    representative = futures[future]
                    try:
                        built_index, value = future.result()
                        self._accept_symmetry_built(built_index, value)
                    except Exception as error:
                        self._release_claim(representative)
                        if first_error is None:
                            first_error = error
                if first_error is not None:
                    raise first_error

        for representative, requested in groups.items():
            value = self._representative_cache[representative]
            source = self._representative_sources[representative]
            for index in requested:
                self.cache[index] = self._materialize(value, representative, index)
                self.sources[index] = (
                    source
                    if index == representative
                    else f"sampling-symmetry:{source}"
                )
                self._note_run_record(index)
        return [self.cache[index] for index in indices]

    def _accept_built(self, index, value):
        index = tuple(index)
        self.cache[index] = value
        self.sources[index] = "built"
        path = self.path(index)
        if path is not None:
            self._write(path, value)
        if self.database is not None:
            specification = self.database_key(index)
            metadata = (
                None
                if self.database_metadata is None
                else self.database_metadata(index)
            )
            try:
                _key, inserted = self.database.put(
                    specification, value, metadata=metadata
                )
                self.database_writes += int(inserted)
            finally:
                self._release_claim(index)
        self.built += 1
        self._note_run_record(index)
        if self.progress is not None:
            self.progress(index, self.stats)

    def get_many(self, indices):
        if self.representative_index is not None:
            return self._get_many_symmetry(indices)
        indices = list(dict.fromkeys(self._index(index) for index in indices))
        self.requested += len(indices)
        self.points.update(indices)
        missing = []
        for index in indices:
            if index in self.cache:
                self.memory_hits += 1
                self._note_run_record(index)
                continue
            specification = (
                None if self.database is None else self.database_key(index)
            )
            if specification is not None:
                value = self.database.get(specification)
                if value is not None:
                    self.cache[index] = value
                    self.sources[index] = "database"
                    self.database_hits += 1
                    self._note_run_record(index)
                    continue
            path = self.path(index)
            if path is not None and path.is_file():
                self.cache[index] = self._read(path)
                self.sources[index] = "point-cache"
                self.restored += 1
                if specification is not None:
                    metadata = (
                        None
                        if self.database_metadata is None
                        else self.database_metadata(index)
                    )
                    _key, inserted = self.database.put(
                        specification, self.cache[index], metadata=metadata
                    )
                    self.database_writes += int(inserted)
                    self.database_migrations += int(inserted)
                self._note_run_record(index)
            else:
                missing.append(index)
        if missing:
            if self.builder is None:
                raise FileNotFoundError(
                    f"No cached electronic record for {missing[0]} and no builder"
                )
            claimed = []
            build = []
            try:
                for index in missing:
                    if self.database is None:
                        build.append(index)
                        continue
                    specification = self.database_key(index)
                    status = self.database.claim(
                        specification,
                        self.database_owner,
                        ttl=self.claim_ttl,
                    )
                    if status == "acquired":
                        claimed.append(index)
                        build.append(index)
                    elif status == "complete":
                        value = self.database.get(specification)
                        self.cache[index] = value
                        self.sources[index] = "database"
                        self.database_hits += 1
                        self._note_run_record(index)
                    else:
                        owner = next(
                            claim["owner"]
                            for claim in self.database.active_claims()
                            if claim["record_id"]
                            == self.database.identifier(specification)
                        )
                        raise RuntimeError(
                            f"electronic record {index} is already being calculated "
                            f"by {owner}"
                        )
            except Exception:
                for index in claimed:
                    self._release_claim(index)
                raise
            if build:
                self.batches += 1
            if self.workers == 1:
                for number, index in enumerate(build):
                    try:
                        value = self.builder(index)
                        self._accept_built(index, value)
                    except Exception:
                        for pending in build[number:]:
                            self._release_claim(pending)
                        raise
            elif build:
                futures = {
                    self._pool().submit(_build, self.builder, index): index
                    for index in build
                }
                first_error = None
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        built_index, value = future.result()
                        self._accept_built(built_index, value)
                    except Exception as error:
                        self._release_claim(index)
                        if first_error is None:
                            first_error = error
                if first_error is not None:
                    raise first_error
        return [self.cache[index] for index in indices]

    def get(self, index):
        index = self._index(index)
        self.get_many((index,))
        return self.cache[index]

    @property
    def stats(self):
        return {
            "requested": int(self.requested),
            "unique_requested": len(self.points),
            "resident": len(self.cache),
            "memory_hits": int(self.memory_hits),
            "restored": int(self.restored),
            "database_hits": int(self.database_hits),
            "database_writes": int(self.database_writes),
            "database_migrations": int(self.database_migrations),
            "built": int(self.built),
            "batches": int(self.batches),
            "representatives": len(self._representative_cache),
            "symmetry_hits": int(self.symmetry_hits),
            "workers": int(self.workers),
        }

    def close(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class ProcrustesOracle:
    """Expose aligned local Hamiltonians and overlaps from lazy frames."""

    def __init__(
        self,
        frames,
        anchor,
        *,
        frame,
        energies,
        overlap,
        overlap_protocol=None,
        energy_shift=0.0,
    ):
        self.frames = frames
        self.shape = frames.shape
        self.anchor = frames._index(anchor)
        self.frame_of = frame
        self.energies_of = energies
        self.overlap_of = overlap
        self.overlap_protocol = overlap_protocol
        self.energy_shift = (
            None if energy_shift is None else float(energy_shift)
        )
        self._gauges = {}
        self._locals = {}
        self._blocks = {}
        self.persistent_overlap_hits = 0
        self.persistent_overlap_writes = 0

    def _persistent_overlap_key(self, left, right):
        left_id = self.frames.record_id(left)
        right_id = self.frames.record_id(right)
        left_view = self.frames.view_key(left)
        right_view = self.frames.view_key(right)
        if left_view is None and right_view is None:
            return left_id, right_id, self.overlap_protocol, False

        left_token = canonical_json({"record": left_id, "view": left_view})
        right_token = canonical_json({"record": right_id, "view": right_view})
        reverse = right_token < left_token
        if reverse:
            left_id, right_id = right_id, left_id
            left_view, right_view = right_view, left_view
        protocol = {
            "base": self.overlap_protocol,
            "sampling_symmetry_views": [left_view, right_view],
            "version": 1,
        }
        return left_id, right_id, protocol, reverse

    def _raw_overlap(self, left, right, records):
        database = self.frames.database
        if database is not None and self.overlap_protocol is not None:
            left_id, right_id, protocol, reverse = self._persistent_overlap_key(
                left, right
            )
            raw = database.get_overlap(left_id, right_id, protocol)
            if raw is not None:
                self.persistent_overlap_hits += 1
                return raw.conj().T if reverse else raw
        raw = self.overlap_of(
            self.frame_of(records[left]), self.frame_of(records[right])
        )
        if database is not None and self.overlap_protocol is not None:
            _key, inserted = database.put_overlap(
                left_id,
                right_id,
                protocol,
                raw.conj().T if reverse else raw,
            )
            self.persistent_overlap_writes += int(inserted)
        return raw

    def _records(self, indices):
        indices = list(
            dict.fromkeys([*(tuple(index) for index in indices), self.anchor])
        )
        records = self.frames.get_many(indices)
        return dict(zip(indices, records))

    def gauges(self, indices):
        indices = list(dict.fromkeys(self.frames._index(index) for index in indices))
        missing = [index for index in indices if index not in self._gauges]
        if missing:
            records = self._records(missing)
            for index in missing:
                block = self._raw_overlap(index, self.anchor, records)
                self._gauges[index] = procrustes(block)[0]
        return [self._gauges[index] for index in indices]

    def hamiltonian_many(self, indices):
        indices = [self.frames._index(index) for index in indices]
        missing = [
            index for index in dict.fromkeys(indices) if index not in self._locals
        ]
        records = self._records(missing)
        gauges = dict(zip(missing, self.gauges(missing)))
        if self.energy_shift is None:
            self.energy_shift = float(
                np.min(self.energies_of(records[self.anchor]))
            )
        for index in missing:
            energy = np.asarray(self.energies_of(records[index]), dtype=float)
            gauge = gauges[index]
            diagonal = np.diag(energy - self.energy_shift)
            self._locals[index] = gauge.conj().T @ diagonal @ gauge
        return np.asarray([self._locals[index] for index in indices])

    def overlap_many(self, pairs):
        pairs = [
            (self.frames._index(left), self.frames._index(right))
            for left, right in pairs
        ]
        missing = []
        for pair in dict.fromkeys(pairs):
            if pair in self._blocks:
                continue
            reverse = (pair[1], pair[0])
            if reverse in self._blocks:
                self._blocks[pair] = self._blocks[reverse].conj().T
            else:
                missing.append(pair)
        indices = list(dict.fromkeys(index for pair in missing for index in pair))
        records = self._records(indices)
        gauges = dict(zip(indices, self.gauges(indices)))
        for left, right in missing:
            raw = self._raw_overlap(left, right, records)
            self._blocks[(left, right)] = (
                gauges[left].conj().T @ raw @ gauges[right]
            )
        return np.asarray([self._blocks[pair] for pair in pairs])

    def raw_overlap_many(self, pairs):
        """Return and persist unaligned electronic overlaps for sample pairs."""

        pairs = [
            (self.frames._index(left), self.frames._index(right))
            for left, right in pairs
        ]
        indices = list(dict.fromkeys(index for pair in pairs for index in pair))
        records = self._records(indices)
        return np.asarray(
            [self._raw_overlap(left, right, records) for left, right in pairs]
        )

    @property
    def stats(self):
        return {
            "aligned_hamiltonians": len(self._locals),
            "aligned_links": len(self._blocks),
            "gauges": len(self._gauges),
            "persistent_overlap_hits": int(self.persistent_overlap_hits),
            "persistent_overlap_writes": int(self.persistent_overlap_writes),
            "frames": self.frames.stats,
        }


class FeatureOracle:
    """Nyström feature map of an aligned electronic overlap oracle."""

    def __init__(self, oracle, anchors, *, tolerance=1.0e-10, max_rank=None):
        self.oracle = oracle
        self.shape = tuple(int(size) for size in oracle.shape)
        self.anchors = tuple(dict.fromkeys(tuple(map(int, index)) for index in anchors))
        if not self.anchors:
            raise ValueError("feature factorization requires at least one anchor")
        tolerance = float(tolerance)
        if tolerance <= 0.0:
            raise ValueError("feature tolerance must be positive")
        blocks = oracle.overlap_many(
            [(left, right) for left in self.anchors for right in self.anchors]
        )
        self.nstates = int(blocks.shape[-1])
        gram = blocks.reshape(
            len(self.anchors),
            len(self.anchors),
            self.nstates,
            self.nstates,
        ).transpose(0, 2, 1, 3).reshape(
            len(self.anchors) * self.nstates,
            len(self.anchors) * self.nstates,
        )
        gram = 0.5 * (gram + gram.conj().T)
        values, vectors = np.linalg.eigh(gram)
        scale = max(float(values[-1]), 1.0)
        if values[0] < -tolerance * scale:
            raise ValueError(
                "anchor overlap Gram matrix is not positive semidefinite: "
                f"minimum eigenvalue {values[0]:.3e}"
            )
        keep = np.flatnonzero(values > tolerance * scale)
        if max_rank is not None:
            max_rank = int(max_rank)
            if max_rank < 1:
                raise ValueError("max_rank must be positive")
            keep = keep[-max_rank:]
        if not len(keep):
            raise ValueError("anchor overlap Gram matrix has zero numerical rank")
        self.eigenvalues = np.asarray(values[keep], dtype=float)
        self.transform = (
            vectors[:, keep] / np.sqrt(self.eigenvalues)[None, :]
        ).conj().T
        self.rank = len(keep)
        self._features = {}
        self.points = set(self.anchors)

    def feature_many(self, indices):
        indices = [tuple(map(int, index)) for index in indices]
        missing = [
            index for index in dict.fromkeys(indices) if index not in self._features
        ]
        if missing:
            blocks = self.oracle.overlap_many(
                [(anchor, index) for index in missing for anchor in self.anchors]
            ).reshape(len(missing), len(self.anchors) * self.nstates, self.nstates)
            values = np.einsum("ra,nab->nrb", self.transform, blocks, optimize=True)
            for index, value in zip(missing, values):
                self._features[index] = value
            self.points.update(missing)
        return np.asarray([self._features[index] for index in indices])

    def feature(self, index):
        return self.feature_many((index,))[0]

    def overlap_many(self, pairs, *, diagonal_exact=True):
        pairs = [(tuple(left), tuple(right)) for left, right in pairs]
        points = list(dict.fromkeys(index for pair in pairs for index in pair))
        features = dict(zip(points, self.feature_many(points)))
        blocks = np.asarray([
            features[left].conj().T @ features[right]
            for left, right in pairs
        ])
        if diagonal_exact:
            identity = np.eye(self.nstates, dtype=complex)
            for pair, block in zip(pairs, blocks):
                if pair[0] == pair[1]:
                    block[...] = identity
        return blocks


def synchronize_features(
    oracle,
    points,
    pairs,
    feature_rank,
    *,
    anchor=None,
    penalty=10.0,
    smoothness=0.0,
    curvature=0.0,
    triples=None,
    initial=None,
    real_tolerance=1.0e-12,
    maxiter=500,
    gtol=1.0e-8,
    seed=0,
):
    """Synchronize one pinned feature map over a sampled overlap graph."""
    from scipy.optimize import minimize

    shape = tuple(int(size) for size in oracle.shape)
    ndim = len(shape)

    def checked(index):
        index = tuple(int(value) for value in index)
        if len(index) != ndim or any(
            value < 0 or value >= size for value, size in zip(index, shape)
        ):
            raise IndexError(f"grid index {index} is outside {shape}")
        return index

    points = tuple(dict.fromkeys(checked(index) for index in points))
    if len(points) < 2:
        raise ValueError("feature synchronization requires at least two sampled points")
    point_ids = {index: offset for offset, index in enumerate(points)}
    pairs = tuple(
        dict.fromkeys((checked(left), checked(right)) for left, right in pairs)
    )
    if not pairs:
        raise ValueError("feature synchronization requires overlap pairs")
    if any(left not in point_ids or right not in point_ids for left, right in pairs):
        raise ValueError("overlap-pair endpoints must belong to the sampled points")
    adjacency = {point: set() for point in points}
    for left, right in pairs:
        if left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    reached = {points[0]}
    frontier = [points[0]]
    while frontier:
        point = frontier.pop()
        for neighbor in adjacency[point] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    if len(reached) != len(points):
        raise ValueError("the sampled overlap graph must be connected")
    blocks = np.asarray(oracle.overlap_many(pairs))
    magnitude = max(float(np.max(np.abs(blocks))), 1.0)
    real_problem = float(np.max(np.abs(blocks.imag))) <= float(real_tolerance) * magnitude
    blocks = blocks.real if real_problem else blocks.astype(complex, copy=False)
    nstates = int(blocks.shape[-1])
    feature_rank = int(feature_rank)
    if feature_rank < nstates:
        raise ValueError("feature_rank must be at least nstates")
    smoothness = float(smoothness)
    curvature = float(curvature)
    if smoothness < 0.0 or curvature < 0.0:
        raise ValueError("smoothness and curvature must be nonnegative")
    anchor = (
        points[0]
        if anchor is None
        else checked(anchor)
    )
    if anchor not in point_ids:
        raise ValueError("the synchronization anchor must be a sampled point")
    anchor_id = point_ids[anchor]
    edges = tuple((point_ids[left], point_ids[right]) for left, right in pairs)
    left_ids = np.asarray([left for left, _right in edges], dtype=int)
    right_ids = np.asarray([right for _left, right in edges], dtype=int)
    triples = () if triples is None else tuple(
        (checked(left), checked(center), checked(right))
        for left, center, right in triples
    )
    if any(point not in point_ids for triple in triples for point in triple):
        raise ValueError("curvature-triple points must belong to sampled points")
    triple_ids = np.asarray(
        [[point_ids[point] for point in triple] for triple in triples], dtype=int
    ).reshape(-1, 3)
    npoints = len(points)
    free = np.asarray([index for index in range(npoints) if index != anchor_id])
    if initial is None:
        dtype = float if real_problem else complex
        rng = np.random.default_rng(seed)
        random = rng.standard_normal((npoints, feature_rank, nstates))
        if not real_problem:
            random = random + 1j * rng.standard_normal(random.shape)
        features = np.asarray(
            [np.linalg.qr(value, mode="reduced")[0] for value in random],
            dtype=dtype,
        )
        anchor_basis = np.linalg.qr(features[anchor_id], mode="complete")[0]
        features = np.einsum(
            "rs,psa->pra", anchor_basis.conj().T, features, optimize=True
        )
    else:
        initial = np.asarray(initial)
        if real_problem:
            initial_magnitude = max(float(np.max(np.abs(initial))), 1.0)
            if float(np.max(np.abs(initial.imag))) > float(real_tolerance) * initial_magnitude:
                raise ValueError("real overlap targets require a real feature warm start")
            features = initial.real.copy()
        else:
            features = initial.astype(complex, copy=True)
        expected = (npoints, feature_rank, nstates)
        if features.shape != expected or not np.all(np.isfinite(features)):
            raise ValueError(f"initial features must be finite with shape {expected}")
    features[anchor_id] = 0.0
    features[anchor_id, :nstates, :] = np.eye(nstates)

    def pack(values):
        values = values[free].reshape(-1)
        return values if real_problem else np.concatenate((values.real, values.imag))

    def unpack(vector):
        count = len(free) * feature_rank * nstates
        values = vector if real_problem else vector[:count] + 1j * vector[count:]
        output = features.copy()
        output[free] = values.reshape(len(free), feature_rank, nstates)
        return output

    edge_scale = max(len(edges), 1)
    point_scale = max(len(free), 1)

    def objective(vector):
        values = unpack(vector)
        gradient = np.zeros_like(values)
        left_values = values[left_ids]
        right_values = values[right_ids]
        error = np.einsum(
            "era,erb->eab", left_values.conj(), right_values, optimize=True
        ) - blocks
        loss = float(np.vdot(error, error).real) / edge_scale
        left_gradient = 2.0 * np.einsum(
            "erb,eab->era", right_values, error.conj(), optimize=True
        ) / edge_scale
        right_gradient = 2.0 * np.einsum(
            "era,eab->erb", left_values, error, optimize=True
        ) / edge_scale
        if smoothness:
            difference = right_values - left_values
            loss += smoothness * float(np.vdot(difference, difference).real) / edge_scale
            left_gradient -= 2.0 * smoothness * difference / edge_scale
            right_gradient += 2.0 * smoothness * difference / edge_scale
        if curvature and len(triple_ids):
            second = (
                values[triple_ids[:, 0]]
                - 2.0 * values[triple_ids[:, 1]]
                + values[triple_ids[:, 2]]
            )
            triple_scale = len(triple_ids)
            loss += (
                curvature * float(np.vdot(second, second).real) / triple_scale
            )
            second_gradient = 2.0 * curvature * second / triple_scale
            np.add.at(gradient, triple_ids[:, 0], second_gradient)
            np.add.at(gradient, triple_ids[:, 1], -2.0 * second_gradient)
            np.add.at(gradient, triple_ids[:, 2], second_gradient)
        np.add.at(gradient, left_ids, left_gradient)
        np.add.at(gradient, right_ids, right_gradient)
        identity = np.eye(nstates)
        free_values = values[free]
        defect = np.einsum(
            "pra,prb->pab", free_values.conj(), free_values, optimize=True
        ) - identity
        loss += penalty * float(np.vdot(defect, defect).real) / point_scale
        gradient[free] += 4.0 * penalty * np.einsum(
            "pra,pab->prb", free_values, defect, optimize=True
        ) / point_scale
        flat = gradient[free].reshape(-1)
        packed_gradient = (
            flat if real_problem else np.concatenate((flat.real, flat.imag))
        )
        return loss, packed_gradient

    result = minimize(
        objective,
        pack(features),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "gtol": float(gtol), "ftol": 1.0e-14},
    )
    unconstrained = unpack(result.x)
    fitted = isometric_frames(unconstrained)
    fitted[anchor_id] = 0.0
    fitted[anchor_id, :nstates, :] = np.eye(nstates)
    errors = np.asarray(
        [
            np.linalg.norm(fitted[left].conj().T @ fitted[right] - target)
            / max(np.linalg.norm(target), np.finfo(float).tiny)
            for (left, right), target in zip(edges, blocks)
        ]
    )
    orthogonality = np.asarray(
        [
            np.linalg.norm(value.conj().T @ value - np.eye(nstates))
            for value in fitted
        ]
    )
    unconstrained_orthogonality = np.asarray(
        [
            np.linalg.norm(value.conj().T @ value - np.eye(nstates))
            for value in unconstrained
        ]
    )
    final_loss = float(np.sum(errors**2)) / edge_scale
    if smoothness:
        difference = fitted[right_ids] - fitted[left_ids]
        final_loss += smoothness * float(np.vdot(difference, difference).real) / edge_scale
    if curvature and len(triple_ids):
        second = (
            fitted[triple_ids[:, 0]]
            - 2.0 * fitted[triple_ids[:, 1]]
            + fitted[triple_ids[:, 2]]
        )
        final_loss += curvature * float(np.vdot(second, second).real) / len(triple_ids)
    return fitted, {
        "backend": "sampled-graph-feature-synchronization",
        "feature_rank": feature_rank,
        "anchor": anchor,
        "points": npoints,
        "pairs": len(edges),
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "retracted_objective": final_loss,
        "maximum_relative_link_error": float(np.max(errors)),
        "rms_relative_link_error": float(np.sqrt(np.mean(errors**2))),
        "maximum_orthogonality_defect": float(np.max(orthogonality)),
        "unconstrained_maximum_orthogonality_defect": float(
            np.max(unconstrained_orthogonality)
        ),
        "isometry": "exact-polar-retraction",
        "penalty": float(penalty),
        "smoothness": smoothness,
        "curvature": curvature,
        "triples": len(triple_ids),
        "warm_started": initial is not None,
        "real_valued": bool(real_problem),
        "real_tolerance": float(real_tolerance),
    }


def optimize_link_features(
    oracle,
    feature_rank,
    *,
    anchor=None,
    penalty=10.0,
    smoothness=0.0,
    maxiter=500,
    gtol=1.0e-8,
    seed=0,
):
    """Fit pinned features from every nearest link on a product grid."""
    shape = tuple(int(size) for size in oracle.shape)
    points = tuple(np.ndindex(shape))
    pairs = []
    for left in points:
        for axis, size in enumerate(shape):
            if left[axis] + 1 >= size:
                continue
            right = list(left)
            right[axis] += 1
            pairs.append((left, tuple(right)))
    fitted, info = synchronize_features(
        oracle,
        points,
        pairs,
        feature_rank,
        anchor=(tuple(size // 2 for size in shape) if anchor is None else anchor),
        penalty=penalty,
        smoothness=smoothness,
        maxiter=maxiter,
        gtol=gtol,
        seed=seed,
    )
    info = dict(info)
    info["backend"] = "pinned-global-link-feature-lbfgs"
    info["links"] = info["pairs"]
    return fitted.reshape(*shape, feature_rank, fitted.shape[-1]), info


__all__ = [
    "FeatureOracle",
    "Frames",
    "ProcrustesOracle",
    "isometric_frames",
    "optimize_link_features",
    "synchronize_features",
]
