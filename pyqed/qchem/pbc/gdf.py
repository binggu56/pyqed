"""Native periodic Gaussian density fitting for SCF consumers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import threading
import time

import numpy as np


@dataclass(frozen=True)
class DiskCDERI:
    """Descriptor for one dense or Hermitian-packed disk-backed factor block."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    packed: bool = False
    nao: int | None = None

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize

    def array(self):
        return np.load(self.path, mmap_mode="r")


@dataclass(frozen=True)
class PackedHermitianCDERI:
    """Lower-triangular storage for Hermitian q=0 AO-pair factors."""

    values: np.ndarray
    nao: int

    @classmethod
    def from_dense(cls, block):
        block = np.asarray(block, dtype=np.complex128)
        if block.ndim != 3 or block.shape[1] != block.shape[2]:
            raise ValueError("A cderi block must have shape (rank, nao, nao).")
        block = 0.5 * (block + block.conj().transpose(0, 2, 1))
        diagonal = np.diag_indices(block.shape[1])
        rows, cols = np.tril_indices(block.shape[1], k=-1)
        return cls(
            values=np.ascontiguousarray(
                np.concatenate(
                    (
                        block[:, diagonal[0], diagonal[1]],
                        block[:, rows, cols],
                    ),
                    axis=1,
                )
            ),
            nao=int(block.shape[1]),
        )

    @property
    def rank(self):
        return int(self.values.shape[0])

    def to_dense(self):
        out = np.zeros(
            (self.rank, self.nao, self.nao), dtype=np.complex128
        )
        diagonal = np.diag_indices(self.nao)
        out[:, diagonal[0], diagonal[1]] = self.values[:, : self.nao]
        rows, cols = np.tril_indices(self.nao, k=-1)
        offset = self.nao
        lower = self.values[:, offset:]
        out[:, rows, cols] = lower
        out[:, cols, rows] = lower.conj()
        return out

    def contract_density(self, density):
        density = np.asarray(density)
        if density.shape != (self.nao, self.nao):
            raise ValueError(f"density must have shape ({self.nao}, {self.nao}).")
        diagonal = np.diag_indices(self.nao)
        result = np.einsum(
            "Pi,i->P",
            self.values[:, : self.nao],
            density[diagonal],
            optimize=True,
        )
        rows, cols = np.tril_indices(self.nao, k=-1)
        lower = self.values[:, self.nao :]
        result += np.einsum(
            "Pi,i->P", lower, density[cols, rows], optimize=True
        )
        result += np.einsum(
            "Pi,i->P", lower.conj(), density[rows, cols], optimize=True
        )
        return result


def _cderi_is_hermitian(block, tolerance):
    block = np.asarray(block)
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("gdf_hermitian_pack_tol must be non-negative and finite.")
    if block.size == 0:
        return True
    scale = max(float(np.max(np.abs(block))), np.finfo(float).tiny)
    residual = float(
        np.max(np.abs(block - block.conj().transpose(0, 2, 1)))
    )
    return residual <= tolerance * scale


def _wrap_scaled(values):
    return ((np.asarray(values, dtype=float) + 0.5) % 1.0) - 0.5


class _KMeshReference:
    """Integral-only k-mesh view, independent of GW transition spaces."""

    def __init__(self, mf):
        self._pbc_mf = mf
        self.cell = mf.cell
        self.kpts = np.asarray(mf.kpts, dtype=float).reshape(-1, 3)
        self.nkpts = int(len(self.kpts))
        self.reciprocal_vectors = 2.0 * np.pi * np.linalg.inv(
            np.asarray(self.cell.lattice_vectors, dtype=float)
        ).T
        self.scaled_kpts = _wrap_scaled(
            self.kpts @ np.linalg.inv(self.reciprocal_vectors)
        )

    def scaled_to_cartesian(self, scaled):
        return np.asarray(scaled, dtype=float) @ self.reciprocal_vectors

    def cartesian_to_scaled(self, kpts):
        return np.asarray(kpts, dtype=float) @ np.linalg.inv(
            self.reciprocal_vectors
        )

    def find_kpoint_index(self, kvec, tol=1.0e-8):
        target = _wrap_scaled(self.cartesian_to_scaled(kvec))
        delta = _wrap_scaled(self.scaled_kpts - target)
        distances = np.max(np.abs(delta), axis=1)
        index = int(np.argmin(distances))
        if distances[index] > tol:
            raise ValueError("k+q point is not present in the SCF k mesh.")
        return index

    def qpoint_mesh(self, tol=1.0e-8):
        scaled_qpts = []
        for k_to in self.scaled_kpts:
            for k_from in self.scaled_kpts:
                q_scaled = _wrap_scaled(k_to - k_from)
                if not any(
                    np.max(np.abs(_wrap_scaled(q_scaled - old))) <= tol
                    for old in scaled_qpts
                ):
                    scaled_qpts.append(q_scaled)
        scaled_qpts.sort(
            key=lambda q: (
                np.linalg.norm(q) > tol,
                tuple(np.round(q, 12)),
            )
        )
        return self.scaled_to_cartesian(np.asarray(scaled_qpts, dtype=float))


class _KMeshSpace:
    """Small protocol consumed by the native periodic integral kernels."""

    def __init__(self, mf):
        self.reference = _KMeshReference(mf)
        self.qpts = self.reference.qpoint_mesh()
        self.q0_index = self.find_qpoint_index(np.zeros(3))
        self.q_index_by_kpair = np.empty(
            (self.reference.nkpts, self.reference.nkpts), dtype=np.int64
        )
        for k_index, kvec in enumerate(self.reference.kpts):
            for kq_index, kqvec in enumerate(self.reference.kpts):
                self.q_index_by_kpair[k_index, kq_index] = self.find_qpoint_index(
                    kqvec - kvec
                )

    def normalize_q_index(self, q_index):
        index = int(q_index)
        if index < 0 or index >= len(self.qpts):
            raise IndexError(f"q_index {index} is out of range for {len(self.qpts)} q points.")
        return index

    def find_qpoint_index(self, qvec, tol=1.0e-8):
        scaled = _wrap_scaled(self.reference.cartesian_to_scaled(self.qpts))
        target = _wrap_scaled(self.reference.cartesian_to_scaled(qvec))
        distances = np.max(np.abs(_wrap_scaled(scaled - target)), axis=1)
        index = int(np.argmin(distances))
        if distances[index] > tol:
            raise ValueError("Requested q point is not present in the SCF difference mesh.")
        return index


@dataclass
class PeriodicGDF:
    """Persistent q-resolved periodic GDF backend.

    The expensive auxiliary metrics and three-center AO image integrals use the
    native range-separated kernels.  Whitened AO factors are cached independently
    of orbitals, so every SCF cycle contracts the new density directly without
    constructing a GW transition space or transforming all AO pairs to MOs.

    ``aux_min_exponent`` applies an explicit primitive exponent floor to the
    fitting basis, matching PySCF GDF's ``exp_to_discard`` semantics.  It is an
    opt-in convergence control because pruning changes the auxiliary space.

    When ``metric_tol`` is omitted, the auxiliary-metric pseudoinverse uses
    ``max(1e-14, 0.1 * precision)``.  The precision-aware floor removes modes
    below the reliable integral scale while preserving an explicit numeric
    override for controlled convergence studies.
    """

    mf: object
    auxbasis: str | None = None
    precision: float | None = None
    reciprocal_kernel: str | None = None
    recip_cut: int | None = None
    omega: float | str | None = None
    mesh: tuple[int, int, int] | str | None = None
    pair_cut: int | str | None = None
    pair_screen_tol: float | None = None
    image_cut: int | tuple[int, ...] | str | None = None
    aux_min_exponent: float | None = None
    metric_tol: float | None = None
    metric_relative_tol: float | None = None
    g2_tol: float = 1.0e-16
    storage: str = "auto"
    max_memory_mb: float | None = None
    cache_dir: str | None = None
    release_raw_ao: bool = True
    stream_pairs: bool = False
    stream_pair_batch_size: int | str | None = None
    stream_pair_batch_mb: float = 128.0
    _space: _KMeshSpace = field(init=False, repr=False)
    _cderi_cache: dict = field(default_factory=dict, init=False, repr=False)
    _q_metadata: dict = field(default_factory=dict, init=False, repr=False)
    _configuration_key: tuple | None = field(default=None, init=False, repr=False)
    _memory_bytes: int = field(default=0, init=False, repr=False)
    _disk_dir: Path | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(init=False, repr=False)
    _q_locks: list = field(init=False, repr=False)
    build_timings: dict = field(default_factory=dict, init=False)
    multi_q_build_timings: list = field(default_factory=list, init=False)

    def __post_init__(self):
        mf = self.mf
        if not getattr(mf.cell, "built", False):
            mf.cell.build()
        if getattr(mf, "_basis", None) is None:
            mf._validate()
            mf._periodic_setup()
        if self.auxbasis is not None:
            mf.gdf_auxbasis = self.auxbasis
        if self.precision is not None:
            mf.gdf_precision = float(self.precision)
        if self.reciprocal_kernel is not None:
            mf.gdf_reciprocal_kernel = str(self.reciprocal_kernel)
        if self.recip_cut is not None:
            mf.gdf_recip_cut = int(self.recip_cut)
        if self.omega is not None:
            mf.gdf_omega = self.omega
        if self.mesh is not None:
            mf.gdf_mesh = self.mesh
        if self.pair_cut is not None:
            mf.gdf_pair_cut = self.pair_cut
        if self.pair_screen_tol is not None:
            mf.gdf_pair_screen_tol = float(self.pair_screen_tol)
        if self.image_cut is not None:
            mf.gdf_short_range_cut = self.image_cut
        if self.aux_min_exponent is not None:
            value = float(self.aux_min_exponent)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "aux_min_exponent must be non-negative and finite."
                )
            mf.gdf_aux_min_exponent = value
        if self.metric_tol is not None:
            mf.gdf_metric_tol = float(self.metric_tol)
        if self.metric_relative_tol is not None:
            value = float(self.metric_relative_tol)
            if not np.isfinite(value) or value < 0.0 or value >= 1.0:
                raise ValueError(
                    "metric_relative_tol must be finite and in [0, 1)."
                )
            mf.gdf_metric_relative_tol = value
        self.storage = str(self.storage).strip().lower()
        if self.storage not in ("auto", "memory", "disk"):
            raise ValueError("storage must be 'auto', 'memory', or 'disk'.")
        if self.max_memory_mb is None:
            self.max_memory_mb = float(getattr(mf, "max_memory", 512.0))
        self.max_memory_mb = float(self.max_memory_mb)
        if not np.isfinite(self.max_memory_mb) or self.max_memory_mb < 0.0:
            raise ValueError("max_memory_mb must be a non-negative finite value.")
        self.stream_pair_batch_mb = float(self.stream_pair_batch_mb)
        if (
            not np.isfinite(self.stream_pair_batch_mb)
            or self.stream_pair_batch_mb <= 0.0
        ):
            raise ValueError("stream_pair_batch_mb must be a positive finite value.")
        if self.stream_pair_batch_size is not None:
            value = self.stream_pair_batch_size
            if isinstance(value, str) and value.strip().lower() == "auto":
                self.stream_pair_batch_size = None
            else:
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError("stream_pair_batch_size must be a positive integer.")
                try:
                    integer = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "stream_pair_batch_size must be a positive integer."
                    ) from exc
                if integer <= 0 or (
                    not isinstance(value, str) and integer != value
                ):
                    raise ValueError("stream_pair_batch_size must be a positive integer.")
                self.stream_pair_batch_size = integer
        self._space = _KMeshSpace(mf)
        self._lock = threading.RLock()
        self._q_locks = [threading.RLock() for _qvec in self._space.qpts]

    @property
    def qpts(self):
        return self._space.qpts

    @property
    def nkpts(self):
        return self._space.reference.nkpts

    def find_qpoint_index(self, qvec, tol=1.0e-8):
        return self._space.find_qpoint_index(qvec, tol=tol)

    @staticmethod
    def _freeze_setting(value):
        if isinstance(value, np.ndarray):
            return (value.shape, str(value.dtype), value.tobytes())
        if isinstance(value, (list, tuple)):
            return tuple(PeriodicGDF._freeze_setting(item) for item in value)
        if isinstance(value, dict):
            return tuple(
                sorted((key, PeriodicGDF._freeze_setting(item)) for key, item in value.items())
            )
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    def _current_configuration_key(self):
        mf = self.mf
        options = tuple(
            sorted(
                (name, self._freeze_setting(value))
                for name, value in vars(mf).items()
                if name.startswith("gdf_") or name.startswith("df_")
            )
        )
        return (
            id(getattr(mf.cell.unit_molecule, "_bas", None)),
            self._freeze_setting(np.asarray(mf.kpts, dtype=float)),
            self._freeze_setting(np.asarray(mf.cell.lattice_vectors, dtype=float)),
            float(mf.eta),
            int(mf.recip_cut),
            int(mf.pair_cut),
            options,
        )

    def _sync_configuration(self):
        key = self._current_configuration_key()
        if self._configuration_key is not None and key != self._configuration_key:
            self.clear()
        self._configuration_key = key
        return key

    @property
    def memory_bytes(self):
        return int(self._memory_bytes)

    @property
    def disk_bytes(self):
        return int(
            sum(
                factor.nbytes
                for factor in self._cderi_cache.values()
                if isinstance(factor, DiskCDERI)
            )
        )

    @property
    def cache_files(self):
        return tuple(
            factor.path
            for factor in self._cderi_cache.values()
            if isinstance(factor, DiskCDERI)
        )

    def _ensure_disk_dir(self):
        if self._disk_dir is None:
            if self.cache_dir is None:
                parent = None
            else:
                parent_path = Path(self.cache_dir)
                parent_path.mkdir(parents=True, exist_ok=True)
                parent = str(parent_path)
            self._disk_dir = Path(
                tempfile.mkdtemp(prefix="pyqed-gdf-", dir=parent)
            )
        return self._disk_dir

    def _use_disk(self, nbytes):
        if self.storage == "disk":
            return True
        if self.storage == "memory":
            return False
        limit = int(self.max_memory_mb * 1.0e6)
        return self._memory_bytes + int(nbytes) > limit

    def _cache_array(self, key, array, *, packed=False, nao=None):
        array = np.ascontiguousarray(array)
        with self._lock:
            if not self._use_disk(array.nbytes):
                self._memory_bytes += int(array.nbytes)
                if packed:
                    return PackedHermitianCDERI(array, int(nao))
                return array

            directory = self._ensure_disk_dir()
            q_index, k_index, kq_index = key[-3:]
            path = directory / f"q{q_index:05d}-k{k_index:05d}-{kq_index:05d}.npy"
            mapped = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=array.dtype,
                shape=array.shape,
            )
            mapped[...] = array
            mapped.flush()
            del mapped
            return DiskCDERI(
                path=str(path),
                shape=tuple(int(value) for value in array.shape),
                dtype=np.dtype(array.dtype).str,
                packed=bool(packed),
                nao=None if nao is None else int(nao),
            )

    def _store(self, q_index, pair_keys=None):
        from pyqed.pbc.gw.integrals import (
            _gdf_aux_coord_type,
            _gdf_auxbasis_name,
            _gdf_auxiliary_basis,
            _gdf_backend_settings,
            _gdf_g_block_size,
            _gdf_image_cut_key,
            _gdf_metric_tol,
            _gdf_pair_screen_tol,
            _gdf_q_ao_store,
            _gdf_rs_shell_engine,
            _gdf_short_range_cut,
            _gdf_short_range_screen_tol,
            _gdf_uses_short_range,
            _gdf_weighted_aux_screen_tol,
            _pair_keys_for_q,
        )

        q_index = self._space.normalize_q_index(q_index)
        ref = self._space.reference
        mf = self.mf
        auxbasis = _gdf_auxbasis_name(ref)
        coord_type = _gdf_aux_coord_type(ref)
        factor_threshold = max(float(self.g2_tol), _gdf_metric_tol(ref))
        (
            recip_cut,
            pair_cut,
            mesh,
            recip_key,
            kernel,
            omega,
            kernel_key,
            _auto_info,
        ) = _gdf_backend_settings(ref)
        short_range_cut = _gdf_short_range_cut(ref)
        short_range_key = (
            _gdf_image_cut_key(short_range_cut)
            if _gdf_uses_short_range(kernel)
            else None
        )
        pair_screen_tol = _gdf_pair_screen_tol(ref)
        short_range_screen_tol = _gdf_short_range_screen_tol(ref)
        rs_engine = _gdf_rs_shell_engine(ref, kernel, omega, mesh)
        aux = _gdf_auxiliary_basis(self._space, auxbasis, coord_type)
        if pair_keys is None:
            pair_keys = list(_pair_keys_for_q(self._space, q_index))
        else:
            pair_keys = [
                (int(k_index), int(kq_index)) for k_index, kq_index in pair_keys
            ]
        nao = int(mf.cell.nao)
        g_block_size = _gdf_g_block_size(
            mf,
            mesh=mesh,
            naux=aux.naux,
            nao_pair=nao * nao,
            nkpts=len(pair_keys),
            force_stream=_gdf_weighted_aux_screen_tol(ref) > 0.0,
        )
        timings = {"q_index": int(q_index), "consumer": "periodic_gdf_scf"}
        store = _gdf_q_ao_store(
            self._space,
            q_index,
            aux,
            auxbasis,
            coord_type,
            factor_threshold,
            pair_keys,
            recip_cut,
            pair_cut,
            mesh,
            recip_key,
            kernel,
            omega,
            kernel_key,
            short_range_key,
            pair_screen_tol,
            short_range_screen_tol,
            g_block_size,
            timings=timings,
            rs_engine=rs_engine,
        )
        self._q_metadata[(self._configuration_key, int(q_index))] = {
            "auxbasis": auxbasis,
            "aux_coord_type": coord_type,
            "naux_cart": int(aux.ncart),
            "factor_threshold": float(factor_threshold),
            "metric_rank": int(store.metric_invsqrt.shape[1]),
            "metric_eigenvalues": np.array(store.metric_eigenvalues, copy=True),
            "factor_method": (
                "periodic_auxiliary_gdf"
                if kernel == "full"
                else (
                    "periodic_auxiliary_gdf:long_range_reciprocal"
                    if kernel == "long_range"
                    else "periodic_auxiliary_gdf:range_separated"
                )
            ),
        }
        return store, timings

    def _pair_batches(self, pair_keys):
        pair_keys = list(pair_keys)
        if not pair_keys:
            return [], {
                "stream_pair_batch_size": 0,
                "stream_pair_raw_block_bytes": 0,
                "stream_pair_workspace_budget_bytes": 0,
                "stream_pair_workspace_factor": 0,
                "stream_pair_estimated_workspace_bytes": 0,
            }
        if not self.stream_pairs:
            batch_size = len(pair_keys)
            raw_block_bytes = 0
            budget_bytes = 0
            workspace_factor = 0
        else:
            from pyqed.pbc.gw.integrals import (
                _gdf_aux_coord_type,
                _gdf_auxbasis_name,
                _gdf_auxiliary_basis,
                _gdf_backend_settings,
                _gdf_short_range_workers,
                _gdf_uses_short_range,
            )

            ref = self._space.reference
            aux = _gdf_auxiliary_basis(
                self._space,
                _gdf_auxbasis_name(ref),
                _gdf_aux_coord_type(ref),
            )
            nao = int(self.mf.cell.nao)
            raw_block_bytes = int(
                max(int(aux.ncart), int(aux.naux))
                * nao
                * nao
                * np.dtype(np.complex128).itemsize
            )
            budget_bytes = int(self.stream_pair_batch_mb * 1.0e6)
            kernel = _gdf_backend_settings(ref)[4]
            workspace_factor = (
                3 * _gdf_short_range_workers(self.mf) + 4
                if _gdf_uses_short_range(kernel)
                else 4
            )
            if self.stream_pair_batch_size is None:
                workspace_bytes_per_pair = max(
                    1,
                    workspace_factor * raw_block_bytes,
                )
                batch_size = max(1, budget_bytes // workspace_bytes_per_pair)
            else:
                batch_size = int(self.stream_pair_batch_size)
            batch_size = min(batch_size, len(pair_keys))
        batches = [
            pair_keys[start : start + batch_size]
            for start in range(0, len(pair_keys), batch_size)
        ]
        return batches, {
            "stream_pair_batch_size": int(batch_size),
            "stream_pair_raw_block_bytes": int(raw_block_bytes),
            "stream_pair_workspace_budget_bytes": int(budget_bytes),
            "stream_pair_workspace_factor": int(workspace_factor),
            "stream_pair_estimated_workspace_bytes": int(
                workspace_factor * raw_block_bytes * batch_size
            ),
        }

    def _resolve_factor(self, factor):
        if not isinstance(factor, DiskCDERI):
            return factor
        array = factor.array()
        if factor.packed:
            return PackedHermitianCDERI(array, int(factor.nao))
        return array

    def cderi(self, q_index, k_index, kq_index=None):
        """Return one whitened AO three-center block ``(rank, nao, nao)``."""

        factor = self._cderi_factor(q_index, k_index, kq_index)
        factor = self._resolve_factor(factor)
        if isinstance(factor, PackedHermitianCDERI):
            return factor.to_dense()
        return factor

    def _materialize_q(self, q_index):
        from pyqed.pbc.gw.integrals import (
            _gdf_self_opposite_pair_sources,
            _gdf_should_use_opposite_q,
        )

        q_index = self._space.normalize_q_index(q_index)
        configuration_key = self._sync_configuration()
        pair_keys = self.pair_keys(q_index)
        cache_keys = {
            pair: (configuration_key, q_index, int(pair[0]), int(pair[1]))
            for pair in pair_keys
        }
        if all(key in self._cderi_cache for key in cache_keys.values()):
            return

        with self._q_locks[q_index]:
            configuration_key = self._sync_configuration()
            cache_keys = {
                pair: (configuration_key, q_index, int(pair[0]), int(pair[1]))
                for pair in pair_keys
            }
            if all(key in self._cderi_cache for key in cache_keys.values()):
                return

            source_q = _gdf_should_use_opposite_q(self._space, q_index)
            if source_q is not None:
                self._materialize_q(source_q)
                for k_index, kq_index in pair_keys:
                    source = self.cderi(source_q, kq_index, k_index)
                    block = np.ascontiguousarray(
                        source.conj().transpose(0, 2, 1)
                    )
                    key = cache_keys[(k_index, kq_index)]
                    with self._lock:
                        self._cderi_cache[key] = self._cache_array(key, block)
                source_metadata = self.metric_info(source_q)
                self._q_metadata[(configuration_key, q_index)] = {
                    **source_metadata,
                    "metric_eigenvalues": np.array(
                        source_metadata["metric_eigenvalues"], copy=True
                    ),
                    "factor_method": (
                        f'{source_metadata["factor_method"]}:opposite_q_conjugate'
                    ),
                }
                self.build_timings[int(q_index)] = {
                    "q_index": int(q_index),
                    "consumer": "periodic_gdf_scf",
                    "opposite_q_source": int(source_q),
                    "opposite_q_conjugate_reuse": True,
                }
                return

            missing_pairs = [
                pair for pair in pair_keys if cache_keys[pair] not in self._cderi_cache
            ]
            source_pairs, source_by_pair = _gdf_self_opposite_pair_sources(
                self._space,
                q_index,
                missing_pairs,
            )
            targets_by_source = {source: [] for source in source_pairs}
            for pair in missing_pairs:
                targets_by_source[source_by_pair[pair]].append(pair)
            batches, batch_info = self._pair_batches(source_pairs)
            batch_timings = []
            total_t0 = time.perf_counter()
            for batch in batches:
                requested = batch if self.stream_pairs else None
                store, timings = self._store(q_index, requested)
                batch_timings.append(timings)
                for source in batch:
                    source_ao = store.ao_blocks[source]
                    for k_index, kq_index in targets_by_source[source]:
                        key = cache_keys[(k_index, kq_index)]
                        ao = (
                            source_ao
                            if (k_index, kq_index) == source
                            else source_ao.conj().transpose(0, 2, 1)
                        )
                        block = np.ascontiguousarray(
                            np.einsum(
                                "Pa,Pmn->amn",
                                store.metric_invsqrt.conj(),
                                ao,
                                optimize=True,
                            )
                        )
                        pack_tol = float(
                            getattr(self.mf, "gdf_hermitian_pack_tol", 1.0e-10)
                        )
                        can_pack = (
                            q_index == self._space.q0_index
                            and k_index == kq_index
                            and _cderi_is_hermitian(block, pack_tol)
                        )
                        if can_pack:
                            packed = PackedHermitianCDERI.from_dense(block)
                            factor = self._cache_array(
                                key,
                                packed.values,
                                packed=True,
                                nao=packed.nao,
                            )
                        else:
                            factor = self._cache_array(key, block)
                        with self._lock:
                            self._cderi_cache[key] = factor
                    if self.release_raw_ao:
                        store.ao_blocks.pop(source, None)
                        for target in targets_by_source[source]:
                            store.ao_blocks.pop(target, None)

            summary = dict(batch_timings[-1]) if batch_timings else {
                "q_index": int(q_index),
                "consumer": "periodic_gdf_scf",
            }
            second_keys = {
                key
                for timings in batch_timings
                for key, value in timings.items()
                if key.endswith("_seconds")
                and isinstance(value, (int, float, np.integer, np.floating))
            }
            for key in second_keys:
                summary[key] = float(
                    sum(float(timings.get(key, 0.0)) for timings in batch_timings)
                )
            summary.update(batch_info)
            summary["stream_pair_requested_pair_count"] = int(len(missing_pairs))
            summary["stream_pair_source_pair_count"] = int(len(source_pairs))
            summary["stream_pair_self_opposite_pair_reuses"] = int(
                len(missing_pairs) - len(source_pairs)
            )
            summary["stream_pair_batches"] = int(len(batches))
            summary["stream_pair_batch_pair_counts"] = [
                int(len(batch)) for batch in batches
            ]
            summary["stream_pair_batch_summaries"] = [
                {
                    "pair_keys": [[int(left), int(right)] for left, right in batch],
                    "q_ao_store_build_seconds": float(
                        timings.get("q_ao_store_build_seconds", 0.0)
                    ),
                    "three_center_short_range_seconds": float(
                        timings.get("three_center_short_range_seconds", 0.0)
                    ),
                    "pair_ft_stream_g_vectors_seconds": float(
                        timings.get("pair_ft_stream_g_vectors_seconds", 0.0)
                    ),
                }
                for batch, timings in zip(batches, batch_timings)
            ]
            summary["total_seconds"] = float(time.perf_counter() - total_t0)
            self.build_timings[int(q_index)] = summary

    def _cderi_factor(self, q_index, k_index, kq_index=None):
        """Return the internally stored dense or packed cderi representation."""

        q_index = self._space.normalize_q_index(q_index)
        k_index = int(k_index)
        if kq_index is None:
            qvec = self.qpts[q_index]
            ref = self._space.reference
            kq_index = ref.find_kpoint_index(ref.kpts[k_index] + qvec)
        kq_index = int(kq_index)
        configuration_key = self._sync_configuration()
        key = (configuration_key, q_index, k_index, kq_index)
        factor = self._cderi_cache.get(key)
        if factor is not None:
            return factor
        self._materialize_q(q_index)
        return self._cderi_cache[key]

    def packed_cderi(self, k_index):
        """Return persistent packed q=0 factors for one k point."""

        factor = self._cderi_factor(self._space.q0_index, k_index, k_index)
        factor = self._resolve_factor(factor)
        if not isinstance(factor, PackedHermitianCDERI):
            raise RuntimeError("The q=0 same-k cderi block is not Hermitian-packed.")
        return factor

    def _q_is_materialized(self, q_index):
        q_index = self._space.normalize_q_index(q_index)
        configuration_key = self._sync_configuration()
        return all(
            (
                configuration_key,
                q_index,
                int(k_index),
                int(kq_index),
            ) in self._cderi_cache
            for k_index, kq_index in self.pair_keys(q_index)
        )

    @staticmethod
    def _chunks(values, size):
        return [values[start : start + size] for start in range(0, len(values), size)]

    def _multi_q_prebuild_batches(self, q_indices):
        from pyqed.pbc.gw.integrals import (
            _gdf_aux_coord_type,
            _gdf_auxbasis_name,
            _gdf_auxiliary_basis,
            _gdf_backend_settings,
            _gdf_rs_aux_engine,
            _gdf_rs_compact_auxiliary_basis,
            _gdf_short_range_workers,
            _gdf_uses_short_range,
        )

        q_indices = list(q_indices)
        if not self.stream_pairs or len(q_indices) < 2:
            return []
        ref = self._space.reference
        settings = _gdf_backend_settings(ref)
        mesh, kernel, omega = settings[2], settings[4], settings[5]
        if not _gdf_uses_short_range(kernel):
            return []
        pair_counts = [len(self.pair_keys(q_index)) for q_index in q_indices]
        max_pairs_per_q = max(pair_counts, default=0)
        if max_pairs_per_q == 0:
            return []
        aux = _gdf_auxiliary_basis(
            self._space,
            _gdf_auxbasis_name(ref),
            _gdf_aux_coord_type(ref),
        )
        rs_aux_engine = _gdf_rs_aux_engine(ref, aux, kernel, omega, mesh)
        compact_aux = _gdf_rs_compact_auxiliary_basis(aux, rs_aux_engine)
        if compact_aux.ncart == 0:
            return []
        workers = _gdf_short_range_workers(self.mf)
        itemsize = np.dtype(np.complex128).itemsize
        raw_pair_bytes = int(
            max(compact_aux.ncart, compact_aux.naux)
            * int(self.mf.cell.nao) ** 2
            * itemsize
        )
        pair_workspace_per_q = int(
            (3 * workers + 4) * raw_pair_bytes * max_pairs_per_q
        )
        metric_workspace_per_q = int(
            (workers + 1) * compact_aux.ncart**2 * itemsize
        )
        budget_q_batch_size = max(
            1,
            int(self.stream_pair_batch_mb * 1.0e6)
            // max(1, pair_workspace_per_q + metric_workspace_per_q),
        )
        if self.stream_pair_batch_size is None:
            q_batch_size = budget_q_batch_size
        else:
            q_batch_size = min(
                int(self.stream_pair_batch_size) // max_pairs_per_q,
                budget_q_batch_size,
            )
        if q_batch_size < 2:
            return []
        return [
            batch
            for batch in self._chunks(
                q_indices,
                min(q_batch_size, len(q_indices)),
            )
            if len(batch) >= 2
        ]

    def _materialize_many(self, q_indices, workers):
        q_indices = list(q_indices)
        if not q_indices:
            return
        worker_count = max(1, min(int(workers), len(q_indices)))
        if worker_count == 1:
            for q_index in q_indices:
                self._materialize_q(q_index)
            return
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            tuple(executor.map(self._materialize_q, q_indices))

    def _prebuild_and_materialize_q_batch(self, q_indices, workers):
        from pyqed.pbc.gw.integrals import (
            _gdf_mf_cache,
            _gdf_pair_ft_workers,
            _gdf_prebuild_short_range_q_batch,
            _gdf_short_range_workers,
        )

        q_indices = list(q_indices)
        short_range_cache = _gdf_mf_cache(
            self.mf,
            "three_center_ao_short_range",
        )
        metric_cache = _gdf_mf_cache(self.mf, "aux_metric_short_range")
        existing_keys = set(short_range_cache)
        existing_metric_keys = set(metric_cache)
        timings = {
            "q_indices": [int(q_index) for q_index in q_indices],
            "qpoints": int(len(q_indices)),
        }
        t0 = time.perf_counter()
        try:
            prebuilt = _gdf_prebuild_short_range_q_batch(
                self._space,
                q_indices,
                timings=timings,
            )
            timings["prebuilt_pair_blocks"] = int(prebuilt)
            workspace_budget = int(self.stream_pair_batch_mb * 1.0e6)
            workspace_upper_bound = int(
                timings.get("three_center_sr_group_workspace_bytes_upper_bound", 0)
                + timings.get(
                    "aux_metric_short_range_workspace_bytes_upper_bound",
                    0,
                )
            )
            timings["stream_pair_workspace_budget_bytes"] = workspace_budget
            timings["stream_pair_workspace_upper_bound_bytes"] = (
                workspace_upper_bound
            )
            timings["stream_pair_workspace_within_budget"] = bool(
                workspace_upper_bound <= workspace_budget
            )
            inner_workers = max(
                _gdf_pair_ft_workers(self.mf),
                _gdf_short_range_workers(self.mf),
            )
            materialize_workers = (
                1 if prebuilt and inner_workers > 1 else int(workers)
            )
            timings["inner_workers"] = int(inner_workers)
            timings["materialize_workers"] = int(materialize_workers)
            timings["nested_parallelism_avoided"] = bool(
                prebuilt and inner_workers > 1 and int(workers) > 1
            )
            self._materialize_many(q_indices, materialize_workers)
        finally:
            added_keys = set(short_range_cache) - existing_keys
            timings["unconsumed_pair_blocks"] = int(len(added_keys))
            for key in added_keys:
                short_range_cache.pop(key, None)
            added_metric_keys = set(metric_cache) - existing_metric_keys
            timings["unconsumed_metric_blocks"] = int(len(added_metric_keys))
            for key in added_metric_keys:
                metric_cache.pop(key, None)
            timings["total_seconds"] = float(time.perf_counter() - t0)
            self.multi_q_build_timings.append(timings)

    def build(self, q_indices=None, workers=None):
        """Prebuild persistent AO factors for selected momentum transfers."""

        from pyqed.pbc.gw.integrals import _gdf_should_use_opposite_q

        if q_indices is None:
            q_indices = range(len(self.qpts))
        q_indices = list(
            dict.fromkeys(self._space.normalize_q_index(q) for q in q_indices)
        )
        if workers is None:
            workers = getattr(self.mf, "gdf_prebuild_workers", 1)
        workers = max(1, min(int(workers), max(1, len(q_indices))))
        pending = [q_index for q_index in q_indices if not self._q_is_materialized(q_index)]
        source_indices = []
        for q_index in pending:
            source_q = _gdf_should_use_opposite_q(self._space, q_index)
            source_q = q_index if source_q is None else int(source_q)
            if source_q not in source_indices and not self._q_is_materialized(source_q):
                source_indices.append(source_q)

        batches = self._multi_q_prebuild_batches(source_indices)
        if batches:
            for batch in batches:
                self._prebuild_and_materialize_q_batch(batch, workers)
            remaining = [
                q_index for q_index in pending if not self._q_is_materialized(q_index)
            ]
            self._materialize_many(remaining, workers)
            return self

        self._materialize_many(pending, workers)
        return self

    def metric_info(self, q_index):
        """Return metric metadata retained after raw AO blocks are released."""

        q_index = self._space.normalize_q_index(q_index)
        configuration_key = self._sync_configuration()
        key = (configuration_key, q_index)
        if key not in self._q_metadata:
            self._materialize_q(q_index)
        return self._q_metadata[key]

    def pair_keys(self, q_index):
        from pyqed.pbc.gw.integrals import _pair_keys_for_q

        return _pair_keys_for_q(self._space, self._space.normalize_q_index(q_index))

    def get_jk(self, dm):
        """Contract AO densities with persistent GDF factors."""

        densities = [np.asarray(dm)] if self.nkpts == 1 and np.asarray(dm).ndim == 2 else list(dm)
        if len(densities) != self.nkpts:
            raise ValueError(
                f"dm must provide one AO density for each of {self.nkpts} k-points."
            )
        nao = int(self.mf.cell.nao)
        for density in densities:
            if density.shape != (nao, nao):
                raise ValueError(f"Each AO density must have shape ({nao}, {nao}).")

        q0 = self._space.q0_index
        density_factor = None
        for k_index, density in enumerate(densities):
            block = self._resolve_factor(
                self._cderi_factor(q0, k_index, k_index)
            )
            if density_factor is None:
                rank = (
                    block.rank
                    if isinstance(block, PackedHermitianCDERI)
                    else block.shape[0]
                )
                density_factor = np.zeros(rank, dtype=np.complex128)
            if isinstance(block, PackedHermitianCDERI):
                density_factor += block.contract_density(density)
            else:
                density_factor += np.einsum(
                    "Pij,ji->P", block, density, optimize=True
                )
        density_factor /= self.nkpts

        vj = []
        vk = []
        for k_index in range(self.nkpts):
            diagonal = self.cderi(q0, k_index, k_index)
            j_block = np.einsum(
                "Pij,P->ij", diagonal, density_factor.conj(), optimize=True
            )
            k_block = np.zeros((nao, nao), dtype=np.complex128)
            for kq_index, density in enumerate(densities):
                q_index = int(self._space.q_index_by_kpair[k_index, kq_index])
                block = self.cderi(q_index, k_index, kq_index)
                k_block += np.einsum(
                    "Pim,mn,Pjn->ij",
                    block,
                    density,
                    block.conj(),
                    optimize=True,
                )
            vj.append(0.5 * (j_block + j_block.conj().T))
            k_block /= self.nkpts
            vk.append(0.5 * (k_block + k_block.conj().T))
        return np.asarray(vj), np.asarray(vk)

    def get_jk_response(self, dm_q, q_index):
        """Contract non-Hermitian ``k -> k+q`` AO response densities.

        ``dm_q[k]`` has rows at ``k+q`` and columns at ``k``.  Returned J/K
        blocks use the same orientation and are not Hermitian-symmetrized.
        """

        q_index = self._space.normalize_q_index(q_index)
        pair_keys = self.pair_keys(q_index)
        pair_by_k = {int(k): int(kq) for k, kq in pair_keys}
        if len(pair_by_k) != self.nkpts:
            raise RuntimeError("The q block does not map every SCF k point.")
        densities = list(dm_q)
        if len(densities) != self.nkpts:
            raise ValueError(
                f"dm_q must provide one response density for each of "
                f"{self.nkpts} k-points."
            )
        nao = int(self.mf.cell.nao)
        densities = [np.asarray(density, dtype=np.complex128) for density in densities]
        if any(density.shape != (nao, nao) for density in densities):
            raise ValueError(f"Each response density must have shape ({nao}, {nao}).")

        density_factor = None
        for k_index, density in enumerate(densities):
            kq_index = pair_by_k[k_index]
            block = self.cderi(q_index, k_index, kq_index)
            if density_factor is None:
                density_factor = np.zeros(block.shape[0], dtype=np.complex128)
            if block.shape[0] != len(density_factor):
                raise RuntimeError("Inconsistent auxiliary rank within one q block.")
            density_factor += np.einsum(
                "Pij,ji->P", block, density, optimize=True
            )
        density_factor /= self.nkpts

        vj = []
        vk = []
        q_index_by_pair = np.asarray(self._space.q_index_by_kpair, dtype=int)
        for k_index in range(self.nkpts):
            kq_index = pair_by_k[k_index]
            q_block = self.cderi(q_index, k_index, kq_index)
            j_block = np.einsum(
                "Pij,P->ji",
                q_block.conj(),
                density_factor,
                optimize=True,
            )
            k_block = np.zeros((nao, nao), dtype=np.complex128)
            for source_k in range(self.nkpts):
                source_kq = pair_by_k[source_k]
                transfer = int(q_index_by_pair[k_index, source_k])
                left = self.cderi(transfer, kq_index, source_kq)
                right = self.cderi(transfer, k_index, source_k)
                k_block += np.einsum(
                    "Pam,mn,Pbn->ab",
                    left,
                    densities[source_k],
                    right.conj(),
                    optimize=True,
                )
            vj.append(j_block)
            vk.append(k_block / self.nkpts)
        vj = np.asarray(vj)
        vk = np.asarray(vk)

        minus_q_index = self.find_qpoint_index(-np.asarray(self.qpts[q_index]))
        if int(minus_q_index) == int(q_index):
            scale = max(
                max(float(np.max(np.abs(density))) for density in densities),
                1.0,
            )
            residual = max(
                float(
                    np.max(
                        np.abs(
                            densities[kq_index]
                            - densities[k_index].conj().T
                        )
                    )
                )
                for k_index, kq_index in pair_by_k.items()
            )
            if residual <= 1.0e-10 * scale:
                visited = set()
                for k_index, kq_index in pair_by_k.items():
                    pair = tuple(sorted((k_index, kq_index)))
                    if pair in visited:
                        continue
                    visited.add(pair)
                    for blocks in (vj, vk):
                        average = 0.5 * (
                            blocks[k_index] + blocks[kq_index].conj().T
                        )
                        blocks[k_index] = average
                        blocks[kq_index] = average.conj().T
        return vj, vk

    def clear(self):
        """Drop whitened blocks owned by this backend."""

        with self._lock:
            files = set(self.cache_files)
            self._cderi_cache.clear()
            self._q_metadata.clear()
            self.build_timings.clear()
            self.multi_q_build_timings.clear()
            self._memory_bytes = 0
            self._configuration_key = None
            for filename in files:
                Path(filename).unlink(missing_ok=True)
            if self._disk_dir is not None:
                try:
                    self._disk_dir.rmdir()
                except OSError:
                    pass
                self._disk_dir = None

    close = clear
