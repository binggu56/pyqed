"""Packed SU(2) qchem sweep-table helpers.

The objects in this module are the Python-owned ABI for the long-term native
SU(2) qchem sweep path.  They keep sector/channel metadata in integer arrays
and dense payloads in contiguous pools, while preserving the existing Python
reference grouped tables as the fallback source of truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

_USE_PACKED_COMPILED_TERMS = False
_DEBUG_PACKED_COMPILED_TERMS = False
_USE_CYTHON_PARENT_BLOCKS = False
_RANK_COUPLED_FACTOR_METADATA_CACHE = {}
_FACTOR_MATCH_LAYOUT_CACHE = {}
_COMPONENT_PARENT_BLOCK_LAYOUT_CACHE = {}
_BASIS_SIGNATURE_CACHE = {}
_COMPONENT_PARENT_BLOCK_LAYOUT_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "puts": 0,
}
_FACTOR_GROUP_PACK_CACHE_MAX_SIZE = 2048
_SU2_KERNEL_MODULE = None
_SU2_KERNEL_IMPORT_ATTEMPTED = False


def _sort_key(value):
    return repr(value)


def _as_array(value, *, dtype=None):
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _array_tuple(array):
    return tuple(int(value) for value in np.asarray(array, dtype=np.int64).reshape(-1))


def _has_nonzero(array, tol=0.0):
    arr = array if isinstance(array, np.ndarray) else np.asarray(array)
    if not arr.size:
        return False
    if float(tol) <= 0.0:
        return bool(arr.any())
    return bool(np.any(np.abs(arr) > float(tol)))


def _codec_signature(codec):
    return tuple(repr(sector) for sector in tuple(getattr(codec, "sectors", ())))


def _su2_kernel_module():
    global _SU2_KERNEL_IMPORT_ATTEMPTED
    global _SU2_KERNEL_MODULE

    if not _SU2_KERNEL_IMPORT_ATTEMPTED:
        _SU2_KERNEL_IMPORT_ATTEMPTED = True
        try:
            from pyqed.mps.nonabelian import _su2_kernel as module
        except Exception:
            module = None
        _SU2_KERNEL_MODULE = module
    return _SU2_KERNEL_MODULE


def _factorize_rank_coupled_boundary_term(
    boundary_block,
    W_block,
    *,
    representation,
    left_reference,
    right_reference,
):
    module = _su2_kernel_module()
    if module is not None:
        if str(representation) == "rank_coupled_left_factor_by_ket":
            kernel = getattr(module, "factorize_rank_coupled_left", None)
            if kernel is not None:
                factor = kernel(boundary_block, W_block)
                if factor is not None:
                    return np.asarray(factor)
        else:
            kernel = getattr(module, "factorize_rank_coupled_right", None)
            if kernel is not None:
                factor = kernel(W_block, boundary_block)
                if factor is not None:
                    return np.asarray(factor)
    if str(representation) == "rank_coupled_left_factor_by_ket":
        return np.asarray(left_reference(boundary_block, W_block))
    return np.asarray(right_reference(W_block, boundary_block))


def _rank_coupled_factor_kernels(boundary_table, W, representation):
    module = _su2_kernel_module()
    if module is None:
        return None, None
    real_dtype = not np.iscomplexobj(boundary_table.block_pool.data)
    try:
        real_dtype = real_dtype and not np.issubdtype(
            np.dtype(getattr(W, "dtype", float)),
            np.complexfloating,
        )
    except TypeError:
        real_dtype = False
    if str(representation) == "rank_coupled_left_factor_by_ket":
        return (
            getattr(
                module,
                "factorize_rank_coupled_left_real"
                if real_dtype
                else "factorize_rank_coupled_left",
                None,
            ),
            None,
        )
    return (
        None,
        getattr(
            module,
            "factorize_rank_coupled_right_real"
            if real_dtype
            else "factorize_rank_coupled_right",
            None,
        ),
    )


@dataclass(frozen=True)
class PackedArrayPool:
    """Contiguous storage for variable-shape dense blocks."""

    data: np.ndarray
    offsets: np.ndarray
    shape_offsets: np.ndarray
    shapes: np.ndarray
    _shape_cache: object = field(default=None, compare=False, repr=False)
    _array_cache: object = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        n_arrays = max(0, int(self.offsets.size) - 1)
        if self._shape_cache is None:
            object.__setattr__(self, "_shape_cache", [None] * n_arrays)
        if self._array_cache is None:
            object.__setattr__(self, "_array_cache", [None] * n_arrays)

    @classmethod
    def from_arrays(cls, arrays):
        raw_arrays = tuple(np.asarray(array) for array in arrays)
        dtype = np.result_type(*(array.dtype for array in raw_arrays), float)
        arrays = tuple(_as_array(array, dtype=dtype) for array in raw_arrays)
        offsets = [0]
        shape_offsets = [0]
        shapes = []
        flat = []
        for array in arrays:
            flat.append(array.reshape(-1))
            offsets.append(offsets[-1] + int(array.size))
            shapes.extend(int(dim) for dim in array.shape)
            shape_offsets.append(shape_offsets[-1] + int(array.ndim))
        data = (
            np.concatenate(flat).astype(dtype, copy=False)
            if flat
            else np.zeros(0, dtype=float)
        )
        return cls(
            data=np.ascontiguousarray(data),
            offsets=np.asarray(offsets, dtype=np.int64),
            shape_offsets=np.asarray(shape_offsets, dtype=np.int64),
            shapes=np.asarray(shapes, dtype=np.int64),
        )

    @property
    def n_arrays(self):
        return max(0, int(self.offsets.size) - 1)

    @property
    def stored_elements(self):
        return int(self.data.size)

    def shape(self, index):
        index = int(index)
        cached = self._shape_cache[index]
        if cached is not None:
            return cached
        start = int(self.shape_offsets[index])
        stop = int(self.shape_offsets[index + 1])
        shape = tuple(int(dim) for dim in self.shapes[start:stop])
        self._shape_cache[index] = shape
        return shape

    def array(self, index):
        index = int(index)
        cached = self._array_cache[index]
        if cached is not None:
            return cached
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        array = self.data[start:stop].reshape(self.shape(index))
        self._array_cache[index] = array
        return array

    @property
    def stats(self):
        return {
            "n_arrays": int(self.n_arrays),
            "stored_elements": int(self.stored_elements),
            "shape_entries": int(self.shapes.size),
        }


@dataclass(frozen=True)
class SectorCodec:
    """Stable sector-to-integer map for one packed table."""

    sectors: tuple
    index_map: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        if not self.index_map:
            object.__setattr__(
                self,
                "index_map",
                {sector: idx for idx, sector in enumerate(self.sectors)},
            )

    @classmethod
    def from_iterable(cls, sectors):
        unique = {sector: None for sector in sectors}
        return cls(tuple(sorted(unique, key=_sort_key)))

    @property
    def index(self):
        return self.index_map

    def id(self, sector):
        return self.index[sector]

    @property
    def stats(self):
        return {"n_sectors": int(len(self.sectors))}


@dataclass(frozen=True)
class PackedSU2BoundaryTable:
    """Packed rank-coupled boundary blocks grouped by ket sector."""

    side: str
    bond: int
    representation: str
    sector_codec: SectorCodec
    ket_sector_ids: np.ndarray
    entry_offsets: np.ndarray
    out_sector_ids: np.ndarray
    channel_offsets: np.ndarray
    channel_ids: np.ndarray
    block_pool: PackedArrayPool
    _channel_index_cache: object = field(default=None, compare=False, repr=False)
    _channel_lookup_cache: object = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if self._channel_index_cache is None:
            object.__setattr__(self, "_channel_index_cache", None)

    @classmethod
    def from_grouped(cls, grouped_by_ket, *, side, bond, representation):
        if str(representation) != "rank_coupled_by_ket":
            return None
        rows = []
        arrays = []
        for q_ket, entries in sorted(grouped_by_ket.items(), key=lambda item: _sort_key(item[0])):
            row = []
            for q_out, channel_blocks in tuple(entries or ()):
                channel_items = []
                for channel, block in sorted(
                    dict(channel_blocks or {}).items(),
                    key=lambda item: int(item[0]),
                ):
                    channel_items.append((int(channel), len(arrays)))
                    arrays.append(block)
                if channel_items:
                    row.append((q_out, tuple(channel_items)))
            rows.append((q_ket, tuple(row)))
        return cls.from_rows(rows, arrays, side=side, bond=bond, representation=representation)

    @classmethod
    def from_rows(cls, rows, arrays, *, side, bond, representation):
        if str(representation) != "rank_coupled_by_ket":
            return None
        sectors = []
        for q_ket, entries in rows:
            sectors.append(q_ket)
            for q_out, _channels in tuple(entries or ()):
                sectors.append(q_out)
        codec = SectorCodec.from_iterable(sectors)
        sector_index = codec.index
        ket_ids = []
        entry_offsets = [0]
        out_ids = []
        channel_offsets = [0]
        channel_ids = []
        ordered_arrays = []
        for q_ket, entries in rows:
            ket_ids.append(sector_index[q_ket])
            for q_out, channels in entries:
                out_ids.append(sector_index[q_out])
                for channel, array_index in channels:
                    channel_ids.append(int(channel))
                    ordered_arrays.append(arrays[int(array_index)])
                channel_offsets.append(len(channel_ids))
            entry_offsets.append(len(out_ids))
        return cls(
            side=str(side),
            bond=int(bond),
            representation=str(representation),
            sector_codec=codec,
            ket_sector_ids=np.asarray(ket_ids, dtype=np.int64),
            entry_offsets=np.asarray(entry_offsets, dtype=np.int64),
            out_sector_ids=np.asarray(out_ids, dtype=np.int64),
            channel_offsets=np.asarray(channel_offsets, dtype=np.int64),
            channel_ids=np.asarray(channel_ids, dtype=np.int64),
            block_pool=PackedArrayPool.from_arrays(ordered_arrays),
        )

    @property
    def n_ket_sectors(self):
        return int(self.ket_sector_ids.size)

    @property
    def n_entries(self):
        return int(self.out_sector_ids.size)

    @property
    def n_channel_blocks(self):
        return int(self.channel_ids.size)

    def channel_index_maps(self):
        """Return cached ``channel -> block_pool index`` maps for each entry."""

        cached = self._channel_index_cache
        if cached is not None:
            return cached
        maps = []
        for entry_idx in range(self.n_entries):
            channel_start = int(self.channel_offsets[entry_idx])
            channel_stop = int(self.channel_offsets[entry_idx + 1])
            maps.append(
                {
                    int(self.channel_ids[channel_idx]): channel_idx
                    for channel_idx in range(channel_start, channel_stop)
                }
            )
        object.__setattr__(self, "_channel_index_cache", tuple(maps))
        return self._channel_index_cache

    def channel_index_lookup(self, max_channel):
        """
        Return a dense ``[entry, channel] -> block_pool index`` lookup table.

        Rank-coupled qchem MPO channel ids are compact enough in normal use
        that this avoids rebuilding a Python dictionary for every boundary
        entry while constructing factor tables.
        """

        max_channel = int(max_channel)
        if max_channel < 0:
            return None
        if max_channel > 4096:
            return None
        cached = self._channel_lookup_cache
        if cached is not None and int(cached[0]) >= max_channel:
            return cached[1]
        lookup = np.full(
            (int(self.n_entries), max_channel + 1),
            -1,
            dtype=np.int64,
        )
        for entry_idx in range(self.n_entries):
            channel_start = int(self.channel_offsets[entry_idx])
            channel_stop = int(self.channel_offsets[entry_idx + 1])
            for channel_idx in range(channel_start, channel_stop):
                channel = int(self.channel_ids[channel_idx])
                if 0 <= channel <= max_channel:
                    lookup[int(entry_idx), channel] = int(channel_idx)
        object.__setattr__(self, "_channel_lookup_cache", (max_channel, lookup))
        return lookup

    @property
    def stats(self):
        return {
            "kind": "packed_su2_boundary_table",
            "side": str(self.side),
            "bond": int(self.bond),
            "representation": str(self.representation),
            "n_ket_sectors": int(self.n_ket_sectors),
            "n_entries": int(self.n_entries),
            "n_channel_blocks": int(self.n_channel_blocks),
            "sectors": int(len(self.sector_codec.sectors)),
            "block_pool": self.block_pool.stats,
        }


class PackedRankCoupledBoundaryPayloads(Mapping):
    """
    Lazy legacy payload view backed by a packed rank-coupled boundary table.

    The moving qchem path consumes ``packed_table`` directly.  The mapping API
    exists only for reference/debug paths that still expect
    ``(q_out, q_in, channel) -> block`` payload dictionaries.
    """

    def __init__(self, packed_table):
        if not isinstance(packed_table, PackedSU2BoundaryTable):
            raise TypeError("PackedRankCoupledBoundaryPayloads requires a packed boundary table.")
        self.packed_table = packed_table
        self._dict_cache = None

    def _materialized(self):
        if self._dict_cache is not None:
            return self._dict_cache
        sectors = tuple(self.packed_table.sector_codec.sectors)
        payloads = {}
        for row_idx, ket_id in enumerate(self.packed_table.ket_sector_ids):
            q_in = sectors[int(ket_id)]
            entry_start = int(self.packed_table.entry_offsets[row_idx])
            entry_stop = int(self.packed_table.entry_offsets[row_idx + 1])
            for entry_idx in range(entry_start, entry_stop):
                q_out = sectors[int(self.packed_table.out_sector_ids[entry_idx])]
                channel_start = int(self.packed_table.channel_offsets[entry_idx])
                channel_stop = int(self.packed_table.channel_offsets[entry_idx + 1])
                for channel_idx in range(channel_start, channel_stop):
                    channel = int(self.packed_table.channel_ids[channel_idx])
                    payloads[(q_out, q_in, channel)] = self.packed_table.block_pool.array(
                        channel_idx
                    )
        self._dict_cache = payloads
        return payloads

    def __getitem__(self, key):
        return self._materialized()[key]

    def __iter__(self):
        return iter(self._materialized())

    def __len__(self):
        return int(self.packed_table.n_channel_blocks)

    def items(self):
        return self._materialized().items()

    def values(self):
        return self._materialized().values()

    @property
    def stats(self):
        return {
            "kind": "packed_rank_coupled_boundary_payloads",
            "n_payloads": int(len(self)),
            "packed_table": self.packed_table.stats,
            "materialized": bool(self._dict_cache is not None),
        }


@dataclass(frozen=True)
class PackedSU2FactorTable:
    """Packed rank-coupled one-site factor table."""

    side: str
    bond: int
    representation: str
    boundary_codec: SectorCodec
    physical_codec: SectorCodec
    family_labels: tuple
    key_boundary_ids: np.ndarray
    key_physical_ids: np.ndarray
    entry_offsets: np.ndarray
    out_boundary_ids: np.ndarray
    out_physical_ids: np.ndarray
    middle_ids: np.ndarray
    family_offsets: np.ndarray
    family_ids: np.ndarray
    factor_indices: np.ndarray
    factor_pool: PackedArrayPool
    key_index_map: dict = field(default_factory=dict, compare=False, repr=False)
    _families_cache: object = field(default=None, compare=False, repr=False)
    _layout_signature_cache: object = field(default=None, compare=False, repr=False)
    _shape_signature_cache: object = field(default=None, compare=False, repr=False)
    _factor_group_pack_cache: object = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    _factor_group_pack_cache_stats: object = field(
        default_factory=lambda: {"hits": 0, "misses": 0, "puts": 0, "clears": 0},
        compare=False,
        repr=False,
    )

    def __post_init__(self):
        if not self.key_index_map:
            object.__setattr__(
                self,
                "key_index_map",
                {
                    (int(boundary_id), int(phys_id)): idx
                    for idx, (boundary_id, phys_id) in enumerate(
                        zip(self.key_boundary_ids, self.key_physical_ids)
                    )
                },
            )
        if self._families_cache is None:
            object.__setattr__(self, "_families_cache", [None] * int(self.n_entries))

    @classmethod
    def from_grouped(cls, grouped_by_key, *, side, bond, representation):
        if str(representation) not in {
            "rank_coupled_left_factor_by_ket",
            "rank_coupled_right_factor_by_ket",
        }:
            return None
        boundary_sectors = []
        physical_sectors = []
        rows = []
        arrays = []
        family_names = set()
        for key, entries in sorted(grouped_by_key.items(), key=lambda item: repr(item[0])):
            if len(key) != 2:
                return None
            q_boundary, q_phys = key
            boundary_sectors.append(q_boundary)
            physical_sectors.append(q_phys)
            row = []
            for entry in tuple(entries or ()):
                if len(entry) < 4:
                    return None
                q_out_boundary, q_out_phys, middle_idx, factor = entry[:4]
                families = tuple(str(name) for name in (entry[4] if len(entry) > 4 else ()))
                boundary_sectors.append(q_out_boundary)
                physical_sectors.append(q_out_phys)
                family_names.update(families)
                row.append(
                    (
                        q_out_boundary,
                        q_out_phys,
                        int(middle_idx),
                        len(arrays),
                        families,
                    )
                )
                arrays.append(factor)
            rows.append((q_boundary, q_phys, tuple(row)))
        return cls.from_rows(
            rows,
            arrays,
            side=side,
            bond=bond,
            representation=representation,
            boundary_sectors=boundary_sectors,
            physical_sectors=physical_sectors,
        )

    @classmethod
    def from_rows(
        cls,
        rows,
        arrays,
        *,
        side,
        bond,
        representation,
        boundary_sectors=(),
        physical_sectors=(),
    ):
        """Pack normalized factor-table rows into integer/offset storage."""

        boundary_sectors = list(boundary_sectors)
        physical_sectors = list(physical_sectors)
        family_names = set()
        for q_boundary, q_phys, entries in rows:
            boundary_sectors.append(q_boundary)
            physical_sectors.append(q_phys)
            for q_out_boundary, q_out_phys, _middle_idx, _array_index, families in entries:
                boundary_sectors.append(q_out_boundary)
                physical_sectors.append(q_out_phys)
                family_names.update(tuple(str(name) for name in families))
        boundary_codec = SectorCodec.from_iterable(boundary_sectors)
        physical_codec = SectorCodec.from_iterable(physical_sectors)
        boundary_index = boundary_codec.index
        physical_index = physical_codec.index
        family_labels = tuple(sorted(family_names))
        family_index = {name: idx for idx, name in enumerate(family_labels)}
        key_boundary_ids = []
        key_physical_ids = []
        entry_offsets = [0]
        out_boundary_ids = []
        out_physical_ids = []
        middle_ids = []
        family_offsets = [0]
        family_ids = []
        factor_indices = []
        for q_boundary, q_phys, entries in rows:
            key_boundary_ids.append(boundary_index[q_boundary])
            key_physical_ids.append(physical_index[q_phys])
            for q_out_boundary, q_out_phys, middle_idx, _array_index, families in entries:
                out_boundary_ids.append(boundary_index[q_out_boundary])
                out_physical_ids.append(physical_index[q_out_phys])
                middle_ids.append(int(middle_idx))
                factor_indices.append(int(_array_index))
                for name in families:
                    family_ids.append(family_index[str(name)])
                family_offsets.append(len(family_ids))
            entry_offsets.append(len(out_boundary_ids))
        return cls(
            side=str(side),
            bond=int(bond),
            representation=str(representation),
            boundary_codec=boundary_codec,
            physical_codec=physical_codec,
            family_labels=family_labels,
            key_boundary_ids=np.asarray(key_boundary_ids, dtype=np.int64),
            key_physical_ids=np.asarray(key_physical_ids, dtype=np.int64),
            entry_offsets=np.asarray(entry_offsets, dtype=np.int64),
            out_boundary_ids=np.asarray(out_boundary_ids, dtype=np.int64),
            out_physical_ids=np.asarray(out_physical_ids, dtype=np.int64),
            middle_ids=np.asarray(middle_ids, dtype=np.int64),
            family_offsets=np.asarray(family_offsets, dtype=np.int64),
            family_ids=np.asarray(family_ids, dtype=np.int64),
            factor_indices=np.asarray(factor_indices, dtype=np.int64),
            factor_pool=PackedArrayPool.from_arrays(arrays),
        )

    @classmethod
    def from_integer_data(
        cls,
        *,
        side,
        bond,
        representation,
        boundary_codec,
        physical_codec,
        family_labels,
        key_boundary_ids,
        key_physical_ids,
        entry_offsets,
        out_boundary_ids,
        out_physical_ids,
        middle_ids,
        family_offsets,
        family_ids,
        factor_indices,
        arrays,
    ):
        """Build a packed factor table from already encoded rows."""

        return cls(
            side=str(side),
            bond=int(bond),
            representation=str(representation),
            boundary_codec=boundary_codec,
            physical_codec=physical_codec,
            family_labels=tuple(str(name) for name in family_labels),
            key_boundary_ids=np.asarray(key_boundary_ids, dtype=np.int64),
            key_physical_ids=np.asarray(key_physical_ids, dtype=np.int64),
            entry_offsets=np.asarray(entry_offsets, dtype=np.int64),
            out_boundary_ids=np.asarray(out_boundary_ids, dtype=np.int64),
            out_physical_ids=np.asarray(out_physical_ids, dtype=np.int64),
            middle_ids=np.asarray(middle_ids, dtype=np.int64),
            family_offsets=np.asarray(family_offsets, dtype=np.int64),
            family_ids=np.asarray(family_ids, dtype=np.int64),
            factor_indices=np.asarray(factor_indices, dtype=np.int64),
            factor_pool=PackedArrayPool.from_arrays(arrays),
        )

    @property
    def n_keys(self):
        return int(self.key_boundary_ids.size)

    @property
    def n_entries(self):
        return int(self.out_boundary_ids.size)

    def _key_index(self):
        return self.key_index_map

    def key_id(self, q_boundary, q_phys):
        try:
            return (
                self.boundary_codec.index[q_boundary],
                self.physical_codec.index[q_phys],
            )
        except KeyError:
            return None

    def entry_range_for_key(self, q_boundary, q_phys):
        key = self.key_id(q_boundary, q_phys)
        if key is None:
            return None
        return self.entry_range_for_ids(*key)

    def entry_range_for_ids(self, boundary_id, phys_id):
        key = (int(boundary_id), int(phys_id))
        idx = self._key_index().get(key)
        if idx is None:
            return None
        return int(self.entry_offsets[idx]), int(self.entry_offsets[idx + 1])

    def factor(self, entry_idx):
        return self.factor_pool.array(int(self.factor_indices[int(entry_idx)]))

    def packed_component_parent_group(self, group, role):
        """
        Return a cached stacked/packed factor group for parent-block builds.

        ``role`` is ``"left"`` for arrays shaped as
        ``(t,l,k,w,a,b)`` and ``"right"`` for arrays shaped as
        ``(t,w,q,r,d,c)``.  The returned value is ``(stack, meta)`` where
        ``meta`` is ``None`` for unsupported ranks or ``(matrix, dims)`` for
        the GEMM-friendly packed representation.
        """

        role = str(role)
        if role not in {"left", "right"}:
            raise ValueError(f"Unknown component parent factor role {role!r}.")
        group = tuple(int(idx) for idx in group)
        key = (role, group)
        cached = self._factor_group_pack_cache.get(key)
        if cached is not None:
            self._factor_group_pack_cache_stats["hits"] += 1
            return cached
        self._factor_group_pack_cache_stats["misses"] += 1
        stack = np.ascontiguousarray(
            np.stack([self.factor(int(idx)) for idx in group], axis=0)
        )
        if stack.ndim != 6:
            packed = (stack, None)
        elif role == "left":
            tdim, ldim, kdim, wdim, adim, bdim = (
                int(dim) for dim in stack.shape
            )
            mat = np.ascontiguousarray(
                stack.transpose(1, 4, 2, 5, 0, 3).reshape(
                    ldim * adim * kdim * bdim,
                    tdim * wdim,
                )
            )
            packed = (stack, (mat, (tdim, ldim, kdim, wdim, adim, bdim)))
        else:
            tdim, wdim, qdim, rdim, ddim, cdim = (
                int(dim) for dim in stack.shape
            )
            mat = np.ascontiguousarray(
                stack.transpose(0, 1, 4, 2, 5, 3).reshape(
                    tdim * wdim,
                    ddim * qdim * cdim * rdim,
                )
            )
            packed = (stack, (mat, (tdim, wdim, qdim, rdim, ddim, cdim)))
        if len(self._factor_group_pack_cache) >= _FACTOR_GROUP_PACK_CACHE_MAX_SIZE:
            self._factor_group_pack_cache.clear()
            self._factor_group_pack_cache_stats["clears"] += 1
        self._factor_group_pack_cache[key] = packed
        self._factor_group_pack_cache_stats["puts"] += 1
        return packed

    def families(self, entry_idx):
        entry_idx = int(entry_idx)
        cached = self._families_cache[entry_idx]
        if cached is not None:
            return cached
        start = int(self.family_offsets[entry_idx])
        stop = int(self.family_offsets[entry_idx + 1])
        cached = tuple(str(self.family_labels[int(idx)]) for idx in self.family_ids[start:stop])
        self._families_cache[entry_idx] = cached
        return cached

    @property
    def stats(self):
        return {
            "kind": "packed_su2_factor_table",
            "side": str(self.side),
            "bond": int(self.bond),
            "representation": str(self.representation),
            "n_keys": int(self.n_keys),
            "n_entries": int(self.n_entries),
            "n_unique_factors": int(self.factor_pool.n_arrays),
            "boundary_sectors": int(len(self.boundary_codec.sectors)),
            "physical_sectors": int(len(self.physical_codec.sectors)),
            "families": int(len(self.family_labels)),
            "factor_pool": self.factor_pool.stats,
            "factor_group_pack_cache": {
                "size": int(len(self._factor_group_pack_cache)),
                **{
                    str(key): int(value)
                    for key, value in self._factor_group_pack_cache_stats.items()
                },
            },
        }


def pack_rank_coupled_factor_table_from_boundary(
    boundary_table,
    W,
    *,
    side,
    bond,
    representation,
):
    """
    Build a packed rank-coupled one-site factor table from a packed boundary.

    This is the packed SU(2) qchem route used by moving environments: it
    consumes integer/offset boundary rows directly and avoids first rebuilding
    the legacy ``{sector: entries}`` factor-table dictionary.
    """

    if not isinstance(boundary_table, PackedSU2BoundaryTable):
        return None
    if str(representation) not in {
        "rank_coupled_left_factor_by_ket",
        "rank_coupled_right_factor_by_ket",
    }:
        return None
    from .renormalized import (
        _symbolic_transition_families_by_channel,
        factorize_left_two_site_dense_term,
        factorize_right_two_site_dense_term,
        group_rank_coupled_reduced_blocks_by_input,
    )

    rep = str(representation)
    family_side = "right" if rep == "rank_coupled_left_factor_by_ket" else "left"
    metadata_key = (id(W), rep, str(family_side))
    metadata = _RANK_COUPLED_FACTOR_METADATA_CACHE.get(metadata_key)
    if metadata is None:
        w_blocks_by_in = group_rank_coupled_reduced_blocks_by_input(W)
        family_by_middle = _symbolic_transition_families_by_channel(W, side=family_side)
        physical_sectors = []
        w_schedule = []
        for q_phys_ket, w_entries in sorted(
            w_blocks_by_in.items(),
            key=lambda item: _sort_key(item[0]),
        ):
            physical_sectors.append(q_phys_ket)
            out_entries = []
            for q_phys_out, W_blocks in tuple(w_entries or ()):
                physical_sectors.append(q_phys_out)
                out_entries.append(
                    (
                        q_phys_out,
                        tuple(
                            (int(pair[0]), int(pair[1]), block)
                            for pair, block in sorted(
                                dict(W_blocks or {}).items(),
                                key=lambda item: (int(item[0][0]), int(item[0][1])),
                            )
                        ),
                    )
                )
            w_schedule.append((q_phys_ket, tuple(out_entries)))
        physical_codec = SectorCodec.from_iterable(physical_sectors)
        family_labels = tuple(
            sorted(
                {
                    str(name)
                    for names in family_by_middle.values()
                    for name in tuple(names or ())
                }
            )
        )
        family_index = {name: idx for idx, name in enumerate(family_labels)}
        metadata = (
            tuple(w_schedule),
            physical_codec,
            family_by_middle,
            family_labels,
            family_index,
        )
        _RANK_COUPLED_FACTOR_METADATA_CACHE[metadata_key] = metadata
        if len(_RANK_COUPLED_FACTOR_METADATA_CACHE) > 128:
            _RANK_COUPLED_FACTOR_METADATA_CACHE.clear()
    if len(metadata) == 2:
        w_blocks_by_in, family_by_middle = metadata
        physical_sectors = []
        w_schedule = []
        for q_phys_ket, w_entries in sorted(
            w_blocks_by_in.items(),
            key=lambda item: _sort_key(item[0]),
        ):
            physical_sectors.append(q_phys_ket)
            out_entries = []
            for q_phys_out, W_blocks in tuple(w_entries or ()):
                physical_sectors.append(q_phys_out)
                out_entries.append(
                    (
                        q_phys_out,
                        tuple(
                            (int(pair[0]), int(pair[1]), block)
                            for pair, block in sorted(
                                dict(W_blocks or {}).items(),
                                key=lambda item: (int(item[0][0]), int(item[0][1])),
                            )
                        ),
                    )
                )
            w_schedule.append((q_phys_ket, tuple(out_entries)))
        w_schedule = tuple(w_schedule)
        physical_codec = SectorCodec.from_iterable(physical_sectors)
        family_labels = tuple(
            sorted(
                {
                    str(name)
                    for names in family_by_middle.values()
                    for name in tuple(names or ())
                }
            )
        )
        family_index = {name: idx for idx, name in enumerate(family_labels)}
    elif len(metadata) == 3:
        w_schedule, physical_sectors, family_by_middle = metadata
        physical_codec = SectorCodec.from_iterable(physical_sectors)
        family_labels = tuple(
            sorted(
                {
                    str(name)
                    for names in family_by_middle.values()
                    for name in tuple(names or ())
                }
            )
        )
        family_index = {name: idx for idx, name in enumerate(family_labels)}
    else:
        w_schedule, physical_codec, family_by_middle, family_labels, family_index = metadata
    physical_index = physical_codec.index
    left_factor_kernel, right_factor_kernel = _rank_coupled_factor_kernels(
        boundary_table,
        W,
        rep,
    )
    key_boundary_ids = []
    key_physical_ids = []
    entry_offsets = [0]
    out_boundary_ids = []
    out_physical_ids = []
    middle_ids = []
    family_offsets = [0]
    family_ids = []
    factor_indices = []
    arrays = []
    factor_cache = {}
    factor_index_cache = {}
    max_boundary_channel = -1
    for _q_phys_ket, w_entries in w_schedule:
        for _q_phys_out, W_blocks in w_entries:
            for left_channel, right_channel, _W_block in W_blocks:
                channel = (
                    int(left_channel)
                    if rep == "rank_coupled_left_factor_by_ket"
                    else int(right_channel)
                )
                max_boundary_channel = max(max_boundary_channel, channel)
    channel_lookup = boundary_table.channel_index_lookup(max_boundary_channel)
    channel_maps = None if channel_lookup is not None else boundary_table.channel_index_maps()

    for key_row, q_boundary_ket_id in enumerate(boundary_table.ket_sector_ids):
        q_boundary_ket_id = int(q_boundary_ket_id)
        entry_start = int(boundary_table.entry_offsets[key_row])
        entry_stop = int(boundary_table.entry_offsets[key_row + 1])
        if entry_start == entry_stop:
            continue
        for q_phys_ket, w_entries in w_schedule:
            row_key_index = len(key_boundary_ids)
            row_entry_start = len(out_boundary_ids)
            for entry_idx in range(entry_start, entry_stop):
                q_boundary_out_id = int(boundary_table.out_sector_ids[entry_idx])
                channel_to_array = (
                    None if channel_maps is None else channel_maps[entry_idx]
                )
                for q_phys_out, W_blocks in w_entries:
                    q_phys_out_id = int(physical_index[q_phys_out])
                    for left_channel, right_channel, W_block in W_blocks:
                        if rep == "rank_coupled_left_factor_by_ket":
                            boundary_channel = int(left_channel)
                            middle_idx = int(right_channel)
                        else:
                            middle_idx = int(left_channel)
                            boundary_channel = int(right_channel)
                        if channel_lookup is not None:
                            if (
                                boundary_channel < 0
                                or boundary_channel > max_boundary_channel
                            ):
                                array_idx = -1
                            else:
                                array_idx = int(
                                    channel_lookup[
                                        int(entry_idx),
                                        int(boundary_channel),
                                    ]
                                )
                        else:
                            array_idx = channel_to_array.get(int(boundary_channel), -1)
                        if array_idx < 0:
                            continue
                        boundary_block = boundary_table.block_pool.array(array_idx)
                        cache_key = (array_idx, id(W_block), str(representation))
                        factor_idx = factor_index_cache.get(cache_key)
                        if factor_idx is None:
                            if (
                                rep == "rank_coupled_left_factor_by_ket"
                                and left_factor_kernel is not None
                            ):
                                factor = left_factor_kernel(boundary_block, W_block)
                                if factor is None:
                                    factor = np.asarray(
                                        factorize_left_two_site_dense_term(
                                            boundary_block,
                                            W_block,
                                        )
                                    )
                                else:
                                    factor = np.asarray(factor)
                            elif right_factor_kernel is not None:
                                factor = right_factor_kernel(W_block, boundary_block)
                                if factor is None:
                                    factor = np.asarray(
                                        factorize_right_two_site_dense_term(
                                            W_block,
                                            boundary_block,
                                        )
                                    )
                                else:
                                    factor = np.asarray(factor)
                            else:
                                factor = _factorize_rank_coupled_boundary_term(
                                    boundary_block,
                                    W_block,
                                    representation=rep,
                                    left_reference=factorize_left_two_site_dense_term,
                                    right_reference=factorize_right_two_site_dense_term,
                                )
                            factor_cache[cache_key] = factor
                            factor_idx = len(arrays)
                            factor_index_cache[cache_key] = factor_idx
                            arrays.append(factor)
                        out_boundary_ids.append(q_boundary_out_id)
                        out_physical_ids.append(q_phys_out_id)
                        middle_ids.append(int(middle_idx))
                        factor_indices.append(int(factor_idx))
                        for name in tuple(family_by_middle.get(int(middle_idx), ())):
                            family_ids.append(int(family_index[str(name)]))
                        family_offsets.append(len(family_ids))
            if len(out_boundary_ids) > row_entry_start:
                key_boundary_ids.append(q_boundary_ket_id)
                key_physical_ids.append(int(physical_index[q_phys_ket]))
                entry_offsets.append(len(out_boundary_ids))
            elif len(key_boundary_ids) != row_key_index:
                raise AssertionError("Unexpected packed SU2 factor row state.")
    if not key_boundary_ids:
        return None
    return PackedSU2FactorTable.from_integer_data(
        side=side,
        bond=bond,
        representation=representation,
        boundary_codec=boundary_table.sector_codec,
        physical_codec=physical_codec,
        family_labels=family_labels,
        key_boundary_ids=key_boundary_ids,
        key_physical_ids=key_physical_ids,
        entry_offsets=entry_offsets,
        out_boundary_ids=out_boundary_ids,
        out_physical_ids=out_physical_ids,
        middle_ids=middle_ids,
        family_offsets=family_offsets,
        family_ids=family_ids,
        factor_indices=factor_indices,
        arrays=arrays,
    )


def pack_rank_coupled_boundary_table_from_payloads(
    numeric_payloads,
    *,
    active_channels=None,
    side,
    bond,
    representation="rank_coupled_by_ket",
    tol=0.0,
):
    """
    Pack a rank-coupled boundary table directly from symbolic numeric payloads.

    ``numeric_payloads`` is keyed by ``(q_out, q_in, channel)``.  This avoids
    first materializing the legacy ``{q_ket: ((q_out, {channel: block}), ...)}``
    dictionary during boundary advancement.
    """

    if str(representation) != "rank_coupled_by_ket":
        return None
    packed_payload = getattr(numeric_payloads, "packed_table", None)
    if isinstance(packed_payload, PackedSU2BoundaryTable):
        active = None if active_channels is None else {int(channel) for channel in active_channels}
        packed_channels = {int(channel) for channel in packed_payload.channel_ids}
        if (
            packed_payload.representation == str(representation)
            and (active is None or packed_channels.issubset(active))
        ):
            return packed_payload
    active = None if active_channels is None else {int(channel) for channel in active_channels}
    by_ket = {}
    arrays = []
    for key, block in sorted(dict(numeric_payloads or {}).items(), key=lambda item: repr(item[0])):
        if len(key) != 3:
            continue
        q_out, q_in, channel = key
        if channel is None:
            continue
        channel = int(channel)
        if active is not None and channel not in active:
            continue
        arr = np.asarray(block)
        if not _has_nonzero(arr, tol):
            continue
        by_ket.setdefault(q_in, {}).setdefault(q_out, []).append((channel, len(arrays)))
        arrays.append(arr)
    rows = []
    for q_ket, out_map in sorted(by_ket.items(), key=lambda item: _sort_key(item[0])):
        entries = []
        for q_out, channels in sorted(out_map.items(), key=lambda item: _sort_key(item[0])):
            entries.append((q_out, tuple(sorted(channels, key=lambda item: int(item[0])))))
        rows.append((q_ket, tuple(entries)))
    if not rows:
        return None
    return PackedSU2BoundaryTable.from_rows(
        rows,
        arrays,
        side=side,
        bond=bond,
        representation=representation,
    )


def pack_rank_coupled_boundary_table_from_block_map(
    block_map,
    *,
    active_channels=None,
    side,
    bond,
    representation="rank_coupled_by_ket",
    tol=0.0,
):
    """
    Pack a rank-coupled boundary table directly from a sector-pair block map.
    """

    if str(representation) != "rank_coupled_by_ket":
        return None
    active = None if active_channels is None else {int(channel) for channel in active_channels}
    rows_by_ket = {}
    arrays = []
    for (q_out, q_in), blocks in sorted(dict(block_map or {}).items(), key=lambda item: repr(item[0])):
        channel_items = []
        for channel, block in enumerate(tuple(blocks or ())):
            if active is not None and int(channel) not in active:
                continue
            arr = np.asarray(block)
            if not _has_nonzero(arr, tol):
                continue
            channel_items.append((int(channel), len(arrays)))
            arrays.append(arr)
        if channel_items:
            rows_by_ket.setdefault(q_in, []).append(
                (q_out, tuple(sorted(channel_items, key=lambda item: int(item[0]))))
            )
    rows = tuple(
        (q_ket, tuple(entries))
        for q_ket, entries in sorted(rows_by_ket.items(), key=lambda item: _sort_key(item[0]))
    )
    if not rows:
        return None
    return PackedSU2BoundaryTable.from_rows(
        rows,
        arrays,
        side=side,
        bond=bond,
        representation=representation,
    )


def filter_rank_coupled_boundary_table_channels(boundary_table, active_channels):
    """Return ``boundary_table`` filtered to active channel ids."""

    if not isinstance(boundary_table, PackedSU2BoundaryTable):
        return None
    if str(boundary_table.representation) != "rank_coupled_by_ket":
        return None
    active = None if active_channels is None else {int(channel) for channel in active_channels}
    if active is None:
        return boundary_table
    packed_channels = {int(channel) for channel in boundary_table.channel_ids}
    if packed_channels.issubset(active):
        return boundary_table
    sectors = tuple(boundary_table.sector_codec.sectors)
    rows = []
    arrays = []
    for row_idx, ket_id in enumerate(boundary_table.ket_sector_ids):
        q_ket = sectors[int(ket_id)]
        entries = []
        entry_start = int(boundary_table.entry_offsets[row_idx])
        entry_stop = int(boundary_table.entry_offsets[row_idx + 1])
        for entry_idx in range(entry_start, entry_stop):
            q_out = sectors[int(boundary_table.out_sector_ids[entry_idx])]
            channels = []
            channel_start = int(boundary_table.channel_offsets[entry_idx])
            channel_stop = int(boundary_table.channel_offsets[entry_idx + 1])
            for channel_idx in range(channel_start, channel_stop):
                channel = int(boundary_table.channel_ids[channel_idx])
                if channel not in active:
                    continue
                channels.append((channel, len(arrays)))
                arrays.append(boundary_table.block_pool.array(channel_idx))
            if channels:
                entries.append((q_out, tuple(channels)))
        if entries:
            rows.append((q_ket, tuple(entries)))
    if not rows:
        return None
    return PackedSU2BoundaryTable.from_rows(
        rows,
        arrays,
        side=boundary_table.side,
        bond=boundary_table.bond,
        representation=boundary_table.representation,
    )


def pack_side_operator_table(grouped_by_ket, *, side, bond, representation):
    """Pack a side operator table when the representation is supported."""

    if str(representation) == "rank_coupled_by_ket":
        return PackedSU2BoundaryTable.from_grouped(
            grouped_by_ket,
            side=side,
            bond=bond,
            representation=representation,
        )
    if str(representation) in {
        "rank_coupled_left_factor_by_ket",
        "rank_coupled_right_factor_by_ket",
    }:
        return PackedSU2FactorTable.from_grouped(
            grouped_by_ket,
            side=side,
            bond=bond,
            representation=representation,
        )
    return None


@dataclass
class PackedSU2QChemCompiledTerms:
    """Lightweight compiled local terms backed by packed SU(2) qchem tables."""

    basis: object
    plan: object
    in_indices: np.ndarray
    out_indices: np.ndarray
    left_indices: np.ndarray
    right_indices: np.ndarray
    match_backend: str = "cython"

    def __post_init__(self):
        self.in_indices = np.asarray(self.in_indices, dtype=np.int64)
        self.out_indices = np.asarray(self.out_indices, dtype=np.int64)
        self.left_indices = np.asarray(self.left_indices, dtype=np.int64)
        self.right_indices = np.asarray(self.right_indices, dtype=np.int64)
        self.items = tuple(() for _entry in self.basis)
        self._family_names_cache = None
        self._family_term_counts_cache = None
        self._diag_match_cache = None
        self._block_matrix_cache = {}
        self.qchem_packed_entry_kernel_provider = True
        self.su2_qchem_sweep_plan_object = self.plan
        self.su2_qchem_factor_match_backend = str(self.match_backend)
        self.su2_qchem_factor_match_count = int(self.in_indices.size)

    @property
    def total_dim(self):
        return int(self.basis.size)

    @property
    def family_names(self):
        if self._family_names_cache is not None:
            return self._family_names_cache
        names = set(self.plan.left_factor_table.family_labels)
        names.update(self.plan.right_factor_table.family_labels)
        self._family_names_cache = tuple(sorted(str(name) for name in names if name is not None))
        return self._family_names_cache

    @property
    def family_term_counts(self):
        if self._family_term_counts_cache is not None:
            return self._family_term_counts_cache
        counts = {}
        if not int(self.left_indices.size):
            self._family_term_counts_cache = counts
            return self._family_term_counts_cache

        labels = tuple(
            sorted(
                {
                    str(label)
                    for table in (
                        self.plan.left_factor_table,
                        self.plan.right_factor_table,
                    )
                    for label in table.family_labels
                    if label is not None
                }
            )
        )
        if not labels:
            self._family_term_counts_cache = {
                "unlabeled": int(self.left_indices.size)
            }
            return self._family_term_counts_cache
        if len(labels) > 63:
            left_names = self._entry_family_name_sets(self.plan.left_factor_table)
            right_names = self._entry_family_name_sets(self.plan.right_factor_table)
            right_n = max(int(self.plan.right_factor_table.n_entries), 1)
            pair_keys = (
                np.asarray(self.left_indices, dtype=np.int64) * np.int64(right_n)
                + np.asarray(self.right_indices, dtype=np.int64)
            )
            unique_pairs, multiplicities = np.unique(pair_keys, return_counts=True)
            for pair_key, multiplicity in zip(unique_pairs, multiplicities):
                lidx, ridx = divmod(int(pair_key), right_n)
                names = left_names[int(lidx)] | right_names[int(ridx)]
                if not names:
                    names = {"unlabeled"}
                for name in names:
                    counts[str(name)] = counts.get(str(name), 0) + int(multiplicity)
            self._family_term_counts_cache = dict(sorted(counts.items()))
            return self._family_term_counts_cache

        label_index = {label: idx for idx, label in enumerate(labels)}
        left_masks = self._entry_family_masks(self.plan.left_factor_table, label_index)
        right_masks = self._entry_family_masks(self.plan.right_factor_table, label_index)
        match_masks = (
            left_masks[np.asarray(self.left_indices, dtype=np.int64)]
            | right_masks[np.asarray(self.right_indices, dtype=np.int64)]
        )
        for label, idx in label_index.items():
            counts[str(label)] = int(np.count_nonzero(match_masks & np.uint64(1 << idx)))
        unlabeled = int(np.count_nonzero(match_masks == np.uint64(0)))
        if unlabeled:
            counts["unlabeled"] = unlabeled
        self._family_term_counts_cache = dict(sorted(counts.items()))
        return self._family_term_counts_cache

    @staticmethod
    def _entry_family_masks(table, label_index):
        masks = np.zeros(int(table.n_entries), dtype=np.uint64)
        table_labels = tuple(str(label) for label in table.family_labels)
        for entry_idx in range(int(table.n_entries)):
            mask = 0
            start = int(table.family_offsets[entry_idx])
            stop = int(table.family_offsets[entry_idx + 1])
            for family_id in table.family_ids[start:stop]:
                label = table_labels[int(family_id)]
                bit = label_index.get(str(label))
                if bit is not None:
                    mask |= 1 << int(bit)
            masks[entry_idx] = np.uint64(mask)
        return masks

    @staticmethod
    def _entry_family_name_sets(table):
        labels = tuple(str(label) for label in table.family_labels)
        out = []
        for entry_idx in range(int(table.n_entries)):
            start = int(table.family_offsets[entry_idx])
            stop = int(table.family_offsets[entry_idx + 1])
            out.append({labels[int(idx)] for idx in table.family_ids[start:stop]})
        return tuple(out)

    def _diagonal_match_indices(self):
        cached = self._diag_match_cache
        if cached is not None:
            return cached
        matches = [[] for _entry in self.basis]
        for match_idx, (in_idx, out_idx) in enumerate(zip(self.in_indices, self.out_indices)):
            if int(in_idx) == int(out_idx):
                matches[int(in_idx)].append(int(match_idx))
        self._diag_match_cache = tuple(tuple(row) for row in matches)
        return self._diag_match_cache

    def _single_kernel(self, lidx, ridx, input_entry, output_entry):
        left = self.plan.left_factor_table.factor(int(lidx))
        right = self.plan.right_factor_table.factor(int(ridx))
        kernel = np.einsum(
            "lkwab,wqrdc->ladqkbcr",
            np.asarray(left),
            np.asarray(right),
            optimize=False,
        )
        return np.ascontiguousarray(
            kernel.reshape(int(output_entry.size), int(input_entry.size))
        )

    @property
    def block_matrices(self):
        cached = getattr(self, "_block_matrices_cache", None)
        if cached is not None:
            return cached
        blocks = tuple(self.block_matrix_for(entry) for entry in self.basis)
        self._block_matrices_cache = blocks
        return blocks

    def block_matrix_for(self, entry_or_key):
        entry = (
            entry_or_key
            if hasattr(entry_or_key, "key")
            else self.basis.entry_for_key(entry_or_key)
        )
        in_idx = int(self.basis.entry_index(entry.key))
        if in_idx in self._block_matrix_cache:
            return self._block_matrix_cache[in_idx]
        block = None
        grouped = {}
        for match_idx in self._diagonal_match_indices()[in_idx]:
            lidx = int(self.left_indices[int(match_idx)])
            ridx = int(self.right_indices[int(match_idx)])
            left = self.plan.left_factor_table.factor(lidx)
            right = self.plan.right_factor_table.factor(ridx)
            key = (
                tuple(int(dim) for dim in left.shape),
                tuple(int(dim) for dim in right.shape),
            )
            bucket = grouped.setdefault(key, {"left": [], "right": []})
            bucket["left"].append(left)
            bucket["right"].append(right)
        for key in sorted(grouped):
            bucket = grouped[key]
            kernel = self.plan._factorized_kernel(
                np.ascontiguousarray(np.stack(bucket["left"], axis=0)),
                np.ascontiguousarray(np.stack(bucket["right"], axis=0)),
                entry,
                entry,
            )
            block = kernel if block is None else block + kernel
        self._block_matrix_cache[in_idx] = block
        return block

    def entry_kernel_items(self, *, max_block_kernel_elements=None):
        """
        Materialize entry-level dense kernels for solver structural analysis.

        :returns: Tuple of ``(input_entry_index, output_entry_index, kernel)``.
        """

        cache_key = (
            None
            if max_block_kernel_elements is None
            else int(max_block_kernel_elements)
        )
        cached = getattr(self, "_entry_kernel_items_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        grouped = {}
        left_factor_cache = [None] * int(self.plan.left_factor_table.n_entries)
        right_factor_cache = [None] * int(self.plan.right_factor_table.n_entries)
        left_shape_cache = [None] * int(self.plan.left_factor_table.n_entries)
        right_shape_cache = [None] * int(self.plan.right_factor_table.n_entries)
        for in_idx, out_idx, lidx, ridx in zip(
            self.in_indices,
            self.out_indices,
            self.left_indices,
            self.right_indices,
        ):
            in_entry = self.basis[int(in_idx)]
            out_entry = self.basis[int(out_idx)]
            lidx = int(lidx)
            ridx = int(ridx)
            left = left_factor_cache[lidx]
            if left is None:
                left = self.plan.left_factor_table.factor(lidx)
                left_factor_cache[lidx] = left
            right = right_factor_cache[ridx]
            if right is None:
                right = self.plan.right_factor_table.factor(ridx)
                right_factor_cache[ridx] = right
            left_shape = left_shape_cache[lidx]
            if left_shape is None:
                left_shape = tuple(int(dim) for dim in left.shape)
                left_shape_cache[lidx] = left_shape
            right_shape = right_shape_cache[ridx]
            if right_shape is None:
                right_shape = tuple(int(dim) for dim in right.shape)
                right_shape_cache[ridx] = right_shape
            key = (
                int(in_idx),
                int(out_idx),
                left_shape,
                right_shape,
            )
            bucket = grouped.setdefault(key, {"left": [], "right": []})
            bucket["left"].append(left)
            bucket["right"].append(right)
        items = []
        for key in sorted(grouped, key=lambda item: (item[0], self.basis[item[1]].offset)):
            in_idx, out_idx, _left_shape, _right_shape = key
            in_entry = self.basis[int(in_idx)]
            out_entry = self.basis[int(out_idx)]
            elements = int(in_entry.size) * int(out_entry.size)
            if (
                max_block_kernel_elements is not None
                and elements > int(max_block_kernel_elements)
            ):
                return None
            bucket = grouped[key]
            kernel = self.plan._factorized_kernel(
                np.ascontiguousarray(np.stack(bucket["left"], axis=0)),
                np.ascontiguousarray(np.stack(bucket["right"], axis=0)),
                in_entry,
                out_entry,
            )
            items.append((int(in_idx), int(out_idx), kernel))
        out = tuple(items)
        self._entry_kernel_items_cache = (cache_key, out)
        return out

    def apply_packed(self, vector, *, base_dtype):
        vec = np.asarray(vector)
        out = np.zeros(int(self.total_dim), dtype=np.result_type(base_dtype, vec.dtype))
        items = self.entry_kernel_items(max_block_kernel_elements=None)
        if items is None:
            return out
        for in_idx, out_idx, kernel in items:
            in_entry = self.basis[int(in_idx)]
            out_entry = self.basis[int(out_idx)]
            out[out_entry.slice] += np.asarray(kernel) @ vec[in_entry.slice]
        return out

    def packed_matvec(self, *, base_dtype, backend, out_entries=None, block_matrices=None):
        def packed_apply(vector):
            return self.apply_packed(vector, base_dtype=base_dtype)

        packed_apply.backend = backend
        packed_apply.basis = self.basis
        packed_apply.compiled_factorized_terms = self
        packed_apply.out_entries = out_entries
        packed_apply.block_matrices = block_matrices
        packed_apply.dense_matrix = None
        packed_apply.family_names = self.family_names
        packed_apply.family_term_counts = self.family_term_counts
        return packed_apply

    def build_component_parent_blocks(self, components, component_dims):
        return self.plan.build_component_parent_blocks(
            self.basis,
            components,
            component_dims,
            use_matches=True,
        )


@dataclass(frozen=True)
class SU2QChemSweepPlan:
    """Packed local factor schedule for one SU(2) qchem bond."""

    bond: int
    left_factor_table: PackedSU2FactorTable
    right_factor_table: PackedSU2FactorTable
    left_boundary_table: PackedSU2BoundaryTable | None = None
    right_boundary_table: PackedSU2BoundaryTable | None = None
    _factor_matches_cache: dict = field(default_factory=dict, compare=False, repr=False)
    _factor_matches_cache_stats: dict = field(
        default_factory=lambda: {
            "local_hits": 0,
            "layout_hits": 0,
            "misses": 0,
            "puts": 0,
        },
        compare=False,
        repr=False,
    )
    _compiled_terms_cache: dict = field(default_factory=dict, compare=False, repr=False)
    _compiled_terms_cache_stats: dict = field(
        default_factory=lambda: {
            "hits": 0,
            "misses": 0,
            "puts": 0,
        },
        compare=False,
        repr=False,
    )

    @property
    def supported(self):
        return (
            isinstance(self.left_factor_table, PackedSU2FactorTable)
            and isinstance(self.right_factor_table, PackedSU2FactorTable)
        )

    @property
    def stats(self):
        return {
            "kind": "su2_qchem_sweep_plan",
            "bond": int(self.bond),
            "supported": bool(self.supported),
            "left_boundary_table": (
                None
                if self.left_boundary_table is None
                else self.left_boundary_table.stats
            ),
            "right_boundary_table": (
                None
                if self.right_boundary_table is None
                else self.right_boundary_table.stats
            ),
            "left_factor_table": self.left_factor_table.stats,
            "right_factor_table": self.right_factor_table.stats,
            "factor_match_cache_size": int(len(self._factor_matches_cache)),
            "factor_match_layout_cache_size": int(len(_FACTOR_MATCH_LAYOUT_CACHE)),
            "component_parent_block_layout_cache_size": int(
                len(_COMPONENT_PARENT_BLOCK_LAYOUT_CACHE)
            ),
            "component_parent_block_layout_cache_stats": {
                str(key): int(value)
                for key, value in _COMPONENT_PARENT_BLOCK_LAYOUT_CACHE_STATS.items()
            },
            "factor_match_cache_stats": {
                str(key): int(value)
                for key, value in self._factor_matches_cache_stats.items()
            },
            "compiled_terms_cache_size": int(len(self._compiled_terms_cache)),
            "compiled_terms_cache_stats": {
                str(key): int(value)
                for key, value in self._compiled_terms_cache_stats.items()
            },
        }

    @staticmethod
    def _basis_signature(basis):
        """
        Return a stable structural signature for a packed two-site basis.

        The packed qchem plan is independent of the Davidson basis, but the
        integer input/output match schedule depends on the basis entry layout.
        This signature lets one plan reuse that schedule while avoiding stale
        matches after sector support changes.
        """

        cache_key = id(basis)
        cached = _BASIS_SIGNATURE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        entries = tuple(getattr(basis, "entries", tuple(basis)))
        signature = (
            "packed_two_site_basis",
            int(getattr(basis, "size", sum(int(entry.size) for entry in entries))),
            tuple(
                (
                    getattr(entry, "key", None),
                    tuple(int(dim) for dim in getattr(entry, "shape", ())),
                    int(getattr(entry, "offset", 0)),
                    int(getattr(entry, "size", 0)),
                )
                for entry in entries
            ),
        )
        if len(_BASIS_SIGNATURE_CACHE) > 512:
            _BASIS_SIGNATURE_CACHE.clear()
        _BASIS_SIGNATURE_CACHE[cache_key] = signature
        return signature

    @staticmethod
    def _factor_table_layout_signature(table):
        """Return the value-independent layout signature for a factor table."""

        cached = getattr(table, "_layout_signature_cache", None)
        if cached is not None:
            return cached
        signature = (
            str(table.side),
            int(table.bond),
            str(table.representation),
            _codec_signature(table.boundary_codec),
            _codec_signature(table.physical_codec),
            tuple(str(label) for label in table.family_labels),
            _array_tuple(table.key_boundary_ids),
            _array_tuple(table.key_physical_ids),
            _array_tuple(table.entry_offsets),
            _array_tuple(table.out_boundary_ids),
            _array_tuple(table.out_physical_ids),
            _array_tuple(table.middle_ids),
            _array_tuple(table.family_offsets),
            _array_tuple(table.family_ids),
        )
        try:
            object.__setattr__(table, "_layout_signature_cache", signature)
        except Exception:
            pass
        return signature

    @staticmethod
    def _factor_table_shape_signature(table):
        """Return the value-independent factor-shape signature for a table."""

        cached = getattr(table, "_shape_signature_cache", None)
        if cached is not None:
            return cached
        pool = table.factor_pool
        signature = (
            _array_tuple(table.factor_indices),
            _array_tuple(pool.shape_offsets),
            _array_tuple(pool.shapes),
        )
        try:
            object.__setattr__(table, "_shape_signature_cache", signature)
        except Exception:
            pass
        return signature

    def _factor_match_layout_cache_key(self, basis):
        return (
            "su2_qchem_factor_matches",
            self._factor_table_layout_signature(self.left_factor_table),
            self._factor_table_layout_signature(self.right_factor_table),
            self._basis_signature(basis),
        )

    def _component_parent_block_layout_cache_key(self, basis, components):
        return (
            "su2_qchem_component_parent_block_layout",
            self._factor_table_layout_signature(self.left_factor_table),
            self._factor_table_layout_signature(self.right_factor_table),
            self._factor_table_shape_signature(self.left_factor_table),
            self._factor_table_shape_signature(self.right_factor_table),
            self._basis_signature(basis),
            tuple(tuple(int(idx) for idx in component) for component in components),
        )

    def _integer_out_index(self, basis):
        left_boundary_index = self.left_factor_table.boundary_codec.index
        left_physical_index = self.left_factor_table.physical_codec.index
        right_boundary_index = self.right_factor_table.boundary_codec.index
        right_physical_index = self.right_factor_table.physical_codec.index
        out_index = {}
        for out_idx, out_entry in enumerate(basis):
            q_lb, q_p1b, q_p2b, q_rb = out_entry.key
            try:
                out_index[
                    (
                        int(left_boundary_index[q_lb]),
                        int(left_physical_index[q_p1b]),
                        int(right_physical_index[q_p2b]),
                        int(right_boundary_index[q_rb]),
                    )
                ] = int(out_idx)
            except KeyError:
                continue
        return out_index

    def _integer_basis_arrays(self, basis):
        left_boundary_index = self.left_factor_table.boundary_codec.index
        left_physical_index = self.left_factor_table.physical_codec.index
        right_boundary_index = self.right_factor_table.boundary_codec.index
        right_physical_index = self.right_factor_table.physical_codec.index
        left_ids = []
        p1_ids = []
        p2_ids = []
        right_ids = []
        for entry in basis:
            q_l, q_p1, q_p2, q_r = entry.key
            left_ids.append(int(left_boundary_index.get(q_l, -1)))
            p1_ids.append(int(left_physical_index.get(q_p1, -1)))
            p2_ids.append(int(right_physical_index.get(q_p2, -1)))
            right_ids.append(int(right_boundary_index.get(q_r, -1)))
        return (
            np.asarray(left_ids, dtype=np.int64),
            np.asarray(p1_ids, dtype=np.int64),
            np.asarray(p2_ids, dtype=np.int64),
            np.asarray(right_ids, dtype=np.int64),
        )

    def _dense_lookup_tables(self, basis):
        left_key_map = np.full(
            (
                len(self.left_factor_table.boundary_codec.sectors),
                len(self.left_factor_table.physical_codec.sectors),
            ),
            -1,
            dtype=np.int64,
        )
        for idx, (boundary_id, phys_id) in enumerate(
            zip(
                self.left_factor_table.key_boundary_ids,
                self.left_factor_table.key_physical_ids,
            )
        ):
            left_key_map[int(boundary_id), int(phys_id)] = int(idx)
        right_key_map = np.full(
            (
                len(self.right_factor_table.boundary_codec.sectors),
                len(self.right_factor_table.physical_codec.sectors),
            ),
            -1,
            dtype=np.int64,
        )
        for idx, (boundary_id, phys_id) in enumerate(
            zip(
                self.right_factor_table.key_boundary_ids,
                self.right_factor_table.key_physical_ids,
            )
        ):
            right_key_map[int(boundary_id), int(phys_id)] = int(idx)
        out_map = np.full(
            (
                len(self.left_factor_table.boundary_codec.sectors),
                len(self.left_factor_table.physical_codec.sectors),
                len(self.right_factor_table.physical_codec.sectors),
                len(self.right_factor_table.boundary_codec.sectors),
            ),
            -1,
            dtype=np.int64,
        )
        for key, out_idx in self._integer_out_index(basis).items():
            out_map[key] = int(out_idx)
        return left_key_map, right_key_map, out_map

    def factor_matches(self, basis):
        """
        Return packed ``(input, output, left_entry, right_entry)`` match arrays.
        """

        if not self.supported:
            return None
        cache_key = self._basis_signature(basis)
        cached = self._factor_matches_cache.get(cache_key)
        if cached is not None:
            self._factor_matches_cache_stats["local_hits"] += 1
            return cached
        layout_cache_key = self._factor_match_layout_cache_key(basis)
        cached = _FACTOR_MATCH_LAYOUT_CACHE.get(layout_cache_key)
        if cached is not None:
            self._factor_matches_cache[cache_key] = cached
            self._factor_matches_cache_stats["layout_hits"] += 1
            return cached
        try:
            from pyqed.mps.nonabelian import _su2_kernel as kernel

            build_matches = getattr(kernel, "build_su2_qchem_factor_matches")
        except Exception:
            build_matches = None
        if build_matches is None:
            return None
        left_ids, p1_ids, p2_ids, right_ids = self._integer_basis_arrays(basis)
        left_key_map, right_key_map, out_map = self._dense_lookup_tables(basis)
        matches = build_matches(
            left_ids,
            p1_ids,
            p2_ids,
            right_ids,
            left_key_map,
            right_key_map,
            out_map,
            self.left_factor_table.entry_offsets,
            self.left_factor_table.out_boundary_ids,
            self.left_factor_table.out_physical_ids,
            self.left_factor_table.middle_ids,
            self.right_factor_table.entry_offsets,
            self.right_factor_table.out_boundary_ids,
            self.right_factor_table.out_physical_ids,
            self.right_factor_table.middle_ids,
        )
        if matches is not None:
            self._factor_matches_cache[cache_key] = matches
            if len(_FACTOR_MATCH_LAYOUT_CACHE) > 256:
                _FACTOR_MATCH_LAYOUT_CACHE.clear()
            _FACTOR_MATCH_LAYOUT_CACHE[layout_cache_key] = matches
            self._factor_matches_cache_stats["puts"] += 1
        else:
            self._factor_matches_cache_stats["misses"] += 1
        return matches

    @staticmethod
    def _entry_component_map(basis, components):
        entry_to_component = {}
        for comp_idx, component in enumerate(tuple(components or ())):
            cursor = 0
            for entry_idx in component:
                entry = basis.entries[int(entry_idx)]
                entry_to_component[int(entry_idx)] = (
                    int(comp_idx),
                    slice(cursor, cursor + int(entry.size)),
                )
                cursor += int(entry.size)
        return entry_to_component

    @staticmethod
    def _factorized_kernel(left_stack, right_stack, input_entry, output_entry):
        left = np.asarray(left_stack)
        right = np.asarray(right_stack)
        if left.ndim != 6 or right.ndim != 6:
            kernel = np.einsum(
                "tlkwab,twqrdc->ladqkbcr",
                left,
                right,
                optimize=False,
            )
            return np.ascontiguousarray(
                kernel.reshape(int(output_entry.size), int(input_entry.size))
            )
        tdim, ldim, kdim, wdim, adim, bdim = (
            int(dim) for dim in left.shape
        )
        r_tdim, r_wdim, qdim, rdim, ddim, cdim = (
            int(dim) for dim in right.shape
        )
        if tdim != r_tdim or wdim != r_wdim:
            kernel = np.einsum(
                "tlkwab,twqrdc->ladqkbcr",
                left,
                right,
                optimize=False,
            )
            return np.ascontiguousarray(
                kernel.reshape(int(output_entry.size), int(input_entry.size))
            )
        left_mat = np.ascontiguousarray(
            left.transpose(1, 4, 2, 5, 0, 3).reshape(
                ldim * adim * kdim * bdim,
                tdim * wdim,
            )
        )
        right_mat = np.ascontiguousarray(
            right.transpose(0, 1, 4, 2, 5, 3).reshape(
                tdim * wdim,
                ddim * qdim * cdim * rdim,
            )
        )
        kernel = (left_mat @ right_mat).reshape(
            ldim,
            adim,
            kdim,
            bdim,
            ddim,
            qdim,
            cdim,
            rdim,
        )
        kernel = kernel.transpose(0, 1, 4, 5, 2, 3, 6, 7)
        return np.ascontiguousarray(
            kernel.reshape(int(output_entry.size), int(input_entry.size))
        )

    def _compile_factorized_terms_from_matches(self, basis, matches):
        from .local_operator import CompiledFactorizedBlock, CompiledFactorizedTerms

        in_indices, out_indices, left_indices, right_indices = matches
        grouped_by_input = [{} for _entry in basis]
        combined_family_labels = tuple(self.left_factor_table.family_labels) + tuple(
            self.right_factor_table.family_labels
        )
        left_factor_cache = [None] * int(self.left_factor_table.n_entries)
        right_factor_cache = [None] * int(self.right_factor_table.n_entries)
        left_shape_cache = [None] * int(self.left_factor_table.n_entries)
        right_shape_cache = [None] * int(self.right_factor_table.n_entries)
        left_family_cache = [None] * int(self.left_factor_table.n_entries)
        right_family_cache = [None] * int(self.right_factor_table.n_entries)
        for in_idx, out_idx, lidx, ridx in zip(
            in_indices,
            out_indices,
            left_indices,
            right_indices,
        ):
            in_idx = int(in_idx)
            out_idx = int(out_idx)
            lidx = int(lidx)
            ridx = int(ridx)
            left_factor = left_factor_cache[lidx]
            if left_factor is None:
                left_factor = self.left_factor_table.factor(lidx)
                left_factor_cache[lidx] = left_factor
            right_factor = right_factor_cache[ridx]
            if right_factor is None:
                right_factor = self.right_factor_table.factor(ridx)
                right_factor_cache[ridx] = right_factor
            left_shape = left_shape_cache[lidx]
            if left_shape is None:
                left_shape = tuple(int(dim) for dim in left_factor.shape)
                left_shape_cache[lidx] = left_shape
            right_shape = right_shape_cache[ridx]
            if right_shape is None:
                right_shape = tuple(int(dim) for dim in right_factor.shape)
                right_shape_cache[ridx] = right_shape
            key = (
                out_idx,
                left_shape,
                right_shape,
            )
            bucket = grouped_by_input[in_idx].setdefault(
                key,
                {"left": [], "right": [], "family_ids": set()},
            )
            bucket["left"].append(left_factor)
            bucket["right"].append(right_factor)
            left_family_ids = left_family_cache[lidx]
            if left_family_ids is None:
                left_family_ids = tuple(
                    int(idx)
                    for idx in self.left_factor_table.family_ids[
                        int(self.left_factor_table.family_offsets[lidx]):int(
                            self.left_factor_table.family_offsets[lidx + 1]
                        )
                    ]
                )
                left_family_cache[lidx] = left_family_ids
            right_family_ids = right_family_cache[ridx]
            if right_family_ids is None:
                shift = len(self.left_factor_table.family_labels)
                right_family_ids = tuple(
                    int(idx) + shift
                    for idx in self.right_factor_table.family_ids[
                        int(self.right_factor_table.family_offsets[ridx]):int(
                            self.right_factor_table.family_offsets[ridx + 1]
                        )
                    ]
                )
                right_family_cache[ridx] = right_family_ids
            bucket["family_ids"].update(left_family_ids)
            bucket["family_ids"].update(right_family_ids)
        compiled_items = []
        for in_idx, grouped in enumerate(grouped_by_input):
            in_entry = basis[int(in_idx)]
            compiled_terms = []
            for shape_key in sorted(grouped, key=lambda key: basis[key[0]].offset):
                out_idx = int(shape_key[0])
                bucket = grouped[shape_key]
                family_names = tuple(
                    sorted(
                        {
                            str(combined_family_labels[int(idx)])
                            for idx in bucket.get("family_ids", ())
                        }
                    )
                )
                compiled_terms.append(
                    CompiledFactorizedBlock(
                        input_entry=in_entry,
                        output_entry=basis[out_idx],
                        left_stack=np.ascontiguousarray(
                            np.stack(bucket["left"], axis=0)
                        ),
                        right_stack=np.ascontiguousarray(
                            np.stack(bucket["right"], axis=0)
                        ),
                        family_names=family_names,
                    )
                )
            compiled_items.append(tuple(compiled_terms))
        compiled = CompiledFactorizedTerms(basis=basis, items=tuple(compiled_items))
        compiled.su2_qchem_factor_match_backend = "cython"
        compiled.su2_qchem_factor_match_count = int(len(in_indices))
        compiled.su2_qchem_sweep_plan_object = self

        def build_component_parent_blocks(components, component_dims):
            return self.build_component_parent_blocks_from_matches(
                basis,
                components,
                component_dims,
                in_indices,
                out_indices,
                left_indices,
                right_indices,
            )

        compiled.build_component_parent_blocks = build_component_parent_blocks
        return compiled

    def build_factorized_terms(self, basis):
        """
        Build reference-compatible factorized terms from packed integer tables.

        This method is deliberately still Python-level, but all schedule
        lookups use integer/offset arrays.  The Cython implementation consumes
        the same fields without changing the sweep API.
        """

        if not self.supported:
            return None
        out_index = basis.index_by_key()
        terms = {}
        right_middle_cache = {}
        for in_entry in basis:
            q_lk, q_p1k, q_p2k, q_rk = in_entry.key
            left_range = self.left_factor_table.entry_range_for_key(q_lk, q_p1k)
            right_range = self.right_factor_table.entry_range_for_key(q_rk, q_p2k)
            if left_range is None or right_range is None:
                terms[in_entry.key] = ()
                continue
            right_key = right_range
            right_by_middle = right_middle_cache.get(right_key)
            if right_by_middle is None:
                right_by_middle = {}
                for ridx in range(right_range[0], right_range[1]):
                    middle = int(self.right_factor_table.middle_ids[ridx])
                    right_by_middle.setdefault(middle, []).append(ridx)
                right_middle_cache[right_key] = right_by_middle
            in_terms = []
            for lidx in range(left_range[0], left_range[1]):
                middle = int(self.left_factor_table.middle_ids[lidx])
                q_lb = self.left_factor_table.boundary_codec.sectors[
                    int(self.left_factor_table.out_boundary_ids[lidx])
                ]
                q_p1b = self.left_factor_table.physical_codec.sectors[
                    int(self.left_factor_table.out_physical_ids[lidx])
                ]
                left_families = self.left_factor_table.families(lidx)
                for ridx in right_by_middle.get(middle, ()):
                    q_rb = self.right_factor_table.boundary_codec.sectors[
                        int(self.right_factor_table.out_boundary_ids[ridx])
                    ]
                    q_p2b = self.right_factor_table.physical_codec.sectors[
                        int(self.right_factor_table.out_physical_ids[ridx])
                    ]
                    out_idx = out_index.get((q_lb, q_p1b, q_p2b, q_rb))
                    if out_idx is None:
                        continue
                    families = tuple(
                        sorted(
                            {
                                str(name)
                                for name in left_families + self.right_factor_table.families(ridx)
                                if name is not None
                            }
                        )
                    )
                    in_terms.append(
                        (
                            out_idx,
                            self.left_factor_table.factor(lidx),
                            self.right_factor_table.factor(ridx),
                            families,
                        )
                    )
            terms[in_entry.key] = tuple(in_terms)
        return basis.out_entries, terms

    def compile_factorized_terms(self, basis, *, prefer_packed=False):
        """
        Compile packed factor schedules directly into block kernels.

        This avoids materializing the legacy ``factorized_terms`` dictionary.
        The returned object is still the existing Python
        ``CompiledFactorizedTerms`` container, so the solver and
        orthonormalized table builders remain unchanged.
        """

        if not self.supported:
            return None
        cache_key = (
            self._basis_signature(basis),
            bool(prefer_packed),
            bool(_USE_PACKED_COMPILED_TERMS),
            bool(_DEBUG_PACKED_COMPILED_TERMS),
        )
        cached = self._compiled_terms_cache.get(cache_key)
        if cached is not None:
            self._compiled_terms_cache_stats["hits"] += 1
            return cached
        self._compiled_terms_cache_stats["misses"] += 1
        matches = self.factor_matches(basis)
        if matches is not None and (_USE_PACKED_COMPILED_TERMS or prefer_packed):
            in_indices, out_indices, left_indices, right_indices = matches
            packed = PackedSU2QChemCompiledTerms(
                basis=basis,
                plan=self,
                in_indices=in_indices,
                out_indices=out_indices,
                left_indices=left_indices,
                right_indices=right_indices,
                match_backend="cython",
            )
            if _DEBUG_PACKED_COMPILED_TERMS:
                legacy = self._compile_factorized_terms_from_matches(basis, matches)
                rng = np.random.default_rng(123)
                probe = rng.normal(size=basis.size) + 1j * rng.normal(size=basis.size)
                ref = legacy.apply_packed(probe, base_dtype=complex)
                got = packed.apply_packed(probe, base_dtype=complex)
                scale = max(float(np.linalg.norm(ref)), 1.0)
                residual = float(np.linalg.norm(ref - got) / scale)
                if residual > 1.0e-10:
                    raise RuntimeError(
                        "Packed SU2 qchem compiled terms disagree with legacy "
                        f"compiled terms: residual={residual:.3e}"
                    )
            self._compiled_terms_cache[cache_key] = packed
            self._compiled_terms_cache_stats["puts"] += 1
            return packed
        if matches is not None:
            compiled = self._compile_factorized_terms_from_matches(basis, matches)
            self._compiled_terms_cache[cache_key] = compiled
            self._compiled_terms_cache_stats["puts"] += 1
            return compiled
        from .local_operator import CompiledFactorizedBlock, CompiledFactorizedTerms

        out_index = self._integer_out_index(basis)
        right_middle_cache = {}
        compiled_items = []
        for in_entry in basis:
            q_lk, q_p1k, q_p2k, q_rk = in_entry.key
            left_key = self.left_factor_table.key_id(q_lk, q_p1k)
            right_key = self.right_factor_table.key_id(q_rk, q_p2k)
            if left_key is None or right_key is None:
                compiled_items.append(())
                continue
            left_range = self.left_factor_table.entry_range_for_ids(*left_key)
            right_range = self.right_factor_table.entry_range_for_ids(*right_key)
            if left_range is None or right_range is None:
                compiled_items.append(())
                continue
            right_by_middle = right_middle_cache.get(right_range)
            if right_by_middle is None:
                right_by_middle = {}
                for ridx in range(right_range[0], right_range[1]):
                    middle = int(self.right_factor_table.middle_ids[ridx])
                    right_by_middle.setdefault(middle, []).append(ridx)
                right_middle_cache[right_range] = right_by_middle
            grouped = {}
            for lidx in range(left_range[0], left_range[1]):
                middle = int(self.left_factor_table.middle_ids[lidx])
                q_lb_id = int(self.left_factor_table.out_boundary_ids[lidx])
                q_p1b_id = int(self.left_factor_table.out_physical_ids[lidx])
                left_factor = self.left_factor_table.factor(lidx)
                left_family_start = int(self.left_factor_table.family_offsets[lidx])
                left_family_stop = int(self.left_factor_table.family_offsets[lidx + 1])
                for ridx in right_by_middle.get(middle, ()):
                    out_idx = out_index.get(
                        (
                            q_lb_id,
                            q_p1b_id,
                            int(self.right_factor_table.out_physical_ids[ridx]),
                            int(self.right_factor_table.out_boundary_ids[ridx]),
                        )
                    )
                    if out_idx is None:
                        continue
                    right_factor = self.right_factor_table.factor(ridx)
                    key = (
                        int(out_idx),
                        tuple(int(dim) for dim in left_factor.shape),
                        tuple(int(dim) for dim in right_factor.shape),
                    )
                    bucket = grouped.setdefault(
                        key,
                        {"left": [], "right": [], "families": []},
                    )
                    bucket["left"].append(left_factor)
                    bucket["right"].append(right_factor)
                    bucket.setdefault("family_ids", set()).update(
                        int(idx)
                        for idx in self.left_factor_table.family_ids[
                            left_family_start:left_family_stop
                        ]
                    )
                    bucket["family_ids"].update(
                        int(idx) + len(self.left_factor_table.family_labels)
                        for idx in self.right_factor_table.family_ids[
                            int(self.right_factor_table.family_offsets[ridx]):int(
                                self.right_factor_table.family_offsets[ridx + 1]
                            )
                        ]
                    )
            compiled_terms = []
            combined_family_labels = tuple(self.left_factor_table.family_labels) + tuple(
                self.right_factor_table.family_labels
            )
            for shape_key in sorted(grouped, key=lambda key: basis[key[0]].offset):
                out_idx = int(shape_key[0])
                bucket = grouped[shape_key]
                family_names = tuple(
                    sorted(
                        {
                            str(combined_family_labels[int(idx)])
                            for idx in bucket.get("family_ids", ())
                        }
                    )
                )
                compiled_terms.append(
                    CompiledFactorizedBlock(
                        input_entry=in_entry,
                        output_entry=basis[out_idx],
                        left_stack=np.ascontiguousarray(
                            np.stack(bucket["left"], axis=0)
                        ),
                        right_stack=np.ascontiguousarray(
                            np.stack(bucket["right"], axis=0)
                        ),
                        family_names=family_names,
                    )
                )
            compiled_items.append(tuple(compiled_terms))
        compiled = CompiledFactorizedTerms(basis=basis, items=tuple(compiled_items))
        self._compiled_terms_cache[cache_key] = compiled
        self._compiled_terms_cache_stats["puts"] += 1
        return compiled

    def build_component_parent_blocks(
        self,
        basis,
        components,
        component_dims,
        *,
        use_matches=True,
    ):
        """
        Build component-parent dense blocks directly from packed factor tables.

        This is the block2-like qchem path below the Python reference
        ``CompiledFactorizedBlock`` layer: output matching uses integer sector
        ids and same-shape terms are stacked before the dense kernel is formed.
        """

        if not self.supported:
            return None
        if components is None:
            return None
        matches = self.factor_matches(basis) if use_matches else None
        if matches is not None:
            return self.build_component_parent_blocks_from_matches(
                basis,
                components,
                component_dims,
                *matches,
            )
        entry_to_component = self._entry_component_map(basis, components)
        if len(entry_to_component) != len(tuple(basis.entries)):
            return None
        out_index = self._integer_out_index(basis)
        right_middle_cache = {}
        blocks = {}
        for in_idx, in_entry in enumerate(basis):
            in_info = entry_to_component.get(int(in_idx))
            if in_info is None:
                return None
            in_comp, in_slice = in_info
            q_lk, q_p1k, q_p2k, q_rk = in_entry.key
            left_key = self.left_factor_table.key_id(q_lk, q_p1k)
            right_key = self.right_factor_table.key_id(q_rk, q_p2k)
            if left_key is None or right_key is None:
                continue
            left_range = self.left_factor_table.entry_range_for_ids(*left_key)
            right_range = self.right_factor_table.entry_range_for_ids(*right_key)
            if left_range is None or right_range is None:
                continue
            right_by_middle = right_middle_cache.get(right_range)
            if right_by_middle is None:
                right_by_middle = {}
                for ridx in range(right_range[0], right_range[1]):
                    middle = int(self.right_factor_table.middle_ids[ridx])
                    right_by_middle.setdefault(middle, []).append(ridx)
                right_middle_cache[right_range] = right_by_middle
            grouped = {}
            for lidx in range(left_range[0], left_range[1]):
                middle = int(self.left_factor_table.middle_ids[lidx])
                q_lb_id = int(self.left_factor_table.out_boundary_ids[lidx])
                q_p1b_id = int(self.left_factor_table.out_physical_ids[lidx])
                left_factor = self.left_factor_table.factor(lidx)
                left_shape = tuple(int(dim) for dim in left_factor.shape)
                for ridx in right_by_middle.get(middle, ()):
                    out_idx = out_index.get(
                        (
                            q_lb_id,
                            q_p1b_id,
                            int(self.right_factor_table.out_physical_ids[ridx]),
                            int(self.right_factor_table.out_boundary_ids[ridx]),
                        )
                    )
                    if out_idx is None:
                        continue
                    out_info = entry_to_component.get(int(out_idx))
                    if out_info is None:
                        return None
                    out_comp, out_slice = out_info
                    right_factor = self.right_factor_table.factor(ridx)
                    key = (
                        int(out_idx),
                        int(out_comp),
                        int(out_slice.start),
                        int(out_slice.stop),
                        left_shape,
                        tuple(int(dim) for dim in right_factor.shape),
                    )
                    bucket = grouped.setdefault(key, {"left": [], "right": []})
                    bucket["left"].append(left_factor)
                    bucket["right"].append(right_factor)
            for shape_key in sorted(grouped, key=lambda key: basis[key[0]].offset):
                out_idx, out_comp, out_start, out_stop, _left_shape, _right_shape = shape_key
                output_entry = basis[int(out_idx)]
                bucket = grouped[shape_key]
                kernel = self._factorized_kernel(
                    np.ascontiguousarray(np.stack(bucket["left"], axis=0)),
                    np.ascontiguousarray(np.stack(bucket["right"], axis=0)),
                    in_entry,
                    output_entry,
                )
                block_key = (int(in_comp), int(out_comp))
                block = blocks.get(block_key)
                if block is None:
                    block = np.zeros(
                        (
                            int(component_dims[int(out_comp)]),
                            int(component_dims[int(in_comp)]),
                        ),
                        dtype=complex,
                    )
                    blocks[block_key] = block
                block[slice(int(out_start), int(out_stop)), in_slice] += np.asarray(
                    kernel,
                    dtype=complex,
                )
        return tuple(
            (in_comp, out_comp, np.ascontiguousarray(block))
            for (in_comp, out_comp), block in sorted(blocks.items())
        )

    def build_component_parent_blocks_from_matches(
        self,
        basis,
        components,
        component_dims,
        in_indices,
        out_indices,
        left_indices,
        right_indices,
    ):
        """
        Build component-parent dense blocks from precomputed packed matches.
        """

        if not self.supported or components is None:
            return None
        native_blocks = self._build_component_parent_blocks_native(
            basis,
            components,
            component_dims,
            in_indices,
            out_indices,
            left_indices,
            right_indices,
        )
        if native_blocks is not None:
            return native_blocks
        schedule = self._component_parent_block_layout(
            basis,
            components,
            in_indices,
            out_indices,
            left_indices,
            right_indices,
        )
        if schedule is not None:
            return self._build_component_parent_blocks_from_layout(
                basis,
                component_dims,
                schedule,
            )
        entry_to_component = self._entry_component_map(basis, components)
        if len(entry_to_component) != len(tuple(basis.entries)):
            return None
        grouped_by_input = {}
        for in_idx, out_idx, lidx, ridx in zip(
            np.asarray(in_indices, dtype=np.int64),
            np.asarray(out_indices, dtype=np.int64),
            np.asarray(left_indices, dtype=np.int64),
            np.asarray(right_indices, dtype=np.int64),
        ):
            in_idx = int(in_idx)
            out_idx = int(out_idx)
            in_info = entry_to_component.get(in_idx)
            out_info = entry_to_component.get(out_idx)
            if in_info is None or out_info is None:
                return None
            in_comp, in_slice = in_info
            out_comp, out_slice = out_info
            left_factor = self.left_factor_table.factor(int(lidx))
            right_factor = self.right_factor_table.factor(int(ridx))
            key = (
                in_idx,
                out_idx,
                int(in_comp),
                int(out_comp),
                int(in_slice.start),
                int(in_slice.stop),
                int(out_slice.start),
                int(out_slice.stop),
                tuple(int(dim) for dim in left_factor.shape),
                tuple(int(dim) for dim in right_factor.shape),
            )
            bucket = grouped_by_input.setdefault(key, {"left": [], "right": []})
            bucket["left"].append(left_factor)
            bucket["right"].append(right_factor)
        blocks = {}
        for key in sorted(grouped_by_input, key=lambda item: (item[2], item[3], basis[item[1]].offset)):
            (
                in_idx,
                out_idx,
                in_comp,
                out_comp,
                in_start,
                in_stop,
                out_start,
                out_stop,
                _left_shape,
                _right_shape,
            ) = key
            bucket = grouped_by_input[key]
            kernel = self._factorized_kernel(
                np.ascontiguousarray(np.stack(bucket["left"], axis=0)),
                np.ascontiguousarray(np.stack(bucket["right"], axis=0)),
                basis[int(in_idx)],
                basis[int(out_idx)],
            )
            block_key = (int(in_comp), int(out_comp))
            block = blocks.get(block_key)
            if block is None:
                block = np.zeros(
                    (
                        int(component_dims[int(out_comp)]),
                        int(component_dims[int(in_comp)]),
                    ),
                    dtype=complex,
                )
                blocks[block_key] = block
            block[slice(int(out_start), int(out_stop)), slice(int(in_start), int(in_stop))] += np.asarray(
                kernel,
                dtype=complex,
            )
        return tuple(
            (in_comp, out_comp, np.ascontiguousarray(block))
            for (in_comp, out_comp), block in sorted(blocks.items())
        )

    def _component_parent_block_layout(
        self,
        basis,
        components,
        in_indices,
        out_indices,
        left_indices,
        right_indices,
    ):
        cache_key = self._component_parent_block_layout_cache_key(basis, components)
        cached = _COMPONENT_PARENT_BLOCK_LAYOUT_CACHE.get(cache_key)
        if cached is not None:
            _COMPONENT_PARENT_BLOCK_LAYOUT_CACHE_STATS["hits"] += 1
            return cached
        _COMPONENT_PARENT_BLOCK_LAYOUT_CACHE_STATS["misses"] += 1
        entry_to_component = self._entry_component_map(basis, components)
        entries = tuple(basis.entries)
        if len(entry_to_component) != len(entries):
            return None
        left_shape_by_entry = tuple(
            self.left_factor_table.factor_pool.shape(
                int(self.left_factor_table.factor_indices[entry_idx])
            )
            for entry_idx in range(int(self.left_factor_table.n_entries))
        )
        right_shape_by_entry = tuple(
            self.right_factor_table.factor_pool.shape(
                int(self.right_factor_table.factor_indices[entry_idx])
            )
            for entry_idx in range(int(self.right_factor_table.n_entries))
        )
        grouped = {}
        for in_idx, out_idx, lidx, ridx in zip(
            np.asarray(in_indices, dtype=np.int64),
            np.asarray(out_indices, dtype=np.int64),
            np.asarray(left_indices, dtype=np.int64),
            np.asarray(right_indices, dtype=np.int64),
        ):
            in_idx = int(in_idx)
            out_idx = int(out_idx)
            in_info = entry_to_component.get(in_idx)
            out_info = entry_to_component.get(out_idx)
            if in_info is None or out_info is None:
                return None
            in_comp, in_slice = in_info
            out_comp, out_slice = out_info
            key = (
                in_idx,
                out_idx,
                int(in_comp),
                int(out_comp),
                int(in_slice.start),
                int(in_slice.stop),
                int(out_slice.start),
                int(out_slice.stop),
                left_shape_by_entry[int(lidx)],
                right_shape_by_entry[int(ridx)],
            )
            bucket = grouped.setdefault(key, ([], []))
            bucket[0].append(int(lidx))
            bucket[1].append(int(ridx))
        schedule = tuple(
            (
                int(key[0]),
                int(key[1]),
                int(key[2]),
                int(key[3]),
                int(key[4]),
                int(key[5]),
                int(key[6]),
                int(key[7]),
                tuple(int(idx) for idx in grouped[key][0]),
                tuple(int(idx) for idx in grouped[key][1]),
            )
            for key in sorted(
                grouped,
                key=lambda item: (int(item[2]), int(item[3]), basis[int(item[1])].offset),
            )
        )
        if len(_COMPONENT_PARENT_BLOCK_LAYOUT_CACHE) > 256:
            _COMPONENT_PARENT_BLOCK_LAYOUT_CACHE.clear()
        _COMPONENT_PARENT_BLOCK_LAYOUT_CACHE[cache_key] = schedule
        _COMPONENT_PARENT_BLOCK_LAYOUT_CACHE_STATS["puts"] += 1
        return schedule

    def _build_component_parent_blocks_from_layout(
        self,
        basis,
        component_dims,
        schedule,
    ):
        schedule = tuple(schedule)
        def packed_kernel(left_stack, left_meta, right_stack, right_meta, in_idx, out_idx):
            if left_meta is None or right_meta is None:
                return self._factorized_kernel(
                    left_stack,
                    right_stack,
                    basis[int(in_idx)],
                    basis[int(out_idx)],
                )
            left_mat, left_dims = left_meta
            right_mat, right_dims = right_meta
            tdim, ldim, kdim, wdim, adim, bdim = left_dims
            r_tdim, r_wdim, qdim, rdim, ddim, cdim = right_dims
            if tdim != r_tdim or wdim != r_wdim:
                return self._factorized_kernel(
                    left_stack,
                    right_stack,
                    basis[int(in_idx)],
                    basis[int(out_idx)],
                )
            kernel = (left_mat @ right_mat).reshape(
                ldim,
                adim,
                kdim,
                bdim,
                ddim,
                qdim,
                cdim,
                rdim,
            )
            kernel = kernel.transpose(0, 1, 4, 5, 2, 3, 6, 7)
            return np.ascontiguousarray(
                kernel.reshape(
                    int(basis[int(out_idx)].size),
                    int(basis[int(in_idx)].size),
                )
            )

        blocks = {}
        component_pairs = {
            (int(item[2]), int(item[3]))
            for item in schedule
        }
        skipped_reverse_pairs = set()
        for (
            in_idx,
            out_idx,
            in_comp,
            out_comp,
            in_start,
            in_stop,
            out_start,
            out_stop,
            left_group,
            right_group,
        ) in schedule:
            block_key = (int(in_comp), int(out_comp))
            reverse_key = (int(out_comp), int(in_comp))
            if (
                block_key != reverse_key
                and reverse_key in component_pairs
                and reverse_key < block_key
            ):
                skipped_reverse_pairs.add(block_key)
                continue
            left_stack, left_meta = self.left_factor_table.packed_component_parent_group(
                left_group,
                "left",
            )
            right_stack, right_meta = self.right_factor_table.packed_component_parent_group(
                right_group,
                "right",
            )
            kernel = packed_kernel(
                left_stack,
                left_meta,
                right_stack,
                right_meta,
                in_idx,
                out_idx,
            )
            block_key = (int(in_comp), int(out_comp))
            block = blocks.get(block_key)
            if block is None:
                block = np.zeros(
                    (
                        int(component_dims[int(out_comp)]),
                        int(component_dims[int(in_comp)]),
                    ),
                    dtype=complex,
                )
                blocks[block_key] = block
            block[slice(int(out_start), int(out_stop)), slice(int(in_start), int(in_stop))] += np.asarray(
                kernel,
                dtype=complex,
            )
        for block_key in tuple(skipped_reverse_pairs):
            reverse_key = (int(block_key[1]), int(block_key[0]))
            reverse_block = blocks.get(reverse_key)
            if reverse_block is not None and block_key not in blocks:
                blocks[block_key] = np.ascontiguousarray(reverse_block.conj().T)
        return tuple(
            (in_comp, out_comp, np.ascontiguousarray(block))
            for (in_comp, out_comp), block in sorted(blocks.items())
        )

    def _build_component_parent_blocks_native(
        self,
        basis,
        components,
        component_dims,
        in_indices,
        out_indices,
        left_indices,
        right_indices,
    ):
        if not _USE_CYTHON_PARENT_BLOCKS:
            return None
        try:
            from pyqed.mps.nonabelian import _su2_kernel as kernel

            build_blocks = getattr(
                kernel,
                "build_su2_qchem_parent_blocks_from_matches",
            )
        except Exception:
            build_blocks = None
        if build_blocks is None:
            return None
        left_pool = self.left_factor_table.factor_pool
        right_pool = self.right_factor_table.factor_pool
        if np.iscomplexobj(left_pool.data) or np.iscomplexobj(right_pool.data):
            return None
        entry_to_component = self._entry_component_map(basis, components)
        if len(entry_to_component) != len(tuple(basis.entries)):
            return None
        shapes = np.asarray(
            [tuple(int(dim) for dim in entry.shape) for entry in basis],
            dtype=np.int64,
        )
        comp_ids = np.full(len(tuple(basis.entries)), -1, dtype=np.int64)
        starts = np.zeros(len(tuple(basis.entries)), dtype=np.int64)
        for entry_idx, (comp_idx, entry_slice) in entry_to_component.items():
            comp_ids[int(entry_idx)] = int(comp_idx)
            starts[int(entry_idx)] = int(entry_slice.start)
        return build_blocks(
            shapes,
            comp_ids,
            starts,
            np.asarray(component_dims, dtype=np.int64),
            in_indices,
            out_indices,
            left_indices,
            right_indices,
            left_pool.data,
            left_pool.offsets,
            left_pool.shape_offsets,
            left_pool.shapes,
            self.left_factor_table.factor_indices,
            right_pool.data,
            right_pool.offsets,
            right_pool.shape_offsets,
            right_pool.shapes,
            self.right_factor_table.factor_indices,
        )
