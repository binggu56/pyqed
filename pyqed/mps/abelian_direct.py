"""Small Abelian direct-family building blocks for spatial DMRG."""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field

import numpy as np

try:  # Optional fast block kernels; all callers keep NumPy fallbacks.
    from pyqed.mps import packed_cython as _packed_cython
except Exception:  # pragma: no cover - optional extension import guard
    _packed_cython = None

_cpp_davidson = None
_cpp_davidson_checked = False


_LEFT_IDENTITY_ADVANCE_GROUP_CACHE = {}
_RIGHT_IDENTITY_ADVANCE_GROUP_CACHE = {}
_LEFT_ADVANCE_GROUP_CACHE = {}
_RIGHT_ADVANCE_GROUP_CACHE = {}
_IDENTITY_ADVANCE_GROUP_CACHE_LIMIT = 4096
_PACKED_LOCAL_PAYLOAD_STATS = Counter()
_PACKED_BOUNDARY_ADVANCE_PAYLOAD_STATS = Counter()
_ABELIAN_ENVIRONMENT_ADVANCE_PAYLOAD_STATS = Counter()
_ABELIAN_SVD_KERNEL_STATS = Counter()
_ABELIAN_SVD_KERNEL_LAST_ERROR = ""


def _cpp_table_kernel(name):
    global _cpp_davidson
    global _cpp_davidson_checked
    if not bool(_cpp_davidson_checked):
        _cpp_davidson_checked = True
        try:  # Optional pybind helpers for hot table resolve/store loops.
            from pyqed.mps import cpp_davidson as module
        except Exception:  # pragma: no cover - optional extension import guard
            module = None
        _cpp_davidson = module
    if _cpp_davidson is None:
        return None
    if not bool(getattr(_cpp_davidson, "CPP_DAVIDSON_AVAILABLE", False)):
        return None
    kernel = getattr(_cpp_davidson, str(name), None)
    if kernel is None:
        return None
    return kernel


class AbelianPackedBoundaryTensor:
    """Columnar boundary tensor used by packed direct-family route builders."""

    __slots__ = ("keys", "blocks", "dirs", "qns", "rank", "source", "_data", "_signature")

    _pyqed_packed_boundary_tensor = True

    def __init__(
        self,
        keys,
        blocks,
        *,
        dirs=(),
        qns=None,
        source="packed_boundary_tensor",
        assume_unique=False,
    ):
        keys = tuple(tuple(key) for key in (keys or ()))
        blocks = tuple(np.asarray(block) for block in (blocks or ()))
        if len(keys) != len(blocks):
            raise ValueError("packed boundary tensor keys/blocks length mismatch")
        duplicate = False
        if not bool(assume_unique):
            seen = set()
            for key in keys:
                if key in seen:
                    duplicate = True
                    break
                seen.add(key)
        if duplicate:
            data = OrderedDict()
            for key, block in zip(keys, blocks):
                old = data.get(key)
                if old is None:
                    data[key] = block
                else:
                    if tuple(old.shape) != tuple(block.shape):
                        raise ValueError(
                            "packed boundary tensor duplicate key has incompatible block shape"
                        )
                    data[key] = old + block
            self.keys = tuple(data.keys())
            self.blocks = tuple(data.values())
            self._data = data
        else:
            self.keys = keys
            self.blocks = blocks
            self._data = None
        self.dirs = list(dirs or ())
        self.qns = qns
        self.rank = len(self.dirs)
        self.source = str(source)
        self._signature = None

    @classmethod
    def from_tensor(cls, tensor, *, source="packed_boundary_tensor"):
        if is_abelian_packed_boundary_tensor(tensor):
            return tensor
        data = getattr(tensor, "data", None)
        if data is None:
            raise TypeError("boundary tensor must provide data blocks")
        items = tuple(data.items())
        return cls(
            tuple(key for key, _block in items),
            tuple(block for _key, block in items),
            dirs=getattr(tensor, "dirs", ()),
            qns=getattr(tensor, "qns", None),
            source=source,
            assume_unique=True,
        )

    @property
    def data(self):
        if self._data is None:
            self._data = OrderedDict(
                (key, block) for key, block in zip(self.keys, self.blocks)
            )
        return self._data

    def __len__(self):
        return len(self.keys)

    def __bool__(self):
        return bool(self.keys)

    def block_shape_signature(self):
        return tuple(
            (repr(key), tuple(int(dim) for dim in np.asarray(block).shape))
            for key, block in zip(self.keys, self.blocks)
        )

    def structural_signature(self):
        """Stable key for exact packed-entry coalescing."""

        if self._signature is None:
            items = []
            for key, block in zip(self.keys, self.blocks):
                arr = np.ascontiguousarray(block)
                digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
                items.append(
                    (
                        tuple(repr(qn) for qn in key),
                        tuple(int(dim) for dim in arr.shape),
                        str(arr.dtype),
                        digest,
                    )
                )
            self._signature = (
                "abelian_packed_boundary_tensor",
                tuple(self.dirs),
                tuple(items),
            )
        return self._signature


def is_abelian_packed_boundary_tensor(tensor):
    return bool(getattr(tensor, "_pyqed_packed_boundary_tensor", False))


def abelian_packed_boundary_advance_payload_stats():
    return {str(key): int(value) for key, value in _PACKED_BOUNDARY_ADVANCE_PAYLOAD_STATS.items()}


def abelian_environment_advance_payload_stats():
    return {
        str(key): int(value)
        for key, value in _ABELIAN_ENVIRONMENT_ADVANCE_PAYLOAD_STATS.items()
    }


def abelian_svd_kernel_stats():
    stats = {str(key): int(value) for key, value in _ABELIAN_SVD_KERNEL_STATS.items()}
    if _ABELIAN_SVD_KERNEL_LAST_ERROR:
        stats["cpp_kernel_last_error"] = _ABELIAN_SVD_KERNEL_LAST_ERROR
        stats["cpp_full_split_last_error"] = _ABELIAN_SVD_KERNEL_LAST_ERROR
    return stats


def pack_abelian_boundary_tensor(tensor, *, source="packed_boundary_tensor"):
    if tensor is None or is_abelian_packed_boundary_tensor(tensor):
        return tensor
    return AbelianPackedBoundaryTensor.from_tensor(tensor, source=source)


@dataclass
class AbelianPackedTensorViewCache:
    """Cache packed views and conjugates for boundary/MPS tensors."""

    source_prefix: str = "direct_family"
    _view_cache: dict = field(default_factory=dict, init=False)
    _conj_cache: dict = field(default_factory=dict, init=False)
    _created: int = 0
    _blocks: int = 0
    _discarded: int = 0
    _last_source: str = ""

    def view(self, tensor, source):
        if is_abelian_packed_boundary_tensor(tensor):
            return tensor
        key = id(tensor)
        cached = self._view_cache.get(key)
        if cached is not None:
            return cached
        packed = pack_abelian_boundary_tensor(
            tensor,
            source=f"{self.source_prefix}_{source}",
        )
        self._view_cache[key] = packed
        self._created += 1
        self._blocks += int(len(packed))
        self._last_source = str(source)
        return packed

    def conj(self, tensor, source):
        key = id(tensor)
        cached = self._conj_cache.get(key)
        if cached is not None:
            return cached
        tensor = self.view(tensor, f"{source}_base")
        result = conjugate_abelian_packed_boundary_tensor(
            tensor,
            source=source,
        )
        self._conj_cache[key] = result
        return result

    def discard(self, *tensors):
        removed = 0
        for tensor in tensors:
            key = id(tensor)
            if key in self._view_cache:
                self._view_cache.pop(key, None)
                removed += 1
            if key in self._conj_cache:
                self._conj_cache.pop(key, None)
                removed += 1
        self._discarded += int(removed)
        return int(removed)

    @property
    def stats(self):
        return {
            "created": int(self._created),
            "blocks": int(self._blocks),
            "discarded": int(self._discarded),
            "last_source": str(self._last_source),
            "view_cache": int(len(self._view_cache)),
            "conj_cache": int(len(self._conj_cache)),
        }


def abelian_local_layout_from_data(data):
    """Return a stable ``((sector_key, block_shape), ...)`` local layout."""

    data = data or {}
    keys = tuple(sorted(data))
    return tuple(
        (tuple(key), tuple(int(dim) for dim in np.asarray(data[key]).shape))
        for key in keys
    )


def abelian_local_layout_from_tensor(tensor):
    return abelian_local_layout_from_data(getattr(tensor, "data", {}) or {})


def abelian_qns_from_layout(layout, proto=None):
    """Build axis quantum-number lists from a local block layout."""

    layout = tuple(layout or ())
    if not layout:
        if proto is not None:
            return [list(axis_qns) for axis_qns in getattr(proto, "qns", ())]
        return []
    rank = len(layout[0][0])
    if proto is not None and len(getattr(proto, "qns", ())) == rank:
        qns = [list(axis_qns) for axis_qns in proto.qns]
    else:
        qns = [sorted({key[axis] for key, _shape in layout}) for axis in range(rank)]
    for axis in range(rank):
        seen = set(qns[axis])
        extras = sorted({key[axis] for key, _shape in layout if key[axis] not in seen})
        qns[axis].extend(extras)
    return qns


def abelian_local_layout_size(layout):
    return int(sum(np.prod(shape, dtype=int) for _key, shape in tuple(layout or ())))


@dataclass(frozen=True)
class AbelianLocalVectorLayout:
    """Flat-vector adapter for Abelian local block layouts."""

    layout: tuple
    qns: tuple = ()
    dirs: tuple = ()

    @classmethod
    def from_tensor(cls, tensor):
        layout = abelian_local_layout_from_tensor(tensor)
        qns = abelian_qns_from_layout(layout, tensor)
        dirs = tuple(getattr(tensor, "dirs", ()))
        return cls(
            layout=layout,
            qns=tuple(tuple(axis_qns) for axis_qns in qns),
            dirs=dirs,
        )

    @classmethod
    def from_layout(cls, layout, *, proto=None, dirs=None, qns=None):
        layout = tuple(
            (tuple(key), tuple(int(dim) for dim in shape))
            for key, shape in tuple(layout or ())
        )
        if qns is None:
            qns = abelian_qns_from_layout(layout, proto)
        if dirs is None and proto is not None:
            dirs = getattr(proto, "dirs", ())
        return cls(
            layout=layout,
            qns=tuple(tuple(axis_qns) for axis_qns in (qns or ())),
            dirs=tuple(dirs or ()),
        )

    @property
    def size(self):
        return abelian_local_layout_size(self.layout)

    @property
    def offsets(self):
        result = {}
        pos = 0
        for key, shape in self.layout:
            n = int(np.prod(shape, dtype=int))
            result[key] = (pos, n)
            pos += n
        return result, pos

    @property
    def entries(self):
        pos = 0
        entries = []
        for key, shape in self.layout:
            n = int(np.prod(shape, dtype=int))
            entries.append((key, shape, pos, n))
            pos += n
        return tuple(entries)

    def flatten_data(self, data, *, dtype=None):
        global _ABELIAN_SVD_KERNEL_LAST_ERROR

        if dtype is None:
            blocks = [
                np.asarray(block).dtype
                for block in (data or {}).values()
            ]
            dtype = np.result_type(*(blocks or [complex]))
        dtype = np.dtype(dtype)
        native_flatten = _cpp_table_kernel("abelian_flatten_data_to_layout")
        if native_flatten is not None and dtype != np.dtype(object):
            try:
                out = np.asarray(native_flatten(data or {}, self.layout))
                _ABELIAN_SVD_KERNEL_STATS["cpp_flatten_calls"] += 1
                _ABELIAN_SVD_KERNEL_STATS["cpp_flatten_blocks"] += int(len(self.layout))
                _ABELIAN_SVD_KERNEL_STATS["cpp_flatten_dim"] += int(out.size)
                if dtype.kind == "c":
                    return out.astype(dtype, copy=False)
                return out.real.astype(dtype, copy=False)
            except Exception as exc:
                _ABELIAN_SVD_KERNEL_STATS["cpp_flatten_failures"] += 1
                _ABELIAN_SVD_KERNEL_LAST_ERROR = repr(exc)
        chunks = []
        for key, shape in self.layout:
            block = (data or {}).get(key)
            if block is None:
                chunks.append(np.zeros(int(np.prod(shape, dtype=int)), dtype=dtype))
            else:
                chunks.append(np.asarray(block, dtype=dtype).reshape(-1))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=dtype)

    def flatten_tensor(self, tensor, *, dtype=None):
        return self.flatten_data(getattr(tensor, "data", {}) or {}, dtype=dtype)

    def unflatten_data(self, vec, *, drop_zero_blocks=False, zero_tol=0.0):
        data = {}
        arr = np.asarray(vec)
        pos = 0
        for key, shape in self.layout:
            n = int(np.prod(shape, dtype=int))
            block_vec = arr[pos:pos + n]
            if drop_zero_blocks and float(np.linalg.norm(block_vec)) <= float(zero_tol):
                pos += n
                continue
            data[key] = block_vec.reshape(shape).copy()
            pos += n
        return data

    def basis_data(self, flat_index, *, dtype=complex):
        flat_index = int(flat_index)
        for key, shape, start, n in self.entries:
            if start <= flat_index < start + n:
                arr = np.zeros(shape, dtype=dtype)
                arr.reshape(-1)[flat_index - start] = 1.0
                return {key: arr}
        raise IndexError("flat basis index out of local layout range")

    def zero_data(self, *, dtype=complex):
        return {
            key: np.zeros(tuple(shape), dtype=dtype)
            for key, shape in self.layout
        }


def abelian_block_data_dtype(*objects):
    """Infer a result dtype from nested Abelian block-data carriers."""

    dtypes = []

    def collect(obj):
        if obj is None:
            return
        data = getattr(obj, "data", None)
        if isinstance(data, dict):
            for block in data.values():
                collect(block)
            return
        if isinstance(obj, dict):
            for value in obj.values():
                collect(value)
            return
        if isinstance(obj, (list, tuple)):
            for value in obj:
                collect(value)
            return
        try:
            arr = np.asarray(obj)
        except Exception:
            return
        if arr.dtype != object:
            dtypes.append(arr.dtype)

    for obj in objects:
        collect(obj)
    return np.result_type(*(dtypes or [float]))


def abelian_flatten_to_layout(tensor, layout, *, dtype=None, proto=None):
    if dtype is None:
        dtype = abelian_block_data_dtype(tensor)
    return AbelianLocalVectorLayout.from_layout(
        layout,
        proto=tensor if proto is None else proto,
    ).flatten_tensor(tensor, dtype=dtype)


def abelian_unflatten_data_from_layout(
    vec,
    layout,
    *,
    proto=None,
    qns=None,
    dirs=None,
    drop_zero_blocks=False,
    zero_tol=0.0,
):
    local_layout = AbelianLocalVectorLayout.from_layout(
        layout,
        proto=proto,
        qns=qns,
        dirs=dirs,
    )
    return (
        local_layout.unflatten_data(
            vec,
            drop_zero_blocks=drop_zero_blocks,
            zero_tol=zero_tol,
        ),
        [list(axis_qns) for axis_qns in local_layout.qns],
        list(local_layout.dirs),
    )


def abelian_zero_data_from_layout(layout, *, proto=None, dtype=complex):
    local_layout = AbelianLocalVectorLayout.from_layout(layout, proto=proto)
    return (
        local_layout.zero_data(dtype=dtype),
        [list(axis_qns) for axis_qns in local_layout.qns],
        list(local_layout.dirs),
    )


def abelian_layout_offsets(layout):
    return AbelianLocalVectorLayout.from_layout(layout).offsets


def abelian_sector_signature(key, dirs):
    total = None
    for qn, direction in zip(key, dirs):
        term = qn * int(direction)
        if total is None:
            total = term
            continue
        try:
            total = total + term
        except TypeError:
            if term == 0:
                continue
            if total == 0:
                total = term
                continue
            return (total, term)
    return total


def abelian_two_site_mps_flow_valid(key):
    if len(key) != 4:
        return True
    try:
        return key[0] + key[2] + key[3] == key[1]
    except TypeError:
        return True


def abelian_axis_sector_dims(tensor, axis):
    dims = {}
    axis = int(axis)
    try:
        for qn, count in Counter(tensor.qns[axis]).items():
            dims[qn] = max(int(dims.get(qn, 0)), int(count))
    except Exception:
        pass
    for key, block in getattr(tensor, "data", {}).items():
        if len(key) <= axis:
            continue
        dims[key[axis]] = max(
            int(dims.get(key[axis], 0)),
            int(np.asarray(block).shape[axis]),
        )
    return dims


def abelian_merge_axis_sector_dims(dims, tensor, axis):
    for qn, dim in abelian_axis_sector_dims(tensor, axis).items():
        dims[qn] = max(int(dims.get(qn, 0)), int(dim))


def abelian_safe_two_site_layout_map(proto, w_sites=()):
    """All two-site blocks preserving fixed outer MPS sectors and charge flow."""

    if getattr(proto, "rank", None) != 4:
        return None
    left_dims = abelian_axis_sector_dims(proto, 0)
    right_dims = abelian_axis_sector_dims(proto, 1)
    phys_left_dims = abelian_axis_sector_dims(proto, 2)
    phys_right_dims = abelian_axis_sector_dims(proto, 3)
    w_sites = tuple(w_sites or ())
    if len(w_sites) >= 2:
        abelian_merge_axis_sector_dims(phys_left_dims, w_sites[0], 2)
        abelian_merge_axis_sector_dims(phys_left_dims, w_sites[0], 3)
        abelian_merge_axis_sector_dims(phys_right_dims, w_sites[1], 2)
        abelian_merge_axis_sector_dims(phys_right_dims, w_sites[1], 3)
    if not left_dims or not right_dims or not phys_left_dims or not phys_right_dims:
        return None

    allowed = {}
    for q_left, d_left in left_dims.items():
        for q_phys_left, d_phys_left in phys_left_dims.items():
            for q_phys_right, d_phys_right in phys_right_dims.items():
                try:
                    q_right = q_left + q_phys_left + q_phys_right
                except TypeError:
                    return None
                d_right = right_dims.get(q_right)
                if d_right is None:
                    continue
                allowed[(q_left, q_right, q_phys_left, q_phys_right)] = (
                    int(d_left),
                    int(d_right),
                    int(d_phys_left),
                    int(d_phys_right),
                )
    return allowed


def abelian_layout_from_map(layout_map):
    return tuple((key, layout_map[key]) for key in sorted(layout_map))


def abelian_merge_layout_tensor(
    layout_map,
    tensor,
    *,
    dirs=None,
    allowed_signatures=None,
    allowed_layout_map=None,
    require_two_site_mps_flow=False,
):
    """Merge tensor block shapes into a mutable layout map."""

    changed = False
    for key, block in (getattr(tensor, "data", None) or {}).items():
        key = tuple(key)
        if require_two_site_mps_flow and not abelian_two_site_mps_flow_valid(key):
            return None, False
        if allowed_signatures is not None and dirs is not None:
            if abelian_sector_signature(key, dirs) not in allowed_signatures:
                return None, False
        shape = tuple(int(dim) for dim in np.asarray(block).shape)
        if allowed_layout_map is not None:
            allowed_shape = allowed_layout_map.get(key)
            if allowed_shape is None or tuple(allowed_shape) != shape:
                return None, False
        old = layout_map.get(key)
        if old is None:
            layout_map[key] = shape
            changed = True
        elif tuple(old) != shape:
            return None, False
    return abelian_layout_from_map(layout_map), changed


@dataclass(frozen=True)
class AbelianLayoutTruncationResult:
    layout_map: dict | None
    truncated: bool
    retained_blocks: int | None = None
    retained_norm: float | None = None


def abelian_truncate_layout_map_by_norm(
    layout_map,
    block_data,
    max_dim,
    *,
    current_dim=None,
):
    """Keep the largest-norm blocks fitting within ``max_dim``."""

    layout_map = {
        tuple(key): tuple(int(dim) for dim in shape)
        for key, shape in (layout_map or {}).items()
    }
    max_dim = int(max_dim)
    if current_dim is None:
        current_dim = abelian_local_layout_size(abelian_layout_from_map(layout_map))
    if max_dim <= 0 or int(current_dim) <= max_dim:
        return AbelianLayoutTruncationResult(dict(layout_map), False)

    data = getattr(block_data, "data", None)
    if data is None:
        data = block_data or {}
    entries = []
    total_norm_sq = 0.0
    for key, shape in layout_map.items():
        n = int(np.prod(shape, dtype=int))
        block = data.get(key)
        norm_sq = 0.0 if block is None else float(np.linalg.norm(block)) ** 2
        total_norm_sq += norm_sq
        entries.append((norm_sq, n, key, tuple(shape)))
    entries.sort(key=lambda item: (-item[0], item[1], repr(item[2])))

    selected = {}
    used_dim = 0
    retained_norm_sq = 0.0
    for norm_sq, n, key, shape in entries:
        if n > max_dim:
            continue
        if used_dim + n > max_dim:
            continue
        selected[key] = shape
        used_dim += n
        retained_norm_sq += norm_sq
    if not selected:
        return AbelianLayoutTruncationResult(None, False)
    retained_norm = (
        1.0
        if total_norm_sq <= 1.0e-30
        else float(math.sqrt(max(retained_norm_sq, 0.0) / total_norm_sq))
    )
    return AbelianLayoutTruncationResult(
        selected,
        True,
        int(len(selected)),
        retained_norm,
    )


def abelian_project_block_data_to_layout(
    data,
    layout,
    *,
    proto=None,
    qns=None,
    dirs=None,
    dtype=None,
    extra_policy="ignore",
    extra_zero_tol=0.0,
):
    """Project block data onto a fixed flat layout, or return ``None``."""

    layout = tuple(
        (tuple(key), tuple(int(dim) for dim in shape))
        for key, shape in tuple(layout or ())
    )
    data = data or {}
    layout_shapes = {key: shape for key, shape in layout}
    policy = str(extra_policy or "ignore").strip().lower()
    for key, block in data.items():
        key = tuple(key)
        expected = layout_shapes.get(key)
        if expected is None:
            if policy in {"forbid", "reject", "error"}:
                return None
            if policy in {"zero", "zero_only", "allow_zero"}:
                if float(np.linalg.norm(block)) > float(extra_zero_tol):
                    return None
            continue
        if tuple(np.asarray(block).shape) != tuple(expected):
            return None
    local_layout = AbelianLocalVectorLayout.from_layout(
        layout,
        proto=proto,
        qns=qns,
        dirs=dirs,
    )
    return local_layout.flatten_data(data, dtype=dtype)


def abelian_project_tensor_to_layout(tensor, layout, **kwargs):
    return abelian_project_block_data_to_layout(
        getattr(tensor, "data", {}) or {},
        layout,
        proto=tensor if "proto" not in kwargs else kwargs.pop("proto"),
        **kwargs,
    )


@dataclass(frozen=True)
class AbelianLayoutProjectionResult:
    flat: np.ndarray | None
    discarded_blocks: int = 0
    discarded_norm_sq: float = 0.0


def abelian_project_tensor_to_layout_with_stats(tensor, layout, **kwargs):
    """Project tensor data to ``layout`` and count ignored extra blocks."""

    layout = tuple(
        (tuple(key), tuple(int(dim) for dim in shape))
        for key, shape in tuple(layout or ())
    )
    data = getattr(tensor, "data", {}) or {}
    layout_shapes = {key: shape for key, shape in layout}
    discarded_blocks = 0
    discarded_norm_sq = 0.0
    for key, block in data.items():
        key = tuple(key)
        expected = layout_shapes.get(key)
        if expected is None:
            discarded_blocks += 1
            discarded_norm_sq += float(np.linalg.norm(block)) ** 2
            continue
        if tuple(np.asarray(block).shape) != expected:
            return AbelianLayoutProjectionResult(
                None,
                int(discarded_blocks),
                float(discarded_norm_sq),
            )
    flat = abelian_project_tensor_to_layout(
        tensor,
        layout,
        extra_policy="ignore",
        **kwargs,
    )
    return AbelianLayoutProjectionResult(
        flat,
        int(discarded_blocks),
        float(discarded_norm_sq),
    )


def abelian_block_index(tensor, axes):
    axes = tuple(int(axis) for axis in tuple(axes or ()))
    idx = {}
    for key, block in (getattr(tensor, "data", None) or {}).items():
        key = tuple(key)
        bucket = tuple(key[axis] for axis in axes)
        entries = idx.get(bucket)
        if entries is None:
            idx[bucket] = [(key, block)]
        else:
            entries.append((key, block))
    return idx


def abelian_qchem_diagonal_contribution(e_blk, w1_blk, w2_blk, f_blk):
    if e_blk.shape[1] != e_blk.shape[2]:
        return None
    if w1_blk.shape[2] != w1_blk.shape[3]:
        return None
    if w2_blk.shape[2] != w2_blk.shape[3]:
        return None
    if f_blk.shape[1] != f_blk.shape[2]:
        return None
    e_diag = np.einsum("aii->ai", e_blk, optimize=True)
    w1_diag = np.einsum("abuu->abu", w1_blk, optimize=True)
    w2_diag = np.einsum("bcvv->bcv", w2_blk, optimize=True)
    f_diag = np.einsum("cll->cl", f_blk, optimize=True)
    return np.einsum(
        "ai,abu,bcv,cl->iluv",
        e_diag,
        w1_diag,
        w2_diag,
        f_diag,
        optimize=True,
    )


@dataclass(frozen=True)
class AbelianFlatJacobiDiagonalResult:
    flat: np.ndarray | None
    block_data: dict | None = None
    candidate_entries: int = 0
    contributions: int = 0
    diagonal_blocks: int = 0
    rejected_reason: str | None = None


def abelian_flat_qchem_jacobi_diagonal(layout, e, w_sites, f):
    """Build a flat two-site qchem Jacobi diagonal from block-data tensors."""

    layout = tuple(
        (tuple(key), tuple(int(dim) for dim in shape))
        for key, shape in tuple(layout or ())
    )
    if len(tuple(w_sites or ())) < 2:
        return AbelianFlatJacobiDiagonalResult(None, rejected_reason="missing_w_sites")
    dtype = np.result_type(
        abelian_block_data_dtype(e, w_sites[0], w_sites[1], f),
        complex,
    )
    diag_data = {
        key: np.zeros(tuple(shape), dtype=dtype)
        for key, shape in layout
    }
    contributions = 0
    candidate_entries = 0

    e_diag_by_ket_bra = abelian_block_index(e, (2, 1))
    w1_diag_by_left_in_out = abelian_block_index(w_sites[0], (0, 3, 2))
    w2_diag_by_left_in_out = abelian_block_index(w_sites[1], (0, 3, 2))
    f_diag_by_mpo_ket_bra = abelian_block_index(f, (0, 2, 1))
    for a_key, a_shape in layout:
        if len(a_key) != 4 or len(a_shape) != 4:
            return AbelianFlatJacobiDiagonalResult(
                None,
                rejected_reason="unsupported_layout",
            )
        left_qn, right_qn, p1_in, p2_in = a_key
        for e_key, e_blk in e_diag_by_ket_bra.get((left_qn, left_qn), ()):
            for w1_key, w1_blk in w1_diag_by_left_in_out.get(
                (e_key[0], p1_in, p1_in),
                (),
            ):
                channel = w1_key[1]
                for w2_key, w2_blk in w2_diag_by_left_in_out.get(
                    (channel, p2_in, p2_in),
                    (),
                ):
                    for _f_key, f_blk in f_diag_by_mpo_ket_bra.get(
                        (w2_key[1], right_qn, right_qn),
                        (),
                    ):
                        candidate_entries += 1
                        contrib = abelian_qchem_diagonal_contribution(
                            e_blk,
                            w1_blk,
                            w2_blk,
                            f_blk,
                        )
                        if contrib is None or tuple(contrib.shape) != tuple(a_shape):
                            continue
                        diag_data[a_key] += contrib
                        contributions += 1

    if contributions == 0:
        return AbelianFlatJacobiDiagonalResult(
            None,
            diag_data,
            int(candidate_entries),
            0,
            int(len(diag_data)),
            rejected_reason="no_contributions",
        )
    flat = AbelianLocalVectorLayout.from_layout(layout).flatten_data(
        diag_data,
        dtype=np.complex128,
    )
    return AbelianFlatJacobiDiagonalResult(
        flat,
        diag_data,
        int(candidate_entries),
        int(contributions),
        int(len(diag_data)),
    )


@dataclass(frozen=True)
class AbelianBlockPreconditionerBuildResult:
    blocks: dict
    used_dim: int = 0
    attempted_blocks: int = 0
    failed_blocks: int = 0
    skipped_blocks: int = 0
    columns: int = 0


def abelian_build_block_preconditioner_blocks(
    layout,
    matvec_flat,
    *,
    max_block_dim,
    max_total_dim=0,
    dtype=complex,
):
    """Build small dense sector-block preconditioners using a flat matvec."""

    layout = tuple(
        (tuple(key), tuple(int(dim) for dim in shape))
        for key, shape in tuple(layout or ())
    )
    offsets, total_dim = abelian_layout_offsets(layout)
    max_block_dim = int(max_block_dim)
    max_total_dim = int(max_total_dim)
    used_dim = 0
    attempted = 0
    failed = 0
    skipped = 0
    columns = 0
    blocks = {}
    for key, _shape in layout:
        start, n = offsets[key]
        start = int(start)
        n = int(n)
        if n <= 1 or n > max_block_dim:
            skipped += 1
            continue
        if max_total_dim > 0 and used_dim + n > max_total_dim:
            skipped += 1
            continue
        attempted += 1
        mat = np.zeros((n, n), dtype=dtype)
        block_failed = False
        for col in range(n):
            basis = np.zeros(int(total_dim), dtype=dtype)
            basis[start + col] = 1.0
            flat = matvec_flat(basis, layout)
            columns += 1
            if flat is None:
                block_failed = True
                break
            flat = np.asarray(flat, dtype=dtype).reshape(int(total_dim))
            mat[:, col] = flat[start:start + n]
        if block_failed:
            failed += 1
            continue
        blocks[key] = (start, n, 0.5 * (mat + mat.conj().T))
        used_dim += n
    return AbelianBlockPreconditionerBuildResult(
        blocks,
        int(used_dim),
        int(attempted),
        int(failed),
        int(skipped),
        int(columns),
    )


def abelian_apply_block_preconditioner(resid, theta, base, blocks):
    if not blocks:
        return base
    resid = np.asarray(resid)
    out = np.asarray(base, dtype=np.result_type(np.asarray(base).dtype, complex)).copy()
    theta = complex(theta)
    for _key, (start, n, mat) in dict(blocks).items():
        start = int(start)
        n = int(n)
        system = theta * np.eye(n, dtype=complex) - np.asarray(mat, dtype=complex)
        rhs = resid[start:start + n]
        try:
            out[start:start + n] = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            out[start:start + n] = np.linalg.pinv(system) @ rhs
    return out


def abelian_apply_jacobi_preconditioner(resid, theta, diagonal, *, floor=1.0e-8):
    resid = np.asarray(resid)
    diagonal = np.asarray(diagonal)
    if int(diagonal.size) != int(resid.size):
        return None
    denom = complex(theta) - diagonal.reshape(resid.shape)
    floor = float(floor)
    finite = np.isfinite(np.real(denom)) & np.isfinite(np.imag(denom))
    small = np.abs(denom) < floor
    replace = (~finite) | small
    if np.any(replace):
        sign = np.where(np.real(denom) >= 0.0, 1.0, -1.0)
        denom = np.where(replace, sign * floor, denom)
    return resid / denom


def abelian_extend_projected_hamiltonian(projected, basis, image):
    """Append one column/row to a Davidson projected Hamiltonian."""

    basis = tuple(np.asarray(vec) for vec in tuple(basis or ()))
    image = np.asarray(image)
    m = len(basis)
    old = np.asarray(projected)
    dtype = np.result_type(old.dtype, image.dtype, *(vec.dtype for vec in basis))
    out = np.zeros((m, m), dtype=dtype)
    if m > 1:
        out[:-1, :-1] = old
    for i, vec in enumerate(basis):
        el = np.vdot(vec, image)
        out[i, m - 1] = el
        out[m - 1, i] = el.conjugate()
    return out


@dataclass(frozen=True)
class AbelianDavidsonRitzResult:
    energy: complex
    vector: np.ndarray
    image: np.ndarray
    residual: np.ndarray
    residual_norm: float
    coefficients: np.ndarray


def abelian_lowest_ritz_state(projected, basis, images):
    """Return the lowest Ritz state from flat Davidson basis vectors."""

    basis = tuple(np.asarray(vec) for vec in tuple(basis or ()))
    images = tuple(np.asarray(vec) for vec in tuple(images or ()))
    if not basis or len(basis) != len(images):
        return None
    projected = np.asarray(projected)
    values, coeffs = np.linalg.eigh(projected)
    idx = np.argsort(np.real(values))
    values = values[idx]
    coeffs = coeffs[:, idx]
    coeff = coeffs[:, 0]
    dtype = np.result_type(*(vec.dtype for vec in basis), coeff.dtype)
    ritz_vec = np.zeros_like(basis[0], dtype=dtype)
    ritz_image = np.zeros_like(ritz_vec)
    for i, coeff_i in enumerate(coeff):
        ritz_vec = ritz_vec + basis[i] * coeff_i
        ritz_image = ritz_image + images[i] * coeff_i
    energy = values[0]
    residual = ritz_image - ritz_vec * energy
    return AbelianDavidsonRitzResult(
        complex(energy),
        ritz_vec,
        ritz_image,
        residual,
        float(np.linalg.norm(residual)),
        np.asarray(coeff),
    )


def abelian_restart_basis_from_vector(vec, *, min_norm=1.0e-12):
    vec = np.asarray(vec)
    norm = float(np.linalg.norm(vec))
    if norm < float(min_norm):
        return None
    return vec / norm


@dataclass(frozen=True)
class AbelianNormalizedFlatVector:
    vector: np.ndarray | None
    norm: float
    accepted: bool


def abelian_normalize_flat_vector(vec, *, min_norm=1.0e-12):
    vec = np.asarray(vec)
    norm = float(np.linalg.norm(vec))
    if norm < float(min_norm):
        return AbelianNormalizedFlatVector(None, norm, False)
    return AbelianNormalizedFlatVector(vec / norm, norm, True)


def _abelian_stable_sector_sort_key(value):
    if hasattr(value, "labels") and hasattr(value, "components"):
        return (
            type(value).__name__,
            tuple(str(label) for label in value.labels),
            tuple(
                _abelian_stable_sector_sort_key(component)
                for component in value.components
            ),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_abelian_stable_sector_sort_key(item) for item in value))
    if isinstance(value, (np.integer, int)):
        return ("int", int(value))
    if isinstance(value, (np.floating, float)):
        return ("float", float(value))
    if isinstance(value, str):
        return ("str", value)
    return (type(value).__name__, repr(value))


def _abelian_cluster_sorted_values(values, rtol=1.0e-10, atol=1.0e-12):
    values = np.asarray(values, dtype=float)
    clusters = []
    start = 0
    n = len(values)
    while start < n:
        stop = start + 1
        reference = float(abs(values[start]))
        while stop < n:
            current = float(abs(values[stop]))
            tol = max(float(atol), float(rtol) * max(reference, current))
            if abs(float(values[stop]) - float(values[start])) > tol:
                break
            stop += 1
        clusters.append((start, stop))
        start = stop
    return clusters


def _abelian_sort_singular_entries(entries, rtol=1.0e-10, atol=1.0e-12):
    if not entries:
        return []
    ordered = sorted(
        entries,
        key=lambda item: (
            -float(np.real(item[0])),
            _abelian_stable_sector_sort_key(item[1]),
            int(item[2]),
        ),
    )
    values = [float(np.real(item[0])) for item in ordered]
    stable = []
    for start, stop in _abelian_cluster_sorted_values(values, rtol=rtol, atol=atol):
        stable.extend(
            sorted(
                ordered[start:stop],
                key=lambda item: (_abelian_stable_sector_sort_key(item[1]), int(item[2])),
            )
        )
    return stable


def _abelian_fix_vector_phase_inplace(vec):
    if vec.size == 0:
        return vec
    idx = int(np.argmax(np.abs(vec)))
    ref = vec[idx]
    if abs(ref) <= 0.0:
        return vec
    if np.iscomplexobj(vec):
        vec *= np.conj(ref) / abs(ref)
    elif ref < 0:
        vec *= -1.0
    return vec


def _abelian_canonical_subspace_rotation(left_basis):
    ncol = left_basis.shape[1]
    if ncol <= 1:
        return np.eye(ncol, dtype=left_basis.dtype)
    dtype = np.result_type(
        left_basis.dtype,
        np.complex128 if np.iscomplexobj(left_basis) else np.float64,
    )
    vectors = []
    tol = 100.0 * np.finfo(float).eps * max(1, ncol)
    for row in range(left_basis.shape[0]):
        candidate = np.asarray(left_basis[row, :].conj(), dtype=dtype).copy()
        for _ in range(2):
            for basis_vec in vectors:
                candidate -= basis_vec * np.vdot(basis_vec, candidate)
        norm = np.linalg.norm(candidate)
        if norm <= tol:
            continue
        candidate /= norm
        _abelian_fix_vector_phase_inplace(candidate)
        vectors.append(candidate)
        if len(vectors) == ncol:
            break
    if len(vectors) != ncol:
        return np.eye(ncol, dtype=dtype)
    return np.column_stack(vectors)


def _abelian_canonicalize_svd_pair(U, S, Vt, rtol=1.0e-10, atol=1.0e-12):
    if len(S) == 0:
        return U, S, Vt
    for start, stop in _abelian_cluster_sorted_values(S, rtol=rtol, atol=atol):
        if stop - start <= 1:
            continue
        rotation = _abelian_canonical_subspace_rotation(U[:, start:stop])
        U[:, start:stop] = U[:, start:stop] @ rotation
        Vt[start:stop, :] = rotation.conj().T @ Vt[start:stop, :]
    for i in range(min(U.shape[1], Vt.shape[0])):
        idx = int(np.argmax(np.abs(U[:, i])))
        ref = U[idx, i]
        if abs(ref) <= 0.0:
            continue
        if np.iscomplexobj(U) or np.iscomplexobj(Vt):
            phase = np.conj(ref) / abs(ref)
            U[:, i] *= phase
            Vt[i, :] *= np.conj(phase)
        elif ref < 0:
            U[:, i] *= -1.0
            Vt[i, :] *= -1.0
    return U, S, Vt


def _abelian_canonicalize_basis_phases(basis):
    for i in range(basis.shape[1]):
        _abelian_fix_vector_phase_inplace(basis[:, i])
    return basis


def _abelian_canonicalize_density_basis(basis, strengths, rtol=1.0e-10, atol=1.0e-12):
    if len(strengths) == 0:
        return basis
    for start, stop in _abelian_cluster_sorted_values(strengths, rtol=rtol, atol=atol):
        if stop - start <= 1:
            continue
        rotation = _abelian_canonical_subspace_rotation(basis[:, start:stop])
        basis[:, start:stop] = basis[:, start:stop] @ rotation
    return _abelian_canonicalize_basis_phases(basis)


@dataclass(frozen=True)
class AbelianTwoSiteSVDResult:
    u_data: dict
    v_data: dict
    s_data: dict
    bond_qns: list
    truncation_error: float
    kept_states: int


def _pack_two_site_svd_integer_sector_ids(data):
    sector_by_token = {}
    entries = []
    for qn_tuple, block in (data or {}).items():
        qn_tuple = tuple(qn_tuple)
        if len(qn_tuple) != 4:
            raise ValueError("two-site SVD expects rank-4 permuted block keys")
        q_mid = qn_tuple[0] + qn_tuple[1]
        sectors = (qn_tuple[0], qn_tuple[1], qn_tuple[2], qn_tuple[3], q_mid)
        tokens = tuple(_abelian_stable_sector_sort_key(sector) for sector in sectors)
        for token, sector in zip(tokens, sectors):
            sector_by_token.setdefault(token, sector)
        entries.append((tokens, block))
    token_to_id = {
        token: idx
        for idx, token in enumerate(sorted(sector_by_token))
    }
    id_to_sector = {
        idx: sector_by_token[token]
        for token, idx in token_to_id.items()
    }
    packed = OrderedDict()
    for tokens, block in entries:
        packed[tuple(token_to_id[token] for token in tokens)] = np.asarray(block)
    return packed, id_to_sector


def _decode_two_site_svd_integer_sector_ids(result, id_to_sector):
    if len(result) == 7:
        u_data, v_data, s_data, bond_qns, trunc, kept, native_stats = result
    else:
        u_data, v_data, s_data, bond_qns, trunc, kept = result
        native_stats = {}
    decoded_u = OrderedDict(
        (
            (
                id_to_sector[int(key[0])],
                id_to_sector[int(key[1])],
                id_to_sector[int(key[2])],
            ),
            np.asarray(block),
        )
        for key, block in u_data.items()
    )
    decoded_v = OrderedDict(
        (
            (
                id_to_sector[int(key[0])],
                id_to_sector[int(key[1])],
                id_to_sector[int(key[2])],
            ),
            np.asarray(block),
        )
        for key, block in v_data.items()
    )
    decoded_s = OrderedDict(
        (id_to_sector[int(key)], np.asarray(block))
        for key, block in s_data.items()
    )
    decoded_bond_qns = [id_to_sector[int(qn)] for qn in bond_qns]
    return (
        decoded_u,
        decoded_v,
        decoded_s,
        decoded_bond_qns,
        float(trunc),
        int(kept),
        native_stats,
    )


def _pack_two_site_split_integer_sector_ids(data):
    sector_by_token = {}
    entries = []
    for qn_tuple, block in (data or {}).items():
        qn_tuple = tuple(qn_tuple)
        if len(qn_tuple) != 4:
            raise ValueError("two-site split expects rank-4 block keys")
        q_mid = qn_tuple[0] + qn_tuple[2]
        sectors = (qn_tuple[0], qn_tuple[1], qn_tuple[2], qn_tuple[3], q_mid)
        tokens = tuple(_abelian_stable_sector_sort_key(sector) for sector in sectors)
        for token, sector in zip(tokens, sectors):
            sector_by_token.setdefault(token, sector)
        entries.append((tokens, block))
    token_to_id = {
        token: idx
        for idx, token in enumerate(sorted(sector_by_token))
    }
    id_to_sector = {
        idx: sector_by_token[token]
        for token, idx in token_to_id.items()
    }
    packed = OrderedDict()
    for tokens, block in entries:
        packed[tuple(token_to_id[token] for token in tokens)] = np.asarray(block)
    return packed, id_to_sector


def _pack_two_site_split_layout_integer_sector_ids(layout):
    sector_by_token = {}
    entries = []
    for qn_tuple, shape in tuple(layout or ()):
        qn_tuple = tuple(qn_tuple)
        if len(qn_tuple) != 4:
            raise ValueError("two-site flat split expects rank-4 layout keys")
        q_mid = qn_tuple[0] + qn_tuple[2]
        sectors = (qn_tuple[0], qn_tuple[1], qn_tuple[2], qn_tuple[3], q_mid)
        tokens = tuple(_abelian_stable_sector_sort_key(sector) for sector in sectors)
        for token, sector in zip(tokens, sectors):
            sector_by_token.setdefault(token, sector)
        entries.append((tokens, tuple(int(dim) for dim in shape)))
    token_to_id = {
        token: idx
        for idx, token in enumerate(sorted(sector_by_token))
    }
    id_to_sector = {
        idx: sector_by_token[token]
        for token, idx in token_to_id.items()
    }
    packed_layout = tuple(
        (tuple(token_to_id[token] for token in tokens), shape)
        for tokens, shape in entries
    )
    return packed_layout, id_to_sector


def _decode_two_site_split_integer_sector_ids(result, id_to_sector):
    if len(result) == 7:
        a_data, b_data, s_data, bond_qns, trunc, kept, native_stats = result
    else:
        a_data, b_data, s_data, bond_qns, trunc, kept = result
        native_stats = {}
    decoded_a = OrderedDict(
        (
            (
                id_to_sector[int(key[0])],
                id_to_sector[int(key[1])],
                id_to_sector[int(key[2])],
            ),
            np.asarray(block),
        )
        for key, block in a_data.items()
    )
    decoded_b = OrderedDict(
        (
            (
                id_to_sector[int(key[0])],
                id_to_sector[int(key[1])],
                id_to_sector[int(key[2])],
            ),
            np.asarray(block),
        )
        for key, block in b_data.items()
    )
    decoded_s = OrderedDict(
        (id_to_sector[int(key)], np.asarray(block))
        for key, block in s_data.items()
    )
    decoded_bond_qns = [id_to_sector[int(qn)] for qn in bond_qns]
    return (
        decoded_a,
        decoded_b,
        decoded_s,
        decoded_bond_qns,
        float(trunc),
        int(kept),
        native_stats,
    )


def _pack_adjacent_site_merge_integer_sector_ids(left_data, right_data):
    sector_by_token = {}
    left_entries = []
    right_entries = []
    for entries, data in ((left_entries, left_data), (right_entries, right_data)):
        for qn_tuple, block in (data or {}).items():
            qn_tuple = tuple(qn_tuple)
            if len(qn_tuple) != 3:
                raise ValueError("adjacent site merge expects rank-3 block keys")
            tokens = tuple(
                _abelian_stable_sector_sort_key(sector)
                for sector in qn_tuple
            )
            for token, sector in zip(tokens, qn_tuple):
                sector_by_token.setdefault(token, sector)
            entries.append((tokens, block))
    token_to_id = {
        token: idx
        for idx, token in enumerate(sorted(sector_by_token))
    }
    id_to_sector = {
        idx: sector_by_token[token]
        for token, idx in token_to_id.items()
    }
    packed_left = OrderedDict(
        (tuple(token_to_id[token] for token in tokens), np.asarray(block))
        for tokens, block in left_entries
    )
    packed_right = OrderedDict(
        (tuple(token_to_id[token] for token in tokens), np.asarray(block))
        for tokens, block in right_entries
    )
    return packed_left, packed_right, id_to_sector


def _decode_adjacent_site_merge_integer_sector_ids(result, id_to_sector):
    if len(result) == 2:
        data, native_stats = result
    else:
        data = result
        native_stats = {}
    decoded = OrderedDict(
        (
            (
                id_to_sector[int(key[0])],
                id_to_sector[int(key[1])],
                id_to_sector[int(key[2])],
                id_to_sector[int(key[3])],
            ),
            np.asarray(block),
        )
        for key, block in (data or {}).items()
    )
    return decoded, native_stats


def _decode_adjacent_site_merge_layout_integer_sector_ids(layout, id_to_sector):
    return tuple(
        (
            tuple(id_to_sector[int(qn)] for qn in tuple(key)),
            tuple(int(dim) for dim in shape),
        )
        for key, shape in tuple(layout or ())
    )


def abelian_two_site_svd_from_permuted_data(data, *, m_max=None):
    """Split permuted two-site block data ``(L, pL, R, pR)`` by SVD."""

    global _ABELIAN_SVD_KERNEL_LAST_ERROR
    native_split = _cpp_table_kernel("abelian_two_site_svd_from_permuted_data")
    if native_split is not None:
        try:
            packed_data, id_to_sector = _pack_two_site_svd_integer_sector_ids(data)
            result = native_split(packed_data, m_max)
            (
                u_data,
                v_data,
                s_data,
                bond_qns,
                trunc,
                kept,
                native_stats,
            ) = _decode_two_site_svd_integer_sector_ids(result, id_to_sector)
            _ABELIAN_SVD_KERNEL_STATS["cpp_full_split_calls"] += 1
            _ABELIAN_SVD_KERNEL_STATS["cpp_full_split_blocks"] += int(len(data or {}))
            if native_stats:
                _ABELIAN_SVD_KERNEL_STATS["cpp_full_split_sectors"] += int(
                    native_stats.get("sectors", 0)
                )
                _ABELIAN_SVD_KERNEL_STATS["cpp_full_split_rows"] += int(
                    native_stats.get("rows", 0)
                )
                _ABELIAN_SVD_KERNEL_STATS["cpp_full_split_cols"] += int(
                    native_stats.get("cols", 0)
                )
            return AbelianTwoSiteSVDResult(
                u_data,
                v_data,
                s_data,
                bond_qns,
                float(trunc),
                int(kept),
            )
        except Exception as exc:
            _ABELIAN_SVD_KERNEL_STATS["cpp_full_split_failures"] += 1
            _ABELIAN_SVD_KERNEL_LAST_ERROR = repr(exc)
    return _abelian_two_site_svd_from_permuted_data_python(data, m_max=m_max)


def _abelian_two_site_svd_from_permuted_data_python(data, *, m_max=None):
    """Split permuted two-site block data ``(L, pL, R, pR)`` by SVD."""

    blocks_by_q_mid = {}
    row_map = {}
    col_map = {}
    for qn_tuple, block in (data or {}).items():
        qn_tuple = tuple(qn_tuple)
        if len(qn_tuple) != 4 or getattr(block, "ndim", None) != 4:
            raise ValueError("two-site SVD expects rank-4 permuted block data")
        q_left, q_phys_left, q_right, q_phys_right = qn_tuple
        q_mid = q_left + q_phys_left
        blocks_by_q_mid.setdefault(q_mid, [])
        row_map.setdefault(q_mid, set())
        col_map.setdefault(q_mid, set())
        blocks_by_q_mid[q_mid].append((qn_tuple, np.asarray(block)))
        row_map[q_mid].add((q_left, q_phys_left))
        col_map[q_mid].add((q_right, q_phys_right))

    sv_list = []
    u_store = {}
    v_store = {}
    s_store = {}
    for q_mid in sorted(blocks_by_q_mid, key=_abelian_stable_sector_sort_key):
        entries = blocks_by_q_mid[q_mid]
        rows = sorted(row_map[q_mid], key=_abelian_stable_sector_sort_key)
        cols = sorted(col_map[q_mid], key=_abelian_stable_sector_sort_key)
        r_starts = {}
        c_starts = {}
        r_dim = 0
        c_dim = 0
        for row in rows:
            for qn, block in entries:
                if (qn[0], qn[1]) == row:
                    r_starts[row] = r_dim
                    r_dim += int(block.shape[0]) * int(block.shape[1])
                    break
        for col in cols:
            for qn, block in entries:
                if (qn[2], qn[3]) == col:
                    c_starts[col] = c_dim
                    c_dim += int(block.shape[2]) * int(block.shape[3])
                    break
        matrix = np.zeros((r_dim, c_dim), dtype=entries[0][1].dtype)
        for qn, block in entries:
            r0 = r_starts[(qn[0], qn[1])]
            c0 = c_starts[(qn[2], qn[3])]
            left_dim = int(block.shape[0]) * int(block.shape[1])
            right_dim = int(block.shape[2]) * int(block.shape[3])
            matrix[r0:r0 + left_dim, c0:c0 + right_dim] = block.reshape(
                left_dim,
                right_dim,
            )
        svd_kernel = _cpp_table_kernel("lapack_svd")
        if svd_kernel is not None:
            try:
                U, S, Vt = svd_kernel(matrix)
                U = np.asarray(U)
                S = np.asarray(S)
                Vt = np.asarray(Vt)
                _ABELIAN_SVD_KERNEL_STATS["cpp_lapack_calls"] += 1
                _ABELIAN_SVD_KERNEL_STATS["cpp_lapack_rows"] += int(matrix.shape[0])
                _ABELIAN_SVD_KERNEL_STATS["cpp_lapack_cols"] += int(matrix.shape[1])
            except Exception:
                _ABELIAN_SVD_KERNEL_STATS["cpp_lapack_failures"] += 1
                U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
                _ABELIAN_SVD_KERNEL_STATS["numpy_calls"] += 1
        else:
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
            _ABELIAN_SVD_KERNEL_STATS["numpy_calls"] += 1
        U, S, Vt = _abelian_canonicalize_svd_pair(U, S, Vt)
        s_store[q_mid] = S
        for i, s in enumerate(S):
            sv_list.append((s, q_mid, i))
        u_store[q_mid] = (U, rows, r_starts, entries)
        v_store[q_mid] = (Vt, cols, c_starts, entries)

    sv_list = _abelian_sort_singular_entries(sv_list)
    full_sq_norm = sum(float(np.real(s)) ** 2 for s, _q_mid, _i in sv_list)
    if m_max is not None:
        sv_list = sv_list[: int(m_max)]
    kept = {}
    for _s, q_mid, i in sv_list:
        kept.setdefault(q_mid, []).append(int(i))
    kept_sq_norm = sum(
        float(np.real(s_store[q_mid][i])) ** 2
        for q_mid, idxs in kept.items()
        for i in idxs
    )
    trunc_err = 0.0
    if full_sq_norm > 1.0e-12:
        trunc_err = 1.0 - kept_sq_norm / full_sq_norm

    final_u = {}
    final_v = {}
    final_s = {}
    bond_qns = []
    for q_mid, idxs in kept.items():
        idxs = sorted(idxs)
        U, rows, r_starts, entries = u_store[q_mid]
        Vt, cols, c_starts, entries = v_store[q_mid]
        final_s[q_mid] = np.diag(s_store[q_mid][idxs])
        bond_qns.extend([q_mid] * len(idxs))
        for row in rows:
            d1 = d2 = None
            for qn, block in entries:
                if (qn[0], qn[1]) == row:
                    d1, d2 = int(block.shape[0]), int(block.shape[1])
                    break
            if d1 is None:
                continue
            r0 = r_starts[row]
            final_u[(row[0], row[1], q_mid)] = U[
                r0:r0 + d1 * d2,
                idxs,
            ].reshape(d1, d2, len(idxs))
        for col in cols:
            d3 = d4 = None
            for qn, block in entries:
                if (qn[2], qn[3]) == col:
                    d3, d4 = int(block.shape[2]), int(block.shape[3])
                    break
            if d3 is None:
                continue
            c0 = c_starts[col]
            final_v[(q_mid, col[0], col[1])] = Vt[
                idxs,
                c0:c0 + d3 * d4,
            ].reshape(len(idxs), d3, d4)
    return AbelianTwoSiteSVDResult(
        final_u,
        final_v,
        final_s,
        bond_qns,
        float(trunc_err),
        int(sum(len(v) for v in kept.values())),
    )


def abelian_state_averaged_two_site_svd_from_permuted_data(
    data_list,
    weights,
    direction,
    *,
    m_max=None,
):
    """State-averaged two-site SVD from permuted ``(L, pL, R, pR)`` data."""

    data_list = tuple(data_list or ())
    if not data_list:
        return AbelianTwoSiteSVDResult({}, {}, {}, [], 0.0, 0)
    weights = tuple(
        float(weight)
        for weight in (() if weights is None else tuple(weights))
    )
    if len(weights) != len(data_list):
        raise ValueError("state-averaged SVD weights must match state count")
    direction = str(direction)

    blocks_by_q_mid = {}
    row_map = {}
    col_map = {}
    matrix_entries = {idx: {} for idx in range(len(data_list))}
    for state_index, data in enumerate(data_list):
        for qn_tuple, block in (data or {}).items():
            qn_tuple = tuple(qn_tuple)
            if len(qn_tuple) != 4 or getattr(block, "ndim", None) != 4:
                raise ValueError("state-averaged two-site SVD expects rank-4 data")
            q_left, q_phys_left, q_right, q_phys_right = qn_tuple
            q_mid = q_left + q_phys_left
            if q_mid not in blocks_by_q_mid:
                blocks_by_q_mid[q_mid] = []
                row_map[q_mid] = set()
                col_map[q_mid] = set()
            if state_index == 0:
                blocks_by_q_mid[q_mid].append(qn_tuple)
                row_map[q_mid].add((q_left, q_phys_left))
                col_map[q_mid].add((q_right, q_phys_right))
            matrix_entries[state_index].setdefault(q_mid, []).append(
                (qn_tuple, np.asarray(block))
            )

    sv_list = []
    u_store = {}
    v_store = {}
    s_store = {}
    for q_mid in sorted(blocks_by_q_mid, key=_abelian_stable_sector_sort_key):
        rows = sorted(row_map[q_mid], key=_abelian_stable_sector_sort_key)
        cols = sorted(col_map[q_mid], key=_abelian_stable_sector_sort_key)
        r_starts = {}
        c_starts = {}
        r_dim = 0
        c_dim = 0
        entries_0 = matrix_entries[0][q_mid]
        for row in rows:
            for qn, block in entries_0:
                if (qn[0], qn[1]) == row:
                    r_starts[row] = r_dim
                    r_dim += int(block.shape[0]) * int(block.shape[1])
                    break
        for col in cols:
            for qn, block in entries_0:
                if (qn[2], qn[3]) == col:
                    c_starts[col] = c_dim
                    c_dim += int(block.shape[2]) * int(block.shape[3])
                    break

        matrices = []
        for state_index in range(len(data_list)):
            matrix = np.zeros((r_dim, c_dim), dtype=entries_0[0][1].dtype)
            for qn, block in matrix_entries[state_index][q_mid]:
                r0 = r_starts[(qn[0], qn[1])]
                c0 = c_starts[(qn[2], qn[3])]
                left_dim = int(block.shape[0]) * int(block.shape[1])
                right_dim = int(block.shape[2]) * int(block.shape[3])
                matrix[r0:r0 + left_dim, c0:c0 + right_dim] = block.reshape(
                    left_dim,
                    right_dim,
                )
            matrices.append(matrix)

        if direction == "right":
            rho = np.zeros((r_dim, r_dim), dtype=matrices[0].dtype)
            for state_index, matrix in enumerate(matrices):
                rho += weights[state_index] * (matrix @ matrix.conj().T)
            S2, U = np.linalg.eigh(rho)
            order = np.argsort(-S2, kind="mergesort")
            S2, U = S2[order], U[:, order]
            S = np.sqrt(np.clip(S2, 0.0, None))
            U = _abelian_canonicalize_density_basis(U, S)
            S_inv = np.zeros_like(S)
            S_inv[S > 1.0e-12] = 1.0 / S[S > 1.0e-12]
            Vt = np.diag(S_inv) @ U.conj().T @ matrices[0]
        else:
            rho = np.zeros((c_dim, c_dim), dtype=matrices[0].dtype)
            for state_index, matrix in enumerate(matrices):
                rho += weights[state_index] * (matrix.conj().T @ matrix)
            S2, V = np.linalg.eigh(rho)
            order = np.argsort(-S2, kind="mergesort")
            S2, V = S2[order], V[:, order]
            S = np.sqrt(np.clip(S2, 0.0, None))
            V = _abelian_canonicalize_density_basis(V, S)
            Vt = V.conj().T
            S_inv = np.zeros_like(S)
            S_inv[S > 1.0e-12] = 1.0 / S[S > 1.0e-12]
            U = matrices[0] @ V @ np.diag(S_inv)

        s_store[q_mid] = S
        for i, s in enumerate(S):
            sv_list.append((s, q_mid, i))
        u_store[q_mid] = (U, rows, r_starts, entries_0)
        v_store[q_mid] = (Vt, cols, c_starts, entries_0)

    sv_list = _abelian_sort_singular_entries(sv_list)
    if m_max is not None:
        sv_list = sv_list[: int(m_max)]
    kept = {}
    for _s, q_mid, i in sv_list:
        kept.setdefault(q_mid, []).append(int(i))

    final_u = {}
    final_v = {}
    final_s = {}
    bond_qns = []
    for q_mid, idxs in kept.items():
        idxs = sorted(idxs)
        U, rows, r_starts, entries_0 = u_store[q_mid]
        Vt, cols, c_starts, entries_0 = v_store[q_mid]
        final_s[q_mid] = np.diag(s_store[q_mid][idxs])
        bond_qns.extend([q_mid] * len(idxs))
        for row in rows:
            d1 = d2 = None
            for qn, block in entries_0:
                if (qn[0], qn[1]) == row:
                    d1, d2 = int(block.shape[0]), int(block.shape[1])
                    break
            if d1 is None:
                continue
            r0 = r_starts[row]
            final_u[(row[0], row[1], q_mid)] = U[
                r0:r0 + d1 * d2,
                idxs,
            ].reshape(d1, d2, len(idxs))
        for col in cols:
            d3 = d4 = None
            for qn, block in entries_0:
                if (qn[2], qn[3]) == col:
                    d3, d4 = int(block.shape[2]), int(block.shape[3])
                    break
            if d3 is None:
                continue
            c0 = c_starts[col]
            final_v[(q_mid, col[0], col[1])] = Vt[
                idxs,
                c0:c0 + d3 * d4,
            ].reshape(len(idxs), d3, d4)
    return AbelianTwoSiteSVDResult(
        final_u,
        final_v,
        final_s,
        bond_qns,
        0.0,
        int(sum(len(v) for v in kept.values())),
    )


def abelian_multiply_u_s_data(u_data, s_data):
    """Contract diagonal/block singular values into U block data."""

    out = {}
    for key, block in (u_data or {}).items():
        key = tuple(key)
        if len(key) != 3:
            continue
        q_mid = key[2]
        singular = (s_data or {}).get(q_mid)
        if singular is None:
            continue
        out[key] = np.tensordot(block, singular, axes=([2], [0]))
    return out


def abelian_multiply_s_v_data(s_data, v_data):
    """Contract diagonal/block singular values into V block data."""

    out = {}
    for key, block in (v_data or {}).items():
        key = tuple(key)
        if len(key) != 3:
            continue
        q_mid = key[0]
        singular = (s_data or {}).get(q_mid)
        if singular is None:
            continue
        out[key] = np.tensordot(singular, block, axes=([1], [0]))
    return out


@dataclass(frozen=True)
class AbelianTwoSiteSplitResult:
    a_data: dict
    b_data: dict
    a_qns: list
    b_qns: list
    a_dirs: list
    b_dirs: list
    s_data: dict
    bond_qns: list
    truncation_error: float
    kept_states: int


def _abelian_block_data_result_dtype(data, *extra):
    dtypes = []
    for block in (data or {}).values():
        try:
            dtype = np.asarray(block).dtype
        except Exception:
            return np.dtype(object)
        dtypes.append(dtype)
    for value in extra:
        try:
            dtypes.append(np.asarray(value).dtype)
        except Exception:
            return np.dtype(object)
    return np.result_type(*(dtypes or [complex]))


class AbelianSiteTensorData:
    """Native Abelian site tensor carrier backed by plain block data."""

    __slots__ = ("data", "qns", "dirs", "_layout_signature")

    _pyqed_abelian_site_tensor_data = True

    def __init__(self, data, qns, dirs, *, copy=True):
        self.data = OrderedDict(
            (
                tuple(key),
                np.array(block, copy=bool(copy)),
            )
            for key, block in (data or {}).items()
        )
        self.qns = tuple(tuple(axis_qns) for axis_qns in (qns or ()))
        self.dirs = tuple(int(d) for d in (dirs or ()))
        self._layout_signature = None

    @property
    def rank(self):
        return len(self.dirs)

    @property
    def shape(self):
        dims = []
        for axis in range(self.rank):
            total = 0
            seen = set()
            for key, block in self.data.items():
                if axis >= len(key) or key[axis] in seen:
                    continue
                total += int(np.asarray(block).shape[axis])
                seen.add(key[axis])
            dims.append(total)
        return tuple(dims)

    def copy(self):
        return type(self)(self.data, self.qns, self.dirs, copy=True)

    def _binary_op(self, other, sign=1.0):
        other_data = getattr(other, "data", None)
        if other_data is None:
            return NotImplemented
        out = OrderedDict((key, np.asarray(block).copy()) for key, block in self.data.items())
        for key, block in (other_data or {}).items():
            key = tuple(key)
            contrib = np.asarray(block) * sign
            old = out.get(key)
            out[key] = contrib.copy() if old is None else old + contrib
        return type(self)(out, self.qns, self.dirs, copy=False)

    def __add__(self, other):
        return self._binary_op(other, sign=1.0)

    def __sub__(self, other):
        return self._binary_op(other, sign=-1.0)

    def __mul__(self, scalar):
        return self.scaled(scalar)

    def __rmul__(self, scalar):
        return self.scaled(scalar)

    def __truediv__(self, scalar):
        return self.scaled(1.0 / scalar)

    def conj(self):
        return type(self)(
            {key: np.asarray(block).conj() for key, block in self.data.items()},
            self.qns,
            self.dirs,
            copy=False,
        )

    def scaled(self, scalar):
        global _ABELIAN_SVD_KERNEL_LAST_ERROR

        native_scale = _cpp_table_kernel("abelian_scale_block_data")
        if native_scale is not None:
            try:
                dtype = np.dtype(_abelian_block_data_result_dtype(self.data, scalar))
                if dtype.kind == "c":
                    out = native_scale(self.data, complex(scalar))
                    _ABELIAN_SVD_KERNEL_STATS["cpp_block_scale_calls"] += 1
                    _ABELIAN_SVD_KERNEL_STATS["cpp_block_scale_blocks"] += int(
                        len(self.data)
                    )
                    return type(self)(out, self.qns, self.dirs, copy=False)
            except Exception as exc:
                _ABELIAN_SVD_KERNEL_STATS["cpp_block_scale_failures"] += 1
                _ABELIAN_SVD_KERNEL_LAST_ERROR = repr(exc)
        return type(self)(
            {key: np.asarray(block) * scalar for key, block in self.data.items()},
            self.qns,
            self.dirs,
            copy=False,
        )

    def norm(self):
        global _ABELIAN_SVD_KERNEL_LAST_ERROR

        native_norm = _cpp_table_kernel("abelian_block_data_norm")
        if native_norm is not None:
            try:
                norm = float(native_norm(self.data))
                _ABELIAN_SVD_KERNEL_STATS["cpp_block_norm_calls"] += 1
                _ABELIAN_SVD_KERNEL_STATS["cpp_block_norm_blocks"] += int(len(self.data))
                return norm
            except Exception as exc:
                _ABELIAN_SVD_KERNEL_STATS["cpp_block_norm_failures"] += 1
                _ABELIAN_SVD_KERNEL_LAST_ERROR = repr(exc)
        total = 0.0
        for block in self.data.values():
            arr = np.asarray(block)
            total += float(np.vdot(arr.reshape(-1), arr.reshape(-1)).real)
        return float(np.sqrt(total))

    def dot(self, other):
        total = 0.0
        other_data = getattr(other, "data", {}) or {}
        for key, block in self.data.items():
            if key in other_data:
                total += np.vdot(block, other_data[key])
        return total

    def block_layout(self):
        return tuple(
            (key, tuple(int(dim) for dim in np.asarray(block).shape))
            for key, block in self.data.items()
        )

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
            axes = axes[0]
        return abelian_transpose_tensor_data(
            self,
            axes,
            carrier=type(self),
        )


class AbelianEnvironmentTensorData(AbelianSiteTensorData):
    """Native Abelian environment tensor carrier backed by plain block data."""

    _pyqed_abelian_environment_tensor_data = True


def is_abelian_site_tensor_data(tensor):
    return bool(getattr(tensor, "_pyqed_abelian_site_tensor_data", False))


def is_abelian_environment_tensor_data(tensor):
    return bool(getattr(tensor, "_pyqed_abelian_environment_tensor_data", False))


def _abelian_tensor_data_view(tensor, *, conj=False, carrier=AbelianSiteTensorData):
    data = getattr(tensor, "data", None)
    if data is None:
        raise TypeError("native Abelian tensor contraction requires block data")
    dirs = list(getattr(tensor, "dirs", ()))
    if conj:
        data = {
            tuple(key): np.asarray(block).conj()
            for key, block in (data or {}).items()
        }
        dirs = [-int(d) for d in dirs]
    return carrier(
        data,
        getattr(tensor, "qns", ()),
        dirs,
        copy=False,
    )


def _abelian_normalize_axes(axes):
    a_ax, b_ax = axes
    if isinstance(a_ax, int):
        a_ax = [a_ax]
    if isinstance(b_ax, int):
        b_ax = [b_ax]
    return [int(axis) for axis in a_ax], [int(axis) for axis in b_ax]


def abelian_tensor_data_tensordot(
    a_tensor,
    b_tensor,
    axes,
    *,
    carrier=AbelianSiteTensorData,
):
    """Block-data-only equivalent of the legacy Abelian ``tensordot``."""

    a = _abelian_tensor_data_view(a_tensor, carrier=AbelianSiteTensorData)
    b = _abelian_tensor_data_view(b_tensor, carrier=AbelianSiteTensorData)
    a_ax, b_ax = _abelian_normalize_axes(axes)
    free_a = [axis for axis in range(a.rank) if axis not in a_ax]
    free_b = [axis for axis in range(b.rank) if axis not in b_ax]
    out_qns = [a.qns[axis] for axis in free_a] + [b.qns[axis] for axis in free_b]
    out_dirs = [a.dirs[axis] for axis in free_a] + [b.dirs[axis] for axis in free_b]

    b_by_contract = defaultdict(list)
    for b_key, b_block in b.data.items():
        contract_key = tuple(b_key[axis] for axis in b_ax)
        b_by_contract[contract_key].append((b_key, b_block))

    out = OrderedDict()
    for a_key, a_block in a.data.items():
        contract_key = tuple(a_key[axis] for axis in a_ax)
        for b_key, b_block in b_by_contract.get(contract_key, ()):
            block = np.tensordot(a_block, b_block, axes=(a_ax, b_ax))
            out_key = tuple(a_key[axis] for axis in free_a) + tuple(
                b_key[axis] for axis in free_b
            )
            old = out.get(out_key)
            out[out_key] = block if old is None else old + block
    return carrier(out, out_qns, out_dirs, copy=False)


def abelian_transpose_tensor_data(
    tensor,
    axes,
    *,
    carrier=AbelianSiteTensorData,
):
    """Permute a native block-data tensor without a legacy tensor wrapper."""

    axes = tuple(int(axis) for axis in axes)
    view = _abelian_tensor_data_view(tensor, carrier=AbelianSiteTensorData)
    return carrier(
        OrderedDict(
            (
                tuple(key[axis] for axis in axes),
                np.asarray(block).transpose(axes),
            )
            for key, block in view.data.items()
        ),
        [view.qns[axis] for axis in axes],
        [view.dirs[axis] for axis in axes],
        copy=False,
    )


def abelian_contract_from_left_data(w_tensor, a_tensor, e_tensor, b_tensor):
    """Advance a left environment using native Abelian block data."""

    cpp_payload = _cpp_table_kernel("abelian_left_environment_advance_data")
    if cpp_payload is not None:
        try:
            keys, blocks, qns, dirs = cpp_payload(
                w_tensor,
                a_tensor,
                e_tensor,
                b_tensor,
            )
            data = OrderedDict(
                (tuple(key), np.asarray(block))
                for key, block in zip(keys, blocks)
            )
            _ABELIAN_ENVIRONMENT_ADVANCE_PAYLOAD_STATS["left"] += 1
            return AbelianEnvironmentTensorData(
                data,
                qns,
                dirs,
                copy=False,
            )
        except Exception:
            _ABELIAN_ENVIRONMENT_ADVANCE_PAYLOAD_STATS["left_failures"] += 1

    a_conj = _abelian_tensor_data_view(
        a_tensor,
        conj=True,
        carrier=AbelianSiteTensorData,
    )
    temp = abelian_tensor_data_tensordot(e_tensor, a_conj, ([1], [0]))
    temp = abelian_tensor_data_tensordot(temp, w_tensor, ([0, 3], [0, 2]))
    temp = abelian_tensor_data_tensordot(temp, b_tensor, ([0, 3], [0, 2]))
    return abelian_transpose_tensor_data(
        temp,
        (1, 0, 2),
        carrier=AbelianEnvironmentTensorData,
    )


def abelian_contract_from_right_data(w_tensor, a_tensor, f_tensor, b_tensor):
    """Advance a right environment using native Abelian block data."""

    cpp_payload = _cpp_table_kernel("abelian_right_environment_advance_data")
    if cpp_payload is not None:
        try:
            keys, blocks, qns, dirs = cpp_payload(
                w_tensor,
                a_tensor,
                f_tensor,
                b_tensor,
            )
            data = OrderedDict(
                (tuple(key), np.asarray(block))
                for key, block in zip(keys, blocks)
            )
            _ABELIAN_ENVIRONMENT_ADVANCE_PAYLOAD_STATS["right"] += 1
            return AbelianEnvironmentTensorData(
                data,
                qns,
                dirs,
                copy=False,
            )
        except Exception:
            _ABELIAN_ENVIRONMENT_ADVANCE_PAYLOAD_STATS["right_failures"] += 1

    a_conj = _abelian_tensor_data_view(
        a_tensor,
        conj=True,
        carrier=AbelianSiteTensorData,
    )
    temp = abelian_tensor_data_tensordot(a_conj, f_tensor, ([1], [1]))
    temp = abelian_tensor_data_tensordot(temp, w_tensor, ([2, 1], [1, 2]))
    temp = abelian_tensor_data_tensordot(temp, b_tensor, ([1, 3], [1, 2]))
    return abelian_transpose_tensor_data(
        temp,
        (1, 0, 2),
        carrier=AbelianEnvironmentTensorData,
    )


def abelian_merge_adjacent_site_tensors(left, right):
    """Merge adjacent native site tensors into two-site ``(L, R, pL, pR)`` data."""

    global _ABELIAN_SVD_KERNEL_LAST_ERROR
    native_merge = _cpp_table_kernel("abelian_merge_adjacent_site_tensors_data")
    if (
        native_merge is not None
        and is_abelian_site_tensor_data(left)
        and is_abelian_site_tensor_data(right)
    ):
        try:
            if int(getattr(left, "rank", 0)) != 3 or int(getattr(right, "rank", 0)) != 3:
                raise ValueError("adjacent site merge expects rank-3 site tensors")
            packed_left, packed_right, id_to_sector = _pack_adjacent_site_merge_integer_sector_ids(
                left.data,
                right.data,
            )
            result = native_merge(packed_left, packed_right)
            data, native_stats = _decode_adjacent_site_merge_integer_sector_ids(
                result,
                id_to_sector,
            )
            _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_calls"] += 1
            if native_stats:
                _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_left_blocks"] += int(
                    native_stats.get("left_blocks", 0)
                )
                _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_right_blocks"] += int(
                    native_stats.get("right_blocks", 0)
                )
                _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_pairs"] += int(
                    native_stats.get("pairs", 0)
                )
                _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_blocks"] += int(
                    native_stats.get("out_blocks", 0)
                )
                _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_inner_dim"] += int(
                    native_stats.get("inner_dim", 0)
                )
            return AbelianSiteTensorData(
                data,
                [left.qns[0], right.qns[1], left.qns[2], right.qns[2]],
                [left.dirs[0], right.dirs[1], left.dirs[2], right.dirs[2]],
                copy=False,
            )
        except Exception as exc:
            _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_failures"] += 1
            _ABELIAN_SVD_KERNEL_LAST_ERROR = repr(exc)
    return _abelian_merge_adjacent_site_tensors_python(left, right)


def abelian_merge_normalize_adjacent_site_tensors(left, right):
    """Merge adjacent native site tensors and normalize the two-site data."""

    global _ABELIAN_SVD_KERNEL_LAST_ERROR
    native_merge = _cpp_table_kernel("abelian_merge_normalize_adjacent_site_tensors_data")
    if (
        native_merge is not None
        and is_abelian_site_tensor_data(left)
        and is_abelian_site_tensor_data(right)
    ):
        try:
            if int(getattr(left, "rank", 0)) != 3 or int(getattr(right, "rank", 0)) != 3:
                raise ValueError("adjacent site merge expects rank-3 site tensors")
            packed_left, packed_right, id_to_sector = _pack_adjacent_site_merge_integer_sector_ids(
                left.data,
                right.data,
            )
            result = native_merge(packed_left, packed_right)
            data, norm, native_stats = result
            decoded, native_stats = _decode_adjacent_site_merge_integer_sector_ids(
                (data, native_stats),
                id_to_sector,
            )
            _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_normalize_calls"] += 1
            if native_stats:
                _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_normalize_pairs"] += int(
                    native_stats.get("pairs", 0)
                )
                _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_normalize_blocks"] += int(
                    native_stats.get("out_blocks", 0)
                )
                _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_normalize_scaled_blocks"] += int(
                    native_stats.get("scaled_blocks", 0)
                )
            return (
                AbelianSiteTensorData(
                    decoded,
                    [left.qns[0], right.qns[1], left.qns[2], right.qns[2]],
                    [left.dirs[0], right.dirs[1], left.dirs[2], right.dirs[2]],
                    copy=False,
                ),
                float(norm),
            )
        except Exception as exc:
            _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_normalize_failures"] += 1
            _ABELIAN_SVD_KERNEL_LAST_ERROR = repr(exc)

    merged = abelian_merge_adjacent_site_tensors(left, right)
    norm = merged.norm()
    if norm > 0.0:
        merged = merged * (1.0 / norm)
    return merged, float(norm)


def abelian_merge_normalize_flatten_adjacent_site_tensors(left, right):
    """Merge, normalize, and flatten adjacent native site tensors."""

    global _ABELIAN_SVD_KERNEL_LAST_ERROR
    native_merge = _cpp_table_kernel(
        "abelian_merge_normalize_flatten_adjacent_site_tensors_data"
    )
    if (
        native_merge is not None
        and is_abelian_site_tensor_data(left)
        and is_abelian_site_tensor_data(right)
    ):
        try:
            if int(getattr(left, "rank", 0)) != 3 or int(getattr(right, "rank", 0)) != 3:
                raise ValueError("adjacent site merge expects rank-3 site tensors")
            packed_left, packed_right, id_to_sector = _pack_adjacent_site_merge_integer_sector_ids(
                left.data,
                right.data,
            )
            data, norm, packed_layout, flat, native_stats = native_merge(
                packed_left,
                packed_right,
            )
            decoded, native_stats = _decode_adjacent_site_merge_integer_sector_ids(
                (data, native_stats),
                id_to_sector,
            )
            layout = _decode_adjacent_site_merge_layout_integer_sector_ids(
                packed_layout,
                id_to_sector,
            )
            flat = np.asarray(flat)
            _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_normalize_flatten_calls"] += 1
            if native_stats:
                _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_normalize_flatten_pairs"] += int(
                    native_stats.get("pairs", 0)
                )
                _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_normalize_flatten_blocks"] += int(
                    native_stats.get("out_blocks", 0)
                )
                _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_normalize_flatten_dim"] += int(
                    native_stats.get("flat_dim", flat.size)
                )
            return (
                AbelianSiteTensorData(
                    decoded,
                    [left.qns[0], right.qns[1], left.qns[2], right.qns[2]],
                    [left.dirs[0], right.dirs[1], left.dirs[2], right.dirs[2]],
                    copy=False,
                ),
                float(norm),
                flat,
                layout,
            )
        except Exception as exc:
            _ABELIAN_SVD_KERNEL_STATS["cpp_site_merge_normalize_flatten_failures"] += 1
            _ABELIAN_SVD_KERNEL_LAST_ERROR = repr(exc)

    merged, norm = abelian_merge_normalize_adjacent_site_tensors(left, right)
    layout = abelian_local_layout_from_tensor(merged)
    flat = AbelianLocalVectorLayout.from_layout(
        layout,
        proto=merged,
    ).flatten_tensor(merged)
    return merged, float(norm), flat, layout


def _abelian_merge_adjacent_site_tensors_python(left, right):
    """Reference Python adjacent-site merge for native site tensors."""

    merged_lprp = abelian_tensor_data_tensordot(
        left,
        right,
        ([1], [0]),
        carrier=AbelianSiteTensorData,
    )
    return abelian_transpose_tensor_data(
        merged_lprp,
        (0, 2, 1, 3),
        carrier=AbelianSiteTensorData,
    )


def abelian_right_canonicalize_site_tensors(factors, *, max_bond_dim=None):
    """Put native Abelian site tensors in right-canonical sweep form."""

    if not factors:
        return factors
    if not is_abelian_site_tensor_data(factors[0]):
        return factors
    out = [
        factor.copy() if hasattr(factor, "copy") else factor
        for factor in factors
    ]
    for site in range(len(out) - 1, 0, -1):
        aa = abelian_merge_adjacent_site_tensors(out[site - 1], out[site])
        split = abelian_split_two_site_svd_data(
            aa.data,
            qns=aa.qns,
            dirs=aa.dirs,
            direction="left",
            m_max=max_bond_dim,
        )
        update = abelian_site_tensors_from_split(split)
        out[site - 1] = update.left
        out[site] = update.right
    return out


@dataclass(frozen=True)
class AbelianTwoSiteUpdateData:
    """Native post-SVD two-site update before any legacy tensor wrapping."""

    left: AbelianSiteTensorData
    right: AbelianSiteTensorData
    s_data: dict
    bond_qns: tuple
    truncation_error: float
    kept_states: int


def abelian_site_tensors_from_split(split, *, copy=False):
    """Convert a split result into native left/right site tensor carriers."""

    copy = bool(copy)
    left_blocks = int(len(getattr(split, "a_data", {}) or {}))
    right_blocks = int(len(getattr(split, "b_data", {}) or {}))
    _ABELIAN_SVD_KERNEL_STATS["site_update_wrap_calls"] += 1
    _ABELIAN_SVD_KERNEL_STATS["site_update_wrap_blocks"] += left_blocks + right_blocks
    if not copy:
        _ABELIAN_SVD_KERNEL_STATS["site_update_wrap_nocopy_blocks"] += (
            left_blocks + right_blocks
        )
    return AbelianTwoSiteUpdateData(
        AbelianSiteTensorData(split.a_data, split.a_qns, split.a_dirs, copy=copy),
        AbelianSiteTensorData(split.b_data, split.b_qns, split.b_dirs, copy=copy),
        OrderedDict(
            (key, np.array(block, copy=copy))
            for key, block in (split.s_data or {}).items()
        ),
        tuple(split.bond_qns or ()),
        float(split.truncation_error),
        int(split.kept_states),
    )


def abelian_split_two_site_svd_data(
    data,
    *,
    qns=None,
    dirs=None,
    direction="right",
    m_max=None,
):
    """Split two-site data ``(L, R, pL, pR)`` into MPS tensors."""

    global _ABELIAN_SVD_KERNEL_LAST_ERROR
    native_split = _cpp_table_kernel("abelian_split_two_site_svd_data")
    if native_split is not None:
        try:
            packed_data, id_to_sector = _pack_two_site_split_integer_sector_ids(data)
            result = native_split(packed_data, str(direction), m_max)
            (
                a_data,
                b_data,
                s_data,
                bond_qns,
                trunc,
                kept,
                native_stats,
            ) = _decode_two_site_split_integer_sector_ids(result, id_to_sector)
            _ABELIAN_SVD_KERNEL_STATS["cpp_split_update_calls"] += 1
            _ABELIAN_SVD_KERNEL_STATS["cpp_split_update_blocks"] += int(len(data or {}))
            if native_stats:
                _ABELIAN_SVD_KERNEL_STATS["cpp_split_update_sectors"] += int(
                    native_stats.get("sectors", 0)
                )
                _ABELIAN_SVD_KERNEL_STATS["cpp_split_update_rows"] += int(
                    native_stats.get("rows", 0)
                )
                _ABELIAN_SVD_KERNEL_STATS["cpp_split_update_cols"] += int(
                    native_stats.get("cols", 0)
                )
            qns = [list(axis_qns) for axis_qns in (qns or ([], [], [], []))]
            dirs = list(dirs or ([-1, 1, 1, 1]))
            if len(qns) != 4:
                qns = (qns + [list(bond_qns)] * 4)[:4]
            if len(dirs) != 4:
                dirs = (dirs + [1] * 4)[:4]
            return AbelianTwoSiteSplitResult(
                a_data,
                b_data,
                [list(qns[0]), list(bond_qns), list(qns[2])],
                [list(bond_qns), list(qns[1]), list(qns[3])],
                [int(dirs[0]), 1, int(dirs[2])],
                [-1, int(dirs[1]), int(dirs[3])],
                s_data,
                list(bond_qns),
                float(trunc),
                int(kept),
            )
        except Exception as exc:
            _ABELIAN_SVD_KERNEL_STATS["cpp_split_update_failures"] += 1
            _ABELIAN_SVD_KERNEL_LAST_ERROR = repr(exc)
    return _abelian_split_two_site_svd_data_python(
        data,
        qns=qns,
        dirs=dirs,
        direction=direction,
        m_max=m_max,
    )


def _abelian_split_two_site_svd_data_python(
    data,
    *,
    qns=None,
    dirs=None,
    direction="right",
    m_max=None,
):
    """Split two-site data ``(L, R, pL, pR)`` into MPS tensors."""

    permuted = {}
    for key, block in (data or {}).items():
        key = tuple(key)
        if len(key) != 4 or getattr(block, "ndim", None) != 4:
            raise ValueError("two-site split expects rank-4 block data")
        permuted[(key[0], key[2], key[1], key[3])] = np.asarray(block).transpose(
            0,
            2,
            1,
            3,
        )
    svd = _abelian_two_site_svd_from_permuted_data_python(permuted, m_max=m_max)
    direction = str(direction)
    if direction == "right":
        a_source = svd.u_data
        b_data = abelian_multiply_s_v_data(svd.s_data, svd.v_data)
    else:
        a_source = abelian_multiply_u_s_data(svd.u_data, svd.s_data)
        b_data = dict(svd.v_data)
    a_data = {
        (key[0], key[2], key[1]): np.asarray(block).transpose(0, 2, 1)
        for key, block in a_source.items()
    }

    qns = [list(axis_qns) for axis_qns in (qns or ([], [], [], []))]
    dirs = list(dirs or ([-1, 1, 1, 1]))
    if len(qns) != 4:
        qns = (qns + [list(svd.bond_qns)] * 4)[:4]
    if len(dirs) != 4:
        dirs = (dirs + [1] * 4)[:4]
    return AbelianTwoSiteSplitResult(
        a_data,
        b_data,
        [list(qns[0]), list(svd.bond_qns), list(qns[2])],
        [list(svd.bond_qns), list(qns[1]), list(qns[3])],
        [int(dirs[0]), 1, int(dirs[2])],
        [-1, int(dirs[1]), int(dirs[3])],
        svd.s_data,
        list(svd.bond_qns),
        float(svd.truncation_error),
        int(svd.kept_states),
    )


def _abelian_split_result_from_flat_kernel(
    native_split,
    vec,
    layout,
    *,
    qns=None,
    dirs=None,
    direction="right",
    m_max=None,
    stat_prefix="cpp_flat_split_update",
):
    packed_layout, id_to_sector = _pack_two_site_split_layout_integer_sector_ids(layout)
    result = native_split(vec, packed_layout, str(direction), m_max)
    (
        a_data,
        b_data,
        s_data,
        bond_qns,
        trunc,
        kept,
        native_stats,
    ) = _decode_two_site_split_integer_sector_ids(result, id_to_sector)
    _ABELIAN_SVD_KERNEL_STATS[f"{stat_prefix}_calls"] += 1
    _ABELIAN_SVD_KERNEL_STATS[f"{stat_prefix}_blocks"] += int(len(layout or ()))
    if native_stats:
        _ABELIAN_SVD_KERNEL_STATS[f"{stat_prefix}_sectors"] += int(
            native_stats.get("sectors", 0)
        )
        _ABELIAN_SVD_KERNEL_STATS[f"{stat_prefix}_rows"] += int(
            native_stats.get("rows", 0)
        )
        _ABELIAN_SVD_KERNEL_STATS[f"{stat_prefix}_cols"] += int(
            native_stats.get("cols", 0)
        )
        _ABELIAN_SVD_KERNEL_STATS[f"{stat_prefix}_dim"] += int(
            native_stats.get("flat_dim", 0)
        )
    local_layout = AbelianLocalVectorLayout.from_layout(
        layout,
        qns=qns,
        dirs=dirs,
    )
    qns = [list(axis_qns) for axis_qns in local_layout.qns]
    dirs = list(local_layout.dirs)
    if len(qns) != 4:
        qns = (qns + [list(bond_qns)] * 4)[:4]
    if len(dirs) != 4:
        dirs = (dirs + [1] * 4)[:4]
    return AbelianTwoSiteSplitResult(
        a_data,
        b_data,
        [list(qns[0]), list(bond_qns), list(qns[2])],
        [list(bond_qns), list(qns[1]), list(qns[3])],
        [int(dirs[0]), 1, int(dirs[2])],
        [-1, int(dirs[1]), int(dirs[3])],
        s_data,
        list(bond_qns),
        float(trunc),
        int(kept),
    )


def abelian_split_flat_two_site_svd_data_from_kernel(
    native_split,
    vec,
    layout,
    *,
    qns=None,
    dirs=None,
    direction="right",
    m_max=None,
):
    """Split flat two-site data through an explicitly supplied native kernel."""

    global _ABELIAN_SVD_KERNEL_LAST_ERROR
    try:
        return _abelian_split_result_from_flat_kernel(
            native_split,
            vec,
            layout,
            qns=qns,
            dirs=dirs,
            direction=direction,
            m_max=m_max,
        )
    except Exception as exc:
        _ABELIAN_SVD_KERNEL_STATS["cpp_flat_split_update_failures"] += 1
        _ABELIAN_SVD_KERNEL_LAST_ERROR = repr(exc)
        raise


def abelian_split_flat_two_site_svd_data(
    vec,
    layout,
    *,
    qns=None,
    dirs=None,
    direction="right",
    m_max=None,
):
    """Split a flat two-site vector without constructing a tensor wrapper."""

    global _ABELIAN_SVD_KERNEL_LAST_ERROR
    native_split = _cpp_table_kernel("abelian_split_flat_two_site_svd_data")
    if native_split is not None:
        try:
            return _abelian_split_result_from_flat_kernel(
                native_split,
                vec,
                layout,
                qns=qns,
                dirs=dirs,
                direction=direction,
                m_max=m_max,
            )
        except Exception as exc:
            _ABELIAN_SVD_KERNEL_STATS["cpp_flat_split_update_failures"] += 1
            _ABELIAN_SVD_KERNEL_LAST_ERROR = repr(exc)

    data, qns, dirs = abelian_unflatten_data_from_layout(
        vec,
        layout,
        qns=qns,
        dirs=dirs,
    )
    return abelian_split_two_site_svd_data(
        data,
        qns=qns,
        dirs=dirs,
        direction=direction,
        m_max=m_max,
    )


def abelian_split_state_averaged_two_site_svd_data(
    data_list,
    weights,
    *,
    qns=None,
    dirs=None,
    direction="right",
    m_max=None,
):
    """State-averaged split of ``(L, R, pL, pR)`` data into MPS tensors."""

    permuted_list = []
    for data in tuple(data_list or ()):
        permuted = {}
        for key, block in (data or {}).items():
            key = tuple(key)
            if len(key) != 4 or getattr(block, "ndim", None) != 4:
                raise ValueError("state-averaged two-site split expects rank-4 data")
            permuted[(key[0], key[2], key[1], key[3])] = np.asarray(block).transpose(
                0,
                2,
                1,
                3,
            )
        permuted_list.append(permuted)
    svd = abelian_state_averaged_two_site_svd_from_permuted_data(
        permuted_list,
        weights,
        direction,
        m_max=m_max,
    )
    direction = str(direction)
    if direction == "right":
        a_source = svd.u_data
        b_data = abelian_multiply_s_v_data(svd.s_data, svd.v_data)
    else:
        a_source = abelian_multiply_u_s_data(svd.u_data, svd.s_data)
        b_data = dict(svd.v_data)
    a_data = {
        (key[0], key[2], key[1]): np.asarray(block).transpose(0, 2, 1)
        for key, block in a_source.items()
    }

    qns = [list(axis_qns) for axis_qns in (qns or ([], [], [], []))]
    dirs = list(dirs or ([-1, 1, 1, 1]))
    if len(qns) != 4:
        qns = (qns + [list(svd.bond_qns)] * 4)[:4]
    if len(dirs) != 4:
        dirs = (dirs + [1] * 4)[:4]
    return AbelianTwoSiteSplitResult(
        a_data,
        b_data,
        [list(qns[0]), list(svd.bond_qns), list(qns[2])],
        [list(svd.bond_qns), list(qns[1]), list(qns[3])],
        [int(dirs[0]), 1, int(dirs[2])],
        [-1, int(dirs[1]), int(dirs[3])],
        svd.s_data,
        list(svd.bond_qns),
        float(svd.truncation_error),
        int(svd.kept_states),
    )


def abelian_orthogonalize_candidate(candidate, basis, *, passes=2, min_norm=1.0e-9):
    q = np.asarray(candidate).copy()
    if not np.all(np.isfinite(q)):
        return None, float("nan")
    basis = tuple(np.asarray(vec) for vec in tuple(basis or ()))
    for _ in range(int(passes)):
        for vec in basis:
            q = q - vec * np.vdot(vec, q)
            if not np.all(np.isfinite(q)):
                return None, float("nan")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < float(min_norm):
        return None, norm
    return q / norm, norm


def abelian_remap_flat_layout(vec, old_layout, new_layout):
    """Move a flat vector from one compatible Abelian block layout to another."""

    old_offsets, _old_dim = AbelianLocalVectorLayout.from_layout(old_layout).offsets
    new_offsets, new_dim = AbelianLocalVectorLayout.from_layout(new_layout).offsets
    out = np.zeros(int(new_dim), dtype=np.asarray(vec).dtype)
    for key, (old_pos, n) in old_offsets.items():
        if key not in new_offsets:
            raise ValueError("Cannot remap a packed vector into a layout missing a block.")
        new_pos, new_n = new_offsets[key]
        if int(n) != int(new_n):
            raise ValueError(
                "Cannot remap a packed vector across incompatible block shapes."
            )
        out[int(new_pos):int(new_pos) + int(n)] = vec[
            int(old_pos):int(old_pos) + int(n)
        ]
    return out


class AbelianDenseBoundaryActionDataTable:
    """Dense block-data action table for one Abelian two-site local layout."""

    def __init__(
        self,
        matrix,
        layout,
        qns,
        dirs,
        *,
        bond=None,
        source="exact_residual_table",
        boundary_family_tables=None,
        channel_matrices=None,
    ):
        self.matrix = np.asarray(matrix, dtype=complex)
        self.vector_layout = AbelianLocalVectorLayout.from_layout(
            layout,
            qns=qns,
            dirs=dirs,
        )
        self.layout = self.vector_layout.layout
        self.qns = [list(q) for q in self.vector_layout.qns]
        self.dirs = list(self.vector_layout.dirs)
        self.bond = None if bond is None else int(bond)
        self.source = str(source)
        self.boundary_family_tables = tuple(boundary_family_tables or ())
        self.channel_matrices = {
            str(name): np.asarray(value, dtype=complex)
            for name, value in (channel_matrices or {}).items()
        }

    @staticmethod
    def _size(layout):
        return AbelianLocalVectorLayout.from_layout(layout).size

    def flatten_data(self, data):
        return self.vector_layout.flatten_data(data, dtype=complex)

    def unflatten_data(self, vec):
        return self.vector_layout.unflatten_data(vec)

    def apply_data(self, data):
        return self.unflatten_data(self.matrix @ self.flatten_data(data))

    def apply_channels_data(self, data):
        vec = self.flatten_data(data)
        return {
            name: self.unflatten_data(matrix @ vec)
            for name, matrix in self.channel_matrices.items()
        }

    @property
    def dim(self):
        return int(self.matrix.shape[0])

    @property
    def nnz(self):
        return int(np.count_nonzero(np.abs(self.matrix) > 1.0e-14))

    @property
    def stats(self):
        active = []
        table_stats = []
        for table in self.boundary_family_tables:
            if table is None:
                continue
            active.extend(str(name) for name in table.active_family_names)
            table_stats.append({
                "side": str(table.side),
                "bond": int(table.bond),
                "active_family_names": table.active_family_names,
                "n_channels": int(table.n_channels),
                "symbolic_terms": int(table.symbolic_terms),
                "stored_elements": int(table.stored_elements),
            })
        return {
            "kind": "abelian_complementary_boundary_action_table",
            "source": self.source,
            "bond": self.bond,
            "dim": int(self.dim),
            "nnz": int(self.nnz),
            "stored_elements": int(np.asarray(self.matrix).size),
            "layout_blocks": int(len(self.layout)),
            "norm": float(np.linalg.norm(self.matrix)),
            "active_family_names": tuple(sorted(set(active))),
            "boundary_family_tables": tuple(table_stats),
            "boundary_channel_tables": tuple(
                {
                    "name": str(name),
                    "dim": int(matrix.shape[0]),
                    "nnz": int(np.count_nonzero(np.abs(matrix) > 1.0e-14)),
                    "stored_elements": int(matrix.size),
                    "norm": float(np.linalg.norm(matrix)),
                }
                for name, matrix in self.channel_matrices.items()
            ),
        }


class AbelianSparseBoundaryActionDataTable:
    """Sparse flat block-data action table for Abelian local solves."""

    def __init__(
        self,
        rows,
        cols,
        values,
        dim,
        layout,
        qns,
        dirs,
        *,
        bond=None,
        source="sparse_flat_complementary_action",
        boundary_family_tables=None,
        kernel_backend=None,
    ):
        self.dim = int(dim)
        raw_rows = self._flat_sparse_input(rows, np.int64)
        raw_cols = self._flat_sparse_input(cols, np.int64)
        raw_values = self._flat_sparse_input(values, np.complex128)
        if not (raw_rows.size == raw_cols.size == raw_values.size):
            raise ValueError("sparse action table rows, cols, and values differ in size")
        self.raw_nnz = int(raw_values.size)
        rows, cols, values = self._canonical_coo(
            raw_rows,
            raw_cols,
            raw_values,
            self.dim,
        )
        self.rows = rows
        self.cols = cols
        self.values = values
        self.indptr = self._csr_indptr(self.rows, self.dim)
        self.indices = self.cols.copy()
        self._set_vector_layout(layout, qns=qns, dirs=dirs)
        self.bond = None if bond is None else int(bond)
        self.source = str(source)
        self.boundary_family_tables = tuple(boundary_family_tables or ())
        self.storage = "sparse_csr"
        self.kernel_backend = kernel_backend

    @classmethod
    def from_csr(
        cls,
        indptr,
        indices,
        values,
        dim,
        layout,
        qns,
        dirs,
        *,
        raw_nnz=None,
        bond=None,
        source="sparse_flat_complementary_action",
        boundary_family_tables=None,
        kernel_backend=None,
    ):
        obj = cls.__new__(cls)
        obj.dim = int(dim)
        obj.indptr = np.ascontiguousarray(indptr, dtype=np.int64).reshape(-1)
        obj.indices = np.ascontiguousarray(indices, dtype=np.int64).reshape(-1)
        obj.values = np.ascontiguousarray(values, dtype=np.complex128).reshape(-1)
        if obj.indptr.size != obj.dim + 1:
            raise ValueError("CSR action table indptr size does not match dimension")
        if obj.indices.size != obj.values.size:
            raise ValueError("CSR action table indices and values differ in size")
        counts = np.diff(obj.indptr)
        obj.rows = np.repeat(np.arange(obj.dim, dtype=np.int64), counts)
        obj.cols = obj.indices.copy()
        obj.raw_nnz = int(obj.values.size if raw_nnz is None else raw_nnz)
        obj._set_vector_layout(layout, qns=qns, dirs=dirs)
        obj.bond = None if bond is None else int(bond)
        obj.source = str(source)
        obj.boundary_family_tables = tuple(boundary_family_tables or ())
        obj.storage = "sparse_csr"
        obj.kernel_backend = kernel_backend
        return obj

    def _set_vector_layout(self, layout, *, qns, dirs):
        self.vector_layout = AbelianLocalVectorLayout.from_layout(
            layout,
            qns=qns,
            dirs=dirs,
        )
        self.layout = self.vector_layout.layout
        self.qns = [list(q) for q in self.vector_layout.qns]
        self.dirs = list(self.vector_layout.dirs)

    @staticmethod
    def _flat_sparse_input(items, dtype):
        if isinstance(items, np.ndarray):
            return np.asarray(items, dtype=dtype).reshape(-1)
        if not isinstance(items, (list, tuple)):
            return np.asarray(items, dtype=dtype).reshape(-1)
        if not items:
            return np.zeros(0, dtype=dtype)
        if any(isinstance(item, np.ndarray) for item in items):
            chunks = [
                np.asarray(item, dtype=dtype).reshape(-1)
                for item in items
                if np.asarray(item).size
            ]
            if not chunks:
                return np.zeros(0, dtype=dtype)
            return np.concatenate(chunks)
        return np.asarray(items, dtype=dtype).reshape(-1)

    @staticmethod
    def _canonical_coo(rows, cols, values, dim, tol=1.0e-14):
        if values.size == 0:
            empty_i = np.zeros(0, dtype=np.int64)
            empty_v = np.zeros(0, dtype=np.complex128)
            return empty_i, empty_i.copy(), empty_v
        if (
            np.any(rows < 0)
            or np.any(cols < 0)
            or np.any(rows >= int(dim))
            or np.any(cols >= int(dim))
        ):
            raise ValueError("sparse action table index outside flat dimension")
        keep = np.abs(values) > float(tol)
        rows = rows[keep]
        cols = cols[keep]
        values = values[keep]
        if values.size == 0:
            empty_i = np.zeros(0, dtype=np.int64)
            empty_v = np.zeros(0, dtype=np.complex128)
            return empty_i, empty_i.copy(), empty_v
        order = np.lexsort((cols, rows))
        rows = np.ascontiguousarray(rows[order], dtype=np.int64)
        cols = np.ascontiguousarray(cols[order], dtype=np.int64)
        values = np.ascontiguousarray(values[order], dtype=np.complex128)
        starts = np.empty(values.size, dtype=bool)
        starts[0] = True
        starts[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
        group_starts = np.nonzero(starts)[0]
        summed = np.add.reduceat(values, group_starts)
        out_rows = rows[group_starts]
        out_cols = cols[group_starts]
        keep = np.abs(summed) > float(tol)
        return (
            np.ascontiguousarray(out_rows[keep], dtype=np.int64),
            np.ascontiguousarray(out_cols[keep], dtype=np.int64),
            np.ascontiguousarray(summed[keep], dtype=np.complex128),
        )

    @staticmethod
    def _csr_indptr(rows, dim):
        indptr = np.zeros(int(dim) + 1, dtype=np.int64)
        if rows.size:
            np.add.at(indptr, rows + 1, 1)
            np.cumsum(indptr, out=indptr)
        return indptr

    def flatten_data(self, data):
        return self.vector_layout.flatten_data(data)

    def unflatten_data(self, vec):
        return self.vector_layout.unflatten_data(vec)

    def matvec(self, vec):
        vector = np.asarray(vec, dtype=np.complex128).reshape(int(self.dim))
        kernel = self.kernel_backend
        if (
            self.values.size
            and kernel is not None
            and getattr(kernel, "CYTHON_AVAILABLE", False)
            and getattr(kernel, "sparse_csr_matvec", None) is not None
        ):
            return kernel.sparse_csr_matvec(
                self.indptr,
                self.indices,
                self.values,
                vector,
                int(self.dim),
            )
        if (
            self.values.size
            and kernel is not None
            and getattr(kernel, "CYTHON_AVAILABLE", False)
            and getattr(kernel, "sparse_coo_matvec", None) is not None
        ):
            return kernel.sparse_coo_matvec(
                self.rows,
                self.cols,
                self.values,
                vector,
                int(self.dim),
            )
        out = np.zeros(int(self.dim), dtype=complex)
        if self.values.size:
            np.add.at(out, self.rows, self.values * vector[self.cols])
        return out

    def apply_data(self, data):
        return self.unflatten_data(self.matvec(self.flatten_data(data)))

    @property
    def nnz(self):
        return int(self.values.size)

    @property
    def stats(self):
        active = []
        table_stats = []
        for table in self.boundary_family_tables:
            if table is None:
                continue
            active.extend(str(name) for name in table.active_family_names)
            table_stats.append({
                "side": str(table.side),
                "bond": int(table.bond),
                "active_family_names": table.active_family_names,
                "n_channels": int(table.n_channels),
                "symbolic_terms": int(table.symbolic_terms),
                "stored_elements": int(table.stored_elements),
            })
        return {
            "kind": "abelian_sparse_complementary_boundary_action_table",
            "source": self.source,
            "bond": self.bond,
            "dim": int(self.dim),
            "nnz": int(self.nnz),
            "raw_nnz": int(self.raw_nnz),
            "coalesced_nnz": int(self.nnz),
            "duplicates_removed": int(max(0, self.raw_nnz - self.nnz)),
            "storage": self.storage,
            "stored_elements": int(self.indptr.size + self.indices.size + self.values.size),
            "layout_blocks": int(len(self.layout)),
            "active_family_names": tuple(sorted(set(active))),
            "boundary_family_tables": tuple(table_stats),
        }


class AbelianRenormalizedActionDataTable:
    """Grouped renormalized-operator action table over Abelian block data."""

    def __init__(
        self,
        collected,
        dim,
        layout,
        qns,
        dirs,
        *,
        bond=None,
        source="renormalized_operator_action_table",
        boundary_family_tables=None,
        max_dense_block_elements=0,
        sparse_density_threshold=0.75,
        kernel_backend=None,
    ):
        self.collected = collected
        self.dim = int(dim)
        self._set_vector_layout(layout, qns=qns, dirs=dirs)
        self.bond = None if bond is None else int(bond)
        self.source = str(source)
        self.boundary_family_tables = tuple(boundary_family_tables or ())
        self.storage = "renormalized_operator_table"
        self.kernel_backend = kernel_backend
        self.block_matrices = None
        self.block_in_starts = None
        self.block_out_starts = None
        self.block_in_sizes = None
        self.block_out_sizes = None
        self.block_matrix_elements = 0
        self.block_sparse_rows = None
        self.block_sparse_cols = None
        self.block_sparse_values = None
        self.block_sparse_nnz = 0
        self._diagonal_cache = None
        groups = collected.get("matvec_groups")
        kernel = self.kernel_backend
        if groups is not None and int(max_dense_block_elements) > 0:
            dims_array = np.asarray(collected.get("group_dims_array"), dtype=np.int64)
            if dims_array.ndim == 2 and dims_array.shape[1] == 8:
                in_sizes = (
                    dims_array[:, 4]
                    * dims_array[:, 5]
                    * dims_array[:, 6]
                    * dims_array[:, 7]
                )
                out_sizes = (
                    dims_array[:, 0]
                    * dims_array[:, 1]
                    * dims_array[:, 2]
                    * dims_array[:, 3]
                )
                capacity = int(np.sum(in_sizes * out_sizes, dtype=np.int64))
                if (
                    capacity <= int(max_dense_block_elements)
                    and kernel is not None
                    and getattr(kernel, "CYTHON_AVAILABLE", False)
                    and getattr(kernel, "direct_operator_groups_dense_blocks", None)
                    is not None
                    and getattr(kernel, "direct_operator_block_matrices_matvec", None)
                    is not None
                ):
                    blocks, _block_in_sizes, _block_out_sizes = (
                        kernel.direct_operator_groups_dense_blocks(
                            collected["group_left"],
                            collected["group_right"],
                            dims_array,
                            collected.get("group_scales"),
                        )
                    )
                    block_in_starts = np.ascontiguousarray(
                        collected["group_in_starts_array"],
                        dtype=np.int64,
                    )
                    block_out_starts = np.ascontiguousarray(
                        collected["group_out_starts_array"],
                        dtype=np.int64,
                    )
                    blocks, block_in_starts, block_out_starts = (
                        self._coalesced_block_matrices(
                            blocks,
                            block_in_starts,
                            block_out_starts,
                        )
                    )
                    self.block_matrices = tuple(blocks)
                    self.block_in_starts = block_in_starts
                    self.block_out_starts = block_out_starts
                    self.block_in_sizes = np.ascontiguousarray(
                        [block.shape[1] for block in self.block_matrices],
                        dtype=np.int64,
                    )
                    self.block_out_sizes = np.ascontiguousarray(
                        [block.shape[0] for block in self.block_matrices],
                        dtype=np.int64,
                    )
                    self.block_matrix_elements = int(
                        sum(int(block.size) for block in self.block_matrices)
                    )
                    self.storage = "renormalized_operator_block_matrix_table"
                    if (
                        float(sparse_density_threshold) > 0.0
                        and kernel is not None
                        and getattr(kernel, "CYTHON_AVAILABLE", False)
                        and getattr(kernel, "direct_operator_block_sparse_matvec", None)
                        is not None
                    ):
                        rows = []
                        cols = []
                        values = []
                        nnz = 0
                        for block in self.block_matrices:
                            brow, bcol = np.nonzero(np.abs(block) > 1.0e-14)
                            bval = np.ascontiguousarray(
                                block[brow, bcol],
                                dtype=np.complex128,
                            )
                            brow = np.ascontiguousarray(brow, dtype=np.int64)
                            bcol = np.ascontiguousarray(bcol, dtype=np.int64)
                            rows.append(brow)
                            cols.append(bcol)
                            values.append(bval)
                            nnz += int(bval.size)
                        if nnz <= int(
                            float(sparse_density_threshold)
                            * max(1, int(self.block_matrix_elements))
                        ):
                            self.block_sparse_rows = tuple(rows)
                            self.block_sparse_cols = tuple(cols)
                            self.block_sparse_values = tuple(values)
                            self.block_sparse_nnz = int(nnz)
                            self.block_matrices = None
                            self.storage = "renormalized_operator_block_sparse_table"

    @staticmethod
    def _coalesced_block_matrices(blocks, in_starts, out_starts):
        seen = set()
        unique = True
        for block, in_start, out_start in zip(blocks, in_starts, out_starts):
            key = (int(in_start), int(out_start), tuple(np.asarray(block).shape))
            if key in seen:
                unique = False
                break
            seen.add(key)
        if unique:
            return (
                tuple(
                    np.ascontiguousarray(block, dtype=np.complex128)
                    for block in blocks
                ),
                np.ascontiguousarray(in_starts, dtype=np.int64),
                np.ascontiguousarray(out_starts, dtype=np.int64),
            )
        merged = OrderedDict()
        for block, in_start, out_start in zip(blocks, in_starts, out_starts):
            arr = np.asarray(block, dtype=np.complex128)
            key = (int(in_start), int(out_start), tuple(arr.shape))
            current = merged.get(key)
            if current is None:
                merged[key] = np.ascontiguousarray(arr.copy())
            else:
                current += arr
        out_blocks = []
        out_in_starts = []
        out_out_starts = []
        for (in_start, out_start, _shape), block in merged.items():
            out_blocks.append(block)
            out_in_starts.append(int(in_start))
            out_out_starts.append(int(out_start))
        return (
            tuple(out_blocks),
            np.ascontiguousarray(out_in_starts, dtype=np.int64),
            np.ascontiguousarray(out_out_starts, dtype=np.int64),
        )

    def _set_vector_layout(self, layout, *, qns, dirs):
        self.vector_layout = AbelianLocalVectorLayout.from_layout(
            layout,
            qns=qns,
            dirs=dirs,
        )
        self.layout = self.vector_layout.layout
        self.qns = [list(q) for q in self.vector_layout.qns]
        self.dirs = list(self.vector_layout.dirs)

    def flatten_data(self, data):
        return self.vector_layout.flatten_data(data)

    def unflatten_data(self, vec):
        return self.vector_layout.unflatten_data(vec)

    def matvec(self, vec):
        vector = np.asarray(vec, dtype=np.complex128).reshape(int(self.dim))
        collected = self.collected
        kernel = self.kernel_backend
        if (
            self.block_sparse_values is not None
            and kernel is not None
            and getattr(kernel, "CYTHON_AVAILABLE", False)
            and getattr(kernel, "direct_operator_block_sparse_matvec", None)
            is not None
        ):
            return kernel.direct_operator_block_sparse_matvec(
                self.block_sparse_rows,
                self.block_sparse_cols,
                self.block_sparse_values,
                self.block_in_starts,
                self.block_out_starts,
                vector,
                int(self.dim),
            )
        if (
            self.block_matrices is not None
            and kernel is not None
            and getattr(kernel, "CYTHON_AVAILABLE", False)
            and getattr(kernel, "direct_operator_block_matrices_matvec", None)
            is not None
        ):
            return kernel.direct_operator_block_matrices_matvec(
                self.block_matrices,
                self.block_in_starts,
                self.block_out_starts,
                vector,
                int(self.dim),
            )
        groups = collected.get("matvec_groups")
        if (
            groups is not None
            and kernel is not None
            and getattr(kernel, "CYTHON_AVAILABLE", False)
            and getattr(kernel, "direct_operator_groups_matvec", None) is not None
        ):
            return kernel.direct_operator_groups_matvec(
                collected["group_left"],
                collected["group_right"],
                collected["group_dims_array"],
                collected["group_in_starts_array"],
                collected["group_out_starts_array"],
                vector,
                int(self.dim),
                collected.get("group_scales"),
            )
        if (
            kernel is not None
            and getattr(kernel, "CYTHON_AVAILABLE", False)
            and getattr(kernel, "direct_operator_entries_matvec", None) is not None
        ):
            return kernel.direct_operator_entries_matvec(
                collected["left"],
                collected["right"],
                collected["dims_array"],
                collected["in_starts_array"],
                collected["out_starts_array"],
                vector,
                int(self.dim),
                collected.get("scales_array"),
            )
        out = np.zeros(int(self.dim), dtype=np.complex128)
        if groups:
            iterator = (
                (
                    group["left"],
                    group["right"],
                    group["dims"],
                    group["in_start"],
                    group["out_start"],
                    group.get("scales"),
                )
                for group in groups
            )
        else:
            scales = collected.get("scales_array")
            if scales is None:
                scales = (None,) * len(collected.get("left", ()))
            iterator = zip(
                collected.get("left", ()),
                collected.get("right", ()),
                collected.get("dims_array", ()),
                collected.get("in_starts_array", ()),
                collected.get("out_starts_array", ()),
                scales,
            )
        for left_stack, right_stack, dims, in_start, out_start, scales in iterator:
            ni, nl, nu, nv, nj, nx, nk, ny = (int(v) for v in dims)
            in_size = nj * nx * nk * ny
            out_size = ni * nl * nu * nv
            block = vector[int(in_start) : int(in_start) + in_size]
            if not np.any(block):
                continue
            a_mat = np.ascontiguousarray(
                block.reshape(nj, nk, nx, ny)
                .transpose(0, 2, 1, 3)
                .reshape(nj * nx, nk * ny)
            )
            tmp = np.matmul(left_stack, a_mat)
            mat_stack = np.matmul(tmp, right_stack)
            if scales is None:
                mat = mat_stack.sum(axis=0)
            elif np.ndim(scales) == 0:
                mat = mat_stack.sum(axis=0) * scales
            else:
                mat = (mat_stack * np.asarray(scales).reshape(-1, 1, 1)).sum(axis=0)
            out_block = (
                mat.reshape(ni, nu, nl, nv)
                .transpose(0, 2, 1, 3)
                .reshape(out_size)
            )
            out[int(out_start) : int(out_start) + out_size] += out_block
        return out

    def diagonal_flat(self):
        if self._diagonal_cache is not None:
            return self._diagonal_cache
        diag = np.zeros(int(self.dim), dtype=np.complex128)
        if self.block_sparse_values is not None:
            for rows, cols, values, in_start, out_start in zip(
                self.block_sparse_rows,
                self.block_sparse_cols,
                self.block_sparse_values,
                self.block_in_starts,
                self.block_out_starts,
            ):
                rows = np.asarray(rows, dtype=np.int64).reshape(-1)
                cols = np.asarray(cols, dtype=np.int64).reshape(-1)
                values = np.asarray(values, dtype=np.complex128).reshape(-1)
                global_rows = int(out_start) + rows
                global_cols = int(in_start) + cols
                keep = global_rows == global_cols
                if np.any(keep):
                    diag[global_rows[keep]] += values[keep]
            self._diagonal_cache = diag
            return diag
        if self.block_matrices is not None:
            for block, in_start, out_start in zip(
                self.block_matrices,
                self.block_in_starts,
                self.block_out_starts,
            ):
                block = np.asarray(block, dtype=np.complex128)
                in_start = int(in_start)
                out_start = int(out_start)
                row0 = max(0, in_start - out_start)
                col0 = max(0, out_start - in_start)
                n = min(block.shape[0] - row0, block.shape[1] - col0)
                if n <= 0:
                    continue
                idx = np.arange(n, dtype=np.int64)
                diag[out_start + row0 + idx] += block[row0 + idx, col0 + idx]
            self._diagonal_cache = diag
            return diag
        basis = np.zeros(int(self.dim), dtype=np.complex128)
        for index in range(int(self.dim)):
            basis[index] = 1.0
            diag[index] = self.matvec(basis)[index]
            basis[index] = 0.0
        self._diagonal_cache = diag
        return diag

    def apply_data(self, data):
        return self.unflatten_data(self.matvec(self.flatten_data(data)))

    @property
    def n_entries(self):
        return int(len(self.collected.get("left", ())))

    @property
    def n_groups(self):
        return int(len(self.collected.get("matvec_groups") or ()))

    @property
    def n_group_channels(self):
        groups = self.collected.get("matvec_groups") or ()
        return int(sum(int(group.get("channels", 0)) for group in groups))

    @property
    def n_block_matrices(self):
        if self.block_sparse_values is not None:
            return int(len(self.block_sparse_values))
        return int(0 if self.block_matrices is None else len(self.block_matrices))

    @property
    def stats(self):
        active = []
        table_stats = []
        for table in self.boundary_family_tables:
            if table is None:
                continue
            active.extend(str(name) for name in table.active_family_names)
            table_stats.append({
                "side": str(table.side),
                "bond": int(table.bond),
                "active_family_names": table.active_family_names,
                "n_channels": int(table.n_channels),
                "symbolic_terms": int(table.symbolic_terms),
                "stored_elements": int(table.stored_elements),
            })
        return {
            "kind": "abelian_renormalized_operator_action_table",
            "source": self.source,
            "bond": self.bond,
            "dim": int(self.dim),
            "entries": int(self.n_entries),
            "groups": int(self.n_groups),
            "group_channels": int(self.n_group_channels),
            "block_matrices": int(self.n_block_matrices),
            "block_matrix_elements": int(self.block_matrix_elements),
            "block_sparse_nnz": int(self.block_sparse_nnz),
            "storage": self.storage,
            "layout_blocks": int(len(self.layout)),
            "active_family_names": tuple(sorted(set(active))),
            "family_names": tuple(self.collected.get("family_names", ())),
            "boundary_family_tables": tuple(table_stats),
        }


class AbelianGroupedRenormalizedDataTable:
    """C++ grouped renormalized-operator table over Abelian block data."""

    def __init__(
        self,
        cpp_table,
        collected,
        dim,
        layout,
        qns,
        dirs,
        *,
        bond=None,
        source="moving_environment_cpp_grouped_renormalized_table",
        boundary_family_tables=None,
    ):
        self.cpp_table = cpp_table
        self.collected = collected
        self.dim = int(dim)
        self._set_vector_layout(layout, qns=qns, dirs=dirs)
        self.bond = None if bond is None else int(bond)
        self.source = str(source)
        self.boundary_family_tables = tuple(boundary_family_tables or ())
        self.storage = str(cpp_table.storage())
        try:
            self.last_refresh_kind = str(cpp_table.last_refresh_kind())
        except Exception:
            self.last_refresh_kind = "build"
        self.block_matrices = None
        self.block_sparse_values = None
        self.block_matrix_elements = int(cpp_table.block_matrix_elements())
        self.block_sparse_nnz = int(cpp_table.block_sparse_nnz())
        self._diagonal_cache = None
        self._moving_environment_cpp_renormalized_table = cpp_table
        self._moving_environment_cpp_renormalized_table_validated = True
        self._moving_environment_structural_key = None
        self.cpp_moving_environment = None
        self.cpp_moving_environment_key = None

    def _set_vector_layout(self, layout, *, qns, dirs):
        self.vector_layout = AbelianLocalVectorLayout.from_layout(
            layout,
            qns=qns,
            dirs=dirs,
        )
        self.layout = self.vector_layout.layout
        self.qns = [list(q) for q in self.vector_layout.qns]
        self.dirs = list(self.vector_layout.dirs)

    @staticmethod
    def _group_payload_arrays(collected):
        group_left = tuple(
            np.ascontiguousarray(block, dtype=np.complex128)
            for block in collected["group_left"]
        )
        group_right = tuple(
            np.ascontiguousarray(block, dtype=np.complex128)
            for block in collected["group_right"]
        )
        group_scales = collected.get("group_scales")
        if group_scales is not None:
            group_scales = tuple(
                None
                if scale is None
                else np.ascontiguousarray(scale, dtype=np.complex128)
                for scale in group_scales
            )
        return (
            group_left,
            group_right,
            np.ascontiguousarray(collected["group_dims_array"], dtype=np.int64),
            np.ascontiguousarray(
                collected["group_in_starts_array"],
                dtype=np.int64,
            ),
            np.ascontiguousarray(
                collected["group_out_starts_array"],
                dtype=np.int64,
            ),
            group_scales,
        )

    @staticmethod
    def _raw_payload_arrays(collected):
        builder = collected.get("raw_builder")
        if builder is not None:
            return (
                tuple(builder.left_entries()),
                tuple(builder.right_entries()),
                np.ascontiguousarray(builder.dims_array(), dtype=np.int64),
                np.ascontiguousarray(builder.in_starts_array(), dtype=np.int64),
                np.ascontiguousarray(builder.out_starts_array(), dtype=np.int64),
                (
                    None
                    if builder.scales_array() is None
                    else np.ascontiguousarray(
                        builder.scales_array(),
                        dtype=np.complex128,
                    )
                ),
            )
        raw_left = tuple(
            np.ascontiguousarray(block, dtype=np.complex128)
            for block in collected["left"]
        )
        raw_right = tuple(
            np.ascontiguousarray(block, dtype=np.complex128)
            for block in collected["right"]
        )
        raw_scales = collected.get("scales_array")
        if raw_scales is not None:
            raw_scales = np.ascontiguousarray(raw_scales, dtype=np.complex128)
        return (
            raw_left,
            raw_right,
            np.ascontiguousarray(collected["dims_array"], dtype=np.int64),
            np.ascontiguousarray(collected["in_starts_array"], dtype=np.int64),
            np.ascontiguousarray(collected["out_starts_array"], dtype=np.int64),
            raw_scales,
        )

    @staticmethod
    def _raw_group_capacity(dims_array, in_starts, out_starts):
        dims_array = np.asarray(dims_array, dtype=np.int64)
        in_starts = np.asarray(in_starts, dtype=np.int64).reshape(-1)
        out_starts = np.asarray(out_starts, dtype=np.int64).reshape(-1)
        if dims_array.ndim != 2 or dims_array.shape[1] != 8:
            return 0
        if in_starts.shape[0] != dims_array.shape[0]:
            return 0
        if out_starts.shape[0] != dims_array.shape[0]:
            return 0
        seen = set()
        capacity = 0
        for dims, in_start, out_start in zip(dims_array, in_starts, out_starts):
            key = (
                tuple(int(value) for value in dims),
                int(in_start),
                int(out_start),
            )
            if key in seen:
                continue
            seen.add(key)
            ni, nl, nu, nv, nj, nx, nk, ny = key[0]
            capacity += int(ni * nl * nu * nv * nj * nx * nk * ny)
        return int(capacity)

    def refresh_from_collected(
        self,
        collected,
        *,
        dim=None,
        layout=None,
        qns=None,
        dirs=None,
        bond=None,
        boundary_family_tables=None,
        sparse_density_threshold=0.0,
    ):
        dim = int(self.dim if dim is None else dim)
        if "group_dims_array" in collected:
            (
                group_left,
                group_right,
                dims_array,
                in_starts,
                out_starts,
                group_scales,
            ) = self._group_payload_arrays(collected)
            self.cpp_table.refresh(
                group_left,
                group_right,
                dims_array,
                in_starts,
                out_starts,
                dim,
                float(sparse_density_threshold),
                group_scales,
            )
        else:
            raw_builder = collected.get("raw_builder")
            if raw_builder is not None:
                if getattr(self.cpp_table, "refresh_from_raw_builder", None) is None:
                    raise RuntimeError(
                        "C++ grouped table does not support raw builder refresh"
                    )
                self.cpp_table.refresh_from_raw_builder(
                    raw_builder,
                    dim,
                    float(sparse_density_threshold),
                )
            else:
                if getattr(self.cpp_table, "refresh_from_raw", None) is None:
                    raise RuntimeError("C++ grouped table does not support raw refresh")
                (
                    raw_left,
                    raw_right,
                    dims_array,
                    in_starts,
                    out_starts,
                    raw_scales,
                ) = self._raw_payload_arrays(collected)
                self.cpp_table.refresh_from_raw(
                    raw_left,
                    raw_right,
                    dims_array,
                    in_starts,
                    out_starts,
                    dim,
                    float(sparse_density_threshold),
                    raw_scales,
                )
        self.collected = collected
        self.dim = int(dim)
        if layout is not None or qns is not None or dirs is not None:
            self._set_vector_layout(
                self.layout if layout is None else layout,
                qns=self.qns if qns is None else qns,
                dirs=self.dirs if dirs is None else dirs,
            )
        if bond is not None:
            self.bond = int(bond)
        if boundary_family_tables is not None:
            self.boundary_family_tables = tuple(boundary_family_tables)
        self.storage = str(self.cpp_table.storage())
        try:
            self.last_refresh_kind = str(self.cpp_table.last_refresh_kind())
        except Exception:
            self.last_refresh_kind = "unknown"
        self.block_matrix_elements = int(self.cpp_table.block_matrix_elements())
        self.block_sparse_nnz = int(self.cpp_table.block_sparse_nnz())
        self._diagonal_cache = None
        self._moving_environment_cpp_renormalized_table = self.cpp_table
        self._moving_environment_cpp_renormalized_table_validated = True
        return self

    def bind_cpp_moving_environment(self, environment, key):
        self.cpp_moving_environment = environment
        self.cpp_moving_environment_key = str(key)
        self._diagonal_cache = None
        return self

    def flatten_data(self, data):
        return self.vector_layout.flatten_data(data)

    def unflatten_data(self, vec):
        return self.vector_layout.unflatten_data(vec)

    def matvec(self, vec):
        vector = np.ascontiguousarray(vec, dtype=np.complex128).reshape(int(self.dim))
        if self.cpp_moving_environment is not None:
            return np.asarray(
                self.cpp_moving_environment.grouped_matvec(
                    self.cpp_moving_environment_key,
                    vector,
                ),
                dtype=np.complex128,
            ).reshape(int(self.dim))
        return np.asarray(self.cpp_table.matvec(vector), dtype=np.complex128).reshape(
            int(self.dim)
        )

    def diagonal_flat(self):
        if self._diagonal_cache is None:
            if self.cpp_moving_environment is not None:
                self._diagonal_cache = np.asarray(
                    self.cpp_moving_environment.grouped_diagonal(
                        self.cpp_moving_environment_key
                    ),
                    dtype=np.complex128,
                ).reshape(int(self.dim))
            else:
                self._diagonal_cache = np.asarray(
                    self.cpp_table.diagonal(),
                    dtype=np.complex128,
                ).reshape(int(self.dim))
        return self._diagonal_cache

    def davidson(self, diagonal, v0, tol, max_iter, restart_dim, accept_unconverged):
        v0 = np.ascontiguousarray(v0, dtype=np.complex128).reshape(int(self.dim))
        if self.cpp_moving_environment is not None:
            return self.cpp_moving_environment.grouped_davidson(
                self.cpp_moving_environment_key,
                v0,
                float(tol),
                int(max_iter),
                int(restart_dim),
                bool(accept_unconverged),
            )
        return self.cpp_table.davidson(
            np.ascontiguousarray(diagonal, dtype=np.complex128),
            v0,
            float(tol),
            int(max_iter),
            int(restart_dim),
            bool(accept_unconverged),
        )

    def apply_data(self, data):
        return self.unflatten_data(self.matvec(self.flatten_data(data)))

    @property
    def n_entries(self):
        if "entry_count" in self.collected:
            return int(self.collected.get("entry_count") or 0)
        prebuilt = self.collected.get("cpp_grouped_table")
        if prebuilt is not None:
            try:
                return int(prebuilt.n_routes())
            except Exception:
                return int(prebuilt.n_group_channels())
        builder = self.collected.get("raw_builder")
        if builder is not None:
            try:
                return int(builder.size())
            except Exception:
                return 0
        return int(len(self.collected.get("left", ())))

    @property
    def n_groups(self):
        return int(self.cpp_table.n_groups())

    @property
    def n_group_channels(self):
        return int(self.cpp_table.n_group_channels())

    @property
    def n_block_matrices(self):
        return int(self.cpp_table.n_blocks())

    @property
    def stats(self):
        active = []
        table_stats = []
        for table in self.boundary_family_tables:
            if table is None:
                continue
            active.extend(str(name) for name in table.active_family_names)
            table_stats.append({
                "side": str(table.side),
                "bond": int(table.bond),
                "active_family_names": table.active_family_names,
                "n_channels": int(table.n_channels),
                "symbolic_terms": int(table.symbolic_terms),
                "stored_elements": int(table.stored_elements),
            })
        return {
            "kind": "moving_environment_cpp_grouped_renormalized_table",
            "source": self.source,
            "bond": self.bond,
            "dim": int(self.dim),
            "entries": int(self.n_entries),
            "groups": int(self.n_groups),
            "group_channels": int(self.n_group_channels),
            "block_matrices": int(self.n_block_matrices),
            "block_matrix_elements": int(self.block_matrix_elements),
            "block_sparse_nnz": int(self.block_sparse_nnz),
            "storage": self.storage,
            "layout_blocks": int(len(self.layout)),
            "active_family_names": tuple(sorted(set(active))),
            "family_names": tuple(self.collected.get("family_names", ())),
            "boundary_family_tables": tuple(table_stats),
        }


class AbelianCompactBlockDataTable:
    """Direct flat block table built from a compact two-site MPO plan."""

    def __init__(self, block_matrices, in_starts, out_starts, dim, layout):
        self.block_matrices = tuple(
            np.ascontiguousarray(block, dtype=np.complex128)
            for block in block_matrices
        )
        self.block_in_starts = np.ascontiguousarray(in_starts, dtype=np.int64)
        self.block_out_starts = np.ascontiguousarray(out_starts, dtype=np.int64)
        self.dim = int(dim)
        self.layout = tuple(layout)
        self.storage = "compact_block_table"
        self.n_entries = int(len(self.block_matrices))
        self.n_groups = int(len(self.block_matrices))
        self.n_group_channels = int(len(self.block_matrices))
        self.n_block_matrices = int(len(self.block_matrices))
        self.block_matrix_elements = int(
            sum(int(block.size) for block in self.block_matrices)
        )
        self.block_sparse_nnz = self.block_matrix_elements
        self._diagonal_cache = None

    def matvec(self, vector):
        vector = np.ascontiguousarray(vector, dtype=np.complex128).reshape(self.dim)
        out = np.zeros(self.dim, dtype=np.complex128)
        for block, in_start, out_start in zip(
            self.block_matrices,
            self.block_in_starts,
            self.block_out_starts,
        ):
            in_start = int(in_start)
            out_start = int(out_start)
            rows, cols = block.shape
            out[out_start:out_start + rows] += block @ vector[in_start:in_start + cols]
        return out

    def diagonal_flat(self):
        if self._diagonal_cache is not None:
            return self._diagonal_cache
        diag = np.zeros(self.dim, dtype=np.complex128)
        for block, in_start, out_start in zip(
            self.block_matrices,
            self.block_in_starts,
            self.block_out_starts,
        ):
            in_start = int(in_start)
            out_start = int(out_start)
            rows, cols = block.shape
            first = max(in_start, out_start)
            last = min(in_start + cols, out_start + rows)
            if last <= first:
                continue
            for global_index in range(first, last):
                diag[global_index] += block[
                    global_index - out_start,
                    global_index - in_start,
                ]
        self._diagonal_cache = diag
        return diag


class AbelianCompactRenormalizedDataTable:
    """Compact renormalized effective-H table for one flat local layout."""

    def __init__(self, cpp_plan, dim, layout):
        self.cpp_plan = cpp_plan
        self.dim = int(dim)
        self.layout = tuple(layout)
        self.storage = "compact_renormalized_table"
        self.backend = "compact_plan"
        self.n_entries = 0
        self.n_groups = 0
        self.n_group_channels = 0
        self.n_block_matrices = 0
        self.block_matrix_elements = 0
        self.block_sparse_nnz = 0
        self.structure_key = None
        self.numeric_token = None
        self._refresh_keys = None
        self._refresh_entries = None
        self._diagonal_routes = None
        self._diagonal_cache = None
        self._diagonal_cache_token = None
        self.cpp_moving_environment = None
        self.cpp_moving_environment_key = None

    def matvec(self, vector):
        vector = np.ascontiguousarray(vector, dtype=np.complex128).reshape(self.dim)
        if self.cpp_moving_environment is not None:
            return np.asarray(
                self.cpp_moving_environment.matvec(
                    self.cpp_moving_environment_key,
                    vector,
                ),
                dtype=np.complex128,
            ).reshape(self.dim)
        return np.asarray(self.cpp_plan.matvec(vector), dtype=np.complex128).reshape(
            self.dim
        )

    def davidson(self, diagonal, v0, tol, max_iter, restart_dim, accept_unconverged):
        if self.cpp_moving_environment is not None:
            return self.cpp_moving_environment.davidson(
                self.cpp_moving_environment_key,
                np.ascontiguousarray(v0, dtype=np.complex128),
                float(tol),
                int(max_iter),
                int(restart_dim),
                bool(accept_unconverged),
            )
        return self.cpp_plan.davidson(
            np.ascontiguousarray(diagonal, dtype=np.complex128),
            np.ascontiguousarray(v0, dtype=np.complex128),
            float(tol),
            int(max_iter),
            int(restart_dim),
            bool(accept_unconverged),
        )

    def install_refresh_recipe(self, *, e_keys, w1_keys, w2_keys, f_keys, entries):
        self._refresh_keys = {
            "e": tuple(e_keys),
            "w1": tuple(w1_keys),
            "w2": tuple(w2_keys),
            "f": tuple(f_keys),
        }
        self._refresh_entries = {
            name: tuple(
                np.ascontiguousarray(group, dtype=np.int64)
                for group in tuple(groups)
            )
            for name, groups in dict(entries).items()
        }

    def install_diagonal_routes(self, routes):
        self._diagonal_routes = np.ascontiguousarray(routes, dtype=np.int64)
        self.n_diagonal_routes = int(self._diagonal_routes.shape[0])

    def bind_cpp_moving_environment(self, environment, key):
        self.cpp_moving_environment = environment
        self.cpp_moving_environment_key = str(key)
        self._diagonal_cache = None
        self._diagonal_cache_token = None
        return self

    def diagonal_flat(self):
        if (
            self._diagonal_routes is None
            or int(self._diagonal_routes.size) == 0
            or not hasattr(self.cpp_plan, "diagonal_from_routes")
        ):
            return None
        if self.cpp_moving_environment is not None:
            try:
                diagonal = np.asarray(
                    self.cpp_moving_environment.diagonal(
                        self.cpp_moving_environment_key
                    ),
                    dtype=np.complex128,
                ).reshape(self.dim)
            except Exception as exc:
                self.last_diagonal_error = str(exc)
                return None
            self._diagonal_cache = diagonal
            self._diagonal_cache_token = self.numeric_token
            return diagonal
        if (
            self._diagonal_cache is not None
            and self._diagonal_cache_token == self.numeric_token
        ):
            return self._diagonal_cache
        try:
            diagonal = np.asarray(
                self.cpp_plan.diagonal_from_routes(self._diagonal_routes),
                dtype=np.complex128,
            ).reshape(self.dim)
        except Exception as exc:
            self.last_diagonal_error = str(exc)
            return None
        self._diagonal_cache = diagonal
        self._diagonal_cache_token = self.numeric_token
        return diagonal

    def refresh_from_operator(self, operator):
        if self._refresh_keys is None or self._refresh_entries is None:
            return False
        if not (
            hasattr(self.cpp_plan, "update_stacks_from_blocks")
            or hasattr(self.cpp_plan, "update_stacks")
        ):
            return False
        keys = self._refresh_keys
        entries = self._refresh_entries
        try:
            e_blocks = tuple(
                np.ascontiguousarray(operator.E.data[key])
                for key in keys["e"]
            )
            w1_blocks = tuple(
                np.ascontiguousarray(operator.W[0].data[key])
                for key in keys["w1"]
            )
            w2_blocks = tuple(
                np.ascontiguousarray(operator.W[1].data[key])
                for key in keys["w2"]
            )
            f_blocks = tuple(
                np.ascontiguousarray(operator.F.data[key])
                for key in keys["f"]
            )
            if hasattr(self.cpp_plan, "update_stacks_from_blocks"):
                self.cpp_plan.update_stacks_from_blocks(
                    e_blocks,
                    entries["r"],
                    w1_blocks,
                    entries["t2"],
                    w2_blocks,
                    entries["t3"],
                    f_blocks,
                    entries["out"],
                )
                self.last_refresh_backend = "cpp_block_refresh"
            else:
                r_e = operator._batched_compact_static_stacks(entries["r"], e_blocks, 0)
                t2_w = operator._batched_compact_static_stacks(entries["t2"], w1_blocks, 1)
                t3_w = operator._batched_compact_static_stacks(entries["t3"], w2_blocks, 1)
                out_f = operator._batched_compact_static_stacks(entries["out"], f_blocks, 1)
                self.cpp_plan.update_stacks(r_e, t2_w, t3_w, out_f)
                self.last_refresh_backend = "python_stack_refresh"
        except Exception as exc:
            self.last_refresh_error = str(exc)
            return False
        self.numeric_token = operator._action_token()
        self._diagonal_cache = None
        self._diagonal_cache_token = None
        if (
            self.cpp_moving_environment is not None
            and self.cpp_moving_environment_key is not None
        ):
            try:
                self.cpp_moving_environment.invalidate_diagonal(
                    self.cpp_moving_environment_key
                )
            except Exception:
                pass
        return True


class AbelianMovingEnvironmentFlatMatvec:
    """Profiled flat-vector matvec wrapper for a moving-environment table."""

    def __init__(self, environment, operator, table, layout, proto_dirs):
        self.environment = environment
        self.operator = operator
        self.table = table
        self.layout = tuple(layout)
        self.proto_dirs = tuple(proto_dirs)
        self.calls = 0
        self.seconds = 0.0
        self._pending_calls = 0
        self._pending_seconds = 0.0
        self._pending_last_seconds = 0.0
        self._diagonal_cache = None

    def bind_operator(self, operator):
        if self.operator is not operator:
            self.flush_profile()
        self.operator = operator
        return self

    def bind_table(self, operator, table, layout, proto_dirs):
        if self.operator is not operator:
            self.flush_profile()
        self.operator = operator
        if self.table is not table:
            self.table = table
        self.layout = tuple(layout)
        self.proto_dirs = tuple(proto_dirs)
        self._diagonal_cache = None
        return self

    def matvec(self, vec):
        start = time.perf_counter()
        out = self.environment.compiled_backend.apply_renormalized_operator_table(
            self.table,
            vec,
        )
        elapsed = float(time.perf_counter() - start)
        self.calls += 1
        self.seconds += elapsed
        self._pending_calls += 1
        self._pending_seconds += elapsed
        self._pending_last_seconds = elapsed
        return np.asarray(out, dtype=np.complex128).reshape(int(self.table.dim))

    def diagonal(self):
        if self._diagonal_cache is None:
            self._diagonal_cache = self.table.diagonal_flat()
        return self._diagonal_cache

    def flush_profile(self):
        calls = int(self._pending_calls)
        if calls <= 0:
            return
        operator = self.operator
        stats = operator.profile_stats.setdefault(
            "packed_flat_complementary_family_action",
            {},
        )
        seconds = float(self._pending_seconds)
        stats["calls"] = int(stats.get("calls", 0)) + calls
        stats["seconds"] = float(stats.get("seconds", 0.0)) + seconds
        stats["last_seconds"] = float(self._pending_last_seconds)
        stats["last_cache"] = "moving_environment_direct"
        stats["compiled_direct_matvec_calls"] = int(
            stats.get("compiled_direct_matvec_calls", 0)
        ) + calls
        stats["compiled_direct_matvec_backend"] = "renormalized_table"
        stats["renormalized_operator_table_calls"] = int(
            stats.get("renormalized_operator_table_calls", 0)
        ) + calls
        stats["compiled_direct_matvec_entries"] = int(
            stats.get("compiled_direct_matvec_entries", 0)
        ) + int(self.table.n_entries) * calls
        stats["compiled_direct_matvec_groups"] = int(
            stats.get("compiled_direct_matvec_groups", 0)
        ) + int(self.table.n_groups) * calls
        stats["compiled_direct_matvec_group_channels"] = int(
            stats.get("compiled_direct_matvec_group_channels", 0)
        ) + int(self.table.n_group_channels) * calls
        stats["renormalized_operator_table_storage"] = str(self.table.storage)
        stats["renormalized_operator_table_block_matrices_last"] = int(
            self.table.n_block_matrices
        )
        stats["renormalized_operator_table_block_matrix_elements_last"] = int(
            self.table.block_matrix_elements
        )
        stats["renormalized_operator_table_block_sparse_nnz_last"] = int(
            self.table.block_sparse_nnz
        )
        stats["moving_environment_direct_matvec_calls"] = int(
            stats.get("moving_environment_direct_matvec_calls", 0)
        ) + calls
        stats["moving_environment_direct_matvec_seconds"] = float(
            stats.get("moving_environment_direct_matvec_seconds", 0.0)
        ) + seconds
        stats["last"] = {
            "dimension": int(self.table.dim),
            "nnz": None,
            "raw_nnz": None,
            "source": "moving_environment_direct_renormalized_table",
            "storage": str(self.table.storage),
            "cache": "moving_environment_direct",
            "bond": None if operator.bond is None else int(operator.bond),
        }
        moving_stats = self.environment.moving_profile_stats
        moving_stats["compiled_flat_matvec_calls"] = int(
            moving_stats.get("compiled_flat_matvec_calls", 0)
        ) + calls
        moving_stats["compiled_flat_matvec_seconds"] = float(
            moving_stats.get("compiled_flat_matvec_seconds", 0.0)
        ) + seconds
        self._pending_calls = 0
        self._pending_seconds = 0.0
        self._pending_last_seconds = 0.0


@dataclass(frozen=True)
class AbelianPackedLocalStateProto:
    """Packed two-site local state layout used by Abelian direct-family validators."""

    tensor: AbelianPackedBoundaryTensor
    source: str = "abelian_packed_local_state_proto"

    @classmethod
    def from_site_tensors(
        cls,
        left,
        right,
        *,
        source="abelian_packed_local_state_proto",
        merge_source="abelian_packed_local_state_proto_merge",
    ):
        left = pack_abelian_boundary_tensor(
            left,
            source=f"{source}_left",
        )
        right = pack_abelian_boundary_tensor(
            right,
            source=f"{source}_right",
        )
        merged = tensordot_abelian_packed_boundary_tensors(
            left,
            right,
            axes=([1], [0]),
            source=merge_source,
        )
        tensor = transpose_abelian_packed_boundary_tensor(
            merged,
            (0, 2, 1, 3),
            source=source,
        )
        return cls(tensor=tensor, source=source)

    @property
    def dirs(self):
        return list(getattr(self.tensor, "dirs", ()))

    @property
    def qns(self):
        return getattr(self.tensor, "qns", None)

    @property
    def keys(self):
        return tuple(getattr(self.tensor, "keys", ()))

    @property
    def blocks(self):
        return tuple(getattr(self.tensor, "blocks", ()))

    def layout(self):
        return tuple(
            (tuple(key), tuple(int(dim) for dim in np.asarray(block).shape))
            for key, block in zip(self.keys, self.blocks)
        )

    def basis(self, key, shape, *, offset=0, source="abelian_packed_local_basis"):
        data = np.zeros(tuple(int(dim) for dim in shape), dtype=complex)
        if data.size:
            data.reshape(-1)[int(offset)] = 1.0
        return AbelianPackedBoundaryTensor(
            (tuple(key),),
            (data,),
        dirs=self.dirs,
        qns=self.qns,
        source=source,
        assume_unique=True,
    )


def scale_abelian_boundary_tensor(
    tensor,
    scalar,
    *,
    source="packed_boundary_tensor_scale",
):
    """Scale packed boundary blocks without materializing a legacy tensor."""

    if is_abelian_packed_boundary_tensor(tensor):
        return AbelianPackedBoundaryTensor(
            tensor.keys,
            tuple(complex(scalar) * np.asarray(block) for block in tensor.blocks),
            dirs=list(getattr(tensor, "dirs", ())),
            qns=getattr(tensor, "qns", None),
            source=source,
            assume_unique=True,
        )
    return tensor * scalar


def add_abelian_packed_boundary_tensors(
    left,
    right,
    *,
    source="packed_boundary_tensor_sum",
):
    """Add two packed/data-view tensors and return a packed tensor."""

    if left is None:
        return right
    if right is None:
        return left
    left_keys, left_blocks, left_dirs, left_qns = abelian_packed_tensor_items(left)
    right_keys, right_blocks, right_dirs, right_qns = abelian_packed_tensor_items(right)
    out = OrderedDict()
    for key, block in zip(left_keys, left_blocks):
        out[tuple(key)] = np.asarray(block)
    for key, block in zip(right_keys, right_blocks):
        key = tuple(key)
        block = np.asarray(block)
        out[key] = block if key not in out else out[key] + block
    qns = left_qns if left_qns is not None else right_qns
    dirs = left_dirs if left_dirs else right_dirs
    return AbelianPackedBoundaryTensor(
        tuple(out.keys()),
        tuple(out.values()),
        dirs=dirs,
        qns=qns,
        source=source,
        assume_unique=True,
    )


def sum_abelian_packed_boundary_terms(
    weighted_terms,
    *,
    scale_source="packed_boundary_tensor_weighted_scale",
    sum_source="packed_boundary_tensor_weighted_sum",
):
    """Return a weighted packed tensor sum, or ``None`` for unsupported inputs."""

    weighted_terms = tuple(weighted_terms or ())
    if not weighted_terms:
        return None
    data = OrderedDict()
    dirs = None
    qns = None
    for tensor, factor in weighted_terms:
        if not is_abelian_packed_boundary_tensor(tensor):
            return None
        keys, blocks, tensor_dirs, tensor_qns = abelian_packed_tensor_items(tensor)
        if dirs is None and tensor_dirs:
            dirs = list(tensor_dirs)
        if qns is None and tensor_qns is not None:
            qns = tensor_qns
        scalar = complex(factor)
        for key, block in zip(keys, blocks):
            key = tuple(key)
            block = np.asarray(block)
            old = data.get(key)
            if old is None:
                if scalar == 1.0:
                    data[key] = block.copy()
                elif scalar == -1.0:
                    data[key] = -block
                else:
                    data[key] = scalar * block
            else:
                if tuple(old.shape) != tuple(block.shape):
                    return None
                if scalar == 1.0:
                    old += block
                elif scalar == -1.0:
                    old -= block
                else:
                    old += scalar * block
    return AbelianPackedBoundaryTensor(
        tuple(data.keys()),
        tuple(data.values()),
        dirs=[] if dirs is None else dirs,
        qns=qns,
        source=sum_source,
        assume_unique=True,
    )


def prune_abelian_packed_boundary_tensor(
    tensor,
    *,
    zero_tol=0.0,
    source="packed_boundary_tensor_pruned",
):
    """Drop numerically zero packed blocks while preserving the packed layout."""

    if tensor is None:
        return None
    if not is_abelian_packed_boundary_tensor(tensor):
        return tensor
    kept_keys = []
    kept_blocks = []
    tol = float(zero_tol)
    for key, block in zip(tensor.keys, tensor.blocks):
        arr = np.asarray(block)
        if arr.size and float(np.linalg.norm(arr.reshape(-1))) <= tol:
            continue
        kept_keys.append(tuple(key))
        kept_blocks.append(arr)
    return AbelianPackedBoundaryTensor(
        tuple(kept_keys),
        tuple(kept_blocks),
        dirs=list(getattr(tensor, "dirs", ())),
        qns=getattr(tensor, "qns", None),
        source=source,
        assume_unique=True,
    )


def packed_same_side_p_product_correction(
    product,
    exact,
    *,
    zero_tol=1.0e-13,
    correction_source="packed_same_side_p_correction",
    source="packed_same_side_p_product_plus_correction",
):
    """Return ``product + (exact - product)`` for same-side P boundary channels.

    Renormalized generator products are only exact before projection/truncation.
    The residual ``exact - product`` is the projected-boundary correction channel.
    """

    if product is None or exact is None:
        return None, None
    if not (
        is_abelian_packed_boundary_tensor(product)
        and is_abelian_packed_boundary_tensor(exact)
    ):
        return None, None
    correction = sum_abelian_packed_boundary_terms(
        (
            (exact, 1.0),
            (product, -1.0),
        ),
        scale_source=f"{correction_source}_scale",
        sum_source=correction_source,
    )
    correction = prune_abelian_packed_boundary_tensor(
        correction,
        zero_tol=zero_tol,
        source=correction_source,
    )
    corrected = sum_abelian_packed_boundary_terms(
        (
            (product, 1.0),
            (correction, 1.0),
        ),
        scale_source=f"{source}_scale",
        sum_source=source,
    )
    corrected = prune_abelian_packed_boundary_tensor(
        corrected,
        zero_tol=zero_tol,
        source=source,
    )
    return corrected, correction


def compose_abelian_packed_boundary_operators(
    first,
    second,
    *,
    reverse=False,
    source="packed_boundary_operator_compose",
    record_failure=None,
):
    """Compose two rank-3 packed boundary operators without a legacy tensor."""

    def fail(reason):
        if callable(record_failure):
            record_failure(str(reason))
        return None

    def add_flux(left, right):
        try:
            return left + right
        except TypeError:
            return NotImplemented

    if not (
        is_abelian_packed_boundary_tensor(first)
        and is_abelian_packed_boundary_tensor(second)
    ):
        return fail("not_packed")
    if (
        len(getattr(first, "dirs", ())) != 3
        or len(getattr(second, "dirs", ())) != 3
    ):
        return fail("rank")
    if tuple(getattr(first, "dirs", ())) != tuple(getattr(second, "dirs", ())):
        return fail("dirs")

    right_by_bra = {}
    right_by_ket = {}
    for key_b, block_b in zip(getattr(second, "keys", ()), getattr(second, "blocks", ())):
        if len(key_b) != 3:
            return fail("right_key_rank")
        arr_b = np.asarray(block_b)
        if arr_b.ndim != 3 or arr_b.shape[0] != 1:
            return fail("right_shape")
        right_by_bra.setdefault(key_b[1], []).append((key_b, arr_b))
        right_by_ket.setdefault(key_b[2], []).append((key_b, arr_b))

    out = OrderedDict()
    qn_sets = [set() for _ in range(3)]
    for key_a, block_a in zip(getattr(first, "keys", ()), getattr(first, "blocks", ())):
        if len(key_a) != 3:
            return fail("left_key_rank")
        arr_a = np.asarray(block_a)
        if arr_a.ndim != 3 or arr_a.shape[0] != 1:
            return fail("left_shape")
        candidates = (
            right_by_ket.get(key_a[1], ())
            if bool(reverse)
            else right_by_bra.get(key_a[2], ())
        )
        for key_b, arr_b in candidates:
            try:
                if bool(reverse):
                    product = arr_b[0] @ arr_a[0]
                    out_flux = add_flux(key_b[0], key_a[0])
                    out_key = (out_flux, key_b[1], key_a[2])
                else:
                    product = arr_a[0] @ arr_b[0]
                    out_flux = add_flux(key_a[0], key_b[0])
                    out_key = (out_flux, key_a[1], key_b[2])
            except ValueError:
                return fail("matmul_shape")
            if out_key[0] is NotImplemented:
                return fail("flux")
            for axis, qn in enumerate(out_key):
                qn_sets[axis].add(qn)
            out_block = product.reshape(1, product.shape[0], product.shape[1])
            old = out.get(out_key)
            if old is not None:
                if old.shape != out_block.shape:
                    return fail("output_shape")
                out[out_key] = old + out_block
            else:
                out[out_key] = out_block.copy()
    if not out:
        return fail("empty")
    return AbelianPackedBoundaryTensor(
        tuple(out.keys()),
        tuple(out.values()),
        dirs=list(getattr(first, "dirs", ())),
        qns=[sorted(qns) for qns in qn_sets],
        source=source,
    )


def _abelian_direct_stored_blocks_for(obj):
    if obj is None:
        return 0
    if is_abelian_packed_boundary_tensor(obj):
        return int(len(obj))
    if isinstance(obj, AbelianPackedIdentityLocalEntry):
        return (
            _abelian_direct_stored_blocks_for(obj.E)
            + _abelian_direct_stored_blocks_for(obj.F)
        )
    if isinstance(obj, AbelianPackedLocalGeneratorEntry):
        return (
            _abelian_direct_stored_blocks_for(obj.E)
            + _abelian_direct_stored_blocks_for(obj.W_left)
            + _abelian_direct_stored_blocks_for(obj.W_right)
            + _abelian_direct_stored_blocks_for(obj.F)
        )
    if isinstance(obj, AbelianPackedDirectFamilyEntries):
        return int(sum(_abelian_direct_stored_blocks_for(entry) for entry in obj))
    if isinstance(obj, AbelianNativeExactPatternFamilyEntries):
        return _abelian_direct_stored_blocks_for(obj.entries)
    if isinstance(obj, (tuple, list)):
        if len(obj) == 3 and not isinstance(obj[0], (tuple, list)):
            E, W_pair, F = obj
            return (
                _abelian_direct_stored_blocks_for(E)
                + sum(_abelian_direct_stored_blocks_for(item) for item in tuple(W_pair or ()))
                + _abelian_direct_stored_blocks_for(F)
            )
        return int(sum(_abelian_direct_stored_blocks_for(item) for item in obj))
    data = getattr(obj, "data", None)
    if data is not None:
        return int(len(data))
    return 0


def _abelian_direct_stored_elements_for(obj):
    if obj is None:
        return 0
    if is_abelian_packed_boundary_tensor(obj):
        return int(sum(int(np.asarray(block).size) for block in obj.blocks))
    if isinstance(obj, AbelianPackedIdentityLocalEntry):
        return (
            _abelian_direct_stored_elements_for(obj.E)
            + _abelian_direct_stored_elements_for(obj.F)
        )
    if isinstance(obj, AbelianPackedLocalGeneratorEntry):
        return (
            _abelian_direct_stored_elements_for(obj.E)
            + _abelian_direct_stored_elements_for(obj.W_left)
            + _abelian_direct_stored_elements_for(obj.W_right)
            + _abelian_direct_stored_elements_for(obj.F)
        )
    if isinstance(obj, AbelianPackedDirectFamilyEntries):
        return int(sum(_abelian_direct_stored_elements_for(entry) for entry in obj))
    if isinstance(obj, AbelianNativeExactPatternFamilyEntries):
        return _abelian_direct_stored_elements_for(obj.entries)
    if isinstance(obj, (tuple, list)):
        if len(obj) == 3 and not isinstance(obj[0], (tuple, list)):
            E, W_pair, F = obj
            return (
                _abelian_direct_stored_elements_for(E)
                + sum(_abelian_direct_stored_elements_for(item) for item in tuple(W_pair or ()))
                + _abelian_direct_stored_elements_for(F)
            )
        return int(sum(_abelian_direct_stored_elements_for(item) for item in obj))
    data = getattr(obj, "data", None)
    if data is not None:
        return int(sum(int(np.asarray(block).size) for block in data.values()))
    return 0


@dataclass(frozen=True)
class AbelianNativeGeneratorOperatorTable:
    """Abelian-native renormalized spin-free generator operators for one boundary."""

    side: str
    bond: int
    operators: dict
    source: str = "abelian_native_spinfree_generator_boundary_table"
    build_seconds: float = 0.0

    @property
    def n_operators(self):
        return int(len(self.operators))

    @property
    def stored_blocks(self):
        return int(
            sum(_abelian_direct_stored_blocks_for(op) for op in self.operators.values())
        )

    @property
    def stored_elements(self):
        return int(
            sum(_abelian_direct_stored_elements_for(op) for op in self.operators.values())
        )

    @property
    def stats(self):
        return {
            "kind": "abelian_native_generator_operator_table",
            "source": str(self.source),
            "side": str(self.side),
            "bond": int(self.bond),
            "n_operators": int(self.n_operators),
            "stored_blocks": int(self.stored_blocks),
            "stored_elements": int(self.stored_elements),
            "build_seconds": float(self.build_seconds),
            "operator_keys": tuple(
                tuple(int(index) for index in key)
                for key in sorted(self.operators)
            ),
        }


@dataclass
class AbelianNativePairBoundaryOperatorTable:
    """Abelian-native table for validated P entries and same-side P operators."""

    side: str
    bond: int
    revision: int = -1
    entries: dict = field(default_factory=dict)
    operators: dict = field(default_factory=dict)
    source: str = "abelian_native_pair_boundary_operator_table"
    build_seconds: float = 0.0
    validated_terms: int = 0
    rejected_terms: int = 0
    resets: int = 0

    def get_operator(self, key):
        return self.operators.get(tuple(key))

    def add_operator(self, key, operator):
        self.operators[tuple(key)] = operator
        return operator

    def get(self, key):
        return self.entries.get(tuple(key))

    def add(self, key, entries):
        entries = tuple(entries or ())
        self.entries[tuple(key)] = entries
        self.validated_terms += 1
        return entries

    def reject(self):
        self.rejected_terms += 1

    def reset_for_revision(self, revision):
        revision = int(revision)
        if int(self.revision) == revision:
            return False
        self.revision = revision
        self.entries.clear()
        self.operators.clear()
        self.validated_terms = 0
        self.rejected_terms = 0
        self.build_seconds = 0.0
        for attr in (
            "_pyqed_same_side_pairs_prebuilt",
            "_pyqed_same_side_route_columns",
        ):
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        self.resets += 1
        return True

    @property
    def n_terms(self):
        return int(len(self.entries))

    @property
    def n_operators(self):
        return int(len(self.operators))

    @property
    def n_entries(self):
        return int(sum(len(entries) for entries in self.entries.values()))

    @property
    def stored_blocks(self):
        return int(
            sum(_abelian_direct_stored_blocks_for(entries) for entries in self.entries.values())
            + sum(_abelian_direct_stored_blocks_for(op) for op in self.operators.values())
        )

    @property
    def stored_elements(self):
        return int(
            sum(_abelian_direct_stored_elements_for(entries) for entries in self.entries.values())
            + sum(_abelian_direct_stored_elements_for(op) for op in self.operators.values())
        )

    @property
    def stats(self):
        return {
            "kind": "abelian_native_pair_boundary_operator_table",
            "source": str(self.source),
            "side": str(self.side),
            "bond": int(self.bond),
            "revision": int(self.revision),
            "n_terms": int(self.n_terms),
            "n_operators": int(self.n_operators),
            "n_entries": int(self.n_entries),
            "stored_blocks": int(self.stored_blocks),
            "stored_elements": int(self.stored_elements),
            "build_seconds": float(self.build_seconds),
            "validated_terms": int(self.validated_terms),
            "rejected_terms": int(self.rejected_terms),
            "resets": int(self.resets),
            "operator_keys": tuple(
                tuple(int(index) for index in key)
                for key in sorted(set(self.entries) | set(self.operators))
            ),
        }


@dataclass
class AbelianNativeExactPatternOperatorTable:
    """Abelian exact JW-pattern boundary table for direct-family actions."""

    side: str
    bond: int
    entries: dict = field(default_factory=dict)
    family_counts: dict = field(default_factory=dict)
    source: str = "abelian_native_exact_jw_pattern_boundary_table"
    build_seconds: float = 0.0
    hits: int = 0
    misses: int = 0
    batch_resolves: int = 0
    batch_stores: int = 0
    cpp_resolves: int = 0
    cpp_stores: int = 0
    evictions: int = 0

    def get(self, key):
        return self.entries.get(key)

    @staticmethod
    def normalize_key(key):
        return (tuple(key[0]), str(key[1]))

    def put(self, key, value, family_name=None):
        key = self.normalize_key(key)
        is_new = key not in self.entries
        self.entries[key] = value
        if is_new and family_name is not None:
            name = str(family_name)
            self.family_counts[name] = int(self.family_counts.get(name, 0)) + 1
        return value

    def resolve_many(self, keys, *, normalized=False):
        keys = tuple(() if keys is None else keys)
        if not bool(normalized):
            keys = tuple(self.normalize_key(key) for key in keys)
        kernel = _cpp_table_kernel("dict_resolve_values_many")
        if kernel is not None:
            try:
                values, missing, missing_positions, hits, misses = kernel(
                    self.entries,
                    keys,
                )
                hits = int(hits)
                misses = int(misses)
                self.batch_resolves += 1
                self.hits += hits
                self.misses += misses
                self.cpp_resolves += 1
                return values, missing, missing_positions, hits, misses
            except Exception:
                pass
        values = [None] * len(keys)
        missing = []
        missing_positions = []
        hits = 0
        misses = 0
        for idx, key in enumerate(keys):
            value = self.entries.get(key)
            if value is None:
                missing.append(key)
                missing_positions.append(idx)
                misses += 1
            else:
                values[idx] = value
                hits += 1
        self.batch_resolves += 1
        self.hits += int(hits)
        self.misses += int(misses)
        return values, tuple(missing), tuple(missing_positions), int(hits), int(misses)

    def put_many(self, keys, values, *, family_name=None, normalized=False):
        keys = tuple(() if keys is None else keys)
        values = tuple(() if values is None else values)
        if not bool(normalized):
            keys = tuple(self.normalize_key(key) for key in keys)
        kernel = _cpp_table_kernel("dict_put_many_values")
        if kernel is not None:
            try:
                stored = kernel(
                    self.entries,
                    keys,
                    values,
                    self.family_counts,
                    None if family_name is None else str(family_name),
                )
                self.batch_stores += 1
                self.cpp_stores += 1
                return int(stored)
            except Exception:
                pass
        stored = 0
        for key, value in zip(keys, values):
            is_new = key not in self.entries
            self.entries[key] = value
            if is_new:
                stored += 1
                if family_name is not None:
                    name = str(family_name)
                    self.family_counts[name] = (
                        int(self.family_counts.get(name, 0)) + 1
                    )
        self.batch_stores += 1
        return int(stored)

    def discard(self, key, *, normalized=False):
        key = self.normalize_key(key) if not bool(normalized) else key
        if key not in self.entries:
            return False
        self.entries.pop(key, None)
        self.evictions += 1
        return True

    @property
    def n_entries(self):
        return int(len(self.entries))

    @property
    def stored_blocks(self):
        return int(
            sum(_abelian_direct_stored_blocks_for(value) for value in self.entries.values())
        )

    @property
    def stored_elements(self):
        return int(
            sum(
                _abelian_direct_stored_elements_for(value)
                for value in self.entries.values()
            )
        )

    @property
    def stats(self):
        return {
            "kind": "abelian_native_exact_pattern_operator_table",
            "source": str(self.source),
            "side": str(self.side),
            "bond": int(self.bond),
            "n_entries": int(self.n_entries),
            "stored_blocks": int(self.stored_blocks),
            "stored_elements": int(self.stored_elements),
            "build_seconds": float(self.build_seconds),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "batch_resolves": int(self.batch_resolves),
            "batch_stores": int(self.batch_stores),
            "cpp_resolves": int(self.cpp_resolves),
            "cpp_stores": int(self.cpp_stores),
            "evictions": int(self.evictions),
            "family_counts": {
                str(name): int(count)
                for name, count in sorted(self.family_counts.items())
            },
        }


@dataclass(frozen=True)
class AbelianNativeExactPatternFamilyEntries:
    """Iterable exact-pattern family entries plus grouping metadata."""

    family_name: str
    entries: tuple
    entry_groups: tuple = ()
    group_keys: tuple = ()
    source: str = "abelian_native_exact_pattern_family_entries"

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return int(len(self.entries))

    def __bool__(self):
        return bool(self.entries)

    def __getitem__(self, index):
        return self.entries[index]

    @property
    def n_groups(self):
        return int(len(self.entry_groups))

    @property
    def n_group_entries(self):
        return int(sum(len(group) for group in self.entry_groups))

    @property
    def stats(self):
        return {
            "kind": "abelian_native_exact_pattern_family_entries",
            "source": str(self.source),
            "family_name": str(self.family_name),
            "n_entries": int(len(self.entries)),
            "n_groups": int(self.n_groups),
            "n_group_entries": int(self.n_group_entries),
            "entry_reduction": int(self.n_group_entries - len(self.entries)),
            "group_sizes": tuple(int(len(group)) for group in self.entry_groups),
            "group_keys": tuple(repr(key) for key in self.group_keys),
        }


@dataclass
class AbelianNativeExactPatternComponentTable:
    """Center-bond Abelian exact JW-pattern component table."""

    bond: int
    families: dict = field(default_factory=dict)
    family_records: dict = field(default_factory=dict)
    source: str = "abelian_native_exact_jw_pattern_component_table"
    build_seconds: float = 0.0

    def get_family_records(self, family_name):
        return self.family_records.get(str(family_name))

    def put_family_records(self, family_name, records):
        records = tuple(records or ())
        self.family_records[str(family_name)] = records
        return records

    def get_family(self, family_name):
        return self.families.get(str(family_name))

    def put_family(
        self,
        family_name,
        entries,
        records=None,
        *,
        compression_policy="auto",
        min_reduction=1,
        max_group_size=None,
    ):
        entries_is_packed = bool(
            getattr(entries, "_pyqed_packed_direct_family_entries", False)
        )
        entries = entries if entries_is_packed else tuple(entries or ())
        if bool(getattr(entries, "_pyqed_planned_direct_family_table_ids", False)):
            snapshot = getattr(entries, "snapshot_table_payloads", None)
            if callable(snapshot):
                entries = snapshot()
        if records is None:
            records = self.family_records.get(str(family_name), ())
        records = tuple(records or ())
        policy = str(compression_policy or "auto").lower().replace("-", "_")
        if policy in {"off", "false", "disabled", "uncompressed"}:
            policy = "none"
        if policy not in {"none", "auto", "structural"}:
            policy = "auto"
        if (
            policy == "auto"
            and max_group_size is not None
            and int(max_group_size) <= 1
        ):
            family_entries = AbelianNativeExactPatternFamilyEntries(
                family_name=str(family_name),
                entries=entries,
            )
            self.families[str(family_name)] = family_entries
            return family_entries

        grouped = {}
        entry_items = tuple(entries or ())
        for index, entry in enumerate(entry_items):
            key = (
                self._record_boundary_pair(records[index])
                if index < len(records)
                else ((), ())
            )
            grouped.setdefault(key, []).append(entry)

        def _direct_sum_w_pair(group):
            if len(group) <= 1:
                return tuple(group)
            parsed = []
            mid_dims = {}
            for entry_index, entry in enumerate(group):
                try:
                    E_term, W_pair, F_term = entry
                    W_left, W_right = W_pair
                except Exception:
                    return tuple(group)
                left_data = getattr(W_left, "data", {}) or {}
                right_data = getattr(W_right, "data", {}) or {}
                left_mid_dims = {}
                right_mid_dims = {}
                for key, block in left_data.items():
                    q_mid = key[1]
                    dim = int(np.asarray(block).shape[1])
                    old = left_mid_dims.get(q_mid)
                    if old is not None and int(old) != int(dim):
                        return tuple(group)
                    left_mid_dims[q_mid] = int(dim)
                for key, block in right_data.items():
                    q_mid = key[0]
                    dim = int(np.asarray(block).shape[0])
                    old = right_mid_dims.get(q_mid)
                    if old is not None and int(old) != int(dim):
                        return tuple(group)
                    right_mid_dims[q_mid] = int(dim)
                mids = set(left_mid_dims).intersection(right_mid_dims)
                if not mids:
                    return tuple(group)
                for q_mid in mids:
                    left_dim = int(left_mid_dims[q_mid])
                    right_dim = int(right_mid_dims[q_mid])
                    if left_dim != right_dim:
                        return tuple(group)
                    mid_dims[(entry_index, q_mid)] = int(left_dim)
                parsed.append((E_term, W_left, W_right, F_term, tuple(sorted(mids))))

            E_term = parsed[0][0]
            F_term = parsed[0][3]
            mid_sectors = tuple(sorted({q for item in parsed for q in item[4]}))
            offsets = {}
            totals = {q_mid: 0 for q_mid in mid_sectors}
            for entry_index, _item in enumerate(parsed):
                for q_mid in mid_sectors:
                    dim = int(mid_dims.get((entry_index, q_mid), 0))
                    offsets[(entry_index, q_mid)] = totals[q_mid]
                    totals[q_mid] += dim
            try:
                W_left_qns = [
                    sorted({q for _E, W_left, _W_right, _F, _mids in parsed for q in W_left.qns[0]}),
                    list(mid_sectors),
                    sorted({q for _E, W_left, _W_right, _F, _mids in parsed for q in W_left.qns[2]}),
                    sorted({q for _E, W_left, _W_right, _F, _mids in parsed for q in W_left.qns[3]}),
                ]
                W_right_qns = [
                    list(mid_sectors),
                    sorted({q for _E, _W_left, W_right, _F, _mids in parsed for q in W_right.qns[1]}),
                    sorted({q for _E, _W_left, W_right, _F, _mids in parsed for q in W_right.qns[2]}),
                    sorted({q for _E, _W_left, W_right, _F, _mids in parsed for q in W_right.qns[3]}),
                ]
            except Exception:
                return tuple(group)

            left_data = {}
            right_data = {}

            def _accumulate_block(data, key, block, axis, start, total_dim):
                shape = list(block.shape)
                shape[axis] = int(total_dim)
                if key not in data:
                    data[key] = np.zeros(shape, dtype=np.asarray(block).dtype)
                elif tuple(data[key].shape) != tuple(shape):
                    return False
                slices = [slice(None)] * len(shape)
                slices[axis] = slice(int(start), int(start) + int(block.shape[axis]))
                data[key][tuple(slices)] = block
                return True

            for entry_index, (_E, W_left, W_right, _F, _mids) in enumerate(parsed):
                for key, block in W_left.data.items():
                    q_mid = key[1]
                    if q_mid not in totals:
                        continue
                    if not _accumulate_block(
                        left_data,
                        key,
                        np.asarray(block),
                        1,
                        offsets[(entry_index, q_mid)],
                        totals[q_mid],
                    ):
                        return tuple(group)
                for key, block in W_right.data.items():
                    q_mid = key[0]
                    if q_mid not in totals:
                        continue
                    if not _accumulate_block(
                        right_data,
                        key,
                        np.asarray(block),
                        0,
                        offsets[(entry_index, q_mid)],
                        totals[q_mid],
                    ):
                        return tuple(group)
            if not left_data or not right_data:
                return tuple(group)
            try:
                W_class = parsed[0][1].__class__
                W_left = W_class(left_data, W_left_qns, parsed[0][1].dirs[:])
                W_right = W_class(right_data, W_right_qns, parsed[0][2].dirs[:])
            except Exception:
                return tuple(group)
            return ((E_term, [W_left, W_right], F_term),)

        def _compressed_group(group):
            group = tuple(group)
            if len(group) <= 1:
                return group
            if (
                policy == "auto"
                and max_group_size is not None
                and int(len(group)) > int(max_group_size)
            ):
                return group
            return _direct_sum_w_pair(group)

        if policy == "none":
            group_items = tuple(
                (key, tuple(grouped[key]))
                for key in sorted(grouped, key=repr)
            )
        else:
            group_items = tuple(
                (
                    key,
                    _compressed_group(grouped[key]),
                )
                for key in sorted(grouped, key=repr)
            )
        compressed_entries = tuple(
            entry
            for _key, group in group_items
            for entry in group
        )
        reduction = int(len(entries) - len(compressed_entries))
        largest_group = max((len(group) for _key, group in group_items), default=0)
        if (
            policy == "auto"
            and (
                reduction < int(min_reduction)
                or (
                    max_group_size is not None
                    and int(largest_group) > int(max_group_size)
                )
            )
        ):
            group_items = tuple(
                (key, tuple(grouped[key]))
                for key in sorted(grouped, key=repr)
            )
            compressed_entries = entry_items
        family_entries = AbelianNativeExactPatternFamilyEntries(
            family_name=str(family_name),
            entries=compressed_entries,
            entry_groups=tuple(group for _key, group in group_items),
            group_keys=tuple(key for key, _group in group_items),
        )
        self.families[str(family_name)] = family_entries
        return family_entries

    @property
    def n_families(self):
        return int(len(self.families))

    @property
    def n_entries(self):
        return int(sum(len(entries) for entries in self.families.values()))

    @property
    def n_records(self):
        return int(sum(len(records) for records in self.family_records.values()))

    @staticmethod
    def _record_local_pair(record):
        try:
            return (str(record[1]), str(record[2]))
        except Exception:
            return ("?", "?")

    @staticmethod
    def _record_boundary_pair(record):
        try:
            return (tuple(record[0]), tuple(record[3]))
        except Exception:
            return ((), ())

    def _record_group_counts(self, indexer):
        out = {}
        for name, records in self.family_records.items():
            groups = {}
            for record in records:
                key = indexer(record)
                groups[key] = int(groups.get(key, 0)) + 1
            out[str(name)] = {
                repr(key): int(count)
                for key, count in sorted(groups.items(), key=lambda item: repr(item[0]))
            }
        return out

    @property
    def stored_blocks(self):
        return int(
            sum(_abelian_direct_stored_blocks_for(entries) for entries in self.families.values())
        )

    @property
    def stored_elements(self):
        return int(
            sum(
                _abelian_direct_stored_elements_for(entries)
                for entries in self.families.values()
            )
        )

    @property
    def stats(self):
        return {
            "kind": "abelian_native_exact_pattern_component_table",
            "source": str(self.source),
            "bond": int(self.bond),
            "n_families": int(self.n_families),
            "n_records": int(self.n_records),
            "n_entries": int(self.n_entries),
            "stored_blocks": int(self.stored_blocks),
            "stored_elements": int(self.stored_elements),
            "build_seconds": float(self.build_seconds),
            "record_counts": {
                str(name): int(len(records))
                for name, records in sorted(self.family_records.items())
            },
            "family_counts": {
                str(name): int(len(entries))
                for name, entries in sorted(self.families.items())
            },
            "record_boundary_counts": self._record_group_counts(
                self._record_boundary_pair
            ),
            "record_local_counts": self._record_group_counts(self._record_local_pair),
        }


def abelian_packed_tensor_items(tensor, *, conj=False):
    """Return columnar ``(keys, blocks, dirs, qns)`` for packed or legacy tensors."""

    if is_abelian_packed_boundary_tensor(tensor):
        keys = tuple(tensor.keys)
        blocks = tuple(tensor.blocks)
        dirs = list(getattr(tensor, "dirs", ()))
        qns = getattr(tensor, "qns", None)
    else:
        data = getattr(tensor, "data", {}) or {}
        keys = tuple(data.keys())
        blocks = tuple(data.values())
        dirs = list(getattr(tensor, "dirs", ()))
        qns = getattr(tensor, "qns", None)
    if conj:
        blocks = tuple(np.asarray(block).conj() for block in blocks)
        dirs = [-int(d) for d in dirs]
    return keys, blocks, dirs, qns


def conjugate_abelian_packed_boundary_tensor(
    tensor,
    *,
    source="packed_boundary_tensor_conj",
):
    keys, blocks, dirs, qns = abelian_packed_tensor_items(tensor, conj=True)
    return AbelianPackedBoundaryTensor(
        keys,
        blocks,
        dirs=dirs,
        qns=qns,
        source=source,
        assume_unique=True,
    )


def transpose_abelian_packed_boundary_tensor(
    tensor,
    axes,
    *,
    source="packed_boundary_tensor_transpose",
):
    axes = tuple(int(axis) for axis in axes)
    keys, blocks, dirs, qns = abelian_packed_tensor_items(tensor)
    return AbelianPackedBoundaryTensor(
        tuple(tuple(key[axis] for axis in axes) for key in keys),
        tuple(np.asarray(block).transpose(axes) for block in blocks),
        dirs=[dirs[axis] for axis in axes],
        qns=None if qns is None else [qns[axis] for axis in axes],
        source=source,
    )


def tensordot_abelian_packed_boundary_tensors(
    left,
    right,
    axes,
    *,
    right_axis_map=None,
    source="packed_boundary_tensor_tensordot",
):
    """Packed blockwise tensor contraction without materializing ``.data`` dicts."""

    left_axes, right_axes = axes
    if isinstance(left_axes, int):
        left_axes = [left_axes]
    if isinstance(right_axes, int):
        right_axes = [right_axes]
    left_axes = tuple(int(axis) for axis in left_axes)
    right_axes = tuple(int(axis) for axis in right_axes)
    left_keys, left_blocks, left_dirs, left_qns = abelian_packed_tensor_items(left)
    right_keys, right_blocks, right_dirs, right_qns = abelian_packed_tensor_items(right)
    left_rank = len(left_dirs)
    right_rank = len(right_dirs)
    free_left = tuple(axis for axis in range(left_rank) if axis not in left_axes)
    free_right = tuple(axis for axis in range(right_rank) if axis not in right_axes)
    right_map = (
        right_axis_map
        if right_axis_map is not None
        else abelian_packed_tensor_axis_map(right, right_axes)
    )
    out = OrderedDict()
    for left_key, left_block in zip(left_keys, left_blocks):
        contract_key = tuple(left_key[axis] for axis in left_axes)
        for right_key, right_block in right_map.get(contract_key, ()):
            out_key = (
                tuple(left_key[axis] for axis in free_left)
                + tuple(right_key[axis] for axis in free_right)
            )
            block = np.tensordot(left_block, right_block, axes=(left_axes, right_axes))
            if out_key in out:
                out[out_key] = out[out_key] + block
            else:
                out[out_key] = block
    qns = None
    if left_qns is not None and right_qns is not None:
        qns = [left_qns[axis] for axis in free_left] + [
            right_qns[axis] for axis in free_right
        ]
    return AbelianPackedBoundaryTensor(
        tuple(out.keys()),
        tuple(out.values()),
        dirs=[left_dirs[axis] for axis in free_left]
        + [right_dirs[axis] for axis in free_right],
        qns=qns,
        source=source,
        assume_unique=True,
    )


def abelian_packed_tensor_axis_map(tensor, axes):
    """Map selected packed sector-key axes to matching blocks."""

    if isinstance(axes, int):
        axes = [axes]
    axes = tuple(int(axis) for axis in axes)
    keys, blocks, _dirs, _qns = abelian_packed_tensor_items(tensor)
    mapped = defaultdict(list)
    for key, block in zip(keys, blocks):
        mapped[tuple(key[axis] for axis in axes)].append((key, block))
    return mapped


def abelian_packed_tensor_axis_qns(tensor, axis):
    """Return sector labels present on one packed tensor axis."""

    axis = int(axis)
    qns = getattr(tensor, "qns", None)
    if qns is not None and axis < len(qns):
        return tuple(qns[axis])
    keys, _blocks, _dirs, _qns = abelian_packed_tensor_items(tensor)
    if not keys:
        return ()
    return tuple(sorted({key[axis] for key in keys if axis < len(key)}))


def make_abelian_packed_local_generator_pair(
    W_left,
    W_right,
    *,
    left_axis=1,
    right_axis=0,
    left_source="direct_family_local_generator_W_left_common",
    right_source="direct_family_local_generator_W_right_common",
):
    """Restrict a packed local-generator W pair to common virtual sectors."""

    if W_left is None or W_right is None:
        return None
    common = set(abelian_packed_tensor_axis_qns(W_left, left_axis)).intersection(
        abelian_packed_tensor_axis_qns(W_right, right_axis)
    )
    if not common:
        return None
    common = tuple(sorted(common))
    W_left = filter_abelian_packed_boundary_tensor_axis(
        W_left,
        left_axis,
        common,
        source=left_source,
    )
    W_right = filter_abelian_packed_boundary_tensor_axis(
        W_right,
        right_axis,
        common,
        source=right_source,
    )
    if not W_left or not W_right:
        return None
    return W_left, W_right, common


def make_abelian_packed_site_operator_from_left(
    local_entries,
    phys_qns,
    left_qns,
    *,
    source="direct_family_site_operator_left",
):
    """Build a packed local operator with known left virtual sectors."""

    left_qns = tuple(left_qns)
    cpp_payload = _cpp_table_kernel("packed_site_operator_from_left_payload")
    if cpp_payload is not None:
        try:
            keys, blocks, qns, dirs = cpp_payload(
                tuple(local_entries or ()),
                tuple(phys_qns or ()),
                left_qns,
            )
            if keys is None:
                return None
            _PACKED_LOCAL_PAYLOAD_STATS["left_site_operator"] += 1
            return AbelianPackedBoundaryTensor(
                tuple(keys),
                tuple(blocks),
                qns=[list(axis) for axis in qns],
                dirs=list(dirs),
                source=source,
                assume_unique=True,
            )
        except Exception:
            _PACKED_LOCAL_PAYLOAD_STATS["left_site_operator_failures"] += 1
            pass
    keys = []
    blocks = []
    right_qns = set()
    for q_left in left_qns:
        for q_out, q_in, flux, coeff in tuple(local_entries or ()):
            q_right = q_left - flux
            right_qns.add(q_right)
            keys.append((q_left, q_right, q_out, q_in))
            block = np.empty((1, 1, 1, 1), dtype=complex)
            block[0, 0, 0, 0] = complex(coeff)
            blocks.append(block)
    if not keys:
        return None
    return AbelianPackedBoundaryTensor(
        tuple(keys),
        tuple(blocks),
        qns=[list(left_qns), sorted(right_qns), list(phys_qns), list(phys_qns)],
        dirs=[-1, 1, 1, -1],
        source=source,
        assume_unique=True,
    )


def make_abelian_packed_site_operator_from_right(
    local_entries,
    phys_qns,
    right_qns,
    *,
    source="direct_family_site_operator_right",
):
    """Build a packed local operator with known right virtual sectors."""

    right_qns = tuple(right_qns)
    cpp_payload = _cpp_table_kernel("packed_site_operator_from_right_payload")
    if cpp_payload is not None:
        try:
            keys, blocks, qns, dirs = cpp_payload(
                tuple(local_entries or ()),
                tuple(phys_qns or ()),
                right_qns,
            )
            if keys is None:
                return None
            _PACKED_LOCAL_PAYLOAD_STATS["right_site_operator"] += 1
            return AbelianPackedBoundaryTensor(
                tuple(keys),
                tuple(blocks),
                qns=[list(axis) for axis in qns],
                dirs=list(dirs),
                source=source,
                assume_unique=True,
            )
        except Exception:
            _PACKED_LOCAL_PAYLOAD_STATS["right_site_operator_failures"] += 1
            pass
    keys = []
    blocks = []
    left_qns = set()
    for q_right in right_qns:
        for q_out, q_in, flux, coeff in tuple(local_entries or ()):
            q_left = q_right + flux
            left_qns.add(q_left)
            keys.append((q_left, q_right, q_out, q_in))
            block = np.empty((1, 1, 1, 1), dtype=complex)
            block[0, 0, 0, 0] = complex(coeff)
            blocks.append(block)
    if not keys:
        return None
    return AbelianPackedBoundaryTensor(
        tuple(keys),
        tuple(blocks),
        qns=[sorted(left_qns), list(right_qns), list(phys_qns), list(phys_qns)],
        dirs=[-1, 1, 1, -1],
        source=source,
        assume_unique=True,
    )


@dataclass
class AbelianSpatialLocalOperatorBuilder:
    """Build and cache Abelian packed spatial local operators."""

    site_qn_maps: object
    local_ops: object = None
    local_ops_factory: object = None
    zero_tol: float = 1.0e-14
    source_prefix: str = "direct_family"
    _site_phys_cache: dict = field(default_factory=dict, init=False)
    _local_piece_entries_cache: dict = field(default_factory=dict, init=False)
    _packed_site_operator_cache: dict = field(default_factory=dict, init=False)

    def _ops(self):
        if self.local_ops is None:
            if self.local_ops_factory is None:
                raise ValueError("local_ops or local_ops_factory is required")
            self.local_ops = self.local_ops_factory()
        return self.local_ops

    def site_phys_data(self, site):
        site = int(site)
        cached = self._site_phys_cache.get(site)
        if cached is not None:
            return cached
        phys_items = tuple(sorted(self.site_qn_maps[site].items()))
        phys_qns = tuple(sorted({qn for _state, qn in phys_items}))
        cached = (phys_items, phys_qns)
        self._site_phys_cache[site] = cached
        return cached

    def local_piece_entries(self, piece, site):
        key = (str(piece), int(site))
        cached = self._local_piece_entries_cache.get(key)
        if cached is not None:
            return cached
        mat = np.asarray(self._ops()[str(piece)], dtype=complex)
        phys_items, phys_qns = self.site_phys_data(site)
        entries = []
        for out_s, q_out in phys_items:
            for in_s, q_in in phys_items:
                coeff = complex(mat[int(out_s), int(in_s)])
                if abs(coeff) <= float(self.zero_tol):
                    continue
                entries.append((q_out, q_in, q_out - q_in, coeff))
        cached = (tuple(entries), phys_qns)
        self._local_piece_entries_cache[key] = cached
        return cached

    def packed_site_operator_from_left(
        self,
        piece,
        site,
        left_qns,
        *,
        source=None,
    ):
        left_qns = tuple(left_qns)
        source = source or f"{self.source_prefix}_site_operator_left"
        key = ("packed_left", str(piece), int(site), left_qns, str(source))
        cached = self._packed_site_operator_cache.get(key)
        if cached is not None:
            return cached
        local_entries, phys_qns = self.local_piece_entries(piece, site)
        op = make_abelian_packed_site_operator_from_left(
            local_entries,
            phys_qns,
            left_qns,
            source=source,
        )
        if op is not None:
            self._packed_site_operator_cache[key] = op
        return op

    def packed_site_operator_from_right(
        self,
        piece,
        site,
        right_qns,
        *,
        source=None,
    ):
        right_qns = tuple(right_qns)
        source = source or f"{self.source_prefix}_site_operator_right"
        key = ("packed_right", str(piece), int(site), right_qns, str(source))
        cached = self._packed_site_operator_cache.get(key)
        if cached is not None:
            return cached
        local_entries, phys_qns = self.local_piece_entries(piece, site)
        op = make_abelian_packed_site_operator_from_right(
            local_entries,
            phys_qns,
            right_qns,
            source=source,
        )
        if op is not None:
            self._packed_site_operator_cache[key] = op
        return op

    @property
    def stats(self):
        return {
            "site_phys_cache": int(len(self._site_phys_cache)),
            "local_piece_entries_cache": int(len(self._local_piece_entries_cache)),
            "packed_site_operator_cache": int(len(self._packed_site_operator_cache)),
            "cpp_left_site_operator_payloads": int(
                _PACKED_LOCAL_PAYLOAD_STATS.get("left_site_operator", 0)
            ),
            "cpp_right_site_operator_payloads": int(
                _PACKED_LOCAL_PAYLOAD_STATS.get("right_site_operator", 0)
            ),
            "cpp_initial_left_payloads": int(
                _PACKED_LOCAL_PAYLOAD_STATS.get("initial_left", 0)
            ),
            "cpp_initial_right_payloads": int(
                _PACKED_LOCAL_PAYLOAD_STATS.get("initial_right", 0)
            ),
            "cpp_payload_failures": int(
                _PACKED_LOCAL_PAYLOAD_STATS.get("left_site_operator_failures", 0)
                + _PACKED_LOCAL_PAYLOAD_STATS.get("right_site_operator_failures", 0)
                + _PACKED_LOCAL_PAYLOAD_STATS.get("initial_left_failures", 0)
                + _PACKED_LOCAL_PAYLOAD_STATS.get("initial_right_failures", 0)
            ),
        }


def make_abelian_packed_initial_left_environment(
    zero_qn,
    *,
    source="direct_family_initial_E",
):
    """Initial packed left boundary environment."""

    cpp_payload = _cpp_table_kernel("packed_initial_left_environment_payload")
    if cpp_payload is not None:
        try:
            keys, blocks, qns, dirs = cpp_payload(zero_qn)
            _PACKED_LOCAL_PAYLOAD_STATS["initial_left"] += 1
            return AbelianPackedBoundaryTensor(
                tuple(keys),
                tuple(blocks),
                dirs=list(dirs),
                qns=[list(axis) for axis in qns],
                source=source,
                assume_unique=True,
            )
        except Exception:
            _PACKED_LOCAL_PAYLOAD_STATS["initial_left_failures"] += 1
            pass
    return AbelianPackedBoundaryTensor(
        ((zero_qn, zero_qn, zero_qn),),
        (np.ones((1, 1, 1), dtype=complex),),
        dirs=[1, -1, 1],
        qns=[[zero_qn], [zero_qn], [zero_qn]],
        source=source,
        assume_unique=True,
    )


def make_abelian_packed_initial_right_environment(
    zero_qn,
    target_qn,
    *,
    source="direct_family_initial_F",
):
    """Initial packed right boundary environment."""

    cpp_payload = _cpp_table_kernel("packed_initial_right_environment_payload")
    if cpp_payload is not None:
        try:
            keys, blocks, qns, dirs = cpp_payload(zero_qn, target_qn)
            _PACKED_LOCAL_PAYLOAD_STATS["initial_right"] += 1
            return AbelianPackedBoundaryTensor(
                tuple(keys),
                tuple(blocks),
                dirs=list(dirs),
                qns=[list(axis) for axis in qns],
                source=source,
                assume_unique=True,
            )
        except Exception:
            _PACKED_LOCAL_PAYLOAD_STATS["initial_right_failures"] += 1
            pass
    return AbelianPackedBoundaryTensor(
        ((zero_qn, target_qn, target_qn),),
        (np.ones((1, 1, 1), dtype=complex),),
        dirs=[-1, 1, -1],
        qns=[[zero_qn], [target_qn], [target_qn]],
        source=source,
        assume_unique=True,
    )


def advance_abelian_packed_left_boundary(
    W,
    A,
    E,
    B,
    *,
    A_conj=None,
    source_prefix="direct_family_left",
):
    """Advance a packed left boundary through one MPS tensor."""

    if A_conj is None:
        A_conj = conjugate_abelian_packed_boundary_tensor(
            A,
            source=f"{source_prefix}_A_conj",
        )
    cpp_payload = _cpp_table_kernel("packed_left_boundary_advance_payload")
    if cpp_payload is not None:
        try:
            keys, blocks, qns, dirs = cpp_payload(W, A_conj, E, B)
            _PACKED_BOUNDARY_ADVANCE_PAYLOAD_STATS["left"] += 1
            return AbelianPackedBoundaryTensor(
                tuple(keys),
                tuple(blocks),
                dirs=list(dirs),
                qns=[list(axis) for axis in qns],
                source=f"{source_prefix}_environment",
                assume_unique=True,
            )
        except Exception:
            _PACKED_BOUNDARY_ADVANCE_PAYLOAD_STATS["left_failures"] += 1
    e_keys, e_blocks, e_dirs, e_qns = abelian_packed_tensor_items(E)
    cache_key = (id(A_conj), id(W), id(B))
    cached_groups = _LEFT_ADVANCE_GROUP_CACHE.get(cache_key)
    if (
        cached_groups is not None
        and cached_groups[0] is A_conj
        and cached_groups[1] is W
        and cached_groups[2] is B
    ):
        a_by_left, w_by_left_phys, b_by_left_phys = (
            cached_groups[3],
            cached_groups[4],
            cached_groups[5],
        )
        a_dirs, a_qns, w_dirs, w_qns, b_dirs, b_qns = (
            cached_groups[6],
            cached_groups[7],
            cached_groups[8],
            cached_groups[9],
            cached_groups[10],
            cached_groups[11],
        )
    else:
        a_keys, a_blocks, a_dirs, a_qns = abelian_packed_tensor_items(A_conj)
        w_keys, w_blocks, w_dirs, w_qns = abelian_packed_tensor_items(W)
        b_keys, b_blocks, b_dirs, b_qns = abelian_packed_tensor_items(B)
        a_by_left = defaultdict(list)
        for key, block in zip(a_keys, a_blocks):
            if len(key) != 3:
                raise ValueError("left advance expects rank-3 MPS blocks")
            a_by_left[key[0]].append((key, np.asarray(block)))
        w_by_left_phys = defaultdict(list)
        for key, block in zip(w_keys, w_blocks):
            if len(key) != 4:
                raise ValueError("left advance expects rank-4 local operator blocks")
            w_by_left_phys[(key[0], key[2])].append((key, np.asarray(block)))
        b_by_left_phys = defaultdict(list)
        for key, block in zip(b_keys, b_blocks):
            if len(key) != 3:
                raise ValueError("left advance expects rank-3 MPS blocks")
            b_by_left_phys[(key[0], key[2])].append((key, np.asarray(block)))
        if len(_LEFT_ADVANCE_GROUP_CACHE) > _IDENTITY_ADVANCE_GROUP_CACHE_LIMIT:
            _LEFT_ADVANCE_GROUP_CACHE.clear()
        _LEFT_ADVANCE_GROUP_CACHE[cache_key] = (
            A_conj,
            W,
            B,
            a_by_left,
            w_by_left_phys,
            b_by_left_phys,
            a_dirs,
            a_qns,
            w_dirs,
            w_qns,
            b_dirs,
            b_qns,
        )
    block_kernel = (
        None
        if _packed_cython is None
        or not getattr(_packed_cython, "CYTHON_AVAILABLE", False)
        else getattr(_packed_cython, "packed_left_boundary_block", None)
    )
    out = OrderedDict()
    for e_key, e_block in zip(e_keys, e_blocks):
        if len(e_key) != 3:
            raise ValueError("left advance expects rank-3 boundary blocks")
        e_block = np.asarray(e_block)
        for a_key, a_block in a_by_left.get(e_key[1], ()):
            for w_key, w_block in w_by_left_phys.get((e_key[0], a_key[2]), ()):
                for b_key, b_block in b_by_left_phys.get((e_key[2], w_key[3]), ()):
                    out_key = (w_key[1], a_key[1], b_key[1])
                    if block_kernel is None:
                        block = np.einsum(
                            "xij,iau,xyuv,jbv->yab",
                            e_block,
                            a_block,
                            w_block,
                            b_block,
                        )
                    else:
                        try:
                            block = block_kernel(e_block, a_block, w_block, b_block)
                        except Exception:
                            block = np.einsum(
                                "xij,iau,xyuv,jbv->yab",
                                e_block,
                                a_block,
                                w_block,
                                b_block,
                            )
                    out[out_key] = block if out_key not in out else out[out_key] + block
    qns = None
    if (
        w_qns is not None
        and a_qns is not None
        and b_qns is not None
    ):
        qns = [
            _packed_axis_qns_from_items(w_qns, 1),
            _packed_axis_qns_from_items(a_qns, 1),
            _packed_axis_qns_from_items(b_qns, 1),
        ]
    return AbelianPackedBoundaryTensor(
        tuple(out.keys()),
        tuple(out.values()),
        dirs=[w_dirs[1], a_dirs[1], b_dirs[1]],
        qns=qns,
        source=f"{source_prefix}_environment",
        assume_unique=True,
    )


def _packed_axis_qns_from_items(qns, axis):
    if qns is None or int(axis) >= len(qns):
        return None
    return list(qns[int(axis)])


def advance_abelian_packed_left_identity_boundary(
    A,
    E,
    B,
    *,
    A_conj=None,
    source_prefix="direct_family_left_identity",
):
    """Advance a packed left boundary through an identity local operator."""

    if A_conj is None:
        A_conj = conjugate_abelian_packed_boundary_tensor(
            A,
            source=f"{source_prefix}_A_conj",
        )
    cpp_payload = _cpp_table_kernel("packed_left_identity_boundary_advance_payload")
    if cpp_payload is not None:
        try:
            keys, blocks, qns, dirs = cpp_payload(A_conj, E, B)
            _PACKED_BOUNDARY_ADVANCE_PAYLOAD_STATS["left_identity"] += 1
            return AbelianPackedBoundaryTensor(
                tuple(keys),
                tuple(blocks),
                dirs=list(dirs),
                qns=[list(axis) for axis in qns],
                source=f"{source_prefix}_environment",
                assume_unique=True,
            )
        except Exception:
            _PACKED_BOUNDARY_ADVANCE_PAYLOAD_STATS["left_identity_failures"] += 1
    e_keys, e_blocks, e_dirs, e_qns = abelian_packed_tensor_items(E)
    cache_key = (id(A_conj), id(B))
    cached_groups = _LEFT_IDENTITY_ADVANCE_GROUP_CACHE.get(cache_key)
    if (
        cached_groups is not None
        and cached_groups[0] is A_conj
        and cached_groups[1] is B
    ):
        a_by_left, b_by_left_phys = cached_groups[2], cached_groups[3]
        a_dirs, a_qns, b_dirs, b_qns = (
            cached_groups[4],
            cached_groups[5],
            cached_groups[6],
            cached_groups[7],
        )
    else:
        a_keys, a_blocks, a_dirs, a_qns = abelian_packed_tensor_items(A_conj)
        b_keys, b_blocks, b_dirs, b_qns = abelian_packed_tensor_items(B)
        a_by_left = defaultdict(list)
        for key, block in zip(a_keys, a_blocks):
            if len(key) != 3:
                raise ValueError("identity left advance expects rank-3 MPS blocks")
            a_by_left[key[0]].append((key, np.asarray(block)))
        b_by_left_phys = defaultdict(list)
        for key, block in zip(b_keys, b_blocks):
            if len(key) != 3:
                raise ValueError("identity left advance expects rank-3 MPS blocks")
            b_by_left_phys[(key[0], key[2])].append((key, np.asarray(block)))
        if len(_LEFT_IDENTITY_ADVANCE_GROUP_CACHE) > _IDENTITY_ADVANCE_GROUP_CACHE_LIMIT:
            _LEFT_IDENTITY_ADVANCE_GROUP_CACHE.clear()
        _LEFT_IDENTITY_ADVANCE_GROUP_CACHE[cache_key] = (
            A_conj,
            B,
            a_by_left,
            b_by_left_phys,
            a_dirs,
            a_qns,
            b_dirs,
            b_qns,
        )
    block_kernel = (
        None
        if _packed_cython is None
        or not getattr(_packed_cython, "CYTHON_AVAILABLE", False)
        else getattr(_packed_cython, "packed_left_identity_boundary_block", None)
    )
    out = OrderedDict()
    for e_key, e_block in zip(e_keys, e_blocks):
        if len(e_key) != 3:
            raise ValueError("identity left advance expects rank-3 boundary blocks")
        e_block = np.asarray(e_block)
        for a_key, a_block in a_by_left.get(e_key[1], ()):
            phys = a_key[2]
            for b_key, b_block in b_by_left_phys.get((e_key[2], phys), ()):
                out_key = (e_key[0], a_key[1], b_key[1])
                if block_kernel is None:
                    block = np.einsum(
                        "xij,iau,jbu->xab",
                        e_block,
                        a_block,
                        b_block,
                    )
                else:
                    try:
                        block = block_kernel(a_block, e_block, b_block)
                    except Exception:
                        block = np.einsum(
                            "xij,iau,jbu->xab",
                            e_block,
                            a_block,
                            b_block,
                        )
                out[out_key] = block if out_key not in out else out[out_key] + block
    qns = None
    if e_qns is not None and a_qns is not None and b_qns is not None:
        qns = [
            _packed_axis_qns_from_items(e_qns, 0),
            _packed_axis_qns_from_items(a_qns, 1),
            _packed_axis_qns_from_items(b_qns, 1),
        ]
    return AbelianPackedBoundaryTensor(
        tuple(out.keys()),
        tuple(out.values()),
        dirs=[e_dirs[0], a_dirs[1], b_dirs[1]],
        qns=qns,
        source=f"{source_prefix}_environment",
        assume_unique=True,
    )


def advance_abelian_packed_right_boundary(
    W,
    A,
    F,
    B,
    *,
    A_conj=None,
    source_prefix="direct_family_right",
):
    """Advance a packed right boundary through one MPS tensor."""

    if A_conj is None:
        A_conj = conjugate_abelian_packed_boundary_tensor(
            A,
            source=f"{source_prefix}_A_conj",
        )
    cpp_payload = _cpp_table_kernel("packed_right_boundary_advance_payload")
    if cpp_payload is not None:
        try:
            keys, blocks, qns, dirs = cpp_payload(W, A_conj, F, B)
            _PACKED_BOUNDARY_ADVANCE_PAYLOAD_STATS["right"] += 1
            return AbelianPackedBoundaryTensor(
                tuple(keys),
                tuple(blocks),
                dirs=list(dirs),
                qns=[list(axis) for axis in qns],
                source=f"{source_prefix}_environment",
                assume_unique=True,
            )
        except Exception:
            _PACKED_BOUNDARY_ADVANCE_PAYLOAD_STATS["right_failures"] += 1
    f_keys, f_blocks, f_dirs, f_qns = abelian_packed_tensor_items(F)
    cache_key = (id(A_conj), id(W), id(B))
    cached_groups = _RIGHT_ADVANCE_GROUP_CACHE.get(cache_key)
    if (
        cached_groups is not None
        and cached_groups[0] is A_conj
        and cached_groups[1] is W
        and cached_groups[2] is B
    ):
        a_by_right, w_by_right_phys, b_by_right_phys = (
            cached_groups[3],
            cached_groups[4],
            cached_groups[5],
        )
        a_dirs, a_qns, w_dirs, w_qns, b_dirs, b_qns = (
            cached_groups[6],
            cached_groups[7],
            cached_groups[8],
            cached_groups[9],
            cached_groups[10],
            cached_groups[11],
        )
    else:
        a_keys, a_blocks, a_dirs, a_qns = abelian_packed_tensor_items(A_conj)
        w_keys, w_blocks, w_dirs, w_qns = abelian_packed_tensor_items(W)
        b_keys, b_blocks, b_dirs, b_qns = abelian_packed_tensor_items(B)
        a_by_right = defaultdict(list)
        for key, block in zip(a_keys, a_blocks):
            if len(key) != 3:
                raise ValueError("right advance expects rank-3 MPS blocks")
            a_by_right[key[1]].append((key, np.asarray(block)))
        w_by_right_phys = defaultdict(list)
        for key, block in zip(w_keys, w_blocks):
            if len(key) != 4:
                raise ValueError("right advance expects rank-4 local operator blocks")
            w_by_right_phys[(key[1], key[2])].append((key, np.asarray(block)))
        b_by_right_phys = defaultdict(list)
        for key, block in zip(b_keys, b_blocks):
            if len(key) != 3:
                raise ValueError("right advance expects rank-3 MPS blocks")
            b_by_right_phys[(key[1], key[2])].append((key, np.asarray(block)))
        if len(_RIGHT_ADVANCE_GROUP_CACHE) > _IDENTITY_ADVANCE_GROUP_CACHE_LIMIT:
            _RIGHT_ADVANCE_GROUP_CACHE.clear()
        _RIGHT_ADVANCE_GROUP_CACHE[cache_key] = (
            A_conj,
            W,
            B,
            a_by_right,
            w_by_right_phys,
            b_by_right_phys,
            a_dirs,
            a_qns,
            w_dirs,
            w_qns,
            b_dirs,
            b_qns,
        )
    block_kernel = (
        None
        if _packed_cython is None
        or not getattr(_packed_cython, "CYTHON_AVAILABLE", False)
        else getattr(_packed_cython, "packed_right_boundary_block", None)
    )
    out = OrderedDict()
    for f_key, f_block in zip(f_keys, f_blocks):
        if len(f_key) != 3:
            raise ValueError("right advance expects rank-3 boundary blocks")
        f_block = np.asarray(f_block)
        for a_key, a_block in a_by_right.get(f_key[1], ()):
            for w_key, w_block in w_by_right_phys.get((f_key[0], a_key[2]), ()):
                for b_key, b_block in b_by_right_phys.get((f_key[2], w_key[3]), ()):
                    out_key = (w_key[0], a_key[0], b_key[0])
                    if block_kernel is None:
                        block = np.einsum(
                            "aip,xij,yxpv,bjv->yab",
                            a_block,
                            f_block,
                            w_block,
                            b_block,
                        )
                    else:
                        try:
                            block = block_kernel(a_block, f_block, w_block, b_block)
                        except Exception:
                            block = np.einsum(
                                "aip,xij,yxpv,bjv->yab",
                                a_block,
                                f_block,
                                w_block,
                                b_block,
                            )
                    out[out_key] = block if out_key not in out else out[out_key] + block
    qns = None
    if (
        w_qns is not None
        and a_qns is not None
        and b_qns is not None
    ):
        qns = [
            _packed_axis_qns_from_items(w_qns, 0),
            _packed_axis_qns_from_items(a_qns, 0),
            _packed_axis_qns_from_items(b_qns, 0),
        ]
    return AbelianPackedBoundaryTensor(
        tuple(out.keys()),
        tuple(out.values()),
        dirs=[w_dirs[0], a_dirs[0], b_dirs[0]],
        qns=qns,
        source=f"{source_prefix}_environment",
        assume_unique=True,
    )


def advance_abelian_packed_right_identity_boundary(
    A,
    F,
    B,
    *,
    A_conj=None,
    source_prefix="direct_family_right_identity",
):
    """Advance a packed right boundary through an identity local operator."""

    if A_conj is None:
        A_conj = conjugate_abelian_packed_boundary_tensor(
            A,
            source=f"{source_prefix}_A_conj",
        )
    cpp_payload = _cpp_table_kernel("packed_right_identity_boundary_advance_payload")
    if cpp_payload is not None:
        try:
            keys, blocks, qns, dirs = cpp_payload(A_conj, F, B)
            _PACKED_BOUNDARY_ADVANCE_PAYLOAD_STATS["right_identity"] += 1
            return AbelianPackedBoundaryTensor(
                tuple(keys),
                tuple(blocks),
                dirs=list(dirs),
                qns=[list(axis) for axis in qns],
                source=f"{source_prefix}_environment",
                assume_unique=True,
            )
        except Exception:
            _PACKED_BOUNDARY_ADVANCE_PAYLOAD_STATS["right_identity_failures"] += 1
    f_keys, f_blocks, f_dirs, f_qns = abelian_packed_tensor_items(F)
    cache_key = (id(A_conj), id(B))
    cached_groups = _RIGHT_IDENTITY_ADVANCE_GROUP_CACHE.get(cache_key)
    if (
        cached_groups is not None
        and cached_groups[0] is A_conj
        and cached_groups[1] is B
    ):
        a_by_right, b_by_right_phys = cached_groups[2], cached_groups[3]
        a_dirs, a_qns, b_dirs, b_qns = (
            cached_groups[4],
            cached_groups[5],
            cached_groups[6],
            cached_groups[7],
        )
    else:
        a_keys, a_blocks, a_dirs, a_qns = abelian_packed_tensor_items(A_conj)
        b_keys, b_blocks, b_dirs, b_qns = abelian_packed_tensor_items(B)
        a_by_right = defaultdict(list)
        for key, block in zip(a_keys, a_blocks):
            if len(key) != 3:
                raise ValueError("identity right advance expects rank-3 MPS blocks")
            a_by_right[key[1]].append((key, np.asarray(block)))
        b_by_right_phys = defaultdict(list)
        for key, block in zip(b_keys, b_blocks):
            if len(key) != 3:
                raise ValueError("identity right advance expects rank-3 MPS blocks")
            b_by_right_phys[(key[1], key[2])].append((key, np.asarray(block)))
        if len(_RIGHT_IDENTITY_ADVANCE_GROUP_CACHE) > _IDENTITY_ADVANCE_GROUP_CACHE_LIMIT:
            _RIGHT_IDENTITY_ADVANCE_GROUP_CACHE.clear()
        _RIGHT_IDENTITY_ADVANCE_GROUP_CACHE[cache_key] = (
            A_conj,
            B,
            a_by_right,
            b_by_right_phys,
            a_dirs,
            a_qns,
            b_dirs,
            b_qns,
        )
    block_kernel = (
        None
        if _packed_cython is None
        or not getattr(_packed_cython, "CYTHON_AVAILABLE", False)
        else getattr(_packed_cython, "packed_right_identity_boundary_block", None)
    )
    out = OrderedDict()
    for f_key, f_block in zip(f_keys, f_blocks):
        if len(f_key) != 3:
            raise ValueError("identity right advance expects rank-3 boundary blocks")
        f_block = np.asarray(f_block)
        for a_key, a_block in a_by_right.get(f_key[1], ()):
            phys = a_key[2]
            for b_key, b_block in b_by_right_phys.get((f_key[2], phys), ()):
                out_key = (f_key[0], a_key[0], b_key[0])
                if block_kernel is None:
                    block = np.einsum(
                        "aip,xij,bjp->xab",
                        a_block,
                        f_block,
                        b_block,
                    )
                else:
                    try:
                        block = block_kernel(a_block, f_block, b_block)
                    except Exception:
                        block = np.einsum(
                            "aip,xij,bjp->xab",
                            a_block,
                            f_block,
                            b_block,
                        )
                out[out_key] = block if out_key not in out else out[out_key] + block
    qns = None
    if f_qns is not None and a_qns is not None and b_qns is not None:
        qns = [
            _packed_axis_qns_from_items(f_qns, 0),
            _packed_axis_qns_from_items(a_qns, 0),
            _packed_axis_qns_from_items(b_qns, 0),
        ]
    return AbelianPackedBoundaryTensor(
        tuple(out.keys()),
        tuple(out.values()),
        dirs=[f_dirs[0], a_dirs[0], b_dirs[0]],
        qns=qns,
        source=f"{source_prefix}_environment",
        assume_unique=True,
    )


def filter_abelian_packed_boundary_tensor_axis(
    tensor,
    axis,
    allowed,
    *,
    source=None,
):
    """Return a packed tensor with blocks restricted on one sector axis."""

    if not is_abelian_packed_boundary_tensor(tensor):
        raise TypeError("expected an AbelianPackedBoundaryTensor")
    axis = int(axis)
    allowed = set(allowed or ())
    if not allowed:
        return AbelianPackedBoundaryTensor(
            (),
            (),
            dirs=list(getattr(tensor, "dirs", ())),
            qns=[[] for _axis in range(len(getattr(tensor, "dirs", ())))],
            source=source or getattr(tensor, "source", "packed_boundary_tensor_filter"),
            assume_unique=True,
        )
    keys = []
    blocks = []
    qn_sets = [set() for _axis in range(len(getattr(tensor, "dirs", ())))]
    for key, block in zip(tensor.keys, tensor.blocks):
        if axis >= len(key) or key[axis] not in allowed:
            continue
        keys.append(key)
        blocks.append(block)
        for key_axis, qn in enumerate(key):
            qn_sets[key_axis].add(qn)
    return AbelianPackedBoundaryTensor(
        tuple(keys),
        tuple(blocks),
        dirs=list(getattr(tensor, "dirs", ())),
        qns=[sorted(qns) for qns in qn_sets],
        source=source or getattr(tensor, "source", "packed_boundary_tensor_filter"),
        assume_unique=True,
    )


class AbelianBoundaryTensorDataView:
    """Legacy ``.data`` view over a packed boundary tensor for validation."""

    __slots__ = ("data", "dirs", "qns", "rank", "source")

    def __init__(self, tensor):
        self.data = tensor.data
        self.dirs = list(getattr(tensor, "dirs", ()))
        qns = getattr(tensor, "qns", None)
        if qns is None:
            qn_sets = [set() for _axis in range(len(self.dirs))]
            for key in getattr(tensor, "keys", ()):
                for axis, qn in enumerate(key):
                    if axis < len(qn_sets):
                        qn_sets[axis].add(qn)
            qns = [sorted(items) for items in qn_sets]
        self.qns = qns
        self.rank = len(self.dirs)
        self.source = str(getattr(tensor, "source", "packed_boundary_tensor_data_view"))


def unpack_abelian_packed_boundary_tensor(tensor):
    if is_abelian_packed_boundary_tensor(tensor):
        return AbelianBoundaryTensorDataView(tensor)
    return tensor


def _packed_tensor_coalesce_key(tensor):
    if is_abelian_packed_boundary_tensor(tensor):
        return tensor.structural_signature()
    return ("object", id(tensor))


class AbelianPackedIdentityLocalEntry:
    """Boundary action entry with identity operators on the active two sites."""

    __slots__ = ("coeff", "E", "F", "source")

    def __init__(self, coeff, E, F, *, source="packed_identity_local"):
        self.coeff = coeff
        self.E = E
        self.F = F
        self.source = source


class AbelianPackedLocalGeneratorEntry:
    """Boundary action entry with explicit active-site local generators."""

    __slots__ = ("coeff", "E", "W_left", "W_right", "F", "source")

    def __init__(
        self,
        coeff,
        E,
        W_left,
        W_right,
        F,
        *,
        source="packed_local_generator",
    ):
        self.coeff = coeff
        self.E = E
        self.W_left = W_left
        self.W_right = W_right
        self.F = F
        self.source = source


class AbelianPackedDirectFamilyEntries:
    """Columnar packed direct-family entries for the C++ raw route builder."""

    __slots__ = (
        "identity_coeffs",
        "identity_E",
        "identity_F",
        "identity_sources",
        "local_coeffs",
        "local_E",
        "local_W_left",
        "local_W_right",
        "local_F",
        "local_sources",
    )

    _pyqed_packed_direct_family_entries = True

    def __init__(self):
        self.identity_coeffs = []
        self.identity_E = []
        self.identity_F = []
        self.identity_sources = []
        self.local_coeffs = []
        self.local_E = []
        self.local_W_left = []
        self.local_W_right = []
        self.local_F = []
        self.local_sources = []

    def append_identity(self, coeff, E, F, *, source="packed_identity_local"):
        self.identity_coeffs.append(complex(coeff))
        self.identity_E.append(E)
        self.identity_F.append(F)
        self.identity_sources.append(str(source))

    def extend_identity(
        self,
        coeffs,
        E_terms,
        F_terms,
        *,
        source="packed_identity_local",
    ):
        coeffs = tuple(coeffs or ())
        n = len(coeffs)
        if n == 0:
            return
        E_terms = tuple(E_terms or ())
        F_terms = tuple(F_terms or ())
        if len(E_terms) != n or len(F_terms) != n:
            raise ValueError("packed identity extension length mismatch")
        self.identity_coeffs.extend(complex(coeff) for coeff in coeffs)
        self.identity_E.extend(E_terms)
        self.identity_F.extend(F_terms)
        self.identity_sources.extend([str(source)] * n)

    def append_local_generator(
        self,
        coeff,
        E,
        W_left,
        W_right,
        F,
        *,
        source="packed_local_generator",
    ):
        self.local_coeffs.append(complex(coeff))
        self.local_E.append(E)
        self.local_W_left.append(W_left)
        self.local_W_right.append(W_right)
        self.local_F.append(F)
        self.local_sources.append(str(source))

    def extend_local_generators(
        self,
        coeffs,
        E_terms,
        W_left_terms,
        W_right_terms,
        F_terms,
        *,
        source="packed_local_generator",
    ):
        coeffs = tuple(coeffs or ())
        n = len(coeffs)
        if n == 0:
            return
        E_terms = tuple(E_terms or ())
        W_left_terms = tuple(W_left_terms or ())
        W_right_terms = tuple(W_right_terms or ())
        F_terms = tuple(F_terms or ())
        if (
            len(E_terms) != n
            or len(W_left_terms) != n
            or len(W_right_terms) != n
            or len(F_terms) != n
        ):
            raise ValueError("packed local-generator extension length mismatch")
        self.local_coeffs.extend(complex(coeff) for coeff in coeffs)
        self.local_E.extend(E_terms)
        self.local_W_left.extend(W_left_terms)
        self.local_W_right.extend(W_right_terms)
        self.local_F.extend(F_terms)
        self.local_sources.extend([str(source)] * n)

    def extend(self, entries):
        if isinstance(entries, AbelianPackedDirectFamilyEntries):
            self.identity_coeffs.extend(entries.identity_coeffs)
            self.identity_E.extend(entries.identity_E)
            self.identity_F.extend(entries.identity_F)
            self.identity_sources.extend(entries.identity_sources)
            self.local_coeffs.extend(entries.local_coeffs)
            self.local_E.extend(entries.local_E)
            self.local_W_left.extend(entries.local_W_left)
            self.local_W_right.extend(entries.local_W_right)
            self.local_F.extend(entries.local_F)
            self.local_sources.extend(entries.local_sources)
            return
        for entry in tuple(entries or ()):
            if isinstance(entry, AbelianPackedIdentityLocalEntry):
                self.append_identity(
                    entry.coeff,
                    entry.E,
                    entry.F,
                    source=entry.source,
                )
            elif isinstance(entry, AbelianPackedLocalGeneratorEntry):
                self.append_local_generator(
                    entry.coeff,
                    entry.E,
                    entry.W_left,
                    entry.W_right,
                    entry.F,
                    source=entry.source,
                )
            else:
                raise TypeError(
                    "packed direct-family buffers only accept packed identity "
                    "or local-generator entries"
                )

    def coalesce_in_place(self, *, tol=1.0e-14):
        """Merge entries that share the same tensor handles and source labels."""

        before = len(self)
        identity_groups = OrderedDict()
        for coeff, E, F, source in zip(
            self.identity_coeffs,
            self.identity_E,
            self.identity_F,
            self.identity_sources,
        ):
            key = (
                _packed_tensor_coalesce_key(E),
                _packed_tensor_coalesce_key(F),
                str(source),
            )
            group = identity_groups.get(key)
            if group is None:
                identity_groups[key] = [complex(coeff), E, F, str(source)]
            else:
                group[0] += complex(coeff)

        local_groups = OrderedDict()
        for coeff, E, W_left, W_right, F, source in zip(
            self.local_coeffs,
            self.local_E,
            self.local_W_left,
            self.local_W_right,
            self.local_F,
            self.local_sources,
        ):
            key = (
                _packed_tensor_coalesce_key(E),
                _packed_tensor_coalesce_key(W_left),
                _packed_tensor_coalesce_key(W_right),
                _packed_tensor_coalesce_key(F),
                str(source),
            )
            group = local_groups.get(key)
            if group is None:
                local_groups[key] = [
                    complex(coeff),
                    E,
                    W_left,
                    W_right,
                    F,
                    str(source),
                ]
            else:
                group[0] += complex(coeff)

        self.identity_coeffs = []
        self.identity_E = []
        self.identity_F = []
        self.identity_sources = []
        cancelled_identity = 0
        for coeff, E, F, source in identity_groups.values():
            if abs(coeff) <= float(tol):
                cancelled_identity += 1
                continue
            self.append_identity(coeff, E, F, source=source)

        self.local_coeffs = []
        self.local_E = []
        self.local_W_left = []
        self.local_W_right = []
        self.local_F = []
        self.local_sources = []
        cancelled_local = 0
        for coeff, E, W_left, W_right, F, source in local_groups.values():
            if abs(coeff) <= float(tol):
                cancelled_local += 1
                continue
            self.append_local_generator(
                coeff,
                E,
                W_left,
                W_right,
                F,
                source=source,
            )
        after = len(self)
        return {
            "before": int(before),
            "after": int(after),
            "reduction": int(before - after),
            "cancelled_identity": int(cancelled_identity),
            "cancelled_local": int(cancelled_local),
            "identity_groups": int(len(identity_groups)),
            "local_groups": int(len(local_groups)),
        }

    @property
    def identity_count(self):
        return int(len(self.identity_coeffs))

    @property
    def local_generator_count(self):
        return int(len(self.local_coeffs))

    @property
    def direct_component_count(self):
        return 0

    def __len__(self):
        return self.identity_count + self.local_generator_count

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        for coeff, E, F, source in zip(
            self.identity_coeffs,
            self.identity_E,
            self.identity_F,
            self.identity_sources,
        ):
            yield AbelianPackedIdentityLocalEntry(coeff, E, F, source=source)
        for coeff, E, W_left, W_right, F, source in zip(
            self.local_coeffs,
            self.local_E,
            self.local_W_left,
            self.local_W_right,
            self.local_F,
            self.local_sources,
        ):
            yield AbelianPackedLocalGeneratorEntry(
                coeff,
                E,
                W_left,
                W_right,
                F,
                source=source,
            )

    def __getitem__(self, index):
        index = int(index)
        n_identity = self.identity_count
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < n_identity:
            return AbelianPackedIdentityLocalEntry(
                self.identity_coeffs[index],
                self.identity_E[index],
                self.identity_F[index],
                source=self.identity_sources[index],
            )
        index -= n_identity
        return AbelianPackedLocalGeneratorEntry(
            self.local_coeffs[index],
            self.local_E[index],
            self.local_W_left[index],
            self.local_W_right[index],
            self.local_F[index],
            source=self.local_sources[index],
        )

    @property
    def stats(self):
        return {
            "kind": "abelian_packed_direct_family_entries",
            "identity_entries": self.identity_count,
            "local_generator_entries": self.local_generator_count,
            "direct_component_entries": 0,
            "entries": len(self),
        }


class AbelianPlannedPackedDirectFamilyEntries:
    """Packed local-generator entries backed by route ids and boundary tables."""

    __slots__ = (
        "local_coeffs",
        "local_left_ids",
        "local_right_ids",
        "left_table_ids",
        "right_table_ids",
        "left_values",
        "right_values",
        "left_table",
        "right_table",
        "local_sources",
        "_local_columns",
        "_table_backed",
        "_schedule_key",
        "source",
    )

    _pyqed_packed_direct_family_entries = True
    _pyqed_planned_direct_family_entries = True

    identity_coeffs = ()
    identity_E = ()
    identity_F = ()
    identity_sources = ()

    def __init__(
        self,
        coeffs,
        left_ids,
        right_ids,
        left_values,
        right_values,
        *,
        left_table_ids=None,
        right_table_ids=None,
        left_table=None,
        right_table=None,
        table_backed=None,
        schedule_key=None,
        source="planned_packed_local_generator",
    ):
        coeffs = np.asarray(coeffs if coeffs is not None else (), dtype=np.complex128)
        left_ids = np.asarray(left_ids if left_ids is not None else (), dtype=np.int64)
        right_ids = np.asarray(right_ids if right_ids is not None else (), dtype=np.int64)
        left_table_ids = np.asarray(
            left_table_ids if left_table_ids is not None else (),
            dtype=np.int64,
        )
        right_table_ids = np.asarray(
            right_table_ids if right_table_ids is not None else (),
            dtype=np.int64,
        )
        if len(left_ids) != len(coeffs) or len(right_ids) != len(coeffs):
            raise ValueError("planned packed direct route length mismatch")
        if left_table_ids.size and left_table_ids.size <= int(left_ids.max(initial=-1)):
            raise ValueError("planned packed direct left table ids are too short")
        if right_table_ids.size and right_table_ids.size <= int(right_ids.max(initial=-1)):
            raise ValueError("planned packed direct right table ids are too short")
        self.local_coeffs = coeffs
        self.local_left_ids = left_ids
        self.local_right_ids = right_ids
        self.left_table_ids = left_table_ids
        self.right_table_ids = right_table_ids
        self.left_values = tuple(left_values or ())
        self.right_values = tuple(right_values or ())
        self.left_table = left_table
        self.right_table = right_table
        self.local_sources = ()
        self._local_columns = None
        if table_backed is None:
            table_backed = bool(
                left_table is not None
                and right_table is not None
                and left_table_ids.size
                and right_table_ids.size
                and np.all(left_table_ids >= 0)
                and np.all(right_table_ids >= 0)
            )
        self._table_backed = bool(table_backed)
        self._schedule_key = schedule_key
        self.source = str(source)

    @property
    def _pyqed_planned_direct_family_table_ids(self):
        return bool(self._table_backed)

    @property
    def _pyqed_planned_direct_family_schedule_key(self):
        return self._schedule_key

    @property
    def left_table_payloads(self):
        return () if self.left_table is None else self.left_table.payloads

    @property
    def right_table_payloads(self):
        return () if self.right_table is None else self.right_table.payloads

    @classmethod
    def from_route_plan(
        cls,
        route_plan,
        boundary_batch,
        *,
        left_table=None,
        right_table=None,
        schedule_key=None,
        source="planned_packed_local_generator",
    ):
        raw_left_table_ids = getattr(boundary_batch, "left_table_ids", ())
        raw_right_table_ids = getattr(boundary_batch, "right_table_ids", ())
        if raw_left_table_ids is None:
            raw_left_table_ids = ()
        if raw_right_table_ids is None:
            raw_right_table_ids = ()
        left_table_ids = np.asarray(raw_left_table_ids, dtype=np.int64)
        right_table_ids = np.asarray(raw_right_table_ids, dtype=np.int64)
        table_backed = bool(
            left_table is not None
            and right_table is not None
            and left_table_ids.size == len(route_plan.left_keys)
            and right_table_ids.size == len(route_plan.right_keys)
            and bool(np.all(left_table_ids >= 0))
            and bool(np.all(right_table_ids >= 0))
        )
        return cls(
            route_plan.pair_coeffs,
            route_plan.pair_left_ids,
            route_plan.pair_right_ids,
            ()
            if table_backed
            else tuple(getattr(boundary_batch, "left_values", ()) or ()),
            ()
            if table_backed
            else tuple(getattr(boundary_batch, "right_values", ()) or ()),
            left_table_ids=left_table_ids,
            right_table_ids=right_table_ids,
            left_table=left_table if table_backed else None,
            right_table=right_table if table_backed else None,
            table_backed=table_backed,
            schedule_key=schedule_key,
            source=source,
        )

    def _materialize_local_columns(self):
        cached = self._local_columns
        if cached is not None:
            return cached
        E_terms = []
        W_left_terms = []
        W_right_terms = []
        F_terms = []
        for left_id, right_id in zip(self.local_left_ids, self.local_right_ids):
            try:
                if self._pyqed_planned_direct_family_table_ids:
                    left_result = self.left_table.payloads[
                        int(self.left_table_ids[int(left_id)])
                    ]
                    right_result = self.right_table.payloads[
                        int(self.right_table_ids[int(right_id)])
                    ]
                else:
                    left_result = self.left_values[int(left_id)]
                    right_result = self.right_values[int(right_id)]
                E_term, W_left = left_result
                W_right, F_term = right_result
            except Exception as exc:
                raise ValueError("planned packed direct boundary payload is missing") from exc
            E_terms.append(E_term)
            W_left_terms.append(W_left)
            W_right_terms.append(W_right)
            F_terms.append(F_term)
        cached = (
            tuple(E_terms),
            tuple(W_left_terms),
            tuple(W_right_terms),
            tuple(F_terms),
        )
        self._local_columns = cached
        return cached

    def snapshot_table_payloads(self):
        """Detach planned entries from mutable boundary-table payload slots."""

        if not self._pyqed_planned_direct_family_table_ids:
            return self
        left_payloads = self.left_table_payloads
        right_payloads = self.right_table_payloads
        left_values = []
        for table_id in self.left_table_ids:
            table_id = int(table_id)
            if table_id < 0 or table_id >= len(left_payloads):
                return self
            payload = left_payloads[table_id]
            if payload is None:
                return self
            left_values.append(payload)
        right_values = []
        for table_id in self.right_table_ids:
            table_id = int(table_id)
            if table_id < 0 or table_id >= len(right_payloads):
                return self
            payload = right_payloads[table_id]
            if payload is None:
                return self
            right_values.append(payload)
        self.left_values = tuple(left_values)
        self.right_values = tuple(right_values)
        self.left_table = None
        self.right_table = None
        self._table_backed = False
        self._local_columns = None
        return self

    @property
    def local_E(self):
        return self._materialize_local_columns()[0]

    @property
    def local_W_left(self):
        return self._materialize_local_columns()[1]

    @property
    def local_W_right(self):
        return self._materialize_local_columns()[2]

    @property
    def local_F(self):
        return self._materialize_local_columns()[3]

    def coalesce_in_place(self, *, tol=1.0e-14):
        return {
            "before": int(len(self)),
            "after": int(len(self)),
            "reduction": 0,
            "cancelled_identity": 0,
            "cancelled_local": 0,
            "identity_groups": 0,
            "local_groups": int(len(self)),
        }

    def extend(self, entries):
        materialized = AbelianPackedDirectFamilyEntries()
        materialized.extend(self)
        materialized.extend(entries)
        self.local_coeffs = np.asarray(materialized.local_coeffs, dtype=np.complex128)
        self.local_left_ids = np.arange(len(materialized.local_coeffs), dtype=np.int64)
        self.local_right_ids = np.arange(len(materialized.local_coeffs), dtype=np.int64)
        self.left_table_ids = np.zeros(0, dtype=np.int64)
        self.right_table_ids = np.zeros(0, dtype=np.int64)
        self.left_values = tuple(
            zip(materialized.local_E, materialized.local_W_left)
        )
        self.right_values = tuple(
            zip(materialized.local_W_right, materialized.local_F)
        )
        self.left_table = None
        self.right_table = None
        self.local_sources = ()
        self._table_backed = False
        self._schedule_key = None
        self._local_columns = (
            tuple(materialized.local_E),
            tuple(materialized.local_W_left),
            tuple(materialized.local_W_right),
            tuple(materialized.local_F),
        )

    @property
    def identity_count(self):
        return 0

    @property
    def local_generator_count(self):
        return int(len(self.local_coeffs))

    @property
    def direct_component_count(self):
        return 0

    def __len__(self):
        return self.local_generator_count

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        E_terms, W_left_terms, W_right_terms, F_terms = self._materialize_local_columns()
        for coeff, E, W_left, W_right, F in zip(
            self.local_coeffs,
            E_terms,
            W_left_terms,
            W_right_terms,
            F_terms,
        ):
            yield AbelianPackedLocalGeneratorEntry(
                coeff,
                E,
                W_left,
                W_right,
                F,
                source=self.source,
            )

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        E_terms, W_left_terms, W_right_terms, F_terms = self._materialize_local_columns()
        return AbelianPackedLocalGeneratorEntry(
            self.local_coeffs[index],
            E_terms[index],
            W_left_terms[index],
            W_right_terms[index],
            F_terms[index],
            source=self.source,
        )

    @property
    def stats(self):
        return {
            "kind": "abelian_planned_packed_direct_family_entries",
            "identity_entries": 0,
            "local_generator_entries": self.local_generator_count,
            "direct_component_entries": 0,
            "entries": len(self),
            "left_values": int(len(self.left_values)),
            "right_values": int(len(self.right_values)),
            "table_backed": bool(self._pyqed_planned_direct_family_table_ids),
            "left_table_ids": int(
                sum(1 for value in self.left_table_ids if int(value) >= 0)
            ),
            "right_table_ids": int(
                sum(1 for value in self.right_table_ids if int(value) >= 0)
            ),
        }


class AbelianSameSidePRouteIdentityEntries:
    """Compact same-side P route rows consumed as identity-local entries."""

    __slots__ = (
        "side",
        "row_ids",
        "row_coeffs",
        "offsets",
        "boundary_ids",
        "factors",
        "boundary_table_ids",
        "boundary_values",
        "boundary_table",
        "identity_tensor",
        "source",
        "_entry_count",
    )

    _pyqed_packed_direct_family_entries = True
    _pyqed_same_side_route_identity_entries = True

    def __init__(
        self,
        *,
        side,
        row_ids,
        row_coeffs,
        route_plan,
        boundary_table_ids=None,
        boundary_values=(),
        boundary_table=None,
        identity_tensor=None,
        source="same_side_p_route_identity",
    ):
        side = str(side)
        if side not in {"left", "right"}:
            raise ValueError("same-side P route side must be 'left' or 'right'")
        row_ids = np.asarray(row_ids if row_ids is not None else (), dtype=np.int64)
        row_coeffs = np.asarray(
            row_coeffs if row_coeffs is not None else (),
            dtype=np.complex128,
        )
        offsets = np.asarray(route_plan.offsets, dtype=np.int64)
        boundary_ids = np.asarray(route_plan.boundary_ids, dtype=np.int64)
        factors = np.asarray(route_plan.factors, dtype=np.complex128)
        boundary_table_ids = np.asarray(
            boundary_table_ids if boundary_table_ids is not None else (),
            dtype=np.int64,
        )
        if row_ids.ndim != 1 or row_coeffs.ndim != 1:
            raise ValueError("same-side P route rows must be rank-1 arrays")
        if row_ids.shape[0] != row_coeffs.shape[0]:
            raise ValueError("same-side P route row ids and coeffs differ")
        if offsets.ndim != 1 or boundary_ids.ndim != 1 or factors.ndim != 1:
            raise ValueError("same-side P route columns must be rank-1 arrays")
        if boundary_ids.shape[0] != factors.shape[0]:
            raise ValueError("same-side P route boundary columns differ")
        if offsets.shape[0] <= 0:
            raise ValueError("same-side P route offsets are empty")
        entry_count = 0
        for row in row_ids:
            row = int(row)
            if row < 0 or row + 1 >= int(offsets.shape[0]):
                raise ValueError("same-side P route row is out of range")
            entry_count += max(0, int(offsets[row + 1]) - int(offsets[row]))
        self.side = side
        self.row_ids = row_ids
        self.row_coeffs = row_coeffs
        self.offsets = offsets
        self.boundary_ids = boundary_ids
        self.factors = factors
        self.boundary_table_ids = boundary_table_ids
        self.boundary_values = tuple(boundary_values or ())
        self.boundary_table = boundary_table
        self.identity_tensor = identity_tensor
        self.source = str(source)
        self._entry_count = int(entry_count)

    @property
    def boundary_payloads(self):
        if self.boundary_table is not None:
            return self.boundary_table.payloads
        return self.boundary_values

    def _boundary_tensor(self, boundary_id):
        boundary_id = int(boundary_id)
        if self.boundary_table_ids.size:
            if boundary_id < 0 or boundary_id >= int(self.boundary_table_ids.shape[0]):
                return None
            table_id = int(self.boundary_table_ids[boundary_id])
            payloads = self.boundary_payloads
            if table_id < 0 or table_id >= len(payloads):
                return None
            return payloads[table_id]
        if boundary_id < 0 or boundary_id >= len(self.boundary_values):
            return None
        return self.boundary_values[boundary_id]

    @property
    def identity_count(self):
        return int(self._entry_count)

    @property
    def local_generator_count(self):
        return 0

    @property
    def direct_component_count(self):
        return 0

    def __len__(self):
        return int(self._entry_count)

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        identity = self.identity_tensor
        for row, row_coeff in zip(self.row_ids, self.row_coeffs):
            start = int(self.offsets[int(row)])
            stop = int(self.offsets[int(row) + 1])
            for item in range(start, stop):
                boundary = self._boundary_tensor(int(self.boundary_ids[item]))
                if boundary is None:
                    continue
                coeff = complex(row_coeff) * complex(self.factors[item])
                if self.side == "left":
                    yield AbelianPackedIdentityLocalEntry(
                        coeff,
                        boundary,
                        identity,
                        source=self.source,
                    )
                else:
                    yield AbelianPackedIdentityLocalEntry(
                        coeff,
                        identity,
                        boundary,
                        source=self.source,
                    )

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError(index)
        for item_index, entry in enumerate(self):
            if item_index == index:
                return entry
        raise IndexError(index)

    @property
    def stats(self):
        return {
            "kind": "abelian_same_side_p_route_identity_entries",
            "side": str(self.side),
            "rows": int(self.row_ids.shape[0]),
            "entries": len(self),
            "boundary_values": int(len(self.boundary_values)),
            "boundary_table_ids": int(
                sum(1 for value in self.boundary_table_ids if int(value) >= 0)
            ),
            "table_backed": bool(self.boundary_table is not None),
        }


class AbelianCompositePackedDirectFamilyEntries:
    """Ordered direct-family chunks consumed without materializing planned parts."""

    __slots__ = ("parts",)

    _pyqed_packed_direct_family_entries = True
    _pyqed_composite_direct_family_entries = True

    def __init__(self, parts=()):
        self.parts = []
        self.extend(parts)

    @staticmethod
    def _as_part(entries):
        if entries is None:
            return None
        if isinstance(entries, AbelianCompositePackedDirectFamilyEntries):
            return entries
        if bool(getattr(entries, "_pyqed_packed_direct_family_entries", False)):
            return entries if len(entries) else None
        packed = AbelianPackedDirectFamilyEntries()
        packed.extend(entries)
        return packed if len(packed) else None

    def append(self, entries):
        part = self._as_part(entries)
        if part is None:
            return
        if isinstance(part, AbelianCompositePackedDirectFamilyEntries):
            self.parts.extend(part.parts)
        else:
            self.parts.append(part)

    def extend(self, entries):
        if isinstance(entries, AbelianCompositePackedDirectFamilyEntries):
            self.parts.extend(entries.parts)
            return
        if bool(getattr(entries, "_pyqed_packed_direct_family_entries", False)):
            self.append(entries)
            return
        try:
            seq = tuple(entries or ())
        except TypeError:
            seq = ()
        if seq and all(
            isinstance(item, AbelianCompositePackedDirectFamilyEntries)
            or bool(getattr(item, "_pyqed_packed_direct_family_entries", False))
            for item in seq
        ):
            for item in seq:
                self.append(item)
            return
        self.append(entries)

    def coalesce_in_place(self, *, tol=1.0e-14):
        return {
            "before": int(len(self)),
            "after": int(len(self)),
            "reduction": 0,
            "cancelled_identity": 0,
            "cancelled_local": 0,
            "identity_groups": int(self.identity_count),
            "local_groups": int(self.local_generator_count),
        }

    @property
    def identity_count(self):
        return int(sum(int(getattr(part, "identity_count", 0)) for part in self.parts))

    @property
    def local_generator_count(self):
        return int(
            sum(int(getattr(part, "local_generator_count", 0)) for part in self.parts)
        )

    @property
    def direct_component_count(self):
        return int(
            sum(int(getattr(part, "direct_component_count", 0)) for part in self.parts)
        )

    def __len__(self):
        return int(sum(len(part) for part in self.parts))

    def __bool__(self):
        return bool(len(self))

    def __iter__(self):
        for part in self.parts:
            yield from part

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError(index)
        offset = 0
        for part in self.parts:
            stop = offset + len(part)
            if index < stop:
                return part[index - offset]
            offset = stop
        raise IndexError(index)

    @property
    def stats(self):
        return {
            "kind": "abelian_composite_packed_direct_family_entries",
            "identity_entries": self.identity_count,
            "local_generator_entries": self.local_generator_count,
            "direct_component_entries": self.direct_component_count,
            "entries": len(self),
            "parts": int(len(self.parts)),
        }


def _apply_abelian_packed_identity_local_action(entry, basis):
    E_keys, E_blocks, _E_dirs, _E_qns = abelian_packed_tensor_items(entry.E)
    F_keys, F_blocks, _F_dirs, _F_qns = abelian_packed_tensor_items(entry.F)
    A_keys, A_blocks, A_dirs, A_qns = abelian_packed_tensor_items(basis)
    e_by_ket_left = defaultdict(list)
    for key, block in zip(E_keys, E_blocks):
        if len(key) != 3:
            raise ValueError("identity-local E tensor must have rank 3")
        e_by_ket_left[key[2]].append((key, np.asarray(block)))
    f_by_mpo_ket_right = defaultdict(list)
    for key, block in zip(F_keys, F_blocks):
        if len(key) != 3:
            raise ValueError("identity-local F tensor must have rank 3")
        f_by_mpo_ket_right[(key[0], key[2])].append((key, np.asarray(block)))

    out = OrderedDict()
    coeff = complex(entry.coeff)
    for a_key, a_block in zip(A_keys, A_blocks):
        if len(a_key) != 4:
            raise ValueError("identity-local basis tensor must have rank 4")
        left_in, right_in, p1, p2 = a_key
        for e_key, e_block in e_by_ket_left.get(left_in, ()):
            for f_key, f_block in f_by_mpo_ket_right.get((e_key[0], right_in), ()):
                out_key = (e_key[1], f_key[1], p1, p2)
                contrib = coeff * np.einsum(
                    "aij,jkxy,alk->ilxy",
                    e_block,
                    np.asarray(a_block),
                    f_block,
                    optimize="greedy",
                )
                out[out_key] = contrib if out_key not in out else out[out_key] + contrib
    if not out:
        return None
    return AbelianPackedBoundaryTensor(
        tuple(out.keys()),
        tuple(out.values()),
        dirs=A_dirs,
        qns=A_qns,
        source="packed_identity_local_action",
    )


def _apply_abelian_packed_generator_local_action(
    E,
    W_left,
    W_right,
    F,
    basis,
    *,
    coeff=1.0,
    source="packed_local_generator_action",
):
    R = tensordot_abelian_packed_boundary_tensors(
        E,
        basis,
        axes=([2], [0]),
        source=f"{source}_EA",
    )
    T2 = tensordot_abelian_packed_boundary_tensors(
        R,
        W_left,
        axes=([0, 3], [0, 3]),
        source=f"{source}_EAW_left",
    )
    T3 = tensordot_abelian_packed_boundary_tensors(
        T2,
        W_right,
        axes=([3, 2], [0, 3]),
        source=f"{source}_EAW_left_W_right",
    )
    T4 = tensordot_abelian_packed_boundary_tensors(
        T3,
        F,
        axes=([3, 1], [0, 2]),
        source=f"{source}_EAWF",
    )
    out = transpose_abelian_packed_boundary_tensor(
        T4,
        (0, 3, 1, 2),
        source=source,
    )
    if complex(coeff) != 1.0:
        out = scale_abelian_boundary_tensor(
            out,
            coeff,
            source=f"{source}_scaled",
        )
    return out


def apply_abelian_packed_local_action_entry(entry, basis):
    """Apply one identity/local-generator direct-family entry to a packed basis."""

    if isinstance(entry, AbelianPackedIdentityLocalEntry):
        return _apply_abelian_packed_identity_local_action(entry, basis)
    if isinstance(entry, AbelianPackedLocalGeneratorEntry):
        return _apply_abelian_packed_generator_local_action(
            entry.E,
            entry.W_left,
            entry.W_right,
            entry.F,
            basis,
            coeff=entry.coeff,
            source="packed_local_generator_action",
        )
    try:
        E, W_pair, F = entry
    except Exception as exc:
        raise TypeError("unsupported packed local action entry") from exc
    if len(W_pair) != 2:
        raise TypeError("local action tuple entry must provide two local operators")
    return _apply_abelian_packed_generator_local_action(
        E,
        W_pair[0],
        W_pair[1],
        F,
        basis,
        source="packed_tuple_local_generator_action",
    )


def apply_abelian_packed_local_action_entries(entries, basis):
    """Apply and sum packed identity/local-generator entries."""

    total = None
    for entry in tuple(entries or ()):
        contribution = apply_abelian_packed_local_action_entry(entry, basis)
        if contribution is None:
            continue
        total = (
            contribution
            if total is None
            else add_abelian_packed_boundary_tensors(
                total,
                contribution,
                source="packed_local_action_sum",
            )
        )
    return total


def compare_abelian_packed_boundary_tensors(left, right):
    """Return ``(same_layout, abs_norm, ref_norm)`` for packed/data-view tensors."""

    if left is None or right is None:
        return left is right, float("inf"), 0.0
    left_keys, left_blocks, _left_dirs, _left_qns = abelian_packed_tensor_items(left)
    right_keys, right_blocks, _right_dirs, _right_qns = abelian_packed_tensor_items(right)
    left_map = {tuple(key): np.asarray(block) for key, block in zip(left_keys, left_blocks)}
    right_map = {
        tuple(key): np.asarray(block) for key, block in zip(right_keys, right_blocks)
    }
    if set(left_map) != set(right_map):
        return False, float("inf"), 0.0
    diff_sq = 0.0
    ref_sq = 0.0
    for key in left_map:
        lhs = left_map[key]
        rhs = right_map[key]
        if tuple(lhs.shape) != tuple(rhs.shape):
            return False, float("inf"), 0.0
        delta = lhs - rhs
        diff_sq += float(np.vdot(delta.reshape(-1), delta.reshape(-1)).real)
        ref_sq += float(np.vdot(rhs.reshape(-1), rhs.reshape(-1)).real)
    return True, float(max(diff_sq, 0.0) ** 0.5), float(max(ref_sq, 0.0) ** 0.5)


def _apply_abelian_packed_local_action_entries_checked(
    entries,
    basis,
    *,
    phase,
    on_error=None,
):
    try:
        return apply_abelian_packed_local_action_entries(entries, basis)
    except Exception as exc:
        if callable(on_error):
            on_error(str(phase), exc)
        return None


def abelian_packed_local_action_apply_clean(
    proto,
    entries,
    *,
    max_iter=4,
    on_error=None,
):
    """Return whether packed entries keep the local-state block layout closed."""

    if proto is None or not entries:
        return None
    layout_map = {key: shape for key, shape in proto.layout()}
    for _iter in range(int(max_iter)):
        changed = False
        layout = tuple((key, layout_map[key]) for key in sorted(layout_map))
        for key, shape in layout:
            basis = proto.basis(
                key,
                shape,
                source="packed_local_action_apply_clean_basis",
            )
            native = _apply_abelian_packed_local_action_entries_checked(
                entries,
                basis,
                phase="apply_clean",
                on_error=on_error,
            )
            if native is None:
                return False
            for out_key, out_shape in AbelianPackedLocalStateProto(native).layout():
                old_shape = layout_map.get(out_key)
                if old_shape is None:
                    layout_map[out_key] = out_shape
                    changed = True
                elif old_shape != out_shape:
                    return False
        if not changed:
            return True
    return False


def abelian_packed_local_action_probe_reference(
    proto,
    candidate_entries,
    reference_entries,
    *,
    max_vectors=4,
    on_error=None,
):
    """Probe two packed local actions on basis vectors and compare outputs."""

    if proto is None or not candidate_entries or not reference_entries:
        return None
    layout = tuple(proto.layout())
    if not layout:
        return False
    checked = 0
    for key, shape in layout:
        n = int(np.prod(shape, dtype=int))
        for offset in range(n):
            basis = proto.basis(
                key,
                shape,
                offset=offset,
                source="packed_local_action_probe_basis",
            )
            native = _apply_abelian_packed_local_action_entries_checked(
                candidate_entries,
                basis,
                phase="probe_native",
                on_error=on_error,
            )
            reference = _apply_abelian_packed_local_action_entries_checked(
                reference_entries,
                basis,
                phase="probe_reference",
                on_error=on_error,
            )
            if native is None or reference is None:
                return False
            same_layout, diff, ref_norm = compare_abelian_packed_boundary_tensors(
                native,
                reference,
            )
            if not same_layout:
                return False
            if diff > 1.0e-10 * max(float(ref_norm), 1.0e-30) + 1.0e-12:
                return False
            checked += 1
            if checked >= int(max_vectors):
                return True
    return checked > 0


def abelian_packed_local_action_matches_reference(
    proto,
    candidate_entries,
    reference_entries,
    *,
    max_iter=4,
    on_error=None,
):
    """Return whether two packed local actions agree over the discovered layout."""

    if proto is None or not candidate_entries or not reference_entries:
        return None
    layout_map = {key: shape for key, shape in proto.layout()}
    for _iter in range(int(max_iter)):
        changed = False
        layout = tuple((key, layout_map[key]) for key in sorted(layout_map))
        for key, shape in layout:
            basis = proto.basis(
                key,
                shape,
                source="packed_local_action_match_basis",
            )
            native = _apply_abelian_packed_local_action_entries_checked(
                candidate_entries,
                basis,
                phase="match_native",
                on_error=on_error,
            )
            reference = _apply_abelian_packed_local_action_entries_checked(
                reference_entries,
                basis,
                phase="match_reference",
                on_error=on_error,
            )
            if native is None or reference is None:
                return False
            same_layout, diff, ref_norm = compare_abelian_packed_boundary_tensors(
                native,
                reference,
            )
            if not same_layout:
                return False
            if diff > 1.0e-10 * max(float(ref_norm), 1.0e-30) + 1.0e-12:
                return False
            for out_key, out_shape in AbelianPackedLocalStateProto(reference).layout():
                old_shape = layout_map.get(out_key)
                if old_shape is None:
                    layout_map[out_key] = out_shape
                    changed = True
                elif old_shape != out_shape:
                    return False
        if not changed:
            return True
    return False


def abelian_typed_direct_entry_buckets(entries):
    """Split direct-family entries by the packed component shape."""

    if isinstance(entries, AbelianPackedDirectFamilyEntries):
        return (
            range(entries.identity_count),
            range(entries.local_generator_count),
            (),
        )
    identities = []
    local_generators = []
    direct_components = []
    for entry in tuple(entries or ()):
        has_e = hasattr(entry, "E")
        has_f = hasattr(entry, "F")
        has_w_left = hasattr(entry, "W_left")
        if has_e and has_f and not has_w_left:
            identities.append(entry)
        elif has_e and has_f and has_w_left:
            local_generators.append(entry)
        else:
            direct_components.append(entry)
    return tuple(identities), tuple(local_generators), tuple(direct_components)


@dataclass(frozen=True)
class AbelianContextualFamilyBuildOptions:
    """Options controlling contextual direct-family entry construction."""

    precompute_boundaries: bool | str = False
    precompute_min_records: int = 2048
    pack_entries: bool = True
    packed_buffer: bool = True
    planned_without_precompute: bool = True
    planned_without_precompute_batch: bool = True
    planned_without_precompute_table_lookup: bool = True
    planned_without_precompute_table_ids_only: bool = True
    snapshot_table_backed_planned_entries: bool = True

    @classmethod
    def from_matvec_options(cls, options):
        options = options or {}
        explicit_precompute = (
            "generator_table_precompute_contextual_boundaries" in options
        )
        precompute = options.get(
            "generator_table_precompute_contextual_boundaries",
            False,
        )
        route_backend = str(
            options.get("generator_table_packed_route_table", "")
        ).strip().lower()
        packed_route_enabled = route_backend not in {
            "",
            "off",
            "false",
            "0",
            "none",
            "python",
            "reference",
        }
        packed_boundary_enabled = bool(
            options.get("generator_table_packed_boundary_tensors", False)
        )
        long_term_packed_path = bool(packed_route_enabled or packed_boundary_enabled)
        if long_term_packed_path and not explicit_precompute:
            precompute = True
        if isinstance(precompute, str):
            precompute = precompute.strip().lower().replace("-", "_")
            if precompute not in {"auto", "true", "false", "1", "0", "yes", "no"}:
                precompute = bool(precompute)
        explicit_precompute_min_records = (
            "generator_table_precompute_contextual_boundaries_min_records" in options
        )
        precompute_min_records = int(
            options.get(
                "generator_table_precompute_contextual_boundaries_min_records",
                0
                if long_term_packed_path and not explicit_precompute_min_records
                else 2048,
            )
            or 0
        )
        return cls(
            precompute_boundaries=precompute,
            precompute_min_records=precompute_min_records,
            pack_entries=bool(
                options.get("generator_table_pack_contextual_direct_entries", True)
            ),
            packed_buffer=bool(
                options.get("generator_table_packed_direct_family_entries", False)
            ),
            planned_without_precompute=bool(
                options.get(
                    "generator_table_planned_contextual_without_precompute",
                    False,
                )
            ),
            planned_without_precompute_batch=bool(
                options.get(
                    "generator_table_planned_contextual_without_precompute_batch",
                    True,
                )
            ),
            planned_without_precompute_table_lookup=bool(
                options.get(
                    "generator_table_planned_contextual_without_precompute_table_lookup",
                    True,
                )
            ),
            planned_without_precompute_table_ids_only=bool(
                options.get(
                    "generator_table_planned_contextual_without_precompute_table_ids_only",
                    True,
                )
            ),
            snapshot_table_backed_planned_entries=bool(
                options.get(
                    "generator_table_snapshot_table_backed_planned_entries",
                    True,
                )
            ),
        )

    def should_precompute(self, records):
        policy = self.precompute_boundaries
        if isinstance(policy, str):
            normalized = policy.strip().lower().replace("-", "_")
            if normalized == "auto":
                count = (
                    records.record_count
                    if isinstance(records, AbelianDirectRoutePlan)
                    else len(records or ())
                )
                return int(count) >= int(self.precompute_min_records)
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        return bool(policy)

    def with_precompute(self, enabled):
        return AbelianContextualFamilyBuildOptions(
            precompute_boundaries=bool(enabled),
            precompute_min_records=self.precompute_min_records,
            pack_entries=self.pack_entries,
            packed_buffer=self.packed_buffer,
            planned_without_precompute=self.planned_without_precompute,
            planned_without_precompute_batch=self.planned_without_precompute_batch,
            planned_without_precompute_table_lookup=(
                self.planned_without_precompute_table_lookup
            ),
            planned_without_precompute_table_ids_only=(
                self.planned_without_precompute_table_ids_only
            ),
            snapshot_table_backed_planned_entries=(
                self.snapshot_table_backed_planned_entries
            ),
        )


@dataclass(frozen=True)
class AbelianContextualBoundaryBatch:
    """Precomputed left/right contextual boundary lookups for one family."""

    left: dict
    right: dict
    left_values: tuple = ()
    right_values: tuple = ()
    left_table_ids: tuple = ()
    right_table_ids: tuple = ()
    left_payload_counts: dict = field(default_factory=dict)
    right_payload_counts: dict = field(default_factory=dict)

    @property
    def left_packed_count(self):
        return int(self.left_payload_counts.get("packed", 0))

    @property
    def right_packed_count(self):
        return int(self.right_payload_counts.get("packed", 0))

    @property
    def packed_boundary_pairs(self):
        total_left = sum(int(value) for value in self.left_payload_counts.values())
        total_right = sum(int(value) for value in self.right_payload_counts.values())
        return bool(
            total_left
            and total_right
            and self.left_packed_count == total_left
            and self.right_packed_count == total_right
        )


@dataclass
class AbelianPackedContextualBoundaryTable:
    """First-class table for packed contextual boundary payloads."""

    side: str
    bond: int = -1
    revision: int = -1
    entries: OrderedDict = field(default_factory=OrderedDict)
    ids: dict = field(default_factory=dict)
    payloads: list = field(default_factory=list)
    family_counts: dict = field(default_factory=dict)
    blocks: int = 0
    hits: int = 0
    misses: int = 0
    puts: int = 0
    batch_resolves: int = 0
    batch_stores: int = 0
    cpp_resolves: int = 0
    cpp_stores: int = 0
    last_batch_size: int = 0
    last_batch_hits: int = 0
    last_batch_misses: int = 0
    resets: int = 0
    evictions: int = 0
    source: str = "abelian_packed_contextual_boundary_table"

    @staticmethod
    def normalize_key(key):
        key = tuple(key)
        if len(key) == 3:
            return (str(key[0]), tuple(key[1]), str(key[2]))
        return (tuple(key[0]), str(key[1]))

    @staticmethod
    def is_normalized_key(key):
        return (
            isinstance(key, tuple)
            and (
                (
                    len(key) == 2
                    and isinstance(key[0], tuple)
                    and isinstance(key[1], str)
                )
                or (
                    len(key) == 3
                    and isinstance(key[0], str)
                    and isinstance(key[1], tuple)
                    and isinstance(key[2], str)
                )
            )
        )

    @staticmethod
    def payload_block_count(value):
        total = 0
        for item in tuple(value or ()):
            total += int(len(getattr(item, "blocks", ()) or ()))
        return int(total)

    def get(self, key):
        key = self.normalize_key(key)
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def resolve_many(self, keys, *, normalized=False, return_ids=False):
        keys = tuple(keys or ())
        if not bool(normalized):
            keys = tuple(self.normalize_key(key) for key in keys)
        kernel = _cpp_table_kernel("dict_resolve_many")
        if kernel is not None:
            try:
                (
                    values,
                    table_ids,
                    missing,
                    missing_positions,
                    hits,
                    misses,
                ) = kernel(self.entries, self.ids, keys)
                hits = int(hits)
                misses = int(misses)
                self.batch_resolves += 1
                self.hits += hits
                self.misses += misses
                self.last_batch_size = int(len(keys))
                self.last_batch_hits = hits
                self.last_batch_misses = misses
                self.cpp_resolves += 1
                if bool(return_ids):
                    return values, table_ids, missing, missing_positions, hits, misses
                return values, missing, missing_positions, hits, misses
            except Exception:
                pass
        values = [None] * len(keys)
        table_ids = [-1] * len(keys)
        missing = []
        missing_positions = []
        hits = 0
        misses = 0
        for idx, key in enumerate(keys):
            value = self.entries.get(key)
            if value is None:
                missing.append(key)
                missing_positions.append(idx)
                misses += 1
            else:
                values[idx] = value
                table_id = self.ids.get(key)
                if table_id is not None:
                    table_ids[idx] = int(table_id)
                hits += 1
        self.batch_resolves += 1
        self.hits += int(hits)
        self.misses += int(misses)
        self.last_batch_size = int(len(keys))
        self.last_batch_hits = int(hits)
        self.last_batch_misses = int(misses)
        if bool(return_ids):
            return (
                values,
                tuple(table_ids),
                tuple(missing),
                tuple(missing_positions),
                int(hits),
                int(misses),
            )
        return values, tuple(missing), tuple(missing_positions), int(hits), int(misses)

    def resolve_ids_many(self, keys, *, normalized=False):
        keys = tuple(keys or ())
        if not bool(normalized):
            keys = tuple(self.normalize_key(key) for key in keys)
        ids = []
        missing = []
        missing_positions = []
        hits = 0
        misses = 0
        for idx, key in enumerate(keys):
            table_id = self.ids.get(key)
            if table_id is None:
                ids.append(-1)
                missing.append(key)
                missing_positions.append(idx)
                misses += 1
            else:
                ids.append(int(table_id))
                hits += 1
        self.batch_resolves += 1
        self.hits += int(hits)
        self.misses += int(misses)
        self.last_batch_size = int(len(keys))
        self.last_batch_hits = int(hits)
        self.last_batch_misses = int(misses)
        return ids, tuple(missing), tuple(missing_positions), int(hits), int(misses)

    def resolve_current_ids_many(self, keys, *, normalized=False):
        """Resolve ids only for entries populated in the current revision."""

        keys = tuple(keys or ())
        if not bool(normalized):
            keys = tuple(self.normalize_key(key) for key in keys)
        kernel = _cpp_table_kernel("dict_resolve_current_ids_many")
        if kernel is not None:
            try:
                (
                    ids,
                    missing,
                    missing_positions,
                    hits,
                    misses,
                ) = kernel(self.entries, self.ids, keys)
                hits = int(hits)
                misses = int(misses)
                self.batch_resolves += 1
                self.hits += hits
                self.misses += misses
                self.last_batch_size = int(len(keys))
                self.last_batch_hits = hits
                self.last_batch_misses = misses
                self.cpp_resolves += 1
                return ids, missing, missing_positions, hits, misses
            except Exception:
                pass
        ids = []
        missing = []
        missing_positions = []
        hits = 0
        misses = 0
        entries = self.entries
        table_ids = self.ids
        missing_marker = object()
        get_entry = entries.get
        get_id = table_ids.get
        for idx, key in enumerate(keys):
            if get_entry(key, missing_marker) is missing_marker:
                ids.append(-1)
                missing.append(key)
                missing_positions.append(idx)
                misses += 1
                continue
            table_id = get_id(key)
            if table_id is None:
                ids.append(-1)
                missing.append(key)
                missing_positions.append(idx)
                misses += 1
                continue
            ids.append(int(table_id))
            hits += 1
        self.batch_resolves += 1
        self.hits += int(hits)
        self.misses += int(misses)
        self.last_batch_size = int(len(keys))
        self.last_batch_hits = int(hits)
        self.last_batch_misses = int(misses)
        return ids, tuple(missing), tuple(missing_positions), int(hits), int(misses)

    def values_for_ids(self, ids):
        values = []
        for table_id in tuple(ids or ()):
            table_id = int(table_id)
            if table_id < 0 or table_id >= len(self.payloads):
                values.append(None)
            else:
                values.append(self.payloads[table_id])
        return values

    def reset_for_revision(self, revision):
        revision = int(revision)
        if int(self.revision) == revision:
            return False
        self.revision = revision
        self.entries.clear()
        self.payloads = [None] * len(self.payloads)
        self.family_counts.clear()
        self.blocks = 0
        self.last_batch_size = 0
        self.last_batch_hits = 0
        self.last_batch_misses = 0
        self.resets += 1
        return True

    def _put_normalized(self, key, value, family_name=None):
        if _contextual_boundary_payload_kind(value) != "packed":
            return False
        previous = self.entries.get(key)
        is_new = previous is None
        if previous is not None:
            self.blocks -= self.payload_block_count(previous)
            table_id = int(self.ids[key])
            self.payloads[table_id] = value
        elif key in self.ids:
            table_id = int(self.ids[key])
            if table_id >= len(self.payloads):
                self.payloads.extend(
                    [None] * (int(table_id) + 1 - len(self.payloads))
                )
            self.payloads[table_id] = value
        else:
            table_id = len(self.payloads)
            self.ids[key] = table_id
            self.payloads.append(value)
        self.entries[key] = value
        self.blocks += self.payload_block_count(value)
        self.puts += 1
        if is_new and family_name is not None:
            family = str(family_name)
            self.family_counts[family] = int(self.family_counts.get(family, 0)) + 1
        return True

    def put(self, key, value, family_name=None):
        return self._put_normalized(
            self.normalize_key(key),
            value,
            family_name=family_name,
        )

    def put_many(self, keys, values, family_name=None, *, normalized=False):
        keys = tuple(keys or ())
        if not bool(normalized):
            keys = tuple(self.normalize_key(key) for key in keys)
        values = tuple(values or ())
        kernel = _cpp_table_kernel("dict_put_many_packed")
        if kernel is not None:
            try:
                stored, block_delta, puts = kernel(
                    self.entries,
                    self.ids,
                    self.payloads,
                    keys,
                    values,
                    True,
                    False,
                    self.family_counts,
                    None if family_name is None else str(family_name),
                )
                self.blocks += int(block_delta)
                self.puts += int(puts)
                self.batch_stores += 1
                self.cpp_stores += 1
                return int(stored)
            except Exception:
                pass
        stored = 0
        for key, value in zip(keys, values):
            if self._put_normalized(key, value, family_name=family_name):
                stored += 1
        self.batch_stores += 1
        return int(stored)

    def put_many_packed(self, keys, values, family_name=None, *, normalized=False):
        """Store known packed contextual payload pairs with less Python overhead."""

        keys = tuple(keys or ())
        if not bool(normalized):
            keys = tuple(self.normalize_key(key) for key in keys)
        values = tuple(values or ())
        kernel = _cpp_table_kernel("dict_put_many_packed")
        if kernel is not None:
            try:
                stored, block_delta, puts = kernel(
                    self.entries,
                    self.ids,
                    self.payloads,
                    keys,
                    values,
                    True,
                    False,
                    self.family_counts,
                    None if family_name is None else str(family_name),
                )
                self.blocks += int(block_delta)
                self.puts += int(puts)
                self.batch_stores += 1
                self.cpp_stores += 1
                return int(stored)
            except Exception:
                pass
        stored = 0
        entries = self.entries
        ids = self.ids
        payloads = self.payloads
        family = None if family_name is None else str(family_name)
        for key, value in zip(keys, values):
            try:
                first, second = value
            except Exception:
                continue
            first_blocks = getattr(first, "blocks", None)
            second_blocks = getattr(second, "blocks", None)
            if not (
                bool(getattr(first, "_pyqed_packed_boundary_tensor", False))
                and bool(getattr(second, "_pyqed_packed_boundary_tensor", False))
            ):
                continue
            previous = entries.get(key)
            is_new = previous is None
            if previous is not None:
                self.blocks -= (
                    int(len(getattr(previous[0], "blocks", ()) or ()))
                    + int(len(getattr(previous[1], "blocks", ()) or ()))
                )
                table_id = int(ids[key])
                payloads[table_id] = value
            elif key in ids:
                table_id = int(ids[key])
                if table_id >= len(payloads):
                    payloads.extend([None] * (int(table_id) + 1 - len(payloads)))
                payloads[table_id] = value
            else:
                table_id = len(payloads)
                ids[key] = table_id
                payloads.append(value)
            entries[key] = value
            self.blocks += (
                int(len(first_blocks or ()))
                + int(len(second_blocks or ()))
            )
            self.puts += 1
            if is_new and family is not None:
                self.family_counts[family] = int(self.family_counts.get(family, 0)) + 1
            stored += 1
        self.batch_stores += 1
        return int(stored)

    def discard(self, key, *, normalized=False):
        key = self.normalize_key(key) if not bool(normalized) else key
        previous = self.entries.pop(key, None)
        if previous is None:
            return False
        self.blocks -= self.payload_block_count(previous)
        table_id = self.ids.get(key)
        if table_id is not None:
            table_id = int(table_id)
            if 0 <= table_id < len(self.payloads):
                self.payloads[table_id] = None
        self.evictions += 1
        return True

    @property
    def n_entries(self):
        return int(len(self.entries))

    @property
    def n_blocks(self):
        return int(self.blocks)

    @property
    def stats(self):
        return {
            "kind": "abelian_packed_contextual_boundary_table",
            "side": str(self.side),
            "bond": int(self.bond),
            "revision": int(self.revision),
            "entries": self.n_entries,
            "blocks": self.n_blocks,
            "ids": int(len(self.ids)),
            "payloads": int(len(self.payloads)),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "puts": int(self.puts),
            "batch_resolves": int(self.batch_resolves),
            "batch_stores": int(self.batch_stores),
            "cpp_resolves": int(self.cpp_resolves),
            "cpp_stores": int(self.cpp_stores),
            "last_batch_size": int(self.last_batch_size),
            "last_batch_hits": int(self.last_batch_hits),
            "last_batch_misses": int(self.last_batch_misses),
            "resets": int(self.resets),
            "evictions": int(self.evictions),
            "families": dict(self.family_counts),
        }


@dataclass
class AbelianSameSidePBoundaryValueTable:
    """Revision-aware table for exact same-side P boundary operator values."""

    side: str
    bond: int = -1
    revision: int = -1
    entries: OrderedDict = field(default_factory=OrderedDict)
    ids: dict = field(default_factory=dict)
    payloads: list = field(default_factory=list)
    blocks: int = 0
    hits: int = 0
    misses: int = 0
    puts: int = 0
    batch_resolves: int = 0
    batch_stores: int = 0
    cpp_resolves: int = 0
    cpp_stores: int = 0
    last_batch_size: int = 0
    last_batch_hits: int = 0
    last_batch_misses: int = 0
    resets: int = 0
    evictions: int = 0
    source: str = "abelian_same_side_p_boundary_value_table"

    @staticmethod
    def normalize_key(key):
        return (tuple(key[0]), str(key[1]))

    @staticmethod
    def payload_block_count(value):
        if value is None:
            return 0
        blocks = getattr(value, "blocks", None)
        if blocks is not None:
            return int(len(blocks))
        data = getattr(value, "data", None)
        if data is not None:
            return int(len(data))
        return 0

    def reset_for_revision(self, revision):
        revision = int(revision)
        if int(self.revision) == revision:
            return False
        self.revision = revision
        self.entries.clear()
        self.payloads = [None] * len(self.payloads)
        self.blocks = 0
        self.last_batch_size = 0
        self.last_batch_hits = 0
        self.last_batch_misses = 0
        self.resets += 1
        return True

    def get(self, key):
        key = self.normalize_key(key)
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def resolve_many(self, keys, *, normalized=False, return_ids=False):
        keys = tuple(keys or ())
        if not bool(normalized):
            keys = tuple(self.normalize_key(key) for key in keys)
        kernel = _cpp_table_kernel("dict_resolve_many")
        if kernel is not None:
            try:
                (
                    values,
                    table_ids,
                    missing,
                    missing_positions,
                    hits,
                    misses,
                ) = kernel(self.entries, self.ids, keys)
                hits = int(hits)
                misses = int(misses)
                self.batch_resolves += 1
                self.hits += hits
                self.misses += misses
                self.last_batch_size = int(len(keys))
                self.last_batch_hits = hits
                self.last_batch_misses = misses
                self.cpp_resolves += 1
                if bool(return_ids):
                    return (
                        values,
                        table_ids,
                        missing,
                        missing_positions,
                        hits,
                        misses,
                    )
                return values, missing, missing_positions, hits, misses
            except Exception:
                pass
        values = [None] * len(keys)
        table_ids = [-1] * len(keys)
        missing = []
        missing_positions = []
        hits = 0
        misses = 0
        for idx, key in enumerate(keys):
            value = self.entries.get(key)
            if value is None:
                missing.append(key)
                missing_positions.append(idx)
                misses += 1
                continue
            values[idx] = value
            table_id = self.ids.get(key)
            if table_id is not None:
                table_ids[idx] = int(table_id)
            hits += 1
        self.batch_resolves += 1
        self.hits += int(hits)
        self.misses += int(misses)
        self.last_batch_size = int(len(keys))
        self.last_batch_hits = int(hits)
        self.last_batch_misses = int(misses)
        if bool(return_ids):
            return (
                values,
                tuple(table_ids),
                tuple(missing),
                tuple(missing_positions),
                int(hits),
                int(misses),
            )
        return values, tuple(missing), tuple(missing_positions), int(hits), int(misses)

    def resolve_current_ids_many(self, keys, *, normalized=False):
        keys = tuple(keys or ())
        if not bool(normalized):
            keys = tuple(self.normalize_key(key) for key in keys)
        kernel = _cpp_table_kernel("dict_resolve_current_ids_many")
        if kernel is not None:
            try:
                (
                    table_ids,
                    missing,
                    missing_positions,
                    hits,
                    misses,
                ) = kernel(self.entries, self.ids, keys)
                hits = int(hits)
                misses = int(misses)
                self.batch_resolves += 1
                self.hits += hits
                self.misses += misses
                self.last_batch_size = int(len(keys))
                self.last_batch_hits = hits
                self.last_batch_misses = misses
                self.cpp_resolves += 1
                return table_ids, missing, missing_positions, hits, misses
            except Exception:
                pass
        table_ids = [-1] * len(keys)
        missing = []
        missing_positions = []
        hits = 0
        misses = 0
        entries = self.entries
        ids = self.ids
        missing_marker = object()
        get_entry = entries.get
        get_id = ids.get
        for idx, key in enumerate(keys):
            if get_entry(key, missing_marker) is missing_marker:
                missing.append(key)
                missing_positions.append(idx)
                misses += 1
                continue
            table_id = get_id(key)
            if table_id is None:
                missing.append(key)
                missing_positions.append(idx)
                misses += 1
                continue
            table_ids[idx] = int(table_id)
            hits += 1
        self.batch_resolves += 1
        self.hits += int(hits)
        self.misses += int(misses)
        self.last_batch_size = int(len(keys))
        self.last_batch_hits = int(hits)
        self.last_batch_misses = int(misses)
        return (
            tuple(table_ids),
            tuple(missing),
            tuple(missing_positions),
            int(hits),
            int(misses),
        )

    def values_for_ids(self, ids):
        values = []
        for table_id in tuple(ids or ()):
            table_id = int(table_id)
            if table_id < 0 or table_id >= len(self.payloads):
                values.append(None)
            else:
                values.append(self.payloads[table_id])
        return values

    def _put_normalized(self, key, value):
        if value is None:
            return False
        previous = self.entries.get(key)
        is_new = previous is None
        if previous is not None:
            self.blocks -= self.payload_block_count(previous)
            table_id = int(self.ids[key])
            self.payloads[table_id] = value
        elif key in self.ids:
            table_id = int(self.ids[key])
            if table_id >= len(self.payloads):
                self.payloads.extend(
                    [None] * (int(table_id) + 1 - len(self.payloads))
                )
            self.payloads[table_id] = value
        else:
            table_id = len(self.payloads)
            self.ids[key] = table_id
            self.payloads.append(value)
        self.entries[key] = value
        self.blocks += self.payload_block_count(value)
        self.puts += 1
        return bool(is_new or previous is not value)

    def put(self, key, value):
        return self._put_normalized(self.normalize_key(key), value)

    def put_many(self, keys, values, *, normalized=False):
        keys = tuple(keys or ())
        if not bool(normalized):
            keys = tuple(self.normalize_key(key) for key in keys)
        values = tuple(values or ())
        kernel = _cpp_table_kernel("dict_put_many_packed")
        if kernel is not None:
            try:
                stored, block_delta, puts = kernel(
                    self.entries,
                    self.ids,
                    self.payloads,
                    keys,
                    values,
                    False,
                    True,
                )
                self.blocks += int(block_delta)
                self.puts += int(puts)
                self.batch_stores += 1
                self.cpp_stores += 1
                return int(stored)
            except Exception:
                pass
        stored = 0
        for normalized_key, value in zip(keys, values):
            if self._put_normalized(normalized_key, value):
                stored += 1
        self.batch_stores += 1
        return int(stored)

    def put_many_packed(self, keys, values, *, normalized=False):
        keys = tuple(keys or ())
        if not bool(normalized):
            keys = tuple(self.normalize_key(key) for key in keys)
        values = tuple(values or ())
        kernel = _cpp_table_kernel("dict_put_many_packed")
        if kernel is not None:
            try:
                stored, block_delta, puts = kernel(
                    self.entries,
                    self.ids,
                    self.payloads,
                    keys,
                    values,
                    False,
                    True,
                )
                self.blocks += int(block_delta)
                self.puts += int(puts)
                self.batch_stores += 1
                self.cpp_stores += 1
                return int(stored)
            except Exception:
                pass
        stored = 0
        entries = self.entries
        ids = self.ids
        payloads = self.payloads
        for normalized_key, value in zip(keys, values):
            if value is None:
                continue
            if not bool(getattr(value, "_pyqed_packed_boundary_tensor", False)):
                if self._put_normalized(normalized_key, value):
                    stored += 1
                continue
            previous = entries.get(normalized_key)
            is_new = previous is None
            if previous is not None:
                self.blocks -= int(len(getattr(previous, "blocks", ()) or ()))
                table_id = int(ids[normalized_key])
                payloads[table_id] = value
            elif normalized_key in ids:
                table_id = int(ids[normalized_key])
                if table_id >= len(payloads):
                    payloads.extend([None] * (int(table_id) + 1 - len(payloads)))
                payloads[table_id] = value
            else:
                table_id = len(payloads)
                ids[normalized_key] = table_id
                payloads.append(value)
            entries[normalized_key] = value
            self.blocks += int(len(getattr(value, "blocks", ()) or ()))
            self.puts += 1
            if is_new or previous is not value:
                stored += 1
        self.batch_stores += 1
        return int(stored)

    def discard(self, key, *, normalized=False):
        key = self.normalize_key(key) if not bool(normalized) else key
        previous = self.entries.pop(key, None)
        if previous is None:
            return False
        self.blocks -= self.payload_block_count(previous)
        table_id = self.ids.get(key)
        if table_id is not None:
            table_id = int(table_id)
            if 0 <= table_id < len(self.payloads):
                self.payloads[table_id] = None
        self.evictions += 1
        return True

    @property
    def n_entries(self):
        return int(len(self.entries))

    @property
    def n_blocks(self):
        return int(self.blocks)

    @property
    def stats(self):
        return {
            "kind": "abelian_same_side_p_boundary_value_table",
            "side": str(self.side),
            "bond": int(self.bond),
            "revision": int(self.revision),
            "entries": self.n_entries,
            "blocks": self.n_blocks,
            "ids": int(len(self.ids)),
            "payloads": int(len(self.payloads)),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "puts": int(self.puts),
            "batch_resolves": int(self.batch_resolves),
            "batch_stores": int(self.batch_stores),
            "cpp_resolves": int(self.cpp_resolves),
            "cpp_stores": int(self.cpp_stores),
            "last_batch_size": int(self.last_batch_size),
            "last_batch_hits": int(self.last_batch_hits),
            "last_batch_misses": int(self.last_batch_misses),
            "resets": int(self.resets),
            "evictions": int(self.evictions),
        }


@dataclass(frozen=True)
class AbelianContextualEntryBuildResult:
    """Result of building one direct contextual family."""

    entries: list | None
    seconds: float


def _increment_counter(stats, key, amount=1):
    stats[str(key)] = int(stats.get(str(key), 0)) + int(amount)


def _contextual_boundary_payload_kind(result):
    if result is None:
        return "missing"
    try:
        first, second = result
    except Exception:
        return "other"
    first_packed = is_abelian_packed_boundary_tensor(first)
    second_packed = is_abelian_packed_boundary_tensor(second)
    if first_packed and second_packed:
        return "packed"
    if first_packed or second_packed:
        return "mixed"
    return "legacy"


def _contextual_boundary_payload_counts(values):
    counts = {"packed": 0, "legacy": 0, "mixed": 0, "missing": 0, "other": 0}
    for value in tuple(values or ()):
        kind = _contextual_boundary_payload_kind(value)
        counts[kind] = int(counts.get(kind, 0)) + 1
    return counts


def make_contextual_family_records(terms, bond):
    """Split full JW-pattern terms into left/local/right contextual records."""

    bond = int(bond)
    return tuple(
        (
            tuple(pattern[:bond]),
            str(pattern[bond]),
            str(pattern[bond + 1]),
            tuple(pattern[bond + 2:]),
            complex(coeff),
        )
        for pattern, coeff in terms
    )


@dataclass(frozen=True)
class AbelianDirectFamilyDispatchPlan:
    """Named direct-family dispatch handle consumed by the C++ owner."""

    family_names: tuple
    build_piece: object

    @classmethod
    def from_builders(cls, builders_by_family):
        builders = {
            str(family_name): builder
            for family_name, builder in (builders_by_family or {}).items()
        }

        def _build_piece(family_name):
            return builders[str(family_name)]()

        return cls(tuple(builders), _build_piece)


@dataclass(frozen=True)
class AbelianDirectFamilyLiteralPlan:
    """Named direct-family payload pieces already prepared for the C++ owner."""

    family_names: tuple
    family_pieces: tuple

    @classmethod
    def from_pieces(cls, pieces_by_family):
        pieces = {
            str(family_name): piece
            for family_name, piece in (pieces_by_family or {}).items()
        }
        return cls(tuple(pieces), tuple(pieces.values()))


@dataclass(frozen=True)
class AbelianContextualFamilyDispatchPlan:
    """Family-name dispatch handle for contextual direct-family payload builds."""

    terms_by_family: dict
    build_family: object

    @classmethod
    def from_pattern_terms(cls, pattern_terms, build_family):
        return cls(
            {
                str(family_name): terms
                for family_name, terms in (pattern_terms or {}).items()
            },
            build_family,
        )

    @property
    def family_names(self):
        return tuple(self.terms_by_family)

    def build_piece(self, family_name):
        family_name = str(family_name)
        return self.build_family(
            family_name,
            self.terms_by_family[family_name],
        )


def contextual_boundary_keys(records):
    """Return unique left/right boundary keys for contextual records."""

    left = OrderedDict()
    right = OrderedDict()
    for left_pattern, left_piece, right_piece, right_pattern, _coeff in records:
        left[(tuple(left_pattern), str(left_piece))] = None
        right[(tuple(right_pattern), str(right_piece))] = None
    left_keys = tuple(
        sorted(left, key=lambda key: (len(key[0]), key[0], key[1]))
    )
    right_keys = tuple(
        sorted(right, key=lambda key: (len(key[0]), tuple(reversed(key[0])), key[0], key[1]))
    )
    return left_keys, right_keys


@dataclass(frozen=True)
class AbelianDirectRoutePlan:
    """Integer route plan for one contextual Abelian direct family.

    The current Python reference path still resolves the boundary tensors, but
    this object separates static route metadata from numeric boundary refreshes.
    That is the shape the native route-table builder needs: unique boundary keys
    plus integer left/right ids and packed coefficients.
    """

    family_name: str
    bond: int
    left_keys: tuple
    right_keys: tuple
    left_ids: np.ndarray
    right_ids: np.ndarray
    coeffs: np.ndarray
    records: tuple = field(repr=False, compare=False)
    pair_left_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    pair_right_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    pair_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.complex128))
    signature: tuple = ()

    @classmethod
    def from_records(cls, family_name, records, *, bond=-1, sort_keys=False):
        records = tuple(records or ())
        if sort_keys:
            left_keys, right_keys = contextual_boundary_keys(records)
            left_index = {key: idx for idx, key in enumerate(left_keys)}
            right_index = {key: idx for idx, key in enumerate(right_keys)}
        else:
            left_keys = []
            right_keys = []
            left_index = {}
            right_index = {}
        left_ids = np.empty(len(records), dtype=np.int64)
        right_ids = np.empty(len(records), dtype=np.int64)
        coeffs = np.empty(len(records), dtype=np.complex128)
        pair_coeffs = OrderedDict()
        for idx, (
            left_pattern,
            left_piece,
            right_piece,
            right_pattern,
            coeff,
        ) in enumerate(records):
            left_key = (tuple(left_pattern), str(left_piece))
            right_key = (tuple(right_pattern), str(right_piece))
            left_id = left_index.get(left_key)
            if left_id is None:
                left_id = len(left_keys)
                left_index[left_key] = left_id
                left_keys.append(left_key)
            right_id = right_index.get(right_key)
            if right_id is None:
                right_id = len(right_keys)
                right_index[right_key] = right_id
                right_keys.append(right_key)
            left_ids[idx] = left_id
            right_ids[idx] = right_id
            coeffs[idx] = complex(coeff)
            pair_key = (int(left_id), int(right_id))
            pair_coeffs[pair_key] = complex(pair_coeffs.get(pair_key, 0.0)) + complex(coeff)
        packed_pairs = tuple(
            (left_id, right_id, coeff)
            for (left_id, right_id), coeff in pair_coeffs.items()
            if abs(complex(coeff)) > 1.0e-14
        )
        if packed_pairs:
            pair_left_ids = np.asarray(
                [left_id for left_id, _right_id, _coeff in packed_pairs],
                dtype=np.int64,
            )
            pair_right_ids = np.asarray(
                [right_id for _left_id, right_id, _coeff in packed_pairs],
                dtype=np.int64,
            )
            pair_coeff_values = np.asarray(
                [coeff for _left_id, _right_id, coeff in packed_pairs],
                dtype=np.complex128,
            )
        else:
            pair_left_ids = np.zeros(0, dtype=np.int64)
            pair_right_ids = np.zeros(0, dtype=np.int64)
            pair_coeff_values = np.zeros(0, dtype=np.complex128)
        pair_signature = tuple(
            (
                int(left_id),
                int(right_id),
                complex(coeff),
            )
            for left_id, right_id, coeff in zip(
                pair_left_ids,
                pair_right_ids,
                pair_coeff_values,
            )
        )
        signature = (
            "abelian_direct_route_plan",
            str(family_name),
            int(bond),
            tuple(left_keys),
            tuple(right_keys),
            pair_signature,
        )
        return cls(
            str(family_name),
            int(bond),
            tuple(left_keys),
            tuple(right_keys),
            left_ids,
            right_ids,
            coeffs,
            records,
            pair_left_ids,
            pair_right_ids,
            pair_coeff_values,
            signature,
        )

    @property
    def record_count(self):
        return int(self.coeffs.shape[0])

    @property
    def left_count(self):
        return int(len(self.left_keys))

    @property
    def right_count(self):
        return int(len(self.right_keys))

    @property
    def pair_count(self):
        return int(self.pair_coeffs.shape[0])

    @property
    def stats(self):
        return {
            "family": str(self.family_name),
            "bond": int(self.bond),
            "records": self.record_count,
            "pairs": self.pair_count,
            "coalesced_records": int(self.record_count - self.pair_count),
            "left_unique": self.left_count,
            "right_unique": self.right_count,
        }

    def iter_records(self):
        left_keys = self.left_keys
        right_keys = self.right_keys
        left_ids = self.left_ids
        right_ids = self.right_ids
        coeffs = self.coeffs
        for idx in range(self.record_count):
            left_pattern, left_piece = left_keys[int(left_ids[idx])]
            right_pattern, right_piece = right_keys[int(right_ids[idx])]
            yield (
                left_pattern,
                left_piece,
                right_piece,
                right_pattern,
                complex(coeffs[idx]),
            )


@dataclass(frozen=True)
class AbelianSameSidePRoutePlan:
    """Static integer route layout for same-side spatial P boundary updates."""

    side: str
    bond: int
    only_new: bool
    raw_keys: np.ndarray = field(repr=False, compare=False)
    offsets: np.ndarray = field(repr=False, compare=False)
    boundary_ids: np.ndarray = field(repr=False, compare=False)
    factors: np.ndarray = field(repr=False, compare=False)
    boundary_keys: tuple
    boundary_parent_ids: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64),
        repr=False,
        compare=False,
    )
    boundary_parent_keys: tuple = ()
    boundary_local_pieces: tuple = ()
    raw_key_tuples: tuple = ()
    term_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    signature: tuple = ()
    source: str = "same_side_p_route_plan"

    @classmethod
    def from_planned_terms(
        cls,
        *,
        side,
        bond,
        planned_terms,
        existing_keys=(),
        only_new=False,
    ):
        existing = {tuple(int(index) for index in key) for key in tuple(existing_keys or ())}
        boundary_index = OrderedDict()
        raw_keys = []
        offsets = []
        term_counts = []
        route_boundary_ids = []
        route_factors = []
        for raw_key, terms in tuple(planned_terms or ()):
            raw_key = tuple(int(index) for index in raw_key)
            if raw_key in existing:
                continue
            row_terms = OrderedDict()
            for boundary_key, factor in tuple(terms or ()):
                normalized_key = (tuple(boundary_key[0]), str(boundary_key[1]))
                row_terms[normalized_key] = (
                    complex(row_terms.get(normalized_key, 0.0))
                    + complex(factor)
                )
            row_terms = tuple(
                (key, factor)
                for key, factor in row_terms.items()
                if abs(complex(factor)) > 1.0e-14
            )
            if not row_terms:
                continue
            offsets.append(int(len(route_boundary_ids)))
            raw_keys.append(raw_key)
            term_counts.append(int(len(row_terms)))
            for normalized_key, factor in row_terms:
                boundary_id = boundary_index.get(normalized_key)
                if boundary_id is None:
                    boundary_id = len(boundary_index)
                    boundary_index[normalized_key] = boundary_id
                route_boundary_ids.append(int(boundary_id))
                route_factors.append(complex(factor))
        offsets.append(int(len(route_boundary_ids)))
        if raw_keys:
            raw_key_array = np.asarray(raw_keys, dtype=np.int64).reshape((-1, 4))
        else:
            raw_key_array = np.zeros((0, 4), dtype=np.int64)
        boundary_keys = tuple(boundary_index.keys())
        parent_index = OrderedDict()
        boundary_parent_ids = []
        boundary_local_pieces = []
        for boundary_key in boundary_keys:
            pattern, piece = boundary_key
            pattern = tuple(pattern)
            if str(side) == "left":
                parent_pattern = tuple(pattern[:-1])
                local_piece = str(pattern[-1]) if pattern else str(piece)
            else:
                parent_pattern = tuple(pattern[1:])
                local_piece = str(pattern[0]) if pattern else str(piece)
            parent_key = (parent_pattern, str(piece))
            parent_id = parent_index.get(parent_key)
            if parent_id is None:
                parent_id = len(parent_index)
                parent_index[parent_key] = parent_id
            boundary_parent_ids.append(int(parent_id))
            boundary_local_pieces.append(str(local_piece))
        signature = (
            "abelian_same_side_p_route_plan",
            str(side),
            int(bond),
            bool(only_new),
            int(len(raw_keys)),
            int(len(route_boundary_ids)),
            int(len(boundary_keys)),
        )
        return cls(
            side=str(side),
            bond=int(bond),
            only_new=bool(only_new),
            raw_keys=raw_key_array,
            offsets=np.asarray(offsets, dtype=np.int64),
            boundary_ids=np.asarray(route_boundary_ids, dtype=np.int64),
            factors=np.asarray(route_factors, dtype=np.complex128),
            boundary_keys=boundary_keys,
            boundary_parent_ids=np.asarray(boundary_parent_ids, dtype=np.int64),
            boundary_parent_keys=tuple(parent_index.keys()),
            boundary_local_pieces=tuple(boundary_local_pieces),
            raw_key_tuples=tuple(raw_keys),
            term_counts=np.asarray(term_counts, dtype=np.int64),
            signature=signature,
        )

    @property
    def records(self):
        return int(self.raw_keys.shape[0])

    @property
    def terms(self):
        return int(self.boundary_ids.shape[0])

    @property
    def boundary_key_count(self):
        return int(len(self.boundary_keys))

    @property
    def boundary_patterns(self):
        return tuple(pattern for pattern, _piece in self.boundary_keys)

    @property
    def stats(self):
        return {
            "kind": "abelian_same_side_p_route_plan",
            "side": str(self.side),
            "bond": int(self.bond),
            "only_new": bool(self.only_new),
            "records": self.records,
            "terms": self.terms,
            "boundary_keys": self.boundary_key_count,
        }

    def as_route_columns(self):
        return {
            "source": self.source,
            "side": str(self.side),
            "bond": int(self.bond),
            "raw_keys": self.raw_keys,
            "offsets": self.offsets,
            "boundary_ids": self.boundary_ids,
            "factors": self.factors,
            "boundary_keys": self.boundary_keys,
            "boundary_parent_ids": self.boundary_parent_ids,
            "boundary_parent_keys": self.boundary_parent_keys,
            "boundary_local_pieces": self.boundary_local_pieces,
            "records": self.records,
            "terms": self.terms,
            "boundary_key_count": self.boundary_key_count,
        }


def merge_abelian_same_side_p_route_plan(
    route_columns,
    boundary_results,
    *,
    operator_table=None,
    require_packed=True,
    enable_row_cache=False,
    source="packed_same_side_p_route_operator",
):
    """Build same-side P boundary operators from route arrays and packed payloads."""

    if isinstance(route_columns, dict):
        raw_keys = np.asarray(route_columns["raw_keys"], dtype=np.int64)
        offsets = np.asarray(route_columns["offsets"], dtype=np.int64)
        boundary_ids = np.asarray(route_columns["boundary_ids"], dtype=np.int64)
        factors = np.asarray(route_columns["factors"], dtype=np.complex128)
    else:
        raw_keys = np.asarray(route_columns.raw_keys, dtype=np.int64)
        offsets = np.asarray(route_columns.offsets, dtype=np.int64)
        boundary_ids = np.asarray(route_columns.boundary_ids, dtype=np.int64)
        factors = np.asarray(route_columns.factors, dtype=np.complex128)
    boundary_results = tuple(boundary_results or ())
    unsupported = 0
    for tensor in boundary_results:
        if tensor is None:
            continue
        if bool(require_packed) and not is_abelian_packed_boundary_tensor(tensor):
            unsupported += 1
    if unsupported:
        return {
            "complete": False,
            "items": (),
            "built": 0,
            "failures": 0,
            "blocks": 0,
            "packed_calls": 0,
            "packed_terms": 0,
            "packed_input_blocks": 0,
            "last_terms": 0,
            "last_output_blocks": 0,
            "unsupported": int(unsupported),
        }

    built_items = []
    failures = 0
    blocks = 0
    packed_terms = 0
    packed_input_blocks = 0
    packed_last_terms = 0
    packed_last_output_blocks = 0
    item_cache = {}
    row_cache = {} if bool(enable_row_cache) else None
    row_cache_hits = 0
    row_cache_builds = 0

    def _items(boundary_id):
        cached = item_cache.get(boundary_id)
        if cached is not None:
            return cached
        tensor = boundary_results[int(boundary_id)]
        keys, tensor_blocks, dirs, qns = abelian_packed_tensor_items(tensor)
        cached = (tuple(keys), tuple(tensor_blocks), list(dirs), qns)
        item_cache[int(boundary_id)] = cached
        return cached

    for row in range(int(raw_keys.shape[0])):
        start = int(offsets[row])
        stop = int(offsets[row + 1])
        if stop <= start:
            failures += 1
            continue
        row_signature = None
        if row_cache is not None:
            row_signature = tuple(
                (int(boundary_ids[item]), complex(factors[item]))
                for item in range(start, stop)
            )
            cached_row = row_cache.get(row_signature)
            if cached_row is not None:
                cached_operator, cached_input_blocks = cached_row
                raw_key = tuple(int(index) for index in raw_keys[row])
                built_items.append((raw_key, cached_operator))
                blocks += int(len(cached_operator))
                terms_count = int(stop - start)
                packed_terms += terms_count
                packed_input_blocks += int(cached_input_blocks)
                packed_last_terms = terms_count
                packed_last_output_blocks = int(len(cached_operator))
                row_cache_hits += 1
                continue
        data = OrderedDict()
        dirs = None
        qns = None
        input_blocks = 0
        row_failed = False
        for item in range(start, stop):
            boundary_id = int(boundary_ids[item])
            if boundary_id < 0 or boundary_id >= len(boundary_results):
                row_failed = True
                break
            tensor = boundary_results[boundary_id]
            if tensor is None:
                row_failed = True
                break
            keys, tensor_blocks, tensor_dirs, tensor_qns = _items(boundary_id)
            if dirs is None and tensor_dirs:
                dirs = list(tensor_dirs)
            if qns is None and tensor_qns is not None:
                qns = tensor_qns
            scalar = complex(factors[item])
            input_blocks += int(len(tensor_blocks))
            for key, block in zip(keys, tensor_blocks):
                key = tuple(key)
                block = np.asarray(block)
                old = data.get(key)
                if old is None:
                    if scalar == 1.0:
                        data[key] = block.copy()
                    elif scalar == -1.0:
                        data[key] = -block
                    else:
                        data[key] = scalar * block
                    continue
                if tuple(old.shape) != tuple(block.shape):
                    row_failed = True
                    break
                if scalar == 1.0:
                    old += block
                elif scalar == -1.0:
                    old -= block
                else:
                    old += scalar * block
            if row_failed:
                break
        if row_failed or not data:
            failures += 1
            continue
        operator = AbelianPackedBoundaryTensor(
            tuple(data.keys()),
            tuple(data.values()),
            dirs=[] if dirs is None else dirs,
            qns=qns,
            source=source,
        )
        if row_cache is not None:
            row_cache[row_signature] = (operator, int(input_blocks))
            row_cache_builds += 1
        raw_key = tuple(int(index) for index in raw_keys[row])
        built_items.append((raw_key, operator))
        blocks += int(len(operator))
        terms_count = int(stop - start)
        packed_terms += terms_count
        packed_input_blocks += int(input_blocks)
        packed_last_terms = terms_count
        packed_last_output_blocks = int(len(operator))

    built = int(raw_keys.shape[0]) - int(failures)
    if failures:
        return {
            "complete": False,
            "items": tuple(built_items),
            "built": int(built),
            "failures": int(failures),
            "blocks": int(blocks),
            "packed_calls": int(built),
            "packed_terms": int(packed_terms),
            "packed_input_blocks": int(packed_input_blocks),
            "last_terms": int(packed_last_terms),
            "last_output_blocks": int(packed_last_output_blocks),
            "row_cache_hits": int(row_cache_hits),
            "row_cache_builds": int(row_cache_builds),
            "unsupported": 0,
        }
    if operator_table is not None:
        for raw_key, operator in built_items:
            operator_table.add_operator(raw_key, operator)
        built_items = []
    return {
        "complete": True,
        "items": tuple(built_items),
        "built": int(built),
        "failures": int(failures),
        "blocks": int(blocks),
        "packed_calls": int(built),
        "packed_terms": int(packed_terms),
        "packed_input_blocks": int(packed_input_blocks),
        "last_terms": int(packed_last_terms),
        "last_output_blocks": int(packed_last_output_blocks),
        "row_cache_hits": int(row_cache_hits),
        "row_cache_builds": int(row_cache_builds),
        "unsupported": 0,
    }


def _abelian_plan_token(value):
    """Return a stable, hashable token for plan signatures."""

    if isinstance(value, np.ndarray):
        return tuple(_abelian_plan_token(item) for item in value.reshape(-1).tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.complexfloating, complex)):
        value = complex(value)
        return (float(value.real), float(value.imag))
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        return tuple(_abelian_plan_token(item) for item in value)
    if isinstance(value, list):
        return tuple(_abelian_plan_token(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    _abelian_plan_token(key),
                    _abelian_plan_token(item),
                )
                for key, item in value.items()
            )
        )
    if hasattr(value, "labels") and hasattr(value, "components"):
        return (
            type(value).__name__,
            tuple(str(label) for label in value.labels),
            tuple(_abelian_plan_token(item) for item in value.components),
        )
    return repr(value)


def _abelian_layout_signature(layout):
    items = []
    for key, shape in tuple(layout or ()):
        items.append(
            (
                tuple(_abelian_plan_token(qn) for qn in tuple(key)),
                tuple(int(dim) for dim in tuple(shape)),
            )
        )
    return tuple(items)


def _safe_abelian_sector_signature(key, dirs):
    try:
        if len(tuple(key)) == len(tuple(dirs)):
            return _abelian_plan_token(abelian_sector_signature(tuple(key), tuple(dirs)))
    except Exception:
        pass
    return tuple(_abelian_plan_token(qn) for qn in tuple(key))


@dataclass(frozen=True)
class AbelianSymmetryAdapter:
    """Stable symmetry/layout descriptor for packed Abelian local actions."""

    name: str = "u1"
    dirs: tuple = ()
    layout_signature: tuple = ()
    sector_signatures: tuple = ()

    @classmethod
    def from_layout(cls, layout, *, dirs=(), name="u1"):
        dirs = tuple(int(direction) for direction in tuple(dirs or ()))
        layout_signature = _abelian_layout_signature(layout)
        sector_signatures = tuple(
            _safe_abelian_sector_signature(key, dirs)
            for key, _shape in tuple(layout or ())
        )
        return cls(
            str(name),
            dirs,
            layout_signature,
            sector_signatures,
        )

    @property
    def signature(self):
        return (
            str(self.name),
            tuple(self.dirs),
            tuple(self.layout_signature),
            tuple(self.sector_signatures),
        )

    def compatible_layout(self, layout):
        return _abelian_layout_signature(layout) == self.layout_signature

    @property
    def stats(self):
        return {
            "kind": "abelian_symmetry_adapter",
            "name": str(self.name),
            "dirs": tuple(self.dirs),
            "sectors": int(len(self.layout_signature)),
            "sector_signatures": int(len(set(self.sector_signatures))),
        }


@dataclass(frozen=True)
class AbelianOperatorFamilyPlan:
    """Block2-style integer plan for one direct operator family."""

    family_name: str
    bond: int
    family_kind: str
    route_plan: AbelianDirectRoutePlan = field(repr=False, compare=False)
    signature: tuple = ()

    @classmethod
    def from_route_plan(cls, route_plan, *, family_kind="contextual_direct"):
        route_plan = route_plan
        signature = (
            str(family_kind),
            str(route_plan.family_name),
            int(route_plan.bond),
            tuple(
                (
                    tuple(_abelian_plan_token(part) for part in left_pattern),
                    str(left_piece),
                )
                for left_pattern, left_piece in route_plan.left_keys
            ),
            tuple(
                (
                    tuple(_abelian_plan_token(part) for part in right_pattern),
                    str(right_piece),
                )
                for right_pattern, right_piece in route_plan.right_keys
            ),
            tuple(int(value) for value in route_plan.left_ids.tolist()),
            tuple(int(value) for value in route_plan.right_ids.tolist()),
            tuple(_abelian_plan_token(value) for value in route_plan.coeffs.tolist()),
            tuple(int(value) for value in route_plan.pair_left_ids.tolist()),
            tuple(int(value) for value in route_plan.pair_right_ids.tolist()),
            tuple(
                _abelian_plan_token(value)
                for value in route_plan.pair_coeffs.tolist()
            ),
        )
        return cls(
            str(route_plan.family_name),
            int(route_plan.bond),
            str(family_kind),
            route_plan,
            signature,
        )

    @property
    def stats(self):
        stats = dict(self.route_plan.stats)
        stats["kind"] = "abelian_operator_family_plan"
        stats["family_kind"] = str(self.family_kind)
        stats["signature_terms"] = int(len(self.signature))
        return stats

    def cache_key(self, *, layout_signature=(), revision=0):
        return (
            "abelian_operator_family_plan",
            int(revision),
            tuple(layout_signature or ()),
            self.signature,
        )


def _packed_boundary_table_signature(table):
    if table is None:
        return None
    return (
        str(getattr(table, "side", "")),
        int(getattr(table, "bond", -1)),
        str(getattr(table, "source", "")),
    )


@dataclass(frozen=True)
class AbelianMovingEnvironmentTables:
    """Persistent packed left/right boundary table handles for one bond."""

    bond: int
    revision: int = 0
    left_table: AbelianPackedContextualBoundaryTable | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    right_table: AbelianPackedContextualBoundaryTable | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    left_signature: tuple | None = None
    right_signature: tuple | None = None

    @classmethod
    def from_contextual_builder(cls, builder, *, bond=-1, revision=0):
        left_table = getattr(builder, "left_packed_boundary_table", None)
        right_table = getattr(builder, "right_packed_boundary_table", None)
        return cls(
            int(bond),
            int(revision),
            left_table,
            right_table,
            _packed_boundary_table_signature(left_table),
            _packed_boundary_table_signature(right_table),
        )

    @property
    def signature(self):
        return (
            "abelian_moving_environment_tables",
            int(self.bond),
            self.left_signature,
            self.right_signature,
        )

    @property
    def stats(self):
        left_entries = 0 if self.left_table is None else self.left_table.n_entries
        right_entries = 0 if self.right_table is None else self.right_table.n_entries
        return {
            "kind": "abelian_moving_environment_tables",
            "bond": int(self.bond),
            "revision": int(self.revision),
            "left_entries": int(left_entries),
            "right_entries": int(right_entries),
            "left_signature": self.left_signature,
            "right_signature": self.right_signature,
        }


@dataclass(frozen=True)
class AbelianLocalActionPlan:
    """Packed local-action plan keyed by symmetry, family, and boundary layout."""

    symmetry: AbelianSymmetryAdapter
    family_plan: AbelianOperatorFamilyPlan
    moving_tables: AbelianMovingEnvironmentTables
    backend: str = "python"
    left_table_ids: tuple = ()
    right_table_ids: tuple = ()
    layout_signature: tuple = ()

    @classmethod
    def from_boundary_batch(
        cls,
        *,
        family_plan,
        moving_tables,
        boundary_batch,
        layout=(),
        dirs=(),
        backend="python",
        symmetry_name="u1",
    ):
        symmetry = AbelianSymmetryAdapter.from_layout(
            layout,
            dirs=dirs,
            name=symmetry_name,
        )
        layout_signature = symmetry.layout_signature
        return cls(
            symmetry,
            family_plan,
            moving_tables,
            str(backend),
            tuple(int(value) for value in getattr(boundary_batch, "left_table_ids", ()) or ()),
            tuple(int(value) for value in getattr(boundary_batch, "right_table_ids", ()) or ()),
            layout_signature,
        )

    @property
    def signature(self):
        return (
            "abelian_local_action_plan",
            str(self.backend),
            self.symmetry.signature,
            self.family_plan.signature,
            self.moving_tables.signature,
            tuple(self.left_table_ids),
            tuple(self.right_table_ids),
            tuple(self.layout_signature),
        )

    def cache_key(self):
        return self.signature

    def stale_for(self, *, layout=(), moving_tables=None):
        if layout and _abelian_layout_signature(layout) != self.layout_signature:
            return True
        if moving_tables is not None and moving_tables.signature != self.moving_tables.signature:
            return True
        return False

    @property
    def stats(self):
        return {
            "kind": "abelian_local_action_plan",
            "backend": str(self.backend),
            "family": str(self.family_plan.family_name),
            "bond": int(self.family_plan.bond),
            "left_table_ids": int(sum(1 for value in self.left_table_ids if int(value) >= 0)),
            "right_table_ids": int(sum(1 for value in self.right_table_ids if int(value) >= 0)),
            "layout_sectors": int(len(self.layout_signature)),
            "moving_revision": int(self.moving_tables.revision),
        }


@dataclass
class AbelianLocalActionPlanCache:
    """Small signature cache that rebuilds only when layout/table shape changes."""

    plans: dict = field(default_factory=dict)
    stats: dict = field(
        default_factory=lambda: {
            "builds": 0,
            "hits": 0,
            "invalidations": 0,
        }
    )

    def get_or_build(self, key, factory):
        try:
            hash(key)
        except TypeError:
            key = _abelian_plan_token(key)
        plan = self.plans.get(key)
        if plan is not None:
            self.stats["hits"] = int(self.stats.get("hits", 0)) + 1
            self.stats["last_hit_key_hash"] = int(hash(key))
            return plan, True
        plan = factory()
        self.plans[key] = plan
        self.stats["builds"] = int(self.stats.get("builds", 0)) + 1
        self.stats["last_build_key_hash"] = int(hash(key))
        self.stats["size"] = int(len(self.plans))
        return plan, False

    def invalidate(self, predicate=None):
        if predicate is None:
            count = len(self.plans)
            self.plans.clear()
        else:
            keys = [key for key, plan in self.plans.items() if predicate(key, plan)]
            for key in keys:
                self.plans.pop(key, None)
            count = len(keys)
        self.stats["invalidations"] = int(self.stats.get("invalidations", 0)) + int(count)
        self.stats["size"] = int(len(self.plans))
        return int(count)


def abelian_generator_owner_from_support(support, bond, nsites):
    """Classify a generator support as left/right/local for a two-site center."""

    support = set(int(site) for site in support)
    bond = int(bond)
    nsites = int(nsites)
    if support and all(0 <= site < bond for site in support):
        return "left"
    if support and all(bond + 2 <= site < nsites for site in support):
        return "right"
    if support and all(bond <= site <= bond + 1 for site in support):
        return "local"
    return None


def abelian_generator_region_from_support(support, bond, nsites):
    """Classify support by which center-adjacent regions it touches."""

    support = set(int(site) for site in support)
    bond = int(bond)
    nsites = int(nsites)
    has_left = any(0 <= site < bond for site in support)
    has_local = any(bond <= site <= bond + 1 for site in support)
    has_right = any(bond + 2 <= site < nsites for site in support)
    if has_left and has_local and not has_right:
        return "left_local"
    if not has_left and has_local and has_right:
        return "local_right"
    if has_left and not has_local and not has_right:
        return "left"
    if not has_left and not has_local and has_right:
        return "right"
    if not has_left and has_local and not has_right:
        return "local"
    if has_left and has_local and has_right:
        return "left_local_right"
    if has_left and not has_local and has_right:
        return "left_right"
    return "empty"


def native_p_owner_records(p_entries, support_lookup, bond, nsites):
    """Return native P raw keys with left/right pair ownership labels."""

    owner_cache = {}

    def _owner(p, q):
        key = (int(p), int(q))
        if key not in owner_cache:
            owner_cache[key] = abelian_generator_owner_from_support(
                support_lookup(*key),
                bond,
                nsites,
            )
        return owner_cache[key]

    records = []
    for key in p_entries:
        p, q, r, s = (int(index) for index in key)
        records.append(
            (
                (p, q, r, s),
                _owner(p, q),
                _owner(r, s),
            )
        )
    return tuple(records)


def _coalesced_packed_identity_local_entries(component_entries):
    entries = tuple(component_entries or ())
    groups = OrderedDict()
    other = []
    packed = 0
    for entry in entries:
        if not isinstance(entry, AbelianPackedIdentityLocalEntry):
            other.append(entry)
            continue
        packed += 1
        E = getattr(entry, "E", None)
        F = getattr(entry, "F", None)
        key = (
            _packed_tensor_coalesce_key(E),
            _packed_tensor_coalesce_key(F),
            str(getattr(entry, "source", "")),
        )
        group = groups.get(key)
        if group is None:
            groups[key] = [
                E,
                F,
                str(getattr(entry, "source", "")),
                complex(entry.coeff),
                entry,
            ]
        else:
            group[3] += complex(entry.coeff)

    if not groups:
        return entries, 0, 0, 0

    coalesced = []
    cancelled = 0
    for E, F, source, coeff, original in groups.values():
        if abs(coeff) <= 1.0e-14:
            cancelled += 1
            continue
        if complex(getattr(original, "coeff", 0.0)) == coeff:
            coalesced.append(original)
        else:
            coalesced.append(
                AbelianPackedIdentityLocalEntry(
                    coeff,
                    E,
                    F,
                    source=source,
                )
            )
    return tuple(coalesced) + tuple(other), packed, len(coalesced), cancelled


def _coalesced_packed_local_generator_entries(component_entries):
    entries = tuple(component_entries or ())
    if not any(
        isinstance(entry, AbelianPackedLocalGeneratorEntry)
        for entry in entries
    ):
        return entries, 0, 0, 0
    groups = OrderedDict()
    other = []
    packed = 0
    for entry in entries:
        if not isinstance(entry, AbelianPackedLocalGeneratorEntry):
            other.append(entry)
            continue
        packed += 1
        E = getattr(entry, "E", None)
        W_left = getattr(entry, "W_left", None)
        W_right = getattr(entry, "W_right", None)
        F = getattr(entry, "F", None)
        key = (
            _packed_tensor_coalesce_key(E),
            _packed_tensor_coalesce_key(W_left),
            _packed_tensor_coalesce_key(W_right),
            _packed_tensor_coalesce_key(F),
            str(getattr(entry, "source", "")),
        )
        group = groups.get(key)
        if group is None:
            groups[key] = [
                E,
                W_left,
                W_right,
                F,
                str(getattr(entry, "source", "")),
                complex(entry.coeff),
                entry,
            ]
        else:
            group[5] += complex(entry.coeff)

    if not groups:
        return entries, 0, 0, 0

    coalesced = []
    cancelled = 0
    for E, W_left, W_right, F, source, coeff, original in groups.values():
        if abs(coeff) <= 1.0e-14:
            cancelled += 1
            continue
        if complex(getattr(original, "coeff", 0.0)) == coeff:
            coalesced.append(original)
        else:
            coalesced.append(
                AbelianPackedLocalGeneratorEntry(
                    coeff,
                    E,
                    W_left,
                    W_right,
                    F,
                    source=source,
                )
            )
    return tuple(coalesced) + tuple(other), packed, len(coalesced), cancelled


class AbelianContextualDirectFamilyBuilder:
    """Build contextual direct-family entries from sweep-provided callbacks."""

    def __init__(
        self,
        *,
        stats,
        record_phase,
        left_builder,
        right_builder,
        fallback_builder,
        entry_cls=AbelianPackedLocalGeneratorEntry,
        left_batch_builder=None,
        right_batch_builder=None,
        left_packed_boundary_table=None,
        right_packed_boundary_table=None,
        enable_packed_boundary_tables=True,
        boundary_batch_cache=None,
        planned_entries_cache=None,
        left_boundary_cache_token=None,
        right_boundary_cache_token=None,
        boundary_batch_owner=None,
    ):
        self.stats = stats
        self.record_phase = record_phase
        self.left_builder = left_builder
        self.right_builder = right_builder
        self.fallback_builder = fallback_builder
        self.entry_cls = entry_cls
        self.left_batch_builder = left_batch_builder
        self.right_batch_builder = right_batch_builder
        self.left_cache = {}
        self.right_cache = {}
        self.boundary_batch_cache = (
            boundary_batch_cache if boundary_batch_cache is not None else {}
        )
        self.planned_entries_cache = (
            planned_entries_cache if planned_entries_cache is not None else {}
        )
        self.left_boundary_cache_token = left_boundary_cache_token
        self.right_boundary_cache_token = right_boundary_cache_token
        self.boundary_batch_owner = boundary_batch_owner
        self.left_packed_boundary_table = None
        self.right_packed_boundary_table = None
        if bool(enable_packed_boundary_tables):
            self.left_packed_boundary_table = (
                left_packed_boundary_table
                if left_packed_boundary_table is not None
                else AbelianPackedContextualBoundaryTable(side="left")
            )
            self.right_packed_boundary_table = (
                right_packed_boundary_table
                if right_packed_boundary_table is not None
                else AbelianPackedContextualBoundaryTable(side="right")
            )

    @staticmethod
    def _normalized_boundary_keys(keys, family_name=None):
        raw_keys = tuple(keys or ())
        if all(
            AbelianPackedContextualBoundaryTable.is_normalized_key(key)
            for key in raw_keys
        ):
            normalized = raw_keys
        else:
            normalized = tuple(
                AbelianPackedContextualBoundaryTable.normalize_key(key)
                for key in raw_keys
            )
        if family_name is None:
            return normalized
        family = str(family_name)
        return tuple(
            key
            if len(tuple(key)) == 3
            else (family, tuple(key[0]), str(key[1]))
            for key in normalized
        )

    @staticmethod
    def _batch_build_missing(
        keys,
        cache,
        builder,
        batch_builder,
        family_name,
        packed_table=None,
        table_ids_only=False,
        debug_stats=None,
        debug_side=None,
    ):
        """Resolve boundary keys through cache, then an optional batch callback."""

        raw_keys = AbelianContextualDirectFamilyBuilder._normalized_boundary_keys(keys)
        storage_keys = AbelianContextualDirectFamilyBuilder._normalized_boundary_keys(
            keys,
            family_name=family_name,
        )
        debug_prefix = f"{debug_side}_" if debug_side else ""
        owner = getattr(
            AbelianContextualDirectFamilyBuilder,
            "_boundary_batch_owner_current",
            None,
        )
        if owner is not None and hasattr(owner, "resolve_contextual_boundary_batch"):
            native_plan_key = str(
                getattr(
                    batch_builder,
                    "_pyqed_cpp_contextual_batch_plan_key",
                    "",
                )
                or ""
            )
            owner_table_id_resolver = getattr(
                owner,
                "resolve_contextual_boundary_table_ids_from_plan",
                None,
            )
            if (
                bool(table_ids_only)
                and packed_table is not None
                and native_plan_key
                and owner_table_id_resolver is not None
            ):
                try:
                    result = owner_table_id_resolver(
                        native_plan_key,
                        tuple(raw_keys),
                        packed_table,
                        family_name,
                        debug_stats if debug_stats is not None else None,
                    )
                    table_ids = tuple(result[1])
                    complete = bool(
                        len(table_ids) == len(raw_keys)
                        and all(int(table_id) >= 0 for table_id in table_ids)
                    )
                    if complete:
                        values = list(result[0])
                        if not values:
                            values = [True]
                        if debug_stats is not None:
                            _increment_counter(
                                debug_stats,
                                f"{debug_prefix}owner_resolve_calls",
                            )
                            _increment_counter(
                                debug_stats,
                                f"{debug_prefix}owner_native_plan_table_id_calls",
                            )
                            _increment_counter(
                                debug_stats,
                                f"{debug_prefix}owner_native_plan_table_id_successes",
                            )
                        return (
                            values,
                            table_ids,
                            int(result[2]),
                            int(result[3]),
                            float(result[4]),
                            bool(result[5]),
                        )
                    if debug_stats is not None:
                        _increment_counter(
                            debug_stats,
                            f"{debug_prefix}owner_native_plan_table_id_incomplete",
                        )
                except Exception as exc:
                    if debug_stats is not None:
                        _increment_counter(
                            debug_stats,
                            f"{debug_prefix}owner_native_plan_table_id_failures",
                        )
                        debug_stats[
                            f"{debug_prefix}owner_native_plan_table_id_last_error"
                        ] = repr(exc)
            try:
                result = owner.resolve_contextual_boundary_batch(
                    tuple(raw_keys),
                    cache,
                    builder,
                    batch_builder if callable(batch_builder) else None,
                    family_name,
                    packed_table,
                    bool(table_ids_only),
                    debug_stats if debug_stats is not None else None,
                    str(debug_side or ""),
                )
                owner_stats = debug_stats
                if owner_stats is not None:
                    _increment_counter(
                        owner_stats,
                        f"{debug_prefix}owner_resolve_calls",
                    )
                return (
                    list(result[0]),
                    tuple(result[1]),
                    int(result[2]),
                    int(result[3]),
                    float(result[4]),
                    bool(result[5]),
                )
            except Exception as exc:
                if debug_stats is not None:
                    _increment_counter(
                        debug_stats,
                        f"{debug_prefix}owner_resolve_failures",
                    )
                    debug_stats[f"{debug_prefix}owner_resolve_last_error"] = repr(exc)

        def _debug_increment(name, amount=1):
            if debug_stats is not None:
                _increment_counter(debug_stats, debug_prefix + str(name), amount)

        def _debug_set(name, value):
            if debug_stats is not None:
                debug_stats[debug_prefix + str(name)] = value

        values = [None] * len(raw_keys)
        table_ids = [-1] * len(raw_keys)
        missing = []
        missing_storage_keys = []
        missing_positions = []
        hits = 0
        misses = 0
        build_seconds = 0.0
        batch_used = False
        if packed_table is not None:
            if bool(table_ids_only) and hasattr(
                packed_table,
                "resolve_current_ids_many",
            ):
                (
                    resolved_table_ids,
                    table_missing,
                    table_missing_positions,
                    table_hits,
                    _table_misses,
                ) = packed_table.resolve_current_ids_many(
                    storage_keys,
                    normalized=True,
                )
            else:
                (
                    table_values,
                    resolved_table_ids,
                    table_missing,
                    table_missing_positions,
                    table_hits,
                    _table_misses,
                ) = packed_table.resolve_many(
                    storage_keys,
                    normalized=True,
                    return_ids=True,
                )
                if not bool(table_ids_only):
                    values[:] = table_values
            table_ids[:] = list(resolved_table_ids)
            hits += int(table_hits)
            unresolved = tuple(
                (
                    int(pos),
                    raw_keys[int(pos)],
                    storage_keys[int(pos)],
                )
                for pos in table_missing_positions
            )
        else:
            unresolved = tuple(
                (idx, raw_key, storage_key)
                for idx, (raw_key, storage_key) in enumerate(
                    zip(raw_keys, storage_keys)
                )
            )
        cache_missing = object()
        cache_table_keys = []
        cache_table_values = []
        cache_table_positions = []
        for idx, raw_key, storage_key in unresolved:
            cached_value = cache.get(
                raw_key,
                cache.get(storage_key, cache_missing),
            )
            if cached_value is not cache_missing:
                values[idx] = cached_value
                hits += 1
                if (
                    packed_table is not None
                    and int(table_ids[idx]) < 0
                    and _contextual_boundary_payload_kind(cached_value) == "packed"
                ):
                    cache_table_keys.append(storage_key)
                    cache_table_values.append(cached_value)
                    cache_table_positions.append(idx)
            else:
                missing.append(raw_key)
                missing_storage_keys.append(storage_key)
                missing_positions.append(idx)
        if cache_table_keys:
            packed_table.put_many(
                tuple(cache_table_keys),
                tuple(cache_table_values),
                family_name=family_name,
                normalized=True,
            )
            table_ids_by_key = packed_table.ids
            for pos, storage_key in zip(cache_table_positions, cache_table_keys):
                table_ids[int(pos)] = int(table_ids_by_key.get(storage_key, -1))
        if missing and callable(batch_builder):
            _debug_increment("batch_attempts")
            _debug_increment("batch_requested_keys", len(missing))
            _debug_set("last_batch_missing", int(len(missing)))
            _debug_set("last_batch_family", str(family_name))
            _debug_set("last_batch_table_ids_only", bool(table_ids_only))
            t_build = time.perf_counter()
            batch_status = "returned"
            built_len = -1
            try:
                built = batch_builder(tuple(missing), family_name=family_name)
            except Exception as exc:
                built = None
                batch_status = "error"
                _debug_increment("batch_errors")
                _debug_set("last_batch_error", repr(exc))
            build_seconds += time.perf_counter() - t_build
            if built is None:
                if batch_status != "error":
                    _debug_increment("batch_none")
                    batch_status = "none"
            else:
                try:
                    built_len = len(built)
                except Exception as exc:
                    built_len = -1
                    batch_status = "len_error"
                    _debug_increment("batch_len_errors")
                    _debug_set("last_batch_len_error", repr(exc))
                _debug_set("last_batch_returned", int(built_len))
                if built_len != len(missing):
                    _debug_increment("batch_len_mismatch")
                    _debug_set("last_batch_expected", int(len(missing)))
                    if batch_status == "returned":
                        batch_status = "len_mismatch"
                elif batch_status == "returned":
                    missing_results = sum(result is None for result in built)
                    _debug_increment("batch_complete")
                    _debug_increment("batch_returned_keys", built_len)
                    _debug_set("last_batch_missing_results", int(missing_results))
            _debug_set("last_batch_status", batch_status)
            if built is not None and built_len == len(missing):
                batch_used = True
                if packed_table is not None:
                    if bool(table_ids_only) and hasattr(
                        packed_table,
                        "put_many_packed",
                    ):
                        packed_table.put_many_packed(
                            missing_storage_keys,
                            built,
                            family_name=family_name,
                            normalized=True,
                        )
                    else:
                        packed_table.put_many(
                            missing_storage_keys,
                            built,
                            family_name=family_name,
                            normalized=True,
                        )
                for pos, storage_key, result in zip(
                    missing_positions,
                    missing_storage_keys,
                    built,
                ):
                    if packed_table is not None:
                        table_ids[pos] = int(packed_table.ids.get(storage_key, -1))
                    if not bool(table_ids_only):
                        values[pos] = result
                    if bool(table_ids_only):
                        if int(table_ids[pos]) < 0:
                            cache[storage_key] = result
                    elif (
                        packed_table is None
                        or _contextual_boundary_payload_kind(result) != "packed"
                    ):
                        cache[storage_key] = result
                    misses += 1
                missing = []
                missing_positions = []
        for pos, raw_key, storage_key in zip(
            missing_positions,
            missing,
            missing_storage_keys,
        ):
            pattern, piece = raw_key
            t_build = time.perf_counter()
            try:
                result = builder(pattern, piece, family_name=family_name)
            except Exception:
                result = None
            build_seconds += time.perf_counter() - t_build
            if (
                packed_table is None
                or _contextual_boundary_payload_kind(result) != "packed"
            ):
                cache[storage_key] = result
            if packed_table is not None:
                packed_table.put_many(
                    (storage_key,),
                    (result,),
                    family_name=family_name,
                    normalized=True,
                )
                table_ids[pos] = int(packed_table.ids.get(storage_key, -1))
            if not bool(table_ids_only):
                values[pos] = result
            misses += 1
        if bool(table_ids_only) and table_ids and all(
            int(table_id) >= 0 for table_id in table_ids
        ):
            values = [True]
        return values, tuple(table_ids), hits, misses, build_seconds, batch_used

    def _batch_build_missing_cached(
        self,
        side,
        keys,
        cache,
        builder,
        batch_builder,
        family_name,
        packed_table=None,
        layout_token=None,
        table_ids_only=False,
    ):
        raw_keys = tuple(keys or ())
        token = (
            self.left_boundary_cache_token
            if str(side) == "left"
            else self.right_boundary_cache_token
        )
        keys = None
        if layout_token is None:
            keys = self._normalized_boundary_keys(raw_keys)
            layout_marker = keys
        else:
            layout_marker = layout_token
        cache_key = (
            "contextual_boundary_batch",
            str(side),
            token,
            str(family_name),
            id(packed_table) if packed_table is not None else None,
            bool(table_ids_only),
            layout_marker,
        )
        cache_stats = self.stats.setdefault(
            "contextual_boundary_batch_cache",
            {"hits": 0, "misses": 0, "stores": 0},
        )
        cached = self.boundary_batch_cache.get(cache_key)
        if cached is not None:
            cache_stats["hits"] = int(cache_stats.get("hits", 0)) + 1
            cache_stats["last_side"] = str(side)
            cached_key_count = len(cached[1] if bool(table_ids_only) else cached[0])
            cache_stats["last_keys"] = int(cached_key_count)
            cache_stats["last_layout_token"] = bool(layout_token is not None)
            cache_stats["cache_size"] = int(len(self.boundary_batch_cache))
            values, table_ids = cached
            hit_count = int(len(table_ids) if bool(table_ids_only) else len(values))
            return (
                list(values),
                tuple(table_ids),
                hit_count,
                0,
                0.0,
                False,
            )
        cache_stats["misses"] = int(cache_stats.get("misses", 0)) + 1
        if keys is None:
            keys = self._normalized_boundary_keys(raw_keys)
        (
            values,
            table_ids,
            hits,
            misses,
            build_seconds,
            batch_used,
        ) = (None, None, None, None, None, None)
        old_owner = getattr(
            AbelianContextualDirectFamilyBuilder,
            "_boundary_batch_owner_current",
            None,
        )
        AbelianContextualDirectFamilyBuilder._boundary_batch_owner_current = (
            self.boundary_batch_owner
        )
        try:
            (
                values,
                table_ids,
                hits,
                misses,
                build_seconds,
                batch_used,
            ) = self._batch_build_missing(
                keys,
                cache,
                builder,
                batch_builder,
                family_name,
                packed_table,
                table_ids_only=table_ids_only,
                debug_stats=self.stats.setdefault(
                    "contextual_route_lazy_pack",
                    {"calls": 0},
                ),
                debug_side=str(side),
            )
        finally:
            AbelianContextualDirectFamilyBuilder._boundary_batch_owner_current = old_owner
        if values and all(value is not None for value in values):
            self.boundary_batch_cache[cache_key] = (tuple(values), tuple(table_ids))
            cache_stats["stores"] = int(cache_stats.get("stores", 0)) + 1
        elif bool(table_ids_only) and table_ids and all(
            int(table_id) >= 0 for table_id in table_ids
        ):
            self.boundary_batch_cache[cache_key] = (tuple(values), tuple(table_ids))
            cache_stats["stores"] = int(cache_stats.get("stores", 0)) + 1
        cache_stats["last_side"] = str(side)
        cache_stats["last_keys"] = int(len(keys))
        cache_stats["last_layout_token"] = bool(layout_token is not None)
        cache_stats["last_table_ids_only"] = bool(table_ids_only)
        cache_stats["cache_size"] = int(len(self.boundary_batch_cache))
        return values, table_ids, hits, misses, build_seconds, batch_used

    def precompute_boundaries(self, family_name, records):
        if isinstance(records, AbelianDirectRoutePlan):
            route_plan = records
            record_count = route_plan.record_count
            left_keys = route_plan.left_keys
            right_keys = route_plan.right_keys
            left_results = {}
            right_results = {}
            left_values = [None] * len(left_keys)
            right_values = [None] * len(right_keys)
        else:
            route_plan = None
            record_count = len(records or ())
            left_keys = right_keys = None
            left_results = {}
            right_results = {}
            left_values = []
            right_values = []
        if not record_count:
            return AbelianContextualBoundaryBatch(left_results, right_results)

        t_precompute = time.perf_counter()
        if route_plan is None:
            left_keys, right_keys = contextual_boundary_keys(records)
            route_cache_token = None
        else:
            route_cache_token = ("route_plan_object", id(route_plan))
        left_build_keys = tuple(left_keys)
        right_build_keys = tuple(right_keys)
        left_cache = self.left_cache
        right_cache = self.right_cache
        left_hits = left_misses = 0
        right_hits = right_misses = 0
        left_build_seconds = 0.0
        right_build_seconds = 0.0
        left_batch_used = False
        right_batch_used = False
        owner_precompute_used = False
        owner = self.boundary_batch_owner
        owner_precompute = (
            getattr(owner, "precompute_contextual_boundaries", None)
            if owner is not None
            else None
        )
        owner_precompute_from_builders = (
            getattr(owner, "precompute_contextual_boundaries_from_builders", None)
            if owner is not None
            else None
        )
        owner_install_batch_builder = (
            getattr(owner, "install_contextual_boundary_batch_builder", None)
            if owner is not None
            else None
        )
        owner_resolve_from_builder = (
            getattr(owner, "resolve_contextual_boundary_batch_from_builder", None)
            if owner is not None
            else None
        )
        owner_resolve_table_ids_from_plan = (
            getattr(owner, "resolve_contextual_boundary_table_ids_from_plan", None)
            if owner is not None
            else None
        )
        owner_stats = self.stats.setdefault(
            "contextual_boundary_precompute_owner",
            {"calls": 0},
        )
        owner_table_ids_only = bool(
            route_plan is not None
            and self.left_packed_boundary_table is not None
            and self.right_packed_boundary_table is not None
        )
        precompute_cache_key = None
        if route_plan is not None:
            precompute_cache_key = (
                "contextual_boundary_precompute_result",
                str(family_name),
                self.left_boundary_cache_token,
                self.right_boundary_cache_token,
                route_cache_token,
                id(self.left_packed_boundary_table),
                id(self.right_packed_boundary_table),
                bool(owner_table_ids_only),
            )
            precompute_cache_stats = self.stats.setdefault(
                "contextual_boundary_precompute_cache",
                {"hits": 0, "misses": 0, "stores": 0},
            )
            cached_batch = self.boundary_batch_cache.get(precompute_cache_key)
            if cached_batch is not None:
                _increment_counter(precompute_cache_stats, "hits")
                boundary_cache_stats = self.stats.setdefault(
                    "contextual_boundary_batch_cache",
                    {"hits": 0, "misses": 0, "stores": 0},
                )
                _increment_counter(boundary_cache_stats, "hits", 2)
                left_unique = len(left_keys)
                right_unique = len(right_keys)
                phase_stats = self.stats.setdefault(
                    "contextual_boundary_precompute",
                    {"calls": 0},
                )
                _increment_counter(phase_stats, "calls")
                _increment_counter(phase_stats, "cache_hits")
                _increment_counter(phase_stats, "records", record_count)
                _increment_counter(phase_stats, "left_unique", left_unique)
                _increment_counter(phase_stats, "right_unique", right_unique)
                _increment_counter(phase_stats, "left_hits", left_unique)
                _increment_counter(phase_stats, "right_hits", right_unique)
                phase_stats["last_left_unique"] = int(left_unique)
                phase_stats["last_right_unique"] = int(right_unique)
                phase_stats["last_owner_used"] = False
                self.record_phase(
                    "contextual_boundary_precompute",
                    time.perf_counter() - t_precompute,
                    records=record_count,
                    left_unique=left_unique,
                    right_unique=right_unique,
                    left_hits=left_unique,
                    left_misses=0,
                    left_build_seconds=0.0,
                    right_hits=right_unique,
                    right_misses=0,
                    right_build_seconds=0.0,
                    left_batch=0,
                    right_batch=0,
                    left_packed=cached_batch.left_payload_counts.get("packed", 0),
                    right_packed=cached_batch.right_payload_counts.get("packed", 0),
                    left_legacy=cached_batch.left_payload_counts.get("legacy", 0),
                    right_legacy=cached_batch.right_payload_counts.get("legacy", 0),
                    left_table_ids=sum(
                        1
                        for table_id in cached_batch.left_table_ids
                        if int(table_id) >= 0
                    ),
                    right_table_ids=sum(
                        1
                        for table_id in cached_batch.right_table_ids
                        if int(table_id) >= 0
                    ),
                    owner=0,
                    cache_hit=1,
                )
                return cached_batch
            _increment_counter(precompute_cache_stats, "misses")
        if owner_precompute is not None:
            _increment_counter(owner_stats, "calls")
            try:
                debug_stats = self.stats.setdefault(
                    "contextual_route_lazy_pack",
                    {"calls": 0},
                )
                if (
                    owner_precompute_from_builders is not None
                    and owner_install_batch_builder is not None
                ):
                    owner_install_batch_builder_auto = getattr(
                        owner,
                        "install_contextual_boundary_batch_builder_auto",
                        None,
                    )

                    def _native_batch_plan_key(batch_builder):
                        key = getattr(
                            batch_builder,
                            "_pyqed_cpp_contextual_batch_plan_key",
                            "",
                        )
                        return str(key) if key else ""

                    legacy_token = None

                    def _legacy_owner_key(side):
                        nonlocal legacy_token
                        if legacy_token is None:
                            legacy_token = (
                                "contextual_boundary_batch_builder",
                                id(self),
                                str(family_name),
                                getattr(route_plan, "signature", None)
                                if route_plan is not None
                                else None,
                                id(self.left_packed_boundary_table),
                                id(self.right_packed_boundary_table),
                            )
                        return repr(legacy_token + (str(side),))

                    def _install_owner_batch_builder(
                        cache,
                        builder,
                        batch_builder,
                        packed_table,
                        side,
                    ):
                        native_key = _native_batch_plan_key(batch_builder)
                        if owner_install_batch_builder_auto is not None:
                            try:
                                return owner_install_batch_builder_auto(
                                    cache,
                                    builder,
                                    batch_builder,
                                    family_name,
                                    packed_table,
                                    side,
                                    native_key,
                                )
                                return _legacy_owner_key(side)
                            except TypeError:
                                pass
                        owner_key = _legacy_owner_key(side)
                        if native_key:
                            try:
                                owner_install_batch_builder(
                                    owner_key,
                                    cache,
                                    builder,
                                    batch_builder,
                                    family_name,
                                    packed_table,
                                    side,
                                    native_key,
                                )
                                return owner_key
                            except TypeError:
                                pass
                        owner_install_batch_builder(
                            owner_key,
                            cache,
                            builder,
                            batch_builder,
                            family_name,
                            packed_table,
                            side,
                        )
                        return owner_key

                    left_owner_key = _install_owner_batch_builder(
                        left_cache,
                        self.left_builder,
                        self.left_batch_builder if route_plan is not None else None,
                        self.left_packed_boundary_table,
                        "left",
                    )
                    right_owner_key = _install_owner_batch_builder(
                        right_cache,
                        self.right_builder,
                        self.right_batch_builder if route_plan is not None else None,
                        self.right_packed_boundary_table,
                        "right",
                    )
                    _increment_counter(owner_stats, "keyed_calls")
                    owner_stats["last_keyed"] = True
                    use_side_resolve = bool(
                        owner_resolve_from_builder is not None
                        and self.stats.get(
                            "contextual_boundary_precompute_side_cache_enabled",
                            False,
                        )
                    )
                    if use_side_resolve:
                        max_content_keys = int(
                            self.stats.get(
                                "contextual_boundary_side_cache_max_content_keys",
                                getattr(
                                    self,
                                    "contextual_boundary_side_cache_max_content_keys",
                                    512,
                                ),
                            )
                        )

                        def _side_key_marker(keys):
                            keys = tuple(keys or ())
                            if len(keys) <= max_content_keys:
                                return keys
                            if route_plan is not None:
                                return (
                                    "route_plan_side",
                                    route_cache_token,
                                    int(len(keys)),
                                )
                            return (int(len(keys)), id(keys))

                        def _resolve_side(
                            side,
                            owner_key,
                            keys,
                            packed_table,
                            native_plan_key,
                        ):
                            keys = tuple(keys or ())
                            side_text = str(side)
                            side_token = (
                                self.left_boundary_cache_token
                                if side_text == "left"
                                else self.right_boundary_cache_token
                            )
                            side_cache_key = (
                                "contextual_boundary_side_precompute_result",
                                side_text,
                                side_token,
                                str(family_name),
                                id(packed_table),
                                bool(owner_table_ids_only),
                                _side_key_marker(keys),
                            )
                            side_cache_stats = self.stats.setdefault(
                                "contextual_boundary_side_precompute_cache",
                                {"hits": 0, "misses": 0, "stores": 0},
                            )
                            cached = self.boundary_batch_cache.get(side_cache_key)
                            if cached is not None:
                                _increment_counter(side_cache_stats, "hits")
                                values, table_ids = cached
                                return (
                                    tuple(values),
                                    tuple(table_ids),
                                    int(len(table_ids) or len(values)),
                                    0,
                                    0.0,
                                    False,
                                )
                            _increment_counter(side_cache_stats, "misses")
                            if bool(owner_table_ids_only):
                                _increment_counter(
                                    side_cache_stats,
                                    "native_plan_attempts",
                                )
                                if owner_resolve_table_ids_from_plan is None:
                                    _increment_counter(
                                        side_cache_stats,
                                        "native_plan_no_method",
                                    )
                                if not native_plan_key:
                                    _increment_counter(
                                        side_cache_stats,
                                        "native_plan_no_key",
                                    )
                            def _resolve_from_builder(resolve_keys, use_native_plan=True):
                                resolve_keys = tuple(resolve_keys or ())
                                try:
                                    return owner_resolve_from_builder(
                                        owner_key,
                                        resolve_keys,
                                        bool(owner_table_ids_only),
                                        debug_stats,
                                        bool(use_native_plan),
                                    )
                                except TypeError:
                                    if bool(use_native_plan):
                                        return owner_resolve_from_builder(
                                            owner_key,
                                            resolve_keys,
                                            bool(owner_table_ids_only),
                                            debug_stats,
                                        )
                                    raise

                            def _payload_pair_difference(candidate, reference):
                                if candidate is None or reference is None:
                                    return candidate is reference, float("inf"), float("inf")
                                try:
                                    candidate_pair = tuple(candidate)
                                    reference_pair = tuple(reference)
                                except Exception:
                                    return False, float("inf"), float("inf")
                                if len(candidate_pair) != len(reference_pair):
                                    return False, float("inf"), float("inf")
                                max_abs = 0.0
                                max_rel = 0.0
                                for lhs, rhs in zip(candidate_pair, reference_pair):
                                    same, diff, ref_norm = compare_abelian_packed_boundary_tensors(
                                        lhs,
                                        rhs,
                                    )
                                    if not same:
                                        return False, float("inf"), float("inf")
                                    diff = float(diff)
                                    ref_norm = float(ref_norm)
                                    max_abs = max(max_abs, diff)
                                    max_rel = max(max_rel, diff / max(ref_norm, 1.0e-30))
                                return True, max_abs, max_rel

                            def _discard_boundary_keys(discard_keys):
                                if packed_table is None:
                                    return 0
                                discard = getattr(packed_table, "discard", None)
                                if not callable(discard):
                                    return 0
                                storage_keys = self._normalized_boundary_keys(
                                    discard_keys,
                                    family_name=family_name,
                                )
                                count = 0
                                for storage_key in storage_keys:
                                    try:
                                        if discard(storage_key, normalized=True):
                                            count += 1
                                    except Exception:
                                        pass
                                return int(count)

                            def _debug_value(name, default):
                                getter = getattr(debug_stats, "get", None)
                                if not callable(getter):
                                    return default
                                return getter(name, default)

                            def _validate_native_table_ids(native_table_ids):
                                if not bool(_debug_value("validate_native_plan_table_ids", False)):
                                    return True
                                if packed_table is None or not hasattr(packed_table, "values_for_ids"):
                                    _increment_counter(
                                        side_cache_stats,
                                        "native_plan_validation_unavailable",
                                    )
                                    return True
                                limit = int(_debug_value("native_plan_validation_limit", -1))
                                if limit == 0:
                                    return True
                                tol = float(
                                    _debug_value("native_plan_validation_tol", 1.0e-10)
                                    or 0.0
                                )
                                fail_fast = bool(
                                    _debug_value("native_plan_validation_fail_fast", False)
                                )
                                sample_size = len(keys) if limit < 0 else min(int(limit), len(keys))
                                if sample_size <= 0:
                                    return True
                                sample_keys = tuple(keys[:sample_size])
                                sample_ids = tuple(native_table_ids[:sample_size])
                                native_values = tuple(packed_table.values_for_ids(sample_ids))
                                _increment_counter(
                                    side_cache_stats,
                                    "native_plan_validation_calls",
                                )
                                _increment_counter(
                                    side_cache_stats,
                                    "native_plan_validation_keys",
                                    sample_size,
                                )
                                side_cache_stats["last_native_plan_validation_keys"] = int(sample_size)
                                evicted = _discard_boundary_keys(sample_keys)
                                _increment_counter(
                                    side_cache_stats,
                                    "native_plan_validation_evictions",
                                    evicted,
                                )
                                if bool(_debug_value("native_plan_validation_cold", False)):
                                    batch_builder = (
                                        self.left_batch_builder
                                        if side_text == "left"
                                        else self.right_batch_builder
                                    )
                                    clear_cache = getattr(
                                        batch_builder,
                                        "_pyqed_clear_contextual_boundary_cache",
                                        None,
                                    )
                                    if callable(clear_cache):
                                        clear_cache()
                                        _increment_counter(
                                            side_cache_stats,
                                            "native_plan_validation_cold_clears",
                                        )
                                try:
                                    reference = _resolve_from_builder(
                                        sample_keys,
                                        use_native_plan=False,
                                    )
                                except Exception as exc:
                                    _increment_counter(
                                        side_cache_stats,
                                        "native_plan_validation_failures",
                                    )
                                    side_cache_stats["native_plan_validation_last_error"] = repr(exc)
                                    if fail_fast:
                                        raise
                                    return True
                                reference_ids = tuple(reference[1])
                                if len(reference_ids) != sample_size or any(
                                    int(table_id) < 0 for table_id in reference_ids
                                ):
                                    _increment_counter(
                                        side_cache_stats,
                                        "native_plan_validation_failures",
                                    )
                                    side_cache_stats[
                                        "native_plan_validation_last_error"
                                    ] = "reference_table_ids_incomplete"
                                    if fail_fast:
                                        raise RuntimeError(
                                            "contextual native plan validation could not "
                                            "build reference ids"
                                        )
                                    return True
                                reference_values = tuple(packed_table.values_for_ids(reference_ids))
                                max_abs = 0.0
                                max_rel = 0.0
                                for idx, (got, ref) in enumerate(zip(native_values, reference_values)):
                                    same, abs_diff, rel_diff = _payload_pair_difference(got, ref)
                                    max_abs = max(max_abs, float(abs_diff))
                                    max_rel = max(max_rel, float(rel_diff))
                                    if (
                                        not same
                                        or (
                                            float(abs_diff) > tol
                                            and float(rel_diff) > tol
                                        )
                                    ):
                                        _increment_counter(
                                            side_cache_stats,
                                            "native_plan_validation_mismatches",
                                        )
                                        side_cache_stats[
                                            "native_plan_validation_last_side"
                                        ] = side_text
                                        side_cache_stats[
                                            "native_plan_validation_last_index"
                                        ] = int(idx)
                                        try:
                                            pattern, piece = sample_keys[idx]
                                            side_cache_stats[
                                                "native_plan_validation_last_pattern"
                                            ] = tuple(str(item) for item in pattern)
                                            side_cache_stats[
                                                "native_plan_validation_last_piece"
                                            ] = str(piece)
                                        except Exception:
                                            side_cache_stats[
                                                "native_plan_validation_last_key"
                                            ] = repr(sample_keys[idx])
                                        try:
                                            got_pair = tuple(got)
                                            ref_pair = tuple(ref)
                                            side_cache_stats[
                                                "native_plan_validation_last_native_key_counts"
                                            ] = tuple(
                                                len(abelian_packed_tensor_items(item)[0])
                                                for item in got_pair
                                            )
                                            side_cache_stats[
                                                "native_plan_validation_last_ref_key_counts"
                                            ] = tuple(
                                                len(abelian_packed_tensor_items(item)[0])
                                                for item in ref_pair
                                            )
                                            side_cache_stats[
                                                "native_plan_validation_last_native_qns"
                                            ] = tuple(
                                                tuple(tuple(axis) for axis in getattr(item, "qns", ()))
                                                for item in got_pair
                                            )
                                            side_cache_stats[
                                                "native_plan_validation_last_ref_qns"
                                            ] = tuple(
                                                tuple(tuple(axis) for axis in getattr(item, "qns", ()))
                                                for item in ref_pair
                                            )
                                            side_cache_stats[
                                                "native_plan_validation_last_native_sources"
                                            ] = tuple(
                                                str(getattr(item, "source", ""))
                                                for item in got_pair
                                            )
                                            side_cache_stats[
                                                "native_plan_validation_last_ref_sources"
                                            ] = tuple(
                                                str(getattr(item, "source", ""))
                                                for item in ref_pair
                                            )
                                        except Exception:
                                            pass
                                        side_cache_stats[
                                            "native_plan_validation_max_abs"
                                        ] = float(max_abs)
                                        side_cache_stats[
                                            "native_plan_validation_max_rel"
                                        ] = float(max_rel)
                                        _discard_boundary_keys(keys)
                                        if fail_fast:
                                            raise RuntimeError(
                                                "contextual native plan validation failed "
                                                f"side={side_text} abs={abs_diff:.3e} "
                                                f"rel={rel_diff:.3e}"
                                            )
                                        return False
                                _increment_counter(
                                    side_cache_stats,
                                    "native_plan_validation_matches",
                                    sample_size,
                                )
                                side_cache_stats["native_plan_validation_max_abs"] = max(
                                    float(
                                        side_cache_stats.get(
                                            "native_plan_validation_max_abs",
                                            0.0,
                                        )
                                    ),
                                    float(max_abs),
                                )
                                side_cache_stats["native_plan_validation_max_rel"] = max(
                                    float(
                                        side_cache_stats.get(
                                            "native_plan_validation_max_rel",
                                            0.0,
                                        )
                                    ),
                                    float(max_rel),
                                )
                                return True

                            used_native_plan = False
                            if (
                                bool(owner_table_ids_only)
                                and owner_resolve_table_ids_from_plan is not None
                                and native_plan_key
                            ):
                                _increment_counter(
                                    side_cache_stats,
                                    "native_plan_calls",
                                )
                                used_native_plan = True
                                resolved = owner_resolve_table_ids_from_plan(
                                    native_plan_key,
                                    keys,
                                    packed_table,
                                    family_name,
                                    debug_stats,
                                )
                            else:
                                resolved = _resolve_from_builder(keys)
                            if used_native_plan:
                                native_table_ids = tuple(resolved[1])
                                native_complete = bool(
                                    len(native_table_ids) == len(keys)
                                    and all(int(table_id) >= 0 for table_id in native_table_ids)
                                )
                                if not native_complete:
                                    _increment_counter(
                                        side_cache_stats,
                                        "native_plan_fallbacks",
                                    )
                                    resolved = _resolve_from_builder(
                                        keys,
                                        use_native_plan=False,
                                    )
                                elif not _validate_native_table_ids(native_table_ids):
                                    _increment_counter(
                                        side_cache_stats,
                                        "native_plan_validation_fallbacks",
                                    )
                                    _increment_counter(
                                        side_cache_stats,
                                        "native_plan_fallbacks",
                                    )
                                    resolved = _resolve_from_builder(
                                        keys,
                                        use_native_plan=False,
                                    )
                            values = tuple(resolved[0])
                            table_ids = tuple(resolved[1])
                            hits = int(resolved[2])
                            misses = int(resolved[3])
                            build_seconds = float(resolved[4])
                            batch_used = bool(resolved[5])
                            if bool(owner_table_ids_only):
                                table_id_array = np.asarray(
                                    table_ids,
                                    dtype=np.int64,
                                )
                                complete = bool(
                                    table_id_array.size == len(keys)
                                    and bool(np.all(table_id_array >= 0))
                                )
                            else:
                                complete = bool(
                                    values
                                    and all(value is not None for value in values)
                                )
                            if complete:
                                self.boundary_batch_cache[side_cache_key] = (
                                    values,
                                    table_ids,
                                )
                                _increment_counter(side_cache_stats, "stores")
                            return (
                                values,
                                table_ids,
                                hits,
                                misses,
                                build_seconds,
                                batch_used,
                            )

                        (
                            left_values,
                            left_table_ids,
                            left_hits,
                            left_misses,
                            left_build_seconds,
                            left_batch_used,
                        ) = _resolve_side(
                            "left",
                            left_owner_key,
                            left_build_keys,
                            self.left_packed_boundary_table,
                            _native_batch_plan_key(
                                self.left_batch_builder
                                if route_plan is not None
                                else None
                            ),
                        )
                        (
                            right_values,
                            right_table_ids,
                            right_hits,
                            right_misses,
                            right_build_seconds,
                            right_batch_used,
                        ) = _resolve_side(
                            "right",
                            right_owner_key,
                            right_build_keys,
                            self.right_packed_boundary_table,
                            _native_batch_plan_key(
                                self.right_batch_builder
                                if route_plan is not None
                                else None
                            ),
                        )
                    else:
                        (
                            left_values,
                            right_values,
                            left_table_ids,
                            right_table_ids,
                            left_hits,
                            left_misses,
                            right_hits,
                            right_misses,
                            left_build_seconds,
                            right_build_seconds,
                            left_batch_used,
                            right_batch_used,
                        ) = owner_precompute_from_builders(
                            left_owner_key,
                            right_owner_key,
                            tuple(left_build_keys),
                            tuple(right_build_keys),
                            debug_stats,
                            table_ids_only=owner_table_ids_only,
                        )
                else:
                    owner_stats["last_keyed"] = False
                    (
                        left_values,
                        right_values,
                        left_table_ids,
                        right_table_ids,
                        left_hits,
                        left_misses,
                        right_hits,
                        right_misses,
                        left_build_seconds,
                        right_build_seconds,
                        left_batch_used,
                        right_batch_used,
                    ) = owner_precompute(
                        tuple(left_build_keys),
                        tuple(right_build_keys),
                        left_cache,
                        right_cache,
                        self.left_builder,
                        self.right_builder,
                        self.left_batch_builder if route_plan is not None else None,
                        self.right_batch_builder if route_plan is not None else None,
                        family_name,
                        self.left_packed_boundary_table,
                        self.right_packed_boundary_table,
                        debug_stats,
                        table_ids_only=owner_table_ids_only,
                    )
                left_values = list(left_values)
                right_values = list(right_values)
                left_table_ids = tuple(left_table_ids)
                right_table_ids = tuple(right_table_ids)
                left_hits = int(left_hits)
                left_misses = int(left_misses)
                right_hits = int(right_hits)
                right_misses = int(right_misses)
                left_build_seconds = float(left_build_seconds)
                right_build_seconds = float(right_build_seconds)
                left_batch_used = bool(left_batch_used)
                right_batch_used = bool(right_batch_used)
                owner_precompute_used = True
                _increment_counter(owner_stats, "successes")
            except Exception as exc:
                _increment_counter(owner_stats, "failures")
                owner_stats["last_error"] = repr(exc)
        if not owner_precompute_used:
            (
                left_values,
                left_table_ids,
                left_hits,
                left_misses,
                left_build_seconds,
                left_batch_used,
            ) = (
                self._batch_build_missing_cached(
                    "left",
                    left_build_keys,
                    left_cache,
                    self.left_builder,
                    self.left_batch_builder if route_plan is not None else None,
                    family_name,
                    self.left_packed_boundary_table,
                    layout_token=(
                        ("route_plan", route_cache_token, "left")
                        if route_plan is not None
                        else None
                    ),
                )
            )
            (
                right_values,
                right_table_ids,
                right_hits,
                right_misses,
                right_build_seconds,
                right_batch_used,
            ) = (
                self._batch_build_missing_cached(
                    "right",
                    right_build_keys,
                    right_cache,
                    self.right_builder,
                    self.right_batch_builder if route_plan is not None else None,
                    family_name,
                    self.right_packed_boundary_table,
                    layout_token=(
                        ("route_plan", route_cache_token, "right")
                        if route_plan is not None
                        else None
                    ),
                )
            )
        owner_stats["last_used"] = bool(owner_precompute_used)
        owner_stats["last_left_keys"] = int(len(left_build_keys))
        owner_stats["last_right_keys"] = int(len(right_build_keys))
        owner_stats["last_table_ids_only"] = bool(owner_table_ids_only)
        if route_plan is None:
            left_results = {
                (tuple(pattern), str(piece)): result
                for (pattern, piece), result in zip(left_build_keys, left_values)
            }
            right_results = {
                (tuple(pattern), str(piece)): result
                for (pattern, piece), result in zip(right_build_keys, right_values)
            }
        left_unique = len(left_keys) if route_plan is not None else len(left_results)
        right_unique = len(right_keys) if route_plan is not None else len(right_results)
        left_table_id_hits = sum(1 for table_id in left_table_ids if int(table_id) >= 0)
        right_table_id_hits = sum(1 for table_id in right_table_ids if int(table_id) >= 0)
        left_table_backed = bool(
            owner_table_ids_only
            and len(left_table_ids) == int(left_unique)
            and left_table_id_hits == int(left_unique)
        )
        right_table_backed = bool(
            owner_table_ids_only
            and len(right_table_ids) == int(right_unique)
            and right_table_id_hits == int(right_unique)
        )
        left_payload_counts = (
            {"packed": int(left_unique)}
            if left_table_backed
            else _contextual_boundary_payload_counts(left_values)
        )
        right_payload_counts = (
            {"packed": int(right_unique)}
            if right_table_backed
            else _contextual_boundary_payload_counts(right_values)
        )

        phase_stats = self.stats.setdefault(
            "contextual_boundary_precompute",
            {"calls": 0},
        )
        _increment_counter(phase_stats, "calls")
        _increment_counter(phase_stats, "records", record_count)
        _increment_counter(phase_stats, "left_unique", left_unique)
        _increment_counter(phase_stats, "right_unique", right_unique)
        _increment_counter(phase_stats, "left_hits", left_hits)
        _increment_counter(phase_stats, "left_misses", left_misses)
        _increment_counter(phase_stats, "right_hits", right_hits)
        _increment_counter(phase_stats, "right_misses", right_misses)
        phase_stats["left_build_seconds"] = (
            float(phase_stats.get("left_build_seconds", 0.0))
            + float(left_build_seconds)
        )
        phase_stats["right_build_seconds"] = (
            float(phase_stats.get("right_build_seconds", 0.0))
            + float(right_build_seconds)
        )
        phase_stats["last_left_build_seconds"] = float(left_build_seconds)
        phase_stats["last_right_build_seconds"] = float(right_build_seconds)
        phase_stats["last_left_unique"] = int(left_unique)
        phase_stats["last_right_unique"] = int(right_unique)
        phase_stats["last_left_cache_size"] = int(len(left_cache))
        phase_stats["last_right_cache_size"] = int(len(right_cache))
        _increment_counter(phase_stats, "left_table_ids", left_table_id_hits)
        _increment_counter(phase_stats, "right_table_ids", right_table_id_hits)
        phase_stats["last_left_table_ids"] = int(left_table_id_hits)
        phase_stats["last_right_table_ids"] = int(right_table_id_hits)
        for kind, count in left_payload_counts.items():
            _increment_counter(phase_stats, f"left_payload_{kind}", count)
            phase_stats[f"last_left_payload_{kind}"] = int(count)
        for kind, count in right_payload_counts.items():
            _increment_counter(phase_stats, f"right_payload_{kind}", count)
            phase_stats[f"last_right_payload_{kind}"] = int(count)
        if (
            self.left_packed_boundary_table is not None
            or self.right_packed_boundary_table is not None
        ):
            packed_table_stats = self.stats.setdefault(
                "packed_contextual_boundary_tables",
                {},
            )
            if self.left_packed_boundary_table is not None:
                packed_table_stats["left"] = self.left_packed_boundary_table.stats
            if self.right_packed_boundary_table is not None:
                packed_table_stats["right"] = self.right_packed_boundary_table.stats
        _increment_counter(phase_stats, "left_batch_calls", int(left_batch_used))
        _increment_counter(phase_stats, "right_batch_calls", int(right_batch_used))
        _increment_counter(
            phase_stats,
            "owner_calls",
            int(owner_precompute is not None),
        )
        _increment_counter(
            phase_stats,
            "owner_successes",
            int(owner_precompute_used),
        )
        phase_stats["last_owner_used"] = bool(owner_precompute_used)
        self.record_phase(
            "contextual_boundary_precompute",
            time.perf_counter() - t_precompute,
            records=record_count,
            left_unique=left_unique,
            right_unique=right_unique,
            left_hits=left_hits,
            left_misses=left_misses,
            left_build_seconds=left_build_seconds,
            right_hits=right_hits,
            right_misses=right_misses,
            right_build_seconds=right_build_seconds,
            left_batch=int(left_batch_used),
            right_batch=int(right_batch_used),
            left_packed=left_payload_counts.get("packed", 0),
            right_packed=right_payload_counts.get("packed", 0),
            left_legacy=left_payload_counts.get("legacy", 0),
            right_legacy=right_payload_counts.get("legacy", 0),
            left_table_ids=left_table_id_hits,
            right_table_ids=right_table_id_hits,
            owner=int(owner_precompute_used),
        )
        boundary_batch = AbelianContextualBoundaryBatch(
            left_results,
            right_results,
            tuple(left_values),
            tuple(right_values),
            tuple(left_table_ids),
            tuple(right_table_ids),
            left_payload_counts,
            right_payload_counts,
        )
        if precompute_cache_key is not None:
            cacheable = bool(
                (left_table_backed and right_table_backed)
                or (
                    all(value is not None for value in tuple(left_values))
                    and all(value is not None for value in tuple(right_values))
                )
            )
            if cacheable:
                self.boundary_batch_cache[precompute_cache_key] = boundary_batch
                precompute_cache_stats = self.stats.setdefault(
                    "contextual_boundary_precompute_cache",
                    {"hits": 0, "misses": 0, "stores": 0},
                )
                _increment_counter(precompute_cache_stats, "stores")
                precompute_cache_stats["cache_size"] = int(
                    precompute_cache_stats.get("stores", 0)
                )
        return boundary_batch

    def boundary_pair(
        self,
        family_name,
        left_pattern,
        left_piece,
        right_piece,
        right_pattern,
        *,
        precompute_boundaries,
        boundary_batch,
    ):
        if precompute_boundaries:
            left_result = boundary_batch.left.get(
                (tuple(left_pattern), str(left_piece))
            )
            right_result = boundary_batch.right.get(
                (tuple(right_pattern), str(right_piece))
            )
        else:
            left_result = self.left_builder(
                left_pattern,
                left_piece,
                family_name=family_name,
            )
            right_result = self.right_builder(
                right_pattern,
                right_piece,
                family_name=family_name,
            )
        if left_result is None or right_result is None:
            raise ValueError("empty contextual operator")
        return left_result, right_result

    def build_entries(
        self,
        family_name,
        records,
        *,
        options,
        boundary_batch=None,
    ):
        if boundary_batch is None:
            boundary_batch = AbelianContextualBoundaryBatch({}, {})
        route_plan = records if isinstance(records, AbelianDirectRoutePlan) else None
        if route_plan is not None:
            family_name = route_plan.family_name
            record_count = route_plan.record_count
            route_cache_token = ("route_plan_object", id(route_plan))
        else:
            record_count = len(records or ())
            route_cache_token = None
        use_packed_buffer = bool(options.pack_entries and options.packed_buffer)
        entries = AbelianPackedDirectFamilyEntries() if use_packed_buffer else []
        t_entries = time.perf_counter()
        left_cache = self.left_cache
        right_cache = self.right_cache
        left_hits = left_misses = 0
        right_hits = right_misses = 0
        contextual_terms = 0
        fallback_terms = 0
        failed_terms = 0
        entry_source = f"contextual_{family_name}_local_generator_csr"
        append_entry = None if use_packed_buffer else entries.append
        entry_cls = self.entry_cls
        left_builder = self.left_builder
        right_builder = self.right_builder
        fallback_builder = self.fallback_builder

        def _snapshot_table_backed_planned_entries(planned_entries):
            if not bool(
                getattr(
                    planned_entries,
                    "_pyqed_planned_direct_family_table_ids",
                    False,
                )
            ):
                return planned_entries
            if not bool(options.snapshot_table_backed_planned_entries):
                snapshot_stats = self.stats.setdefault(
                    "contextual_planned_entry_snapshots",
                    {"calls": 0, "entries": 0},
                )
                _increment_counter(snapshot_stats, "kept_table_backed")
                _increment_counter(snapshot_stats, "kept_entries", int(len(planned_entries)))
                snapshot_stats["last_kept_entries"] = int(len(planned_entries))
                return planned_entries
            snapshot = getattr(planned_entries, "snapshot_table_payloads", None)
            if not callable(snapshot):
                return planned_entries
            snapshot_stats = self.stats.setdefault(
                "contextual_planned_entry_snapshots",
                {"calls": 0, "entries": 0},
            )
            snapshot_stats["calls"] = int(snapshot_stats.get("calls", 0)) + 1
            snapshot_stats["entries"] = (
                int(snapshot_stats.get("entries", 0))
                + int(len(planned_entries))
            )
            snapshot_stats["last_entries"] = int(len(planned_entries))
            return snapshot()

        if route_plan is not None:
            left_keys = route_plan.left_keys
            right_keys = route_plan.right_keys
            left_ids = route_plan.left_ids
            right_ids = route_plan.right_ids
            coeffs = route_plan.coeffs
            pair_left_ids = route_plan.pair_left_ids
            pair_right_ids = route_plan.pair_right_ids
            pair_coeffs = route_plan.pair_coeffs
            left_values = tuple(getattr(boundary_batch, "left_values", ()) or ())
            right_values = tuple(getattr(boundary_batch, "right_values", ()) or ())
            left_table_ids = tuple(getattr(boundary_batch, "left_table_ids", ()) or ())
            right_table_ids = tuple(getattr(boundary_batch, "right_table_ids", ()) or ())
            left_table_id_array = np.asarray(left_table_ids, dtype=np.int64)
            right_table_id_array = np.asarray(right_table_ids, dtype=np.int64)
            precomputed_table_ids = bool(
                options.precompute_boundaries
                and getattr(boundary_batch, "packed_boundary_pairs", False)
                and self.left_packed_boundary_table is not None
                and self.right_packed_boundary_table is not None
                and left_table_id_array.size == len(left_keys)
                and right_table_id_array.size == len(right_keys)
                and bool(np.all(left_table_id_array >= 0))
                and bool(np.all(right_table_id_array >= 0))
            )
            use_precomputed_arrays = bool(
                options.precompute_boundaries
                and (
                    (
                        len(left_values) == len(left_keys)
                        and len(right_values) == len(right_keys)
                    )
                    or precomputed_table_ids
                )
            )
            use_planned_arrays = bool(use_precomputed_arrays)
            if (
                not use_planned_arrays
                and not options.precompute_boundaries
                and options.planned_without_precompute
                and use_packed_buffer
                and entry_cls is AbelianPackedLocalGeneratorEntry
            ):
                t_lazy = time.perf_counter()
                use_lazy_batch = bool(
                    getattr(options, "planned_without_precompute_batch", True)
                )
                use_lazy_table = bool(
                    getattr(
                        options,
                        "planned_without_precompute_table_lookup",
                        True,
                    )
                )
                use_lazy_table_ids_only = bool(
                    use_lazy_table
                    and getattr(
                        options,
                        "planned_without_precompute_table_ids_only",
                        True,
                    )
                )

                def _lookup_unique_boundaries(keys, cache, builder):
                    values = []
                    hits = 0
                    misses = 0
                    packed = True
                    for pattern, piece in keys:
                        key = (tuple(pattern), str(piece))
                        if key in cache:
                            result = cache[key]
                            hits += 1
                        else:
                            try:
                                result = builder(
                                    tuple(pattern),
                                    str(piece),
                                    family_name=family_name,
                                )
                            except Exception:
                                result = None
                            cache[key] = result
                            misses += 1
                        if (
                            result is None
                            or _contextual_boundary_payload_kind(result) != "packed"
                        ):
                            packed = False
                        values.append(result)
                    return tuple(values), int(hits), int(misses), bool(packed)

                lazy_left_build_seconds = 0.0
                lazy_right_build_seconds = 0.0
                lazy_left_batch_used = False
                lazy_right_batch_used = False
                if use_lazy_batch or use_lazy_table:
                    left_lazy_table = (
                        self.left_packed_boundary_table if use_lazy_table else None
                    )
                    right_lazy_table = (
                        self.right_packed_boundary_table if use_lazy_table else None
                    )
                    (
                        lazy_left_values,
                        left_table_ids,
                        lazy_left_hits,
                        lazy_left_misses,
                        lazy_left_build_seconds,
                        lazy_left_batch_used,
                    ) = self._batch_build_missing_cached(
                        "left",
                        left_keys,
                        left_cache,
                        left_builder,
                        self.left_batch_builder if use_lazy_batch else None,
                        family_name,
                        left_lazy_table,
                        layout_token=("route_plan", route_cache_token, "left"),
                        table_ids_only=use_lazy_table_ids_only,
                    )
                    (
                        lazy_right_values,
                        right_table_ids,
                        lazy_right_hits,
                        lazy_right_misses,
                        lazy_right_build_seconds,
                        lazy_right_batch_used,
                    ) = self._batch_build_missing_cached(
                        "right",
                        right_keys,
                        right_cache,
                        right_builder,
                        self.right_batch_builder if use_lazy_batch else None,
                        family_name,
                        right_lazy_table,
                        layout_token=("route_plan", route_cache_token, "right"),
                        table_ids_only=use_lazy_table_ids_only,
                    )
                    lazy_left_values = tuple(lazy_left_values)
                    lazy_right_values = tuple(lazy_right_values)
                    if use_lazy_table_ids_only:
                        lazy_left_table_id_array = np.asarray(
                            left_table_ids,
                            dtype=np.int64,
                        )
                        lazy_right_table_id_array = np.asarray(
                            right_table_ids,
                            dtype=np.int64,
                        )
                        if bool(np.all(lazy_left_table_id_array >= 0)):
                            lazy_left_values = tuple(
                                left_lazy_table.values_for_ids(left_table_ids)
                            )
                        else:
                            lazy_left_values = tuple(
                                left_cache.get(
                                    (str(family_name), tuple(pattern), str(piece))
                                )
                                for pattern, piece in left_keys
                            )
                        if bool(np.all(lazy_right_table_id_array >= 0)):
                            lazy_right_values = tuple(
                                right_lazy_table.values_for_ids(right_table_ids)
                            )
                        else:
                            lazy_right_values = tuple(
                                right_cache.get(
                                    (str(family_name), tuple(pattern), str(piece))
                                )
                                for pattern, piece in right_keys
                            )
                        lazy_left_packed = bool(
                            lazy_left_values
                            and lazy_left_table_id_array.size == len(left_keys)
                            and bool(np.all(lazy_left_table_id_array >= 0))
                            and all(
                                _contextual_boundary_payload_kind(result) == "packed"
                                for result in lazy_left_values
                            )
                        )
                        lazy_right_packed = bool(
                            lazy_right_values
                            and lazy_right_table_id_array.size == len(right_keys)
                            and bool(np.all(lazy_right_table_id_array >= 0))
                            and all(
                                _contextual_boundary_payload_kind(result) == "packed"
                                for result in lazy_right_values
                            )
                        )
                    else:
                        lazy_left_packed = all(
                            _contextual_boundary_payload_kind(result) == "packed"
                            for result in lazy_left_values
                        )
                        lazy_right_packed = all(
                            _contextual_boundary_payload_kind(result) == "packed"
                            for result in lazy_right_values
                        )
                else:
                    (
                        lazy_left_values,
                        lazy_left_hits,
                        lazy_left_misses,
                        lazy_left_packed,
                    ) = _lookup_unique_boundaries(left_keys, left_cache, left_builder)
                    (
                        lazy_right_values,
                        lazy_right_hits,
                        lazy_right_misses,
                        lazy_right_packed,
                    ) = _lookup_unique_boundaries(right_keys, right_cache, right_builder)
                    left_table_ids = tuple(-1 for _ in lazy_left_values)
                    right_table_ids = tuple(-1 for _ in lazy_right_values)
                for key, value in zip(left_keys, lazy_left_values):
                    left_cache[key] = value
                for key, value in zip(right_keys, lazy_right_values):
                    right_cache[key] = value
                left_hits += int(lazy_left_hits)
                left_misses += int(lazy_left_misses)
                right_hits += int(lazy_right_hits)
                right_misses += int(lazy_right_misses)
                left_table_id_array = np.asarray(left_table_ids, dtype=np.int64)
                right_table_id_array = np.asarray(right_table_ids, dtype=np.int64)
                lazy_left_table_id_hits = int(
                    np.count_nonzero(left_table_id_array >= 0)
                )
                lazy_right_table_id_hits = int(
                    np.count_nonzero(right_table_id_array >= 0)
                )
                lazy_stats = self.stats.setdefault(
                    "contextual_route_lazy_pack",
                    {"calls": 0},
                )
                _increment_counter(lazy_stats, "calls")
                _increment_counter(lazy_stats, "left_hits", lazy_left_hits)
                _increment_counter(lazy_stats, "left_misses", lazy_left_misses)
                _increment_counter(lazy_stats, "right_hits", lazy_right_hits)
                _increment_counter(lazy_stats, "right_misses", lazy_right_misses)
                _increment_counter(lazy_stats, "left_batch_calls", int(lazy_left_batch_used))
                _increment_counter(lazy_stats, "right_batch_calls", int(lazy_right_batch_used))
                _increment_counter(lazy_stats, "left_table_ids", lazy_left_table_id_hits)
                _increment_counter(lazy_stats, "right_table_ids", lazy_right_table_id_hits)
                lazy_stats["last_left_unique"] = int(len(left_keys))
                lazy_stats["last_right_unique"] = int(len(right_keys))
                lazy_stats["last_left_table_ids"] = int(lazy_left_table_id_hits)
                lazy_stats["last_right_table_ids"] = int(lazy_right_table_id_hits)
                lazy_stats["table_lookup"] = bool(use_lazy_table)
                lazy_stats["table_ids_only"] = bool(use_lazy_table_ids_only)
                lazy_stats["last_left_build_seconds"] = float(lazy_left_build_seconds)
                lazy_stats["last_right_build_seconds"] = float(lazy_right_build_seconds)
                lazy_stats["left_build_seconds"] = (
                    float(lazy_stats.get("left_build_seconds", 0.0))
                    + float(lazy_left_build_seconds)
                )
                lazy_stats["right_build_seconds"] = (
                    float(lazy_stats.get("right_build_seconds", 0.0))
                    + float(lazy_right_build_seconds)
                )
                lazy_stats["last_seconds"] = float(time.perf_counter() - t_lazy)
                lazy_stats["seconds"] = (
                    float(lazy_stats.get("seconds", 0.0))
                    + float(lazy_stats["last_seconds"])
                )
                if lazy_left_packed and lazy_right_packed:
                    left_values = () if use_lazy_table_ids_only else lazy_left_values
                    right_values = () if use_lazy_table_ids_only else lazy_right_values
                    boundary_batch = AbelianContextualBoundaryBatch(
                        {},
                        {},
                        left_values,
                        right_values,
                        left_table_ids,
                        right_table_ids,
                        (
                            {"packed": len(left_table_ids)}
                            if use_lazy_table_ids_only
                            else _contextual_boundary_payload_counts(left_values)
                        ),
                        (
                            {"packed": len(right_table_ids)}
                            if use_lazy_table_ids_only
                            else _contextual_boundary_payload_counts(right_values)
                        ),
                    )
                    use_planned_arrays = True
                    _increment_counter(lazy_stats, "planned_calls")
                    _increment_counter(lazy_stats, "planned_entries", int(record_count))
                    _increment_counter(lazy_stats, "planned_compact_pairs", int(pair_coeffs.shape[0]))
                else:
                    _increment_counter(lazy_stats, "fallbacks")
                    lazy_stats["last_fallback_reason"] = "nonpacked_or_missing_boundary"
            route_fast_packed = False
            if (
                use_planned_arrays
                and use_packed_buffer
                and entry_cls is AbelianPackedLocalGeneratorEntry
            ):
                planned_count = int(record_count)
                compact_pair_count = int(pair_coeffs.shape[0])
                route_fast_packed = bool(
                    getattr(boundary_batch, "packed_boundary_pairs", False)
                )
                if not route_fast_packed:
                    route_fast_packed = True
                    for route_idx in range(planned_count):
                        left_id = int(left_ids[route_idx])
                        right_id = int(right_ids[route_idx])
                        left_result = left_values[left_id]
                        right_result = right_values[right_id]
                        if left_result is None or right_result is None:
                            route_fast_packed = False
                            break
                        try:
                            E_term, W_left = left_result
                            W_right, F_term = right_result
                        except Exception:
                            route_fast_packed = False
                            break
                if route_fast_packed:
                    left_table_id_array = np.asarray(left_table_ids, dtype=np.int64)
                    right_table_id_array = np.asarray(right_table_ids, dtype=np.int64)
                    table_backed_possible = bool(
                        self.left_packed_boundary_table is not None
                        and self.right_packed_boundary_table is not None
                        and left_table_id_array.size == len(left_keys)
                        and right_table_id_array.size == len(right_keys)
                        and bool(np.all(left_table_id_array >= 0))
                        and bool(np.all(right_table_id_array >= 0))
                    )
                    left_table_ids_tuple = tuple(
                        int(value) for value in left_table_id_array
                    )
                    right_table_ids_tuple = tuple(
                        int(value) for value in right_table_id_array
                    )
                    stable_table_key = bool(
                        table_backed_possible
                        and not bool(options.snapshot_table_backed_planned_entries)
                    )

                    def _table_token(table):
                        if table is None:
                            return None
                        if stable_table_key:
                            return (
                                id(table),
                                int(getattr(table, "revision", -1)),
                            )
                        return (
                            id(table),
                            int(getattr(table, "revision", -1)),
                            int(getattr(table, "puts", 0)),
                            int(getattr(table, "evictions", 0)),
                            int(len(getattr(table, "entries", {}) or {})),
                            int(len(getattr(table, "payloads", ()) or ())),
                        )

                    planned_cache_key = (
                        "planned_direct_family_entries",
                        route_cache_token,
                        _table_token(self.left_packed_boundary_table),
                        _table_token(self.right_packed_boundary_table),
                        left_table_ids_tuple,
                        right_table_ids_tuple,
                        entry_source,
                    )
                    planned_cache_stats = self.stats.setdefault(
                        "contextual_planned_entry_cache",
                        {"hits": 0, "builds": 0},
                    )
                    entries = self.planned_entries_cache.get(planned_cache_key)
                    if entries is not None:
                        planned_cache_stats["hits"] = (
                            int(planned_cache_stats.get("hits", 0)) + 1
                        )
                    else:
                        owner_entries = None
                        owner = self.boundary_batch_owner
                        owner_builder = (
                            None
                            if owner is None
                            else getattr(
                                owner,
                                "build_planned_direct_family_entries_from_route",
                                None,
                            )
                        )
                        if owner_builder is not None:
                            try:
                                owner_entries = owner_builder(
                                    AbelianPlannedPackedDirectFamilyEntries,
                                    route_plan,
                                    boundary_batch,
                                    self.left_packed_boundary_table,
                                    self.right_packed_boundary_table,
                                    entry_source,
                                )
                            except Exception as exc:
                                planned_cache_stats[
                                    "owner_build_failures"
                                ] = (
                                    int(
                                        planned_cache_stats.get(
                                            "owner_build_failures",
                                            0,
                                        )
                                    )
                                    + 1
                                )
                                planned_cache_stats["owner_build_last_error"] = (
                                    repr(exc)
                                )
                                owner_entries = None
                        if owner_entries is not None:
                            entries = owner_entries
                            planned_cache_stats["owner_builds"] = (
                                int(planned_cache_stats.get("owner_builds", 0))
                                + 1
                            )
                            planned_cache_stats["owner_entries"] = (
                                int(planned_cache_stats.get("owner_entries", 0))
                                + int(len(entries))
                            )
                            planned_cache_stats["backend_actual"] = (
                                "cpp_moving_environment"
                            )
                        else:
                            entries = AbelianPlannedPackedDirectFamilyEntries.from_route_plan(
                                route_plan,
                                boundary_batch,
                                left_table=self.left_packed_boundary_table,
                                right_table=self.right_packed_boundary_table,
                                source=entry_source,
                            )
                            planned_cache_stats["python_builds"] = (
                                int(planned_cache_stats.get("python_builds", 0))
                                + 1
                            )
                            planned_cache_stats["python_entries"] = (
                                int(planned_cache_stats.get("python_entries", 0))
                                + int(len(entries))
                            )
                            planned_cache_stats["backend_actual"] = "python"
                        entries = _snapshot_table_backed_planned_entries(entries)
                        if bool(entries._pyqed_planned_direct_family_table_ids):
                            self.planned_entries_cache[planned_cache_key] = entries
                        planned_cache_stats["builds"] = (
                            int(planned_cache_stats.get("builds", 0)) + 1
                        )
                    planned_cache_stats["last_entries"] = int(len(entries))
                    planned_cache_stats["last_compact_pairs"] = int(
                        compact_pair_count
                    )
                    planned_cache_stats["last_stable_table_key"] = bool(stable_table_key)
                    planned_cache_stats["cache_size"] = int(
                        len(self.planned_entries_cache)
                    )
                    contextual_terms += record_count
                    fast_stats = self.stats.setdefault(
                        "contextual_route_fast_pack",
                        {"calls": 0},
                    )
                    _increment_counter(fast_stats, "calls")
                    _increment_counter(fast_stats, "entries", compact_pair_count)
                    _increment_counter(
                        fast_stats,
                        "compact_pairs",
                        compact_pair_count,
                    )
                    _increment_counter(
                        fast_stats,
                        "coalesced_records",
                        planned_count - compact_pair_count,
                    )
                    fast_stats["last_entries"] = int(compact_pair_count)
                    fast_stats["last_records"] = int(record_count)
                    fast_stats["last_compact_pairs"] = int(compact_pair_count)
                    fast_stats["last_coalesced_records"] = int(
                        planned_count - compact_pair_count
                    )
                    _increment_counter(
                        fast_stats,
                        "planned_entries",
                        compact_pair_count,
                    )
                    _increment_counter(fast_stats, "planned_calls")
                    left_table_id_hits = int(
                        np.count_nonzero(left_table_id_array >= 0)
                    )
                    right_table_id_hits = int(
                        np.count_nonzero(right_table_id_array >= 0)
                    )
                    _increment_counter(
                        fast_stats,
                        "left_table_ids",
                        left_table_id_hits,
                    )
                    _increment_counter(
                        fast_stats,
                        "right_table_ids",
                        right_table_id_hits,
                    )
                    fast_stats["last_left_table_ids"] = int(left_table_id_hits)
                    fast_stats["last_right_table_ids"] = int(right_table_id_hits)
                    if bool(entries._pyqed_planned_direct_family_table_ids):
                        _increment_counter(fast_stats, "table_backed_calls")
                        _increment_counter(
                            fast_stats,
                            "table_backed_entries",
                            planned_count,
                        )
                        fast_stats["last_table_backed"] = True
                    else:
                        fast_stats["last_table_backed"] = False
                    if getattr(boundary_batch, "packed_boundary_pairs", False):
                        _increment_counter(fast_stats, "packed_boundary_calls")
                        _increment_counter(
                            fast_stats,
                            "packed_boundary_entries",
                            planned_count,
                        )
                        fast_stats["last_boundary_payload"] = "packed"
                    else:
                        _increment_counter(fast_stats, "nonpacked_boundary_calls")
                        _increment_counter(
                            fast_stats,
                            "nonpacked_boundary_entries",
                            planned_count,
                        )
                        fast_stats["last_boundary_payload"] = "mixed_or_legacy"
            if not route_fast_packed:
                for route_idx in range(record_count):
                    left_id = int(left_ids[route_idx])
                    right_id = int(right_ids[route_idx])
                    left_key = left_keys[left_id]
                    right_key = right_keys[right_id]
                    left_pattern, left_piece = left_key
                    right_pattern, right_piece = right_key
                    coeff = coeffs[route_idx]
                    try:
                        if options.precompute_boundaries:
                            if use_precomputed_arrays:
                                left_result = left_values[left_id]
                                right_result = right_values[right_id]
                            else:
                                left_result = boundary_batch.left.get(left_key)
                                right_result = boundary_batch.right.get(right_key)
                            if left_result is None or right_result is None:
                                raise ValueError("empty contextual operator")
                        else:
                            if left_key in left_cache:
                                left_result = left_cache[left_key]
                                left_hits += 1
                            else:
                                try:
                                    left_result = left_builder(
                                        left_pattern,
                                        left_piece,
                                        family_name=family_name,
                                    )
                                except Exception:
                                    left_result = None
                                left_cache[left_key] = left_result
                                left_misses += 1
                            if right_key in right_cache:
                                right_result = right_cache[right_key]
                                right_hits += 1
                            else:
                                try:
                                    right_result = right_builder(
                                        right_pattern,
                                        right_piece,
                                        family_name=family_name,
                                    )
                                except Exception:
                                    right_result = None
                                right_cache[right_key] = right_result
                                right_misses += 1
                            if left_result is None or right_result is None:
                                raise ValueError("empty contextual operator")
                        E_term, W_left = left_result
                        W_right, F_term = right_result
                        contextual_terms += 1
                    except Exception:
                        try:
                            E_term, W_left, W_right, F_term = fallback_builder(
                                left_pattern,
                                left_piece,
                                right_piece,
                                right_pattern,
                            )
                            fallback_terms += 1
                        except Exception:
                            failed_terms += 1
                            if contextual_terms:
                                _increment_counter(
                                    self.stats,
                                    "contextual_recursive_terms",
                                    contextual_terms,
                                )
                            if fallback_terms:
                                _increment_counter(
                                    self.stats,
                                    "fallback_full_pattern_terms",
                                    fallback_terms,
                                )
                            _increment_counter(self.stats, "failed_terms", failed_terms)
                            return AbelianContextualEntryBuildResult(
                                None,
                                time.perf_counter() - t_entries,
                            )

                    if options.pack_entries:
                        if use_packed_buffer and entry_cls is AbelianPackedLocalGeneratorEntry:
                            entries.append_local_generator(
                                coeff,
                                E_term,
                                W_left,
                                W_right,
                                F_term,
                                source=entry_source,
                            )
                        else:
                            append_entry(
                                entry_cls(
                                    coeff,
                                    E_term,
                                    W_left,
                                    W_right,
                                    F_term,
                                    source=entry_source,
                                )
                            )
                    else:
                        append_entry((E_term, [W_left * complex(coeff), W_right], F_term))
        else:
            for left_pattern, left_piece, right_piece, right_pattern, coeff in (
                records or ()
            ):
                try:
                    if options.precompute_boundaries:
                        left_result, right_result = self.boundary_pair(
                            family_name,
                            left_pattern,
                            left_piece,
                            right_piece,
                            right_pattern,
                            precompute_boundaries=True,
                            boundary_batch=boundary_batch,
                        )
                    else:
                        left_key = (left_pattern, left_piece)
                        if left_key in left_cache:
                            left_result = left_cache[left_key]
                            left_hits += 1
                        else:
                            try:
                                left_result = left_builder(
                                    left_pattern,
                                    left_piece,
                                    family_name=family_name,
                                )
                            except Exception:
                                left_result = None
                            left_cache[left_key] = left_result
                            left_misses += 1
                        right_key = (right_pattern, right_piece)
                        if right_key in right_cache:
                            right_result = right_cache[right_key]
                            right_hits += 1
                        else:
                            try:
                                right_result = right_builder(
                                    right_pattern,
                                    right_piece,
                                    family_name=family_name,
                                )
                            except Exception:
                                right_result = None
                            right_cache[right_key] = right_result
                            right_misses += 1
                        if left_result is None or right_result is None:
                            raise ValueError("empty contextual operator")
                    E_term, W_left = left_result
                    W_right, F_term = right_result
                    contextual_terms += 1
                except Exception:
                    try:
                        E_term, W_left, W_right, F_term = fallback_builder(
                            left_pattern,
                            left_piece,
                            right_piece,
                            right_pattern,
                        )
                        fallback_terms += 1
                    except Exception:
                        failed_terms += 1
                        if contextual_terms:
                            _increment_counter(
                                self.stats,
                                "contextual_recursive_terms",
                                contextual_terms,
                            )
                        if fallback_terms:
                            _increment_counter(
                                self.stats,
                                "fallback_full_pattern_terms",
                                fallback_terms,
                            )
                        _increment_counter(self.stats, "failed_terms", failed_terms)
                        return AbelianContextualEntryBuildResult(
                            None,
                            time.perf_counter() - t_entries,
                        )

                if options.pack_entries:
                    if use_packed_buffer and entry_cls is AbelianPackedLocalGeneratorEntry:
                        entries.append_local_generator(
                            coeff,
                            E_term,
                            W_left,
                            W_right,
                            F_term,
                            source=entry_source,
                        )
                    else:
                        append_entry(
                            entry_cls(
                                coeff,
                                E_term,
                                W_left,
                                W_right,
                                F_term,
                                source=entry_source,
                            )
                        )
                else:
                    append_entry((E_term, [W_left * complex(coeff), W_right], F_term))
        if contextual_terms:
            _increment_counter(
                self.stats,
                "contextual_recursive_terms",
                contextual_terms,
            )
        if fallback_terms:
            _increment_counter(
                self.stats,
                "fallback_full_pattern_terms",
                fallback_terms,
            )
        if not options.precompute_boundaries and record_count:
            cache_stats = self.stats.setdefault(
                "contextual_lazy_boundary_cache",
                {"calls": 0},
            )
            _increment_counter(cache_stats, "calls")
            _increment_counter(cache_stats, "left_hits", left_hits)
            _increment_counter(cache_stats, "left_misses", left_misses)
            _increment_counter(cache_stats, "right_hits", right_hits)
            _increment_counter(cache_stats, "right_misses", right_misses)
            cache_stats["last_left_unique"] = int(len(left_cache))
            cache_stats["last_right_unique"] = int(len(right_cache))
        return AbelianContextualEntryBuildResult(
            entries,
            time.perf_counter() - t_entries,
        )


class AbelianContextualComponentStore:
    """Store exact contextual family entries and apply optional compression."""

    def __init__(
        self,
        *,
        component_table,
        family_options,
        matvec_options,
        stats,
        record_phase,
        validate_entries,
        bond,
    ):
        self.component_table = component_table
        self.family_options = family_options
        self.matvec_options = matvec_options or {}
        self.stats = stats
        self.record_phase = record_phase
        self.validate_entries = validate_entries
        self.bond = int(bond)

    def compression_options(self):
        policy = getattr(
            self.family_options,
            "exact_component_compression_policy",
            "auto",
        )
        validate_compressed = bool(
            getattr(
                self.family_options,
                "exact_component_compression_validate",
                True,
            )
        )
        min_reduction = int(
            getattr(
                self.family_options,
                "exact_component_compression_min_reduction",
                1,
            )
        )
        max_group_size = int(
            getattr(
                self.family_options,
                "exact_component_compression_max_group_size",
                64,
            )
        )
        if (
            str(policy).lower().replace("-", "_") == "auto"
            and (
                (
                    "native_boundary_p" in self.stats
                    and str(
                        self.stats["native_boundary_p"].get(
                            "validation_policy",
                            "",
                        )
                    )
                    == "off"
                )
                or bool(
                    self.matvec_options.get(
                        "generator_table_allow_planned_packed_contextual_entries",
                        False,
                    )
                )
            )
        ):
            fast_cap = int(
                self.matvec_options.get(
                    "generator_table_exact_component_compression_fast_max_group_size",
                    1,
                )
            )
            if fast_cap > 0:
                max_group_size = (
                    fast_cap
                    if max_group_size <= 0
                    else min(int(max_group_size), fast_cap)
                )
        return policy, validate_compressed, min_reduction, max_group_size

    def store(self, family_name, original_entries, records):
        policy, validate_compressed, min_reduction, max_group_size = (
            self.compression_options()
        )
        original_is_packed = bool(
            getattr(original_entries, "_pyqed_packed_direct_family_entries", False)
        )
        entries_to_store = (
            original_entries if original_is_packed else tuple(original_entries or ())
        )
        can_fast_coalesce = (
            str(policy).lower().replace("-", "_") == "auto"
            and max_group_size is not None
            and int(max_group_size) <= 1
            and bool(
                self.matvec_options.get(
                    "generator_table_coalesce_contextual_entries",
                    False,
                )
            )
        )
        if can_fast_coalesce:
            before = int(len(entries_to_store))
            if original_is_packed:
                before_identity = int(entries_to_store.identity_count)
                before_local = int(entries_to_store.local_generator_count)
                packed_stats = entries_to_store.coalesce_in_place()
                packed_identity = before_identity
                unique_identity = int(entries_to_store.identity_count)
                cancelled_identity = int(packed_stats["cancelled_identity"])
                packed_local = before_local
                unique_local = int(entries_to_store.local_generator_count)
                cancelled_local = int(packed_stats["cancelled_local"])
            else:
                (
                    entries_to_store,
                    packed_identity,
                    unique_identity,
                    cancelled_identity,
                ) = _coalesced_packed_identity_local_entries(entries_to_store)
                (
                    entries_to_store,
                    packed_local,
                    unique_local,
                    cancelled_local,
                ) = _coalesced_packed_local_generator_entries(entries_to_store)
            after = int(len(entries_to_store))
            if packed_identity or packed_local:
                coalesce_stats = self.stats.setdefault(
                    "contextual_entry_coalesce",
                    {"calls": 0},
                )
                _increment_counter(coalesce_stats, "calls")
                _increment_counter(coalesce_stats, "original_entries", before)
                _increment_counter(coalesce_stats, "stored_entries", after)
                _increment_counter(coalesce_stats, "packed_identity", packed_identity)
                _increment_counter(coalesce_stats, "unique_identity", unique_identity)
                _increment_counter(
                    coalesce_stats,
                    "cancelled_identity",
                    cancelled_identity,
                )
                _increment_counter(coalesce_stats, "packed_local", packed_local)
                _increment_counter(coalesce_stats, "unique_local", unique_local)
                _increment_counter(coalesce_stats, "cancelled_local", cancelled_local)
                coalesce_stats["last_family"] = str(family_name)
                coalesce_stats["last_bond"] = self.bond
                coalesce_stats["last_reduction"] = int(before - after)
        t_put_family = time.perf_counter()
        entries = self.component_table.put_family(
            family_name,
            entries_to_store,
            records=records,
            compression_policy=policy,
            min_reduction=min_reduction,
            max_group_size=None if max_group_size <= 0 else max_group_size,
        )
        self.record_phase(
            "component_table_put_family",
            time.perf_counter() - t_put_family,
            original_entries=len(original_entries),
            stored_entries=len(entries),
        )
        if (
            validate_compressed
            and len(entries) < len(original_entries)
            and not self.validate_entries(
                tuple(entries),
                original_entries,
                max_vectors=int(
                    getattr(
                        self.family_options,
                        "exact_component_compression_validation_vectors",
                        1,
                    )
                ),
            )
        ):
            entries = self.component_table.put_family(
                family_name,
                original_entries,
                records=records,
                compression_policy="none",
            )
            validation_stats = self.stats.setdefault(
                "native_exact_component_compression_validation",
                {"rejected": 0},
            )
            _increment_counter(validation_stats, "rejected")
            validation_stats["last_family"] = str(family_name)
            validation_stats["last_bond"] = self.bond
        elif validate_compressed and len(entries) < len(original_entries):
            validation_stats = self.stats.setdefault(
                "native_exact_component_compression_validation",
                {"accepted": 0},
            )
            _increment_counter(validation_stats, "accepted")
            validation_stats["last_family"] = str(family_name)
            validation_stats["last_bond"] = self.bond
        return entries
