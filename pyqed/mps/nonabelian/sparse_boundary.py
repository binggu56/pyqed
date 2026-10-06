"""Experimental sparse storage and bounded gathering for reduced SU(2) boundaries.

Contractions use the existing real compiled kernel in bounded output batches.
No numerical truncation is applied. One-site reduced adjoints can consume
these blocks lazily; the solver's default boundary dispatch remains dense.
"""
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import replace
import time
import tempfile

import numpy as np
from scipy.sparse import bsr_matrix

from . import environment as env
from .su2_qchem_plan import PackedArrayPool


class _DiskBlocks(Mapping):
    """Temporary append-only BSR arrays; only offsets and shapes stay resident.

    A channel is read once into an immutable buffer shared by its array views.
    Those views own the buffer and remain valid across subsequent file reads.
    """

    def __init__(self, directory, budget):
        self.file = tempfile.TemporaryFile(dir=directory)
        self.budget = int(budget)
        self.records, self.specs, self.support = {}, {}, {}
        self.nbytes = self.disk_bytes = 0

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    def __contains__(self, key):
        return key in self.records

    def __setitem__(self, key, parts):
        if key in self.records:
            raise ValueError('Disk boundary blocks are append-only')
        size = sum(p.data.nbytes+p.indices.nbytes+p.indptr.nbytes for p in parts)
        required = self.disk_bytes + size
        if required > self.budget:
            raise MemoryError('Sparse boundary disk storage exceeds budget: '
                              f'{required} B required, limit {self.budget} B')
        self.file.seek(self.disk_bytes)
        start = self.disk_bytes
        layout = []
        for part in parts:
            # BSR index lengths follow from data shape; retain one descriptor
            # per component instead of three nested array descriptors.
            layout.append((part.data.shape, part.data.dtype, part.indices.dtype,
                           part.indptr.dtype, part.indptr.size))
            for array in (part.data, part.indices, part.indptr):
                array.tofile(self.file)
        self.disk_bytes = self.file.tell()
        self.records[key] = start, size, layout
        self.specs[key] = (len(parts), *parts[0].shape)
        self.support[key] = any(np.any(p.data) for p in parts)
        self.nbytes += size

    def __getitem__(self, key):
        start, size, layout = self.records[key]
        self.file.seek(start)
        payload = self.file.read(size)
        if len(payload) != size:
            raise OSError('Incomplete sparse boundary payload')
        _, rows, columns = self.specs[key]
        parts = []
        offset = 0
        for shape, dtype, index_dtype, pointer_dtype, pointer_count in layout:
            count = int(np.prod(shape))
            data = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape)
            offset += data.nbytes
            indices = np.frombuffer(payload, dtype=index_dtype, count=shape[0], offset=offset)
            offset += indices.nbytes
            indptr = np.frombuffer(payload, dtype=pointer_dtype, count=pointer_count, offset=offset)
            offset += indptr.nbytes
            parts.append(bsr_matrix((data, indices, indptr), shape=(rows, columns), copy=False))
        return tuple(parts)


class _SparseChannels(Mapping):
    def __init__(self, boundary, indices):
        self.boundary, self.indices = boundary, indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def __contains__(self, channel):
        return channel in self.indices

    def __getitem__(self, channel):
        return self.boundary.array(self.indices[channel])


class SparseBoundary:
    """Real rank-coupled blocks with an explicitly bounded dense decode cache.

    Keys are ``(bra_sector, ket_sector, channel)``. Values are component-first
    dense arrays or tuples of BSR matrices. Treat stored blocks as immutable;
    create a new boundary when their values change. Bounds cover individual
    numerical arenas and retained sparse arrays, not total process memory.
    """

    def __init__(self, blocks, *, side, bond, decode_bytes=8 * 2**20):
        if side not in ('left', 'right') or decode_bytes < 0:
            raise ValueError('Expected left/right side and nonnegative decode budget')
        self.side, self.bond = side, int(bond)
        self.blocks = {}
        self._decoded = OrderedDict()
        self._decoded_bytes = 0
        self.decode_bytes = decode_bytes
        disk_backed = isinstance(blocks, _DiskBlocks)
        for key, value in (() if disk_backed else blocks.items()):
            parts = []
            for component in value:
                if isinstance(component, bsr_matrix):
                    if np.iscomplexobj(component.data):
                        raise ValueError('SparseBoundary currently requires real blocks')
                    parts.append(component)
                else:
                    array = np.asarray(component)
                    if array.ndim != 2 or np.iscomplexobj(array):
                        raise ValueError('Expected real component matrices')
                    tile = 2 if all(d % 2 == 0 for d in array.shape) else 1
                    parts.append(bsr_matrix(array, blocksize=(tile, tile)))
            if not parts or any(min(p.shape) <= 0 or p.shape != parts[0].shape for p in parts):
                raise ValueError('Boundary components must have matching nonempty shapes')
            self.blocks[key] = tuple(parts)
        if disk_backed:
            self.blocks = blocks
        specs = blocks.specs if disk_backed else {k: (len(v), *v[0].shape) for k, v in self.blocks.items()}
        self._specs = specs
        self._support = (blocks.support if disk_backed else
                         {k: any(np.any(p.data) for p in v) for k, v in self.blocks.items()})
        table, indices = env._empty_packed_boundary_from_specs(specs, side=side, bond=bond)
        self.keys = [None] * len(indices)
        self._channel_indices = {}
        for key, index in indices.items():
            self.keys[index] = key
            qb, qk, channel = key
            self._channel_indices.setdefault((qb, qk), {})[channel] = index
        self.offsets = table.block_pool.offsets
        self.shape_offsets = table.block_pool.shape_offsets
        self.shapes = table.block_pool.shapes
        self._table = table
        self.nbytes = blocks.nbytes if disk_backed else sum(
            p.data.nbytes + p.indices.nbytes + p.indptr.nbytes
            for parts in self.blocks.values() for p in parts)
        self.last_advance = None

    @property
    def decode_bytes(self):
        return self._decode_bytes

    @decode_bytes.setter
    def decode_bytes(self, value):
        if not np.isfinite(value) or value < 0:
            raise ValueError('Decode budget must be finite and nonnegative')
        self._decode_bytes = int(value)
        while self._decoded and self._decoded_bytes > self._decode_bytes:
            _, old = self._decoded.popitem(last=False)
            self._decoded_bytes -= old.nbytes

    @property
    def data(self):
        raise RuntimeError('Sparse boundaries have no dense arena; gather specific blocks')

    def release_decoded(self):
        """Discard dense cached copies while preserving immutable sparse values."""
        self._decoded.clear()
        self._decoded_bytes = 0

    def has_nonzero(self, index):
        """Conservative exact support without decoding immutable sparse blocks.

        Explicit BSR zeros are excluded. Duplicate entries that cancel may
        retain an unnecessary route, but never discard a contributing one.
        """
        return self._support[self.keys[index]]

    @property
    def table(self):
        # A stored table pointing back to self would retain old boundary arrays
        # until cyclic garbage collection during a sweep.
        return replace(self._table, block_pool=self)

    def get(self, sector_pair, default=None):
        """Expose channels lazily for reduced local contractions."""
        indices = self._channel_indices.get(sector_pair)
        return default if indices is None else _SparseChannels(self, indices)

    @classmethod
    def from_reduced(cls, block, *, side, bond, **kwargs):
        """Pack a real channel-resolved endpoint or reference boundary."""
        blocks = {}
        for (qb, qk), channels in block.data.items():
            for channel, array in channels.items():
                if env._has_material_imaginary_part(array):
                    raise NotImplementedError('Sparse boundaries require real values')
                blocks[qb, qk, channel] = np.asarray(array).real
        return cls(blocks, side=side, bond=bond, **kwargs)

    @property
    def stats(self):
        return dict(kind='sparse_reduced_boundary', n_arrays=len(self.keys),
                    stored_bytes=self.nbytes, decoded_bytes=self._decoded_bytes,
                    resident_sparse_bytes=0 if isinstance(self.blocks, _DiskBlocks) else self.nbytes,
                    disk_bytes=getattr(self.blocks, 'disk_bytes', 0),
                    dense_equivalent_bytes=int(self.offsets[-1])*8)

    def array(self, index):
        """Decode one block, retaining at most ``decode_bytes`` of dense data."""
        index = int(index)
        cached = self._decoded.get(index)
        if cached is not None:
            self._decoded.move_to_end(index)
            return cached
        # SciPy's direct BSR decoder builds COO indices; CSR uses its dense
        # conversion kernel without that intermediate coordinate expansion.
        parts = self.blocks[self.keys[index]]
        dtype = np.result_type(*(p.dtype for p in parts))
        array = np.empty((len(parts), *parts[0].shape), dtype=dtype)
        for component, part in enumerate(parts):
            part.tocsr().astype(dtype, copy=False).toarray(out=array[component])
        if array.nbytes <= self.decode_bytes:
            while self._decoded and self._decoded_bytes + array.nbytes > self.decode_bytes:
                _, old = self._decoded.popitem(last=False)
                self._decoded_bytes -= old.nbytes
            self._decoded[index] = array
            self._decoded_bytes += array.nbytes
        return array

    def advance(self, core, bra, ket=None, *, moving_environment,
                arena_bytes=16 * 2**20, storage_bytes=256 * 2**20,
                spill_directory=None, disk_bytes=8 * 2**30, release_recoupling=False):
        """Advance real Hamiltonian blocks with bounded parent/output gatherings.

        Identity-metric-specific transport and genuinely complex tensors are
        unsupported. Scratch slots beyond the physical chain are released on every exit.
        Existing physical boundary slots are preserved.
        With ``spill_directory``, child BSR arrays are written to a temporary
        file instead of retained in RAM. ``disk_bytes`` limits that file;
        ``storage_bytes`` then limits one decoded sparse block. Metadata,
        decode caches, and active contraction arenas have separate lifetimes.
        ``release_recoupling`` drops the core cache once routes own the needed
        coefficients, avoiding overlap with their packed copies during transport.
        """
        if (not np.isfinite(arena_bytes) or not np.isfinite(storage_bytes)
                or not np.isfinite(disk_bytes)
                or arena_bytes <= 0 or storage_bytes <= 0 or disk_bytes <= 0):
            raise ValueError('Boundary budgets must be positive')
        ket = bra if ket is None else ket
        if getattr(core, 'fully_reduced_identity', False):
            raise NotImplementedError('Sparse identity-metric transport is not implemented')
        if any(env._has_material_imaginary_part(a)
               for site in (bra, ket) for a in site.data.values()):
            raise NotImplementedError('Sparse boundary execution currently requires real tensors')
        started = time.perf_counter()
        plan = env._plan_rank_coupled_boundary(
            core, bra, self.table, ket, side=self.side, require_real=True, grouped_routes=True)
        bond = self.bond + (1 if self.side == 'left' else -1)
        stats = dict(parent_bytes=self.nbytes, stored_bytes=0, maximum_gather_bytes=0,
                     maximum_output_bytes=0, completed_blocks=0, completed_batches=0, completed_gathers=0)
        self.last_advance = stats
        if plan is None:
            stats['seconds'] = time.perf_counter() - started
            return type(self)({}, side=self.side, bond=bond, decode_bytes=self.decode_bytes)
        aa, bb, ww, _, _, routes, specs = plan
        del plan
        cache_bytes = sum(a.nbytes for blocks in core._environment_reduced_block_cache.values()
                          for a in blocks.values())
        if release_recoupling:
            core._environment_reduced_block_cache.clear()
        pools = []
        for arrays in (aa, bb, ww):
            pools.append(PackedArrayPool.from_arrays(arrays))
            arrays.clear()
        stats.update(route_count=sum(len(group)//4 for group in routes.values()),
                     output_block_count=len(specs),
                     route_index_bytes=sum(len(group)*group.itemsize for group in routes.values()),
                     reduced_cache_array_bytes=cache_bytes,
                     route_pool_bytes=sum(p.data.nbytes+p.offsets.nbytes+
                         p.shape_offsets.nbytes+p.shapes.nbytes for p in pools))
        groups = routes
        del routes, aa, bb, ww, arrays
        blocks = {} if spill_directory is None else _DiskBlocks(spill_directory, disk_bytes)
        owner = moving_environment
        scratch_parent = int(owner.system_stats["n_sites"]) + 1
        scratch_child = scratch_parent + 1

        def batches():
            keys, parents, gather, output = [], set(), 0, 0
            for key, source in groups.items():
                ids = set(source[::4])
                size = int(np.prod(specs[key])) * 8
                parent_size = sum(int(self.offsets[i+1]-self.offsets[i])*8 for i in ids)
                if size > arena_bytes:
                    raise MemoryError('Sparse boundary dense gathering exceeds arena budget: '
                                      f'parent {parent_size} B, output {size} B, limit {arena_bytes} B')
                extra = sum(int(self.offsets[i+1]-self.offsets[i])*8 for i in ids-parents)
                if keys and max(gather+extra, output+size) > arena_bytes:
                    yield keys, sorted(parents), gather, output
                    keys, parents, gather, output = [], set(), 0, 0
                    extra = parent_size
                keys.append(key)
                parents.update(ids)
                gather += extra
                output += size
            if keys:
                yield keys, sorted(parents), gather, output

        def gatherings(keys, outputs):
            chunk, parents, size = [], set(), 0
            for key in keys:
                indices = iter(groups[key])
                for p, a, b, w in zip(indices, indices, indices, indices, strict=True):
                    required = int(self.offsets[p+1]-self.offsets[p])*8
                    if required > arena_bytes:
                        raise MemoryError('Sparse boundary parent block exceeds arena budget: '
                                          f'{required} B required, limit {arena_bytes} B')
                    extra = 0 if p in parents else required
                    if chunk and size+extra > arena_bytes:
                        yield chunk, sorted(parents), size, False
                        chunk, parents, size = [], set(), 0
                        extra = required
                    chunk.append((p, a, b, w, outputs[key]))
                    parents.add(p)
                    size += extra
            if chunk:
                yield chunk, sorted(parents), size, True

        try:
            for revision, (keys, _, _, output_bytes) in enumerate(batches(), 1):
                table, outputs = env._empty_packed_boundary_from_specs(
                    {key: specs[key] for key in keys}, side=self.side, bond=scratch_child)
                labels, topology = env._packed_boundary_labels(table)
                arguments = [arg for p in pools for arg in (p.data, p.offsets, p.shape_offsets, p.shapes)]
                values = None
                for step, (chunk, ids, gather, final) in enumerate(gatherings(keys, outputs)):
                    values = None  # Release the previous view before accumulating.
                    selected_specs = {self.keys[i]: self._specs[self.keys[i]] for i in ids}
                    parent_table, selected = env._empty_packed_boundary_from_specs(
                        selected_specs, side=self.side, bond=scratch_parent)
                    topology_pool = parent_table.block_pool
                    pool = PackedArrayPool(
                        np.empty(int(topology_pool.offsets[-1]), dtype=float),
                        topology_pool.offsets, topology_pool.shape_offsets,
                        topology_pool.shapes)
                    for i in ids:
                        index = selected[self.keys[i]]
                        begin, end = pool.offsets[index:index+2]
                        pool.data[begin:end] = self.array(i).reshape(-1)
                    parent_table = replace(parent_table, block_pool=pool)
                    parent_labels, parent_topology = env._packed_boundary_labels(parent_table)
                    owner.install_boundary(self.side, scratch_parent, pool.data, pool.offsets,
                                           parent_labels, parent_topology, revision)
                    route_array = np.asarray([
                        (selected[self.keys[p]], a, b, w, o) for p, a, b, w, o in chunk], dtype=np.int64)
                    values, _ = owner.advance_boundary(
                        self.side, scratch_parent, scratch_child, route_array, *arguments,
                        table.block_pool.offsets, table.block_pool.shape_offsets, table.block_pool.shapes,
                        labels, topology, revision, accumulate_output=step > 0, finalize_update=final)
                    stats['maximum_gather_bytes'] = max(stats['maximum_gather_bytes'], gather)
                    stats['completed_gathers'] += 1
                    owner.release_boundary(self.side, scratch_parent)
                    del pool, parent_table, route_array
                for key in keys:
                    index = outputs[key]
                    begin, end = table.block_pool.offsets[index:index+2]
                    dense = values[begin:end].reshape(specs[key])
                    tile = 2 if all(d % 2 == 0 for d in dense.shape[1:]) else 1
                    parts = tuple(bsr_matrix(x, blocksize=(tile, tile)) for x in dense)
                    size = sum(p.data.nbytes + p.indices.nbytes + p.indptr.nbytes for p in parts)
                    retained = stats['stored_bytes'] + size if spill_directory is None else size
                    if retained > storage_bytes:
                        raise MemoryError('Sparse boundary retained arrays exceed storage budget: '
                                          f'{retained} B required, limit {storage_bytes} B')
                    blocks[key] = parts
                    stats['stored_bytes'] += size
                    stats['completed_blocks'] += 1
                stats['maximum_output_bytes'] = max(stats['maximum_output_bytes'], output_bytes)
                stats['completed_batches'] += 1
                # BSR parts own their data. Drop both compiled and Python
                # arenas before gathering the next batch.
                owner.release_boundary(self.side, scratch_parent)
                owner.release_boundary(self.side, scratch_child)
                del values, dense, table
        except BaseException:
            if isinstance(blocks, _DiskBlocks):
                blocks.file.close()
            raise
        finally:
            owner.release_boundary(self.side, scratch_parent)
            owner.release_boundary(self.side, scratch_child)
            stats['seconds'] = time.perf_counter() - started
        return type(self)(blocks, side=self.side, bond=bond, decode_bytes=self.decode_bytes)


def iter_right_boundaries(sites, cores, endpoint, *, moving_environment, stride,
                          **budgets):
    """Checkpoint/recompute sparse right blocks in increasing site order.

    Retains O(n/stride + stride) blocks. Budgets apply to each advance, not
    total process memory. Closing the iterator releases retained checkpoints.
    """
    def advance(parent, i):
        child = parent.advance(cores[i], sites[i], moving_environment=moving_environment,
                               **budgets)
        if isinstance(parent.blocks, _DiskBlocks):
            parent.release_decoded()
        return child
    yield from iter_sweep_boundaries(len(sites), endpoint, advance, stride=stride)


def iter_sweep_boundaries(n, endpoint, advance, *, stride=1, direction='lr'):
    """Consume frozen opposite-side checkpoints in either sweep direction.

    ``advance(parent, site)`` contracts the physical site into its parent.
    Traversal order changes, not the physical tensor or sector ordering.
    Retains O(n/stride + stride) blocks without truncation; closing releases
    checkpoints. The caller owns callback-specific decoded-cache lifetimes.
    """
    if int(stride) != stride or stride < 1:
        raise ValueError('stride must be a positive integer')
    if direction not in {'lr', 'rl'}:
        raise ValueError('direction must be lr or rl')
    stride = int(stride)
    order = range(n) if direction == 'lr' else range(n-1, -1, -1)
    blocks = {n-1: endpoint}

    try:
        current = endpoint
        for i in range(n-1, stride-1, -1):
            current = advance(current, order[i])
            if i % stride == 0:
                blocks[i-1] = current
        del current
        for begin in range(0, n, stride):
            end = min(begin+stride, n)-1
            for i in range(end, begin, -1):
                blocks[i-1] = advance(blocks[i], order[i])
            for i in range(begin, end+1):
                yield order[i], blocks.pop(i)
    finally:
        blocks.clear()
