"""Unified locally diabatic representation solver."""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import linear_sum_assignment

from pyqed.dvr import DVR

from . import keo as keo_tools
from . import kinetic as kinetic_tools
from . import overlap as overlap_tools
from .coord import Coord


_LDR_ELECTRONIC_SCANNER = None


def _set_worker_thread_limits(worker_threads):
    if worker_threads is None:
        return
    worker_threads = int(worker_threads)
    if worker_threads < 1:
        raise ValueError("worker_threads must be >= 1")
    value = str(worker_threads)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value


def _init_electronic_worker(electronic, nroots, worker_threads):
    global _LDR_ELECTRONIC_SCANNER
    _set_worker_thread_limits(worker_threads)
    if worker_threads is not None and hasattr(electronic, "direct_ci_workers"):
        electronic.direct_ci_workers = int(worker_threads)
    _LDR_ELECTRONIC_SCANNER = electronic.as_scanner(nstates=int(nroots))


def _scan_electronic_worker(tasks):
    if _LDR_ELECTRONIC_SCANNER is None:
        raise RuntimeError("electronic LDR worker was not initialized")
    output = []
    for index, geometry in tasks:
        if isinstance(geometry, (list, tuple, np.ndarray)):
            geometry = np.asarray(geometry, dtype=float)
        result = _LDR_ELECTRONIC_SCANNER(geometry)
        frame = getattr(result, "frame", None)
        frame = frame() if callable(frame) else result
        energies = np.asarray(getattr(result, "e_tot", None), dtype=float)
        output.append((index, frame, energies))
    return output


def _toeplitz_descriptor(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    column, row = (np.asarray(part) for part in value)
    if column.ndim != 1 or row.ndim != 1 or column.shape != row.shape:
        return None
    return column, row


def _sine_descriptor(value):
    if not isinstance(value, dict):
        return None
    if value.get("kind") != "sine-toeplitz-hankel":
        return None
    column = np.asarray(value.get("column"))
    row = np.asarray(value.get("row"))
    hankel = np.asarray(value.get("hankel"))
    if column.ndim != 1 or row.shape != column.shape:
        return None
    if hankel.shape != (2 * column.size - 1,):
        return None
    return column, row, hankel


def _kinetic_size(value):
    descriptor = _toeplitz_descriptor(value)
    if descriptor is not None:
        return descriptor[0].size
    descriptor = _sine_descriptor(value)
    if descriptor is not None:
        return descriptor[0].size
    shape = np.shape(value)
    return shape[0] if len(shape) == 2 and shape[0] == shape[1] else None


def _kinetic_trace(value):
    descriptor = _toeplitz_descriptor(value)
    if descriptor is not None:
        return descriptor[0].size * descriptor[0][0]
    descriptor = _sine_descriptor(value)
    if descriptor is not None:
        column, _, hankel = descriptor
        indices = np.arange(column.size)
        return column.size * column[0] + np.sum(hankel[2 * indices])
    if sp.issparse(value):
        return np.sum(value.diagonal())
    return np.trace(np.asarray(value))


class LDR:
    """N-dimensional, multi-state LDR dynamics on a product DVR.

    The nuclear coordinates come from a product DVR. With nearest-neighbor
    links, structured axis descriptors drive the prefix-FFT backend without
    constructing the global product-grid kinetic matrix. ``keo`` owns the
    backend-independent nuclear kinetic specification; its default is a lazy
    product-coordinate KEO, while curvilinear calculations can supply
    :func:`pyqed.ldr.keo.podolsky`. Local electronic frames are represented by
    either full overlaps or links. An ab initio calculation can be attached as
    ``LDR(mc, grid=grid, coord=Coord(to_cartesian=geometry, bounds=...),
    states=(...))``;
    calling ``build()`` scans its electronic frames and retains their
    nearest-neighbor overlaps.
    """

    def __init__(
        self,
        source,
        nstates=None,
        *,
        grid=None,
        coord=None,
        states=None,
        keo=None,
        kinetic=None,
        energies=None,
        overlap=None,
        overlaps=None,
        links=None,
        average_paths=False,
        electronic=None,
        scan_provider=None,
        overlap_mode="auto",
        kinetic_backend="auto",
    ):
        if isinstance(source, DVR):
            if grid is not None or coord is not None or states is not None:
                raise ValueError(
                    "grid, coord, and states are only valid when the first "
                    "argument is an electronic driver"
                )
            dvr = source
            coord = Coord(
                bounds=tuple(
                    (float(np.min(axis)), float(np.max(axis)))
                    for axis in dvr.x
                )
            )
            if nstates is None:
                raise TypeError("nstates is required with a precomputed DVR")
            state_indices = None
        else:
            if electronic is not None:
                raise ValueError(
                    "provide the electronic driver as the first argument, not "
                    "also as electronic="
                )
            if nstates is not None:
                raise TypeError(
                    "do not pass nstates with an electronic driver; use states="
                )
            if not isinstance(coord, Coord):
                raise TypeError("coord must be a pyqed.ldr.Coord")
            if coord.to_cartesian is None:
                raise ValueError("electronic LDR coordinates need to_cartesian")
            electronic = source
            if grid is None:
                raise ValueError("an electronic LDR calculation requires grid")
            dvr = grid
            coord.validate_grid(dvr)
            if states is None:
                electronic_nstates = getattr(electronic, "nstates", None)
                if electronic_nstates is None:
                    raise ValueError("states must be provided for an electronic driver")
                state_indices = tuple(range(int(electronic_nstates)))
            else:
                state_indices = tuple(int(state) for state in states)
            if not state_indices or min(state_indices) < 0:
                raise ValueError("states must contain nonnegative electronic-state indices")
            if len(set(state_indices)) != len(state_indices):
                raise ValueError("states must not contain duplicates")
            nstates = len(state_indices)

        self.coord = coord
        self.dvr = dvr
        self.nstates = int(nstates)
        if self.nstates <= 0:
            raise ValueError("nstates must be positive")
        kinetic_backend = str(kinetic_backend).lower().replace("_", "-")
        if kinetic_backend not in {"auto", "generic", "prefix-fft"}:
            raise ValueError(
                "kinetic_backend must be 'auto', 'generic', or 'prefix-fft'"
            )
        if overlap is not None and overlaps is not None:
            raise ValueError("provide overlap or overlaps, not both")
        if overlap is not None and links is not None:
            raise ValueError("provide overlap or links, not both")
        if overlaps is not None and links is not None:
            raise ValueError("provide overlaps or links, not both")
        if overlap is not None:
            overlaps = overlap

        self.shape = dvr.shape
        self.ndim = dvr.ndim
        self.x = dvr.x
        self.points = dvr.points
        self.ngrid = dvr.size
        self.size = self.ngrid * self.nstates
        if keo is not None and kinetic is not None:
            raise ValueError("provide keo or kinetic, not both")
        self.axis_kinetics = None
        self._keo_spec = None
        if keo is None and kinetic is None:
            axis_kinetics = []
            for dvr_axis in dvr.axes:
                axis_descriptor = getattr(dvr_axis, "kinetic_descriptor", None)
                if axis_descriptor is None:
                    axis_descriptor = getattr(dvr_axis, "kinetic_toeplitz", None)
                axis_kinetics.append(
                    axis_descriptor()
                    if axis_descriptor is not None
                    else np.asarray(dvr_axis.t())
                )
            self.axis_kinetics = tuple(axis_kinetics)
            if dvr.ndim == 1:
                kinetic = self.axis_kinetics[0]
            self.keo = keo_tools.product(dvr.axes)
        elif keo is None:
            self.keo = keo_tools.matrix(kinetic)
        else:
            bind = getattr(keo, "bind", None)
            if callable(bind) and getattr(keo, "shape", None) is None:
                self._keo_spec = keo
                self.keo = None
            elif callable(bind):
                keo = bind(
                    coord,
                    grid=dvr,
                    molecule=getattr(electronic, "mol", None),
                )
            if self._keo_spec is None:
                self.keo = keo
                self._validate_keo_shape(keo)
                if isinstance(keo, keo_tools.Matrix):
                    kinetic = keo.matrix
        self.kinetic = kinetic
        if self.kinetic is None and self.axis_kinetics is not None:
            axis_sizes = tuple(map(_kinetic_size, self.axis_kinetics))
            if axis_sizes != self.shape:
                raise ValueError(
                    f"axis kinetic sizes {axis_sizes} != grid shape {self.shape}"
                )
        elif self.kinetic is not None:
            kinetic_size = _kinetic_size(self.kinetic)
            kinetic_shape = (
                (kinetic_size, kinetic_size)
                if kinetic_size is not None
                else np.shape(self.kinetic)
            )
            if kinetic_shape != (self.ngrid, self.ngrid):
                raise ValueError(
                    f"kinetic shape {kinetic_shape} != "
                    f"{(self.ngrid, self.ngrid)}"
                )

        self.average_paths = bool(average_paths)
        overlap_mode = str(overlap_mode).lower().replace("_", "-")
        if overlap_mode not in {"auto", "links", "full", "none"}:
            raise ValueError(
                "overlap_mode must be 'auto', 'links', 'full', or 'none'"
            )
        self.electronic = electronic
        self.scan_provider = scan_provider
        self.state_indices = state_indices
        self.overlap_mode = overlap_mode
        self.kinetic_backend = kinetic_backend
        self.kinetic_info = {"backend": "unbuilt"}
        self._kinetic_operators = {}
        self._kinetic_operator_info = {}
        self.overlaps = None
        self.links = None
        self.energies = energies
        self.electronic_data = None
        self.frames = None
        self.root_indices = None
        self.raw_energies = None
        self.energy_offset = 0.0
        self._electronic_scanner = None
        self.set_overlaps(overlaps=overlaps, links=links)

        self.state = None
        self.states = None
        self.times = None
        self.norm = None
        self.energy = None
        self.history = None
        self.density = None
        self.success = None
        self.message = None

    def _validate_keo_shape(self, keo):
        keo_shape = getattr(keo, "shape", None)
        if keo_shape is None:
            raise TypeError("keo must expose its global nuclear shape")
        if tuple(keo_shape) != (self.ngrid, self.ngrid):
            raise ValueError(
                f"keo shape {tuple(keo_shape)} != {(self.ngrid, self.ngrid)}"
            )

    def _bind_keo(self):
        if self._keo_spec is None:
            return self.keo
        self.keo = self._keo_spec.bind(
            self.coord,
            grid=self.dvr,
            molecule=getattr(self.electronic, "mol", None),
        )
        self._validate_keo_shape(self.keo)
        self._keo_spec = None
        return self.keo

    @staticmethod
    def _electronic_frame(result):
        frame = getattr(result, "frame", None)
        return frame() if callable(frame) else result

    @staticmethod
    def _frame_overlap(left, right):
        overlap = getattr(left, "overlap", None)
        if not callable(overlap):
            raise TypeError(
                "electronic frames must provide overlap(other)"
            )
        return np.asarray(overlap(right), dtype=complex)

    def _point_geometry(self, index, template_mol=None):
        coordinates = np.asarray(
            [self.x[axis][index[axis]] for axis in range(self.ndim)],
            dtype=float,
        )
        value = self.coord.cartesian(coordinates)
        if template_mol is None or not isinstance(value, (list, tuple, np.ndarray)):
            return value
        cartesian = np.asarray(value, dtype=float)
        if cartesian.shape != (template_mol.natom, 3):
            return value
        mol = copy.deepcopy(template_mol)
        mol.set_geom(cartesian)
        mol.build()
        return mol

    def _scan_electronic(
        self,
        *,
        nroots=None,
        n_workers=1,
        worker_threads=1,
        progress=False,
    ):
        n_workers = 1 if n_workers is None else int(n_workers)
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        _set_worker_thread_limits(worker_threads)
        requested = max(self.state_indices) + 1
        existing_energies = getattr(self.electronic, "e_tot", None)
        available = (
            None
            if existing_energies is None
            else int(np.asarray(existing_energies).size)
        )
        if nroots is None:
            nroots = getattr(self.electronic, "nstates", None) or available or requested
        nroots = int(nroots)
        if nroots < requested:
            raise ValueError(
                f"nroots={nroots} does not include requested state "
                f"{max(self.state_indices)}"
            )

        if available is None or available < requested:
            run = getattr(self.electronic, "run", None)
            if not callable(run):
                raise TypeError("electronic driver must provide run() or a solved reference")
            run(nstates=nroots)

        reference = self._electronic_frame(self.electronic)
        as_scanner = getattr(self.electronic, "as_scanner", None)
        if not callable(as_scanner):
            raise TypeError("electronic driver must provide as_scanner()")

        frames = np.empty(self.shape, dtype=object)
        roots = np.empty((*self.shape, self.nstates), dtype=int)
        raw_energies = np.empty((*self.shape, self.nstates), dtype=float)
        indices = tuple(np.ndindex(self.shape))

        def store(index, frame, point_energies, count):
            reference_overlap = abs(self._frame_overlap(reference, frame))
            if reference_overlap.ndim != 2 or reference_overlap.shape[0] < requested:
                raise ValueError(
                    "reference overlap does not contain all requested states"
                )
            weights = reference_overlap[np.asarray(self.state_indices)]
            rows, columns = linear_sum_assignment(-weights)
            if rows.size != self.nstates:
                raise RuntimeError("could not assign every requested electronic state")
            selected = columns[np.argsort(rows)]
            point_energies = np.asarray(point_energies, dtype=float)
            if point_energies.ndim != 1 or point_energies.size <= int(selected.max()):
                raise ValueError("electronic result has incompatible state energies")
            frames[index] = frame
            roots[index] = selected
            raw_energies[index] = point_energies[selected]
            if callable(progress):
                progress(count, self.ngrid, index)
            elif progress:
                print(f"electronic point {count}/{self.ngrid}", flush=True)

        if n_workers == 1:
            if worker_threads is not None and hasattr(self.electronic, "direct_ci_workers"):
                self.electronic.direct_ci_workers = int(worker_threads)
            scanner = as_scanner(nstates=nroots)
            self._electronic_scanner = scanner
            template_mol = getattr(self.electronic, "mol", None)
            for count, index in enumerate(indices, 1):
                result = scanner(
                    self._point_geometry(index, template_mol=template_mol)
                )
                store(
                    index,
                    self._electronic_frame(result),
                    getattr(result, "e_tot", None),
                    count,
                )
        else:
            self._electronic_scanner = None
            tasks = []
            for index in overlap_tools.snake(self.shape):
                coordinates = np.asarray(
                    [self.x[axis][index[axis]] for axis in range(self.ndim)],
                    dtype=float,
                )
                tasks.append((index, self.coord.cartesian(coordinates)))
            worker_count = min(n_workers, self.ngrid)
            base, extra = divmod(len(tasks), worker_count)
            chunks = []
            start = 0
            for worker in range(worker_count):
                stop = start + base + (worker < extra)
                chunks.append(tuple(tasks[start:stop]))
                start = stop
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_electronic_worker,
                initargs=(self.electronic, nroots, worker_threads),
            ) as executor:
                futures = [
                    executor.submit(_scan_electronic_worker, chunk)
                    for chunk in chunks
                ]
                count = 0
                for future in as_completed(futures):
                    for index, frame, point_energies in future.result():
                        count += 1
                        store(index, frame, point_energies, count)

        anchor = tuple(np.argmin(abs(axis)) for axis in self.x)
        self.frames = frames
        self.root_indices = roots
        self.raw_energies = raw_energies
        self.energy_offset = float(raw_energies[anchor][0])
        self.energies = raw_energies - self.energy_offset
        self.electronic_data = {
            "frames": frames,
            "root_indices": roots,
            "raw_energies": raw_energies,
            "energy_offset": self.energy_offset,
        }
        return self

    def scan(
        self,
        *,
        nroots=None,
        n_workers=1,
        worker_threads=1,
        progress=False,
    ):
        """Sample the configured electronic driver on this solver's DVR grid."""

        if self.electronic is None:
            raise RuntimeError("No electronic driver is attached.")
        if self.coord.to_cartesian is not None:
            return self._scan_electronic(
                nroots=nroots,
                n_workers=n_workers,
                worker_threads=worker_threads,
                progress=progress,
            )
        if self.scan_provider is None or not hasattr(self.scan_provider, "_scan_ldr"):
            raise RuntimeError("No nuclear-grid electronic scanner is attached.")

        mode = self.overlap_mode
        scan_mode = "link-only" if mode in {"auto", "links"} else mode
        energies, overlap_data, electronic_data = self.scan_provider._scan_ldr(
            self.electronic,
            overlap_method=scan_mode,
            n_workers=n_workers,
            worker_threads=worker_threads,
        )

        energies = np.asarray(energies)
        expected = (*self.shape, self.nstates)
        if energies.shape != expected:
            raise ValueError(f"scanned energy shape {energies.shape} != {expected}")
        self.energies = energies
        self.electronic_data = electronic_data

        if mode == "none":
            self.set_overlaps()
        elif mode in {"auto", "links"}:
            links = getattr(self.scan_provider, "overlap_links", None)
            if links is None and isinstance(overlap_data, dict):
                links = overlap_data
            if links is None:
                raise RuntimeError("The electronic scan did not produce overlap links.")
            self.set_overlaps(links=links)
        else:
            overlaps = getattr(self.scan_provider, "overlap_matrix", None)
            if overlaps is None and overlap_data is not None:
                overlaps = overlap_data
            if overlaps is None:
                raise RuntimeError("The electronic scan did not produce full overlaps.")
            self.set_overlaps(overlaps=overlaps)
        return self

    def build_links(self):
        """Build nearest-neighbor electronic links from scanned local frames."""

        if self.frames is None or self.root_indices is None:
            raise RuntimeError("scan electronic frames before building links")

        def selected_overlap(left, right):
            value = self._frame_overlap(self.frames[left], self.frames[right])
            return value[np.ix_(self.root_indices[left], self.root_indices[right])]

        links = overlap_tools.nearest(
            self.shape,
            selected_overlap,
        )
        self.set_overlaps(links=links)
        return self

    def build(
        self,
        *,
        nroots=None,
        n_workers=1,
        worker_threads=1,
        progress=False,
    ):
        """Scan an electronic driver and build its nearest-neighbor LDR links."""

        self._bind_keo()
        self.scan(
            nroots=nroots,
            n_workers=n_workers,
            worker_threads=worker_threads,
            progress=progress,
        )
        if self.coord.to_cartesian is not None:
            self.build_links()
        return self

    @classmethod
    def from_domains(
        cls,
        domains,
        npts,
        nstates,
        *,
        mass=None,
        names=None,
        **kwargs,
    ):
        """Construct an LDR solver with a product sine DVR."""

        return cls(
            DVR(domains, npts, mass=mass, names=names),
            nstates,
            **kwargs,
        )

    def set_overlaps(self, *, overlaps=None, links=None):
        """Set one electronic-overlap representation."""

        if overlaps is not None and links is not None:
            raise ValueError("provide full overlaps or links, not both")
        if overlaps is not None:
            overlaps = np.asarray(overlaps, dtype=complex)
            expected = self.ngrid * self.nstates * self.ngrid * self.nstates
            if overlaps.size != expected:
                raise ValueError("overlaps have incompatible grid or state dimensions")
            overlaps = overlaps.reshape(
                self.ngrid,
                self.nstates,
                self.ngrid,
                self.nstates,
            )
        self.overlaps = overlaps
        self.links = links
        self._kinetic_operators.clear()
        self._kinetic_operator_info.clear()
        self.kinetic_info = {"backend": "unbuilt"}
        return self

    def set_diabatic(self, potential, *, representation="full", unitarize=False):
        """Diagonalize a diabatic potential field and construct local overlaps."""

        potential = np.asarray(potential, dtype=complex)
        expected = (*self.shape, self.nstates, self.nstates)
        if potential.shape != expected:
            raise ValueError(f"potential shape {potential.shape} != {expected}")
        energies, frames = np.linalg.eigh(potential)
        self.energies = energies
        self.frames = frames

        if representation == "full":
            self.set_overlaps(overlaps=overlap_tools.from_frames(frames))
        elif representation == "links":
            self.set_overlaps(
                links=overlap_tools.nearest(
                    self.shape,
                    lambda left, right: (
                        frames[left].conj().T @ frames[right]
                    ),
                    unitarize=unitarize,
                )
            )
        else:
            raise ValueError("representation must be 'full' or 'links'")
        return self

    def _energies(self, time=0.0):
        values = self.energies(time) if callable(self.energies) else self.energies
        if values is None:
            return np.zeros((*self.shape, self.nstates), dtype=float)
        values = np.asarray(values)
        expected = (*self.shape, self.nstates)
        if values.shape != expected:
            raise ValueError(f"energies shape {values.shape} != {expected}")
        return values

    def _kinetic_matrix_data(self):
        if self.kinetic is None:
            if self.axis_kinetics is not None:
                return self.dvr.kinetic()
            if self.keo is None and self._keo_spec is not None:
                raise RuntimeError("call LDR.build() to construct the configured KEO")
            to_sparse = getattr(self.keo, "to_sparse", None)
            if callable(to_sparse):
                return to_sparse()
            to_dense = getattr(self.keo, "to_dense", None)
            if callable(to_dense):
                return to_dense()
            raise TypeError("this structured KEO has no direct LDR representation")
        descriptor = _toeplitz_descriptor(self.kinetic)
        if descriptor is not None:
            return scipy.linalg.toeplitz(*descriptor)
        descriptor = _sine_descriptor(self.kinetic)
        if descriptor is not None:
            column, row, hankel = descriptor
            indices = np.arange(column.size)
            return scipy.linalg.toeplitz(column, row) + hankel[
                indices[:, None] + indices[None, :]
            ]
        return self.kinetic

    def kinetic_matrix(self, *, sparse=False):
        """Return the overlap-dressed nuclear kinetic operator."""

        nuclear = self._kinetic_matrix_data()

        if sparse:
            if self.overlaps is not None:
                blocks = self.overlaps.reshape(
                    self.ngrid,
                    self.nstates,
                    self.ngrid,
                    self.nstates,
                )
                return kinetic_tools.dress(
                    nuclear,
                    lambda i, j: blocks[i, :, j, :],
                    nstates=self.nstates,
                    symmetrize=True,
                )
            if self.links is not None:
                return kinetic_tools.linked(
                    nuclear,
                    self.shape,
                    self.links,
                    nstates=self.nstates,
                    average_paths=self.average_paths,
                )
            return sp.kron(
                sp.csr_matrix(nuclear),
                sp.eye(self.nstates, format="csr"),
                format="csr",
            )

        return kinetic_tools.matrix(
            nuclear,
            self.shape,
            self.nstates,
            overlaps=self.overlaps,
            links=self.links,
            average_paths=self.average_paths,
        )

    def kinetic_operator(self, *, backend=None):
        """Return the matrix-free overlap-dressed nuclear kinetic operator."""

        backend = self.kinetic_backend if backend is None else backend
        backend = str(backend).lower().replace("_", "-")
        if backend not in {"auto", "generic", "prefix-fft"}:
            raise ValueError("backend must be 'auto', 'generic', or 'prefix-fft'")
        if backend in self._kinetic_operators:
            self.kinetic_info = dict(self._kinetic_operator_info[backend])
            return self._kinetic_operators[backend]
        if isinstance(self.keo, keo_tools.MPOComponents) and backend == "auto":
            raise NotImplementedError(
                "matrix-free curvilinear KEO propagation uses TNLDR.from_ldr(); "
                "request kinetic_backend='generic' to materialize it explicitly"
            )

        linked = (
            self.links is not None
            and self.overlaps is None
            and not self.average_paths
        )
        eligible_1d = (
            linked
            and self.ndim == 1
            and (self.axis_kinetics is not None or self.kinetic is not None)
        )
        eligible_nd = linked and self.ndim > 1 and self.axis_kinetics is not None
        if backend == "prefix-fft" and not (eligible_1d or eligible_nd):
            raise ValueError(
                "prefix-fft requires a product-grid KEO, nearest-neighbor links, "
                "and linked-product transport"
            )
        if (eligible_1d or eligible_nd) and backend in {"auto", "prefix-fft"}:
            try:
                if eligible_1d:
                    prefix_kinetic = (
                        self.axis_kinetics[0]
                        if self.axis_kinetics is not None
                        else self.kinetic
                    )
                    if sp.issparse(prefix_kinetic):
                        prefix_kinetic = prefix_kinetic.toarray()
                    operator = kinetic_tools.PrefixFFT(prefix_kinetic, self.links)
                else:
                    operator = kinetic_tools.PrefixFFTND(
                        self.axis_kinetics,
                        self.shape,
                        self.links,
                    )
            except (KeyError, ValueError, np.linalg.LinAlgError):
                if backend == "prefix-fft":
                    raise
            else:
                self.kinetic_info = dict(operator.info)
                linear_operator = operator.aslinearoperator()
                self._kinetic_operators[backend] = linear_operator
                self._kinetic_operator_info[backend] = dict(self.kinetic_info)
                return linear_operator

        operator = kinetic_tools.operator(
            self._kinetic_matrix_data(),
            self.shape,
            self.nstates,
            overlaps=self.overlaps,
            links=self.links,
            average_paths=self.average_paths,
        )
        self.kinetic_info = {"backend": "generic"}
        self._kinetic_operators[backend] = operator
        self._kinetic_operator_info[backend] = dict(self.kinetic_info)
        return operator

    def hamiltonian(self, time=0.0, *, sparse=False, matrix_free=False):
        """Return the LDR Hamiltonian at ``time``."""

        diagonal = self._energies(time).reshape(-1)
        if matrix_free:
            kinetic = self.kinetic_operator()

            def matvec(vector):
                vector = np.asarray(vector).reshape(-1)
                return kinetic @ vector + diagonal * vector

            return sla.LinearOperator(
                (self.size, self.size),
                matvec=matvec,
                rmatvec=matvec,
                dtype=np.result_type(kinetic.dtype, diagonal.dtype, complex),
            )
        if sparse:
            return self.kinetic_matrix(sparse=True) + sp.diags(
                diagonal,
                format="csr",
            )
        return self.kinetic_matrix() + np.diag(diagonal)

    def _trace(self, time=0.0):
        if self.kinetic is None and self.axis_kinetics is not None:
            kinetic_trace = sum(
                _kinetic_trace(axis_kinetic) * (self.ngrid // axis_size)
                for axis_kinetic, axis_size in zip(
                    self.axis_kinetics,
                    self.shape,
                )
            )
        elif self.kinetic is not None:
            kinetic_trace = _kinetic_trace(self.kinetic)
        else:
            kinetic_trace = _kinetic_trace(self._kinetic_matrix_data())
        return self.nstates * kinetic_trace + np.sum(self._energies(time))

    @staticmethod
    def _validate_steps(dt, nsteps, nout=1):
        dt = float(dt)
        nsteps = int(nsteps)
        nout = int(nout)
        if dt <= 0.0 or nsteps < 0 or nout <= 0:
            raise ValueError("dt and nout must be positive and nsteps non-negative")
        return dt, nsteps, nout

    def _state_vector(self, state):
        state = np.asarray(state, dtype=complex)
        expected = (*self.shape, self.nstates)
        if state.shape == expected:
            return state.reshape(-1)
        if state.shape == (self.size,):
            return state.copy()
        raise ValueError(f"state shape {state.shape} != {expected} or {(self.size,)}")

    def wavepacket(
        self,
        envelope,
        state=0,
        *,
        anchor=None,
        normalize=True,
        support_threshold=None,
        energy_order=True,
    ):
        """Build a gauge-consistent packet in one energy-ordered adiabatic state."""

        envelope = np.asarray(envelope, dtype=complex)
        if envelope.shape != self.shape:
            raise ValueError(f"envelope shape {envelope.shape} != {self.shape}")
        state = int(state)
        if state < 0 or state >= self.nstates:
            raise ValueError(f"state must lie in [0, {self.nstates})")
        if anchor is None:
            anchor = tuple(np.unravel_index(np.argmax(np.abs(envelope)), self.shape))
        else:
            anchor = tuple(int(index) for index in anchor)

        if energy_order and self.energies is not None:
            labels = np.argsort(self._energies(), axis=-1)[..., state]
        else:
            labels = np.full(self.shape, state, dtype=int)

        support = None
        if support_threshold is not None:
            support_threshold = float(support_threshold)
            if support_threshold < 0.0:
                raise ValueError("support_threshold must be nonnegative")
            scale = float(np.max(np.abs(envelope)))
            if scale == 0.0:
                raise ValueError("wavepacket envelope has zero norm")
            support = np.abs(envelope) > support_threshold * scale
            envelope = np.where(support, envelope, 0.0)

        if self.links is not None:
            gauge = overlap_tools.phase_gauge(
                self.shape,
                self.links,
                state=labels,
                anchor=anchor,
                support=support,
            )
        elif self.overlaps is not None:
            blocks = self.overlaps.reshape(
                self.ngrid,
                self.nstates,
                self.ngrid,
                self.nstates,
            )
            links = {}
            for left in np.ndindex(self.shape):
                left_flat = np.ravel_multi_index(left, self.shape)
                for axis, size in enumerate(self.shape):
                    if left[axis] + 1 >= size:
                        continue
                    right = list(left)
                    right[axis] += 1
                    right = tuple(right)
                    right_flat = np.ravel_multi_index(right, self.shape)
                    links[(axis, left)] = blocks[
                        left_flat, :, right_flat, :
                    ]
            gauge = overlap_tools.phase_gauge(
                self.shape,
                links,
                state=labels,
                anchor=anchor,
                support=support,
            )
        else:
            gauge = np.ones(self.shape, dtype=complex)

        packet = np.zeros((*self.shape, self.nstates), dtype=complex)
        flat_packet = packet.reshape(self.ngrid, self.nstates)
        flat_packet[np.arange(self.ngrid), labels.reshape(-1)] = (
            envelope * gauge
        ).reshape(-1)
        if normalize:
            norm = np.linalg.norm(packet)
            if norm == 0.0:
                raise ValueError("wavepacket envelope has zero norm")
            packet /= norm
        return packet

    def run(
        self,
        state,
        dt,
        nsteps,
        *,
        nout=1,
        t0=0.0,
        matrix_free=True,
        method="expm_multiply",
        absorber=None,
    ):
        r"""Propagate a wavepacket, optionally with $H\to H-iW$."""

        dt, nsteps, nout = self._validate_steps(dt, nsteps, nout)
        vector = self._state_vector(state)
        initial_norm = float(np.vdot(vector, vector).real)
        absorber_diagonal = None
        if absorber is not None:
            absorber = np.asarray(absorber, dtype=float)
            if absorber.shape == self.shape:
                absorber = np.repeat(absorber.reshape(-1), self.nstates)
            elif absorber.shape == (*self.shape, self.nstates):
                absorber = absorber.reshape(-1)
            elif absorber.shape != (self.size,):
                raise ValueError(
                    "absorber must match the nuclear grid or vibronic state"
                )
            if not np.all(np.isfinite(absorber)):
                raise ValueError("absorber values must be finite")
            if np.any(absorber < 0.0):
                raise ValueError("absorber values must be nonnegative")
            absorber_diagonal = absorber

        def effective_hamiltonian(time):
            operator = self.hamiltonian(
                time, matrix_free=matrix_free, sparse=not matrix_free
            )
            if absorber_diagonal is None:
                return operator
            if matrix_free:
                def matvec(value):
                    shape = np.asarray(value).shape
                    value = np.asarray(value).reshape(-1)
                    result = operator @ value - 1j * absorber_diagonal * value
                    return np.asarray(result).reshape(shape)

                def rmatvec(value):
                    shape = np.asarray(value).shape
                    value = np.asarray(value).reshape(-1)
                    result = (
                        operator.rmatvec(value)
                        + 1j * absorber_diagonal * value
                    )
                    return np.asarray(result).reshape(shape)

                return sla.LinearOperator(
                    operator.shape,
                    matvec=matvec,
                    rmatvec=rmatvec,
                    dtype=np.result_type(operator.dtype, complex),
                )
            return operator - 1j * sp.diags(absorber_diagonal, format="csr")

        def effective_trace(time=0.0):
            trace = self._trace(time)
            if absorber_diagonal is not None:
                trace -= 1j * np.sum(absorber_diagonal)
            return trace

        states = [vector.reshape(*self.shape, self.nstates).copy()]
        times = [float(t0)]

        static_interval = (
            method == "expm_multiply"
            and not callable(self.energies)
            and nsteps > 0
            and nsteps % nout == 0
        )
        if static_interval:
            hamiltonian = effective_hamiltonian(float(t0))
            flat_states = sla.expm_multiply(
                -1j * hamiltonian,
                vector,
                start=0.0,
                stop=nsteps * dt,
                num=nsteps // nout + 1,
                endpoint=True,
                traceA=-1j * effective_trace(float(t0)),
            )
            states = [
                value.reshape(*self.shape, self.nstates).copy()
                for value in flat_states
            ]
            times = list(float(t0) + np.arange(len(states)) * nout * dt)
            vector = flat_states[-1]
        else:
            for step in range(1, nsteps + 1):
                midpoint = float(t0) + (step - 0.5) * dt
                if method == "expm_multiply":
                    hamiltonian = effective_hamiltonian(midpoint)
                    vector = sla.expm_multiply(
                        -1j * dt * hamiltonian,
                        vector,
                        traceA=-1j * dt * effective_trace(midpoint),
                    )
                elif method == "expm":
                    hamiltonian = self.hamiltonian(midpoint)
                    if absorber_diagonal is not None:
                        hamiltonian -= 1j * np.diag(absorber_diagonal)
                    vector = scipy.linalg.expm(-1j * dt * hamiltonian) @ vector
                else:
                    raise ValueError("method must be 'expm_multiply' or 'expm'")

                if step % nout == 0 or step == nsteps:
                    states.append(vector.reshape(*self.shape, self.nstates).copy())
                    times.append(float(t0) + step * dt)

        self.state = states[-1]
        self.states = np.asarray(states)
        self.times = np.asarray(times)
        self.norm = np.asarray(
            [np.vdot(value, value).real for value in self.states.reshape(len(states), -1)]
        )
        self.absorbed_probability = initial_norm - self.norm
        self.energy = self.expectation(self.state, time=self.times[-1]).real
        self.success = True
        self.message = "real-time propagation completed"
        return self

    def ground_state(
        self,
        state=None,
        *,
        dt=0.1,
        nsteps=1000,
        tol=1.0e-10,
        matrix_free=True,
    ):
        """Find the ground state by normalized imaginary-time propagation."""

        dt, nsteps, _ = self._validate_steps(dt, nsteps)
        if nsteps == 0:
            raise ValueError("ground_state requires at least one step")
        if tol <= 0.0:
            raise ValueError("tol must be positive")
        if state is None:
            rng = np.random.default_rng(0)
            vector = rng.normal(size=self.size).astype(complex)
        else:
            vector = self._state_vector(state)
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ValueError("initial state has zero norm")
        vector /= norm

        hamiltonian = self.hamiltonian(
            sparse=not matrix_free,
            matrix_free=matrix_free,
        )
        trace = self._trace()
        history = []
        previous = None
        converged = False
        for step in range(1, nsteps + 1):
            vector = sla.expm_multiply(
                -dt * hamiltonian,
                vector,
                traceA=-dt * trace,
            )
            vector /= np.linalg.norm(vector)
            energy = np.vdot(vector, hamiltonian @ vector).real
            history.append((step * dt, energy))
            if previous is not None and abs(energy - previous) < tol:
                converged = True
                break
            previous = energy

        self.state = vector.reshape(*self.shape, self.nstates)
        self.states = self.state[None, ...]
        self.times = np.asarray([step * dt])
        self.norm = np.asarray([1.0])
        self.energy = float(energy)
        self.history = np.asarray(history)
        self.success = converged
        self.message = (
            "imaginary-time propagation converged"
            if converged
            else "imaginary-time propagation reached nsteps"
        )
        return self

    def thermal(self, beta, *, normalize=True):
        r"""Return the canonical density matrix $e^{-\beta H}$ in the LDR basis."""

        beta = float(beta)
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        density = scipy.linalg.expm(-beta * self.hamiltonian())
        if normalize:
            density /= np.trace(density)
        self.density = density.reshape(
            *self.shape,
            self.nstates,
            *self.shape,
            self.nstates,
        )
        return self.density

    def QME(self, density, dt, nsteps, *, nout=1, t0=0.0):
        """Propagate a density matrix with the Liouville-von Neumann equation."""

        dt, nsteps, nout = self._validate_steps(dt, nsteps, nout)
        density = np.asarray(density, dtype=complex).reshape(self.size, self.size)
        densities = [density.copy()]
        times = [float(t0)]
        for step in range(1, nsteps + 1):
            midpoint = float(t0) + (step - 0.5) * dt
            propagator = scipy.linalg.expm(-1j * dt * self.hamiltonian(midpoint))
            density = propagator @ density @ propagator.conj().T
            if step % nout == 0 or step == nsteps:
                densities.append(density.copy())
                times.append(float(t0) + step * dt)
        self.density = np.asarray(densities).reshape(
            len(densities),
            *self.shape,
            self.nstates,
            *self.shape,
            self.nstates,
        )
        self.times = np.asarray(times)
        self.success = True
        self.message = "density-matrix propagation completed"
        return self

    def expectation(self, state, *, time=0.0):
        r"""Return $\langle\Psi|H|\Psi\rangle/\langle\Psi|\Psi\rangle$."""

        vector = self._state_vector(state)
        norm = np.vdot(vector, vector)
        if norm == 0.0:
            raise ValueError("state has zero norm")
        return np.vdot(vector, self.hamiltonian(time, matrix_free=True) @ vector) / norm

    def electronic_density(self, state=None):
        """Return the reduced electronic density matrix."""

        state = self.state if state is None else state
        psi = self._state_vector(state).reshape(self.ngrid, self.nstates)
        return psi.conj().T @ psi

    def nuclear_density(self, state=None):
        """Return the nuclear probability on the product grid."""

        state = self.state if state is None else state
        psi = self._state_vector(state).reshape(self.ngrid, self.nstates)
        return np.sum(np.abs(psi) ** 2, axis=1).reshape(self.shape)
