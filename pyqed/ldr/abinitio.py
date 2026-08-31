"""Ab initio sampling and functional-TT fitting for aligned LDR fields."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from .database import ElectronicDatabase, canonical_json
from .oracle import Frames, ProcrustesOracle


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def _array_fingerprint(value):
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": tuple(int(size) for size in array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


def _protocol_value(value):
    if value is None or isinstance(value, (str, int, float, bool, np.generic)):
        return _jsonable(value)
    if isinstance(value, (tuple, list)):
        return [_protocol_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _protocol_value(item) for key, item in value.items()}
    return repr(value)


def _molecule_identity(electronic):
    molecule = getattr(electronic, "mol", None)
    if molecule is None:
        return {
            "driver": f"{type(electronic).__module__}.{type(electronic).__qualname__}"
        }
    natom = int(getattr(molecule, "natom", 0))
    symbol = getattr(molecule, "atom_symbol", None)
    if callable(symbol):
        symbols = tuple(str(symbol(index)) for index in range(natom))
    else:
        atom_symbols = getattr(molecule, "atom_symbols", ())
        symbols = tuple(map(str, atom_symbols() if callable(atom_symbols) else atom_symbols))
    return {
        "symbols": symbols,
        "charge": int(getattr(molecule, "charge", 0)),
        "spin": int(getattr(molecule, "spin", 0)),
        "basis": _protocol_value(getattr(molecule, "basis", None)),
        "ecp": _protocol_value(getattr(molecule, "ecp", None)),
        "cartesian_ao": bool(getattr(molecule, "cart", False)),
        "symmetry": _protocol_value(getattr(molecule, "groupname", None)),
    }


def _electronic_protocol(electronic, *, nroots):
    molecule = getattr(electronic, "mol", None)
    mean_field = getattr(electronic, "mf", None)
    attributes = (
        "ncas",
        "nelecas",
        "nelecas_spin",
        "ncore",
        "ms2",
        "multiplicity",
        "scan_method",
        "scan_run_kwargs",
        "wfnsym",
        "spin_purification",
        "target_s2",
    )
    method = {
        name: _protocol_value(getattr(electronic, name))
        for name in attributes
        if hasattr(electronic, name)
    }
    orbitals = {
        name: _array_fingerprint(getattr(electronic, name, None))
        for name in ("mo_coeff", "mo_core", "mo_cas")
        if getattr(electronic, name, None) is not None
    }
    return {
        "schema": "pyqed-abinitio-electronic-v2",
        "molecule": _molecule_identity(electronic),
        "driver": f"{type(electronic).__module__}.{type(electronic).__qualname__}",
        "mean_field": (
            None
            if mean_field is None
            else {
                "driver": f"{type(mean_field).__module__}.{type(mean_field).__qualname__}",
                "xc": _protocol_value(getattr(mean_field, "xc", None)),
                "conv_tol": _protocol_value(getattr(mean_field, "conv_tol", None)),
            }
        ),
        "integrals": {
            name: _protocol_value(getattr(molecule, name))
            for name in (
                "builtin_resolved_eri_representation",
                "builtin_resolved_aosym",
                "native_resolved_eri_representation",
                "native_resolved_aosym",
            )
            if molecule is not None and hasattr(molecule, name)
        },
        "method": method,
        "orbitals": orbitals,
        "nroots": int(nroots),
    }


def _automatic_database_path(electronic):
    identity = _molecule_identity(electronic)
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()[:16]
    symbols = identity.get("symbols", ())
    formula = "molecule" if not symbols else "-".join(
        "".join(character for character in symbol if character.isalnum()) or "X"
        for symbol in symbols
    )
    configured = os.environ.get("PYQED_ELECTRONIC_CACHE_DIR")
    if configured is not None:
        root = Path(configured).expanduser()
    elif os.environ.get("XDG_CACHE_HOME"):
        root = Path(os.environ["XDG_CACHE_HOME"]) / "pyqed" / "electronic"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / "pyqed" / "electronic"
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "pyqed" / "electronic"
    else:
        root = Path.home() / ".cache" / "pyqed" / "electronic"
    return root / f"{formula}-{digest}" / "electronic.sqlite"


class _SampleBuilder:
    def __init__(self, grids, geometry, electronic):
        self.grids = grids
        self.geometry = geometry
        self.electronic = electronic

    def __call__(self, index):
        index = tuple(int(value) for value in index)
        coordinates = tuple(
            float(grid[value]) for grid, value in zip(self.grids, index)
        )
        geometry = (
            coordinates if self.geometry is None else self.geometry(coordinates)
        )
        return self.electronic(
            {
                "index": index,
                "coordinates": coordinates,
                "geometry": geometry,
            }
        )


def _electronic_frame(result):
    frame = getattr(result, "frame", None)
    return frame() if callable(frame) else result


def _electronic_frame_record(record):
    return record[0]


def _energy_view(states):
    states = np.asarray(states, dtype=int)

    def selected(record):
        return np.asarray(record[1], dtype=float)[states]

    return selected


def _overlap_view(states):
    states = np.asarray(states, dtype=int)

    def selected(left, right):
        overlap = getattr(left, "overlap", None)
        if not callable(overlap):
            raise TypeError("electronic frames must provide overlap(other)")
        block = np.asarray(overlap(right), dtype=complex)
        return block[np.ix_(states, states)]

    return selected


class _ElectronicStructureBuilder:
    """Lazily sample all requested roots from a native electronic driver."""

    def __init__(self, grids, coord, electronic, nroots=None):
        self.grids = tuple(np.asarray(grid, dtype=float) for grid in grids)
        self.coord = coord
        self.electronic = electronic
        self.nroots = None if nroots is None else int(nroots)
        self._scanner = None

    def _prepare(self):
        if self._scanner is not None:
            return
        energies = getattr(self.electronic, "e_tot", None)
        available = None if energies is None else int(np.asarray(energies).size)
        nroots = self.nroots
        if nroots is None:
            nroots = (
                getattr(self.electronic, "nstates", None)
                or available
                or 1
            )
        nroots = int(nroots)
        if available is None or available < nroots:
            run = getattr(self.electronic, "run", None)
            if not callable(run):
                raise TypeError(
                    "electronic driver must provide run() or a solved reference"
                )
            run(nstates=nroots)
        scanner = getattr(self.electronic, "as_scanner", None)
        if not callable(scanner):
            raise TypeError("electronic driver must provide as_scanner()")
        self._scanner = scanner(nstates=nroots)

    def _geometry(self, coordinates):
        value = self.coord.cartesian(np.asarray(coordinates, dtype=float))
        molecule = getattr(self.electronic, "mol", None)
        if molecule is None or not isinstance(value, (list, tuple, np.ndarray)):
            return value
        cartesian = np.asarray(value, dtype=float)
        if cartesian.shape != (molecule.natom, 3):
            return value
        sample = copy.deepcopy(molecule)
        sample.set_geom(cartesian)
        sample.build()
        return sample

    def __call__(self, index):
        self._prepare()
        index = tuple(int(value) for value in index)
        coordinates = tuple(
            float(grid[value]) for grid, value in zip(self.grids, index)
        )
        result = self._scanner(self._geometry(coordinates))
        frame = _electronic_frame(result)
        energies = np.asarray(getattr(result, "e_tot", None), dtype=float)
        if energies.ndim != 1:
            raise ValueError("electronic result has incompatible state energies")
        if energies.size < int(self.nroots):
            raise ValueError("electronic result returned fewer roots than requested")
        return frame, energies


class AbInitioFit:
    """Sample and fit LDR fields in the anchor-Procrustes P gauge.

    Electronic Hamiltonians are always transformed into the Procrustes gauge
    before interpolation. Raw, nonunitary overlaps are retained as the link
    targets; their polar factors define the gauge but never replace the links.
    """

    def __init__(
        self,
        source,
        nstates=None,
        builder=None,
        *,
        coord=None,
        states=None,
        nroots=None,
        fit_options=None,
        anchor=None,
        frame=None,
        energies=None,
        overlap=None,
        overlap_protocol=None,
        electronic=None,
        cache=None,
        database=None,
        protocol=None,
        geometry=None,
        symmetry=None,
        run_id=None,
        run_metadata=None,
        claim_ttl=7 * 24 * 60 * 60,
        workers=1,
        progress=None,
        energy_shift=None,
    ):
        native_driver = None
        native_states = None
        if coord is not None:
            if nstates is not None or builder is not None:
                raise TypeError(
                    "do not pass nstates or builder with an electronic driver"
                )
            if states is None:
                raise ValueError("states must be provided with an electronic driver")
            if any(
                value is not None
                for value in (frame, energies, overlap, electronic, geometry)
            ):
                raise TypeError(
                    "native AbInitioFit supplies frame, energy, overlap, and geometry"
                )
            native_driver = source
            native_states = tuple(int(state) for state in states)
            if (
                not native_states
                or min(native_states) < 0
                or len(set(native_states)) != len(native_states)
            ):
                raise ValueError("states must contain unique nonnegative indices")
            requested = max(native_states) + 1
            available_energies = getattr(native_driver, "e_tot", None)
            available = (
                None
                if available_energies is None
                else int(np.asarray(available_energies).size)
            )
            if nroots is None:
                nroots = (
                    getattr(native_driver, "nstates", None)
                    or available
                    or requested
                )
            nroots = int(nroots)
            if nroots < requested:
                raise ValueError(
                    f"nroots={nroots} does not include requested state "
                    f"{max(native_states)}"
                )
            requested_degrees = None if fit_options is None else fit_options.get("degrees")
            if requested_degrees is None:
                candidate_shape = (9,) * coord.ndim
            else:
                requested_degrees = (
                    (int(requested_degrees),) * coord.ndim
                    if np.isscalar(requested_degrees)
                    else tuple(int(value) for value in requested_degrees)
                )
                if len(requested_degrees) != coord.ndim:
                    raise ValueError("degrees must contain one value per coordinate")
                candidate_shape = tuple(
                    max(5, degree + 1) for degree in requested_degrees
                )
            candidate_grids = []
            for (lower, upper), count in zip(coord.bounds, candidate_shape):
                nodes = np.cos(np.pi * np.arange(count) / (count - 1))[::-1]
                candidate_grids.append(
                    lower + 0.5 * (nodes + 1.0) * (upper - lower)
                )
            source = tuple(candidate_grids)
            if protocol is None:
                protocol = _electronic_protocol(
                    native_driver,
                    nroots=nroots,
                )
            if database is None:
                database = _automatic_database_path(native_driver)
            if overlap_protocol is None:
                overlap_protocol = {
                    "algorithm": "energy-indexed-native-frame-overlap",
                    "version": 2,
                    "states": native_states,
                }
            builder = _ElectronicStructureBuilder(
                source,
                coord,
                native_driver,
                nroots=nroots,
            )
            nstates = len(native_states)
            frame = _electronic_frame_record
            energies = _energy_view(native_states)
            overlap = _overlap_view(native_states)
            geometry = coord.cartesian
        if states is not None or nroots is not None or fit_options is not None:
            if native_driver is None:
                raise TypeError(
                    "states, nroots, and fit_options require the native electronic API"
                )
        if nstates is None:
            raise TypeError("nstates is required for a callback-based fit")
        self.grids = tuple(
            np.asarray(getattr(grid, "x", grid), dtype=float) for grid in source
        )
        if not self.grids or any(
            grid.ndim != 1 or len(grid) < 3 for grid in self.grids
        ):
            raise ValueError("grids must be one-dimensional arrays of length >= 3")
        self.shape = tuple(len(grid) for grid in self.grids)
        self.nstates = int(nstates)
        if self.nstates < 1:
            raise ValueError("nstates must be positive")
        anchor = (
            tuple(size // 2 for size in self.shape)
            if anchor is None
            else tuple(int(value) for value in anchor)
        )
        if database is not None and protocol is None:
            raise ValueError("a calculation protocol is required with a database")
        self.protocol = protocol
        self.geometry_of = geometry
        self.symmetry = symmetry
        self._symmetry_images = {}
        if symmetry is not None:
            required = ("resolve", "transform_record", "view_key", "metadata")
            missing = [
                name
                for name in required
                if not callable(getattr(symmetry, name, None))
            ]
            if missing:
                raise TypeError(
                    "symmetry is missing: " + ", ".join(missing)
                )
        if builder is not None and electronic is not None:
            raise ValueError("provide builder or electronic, not both")
        self.electronic = electronic
        if electronic is not None:
            builder = _SampleBuilder(self.grids, geometry, electronic)
        self._owns_database = database is not None and not isinstance(
            database, ElectronicDatabase
        )
        self.database = (
            ElectronicDatabase(database) if self._owns_database else database
        )
        if (
            self.database is not None
            and overlap is not None
            and overlap_protocol is None
        ):
            raise ValueError(
                "overlap_protocol is required to persist electronic overlaps"
            )
        self.overlap_protocol = overlap_protocol
        self.run_id = (
            None
            if self.database is None
            else str(run_id or f"abinitio-{uuid.uuid4().hex}")
        )
        self.run_metadata = {} if run_metadata is None else dict(run_metadata)
        if self.database is not None:
            metadata = {
                "driver": type(self).__name__,
                "shape": self.shape,
                "nstates": self.nstates,
                "anchor": anchor,
                "protocol": self.protocol,
                "sampling_symmetry": (
                    None
                    if self.symmetry is None
                    else self.symmetry.metadata()
                ),
                **self.run_metadata,
            }
            self.database.start_run(
                self.run_id, metadata=metadata, status="initialized"
            )
        self.frames = Frames(
            self.shape,
            builder,
            cache_dir=cache,
            database=self.database,
            database_key=(
                None if self.database is None else self._database_key
            ),
            database_metadata=(
                None if self.database is None else self.sample
            ),
            database_run=self.run_id,
            claim_ttl=claim_ttl,
            workers=workers,
            progress=progress,
            representative=(
                None
                if self.symmetry is None
                else self._representative_index
            ),
            transform=(
                None
                if self.symmetry is None
                else self._transform_symmetry_record
            ),
            view_key=(
                None
                if self.symmetry is None
                else self._symmetry_view_key
            ),
        )
        self.anchor = self.frames._index(anchor)
        self.frame_of = frame
        self.energies_of = energies
        self.overlap_of = overlap
        self.requested_energy_shift = energy_shift
        self.oracle = None
        if all(callback is not None for callback in (frame, energies, overlap)):
            self.oracle = ProcrustesOracle(
                self.frames,
                self.anchor,
                frame=frame,
                energies=energies,
                overlap=overlap,
                overlap_protocol=overlap_protocol,
                energy_shift=energy_shift,
            )
        self.energy = None
        self.links = None
        self.feature = None
        self.info = None
        self.config = None
        self.seconds = None
        self.success = False
        self.message = "not fitted"
        self.paths = None
        self.labels = None
        self.metadata = None
        self.fit_options = (
            None if native_driver is None
            else {} if fit_options is None
            else dict(fit_options)
        )
        self._native = native_driver is not None
        if self._native:
            self.electronic_driver = native_driver
            self.states = native_states
            self.coord = coord
            self.database_path = Path(self.database.path)

    @classmethod
    def from_electronic(
        cls,
        electronic,
        *,
        coord,
        states,
        nroots=None,
        **kwargs,
    ):
        """Create a lazy native sampler on an internal candidate grid."""
        return cls(
            electronic,
            coord=coord,
            states=states,
            nroots=nroots,
            **kwargs,
        )

    def adaptive_plan(self, *, degrees=4):
        """Return the internal seed design and adaptive safety budget."""
        from .ttfit import coordinate_fiber_points

        if degrees is None:
            degrees = (4,) * len(self.shape)
        elif np.isscalar(degrees):
            degrees = (int(degrees),) * len(self.shape)
        else:
            degrees = tuple(int(degree) for degree in degrees)
        if len(degrees) != len(self.shape):
            raise ValueError("degrees must contain one value per coordinate")
        points_per_axis = tuple(
            min(size, degree + 1)
            for size, degree in zip(self.shape, degrees)
        )
        points = coordinate_fiber_points(
            self.shape,
            self.anchor,
            points_per_axis=points_per_axis,
        )
        total = int(np.prod(self.shape))
        remaining = max(1, total - len(points))
        batch_size = min(
            remaining,
            32,
            max(8, 2 * int(np.ceil(np.sqrt(total)))),
        )
        return {
            "points": points,
            "target_points": total,
            "batch_size": batch_size,
        }

    def run_adaptive(self, **fit_options):
        """Fit synchronized fields with the internal adaptive sampling policy."""
        reserved = {
            "representation",
            "points",
            "pairs",
            "adaptive_count",
            "adaptive_batch",
            "adaptive_pool",
        }
        supplied = reserved.intersection(fit_options)
        if supplied:
            names = ", ".join(sorted(supplied))
            raise TypeError(
                f"adaptive sampling is internal; remove these fit options: {names}"
            )
        plan = self.adaptive_plan(degrees=fit_options.get("degrees"))
        fit_options.setdefault(
            "feature_rank",
            min(32, 4 * self.nstates * len(self.shape)),
        )
        fit_options.setdefault("feature_strategy", "nystrom")
        fit_options.setdefault("adaptive_energy_atol", 1.0e-4)
        fit_options.setdefault("adaptive_link_rtol", 5.0e-3)
        fit_options.setdefault("adaptive_patience", 2)
        fit_options.setdefault("adaptive_minimum_rounds", 2)
        return self.run(
            representation="adaptive-sync",
            points=plan["points"],
            adaptive_count=plan["target_points"],
            adaptive_batch=plan["batch_size"],
            **fit_options,
        )

    def build(self):
        """Run the native adaptive electronic fit."""
        if self.success:
            return self
        if not self._native:
            raise RuntimeError("callback-based fits use run() with an explicit design")
        options = dict(self.fit_options)
        options.setdefault("degrees", min(6, *(len(axis) - 1 for axis in self.grids)))
        plan = self.adaptive_plan(degrees=options["degrees"])
        progress = self.frames.progress
        if isinstance(progress, (bool, np.bool_)):
            if progress:
                total = plan["target_points"]

                def progress(_index, stats):
                    print(f"electronic point {stats['built']}/{total}", flush=True)
            else:
                progress = None
        self.frames.progress = progress
        with self:
            self.run_adaptive(**options)
        return self

    def direct_product(
        self,
        grid,
        *,
        keo,
        workers=1,
        progress=None,
        energy_shift=None,
    ):
        """Build a database-backed direct-product LDR reference on ``grid``."""
        if not self._native:
            raise RuntimeError("direct-product sampling requires a native electronic driver")
        from pyqed.dvr import DVR
        from pyqed.ldr.core import LDR

        if not isinstance(grid, DVR):
            raise TypeError("grid must be a pyqed.dvr.DVR product grid")
        self.coord.validate_grid(grid)
        builder = _ElectronicStructureBuilder(
            grid.x,
            self.coord,
            self.electronic_driver,
            nroots=self.protocol["nroots"],
        )
        energy_shift = self.energy_shift if energy_shift is None else float(energy_shift)
        reference = type(self)(
            grid.x,
            len(self.states),
            builder,
            frame=_electronic_frame_record,
            energies=_energy_view(self.states),
            overlap=_overlap_view(self.states),
            overlap_protocol=self.overlap_protocol,
            geometry=self.coord.cartesian,
            database=self.database_path,
            protocol=self.protocol,
            workers=workers,
            progress=None,
            energy_shift=energy_shift,
        )
        points = tuple(np.ndindex(grid.shape))
        specification = keo
        bind = getattr(specification, "bind", None)
        if callable(bind) and getattr(specification, "shape", None) is None:
            specification = bind(
                self.coord,
                grid=grid,
                molecule=getattr(self.electronic_driver, "mol", None),
            )
        pairs = []
        pair_keys = []
        for left in points:
            for axis, size in enumerate(grid.shape):
                if left[axis] + 1 >= size:
                    continue
                right = list(left)
                right[axis] += 1
                right = tuple(right)
                pairs.append((left, right))
                pair_keys.append((axis, left))
        if progress:
            total = len(points)

            def progress(_index, stats):
                print(f"direct electronic point {stats['built']}/{total}", flush=True)

        reference.frames.progress = progress if callable(progress) else None
        with reference:
            records = reference.frames.get_many(points)
            blocks = reference.oracle.raw_overlap_many(pairs)
            gauges = np.asarray(reference.oracle.gauges(points))
        energies = np.asarray(
            [reference.energies_of(record) for record in records], dtype=float
        ).reshape(*grid.shape, len(self.states))
        if energy_shift is None:
            energy_shift = float(np.min(energies[reference.anchor]))
        energies -= float(energy_shift)
        links = dict(zip(pair_keys, blocks))
        solver = LDR(
            grid,
            len(self.states),
            keo=specification,
            energies=energies,
            links=links,
            kinetic_backend="generic",
        )
        solver.database_path = self.database_path
        solver.database_info = reference.stats["database"]
        solver.procrustes_gauges = gauges.reshape(
            *grid.shape, len(self.states), len(self.states)
        )
        solver.direct_product_info = {
            "shape": tuple(grid.shape),
            "geometries": len(points),
            "overlap_pairs": len(pairs),
            "overlap_representation": "nearest-link-lpa",
            "action": "linked-product-approximation",
            "gauge": "anchor-procrustes",
            "database_hits": int(reference.frames.stats["database_hits"]),
            "database_writes": int(reference.frames.stats["database_writes"]),
        }
        return solver

    def refine_hamiltonian(
        self,
        coordinates,
        hamiltonians,
        *,
        degrees,
        rank,
        sweeps=30,
        rtol=1.0e-10,
        regularization=1.0e-10,
        seed=0,
    ):
        """Refine the full Procrustes-gauged Hermitian matrix field."""
        from pyqed.mps.functional import FunctionalTT

        if not self.success or self.energy is None:
            raise RuntimeError("a completed matrix-valued fit is required")
        coordinates = np.asarray(coordinates, dtype=float)
        hamiltonians = np.asarray(hamiltonians, dtype=complex)
        if coordinates.ndim != 2 or coordinates.shape[1] != len(self.grids):
            raise ValueError("coordinates have the wrong dimension")
        expected = (len(coordinates), self.nstates, self.nstates)
        if hamiltonians.shape != expected:
            raise ValueError(f"hamiltonians shape {hamiltonians.shape} != {expected}")
        if not np.allclose(
            hamiltonians,
            hamiltonians.conj().swapaxes(-1, -2),
            atol=1.0e-11,
            rtol=1.0e-11,
        ):
            raise ValueError("hamiltonian refinement values must be Hermitian")

        if not hasattr(self, "_hamiltonian_base_coordinates"):
            mesh = np.meshgrid(*self.grids, indexing="ij")
            self._hamiltonian_base_coordinates = np.stack(
                [values.reshape(-1) for values in mesh], axis=1
            )
            if self.oracle is None:
                self._hamiltonian_base_values = np.asarray(
                    self.energy.predict(self._hamiltonian_base_coordinates)
                )
            else:
                points = tuple(np.ndindex(self.shape))
                self._hamiltonian_base_values = np.asarray(
                    self.oracle.hamiltonian_many(points)
                )
            self._hamiltonian_refinement_coordinates = np.empty(
                (0, len(self.grids)), dtype=float
            )
            self._hamiltonian_refinement_values = np.empty(
                (0, self.nstates, self.nstates), dtype=complex
            )
        if len(coordinates):
            self._hamiltonian_refinement_coordinates = np.concatenate(
                (self._hamiltonian_refinement_coordinates, coordinates)
            )
            self._hamiltonian_refinement_values = np.concatenate(
                (self._hamiltonian_refinement_values, hamiltonians)
            )
        training_coordinates = np.concatenate(
            (
                self._hamiltonian_base_coordinates,
                self._hamiltonian_refinement_coordinates,
            )
        )
        training_values = np.concatenate(
            (
                self._hamiltonian_base_values,
                self._hamiltonian_refinement_values,
            )
        )
        degrees = (
            (int(degrees),) * len(self.grids)
            if np.isscalar(degrees)
            else tuple(int(value) for value in degrees)
        )
        if len(degrees) != len(self.grids):
            raise ValueError("degrees must contain one value per coordinate")
        bounds = tuple(
            (float(grid[0]), float(grid[-1])) for grid in self.grids
        )
        refined = FunctionalTT(
            degrees=degrees,
            rank=int(rank),
            bounds=bounds,
            normalization="frobenius",
            hermitian=True,
            regularization=float(regularization),
            sweeps=int(sweeps),
            rtol=float(rtol),
            random_state=int(seed),
        ).fit(training_coordinates, training_values)
        predicted = np.asarray(
            refined.predict(self._hamiltonian_refinement_coordinates)
        )
        scale = max(
            float(np.linalg.norm(self._hamiltonian_refinement_values)),
            np.finfo(float).tiny,
        )
        self.energy = refined
        self.hamiltonian_refinement = {
            "representation": "full-procrustes-gauged-hermitian-matrix",
            "base_points": len(self._hamiltonian_base_coordinates),
            "refinement_points": len(self._hamiltonian_refinement_coordinates),
            "latest_refinement_points": len(coordinates),
            "degrees": degrees,
            "rank": int(rank),
            "ranks": tuple(refined.ranks_),
            "relative_refinement_error": float(
                np.linalg.norm(
                    predicted - self._hamiltonian_refinement_values
                ) / scale
            ),
        }
        return self

    def continuous_fields(self, coordinates, pairs=()):
        """Sample gauged Hamiltonians and graph links at scattered coordinates."""

        if not self._native:
            raise RuntimeError(
                "continuous sampling requires a native electronic driver"
            )
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != len(self.grids):
            raise ValueError("coordinates have the wrong dimension")
        if len(coordinates) < 2:
            raise ValueError("continuous sampling requires at least two points")
        pairs = np.asarray(pairs, dtype=int)
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (npairs, 2)")
        if len(pairs) and (np.min(pairs) < 0 or np.max(pairs) >= len(coordinates)):
            raise IndexError("a continuous-field pair is outside the sample set")
        if self.energy_shift is None:
            self.oracle.hamiltonian_many((self.anchor,))
        anchor_coordinates = np.asarray(self.coordinates(self.anchor), dtype=float)
        source = tuple(
            np.concatenate(([anchor_coordinates[axis]], coordinates[:, axis]))
            for axis in range(len(self.grids))
        )
        builder = _ElectronicStructureBuilder(
            source,
            self.coord,
            self.electronic_driver,
            nroots=self.protocol["nroots"],
        )
        reference = type(self)(
            source,
            self.nstates,
            builder,
            anchor=(0,) * len(self.grids),
            frame=_electronic_frame_record,
            energies=_energy_view(self.states),
            overlap=_overlap_view(self.states),
            overlap_protocol=self.overlap_protocol,
            geometry=self.coord.cartesian,
            database=self.database,
            protocol=self.protocol,
            workers=self.frames.workers,
            progress=None,
            energy_shift=self.energy_shift,
        )
        points = tuple(
            (number,) * len(self.grids)
            for number in range(1, len(coordinates) + 1)
        )
        mapped_pairs = [
            (points[int(left)], points[int(right)]) for left, right in pairs
        ]
        with reference:
            hamiltonians = reference.oracle.hamiltonian_many(points)
            links = (
                reference.oracle.overlap_many(mapped_pairs)
                if mapped_pairs
                else np.empty((0, self.nstates, self.nstates), dtype=complex)
            )
        return {
            "coordinates": coordinates,
            "hamiltonians": np.asarray(hamiltonians),
            "pairs": pairs,
            "links": np.asarray(links),
            "stats": dict(reference.frames.stats),
        }

    def coordinates(self, index):
        """Return generalized coordinates for one product-grid index."""

        index = tuple(int(value) for value in index)
        if len(index) != len(self.shape) or any(
            value < 0 or value >= size
            for value, size in zip(index, self.shape)
        ):
            raise IndexError(f"grid index {index} is outside {self.shape}")
        return tuple(float(grid[value]) for grid, value in zip(self.grids, index))

    def sample_geometry(self, index):
        """Return the physical geometry requested at one grid index."""

        coordinates = self.coordinates(index)
        return coordinates if self.geometry_of is None else self.geometry_of(coordinates)

    def _symmetry_image(self, index):
        index = tuple(int(value) for value in index)
        image = self._symmetry_images.get(index)
        if image is None:
            image = self.symmetry.resolve(self.coordinates(index))
            if len(image.representative_coordinates) != len(self.grids):
                raise ValueError(
                    "sampling-symmetry representative has the wrong coordinate dimension"
                )
            self._symmetry_images[index] = image
        return image

    def _index_for_coordinates(self, coordinates):
        index = []
        tolerance = max(
            float(getattr(self.symmetry, "tolerance", 0.0)), 1.0e-12
        )
        for axis, (grid, value) in enumerate(zip(self.grids, coordinates)):
            matches = np.flatnonzero(
                np.isclose(grid, float(value), atol=tolerance, rtol=1.0e-12)
            )
            if matches.size == 0:
                raise ValueError(
                    f"sampling-symmetry representative coordinate {value:.16g} is "
                    f"not present on grid axis {axis}"
                )
            nearest = matches[np.argmin(np.abs(grid[matches] - float(value)))]
            index.append(int(nearest))
        return tuple(index)

    def _representative_index(self, index):
        return self._index_for_coordinates(
            self._symmetry_image(index).representative_coordinates
        )

    def representative_coordinates(self, index):
        """Return the canonical generalized coordinates for one orbit image."""

        if self.symmetry is None:
            return self.coordinates(index)
        return tuple(self._symmetry_image(index).representative_coordinates)

    def representative_geometry(self, index):
        """Return the canonical geometry stored in the electronic database."""

        coordinates = self.representative_coordinates(index)
        return coordinates if self.geometry_of is None else self.geometry_of(coordinates)

    def expand_points(self, points):
        """Expand explicit grid samples to complete molecular-symmetry orbits."""

        points = tuple(dict.fromkeys(self.frames._index(point) for point in points))
        if self.symmetry is None:
            return points
        expanded = []
        for point in points:
            for coordinates in self.symmetry.images(self.coordinates(point)):
                expanded.append(self._index_for_coordinates(coordinates))
        return tuple(dict.fromkeys(expanded))

    def expand_pairs(self, pairs):
        """Expand explicit links by applying each symmetry operation jointly."""

        normalized = tuple(
            (self.frames._index(left), self.frames._index(right))
            for left, right in pairs
        )
        if self.symmetry is None:
            return tuple(dict.fromkeys(normalized))
        expanded = []
        for left, right in normalized:
            for left_coordinates, right_coordinates in self.symmetry.pair_images(
                self.coordinates(left), self.coordinates(right)
            ):
                expanded.append(
                    (
                        self._index_for_coordinates(left_coordinates),
                        self._index_for_coordinates(right_coordinates),
                    )
                )
        return tuple(dict.fromkeys(expanded))

    def _transform_symmetry_record(self, record, representative, index):
        return self.symmetry.transform_record(
            record,
            self._symmetry_image(index),
            representative_geometry=self.representative_geometry(representative),
            requested_geometry=self.sample_geometry(index),
            protocol=self.protocol,
        )

    def _symmetry_view_key(self, index):
        return self.symmetry.view_key(self._symmetry_image(index))

    def sample(self, index):
        """Describe one grid point for provenance and database queries."""

        index = tuple(int(value) for value in index)
        sample = {
            "index": index,
            "coordinates": self.coordinates(index),
            "geometry": self.sample_geometry(index),
        }
        if self.symmetry is not None:
            image = self._symmetry_image(index)
            sample.update(
                {
                    "representative_index": self._representative_index(index),
                    "representative_coordinates": image.representative_coordinates,
                    "representative_geometry": self.representative_geometry(index),
                    "sampling_symmetry": self.symmetry.view_key(image),
                }
            )
        source = self.frames.sources.get(index)
        if source is not None:
            sample["source"] = source
        if self.database is not None:
            sample["record_id"] = self.database.identifier(
                self._database_key(index)
            )
        return sample

    def _database_key(self, index):
        return {
            "geometry": self.representative_geometry(index),
            "protocol": self.protocol,
        }

    @property
    def energy_shift(self):
        if self.oracle is not None:
            return self.oracle.energy_shift
        return self.requested_energy_shift

    @property
    def stats(self):
        stats = {
            "success": bool(self.success),
            "message": self.message,
            "seconds": self.seconds,
            "frames": self.frames.stats,
            "fit": self.info,
        }
        if self.database is not None:
            stats["database"] = self.database.stats
            stats["run_id"] = self.run_id
        return stats

    def run(
        self,
        *,
        sampler="cross",
        rank=16,
        energy_rank=None,
        link_rank=None,
        degrees=6,
        sweeps=8,
        rtol=1.0e-8,
        validation=128,
        seed=0,
        start_rank=1,
        kick_rank=2,
        fit_sweeps=12,
        initial=32,
        rounds=6,
        sparse_sequence="halton",
        cur_axis=-2,
        cur_slabs=4,
        cur_probes=None,
        cur_rcond=1.0e-10,
        regularization=1.0e-10,
        representation="links",
        feature_rank=None,
        feature_penalty=10.0,
        feature_smoothness=0.0,
        feature_maxiter=500,
        variational_maxiter=500,
        feature_strategy="synchronized",
        points=None,
        pairs=None,
        neighbors=4,
        adaptive_count=None,
        adaptive_batch=8,
        adaptive_pool=4096,
        adaptive_importance=None,
        adaptive_importance_floor=0.1,
        adaptive_energy_atol=None,
        adaptive_link_rtol=None,
        adaptive_patience=1,
        adaptive_minimum_rounds=1,
    ):
        """Sample and fit aligned energy and electronic-transport fields."""
        if self.oracle is None:
            raise RuntimeError(
                "frame, energies, and overlap callbacks are required for fitting"
            )
        from .ttfit import (
            fit_aligned,
            fit_adaptive_sync,
            fit_block_cross,
            fit_cur,
            fit_energy_features,
            fit_sync,
            fit_sparse,
            fit_variational,
        )

        sampler = str(sampler).lower().replace("_", "-")
        if sampler not in {"cross", "block-cross", "sparse", "cur"}:
            raise ValueError(
                "sampler must be 'cross', 'block-cross', 'sparse', or 'cur'"
            )
        representation = str(representation).lower().replace("_", "-")
        if representation not in {
            "links", "features", "sync", "adaptive-sync", "variational"
        }:
            raise ValueError(
                "representation must be 'links', 'features', 'sync', "
                "'adaptive-sync', or 'variational'"
            )
        if representation == "features" and sampler != "cross":
            raise ValueError("feature representation currently requires sampler='cross'")
        if representation in {"sync", "adaptive-sync"} and points is None:
            raise ValueError(f"{representation} representation requires sampled points")
        if representation == "adaptive-sync" and adaptive_count is None:
            raise ValueError("adaptive-sync representation requires adaptive_count")
        if representation == "variational" and pairs is None:
            raise ValueError("variational representation requires sampled pairs")
        requested_points = points
        requested_pairs = pairs
        if points is not None:
            points = self.expand_points((*points, self.anchor))
        if pairs is not None:
            pairs = self.expand_pairs(pairs)
        started = time.perf_counter()
        self.success = False
        self.message = "fitting"
        if self.database is not None:
            self.database.update_run(self.run_id, "fitting")
        try:
            if representation == "variational":
                self.energy, self.feature, self.info = fit_variational(
                    self.oracle,
                    self.grids,
                    self.nstates,
                    pairs,
                    max_rank=rank,
                    feature_rank=feature_rank,
                    degrees=degrees,
                    sweeps=fit_sweeps,
                    rtol=rtol,
                    regularization=regularization,
                    penalty=feature_penalty,
                    smoothness=feature_smoothness,
                    maxiter=variational_maxiter,
                    collocation=int(np.prod(self.shape)),
                    seed=seed,
                )
                self.links = None
            elif representation == "adaptive-sync":
                self.energy, self.feature, self.info = fit_adaptive_sync(
                    self.oracle,
                    self.grids,
                    self.nstates,
                    points,
                    target_points=adaptive_count,
                    batch_size=adaptive_batch,
                    candidate_pool=adaptive_pool,
                    importance=adaptive_importance,
                    importance_floor=adaptive_importance_floor,
                    energy_atol=adaptive_energy_atol,
                    link_rtol=adaptive_link_rtol,
                    patience=adaptive_patience,
                    minimum_rounds=adaptive_minimum_rounds,
                    anchor=self.anchor,
                    max_rank=rank,
                    feature_rank=feature_rank,
                    neighbors=neighbors,
                    degrees=degrees,
                    sweeps=fit_sweeps,
                    rtol=rtol,
                    regularization=regularization,
                    feature_penalty=feature_penalty,
                    feature_smoothness=feature_smoothness,
                    feature_maxiter=feature_maxiter,
                    variational_maxiter=variational_maxiter,
                    feature_strategy=feature_strategy,
                    seed=seed,
                    point_expander=self.expand_points,
                )
                self.links = None
            elif representation == "sync":
                self.energy, self.feature, self.info = fit_sync(
                    self.oracle,
                    self.grids,
                    self.nstates,
                    points,
                    anchor=self.anchor,
                    pairs=pairs,
                    max_rank=rank,
                    feature_rank=feature_rank,
                    neighbors=neighbors,
                    degrees=degrees,
                    sweeps=fit_sweeps,
                    rtol=rtol,
                    regularization=regularization,
                    feature_penalty=feature_penalty,
                    feature_smoothness=feature_smoothness,
                    feature_maxiter=feature_maxiter,
                    variational_maxiter=variational_maxiter,
                    feature_strategy=feature_strategy,
                    seed=seed,
                )
                self.links = None
            elif representation == "features":
                self.energy, self.feature, self.info = fit_energy_features(
                    self.oracle,
                    self.grids,
                    self.nstates,
                    self.anchor,
                    max_rank=rank,
                    energy_rank=energy_rank,
                    feature_rank=feature_rank,
                    feature_penalty=feature_penalty,
                    feature_smoothness=feature_smoothness,
                    feature_maxiter=feature_maxiter,
                    degrees=degrees,
                    sweeps=sweeps,
                    rtol=rtol,
                    validation=validation,
                    seed=seed,
                    start_rank=start_rank,
                    kick_rank=kick_rank,
                )
                self.links = None
            elif sampler == "cur":
                self.energy, self.links, self.info = fit_cur(
                    self.oracle,
                    self.grids,
                    self.nstates,
                    rank=rank,
                    energy_rank=energy_rank,
                    link_rank=link_rank,
                    degrees=degrees,
                    axis=cur_axis,
                    slabs=cur_slabs,
                    probes=cur_probes,
                    seed=seed,
                    rcond=cur_rcond,
                )
            elif sampler == "cross":
                self.energy, self.links, self.info = fit_aligned(
                    self.oracle,
                    self.grids,
                    self.nstates,
                    max_rank=rank,
                    energy_rank=energy_rank,
                    link_rank=link_rank,
                    degrees=degrees,
                    sweeps=sweeps,
                    rtol=rtol,
                    validation=validation,
                    seed=seed,
                    start_rank=start_rank,
                    kick_rank=kick_rank,
                )
            elif sampler == "block-cross":
                self.energy, self.links, self.info = fit_block_cross(
                    self.oracle,
                    self.grids,
                    self.nstates,
                    rank=rank,
                    degrees=degrees,
                    sweeps=sweeps,
                    rtol=rtol,
                    validation=validation,
                    seed=seed,
                    start_rank=start_rank,
                    kick_rank=kick_rank,
                )
            elif sampler == "sparse":
                self.energy, self.links, self.info = fit_sparse(
                    self.oracle,
                    self.grids,
                    self.nstates,
                    rank=rank,
                    energy_rank=energy_rank,
                    link_rank=link_rank,
                    degrees=degrees,
                    initial=initial,
                    validation=validation,
                    rounds=rounds,
                    rtol=rtol,
                    sweeps=sweeps,
                    seed=seed,
                    regularization=regularization,
                    sequence=sparse_sequence,
                )
        except Exception as error:
            self.message = str(error)
            if self.database is not None:
                self.database.update_run(self.run_id, "failed")
            raise
        else:
            self.success = True
            self.message = "fitted"
            if self.database is not None:
                self.database.update_run(self.run_id, "fitted")
        finally:
            self.seconds = time.perf_counter() - started
        self.config = {
            "gauge": "anchor-procrustes",
            "unitarize_links": False,
            "sampler": sampler,
            "rank": int(rank),
            "energy_rank": None if energy_rank is None else int(energy_rank),
            "link_rank": None if link_rank is None else int(link_rank),
            "degrees": degrees,
            "sweeps": int(sweeps),
            "rtol": float(rtol),
            "validation": int(validation),
            "seed": int(seed),
            "start_rank": int(start_rank),
            "kick_rank": int(kick_rank),
            "fit_sweeps": int(fit_sweeps),
            "initial": int(initial),
            "rounds": int(rounds),
            "sparse_sequence": str(sparse_sequence),
            "cur_axis": int(cur_axis),
            "cur_slabs": int(cur_slabs),
            "cur_probes": None if cur_probes is None else int(cur_probes),
            "cur_rcond": float(cur_rcond),
            "regularization": float(regularization),
            "representation": representation,
            "feature_strategy": str(feature_strategy),
            "feature_rank": None if feature_rank is None else int(feature_rank),
            "feature_penalty": float(feature_penalty),
            "feature_smoothness": float(feature_smoothness),
            "feature_maxiter": int(feature_maxiter),
            "variational_maxiter": int(variational_maxiter),
            "points": (
                None
                if points is None
                else [list(map(int, point)) for point in points]
            ),
            "requested_points": (
                None
                if requested_points is None
                else [list(map(int, point)) for point in requested_points]
            ),
            "pairs": (
                None
                if pairs is None
                else [[list(map(int, left)), list(map(int, right))] for left, right in pairs]
            ),
            "requested_pairs": (
                None
                if requested_pairs is None
                else [
                    [list(map(int, left)), list(map(int, right))]
                    for left, right in requested_pairs
                ]
            ),
            "neighbors": int(neighbors),
            "adaptive_count": (
                None if adaptive_count is None else int(adaptive_count)
            ),
            "adaptive_batch": int(adaptive_batch),
            "adaptive_pool": int(adaptive_pool),
            "adaptive_importance_weighted": adaptive_importance is not None,
            "adaptive_importance_floor": float(adaptive_importance_floor),
            "adaptive_energy_atol": (
                None
                if adaptive_energy_atol is None
                else float(adaptive_energy_atol)
            ),
            "adaptive_link_rtol": (
                None
                if adaptive_link_rtol is None
                else float(adaptive_link_rtol)
            ),
            "adaptive_patience": int(adaptive_patience),
            "adaptive_minimum_rounds": int(adaptive_minimum_rounds),
        }
        return self

    def save(self, directory, *, labels=None, metadata=None):
        """Persist fitted fields, grids, and sampling metadata."""
        if not self.success or self.energy is None or (
            self.links is None and self.feature is None
        ):
            raise RuntimeError("run must complete before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        labels = (
            tuple(f"q{axis}" for axis in range(len(self.grids)))
            if labels is None
            else tuple(str(label) for label in labels)
        )
        if len(labels) != len(self.grids) or len(set(labels)) != len(labels):
            raise ValueError("labels must be unique and match the grid count")
        energy_path = directory / "ebar.npz"
        self.energy.save(energy_path)
        link_paths = []
        feature_path = None
        if self.links is not None:
            for label, model in zip(labels, self.links):
                path = directory / f"bar_l_{label}.npz"
                model.save(path)
                link_paths.append(path)
        else:
            feature_path = directory / "y.npz"
            self.feature.save(feature_path)
        grid_path = directory / "grids.npz"
        np.savez(
            grid_path,
            **{f"grid_{axis}": grid for axis, grid in enumerate(self.grids)},
        )
        samples_path = directory / "samples.json"
        samples = [self.sample(index) for index in sorted(self.frames.points)]
        samples_path.write_text(json.dumps(_jsonable(samples), indent=2) + "\n")
        self.labels = labels
        self.metadata = {} if metadata is None else metadata
        summary = {
            "class": type(self).__name__,
            "grid": list(self.shape),
            "nstates": self.nstates,
            "anchor": list(self.anchor),
            "energy_shift": self.energy_shift,
            "labels": list(labels),
            "energy_model": energy_path.name,
            "link_models": [path.name for path in link_paths],
            "feature_model": None if feature_path is None else feature_path.name,
            "grids": grid_path.name,
            "samples": samples_path.name,
            "seconds": self.seconds,
            "success": self.success,
            "message": self.message,
            "config": self.config,
            "sampling": self.info,
            "metadata": self.metadata,
            "database": (
                None if self.database is None else str(self.database.path)
            ),
            "protocol": self.protocol,
            "overlap_protocol": self.overlap_protocol,
            "sampling_symmetry": (
                None
                if self.symmetry is None
                else self.symmetry.metadata()
            ),
            "run_id": self.run_id,
        }
        summary_path = directory / "summary.json"
        summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
        self.paths = {
            "directory": directory,
            "energy": energy_path,
            "links": tuple(link_paths),
            "feature": feature_path,
            "grids": grid_path,
            "samples": samples_path,
            "summary": summary_path,
        }
        return self

    @classmethod
    def load(cls, directory):
        """Restore fitted fields without electronic-structure callbacks."""
        from pyqed.mps.functional import load_field_model

        directory = Path(directory)
        summary = json.loads((directory / "summary.json").read_text())
        with np.load(directory / summary["grids"], allow_pickle=False) as archive:
            grids = tuple(
                np.asarray(archive[f"grid_{axis}"], dtype=float)
                for axis in range(len(summary["grid"]))
            )
        fit = cls(
            grids,
            summary["nstates"],
            anchor=summary["anchor"],
            energy_shift=summary.get("energy_shift"),
        )
        fit.energy = load_field_model(directory / summary["energy_model"])
        fit.links = tuple(
            load_field_model(directory / path)
            for path in summary.get("link_models", ())
        ) or None
        feature_model = summary.get("feature_model")
        fit.feature = (
            None
            if feature_model is None
            else load_field_model(directory / feature_model)
        )
        fit.info = summary.get("sampling")
        fit.config = summary.get("config")
        fit.protocol = summary.get("protocol")
        fit.overlap_protocol = summary.get("overlap_protocol")
        fit.sampling_symmetry_metadata = summary.get("sampling_symmetry")
        fit.run_id = summary.get("run_id")
        fit.seconds = summary.get("seconds")
        fit.success = bool(summary.get("success", True))
        fit.message = "loaded"
        fit.labels = tuple(summary["labels"])
        fit.metadata = summary.get("metadata", {})
        fit.paths = {
            "directory": directory,
            "energy": directory / summary["energy_model"],
            "links": tuple(directory / path for path in summary["link_models"]),
            "feature": (
                None if feature_model is None else directory / feature_model
            ),
            "grids": directory / summary["grids"],
            "samples": (
                None
                if summary.get("samples") is None
                else directory / summary["samples"]
            ),
            "summary": directory / "summary.json",
        }
        return fit

    def close(self):
        if (
            self.database is not None
            and self.database.connection is not None
            and self.run_id is not None
        ):
            status = self.database.run(self.run_id)["status"]
            if status == "initialized":
                self.database.update_run(self.run_id, "sampled")
            elif status == "fitting":
                self.database.update_run(self.run_id, "interrupted")
        self.frames.close()
        if self._owns_database and self.database is not None:
            self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = ["AbInitioFit"]
