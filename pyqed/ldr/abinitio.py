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
        return _jsonable(value.tolist())
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
    For native electronic drivers, molecular permutation symmetry is detected
    on the supplied coordinate chart.  The resulting group, coordinate
    representation, and selected-state representation are owned by the fit.
    Native molecular builds use an uncertainty-calibrated MACE ensemble and
    distill the accepted matrix/endpoint fields to FunctionalTT.  This is an
    adaptation of the MACE architecture of Batatia et al., NeurIPS 35, 11423
    (2022), https://arxiv.org/abs/2206.07697; it adds chart-dependent matrix
    heads and does not inherit interatomic-potential force or accuracy claims.
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
        symmetry="auto",
        run_id=None,
        run_metadata=None,
        claim_ttl=7 * 24 * 60 * 60,
        workers=1,
        progress=None,
        energy_shift=None,
    ):
        native_driver = None
        native_states = None
        symmetry_validation = None
        automatic_symmetry = isinstance(symmetry, str) and symmetry == "auto"
        if isinstance(symmetry, str) and not automatic_symmetry:
            raise ValueError("symmetry must be 'auto', False, or a group action")
        if symmetry is False:
            symmetry = None
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
            if automatic_symmetry:
                from .sampling_symmetry import detect_symmetry

                symmetry, symmetry_validation = detect_symmetry(
                    getattr(native_driver, "mol", None), coord
                )
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
        if automatic_symmetry and native_driver is None:
            symmetry = None
        self._symmetry = symmetry
        self.symmetry_validation = (
            symmetry_validation
            if symmetry_validation is not None
            else {
                "detected": False,
                "group": "C1" if symmetry is None else str(symmetry.name),
                "order": 1 if symmetry is None else int(getattr(symmetry, "order", 1)),
                "reason": "disabled" if symmetry is None else "provided explicitly",
            }
        )
        self.group = "C1" if symmetry is None else str(symmetry.name)
        coordinate_representations = getattr(
            symmetry, "coordinate_representations", None
        )
        self.coord_repr = (
            np.eye(len(self.grids))[None, ...]
            if symmetry is None
            else None
            if coordinate_representations is None
            else np.array(coordinate_representations, copy=True)
        )
        if symmetry is None:
            self.coord_irreps = ("A",)
            self.coord_blocks = (tuple(range(len(self.grids))),)
            self.coord_basis = np.eye(len(self.grids))
            self.irrep_validation = {
                "labels": self.coord_irreps,
                "dimensions": (len(self.grids),),
                "coordinate_blocks": self.coord_blocks,
                "input_basis_is_adapted": True,
                "off_block_error": 0.0,
            }
        elif self.coord_repr is None:
            self.coord_irreps = None
            self.coord_blocks = None
            self.coord_basis = None
            self.irrep_validation = None
        else:
            from .sampling_symmetry import coord_irreps

            (
                self.coord_irreps,
                self.coord_blocks,
                self.coord_basis,
                self.irrep_validation,
            ) = coord_irreps(self.coord_repr, self.group)
        self.state_repr = (
            np.eye(self.nstates, dtype=complex)[None, ...]
            if symmetry is None
            else None
        )
        self.state_validation = None
        self._record_symmetry = (
            symmetry
            if symmetry is not None
            and bool(getattr(symmetry, "supports_record_transport", True))
            else None
        )
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
                    if self._symmetry is None
                    else self._symmetry.metadata()
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
                if self._record_symmetry is None
                else self._representative_index
            ),
            transform=(
                None
                if self._record_symmetry is None
                else self._transform_symmetry_record
            ),
            view_key=(
                None
                if self._record_symmetry is None
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

    def _mace_inputs(self):
        molecule = getattr(self.electronic_driver, "mol", None)
        if molecule is None:
            return None
        natom = int(getattr(molecule, "natom", 0))
        if natom < 2:
            return None
        symbol = getattr(molecule, "atom_symbol", None)
        if callable(symbol):
            species = tuple(str(symbol(index)) for index in range(natom))
        else:
            symbols = getattr(molecule, "atom_symbols", None)
            species = tuple(symbols()) if callable(symbols) else ()
        if len(species) != natom:
            charges = getattr(molecule, "atom_charges", None)
            species = tuple(map(int, charges())) if callable(charges) else ()
        if len(species) != natom:
            return None
        center = np.mean(np.asarray(self.coord.bounds, dtype=float), axis=1)
        try:
            geometry = np.asarray(self.coord.cartesian(center), dtype=float)
        except Exception:
            return None
        return species if geometry.shape == (natom, 3) else None

    def _design(self, count, seed, *, reduce=True):
        from scipy.stats import qmc

        count = int(count)
        if count < 1:
            raise ValueError("sample count must be positive")
        bounds = np.asarray(self.coord.bounds, dtype=float)
        power = int(np.ceil(np.log2(max(2, count))))
        unit = qmc.Sobol(len(bounds), scramble=True, seed=int(seed)).random_base2(power)
        coordinates = qmc.scale(unit[:count], bounds[:, 0], bounds[:, 1])
        return self.reduce_coordinates(coordinates) if reduce else coordinates

    def _database_coordinates(self, *, canonical=True):
        """Return reusable coordinates for the exact current protocol."""

        if self.database is None:
            return np.empty((0, len(self.grids)), dtype=float)
        cached = getattr(self, "_database_coordinate_cache", None)
        if cached is not None:
            coordinates = cached
            if not canonical or self._symmetry is None:
                return np.array(coordinates, copy=True)
            representatives, _operations = self._symmetry.canonicalize_many(
                coordinates, unique=False
            )
            fixed = np.max(np.abs(representatives - coordinates), axis=1) <= max(
                10.0 * self._symmetry.tolerance, 1.0e-10
            )
            return np.unique(coordinates[fixed], axis=0)
        database = self.database
        temporary = getattr(database, "connection", None) is None
        if temporary:
            database = ElectronicDatabase(database.path)
        protocol = canonical_json(self.protocol)
        bounds = np.asarray(self.coord.bounds, dtype=float)
        coordinates = []
        geometry_keys = set()
        try:
            for entry in database.entries():
                specification = entry.get("specification", {})
                if canonical_json(specification.get("protocol")) != protocol:
                    continue
                value = entry.get("metadata", {}).get("coordinates")
                if value is None:
                    continue
                value = np.asarray(value, dtype=float)
                if (
                    value.shape == (len(self.grids),)
                    and np.all(value >= bounds[:, 0] - 1.0e-12)
                    and np.all(value <= bounds[:, 1] + 1.0e-12)
                    and canonical_json(specification.get("geometry"))
                    == canonical_json(self.coord.cartesian(value))
                ):
                    coordinates.append(value)
                    geometry_keys.add(canonical_json(specification.get("geometry")))
        finally:
            if temporary:
                database.close()
        if not coordinates:
            return np.empty((0, len(self.grids)), dtype=float)
        coordinates = np.unique(np.asarray(coordinates), axis=0)
        self._database_coordinate_cache = coordinates
        self._database_geometry_cache = frozenset(geometry_keys)
        if not canonical or self._symmetry is None:
            return np.array(coordinates, copy=True)
        representatives, _operations = self._symmetry.canonicalize_many(
            coordinates, unique=False
        )
        fixed = np.max(np.abs(representatives - coordinates), axis=1) <= max(
            10.0 * self._symmetry.tolerance, 1.0e-10
        )
        return np.unique(coordinates[fixed], axis=0)

    @staticmethod
    def _graph_pairs(coordinates, neighbors=4):
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial.distance import cdist

        coordinates = np.asarray(coordinates, dtype=float)
        if len(coordinates) < 2:
            raise ValueError("at least two coordinates are required for links")
        scale = np.ptp(coordinates, axis=0)
        scale[scale < 1.0e-12] = 1.0
        distances = cdist(coordinates / scale, coordinates / scale)
        tree = minimum_spanning_tree(distances).tocoo()
        pairs = {
            tuple(sorted((int(left), int(right))))
            for left, right in zip(tree.row, tree.col)
        }
        np.fill_diagonal(distances, np.inf)
        count = min(int(neighbors), len(coordinates) - 1)
        for left in range(len(coordinates)):
            nearest = np.argpartition(distances[left], count - 1)[:count]
            pairs.update(tuple(sorted((left, int(right)))) for right in nearest)
        return np.asarray(sorted(pairs), dtype=int)

    @staticmethod
    def _mace_predictions(model, coordinates, pairs):
        hamiltonians = np.asarray(model.neural_energy.predict(coordinates))
        features = np.asarray(model.neural_feature.predict(coordinates))
        links = (
            features[pairs[:, 0]].conj().swapaxes(-1, -2)
            @ features[pairs[:, 1]]
        )
        return hamiltonians, links

    @staticmethod
    def _distilled_predictions(model, coordinates, pairs):
        hamiltonians = np.asarray(model.energy.predict(coordinates))
        features = np.asarray(model.feature.predict(coordinates))
        links = (
            features[pairs[:, 0]].conj().swapaxes(-1, -2)
            @ features[pairs[:, 1]]
        )
        return hamiltonians, links

    @staticmethod
    def _prediction_metrics(hamiltonians, links, fields):
        h_error = np.linalg.norm(
            hamiltonians - fields["hamiltonians"], axis=(-2, -1)
        )
        l_error = np.linalg.norm(links - fields["links"], axis=(-2, -1))
        l_scale = np.maximum(
            np.linalg.norm(fields["links"], axis=(-2, -1)),
            np.finfo(float).tiny,
        )
        return {
            "maximum_hamiltonian_error": float(np.max(h_error)),
            "rms_hamiltonian_error": float(np.sqrt(np.mean(h_error**2))),
            "relative_link_error": float(
                np.linalg.norm(l_error) / np.linalg.norm(l_scale)
            ),
            "maximum_relative_link_error": float(np.max(l_error / l_scale)),
            "hamiltonian_errors": h_error,
            "link_errors": l_error / l_scale,
        }

    @staticmethod
    def _passes_accuracy(metrics, *, h_atol, h_rms, link_rtol):
        return bool(
            metrics["maximum_hamiltonian_error"] <= h_atol
            and metrics["rms_hamiltonian_error"] <= h_rms
            and metrics["relative_link_error"] <= link_rtol
        )

    def _mace_metrics(self, models, fields):
        coordinates = np.asarray(fields["coordinates"], dtype=float)
        pairs = np.asarray(fields["pairs"], dtype=int)
        predictions = [
            self._mace_predictions(model, coordinates, pairs) for model in models
        ]
        h_values = np.asarray([value[0] for value in predictions])
        l_values = np.asarray([value[1] for value in predictions])
        h_mean = np.mean(h_values, axis=0)
        l_mean = np.mean(l_values, axis=0)
        metrics = self._prediction_metrics(h_mean, l_mean, fields)
        h_spread = np.sqrt(
            np.mean(np.linalg.norm(h_values - h_mean, axis=(-2, -1)) ** 2, axis=0)
        )
        l_spread = np.sqrt(
            np.mean(np.linalg.norm(l_values - l_mean, axis=(-2, -1)) ** 2, axis=0)
        )
        metrics.update({
            "hamiltonian_spread": h_spread,
            "link_spread": l_spread / np.maximum(
                np.linalg.norm(fields["links"], axis=(-2, -1)),
                np.finfo(float).tiny,
            ),
        })
        return metrics

    @staticmethod
    def _calibrate_uncertainty(metrics, quantile=0.95):
        floor = np.finfo(float).eps
        h_factor = float(
            np.quantile(
                metrics["hamiltonian_errors"]
                / np.maximum(metrics["hamiltonian_spread"], floor),
                quantile,
                method="higher",
            )
        )
        l_factor = float(
            np.quantile(
                metrics["link_errors"]
                / np.maximum(metrics["link_spread"], floor),
                quantile,
                method="higher",
            )
        )
        return {
            "quantile": float(quantile),
            "hamiltonian_factor": h_factor,
            "link_factor": l_factor,
            "hamiltonian_coverage": float(
                np.mean(
                    metrics["hamiltonian_errors"]
                    <= h_factor * np.maximum(metrics["hamiltonian_spread"], floor)
                )
            ),
            "link_coverage": float(
                np.mean(
                    metrics["link_errors"]
                    <= l_factor * np.maximum(metrics["link_spread"], floor)
                )
            ),
        }

    @staticmethod
    def _uncertainty_coverage(metrics, calibration):
        floor = np.finfo(float).eps
        h_limit = calibration["hamiltonian_factor"] * np.maximum(
            metrics["hamiltonian_spread"], floor
        )
        l_limit = calibration["link_factor"] * np.maximum(
            metrics["link_spread"], floor
        )
        return {
            "hamiltonian_coverage": float(
                np.mean(metrics["hamiltonian_errors"] <= h_limit)
            ),
            "link_coverage": float(np.mean(metrics["link_errors"] <= l_limit)),
            "maximum_hamiltonian_bound": float(np.max(h_limit)),
            "maximum_link_bound": float(np.max(l_limit)),
        }

    def _fit_mace_ensemble(
        self,
        fields,
        *,
        species,
        finite_group,
        feature_rank,
        ensemble,
        epochs,
        seed,
        previous=(),
        encoder_options=None,
        hidden=(96, 96),
        sync_steps=300,
    ):
        from pyqed.ml import MACE

        options = {
            "channels": 20,
            "max_ell": 2,
            "interactions": 2,
            "correlation": 2,
            "radial_basis": 8,
            "radial_mlp": (64, 64),
            "cutoff": 5.0,
            "dtype": "float64",
        }
        options.update({} if encoder_options is None else encoder_options)
        models = []
        synchronized_targets = None
        coordinates = np.asarray(fields["coordinates"], dtype=float)
        anchor_coordinate = np.mean(np.asarray(self.coord.bounds, dtype=float), axis=1)
        anchor = int(np.argmin(np.linalg.norm(coordinates - anchor_coordinate, axis=1)))
        for member in range(int(ensemble)):
            model = MACE(
                self.grids,
                species,
                self.coord.cartesian,
                self.nstates,
                chart_features=True,
                chart_bounds=self.coord.bounds,
                periodic_axes=self.coord.periodic_axes,
                geometry_units="bohr",
                **options,
            ).fit_y(
                (coordinates, fields["hamiltonians"]),
                coordinates,
                fields["pairs"],
                fields["links"],
                feature_targets=synchronized_targets,
                feature_rank=int(feature_rank),
                anchor=anchor,
                feature_objective="links-only",
                ambient_representation="full",
                energy_representation="direct",
                energy_objective="trace-traceless",
                finite_group=finite_group,
                hidden=tuple(int(value) for value in hidden),
                epochs=int(epochs),
                sync_steps=int(sync_steps),
                initial_fit=(previous[member] if member < len(previous) else None),
                seed=int(seed) + 97 * member,
                distill=False,
            )
            if not model.success:
                raise RuntimeError(f"MACE ensemble member {member} failed: {model.message}")
            models.append(model)
            synchronized_targets = np.asarray(model.feature_targets_)
        return tuple(models)

    def _build_mace(self, options):
        species = self._mace_inputs()
        if species is None:
            raise RuntimeError("MACE requires a molecular Cartesian coordinate chart")
        initial = int(options.pop("initial", max(48, 12 * len(self.grids))))
        batch = int(options.pop("batch", max(12, 4 * len(self.grids))))
        maximum = int(options.pop("maximum", max(initial + 3 * batch, 128)))
        calibration_count = int(options.pop("calibration", 32))
        validation_count = int(options.pop("validation", 64))
        ensemble_count = int(options.pop("ensemble", 3))
        epochs = int(options.pop("epochs", 450))
        refinement_epochs = int(options.pop("refinement_epochs", 180))
        maximum_rounds = options.pop("rounds", None)
        maximum_rounds = None if maximum_rounds is None else int(maximum_rounds)
        if maximum_rounds is not None and maximum_rounds < 1:
            raise ValueError("rounds must be positive")
        feature_rank = int(options.pop("feature_rank", 8 * self.nstates))
        feature_rank = int(np.ceil(feature_rank / self.nstates) * self.nstates)
        h_atol = float(options.pop("hamiltonian_atol", 1.5e-3))
        h_rms = float(options.pop("hamiltonian_rms", 5.0e-4))
        link_rtol = float(options.pop("link_rtol", 5.0e-3))
        distill_rtol = float(options.pop("distill_rtol", 2.0e-3))
        coverage_atol = float(options.pop("coverage", 0.90))
        if not 0.0 <= coverage_atol <= 1.0:
            raise ValueError("coverage must lie between zero and one")
        tt_rank = int(options.pop("rank", 64))
        tt_degree = options.pop("degrees", 10)
        tt_degree = int(tt_degree if np.isscalar(tt_degree) else max(tt_degree))
        seed = int(options.pop("seed", 0))
        strict = bool(options.pop("strict", True))
        reuse_database = bool(options.pop("reuse_database", True))
        cache_only = bool(options.pop("cache_only", False))
        verbose = bool(options.pop("verbose", False))
        self._cache_only = cache_only
        if cache_only:
            self.frames.builder = None
        encoder_options = options.pop("encoder", None)
        hidden = options.pop("hidden", (96, 96))
        sync_steps = int(options.pop("sync_steps", 300))
        if options:
            raise TypeError("unknown MACE fit options: " + ", ".join(sorted(options)))

        origin = np.mean(np.asarray(self.coord.bounds, dtype=float), axis=1)
        database_pool = (
            self._database_coordinates()
            if reuse_database
            else np.empty((0, len(self.grids)), dtype=float)
        )
        database_candidates = int(len(database_pool))
        if len(database_pool):
            keep = np.linalg.norm(database_pool - origin, axis=1) > 1.0e-12
            database_pool = database_pool[keep]
            database_pool = database_pool[
                np.random.default_rng(seed + 7).permutation(len(database_pool))
            ]

        def take(count, fallback_seed):
            nonlocal database_pool
            count = int(count)
            selected = database_pool[:count]
            database_pool = database_pool[len(selected):]
            attempts = 0
            while len(selected) < count:
                if cache_only:
                    raise RuntimeError(
                        "the electronic database does not contain enough disjoint "
                        "coordinates for the requested cache-only design"
                    )
                generated = self._design(
                    max(2 * (count - len(selected)), count),
                    fallback_seed + attempts,
                )
                selected = np.unique(np.vstack((selected, generated)), axis=0)
                attempts += 1
            return selected[:count]

        calibration = take(calibration_count, seed + 11)
        validation = take(validation_count, seed + 23)
        training = self.reduce_coordinates(
            np.vstack((origin, take(max(initial - 1, 1), seed + 1)))
        )
        calibration_pairs = self._graph_pairs(calibration)
        validation_pairs = self._graph_pairs(validation)
        calibration_fields = self.continuous_fields(calibration, calibration_pairs)
        validation_fields = self.continuous_fields(validation, validation_pairs)
        finite_group = self.mace_group(feature_rank) if self._symmetry is not None else None
        models = ()
        history = []
        round_index = 0
        while True:
            pairs = self._graph_pairs(training)
            training_fields = self.continuous_fields(training, pairs)
            models = self._fit_mace_ensemble(
                training_fields,
                species=species,
                finite_group=finite_group,
                feature_rank=feature_rank,
                ensemble=ensemble_count,
                epochs=epochs if not models else refinement_epochs,
                seed=seed + 1000 * round_index,
                previous=models,
                encoder_options=encoder_options,
                hidden=hidden,
                sync_steps=sync_steps,
            )
            calibration_metrics = self._mace_metrics(models, calibration_fields)
            validation_metrics = self._mace_metrics(models, validation_fields)
            uncertainty = self._calibrate_uncertainty(calibration_metrics)
            validation_coverage = self._uncertainty_coverage(
                validation_metrics, uncertainty
            )
            accepted = (
                validation_metrics["maximum_hamiltonian_error"] <= h_atol
                and validation_metrics["rms_hamiltonian_error"] <= h_rms
                and validation_metrics["relative_link_error"] <= link_rtol
                and validation_coverage["hamiltonian_coverage"] >= coverage_atol
                and validation_coverage["link_coverage"] >= coverage_atol
            )
            history.append(
                {
                    "round": round_index,
                    "training_points": int(len(training)),
                    "validation": {
                        key: value
                        for key, value in validation_metrics.items()
                        if np.isscalar(value)
                    },
                    "uncertainty": uncertainty,
                    "validation_coverage": validation_coverage,
                    "accepted": bool(accepted),
                }
            )
            if verbose:
                print(f"MACE adaptive round {round_index}: {history[-1]}", flush=True)
            if (
                accepted
                or len(training) >= maximum
                or maximum_rounds is not None
                and round_index + 1 >= maximum_rounds
            ):
                break
            pool_count = max(8 * batch, 128)
            if len(database_pool):
                pool = database_pool[:pool_count]
                database_pool = database_pool[len(pool):]
            else:
                if cache_only:
                    raise RuntimeError(
                        "cache-only adaptive sampling exhausted reusable database points"
                    )
                pool = self._design(pool_count, seed + 100 + round_index)
            bounds = np.asarray(self.coord.bounds, dtype=float)
            scale = np.ptp(bounds, axis=1)
            distance = np.linalg.norm(
                (pool[:, None, :] - training[None, :, :]) / scale,
                axis=-1,
            ).min(axis=1)
            h_values = np.asarray(
                [model.neural_energy.predict(pool) for model in models]
            )
            h_mean = np.mean(h_values, axis=0)
            spread = np.sqrt(
                np.mean(
                    np.linalg.norm(h_values - h_mean, axis=(-2, -1)) ** 2,
                    axis=0,
                )
            )
            score = uncertainty["hamiltonian_factor"] * spread + h_atol * distance
            acquired = pool[np.argsort(score)[-batch:]]
            training = self.reduce_coordinates(np.vstack((training, acquired)))
            round_index += 1

        member_metrics = [
            self._mace_metrics((candidate,), validation_fields)
            for candidate in models
        ]
        member_scores = [
            max(
                metrics["maximum_hamiltonian_error"]
                / max(h_atol, np.finfo(float).tiny),
                metrics["rms_hamiltonian_error"]
                / max(h_rms, np.finfo(float).tiny),
                metrics["relative_link_error"]
                / max(link_rtol, np.finfo(float).tiny),
            )
            for metrics in member_metrics
        ]
        best = int(np.argmin(member_scores))
        model = models[best]
        selected_metrics = member_metrics[best]
        selected_accepted = self._passes_accuracy(
            selected_metrics,
            h_atol=h_atol,
            h_rms=h_rms,
            link_rtol=link_rtol,
        )
        self.mace = model
        self.ensemble = models
        self.energy = None
        self.feature = None
        self.links = None
        self.info = dict(model.info)
        self.info["adaptive"] = history
        self.validation = {
            "calibration": history[-1]["uncertainty"],
            "uncertainty_coverage": history[-1]["validation_coverage"],
            "independent": history[-1]["validation"],
            "independent_hamiltonian_errors": validation_metrics[
                "hamiltonian_errors"
            ],
            "independent_link_errors": validation_metrics["link_errors"],
            "selected_member": {
                key: value
                for key, value in selected_metrics.items()
                if np.isscalar(value)
            },
            "state_symmetry": self.state_validation,
        }
        self.acceptance = {
            "accepted": False,
            "stage": "neural",
            "hamiltonian_atol": h_atol,
            "hamiltonian_rms": h_rms,
            "link_rtol": link_rtol,
            "distill_rtol": distill_rtol,
            "coverage": coverage_atol,
        }
        self.model = "mace"
        self.success = False
        self.message = "neural production acceptance gates failed"
        self.config = {
            "gauge": "anchor-procrustes",
            "unitarize_links": False,
            "model": self.model,
            "feature_rank": feature_rank,
            "ensemble": ensemble_count,
            "training_points": int(len(training)),
            "maximum_points": maximum,
            "maximum_rounds": maximum_rounds,
            "seed": seed,
            "database_reuse": reuse_database,
            "cache_only": cache_only,
            "database_candidates": database_candidates,
        }
        if not history[-1]["accepted"] or not selected_accepted:
            self.acceptance["stage"] = (
                "ensemble" if not history[-1]["accepted"] else "selected-member"
            )
            if strict:
                raise RuntimeError(
                    "MACE fit failed neural production acceptance gates: "
                    f"{self.validation}"
                )
            return self

        model.distill_y(
            rank=tt_rank,
            degree=tt_degree,
            method="cross",
            validation_points=max(128, validation_count),
            seed=seed + 5000,
        )
        distillation = model.info["distillation"]
        final_hamiltonians, final_links = self._distilled_predictions(
            model,
            np.asarray(validation_fields["coordinates"], dtype=float),
            np.asarray(validation_fields["pairs"], dtype=int),
        )
        final_metrics = self._prediction_metrics(
            final_hamiltonians, final_links, validation_fields
        )
        accepted = bool(
            distillation["energy_relative_error"] <= distill_rtol
            and distillation["feature_relative_error"] <= distill_rtol
            and self._passes_accuracy(
                final_metrics,
                h_atol=h_atol,
                h_rms=h_rms,
                link_rtol=link_rtol,
            )
        )
        self.energy = model.energy
        self.feature = model.feature
        self.info = dict(model.info)
        self.info["adaptive"] = history
        self.validation["distillation"] = distillation
        self.validation["final"] = {
            key: value
            for key, value in final_metrics.items()
            if np.isscalar(value)
        }
        self.acceptance.update(accepted=accepted, stage="distillation")
        self.model = "mace-ftt"
        self.config["model"] = self.model
        self.success = accepted
        self.message = "accepted" if accepted else "production acceptance gates failed"
        if strict and not accepted:
            raise RuntimeError(
                "MACE fit failed production acceptance gates: "
                f"{self.validation}"
            )
        return self

    def build(self):
        """Run the native adaptive electronic fit."""
        if self.success:
            return self
        if not self._native:
            raise RuntimeError("callback-based fits use run() with an explicit design")
        options = dict(self.fit_options)
        requested_model = str(options.pop("model", "auto")).lower()
        if requested_model not in {"auto", "mace", "ftt"}:
            raise ValueError("model must be 'auto', 'mace', or 'ftt'")
        use_mace = requested_model == "mace" or (
            requested_model == "auto" and self._mace_inputs() is not None
        )
        if use_mace:
            started = time.perf_counter()
            with self:
                if self.database is not None:
                    self.database.update_run(self.run_id, "fitting")
                try:
                    result = self._build_mace(options)
                except Exception as error:
                    self.message = str(error)
                    if self.database is not None:
                        self.database.update_run(self.run_id, "failed")
                    raise
                else:
                    if self.database is not None:
                        self.database.update_run(
                            self.run_id, "fitted" if self.success else "rejected"
                        )
                finally:
                    self.seconds = time.perf_counter() - started
            return result
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
            self._ensure_state_repr(strict=False)
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
        database = self.database
        if database is not None and getattr(database, "connection", None) is None:
            database = database.path
        reference = type(self)(
            grid.x,
            len(self.states),
            builder,
            frame=_electronic_frame_record,
            energies=_energy_view(self.states),
            overlap=_overlap_view(self.states),
            overlap_protocol=self.overlap_protocol,
            geometry=self.coord.cartesian,
            database=database,
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
        periodic_axes = frozenset(getattr(grid, "periodic_axes", ()))
        for left in points:
            for axis, size in enumerate(grid.shape):
                if left[axis] + 1 >= size and axis not in periodic_axes:
                    continue
                right = list(left)
                right[axis] = (right[axis] + 1) % size
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
        database = self.database
        if database is not None and getattr(database, "connection", None) is None:
            database = database.path
        reference = type(self)(
            source,
            self.nstates,
            None if bool(getattr(self, "_cache_only", False)) else builder,
            anchor=(0,) * len(self.grids),
            frame=_electronic_frame_record,
            energies=_energy_view(self.states),
            overlap=_overlap_view(self.states),
            overlap_protocol=self.overlap_protocol,
            geometry=self.coord.cartesian,
            database=database,
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

    def reduce_coordinates(self, coordinates):
        """Return unique coordinate-orbit representatives for fitting."""

        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != len(self.grids):
            raise ValueError("coordinates have the wrong dimension")
        if self._symmetry is None:
            return np.unique(coordinates, axis=0)
        reduce = getattr(self._symmetry, "canonicalize_many", None)
        if not callable(reduce):
            raise TypeError(
                "the configured symmetry does not reduce scattered coordinates"
            )
        representatives, _inverse, _operations = reduce(coordinates, unique=True)
        return representatives

    def reduce_pairs(self, coordinates, pairs):
        """Jointly reduce pairs while preserving their physical link geometry."""

        coordinates = np.asarray(coordinates, dtype=float)
        pairs = np.asarray(pairs, dtype=int)
        if self._symmetry is None:
            return coordinates, pairs
        reduce = getattr(self._symmetry, "canonicalize_pairs", None)
        if not callable(reduce):
            raise TypeError("the configured symmetry does not reduce scattered pairs")
        representatives, representative_pairs, _operations = reduce(
            coordinates, pairs
        )
        return representatives, representative_pairs

    def orbit(self, coordinates):
        """Return the symmetry orbit of one coordinate vector."""

        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != (len(self.grids),):
            raise ValueError("coordinates have the wrong dimension")
        if self._symmetry is None:
            return coordinates[None, :]
        return np.asarray(self._symmetry.images(coordinates), dtype=float)

    def reduced_size(self, full_size):
        """Return the quotient-domain budget for a full-domain budget."""

        if self._symmetry is None:
            return int(full_size)
        count = getattr(self._symmetry, "representative_count", None)
        return int(full_size) if not callable(count) else int(count(full_size))

    def _state_calibration_coordinates(self, count=2, *, offset=0):
        origin = np.asarray(self._symmetry.origin, dtype=float)
        database = self._database_coordinates(canonical=False)
        representatives = self._database_coordinates(canonical=True)
        if len(database) and len(representatives):
            tolerance = max(float(self._symmetry.tolerance), 1.0e-10)

            def key(value):
                return tuple(np.rint(np.asarray(value) / tolerance).astype(np.int64))

            stored = {key(value) for value in database}
            stored_geometries = getattr(self, "_database_geometry_cache", frozenset())
            complete = []
            scale = np.ptp(np.asarray(self.coord.bounds, dtype=float), axis=1)
            for base in representatives:
                orbit = origin + np.einsum(
                    "gij,j->gi",
                    self.coord_repr,
                    np.asarray(base) - origin,
                    optimize=True,
                )
                orbit_keys = {key(value) for value in orbit}
                if len(orbit_keys) != len(self.coord_repr):
                    continue
                exact_orbit = all(
                    canonical_json(self.coord.cartesian(value)) in stored_geometries
                    for value in orbit
                )
                if orbit_keys.issubset(stored) and exact_orbit:
                    radius = float(np.linalg.norm((base - origin) / scale))
                    if radius > 1.0e-4:
                        complete.append((radius, orbit))
            complete.sort(key=lambda item: item[0])
            selected = (
                complete[: int(count)]
                if int(offset) == 0
                else complete[-int(count):]
                if len(complete) >= int(count) + 2
                else []
            )
            if len(selected) == int(count):
                return np.asarray([orbit for _radius, orbit in selected])
        if bool(getattr(self, "_cache_only", False)):
            raise RuntimeError(
                "the electronic database does not contain enough complete symmetry "
                "orbits for cache-only state-representation validation"
            )
        spans = np.ptp(np.asarray(self.coord.bounds, dtype=float), axis=1)
        axes = np.arange(1, len(spans) + 1, dtype=float)
        orbits = []
        for local_number in range(int(count)):
            number = int(offset) + local_number
            direction = np.sin((number + 1.37) * axes) + 0.4 * np.cos(
                (number + 0.73) * axes
            )
            direction /= max(float(np.max(np.abs(direction))), 1.0)
            amplitude = 0.12 + 0.06 * (local_number % 5)
            base = origin + amplitude * spans * direction
            orbit = origin + np.einsum(
                "gij,j->gi",
                self.coord_repr,
                base - origin,
                optimize=True,
            )
            orbits.append(orbit)
        return np.asarray(orbits)

    def _validate_state_repr(self, count=4):
        validation = self._state_calibration_coordinates(count, offset=11)
        order = validation.shape[1]
        coordinates = validation.reshape(-1, validation.shape[-1])
        pairs = np.asarray(
            [
                (orbit * order, orbit * order + operation)
                for orbit in range(len(validation))
                for operation in range(1, order)
            ],
            dtype=int,
        )
        fields = self.continuous_fields(coordinates, pairs)
        hamiltonians = fields["hamiltonians"].reshape(
            len(validation), order, self.nstates, self.nstates
        )
        errors = []
        for orbit in hamiltonians:
            source = orbit[0] - np.trace(orbit[0]) / self.nstates * np.eye(
                self.nstates
            )
            scale = max(float(np.linalg.norm(source)), np.finfo(float).tiny)
            for operation, expected in zip(self.state_repr, orbit):
                expected = expected - np.trace(expected) / self.nstates * np.eye(
                    self.nstates
                )
                predicted = operation @ source @ operation.conj().T
                errors.append(float(np.linalg.norm(predicted - expected) / scale))
        singular = np.linalg.svd(fields["links"], compute_uv=False)
        return {
            "independent_orbits": int(len(validation)),
            "independent_maximum_covariance_error": max(errors, default=0.0),
            "independent_rms_covariance_error": float(
                np.sqrt(np.mean(np.square(errors))) if errors else 0.0
            ),
            "minimum_manifold_singular_value": float(np.min(singular)),
            "one_percent_manifold_singular_value": float(
                np.quantile(singular, 0.01)
            ),
        }

    def _ensure_state_repr(self, *, strict=True):
        if self.state_repr is not None:
            return self.state_repr
        if self._symmetry is None or self.coord_repr is None:
            return None
        if self.nstates == 1:
            self.state_repr = np.ones((len(self.coord_repr), 1, 1), dtype=complex)
            self.state_validation = {
                "maximum_covariance_error": 0.0,
                "closure_error": 0.0,
                "maximum_null_ratio": 0.0,
                "calibration_orbits": 0,
            }
            return self.state_repr
        calibration = self._state_calibration_coordinates()
        try:
            fields = self.continuous_fields(
                calibration.reshape(-1, calibration.shape[-1])
            )
            hamiltonians = fields["hamiltonians"].reshape(
                calibration.shape[0],
                calibration.shape[1],
                self.nstates,
                self.nstates,
            )
            from .sampling_symmetry import infer_state_repr

            self.state_repr, self.state_validation = infer_state_repr(
                self.coord_repr, hamiltonians
            )
            self.state_validation.update(self._validate_state_repr())
            if (
                self.state_validation["independent_maximum_covariance_error"]
                > 2.0e-3
                or self.state_validation["minimum_manifold_singular_value"] < 0.9
            ):
                raise RuntimeError(
                    "selected-state symmetry validation failed: "
                    f"{self.state_validation}"
                )
        except Exception as error:
            self.state_validation = {"error": str(error)}
            self.symmetry_validation["state_error"] = str(error)
            if strict:
                raise
        return self.state_repr

    def mace_group(self, feature_rank, *, tolerance=None):
        """Return the detected finite-group data consumed by MACE."""

        if self._symmetry is None or self.coord_repr is None:
            return None
        state_repr = self._ensure_state_repr(strict=True)
        return self._symmetry.mace_group(
            state_repr,
            feature_rank=int(feature_rank),
            tolerance=tolerance,
        )

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
            image = self._symmetry.resolve(self.coordinates(index))
            if len(image.representative_coordinates) != len(self.grids):
                raise ValueError(
                    "sampling-symmetry representative has the wrong coordinate dimension"
                )
            self._symmetry_images[index] = image
        return image

    def _index_for_coordinates(self, coordinates):
        index = []
        tolerance = max(
            float(getattr(self._symmetry, "tolerance", 0.0)), 1.0e-12
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

        if self._symmetry is None:
            return self.coordinates(index)
        return tuple(self._symmetry_image(index).representative_coordinates)

    def representative_geometry(self, index):
        """Return the canonical geometry stored in the electronic database."""

        coordinates = self.representative_coordinates(index)
        return coordinates if self.geometry_of is None else self.geometry_of(coordinates)

    def expand_points(self, points):
        """Expand explicit grid samples to complete molecular-symmetry orbits."""

        points = tuple(dict.fromkeys(self.frames._index(point) for point in points))
        if self._symmetry is None:
            return points
        expanded = []
        for point in points:
            for coordinates in self._symmetry.images(self.coordinates(point)):
                expanded.append(self._index_for_coordinates(coordinates))
        return tuple(dict.fromkeys(expanded))

    def expand_pairs(self, pairs):
        """Expand explicit links by applying each symmetry operation jointly."""

        normalized = tuple(
            (self.frames._index(left), self.frames._index(right))
            for left, right in pairs
        )
        if self._symmetry is None:
            return tuple(dict.fromkeys(normalized))
        expanded = []
        for left, right in normalized:
            for left_coordinates, right_coordinates in self._symmetry.pair_images(
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
        return self._record_symmetry.transform_record(
            record,
            self._symmetry_image(index),
            representative_geometry=self.representative_geometry(representative),
            requested_geometry=self.sample_geometry(index),
            protocol=self.protocol,
        )

    def _symmetry_view_key(self, index):
        return self._record_symmetry.view_key(self._symmetry_image(index))

    def sample(self, index):
        """Describe one grid point for provenance and database queries."""

        index = tuple(int(value) for value in index)
        sample = {
            "index": index,
            "coordinates": self.coordinates(index),
            "geometry": self.sample_geometry(index),
        }
        if self._record_symmetry is not None:
            image = self._symmetry_image(index)
            sample.update(
                {
                    "representative_index": self._representative_index(index),
                    "representative_coordinates": image.representative_coordinates,
                    "representative_geometry": self.representative_geometry(index),
                    "sampling_symmetry": self._symmetry.view_key(image),
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
            "geometry": (
                self.representative_geometry(index)
                if self._record_symmetry is not None
                else self.sample_geometry(index)
            ),
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
        mace_path = None
        if getattr(self, "mace", None) is not None:
            mace_path = self.mace.save(directory / "mace.pt")
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
            "mace_model": None if mace_path is None else mace_path.name,
            "grids": grid_path.name,
            "samples": samples_path.name,
            "seconds": self.seconds,
            "success": self.success,
            "message": self.message,
            "config": self.config,
            "sampling": self.info,
            "validation": getattr(self, "validation", None),
            "acceptance": getattr(self, "acceptance", None),
            "model": getattr(self, "model", None),
            "metadata": self.metadata,
            "database": (
                None if self.database is None else str(self.database.path)
            ),
            "protocol": self.protocol,
            "overlap_protocol": self.overlap_protocol,
            "sampling_symmetry": (
                None
                if self._symmetry is None
                else self._symmetry.metadata()
            ),
            "symmetry": {
                "group": self.group,
                "coord_repr": self.coord_repr,
                "state_repr": self.state_repr,
                "coord_irreps": self.coord_irreps,
                "coord_blocks": self.coord_blocks,
                "coord_basis": self.coord_basis,
                "detection": self.symmetry_validation,
                "irreps": self.irrep_validation,
                "states": self.state_validation,
            },
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
            "mace": mace_path,
        }
        return self

    @classmethod
    def load(cls, directory, *, geometry=None):
        """Restore fitted fields without electronic-structure callbacks."""
        from pyqed.mps.functional import load_field_model

        directory = Path(directory)
        summary = json.loads((directory / "summary.json").read_text())
        with np.load(directory / summary["grids"], allow_pickle=False) as archive:
            grids = tuple(
                np.asarray(archive[f"grid_{axis}"], dtype=float)
                for axis in range(len(summary["grid"]))
            )
        symmetry = summary.get("symmetry", {})
        sampling_symmetry = summary.get("sampling_symmetry") or {}
        coordinate_representations = symmetry.get("coord_repr")
        restored_symmetry = False
        if (
            coordinate_representations is not None
            and len(coordinate_representations) > 1
        ):
            from .sampling_symmetry import FiniteGroupSamplingSymmetry

            operations = sampling_symmetry.get("operations")
            if operations is not None and len(operations) != len(
                coordinate_representations
            ):
                operations = None
            restored_symmetry = FiniteGroupSamplingSymmetry(
                coordinate_representations,
                name=symmetry.get(
                    "group", sampling_symmetry.get("name", "finite-group")
                ),
                operations=operations,
                origin=sampling_symmetry.get("origin"),
                tolerance=float(sampling_symmetry.get("tolerance", 1.0e-10)),
            )
        fit = cls(
            grids,
            summary["nstates"],
            anchor=summary["anchor"],
            energy_shift=summary.get("energy_shift"),
            symmetry=restored_symmetry,
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
        fit.validation = summary.get("validation")
        fit.acceptance = summary.get("acceptance")
        fit.model = summary.get("model")
        fit.config = summary.get("config")
        fit.protocol = summary.get("protocol")
        fit.overlap_protocol = summary.get("overlap_protocol")
        fit.sampling_symmetry_metadata = summary.get("sampling_symmetry")
        fit.group = symmetry.get("group", "C1")
        if symmetry.get("coord_repr") is not None:
            fit.coord_repr = np.asarray(symmetry["coord_repr"], dtype=float)
        if symmetry.get("state_repr") is not None:
            fit.state_repr = np.asarray(symmetry["state_repr"])
        fit.coord_irreps = symmetry.get("coord_irreps")
        fit.coord_blocks = symmetry.get("coord_blocks")
        if symmetry.get("coord_basis") is not None:
            fit.coord_basis = np.asarray(symmetry["coord_basis"], dtype=float)
        fit.symmetry_validation = symmetry.get("detection")
        fit.irrep_validation = symmetry.get("irreps")
        fit.state_validation = symmetry.get("states")
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
            "mace": (
                None
                if summary.get("mace_model") is None
                else directory / summary["mace_model"]
            ),
        }
        mace_model = summary.get("mace_model")
        fit.mace_checkpoint = (
            None if mace_model is None else directory / mace_model
        )
        if fit.mace_checkpoint is not None and geometry is not None:
            from pyqed.ml import MACE

            fit.mace = MACE.load(
                fit.mace_checkpoint, geometry, distill=False
            )
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
        if (
            self.database is not None
            and self.database.connection is not None
            and self.run_id is not None
        ):
            self.database.release_claims(self.run_id)
        if self._owns_database and self.database is not None:
            self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = ["AbInitioFit"]
