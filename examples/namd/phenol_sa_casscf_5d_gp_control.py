#!/usr/bin/env python3
r"""Run a matched five-dimensional phenol GP/NGP control on adiabatic S1.

The construction follows the physical question posed by C. Xie, J. Ma,
X. Zhu, D. R. Yarkony, D. Xie, and H. Guo, *J. Am. Chem. Soc.* **138**,
7828--7831 (2016), DOI: 10.1021/jacs.6b03288, and the explicit
GP/DBOC comparison of C. Xie and H. Guo, *Chem. Phys. Lett.* **683**,
222--227 (2017), DOI: 10.1016/j.cplett.2017.02.026.

This is an adaptation rather than a reproduction of either calculation.  It
projects PyQED's three-state SA(6)-CASSCF/MACE field onto adiabatic S1, uses
the five-dimensional numerical Podolsky KEO, and represents the electronic
overlaps by discrete scalar LDR links.  Both controls retain the same
MACE endpoint-overlap magnitude compressed deterministically over
``(R_OH, phi, q16)`` and broadcast over the narrower ``q3`` and ``q10``
ranges, and hence the same approximate finite-link quantum-metric/DBOC
contribution.  GP multiplies that magnitude by the real Z2
branch-cut gauge of the physical inner S1/S2 conical intersection; NGP removes
only this phase.  The branch cut is trivial throughout the initial basin and
excludes the known outer-region root-leak topology.  The adiabatic potential,
link magnitudes, initial state, grid, KEO, CAP, and propagator are otherwise
identical.  The fitted electronic field and borrowed local-P-gauge quasibound
envelope must still be qualified before quantitative lifetimes are claimed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

if os.environ.get("PYQED_NO_MATPLOTLIB") == "1":
    plt = None
else:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
import numpy as np

from pyqed.cache import file_signature as _file_signature, specs_equivalent
from examples.namd.phenol_sa_casscf_3d_gp_control import (
    maximum_spanning_tree_gauge,
    nearest,
    rectangular_loop_phase,
)
from examples.namd.phenol_sa_casscf_5d_ftt_ttldr import (
    _load_operator_cache,
    _save_operator_cache,
)
from examples.namd.phenol_sa_casscf_5d_quasibound import (
    add_constant_to_tt,
    fit_discrete_tt,
    load_interpolated_state,
    mps_marginals,
    sample_mps_indices,
    tt_cores_to_diagonal_mpo,
)
from pyqed.ldr.ttfit import corewise_link_mpo_components
from pyqed.ml import CorrectedMatrixField, MACE, RadialMatrixCorrection
from pyqed.models.phenol_coordinates import PhenolReactiveChart
from pyqed.mps import MPS, MPO
from pyqed.mps.cross import tt_cross, tt_value
from pyqed.mps.functional import FunctionalTT
from pyqed.namd.ttldr import TTLDR, polynomial_cap
from pyqed.units import au2fs


ROOT = Path("dataset/phenol_5d_production")
DEFAULT_CHECKPOINT = ROOT / "model/mace_y_probability_expanded_final_polished/phenol_sa6_5d_mace_y.pt"
DEFAULT_RADIAL_CORRECTION = ROOT / "model/radial_correction_probability_expanded/phenol_sa6_5d_radial_delta.npz"
DEFAULT_KEO_CACHE = ROOT / "cache/quasibound_scalar_keo_65x21x23x21x17"
DEFAULT_INITIAL_STATE = ROOT / "states/s1_origin_5d_quasibound_localwell_h3_corrected/phenol_sa_casscf_5d_s1_quasibound.npz"
BASE_POTENTIAL_CACHE = ROOT / "cache/gp_ngp_s1_scalar_fields_65x21x23x21x17"
BASE_RESIDUAL_CACHE = ROOT / "cache/gp_ngp_s1_base_qualified_65x21x23x21x17"
DEFAULT_FIELD_CACHE = ROOT / "cache/gp_ngp_s1_full_overlap_fields_65x21x23x21x17"
DEFAULT_RESIDUAL_CACHE = ROOT / "cache/gp_ngp_s1_full_overlap_base_qualified_65x21x23x21x17"
DEFAULT_GP_OPERATOR_CACHE = ROOT / "cache/gp_s1_full_overlap_operator_65x21x23x21x17"
DEFAULT_NGP_OPERATOR_CACHE = ROOT / "cache/ngp_s1_full_overlap_operator_65x21x23x21x17"
DEFAULT_OUTPUT = ROOT / "dynamics/gp_ngp_full_overlap_pilot"
COLORS = {"gp": "#0072B2", "ngp": "#D55E00"}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _load_keo(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    axes = tuple(np.load(directory / name) for name in metadata["axes"])
    components = []
    for entries in metadata["components"]:
        components.append(MPO([np.load(directory / entry["path"], mmap_mode="r") for entry in entries]))
    active = metadata["keo_info"]["component_active_axes"]
    if active is None or len(active) != len(components):
        raise ValueError("the scalar KEO cache lacks active-axis labels")
    return axes, tuple((tuple(map(int, axes_)), component) for axes_, component in zip(active, components)), metadata


class ProjectedS1Oracle:
    """Evaluate scalar S1 energies and, when supplied, endpoint features."""

    def __init__(
        self,
        axes,
        energy,
        feature,
        state=1,
        prediction_batch_size=256,
        joint_predictor=None,
    ):
        self.axes = tuple(np.asarray(axis, dtype=float) for axis in axes)
        self.energy = energy
        self.feature = feature
        self.joint_predictor = joint_predictor
        self.state = int(state)
        self.prediction_batch_size = int(prediction_batch_size)
        if self.prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        self.cache = {}
        self.raw_cache = {}
        self.minimum_reference_overlap = 1.0
        self.minimum_link_magnitude = 1.0
        if self.feature is not None:
            self._build_reference_gauge()
        equilibrium = np.asarray(PhenolReactiveChart().equilibrium)
        self.energy_reference = float(self._energies(equilibrium[None])[0])

    def _states(self, coordinates):
        if self.feature is None:
            raise RuntimeError("endpoint features were not supplied to this oracle")
        energies = []
        ambient = []
        for start in range(0, len(coordinates), self.prediction_batch_size):
            points = coordinates[start : start + self.prediction_batch_size]
            if self.joint_predictor is None:
                hamiltonian = np.asarray(self.energy.predict(points), dtype=complex)
                feature = np.asarray(self.feature.predict(points), dtype=complex)
            else:
                predicted = self.joint_predictor(points)
                hamiltonian = np.asarray(predicted["energy"], dtype=complex)
                feature = np.asarray(predicted["feature"], dtype=complex)
            hamiltonian = 0.5 * (
                hamiltonian + hamiltonian.swapaxes(-1, -2).conj()
            )
            batch_energies, vectors = np.linalg.eigh(hamiltonian)
            selected = vectors[:, :, self.state]
            energies.append(batch_energies[:, self.state])
            ambient.append(
                np.einsum("nra,na->nr", feature, selected, optimize=True)
            )
        energies = np.concatenate(energies)
        ambient = np.concatenate(ambient)
        norms = np.linalg.norm(ambient, axis=1)
        if np.any(norms <= 1.0e-12):
            raise RuntimeError("the fitted electronic feature loses the S1 state")
        return energies, ambient / norms[:, None]

    def _energies(self, coordinates):
        values = []
        for start in range(0, len(coordinates), self.prediction_batch_size):
            points = coordinates[start : start + self.prediction_batch_size]
            hamiltonian = np.asarray(self.energy.predict(points), dtype=complex)
            hamiltonian = 0.5 * (
                hamiltonian + hamiltonian.swapaxes(-1, -2).conj()
            )
            values.append(np.linalg.eigvalsh(hamiltonian)[:, self.state])
        return np.concatenate(values)

    def _build_reference_gauge(self):
        radius, torsion = self.axes[:2]
        equilibrium = np.asarray(PhenolReactiveChart().equilibrium)
        rr, pp = np.meshgrid(radius, torsion, indexing="ij")
        coordinates = np.repeat(equilibrium[None], rr.size, axis=0)
        coordinates[:, 0] = rr.ravel()
        coordinates[:, 1] = pp.ravel()
        _energies, vectors = self._states(coordinates)
        self.reference_vectors = vectors.reshape(len(radius), len(torsion), -1)
        radial = np.einsum(
            "ijr,ijr->ij",
            self.reference_vectors[:-1].conj(),
            self.reference_vectors[1:],
            optimize=True,
        )[:, :, None]
        angular = np.einsum(
            "ijr,ijr->ij",
            self.reference_vectors[:, :-1].conj(),
            self.reference_vectors[:, 1:],
            optimize=True,
        )[:, :, None]
        dummy = np.ones((len(radius), len(torsion), 0), dtype=complex)
        anchor = (nearest(radius, equilibrium[0]), nearest(torsion, 0.0), 0)
        phase, self.reference_links, self.tree_minimum = maximum_spanning_tree_gauge(
            (radial, angular, dummy), (len(radius), len(torsion), 1), anchor
        )
        self.reference_phase = phase[:, :, 0]
        self.anchor = anchor

    def _indices_to_coordinates(self, indices):
        return np.column_stack(
            [self.axes[axis][indices[:, axis]] for axis in range(len(self.axes))]
        )

    def nodes(self, indices):
        indices = np.asarray(indices, dtype=int)
        keys = [tuple(map(int, row)) for row in indices]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        if missing:
            missing_indices = np.asarray(missing, dtype=int)
            energies, vectors = self._states(self._indices_to_coordinates(missing_indices))
            reference = self.reference_vectors[missing_indices[:, 0], missing_indices[:, 1]]
            overlap = np.einsum("nr,nr->n", reference.conj(), vectors, optimize=True)
            magnitude = np.abs(overlap)
            if np.any(magnitude <= 1.0e-8):
                point = missing[int(np.argmin(magnitude))]
                raise RuntimeError(f"the spectator-to-planar S1 gauge is singular near {point}")
            phase = (
                self.reference_phase[missing_indices[:, 0], missing_indices[:, 1]]
                * overlap.conj()
                / magnitude
            )
            self.minimum_reference_overlap = min(
                self.minimum_reference_overlap, float(np.min(magnitude))
            )
            self.cache.update(
                (key, (float(value), vector, gauge))
                for key, value, vector, gauge in zip(missing, energies, vectors, phase)
            )
        values = [self.cache[key] for key in keys]
        return (
            np.asarray([value[0] for value in values]),
            np.asarray([value[1] for value in values]),
            np.asarray([value[2] for value in values]),
        )

    def raw_nodes(self, indices):
        """Return eigenstates without imposing the planar reference gauge."""
        indices = np.asarray(indices, dtype=int)
        keys = [tuple(map(int, row)) for row in indices]
        missing = list(dict.fromkeys(key for key in keys if key not in self.raw_cache))
        if missing:
            missing_indices = np.asarray(missing, dtype=int)
            energies, vectors = self._states(
                self._indices_to_coordinates(missing_indices)
            )
            self.raw_cache.update(
                (key, (float(value), vector))
                for key, value, vector in zip(missing, energies, vectors)
            )
        values = [self.raw_cache[key] for key in keys]
        return (
            np.asarray([value[0] for value in values]),
            np.asarray([value[1] for value in values]),
        )

    def potential(self, indices):
        indices = np.asarray(indices, dtype=int)
        energies = self._energies(self._indices_to_coordinates(indices))
        return energies - self.energy_reference

    def link(self, axis, indices, *, strip_phase=False):
        indices = np.asarray(indices, dtype=int)
        right = indices.copy()
        right[:, int(axis)] += 1
        if strip_phase:
            _el, left_vectors = self.raw_nodes(indices)
            _er, right_vectors = self.raw_nodes(right)
            raw = np.einsum(
                "nr,nr->n", left_vectors.conj(), right_vectors, optimize=True
            )
            self.minimum_link_magnitude = min(
                self.minimum_link_magnitude, float(np.min(np.abs(raw)))
            )
            return np.abs(raw).astype(complex)
        _el, left_vectors, left_phase = self.nodes(indices)
        _er, right_vectors, right_phase = self.nodes(right)
        raw = np.einsum("nr,nr->n", left_vectors.conj(), right_vectors, optimize=True)
        signed = left_phase.conj() * raw * right_phase
        self.minimum_link_magnitude = min(
            self.minimum_link_magnitude, float(np.min(np.abs(signed)))
        )
        return signed

    def loop_phases(self):
        radius, torsion = self.axes[:2]
        loops = {
            "inner": ((1.02, -0.15), (1.25, 0.15)),
            "outer": ((1.75, -0.15), (1.98, 0.15)),
        }
        result = {}
        for name, (lower, upper) in loops.items():
            lo = (nearest(radius, lower[0]), nearest(torsion, lower[1]), 0)
            hi = (nearest(radius, upper[0]), nearest(torsion, upper[1]), 0)
            phase, minimum = rectangular_loop_phase(self.reference_links, lo, hi)
            result[name] = {"phase_radian": phase, "minimum_link_magnitude": minimum}
        return result


class DiscreteLinkTT:
    """Directional scalar link represented by discrete TT cores."""

    output_shape_ = (1, 1)

    def __init__(self, cores, shape):
        self.cores = tuple(np.asarray(core) for core in cores)
        self.shape = tuple(map(int, shape))

    def tensor_cores(self, grids):
        if tuple(len(grid) for grid in grids) != self.shape:
            raise ValueError("directional link requested on an incompatible edge grid")
        return [*self.cores, np.ones((1, 1, 1), dtype=complex)]

    def values(self, indices):
        return np.asarray([tt_value(self.cores, row) for row in np.asarray(indices, dtype=int)])


def _field_spec(args, axes):
    return {
        "checkpoint": _file_signature(args.checkpoint),
        "radial_correction": _file_signature(args.radial_correction),
        "grid_shape": [len(axis) for axis in axes],
        "potential_rank": int(args.potential_tt_rank),
        "cross_sweeps": int(args.field_cross_sweeps),
        "cross_rtol": float(args.field_cross_rtol),
        "cross_validation": int(args.field_cross_validation),
        "link_fit": "deterministic-r-phi-q16-reference-tensor-svd",
        "link_tensor_rank": int(args.link_tensor_rank),
        "link_tensor_rtol": float(args.link_tensor_rtol),
        "link_support_validation": int(args.link_support_validation),
        "link_ci_validation": int(args.link_ci_validation),
        "link_validation": int(args.link_validation),
        "link_support_rms_target": float(args.link_support_rms_target),
        "link_ci_rms_target": float(args.link_ci_rms_target),
        "prediction_batch_size": int(args.prediction_batch_size),
        "seed": int(args.seed),
        "state": 1,
        "gauge": "reactive-plane-magnitude-inner-ci-z2-outer-clean-v6",
    }


def _save_field_cache(directory, axes, potential, links, spec, info):
    directory = Path(directory)
    stage = directory.with_name(directory.name + ".tmp")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    files = {"axes": [], "potential": [], "links": {}}
    for site, axis in enumerate(axes):
        name = f"axis_{site:02d}.npy"
        np.save(stage / name, axis)
        files["axes"].append(name)
    for site, core in enumerate(potential):
        name = f"potential_site_{site:02d}.npy"
        np.save(stage / name, core)
        files["potential"].append(name)
    for mode, fields in links.items():
        files["links"][mode] = []
        for axis, cores in enumerate(fields):
            names = []
            for site, core in enumerate(cores):
                name = f"{mode}_axis_{axis:02d}_site_{site:02d}.npy"
                np.save(stage / name, core)
                names.append(name)
            files["links"][mode].append(names)
    metadata = {"version": 1, "spec": spec, "files": files, "info": _jsonable(info)}
    (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if directory.exists():
        shutil.rmtree(directory)
    stage.replace(directory)


def _load_field_cache(directory, axes, spec):
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    if not specs_equivalent(metadata.get("spec"), spec):
        return None
    cached_axes = tuple(np.load(directory / name) for name in metadata["files"]["axes"])
    if any(not np.array_equal(left, right) for left, right in zip(cached_axes, axes)):
        return None
    potential = tuple(np.load(directory / name) for name in metadata["files"]["potential"])
    links = {
        mode: tuple(
            tuple(np.load(directory / name) for name in names)
            for names in fields
        )
        for mode, fields in metadata["files"]["links"].items()
    }
    return potential, links, metadata["info"]


def _load_compatible_potential(directory, axes, spec):
    """Reuse a validated potential when only link construction changed."""
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    potential_keys = (
        "checkpoint",
        "radial_correction",
        "grid_shape",
        "potential_rank",
        "cross_sweeps",
        "cross_rtol",
        "cross_validation",
        "prediction_batch_size",
        "seed",
        "state",
    )
    cached_spec = metadata.get("spec", {})
    if any(
        not specs_equivalent(cached_spec.get(key), spec.get(key))
        for key in potential_keys
    ):
        return None
    cached_axes = tuple(np.load(directory / name) for name in metadata["files"]["axes"])
    if any(not np.array_equal(left, right) for left, right in zip(cached_axes, axes)):
        return None
    potential = tuple(np.load(directory / name) for name in metadata["files"]["potential"])
    return potential, metadata["info"]["potential"]


def _residual_spec(args, axes):
    return {
        "base_field": _file_signature(Path(args.field_cache) / "metadata.json"),
        "checkpoint": _file_signature(args.checkpoint),
        "radial_correction": _file_signature(args.radial_correction),
        "initial_state": _file_signature(args.initial_state),
        "grid_shape": [len(axis) for axis in axes],
        "levels": int(args.potential_residual_levels),
        "rank": int(args.potential_residual_rank),
        "sweeps": int(args.potential_residual_sweeps),
        "validation": int(args.potential_residual_validation),
        "support_samples": int(args.potential_residual_support_samples),
        "guard_samples": int(args.potential_residual_guard_samples),
        "support_target_ev": float(args.potential_residual_target_ev),
        "guard_target_ev": float(args.potential_residual_guard_target_ev),
        "weighted_levels": int(args.potential_residual_weighted_levels),
        "weighted_rank": int(args.potential_residual_weighted_rank),
        "weighted_degree": int(args.potential_residual_weighted_degree),
        "weighted_samples": int(args.potential_residual_weighted_samples),
        "weighted_guard_samples": int(
            args.potential_residual_weighted_guard_samples
        ),
        "weighted_sweeps": int(args.potential_residual_weighted_sweeps),
        "weighted_regularization": float(
            args.potential_residual_weighted_regularization
        ),
        "discrete_levels": int(args.potential_residual_discrete_levels),
        "discrete_rank": int(args.potential_residual_discrete_rank),
        "discrete_samples": int(args.potential_residual_discrete_samples),
        "discrete_guard_samples": int(
            args.potential_residual_discrete_guard_samples
        ),
        "discrete_sweeps": int(args.potential_residual_discrete_sweeps),
        "discrete_regularization": float(
            args.potential_residual_discrete_regularization
        ),
        "seed": int(args.seed),
        "symmetry": "exact phi-and-q16 reflection-folded half-grid",
    }


def _save_residual_cache(directory, axes, corrections, spec, info):
    directory = Path(directory)
    stage = directory.with_name(directory.name + ".tmp")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    files = {"axes": [], "corrections": []}
    for site, axis in enumerate(axes):
        name = f"axis_{site:02d}.npy"
        np.save(stage / name, axis)
        files["axes"].append(name)
    for component, cores in enumerate(corrections):
        names = []
        for site, core in enumerate(cores):
            name = f"correction_{component:02d}_site_{site:02d}.npy"
            np.save(stage / name, core)
            names.append(name)
        files["corrections"].append(names)
    (stage / "metadata.json").write_text(
        json.dumps(
            {"version": 1, "spec": spec, "files": files, "info": _jsonable(info)},
            indent=2,
        )
        + "\n"
    )
    if directory.exists():
        shutil.rmtree(directory)
    stage.replace(directory)


def _load_residual_cache(directory, axes, spec):
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    if not specs_equivalent(metadata.get("spec"), spec):
        return None
    cached_axes = tuple(np.load(directory / name) for name in metadata["files"]["axes"])
    if any(not np.array_equal(left, right) for left, right in zip(cached_axes, axes)):
        return None
    corrections = tuple(
        tuple(np.load(directory / name) for name in names)
        for names in metadata["files"]["corrections"]
    )
    return corrections, metadata["info"]


def _load_compatible_residual(directory, axes, spec):
    """Reuse a residual fit when its only changed input is link metadata."""
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    cached_spec = dict(metadata.get("spec", {}))
    current_spec = dict(spec)
    cached_spec.pop("base_field", None)
    current_spec.pop("base_field", None)
    if not specs_equivalent(cached_spec, current_spec):
        return None
    cached_axes = tuple(np.load(directory / name) for name in metadata["files"]["axes"])
    if any(not np.array_equal(left, right) for left, right in zip(cached_axes, axes)):
        return None
    corrections = tuple(
        tuple(np.load(directory / name) for name in names)
        for names in metadata["files"]["corrections"]
    )
    return corrections, metadata["info"]


def _sum_tt_components(components):
    """Exactly direct-sum scalar TT components without a dense grid."""
    components = [tuple(np.asarray(core) for core in cores) for cores in components]
    if not components:
        raise ValueError("at least one TT component is required")
    if len(components) == 1:
        return tuple(core.copy() for core in components[0])
    sites = len(components[0])
    if any(len(cores) != sites for cores in components):
        raise ValueError("TT components have incompatible lengths")
    result = [np.concatenate([cores[0] for cores in components], axis=2)]
    for site in range(1, sites - 1):
        physical = components[0][site].shape[1]
        left = sum(cores[site].shape[0] for cores in components)
        right = sum(cores[site].shape[2] for cores in components)
        core = np.zeros((left, physical, right), dtype=np.result_type(*[cores[site] for cores in components]))
        lo = ro = 0
        for cores in components:
            block = cores[site]
            core[lo : lo + block.shape[0], :, ro : ro + block.shape[2]] = block
            lo += block.shape[0]
            ro += block.shape[2]
        result.append(core)
    result.append(np.concatenate([cores[-1] for cores in components], axis=0))
    return tuple(result)


def _reflection_pair_tt(cores, reflection_sites=(1, 3)):
    original = [np.asarray(core).copy() for core in cores]
    reflected = [np.asarray(core).copy() for core in cores]
    original[0] *= 0.5
    reflected[0] *= 0.5
    for site in reflection_sites:
        reflected[site] = reflected[site][:, ::-1, :]
    return tuple(original), tuple(reflected)


def _reflection_fold(axes, reflection_sites=(1, 3)):
    """Return a half-grid shape and exact full/half reflection mappings."""
    shape = [len(axis) for axis in axes]
    centers = {}
    for site in reflection_sites:
        axis = np.asarray(axes[site])
        center = len(axis) // 2
        if len(axis) % 2 != 1 or not np.allclose(axis, -axis[::-1]):
            raise ValueError("reflection-folded axes must be odd and centered at zero")
        centers[site] = center
        shape[site] = center + 1

    def to_full(indices):
        full = np.asarray(indices, dtype=int).copy()
        for site, center in centers.items():
            full[:, site] += center
        return full

    def expand(cores):
        expanded = [np.asarray(core).copy() for core in cores]
        for site, center in centers.items():
            mapping = np.abs(np.arange(len(axes[site])) - center)
            expanded[site] = expanded[site][:, mapping, :]
        return tuple(expanded)

    return tuple(shape), to_full, expand


def _potential_values(components, indices):
    indices = np.asarray(indices, dtype=int)
    return sum(
        np.asarray([tt_value(cores, row) for row in indices]).real
        for cores in components
    )


def _potential_error_statistics(error):
    error = 27.211386245988 * np.asarray(error, dtype=float)
    absolute = np.abs(error)
    return {
        "rms_ev": float(np.sqrt(np.mean(error**2))),
        "p95_absolute_ev": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_ev": float(np.max(absolute)),
        "mean_ev": float(np.mean(error)),
    }


def _build_potential_residual(args, axes, base):
    spec = _residual_spec(args, axes)
    cached = _load_residual_cache(args.potential_residual_cache, axes, spec)
    if cached is not None:
        corrections, info = cached
        return _sum_tt_components((base, *corrections)), info, True
    compatible = _load_compatible_residual(BASE_RESIDUAL_CACHE, axes, spec)
    if compatible is not None:
        corrections, info = compatible
        info = dict(info)
        info["cache_reused_after_link_change"] = True
        _save_residual_cache(args.potential_residual_cache, axes, corrections, spec, info)
        return _sum_tt_components((base, *corrections)), info, True
    fit = MACE.load(
        args.checkpoint, PhenolReactiveChart().geometry, device="cpu", distill=False
    )
    energy = CorrectedMatrixField(
        fit.neural_energy, RadialMatrixCorrection.load(args.radial_correction)
    )
    oracle = ProjectedS1Oracle(
        axes, energy, None, prediction_batch_size=args.prediction_batch_size
    )
    shape = tuple(len(axis) for axis in axes)
    support_state = load_interpolated_state(args.initial_state, axes, sites=None)
    support = sample_mps_indices(
        support_state, args.potential_residual_support_samples, args.seed + 701
    )[:, :5]
    rng = np.random.default_rng(args.seed + 709)
    guard = np.column_stack(
        [rng.integers(0, size, args.potential_residual_guard_samples) for size in shape]
    )
    exact_support = oracle.potential(support)
    exact_guard = oracle.potential(guard)
    components = [tuple(base)]
    corrections = []

    def coordinates(indices):
        indices = np.asarray(indices, dtype=int)
        return np.column_stack(
            [axis[indices[:, site]] for site, axis in enumerate(axes)]
        )

    def qualify(level, cross=None):
        support_error = _potential_values(components, support) - exact_support
        guard_error = _potential_values(components, guard) - exact_guard
        record = {
            "level": int(level),
            "support": _potential_error_statistics(support_error),
            "guard": _potential_error_statistics(guard_error),
        }
        if cross is not None:
            record["cross"] = cross
        record["passed"] = bool(
            record["support"]["rms_ev"] <= args.potential_residual_target_ev
            and record["guard"]["rms_ev"] <= args.potential_residual_guard_target_ev
        )
        print(
            f"[potential residual] level {level}: "
            f"support={record['support']['rms_ev']:.6f} eV "
            f"guard={record['guard']['rms_ev']:.6f} eV",
            flush=True,
        )
        return record

    levels = [qualify(0)]
    equilibrium = np.asarray(
        [nearest(axis, value) for axis, value in zip(axes, PhenolReactiveChart().equilibrium)],
        dtype=int,
    )[None]
    folded_shape, folded_to_full, expand_folded = _reflection_fold(axes)
    for level in range(1, args.potential_residual_levels + 1):
        if levels[-1]["passed"]:
            break
        def residual(indices):
            indices = np.asarray(indices, dtype=int)
            return oracle.potential(indices) - _potential_values(components, indices)

        folded_equilibrium = equilibrium.copy()
        for site in (1, 3):
            folded_equilibrium[:, site] = abs(
                folded_equilibrium[:, site] - len(axes[site]) // 2
            )
        offset = float(residual(folded_to_full(folded_equilibrium))[0])
        cores, cross_info = tt_cross(
            folded_shape,
            lambda _index: 0.0,
            batch_evaluator=lambda indices: residual(folded_to_full(indices)) - offset,
            max_rank=args.potential_residual_rank,
            sweeps=args.potential_residual_sweeps,
            rtol=0.0,
            validation=args.potential_residual_validation,
            seed=args.seed + 719 * level,
            start_rank=min(8, args.potential_residual_rank),
            kick_rank=4,
        )
        cores = expand_folded(add_constant_to_tt(cores, offset))
        corrections.append(cores)
        components.append(cores)
        cross_info = dict(cross_info)
        cross_info["representation"] = "reflection-folded-half-grid"
        cross_info["folded_shape"] = list(folded_shape)
        levels.append(qualify(level, cross=cross_info))
        if levels[-1]["passed"]:
            break
    for discrete_level in range(1, args.potential_residual_discrete_levels + 1):
        if levels[-1]["passed"]:
            break
        training_support = sample_mps_indices(
            support_state,
            args.potential_residual_discrete_samples,
            args.seed + 401 * discrete_level,
        )[:, :5]
        training_guard = np.column_stack(
            [
                rng.integers(
                    0,
                    size,
                    size=args.potential_residual_discrete_guard_samples,
                )
                for size in shape
            ]
        )
        training_indices = np.vstack((training_support, training_guard))
        reflected = training_indices.copy()
        for site in (1, 3):
            reflected[:, site] = shape[site] - 1 - reflected[:, site]
        training_indices = np.vstack((training_indices, reflected))
        training_target = (
            oracle.potential(training_indices)
            - _potential_values(components, training_indices)
        )
        validation_target = exact_support - _potential_values(components, support)
        cores, discrete_info = fit_discrete_tt(
            training_indices,
            training_target,
            shape,
            rank=args.potential_residual_discrete_rank,
            sweeps=args.potential_residual_discrete_sweeps,
            regularization=args.potential_residual_discrete_regularization,
            seed=args.seed + 409 * discrete_level,
            validation=(support, validation_target),
        )
        pair = _reflection_pair_tt(cores)
        corrections.extend(pair)
        components.extend(pair)
        discrete_info.update(
            {
                "discrete_level": int(discrete_level),
                "training_support_samples": int(len(training_support)),
                "training_guard_samples": int(len(training_guard)),
                "reflection_augmented_samples": int(len(training_indices)),
            }
        )
        levels.append(
            qualify(len(levels), cross=discrete_info)
        )
    for weighted_level in range(1, args.potential_residual_weighted_levels + 1):
        if levels[-1]["passed"]:
            break
        training_support = sample_mps_indices(
            support_state,
            args.potential_residual_weighted_samples,
            args.seed + 307 * weighted_level,
        )[:, :5]
        training_guard = np.column_stack(
            [
                rng.integers(
                    0,
                    size,
                    size=args.potential_residual_weighted_guard_samples,
                )
                for size in shape
            ]
        )
        training_indices = np.vstack((training_support, training_guard))
        training_target = (
            oracle.potential(training_indices)
            - _potential_values(components, training_indices)
        )
        validation_target = exact_support - _potential_values(components, support)
        model = FunctionalTT(
            degrees=(args.potential_residual_weighted_degree,) * len(shape),
            rank=args.potential_residual_weighted_rank,
            bounds=np.asarray([(axis[0], axis[-1]) for axis in axes]),
            normalization="frobenius",
            hermitian=False,
            regularization=args.potential_residual_weighted_regularization,
            sweeps=args.potential_residual_weighted_sweeps,
            rtol=1.0e-8,
            local_rtol=1.0e-6,
            local_maxiter=30,
            patience=4,
            random_state=args.seed + 311 * weighted_level,
        ).fit(
            coordinates(training_indices),
            training_target,
            validation=(coordinates(support), validation_target),
        )
        pair = _reflection_pair_tt(model.grid_cores(axes))
        corrections.extend(pair)
        components.extend(pair)
        levels.append(
            qualify(
                len(levels),
                cross={
                    "method": "state-weighted-functional-tt",
                    "weighted_level": int(weighted_level),
                    "rank": int(args.potential_residual_weighted_rank),
                    "degree": int(args.potential_residual_weighted_degree),
                    "training_support_samples": int(len(training_support)),
                    "training_guard_samples": int(len(training_guard)),
                    "sweeps": int(model.n_sweeps),
                    "train_relative_error": float(model.error),
                    "validation_relative_error": float(model.validation_error),
                },
            )
        )
    info = {
        "method": "support-qualified direct-MACE TT residuals",
        "levels": levels,
        "passed": bool(levels[-1]["passed"]),
        "combined_ranks": [core.shape[2] for core in _sum_tt_components(components)],
    }
    _save_residual_cache(args.potential_residual_cache, axes, corrections, spec, info)
    return _sum_tt_components(components), info, False


def _phase_sheet_cores(values, edge_shape, *, tolerance=1.0e-12):
    """Embed one exact R/phi phase sheet in a five-coordinate TT."""
    values = np.asarray(values, dtype=complex)
    if values.shape != tuple(edge_shape[:2]):
        raise ValueError("phase sheet and directional edge shape are incompatible")
    u, singular, vh = np.linalg.svd(values, full_matrices=False)
    threshold = float(tolerance) * singular[0] if len(singular) else 0.0
    rank = max(1, int(np.count_nonzero(singular > threshold)))
    cores = [
        (u[:, :rank] * singular[:rank]).reshape(1, values.shape[0], rank),
        vh[:rank].reshape(rank, values.shape[1], 1),
    ]
    cores.extend(
        np.ones((1, int(size), 1), dtype=complex) for size in edge_shape[2:]
    )
    return tuple(cores), rank


def _inner_conical_phase_links(axes, energy, *, prediction_batch_size=256):
    """Return a unit Z2 branch-cut connection for the fitted inner S1/S2 CI."""
    radius, torsion = axes[:2]
    equilibrium = np.asarray(PhenolReactiveChart().equilibrium)
    rr, pp = np.meshgrid(radius, torsion, indexing="ij")
    coordinates = np.repeat(equilibrium[None], rr.size, axis=0)
    coordinates[:, 0] = rr.ravel()
    coordinates[:, 1] = pp.ravel()
    eigenvalues = []
    for start in range(0, len(coordinates), int(prediction_batch_size)):
        points = coordinates[start : start + int(prediction_batch_size)]
        hamiltonian = np.asarray(energy.predict(points), dtype=complex)
        hamiltonian = 0.5 * (
            hamiltonian + hamiltonian.swapaxes(-1, -2).conj()
        )
        eigenvalues.append(np.linalg.eigvalsh(hamiltonian))
    eigenvalues = np.concatenate(eigenvalues).reshape(len(radius), len(torsion), -1)
    gap = eigenvalues[:, :, 2] - eigenvalues[:, :, 1]
    physical_window = (rr >= 0.95) & (rr <= 1.50) & (np.abs(pp) <= 0.25)
    masked = np.where(physical_window, gap, np.inf)
    ci_index = np.unravel_index(np.argmin(masked), masked.shape)
    ci_radius = float(radius[ci_index[0]])
    zero = nearest(torsion, 0.0)
    cut_edge = zero if zero + 1 < len(torsion) else zero - 1
    radial_links = np.ones((len(radius) - 1, len(torsion)), dtype=complex)
    angular_links = np.ones((len(radius), len(torsion) - 1), dtype=complex)
    angular_links[radius > ci_radius, cut_edge] = -1.0
    first_cut = int(np.flatnonzero(radius > ci_radius)[0])
    flux_radius = 0.5 * float(radius[first_cut - 1] + radius[first_cut])
    flux_torsion = 0.5 * float(torsion[cut_edge] + torsion[cut_edge + 1])
    links = (
        radial_links[..., None],
        angular_links[..., None],
        np.ones((len(radius), len(torsion), 0), dtype=complex),
    )
    info = {
        "construction": "real Z2 branch-cut connection",
        "gap_search_window": {"R_angstrom": [0.95, 1.50], "abs_phi_radian": 0.25},
        "gap_minimum_grid_index": list(map(int, ci_index)),
        "gap_minimum_grid_coordinate": [
            float(radius[ci_index[0]]), float(torsion[ci_index[1]])
        ],
        "gap_minimum_ev": float(gap[ci_index] * 27.211386245988),
        "phase_flux_center": [flux_radius, flux_torsion],
        "cut_angular_edge_index": int(cut_edge),
        "cut_starts_radial_index": first_cut,
        "phase_flux_regularization": (
            "the cut starts on the first outer radial node, placing the singular flux on a dual-grid plaquette"
        ),
    }
    return links, info


def _phase_link_cores(reference_links, shape):
    """Encode only the clean inner-CI Z2 phase on every edge grid."""
    result = []
    records = []
    for axis in range(len(shape)):
        edge_shape = list(shape)
        edge_shape[axis] -= 1
        cores = tuple(
            np.ones((1, int(size), 1), dtype=complex) for size in edge_shape
        )
        rank = 1
        if axis < 2:
            values = np.asarray(reference_links[axis][..., 0], dtype=complex)
            magnitude = np.abs(values)
            if np.any(magnitude <= 1.0e-12):
                raise RuntimeError("the planar Berry phase sheet contains a zero link")
            cores, rank = _phase_sheet_cores(values / magnitude, edge_shape)
        result.append(cores)
        records.append(
            {
                "axis": axis,
                "phase_sheet_rank": int(rank),
                "ranks": [int(core.shape[2]) for core in cores[:-1]],
            }
        )
    return tuple(result), records


def _tt_hadamard(left, right):
    """Return the exact elementwise product of two scalar tensor trains."""
    if len(left) != len(right):
        raise ValueError("tensor trains have different numbers of sites")
    result = []
    for a, b in zip(left, right):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape[1] != b.shape[1]:
            raise ValueError("tensor-train physical dimensions differ")
        core = np.einsum("aib,cid->acibd", a, b, optimize=True)
        result.append(
            core.reshape(a.shape[0] * b.shape[0], a.shape[1], a.shape[2] * b.shape[2])
        )
    return tuple(result)


def _dense_tt_svd(values, *, max_rank, rtol):
    """Compress a small dense tensor deterministically into TT cores."""
    values = np.asarray(values)
    shape = values.shape
    cores = []
    remainder = values
    left = 1
    discarded = []
    for site, physical in enumerate(shape[:-1]):
        matrix = remainder.reshape(left * physical, -1)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        threshold = float(rtol) * singular[0] if len(singular) else 0.0
        rank = max(1, min(int(max_rank), int(np.count_nonzero(singular > threshold))))
        discarded.append(float(np.sum(singular[rank:] ** 2)))
        cores.append(u[:, :rank].reshape(left, physical, rank))
        remainder = singular[:rank, None] * vh[:rank]
        left = rank
    cores.append(remainder.reshape(left, shape[-1], 1))
    return tuple(cores), {
        "ranks": [int(core.shape[2]) for core in cores[:-1]],
        "discarded_frobenius_squared": discarded,
    }


def _tt_to_dense(cores):
    values = np.asarray(cores[0])[0]
    for core in cores[1:]:
        values = np.tensordot(values, core, axes=([-1], [0]))
    return values[..., 0]


def _embed_r_phi_q16_cores(cores, edge_shape):
    """Embed R/phi/q16 TT cores while broadcasting across q3 and q10."""
    first, second, last = map(np.asarray, cores)
    carried = second.shape[2]
    identity_q3 = np.repeat(
        np.eye(carried, dtype=second.dtype)[:, None, :], edge_shape[2], axis=1
    )
    identity_q10 = np.repeat(
        np.eye(carried, dtype=second.dtype)[:, None, :], edge_shape[3], axis=1
    )
    return first, second, identity_q3, identity_q10, last


def _full_overlap_link_cores(args, axes, oracle, reference_links):
    """Fit common non-unit link magnitudes, then add phase only to GP.

    The magnitude fit resolves ``R_OH``, ``phi``, and ``q16`` explicitly and
    broadcasts that tensor over the two weaker spectator modes ``q3`` and
    ``q10``.  This approximation is shared exactly by the GP and NGP controls.
    """
    shape = tuple(len(axis) for axis in axes)
    phases, phase_records = _phase_link_cores(reference_links, shape)
    links = {"gp": [], "ngp": []}
    records = []
    rng = np.random.default_rng(args.seed + 809)
    equilibrium = np.asarray(PhenolReactiveChart().equilibrium)
    fixed = np.asarray(
        [nearest(grid, value) for grid, value in zip(axes, equilibrium)], dtype=int
    )
    support_state = load_interpolated_state(args.initial_state, axes, sites=None)
    support = sample_mps_indices(
        support_state, args.link_support_validation, args.seed + 811
    )[:, : len(shape)]
    ci_radius = np.flatnonzero((axes[0] >= 0.95) & (axes[0] <= 1.50))
    ci_torsion = np.flatnonzero(np.abs(axes[1]) <= 0.25)
    ci = np.column_stack(
        [rng.integers(0, size, args.link_ci_validation) for size in shape]
    )
    ci[:, 0] = rng.choice(ci_radius, size=len(ci))
    ci[:, 1] = rng.choice(ci_torsion, size=len(ci))
    if len(support):
        spectators = support[rng.integers(0, len(support), len(ci))]
        ci[:, 2:] = spectators[:, 2:]
    uniform = np.column_stack(
        [rng.integers(0, size, args.link_validation) for size in shape]
    )

    def statistics(axis, indices, cores):
        indices = np.asarray(indices, dtype=int).copy()
        indices[:, axis] = np.minimum(indices[:, axis], shape[axis] - 2)
        exact = oracle.link(axis, indices, strip_phase=True).real
        predicted = np.asarray([tt_value(cores, row) for row in indices]).real
        error = predicted - exact
        return {
            "samples": int(len(indices)),
            "rms_absolute_error": float(np.sqrt(np.mean(error**2))),
            "maximum_absolute_error": float(np.max(np.abs(error))),
            "exact_magnitude_range": [float(np.min(exact)), float(np.max(exact))],
            "predicted_magnitude_range": [
                float(np.min(predicted)), float(np.max(predicted))
            ],
        }

    for axis in range(len(shape)):
        edge_shape = list(shape)
        edge_shape[axis] -= 1
        print(f"[links] constructing axis {axis} reactive-plane magnitude", flush=True)
        rr, pp, zz = np.meshgrid(
            np.arange(edge_shape[0]),
            np.arange(edge_shape[1]),
            np.arange(edge_shape[4]),
            indexing="ij",
        )
        indices = np.repeat(fixed[None], rr.size, axis=0)
        indices[:, 0] = rr.ravel()
        indices[:, 1] = pp.ravel()
        indices[:, 4] = zz.ravel()
        indices[:, axis] = np.minimum(indices[:, axis], edge_shape[axis] - 1)
        sheet = oracle.link(axis, indices, strip_phase=True).real.reshape(
            edge_shape[0], edge_shape[1], edge_shape[4]
        )
        sheet_cores, sheet_info = _dense_tt_svd(
            sheet, max_rank=args.link_tensor_rank, rtol=args.link_tensor_rtol
        )
        reconstructed = _tt_to_dense(sheet_cores).real
        positivity_shift = max(0.0, 1.0e-12 - float(np.min(reconstructed)))
        if positivity_shift:
            sheet_cores = tuple(add_constant_to_tt(sheet_cores, positivity_shift))
            reconstructed += positivity_shift
        sheet_error = reconstructed - sheet
        magnitude = list(_embed_r_phi_q16_cores(sheet_cores, edge_shape))
        magnitude = tuple(magnitude)
        ngp = tuple(np.asarray(core, dtype=complex) for core in magnitude)
        gp = _tt_hadamard(ngp, phases[axis])
        validation = uniform.copy()
        validation[:, axis] = np.minimum(validation[:, axis], edge_shape[axis] - 1)
        gp_values = np.asarray([tt_value(gp, row) for row in validation])
        ngp_values = np.asarray([tt_value(ngp, row) for row in validation])
        magnitude_mismatch = float(
            np.max(np.abs(np.abs(gp_values) - np.abs(ngp_values)))
        )
        links["gp"].append(gp)
        links["ngp"].append(ngp)
        record = {
            "axis": axis,
            "representation": (
                "deterministic R/phi/q16 magnitude TT-SVD broadcast over q3 and q10"
            ),
            "spectator_reference_indices": list(map(int, fixed)),
            "reference_tensor": {
                **sheet_info,
                "shape": list(map(int, sheet.shape)),
                "rms_absolute_error": float(np.sqrt(np.mean(sheet_error**2))),
                "maximum_absolute_error": float(np.max(np.abs(sheet_error))),
                "positivity_shift": float(positivity_shift),
            },
            "reference_sheet_magnitude_range": [
                float(np.min(sheet)), float(np.max(sheet))
            ],
            "edge_shape": list(map(int, edge_shape)),
            "uniform_guard": statistics(axis, uniform, ngp),
            "initial_support": statistics(axis, support, ngp),
            "inner_ci_support": statistics(axis, ci, ngp),
            "gp_ngp_max_magnitude_mismatch": magnitude_mismatch,
            "ngp_ranks": [int(core.shape[2]) for core in ngp[:-1]],
            "gp_ranks": [int(core.shape[2]) for core in gp[:-1]],
        }
        record["passed"] = bool(
            record["initial_support"]["rms_absolute_error"]
            <= args.link_support_rms_target
            and record["inner_ci_support"]["rms_absolute_error"]
            <= args.link_ci_rms_target
        )
        if not record["passed"]:
            raise RuntimeError(
                f"axis {axis} reactive-plane link failed qualification: "
                f"support={record['initial_support']['rms_absolute_error']:.3e}, "
                f"CI={record['inner_ci_support']['rms_absolute_error']:.3e}"
            )
        records.append(record)
        print(
            f"[links] axis {axis}: support="
            f"{record['initial_support']['rms_absolute_error']:.3e}, CI="
            f"{record['inner_ci_support']['rms_absolute_error']:.3e}, "
            f"sheet range=[{np.min(sheet):.6f}, {np.max(sheet):.6f}]",
            flush=True,
        )
    return (
        {mode: tuple(fields) for mode, fields in links.items()},
        {
            "construction": (
                "common deterministic R/phi/q16 endpoint-overlap magnitude TTs "
                "broadcast over q3/q10; "
                "GP alone carries the clean inner-CI Z2 phase"
            ),
            "phase": phase_records,
            "magnitude": records,
            "maximum_gp_ngp_magnitude_mismatch": float(
                max(record["gp_ngp_max_magnitude_mismatch"] for record in records)
            ),
        },
    )


def _encoded_loop_phases(axes, link_cores):
    """Measure the two diagnostic Wilson loops from the encoded TT links."""
    shape = tuple(len(axis) for axis in axes)
    equilibrium = PhenolReactiveChart().equilibrium
    spectators = tuple(
        nearest(axes[axis], equilibrium[axis]) for axis in range(2, len(axes))
    )
    sheets = []
    for axis in range(2):
        edge_shape = list(shape)
        edge_shape[axis] -= 1
        sheet = np.empty(edge_shape[:2], dtype=complex)
        for index in np.ndindex(*edge_shape[:2]):
            full_index = (*index, *spectators)
            sheet[index] = tt_value(link_cores[axis], full_index)
        sheets.append(sheet[..., None])
    dummy = np.ones((shape[0], shape[1], 0), dtype=complex)
    loops = {
        "inner": ((1.02, -0.15), (1.25, 0.15)),
        "outer": ((1.75, -0.15), (1.98, 0.15)),
    }
    result = {}
    for name, (lower, upper) in loops.items():
        lo = (nearest(axes[0], lower[0]), nearest(axes[1], lower[1]), 0)
        hi = (nearest(axes[0], upper[0]), nearest(axes[1], upper[1]), 0)
        phase, minimum = rectangular_loop_phase((*sheets, dummy), lo, hi)
        result[name] = {
            "phase_radian": phase,
            "minimum_link_magnitude": minimum,
        }
    return result


def _build_fields(args, axes):
    spec = _field_spec(args, axes)
    cached = _load_field_cache(args.field_cache, axes, spec)
    if cached is not None:
        return (*cached, True)
    fit = MACE.load(
        args.checkpoint, PhenolReactiveChart().geometry, device="cpu", distill=False
    )
    radial_correction = RadialMatrixCorrection.load(args.radial_correction)
    energy = CorrectedMatrixField(fit.neural_energy, radial_correction)
    if fit.neural_feature is None:
        raise RuntimeError("the fitted MACE checkpoint does not contain endpoint features")

    def predict_endpoints(points):
        values = fit.predict_covariant(points)
        return {
            "energy": values["energy"] + radial_correction.predict(points),
            "feature": values["feature"],
        }

    oracle = ProjectedS1Oracle(
        axes,
        energy,
        fit.neural_feature,
        prediction_batch_size=args.prediction_batch_size,
        joint_predictor=predict_endpoints,
    )
    phase_links, phase_info = _inner_conical_phase_links(
        axes, energy, prediction_batch_size=args.prediction_batch_size
    )
    shape = tuple(len(axis) for axis in axes)
    compatible = _load_compatible_potential(args.field_cache, axes, spec)
    if compatible is None:
        compatible = _load_compatible_potential(BASE_POTENTIAL_CACHE, axes, spec)
    if compatible is None:
        potential, potential_info = tt_cross(
            shape,
            lambda _index: 0.0,
            batch_evaluator=oracle.potential,
            max_rank=args.potential_tt_rank,
            sweeps=args.field_cross_sweeps,
            rtol=args.field_cross_rtol,
            validation=args.field_cross_validation,
            seed=args.seed + 101,
            start_rank=1,
            kick_rank=2,
        )
        rng = np.random.default_rng(args.seed + 307)
        validation_indices = np.column_stack(
            [rng.integers(0, size, args.potential_validation) for size in shape]
        )
        reference = oracle.potential(validation_indices)
        predicted = np.asarray(
            [tt_value(potential, row) for row in validation_indices]
        ).real
        error = predicted - reference
        potential_info = dict(potential_info)
        potential_info["independent_validation_rms_ev"] = float(
            np.sqrt(np.mean(error**2)) * 27.211386245988
        )
        potential_info["independent_validation_max_ev"] = float(
            np.max(np.abs(error)) * 27.211386245988
        )
        potential_info["cache_reused_after_gauge_change"] = False
    else:
        potential, potential_info = compatible
        potential_info = dict(potential_info)
        potential_info["cache_reused_after_gauge_change"] = True
    links, link_info = _full_overlap_link_cores(
        args, axes, oracle, phase_links
    )
    encoded_loops = {
        mode: _encoded_loop_phases(axes, fields) for mode, fields in links.items()
    }
    reference_loops = {}
    loop_boxes = {
        "inner": ((1.02, -0.15), (1.25, 0.15)),
        "outer": ((1.75, -0.15), (1.98, 0.15)),
    }
    for name, (lower, upper) in loop_boxes.items():
        lo = (nearest(axes[0], lower[0]), nearest(axes[1], lower[1]), 0)
        hi = (nearest(axes[0], upper[0]), nearest(axes[1], upper[1]), 0)
        phase, minimum = rectangular_loop_phase(phase_links, lo, hi)
        reference_loops[name] = {
            "phase_radian": phase,
            "minimum_link_magnitude": minimum,
        }
    loop_error = max(
        abs(
            np.angle(
                np.exp(
                    1j
                    * (
                        encoded_loops["gp"][name]["phase_radian"]
                        - reference_loops[name]["phase_radian"]
                    )
                )
            )
        )
        for name in reference_loops
    )
    if loop_error > 1.0e-9:
        raise RuntimeError(f"TT phase sheet changed a Wilson loop by {loop_error:.3e} rad")
    ngp_loop_error = max(
        abs(encoded_loops["ngp"][name]["phase_radian"])
        for name in reference_loops
    )
    if ngp_loop_error > 1.0e-9:
        raise RuntimeError(
            f"NGP overlap magnitudes acquired a loop phase of {ngp_loop_error:.3e} rad"
        )
    info = {
        "potential": potential_info,
        "links": link_info,
        "reference_loop_phases": reference_loops,
        "encoded_loop_phases": encoded_loops["gp"],
        "encoded_ngp_loop_phases": encoded_loops["ngp"],
        "raw_endpoint_loop_phases": oracle.loop_phases(),
        "maximum_encoded_loop_error_radian": loop_error,
        "maximum_ngp_loop_phase_radian": ngp_loop_error,
        "physical_ci": phase_info,
        "link_control": (
            "identical deterministic R/phi/q16 endpoint-overlap magnitudes in "
            "both branches, broadcast over q3/q10; "
            "NGP removes only the clean inner-CI Z2 link phase"
        ),
        "finite_link_metric": (
            "the same qualified R/phi/q16 approximation is retained in GP and "
            "NGP; no separate DBOC is added"
        ),
        "minimum_spectator_to_planar_reference_overlap": float(
            oracle.minimum_reference_overlap
        ),
        "minimum_direct_link_magnitude_sampled": float(
            oracle.minimum_link_magnitude
        ),
        "energy_reference_hartree": oracle.energy_reference,
    }
    _save_field_cache(args.field_cache, axes, potential, links, spec, info)
    return tuple(potential), links, info, False


def _append_scalar_electronic_site(operator):
    return MPO([*operator.factors, np.ones((1, 1, 1, 1), dtype=complex)])


def _operator_spec(args, mode, axes):
    spec = {
        "field": _file_signature(Path(args.field_cache) / "metadata.json"),
        "keo": _file_signature(Path(args.keo_cache) / "metadata.json"),
        "mode": mode,
        "grid_shape": [len(axis) for axis in axes],
        "overlap_rank": int(args.overlap_rank),
        "operator_rank": int(args.operator_rank),
    }
    residual_metadata = Path(args.potential_residual_cache) / "metadata.json"
    if json.loads(residual_metadata.read_text())["files"]["corrections"]:
        spec["potential_residual"] = _file_signature(residual_metadata)
    return spec


def _build_driver(args, mode, axes, labelled_keo, potential_cores, link_cores):
    cache = args.gp_operator_cache if mode == "gp" else args.ngp_operator_cache
    spec = _operator_spec(args, mode, axes)
    cached = _load_operator_cache(cache, spec, axes)
    if cached is not None:
        driver, _axes, payload = cached
        return driver, payload, True
    models = []
    shape = tuple(len(axis) for axis in axes)
    for axis, cores in enumerate(link_cores):
        edge_shape = list(shape)
        edge_shape[axis] -= 1
        models.append(DiscreteLinkTT(cores, edge_shape))
    kinetic, overlap_info = corewise_link_mpo_components(
        models,
        axes,
        labelled_keo,
        1,
        max_rank=args.overlap_rank,
        operator_rank=args.operator_rank,
        split=True,
    )
    potential = _append_scalar_electronic_site(tt_cores_to_diagonal_mpo(potential_cores))
    driver = TTLDR.from_components(
        (*kinetic, potential),
        grids=axes,
        overlap_info=overlap_info,
        potential_info={"backend": "adiabatic-S1-discrete-TT", "ranks": potential.bond_orders()},
    )
    payload = {"overlap_info": overlap_info, "potential_info": driver.potential_info}
    _save_operator_cache(cache, spec, axes, driver, payload)
    return driver, payload, False


def _initial_state(path, axes, sites):
    nuclear = load_interpolated_state(path, axes, sites=None)
    factors = [nuclear._get_std_B(site).copy() for site in range(nuclear.L)]
    factors.append(np.ones((1, 1, 1), dtype=complex))
    state = MPS(factors, sites=sites).right_canonicalize()
    state.normalize()
    return state


def _state_overlap(left, right):
    environment = np.ones((1, 1), dtype=complex)
    for site in range(left.L):
        a = left._get_std_B(site)
        b = right._get_std_B(site)
        environment = np.einsum("ab,api,bpj->ij", environment, a.conj(), b, optimize=True)
    return environment[0, 0]


def _branch_checkpoint_config(args, mode):
    return {
        "version": 1,
        "mode": mode,
        "field": _file_signature(Path(args.field_cache) / "metadata.json"),
        "potential_residual": _file_signature(
            Path(args.potential_residual_cache) / "metadata.json"
        ),
        "keo": _file_signature(Path(args.keo_cache) / "metadata.json"),
        "initial_state": _file_signature(args.initial_state),
        "dt_fs": float(args.time_fs / args.steps),
        "interval": int(args.interval),
        "state_rank": int(args.state_rank),
        "integrator": args.integrator,
        "tdvp2_warmup_steps": int(args.tdvp2_warmup_steps),
        "cutoff": float(args.cutoff),
        "krylov_dim": int(args.krylov_dim),
        "krylov_tol": float(args.krylov_tol),
        "cap_start": float(args.cap_start),
        "cap_strength": float(args.cap_strength),
        "cap_order": int(args.cap_order),
    }


def _save_branch_checkpoint(path, config, step, state, history, seconds):
    payload = {
        "config_json": np.asarray(json.dumps(config, sort_keys=True)),
        "step": np.asarray(step),
        "factor_count": np.asarray(state.L),
        "times_fs": history["times_fs"],
        "norms": history["norms"],
        "cap_yield": history["cap_yield"],
        "absorbed": history["absorbed"],
        "closure": history["closure"],
        "tdvp_truncation_error": history["tdvp_truncation_error"],
        "tdvp_norm_defect": history["tdvp_norm_defect"],
        "seconds": np.asarray(seconds),
    }
    payload.update(
        {f"factor_{site}": state._get_std_B(site) for site in range(state.L)}
    )
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _load_branch_checkpoint(path, config, sites):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        if not specs_equivalent(json.loads(str(saved["config_json"])), config):
            raise RuntimeError(f"existing branch checkpoint has a different configuration: {path}")
        count = int(saved["factor_count"])
        state = MPS(
            [np.asarray(saved[f"factor_{site}"]) for site in range(count)],
            sites=sites,
        )
        history = {
            key: np.asarray(saved[key])
            for key in (
                "times_fs", "norms", "cap_yield", "absorbed", "closure",
                "tdvp_truncation_error", "tdvp_norm_defect",
            )
        }
        return int(saved["step"]), state, history, float(saved["seconds"])


def _branch_segment(args, completed):
    """Choose a checkpoint-sized TDVP segment, respecting hybrid warmup."""
    segment = min(args.checkpoint_steps, args.steps - completed)
    integrator = args.integrator
    if args.integrator == "hybrid":
        if completed < args.tdvp2_warmup_steps:
            integrator = "tdvp2"
            segment = min(segment, args.tdvp2_warmup_steps - completed)
        else:
            integrator = "tdvp"
    return segment, integrator


def _run_branch(args, mode, driver, initial, axes):
    if args.checkpoint_steps < 1:
        raise ValueError("checkpoint-steps must be positive")
    if args.steps % args.interval or args.checkpoint_steps % args.interval:
        raise ValueError("steps and checkpoint-steps must be multiples of interval")
    if args.integrator == "hybrid":
        if args.tdvp2_warmup_steps % args.interval:
            raise ValueError("tdvp2-warmup-steps must be a multiple of interval")
        if not 0 <= args.tdvp2_warmup_steps <= args.steps:
            raise ValueError("tdvp2-warmup-steps must lie between zero and steps")
    cap = polynomial_cap(axes[0], args.cap_start, args.cap_strength, args.cap_order)
    checkpoint = args.output / f"{mode}_checkpoint.npz"
    config = _branch_checkpoint_config(args, mode)
    restored = _load_branch_checkpoint(checkpoint, config, initial.sites)
    if restored is None:
        completed = 0
        current = initial.copy()
        history = None
        previous_seconds = 0.0
    else:
        completed, current, history, previous_seconds = restored
        if completed > args.steps:
            raise RuntimeError(f"checkpoint step {completed} exceeds requested {args.steps}")
        print(f"[{mode}] resumed checkpoint at step {completed}/{args.steps}", flush=True)
    started = time.perf_counter()
    while completed < args.steps:
        segment, segment_integrator = _branch_segment(args, completed)
        driver.run(
            current,
            dt=float(args.time_fs / args.steps / au2fs),
            steps=segment,
            interval=args.interval,
            max_bond=args.state_rank,
            integrator=segment_integrator,
            cutoff=args.cutoff,
            krylov_dim=args.krylov_dim,
            krylov_tol=args.krylov_tol,
            normalize=False,
            progress=args.progress,
            workers=args.workers,
            absorber=cap,
        )
        local_times = completed * config["dt_fs"] + np.asarray(driver.times) * au2fs
        local_norms = np.asarray(driver.norms)
        local_yield = np.asarray(driver.absorber_yields)[:, 0]
        local_truncation = np.asarray(driver.tdvp_truncation_errors)
        local_norm_defect = np.asarray(driver.tdvp_norm_defects)
        if history is None:
            times = local_times
            norms = local_norms
            cap_yield = local_yield
            tdvp_truncation_error = local_truncation
            tdvp_norm_defect = local_norm_defect
        else:
            times = np.concatenate((history["times_fs"], local_times[1:]))
            norms = np.concatenate((history["norms"], local_norms[1:]))
            cap_yield = np.concatenate(
                (history["cap_yield"], history["cap_yield"][-1] + local_yield[1:])
            )
            tdvp_truncation_error = np.concatenate(
                (
                    history["tdvp_truncation_error"],
                    history["tdvp_truncation_error"][-1] + local_truncation[1:],
                )
            )
            tdvp_norm_defect = np.concatenate(
                (
                    history["tdvp_norm_defect"],
                    history["tdvp_norm_defect"][-1] + local_norm_defect[1:],
                )
            )
        absorbed = norms[0] - norms
        history = {
            "times_fs": times,
            "norms": norms,
            "cap_yield": cap_yield,
            "absorbed": absorbed,
            "closure": cap_yield - absorbed,
            "tdvp_truncation_error": tdvp_truncation_error,
            "tdvp_norm_defect": tdvp_norm_defect,
        }
        current = driver.final_state
        completed += segment
        total_seconds = previous_seconds + time.perf_counter() - started
        _save_branch_checkpoint(
            checkpoint, config, completed, current, history, total_seconds
        )
        print(
            f"[{mode}] checkpoint step {completed}/{args.steps} "
            f"({completed * config['dt_fs']:.2f} fs, {segment_integrator})",
            flush=True,
        )
    marginals = mps_marginals(current)[:5]
    result = {
        "mode": mode,
        **history,
        "marginals": marginals,
        "seconds": previous_seconds + time.perf_counter() - started,
        "final_ranks": current.bond_orders(),
        "final_state": current,
    }
    return result


def _selected_modes(mode):
    if mode == "both":
        return ("gp", "ngp")
    if mode in ("gp", "ngp"):
        return (mode,)
    raise ValueError(f"unknown GP/NGP mode: {mode}")


def _save_results(output, axes, initial_marginals, results, field_info, summary):
    payload = {f"axis_{axis}": values for axis, values in enumerate(axes)}
    payload.update({f"initial_marginal_{axis}": values for axis, values in enumerate(initial_marginals)})
    for result in results:
        mode = result["mode"]
        for key in (
            "times_fs", "norms", "cap_yield", "absorbed", "closure",
            "tdvp_truncation_error", "tdvp_norm_defect",
        ):
            payload[f"{key}_{mode}"] = result[key]
        for axis, values in enumerate(result["marginals"]):
            payload[f"final_marginal_{axis}_{mode}"] = values
        state = result["final_state"]
        for site in range(state.L):
            payload[f"final_factor_{site}_{mode}"] = state._get_std_B(site)
    np.savez_compressed(output / "phenol_5d_gp_ngp.npz", **payload)

    if plt is None:
        (output / "summary.json").write_text(
            json.dumps(_jsonable(summary), indent=2) + "\n"
        )
        return

    figure, panels = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    for result in results:
        mode = result["mode"]
        panels[0, 0].plot(result["times_fs"], 100.0 * result["absorbed"], color=COLORS[mode], label=mode.upper())
        panels[0, 1].plot(result["times_fs"], 100.0 * result["cap_yield"], color=COLORS[mode], label=mode.upper())
        panels[1, 0].plot(axes[0], result["marginals"][0], color=COLORS[mode], label=mode.upper())
    panels[1, 0].plot(axes[0], initial_marginals[0], "--", color="0.35", label="initial")
    loops = field_info["reference_loop_phases"]
    names = list(loops)
    x = np.arange(len(names))
    values = np.asarray([abs(loops[name]["phase_radian"]) / np.pi for name in names])
    panels[1, 1].bar(x - 0.18, values, 0.36, color=COLORS["gp"], label="GP")
    panels[1, 1].bar(x + 0.18, np.zeros_like(values), 0.36, color=COLORS["ngp"], label="NGP")
    panels[0, 0].set(xlabel="time (fs)", ylabel="norm loss (%)", title="CAP absorption")
    panels[0, 1].set(xlabel="time (fs)", ylabel="integrated CAP flux (%)", title="Dissociation flux")
    panels[1, 0].set(xlabel=r"$R_{OH}$ ($\mathrm{\AA}$)", ylabel="probability", title="Radial distribution")
    panels[1, 1].set(xticks=x, xticklabels=names, ylabel=r"$|\gamma|/\pi$", ylim=(0.0, 1.08), title="Wilson-loop control")
    for label, panel in zip("abcd", panels.flat):
        panel.grid(alpha=0.18)
        panel.legend(frameon=False)
        panel.text(0.02, 0.96, label, transform=panel.transAxes, va="top", fontweight="bold")
    figure.savefig(output / "phenol_5d_gp_ngp.png", dpi=350)
    figure.savefig(output / "phenol_5d_gp_ngp.pdf")
    plt.close(figure)
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")


def _save_setup_diagnostics(output, axes, potential, links, initial_marginals, field_info):
    if plt is None:
        return
    equilibrium = np.asarray(PhenolReactiveChart().equilibrium)
    fixed = [nearest(axis, value) for axis, value in zip(axes, equilibrium)]
    radial_indices = np.asarray(
        [[index, *fixed[1:]] for index in range(len(axes[0]))], dtype=int
    )
    radial_energy = 27.211386245988 * np.asarray(
        [tt_value(potential, row) for row in radial_indices]
    ).real
    angular_phase = np.empty((len(axes[0]), len(axes[1]) - 1))
    for index in np.ndindex(angular_phase.shape):
        angular_phase[index] = np.angle(
            tt_value(links["gp"][1], (*index, 0, 0, 0))
        )

    figure, panels = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    panels[0, 0].plot(axes[0], radial_energy, color="0.15")
    mesh = panels[0, 1].pcolormesh(
        0.5 * (axes[1][:-1] + axes[1][1:]), axes[0], angular_phase,
        shading="auto", cmap="twilight", vmin=-np.pi, vmax=np.pi,
    )
    figure.colorbar(mesh, ax=panels[0, 1], label=r"$\arg U_R$")
    panels[1, 0].plot(axes[0], initial_marginals[0], color="0.25")
    loops = field_info["encoded_loop_phases"]
    names = list(loops)
    x = np.arange(len(names))
    gp = [abs(loops[name]["phase_radian"]) / np.pi for name in names]
    panels[1, 1].bar(x - 0.18, gp, 0.36, color=COLORS["gp"], label="GP")
    panels[1, 1].bar(x + 0.18, np.zeros(len(names)), 0.36, color=COLORS["ngp"], label="NGP")
    panels[0, 0].set(
        xlabel=r"$R_{OH}$ ($\mathrm{\AA}$)", ylabel=r"$E_1-E_1(q_0)$ (eV)",
        title="Compressed adiabatic S1 cut",
    )
    panels[0, 1].set(
        xlabel=r"$\phi$ (rad)", ylabel=r"$R_{OH}$ ($\mathrm{\AA}$)",
        title="Encoded angular Berry-link phase",
    )
    panels[1, 0].set(
        xlabel=r"$R_{OH}$ ($\mathrm{\AA}$)", ylabel="probability",
        title="Initial nuclear envelope",
    )
    panels[1, 1].set(
        xticks=x, xticklabels=names, ylabel=r"$|\gamma|/\pi$", ylim=(0.0, 1.08),
        title="Matched topological control",
    )
    for label, panel in zip("abcd", panels.flat):
        panel.grid(alpha=0.18)
        panel.text(0.02, 0.96, label, transform=panel.transAxes, va="top", fontweight="bold")
    panels[1, 1].legend(frameon=False)
    figure.savefig(output / "phenol_5d_gp_ngp_setup.png", dpi=350)
    figure.savefig(output / "phenol_5d_gp_ngp_setup.pdf")
    plt.close(figure)

    figure, panels = plt.subplots(2, 3, figsize=(11.0, 6.6), constrained_layout=True)
    direction_names = (
        r"$R_{OH}$ link",
        r"$\phi$ link",
        r"$q_3$ link",
        r"$q_{10}$ link",
        r"$q_{16}$ link",
    )
    shape = tuple(len(axis) for axis in axes)
    for direction, panel in enumerate(panels.flat[:5]):
        if direction == 0:
            coordinate = 0.5 * (axes[0][:-1] + axes[0][1:])
            indices = np.repeat(np.asarray(fixed)[None], len(coordinate), axis=0)
            indices[:, 0] = np.arange(len(coordinate))
        else:
            coordinate = axes[0]
            indices = np.repeat(np.asarray(fixed)[None], len(coordinate), axis=0)
            indices[:, 0] = np.arange(len(coordinate))
            indices[:, direction] = min(fixed[direction], shape[direction] - 2)
        gp = np.abs(
            np.asarray([tt_value(links["gp"][direction], row) for row in indices])
        )
        ngp = np.abs(
            np.asarray([tt_value(links["ngp"][direction], row) for row in indices])
        )
        panel.plot(coordinate, gp, color=COLORS["gp"], label="GP")
        panel.plot(
            coordinate, ngp, "--", color=COLORS["ngp"], linewidth=1.3, label="NGP"
        )
        panel.set(
            xlabel=r"$R_{OH}$ ($\mathrm{\AA}$)",
            ylabel=r"$|U(q,q+\Delta q)|$",
            title=direction_names[direction],
            ylim=(-0.03, 1.04),
        )
        panel.grid(alpha=0.18)
    records = field_info["links"]["magnitude"]
    x = np.arange(len(records))
    support = [record["initial_support"]["rms_absolute_error"] for record in records]
    ci = [record["inner_ci_support"]["rms_absolute_error"] for record in records]
    panels[1, 2].bar(x - 0.18, support, 0.36, color="#009E73", label="initial support")
    panels[1, 2].bar(x + 0.18, ci, 0.36, color="#CC79A7", label="inner CI")
    panels[1, 2].set(
        xticks=x,
        xticklabels=(r"$R$", r"$\phi$", r"$q_3$", r"$q_{10}$", r"$q_{16}$"),
        ylabel="overlap-magnitude RMS error",
        title="5D qualification",
    )
    panels[1, 2].grid(axis="y", alpha=0.18)
    panels[0, 0].legend(frameon=False)
    panels[1, 2].legend(frameon=False)
    for label, panel in zip("abcdef", panels.flat):
        panel.text(
            0.02, 0.96, label, transform=panel.transAxes,
            va="top", fontweight="bold",
        )
    figure.savefig(output / "phenol_5d_gp_ngp_link_magnitudes.png", dpi=350)
    figure.savefig(output / "phenol_5d_gp_ngp_link_magnitudes.pdf")
    plt.close(figure)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    modes = _selected_modes(args.mode)
    axes, labelled_keo, keo_metadata = _load_keo(args.keo_cache)
    print(
        "[fields] building/loading adiabatic S1 and full-magnitude link TTs",
        flush=True,
    )
    potential, links, field_info, field_hit = _build_fields(args, axes)
    print("[potential] building/loading support-qualified residual", flush=True)
    potential, residual_info, residual_hit = _build_potential_residual(
        args, axes, potential
    )
    field_info = dict(field_info)
    field_info["residual"] = residual_info
    drivers = {}
    operator_info = {}
    for mode in modes:
        drivers[mode], payload, hit = _build_driver(
            args, mode, axes, labelled_keo, potential, links[mode]
        )
        operator_info[mode] = {"cache_hit": hit, "operator_ranks": drivers[mode].operator_ranks, "payload": payload}

    initial = _initial_state(
        args.initial_state, axes, drivers[modes[0]].components[0].input_sites
    )
    initial_marginals = mps_marginals(initial)[:5]
    results = []
    if not args.setup_only:
        for mode in modes:
            print(f"[{mode}] starting matched projected-S1 propagation", flush=True)
            results.append(_run_branch(args, mode, drivers[mode], initial, axes))
    final_fidelity = None
    by_mode = {result["mode"]: result for result in results}
    if set(by_mode) == {"gp", "ngp"}:
        overlap = _state_overlap(
            by_mode["gp"]["final_state"], by_mode["ngp"]["final_state"]
        )
        denominator = (
            by_mode["gp"]["final_state"].norm_squared()
            * by_mode["ngp"]["final_state"].norm_squared()
        )
        final_fidelity = float(abs(overlap) ** 2 / denominator)
    summary = {
        "method": "matched projected-S1 discrete-link GP/NGP control",
        "fidelity": (
            "adaptation; scalar adiabatic S1 control retaining the same finite-link "
            "metric/DBOC approximation in both branches, with NGP removing only "
            "link phase; the magnitude is resolved over R/phi/q16 and broadcast "
            "over q3/q10; not a reproduction of the cited calculations"
        ),
        "references": [
            "C. Xie et al., J. Am. Chem. Soc. 138, 7828-7831 (2016), DOI: 10.1021/jacs.6b03288",
            "C. Xie and H. Guo, Chem. Phys. Lett. 683, 222-227 (2017), DOI: 10.1016/j.cplett.2017.02.026",
        ],
        "grid_shape": [len(axis) for axis in axes],
        "grid_bounds": [[float(axis[0]), float(axis[-1])] for axis in axes],
        "initial_state": str(args.initial_state),
        "initial_boundary_probabilities": [float(values[0] + values[-1]) for values in initial_marginals],
        "keo": {"cache": str(args.keo_cache), "info": keo_metadata["keo_info"]},
        "fields": {
            "cache_hit": field_hit,
            "residual_cache_hit": residual_hit,
            "info": field_info,
        },
        "operators": operator_info,
        "dynamics": {
            "modes": list(modes),
            "setup_only": bool(args.setup_only),
            "time_fs": float(args.time_fs),
            "steps": int(args.steps),
            "checkpoint_steps": int(args.checkpoint_steps),
            "state_rank": int(args.state_rank),
            "integrator": args.integrator,
            "tdvp2_warmup_steps": int(args.tdvp2_warmup_steps),
            "cap_start_angstrom": float(args.cap_start),
            "cap_strength_hartree": float(args.cap_strength),
            "results": [
                {
                    "mode": result["mode"],
                    "final_norm": float(result["norms"][-1]),
                    "final_cap_yield": float(result["cap_yield"][-1]),
                    "maximum_closure_defect": float(np.max(np.abs(result["closure"]))),
                    "cumulative_tdvp_truncation_error": float(
                        result["tdvp_truncation_error"][-1]
                    ),
                    "cumulative_tdvp_norm_defect": float(
                        result["tdvp_norm_defect"][-1]
                    ),
                    "seconds": float(result["seconds"]),
                    "final_ranks": result["final_ranks"],
                }
                for result in results
            ],
            "final_gp_ngp_fidelity": final_fidelity,
        },
    }
    _save_setup_diagnostics(
        args.output, axes, potential, links, initial_marginals, field_info
    )
    if results:
        _save_results(args.output, axes, initial_marginals, results, field_info, summary)
    else:
        (args.output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print(json.dumps(_jsonable(summary), indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--radial-correction", type=Path, default=DEFAULT_RADIAL_CORRECTION)
    parser.add_argument("--keo-cache", type=Path, default=DEFAULT_KEO_CACHE)
    parser.add_argument("--initial-state", type=Path, default=DEFAULT_INITIAL_STATE)
    parser.add_argument("--field-cache", type=Path, default=DEFAULT_FIELD_CACHE)
    parser.add_argument(
        "--potential-residual-cache", type=Path, default=DEFAULT_RESIDUAL_CACHE
    )
    parser.add_argument("--gp-operator-cache", type=Path, default=DEFAULT_GP_OPERATOR_CACHE)
    parser.add_argument("--ngp-operator-cache", type=Path, default=DEFAULT_NGP_OPERATOR_CACHE)
    parser.add_argument("--potential-tt-rank", type=int, default=24)
    parser.add_argument("--potential-validation", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=256)
    parser.add_argument("--potential-residual-levels", type=int, default=1)
    parser.add_argument("--potential-residual-rank", type=int, default=32)
    parser.add_argument("--potential-residual-sweeps", type=int, default=13)
    parser.add_argument("--potential-residual-validation", type=int, default=256)
    parser.add_argument("--potential-residual-support-samples", type=int, default=2048)
    parser.add_argument("--potential-residual-guard-samples", type=int, default=1024)
    parser.add_argument("--potential-residual-target-ev", type=float, default=0.012)
    parser.add_argument("--potential-residual-guard-target-ev", type=float, default=0.025)
    parser.add_argument("--potential-residual-discrete-levels", type=int, default=0)
    parser.add_argument("--potential-residual-discrete-rank", type=int, default=12)
    parser.add_argument(
        "--potential-residual-discrete-samples", type=int, default=8192
    )
    parser.add_argument(
        "--potential-residual-discrete-guard-samples", type=int, default=8192
    )
    parser.add_argument("--potential-residual-discrete-sweeps", type=int, default=12)
    parser.add_argument(
        "--potential-residual-discrete-regularization", type=float, default=1.0e-6
    )
    parser.add_argument("--potential-residual-weighted-levels", type=int, default=0)
    parser.add_argument("--potential-residual-weighted-rank", type=int, default=16)
    parser.add_argument("--potential-residual-weighted-degree", type=int, default=8)
    parser.add_argument(
        "--potential-residual-weighted-samples", type=int, default=8192
    )
    parser.add_argument(
        "--potential-residual-weighted-guard-samples", type=int, default=4096
    )
    parser.add_argument("--potential-residual-weighted-sweeps", type=int, default=12)
    parser.add_argument(
        "--potential-residual-weighted-regularization", type=float, default=1.0e-8
    )
    parser.add_argument("--field-cross-sweeps", type=int, default=13)
    parser.add_argument("--field-cross-rtol", type=float, default=1.0e-4)
    parser.add_argument("--field-cross-validation", type=int, default=128)
    parser.add_argument("--link-validation", type=int, default=1024)
    parser.add_argument("--link-support-validation", type=int, default=2048)
    parser.add_argument("--link-ci-validation", type=int, default=2048)
    parser.add_argument("--link-tensor-rank", type=int, default=32)
    parser.add_argument("--link-tensor-rtol", type=float, default=1.0e-8)
    parser.add_argument("--link-support-rms-target", type=float, default=5.0e-2)
    parser.add_argument("--link-ci-rms-target", type=float, default=9.0e-2)
    parser.add_argument("--overlap-rank", type=int, default=24)
    parser.add_argument("--operator-rank", type=int, default=48)
    parser.add_argument("--state-rank", type=int, default=24)
    parser.add_argument(
        "--mode", choices=("both", "gp", "ngp"), default="both",
        help="propagate both controls or one independently (for cluster arrays)",
    )
    parser.add_argument("--time-fs", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--checkpoint-steps", type=int, default=20)
    parser.add_argument(
        "--integrator", choices=("tdvp", "tdvp2", "hybrid"), default="hybrid"
    )
    parser.add_argument("--tdvp2-warmup-steps", type=int, default=5)
    parser.add_argument("--cutoff", type=float, default=1.0e-10)
    parser.add_argument("--krylov-dim", type=int, default=8)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cap-start", type=float, default=2.45)
    parser.add_argument("--cap-strength", type=float, default=0.02)
    parser.add_argument("--cap-order", type=int, default=4)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
