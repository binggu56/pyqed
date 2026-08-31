#!/usr/bin/env python3
"""Real CASCI full-DVR reference for the two-coordinate pyrazine CGLDR test.

The calculation uses CASCI(4e,4o)/STO-3G at every point of a two-dimensional
normal-coordinate DVR. A six-root window tracks the target reference states
through energy-order crossings. The full reference retains direct CASCI
overlaps along both DVR axes. CGLDR either uses a single local quadratic
expansion at ``Q_coupling = 0`` or a separable piecewise-Hermite expansion
with selectable boundary or interior outer anchors. Interior placement reserves
the boundary points for validation and uses linear extrapolation there.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import expm_multiply

from examples.qchem.mcscf.pyrazine_lvc_from_casci import (
    PYRAZINE_GEOMETRY_BOHR,
    build_pyrazine,
)
from pyqed.dvr import DVR, HermiteDVR
from pyqed.ldr import CGLDR, ElectronicPartition, SeparableHamiltonian
from pyqed.ldr.coarse_grained import CGLDRElectronicData
from pyqed.ldr.observables import (
    mps_to_array,
    nuclear_density_distance,
    nuclear_observables,
)
from pyqed.mps.mps import MPS
from pyqed.qchem.mcscf.casci import CASCI, overlap
from pyqed.units import (
    amu_to_au,
    au2fs,
    au2wavenumber,
    wavenumber2hartree,
)


STATE_IDS = (1, 2)
COORDINATE_NAMES = ("Q_tuning", "Q_coupling")
_CASCI_WORKER_REFERENCE = None
_CASCI_WORKER_OPTIONS = None


@dataclass(frozen=True)
class SelectedModes:
    """Two dimensionless Cartesian displacement modes."""

    displacements: np.ndarray
    frequencies: np.ndarray
    hessian_indices: np.ndarray
    coupling_strengths: np.ndarray
    tuning_strengths: np.ndarray

    def __post_init__(self):
        displacements = np.asarray(self.displacements, dtype=float)
        frequencies = np.asarray(self.frequencies, dtype=float)
        indices = np.asarray(self.hessian_indices, dtype=int)
        coupling = np.asarray(self.coupling_strengths, dtype=float)
        tuning = np.asarray(self.tuning_strengths, dtype=float)
        if displacements.ndim != 3 or displacements.shape[0] != 2:
            raise ValueError("displacements must have shape (2, natom, 3)")
        if frequencies.shape != (2,) or np.any(frequencies <= 0.0):
            raise ValueError("frequencies must contain two positive values")
        if indices.shape != (2,):
            raise ValueError("hessian_indices must have shape (2,)")
        if coupling.shape != (2,) or tuning.shape != (2,):
            raise ValueError("mode strengths must have shape (2,)")
        object.__setattr__(self, "displacements", displacements)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "hessian_indices", indices)
        object.__setattr__(self, "coupling_strengths", coupling)
        object.__setattr__(self, "tuning_strengths", tuning)

    def to_npz(self, filename):
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            filename,
            displacements=self.displacements,
            frequencies=self.frequencies,
            hessian_indices=self.hessian_indices,
            coupling_strengths=self.coupling_strengths,
            tuning_strengths=self.tuning_strengths,
        )

    @classmethod
    def from_npz(cls, filename):
        with np.load(filename) as archive:
            return cls(
                displacements=archive["displacements"],
                frequencies=archive["frequencies"],
                hessian_indices=archive["hessian_indices"],
                coupling_strengths=archive["coupling_strengths"],
                tuning_strengths=archive["tuning_strengths"],
            )


@dataclass(frozen=True)
class TrackedCASCIPoint:
    """A CASCI point plus roots matched to the target reference states."""

    casci: object
    state_ids: tuple[int, ...]
    reference_overlaps: np.ndarray

    def __post_init__(self):
        state_ids = tuple(int(state_id) for state_id in self.state_ids)
        strengths = np.asarray(self.reference_overlaps, dtype=float)
        if len(state_ids) != len(STATE_IDS) or len(set(state_ids)) != len(state_ids):
            raise ValueError("Tracked points require two unique target roots.")
        if strengths.shape != (len(STATE_IDS),):
            raise ValueError("reference_overlaps must have shape (2,).")
        object.__setattr__(self, "state_ids", state_ids)
        object.__setattr__(self, "reference_overlaps", strengths)


def reference_geometry():
    return np.asarray([row[1:] for row in PYRAZINE_GEOMETRY_BOHR], dtype=float)


def run_casci_point(
    geometry,
    *,
    basis="sto-3g",
    eri="factors",
    nstates=3,
    orbital_reference=None,
):
    """Run an independent RHF/CASCI(4e,4o) point calculation."""
    mol = build_pyrazine(basis, eri, coords=geometry)
    mf = mol.RHF().run()
    mo_coeff = None
    if orbital_reference is not None:
        cross_overlap = asymmetric_ao_overlap(
            orbital_reference.mol,
            mol,
        )
        mo_coeff, _matched_overlaps = match_orbitals(
            orbital_reference.mo_coeff,
            mf.mo_coeff,
            cross_overlap,
        )
    return CASCI(mf, ncas=4, nelecas=4).run(
        nstates=nstates,
        mo_coeff=mo_coeff,
    )


def asymmetric_ao_overlap(left_mol, right_mol):
    from pyqed.qchem.hf.rhf import _cross_ao_overlap_matrix

    return _cross_ao_overlap_matrix(left_mol, right_mol)


def match_orbitals(reference_coeff, current_coeff, cross_ao_overlap):
    """Reorder and phase current MOs by maximum overlap with reference MOs."""
    reference_coeff = np.asarray(reference_coeff)
    current_coeff = np.asarray(current_coeff)
    cross_ao_overlap = np.asarray(cross_ao_overlap)
    mo_overlap = (
        reference_coeff.T.conj()
        @ cross_ao_overlap
        @ current_coeff
    )
    reference_indices, current_indices = linear_sum_assignment(
        -np.abs(mo_overlap)
    )
    order = current_indices[np.argsort(reference_indices)]
    tracked = current_coeff[:, order].astype(
        np.result_type(current_coeff, complex),
        copy=True,
    )
    diagonal = mo_overlap[np.arange(len(order)), order]
    nonzero = np.abs(diagonal) > 0.0
    tracked[:, nonzero] *= np.exp(-1j * np.angle(diagonal[nonzero]))
    if np.allclose(tracked.imag, 0.0, atol=1.0e-14):
        tracked = tracked.real
    return tracked, np.abs(diagonal)


def select_state_ids(reference_to_point_overlap, target_ids=STATE_IDS):
    """Assign unique point roots to target reference roots by overlap."""
    overlaps = np.abs(np.asarray(reference_to_point_overlap))
    target_ids = tuple(int(state_id) for state_id in target_ids)
    if overlaps.ndim != 2:
        raise ValueError("reference_to_point_overlap must be a matrix")
    if not target_ids or max(target_ids) >= overlaps.shape[0]:
        raise ValueError("target reference root is absent from overlap matrix")
    target_overlaps = overlaps[np.asarray(target_ids)]
    target_rows, point_roots = linear_sum_assignment(-target_overlaps)
    point_roots = point_roots[np.argsort(target_rows)]
    strengths = target_overlaps[
        np.arange(len(target_ids)),
        point_roots,
    ]
    return tuple(int(root) for root in point_roots), strengths


def track_casci_states(reference, point):
    state_ids, strengths = select_state_ids(overlap(reference, point))
    return TrackedCASCIPoint(point, state_ids, strengths)


def _initialize_casci_worker(
    geometry,
    basis,
    eri,
    root_window,
):
    global _CASCI_WORKER_REFERENCE, _CASCI_WORKER_OPTIONS
    _CASCI_WORKER_REFERENCE = run_casci_point(
        geometry,
        basis=basis,
        eri=eri,
        nstates=root_window,
    )
    _CASCI_WORKER_OPTIONS = {
        "basis": basis,
        "eri": eri,
        "nstates": root_window,
    }


def _run_casci_worker(task):
    index, geometry = task
    if _CASCI_WORKER_REFERENCE is None or _CASCI_WORKER_OPTIONS is None:
        raise RuntimeError("CASCI worker was not initialized")
    point = run_casci_point(
        geometry,
        **_CASCI_WORKER_OPTIONS,
        orbital_reference=_CASCI_WORKER_REFERENCE,
    )
    tracked = track_casci_states(_CASCI_WORKER_REFERENCE, point)
    energies = np.asarray(point.e_tot)[list(tracked.state_ids)]
    return index, tracked, energies


def pyscf_normal_modes(geometry, *, basis="sto-3g"):
    """Return positive-frequency dimensionless RHF normal modes."""
    try:
        from pyscf import gto, scf
        from pyscf.hessian import thermo
    except ImportError as exc:
        raise ImportError(
            "Selecting pyrazine normal modes requires PySCF."
        ) from exc

    atoms = [
        [row[0], *xyz]
        for row, xyz in zip(PYRAZINE_GEOMETRY_BOHR, geometry)
    ]
    mol = gto.M(atom=atoms, unit="Bohr", basis=basis, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    hessian = mf.Hessian().kernel()
    analysis = thermo.harmonic_analysis(
        mol,
        hessian,
        imaginary_freq=False,
    )
    frequencies_cm1 = np.asarray(
        analysis["freq_wavenumber"],
        dtype=float,
    )
    modes = np.asarray(analysis["norm_mode"], dtype=float)
    positive = np.flatnonzero(
        np.isfinite(frequencies_cm1) & (frequencies_cm1 > 0.0)
    )
    frequencies = frequencies_cm1[positive] * wavenumber2hartree
    modes = modes[positive]
    dimensionless = (
        modes
        / np.sqrt(amu_to_au * frequencies)[:, None, None]
    )
    return dimensionless, frequencies, positive


def select_casci_modes(
    casci,
    geometry,
    *,
    basis="sto-3g",
):
    """Select distinct modes with the strongest CASCI coupling and tuning."""
    modes, frequencies, hessian_indices = pyscf_normal_modes(
        geometry,
        basis=basis,
    )
    first, _second = casci.vibronic_couplings(
        state_ids=STATE_IDS,
        modes=modes,
    )
    coupling = np.abs(first[0, 1])
    tuning = 0.5 * np.abs(first[1, 1] - first[0, 0])
    coupling_local = int(np.argmax(coupling))
    tuning_candidates = np.arange(len(modes))
    tuning_candidates = tuning_candidates[tuning_candidates != coupling_local]
    tuning_local = int(tuning_candidates[np.argmax(tuning[tuning_candidates])])

    # CGLDR samples the tuning mode and expands the coupling mode.
    selected = np.asarray([tuning_local, coupling_local], dtype=int)
    return SelectedModes(
        displacements=modes[selected],
        frequencies=frequencies[selected],
        hessian_indices=hessian_indices[selected],
        coupling_strengths=coupling[selected],
        tuning_strengths=tuning[selected],
    )


def load_or_select_modes(
    reference,
    geometry,
    *,
    basis="sto-3g",
    cache=None,
    force=False,
):
    cache = None if cache is None else Path(cache)
    if cache is not None and cache.exists() and not force:
        return SelectedModes.from_npz(cache)
    modes = select_casci_modes(reference, geometry, basis=basis)
    if cache is not None:
        modes.to_npz(cache)
    return modes


def geometry_at(reference, modes, coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    modes = np.asarray(modes, dtype=float)
    if modes.ndim != 3:
        raise ValueError("modes must have shape (nmodes, natom, 3)")
    if coordinates.shape != (len(modes),):
        raise ValueError(
            f"coordinates must have shape ({len(modes)},)"
        )
    return np.asarray(reference) + np.einsum(
        "m,mAx->Ax",
        coordinates,
        modes,
        optimize=True,
    )


def build_dvr(
    selected_modes,
    *,
    npts=(8, 5),
):
    if npts[1] % 2 != 1:
        raise ValueError(
            "The coupling DVR needs an odd number of points so Q_coupling=0 "
            "is an electronic expansion line."
        )
    axes = tuple(
        HermiteDVR(
            npts=count,
            mass=1.0 / omega,
            omega=omega,
            center=0.0,
        )
        for count, omega in zip(npts, selected_modes.frequencies)
    )
    return DVR.from_axes(
        axes,
        names=COORDINATE_NAMES,
    )


def _polar_unitary(matrix):
    u, _singular_values, vh = np.linalg.svd(
        np.asarray(matrix, dtype=complex),
        full_matrices=False,
    )
    return u @ vh


def transport_operator(operator, overlap_matrix):
    """Transport one Hermitian operator through a CASCI overlap block."""
    frame = _polar_unitary(overlap_matrix)
    transported = frame @ operator @ frame.conj().T
    return 0.5 * (transported + transported.conj().T)


def parallel_transport_frame(overlaps, reference, target):
    """Accumulate nearest-neighbor polar links from ``target`` to ``reference``."""
    overlaps = np.asarray(overlaps, dtype=complex)
    reference = int(reference)
    target = int(target)
    frame = np.eye(overlaps.shape[-1], dtype=complex)
    singular_values = []
    if target == reference:
        return frame, np.ones(overlaps.shape[-1])
    step = 1 if target > reference else -1
    current = reference
    while current != target:
        following = current + step
        link_overlap = overlaps[current, following]
        singular_values.extend(
            np.linalg.svd(link_overlap, compute_uv=False)
        )
        frame = frame @ _polar_unitary(link_overlap)
        current = following
    return frame, np.asarray(singular_values)


def piecewise_cubic_hermite_field(
    coordinates,
    anchor_coordinates,
    hamiltonians,
    gradients,
    *,
    extrapolation="linear",
):
    """Materialize the separable Hermite expansion for validation."""
    return SeparableHamiltonian.cubic_hermite(
        coordinates,
        anchor_coordinates,
        hamiltonians,
        gradients,
        extrapolation=extrapolation,
    ).evaluate()


def coupling_anchor_indices(dvr, count, *, placement="boundary"):
    """Choose center and outer Hermite-DVR expansion anchors."""
    count = int(count)
    if count not in (1, 2, 3):
        raise ValueError("count must be 1, 2, or 3")
    if placement not in ("boundary", "interior"):
        raise ValueError("placement must be 'boundary' or 'interior'")
    center = int(np.argmin(np.abs(dvr.x[1])))
    if count == 1:
        return (center,)
    if placement == "interior" and dvr.shape[1] < 5:
        raise ValueError(
            "multi-anchor interpolation requires at least five coupling "
            "DVR points so the outer grid points remain validation points"
        )
    outer = (
        (0, dvr.shape[1] - 1)
        if placement == "boundary"
        else (1, dvr.shape[1] - 2)
    )
    return outer if count == 2 else (outer[0], center, outer[1])


def two_anchor_hermite_field(
    coordinates,
    anchor_coordinates,
    hamiltonians,
    gradients,
    hessians=None,
):
    """Interpolate two matrix Taylor jets with a Hermite polynomial."""
    coordinates = np.asarray(coordinates, dtype=float)
    anchors = np.asarray(anchor_coordinates, dtype=float)
    if anchors.shape != (2,) or not anchors[0] < anchors[1]:
        raise ValueError("anchor_coordinates must contain two ordered values")
    hamiltonians = np.asarray(hamiltonians, dtype=complex)
    gradients = np.asarray(gradients, dtype=complex)
    if (
        hamiltonians.shape != gradients.shape
        or hamiltonians.shape[-3] != 2
        or hamiltonians.shape[-1] != hamiltonians.shape[-2]
    ):
        raise ValueError(
            "anchor jets must have matching shape "
            "(*sampled, 2, nstates, nstates)"
        )

    interval = anchors[1] - anchors[0]
    reduced = (coordinates - anchors[0]) / interval
    t2 = reduced**2
    t3 = t2 * reduced
    if hessians is None:
        basis = np.stack((
            2.0 * t3 - 3.0 * t2 + 1.0,
            t3 - 2.0 * t2 + reduced,
            -2.0 * t3 + 3.0 * t2,
            t3 - t2,
        ))
        values = np.stack((
            hamiltonians[..., 0, :, :],
            interval * gradients[..., 0, :, :],
            hamiltonians[..., 1, :, :],
            interval * gradients[..., 1, :, :],
        ))
    else:
        hessians = np.asarray(hessians, dtype=complex)
        if hamiltonians.shape != hessians.shape:
            raise ValueError(
                "anchor Hessians must match the Hamiltonian shape"
            )
        t4 = t3 * reduced
        t5 = t4 * reduced
        basis = np.stack((
            1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5,
            reduced - 6.0 * t3 + 8.0 * t4 - 3.0 * t5,
            0.5 * (t2 - 3.0 * t3 + 3.0 * t4 - t5),
            10.0 * t3 - 15.0 * t4 + 6.0 * t5,
            -4.0 * t3 + 7.0 * t4 - 3.0 * t5,
            0.5 * (t3 - 2.0 * t4 + t5),
        ))
        values = np.stack((
            hamiltonians[..., 0, :, :],
            interval * gradients[..., 0, :, :],
            interval**2 * hessians[..., 0, :, :],
            hamiltonians[..., 1, :, :],
            interval * gradients[..., 1, :, :],
            interval**2 * hessians[..., 1, :, :],
        ))
    sampled_ndim = hamiltonians.ndim - 3
    field = np.einsum(
        "kq,k...ab->...qab",
        basis,
        values,
        optimize=True,
    )
    field = 0.5 * (field + field.swapaxes(-1, -2).conj())
    expected = (
        *hamiltonians.shape[:sampled_ndim],
        coordinates.size,
        hamiltonians.shape[-2],
        hamiltonians.shape[-1],
    )
    if field.shape != expected:
        raise RuntimeError(
            f"interpolated field shape {field.shape} != {expected}"
        )
    return field


def retained_overlap(left, right, *, unitarize=False):
    left_casci = left.casci if isinstance(left, TrackedCASCIPoint) else left
    right_casci = right.casci if isinstance(right, TrackedCASCIPoint) else right
    left_ids = left.state_ids if isinstance(left, TrackedCASCIPoint) else STATE_IDS
    right_ids = (
        right.state_ids if isinstance(right, TrackedCASCIPoint) else STATE_IDS
    )
    block = np.asarray(overlap(left_casci, right_casci), dtype=complex)[
        np.ix_(left_ids, right_ids)
    ]
    return _polar_unitary(block) if unitarize else block


def _line_point_index(axis, fixed, coordinate):
    index = list(fixed)
    index.insert(axis, coordinate)
    return tuple(index)


def all_line_overlaps(points, *, unitarize=False):
    """Compute direct all-pair CASCI overlaps on every DVR coordinate line."""
    points = np.asarray(points, dtype=object)
    shape = points.shape
    output = []
    for axis, count in enumerate(shape):
        other_shape = shape[:axis] + shape[axis + 1 :]
        axis_overlaps = np.empty(
            (*other_shape, count, count, len(STATE_IDS), len(STATE_IDS)),
            dtype=complex,
        )
        for fixed in np.ndindex(*other_shape):
            for bra in range(count):
                axis_overlaps[fixed + (bra, bra)] = np.eye(len(STATE_IDS))
                left = points[_line_point_index(axis, fixed, bra)]
                for ket in range(bra + 1, count):
                    right = points[_line_point_index(axis, fixed, ket)]
                    block = retained_overlap(
                        left,
                        right,
                        unitarize=unitarize,
                    )
                    axis_overlaps[fixed + (bra, ket)] = block
                    axis_overlaps[fixed + (ket, bra)] = block.conj().T
        output.append(axis_overlaps)
    return tuple(output)


def scan_casci_grid(
    dvr,
    reference,
    modes,
    *,
    basis="sto-3g",
    eri="factors",
    unitarize_overlaps=False,
    orbital_reference=None,
    root_window=6,
    workers=1,
):
    """Evaluate independent CASCI objects and direct line overlaps."""
    if not isinstance(workers, (int, np.integer)) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if orbital_reference is None:
        raise ValueError("orbital_reference is required for state tracking")
    points = np.empty(dvr.shape, dtype=object)
    energies = np.empty((*dvr.shape, len(STATE_IDS)))
    tasks = []
    for index in np.ndindex(*dvr.shape):
        coordinates = np.asarray(
            [dvr.x[axis][index[axis]] for axis in range(dvr.ndim)]
        )
        tasks.append((
            index,
            geometry_at(reference, modes, coordinates),
        ))

    if workers == 1:
        for completed, (index, geometry) in enumerate(tasks, start=1):
            point = run_casci_point(
                geometry,
                basis=basis,
                eri=eri,
                nstates=root_window,
                orbital_reference=orbital_reference,
            )
            tracked = track_casci_states(orbital_reference, point)
            points[index] = tracked
            energies[index] = np.asarray(point.e_tot)[list(tracked.state_ids)]
            print(f"[CASCI grid] {completed}/{dvr.size}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_initialize_casci_worker,
            initargs=(reference, basis, eri, root_window),
        ) as executor:
            futures = [
                executor.submit(_run_casci_worker, task)
                for task in tasks
            ]
            for completed, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                index, tracked, point_energies = future.result()
                points[index] = tracked
                energies[index] = point_energies
                print(
                    f"[CASCI grid] {completed}/{dvr.size} "
                    f"({workers} workers)",
                    flush=True,
                )
    line_overlaps = all_line_overlaps(
        points,
        unitarize=unitarize_overlaps,
    )
    return energies, line_overlaps, points


def scan_casci_subset(
    dvr,
    reference,
    modes,
    indices,
    *,
    basis="sto-3g",
    eri="factors",
    orbital_reference=None,
    root_window=6,
    workers=1,
):
    """Evaluate only selected grid points needed to rebuild expansion data."""
    if not isinstance(workers, (int, np.integer)) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if orbital_reference is None:
        raise ValueError("orbital_reference is required for state tracking")
    indices = tuple(tuple(int(value) for value in index) for index in indices)
    points = np.empty(dvr.shape, dtype=object)
    points.fill(None)
    tasks = []
    for index in indices:
        coordinates = np.asarray(
            [dvr.x[axis][index[axis]] for axis in range(dvr.ndim)]
        )
        tasks.append((
            index,
            geometry_at(reference, modes, coordinates),
        ))

    if workers == 1:
        for completed, (index, geometry) in enumerate(tasks, start=1):
            point = run_casci_point(
                geometry,
                basis=basis,
                eri=eri,
                nstates=root_window,
                orbital_reference=orbital_reference,
            )
            points[index] = track_casci_states(orbital_reference, point)
            print(
                f"[CASCI expansion points] {completed}/{len(tasks)}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_initialize_casci_worker,
            initargs=(reference, basis, eri, root_window),
        ) as executor:
            futures = [
                executor.submit(_run_casci_worker, task)
                for task in tasks
            ]
            for completed, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                index, tracked, _point_energies = future.result()
                points[index] = tracked
                print(
                    f"[CASCI expansion points] {completed}/{len(tasks)} "
                    f"({workers} workers)",
                    flush=True,
                )
    return points


def overlap_diagnostics(line_overlaps):
    diagnostics = {}
    for axis, values in enumerate(line_overlaps):
        nearest = np.diagonal(values, offset=1, axis1=-4, axis2=-3)
        nearest = np.moveaxis(nearest, -1, -3)
        singular_values = np.linalg.svd(nearest, compute_uv=False)
        diagnostics[f"axis_{axis}_min_neighbor_singular_value"] = float(
            np.min(singular_values)
        )
        diagnostics[f"axis_{axis}_max_neighbor_singular_value"] = float(
            np.max(singular_values)
        )
    return diagnostics


def build_cgldr_data(
    dvr,
    energies,
    line_overlaps,
    points,
    coupling_mode,
    *,
    energy_zero=None,
    root_window=6,
    coupling_anchors=1,
    anchor_placement="boundary",
):
    """Build a one-, two-, or three-anchor transverse CASCI expansion."""
    if coupling_anchors not in (1, 2, 3):
        raise ValueError("coupling_anchors must be 1, 2, or 3")
    coupling_center = int(np.argmin(np.abs(dvr.x[1])))
    if abs(float(dvr.x[1][coupling_center])) > 1.0e-12:
        raise ValueError("The coupling DVR does not contain Q_coupling=0.")
    if energy_zero is None:
        energy_zero = float(np.min(energies))

    line_energies = np.asarray(energies[:, coupling_center]) - energy_zero
    tuning_line_overlaps = np.asarray(
        line_overlaps[0][coupling_center]
    ).transpose(0, 2, 1, 3)
    metadata = {
        "solver": "CASCI(4e,4o)/STO-3G",
        "state_ids": list(STATE_IDS),
        "root_window": int(root_window),
        "sampled_coordinate": COORDINATE_NAMES[0],
        "expanded_coordinate": COORDINATE_NAMES[1],
        "energy_zero": energy_zero,
        "derivative_source": "CASCI.vibronic_couplings",
        "coupling_anchor_count": int(coupling_anchors),
    }

    if coupling_anchors == 1:
        gradients = np.empty((dvr.shape[0], 1, 2, 2), dtype=complex)
        hessians = np.empty((dvr.shape[0], 1, 1, 2, 2), dtype=complex)
        for tuning_index in range(dvr.shape[0]):
            print(
                f"[CASCI derivatives] {tuning_index + 1}/{dvr.shape[0]}",
                flush=True,
            )
            tracked = points[tuning_index, coupling_center]
            first, second = tracked.casci.vibronic_couplings(
                state_ids=tracked.state_ids,
                modes=np.asarray(coupling_mode)[None, ...],
            )
            gradients[tuning_index, 0] = first[..., 0]
            hessians[tuning_index, 0, 0] = second[..., 0, 0]
        metadata["coupling_anchor_points"] = [0.0]
        return CGLDRElectronicData(
            energies=line_energies,
            overlaps=tuning_line_overlaps,
            hamiltonian_gradients=gradients,
            hamiltonian_hessians=hessians,
            reactive_grids=(np.asarray(dvr.x[0]),),
            metadata=metadata,
        )

    anchor_indices = coupling_anchor_indices(
        dvr,
        coupling_anchors,
        placement=anchor_placement,
    )
    anchor_coordinates = np.asarray(dvr.x[1])[list(anchor_indices)]
    hamiltonians = np.empty(
        (dvr.shape[0], coupling_anchors, 2, 2),
        dtype=complex,
    )
    gradients = np.empty_like(hamiltonians)
    transport_singular_values = []
    coupling_overlaps = np.asarray(line_overlaps[1])
    for tuning_index in range(dvr.shape[0]):
        for anchor_number, anchor_index in enumerate(anchor_indices):
            completed = (
                coupling_anchors * tuning_index + anchor_number + 1
            )
            print(
                f"[CASCI anchor derivatives] {completed}/"
                f"{coupling_anchors * dvr.shape[0]}",
                flush=True,
            )
            tracked = points[tuning_index, anchor_index]
            first, _second = tracked.casci.vibronic_couplings(
                state_ids=tracked.state_ids,
                modes=np.asarray(coupling_mode)[None, ...],
            )
            frame, link_singular_values = parallel_transport_frame(
                coupling_overlaps[tuning_index],
                coupling_center,
                anchor_index,
            )
            transport_singular_values.extend(link_singular_values)
            anchor_hamiltonian = np.diag(
                np.asarray(energies[tuning_index, anchor_index])
                - energy_zero
            )
            hamiltonians[tuning_index, anchor_number] = transport_operator(
                anchor_hamiltonian,
                frame,
            )
            gradients[tuning_index, anchor_number] = transport_operator(
                first[..., 0],
                frame,
            )

    separable_hamiltonian = SeparableHamiltonian.cubic_hermite(
        dvr.x[1],
        anchor_coordinates,
        hamiltonians,
        gradients,
        extrapolation=(
            "linear" if anchor_placement == "interior" else "error"
        ),
    )
    field = separable_hamiltonian.evaluate()
    direct_center = np.zeros((dvr.shape[0], 2, 2))
    states = np.arange(2)
    direct_center[:, states, states] = line_energies
    center_residual = np.linalg.norm(
        field[:, coupling_center] - direct_center,
        axis=(-2, -1),
    )
    direct_field = np.empty_like(field)
    for tuning_index in range(dvr.shape[0]):
        for coupling_index in range(dvr.shape[1]):
            frame, _singular_values = parallel_transport_frame(
                coupling_overlaps[tuning_index],
                coupling_center,
                coupling_index,
            )
            local_hamiltonian = np.diag(
                np.asarray(energies[tuning_index, coupling_index])
                - energy_zero
            )
            direct_field[tuning_index, coupling_index] = (
                transport_operator(local_hamiltonian, frame)
            )
    full_grid_residual = np.linalg.norm(
        field - direct_field,
        axis=(-2, -1),
    )
    boundary_residual = full_grid_residual[:, (0, dvr.shape[1] - 1)]
    interior_residual = full_grid_residual[:, 1:-1]
    metadata.update({
        "coupling_anchor_points": anchor_coordinates.tolist(),
        "interpolation": "piecewise-cubic-Hermite",
        "anchor_policy": f"{anchor_placement}-DVR-points",
        "extrapolation": (
            "linear-outside-anchor-interval"
            if anchor_placement == "interior"
            else "none"
        ),
        "transport": "nearest-neighbor-polar-to-Qc=0",
        "quadratic_anchor_terms": "excluded-in-two-state-subspace",
        "minimum_transport_singular_value": float(
            np.min(transport_singular_values)
        ),
        "maximum_center_hamiltonian_residual": float(
            np.max(center_residual)
        ),
        "maximum_full_grid_hamiltonian_residual": float(
            np.max(full_grid_residual)
        ),
        "maximum_interior_hamiltonian_residual": float(
            np.max(interior_residual)
        ),
        "maximum_boundary_hamiltonian_residual": float(
            np.max(boundary_residual)
        ),
    })
    print(
        f"[{coupling_anchors}-anchor] minimum transport singular value:",
        metadata["minimum_transport_singular_value"],
    )
    print(
        f"[{coupling_anchors}-anchor] maximum center Hamiltonian residual:",
        metadata["maximum_center_hamiltonian_residual"],
    )
    print(
        f"[{coupling_anchors}-anchor] maximum full-grid Hamiltonian residual:",
        metadata["maximum_full_grid_hamiltonian_residual"],
    )
    print(
        f"[{coupling_anchors}-anchor] maximum boundary Hamiltonian residual:",
        metadata["maximum_boundary_hamiltonian_residual"],
    )
    return CGLDRElectronicData(
        energies=line_energies,
        overlaps=tuning_line_overlaps,
        separable_hamiltonian=separable_hamiltonian,
        reactive_grids=(np.asarray(dvr.x[0]),),
        expanded_grids=(np.asarray(dvr.x[1]),),
        metadata=metadata,
    )


def save_full_data(
    filename,
    dvr,
    energies,
    line_overlaps,
    *,
    energy_zero,
    unitarize_overlaps,
    root_window,
    selected_roots=None,
    tracking_strengths=None,
):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "energies": np.asarray(energies) - energy_zero,
        "overlaps_0": line_overlaps[0],
        "overlaps_1": line_overlaps[1],
        "grid_0": dvr.x[0],
        "grid_1": dvr.x[1],
        "state_ids": np.asarray(STATE_IDS),
        "energy_zero": np.asarray(energy_zero),
        "unitarize_overlaps": np.asarray(unitarize_overlaps),
        "root_window": np.asarray(root_window),
    }
    if selected_roots is not None:
        arrays["selected_roots"] = np.asarray(selected_roots, dtype=int)
    if tracking_strengths is not None:
        arrays["tracking_strengths"] = np.asarray(
            tracking_strengths,
            dtype=float,
        )
    np.savez(filename, **arrays)


def load_full_data(filename, dvr, *, unitarize_overlaps, root_window):
    with np.load(filename) as archive:
        for axis in range(2):
            np.testing.assert_allclose(archive[f"grid_{axis}"], dvr.x[axis])
        if tuple(archive["state_ids"]) != STATE_IDS:
            raise ValueError("Cached state_ids do not match (1, 2).")
        if bool(archive["unitarize_overlaps"]) != unitarize_overlaps:
            raise ValueError("Cached overlap unitarization does not match.")
        if int(archive["root_window"]) != root_window:
            raise ValueError("Cached CASCI root window does not match.")
        return (
            np.asarray(archive["energies"]),
            (np.asarray(archive["overlaps_0"]), np.asarray(archive["overlaps_1"])),
            float(archive["energy_zero"]),
        )


def _flat_indices(shape, axis, fixed, nstates):
    indices = np.empty((shape[axis], nstates), dtype=int)
    for coordinate in range(shape[axis]):
        point = list(fixed)
        point.insert(axis, coordinate)
        nuclear = np.ravel_multi_index(tuple(point), shape)
        indices[coordinate] = (
            nstates * nuclear + np.arange(nstates)
        )
    return indices


def build_full_hamiltonian(dvr, energies, line_overlaps):
    """Assemble the full direct-overlap CASCI/LDR Hamiltonian."""
    nstates = energies.shape[-1]
    rows = []
    columns = []
    values = []
    for axis in range(dvr.ndim):
        other_shape = dvr.shape[:axis] + dvr.shape[axis + 1 :]
        kinetic = np.asarray(dvr.axes[axis].t(), dtype=complex)
        for fixed in np.ndindex(*other_shape):
            overlaps_on_line = line_overlaps[axis][fixed]
            block = kinetic[:, :, None, None] * overlaps_on_line
            line = _flat_indices(dvr.shape, axis, fixed, nstates)
            rows.append(np.broadcast_to(
                line[:, None, :, None],
                block.shape,
            ).reshape(-1))
            columns.append(np.broadcast_to(
                line[None, :, None, :],
                block.shape,
            ).reshape(-1))
            values.append(block.reshape(-1))

    dimension = nstates * dvr.size
    kinetic = sp.coo_matrix(
        (
            np.concatenate(values),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(dimension, dimension),
    ).tocsr()
    potential = sp.diags(np.asarray(energies).reshape(-1), format="csr")
    hamiltonian = kinetic + potential
    return 0.5 * (hamiltonian + hamiltonian.getH())


def coupling_frames(dvr, coupling_line_overlaps):
    """Represent each tuning-line electronic frame at every coupling point."""
    center = int(np.argmin(np.abs(dvr.x[1])))
    frames = np.empty((*dvr.shape, len(STATE_IDS), len(STATE_IDS)), dtype=complex)
    for tuning_index in range(dvr.shape[0]):
        overlaps = coupling_line_overlaps[tuning_index]
        for coupling_index in range(dvr.shape[1]):
            frames[tuning_index, coupling_index] = _polar_unitary(
                overlaps[coupling_index, center]
            )
    return frames


def initial_states(dvr, frames, *, width=(1.0, 1.0)):
    """Return identical upper-state Gaussian packets in both representations."""
    if tuple(width) != (1.0, 1.0):
        raise ValueError(
            "Hermite-DVR initialization currently requires width=(1, 1)."
        )
    electronic = np.array([0.0, 1.0], dtype=complex).reshape(1, 2, 1)
    factors = [electronic]
    for axis in dvr.axes:
        if not isinstance(axis, HermiteDVR):
            raise TypeError("The CASCI benchmark requires HermiteDVR axes.")
        factors.append(
            axis.harmonic_state(0).astype(complex).reshape(
                1,
                axis.npts,
                1,
            )
        )
    cg_state = MPS(factors)
    amplitude = np.asarray(mps_to_array(cg_state)[1])
    full_state = amplitude[..., None] * frames[..., :, 1]
    full_state /= np.linalg.norm(full_state)
    return cg_state, full_state


def build_cgldr(dvr, data, *, max_rank=64):
    dynamics = CGLDR(
        dvr,
        ElectronicPartition(
            sampled=(COORDINATE_NAMES[0],),
            expanded=(COORDINATE_NAMES[1],),
            center=(0.0,),
        ),
        state_ids=STATE_IDS,
        tt_options={"max_rank": max_rank},
    )
    dynamics.set_electronic_data(data)
    return dynamics


def propagate_full(hamiltonian, initial, times):
    states = expm_multiply(
        -1j * hamiltonian,
        initial.reshape(-1),
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
        traceA=-1j * hamiltonian.diagonal().sum(),
    )
    return states.reshape(len(times), *initial.shape)


def compare_dynamics(
    dvr,
    cgldr,
    full_hamiltonian,
    frames,
    *,
    time_step=0.5,
    steps=100,
    output_every=5,
):
    cg_initial, full_initial = initial_states(dvr, frames)
    cgldr.run(
        cg_initial,
        time_step=time_step,
        steps=steps,
        output_every=output_every,
        save_data=False,
    )
    cg_states = np.asarray([mps_to_array(state) for state in cgldr.states])
    times = np.linspace(
        0.0,
        time_step * steps,
        len(cg_states),
    )
    full_states = propagate_full(full_hamiltonian, full_initial, times)

    full_observables = nuclear_observables(
        full_states,
        dvr.x,
        electronic_axis=-1,
    )
    cg_observables = nuclear_observables(
        cg_states,
        dvr.x,
        electronic_axis=1,
    )
    distance = nuclear_density_distance(
        full_observables["nuclear_density"],
        cg_observables["nuclear_density"],
    )
    return {
        "times_au": times,
        "times_fs": times * au2fs,
        "full_coordinate_means": full_observables["coordinate_means"],
        "cg_coordinate_means": cg_observables["coordinate_means"],
        "full_coordinate_second_moments": full_observables[
            "coordinate_second_moments"
        ],
        "cg_coordinate_second_moments": cg_observables[
            "coordinate_second_moments"
        ],
        "full_coordinate_covariance": full_observables[
            "coordinate_covariance"
        ],
        "cg_coordinate_covariance": cg_observables[
            "coordinate_covariance"
        ],
        "full_coordinate_variances": full_observables[
            "coordinate_variances"
        ],
        "cg_coordinate_variances": cg_observables[
            "coordinate_variances"
        ],
        "full_autocorrelation": full_observables["autocorrelation"],
        "cg_autocorrelation": cg_observables["autocorrelation"],
        "full_survival_probability": full_observables[
            "survival_probability"
        ],
        "cg_survival_probability": cg_observables[
            "survival_probability"
        ],
        "full_norms": full_observables["norms"],
        "cg_norms": cg_observables["norms"],
        **distance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument(
        "--eri",
        default="factors",
        choices=("dense", "s4", "s8", "direct", "factors", "ri", "auto"),
    )
    parser.add_argument("--n-tuning", type=int, default=8)
    parser.add_argument("--n-coupling", type=int, default=5)
    parser.add_argument("--time-step", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-every", type=int, default=5)
    parser.add_argument("--max-rank", type=int, default=64)
    parser.add_argument(
        "--coupling-anchors",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Number of coupling-coordinate expansion anchor points.",
    )
    parser.add_argument(
        "--anchor-placement",
        choices=("boundary", "interior"),
        default="boundary",
        help=(
            "Place outer interpolation anchors on the DVR boundary or one "
            "point inward. Interior placement uses linear edge extrapolation."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent CASCI grid-point worker processes.",
    )
    parser.add_argument(
        "--root-window",
        type=int,
        default=6,
        help="CASCI roots computed before target-state overlap tracking.",
    )
    parser.add_argument(
        "--mode-cache",
        type=Path,
        default=Path("/private/tmp/pyrazine_casci_modes.npz"),
    )
    parser.add_argument(
        "--full-cache",
        type=Path,
        default=Path("/private/tmp/pyrazine_casci_full_8x5.npz"),
    )
    parser.add_argument(
        "--cg-cache",
        type=Path,
        default=Path("/private/tmp/pyrazine_casci_cg_8x5.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pyrazine_casci_cgldr_dynamics.npz"),
    )
    parser.add_argument("--force-modes", action="store_true")
    parser.add_argument("--force-scan", action="store_true")
    parser.add_argument("--unitarize-overlaps", action="store_true")
    args = parser.parse_args()
    if args.root_window <= max(STATE_IDS):
        parser.error("--root-window must exceed the largest target state id")

    geometry = reference_geometry()
    print(
        f"[reference] RHF/CASCI(4e,4o), {args.root_window} roots",
        flush=True,
    )
    reference = run_casci_point(
        geometry,
        basis=args.basis,
        eri=args.eri,
        nstates=args.root_window,
    )
    selected = load_or_select_modes(
        reference,
        geometry,
        basis=args.basis,
        cache=args.mode_cache,
        force=args.force_modes,
    )
    print(
        "[modes] tuning/coupling Hessian indices:",
        selected.hessian_indices.tolist(),
    )
    print(
        "[modes] frequencies / cm^-1:",
        (selected.frequencies * au2wavenumber).tolist(),
    )
    print("[modes] |F12| / Eh:", selected.coupling_strengths.tolist())
    print("[modes] |Delta F|/2 / Eh:", selected.tuning_strengths.tolist())

    dvr = build_dvr(
        selected,
        npts=(args.n_tuning, args.n_coupling),
    )
    full_cache_exists = args.full_cache.exists()
    cg_cache_exists = args.cg_cache.exists()
    if full_cache_exists and cg_cache_exists and not args.force_scan:
        energies, line_overlaps, energy_zero = load_full_data(
            args.full_cache,
            dvr,
            unitarize_overlaps=args.unitarize_overlaps,
            root_window=args.root_window,
        )
        cg_data = CGLDRElectronicData.from_npz(args.cg_cache)
        cached_anchors = int(
            cg_data.metadata.get("coupling_anchor_count", 1)
        )
        if cached_anchors != args.coupling_anchors:
            raise ValueError(
                f"Cached CGLDR data use {cached_anchors} coupling anchors, "
                f"not {args.coupling_anchors}."
            )
        expected_anchor_policy = f"{args.anchor_placement}-DVR-points"
        if args.coupling_anchors > 1 and (
            cg_data.separable_hamiltonian is None
            or cg_data.metadata.get("anchor_policy")
            != expected_anchor_policy
        ):
            raise ValueError(
                "Cached multi-anchor data do not match the requested separable "
                f"{expected_anchor_policy} representation. Choose a new "
                "--cg-cache or use --force-scan."
            )
        print("[cache] loaded full and CGLDR CASCI data")
    else:
        start = time.perf_counter()
        scanned_full_grid = args.force_scan or not full_cache_exists
        if scanned_full_grid:
            raw_energies, line_overlaps, points = scan_casci_grid(
                dvr,
                geometry,
                selected.displacements,
                basis=args.basis,
                eri=args.eri,
                unitarize_overlaps=args.unitarize_overlaps,
                orbital_reference=reference,
                root_window=args.root_window,
                workers=args.workers,
            )
            flat_points = points.reshape(-1)
            tracking_strengths = np.asarray([
                point.reference_overlaps for point in flat_points
            ])
            selected_roots = np.asarray([
                point.state_ids for point in flat_points
            ]).reshape(*dvr.shape, len(STATE_IDS))
            print(
                "[state tracking] minimum target/reference overlap:",
                float(np.min(tracking_strengths)),
            )
            print(
                "[state tracking] selected point roots:",
                sorted({point.state_ids for point in flat_points}),
            )
            energy_zero = float(np.min(raw_energies))
        else:
            energies, line_overlaps, energy_zero = load_full_data(
                args.full_cache,
                dvr,
                unitarize_overlaps=args.unitarize_overlaps,
                root_window=args.root_window,
            )
            raw_energies = energies + energy_zero
            coupling_indices = coupling_anchor_indices(
                dvr,
                args.coupling_anchors,
                placement=args.anchor_placement,
            )
            expansion_indices = [
                (tuning_index, coupling_index)
                for tuning_index in range(dvr.shape[0])
                for coupling_index in coupling_indices
            ]
            points = scan_casci_subset(
                dvr,
                geometry,
                selected.displacements,
                expansion_indices,
                basis=args.basis,
                eri=args.eri,
                orbital_reference=reference,
                root_window=args.root_window,
                workers=args.workers,
            )
            print("[cache] reused full-grid CASCI data")

        cg_data = build_cgldr_data(
            dvr,
            raw_energies,
            line_overlaps,
            points,
            selected.displacements[1],
            energy_zero=energy_zero,
            root_window=args.root_window,
            coupling_anchors=args.coupling_anchors,
            anchor_placement=args.anchor_placement,
        )
        energies = raw_energies - energy_zero
        if scanned_full_grid:
            save_full_data(
                args.full_cache,
                dvr,
                raw_energies,
                line_overlaps,
                energy_zero=energy_zero,
                unitarize_overlaps=args.unitarize_overlaps,
                root_window=args.root_window,
                selected_roots=selected_roots,
                tracking_strengths=tracking_strengths.reshape(
                    *dvr.shape,
                    len(STATE_IDS),
                ),
            )
        args.cg_cache.parent.mkdir(parents=True, exist_ok=True)
        cg_data.to_npz(args.cg_cache)
        print(f"[electronic data] completed in {time.perf_counter() - start:.2f} s")
        print("[cache] saved CGLDR CASCI data")

    diagnostics = overlap_diagnostics(line_overlaps)
    for name, value in diagnostics.items():
        print(f"[overlap] {name}: {value:.8f}")
    hamiltonian = build_full_hamiltonian(dvr, energies, line_overlaps)
    print(
        f"[Hamiltonian] shape={hamiltonian.shape}, nnz={hamiltonian.nnz}, "
        f"Hermiticity={sp.linalg.norm(hamiltonian - hamiltonian.getH()):.3e}"
    )
    frames = coupling_frames(dvr, line_overlaps[1])
    dynamics = build_cgldr(dvr, cg_data, max_rank=args.max_rank)
    results = compare_dynamics(
        dvr,
        dynamics,
        hamiltonian,
        frames,
        time_step=args.time_step,
        steps=args.steps,
        output_every=args.output_every,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **results,
        npts=np.asarray(dvr.shape),
        frequencies=selected.frequencies,
        hessian_indices=selected.hessian_indices,
        **diagnostics,
    )
    print(
        "[dynamics] final |autocorrelation|, full/CGLDR:",
        abs(results["full_autocorrelation"][-1]),
        abs(results["cg_autocorrelation"][-1]),
    )
    print("[dynamics] maximum nuclear TV:", np.max(results["total_variation"]))
    print(
        "[dynamics] maximum coordinate-mean error:",
        np.max(np.abs(
            results["full_coordinate_means"]
            - results["cg_coordinate_means"]
        )),
    )
    print(
        "[dynamics] maximum coordinate-variance error:",
        np.max(np.abs(
            results["full_coordinate_variances"]
            - results["cg_coordinate_variances"]
        )),
    )
    print("[output]", args.output)


if __name__ == "__main__":
    main()
