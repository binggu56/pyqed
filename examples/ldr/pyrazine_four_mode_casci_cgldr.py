#!/usr/bin/env python3
"""Real CASCI benchmark with two primary and two secondary pyrazine modes.

The optional full reference evaluates CASCI(4e,4o) on the complete
four-dimensional Hermite DVR and retains direct electronic overlaps along every
DVR line. CGLDR samples the two strongest tuning modes and represents a weaker
tuning mode and the strongest coupling mode with either raw analytical
center-state F/G tensors or overlap-transported three-anchor LPA fits.
``--cg-only`` evaluates only the electronic points needed by the selected
secondary model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy.sparse.linalg import expm_multiply

from examples.ldr.pyrazine_casci_cgldr import (
    STATE_IDS,
    _polar_unitary,
    build_full_hamiltonian,
    pyscf_normal_modes,
    reference_geometry,
    retained_overlap,
    run_casci_point,
    scan_casci_grid,
    scan_casci_subset,
)
from pyqed.dvr import DVR, HermiteDVR
from pyqed.ldr import (
    CGLDR,
    CGLDRElectronicData,
    ElectronicPartition,
    SeparableHamiltonian,
)
from pyqed.ldr.observables import (
    mps_to_array,
    nuclear_density_distance,
    nuclear_observables,
)
from pyqed.mps.mps import MPS
from pyqed.units import au2fs, au2wavenumber


COORDINATE_NAMES = (
    "Q_tuning_1",
    "Q_tuning_2",
    "Q_tuning_3",
    "Q_coupling",
)


@dataclass(frozen=True)
class SelectedFourModes:
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
        if displacements.ndim != 3 or displacements.shape[0] != 4:
            raise ValueError(
                "displacements must have shape (4, natom, 3)"
            )
        for name, values in (
            ("frequencies", frequencies),
            ("hessian_indices", indices),
            ("coupling_strengths", coupling),
            ("tuning_strengths", tuning),
        ):
            if values.shape != (4,):
                raise ValueError(f"{name} must have shape (4,)")
        if np.any(frequencies <= 0.0):
            raise ValueError("mode frequencies must be positive")
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


def select_four_modes(casci, geometry, *, basis="sto-3g"):
    """Select three tuning modes and one distinct coupling mode.

    The two strongest tuning modes are primary. The weaker tuning mode and
    strongest coupling mode are secondary, matching the standard structure of
    the four-mode pyrazine model.
    """
    modes, frequencies, hessian_indices = pyscf_normal_modes(
        geometry,
        basis=basis,
    )
    first = casci.vibronic_gradients(
        state_ids=STATE_IDS,
        modes=modes,
    )
    coupling = np.abs(first[0, 1])
    tuning = 0.5 * np.abs(first[1, 1] - first[0, 0])

    coupling_mode = int(np.argmax(coupling))
    tuning_modes = []
    for index in np.argsort(tuning)[::-1]:
        index = int(index)
        if index != coupling_mode and index not in tuning_modes:
            tuning_modes.append(index)
        if len(tuning_modes) == 3:
            break
    if len(tuning_modes) != 3:
        raise ValueError("Could not select three tuning modes")

    selected = np.asarray(
        [tuning_modes[0], tuning_modes[1], tuning_modes[2], coupling_mode],
        dtype=int,
    )
    return SelectedFourModes(
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
        return SelectedFourModes.from_npz(cache)
    selected = select_four_modes(reference, geometry, basis=basis)
    if cache is not None:
        selected.to_npz(cache)
    return selected


def build_dvr(selected_modes, *, npts=(3, 3, 3, 3)):
    npts = tuple(int(value) for value in npts)
    if len(npts) != 4 or any(value < 3 for value in npts):
        raise ValueError("npts must contain four integers of at least three")
    if any(value % 2 != 1 for value in npts[2:]):
        raise ValueError(
            "secondary DVR point counts must be odd so the expansion center "
            "is present"
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
    return DVR.from_axes(axes, names=COORDINATE_NAMES)


def _secondary_centers(dvr):
    centers = tuple(
        int(np.argmin(np.abs(grid))) for grid in dvr.x[2:]
    )
    if any(
        abs(float(dvr.x[axis + 2][center])) > 1.0e-12
        for axis, center in enumerate(centers)
    ):
        raise ValueError("secondary Hermite DVRs must contain zero")
    return centers


def _primary_points(points, centers):
    return np.asarray(points, dtype=object)[
        :,
        :,
        centers[0],
        centers[1],
    ]


def axial_anchor_indices(dvr):
    """Return the center and two axial anchors per secondary coordinate."""
    centers = _secondary_centers(dvr)
    indices = []
    for primary in np.ndindex(*dvr.shape[:2]):
        center = (*primary, *centers)
        indices.append(center)
        for secondary_axis, coordinate in enumerate(centers, start=2):
            if coordinate == 0 or coordinate == dvr.shape[secondary_axis] - 1:
                raise ValueError(
                    "each secondary center needs two neighboring anchors"
                )
            for offset in (-1, 1):
                anchor = list(center)
                anchor[secondary_axis] = coordinate + offset
                indices.append(tuple(anchor))
    return tuple(indices)


def primary_anchor_indices(dvr):
    """Return the secondary-center point for every primary geometry."""
    centers = _secondary_centers(dvr)
    return tuple(
        (*primary, *centers)
        for primary in np.ndindex(*dvr.shape[:2])
    )


def primary_overlaps(points, centers):
    """Return all direct overlap blocks on the two-dimensional primary grid."""
    primary = _primary_points(points, centers)
    shape = primary.shape
    flat = primary.reshape(-1)
    blocks = np.empty(
        (len(flat), len(STATE_IDS), len(flat), len(STATE_IDS)),
        dtype=complex,
    )
    for bra, left in enumerate(flat):
        blocks[bra, :, bra, :] = np.eye(len(STATE_IDS))
        for ket in range(bra + 1, len(flat)):
            block = retained_overlap(left, flat[ket])
            blocks[bra, :, ket, :] = block
            blocks[ket, :, bra, :] = block.conj().T
    return blocks.reshape(
        *shape,
        len(STATE_IDS),
        *shape,
        len(STATE_IDS),
    )


def _analytic_f(point, modes, model="clamped", backend="native"):
    moving_basis = {
        "clamped": "symmetric",
        "relaxed": "rhf-relaxed",
        "parallel": "rhf-relaxed-pt",
    }.get(model)
    if moving_basis is None:
        raise ValueError("F model must be 'clamped', 'relaxed', or 'parallel'")
    return point.casci.vibronic_gradients(
        state_ids=point.state_ids,
        modes=modes,
        moving_basis=moving_basis,
        backend=backend,
    )


def _analytic_fg(
    point,
    modes,
    f_model="clamped",
    g_model="clamped",
    backend="native",
):
    if g_model == "relaxed":
        moving_basis = {
            "relaxed": "rhf-relaxed",
            "parallel": "rhf-relaxed-pt",
        }.get(f_model)
        if moving_basis is None:
            raise ValueError("Relaxed G requires relaxed or parallel F")
        return point.casci.vibronic_couplings(
            state_ids=point.state_ids,
            modes=modes,
            moving_basis=moving_basis,
            backend=backend,
        )
    if g_model != "clamped":
        raise ValueError("G model must be 'clamped' or 'relaxed'")
    first, second = point.casci.vibronic_couplings(
        state_ids=point.state_ids,
        modes=modes,
        backend=backend,
    )
    if f_model != "clamped":
        first = _analytic_f(point, modes, model=f_model, backend=backend)
    return first, second


def build_cgldr_data(
    dvr,
    raw_energies,
    points,
    selected_modes,
    *,
    energy_zero=None,
    root_window=6,
    metadata=None,
    workers=1,
    f_model="clamped",
    g_model="clamped",
    derivative_backend="native",
):
    """Build an analytical-F/G secondary expansion."""
    centers = _secondary_centers(dvr)
    primary_points = _primary_points(points, centers)
    center_energies = np.asarray(raw_energies)[
        :,
        :,
        centers[0],
        centers[1],
    ]
    if energy_zero is None:
        energy_zero = float(np.min(center_energies))
    energies = center_energies - energy_zero
    overlaps = primary_overlaps(points, centers)
    primary_shape = tuple(dvr.shape[:2])
    gradients = np.empty(
        (*primary_shape, 2, len(STATE_IDS), len(STATE_IDS)),
        dtype=complex,
    )
    hessians = np.empty(
        (
            *primary_shape,
            2,
            2,
            len(STATE_IDS),
            len(STATE_IDS),
        ),
        dtype=complex,
    )
    secondary_modes = np.asarray(selected_modes.displacements[2:])
    indices = tuple(np.ndindex(primary_shape))
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        iterator = (
            (
                index,
                _analytic_fg(
                    primary_points[index],
                    secondary_modes,
                    f_model,
                    g_model,
                    derivative_backend,
                ),
            )
            for index in indices
        )
        for completed, (index, (first, second)) in enumerate(iterator, start=1):
            gradients[index] = np.moveaxis(first, -1, 0)
            hessians[index] = np.moveaxis(second, (-2, -1), (0, 1))
            print(
                f"[CASCI secondary derivatives] {completed}/{len(indices)}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _analytic_fg,
                    primary_points[index],
                    secondary_modes,
                    f_model,
                    g_model,
                    derivative_backend,
                ): index
                for index in indices
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                first, second = future.result()
                gradients[index] = np.moveaxis(first, -1, 0)
                hessians[index] = np.moveaxis(second, (-2, -1), (0, 1))
                print(
                    f"[CASCI secondary derivatives] {completed}/{len(indices)} "
                    f"({workers} workers)",
                    flush=True,
                )

    output_metadata = {} if metadata is None else dict(metadata)
    output_metadata.setdefault("solver", "CASCI(4e,4o)/STO-3G")
    representation = {
        ("clamped", "clamped"): "single-center-analytic-fg",
        ("relaxed", "clamped"): "single-center-rhf-relaxed-f-clamped-g",
        ("parallel", "clamped"): "single-center-rhf-parallel-f-clamped-g",
        ("relaxed", "relaxed"): "single-center-rhf-relaxed-fg",
        ("parallel", "relaxed"): "single-center-rhf-parallel-fg",
    }.get((f_model, g_model))
    if representation is None:
        raise ValueError("F model must be 'clamped', 'relaxed', or 'parallel'")
    output_metadata.update({
        "state_ids": list(STATE_IDS),
        "root_window": int(root_window),
        "primary_coordinates": list(COORDINATE_NAMES[:2]),
        "secondary_coordinates": list(COORDINATE_NAMES[2:]),
        "secondary_expansion": "single-center-analytic-F/G",
        "secondary_representation": representation,
        "secondary_mixed_hessian": "included",
        "derivative_character": {
            ("clamped", "clamped"): "clamped-center-state",
            ("relaxed", "clamped"): "rhf-relaxed-F_clamped-G",
            ("parallel", "clamped"): "rhf-parallel-F_clamped-G",
            ("relaxed", "relaxed"): "rhf-relaxed-F/G",
            ("parallel", "relaxed"): "rhf-parallel-F/G",
        }[(f_model, g_model)],
        "derivative_integral_backend": str(derivative_backend),
        "electronic_points_per_primary": 1,
        "energy_zero": float(energy_zero),
    })
    return CGLDRElectronicData(
        energies=energies,
        overlaps=overlaps,
        hamiltonian_gradients=gradients,
        hamiltonian_hessians=hessians,
        reactive_grids=tuple(np.asarray(grid) for grid in dvr.x[:2]),
        expanded_grids=tuple(np.asarray(grid) for grid in dvr.x[2:]),
        metadata=output_metadata,
    )


def build_cardinal_cgldr_data(
    dvr,
    energies,
    frames,
    overlaps,
    *,
    metadata=None,
):
    """Represent the real full-grid secondary potential in the center frame.

    The nuclear coordinate factors are DVR cardinal functions. This avoids the
    noncovariant projected CASCI Hessian while retaining the exact ab initio
    potential values on the pilot secondary grid.
    """
    energies = np.asarray(energies)
    frames = np.asarray(frames)
    expected_energies = (*dvr.shape, len(STATE_IDS))
    expected_frames = (
        *dvr.shape,
        len(STATE_IDS),
        len(STATE_IDS),
    )
    if energies.shape != expected_energies:
        raise ValueError(
            f"energies shape {energies.shape} != {expected_energies}"
        )
    if frames.shape != expected_frames:
        raise ValueError(
            f"frames shape {frames.shape} != {expected_frames}"
        )
    center_field = np.einsum(
        "...ap,...a,...aq->...pq",
        frames.conj(),
        energies,
        frames,
        optimize=True,
    )
    center_field = 0.5 * (
        center_field + center_field.swapaxes(-1, -2).conj()
    )
    primary_shape = tuple(dvr.shape[:2])
    secondary_shape = tuple(dvr.shape[2:])
    nterms = int(np.prod(secondary_shape))
    operators = center_field.reshape(
        *primary_shape,
        nterms,
        len(STATE_IDS),
        len(STATE_IDS),
    )
    factors = [
        np.zeros((nterms, count), dtype=float)
        for count in secondary_shape
    ]
    for term, index in enumerate(np.ndindex(secondary_shape)):
        for axis, coordinate in enumerate(index):
            factors[axis][term, coordinate] = 1.0

    centers = _secondary_centers(dvr)
    center_energies = energies[
        :,
        :,
        centers[0],
        centers[1],
    ]
    output_metadata = {} if metadata is None else dict(metadata)
    output_metadata.update({
        "secondary_representation": "full-grid-cardinal-center-frame",
        "secondary_term_count": nterms,
        "transport": "direct-polar-center-frame",
        "projected_casci_hessian": "excluded",
    })
    return CGLDRElectronicData(
        energies=center_energies,
        overlaps=np.asarray(overlaps),
        separable_hamiltonian=SeparableHamiltonian(
            operators=operators,
            factors=tuple(factors),
        ),
        reactive_grids=tuple(np.asarray(grid) for grid in dvr.x[:2]),
        expanded_grids=tuple(np.asarray(grid) for grid in dvr.x[2:]),
        metadata=output_metadata,
    )


def build_axial_lpa_cgldr_data(
    dvr,
    energies,
    frames,
    overlaps,
    *,
    metadata=None,
):
    """Fit independent three-anchor quadratics for two secondary modes.

    Only the center and the two axial neighbors of each secondary coordinate
    enter the fit. The secondary mixed Hessian is intentionally omitted, so
    the electronic point count per primary geometry is ``1 + 2 M = 5``.
    """
    energies = np.asarray(energies)
    frames = np.asarray(frames)
    expected_energies = (*dvr.shape, len(STATE_IDS))
    expected_frames = (*dvr.shape, len(STATE_IDS), len(STATE_IDS))
    if energies.shape != expected_energies:
        raise ValueError(
            f"energies shape {energies.shape} != {expected_energies}"
        )
    if frames.shape != expected_frames:
        raise ValueError(
            f"frames shape {frames.shape} != {expected_frames}"
        )

    centers = _secondary_centers(dvr)
    anchor_indices = []
    for axis, center in enumerate(centers, start=2):
        if center == 0 or center == dvr.shape[axis] - 1:
            raise ValueError("each secondary center needs two neighboring anchors")
        anchor_indices.append((center - 1, center, center + 1))

    center_field = np.einsum(
        "...ap,...a,...aq->...pq",
        frames.conj(),
        energies,
        frames,
        optimize=True,
    )
    center_field = 0.5 * (
        center_field + center_field.swapaxes(-1, -2).conj()
    )
    primary_shape = tuple(dvr.shape[:2])
    nstates = len(STATE_IDS)
    operators = np.empty((*primary_shape, 5, nstates, nstates), dtype=complex)
    center_slice = (
        slice(None),
        slice(None),
        centers[0],
        centers[1],
    )
    operators[..., 0, :, :] = center_field[center_slice]

    factors = [
        np.ones((5, dvr.shape[axis]), dtype=float)
        for axis in (2, 3)
    ]
    for secondary_axis, indices in enumerate(anchor_indices):
        grid_axis = secondary_axis + 2
        coordinates = np.asarray(dvr.x[grid_axis], dtype=float)
        anchors = np.asarray(indices, dtype=int)
        delta_anchors = coordinates[anchors] - coordinates[centers[secondary_axis]]
        design = np.column_stack((
            np.ones(3),
            delta_anchors,
            0.5 * delta_anchors**2,
        ))
        field_slice = [slice(None), slice(None), centers[0], centers[1]]
        field_slice[grid_axis] = anchors
        samples = center_field[tuple(field_slice)]
        coefficients = np.linalg.solve(
            design,
            np.moveaxis(samples, 2, 0).reshape(3, -1),
        ).reshape(3, *primary_shape, nstates, nstates)
        first_term = 1 + 2 * secondary_axis
        operators[..., first_term, :, :] = coefficients[1]
        operators[..., first_term + 1, :, :] = coefficients[2]

        delta_grid = coordinates - coordinates[centers[secondary_axis]]
        factors[secondary_axis][first_term] = delta_grid
        factors[secondary_axis][first_term + 1] = 0.5 * delta_grid**2

    operators = 0.5 * (
        operators + operators.swapaxes(-1, -2).conj()
    )
    center_energies = energies[center_slice]
    output_metadata = {} if metadata is None else dict(metadata)
    output_metadata.update({
        "secondary_representation": "axial-three-anchor-quadratic-lpa",
        "secondary_anchor_indices": [list(values) for values in anchor_indices],
        "secondary_term_count": 5,
        "electronic_points_per_primary": 5,
        "secondary_mixed_hessian": "excluded",
        "transport": "direct-polar-center-frame",
        "projected_casci_hessian": "excluded",
    })
    return CGLDRElectronicData(
        energies=center_energies,
        overlaps=np.asarray(overlaps),
        separable_hamiltonian=SeparableHamiltonian(
            operators=operators,
            factors=tuple(factors),
        ),
        reactive_grids=tuple(np.asarray(grid) for grid in dvr.x[:2]),
        expanded_grids=tuple(np.asarray(grid) for grid in dvr.x[2:]),
        metadata=output_metadata,
    )


def build_axial_cgldr_data_from_points(
    dvr,
    points,
    *,
    metadata=None,
):
    """Build axial-LPA data from only the sampled CGLDR anchor points."""
    points = np.asarray(points, dtype=object)
    if points.shape != dvr.shape:
        raise ValueError(f"points shape {points.shape} != DVR shape {dvr.shape}")

    indices = axial_anchor_indices(dvr)
    missing = [index for index in indices if points[index] is None]
    if missing:
        raise ValueError(f"missing {len(missing)} axial CGLDR anchor points")

    energies = np.full((*dvr.shape, len(STATE_IDS)), np.nan, dtype=float)
    frames = np.full(
        (*dvr.shape, len(STATE_IDS), len(STATE_IDS)),
        np.nan + 0.0j,
        dtype=complex,
    )
    centers = _secondary_centers(dvr)
    tracking_strengths = []
    for index in indices:
        point = points[index]
        energies[index] = np.asarray(point.casci.e_tot)[list(point.state_ids)]
        tracking_strengths.extend(np.asarray(point.reference_overlaps).tolist())
        center_index = (*index[:2], *centers)
        if index == center_index:
            frames[index] = np.eye(len(STATE_IDS))
        else:
            frames[index] = _polar_unitary(
                retained_overlap(point, points[center_index])
            )

    energy_zero = float(np.nanmin(energies))
    output_metadata = {} if metadata is None else dict(metadata)
    output_metadata.update({
        "energy_zero": energy_zero,
        "electronic_point_count": len(indices),
        "minimum_reference_state_overlap": float(np.min(tracking_strengths)),
    })
    return build_axial_lpa_cgldr_data(
        dvr,
        energies - energy_zero,
        frames,
        primary_overlaps(points, centers),
        metadata=output_metadata,
    )


def build_analytic_cgldr_data_from_points(
    dvr,
    points,
    selected_modes,
    *,
    root_window=6,
    metadata=None,
    workers=1,
    f_model="clamped",
    g_model="clamped",
    derivative_backend="native",
):
    """Build analytical F/G data from primary anchors."""
    points = np.asarray(points, dtype=object)
    if points.shape != dvr.shape:
        raise ValueError(f"points shape {points.shape} != DVR shape {dvr.shape}")

    indices = primary_anchor_indices(dvr)
    missing = [index for index in indices if points[index] is None]
    if missing:
        raise ValueError(f"missing {len(missing)} primary CGLDR anchor points")

    raw_energies = np.full((*dvr.shape, len(STATE_IDS)), np.nan, dtype=float)
    tracking_strengths = []
    for index in indices:
        point = points[index]
        raw_energies[index] = np.asarray(point.casci.e_tot)[list(point.state_ids)]
        tracking_strengths.extend(np.asarray(point.reference_overlaps).tolist())

    center_values = np.asarray([raw_energies[index] for index in indices])
    energy_zero = float(np.min(center_values))
    output_metadata = {} if metadata is None else dict(metadata)
    output_metadata.update({
        "electronic_point_count": len(indices),
        "minimum_reference_state_overlap": float(np.min(tracking_strengths)),
        "derivative_source": {
            ("clamped", "clamped"): "CASCI.vibronic_couplings",
            ("relaxed", "clamped"): (
                "CASCI.vibronic_gradients(rhf-relaxed)+"
                "CASCI.vibronic_couplings(clamped-G)"
            ),
            ("parallel", "clamped"): (
                "CASCI.vibronic_gradients(rhf-relaxed-pt)+"
                "CASCI.vibronic_couplings(clamped-G)"
            ),
            ("relaxed", "relaxed"): (
                "CASCI.vibronic_couplings(rhf-relaxed)"
            ),
            ("parallel", "relaxed"): (
                "CASCI.vibronic_couplings(rhf-relaxed-pt)"
            ),
        }.get((f_model, g_model)),
    })
    if output_metadata["derivative_source"] is None:
        raise ValueError("F model must be 'clamped', 'relaxed', or 'parallel'")
    return build_cgldr_data(
        dvr,
        raw_energies,
        points,
        selected_modes,
        energy_zero=energy_zero,
        root_window=root_window,
        metadata=output_metadata,
        workers=workers,
        f_model=f_model,
        g_model=g_model,
        derivative_backend=derivative_backend,
    )


def full_to_center_frames(points, centers):
    """Project the center electronic frame onto every full-grid local frame."""
    points = np.asarray(points, dtype=object)
    frames = np.empty(
        (*points.shape, len(STATE_IDS), len(STATE_IDS)),
        dtype=complex,
    )
    for index in np.ndindex(points.shape):
        center_index = (
            index[0],
            index[1],
            centers[0],
            centers[1],
        )
        if index == center_index:
            frames[index] = np.eye(len(STATE_IDS))
        else:
            frames[index] = _polar_unitary(
                retained_overlap(points[index], points[center_index])
            )
    return frames


def build_cgldr(dvr, data, *, max_rank=64):
    dynamics = CGLDR(
        dvr,
        ElectronicPartition(
            sampled=COORDINATE_NAMES[:2],
            expanded=COORDINATE_NAMES[2:],
            center=(0.0, 0.0),
        ),
        state_ids=STATE_IDS,
        tt_options={"max_rank": max_rank},
    )
    dynamics.set_electronic_data(data)
    return dynamics


def initial_cg_state(dvr):
    electronic = np.array([0.0, 1.0], dtype=complex).reshape(1, 2, 1)
    factors = [electronic]
    for axis in dvr.axes:
        factors.append(
            axis.harmonic_state(0).astype(complex).reshape(
                1,
                axis.npts,
                1,
            )
        )
    return MPS(factors)


def initial_states(dvr, frames):
    cg_state = initial_cg_state(dvr)
    nuclear = np.asarray(mps_to_array(cg_state)[1])
    full_state = nuclear[..., None] * frames[..., :, 1]
    full_state /= np.linalg.norm(full_state)
    return cg_state, full_state


def run_cgldr_only(
    dvr,
    cgldr,
    *,
    time_step=0.5,
    steps=100,
    output_every=5,
    integrator="hybrid",
    tdvp_cutoff=0.0,
    krylov_dim=12,
    tdvp_warmup_steps=20,
):
    """Propagate CGLDR and collect observables without a full-grid reference."""
    cgldr.run(
        initial_cg_state(dvr),
        time_step=time_step,
        steps=steps,
        output_every=output_every,
        save_data=False,
        integrator=integrator,
        tdvp_options={
            "cutoff": tdvp_cutoff,
            "krylov_dim": krylov_dim,
        },
        tdvp_warmup_steps=tdvp_warmup_steps,
    )
    states = np.asarray([mps_to_array(state) for state in cgldr.states])
    times = np.linspace(0.0, time_step * steps, len(states))
    populations = np.sum(
        np.abs(states) ** 2,
        axis=tuple(range(2, states.ndim)),
    )
    populations /= populations.sum(axis=1, keepdims=True)
    observables = nuclear_observables(states, dvr.x, electronic_axis=1)
    results = {
        "times_au": times,
        "times_fs": times * au2fs,
        "cg_coordinate_means": observables["coordinate_means"],
        "cg_coordinate_variances": observables["coordinate_variances"],
        "cg_autocorrelation": observables["autocorrelation"],
        "cg_norms": observables["norms"],
        "cg_populations": populations,
        "integrator": np.asarray(cgldr.integrator),
        "integrator_history": np.asarray(cgldr.integrator_history),
        "bond_dimensions": np.asarray(cgldr.bond_dimensions),
    }
    if cgldr.integrator != "split":
        results["tdvp_truncation_errors"] = np.asarray(
            cgldr.tdvp_truncation_errors
        )
    return results


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
    cg_states = np.asarray([
        mps_to_array(state) for state in cgldr.states
    ])
    times = np.linspace(
        0.0,
        time_step * steps,
        len(cg_states),
    )
    full_states = expm_multiply(
        -1j * full_hamiltonian,
        full_initial.reshape(-1),
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
        traceA=-1j * full_hamiltonian.diagonal().sum(),
    ).reshape(len(times), *full_initial.shape)

    full_center_states = np.einsum(
        "...ap,t...a->t...p",
        frames.conj(),
        full_states,
        optimize=True,
    )
    cg_center_states = np.moveaxis(cg_states, 1, -1)
    full_populations = np.sum(
        np.abs(full_center_states) ** 2,
        axis=tuple(range(1, full_center_states.ndim - 1)),
    )
    cg_populations = np.sum(
        np.abs(cg_center_states) ** 2,
        axis=tuple(range(1, cg_center_states.ndim - 1)),
    )
    full_populations /= full_populations.sum(axis=1, keepdims=True)
    cg_populations /= cg_populations.sum(axis=1, keepdims=True)

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
        "full_coordinate_variances": full_observables[
            "coordinate_variances"
        ],
        "cg_coordinate_variances": cg_observables[
            "coordinate_variances"
        ],
        "full_autocorrelation": full_observables["autocorrelation"],
        "cg_autocorrelation": cg_observables["autocorrelation"],
        "full_norms": full_observables["norms"],
        "cg_norms": cg_observables["norms"],
        "full_populations": full_populations,
        "cg_populations": cg_populations,
        **distance,
    }


def plot_comparison(results, output):
    """Write a compact full-LDR versus CGLDR validation figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    time_fs = np.asarray(results["times_fs"])
    full_means = np.asarray(results["full_coordinate_means"])
    cg_means = np.asarray(results["cg_coordinate_means"])
    mean_error = np.max(np.abs(full_means - cg_means), axis=1)

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 5.1), constrained_layout=True)
    colors = ("#0072B2", "#D55E00")

    for state, label in enumerate(("S1", "S2")):
        axes[0, 0].plot(
            time_fs,
            results["full_populations"][:, state],
            color=colors[state],
            linewidth=1.7,
            label=f"Full {label}",
        )
        axes[0, 0].plot(
            time_fs,
            results["cg_populations"][:, state],
            color=colors[state],
            linestyle="--",
            linewidth=1.4,
            label=f"CGLDR {label}",
        )
    axes[0, 0].set_ylabel("Population")
    axes[0, 0].legend(frameon=False, ncol=2, handlelength=2.2)

    axes[0, 1].plot(
        time_fs,
        np.abs(results["full_autocorrelation"]) ** 2,
        color="#222222",
        linewidth=1.7,
        label="Full LDR",
    )
    axes[0, 1].plot(
        time_fs,
        np.abs(results["cg_autocorrelation"]) ** 2,
        color="#CC79A7",
        linestyle="--",
        linewidth=1.5,
        label="CGLDR",
    )
    axes[0, 1].set_ylabel(r"$|C(t)|^2$")
    axes[0, 1].legend(frameon=False)

    axes[0, 2].plot(
        time_fs,
        results["total_variation"],
        color="#009E73",
        linewidth=1.6,
    )
    axes[0, 2].set_ylabel("Nuclear-density TV")
    axes[0, 3].plot(
        time_fs,
        mean_error,
        color="#D55E00",
        linewidth=1.6,
    )
    axes[0, 3].set_ylabel(r"max $|\Delta\langle Q\rangle|$")

    coordinate_labels = (
        r"$Q_{\mathrm{tune},1}$",
        r"$Q_{\mathrm{tune},2}$",
        r"$Q_{\mathrm{tune},3}$",
        r"$Q_{\mathrm{couple}}$",
    )
    for axis, label, full, coarse in zip(
        axes[1], coordinate_labels, full_means.T, cg_means.T
    ):
        axis.plot(
            time_fs, full, color="#222222", linewidth=1.7, label="Full LDR"
        )
        axis.plot(
            time_fs, coarse, color="#0072B2", linestyle="--",
            linewidth=1.4, label="CGLDR"
        )
        axis.set_ylabel(rf"$\langle {label[1:-1]}\rangle$")

    for letter, axis in zip("abcdefgh", axes.flat):
        axis.set_xlabel("Time (fs)")
        axis.grid(True, color="#dddddd", linewidth=0.5, alpha=0.8)
        axis.text(
            0.02,
            0.96,
            letter,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        for spine in axis.spines.values():
            spine.set_color("#333333")
    axes[1, 0].legend(frameon=False)
    fig.savefig(pdf)
    fig.savefig(png, dpi=360)
    plt.close(fig)
    return png, pdf


def plot_cgldr_dynamics(results, output):
    """Write a standalone four-mode CGLDR dynamics figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    time_fs = np.asarray(results["times_fs"])
    populations = np.asarray(results["cg_populations"])
    means = np.asarray(results["cg_coordinate_means"])

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(7.2, 7.8),
        sharex=True,
        constrained_layout=True,
    )
    axes[0, 0].plot(time_fs, populations[:, 0], color="#0072B2", label="$S_1$")
    axes[0, 0].plot(time_fs, populations[:, 1], color="#D55E00", label="$S_2$")
    axes[0, 0].set_ylabel("Population")
    axes[0, 0].legend(frameon=False, ncol=2)

    coordinate_labels = ("$Q_{t1}$", "$Q_{t2}$", "$Q_{t3}$", "$Q_c$")
    coordinate_axes = (axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0])
    colors = ("#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    for axis, label, values, color in zip(
        coordinate_axes,
        coordinate_labels,
        means.T,
        colors,
    ):
        axis.plot(time_fs, values, color=color, linewidth=1.6)
        axis.axhline(0.0, color="#777777", linewidth=0.6)
        axis.set_ylabel(rf"$\langle {label[1:-1]}\rangle$")

    axes[2, 1].plot(
        time_fs,
        np.abs(results["cg_autocorrelation"]),
        color="#222222",
        label=r"$|C(t)|$",
    )
    axes[2, 1].plot(
        time_fs,
        results["cg_norms"],
        color="#999999",
        linestyle="--",
        label="Norm",
    )
    axes[2, 1].set_ylabel("Amplitude")
    axes[2, 1].legend(frameon=False)

    for letter, axis in zip("abcdef", axes.flat):
        axis.set_xlabel("Time (fs)")
        axis.grid(True, color="#dddddd", linewidth=0.5)
        axis.text(
            0.02,
            0.95,
            letter,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    fig.savefig(pdf)
    fig.savefig(png, dpi=360)
    plt.close(fig)
    return png, pdf


def validate_cgldr_cache(
    data,
    dvr,
    *,
    basis,
    root_window,
    secondary_model="axial-lpa",
    derivative_backend=None,
):
    """Reject an electronic cache built for a different calculation."""
    metadata = data.metadata
    if metadata.get("ao_basis") != basis:
        raise ValueError("Cached CGLDR AO basis does not match --basis")
    if int(metadata.get("root_window", -1)) != int(root_window):
        raise ValueError("Cached CGLDR root window does not match")
    expected_representation = {
        "axial-lpa": "axial-three-anchor-quadratic-lpa",
        "analytic-fg": "single-center-analytic-fg",
        "relaxed-f": "single-center-rhf-relaxed-f-clamped-g",
        "parallel-f": "single-center-rhf-parallel-f-clamped-g",
        "relaxed-fg": "single-center-rhf-relaxed-fg",
        "parallel-fg": "single-center-rhf-parallel-fg",
    }.get(secondary_model)
    if metadata.get("secondary_representation") != expected_representation:
        raise ValueError("Cached CGLDR secondary representation does not match")
    if derivative_backend is not None and metadata.get(
        "derivative_integral_backend"
    ) != str(derivative_backend):
        raise ValueError("Cached CGLDR derivative backend does not match")
    for cached, grid in zip(data.reactive_grids or (), dvr.x[:2]):
        np.testing.assert_allclose(cached, grid)
    for cached, grid in zip(data.expanded_grids or (), dvr.x[2:]):
        np.testing.assert_allclose(cached, grid)


def save_full_data(
    filename,
    dvr,
    energies,
    line_overlaps,
    frames,
    *,
    energy_zero,
    root_window,
):
    arrays = {
        "energies": np.asarray(energies) - energy_zero,
        "frames": np.asarray(frames),
        "energy_zero": np.asarray(energy_zero),
        "root_window": np.asarray(root_window),
        "state_ids": np.asarray(STATE_IDS),
    }
    for axis, (grid, overlaps) in enumerate(
        zip(dvr.x, line_overlaps)
    ):
        arrays[f"grid_{axis}"] = np.asarray(grid)
        arrays[f"overlaps_{axis}"] = np.asarray(overlaps)
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez(filename, **arrays)


def load_full_data(filename, dvr, *, root_window):
    with np.load(filename) as archive:
        if int(archive["root_window"]) != root_window:
            raise ValueError("Cached root window does not match")
        if tuple(archive["state_ids"]) != STATE_IDS:
            raise ValueError("Cached state IDs do not match")
        for axis, grid in enumerate(dvr.x):
            np.testing.assert_allclose(archive[f"grid_{axis}"], grid)
        return (
            np.asarray(archive["energies"]),
            tuple(
                np.asarray(archive[f"overlaps_{axis}"])
                for axis in range(dvr.ndim)
            ),
            np.asarray(archive["frames"]),
            float(archive["energy_zero"]),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--eri", default="dense")
    parser.add_argument(
        "--derivative-backend",
        choices=("native", "pyscf", "python", "auto"),
        default="native",
        help="Integral backend for analytical F/G derivatives.",
    )
    parser.add_argument("--root-window", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--npts",
        nargs=4,
        type=int,
        default=(3, 3, 3, 3),
        metavar=("N1", "N2", "NS1", "NS2"),
    )
    parser.add_argument("--time-step", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output-every", type=int, default=5)
    parser.add_argument("--max-rank", type=int, default=64)
    parser.add_argument(
        "--integrator",
        choices=("split", "tdvp", "tdvp2", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--tdvp-cutoff", type=float, default=0.0)
    parser.add_argument("--krylov-dim", type=int, default=12)
    parser.add_argument("--tdvp-warmup-steps", type=int, default=20)
    parser.add_argument(
        "--secondary-model",
        choices=(
            "analytic-fg",
            "relaxed-f",
            "parallel-f",
            "relaxed-fg",
            "parallel-fg",
            "axial-lpa",
            "cardinal",
        ),
        default="axial-lpa",
        help=(
            "Use clamped analytical CASCI F/G, canonical or overlap-parallel "
            "RHF-relaxed derivatives, independent three-anchor LPA fits, or "
            "the full secondary cardinal field."
        ),
    )
    parser.add_argument(
        "--mode-cache",
        type=Path,
        default=Path(
            "/private/tmp/pyrazine_casci_four_modes_3tuning_1coupling.npz"
        ),
    )
    parser.add_argument(
        "--full-cache",
        type=Path,
        default=Path(
            "/private/tmp/pyrazine_casci_four_mode_full_canonical_3x3x3x3.npz"
        ),
    )
    parser.add_argument(
        "--cg-cache",
        type=Path,
        default=Path(
            "/private/tmp/pyrazine_casci_four_mode_cg_axial_lpa_3x3.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/private/tmp/pyrazine_casci_four_mode_cgldr_dynamics.npz"
        ),
    )
    parser.add_argument("--force-modes", action="store_true")
    parser.add_argument("--force-scan", action="store_true")
    parser.add_argument(
        "--cg-only",
        action="store_true",
        help=(
            "Evaluate only the electronic points required by the selected "
            "secondary model and run CGLDR without constructing the full-grid "
            "LDR reference."
        ),
    )
    args = parser.parse_args()
    if args.cg_only and args.secondary_model == "cardinal":
        parser.error("--cg-only does not support --secondary-model cardinal")
    analytic_options = {
        "analytic-fg": ("clamped", "clamped"),
        "relaxed-f": ("relaxed", "clamped"),
        "parallel-f": ("parallel", "clamped"),
        "relaxed-fg": ("relaxed", "relaxed"),
        "parallel-fg": ("parallel", "relaxed"),
    }
    analytic_models = set(analytic_options)
    if args.secondary_model in analytic_models and not args.cg_only:
        parser.error(
            "analytical secondary models currently require --cg-only"
        )

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
    print("[modes] Hessian indices:", selected.hessian_indices.tolist())
    print(
        "[modes] frequencies / cm^-1:",
        (selected.frequencies * au2wavenumber).tolist(),
    )
    print("[modes] |F12| / Eh:", selected.coupling_strengths.tolist())
    print("[modes] |Delta F|/2 / Eh:", selected.tuning_strengths.tolist())

    dvr = build_dvr(selected, npts=tuple(args.npts))
    base_metadata = {
        "solver": f"CASCI(4e,4o)/{args.basis}",
        "ao_basis": args.basis,
        "integral_engine": "native",
        "eri_representation": args.eri,
        "state_ids": list(STATE_IDS),
        "root_window": int(args.root_window),
        "primary_coordinates": list(COORDINATE_NAMES[:2]),
        "secondary_coordinates": list(COORDINATE_NAMES[2:]),
        "mode_roles": [
            "primary-tuning",
            "primary-tuning",
            "secondary-tuning",
            "secondary-coupling",
        ],
    }

    if args.cg_only:
        if args.cg_cache.exists() and not args.force_scan:
            cg_data = CGLDRElectronicData.from_npz(args.cg_cache)
            validate_cgldr_cache(
                cg_data,
                dvr,
                basis=args.basis,
                root_window=args.root_window,
                secondary_model=args.secondary_model,
                derivative_backend=(
                    args.derivative_backend
                    if args.secondary_model in analytic_models
                    else None
                ),
            )
            print(
                f"[cache] loaded {args.secondary_model} CGLDR CASCI data"
            )
        else:
            start = time.perf_counter()
            indices = (
                primary_anchor_indices(dvr)
                if args.secondary_model in analytic_models
                else axial_anchor_indices(dvr)
            )
            points = scan_casci_subset(
                dvr,
                geometry,
                selected.displacements,
                indices,
                basis=args.basis,
                eri=args.eri,
                orbital_reference=reference,
                root_window=args.root_window,
                workers=args.workers,
            )
            if args.secondary_model in analytic_models:
                cg_data = build_analytic_cgldr_data_from_points(
                    dvr,
                    points,
                    selected,
                    root_window=args.root_window,
                    metadata=base_metadata,
                    workers=args.workers,
                    f_model=analytic_options[args.secondary_model][0],
                    g_model=analytic_options[args.secondary_model][1],
                    derivative_backend=args.derivative_backend,
                )
            else:
                cg_data = build_axial_cgldr_data_from_points(
                    dvr,
                    points,
                    metadata=base_metadata,
                )
            args.cg_cache.parent.mkdir(parents=True, exist_ok=True)
            cg_data.to_npz(args.cg_cache)
            print(
                f"[CASCI] {len(indices)} CGLDR anchors took "
                f"{time.perf_counter() - start:.1f} s"
            )

        dynamics = build_cgldr(dvr, cg_data, max_rank=args.max_rank)
        results = run_cgldr_only(
            dvr,
            dynamics,
            time_step=args.time_step,
            steps=args.steps,
            output_every=args.output_every,
            integrator=args.integrator,
            tdvp_cutoff=args.tdvp_cutoff,
            krylov_dim=args.krylov_dim,
            tdvp_warmup_steps=args.tdvp_warmup_steps,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **results)
        png, pdf = plot_cgldr_dynamics(results, args.output)
        print(
            "final populations CGLDR:",
            results["cg_populations"][-1].tolist(),
        )
        print(
            "coordinate means CGLDR:",
            results["cg_coordinate_means"][-1].tolist(),
        )
        print(
            "maximum norm error:",
            float(np.max(np.abs(results["cg_norms"] - 1.0))),
        )
        print("saved:", args.output)
        print("figures:", png, pdf)
        return

    def reduced_data(energies, frames, overlaps, metadata):
        if args.secondary_model == "axial-lpa":
            return build_axial_lpa_cgldr_data(
                dvr,
                energies,
                frames,
                overlaps,
                metadata=metadata,
            )
        return build_cardinal_cgldr_data(
            dvr,
            energies,
            frames,
            overlaps,
            metadata=metadata,
        )

    if (
        args.full_cache.exists()
        and args.cg_cache.exists()
        and not args.force_scan
    ):
        energies, line_overlaps, frames, _energy_zero = load_full_data(
            args.full_cache,
            dvr,
            root_window=args.root_window,
        )
        cached_cg_data = CGLDRElectronicData.from_npz(args.cg_cache)
        cached_metadata = dict(cached_cg_data.metadata)
        cached_metadata.update(base_metadata)
        cg_data = reduced_data(
            energies, frames, cached_cg_data.overlaps, cached_metadata
        )
        cg_data.to_npz(args.cg_cache)
        print("[cache] loaded full and CGLDR CASCI data")
    else:
        start = time.perf_counter()
        raw_energies, line_overlaps, points = scan_casci_grid(
            dvr,
            geometry,
            selected.displacements,
            basis=args.basis,
            eri=args.eri,
            unitarize_overlaps=False,
            orbital_reference=reference,
            root_window=args.root_window,
            workers=args.workers,
        )
        energy_zero = float(np.min(raw_energies))
        centers = _secondary_centers(dvr)
        frames = full_to_center_frames(points, centers)
        energies = raw_energies - energy_zero
        overlaps = primary_overlaps(points, centers)
        fresh_metadata = dict(base_metadata)
        fresh_metadata["energy_zero"] = float(energy_zero)
        cg_data = reduced_data(
            energies, frames, overlaps, fresh_metadata
        )
        save_full_data(
            args.full_cache,
            dvr,
            raw_energies,
            line_overlaps,
            frames,
            energy_zero=energy_zero,
            root_window=args.root_window,
        )
        args.cg_cache.parent.mkdir(parents=True, exist_ok=True)
        cg_data.to_npz(args.cg_cache)
        print(
            f"[CASCI] data construction took {time.perf_counter() - start:.1f} s"
        )

    dynamics = build_cgldr(
        dvr,
        cg_data,
        max_rank=args.max_rank,
    )
    full_hamiltonian = build_full_hamiltonian(
        dvr,
        energies,
        line_overlaps,
    )
    results = compare_dynamics(
        dvr,
        dynamics,
        full_hamiltonian,
        frames,
        time_step=args.time_step,
        steps=args.steps,
        output_every=args.output_every,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **results)
    png, pdf = plot_comparison(results, args.output)
    print(
        "final populations full:",
        results["full_populations"][-1].tolist(),
    )
    print(
        "final populations CGLDR:",
        results["cg_populations"][-1].tolist(),
    )
    print("maximum density TV:", float(np.max(results["total_variation"])))
    print(
        "maximum coordinate-mean error:",
        float(np.max(np.abs(
            results["full_coordinate_means"]
            - results["cg_coordinate_means"]
        ))),
    )
    print(
        "maximum |autocorrelation| error:",
        float(np.max(np.abs(
            np.abs(results["full_autocorrelation"])
            - np.abs(results["cg_autocorrelation"])
        ))),
    )
    print("saved:", args.output)
    print("figures:", png, pdf)


if __name__ == "__main__":
    main()
