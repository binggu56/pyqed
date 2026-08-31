#!/usr/bin/env python3
"""SO2 CASCI CGLDR benchmark with selected secondary coordinates.

The cached SO2 linked-LDR scans in ``examples/namd`` use valence coordinates
``(r1, r2, theta)``.  This script turns such a scan into CGLDR data with either
``r1`` and ``r2`` as primary sampled coordinates and ``theta`` as the secondary
expanded coordinate, or with ``q_s = (r1 + r2) / sqrt(2)`` and ``theta`` as
primary sampled coordinates and ``q_a = (r1 - r2) / sqrt(2)`` as the secondary
expanded coordinate. A sparse one-primary/two-secondary model instead samples
only ``q_s`` and uses axial three-anchor expansions for ``(theta, q_a)``.

For a matched ``(q_s, q_a, theta)`` grid, the electronic Hamiltonian along
theta is represented in one theta-center CASCI reference frame using analytic
first- and second-order Hamiltonian derivatives,

    H(q) = H0 + F_q (q - q_c) + 1/2 G_qq (q - q_c)^2.

The curvilinear theta Hessian includes both the projected Cartesian Hessian and
the coordinate-curvature correction.  The sampled ``(q_s, q_a)`` states and
their overlaps are calculated with the same CASCI model.  The older cached-scan
fit remains available only for legacy valence-coordinate scans.  The default
CAS(6e,6o) space keeps the near-degenerate occupied and virtual SO2 orbitals on
the same side of each active-space boundary.

For an expanded ``q_a``, the default is the same single-center quadratic
model.  Selecting multiple odd-numbered anchors evaluates relaxed-PT ``H/F/G``
at bracketing nodes and the central ``q_a`` node, transports the side matrices
into the central electronic frame, and constructs a piecewise-quintic
interpolant. Inner anchor stencils use the outer anchors' quadratic
continuation beyond the interpolation interval.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyqed.dvr import DVR, LegendreDVR, SineDVR
from pyqed.ldr import CGLDR, CGLDRElectronicData, ElectronicPartition
from pyqed.ldr import SeparableHamiltonian
from pyqed.mps.mps import gaussian_state
from pyqed.namd.triatomic import Triatom
from pyqed.units import au2fs


DEFAULT_SCAN_DIR = (
    Path(__file__).resolve().parents[1]
    / "namd"
    / "so2_3d_s2_9x9x9_100fs_sine_legendre_linked_ldr"
)
SQRT2 = np.sqrt(2.0)
REFERENCE_BOND = 2.70
REFERENCE_THETA_DEG = 119.5
REFERENCE_BOND_WIDTH = 0.16
REFERENCE_THETA_WIDTH_DEG = 7.5
DEFAULT_NCAS = 6
DEFAULT_NELECAS = 6
MIN_ACTIVE_GAP = 5.0e-3


@dataclass(frozen=True)
class SO2LinkedScan:
    solver: Triatom
    apes: np.ndarray
    r1: np.ndarray
    r2: np.ndarray
    theta: np.ndarray
    meta: dict


def so2_body_frame(r=2.70, theta=np.deg2rad(119.5)):
    return [
        ["O", (float(r), 0.0, 0.0)],
        ["S", (0.0, 0.0, 0.0)],
        ["O", (float(r) * np.cos(theta), float(r) * np.sin(theta), 0.0)],
    ]


def so2_qs_theta_body_frame(qs, theta, qa=0.0):
    r1 = (float(qs) + float(qa)) / SQRT2
    r2 = (float(qs) - float(qa)) / SQRT2
    return [
        ["O", (r1, 0.0, 0.0)],
        ["S", (0.0, 0.0, 0.0)],
        ["O", (r2 * np.cos(theta), r2 * np.sin(theta), 0.0)],
    ]


def so2_qa_mode(theta):
    return np.array(
        [
            [1.0 / SQRT2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-np.cos(theta) / SQRT2, -np.sin(theta) / SQRT2, 0.0],
        ],
        dtype=float,
    )


def so2_theta_modes(qs, qa, theta):
    """Return ``dR/dtheta`` and ``d2R/dtheta2`` at fixed ``(q_s, q_a)``."""
    r2 = (float(qs) - float(qa)) / SQRT2
    tangent = np.zeros((3, 3), dtype=float)
    curvature = np.zeros((3, 3), dtype=float)
    tangent[2, :2] = (-r2 * np.sin(theta), r2 * np.cos(theta))
    curvature[2, :2] = (-r2 * np.cos(theta), -r2 * np.sin(theta))
    return np.stack((tangent, curvature))


def so2_theta_qa_modes(qs, qa, theta):
    """Return first and second geometry derivatives for ``(theta, q_a)``."""
    theta_tangent, theta_curvature = so2_theta_modes(qs, qa, theta)
    first = np.stack((theta_tangent, so2_qa_mode(theta)))
    curvature = np.zeros((2, 2, 3, 3), dtype=float)
    curvature[0, 0] = theta_curvature
    mixed = np.zeros((3, 3), dtype=float)
    mixed[2, :2] = (
        np.sin(theta) / SQRT2,
        -np.cos(theta) / SQRT2,
    )
    curvature[0, 1] = curvature[1, 0] = mixed
    return first, curvature


def theta_vibronic_couplings(
    point,
    state_ids,
    qs,
    qa,
    theta,
    *,
    moving_basis="rhf-relaxed-pt",
    backend="native",
):
    r"""Return analytic ``dH/dtheta`` and curvilinear ``d2H/dtheta2``.

    For a curvilinear coordinate,

    ``d2H/dtheta2 = R' G R' + R'' F``.
    """
    first, second = point.vibronic_couplings(
        state_ids=state_ids,
        modes=so2_theta_modes(qs, qa, theta),
        moving_basis=moving_basis,
        backend=backend,
    )
    return first[..., 0], second[..., 0, 0] + first[..., 1]


def theta_qa_vibronic_couplings(
    point,
    state_ids,
    qs,
    qa,
    theta,
    *,
    moving_basis="rhf-relaxed-pt",
    backend="native",
):
    r"""Return analytical coordinate ``F`` and ``G`` for ``(theta, q_a)``."""
    first_modes, curvature = so2_theta_qa_modes(qs, qa, theta)
    modes = np.concatenate(
        (
            first_modes,
            curvature[0, 0][None, ...],
            curvature[0, 1][None, ...],
        )
    )
    raw_first, raw_second = point.vibronic_couplings(
        state_ids=state_ids,
        modes=modes,
        moving_basis=moving_basis,
        backend=backend,
    )
    first = np.asarray(raw_first[..., :2])
    second = np.array(raw_second[..., :2, :2], copy=True)
    second[..., 0, 0] += raw_first[..., 2]
    second[..., 0, 1] += raw_first[..., 3]
    second[..., 1, 0] += raw_first[..., 3]
    return first, second


def infer_sine_domain(nodes):
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size < 2:
        raise ValueError("Need at least two sine-DVR nodes to infer a domain.")
    spacing = float(np.mean(np.diff(nodes)))
    return float(nodes[0] - spacing), float(nodes[-1] + spacing)


def infer_legendre_domain(nodes):
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size < 2:
        raise ValueError("Need at least two Legendre-DVR nodes to infer a domain.")
    roots, _weights = np.polynomial.legendre.leggauss(nodes.size)
    half_width = float((nodes[-1] - nodes[0]) / (roots[-1] - roots[0]))
    midpoint = float(0.5 * (nodes[0] + nodes[-1]))
    return midpoint - half_width, midpoint + half_width


def load_so2_linked_scan(scan_dir, *, path_average=True, mass_r=1.0, mass_theta=1.0):
    scan_dir = Path(scan_dir)
    valence_observables = (
        scan_dir / "so2_3d_sine_legendre_linked_ldr_observables.npz"
    )
    transformed_observables = (
        scan_dir / "so2_3d_qs_qa_theta_linked_ldr_observables.npz"
    )
    transformed_grid = scan_dir / "so2_3d_qs_qa_theta_grid.npz"
    if transformed_grid.exists() or transformed_observables.exists():
        observables_path = (
            transformed_grid
            if transformed_grid.exists()
            else transformed_observables
        )
        coordinate_system = "qs-qa-theta"
        axis_keys = ("qs", "qa", "theta")
    else:
        observables_path = valence_observables
        coordinate_system = "valence"
        axis_keys = ("r1", "r2", "theta")
    apes_path = scan_dir / "apes.npz"
    links_path = scan_dir / "overlap_links.npz"
    missing = [
        str(path)
        for path in (observables_path, apes_path, links_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing SO2 cached scan files: " + ", ".join(missing))

    with np.load(observables_path) as archive:
        r1 = np.asarray(archive[axis_keys[0]], dtype=float)
        r2 = np.asarray(archive[axis_keys[1]], dtype=float)
        theta = np.asarray(archive[axis_keys[2]], dtype=float)

    with np.load(apes_path, allow_pickle=True) as archive:
        apes = np.asarray(archive["data"], dtype=float)
        meta = dict(archive["meta"].item()) if "meta" in archive else {}

    expected_shape = (r1.size, r2.size, theta.size)
    if apes.shape[:3] != expected_shape:
        raise ValueError(f"APES grid shape {apes.shape[:3]} != {expected_shape}")
    nstates = int(apes.shape[-1])

    r1_domain = infer_sine_domain(r1)
    r2_domain = infer_sine_domain(r2)
    theta_domain = infer_legendre_domain(theta)
    triatom = Triatom(
        so2_body_frame(),
        basis=meta.get("basis", "631g*"),
        nstates=nstates,
        charge=0,
        spin=0,
        unit="bohr",
        dvr_type=["sine", "sine", "legendre"],
        coordinates=coordinate_system,
    )
    triatom.set_dvr(
        domains=[r1_domain, r2_domain, theta_domain],
        npts=[r1.size, r2.size, theta.size],
        dvr_type=["sine", "sine", "legendre"],
        dvr_params=[
            {"mass": mass_r},
            {"mass": mass_r},
            {"mass": mass_theta},
        ],
    )
    np.testing.assert_allclose(triatom.x[0], r1, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(triatom.x[1], r2, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(triatom.x[2], theta, rtol=0.0, atol=1.0e-10)

    with np.load(links_path, allow_pickle=True) as archive:
        triatom.overlap_links = triatom._unpack_overlap_links(
            archive["axes"],
            archive["indices"],
            archive["data"],
        )
    triatom.overlap_matrix = None
    triatom.overlap_path_average = bool(path_average)
    triatom.apes = apes

    return SO2LinkedScan(
        solver=triatom,
        apes=apes,
        r1=r1,
        r2=r2,
        theta=theta,
        meta={**meta, "coordinates": coordinate_system},
    )


def nearest_index(values, target):
    values = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(values - float(target))))


def parse_state_ids(text, nstates):
    if text is None:
        return tuple(range(nstates))
    if isinstance(text, str):
        text = text.strip().lower()
        if text in {"", "all", "*"}:
            return tuple(range(nstates))
        state_ids = tuple(
            int(item.strip()) for item in text.split(",") if item.strip()
        )
    else:
        state_ids = tuple(int(item) for item in text)
    if not state_ids:
        raise ValueError("state_ids cannot be empty")
    if len(set(state_ids)) != len(state_ids):
        raise ValueError("state_ids must be unique")
    if min(state_ids) < 0 or max(state_ids) >= nstates:
        raise ValueError(f"state_ids must be between 0 and {nstates - 1}")
    return state_ids


def active_state_indices(state_indices, nstates):
    if state_indices is None:
        return np.arange(nstates, dtype=int)
    state_indices = np.asarray(tuple(state_indices), dtype=int)
    if state_indices.ndim != 1 or state_indices.size == 0:
        raise ValueError("state_indices must be a nonempty one-dimensional list")
    if len(set(state_indices.tolist())) != state_indices.size:
        raise ValueError("state_indices must be unique")
    if np.any(state_indices < 0) or np.any(state_indices >= nstates):
        raise ValueError(f"state_indices must be between 0 and {nstates - 1}")
    return state_indices


def parse_grid_indices(text, size, *, name):
    if text is None:
        return tuple(range(size))
    if isinstance(text, str):
        text = text.strip().lower()
        if text in {"", "all", "*"}:
            return tuple(range(size))
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    else:
        values = tuple(int(item) for item in text)
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")
    if min(values) < 0 or max(values) >= size:
        raise ValueError(f"{name} entries must be between 0 and {size - 1}")
    return values


def polar_unitary(matrix):
    u, _singular_values, vh = np.linalg.svd(
        np.asarray(matrix, dtype=complex),
        full_matrices=False,
    )
    return u @ vh


def linked_overlap_block(solver, bra_idx, ket_idx, links, state_indices):
    overlap = solver._linked_overlap_between(
        bra_idx,
        ket_idx,
        links,
        solver.nstates,
    )
    return overlap[np.ix_(state_indices, state_indices)]


def linked_unitary_transport(solver, bra_idx, ket_idx, links, state_indices):
    overlap = linked_overlap_block(solver, bra_idx, ket_idx, links, state_indices)
    return polar_unitary(overlap)


def sampled_overlap_matrix(scan: SO2LinkedScan, theta_index, *, state_indices=None):
    solver = scan.solver
    n1, n2, _ntheta = solver.nx
    state_indices = active_state_indices(state_indices, solver.nstates)
    nstates = state_indices.size
    links = solver.overlap_links
    sampled = list(np.ndindex(n1, n2))
    flat_index = {idx: pos for pos, idx in enumerate(sampled)}
    blocks = np.zeros((n1 * n2, nstates, n1 * n2, nstates), dtype=complex)
    eye = np.eye(nstates, dtype=complex)
    for bra_pos, bra in enumerate(sampled):
        blocks[bra_pos, :, bra_pos, :] = eye
        bra_idx = (bra[0], bra[1], theta_index)
        for ket in sampled[bra_pos + 1:]:
            ket_pos = flat_index[ket]
            ket_idx = (ket[0], ket[1], theta_index)
            overlap = solver._linked_overlap_between(
                bra_idx,
                ket_idx,
                links,
                solver.nstates,
            )
            overlap = polar_unitary(
                overlap[np.ix_(state_indices, state_indices)]
            )
            blocks[bra_pos, :, ket_pos, :] = overlap
            blocks[ket_pos, :, bra_pos, :] = overlap.conj().T
    return blocks.reshape(n1, n2, nstates, n1, n2, nstates)


def theta_center_hamiltonian(
    scan: SO2LinkedScan,
    theta_index,
    *,
    energy_shift=0.0,
    state_indices=None,
):
    solver = scan.solver
    n1, n2, ntheta = solver.nx
    state_indices = active_state_indices(state_indices, solver.nstates)
    nstates = state_indices.size
    operators = np.empty((n1, n2, ntheta, nstates, nstates), dtype=complex)
    shifted_apes = (
        np.asarray(scan.apes, dtype=float)[..., state_indices]
        - float(energy_shift)
    )
    links = solver.overlap_links

    for i, j, k in np.ndindex(n1, n2, ntheta):
        transport = linked_unitary_transport(
            solver,
            (i, j, theta_index),
            (i, j, k),
            links,
            state_indices,
        )
        local = transport @ np.diag(shifted_apes[i, j, k]) @ transport.conj().T
        operators[i, j, k] = 0.5 * (local + local.conj().T)

    return SeparableHamiltonian(
        operators=operators,
        factors=(np.eye(ntheta),),
    )


def theta_anchor_indices(theta, theta_index, anchor_count):
    theta = np.asarray(theta, dtype=float)
    anchor_count = int(anchor_count)
    if anchor_count < 3:
        raise ValueError("theta_anchor_count must be at least 3")
    if anchor_count > theta.size:
        raise ValueError("theta_anchor_count cannot exceed the theta grid size")
    order = np.argsort(np.abs(theta - theta[theta_index]))
    selected = np.sort(order[:anchor_count])
    if theta_index not in selected:
        selected = np.sort(np.r_[selected[:-1], theta_index])
    return selected.astype(int)


def theta_quadratic_derivatives(
    scan: SO2LinkedScan,
    theta_index,
    *,
    anchor_count=3,
    energy_shift=0.0,
    state_indices=None,
):
    solver = scan.solver
    n1, n2, _ntheta = solver.nx
    state_indices = active_state_indices(state_indices, solver.nstates)
    nstates = state_indices.size
    links = solver.overlap_links
    anchors = theta_anchor_indices(scan.theta, theta_index, anchor_count)
    displacement = scan.theta[anchors] - scan.theta[theta_index]
    design = np.column_stack(
        [
            np.ones_like(displacement),
            displacement,
            0.5 * displacement**2,
        ]
    )
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("theta anchors do not span a quadratic fit")

    energies = (
        scan.apes[:, :, theta_index, state_indices]
        - float(energy_shift)
    )
    gradients = np.empty((n1, n2, 1, nstates, nstates), dtype=complex)
    hessians = np.empty((n1, n2, 1, 1, nstates, nstates), dtype=complex)
    shifted_apes = (
        np.asarray(scan.apes, dtype=float)[..., state_indices]
        - float(energy_shift)
    )

    for i, j in np.ndindex(n1, n2):
        samples = []
        for k in anchors:
            transport = linked_unitary_transport(
                solver,
                (i, j, theta_index),
                (i, j, int(k)),
                links,
                state_indices,
            )
            local = transport @ np.diag(shifted_apes[i, j, k]) @ transport.conj().T
            samples.append(0.5 * (local + local.conj().T))
        samples = np.asarray(samples)
        coefficients, *_ = np.linalg.lstsq(
            design,
            samples.reshape(len(anchors), -1),
            rcond=None,
        )
        gradient = coefficients[1].reshape(nstates, nstates)
        hessian = coefficients[2].reshape(nstates, nstates)
        gradients[i, j, 0] = 0.5 * (gradient + gradient.conj().T)
        hessians[i, j, 0, 0] = 0.5 * (hessian + hessian.conj().T)

    return energies, gradients, hessians, anchors


def symmetric_stretch_nodes(scan, indices=None):
    qs, _qa = transformed_stretch_nodes(scan)
    indices = parse_grid_indices(indices, qs.size, name="qs_indices")
    return qs[np.asarray(indices, dtype=int)], indices


def unique_sorted_grid_nodes(values, *, atol=1.0e-12):
    values = np.sort(np.asarray(values, dtype=float).ravel())
    if values.size == 0:
        return values
    nodes = [float(values[0])]
    for value in values[1:]:
        if abs(float(value) - nodes[-1]) > atol:
            nodes.append(float(value))
    return np.asarray(nodes, dtype=float)


def transformed_stretch_nodes(scan):
    """Return the unique full-grid ``q_s`` and ``q_a`` nodes from ``r1/r2``."""
    if scan.r1.shape != scan.r2.shape or not np.allclose(scan.r1, scan.r2):
        raise ValueError("q_a mode requires matching r1 and r2 DVR grids")
    r1 = np.asarray(scan.r1, dtype=float)
    r2 = np.asarray(scan.r2, dtype=float)
    qs = unique_sorted_grid_nodes((r1[:, None] + r2[None, :]) / SQRT2)
    qa = unique_sorted_grid_nodes((r1[:, None] - r2[None, :]) / SQRT2)
    return qs, qa


def centered_grid_window(nodes, npts, *, center):
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 1 or nodes.size == 0:
        raise ValueError("nodes must be a nonempty one-dimensional array")
    if npts is None:
        return nodes
    npts = int(npts)
    if npts < 1:
        raise ValueError("npts must be positive")
    if npts > nodes.size:
        raise ValueError(f"npts={npts} exceeds available nodes={nodes.size}")
    center_index = int(np.argmin(np.abs(nodes - float(center))))
    start = center_index - npts // 2
    start = max(0, min(start, nodes.size - npts))
    return nodes[start:start + npts]


def qa_axis_from_scan(scan, npts=None, half_width=None):
    if half_width is None:
        _qs, qa = transformed_stretch_nodes(scan)
        qa = centered_grid_window(qa, npts, center=0.0)
        if qa.size < 3:
            raise ValueError("qa_npts must be at least 3")
        return SineDVR(*infer_sine_domain(qa), qa.size)
    if npts is None:
        npts = scan.r1.size
    npts = int(npts)
    if npts < 3:
        raise ValueError("qa_npts must be at least 3")
    half_width = float(half_width)
    if half_width <= 0.0:
        raise ValueError("qa_half_width must be positive")
    return SineDVR(-half_width, half_width, npts)


def casci_reference_point(
    geometry,
    *,
    basis,
    charge,
    spin,
    unit,
    ncas,
    nelecas,
    nstates,
    scf_tol,
    scf_max_cycle,
    multiplicity=None,
    eri_workers=1,
    spin_root_cushion=8,
    direct_ci_eigensolver=None,
    direct_ci_auto_spin0=None,
):
    from pyqed import Molecule
    from pyqed.qchem.hf.rhf import RHF
    from pyqed.qchem.mcscf.casci import CASCI

    mol = Molecule(
        atom=geometry,
        basis=basis,
        charge=charge,
        spin=spin,
        unit=unit,
    )
    eri_workers = int(eri_workers)
    if eri_workers < 1:
        raise ValueError("eri_workers must be positive")
    if eri_workers > 1:
        mol.builtin_parallel = True
        mol.builtin_parallel_min_nao = 0
        mol.builtin_eri_workers = eri_workers
    mol.build(eri="dense")
    mf = RHF(mol, verbose=0).run(
        tol=scf_tol,
        max_cycle=scf_max_cycle,
    )
    casci = CASCI(
        mf,
        ncas=ncas,
        nelecas=nelecas,
        spin=spin,
        multiplicity=multiplicity,
        verbose=0,
    )
    casci.spin_root_cushion = int(spin_root_cushion)
    if direct_ci_eigensolver is not None:
        casci.direct_ci_eigensolver = str(direct_ci_eigensolver)
    if direct_ci_auto_spin0 is not None:
        casci.direct_ci_auto_spin0 = bool(direct_ci_auto_spin0)
    casci.run(nstates=nstates, spin_root_cushion=spin_root_cushion)
    return casci


def casci_overlap_active(left, right, state_ids, *, polar=False):
    if hasattr(left, "overlap"):
        overlap = left.overlap(right)
    else:
        from pyqed.qchem.mcscf.casci import overlap as casci_overlap

        overlap = casci_overlap(left, right)
    overlap = np.asarray(overlap, dtype=complex)
    state_ids = np.asarray(state_ids, dtype=int)
    active = overlap[np.ix_(state_ids, state_ids)]
    return polar_unitary(active) if polar else active


def active_space_gaps(point):
    """Return the lower and upper canonical-MO gaps around a CAS space."""
    energies = np.asarray(point.mf.mo_energy, dtype=float)
    first = int(point.ncore)
    stop = first + int(point.ncas)
    lower = np.inf if first == 0 else energies[first] - energies[first - 1]
    upper = np.inf if stop == len(energies) else energies[stop] - energies[stop - 1]
    return float(abs(lower)), float(abs(upper))


def require_smooth_active_space(point, *, min_gap=MIN_ACTIVE_GAP):
    """Reject canonical active spaces whose relaxed derivatives have a pole."""
    gaps = active_space_gaps(point)
    labels = ("core-active", "active-external")
    boundary = int(np.argmin(gaps))
    if gaps[boundary] < float(min_gap):
        raise ValueError(
            f"CASCI {labels[boundary]} orbital gap {gaps[boundary]:.3e} Eh "
            f"is below the analytical F/G threshold {float(min_gap):.3e} Eh. "
            "The canonical active space is not differentiable here; enlarge "
            "or shift it so the near-degenerate pair lies in one orbital space."
        )
    return gaps


def _uses_relaxed_orbitals(moving_basis):
    key = str(moving_basis).lower().replace("_", "-")
    return key in {
        "rhf-relaxed",
        "cphf",
        "relaxed",
        "rhf-relaxed-pt",
        "relaxed-pt",
        "parallel-transport",
        "parallel",
    }


def _build_qa_casci_anchor(task):
    (
        i,
        j,
        anchor_index,
        qs_value,
        theta_value,
        qa_value,
        state_ids,
        basis,
        ncas,
        nelecas,
        casci_nstates,
        scf_tol,
        scf_max_cycle,
        multiplicity,
        moving_basis,
        derivative_backend,
        derivative_workers,
    ) = task
    point = casci_reference_point(
        so2_qs_theta_body_frame(qs_value, theta_value, qa_value),
        basis=basis,
        charge=0,
        spin=0,
        unit="bohr",
        ncas=ncas,
        nelecas=nelecas,
        nstates=casci_nstates,
        scf_tol=scf_tol,
        scf_max_cycle=scf_max_cycle,
        multiplicity=multiplicity,
        eri_workers=derivative_workers,
    )
    if _uses_relaxed_orbitals(moving_basis):
        require_smooth_active_space(point)
    state_indices = np.asarray(state_ids, dtype=int)
    energies = np.asarray(point.e_tot, dtype=float)[state_indices]
    first, second = point.vibronic_couplings(
        state_ids=state_ids,
        modes=so2_qa_mode(theta_value)[None, ...],
        moving_basis=moving_basis,
        backend=derivative_backend,
    )
    return (
        i,
        j,
        anchor_index,
        point.frame(),
        energies,
        first[..., 0],
        second[..., 0, 0],
    )


def _build_qs_casci_anchor(task):
    (
        i,
        anchor_index,
        qs_value,
        theta_value,
        qa_value,
        state_ids,
        basis,
        ncas,
        nelecas,
        casci_nstates,
        scf_tol,
        scf_max_cycle,
        multiplicity,
        moving_basis,
        derivative_backend,
        derivative_workers,
    ) = task
    point = casci_reference_point(
        so2_qs_theta_body_frame(qs_value, theta_value, qa_value),
        basis=basis,
        charge=0,
        spin=0,
        unit="bohr",
        ncas=ncas,
        nelecas=nelecas,
        nstates=casci_nstates,
        scf_tol=scf_tol,
        scf_max_cycle=scf_max_cycle,
        multiplicity=multiplicity,
        eri_workers=derivative_workers,
    )
    if _uses_relaxed_orbitals(moving_basis):
        require_smooth_active_space(point)
    state_indices = np.asarray(state_ids, dtype=int)
    energies = np.asarray(point.e_tot, dtype=float)[state_indices]
    first, second = theta_qa_vibronic_couplings(
        point,
        state_ids,
        qs_value,
        qa_value,
        theta_value,
        moving_basis=moving_basis,
        backend=derivative_backend,
    )
    return i, anchor_index, point.frame(), energies, first, second


def _build_theta_casci_anchor(task):
    (
        i,
        j,
        qs_value,
        qa_value,
        theta_value,
        state_ids,
        basis,
        ncas,
        nelecas,
        casci_nstates,
        scf_tol,
        scf_max_cycle,
        multiplicity,
        moving_basis,
        derivative_backend,
        derivative_workers,
    ) = task
    point = casci_reference_point(
        so2_qs_theta_body_frame(qs_value, theta_value, qa_value),
        basis=basis,
        charge=0,
        spin=0,
        unit="bohr",
        ncas=ncas,
        nelecas=nelecas,
        nstates=casci_nstates,
        scf_tol=scf_tol,
        scf_max_cycle=scf_max_cycle,
        multiplicity=multiplicity,
        eri_workers=derivative_workers,
    )
    if _uses_relaxed_orbitals(moving_basis):
        require_smooth_active_space(point)
    state_indices = np.asarray(state_ids, dtype=int)
    energies = np.asarray(point.e_tot, dtype=float)[state_indices]
    first, second = theta_vibronic_couplings(
        point,
        state_ids,
        qs_value,
        qa_value,
        theta_value,
        moving_basis=moving_basis,
        backend=derivative_backend,
    )
    return i, j, point.frame(), energies, first, second


def build_theta_cgldr_from_casci(
    scan: SO2LinkedScan,
    *,
    theta_center_deg=119.5,
    state_ids=None,
    qs_indices=None,
    qa_indices=None,
    max_rank=64,
    energy_reference="minimum",
    basis="sto-3g",
    ncas=DEFAULT_NCAS,
    nelecas=DEFAULT_NELECAS,
    multiplicity=1,
    scf_tol=1.0e-8,
    scf_max_cycle=80,
    workers=1,
    moving_basis="rhf-relaxed-pt",
    derivative_backend="native",
    derivative_workers=1,
    kinetic_model="valence",
    kinetic_exp_order=10,
    kinetic_exp_scale=1,
    polar_overlap=False,
    electronic_data=None,
):
    """Build CASCI CGLDR with sampled ``(q_s,q_a)`` and expanded ``theta``."""
    if scan.solver.coordinates != "qs-qa-theta":
        raise ValueError(
            "Analytic theta CGLDR requires a matched (q_s, q_a, theta) grid."
        )
    if not isinstance(workers, (int, np.integer)) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    if int(derivative_workers) <= 0:
        raise ValueError("derivative_workers must be positive")
    state_ids = parse_state_ids(state_ids, scan.solver.nstates)
    casci_nstates = max(state_ids) + 1
    qs_indices = parse_grid_indices(
        qs_indices, scan.r1.size, name="qs_indices"
    )
    qa_indices = parse_grid_indices(
        qa_indices, scan.r2.size, name="qa_indices"
    )
    qs = scan.r1[np.asarray(qs_indices, dtype=int)]
    qa = scan.r2[np.asarray(qa_indices, dtype=int)]
    theta_index = nearest_index(scan.theta, np.deg2rad(theta_center_deg))
    theta_center = float(scan.theta[theta_index])

    axes = (
        SineDVR(*infer_sine_domain(qs), len(qs)),
        SineDVR(*infer_sine_domain(qa), len(qa)),
        LegendreDVR(*infer_legendre_domain(scan.theta), len(scan.theta)),
    )
    dvr = DVR.from_axes(axes, names=("qs", "qa", "theta"))
    partition = ElectronicPartition(
        sampled=("qs", "qa"),
        expanded=("theta",),
        center=(theta_center,),
    )
    kinetic_model = str(kinetic_model).lower().replace("_", "-")
    if kinetic_model == "valence":
        kinetic_solver = Triatom(
            so2_body_frame(),
            basis=basis,
            nstates=scan.solver.nstates,
            charge=0,
            spin=0,
            unit="bohr",
            coordinates="qs-qa-theta",
            dvr_type=["sine", "sine", "legendre"],
        )
        kinetic_solver.set_dvr(
            domains=[
                infer_sine_domain(qs),
                infer_sine_domain(qa),
                infer_legendre_domain(scan.theta),
            ],
            npts=[len(qs), len(qa), len(scan.theta)],
            dvr_type=["sine", "sine", "legendre"],
        )
        nuclear_kinetic_mpo = kinetic_solver.buildK_product_mpo(
            max_rank=max_rank,
            symmetrize=True,
        )
        kinetic_label = "valence"
    elif kinetic_model in {"product", "product-dvr", "dvr"}:
        nuclear_kinetic_mpo = None
        kinetic_label = "product-dvr"
    else:
        raise ValueError("kinetic_model must be 'valence' or 'product-dvr'")
    dynamics = CGLDR(
        dvr,
        partition,
        state_ids=state_ids,
        tt_options={"max_rank": max_rank},
        nuclear_kinetic_mpo=nuclear_kinetic_mpo,
        kinetic_exponential_options={
            "order": kinetic_exp_order,
            "scale": kinetic_exp_scale,
        },
    )
    if electronic_data is not None:
        dynamics.set_electronic_data(electronic_data, tolerance=1.0e-6)
        return dynamics, electronic_data

    nqs, nqa, nactive = len(qs), len(qa), len(state_ids)
    points = np.empty((nqs, nqa), dtype=object)
    energies = np.empty((nqs, nqa, nactive), dtype=float)
    gradients = np.empty((nqs, nqa, 1, nactive, nactive), dtype=complex)
    hessians = np.empty((nqs, nqa, 1, 1, nactive, nactive), dtype=complex)
    tasks = [
        (
            i,
            j,
            float(qs_value),
            float(qa_value),
            theta_center,
            tuple(state_ids),
            basis,
            int(ncas),
            int(nelecas),
            int(casci_nstates),
            float(scf_tol),
            int(scf_max_cycle),
            multiplicity,
            moving_basis,
            derivative_backend,
            int(derivative_workers),
        )
        for i, qs_value in enumerate(qs)
        for j, qa_value in enumerate(qa)
    ]
    total = len(tasks)
    if workers == 1:
        results = map(_build_theta_casci_anchor, tasks)
        for count, result in enumerate(results, start=1):
            i, j, point, point_energies, first, second = result
            points[i, j] = point
            energies[i, j] = point_energies
            gradients[i, j, 0] = first
            hessians[i, j, 0, 0] = second
            print(f"[CASCI theta derivatives] {count}/{total}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, total)) as executor:
            futures = [
                executor.submit(_build_theta_casci_anchor, task)
                for task in tasks
            ]
            for count, future in enumerate(as_completed(futures), start=1):
                i, j, point, point_energies, first, second = future.result()
                points[i, j] = point
                energies[i, j] = point_energies
                gradients[i, j, 0] = first
                hessians[i, j, 0, 0] = second
                print(
                    f"[CASCI theta derivatives] {count}/{total} "
                    f"({workers} workers)",
                    flush=True,
                )

    if energy_reference in {"minimum", "theta-center-minimum"}:
        energy_shift = float(np.nanmin(energies))
    elif energy_reference == "zero":
        energy_shift = 0.0
    else:
        raise ValueError(
            "energy_reference must be 'minimum', 'theta-center-minimum', or 'zero'"
        )
    energies -= energy_shift

    overlaps = np.empty(
        (nqs, nqa, nactive, nqs, nqa, nactive),
        dtype=complex,
    )
    flat = list(np.ndindex(nqs, nqa))
    for bra_pos, bra in enumerate(flat):
        overlaps[bra + (slice(None),) + bra + (slice(None),)] = np.eye(nactive)
        for ket in flat[bra_pos + 1:]:
            block = casci_overlap_active(
                points[bra], points[ket], state_ids, polar=polar_overlap
            )
            overlaps[bra + (slice(None),) + ket + (slice(None),)] = block
            overlaps[ket + (slice(None),) + bra + (slice(None),)] = block.conj().T

    metadata = {
        "molecule": "SO2",
        "source": "live CASCI theta derivative grid",
        "source_meta": scan.meta,
        "sampled_coordinates": ["qs", "qa"],
        "expanded_coordinates": ["theta"],
        "state_ids": list(state_ids),
        "qs_indices": list(qs_indices),
        "qa_indices": list(qa_indices),
        "theta_center_index": int(theta_index),
        "theta_center_rad": theta_center,
        "theta_center_deg": float(np.rad2deg(theta_center)),
        "theta_anchor_count": 1,
        "theta_anchor_degrees": [float(np.rad2deg(theta_center))],
        "theta_model": "analytic quadratic",
        "energy_shift": energy_shift,
        "derivative_source": "CASCI.vibronic_couplings",
        "theta_hessian": "R' G R' + R'' F",
        "moving_basis": moving_basis,
        "overlap_form": "polar" if polar_overlap else "raw",
        "secondary_mode": "theta",
        "basis": basis,
        "ncas": int(ncas),
        "nelecas": int(nelecas),
        "active_gap_threshold": MIN_ACTIVE_GAP,
        "multiplicity": multiplicity,
        "workers": workers,
        "derivative_workers": int(derivative_workers),
        "derivative_backend": derivative_backend,
        "kinetic_model": kinetic_label,
        "kinetic_coordinates": ["qs", "qa", "theta"],
        "kinetic_exp_order": int(kinetic_exp_order),
        "kinetic_exp_scale": int(kinetic_exp_scale),
    }
    data = CGLDRElectronicData(
        energies=energies,
        overlaps=overlaps,
        hamiltonian_gradients=gradients,
        hamiltonian_hessians=hessians,
        reactive_grids=(qs, qa),
        expanded_grids=(scan.theta,),
        metadata=metadata,
    )
    dynamics.set_electronic_data(data, tolerance=1.0e-6)
    return dynamics, data


def build_qa_cgldr_from_casci(
    scan: SO2LinkedScan,
    *,
    state_ids=None,
    qs_indices=None,
    theta_indices=None,
    qa_npts=None,
    qa_anchor_count=1,
    qa_anchor_indices=None,
    max_rank=64,
    energy_reference="minimum",
    basis="sto-3g",
    ncas=DEFAULT_NCAS,
    nelecas=DEFAULT_NELECAS,
    multiplicity=1,
    scf_tol=1.0e-8,
    scf_max_cycle=80,
    workers=1,
    moving_basis="rhf-relaxed-pt",
    derivative_backend="native",
    derivative_workers=1,
    qa_half_width=None,
    kinetic_model="valence",
    kinetic_exp_order=10,
    kinetic_exp_scale=1,
    kinetic_svd_tol=0.0,
    polar_overlap=False,
    electronic_data=None,
    anchor_cache_dir=None,
    reuse_anchor_cache=False,
):
    if not isinstance(workers, (int, np.integer)) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    if int(derivative_workers) <= 0:
        raise ValueError("derivative_workers must be positive")
    state_ids = parse_state_ids(state_ids, scan.solver.nstates)
    casci_nstates = max(state_ids) + 1
    qs, qs_indices = symmetric_stretch_nodes(scan, qs_indices)
    theta_indices = parse_grid_indices(
        theta_indices,
        scan.theta.size,
        name="theta_indices",
    )
    theta = scan.theta[np.asarray(theta_indices, dtype=int)]
    qa_axis = qa_axis_from_scan(scan, qa_npts, half_width=qa_half_width)
    qa_anchor_count = int(qa_anchor_count)
    if qa_anchor_count != 1 and (
        qa_anchor_count < 3 or qa_anchor_count % 2 == 0
    ):
        raise ValueError("qa_anchor_count must be 1 or an odd integer >= 3")
    if qa_anchor_count > qa_axis.npts:
        raise ValueError("qa_anchor_count cannot exceed the q_a grid size")
    qa_center_index = int(np.argmin(np.abs(qa_axis.x)))
    if qa_anchor_indices is None:
        if qa_anchor_count == 1:
            qa_anchor_indices = np.array([qa_center_index], dtype=int)
        else:
            qa_anchor_indices = np.rint(
                np.linspace(0, qa_axis.npts - 1, qa_anchor_count)
            ).astype(int)
    else:
        qa_anchor_indices = np.asarray(
            parse_grid_indices(
                qa_anchor_indices,
                qa_axis.npts,
                name="qa_anchor_indices",
            ),
            dtype=int,
        )
    if qa_anchor_indices.size != qa_anchor_count:
        raise ValueError(
            "qa_anchor_indices must contain qa_anchor_count entries"
        )
    if np.any(np.diff(qa_anchor_indices) <= 0):
        raise ValueError("qa_anchor_indices must be strictly increasing")
    center_matches = np.flatnonzero(qa_anchor_indices == qa_center_index)
    if center_matches.size != 1:
        raise ValueError("qa_anchor_indices must include the central q_a node")
    center_anchor = int(center_matches[0])
    if qa_anchor_count > 1 and center_anchor != qa_anchor_count // 2:
        raise ValueError("q_a anchors must symmetrically bracket the central node")
    qa_anchors = np.asarray(qa_axis.x[qa_anchor_indices], dtype=float)
    qa_extrapolation = (
        "error"
        if qa_anchor_indices[0] == 0
        and qa_anchor_indices[-1] == qa_axis.npts - 1
        else "quadratic"
    )

    dvr = DVR.from_axes(
        (
            SineDVR(*infer_sine_domain(qs), len(qs)),
            LegendreDVR(*infer_legendre_domain(theta), len(theta)),
            qa_axis,
        ),
        names=("qs", "theta", "qa"),
    )
    partition = ElectronicPartition(
        sampled=("qs", "theta"),
        expanded=("qa",),
        center=(0.0,),
    )
    kinetic_model = str(kinetic_model).lower().replace("_", "-")
    if kinetic_model == "valence":
        nuclear_kinetic_mpo = scan.solver.buildK_qsqa_mpo(
            dvr.axes,
            max_rank=None,
            symmetrize=True,
            svd_tol=kinetic_svd_tol,
        )
        kinetic_label = "transformed-valence"
    elif kinetic_model in {"product", "product-dvr", "dvr"}:
        nuclear_kinetic_mpo = None
        kinetic_label = "product-dvr"
    else:
        raise ValueError("kinetic_model must be 'valence' or 'product-dvr'")
    dynamics = CGLDR(
        dvr,
        partition,
        state_ids=state_ids,
        tt_options={"max_rank": max_rank},
        nuclear_kinetic_mpo=nuclear_kinetic_mpo,
        kinetic_exponential_options={
            "order": kinetic_exp_order,
            "scale": kinetic_exp_scale,
        },
    )
    if electronic_data is not None:
        dynamics.set_electronic_data(electronic_data, tolerance=1.0e-6)
        return dynamics, electronic_data

    nqs = len(qs)
    ntheta = len(theta)
    nactive = len(state_ids)
    nanchors = len(qa_anchors)
    points = np.empty((nqs, ntheta, nanchors), dtype=object)
    anchor_energies = np.empty(
        (nqs, ntheta, nanchors, nactive),
        dtype=float,
    )
    anchor_gradients = np.empty(
        (nqs, ntheta, nanchors, nactive, nactive),
        dtype=complex,
    )
    anchor_hessians = np.empty_like(anchor_gradients)

    tasks = [
        (
            i,
            j,
            anchor_index,
            float(qs_value),
            float(theta_value),
            float(qa_value),
            tuple(state_ids),
            basis,
            int(ncas),
            int(nelecas),
            int(casci_nstates),
            float(scf_tol),
            int(scf_max_cycle),
            multiplicity,
            moving_basis,
            derivative_backend,
            int(derivative_workers),
        )
        for i, qs_value in enumerate(qs)
        for j, theta_value in enumerate(theta)
        for anchor_index, qa_value in enumerate(qa_anchors)
    ]
    total = len(tasks)

    cache_dir = None
    if anchor_cache_dir is not None:
        cache_dir = Path(anchor_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(task):
        i, j, anchor_index = task[:3]
        qa_index = int(qa_anchor_indices[anchor_index])
        return cache_dir / f"anchor_{i}_{j}_qa{qa_index}.pkl"

    def save_result(task, result):
        if cache_dir is None:
            return
        path = cache_path(task)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(result, stream, pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    def assign_result(result, task=None):
        i, j, anchor_index, point, point_energies, first, second = result
        if task is not None:
            expected_i, expected_j, anchor_index = task[:3]
            if (i, j) != (expected_i, expected_j):
                raise ValueError("Cached q_a anchor has incompatible sampled indices")
        points[i, j, anchor_index] = point
        anchor_energies[i, j, anchor_index] = point_energies
        anchor_gradients[i, j, anchor_index] = first
        anchor_hessians[i, j, anchor_index] = second

    pending = []
    completed = 0
    for task in tasks:
        path = cache_path(task) if cache_dir is not None else None
        if reuse_anchor_cache and path.is_file():
            with path.open("rb") as stream:
                assign_result(pickle.load(stream), task=task)
            completed += 1
        else:
            pending.append(task)
    if completed:
        print(
            f"[CASCI q_a derivatives] restored {completed}/{total} anchors",
            flush=True,
        )

    if workers == 1:
        for count, task in enumerate(pending, start=completed + 1):
            result = _build_qa_casci_anchor(task)
            assign_result(result)
            save_result(task, result)
            print(f"[CASCI q_a derivatives] {count}/{total}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
            futures = {
                executor.submit(_build_qa_casci_anchor, task): task
                for task in pending
            }
            for count, future in enumerate(
                as_completed(futures), start=completed + 1
            ):
                result = future.result()
                assign_result(result)
                save_result(futures[future], result)
                print(
                    f"[CASCI q_a derivatives] {count}/{total} "
                    f"({workers} workers)",
                    flush=True,
                )

    if energy_reference in {"minimum", "theta-center-minimum"}:
        energy_shift = float(np.nanmin(anchor_energies))
    elif energy_reference == "zero":
        energy_shift = 0.0
    else:
        raise ValueError(
            "energy_reference must be 'minimum', 'theta-center-minimum', or 'zero'"
        )
    shifted_anchor_energies = anchor_energies - energy_shift
    energies = shifted_anchor_energies[..., center_anchor, :]
    gradients = anchor_gradients[..., center_anchor, None, :, :]
    hessians = anchor_hessians[..., center_anchor, None, None, :, :]

    separable_hamiltonian = None
    if qa_anchor_count > 1:
        anchor_hamiltonians = np.empty_like(anchor_gradients)
        transported_gradients = np.empty_like(anchor_gradients)
        transported_hessians = np.empty_like(anchor_hessians)
        for index in np.ndindex(nqs, ntheta):
            reference = points[index + (center_anchor,)]
            for anchor_index in range(nanchors):
                if anchor_index == center_anchor:
                    transport = np.eye(nactive, dtype=complex)
                else:
                    transport = casci_overlap_active(
                        reference,
                        points[index + (anchor_index,)],
                        state_ids,
                        polar=True,
                    )
                local_hamiltonian = np.diag(
                    shifted_anchor_energies[index + (anchor_index,)]
                )
                anchor_hamiltonians[index + (anchor_index,)] = (
                    transport @ local_hamiltonian @ transport.conj().T
                )
                transported_gradients[index + (anchor_index,)] = (
                    transport
                    @ anchor_gradients[index + (anchor_index,)]
                    @ transport.conj().T
                )
                transported_hessians[index + (anchor_index,)] = (
                    transport
                    @ anchor_hessians[index + (anchor_index,)]
                    @ transport.conj().T
                )
        anchor_hamiltonians = 0.5 * (
            anchor_hamiltonians
            + anchor_hamiltonians.swapaxes(-1, -2).conj()
        )
        transported_gradients = 0.5 * (
            transported_gradients
            + transported_gradients.swapaxes(-1, -2).conj()
        )
        transported_hessians = 0.5 * (
            transported_hessians
            + transported_hessians.swapaxes(-1, -2).conj()
        )
        separable_hamiltonian = SeparableHamiltonian.quintic_hermite(
            qa_axis.x,
            qa_anchors,
            anchor_hamiltonians,
            transported_gradients,
            transported_hessians,
            extrapolation=qa_extrapolation,
        )

    overlaps = np.empty(
        (nqs, ntheta, nactive, nqs, ntheta, nactive),
        dtype=complex,
    )
    for bra in np.ndindex(nqs, ntheta):
        overlaps[bra + (slice(None),) + bra + (slice(None),)] = np.eye(
            nactive,
            dtype=complex,
        )
    flat = list(np.ndindex(nqs, ntheta))
    for bra_pos, bra in enumerate(flat):
        for ket in flat[bra_pos + 1:]:
            block = casci_overlap_active(
                points[bra + (center_anchor,)],
                points[ket + (center_anchor,)],
                state_ids,
                polar=polar_overlap,
            )
            overlaps[bra + (slice(None),) + ket + (slice(None),)] = block
            overlaps[ket + (slice(None),) + bra + (slice(None),)] = block.conj().T

    metadata = {
        "molecule": "SO2",
        "source": "live CASCI q_a derivative grid",
        "source_meta": scan.meta,
        "sampled_coordinates": ["qs", "theta"],
        "expanded_coordinates": ["qa"],
        "state_ids": list(state_ids),
        "qs_indices": list(qs_indices),
        "theta_indices": list(theta_indices),
        "theta_degrees": np.rad2deg(theta).tolist(),
        "qa_npts": int(qa_axis.npts),
        "qa_half_width": float(qa_axis.xmax),
        "qa_grid": qa_axis.x.tolist(),
        "qa_model": (
            "single-reference-quadratic"
            if qa_anchor_count == 1
            else f"{qa_anchor_count}-anchor-relaxed-pt-quintic"
        ),
        "qa_anchor_count": qa_anchor_count,
        "qa_anchor_indices": qa_anchor_indices.tolist(),
        "qa_anchor_values": qa_anchors.tolist(),
        "qa_extrapolation": qa_extrapolation,
        "energy_shift": energy_shift,
        "derivative_source": "CASCI.vibronic_couplings:basis_derivatives+RDM",
        "moving_basis": moving_basis,
        "overlap_form": "polar" if polar_overlap else "raw",
        "secondary_mode": "qa",
        "basis": basis,
        "ncas": int(ncas),
        "nelecas": int(nelecas),
        "active_gap_threshold": MIN_ACTIVE_GAP,
        "multiplicity": multiplicity,
        "workers": int(workers),
        "derivative_workers": int(derivative_workers),
        "derivative_backend": derivative_backend,
        "kinetic_model": kinetic_label,
        "kinetic_coordinates": ["qs", "theta", "qa"],
        "kinetic_exp_order": int(kinetic_exp_order),
        "kinetic_exp_scale": int(kinetic_exp_scale),
        "kinetic_svd_tol": float(kinetic_svd_tol),
    }
    data = CGLDRElectronicData(
        energies=energies,
        overlaps=overlaps,
        hamiltonian_gradients=gradients,
        hamiltonian_hessians=hessians,
        separable_hamiltonian=separable_hamiltonian,
        reactive_grids=(qs, theta),
        expanded_grids=(qa_axis.x,),
        metadata=metadata,
    )
    dynamics.set_electronic_data(data, tolerance=1.0e-6)
    return dynamics, data


def build_qs_cgldr_from_casci(
    scan: SO2LinkedScan,
    *,
    state_ids=None,
    qs_indices=None,
    qa_npts=None,
    theta_anchor_indices=None,
    qa_anchor_indices=None,
    max_rank=64,
    energy_reference="minimum",
    basis="sto-3g",
    ncas=DEFAULT_NCAS,
    nelecas=DEFAULT_NELECAS,
    multiplicity=1,
    scf_tol=1.0e-8,
    scf_max_cycle=80,
    workers=1,
    moving_basis="rhf-relaxed-pt",
    derivative_backend="native",
    derivative_workers=1,
    qa_half_width=None,
    kinetic_model="valence",
    kinetic_exp_order=10,
    kinetic_exp_scale=1,
    kinetic_svd_tol=0.0,
    polar_overlap=False,
    electronic_data=None,
):
    """Build SO2 CGLDR data with ``q_s`` sampled and ``(theta, q_a)`` expanded."""
    if not isinstance(workers, (int, np.integer)) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    if int(derivative_workers) <= 0:
        raise ValueError("derivative_workers must be positive")
    state_ids = parse_state_ids(state_ids, scan.solver.nstates)
    casci_nstates = max(state_ids) + 1
    qs, qs_indices = symmetric_stretch_nodes(scan, qs_indices)
    theta = np.asarray(scan.theta, dtype=float)
    qa_axis = qa_axis_from_scan(scan, qa_npts, half_width=qa_half_width)
    qa = np.asarray(qa_axis.x, dtype=float)
    theta_center_index = nearest_index(theta, np.deg2rad(REFERENCE_THETA_DEG))
    qa_center_index = nearest_index(qa, 0.0)

    if theta_anchor_indices is None:
        theta_anchor_indices = (
            theta_center_index - 1,
            theta_center_index,
            theta_center_index + 1,
        )
    if qa_anchor_indices is None:
        stride = max(1, min(3, qa_center_index, len(qa) - qa_center_index - 1))
        qa_anchor_indices = (
            qa_center_index - stride,
            qa_center_index,
            qa_center_index + stride,
        )

    def validate_anchors(indices, size, center_index, name):
        indices = np.asarray(
            parse_grid_indices(indices, size, name=name),
            dtype=int,
        )
        if indices.size != 3 or np.any(np.diff(indices) <= 0):
            raise ValueError(f"{name} must contain three increasing indices")
        if indices[1] != center_index:
            raise ValueError(f"{name} must bracket the central DVR node")
        return indices

    theta_anchor_indices = validate_anchors(
        theta_anchor_indices,
        len(theta),
        theta_center_index,
        "theta_anchor_indices",
    )
    qa_anchor_indices = validate_anchors(
        qa_anchor_indices,
        len(qa),
        qa_center_index,
        "qa_anchor_indices",
    )
    theta_anchors = theta[theta_anchor_indices]
    qa_anchors = qa[qa_anchor_indices]
    centers = (theta[theta_center_index], qa[qa_center_index])
    extrapolation = (
        "error"
        if theta_anchor_indices[0] == 0
        and theta_anchor_indices[-1] == len(theta) - 1
        else "quadratic",
        "error"
        if qa_anchor_indices[0] == 0
        and qa_anchor_indices[-1] == len(qa) - 1
        else "quadratic",
    )

    dvr = DVR.from_axes(
        (
            SineDVR(*infer_sine_domain(qs), len(qs)),
            LegendreDVR(*infer_legendre_domain(theta), len(theta)),
            qa_axis,
        ),
        names=("qs", "theta", "qa"),
    )
    partition = ElectronicPartition(
        sampled=("qs",),
        expanded=("theta", "qa"),
        center=centers,
    )
    kinetic_model = str(kinetic_model).lower().replace("_", "-")
    if kinetic_model == "valence":
        nuclear_kinetic_mpo = scan.solver.buildK_qsqa_mpo(
            dvr.axes,
            max_rank=None,
            symmetrize=True,
            svd_tol=kinetic_svd_tol,
        )
        kinetic_label = "transformed-valence"
    elif kinetic_model in {"product", "product-dvr", "dvr"}:
        nuclear_kinetic_mpo = None
        kinetic_label = "product-dvr"
    else:
        raise ValueError("kinetic_model must be 'valence' or 'product-dvr'")
    dynamics = CGLDR(
        dvr,
        partition,
        state_ids=state_ids,
        tt_options={"max_rank": max_rank},
        nuclear_kinetic_mpo=nuclear_kinetic_mpo,
        kinetic_exponential_options={
            "order": kinetic_exp_order,
            "scale": kinetic_exp_scale,
        },
    )
    if electronic_data is not None:
        dynamics.set_electronic_data(electronic_data, tolerance=1.0e-6)
        return dynamics, electronic_data

    anchor_coordinates = [
        (centers[0], centers[1]),
        (theta_anchors[0], centers[1]),
        (theta_anchors[2], centers[1]),
        (centers[0], qa_anchors[0]),
        (centers[0], qa_anchors[2]),
    ]
    nqs = len(qs)
    nactive = len(state_ids)
    nanchor_points = len(anchor_coordinates)
    points = np.empty((nqs, nanchor_points), dtype=object)
    anchor_energies = np.empty((nqs, nanchor_points, nactive), dtype=float)
    anchor_gradients = np.empty(
        (nqs, nanchor_points, nactive, nactive, 2),
        dtype=complex,
    )
    anchor_hessians = np.empty(
        (nqs, nanchor_points, nactive, nactive, 2, 2),
        dtype=complex,
    )
    tasks = [
        (
            i,
            anchor_index,
            float(qs_value),
            float(theta_value),
            float(qa_value),
            tuple(state_ids),
            basis,
            int(ncas),
            int(nelecas),
            int(casci_nstates),
            float(scf_tol),
            int(scf_max_cycle),
            multiplicity,
            moving_basis,
            derivative_backend,
            int(derivative_workers),
        )
        for i, qs_value in enumerate(qs)
        for anchor_index, (theta_value, qa_value) in enumerate(anchor_coordinates)
    ]
    total = len(tasks)

    def save_anchor(result):
        i, anchor_index, point, point_energies, first, second = result
        points[i, anchor_index] = point
        anchor_energies[i, anchor_index] = point_energies
        anchor_gradients[i, anchor_index] = first
        anchor_hessians[i, anchor_index] = second

    if workers == 1:
        for count, task in enumerate(tasks, start=1):
            save_anchor(_build_qs_casci_anchor(task))
            print(f"[CASCI q_s | theta,q_a anchors] {count}/{total}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, total)) as executor:
            futures = [
                executor.submit(_build_qs_casci_anchor, task)
                for task in tasks
            ]
            for count, future in enumerate(as_completed(futures), start=1):
                save_anchor(future.result())
                print(
                    f"[CASCI q_s | theta,q_a anchors] {count}/{total} "
                    f"({workers} workers)",
                    flush=True,
                )

    if energy_reference in {"minimum", "theta-center-minimum"}:
        energy_shift = float(np.nanmin(anchor_energies))
    elif energy_reference == "zero":
        energy_shift = 0.0
    else:
        raise ValueError(
            "energy_reference must be 'minimum', 'theta-center-minimum', or 'zero'"
        )
    anchor_energies -= energy_shift
    center_hamiltonian = np.empty((nqs, nactive, nactive), dtype=complex)
    axial_hamiltonians = (
        np.empty((nqs, 3, nactive, nactive), dtype=complex),
        np.empty((nqs, 3, nactive, nactive), dtype=complex),
    )
    axial_gradients = tuple(np.empty_like(values) for values in axial_hamiltonians)
    axial_hessians = tuple(np.empty_like(values) for values in axial_hamiltonians)
    center_gradients = np.empty((nqs, 2, nactive, nactive), dtype=complex)
    center_hessians = np.empty(
        (nqs, 2, 2, nactive, nactive),
        dtype=complex,
    )
    theta_records = (1, 0, 2)
    qa_records = (3, 0, 4)
    for i in range(nqs):
        reference = points[i, 0]
        transported_h = []
        transported_f = []
        transported_g = []
        for anchor_index in range(nanchor_points):
            transport = (
                np.eye(nactive, dtype=complex)
                if anchor_index == 0
                else casci_overlap_active(
                    reference,
                    points[i, anchor_index],
                    state_ids,
                    polar=True,
                )
            )
            transported_h.append(
                transport
                @ np.diag(anchor_energies[i, anchor_index])
                @ transport.conj().T
            )
            transported_f.append(np.stack([
                transport
                @ anchor_gradients[i, anchor_index, :, :, axis]
                @ transport.conj().T
                for axis in range(2)
            ]))
            transported_g.append(np.stack([
                np.stack([
                    transport
                    @ anchor_hessians[i, anchor_index, :, :, first, second]
                    @ transport.conj().T
                    for second in range(2)
                ])
                for first in range(2)
            ]))
        transported_h = np.asarray(transported_h)
        transported_f = np.asarray(transported_f)
        transported_g = np.asarray(transported_g)
        transported_h = 0.5 * (
            transported_h + transported_h.swapaxes(-1, -2).conj()
        )
        transported_f = 0.5 * (
            transported_f + transported_f.swapaxes(-1, -2).conj()
        )
        transported_g = 0.5 * (
            transported_g + transported_g.swapaxes(-1, -2).conj()
        )
        transported_g = 0.5 * (
            transported_g + transported_g.swapaxes(1, 2)
        )
        center_hamiltonian[i] = transported_h[0]
        center_gradients[i] = transported_f[0]
        center_hessians[i] = transported_g[0]
        for axis, records in enumerate((theta_records, qa_records)):
            axial_hamiltonians[axis][i] = transported_h[list(records)]
            axial_gradients[axis][i] = transported_f[list(records), axis]
            axial_hessians[axis][i] = transported_g[list(records), axis, axis]

    separable_hamiltonian = SeparableHamiltonian.axial_quintic_hermite(
        (theta, qa),
        (theta_anchors, qa_anchors),
        axial_hamiltonians,
        axial_gradients,
        axial_hessians,
        center_hamiltonian=center_hamiltonian,
        centers=centers,
        mixed_hessians=center_hessians,
        extrapolation=extrapolation,
    )
    overlaps = np.empty(
        (nqs, nactive, nqs, nactive),
        dtype=complex,
    )
    for bra in range(nqs):
        overlaps[bra, :, bra, :] = np.eye(nactive, dtype=complex)
        for ket in range(bra + 1, nqs):
            block = casci_overlap_active(
                points[bra, 0],
                points[ket, 0],
                state_ids,
                polar=polar_overlap,
            )
            overlaps[bra, :, ket, :] = block
            overlaps[ket, :, bra, :] = block.conj().T

    metadata = {
        "molecule": "SO2",
        "source": "live CASCI axial theta/q_a derivative grid",
        "source_meta": scan.meta,
        "sampled_coordinates": ["qs"],
        "expanded_coordinates": ["theta", "qa"],
        "state_ids": list(state_ids),
        "qs_indices": list(qs_indices),
        "theta_grid_degrees": np.rad2deg(theta).tolist(),
        "qa_grid": qa.tolist(),
        "theta_anchor_indices": theta_anchor_indices.tolist(),
        "theta_anchor_degrees": np.rad2deg(theta_anchors).tolist(),
        "qa_anchor_indices": qa_anchor_indices.tolist(),
        "qa_anchor_values": qa_anchors.tolist(),
        "secondary_model": "axial-three-anchor-relaxed-pt-quintic",
        "anchor_geometry_count_per_qs": nanchor_points,
        "mixed_hessian": "analytical center G_theta_qa",
        "extrapolation": list(extrapolation),
        "energy_shift": energy_shift,
        "derivative_source": "CASCI.vibronic_couplings:basis_derivatives+RDM",
        "moving_basis": moving_basis,
        "anchor_transport": "polar",
        "overlap_form": "polar" if polar_overlap else "raw",
        "basis": basis,
        "ncas": int(ncas),
        "nelecas": int(nelecas),
        "active_gap_threshold": MIN_ACTIVE_GAP,
        "multiplicity": multiplicity,
        "workers": workers,
        "derivative_workers": int(derivative_workers),
        "derivative_backend": derivative_backend,
        "kinetic_model": kinetic_label,
        "kinetic_coordinates": ["qs", "theta", "qa"],
        "kinetic_exp_order": int(kinetic_exp_order),
        "kinetic_exp_scale": int(kinetic_exp_scale),
        "kinetic_svd_tol": float(kinetic_svd_tol),
    }
    data = CGLDRElectronicData(
        energies=anchor_energies[:, 0],
        overlaps=overlaps,
        hamiltonian_gradients=center_gradients,
        hamiltonian_hessians=center_hessians,
        separable_hamiltonian=separable_hamiltonian,
        reactive_grids=(qs,),
        expanded_grids=(theta, qa),
        metadata=metadata,
    )
    dynamics.set_electronic_data(data, tolerance=1.0e-6)
    return dynamics, data


def build_cgldr_from_scan(
    scan: SO2LinkedScan,
    *,
    theta_center_deg=119.5,
    state_ids=None,
    max_rank=64,
    energy_reference="minimum",
    theta_model="quadratic",
    theta_anchor_count=3,
    kinetic_model="valence",
    kinetic_exp_order=10,
    kinetic_exp_scale=1,
):
    theta_target = np.deg2rad(theta_center_deg)
    theta_index = nearest_index(scan.theta, theta_target)
    theta_center = float(scan.theta[theta_index])
    state_ids = parse_state_ids(state_ids, scan.solver.nstates)
    state_indices = np.asarray(state_ids, dtype=int)

    axes = (
        SineDVR(*infer_sine_domain(scan.r1), scan.r1.size),
        SineDVR(*infer_sine_domain(scan.r2), scan.r2.size),
        LegendreDVR(*infer_legendre_domain(scan.theta), scan.theta.size),
    )
    coordinate_names = tuple(scan.solver.coordinate_labels)
    dvr = DVR.from_axes(axes, names=coordinate_names)
    partition = ElectronicPartition(
        sampled=coordinate_names[:2],
        expanded=(coordinate_names[2],),
        center=(theta_center,),
    )
    kinetic_model = str(kinetic_model).lower().replace("_", "-")
    if kinetic_model == "valence":
        nuclear_kinetic_mpo = scan.solver.buildK_product_mpo(
            max_rank=max_rank,
            symmetrize=True,
        )
    elif kinetic_model in {"product", "product-dvr", "dvr"}:
        kinetic_model = "product-dvr"
        nuclear_kinetic_mpo = None
    else:
        raise ValueError("kinetic_model must be 'valence' or 'product-dvr'")

    dynamics = CGLDR(
        dvr,
        partition,
        state_ids=state_ids,
        tt_options={"max_rank": max_rank},
        nuclear_kinetic_mpo=nuclear_kinetic_mpo,
        kinetic_exponential_options={
            "order": kinetic_exp_order,
            "scale": kinetic_exp_scale,
        },
    )

    if energy_reference == "minimum":
        energy_shift = float(np.nanmin(scan.apes))
    elif energy_reference == "theta-center-minimum":
        energy_shift = float(np.nanmin(scan.apes[:, :, theta_index, :]))
    elif energy_reference == "zero":
        energy_shift = 0.0
    else:
        raise ValueError(
            "energy_reference must be 'minimum', 'theta-center-minimum', or 'zero'"
        )

    metadata = {
        "molecule": "SO2",
        "source": (
            f"{scan.meta.get('electronic_method', 'electronic')} "
            "linked-LDR cache"
        ),
        "source_meta": scan.meta,
        "sampled_coordinates": list(coordinate_names[:2]),
        "expanded_coordinates": [coordinate_names[2]],
        "state_ids": list(state_ids),
        "theta_center_index": theta_index,
        "theta_center_rad": theta_center,
        "theta_center_deg": float(np.rad2deg(theta_center)),
        "energy_shift": energy_shift,
        "hamiltonian_frame": "theta-center linked polar-unitary transport",
        "theta_transport": "polar-unitary",
        "theta_model": theta_model,
        "kinetic_model": kinetic_model,
        "kinetic_exp_order": int(kinetic_exp_order),
        "kinetic_exp_scale": int(kinetic_exp_scale),
    }
    theta_model = str(theta_model).lower().replace("_", "-")
    if theta_model in {"single-reference", "single", "center", "quadratic"}:
        theta_model_label = (
            "quadratic" if theta_model == "quadratic" else "single-reference"
        )
        energies, gradients, hessians, anchors = theta_quadratic_derivatives(
            scan,
            theta_index,
            anchor_count=theta_anchor_count,
            energy_shift=energy_shift,
            state_indices=state_indices,
        )
        metadata.update({
            "theta_model": theta_model_label,
            "theta_reference_indices": [int(theta_index)],
            "theta_reference_degrees": [float(np.rad2deg(scan.theta[theta_index]))],
            "theta_anchor_indices": anchors.tolist(),
            "theta_anchor_degrees": np.rad2deg(scan.theta[anchors]).tolist(),
            "theta_anchor_count": int(len(anchors)),
            "diagnostic": (
                "single theta-center electronic reference with local "
                "quadratic theta PES"
            ),
        })
        data = CGLDRElectronicData(
            energies=energies,
            overlaps=sampled_overlap_matrix(
                scan,
                theta_index,
                state_indices=state_indices,
            ),
            hamiltonian_gradients=gradients,
            hamiltonian_hessians=hessians,
            reactive_grids=(scan.r1, scan.r2),
            expanded_grids=(scan.theta,),
            metadata=metadata,
        )
    elif theta_model == "collocation":
        metadata.update({
            "theta_anchor_indices": list(range(scan.theta.size)),
            "theta_anchor_degrees": np.rad2deg(scan.theta).tolist(),
            "theta_anchor_count": int(scan.theta.size),
            "diagnostic": "full-theta collocation, not coarse-grained",
        })
        data = CGLDRElectronicData(
            energies=scan.apes[:, :, theta_index, state_indices] - energy_shift,
            overlaps=sampled_overlap_matrix(
                scan,
                theta_index,
                state_indices=state_indices,
            ),
            separable_hamiltonian=theta_center_hamiltonian(
                scan,
                theta_index,
                energy_shift=energy_shift,
                state_indices=state_indices,
            ),
            reactive_grids=(scan.r1, scan.r2),
            expanded_grids=(scan.theta,),
            metadata=metadata,
        )
    else:
        raise ValueError(
            "theta_model must be 'single-reference', 'quadratic', or "
            "'collocation'"
        )
    dynamics.set_electronic_data(data, tolerance=1.0e-6)
    return dynamics, data


def initial_state(dynamics, *, state, center, width):
    weights = [
        np.asarray(
            getattr(axis, "w", np.full(len(axis.x), float(axis.dx))),
            dtype=float,
        )
        for axis in dynamics.axes
    ]
    return gaussian_state(
        dynamics.x,
        state=state,
        nstates=dynamics.nstates,
        center=center,
        width=width,
        weights=weights,
    ).normalize()


def parse_triplet(text, *, degree_indices=()):
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("Expected exactly three comma-separated values.")
    for index in degree_indices:
        values[int(index)] = np.deg2rad(values[int(index)])
    return tuple(values)


def secondary_hessian_eigenvalue_range(data, mask=None):
    if data.hamiltonian_hessians is None:
        return None
    theta_hessian = data.hamiltonian_hessians[..., 0, 0, :, :]
    if mask is not None:
        theta_hessian = theta_hessian[np.asarray(mask, dtype=bool)]
    eigenvalues = np.linalg.eigvalsh(
        theta_hessian.reshape(
            -1,
            theta_hessian.shape[-2],
            theta_hessian.shape[-1],
        )
    )
    return float(eigenvalues.min()), float(eigenvalues.max()), int(eigenvalues.size)


def sampled_product_gaussian_support(grids, center, width, *, threshold=1.0e-3):
    center = tuple(float(value) for value in center)
    width = tuple(float(value) for value in width)
    weights = 1.0
    shape = tuple(len(grid) for grid in grids)
    for axis, grid in enumerate(grids):
        reshape = [1] * len(grids)
        reshape[axis] = len(grid)
        weights = weights * np.exp(
            -2.0 * ((np.asarray(grid).reshape(reshape) - center[axis])
                    / width[axis]) ** 2
        )
    weights = np.array(np.broadcast_to(weights, shape), copy=True)
    weights /= np.sum(weights)
    return weights > float(threshold)


def default_initial_packet_spec(secondary_mode, coordinates="valence"):
    """Return default packet strings and angular coordinate positions."""
    secondary_mode = str(secondary_mode).lower()
    coordinates = str(coordinates).lower().replace("_", "-")
    if secondary_mode == "theta" and coordinates == "qs-qa-theta":
        return (
            f"{SQRT2 * REFERENCE_BOND},0.0,{REFERENCE_THETA_DEG}",
            f"{REFERENCE_BOND_WIDTH},{REFERENCE_BOND_WIDTH},"
            f"{REFERENCE_THETA_WIDTH_DEG}",
            (2,),
        )
    if secondary_mode in {"qa", "theta-qa"}:
        # The r1/r2 -> q_s/q_a transform is orthogonal, so the Gaussian width
        # remains 0.16 bohr on both q_s and q_a.
        return (
            f"{SQRT2 * REFERENCE_BOND},{REFERENCE_THETA_DEG},0.0",
            f"{REFERENCE_BOND_WIDTH},{REFERENCE_THETA_WIDTH_DEG},"
            f"{REFERENCE_BOND_WIDTH}",
            (1,),
        )
    if secondary_mode == "theta":
        return (
            f"{REFERENCE_BOND},{REFERENCE_BOND},{REFERENCE_THETA_DEG}",
            f"{REFERENCE_BOND_WIDTH},{REFERENCE_BOND_WIDTH},"
            f"{REFERENCE_THETA_WIDTH_DEG}",
            (2,),
        )
    raise ValueError("secondary_mode must be 'theta', 'qa', or 'theta-qa'")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/so2_cgldr"))
    parser.add_argument("--max-rank", type=int, default=64)
    parser.add_argument(
        "--secondary-mode",
        choices=("theta", "qa", "theta-qa"),
        default="theta",
        help=(
            "Expanded coordinates: theta, q_a, or both with only q_s "
            "sampled."
        ),
    )
    parser.add_argument(
        "--state-ids",
        default="all",
        help=(
            "Comma-separated scan-root ids to keep, or 'all'. Use "
            "--initial-state to choose the launched electronic state."
        ),
    )
    parser.add_argument("--theta-center-deg", type=float, default=REFERENCE_THETA_DEG)
    parser.add_argument(
        "--energy-reference",
        choices=("minimum", "theta-center-minimum", "zero"),
        default="minimum",
    )
    parser.add_argument(
        "--theta-model",
        choices=("single-reference", "quadratic", "collocation"),
        default="single-reference",
    )
    parser.add_argument("--theta-anchor-count", type=int, default=3)
    parser.add_argument(
        "--theta-anchors",
        default=None,
        help="Three comma-separated theta DVR indices for theta-qa mode.",
    )
    parser.add_argument(
        "--theta-derivatives",
        choices=("linked-fit", "analytic"),
        default="linked-fit",
        help=(
            "Use a polar-transported CASCI theta stencil (validated default) "
            "or experimental analytic truncated-state F/G tensors."
        ),
    )
    parser.add_argument(
        "--kinetic-model",
        choices=("valence", "product-dvr"),
        default="valence",
        help="Use the triatomic valence vibrational KEO MPO or old product-DVR kinetic.",
    )
    parser.add_argument("--kinetic-exp-order", type=int, default=10)
    parser.add_argument("--kinetic-exp-scale", type=int, default=1)
    parser.add_argument(
        "--polar-overlap",
        action="store_true",
        help="Discard overlap singular values and retain only polar transport.",
    )
    parser.add_argument(
        "--kinetic-svd-tol",
        type=float,
        default=0.0,
        help="Relative SVD cutoff for transformed q_s/theta/q_a KEO factors.",
    )
    parser.add_argument("--qs-indices", default="all")
    parser.add_argument("--qa-indices", default="all")
    parser.add_argument("--theta-indices", default="all")
    parser.add_argument("--qa-npts", type=int, default=None)
    parser.add_argument(
        "--qa-anchor-count",
        type=int,
        default=1,
        help=(
            "Use one relaxed-PT q_a anchor with a quadratic expansion, or "
            "an odd center-containing stencil with piecewise-quintic interpolation."
        ),
    )
    parser.add_argument(
        "--qa-anchors",
        default=None,
        help=(
            "Comma-separated q_a DVR anchor indices. Examples include "
            "'1,4,7' and the center-dense five-anchor stencil '1,3,4,5,7'."
        ),
    )
    parser.add_argument(
        "--qa-half-width",
        type=float,
        default=None,
        help="Sine-DVR q_a box half-width in bohr; defaults to the full r-domain transform.",
    )
    parser.add_argument("--casci-basis", default="sto-3g")
    parser.add_argument("--ncas", type=int, default=DEFAULT_NCAS)
    parser.add_argument("--nelecas", type=int, default=DEFAULT_NELECAS)
    parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="CASCI spin multiplicity; the SO2 photodynamics default is singlet.",
    )
    parser.add_argument(
        "--moving-basis",
        choices=("symmetric", "rhf-relaxed", "rhf-relaxed-pt"),
        default="rhf-relaxed-pt",
        help="Electronic-frame transport used for analytical F/G.",
    )
    parser.add_argument(
        "--derivative-backend",
        choices=("native", "pyscf", "auto", "python"),
        default="native",
    )
    parser.add_argument(
        "--derivative-workers",
        type=int,
        default=1,
        help="Native ERI worker threads inside each independent CASCI anchor.",
    )
    parser.add_argument("--scf-tol", type=float, default=1.0e-8)
    parser.add_argument("--scf-max-cycle", type=int, default=80)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent CASCI anchor worker processes.",
    )
    parser.add_argument("--no-path-average", action="store_true")
    parser.add_argument(
        "--initial-state",
        type=int,
        default=2,
        help="Initial scan-root id; it must be included in --state-ids.",
    )
    parser.add_argument("--center", default=None)
    parser.add_argument("--width", default=None)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--output-every", type=int, default=5)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--reuse-electronic-data",
        action="store_true",
        help="Reuse so2_cgldr_electronic_data.npz from --outdir.",
    )
    parser.add_argument(
        "--reuse-anchor-cache",
        action="store_true",
        help="Resume completed q_a CASCI anchors from the output directory.",
    )
    parser.add_argument(
        "--electronic-data",
        type=Path,
        default=None,
        help="Load an explicit CGLDR electronic-data NPZ instead of rebuilding.",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    args.outdir.mkdir(parents=True, exist_ok=True)
    scan = load_so2_linked_scan(
        args.scan_dir,
        path_average=not args.no_path_average,
    )
    state_ids = parse_state_ids(args.state_ids, scan.solver.nstates)
    if args.secondary_mode == "theta-qa":
        electronic_data = None
        electronic_data_path = args.outdir / "so2_cgldr_electronic_data.npz"
        if args.electronic_data is not None:
            electronic_data = CGLDRElectronicData.from_npz(args.electronic_data)
        elif args.reuse_electronic_data:
            electronic_data = CGLDRElectronicData.from_npz(electronic_data_path)
        dynamics, data = build_qs_cgldr_from_casci(
            scan,
            state_ids=state_ids,
            qs_indices=args.qs_indices,
            qa_npts=args.qa_npts,
            theta_anchor_indices=args.theta_anchors,
            qa_anchor_indices=args.qa_anchors,
            qa_half_width=args.qa_half_width,
            max_rank=args.max_rank,
            energy_reference=args.energy_reference,
            basis=args.casci_basis,
            ncas=args.ncas,
            nelecas=args.nelecas,
            multiplicity=args.multiplicity,
            scf_tol=args.scf_tol,
            scf_max_cycle=args.scf_max_cycle,
            workers=args.workers,
            moving_basis=args.moving_basis,
            derivative_backend=args.derivative_backend,
            derivative_workers=args.derivative_workers,
            kinetic_model=args.kinetic_model,
            kinetic_exp_order=args.kinetic_exp_order,
            kinetic_exp_scale=args.kinetic_exp_scale,
            kinetic_svd_tol=args.kinetic_svd_tol,
            polar_overlap=args.polar_overlap,
            electronic_data=electronic_data,
        )
        sampled_grids = data.reactive_grids
        secondary_label = "theta"
    elif args.secondary_mode == "qa":
        electronic_data = None
        electronic_data_path = args.outdir / "so2_cgldr_electronic_data.npz"
        if args.electronic_data is not None:
            electronic_data = CGLDRElectronicData.from_npz(args.electronic_data)
        elif args.reuse_electronic_data:
            electronic_data = CGLDRElectronicData.from_npz(electronic_data_path)
        dynamics, data = build_qa_cgldr_from_casci(
            scan,
            state_ids=state_ids,
            qs_indices=args.qs_indices,
            theta_indices=args.theta_indices,
            qa_npts=args.qa_npts,
            qa_anchor_count=args.qa_anchor_count,
            qa_anchor_indices=args.qa_anchors,
            qa_half_width=args.qa_half_width,
            max_rank=args.max_rank,
            energy_reference=args.energy_reference,
            basis=args.casci_basis,
            ncas=args.ncas,
            nelecas=args.nelecas,
            multiplicity=args.multiplicity,
            scf_tol=args.scf_tol,
            scf_max_cycle=args.scf_max_cycle,
            workers=args.workers,
            moving_basis=args.moving_basis,
            derivative_backend=args.derivative_backend,
            derivative_workers=args.derivative_workers,
            kinetic_model=args.kinetic_model,
            kinetic_exp_order=args.kinetic_exp_order,
            kinetic_exp_scale=args.kinetic_exp_scale,
            kinetic_svd_tol=args.kinetic_svd_tol,
            polar_overlap=args.polar_overlap,
            electronic_data=electronic_data,
            anchor_cache_dir=args.outdir / "qa_anchor_cache",
            reuse_anchor_cache=args.reuse_anchor_cache,
        )
        sampled_grids = data.reactive_grids
        secondary_label = "q_a"
    else:
        if (
            scan.solver.coordinates == "qs-qa-theta"
            and args.theta_derivatives == "analytic"
        ):
            electronic_data = None
            electronic_data_path = (
                args.outdir / "so2_cgldr_electronic_data.npz"
            )
            if args.electronic_data is not None:
                electronic_data = CGLDRElectronicData.from_npz(
                    args.electronic_data
                )
            elif args.reuse_electronic_data:
                electronic_data = CGLDRElectronicData.from_npz(
                    electronic_data_path
                )
            dynamics, data = build_theta_cgldr_from_casci(
                scan,
                theta_center_deg=args.theta_center_deg,
                state_ids=state_ids,
                qs_indices=args.qs_indices,
                qa_indices=args.qa_indices,
                max_rank=args.max_rank,
                energy_reference=args.energy_reference,
                basis=args.casci_basis,
                ncas=args.ncas,
                nelecas=args.nelecas,
                multiplicity=args.multiplicity,
                scf_tol=args.scf_tol,
                scf_max_cycle=args.scf_max_cycle,
                workers=args.workers,
                moving_basis=args.moving_basis,
                derivative_backend=args.derivative_backend,
                derivative_workers=args.derivative_workers,
                kinetic_model=args.kinetic_model,
                kinetic_exp_order=args.kinetic_exp_order,
                kinetic_exp_scale=args.kinetic_exp_scale,
                polar_overlap=args.polar_overlap,
                electronic_data=electronic_data,
            )
            sampled_grids = data.reactive_grids
        else:
            dynamics, data = build_cgldr_from_scan(
                scan,
                theta_center_deg=args.theta_center_deg,
                state_ids=state_ids,
                max_rank=args.max_rank,
                energy_reference=args.energy_reference,
                theta_model=args.theta_model,
                theta_anchor_count=args.theta_anchor_count,
                kinetic_model=args.kinetic_model,
                kinetic_exp_order=args.kinetic_exp_order,
                kinetic_exp_scale=args.kinetic_exp_scale,
            )
            sampled_grids = (scan.r1, scan.r2)
        secondary_label = "theta"
    default_center, default_width, angle_indices = default_initial_packet_spec(
        args.secondary_mode,
        coordinates=scan.solver.coordinates,
    )
    data.to_npz(args.outdir / "so2_cgldr_electronic_data.npz")

    dt = args.dt_fs / au2fs
    center = parse_triplet(args.center or default_center, degree_indices=angle_indices)
    width = parse_triplet(args.width or default_width, degree_indices=angle_indices)
    summary = (
        "[cgldr] built SO2 CGLDR data: "
        f"grid={dynamics.npts}, nstates={dynamics.nstates}, "
        f"state_ids={data.metadata['state_ids']}, "
        f"secondary_mode={args.secondary_mode}, "
        f"kinetic_model={data.metadata['kinetic_model']}"
    )
    if args.secondary_mode == "theta":
        summary += (
            f", theta_center={np.rad2deg(dynamics.partition.center[0]):.4f} deg, "
            f"theta_model={data.metadata['theta_model']}, "
            f"theta_anchors={data.metadata['theta_anchor_count']}"
        )
    print(summary)
    if args.secondary_mode == "theta":
        print(
            "[cgldr] theta anchor degrees = "
            f"{np.array2string(np.asarray(data.metadata['theta_anchor_degrees']), precision=4)}"
        )
    elif args.secondary_mode == "qa":
        print(
            "[cgldr] sampled theta degrees = "
            f"{np.array2string(np.asarray(data.metadata['theta_degrees']), precision=4)}"
        )
        print(
            "[cgldr] q_a anchors = "
            f"{np.array2string(np.asarray(data.metadata['qa_anchor_values']), precision=6)} "
            f"({data.metadata['qa_model']})"
        )
    else:
        print(
            "[cgldr] theta anchors / deg = "
            f"{np.array2string(np.asarray(data.metadata['theta_anchor_degrees']), precision=4)}"
        )
        print(
            "[cgldr] q_a anchors / bohr = "
            f"{np.array2string(np.asarray(data.metadata['qa_anchor_values']), precision=6)}"
        )
    global_theta_range = secondary_hessian_eigenvalue_range(data)
    if global_theta_range is not None:
        print(
            f"[cgldr] {secondary_label} Hessian eigenvalue range, "
            "full sampled grid = "
            f"{global_theta_range[0]:.6e} .. {global_theta_range[1]:.6e}"
        )
        support = sampled_product_gaussian_support(
            sampled_grids,
            center[:2],
            width[:2],
        )
        support_theta_range = secondary_hessian_eigenvalue_range(data, support)
        print(
            f"[cgldr] {secondary_label} Hessian eigenvalue range, packet support "
            f"(w>1e-3, {support.sum()} stretch points) = "
            f"{support_theta_range[0]:.6e} .. {support_theta_range[1]:.6e}"
        )
    print(f"[cgldr] electronic data: {args.outdir / 'so2_cgldr_electronic_data.npz'}")

    if args.build_only or args.steps == 0:
        print(f"[timing] build-only completed in {time.perf_counter() - start:.2f} s")
        return

    if args.initial_state not in state_ids:
        raise ValueError("--initial-state must be included in --state-ids")
    active_initial_state = state_ids.index(args.initial_state)
    packet = initial_state(
        dynamics,
        state=active_initial_state,
        center=center,
        width=width,
    )
    dynamics.output_folder = str(args.outdir)
    dynamics.run(
        packet,
        time_step=dt,
        steps=args.steps,
        output_every=args.output_every,
        save_data=False,
    )
    populations = dynamics.compute_populations(plot=False, femtoseconds=True)
    coordinates = dynamics.compute_coordinate_expectations(femtoseconds=True)
    np.savez(
        args.outdir / "so2_cgldr_dynamics.npz",
        populations=populations,
        coordinate_expectations=coordinates["means"],
        coordinate_variances=coordinates["variances"],
        coordinate_names=np.asarray(coordinates["names"]),
        times_fs=dynamics.times,
        metadata_json=np.array(json.dumps(data.metadata, sort_keys=True)),
    )
    print(f"[cgldr] dynamics: {args.outdir / 'so2_cgldr_dynamics.npz'}")
    print(f"[cgldr] final populations: {np.array2string(populations[-1], precision=6)}")
    print(
        "[cgldr] final coordinate expectations: "
        f"{dict(zip(coordinates['names'], coordinates['means'][-1]))}"
    )
    print(f"[timing] completed in {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    main()
