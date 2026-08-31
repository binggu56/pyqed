"""Liquid-phase helpers for LDR/LDRFG quantum dynamics.

The routines here intentionally keep the liquid model outside the electronic
solver.  Cheap analytic tests can reduce an MD trajectory to one collective
coordinate, while the embedded methanol workflow also builds a full-coordinate
single frozen-Gaussian path over selected intramolecular modes plus all solvent
Cartesian coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyqed.units import atomic_mass, amu2au, au2angstrom, au2fs

from .ldrfg import LDRFG


TIP3P_CHARGES = np.array([-0.834, 0.417, 0.417], dtype=float)


@dataclass(frozen=True)
class XYZFrame:
    """One XYZ trajectory frame in atomic units."""

    symbols: tuple[str, ...]
    positions: np.ndarray
    time: float | None = None


@dataclass(frozen=True)
class MethanolFGCoordinateFrame:
    """One full-coordinate frozen-Gaussian center for methanol in liquid."""

    labels: tuple[str, ...]
    groups: tuple[str, ...]
    centers: np.ndarray
    momenta: np.ndarray
    masses: np.ndarray
    widths: np.ndarray
    time: float | None = None
    source_frame: int | None = None


@dataclass(frozen=True)
class EmbeddedLDRFGTDVPModel:
    """Path-linearized embedded LDRFG model for one moving liquid FG."""

    solver: LDRFG
    reference_q: np.ndarray
    reference_p: np.ndarray
    reference_index: int
    labels: tuple[str, ...]
    groups: tuple[str, ...]
    force_model: str
    electronic_force_source: str
    classical_force_source: str
    electronic_gradient_rank: int
    electronic_gradient_residual: float


@dataclass(frozen=True)
class SolventEmbeddedLDRSnapshot:
    """Solvent-conditioned LDR electronic data for one liquid frame."""

    bond_grid: np.ndarray
    apes: np.ndarray
    overlap: np.ndarray
    point_charge_coords: np.ndarray
    point_charge_charges: np.ndarray
    solvent_coordinate: float | None = None
    electronic_objects: tuple[object, ...] | None = None


@dataclass(frozen=True)
class SolventEmbeddedLDRTrajectory:
    """A sequence of solvent-conditioned LDR electronic snapshots."""

    snapshots: tuple[SolventEmbeddedLDRSnapshot, ...]
    times: np.ndarray

    @property
    def bond_grid(self):
        return self.snapshots[0].bond_grid


@dataclass
class LiquidAvoidedCrossingLDRModel:
    """One LDR coordinate coupled to one solvent collective coordinate.

    The local two-state Hamiltonian is the avoided-crossing model

    ``[[z, delta], [delta, -z]]``, ``z = a_x*x + a_q*q``,

    plus harmonic scalar confinement.  The adiabatic eigenvectors define the
    LDR electronic overlaps and the frozen-coordinate Berry connection used by
    :class:`pyqed.namd.LDRFG`.
    """

    x_grid: np.ndarray
    kinetic_x: np.ndarray
    mass_y: float = 1.0
    a_x: float = 1.0
    a_q: float = 0.8
    delta: float = 0.2
    k_x: float = 0.02
    k_q: float = 0.03
    gamma_y: float = 1.0

    def __post_init__(self):
        self.x_grid = np.asarray(self.x_grid, dtype=float)
        self.kinetic_x = np.asarray(self.kinetic_x, dtype=complex)
        if self.x_grid.ndim != 1:
            raise ValueError("x_grid must be one-dimensional.")
        if self.kinetic_x.shape != (self.x_grid.size, self.x_grid.size):
            raise ValueError("kinetic_x shape must be (len(x_grid), len(x_grid)).")

    def z(self, q):
        q0 = float(np.asarray(q, dtype=float)[0])
        return self.a_x * self.x_grid + self.a_q * q0

    def theta(self, q):
        return 0.5 * np.arctan2(self.delta, self.z(q))

    def dtheta_dq(self, q):
        z = self.z(q)
        return -0.5 * self.delta * self.a_q / (z * z + self.delta * self.delta)

    def electronic_vectors(self, q):
        theta = self.theta(q)
        c = np.cos(theta)
        s = np.sin(theta)
        vecs = np.empty((self.x_grid.size, 2, 2), dtype=float)
        vecs[:, :, 0] = np.stack((-s, c), axis=-1)
        vecs[:, :, 1] = np.stack((c, s), axis=-1)
        return vecs

    def electronic_vector_gradients(self, q):
        theta = self.theta(q)
        dtheta = self.dtheta_dq(q)
        c = np.cos(theta)
        s = np.sin(theta)
        grads = np.empty((self.x_grid.size, 2, 2), dtype=float)
        grads[:, :, 0] = np.stack((-c * dtheta, -s * dtheta), axis=-1)
        grads[:, :, 1] = np.stack((-s * dtheta, c * dtheta), axis=-1)
        return grads

    def energies(self, q):
        q0 = float(np.asarray(q, dtype=float)[0])
        z = self.z(q)
        rho = np.sqrt(z * z + self.delta * self.delta)
        scalar = 0.5 * self.k_x * self.x_grid**2 + 0.5 * self.k_q * q0**2
        return np.stack((scalar - rho, scalar + rho), axis=-1)

    def grad_energies(self, q):
        q0 = float(np.asarray(q, dtype=float)[0])
        z = self.z(q)
        rho = np.sqrt(z * z + self.delta * self.delta)
        drho = self.a_q * z / rho
        dscalar = self.k_q * q0
        return np.stack((dscalar - drho, dscalar + drho), axis=-1)[None, :, :]

    def overlap(self, q):
        vecs = self.electronic_vectors(q)
        return np.einsum("mdb,nda->mbna", vecs, vecs)

    def grad_overlap(self, q):
        vecs = self.electronic_vectors(q)
        dvecs = self.electronic_vector_gradients(q)
        grad = np.einsum("mdb,nda->mbna", dvecs, vecs)
        grad += np.einsum("mdb,nda->mbna", vecs, dvecs)
        return grad[None, :, :, :, :]

    def berry(self, q):
        vecs = self.electronic_vectors(q)
        dvecs = self.electronic_vector_gradients(q)
        local = np.einsum("ndb,nda->nba", vecs, dvecs)
        berry = np.zeros((1, self.x_grid.size, 2, self.x_grid.size, 2), dtype=float)
        for n in range(self.x_grid.size):
            berry[0, n, :, n, :] = local[n]
        return berry

    def solver(self, *, include_berry=True):
        return LDRFG(
            self.kinetic_x,
            masses_y=[self.mass_y],
            energies=self.energies,
            overlap=self.overlap,
            grad_energies=self.grad_energies,
            grad_overlap=self.grad_overlap,
            berry=self.berry if include_berry else None,
            gamma=np.array([[self.gamma_y]], dtype=float),
        )


@dataclass
class PhaseGaugedLiquidLDRModel:
    """q-dependent adiabatic phase transform of a liquid LDR model."""

    base_model: LiquidAvoidedCrossingLDRModel
    phase_offsets: np.ndarray
    phase_slopes: np.ndarray

    def __post_init__(self):
        self.phase_offsets = np.asarray(self.phase_offsets, dtype=float)
        self.phase_slopes = np.asarray(self.phase_slopes, dtype=float)
        nstates = self.base_model.energies([0.0]).shape[1]
        expected = (self.x_grid.size, nstates)
        if self.phase_offsets.shape != expected:
            raise ValueError(f"phase_offsets shape {self.phase_offsets.shape} != {expected}.")
        if self.phase_slopes.shape != expected:
            raise ValueError(f"phase_slopes shape {self.phase_slopes.shape} != {expected}.")

    @property
    def x_grid(self):
        return self.base_model.x_grid

    @property
    def kinetic_x(self):
        return self.base_model.kinetic_x

    @property
    def mass_y(self):
        return self.base_model.mass_y

    @property
    def gamma_y(self):
        return self.base_model.gamma_y

    def phase_angles(self, q):
        q0 = float(np.asarray(q, dtype=float)[0])
        return self.phase_offsets + self.phase_slopes * q0

    def phase(self, q):
        return np.exp(1j * self.phase_angles(q))

    def phase_gradient(self, q):
        return 1j * self.phase_slopes * self.phase(q)

    def electronic_vectors(self, q):
        return self.base_model.electronic_vectors(q).astype(complex) * self.phase(q)[:, None, :]

    def electronic_vector_gradients(self, q):
        phase = self.phase(q)
        return (
            self.base_model.electronic_vector_gradients(q).astype(complex) * phase[:, None, :]
            + self.base_model.electronic_vectors(q).astype(complex) * self.phase_gradient(q)[:, None, :]
        )

    def energies(self, q):
        return self.base_model.energies(q)

    def grad_energies(self, q):
        return self.base_model.grad_energies(q)

    def overlap(self, q):
        vecs = self.electronic_vectors(q)
        return np.einsum("mdb,nda->mbna", np.conjugate(vecs), vecs)

    def grad_overlap(self, q):
        vecs = self.electronic_vectors(q)
        dvecs = self.electronic_vector_gradients(q)
        grad = np.einsum("mdb,nda->mbna", np.conjugate(dvecs), vecs)
        grad += np.einsum("mdb,nda->mbna", np.conjugate(vecs), dvecs)
        return grad[None, :, :, :, :]

    def berry(self, q):
        vecs = self.electronic_vectors(q)
        dvecs = self.electronic_vector_gradients(q)
        local = np.einsum("ndb,nda->nba", np.conjugate(vecs), dvecs)
        berry = np.zeros((1, self.x_grid.size, local.shape[1], self.x_grid.size, local.shape[2]), dtype=complex)
        for n in range(self.x_grid.size):
            berry[0, n, :, n, :] = local[n]
        return berry

    def solver(self, *, include_berry=True):
        return LDRFG(
            self.kinetic_x,
            masses_y=[self.mass_y],
            energies=self.energies,
            overlap=self.overlap,
            grad_energies=self.grad_energies,
            grad_overlap=self.grad_overlap,
            berry=self.berry if include_berry else None,
            gamma=np.array([[self.gamma_y]], dtype=float),
        )


def read_xyz_trajectory(path, *, stride=1, max_frames=None, length_unit="bohr"):
    """Read an XYZ trajectory into :class:`XYZFrame` records.

    ``XYZTrajectoryWriter`` stores MD time in atomic units in the comment line.
    Coordinates are converted to Bohr when ``length_unit='angstrom'``.
    """

    path = Path(path)
    lines = path.read_text().splitlines()
    frames = []
    index = 0
    frame_index = 0
    scale = 1.0
    if length_unit.lower() in {"angstrom", "ang", "a"}:
        scale = 1.0 / au2angstrom
    elif length_unit.lower() not in {"bohr", "au", "atomic"}:
        raise ValueError("length_unit must be 'bohr' or 'angstrom'.")

    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        natoms = int(lines[index])
        comment = lines[index + 1].strip()
        if frame_index % int(stride) == 0:
            symbols = []
            positions = []
            for line in lines[index + 2 : index + 2 + natoms]:
                fields = line.split()
                symbols.append(fields[0])
                positions.append([float(fields[1]), float(fields[2]), float(fields[3])])
            frames.append(
                XYZFrame(
                    tuple(symbols),
                    np.asarray(positions, dtype=float) * scale,
                    _parse_xyz_time(comment),
                )
            )
            if max_frames is not None and len(frames) >= int(max_frames):
                break
        frame_index += 1
        index += natoms + 2
    return frames


def solvent_electric_field_coordinate(
    frames,
    *,
    solute_atoms,
    axis_atoms=(0, 1),
    box_size=None,
    normalize=True,
    scale=1.0,
):
    """Reduce solvent frames to a scalar electric field along a solute axis."""

    if len(frames) == 0:
        raise ValueError("at least one frame is required.")
    solute_atoms = int(solute_atoms)
    axis_atoms = tuple(int(i) for i in axis_atoms)
    raw = []
    for frame in frames:
        positions = np.asarray(frame.positions, dtype=float)
        center = np.mean(positions[:solute_atoms], axis=0)
        axis = positions[axis_atoms[1]] - positions[axis_atoms[0]]
        norm = np.linalg.norm(axis)
        if norm == 0.0:
            raise ValueError("axis atoms define a zero-length direction.")
        direction = axis / norm
        solvent_positions = positions[solute_atoms:]
        charges = _solvent_charges(len(solvent_positions))
        delta = center - solvent_positions
        if box_size is not None:
            box = np.asarray(box_size, dtype=float)
            delta -= box * np.round(delta / box)
        r = np.linalg.norm(delta, axis=1)
        active = r > 1.0e-12
        field = np.sum(charges[active, None] * delta[active] / r[active, None] ** 3, axis=0)
        raw.append(float(np.dot(field, direction)))

    q = np.asarray(raw, dtype=float) * float(scale)
    if normalize and q.size > 1:
        std = float(np.std(q))
        if std > 0.0:
            q = (q - float(np.mean(q))) / std
        else:
            q = q - float(np.mean(q))
    times = np.asarray(
        [float(i) if frame.time is None else float(frame.time) for i, frame in enumerate(frames)],
        dtype=float,
    )
    return times, q


def solvent_point_charges_from_frame(frame, *, solute_atoms, charges=None):
    """Return solvent point charges from a frame with water O-H-H triplets."""

    positions = np.asarray(frame.positions, dtype=float)
    solute_atoms = int(solute_atoms)
    solvent_positions = positions[solute_atoms:]
    if charges is None:
        solvent_charges = _solvent_charges(len(solvent_positions))
    else:
        solvent_charges = np.asarray(charges, dtype=float)
        if solvent_charges.shape != (len(solvent_positions),):
            raise ValueError(
                f"charges shape {solvent_charges.shape} != {(len(solvent_positions),)}."
            )
    return solvent_positions.copy(), solvent_charges.copy()


def methanol_full_fg_coordinate_path(
    frames,
    *,
    solute_atoms=6,
    include=("oh_stretch", "coh_bend", "solvent_cartesian"),
    source_frame_indices=None,
    width_by_group=None,
):
    """Return a single-FG coordinate path for methanol plus all solvent atoms.

    The C-O stretch is intentionally excluded because the methanol embedded
    workflow keeps that coordinate on the LDR/DVR grid.  Coordinates are the
    hydroxyl O-H stretch, the C-O-H angle, and solvent Cartesian coordinates in
    a methanol body frame whose origin is the C-O midpoint.
    """

    frames = tuple(frames)
    if not frames:
        raise ValueError("at least one frame is required.")
    solute_atoms = int(solute_atoms)
    include = tuple(include)
    allowed = {"oh_stretch", "coh_bend", "solvent_cartesian"}
    unknown = sorted(set(include) - allowed)
    if unknown:
        raise ValueError(f"unknown methanol FG coordinate groups: {unknown}.")
    if solute_atoms < 6:
        raise ValueError("methanol FG coordinates require at least 6 solute atoms.")
    source_frame_indices = (
        list(range(len(frames)))
        if source_frame_indices is None
        else [int(index) for index in source_frame_indices]
    )
    if len(source_frame_indices) != len(frames):
        raise ValueError("source_frame_indices length must match frames.")

    first_labels, first_groups, first_centers = _methanol_fg_centers(
        frames[0],
        solute_atoms=solute_atoms,
        include=include,
    )
    masses = _methanol_fg_masses(frames[0], first_labels, first_groups, solute_atoms=solute_atoms)
    widths = _methanol_fg_widths(first_groups, width_by_group=width_by_group)
    centers = []
    labels_by_frame = []
    groups_by_frame = []
    for frame in frames:
        labels, groups, frame_centers = _methanol_fg_centers(
            frame,
            solute_atoms=solute_atoms,
            include=include,
        )
        labels_by_frame.append(labels)
        groups_by_frame.append(groups)
        centers.append(frame_centers)
    centers = np.asarray(centers, dtype=float)
    times = np.asarray(
        [float(index) if frame.time is None else float(frame.time) for index, frame in enumerate(frames)],
        dtype=float,
    )
    momenta = _finite_difference_momenta(centers, times, masses)

    path = []
    for index, frame in enumerate(frames):
        path.append(
            MethanolFGCoordinateFrame(
                labels=labels_by_frame[index],
                groups=groups_by_frame[index],
                centers=centers[index].copy(),
                momenta=momenta[index].copy(),
                masses=masses.copy(),
                widths=widths.copy(),
                time=None if frame.time is None else float(frame.time),
                source_frame=int(source_frame_indices[index]),
            )
        )
    return tuple(path)


def methanol_fg_path_diagnostics(path, *, width_scaled_jump_threshold=10.0):
    """Return readiness diagnostics for a methanol full-coordinate FG path."""

    path = tuple(path)
    if not path:
        raise ValueError("at least one FG frame is required.")
    labels0 = tuple(path[0].labels)
    groups0 = tuple(path[0].groups)
    centers = np.asarray([frame.centers for frame in path], dtype=float)
    momenta = np.asarray([frame.momenta for frame in path], dtype=float)
    masses = np.asarray(path[0].masses, dtype=float)
    widths = np.asarray(path[0].widths, dtype=float)
    labels_stable = all(tuple(frame.labels) == labels0 for frame in path)
    groups_stable = all(tuple(frame.groups) == groups0 for frame in path)
    masses_stable = all(np.allclose(np.asarray(frame.masses, dtype=float), masses) for frame in path)
    widths_stable = all(np.allclose(np.asarray(frame.widths, dtype=float), widths) for frame in path)
    finite_centers = bool(np.all(np.isfinite(centers)))
    finite_momenta = bool(np.all(np.isfinite(momenta)))
    positive_masses = bool(np.all(np.isfinite(masses)) and np.all(masses > 0.0))
    positive_widths = bool(np.all(np.isfinite(widths)) and np.all(widths > 0.0))

    if centers.shape[0] > 1:
        delta = np.diff(centers, axis=0)
        displacement_norm = np.linalg.norm(delta, axis=1)
        width_scaled = np.sqrt(np.sum(widths[None, :] * delta * delta, axis=1))
        overlap_magnitude = np.exp(-0.25 * width_scaled * width_scaled)
    else:
        displacement_norm = np.zeros(0, dtype=float)
        width_scaled = np.zeros(0, dtype=float)
        overlap_magnitude = np.ones(0, dtype=float)
    max_width_scaled = float(np.max(width_scaled)) if width_scaled.size else 0.0
    min_overlap = float(np.min(overlap_magnitude)) if overlap_magnitude.size else 1.0
    jump_ready = bool(max_width_scaled <= float(width_scaled_jump_threshold))
    ready = bool(
        labels_stable
        and groups_stable
        and masses_stable
        and widths_stable
        and finite_centers
        and finite_momenta
        and positive_masses
        and positive_widths
        and jump_ready
    )
    failed = []
    for name, ok in (
        ("labels", labels_stable),
        ("groups", groups_stable),
        ("masses", masses_stable and positive_masses),
        ("widths", widths_stable and positive_widths),
        ("centers", finite_centers),
        ("momenta", finite_momenta),
        ("frame_jumps", jump_ready),
    ):
        if not ok:
            failed.append(name)
    return {
        "fg_path_ready": ready,
        "verdict": "ready" if ready else "fg_path_limited",
        "recommendation": (
            "Full-coordinate methanol FG path is stable for the requested frames."
            if ready
            else "Reduce frame stride, check methanol body-frame construction, or widen FG coordinates."
        ),
        "failed_checks": failed,
        "frame_count": int(len(path)),
        "coordinate_count": int(len(labels0)),
        "labels": list(labels0),
        "groups": list(groups0),
        "group_counts": _count_items(groups0),
        "source_frames": [
            None if frame.source_frame is None else int(frame.source_frame) for frame in path
        ],
        "times": [None if frame.time is None else float(frame.time) for frame in path],
        "labels_stable": bool(labels_stable),
        "groups_stable": bool(groups_stable),
        "masses_stable": bool(masses_stable),
        "widths_stable": bool(widths_stable),
        "finite_centers": finite_centers,
        "finite_momenta": finite_momenta,
        "positive_masses": positive_masses,
        "positive_widths": positive_widths,
        "width_scaled_jump_threshold": float(width_scaled_jump_threshold),
        "max_width_scaled_displacement": max_width_scaled,
        "displacement_norm": displacement_norm.tolist(),
        "width_scaled_displacement": width_scaled.tolist(),
        "gaussian_overlap_magnitude": overlap_magnitude.tolist(),
        "min_gaussian_overlap_magnitude": min_overlap,
        "center_shape": list(centers.shape),
        "momenta_shape": list(momenta.shape),
    }


def embedded_ldrfg_path_linearized_model(
    snapshots,
    fg_path,
    kinetic_x,
    *,
    reference_index=0,
    classical_force=None,
):
    """Build a one-FG TDVP model from embedded LDR snapshots along an FG path.

    The electronic Hamiltonian is linearized around ``reference_index`` by a
    least-squares fit of APES changes against full-coordinate FG displacements.
    This gives the TDVP solver a genuine ``-<C|dH/dQ|C>`` electronic force, but
    the derivative is still an audit-level path surrogate rather than a full
    arbitrary-displacement QM/MM gradient.
    """

    snapshots = _validate_embedded_snapshots(snapshots)
    fg_path = tuple(fg_path)
    if len(snapshots) != len(fg_path):
        raise ValueError(f"snapshots length {len(snapshots)} != fg_path length {len(fg_path)}.")
    if len(fg_path) == 0:
        raise ValueError("fg_path must contain at least one frame.")
    reference_index = int(reference_index)
    if reference_index < 0 or reference_index >= len(fg_path):
        raise ValueError("reference_index is outside the FG path.")

    reference = fg_path[reference_index]
    labels = tuple(reference.labels)
    groups = tuple(reference.groups)
    q_ref = np.asarray(reference.centers, dtype=float)
    p_ref = np.asarray(reference.momenta, dtype=float)
    masses = np.asarray(reference.masses, dtype=float)
    widths = np.asarray(reference.widths, dtype=float)
    if q_ref.ndim != 1 or p_ref.shape != q_ref.shape:
        raise ValueError("reference FG centers and momenta must be one-dimensional and aligned.")
    if masses.shape != q_ref.shape or widths.shape != q_ref.shape:
        raise ValueError("reference FG masses and widths must match centers.")
    if np.any(masses <= 0.0) or np.any(widths <= 0.0):
        raise ValueError("reference FG masses and widths must be positive.")
    for frame in fg_path:
        if tuple(frame.labels) != labels or tuple(frame.groups) != groups:
            raise ValueError("FG path labels and groups must be stable.")
        if np.asarray(frame.centers, dtype=float).shape != q_ref.shape:
            raise ValueError("FG path centers must have a stable shape.")

    apes_sequence = np.asarray([snapshot.apes for snapshot in snapshots], dtype=float)
    apes_ref = apes_sequence[reference_index]
    overlap_ref = np.asarray(snapshots[reference_index].overlap, dtype=complex)
    q_sequence = np.asarray([frame.centers for frame in fg_path], dtype=float)
    x = q_sequence - q_ref[None, :]
    y = apes_sequence.reshape(len(snapshots), -1) - apes_ref.reshape(1, -1)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("FG path and embedded APES must be finite.")
    if len(fg_path) > 1 and np.linalg.norm(x) > 0.0:
        beta, residuals, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if np.asarray(residuals).size:
            gradient_residual = float(np.sqrt(np.sum(residuals) / max(1, y.size)))
        else:
            gradient_residual = float(np.linalg.norm(x @ beta - y) / max(1, y.size) ** 0.5)
    else:
        beta = np.zeros((q_ref.size, y.shape[1]), dtype=float)
        rank = 0
        gradient_residual = 0.0
    grad_apes = beta.reshape(q_ref.size, *apes_ref.shape)

    def energies(q):
        dq = np.asarray(q, dtype=float) - q_ref
        return apes_ref + np.einsum("j,jga->ga", dq, grad_apes, optimize=True)

    def grad_energies(_q):
        return grad_apes

    def overlap(_q):
        return overlap_ref

    solver = LDRFG(
        kinetic_x,
        masses,
        energies=energies,
        overlap=overlap,
        grad_energies=grad_energies,
        grad_overlap=None,
        gamma=np.diag(widths),
    )
    return EmbeddedLDRFGTDVPModel(
        solver=solver,
        reference_q=q_ref.copy(),
        reference_p=p_ref.copy(),
        reference_index=reference_index,
        labels=labels,
        groups=groups,
        force_model="path_linearized_embedded_ldrfg",
        electronic_force_source="least_squares_apES_gradient_along_fg_path",
        classical_force_source="callback" if classical_force is not None else "none",
        electronic_gradient_rank=int(rank),
        electronic_gradient_residual=gradient_residual,
    )


def methanol_fg_path_classical_forces(fg_path):
    """Estimate classical forces on FG coordinates from path momenta."""

    fg_path = tuple(fg_path)
    if len(fg_path) == 0:
        raise ValueError("fg_path must contain at least one frame.")
    momenta = np.asarray([frame.momenta for frame in fg_path], dtype=float)
    if momenta.ndim != 2:
        raise ValueError("FG path momenta must form a two-dimensional array.")
    if len(fg_path) == 1:
        return np.zeros_like(momenta)
    times = np.asarray(
        [float(index) if frame.time is None else float(frame.time) for index, frame in enumerate(fg_path)],
        dtype=float,
    )
    if np.any(np.diff(times) <= 0.0):
        times = np.arange(len(fg_path), dtype=float)
    return np.gradient(momenta, times, axis=0, edge_order=1)


def methanol_fg_path_force_callback(fg_path, forces=None):
    """Return a nearest-anchor classical force callback for an FG path."""

    fg_path = tuple(fg_path)
    if len(fg_path) == 0:
        raise ValueError("fg_path must contain at least one frame.")
    centers = np.asarray([frame.centers for frame in fg_path], dtype=float)
    if forces is None:
        forces = methanol_fg_path_classical_forces(fg_path)
    forces = np.asarray(forces, dtype=float)
    if forces.shape != centers.shape:
        raise ValueError(f"forces shape {forces.shape} != centers shape {centers.shape}.")

    def callback(q, *, t=None, step=None):
        q = np.asarray(q, dtype=float)
        if step is not None:
            index = int(np.clip(int(step), 0, len(fg_path) - 1))
        else:
            index = int(np.argmin(np.linalg.norm(centers - q[None, :], axis=1)))
        return forces[index]

    return callback


def propagate_liquid_ldrfg_tdvp(
    model,
    c0,
    q0,
    p0,
    times,
    *,
    classical_force=None,
    normalize=True,
):
    """Propagate coupled ``C,Q,P`` liquid LDRFG TDVP dynamics."""

    solver = model.solver if isinstance(model, EmbeddedLDRFGTDVPModel) else model
    if not isinstance(solver, LDRFG):
        raise TypeError("model must be an EmbeddedLDRFGTDVPModel or LDRFG instance.")
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times must be a non-empty one-dimensional array.")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    c = np.asarray(c0, dtype=complex)
    q = np.asarray(q0, dtype=float)
    p = np.asarray(p0, dtype=float)
    solver._validate_c(c)
    solver._validate_q(q)
    solver._validate_p(p)

    c_history = np.zeros((times.size, *c.shape), dtype=complex)
    q_history = np.zeros((times.size, solver.ny), dtype=float)
    p_history = np.zeros_like(q_history)
    electronic_force_history = np.zeros_like(q_history)
    classical_force_history = np.zeros_like(q_history)
    total_force_history = np.zeros_like(q_history)
    norm = np.zeros(times.size, dtype=float)
    energy = np.zeros(times.size, dtype=float)

    def call_classical(q_value, t_value, step_value):
        if classical_force is None:
            return np.zeros(solver.ny, dtype=float)
        try:
            value = classical_force(q_value, t=t_value, step=step_value)
        except TypeError:
            try:
                value = classical_force(q_value, t_value, step_value)
            except TypeError:
                value = classical_force(q_value)
        value = np.asarray(value, dtype=float)
        if value.shape != (solver.ny,):
            raise ValueError(f"classical force shape {value.shape} != {(solver.ny,)}.")
        return value

    for index, t_value in enumerate(times):
        c_history[index] = c
        q_history[index] = q
        p_history[index] = p
        electronic_force_history[index] = solver.force(c, q, p)
        classical_force_history[index] = call_classical(q, float(t_value), index)
        total_force_history[index] = electronic_force_history[index] + classical_force_history[index]
        c_flat = c.reshape(-1)
        norm[index] = float(np.vdot(c_flat, c_flat).real)
        energy_value = complex(solver.energy(c, q, p))
        if abs(energy_value.imag) > 1.0e-9 * max(1.0, abs(energy_value.real)):
            raise ValueError(f"LDRFG energy has a non-negligible imaginary component: {energy_value!r}.")
        energy[index] = float(energy_value.real)
        if index == times.size - 1:
            break
        dt = float(times[index + 1] - t_value)
        c_half = solver.propagate_coefficients(c, q, p, 0.5 * dt)
        f0 = solver.force(c_half, q, p) + call_classical(q, float(t_value), index)
        p_half = p + 0.5 * dt * f0
        q_new = q + dt * solver.inv_masses_y * p_half
        f1 = solver.force(c_half, q_new, p_half) + call_classical(
            q_new,
            float(times[index + 1]),
            index + 1,
        )
        p_new = p_half + 0.5 * dt * f1
        c_new = solver.propagate_coefficients(c_half, q_new, p_new, 0.5 * dt)
        if normalize:
            c_norm = np.sqrt(np.vdot(c_new.reshape(-1), c_new.reshape(-1)).real)
            if c_norm <= 0.0:
                raise ValueError("LDRFG coefficient norm vanished during propagation.")
            c_new = c_new / c_norm
        c, q, p = c_new, q_new, p_new

    return {
        "times": times.copy(),
        "coefficients": c_history,
        "q": q_history,
        "p": p_history,
        "electronic_force": electronic_force_history,
        "classical_force": classical_force_history,
        "total_force": total_force_history,
        "norm": norm,
        "energy": energy,
        "force_model": model.force_model if isinstance(model, EmbeddedLDRFGTDVPModel) else "ldrfg",
        "electronic_force_source": (
            model.electronic_force_source if isinstance(model, EmbeddedLDRFGTDVPModel) else "solver"
        ),
        "classical_force_source": "callback" if classical_force is not None else (
            model.classical_force_source if isinstance(model, EmbeddedLDRFGTDVPModel) else "none"
        ),
    }


def _methanol_fg_centers(frame, *, solute_atoms, include):
    positions = np.asarray(frame.positions, dtype=float)
    symbols = tuple(str(symbol) for symbol in frame.symbols)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("frame positions must have shape (natoms, 3).")
    if len(symbols) != positions.shape[0]:
        raise ValueError("frame symbols and positions length mismatch.")
    if positions.shape[0] < solute_atoms:
        raise ValueError("frame contains fewer atoms than solute_atoms.")
    c = positions[0]
    o = positions[1]
    h_oh = positions[5]
    labels = []
    groups = []
    values = []
    if "oh_stretch" in include:
        labels.append("methanol:O-H")
        groups.append("oh_stretch")
        values.append(float(np.linalg.norm(h_oh - o)))
    if "coh_bend" in include:
        labels.append("methanol:C-O-H")
        groups.append("coh_bend")
        values.append(float(_angle(c - o, h_oh - o)))
    if "solvent_cartesian" in include:
        origin, axes = _methanol_body_frame(positions)
        solvent = positions[solute_atoms:]
        body = (solvent - origin[None, :]) @ axes.T
        for atom_index, (symbol, coords) in enumerate(zip(symbols[solute_atoms:], body), start=solute_atoms):
            for axis_label, value in zip(("x", "y", "z"), coords):
                labels.append(f"solvent:{atom_index}:{symbol}:{axis_label}")
                groups.append("solvent_cartesian")
                values.append(float(value))
    return tuple(labels), tuple(groups), np.asarray(values, dtype=float)


def _methanol_body_frame(positions):
    c = np.asarray(positions[0], dtype=float)
    o = np.asarray(positions[1], dtype=float)
    h_oh = np.asarray(positions[5], dtype=float)
    origin = 0.5 * (c + o)
    z_axis = _unit(o - c, "C-O body-frame axis")
    h_vec = h_oh - o
    x_vec = h_vec - np.dot(h_vec, z_axis) * z_axis
    if np.linalg.norm(x_vec) < 1.0e-12:
        x_vec = _orthogonal_fallback(z_axis)
    x_axis = _unit(x_vec, "C-O-H body-frame axis")
    y_axis = _unit(np.cross(z_axis, x_axis), "methanol body-frame y axis")
    x_axis = _unit(np.cross(y_axis, z_axis), "methanol body-frame x axis")
    return origin, np.vstack((x_axis, y_axis, z_axis))


def _methanol_fg_masses(frame, labels, groups, *, solute_atoms):
    positions = np.asarray(frame.positions, dtype=float)
    symbols = tuple(str(symbol).upper() for symbol in frame.symbols)
    m_h = atomic_mass["H"] * amu2au
    m_o = atomic_mass["O"] * amu2au
    reduced_oh = m_h * m_o / (m_h + m_o)
    oh_length = float(np.linalg.norm(positions[5] - positions[1]))
    bend_mass = max(reduced_oh * oh_length * oh_length, 1.0)
    masses = []
    solvent_mass_by_atom = {
        atom_index: atomic_mass.get(symbols[atom_index], 12.0) * amu2au
        for atom_index in range(solute_atoms, len(symbols))
    }
    for label, group in zip(labels, groups):
        if group == "oh_stretch":
            masses.append(reduced_oh)
        elif group == "coh_bend":
            masses.append(bend_mass)
        elif group == "solvent_cartesian":
            atom_index = int(label.split(":")[1])
            masses.append(solvent_mass_by_atom[atom_index])
        else:
            masses.append(1.0)
    return np.asarray(masses, dtype=float)


def _methanol_fg_widths(groups, *, width_by_group=None):
    defaults = {
        "oh_stretch": 1.0,
        "coh_bend": 4.0,
        "solvent_cartesian": 0.25,
    }
    if width_by_group:
        defaults.update({str(key): float(value) for key, value in width_by_group.items()})
    return np.asarray([defaults.get(group, 1.0) for group in groups], dtype=float)


def _finite_difference_momenta(centers, times, masses):
    centers = np.asarray(centers, dtype=float)
    times = np.asarray(times, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if centers.shape[0] == 1:
        return np.zeros_like(centers)
    if times.shape != (centers.shape[0],) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        times = np.arange(centers.shape[0], dtype=float)
    velocities = np.empty_like(centers)
    for coord in range(centers.shape[1]):
        velocities[:, coord] = np.gradient(centers[:, coord], times, edge_order=1)
    return velocities * masses[None, :]


def _angle(left, right):
    left = _unit(left, "angle vector")
    right = _unit(right, "angle vector")
    return float(np.arccos(np.clip(np.dot(left, right), -1.0, 1.0)))


def _unit(vector, label):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-14:
        raise ValueError(f"{label} has zero length.")
    return vector / norm


def _orthogonal_fallback(axis):
    axis = np.asarray(axis, dtype=float)
    candidates = np.eye(3)
    candidate = min(candidates, key=lambda item: abs(float(np.dot(item, axis))))
    return candidate - np.dot(candidate, axis) * axis


def _count_items(values):
    counts = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def h2_bond_geometry(bond_length, *, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    """Return a two-atom H2 geometry for an LDR bond coordinate."""

    center = np.asarray(center, dtype=float)
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("axis must not be zero.")
    direction = axis / norm
    half = 0.5 * float(bond_length) * direction
    return np.vstack((center - half, center + half))


def solute_bond_distance_geometry_builder(
    frame,
    *,
    solute_atoms,
    atom_pair=(0, 1),
    moving_atoms=None,
):
    """Return a geometry builder for one solute bond-distance coordinate.

    The builder preserves the solute geometry from ``frame`` except for atoms
    listed in ``moving_atoms``.  Those atoms are translated along the current
    ``atom_pair`` axis so the pair distance equals the requested LDR coordinate.
    This is useful for first liquid-phase solute scans such as a C-O or O-H
    stretch in a solvated molecule.
    """

    positions = np.asarray(frame.positions[: int(solute_atoms)], dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("frame positions must have shape (natoms, 3).")
    atom_i, atom_j = (int(atom_pair[0]), int(atom_pair[1]))
    if not (0 <= atom_i < positions.shape[0] and 0 <= atom_j < positions.shape[0]):
        raise ValueError("atom_pair indices must be inside the solute atom range.")
    reference_axis = positions[atom_j] - positions[atom_i]
    reference_distance = float(np.linalg.norm(reference_axis))
    if reference_distance == 0.0:
        raise ValueError("atom_pair defines a zero-length direction.")
    if moving_atoms is None:
        moving_atoms = (atom_j,)
    moving_atoms = tuple(int(index) for index in moving_atoms)
    for index in moving_atoms:
        if not 0 <= index < positions.shape[0]:
            raise ValueError("moving_atoms indices must be inside the solute atom range.")

    def builder(distance, frame=None, **_kwargs):
        if frame is None:
            source_positions = positions
        else:
            source_positions = np.asarray(frame.positions[: int(solute_atoms)], dtype=float)
            if source_positions.shape != positions.shape:
                raise ValueError(f"frame solute positions shape {source_positions.shape} != {positions.shape}.")
        axis = source_positions[atom_j] - source_positions[atom_i]
        current_distance = float(np.linalg.norm(axis))
        if current_distance == 0.0:
            raise ValueError("atom_pair defines a zero-length direction.")
        geometry = source_positions.copy()
        displacement = (float(distance) - current_distance) * axis / current_distance
        geometry[list(moving_atoms)] += displacement
        return geometry

    return builder


def embedded_casci_ldr_snapshot(
    frame,
    coordinate_grid,
    *,
    solute_symbols=None,
    geometry_builder,
    solute_atoms,
    basis="sto-3g",
    charge=0,
    nstates=2,
    ncas=2,
    nelecas=2,
    spin=0,
    method="direct_ci",
    run_kwargs=None,
    reference_run_kwargs=None,
    electronic_runner=None,
    keep_objects=True,
    solvent_charges=None,
    solvent_coordinate=None,
    solvent_coordinate_builder=None,
):
    """Build a solvent-embedded CASCI LDR snapshot for an arbitrary solute."""

    if solute_symbols is None:
        solute_symbols = tuple(frame.symbols[: int(solute_atoms)])
    else:
        solute_symbols = tuple(str(symbol) for symbol in solute_symbols)
    if len(solute_symbols) != int(solute_atoms):
        raise ValueError("solute_symbols length must match solute_atoms.")

    def runner(geometry, point_charge_coords, point_charge_charges):
        if electronic_runner is not None:
            return electronic_runner(geometry, point_charge_coords, point_charge_charges)
        return _embedded_casci_point(
            solute_symbols,
            geometry,
            point_charge_coords,
            point_charge_charges,
            basis=basis,
            charge=charge,
            nstates=nstates,
            ncas=ncas,
            nelecas=nelecas,
            spin=spin,
            method=method,
            run_kwargs=run_kwargs,
            reference_run_kwargs=reference_run_kwargs,
        )

    return solvent_embedded_ldr_snapshot(
        frame,
        coordinate_grid,
        geometry_builder=geometry_builder,
        electronic_runner=runner,
        solute_atoms=solute_atoms,
        nstates=nstates,
        solvent_charges=solvent_charges,
        solvent_coordinate=solvent_coordinate,
        solvent_coordinate_builder=solvent_coordinate_builder,
        keep_objects=keep_objects,
    )


def build_embedded_casci_ldr_trajectory(
    frames,
    coordinate_grid,
    *,
    frame_indices=None,
    **snapshot_kwargs,
):
    """Build arbitrary-solute embedded CASCI LDR data for multiple frames."""

    frames = tuple(frames)
    if len(frames) == 0:
        raise ValueError("at least one frame is required.")
    if frame_indices is None:
        frame_indices = range(len(frames))
    frame_indices = tuple(int(i) for i in frame_indices)
    snapshots = []
    times = []
    for frame_index in frame_indices:
        frame = frames[frame_index]
        snapshots.append(
            embedded_casci_ldr_snapshot(
                frame,
                coordinate_grid,
                **snapshot_kwargs,
            )
        )
        times.append(float(frame_index) if frame.time is None else float(frame.time))
    times = _strictly_increasing_times(times)
    return SolventEmbeddedLDRTrajectory(tuple(snapshots), times)


def embedded_h2_casci_ldr_snapshot(
    frame,
    bond_grid,
    *,
    solute_atoms=2,
    axis_atoms=(0, 1),
    center=None,
    basis="sto-3g",
    nstates=2,
    ncas=2,
    nelecas=2,
    spin=0,
    method="direct_ci",
    run_kwargs=None,
    reference_run_kwargs=None,
    electronic_runner=None,
    keep_objects=True,
):
    """Build solvent-embedded H2 CASCI APES and LDR overlaps for one frame.

    The default path uses PyQED RHF/CASCI embedded in the solvent point charges.
    ``electronic_runner`` is an optional test hook with signature
    ``runner(geometry, point_charge_coords, point_charge_charges)`` returning
    ``(energies, electronic_object)``.
    """

    positions = np.asarray(frame.positions, dtype=float)
    if center is None:
        center = np.mean(positions[: int(solute_atoms)], axis=0)
    axis_atoms = tuple(int(i) for i in axis_atoms)
    axis = positions[axis_atoms[1]] - positions[axis_atoms[0]]

    def geometry_builder(bond, **_kwargs):
        return h2_bond_geometry(float(bond), center=center, axis=axis)

    def runner(geometry, point_charge_coords, point_charge_charges):
        if electronic_runner is None:
            return _embedded_h2_casci_point(
                geometry,
                point_charge_coords,
                point_charge_charges,
                basis=basis,
                nstates=nstates,
                ncas=ncas,
                nelecas=nelecas,
                spin=spin,
                method=method,
                run_kwargs=run_kwargs,
                reference_run_kwargs=reference_run_kwargs,
            )
        return electronic_runner(geometry, point_charge_coords, point_charge_charges)

    try:
        _, q_values = solvent_electric_field_coordinate(
            [frame],
            solute_atoms=solute_atoms,
            axis_atoms=axis_atoms,
            normalize=False,
        )
        solvent_coordinate = float(q_values[0])
    except Exception:
        solvent_coordinate = None

    return solvent_embedded_ldr_snapshot(
        frame,
        bond_grid,
        geometry_builder=geometry_builder,
        electronic_runner=runner,
        solute_atoms=solute_atoms,
        nstates=nstates,
        keep_objects=keep_objects,
        solvent_coordinate=solvent_coordinate,
    )


def solvent_embedded_ldr_snapshot(
    frame,
    coordinate_grid,
    *,
    geometry_builder,
    electronic_runner,
    solute_atoms,
    nstates=2,
    solvent_charges=None,
    solvent_coordinate=None,
    solvent_coordinate_builder=None,
    keep_objects=True,
):
    """Build one solvent-embedded LDR APES/overlap snapshot.

    ``geometry_builder`` maps each LDR coordinate value to a solute geometry.
    It is called as ``geometry_builder(value, frame=..., point_charge_coords=...,
    point_charge_charges=...)``.  ``electronic_runner`` then evaluates that
    geometry in the solvent point-charge field and returns
    ``(energies, electronic_object)``.  The returned electronic objects must
    expose ``wavefunction_overlap`` or ``overlap``; otherwise the CASCI overlap
    helper is used.
    """

    coordinate_grid = np.asarray(coordinate_grid, dtype=float)
    if coordinate_grid.ndim != 1 or coordinate_grid.size == 0:
        raise ValueError("coordinate_grid must be a non-empty one-dimensional array.")
    if int(nstates) <= 0:
        raise ValueError("nstates must be positive.")

    pc_coords, pc_charges = solvent_point_charges_from_frame(
        frame,
        solute_atoms=solute_atoms,
        charges=solvent_charges,
    )

    objects = []
    apes = np.zeros((coordinate_grid.size, int(nstates)), dtype=float)
    for index, coordinate in enumerate(coordinate_grid):
        geometry = geometry_builder(
            float(coordinate),
            frame=frame,
            point_charge_coords=pc_coords,
            point_charge_charges=pc_charges,
        )
        energies, obj = electronic_runner(geometry, pc_coords, pc_charges)
        energies = np.asarray(energies, dtype=float)
        if energies.shape[0] < int(nstates):
            raise ValueError(f"electronic runner returned {energies.shape[0]} states, expected {nstates}.")
        apes[index] = energies[: int(nstates)]
        objects.append(obj)

    overlap_tensor = np.zeros(
        (coordinate_grid.size, int(nstates), coordinate_grid.size, int(nstates)),
        dtype=complex,
    )
    for i, left in enumerate(objects):
        for j, right in enumerate(objects):
            block = _electronic_overlap_block(left, right, int(nstates), diagonal=i == j)
            if block.shape[0] < int(nstates) or block.shape[1] < int(nstates):
                raise ValueError(f"overlap block shape {block.shape} does not contain {nstates} states.")
            overlap_tensor[i, :, j, :] = block[: int(nstates), : int(nstates)]

    if solvent_coordinate_builder is not None:
        solvent_coordinate = solvent_coordinate_builder(
            frame,
            point_charge_coords=pc_coords,
            point_charge_charges=pc_charges,
        )
    if solvent_coordinate is not None:
        solvent_coordinate = float(solvent_coordinate)

    return SolventEmbeddedLDRSnapshot(
        bond_grid=coordinate_grid.copy(),
        apes=apes,
        overlap=overlap_tensor,
        point_charge_coords=pc_coords,
        point_charge_charges=pc_charges,
        solvent_coordinate=solvent_coordinate,
        electronic_objects=tuple(objects) if keep_objects else None,
    )


def build_solvent_embedded_ldr_trajectory(
    frames,
    coordinate_grid,
    *,
    frame_indices=None,
    **snapshot_kwargs,
):
    """Build generic solvent-embedded LDR electronic data for multiple frames."""

    frames = tuple(frames)
    if len(frames) == 0:
        raise ValueError("at least one frame is required.")
    if frame_indices is None:
        frame_indices = range(len(frames))
    frame_indices = tuple(int(i) for i in frame_indices)
    snapshots = []
    times = []
    for frame_index in frame_indices:
        frame = frames[frame_index]
        snapshots.append(
            solvent_embedded_ldr_snapshot(
                frame,
                coordinate_grid,
                **snapshot_kwargs,
            )
        )
        times.append(float(frame_index) if frame.time is None else float(frame.time))
    times = _strictly_increasing_times(times)
    return SolventEmbeddedLDRTrajectory(tuple(snapshots), times)


def build_embedded_h2_casci_ldr_trajectory(
    frames,
    bond_grid,
    *,
    frame_indices=None,
    **snapshot_kwargs,
):
    """Build embedded H2 LDR electronic data for multiple liquid frames."""

    frames = tuple(frames)
    if len(frames) == 0:
        raise ValueError("at least one frame is required.")
    if frame_indices is None:
        frame_indices = range(len(frames))
    frame_indices = tuple(int(i) for i in frame_indices)
    snapshots = tuple(
        embedded_h2_casci_ldr_snapshot(
            frames[frame_index],
            bond_grid,
            **snapshot_kwargs,
        )
        for frame_index in frame_indices
    )
    times = _strictly_increasing_times(
        [
            float(frame_index) if frames[frame_index].time is None else float(frames[frame_index].time)
            for frame_index in frame_indices
        ]
    )
    return SolventEmbeddedLDRTrajectory(
        snapshots,
        times,
    )


def embedded_ldr_hamiltonian(snapshot, kinetic_x, *, symmetrize=True):
    """Return the flattened LDR Hamiltonian for an embedded snapshot."""

    kinetic_x = np.asarray(kinetic_x, dtype=complex)
    apes = np.asarray(snapshot.apes, dtype=float)
    overlap = np.asarray(snapshot.overlap, dtype=complex)
    ngrid, nstates = apes.shape
    if kinetic_x.shape != (ngrid, ngrid):
        raise ValueError(f"kinetic_x shape {kinetic_x.shape} != {(ngrid, ngrid)}.")
    if overlap.shape != (ngrid, nstates, ngrid, nstates):
        raise ValueError(
            f"overlap shape {overlap.shape} != {(ngrid, nstates, ngrid, nstates)}."
        )

    h_tensor = np.einsum("mn,mbna->mbna", kinetic_x, overlap, optimize=True)
    for n in range(ngrid):
        for state in range(nstates):
            h_tensor[n, state, n, state] += apes[n, state]
    h = h_tensor.reshape(ngrid * nstates, ngrid * nstates)
    if symmetrize:
        h = 0.5 * (h + h.conj().T)
    return h


def embedded_ldr_trajectory_diagnostics(snapshots, times=None, kinetic_x=None):
    """Return quality diagnostics for a solvent-embedded LDR trajectory."""

    snapshots = _validate_embedded_snapshots(snapshots)
    nsnap = len(snapshots)
    times_array = None
    if times is not None:
        times_array = np.asarray(times, dtype=float)
        if times_array.shape != (nsnap,):
            raise ValueError(f"times shape {times_array.shape} != {(nsnap,)}.")
        if nsnap > 1 and np.any(np.diff(times_array) <= 0.0):
            raise ValueError("times must be strictly increasing.")

    ngrid, nstates = snapshots[0].apes.shape
    identity = np.eye(nstates, dtype=complex)
    gap_min = np.full(nsnap, np.nan, dtype=float)
    apes_min = np.zeros((nsnap, nstates), dtype=float)
    apes_max = np.zeros((nsnap, nstates), dtype=float)
    overlap_identity_error = np.zeros(nsnap, dtype=float)
    overlap_hermiticity_error = np.zeros(nsnap, dtype=float)
    overlap_eigenvalue_min = np.zeros(nsnap, dtype=float)
    overlap_eigenvalue_max = np.zeros(nsnap, dtype=float)
    solvent_coordinate = np.full(nsnap, np.nan, dtype=float)
    hamiltonian_hermiticity_error = np.full(nsnap, np.nan, dtype=float)
    hamiltonian_trace = np.full(nsnap, np.nan, dtype=float)

    for index, snapshot in enumerate(snapshots):
        apes = np.asarray(snapshot.apes, dtype=float)
        overlap = np.asarray(snapshot.overlap, dtype=complex)
        if not np.all(np.isfinite(apes)):
            raise ValueError("embedded APES contains non-finite values.")
        if not np.all(np.isfinite(overlap)):
            raise ValueError("embedded overlap contains non-finite values.")
        apes_min[index] = np.min(apes, axis=0)
        apes_max[index] = np.max(apes, axis=0)
        if nstates >= 2:
            gap_min[index] = float(np.min(apes[:, 1] - apes[:, 0]))
        overlap_matrix = overlap.reshape(ngrid * nstates, ngrid * nstates)
        overlap_h = 0.5 * (overlap_matrix + overlap_matrix.conj().T)
        overlap_hermiticity_error[index] = float(np.max(np.abs(overlap_matrix - overlap_matrix.conj().T)))
        overlap_eigs = np.linalg.eigvalsh(overlap_h)
        overlap_eigenvalue_min[index] = float(np.min(overlap_eigs).real)
        overlap_eigenvalue_max[index] = float(np.max(overlap_eigs).real)
        diagonal_errors = [
            np.max(np.abs(overlap[grid_index, :, grid_index, :] - identity))
            for grid_index in range(ngrid)
        ]
        overlap_identity_error[index] = float(np.max(diagonal_errors))
        if snapshot.solvent_coordinate is not None:
            solvent_coordinate[index] = float(snapshot.solvent_coordinate)
        if kinetic_x is not None:
            hamiltonian = embedded_ldr_hamiltonian(snapshot, kinetic_x, symmetrize=False)
            hamiltonian_hermiticity_error[index] = float(
                np.max(np.abs(hamiltonian - hamiltonian.conj().T))
            )
            hamiltonian_trace[index] = float(np.trace(hamiltonian).real)

    apes_frame_rms_delta = np.zeros(max(nsnap - 1, 0), dtype=float)
    if nsnap > 1:
        apes_sequence = np.asarray([snapshot.apes for snapshot in snapshots], dtype=float)
        apes_frame_rms_delta = np.sqrt(np.mean(np.diff(apes_sequence, axis=0) ** 2, axis=(1, 2)))

    solvent_coordinate_velocity = np.full(nsnap, np.nan, dtype=float)
    if times_array is not None and nsnap > 1 and np.all(np.isfinite(solvent_coordinate)):
        solvent_coordinate_velocity = np.gradient(solvent_coordinate, times_array)

    finite = (
        np.all(np.isfinite(gap_min) | np.isnan(gap_min))
        and np.all(np.isfinite(apes_min))
        and np.all(np.isfinite(apes_max))
        and np.all(np.isfinite(overlap_identity_error))
        and np.all(np.isfinite(overlap_hermiticity_error))
        and np.all(np.isfinite(overlap_eigenvalue_min))
        and np.all(np.isfinite(overlap_eigenvalue_max))
    )
    return {
        "gap_min": gap_min,
        "apes_min": apes_min,
        "apes_max": apes_max,
        "apes_frame_rms_delta": apes_frame_rms_delta,
        "overlap_identity_error": overlap_identity_error,
        "overlap_hermiticity_error": overlap_hermiticity_error,
        "overlap_eigenvalue_min": overlap_eigenvalue_min,
        "overlap_eigenvalue_max": overlap_eigenvalue_max,
        "solvent_coordinate": solvent_coordinate,
        "solvent_coordinate_velocity": solvent_coordinate_velocity,
        "hamiltonian_hermiticity_error": hamiltonian_hermiticity_error,
        "hamiltonian_trace": hamiltonian_trace,
        "finite": bool(finite),
    }


def embedded_ldr_frame_overlap_diagnostics(snapshots, times=None):
    """Return electronic overlap diagnostics between consecutive liquid frames."""

    snapshots = _validate_embedded_snapshots(snapshots)
    if len(snapshots) < 2:
        raise ValueError("at least two embedded snapshots are required.")
    times_array = None
    if times is not None:
        times_array = np.asarray(times, dtype=float)
        if times_array.shape != (len(snapshots),):
            raise ValueError(f"times shape {times_array.shape} != {(len(snapshots),)}.")
        if np.any(np.diff(times_array) <= 0.0):
            raise ValueError("times must be strictly increasing.")

    ngrid, nstates = snapshots[0].apes.shape
    identity = np.eye(nstates, dtype=complex)
    overlap_sequence = np.zeros((len(snapshots) - 1, ngrid, nstates, nstates), dtype=complex)
    unitary_transport_sequence = np.zeros_like(overlap_sequence)
    deviation = np.zeros((len(snapshots) - 1, ngrid), dtype=float)
    unitarity_error = np.zeros((len(snapshots) - 1, ngrid), dtype=float)
    mixing_norm = np.zeros((len(snapshots) - 1, ngrid), dtype=float)
    diagonal_abs_min = np.zeros((len(snapshots) - 1, ngrid), dtype=float)
    diagonal_abs_max = np.zeros((len(snapshots) - 1, ngrid), dtype=float)
    singular_value_min = np.zeros((len(snapshots) - 1, ngrid), dtype=float)
    singular_value_max = np.zeros((len(snapshots) - 1, ngrid), dtype=float)
    polar_residual = np.zeros((len(snapshots) - 1, ngrid), dtype=float)
    unitary_transport_deviation = np.zeros((len(snapshots) - 1, ngrid), dtype=float)
    unitary_transport_unitarity_error = np.zeros((len(snapshots) - 1, ngrid), dtype=float)
    unitary_transport_mixing_norm = np.zeros((len(snapshots) - 1, ngrid), dtype=float)
    phase_invariant_deviation = np.zeros((len(snapshots) - 1, ngrid), dtype=float)
    phase_invariant_mixing_norm = np.zeros((len(snapshots) - 1, ngrid), dtype=float)
    phase_aligned_unitary_transport_sequence = np.zeros_like(overlap_sequence)

    for step, (left_snapshot, right_snapshot) in enumerate(zip(snapshots[:-1], snapshots[1:])):
        left_objects = left_snapshot.electronic_objects
        right_objects = right_snapshot.electronic_objects
        if left_objects is None or right_objects is None:
            raise ValueError("frame-overlap diagnostics require snapshots with electronic_objects.")
        if len(left_objects) != ngrid or len(right_objects) != ngrid:
            raise ValueError("electronic_objects length must match the LDR coordinate grid.")
        for grid_index, (left, right) in enumerate(zip(left_objects, right_objects)):
            block = _electronic_overlap_block(left, right, nstates, diagonal=False)
            if block.shape[0] < nstates or block.shape[1] < nstates:
                raise ValueError(f"frame overlap block shape {block.shape} does not contain {nstates} states.")
            block = block[:nstates, :nstates]
            transport, singular_values = _closest_unitary(block)
            overlap_sequence[step, grid_index] = block
            unitary_transport_sequence[step, grid_index] = transport
            deviation[step, grid_index] = float(np.linalg.norm(block - identity))
            unitarity_error[step, grid_index] = float(np.linalg.norm(block.conj().T @ block - identity))
            offdiagonal = block.copy()
            offdiagonal[np.diag_indices(nstates)] = 0.0
            mixing_norm[step, grid_index] = float(np.linalg.norm(offdiagonal))
            diagonal_abs = np.abs(np.diag(block))
            diagonal_abs_min[step, grid_index] = float(np.min(diagonal_abs))
            diagonal_abs_max[step, grid_index] = float(np.max(diagonal_abs))
            singular_value_min[step, grid_index] = float(np.min(singular_values))
            singular_value_max[step, grid_index] = float(np.max(singular_values))
            polar_residual[step, grid_index] = float(np.linalg.norm(block - transport))
            unitary_transport_deviation[step, grid_index] = float(np.linalg.norm(transport - identity))
            unitary_transport_unitarity_error[step, grid_index] = float(
                np.linalg.norm(transport.conj().T @ transport - identity)
            )
            transport_offdiagonal = transport.copy()
            transport_offdiagonal[np.diag_indices(nstates)] = 0.0
            unitary_transport_mixing_norm[step, grid_index] = float(np.linalg.norm(transport_offdiagonal))
            phase_aligned = _phase_align_unitary_transport(transport)
            phase_aligned_unitary_transport_sequence[step, grid_index] = phase_aligned
            phase_invariant_deviation[step, grid_index] = float(np.linalg.norm(phase_aligned - identity))
            phase_aligned_offdiagonal = phase_aligned.copy()
            phase_aligned_offdiagonal[np.diag_indices(nstates)] = 0.0
            phase_invariant_mixing_norm[step, grid_index] = float(np.linalg.norm(phase_aligned_offdiagonal))

    frame_overlap_speed = np.full_like(deviation, np.nan)
    unitary_transport_speed = np.full_like(unitary_transport_deviation, np.nan)
    if times_array is not None:
        dt = np.diff(times_array)
        frame_overlap_speed = deviation / dt[:, None]
        unitary_transport_speed = unitary_transport_deviation / dt[:, None]

    return {
        "overlap_sequence": overlap_sequence,
        "unitary_transport_sequence": unitary_transport_sequence,
        "phase_aligned_unitary_transport_sequence": phase_aligned_unitary_transport_sequence,
        "deviation": deviation,
        "unitarity_error": unitarity_error,
        "mixing_norm": mixing_norm,
        "diagonal_abs_min": diagonal_abs_min,
        "diagonal_abs_max": diagonal_abs_max,
        "singular_value_min": singular_value_min,
        "singular_value_max": singular_value_max,
        "polar_residual": polar_residual,
        "unitary_transport_deviation": unitary_transport_deviation,
        "unitary_transport_unitarity_error": unitary_transport_unitarity_error,
        "unitary_transport_mixing_norm": unitary_transport_mixing_norm,
        "phase_invariant_deviation": phase_invariant_deviation,
        "phase_invariant_mixing_norm": phase_invariant_mixing_norm,
        "frame_overlap_speed": frame_overlap_speed,
        "unitary_transport_speed": unitary_transport_speed,
        "deviation_max": float(np.max(deviation)),
        "mixing_norm_max": float(np.max(mixing_norm)),
        "unitarity_error_max": float(np.max(unitarity_error)),
        "diagonal_abs_min_global": float(np.min(diagonal_abs_min)),
        "diagonal_abs_max_global": float(np.max(diagonal_abs_max)),
        "singular_value_min_global": float(np.min(singular_value_min)),
        "singular_value_max_global": float(np.max(singular_value_max)),
        "polar_residual_max": float(np.max(polar_residual)),
        "unitary_transport_deviation_max": float(np.max(unitary_transport_deviation)),
        "unitary_transport_mixing_norm_max": float(np.max(unitary_transport_mixing_norm)),
        "unitary_transport_unitarity_error_max": float(np.max(unitary_transport_unitarity_error)),
        "phase_invariant_deviation_max": float(np.max(phase_invariant_deviation)),
        "phase_invariant_mixing_norm_max": float(np.max(phase_invariant_mixing_norm)),
    }


def embedded_ldr_transport_holonomy(frame_overlap_diagnostics, *, transport="unitary"):
    """Accumulate electronic frame transport along a liquid trajectory.

    The returned cumulative transport maps electronic coefficients from the
    current liquid frame back to the first frame for each LDR coordinate-grid
    point.  For coefficient propagation from the first frame to the current
    frame, use the conjugate transpose of the cumulative block.
    """

    transport = _normalize_frame_transport(transport)
    if transport is None:
        transport = "unitary"
    sequence = np.asarray(_frame_overlap_transport_array(frame_overlap_diagnostics, transport), dtype=complex)
    if sequence.ndim != 4 or sequence.shape[2] != sequence.shape[3]:
        raise ValueError("transport sequence must have shape (nsteps, ngrid, nstates, nstates).")

    nsteps, ngrid, nstates, _ = sequence.shape
    identity = np.eye(nstates, dtype=complex)
    cumulative = np.zeros((nsteps + 1, ngrid, nstates, nstates), dtype=complex)
    cumulative[0] = identity
    for step in range(nsteps):
        cumulative[step + 1] = np.einsum("gab,gbc->gac", cumulative[step], sequence[step], optimize=True)

    unitarity_error = np.zeros((nsteps + 1, ngrid), dtype=float)
    deviation = np.zeros((nsteps + 1, ngrid), dtype=float)
    mixing_norm = np.zeros((nsteps + 1, ngrid), dtype=float)
    eigenphase = np.zeros((nsteps + 1, ngrid, nstates), dtype=float)
    for step in range(nsteps + 1):
        for grid_index in range(ngrid):
            block = cumulative[step, grid_index]
            unitarity_error[step, grid_index] = float(np.linalg.norm(block.conj().T @ block - identity))
            deviation[step, grid_index] = float(np.linalg.norm(block - identity))
            offdiagonal = block.copy()
            offdiagonal[np.diag_indices(nstates)] = 0.0
            mixing_norm[step, grid_index] = float(np.linalg.norm(offdiagonal))
            eigenphase[step, grid_index] = np.sort(np.angle(np.linalg.eigvals(block)))

    return {
        "transport": transport,
        "cumulative_transport": cumulative,
        "unitarity_error": unitarity_error,
        "deviation": deviation,
        "mixing_norm": mixing_norm,
        "eigenphase": eigenphase,
        "final_transport": cumulative[-1],
        "final_eigenphase": eigenphase[-1],
        "unitarity_error_max": float(np.max(unitarity_error)),
        "deviation_max": float(np.max(deviation)),
        "mixing_norm_max": float(np.max(mixing_norm)),
        "final_deviation_max": float(np.max(deviation[-1])),
        "final_mixing_norm_max": float(np.max(mixing_norm[-1])),
        "final_eigenphase_abs_max": float(np.max(np.abs(eigenphase[-1]))),
    }


def embedded_ldr_geometric_hotspots(
    frame_overlap_diagnostics,
    times=None,
    *,
    coordinate_grid=None,
    frame_indices=None,
    source_frame_indices=None,
    top_k=5,
):
    """Rank liquid-frame intervals by embedded-LDR geometric activity."""

    unitary_deviation = np.asarray(frame_overlap_diagnostics["unitary_transport_deviation"], dtype=float)
    unitary_mixing = np.asarray(frame_overlap_diagnostics["unitary_transport_mixing_norm"], dtype=float)
    phase_deviation = np.asarray(
        frame_overlap_diagnostics.get("phase_invariant_deviation", unitary_deviation),
        dtype=float,
    )
    phase_mixing = np.asarray(
        frame_overlap_diagnostics.get("phase_invariant_mixing_norm", unitary_mixing),
        dtype=float,
    )
    raw_deviation = np.asarray(frame_overlap_diagnostics["deviation"], dtype=float)
    unitarity_error = np.asarray(frame_overlap_diagnostics["unitarity_error"], dtype=float)
    polar_residual = np.asarray(frame_overlap_diagnostics["polar_residual"], dtype=float)
    singular_min = np.asarray(frame_overlap_diagnostics["singular_value_min"], dtype=float)
    singular_max = np.asarray(frame_overlap_diagnostics["singular_value_max"], dtype=float)
    if unitary_deviation.ndim != 2:
        raise ValueError("frame-overlap diagnostics must contain step-by-grid arrays.")
    nsteps, ngrid = unitary_deviation.shape
    expected = (nsteps, ngrid)
    for name, values in {
        "unitary_transport_mixing_norm": unitary_mixing,
        "phase_invariant_deviation": phase_deviation,
        "phase_invariant_mixing_norm": phase_mixing,
        "deviation": raw_deviation,
        "unitarity_error": unitarity_error,
        "polar_residual": polar_residual,
        "singular_value_min": singular_min,
        "singular_value_max": singular_max,
    }.items():
        if values.shape != expected:
            raise ValueError(f"{name} shape {values.shape} != {expected}.")

    times_array = None
    if times is not None:
        times_array = np.asarray(times, dtype=float)
        if times_array.shape != (nsteps + 1,):
            raise ValueError(f"times shape {times_array.shape} != {(nsteps + 1,)}.")
        if np.any(np.diff(times_array) <= 0.0):
            raise ValueError("times must be strictly increasing.")
    coordinate_grid = None if coordinate_grid is None else np.asarray(coordinate_grid, dtype=float)
    if coordinate_grid is not None and coordinate_grid.shape != (ngrid,):
        raise ValueError(f"coordinate_grid shape {coordinate_grid.shape} != {(ngrid,)}.")
    frame_indices = None if frame_indices is None else np.asarray(frame_indices, dtype=int)
    if frame_indices is not None and frame_indices.shape != (nsteps + 1,):
        raise ValueError(f"frame_indices shape {frame_indices.shape} != {(nsteps + 1,)}.")
    source_frame_indices = (
        None if source_frame_indices is None else np.asarray(source_frame_indices, dtype=int)
    )
    if source_frame_indices is not None and source_frame_indices.shape != (nsteps + 1,):
        raise ValueError(f"source_frame_indices shape {source_frame_indices.shape} != {(nsteps + 1,)}.")

    singular_leakage = np.maximum(np.abs(1.0 - singular_min), np.abs(singular_max - 1.0))
    leakage = np.maximum(polar_residual, unitarity_error)
    geometric_score = np.max(phase_deviation, axis=1)
    mixing_score = np.max(phase_mixing, axis=1)
    raw_score = np.max(raw_deviation, axis=1)
    leakage_score = np.max(leakage, axis=1)
    singular_leakage_score = np.max(singular_leakage, axis=1)
    score = geometric_score + leakage_score
    dominant_source = np.asarray(
        [
            _geometric_hotspot_source(geometric, leakage_value)
            for geometric, leakage_value in zip(geometric_score, leakage_score)
        ],
        dtype=object,
    )
    grid_index = np.argmax(phase_deviation + leakage, axis=1).astype(int)
    order = np.lexsort((np.arange(nsteps), -score))
    top_indices = order[: min(int(top_k), nsteps)]

    records = []
    for step in top_indices:
        grid = int(grid_index[step])
        record = {
            "step": int(step),
            "grid_index": grid,
            "score": float(score[step]),
            "geometric_score": float(geometric_score[step]),
            "mixing_score": float(mixing_score[step]),
            "raw_score": float(raw_score[step]),
            "leakage_score": float(leakage_score[step]),
            "singular_leakage_score": float(singular_leakage_score[step]),
            "dominant_source": str(dominant_source[step]),
            "unitary_transport_deviation": float(unitary_deviation[step, grid]),
            "unitary_transport_mixing_norm": float(unitary_mixing[step, grid]),
            "phase_invariant_deviation": float(phase_deviation[step, grid]),
            "phase_invariant_mixing_norm": float(phase_mixing[step, grid]),
            "raw_deviation": float(raw_deviation[step, grid]),
            "unitarity_error": float(unitarity_error[step, grid]),
            "polar_residual": float(polar_residual[step, grid]),
            "singular_value_min": float(singular_min[step, grid]),
            "singular_value_max": float(singular_max[step, grid]),
        }
        if coordinate_grid is not None:
            record["coordinate"] = float(coordinate_grid[grid])
        if frame_indices is not None:
            record.update(
                {
                    "frame_start": int(frame_indices[step]),
                    "frame_end": int(frame_indices[step + 1]),
                }
            )
        if source_frame_indices is not None:
            record.update(
                {
                    "source_frame_start": int(source_frame_indices[step]),
                    "source_frame_end": int(source_frame_indices[step + 1]),
                }
            )
        if times_array is not None:
            record.update(
                {
                    "time_start": float(times_array[step]),
                    "time_end": float(times_array[step + 1]),
                    "time_mid": float(0.5 * (times_array[step] + times_array[step + 1])),
                    "dt": float(times_array[step + 1] - times_array[step]),
                }
            )
        records.append(record)

    return {
        "score": score,
        "geometric_score": geometric_score,
        "mixing_score": mixing_score,
        "raw_score": raw_score,
        "leakage_score": leakage_score,
        "singular_leakage_score": singular_leakage_score,
        "dominant_source": dominant_source,
        "grid_index": grid_index,
        "top_indices": top_indices.astype(int),
        "coordinate_grid": None if coordinate_grid is None else coordinate_grid.copy(),
        "frame_indices": None if frame_indices is None else frame_indices.copy(),
        "source_frame_indices": None if source_frame_indices is None else source_frame_indices.copy(),
        "records": records,
        "score_max": float(np.max(score)),
        "geometric_score_max": float(np.max(geometric_score)),
        "mixing_score_max": float(np.max(mixing_score)),
        "leakage_score_max": float(np.max(leakage_score)),
        "singular_leakage_score_max": float(np.max(singular_leakage_score)),
    }


def embedded_ldr_geometric_quality(
    frame_overlap_diagnostics,
    hotspots=None,
    *,
    leakage_tolerance=1.0e-2,
    geometric_tolerance=1.0e-6,
):
    """Assess whether embedded liquid-frame geometry is interpretable."""

    leakage_tolerance = float(leakage_tolerance)
    geometric_tolerance = float(geometric_tolerance)
    if leakage_tolerance < 0.0:
        raise ValueError("leakage_tolerance must be non-negative.")
    if geometric_tolerance < 0.0:
        raise ValueError("geometric_tolerance must be non-negative.")
    if hotspots is None:
        hotspots = embedded_ldr_geometric_hotspots(frame_overlap_diagnostics)

    leakage_max = float(hotspots["leakage_score_max"])
    geometric_max = float(hotspots["geometric_score_max"])
    mixing_max = float(hotspots["mixing_score_max"])
    top_source = "none"
    top_record = None
    if hotspots["records"]:
        top_record = hotspots["records"][0]
        top_source = top_record["dominant_source"]

    subspace_unitary = leakage_max <= leakage_tolerance
    geometry_visible = geometric_max > geometric_tolerance
    if not subspace_unitary:
        verdict = "leakage_limited"
        recommendation = "increase embedded states or active space before interpreting geometric transport"
    elif geometry_visible:
        verdict = "ready"
        recommendation = "geometric transport is resolved within the retained state subspace"
    else:
        verdict = "geometry_quiet"
        recommendation = "no appreciable unitary geometric transport in this sampled liquid path"

    return {
        "verdict": verdict,
        "recommendation": recommendation,
        "subspace_unitary": bool(subspace_unitary),
        "geometry_visible": bool(geometry_visible),
        "leakage_tolerance": leakage_tolerance,
        "geometric_tolerance": geometric_tolerance,
        "leakage_score_max": leakage_max,
        "geometric_score_max": geometric_max,
        "mixing_score_max": mixing_max,
        "top_source": top_source,
        "top_record": top_record,
    }


def embedded_ldr_geometric_signal_summary(
    frame_overlap_diagnostics,
    *,
    leakage_tolerance=1.0e-2,
    geometric_tolerance=1.0e-6,
):
    """Summarize phase-aligned geometric signal over all frame/grid blocks."""

    leakage_tolerance = float(leakage_tolerance)
    geometric_tolerance = float(geometric_tolerance)
    if leakage_tolerance < 0.0:
        raise ValueError("leakage_tolerance must be non-negative.")
    if geometric_tolerance < 0.0:
        raise ValueError("geometric_tolerance must be non-negative.")

    fallback_deviation = np.asarray(frame_overlap_diagnostics["unitary_transport_deviation"], dtype=float)
    fallback_mixing = np.asarray(frame_overlap_diagnostics["unitary_transport_mixing_norm"], dtype=float)
    deviation = np.asarray(
        frame_overlap_diagnostics.get("phase_invariant_deviation", fallback_deviation),
        dtype=float,
    )
    mixing = np.asarray(
        frame_overlap_diagnostics.get("phase_invariant_mixing_norm", fallback_mixing),
        dtype=float,
    )
    unitarity_error = np.asarray(frame_overlap_diagnostics["unitarity_error"], dtype=float)
    polar_residual = np.asarray(frame_overlap_diagnostics["polar_residual"], dtype=float)
    singular_min = np.asarray(frame_overlap_diagnostics["singular_value_min"], dtype=float)
    singular_max = np.asarray(frame_overlap_diagnostics["singular_value_max"], dtype=float)
    expected = deviation.shape
    if deviation.ndim != 2:
        raise ValueError("geometric signal arrays must have shape (nsteps, ngrid).")
    for name, values in {
        "phase_invariant_mixing_norm": mixing,
        "unitarity_error": unitarity_error,
        "polar_residual": polar_residual,
        "singular_value_min": singular_min,
        "singular_value_max": singular_max,
    }.items():
        if values.shape != expected:
            raise ValueError(f"{name} shape {values.shape} != {expected}.")

    leakage = np.maximum(unitarity_error, polar_residual)
    singular_leakage = np.maximum(np.abs(1.0 - singular_min), np.abs(singular_max - 1.0))
    visible = deviation > geometric_tolerance
    subspace_unitary = leakage <= leakage_tolerance
    interpretable_visible = visible & subspace_unitary
    total = int(deviation.size)
    step_visible = np.any(visible, axis=1)
    step_subspace_unitary = np.all(subspace_unitary, axis=1)
    step_interpretable_visible = np.any(interpretable_visible, axis=1)
    max_index = np.unravel_index(int(np.argmax(deviation)), deviation.shape)
    if np.any(interpretable_visible):
        interpretable_values = np.where(interpretable_visible, deviation, -np.inf)
        interpretable_max_index = np.unravel_index(int(np.argmax(interpretable_values)), deviation.shape)
        interpretable_max = float(deviation[interpretable_max_index])
    else:
        interpretable_max_index = None
        interpretable_max = 0.0

    return {
        "leakage_tolerance": leakage_tolerance,
        "geometric_tolerance": geometric_tolerance,
        "geometric_deviation_max": float(np.max(deviation)),
        "geometric_deviation_mean": float(np.mean(deviation)),
        "geometric_deviation_rms": float(np.sqrt(np.mean(deviation * deviation))),
        "geometric_mixing_max": float(np.max(mixing)),
        "geometric_mixing_mean": float(np.mean(mixing)),
        "geometric_mixing_rms": float(np.sqrt(np.mean(mixing * mixing))),
        "leakage_max": float(np.max(leakage)),
        "leakage_mean": float(np.mean(leakage)),
        "singular_leakage_max": float(np.max(singular_leakage)),
        "visible_count": int(np.count_nonzero(visible)),
        "visible_fraction": float(np.count_nonzero(visible) / total),
        "subspace_unitary_count": int(np.count_nonzero(subspace_unitary)),
        "subspace_unitary_fraction": float(np.count_nonzero(subspace_unitary) / total),
        "interpretable_visible_count": int(np.count_nonzero(interpretable_visible)),
        "interpretable_visible_fraction": float(np.count_nonzero(interpretable_visible) / total),
        "visible_step_count": int(np.count_nonzero(step_visible)),
        "visible_step_fraction": float(np.count_nonzero(step_visible) / deviation.shape[0]),
        "subspace_unitary_step_count": int(np.count_nonzero(step_subspace_unitary)),
        "subspace_unitary_step_fraction": float(np.count_nonzero(step_subspace_unitary) / deviation.shape[0]),
        "interpretable_visible_step_count": int(np.count_nonzero(step_interpretable_visible)),
        "interpretable_visible_step_fraction": float(
            np.count_nonzero(step_interpretable_visible) / deviation.shape[0]
        ),
        "max_geometric_step": int(max_index[0]),
        "max_geometric_grid_index": int(max_index[1]),
        "max_interpretable_geometric_step": None if interpretable_max_index is None else int(interpretable_max_index[0]),
        "max_interpretable_geometric_grid_index": (
            None if interpretable_max_index is None else int(interpretable_max_index[1])
        ),
        "max_interpretable_geometric_deviation": interpretable_max,
    }


def embedded_ldr_geometric_state_convergence(
    trajectories,
    *,
    labels=None,
    coordinate_grid=None,
    frame_indices=None,
    source_frame_indices=None,
    leakage_tolerance=1.0e-2,
    geometric_tolerance=1.0e-6,
    top_k=1,
):
    """Compare embedded-LDR geometric quality across retained state counts.

    Each trajectory must contain the same liquid-frame sequence but may retain a
    different number of electronic states.  This is a lightweight convergence
    gate for liquid-phase geometric dynamics: if leakage falls as more states
    are retained, the current calculation is state-space limited rather than a
    reliable geometric transport result.
    """

    if isinstance(trajectories, dict):
        items = list(trajectories.items())
    else:
        trajectories = tuple(trajectories)
        if labels is None:
            labels = [None] * len(trajectories)
        labels = tuple(labels)
        if len(labels) != len(trajectories):
            raise ValueError("labels length must match trajectories.")
        items = list(zip(labels, trajectories))
    if len(items) == 0:
        raise ValueError("at least one trajectory is required.")

    records = []
    for label, trajectory in items:
        if isinstance(trajectory, SolventEmbeddedLDRTrajectory):
            snapshots = trajectory.snapshots
            times = trajectory.times
            grid = trajectory.bond_grid if coordinate_grid is None else coordinate_grid
        else:
            snapshots = _validate_embedded_snapshots(trajectory)
            times = None
            grid = coordinate_grid
        snapshots = _validate_embedded_snapshots(snapshots)
        nstates = int(snapshots[0].apes.shape[1])
        label = f"nstates={nstates}" if label is None else str(label)
        frame_overlap = embedded_ldr_frame_overlap_diagnostics(snapshots, times)
        hotspots = embedded_ldr_geometric_hotspots(
            frame_overlap,
            times=times,
            coordinate_grid=grid,
            frame_indices=frame_indices,
            source_frame_indices=source_frame_indices,
            top_k=top_k,
        )
        quality = embedded_ldr_geometric_quality(
            frame_overlap,
            hotspots,
            leakage_tolerance=leakage_tolerance,
            geometric_tolerance=geometric_tolerance,
        )
        records.append(
            {
                "label": label,
                "nstates": nstates,
                "verdict": quality["verdict"],
                "recommendation": quality["recommendation"],
                "subspace_unitary": bool(quality["subspace_unitary"]),
                "geometry_visible": bool(quality["geometry_visible"]),
                "leakage_score_max": float(quality["leakage_score_max"]),
                "geometric_score_max": float(quality["geometric_score_max"]),
                "mixing_score_max": float(quality["mixing_score_max"]),
                "top_source": quality["top_source"],
                "top_record": quality["top_record"],
            }
        )

    ordered = sorted(records, key=lambda record: (record["nstates"], record["label"]))
    ready = [record for record in ordered if record["verdict"] == "ready"]
    if ready:
        recommended = ready[0]
    else:
        recommended = min(
            ordered,
            key=lambda record: (
                record["leakage_score_max"],
                -record["geometric_score_max"],
                record["nstates"],
            ),
        )
    leakage = np.asarray([record["leakage_score_max"] for record in ordered], dtype=float)
    geometric = np.asarray([record["geometric_score_max"] for record in ordered], dtype=float)
    mixing = np.asarray([record["mixing_score_max"] for record in ordered], dtype=float)
    nstates = np.asarray([record["nstates"] for record in ordered], dtype=int)
    leakage_monotonic = True
    if leakage.size > 1:
        leakage_monotonic = bool(np.all(np.diff(leakage) <= 1.0e-12 * np.maximum(1.0, leakage[:-1])))

    return {
        "records": records,
        "ordered_records": ordered,
        "recommended_label": recommended["label"],
        "recommended_nstates": int(recommended["nstates"]),
        "recommended_verdict": recommended["verdict"],
        "all_ready": bool(all(record["verdict"] == "ready" for record in records)),
        "any_ready": bool(len(ready) > 0),
        "leakage_monotonic_nonincreasing": leakage_monotonic,
        "nstates": nstates,
        "leakage_score_max": leakage,
        "geometric_score_max": geometric,
        "mixing_score_max": mixing,
    }


def propagate_embedded_ldr_snapshots(
    snapshots,
    times,
    kinetic_x,
    *,
    initial_state=0,
    packet_center=None,
    packet_width=None,
    normalize=True,
    frame_transport=None,
    frame_overlap_diagnostics=None,
    substeps=1,
):
    """Propagate an LDR packet with a time-dependent embedded-LDR Hamiltonian."""

    snapshots = _validate_embedded_snapshots(snapshots)
    if len(snapshots) < 2:
        raise ValueError("at least two embedded snapshots are required.")
    times = np.asarray(times, dtype=float)
    if times.shape != (len(snapshots),):
        raise ValueError(f"times shape {times.shape} != {(len(snapshots),)}.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    substeps = int(substeps)
    if substeps < 1:
        raise ValueError("substeps must be a positive integer.")

    first = snapshots[0]
    bond_grid = np.asarray(first.bond_grid, dtype=float)
    nstates = int(first.apes.shape[1])
    frame_transport = _normalize_frame_transport(frame_transport)
    transport_sequence = None
    if frame_transport is not None:
        if frame_overlap_diagnostics is None:
            frame_overlap_diagnostics = embedded_ldr_frame_overlap_diagnostics(snapshots, times)
        transport_sequence = _frame_transport_sequence(
            frame_overlap_diagnostics,
            frame_transport,
            nsteps=len(snapshots) - 1,
            ngrid=bond_grid.size,
            nstates=nstates,
        )
    if packet_center is None:
        packet_center = float(bond_grid[len(bond_grid) // 2])
    if packet_width is None:
        span = float(np.max(bond_grid) - np.min(bond_grid))
        packet_width = max(span / 3.0, 1.0e-6)
    c = initial_ldr_packet(
        bond_grid,
        center=packet_center,
        width=packet_width,
        state=initial_state,
        nstates=nstates,
    )

    hamiltonians = [embedded_ldr_hamiltonian(snapshot, kinetic_x) for snapshot in snapshots]
    populations = [_populations(c)]
    norms = [float(np.vdot(c.ravel(), c.ravel()).real)]
    energies = [_expectation(hamiltonians[0], c)]

    from scipy.sparse.linalg import expm_multiply

    c_flat = c.reshape(-1)
    for left, right in zip(range(len(snapshots) - 1), range(1, len(snapshots))):
        dt = float(times[right] - times[left])
        transport_matrix = None
        right_hamiltonian = hamiltonians[right]
        if transport_sequence is not None:
            transport_matrix = _block_diagonal_frame_transport(transport_sequence[left])
            right_hamiltonian = transport_matrix @ right_hamiltonian @ transport_matrix.conj().T
        dt_sub = dt / substeps
        for substep in range(substeps):
            fraction_mid = (substep + 0.5) / substeps
            h_mid = (1.0 - fraction_mid) * hamiltonians[left] + fraction_mid * right_hamiltonian
            c_flat = expm_multiply((-1j * dt_sub) * h_mid, c_flat)
        if transport_matrix is not None:
            c_flat = transport_matrix.conj().T @ c_flat
        if normalize:
            norm = np.sqrt(np.vdot(c_flat, c_flat).real)
            if norm == 0.0:
                raise ValueError("LDR packet norm vanished during embedded propagation.")
            c_flat = c_flat / norm
        c = c_flat.reshape(bond_grid.size, nstates)
        populations.append(_populations(c))
        norms.append(float(np.vdot(c_flat, c_flat).real))
        energies.append(_expectation(hamiltonians[right], c))

    return {
        "times": times.copy(),
        "bond_grid": bond_grid.copy(),
        "populations": np.asarray(populations, dtype=float),
        "norm": np.asarray(norms, dtype=float),
        "energy": np.asarray(energies, dtype=float),
        "hamiltonian_trace": np.asarray([np.trace(h).real for h in hamiltonians], dtype=float),
        "frame_transport": "none" if frame_transport is None else frame_transport,
        "substeps": int(substeps),
    }


def compare_embedded_ldr_to_static(
    snapshots,
    times,
    kinetic_x,
    *,
    static_index=0,
    initial_state=0,
    packet_center=None,
    packet_width=None,
    normalize=True,
    frame_transport=None,
    frame_overlap_diagnostics=None,
    substeps=1,
):
    """Compare time-dependent embedded LDR propagation to a frozen snapshot."""

    snapshots = _validate_embedded_snapshots(snapshots)
    if len(snapshots) < 2:
        raise ValueError("at least two embedded snapshots are required.")
    static_index = int(static_index)
    if not 0 <= static_index < len(snapshots):
        raise ValueError("static_index must select one embedded snapshot.")
    common = {
        "initial_state": initial_state,
        "packet_center": packet_center,
        "packet_width": packet_width,
        "normalize": normalize,
        "substeps": substeps,
    }
    liquid = propagate_embedded_ldr_snapshots(
        snapshots,
        times,
        kinetic_x,
        frame_transport=frame_transport,
        frame_overlap_diagnostics=frame_overlap_diagnostics,
        **common,
    )
    static_snapshots = tuple(snapshots[static_index] for _ in snapshots)
    static = propagate_embedded_ldr_snapshots(static_snapshots, times, kinetic_x, **common)
    return {
        "liquid": liquid,
        "static": static,
        "population_delta": liquid["populations"] - static["populations"],
        "energy_delta": liquid["energy"] - static["energy"],
        "static_index": static_index,
    }


def compare_embedded_geometric_contribution(
    snapshots,
    times,
    kinetic_x,
    *,
    frame_transport="phase_aligned",
    initial_state=0,
    packet_center=None,
    packet_width=None,
    normalize=True,
    frame_overlap_diagnostics=None,
    substeps=1,
):
    """Compare embedded LDR propagation with frame geometry on and off.

    ``with_geometry`` propagates with the requested consecutive-frame
    electronic transport, while ``without_geometry`` uses the same
    time-dependent embedded Hamiltonians without basis transport.  This is the
    embedded analogue of :func:`compare_liquid_geometric_contribution`.
    """

    snapshots = _validate_embedded_snapshots(snapshots)
    if len(snapshots) < 2:
        raise ValueError("at least two embedded snapshots are required.")
    frame_transport = _normalize_frame_transport(frame_transport)
    if frame_transport is None:
        raise ValueError("frame_transport must enable geometry for this comparison.")
    common = {
        "initial_state": initial_state,
        "packet_center": packet_center,
        "packet_width": packet_width,
        "normalize": normalize,
        "substeps": substeps,
    }
    with_geometry = propagate_embedded_ldr_snapshots(
        snapshots,
        times,
        kinetic_x,
        frame_transport=frame_transport,
        frame_overlap_diagnostics=frame_overlap_diagnostics,
        **common,
    )
    without_geometry = propagate_embedded_ldr_snapshots(
        snapshots,
        times,
        kinetic_x,
        **common,
    )
    population_delta = with_geometry["populations"] - without_geometry["populations"]
    energy_delta = with_geometry["energy"] - without_geometry["energy"]
    return {
        "with_geometry": with_geometry,
        "without_geometry": without_geometry,
        "population_delta": population_delta,
        "energy_delta": energy_delta,
        "population_delta_max_abs": float(np.max(np.abs(population_delta))),
        "population_delta_rms": float(np.sqrt(np.mean(population_delta * population_delta))),
        "population_delta_final_norm": float(np.linalg.norm(population_delta[-1])),
        "energy_delta_max_abs": float(np.max(np.abs(energy_delta))),
        "energy_delta_final": float(energy_delta[-1]),
        "with_geometry_norm_max_error": float(np.max(np.abs(with_geometry["norm"] - 1.0))),
        "without_geometry_norm_max_error": float(np.max(np.abs(without_geometry["norm"] - 1.0))),
        "frame_transport": frame_transport,
    }


def embedded_ldr_geometric_step_diagnostics(
    geometric_control,
    times=None,
    *,
    frame_indices=None,
    source_frame_indices=None,
):
    """Return per-interval embedded geometric population-effect diagnostics."""

    result = liquid_ldr_geometric_step_diagnostics(geometric_control, times=times)
    nsteps = int(result["step_score"].size)
    frame_transport = geometric_control.get("frame_transport")
    if frame_transport is not None:
        result["frame_transport"] = frame_transport
    if frame_indices is not None:
        frame_indices = np.asarray(frame_indices, dtype=int)
        if frame_indices.shape != (nsteps + 1,):
            raise ValueError(f"frame_indices shape {frame_indices.shape} != {(nsteps + 1,)}.")
        result.update(
            {
                "frame_start": frame_indices[:-1],
                "frame_end": frame_indices[1:],
                "frame_indices": frame_indices.copy(),
            }
        )
    if source_frame_indices is not None:
        source_frame_indices = np.asarray(source_frame_indices, dtype=int)
        if source_frame_indices.shape != (nsteps + 1,):
            raise ValueError(
                f"source_frame_indices shape {source_frame_indices.shape} != {(nsteps + 1,)}."
            )
        result.update(
            {
                "source_frame_start": source_frame_indices[:-1],
                "source_frame_end": source_frame_indices[1:],
                "source_frame_indices": source_frame_indices.copy(),
            }
        )
    return result


def embedded_ldr_geometric_population_hotspots(
    geometric_control,
    times=None,
    *,
    frame_indices=None,
    source_frame_indices=None,
    top_k=5,
):
    """Rank embedded liquid-frame intervals by geometric population change."""

    if top_k < 1:
        return []
    step_diagnostics = embedded_ldr_geometric_step_diagnostics(
        geometric_control,
        times=times,
        frame_indices=frame_indices,
        source_frame_indices=source_frame_indices,
    )
    population_delta = np.asarray(geometric_control["population_delta"], dtype=float)
    if population_delta.ndim != 2:
        raise ValueError("population_delta must be a two-dimensional array.")
    if population_delta.shape[0] != int(step_diagnostics["step_score"].size) + 1:
        raise ValueError("population_delta must have one row per time sample.")

    records = []
    for step in range(step_diagnostics["step_score"].size):
        step_delta = step_diagnostics["population_delta_step"][step]
        dominant_state = int(step_diagnostics["dominant_state"][step])
        record = {
            "step": int(step),
            "dominant_state": dominant_state,
            "dominant_population_delta_step": float(
                step_diagnostics["dominant_population_delta_step"][step]
            ),
            "population_delta_step": step_delta.tolist(),
            "population_delta_start": population_delta[step].tolist(),
            "population_delta_end": population_delta[step + 1].tolist(),
            "score": float(step_diagnostics["step_score"][step]),
            "frame_transport": str(geometric_control.get("frame_transport", "unknown")),
        }
        for key in ("time_start", "time_end", "time_mid", "time_start_fs", "time_end_fs", "time_mid_fs"):
            if key in step_diagnostics:
                record[key] = float(step_diagnostics[key][step])
        for key in ("frame_start", "frame_end", "source_frame_start", "source_frame_end"):
            if key in step_diagnostics:
                record[key] = int(step_diagnostics[key][step])
        records.append(record)

    records.sort(key=lambda item: (-item["score"], item["step"]))
    return records[: int(top_k)]


def embedded_ldr_geometric_population_signal_summary(
    geometric_control,
    *,
    hotspots=None,
    geometric_tolerance=1.0e-8,
):
    """Summarize embedded frame-transport population signal strength."""

    summary = liquid_ldr_geometric_signal_summary(
        geometric_control,
        hotspots=hotspots,
        geometric_tolerance=geometric_tolerance,
    )
    summary["frame_transport"] = str(geometric_control.get("frame_transport", "unknown"))
    return summary


def embedded_ldr_geometric_population_quality(
    geometric_control,
    *,
    signal_summary=None,
    population_tolerance=1.0e-8,
    norm_tolerance=1.0e-10,
    min_steps=1,
):
    """Classify whether embedded frame-transport population effects are usable."""

    if signal_summary is None:
        signal_summary = embedded_ldr_geometric_population_signal_summary(
            geometric_control,
            geometric_tolerance=population_tolerance,
        )
    quality = liquid_ldr_geometric_quality(
        geometric_control,
        signal_summary=signal_summary,
        population_tolerance=population_tolerance,
        norm_tolerance=norm_tolerance,
        min_steps=min_steps,
    )
    recommendations = {
        "too_short": "Use at least two embedded liquid frames before interpreting frame-transport population effects.",
        "norm_limited": "Tighten embedded propagation accuracy or timestep before interpreting transported/untransported differences.",
        "geometry_quiet": "Embedded transported/untransported population differences are below the requested visibility tolerance.",
        "ready": "Embedded liquid LDR geometric population signal is visible and norm-stable for this trajectory.",
    }
    quality["recommendation"] = recommendations.get(quality["verdict"], quality["recommendation"])
    quality["frame_transport"] = str(geometric_control.get("frame_transport", "unknown"))
    return quality


def embedded_ldr_geometric_population_stride_convergence(
    snapshots,
    times,
    kinetic_x,
    strides,
    *,
    frame_transport="phase_aligned",
    initial_state=0,
    packet_center=None,
    packet_width=None,
    normalize=True,
    substeps=1,
    source_frame_indices=None,
    population_tolerance=1.0e-8,
    norm_tolerance=1.0e-10,
    min_steps=1,
    top_k=1,
    population_retention_tolerance=0.5,
    path_length_retention_tolerance=0.5,
):
    """Check embedded geometric population diagnostics under frame downsampling."""

    snapshots = _validate_embedded_snapshots(snapshots)
    times = np.asarray(times, dtype=float)
    if times.shape != (len(snapshots),):
        raise ValueError(f"times shape {times.shape} != {(len(snapshots),)}.")
    if len(snapshots) < 2:
        raise ValueError("at least two embedded snapshots are required.")
    strides = [int(stride) for stride in strides]
    if not strides:
        raise ValueError("at least one stride is required.")
    if any(stride <= 0 for stride in strides):
        raise ValueError("strides must be positive integers.")
    source_frame_indices = (
        np.arange(len(snapshots), dtype=int)
        if source_frame_indices is None
        else np.asarray(source_frame_indices, dtype=int)
    )
    if source_frame_indices.shape != (len(snapshots),):
        raise ValueError(f"source_frame_indices shape {source_frame_indices.shape} != {(len(snapshots),)}.")

    deduped_strides = []
    for stride in strides:
        if stride not in deduped_strides:
            deduped_strides.append(stride)

    records = []
    for stride in deduped_strides:
        indices = np.arange(0, len(snapshots), stride, dtype=int)
        if indices[-1] != len(snapshots) - 1:
            indices = np.concatenate((indices, np.asarray([len(snapshots) - 1], dtype=int)))
        sampled_snapshots = tuple(snapshots[int(index)] for index in indices)
        sampled_times = times[indices]
        sampled_source = source_frame_indices[indices]
        frame_overlap = embedded_ldr_frame_overlap_diagnostics(sampled_snapshots, sampled_times)
        control = compare_embedded_geometric_contribution(
            sampled_snapshots,
            sampled_times,
            kinetic_x,
            frame_transport=frame_transport,
            initial_state=initial_state,
            packet_center=packet_center,
            packet_width=packet_width,
            normalize=normalize,
            frame_overlap_diagnostics=frame_overlap,
            substeps=substeps,
        )
        hotspots = embedded_ldr_geometric_population_hotspots(
            control,
            times=sampled_times,
            frame_indices=indices,
            source_frame_indices=sampled_source,
            top_k=top_k,
        )
        signal = embedded_ldr_geometric_population_signal_summary(
            control,
            hotspots=hotspots,
            geometric_tolerance=population_tolerance,
        )
        quality = embedded_ldr_geometric_population_quality(
            control,
            signal_summary=signal,
            population_tolerance=population_tolerance,
            norm_tolerance=norm_tolerance,
            min_steps=min_steps,
        )
        top_hotspot = hotspots[0] if hotspots else None
        records.append(
            {
                "stride": int(stride),
                "indices": [int(index) for index in indices],
                "source_frame_indices": [int(index) for index in sampled_source],
                "sample_count": int(indices.size),
                "step_count": int(indices.size - 1),
                "time_start": float(sampled_times[0]),
                "time_end": float(sampled_times[-1]),
                "time_start_fs": float(sampled_times[0] * au2fs),
                "time_end_fs": float(sampled_times[-1] * au2fs),
                "population_delta_max_abs": float(signal["population_delta_max_abs"]),
                "population_delta_final_norm": float(signal["population_delta_final_norm"]),
                "population_delta_path_length": float(signal["population_delta_path_length"]),
                "top_step_score_fraction": float(signal["top_step_score_fraction"]),
                "top3_step_score_fraction": float(signal["top3_step_score_fraction"]),
                "effective_step_count": float(signal["effective_step_count"]),
                "norm_error_max": float(quality["norm_error_max"]),
                "quality_verdict": quality["verdict"],
                "quality_recommendation": quality["recommendation"],
                "geometry_visible": bool(quality["geometry_visible"]),
                "norm_stable": bool(quality["norm_stable"]),
                "enough_steps": bool(quality["enough_steps"]),
                "top_hotspot": top_hotspot,
                "top_hotspot_score": float(top_hotspot["score"]) if top_hotspot is not None else 0.0,
                "top_hotspot_step": int(top_hotspot["step"]) if top_hotspot is not None else None,
            }
        )

    baseline = records[0]
    baseline_population = float(baseline["population_delta_max_abs"])
    baseline_path_length = float(baseline["population_delta_path_length"])
    for record in records:
        record["population_delta_max_abs_relative_to_baseline"] = _safe_ratio(
            record["population_delta_max_abs"],
            baseline_population,
        )
        record["population_delta_path_length_relative_to_baseline"] = _safe_ratio(
            record["population_delta_path_length"],
            baseline_path_length,
        )

    ready_records = [record for record in records if record["quality_verdict"] == "ready"]
    retained_ready_records = [
        record
        for record in ready_records
        if _ratio_at_least(
            record["population_delta_max_abs_relative_to_baseline"],
            population_retention_tolerance,
        )
        and _ratio_at_least(
            record["population_delta_path_length_relative_to_baseline"],
            path_length_retention_tolerance,
        )
    ]
    recommended = retained_ready_records[-1] if retained_ready_records else (ready_records[0] if ready_records else None)
    return {
        "records": records,
        "baseline_stride": int(baseline["stride"]),
        "recommended_stride": None if recommended is None else int(recommended["stride"]),
        "any_ready": bool(ready_records),
        "all_ready": bool(len(ready_records) == len(records)),
        "frame_transport": _normalize_frame_transport(frame_transport),
        "population_retention_tolerance": float(population_retention_tolerance),
        "path_length_retention_tolerance": float(path_length_retention_tolerance),
        "population_tolerance": float(population_tolerance),
        "norm_tolerance": float(norm_tolerance),
        "min_steps": int(min_steps),
        "substeps": int(substeps),
    }


def embedded_ldr_geometric_readiness(
    frame_quality,
    *,
    population_quality=None,
    substep_convergence=None,
    stride_convergence=None,
    state_convergence=None,
    frame_step_convergence=None,
    fg_path_diagnostics=None,
):
    """Combine embedded frame-transport and population-effect readiness checks."""

    checks = []

    frame_ready = str(frame_quality.get("verdict", "unknown")) == "ready"
    checks.append(
        {
            "name": "frame_quality",
            "ready": bool(frame_ready),
            "detail": frame_quality.get("verdict", "unknown"),
            "recommendation": frame_quality.get(
                "recommendation",
                "Run embedded frame-overlap geometric quality diagnostics.",
            ),
        }
    )

    if population_quality is not None:
        population_ready = str(population_quality.get("verdict", "unknown")) == "ready"
        checks.append(
            {
                "name": "population_quality",
                "ready": bool(population_ready),
                "detail": population_quality.get("verdict", "unknown"),
                "recommendation": population_quality.get(
                    "recommendation",
                    "Run transported/untransported embedded population quality diagnostics.",
                ),
            }
        )

    if substep_convergence is not None:
        ready = bool(substep_convergence.get("recommended_ready", False))
        checks.append(
            {
                "name": "substeps",
                "ready": ready,
                "detail": f"recommended_substeps={substep_convergence.get('recommended_substeps')}",
                "recommendation": _ready_or_recommend(
                    ready,
                    "Use the recommended embedded LDR substeps for transported propagation.",
                ),
            }
        )

    if stride_convergence is not None:
        ready = bool(stride_convergence.get("any_ready", False))
        recommended_stride = stride_convergence.get("recommended_stride")
        checks.append(
            {
                "name": "stride",
                "ready": ready,
                "detail": f"recommended_stride={recommended_stride}",
                "recommendation": _ready_or_recommend(
                    ready,
                    "Use a smaller embedded trajectory stride before interpreting geometric population effects.",
                ),
            }
        )

    if state_convergence is not None:
        ready = bool(state_convergence.get("any_ready", False))
        checks.append(
            {
                "name": "state_convergence",
                "ready": ready,
                "detail": f"recommended_nstates={state_convergence.get('recommended_nstates')}",
                "recommendation": _ready_or_recommend(
                    ready,
                    "Increase embedded states or active space before interpreting geometric transport.",
                ),
            }
        )

    if frame_step_convergence is not None:
        ready = bool(frame_step_convergence.get("any_subspace_unitary", False))
        checks.append(
            {
                "name": "frame_step_convergence",
                "ready": ready,
                "detail": f"recommended_frame_step={frame_step_convergence.get('recommended_frame_step')}",
                "recommendation": _ready_or_recommend(
                    ready,
                    "Use a smaller liquid-frame stride or improve embedded state tracking.",
                ),
            }
        )

    if fg_path_diagnostics is not None:
        ready = bool(fg_path_diagnostics.get("fg_path_ready", False))
        checks.append(
            {
                "name": "fg_path",
                "ready": ready,
                "detail": (
                    f"coordinate_count={fg_path_diagnostics.get('coordinate_count')} "
                    f"max_width_scaled_displacement="
                    f"{fg_path_diagnostics.get('max_width_scaled_displacement')}"
                ),
                "recommendation": fg_path_diagnostics.get(
                    "recommendation",
                    "Reduce frame stride or check full-coordinate FG path construction.",
                ),
            }
        )

    failed = [check for check in checks if not check["ready"]]
    if failed:
        verdict = f"{failed[0]['name']}_limited"
        recommendation = failed[0]["recommendation"]
    else:
        verdict = "ready"
        recommendation = "Embedded liquid LDR geometric diagnostics are ready for the requested checks."

    return {
        "verdict": verdict,
        "ready": bool(not failed),
        "recommendation": recommendation,
        "checks": checks,
        "failed_checks": [check["name"] for check in failed],
    }


def embedded_ldr_comparison_metrics(comparison):
    """Return compact scalar metrics for an embedded liquid/static comparison."""

    population_delta = np.asarray(comparison["population_delta"], dtype=float)
    energy_delta = np.asarray(comparison["energy_delta"], dtype=float)
    liquid = comparison["liquid"]
    static = comparison["static"]
    liquid_norm = np.asarray(liquid["norm"], dtype=float)
    static_norm = np.asarray(static["norm"], dtype=float)
    return {
        "population_delta_max_abs": float(np.max(np.abs(population_delta))),
        "population_delta_rms": float(np.sqrt(np.mean(population_delta * population_delta))),
        "population_delta_final_norm": float(np.linalg.norm(population_delta[-1])),
        "energy_delta_max_abs": float(np.max(np.abs(energy_delta))),
        "energy_delta_final": float(energy_delta[-1]),
        "liquid_norm_max_error": float(np.max(np.abs(liquid_norm - 1.0))),
        "static_norm_max_error": float(np.max(np.abs(static_norm - 1.0))),
        "static_reference_frame": int(comparison["static_index"]),
    }


def second_derivative_kinetic(npts, dx, mass=1.0):
    """Return a simple central-difference kinetic matrix."""

    kinetic = np.diag(np.full(npts, 1.0 / (mass * dx * dx)))
    kinetic += np.diag(np.full(npts - 1, -0.5 / (mass * dx * dx)), k=1)
    kinetic += np.diag(np.full(npts - 1, -0.5 / (mass * dx * dx)), k=-1)
    return kinetic


def initial_ldr_packet(x_grid, *, center=-1.0, width=0.7, state=0, nstates=2):
    """Build a normalized LDR packet on one adiabatic state."""

    x_grid = np.asarray(x_grid, dtype=float)
    c = np.zeros((x_grid.size, int(nstates)), dtype=complex)
    envelope = np.exp(-0.5 * ((x_grid - float(center)) / float(width)) ** 2)
    norm = np.linalg.norm(envelope)
    if norm == 0.0:
        raise ValueError("initial packet has zero norm.")
    c[:, int(state)] = envelope / norm
    return c


def propagate_liquid_ldr(
    model,
    q_path,
    times,
    *,
    initial_state=0,
    initial_coefficients=None,
    packet_center=-1.0,
    packet_width=0.7,
    normalize=True,
    include_berry=True,
    substeps=1,
):
    """Propagate LDRFG coefficients along a prescribed liquid coordinate path."""

    q_path = np.asarray(q_path, dtype=float)
    times = np.asarray(times, dtype=float)
    if q_path.ndim != 1 or times.ndim != 1 or q_path.shape != times.shape:
        raise ValueError("q_path and times must be one-dimensional arrays with the same shape.")
    if q_path.size < 2:
        raise ValueError("at least two liquid-coordinate samples are required.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    substeps = int(substeps)
    if substeps < 1:
        raise ValueError("substeps must be a positive integer.")

    solver = model.solver(include_berry=include_berry)
    if initial_coefficients is None:
        c = initial_ldr_packet(
            model.x_grid,
            center=packet_center,
            width=packet_width,
            state=initial_state,
        )
    else:
        c = np.asarray(initial_coefficients, dtype=complex).copy()
        nstates = model.energies([q_path[0]]).shape[1]
        if c.shape != (model.x_grid.size, nstates):
            raise ValueError(f"initial_coefficients shape {c.shape} != {(model.x_grid.size, nstates)}.")
        norm = np.sqrt(np.vdot(c.ravel(), c.ravel()).real)
        if norm == 0.0:
            raise ValueError("initial_coefficients has zero norm.")
        if normalize:
            c = c / norm
    pops = [_populations(c)]
    norms = [float(np.vdot(c.ravel(), c.ravel()).real)]
    energies = [float(solver.energy(c, [q_path[0]], [0.0]).real)]

    for left, right in zip(range(q_path.size - 1), range(1, q_path.size)):
        dt = float(times[right] - times[left])
        q_dot = float((q_path[right] - q_path[left]) / dt)
        p_mid = np.array([model.mass_y * q_dot], dtype=float)
        dt_sub = dt / substeps
        for substep in range(substeps):
            fraction_mid = (substep + 0.5) / substeps
            q_mid = np.array([q_path[left] + fraction_mid * (q_path[right] - q_path[left])], dtype=float)
            c = solver.propagate_coefficients(c, q_mid, p_mid, dt_sub)
            if normalize:
                norm = np.sqrt(np.vdot(c.ravel(), c.ravel()).real)
                if norm == 0.0:
                    raise ValueError("LDR packet norm vanished during propagation.")
                c = c / norm
        pops.append(_populations(c))
        norms.append(float(np.vdot(c.ravel(), c.ravel()).real))
        energies.append(float(solver.energy(c, [q_path[right]], [p_mid[0]]).real))

    return {
        "times": times.copy(),
        "q": q_path.copy(),
        "populations": np.asarray(pops, dtype=float),
        "norm": np.asarray(norms, dtype=float),
        "energy": np.asarray(energies, dtype=float),
    }


def liquid_ldr_diagnostics(model, q_path, times):
    """Return solvent-path diagnostics for a liquid-driven LDRFG model."""

    q_path = np.asarray(q_path, dtype=float)
    times = np.asarray(times, dtype=float)
    if q_path.ndim != 1 or times.ndim != 1 or q_path.shape != times.shape:
        raise ValueError("q_path and times must be one-dimensional arrays with the same shape.")
    if q_path.size < 2:
        raise ValueError("at least two liquid-coordinate samples are required.")

    q_dot = np.gradient(q_path, times)
    gaps = []
    berry_norms = []
    for q_value in q_path:
        q = np.array([q_value], dtype=float)
        energies = np.asarray(model.energies(q), dtype=float)
        if energies.shape[1] < 2:
            gaps.append(0.0)
        else:
            gaps.append(float(np.min(energies[:, 1] - energies[:, 0])))
        berry = np.asarray(model.berry(q), dtype=complex).reshape(model.x_grid.size * 2, model.x_grid.size * 2)
        berry_norms.append(float(np.linalg.norm(berry)))

    berry_norms = np.asarray(berry_norms, dtype=float)
    geometric_speed = np.abs(q_dot) * berry_norms
    return {
        "q_dot": q_dot,
        "gap_min": np.asarray(gaps, dtype=float),
        "berry_norm": berry_norms,
        "geometric_speed": geometric_speed,
    }


def compare_liquid_to_static_ldr(
    model,
    q_path,
    times,
    *,
    q_static=None,
    initial_state=0,
    initial_coefficients=None,
    packet_center=-1.0,
    packet_width=0.7,
    substeps=1,
):
    """Propagate liquid-driven and static-bath LDRFG references side by side."""

    q_path = np.asarray(q_path, dtype=float)
    times = np.asarray(times, dtype=float)
    if q_static is None:
        q_static = float(np.mean(q_path))
    static_path = np.full_like(q_path, float(q_static), dtype=float)
    common = {
        "initial_state": initial_state,
        "initial_coefficients": initial_coefficients,
        "packet_center": packet_center,
        "packet_width": packet_width,
        "substeps": substeps,
    }
    liquid = propagate_liquid_ldr(model, q_path, times, **common)
    static = propagate_liquid_ldr(model, static_path, times, **common)
    diagnostics = liquid_ldr_diagnostics(model, q_path, times)
    return {
        "liquid": liquid,
        "static": static,
        "diagnostics": diagnostics,
        "population_delta": liquid["populations"] - static["populations"],
        "q_static": float(q_static),
    }


def compare_liquid_geometric_contribution(
    model,
    q_path,
    times,
    *,
    initial_state=0,
    initial_coefficients=None,
    packet_center=-1.0,
    packet_width=0.7,
    substeps=1,
):
    """Compare liquid-driven LDR propagation with Berry coupling on and off."""

    common = {
        "initial_state": initial_state,
        "initial_coefficients": initial_coefficients,
        "packet_center": packet_center,
        "packet_width": packet_width,
        "substeps": substeps,
    }
    with_geometry = propagate_liquid_ldr(model, q_path, times, include_berry=True, **common)
    without_geometry = propagate_liquid_ldr(model, q_path, times, include_berry=False, **common)
    population_delta = with_geometry["populations"] - without_geometry["populations"]
    energy_delta = with_geometry["energy"] - without_geometry["energy"]
    return {
        "with_geometry": with_geometry,
        "without_geometry": without_geometry,
        "population_delta": population_delta,
        "energy_delta": energy_delta,
        "population_delta_max_abs": float(np.max(np.abs(population_delta))),
        "population_delta_rms": float(np.sqrt(np.mean(population_delta * population_delta))),
        "population_delta_final_norm": float(np.linalg.norm(population_delta[-1])),
        "energy_delta_max_abs": float(np.max(np.abs(energy_delta))),
        "energy_delta_final": float(energy_delta[-1]),
        "with_geometry_norm_max_error": float(np.max(np.abs(with_geometry["norm"] - 1.0))),
        "without_geometry_norm_max_error": float(np.max(np.abs(without_geometry["norm"] - 1.0))),
    }


def liquid_ldr_geometric_step_diagnostics(
    geometric_control,
    times=None,
    q_path=None,
    path_diagnostics=None,
):
    """Return per-interval Berry/no-Berry population-delta diagnostics."""

    population_delta = np.asarray(geometric_control["population_delta"], dtype=float)
    if population_delta.ndim != 2:
        raise ValueError("population_delta must be a two-dimensional array.")
    if population_delta.shape[0] < 2:
        step_delta = np.zeros((0, population_delta.shape[1]), dtype=float)
    else:
        step_delta = np.diff(population_delta, axis=0)
    step_score = np.linalg.norm(step_delta, axis=1)
    if step_delta.shape[0]:
        dominant_state = np.argmax(np.abs(step_delta), axis=1).astype(int)
        dominant_step = step_delta[np.arange(step_delta.shape[0]), dominant_state]
    else:
        dominant_state = np.zeros(0, dtype=int)
        dominant_step = np.zeros(0, dtype=float)
    result = {
        "population_delta_step": step_delta,
        "step_score": step_score,
        "cumulative_path_length": np.concatenate(([0.0], np.cumsum(step_score))),
        "dominant_state": dominant_state,
        "dominant_population_delta_step": dominant_step,
    }
    if times is not None:
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.shape[0] != population_delta.shape[0]:
            raise ValueError("times must be one-dimensional with one value per population sample.")
        result.update(
            {
                "time_start": times[:-1],
                "time_end": times[1:],
                "time_mid": 0.5 * (times[:-1] + times[1:]),
                "time_start_fs": times[:-1] * au2fs,
                "time_end_fs": times[1:] * au2fs,
                "time_mid_fs": 0.5 * (times[:-1] + times[1:]) * au2fs,
            }
        )
    if q_path is not None:
        q_path = np.asarray(q_path, dtype=float)
        if q_path.ndim != 1 or q_path.shape[0] != population_delta.shape[0]:
            raise ValueError("q_path must be one-dimensional with one value per population sample.")
        result.update(
            {
                "q_start": q_path[:-1],
                "q_end": q_path[1:],
                "q_mid": 0.5 * (q_path[:-1] + q_path[1:]),
                "q_delta": np.diff(q_path),
                "abs_q_delta": np.abs(np.diff(q_path)),
            }
        )
    if path_diagnostics is not None:
        geometric_speed = np.asarray(path_diagnostics["geometric_speed"], dtype=float)
        gap_min = np.asarray(path_diagnostics["gap_min"], dtype=float)
        if geometric_speed.ndim != 1 or geometric_speed.shape[0] != population_delta.shape[0]:
            raise ValueError("path_diagnostics['geometric_speed'] must have one value per population sample.")
        if gap_min.ndim != 1 or gap_min.shape[0] != population_delta.shape[0]:
            raise ValueError("path_diagnostics['gap_min'] must have one value per population sample.")
        gap_min_mean = 0.5 * (gap_min[:-1] + gap_min[1:])
        result.update(
            {
                "geometric_speed_start": geometric_speed[:-1],
                "geometric_speed_end": geometric_speed[1:],
                "geometric_speed_mean": 0.5 * (geometric_speed[:-1] + geometric_speed[1:]),
                "gap_min_start": gap_min[:-1],
                "gap_min_end": gap_min[1:],
                "gap_min_mean": gap_min_mean,
                "inverse_gap_min_mean": np.divide(
                    1.0,
                    gap_min_mean,
                    out=np.full(gap_min_mean.shape, np.nan, dtype=float),
                    where=gap_min_mean != 0.0,
                ),
            }
        )
    return result


def liquid_ldr_geometric_driver_correlations(step_diagnostics):
    """Correlate Berry step scores with liquid-path step descriptors."""

    step_score = np.asarray(step_diagnostics["step_score"], dtype=float)
    correlations = {}
    drivers = {
        "abs_q_delta": step_diagnostics.get("abs_q_delta"),
        "geometric_speed_mean": step_diagnostics.get("geometric_speed_mean"),
        "gap_min_mean": step_diagnostics.get("gap_min_mean"),
    }
    if drivers["gap_min_mean"] is not None:
        gap = np.asarray(drivers["gap_min_mean"], dtype=float)
        drivers["inverse_gap_min_mean"] = np.divide(
            1.0,
            gap,
            out=np.full_like(gap, np.nan, dtype=float),
            where=gap != 0.0,
        )
    for label, values in drivers.items():
        if values is None:
            correlations[label] = None
        else:
            correlations[label] = _pearson_or_none(step_score, np.asarray(values, dtype=float))
    return correlations


def liquid_ldr_geometric_hotspots(
    model,
    q_path,
    times,
    *,
    geometric_control=None,
    diagnostics=None,
    top_k=5,
    initial_state=0,
    packet_center=-1.0,
    packet_width=0.7,
):
    """Rank liquid-trajectory intervals by Berry-induced LDR population change."""

    q_path = np.asarray(q_path, dtype=float)
    times = np.asarray(times, dtype=float)
    if q_path.ndim != 1 or times.ndim != 1 or q_path.shape != times.shape:
        raise ValueError("q_path and times must be one-dimensional arrays with the same shape.")
    if q_path.size < 2:
        raise ValueError("at least two liquid-coordinate samples are required.")
    if top_k < 1:
        return []
    if geometric_control is None:
        geometric_control = compare_liquid_geometric_contribution(
            model,
            q_path,
            times,
            initial_state=initial_state,
            packet_center=packet_center,
            packet_width=packet_width,
        )
    if diagnostics is None:
        diagnostics = liquid_ldr_diagnostics(model, q_path, times)

    step_diagnostics = liquid_ldr_geometric_step_diagnostics(
        geometric_control,
        times=times,
        q_path=q_path,
        path_diagnostics=diagnostics,
    )
    population_delta = np.asarray(geometric_control["population_delta"], dtype=float)
    geometric_speed = np.asarray(diagnostics["geometric_speed"], dtype=float)
    gap_min = np.asarray(diagnostics["gap_min"], dtype=float)
    if population_delta.shape[0] != q_path.size:
        raise ValueError("geometric_control population_delta must have one row per time sample.")
    if geometric_speed.shape != q_path.shape or gap_min.shape != q_path.shape:
        raise ValueError("diagnostics arrays must have one value per time sample.")

    driver_scores = _liquid_hotspot_driver_scores(step_diagnostics)
    records = []
    for step in range(q_path.size - 1):
        step_delta = step_diagnostics["population_delta_step"][step]
        dominant_state = int(step_diagnostics["dominant_state"][step])
        score = float(step_diagnostics["step_score"][step])
        step_driver_scores = {
            label: float(values[step])
            for label, values in driver_scores.items()
            if np.isfinite(values[step])
        }
        if step_driver_scores:
            dominant_driver = max(step_driver_scores, key=lambda label: step_driver_scores[label])
            dominant_driver_score = step_driver_scores[dominant_driver]
        else:
            dominant_driver = None
            dominant_driver_score = 0.0
        records.append(
            {
                "step": int(step),
                "time_start": float(step_diagnostics["time_start"][step]),
                "time_end": float(step_diagnostics["time_end"][step]),
                "time_mid": float(step_diagnostics["time_mid"][step]),
                "time_start_fs": float(step_diagnostics["time_start_fs"][step]),
                "time_end_fs": float(step_diagnostics["time_end_fs"][step]),
                "time_mid_fs": float(step_diagnostics["time_mid_fs"][step]),
                "q_start": float(step_diagnostics["q_start"][step]),
                "q_end": float(step_diagnostics["q_end"][step]),
                "q_mid": float(step_diagnostics["q_mid"][step]),
                "q_delta": float(step_diagnostics["q_delta"][step]),
                "abs_q_delta": float(step_diagnostics["abs_q_delta"][step]),
                "geometric_speed_start": float(step_diagnostics["geometric_speed_start"][step]),
                "geometric_speed_end": float(step_diagnostics["geometric_speed_end"][step]),
                "geometric_speed_mean": float(step_diagnostics["geometric_speed_mean"][step]),
                "gap_min_mean": float(step_diagnostics["gap_min_mean"][step]),
                "inverse_gap_min_mean": float(step_diagnostics["inverse_gap_min_mean"][step]),
                "driver_scores": step_driver_scores,
                "dominant_driver": dominant_driver,
                "dominant_driver_score": float(dominant_driver_score),
                "dominant_state": dominant_state,
                "dominant_population_delta_step": float(
                    step_diagnostics["dominant_population_delta_step"][step]
                ),
                "population_delta_step": step_delta.tolist(),
                "population_delta_start": population_delta[step].tolist(),
                "population_delta_end": population_delta[step + 1].tolist(),
                "score": score,
            }
        )

    records.sort(key=lambda item: (-item["score"], item["step"]))
    return records[: int(top_k)]


def _liquid_hotspot_driver_scores(step_diagnostics):
    drivers = {
        "abs_q_delta": step_diagnostics.get("abs_q_delta"),
        "geometric_speed_mean": step_diagnostics.get("geometric_speed_mean"),
        "inverse_gap_min_mean": step_diagnostics.get("inverse_gap_min_mean"),
    }
    scores = {}
    for label, values in drivers.items():
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        normalized = np.zeros(values.shape, dtype=float)
        if np.any(finite):
            max_abs = float(np.max(np.abs(values[finite])))
            if max_abs > 0.0:
                normalized[finite] = np.abs(values[finite]) / max_abs
        scores[label] = normalized
    return scores


def liquid_ldr_hotspot_driver_summary(hotspots):
    """Summarize which liquid drivers dominate ranked analytic LDR hot spots."""

    records = list(hotspots or [])
    labels = sorted(
        {
            str(label)
            for record in records
            for label in (record.get("driver_scores") or {}).keys()
        }
    )
    counts = {label: 0 for label in labels}
    score_sums = {label: 0.0 for label in labels}
    driver_score_sums = {label: 0.0 for label in labels}
    top_by_driver = {label: None for label in labels}

    for record in records:
        driver = record.get("dominant_driver")
        if driver is None:
            continue
        driver = str(driver)
        if driver not in counts:
            counts[driver] = 0
            score_sums[driver] = 0.0
            driver_score_sums[driver] = 0.0
            top_by_driver[driver] = None
            labels.append(driver)
        score = float(record.get("score", 0.0))
        driver_score = float(record.get("dominant_driver_score", 0.0))
        counts[driver] += 1
        score_sums[driver] += score
        driver_score_sums[driver] += driver_score
        current = top_by_driver[driver]
        if current is None or score > float(current.get("score", 0.0)):
            top_by_driver[driver] = record

    dominant_driver = None
    if score_sums:
        dominant_driver = max(score_sums, key=lambda label: (score_sums[label], counts[label], label))
        if score_sums[dominant_driver] <= 0.0 and counts[dominant_driver] == 0:
            dominant_driver = None
    total_score = float(sum(score_sums.values()))
    labels = sorted(counts)
    return {
        "hotspot_count": int(len(records)),
        "drivers": labels,
        "dominant_driver": dominant_driver,
        "dominant_driver_count": int(counts[dominant_driver]) if dominant_driver is not None else 0,
        "dominant_driver_score_sum": float(score_sums[dominant_driver])
        if dominant_driver is not None
        else 0.0,
        "score_sum": total_score,
        "count_by_driver": {label: int(counts[label]) for label in labels},
        "score_sum_by_driver": {label: float(score_sums[label]) for label in labels},
        "score_fraction_by_driver": {
            label: float(score_sums[label] / total_score) if total_score > 0.0 else 0.0
            for label in labels
        },
        "driver_score_sum_by_driver": {label: float(driver_score_sums[label]) for label in labels},
        "top_hotspot_by_driver": {
            label: top_by_driver[label] for label in labels if top_by_driver[label] is not None
        },
    }


def _pearson_or_none(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 2:
        return None
    left = left[finite]
    right = right[finite]
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denom = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denom == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denom)


def _step_score_localization(step_scores):
    step_scores = np.asarray(step_scores, dtype=float)
    finite_scores = step_scores[np.isfinite(step_scores)]
    if finite_scores.size == 0:
        return {
            "step_score_sum": 0.0,
            "top_step_index": None,
            "top_step_score_fraction": 0.0,
            "top3_step_score_fraction": 0.0,
            "top5_step_score_fraction": 0.0,
            "effective_step_count": 0.0,
        }
    total = float(np.sum(finite_scores))
    if total <= 0.0:
        return {
            "step_score_sum": total,
            "top_step_index": None,
            "top_step_score_fraction": 0.0,
            "top3_step_score_fraction": 0.0,
            "top5_step_score_fraction": 0.0,
            "effective_step_count": 0.0,
        }
    order = np.argsort(step_scores)[::-1]
    finite_order = [int(index) for index in order if np.isfinite(step_scores[index])]
    square_sum = float(np.sum(finite_scores * finite_scores))
    return {
        "step_score_sum": total,
        "top_step_index": finite_order[0] if finite_order else None,
        "top_step_score_fraction": float(step_scores[finite_order[0]] / total) if finite_order else 0.0,
        "top3_step_score_fraction": float(np.sum(step_scores[finite_order[:3]]) / total),
        "top5_step_score_fraction": float(np.sum(step_scores[finite_order[:5]]) / total),
        "effective_step_count": float(total * total / square_sum) if square_sum > 0.0 else 0.0,
    }


def liquid_ldr_geometric_signal_summary(
    geometric_control,
    *,
    hotspots=None,
    geometric_tolerance=1.0e-8,
):
    """Summarize Berry/no-Berry LDR signal strength across a liquid path."""

    population_delta = np.asarray(geometric_control["population_delta"], dtype=float)
    energy_delta = np.asarray(geometric_control["energy_delta"], dtype=float)
    if population_delta.ndim != 2:
        raise ValueError("population_delta must be a two-dimensional array.")
    if energy_delta.shape != (population_delta.shape[0],):
        raise ValueError("energy_delta must have one value per population sample.")
    step_diagnostics = liquid_ldr_geometric_step_diagnostics(geometric_control)
    step_scores = step_diagnostics["step_score"]
    localization = _step_score_localization(step_scores)

    sample_norm = np.linalg.norm(population_delta, axis=1)
    peak_flat = int(np.argmax(np.abs(population_delta)))
    peak_sample, peak_state = np.unravel_index(peak_flat, population_delta.shape)
    visible_mask = step_scores > float(geometric_tolerance)
    summary = {
        "sample_count": int(population_delta.shape[0]),
        "state_count": int(population_delta.shape[1]),
        "step_count": int(step_scores.size),
        "geometric_tolerance": float(geometric_tolerance),
        "population_delta_max_abs": float(np.max(np.abs(population_delta))),
        "population_delta_rms": float(np.sqrt(np.mean(population_delta * population_delta))),
        "population_delta_final_norm": float(sample_norm[-1]),
        "population_delta_sample_norm_sum": float(np.sum(sample_norm)),
        "population_delta_path_length": float(step_diagnostics["cumulative_path_length"][-1]),
        "population_delta_peak_sample": int(peak_sample),
        "population_delta_peak_state": int(peak_state),
        "population_delta_peak_value": float(population_delta[peak_sample, peak_state]),
        "energy_delta_max_abs": float(np.max(np.abs(energy_delta))),
        "energy_delta_final": float(energy_delta[-1]),
        "step_score_max": float(np.max(step_scores)) if step_scores.size else 0.0,
        "step_score_mean": float(np.mean(step_scores)) if step_scores.size else 0.0,
        "step_score_sum": localization["step_score_sum"],
        "top_step_index": localization["top_step_index"],
        "top_step_score_fraction": localization["top_step_score_fraction"],
        "top3_step_score_fraction": localization["top3_step_score_fraction"],
        "top5_step_score_fraction": localization["top5_step_score_fraction"],
        "effective_step_count": localization["effective_step_count"],
        "visible_step_count": int(np.count_nonzero(visible_mask)),
        "visible_step_fraction": float(np.count_nonzero(visible_mask) / step_scores.size)
        if step_scores.size
        else 0.0,
    }
    if hotspots:
        summary["hotspot_count"] = int(len(hotspots))
        summary["top_hotspot"] = hotspots[0]
        summary["top_hotspot_score"] = float(hotspots[0]["score"])
        summary["hotspot_driver_summary"] = liquid_ldr_hotspot_driver_summary(hotspots)
    else:
        summary["hotspot_count"] = 0
        summary["top_hotspot"] = None
        summary["top_hotspot_score"] = 0.0
        summary["hotspot_driver_summary"] = liquid_ldr_hotspot_driver_summary([])
    return summary


def liquid_ldr_geometric_quality(
    geometric_control,
    *,
    signal_summary=None,
    population_tolerance=1.0e-4,
    norm_tolerance=1.0e-10,
    min_steps=2,
):
    """Classify whether an analytic liquid LDR Berry signal is numerically usable."""

    if signal_summary is None:
        signal_summary = liquid_ldr_geometric_signal_summary(geometric_control)
    norm_error = max(
        float(geometric_control.get("with_geometry_norm_max_error", np.inf)),
        float(geometric_control.get("without_geometry_norm_max_error", np.inf)),
    )
    step_count = int(signal_summary.get("step_count", 0))
    signal_max = float(signal_summary.get("population_delta_max_abs", 0.0))
    geometry_visible = signal_max > float(population_tolerance)
    norm_stable = norm_error <= float(norm_tolerance)
    enough_steps = step_count >= int(min_steps)
    if not enough_steps:
        verdict = "too_short"
        recommendation = "Use at least two liquid-trajectory intervals before interpreting Berry signal quality."
    elif not norm_stable:
        verdict = "norm_limited"
        recommendation = "Tighten propagation accuracy or timestep before interpreting Berry/no-Berry differences."
    elif not geometry_visible:
        verdict = "geometry_quiet"
        recommendation = "Berry/no-Berry population differences are below the requested visibility tolerance."
    else:
        verdict = "ready"
        recommendation = "Analytic liquid LDR Berry signal is visible and norm-stable for this trajectory."
    return {
        "verdict": verdict,
        "recommendation": recommendation,
        "population_tolerance": float(population_tolerance),
        "norm_tolerance": float(norm_tolerance),
        "min_steps": int(min_steps),
        "norm_error_max": norm_error,
        "population_delta_max_abs": signal_max,
        "step_count": step_count,
        "geometry_visible": bool(geometry_visible),
        "norm_stable": bool(norm_stable),
        "enough_steps": bool(enough_steps),
        "effective_step_count": float(signal_summary.get("effective_step_count", 0.0)),
        "top_step_score_fraction": float(signal_summary.get("top_step_score_fraction", 0.0)),
    }


def liquid_ldr_geometric_stride_convergence(
    model,
    q_path,
    times,
    strides,
    *,
    initial_state=0,
    packet_center=-1.0,
    packet_width=0.7,
    population_tolerance=1.0e-4,
    norm_tolerance=1.0e-10,
    min_steps=2,
    top_k=1,
    population_retention_tolerance=0.5,
    path_length_retention_tolerance=0.5,
    substeps=1,
):
    """Check analytic liquid LDR Berry diagnostics under temporal downsampling."""

    q_path = np.asarray(q_path, dtype=float)
    times = np.asarray(times, dtype=float)
    if q_path.ndim != 1 or times.ndim != 1 or q_path.shape != times.shape:
        raise ValueError("q_path and times must be one-dimensional arrays with the same shape.")
    if q_path.size < 2:
        raise ValueError("at least two liquid-coordinate samples are required.")
    strides = [int(stride) for stride in strides]
    if not strides:
        raise ValueError("at least one stride is required.")
    if any(stride <= 0 for stride in strides):
        raise ValueError("strides must be positive integers.")

    deduped_strides = []
    for stride in strides:
        if stride not in deduped_strides:
            deduped_strides.append(stride)

    records = []
    for stride in deduped_strides:
        indices = np.arange(0, q_path.size, stride, dtype=int)
        if indices[-1] != q_path.size - 1:
            indices = np.concatenate((indices, np.asarray([q_path.size - 1], dtype=int)))
        q_sample = q_path[indices]
        time_sample = times[indices]
        geometric_control = compare_liquid_geometric_contribution(
            model,
            q_sample,
            time_sample,
            initial_state=initial_state,
            packet_center=packet_center,
            packet_width=packet_width,
            substeps=substeps,
        )
        diagnostics = liquid_ldr_diagnostics(model, q_sample, time_sample)
        hotspots = liquid_ldr_geometric_hotspots(
            model,
            q_sample,
            time_sample,
            geometric_control=geometric_control,
            diagnostics=diagnostics,
            top_k=top_k,
            initial_state=initial_state,
            packet_center=packet_center,
            packet_width=packet_width,
        )
        signal = liquid_ldr_geometric_signal_summary(geometric_control, hotspots=hotspots)
        quality = liquid_ldr_geometric_quality(
            geometric_control,
            signal_summary=signal,
            population_tolerance=population_tolerance,
            norm_tolerance=norm_tolerance,
            min_steps=min_steps,
        )
        top_hotspot = hotspots[0] if hotspots else None
        records.append(
            {
                "stride": int(stride),
                "indices": [int(index) for index in indices],
                "sample_count": int(q_sample.size),
                "step_count": int(q_sample.size - 1),
                "time_start": float(time_sample[0]),
                "time_end": float(time_sample[-1]),
                "time_start_fs": float(time_sample[0] * au2fs),
                "time_end_fs": float(time_sample[-1] * au2fs),
                "q_start": float(q_sample[0]),
                "q_end": float(q_sample[-1]),
                "population_delta_max_abs": float(signal["population_delta_max_abs"]),
                "population_delta_final_norm": float(signal["population_delta_final_norm"]),
                "population_delta_path_length": float(signal["population_delta_path_length"]),
                "top_step_score_fraction": float(signal["top_step_score_fraction"]),
                "top3_step_score_fraction": float(signal["top3_step_score_fraction"]),
                "effective_step_count": float(signal["effective_step_count"]),
                "norm_error_max": float(quality["norm_error_max"]),
                "quality_verdict": quality["verdict"],
                "quality_recommendation": quality["recommendation"],
                "geometry_visible": bool(quality["geometry_visible"]),
                "norm_stable": bool(quality["norm_stable"]),
                "enough_steps": bool(quality["enough_steps"]),
                "top_hotspot": top_hotspot,
                "top_hotspot_score": float(top_hotspot["score"]) if top_hotspot is not None else 0.0,
                "top_hotspot_step": int(top_hotspot["step"]) if top_hotspot is not None else None,
                "top_hotspot_time_start_fs": float(top_hotspot["time_start_fs"])
                if top_hotspot is not None
                else None,
                "top_hotspot_time_end_fs": float(top_hotspot["time_end_fs"])
                if top_hotspot is not None
                else None,
            }
        )

    baseline = records[0]
    baseline_population = float(baseline["population_delta_max_abs"])
    baseline_path_length = float(baseline["population_delta_path_length"])
    for record in records:
        record["population_delta_max_abs_relative_to_baseline"] = _safe_ratio(
            record["population_delta_max_abs"],
            baseline_population,
        )
        record["population_delta_path_length_relative_to_baseline"] = _safe_ratio(
            record["population_delta_path_length"],
            baseline_path_length,
        )

    ready_records = [record for record in records if record["quality_verdict"] == "ready"]
    retained_ready_records = [
        record
        for record in ready_records
        if _ratio_at_least(
            record["population_delta_max_abs_relative_to_baseline"],
            population_retention_tolerance,
        )
        and _ratio_at_least(
            record["population_delta_path_length_relative_to_baseline"],
            path_length_retention_tolerance,
        )
    ]
    recommended = retained_ready_records[-1] if retained_ready_records else (ready_records[0] if ready_records else None)
    return {
        "records": records,
        "baseline_stride": int(baseline["stride"]),
        "recommended_stride": None if recommended is None else int(recommended["stride"]),
        "any_ready": bool(ready_records),
        "all_ready": bool(len(ready_records) == len(records)),
        "population_retention_tolerance": float(population_retention_tolerance),
        "path_length_retention_tolerance": float(path_length_retention_tolerance),
        "population_tolerance": float(population_tolerance),
        "norm_tolerance": float(norm_tolerance),
        "min_steps": int(min_steps),
        "substeps": int(substeps),
    }


def liquid_ldr_geometric_gauge_invariance(
    model,
    q_path,
    times,
    *,
    phase_offsets=None,
    phase_slopes=None,
    initial_state=0,
    packet_center=-1.0,
    packet_width=0.7,
    tolerance=1.0e-4,
    substeps=4,
):
    """Compare liquid LDR propagation before and after a q-dependent phase gauge."""

    q_path = np.asarray(q_path, dtype=float)
    times = np.asarray(times, dtype=float)
    if q_path.ndim != 1 or times.ndim != 1 or q_path.shape != times.shape:
        raise ValueError("q_path and times must be one-dimensional arrays with the same shape.")
    if q_path.size < 2:
        raise ValueError("at least two liquid-coordinate samples are required.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing.")

    ngrid = model.x_grid.size
    nstates = model.energies([q_path[0]]).shape[1]
    if phase_offsets is None:
        grid_index = np.arange(ngrid, dtype=float)[:, None]
        state_index = np.arange(nstates, dtype=float)[None, :]
        phase_offsets = 0.11 * (grid_index + 1.0) * (state_index + 0.5)
    if phase_slopes is None:
        x_scaled = model.x_grid - float(np.mean(model.x_grid))
        if np.max(np.abs(x_scaled)) > 0.0:
            x_scaled = x_scaled / np.max(np.abs(x_scaled))
        state_index = np.arange(nstates, dtype=float)[None, :]
        phase_slopes = 0.17 * x_scaled[:, None] * (state_index + 1.0)

    gauged_model = PhaseGaugedLiquidLDRModel(model, phase_offsets, phase_slopes)
    reference_initial = initial_ldr_packet(
        model.x_grid,
        center=packet_center,
        width=packet_width,
        state=initial_state,
        nstates=nstates,
    )
    gauged_initial = reference_initial * np.conjugate(gauged_model.phase([q_path[0]]))

    reference_with = propagate_liquid_ldr(
        model,
        q_path,
        times,
        initial_coefficients=reference_initial,
        include_berry=True,
        substeps=substeps,
    )
    gauged_with = propagate_liquid_ldr(
        gauged_model,
        q_path,
        times,
        initial_coefficients=gauged_initial,
        include_berry=True,
        substeps=substeps,
    )
    reference_without = propagate_liquid_ldr(
        model,
        q_path,
        times,
        initial_coefficients=reference_initial,
        include_berry=False,
        substeps=substeps,
    )
    gauged_without = propagate_liquid_ldr(
        gauged_model,
        q_path,
        times,
        initial_coefficients=gauged_initial,
        include_berry=False,
        substeps=substeps,
    )

    with_delta = reference_with["populations"] - gauged_with["populations"]
    without_delta = reference_without["populations"] - gauged_without["populations"]
    with_norm_delta = reference_with["norm"] - gauged_with["norm"]
    without_norm_delta = reference_without["norm"] - gauged_without["norm"]
    max_with = float(np.max(np.abs(with_delta)))
    max_without = float(np.max(np.abs(without_delta)))
    gauge_ready = bool(max_with <= float(tolerance))
    if gauge_ready:
        recommendation = "Berry-enabled liquid LDR populations are stable under the tested q-dependent phase gauge."
    else:
        recommendation = "Increase LDR substeps or reduce the liquid trajectory timestep before using strict gauge-invariance thresholds."
    return {
        "gauge_ready": gauge_ready,
        "recommendation": recommendation,
        "tolerance": float(tolerance),
        "substeps": int(substeps),
        "phase_offsets": np.asarray(phase_offsets, dtype=float).tolist(),
        "phase_slopes": np.asarray(phase_slopes, dtype=float).tolist(),
        "with_geometry_population_delta": with_delta,
        "without_geometry_population_delta": without_delta,
        "with_geometry_population_delta_max_abs": max_with,
        "without_geometry_population_delta_max_abs": max_without,
        "with_geometry_norm_delta_max_abs": float(np.max(np.abs(with_norm_delta))),
        "without_geometry_norm_delta_max_abs": float(np.max(np.abs(without_norm_delta))),
        "with_geometry_reference_populations": reference_with["populations"],
        "with_geometry_gauged_populations": gauged_with["populations"],
        "without_geometry_reference_populations": reference_without["populations"],
        "without_geometry_gauged_populations": gauged_without["populations"],
    }


def liquid_ldr_geometric_gauge_substep_convergence(
    model,
    q_path,
    times,
    substeps,
    *,
    phase_offsets=None,
    phase_slopes=None,
    initial_state=0,
    packet_center=-1.0,
    packet_width=0.7,
    tolerance=1.0e-4,
):
    """Check gauge-invariance error as the liquid LDR internal substeps are refined."""

    substeps = [int(value) for value in substeps]
    if not substeps:
        raise ValueError("at least one substep count is required.")
    if any(value <= 0 for value in substeps):
        raise ValueError("substep counts must be positive integers.")

    deduped = []
    for value in substeps:
        if value not in deduped:
            deduped.append(value)

    records = []
    for value in deduped:
        diagnostic = liquid_ldr_geometric_gauge_invariance(
            model,
            q_path,
            times,
            phase_offsets=phase_offsets,
            phase_slopes=phase_slopes,
            initial_state=initial_state,
            packet_center=packet_center,
            packet_width=packet_width,
            tolerance=tolerance,
            substeps=value,
        )
        records.append(
            {
                "substeps": int(value),
                "gauge_ready": bool(diagnostic["gauge_ready"]),
                "recommendation": diagnostic["recommendation"],
                "with_geometry_population_delta_max_abs": float(
                    diagnostic["with_geometry_population_delta_max_abs"]
                ),
                "without_geometry_population_delta_max_abs": float(
                    diagnostic["without_geometry_population_delta_max_abs"]
                ),
                "with_geometry_norm_delta_max_abs": float(diagnostic["with_geometry_norm_delta_max_abs"]),
                "without_geometry_norm_delta_max_abs": float(
                    diagnostic["without_geometry_norm_delta_max_abs"]
                ),
            }
        )

    baseline_error = records[0]["with_geometry_population_delta_max_abs"]
    for record in records:
        record["with_geometry_error_relative_to_baseline"] = _safe_ratio(
            record["with_geometry_population_delta_max_abs"],
            baseline_error,
        )

    ready_records = [record for record in records if record["gauge_ready"]]
    recommended = ready_records[0] if ready_records else min(
        records,
        key=lambda record: (record["with_geometry_population_delta_max_abs"], record["substeps"]),
    )
    errors = np.asarray([record["with_geometry_population_delta_max_abs"] for record in records], dtype=float)
    monotonic_nonincreasing = True
    if errors.size > 1:
        monotonic_nonincreasing = bool(np.all(np.diff(errors) <= 1.0e-12 * np.maximum(1.0, errors[:-1])))
    return {
        "records": records,
        "recommended_substeps": int(recommended["substeps"]),
        "recommended_gauge_ready": bool(recommended["gauge_ready"]),
        "any_ready": bool(ready_records),
        "all_ready": bool(len(ready_records) == len(records)),
        "tolerance": float(tolerance),
        "error_monotonic_nonincreasing": monotonic_nonincreasing,
    }


def liquid_ldr_substep_convergence(
    model,
    q_path,
    times,
    substeps,
    *,
    initial_state=0,
    packet_center=-1.0,
    packet_width=0.7,
    population_tolerance=1.0e-4,
    geometric_tolerance=1.0e-4,
    norm_tolerance=1.0e-10,
):
    """Check physical liquid LDR population convergence with internal substeps."""

    substeps = [int(value) for value in substeps]
    if not substeps:
        raise ValueError("at least one substep count is required.")
    if any(value <= 0 for value in substeps):
        raise ValueError("substep counts must be positive integers.")
    deduped = []
    for value in substeps:
        if value not in deduped:
            deduped.append(value)
    ordered = sorted(deduped)

    controls = {}
    for value in ordered:
        controls[value] = compare_liquid_geometric_contribution(
            model,
            q_path,
            times,
            initial_state=initial_state,
            packet_center=packet_center,
            packet_width=packet_width,
            substeps=value,
        )

    reference_substeps = ordered[-1]
    reference = controls[reference_substeps]
    reference_pop = np.asarray(reference["with_geometry"]["populations"], dtype=float)
    reference_no_berry = np.asarray(reference["without_geometry"]["populations"], dtype=float)
    reference_geo = np.asarray(reference["population_delta"], dtype=float)

    records = []
    for value in ordered:
        control = controls[value]
        pop_delta = np.asarray(control["with_geometry"]["populations"], dtype=float) - reference_pop
        no_berry_delta = np.asarray(control["without_geometry"]["populations"], dtype=float) - reference_no_berry
        geo_delta = np.asarray(control["population_delta"], dtype=float) - reference_geo
        norm_error = max(
            float(control["with_geometry_norm_max_error"]),
            float(control["without_geometry_norm_max_error"]),
        )
        population_error = float(np.max(np.abs(pop_delta)))
        geometric_error = float(np.max(np.abs(geo_delta)))
        no_berry_error = float(np.max(np.abs(no_berry_delta)))
        records.append(
            {
                "substeps": int(value),
                "is_reference": bool(value == reference_substeps),
                "population_error_max_abs": population_error,
                "population_error_rms": float(np.sqrt(np.mean(pop_delta * pop_delta))),
                "population_final_error_norm": float(np.linalg.norm(pop_delta[-1])),
                "no_berry_population_error_max_abs": no_berry_error,
                "geometric_population_delta_error_max_abs": geometric_error,
                "geometric_population_delta_error_rms": float(np.sqrt(np.mean(geo_delta * geo_delta))),
                "geometric_population_delta_final_error_norm": float(np.linalg.norm(geo_delta[-1])),
                "population_delta_max_abs": float(control["population_delta_max_abs"]),
                "norm_error_max": norm_error,
                "population_converged": bool(population_error <= float(population_tolerance)),
                "geometric_converged": bool(geometric_error <= float(geometric_tolerance)),
                "norm_stable": bool(norm_error <= float(norm_tolerance)),
            }
        )

    for record in records:
        record["ready"] = bool(
            record["population_converged"] and record["geometric_converged"] and record["norm_stable"]
        )

    ready_records = [record for record in records if record["ready"]]
    recommended = ready_records[0] if ready_records else min(
        records,
        key=lambda record: (
            record["population_error_max_abs"] + record["geometric_population_delta_error_max_abs"],
            record["substeps"],
        ),
    )
    pop_errors = np.asarray([record["population_error_max_abs"] for record in records], dtype=float)
    geo_errors = np.asarray([record["geometric_population_delta_error_max_abs"] for record in records], dtype=float)
    return {
        "records": records,
        "reference_substeps": int(reference_substeps),
        "recommended_substeps": int(recommended["substeps"]),
        "recommended_ready": bool(recommended["ready"]),
        "any_ready": bool(ready_records),
        "all_ready": bool(len(ready_records) == len(records)),
        "population_tolerance": float(population_tolerance),
        "geometric_tolerance": float(geometric_tolerance),
        "norm_tolerance": float(norm_tolerance),
        "population_error_monotonic_nonincreasing": _monotonic_nonincreasing(pop_errors),
        "geometric_error_monotonic_nonincreasing": _monotonic_nonincreasing(geo_errors),
    }


def embedded_ldr_substep_convergence(
    snapshots,
    times,
    kinetic_x,
    substeps,
    *,
    frame_transport="phase_aligned",
    initial_state=0,
    packet_center=None,
    packet_width=None,
    normalize=True,
    frame_overlap_diagnostics=None,
    population_tolerance=1.0e-4,
    geometric_tolerance=1.0e-4,
    norm_tolerance=1.0e-10,
):
    """Check embedded LDR population convergence with internal substeps."""

    substeps = [int(value) for value in substeps]
    if not substeps:
        raise ValueError("at least one substep count is required.")
    if any(value <= 0 for value in substeps):
        raise ValueError("substep counts must be positive integers.")
    deduped = []
    for value in substeps:
        if value not in deduped:
            deduped.append(value)
    ordered = sorted(deduped)

    controls = {}
    for value in ordered:
        controls[value] = compare_embedded_geometric_contribution(
            snapshots,
            times,
            kinetic_x,
            frame_transport=frame_transport,
            initial_state=initial_state,
            packet_center=packet_center,
            packet_width=packet_width,
            normalize=normalize,
            frame_overlap_diagnostics=frame_overlap_diagnostics,
            substeps=value,
        )

    reference_substeps = ordered[-1]
    reference = controls[reference_substeps]
    reference_pop = np.asarray(reference["with_geometry"]["populations"], dtype=float)
    reference_plain = np.asarray(reference["without_geometry"]["populations"], dtype=float)
    reference_geo = np.asarray(reference["population_delta"], dtype=float)

    records = []
    for value in ordered:
        control = controls[value]
        pop_delta = np.asarray(control["with_geometry"]["populations"], dtype=float) - reference_pop
        plain_delta = np.asarray(control["without_geometry"]["populations"], dtype=float) - reference_plain
        geo_delta = np.asarray(control["population_delta"], dtype=float) - reference_geo
        norm_error = max(
            float(control["with_geometry_norm_max_error"]),
            float(control["without_geometry_norm_max_error"]),
        )
        population_error = float(np.max(np.abs(pop_delta)))
        geometric_error = float(np.max(np.abs(geo_delta)))
        plain_error = float(np.max(np.abs(plain_delta)))
        records.append(
            {
                "substeps": int(value),
                "is_reference": bool(value == reference_substeps),
                "population_error_max_abs": population_error,
                "population_error_rms": float(np.sqrt(np.mean(pop_delta * pop_delta))),
                "population_final_error_norm": float(np.linalg.norm(pop_delta[-1])),
                "untransported_population_error_max_abs": plain_error,
                "geometric_population_delta_error_max_abs": geometric_error,
                "geometric_population_delta_error_rms": float(np.sqrt(np.mean(geo_delta * geo_delta))),
                "geometric_population_delta_final_error_norm": float(np.linalg.norm(geo_delta[-1])),
                "population_delta_max_abs": float(control["population_delta_max_abs"]),
                "norm_error_max": norm_error,
                "population_converged": bool(population_error <= float(population_tolerance)),
                "geometric_converged": bool(geometric_error <= float(geometric_tolerance)),
                "norm_stable": bool(norm_error <= float(norm_tolerance)),
            }
        )

    for record in records:
        record["ready"] = bool(
            record["population_converged"] and record["geometric_converged"] and record["norm_stable"]
        )

    ready_records = [record for record in records if record["ready"]]
    recommended = ready_records[0] if ready_records else min(
        records,
        key=lambda record: (
            record["population_error_max_abs"] + record["geometric_population_delta_error_max_abs"],
            record["substeps"],
        ),
    )
    pop_errors = np.asarray([record["population_error_max_abs"] for record in records], dtype=float)
    geo_errors = np.asarray([record["geometric_population_delta_error_max_abs"] for record in records], dtype=float)
    return {
        "records": records,
        "reference_substeps": int(reference_substeps),
        "recommended_substeps": int(recommended["substeps"]),
        "recommended_ready": bool(recommended["ready"]),
        "any_ready": bool(ready_records),
        "all_ready": bool(len(ready_records) == len(records)),
        "frame_transport": _normalize_frame_transport(frame_transport),
        "population_tolerance": float(population_tolerance),
        "geometric_tolerance": float(geometric_tolerance),
        "norm_tolerance": float(norm_tolerance),
        "population_error_monotonic_nonincreasing": _monotonic_nonincreasing(pop_errors),
        "geometric_error_monotonic_nonincreasing": _monotonic_nonincreasing(geo_errors),
    }


def liquid_ldr_geometric_readiness(
    quality,
    *,
    substep_convergence=None,
    gauge_check=None,
    gauge_substep_convergence=None,
    stride_convergence=None,
):
    """Combine liquid LDR geometric quality and optional convergence diagnostics."""

    checks = []

    quality_ready = str(quality.get("verdict", "unknown")) == "ready"
    checks.append(
        {
            "name": "quality",
            "ready": bool(quality_ready),
            "detail": quality.get("verdict", "unknown"),
            "recommendation": quality.get("recommendation", "Run analytic liquid LDR quality diagnostics."),
        }
    )

    if substep_convergence is not None:
        ready = bool(substep_convergence.get("recommended_ready", False))
        checks.append(
            {
                "name": "substeps",
                "ready": ready,
                "detail": f"recommended_substeps={substep_convergence.get('recommended_substeps')}",
                "recommendation": _ready_or_recommend(
                    ready,
                    "Use the recommended liquid LDR substeps for the main propagation.",
                ),
            }
        )

    if gauge_check is not None:
        ready = bool(gauge_check.get("gauge_ready", False))
        checks.append(
            {
                "name": "gauge",
                "ready": ready,
                "detail": f"substeps={gauge_check.get('substeps')}",
                "recommendation": gauge_check.get(
                    "recommendation",
                    "Increase gauge-check substeps or relax the gauge tolerance.",
                ),
            }
        )

    if gauge_substep_convergence is not None:
        ready = bool(gauge_substep_convergence.get("recommended_gauge_ready", False))
        checks.append(
            {
                "name": "gauge_substeps",
                "ready": ready,
                "detail": f"recommended_substeps={gauge_substep_convergence.get('recommended_substeps')}",
                "recommendation": _ready_or_recommend(
                    ready,
                    "Use the recommended gauge-check substeps or refine the trajectory timestep.",
                ),
            }
        )

    if stride_convergence is not None:
        ready = bool(stride_convergence.get("any_ready", False))
        recommended_stride = stride_convergence.get("recommended_stride")
        baseline_stride = stride_convergence.get("baseline_stride")
        checks.append(
            {
                "name": "stride",
                "ready": ready,
                "detail": f"recommended_stride={recommended_stride}",
                "recommendation": _stride_recommendation(ready, recommended_stride, baseline_stride),
            }
        )

    failed = [check for check in checks if not check["ready"]]
    if failed:
        verdict = f"{failed[0]['name']}_limited"
        recommendation = failed[0]["recommendation"]
    else:
        verdict = "ready"
        recommendation = "Liquid-phase geometric LDR diagnostics are ready for the requested checks."

    return {
        "verdict": verdict,
        "ready": bool(not failed),
        "recommendation": recommendation,
        "checks": checks,
        "failed_checks": [check["name"] for check in failed],
    }


def _safe_ratio(value, baseline):
    value = float(value)
    baseline = float(baseline)
    if baseline == 0.0:
        return None
    return float(value / baseline)


def _ratio_at_least(value, threshold):
    if value is None:
        return False
    return float(value) >= float(threshold)


def _monotonic_nonincreasing(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return True
    return bool(np.all(np.diff(values) <= 1.0e-12 * np.maximum(1.0, values[:-1])))


def _ready_or_recommend(ready, message):
    if ready:
        return "Ready."
    return message


def _stride_recommendation(ready, recommended_stride, baseline_stride):
    if not ready:
        return "Run a longer or more finely sampled liquid trajectory before interpreting stride convergence."
    if recommended_stride == baseline_stride:
        return "Use the baseline stride for geometric hot-spot interpretation."
    return "Use the recommended stride only for coarse screening; use the baseline stride for detailed hot spots."


def _parse_xyz_time(comment):
    if not comment.startswith("time="):
        return None
    try:
        return float(comment.split("=", 1)[1].split()[0])
    except ValueError:
        return None


def _solvent_charges(natoms):
    if natoms % 3 != 0:
        raise ValueError("solvent atoms after solute must be O-H-H water triplets.")
    return np.tile(TIP3P_CHARGES, natoms // 3)


def _populations(c):
    c = np.asarray(c, dtype=complex)
    return np.sum(np.abs(c) ** 2, axis=0).real


def _expectation(operator, c):
    c_flat = np.asarray(c, dtype=complex).reshape(-1)
    return float((np.vdot(c_flat, operator @ c_flat) / np.vdot(c_flat, c_flat)).real)


def _electronic_overlap_block(left, right, nstates, *, diagonal=False):
    if left is None or right is None:
        if diagonal:
            return np.eye(int(nstates), dtype=complex)
        return np.zeros((int(nstates), int(nstates)), dtype=complex)
    if hasattr(left, "wavefunction_overlap"):
        return np.asarray(left.wavefunction_overlap(right), dtype=complex)
    if hasattr(left, "overlap"):
        return np.asarray(left.overlap(right), dtype=complex)

    from pyqed.qchem.mcscf.casci import overlap as casci_overlap

    return np.asarray(casci_overlap(left, right), dtype=complex)


def _closest_unitary(matrix):
    left, singular_values, right_h = np.linalg.svd(np.asarray(matrix, dtype=complex), full_matrices=False)
    return left @ right_h, singular_values


def _phase_align_unitary_transport(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    diagonal = np.diag(matrix)
    phases = np.ones(diagonal.shape, dtype=complex)
    active = np.abs(diagonal) > 1.0e-12
    phases[active] = diagonal[active] / np.abs(diagonal[active])
    return matrix @ np.diag(np.conj(phases))


def _normalize_frame_transport(frame_transport):
    if frame_transport is None or frame_transport is False:
        return None
    if frame_transport is True:
        return "unitary"
    frame_transport = str(frame_transport).strip().lower()
    if frame_transport in {"none", "off", "false", "0"}:
        return None
    if frame_transport in {"phase", "phase-aligned", "phase_aligned", "parallel"}:
        return "phase_aligned"
    if frame_transport not in {"unitary", "raw", "phase_aligned"}:
        raise ValueError("frame_transport must be None, 'unitary', 'phase_aligned', or 'raw'.")
    return frame_transport


def _frame_overlap_transport_array(frame_overlap_diagnostics, frame_transport):
    frame_transport = _normalize_frame_transport(frame_transport)
    if frame_transport is None:
        frame_transport = "unitary"
    keys = {
        "unitary": "unitary_transport_sequence",
        "phase_aligned": "phase_aligned_unitary_transport_sequence",
        "raw": "overlap_sequence",
    }
    key = keys[frame_transport]
    if key not in frame_overlap_diagnostics:
        raise ValueError(f"frame_overlap_diagnostics does not contain {key!r}.")
    return frame_overlap_diagnostics[key]


def _frame_transport_sequence(frame_overlap_diagnostics, frame_transport, *, nsteps, ngrid, nstates):
    sequence = np.asarray(_frame_overlap_transport_array(frame_overlap_diagnostics, frame_transport), dtype=complex)
    expected = (int(nsteps), int(ngrid), int(nstates), int(nstates))
    if sequence.shape != expected:
        raise ValueError(f"{frame_transport} transport sequence shape {sequence.shape} != {expected}.")
    return sequence


def _block_diagonal_frame_transport(transport_blocks):
    transport_blocks = np.asarray(transport_blocks, dtype=complex)
    if transport_blocks.ndim != 3 or transport_blocks.shape[1] != transport_blocks.shape[2]:
        raise ValueError("transport_blocks must have shape (ngrid, nstates, nstates).")
    ngrid, nstates, _ = transport_blocks.shape
    matrix = np.zeros((ngrid * nstates, ngrid * nstates), dtype=complex)
    for grid_index, block in enumerate(transport_blocks):
        start = grid_index * nstates
        stop = start + nstates
        matrix[start:stop, start:stop] = block
    return matrix


def _geometric_hotspot_source(geometric_score, leakage_score):
    geometric_score = float(geometric_score)
    leakage_score = float(leakage_score)
    scale = max(abs(geometric_score), abs(leakage_score), 1.0)
    tol = 1.0e-12 * scale
    if geometric_score <= tol and leakage_score <= tol:
        return "quiet"
    if leakage_score > 2.0 * max(geometric_score, tol):
        return "leakage"
    if geometric_score > 2.0 * max(leakage_score, tol):
        return "geometric"
    return "mixed"


def _strictly_increasing_times(times):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if times.size == 0:
        raise ValueError("at least one time is required.")
    if np.all(np.diff(times) > 0.0):
        return times
    return np.arange(times.size, dtype=float)


def _validate_embedded_snapshots(snapshots):
    snapshots = tuple(snapshots)
    if len(snapshots) == 0:
        raise ValueError("at least one embedded snapshot is required.")
    first_grid = np.asarray(snapshots[0].bond_grid, dtype=float)
    first_shape = np.asarray(snapshots[0].apes).shape
    if first_grid.ndim != 1:
        raise ValueError("snapshot bond_grid must be one-dimensional.")
    if len(first_shape) != 2:
        raise ValueError("snapshot apes must be two-dimensional.")
    ngrid, nstates = first_shape
    expected_overlap_shape = (ngrid, nstates, ngrid, nstates)
    for index, snapshot in enumerate(snapshots):
        bond_grid = np.asarray(snapshot.bond_grid, dtype=float)
        apes = np.asarray(snapshot.apes)
        overlap = np.asarray(snapshot.overlap)
        if bond_grid.shape != first_grid.shape or not np.allclose(bond_grid, first_grid):
            raise ValueError(f"snapshot {index} bond_grid does not match the first snapshot.")
        if apes.shape != first_shape:
            raise ValueError(f"snapshot {index} apes shape {apes.shape} != {first_shape}.")
        if overlap.shape != expected_overlap_shape:
            raise ValueError(
                f"snapshot {index} overlap shape {overlap.shape} != {expected_overlap_shape}."
            )
    return snapshots


def _embedded_h2_casci_point(
    geometry,
    pc_coords,
    pc_charges,
    *,
    basis,
    nstates,
    ncas,
    nelecas,
    spin,
    method,
    run_kwargs,
    reference_run_kwargs,
):
    return _embedded_casci_point(
        ("H", "H"),
        geometry,
        pc_coords,
        pc_charges,
        basis=basis,
        charge=0,
        nstates=nstates,
        ncas=ncas,
        nelecas=nelecas,
        spin=spin,
        method=method,
        run_kwargs=run_kwargs,
        reference_run_kwargs=reference_run_kwargs,
    )


def _embedded_casci_point(
    symbols,
    geometry,
    pc_coords,
    pc_charges,
    *,
    basis,
    charge,
    nstates,
    ncas,
    nelecas,
    spin,
    method,
    run_kwargs,
    reference_run_kwargs,
):
    from pyqed import Molecule
    from pyqed.qchem import embed_point_charges
    from pyqed.qchem.mcscf.casci import CASCI

    symbols = tuple(str(symbol) for symbol in symbols)
    geometry = np.asarray(geometry, dtype=float)
    if geometry.shape != (len(symbols), 3):
        raise ValueError(f"geometry shape {geometry.shape} != {(len(symbols), 3)}.")
    atom = "; ".join(
        f"{symbol} {x:.16g} {y:.16g} {z:.16g}"
        for symbol, (x, y, z) in zip(symbols, geometry)
    )
    mol = Molecule(atom=atom, unit="bohr", basis=basis, charge=int(charge), spin=spin)
    mol.build()
    mf = mol.RHF()
    mc = CASCI(mf, ncas=int(ncas), nelecas=nelecas, spin=spin, verbose=0)
    embedded = embed_point_charges(
        mc,
        pc_coords,
        pc_charges,
        run_kwargs={} if run_kwargs is None else dict(run_kwargs),
        reference_run_kwargs=(
            {"verbose": 0, "max_cycle": 100}
            if reference_run_kwargs is None
            else dict(reference_run_kwargs)
        ),
    )
    embedded.run(nstates=int(nstates), method=method)
    energies = np.asarray(embedded.method.e_tot, dtype=float)
    return energies, embedded.method
