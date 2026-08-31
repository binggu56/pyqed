"""GDVR real-space RT-TDHF determinant frames for LDR nuclear grids."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
from scipy.linalg import det, eigh, expm
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

from pyqed.ldr import kinetic as kinetic_tools
from pyqed.ldr import overlap as overlap_tools
from pyqed.qchem.gdvr import RTTDHF


def _occupied_spatial_orbitals(mf, *, thresh=1.0e-12):
    coeff = np.asarray(mf.mo_coeff, dtype=complex)
    occ = np.asarray(getattr(mf, "mo_occ", None), dtype=float)
    if occ.shape == (coeff.shape[1],):
        keep = occ > float(thresh)
    else:
        keep = np.arange(coeff.shape[1]) < int(round(mf.mol.nelec / 2))
    if not np.any(keep):
        raise ValueError("GDVR RHF reference has no occupied orbitals.")
    return coeff[:, keep]


def _orthonormal_columns(coefficients, *, thresh=1.0e-12):
    coefficients = np.asarray(coefficients, dtype=complex)
    metric = coefficients.conj().T @ coefficients
    evals, evecs = eigh(0.5 * (metric + metric.conj().T))
    if evals.size == 0 or np.any(evals < float(thresh)):
        raise ValueError("Occupied GDVR orbital frame is singular.")
    return coefficients @ evecs @ np.diag(evals**-0.5)


def gdvr_det_overlap(left, right, *, phase="full"):
    """Closed-shell determinant overlap for two same-grid GDVR frames."""

    if left.size != right.size or left.M != right.M:
        raise ValueError("GDVR determinant overlaps require matching basis sizes.")
    if not np.allclose(left.z, right.z, rtol=0.0, atol=1.0e-12):
        raise ValueError("GDVR determinant overlaps require a shared z grid.")

    c_left = getattr(left.mol, "c_list", None)
    c_right = getattr(right.mol, "c_list", None)
    if c_left is None and c_right is None:
        orbital_overlap = left.overlap_orbitals.conj().T @ right.overlap_orbitals
    elif c_left is None or c_right is None:
        raise ValueError("Both GDVR frames must provide c_list for determinant overlaps.")
    else:
        context_left = left.mol._gdvr_build_context
        context_right = right.mol._gdvr_build_context
        if not (
            np.array_equal(context_left["alphas"], context_right["alphas"])
            and np.array_equal(context_left["centers"], context_right["centers"])
            and tuple(context_left["labels"]) == tuple(context_right["labels"])
        ):
            raise ValueError("GDVR determinant overlaps require a shared primitive transverse basis.")
        basis_overlap = np.einsum(
            "kpa,pq,kqb->kab",
            np.stack(c_left).conj(),
            context_left["S_prim"],
            np.stack(c_right),
            optimize=True,
        )
        orbitals_left = left.overlap_orbitals.reshape(len(left.z), left.M, -1)
        orbitals_right = right.overlap_orbitals.reshape(len(right.z), right.M, -1)
        orbital_overlap = np.einsum(
            "kai,kab,kbj->ij",
            orbitals_left.conj(),
            basis_overlap,
            orbitals_right,
            optimize=True,
        )
    spatial_det = det(orbital_overlap)
    if phase == "full":
        left_phase = getattr(left, "det_phase", 1.0 + 0.0j)
        right_phase = getattr(right, "det_phase", 1.0 + 0.0j)
    elif phase == "transport":
        left_phase = getattr(left, "det_phase", 1.0 + 0.0j) / getattr(left, "dynamical_phase", 1.0 + 0.0j)
        right_phase = getattr(right, "det_phase", 1.0 + 0.0j) / getattr(right, "dynamical_phase", 1.0 + 0.0j)
    elif phase == "raw":
        left_phase = right_phase = 1.0 + 0.0j
    else:
        raise ValueError("phase must be 'full', 'transport', or 'raw'.")
    return np.conj(left_phase) * right_phase * spatial_det * spatial_det


class GDVRFrame:
    """One local real-space GDVR RT-TDHF determinant frame."""

    def __init__(
        self,
        mf,
        *,
        field=None,
        interaction=None,
        cap=None,
        nuclear_dipole=None,
        s_thresh=1.0e-12,
    ):
        self.rt = RTTDHF(mf, interaction=interaction, field=field, cap=cap)
        self.mf = mf
        self.mol = mf.mol
        self.field = field
        self.interaction = interaction
        self.cap = cap
        self.nuclear_dipole = np.zeros(3) if nuclear_dipole is None else np.asarray(
            nuclear_dipole,
            dtype=float,
        ).reshape(3)
        self.s_thresh = float(s_thresh)
        self.z = np.asarray(self.mol.z, dtype=float).copy()
        self.M = int(self.mol.shapes["M"])
        self.size = int(self.rt.size)

        spatial = _occupied_spatial_orbitals(mf, thresh=self.s_thresh)
        self.weighted_orbitals = np.sqrt(2.0) * spatial
        self.det_phase = 1.0 + 0.0j
        self.dynamical_phase = 1.0 + 0.0j

    @property
    def nocc(self):
        return int(self.weighted_orbitals.shape[1])

    @property
    def overlap_orbitals(self):
        return _orthonormal_columns(self.weighted_orbitals, thresh=self.s_thresh)

    def copy(self):
        other = object.__new__(type(self))
        other.rt = self.rt
        other.mf = self.mf
        other.mol = self.mol
        other.field = self.field
        other.interaction = self.interaction
        other.cap = self.cap
        other.nuclear_dipole = self.nuclear_dipole.copy()
        other.s_thresh = self.s_thresh
        other.z = self.z
        other.M = self.M
        other.size = self.size
        other.weighted_orbitals = self.weighted_orbitals.copy()
        other.det_phase = complex(self.det_phase)
        other.dynamical_phase = complex(self.dynamical_phase)
        return other

    def density(self):
        return self.rt.density_from_orbitals(self.weighted_orbitals)

    def energy(self):
        return float(self.rt.energy(self.density()))

    def phase_energy(self, time=0.0, density=None):
        """Electronic energy used for the determinant action phase."""

        dm = self.density() if density is None else np.asarray(density, dtype=complex)
        obs = self.rt.sample_observables(dm, time=time, field=self.field, include_velocity=False)
        dipole = obs["dipole"] + self.nuclear_dipole
        return float(obs["energy"] - np.dot(obs["field"], dipole))

    def dipole_moment(self):
        return self.rt.dipole_moment(self.density()) + self.nuclear_dipole

    def dipole_acceleration(self, time=0.0):
        dm = self.density()
        obs = self.rt.sample_observables(dm, time=time, field=self.field)
        return obs["dipole_acceleration"]

    def electron_count(self):
        return float(self.rt.electron_count(self.density()))

    def step(self, time, dt):
        old_orbitals = self.overlap_orbitals
        old_density = self.density()
        old_energy = self.phase_energy(time=time, density=old_density)
        new_weighted_orbitals = self.rt.step_orbitals(
            self.weighted_orbitals,
            float(time),
            float(dt),
            field=self.field,
        )
        new_density = self.rt.density_from_orbitals(new_weighted_orbitals)
        new_energy = self.phase_energy(time=float(time) + float(dt), density=new_density)
        new_orbitals = _orthonormal_columns(new_weighted_orbitals, thresh=self.s_thresh)
        raw_local_overlap = det(old_orbitals.conj().T @ new_orbitals) ** 2
        if abs(raw_local_overlap) > self.s_thresh:
            raw_phase = raw_local_overlap / abs(raw_local_overlap)
        else:
            raw_phase = 1.0 + 0.0j
        phase_energy = 0.5 * (old_energy + new_energy)
        dynamical_step = np.exp(-1j * phase_energy * float(dt))
        self.dynamical_phase *= dynamical_step
        self.det_phase *= dynamical_step / raw_phase
        if abs(self.dynamical_phase) > 0.0:
            self.dynamical_phase /= abs(self.dynamical_phase)
        if abs(self.det_phase) > 0.0:
            self.det_phase /= abs(self.det_phase)
        self.weighted_orbitals = new_weighted_orbitals
        return self

    def propagate(self, time, dt, substeps=1):
        substeps = int(substeps)
        if substeps <= 0:
            raise ValueError("substeps must be positive.")
        h = float(dt) / substeps
        t0 = float(time)
        for istep in range(substeps):
            self.step(t0 + istep * h, h)
        return self


@dataclass(frozen=True)
class Trajectory:
    times: np.ndarray
    coefficients: np.ndarray
    overlaps: np.ndarray | None
    hamiltonians: np.ndarray | None
    electronic_energies: np.ndarray
    electronic_dipoles: np.ndarray
    electronic_dipole_accelerations: np.ndarray
    electron_counts: np.ndarray
    fields: np.ndarray
    timings: dict | None = None

    @property
    def kinetic_hamiltonians(self):
        return self.hamiltonians

    @property
    def norm(self):
        return np.sum(np.abs(self.coefficients) ** 2, axis=1)

    @property
    def coordinate_density(self):
        return np.abs(self.coefficients) ** 2

    @property
    def weighted_dipole(self):
        return np.einsum("ti,tix->tx", self.coordinate_density, self.electronic_dipoles)

    @property
    def weighted_dipole_acceleration(self):
        return np.einsum(
            "ti,tix->tx",
            self.coordinate_density,
            self.electronic_dipole_accelerations,
        )

    @property
    def weighted_electron_count(self):
        return np.einsum("ti,ti->t", self.coordinate_density, self.electron_counts)


class Solver:
    """Sine-DVR LDR solver with time-dependent GDVR determinant overlaps.

    For one electronic determinant frame per nuclear grid point, the LDR
    coefficient Hamiltonian is

        H_ij(t) = T_ij S_ij(t),

    where ``S_ij(t)`` is the closed-shell determinant overlap between the
    RT-TDHF electronic frames at nuclear points i and j.  The local electronic
    energy term is already carried by the time-dependent electronic frame and
    cancels against the time-basis gauge term in the projected equation.  Each
    frame also carries a scalar determinant action phase so the orbital TDHF
    gauge gives the many-electron phase required by that cancellation.
    """

    def __init__(
        self,
        *,
        nuclear,
        electronic,
        hbar=1.0,
        overlap_method="full",
        grid_shape=None,
        lpa_unitarize_links=False,
        kinetic_sparse_tol=0.0,
        propagation_workers=1,
        electronic_substeps=1,
        phase_representation="action",
    ):
        if not hasattr(nuclear, "points"):
            raise TypeError("nuclear must provide product-grid points.")
        kinetic_builder = getattr(nuclear, "kinetic", None)
        if not callable(kinetic_builder):
            kinetic_builder = getattr(nuclear, "t", None)
        if not callable(kinetic_builder):
            raise TypeError("nuclear must provide kinetic() or t().")

        grid = np.asarray(nuclear.points, dtype=float)
        if grid.ndim == 0:
            raise ValueError("grid must contain at least one nuclear point.")
        kinetic = kinetic_builder()
        if hasattr(kinetic, "toarray"):
            kinetic = kinetic.toarray()
        self.nuclear = nuclear
        self.grid = grid.reshape(-1) if grid.ndim == 1 else grid.reshape(grid.shape[0], -1)
        self.kinetic = np.asarray(kinetic, dtype=complex)
        self.frames = [frame.copy() for frame in electronic]
        self.hbar = float(hbar)
        self.overlap_method = str(overlap_method)
        self.lpa_unitarize_links = bool(lpa_unitarize_links)
        self.kinetic_sparse_tol = float(kinetic_sparse_tol)
        self.propagation_workers = int(propagation_workers)
        self.electronic_substeps = int(electronic_substeps)
        self.phase_representation = str(phase_representation)
        if self.kinetic.shape != (self.ngrid, self.ngrid):
            raise ValueError("kinetic must have shape (ngrid, ngrid).")
        if len(self.frames) != self.ngrid:
            raise ValueError("electronic length must equal nuclear grid size.")
        if self.hbar <= 0.0:
            raise ValueError("hbar must be positive.")
        if self.kinetic_sparse_tol < 0.0:
            raise ValueError("kinetic_sparse_tol must be non-negative.")
        if self.propagation_workers <= 0:
            raise ValueError("propagation_workers must be positive.")
        if self.electronic_substeps <= 0:
            raise ValueError("electronic_substeps must be positive.")
        if self.overlap_method not in {"full", "lpa"}:
            raise ValueError("overlap_method must be 'full' or 'lpa'.")
        if self.phase_representation not in {"action", "pes"}:
            raise ValueError("phase_representation must be 'action' or 'pes'.")
        nocc = self.frames[0].nocc
        if any(frame.nocc != nocc for frame in self.frames):
            raise ValueError("All GDVR RT-TDHF frames must have the same occupied rank.")
        self.grid_shape, self.grid_indices, self.flat_index = self._prepare_product_grid(grid_shape)
        self.lpa_edges = self._prepare_lpa_edges()
        self.last_step_timings = {}

    @property
    def ngrid(self):
        return int(self.grid.shape[0])

    @property
    def ndim(self):
        return 1 if self.grid.ndim == 1 else int(self.grid.shape[1])

    @property
    def points(self):
        return self.grid[:, None] if self.grid.ndim == 1 else self.grid

    def _prepare_product_grid(self, grid_shape):
        if grid_shape is not None:
            shape = tuple(int(n) for n in grid_shape)
            if any(n <= 0 for n in shape):
                raise ValueError("grid_shape entries must be positive.")
            if int(np.prod(shape)) != self.ngrid:
                raise ValueError("grid_shape product must equal ngrid.")
            indices = np.asarray(list(np.ndindex(shape)), dtype=int)
            flat_index = {tuple(idx): i for i, idx in enumerate(indices)}
            return shape, indices, flat_index

        if self.grid.ndim == 1:
            shape = (self.ngrid,)
            indices = np.arange(self.ngrid, dtype=int)[:, None]
            flat_index = {(int(i),): int(i) for i in range(self.ngrid)}
            return shape, indices, flat_index

        points = self.points
        axes = [np.unique(points[:, axis]) for axis in range(points.shape[1])]
        shape = tuple(len(axis) for axis in axes)
        if int(np.prod(shape)) != self.ngrid:
            if self.overlap_method == "lpa":
                raise ValueError("LPA overlap construction requires a complete product grid.")
            return None, None, None

        indices = np.empty((self.ngrid, points.shape[1]), dtype=int)
        seen = set()
        for point_index, point in enumerate(points):
            multi_index = []
            for axis, values in enumerate(axes):
                nearest = int(np.argmin(np.abs(values - point[axis])))
                if not np.isclose(values[nearest], point[axis], rtol=0.0, atol=1.0e-12):
                    if self.overlap_method == "lpa":
                        raise ValueError("LPA could not assign grid point to a product-grid axis.")
                    return None, None, None
                multi_index.append(nearest)
            multi_index = tuple(multi_index)
            if multi_index in seen:
                if self.overlap_method == "lpa":
                    raise ValueError("LPA product-grid assignment has duplicate points.")
                return None, None, None
            seen.add(multi_index)
            indices[point_index] = multi_index
        flat_index = {tuple(idx): i for i, idx in enumerate(indices)}
        return shape, indices, flat_index

    def _prepare_lpa_edges(self):
        if self.overlap_method != "lpa":
            return ()
        if self.grid_shape is None:
            raise ValueError("LPA overlap construction requires grid_shape or an inferable product grid.")

        return overlap_tools.layout(self.grid_shape)[2]

    def _map_frames(self, func, frames=None):
        frames = self.frames if frames is None else list(frames)
        workers = min(self.propagation_workers, len(frames))
        if workers <= 1:
            return [func(frame) for frame in frames]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(func, frames))

    def _half_substeps(self):
        return max(1, int(np.ceil(0.5 * self.electronic_substeps)))

    def _overlap_phase(self, phase=None):
        if phase is not None:
            return str(phase)
        return "transport" if self.phase_representation == "pes" else "full"

    def _full_overlap_matrix(self, phase=None):
        phase = self._overlap_phase(phase)
        overlap = np.empty((self.ngrid, self.ngrid), dtype=complex)
        for i, left in enumerate(self.frames):
            overlap[i, i] = 1.0
            for j in range(i + 1, self.ngrid):
                value = gdvr_det_overlap(left, self.frames[j], phase=phase)
                overlap[i, j] = value
                overlap[j, i] = value.conjugate()
        return overlap

    def _lpa_overlap_links(self, phase=None):
        phase = self._overlap_phase(phase)
        _, flat_index, _ = overlap_tools.layout(self.grid_shape)
        return overlap_tools.nearest(
            self.grid_shape,
            lambda left, right: gdvr_det_overlap(
                self.frames[flat_index[left]],
                self.frames[flat_index[right]],
                phase=phase,
            ),
            unitarize=self.lpa_unitarize_links,
        )

    def _lpa_overlap_between(self, bra_idx, ket_idx, links):
        return overlap_tools.between(bra_idx, ket_idx, links)

    def _lpa_overlap_matrix(self, phase=None):
        return overlap_tools.dense(
            self.grid_shape,
            self._lpa_overlap_links(phase=phase),
        )

    def overlap_matrix(self, phase=None):
        if self.overlap_method == "full":
            return self._full_overlap_matrix(phase=phase)
        return self._lpa_overlap_matrix(phase=phase)

    def kinetic_matrix(self, phase=None):
        return self.kinetic * self.overlap_matrix(phase=phase)

    def _lpa_sparse_kinetic_matrix(self, phase=None):
        if self.overlap_method != "lpa":
            raise ValueError("hamiltonian_method='lpa-sparse' requires overlap_method='lpa'.")
        phase = self._overlap_phase(phase)

        return kinetic_tools.linked(
            self.kinetic,
            self.grid_shape,
            self._lpa_overlap_links(phase=phase),
            threshold=self.kinetic_sparse_tol,
        )

    def electronic_energy_vector(self, time=0.0):
        """Return local electronic energies as diagnostics, not PES terms."""

        energies = np.empty(self.ngrid, dtype=float)
        for i, frame in enumerate(self.frames):
            dm = frame.density()
            energies[i] = frame.rt.sample_observables(dm, time=time, field=frame.field)["energy"]
        return energies

    def phase_energy_vector(self, time=0.0):
        return np.array([frame.phase_energy(time=time) for frame in self.frames], dtype=float)

    def hamiltonian_matrix(self, time=0.0):
        hamiltonian = self.kinetic_matrix()
        if self.phase_representation == "pes":
            hamiltonian += np.diag(self.phase_energy_vector(time=time))
        return 0.5 * (hamiltonian + hamiltonian.conj().T)

    def propagation_hamiltonian(self, time=0.0, *, hamiltonian_method="dense"):
        if hamiltonian_method == "dense":
            return self.hamiltonian_matrix(time=time)
        if hamiltonian_method == "lpa-sparse":
            hamiltonian = self._lpa_sparse_kinetic_matrix()
            if self.phase_representation == "pes":
                hamiltonian = hamiltonian + diags(self.phase_energy_vector(time=time), format="csr")
            return hamiltonian
        raise ValueError(f"unknown Hamiltonian method {hamiltonian_method!r}.")

    def static_ldr_hamiltonian(self, time=0.0, *, hamiltonian_method="dense"):
        """Static-picture LDR Hamiltonian used to prepare initial states."""

        if hamiltonian_method == "lpa-sparse":
            h0 = self._lpa_sparse_kinetic_matrix(phase="transport")
        else:
            h0 = self.kinetic_matrix(phase="transport")
        if hasattr(h0, "toarray"):
            h0 = h0.toarray()
        h0 = np.asarray(h0, dtype=complex)
        h0 = 0.5 * (h0 + h0.conj().T)
        h0 += np.diag(self.phase_energy_vector(time=time))
        return 0.5 * (h0 + h0.conj().T)

    def ground_state(self, time=0.0, *, hamiltonian_method="dense"):
        """Lowest eigenstate of ``T*S(0) + diag(E_i(0))`` for initialization."""

        h0 = self.static_ldr_hamiltonian(time=time, hamiltonian_method=hamiltonian_method)
        energies, states = eigh(h0)
        coeff = np.asarray(states[:, 0], dtype=complex)
        pivot = int(np.argmax(np.abs(coeff)))
        if np.abs(coeff[pivot]) > 0.0:
            coeff *= np.exp(-1j * np.angle(coeff[pivot]))
        return coeff / np.linalg.norm(coeff), float(energies[0])

    def coefficient_ground_state(self, time=0.0, *, hamiltonian_method="dense"):
        """Lowest eigenstate of the coefficient Hamiltonian used for propagation."""

        h0 = self.propagation_hamiltonian(time=time, hamiltonian_method=hamiltonian_method)
        if hasattr(h0, "toarray"):
            h0 = h0.toarray()
        h0 = 0.5 * (np.asarray(h0, dtype=complex) + np.asarray(h0, dtype=complex).conj().T)
        evals, evecs = eigh(h0)
        coeff = np.asarray(evecs[:, 0], dtype=complex)
        pivot = int(np.argmax(np.abs(coeff)))
        if np.abs(coeff[pivot]) > 0.0:
            coeff *= np.exp(-1j * np.angle(coeff[pivot]))
        return coeff / np.linalg.norm(coeff), evals

    def step(self, coefficients, time, dt, *, coefficient_propagator="dense", hamiltonian_method="dense"):
        timings = {}
        t0 = perf_counter()
        mid_frames = self._map_frames(
            lambda frame: frame.copy().propagate(time, 0.5 * dt, substeps=self._half_substeps())
        )
        timings["electronic_midpoint_seconds"] = perf_counter() - t0
        old_frames = self.frames
        self.frames = mid_frames
        t0 = perf_counter()
        h_mid = self.propagation_hamiltonian(
            time=time + 0.5 * dt,
            hamiltonian_method=hamiltonian_method,
        )
        timings["ldr_hamiltonian_seconds"] = perf_counter() - t0
        self.frames = old_frames

        generator = -1j * h_mid * float(dt) / self.hbar
        coefficients = np.asarray(coefficients, dtype=complex)
        t0 = perf_counter()
        if coefficient_propagator == "dense":
            if hamiltonian_method != "dense":
                generator = generator.toarray()
            next_coefficients = expm(generator) @ coefficients
        elif coefficient_propagator == "expm-multiply":
            next_coefficients = expm_multiply(generator, coefficients)
        else:
            raise ValueError(f"unknown coefficient propagator {coefficient_propagator!r}.")
        timings["coefficient_seconds"] = perf_counter() - t0
        t0 = perf_counter()
        self.frames = self._map_frames(
            lambda frame: frame.propagate(time, dt, substeps=self.electronic_substeps),
            self.frames,
        )
        timings["electronic_full_seconds"] = perf_counter() - t0
        self.last_step_timings = timings
        return next_coefficients

    def run(
        self,
        coefficients0,
        *,
        dt,
        nsteps,
        t0=0.0,
        store_hamiltonians=False,
        store_overlaps=True,
        progress_every=0,
        coefficient_propagator="dense",
        hamiltonian_method="dense",
    ):
        coefficients = np.asarray(coefficients0, dtype=complex).reshape(-1)
        if coefficients.shape != (self.ngrid,):
            raise ValueError(f"coefficients0 shape {coefficients.shape} != {(self.ngrid,)}.")
        nsteps = int(nsteps)
        if nsteps < 0:
            raise ValueError("nsteps must be non-negative.")
        dt = float(dt)

        times = float(t0) + dt * np.arange(nsteps + 1, dtype=float)
        coeffs = np.empty((nsteps + 1, self.ngrid), dtype=complex)
        overlaps = (
            np.empty((nsteps + 1, self.ngrid, self.ngrid), dtype=complex)
            if store_overlaps
            else None
        )
        hamiltonian_hist = (
            np.empty((nsteps + 1, self.ngrid, self.ngrid), dtype=complex)
            if store_hamiltonians
            else None
        )
        energies = np.empty((nsteps + 1, self.ngrid), dtype=float)
        dipoles = np.empty((nsteps + 1, self.ngrid, 3), dtype=float)
        accelerations = np.empty((nsteps + 1, self.ngrid, 3), dtype=float)
        electron_counts = np.empty((nsteps + 1, self.ngrid), dtype=float)
        fields = np.empty((nsteps + 1, 3), dtype=float)
        timings = {
            "overlap_seconds": 0.0,
            "hamiltonian_history_seconds": 0.0,
            "observable_seconds": 0.0,
            "electronic_midpoint_seconds": 0.0,
            "ldr_hamiltonian_seconds": 0.0,
            "coefficient_seconds": 0.0,
            "electronic_full_seconds": 0.0,
        }

        for step, time in enumerate(times):
            if progress_every and (step == 0 or step == nsteps or step % int(progress_every) == 0):
                print(f"[rtldr] propagated {step}/{nsteps}", flush=True)
            coeffs[step] = coefficients
            if overlaps is not None:
                t0 = perf_counter()
                overlaps[step] = self.overlap_matrix()
                timings["overlap_seconds"] += perf_counter() - t0
            if hamiltonian_hist is not None:
                t0 = perf_counter()
                hamiltonian_hist[step] = self.hamiltonian_matrix(time=time)
                timings["hamiltonian_history_seconds"] += perf_counter() - t0
            fields[step] = self.frames[0].rt.field_vector(time, field=self.frames[0].field)
            t0 = perf_counter()
            frame_observables = self._map_frames(
                lambda frame: frame.rt.sample_observables(
                    frame.density(),
                    time=time,
                    field=frame.field,
                    include_velocity=False,
                )
            )
            timings["observable_seconds"] += perf_counter() - t0
            for i, obs in enumerate(frame_observables):
                energies[step, i] = obs["energy"]
                dipoles[step, i] = obs["dipole"] + self.frames[i].nuclear_dipole
                accelerations[step, i] = obs["dipole_acceleration"]
                electron_counts[step, i] = obs["electron_count"]
            if step < nsteps:
                coefficients = self.step(
                    coefficients,
                    time,
                    dt,
                    coefficient_propagator=coefficient_propagator,
                    hamiltonian_method=hamiltonian_method,
                )
                for key, value in self.last_step_timings.items():
                    timings[key] = timings.get(key, 0.0) + float(value)

        timings["total_measured_seconds"] = float(sum(timings.values()))
        timings["nsteps"] = int(nsteps)
        timings["ngrid"] = int(self.ngrid)
        timings["propagation_workers"] = int(self.propagation_workers)
        timings["electronic_substeps"] = int(self.electronic_substeps)
        timings["phase_representation"] = self.phase_representation

        return Trajectory(
            times=times,
            coefficients=coeffs,
            overlaps=overlaps,
            hamiltonians=hamiltonian_hist,
            electronic_energies=energies,
            electronic_dipoles=dipoles,
            electronic_dipole_accelerations=accelerations,
            electron_counts=electron_counts,
            fields=fields,
            timings=timings,
        )
