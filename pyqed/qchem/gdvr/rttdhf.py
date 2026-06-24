"""Real-time TDHF propagation in the orthonormal GDVR basis."""

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from pyqed.qchem.gdvr.rhf import fock_2e_slice_collocated, prepare_gdvr_fock_builder


def _axis_to_index(axis):
    if isinstance(axis, str):
        key = axis.lower()
        if key == "x":
            return 0
        if key == "y":
            return 1
        if key == "z":
            return 2
        raise ValueError("axis must be one of 'x', 'y', or 'z'.")
    idx = int(axis)
    if idx not in (0, 1, 2):
        raise ValueError("axis index must be 0, 1, or 2.")
    return idx


def _hermitian(matrix):
    matrix = np.asarray(matrix)
    return 0.5 * (matrix + matrix.conj().T)


def cap_operator_from_z(z, M=1, width=2.0, strength=0.005, order=2):
    """
    Build a diagonal complex-absorbing-potential profile W(z).

    The real-time Hamiltonian uses ``H_eff = H - i W`` with ``W >= 0`` near
    both z-grid edges.  ``width`` is measured inward from the first and last
    supplied grid points.
    """
    z = np.asarray(z, dtype=float).reshape(-1)
    M = int(M)
    width = float(width)
    strength = float(strength)
    order = int(order)
    if z.size == 0:
        raise ValueError("z grid must be non-empty.")
    if M <= 0:
        raise ValueError("M must be positive.")
    if width <= 0.0:
        raise ValueError("CAP width must be positive.")
    if strength < 0.0:
        raise ValueError("CAP strength must be non-negative.")
    if order < 1:
        raise ValueError("CAP order must be positive.")

    zmin = float(np.min(z))
    zmax = float(np.max(z))
    if width > 0.5 * (zmax - zmin):
        raise ValueError("CAP width must not exceed half the z-grid span.")

    left_start = zmin + width
    right_start = zmax - width
    profile = np.zeros_like(z)
    left = z < left_start
    right = z > right_start
    profile[left] = ((left_start - z[left]) / width) ** order
    profile[right] = ((z[right] - right_start) / width) ** order
    return np.diag(np.repeat(strength * profile, M))


class RTTDHF:
    """
    Real-time TDHF propagation for a converged GDVR RHF reference.

    The GDVR RHF basis is treated as orthonormal, matching the static GDVR SCF
    diagonalization. The density matrix evolves by a midpoint unitary step with
    the same collocated Coulomb/exchange Fock build used by GDVR RHF.
    """

    def __init__(self, mf, interaction=None, field=None, cap=None):
        if mf.mo_coeff is None or mf.dm is None:
            raise ValueError("Run GDVR RHF before starting real-time TDHF.")
        if mf.mol.hcore is None or mf.mol.eri_j is None or mf.mol.eri_k is None:
            raise ValueError("The GDVR molecule must contain hcore and ERI kernels.")
        if mf.mol.shapes is None:
            raise ValueError("The GDVR molecule must contain shape metadata.")

        self._scf = mf
        self.mol = mf.mol
        self.field = field
        self.interaction = None if interaction is None else np.asarray(interaction)

        self.Nz = int(self.mol.shapes["Nz"])
        self.M = int(self.mol.shapes["M"])
        self.size = self.Nz * self.M
        self.hcore = np.asarray(self.mol.hcore, dtype=float)
        if self.hcore.shape != (self.size, self.size):
            raise ValueError("GDVR hcore shape does not match Nz * M.")
        self._jk_builder = prepare_gdvr_fock_builder(
            self.mol.eri_j,
            self.mol.eri_k,
            self.Nz,
            self.M,
        )

        self.e_nuc = float(self.mol.nuclear_repulsion_energy())
        self.dm = _hermitian(np.asarray(mf.dm, dtype=complex))
        self.cap = self._operator_from_diagonal_or_matrix(cap, name="cap") if cap is not None else None

        self.times = None
        self.dms = None
        self.energies = None
        self.dipoles = None
        self.dipole_velocities = None
        self.dipole_accelerations = None
        self.fields = None
        self.electron_counts = None
        self.orbitals = None
        self.orbital_occupations = None
        self.propagation_method = None

    def _operator_from_diagonal_or_matrix(self, operator, name):
        op = np.asarray(operator)
        if op.ndim == 1:
            if op.size == self.Nz:
                op = np.repeat(op, self.M)
            if op.size != self.size:
                raise ValueError(f"{name} diagonal must have length Nz or Nz * M.")
            op = np.diag(op)
        if op.shape != (self.size, self.size):
            raise ValueError(f"{name} must have shape (N, N), length N, or length Nz.")
        op = _hermitian(op)
        if np.max(np.abs(op.imag)) < 1e-14:
            op = op.real
        if np.any(np.diag(op).real < -1e-14):
            raise ValueError(f"{name} must be a non-negative absorbing potential.")
        return np.asarray(op, dtype=complex)

    def position_operator(self, axis="z"):
        if hasattr(self.mol, "position_operator"):
            return self.mol.position_operator(axis)
        idx = _axis_to_index(axis)
        if idx != 2:
            raise NotImplementedError("GDVR currently has a built-in position operator only along z.")
        z = np.asarray(self.mol.z, dtype=float)
        if z.shape != (self.Nz,):
            raise ValueError("GDVR z grid is missing or has the wrong shape.")
        return np.diag(np.repeat(z, self.M))

    def get_interaction(self):
        if self.interaction is None:
            self.interaction = self.position_operator("z")
        op = np.asarray(self.interaction)
        if op.shape == (self.size, self.size):
            return _hermitian(op)
        if op.shape == (3, self.size, self.size):
            return np.stack([_hermitian(block) for block in op], axis=0)
        raise ValueError("interaction must have shape (N, N) or (3, N, N).")

    def field_vector(self, time, field=None):
        source = self.field if field is None else field
        if source is None:
            return np.zeros(3)

        value = source(time) if callable(source) else source
        vec = np.asarray(value, dtype=float)
        if vec.ndim == 0:
            out = np.zeros(3)
            out[2] = float(vec)
            return out

        vec = vec.reshape(-1)
        if vec.size != 3:
            raise ValueError("field must evaluate to a scalar or a length-3 vector.")
        return vec

    def _field_coupling_from_vector(self, field_vec):
        field_vec = np.asarray(field_vec, dtype=float)
        if not np.any(field_vec):
            return np.zeros_like(self.hcore, dtype=complex)

        interaction = self.get_interaction()
        if interaction.ndim == 2:
            return field_vec[2] * interaction
        return np.einsum("x,xij->ij", field_vec, interaction, optimize=True)

    def _field_coupling(self, time, field=None):
        return self._field_coupling_from_vector(self.field_vector(time, field=field))

    def field_free_fock(self, dm):
        dm = _hermitian(np.asarray(dm, dtype=complex))
        f2e = fock_2e_slice_collocated(
            dm,
            self._jk_builder,
            None,
            self.Nz,
            self.M,
            k_scale=1.0,
        )
        return _hermitian(self.hcore + f2e)

    def get_fock(self, dm, time=0.0, field=None):
        return _hermitian(self.field_free_fock(dm) - self._field_coupling(time, field=field))

    def get_effective_fock(self, dm, time=0.0, field=None):
        fock = np.asarray(self.get_fock(dm, time=time, field=field), dtype=complex)
        if self.cap is not None:
            fock = fock - 1j * self.cap
        return fock

    def energy(self, dm=None, field_free_fock=None):
        if dm is None:
            dm = self.dm
        dm = _hermitian(np.asarray(dm, dtype=complex))
        fock = self.field_free_fock(dm) if field_free_fock is None else field_free_fock
        return (0.5 * np.einsum("ij,ji->", self.hcore + fock, dm, optimize=True)).real + self.e_nuc

    def electron_count(self, dm=None):
        if dm is None:
            dm = self.dm
        return np.trace(np.asarray(dm)).real

    def dipole_moment(self, dm=None):
        if dm is None:
            dm = self.dm
        dm = np.asarray(dm)
        interaction = self.get_interaction()
        if interaction.ndim == 2:
            out = np.zeros(3)
            out[2] = np.einsum("ij,ji->", interaction, dm, optimize=True).real
            return out
        return np.einsum("xij,ji->x", interaction, dm, optimize=True).real

    def dipole_velocity(self, dm=None, time=0.0, field=None, fock=None):
        if dm is None:
            dm = self.dm
        dm = np.asarray(dm)
        if fock is None:
            fock = self.get_fock(dm, time=time, field=field)
        interaction = self.get_interaction()

        def _one(op):
            comm = op @ fock - fock @ op
            return (-1j * np.einsum("ij,ji->", comm, dm, optimize=True)).real

        if interaction.ndim == 2:
            out = np.zeros(3)
            out[2] = _one(interaction)
            return out
        return np.array([_one(op) for op in interaction], dtype=float)

    def dipole_acceleration_force(self, dm=None, time=0.0, field=None, fock=None):
        if dm is None:
            dm = self.dm
        dm = np.asarray(dm)
        if fock is None:
            fock = self.get_fock(dm, time=time, field=field)
        interaction = self.get_interaction()

        def _one(op):
            comm = op @ fock - fock @ op
            double_comm = comm @ fock - fock @ comm
            return (-np.einsum("ij,ji->", double_comm, dm, optimize=True)).real

        if interaction.ndim == 2:
            out = np.zeros(3)
            out[2] = _one(interaction)
            return out
        return np.array([_one(op) for op in interaction], dtype=float)

    def sample_observables(self, dm, time=0.0, field=None, include_velocity=True):
        dm = _hermitian(np.asarray(dm, dtype=complex))
        field_vec = self.field_vector(time, field=field)
        field_free_fock = self.field_free_fock(dm)
        fock = _hermitian(field_free_fock - self._field_coupling_from_vector(field_vec))
        out = {
            "energy": self.energy(dm, field_free_fock=field_free_fock),
            "dipole": self.dipole_moment(dm),
            "dipole_acceleration": self.dipole_acceleration_force(dm, fock=fock),
            "field": field_vec,
            "electron_count": self.electron_count(dm),
        }
        if include_velocity:
            out["dipole_velocity"] = self.dipole_velocity(dm, fock=fock)
        return out

    def occupied_orbitals(self, dm0=None, thresh=1e-10, return_occupations=False):
        """Return orthonormal occupied orbitals, optionally with occupations."""
        if dm0 is not None:
            evals, evecs = np.linalg.eigh(_hermitian(np.asarray(dm0, dtype=complex)))
            keep = evals > float(thresh)
            if not np.any(keep):
                raise ValueError("dm0 contains no occupied natural orbitals.")
            orbitals = evecs[:, keep]
            occupations = evals[keep].real
            if return_occupations:
                return orbitals, occupations
            return orbitals

        coeff = np.asarray(self._scf.mo_coeff, dtype=complex)
        occ = np.asarray(self._scf.mo_occ, dtype=float)
        if coeff.shape != (self.size, self.size) or occ.shape != (self.size,):
            raise ValueError("GDVR RHF orbital coefficients/occupations have inconsistent shapes.")
        keep = occ > float(thresh)
        if not np.any(keep):
            raise ValueError("GDVR RHF reference has no occupied orbitals.")
        orbitals = coeff[:, keep]
        occupations = occ[keep]
        if return_occupations:
            return orbitals, occupations
        return orbitals

    @staticmethod
    def density_from_orbitals(orbitals, occupations=None):
        """Build ``D = C diag(occupations) C^H`` from orbital columns."""
        orbitals = np.asarray(orbitals, dtype=complex)
        if orbitals.ndim != 2:
            raise ValueError("orbitals must be a two-dimensional coefficient matrix.")
        if occupations is None:
            return _hermitian(orbitals @ orbitals.conj().T)

        occupations = np.asarray(occupations, dtype=float).reshape(-1)
        if occupations.shape != (orbitals.shape[1],):
            raise ValueError("occupations must have one entry per orbital column.")
        return _hermitian((orbitals * occupations[np.newaxis, :]) @ orbitals.conj().T)

    def kick(self, dm=None, strength=1e-4, axis="z", interaction=None):
        if dm is None:
            dm = self.dm
        op = self.get_interaction() if interaction is None else np.asarray(interaction)
        if op.ndim == 3:
            op = op[_axis_to_index(axis)]
        op = _hermitian(op)
        u = expm(-1j * float(strength) * op)
        return _hermitian(u @ np.asarray(dm, dtype=complex) @ u.conj().T)

    def kick_orbitals(self, orbitals, strength=1e-4, axis="z", interaction=None):
        op = self.get_interaction() if interaction is None else np.asarray(interaction)
        if op.ndim == 3:
            op = op[_axis_to_index(axis)]
        op = _hermitian(op)
        u = expm(-1j * float(strength) * op)
        return u @ np.asarray(orbitals, dtype=complex)

    def step(self, dm, time, dt, field=None):
        dm = _hermitian(np.asarray(dm, dtype=complex))
        fock_0 = self.get_effective_fock(dm, time=time, field=field)
        u_half = expm(-0.5j * float(dt) * fock_0)
        dm_half = _hermitian(u_half @ dm @ u_half.conj().T)

        fock_half = self.get_effective_fock(dm_half, time=time + 0.5 * float(dt), field=field)
        u = expm(-1j * float(dt) * fock_half)
        return _hermitian(u @ dm @ u.conj().T)

    def step_orbitals(self, orbitals, time, dt, field=None, occupations=None):
        orbitals = np.asarray(orbitals, dtype=complex)
        dm = self.density_from_orbitals(orbitals, occupations=occupations)
        fock_0 = self.get_effective_fock(dm, time=time, field=field)
        orbitals_half = expm_multiply(-0.5j * float(dt) * fock_0, orbitals)

        dm_half = self.density_from_orbitals(orbitals_half, occupations=occupations)
        fock_half = self.get_effective_fock(dm_half, time=time + 0.5 * float(dt), field=field)
        return expm_multiply(-1j * float(dt) * fock_half, orbitals)

    def _allocate_trajectory(self, nsteps, store_dm, store_orbitals=False, norb=None):
        dms = None
        if store_dm:
            dms = np.zeros((nsteps + 1, self.size, self.size), dtype=complex)
        orbitals = None
        if store_orbitals:
            if norb is None:
                raise ValueError("norb is required when store_orbitals=True.")
            orbitals = np.zeros((nsteps + 1, self.size, int(norb)), dtype=complex)
        return dms, orbitals

    def _finalize_run(
        self,
        times,
        dms,
        orbitals,
        energies,
        dipoles,
        dipole_velocities,
        dipole_accelerations,
        fields,
        electron_counts,
        dm,
        method,
        orbital_occupations=None,
    ):
        self.times = times
        self.dms = dms
        self.orbitals = orbitals
        self.energies = energies
        self.dipoles = dipoles
        self.dipole_velocities = dipole_velocities
        self.dipole_accelerations = dipole_accelerations
        self.fields = fields
        self.electron_counts = electron_counts
        self.dm = dm
        self.propagation_method = str(method)
        if orbital_occupations is None:
            self.orbital_occupations = None
        else:
            self.orbital_occupations = np.asarray(orbital_occupations, dtype=float).copy()
        return self

    def run_density(self, dt, nsteps, dm0=None, field=None, t0=0.0, store_dm=True, kick=None):
        dt = float(dt)
        nsteps = int(nsteps)
        if nsteps < 0:
            raise ValueError("nsteps must be non-negative.")

        dm = _hermitian(np.asarray(self._scf.dm if dm0 is None else dm0, dtype=complex))
        if kick is not None:
            dm = self.kick(dm=dm, **kick)

        times = float(t0) + dt * np.arange(nsteps + 1, dtype=float)
        dms, orbitals = self._allocate_trajectory(nsteps, store_dm, store_orbitals=False)
        energies = np.zeros(nsteps + 1, dtype=float)
        dipoles = np.zeros((nsteps + 1, 3), dtype=float)
        dipole_velocities = np.zeros((nsteps + 1, 3), dtype=float)
        dipole_accelerations = np.zeros((nsteps + 1, 3), dtype=float)
        fields = np.zeros((nsteps + 1, 3), dtype=float)
        electron_counts = np.zeros(nsteps + 1, dtype=float)

        for istep, time in enumerate(times):
            if store_dm:
                dms[istep] = dm
            obs = self.sample_observables(dm, time=time, field=field)
            energies[istep] = obs["energy"]
            dipoles[istep] = obs["dipole"]
            dipole_velocities[istep] = obs["dipole_velocity"]
            dipole_accelerations[istep] = obs["dipole_acceleration"]
            fields[istep] = obs["field"]
            electron_counts[istep] = obs["electron_count"]

            if istep < nsteps:
                dm = self.step(dm, time, dt, field=field)

        return self._finalize_run(
            times,
            dms,
            orbitals,
            energies,
            dipoles,
            dipole_velocities,
            dipole_accelerations,
            fields,
            electron_counts,
            dm,
            method="density",
        )

    def run_orbitals(
        self,
        dt,
        nsteps,
        dm0=None,
        field=None,
        t0=0.0,
        store_dm=True,
        store_orbitals=False,
        kick=None,
    ):
        dt = float(dt)
        nsteps = int(nsteps)
        if nsteps < 0:
            raise ValueError("nsteps must be non-negative.")

        orbitals_current, occupations = self.occupied_orbitals(dm0=dm0, return_occupations=True)
        if kick is not None:
            orbitals_current = self.kick_orbitals(orbitals_current, **kick)

        times = float(t0) + dt * np.arange(nsteps + 1, dtype=float)
        dms, orbitals = self._allocate_trajectory(
            nsteps,
            store_dm,
            store_orbitals=store_orbitals,
            norb=orbitals_current.shape[1],
        )
        energies = np.zeros(nsteps + 1, dtype=float)
        dipoles = np.zeros((nsteps + 1, 3), dtype=float)
        dipole_velocities = np.zeros((nsteps + 1, 3), dtype=float)
        dipole_accelerations = np.zeros((nsteps + 1, 3), dtype=float)
        fields = np.zeros((nsteps + 1, 3), dtype=float)
        electron_counts = np.zeros(nsteps + 1, dtype=float)

        dm = None
        for istep, time in enumerate(times):
            dm = self.density_from_orbitals(orbitals_current, occupations=occupations)
            if store_dm:
                dms[istep] = dm
            if store_orbitals:
                orbitals[istep] = orbitals_current
            obs = self.sample_observables(dm, time=time, field=field)
            energies[istep] = obs["energy"]
            dipoles[istep] = obs["dipole"]
            dipole_velocities[istep] = obs["dipole_velocity"]
            dipole_accelerations[istep] = obs["dipole_acceleration"]
            fields[istep] = obs["field"]
            electron_counts[istep] = obs["electron_count"]

            if istep < nsteps:
                orbitals_current = self.step_orbitals(
                    orbitals_current,
                    time,
                    dt,
                    field=field,
                    occupations=occupations,
                )

        return self._finalize_run(
            times,
            dms,
            orbitals,
            energies,
            dipoles,
            dipole_velocities,
            dipole_accelerations,
            fields,
            electron_counts,
            dm,
            method="orbital",
            orbital_occupations=occupations,
        )

    def run(
        self,
        dt,
        nsteps,
        dm0=None,
        field=None,
        t0=0.0,
        store_dm=True,
        kick=None,
        method="density",
        store_orbitals=False,
    ):
        key = str(method).strip().lower()
        if key in {"density", "dm"}:
            if store_orbitals:
                raise ValueError("store_orbitals is available only with method='orbital'.")
            return self.run_density(dt, nsteps, dm0=dm0, field=field, t0=t0, store_dm=store_dm, kick=kick)
        if key in {"orbital", "orbitals", "krylov", "expm_multiply"}:
            return self.run_orbitals(
                dt,
                nsteps,
                dm0=dm0,
                field=field,
                t0=t0,
                store_dm=store_dm,
                store_orbitals=store_orbitals,
                kick=kick,
            )
        raise ValueError("method must be 'density' or 'orbital'.")
