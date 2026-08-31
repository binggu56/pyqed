"""Time-dependent nuclear wavepacket scattering."""

from __future__ import annotations

import numpy as np

from pyqed.mps.decompose import decompose
from pyqed.mps.mps import MPS, MPO
from pyqed.mps.tdmps import TDMPS


def _full_rank(shape):
    shape = tuple(int(value) for value in shape)
    return max(
        [1]
        + [
            min(int(np.prod(shape[:split])), int(np.prod(shape[split:])))
            for split in range(1, len(shape))
        ]
    )


def _diagonal_mpo(values):
    values = np.asarray(values)
    cores = decompose(values, rank=_full_rank(values.shape))
    factors = []
    for core in cores:
        left, dim, right = core.shape
        factor = np.zeros((left, right, dim, dim), dtype=core.dtype)
        diagonal = np.arange(dim)
        factor[:, :, diagonal, diagonal] = core.transpose(0, 2, 1)
        factors.append(factor)
    return MPO(factors)


def _operator(value, dims, *, name, nonnegative=False):
    if isinstance(value, MPO):
        if tuple(value.dims) != dims or tuple(value.input_dims) != dims:
            raise ValueError(f"{name} MPO dimensions must be {dims}")
        return value

    values = np.asarray(value)
    if values.shape != dims:
        raise ValueError(f"{name} shape {values.shape} != {dims}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if np.max(np.abs(np.imag(values))) > 1.0e-13:
        raise ValueError(f"diagonal {name} must be real")
    values = np.real(values)
    if nonnegative and np.min(values) < -1.0e-13:
        raise ValueError("absorber must be nonnegative")
    return _diagonal_mpo(values)


class _TDDMRGRun:
    def __init__(
        self,
        scattering,
        *,
        max_bond,
        cutoff,
        integrator,
        krylov_dim,
        krylov_tol,
        normalize,
    ):
        self.scattering = scattering
        self.max_bond = int(max_bond)
        self.cutoff = float(cutoff)
        self.integrator = str(integrator)
        self.krylov_dim = int(krylov_dim)
        self.krylov_tol = float(krylov_tol)
        self.normalize = bool(normalize)
        if self.max_bond <= 0:
            raise ValueError("max_bond must be positive")
        if self.cutoff < 0.0:
            raise ValueError("cutoff must be nonnegative")
        if self.krylov_dim <= 0:
            raise ValueError("krylov_dim must be positive")
        if self.krylov_tol <= 0.0:
            raise ValueError("krylov_tol must be positive")

    def run(
        self,
        psi0,
        *,
        dt,
        steps,
        observables=None,
        interval=1,
        t0=0.0,
        progress=True,
        **kwargs,
    ):
        """Propagate the incoming MPS and populate the scattering calculation."""
        if not isinstance(psi0, MPS):
            raise TypeError("psi0 must be an MPS")
        if tuple(psi0.dims) != tuple(self.scattering.hamiltonian.input_dims):
            raise ValueError(
                f"psi0 dimensions {tuple(psi0.dims)} != "
                f"{tuple(self.scattering.hamiltonian.input_dims)}"
            )
        dt = float(dt)
        steps = int(steps)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if steps < 0:
            raise ValueError("steps must be nonnegative")

        dynamics = TDMPS(
            self.scattering.hamiltonian,
            D=self.max_bond,
            normalize=self.normalize,
        )
        kwargs.setdefault("track_energy", not self.scattering.has_absorber)
        dynamics.run(
            psi0.copy(),
            dt=dt,
            steps=steps,
            e_ops=[] if observables is None else tuple(observables),
            interval=interval,
            t0=t0,
            integrator=self.integrator,
            cutoff=self.cutoff,
            krylov_dim=self.krylov_dim,
            krylov_tol=self.krylov_tol,
            krylov_method=("arnoldi" if self.scattering.has_absorber else "lanczos"),
            progress=progress,
            **kwargs,
        )

        calculation = self.scattering
        calculation.driver = dynamics
        calculation.final_state = dynamics.final_state
        calculation.times = np.concatenate(([float(t0)], dynamics.times))
        calculation.observables = dynamics.observables
        checkpoint_steps = np.rint((dynamics.times - float(t0)) / dt).astype(int)
        checkpoint_norms = dynamics.pre_normalization_norms[checkpoint_steps - 1]
        calculation.norms = np.concatenate(
            ([float(np.sqrt(np.real(psi0.norm_squared())))], checkpoint_norms)
        )
        calculation.success = True
        calculation.message = "TDDMRG propagation completed"
        return calculation


class WavepacketScattering:
    r"""Single-surface wavepacket scattering with

    .. math:: H = T + V - iW.

    ``T`` must be an MPO. Diagonal ``V`` and ``W`` may be supplied either as
    grid tensors or as MPOs. ``W`` is a nonnegative complex absorbing
    potential.
    """

    def __init__(self, *, kinetic, potential, absorber=None):
        if not isinstance(kinetic, MPO):
            raise TypeError("kinetic must be an MPO")
        if kinetic.dims != kinetic.input_dims:
            raise ValueError("kinetic must be a square MPO")
        self.dims = tuple(int(value) for value in kinetic.dims)
        self.kinetic = kinetic
        self.potential = _operator(potential, self.dims, name="potential")
        self.absorber = (
            None
            if absorber is None
            else _operator(absorber, self.dims, name="absorber", nonnegative=True)
        )
        self.has_absorber = self.absorber is not None
        self.hamiltonian = self.kinetic + self.potential
        if self.absorber is not None:
            self.hamiltonian = self.hamiltonian + (-1j) * self.absorber

        self.driver = None
        self.final_state = None
        self.times = None
        self.observables = None
        self.norms = None
        self.success = False
        self.message = "not run"

    def tddmrg(
        self,
        *,
        max_bond=128,
        cutoff=1.0e-10,
        integrator="tdvp2",
        krylov_dim=12,
        krylov_tol=1.0e-12,
        normalize=None,
    ):
        """Bind the existing MPS time propagator to this scattering problem."""
        if normalize is None:
            normalize = not self.has_absorber
        if self.has_absorber and normalize:
            raise ValueError("CAP propagation cannot normalize after each step")
        return _TDDMRGRun(
            self,
            max_bond=max_bond,
            cutoff=cutoff,
            integrator=integrator,
            krylov_dim=krylov_dim,
            krylov_tol=krylov_tol,
            normalize=normalize,
        )


__all__ = ["WavepacketScattering"]
