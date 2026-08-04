"""High-level ab initio strong-field drivers."""

from collections.abc import Mapping

import numpy as np


def _axis_index(axis):
    if isinstance(axis, str):
        key = axis.strip().lower()
        if key in {"x", "y", "z"}:
            return {"x": 0, "y": 1, "z": 2}[key]
        raise ValueError("axis must be 'x', 'y', or 'z'.")
    index = int(axis)
    if index not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    return index


def _field_frequency(field):
    for name in ("omega", "omegac", "frequency", "freq"):
        value = getattr(field, name, None)
        if value is not None:
            return float(value)
    return None


class HHG:
    """Fixed-nuclei ab initio high-harmonic-generation calculation."""

    def __init__(
        self,
        mol,
        *,
        method="rt-tdhf",
        field,
        omega=None,
        axis="z",
        cap=None,
        reference=None,
        reference_options=None,
        window="hann",
        zero_pad=4,
    ):
        self.mol = mol
        self.method = str(method).strip().lower().replace("_", "-")
        self.field = field
        self.omega0 = _field_frequency(field) if omega is None else float(omega)
        self.axis = axis
        self.cap_options = cap
        self.reference = reference
        self.reference_options = dict(reference_options or {})
        self.window = str(window).strip().lower()
        self.zero_pad = int(zero_pad)

        self.dynamics = None
        self.trajectory = None
        self.time = None
        self.dipole = None
        self.acceleration = None
        self.frequency = None
        self.harmonic_order = None
        self.intensity = None
        self.normalized_intensity = None

        if self.method not in {"rt-tdhf", "rttdhf"}:
            raise NotImplementedError(
                f"HHG method {method!r} is not implemented; use 'rt-tdhf'."
            )
        if self.omega0 is None or self.omega0 <= 0.0:
            raise ValueError(
                "Provide omega or use a field with an omega/frequency attribute."
            )
        if self.zero_pad < 1:
            raise ValueError("zero_pad must be at least 1.")
        _axis_index(axis)

    def _build_reference(self):
        if self.reference is None:
            self.reference = self.mol.RHF().run(**self.reference_options)
        return self.reference

    def _build_cap(self):
        cap = self.cap_options
        if cap is None or cap is False:
            return None
        if cap is True:
            return self.mol.cap()
        if isinstance(cap, Mapping):
            return self.mol.cap(**cap)
        return np.asarray(cap)

    def _window_values(self, size):
        if self.window in {"hann", "hanning"}:
            return np.hanning(size)
        if self.window in {"none", "boxcar", "rectangular"}:
            return np.ones(size)
        raise ValueError("window must be 'hann' or 'none'.")

    def _analyze(self):
        time = np.asarray(self.trajectory.times, dtype=float)
        if time.size < 4:
            raise ValueError("HHG analysis requires at least four time samples.")
        dt = np.diff(time)
        if not np.allclose(dt, dt[0], rtol=1.0e-8, atol=1.0e-12):
            raise ValueError("HHG analysis requires uniformly spaced time samples.")

        axis = _axis_index(self.axis)
        dipole = np.asarray(self.trajectory.dipoles[:, axis], dtype=float)
        accelerations = getattr(self.trajectory, "dipole_accelerations", None)
        if accelerations is None:
            acceleration = np.gradient(np.gradient(dipole, dt[0]), dt[0])
        else:
            acceleration = np.asarray(accelerations[:, axis], dtype=float)

        signal = acceleration - float(np.mean(acceleration))
        nfft = self.zero_pad * time.size
        frequency = 2.0 * np.pi * np.fft.rfftfreq(nfft, d=float(dt[0]))
        intensity = np.abs(
            np.fft.rfft(signal * self._window_values(time.size), n=nfft)
        ) ** 2
        normalized = intensity.copy()
        scale = float(np.max(normalized[1:])) if normalized.size > 1 else 0.0
        if scale > 0.0:
            normalized /= scale

        self.time = time
        self.dipole = dipole
        self.acceleration = acceleration
        self.frequency = frequency
        self.harmonic_order = frequency / self.omega0
        self.intensity = intensity
        self.normalized_intensity = normalized

    def run(
        self,
        *,
        dt,
        nsteps=None,
        t_final=None,
        t0=0.0,
        store_dm=False,
        kick=None,
        propagator="density",
        store_orbitals=False,
    ):
        """Build the reference, propagate it, and calculate the HHG spectrum."""
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if nsteps is None:
            if t_final is None:
                t_final = getattr(self.field, "duration", None)
            if t_final is None:
                raise ValueError(
                    "Provide nsteps or t_final, or use a field with a duration attribute."
                )
            nsteps = int(np.ceil((float(t_final) - float(t0)) / dt))
        nsteps = int(nsteps)
        if nsteps < 1:
            raise ValueError("nsteps must be positive.")

        reference = self._build_reference()
        self.dynamics = reference.RTTDHF(
            interaction=self.mol.dipole_operator(self.axis),
            field=self.field,
            cap=self._build_cap(),
        )
        self.trajectory = self.dynamics.run(
            dt=dt,
            nsteps=nsteps,
            t0=t0,
            store_dm=store_dm,
            kick=kick,
            method=propagator,
            store_orbitals=store_orbitals,
        )
        self._analyze()
        return self
