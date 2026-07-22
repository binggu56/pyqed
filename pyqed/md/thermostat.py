"""Temperature-coupling helpers for molecular dynamics."""

from __future__ import annotations

import numpy as np


class BerendsenThermostat:
    """Weak velocity-rescaling thermostat for equilibration callbacks.

    The callback form is intended for :class:`pyqed.md.MDEngine`, where it is
    called every ``interval`` MD steps.  ``timestep_fs`` should be the base MD
    timestep, not the callback period.
    """

    def __init__(
        self,
        atoms,
        target_temperature_K,
        tau_fs,
        timestep_fs,
        interval=1,
        remove_center_of_mass=True,
        min_scale=0.8,
        max_scale=1.25,
    ):
        target_temperature_K = float(target_temperature_K)
        tau_fs = float(tau_fs)
        timestep_fs = float(timestep_fs)
        interval = int(interval)
        min_scale = float(min_scale)
        max_scale = float(max_scale)
        if target_temperature_K <= 0.0:
            raise ValueError("target_temperature_K must be positive.")
        if tau_fs <= 0.0:
            raise ValueError("tau_fs must be positive.")
        if timestep_fs <= 0.0:
            raise ValueError("timestep_fs must be positive.")
        if interval <= 0:
            raise ValueError("interval must be positive.")
        if min_scale <= 0.0 or max_scale <= 0.0 or min_scale > max_scale:
            raise ValueError("min_scale and max_scale must be positive and ordered.")

        self.atoms = atoms
        self.target_temperature_K = target_temperature_K
        self.tau_fs = tau_fs
        self.timestep_fs = timestep_fs
        self.interval = interval
        self.remove_center_of_mass = bool(remove_center_of_mass)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.calls = 0
        self.last_temperature_K = None
        self.last_scale = None

    def __call__(self):
        return self.apply()

    def apply(self):
        """Apply one Berendsen temperature-coupling update."""
        current = float(
            self.atoms.get_temperature(remove_center_of_mass=self.remove_center_of_mass)
        )
        self.last_temperature_K = current
        if current <= 0.0 or not np.isfinite(current):
            self.last_scale = 1.0
            self.calls += 1
            return 1.0

        coupling = min((self.timestep_fs * self.interval) / self.tau_fs, 1.0)
        scale_squared = 1.0 + coupling * (self.target_temperature_K / current - 1.0)
        if scale_squared <= 0.0 or not np.isfinite(scale_squared):
            scale = self.min_scale
        else:
            scale = float(np.sqrt(scale_squared))
            scale = min(max(scale, self.min_scale), self.max_scale)

        self.atoms.set_momenta(
            self.atoms.get_momenta() * scale,
            apply_constraint=False,
        )
        self.last_scale = scale
        self.calls += 1
        return scale
