"""PyQED molecular-dynamics engine façade."""

from dataclasses import dataclass

from .io import EnergyLogger, XYZTrajectoryWriter
from .langevin import Langevin
from .restart import write_restart
from .verlet import VelocityVerlet
from pyqed.units import fs


@dataclass
class MDState:
    """Compact snapshot of a PyQED MD run."""

    step: int
    time: float
    potential_energy: float
    kinetic_energy: float
    total_energy: float
    temperature_K: float


class MDEngine:
    """Small MD run-loop around PyQED atoms, calculators, and integrators.

    The engine intentionally wraps the existing tested integrators rather than
    replacing them.  It provides one stable place for logging, trajectory
    writing, callbacks, restarts, and ensemble selection.
    """

    def __init__(
        self,
        atoms,
        timestep,
        ensemble="nve",
        temperature_K=None,
        friction=None,
        friction_per_ps=None,
        trajectory=None,
        trajectory_interval=1,
        logfile=None,
        log_interval=1,
        restart=None,
        restart_interval=None,
        callbacks=None,
    ):
        self.atoms = atoms
        self.timestep = float(timestep)
        self.ensemble = str(ensemble).lower()
        self.restart = restart
        self.restart_interval = _optional_positive_interval(restart_interval, "restart_interval")
        trajectory_interval = _positive_interval(trajectory_interval, "trajectory_interval")
        log_interval = _positive_interval(log_interval, "log_interval")
        self._managed_observers = []
        self.dynamics = self._make_dynamics(
            temperature_K=temperature_K,
            friction=friction,
            friction_per_ps=friction_per_ps,
        )

        if trajectory is not None:
            writer = XYZTrajectoryWriter(atoms, trajectory, dynamics=self.dynamics)
            self._managed_observers.append(writer)
            self.dynamics.attach(writer, interval=trajectory_interval)
        if logfile is not None:
            logger = EnergyLogger(atoms, logfile, dynamics=self.dynamics)
            self._managed_observers.append(logger)
            self.dynamics.attach(logger, interval=log_interval)
        for callback in callbacks or ():
            if isinstance(callback, tuple):
                function, interval = callback
                self.attach(function, interval=interval)
            else:
                self.attach(callback)
        if self.restart is not None and self.restart_interval is not None:
            self.dynamics.attach(self._write_restart_observer, interval=self.restart_interval)

    @property
    def step_index(self):
        return self.dynamics.get_number_of_steps()

    @property
    def time(self):
        return self.dynamics.get_time()

    def attach(self, callback, interval=1, *args, **kwargs):
        """Attach a callback that is called every ``interval`` MD steps."""
        self.dynamics.attach(callback, interval=_positive_interval(interval, "interval"), *args, **kwargs)

    def step(self, steps=1):
        """Advance the simulation by ``steps`` integration steps."""
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        self.dynamics.run(steps)
        return self.state()

    def run(self, steps):
        """Run MD and return the final :class:`MDState`."""
        return self.step(steps)

    def state(self):
        """Return energy, temperature, step, and time for the current snapshot."""
        potential = float(self.atoms.get_potential_energy())
        kinetic = float(self.atoms.get_kinetic_energy())
        return MDState(
            step=int(self.step_index),
            time=float(self.time),
            potential_energy=potential,
            kinetic_energy=kinetic,
            total_energy=potential + kinetic,
            temperature_K=float(self.atoms.get_temperature()),
        )

    def write_restart(self, filename=None):
        """Write a restart file for the current MD state."""
        target = self.restart if filename is None else filename
        if target is None:
            raise ValueError("filename is required when no restart path was configured.")
        write_restart(
            self.atoms,
            target,
            step=self.step_index,
            time=self.time,
            metadata={"engine": "pyqed", "ensemble": self.ensemble},
        )

    def close(self):
        """Close managed log and trajectory writers."""
        for observer in self._managed_observers:
            close = getattr(observer, "close", None)
            if close is not None:
                close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def _make_dynamics(self, temperature_K=None, friction=None, friction_per_ps=None):
        if self.ensemble in {"nve", "microcanonical"}:
            return VelocityVerlet(self.atoms, self.timestep)
        if self.ensemble in {"nvt", "langevin"}:
            if temperature_K is None:
                raise ValueError("temperature_K is required for Langevin/NVT MD.")
            if friction is not None and friction_per_ps is not None:
                raise ValueError("Specify only one of friction or friction_per_ps.")
            if friction is None:
                if friction_per_ps is None:
                    raise ValueError("friction or friction_per_ps is required for Langevin/NVT MD.")
                friction = friction_ps_to_atomic_units(friction_per_ps)
            return Langevin(
                self.atoms,
                self.timestep,
                temperature_K=temperature_K,
                friction=friction,
            )
        raise ValueError("ensemble must be 'nve' or 'langevin'.")

    def _write_restart_observer(self):
        self.write_restart()


def _positive_interval(value, name):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _optional_positive_interval(value, name):
    if value is None:
        return None
    return _positive_interval(value, name)


def friction_ps_to_atomic_units(friction_per_ps):
    """Convert a friction coefficient in ps^-1 to inverse atomic time."""
    return float(friction_per_ps) / (1000.0 * fs)
