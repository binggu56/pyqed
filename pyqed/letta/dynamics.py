"""Stateful real-time drivers for native LETTA TDVP."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
import pickle

from .core import LETTA
from .tdvp import LETTATDVPEngine, letta_structural_rank_caps


def resolve_letta_backend(operator, backend="auto", *, device=None, max_bond=None):
    """Resolve ``auto`` to NumPy or PyTorch without importing Torch eagerly."""
    key = str(backend).lower()
    if key not in {"auto", "numpy", "torch"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'.")
    if key != "auto":
        return key
    try:
        import torch  # noqa: F401
    except ImportError:
        return "numpy"
    if device is not None and str(device) != "cpu":
        return "torch"
    if max_bond is None:
        return "numpy"
    if isinstance(max_bond, Integral):
        maximum = int(max_bond)
    else:
        finite = [int(value) for value in max_bond if value is not None]
        maximum = max(finite, default=1)
    workload = operator.nsites * max(operator.dims) ** 2
    return "torch" if workload >= 512 and maximum >= 6 else "numpy"


class TDVP:
    """Backend-dispatching window-2 LETTA TDVP engine."""

    def __init__(
        self,
        operator,
        *,
        backend="auto",
        device=None,
        torch_num_threads=None,
        channel_mode="auto",
        **options,
    ):
        self.source_operator = operator
        self.requested_backend = str(backend).lower()
        self.backend = resolve_letta_backend(
            operator,
            self.requested_backend,
            device=device,
            max_bond=options.get("max_bond"),
        )
        if self.backend == "torch":
            from .torch_tdvp import TorchTDVP

            self.implementation = TorchTDVP(
                operator,
                device=device,
                num_threads=torch_num_threads,
                channel_mode=channel_mode,
                **options,
            )
        else:
            self.implementation = LETTATDVPEngine(
                operator, channel_mode=channel_mode, **options
            )

    def __getattr__(self, name):
        implementation = self.__dict__.get("implementation")
        if implementation is None:
            raise AttributeError(name)
        return getattr(implementation, name)

    @property
    def state(self):
        return self.implementation.state

    @state.setter
    def state(self, value):
        self.implementation.state = value

    @property
    def prepared(self):
        return self.implementation.prepared

    @prepared.setter
    def prepared(self, value):
        self.implementation.prepared = bool(value)

    @property
    def integrator(self):
        return self.implementation.integrator

    def prepare_state(self, state):
        method = getattr(self.implementation, "prepare_state", None)
        return state if method is None else method(state)

    def set_integrator(self, value):
        self.implementation.set_integrator(value)
        return self

    def reset(self):
        self.implementation.reset()
        return self

    def step(self, state, dt, *, normalize=False, return_info=True):
        return self.implementation.step(
            state, dt, normalize=normalize, return_info=return_info
        )


class LETTAEvolution:
    """Hybrid two-site-to-one-site LETTA time evolution.

    Two-site TDVP grows the virtual ranks.  Once every bond has reached its
    structural/rank cap for ``saturation_steps`` consecutive steps, or once
    ``force_switch_time`` is reached, subsequent steps use fixed-rank
    one-site TDVP.
    """

    def __init__(
        self,
        operator,
        *,
        max_bond,
        cutoff=0.0,
        krylov_dim=20,
        krylov_tol=1.0e-10,
        saturation_steps=4,
        force_switch_time=None,
        backend="auto",
        device=None,
        torch_num_threads=None,
        channel_mode="auto",
    ):
        self.operator = operator
        self.max_bond = int(max_bond)
        if self.max_bond < 1:
            raise ValueError("max_bond must be positive.")
        self.cutoff = float(cutoff)
        self.krylov_dim = int(krylov_dim)
        self.krylov_tol = float(krylov_tol)
        self.saturation_steps = int(saturation_steps)
        if self.saturation_steps < 1:
            raise ValueError("saturation_steps must be positive.")
        self.force_switch_time = (
            None if force_switch_time is None else float(force_switch_time)
        )
        self.requested_backend = str(backend).lower()
        self.backend = resolve_letta_backend(
            operator, self.requested_backend, device=device,
            max_bond=self.max_bond,
        )
        self.device = None if device is None else str(device)
        self.torch_num_threads = (
            None if torch_num_threads is None else int(torch_num_threads)
        )
        self.channel_mode = str(channel_mode).lower()
        self.rank_caps = letta_structural_rank_caps(operator.dims, self.max_bond)
        options = {
            "integrator": "tdvp2",
            "max_bond": self.max_bond,
            "cutoff": self.cutoff,
            "krylov_dim": self.krylov_dim,
            "krylov_tol": self.krylov_tol,
        }
        self.engine = TDVP(
            operator,
            backend=self.backend,
            device=self.device,
            torch_num_threads=self.torch_num_threads,
            channel_mode=self.channel_mode,
            **options,
        )
        self.state = None
        self.time = 0.0
        self.step_index = 0
        self.mode = "tdvp2"
        self.saturation_streak = 0
        self.switch_step = None
        self.switch_time = None
        self.switch_reason = None
        self.history = []
        self.observations = []
        self.success = False
        self.message = "not started"

    def _consider_switch(self, ranks):
        if self.mode != "tdvp2":
            return
        saturated = all(rank >= cap for rank, cap in zip(ranks, self.rank_caps))
        self.saturation_streak = self.saturation_streak + 1 if saturated else 0
        reason = None
        if self.saturation_streak >= self.saturation_steps:
            reason = "rank saturation"
        if self.force_switch_time is not None and self.time >= self.force_switch_time:
            reason = "forced switch time"
        if reason is not None:
            self.mode = "tdvp1"
            self.engine.set_integrator("tdvp1")
            self.switch_step = self.step_index
            self.switch_time = self.time
            self.switch_reason = reason

    def step(self, state, dt, *, normalize=False):
        """Advance ``state`` once and return ``(state, diagnostics)``."""
        mode_used = self.mode
        output, info = self.engine.step(state, dt, normalize=normalize, return_info=True)
        self.state = output
        self.step_index += 1
        self.time += float(dt)
        ranks = tuple(info["ranks"])
        self._consider_switch(ranks)
        info = {
            **info,
            "step": self.step_index,
            "time": self.time,
            "mode_used": mode_used,
            "next_mode": self.mode,
            "saturation_streak": self.saturation_streak,
        }
        self.history.append(info)
        return output, info

    def run(
        self,
        state,
        dt,
        nsteps,
        *,
        observer=None,
        normalize=False,
        checkpoint=None,
        checkpoint_interval=None,
    ):
        """Advance for ``nsteps`` and retain states/diagnostics on the driver."""
        if not isinstance(state, LETTA) and not hasattr(state, "to_letta"):
            raise TypeError("state must be a LETTA or backend LETTA state.")
        nsteps = int(nsteps)
        if nsteps < 0:
            raise ValueError("nsteps must be non-negative.")
        current = state
        if observer is not None and not self.observations:
            self.observations.append(observer(self.time, current))
        for _ in range(nsteps):
            current, _ = self.step(current, dt, normalize=normalize)
            if observer is not None:
                self.observations.append(observer(self.time, current))
            if (
                checkpoint is not None
                and checkpoint_interval is not None
                and self.step_index % int(checkpoint_interval) == 0
            ):
                self.save_checkpoint(checkpoint)
        self.state = current
        self.success = True
        self.message = f"completed {nsteps} steps"
        if checkpoint is not None:
            self.save_checkpoint(checkpoint)
        return current

    def save_checkpoint(self, path):
        """Save driver and LETTA state needed for an exact continuation."""
        if self.state is None:
            raise ValueError("No evolved state is available to checkpoint.")
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        portable_state = (
            self.state.to_letta() if hasattr(self.state, "to_letta") else self.state
        )
        payload = {
            "format": "pyqed.letta.LETTAEvolution.checkpoint",
            "version": 1,
            "state": portable_state.to_state_dict(),
            "time": self.time,
            "step_index": self.step_index,
            "mode": self.mode,
            "saturation_streak": self.saturation_streak,
            "switch_step": self.switch_step,
            "switch_time": self.switch_time,
            "switch_reason": self.switch_reason,
            "history": self.history,
            "observations": self.observations,
            "config": {
                "max_bond": self.max_bond,
                "cutoff": self.cutoff,
                "krylov_dim": self.krylov_dim,
                "krylov_tol": self.krylov_tol,
                "saturation_steps": self.saturation_steps,
                "force_switch_time": self.force_switch_time,
                "backend": self.requested_backend,
                "device": self.device,
                "torch_num_threads": self.torch_num_threads,
                "channel_mode": self.channel_mode,
            },
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load_checkpoint(cls, path, operator):
        """Restore a driver; the immutable Hamiltonian is supplied explicitly."""
        with Path(path).expanduser().open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("format") != "pyqed.letta.LETTAEvolution.checkpoint":
            raise ValueError("Not a pyqed LETTAEvolution checkpoint.")
        driver = cls(operator, **payload["config"])
        portable_state = LETTA.from_state_dict(payload["state"])
        driver.state = driver.engine.prepare_state(portable_state) if (
            driver.backend == "torch"
        ) else portable_state
        driver.time = float(payload["time"])
        driver.step_index = int(payload["step_index"])
        driver.mode = str(payload["mode"])
        driver.saturation_streak = int(payload["saturation_streak"])
        driver.switch_step = payload["switch_step"]
        driver.switch_time = payload["switch_time"]
        driver.switch_reason = payload["switch_reason"]
        driver.history = list(payload["history"])
        driver.observations = list(payload["observations"])
        driver.engine.set_integrator(driver.mode)
        driver.engine.prepared = True
        driver.engine.state = driver.state
        return driver


__all__ = ["LETTAEvolution", "TDVP", "resolve_letta_backend"]
