"""Directional corner-transfer renormalization for finite PEPS layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from .contraction import BoundaryMPSContractor, shared_executor


def _rotate_clockwise(layers):
    nrows = len(layers)
    ncols = len(layers[0])
    rotated = [[None for _ in range(nrows)] for _ in range(ncols)]
    for row in range(nrows):
        for col in range(ncols):
            rotated[col][nrows - 1 - row] = np.asarray(layers[row][col]).transpose(
                3, 0, 1, 2
            )
    return tuple(tuple(row) for row in rotated)


def _directional_layers(layers):
    top = tuple(tuple(np.asarray(tensor) for tensor in row) for row in layers)
    left = _rotate_clockwise(top)
    bottom = _rotate_clockwise(left)
    right = _rotate_clockwise(bottom)
    return {
        "top": top,
        "right": right,
        "bottom": bottom,
        "left": left,
    }


def _rotate_replacements_clockwise(replacements, shape):
    nrows, ncols = shape
    rotated = {}
    for (row, col), tensor in replacements.items():
        rotated[(col, nrows - 1 - row)] = np.asarray(tensor).transpose(3, 0, 1, 2)
    return rotated, (ncols, nrows)


def _representative_corner(info):
    rows = info.get("row_schmidt_values", ())
    if not rows:
        return np.ones((1, 1))
    row = rows[len(rows) // 2]
    if not row:
        return np.ones((1, 1))
    values = np.asarray(row[len(row) // 2])
    if values.size == 0:
        return np.ones((1, 1))
    scale = float(np.linalg.norm(values))
    normalized = values if scale == 0.0 else values / scale
    return np.diag(normalized)


@dataclass
class CTMRGEnvironment:
    """Renormalized finite-PEPS corner environment and diagnostics."""

    value: complex
    chi: int
    corners: dict
    edges: dict
    directional_values: dict
    history: tuple
    converged: bool
    residual: float
    directional_spread: float
    warm_started: bool = False
    evaluators: dict = field(default_factory=dict, repr=False)

    def diagnostics(self):
        """Return contraction diagnostics in the public contractor format."""

        return {
            "method": "ctmrg",
            "chi": self.chi,
            "converged": self.converged,
            "iterations": len(self.history),
            "residual": self.residual,
            "directional_spread": self.directional_spread,
            "corners": self.corners,
            "edges": self.edges,
            "history": self.history,
            "max_relative_error": max(
                edge["max_relative_error"] for edge in self.edges.values()
            ),
            "environment_reused": False,
            "environment_builds": 1,
            "warm_started": self.warm_started,
        }

    def contract_replacements(self, replacements, *, return_info=False):
        """Evaluate local replacement layers in the cached four directions."""

        values, infos = self.contract_many(
            (replacements,),
            return_info=True,
        )
        return (values[0], infos[0]) if return_info else values[0]

    def contract_many(
        self,
        replacement_maps,
        *,
        return_info=False,
        workers=1,
    ):
        """Evaluate many observable channels in four batched directions."""

        if not self.evaluators:
            raise RuntimeError("this CTMRG environment has no observable evaluators.")
        directional_maps = {name: [] for name in self.evaluators}
        for replacements in replacement_maps:
            maps = {"top": dict(replacements)}
            shape = self.evaluators["top"].original_shape
            maps["left"], shape = _rotate_replacements_clockwise(maps["top"], shape)
            maps["bottom"], shape = _rotate_replacements_clockwise(maps["left"], shape)
            maps["right"], _ = _rotate_replacements_clockwise(maps["bottom"], shape)
            for name in directional_maps:
                directional_maps[name].append(maps[name])

        def contract_direction(item):
            name, evaluator = item
            values, infos = evaluator.contract_many(
                directional_maps[name],
                return_info=True,
                workers=1,
            )
            return name, values, infos

        items = list(self.evaluators.items())
        if workers > 1:
            contracted = shared_executor(min(int(workers), len(items))).map(
                contract_direction,
                items,
            )
        else:
            contracted = map(contract_direction, items)
        directional_values = {}
        directional_infos = {}
        for name, values, infos in contracted:
            directional_values[name] = values
            directional_infos[name] = infos

        values = []
        infos = []
        njobs = len(next(iter(directional_values.values()), ()))
        for job in range(njobs):
            job_values = {
                name: items[job] for name, items in directional_values.items()
            }
            job_infos = {
                name: items[job] for name, items in directional_infos.items()
            }
            value = sum(job_values.values()) / len(job_values)
            scale = max(1.0, abs(value))
            values.append(value)
            infos.append(
                {
                    "method": "ctmrg",
                    "chi": self.chi,
                    "directional_values": job_values,
                    "directional_spread": max(
                        abs(item - value) for item in job_values.values()
                    ) / scale,
                    "max_relative_error": max(
                        item["max_relative_error"] for item in job_infos.values()
                    ),
                    "environment_reused": True,
                    "environment_builds": 0,
                    "directional_contractions": job_infos,
                    "batched_frontier": True,
                }
            )
        values = tuple(values)
        infos = tuple(infos)
        return (values, infos) if return_info else values


@dataclass
class CTMRGContractor:
    r"""Four-direction finite corner-transfer matrix renormalization.

    Each move absorbs a complete layer into a boundary state and truncates its
    Schmidt spaces to ``chi``. The four orientations provide renormalized
    corner spectra and an internal directional-consistency diagnostic. ``chi``
    grows geometrically from ``initial_chi`` to the requested final value.
    """

    chi: int = 64
    initial_chi: int = 4
    max_iterations: int = 16
    tolerance: float = 1.0e-10
    rtol: float = 1.0e-10
    atol: float = 0.0

    def __post_init__(self):
        for name in ("chi", "initial_chi", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer.")
            setattr(self, name, int(value))
        self.initial_chi = min(self.initial_chi, self.chi)
        for name in ("tolerance", "rtol", "atol"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            setattr(self, name, value)
        self.environment = None

    @staticmethod
    def _contract_direction(layers, chi, rtol, atol):
        contractor = BoundaryMPSContractor(
            max_bond=chi,
            rtol=rtol,
            atol=atol,
            direction="rows",
        )
        value, info = contractor.contract(layers, return_info=True)
        info = dict(info)
        info["row_schmidt_values"] = tuple(
            row.get("schmidt_values", ())
            for row in info["row_compressions"]
        )
        return value, info

    def run(self, layers, *, warm_start=None, cache_observables=False):
        """Build and return the converged finite CTMRG environment."""

        directions = _directional_layers(layers)
        current_chi = self.initial_chi
        previous = None
        previous_chi = None
        warm_started = False
        if warm_start is not None:
            if not isinstance(warm_start, CTMRGEnvironment):
                raise TypeError("warm_start must be a CTMRGEnvironment or None.")
            current_chi = min(self.chi, max(self.initial_chi, warm_start.chi))
            previous = warm_start.value
            previous_chi = current_chi
            warm_started = True
        history = []
        final_values = {}
        final_infos = {}
        converged = False
        residual = np.inf
        spread = np.inf
        for iteration in range(self.max_iterations):
            values = {}
            infos = {}
            for name, oriented in directions.items():
                values[name], infos[name] = self._contract_direction(
                    oriented,
                    current_chi,
                    self.rtol,
                    self.atol,
                )
            estimate = sum(values.values()) / len(values)
            scale = max(1.0, abs(estimate))
            spread = max(abs(value - estimate) for value in values.values()) / scale
            residual = (
                np.inf
                if previous is None
                else abs(estimate - previous) / max(1.0, abs(estimate), abs(previous))
            )
            history.append(
                {
                    "iteration": iteration,
                    "chi": current_chi,
                    "value": estimate,
                    "residual": float(residual),
                    "directional_spread": float(spread),
                    "max_truncation_error": max(
                        info["max_relative_error"] for info in infos.values()
                    ),
                    "warm_started": warm_started,
                }
            )
            final_values = values
            final_infos = infos
            if current_chi < self.chi:
                previous_chi = current_chi
                current_chi = min(self.chi, 2 * current_chi)
                previous = estimate
                continue
            if (
                previous is not None
                and residual <= self.tolerance
                and previous_chi == current_chi
            ):
                converged = True
                break
            previous = estimate
            previous_chi = current_chi

        estimate = sum(final_values.values()) / len(final_values)
        final_chi = int(history[-1]["chi"])
        corner_names = {
            "top": "northwest",
            "right": "northeast",
            "bottom": "southeast",
            "left": "southwest",
        }
        corners = {
            corner_names[name]: _representative_corner(info)
            for name, info in final_infos.items()
        }
        edges = {
            name: {
                "row_bond_dims": info["row_bond_dims"],
                "discarded_weight": info["discarded_weight"],
                "max_relative_error": info["max_relative_error"],
            }
            for name, info in final_infos.items()
        }
        evaluators = {}
        if cache_observables:
            evaluators = {
                name: BoundaryMPSContractor(
                    max_bond=final_chi,
                    rtol=self.rtol,
                    atol=self.atol,
                    direction="rows",
                ).build_environment(oriented)
                for name, oriented in directions.items()
            }
        self.environment = CTMRGEnvironment(
            value=estimate,
            chi=final_chi,
            corners=corners,
            edges=edges,
            directional_values=final_values,
            history=tuple(history),
            converged=converged,
            residual=float(residual),
            directional_spread=float(spread),
            warm_started=warm_started,
            evaluators=evaluators,
        )
        return self.environment

    def contract(self, layers, *, return_info=False):
        environment = self.run(layers)
        info = environment.diagnostics()
        return (environment.value, info) if return_info else environment.value


__all__ = ["CTMRGContractor", "CTMRGEnvironment"]
