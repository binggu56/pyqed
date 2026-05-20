#!/usr/bin/env python3
"""Four-mode pyrazine LDRFG demonstration.

The model uses the four-mode pyrazine linear vibronic Hamiltonian from the
legacy sparse-grid example.  A configurable subset of normal coordinates is
represented on a sine-DVR/LDR grid; the remaining coordinates are propagated as
one moving frozen Gaussian packet with the dense LDRFG equations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.dvr import SineDVR
from pyqed.namd import LDRFG
from pyqed.units import au2ev, au2fs, wavenum2au


MODE_LABELS = ("nu1", "nu6a", "nu9a", "nu10a")
MODE_FREQ_CM = np.array([1015.0, 596.0, 1230.0, 919.0])
MODE_FREQ_AU = MODE_FREQ_CM * wavenum2au
MODE_MASSES = 1.0 / MODE_FREQ_AU
ESHIFT = np.array([3.94, 4.89]) / au2ev
KAPPA_1 = np.array([-0.0470, -0.0964, 0.1594, 0.0]) / au2ev
KAPPA_2 = np.array([-0.2012, 0.1193, 0.0484, 0.0]) / au2ev
GAMMA_10A = -0.018 / au2ev
LAMBDA_10A = 0.1825 / au2ev
NSTATES = 3


def pyrazine_4mode_diabatic(coords: np.ndarray) -> np.ndarray:
    """Return the 3-state diabatic Hamiltonian at one four-mode geometry."""

    q = np.asarray(coords, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"coords shape {q.shape} != (4,).")

    harmonic = 0.5 * np.dot(MODE_FREQ_AU, q * q)
    h = np.zeros((NSTATES, NSTATES), dtype=float)
    h[0, 0] = harmonic
    h[1, 1] = harmonic + np.dot(KAPPA_1, q) + ESHIFT[0] + GAMMA_10A * q[3] ** 2
    h[2, 2] = harmonic + np.dot(KAPPA_2, q) + ESHIFT[1] + GAMMA_10A * q[3] ** 2
    h[1, 2] = h[2, 1] = LAMBDA_10A * q[3]
    return h


def pyrazine_4mode_diabatic_gradient(coords: np.ndarray) -> np.ndarray:
    """Return dH/dq_j with shape ``(4, 3, 3)``."""

    q = np.asarray(coords, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"coords shape {q.shape} != (4,).")

    grad = np.zeros((4, NSTATES, NSTATES), dtype=float)
    for mode in range(4):
        base = MODE_FREQ_AU[mode] * q[mode]
        grad[mode, 0, 0] = base
        grad[mode, 1, 1] = base + KAPPA_1[mode]
        grad[mode, 2, 2] = base + KAPPA_2[mode]
    grad[3, 1, 1] += 2.0 * GAMMA_10A * q[3]
    grad[3, 2, 2] += 2.0 * GAMMA_10A * q[3]
    grad[3, 1, 2] = grad[3, 2, 1] = LAMBDA_10A
    return grad


def _align_vectors(reference: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    aligned = np.array(vectors, copy=True)
    for grid_index in range(reference.shape[0]):
        for state in range(reference.shape[2]):
            if np.dot(reference[grid_index, :, state], aligned[grid_index, :, state]) < 0.0:
                aligned[grid_index, :, state] *= -1.0
    return aligned


def _canonicalize_vectors(vectors: np.ndarray) -> np.ndarray:
    canonical = np.array(vectors, copy=True)
    for grid_index in range(canonical.shape[0]):
        for state in range(canonical.shape[2]):
            pivot = int(np.argmax(np.abs(canonical[grid_index, :, state])))
            if canonical[grid_index, pivot, state] < 0.0:
                canonical[grid_index, :, state] *= -1.0
    return canonical


def _tensor_product_kinetic(dvrs: list[SineDVR]) -> np.ndarray:
    kinetic = None
    for axis, dvr in enumerate(dvrs):
        factors = []
        for j, other in enumerate(dvrs):
            factors.append(sp.csr_matrix(dvr.t()) if j == axis else sp.eye(other.npts, format="csr"))
        term = factors[0]
        for factor in factors[1:]:
            term = sp.kron(term, factor, format="csr")
        kinetic = term if kinetic is None else kinetic + term
    return np.asarray(kinetic.toarray(), dtype=complex)


def _tensor_product_grid(dvrs: list[SineDVR]) -> np.ndarray:
    meshes = np.meshgrid(*[dvr.x for dvr in dvrs], indexing="ij")
    return np.stack([mesh.reshape(-1) for mesh in meshes], axis=-1)


@dataclass
class Pyrazine4ModeLDRFGModel:
    """Adapter from four-mode pyrazine LVC data to the LDRFG solver."""

    ldr_mode_indices: tuple[int, ...] = (0, 3)
    npts: int = 9
    qmax: float = 6.0
    gaussian_width: float = 1.0
    finite_difference_step: float = 1.0e-5
    include_berry: bool = True

    def __post_init__(self) -> None:
        if not self.ldr_mode_indices:
            raise ValueError("At least one LDR mode is required.")
        if len(set(self.ldr_mode_indices)) != len(self.ldr_mode_indices):
            raise ValueError("LDR mode indices must be unique.")
        if any(mode < 0 or mode >= 4 for mode in self.ldr_mode_indices):
            raise ValueError("LDR mode indices must be between 0 and 3.")

        self.fg_mode_indices = tuple(mode for mode in range(4) if mode not in self.ldr_mode_indices)
        self.dvrs = [
            SineDVR(-self.qmax, self.qmax, self.npts, mass=MODE_MASSES[mode])
            for mode in self.ldr_mode_indices
        ]
        self.ldr_grid = _tensor_product_grid(self.dvrs)
        self.kinetic_x = _tensor_product_kinetic(self.dvrs)
        self.masses_y = MODE_MASSES[list(self.fg_mode_indices)]
        self.gamma_y = np.eye(len(self.fg_mode_indices)) * self.gaussian_width

    @property
    def ngrid(self) -> int:
        return self.ldr_grid.shape[0]

    @property
    def ny(self) -> int:
        return len(self.fg_mode_indices)

    def full_coords(self, q_fg: np.ndarray) -> np.ndarray:
        q_fg = np.asarray(q_fg, dtype=float)
        if q_fg.shape != (self.ny,):
            raise ValueError(f"q_fg shape {q_fg.shape} != {(self.ny,)}.")
        coords = np.zeros((self.ngrid, 4), dtype=float)
        coords[:, list(self.ldr_mode_indices)] = self.ldr_grid
        coords[:, list(self.fg_mode_indices)] = q_fg
        return coords

    def electronic_vectors(self, q_fg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coords = self.full_coords(q_fg)
        energies = np.empty((self.ngrid, NSTATES), dtype=float)
        vectors = np.empty((self.ngrid, NSTATES, NSTATES), dtype=float)
        for grid_index, geom in enumerate(coords):
            energies[grid_index], vectors[grid_index] = np.linalg.eigh(pyrazine_4mode_diabatic(geom))
        vectors = _canonicalize_vectors(vectors)
        return energies, vectors

    def energies(self, q_fg: np.ndarray) -> np.ndarray:
        energies, _ = self.electronic_vectors(q_fg)
        return energies

    def overlap(self, q_fg: np.ndarray) -> np.ndarray:
        _, vectors = self.electronic_vectors(q_fg)
        return np.einsum("mdb,nda->mbna", vectors, vectors)

    def grad_energies(self, q_fg: np.ndarray) -> np.ndarray:
        q_fg = np.asarray(q_fg, dtype=float)
        energies, vectors = self.electronic_vectors(q_fg)
        coords = self.full_coords(q_fg)
        grad = np.empty((self.ny, self.ngrid, NSTATES), dtype=float)
        for fg_axis, mode in enumerate(self.fg_mode_indices):
            for grid_index, geom in enumerate(coords):
                dh = pyrazine_4mode_diabatic_gradient(geom)[mode]
                grad[fg_axis, grid_index] = np.einsum(
                    "da,de,ea->a",
                    vectors[grid_index],
                    dh,
                    vectors[grid_index],
                )
        return grad

    def grad_overlap(self, q_fg: np.ndarray) -> np.ndarray:
        q_fg = np.asarray(q_fg, dtype=float)
        eps = self.finite_difference_step
        grad = np.empty((self.ny, self.ngrid, NSTATES, self.ngrid, NSTATES), dtype=complex)
        for fg_axis in range(self.ny):
            qp = q_fg.copy()
            qm = q_fg.copy()
            qp[fg_axis] += eps
            qm[fg_axis] -= eps
            grad[fg_axis] = (self.overlap(qp) - self.overlap(qm)) / (2.0 * eps)
        return grad

    def berry(self, q_fg: np.ndarray) -> np.ndarray:
        q_fg = np.asarray(q_fg, dtype=float)
        eps = self.finite_difference_step
        _, reference = self.electronic_vectors(q_fg)
        berry = np.zeros((self.ny, self.ngrid, NSTATES, self.ngrid, NSTATES), dtype=float)
        for fg_axis in range(self.ny):
            qp = q_fg.copy()
            qm = q_fg.copy()
            qp[fg_axis] += eps
            qm[fg_axis] -= eps
            _, vp = self.electronic_vectors(qp)
            _, vm = self.electronic_vectors(qm)
            dv = (_align_vectors(reference, vp) - _align_vectors(reference, vm)) / (2.0 * eps)
            local = np.einsum("ndb,nda->nba", reference, dv)
            for grid_index in range(self.ngrid):
                berry[fg_axis, grid_index, :, grid_index, :] = local[grid_index]
        return berry

    def solver(self) -> LDRFG:
        return LDRFG(
            self.kinetic_x,
            masses_y=self.masses_y,
            energies=self.energies,
            overlap=self.overlap,
            grad_energies=self.grad_energies,
            grad_overlap=self.grad_overlap,
            berry=self.berry if self.include_berry else None,
            gamma=self.gamma_y if self.ny else None,
        )


def initial_ldrfg_state(model: Pyrazine4ModeLDRFGModel, state: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    envelope = np.ones(model.ngrid, dtype=float)
    for axis, mode in enumerate(model.ldr_mode_indices):
        envelope *= np.exp(-0.5 * MODE_FREQ_AU[mode] * MODE_MASSES[mode] * model.ldr_grid[:, axis] ** 2)
    envelope /= np.linalg.norm(envelope)
    c = np.zeros((model.ngrid, NSTATES), dtype=complex)
    c[:, state] = envelope
    q = np.zeros(model.ny, dtype=float)
    p = np.zeros(model.ny, dtype=float)
    return c, q, p


def run_demo(
    npts: int = 7,
    qmax: float = 6.0,
    tmax_fs: float = 5.0,
    nsteps: int = 100,
    initial_state: int = 2,
    ldr_modes: tuple[int, ...] = (0, 3),
    gaussian_width: float = 1.0,
    include_berry: bool = True,
) -> dict[str, np.ndarray]:
    model = Pyrazine4ModeLDRFGModel(
        ldr_mode_indices=ldr_modes,
        npts=npts,
        qmax=qmax,
        gaussian_width=gaussian_width,
        include_berry=include_berry,
    )
    solver = model.solver()
    c, q, p = initial_ldrfg_state(model, state=initial_state)

    dt = (tmax_fs / au2fs) / nsteps
    times_fs = np.linspace(0.0, tmax_fs, nsteps + 1)
    populations = np.empty((nsteps + 1, NSTATES), dtype=float)
    q_history = np.empty((nsteps + 1, model.ny), dtype=float)
    p_history = np.empty((nsteps + 1, model.ny), dtype=float)
    energy = np.empty(nsteps + 1, dtype=float)

    for step in range(nsteps + 1):
        populations[step] = np.sum(np.abs(c) ** 2, axis=0).real
        q_history[step] = q
        p_history[step] = p
        energy[step] = solver.energy(c, q, p).real
        if step == nsteps:
            break
        c, q, p = solver.step_rk4(c, q, p, dt)
        c /= np.sqrt(np.vdot(c.ravel(), c.ravel()))

    return {
        "times_fs": times_fs,
        "populations": populations,
        "q": q_history,
        "p": p_history,
        "energy": energy,
        "ldr_modes": np.asarray(ldr_modes, dtype=int),
        "fg_modes": np.asarray(model.fg_mode_indices, dtype=int),
        "gaussian_width": np.asarray(model.gaussian_width),
        "ldr_grid": model.ldr_grid,
    }


def plot_demo(result: dict[str, np.ndarray], outpath: Path) -> None:
    fig, (ax_pop, ax_q, ax_e) = plt.subplots(
        3,
        1,
        figsize=(7.0, 7.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.6, 1.8, 1.4]},
    )
    labels = ["S0", "S1", "S2"]
    for state, label in enumerate(labels):
        ax_pop.plot(result["times_fs"], result["populations"][:, state], lw=2.0, label=label)
    ax_pop.set_ylabel("adiabatic population")
    ax_pop.set_ylim(-0.03, 1.03)
    ax_pop.legend(frameon=False, ncol=3)

    for axis, mode in enumerate(result["fg_modes"]):
        ax_q.plot(result["times_fs"], result["q"][:, axis], lw=1.8, label=f"Q_{MODE_LABELS[mode]}")
    ax_q.set_ylabel("FG center")
    ax_q.legend(frameon=False, ncol=min(3, max(1, len(result["fg_modes"]))))

    drift = result["energy"] - result["energy"][0]
    ax_e.plot(result["times_fs"], drift, color="0.2", lw=1.8)
    ax_e.set_xlabel("time / fs")
    ax_e.set_ylabel("energy drift")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _parse_modes(value: str) -> tuple[int, ...]:
    if not value:
        raise argparse.ArgumentTypeError("mode list must not be empty")
    modes = tuple(int(item.strip()) for item in value.split(","))
    if any(mode < 0 or mode >= 4 for mode in modes):
        raise argparse.ArgumentTypeError("mode indices must be 0, 1, 2, or 3")
    return modes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npts", type=int, default=7)
    parser.add_argument("--qmax", type=float, default=6.0)
    parser.add_argument("--tmax-fs", type=float, default=5.0)
    parser.add_argument("--nsteps", type=int, default=100)
    parser.add_argument("--initial-state", type=int, choices=range(NSTATES), default=2)
    parser.add_argument("--ldr-modes", type=_parse_modes, default=(0, 3))
    parser.add_argument("--gaussian-width", type=float, default=1.0)
    parser.add_argument("--no-berry", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/pyrazine_4mode_ldrfg"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    result = run_demo(
        npts=args.npts,
        qmax=args.qmax,
        tmax_fs=args.tmax_fs,
        nsteps=args.nsteps,
        initial_state=args.initial_state,
        ldr_modes=args.ldr_modes,
        gaussian_width=args.gaussian_width,
        include_berry=not args.no_berry,
    )
    prefix = "pyrazine_4mode_ldrfg"
    plot_path = args.outdir / f"{prefix}_dynamics.png"
    data_path = args.outdir / f"{prefix}_dynamics.npz"
    plot_demo(result, plot_path)
    np.savez_compressed(data_path, **result)

    print(f"[plot] {plot_path}")
    print(f"[data] {data_path}")
    print("[ldr modes]", [MODE_LABELS[i] for i in result["ldr_modes"]])
    print("[fg modes]", [MODE_LABELS[i] for i in result["fg_modes"]])
    print("[gaussian width]", float(result["gaussian_width"]))
    print("[final populations]", np.array2string(result["populations"][-1], precision=8))
    print(
        "[norm] min={:.12f} max={:.12f}".format(
            float(result["populations"].sum(axis=1).min()),
            float(result["populations"].sum(axis=1).max()),
        )
    )
    print("[energy drift]", float(result["energy"][-1] - result["energy"][0]))


if __name__ == "__main__":
    main()
