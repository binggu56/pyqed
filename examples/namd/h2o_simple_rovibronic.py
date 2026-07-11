#!/usr/bin/env python3
"""Small full rovibronic H2O LDR example.

This is a deliberately lightweight model calculation:

* nuclear coordinates are full triatomic internal coordinates ``(r1, r2, theta)``
  for H-O-H;
* rotation is included through one conserved fixed-``J, Jz`` sector;
* two electronic states are propagated;
* nearest-neighbor electronic overlaps are analytic linked rotations.

Replace ``model_apes`` and ``model_overlap_links`` with CASCI/CASSCF data to turn
this template into an ab initio H2O rovibronic LDR run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.namd.triatomic import Triatom
from pyqed.units import au2fs

HARTREE_TO_EV = 27.211386245988
FS_TO_AU = 1.0 / au2fs


def optional_int(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "full"}:
        return None
    return int(value)


def h2o_body_frame(r: float = 1.81, theta_deg: float = 104.5):
    theta = np.deg2rad(theta_deg)
    return [
        ["H", (float(r), 0.0, 0.0)],
        ["O", (0.0, 0.0, 0.0)],
        ["H", (float(r) * np.cos(theta), float(r) * np.sin(theta), 0.0)],
    ]


def make_solver(args) -> Triatom:
    solver = Triatom(
        h2o_body_frame(),
        nstates=2,
        charge=0,
        spin=0,
        unit="bohr",
        J=args.J,
        Jz=args.Jz,
    )
    solver.set_dvr(
        domains=[
            [args.r_min, args.r_max],
            [args.r_min, args.r_max],
            [np.deg2rad(args.theta_min_deg), np.deg2rad(args.theta_max_deg)],
        ],
        npts=[args.n_r, args.n_r, args.n_theta],
        dvr_type=["sine", "sine", "legendre"],
    )
    solver.overlap_path_average = True
    return solver


def model_apes(solver: Triatom) -> np.ndarray:
    """Two smooth adiabatic-like H2O surfaces in Hartree."""
    r1, r2, theta = np.meshgrid(*solver.x, indexing="ij")
    re = 1.81
    te = np.deg2rad(104.5)
    qsym = 0.5 * (r1 + r2) - re
    qasym = 0.5 * (r1 - r2)
    qbend = theta - te

    # Soft curvatures chosen for a visible demo on a coarse grid.
    e0 = 0.055 * qsym**2 + 0.095 * qasym**2 + 0.018 * qbend**2
    e1 = (
        0.070
        + 0.050 * (qsym - 0.10) ** 2
        + 0.080 * (qasym + 0.08) ** 2
        + 0.015 * (qbend - 0.16) ** 2
    )
    return np.stack([e0, e1], axis=-1)


def mixing_angle(q) -> float:
    """Analytic electronic-frame angle alpha(q) for model LDR links."""
    r1, r2, theta = q
    te = np.deg2rad(104.5)
    bend = np.tanh((theta - te) / 0.13)
    asym = np.tanh((r1 - r2) / 0.22)
    envelope = np.exp(-((0.5 * (r1 + r2) - 1.86) / 0.42) ** 2)
    return 0.42 * envelope * bend + 0.28 * envelope * asym


def rotation_link(delta_alpha: float) -> np.ndarray:
    c = np.cos(delta_alpha)
    s = np.sin(delta_alpha)
    return np.array([[c, -s], [s, c]], dtype=complex)


def model_overlap_links(solver: Triatom) -> dict[tuple[int, tuple[int, ...]], np.ndarray]:
    """Nearest-neighbor electronic overlap links from alpha(q_j)-alpha(q_i)."""
    links = {}
    for axis in range(solver.ndim):
        for idx in solver._grid_indices():
            if idx[axis] + 1 >= solver.nx[axis]:
                continue
            nxt = list(idx)
            nxt[axis] += 1
            nxt = tuple(nxt)
            qi = tuple(solver.x[a][idx[a]] for a in range(solver.ndim))
            qj = tuple(solver.x[a][nxt[a]] for a in range(solver.ndim))
            links[(axis, idx)] = rotation_link(mixing_angle(qj) - mixing_angle(qi))
    return links


def initial_packet(
    solver: Triatom,
    *,
    state: int,
    rot_index: int,
    sigma_r: float,
    sigma_theta_deg: float,
) -> np.ndarray:
    r1, r2, theta = np.meshgrid(*solver.x, indexing="ij")
    center = np.array([1.81, 1.81, np.deg2rad(104.5)])
    sigma_theta = np.deg2rad(sigma_theta_deg)
    scalar = np.exp(
        -0.5 * ((r1 - center[0]) / sigma_r) ** 2
        -0.5 * ((r2 - center[1]) / sigma_r) ** 2
        -0.5 * ((theta - center[2]) / sigma_theta) ** 2
    )

    psi_values = np.zeros((*solver.nx, solver.nrot, solver.nstates), dtype=complex)
    psi_values[..., rot_index, state] = scalar
    psi = solver.to_quadrature_normalized(psi_values)
    return psi / solver.norm(psi)


def electronic_populations(psilist) -> np.ndarray:
    return np.asarray([np.sum(np.abs(psi) ** 2, axis=(0, 1, 2, 3)) for psi in psilist])


def rotational_populations(psilist) -> np.ndarray:
    return np.asarray([np.sum(np.abs(psi) ** 2, axis=(0, 1, 2, 4)) for psi in psilist])


def theta_density(solver: Triatom, psilist) -> np.ndarray:
    weights = np.asarray(solver.w[2], dtype=float)
    out = []
    for psi in psilist:
        marginal = np.sum(np.abs(psi) ** 2, axis=(0, 1, 3, 4))
        out.append(marginal / weights)
    return np.asarray(out)


def plot_populations(times_fs, electronic, rotational, outpath: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), constrained_layout=True)
    for state in range(electronic.shape[1]):
        axes[0].plot(times_fs, electronic[:, state], marker="o", label=f"S{state}")
    axes[0].set_xlabel("time / fs")
    axes[0].set_ylabel("population")
    axes[0].set_title("electronic")
    axes[0].legend()

    for r in range(rotational.shape[1]):
        axes[1].plot(times_fs, rotational[:, r], lw=1.4, label=f"rot {r}")
    axes[1].set_xlabel("time / fs")
    axes[1].set_ylabel("population")
    axes[1].set_title("fixed-J, Jz rotational sector")
    axes[1].legend(ncol=2, fontsize=8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_theta(times_fs, theta_deg, rho_theta, outpath: Path):
    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    image = ax.imshow(
        rho_theta.T,
        origin="lower",
        extent=[times_fs[0], times_fs[-1], theta_deg[0], theta_deg[-1]],
        aspect="auto",
        cmap="magma",
    )
    ax.set_xlabel("time / fs")
    ax.set_ylabel("H-O-H angle / deg")
    ax.set_title("theta density")
    fig.colorbar(image, ax=ax)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_wavepacket_snapshots(solver: Triatom, result: dict, times_fs, outpath: Path):
    chosen = np.linspace(0, len(times_fs) - 1, min(4, len(times_fs)), dtype=int)
    fig, axes = plt.subplots(
        1,
        len(chosen),
        figsize=(3.0 * len(chosen), 3.0),
        squeeze=False,
        constrained_layout=True,
    )
    extent = [solver.x[0][0], solver.x[0][-1], solver.x[1][0], solver.x[1][-1]]
    image = None
    for ax, it in zip(axes[0], chosen):
        rho = np.sum(np.abs(result["psilist"][it]) ** 2, axis=(2, 3, 4))
        peak = float(rho.max())
        if peak > 0.0:
            rho = rho / peak
        image = ax.imshow(
            rho.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            interpolation="bicubic",
        )
        ax.set_title(f"{times_fs[it]:.1f} fs")
        ax.set_xlabel("r1 / bohr")
        ax.set_ylabel("r2 / bohr")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="relative density")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--J", type=int, default=1)
    parser.add_argument(
        "--Jz",
        type=optional_int,
        default=0,
        help="Conserved space-fixed projection. Use '--Jz none' for the full M space.",
    )
    parser.add_argument("--n-r", type=int, default=4)
    parser.add_argument("--n-theta", type=int, default=5)
    parser.add_argument("--r-min", type=float, default=1.55)
    parser.add_argument("--r-max", type=float, default=2.25)
    parser.add_argument("--theta-min-deg", type=float, default=84.0)
    parser.add_argument("--theta-max-deg", type=float, default=124.0)
    parser.add_argument("--initial-state", type=int, default=1)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--tmax-fs", type=float, default=4.0)
    parser.add_argument("--nout", type=int, default=2)
    parser.add_argument("--sigma-r", type=float, default=0.13)
    parser.add_argument("--sigma-theta-deg", type=float, default=7.0)
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/h2o_simple_rovibronic"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    solver = make_solver(args)
    solver.apes = model_apes(solver)
    solver.overlap_links = model_overlap_links(solver)

    rot_index = solver.nrot // 2
    psi0 = initial_packet(
        solver,
        state=args.initial_state,
        rot_index=rot_index,
        sigma_r=args.sigma_r,
        sigma_theta_deg=args.sigma_theta_deg,
    )

    dt = args.dt_fs * FS_TO_AU
    nt = int(round(args.tmax_fs / args.dt_fs))
    result = solver.run(
        psi0,
        dt=dt,
        nt=nt,
        nout=args.nout,
        kinetic_propagator="expm_multiply",
        kinetic_action="matrix-free",
    )

    times_fs = result["times"] * au2fs
    electronic = electronic_populations(result["psilist"])
    rotational = rotational_populations(result["psilist"])
    rho_theta = theta_density(solver, result["psilist"])

    npz_path = args.outdir / "h2o_simple_rovibronic.npz"
    np.savez_compressed(
        npz_path,
        times_fs=times_fs,
        r1=solver.x[0],
        r2=solver.x[1],
        theta=solver.x[2],
        theta_deg=np.rad2deg(solver.x[2]),
        apes=solver.apes,
        psi_t=np.asarray(result["psilist"]),
        electronic_populations=electronic,
        rotational_populations=rotational,
        theta_density=rho_theta,
        J=solver.J,
        Jz=-999 if solver.Jz is None else solver.Jz,
        nrot=solver.nrot,
        rot_index=rot_index,
    )

    plot_populations(times_fs, electronic, rotational, args.outdir / "h2o_populations.png")
    plot_theta(times_fs, np.rad2deg(solver.x[2]), rho_theta, args.outdir / "h2o_theta_density.png")
    plot_wavepacket_snapshots(solver, result, times_fs, args.outdir / "h2o_wavepacket_r1r2.png")

    print(f"grid={solver.nx}, J={solver.J}, Jz={solver.Jz}, nrot={solver.nrot}, nstates={solver.nstates}")
    print(f"dimension={np.prod(solver.nx) * solver.nrot * solver.nstates}")
    print(f"APES range={(solver.apes.min() * HARTREE_TO_EV):.4f}..{(solver.apes.max() * HARTREE_TO_EV):.4f} eV")
    print(f"initial norm={solver.norm(psi0):.12f}")
    print(f"final norm={solver.norm(result['psilist'][-1]):.12f}")
    print(f"npz={npz_path}")
    print(f"population_png={args.outdir / 'h2o_populations.png'}")
    print(f"theta_png={args.outdir / 'h2o_theta_density.png'}")
    print(f"wavepacket_png={args.outdir / 'h2o_wavepacket_r1r2.png'}")


if __name__ == "__main__":
    main()
