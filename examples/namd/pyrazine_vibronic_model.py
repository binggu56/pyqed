#!/usr/bin/env python3
"""Two-mode pyrazine sine-DVR LDR vibronic-coupling benchmark.

This is a deliberately plain diabatic model used as a control problem for the
LDR experiments.  By default the Hamiltonian is represented on local adiabatic
PESs, so the sine-DVR kinetic energy is multiplied by the electronic overlap
between adiabatic eigenvectors at different DVR grid points.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
from numpy.polynomial.legendre import leggauss
from scipy.sparse.linalg import expm_multiply

from pyqed.dvr import SineDVR
from pyqed.units import au2ev, au2fs, wavenum2au


def pyrazine_potential(x, y):
    """Return a 3-state diabatic pyrazine potential on a 2D grid.

    Coordinates are dimensionless normal coordinates.  States are
    ``S0, S1, S2``.  Only ``S1`` and ``S2`` are vibronically coupled.
    """

    freq_c = 952.0 * wavenum2au
    freq_t = 597.0 * wavenum2au
    e_shift = np.array([31800.0, 39000.0]) * wavenum2au
    kappa = np.array([-847.0, 1202.0]) * wavenum2au
    lam = 2110.0 * wavenum2au

    xg, yg = np.meshgrid(x, y, indexing="ij")
    base = 0.5 * freq_c * xg**2 + 0.5 * freq_t * yg**2

    v = np.zeros((len(x), len(y), 3, 3), dtype=float)
    v[:, :, 0, 0] = base
    v[:, :, 1, 1] = base + kappa[0] * yg + e_shift[0]
    v[:, :, 2, 2] = base + kappa[1] * yg + e_shift[1]
    v[:, :, 1, 2] = lam * xg
    v[:, :, 2, 1] = lam * xg
    return v


def adiabatic_representation(v):
    """Return adiabatic energies and diabatic-to-adiabatic eigenvectors."""

    ngrid = v.shape[0] * v.shape[1]
    energies, vectors = np.linalg.eigh(v.reshape(ngrid, 3, 3))
    return energies.reshape(v.shape[0], v.shape[1], 3), vectors.reshape(v.shape[0], v.shape[1], 3, 3)


def build_sine_dvr_ldr_hamiltonian(x_dvr, y_dvr, representation="adiabatic"):
    """Build the sine-DVR LDR Hamiltonian."""

    nstates = 3
    nx = x_dvr.npts
    ny = y_dvr.npts
    ngrid = nx * ny
    tx = sp.csr_matrix(x_dvr.t())
    ty = sp.csr_matrix(y_dvr.t())
    ix = sp.eye(nx, format="csr")
    iy = sp.eye(ny, format="csr")
    kinetic_nuclear = sp.kron(tx, iy, format="csr") + sp.kron(ix, ty, format="csr")

    v_diabatic = pyrazine_potential(x_dvr.x, y_dvr.x)
    if representation == "diabatic":
        kinetic = sp.kron(kinetic_nuclear, sp.eye(nstates, format="csr"), format="csr")
        v = v_diabatic.reshape(ngrid, nstates, nstates)
        rows = []
        cols = []
        data = []
        for g in range(ngrid):
            block = v[g]
            nz_a, nz_b = np.nonzero(np.abs(block) > 0.0)
            rows.extend((g * nstates + nz_a).tolist())
            cols.extend((g * nstates + nz_b).tolist())
            data.extend(block[nz_a, nz_b].tolist())
        potential = sp.csr_matrix((data, (rows, cols)), shape=(ngrid * nstates, ngrid * nstates))
        adiabatic_energies, _ = adiabatic_representation(v_diabatic)
        return kinetic + potential, v_diabatic, adiabatic_energies

    if representation != "adiabatic":
        raise ValueError(f"unknown representation {representation!r}")

    adiabatic_energies, vectors = adiabatic_representation(v_diabatic)
    vectors = vectors.reshape(ngrid, nstates, nstates)
    kinetic_nuclear = kinetic_nuclear.tocoo()
    rows = []
    cols = []
    data = []
    for gi, gj, tij in zip(kinetic_nuclear.row, kinetic_nuclear.col, kinetic_nuclear.data):
        overlap = vectors[gi].T @ vectors[gj]
        for state_i in range(nstates):
            row0 = gi * nstates + state_i
            for state_j in range(nstates):
                value = tij * overlap[state_i, state_j]
                if value != 0.0:
                    rows.append(row0)
                    cols.append(gj * nstates + state_j)
                    data.append(value)
    kinetic = sp.csr_matrix((data, (rows, cols)), shape=(ngrid * nstates, ngrid * nstates))
    potential = sp.diags(adiabatic_energies.reshape(ngrid, nstates).reshape(-1), format="csr")
    return kinetic + potential, v_diabatic, adiabatic_energies


def sine_basis_values(dvr, x):
    n = np.arange(1, dvr.npts + 1)
    return np.sqrt(2.0 / dvr.L) * np.sin(np.outer(np.asarray(x) - dvr.xmin, n) * np.pi / dvr.L)


def projected_harmonic_ground(dvr, nquad=1000):
    """Project pi^-1/4 exp(-Q^2/2) into the truncated sine FBR."""

    q, w = leggauss(nquad)
    x = 0.5 * (dvr.xmax - dvr.xmin) * q + 0.5 * (dvr.xmax + dvr.xmin)
    wx = 0.5 * (dvr.xmax - dvr.xmin) * w
    values = np.pi ** (-0.25) * np.exp(-0.5 * x**2)
    return sine_basis_values(dvr, x).T @ (wx * values)


def initial_packet_dvr(x_dvr, y_dvr, state):
    coeff_fbr = np.outer(projected_harmonic_ground(x_dvr), projected_harmonic_ground(y_dvr))
    coeff_dvr = x_dvr.fbr2dvr().T @ coeff_fbr @ y_dvr.fbr2dvr()
    psi = np.zeros((x_dvr.npts, y_dvr.npts, 3), dtype=complex)
    psi[:, :, state] = coeff_dvr
    psi = psi.reshape(-1)
    return psi / np.linalg.norm(psi)


def populations_from_vector(psi, ngrid, nstates=3):
    return np.sum(np.abs(psi.reshape(ngrid, nstates)) ** 2, axis=0).real


def sine_dvr_ldr_run(qmax, npts, tmax_fs, nsnapshots, initial_state=2, representation="adiabatic"):
    freq_c = 952.0 * wavenum2au
    freq_t = 597.0 * wavenum2au
    x_dvr = SineDVR(-qmax, qmax, npts, mass=1.0 / freq_c)
    y_dvr = SineDVR(-qmax, qmax, npts, mass=1.0 / freq_t)
    hamiltonian, potential, adiabatic_energies = build_sine_dvr_ldr_hamiltonian(
        x_dvr, y_dvr, representation=representation
    )
    psi0 = initial_packet_dvr(x_dvr, y_dvr, initial_state)
    times = np.linspace(0.0, tmax_fs, nsnapshots)
    states = expm_multiply(
        -1j * hamiltonian,
        psi0,
        start=0.0,
        stop=float(tmax_fs / au2fs),
        num=nsnapshots,
        traceA=-1j * hamiltonian.diagonal().sum(),
    )
    ngrid = npts * npts
    pops = np.asarray([populations_from_vector(state, ngrid) for state in states])
    return {
        "times_fs": times,
        "populations": pops,
        "states": states.reshape(nsnapshots, npts, npts, 3),
        "potential": potential,
        "adiabatic_energies": adiabatic_energies,
        "representation": representation,
        "x": x_dvr.x,
        "y": y_dvr.x,
        "x_edges": cell_edges(x_dvr),
        "y_edges": cell_edges(y_dvr),
        "dx": x_dvr.dx,
        "dy": y_dvr.dx,
        "hamiltonian_nnz": hamiltonian.nnz,
        "dim": hamiltonian.shape[0],
    }


def cell_edges(dvr):
    centers = np.asarray(dvr.x)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = dvr.xmin
    edges[-1] = dvr.xmax
    return edges


def plot_populations(times_fs, pops, outpath, representation):
    fig, (ax_pop, ax_norm) = plt.subplots(
        2,
        1,
        figsize=(6.8, 5.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    labels = ["S0", "S1", "S2"]
    colors = ["0.35", "tab:blue", "tab:orange"]
    for state, (label, color) in enumerate(zip(labels, colors)):
        ax_pop.plot(times_fs, pops[:, state], lw=2.0, color=color, label=label)
    ax_norm.plot(times_fs, pops.sum(axis=1) - 1.0, color="k", lw=1.8)
    ax_pop.set_ylabel(f"{representation} population")
    ax_pop.set_ylim(-0.03, 1.03)
    ax_pop.legend(frameon=False, ncol=3)
    ax_norm.set_xlabel("time / fs")
    ax_norm.set_ylabel("norm - 1")
    ax_norm.set_ylim(-1.0e-10, 1.0e-10)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_wavepackets(result, snapshot_times, outpath):
    x = result["x"]
    y = result["y"]
    x_edges = result["x_edges"]
    y_edges = result["y_edges"]
    times = result["times_fs"]
    states = result["states"]
    pops = result["populations"]
    selected = [int(np.argmin(np.abs(times - t))) for t in snapshot_times]
    labels = ["S0", "S1", "S2"]
    cmaps = ["cividis", "viridis", "plasma"]
    ye, xe = np.meshgrid(y_edges, x_edges, indexing="xy")
    cell_area = result["dx"] * result["dy"]

    fig, axes = plt.subplots(
        3,
        len(selected),
        figsize=(2.7 * len(selected), 7.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    last = None
    for state in range(3):
        vmax = float(np.max(np.abs(states[:, :, :, state]) ** 2))
        if vmax == 0.0:
            vmax = 1.0
        for col, idx in enumerate(selected):
            ax = axes[state, col]
            density = np.abs(states[idx, :, :, state]) ** 2 / cell_area
            last = ax.pcolormesh(ye, xe, density, shading="flat", cmap=cmaps[state], vmin=0.0, vmax=vmax / cell_area)
            ax.scatter(np.tile(y, len(x)), np.repeat(x, len(y)), s=1.5, color="white", alpha=0.18, linewidths=0)
            if state == 0:
                ax.set_title(f"{times[idx]:g} fs")
            if col == 0:
                ax.set_ylabel(f"{labels[state]}\nQ_c")
            if state == 2:
                ax.set_xlabel("Q_t")
            ax.text(
                0.04,
                0.93,
                f"P={pops[idx, state]:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 1.8},
            )
            ax.set_aspect("equal", adjustable="box")
    fig.colorbar(last, ax=axes, shrink=0.84, label="DVR cell density")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_surfaces(x, y, adiabatic_energies, outpath):
    adiabatic = adiabatic_energies * au2ev
    yg, xg = np.meshgrid(y, x, indexing="xy")

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.5), constrained_layout=True)
    panels = [
        ("adiabatic S0 / eV", adiabatic[:, :, 0], "Greys"),
        ("adiabatic S1 / eV", adiabatic[:, :, 1], "viridis"),
        ("adiabatic S2 / eV", adiabatic[:, :, 2], "magma"),
        ("S2-S1 gap / eV", adiabatic[:, :, 2] - adiabatic[:, :, 1], "cividis"),
    ]
    for ax, (title, values, cmap) in zip(axes, panels):
        im = ax.contourf(yg, xg, values, levels=32, cmap=cmap)
        ax.contour(yg, xg, values, levels=8, colors="k", linewidths=0.25, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("Q_t")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("Q_c")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npts", type=int, default=49)
    parser.add_argument("--qmax", type=float, default=8.0)
    parser.add_argument("--tmax-fs", type=float, default=80.0)
    parser.add_argument("--nsnapshots", type=int, default=401)
    parser.add_argument("--initial-state", type=int, default=2)
    parser.add_argument("--representation", choices=("adiabatic", "diabatic"), default="adiabatic")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/private/tmp/pyrazine_vibronic_model"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    result = sine_dvr_ldr_run(
        args.qmax,
        args.npts,
        args.tmax_fs,
        args.nsnapshots,
        args.initial_state,
        representation=args.representation,
    )

    prefix = f"pyrazine_sine_dvr_ldr_{args.representation}"
    pop_png = args.outdir / f"{prefix}_populations.png"
    wav_png = args.outdir / f"{prefix}_wavepackets.png"
    pes_png = args.outdir / f"{prefix}_surfaces.png"
    data_path = args.outdir / f"{prefix}_dynamics.npz"

    plot_populations(result["times_fs"], result["populations"], pop_png, result["representation"])
    plot_wavepackets(result, [0.0, 10.0, 20.0, 40.0, 80.0], wav_png)
    plot_surfaces(result["x"], result["y"], result["adiabatic_energies"], pes_png)
    np.savez_compressed(
        data_path,
        times_fs=result["times_fs"],
        populations=result["populations"],
        x=result["x"],
        y=result["y"],
        x_edges=result["x_edges"],
        y_edges=result["y_edges"],
        states=result["states"],
        potential=result["potential"],
        adiabatic_energies=result["adiabatic_energies"],
        representation=result["representation"],
        dim=result["dim"],
        hamiltonian_nnz=result["hamiltonian_nnz"],
    )

    print(f"[plot populations] {pop_png}")
    print(f"[plot wavepackets] {wav_png}")
    print(f"[plot surfaces] {pes_png}")
    print(f"[data] {data_path}")
    print(f"[size] dim={result['dim']}, H nnz={result['hamiltonian_nnz']}")
    print("[final populations]", np.array2string(result["populations"][-1], precision=8))
    print(
        "[norm] min={:.12f} max={:.12f}".format(
            float(result["populations"].sum(axis=1).min()),
            float(result["populations"].sum(axis=1).max()),
        )
    )


if __name__ == "__main__":
    main()
