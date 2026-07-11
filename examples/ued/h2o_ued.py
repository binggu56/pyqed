#!/usr/bin/env python3
"""Minimal aligned H2O UED example with CASCI electronic densities."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed import au2angstrom, au2fs
from pyqed.namd.triatomic import Triatom
from pyqed.ued.ued import UED


SYMBOLS = ("H", "O", "H")
R_EQ = 1.81
THETA_EQ = np.deg2rad(104.5)
R_RANGE = (1.65, 2.10)
THETA_RANGE = np.deg2rad((90.0, 120.0))
BASIS = "sto-3g"
N_R = 3
N_THETA = 3
N_S = 41
S_MAX = 8.0  # Angstrom^-1
NCAS = 4
NELECAS = 4
NSTATES = 1
STATE = 0
DT_FS = 0.2
TMAX_FS = 10.0
UED_DT_FS = 1.0
OUTDIR = Path("/private/tmp/h2o_ued")


def h2o_xyz(r1=R_EQ, r2=R_EQ, theta=THETA_EQ):
    return np.array(
        [
            [r1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [r2 * np.cos(theta), r2 * np.sin(theta), 0.0],
        ]
    )


def make_triatom(n_r, n_theta, basis, nstates):
    triatom = Triatom(
        [["H", tuple(h2o_xyz()[0])], ["O", (0.0, 0.0, 0.0)], ["H", tuple(h2o_xyz()[2])]],
        basis=basis,
        nstates=nstates,
        charge=0,
        spin=0,
        unit="bohr",
    )
    triatom.set_dvr(
        domains=[R_RANGE, R_RANGE, THETA_RANGE],
        npts=[n_r, n_r, n_theta],
        dvr_type="sine",
    )
    return triatom


def detector(s_max, n_s):
    axis = np.linspace(-s_max, s_max, n_s)
    sx, sy = np.meshgrid(axis, axis, indexing="ij")
    s = np.column_stack(
        [sx.ravel() * au2angstrom, sy.ravel() * au2angstrom, np.zeros(n_s * n_s)]
    )
    return axis, axis, s


def gaussian_packet(triatom, nstates, state):
    r1, r2, theta = np.meshgrid(*triatom.x, indexing="ij")
    packet = np.exp(
        -70.0 * (r1 - R_EQ) ** 2
        -70.0 * (r2 - R_EQ) ** 2
        -70.0 * (theta - THETA_EQ) ** 2
    ).astype(complex)
    packet /= np.sqrt(np.sum(np.abs(packet) ** 2))

    psi = np.zeros((*triatom.nx, nstates), dtype=complex)
    psi[..., state] = packet
    return psi


def plot(path, sx, sy, times_fs, intensity):
    frames = np.unique(np.linspace(0, len(times_fs) - 1, min(3, len(times_fs)), dtype=int))
    fig, axes = plt.subplots(1, len(frames), figsize=(4.1 * len(frames), 3.8), dpi=200)
    axes = np.atleast_1d(axes)
    vmax = max(float(intensity.max()), 1e-30)

    for ax, frame in zip(axes, frames):
        im = ax.imshow(
            (intensity[frame] / vmax).T,
            origin="lower",
            extent=[sx[0], sx[-1], sy[0], sy[-1]],
            aspect="equal",
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set(
            xlabel="sx (A^-1)",
            ylabel="sy (A^-1)",
            title=f"t = {times_fs[frame]:.1f} fs",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    if not 0 <= STATE < NSTATES:
        raise ValueError("STATE must satisfy 0 <= STATE < NSTATES.")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    triatom = make_triatom(N_R, N_THETA, BASIS, NSTATES)
    sx, sy, s = detector(S_MAX, N_S)
    psi0 = gaussian_packet(triatom, NSTATES, STATE)
    cwd = Path.cwd()
    try:
        os.chdir(OUTDIR)
        triatom.scan_pes(
            basis=BASIS,
            nstates=NSTATES,
            ncas=NCAS,
            nelecas=NELECAS,
            overlap_method="link-only",
            electronic_method="casci",
        )
    finally:
        os.chdir(cwd)

    nt = int(round(TMAX_FS / DT_FS))
    nout = max(1, int(round(UED_DT_FS / DT_FS)))
    result = triatom.run(
        psi0,
        dt=DT_FS / au2fs,
        nt=nt,
        nout=nout,
        kinetic_propagator="dense",
    )

    ued = UED(triatom, aligned=True)
    signal = ued.run(s, verbose=True)
    times_fs = np.asarray(signal["times"], dtype=float) * au2fs
    intensity = signal["I_signal"].reshape(len(times_fs), N_S, N_S)
    intensity_plot = intensity / max(float(intensity.max()), 1e-30)

    npz_path = OUTDIR / "h2o_ued.npz"
    png_path = OUTDIR / "h2o_ued.png"
    np.savez_compressed(
        npz_path,
        sx_angstrom_inv=sx,
        sy_angstrom_inv=sy,
        s_bohr_inv=s,
        r1_bohr=triatom.x[0],
        r2_bohr=triatom.x[1],
        theta_rad=triatom.x[2],
        times_fs=times_fs,
        psi0=psi0,
        psilist=np.asarray(result["psilist"]),
        pes=triatom.apes,
        rho_el_FT_ij=ued.electronic_fts,
        I_signal=intensity,
        I_signal_normalized=intensity_plot,
        I_total=signal["I_total"].reshape(len(times_fs), N_S, N_S),
        norms=signal["norms"],
    )
    plot(png_path, sx, sy, times_fs, intensity)

    print(f"npz={npz_path}")
    print(f"png={png_path}")
    print(f"norm range={signal['norms'].min():.10f} {signal['norms'].max():.10f}")


if __name__ == "__main__":
    main()
