#!/usr/bin/env python3
"""Aligned 2D LDR-UED demo for H2O.

The nuclear grid uses a reduced H2O model with symmetric O-H stretch ``r`` and
bend angle ``theta``.  The asymmetric stretch is frozen by setting both O-H
distances equal.  Electronic densities are computed with PyQED RHF and the
native analytic AO Fourier transform.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed import au2angstrom
from pyqed.dvr.dvr_1d import kinetic
from pyqed.qchem.fourier import AOPairFTPlan, has_compiled_ao_ft
from pyqed.qchem.mol import Molecule
from pyqed.ued.ued import UED, electron_density_ft
from pyqed.units import amu2au


MASS_O = 15.999
MASS_H = 1.008
SYMBOLS = ("O", "H", "H")


def h2o_symmetric_coords(r_oh: float, theta: float) -> np.ndarray:
    """Return center-of-mass O,H,H coordinates in bohr."""
    half = 0.5 * theta
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [r_oh * np.sin(half), r_oh * np.cos(half), 0.0],
            [-r_oh * np.sin(half), r_oh * np.cos(half), 0.0],
        ],
        dtype=float,
    )
    masses = np.array([MASS_O, MASS_H, MASS_H], dtype=float)
    coords -= np.einsum("a,ax->x", masses, coords) / masses.sum()
    return coords


def atom_string(coords: np.ndarray) -> str:
    return "; ".join(
        f"{sym} {xyz[0]:.16g} {xyz[1]:.16g} {xyz[2]:.16g}"
        for sym, xyz in zip(SYMBOLS, coords)
    )


def build_h2o(coords: np.ndarray, basis: str) -> Molecule:
    mol = Molecule(atom=atom_string(coords), unit="bohr", basis=basis, spin=0)
    mol.build(driver="builtin", eri="dense", aosym="s1")
    return mol


def momentum_grid(smax_ang: float, n_s: int):
    axis = np.linspace(-smax_ang, smax_ang, n_s)
    sx, sy = np.meshgrid(axis, axis, indexing="ij")
    s_vectors = np.column_stack(
        [
            sx.ravel() * au2angstrom,
            sy.ravel() * au2angstrom,
            np.zeros(n_s * n_s),
        ]
    )
    return axis, axis, s_vectors


class H2OSymmetricBendGrid:
    """Minimal LDR-like grid object for UED(aligned=True)."""

    def __init__(self, r_grid: np.ndarray, theta_grid: np.ndarray):
        self.x = [np.asarray(r_grid, dtype=float), np.asarray(theta_grid, dtype=float)]
        self.nx = [len(axis) for axis in self.x]
        self.ndim = 2
        self.nstates = 1
        self.dx = [
            float(self.x[0][1] - self.x[0][0]),
            float(self.x[1][1] - self.x[1][0]),
        ]
        self.dv = float(np.prod(self.dx))
        self._coords = None

    def internal_to_xyz(self, r_oh: float, theta: float) -> np.ndarray:
        return h2o_symmetric_coords(r_oh, theta)

    def cartesian_grid(self, copy=True) -> np.ndarray:
        if self._coords is None:
            self._coords = np.empty((*self.nx, len(SYMBOLS), 3), dtype=float)
            for i, r_oh in enumerate(self.x[0]):
                for j, theta in enumerate(self.x[1]):
                    self._coords[i, j] = self.internal_to_xyz(float(r_oh), float(theta))
        return np.array(self._coords, copy=True) if copy else self._coords


def scan_electronic_grid(r_grid, theta_grid, s_vectors, basis):
    n_r, n_th, n_s = len(r_grid), len(theta_grid), len(s_vectors)
    energies = np.zeros((n_r, n_th), dtype=float)
    ft_ii = np.zeros((n_r, n_th, 1, n_s), dtype=complex)
    ft_ij = np.zeros((n_r, n_th, 1, 1, n_s), dtype=complex)
    coords_grid = np.empty((n_r, n_th, len(SYMBOLS), 3), dtype=float)

    ref_coords = h2o_symmetric_coords(float(r_grid[n_r // 2]), float(theta_grid[n_th // 2]))
    ref_mol = build_h2o(ref_coords, basis)
    plan = AOPairFTPlan.from_molecule(ref_mol) if has_compiled_ao_ft() else None

    total = n_r * n_th
    count = 0
    for i, r_oh in enumerate(r_grid):
        for j, theta in enumerate(theta_grid):
            count += 1
            coords = h2o_symmetric_coords(float(r_oh), float(theta))
            mol = build_h2o(coords, basis)
            mf = mol.RHF().run(tol=1e-10, conv_tol_dm=1e-8, max_cycle=80)
            dm = mf.make_rdm1()
            dm1 = dm[None, :, :]
            tdm1 = dm[None, None, :, :]
            if plan is not None:
                origins = plan.origins_from_atom_coords(mol.atom_coords())
                ft_diag, ft_trans = plan.contract(
                    dm1,
                    tdm1,
                    s_vectors,
                    origins=origins,
                    compiled=True,
                )
            else:
                ft_diag, ft_trans = electron_density_ft(
                    dm1,
                    tdm1,
                    mol,
                    s_vectors,
                    backend="native",
                    ao_ft_compiled=False,
                )
            energies[i, j] = mf.e_tot
            ft_ii[i, j] = ft_diag
            ft_ij[i, j] = ft_trans
            coords_grid[i, j] = coords
            print(
                f"[electronic] {count:3d}/{total}: "
                f"r={r_oh:.4f} theta={np.rad2deg(theta):.2f} E={mf.e_tot:.10f}"
            )

    return energies, ft_ii, ft_ij, coords_grid


def vibrational_ground_packet(r_grid, theta_grid, pes):
    """Ground packet for a simple diagonal 2D kinetic model."""
    m_o = MASS_O * amu2au
    m_h = MASS_H * amu2au
    mu_oh = m_o * m_h / (m_o + m_h)
    stretch_mass = 2.0 * mu_oh
    r_eq = float(r_grid[np.unravel_index(np.argmin(pes), pes.shape)[0]])
    bend_inertia = 0.5 * m_h * r_eq**2

    t_r = kinetic(r_grid, mass=stretch_mass, dvr="sine")
    t_th = kinetic(theta_grid, mass=bend_inertia, dvr="sine")
    h = (
        np.kron(t_r, np.eye(len(theta_grid)))
        + np.kron(np.eye(len(r_grid)), t_th)
        + np.diag(pes.ravel())
    )
    h = 0.5 * (h + h.T)
    evals, evecs = np.linalg.eigh(h)
    chi = evecs[:, 0].reshape(len(r_grid), len(theta_grid)).astype(complex)
    dv = float((r_grid[1] - r_grid[0]) * (theta_grid[1] - theta_grid[0]))
    chi /= np.sqrt(np.sum(np.abs(chi) ** 2) * dv)
    return chi, float(evals[0])


def plot_results(path, r_grid, theta_grid, sx, sy, pes, chi, intensity):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), dpi=200)

    theta_deg = np.rad2deg(theta_grid)
    extent_rt = [r_grid[0], r_grid[-1], theta_deg[0], theta_deg[-1]]
    pes_rel = (pes - pes.min()) * 27.211386245988
    im0 = axes[0].imshow(
        pes_rel.T,
        origin="lower",
        extent=extent_rt,
        aspect="auto",
        cmap="viridis",
    )
    axes[0].set_xlabel("O-H r (bohr)")
    axes[0].set_ylabel("H-O-H angle (deg)")
    axes[0].set_title("RHF PES")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="eV")

    density = np.abs(chi) ** 2
    im1 = axes[1].imshow(
        density.T,
        origin="lower",
        extent=extent_rt,
        aspect="auto",
        cmap="magma",
    )
    axes[1].set_xlabel("O-H r (bohr)")
    axes[1].set_title("2D ground packet")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        intensity.T,
        origin="lower",
        extent=[sx[0], sx[-1], sy[0], sy[-1]],
        aspect="equal",
        cmap="inferno",
    )
    axes[2].set_xlabel(r"$s_x$ ($\AA^{-1}$)")
    axes[2].set_ylabel(r"$s_y$ ($\AA^{-1}$)")
    axes[2].set_title("Aligned UED")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--n-r", type=int, default=9)
    parser.add_argument("--n-theta", type=int, default=9)
    parser.add_argument("--r-min", type=float, default=1.65)
    parser.add_argument("--r-max", type=float, default=2.10)
    parser.add_argument("--theta-min-deg", type=float, default=90.0)
    parser.add_argument("--theta-max-deg", type=float, default=120.0)
    parser.add_argument("--n-s", type=int, default=81)
    parser.add_argument("--s-max", type=float, default=8.0, help="A^-1")
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/h2o_ldr_ued_2d"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    r_grid = np.linspace(args.r_min, args.r_max, args.n_r)
    theta_grid = np.deg2rad(
        np.linspace(args.theta_min_deg, args.theta_max_deg, args.n_theta)
    )
    sx, sy, s_vectors = momentum_grid(args.s_max, args.n_s)

    pes, ft_ii, ft_ij, coords_grid = scan_electronic_grid(
        r_grid,
        theta_grid,
        s_vectors,
        args.basis,
    )
    chi, vib_energy = vibrational_ground_packet(r_grid, theta_grid, pes)

    ldr = H2OSymmetricBendGrid(r_grid, theta_grid)
    ued = UED(
        ldr=ldr,
        symbols=SYMBOLS,
        s_vectors=s_vectors,
        electronic_fts=ft_ij,
        aligned=True,
    )
    signal = ued.run(chi, verbose=True)
    intensity = signal["I_signal"][0].reshape(args.n_s, args.n_s)
    scale = intensity.max()
    if scale > 0:
        intensity = intensity / scale

    npz_path = args.outdir / "h2o_ldr_ued_2d.npz"
    np.savez_compressed(
        npz_path,
        r_bohr=r_grid,
        theta_rad=theta_grid,
        sx_angstrom_inv=sx,
        sy_angstrom_inv=sy,
        s_vectors_bohr_inv=s_vectors,
        coords_bohr=coords_grid,
        pes=pes,
        rho_el_FT_ii=ft_ii,
        rho_el_FT_ij=ft_ij,
        chi=chi,
        vib_energy=vib_energy,
        I_total=signal["I_total"][0].reshape(args.n_s, args.n_s),
        I_nuc=signal["I_nuc"][0].reshape(args.n_s, args.n_s),
        I_el=signal["I_el"][0].reshape(args.n_s, args.n_s),
        I_cross=signal["I_cross"][0].reshape(args.n_s, args.n_s),
        I_signal=intensity,
        norm=signal["norms"][0],
    )

    png_path = args.outdir / "h2o_ldr_ued_2d.png"
    plot_results(png_path, r_grid, theta_grid, sx, sy, pes, chi, intensity)

    print(f"npz={npz_path}")
    print(f"png={png_path}")
    print(f"vib_energy={vib_energy:.10f}")
    print(f"norm={signal['norms'][0]:.10f}")
    print(f"I range={intensity.min():.8e} {intensity.max():.8e}")


if __name__ == "__main__":
    main()
