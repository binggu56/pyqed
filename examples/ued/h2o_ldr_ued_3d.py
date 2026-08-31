#!/usr/bin/env python3
"""Aligned 3D LDR-UED demo for H2O using the analytical triatomic G-matrix.

The nuclear grid uses the full triatomic internal-coordinate space
``(r1, r2, theta)`` for H-O-H.  The nuclear kinetic energy is built by
``pyqed.namd.triatomic.Triatom.buildK()``, which includes the analytical
triatomic G-matrix and coordinate-coupling terms for J=0.
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

from pyqed.units import au2ev
from pyqed import au2angstrom
from pyqed.namd.triatomic import Triatom
from pyqed.qchem.fourier import AOPairFTPlan, has_compiled_ao_ft
from pyqed.qchem.mol import Molecule
from pyqed.ued.ued import UED, electron_density_ft


SYMBOLS = ("H", "O", "H")


def build_h2o(coords: np.ndarray, basis: str) -> Molecule:
    atom = "; ".join(
        f"{sym} {xyz[0]:.16g} {xyz[1]:.16g} {xyz[2]:.16g}"
        for sym, xyz in zip(SYMBOLS, coords)
    )
    mol = Molecule(atom=atom, unit="bohr", basis=basis, spin=0)
    mol.build(eri="dense", aosym="s1")
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


def make_triatom(args) -> Triatom:
    r_eq = 1.81
    theta_eq = np.deg2rad(104.5)
    atom = [
        ["H", (r_eq, 0.0, 0.0)],
        ["O", (0.0, 0.0, 0.0)],
        ["H", (r_eq * np.cos(theta_eq), r_eq * np.sin(theta_eq), 0.0)],
    ]
    mol = Triatom(atom, basis=args.basis, nstates=1, charge=0, spin=0, unit="bohr")
    mol.set_dvr(
        domains=[
            [args.r1_min, args.r1_max],
            [args.r2_min, args.r2_max],
            [np.deg2rad(args.theta_min_deg), np.deg2rad(args.theta_max_deg)],
        ],
        npts=[args.n_r1, args.n_r2, args.n_theta],
        dvr_type=args.dvr_type,
    )
    return mol


def scan_electronic_grid(triatom: Triatom, s_vectors: np.ndarray, basis: str):
    n_s = len(s_vectors)
    pes = np.zeros(tuple(triatom.nx), dtype=float)
    ft_ii = np.zeros((*triatom.nx, 1, n_s), dtype=complex)
    ft_ij = np.zeros((*triatom.nx, 1, 1, n_s), dtype=complex)
    coords_grid = triatom.cartesian_grid(copy=False)

    ref_idx = tuple(n // 2 for n in triatom.nx)
    ref_mol = build_h2o(coords_grid[ref_idx], basis)
    plan = AOPairFTPlan.from_molecule(ref_mol) if has_compiled_ao_ft() else None

    total = int(np.prod(triatom.nx))
    for count, idx in enumerate(np.ndindex(*triatom.nx), start=1):
        coords = coords_grid[idx]
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
        pes[idx] = mf.e_tot
        ft_ii[idx] = ft_diag
        ft_ij[idx] = ft_trans
        q = [triatom.x[axis][idx[axis]] for axis in range(triatom.ndim)]
        print(
            f"[electronic] {count:4d}/{total}: "
            f"r1={q[0]:.4f} r2={q[1]:.4f} "
            f"theta={np.rad2deg(q[2]):.2f} E={mf.e_tot:.10f}"
        )

    return pes, ft_ii, ft_ij, np.array(coords_grid, copy=True)


def vibrational_ground_packet(triatom: Triatom, pes: np.ndarray, sparse=False):
    triatom.apes = pes[..., None]
    kinetic = triatom.buildK(sparse=sparse)
    if sparse:
        from scipy.sparse import diags
        from scipy.sparse.linalg import eigsh

        h = kinetic + diags(pes.ravel(), format="csr")
        evals, evecs = eigsh(h, k=1, which="SA")
        order = np.argsort(evals)
        energy = float(evals[order[0]])
        chi = evecs[:, order[0]].reshape(*triatom.nx)
    else:
        h = kinetic + np.diag(pes.ravel())
        h = 0.5 * (h + h.conj().T)
        evals, evecs = np.linalg.eigh(h)
        energy = float(evals[0].real)
        chi = evecs[:, 0].reshape(*triatom.nx)

    chi = chi.astype(complex, copy=False)
    chi /= np.sqrt(np.sum(np.abs(chi) ** 2))
    return chi[..., None], energy


def plot_results(path, triatom, sx, sy, pes, chi, intensity):
    r1_grid, r2_grid, theta_grid = triatom.x
    theta_mid = len(theta_grid) // 2
    r2_mid = len(r2_grid) // 2
    theta_deg = np.rad2deg(theta_grid)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), dpi=200)

    pes_slice = (pes[:, :, theta_mid] - pes.min()) * au2ev
    im0 = axes[0].imshow(
        pes_slice.T,
        origin="lower",
        extent=[r1_grid[0], r1_grid[-1], r2_grid[0], r2_grid[-1]],
        aspect="auto",
        cmap="viridis",
    )
    axes[0].set_xlabel("O-H r1 (bohr)")
    axes[0].set_ylabel("O-H r2 (bohr)")
    axes[0].set_title(f"PES at theta={theta_deg[theta_mid]:.1f} deg")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="eV")

    density_slice = np.abs(chi[:, r2_mid, :, 0]) ** 2
    im1 = axes[1].imshow(
        density_slice.T,
        origin="lower",
        extent=[r1_grid[0], r1_grid[-1], theta_deg[0], theta_deg[-1]],
        aspect="auto",
        cmap="magma",
    )
    axes[1].set_xlabel("O-H r1 (bohr)")
    axes[1].set_ylabel("H-O-H angle (deg)")
    axes[1].set_title(f"packet at r2={r2_grid[r2_mid]:.2f} bohr")
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
    axes[2].set_title("Aligned 3D LDR-UED")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--dvr-type", default="sine")
    parser.add_argument("--n-r1", type=int, default=5)
    parser.add_argument("--n-r2", type=int, default=5)
    parser.add_argument("--n-theta", type=int, default=5)
    parser.add_argument("--r1-min", type=float, default=1.65)
    parser.add_argument("--r1-max", type=float, default=2.10)
    parser.add_argument("--r2-min", type=float, default=1.65)
    parser.add_argument("--r2-max", type=float, default=2.10)
    parser.add_argument("--theta-min-deg", type=float, default=90.0)
    parser.add_argument("--theta-max-deg", type=float, default=120.0)
    parser.add_argument("--n-s", type=int, default=61)
    parser.add_argument("--s-max", type=float, default=8.0, help="A^-1")
    parser.add_argument("--sparse-kinetic", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/h2o_ldr_ued_3d"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    triatom = make_triatom(args)
    sx, sy, s_vectors = momentum_grid(args.s_max, args.n_s)

    pes, ft_ii, ft_ij, coords = scan_electronic_grid(triatom, s_vectors, args.basis)
    chi, vib_energy = vibrational_ground_packet(
        triatom,
        pes,
        sparse=args.sparse_kinetic,
    )

    signal = UED(
        ldr=triatom,
        s_vectors=s_vectors,
        electronic_fts=ft_ij,
        aligned=True,
    ).run(chi, verbose=True)
    intensity = signal["I_signal"][0].reshape(args.n_s, args.n_s)
    scale = intensity.max()
    if scale > 0:
        intensity = intensity / scale

    npz_path = args.outdir / "h2o_ldr_ued_3d.npz"
    np.savez_compressed(
        npz_path,
        r1_bohr=triatom.x[0],
        r2_bohr=triatom.x[1],
        theta_rad=triatom.x[2],
        sx_angstrom_inv=sx,
        sy_angstrom_inv=sy,
        s_vectors_bohr_inv=s_vectors,
        coords_bohr=coords,
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

    png_path = args.outdir / "h2o_ldr_ued_3d.png"
    plot_results(png_path, triatom, sx, sy, pes, chi, intensity)

    print(f"npz={npz_path}")
    print(f"png={png_path}")
    print(f"vib_energy={vib_energy:.10f}")
    print(f"norm={signal['norms'][0]:.10f}")
    print(f"I range={intensity.min():.8e} {intensity.max():.8e}")


if __name__ == "__main__":
    main()
