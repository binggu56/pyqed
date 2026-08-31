#!/usr/bin/env python3
"""Scan conventional GTO-RHF and GDVR-RHF PESs for linear H4."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.gdvr import AtomicChain
from pyqed.qchem.gdvr.rhf import (
    _get_newton_context,
    eri_JK_from_kernels_M1,
    rebuild_Hcore_from_d,
)
from pyqed.qchem.hf import RHF


Z_OUTER = 2.0
Z_INNER = 2.0 / 3.0


def geometry(q_plus, q_minus):
    """Return z positions for collective translations of H2 and H3."""
    root2 = np.sqrt(2.0)
    z2 = -Z_INNER + (q_plus + q_minus) / root2
    z3 = Z_INNER + (q_plus - q_minus) / root2
    return np.array([-Z_OUTER, z2, z3, Z_OUTER], dtype=float)


def atom_string(z):
    return "; ".join(f"H 0 0 {value:.16g}" for value in z)


def gto_rhf_energy(z, basis, conv):
    mol = Molecule(atom=atom_string(z), unit="bohr", basis=basis)
    mol.build()
    mf = RHF(mol).run(tol=conv, verbose=0)
    if not mf.converged:
        raise RuntimeError("GTO RHF did not converge")
    return float(mf.e_tot)


def set_transverse_guess(mol, coefficients):
    """Rebuild a new geometry with transverse coefficients from a neighbor."""
    ctx = _get_newton_context(mol)
    d_stack = np.asarray(coefficients, float).copy()
    overlap = ctx["S_prim"]
    expected = (int(mol.shapes["Nz"]), overlap.shape[0])
    if d_stack.shape != expected:
        raise ValueError(f"Transverse continuation guess has shape {d_stack.shape}, expected {expected}")
    for n, vector in enumerate(d_stack):
        d_stack[n] = vector / np.sqrt(float(vector @ overlap @ vector))
    mol.c_list = [vector[:, None] for vector in d_stack]
    mol.hcore = rebuild_Hcore_from_d(
        d_stack,
        ctx["z"],
        ctx["Kz"],
        overlap,
        ctx["T_prim"],
        ctx["alphas"],
        ctx["centers"],
        ctx["labels"],
        ctx["nuclei"],
        h_local_ops=ctx["h_local_ops"],
        h1_nm=ctx["h1_nm"],
    )
    mol.eri_j, mol.eri_k = eri_JK_from_kernels_M1(
        mol.c_list, ctx["K_h"], ctx["Kx_h"]
    )


def gdvr_rhf_energy(z, args, transverse_guess=None, dm0=None):
    coords = [[0.0, 0.0, value] for value in z]
    mol = AtomicChain(elements=["H"] * 4, coords=coords)
    mol.build(
        Lz=args.lz,
        Nz=args.nz,
        M=1,
        transverse_basis=args.transverse_basis,
        dvr_method="sine",
        verbose=False,
    )
    if transverse_guess is not None:
        set_transverse_guess(mol, transverse_guess)
    mf = mol.RHF().run(
        conv=args.conv,
        max_iter=100,
        newton=False,
        verbose=False,
        dm0=dm0,
    )
    mf.newton(
        tol=args.gdvr_tol,
        max_cycles=args.gdvr_cycles,
        sweeps=args.gdvr_sweeps,
        ridge=0.5,
        trust_step=0.5,
        trust_radius=1.0,
        scf_conv=args.conv,
        scf_max_iter=100,
        verbose=False,
    )
    history = np.asarray(mf.info["newton_energy_history"], float)
    return (
        float(mf.e_tot),
        bool(mf.info.get("newton_converged", False)),
        int(mf.info["newton_cycles"]),
        float(abs(history[-1] - history[-2])),
        np.vstack([np.asarray(c[:, 0], float) for c in mf.mol.c_list]),
        np.asarray(mf.dm, float),
    )


def save_scan(
    path,
    q_plus,
    q_minus,
    atomic_z,
    gto,
    gdvr,
    converged,
    cycles,
    delta,
    args,
    transverse_coefficients=None,
    density_matrices=None,
):
    data = dict(
        q_plus=q_plus,
        q_minus=q_minus,
        atomic_z=atomic_z,
        gto_rhf_energy=gto,
        gdvr_rhf_energy=gdvr,
        gdvr_converged=converged,
        gdvr_newton_cycles=cycles,
        gdvr_newton_delta=delta,
        gdvr_newton_tol=args.gdvr_tol,
        gdvr_newton_max_cycles=args.gdvr_cycles,
        gdvr_newton_sweeps=args.gdvr_sweeps,
        basis=args.basis,
        transverse_basis=args.transverse_basis,
        lz=args.lz,
        nz=args.nz,
        continuation=args.continuation,
    )
    if transverse_coefficients is not None:
        data["gdvr_transverse_coefficients"] = transverse_coefficients
        data["gdvr_density_matrices"] = density_matrices
    np.savez_compressed(path, **data)


def scan_order(q_plus, q_minus):
    """Visit an anchor near the minimum, then propagate to adjacent points."""
    i_order = np.argsort(np.abs(q_plus), kind="stable")
    j_anchor = int(np.argmin(np.abs(q_minus + 0.25)))
    points = []
    for i in i_order:
        points.append((int(i), j_anchor))
        for offset in range(1, len(q_minus)):
            if j_anchor - offset >= 0:
                points.append((int(i), j_anchor - offset))
            if j_anchor + offset < len(q_minus):
                points.append((int(i), j_anchor + offset))
    return points


def nearest_saved_point(i, j, q_plus, q_minus, available):
    candidates = np.argwhere(available)
    if candidates.size == 0:
        return None
    distances = (
        (q_plus[candidates[:, 0]] - q_plus[i]) ** 2
        + (q_minus[candidates[:, 1]] - q_minus[j]) ** 2
    )
    return tuple(candidates[int(np.argmin(distances))])


def plot_surfaces(path, q_plus, q_minus, gto, gdvr, args):
    gto_rel = 1000.0 * (gto - np.nanmin(gto))
    gdvr_rel = 1000.0 * (gdvr - np.nanmin(gdvr))
    shape_delta = gdvr_rel - gto_rel
    if len(q_plus) == 1:
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), constrained_layout=True)
        axes[0].plot(q_minus, gto_rel[0], "o-", label="GTO-RHF")
        axes[0].plot(q_minus, gdvr_rel[0], "o-", label="GDVR-RHF")
        axes[0].set_ylabel("Relative energy (mEh)")
        axes[0].legend(frameon=False)
        axes[1].plot(q_minus, shape_delta[0], "o-", color="tab:red")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_ylabel("GDVR − GTO shape (mEh)")
        for ax in axes:
            ax.set_xlabel(r"$q_-$ (bohr)")
            ax.grid(alpha=0.2)
        fig.suptitle(
            rf"Linear H$_4$, $q_+={q_plus[0]:g}$; $N_z={args.nz}$, "
            rf"$L_z={args.lz:g}$ bohr"
        )
        fig.savefig(path, dpi=240)
        plt.close(fig)
        return

    x, y = np.meshgrid(q_minus, q_plus)
    energy_levels = np.linspace(0.0, max(gto_rel.max(), gdvr_rel.max()), 25)
    delta_bound = max(abs(shape_delta.min()), abs(shape_delta.max()))
    panels = (
        (
            gto_rel,
            "GTO-RHF relative PES",
            "viridis",
            "Relative energy (mEh)",
            energy_levels,
        ),
        (
            gdvr_rel,
            f"GDVR-RHF ({args.transverse_basis} transverse)",
            "viridis",
            "Relative energy (mEh)",
            energy_levels,
        ),
        (
            shape_delta,
            "GDVR − GTO PES shape",
            "coolwarm",
            "Difference (mEh)",
            np.linspace(-delta_bound, delta_bound, 25),
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    for ax, (values, title, cmap, label, levels) in zip(axes, panels):
        image = ax.contourf(x, y, values, levels=levels, cmap=cmap)
        ax.scatter(x, y, s=9, color="black", alpha=0.45)
        ax.set_title(title)
        ax.set_xlabel(r"$q_-$ (bohr)")
        ax.set_ylabel(r"$q_+$ (bohr)")
        fig.colorbar(image, ax=ax, label=label)
    fig.suptitle(
        rf"Linear H$_4$: $z_1=-2$, $z_4=+2$ bohr; "
        rf"$N_z={args.nz}$, $L_z={args.lz:g}$ bohr"
    )
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-plus-min", type=float, default=-0.5)
    parser.add_argument("--q-plus-max", type=float, default=0.5)
    parser.add_argument("--nq-plus", type=int, default=5)
    parser.add_argument("--q-minus-min", type=float, default=-0.5)
    parser.add_argument("--q-minus-max", type=float, default=0.5)
    parser.add_argument("--nq-minus", type=int, default=5)
    parser.add_argument("--basis", default="d-aug-cc-pvdz")
    parser.add_argument("--transverse-basis", default="d-aug-cc-pvdz")
    parser.add_argument("--lz", type=float, default=6.0)
    parser.add_argument("--nz", type=int, default=41)
    parser.add_argument("--conv", type=float, default=1.0e-8)
    parser.add_argument("--gdvr-tol", type=float, default=1.0e-6)
    parser.add_argument("--gdvr-cycles", type=int, default=180)
    parser.add_argument("--gdvr-sweeps", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--independent-starts",
        action="store_false",
        dest="continuation",
        help="disable nearest-neighbor continuation of Newton transverse coefficients",
    )
    parser.set_defaults(continuation=True)
    parser.add_argument(
        "--output", type=Path, default=Path("h4_collective_hf_gdvr_newton_pes.npz")
    )
    args = parser.parse_args()

    if args.nq_plus < 1 or args.nq_minus < 1:
        raise ValueError("nq-plus and nq-minus must be positive")
    q_plus = (
        np.array([0.5 * (args.q_plus_min + args.q_plus_max)])
        if args.nq_plus == 1
        else np.linspace(args.q_plus_min, args.q_plus_max, args.nq_plus)
    )
    q_minus = (
        np.array([0.5 * (args.q_minus_min + args.q_minus_max)])
        if args.nq_minus == 1
        else np.linspace(args.q_minus_min, args.q_minus_max, args.nq_minus)
    )
    shape = (args.nq_plus, args.nq_minus)
    gto = np.full(shape, np.nan)
    gdvr = np.full_like(gto, np.nan)
    gdvr_converged = np.zeros_like(gto, dtype=bool)
    gdvr_cycles = np.zeros_like(gto, dtype=int)
    gdvr_delta = np.full_like(gto, np.nan)
    atomic_z = np.empty((*shape, 4))
    transverse_coefficients = None
    density_matrices = None
    if args.resume and args.output.exists():
        previous = np.load(args.output)
        if not (
            np.array_equal(previous["q_plus"], q_plus)
            and np.array_equal(previous["q_minus"], q_minus)
        ):
            raise ValueError("Cannot resume: saved and requested collective-coordinate grids differ")
        gto[:] = previous["gto_rhf_energy"]
        gdvr[:] = previous["gdvr_rhf_energy"]
        gdvr_converged[:] = previous["gdvr_converged"]
        gdvr_cycles[:] = previous["gdvr_newton_cycles"]
        gdvr_delta[:] = previous["gdvr_newton_delta"]
        atomic_z[:] = previous["atomic_z"]
        if args.continuation:
            transverse_coefficients = previous["gdvr_transverse_coefficients"].copy()
            density_matrices = previous["gdvr_density_matrices"].copy()
    started = time.perf_counter()
    total = args.nq_plus * args.nq_minus

    points = scan_order(q_plus, q_minus) if args.continuation else list(np.ndindex(shape))
    for i, j in points:
        qp = q_plus[i]
        qm = q_minus[j]
        if np.isfinite(gto[i, j]) and np.isfinite(gdvr[i, j]) and gdvr_converged[i, j]:
            print(f"[{i * args.nq_minus + j + 1:3d}/{total}] resumed", flush=True)
            continue
        z = geometry(qp, qm)
        if not np.all(np.diff(z) > 0.0):
            raise ValueError(f"Atom crossing at q_plus={qp}, q_minus={qm}")
        atomic_z[i, j] = z
        point_started = time.perf_counter()
        gto[i, j] = gto_rhf_energy(z, args.basis, args.conv)
        guess_index = None
        if args.continuation and transverse_coefficients is not None:
            guess_index = nearest_saved_point(
                i,
                j,
                q_plus,
                q_minus,
                gdvr_converged,
            )
        transverse_guess = (
            None if guess_index is None else transverse_coefficients[guess_index]
        )
        dm0 = None if guess_index is None else density_matrices[guess_index]
        (
            gdvr[i, j],
            gdvr_converged[i, j],
            gdvr_cycles[i, j],
            gdvr_delta[i, j],
            point_coefficients,
            point_density,
        ) = gdvr_rhf_energy(z, args, transverse_guess=transverse_guess, dm0=dm0)
        if transverse_coefficients is None:
            transverse_coefficients = np.full((*shape, *point_coefficients.shape), np.nan)
            density_matrices = np.full((*shape, *point_density.shape), np.nan)
        transverse_coefficients[i, j] = point_coefficients
        density_matrices[i, j] = point_density
        save_scan(
            args.output,
            q_plus,
            q_minus,
            atomic_z,
            gto,
            gdvr,
            gdvr_converged,
            gdvr_cycles,
            gdvr_delta,
            args,
            transverse_coefficients,
            density_matrices,
        )
        print(
            f"[{i * args.nq_minus + j + 1:3d}/{total}] "
            f"q+= {qp:+.3f} q-= {qm:+.3f} "
            f"GTO={gto[i, j]:+.10f} GDVR={gdvr[i, j]:+.10f} "
            f"Newton={'ok' if gdvr_converged[i, j] else 'unconverged'} "
            f"cycles={gdvr_cycles[i, j]} dE={gdvr_delta[i, j]:.2e} "
            f"seed={'none' if guess_index is None else guess_index} "
            f"({time.perf_counter() - point_started:.2f} s)",
            flush=True,
        )

    save_scan(
        args.output,
        q_plus,
        q_minus,
        atomic_z,
        gto,
        gdvr,
        gdvr_converged,
        gdvr_cycles,
        gdvr_delta,
        args,
        transverse_coefficients,
        density_matrices,
    )
    figure = args.output.with_suffix(".png")
    plot_surfaces(figure, q_plus, q_minus, gto, gdvr, args)
    print(f"Saved {args.output}")
    print(f"Saved {figure}")
    print(f"Elapsed {time.perf_counter() - started:.2f} s")


if __name__ == "__main__":
    main()
