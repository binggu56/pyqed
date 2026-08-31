#!/usr/bin/env python3
"""Minimal A + BC Jacobi-coordinate ab initio calculation with unified LDR.

This is intentionally tiny: (1) build a 3D H + OH type Jacobi DVR grid,
`(r, R, gamma)`, (2) scan a CASCI PES plus linked LDR overlaps,
(3) run one LDR propagation step unless `--no-propagate` is given.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.namd.triatomic import Triatom
from pyqed.dvr import LegendreDVR, SineDVR
from pyqed.units import au2fs


def h3plus_jacobi_ref(r: float = 1.5, R: float = 3.2, gamma: float = np.pi / 3.0):
    """Reference geometry in cartesian coordinates (bohr).

    Atom order is always interpreted as A + BC for
    ``coordinates='jacobi'`` in the internal scan mapping:

    ``q = (r, R, gamma)``.
    """
    return [
        ["H", (float(R * np.cos(gamma)), float(R * np.sin(gamma)), 0.0)],
        ["H", (-0.75, 0.0, 0.0)],
        ["H", (0.75, 0.0, 0.0)],
    ]


def _state_packet(
    solver: Triatom,
    state: int,
    width: float = 5.0,
    widths=None,
    momenta=None,
    center=None,
    use_reference_projection: bool = True,
):
    """Construct a normalized vibronic Gaussian in the nuclear tensor grid."""
    center = np.array([axis[len(axis) // 2] for axis in solver.x], dtype=float) if center is None else np.asarray(center, dtype=float)
    if center.shape != (solver.ndim,):
        raise ValueError("center must have one value per coordinate.")
    if use_reference_projection:
        return solver.projected_initial_packet(
            state=state,
            width=width,
            widths=widths,
            center=center,
            momenta=momenta,
        )

    widths = np.ones(solver.ndim, dtype=float) * width if widths is None else np.asarray(widths, dtype=float)
    if widths.shape != (solver.ndim,):
        raise ValueError("widths must match the number of dimensions.")
    momenta = (
        np.zeros(solver.ndim, dtype=float)
        if momenta is None
        else np.asarray(momenta, dtype=float)
    )
    if momenta.shape != (solver.ndim,):
        raise ValueError("momenta must match the number of dimensions.")
    psi = np.zeros((*solver.nx, solver.nstates), dtype=complex)
    for idx in np.ndindex(*solver.nx):
        q = np.array([solver.x[i][idx[i]] for i in range(solver.ndim)], dtype=float)
        displacement = q - center
        amp = np.exp(
            -np.dot(widths, displacement**2)
            + 1j * np.dot(momenta, displacement)
        )
        psi[idx + (state,)] = amp
    psi = solver.to_quadrature_normalized(psi)
    norm = solver.norm(psi)
    if norm == 0.0:
        raise RuntimeError("initial wavepacket norm is zero")
    return psi / norm


def cap_profile_1d(x, width_min, width_max, strength, order):
    x = np.asarray(x, dtype=float)
    cap = np.zeros_like(x, dtype=float)
    if strength <= 0.0:
        return cap
    if width_min > 0.0:
        left = float(np.asarray(x)[0]) + float(width_min)
        mask = x < left
        cap[mask] += strength * ((left - x[mask]) / float(width_min)) ** int(order)
    if width_max > 0.0:
        right = float(np.asarray(x)[-1]) - float(width_max)
        mask = x > right
        cap[mask] += strength * ((x[mask] - right) / float(width_max)) ** int(order)
    return cap


def build_cap_profile(dvrs, strength, width_min, width_max, order):
    r_cap = cap_profile_1d(dvrs[0].x, width_min, width_max, strength, order)
    R_cap = cap_profile_1d(dvrs[1].x, width_min, width_max, strength, order)
    gamma = np.zeros_like(dvrs[2].x, dtype=float)
    cap = (r_cap[:, None, None] + R_cap[None, :, None] + gamma[None, None, :])
    return cap


def asymptotic_region_populations(solver, psilist, asymptotic_r):
    """Compute channel populations for grid points with R >= asymptotic_r."""
    mask = solver.x[1] >= asymptotic_r
    mask = mask[None, :, None]  # shape broadcast to (*nx, 1)
    pops = []
    for psi in psilist:
        rho = np.abs(psi) ** 2
        region = rho * mask[..., None]
        pops.append(np.sum(region, axis=(0, 1, 2)))
    return np.asarray(pops)


def _parse_time_list(value: str) -> list[float]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    return out


def plot_wavepackets(
    solver,
    psilist: Sequence[np.ndarray],
    times_au,
    outpath: Path,
    snapshot_times_fs,
    center=None,
    normalization="state",
    vmax_percentile=99.5,
    log_scale=False,
):
    """Plot r-R nuclear density (state-resolved, integrated over gamma)."""
    times_fs = np.asarray(times_au) * au2fs
    requested = _parse_time_list(snapshot_times_fs) if isinstance(snapshot_times_fs, str) else list(snapshot_times_fs)
    if not requested:
        requested = [0.0, float(times_fs[-1])]
    chosen = []
    for target in sorted(set(requested)):
        idx = int(np.argmin(np.abs(times_fs - target)))
        if idx not in chosen:
            chosen.append(idx)
    if not chosen:
        return

    ncols = len(chosen)
    nrows = solver.nstates
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    r_grid = solver.x[0]
    R_grid = solver.x[1]
    r_mesh, R_mesh = np.meshgrid(r_grid, R_grid, indexing="ij")
    global_max = 0.0
    densities = {}
    state_max = {state: 0.0 for state in range(solver.nstates)}
    for idx in chosen:
        for state in range(solver.nstates):
            rho = np.sum(np.abs(psilist[idx][..., state]) ** 2, axis=2)
            densities[(state, idx)] = rho
            state_max[state] = max(state_max[state], float(np.max(rho)))
            global_max = max(global_max, float(np.max(rho)))

    if vmax_percentile is None:
        vmax_percentile = 99.5
    vmax_percentile = float(vmax_percentile)
    if vmax_percentile <= 0 or vmax_percentile > 100:
        vmax_percentile = 99.5

    # Robust global maxima from high quantile to avoid a single extreme point dominating
    # all subplots.
    flat = np.hstack([densities[(state, idx)].ravel() for state in range(solver.nstates) for idx in chosen])
    safe_vmax = float(np.quantile(flat, vmax_percentile / 100.0)) if flat.size else 1.0
    safe_vmax = max(safe_vmax, 1.0e-16)

    mesh = None
    for row, state in enumerate(range(solver.nstates)):
        for col, idx in enumerate(chosen):
            ax = axes[row, col]
            rho = densities[(state, idx)]

            if normalization == "state":
                denom = float(np.quantile(rho[rho > 0], vmax_percentile / 100.0)) if np.any(rho > 0) else state_max[state]
                vmax = max(denom, 1.0e-16)
            elif normalization == "global":
                vmax = safe_vmax
            else:
                vmax = global_max
            vmin = 1.0e-16 if log_scale else 0.0

            if log_scale:
                v = np.clip(rho, vmin, None)
                plotted = np.log10(v)
                plotted_vmin = float(np.min(plotted))
                plotted_vmax = float(np.max(plotted))
                if np.isclose(plotted_vmin, plotted_vmax):
                    plotted_vmax = plotted_vmin + 1.0
            else:
                plotted = rho
                plotted_vmin = vmin
                plotted_vmax = vmax

            mesh = ax.pcolormesh(
                r_mesh,
                R_mesh,
                plotted,
                shading="auto",
                cmap="magma",
                vmin=plotted_vmin,
                vmax=plotted_vmax,
            )
            ax.set_aspect("auto", adjustable="box")
            if row == 0:
                ax.set_title(f"{times_fs[idx]:.2f} fs")
            if col == 0:
                ax.set_ylabel(f"S{state} / log₁₀|ψ|²" if log_scale else f"S{state} / |ψ|²")
            if row == nrows - 1:
                ax.set_xlabel("r / bohr")
            else:
                ax.set_xticklabels([])
            if row != nrows - 1:
                ax.set_xticks([])

            if center is not None and not log_scale:
                ax.plot(center[0], center[1], "r+", markersize=8, markeredgewidth=1.2, alpha=0.9)

    label = "log10(∫|ψ|² dγ)" if log_scale else "∫|ψ|² dγ"
    # Shared x/y labels to reduce clutter.
    for ax in axes.ravel():
        ax.set_xlim(float(solver.x[0][0]), float(solver.x[0][-1]))
        ax.set_ylim(float(solver.x[1][0]), float(solver.x[1][-1]))

    fig.colorbar(mesh, ax=axes.ravel().tolist(), label=label)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _resolve_initial_packet_center_and_momentum(args, solver):
    """Choose default center/momentum for nonadiabatic scattering setups."""
    momenta = np.asarray(args.initial_momentum, dtype=float)
    center = np.array([axis[len(axis) // 2] for axis in solver.x], dtype=float)
    if args.initial_center is not None:
        center = np.asarray(args.initial_center, dtype=float)
        if center.shape != (solver.ndim,):
            raise ValueError("initial-center must provide one value per coordinate.")
    elif args.scattering:
        # Put packet near the large-R end of the grid and launch toward smaller R.
        r_axis, R_axis, gamma_axis = solver.x
        center = np.array(
            [
                float(r_axis[len(r_axis) // 2]),
                float(R_axis[-1] - 0.15 * (R_axis[-1] - R_axis[0])),
                float(gamma_axis[len(gamma_axis) // 2]),
            ],
            dtype=float,
        )
        if np.allclose(momenta, 0.0):
            momenta = np.array([0.0, -0.002, 0.0], dtype=float)
            print(
                "[scattering] No momentum supplied; using default incoming R momentum "
                "(PR = -0.002 au)."
            )
    return center, momenta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--nstates", type=int, default=2)
    parser.add_argument("--ncas", type=int, default=3)
    parser.add_argument("--nelecas", type=int, default=2)
    parser.add_argument("--n-r", type=int, default=3)
    parser.add_argument("--n-R", type=int, default=3)
    parser.add_argument("--n-gamma", type=int, default=7)
    parser.add_argument("--r-min", type=float, default=0.85)
    parser.add_argument("--r-max", type=float, default=3.0)
    parser.add_argument("--R-min", type=float, default=1.8)
    parser.add_argument("--R-max", type=float, default=5.0)
    parser.add_argument("--gamma-min", type=float, default=35.0)
    parser.add_argument("--gamma-max", type=float, default=175.0)
    parser.add_argument("--overlap", choices=("links", "full"), default="links")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument(
        "--no-propagate",
        action="store_true",
        help="Only run PES scan and skip LDR dynamics.",
    )
    parser.add_argument("--dt-fs", type=float, default=0.02)
    parser.add_argument("--nt", type=int, default=20)
    parser.add_argument("--nout", type=int, default=1)
    parser.add_argument("--scattering", action="store_true", help="Enable nonadiabatic-scattering-oriented defaults.")
    parser.add_argument("--initial-state", type=int, default=1)
    parser.add_argument("--width", type=float, default=5.0)
    parser.add_argument(
        "--initial-center",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "R2", "THETA"),
        help="Initial packet center in internal coordinates (overrides automatic default).",
    )
    parser.add_argument(
        "--initial-reference-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Prepare initial packet by projecting the chosen local basis at a center "
            "reference grid point (recommended for linked-overlap propagation). "
            "Disable for a pure nuclear Gaussian in one adiabatic component."
        ),
    )
    parser.add_argument(
        "--initial-momentum",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("PR", "PR_COM", "PTHETA"),
        help="Initial nuclear momenta (a.u.) along (r, R, gamma).",
    )
    parser.add_argument(
        "--cap-strength",
        type=float,
        default=0.0,
        help="CAP strength in Hartree. H -> H - i W with W >= 0.",
    )
    parser.add_argument(
        "--cap-width",
        type=float,
        default=0.6,
        help="Distance from each edge on r and R where CAP starts (bohr).",
    )
    parser.add_argument(
        "--cap-order",
        type=int,
        default=2,
        help="Polynomial order for the CAP profile.",
    )
    parser.add_argument(
        "--asymptotic-r",
        type=float,
        default=None,
        help=(
            "If set, report population in region R >= this value (au), useful for "
            "nonadiabatic scattering channel yields."
        ),
    )
    parser.add_argument("--plot-wavepackets", action="store_true", help="Plot selected wavepacket snapshots.")
    parser.add_argument(
        "--wavepacket-times-fs",
        type=str,
        default="",
        help=(
            "Comma-separated times (fs) for wavepacket snapshots. If empty, "
            "defaults to first/mid/final frames."
        ),
    )
    parser.add_argument(
        "--save-wavepackets",
        action="store_true",
        help="Save full wavefunction history (can be large).",
    )
    parser.add_argument(
        "--wavepacket-normalization",
        choices=("global", "state", "raw"),
        default="state",
        help="Color scaling mode per subplot row. 'state' (default) is usually clearer.",
    )
    parser.add_argument(
        "--wavepacket-vmax-percentile",
        type=float,
        default=99.5,
        help="Robust vmax percentile for color scaling; helps suppress single-point outliers.",
    )
    parser.add_argument(
        "--wavepacket-log",
        action="store_true",
        help="Plot log10-projected density to show weak tails.",
    )
    parser.add_argument("--outdir", type=Path, default=Path(__file__).with_name("h3plus_minimal_jacobi_abinitio_ldr"))
    args = parser.parse_args()

    if args.initial_state < 0 or args.initial_state >= args.nstates:
        raise ValueError("initial-state must be in [0, nstates).")

    args.outdir.mkdir(parents=True, exist_ok=True)

    mol = Triatom(
        h3plus_jacobi_ref(),
        basis=args.basis,
        nstates=args.nstates,
        charge=1,
        spin=0,
        unit="bohr",
    )

    dvrs = [
        SineDVR(args.r_min, args.r_max, args.n_r),
        SineDVR(args.R_min, args.R_max, args.n_R),
        LegendreDVR(
            np.deg2rad(args.gamma_min),
            np.deg2rad(args.gamma_max),
            args.n_gamma,
        ),
    ]
    electronic = mol.casci(
        ncas=args.ncas,
        nelecas=args.nelecas,
    )
    solver = mol.ldr(
        coordinates="jacobi",
        jacobi_atoms=(0, (1, 2)),
        dvrs=dvrs,
        electronic=electronic,
        overlap=args.overlap,
    )

    solver.scan(
        n_workers=args.n_workers,
    )

    if args.cap_strength > 0.0:
        cap_profile = build_cap_profile(
            solver.dvr.axes,
            strength=args.cap_strength,
            width_min=args.cap_width,
            width_max=args.cap_width,
            order=args.cap_order,
        )
        solver.energies = solver.energies.astype(complex) - 1j * cap_profile[..., None]
        print(
            "[CAP] "
            f"strength={args.cap_strength:g}, width={args.cap_width:g}, "
            f"order={int(args.cap_order)}, "
            f"max W={float(np.max(cap_profile)):.6g} Eh"
        )
    else:
        print("[CAP] disabled; set --cap-strength > 0 for scattering boundary absorption.")

    print("APES shape:", solver.energies.shape)
    print("APES min (Eh):", np.nanmin(solver.energies))
    print("APES max (Eh):", np.nanmax(solver.energies))

    if args.no_propagate:
        print("Scan done. Omit --no-propagate to continue with LDR propagation.")
        return

    if args.scattering and args.cap_strength <= 0.0:
        print(
            "[scattering] CAP is recommended for reactive/aborting flux. "
            "Set --cap-strength > 0 for outgoing-wave absorption."
        )

    # ---- LDR propagation block ----
    widths = None
    center, momenta = _resolve_initial_packet_center_and_momentum(args, mol)
    print(
        f"[initial Gaussian] center=(r, R, gamma)={tuple(float(v) for v in center)} a.u. "
        f"({float(np.rad2deg(center[2])):.6f} deg for gamma)"
    )
    print(f"[initial Gaussian] momenta=(pr, pR, pγ)={tuple(float(v) for v in momenta)} a.u.")
    psi0 = _state_packet(
        mol,
        state=args.initial_state,
        width=args.width,
        widths=widths,
        center=center,
        use_reference_projection=args.initial_reference_projection,
        momenta=momenta,
    )
    solver.run(
        psi0,
        dt=args.dt_fs / au2fs,
        nsteps=args.nt,
        nout=args.nout,
    )

    populations = np.asarray([
        np.diag(solver.electronic_density(psi)).real
        for psi in solver.states
    ])
    norms = solver.norm
    print("times / fs:", solver.times * au2fs)
    print("populations:")
    print(populations)
    print("norm range:", float(norms.min()), float(norms.max()))

    if args.asymptotic_r is not None:
        asym_pop = asymptotic_region_populations(
            solver, solver.states, args.asymptotic_r
        )
        print(f"asymptotic populations (R >= {args.asymptotic_r:g} au) [state resolved]:")
        print(asym_pop[-1])

    out = args.outdir / "h3plus_minimal_jacobi_abinitio_ldr_dynamics.npz"
    times_fs = solver.times * au2fs
    np.savez(
        out,
        times=solver.times,
        populations=populations,
        norms=norms,
        asymptotic_r=args.asymptotic_r if args.asymptotic_r is not None else np.nan,
        asymptotic_populations=asym_pop if args.asymptotic_r is not None else None,
        times_fs=times_fs,
        psilist=solver.states if args.save_wavepackets else None,
    )
    print(f"Saved dynamics -> {out}")

    if args.plot_wavepackets:
        snapshot_times = args.wavepacket_times_fs.strip()
        if not snapshot_times:
            t0, t_end = times_fs[0], times_fs[-1]
            third = t_end / 3.0
            snapshot_times = f"{t0:g},{third:g},{2*third:g},{t_end:g}"
        wavepacket_out = args.outdir / "h3plus_minimal_jacobi_abinitio_ldr_wavepackets.png"
        plot_wavepackets(
            solver,
            solver.states,
            solver.times,
            wavepacket_out,
            snapshot_times,
            center=center,
            normalization=args.wavepacket_normalization,
            vmax_percentile=args.wavepacket_vmax_percentile,
            log_scale=args.wavepacket_log,
        )
        print(f"Saved wavepackets -> {wavepacket_out}")


if __name__ == "__main__":
    main()
