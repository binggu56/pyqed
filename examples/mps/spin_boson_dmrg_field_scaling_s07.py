"""Finite-field DMRG scaling for the sub-Ohmic spin-boson Wilson chain.

This script measures the aligned impurity order parameter

    m = -sign(epsilon) <sigma_z>

on finite Wilson chains and performs a small two-variable scaling-collapse
search,

    m_N(alpha, eps) = Lambda^(-N beta/nu)
        F((alpha-alpha_c) Lambda^(N/nu), eps Lambda^(N y_h)).

For the long-range Ising/spin-boson critical point the default field scaling
dimension is y_h=(1+s)/2.  The resulting exponents are meant as a local DMRG
diagnostic; convergence still needs the usual checks in N, boson cutoff, bond
dimension, field window, and Lambda.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pyqed.mps.mps import MPS
from pyqed.narg import (
    log_discretized_spin_boson_wilson_chain,
    spin_boson_product_mps,
    spin_boson_wilson_dmrg,
)


@dataclass
class CollapseFit:
    alpha_c: float
    inv_nu: float
    beta_over_nu: float
    y_h: float
    score: float
    npoints: int
    effective_pairs: float

    @property
    def nu(self) -> float:
        return 1.0 / self.inv_nu

    @property
    def beta(self) -> float:
        return self.beta_over_nu / self.inv_nu


def _float_array(text: str) -> np.ndarray:
    values = [float(item) for item in text.replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("array must contain at least one value")
    return np.asarray(values, dtype=float)


def _extend_mps_with_local_ground(state: MPS | None, chain, *, params: dict):
    if state is None:
        return None
    if len(state) != chain.nmodes:
        return None
    if params["displacements"] not in (None, False, "none", "zero", "false"):
        return None
    product = spin_boson_product_mps(
        chain,
        params["nboson"],
        basis=params["basis"],
        displacements=params["displacements"],
        parent_dim=params["parent_dim"],
        dvr_qmax=params["dvr_qmax"],
    )
    local = product.factors[-1].copy()
    return MPS([factor.copy() for factor in state.factors] + [local], labels=["lv", "p", "rv"])


def _checkpoint_name(alpha: float, epsilon: float) -> str:
    return f"alpha_{alpha:.8f}_eps_{epsilon:.3e}.npz".replace("+", "")


def _load_checkpoint(path: Path, nvalues: np.ndarray):
    if not path.exists():
        return None
    try:
        data = np.load(path)
        saved_n = np.asarray(data["nvalues"], dtype=int)
        saved_mags = np.asarray(data["magnetizations"], dtype=float)
        saved_energies = np.asarray(data["energies"], dtype=float)
        saved_converged = np.asarray(data["converged"], dtype=bool)
        saved_seconds = np.asarray(data["seconds"], dtype=float)
    except Exception:
        return None

    mags = np.full(len(nvalues), np.nan, dtype=float)
    energies = np.full((len(nvalues), 1), np.nan, dtype=float)
    converged = np.zeros(len(nvalues), dtype=bool)
    seconds = np.full(len(nvalues), np.nan, dtype=float)
    n_to_index = {int(nmode): index for index, nmode in enumerate(nvalues)}
    for saved_index, nmode in enumerate(saved_n):
        index = n_to_index.get(int(nmode))
        if index is None:
            continue
        mags[index] = float(saved_mags[saved_index])
        energy_row = np.asarray(saved_energies[saved_index], dtype=float).reshape(-1)
        energies[index, 0] = float(energy_row[0])
        converged[index] = bool(saved_converged[saved_index])
        seconds[index] = float(saved_seconds[saved_index])
    return mags, energies, converged, seconds


def dmrg_field_curve(
    alpha: float,
    epsilon: float,
    *,
    params: dict,
    nvalues: np.ndarray,
    checkpoint_path: Path | None = None,
):
    """Return m_z and energies for fixed alpha/epsilon over increasing N."""
    mags = np.full(len(nvalues), np.nan, dtype=float)
    energies = np.full((len(nvalues), 1), np.nan, dtype=float)
    converged = np.zeros(len(nvalues), dtype=bool)
    seconds = np.full(len(nvalues), np.nan, dtype=float)
    if checkpoint_path is not None:
        loaded = _load_checkpoint(checkpoint_path, nvalues)
        if loaded is not None:
            mags, energies, converged, seconds = loaded
            loaded_n = nvalues[np.isfinite(mags)]
            if len(loaded_n):
                print(
                    f"loaded alpha={alpha:.8f} eps={epsilon:.3e} "
                    f"N={int(loaded_n[0])}..{int(loaded_n[-1])}",
                    flush=True,
                )
            if np.all(np.isfinite(mags)):
                return mags, energies, converged, seconds

    previous_state = None
    for nindex, nmodes in enumerate(nvalues):
        if np.isfinite(mags[nindex]):
            continue
        chain = log_discretized_spin_boson_wilson_chain(
            int(nmodes),
            alpha=float(alpha),
            Lambda=params["Lambda"],
            s=params["s"],
            omegac=1.0,
            epsilon=float(epsilon),
            delta=params["delta"],
        )
        init_guess = _extend_mps_with_local_ground(previous_state, chain, params=params)
        start = perf_counter()
        result = spin_boson_wilson_dmrg(
            chain,
            nboson=params["nboson"],
            bond_dim=params["bond_dim"],
            nsweeps=params["nsweeps"],
            nstates=1,
            basis=params["basis"],
            displacements=params["displacements"],
            parent_dim=params["parent_dim"],
            dvr_qmax=params["dvr_qmax"],
            init_guess=init_guess,
            sweep_tol=params["sweep_tol"],
            davidson_tol=params["davidson_tol"],
            davidson_max_iter=params["davidson_max_iter"],
            noise=params["noise"],
            noise_decay=params["noise_decay"],
            not_conv_err=False,
        )
        previous_state = result.dmrg.ground_state
        elapsed = perf_counter() - start
        mags[nindex] = float(result.magnetization)
        energies[nindex, 0] = float(np.asarray(result.energies, dtype=float).reshape(-1)[0])
        converged[nindex] = bool(result.dmrg.converged)
        seconds[nindex] = float(elapsed)

        aligned = _aligned_magnetization(float(result.magnetization), float(epsilon))
        print(
            f"alpha={alpha:.8f} eps={epsilon:.3e} N={int(nmodes):2d} "
            f"mz={float(result.magnetization): .9e} aligned={aligned:.9e} "
            f"conv={int(converged[nindex])} seconds={elapsed:.3f}",
            flush=True,
        )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            completed = np.isfinite(mags)
            np.savez(
                checkpoint_path,
                alpha=np.asarray(float(alpha)),
                epsilon=np.asarray(float(epsilon)),
                nvalues=np.asarray(nvalues[completed], dtype=int),
                magnetizations=np.asarray(mags[completed], dtype=float),
                energies=np.asarray(energies[completed], dtype=float),
                converged=np.asarray(converged[completed], dtype=bool),
                seconds=np.asarray(seconds[completed], dtype=float),
            )
    return mags, energies, converged, seconds


def _aligned_magnetization(mz: float, epsilon: float) -> float:
    if abs(float(epsilon)) <= 0.0:
        return abs(float(mz))
    return float(-np.sign(epsilon) * mz)


def _collapse_points(
    alphas: np.ndarray,
    epsilons: np.ndarray,
    nvalues: np.ndarray,
    aligned_mags: np.ndarray,
    *,
    alpha_c: float,
    inv_nu: float,
    beta_over_nu: float,
    y_h: float,
    Lambda: float,
    min_m: float,
    max_m: float,
):
    coords = []
    values = []
    ns = []
    for ieps, epsilon in enumerate(epsilons):
        if epsilon <= 0.0:
            continue
        for ialpha, alpha in enumerate(alphas):
            for inum, nmode in enumerate(nvalues):
                mag = float(aligned_mags[ieps, ialpha, inum])
                if not np.isfinite(mag) or mag < min_m or mag > max_m:
                    continue
                scale = float(Lambda) ** float(nmode)
                x_alpha = (float(alpha) - float(alpha_c)) * scale ** float(inv_nu)
                x_field = float(epsilon) * scale ** float(y_h)
                if x_field <= 0.0:
                    continue
                y_value = mag * scale ** float(beta_over_nu)
                if y_value <= 0.0 or not np.isfinite(y_value):
                    continue
                coords.append((x_alpha, np.log(x_field)))
                values.append(np.log(y_value))
                ns.append(int(nmode))
    return np.asarray(coords, dtype=float), np.asarray(values, dtype=float), np.asarray(ns, dtype=int)


def _collapse_score(
    alphas: np.ndarray,
    epsilons: np.ndarray,
    nvalues: np.ndarray,
    aligned_mags: np.ndarray,
    *,
    alpha_c: float,
    inv_nu: float,
    beta_over_nu: float,
    y_h: float,
    Lambda: float,
    min_m: float,
    max_m: float,
    width: float,
) -> tuple[float, int, float]:
    coords, values, ns = _collapse_points(
        alphas,
        epsilons,
        nvalues,
        aligned_mags,
        alpha_c=alpha_c,
        inv_nu=inv_nu,
        beta_over_nu=beta_over_nu,
        y_h=y_h,
        Lambda=Lambda,
        min_m=min_m,
        max_m=max_m,
    )
    npoints = int(len(values))
    if npoints < 8 or len(np.unique(ns)) < 2:
        return np.inf, npoints, 0.0

    center = np.nanmean(coords, axis=0)
    spread = np.nanstd(coords, axis=0)
    spread[spread < 1e-12] = 1.0
    coords = (coords - center) / spread
    delta = coords[:, None, :] - coords[None, :, :]
    distance2 = np.sum(delta * delta, axis=2)
    pair_mask = np.triu(ns[:, None] != ns[None, :], k=1)
    weights = np.exp(-0.5 * distance2 / max(float(width), 1e-6) ** 2) * pair_mask
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-14:
        return np.inf, npoints, 0.0

    value_delta = values[:, None] - values[None, :]
    score = float(np.sum(weights * value_delta * value_delta) / weight_sum)
    effective_pairs = weight_sum * weight_sum / max(float(np.sum(weights * weights)), 1e-14)
    if effective_pairs < max(10.0, 0.5 * npoints):
        score *= max(10.0, 0.5 * npoints) / max(effective_pairs, 1e-12)
    return score, npoints, float(effective_pairs)


def grid_collapse(
    alphas: np.ndarray,
    epsilons: np.ndarray,
    nvalues: np.ndarray,
    aligned_mags: np.ndarray,
    *,
    alpha_c_grid: np.ndarray,
    inv_nu_grid: np.ndarray,
    beta_over_nu_grid: np.ndarray,
    y_h: float,
    Lambda: float,
    min_m: float,
    max_m: float,
    width: float,
) -> CollapseFit:
    best: CollapseFit | None = None
    for alpha_c in alpha_c_grid:
        for inv_nu in inv_nu_grid:
            for beta_over_nu in beta_over_nu_grid:
                score, npoints, effective_pairs = _collapse_score(
                    alphas,
                    epsilons,
                    nvalues,
                    aligned_mags,
                    alpha_c=float(alpha_c),
                    inv_nu=float(inv_nu),
                    beta_over_nu=float(beta_over_nu),
                    y_h=float(y_h),
                    Lambda=float(Lambda),
                    min_m=float(min_m),
                    max_m=float(max_m),
                    width=float(width),
                )
                if best is None or score < best.score:
                    best = CollapseFit(
                        alpha_c=float(alpha_c),
                        inv_nu=float(inv_nu),
                        beta_over_nu=float(beta_over_nu),
                        y_h=float(y_h),
                        score=float(score),
                        npoints=int(npoints),
                        effective_pairs=float(effective_pairs),
                    )
    if best is None:
        raise RuntimeError("collapse grid is empty")
    return best


def _plot_results(
    output: Path,
    *,
    alphas: np.ndarray,
    epsilons: np.ndarray,
    nvalues: np.ndarray,
    aligned_mags: np.ndarray,
    free_fit: CollapseFit,
    fixed_fit: CollapseFit | None,
    Lambda: float,
):
    fit = fixed_fit if fixed_fit is not None else free_fit
    mid_eps_index = len(epsilons) // 2
    alpha_index = int(np.argmin(np.abs(alphas - fit.alpha_c)))

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    for inum, nmode in enumerate(nvalues):
        ax.plot(alphas, aligned_mags[mid_eps_index, :, inum], marker="o", label=f"N={int(nmode)}")
    ax.axvline(fit.alpha_c, color="k", linewidth=1.0, alpha=0.45)
    ax.set_title(f"order parameter, epsilon={epsilons[mid_eps_index]:.1e}")
    ax.set_xlabel("alpha")
    ax.set_ylabel(r"$-\mathrm{sign}(\epsilon)\langle\sigma_z\rangle$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for inum, nmode in enumerate(nvalues):
        ax.loglog(epsilons, aligned_mags[:, alpha_index, inum], marker="o", label=f"N={int(nmode)}")
    ax.set_title(f"field response, alpha={alphas[alpha_index]:.4f}")
    ax.set_xlabel("epsilon")
    ax.set_ylabel(r"$-\mathrm{sign}(\epsilon)\langle\sigma_z\rangle$")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1, 0]
    for inum, nmode in enumerate(nvalues):
        scale = float(Lambda) ** float(nmode)
        x = (alphas[None, :] - fit.alpha_c) * scale ** fit.inv_nu
        y = aligned_mags[:, :, inum] * scale ** fit.beta_over_nu
        for ieps, epsilon in enumerate(epsilons):
            ax.plot(x[0], y[ieps], marker="o", linestyle="", label=f"eps={epsilon:.0e}" if inum == 0 else None)
    ax.set_title(
        rf"collapse slice: $\alpha_c$={fit.alpha_c:.4f}, "
        rf"$\nu$={fit.nu:.3f}, $\beta$={fit.beta:.3f}"
    )
    ax.set_xlabel(r"$(\alpha-\alpha_c)\Lambda^{N/\nu}$")
    ax.set_ylabel(r"$m\Lambda^{N\beta/\nu}$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for inum, nmode in enumerate(nvalues):
        scale = float(Lambda) ** float(nmode)
        x = epsilons * scale ** fit.y_h
        y = aligned_mags[:, alpha_index, inum] * scale ** fit.beta_over_nu
        ax.loglog(x, y, marker="o", label=f"N={int(nmode)}")
    ax.set_title(r"critical-field scaling at nearest alpha")
    ax.set_xlabel(r"$\epsilon\Lambda^{Ny_h}$")
    ax.set_ylabel(r"$m\Lambda^{N\beta/\nu}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmin", type=int, default=7)
    parser.add_argument("--nmax", type=int, default=10)
    parser.add_argument("--nboson", type=int, default=5)
    parser.add_argument("--bond-dim", type=int, default=24)
    parser.add_argument("--nsweeps", type=int, default=4)
    parser.add_argument("--Lambda", type=float, default=1.5)
    parser.add_argument("--s", type=float, default=0.7)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument(
        "--basis",
        choices=("fock", "dvr", "gh-dvr", "sine-dvr", "displaced-dvr"),
        default="fock",
    )
    parser.add_argument(
        "--displacements",
        choices=("none", "auto"),
        default="none",
    )
    parser.add_argument("--parent-dim", type=int, default=None)
    parser.add_argument("--dvr-qmax", type=float, default=14.0)
    parser.add_argument("--sweep-tol", type=float, default=1e-7)
    parser.add_argument("--davidson-tol", type=float, default=1e-7)
    parser.add_argument("--davidson-max-iter", type=int, default=60)
    parser.add_argument("--noise", type=float, default=1e-6)
    parser.add_argument("--noise-decay", type=float, default=0.25)
    parser.add_argument(
        "--alphas",
        type=_float_array,
        default=np.array([0.46, 0.48, 0.49, 0.50, 0.52]),
    )
    parser.add_argument(
        "--epsilons",
        type=_float_array,
        default=np.array([1e-7, 3e-7, 1e-6, 3e-6, 1e-5]),
    )
    parser.add_argument("--y-h", type=float, default=None)
    parser.add_argument("--alpha-c-min", type=float, default=0.455)
    parser.add_argument("--alpha-c-max", type=float, default=0.525)
    parser.add_argument("--alpha-c-points", type=int, default=29)
    parser.add_argument("--invnu-min", type=float, default=0.20)
    parser.add_argument("--invnu-max", type=float, default=0.70)
    parser.add_argument("--invnu-points", type=int, default=26)
    parser.add_argument("--beta-over-nu-min", type=float, default=0.05)
    parser.add_argument("--beta-over-nu-max", type=float, default=0.25)
    parser.add_argument("--beta-over-nu-points", type=int, default=21)
    parser.add_argument("--fit-min-m", type=float, default=1e-5)
    parser.add_argument("--fit-max-m", type=float, default=0.85)
    parser.add_argument("--collapse-width", type=float, default=0.65)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("spin_boson_dmrg_field_scaling_s07.png"),
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.nmax < args.nmin:
        raise ValueError("nmax must be at least nmin")

    params = {
        "nboson": args.nboson,
        "bond_dim": args.bond_dim,
        "nsweeps": args.nsweeps,
        "Lambda": args.Lambda,
        "s": args.s,
        "delta": args.delta,
        "basis": args.basis,
        "displacements": None if args.displacements == "none" else args.displacements,
        "parent_dim": args.parent_dim,
        "dvr_qmax": args.dvr_qmax,
        "sweep_tol": args.sweep_tol,
        "davidson_tol": args.davidson_tol,
        "davidson_max_iter": args.davidson_max_iter,
        "noise": args.noise,
        "noise_decay": args.noise_decay,
    }
    nvalues = np.arange(args.nmin, args.nmax + 1, dtype=int)
    alphas = np.asarray(args.alphas, dtype=float)
    epsilons = np.asarray(args.epsilons, dtype=float)
    y_h = 0.5 * (1.0 + float(args.s)) if args.y_h is None else float(args.y_h)

    print("DMRG finite-field spin-boson scaling")
    print(f"s={args.s:.6f} Lambda={args.Lambda:.6f} delta={args.delta:.6f} y_h={y_h:.6f}")
    print(f"N range={args.nmin}..{args.nmax} nboson={args.nboson} D={args.bond_dim}")
    print(f"basis={args.basis} nsweeps={args.nsweeps} displacements={args.displacements}")
    print("alphas  :", " ".join(f"{alpha:.6f}" for alpha in alphas))
    print("epsilons:", " ".join(f"{epsilon:.3e}" for epsilon in epsilons))
    print()

    mags = np.empty((len(epsilons), len(alphas), len(nvalues)), dtype=float)
    energies = np.empty((len(epsilons), len(alphas), len(nvalues), 1), dtype=float)
    converged = np.empty((len(epsilons), len(alphas), len(nvalues)), dtype=bool)
    seconds = np.empty((len(epsilons), len(alphas), len(nvalues)), dtype=float)

    for ieps, epsilon in enumerate(epsilons):
        for ialpha, alpha in enumerate(alphas):
            checkpoint_path = None
            if args.checkpoint_dir is not None:
                checkpoint_path = args.checkpoint_dir / _checkpoint_name(float(alpha), float(epsilon))
            curve_mags, curve_energies, curve_conv, curve_seconds = dmrg_field_curve(
                float(alpha),
                float(epsilon),
                params=params,
                nvalues=nvalues,
                checkpoint_path=checkpoint_path,
            )
            mags[ieps, ialpha] = curve_mags
            energies[ieps, ialpha, :, 0] = curve_energies[:, 0]
            converged[ieps, ialpha] = curve_conv
            seconds[ieps, ialpha] = curve_seconds

    aligned_mags = np.empty_like(mags)
    for ieps, epsilon in enumerate(epsilons):
        aligned_mags[ieps] = np.vectorize(_aligned_magnetization)(mags[ieps], float(epsilon))

    alpha_c_grid = np.linspace(args.alpha_c_min, args.alpha_c_max, args.alpha_c_points)
    inv_nu_grid = np.linspace(args.invnu_min, args.invnu_max, args.invnu_points)
    beta_over_nu_grid = np.linspace(
        args.beta_over_nu_min,
        args.beta_over_nu_max,
        args.beta_over_nu_points,
    )
    free_fit = grid_collapse(
        alphas,
        epsilons,
        nvalues,
        aligned_mags,
        alpha_c_grid=alpha_c_grid,
        inv_nu_grid=inv_nu_grid,
        beta_over_nu_grid=beta_over_nu_grid,
        y_h=y_h,
        Lambda=args.Lambda,
        min_m=args.fit_min_m,
        max_m=args.fit_max_m,
        width=args.collapse_width,
    )
    fixed_beta_over_nu = 0.5 * (1.0 - float(args.s))
    fixed_fit = grid_collapse(
        alphas,
        epsilons,
        nvalues,
        aligned_mags,
        alpha_c_grid=alpha_c_grid,
        inv_nu_grid=inv_nu_grid,
        beta_over_nu_grid=np.asarray([fixed_beta_over_nu], dtype=float),
        y_h=y_h,
        Lambda=args.Lambda,
        min_m=args.fit_min_m,
        max_m=args.fit_max_m,
        width=args.collapse_width,
    )

    print()
    print("collapse fits")
    print("type        alpha_c    1/nu      nu        beta/nu   beta      score       points  eff_pairs")
    for label, fit in (("free", free_fit), ("fixed-bn", fixed_fit)):
        print(
            f"{label:10s}  {fit.alpha_c:8.5f}  {fit.inv_nu:8.5f}  "
            f"{fit.nu:8.5f}  {fit.beta_over_nu:8.5f}  {fit.beta:8.5f}  "
            f"{fit.score:10.4e}  {fit.npoints:6d}  {fit.effective_pairs:9.2f}"
        )
    print()
    print("fixed-bn uses beta/nu=(1-s)/2; y_h=(1+s)/2 unless --y-h is supplied.")

    _plot_results(
        args.output,
        alphas=alphas,
        epsilons=epsilons,
        nvalues=nvalues,
        aligned_mags=aligned_mags,
        free_fit=free_fit,
        fixed_fit=fixed_fit,
        Lambda=args.Lambda,
    )
    print()
    print(args.output)

    data_output = args.data_output
    if data_output is None:
        data_output = args.output.with_suffix(".npz")
    data_output.parent.mkdir(parents=True, exist_ok=True)
    fit_table = np.asarray(
        [
            [
                free_fit.alpha_c,
                free_fit.inv_nu,
                free_fit.nu,
                free_fit.beta_over_nu,
                free_fit.beta,
                free_fit.y_h,
                free_fit.score,
                free_fit.npoints,
                free_fit.effective_pairs,
            ],
            [
                fixed_fit.alpha_c,
                fixed_fit.inv_nu,
                fixed_fit.nu,
                fixed_fit.beta_over_nu,
                fixed_fit.beta,
                fixed_fit.y_h,
                fixed_fit.score,
                fixed_fit.npoints,
                fixed_fit.effective_pairs,
            ],
        ],
        dtype=float,
    )
    np.savez(
        data_output,
        alphas=alphas,
        epsilons=epsilons,
        nvalues=nvalues,
        magnetizations=mags,
        aligned_magnetizations=aligned_mags,
        energies=energies,
        converged=converged,
        seconds=seconds,
        fit_table=fit_table,
        fit_labels=np.asarray(["free", "fixed-beta-over-nu"]),
        params=np.asarray(
            [
                args.s,
                args.Lambda,
                args.delta,
                args.nboson,
                args.bond_dim,
                args.nsweeps,
                y_h,
                args.fit_min_m,
                args.fit_max_m,
            ],
            dtype=float,
        ),
    )
    print(data_output)


if __name__ == "__main__":
    main()

