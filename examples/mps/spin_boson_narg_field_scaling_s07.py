"""Finite-field NARG scaling for the sub-Ohmic spin-boson Wilson chain.

This mirrors ``spin_boson_dmrg_field_scaling_s07.py`` but replaces the finite
two-site DMRG solver by Wilson-chain NARG.  The default solver is the
orthonormal conditional-basis NARG: the added oscillator is represented by a
sine DVR grid, the block is diagonalized conditionally at each grid point, and
the total basis has identity overlap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
for path in (ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib

matplotlib.use("Agg")

import numpy as np

from pyqed.narg import (
    SpinBosonWilsonAdiabaticNARG,
    SpinBosonWilsonNARG,
    log_discretized_spin_boson_wilson_chain,
    spin_boson_narg_step_observables,
)
from spin_boson_dmrg_field_scaling_s07 import (
    _aligned_magnetization,
    _checkpoint_name,
    _float_array,
    _load_checkpoint,
    _plot_results,
    grid_collapse,
)


def _conditional_states_arg(text):
    key = str(text).strip().lower()
    if key in {"full", "none", "d", "bond", "bond-dim", "bond_dim"}:
        return None
    try:
        value = int(key)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("n-conditional-states must be an integer or 'full'") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("n-conditional-states must be positive or 'full'")
    return value


def grid_collapse_over_y_h(
    alphas: np.ndarray,
    epsilons: np.ndarray,
    nvalues: np.ndarray,
    aligned_mags: np.ndarray,
    *,
    alpha_c_grid: np.ndarray,
    inv_nu_grid: np.ndarray,
    beta_over_nu_grid: np.ndarray,
    y_h_grid: np.ndarray,
    Lambda: float,
    min_m: float,
    max_m: float,
    width: float,
):
    """Run the existing collapse score over a grid of field dimensions."""
    best = None
    for y_h in np.asarray(y_h_grid, dtype=float):
        fit = grid_collapse(
            alphas,
            epsilons,
            nvalues,
            aligned_mags,
            alpha_c_grid=alpha_c_grid,
            inv_nu_grid=inv_nu_grid,
            beta_over_nu_grid=beta_over_nu_grid,
            y_h=float(y_h),
            Lambda=Lambda,
            min_m=min_m,
            max_m=max_m,
            width=width,
        )
        if best is None or fit.score < best.score:
            best = fit
    if best is None:
        raise RuntimeError("collapse grid is empty")
    return best


def _run_narg_result(alpha: float, epsilon: float, nmodes: int, *, params: dict, nroots: int):
    chain = log_discretized_spin_boson_wilson_chain(
        int(nmodes),
        alpha=float(alpha),
        Lambda=params["Lambda"],
        s=params["s"],
        omegac=1.0,
        epsilon=float(epsilon),
        delta=params["delta"],
    )
    if params["solver"] == "adiabatic":
        return SpinBosonWilsonAdiabaticNARG(
            chain,
            nboson=params["nboson"],
            bond_dim=params["bond_dim"],
            n_conditional_states=params["n_conditional_states"],
            dvr_qmax=params["dvr_qmax"],
            nrg_rescale=params["nrg_rescale"],
            nrg_Lambda=params["Lambda"],
            nrg_rescale_power=params["nrg_rescale_power"],
            nrg_scale=params["nrg_scale"],
            diagonalization_method=params["diagonalization_method"],
            sparse_diagonalization_threshold=params["sparse_diagonalization_threshold"],
            diagonalization_tol=params["diagonalization_tol"],
            diagonalization_maxiter=params["diagonalization_maxiter"],
            diagonalization_ncv=params["diagonalization_ncv"],
            initial_product_vectors=params.get("initial_product_vectors"),
            store_step_vectors=params.get("store_step_vectors", False),
            full_conditional_shortcut=params["full_conditional_shortcut"],
        ).run(nroots=nroots)
    return SpinBosonWilsonNARG(
        chain,
        nboson=params["nboson"],
        bond_dim=params["bond_dim"],
        basis=params["basis"],
        displacements=params["displacements"],
        parent_dim=params["parent_dim"],
        dvr_qmax=params["dvr_qmax"],
        nrg_rescale=params["nrg_rescale"],
        nrg_Lambda=params["Lambda"],
        nrg_rescale_power=params["nrg_rescale_power"],
        nrg_scale=params["nrg_scale"],
        diagonalization_method=params["diagonalization_method"],
        sparse_diagonalization_threshold=params["sparse_diagonalization_threshold"],
        diagonalization_tol=params["diagonalization_tol"],
        diagonalization_maxiter=params["diagonalization_maxiter"],
        diagonalization_ncv=params["diagonalization_ncv"],
        initial_product_vectors=params.get("initial_product_vectors"),
        store_step_vectors=params.get("store_step_vectors", False),
    ).run(nroots=nroots)


def narg_field_curve(
    alpha: float,
    epsilon: float,
    *,
    params: dict,
    nvalues: np.ndarray,
    checkpoint_path: Path | None = None,
):
    """Return NARG ``<sigma_z>`` and ground energies over increasing N."""
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

    if params.get("max_chain_extraction", True):
        start = perf_counter()
        result = _run_narg_result(
            alpha,
            epsilon,
            int(np.max(nvalues)),
            params=params,
            nroots=1,
        )
        elapsed = perf_counter() - start
        observables = spin_boson_narg_step_observables(result, nvalues=nvalues, nlevels=1)
        if params.get("recycle_eigenvectors", False):
            params["initial_product_vectors"] = [
                None if step.product_vectors is None else step.product_vectors.copy()
                for step in result.steps
            ]
        mags[:] = observables.magnetizations[:, 0]
        energies[:, 0] = observables.energies[:, 0]
        converged[:] = np.isfinite(mags) & np.isfinite(energies[:, 0])
        seconds[:] = float(elapsed)
        for nindex, nmodes in enumerate(nvalues):
            aligned = _aligned_magnetization(mags[nindex], float(epsilon))
            print(
                f"alpha={alpha:.8f} eps={epsilon:.3e} N={int(nmodes):2d} "
                f"mz={mags[nindex]: .9e} aligned={aligned:.9e} "
                f"kept={int(observables.kept[nindex])} seconds={elapsed:.3f}",
                flush=True,
            )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                checkpoint_path,
                alpha=np.asarray(float(alpha)),
                epsilon=np.asarray(float(epsilon)),
                nvalues=np.asarray(nvalues, dtype=int),
                magnetizations=np.asarray(mags, dtype=float),
                energies=np.asarray(energies, dtype=float),
                converged=np.asarray(converged, dtype=bool),
                seconds=np.asarray(seconds, dtype=float),
            )
        return mags, energies, converged, seconds

    for nindex, nmodes in enumerate(nvalues):
        if np.isfinite(mags[nindex]):
            continue

        start = perf_counter()
        result = _run_narg_result(alpha, epsilon, int(nmodes), params=params, nroots=1)
        elapsed = perf_counter() - start
        if params.get("recycle_eigenvectors", False):
            params["initial_product_vectors"] = [
                None if step.product_vectors is None else step.product_vectors.copy()
                for step in result.steps
            ]

        mz = float(np.real_if_close(np.asarray(result.magnetizations)[0]))
        mags[nindex] = mz
        energies[nindex, 0] = float(np.asarray(result.energies, dtype=float).reshape(-1)[0])
        converged[nindex] = True
        seconds[nindex] = float(elapsed)

        aligned = _aligned_magnetization(mz, float(epsilon))
        print(
            f"alpha={alpha:.8f} eps={epsilon:.3e} N={int(nmodes):2d} "
            f"mz={mz: .9e} aligned={aligned:.9e} "
            f"kept={result.effective_hamiltonian.shape[0]} seconds={elapsed:.3f}",
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmin", type=int, default=7)
    parser.add_argument("--nmax", type=int, default=13)
    parser.add_argument("--nboson", type=int, default=16)
    parser.add_argument("--bond-dim", type=int, default=64)
    parser.add_argument("--Lambda", type=float, default=1.5)
    parser.add_argument("--s", type=float, default=0.7)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--solver", choices=("adiabatic", "standard"), default="adiabatic")
    parser.add_argument("--basis", choices=("fock", "dvr", "gh-dvr", "sine-dvr", "displaced-dvr"), default="fock")
    parser.add_argument("--displacements", choices=("none", "auto"), default="none")
    parser.add_argument("--parent-dim", type=int, default=None)
    parser.add_argument("--n-conditional-states", type=_conditional_states_arg, default=None)
    parser.add_argument("--dvr-qmax", type=float, default=8.0)
    parser.add_argument("--nrg-rescale", action="store_true")
    parser.add_argument("--nrg-rescale-power", type=float, default=1.0)
    parser.add_argument("--nrg-scale", choices=("lambda", "onsite"), default="lambda")
    parser.add_argument(
        "--diagonalization-method",
        choices=("dense", "iterative", "lobpcg", "auto"),
        default="auto",
    )
    parser.add_argument("--sparse-diagonalization-threshold", type=int, default=2048)
    parser.add_argument("--diagonalization-tol", type=float, default=1.0e-10)
    parser.add_argument("--diagonalization-maxiter", type=int, default=None)
    parser.add_argument("--diagonalization-ncv", type=int, default=None)
    parser.add_argument("--recycle-eigenvectors", action="store_true")
    parser.add_argument("--no-full-conditional-shortcut", action="store_true")
    parser.add_argument(
        "--per-n-runs",
        action="store_true",
        help="Run each Wilson length independently instead of extracting prefixes from one Nmax run.",
    )
    parser.add_argument(
        "--alphas",
        type=_float_array,
        default=np.array([0.06, 0.07, 0.08, 0.09, 0.10, 0.11]),
    )
    parser.add_argument(
        "--epsilons",
        type=_float_array,
        default=np.array([1e-7, 3e-7, 1e-6, 3e-6, 1e-5]),
    )
    parser.add_argument("--y-h", type=float, default=None)
    parser.add_argument("--alpha-c-min", type=float, default=0.055)
    parser.add_argument("--alpha-c-max", type=float, default=0.115)
    parser.add_argument("--alpha-c-points", type=int, default=31)
    parser.add_argument("--invnu-min", type=float, default=0.20)
    parser.add_argument("--invnu-max", type=float, default=0.90)
    parser.add_argument("--invnu-points", type=int, default=36)
    parser.add_argument("--beta-over-nu-min", type=float, default=0.05)
    parser.add_argument("--beta-over-nu-max", type=float, default=0.25)
    parser.add_argument("--beta-over-nu-points", type=int, default=21)
    parser.add_argument("--fixed-beta-over-nu", type=float, default=None)
    parser.add_argument("--fit-min-m", type=float, default=3e-6)
    parser.add_argument("--fit-max-m", type=float, default=0.40)
    parser.add_argument("--collapse-width", type=float, default=0.65)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("spin_boson_narg_field_scaling_s07.png"),
    )
    parser.add_argument("--data-output", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.nmax < args.nmin:
        raise ValueError("nmax must be at least nmin")
    params = {
        "nboson": args.nboson,
        "bond_dim": args.bond_dim,
        "Lambda": args.Lambda,
        "s": args.s,
        "delta": args.delta,
        "solver": args.solver,
        "basis": args.basis,
        "displacements": None if args.displacements == "none" else args.displacements,
        "parent_dim": args.parent_dim,
        "n_conditional_states": args.n_conditional_states,
        "dvr_qmax": args.dvr_qmax,
        "nrg_rescale": args.nrg_rescale,
        "nrg_rescale_power": args.nrg_rescale_power,
        "nrg_scale": args.nrg_scale,
        "diagonalization_method": args.diagonalization_method,
        "sparse_diagonalization_threshold": args.sparse_diagonalization_threshold,
        "diagonalization_tol": args.diagonalization_tol,
        "diagonalization_maxiter": args.diagonalization_maxiter,
        "diagonalization_ncv": args.diagonalization_ncv,
        "initial_product_vectors": None,
        "store_step_vectors": args.recycle_eigenvectors,
        "recycle_eigenvectors": args.recycle_eigenvectors,
        "full_conditional_shortcut": not args.no_full_conditional_shortcut,
        "max_chain_extraction": not args.per_n_runs,
    }
    nvalues = np.arange(args.nmin, args.nmax + 1, dtype=int)
    alphas = np.asarray(args.alphas, dtype=float)
    epsilons = np.asarray(args.epsilons, dtype=float)
    y_h = 0.5 * (1.0 + float(args.s)) if args.y_h is None else float(args.y_h)

    print("NARG finite-field spin-boson scaling")
    print(f"s={args.s:.6f} Lambda={args.Lambda:.6f} delta={args.delta:.6f} y_h={y_h:.6f}")
    print(f"N range={args.nmin}..{args.nmax} nboson={args.nboson} D={args.bond_dim}")
    display_basis = "sine-dvr" if args.solver == "adiabatic" else args.basis
    print(
        f"solver={args.solver} n_cond={args.n_conditional_states if args.n_conditional_states is not None else 'full'} "
        f"basis={display_basis} dvr_qmax={args.dvr_qmax}"
    )
    print(f"nrg_rescale={args.nrg_rescale} nrg_scale={args.nrg_scale}")
    print(
        f"diagonalization={args.diagonalization_method} "
        f"threshold={args.sparse_diagonalization_threshold} "
        f"tol={args.diagonalization_tol:.1e} maxiter={args.diagonalization_maxiter} "
        f"ncv={args.diagonalization_ncv if args.diagonalization_ncv is not None else 'auto'}"
    )
    print(f"full_conditional_shortcut={not args.no_full_conditional_shortcut}")
    print(f"max_chain_extraction={not args.per_n_runs}")
    print(f"recycle_eigenvectors={args.recycle_eigenvectors}")
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
            curve_mags, curve_energies, curve_conv, curve_seconds = narg_field_curve(
                float(alpha),
                float(epsilon),
                params=params,
                nvalues=nvalues,
                checkpoint_path=checkpoint_path,
            )
            mags[ieps, ialpha] = curve_mags
            energies[ieps, ialpha] = curve_energies
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
    fixed_beta_over_nu = (
        0.5 * (1.0 - float(args.s))
        if args.fixed_beta_over_nu is None
        else float(args.fixed_beta_over_nu)
    )
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
    print(
        "fixed-bn uses --fixed-beta-over-nu when supplied; otherwise "
        "beta/nu=(1-s)/2. y_h=(1+s)/2 unless --y-h is supplied."
    )

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
        solver=np.asarray(args.solver),
        params=np.asarray(
            [
                args.s,
                args.Lambda,
                args.delta,
                args.nboson,
                args.bond_dim,
                args.n_conditional_states if args.n_conditional_states is not None else -1,
                y_h,
                args.fit_min_m,
                args.fit_max_m,
                1.0 if not args.per_n_runs else 0.0,
            ],
            dtype=float,
        ),
    )
    print(data_output)


if __name__ == "__main__":
    main()


