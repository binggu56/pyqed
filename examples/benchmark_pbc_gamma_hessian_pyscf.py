#!/usr/bin/env python3
"""Benchmark periodic H2 KRHF curvatures against PySCF FFTDF."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

from pyqed.qchem.pbc import Cell


BASE_COORDS = np.asarray([[2.3, 3.0, 3.0], [3.7, 3.0, 3.0]])
LATTICE = np.eye(3) * 6.0
AXIS_LABELS = (r"$x$", r"$y$", r"$z$")


def _pyqed_energy(
    coords,
    *,
    recip_cut="auto",
    recip_precision=1.0e-8,
    conv_tol=1.0e-13,
):
    cell = Cell(
        atom=[("H", tuple(position)) for position in coords],
        a=LATTICE,
        basis="sto-3g",
        unit="bohr",
        spin=0,
        dimension=3,
        integral_options={"eri_representation": "direct"},
    ).build()
    mean_field = cell.KRHF(
        nk=1,
        eta=0.5,
        real_cut=2,
        pair_cut=2,
        recip_cut=recip_cut,
        recip_precision=recip_precision,
        one_body_nuclear_cut=2,
        jk_builder="reciprocal",
    )
    started = time.perf_counter()
    mean_field.run(max_cycle=80, conv_tol=conv_tol, conv_tol_dm=1.0e-11)
    seconds = time.perf_counter() - started
    if not mean_field.converged:
        raise RuntimeError("PyQED KRHF did not converge.")
    return float(mean_field.e_tot), float(seconds)


def _pyscf_energy(coords, *, mesh, conv_tol=1.0e-13):
    try:
        from pyscf.pbc import gto, scf
    except ImportError as error:
        raise RuntimeError("This benchmark requires the optional PySCF package.") from error

    cell = gto.Cell(
        atom=[("H", tuple(position)) for position in coords],
        a=LATTICE,
        basis="sto-3g",
        unit="B",
        spin=0,
        verbose=0,
        precision=1.0e-12,
        cart=True,
    )
    cell.build()
    cell.mesh = np.asarray([mesh, mesh, mesh], dtype=int)
    mean_field = scf.KRHF(
        cell,
        kpts=np.zeros((1, 3)),
        exxdiv="ewald",
    )
    mean_field.conv_tol = conv_tol
    mean_field.conv_tol_grad = 1.0e-11
    mean_field.max_cycle = 80
    started = time.perf_counter()
    energy = mean_field.kernel()
    seconds = time.perf_counter() - started
    if not mean_field.converged:
        raise RuntimeError("PySCF KRHF did not converge.")
    return float(energy), float(seconds)


def _diagonal_curvatures(energy_function, steps):
    cache = {}

    def cached_energy(coords):
        key = tuple(np.round(np.asarray(coords).ravel(), 12))
        if key not in cache:
            cache[key] = energy_function(coords)
        return cache[key][0]

    reference_energy = cached_energy(BASE_COORDS)
    curvatures = {}
    for step in steps:
        diagonal = np.zeros(3)
        for axis in range(3):
            plus = BASE_COORDS.copy()
            minus = BASE_COORDS.copy()
            plus[0, axis] += step
            minus[0, axis] -= step
            diagonal[axis] = (
                cached_energy(plus)
                - 2.0 * reference_energy
                + cached_energy(minus)
            ) / step**2
        curvatures[str(step)] = diagonal.tolist()
    timings = [value[1] for value in cache.values()]
    return {
        "reference_energy": reference_energy,
        "curvatures": curvatures,
        "energy_evaluations": len(cache),
        "total_scf_seconds": float(np.sum(timings)),
        "mean_scf_seconds": float(np.mean(timings)),
    }


def _richardson_extrapolation(curvatures):
    steps = sorted(float(step) for step in curvatures)
    if len(steps) < 2:
        raise ValueError("Richardson extrapolation requires at least two steps.")
    fine, coarse = steps[:2]
    ratio2 = (coarse / fine) ** 2
    fine_value = np.asarray(curvatures[str(fine)])
    coarse_value = np.asarray(curvatures[str(coarse)])
    return ((ratio2 * fine_value - coarse_value) / (ratio2 - 1.0)).tolist()


def _analytic_hessian(recip_precision):
    cell = Cell(
        atom=[("H", tuple(position)) for position in BASE_COORDS],
        a=LATTICE,
        basis="sto-3g",
        unit="bohr",
        spin=0,
        dimension=3,
        integral_options={"eri_representation": "direct"},
    ).build()
    mean_field = cell.KRHF(
        nk=1,
        eta=0.5,
        real_cut=2,
        pair_cut=2,
        recip_cut="auto",
        recip_precision=recip_precision,
        one_body_nuclear_cut=2,
        jk_builder="reciprocal",
    )
    started = time.perf_counter()
    mean_field.run(max_cycle=80, conv_tol=1.0e-13, conv_tol_dm=1.0e-11)
    scf_seconds = time.perf_counter() - started
    driver = mean_field.Hessian()
    hessian = driver.kernel(
        second_derivative_backend="analytic",
        symmetrize=False,
        enforce_acoustic_sum_rule=False,
    )
    return {
        "settings": {
            "real_cut": 2,
            "pair_cut": 2,
            "recip_cut": mean_field.recip_cut,
            "recip_precision": recip_precision,
            "recip_estimated_tail": mean_field.recip_auto_info["estimated_tail"],
            "pair_ft_screen_tol": mean_field.pair_ft_screen_tol,
            "one_body_screen_tol": mean_field.one_body_screen_tol,
        },
        "reference_energy": float(mean_field.e_tot),
        "diagonal": np.diag(hessian)[:3].tolist(),
        "scf_seconds": float(scf_seconds),
        "hessian_seconds": driver.seconds,
        "cphf_residual": driver.response.residual_norm,
    }


def _plot(results, output):
    steps = np.asarray(results["settings"]["steps"], dtype=float)
    pyqed = np.asarray(results["analytic_hessian"]["diagonal"])
    pyqed_fd = np.asarray(results["methods"]["PyQED"]["richardson_extrapolated"])
    pyscf = np.asarray(
        results["methods"]["PySCF FFTDF"]["richardson_extrapolated"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0))
    colors = {"PyQED": "#0072B2", "PySCF FFTDF": "#D55E00"}
    positions = np.arange(3)
    axes[0].scatter(
        positions - 0.12,
        pyqed,
        color=colors["PyQED"],
        marker="o",
        s=34,
        label="PyQED analytic",
        zorder=3,
    )
    axes[0].scatter(
        positions,
        pyqed_fd,
        color="#666666",
        marker="x",
        s=34,
        label="PyQED FD extrap.",
        zorder=3,
    )
    axes[0].scatter(
        positions + 0.12,
        pyscf,
        facecolor="white",
        edgecolor=colors["PySCF FFTDF"],
        marker="s",
        s=38,
        label="PySCF FD extrap.",
        zorder=3,
    )
    axes[0].axhline(0.0, color="#666666", linewidth=0.8)
    axes[0].set_xticks(positions, AXIS_LABELS)
    axes[0].set_ylabel(r"$H_{0\alpha,0\alpha}$ ($E_h/a_0^2$)")
    axes[0].set_title("Zero-step curvature")
    axes[0].legend(frameon=False, fontsize=7.5)

    for axis, label, marker in zip(range(3), AXIS_LABELS, ("o", "s", "^")):
        error = []
        for step in steps:
            key = str(float(step))
            left = results["methods"]["PyQED"]["curvatures"][key][axis]
            right = results["methods"]["PySCF FFTDF"]["curvatures"][key][axis]
            error.append(abs(left - right))
        axes[1].plot(
            steps,
            error,
            color=("#0072B2", "#009E73", "#CC79A7")[axis],
            marker=marker,
            linewidth=1.3,
            label=label,
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xticks(steps)
    axes[1].set_xticklabels(
        [rf"${step / 1.0e-3:g}\times 10^{{-3}}$" for step in steps]
    )
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[1].set_xlabel(r"Finite-difference step ($a_0$)")
    axes[1].set_ylabel(r"$|H^{\rm PyQED}-H^{\rm PySCF}|$")
    axes[1].set_title("Cross-code difference")
    axes[1].legend(frameon=False, fontsize=8, ncol=3)

    cuts = np.asarray(results["energy_convergence"]["recip_cuts"], dtype=int)
    energy_error = np.asarray(results["energy_convergence"]["absolute_error"])
    axes[2].semilogy(
        cuts,
        energy_error,
        color="#0072B2",
        marker="o",
        linewidth=1.3,
    )
    axes[2].set_xlabel("PyQED reciprocal cut")
    axes[2].set_ylabel(r"$|E-E_{\rm PySCF}|$ ($E_h$)")
    axes[2].set_title("Total-energy convergence")
    axes[2].set_xticks(cuts)
    auto_cut = int(results["analytic_hessian"]["settings"]["recip_cut"])
    axes[2].axvline(
        auto_cut,
        color="#D55E00",
        linestyle="--",
        linewidth=1.1,
        label="automatic",
    )
    axes[2].legend(frameon=False, fontsize=8)

    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(
            -0.1,
            1.06,
            label,
            transform=axis.transAxes,
            va="bottom",
            ha="left",
            fontweight="bold",
            clip_on=False,
        )
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    png = output.with_suffix(".png")
    fig.savefig(png, dpi=320)
    plt.close(fig)
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=41)
    parser.add_argument("--recip-precision", type=float, default=1.0e-8)
    parser.add_argument("--energy-cuts", type=int, nargs="+", default=range(3, 11))
    parser.add_argument("--steps", type=float, nargs="+", default=(2.0e-3, 1.0e-3))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gamma_hessian_pyscf_benchmark.pdf"),
    )
    args = parser.parse_args()

    steps = sorted({float(step) for step in args.steps}, reverse=True)
    pyqed = _diagonal_curvatures(
        lambda coords: _pyqed_energy(
            coords,
            recip_precision=args.recip_precision,
        ),
        steps,
    )
    pyscf = _diagonal_curvatures(
        lambda coords: _pyscf_energy(coords, mesh=args.mesh),
        steps,
    )
    pyqed["richardson_extrapolated"] = _richardson_extrapolation(
        pyqed["curvatures"]
    )
    pyscf["richardson_extrapolated"] = _richardson_extrapolation(
        pyscf["curvatures"]
    )
    energy_values = []
    energy_seconds = []
    for cut in args.energy_cuts:
        energy, seconds = _pyqed_energy(BASE_COORDS, recip_cut=int(cut))
        energy_values.append(energy)
        energy_seconds.append(seconds)

    differences = {}
    for step in steps:
        key = str(step)
        differences[key] = (
            np.asarray(pyqed["curvatures"][key])
            - np.asarray(pyscf["curvatures"][key])
        ).tolist()
    results = {
        "system": "periodic H2, 6 bohr cubic cell, STO-3G, Gamma KRHF",
        "settings": {
            "steps": steps,
            "pyscf_mesh": args.mesh,
            "pyqed_recip_precision": args.recip_precision,
            "pyqed_real_cut": 2,
            "pyqed_pair_cut": 2,
        },
        "methods": {"PyQED": pyqed, "PySCF FFTDF": pyscf},
        "curvature_difference": differences,
        "energy_convergence": {
            "recip_cuts": list(args.energy_cuts),
            "pyqed_energy": energy_values,
            "absolute_error": np.abs(
                np.asarray(energy_values) - pyscf["reference_energy"]
            ).tolist(),
            "seconds": energy_seconds,
        },
    }
    results["analytic_hessian"] = _analytic_hessian(args.recip_precision)
    results["settings"]["pyqed_resolved_recip_cut"] = results[
        "analytic_hessian"
    ]["settings"]["recip_cut"]

    data = args.output.with_suffix(".json")
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    png = _plot(results, args.output)
    tight_key = str(float(min(steps)))
    delta = np.asarray(differences[tight_key])
    analytic_delta = np.asarray(results["analytic_hessian"]["diagonal"]) - np.asarray(
        pyscf["richardson_extrapolated"]
    )
    print(f"PyQED energy: {pyqed['reference_energy']:.12f} Eh")
    print(f"PySCF energy: {pyscf['reference_energy']:.12f} Eh")
    print(
        "PyQED automatic reciprocal cut: "
        f"{results['settings']['pyqed_resolved_recip_cut']}"
    )
    print(f"tight-step curvature delta: {delta}")
    print(f"max |curvature delta|: {np.max(np.abs(delta)):.3e} Eh/a0^2")
    print(f"analytic vs extrapolated PySCF delta: {analytic_delta}")
    print(
        "max |analytic delta|: "
        f"{np.max(np.abs(analytic_delta)):.3e} Eh/a0^2"
    )
    print(
        "PyQED analytic Hessian: "
        f"{results['analytic_hessian']['hessian_seconds']:.3f} s"
    )
    print(f"PyQED SCF total: {pyqed['total_scf_seconds']:.3f} s")
    print(f"PySCF SCF total: {pyscf['total_scf_seconds']:.3f} s")
    print(f"wrote {args.output}")
    print(f"wrote {png}")
    print(f"wrote {data}")


if __name__ == "__main__":
    main()
