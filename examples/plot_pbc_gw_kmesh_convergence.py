#!/usr/bin/env python3
"""Plot k-mesh convergence from periodic PyQED/PySCF GW benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyqed.units import au2ev


def _gap(values):
    values = np.asarray(values, dtype=float)
    return float((np.min(values[:, 1]) - np.max(values[:, 0])) * au2ev)


def _load(path):
    payload = json.loads(Path(path).read_text())
    mesh = tuple(int(value) for value in payload["kmesh"])
    if len(set(mesh)) != 1:
        raise ValueError(f"Expected a cubic k mesh; got {mesh} in {path}.")
    return {
        "mesh": mesh,
        "n": mesh[0],
        "nkpts": int(np.prod(mesh)),
        "gamma_centered": bool(payload.get("gamma_centered", False)),
        "gamma_only": bool(payload.get("gamma_only", False)),
        "pyqed_gap_eV": _gap(payload["pyqed"]["qp_Ha"]),
        "pyscf_gap_eV": _gap(payload["pyscf"]["qp_Ha"]),
        "max_qp_error_meV": float(
            payload["comparison"]["max_abs_qp_error_meV"]
        ),
        "pyqed_seconds": float(
            sum(
                payload["pyqed"][key]
                for key in ("gdf_seconds", "scf_seconds", "gw_seconds")
            )
        ),
        "pyscf_seconds": float(
            sum(
                payload["pyscf"][key]
                for key in ("gdf_seconds", "scf_seconds", "gw_seconds")
            )
        ),
    }


def plot(rows, output):
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: row["n"])
    gamma_only = all(row["gamma_only"] for row in rows)
    n = np.asarray([row["n"] for row in rows], dtype=int)
    pyqed_gap = np.asarray([row["pyqed_gap_eV"] for row in rows])
    pyscf_gap = np.asarray([row["pyscf_gap_eV"] for row in rows])
    error = np.asarray([row["max_qp_error_meV"] for row in rows])
    pyqed_time = np.asarray([row["pyqed_seconds"] for row in rows])
    pyscf_time = np.asarray([row["pyscf_seconds"] for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(9.7, 3.35))
    axes[0].plot(
        n,
        pyscf_gap,
        color="#D55E00",
        marker="o",
        linewidth=1.3,
        label="PySCF",
    )
    axes[0].plot(
        n,
        pyqed_gap,
        color="#0072B2",
        marker="x",
        markersize=6.0,
        markeredgewidth=1.3,
        linewidth=0,
        label="PyQED",
    )
    axes[0].set_ylabel(
        r"$E_g^{\mathrm{QP}}(\Gamma)$ (eV)"
        if gamma_only
        else r"Sampled $E_g^{\mathrm{QP}}$ (eV)"
    )
    axes[0].legend(frameon=False, fontsize=8.5)

    axes[1].plot(
        n,
        error,
        color="#009E73",
        marker="s",
        linewidth=1.3,
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$\max|\Delta E^{\mathrm{QP}}|$ (meV)")

    axes[2].plot(
        n,
        pyqed_time,
        color="#0072B2",
        marker="o",
        linewidth=1.3,
        label="PyQED",
    )
    axes[2].plot(
        n,
        pyscf_time,
        color="#D55E00",
        marker="s",
        linewidth=1.3,
        linestyle="--",
        label="PySCF",
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Total time (s)")
    axes[2].legend(frameon=False, fontsize=8.5)

    for panel, axis in zip(("a", "b", "c"), axes):
        axis.set_xlabel(r"Cubic mesh size $n$ in $n^3$")
        axis.set_xticks(n)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            -0.08,
            1.04,
            panel,
            transform=axis.transAxes,
            va="bottom",
            ha="left",
            fontweight="bold",
            clip_on=False,
        )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.92, wspace=0.38)

    output = Path(output)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    fig.savefig(png, dpi=360, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gw_kmesh_convergence"),
    )
    args = parser.parse_args()
    rows = [_load(path) for path in args.inputs]
    png, pdf = plot(rows, args.output)
    print(json.dumps({"rows": rows, "figure_png": str(png), "figure_pdf": str(pdf)}, indent=2))


if __name__ == "__main__":
    main()
