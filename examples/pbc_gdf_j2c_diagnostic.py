#!/usr/bin/env python3
"""Compare raw native and PySCF periodic auxiliary Coulomb metrics by q."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from pbc_gdf_validation import (
    CASES,
    _native_space,
    _parse_kmesh,
    _pyscf_reference,
)
from pyqed.pbc.gw.integrals import (
    _gdf_aux_coord_type,
    _gdf_auxiliary_basis,
    _gdf_auxbasis_name,
    _periodic_gdf_aux_metric,
)


def _pyscf_j2c(mf, qpts):
    from pyscf.pbc.df.gdf_builder import _CCGDFBuilder
    from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder

    with_df = mf.with_df
    if with_df._prefer_ccdf or mf.cell.omega > 0:
        builder = _CCGDFBuilder(mf.cell, with_df.auxcell, with_df.kpts)
        builder.eta = with_df.eta
    else:
        builder = _RSGDFBuilder(mf.cell, with_df.auxcell, with_df.kpts)
    builder.mesh = with_df.mesh
    builder.linear_dep_threshold = with_df.linear_dep_threshold
    builder.build()
    return [np.asarray(metric) for metric in builder.get_2c2e(np.asarray(qpts))]


def _metric_row(q_index, qvec, native, reference, threshold):
    native = 0.5 * (native + native.conj().T)
    reference = 0.5 * (reference + reference.conj().T)
    if native.shape != reference.shape:
        raise ValueError(
            f"J2c shape mismatch at q={q_index}: {native.shape} != {reference.shape}."
        )
    delta = native - reference
    native_eig = np.linalg.eigvalsh(native)
    reference_eig = np.linalg.eigvalsh(reference)
    scale = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return {
        "q_index": int(q_index),
        "qvec": np.asarray(qvec, dtype=float).tolist(),
        "relative_error": float(np.linalg.norm(delta) / scale),
        "max_abs_error": float(np.max(np.abs(delta))),
        "native_min_eigenvalue": float(native_eig[0]),
        "pyscf_min_eigenvalue": float(reference_eig[0]),
        "native_rank": int(np.count_nonzero(native_eig > threshold)),
        "pyscf_rank": int(np.count_nonzero(reference_eig > threshold)),
        "native_negative_count": int(np.count_nonzero(native_eig < -threshold)),
        "pyscf_negative_count": int(np.count_nonzero(reference_eig < -threshold)),
        "native_eigenvalues": native_eig.tolist(),
        "pyscf_eigenvalues": reference_eig.tolist(),
    }


def diagnose(args):
    case = replace(CASES[args.case], kmesh=args.kmesh)
    cell, kpts, pyscf_mf, *_ = _pyscf_reference(
        case,
        args.reference_precision,
        aux_min_exponent=args.aux_min_exponent,
        metric_tol=args.metric_tol,
    )
    seed_space = _native_space(
        case,
        cell,
        kpts,
        pyscf_mf,
        args.precision,
        aux_min_exponent=args.aux_min_exponent,
        metric_tol=args.metric_tol,
    )
    reference_metrics = _pyscf_j2c(pyscf_mf, seed_space.qpts)

    studies = []
    for mode in args.rs_aux_partition:
        space = _native_space(
            case,
            cell,
            kpts,
            pyscf_mf,
            args.precision,
            reciprocal_kernel=args.reciprocal_kernel,
            rs_aux_partition=mode,
            aux_min_exponent=args.aux_min_exponent,
            metric_tol=args.metric_tol,
        )
        ref = space.reference
        auxbasis = _gdf_auxbasis_name(ref)
        aux = _gdf_auxiliary_basis(
            space,
            auxbasis,
            _gdf_aux_coord_type(ref),
        )
        rows = []
        for q_index, reference in enumerate(reference_metrics):
            native = _periodic_gdf_aux_metric(
                space,
                q_index,
                aux,
                args.g2_tol,
            )
            rows.append(
                _metric_row(
                    q_index,
                    space.qpts[q_index],
                    native,
                    reference,
                    args.metric_tol,
                )
            )
        studies.append(
            {
                "rs_aux_partition": mode,
                "q_blocks": rows,
                "max_relative_error": max(row["relative_error"] for row in rows),
                "min_native_eigenvalue": min(
                    row["native_min_eigenvalue"] for row in rows
                ),
            }
        )
    return {
        "case": case.name,
        "kmesh": list(case.kmesh),
        "precision": float(args.precision),
        "reference_precision": float(args.reference_precision),
        "metric_tol": float(args.metric_tol),
        "aux_min_exponent": float(args.aux_min_exponent),
        "reciprocal_kernel": args.reciprocal_kernel,
        "studies": studies,
    }


def _plot(payload, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    colors = {"smooth": "#0072B2", "off": "#D55E00", "all": "#009E73"}
    for study in payload["studies"]:
        mode = study["rs_aux_partition"]
        rows = study["q_blocks"]
        q = np.asarray([row["q_index"] for row in rows])
        error = np.asarray([row["relative_error"] for row in rows])
        minimum = np.asarray([row["native_min_eigenvalue"] for row in rows])
        axes[0].plot(
            q,
            error,
            marker="o",
            linewidth=1.2,
            color=colors[mode],
            label=mode,
        )
        axes[1].plot(
            q,
            minimum,
            marker="o",
            linewidth=1.2,
            color=colors[mode],
            label=mode,
        )
    reference_rows = payload["studies"][0]["q_blocks"]
    axes[1].plot(
        [row["q_index"] for row in reference_rows],
        [row["pyscf_min_eigenvalue"] for row in reference_rows],
        color="0.25",
        linestyle="--",
        linewidth=1.0,
        marker="x",
        label="PySCF",
    )
    axes[0].set_yscale("log")
    axes[1].set_yscale("symlog", linthresh=1.0e-12)
    axes[0].set_ylabel(r"$\|J_{2c}-J_{2c}^{\mathrm{PySCF}}\|_F/\|J_{2c}^{\mathrm{PySCF}}\|_F$")
    axes[1].set_ylabel(r"$\lambda_{\min}(J_{2c})$ (Ha)")
    for panel, axis in zip(("a", "b"), axes):
        axis.set_xlabel(r"Momentum-transfer index $q$")
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            0.02,
            0.97,
            panel,
            transform=axis.transAxes,
            va="top",
            fontweight="bold",
        )
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.17, top=0.97, wspace=0.35)

    output = Path(output)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    fig.savefig(png, dpi=360, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="diamond")
    parser.add_argument("--kmesh", type=_parse_kmesh, default=(2, 2, 2))
    parser.add_argument("--precision", type=float, default=1.0e-12)
    parser.add_argument("--reference-precision", type=float, default=1.0e-12)
    parser.add_argument("--metric-tol", type=float)
    parser.add_argument("--aux-min-exponent", type=float, default=0.0)
    parser.add_argument("--g2-tol", type=float, default=1.0e-14)
    parser.add_argument(
        "--reciprocal-kernel",
        choices=("full", "range_separated"),
        default="range_separated",
    )
    parser.add_argument(
        "--rs-aux-partition",
        action="append",
        choices=("smooth", "off", "all"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_diamond_222_j2c_diagnostic.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("/private/tmp/pbc_diamond_222_j2c_diagnostic"),
    )
    args = parser.parse_args()
    if args.metric_tol is None:
        args.metric_tol = max(1.0e-14, 0.1 * float(args.precision))
    if args.rs_aux_partition is None:
        args.rs_aux_partition = ["smooth", "off", "all"]

    payload = diagnose(args)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    png, pdf = _plot(payload, args.figure)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "figure_png": str(png),
                "figure_pdf": str(pdf),
                "studies": [
                    {
                        "rs_aux_partition": study["rs_aux_partition"],
                        "max_relative_error": study["max_relative_error"],
                        "min_native_eigenvalue": study["min_native_eigenvalue"],
                    }
                    for study in payload["studies"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
