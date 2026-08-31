#!/usr/bin/env python3
"""Benchmark native PyQED periodic G0W0 against PySCF KGWAC.

The primary comparison is end to end: each package builds and converges its
own GDF-KRHF reference before evaluating linearized analytic-continuation GW.
An optional aligned diagnostic runs the PyQED GW solver from the PySCF mean
field and PySCF GDF factors to isolate solver/convention differences.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from pbc_gdf_validation import (
    CASES,
    _native_space,
    _parse_kmesh,
    _pyscf_reference,
    _seed_native_reference,
)
from pyqed.pbc.gw import (
    GDF,
    PYSCF_GDF,
    KPointTransitionSpace,
    diagonal_g0w0,
)
from pyqed.units import au2ev, au2mev


def _native_krhf(
    case,
    *,
    precision,
    aux_min_exponent,
    metric_tol,
    workers,
):
    cell, kpts, _ = _seed_native_reference(case)
    mf = cell.KRHF(
        kpts=kpts,
        eta=0.5,
        real_cut="auto",
        pair_cut="auto",
        recip_cut=5,
        jk_builder="gdf",
    )
    mf.density_fit(
        auxbasis=case.auxbasis,
        precision=float(precision),
        mesh="auto",
        omega="auto",
        pair_cut="auto",
        stream_pairs=True,
        metric_tol=float(metric_tol),
    )
    mf.gdf_aux_min_exponent = float(aux_min_exponent)
    mf.gdf_reciprocal_kernel = "range_separated"
    mf.gdf_rs_aux_partition = "smooth"
    mf.gdf_pair_ft_screen_tol = 0.0
    mf.gdf_self_opposite_pair_reuse = True

    started = time.perf_counter()
    mf.with_df.build(workers=min(int(workers), max(1, len(kpts))))
    df_seconds = time.perf_counter() - started
    started = time.perf_counter()
    mf.run(max_cycle=80, conv_tol=1.0e-10, conv_tol_dm=1.0e-8)
    scf_seconds = time.perf_counter() - started
    if not mf.converged:
        raise RuntimeError(f"Native KRHF did not converge for {case.name}.")
    return cell, kpts, mf, df_seconds, scf_seconds


def _run_pyscf_kgw(mf, *, bands, kptlist, ac_nw, finite_size):
    from pyscf.pbc import gw

    kgw = gw.KGW(mf, freq_int="ac")
    kgw.linearized = True
    kgw.ac = "twopole"
    kgw.fc = bool(finite_size)
    kgw.verbose = 0
    started = time.perf_counter()
    qp = kgw.kernel(
        orbs=list(bands),
        kptlist=list(kptlist),
        nw=int(ac_nw),
    )
    seconds = time.perf_counter() - started
    if not kgw.converged:
        raise RuntimeError("PySCF KRGWAC did not converge.")
    return np.asarray(qp, dtype=float), seconds


def _run_pyqed_gw(
    space,
    *,
    component,
    bands,
    kptlist,
    ac_nw,
    finite_size,
    head_method,
):
    started = time.perf_counter()
    result = diagonal_g0w0(
        space,
        coulomb_component=component,
        direct_scale=1.0,
        linearized=True,
        frequency_integration="ac",
        ac_nw=int(ac_nw),
        finite_size_correction=bool(finite_size),
        finite_size_head_method=str(head_method),
        qp_bands={int(k_index): list(bands) for k_index in kptlist},
    )
    return result, time.perf_counter() - started


def _selected(values, bands, kptlist):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("Orbital energies must have shape (nmo,) or (nkpts, nmo).")
    return values[np.asarray(kptlist, dtype=int)[:, None], np.asarray(bands, dtype=int)]


def _gamma_index(kpts, tol=1.0e-10):
    norms = np.linalg.norm(np.asarray(kpts, dtype=float), axis=1)
    index = int(np.argmin(norms))
    if norms[index] > float(tol):
        raise ValueError(
            "The requested Gamma-only benchmark needs a k mesh containing Gamma; "
            "use --gamma-centered for an even mesh."
        )
    return index


def _plot(payload, output):
    import matplotlib.pyplot as plt

    bands = payload["bands"]
    pyscf_qp = np.asarray(payload["pyscf"]["qp_Ha"], dtype=float)
    native_qp = np.asarray(payload["pyqed"]["qp_Ha"], dtype=float)
    x = np.arange(pyscf_qp.shape[0])
    zero = float(np.max(pyscf_qp[:, 0]))
    colors = ("#0072B2", "#D55E00")
    labels = ("VBM", "CBM")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.7, 5.3),
        sharex=True,
        gridspec_kw={"height_ratios": (1.45, 1.0), "hspace": 0.08},
    )
    for iband, (color, label) in enumerate(zip(colors, labels)):
        axes[0].plot(
            x,
            (pyscf_qp[:, iband] - zero) * au2ev,
            color=color,
            marker="o",
            markersize=4.2,
            linewidth=1.25,
            label=f"PySCF {label}",
        )
        axes[0].plot(
            x,
            (native_qp[:, iband] - zero) * au2ev,
            color=color,
            marker="x",
            markersize=5.0,
            linewidth=0,
            markeredgewidth=1.15,
            label=f"PyQED {label}",
        )

    delta = (native_qp - pyscf_qp) * au2mev
    for iband, (color, label) in enumerate(zip(colors, labels)):
        axes[1].plot(
            x,
            delta[:, iband],
            color=color,
            marker="o",
            markersize=4.2,
            linewidth=1.15,
            label=label,
        )
    aligned = payload.get("aligned_diagnostic")
    head_aligned = payload.get("head_aligned_diagnostic")
    if head_aligned is not None:
        head_aligned_qp = np.asarray(head_aligned["qp_Ha"], dtype=float)
        head_aligned_delta = (head_aligned_qp - pyscf_qp) * au2mev
        axes[1].plot(
            x,
            np.max(np.abs(head_aligned_delta), axis=1),
            color="#CC79A7",
            marker="^",
            markersize=4.0,
            linewidth=1.0,
            linestyle=":",
            label="Head-aligned max abs.",
        )
    if aligned is not None:
        aligned_qp = np.asarray(aligned["qp_Ha"], dtype=float)
        aligned_delta = (aligned_qp - pyscf_qp) * au2mev
        axes[1].plot(
            x,
            np.max(np.abs(aligned_delta), axis=1),
            color="#009E73",
            marker="s",
            markersize=3.8,
            linewidth=1.0,
            linestyle="--",
            label="Aligned max abs.",
        )

    axes[0].set_ylabel(r"$E^{\mathrm{QP}}-E_{\mathrm{VBM}}$ (eV)")
    axes[1].set_ylabel(r"$E^{\mathrm{QP}}_{\mathrm{PyQED}}-E^{\mathrm{QP}}_{\mathrm{PySCF}}$ (meV)")
    axes[1].set_xlabel("Target k-point index")
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    axes[1].legend(frameon=False, fontsize=8.0, ncol=2)
    for panel, axis in zip(("a", "b"), axes):
        axis.text(
            0.01,
            0.95,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[1].set_xticks(x)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.11, top=0.88)

    output = Path(output)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    fig.savefig(png, dpi=360, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def benchmark(args):
    import pyscf

    case = replace(
        CASES[args.case],
        kmesh=args.kmesh,
        gamma_centered=bool(args.gamma_centered),
    )
    bands = None

    (
        _,
        _,
        pyscf_mf,
        pyscf_scf_energy,
        pyscf_df_seconds,
        pyscf_scf_seconds,
    ) = _pyscf_reference(
        case,
        args.reference_precision,
        aux_min_exponent=args.aux_min_exponent,
        metric_tol=args.metric_tol,
        force_metric_eig=args.pyscf_metric_eig,
    )
    cell, kpts, native_mf, native_df_seconds, native_scf_seconds = _native_krhf(
        case,
        precision=args.precision,
        aux_min_exponent=args.aux_min_exponent,
        metric_tol=args.metric_tol,
        workers=args.workers,
    )
    nocc = cell.nelectron // 2
    bands = (max(0, nocc - 1), min(cell.nao - 1, nocc))
    kptlist = (
        (_gamma_index(kpts),)
        if args.gamma_only
        else tuple(range(len(kpts)))
    )

    native_space = KPointTransitionSpace(native_mf, qpts="mesh")
    native_gw, native_gw_seconds = _run_pyqed_gw(
        native_space,
        component=GDF,
        bands=bands,
        kptlist=kptlist,
        ac_nw=args.ac_nw,
        finite_size=args.finite_size,
        head_method="builtin_gradient",
    )
    pyscf_qp_full, pyscf_gw_seconds = _run_pyscf_kgw(
        pyscf_mf,
        bands=bands,
        kptlist=kptlist,
        ac_nw=args.ac_nw,
        finite_size=args.finite_size,
    )

    native_qp = _selected(native_gw.e_qp, bands, kptlist)
    pyscf_qp = _selected(pyscf_qp_full, bands, kptlist)
    native_mo = _selected(native_mf.mo_energy, bands, kptlist)
    pyscf_mo = _selected(pyscf_mf.mo_energy, bands, kptlist)
    qp_delta = native_qp - pyscf_qp
    mf_delta = native_mo - pyscf_mo

    payload = {
        "case": case.name,
        "kmesh": list(case.kmesh),
        "gamma_centered": bool(case.gamma_centered),
        "gamma_only": bool(args.gamma_only),
        "target_k_indices": list(kptlist),
        "target_kpts": np.asarray(kpts)[list(kptlist)].tolist(),
        "nkpts": int(len(kpts)),
        "nao": int(cell.nao),
        "bands": list(bands),
        "ac_nw": int(args.ac_nw),
        "finite_size_correction": bool(args.finite_size),
        "precision": float(args.precision),
        "reference_precision": float(args.reference_precision),
        "aux_min_exponent": float(args.aux_min_exponent),
        "metric_tol": float(args.metric_tol),
        "pyscf_metric_eig": bool(args.pyscf_metric_eig),
        "pyscf_version": str(pyscf.__version__),
        "pyqed": {
            "scf_energy_Ha": float(native_mf.e_tot),
            "mo_energy_Ha": native_mo.tolist(),
            "qp_Ha": native_qp.tolist(),
            "gdf_seconds": float(native_df_seconds),
            "scf_seconds": float(native_scf_seconds),
            "gw_seconds": float(native_gw_seconds),
            "finite_size_method": native_gw.info.get("finite_size_method"),
        },
        "pyscf": {
            "scf_energy_Ha": float(pyscf_scf_energy),
            "mo_energy_Ha": pyscf_mo.tolist(),
            "qp_Ha": pyscf_qp.tolist(),
            "gdf_seconds": float(pyscf_df_seconds),
            "scf_seconds": float(pyscf_scf_seconds),
            "gw_seconds": float(pyscf_gw_seconds),
        },
        "comparison": {
            "scf_total_energy_error_Ha": float(native_mf.e_tot - pyscf_scf_energy),
            "max_abs_mf_orbital_error_Ha": float(np.max(np.abs(mf_delta))),
            "max_abs_mf_orbital_error_meV": float(
                au2mev * np.max(np.abs(mf_delta))
            ),
            "max_abs_qp_error_Ha": float(np.max(np.abs(qp_delta))),
            "max_abs_qp_error_meV": float(au2mev * np.max(np.abs(qp_delta))),
            "rms_qp_error_meV": float(
                au2mev * np.sqrt(np.mean(np.abs(qp_delta) ** 2))
            ),
            "qp_error_meV": (au2mev * qp_delta).tolist(),
        },
    }

    if not args.skip_aligned:
        if args.finite_size:
            head_aligned_gw, head_aligned_seconds = _run_pyqed_gw(
                native_space,
                component=GDF,
                bands=bands,
                kptlist=kptlist,
                ac_nw=args.ac_nw,
                finite_size=True,
                head_method="pyscf_gradient",
            )
            head_aligned_qp = _selected(
                head_aligned_gw.e_qp, bands, kptlist
            )
            head_aligned_delta = head_aligned_qp - pyscf_qp
            payload["head_aligned_diagnostic"] = {
                "description": (
                    "Native PyQED KRHF/GDF/GW with the benchmark-only "
                    "PySCF grid gradient head"
                ),
                "qp_Ha": head_aligned_qp.tolist(),
                "seconds": float(head_aligned_seconds),
                "max_abs_qp_error_Ha": float(
                    np.max(np.abs(head_aligned_delta))
                ),
                "max_abs_qp_error_meV": float(
                    au2mev * np.max(np.abs(head_aligned_delta))
                ),
            }
        aligned_space = _native_space(
            case,
            cell,
            kpts,
            pyscf_mf,
            args.precision,
            reciprocal_kernel="range_separated",
            rs_aux_partition="smooth",
            aux_min_exponent=args.aux_min_exponent,
            metric_tol=args.metric_tol,
        )
        aligned_gw, aligned_seconds = _run_pyqed_gw(
            aligned_space,
            component=PYSCF_GDF,
            bands=bands,
            kptlist=kptlist,
            ac_nw=args.ac_nw,
            finite_size=args.finite_size,
            head_method="pyscf_gradient",
        )
        aligned_qp = _selected(aligned_gw.e_qp, bands, kptlist)
        aligned_delta = aligned_qp - pyscf_qp
        payload["aligned_diagnostic"] = {
            "description": "PyQED GW on PySCF KRHF orbitals and GDF factors",
            "qp_Ha": aligned_qp.tolist(),
            "seconds": float(aligned_seconds),
            "max_abs_qp_error_Ha": float(np.max(np.abs(aligned_delta))),
            "max_abs_qp_error_meV": float(
                au2mev * np.max(np.abs(aligned_delta))
            ),
        }
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=tuple(CASES),
        default="lih-rocksalt-2k-svp-solid",
    )
    parser.add_argument("--kmesh", type=_parse_kmesh, default=(4, 2, 2))
    parser.add_argument("--gamma-centered", action="store_true")
    parser.add_argument(
        "--gamma-only",
        action="store_true",
        help="solve QP roots only at Gamma while retaining the full mesh in screening",
    )
    parser.add_argument("--precision", type=float, default=1.0e-12)
    parser.add_argument("--reference-precision", type=float, default=1.0e-12)
    parser.add_argument("--aux-min-exponent", type=float, default=0.075)
    parser.add_argument("--metric-tol", type=float)
    parser.add_argument("--pyscf-metric-eig", action="store_true")
    parser.add_argument("--ac-nw", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-finite-size", dest="finite_size", action="store_false")
    parser.add_argument("--skip-aligned", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_lih_422_pyqed_pyscf_kgw.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("/private/tmp/pbc_lih_422_pyqed_pyscf_kgw"),
    )
    args = parser.parse_args()
    metric_tol_was_explicit = args.metric_tol is not None
    if args.metric_tol is None:
        args.metric_tol = max(1.0e-14, 0.1 * float(args.precision))

    payload = benchmark(args)
    payload["metric_tol_policy"] = (
        "explicit" if metric_tol_was_explicit else "precision_auto"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    png, pdf = _plot(payload, args.figure)
    summary = {
        "output": str(args.output),
        "figure_png": str(png),
        "figure_pdf": str(pdf),
        **payload["comparison"],
    }
    if "aligned_diagnostic" in payload:
        summary["aligned_max_abs_qp_error_meV"] = payload[
            "aligned_diagnostic"
        ]["max_abs_qp_error_meV"]
    if "head_aligned_diagnostic" in payload:
        summary["head_aligned_max_abs_qp_error_meV"] = payload[
            "head_aligned_diagnostic"
        ]["max_abs_qp_error_meV"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
