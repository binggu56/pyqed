#!/usr/bin/env python3
"""Run a native periodic H2 KRHF -> GW -> BSE workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import au2ev
from pyqed.pbc.gw import KBSE, KGW, KTDA
from pyqed.qchem.pbc import Cell


HARTREE_TO_EV = au2ev


def _mesh(value):
    mesh = tuple(int(item) for item in str(value).split(","))
    if len(mesh) != 3 or any(item <= 0 for item in mesh):
        raise argparse.ArgumentTypeError("kmesh must contain three positive integers")
    return mesh


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _band_gaps(energy, occupation):
    energy = np.asarray(energy, dtype=float)
    occupation = np.asarray(occupation, dtype=float)
    occupied = occupation > 1.0e-8
    virtual = ~occupied
    if not np.any(occupied) or not np.any(virtual):
        return {"fundamental_Ha": None, "minimum_direct_Ha": None}
    direct = [
        np.min(row[virtual[k]]) - np.max(row[occupied[k]])
        for k, row in enumerate(energy)
    ]
    return {
        "fundamental_Ha": float(np.min(energy[virtual]) - np.max(energy[occupied])),
        "minimum_direct_Ha": float(np.min(direct)),
    }


def _plot_results(path, mo_energy, qp_energy, occupation, tda_energy, bse_energy):
    mo_energy = np.asarray(mo_energy, dtype=float)
    qp_energy = np.asarray(qp_energy, dtype=float)
    occupation = np.asarray(occupation, dtype=float)
    tda_ev = HARTREE_TO_EV * np.asarray(tda_energy, dtype=float)
    bse_ev = HARTREE_TO_EV * np.asarray(bse_energy, dtype=float)
    k_index = np.arange(mo_energy.shape[0])
    vbm = float(np.max(mo_energy[occupation > 1.0e-8]))

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.86, wspace=0.34)

    ax = axes[0]
    colors = ("#4C78A8", "#E45756")
    for band in range(mo_energy.shape[1]):
        occupied_band = bool(np.any(occupation[:, band] > 1.0e-8))
        color = colors[0 if occupied_band else 1]
        label_suffix = "occupied" if occupied_band else "virtual"
        ax.plot(
            k_index,
            HARTREE_TO_EV * (mo_energy[:, band] - vbm),
            "o--",
            color=color,
            markerfacecolor="white",
            linewidth=1.2,
            markersize=4.5,
            label=f"KRHF {label_suffix}",
        )
        ax.plot(
            k_index,
            HARTREE_TO_EV * (qp_energy[:, band] - vbm),
            "s-",
            color=color,
            linewidth=1.4,
            markersize=4.2,
            label=f"GW {label_suffix}",
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, loc="best")
    ax.axhline(0.0, color="0.45", linewidth=0.8)
    ax.set_xticks(k_index)
    ax.set_xlabel("k-point index")
    ax.set_ylabel("Energy relative to KRHF VBM (eV)")
    ax.set_title("Quasiparticle bands")
    ax.grid(axis="y", color="0.9", linewidth=0.7)

    ax = axes[1]
    nroot = max(len(tda_ev), len(bse_ev))
    root = np.arange(nroot)
    width = 0.34
    if len(tda_ev):
        ax.bar(
            root[: len(tda_ev)] - width / 2,
            tda_ev,
            width,
            color="#59A14F",
            edgecolor="black",
            linewidth=0.6,
            label="TDA",
        )
    if len(bse_ev):
        ax.bar(
            root[: len(bse_ev)] + width / 2,
            bse_ev,
            width,
            color="#F28E2B",
            edgecolor="black",
            linewidth=0.6,
            hatch="//",
            label="BSE",
        )
    qp_direct_gap = _band_gaps(qp_energy, occupation)["minimum_direct_Ha"]
    if qp_direct_gap is not None:
        ax.axhline(
            HARTREE_TO_EV * qp_direct_gap,
            color="0.25",
            linestyle=":",
            linewidth=1.2,
            label="QP direct gap",
        )
    ax.set_xticks(root, [f"S{index + 1}" for index in root])
    ax.set_ylabel("Excitation energy (eV)")
    ax.set_title(r"Vertical $q=0$ excitations")
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    values = np.concatenate((tda_ev, bse_ev))
    ymax = max(
        float(np.max(values)) if values.size else 1.0,
        HARTREE_TO_EV * qp_direct_gap if qp_direct_gap is not None else 0.0,
    )
    ax.set_ylim(0.0, 1.28 * ymax)
    ax.legend(frameon=False, loc="upper center", ncol=3)

    for label, ax in zip(("a", "b"), axes):
        ax.text(
            -0.14,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = path.with_suffix(".pdf")
    png_path = path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kmesh", type=_mesh, default=(2, 1, 1))
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--auxbasis", default="def2-svp-jkfit")
    parser.add_argument("--precision", type=float, default=1.0e-8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-memory-mb", type=float, default=1024.0)
    parser.add_argument("--gw-method", choices=("g0w0", "evgw", "gnw0"), default="g0w0")
    parser.add_argument("--gw-max-cycle", type=int, default=1)
    parser.add_argument(
        "--coulomb-component",
        choices=("gdf", "full_ewald", "reciprocal_ewald_lr"),
        default="gdf",
    )
    parser.add_argument(
        "--bse-screening-energy", choices=("mf", "qp"), default="qp"
    )
    parser.add_argument("--nroots", type=int, default=2)
    parser.add_argument("--skip-bse", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_h2_gw_bse.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("/private/tmp/pbc_h2_gw_bse.pdf"),
    )
    args = parser.parse_args()

    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis=args.basis,
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        kpts=cell.make_kpts(args.kmesh),
        eta=0.5,
        real_cut=2,
        pair_cut=2,
        recip_cut=5,
    ).density_fit(
        auxbasis=args.auxbasis,
        precision=args.precision,
        mesh="auto",
        omega="auto",
        pair_cut="auto",
        image_cut="auto",
        storage="auto",
        max_memory_mb=args.max_memory_mb,
        stream_pairs=True,
    )

    timings = {}
    started = time.perf_counter()
    mf.with_df.build(workers=args.workers)
    timings["gdf_prebuild_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    mf.run(max_cycle=80, conv_tol=1.0e-10, conv_tol_dm=1.0e-8)
    timings["krhf_seconds_after_prebuild"] = time.perf_counter() - started
    if not mf.converged:
        raise RuntimeError("Native GDF-KRHF did not converge.")

    gw_options = {
        "backend": "periodic",
        "coulomb_component": args.coulomb_component,
        "direct_scale": 1.0,
        "prebuild_gdf": args.coulomb_component == "gdf",
        "prebuild_gdf_workers": args.workers,
    }
    if args.gw_method in ("evgw", "gnw0"):
        gw_options.update(
            max_cycle=args.gw_max_cycle,
            solve_roots=False,
        )
    started = time.perf_counter()
    gw = getattr(KGW(mf, eta=1.0e-3), args.gw_method)(**gw_options)
    timings["gw_seconds"] = time.perf_counter() - started

    tda_energy = np.zeros(0, dtype=float)
    bse_energy = np.zeros(0, dtype=float)
    nroots = 0
    if not args.skip_bse:
        occupation = np.asarray(mf.mo_occ, dtype=float).reshape(mf.nkpts, -1)
        ntransitions = sum(
            int(np.count_nonzero(row > 1.0e-8))
            * int(np.count_nonzero(row <= 1.0e-8))
            for row in occupation
        )
        nroots = min(int(args.nroots), ntransitions)
        if nroots <= 0:
            raise RuntimeError("The q=0 transition space is empty.")
        common = {
            "backend": "periodic",
            "q_index": 0,
            "coulomb_component": args.coulomb_component,
            "direct_scale": 1.0,
            "nroots": nroots,
            "return_vectors": True,
            "screening_from_qp": args.bse_screening_energy == "qp",
        }
        started = time.perf_counter()
        tda = KTDA(gw).run(**common)
        timings["tda_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        bse = KBSE(gw).run(**common)
        timings["bse_seconds"] = time.perf_counter() - started
        tda_energy = np.asarray(tda.e, dtype=float)
        bse_energy = np.asarray(bse.e, dtype=float)

    mo_energy = np.asarray(mf.mo_energy, dtype=float).reshape(mf.nkpts, -1)
    qp_energy = np.asarray(gw.e_qp, dtype=float).reshape(mf.nkpts, -1)
    occupation = np.asarray(mf.mo_occ, dtype=float).reshape(mf.nkpts, -1)
    figure_pdf, figure_png = _plot_results(
        args.figure,
        mo_energy,
        qp_energy,
        occupation,
        tda_energy,
        bse_energy,
    )
    result = {
        "system": "periodic H2",
        "kmesh": args.kmesh,
        "nkpts": int(mf.nkpts),
        "nao": int(cell.nao),
        "basis": args.basis,
        "auxbasis": args.auxbasis,
        "gdf_precision": float(args.precision),
        "gdf_factor_memory_bytes": int(mf.with_df.memory_bytes),
        "gdf_factor_disk_bytes": int(mf.with_df.disk_bytes),
        "krhf_converged": bool(mf.converged),
        "krhf_iterations": int(mf.niter),
        "krhf_energy_Ha": float(mf.e_tot),
        "mo_energy_Ha": mo_energy,
        "qp_energy_Ha": qp_energy,
        "occupation": occupation,
        "mf_gap": _band_gaps(mo_energy, occupation),
        "gw_gap": _band_gaps(qp_energy, occupation),
        "gw_method": args.gw_method,
        "gw_converged": bool(gw.info.get("all_converged", gw.info.get("converged", False))),
        "coulomb_component": args.coulomb_component,
        "tda_energy_Ha": tda_energy,
        "bse_energy_Ha": bse_energy,
        "requested_bse_roots": int(args.nroots),
        "resolved_bse_roots": int(nroots),
        "timings": timings,
        "gdf_build_timings": mf.with_df.build_timings,
        "gw_cache_sizes": gw._periodic_cache.sizes(),
        "figure_pdf": str(figure_pdf),
        "figure_png": str(figure_png),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    mf_gap_ev = HARTREE_TO_EV * result["mf_gap"]["fundamental_Ha"]
    gw_gap_ev = HARTREE_TO_EV * result["gw_gap"]["fundamental_Ha"]
    print(
        f"KRHF converged in {mf.niter} cycles: E = {mf.e_tot:.12f} Ha\n"
        f"Fundamental gap: KRHF {mf_gap_ev:.6f} eV, "
        f"{args.gw_method} {gw_gap_ev:.6f} eV\n"
        f"TDA roots (eV): {HARTREE_TO_EV * tda_energy}\n"
        f"BSE roots (eV): {HARTREE_TO_EV * bse_energy}\n"
        f"Timings (s): {json.dumps(_json_ready(timings), sort_keys=True)}\n"
        f"Wrote {args.output}\n"
        f"Wrote {figure_pdf}\n"
        f"Wrote {figure_png}"
    )


if __name__ == "__main__":
    main()
