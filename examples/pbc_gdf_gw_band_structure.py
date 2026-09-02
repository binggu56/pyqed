#!/usr/bin/env python3
"""Plot direct native-GDF KRHF and G0W0 bands on an arbitrary k path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.pbc.gw import KGW
from pyqed.qchem.pbc import Cell
from pyqed.units import au2ev


def _mesh(value):
    mesh = tuple(int(item) for item in str(value).split(","))
    if len(mesh) != 3 or any(item <= 0 for item in mesh):
        raise argparse.ArgumentTypeError("kmesh must contain three positive integers")
    return mesh


def _plot(output, scaled_path, mf_energy, qp_energy, energy_zero):
    mf_shifted = (np.asarray(mf_energy) - float(energy_zero)) * au2ev
    qp_shifted = (np.asarray(qp_energy) - float(energy_zero)) * au2ev
    fig, axis = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    for band in range(mf_shifted.shape[1]):
        axis.plot(
            scaled_path,
            mf_shifted[:, band],
            color="#666666",
            linewidth=1.2,
            linestyle="--",
            label="KRHF" if band == 0 else None,
        )
        axis.plot(
            scaled_path,
            qp_shifted[:, band],
            color="#0072B2",
            linewidth=1.6,
            label=r"$G_0W_0$" if band == 0 else None,
        )
    axis.axhline(0.0, color="#222222", linewidth=0.8)
    axis.set(
        xlim=(float(scaled_path[0]), float(scaled_path[-1])),
        xlabel=r"Scaled momentum along $k_x$",
        ylabel=r"Energy $-E_{\mathrm{VBM}}$ (eV)",
    )
    axis.set_xticks(
        [float(scaled_path[0]), float(scaled_path[-1])],
        [r"$\Gamma$", "X"],
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kmesh", type=_mesh, default=(2, 2, 2))
    parser.add_argument("--path-points", type=int, default=10)
    parser.add_argument("--precision", type=float, default=1.0e-8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eta", type=float, default=1.0e-3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_h2_gdf_gw_band_structure"),
    )
    args = parser.parse_args()
    if args.path_points < 2:
        parser.error("--path-points must be at least 2")

    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    kpts = cell.make_kpts(args.kmesh)
    mf = cell.KRHF(
        kpts=kpts,
        eta=0.5,
        real_cut=2,
        pair_cut=2,
        recip_cut=5,
    ).density_fit(
        auxbasis="def2-svp-jkfit",
        precision=args.precision,
        storage="memory",
        stream_pairs=True,
    )

    started = time.perf_counter()
    mf.with_df.build(workers=args.workers)
    gdf_seconds = time.perf_counter() - started
    started = time.perf_counter()
    mf.run(max_cycle=40, conv_tol=1.0e-10, conv_tol_dm=1.0e-8)
    scf_seconds = time.perf_counter() - started
    if not mf.converged:
        raise RuntimeError("GDF-KRHF did not converge")

    started = time.perf_counter()
    gw = KGW(mf, eta=args.eta).g0w0(
        backend="periodic",
        coulomb_component="gdf",
        direct_scale=1.0,
        qp_bands=list(range(cell.nao)),
        intermediate_bands=list(range(cell.nao)),
        prebuild_screening=True,
    )
    gw_seconds = time.perf_counter() - started
    mesh_bands = gw.band_structure(
        kpts=mf.kpts,
        qp_bands=list(range(cell.nao)),
        intermediate_bands=list(range(cell.nao)),
        reference="none",
        pair_workers=args.workers,
    )
    mesh_error = float(np.max(np.abs(mesh_bands["qp_energy"] - gw.e_qp)))

    scaled_path = np.linspace(0.0, 0.5, args.path_points)
    scaled_kpts = np.zeros((args.path_points, 3), dtype=float)
    scaled_kpts[:, 0] = scaled_path
    started = time.perf_counter()
    bands = gw.band_structure(
        scaled_kpts=scaled_kpts,
        qp_bands=list(range(cell.nao)),
        intermediate_bands=list(range(cell.nao)),
        reference="none",
        pair_workers=args.workers,
    )
    band_seconds = time.perf_counter() - started

    output = args.output.expanduser().resolve()
    nocc = cell.nelectron // 2
    path_vbm = float(np.max(np.asarray(bands["mo_energy"])[:, :nocc]))
    _plot(
        output,
        scaled_path,
        bands["mo_energy"],
        bands["qp_energy"],
        path_vbm,
    )
    payload = {
        "case": "h2-cubic",
        "kmesh": list(args.kmesh),
        "path_scaled": scaled_kpts.tolist(),
        "precision": float(args.precision),
        "eta_Ha": float(args.eta),
        "fermi_Ha": float(bands["e_fermi"]),
        "path_vbm_Ha": path_vbm,
        "mf_energy_Ha": np.asarray(bands["mo_energy"]).tolist(),
        "qp_energy_Ha": np.asarray(bands["qp_energy"]).tolist(),
        "sigma_c_Ha": np.asarray(bands["sigma_c"]).real.tolist(),
        "mesh_identity_error_Ha": mesh_error,
        "timings": {
            "gdf_seconds": float(gdf_seconds),
            "scf_seconds": float(scf_seconds),
            "mesh_gw_seconds": float(gw_seconds),
            "band_seconds": float(band_seconds),
            "band": bands["info"],
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    mf.with_df.close()
    print(f"figure: {output.with_suffix('.png')}")
    print(f"json: {output.with_suffix('.json')}")
    print(f"mesh identity error: {mesh_error:.3e} Ha")


if __name__ == "__main__":
    main()
