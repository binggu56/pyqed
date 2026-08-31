#!/usr/bin/env python3
"""Run a native rocksalt LiH KRHF -> GW -> photoemission workflow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.pbc.gw import KGW
from pyqed.qchem.pbc import Cell
from pyqed.units import au2ev


HARTREE_TO_EV = float(au2ev)


def _mesh(value):
    mesh = tuple(int(item) for item in str(value).split(","))
    if len(mesh) != 3 or any(item <= 0 for item in mesh):
        raise argparse.ArgumentTypeError("kmesh must contain three positive integers")
    return mesh


def _positive_integer_or_auto(value):
    if str(value).strip().lower() == "auto":
        return None
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("value must be 'auto' or a positive integer")
    return integer


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _rocksalt_cell(lattice_constant, basis):
    half = 0.5 * float(lattice_constant)
    lattice = np.asarray(
        [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]],
        dtype=float,
    )
    return Cell(
        atom=f"Li 0 0 0; H {half} {half} {half}",
        a=lattice,
        basis=basis,
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()


def _selected_valence_bands(qp_energy, occupation, binding_max_ev, requested):
    occupied = occupation > 1.0e-8
    occupied_bands = np.flatnonzero(np.any(occupied, axis=0)).tolist()
    if requested != "auto":
        bands = sorted({int(item) for item in requested.split(",") if item.strip()})
        if not bands:
            raise ValueError("--bands must be 'auto' or a comma-separated band list")
        if min(bands) < 0 or max(bands) >= qp_energy.shape[1]:
            raise ValueError("--bands contains an out-of-range band index")
        return bands

    vbm = float(np.max(qp_energy[occupied]))
    limit = float(binding_max_ev) / HARTREE_TO_EV
    bands = []
    for band in occupied_bands:
        mask = occupied[:, band]
        if np.any(mask) and float(np.min(vbm - qp_energy[mask, band])) <= limit:
            bands.append(int(band))
    if not bands:
        bands = [int(occupied_bands[-1])]
    return bands


def _peak_rows(peaks):
    return [
        {
            "k_index": int(target[0]),
            "band_index": int(target[1]),
            "binding_energy_eV": float(binding),
            "intensity": float(intensity),
        }
        for target, binding, intensity in zip(
            peaks.targets,
            peaks.binding_energies,
            peaks.intensities,
        )
    ]


def _write_csv(path, binding_ev, kinetic_ev, spectral, photoemission):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "binding_energy_eV",
                "kinetic_energy_eV",
                "intrinsic_signal_per_Ha",
                "raw_photoemission_signal",
                "detector_photoemission_signal",
            ]
        )
        writer.writerows(
            zip(
                binding_ev,
                kinetic_ev,
                spectral.signal,
                photoemission.raw_signal,
                photoemission.signal,
            )
        )


def _plot_results(path, binding_ev, spectral, photoemission, qp_binding_ev):
    intrinsic = np.asarray(spectral.signal, dtype=float)
    measured = np.asarray(photoemission.signal, dtype=float)
    target_intensity = np.asarray(photoemission.target_intensity, dtype=float)
    intrinsic_norm = intrinsic / max(float(np.max(intrinsic)), np.finfo(float).tiny)
    measured_norm = measured / max(float(np.max(measured)), np.finfo(float).tiny)
    target_norm = target_intensity / max(
        float(np.max(target_intensity)), np.finfo(float).tiny
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35))
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.19, top=0.86, wspace=0.36)

    ax = axes[0]
    ax.plot(binding_ev, intrinsic_norm, color="#4C78A8", linewidth=1.5, label="GW spectral")
    ax.plot(binding_ev, measured_norm, color="#E45756", linewidth=1.5, label="PES weighted")
    ax.fill_between(binding_ev, 0.0, measured_norm, color="#E45756", alpha=0.18)
    ax.set_xlim(float(binding_ev[0]), float(binding_ev[-1]))
    ax.set_ylim(0.0, 1.06)
    ax.set_xlabel("Binding energy from VBM (eV)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title("LiH valence photoemission")
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    image = ax.imshow(
        target_norm,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(binding_ev[0], binding_ev[-1], -0.5, len(spectral.targets) - 0.5),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    rows = np.arange(len(spectral.targets))
    inside = (qp_binding_ev >= binding_ev[0]) & (qp_binding_ev <= binding_ev[-1])
    ax.scatter(
        qp_binding_ev[inside],
        rows[inside],
        s=24,
        facecolors="none",
        edgecolors="#56B4E9",
        linewidths=0.9,
        label="GW QP",
    )
    if len(rows) <= 16:
        labels = [f"k{int(k)} b{int(b)}" for k, b in spectral.targets]
        ax.set_yticks(rows, labels)
    ax.set_xlim(float(binding_ev[0]), float(binding_ev[-1]))
    ax.set_xlabel("Binding energy from VBM (eV)")
    ax.set_ylabel("Spectral target")
    ax.set_title("Matrix-element-resolved intensity")
    if np.any(inside):
        legend = ax.legend(frameon=False, loc="upper right")
        plt.setp(legend.get_texts(), color="white")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Relative target intensity")

    for label, ax in zip(("a", "b"), axes):
        ax.text(
            -0.15,
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
    parser.add_argument("--kmesh", type=_mesh, default=(1, 1, 1))
    parser.add_argument("--lattice-constant", type=float, default=7.72)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--auxbasis", default="def2-svp-jkfit")
    parser.add_argument("--precision", type=float, default=1.0e-8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-memory-mb", type=float, default=1024.0)
    parser.add_argument("--cache-dir")
    parser.add_argument("--stream-pair-batch-mb", type=float, default=128.0)
    parser.add_argument(
        "--stream-pair-batch-size",
        type=_positive_integer_or_auto,
        default=None,
    )
    parser.add_argument("--bands", default="auto")
    parser.add_argument("--binding-max-ev", type=float, default=10.0)
    parser.add_argument("--npoints", type=int, default=801)
    parser.add_argument("--eta-ev", type=float, default=0.08)
    parser.add_argument("--intrinsic-broadening-ev", type=float, default=0.08)
    parser.add_argument("--photon-energy-ev", type=float, default=80.0)
    parser.add_argument("--work-function-ev", type=float, default=4.5)
    parser.add_argument("--inner-potential-ev", type=float, default=10.0)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--resolution-ev", type=float, default=0.15)
    parser.add_argument("--momentum-broadening", type=float, default=0.25)
    parser.add_argument("--finite-size-correction", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_lih_gw_pes.json"),
    )
    parser.add_argument("--npz", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()

    if args.binding_max_ev <= 0.0:
        parser.error("--binding-max-ev must be positive")
    if args.npoints < 2:
        parser.error("--npoints must be at least two")
    if args.eta_ev <= 0.0 or args.intrinsic_broadening_ev <= 0.0:
        parser.error("spectral broadenings must be positive")

    npz_path = args.npz or args.output.with_suffix(".npz")
    csv_path = args.csv or args.output.with_suffix(".csv")
    figure_path = args.figure or args.output.with_suffix(".pdf")

    cell = _rocksalt_cell(args.lattice_constant, args.basis)
    mf = cell.KRHF(
        kpts=cell.make_kpts(args.kmesh),
        eta=0.5,
        real_cut="auto",
        pair_cut="auto",
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
        cache_dir=args.cache_dir,
        stream_pairs=True,
        stream_pair_batch_mb=args.stream_pair_batch_mb,
        stream_pair_batch_size=args.stream_pair_batch_size,
    )

    timings = {}
    started = time.perf_counter()
    mf.with_df.build(workers=args.workers)
    timings["gdf_prebuild_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    mf.run(max_cycle=80, conv_tol=1.0e-10, conv_tol_dm=1.0e-8)
    timings["krhf_seconds_after_prebuild"] = time.perf_counter() - started
    if not mf.converged:
        raise RuntimeError("Native GDF-KRHF did not converge for rocksalt LiH.")

    occupation = np.asarray(mf.mo_occ, dtype=float).reshape(mf.nkpts, -1)
    occupied_bands = np.flatnonzero(np.any(occupation > 1.0e-8, axis=0)).tolist()
    started = time.perf_counter()
    gw = KGW(mf, eta=args.eta_ev / HARTREE_TO_EV).g0w0(
        backend="periodic",
        coulomb_component="gdf",
        direct_scale=1.0,
        frequency_integration="poles",
        linearized=True,
        qp_bands=occupied_bands,
        finite_size_correction=args.finite_size_correction,
        prebuild_gdf=True,
        prebuild_gdf_workers=args.workers,
    )
    timings["gw_seconds"] = time.perf_counter() - started
    if not gw.converged:
        raise RuntimeError("Periodic G0W0 did not converge for rocksalt LiH.")

    qp_energy = np.asarray(gw.e_qp, dtype=float).reshape(mf.nkpts, -1)
    selected_bands = _selected_valence_bands(
        qp_energy,
        occupation,
        args.binding_max_ev,
        args.bands,
    )
    binding_grid_ev = np.linspace(0.0, args.binding_max_ev, args.npoints)
    started = time.perf_counter()
    spectral = gw.spectral_function(
        binding_grid=binding_grid_ev,
        units="ev",
        bands=selected_bands,
        occupied_only=True,
        energy_reference="vbm",
        eta=args.eta_ev / HARTREE_TO_EV,
        broadening=args.intrinsic_broadening_ev / HARTREE_TO_EV,
        finite_size_correction=args.finite_size_correction,
    )
    timings["spectral_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    photoemission = gw.experimental_pes(
        spectral_result=spectral,
        photon_energy=args.photon_energy_ev,
        work_function=args.work_function_ev,
        units="ev",
        direction=(0.0, 0.0, 1.0),
        polarization=None,
        surface_normal=(0.0, 0.0, 1.0),
        inner_potential=args.inner_potential_ev,
        temperature=args.temperature_k,
        energy_resolution=args.resolution_ev,
        binding_offset=0.0,
        momentum_broadening=args.momentum_broadening,
    )
    timings["photoemission_seconds"] = time.perf_counter() - started

    binding_ev = np.asarray(spectral.binding_energies) * HARTREE_TO_EV
    kinetic_ev = np.asarray(photoemission.kinetic_energies) * HARTREE_TO_EV
    vbm = float(np.max(qp_energy[occupation > 1.0e-8]))
    qp_binding_ev = HARTREE_TO_EV * np.asarray(
        [vbm - qp_energy[k_index, band_index] for k_index, band_index in spectral.targets]
    )
    grid_spacing = float(binding_ev[1] - binding_ev[0])
    min_distance = max(1, int(round(0.10 / grid_spacing)))
    spectral_peaks = spectral.peaks(
        source="spectral_function",
        units="ev",
        min_distance=min_distance,
        max_peaks=24,
    )
    pes_peaks = photoemission.peaks(
        source="signal",
        units="ev",
        min_distance=min_distance,
        max_peaks=12,
    )

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        binding_energy_eV=binding_ev,
        kinetic_energy_eV=kinetic_ev,
        omega_Ha=spectral.omega,
        targets=spectral.targets,
        kpoints_bohr_inverse=spectral.kpoints,
        occupations=spectral.occupations,
        target_weights=spectral.target_weights,
        spectral_function_per_Ha=spectral.spectral_function,
        intrinsic_signal_per_Ha=spectral.signal,
        sigma_c_Ha=spectral.sigma_c,
        green_function_per_Ha=spectral.green_function,
        matrix_elements=photoemission.matrix_elements,
        matrix_strengths=photoemission.matrix_strengths,
        momentum_weights=photoemission.momentum_weights,
        fermi_factors=photoemission.fermi_factors,
        target_photoemission_intensity=photoemission.target_intensity,
        raw_photoemission_signal=photoemission.raw_signal,
        detector_photoemission_signal=photoemission.signal,
        qp_binding_energy_eV=qp_binding_ev,
    )
    _write_csv(csv_path, binding_ev, kinetic_ev, spectral, photoemission)
    figure_pdf, figure_png = _plot_results(
        figure_path,
        binding_ev,
        spectral,
        photoemission,
        qp_binding_ev,
    )

    result = {
        "system": "rocksalt LiH",
        "lattice_constant_bohr": float(args.lattice_constant),
        "primitive_lattice_bohr": np.asarray(cell.lattice_vectors),
        "kmesh": args.kmesh,
        "nkpts": int(mf.nkpts),
        "nao": int(cell.nao),
        "nelectron_per_cell": int(cell.nelectron),
        "basis": args.basis,
        "auxbasis": args.auxbasis,
        "gdf_precision": float(args.precision),
        "krhf_real_cut": int(mf.real_cut),
        "krhf_pair_cut": int(mf.pair_cut),
        "krhf_one_body_nuclear_cut": int(mf.one_body_nuclear_cut),
        "krhf_one_body_screen_tol": float(mf.one_body_screen_tol),
        "gdf_stream_pair_batch_mb": float(args.stream_pair_batch_mb),
        "gdf_stream_pair_batch_size": args.stream_pair_batch_size,
        "gdf_factor_memory_bytes": int(mf.with_df.memory_bytes),
        "gdf_factor_disk_bytes": int(mf.with_df.disk_bytes),
        "krhf_converged": bool(mf.converged),
        "krhf_iterations": int(mf.niter),
        "krhf_energy_Ha_per_cell": float(mf.e_tot),
        "krhf_integral_build_timings": dict(mf.integral_build_timings),
        "mo_energy_Ha": np.asarray(mf.mo_energy).reshape(mf.nkpts, -1),
        "qp_energy_Ha": qp_energy,
        "occupation": occupation,
        "gw_converged": bool(gw.converged),
        "gw_info": gw.info,
        "selected_spectral_bands": selected_bands,
        "spectral_targets": spectral.targets,
        "binding_range_eV": [float(binding_ev[0]), float(binding_ev[-1])],
        "binding_reference": "valence-band maximum",
        "spectral_peaks": _peak_rows(spectral_peaks),
        "photoemission_peaks": _peak_rows(pes_peaks),
        "spectral_info": spectral.info,
        "photoemission_info": photoemission.info,
        "timings": timings,
        "gdf_build_timings": mf.with_df.build_timings,
        "gdf_multi_q_build_timings": mf.with_df.multi_q_build_timings,
        "gw_cache_sizes": gw._periodic_cache.sizes(),
        "npz": str(npz_path),
        "csv": str(csv_path),
        "figure_pdf": str(figure_pdf),
        "figure_png": str(figure_png),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(result), indent=2) + "\n")

    peak_text = ", ".join(
        f"{row['binding_energy_eV']:.3f}" for row in result["photoemission_peaks"]
    ) or "none"
    print(
        f"KRHF converged in {mf.niter} cycles: E = {mf.e_tot:.12f} Ha/cell\n"
        f"G0W0 targets: {len(spectral.targets)} across bands {selected_bands}\n"
        f"PES peaks in 0-{args.binding_max_ev:g} eV: {peak_text} eV\n"
        f"Timings (s): {json.dumps(_json_ready(timings), sort_keys=True)}\n"
        f"Wrote {args.output}\n"
        f"Wrote {npz_path}\n"
        f"Wrote {csv_path}\n"
        f"Wrote {figure_pdf}\n"
        f"Wrote {figure_png}"
    )
    mf.with_df.close()


if __name__ == "__main__":
    main()
