#!/usr/bin/env python3
"""Benchmark native PyQED periodic G0W0 against PySCF KGWAC.

The primary comparison is end to end: each package builds and converges its
own GDF-KRHF reference before evaluating linearized analytic-continuation GW.
An optional aligned diagnostic runs the PyQED GW solver from the PySCF mean
field and PySCF GDF factors to isolate solver/convention differences.
``--isolated-memory`` runs the two primary calculations in sequential fresh
processes so their RSS baselines and allocator arenas cannot contaminate one
another.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
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


class _PeakRSSSampler:
    """Sample process RSS while one benchmark phase is active."""

    def __init__(self, interval=0.05):
        self.interval = float(interval)
        self.start_rss_mb = None
        self.peak_rss_mb = None
        self.end_rss_mb = None
        self.peak_increment_mb = None
        self.retained_increment_mb = None
        self._stop = threading.Event()
        self._thread = None

    def _rss_mb(self):
        import psutil

        return float(psutil.Process().memory_info().rss) / 1.0e6

    def _sample(self):
        while not self._stop.wait(self.interval):
            self.peak_rss_mb = max(self.peak_rss_mb, self._rss_mb())

    def __enter__(self):
        self.start_rss_mb = self._rss_mb()
        self.peak_rss_mb = self.start_rss_mb
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.end_rss_mb = self._rss_mb()
        self.peak_rss_mb = max(self.peak_rss_mb, self.end_rss_mb)
        self._stop.set()
        self._thread.join()
        self.peak_increment_mb = max(
            0.0,
            self.peak_rss_mb - self.start_rss_mb,
        )
        self.retained_increment_mb = max(
            0.0,
            self.end_rss_mb - self.start_rss_mb,
        )

    def as_dict(self):
        return {
            "start_rss_mb": float(self.start_rss_mb),
            "peak_rss_mb": float(self.peak_rss_mb),
            "end_rss_mb": float(self.end_rss_mb),
            "peak_increment_mb": float(self.peak_increment_mb),
            "retained_increment_mb": float(self.retained_increment_mb),
        }


def _pyqed_krhf(
    case,
    *,
    precision,
    aux_min_exponent,
    metric_tol,
    workers,
    stream_pair_batch_mb=128.0,
    folded_batch_mb=128.0,
    storage="auto",
    max_memory_mb=512.0,
):
    cell, kpts, _ = _seed_native_reference(case)
    mf = cell.KRHF(
        kpts=kpts,
        eta=0.5,
        real_cut="auto",
        pair_cut="auto",
        recip_cut=5,
        jk_builder="gdf",
        occupation_mode=case.occupation_mode,
    )
    mf.density_fit(
        auxbasis=case.auxbasis,
        precision=float(precision),
        mesh="auto",
        omega="auto",
        pair_cut="auto",
        stream_pairs=True,
        stream_pair_batch_mb=float(stream_pair_batch_mb),
        folded_batch_mb=float(folded_batch_mb),
        storage=str(storage),
        max_memory_mb=float(max_memory_mb),
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
    return values[
        np.asarray(kptlist, dtype=int)[:, None],
        np.asarray(bands, dtype=int),
    ]


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


def _plot_performance(payload, output):
    import matplotlib.pyplot as plt

    phases = ("GDF", "KRHF", r"$G_0W_0$")
    timing_keys = ("gdf_seconds", "scf_seconds", "gw_seconds")
    pyqed_times = np.asarray([payload["pyqed"][key] for key in timing_keys])
    pyscf_times = np.asarray([payload["pyscf"][key] for key in timing_keys])
    qp_errors = np.asarray(payload["comparison"]["qp_error_meV"])[0]
    memory = payload.get("memory")

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.15))
    x = np.arange(len(phases))
    width = 0.34
    axes[0].bar(
        x - width / 2,
        pyqed_times,
        width,
        color="#0072B2",
        label="PyQED",
    )
    axes[0].bar(
        x + width / 2,
        pyscf_times,
        width,
        color="#D55E00",
        label="PySCF",
    )
    axes[0].set_xticks(x, phases)
    axes[0].set_ylabel("Wall time (s)")
    axes[0].legend(frameon=False, fontsize=8.5)

    error_x = np.arange(2)
    axes[1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1].scatter(
        error_x,
        qp_errors,
        color=("#0072B2", "#D55E00"),
        marker="o",
        s=38,
        zorder=3,
    )
    axes[1].vlines(
        error_x,
        0.0,
        qp_errors,
        color=("#0072B2", "#D55E00"),
        linewidth=1.2,
    )
    axes[1].set_xticks(error_x, ("VBM", "CBM"))
    axes[1].set_ylabel(r"$E^{\rm QP}_{\rm PyQED}-E^{\rm QP}_{\rm PySCF}$ (meV)")
    bound = max(0.005, 1.25 * float(np.max(np.abs(qp_errors))))
    axes[1].set_ylim(-bound, bound)

    if memory is None:
        axes[2].axis("off")
    else:
        if memory.get("isolated_processes", False):
            labels = (
                "Process\npeak",
                "Retained\nabove base",
                "$G_0W_0$\nextra",
            )
            pyqed_memory = (
                memory["pyqed_process"]["peak_rss_mb"],
                memory["pyqed_process"]["retained_increment_mb"],
                memory["pyqed_gw"]["peak_increment_mb"],
            )
            pyscf_memory = (
                memory["pyscf_process"]["peak_rss_mb"],
                memory["pyscf_process"]["retained_increment_mb"],
                memory["pyscf_gw"]["peak_increment_mb"],
            )
            memory_ylabel = "RSS (MB)"
        else:
            labels = ("Reference", r"$G_0W_0$")
            pyqed_memory = (
                memory["pyqed_reference"]["peak_increment_mb"],
                memory["pyqed_gw"]["peak_increment_mb"],
            )
            pyscf_memory = (
                memory["pyscf_reference"]["peak_increment_mb"],
                memory["pyscf_gw"]["peak_increment_mb"],
            )
            memory_ylabel = "Peak RSS increment (MB)"
        memory_x = np.arange(len(labels))
        axes[2].bar(
            memory_x - width / 2,
            pyqed_memory,
            width,
            color="#0072B2",
        )
        axes[2].bar(
            memory_x + width / 2,
            pyscf_memory,
            width,
            color="#D55E00",
        )
        axes[2].set_xticks(memory_x, labels)
        axes[2].set_yscale("log")
        axes[2].set_ylabel(memory_ylabel)

    for label, axis in zip(("a", "b", "c"), axes):
        if not axis.axison:
            continue
        axis.text(
            -0.13,
            1.03,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            clip_on=False,
        )
        axis.grid(axis="y", color="0.9", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.18,
        top=0.91,
        wspace=0.42,
    )

    output = Path(output)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    fig.savefig(png, dpi=360, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _isolated_package_benchmark(args, package):
    """Run one package reference and GW calculation in a fresh process."""

    if package not in ("pyqed", "pyscf"):
        raise ValueError("package must be 'pyqed' or 'pyscf'.")
    if package == "pyscf":
        import pyscf

        package_version = str(pyscf.__version__)
    else:
        package_version = None

    case = replace(
        CASES[args.case],
        kmesh=args.kmesh,
        gamma_centered=bool(args.gamma_centered),
    )
    with _PeakRSSSampler() as process_memory:
        if package == "pyqed":
            with _PeakRSSSampler() as reference_memory:
                cell, kpts, mean_field, gdf_seconds, scf_seconds = _pyqed_krhf(
                    case,
                    precision=args.precision,
                    aux_min_exponent=args.aux_min_exponent,
                    metric_tol=args.metric_tol,
                    workers=args.workers,
                    stream_pair_batch_mb=args.stream_pair_batch_mb,
                    folded_batch_mb=args.folded_batch_mb,
                    storage=args.storage,
                    max_memory_mb=args.max_memory_mb,
                )
            scf_energy = float(mean_field.e_tot)
        else:
            with _PeakRSSSampler() as reference_memory:
                (
                    cell,
                    kpts,
                    mean_field,
                    scf_energy,
                    gdf_seconds,
                    scf_seconds,
                ) = _pyscf_reference(
                    case,
                    args.reference_precision,
                    aux_min_exponent=args.aux_min_exponent,
                    metric_tol=args.metric_tol,
                    force_metric_eig=args.pyscf_metric_eig,
                )

        nocc = cell.nelectron // 2
        bands = (max(0, nocc - 1), min(cell.nao - 1, nocc))
        kptlist = (
            (_gamma_index(kpts),)
            if args.gamma_only
            else tuple(range(len(kpts)))
        )
        with _PeakRSSSampler() as gw_memory:
            if package == "pyqed":
                result, gw_seconds = _run_pyqed_gw(
                    KPointTransitionSpace(mean_field, qpts="mesh"),
                    component=GDF,
                    bands=bands,
                    kptlist=kptlist,
                    ac_nw=args.ac_nw,
                    finite_size=args.finite_size,
                    head_method="builtin_gradient",
                )
                qp = _selected(result.e_qp, bands, kptlist)
                ac_profile = result.info.get("ac_profile", {})
                finite_size_method = result.info.get("finite_size_method")
            else:
                qp_full, gw_seconds = _run_pyscf_kgw(
                    mean_field,
                    bands=bands,
                    kptlist=kptlist,
                    ac_nw=args.ac_nw,
                    finite_size=args.finite_size,
                )
                qp = _selected(qp_full, bands, kptlist)
                ac_profile = None
                finite_size_method = None
        mo_energy = _selected(mean_field.mo_energy, bands, kptlist)

    calculation = {
        "scf_energy_Ha": float(scf_energy),
        "mo_energy_Ha": mo_energy.tolist(),
        "qp_Ha": qp.tolist(),
        "gdf_seconds": float(gdf_seconds),
        "scf_seconds": float(scf_seconds),
        "gw_seconds": float(gw_seconds),
    }
    if ac_profile is not None:
        calculation["finite_size_method"] = finite_size_method
        calculation["ac_profile"] = ac_profile
    return {
        "package": package,
        "package_version": package_version,
        "metadata": {
            "case": case.name,
            "kmesh": list(case.kmesh),
            "gamma_centered": bool(case.gamma_centered),
            "gamma_only": bool(args.gamma_only),
            "target_k_indices": list(kptlist),
            "target_kpts": np.asarray(kpts)[list(kptlist)].tolist(),
            "nkpts": int(len(kpts)),
            "nao": int(cell.nao),
            "bands": list(bands),
        },
        "calculation": calculation,
        "memory": {
            "process": process_memory.as_dict(),
            "reference": reference_memory.as_dict(),
            "gw": gw_memory.as_dict(),
        },
    }


def _isolated_worker_command(args, package, output):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--case",
        str(args.case),
        "--kmesh",
        ",".join(str(value) for value in args.kmesh),
        "--precision",
        str(args.precision),
        "--reference-precision",
        str(args.reference_precision),
        "--aux-min-exponent",
        str(args.aux_min_exponent),
        "--metric-tol",
        str(args.metric_tol),
        "--ac-nw",
        str(args.ac_nw),
        "--workers",
        str(args.workers),
        "--stream-pair-batch-mb",
        str(args.stream_pair_batch_mb),
        "--folded-batch-mb",
        str(args.folded_batch_mb),
        "--storage",
        str(args.storage),
        "--max-memory-mb",
        str(args.max_memory_mb),
        "--isolated-worker",
        package,
        "--worker-output",
        str(output),
    ]
    if args.gamma_centered:
        command.append("--gamma-centered")
    if args.gamma_only:
        command.append("--gamma-only")
    if args.pyscf_metric_eig:
        command.append("--pyscf-metric-eig")
    if not args.finite_size:
        command.append("--no-finite-size")
    return command


def _isolated_benchmark(args):
    rows = {}
    with tempfile.TemporaryDirectory(prefix="pyqed-pbc-gw-memory-") as directory:
        for package in ("pyqed", "pyscf"):
            output = Path(directory) / f"{package}.json"
            print(f"Starting isolated {package} benchmark...", flush=True)
            completed = subprocess.run(
                _isolated_worker_command(args, package, output),
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Isolated {package} benchmark failed:\n"
                    f"{completed.stdout}{completed.stderr}"
                )
            rows[package] = json.loads(output.read_text())

    pyqed = rows["pyqed"]
    pyscf = rows["pyscf"]
    if pyqed["metadata"] != pyscf["metadata"]:
        raise RuntimeError("Isolated workers returned inconsistent benchmark metadata.")
    metadata = pyqed["metadata"]
    native_qp = np.asarray(pyqed["calculation"]["qp_Ha"], dtype=float)
    pyscf_qp = np.asarray(pyscf["calculation"]["qp_Ha"], dtype=float)
    native_mo = np.asarray(pyqed["calculation"]["mo_energy_Ha"], dtype=float)
    pyscf_mo = np.asarray(pyscf["calculation"]["mo_energy_Ha"], dtype=float)
    qp_delta = native_qp - pyscf_qp
    mf_delta = native_mo - pyscf_mo
    native_energy = float(pyqed["calculation"]["scf_energy_Ha"])
    pyscf_energy = float(pyscf["calculation"]["scf_energy_Ha"])
    pyqed_memory = pyqed["memory"]
    pyscf_memory = pyscf["memory"]

    return {
        **metadata,
        "ac_nw": int(args.ac_nw),
        "finite_size_correction": bool(args.finite_size),
        "precision": float(args.precision),
        "reference_precision": float(args.reference_precision),
        "aux_min_exponent": float(args.aux_min_exponent),
        "metric_tol": float(args.metric_tol),
        "stream_pair_batch_mb": float(args.stream_pair_batch_mb),
        "folded_batch_mb": float(args.folded_batch_mb),
        "storage": str(args.storage),
        "max_memory_mb": float(args.max_memory_mb),
        "pyscf_metric_eig": bool(args.pyscf_metric_eig),
        "pyscf_version": pyscf["package_version"],
        "memory": {
            "isolated_processes": True,
            "pyqed_process": pyqed_memory["process"],
            "pyscf_process": pyscf_memory["process"],
            "pyqed_reference": pyqed_memory["reference"],
            "pyscf_reference": pyscf_memory["reference"],
            "pyqed_gw": pyqed_memory["gw"],
            "pyscf_gw": pyscf_memory["gw"],
            "process_peak_rss_mb": float(
                max(
                    pyqed_memory["process"]["peak_rss_mb"],
                    pyscf_memory["process"]["peak_rss_mb"],
                )
            ),
        },
        "pyqed": pyqed["calculation"],
        "pyscf": pyscf["calculation"],
        "comparison": {
            "scf_total_energy_error_Ha": native_energy - pyscf_energy,
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


def benchmark(args):
    import pyscf

    case = replace(
        CASES[args.case],
        kmesh=args.kmesh,
        gamma_centered=bool(args.gamma_centered),
    )
    bands = None

    with _PeakRSSSampler() as pyscf_reference_memory:
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
    with _PeakRSSSampler() as pyqed_reference_memory:
        (
            cell,
            kpts,
            native_mf,
            native_df_seconds,
            native_scf_seconds,
        ) = _pyqed_krhf(
            case,
            precision=args.precision,
            aux_min_exponent=args.aux_min_exponent,
            metric_tol=args.metric_tol,
            workers=args.workers,
            stream_pair_batch_mb=args.stream_pair_batch_mb,
            folded_batch_mb=args.folded_batch_mb,
            storage=args.storage,
            max_memory_mb=args.max_memory_mb,
        )
    nocc = cell.nelectron // 2
    bands = (max(0, nocc - 1), min(cell.nao - 1, nocc))
    kptlist = (
        (_gamma_index(kpts),)
        if args.gamma_only
        else tuple(range(len(kpts)))
    )

    native_space = KPointTransitionSpace(native_mf, qpts="mesh")
    with _PeakRSSSampler() as pyqed_gw_memory:
        native_gw, native_gw_seconds = _run_pyqed_gw(
            native_space,
            component=GDF,
            bands=bands,
            kptlist=kptlist,
            ac_nw=args.ac_nw,
            finite_size=args.finite_size,
            head_method="builtin_gradient",
        )
    with _PeakRSSSampler() as pyscf_gw_memory:
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
        "stream_pair_batch_mb": float(args.stream_pair_batch_mb),
        "folded_batch_mb": float(args.folded_batch_mb),
        "storage": str(args.storage),
        "max_memory_mb": float(args.max_memory_mb),
        "pyscf_metric_eig": bool(args.pyscf_metric_eig),
        "pyscf_version": str(pyscf.__version__),
        "memory": {
            "pyscf_reference": pyscf_reference_memory.as_dict(),
            "pyqed_reference": pyqed_reference_memory.as_dict(),
            "pyqed_gw": pyqed_gw_memory.as_dict(),
            "pyscf_gw": pyscf_gw_memory.as_dict(),
            "process_peak_rss_mb": float(
                max(
                    pyscf_reference_memory.peak_rss_mb,
                    pyqed_reference_memory.peak_rss_mb,
                    pyqed_gw_memory.peak_rss_mb,
                    pyscf_gw_memory.peak_rss_mb,
                )
            ),
        },
        "pyqed": {
            "scf_energy_Ha": float(native_mf.e_tot),
            "mo_energy_Ha": native_mo.tolist(),
            "qp_Ha": native_qp.tolist(),
            "gdf_seconds": float(native_df_seconds),
            "scf_seconds": float(native_scf_seconds),
            "gw_seconds": float(native_gw_seconds),
            "finite_size_method": native_gw.info.get("finite_size_method"),
            "ac_profile": native_gw.info.get("ac_profile", {}),
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
    parser.add_argument("--stream-pair-batch-mb", type=float, default=128.0)
    parser.add_argument("--folded-batch-mb", type=float, default=128.0)
    parser.add_argument(
        "--storage",
        choices=("auto", "memory", "disk"),
        default="auto",
    )
    parser.add_argument("--max-memory-mb", type=float, default=512.0)
    parser.add_argument("--no-finite-size", dest="finite_size", action="store_false")
    parser.add_argument("--skip-aligned", action="store_true")
    parser.add_argument(
        "--isolated-memory",
        action="store_true",
        help="run PyQED and PySCF in separate sequential processes",
    )
    parser.add_argument(
        "--isolated-worker",
        choices=("pyqed", "pyscf"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
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

    if args.isolated_worker is not None:
        if args.worker_output is None:
            parser.error("--isolated-worker requires --worker-output")
        payload = _isolated_package_benchmark(args, args.isolated_worker)
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(json.dumps(payload, indent=2) + "\n")
        return

    payload = _isolated_benchmark(args) if args.isolated_memory else benchmark(args)
    payload["metric_tol_policy"] = (
        "explicit" if metric_tol_was_explicit else "precision_auto"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    png, pdf = _plot(payload, args.figure)
    performance_png, performance_pdf = _plot_performance(
        payload,
        args.figure.with_name(args.figure.name + "_performance"),
    )
    summary = {
        "output": str(args.output),
        "figure_png": str(png),
        "figure_pdf": str(pdf),
        "performance_figure_png": str(performance_png),
        "performance_figure_pdf": str(performance_pdf),
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
