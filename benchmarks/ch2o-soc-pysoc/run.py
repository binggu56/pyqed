#!/usr/bin/env python3
"""Benchmark PyQED one-electron SOC AO integrals against PySOC/MolSOC.

This is a matched operator-level comparison.  It intentionally does not
compare PyQED CASCI state couplings with PySOC LR-TDDFT/TD-DFTB state
couplings, because those wavefunction models are not equivalent.

Reference
---------
X. Gao, S. Bai, D. Fazzi, T. Niehaus, M. Barbatti, and W. Thiel,
J. Chem. Theory Comput. 13, 515-524 (2017), DOI: 10.1021/acs.jctc.6b00915.

PySOC uses MolSOC for the AO spin-orbit integrals.  The fitted TD-DFTB basis
is stored in an unnormalized primitive convention.  Libcint normalizes
primitives and contractions, so this script removes the primitive factors
when constructing the basis and transforms the resulting normalized AO
matrix back to PySOC's raw fitted-basis convention before comparison.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
from pyscf import gto

import pyqed
from pyqed.qchem.soc import get_pvxp_ao


COLORS = {"x": "#0072B2", "y": "#D55E00", "z": "#009E73"}
MARKERS = {"x": "o", "y": "s", "z": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        required=True,
        help="Figure path without extension; PDF and PNG are written.",
    )
    return parser.parse_args()


def _basis_for_libcint(reference: dict) -> dict[str, list]:
    basis = {}
    for element, shells in reference["basis"].items():
        converted = []
        for shell in shells:
            angular_momentum = int(shell["angular_momentum"])
            primitives = [
                [float(exponent), float(coefficient) / gto.gto_norm(angular_momentum, exponent)]
                for exponent, coefficient in shell["primitives"]
            ]
            converted.append([angular_momentum, *primitives])
        basis[element] = converted
    return basis


def _build_molecule(config: dict, reference: dict):
    atoms = [
        [entry["element"], tuple(float(value) for value in entry["coordinates"])]
        for entry in config["system"]["atoms"]
    ]
    return gto.M(
        atom=atoms,
        basis=_basis_for_libcint(reference),
        unit=config["system"]["unit"],
        charge=int(config["system"]["charge"]),
        spin=int(config["system"]["spin"]),
        cart=False,
        verbose=0,
    )


def _component_metrics(reference: np.ndarray, calculated: np.ndarray, floor: float) -> dict:
    residual = calculated - reference
    mask = np.abs(reference) > floor
    return {
        "max_reference_abs": float(np.max(np.abs(reference))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "rms_abs_error": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
        "relative_frobenius_error": float(np.linalg.norm(residual) / np.linalg.norm(reference)),
        "max_relative_error_above_floor": float(
            np.max(np.abs(residual[mask]) / np.abs(reference[mask]))
        ),
        "reference_floor": float(floor),
        "points_above_floor": int(np.count_nonzero(mask)),
    }


def _plot(reference: np.ndarray, calculated: np.ndarray, output_prefix: Path) -> None:
    try:
        import ultraplot as uplt
    except ImportError as exc:
        raise SystemExit("UltraPlot is required to generate the benchmark figure.") from exc

    uplt.rc.update(
        {
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.8,
            "legend.fontsize": 8.5,
            "tick.labelsize": 8.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axs = uplt.subplots(ncols=2, refwidth=3.15, refheight=2.55, share=False, wspace=1.35)
    components = ("x", "y", "z")

    limit = 1.06 * max(float(np.max(np.abs(reference))), float(np.max(np.abs(calculated))))
    for index, component in enumerate(components):
        axs[0].scatter(
            reference[index].ravel(),
            calculated[index].ravel(),
            s=15,
            marker=MARKERS[component],
            color=COLORS[component],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
            label=rf"${component}$",
        )
    axs[0].plot([-limit, limit], [-limit, limit], color="0.25", lw=1.0, ls="--", label="identity")
    axs[0].format(
        xlabel="PySOC/MolSOC AO integral",
        ylabel="PyQED AO integral (sign-aligned)",
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        aspect="equal",
        title="AO integral agreement",
        grid=False,
    )
    axs[0].legend(loc="upper left", ncols=1, frame=False)
    axs[0].text(0.0, 1.06, "a", transform=axs[0].transAxes, fontweight="bold", fontsize=10)

    for index, component in enumerate(components):
        ref_abs = np.abs(reference[index].ravel())
        err_abs = np.abs(calculated[index].ravel() - reference[index].ravel())
        mask = ref_abs > 1.0e-6
        axs[1].scatter(
            ref_abs[mask],
            np.maximum(err_abs[mask], 1.0e-12),
            s=15,
            marker=MARKERS[component],
            color=COLORS[component],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
        )
    axs[1].format(
        xlabel=r"$|\mathrm{PySOC/MolSOC}|$",
        ylabel="Absolute residual",
        xscale="log",
        yscale="log",
        xformatter="sci",
        yformatter="sci",
        title="Residual by integral magnitude",
        grid=False,
    )
    axs[1].text(0.0, 1.06, "b", transform=axs[1].transAxes, fontweight="bold", fontsize=10)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    config = json.loads(args.input.read_text())
    reference_data = json.loads(args.reference.read_text())
    molecule = _build_molecule(config, reference_data)

    reference_overlap = np.asarray(reference_data["raw_overlap"], dtype=float)
    reference_soc = np.asarray(reference_data["raw_soc_one"], dtype=float)
    if reference_soc.shape != (3, molecule.nao_nr(), molecule.nao_nr()):
        raise ValueError("PySOC reference SOC tensor has an unexpected shape.")

    normalized_overlap = reference_overlap / np.sqrt(
        np.outer(np.diag(reference_overlap), np.diag(reference_overlap))
    )
    overlap_error = float(np.max(np.abs(molecule.intor("int1e_ovlp") - normalized_overlap)))

    pyqed_normalized = get_pvxp_ao(molecule, one_center=False)
    raw_norms = np.sqrt(np.diag(reference_overlap))
    pyqed_raw = pyqed_normalized * np.outer(raw_norms, raw_norms)[None, :, :]
    sign_alignment = float(config["method"]["sign_alignment"])
    pyqed_aligned = sign_alignment * pyqed_raw

    floor = float(config["method"]["reference_magnitude_floor"])
    metrics = _component_metrics(reference_soc, pyqed_aligned, floor)
    per_component = {
        component: _component_metrics(reference_soc[index], pyqed_aligned[index], floor)
        for index, component in enumerate(config["method"]["component_order"])
    }
    tolerance = float(config["validation"]["absolute_tolerance"])
    passed = metrics["relative_frobenius_error"] <= tolerance

    result = {
        "benchmark_id": "ch2o-soc-pysoc",
        "input": config,
        "reference": reference_data["provenance"],
        "runtime": {
            "python_version": platform.python_version(),
            "pyqed_version": getattr(pyqed, "__version__", "unknown"),
            "numpy_version": np.__version__,
            "pyscf_version": __import__("pyscf").__version__,
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        "basis_mapping": {
            "number_of_aos": int(molecule.nao_nr()),
            "max_normalized_overlap_error": overlap_error,
        },
        "metrics": metrics,
        "component_metrics": per_component,
        "validation": {
            "quantity": config["validation"]["quantity"],
            "tolerance": tolerance,
            "observed_difference": metrics["relative_frobenius_error"],
            "passed": bool(passed),
        },
        "pysoc_raw_soc_one": reference_soc.tolist(),
        "pyqed_sign_aligned_raw_soc_one": pyqed_aligned.tolist(),
        "residual": (pyqed_aligned - reference_soc).tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    _plot(reference_soc, pyqed_aligned, args.figure_prefix)

    print(f"PySOC maximum |integral|: {metrics['max_reference_abs']:.9f}")
    print(f"Maximum absolute error:  {metrics['max_abs_error']:.9e}")
    print(f"RMS absolute error:      {metrics['rms_abs_error']:.9e}")
    print(f"Relative Frobenius error:{metrics['relative_frobenius_error']:.9e}")
    print(f"Basis overlap max error: {overlap_error:.9e}")
    print(f"Validation: {'PASS' if passed else 'FAIL'} (tolerance {tolerance:.3e})")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
