"""Diagnose direct spherical J/K accuracy for pyrazine/aug-cc-pVDZ."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian
from pyscf.lib import param

from benchmark_pyrazine_spherical_jk import ATOM
from pyqed.qchem import Molecule
from pyqed.qchem.basis import _basis_path, direct_jk_spherical_cpp
from pyqed.units import au2angstrom


def error_metrics(candidate, reference, labels):
    difference = candidate - reference
    flat_index = int(np.argmax(np.abs(difference)))
    index = tuple(int(value) for value in np.unravel_index(flat_index, difference.shape))
    reference_scale = float(np.max(np.abs(reference)))
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(difference * difference))),
        "relative_frobenius": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        "max_abs_over_reference_max": float(
            np.max(np.abs(difference)) / reference_scale
        ),
        "max_index": list(index),
        "max_labels": [labels[index[0]], labels[index[1]]],
        "candidate_at_max": float(candidate[index]),
        "reference_at_max": float(reference[index]),
        "signed_error_at_max": float(difference[index]),
    }


def add_error_image(axis, values, title):
    maximum = float(np.max(np.abs(values)))
    scale = maximum if maximum > 0.0 else 1.0
    image = axis.imshow(
        values,
        origin="lower",
        cmap="coolwarm",
        norm=mcolors.Normalize(vmin=-scale, vmax=scale),
        interpolation="nearest",
        rasterized=True,
    )
    axis.set(title=title, xlabel="AO column", ylabel="AO row")
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    axis.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--tag", default="current")
    args = parser.parse_args()

    build_start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mol = Molecule(atom=ATOM, basis="aug-cc-pvdz", unit="angstrom")
        mol.build(options={
                "coord_type": "spherical",
                "eri_representation": "direct",
                "eri_backend": "rys",
                "eri_screen_tol": 0.0,
                "parallel": True,
                "eri_workers": args.workers,
                "parallel_min_nao": 0,
            },
        )
    build_seconds = time.perf_counter() - build_start
    data = mol._builtin_direct_jk_data
    rng = np.random.default_rng(20260823)
    density = rng.normal(size=(mol.nao, mol.nao))
    density += density.T
    common = (
        data["shells"], data["origins"], data["exps"], data["weights"],
        data["nprim"], data["pair_bounds"], data["transform"], density,
    )

    rys_start = time.perf_counter()
    rys = direct_jk_spherical_cpp(
        *common,
        screen_tol=0.0,
        workers=args.workers,
        rys_max_rank=data["rys_max_rank"],
        native_plan=data["native_plan"],
        symmetric_density=True,
    )
    rys_seconds = time.perf_counter() - rys_start
    if rys is None:
        raise RuntimeError("native Rys J/K failed")

    os_start = time.perf_counter()
    os_result = direct_jk_spherical_cpp(
        *common,
        screen_tol=0.0,
        workers=args.workers,
        rys_max_rank=-1,
        native_plan=data["native_plan"],
        symmetric_density=True,
    )
    os_seconds = time.perf_counter() - os_start
    if os_result is None:
        raise RuntimeError("native Obara-Saika J/K failed")

    basis_path = _basis_path("aug-cc-pvdz")
    pyscf_basis = {
        element: parse_gaussian.load(basis_path, element)
        for element in ("H", "C", "N")
    }
    pyscf_mol = gto.M(
        atom=ATOM,
        basis=pyscf_basis,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    pyscf_start = time.perf_counter()
    pyscf_j, pyscf_k = scf.hf.get_jk(pyscf_mol, density, hermi=1)
    pyscf_seconds = time.perf_counter() - pyscf_start
    pyscf_overlap = pyscf_mol.intor_symmetric("int1e_ovlp")
    labels = list(pyscf_mol.ao_labels())
    aligned_atom = list(zip(mol.atom_symbols(), mol.atom_coords().tolist()))
    aligned_mol = gto.M(
        atom=aligned_atom,
        basis=pyscf_basis,
        unit="Bohr",
        cart=False,
        verbose=0,
    )
    aligned_j, aligned_k = scf.hf.get_jk(aligned_mol, density, hermi=1)
    aligned_overlap = aligned_mol.intor_symmetric("int1e_ovlp")

    metrics = {
        "molecule": "pyrazine",
        "basis": "aug-cc-pVDZ",
        "nao": int(mol.nao),
        "workers": args.workers,
        "screen_tol": 0.0,
        "density_max_abs": float(np.max(np.abs(density))),
        "density_frobenius": float(np.linalg.norm(density)),
        "pyqed_bohr_to_angstrom": float(au2angstrom),
        "pyscf_bohr_to_angstrom": float(param.BOHR),
        "relative_length_conversion_difference": float(
            au2angstrom / param.BOHR - 1.0
        ),
        "build_seconds": build_seconds,
        "rys_seconds": rys_seconds,
        "obara_saika_seconds": os_seconds,
        "pyscf_seconds": pyscf_seconds,
        "rys_vs_pyscf_j": error_metrics(rys[0], pyscf_j, labels),
        "rys_vs_pyscf_k": error_metrics(rys[1], pyscf_k, labels),
        "rys_vs_obara_saika_j": error_metrics(rys[0], os_result[0], labels),
        "rys_vs_obara_saika_k": error_metrics(rys[1], os_result[1], labels),
        "overlap_vs_pyscf": error_metrics(mol.overlap, pyscf_overlap, labels),
        "rys_vs_bohr_aligned_pyscf_j": error_metrics(rys[0], aligned_j, labels),
        "rys_vs_bohr_aligned_pyscf_k": error_metrics(rys[1], aligned_k, labels),
        "overlap_vs_bohr_aligned_pyscf": error_metrics(
            mol.overlap, aligned_overlap, labels
        ),
    }

    differences = {
        "Rys − PySCF J": rys[0] - pyscf_j,
        "Rys − PySCF K": rys[1] - pyscf_k,
        "PyQED − PySCF overlap": mol.overlap - pyscf_overlap,
        "Rys − Bohr-aligned PySCF J": rys[0] - aligned_j,
        "Rys − Bohr-aligned PySCF K": rys[1] - aligned_k,
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), constrained_layout=True)
    for axis, (title, values) in zip(axes.flat[:5], differences.items()):
        add_error_image(axis, values, title)
    relative_names = (
        "J: PySCF", "K: PySCF", "J: aligned", "K: aligned",
        "J: OS", "K: OS", "Overlap",
    )
    relative_values = (
        metrics["rys_vs_pyscf_j"]["relative_frobenius"],
        metrics["rys_vs_pyscf_k"]["relative_frobenius"],
        metrics["rys_vs_bohr_aligned_pyscf_j"]["relative_frobenius"],
        metrics["rys_vs_bohr_aligned_pyscf_k"]["relative_frobenius"],
        metrics["rys_vs_obara_saika_j"]["relative_frobenius"],
        metrics["rys_vs_obara_saika_k"]["relative_frobenius"],
        metrics["overlap_vs_pyscf"]["relative_frobenius"],
    )
    axes[1, 2].bar(np.arange(len(relative_names)), relative_values, color="#0072B2")
    axes[1, 2].set(
        title="Relative Frobenius errors",
        ylabel="Relative error",
        xticks=np.arange(len(relative_names)),
        xticklabels=relative_names,
        yscale="log",
    )
    axes[1, 2].tick_params(axis="x", rotation=30)
    axes[1, 2].grid(axis="y", alpha=0.25)
    axes[1, 2].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Pyrazine/aug-cc-pVDZ direct spherical J/K accuracy", fontsize=13)

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f"pyrazine_aug_cc_pvdz_accuracy_{args.tag}"
    with stem.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    fig.savefig(stem.with_suffix(".png"), dpi=360)
    fig.savefig(stem.with_suffix(".pdf"))
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
