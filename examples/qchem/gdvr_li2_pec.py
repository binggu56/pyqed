#!/usr/bin/env python3
"""Ground-state GDVR RHF potential-energy curve for Li2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem.gdvr import AtomicChain, Molecule, local_ecp_terms_from_pyscf


AU_ENERGY_EV = 27.211386245988


def _parse_r_values(args):
    if args.r_values:
        values = np.asarray(args.r_values, dtype=float)
    else:
        values = np.linspace(float(args.r_min), float(args.r_max), int(args.npts))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Need at least one Li-Li distance.")
    if np.any(values <= 0.0):
        raise ValueError("Li-Li distances must be positive.")
    return values


def _build_li2_model(r_bohr, args):
    z0 = 0.5 * float(r_bohr)
    coords = [(0.0, 0.0, -z0), (0.0, 0.0, z0)]
    if args.core_model == "all-electron":
        return AtomicChain(["Li", "Li"], coords=coords)
    if args.core_model == "valence-charge":
        return Molecule([1.0, 1.0], coords=coords, nelec=2, spin=0)
    if args.core_model == "softcore-pp":
        return Molecule(
            [1.0, 1.0],
            coords=coords,
            nelec=2,
            spin=0,
            softcore_radii=[float(args.pp_rc), float(args.pp_rc)],
            basis_charges=[3.0, 3.0],
        )
    if args.core_model in ("real-ecp-local", "real-ecp-scalar", "real-ecp-semilocal"):
        scalarize_nonlocal = args.core_model == "real-ecp-scalar"
        ecp = local_ecp_terms_from_pyscf(
            "Li",
            args.ecp_name,
            scalarize_nonlocal=scalarize_nonlocal,
        )
        use_semilocal = args.core_model == "real-ecp-semilocal"
        slice_ecp = (
            local_ecp_terms_from_pyscf("Li", args.ecp_name, scalarize_nonlocal=True)
            if use_semilocal
            else ecp
        )
        valence_charge = float(3 - int(ecp["core_electrons"]))
        metadata = {
            "ecp_name": str(args.ecp_name),
            "source": "pyscf.gto.basis.load_ecp",
            "local_terms": [list(term) for term in ecp["local_terms"]],
            "semilocal_terms": [list(term) for term in ecp["semilocal_terms"]],
            "semilocal_channels": sorted({int(term[0]) for term in ecp["semilocal_terms"]}) if use_semilocal else [],
            "omitted_nonlocal_channels": [] if use_semilocal else list(ecp["omitted_nonlocal_channels"]),
            "scalarized_nonlocal_channels": list(ecp["scalarized_nonlocal_channels"]),
            "approximation": (
                "semilocal projectors"
                if use_semilocal
                else (
                    "scalar local channel only"
                    if not scalarize_nonlocal
                    else "scalarized semilocal channels"
                )
            ),
        }
        return Molecule(
            [valence_charge, valence_charge],
            coords=coords,
            nelec=2,
            spin=0,
            basis_charges=[3.0, 3.0],
            local_ecp_terms=[ecp["local_terms"], ecp["local_terms"]],
            semilocal_ecp_terms=(
                [ecp["semilocal_terms"], ecp["semilocal_terms"]]
                if use_semilocal
                else None
            ),
            slice_local_ecp_terms=(
                [slice_ecp["local_terms"], slice_ecp["local_terms"]]
                if use_semilocal
                else None
            ),
            ecp_metadata=metadata,
        )
    raise ValueError(f"unknown core model {args.core_model!r}")


def _run_point(r_bohr, args):
    if bool(args.transverse_opt) and int(args.m) != 1:
        raise ValueError("Transverse orbital optimization currently requires --m 1.")

    mol = _build_li2_model(r_bohr, args)
    build_start = time.perf_counter()
    mol.build(
        Lz=float(args.lz),
        Nz=int(args.nz),
        M=int(args.m),
        transverse_basis=args.transverse_basis,
        verbose=bool(args.verbose),
        dvr_method=args.dvr_method,
    )
    build_seconds = time.perf_counter() - build_start

    scf_start = time.perf_counter()
    mf = mol.RHF().run(
        conv=float(args.scf_conv),
        max_iter=int(args.scf_max_iter),
        verbose=bool(args.verbose),
    )
    scf_seconds = time.perf_counter() - scf_start
    e_before_transverse_opt = None
    transverse_opt_seconds = 0.0

    if bool(args.transverse_opt):
        e_before_transverse_opt = float(mf.e_tot)
        opt_start = time.perf_counter()
        mf.newton(
            tol=float(args.transverse_opt_tol),
            max_cycles=int(args.transverse_opt_cycles),
            sweep_iterations=int(args.transverse_opt_sweeps),
            ridge=float(args.transverse_opt_ridge),
            trust_step=float(args.transverse_opt_step),
            trust_radius=float(args.transverse_opt_radius),
            scf_conv=float(args.scf_conv),
            scf_max_iter=int(args.scf_max_iter),
            verbose=bool(args.verbose),
        )
        transverse_opt_seconds = time.perf_counter() - opt_start

    return {
        "R_bohr": float(r_bohr),
        "E_RHF_Ha": float(mf.e_tot),
        "E_before_transverse_opt_Ha": e_before_transverse_opt,
        "E_nuc_Ha": float(mol.nuclear_repulsion_energy()),
        "HOMO_Ha": float(mf.mo_energy[int(mol.nelec // 2) - 1]),
        "LUMO_Ha": float(mf.mo_energy[int(mol.nelec // 2)]),
        "gap_Ha": float(mf.mo_energy[int(mol.nelec // 2)] - mf.mo_energy[int(mol.nelec // 2) - 1]),
        "nelec": int(mol.nelec),
        "core_model": args.core_model,
        "ecp_name": mol.ecp_metadata.get("ecp_name", ""),
        "ecp_omitted_nonlocal_channels": " ".join(
            str(x) for x in mol.ecp_metadata.get("omitted_nonlocal_channels", [])
        ),
        "ecp_scalarized_nonlocal_channels": " ".join(
            str(x) for x in mol.ecp_metadata.get("scalarized_nonlocal_channels", [])
        ),
        "build_seconds": float(build_seconds),
        "scf_seconds": float(scf_seconds),
        "transverse_opt_seconds": float(transverse_opt_seconds),
        "transverse_opt_cycles": int(mf.info.get("newton_cycles", 0)) if bool(args.transverse_opt) else 0,
        "transverse_opt_converged": bool(mf.info.get("newton_converged", False)) if bool(args.transverse_opt) else False,
        "transverse_opt_history_Ha": [
            float(x) for x in mf.info.get("newton_energy_history", [])
        ] if bool(args.transverse_opt) else [],
        "scf_cycles": int(mf.info.get("cycles", -1)) if mf.info is not None else -1,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-values", type=float, nargs="*", default=None)
    parser.add_argument("--r-min", type=float, default=3.5)
    parser.add_argument("--r-max", type=float, default=8.0)
    parser.add_argument("--npts", type=int, default=7)
    parser.add_argument("--lz", type=float, default=8.0)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument(
        "--core-model",
        choices=(
            "all-electron",
            "valence-charge",
            "softcore-pp",
            "real-ecp-local",
            "real-ecp-scalar",
            "real-ecp-semilocal",
        ),
        default="real-ecp-scalar",
        help=(
            "all-electron uses Z=3 Li nuclei and 6 electrons; valence-charge "
            "uses a crude +1/+1 two-electron Coulomb proxy; softcore-pp uses "
            "+1/+1, two electrons, Li transverse exponents, and a smoothed "
            "local core potential; real-ecp-local uses the local channel from "
            "a named PySCF Li ECP and omits semilocal projectors; "
            "real-ecp-scalar additionally folds semilocal radial terms into "
            "the scalar GDVR potential; real-ecp-semilocal keeps semilocal "
            "projectors as nonlocal one-electron blocks."
        ),
    )
    parser.add_argument("--pp-rc", type=float, default=0.75, help="Soft-core PP radius in bohr.")
    parser.add_argument("--ecp-name", default="bfd", help="PySCF ECP name for real-ECP core models.")
    parser.add_argument("--transverse-basis", default="sto3g")
    parser.add_argument("--dvr-method", choices=("sine", "exp", "sinc"), default="exp")
    parser.add_argument("--scf-conv", type=float, default=1.0e-8)
    parser.add_argument("--scf-max-iter", type=int, default=100)
    parser.add_argument("--transverse-opt", action="store_true", help="Optimize the M=1 transverse slice orbital after RHF.")
    parser.add_argument("--transverse-opt-cycles", type=int, default=20)
    parser.add_argument("--transverse-opt-sweeps", type=int, default=2)
    parser.add_argument("--transverse-opt-tol", type=float, default=1.0e-8)
    parser.add_argument("--transverse-opt-ridge", type=float, default=0.5)
    parser.add_argument("--transverse-opt-step", type=float, default=0.5)
    parser.add_argument("--transverse-opt-radius", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("/private/tmp/gdvr_li2_pec.png"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    r_values = _parse_r_values(args)

    rows = []
    total_start = time.perf_counter()
    for idx, r_bohr in enumerate(r_values, start=1):
        print(f"[{idx}/{len(r_values)}] R = {r_bohr:.6f} bohr", flush=True)
        row = _run_point(r_bohr, args)
        rows.append(row)
        print(
            "    E = {E_RHF_Ha:.12f} Ha  gap = {gap_Ha:.6f} Ha  "
            "build = {build_seconds:.2f}s  scf = {scf_seconds:.2f}s".format(**row),
            flush=True,
        )

    energies = np.asarray([row["E_RHF_Ha"] for row in rows], dtype=float)
    min_idx = int(np.argmin(energies))
    for row in rows:
        row["E_relative_eV"] = float((row["E_RHF_Ha"] - energies[min_idx]) * AU_ENERGY_EV)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.csv if args.csv is not None else args.out.with_suffix(".csv")
    json_path = args.json if args.json is not None else args.out.with_suffix(".summary.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "R_bohr",
        "E_RHF_Ha",
        "E_before_transverse_opt_Ha",
        "E_relative_eV",
        "E_nuc_Ha",
        "HOMO_Ha",
        "LUMO_Ha",
        "gap_Ha",
        "core_model",
        "ecp_name",
        "ecp_omitted_nonlocal_channels",
        "ecp_scalarized_nonlocal_channels",
        "build_seconds",
        "scf_seconds",
        "transverse_opt_seconds",
        "transverse_opt_cycles",
        "transverse_opt_converged",
        "scf_cycles",
    ]
    with open(csv_path, "w") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in header) + "\n")

    r = np.asarray([row["R_bohr"] for row in rows], dtype=float)
    rel_ev = np.asarray([row["E_relative_eV"] for row in rows], dtype=float)
    gaps = np.asarray([row["gap_Ha"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.5), sharex=True)
    axes[0].plot(r, energies, marker="o", lw=1.8)
    axes[0].axvline(r[min_idx], color="0.4", ls="--", lw=1.0)
    axes[0].set_ylabel("RHF energy (Ha)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(r, rel_ev, marker="o", lw=1.8, label="relative energy")
    axes[1].plot(r, gaps * AU_ENERGY_EV, marker="s", lw=1.2, label="HOMO-LUMO gap")
    axes[1].axvline(r[min_idx], color="0.4", ls="--", lw=1.0)
    axes[1].set_xlabel("Li-Li distance (bohr)")
    axes[1].set_ylabel("energy (eV)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(
        "Li2 GDVR RHF PEC, "
        f"Nz={args.nz}, M={args.m}, Lz={args.lz:g}, {args.transverse_basis}, "
        f"{args.core_model}"
        + (f" ({args.ecp_name})" if args.core_model.startswith("real-ecp") else "")
    )
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "grid": {
            "Lz_bohr": float(args.lz),
            "Nz": int(args.nz),
            "M": int(args.m),
            "transverse_basis": args.transverse_basis,
            "dvr_method": args.dvr_method,
        },
        "transverse_opt": {
            "enabled": bool(args.transverse_opt),
            "cycles": int(args.transverse_opt_cycles),
            "sweeps": int(args.transverse_opt_sweeps),
            "tol": float(args.transverse_opt_tol),
            "ridge": float(args.transverse_opt_ridge),
            "step": float(args.transverse_opt_step),
            "radius": float(args.transverse_opt_radius),
        },
        "core_model": args.core_model,
        "pp_rc_bohr": None if args.core_model != "softcore-pp" else float(args.pp_rc),
        "ecp": None
        if not args.core_model.startswith("real-ecp")
        else {
            "name": str(args.ecp_name),
            "source": "pyscf.gto.basis.load_ecp",
            "approximation": (
                "scalar local channel only; semilocal projectors are omitted"
                if args.core_model == "real-ecp-local"
                else (
                    "semilocal projectors retained as nonlocal GDVR blocks"
                    if args.core_model == "real-ecp-semilocal"
                    else "semilocal radial terms folded into the scalar GDVR potential"
                )
            ),
        },
        "scan": rows,
        "minimum_sampled": {
            "R_bohr": float(r[min_idx]),
            "E_RHF_Ha": float(energies[min_idx]),
            "index": int(min_idx),
        },
        "timing_seconds": float(time.perf_counter() - total_start),
        "files": {
            "plot_png": str(args.out),
            "csv": str(csv_path),
            "summary_json": str(json_path),
        },
    }
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Minimum sampled R: {r[min_idx]:.6f} bohr")
    print(f"Minimum sampled E: {energies[min_idx]:.12f} Ha")
    print(f"Saved figure:      {args.out}")
    print(f"Saved CSV:         {csv_path}")
    print(f"Saved summary:     {json_path}")


if __name__ == "__main__":
    main()
