#!/usr/bin/env python3
"""Stress-validate PyQED CASPT2 against OpenMolcas.

The matrix covers a third basis, excited roots, MS/XMS state interaction,
real and imaginary shifts, and a near-degenerate H4 rectangle-to-square scan.
All comparisons use fixed-orbital singlet CASCI references, full internal
contraction, no frozen orbitals, and zero IPEA shift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from pyqed.qchem import CASCI, MSCASPT2, XMSCASPT2, Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.caspt2 import CASPT2


FLOAT = r"[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?"
SS_REFERENCE_MATCH_TOL = 2.0e-7
FINAL_BLOCK = re.compile(
    r"FINAL\s+CASPT2\s+RESULT:(.*?)(?=\n\s*\+\+\s+Denominators|\Z)",
    re.IGNORECASE | re.DOTALL,
)
MS_ENERGY = re.compile(
    rf"::\s+(MS|XMS)-CASPT2\s+Root\s+(\d+)\s+Total\s+energy:\s*({FLOAT})",
    re.IGNORECASE,
)
RASSCF_ENERGY = re.compile(
    rf"::\s+RASSCF\s+root\s+number\s+(\d+)\s+Total\s+energy:\s*({FLOAT})",
    re.IGNORECASE,
)


def _number(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    return None if match is None else float(match.group(1).replace("D", "E").replace("d", "e"))


def parse_openmolcas(text):
    roots = []
    for block in FINAL_BLOCK.findall(text):
        roots.append(
            {
                "reference_energy_hartree": _number(rf"Reference\s+energy:\s*({FLOAT})", block),
                "e_corr_nonvariational_hartree": _number(
                    rf"E2\s*\(Non-variational\):\s*({FLOAT})", block
                ),
                "shift_correction_hartree": _number(
                    rf"Shift\s+correction:\s*({FLOAT})", block
                ),
                "e_corr_hartree": _number(rf"E2\s*\(Variational\):\s*({FLOAT})", block),
                "total_energy_hartree": _number(rf"Total\s+energy:\s*({FLOAT})", block),
                "reference_weight": _number(rf"Reference\s+weight:\s*({FLOAT})", block),
                "residual_norm": _number(rf"Residual\s+norm:\s*({FLOAT})", block),
            }
        )
    multistate = {}
    for variant, root, energy in MS_ENERGY.findall(text):
        multistate.setdefault(variant.lower(), []).append(
            (int(root), float(energy.replace("D", "E").replace("d", "e")))
        )
    return {
        "roots": roots,
        "rasscf_energies_hartree": [
            float(value.replace("D", "E").replace("d", "e"))
            for _root, value in RASSCF_ENERGY.findall(text)
        ],
        "multistate_energies_hartree": {
            key: [energy for _root, energy in sorted(values)]
            for key, values in multistate.items()
        },
    }


def build_reference(atom, basis, ncas, nelecas, nstates):
    mol = Molecule(atom=atom, unit="angstrom", basis=basis)
    mol.build()
    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=ncas, nelecas=nelecas).run(
        nstates=nstates,
        method="direct_spin0",
    )
    return mol, mc


def atom_records(atom):
    records = []
    for item in atom.split(";"):
        symbol, x, y, z = item.split()
        records.append((symbol, float(x), float(y), float(z)))
    return records


def write_openmolcas_input(
    directory,
    name,
    atom,
    basis,
    ncas,
    nelecas,
    inactive,
    nroots,
    variant="ss",
    root=1,
    real_shift=0.0,
    imaginary_shift=0.0,
):
    directory.mkdir(parents=True, exist_ok=True)
    xyz = directory / f"{name}.xyz"
    records = atom_records(atom)
    xyz.write_text(
        "\n".join(
            [str(len(records)), name]
            + [f"{symbol} {x:.12f} {y:.12f} {z:.12f}" for symbol, x, y, z in records]
        )
        + "\n",
        encoding="utf-8",
    )
    if variant == "ss":
        state_line = f"MULTISTATE = 1 {root}"
    elif variant == "ms":
        state_line = "MULTISTATE = " + str(nroots) + " " + " ".join(
            str(index) for index in range(1, nroots + 1)
        )
    elif variant == "xms":
        state_line = "XMULTISTATE = " + str(nroots) + " " + " ".join(
            str(index) for index in range(1, nroots + 1)
        )
    else:
        raise ValueError(f"Unknown variant {variant!r}.")
    shift_lines = []
    if real_shift:
        shift_lines.append(f"SHIFT = {real_shift:.12g}")
    if imaginary_shift:
        shift_lines.append(f"IMAGINARY = {imaginary_shift:.12g}")
    path = directory / f"{name}_{variant}_{time.time_ns()}.input"
    path.write_text(
        "\n".join(
            [
                "&GATEWAY",
                f"Coord = {xyz.name}",
                f"Basis = {basis}",
                "Group = C1",
                "",
                "&SEWARD",
                "",
                "&SCF",
                "Charge = 0",
                "Spin = 1",
                "",
                "&RASSCF",
                "Spin = 1",
                f"Nactel = {nelecas} 0 0",
                f"Inactive = {inactive}",
                f"Ras2 = {ncas}",
                f"CIRoot = {nroots} {nroots} 1",
                "CIONLY",
                "",
                "&CASPT2",
                "Frozen = 0",
                "IPEA = 0.0",
                state_line,
                *shift_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def run_openmolcas(command, path, timeout):
    start = time.perf_counter()
    proc = subprocess.run(
        [command, path.name],
        cwd=path.parent,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - start
    output = proc.stdout + "\n" + proc.stderr
    log = path.with_suffix(".log")
    log.write_text(output, encoding="utf-8")
    parsed = parse_openmolcas(output)
    parsed.update(
        status="ok" if proc.returncode == 0 else "failed",
        returncode=proc.returncode,
        elapsed_s=elapsed,
        input_path=str(path),
        log_path=str(log),
    )
    if proc.returncode:
        raise RuntimeError(f"OpenMolcas failed for {path}; see {log}")
    return parsed


def pyqed_ss(mc, root=0, real_shift=0.0, imaginary_shift=0.0):
    pt = CASPT2(
        mc,
        root=root,
        real_shift=real_shift,
        imaginary_shift=imaginary_shift,
    )
    start = time.perf_counter()
    pt.run()
    return {
        "reference_energy_hartree": float(np.asarray(mc.e_tot)[root]),
        "e_corr_nonvariational_hartree": float(pt.e_corr_nonvariational),
        "shift_correction_hartree": float(pt.shift_correction),
        "e_corr_hartree": float(pt.e_corr),
        "total_energy_hartree": float(pt.e_tot),
        "reference_weight": float(pt.reference_weight),
        "residual_norm": float(pt.contracted_residual_norm),
        "contracted_rank": int(pt.contracted_basis_rank),
        "external_determinants": len(pt.external_determinants),
        "elapsed_s": time.perf_counter() - start,
    }


def matched_ss(pyqed, openmolcas):
    external = openmolcas["roots"][-1]
    reference_delta = (
        pyqed["reference_energy_hartree"] - external["reference_energy_hartree"]
    )
    return {
        "pyqed": pyqed,
        "openmolcas": external,
        "reference_delta_hartree": reference_delta,
        "reference_matched": abs(reference_delta) <= SS_REFERENCE_MATCH_TOL,
        "comparison_valid": abs(reference_delta) <= SS_REFERENCE_MATCH_TOL,
        "total_delta_hartree": pyqed["total_energy_hartree"]
        - external["total_energy_hartree"],
        "correction_delta_hartree": pyqed["e_corr_hartree"]
        - external["e_corr_hartree"],
    }


def lih_validation(args, work):
    atom = "Li 0 0 0; H 0 0 1.6"
    mol, mc = build_reference(atom, "sto-3g", 2, 2, 2)
    del mol
    inactive = 1
    excited = []
    for root in range(2):
        path = write_openmolcas_input(
            work / "lih_excited", f"lih_root{root + 1}", atom, "STO-3G", 2, 2,
            inactive, 2, root=root + 1,
        )
        excited.append(
            matched_ss(
                pyqed_ss(mc, root=root),
                run_openmolcas(args.openmolcas_command, path, args.timeout),
            )
        )

    multi = {}
    for variant, cls in (("ms", MSCASPT2), ("xms", XMSCASPT2)):
        driver = cls(mc, roots=(0, 1))
        start = time.perf_counter()
        energies = np.asarray(driver.run(), dtype=float)
        path = write_openmolcas_input(
            work / "lih_multistate", f"lih_{variant}", atom, "STO-3G", 2, 2,
            inactive, 2, variant=variant,
        )
        external = run_openmolcas(args.openmolcas_command, path, args.timeout)
        external_energies = np.asarray(
            external["multistate_energies_hartree"][variant], dtype=float
        )
        multi[variant] = {
            "pyqed_energies_hartree": energies.tolist(),
            "openmolcas_energies_hartree": external_energies.tolist(),
            "delta_hartree": (energies - external_energies).tolist(),
            "max_abs_delta_hartree": float(np.max(np.abs(energies - external_energies))),
            "effective_hamiltonian_hartree": driver.effective_hamiltonian.tolist(),
            "elapsed_s": time.perf_counter() - start,
            "openmolcas": external,
        }

    shifts = []
    for root in range(2):
        for kind, values in (
            ("real", (0.05, 0.10, 0.20)),
            ("imaginary", (0.05, 0.10, 0.20)),
        ):
            for value in values:
                options = {
                    "real_shift": value if kind == "real" else 0.0,
                    "imaginary_shift": value if kind == "imaginary" else 0.0,
                }
                path = write_openmolcas_input(
                    work / "lih_shifts",
                    f"lih_root{root + 1}_{kind}_{value:g}",
                    atom,
                    "STO-3G",
                    2,
                    2,
                    inactive,
                    2,
                    root=root + 1,
                    real_shift=options["real_shift"],
                    imaginary_shift=options["imaginary_shift"],
                )
                record = matched_ss(
                    pyqed_ss(mc, root=root, **options),
                    run_openmolcas(args.openmolcas_command, path, args.timeout),
                )
                record.update(kind=kind, root=root, shift_hartree=value)
                shifts.append(record)
    return {"excited_roots": excited, "multistate": multi, "shifts": shifts}


def third_basis_validation(args, work):
    atom = "Li 0 0 0; H 0 0 1.6"
    _mol, mc = build_reference(atom, "6-31g", 2, 2, 1)
    path = write_openmolcas_input(
        work / "lih_631g", "lih_631g", atom, "6-31G", 2, 2, 1, 1
    )
    return matched_ss(
        pyqed_ss(mc),
        run_openmolcas(args.openmolcas_command, path, args.timeout),
    )


def intruder_validation(args, work):
    atom = "Li 0 0 0; H 0 0 5.0"
    _mol, mc = build_reference(atom, "sto-3g", 2, 2, 2)
    try:
        pyqed_ss(mc, root=1)
    except ZeroDivisionError as exc:
        unshifted = {"status": "intruder_detected", "message": str(exc)}
    else:
        unshifted = {"status": "unexpected_success"}
    shifts = []
    for kind, values in (
        ("real", (0.05, 0.10, 0.20)),
        ("imaginary", (0.05, 0.10, 0.20)),
    ):
        for value in values:
            options = {
                "real_shift": value if kind == "real" else 0.0,
                "imaginary_shift": value if kind == "imaginary" else 0.0,
            }
            path = write_openmolcas_input(
                work / "lih_intruder",
                f"lih_5p0_root2_{kind}_{value:g}",
                atom,
                "STO-3G",
                2,
                2,
                1,
                2,
                root=2,
                real_shift=options["real_shift"],
                imaginary_shift=options["imaginary_shift"],
            )
            record = matched_ss(
                pyqed_ss(mc, root=1, **options),
                run_openmolcas(args.openmolcas_command, path, args.timeout),
            )
            record.update(kind=kind, shift_hartree=value)
            shifts.append(record)
    return {
        "molecule": "LiH",
        "bond_angstrom": 5.0,
        "root": 1,
        "unshifted": unshifted,
        "shifts": shifts,
        "external_comparison_valid": all(item["comparison_valid"] for item in shifts),
    }


def h4_atom(aspect, side=2.5):
    half_x = side / 2.0
    half_y = side * aspect / 2.0
    return (
        f"H {-half_x} {-half_y} 0; H {half_x} {-half_y} 0; "
        f"H {half_x} {half_y} 0; H {-half_x} {half_y} 0"
    )


def h4_scan(args, work):
    records = []
    for aspect in (0.80, 0.90, 0.97, 1.00, 1.03, 1.10, 1.20):
        atom = h4_atom(aspect, side=args.h4_side)
        _mol, mc = build_reference(atom, "6-31g", 4, 4, 2)
        xms = XMSCASPT2(mc, roots=(0, 1))
        start = time.perf_counter()
        pyqed_energies = np.asarray(xms.run(), dtype=float)
        pyqed_elapsed = time.perf_counter() - start
        path = write_openmolcas_input(
            work / "h4_xms_scan", f"h4_{aspect:.2f}", atom, "6-31G", 4, 4,
            0, 2, variant="xms",
        )
        external = run_openmolcas(args.openmolcas_command, path, args.timeout)
        openmolcas_energies = np.asarray(
            external["multistate_energies_hartree"]["xms"], dtype=float
        )
        reference_delta = np.asarray(mc.e_tot, dtype=float) - np.asarray(
            external["rasscf_energies_hartree"], dtype=float
        )
        records.append(
            {
                "aspect_ratio": aspect,
                "side_angstrom": args.h4_side,
                "pyqed_reference_energies_hartree": np.asarray(mc.e_tot).tolist(),
                "openmolcas_reference_energies_hartree": external[
                    "rasscf_energies_hartree"
                ],
                "reference_delta_hartree": reference_delta.tolist(),
                "reference_matched": bool(np.max(np.abs(reference_delta)) <= 1.0e-7),
                "pyqed_xms_energies_hartree": pyqed_energies.tolist(),
                "openmolcas_xms_energies_hartree": openmolcas_energies.tolist(),
                "delta_hartree": (pyqed_energies - openmolcas_energies).tolist(),
                "pyqed_gap_hartree": float(np.diff(pyqed_energies)[0]),
                "openmolcas_gap_hartree": float(np.diff(openmolcas_energies)[0]),
                "reference_weights": [
                    root["reference_weight"] for root in external["roots"]
                ],
                "pyqed_elapsed_s": pyqed_elapsed,
                "openmolcas": external,
            }
        )
    return records


def plot_results(results, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    labels = ["LiH/6-31G", "LiH SS S0", "LiH SS S1", "LiH MS", "LiH XMS"]
    errors = [abs(results["third_basis"]["total_delta_hartree"])]
    errors.extend(abs(item["total_delta_hartree"]) for item in results["lih"]["excited_roots"])
    errors.extend(
        results["lih"]["multistate"][variant]["max_abs_delta_hartree"]
        for variant in ("ms", "xms")
    )
    axes[0].bar(labels, np.maximum(errors, 1.0e-14))
    axes[0].set_yscale("log")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylabel(r"maximum $|\Delta E|$ ($E_h$)")
    axes[0].set_title("Independent OpenMolcas agreement")

    colors = {"real": "tab:blue", "imaginary": "tab:orange"}
    for root, marker in ((0, "o"), (1, "s")):
        for kind in ("real", "imaginary"):
            selected = [
                item
                for item in results["lih"]["shifts"]
                if item["kind"] == kind and item["root"] == root
            ]
            axes[1].plot(
                [item["shift_hartree"] for item in selected],
                [abs(item["total_delta_hartree"]) for item in selected],
                marker + "-",
                label=f"{kind}, S{root}",
                color=colors[kind],
                alpha=1.0 if root == 0 else 0.65,
            )
    for kind in ("real", "imaginary"):
        selected = [
            item
            for item in results["intruder"]["shifts"]
            if item["kind"] == kind and item["comparison_valid"]
        ]
        if selected:
            axes[1].plot(
                [item["shift_hartree"] for item in selected],
                [abs(item["total_delta_hartree"]) for item in selected],
                "^--",
                label=f"{kind}, intruder",
                color=colors[kind],
                alpha=0.4,
            )
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"shift ($E_h$)")
    axes[1].set_ylabel(r"$|\Delta E|$ ($E_h$)")
    axes[1].set_title("Shifted SS-CASPT2")
    axes[1].legend(frameon=False)

    scan = results["h4_xms_scan"]
    aspect = np.array([item["aspect_ratio"] for item in scan])
    matched = np.array([item["reference_matched"] for item in scan], dtype=bool)
    pyqed_gap = np.array([item["pyqed_gap_hartree"] for item in scan], dtype=float)
    openmolcas_gap = np.array(
        [item["openmolcas_gap_hartree"] for item in scan], dtype=float
    )
    axes[2].plot(aspect, np.where(matched, pyqed_gap, np.nan), "o-", label="PyQED")
    axes[2].plot(
        aspect,
        np.where(matched, openmolcas_gap, np.nan),
        "x--", label="OpenMolcas",
    )
    if np.any(~matched):
        axes[2].scatter(
            aspect[~matched],
            openmolcas_gap[~matched],
            marker="X",
            s=70,
            color="tab:red",
            label="unmatched CAS reference",
            zorder=4,
        )
    axes[2].axvline(1.0, color="0.75", linewidth=1)
    axes[2].set_xlabel("H4 rectangle aspect ratio")
    axes[2].set_ylabel(r"XMS S1-S0 gap ($E_h$)")
    axes[2].set_title(f"Near-degenerate H4/6-31G, side={scan[0]['side_angstrom']:.1f} Å")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openmolcas-command", default="pymolcas")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--h4-side", type=float, default=2.5)
    parser.add_argument("--work-dir", default="/private/tmp/pyqed-caspt2-stress")
    parser.add_argument("--out", default="/private/tmp/pyqed-caspt2-stress.json")
    parser.add_argument("--plot", default="/private/tmp/pyqed-caspt2-stress.png")
    args = parser.parse_args()
    work = Path(args.work_dir)
    results = {
        "method": {
            "reference": "fixed-orbital singlet CASCI",
            "contraction": "fully internally contracted",
            "ipea_shift_hartree": 0.0,
            "frozen_orbitals": 0,
            "ss_reference_match_tolerance_hartree": SS_REFERENCE_MATCH_TOL,
            "h4_reference_match_tolerance_hartree": 1.0e-7,
        },
        "third_basis": third_basis_validation(args, work),
        "lih": lih_validation(args, work),
        "intruder": intruder_validation(args, work),
        "h4_xms_scan": h4_scan(args, work),
        "unsupported": [
            "nonzero IPEA shift",
            "real- or imaginary-shifted MS/XMS effective couplings",
        ],
    }
    all_deltas = [abs(results["third_basis"]["total_delta_hartree"])]
    all_deltas.extend(
        abs(item["total_delta_hartree"]) for item in results["lih"]["excited_roots"]
    )
    all_deltas.extend(
        abs(delta)
        for variant in results["lih"]["multistate"].values()
        for delta in variant["delta_hartree"]
    )
    all_deltas.extend(
        abs(delta)
        for item in results["h4_xms_scan"]
        if item["reference_matched"]
        for delta in item["delta_hartree"]
    )
    unmatched = [
        f"h4_aspect_{item['aspect_ratio']:.2f}"
        for item in results["h4_xms_scan"]
        if not item["reference_matched"]
    ]
    valid_shifted = [
        item
        for item in results["lih"]["shifts"] + results["intruder"]["shifts"]
        if item["comparison_valid"]
    ]
    valid_intruder = [
        item for item in results["intruder"]["shifts"] if item["comparison_valid"]
    ]
    results["summary"] = {
        "max_abs_energy_delta_hartree": max(all_deltas),
        "max_abs_shifted_total_delta_hartree": max(
            abs(item["total_delta_hartree"])
            for item in valid_shifted
        ),
        "max_abs_intruder_shift_delta_hartree": (
            max(abs(item["total_delta_hartree"]) for item in valid_intruder)
            if valid_intruder
            else None
        ),
        "intruder_external_comparison_valid": bool(valid_intruder),
        "minimum_h4_xms_gap_hartree": min(
            item["openmolcas_gap_hartree"]
            for item in results["h4_xms_scan"]
            if item["reference_matched"]
        ),
        "minimum_openmolcas_reference_weight": min(
            weight
            for item in results["h4_xms_scan"]
            for weight in item["reference_weights"]
            if weight is not None
        ),
        "unmatched_reference_cases": unmatched,
    }
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_results(results, args.plot)
    print(json.dumps(results["summary"], indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
