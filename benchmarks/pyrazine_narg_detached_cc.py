"""Benchmark Abelian detached-frame plus CC NARG on pyrazine."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYQED_MPS_DISABLE_CPP_DAVIDSON", "1")

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.pyrazine_narg_overlap_casci import _geometry, _plain
from pyqed.narg.qchem.abelian import NARG
from pyqed.qchem import Molecule


def _run(geometry, args):
    mol = Molecule(atom=geometry, unit="bohr", basis=args.basis)
    mol.build(
        eri="cd",
        options={
            "low_rank_tol": args.cd_tol,
            "parallel": False,
            "eri_screen_tol": 0.0,
        },
    )
    mf = mol.RHF().run(tol=1.0e-10, max_cycle=100, verbose=0)
    solver = NARG(
        mf,
        ncas=args.ncas,
        nelecas=args.nelecas,
        D=args.D,
        nstates=1,
        n0=args.n0,
        growth_sites=1,
        dressing="detached+cc",
        chi=args.chi,
        fast=True,
        store_tensors=False,
        cc_response_tol=args.cc_response_tol,
    )
    started = time.perf_counter()
    solver.run()
    seconds = time.perf_counter() - started
    return solver, seconds


def _history_summary(solver):
    detached = solver.detached_history
    cc = solver.dressing_history
    return {
        "growth_steps": len(detached),
        "maximum_frame_rank": max(
            (int(item["frame_rank"]) for item in detached),
            default=0,
        ),
        "maximum_detached_dimension": max(
            (int(item["detached_dim"]) for item in detached),
            default=0,
        ),
        "maximum_retained_dimension": max(
            (int(item["retained_dim"]) for item in detached),
            default=0,
        ),
        "maximum_cc_response_rank": max(
            (int(item["response_rank"]) for item in cc),
            default=0,
        ),
        "maximum_sector_leakage": max(
            (float(item["maximum_sector_leakage"]) for item in detached),
            default=0.0,
        ),
        "sector_label_corrections": sum(
            int(item["sector_label_corrections"]) for item in detached
        ),
        "total_detached_improvement_hartree": float(
            sum(float(item["detached_improvement"]) for item in detached)
        ),
        "total_cc_sector_energy_gain_hartree": float(
            sum(float(item["sector_energy_gain"]) for item in cc)
        ),
    }


def _plot(records, output):
    labels = ["$Q_1$", "$Q_2$"][: len(records)]
    errors = [record["energy_error_hartree"] for record in records]
    timings = [record["seconds"] for record in records]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), constrained_layout=True)
    axes[0].bar(labels, errors, color="#176B87", width=0.58)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel(r"$E_{\rm NARG}-E_{\rm CASCI}$ (hartree)")
    axes[1].bar(labels, timings, color="#D1495B", width=0.58)
    axes[1].set_ylabel("NARG time (s)")
    for label, axis in zip("ab", axes):
        axis.text(-0.16, 1.04, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(output, dpi=320)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--ncas", type=int, default=14)
    parser.add_argument("--nelecas", type=int, default=14)
    parser.add_argument("--D", type=int, default=512)
    parser.add_argument("--chi", type=int)
    parser.add_argument("--n0", type=int, default=4)
    parser.add_argument("--geometries", type=int, choices=(1, 2), default=2)
    parser.add_argument("--coupling-displacement", type=float, default=0.02)
    parser.add_argument("--cd-tol", type=float, default=1.0e-10)
    parser.add_argument("--cc-response-tol", type=float, default=1.0e-10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/pyrazine_narg_detached_cc"),
    )
    args = parser.parse_args()
    args.chi = 8 * args.D if args.chi is None else args.chi
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = json.loads(args.reference.read_text())
    casci = np.asarray(reference["casci_energies_hartree"], dtype=float)[:, 0]
    geometries = [
        _geometry(1.0),
        _geometry(1.0, args.coupling_displacement),
    ][: args.geometries]
    records = []
    for index, geometry in enumerate(geometries):
        print(f"geometry {index + 1}", flush=True)
        solver, seconds = _run(geometry, args)
        energy = float(np.asarray(solver.e_tot).reshape(-1)[0])
        record = {
            "geometry": index + 1,
            "energy_hartree": energy,
            "casci_energy_hartree": float(casci[index]),
            "energy_error_hartree": energy - float(casci[index]),
            "seconds": seconds,
            **_history_summary(solver),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    payload = {
        "system": f"pyrazine/{args.basis} CAS({args.nelecas},{args.ncas})",
        "symmetry": "abelian_number_sz",
        "dressing": "detached_cc",
        "D": args.D,
        "chi": args.chi,
        "n0": args.n0,
        "cc_response_tol": args.cc_response_tol,
        "records": records,
        "cross_geometry_overlap_available": False,
    }
    (args.output_dir / "pyrazine_narg_detached_cc.json").write_text(
        json.dumps(_plain(payload), indent=2) + "\n"
    )
    _plot(records, args.output_dir / "pyrazine_narg_detached_cc.png")
    print(json.dumps(_plain(payload), indent=2), flush=True)


if __name__ == "__main__":
    main()
