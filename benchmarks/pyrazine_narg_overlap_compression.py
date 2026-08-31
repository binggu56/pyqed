"""Scan controlled bond caps for cross-geometry SU(2)-NARG overlap."""

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
from pyqed.narg.qchem.su2 import NARG
from pyqed.qchem import Molecule


def _complex_matrix(values):
    return np.asarray(
        [
            [
                complex(value["real"], value["imag"])
                if isinstance(value, dict)
                else complex(value)
                for value in row
            ]
            for row in values
        ],
        dtype=complex,
    )


def _run_narg(geometry, args):
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
    started = time.perf_counter()
    narg = NARG(
        mf,
        ncas=args.ncas,
        nelecas=args.nelecas,
        D=args.D,
        nstates=args.nstates,
        target_j2=0,
        su2_backend=args.su2_backend,
        threads=args.threads,
        carry_rdm_operators=False,
        carry_spin_rdm_operators=False,
    ).run()
    return narg, time.perf_counter() - started


def _plot(records, output):
    caps = np.asarray([record["max_bond"] for record in records])
    timings = np.asarray([record["seconds"] for record in records])
    peaks = np.asarray([record["peak_reduced_bond_dimension"] for record in records])
    compression = np.asarray([record["max_compression_error"] for record in records])
    total = np.asarray([record["max_casci_error"] for record in records])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    axes[0].plot(caps, timings, "o-", color="#176B87", label="Overlap time")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Overlap bond cap $D_{\rm ovlp}$")
    axes[0].set_ylabel("Time (s)")
    peak_axis = axes[0].twinx()
    peak_axis.plot(caps, peaks, "s--", color="#D1495B", label="Peak bond")
    peak_axis.set_ylabel("Peak reduced bond")

    axes[1].plot(caps, compression, "o-", color="#6A4C93", label="Compression")
    axes[1].plot(caps, total, "s-", color="#D1495B", label="vs CASCI")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Overlap bond cap $D_{\rm ovlp}$")
    axes[1].set_ylabel("Maximum overlap-magnitude error")
    axes[1].legend(frameon=False)
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
    parser.add_argument("--nstates", type=int, default=4)
    parser.add_argument("--D", type=int, default=256)
    parser.add_argument(
        "--caps",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--cutoff", type=float, default=1.0e-12)
    parser.add_argument("--coupling-displacement", type=float, default=0.02)
    parser.add_argument("--cd-tol", type=float, default=1.0e-10)
    parser.add_argument("--su2-backend", default="python")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/pyrazine_narg_overlap_compression"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = json.loads(args.reference.read_text())
    casci_overlap = np.asarray(reference["casci_overlap"], dtype=complex)
    exact_narg_overlap = _complex_matrix(reference["narg_overlap"])
    reference_energies = np.asarray(reference["narg_energies_hartree"])

    print("geometry 1", flush=True)
    first, first_seconds = _run_narg(_geometry(1.0), args)
    print(f"NARG: {first_seconds:.3f} s", flush=True)
    print("geometry 2", flush=True)
    second, second_seconds = _run_narg(
        _geometry(1.0, args.coupling_displacement),
        args,
    )
    print(f"NARG: {second_seconds:.3f} s", flush=True)
    energies = np.asarray([first.e_tot, second.e_tot])
    energy_reproduction_error = float(np.max(np.abs(energies - reference_energies)))

    records = []
    matrices = []
    for cap in args.caps:
        started = time.perf_counter()
        overlap, info = first.overlap(
            second,
            orbital_split="auto",
            orbital_map_threshold=0.0,
            cutoff=args.cutoff,
            max_bond=cap,
            return_info=True,
        )
        seconds = time.perf_counter() - started
        overlap = np.asarray(overlap)
        transform = info["transforms"]["ket"][0]
        record = {
            "max_bond": int(cap),
            "seconds": seconds,
            "max_compression_error": float(
                np.max(np.abs(np.abs(overlap) - np.abs(exact_narg_overlap)))
            ),
            "rms_compression_error": float(
                np.sqrt(np.mean((np.abs(overlap) - np.abs(exact_narg_overlap)) ** 2))
            ),
            "max_casci_error": float(
                np.max(np.abs(np.abs(overlap) - np.abs(casci_overlap)))
            ),
            "peak_reduced_bond_dimension": int(
                transform["peak_reduced_bond_dimension"]
            ),
            "sum_discarded_weight": float(transform["sum_gate_discarded_weight"]),
            "max_discarded_weight": float(transform["max_gate_discarded_weight"]),
            "compiled_channel_mix_batches": int(
                transform["compiled_channel_mix_batches"]
            ),
        }
        records.append(record)
        matrices.append(overlap)
        print(json.dumps(record), flush=True)

    payload = {
        "system": f"pyrazine/{args.basis} CAS({args.nelecas},{args.ncas})",
        "D": args.D,
        "cutoff": args.cutoff,
        "reference": str(args.reference),
        "narg_state_seconds": [first_seconds, second_seconds],
        "energy_reproduction_error_hartree": energy_reproduction_error,
        "records": records,
    }
    (args.output_dir / "pyrazine_narg_overlap_compression.json").write_text(
        json.dumps(_plain(payload), indent=2) + "\n"
    )
    np.savez(
        args.output_dir / "pyrazine_narg_overlap_compression.npz",
        caps=np.asarray(args.caps),
        overlaps=np.asarray(matrices),
        casci_overlap=casci_overlap,
        exact_narg_overlap=exact_narg_overlap,
    )
    _plot(records, args.output_dir / "pyrazine_narg_overlap_compression.png")
    print(json.dumps(_plain(payload), indent=2), flush=True)


if __name__ == "__main__":
    main()
