"""Run bounded-memory native GDF-KRHF for small crystalline benchmarks."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from pyqed.qchem.pbc import Cell


def _fcc_primitive(length):
    half = 0.5 * float(length)
    return np.asarray(
        [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]],
        dtype=float,
    )


CASES = {
    "diamond": (
        f"C 0 0 0; C {6.74 / 4:.12f} {6.74 / 4:.12f} {6.74 / 4:.12f}",
        _fcc_primitive(6.74),
    ),
    "bn": (
        f"B 0 0 0; N {6.83 / 4:.12f} {6.83 / 4:.12f} {6.83 / 4:.12f}",
        _fcc_primitive(6.83),
    ),
}


def _mesh(value):
    values = tuple(int(item) for item in str(value).split(","))
    if len(values) != 3 or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("mesh must contain three positive integers")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="diamond")
    parser.add_argument("--kmesh", type=_mesh, default=(1, 1, 1))
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--auxbasis", default="def2-svp-jkfit")
    parser.add_argument("--precision", type=float, default=1.0e-8)
    parser.add_argument("--storage", choices=("auto", "memory", "disk"), default="auto")
    parser.add_argument("--max-memory-mb", type=float, default=2048.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-cycle", type=int, default=80)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_gdf_production.json"),
    )
    args = parser.parse_args()

    atom, lattice = CASES[args.case]
    cell = Cell(
        atom=atom,
        a=lattice,
        basis=args.basis,
        unit="bohr",
        dimension=3,
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
        omega="auto",
        mesh="auto",
        pair_cut="auto",
        image_cut="auto",
        storage=args.storage,
        max_memory_mb=args.max_memory_mb,
        cache_dir=str(args.cache_dir),
        stream_pairs=True,
    )
    mf.gdf_reciprocal_kernel = "range_separated"

    started = time.perf_counter()
    mf.with_df.build(workers=args.workers)
    prebuild_seconds = time.perf_counter() - started
    started = time.perf_counter()
    mf.run(max_cycle=args.max_cycle, conv_tol=1.0e-10, conv_tol_dm=1.0e-8)
    scf_seconds = time.perf_counter() - started

    result = {
        "case": args.case,
        "kmesh": args.kmesh,
        "nkpts": int(mf.nkpts),
        "nao": int(cell.nao),
        "converged": bool(mf.converged),
        "iterations": int(mf.niter),
        "energy_Ha": float(mf.e_tot),
        "prebuild_seconds": float(prebuild_seconds),
        "scf_seconds_after_prebuild": float(scf_seconds),
        "factor_memory_bytes": int(mf.with_df.memory_bytes),
        "factor_disk_bytes": int(mf.with_df.disk_bytes),
        "cache_files": list(mf.with_df.cache_files),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not args.keep_cache:
        mf.with_df.close()


if __name__ == "__main__":
    main()
