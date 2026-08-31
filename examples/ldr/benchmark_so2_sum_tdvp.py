#!/usr/bin/env python3
"""Benchmark one cached SO2 component-TDVP step without reference overhead."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import time

import numpy as np

from examples.ldr.so2_casci_cgldr import DEFAULT_SCAN_DIR, load_so2_linked_scan
from examples.ldr.so2_casci_cgldr_dense import dense_kinetic, nuclear_packet
from examples.ldr.so2_procrustes_tdvp import initial_state
from examples.ldr.so2_procrustes_tt import DEFAULT_REFERENCE, DEFAULT_TWO
from pyqed.mps.decompose import tt_to_tensor
from pyqed.mps.tdvp import TDVPEngine
from pyqed.units import au2fs


DEFAULT_HAMILTONIAN = Path(
    "/private/tmp/so2_procrustes_tdvp_ftt_r48_20fs_optimized/hamiltonian.pkl"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hamiltonian", type=Path, default=DEFAULT_HAMILTONIAN)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--gauge", type=Path, default=DEFAULT_TWO)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--workers", type=int, nargs="+", default=(1, 2, 4, 6))
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--state-rank", type=int, default=48)
    parser.add_argument("--krylov-dim", type=int, default=12)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-11)
    args = parser.parse_args()

    with args.hamiltonian.open("rb") as stream:
        hamiltonian = pickle.load(stream)["hamiltonian"]
    with np.load(args.reference) as archive:
        grids = tuple(
            np.asarray(archive[name]) for name in ("qs", "theta", "qa")
        )
    with np.load(args.gauge) as archive:
        gauge = np.asarray(archive["gauge"], dtype=complex)
        primary_gauge = np.asarray(archive["primary_gauge"], dtype=complex)
    scan = load_so2_linked_scan(args.scan_dir)
    _kinetic, axes = dense_kinetic(scan, *grids)
    packet = nuclear_packet(*grids, axes)
    state, _physical = initial_state(
        packet,
        primary_gauge,
        gauge,
        2,
        args.state_rank,
    )

    reference = None
    for workers in args.workers:
        engine = TDVPEngine(
            hamiltonian,
            max_bond=args.state_rank,
            cutoff=1.0e-11,
            krylov_dim=args.krylov_dim,
            krylov_tol=args.krylov_tol,
            workers=workers,
        )
        started = time.perf_counter()
        output, info = engine.step(state, args.dt_fs / au2fs)
        elapsed = time.perf_counter() - started
        engine.close()
        values = np.asarray(
            tt_to_tensor(
                [output._get_std_B(site) for site in range(output.L)]
            )
        ).reshape(-1)
        if reference is None:
            reference = values
            error = 0.0
        else:
            phase = np.vdot(reference, values)
            values *= np.exp(-1j * np.angle(phase))
            error = float(np.max(np.abs(values - reference)))
        print(
            f"workers={workers}: {elapsed:.6f} s, "
            f"max error={error:.3e}, "
            f"truncation={info['truncation_error']:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
