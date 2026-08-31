#!/usr/bin/env python3
"""Isolated block2 worker for the large-CAS SU(2) benchmark."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes


def _rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bond-dim", type=int, required=True)
    parser.add_argument("--half-sweeps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--davidson-tol", type=float, required=True)
    parser.add_argument("--nstates", type=int, required=True)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as active:
        ncas = int(active["ncas"])
        n_elec = int(active["n_elec"])
        spin = int(active["spin"])
        ecore = float(active["ecore"])
        h1e = np.ascontiguousarray(active["h1e"])
        g2e = np.ascontiguousarray(active["g2e"])
        orb_sym = np.asarray(active["orb_sym"], dtype=np.int32).tolist()

    rss_before = _rss_bytes()
    total_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pyqed-block2-large-cas-") as scratch:
        driver = DMRGDriver(
            scratch=scratch,
            symm_type=SymmetryTypes.SU2,
            n_threads=1,
        )
        driver.bw.b.Random.rand_seed(args.seed)
        setup_started = time.perf_counter()
        driver.initialize_system(
            n_sites=ncas,
            n_elec=n_elec,
            spin=spin,
            orb_sym=orb_sym,
        )
        mpo = driver.get_qc_mpo(h1e, g2e, ecore=ecore, iprint=0)
        system_seconds = time.perf_counter() - setup_started

        state_started = time.perf_counter()
        ket = driver.get_random_mps(
            tag="KET",
            bond_dim=args.bond_dim,
            nroots=args.nstates,
        )
        state_initialization_seconds = time.perf_counter() - state_started

        sweep_started = time.perf_counter()
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=args.half_sweeps,
            bond_dims=[args.bond_dim] * args.half_sweeps,
            noises=[0.0] * args.half_sweeps,
            thrds=[args.davidson_tol**2] * args.half_sweeps,
            tol=1.0e-9,
            iprint=0,
            dav_max_iter=100,
            dav_def_max_size=50,
        )
        sweep_seconds = time.perf_counter() - sweep_started

        expectation_started = time.perf_counter()
        if args.nstates == 1:
            exported_energies = [
                float(
                    driver.expectation(ket, mpo, ket, iprint=0)
                    / driver.expectation(
                        ket,
                        driver.get_identity_mpo(),
                        ket,
                        iprint=0,
                    )
                )
            ]
        else:
            roots = [
                driver.split_mps(ket, root, f"ROOT{root}")
                for root in range(args.nstates)
            ]
            identity = driver.get_identity_mpo()
            exported_energies = [
                float(
                    driver.expectation(root, mpo, root, iprint=0)
                    / driver.expectation(root, identity, root, iprint=0)
                )
                for root in roots
            ]
        expectation_seconds = time.perf_counter() - expectation_started

    solver_energies = [
        float(value) for value in np.asarray(energy).reshape(-1)
    ]
    exported_energy = exported_energies[0]
    result = {
        "backend": "block2-su2",
        "energy": exported_energy,
        "state_energies": exported_energies,
        "solver_energy": solver_energies[0],
        "solver_state_energies": solver_energies,
        "returned_mps_total_energy": exported_energy,
        "reported_expectation_error": abs(
            solver_energies[0] - exported_energy
        ),
        "expectation_seconds": float(expectation_seconds),
        "system_seconds": float(system_seconds),
        "state_initialization_seconds": float(
            state_initialization_seconds
        ),
        "run_seconds": float(sweep_seconds),
        "sweep_seconds": float(sweep_seconds),
        "total_seconds": float(time.perf_counter() - total_started),
        "bond_updates": int(max(0, ncas - 1) * args.half_sweeps),
        "bond_updates_per_second": float(
            max(0, ncas - 1) * args.half_sweeps
            / max(sweep_seconds, 1.0e-15)
        ),
        "davidson_tol": float(args.davidson_tol),
        "peak_rss_bytes": int(_rss_bytes()),
        "peak_rss_delta_bytes": max(0, int(_rss_bytes() - rss_before)),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
