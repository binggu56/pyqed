#!/usr/bin/env python3
"""Isolated block2 worker for the large-CAS SU(2) benchmark."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
from pyqed import num_workers


def _rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bond-dim", type=int, required=True)
    parser.add_argument("--half-sweeps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--davidson-tol", type=float, required=True)
    parser.add_argument("--nstates", type=int, required=True)
    parser.add_argument("--threads", type=int, default=num_workers())
    parser.add_argument("--sweep-tol", type=float, default=1.0e-9)
    parser.add_argument("--memory-gib", type=float, default=1.)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument('--shared-state', type=Path,
                        help='Export the actual initial singlet MPS in left-coupled reduced blocks.')
    parser.add_argument('--fixed-sweeps', action='store_true')
    parser.add_argument("--mpo-algorithm", choices=('bipartite','conventional'), default='bipartite')
    args = parser.parse_args()
    if args.threads < 1 or args.memory_gib <= 0 or args.sweep_tol <= 0:
        parser.error('Use positive thread count, memory budget, and sweep tolerance.')

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
            n_threads=args.threads,
            n_mkl_threads=1,
            stack_mem=int(args.memory_gib * 1024**3),
        )
        driver.bw.b.Random.rand_seed(args.seed)
        setup_started = time.perf_counter()
        driver.initialize_system(
            n_sites=ncas,
            n_elec=n_elec,
            spin=spin,
            orb_sym=orb_sym,
        )
        mpo = driver.get_qc_mpo(h1e, g2e, ecore=ecore, iprint=0,
            algo_type=(MPOAlgorithmTypes.Conventional if args.mpo_algorithm == 'conventional'
                       else MPOAlgorithmTypes.FastBipartite))
        system_seconds = time.perf_counter() - setup_started

        state_started = time.perf_counter()
        ket = driver.get_random_mps(
            tag="KET",
            bond_dim=args.bond_dim,
            nroots=args.nstates,
            center=0,
            dot=1 if args.shared_state else 2,
        )
        initial_energy = None
        interchange_seconds = 0.
        if args.shared_state:
            interchange_started = time.perf_counter()
            if spin != 0 or args.nstates != 1:
                raise ValueError('Shared-state benchmark currently requires a single singlet.')
            from pyblock2.algebra.io import MPSTools, TensorTools
            driver.align_mps_center(ket, ncas-1)
            if ket.canonical_form[-1] == 'S':
                ket.flip_fused_form(ket.center, None, driver.prule)
                ket.save_data()
            tensors = MPSTools.from_block2(ket).tensors
            # The public converter treats a fused singlet boundary as an
            # unfused rank-two tensor when its ket quantum number is vacuum.
            ket.info.load_left_dims(ncas-1)
            left, physical = ket.info.left_dims[ncas-1], ket.info.basis[ncas-1]
            fused = physical.__class__.tensor_product_ref(
                left, physical, ket.info.left_dims_fci[ncas])
            connection = physical.__class__.get_connection_info(left, physical, fused)
            ket.load_tensor(ncas-1)
            tensors[-1] = TensorTools.from_block2_fused(ket.tensors[ncas-1],
                left, physical, fused, connection)
            ket.unload_tensor(ncas-1)
            fused.deallocate()
            left.deallocate()
            arrays, labels = {}, []
            for i, tensor in enumerate(tensors):
                for block in tensor.blocks:
                    qs, value = list(block.q_labels), block.reduced
                    if i == 0 and len(qs) == 2:
                        qs.insert(0, driver.vacuum)
                        value = value[None, ...]
                    if i == ncas-1 and len(qs) == 2:
                        qs.append(driver.target)
                        value = value[..., None]
                    labels.append([i, [[q.n, q.twos, q.pg] for q in qs]])
                    arrays[f'b{len(labels)-1}'] = value
            np.savez(args.shared_state, labels=json.dumps(labels), **arrays)
            initial_energy = float(driver.expectation(ket, mpo, ket, iprint=0)
                / driver.expectation(ket, driver.get_identity_mpo(), ket, iprint=0))
            driver.align_mps_center(ket, 0)
            ket = driver.adjust_mps(ket, dot=2)[0]
            interchange_seconds = time.perf_counter()-interchange_started
        state_initialization_seconds = time.perf_counter() - state_started - interchange_seconds

        sweep_started = time.perf_counter()
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=args.half_sweeps,
            bond_dims=[args.bond_dim] * args.half_sweeps,
            noises=[0.0] * args.half_sweeps,
            thrds=[args.davidson_tol**2] * args.half_sweeps,
            tol=0. if args.fixed_sweeps else args.sweep_tol,
            iprint=args.verbose,
            dav_max_iter=100,
            dav_def_max_size=50,
        )
        sweep_seconds = time.perf_counter() - sweep_started
        sweep_energies = np.asarray(driver._dmrg.energies, dtype=float).reshape(-1, args.nstates).tolist()
        completed = len(sweep_energies)

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
        "initial_energy": initial_energy,
        "fixed_sweeps": args.fixed_sweeps,
        "interchange_seconds": interchange_seconds,
        "version": version('block2'),
        "mpo_algorithm": args.mpo_algorithm,
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
        "total_seconds": float(time.perf_counter() - total_started - interchange_seconds),
        "bond_updates": int(max(0, ncas - 1) * completed),
        "bond_updates_per_second": float(
            max(0, ncas - 1) * completed
            / max(sweep_seconds, 1.0e-15)
        ),
        "davidson_tol": float(args.davidson_tol),
        "sweep_tol": float(args.sweep_tol),
        "half_sweeps": completed,
        "sweep_energies": sweep_energies,
        "threads": args.threads,
        "memory_gib": args.memory_gib,
        "peak_rss_bytes": int(_rss_bytes()),
        "peak_rss_delta_bytes": max(0, int(_rss_bytes() - rss_before)),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
