"""Validate and profile native two-site SU(2)-LETTA on Hubbard chains.

The fixed-``Sz`` determinant reference is independent of the reduced tensor
network implementation and is exact for the requested small chain.  Larger
orbital counts can be benchmarked without the reference via ``--no-exact``.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np

from pyqed.letta import SU2LETTA


def _bitstrings(nsites, nelec):
    return tuple(
        sum(1 << site for site in occupied)
        for occupied in combinations(range(nsites), nelec)
    )


def _hop(bits, target, source):
    if not (bits >> source) & 1 or ((bits >> target) & 1 and target != source):
        return None
    sign = -1 if (bits & ((1 << source) - 1)).bit_count() % 2 else 1
    updated = bits ^ (1 << source)
    sign *= -1 if (updated & ((1 << target) - 1)).bit_count() % 2 else 1
    return updated | (1 << target), sign


def exact_hubbard_energy(h1e, interaction, *, nelec, spin=0):
    """Return the exact fixed-``N`` and fixed-``Sz`` Hubbard energy."""
    h1e = np.asarray(h1e, dtype=float)
    nsites = h1e.shape[0]
    nup = (int(nelec) + int(spin)) // 2
    ndown = int(nelec) - nup
    if 2 * nup != int(nelec) + int(spin):
        raise ValueError("nelec and spin must define integral up/down populations.")
    basis = tuple(
        (up, down)
        for up in _bitstrings(nsites, nup)
        for down in _bitstrings(nsites, ndown)
    )
    positions = {state: index for index, state in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), dtype=float)
    for column, (up, down) in enumerate(basis):
        matrix[column, column] += float(interaction) * (up & down).bit_count()
        for target in range(nsites):
            for source in range(nsites):
                coefficient = h1e[target, source]
                if coefficient == 0.0:
                    continue
                for species, bits in enumerate((up, down)):
                    result = _hop(bits, target, source)
                    if result is None:
                        continue
                    updated, sign = result
                    state = (updated, down) if species == 0 else (up, updated)
                    matrix[positions[state], column] += coefficient * sign
    matrix = 0.5 * (matrix + matrix.T)
    return float(np.linalg.eigvalsh(matrix)[0])


def hubbard_chain(nsites, *, hopping, interaction, periodic=False):
    h1e = np.zeros((nsites, nsites), dtype=float)
    graph = []
    for site in range(nsites - 1):
        h1e[site, site + 1] = h1e[site + 1, site] = -float(hopping)
        graph.append((site, site + 1))
    if periodic and nsites > 2:
        h1e[0, -1] = h1e[-1, 0] = -float(hopping)
        graph.append((0, nsites - 1))
    eri = np.zeros((nsites, nsites, nsites, nsites), dtype=float)
    for site in range(nsites):
        eri[site, site, site, site] = float(interaction)
    return h1e, eri, tuple(graph)


def run_case(args, *, D, workers, reference):
    h1e, eri, graph = hubbard_chain(
        args.nsites,
        hopping=args.hopping,
        interaction=args.interaction,
        periodic=args.periodic,
    )
    tracemalloc.start()
    started = time.perf_counter()
    state = SU2LETTA.from_integrals(
        h1e,
        eri,
        nelec=args.nelec,
        spin=args.spin,
        graph=graph,
        D=D,
        seed=args.seed,
        workers=workers,
    )
    try:
        state.run(
            nsweeps=args.cycles,
            algorithm="two_site",
            tol=args.tol,
            residual_tol=args.residual_tol,
            truncation_tol=args.truncation_tol,
            consecutive_cycles=args.consecutive_cycles,
            pair_cutoff=args.pair_cutoff,
        )
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
        record = {
            "nsites": int(args.nsites),
            "nelec": int(args.nelec),
            "spin": int(args.spin),
            "D": int(D),
            "workers": int(workers),
            "energy": float(state.energy),
            "reference_energy": reference,
            "energy_error": None if reference is None else float(state.energy - reference),
            "elapsed_s": float(elapsed),
            "python_peak_bytes": int(peak),
            "state_storage_bytes": int(state.storage_nbytes),
            "nparameters": int(state.nparameters),
            "frontier_states": list(state.frontier_states),
            "convergence": state.convergence_summary,
        }
        if args.checkpoint_dir:
            checkpoint = Path(args.checkpoint_dir) / (
                f"su2_letta_n{args.nsites}_D{D}_w{workers}.chk"
            )
            state.save_checkpoint(checkpoint)
            record["checkpoint"] = str(checkpoint)
        return record
    finally:
        state.close()
        tracemalloc.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsites", type=int, default=4)
    parser.add_argument("--nelec", type=int)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--D", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--workers", type=int, nargs="+", default=[1])
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--hopping", type=float, default=1.0)
    parser.add_argument("--interaction", type=float, default=4.0)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--tol", type=float, default=1.0e-9)
    parser.add_argument("--residual-tol", type=float, default=1.0e-8)
    parser.add_argument("--truncation-tol", type=float, default=1.0e-6)
    parser.add_argument("--pair-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--consecutive-cycles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-exact", action="store_true")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.nelec is None:
        args.nelec = args.nsites

    h1e, _eri, _graph = hubbard_chain(
        args.nsites,
        hopping=args.hopping,
        interaction=args.interaction,
        periodic=args.periodic,
    )
    reference = None
    if not args.no_exact:
        reference = exact_hubbard_energy(
            h1e,
            args.interaction,
            nelec=args.nelec,
            spin=args.spin,
        )
    records = [
        run_case(args, D=D, workers=workers, reference=reference)
        for D in args.D
        for workers in args.workers
    ]
    payload = {
        "model": "spatial-orbital Hubbard chain",
        "exact_reference": reference,
        "records": records,
    }
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
