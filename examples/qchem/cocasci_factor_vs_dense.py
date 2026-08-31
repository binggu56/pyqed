#!/usr/bin/env python3
"""Compare dense and factor-only builtin COCAS on H4 / STO-3G / CAS(4,4)."""

import contextlib
import io
import logging
import time
from statistics import mean

from pyqed.qchem import COCAS, Molecule


ATOM = "H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4"
BASIS = "sto-3g"
NCAS = 4
NELECAS = 4
REPEATS = 3


def run_case(label, options, repeats=REPEATS):
    build_times = []
    rhf_times = []
    cocasci_times = []
    e_rhf = None
    e_cocasci = None
    cycles = None
    backend = None

    for _ in range(repeats):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            t0 = time.perf_counter()

            mol = Molecule(atom=ATOM, unit="angstrom", basis=BASIS)
            mol.build(options=options)

            t1 = time.perf_counter()

            mf = mol.RHF().run()

            t2 = time.perf_counter()

            mc = COCAS(mf, ncas=NCAS, nelecas=NELECAS, max_cycles=20).run(nstates=1)

            t3 = time.perf_counter()

        build_times.append(t1 - t0)
        rhf_times.append(t2 - t1)
        cocasci_times.append(t3 - t2)
        e_rhf = float(mf.e_tot)
        e_cocasci = float(mc.e_tot[0] if hasattr(mc.e_tot, "__len__") else mc.e_tot)
        cycles = len(getattr(mc, "e_history", []))
        backend = getattr(mc, "solver_backend", None)

    return {
        "label": label,
        "build": mean(build_times),
        "rhf": mean(rhf_times),
        "cocasci": mean(cocasci_times),
        "total": mean(build_times) + mean(rhf_times) + mean(cocasci_times),
        "e_rhf": e_rhf,
        "e_cocasci": e_cocasci,
        "cycles": cycles,
        "backend": backend,
    }


def print_result(result, reference_total=None, reference_energy=None):
    print(result["label"])
    print(f"  build    {result['build']:.6f} s")
    print(f"  rhf      {result['rhf']:.6f} s")
    print(f"  cocasci  {result['cocasci']:.6f} s")
    print(f"  total    {result['total']:.6f} s")
    print(f"  E(HF)    {result['e_rhf']:.15f}")
    print(f"  E(COCAS) {result['e_cocasci']:.15f}")
    print(f"  cycles   {result['cycles']}")
    print(f"  backend  {result['backend']}")
    if reference_total is not None:
        print(f"  speedup  {reference_total / result['total']:.3f}x")
    if reference_energy is not None:
        print(f"  dE       {result['e_cocasci'] - reference_energy:+.3e} Eh")
    print()


def main():
    logging.getLogger().setLevel(logging.ERROR)

    dense = run_case("dense", {"eri_representation": "dense"})
    factors = run_case(
        "factors_only",
        {"eri_representation": "factors", "low_rank_tol": 1e-8},
    )

    print("H4 / STO-3G / CAS(4,4) COCAS")
    print()
    print_result(dense)
    print_result(
        factors,
        reference_total=dense["total"],
        reference_energy=dense["e_cocasci"],
    )


if __name__ == "__main__":
    main()
