#!/usr/bin/env python3
"""Compare dense and factor-only builtin CASSCF on H4 / STO-3G / CAS(4,4)."""

import contextlib
import io
import logging
import time
from statistics import mean

from pyqed.qchem import CASSCF, Molecule


ATOM = "H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4"
BASIS = "sto-3g"
NCAS = 4
NELECAS = 4
REPEATS = 3


def run_case(label, options, repeats=REPEATS):
    build_times = []
    rhf_times = []
    casscf_times = []
    e_rhf = None
    e_casscf = None
    cycles = None

    for _ in range(repeats):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            t0 = time.perf_counter()

            mol = Molecule(atom=ATOM, unit="angstrom", basis=BASIS)
            mol.build(driver="builtin", options=options)

            t1 = time.perf_counter()

            mf = mol.RHF().run()

            t2 = time.perf_counter()

            mc = CASSCF(mf, ncas=NCAS, nelecas=NELECAS, max_cycle=20).run(nstates=1)

            t3 = time.perf_counter()

        build_times.append(t1 - t0)
        rhf_times.append(t2 - t1)
        casscf_times.append(t3 - t2)
        e_rhf = float(mf.e_tot)
        e_casscf = float(mc.e_tot[0])
        cycles = len(mc.history)

    return {
        "label": label,
        "build": mean(build_times),
        "rhf": mean(rhf_times),
        "casscf": mean(casscf_times),
        "total": mean(build_times) + mean(rhf_times) + mean(casscf_times),
        "e_rhf": e_rhf,
        "e_casscf": e_casscf,
        "cycles": cycles,
    }


def print_result(result, reference_total=None, reference_energy=None):
    print(result["label"])
    print(f"  build   {result['build']:.6f} s")
    print(f"  rhf     {result['rhf']:.6f} s")
    print(f"  casscf  {result['casscf']:.6f} s")
    print(f"  total   {result['total']:.6f} s")
    print(f"  E(HF)   {result['e_rhf']:.15f}")
    print(f"  E(CASSCF) {result['e_casscf']:.15f}")
    print(f"  cycles  {result['cycles']}")
    if reference_total is not None:
        print(f"  speedup {reference_total / result['total']:.3f}x")
    if reference_energy is not None:
        print(f"  dE      {result['e_casscf'] - reference_energy:+.3e} Eh")
    print()


def main():
    logging.getLogger().setLevel(logging.ERROR)

    dense = run_case("dense", {"eri_representation": "dense"})
    factors = run_case(
        "factors_only",
        {"eri_representation": "factors", "low_rank_tol": 1e-8},
    )

    print("H4 / STO-3G / CAS(4,4)")
    print()
    print_result(dense)
    print_result(
        factors,
        reference_total=dense["total"],
        reference_energy=dense["e_casscf"],
    )


if __name__ == "__main__":
    main()
