#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from scipy import linalg

from pyqed.mps.nonabelian import (
    SweepDriver,
    build_random_spatial_mps,
    build_spatial_hubbard_mpo,
    spatial_annihilate_down,
    spatial_annihilate_up,
    spatial_create_down,
    spatial_create_up,
    spatial_double_occupancy,
    spatial_number,
    spatial_parity,
)

try:
    from tenpy.algorithms import dmrg as tenpy_dmrg
    from tenpy.models.hubbard import FermiHubbardChain
    from tenpy.networks.mps import MPS

    _HAVE_TENPY = True
except Exception:  # pragma: no cover - optional dependency for example script
    tenpy_dmrg = None
    FermiHubbardChain = None
    MPS = None
    _HAVE_TENPY = False


def _kron_all(ops):
    out = np.asarray(ops[0], dtype=complex)
    for op in ops[1:]:
        out = np.kron(out, np.asarray(op, dtype=complex))
    return out


def _jw_bilinear_dense(nsites, left_site, left_operator, right_site, right_operator, parity):
    if left_site >= right_site:
        raise ValueError("_jw_bilinear_dense requires left_site < right_site.")
    ident = np.eye(parity.shape[0], dtype=complex)
    ops = [ident.copy() for _ in range(nsites)]
    ops[left_site] = np.asarray(left_operator @ parity, dtype=complex)
    for site in range(left_site + 1, right_site):
        ops[site] = np.asarray(parity, dtype=complex)
    ops[right_site] = np.asarray(right_operator, dtype=complex)
    return _kron_all(ops)


def _dense_spatial_hubbard_hamiltonian(nsites, *, hopping_t, chemical_potential, onsite_u):
    ident = np.eye(4, dtype=complex)
    parity = spatial_parity().as_dense().astype(complex)
    number = spatial_number().as_dense().astype(complex)
    doublon = spatial_double_occupancy().as_dense().astype(complex)
    c_up = spatial_annihilate_up().as_dense().astype(complex)
    cd_up = spatial_create_up().as_dense().astype(complex)
    c_down = spatial_annihilate_down().as_dense().astype(complex)
    cd_down = spatial_create_down().as_dense().astype(complex)

    h = np.zeros((4**nsites, 4**nsites), dtype=complex)
    for site in range(nsites):
        ops = [ident.copy() for _ in range(nsites)]
        ops[site] = -chemical_potential * number + onsite_u * doublon
        h += _kron_all(ops)

    for site in range(nsites - 1):
        h += -hopping_t * _jw_bilinear_dense(nsites, site, cd_up, site + 1, c_up, parity)
        h += +hopping_t * _jw_bilinear_dense(nsites, site, c_up, site + 1, cd_up, parity)
        h += -hopping_t * _jw_bilinear_dense(nsites, site, cd_down, site + 1, c_down, parity)
        h += +hopping_t * _jw_bilinear_dense(nsites, site, c_down, site + 1, cd_down, parity)
    return h


def _sector_basis_indices(nsites, *, target_charge, target_two_sz):
    local_charge = (0, 1, 1, 2)
    local_two_sz = (0, 1, -1, 0)
    indices = []
    for state_index in range(4**nsites):
        value = state_index
        charge = 0
        two_sz = 0
        for _site in range(nsites):
            digit = value % 4
            value //= 4
            charge += local_charge[digit]
            two_sz += local_two_sz[digit]
        if charge == target_charge and two_sz == target_two_sz:
            indices.append(state_index)
    return np.asarray(indices, dtype=int)


def _ed_ground_state_energy(nsites, *, hopping_t, onsite_u, chemical_potential):
    start = time.perf_counter()
    h = _dense_spatial_hubbard_hamiltonian(
        nsites,
        hopping_t=hopping_t,
        onsite_u=onsite_u,
        chemical_potential=chemical_potential,
    )
    target_charge = int(nsites)
    target_two_sz = 0 if nsites % 2 == 0 else 1
    indices = _sector_basis_indices(
        nsites,
        target_charge=target_charge,
        target_two_sz=target_two_sz,
    )
    h = h[np.ix_(indices, indices)]
    energy = float(
        np.real(
            linalg.eigvalsh(
                h,
                subset_by_index=(0, 0),
                check_finite=False,
            )[0]
        )
    )
    return {
        "energy": energy,
        "time_s": time.perf_counter() - start,
        "status": "ok",
    }


def _half_filled_product_state(length):
    state = []
    for site in range(length):
        state.append("up" if site % 2 == 0 else "down")
    return state


def _tenpy_ground_state_energy(
    length,
    *,
    max_bond,
    hopping_t,
    onsite_u,
    chemical_potential,
    max_sweeps,
    max_e_err,
):
    if not _HAVE_TENPY:
        return {"status": "unavailable"}

    start = time.perf_counter()
    logging_threshold = logging.root.manager.disable
    try:
        logging.disable(logging.INFO)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="unit_cell_width is a new argument for MPS and similar classes.*",
            )
            model = FermiHubbardChain(
                {
                    "L": int(length),
                    "t": float(hopping_t),
                    "U": float(onsite_u),
                    "mu": float(chemical_potential),
                    "bc_MPS": "finite",
                    "cons_N": "N",
                    "cons_Sz": "Sz",
                }
            )
            psi = MPS.from_product_state(
                model.lat.mps_sites(),
                _half_filled_product_state(length),
                bc="finite",
            )
            info = tenpy_dmrg.run(
                psi,
                model,
                {
                    "mixer": False,
                    "max_sweeps": int(max_sweeps),
                    "max_E_err": float(max_e_err),
                    "trunc_params": {
                        "chi_max": int(max_bond),
                        "svd_min": 1e-14,
                    },
                },
            )
    finally:
        logging.disable(logging_threshold)
    sweep_stats = info.get("sweep_statistics", {})
    return {
        "energy": float(np.real(info["E"])),
        "time_s": time.perf_counter() - start,
        "status": "ok",
        "nsweeps": len(sweep_stats.get("E", ())),
        "chi": list(getattr(psi, "chi", ())),
    }


def _chain_energy_converged(driver, *, energy_tol):
    if driver.converged:
        return True
    if len(driver.history) < 2:
        return False
    curr_entry = driver.history[-1]
    prev_entry = None
    for candidate in reversed(driver.history[:-1]):
        if candidate.get("direction") == curr_entry.get("direction"):
            prev_entry = candidate
            break
    if prev_entry is None:
        prev_entry = driver.history[-2]
    prev = prev_entry.get("energy")
    curr = curr_entry.get("energy")
    if prev is not None and curr is not None:
        if abs(float(curr) - float(prev)) <= float(energy_tol):
            return True

    best_history = []
    best = None
    for entry in driver.history:
        energy = entry.get("energy")
        if energy is None:
            continue
        best = float(energy) if best is None else min(best, float(energy))
        best_history.append(best)
    if len(best_history) >= 3:
        return abs(best_history[-1] - best_history[-3]) <= float(energy_tol)
    return False


def _pyqed_ground_state_energy(
    length,
    *,
    max_bond,
    seed,
    bond_multiplicity,
    hopping_t,
    onsite_u,
    chemical_potential,
    max_nsweeps,
    cutoff,
    energy_tol,
):
    start = time.perf_counter()
    sites = build_random_spatial_mps(
        int(length),
        seed=int(seed),
        bond_multiplicity=int(bond_multiplicity),
    )
    mpo = build_spatial_hubbard_mpo(
        sites,
        hopping_t=float(hopping_t),
        onsite_u=float(onsite_u),
        chemical_potential=float(chemical_potential),
    )
    driver = SweepDriver(
        [site.copy() for site in sites],
        nsweeps=int(max_nsweeps),
        mpo_factors=mpo,
        max_bond=int(max_bond),
        cutoff=float(cutoff),
    )
    driver.run()
    return {
        "energy": driver.last_energy,
        "time_s": time.perf_counter() - start,
        "status": "ok",
        "nsweeps": driver.ncompleted,
        "converged": _chain_energy_converged(driver, energy_tol=energy_tol),
        "history": [entry.get("energy") for entry in driver.history],
        "local_solver_kwargs": [entry.get("local_solver_kwargs") for entry in driver.history],
    }


def _energy_error(energy, reference):
    if energy is None or reference is None:
        return None
    return abs(float(energy) - float(reference))


def _iter_cases(lengths, max_bonds, hopping_ts, onsite_us, chemical_potentials):
    for length, max_bond, hopping_t, onsite_u, chemical_potential in product(
        lengths,
        max_bonds,
        hopping_ts,
        onsite_us,
        chemical_potentials,
    ):
        yield {
            "length": int(length),
            "max_bond": int(max_bond),
            "hopping_t": float(hopping_t),
            "onsite_u": float(onsite_u),
            "chemical_potential": float(chemical_potential),
        }


def run_case(
    *,
    length,
    max_bond,
    hopping_t,
    onsite_u,
    chemical_potential,
    seed,
    bond_multiplicity,
    pyqed_max_nsweeps,
    pyqed_cutoff,
    pyqed_energy_tol,
    run_ed,
    ed_max_length,
    run_tenpy,
    tenpy_max_sweeps,
    tenpy_max_e_err,
):
    row = {
        "length": int(length),
        "max_bond": int(max_bond),
        "hopping_t": float(hopping_t),
        "onsite_u": float(onsite_u),
        "chemical_potential": float(chemical_potential),
    }

    pyqed_result = _pyqed_ground_state_energy(
        length,
        max_bond=max_bond,
        seed=seed,
        bond_multiplicity=bond_multiplicity,
        hopping_t=hopping_t,
        onsite_u=onsite_u,
        chemical_potential=chemical_potential,
        max_nsweeps=pyqed_max_nsweeps,
        cutoff=pyqed_cutoff,
        energy_tol=pyqed_energy_tol,
    )
    row["pyqed"] = pyqed_result

    ed_result = {"status": "skipped"}
    if run_ed and int(length) <= int(ed_max_length):
        ed_result = _ed_ground_state_energy(
            length,
            hopping_t=hopping_t,
            onsite_u=onsite_u,
            chemical_potential=chemical_potential,
        )
    row["ed"] = ed_result

    tenpy_result = {"status": "skipped"}
    if run_tenpy:
        tenpy_result = _tenpy_ground_state_energy(
            length,
            max_bond=max_bond,
            hopping_t=hopping_t,
            onsite_u=onsite_u,
            chemical_potential=chemical_potential,
            max_sweeps=tenpy_max_sweeps,
            max_e_err=tenpy_max_e_err,
        )
    row["tenpy"] = tenpy_result

    ed_energy = ed_result.get("energy")
    tenpy_energy = tenpy_result.get("energy")
    pyqed_energy = pyqed_result.get("energy")

    row["pyqed_error_vs_ed"] = _energy_error(pyqed_energy, ed_energy)
    row["tenpy_error_vs_ed"] = _energy_error(tenpy_energy, ed_energy)
    row["pyqed_error_vs_tenpy"] = _energy_error(pyqed_energy, tenpy_energy)
    return row


def _fmt_float(value, *, digits=6, sci=False):
    if value is None:
        return "n/a"
    if sci:
        return f"{float(value):.3e}"
    return f"{float(value):.{digits}f}"


def _format_solver_kwargs(items):
    if not items:
        return "[]"
    chunks = []
    for idx, item in enumerate(items):
        if not item:
            chunks.append(f"s{idx}={{}}")
            continue
        joined = ", ".join(f"{key}={value!r}" for key, value in sorted(item.items()))
        chunks.append(f"s{idx}=" + "{" + joined + "}")
    return "[" + ", ".join(chunks) + "]"


def _format_table(rows):
    headers = [
        "L",
        "D",
        "t",
        "U",
        "mu",
        "pyqed_time_s",
        "pyqed_nsweeps",
        "pyqed_conv",
        "pyqed_E",
        "pyqed_err_ed",
        "tenpy_time_s",
        "tenpy_E",
        "tenpy_err_ed",
        "ed_time_s",
        "ed_E",
        "|pyqed-tenpy|",
    ]
    formatted = []
    for row in rows:
        pyqed = row["pyqed"]
        tenpy = row["tenpy"]
        ed = row["ed"]
        formatted.append(
            {
                "L": str(row["length"]),
                "D": str(row["max_bond"]),
                "t": _fmt_float(row["hopping_t"], digits=2),
                "U": _fmt_float(row["onsite_u"], digits=2),
                "mu": _fmt_float(row["chemical_potential"], digits=2),
                "pyqed_time_s": _fmt_float(pyqed.get("time_s")),
                "pyqed_nsweeps": str(pyqed.get("nsweeps", "n/a")),
                "pyqed_conv": str(pyqed.get("converged", "n/a")),
                "pyqed_E": _fmt_float(pyqed.get("energy"), digits=12),
                "pyqed_err_ed": _fmt_float(row.get("pyqed_error_vs_ed"), sci=True),
                "tenpy_time_s": _fmt_float(tenpy.get("time_s")),
                "tenpy_E": _fmt_float(tenpy.get("energy"), digits=12),
                "tenpy_err_ed": _fmt_float(row.get("tenpy_error_vs_ed"), sci=True),
                "ed_time_s": _fmt_float(ed.get("time_s")),
                "ed_E": _fmt_float(ed.get("energy"), digits=12),
                "|pyqed-tenpy|": _fmt_float(row.get("pyqed_error_vs_tenpy"), sci=True),
            }
        )
    widths = {
        header: max(len(header), *(len(item[header]) for item in formatted))
        for header in headers
    }
    lines = []
    lines.append("  ".join(header.ljust(widths[header]) for header in headers))
    lines.append("  ".join("-" * widths[header] for header in headers))
    for item in formatted:
        lines.append("  ".join(item[header].ljust(widths[header]) for header in headers))
    lines.append("")
    for row in rows:
        pyqed = row["pyqed"]
        tenpy = row["tenpy"]
        lines.append(
            "case "
            f"L={row['length']} D={row['max_bond']} t={row['hopping_t']} "
            f"U={row['onsite_u']} mu={row['chemical_potential']}"
        )
        lines.append(
            "  pyqed_local_solver="
            + _format_solver_kwargs(pyqed.get("local_solver_kwargs"))
        )
        if pyqed.get("history"):
            lines.append(
                "  pyqed_history=[" + ", ".join(f"{float(x):.12f}" for x in pyqed["history"]) + "]"
            )
        if tenpy.get("chi"):
            lines.append(
                "  tenpy_chi=[" + ", ".join(str(int(x)) for x in tenpy["chi"]) + "]"
            )
    return "\n".join(lines)


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark non-Abelian Hubbard chain sweeps across L/D/parameter grids, "
            "with optional ED and TeNPy reference energies."
        )
    )
    parser.add_argument("--lengths", nargs="+", type=int, default=[4, 6])
    parser.add_argument("--max-bonds", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--hopping-ts", nargs="+", type=float, default=[1.0])
    parser.add_argument("--onsite-us", nargs="+", type=float, default=[4.0, 8.0])
    parser.add_argument("--chemical-potentials", nargs="+", type=float, default=[0.0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bond-multiplicity", type=int, default=4)
    parser.add_argument("--pyqed-max-nsweeps", type=int, default=2)
    parser.add_argument("--pyqed-cutoff", type=float, default=0.0)
    parser.add_argument("--pyqed-energy-tol", type=float, default=1e-8)
    parser.add_argument("--skip-ed", action="store_true")
    parser.add_argument(
        "--ed-max-length",
        type=int,
        default=6,
        help="Only run dense ED for L <= this value.",
    )
    parser.add_argument("--skip-tenpy", action="store_true")
    parser.add_argument("--tenpy-max-sweeps", type=int, default=4)
    parser.add_argument("--tenpy-max-e-err", type=float, default=1e-10)
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format for stdout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to write the full benchmark payload.",
    )
    args = parser.parse_args()

    rows = [
        run_case(
            length=case["length"],
            max_bond=case["max_bond"],
            hopping_t=case["hopping_t"],
            onsite_u=case["onsite_u"],
            chemical_potential=case["chemical_potential"],
            seed=args.seed,
            bond_multiplicity=args.bond_multiplicity,
            pyqed_max_nsweeps=args.pyqed_max_nsweeps,
            pyqed_cutoff=args.pyqed_cutoff,
            pyqed_energy_tol=args.pyqed_energy_tol,
            run_ed=not args.skip_ed,
            ed_max_length=args.ed_max_length,
            run_tenpy=not args.skip_tenpy,
            tenpy_max_sweeps=args.tenpy_max_sweeps,
            tenpy_max_e_err=args.tenpy_max_e_err,
        )
        for case in _iter_cases(
            args.lengths,
            args.max_bonds,
            args.hopping_ts,
            args.onsite_us,
            args.chemical_potentials,
        )
    ]

    payload = {
        "settings": {
            "lengths": args.lengths,
            "max_bonds": args.max_bonds,
            "hopping_ts": args.hopping_ts,
            "onsite_us": args.onsite_us,
            "chemical_potentials": args.chemical_potentials,
            "seed": args.seed,
            "bond_multiplicity": args.bond_multiplicity,
            "pyqed_max_nsweeps": args.pyqed_max_nsweeps,
            "pyqed_cutoff": args.pyqed_cutoff,
            "pyqed_energy_tol": args.pyqed_energy_tol,
            "run_ed": not args.skip_ed,
            "ed_max_length": args.ed_max_length,
            "run_tenpy": not args.skip_tenpy,
            "tenpy_max_sweeps": args.tenpy_max_sweeps,
            "tenpy_max_e_err": args.tenpy_max_e_err,
            "tenpy_available": _HAVE_TENPY,
        },
        "rows": rows,
    }

    if args.output is not None:
        args.output.write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )

    if args.format == "json":
        print(json.dumps(payload, indent=2, default=_json_default))
    else:
        print(_format_table(rows))


if __name__ == "__main__":
    main()
