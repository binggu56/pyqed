#!/usr/bin/env python3
"""Reduced SU(2) two-site DMRG for a finite spinful Hubbard lattice.

Adaptation of White, Phys. Rev. Lett. 69, 2863 (1992),
https://doi.org/10.1103/PhysRevLett.69.2863, with SU(2) multiplet truncation
and normal/complementary operators. This is a finite-bond variational
calculation, not an exact reproduction or a convergence guarantee. Open or
periodic lattice bonds are mapped to an open MPS. D counts total multiplets;
one sweep here is one directional pass. The local norm is explicitly retained.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from examples.mps.hubbard_2d_ed import square_lattice_bonds
from pyqed import num_workers
from pyqed.mps.nonabelian import (
    build_random_reduced_spatial_mps, merge_mps_sites, svd_two_site,
)
from pyqed.mps.nonabelian.environment import contract_chain_expectation
from pyqed.mps.nonabelian.sweep import (
    _identity_mpo_factors_for_sites_and_mpo,
    run_sweeps,
)


def lattice_site_order(lx, ly, *, ordering="row-major"):
    """Return row-major lattice labels in the chosen MPS order."""
    key = ordering.lower().replace("_", "-")
    if key in {"row", "row-major", "rowmajor"}:
        return list(range(lx * ly))
    if key == "snake":
        return [x + lx*y for y in range(ly)
                for x in (range(lx) if y % 2 == 0 else range(lx-1, -1, -1))]
    raise ValueError("ordering must be 'row-major' or 'snake'.")


def build_2d_hubbard_mpo(sites, lx, ly, *, hopping_t, onsite_u,
                         chemical_potential=0., ordering="snake",
                         periodic_x=False, periodic_y=False, n_threads=None):
    """Build reduced normal/complementary Hubbard operators on an open MPS.

    Uses the existing SU(2) chemistry operator builder with only on-site
    interactions; it never constructs a determinant Hamiltonian. The current
    owner accepts an n**4 integral container (13 MiB for 36 sites), populated
    only on the on-site diagonal. This limits larger-lattice setup scalability.
    """
    from pyqed.mps.nonabelian._su2_kernel import SU2MovingEnvironment
    from pyqed.qchem.dmrg.backends.reduced import build_su2_normal_complementary_mpo

    n = lx * ly
    if len(sites) != n or not all(
        s.metadata.get("physical_basis") == "fully_reduced_su2" for s in sites
    ):
        raise ValueError("Supply one fully reduced SU(2) tensor per lattice site.")
    order = lattice_site_order(lx, ly, ordering=ordering)
    index = {site: i for i, site in enumerate(order)}
    h1 = -float(chemical_potential) * np.eye(n)
    for a, b in square_lattice_bonds(lx, ly, periodic_x=periodic_x, periodic_y=periodic_y):
        h1[index[a], index[b]] = h1[index[b], index[a]] = -float(hopping_t)
    eri = np.zeros((n,)*4)
    i = np.arange(n)
    eri[i, i, i, i] = float(onsite_u)
    target = sites[-1].qns[2][0]
    owner = SU2MovingEnvironment(h1, eri, target.charge,
                                 two_s=target.irrep.two_j, cutoff=1e-12,
                                 include_half=True)
    owner.set_num_threads(num_workers() if n_threads is None else n_threads)
    mpo = build_su2_normal_complementary_mpo(owner, fully_reduced=True)
    for factor in mpo:
        object.__setattr__(factor, "normal_complementary_right_dual", True)
    return mpo


def normalized_energy(sites, mpo):
    """Fresh recursive reduced contractions, independent of sweep caches."""
    identity = _identity_mpo_factors_for_sites_and_mpo(sites, mpo)
    numerator = contract_chain_expectation(sites, mpo)
    norm = contract_chain_expectation(sites, identity)
    if norm.real <= 0 or abs(norm.imag) > 1e-10 * norm.real:
        raise ValueError(f"Invalid state norm: {norm}")
    energy = numerator / norm
    if abs(energy.imag) > 1e-9:
        raise ValueError(f"Nonreal energy: {energy}")
    return float(energy.real), float(norm.real)


def calculate(lx=6, ly=6, *, D=30, nsweeps=1, hopping=1., onsite_u=4.,
              mu=0., ordering="snake", periodic_x=False, periodic_y=False,
              seed=7, bond_multiplicity=2, cutoff=1e-10, n_threads=None,
              itermax=500, max_space=96, residual_tol=1e-7, verbose=2):
    """Run metric-aware reduced sweeps and verify energy/local residuals.

    Two-site SU(2) adaptation of White, PRL 69, 2863 (1992),
    doi:10.1103/PhysRevLett.69.2863. Finite multiplet truncation and local
    residual tolerances do not guarantee convergence to the ground state.
    """
    started = time.perf_counter()
    sites = build_random_reduced_spatial_mps(
        lx*ly, seed=seed, bond_multiplicity=bond_multiplicity,
        cap_multiplicity=True)
    # Initial-state compression deliberately truncates; canonical gauge moves
    # alone must preserve a state and therefore reject this use of a bond cap.
    for bond in range(len(sites)-2, -1, -1):
        left, right, _, _, _ = svd_two_site(
            merge_mps_sites(sites[bond], sites[bond+1]), max_bond=D,
            max_bond_mode="reduced", cutoff=cutoff, absorb="left")
        sites[bond:bond+2] = [left, right]
    print("Initial MPS compressed to the total multiplet budget", flush=True)
    mpo = build_2d_hubbard_mpo(
        sites, lx, ly, hopping_t=hopping, onsite_u=onsite_u,
        chemical_potential=mu, ordering=ordering, periodic_x=periodic_x,
        periodic_y=periodic_y, n_threads=n_threads)
    initial_energy, _ = normalized_energy(sites, mpo)
    owner = mpo[0].normal_complementary_owner
    setup_s = time.perf_counter() - started
    print(f"Setup {setup_s:.3f} s; initial E/t={initial_energy/hopping:.12f}", flush=True)
    tick = time.perf_counter()
    result = run_sweeps(
        sites, nsweeps=nsweeps, mpo_factors=mpo, max_bond=D,
        max_bond_mode="reduced", cutoff=cutoff, canonical_local_norm=False,
        prefer_reduced_local_operator=True, su2_moving_environment=owner,
        compute_final_expectation=True,
        local_solver_kwargs=dict(itermax=itermax, max_space=max_space,
                                 tol=1e-10, tol_residual=residual_tol,
                                 couple_physical=False),
        profile=True, verbose=verbose)
    sweep_s = time.perf_counter() - tick
    tick = time.perf_counter()
    energy, norm = normalized_energy(result["sites"], mpo)
    updates = [u for h in result["history"] for u in h["updates"]]
    objectives = [u["local_objective"] for u in updates]
    max_residual = max(float(o["residual"]) for o in objectives)
    failed = [i for i, o in enumerate(objectives)
              if not o.get("davidson_converged", False)
              or float(o["residual"]) > residual_tol]
    sweep_energy = float(result["history"][-1]["energy"])
    verification = dict(energy=energy, norm=norm, sweep_energy=sweep_energy,
                        energy_difference=abs(energy-sweep_energy),
                        max_local_residual=max_residual, failed_local_solves=failed,
                        max_kept=max(len(s.qns[2]) for s in result["sites"][:-1]))
    result.update(verification)
    result.update(setup_s=setup_s, sweep_s=sweep_s,
                  verification_s=time.perf_counter()-tick,
                  total_s=time.perf_counter()-started,
                  initial_energy=initial_energy, threading=owner.threading_info)
    result["validated"] = not failed and abs(energy-sweep_energy) < 1e-8
    return result


def save_results(result, output, parameters):
    """Save a restart state, scalar diagnostics, and reproducible figures."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "state.pkl").open("wb") as stream:
        pickle.dump(result["sites"], stream)
    history = []
    for entry in result["history"]:
        updates = []
        for u in entry["updates"]:
            o = u["local_objective"]
            updates.append(dict(bond=int(u["bond"]), energy=float(o["energy"]),
                residual=float(o["residual"]), converged=bool(o["davidson_converged"]),
                iterations=int(o["davidson_iterations"]), kept=u.get("kept"),
                trunc_err=float(u["trunc_err"])))
        history.append(dict(direction=entry["direction"], energy=float(entry["energy"]),
                            energy_source=entry.get("energy_source"), updates=updates))
    summary = {k: result[k] for k in (
        "energy", "norm", "sweep_energy", "energy_difference", "initial_energy",
        "max_local_residual", "failed_local_solves", "max_kept", "validated",
        "setup_s", "sweep_s", "verification_s", "total_s", "threading", "converged")}
    summary.update(parameters=parameters, history=history)
    (output / "results.json").write_text(json.dumps(summary, indent=2))
    plot_results(output)
    return summary


def plot_results(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads((output / "results.json").read_text())
    rows = [u for h in data["history"] for u in h["updates"]]
    x = np.arange(1, len(rows)+1)
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(x, [u["energy"] for u in rows], "o-", ms=3, label="Local objective")
    axes[0].axhline(data["energy"], ls="--", color="gray", label="Verified final energy")
    axes[0].set_ylabel("Energy"); axes[0].legend(fontsize=8)
    axes[1].semilogy(x, np.maximum([u["residual"] for u in rows], 1e-16), "o-", ms=3)
    axes[1].set_ylabel("Local residual")
    axes[2].plot(x, [u["trunc_err"] for u in rows], "o-", ms=3)
    axes[2].set(xlabel="Bond update", ylabel="Discarded weight")
    p = data["parameters"]
    fig.suptitle(f"{p['lx']}×{p['ly']} SU(2) Hubbard, U/t={p['onsite_u']/p['hopping']:g}, "
                 f"total D={p['D']}\nSweep {data['sweep_s']:.2f} s; "
                 f"validated={data['validated']}; ground state not converged")
    fig.tight_layout()
    fig.savefig(output / "hubbard_su2_energy_residuals.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx", type=int, default=4)
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("-t", "--hopping", type=float, default=1.)
    parser.add_argument("-U", "--onsite-u", type=float, default=4.)
    parser.add_argument("--mu", type=float, default=0.)
    parser.add_argument("--periodic-x", action="store_true")
    parser.add_argument("--periodic-y", action="store_true")
    parser.add_argument("--ordering", choices=("row-major", "snake"), default="snake")
    parser.add_argument("--max-bond", dest="D", type=int, default=30)
    parser.add_argument("--bond-multiplicity", type=int, default=2)
    parser.add_argument("--nsweeps", type=int, default=1)
    parser.add_argument("--cutoff", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-threads", type=int)
    parser.add_argument("--itermax", type=int, default=500)
    parser.add_argument("--max-space", type=int, default=96)
    parser.add_argument("--residual-tol", type=float, default=1e-7)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/hubbard_su2_2d"))
    args = vars(parser.parse_args())
    output = args.pop("output")
    result = calculate(**args)
    summary = save_results(result, output, args)
    print(json.dumps({k:v for k,v in summary.items() if k != "history"}, indent=2), flush=True)
    if not result["validated"]:
        raise RuntimeError("Calculation failed local residual or final energy validation.")


if __name__ == "__main__":
    main()
