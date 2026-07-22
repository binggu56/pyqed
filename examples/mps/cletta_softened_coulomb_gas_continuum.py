"""Neutralized 1D softened-Coulomb/Yukawa gas with continuum cMPS/cLETTA.

The bare infinite uniform Coulomb Hartree term diverges.  This example uses a
neutral-background/connected interaction energy,

    int_0^inf dr V(r) ( < :n(r)n(0): > - n^2 ),

with a softened screened-Coulomb kernel

    V(r) = g exp(-kappa r) / sqrt(r^2 + a^2).

The kernel is compressed by a nonlinear exponential fit and the wavefunction
memory poles remain independent variational cLETTA parameters.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pyqed.mps import (
    ContinuousMPS,
    fit_exponential_kernel_nonlinear,
    pack_canonical_parameters,
    softened_yukawa_kernel,
)


def _product_state(density):
    theta = pack_canonical_parameters([], np.array([[np.sqrt(float(density))]]))
    return ContinuousMPS.from_canonical_parameters(theta, bond_dim=1)


def _apply_values(state, values):
    state.energy = values["energy_density"]
    state.density = values["density"]
    state.kinetic = values["kinetic"]
    state.interaction = values["interaction"]
    state.raw_density = values["raw_density"]
    state.scale = values["scale"]
    return state


def _row(label, state, *, cutoff="", contraction_backend=""):
    row = {
        "ansatz": label,
        "cutoff": cutoff,
        "contraction_backend": contraction_backend,
        "energy": float(state.energy),
        "kinetic": float(state.kinetic),
        "interaction": float(state.interaction),
        "raw_density": float(state.raw_density),
        "scale": float(state.scale),
        "success": bool(state.success) if state.success is not None else True,
        "nfev": int(state.nfev),
        "memory_rates": "",
        "memory_frequencies": "",
        "tie_norm": "",
    }
    if state.cletta_decay_rates is not None:
        row["memory_rates"] = ";".join(f"{value:.12g}" for value in state.cletta_decay_rates)
    if state.cletta_frequencies is not None:
        row["memory_frequencies"] = ";".join(f"{value:.12g}" for value in state.cletta_frequencies)
    if state.cletta_tie_matrices is not None:
        row["tie_norm"] = float(np.linalg.norm(state.cletta_tie_matrices))
    return row


def run(args):
    distances = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, args.fit_near_max, args.fit_near_points),
                np.geomspace(
                    args.fit_near_max / (args.fit_near_points - 1),
                    args.fit_rmax,
                    args.fit_tail_points,
                ),
            ]
        )
    )
    target = softened_yukawa_kernel(
        distances,
        strength=args.strength,
        screening=args.screening,
        softening=args.softening,
    )
    fit = fit_exponential_kernel_nonlinear(
        distances,
        target,
        rank=args.fit_rank,
        starts=args.fit_starts,
        max_nfev=args.fit_max_nfev,
        rate_offset=args.screening,
    )
    rates = fit["decay_rates"]
    strengths = fit["strengths"]

    print(
        f"soft screened Coulomb kappa={args.screening:g} "
        f"nonlinear rank={args.fit_rank} terms={len(rates)} "
        f"rel_error={fit['rel_error']:.3e} max_abs={fit['max_abs_error']:.3e} "
        f"rms_rel={fit['relative_rms_error']:.3e} "
        f"max_rel={fit['max_rel_error']:.3e}"
    )
    print("Hamiltonian decays:", " ".join(f"{value:.8g}" for value in rates))
    print("Hamiltonian weights:", " ".join(f"{value:.8g}" for value in strengths))

    rows = []
    product = _product_state(args.density)
    product_values = product.exponential_bose_gas_fixed_density_observables(
        decay_rates=rates,
        strengths=strengths,
        density=args.density,
        connected=True,
    )
    _apply_values(product, product_values)
    rows.append(_row("product-neutral", product))

    cmps_states = {}
    for index, bond_dim in enumerate(dict.fromkeys(args.cmps_bond_dims)):
        cmps = ContinuousMPS.optimize_exponential_bose_gas_fixed_density(
            bond_dim=bond_dim,
            decay_rates=rates,
            strengths=strengths,
            density=args.density,
            connected=True,
            restarts=args.cmps_restarts,
            seed=args.seed + 100 + 17 * index,
            maxiter=args.cmps_maxiter,
            regularization=args.regularization,
            density_gauge_penalty=args.density_gauge_penalty,
        )
        cmps_states[int(bond_dim)] = cmps
        rows.append(_row(f"cMPS-D{bond_dim}", cmps))

    cutoffs = args.cutoffs if args.cutoffs else [args.cutoff]
    seed_parameters = []
    for cutoff in cutoffs:
        cletta = ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
            bond_dim=args.bond_dim,
            interaction_decay_rates=rates,
            strengths=strengths,
            density=args.density,
            connected=True,
            num_modes=args.num_modes,
            depth=cutoff,
            seed_parameters=seed_parameters,
            seed_base_thetas=(
                [cmps_states[args.bond_dim].theta]
                if args.seed_from_cmps and args.bond_dim in cmps_states
                else ()
            ),
            restarts=args.restarts,
            seed=args.seed + 17 * int(cutoff),
            maxiter=args.maxiter,
            regularization=args.cletta_regularization,
            density_gauge_penalty=args.density_gauge_penalty,
            rate_bounds=(args.min_memory_decay, args.max_memory_decay),
            frequency_bounds=(-args.max_memory_frequency, args.max_memory_frequency),
            tie_scale=args.tie_scale,
        )
        if args.contraction_backend != "explicit":
            hierarchy_values = cletta.cletta_base.cletta_exponential_bose_gas_fixed_density_observables(
                cletta.cletta_tie_matrices,
                cletta.cletta_decay_rates,
                interaction_decay_rates=rates,
                strengths=strengths,
                density=args.density,
                depth=cutoff,
                frequencies=cletta.cletta_frequencies,
                connected=True,
                contraction_backend=args.contraction_backend,
            )
            _apply_values(cletta, hierarchy_values)
        seed_parameters = [cletta.cletta_parameters]
        rows.append(
            _row(
                f"cLETTA-D{args.bond_dim}-M{args.num_modes}",
                cletta,
                cutoff=cutoff,
                contraction_backend=args.contraction_backend,
            )
        )

    for row in rows:
        cutoff = f" L={row['cutoff']}" if row["cutoff"] != "" else ""
        print(
            f"{row['ansatz']:>22s}{cutoff:>8s} "
            f"E={row['energy']:.10f} T={row['kinetic']:.10f} "
            f"Vconn={row['interaction']:.10f} success={row['success']}"
        )
        if row["contraction_backend"]:
            print(f"  contraction={row['contraction_backend']}")
        if row["memory_rates"]:
            print(f"  memory rates=[{row['memory_rates']}]")
            print(f"  memory frequencies=[{row['memory_frequencies']}] tie_norm={row['tie_norm']}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        keys = list(rows[0].keys())
        fit_keys = [
            "fit_rel_error",
            "fit_max_abs_error",
            "fit_max_rel_error",
            "fit_relative_rms_error",
            "fit_decays",
            "fit_strengths",
        ]
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys + fit_keys)
            writer.writeheader()
            for row in rows:
                out = dict(row)
                out["fit_rel_error"] = fit["rel_error"]
                out["fit_max_abs_error"] = fit["max_abs_error"]
                out["fit_max_rel_error"] = fit["max_rel_error"]
                out["fit_relative_rms_error"] = fit["relative_rms_error"]
                out["fit_decays"] = ";".join(f"{value:.12g}" for value in rates)
                out["fit_strengths"] = ";".join(f"{value:.12g}" for value in strengths)
                writer.writerow(out)
        print(f"wrote {output}")

    if args.figure:
        import matplotlib.pyplot as plt

        figure = Path(args.figure)
        figure.parent.mkdir(parents=True, exist_ok=True)
        labels = [
            row["ansatz"] if row["cutoff"] == "" else f"{row['ansatz']} L={row['cutoff']}"
            for row in rows
        ]
        energy = np.array([row["energy"] for row in rows], dtype=float)
        kinetic = np.array([row["kinetic"] for row in rows], dtype=float)
        interaction = np.array([row["interaction"] for row in rows], dtype=float)
        x = np.arange(len(rows))

        fig, (ax_energy, ax_fit) = plt.subplots(1, 2, figsize=(9.4, 3.8))
        ax_energy.bar(x, kinetic, label="kinetic", color="#cc6677")
        ax_energy.bar(x, interaction, bottom=kinetic, label="connected interaction", color="#4477aa")
        ax_energy.plot(x, energy, "ko-", lw=1, ms=4)
        ax_energy.axhline(0.0, color="black", lw=0.8)
        ax_energy.set_xticks(x, labels, rotation=18, ha="right")
        ax_energy.set_ylabel("neutral energy density")
        ax_energy.grid(axis="y", alpha=0.25)
        ax_energy.legend(frameon=False)

        rgrid = np.unique(
            np.concatenate(
                [np.linspace(0.0, args.fit_near_max, 250), np.geomspace(0.02, args.fit_rmax, 500)]
            )
        )
        exact = softened_yukawa_kernel(
            rgrid,
            strength=args.strength,
            screening=args.screening,
            softening=args.softening,
        )
        fitted = np.exp(-rgrid[:, None] * rates[None, :]) @ strengths
        ax_fit.plot(rgrid, exact, label="soft screened Coulomb", color="#228833")
        ax_fit.plot(rgrid, fitted, "--", label="nonlinear fit", color="#aa3377")
        ax_fit.set_xlabel("r")
        ax_fit.set_ylabel("V(r)")
        ax_fit.set_title(f"Fit to r={args.fit_rmax:g}, max rel. error {fit['max_rel_error']:.1e}")
        ax_fit.set_xscale("symlog", linthresh=0.5)
        ax_fit.grid(alpha=0.25)
        ax_fit.legend(frameon=False)

        fig.tight_layout()
        fig.savefig(figure, dpi=200)
        print(f"wrote {figure}")

    return rows, fit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--softening", type=float, default=0.5)
    parser.add_argument("--screening", type=float, default=0.0)
    parser.add_argument("--fit-near-max", type=float, default=2.0)
    parser.add_argument("--fit-near-points", type=int, default=101)
    parser.add_argument("--fit-rmax", type=float, default=64.0)
    parser.add_argument("--fit-tail-points", type=int, default=240)
    parser.add_argument("--fit-rank", type=int, default=12)
    parser.add_argument("--fit-starts", type=int, default=1)
    parser.add_argument("--fit-max-nfev", type=int, default=3000)
    parser.add_argument("--cmps-bond-dims", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--cmps-restarts", type=int, default=3)
    parser.add_argument("--cmps-maxiter", type=int, default=180)
    parser.add_argument("--bond-dim", type=int, default=2)
    parser.add_argument("--num-modes", type=int, default=2)
    parser.add_argument("--cutoff", type=int, default=1)
    parser.add_argument("--cutoffs", type=int, nargs="*", default=None)
    parser.add_argument(
        "--contraction-backend",
        choices=("explicit", "hierarchy", "heom"),
        default="hierarchy",
    )
    parser.add_argument("--seed-from-cmps", action="store_true", default=True)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--regularization", type=float, default=1.0e-10)
    parser.add_argument("--cletta-regularization", type=float, default=1.0e-6)
    parser.add_argument("--density-gauge-penalty", type=float, default=1.0e-3)
    parser.add_argument("--tie-scale", type=float, default=0.001)
    parser.add_argument("--min-memory-decay", type=float, default=0.02)
    parser.add_argument("--max-memory-decay", type=float, default=5.0)
    parser.add_argument("--max-memory-frequency", type=float, default=5.0)
    parser.add_argument(
        "--output",
        default="/private/tmp/cletta_softened_coulomb_gas_continuum.csv",
    )
    parser.add_argument(
        "--figure",
        default="/private/tmp/cletta_softened_coulomb_gas_continuum.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
