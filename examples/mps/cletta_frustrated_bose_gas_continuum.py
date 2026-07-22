"""Pseudomode cLETTA pilot for a frustrated continuum Bose gas.

The infinite-line Hamiltonian density is

    H/L = <d psi^dag d psi> + g <psi^dag psi^dag psi psi>
          + int_0^inf dr V(r) < :n(r)n(0): >,

with a short-range-attractive/long-range-repulsive (SALR) kernel

    V(r) = sum_j w_j exp(-kappa_j r).

Its full-line Fourier transform is a compact sum of poles,

    V_tilde(k) = 2 sum_j w_j kappa_j / (kappa_j^2 + k^2).

The defaults give a quadratic response ``k^2 + rho V_tilde(k)`` whose
minimum occurs at nonzero momentum.  A conjugate pseudomode pair is initialized
at poles ``gamma +/- i k_star`` and optimized at successive total-occupation
cutoffs L.  This realizes a Brazovskii-like finite-wavevector tendency without
putting a fourth spatial derivative into a generic cMPS ansatz.
"""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path

import numpy as np

from pyqed.mps import ContinuousMPS, canonical_parameter_size, pack_canonical_parameters


def product_state(density):
    theta = pack_canonical_parameters([], np.array([[np.sqrt(float(density))]]))
    return ContinuousMPS.from_canonical_parameters(theta, bond_dim=1)


def real_space_kernel(distance, decay_rates, strengths):
    distance = np.asarray(distance, dtype=float)
    rates = np.asarray(decay_rates, dtype=float)
    weights = np.asarray(strengths, dtype=float)
    return np.sum(
        weights.reshape((-1,) + (1,) * distance.ndim)
        * np.exp(-rates.reshape((-1,) + (1,) * distance.ndim) * distance),
        axis=0,
    )


def interaction_spectrum(momentum, decay_rates, strengths):
    """Return the full-line Fourier transform of the exponential kernel."""
    momentum = np.asarray(momentum, dtype=float)
    rates = np.asarray(decay_rates, dtype=float)
    weights = np.asarray(strengths, dtype=float)
    shape = (-1,) + (1,) * momentum.ndim
    return 2.0 * np.sum(
        (weights * rates).reshape(shape)
        / (rates.reshape(shape) ** 2 + momentum**2),
        axis=0,
    )


def quadratic_spectrum(momentum, decay_rates, strengths, density=1.0):
    """Mean-field quadratic response used only to locate the soft momentum."""
    momentum = np.asarray(momentum, dtype=float)
    return momentum**2 + float(density) * interaction_spectrum(
        momentum, decay_rates, strengths
    )


def preferred_wavevector(decay_rates, strengths, density=1.0, momentum_max=None):
    """Locate the global minimum of the quadratic response on ``k >= 0``."""
    from scipy.optimize import minimize_scalar

    rates = np.asarray(decay_rates, dtype=float)
    if momentum_max is None:
        momentum_max = 8.0 * max(float(np.max(rates)), float(density), 1.0)
    objective = lambda k: float(quadratic_spectrum(k, rates, strengths, density))
    result = minimize_scalar(objective, bounds=(0.0, float(momentum_max)), method="bounded")
    candidates = [(0.0, objective(0.0)), (float(momentum_max), objective(momentum_max))]
    if result.success:
        candidates.append((float(result.x), float(result.fun)))
    return min(candidates, key=lambda item: item[1])


def connected_density_correlation(state, distances):
    """Evaluate the fixed-density correlator for a possibly noncanonical state."""
    distances = np.atleast_1d(np.asarray(distances, dtype=float))
    left, right, dominant = state.dominant_fixed_points()
    transfer = state.transfer_matrix()
    insertion = np.kron(state.r, state.r.conj())
    density = np.real_if_close(np.vdot(left, insertion @ right))
    scale = float(state.scale)
    shifted = scale * (transfer - dominant * np.eye(transfer.shape[0]))
    eigenvalues, eigenvectors = np.linalg.eig(shifted)
    coefficients = np.linalg.solve(eigenvectors, insertion @ right)
    values = []
    for distance in distances:
        evolved = eigenvectors @ (np.exp(eigenvalues * distance) * coefficients)
        value = np.vdot(left, insertion @ evolved) - density * density
        values.append(scale**2 * value)
    return np.real_if_close(np.asarray(values))


def attenuated_memory_seeds(parameters, bond_dim):
    """Seed a deeper cutoff between the previous tie and the zero-tie state."""
    parameters = np.asarray(parameters, dtype=float)
    base_size = canonical_parameter_size(int(bond_dim))
    tie_slice = slice(base_size, base_size + int(bond_dim) ** 2)
    seeds = []
    for factor in (1.0, 0.75, 0.5, 0.25, 0.0):
        seed = np.array(parameters, copy=True)
        seed[tie_slice] *= factor
        seeds.append(seed)
    return seeds


def noisy_cmps_memory_seeds(
    base_theta,
    bond_dim,
    *,
    count,
    noise,
    rng,
    memory_rate,
    memory_frequency,
    optimize_memory_poles=False,
):
    """Lift a cMPS optimum to cLETTA and weakly break the zero-memory seed."""
    base_theta = np.asarray(base_theta, dtype=float)
    bond_dim = int(bond_dim)
    seeds = []
    for _ in range(int(count)):
        noisy_base = base_theta + float(noise) * rng.normal(size=base_theta.size)
        noisy_tie = float(noise) * rng.normal(size=bond_dim * bond_dim)
        pieces = [noisy_base, noisy_tie]
        if optimize_memory_poles:
            pieces.extend(
                [
                    np.array([np.log(float(memory_rate))]),
                    np.array([float(memory_frequency)]),
                ]
            )
        seeds.append(np.concatenate(pieces))
    return seeds


def populate_product_observables(state, values):
    for name in (
        "energy_density",
        "density",
        "kinetic",
        "contact",
        "interaction",
        "raw_density",
        "scale",
    ):
        attribute = "energy" if name == "energy_density" else name
        setattr(state, attribute, values[name])
    state.success = True
    state.nfev = 0


def state_row(label, state, *, cutoff="", base_bond_dim="", num_modes=""):
    row = {
        "ansatz": label,
        "cutoff_L": cutoff,
        "base_bond_dim": base_bond_dim,
        "memory_modes": num_modes,
        "memory_dim": "",
        "effective_bond_dim": int(state.bond_dim),
        "parameter_count": "",
        "energy": float(state.energy),
        "kinetic": float(state.kinetic),
        "contact": float(state.contact),
        "interaction": float(state.interaction),
        "raw_density": float(state.raw_density),
        "scale": float(state.scale),
        "success": bool(state.success) if state.success is not None else True,
        "nfev": int(state.nfev),
        "memory_rate": "",
        "memory_frequency": "",
        "tie_norm": "",
    }
    if cutoff != "":
        memory_dim = comb(int(num_modes) + int(cutoff), int(cutoff))
        row["memory_dim"] = memory_dim
        row["parameter_count"] = int(np.asarray(state.cletta_parameters).size)
        row["memory_rate"] = float(state.cletta_decay_rates[0])
        row["memory_frequency"] = abs(float(state.cletta_frequencies[0]))
        row["tie_norm"] = float(np.linalg.norm(state.cletta_tie_matrices))
    elif label.startswith("cMPS"):
        row["parameter_count"] = int(state.canonical_parameters().size)
    return row


def run(args):
    rates = np.asarray(args.decay_rates, dtype=float)
    strengths = np.asarray(args.strengths, dtype=float)
    if rates.shape != strengths.shape:
        raise ValueError("--decay-rates and --strengths must have the same length.")
    if np.any(rates <= 0.0) or np.any(~np.isfinite(rates)):
        raise ValueError("all decay rates must be finite and positive.")
    if float(args.contact_coupling) < 0.0:
        raise ValueError("--contact-coupling must be nonnegative for this stable pilot.")

    k_star, response_minimum = preferred_wavevector(
        rates, strengths, density=args.density, momentum_max=args.momentum_max
    )
    if k_star <= 1.0e-6:
        raise ValueError(
            "the chosen kernel has no resolved finite-wavevector response minimum; "
            "choose competing strengths or decay rates."
        )
    memory_rate = (
        float(args.memory_decay_rate)
        if args.memory_decay_rate is not None
        else max(0.35 * k_star, 1.0e-3)
    )
    memory_frequencies = np.array([k_star, -k_star])
    print(
        f"soft momentum k*={k_star:.8f}, wavelength={2.0 * np.pi / k_star:.8f}, "
        f"quadratic minimum={response_minimum:.8f}"
    )
    print(f"initial pseudomode poles: {memory_rate:.8f} +/- i {k_star:.8f}")

    rows = []
    states = []
    product = product_state(args.density)
    values = product.exponential_bose_gas_fixed_density_observables(
        decay_rates=rates,
        strengths=strengths,
        density=args.density,
        contact_coupling=args.contact_coupling,
    )
    populate_product_observables(product, values)
    rows.append(state_row("product", product))
    states.append(("product", product))

    cmps_base_seeds = []
    for index, bond_dim in enumerate(args.cmps_bond_dims):
        if int(bond_dim) < 1:
            continue
        state = ContinuousMPS.optimize_exponential_bose_gas_fixed_density(
            bond_dim=int(bond_dim),
            decay_rates=rates,
            strengths=strengths,
            density=args.density,
            contact_coupling=args.contact_coupling,
            restarts=args.cmps_restarts,
            seed=args.seed + 100 + index,
            maxiter=args.cmps_maxiter,
            regularization=args.regularization,
            density_gauge_penalty=args.density_gauge_penalty,
        )
        label = f"cMPS-D{int(bond_dim)}"
        rows.append(state_row(label, state))
        states.append((label, state))
        if int(bond_dim) == int(args.bond_dim):
            cmps_base_seeds.append(np.array(state.theta, copy=True))

    previous_parameters = None
    for index, cutoff in enumerate(sorted(set(int(value) for value in args.cutoffs))):
        if cutoff < 1:
            raise ValueError("all cLETTA cutoffs must be at least one.")
        seeds = (
            []
            if previous_parameters is None
            else attenuated_memory_seeds(previous_parameters, args.bond_dim)
        )
        if cutoff == int(args.cmps_noise_cutoff):
            rng = np.random.default_rng(args.seed + 1000 + index)
            for base_theta in cmps_base_seeds:
                seeds.extend(
                    noisy_cmps_memory_seeds(
                        base_theta,
                        args.bond_dim,
                        count=args.cmps_noise_seeds,
                        noise=args.cmps_seed_noise,
                        rng=rng,
                        memory_rate=memory_rate,
                        memory_frequency=k_star,
                        optimize_memory_poles=args.optimize_memory_poles,
                    )
                )
            if cmps_base_seeds and args.cmps_noise_seeds:
                print(
                    f"L={cutoff}: added {len(cmps_base_seeds) * args.cmps_noise_seeds} "
                    f"noisy cMPS seeds (sigma={args.cmps_seed_noise:g})"
                )
        contraction_backend = args.contraction_backend
        if contraction_backend == "auto":
            contraction_backend = "hierarchy_iterative" if cutoff >= 4 else "explicit"
        if contraction_backend == "hierarchy_iterative":
            print(f"L={cutoff}: using matrix-free two-sided HEOM contraction")
        state = ContinuousMPS.optimize_exponential_bose_gas_cletta_fixed_density(
            bond_dim=args.bond_dim,
            interaction_decay_rates=rates,
            strengths=strengths,
            density=args.density,
            contact_coupling=args.contact_coupling,
            num_modes=2,
            depth=cutoff,
            memory_decay_rates=[memory_rate, memory_rate],
            memory_frequencies=memory_frequencies,
            optimize_memory_rates=args.optimize_memory_poles,
            optimize_memory_frequencies=args.optimize_memory_poles,
            conjugate_pair=True,
            seed_parameters=seeds,
            seed_base_thetas=([] if args.skip_zero_cmps_seed else cmps_base_seeds),
            restarts=max(
                args.restarts,
                len(seeds) + (0 if args.skip_zero_cmps_seed else len(cmps_base_seeds)),
            ),
            seed=args.seed + index,
            maxiter=args.maxiter,
            regularization=args.regularization,
            density_gauge_penalty=args.density_gauge_penalty,
            tie_scale=args.tie_scale,
            use_jax=not args.no_jax,
            frequency_bounds=(-args.frequency_bound, args.frequency_bound),
            contraction_backend=contraction_backend,
            iterative_tolerance=args.iterative_tolerance,
            iterative_maxiter=args.iterative_maxiter,
        )
        previous_parameters = np.array(state.cletta_parameters, copy=True)
        label = f"cLETTA-D{args.bond_dim}-M2-L{cutoff}"
        rows.append(
            state_row(
                label,
                state,
                cutoff=cutoff,
                base_bond_dim=args.bond_dim,
                num_modes=2,
            )
        )
        states.append((label, state))

    previous_energy = None
    for row in rows:
        delta = ""
        if row["cutoff_L"] != "":
            if previous_energy is not None:
                delta = f" dE_L={row['energy'] - previous_energy:+.3e}"
            previous_energy = row["energy"]
        memory = (
            f" L={row['cutoff_L']} dmem={row['memory_dim']}"
            if row["cutoff_L"] != ""
            else ""
        )
        print(
            f"{row['ansatz']:>18s}{memory:>14s} E={row['energy']:.10f} "
            f"T={row['kinetic']:.7f} V={row['interaction']:.7f} "
            f"C={row['contact']:.7f}{delta}"
        )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        extra = {
            "soft_wavevector": k_star,
            "soft_wavelength": 2.0 * np.pi / k_star,
            "quadratic_minimum": response_minimum,
            "contact_coupling": args.contact_coupling,
        }
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) + list(extra))
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, **extra})
        print(f"wrote {output}")

    if args.figure:
        import matplotlib.pyplot as plt

        figure = Path(args.figure)
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))

        momentum = np.linspace(0.0, args.momentum_max, 600)
        axes[0].plot(
            momentum,
            quadratic_spectrum(momentum, rates, strengths, args.density),
            color="#4477aa",
        )
        axes[0].axvline(k_star, color="#cc6677", ls="--", label=rf"$k_\star={k_star:.2f}$")
        axes[0].set(xlabel=r"$k$", ylabel=r"$k^2+\rho\widetilde V(k)$")
        axes[0].legend(frameon=False)

        cletta_rows = [row for row in rows if row["cutoff_L"] != ""]
        axes[1].plot(
            [row["cutoff_L"] for row in cletta_rows],
            [row["energy"] for row in cletta_rows],
            "o-",
            color="#228833",
            label="cLETTA",
        )
        for row in rows:
            if str(row["ansatz"]).startswith("cMPS"):
                axes[1].axhline(row["energy"], ls="--", lw=1, label=row["ansatz"])
        axes[1].set(xlabel=r"pseudomode cutoff $L$", ylabel="energy density")
        axes[1].legend(frameon=False, fontsize=8)

        distance = np.linspace(0.0, args.correlation_range, args.correlation_points)
        for label, state in states[1:]:
            correlation = connected_density_correlation(state, distance)
            axes[2].plot(distance, np.real(correlation), label=label)
        axes[2].axvline(2.0 * np.pi / k_star, color="0.5", ls=":", lw=1)
        axes[2].set(xlabel=r"$x$", ylabel=r"$\langle n(x)n(0)\rangle_c$")
        axes[2].legend(frameon=False, fontsize=7)

        for axis in axes:
            axis.grid(alpha=0.22)
        fig.tight_layout()
        fig.savefig(figure, dpi=200)
        print(f"wrote {figure}")

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--decay-rates", nargs="+", type=float, default=[0.45, 2.2])
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.8, -1.6])
    parser.add_argument("--contact-coupling", type=float, default=0.5)
    parser.add_argument("--cmps-bond-dims", nargs="*", type=int, default=[2, 3])
    parser.add_argument("--cmps-restarts", type=int, default=4)
    parser.add_argument("--cmps-maxiter", type=int, default=240)
    parser.add_argument("--bond-dim", type=int, default=2)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--cmps-noise-cutoff",
        type=int,
        default=3,
        help="cutoff receiving direct noisy seeds from the optimized matching-D cMPS",
    )
    parser.add_argument("--cmps-noise-seeds", type=int, default=8)
    parser.add_argument("--cmps-seed-noise", type=float, default=0.03)
    parser.add_argument(
        "--skip-zero-cmps-seed",
        action="store_true",
        help="omit the exact zero-tie fallback when direct noisy cMPS seeds are present",
    )
    parser.add_argument("--memory-decay-rate", type=float, default=None)
    parser.add_argument(
        "--optimize-memory-poles",
        action="store_true",
        help="vary gamma and omega; by default the finite-k poles remain fixed",
    )
    parser.add_argument("--frequency-bound", type=float, default=8.0)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=260)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--regularization", type=float, default=1.0e-10)
    parser.add_argument("--density-gauge-penalty", type=float, default=1.0e-4)
    parser.add_argument("--tie-scale", type=float, default=0.04)
    parser.add_argument("--no-jax", action="store_true")
    parser.add_argument(
        "--contraction-backend",
        choices=["auto", "explicit", "hierarchy_iterative"],
        default="auto",
        help="auto selects matrix-free two-sided HEOM at L >= 4",
    )
    parser.add_argument("--iterative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--iterative-maxiter", type=int, default=None)
    parser.add_argument("--momentum-max", type=float, default=3.0)
    parser.add_argument("--correlation-range", type=float, default=16.0)
    parser.add_argument("--correlation-points", type=int, default=240)
    parser.add_argument(
        "--output", default="/private/tmp/cletta_frustrated_bose_gas_continuum.csv"
    )
    parser.add_argument(
        "--figure", default="/private/tmp/cletta_frustrated_bose_gas_continuum.png"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
