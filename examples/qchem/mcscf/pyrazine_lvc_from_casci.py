#!/usr/bin/env python3
"""Build a two-mode pyrazine LVC model from native CASCI derivatives.

This is a pipeline benchmark for ``LVC.from_casci()`` rather than a strict
parameter reproduction.  The literature two-mode pyrazine model uses diabatic
states and fitted normal modes; this script uses the current ab-initio CASCI
states and either quick guessed modes or normal modes selected by nearest RHF
frequency.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pyqed.models.pyrazine import pyrazine_2mode_lvc
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casci import CASCI
from pyqed.units import amu_to_au, au2wavenumber, wavenumber2hartree


PYRAZINE_GEOMETRY_BOHR = [
    ["N", 0.0000000000, 0.0000046126, 2.9751681209],
    ["C", 0.0000000000, 2.0213606485, 1.3447521663],
    ["C", 0.0000000000, 2.0213594563, -1.3447637764],
    ["N", 0.0000000000, -0.0000049244, -2.9751696399],
    ["C", 0.0000000000, -2.0213693403, -1.3447570196],
    ["C", 0.0000000000, -2.0213627060, 1.3447652675],
    ["H", 0.0000000000, 3.8979353927, 2.1970440670],
    ["H", 0.0000000000, 3.8979280273, -2.1970658170],
    ["H", 0.0000000000, -3.8979425319, -2.1970514056],
    ["H", 0.0000000000, -3.8979294535, 2.1970704549],
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare a CASCI-derived pyrazine LVC against the standard two-mode model."
    )
    parser.add_argument("--basis", default="sto-3g", help="AO basis for RHF, Hessian, and CASCI.")
    parser.add_argument(
        "--driver",
        default="gbasis",
        choices=("gbasis", "builtin"),
        help="Integral backend for the RHF/CASCI derivative calculation.",
    )
    parser.add_argument(
        "--reference-geometry",
        default="pyscf-casscf-opt",
        choices=("input", "pyscf-rhf-opt", "pyscf-casscf-opt"),
        help="Reference geometry used before building modes and the CASCI LVC.",
    )
    parser.add_argument(
        "--rhf-opt-maxiter",
        type=int,
        default=20,
        help="Maximum SciPy BFGS iterations for PySCF reference optimizations.",
    )
    parser.add_argument(
        "--rhf-opt-gtol",
        type=float,
        default=1.0e-3,
        help="Cartesian gradient convergence threshold for PySCF reference optimizations.",
    )
    parser.add_argument("--ncas", type=int, default=4, help="Number of active orbitals.")
    parser.add_argument("--nelecas", type=int, default=4, help="Number of active electrons.")
    parser.add_argument("--nstates", type=int, default=3, help="Number of CASCI roots.")
    parser.add_argument(
        "--mc-method",
        default="casscf",
        choices=("casci", "casscf"),
        help="Use RHF-orbital CASCI or optimize orbitals with PyQED CASSCF before LVC construction.",
    )
    parser.add_argument(
        "--target-frequencies",
        type=float,
        nargs=2,
        default=(952.0, 597.0),
        metavar=("COUPLING_CM1", "TUNING_CM1"),
        help="Literature frequencies used to pick the two nearest RHF modes.",
    )
    parser.add_argument(
        "--eri",
        default="factors",
        choices=("dense", "s4", "s8", "direct", "factors", "ri", "auto"),
        help="Native ERI representation.",
    )
    parser.add_argument(
        "--mode-source",
        default="guess",
        choices=("guess", "hessian"),
        help=(
            "Use quick symmetry-adapted guessed modes, or compute RHF Hessian "
            "modes and pick the nearest frequencies."
        ),
    )
    parser.add_argument(
        "--stationarize-s0",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move the reference along the selected modes so S0 has near-zero projected slope.",
    )
    parser.add_argument(
        "--subtract-common-s0-slope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subtract the raw S0 diagonal slope from all diagonal electronic LVC terms.",
    )
    parser.add_argument(
        "--stationary-iterations",
        type=int,
        default=1,
        help="Newton iterations used by --stationarize-s0.",
    )
    parser.add_argument(
        "--stationary-hessian",
        default="fd",
        choices=("fd", "analytic"),
        help="Jacobian used for S0 stationarization: finite-difference slopes or analytic G block.",
    )
    parser.add_argument(
        "--stationary-fd-step",
        type=float,
        default=0.02,
        help="Dimensionless normal-coordinate step for finite-difference slope Jacobians.",
    )
    parser.add_argument(
        "--max-stationary-step",
        type=float,
        default=0.25,
        help="Maximum dimensionless normal-coordinate Newton step per iteration.",
    )
    return parser.parse_args()


def build_pyrazine(basis, driver, eri, coords=None):
    atom = PYRAZINE_GEOMETRY_BOHR
    if coords is not None:
        coords = np.asarray(coords, dtype=float)
        atom = [
            [row[0], float(x), float(y), float(z)]
            for row, (x, y, z) in zip(PYRAZINE_GEOMETRY_BOHR, coords)
        ]
    mol = Molecule(atom=atom, unit="bohr", basis=basis)
    if driver == "builtin":
        mol.build(driver="builtin", eri=eri)
    else:
        mol.build(driver=driver)
    return mol


def run_reference(coords, args):
    mol = build_pyrazine(args.basis, args.driver, args.eri, coords=coords)
    mf = mol.RHF().run()
    if args.mc_method == "casscf":
        from pyqed.qchem.mcscf.casscf import CASSCF

        casscf = CASSCF(mf, ncas=args.ncas, nelecas=args.nelecas).run(nstates=args.nstates)
        mc = casscf.casci
    else:
        mc = CASCI(mf, ncas=args.ncas, nelecas=args.nelecas).run(nstates=args.nstates)
    return mol, mf, mc


def optimize_reference_geometry_pyscf(coords, args):
    try:
        from pyscf import gto, grad, mcscf, scf
    except ImportError as exc:
        raise ImportError(
            "--reference-geometry pyscf-rhf-opt/pyscf-casscf-opt requires PySCF. "
            "Use --reference-geometry input to skip this step."
        ) from exc
    from scipy.optimize import minimize

    symbols = [row[0] for row in PYRAZINE_GEOMETRY_BOHR]
    x0 = np.asarray(coords, dtype=float).reshape(-1)
    cache = {"x": None, "value": None}
    history = []

    def evaluate(x):
        x = np.asarray(x, dtype=float)
        if cache["x"] is not None and np.allclose(x, cache["x"]):
            return cache["value"]

        atom = [[symbol, *xyz] for symbol, xyz in zip(symbols, x.reshape(-1, 3))]
        mol = gto.M(atom=atom, unit="Bohr", basis=args.basis, verbose=0)
        mf = scf.RHF(mol).run(verbose=0)
        if args.reference_geometry == "pyscf-casscf-opt":
            mc = mcscf.CASSCF(mf, args.ncas, args.nelecas)
            mc.max_cycle_macro = 30
            mc.kernel()
            gradient = mc.nuc_grad_method().kernel().reshape(-1)
            energy = float(mc.e_tot)
        else:
            gradient = grad.RHF(mf).kernel().reshape(-1)
            energy = float(mf.e_tot)
        value = (energy, np.asarray(gradient, dtype=float))
        history.append(
            {
                "energy": value[0],
                "gradient_norm": float(np.linalg.norm(value[1])),
                "gradient_max": float(np.max(np.abs(value[1]))),
            }
        )
        cache["x"] = x.copy()
        cache["value"] = value
        return value

    result = minimize(
        fun=lambda x: evaluate(x)[0],
        x0=x0,
        jac=lambda x: evaluate(x)[1],
        method="BFGS",
        options={"gtol": args.rhf_opt_gtol, "maxiter": args.rhf_opt_maxiter},
    )
    return result.x.reshape(-1, 3), result, history


def select_nearest_modes(frequencies_cm1, targets_cm1):
    frequencies_cm1 = np.asarray(frequencies_cm1, dtype=float)
    selected = []
    for target in targets_cm1:
        candidates = [
            idx
            for idx, freq in enumerate(frequencies_cm1)
            if idx not in selected and np.isfinite(freq) and freq > 50.0
        ]
        if not candidates:
            raise RuntimeError("No positive vibrational modes are available for selection.")
        selected.append(min(candidates, key=lambda idx: abs(frequencies_cm1[idx] - target)))
    return np.asarray(selected, dtype=int)


def dimensionless_displacement_modes(vib, indices):
    modes = np.asarray(vib["modes"], dtype=float)[indices]
    frequencies_au = np.asarray(vib["freq_au"], dtype=float)[indices]
    return modes / np.sqrt(amu_to_au * frequencies_au)[:, None, None]


def dimensionless_guessed_modes(mol, target_frequencies_cm1):
    coords = np.asarray(mol.atom_coords(), dtype=float)
    masses = np.asarray(mol.atom_mass_list(), dtype=float) * amu_to_au
    ring_center = coords[:6].mean(axis=0)

    coupling = np.zeros_like(coords)
    coupling[:, 0] = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -0.7, 0.7, -0.7, 0.7])

    tuning = np.zeros_like(coords)
    tuning[:, 1:] = coords[:, 1:] - ring_center[1:]

    patterns = np.stack([coupling, tuning])
    frequencies_au = np.asarray(target_frequencies_cm1, dtype=float) * wavenumber2hartree
    modes = []
    for pattern, omega in zip(patterns, frequencies_au):
        mass_norm = np.sqrt(np.einsum("A,Ax,Ax->", masses, pattern, pattern))
        modes.append(pattern / mass_norm / np.sqrt(omega))
    return np.asarray(modes), frequencies_au, np.asarray(target_frequencies_cm1, dtype=float), np.array([-1, -1])


def get_modes(mol, args):
    if args.mode_source == "guess":
        print("Using quick symmetry-adapted guessed pyrazine modes.", flush=True)
        return dimensionless_guessed_modes(mol, args.target_frequencies)

    print("Running native RHF Hessian for mode selection...", flush=True)
    hessian_mol = build_pyrazine(args.basis, "builtin", args.eri, coords=mol.atom_coords())
    mf = hessian_mol.RHF().run()
    hessian = mf.Hessian().run()
    vib = hessian.vibrational_analysis()
    mode_indices = select_nearest_modes(vib["freq_cm1"], args.target_frequencies)
    modes = dimensionless_displacement_modes(vib, mode_indices)
    frequencies_au = np.asarray(vib["freq_au"], dtype=float)[mode_indices]
    frequencies_cm1 = np.asarray(vib["freq_cm1"], dtype=float)[mode_indices]
    return modes, frequencies_au, frequencies_cm1, mode_indices


def build_lvc_from_reference(mc, modes, frequencies_au, args):
    from pyqed.qchem.vibronic import LVC

    return LVC.from_casci(
        mc,
        modes=modes,
        frequencies=frequencies_au,
        state_ids=range(args.nstates),
        mode_ids=np.array([10, 6]),
        return_quadratic=True,
    )


def stationarize_s0(coords, modes, frequencies_au, args):
    coords = np.asarray(coords, dtype=float).copy()
    total_q = np.zeros(modes.shape[0], dtype=float)
    history = []

    for _iteration in range(max(0, args.stationary_iterations)):
        mol, _mf, mc = run_reference(coords, args)
        lvc, quadratic = build_lvc_from_reference(mc, modes, frequencies_au, args)
        slope = np.asarray(lvc.vibronic_couplings()[0, 0], dtype=float)

        if args.stationary_hessian == "fd":
            hess = np.zeros((modes.shape[0], modes.shape[0]), dtype=float)
            for mode_id in range(modes.shape[0]):
                trial = coords + args.stationary_fd_step * modes[mode_id]
                _trial_mol, _trial_mf, trial_mc = run_reference(trial, args)
                trial_lvc, _trial_quadratic = build_lvc_from_reference(
                    trial_mc,
                    modes,
                    frequencies_au,
                    args,
                )
                trial_slope = np.asarray(trial_lvc.vibronic_couplings()[0, 0], dtype=float)
                hess[:, mode_id] = (trial_slope - slope) / args.stationary_fd_step
        else:
            hess = np.asarray(quadratic[0, 0], dtype=float)

        step = -np.linalg.pinv(hess, rcond=1.0e-10) @ slope
        step_norm = float(np.linalg.norm(step))
        if step_norm > args.max_stationary_step:
            step = step * (args.max_stationary_step / step_norm)
        coords = coords + np.einsum("m,mAx->Ax", step, modes, optimize=True)
        total_q += step
        history.append(
            {
                "slope": slope.copy(),
                "hessian": hess.copy(),
                "step": step.copy(),
                "step_norm": float(np.linalg.norm(step)),
                "hessian_source": args.stationary_hessian,
                "energy": float(lvc.reference_energies[0]),
                "mol": mol,
                "mc": mc,
            }
        )

    mol, mf, mc = run_reference(coords, args)
    return coords, total_q, history, mol, mf, mc


def print_vector_table(title, labels, values, reference):
    print(title)
    print("  {:>12s} {:>16s} {:>16s} {:>16s}".format("quantity", "CASCI-LVC", "literature", "difference"))
    for label, value, ref in zip(labels, values, reference):
        print(f"  {label:>12s} {value:16.3f} {ref:16.3f} {value - ref:16.3f}")
    print()


def main():
    args = parse_args()

    print("Building pyrazine molecule...", flush=True)
    mol = build_pyrazine(args.basis, args.driver, args.eri)
    opt_history = []
    opt_result = None
    if args.reference_geometry != "input":
        method_label = "CASSCF" if args.reference_geometry == "pyscf-casscf-opt" else "RHF"
        print(f"Optimizing S0 reference geometry with PySCF {method_label} gradients...", flush=True)
        coords, opt_result, opt_history = optimize_reference_geometry_pyscf(mol.atom_coords(), args)
        mol = build_pyrazine(args.basis, args.driver, args.eri, coords=coords)

    modes, frequencies_au, frequencies_cm1, mode_indices = get_modes(mol, args)

    stationarity_history = []
    total_q = np.zeros(modes.shape[0], dtype=float)
    if args.stationarize_s0:
        print("Stationarizing S0 in the selected mode subspace...", flush=True)
        coords, total_q, stationarity_history, mol, _mf, mc = stationarize_s0(
            mol.atom_coords(),
            modes,
            frequencies_au,
            args,
        )
    else:
        print(f"Running PyQED RHF/{args.mc_method.upper()} at the input reference geometry...", flush=True)
        mol, _mf, mc = run_reference(mol.atom_coords(), args)

    print("Building LVC from analytic CASCI BO-Hamiltonian derivatives...", flush=True)
    lvc, quadratic = build_lvc_from_reference(mc, modes, frequencies_au, args)
    literature = pyrazine_2mode_lvc(units="cm^-1")

    energies_cm1 = (lvc.reference_energies - lvc.reference_energies[0]) * au2wavenumber
    raw_couplings_cm1 = lvc.vibronic_couplings() * au2wavenumber
    couplings_cm1 = raw_couplings_cm1.copy()
    if args.subtract_common_s0_slope:
        diag_idx = np.diag_indices(lvc.nstates)
        for mode in range(lvc.nmodes):
            couplings_cm1[diag_idx[0], diag_idx[1], mode] -= raw_couplings_cm1[0, 0, mode]
    literature_couplings = literature.vibronic_couplings()

    print()
    print("Pyrazine two-mode LVC.from_casci benchmark")
    print(f"  basis: {args.basis}")
    print(f"  driver: {args.driver}")
    print(f"  reference geometry: {args.reference_geometry}")
    if opt_result is not None and opt_history:
        last_opt = opt_history[-1]
        print(
            "  reference opt: "
            f"success={bool(opt_result.success)}, "
            f"energy={last_opt['energy']:.12f} Eh, "
            f"|grad|={last_opt['gradient_norm']:.3e}, "
            f"max|grad|={last_opt['gradient_max']:.3e}"
        )
    print(f"  active space: CAS({args.nelecas},{args.ncas})")
    print(f"  MC method: {args.mc_method.upper()}")
    print(f"  mode source: {args.mode_source}")
    print(f"  S0 stationarized: {args.stationarize_s0}")
    print(f"  common S0 slope subtracted: {args.subtract_common_s0_slope}")
    if args.stationarize_s0:
        print(
            "  accumulated reference shift q: "
            + ", ".join(f"{value:.6f}" for value in total_q)
        )
        for idx, record in enumerate(stationarity_history, start=1):
            slope_cm1 = record["slope"] * au2wavenumber
            step = record["step"]
            print(
                f"  stationarization {idx}: "
                f"S0 slope before = [{slope_cm1[0]:.3f}, {slope_cm1[1]:.3f}] cm^-1, "
                f"dq = [{step[0]:.6f}, {step[1]:.6f}], "
                f"Jacobian = {record['hessian_source']}"
            )
    if args.mode_source == "hessian":
        print(f"  selected Hessian mode indices: {mode_indices.tolist()}")
    print(
        "  selected frequencies (cm^-1): "
        + ", ".join(f"{freq:.3f}" for freq in frequencies_cm1)
    )
    print(
        "  target frequencies (cm^-1):   "
        + ", ".join(f"{freq:.3f}" for freq in args.target_frequencies)
    )
    print()

    print_vector_table(
        "Vertical energies relative to S0 (cm^-1)",
        [f"S{i}" for i in range(args.nstates)],
        energies_cm1,
        literature.reference_energies[: args.nstates],
    )

    print(
        "Final raw S0 slopes from electronic_hamiltonian_derivative() (cm^-1): "
        + ", ".join(f"{value:.6f}" for value in raw_couplings_cm1[0, 0])
    )
    print(
        "Final model S0 slopes after common-term removal (cm^-1): "
        + ", ".join(f"{value:.6f}" for value in couplings_cm1[0, 0])
    )
    print()

    diag_labels = []
    diag_values = []
    diag_refs = []
    mode_names = ("10a", "6a")
    for state in range(args.nstates):
        for mode in range(2):
            diag_labels.append(f"S{state}/{mode_names[mode]}")
            diag_values.append(couplings_cm1[state, state, mode])
            diag_refs.append(literature_couplings[state, state, mode])
    print_vector_table(
        "Diagonal electronic linear couplings dE/dQ (cm^-1)",
        diag_labels,
        diag_values,
        diag_refs,
    )

    print_vector_table(
        "Interstate linear couplings <Sa|dH/dQ|Sb> (cm^-1)",
        ["S1-S2/10a", "S1-S2/6a"],
        [raw_couplings_cm1[1, 2, 0], raw_couplings_cm1[1, 2, 1]],
        [literature_couplings[1, 2, 0], literature_couplings[1, 2, 1]],
    )

    print("Quadratic derivative tensor")
    print(f"  shape: {quadratic.shape}")
    print(f"  max |G| in selected modes: {np.max(np.abs(quadratic)) * au2wavenumber:.3f} cm^-1")
    print()
    print("Note: signs of normal modes and electronic phases are arbitrary; compare magnitudes first.")


if __name__ == "__main__":
    main()
