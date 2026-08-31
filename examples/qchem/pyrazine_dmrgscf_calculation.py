#!/usr/bin/env python3
"""Run and plot a factorized Abelian DMRG-SCF calculation for pyrazine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg import DMRGSCF


ATOM = """
N   0.000000   1.340000   0.000000
C   1.160000   0.670000   0.000000
C   1.160000  -0.670000   0.000000
N   0.000000  -1.340000   0.000000
C  -1.160000  -0.670000   0.000000
C  -1.160000   0.670000   0.000000
H   2.095000   1.210000   0.000000
H   2.095000  -1.210000   0.000000
H  -2.095000  -1.210000   0.000000
H  -2.095000   1.210000   0.000000
"""


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _plot(mc, timings, path, *, ncas, nelecas, bond_dimension):
    energies = np.asarray(mc.e_history, dtype=float).reshape(-1)
    cycles = np.arange(energies.size)
    diagnostics = list(mc.macro_diagnostics or [])
    gradient_cycles = np.asarray(
        [row.get("macro", index + 1) for index, row in enumerate(diagnostics)],
        dtype=int,
    )
    gradients = np.asarray(
        [row.get("gn", np.nan) for row in diagnostics],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)
    error = (energies - np.min(energies)) * 1000.0
    axes[0].plot(cycles, error, "o-", color="#0072B2", label="Energy")
    axes[0].axhline(0.0, color="0.45", lw=0.8)
    axes[0].set(
        xlabel="DMRG-SCF macro iteration",
        ylabel=r"$E-E_{\mathrm{best}}$ (m$E_h$)",
        title="Orbital-optimization convergence",
    )
    finite = np.isfinite(gradients) & (gradients > 0.0)
    if np.any(finite):
        gradient_axis = axes[0].twinx()
        gradient_axis.semilogy(
            gradient_cycles[finite],
            gradients[finite],
            "s--",
            color="#D55E00",
            label="Orbital gradient",
        )
        gradient_axis.set_ylabel("Orbital-gradient norm", color="#D55E00")
        gradient_axis.tick_params(axis="y", labelcolor="#D55E00")

    labels = ["CD build", "RHF", "DMRG-SCF"]
    values = [timings["build"], timings["rhf"], timings["dmrgscf"]]
    axes[1].bar(labels, values, color=["#56B4E9", "#009E73", "#CC79A7"])
    axes[1].set(ylabel="Wall time (s)", title="Calculation timing")
    for index, value in enumerate(values):
        axes[1].text(index, value, f"{value:.1f}", ha="center", va="bottom")

    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Pyrazine/aug-cc-pVDZ charge/$S_z$ DMRG-SCF "
        f"CAS({nelecas},{ncas}), $D={bond_dimension}$"
    )
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/private/tmp/pyqed-pyrazine-dmrgscf")
    parser.add_argument("--ncas", type=int, default=10)
    parser.add_argument("--nelecas", type=int, default=10)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--macro-cycles", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cd-tol", type=float, default=1.0e-8)
    parser.add_argument("--macro-tol", type=float, default=1.0e-6)
    parser.add_argument("--sweep-tol", type=float, default=1.0e-7)
    parser.add_argument("--orb-grad-tol", type=float, default=1.0e-4)
    parser.add_argument("--optimizer-max-steps", type=int, default=4)
    parser.add_argument("--allow-unconverged", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    timings = {}
    total_start = time.perf_counter()

    print("[1/3] Building spherical matrix-free CD integrals", flush=True)
    started = time.perf_counter()
    mol = Molecule(atom=ATOM, basis="aug-cc-pvdz", unit="angstrom")
    mol.build(
        eri="cd",
        options={
            "coord_type": "spherical",
            "low_rank_tol": args.cd_tol,
            "eri_screen_tol": 0.0,
            "parallel": True,
            "eri_workers": args.workers,
            "parallel_min_nao": 0,
            "rys_cache_mib": 256,
        },
    )
    timings["build"] = time.perf_counter() - started

    print("[2/3] Running factorized RHF", flush=True)
    started = time.perf_counter()
    mf = mol.RHF().run(tol=1.0e-10, conv_tol_grad=1.0e-5, verbose=0)
    timings["rhf"] = time.perf_counter() - started

    print(
        f"[3/3] Running charge/Sz DMRG-SCF CAS({args.nelecas},{args.ncas}), "
        f"D={args.D}",
        flush=True,
    )
    mc = DMRGSCF(
        mf,
        ncas=args.ncas,
        nelecas=args.nelecas,
        D=args.D,
        max_cycles=args.macro_cycles,
        macro_tol=args.macro_tol,
        dmrg_conv_tol=args.sweep_tol,
        symmetry="sz",
        site="spatial",
        spatial_abelian_mpo="auto",
        init_guess="hf",
        verbose=0,
    )
    started = time.perf_counter()
    mc.run(
        nstates=1,
        nsweeps=args.sweeps,
        sweep_tol=args.sweep_tol,
        orb_grad_tol=args.orb_grad_tol,
        optimizer="RCG",
        optimizer_tol=1.0e-5,
        optimizer_max_steps=args.optimizer_max_steps,
        optimizer_max_step_norm=0.20,
        macro_trust_radius=0.20,
        warm_start_bonds=True,
        mixer_zero_block_noise_scale=0.0,
        compute_s2=False,
        require_conv=not args.allow_unconverged,
    )
    timings["dmrgscf"] = time.perf_counter() - started
    timings["total"] = time.perf_counter() - total_start

    stem = output_dir / "pyrazine_aug_cc_pvdz_charge_sz_dmrgscf"
    figure_path = stem.with_suffix(".png")
    results_path = stem.with_suffix(".json")
    _plot(
        mc,
        timings,
        figure_path,
        ncas=args.ncas,
        nelecas=args.nelecas,
        bond_dimension=args.D,
    )

    factor_shape = tuple(int(value) for value in mol.eri_factors.shape)
    active_info = dict(getattr(mc.casci, "build_info", {}) or {})
    dmrg_engine = getattr(mc.casci, "dmrg", None)
    environment_profile = dict(
        getattr(dmrg_engine, "environment_profile", {}) or {}
    )
    moving_profile = dict(environment_profile.get("moving_environment", {}) or {})
    result = {
        "system": "pyrazine",
        "geometry": "idealized planar D2h",
        "basis": "aug-cc-pVDZ",
        "coordinate_type": "spherical",
        "integrals": "matrix-free pivoted CD",
        "cd_tolerance": args.cd_tol,
        "factor_shape": factor_shape,
        "factor_storage": mol._builtin_build_info.get("factor_storage"),
        "active_space": {
            "electrons": args.nelecas,
            "orbitals": args.ncas,
            "selection": "canonical orbitals around the Fermi level",
            "ncore": int(mc.ncore),
        },
        "symmetry": "Abelian charge/Sz",
        "bond_dimension": args.D,
        "maximum_sweeps": args.sweeps,
        "maximum_macro_cycles": args.macro_cycles,
        "rhf_energy_hartree": mf.e_tot,
        "dmrgscf_energy_hartree": mc.e_tot,
        "correlation_energy_hartree": np.asarray(mc.e_tot) - mf.e_tot,
        "converged": mc.converged,
        "macro_converged": mc.macro_converged,
        "solver_converged": mc.solver_converged,
        "macro_iterations": mc.macro_iterations,
        "spin_square": getattr(mc.casci, "s2", None),
        "energy_history_hartree": mc.e_history,
        "macro_diagnostics": mc.macro_diagnostics,
        "factorized_orbital_integrals": mc.use_cholesky_integrals,
        "spatial_rdm2_algorithm": getattr(mc.casci, "spatial_rdm2_algorithm", None),
        "resolved_spatial_rdm2_algorithm": mc.casci._resolve_spatial_rdm2_algorithm(),
        "spatial_rdm_diagnostics": getattr(mc.casci, "spatial_rdm_diagnostics", None),
        "dmrg_path": {
            "requested_spatial_mpo": getattr(mc.casci, "spatial_abelian_mpo", None),
            "representation": active_info.get("representation"),
            "pipeline": active_info.get("pipeline"),
            "family_environment_backend": active_info.get(
                "spatial_family_environment_backend"
            ),
            "performance": active_info.get("dmrg_performance"),
            "build_timings_seconds": active_info.get("build_timings"),
            "resolved_abelian_options": active_info.get(
                "resolved_abelian_matvec_options"
            ),
            "moving_environment_seconds": {
                str(key): value
                for key, value in moving_profile.items()
                if str(key).endswith("_seconds")
            },
        },
        "timing_seconds": timings,
        "peak_rss_bytes": _rss_bytes(),
    }
    results_path.write_text(json.dumps(_plain(result), indent=2) + "\n")

    print("\nDMRG-SCF calculation complete")
    print(f"E(RHF)      = {mf.e_tot:.12f} Ha")
    print(f"E(DMRG-SCF) = {float(np.asarray(mc.e_tot)):.12f} Ha")
    print(f"Converged   = {mc.converged}")
    print(f"Macro cycles= {mc.macro_iterations}")
    print(f"CD factors  = {factor_shape}")
    print(f"Peak RSS    = {_rss_bytes() / 2**30:.3f} GiB")
    print(f"Results     = {results_path}")
    print(f"Figure      = {figure_path}")


if __name__ == "__main__":
    main()
