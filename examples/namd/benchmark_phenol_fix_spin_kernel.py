#!/usr/bin/env python3
"""Benchmark fused and separate native ``fix_spin`` kernels for phenol CAS(10,10)."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import au2mev
from pyqed.models.phenol_coordinates import PHENOL_SPECIES
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf import direct_ci


NCAS = 10
NELECAS = 10
NCORE = 20
HARTREE_TO_MEV = au2mev


def build_mean_field(geometry, basis):
    mol = Molecule(
        atom=list(zip(PHENOL_SPECIES, np.asarray(geometry))),
        unit="angstrom",
        basis=basis,
        charge=0,
        spin=0,
    )
    pmol = mol.topyscf()
    pmol.build(verbose=0)
    mol.nao = mol.nmo = pmol.nao
    mol.nbas = pmol.nbas
    mf = mol.RHF(verbose=0).run(tol=1.0e-10, max_cycle=100, density_fit=True)
    mf.eri_factors = np.vstack([np.asarray(block) for block in mf._pyscf_mf.with_df.loop()])
    mf.eri = None
    mf.cholesky_jk = True
    return mf


def operator_data(h1, eri_same, eri_cross, binary, connectivity):
    diagonal = direct_ci._compute_diag_compact(h1, eri_same, eri_cross, binary)
    alpha_cross = direct_ci._spin_string_cross_diagonal(
        eri_cross, connectivity.alpha_occ
    )
    beta_cross = direct_ci._spin_string_cross_diagonal(
        eri_cross, connectivity.beta_occ
    )
    return diagonal, alpha_cross, beta_cross


def native_solve(
    operator,
    pair_left,
    pair_right,
    connectivity,
    guess,
    *,
    workers,
    nroots,
    penalty=None,
):
    h1, eri_same, eri_cross, diagonal, alpha_cross, beta_cross = operator
    result = direct_ci._davidson_spin0_pair_cpp(
        h1,
        eri_same,
        eri_cross,
        diagonal,
        pair_left,
        pair_right,
        connectivity.alpha_occ,
        connectivity.beta_occ,
        connectivity.I_A,
        connectivity.J_A,
        connectivity.p_A,
        connectivity.q_A,
        connectivity.phase_A,
        connectivity.I_B,
        connectivity.J_B,
        connectivity.p_B,
        connectivity.q_B,
        connectivity.phase_B,
        connectivity.I_AA,
        connectivity.J_AA,
        connectivity.p_AA,
        connectivity.q_AA,
        connectivity.r_AA,
        connectivity.s_AA,
        connectivity.phase_AA,
        connectivity.I_BB,
        connectivity.J_BB,
        connectivity.p_BB,
        connectivity.q_BB,
        connectivity.r_BB,
        connectivity.s_BB,
        connectivity.phase_BB,
        connectivity.alpha_offsets,
        connectivity.beta_offsets,
        connectivity.alpha_order,
        connectivity.beta_order,
        alpha_cross,
        beta_cross,
        connectivity.alpha_ordered_I,
        connectivity.alpha_ordered_J,
        connectivity.alpha_ordered_phase,
        connectivity.beta_ordered_I,
        connectivity.beta_ordered_J,
        connectivity.beta_ordered_phase,
        workers=workers,
        nroots=nroots,
        guess=guess,
        energy_tol=1.0e-9,
        residual_tol=3.2e-5,
        max_cycle=150,
        max_subspace=120,
        spin_penalty=penalty,
    )
    if result is None:
        raise RuntimeError("native spin-zero Davidson did not return a result")
    return result


def plot_results(result, output):
    labels = ["Fused penalty\n(general H kernel)", "Separate penalty\n(packed-BLAS H)"]
    old = np.asarray(result["wall_seconds"]["fused"], dtype=float)
    new = np.asarray(result["wall_seconds"]["separate"], dtype=float)
    medians = [np.median(old), np.median(new)]
    energy_error = np.maximum(
        np.asarray(result["state_energy_error_mev"], dtype=float), 1.0e-12
    )
    spin = np.maximum(np.abs(np.asarray(result["state_s2"], dtype=float)), 1.0e-14)

    colors = ["#7A7A7A", "#0072B2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    x = np.arange(2)
    axes[0].bar(x, medians, color=colors, width=0.62, edgecolor="black", linewidth=0.7)
    for index, samples in enumerate((old, new)):
        offsets = np.linspace(-0.10, 0.10, samples.size)
        axes[0].scatter(
            np.full(samples.size, index) + offsets,
            samples,
            color="white",
            edgecolor="black",
            linewidth=0.7,
            s=24,
            zorder=3,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Six-root Davidson wall time (s)")
    axes[0].set_title(
        f"{result['speedup']:.2f}$\\times$ faster",
        fontsize=10,
    )
    axes[0].grid(axis="y", color="0.88", linewidth=0.7)
    axes[0].set_axisbelow(True)

    roots = np.arange(1, len(energy_error) + 1)
    axes[1].semilogy(
        roots,
        energy_error,
        "o-",
        color="#D55E00",
        label=r"$|E_{\rm separate}-E_{\rm fused}|$ (meV)",
    )
    axes[1].semilogy(
        roots,
        spin,
        "s--",
        color="#009E73",
        label=r"$|\langle S^2\rangle|$",
    )
    axes[1].set_xlabel("State")
    axes[1].set_ylabel("Absolute error / spin residual")
    axes[1].set_xticks(roots)
    axes[1].grid(color="0.90", linewidth=0.7)
    axes[1].legend(frameon=False, fontsize=8, loc="best")

    for label, axis in zip(("a", "b"), axes):
        axis.text(-0.14, 1.04, label, transform=axis.transAxes, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)

    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_penalty.npz"),
    )
    parser.add_argument("--basis", default="6-31+g*")
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--nroots", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/phenol_fix_spin_kernel_benchmark"),
    )
    args = parser.parse_args()

    seed = np.load(args.seed, allow_pickle=True)
    print("building phenol mean field", flush=True)
    mf = build_mean_field(seed["geometry"], args.basis)
    solver = direct_ci.CASCI(
        mf,
        ncas=NCAS,
        nelecas=NELECAS,
        ncore=NCORE,
        multiplicity=1,
        tol=1.0e-9,
        verbose=0,
    )
    solver.mo_coeff = np.asarray(seed["mo_coeff"])
    solver.mo_core, solver.mo_cas = direct_ci._slice_active_orbitals(
        solver.mo_coeff, NCORE, NCAS
    )
    occupations = direct_ci._reference_active_occupations(
        solver.nelecas_spin, NCAS
    )
    binary = direct_ci.get_fci_string_basis(mo_occ=occupations)
    solver.binary = binary
    physical_h1, physical_same, physical_cross, energy_core = (
        solver.get_direct_compact_integrals(use_cholesky=True)
    )
    print("building CAS(10,10) connectivity", flush=True)
    connectivity = direct_ci.build_spin_string_connectivity(binary)
    pairs = direct_ci._rectangular_spin0_pairs(binary)
    pair_left, pair_right, pair_same = direct_ci._spin0_pair_arrays(pairs)
    guess = direct_ci._project_spin0_guess(
        [state.reshape(-1) for state in np.asarray(seed["ci"])[: args.nroots]],
        pair_left,
        pair_right,
        pair_same,
        binary.shape[0],
    )

    physical_diag = operator_data(
        physical_h1, physical_same, physical_cross, binary, connectivity
    )
    physical = (physical_h1, physical_same, physical_cross, *physical_diag)
    s2_h1, s2_h2 = direct_ci.build_spin_square_operator(NCAS)
    penalty_h1 = s2_h1[0]
    penalty_same = s2_h2[0, 0]
    penalty_cross = s2_h2[0, 1]
    penalty_diag = operator_data(
        penalty_h1, penalty_same, penalty_cross, binary, connectivity
    )
    penalty = (
        penalty_h1,
        penalty_same,
        penalty_cross,
        *penalty_diag,
        args.shift,
    )
    fused_h1 = physical_h1 + args.shift * penalty_h1
    fused_same = physical_same + args.shift * penalty_same
    fused_cross = physical_cross + args.shift * penalty_cross
    fused_diag = operator_data(
        fused_h1, fused_same, fused_cross, binary, connectivity
    )
    fused = (fused_h1, fused_same, fused_cross, *fused_diag)

    # Populate native workspaces once before timing dynamic-library and allocator warmup.
    print("warming fused native Davidson", flush=True)
    native_solve(
        fused,
        pair_left,
        pair_right,
        connectivity,
        guess,
        workers=args.workers,
        nroots=args.nroots,
    )
    print("warming separate native Davidson", flush=True)
    native_solve(
        physical,
        pair_left,
        pair_right,
        connectivity,
        guess,
        workers=args.workers,
        nroots=args.nroots,
        penalty=penalty,
    )
    print("timing native Davidson", flush=True)

    timings = {"fused": [], "separate": []}
    solutions = {}
    for repeat in range(args.repeats):
        order = ("fused", "separate") if repeat % 2 == 0 else ("separate", "fused")
        for mode in order:
            start = time.perf_counter()
            solutions[mode] = native_solve(
                fused if mode == "fused" else physical,
                pair_left,
                pair_right,
                connectivity,
                guess,
                workers=args.workers,
                nroots=args.nroots,
                penalty=None if mode == "fused" else penalty,
            )
            timings[mode].append(time.perf_counter() - start)

    fused_energy, _ = solutions["fused"]
    separate_energy, separate_vectors = solutions["separate"]
    state_s2 = []
    for root in range(args.nroots):
        sigma_s2 = direct_ci._sigma_compact_spin0_pair(
            penalty_h1,
            penalty_same,
            penalty_cross,
            penalty_diag[0],
            separate_vectors[:, root],
            pair_left,
            pair_right,
            connectivity.alpha_occ,
            connectivity.beta_occ,
            connectivity.I_A,
            connectivity.J_A,
            connectivity.p_A,
            connectivity.q_A,
            connectivity.phase_A,
            connectivity.I_B,
            connectivity.J_B,
            connectivity.p_B,
            connectivity.q_B,
            connectivity.phase_B,
            connectivity.I_AA,
            connectivity.J_AA,
            connectivity.p_AA,
            connectivity.q_AA,
            connectivity.r_AA,
            connectivity.s_AA,
            connectivity.phase_AA,
            connectivity.I_BB,
            connectivity.J_BB,
            connectivity.p_BB,
            connectivity.q_BB,
            connectivity.r_BB,
            connectivity.s_BB,
            connectivity.phase_BB,
            connectivity.alpha_offsets,
            connectivity.beta_offsets,
            connectivity.alpha_order,
            connectivity.beta_order,
            penalty_diag[1],
            penalty_diag[2],
            connectivity.alpha_ordered_I,
            connectivity.alpha_ordered_J,
            connectivity.alpha_ordered_phase,
            connectivity.beta_ordered_I,
            connectivity.beta_ordered_J,
            connectivity.beta_ordered_phase,
            1,
        )
        state_s2.append(float(np.dot(separate_vectors[:, root], sigma_s2)))

    old_median = float(np.median(timings["fused"]))
    new_median = float(np.median(timings["separate"]))
    result = {
        "system": "phenol",
        "method": f"SA({args.nroots})-CASCI(10e,10o)/{args.basis}",
        "seed": str(args.seed),
        "shift": args.shift,
        "workers": args.workers,
        "repeats": args.repeats,
        "ndeterminants": int(binary.shape[0]),
        "npair": int(pair_left.size),
        "cblas": bool(direct_ci.direct_ci_capabilities()["cblas"]),
        "wall_seconds": timings,
        "median_wall_seconds": {"fused": old_median, "separate": new_median},
        "speedup": old_median / new_median,
        "energies_hartree": (separate_energy + energy_core).tolist(),
        "state_energy_error_mev": (
            np.abs(separate_energy - fused_energy) * HARTREE_TO_MEV
        ).tolist(),
        "state_s2": state_s2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    png, pdf = plot_results(result, args.output)
    print(json.dumps(result, indent=2))
    print(png)
    print(pdf)
    print(json_path)


if __name__ == "__main__":
    main()
