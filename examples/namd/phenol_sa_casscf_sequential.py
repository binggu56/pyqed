#!/usr/bin/env python3
"""Sequential, orbital-relaxed phenol SA(6)-CASSCF O--H cuts.

Each backend/direction pair is one sequential chain.  Orbitals and all CI
roots are transported from the preceding geometry; processes are used only
across independent chains, never across unordered points in one chain.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from matplotlib.lines import Line2D
from pyscf import fci, gto, mcscf, scf

from pyqed.units import au2ev
from pyqed.models.phenol_coordinates import PHENOL_SPECIES, PhenolReactiveChart
from pyqed.qchem import CASSCF as PyQEDCASSCF
from pyqed.qchem import Molecule as PyQEDMolecule

HARTREE_TO_EV = au2ev
NCAS = 10
NELECAS = 10
NCORE = 20


def geometry_at(distance):
    chart = PhenolReactiveChart()
    coordinate = np.array(chart.equilibrium, copy=True)
    coordinate[0] = float(distance)
    return np.asarray(chart.geometry(coordinate))


def pyscf_molecule(geometry, basis):
    return gto.M(
        atom=list(zip(PHENOL_SPECIES, np.asarray(geometry))),
        unit="Angstrom",
        basis=basis,
        charge=0,
        spin=0,
        symmetry=False,
        verbose=0,
        max_memory=12000,
    )


def singlet_solver(mol, nstates):
    solver = fci.addons.fix_spin_(fci.direct_spin1.FCI(mol), shift=1.0, ss=0)
    solver.nroots = int(nstates)
    solver.conv_tol = 1.0e-9
    solver.max_cycle = 150
    return solver


def metric_orthonormalize(coefficients, overlap):
    coefficients = np.asarray(coefficients)
    metric = coefficients.T @ overlap @ coefficients
    values, vectors = np.linalg.eigh(metric)
    if np.min(values) < 1.0e-10:
        raise RuntimeError("projected molecular orbitals lost numerical rank")
    return coefficients @ ((vectors * values**-0.5) @ vectors.T)


def project_orbitals(previous_geometry, previous_mo, geometry, basis):
    old_mol = pyscf_molecule(previous_geometry, basis)
    new_mol = pyscf_molecule(geometry, basis)
    projected = scf.addons.project_mo_nr2nr(old_mol, previous_mo, new_mol)
    overlap = new_mol.intor_symmetric("int1e_ovlp")
    transported = np.empty_like(projected)
    completed = np.empty((projected.shape[0], 0), dtype=projected.dtype)
    blocks = (
        slice(0, NCORE),
        slice(NCORE, NCORE + NCAS),
        slice(NCORE + NCAS, projected.shape[1]),
    )
    for block in blocks:
        candidate = np.array(projected[:, block], copy=True)
        if completed.shape[1]:
            candidate -= completed @ (completed.T @ overlap @ candidate)
        candidate = metric_orthonormalize(candidate, overlap)
        transported[:, block] = candidate
        completed = np.column_stack((completed, candidate))
    error = np.linalg.norm(
        transported.T @ overlap @ transported - np.eye(transported.shape[1])
    )
    if error > 1.0e-8:
        raise RuntimeError(f"blockwise orbital transport lost orthogonality: {error:.3e}")
    return transported


def align_reference_subspaces(reference_mo, transported_mo, geometry, basis):
    """Align optimized reference subspaces to a transported orbital gauge."""

    overlap = pyscf_molecule(geometry, basis).intor_symmetric("int1e_ovlp")
    reference_mo = np.asarray(reference_mo)
    transported_mo = np.asarray(transported_mo)
    aligned = np.empty_like(reference_mo)
    singular_values = []
    blocks = (
        slice(0, NCORE),
        slice(NCORE, NCORE + NCAS),
        slice(NCORE + NCAS, reference_mo.shape[1]),
    )
    for block in blocks:
        metric = reference_mo[:, block].T @ overlap @ transported_mo[:, block]
        left, singular, right_h = np.linalg.svd(metric)
        aligned[:, block] = reference_mo[:, block] @ (left @ right_h)
        singular_values.append(singular)
    return aligned, singular_values[1]


def active_continuity(left, right, basis):
    left_mol = pyscf_molecule(left["geometry"], basis)
    right_mol = pyscf_molecule(right["geometry"], basis)
    cross = gto.intor_cross("int1e_ovlp", left_mol, right_mol)
    active_overlap = (
        left["mo_coeff"][:, NCORE : NCORE + NCAS].T
        @ cross
        @ right["mo_coeff"][:, NCORE : NCORE + NCAS]
    )
    root_overlap = np.empty((len(left["ci"]), len(right["ci"])))
    for bra in range(root_overlap.shape[0]):
        for ket in range(root_overlap.shape[1]):
            root_overlap[bra, ket] = abs(
                fci.addons.overlap(
                    left["ci"][bra],
                    right["ci"][ket],
                    NELECAS,
                    (NELECAS // 2, NELECAS // 2),
                    active_overlap,
                )
            )
    return np.linalg.svd(active_overlap, compute_uv=False), root_overlap


def load_record(path):
    with np.load(path, allow_pickle=False) as archive:
        record = {key: np.asarray(archive[key]) for key in archive.files}
    if "energies" not in record and "e_states" in record:
        record["energies"] = record["e_states"]
    return record


def save_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **record)


def _pyscf_casscf(geometry, basis, nstates, mo0, ci0, options):
    mol = pyscf_molecule(geometry, basis)
    mf = scf.RHF(mol).density_fit().run(
        conv_tol=options["scf_tol"], max_cycle=100, verbose=0
    )
    if not mf.converged:
        mf = mf.newton().run(conv_tol=options["scf_tol"], max_cycle=80, verbose=0)

    macro_history = []
    current_mo = np.asarray(mo0)
    current_ci = None if ci0 is None else [np.asarray(state) for state in ci0]
    final = None
    for attempt in range(options["restarts"] + 1):
        driver = mcscf.DFCASSCF(mf, NCAS, NELECAS)
        driver.fcisolver = singlet_solver(mol, nstates)
        driver = driver.state_average_([1.0 / nstates] * nstates)
        driver.max_cycle_macro = options["macro_cycles"]
        driver.max_cycle_micro = options["micro_cycles"]
        driver.conv_tol = options["conv_tol"]
        driver.conv_tol_grad = options["conv_grad"]
        driver.max_memory = 12000

        def callback(environment):
            macro_history.append(
                (
                    float(environment.get("imacro", len(macro_history))),
                    float(environment.get("e_tot", np.nan)),
                    float(environment.get("norm_gorb", np.nan)),
                )
            )

        driver.callback = callback
        driver.kernel(current_mo, ci0=current_ci)
        final = driver
        current_mo = np.asarray(driver.mo_coeff)
        current_ci = [np.asarray(state) for state in driver.ci]
        last_grad = macro_history[-1][2] if macro_history else np.inf
        if driver.converged or last_grad <= options["conv_grad"]:
            break

    energies = np.asarray(final.e_states)
    states = np.asarray(final.ci)
    spins = np.asarray(
        [
            fci.spin_op.spin_square0(
                state, NCAS, (NELECAS // 2, NELECAS // 2)
            )[0]
            for state in states
        ]
    )
    gradient = macro_history[-1][2] if macro_history else np.nan
    relaxed = bool(final.converged or gradient <= options["conv_grad"])
    return {
        "geometry": np.asarray(geometry),
        "mo_coeff": np.asarray(final.mo_coeff),
        "ci": states,
        "energies": energies,
        "spins": spins,
        "scf_converged": np.asarray(bool(mf.converged)),
        "converged": np.asarray(bool(final.converged)),
        "orbital_relaxed": np.asarray(relaxed),
        "orbital_gradient": np.asarray(gradient),
        "macro_history": np.asarray(macro_history, dtype=float).reshape(-1, 3),
    }


def _pyqed_casscf(geometry, basis, nstates, mo0, ci0, options):
    mol = PyQEDMolecule(
        atom=list(zip(PHENOL_SPECIES, np.asarray(geometry))),
        unit="angstrom",
        basis=basis,
        charge=0,
        spin=0,
    )
    # Use the same PySCF density-fitting factors as the reference calculation so
    # that this comparison isolates the CASSCF optimizer and CI solver.  PyQED's
    # builtin integral driver also recognizes 6-31+G* natively.
    pmol = mol.topyscf()
    pmol.build(verbose=0)
    mol.nao = mol.nmo = pmol.nao
    mol.nbas = pmol.nbas
    mf = mol.RHF(verbose=0).run(
        tol=options["scf_tol"], max_cycle=100, density_fit=True
    )
    factors = np.vstack(
        [np.asarray(block) for block in mf._pyscf_mf.with_df.loop()]
    )
    mf.nao = mf.nmo = mol.nao
    mf.eri = None
    mf.eri_factors = factors
    mf.cholesky_jk = True
    mf.low_rank_jk = True
    mol.eri = None
    mol.eri_factors = factors

    current_mo = metric_orthonormalize(np.asarray(mo0), np.asarray(mol.overlap))
    current_ci = None if ci0 is None else [np.asarray(state) for state in ci0]
    combined_history = []
    combined_zero_step_recoveries = []
    final = None
    for _attempt in range(options["restarts"] + 1):
        driver = PyQEDCASSCF(
            mf,
            ncas=NCAS,
            nelecas=NELECAS,
            multiplicity=1,
            max_cycle=options["macro_cycles"],
            max_micro_cycle=options["micro_cycles"],
            conv_tol=options["conv_tol"],
            conv_tol_grad=options["conv_grad"],
            conv_tol_grad_relaxed=options["conv_grad"],
            conv_tol_step=options["conv_step"],
            optimizer=options["pyqed_optimizer"],
            max_step=options["pyqed_max_step"],
            coupling=options["pyqed_coupling"],
            ah_max_cycle=options["pyqed_ah_cycles"],
            ah_max_subspace=options["pyqed_ah_subspace"],
            ah_pspace_max_cycle=options["pyqed_ah_cycles"],
            ah_conv_tol=options["pyqed_ah_tol"],
            ah_adaptive_trust=True,
            keyframe_interval=options["pyqed_keyframe_interval"],
            keyframe_gradient_trust=options["pyqed_keyframe_gradient_trust"],
            active_overlap_floor=options["pyqed_active_overlap_floor"],
            micro_ci_mode=options["pyqed_micro_ci_mode"],
            use_cholesky=True,
            verbose=1,
        )
        driver.state_average([1.0 / nstates] * nstates)
        driver.fix_spin(ss=0, shift=options["spin_shift"])
        try:
            driver.run(nstates=nstates, mo_coeff=current_mo, ci0=current_ci)
        except RuntimeError:
            combined_history.extend(driver.history)
            combined_zero_step_recoveries.extend(driver.zero_step_recovery_history)
            if driver.casci is None or _attempt >= options["restarts"]:
                raise
            current_mo = np.asarray(driver.mo_coeff)
            current_ci = [np.asarray(state) for state in driver.casci.ci]
            continue
        final = driver
        combined_history.extend(driver.history)
        combined_zero_step_recoveries.extend(driver.zero_step_recovery_history)
        current_mo = np.asarray(driver.mo_coeff)
        current_ci = [np.asarray(state) for state in driver.ci]
        gradient = float(driver.history[-1]["gradient_norm"])
        if driver.converged or gradient <= options["conv_grad"]:
            break

    energies = np.asarray(final.e_tot)
    states = np.asarray(final.ci)
    spins = np.asarray([final.spin_square(state) for state in range(nstates)])
    gradient = float(final.history[-1]["gradient_norm"])
    history = np.asarray(
        [
            (
                row.get("cycle", row.get("macro_cycle", index + 1)),
                row["energy"],
                row["gradient_norm"],
            )
            for index, row in enumerate(combined_history)
        ],
        dtype=float,
    )
    ci_diagnostics = getattr(final.casci, "direct_ci_diagnostics", {})
    relaxed = bool(final.converged or gradient <= options["conv_grad"])
    return {
        "geometry": np.asarray(geometry),
        "mo_coeff": np.asarray(final.mo_coeff),
        "ci": states,
        "energies": energies,
        "spins": spins,
        "scf_converged": np.asarray(bool(mf.converged)),
        "converged": np.asarray(bool(final.converged)),
        "orbital_relaxed": np.asarray(relaxed),
        "orbital_gradient": np.asarray(gradient),
        "macro_history": history.reshape(-1, 3),
        "active_overlap_history": np.asarray(final.active_overlap_history),
        "keyframe_refreshes": np.asarray(
            sum("keyframe_refresh" in row for row in final.micro_history)
        ),
        "rejected_step_rollbacks": np.asarray(
            sum(bool(row.get("rejected_step_rolled_back", False)) for row in final.micro_history)
        ),
        "zero_step_recoveries": np.asarray(len(combined_zero_step_recoveries)),
        "zero_step_recovery_history": np.asarray(
            [
                (row["macro"], row["energy"], row["gradient_norm"])
                for row in combined_zero_step_recoveries
            ],
            dtype=float,
        ).reshape(-1, 3),
        "external_restarts": np.asarray(_attempt),
        "solver_backend": np.asarray(str(final.casci.solver_backend)),
        "ci_iterations": np.asarray(int(ci_diagnostics.get("iterations", -1))),
        "ci_requested_nstates": np.asarray(
            int(ci_diagnostics.get("requested_nstates", nstates))
        ),
        "ci_solved_nstates": np.asarray(
            int(ci_diagnostics.get("solved_nstates", nstates))
        ),
    }


def run_casscf(backend, geometry, basis, nstates, mo0, ci0, options):
    started = time.perf_counter()
    if backend == "pyscf":
        record = _pyscf_casscf(geometry, basis, nstates, mo0, ci0, options)
    elif backend == "pyqed":
        record = _pyqed_casscf(geometry, basis, nstates, mo0, ci0, options)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    record["wall_seconds"] = np.asarray(time.perf_counter() - started)
    record["backend"] = np.asarray(backend)
    if record["ci"].shape[0] != nstates:
        raise RuntimeError(f"{backend} returned CI shape {record['ci'].shape}")
    if np.max(np.abs(record["spins"])) > 1.0e-5:
        raise RuntimeError(f"{backend} returned spin-contaminated roots: {record['spins']}")
    if not bool(record["orbital_relaxed"]):
        raise RuntimeError(
            f"{backend} CASSCF did not relax orbitals; final |g|="
            f"{float(record['orbital_gradient']):.3e}"
        )
    return record


def anchor_worker(task):
    backend, output, seed_path, basis, nstates, distance, options = task
    output = Path(output)
    path = output / backend / "anchor.npz"
    if path.exists():
        record = load_record(path)
        if bool(record.get("orbital_relaxed", False)):
            return backend, str(path), "cached"
    seed = load_record(seed_path)
    record = run_casscf(
        backend,
        geometry_at(distance),
        basis,
        nstates,
        seed["mo_coeff"],
        seed["ci"],
        options,
    )
    record["distance"] = np.asarray(distance)
    save_record(path, record)
    return backend, str(path), "computed"


def chain_worker(task):
    backend, direction, distances, output, anchor_path, basis, nstates, options = task
    output = Path(output)
    previous = load_record(anchor_path)
    completed = []
    for distance in distances:
        path = output / backend / direction / f"r{distance:.5f}.npz"
        if path.exists():
            record = load_record(path)
            if not bool(record.get("orbital_relaxed", False)):
                path.unlink()
                record = None
        else:
            record = None
        if record is None:
            geometry = geometry_at(distance)
            transported_mo = project_orbitals(
                previous["geometry"], previous["mo_coeff"], geometry, basis
            )
            mo0 = transported_mo
            reference_dir = options.get("pyqed_reference_dir")
            reference_singular = None
            if backend == "pyqed" and reference_dir:
                reference_path = (
                    Path(reference_dir)
                    / "pyscf"
                    / direction
                    / f"r{distance:.5f}.npz"
                )
                if not reference_path.exists():
                    raise FileNotFoundError(
                        f"missing converged PySCF orbital reference: {reference_path}"
                    )
                reference = load_record(reference_path)
                mo0, reference_singular = align_reference_subspaces(
                    reference["mo_coeff"], transported_mo, geometry, basis
                )
            record = run_casscf(
                backend, geometry, basis, nstates, mo0, previous["ci"], options
            )
            singular, overlaps = active_continuity(previous, record, basis)
            record["active_singular_values"] = singular
            record["previous_root_overlap"] = overlaps
            record["distance"] = np.asarray(distance)
            if reference_singular is not None:
                record["reference_active_singular_values"] = reference_singular
                record["initialization"] = np.asarray(
                    "pyscf_subspaces_procrustes_aligned_to_previous_pyqed"
                )
            save_record(path, record)
        previous = record
        completed.append(str(path))
        gaps = (record["energies"][1:3] - record["energies"][:2]) * HARTREE_TO_EV
        print(
            f"[{backend}:{direction}] R={distance:.5f}  "
            f"gaps={gaps[0]:.4f},{gaps[1]:.4f} eV  "
            f"|g|={float(record['orbital_gradient']):.2e}",
            flush=True,
        )
    return backend, direction, completed


def collect_backend(output, backend, distances, anchor_distance):
    anchor = load_record(output / backend / "anchor.npz")
    records = []
    for distance in distances:
        if np.isclose(distance, anchor_distance, atol=1.0e-8):
            record = anchor
        else:
            direction = "increasing" if distance > anchor_distance else "decreasing"
            record = load_record(output / backend / direction / f"r{distance:.5f}.npz")
        records.append(record)
    return anchor, records


def plot_results(output, backends, distances, anchor_distance):
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 6))
    figure, panels = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    assembled = {}
    for column, backend in enumerate(backends):
        backend_label = {"pyscf": "PySCF", "pyqed": "PyQED"}[backend]
        anchor, records = collect_backend(output, backend, distances, anchor_distance)
        energies = np.asarray([record["energies"] for record in records])
        relative = (energies - anchor["energies"][0]) * HARTREE_TO_EV
        assembled[backend] = (records, energies, relative)
        for state, color in enumerate(colors):
            panels[0, column].plot(
                distances, relative[:, state], "o-", ms=3.0, lw=1.2,
                color=color, label=f"S{state}"
            )
        panels[0, column].set(
            xlabel=r"$R_{\rm OH}$ ($\AA$)",
            ylabel=r"$E_i(R)-E_{S_0}(R_{\rm eq})$ (eV)",
            title=f"{backend_label} SA(6)-CASSCF(10,10)",
        )
        panels[0, column].legend(ncol=2, fontsize=7.5)

        gradients = [float(record["orbital_gradient"]) for record in records]
        panels[1, column].semilogy(distances, gradients, "o-", label="orbital gradient")
        continuity_x, continuity = [], []
        for distance, record in zip(distances, records):
            if "active_singular_values" in record:
                continuity_x.append(distance)
                continuity.append(np.min(record["active_singular_values"]))
        twin = panels[1, column].twinx()
        if continuity:
            twin.plot(continuity_x, continuity, "s--", color="tab:orange", label="active overlap")
        panels[1, column].axhline(1.0e-3, color="0.45", ls=":", lw=1)
        panels[1, column].set(
            xlabel=r"$R_{\rm OH}$ ($\AA$)", ylabel=r"final $|g_{\rm orb}|$",
            title="Relaxation and transported-space continuity",
        )
        twin.set_ylabel("minimum active-overlap singular value")
        twin.set_ylim(0, 1.05)

    png = output / "phenol_sa6_casscf_sequential.png"
    pdf = output / "phenol_sa6_casscf_sequential.pdf"
    figure.savefig(png, dpi=280)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf, assembled


def plot_backend_comparison(output, distances, assembled):
    if not {"pyscf", "pyqed"}.issubset(assembled):
        return None, None, None
    reference = assembled["pyscf"][2]
    this_work = assembled["pyqed"][2]
    error_mev = np.abs(this_work - reference) * 1000.0
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, reference.shape[1]))
    figure, panels = plt.subplots(
        1, 2, figsize=(10.0, 4.1), constrained_layout=True,
        gridspec_kw={"width_ratios": (1.55, 1.0)},
    )
    for state, color in enumerate(colors):
        panels[0].plot(
            distances, reference[:, state], "-o", color=color, ms=3.8,
            lw=1.15, label=f"S{state}",
        )
        panels[0].plot(
            distances, this_work[:, state], linestyle="none", marker="s",
            ms=4.1, markerfacecolor="none", markeredgewidth=0.9,
            markeredgecolor=color,
        )
        panels[1].plot(
            distances, error_mev[:, state], "-o", color=color, ms=3.2,
            lw=1.05, label=f"S{state}",
        )
    state_legend = panels[0].legend(ncol=2, fontsize=7.6, loc="upper right")
    panels[0].add_artist(state_legend)
    panels[0].legend(
        handles=[
            Line2D([], [], color="0.2", marker="o", label="Reference (PySCF)"),
            Line2D(
                [], [], color="0.2", linestyle="none", marker="s",
                markerfacecolor="none", label="This work (PyQED)",
            ),
        ],
        fontsize=7.6,
        loc="center right",
    )
    panels[0].set(
        xlabel=r"$R_{\rm OH}$ ($\AA$)",
        ylabel=r"$E_i(R)-E_{S_0}(R_{\rm eq})$ (eV)",
        title="Fully relaxed adiabatic PESs",
    )
    panels[1].axhline(1.0, color="0.5", lw=0.9, ls=":")
    panels[1].set(
        xlabel=r"$R_{\rm OH}$ ($\AA$)",
        ylabel="absolute backend difference (meV)",
        title="PyQED versus PySCF",
    )
    panels[1].set_ylim(bottom=0)
    png = output / "phenol_sa6_casscf_apes_comparison.png"
    pdf = output / "phenol_sa6_casscf_apes_comparison.pdf"
    figure.savefig(png, dpi=280)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf, error_mev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31+g*")
    parser.add_argument("--nstates", type=int, default=6)
    parser.add_argument("--backends", nargs="+", choices=("pyscf", "pyqed"), default=["pyscf", "pyqed"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--macro-cycles", type=int, default=50)
    parser.add_argument("--micro-cycles", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--conv-tol", type=float, default=2.0e-7)
    parser.add_argument("--conv-grad", type=float, default=1.0e-5)
    parser.add_argument("--conv-step", type=float, default=1.0e-3)
    parser.add_argument("--scf-tol", type=float, default=1.0e-9)
    parser.add_argument("--spin-shift", type=float, default=1.0)
    parser.add_argument(
        "--pyqed-optimizer", choices=("AH", "LBFGS", "DIAG"), default="AH"
    )
    parser.add_argument("--pyqed-max-step", type=float, default=0.025)
    parser.add_argument(
        "--pyqed-coupling",
        choices=("qn", "relaxed_fd", "full"),
        default="qn",
    )
    parser.add_argument("--pyqed-ah-cycles", type=int, default=20)
    parser.add_argument("--pyqed-ah-subspace", type=int, default=24)
    parser.add_argument("--pyqed-ah-tol", type=float, default=1.0e-7)
    parser.add_argument("--pyqed-keyframe-interval", type=int, default=4)
    parser.add_argument("--pyqed-keyframe-gradient-trust", type=float, default=3.0)
    parser.add_argument("--pyqed-active-overlap-floor", type=float, default=0.35)
    parser.add_argument(
        "--pyqed-micro-ci-mode",
        choices=("keyframe", "full"),
        default="keyframe",
    )
    parser.add_argument(
        "--pyqed-reference-dir",
        type=Path,
        help="Completed PySCF cut whose optimized subspaces seed PyQED after "
        "Procrustes alignment to the transported PyQED frame.",
    )
    parser.add_argument("--seed-anchor", type=Path, default=Path("/private/tmp/phenol_sa6_penalty.npz"))
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/phenol_sa6_casscf_sequential"))
    parser.add_argument(
        "--distances", type=float, nargs="*",
        default=[0.90, 0.94, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.55, 1.70, 1.85, 1.95, 2.05, 2.20, 2.50, 3.00],
    )
    args = parser.parse_args()
    if not args.seed_anchor.exists():
        raise FileNotFoundError(f"SA-CASSCF seed anchor not found: {args.seed_anchor}")
    args.output.mkdir(parents=True, exist_ok=True)
    anchor_distance = float(PhenolReactiveChart().equilibrium[0])
    distances = sorted(set(float(value) for value in args.distances) | {anchor_distance})
    options = {
        "macro_cycles": args.macro_cycles,
        "micro_cycles": args.micro_cycles,
        "restarts": args.restarts,
        "conv_tol": args.conv_tol,
        "conv_grad": args.conv_grad,
        "conv_step": args.conv_step,
        "scf_tol": args.scf_tol,
        "spin_shift": args.spin_shift,
        "pyqed_optimizer": args.pyqed_optimizer,
        "pyqed_max_step": args.pyqed_max_step,
        "pyqed_coupling": args.pyqed_coupling,
        "pyqed_ah_cycles": args.pyqed_ah_cycles,
        "pyqed_ah_subspace": args.pyqed_ah_subspace,
        "pyqed_ah_tol": args.pyqed_ah_tol,
        "pyqed_keyframe_interval": args.pyqed_keyframe_interval,
        "pyqed_keyframe_gradient_trust": args.pyqed_keyframe_gradient_trust,
        "pyqed_active_overlap_floor": args.pyqed_active_overlap_floor,
        "pyqed_micro_ci_mode": args.pyqed_micro_ci_mode,
        "pyqed_reference_dir": (
            None if args.pyqed_reference_dir is None else str(args.pyqed_reference_dir)
        ),
    }
    context = mp.get_context("spawn")
    anchors = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(args.backends)), mp_context=context) as pool:
        futures = [
            pool.submit(
                anchor_worker,
                (backend, str(args.output), str(args.seed_anchor), args.basis, args.nstates, anchor_distance, options),
            )
            for backend in args.backends
        ]
        for future in as_completed(futures):
            backend, path, status = future.result()
            anchors[backend] = path
            print(f"[{backend}:anchor] {status}: {path}", flush=True)

    increasing = [value for value in distances if value > anchor_distance]
    decreasing = sorted(
        [value for value in distances if value < anchor_distance], reverse=True
    )
    tasks = []
    for backend in args.backends:
        if increasing:
            tasks.append((backend, "increasing", increasing, str(args.output), anchors[backend], args.basis, args.nstates, options))
        if decreasing:
            tasks.append((backend, "decreasing", decreasing, str(args.output), anchors[backend], args.basis, args.nstates, options))
    if tasks:
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(tasks)), mp_context=context
        ) as pool:
            futures = [pool.submit(chain_worker, task) for task in tasks]
            for future in as_completed(futures):
                backend, direction, completed = future.result()
                print(
                    f"[{backend}:{direction}] complete ({len(completed)} geometries)",
                    flush=True,
                )

    png, pdf, assembled = plot_results(
        args.output, args.backends, np.asarray(distances), anchor_distance
    )
    comparison_png, comparison_pdf, error_mev = plot_backend_comparison(
        args.output, np.asarray(distances), assembled
    )
    summary = {
        "method": f"SA({args.nstates})-CASSCF(10e,10o)/{args.basis}",
        "orbital_relaxed_at_every_geometry": True,
        "sequential_transport": ["full MO frame", "all CI roots"],
        "parallelization": "independent backend/direction chains only",
        "backends": args.backends,
        "workers": args.workers,
        "distances_angstrom": distances,
        "anchor_distance_angstrom": anchor_distance,
        "gradient_threshold": args.conv_grad,
        "spin_constraint": "fix_spin(ss=0)",
        "spin_shift": args.spin_shift,
        "pyqed_optimizer": args.pyqed_optimizer,
        "pyqed_coupling": args.pyqed_coupling,
        "pyqed_max_step": args.pyqed_max_step,
        "pyqed_ah_cycles": args.pyqed_ah_cycles,
        "pyqed_ah_subspace": args.pyqed_ah_subspace,
        "pyqed_keyframe_interval": args.pyqed_keyframe_interval,
        "pyqed_active_overlap_floor": args.pyqed_active_overlap_floor,
        "pyqed_micro_ci_mode": args.pyqed_micro_ci_mode,
        "pyqed_reference_dir": (
            None if args.pyqed_reference_dir is None else str(args.pyqed_reference_dir)
        ),
        "pyqed_orbital_initialization": (
            "sequentially transported PyQED orbitals"
            if args.pyqed_reference_dir is None
            else "optimized PySCF core/active/external subspaces, blockwise "
            "Procrustes-aligned to the sequentially transported PyQED frame, "
            "followed by full PyQED orbital relaxation"
        ),
        "figure": str(png),
        "figure_pdf": str(pdf),
    }
    if comparison_png is not None:
        pyscf_energies = assembled["pyscf"][1]
        pyqed_energies = assembled["pyqed"][1]
        summary.update(
            {
                "comparison_figure": str(comparison_png),
                "comparison_figure_pdf": str(comparison_pdf),
                "max_backend_error_mev": float(np.max(error_mev)),
                "rms_backend_error_mev": float(np.sqrt(np.mean(error_mev**2))),
            }
        )
        for label, energies in (
            ("pyscf", pyscf_energies),
            ("pyqed", pyqed_energies),
        ):
            gap12 = (energies[:, 2] - energies[:, 1]) * HARTREE_TO_EV
            gap01 = (energies[:, 1] - energies[:, 0]) * HARTREE_TO_EV
            index12 = int(np.argmin(gap12))
            index01 = int(np.argmin(gap01))
            summary[f"{label}_minimum_s1_s2_gap_ev"] = float(gap12[index12])
            summary[f"{label}_minimum_s1_s2_gap_distance_angstrom"] = distances[index12]
            summary[f"{label}_minimum_s0_s1_gap_ev"] = float(gap01[index01])
            summary[f"{label}_minimum_s0_s1_gap_distance_angstrom"] = distances[index01]
        np.savez_compressed(
            args.output / "phenol_sa6_casscf_full_cut.npz",
            distances=np.asarray(distances),
            pyscf_energies=pyscf_energies,
            pyqed_energies=pyqed_energies,
            backend_error_mev=error_mev,
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
