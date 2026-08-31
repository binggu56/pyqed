#!/usr/bin/env python3
"""Ab initio excited-state partial charges along phenol O--H stretching.

The calculation uses RHF/CIS/STO-3G and constructs a state-specific
one-particle density for every CIS root.  Mulliken and meta-Lowdin charges are
then evaluated from those AO density matrices.  The small basis makes this a
quick method demonstration, not a converged photodissociation calculation.

Run from the repository root with

    PYTHONPATH=. python examples/ldr/phenol_abinitio_partial_charges.py
"""

from __future__ import annotations
from pyqed.units import au2ev

import argparse
import json
import os
import traceback
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
from pyscf import gto, scf, tdscf
from scipy.optimize import linear_sum_assignment

HARTREE_TO_EV = au2ev
ATOM_LABELS = ("C1", "C2", "C3", "C4", "C5", "C6", "O", "H_O", "H2", "H3", "H4", "H5", "H6")


def phenol_geometry(r_oh):
    """Return a fixed planar phenol geometry with one scanned O--H distance."""

    r_cc, r_co, r_ch = 1.397, 1.360, 1.080
    carbons = []
    for index in range(6):
        angle = 2.0 * np.pi * index / 6.0
        carbons.append(r_cc * np.array((np.cos(angle), np.sin(angle), 0.0)))

    atoms = [("C", xyz) for xyz in carbons]
    oxygen = carbons[0] + np.array((r_co, 0.0, 0.0))
    atoms.extend((("O", oxygen), ("H", oxygen + np.array((r_oh, 0.0, 0.0)))))
    for index in range(1, 6):
        direction = carbons[index] / np.linalg.norm(carbons[index])
        atoms.append(("H", carbons[index] + r_ch * direction))
    return atoms


def excited_density(mf, amplitude):
    r"""Return the unrelaxed singlet TDA density for normalized $X_{ia}$."""

    x = np.asarray(amplitude, dtype=float)
    x = x / np.linalg.norm(x)
    occupied = mf.mo_occ > 0
    virtual = mf.mo_occ == 0
    density_mo = np.diag(mf.mo_occ.copy())
    # Spin-summed CIS/TDA difference density: -XX^T in occupied space and
    # +X^TX in virtual space.  Its trace is zero.
    density_mo[np.ix_(occupied, occupied)] -= x @ x.T
    density_mo[np.ix_(virtual, virtual)] += x.T @ x
    return mf.mo_coeff @ density_mo @ mf.mo_coeff.T


def atomic_charges(mol, density, overlap):
    _, mulliken = scf.hf.mulliken_pop(mol, density, s=overlap, verbose=0)
    _, lowdin = scf.hf.mulliken_pop_meta_lowdin_ao(mol, density, verbose=0)
    return np.asarray(mulliken), np.asarray(lowdin)


def dominant_excitations(mf, amplitude, nkeep=3):
    x = np.asarray(amplitude)
    occupied = np.flatnonzero(mf.mo_occ > 0)
    virtual = np.flatnonzero(mf.mo_occ == 0)
    order = np.argsort(np.abs(x).ravel())[::-1][:nkeep]
    result = []
    for flat in order:
        i, a = np.unravel_index(flat, x.shape)
        result.append((int(occupied[i]), int(virtual[a]), float(x[i, a])))
    return result


def calculate_point(r_oh, basis, nstates):
    mol = gto.M(
        atom=phenol_geometry(r_oh), basis=basis, unit="Angstrom",
        charge=0, spin=0, symmetry=False, verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-9
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        mf = mf.newton().run()
    if not mf.converged:
        raise RuntimeError(f"SCF failed at R_OH={r_oh:.3f} angstrom")

    tda = tdscf.TDA(mf)
    tda.nstates = nstates
    tda.conv_tol = 1.0e-7
    tda.kernel()
    if not np.all(tda.converged):
        raise RuntimeError(f"TDA failed at R_OH={r_oh:.3f} angstrom")

    overlap = mf.get_ovlp()
    ground_mulliken, ground_lowdin = atomic_charges(mol, mf.make_rdm1(), overlap)
    amplitudes = np.array([xy[0] for xy in tda.xy])
    mulliken, lowdin = [], []
    for amplitude in amplitudes:
        density = excited_density(mf, amplitude)
        charge_m, charge_l = atomic_charges(mol, density, overlap)
        mulliken.append(charge_m)
        lowdin.append(charge_l)
    return {
        "scf_energy": float(mf.e_tot),
        "excitation_ev": np.asarray(tda.e) * HARTREE_TO_EV,
        "amplitudes": amplitudes,
        "mulliken": np.asarray(mulliken),
        "lowdin": np.asarray(lowdin),
        "ground_mulliken": ground_mulliken,
        "ground_lowdin": ground_lowdin,
        "dominant": [dominant_excitations(mf, x) for x in amplitudes],
        "mol": mol,
        "occupied_mo": mf.mo_coeff[:, mf.mo_occ > 0],
        "virtual_mo": mf.mo_coeff[:, mf.mo_occ == 0],
    }


def track_roots(points):
    """Track roots by TDA-amplitude overlap, then apply each permutation."""

    npoints = len(points)
    nstates = len(points[0]["excitation_ev"])
    tracked = {key: np.empty((npoints, nstates, *np.asarray(points[0][key]).shape[1:]))
               for key in ("excitation_ev", "amplitudes", "mulliken", "lowdin")}
    permutations = np.empty((npoints, nstates), dtype=int)
    permutations[0] = np.arange(nstates)
    previous = points[0]["amplitudes"]
    for key in tracked:
        tracked[key][0] = points[0][key]

    for index in range(1, npoints):
        current = points[index]["amplitudes"]
        ao_overlap = gto.intor_cross("int1e_ovlp", points[index - 1]["mol"], points[index]["mol"])
        occupied_overlap = (
            points[index - 1]["occupied_mo"].T @ ao_overlap @ points[index]["occupied_mo"]
        )
        virtual_overlap = (
            points[index - 1]["virtual_mo"].T @ ao_overlap @ points[index]["virtual_mo"]
        )
        # Overlap of the singly excited parts, including the changing AO/MO
        # bases.  This is substantially safer than comparing amplitude array
        # indices at two geometries.
        overlap = np.abs(
            np.einsum(
                "sia,ij,tjb,ab->st",
                previous,
                occupied_overlap,
                current,
                virtual_overlap,
                optimize=True,
            )
        )
        rows, cols = linear_sum_assignment(-overlap)
        permutation = cols[np.argsort(rows)]
        permutations[index] = permutation
        for key in tracked:
            tracked[key][index] = points[index][key][permutation]
        previous = current[permutation]
    tracked["permutations"] = permutations
    return tracked


def charge_track(points, key="lowdin"):
    """Track roots using only full-molecule state-specific charge vectors."""

    npoints = len(points)
    nstates = len(points[0]["excitation_ev"])
    permutations = np.empty((npoints, nstates), dtype=int)
    permutations[0] = np.arange(nstates)
    previous = points[0][key]
    for index in range(1, npoints):
        current = points[index][key]
        cost = np.linalg.norm(previous[:, None, :] - current[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        permutation = cols[np.argsort(rows)]
        permutations[index] = permutation
        previous = current[permutation]
    return permutations


def plot_results(r_oh, data, output):
    energies = data["excitation_ev"]
    mulliken = data["mulliken"]
    lowdin = data["lowdin"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, energies.shape[1]))
    for state, color in enumerate(colors):
        axes[0].plot(r_oh, energies[:, state], "o-", color=color, label=f"tracked S{state + 1}")
        axes[1].plot(r_oh, mulliken[:, state, 7], "o-", color=color)
        axes[2].plot(r_oh, lowdin[:, state, 7], "o-", color=color)
    axes[0].set_ylabel("vertical excitation energy (eV)")
    axes[1].set_ylabel(r"Mulliken charge on dissociating H ($e$)")
    axes[2].set_ylabel(r"Löwdin charge on dissociating H ($e$)")
    for axis in axes:
        axis.set_xlabel(r"$R_{\mathrm{OH}}$ (angstrom)")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Ab initio RHF/CIS excited states")
    axes[1].set_title("State-specific density")
    axes[2].set_title("Population-scheme check")
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--nstates", type=int, default=4)
    parser.add_argument("--npoints", type=int, default=9)
    parser.add_argument("--rmin", type=float, default=0.90)
    parser.add_argument("--rmax", type=float, default=1.15)
    parser.add_argument("--outdir", type=Path, default=Path("/private/tmp/phenol_abinitio_charges"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    r_oh = np.linspace(args.rmin, args.rmax, args.npoints)
    points = []
    for index, distance in enumerate(r_oh):
        print(f"[{index + 1}/{len(r_oh)}] R_OH={distance:.3f} angstrom", flush=True)
        try:
            points.append(calculate_point(distance, args.basis, args.nstates))
        except Exception:
            traceback.print_exc()
            raise
    tracked = track_roots(points)
    charge_permutations = charge_track(points)

    figure = args.outdir / "phenol_abinitio_partial_charges.png"
    data_file = args.outdir / "phenol_abinitio_partial_charges.npz"
    summary_file = args.outdir / "phenol_abinitio_partial_charges.json"
    plot_results(r_oh, tracked, figure)
    np.savez(
        data_file, r_oh_angstrom=r_oh,
        excitation_ev=tracked["excitation_ev"],
        mulliken_charges=tracked["mulliken"],
        lowdin_charges=tracked["lowdin"],
        root_permutations=tracked["permutations"],
        charge_root_permutations=charge_permutations,
        atom_labels=np.asarray(ATOM_LABELS),
    )
    hydrogen_m = tracked["mulliken"][:, :, 7]
    hydrogen_l = tracked["lowdin"][:, :, 7]
    summary = {
        "method": f"RHF/CIS/{args.basis}",
        "density": "state-specific CIS one-particle density",
        "r_oh_angstrom": r_oh.tolist(),
        "hydrogen_mulliken_range": [float(hydrogen_m.min()), float(hydrogen_m.max())],
        "hydrogen_lowdin_range": [float(hydrogen_l.min()), float(hydrogen_l.max())],
        "max_mulliken_lowdin_disagreement": float(np.max(np.abs(hydrogen_m - hydrogen_l))),
        "charge_tracking_root_accuracy": float(
            np.mean(charge_permutations[1:] == tracked["permutations"][1:])
        ),
        "charge_tracking_all_points_exact": bool(
            np.all(charge_permutations[1:] == tracked["permutations"][1:])
        ),
    }
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"figure: {figure}")
    print(f"data: {data_file}")


if __name__ == "__main__":
    main()
