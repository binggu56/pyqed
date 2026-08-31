#!/usr/bin/env python3
"""Full rectilinear CASCI/LDR reference for the H3+ S1/S2 dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from examples.ldr.h3plus_rectilinear_cgldr import h3plus_geometry
from pyqed.dvr import DVR
from pyqed.ldr.observables import nuclear_observables
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casci import CASCI, overlap
from pyqed.units import amu_to_au, au2fs


STATE_IDS = (1, 2)


def run_casci_point(coordinates, *, basis="sto-3g"):
    geometry = h3plus_geometry(coordinates)
    mol = Molecule(
        atom=[["H", *xyz] for xyz in geometry],
        unit="bohr",
        basis=basis,
        charge=1,
        spin=0,
    )
    mol.build()
    mf = mol.RHF().run()
    return CASCI(mf, ncas=3, nelecas=2).run(nstates=3)


def state_overlap(left, right, *, unitarize=False):
    block = np.asarray(overlap(left, right), dtype=complex)[
        np.ix_(STATE_IDS, STATE_IDS)
    ]
    if not unitarize:
        return block
    u, _, vh = np.linalg.svd(block, full_matrices=False)
    return u @ vh


def scan_electronic_data(dvr, *, basis="sto-3g", unitarize_links=False):
    """Scan energies and nearest-neighbor S1/S2 overlap links."""
    nqs, nqx, nqy = dvr.shape
    energies = np.empty((nqs, nqx, nqy, 2))
    links = (
        np.empty((nqs - 1, nqx, nqy, 2, 2), dtype=complex),
        np.empty((nqs, nqx - 1, nqy, 2, 2), dtype=complex),
        np.empty((nqs, nqx, nqy - 1, 2, 2), dtype=complex),
    )
    previous_plane = None
    completed = 0
    total = dvr.size

    for i, qs in enumerate(dvr.x[0]):
        current_plane = [[None] * nqy for _ in range(nqx)]
        for j, qx in enumerate(dvr.x[1]):
            for k, qy in enumerate(dvr.x[2]):
                point = run_casci_point(
                    {"Qs": float(qs), "Qx": float(qx), "Qy": float(qy)},
                    basis=basis,
                )
                energies[i, j, k] = np.asarray(point.e_tot)[list(STATE_IDS)]
                if i:
                    links[0][i - 1, j, k] = state_overlap(
                        previous_plane[j][k],
                        point,
                        unitarize=unitarize_links,
                    )
                if j:
                    links[1][i, j - 1, k] = state_overlap(
                        current_plane[j - 1][k],
                        point,
                        unitarize=unitarize_links,
                    )
                if k:
                    links[2][i, j, k - 1] = state_overlap(
                        current_plane[j][k - 1],
                        point,
                        unitarize=unitarize_links,
                    )
                current_plane[j][k] = point
                completed += 1
                if completed == 1 or completed % max(1, total // 20) == 0:
                    print(f"[scan] {completed}/{total}")
        previous_plane = current_plane
    return energies, links


def save_electronic_data(
    filename,
    dvr,
    energies,
    links,
    *,
    unitarize_links=False,
):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        filename,
        energies=energies,
        links_0=links[0],
        links_1=links[1],
        links_2=links[2],
        grid_0=dvr.x[0],
        grid_1=dvr.x[1],
        grid_2=dvr.x[2],
        state_ids=np.asarray(STATE_IDS),
        basis=np.asarray("sto-3g"),
        unitarize_links=np.asarray(unitarize_links),
    )


def load_electronic_data(filename, dvr, *, unitarize_links=False):
    archive = np.load(filename)
    for axis in range(3):
        np.testing.assert_allclose(archive[f"grid_{axis}"], dvr.x[axis])
    if tuple(archive["state_ids"]) != STATE_IDS:
        raise ValueError("Cached state_ids do not match (1, 2).")
    cached_unitarization = bool(archive["unitarize_links"])
    if cached_unitarization != unitarize_links:
        raise ValueError(
            "Cached overlap-link unitarization does not match this run."
        )
    return archive["energies"], tuple(
        archive[f"links_{axis}"] for axis in range(3)
    )


def linked_line_overlaps(neighbor_links):
    """Build all pairwise transports along one DVR line."""
    neighbor_links = np.asarray(neighbor_links, dtype=complex)
    npoints = neighbor_links.shape[0] + 1
    nstates = neighbor_links.shape[-1]
    transports = np.empty((npoints, npoints, nstates, nstates), dtype=complex)
    for start in range(npoints):
        transports[start, start] = np.eye(nstates)
        running = np.eye(nstates, dtype=complex)
        for stop in range(start + 1, npoints):
            running = running @ neighbor_links[stop - 1]
            transports[start, stop] = running
            transports[stop, start] = running.conj().T
    return transports


def secondary_electronic_frames(dvr, links):
    """Map the electronic frame at ``Qx=Qy=0`` onto every secondary point."""
    qx_reference = int(np.argmin(np.abs(dvr.x[1])))
    qy_reference = int(np.argmin(np.abs(dvr.x[2])))
    frames = np.empty((*dvr.shape, 2, 2), dtype=complex)
    for i in range(dvr.shape[0]):
        qx_transports = linked_line_overlaps(
            links[1][i, :, qy_reference]
        )
        for j in range(dvr.shape[1]):
            qy_transports = linked_line_overlaps(links[2][i, j, :])
            for k in range(dvr.shape[2]):
                local_to_center = (
                    qy_transports[k, qy_reference]
                    @ qx_transports[j, qx_reference]
                )
                u, _, vh = np.linalg.svd(
                    local_to_center,
                    full_matrices=False,
                )
                frames[i, j, k] = u @ vh
    return frames


def _flat_indices(shape, axis, fixed):
    indices = np.empty((shape[axis], 2), dtype=int)
    for coordinate in range(shape[axis]):
        point = list(fixed)
        point.insert(axis, coordinate)
        nuclear = np.ravel_multi_index(tuple(point), shape)
        indices[coordinate] = (2 * nuclear, 2 * nuclear + 1)
    return indices


def build_sparse_hamiltonian(dvr, energies, links):
    """Assemble the full linked-overlap LDR Hamiltonian."""
    shape = dvr.shape
    rows = []
    cols = []
    values = []

    for axis in range(3):
        other_shape = shape[:axis] + shape[axis + 1 :]
        kinetic = np.asarray(dvr.axes[axis].t(), dtype=complex)
        for fixed in np.ndindex(*other_shape):
            selector = list(fixed)
            selector.insert(axis, slice(None))
            line_links = links[axis][tuple(selector)]
            transports = linked_line_overlaps(line_links)
            block = kinetic[:, :, None, None] * transports
            line = _flat_indices(shape, axis, fixed)
            block_rows = np.broadcast_to(
                line[:, None, :, None],
                block.shape,
            )
            block_cols = np.broadcast_to(
                line[None, :, None, :],
                block.shape,
            )
            rows.append(block_rows.reshape(-1))
            cols.append(block_cols.reshape(-1))
            values.append(block.reshape(-1))

    dimension = 2 * int(np.prod(shape))
    nuclear = sp.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dimension, dimension),
    ).tocsr()
    potential = sp.diags(np.asarray(energies).reshape(-1), format="csr")
    hamiltonian = nuclear + potential
    return 0.5 * (hamiltonian + hamiltonian.getH())


def initial_wavepacket(dvr, *, electronic_frames=None):
    grids = np.meshgrid(*dvr.x, indexing="ij")
    centers = (-0.20, -0.015, 0.0)
    sigma = (0.06, 0.04, 0.04)
    amplitude = np.ones(dvr.shape)
    for grid, center, width in zip(grids, centers, sigma):
        amplitude *= np.exp(-0.25 * ((grid - center) / width) ** 2)
    state = np.zeros((*dvr.shape, 2), dtype=complex)
    if electronic_frames is None:
        state[..., 1] = amplitude
    else:
        state[...] = amplitude[..., None] * electronic_frames[..., :, 1]
    state /= np.linalg.norm(state)
    return state


def propagate(
    hamiltonian,
    initial,
    *,
    final_time,
    samples,
    electronic_frames=None,
):
    trace = -1j * hamiltonian.diagonal().sum()
    states = expm_multiply(
        -1j * hamiltonian,
        initial.reshape(-1),
        start=0.0,
        stop=final_time,
        num=samples,
        endpoint=True,
        traceA=trace,
    )
    states = states.reshape(samples, *initial.shape)
    population_states = states
    if electronic_frames is not None:
        population_states = np.einsum(
            "...ab,t...a->t...b",
            electronic_frames.conj(),
            states,
            optimize=True,
        )
    populations = np.sum(
        np.abs(population_states) ** 2,
        axis=tuple(range(1, states.ndim - 1)),
    )
    times = np.linspace(0.0, final_time, samples)
    return times, states, populations


def plot_populations(times, populations, output, *, comparison=None):
    fig, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    colors = ("tab:blue", "tab:orange")
    for state, color in enumerate(colors):
        axis.plot(
            times * au2fs,
            populations[:, state],
            color=color,
            label=rf"full LDR $S_{state + 1}$",
            linewidth=2,
        )
    if comparison is not None:
        archive = np.load(comparison)
        for state, color in enumerate(colors):
            axis.plot(
                archive["times_fs"],
                archive["populations"][:, state],
                color=color,
                linestyle="--",
                label=rf"CGLDR $S_{state + 1}$",
                linewidth=1.8,
            )
    axis.set(
        xlabel="Time / fs",
        ylabel="Electronic population",
        ylim=(-0.02, 1.02),
        title=r"Rectilinear CASCI dynamics for H$_3^+$",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def build_dvr(npts):
    reference = h3plus_geometry({"Qs": 0.0, "Qx": 0.0, "Qy": 0.0})
    mol = Molecule(
        atom=[["H", *xyz] for xyz in reference],
        unit="bohr",
        basis="sto-3g",
        charge=1,
    )
    mol.build()
    mass = float(mol.atom_mass_list()[0]) * amu_to_au
    return DVR(
        domains=((-0.40, 0.80), (-0.20, 0.20), (-0.20, 0.20)),
        npts=npts,
        mass=(mass, mass, mass),
        names=("Qs", "Qx", "Qy"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npts", type=int, nargs=3, default=(64, 9, 9))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("/private/tmp/h3plus_full_rectilinear_casci_ldr.npz"),
    )
    parser.add_argument("--force-scan", action="store_true")
    parser.add_argument(
        "--unitarize-overlap-links",
        action="store_true",
        help="Replace each CASCI neighbor overlap by its polar unitary.",
    )
    parser.add_argument("--final-time-fs", type=float, default=2.0)
    parser.add_argument("--samples", type=int, default=34)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/h3plus_full_rectilinear_casci_ldr_dynamics.npz"),
    )
    parser.add_argument("--plot", type=Path)
    parser.add_argument(
        "--comparison",
        type=Path,
        help="Optional CGLDR dynamics NPZ to overlay on the population plot.",
    )
    args = parser.parse_args()

    dvr = build_dvr(tuple(args.npts))
    if args.cache.exists() and not args.force_scan:
        energies, links = load_electronic_data(
            args.cache,
            dvr,
            unitarize_links=args.unitarize_overlap_links,
        )
        print(f"[cache] loaded {args.cache}")
    else:
        start = time.perf_counter()
        energies, links = scan_electronic_data(
            dvr,
            unitarize_links=args.unitarize_overlap_links,
        )
        save_electronic_data(
            args.cache,
            dvr,
            energies,
            links,
            unitarize_links=args.unitarize_overlap_links,
        )
        print(f"[scan] completed in {time.perf_counter() - start:.2f} s")
        print(f"[cache] saved {args.cache}")

    hamiltonian = build_sparse_hamiltonian(dvr, energies, links)
    hermiticity = sp.linalg.norm(hamiltonian - hamiltonian.getH())
    print(
        f"[hamiltonian] shape={hamiltonian.shape}, "
        f"nnz={hamiltonian.nnz}, hermiticity={hermiticity:.3e}"
    )
    electronic_frames = secondary_electronic_frames(dvr, links)
    initial = initial_wavepacket(
        dvr,
        electronic_frames=electronic_frames,
    )
    times, states, populations = propagate(
        hamiltonian,
        initial,
        final_time=args.final_time_fs / au2fs,
        samples=args.samples,
        electronic_frames=electronic_frames,
    )
    norms = np.sum(np.abs(states) ** 2, axis=tuple(range(1, states.ndim)))
    observables = nuclear_observables(
        states,
        dvr.x,
        electronic_axis=-1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        times_au=times,
        times_fs=times * au2fs,
        populations=populations,
        norms=norms,
        npts=np.asarray(dvr.shape),
        nuclear_density=observables["nuclear_density"],
        coordinate_means=observables["coordinate_means"],
        coordinate_covariance=observables["coordinate_covariance"],
        survival_probability=observables["survival_probability"],
    )
    print("[result] final populations:", populations[-1])
    print("[result] norm range:", norms.min(), norms.max())
    print(f"[result] saved {args.output}")
    if args.plot is not None:
        plot_populations(
            times,
            populations,
            args.plot,
            comparison=args.comparison,
        )
        print(f"[plot] {args.plot}")


if __name__ == "__main__":
    main()
