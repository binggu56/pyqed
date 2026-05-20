#!/usr/bin/env python3
"""H3+ sparse-APES FE-DVR LDR with local/linked overlaps vs sine DVR-LDR.

This fixed-theta benchmark uses AM1/MECI electronic states for speed.  The
reference calculation is sine DVR-LDR with APES and electronic overlaps on the
sine grid.  The comparison calculation propagates on an FE-DVR grid, uses APES
interpolated from a sparse grid, and compares direct local electronic overlaps
against linked nearest-neighbor products on the nonzero kinetic couplings of the
LDR Hamiltonian.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.dvr import FEDVR, SineDVR
from pyqed.qchem import Molecule
from pyqed.smolyak.interpolator import SparseInterpolator
from pyqed.units import amu2au, au2fs

from h3plus_fedvr_fixed_theta import (
    fixed_theta_stretch_kinetic,
    h3plus_body_frame,
    initial_packet,
    region_masks,
    region_population,
)


def run_am1_meci(r1, r2, theta, args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pyqed.qchem.semiempirical.am1 import RAM1

    mol = Molecule(
        atom=h3plus_body_frame(r1, r2, theta),
        charge=1,
        spin=0,
        unit="bohr",
    )
    mf = RAM1(mol).run(
        conv_tol=args.scf_tol,
        max_cycle=args.max_cycle,
        verbose=0,
        damping=args.damping,
    )
    return mf.MECI(nstates=args.nstates, ncas=args.ncas).run()


def electronic_energies(obj, nstates):
    energies = getattr(obj, "e_tot", getattr(obj, "e", None))
    if energies is None:
        raise AttributeError("MECI object has neither e_tot nor e.")
    return np.asarray(energies, dtype=float)[:nstates]


def electronic_overlap(left, right, nstates):
    if left is right:
        return np.eye(nstates, dtype=complex)
    block = np.asarray(left.wavefunction_overlap(right), dtype=complex)
    if block.shape != (nstates, nstates):
        raise ValueError(f"Overlap block shape {block.shape} != {(nstates, nstates)}")
    return block


def grid_points(r1_dvr, r2_dvr):
    r1, r2 = np.meshgrid(r1_dvr.x, r2_dvr.x, indexing="ij")
    return np.column_stack([r1.reshape(-1), r2.reshape(-1)])


def scan_grid_objects(r1_dvr, r2_dvr, theta, args, label):
    objects = np.empty((r1_dvr.npts, r2_dvr.npts), dtype=object)
    apes = np.zeros((r1_dvr.npts, r2_dvr.npts, args.nstates), dtype=float)
    total = r1_dvr.npts * r2_dvr.npts
    t0 = time.perf_counter()
    count = 0
    for i, r1 in enumerate(r1_dvr.x):
        for j, r2 in enumerate(r2_dvr.x):
            count += 1
            obj = run_am1_meci(r1, r2, theta, args)
            objects[i, j] = obj
            apes[i, j] = electronic_energies(obj, args.nstates)
            print(
                f"[{label} scan] {count:3d}/{total}: "
                f"r1={r1:.6f} r2={r2:.6f} E0={apes[i, j, 0]:.10f}"
            )
    return apes, objects, time.perf_counter() - t0


def sparse_grid_apes_on_targets(target, theta, args):
    interval = np.array(
        [[args.r_min, args.r_min], [args.r_max, args.r_max]],
        dtype=float,
    )
    samples = {}

    def sample(points):
        values = np.empty((len(points), args.nstates), dtype=float)
        for i, (r1, r2) in enumerate(points):
            key = (round(float(r1), 13), round(float(r2), 13))
            if key not in samples:
                obj = run_am1_meci(r1, r2, theta, args)
                samples[key] = electronic_energies(obj, args.nstates)
                print(
                    f"[sparse APES scan] {len(samples):3d}: "
                    f"r1={r1:.6f} r2={r2:.6f} E0={samples[key][0]:.10f}"
                )
            values[i] = samples[key]
        return values

    interpolated = np.zeros((len(target), args.nstates), dtype=float)
    for state in range(args.nstates):
        interpolator = SparseInterpolator(
            args.interp_level,
            2,
            interpolation_type=args.interp_type,
            interpolation_interval=interval,
            tol=0.0,
        )

        def state_energy(points, state=state):
            return sample(points)[:, state]

        interpolated[:, state] = interpolator.fit(state_energy, target)
    return interpolated, len(samples)


def interpolate_tensor_apes_to_dvr(r1_axis, r2_axis, apes, r1_dvr, r2_dvr):
    target = grid_points(r1_dvr, r2_dvr)
    interpolated = np.zeros((len(target), apes.shape[-1]), dtype=float)
    for state in range(apes.shape[-1]):
        interp = RegularGridInterpolator(
            (np.asarray(r1_axis), np.asarray(r2_axis)),
            apes[:, :, state],
            bounds_error=False,
            fill_value=None,
        )
        interpolated[:, state] = interp(target)
    return interpolated.reshape(r1_dvr.npts, r2_dvr.npts, apes.shape[-1])


def build_local_overlap_ldr_hamiltonian(r1_dvr, r2_dvr, theta, apes, objects, args):
    proton_mass = 1.00782503223 * amu2au
    masses_au = np.array([proton_mass, proton_mass, proton_mass])
    kinetic = fixed_theta_stretch_kinetic(r1_dvr, r2_dvr, theta, masses_au).tocoo()
    flat_objects = objects.reshape(-1)
    ngrid = len(flat_objects)
    nstates = args.nstates

    rows, cols, data = [], [], []
    overlap_cache = {}
    t0 = time.perf_counter()
    for i, j, value in zip(kinetic.row, kinetic.col, kinetic.data):
        if abs(value) <= args.kinetic_threshold:
            continue
        key = (int(i), int(j))
        if key not in overlap_cache:
            overlap_cache[key] = electronic_overlap(flat_objects[i], flat_objects[j], nstates)
        block = value * overlap_cache[key]
        nz_a, nz_b = np.nonzero(np.abs(block) > 1.0e-14)
        rows.extend((i * nstates + nz_a).tolist())
        cols.extend((j * nstates + nz_b).tolist())
        data.extend(block[nz_a, nz_b].tolist())
    overlap_time = time.perf_counter() - t0

    dim = ngrid * nstates
    kinetic_ldr = sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))
    kinetic_ldr = 0.5 * (kinetic_ldr + kinetic_ldr.getH())
    potential = sp.diags(apes.reshape(-1), format="csr", dtype=complex)
    return kinetic_ldr + potential, overlap_time, len(overlap_cache)


def build_neighbor_links(objects, nstates):
    nx, ny = objects.shape
    x_links = np.empty((max(nx - 1, 0), ny), dtype=object)
    y_links = np.empty((nx, max(ny - 1, 0)), dtype=object)
    t0 = time.perf_counter()
    count = 0
    for i in range(nx - 1):
        for j in range(ny):
            x_links[i, j] = electronic_overlap(objects[i, j], objects[i + 1, j], nstates)
            count += 1
    for i in range(nx):
        for j in range(ny - 1):
            y_links[i, j] = electronic_overlap(objects[i, j], objects[i, j + 1], nstates)
            count += 1
    return x_links, y_links, time.perf_counter() - t0, count


def linked_path_overlap(left, right, shape, x_links, y_links, nstates):
    if left == right:
        return np.eye(nstates, dtype=complex)

    start = np.array(np.unravel_index(left, shape), dtype=int)
    target = np.array(np.unravel_index(right, shape), dtype=int)

    def step_block(current, axis, direction):
        i, j = int(current[0]), int(current[1])
        if axis == 0 and direction > 0:
            return x_links[i, j]
        if axis == 0:
            return x_links[i - 1, j].conj().T
        if direction > 0:
            return y_links[i, j]
        return y_links[i, j - 1].conj().T

    def path_product(order):
        current = start.copy()
        block = np.eye(nstates, dtype=complex)
        for axis in order:
            while current[axis] != target[axis]:
                direction = 1 if target[axis] > current[axis] else -1
                block = block @ step_block(current, axis, direction)
                current[axis] += direction
        return block

    # Average the two Manhattan paths to reduce axis-order dependence.
    return 0.5 * (path_product((0, 1)) + path_product((1, 0)))


def build_linked_overlap_ldr_hamiltonian(r1_dvr, r2_dvr, theta, apes, objects, args):
    proton_mass = 1.00782503223 * amu2au
    masses_au = np.array([proton_mass, proton_mass, proton_mass])
    kinetic = fixed_theta_stretch_kinetic(r1_dvr, r2_dvr, theta, masses_au).tocoo()
    ngrid = objects.size
    nstates = args.nstates
    shape = objects.shape
    x_links, y_links, link_time, link_count = build_neighbor_links(objects, nstates)

    rows, cols, data = [], [], []
    overlap_cache = {}
    t0 = time.perf_counter()
    for i, j, value in zip(kinetic.row, kinetic.col, kinetic.data):
        if abs(value) <= args.kinetic_threshold:
            continue
        key = (int(i), int(j))
        if key not in overlap_cache:
            overlap_cache[key] = linked_path_overlap(i, j, shape, x_links, y_links, nstates)
        block = value * overlap_cache[key]
        nz_a, nz_b = np.nonzero(np.abs(block) > 1.0e-14)
        rows.extend((i * nstates + nz_a).tolist())
        cols.extend((j * nstates + nz_b).tolist())
        data.extend(block[nz_a, nz_b].tolist())
    path_time = time.perf_counter() - t0

    dim = ngrid * nstates
    kinetic_ldr = sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))
    kinetic_ldr = 0.5 * (kinetic_ldr + kinetic_ldr.getH())
    potential = sp.diags(apes.reshape(-1), format="csr", dtype=complex)
    return kinetic_ldr + potential, link_time + path_time, link_count, {
        "link_time": link_time,
        "path_time": path_time,
        "kinetic_edge_blocks": len(overlap_cache),
    }


def nearest_index_map(source_axis, target_axis):
    source_axis = np.asarray(source_axis, dtype=float)
    target_axis = np.asarray(target_axis, dtype=float)
    return np.asarray([int(np.argmin(np.abs(source_axis - x))) for x in target_axis], dtype=int)


def build_anchor_linked_overlap_ldr_hamiltonian(
    r1_dvr,
    r2_dvr,
    theta,
    apes,
    anchor_r1,
    anchor_r2,
    anchor_objects,
    args,
):
    proton_mass = 1.00782503223 * amu2au
    masses_au = np.array([proton_mass, proton_mass, proton_mass])
    kinetic = fixed_theta_stretch_kinetic(r1_dvr, r2_dvr, theta, masses_au).tocoo()
    ngrid = r1_dvr.npts * r2_dvr.npts
    nstates = args.nstates
    x_links, y_links, link_time, link_count = build_neighbor_links(anchor_objects, nstates)

    r1_map = nearest_index_map(anchor_r1, r1_dvr.x)
    r2_map = nearest_index_map(anchor_r2, r2_dvr.x)
    anchor_shape = anchor_objects.shape

    def anchor_flat(fe_flat):
        i, j = np.unravel_index(int(fe_flat), (r1_dvr.npts, r2_dvr.npts))
        return int(np.ravel_multi_index((r1_map[i], r2_map[j]), anchor_shape))

    rows, cols, data = [], [], []
    overlap_cache = {}
    t0 = time.perf_counter()
    for i, j, value in zip(kinetic.row, kinetic.col, kinetic.data):
        if abs(value) <= args.kinetic_threshold:
            continue
        ai = anchor_flat(i)
        aj = anchor_flat(j)
        key = (ai, aj)
        if key not in overlap_cache:
            overlap_cache[key] = linked_path_overlap(ai, aj, anchor_shape, x_links, y_links, nstates)
        block = value * overlap_cache[key]
        nz_a, nz_b = np.nonzero(np.abs(block) > 1.0e-14)
        rows.extend((i * nstates + nz_a).tolist())
        cols.extend((j * nstates + nz_b).tolist())
        data.extend(block[nz_a, nz_b].tolist())
    path_time = time.perf_counter() - t0

    dim = ngrid * nstates
    kinetic_ldr = sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))
    kinetic_ldr = 0.5 * (kinetic_ldr + kinetic_ldr.getH())
    potential = sp.diags(apes.reshape(-1), format="csr", dtype=complex)
    return kinetic_ldr + potential, link_time + path_time, link_count, {
        "link_time": link_time,
        "path_time": path_time,
        "kinetic_edge_blocks": len(overlap_cache),
        "anchor_points": anchor_objects.size,
    }


def nearest_grid_index(r1_dvr, r2_dvr, center):
    i = int(np.argmin(np.abs(np.asarray(r1_dvr.x) - center[0])))
    j = int(np.argmin(np.abs(np.asarray(r2_dvr.x) - center[1])))
    return int(np.ravel_multi_index((i, j), (r1_dvr.npts, r2_dvr.npts)))


def projected_state_packet(r1_dvr, r2_dvr, state, nstates, center, width, objects=None, anchor=None):
    scalar = initial_packet(r1_dvr, r2_dvr, center, width)
    coeff = np.zeros((scalar.size, nstates), dtype=complex)
    if objects is not None:
        flat_objects = objects.reshape(-1)
        ref_idx = nearest_grid_index(r1_dvr, r2_dvr, center)
        ref = flat_objects[ref_idx]
        for i, obj in enumerate(flat_objects):
            coeff[i] = electronic_overlap(obj, ref, nstates)[:, state]
    elif anchor is not None:
        anchor_r1 = np.asarray(anchor["r1"], dtype=float)
        anchor_r2 = np.asarray(anchor["r2"], dtype=float)
        anchor_shape = anchor["objects"].shape
        x_links, y_links, _, _ = build_neighbor_links(anchor["objects"], nstates)
        r1_map = nearest_index_map(anchor_r1, r1_dvr.x)
        r2_map = nearest_index_map(anchor_r2, r2_dvr.x)
        ref_a = int(np.argmin(np.abs(anchor_r1 - center[0])))
        ref_b = int(np.argmin(np.abs(anchor_r2 - center[1])))
        ref_idx = int(np.ravel_multi_index((ref_a, ref_b), anchor_shape))
        for flat in range(scalar.size):
            i, j = np.unravel_index(flat, (r1_dvr.npts, r2_dvr.npts))
            anchor_idx = int(np.ravel_multi_index((r1_map[i], r2_map[j]), anchor_shape))
            coeff[flat] = linked_path_overlap(anchor_idx, ref_idx, anchor_shape, x_links, y_links, nstates)[:, state]
    else:
        coeff[:, state] = 1.0

    packet = (scalar[:, None] * coeff).reshape(-1)
    norm = np.linalg.norm(packet)
    if norm == 0.0:
        raise ValueError("Projected packet has zero norm.")
    return packet / norm


def parse_snapshot_times(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def snapshot_indices(times_fs, snapshot_times_fs):
    indices = {}
    times_fs = np.asarray(times_fs, dtype=float)
    for target in snapshot_times_fs:
        idx = int(np.argmin(np.abs(times_fs - target)))
        indices[idx] = times_fs[idx]
    return indices


def propagate_populations(H, r1_dvr, r2_dvr, psi0, nstates, times_fs, snapshot_times_fs=()):
    t0 = time.perf_counter()
    evals, evecs = la.eigh(H.toarray())
    eig_time = time.perf_counter() - t0
    amplitudes = evecs.conj().T @ psi0
    lower, upper, bridge = region_masks(r1_dvr, r2_dvr)
    electronic = np.zeros((len(times_fs), nstates), dtype=float)
    region = np.zeros((len(times_fs), 4), dtype=float)
    snap_index = snapshot_indices(times_fs, snapshot_times_fs)
    snapshots = []
    t0 = time.perf_counter()
    for i, time_fs in enumerate(times_fs):
        coeff = evecs @ (np.exp(-1j * evals * time_fs / au2fs) * amplitudes)
        psi = coeff.reshape(-1, nstates)
        electronic[i] = np.sum(np.abs(psi) ** 2, axis=0)
        region[i] = region_population(np.linalg.norm(psi, axis=1), lower, upper, bridge)
        if i in snap_index:
            density = np.sum(np.abs(psi) ** 2, axis=1).reshape(r1_dvr.npts, r2_dvr.npts)
            snapshots.append((snap_index[i], density))
    prop_time = time.perf_counter() - t0
    return electronic, region, snapshots, eig_time, prop_time


def grid_edges(axis):
    axis = np.asarray(axis, dtype=float)
    if axis.size == 1:
        return np.array([axis[0] - 0.5, axis[0] + 0.5])
    mids = 0.5 * (axis[1:] + axis[:-1])
    edges = np.empty(axis.size + 1, dtype=float)
    edges[1:-1] = mids
    edges[0] = axis[0] - 0.5 * (axis[1] - axis[0])
    edges[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    return edges


def plot_density_snapshots(r1_dvr, r2_dvr, snapshots, title, outpath):
    if not snapshots:
        return
    ncols = len(snapshots)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(2.5 * ncols, 2.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if ncols == 1:
        axes = [axes]
    vmax = max(float(np.max(density)) for _, density in snapshots)
    r1_edges = grid_edges(r1_dvr.x)
    r2_edges = grid_edges(r2_dvr.x)
    for ax, (time_fs, density) in zip(axes, snapshots):
        mesh = ax.pcolormesh(
            r2_edges,
            r1_edges,
            density,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.plot([r2_dvr.x[0], r2_dvr.x[-1]], [r2_dvr.x[0], r2_dvr.x[-1]], color="w", lw=0.7, alpha=0.55)
        ax.set_title(f"{time_fs:.1f} fs", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("r2 / bohr")
    axes[0].set_ylabel("r1 / bohr")
    fig.suptitle(title, fontsize=10)
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.82)
    cbar.set_label("nuclear density")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_electronic_populations(times, series, outpath):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    for label, values, color, linestyle in series:
        ax.plot(times, values[:, 0], color=color, lw=2, ls=linestyle, label=f"{label} state 0")
        ax.plot(times, values[:, 1], color=color, lw=1.5, ls=":", label=f"{label} state 1")
    ax.set_xlabel("time / fs")
    ax.set_ylabel("electronic population")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_region_populations(times, series, outpath):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    for label, values, color, linestyle in series:
        ax.plot(times, values[:, 0], color=color, lw=2, ls=linestyle, label=f"{label} r1 < r2")
        ax.plot(times, values[:, 1], color=color, lw=1.5, ls=":", label=f"{label} r1 > r2")
    ax.set_xlabel("time / fs")
    ax.set_ylabel("region population")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def run_case(
    label,
    r1_dvr,
    r2_dvr,
    theta,
    apes,
    objects,
    args,
    times,
    center,
    overlap_mode="local",
    anchor=None,
):
    t0 = time.perf_counter()
    if overlap_mode == "local":
        H, overlap_time, nblocks = build_local_overlap_ldr_hamiltonian(
            r1_dvr,
            r2_dvr,
            theta,
            apes,
            objects,
            args,
        )
        overlap_extra = None
    elif overlap_mode == "linked":
        H, overlap_time, nblocks, overlap_extra = build_linked_overlap_ldr_hamiltonian(
            r1_dvr,
            r2_dvr,
            theta,
            apes,
            objects,
            args,
        )
    elif overlap_mode == "anchor-linked":
        if anchor is None:
            raise ValueError("anchor-linked mode requires anchor data.")
        H, overlap_time, nblocks, overlap_extra = build_anchor_linked_overlap_ldr_hamiltonian(
            r1_dvr,
            r2_dvr,
            theta,
            apes,
            anchor["r1"],
            anchor["r2"],
            anchor["objects"],
            args,
        )
    else:
        raise ValueError(f"Unknown overlap mode {overlap_mode!r}")
    build_time = time.perf_counter() - t0
    psi0 = projected_state_packet(
        r1_dvr,
        r2_dvr,
        args.initial_state,
        args.nstates,
        center,
        args.packet_width,
        objects=objects,
        anchor=anchor if overlap_mode == "anchor-linked" else None,
    )
    electronic, region, snapshots, eig_time, prop_time = propagate_populations(
        H,
        r1_dvr,
        r2_dvr,
        psi0,
        args.nstates,
        times,
        snapshot_times_fs=args.snapshot_times,
    )
    print(f"\n[{label}]")
    print(f"[size] points={r1_dvr.npts * r2_dvr.npts}, dim={H.shape[0]}, H nnz={H.nnz}")
    if overlap_mode == "local":
        print(f"[overlap] exact kinetic-edge overlap blocks={nblocks}, time={overlap_time:.6f} s")
    else:
        print(f"[overlap] nearest-neighbor link blocks={nblocks}, time={overlap_time:.6f} s")
        print(
            "[overlap] linked path blocks="
            f"{overlap_extra['kinetic_edge_blocks']}, "
            f"link={overlap_extra['link_time']:.6f} s, "
            f"path={overlap_extra['path_time']:.6f} s"
        )
        if "anchor_points" in overlap_extra:
            print(f"[overlap] anchor electronic points={overlap_extra['anchor_points']}")
    print("[initial electronic]", np.array2string(electronic[0], precision=8))
    print("[final electronic]", np.array2string(electronic[-1], precision=8))
    print("[initial region]", np.array2string(region[0], precision=8))
    print("[final region]", np.array2string(region[-1], precision=8))
    print(f"[timing] H build {build_time:.6f} s")
    print(f"[timing] eig {eig_time:.6f} s")
    print(f"[timing] propagation {prop_time:.6f} s")
    return {
        "H": H,
        "electronic": electronic,
        "region": region,
        "snapshots": snapshots,
        "build_time": build_time,
        "eig_time": eig_time,
        "prop_time": prop_time,
        "overlap_blocks": nblocks,
        "overlap_mode": overlap_mode,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-min", type=float, default=0.90)
    parser.add_argument("--r-max", type=float, default=3.20)
    parser.add_argument("--theta-deg", type=float, default=60.0)
    parser.add_argument("--nstates", type=int, default=2)
    parser.add_argument("--ncas", type=int, default=3)
    parser.add_argument("--scf-tol", type=float, default=1.0e-9)
    parser.add_argument("--max-cycle", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--sine-npts", type=int, default=8)
    parser.add_argument("--n-elements", type=int, default=4)
    parser.add_argument("--n-lobatto", type=int, default=4)
    parser.add_argument("--interp-level", type=int, default=5)
    parser.add_argument("--interp-type", choices=("CH", "CC"), default="CH")
    parser.add_argument("--initial-state", type=int, default=0)
    parser.add_argument("--packet-r1", type=float, default=1.36)
    parser.add_argument("--packet-r2", type=float, default=2.28)
    parser.add_argument("--packet-width", type=float, default=5.0)
    parser.add_argument("--nt", type=int, default=160)
    parser.add_argument("--dt-fs", type=float, default=0.05)
    parser.add_argument("--nout", type=int, default=2)
    parser.add_argument("--snapshot-fs", type=str, default="0,2,4,8")
    parser.add_argument("--kinetic-threshold", type=float, default=1.0e-13)
    parser.add_argument(
        "--linked-only",
        action="store_true",
        help="Use linked electronic overlaps for sine and FE comparisons and skip exact local propagations.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name("h3plus_sparse_apes_fedvr_local_overlap_vs_sine_ldr"),
    )
    args = parser.parse_args()

    if args.nstates < 2:
        raise ValueError("Use at least two states for electronic-population comparison.")
    args.outdir.mkdir(parents=True, exist_ok=True)
    theta = np.deg2rad(args.theta_deg)
    center = (args.packet_r1, args.packet_r2)
    times = np.arange(args.nt // args.nout + 1) * args.nout * args.dt_fs
    args.snapshot_times = parse_snapshot_times(args.snapshot_fs)

    sine_r1 = SineDVR(args.r_min, args.r_max, args.sine_npts)
    sine_r2 = SineDVR(args.r_min, args.r_max, args.sine_npts)
    sine_apes, sine_objects, sine_scan = scan_grid_objects(
        sine_r1,
        sine_r2,
        theta,
        args,
        "sine",
    )
    print(f"[sine scan] time={sine_scan:.6f} s")
    sine_mode = "linked" if args.linked_only else "local"
    sine_label = "sine DVR-LDR AM1 APES + linked overlaps" if args.linked_only else "sine DVR-LDR exact AM1 APES"
    sine = run_case(
        sine_label,
        sine_r1,
        sine_r2,
        theta,
        sine_apes,
        sine_objects,
        args,
        times,
        center,
        overlap_mode=sine_mode,
    )

    fe_r1 = FEDVR(args.r_min, args.r_max, args.n_elements, args.n_lobatto)
    fe_r2 = FEDVR(args.r_min, args.r_max, args.n_elements, args.n_lobatto)
    anchor = {
        "r1": sine_r1.x.copy(),
        "r2": sine_r2.x.copy(),
        "apes": sine_apes,
        "objects": sine_objects,
    }
    t0 = time.perf_counter()
    anchor_apes_on_fe = interpolate_tensor_apes_to_dvr(
        anchor["r1"],
        anchor["r2"],
        anchor["apes"],
        fe_r1,
        fe_r2,
    )
    anchor_interp_time = time.perf_counter() - t0
    print(
        "[anchor APES] interpolated sine-grid electronic-data APES "
        f"to FE nodes in {anchor_interp_time:.6f} s"
    )

    fe_anchor_linked = run_case(
        "FE-DVR LDR interpolated APES + anchor linked overlaps",
        fe_r1,
        fe_r2,
        theta,
        anchor_apes_on_fe,
        None,
        args,
        times,
        center,
        overlap_mode="anchor-linked",
        anchor=anchor,
    )

    fe_exact_apes, fe_objects, fe_scan = scan_grid_objects(
        fe_r1,
        fe_r2,
        theta,
        args,
        "FE overlap",
    )
    print(f"[FE overlap scan] time={fe_scan:.6f} s")
    apes_err = anchor_apes_on_fe - fe_exact_apes
    print(
        "[anchor APES] vs exact FE-node APES: "
        f"rmse={np.sqrt(np.mean(apes_err**2)):.6e}, "
        f"max={np.max(np.abs(apes_err)):.6e}"
    )

    fe_exact = None
    if not args.linked_only:
        fe_exact = run_case(
            "FE-DVR LDR exact AM1 APES + local overlaps",
            fe_r1,
            fe_r2,
            theta,
            fe_exact_apes,
            fe_objects,
            args,
            times,
            center,
        )
    fe_linked = run_case(
        "FE-DVR LDR exact AM1 APES + FE-grid linked overlaps",
        fe_r1,
        fe_r2,
        theta,
        fe_exact_apes,
        fe_objects,
        args,
        times,
        center,
        overlap_mode="linked",
    )

    pop_path = args.outdir / (
        f"h3plus_am1_anchor_linked_fe_e{args.n_elements}p{args.n_lobatto}_"
        f"sine{args.sine_npts}_electronic.png"
    )
    region_path = args.outdir / (
        f"h3plus_am1_anchor_linked_fe_e{args.n_elements}p{args.n_lobatto}_"
        f"sine{args.sine_npts}_regions.png"
    )
    electronic_series = [
        ("sine linked" if args.linked_only else "sine exact", sine["electronic"], "tab:blue", "-"),
        ("FE linked", fe_linked["electronic"], "tab:red", "--"),
        ("anchor APES+linked", fe_anchor_linked["electronic"], "tab:green", "-."),
    ]
    region_series = [
        ("sine linked" if args.linked_only else "sine exact", sine["region"], "tab:blue", "-"),
        ("FE linked", fe_linked["region"], "tab:red", "--"),
        ("anchor APES+linked", fe_anchor_linked["region"], "tab:green", "-."),
    ]
    if fe_exact is not None:
        electronic_series.insert(1, ("FE exact local", fe_exact["electronic"], "tab:orange", "-"))
        region_series.insert(1, ("FE exact local", fe_exact["region"], "tab:orange", "-"))
    plot_electronic_populations(times, electronic_series, pop_path)
    plot_region_populations(times, region_series, region_path)
    sine_snap_path = args.outdir / (
        f"h3plus_am1_sine{args.sine_npts}_wavepackets.png"
    )
    fe_exact_snap_path = args.outdir / (
        f"h3plus_am1_fe_e{args.n_elements}p{args.n_lobatto}_exact_wavepackets.png"
    )
    fe_linked_snap_path = args.outdir / (
        f"h3plus_am1_fe_e{args.n_elements}p{args.n_lobatto}_linked_wavepackets.png"
    )
    anchor_snap_path = args.outdir / (
        f"h3plus_am1_fe_e{args.n_elements}p{args.n_lobatto}_anchor_linked_wavepackets.png"
    )
    plot_density_snapshots(
        sine_r1,
        sine_r2,
        sine["snapshots"],
        "sine DVR-LDR exact",
        sine_snap_path,
    )
    if fe_exact is not None:
        plot_density_snapshots(
            fe_r1,
            fe_r2,
            fe_exact["snapshots"],
            "FE-DVR LDR exact local",
            fe_exact_snap_path,
        )
    plot_density_snapshots(
        fe_r1,
        fe_r2,
        fe_linked["snapshots"],
        "FE-DVR LDR linked",
        fe_linked_snap_path,
    )
    plot_density_snapshots(
        fe_r1,
        fe_r2,
        fe_anchor_linked["snapshots"],
        "FE-DVR anchor APES + linked",
        anchor_snap_path,
    )
    print(f"[plot] {pop_path}")
    print(f"[plot] {region_path}")
    print(f"[plot] {sine_snap_path}")
    if fe_exact is not None:
        print(f"[plot] {fe_exact_snap_path}")
    print(f"[plot] {fe_linked_snap_path}")
    print(f"[plot] {anchor_snap_path}")


if __name__ == "__main__":
    main()
