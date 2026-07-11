#!/usr/bin/env python3
"""Compare 2D sparse-grid and sparse FE-DVR overlap LDR against sine-DVR LDR.

The model is a two-state avoided crossing in two nuclear coordinates.  The
sine-DVR, FE-DVR, sparse FE-DVR, and sparse-grid calculations all use
adiabatic energies and full electronic overlap blocks, including a deliberately
discontinuous sign gauge.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.dvr import FEDVR, SineDVR
from pyqed.smolyak.sg import SparseGridLDR


def diabatic_potential(points):
    x = points[:, 0]
    y = points[:, 1]
    kx = 0.030
    ky = 0.018
    shift = 1.25
    v11 = kx * (x + shift) ** 2 + ky * y**2
    v22 = kx * (x - shift) ** 2 + ky * y**2 + 0.006 * x
    v12 = 0.035 * np.exp(-0.55 * (x**2 + 0.7 * y**2))

    out = np.zeros((len(points), 2, 2), dtype=float)
    out[:, 0, 0] = v11
    out[:, 1, 1] = v22
    out[:, 0, 1] = v12
    out[:, 1, 0] = v12
    return out


def adiabatic_data(points, gauge=False):
    energies, vectors = np.linalg.eigh(diabatic_potential(points))
    if gauge:
        signs = np.ones((len(points), 2), dtype=float)
        signs[:, 0] = np.where(np.sin(11.7 * points[:, 0] + 5.3 * points[:, 1]) >= 0, 1.0, -1.0)
        signs[:, 1] = np.where(np.cos(7.1 * points[:, 0] - 13.2 * points[:, 1]) >= 0, 1.0, -1.0)
        vectors = vectors * signs[:, None, :]
    return energies, vectors


def electronic_overlap(vectors):
    return np.einsum("ica,jcb->iajb", vectors, vectors)


def sine_grid(domain, npts):
    x_dvr = SineDVR(domain[0], domain[1], npts)
    y_dvr = SineDVR(domain[0], domain[1], npts)
    xx, yy = np.meshgrid(x_dvr.x, y_dvr.x, indexing="ij")
    points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    return x_dvr, y_dvr, points


def build_sine_ldr(domain, npts, mass):
    x_dvr, y_dvr, points = sine_grid(domain, npts)
    x_dvr.mass = mass
    y_dvr.mass = mass
    ngrid = npts * npts
    tx = sp.csr_matrix(x_dvr.t())
    ty = sp.csr_matrix(y_dvr.t())
    identity = sp.eye(npts, format="csr")
    kinetic = sp.kron(tx, identity, format="csr") + sp.kron(identity, ty, format="csr")
    energies, vectors = adiabatic_data(points, gauge=True)
    overlap = electronic_overlap(vectors)
    hamiltonian = build_sparse_overlap_matrix(kinetic, overlap)

    diagonal = np.empty(2 * ngrid, dtype=float)
    diagonal[0::2] = energies[:, 0]
    diagonal[1::2] = energies[:, 1]
    potential = sp.diags(diagonal, format="csr")
    return hamiltonian + potential, points, overlap


def as_pair(value):
    if np.isscalar(value):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError("Expected a scalar or a length-2 value.")
    return int(value[0]), int(value[1])


def fedvr_grid(domain, n_elements, n_lobatto, mass):
    n_elements_x, n_elements_y = as_pair(n_elements)
    x_dvr = FEDVR(domain[0], domain[1], n_elements_x, n_lobatto, mass=mass)
    y_dvr = FEDVR(domain[0], domain[1], n_elements_y, n_lobatto, mass=mass)
    xx, yy = np.meshgrid(x_dvr.x, y_dvr.x, indexing="ij")
    wx, wy = np.meshgrid(x_dvr.w, y_dvr.w, indexing="ij")
    points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    weights = (wx * wy).reshape(-1)
    return x_dvr, y_dvr, points, weights


def build_fedvr_ldr(domain, n_elements, n_lobatto, mass):
    x_dvr, y_dvr, points, weights = fedvr_grid(domain, n_elements, n_lobatto, mass)
    nx = x_dvr.npts
    ny = y_dvr.npts
    ngrid = nx * ny
    tx = x_dvr.kinetic_sparse()
    ty = y_dvr.kinetic_sparse()
    kinetic = sp.kron(tx, sp.eye(ny, format="csr"), format="csr") + sp.kron(
        sp.eye(nx, format="csr"),
        ty,
        format="csr",
    )
    energies, vectors = adiabatic_data(points, gauge=True)
    overlap = electronic_overlap(vectors)
    hamiltonian = build_sparse_overlap_matrix(kinetic, overlap)

    diagonal = np.empty(2 * ngrid, dtype=float)
    diagonal[0::2] = energies[:, 0]
    diagonal[1::2] = energies[:, 1]
    potential = sp.diags(diagonal, format="csr")
    return hamiltonian + potential, points, weights


def build_sparse_fedvr_ldr(
    domain,
    n_elements,
    n_lobatto,
    mass,
    center,
    y_max,
    packet_radius,
):
    x_dvr, y_dvr, points, weights = fedvr_grid(domain, n_elements, n_lobatto, mass)
    nx = x_dvr.npts
    ny = y_dvr.npts
    tx = x_dvr.kinetic_sparse()
    ty = y_dvr.kinetic_sparse()
    kinetic_full = sp.kron(tx, sp.eye(ny, format="csr"), format="csr") + sp.kron(
        sp.eye(nx, format="csr"),
        ty,
        format="csr",
    )

    center = np.asarray(center, dtype=float)
    active = (np.abs(points[:, 1]) <= y_max) | (
        np.linalg.norm(points - center, axis=1) <= packet_radius
    )
    if active.sum() == 0:
        raise ValueError("Sparse FE-DVR pruning removed every basis function.")

    active_points = points[active]
    active_weights = weights[active]
    kinetic = kinetic_full[active][:, active].tocsr()
    energies, vectors = adiabatic_data(active_points, gauge=True)
    overlap = electronic_overlap(vectors)
    hamiltonian = build_sparse_overlap_matrix(kinetic, overlap)

    diagonal = np.empty(2 * len(active_points), dtype=float)
    diagonal[0::2] = energies[:, 0]
    diagonal[1::2] = energies[:, 1]
    potential = sp.diags(diagonal, format="csr")
    return hamiltonian + potential, active_points, active_weights, active


def dyadic_node_levels(active_indices, max_level):
    levels = []
    for index in active_indices:
        index = int(index)
        twos = 0
        while index % 2 == 0:
            twos += 1
            index //= 2
        levels.append(max_level - twos)
    return np.asarray(levels, dtype=int)


def close_under_kinetic(active, kinetic, steps):
    active = np.asarray(active, dtype=bool).copy()
    coo = kinetic.tocoo()
    for _ in range(int(steps)):
        connected = active[coo.row] | active[coo.col]
        if not np.any(connected):
            break
        expanded = active.copy()
        expanded[coo.row[connected]] = True
        expanded[coo.col[connected]] = True
        if np.array_equal(expanded, active):
            break
        active = expanded
    return active


def build_direct_sparse_fedvr_ldr(domain, max_level, sparse_level, mass, closure_steps=0):
    n_elements = 2 ** (max_level - 1)
    x_dvr = FEDVR(domain[0], domain[1], n_elements, 3, mass=mass)
    y_dvr = FEDVR(domain[0], domain[1], n_elements, 3, mass=mass)
    nx = x_dvr.npts
    ny = y_dvr.npts

    level_x = dyadic_node_levels(x_dvr.active, max_level)
    level_y = dyadic_node_levels(y_dvr.active, max_level)
    active_matrix = level_x[:, None] + level_y[None, :] <= sparse_level
    active = active_matrix.reshape(-1)

    xx, yy = np.meshgrid(x_dvr.x, y_dvr.x, indexing="ij")
    wx, wy = np.meshgrid(x_dvr.w, y_dvr.w, indexing="ij")
    points_full = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    weights_full = (wx * wy).reshape(-1)

    tx = x_dvr.kinetic_sparse()
    ty = y_dvr.kinetic_sparse()
    kinetic_full = sp.kron(tx, sp.eye(ny, format="csr"), format="csr") + sp.kron(
        sp.eye(nx, format="csr"),
        ty,
        format="csr",
    )
    active = close_under_kinetic(active, kinetic_full, closure_steps)
    kinetic = kinetic_full[active][:, active].tocsr()
    points = points_full[active]
    weights = weights_full[active]

    energies, vectors = adiabatic_data(points, gauge=True)
    overlap = electronic_overlap(vectors)
    hamiltonian = build_sparse_overlap_matrix(kinetic, overlap)

    diagonal = np.empty(2 * len(points), dtype=float)
    diagonal[0::2] = energies[:, 0]
    diagonal[1::2] = energies[:, 1]
    potential = sp.diags(diagonal, format="csr")
    return hamiltonian + potential, points, weights, active, points_full


def fedvr_elements_from_level(level):
    return 2 ** int(level)


def smolyak_fedvr_terms(level):
    terms = []
    for total, coeff in ((level, 1.0), (level - 1, -1.0)):
        if total < 2:
            continue
        for level_x in range(1, total):
            level_y = total - level_x
            terms.append((coeff, level_x, level_y))
    return terms


def run_smolyak_fedvr_combination(domain, level, n_lobatto, mass, center, width, state, times):
    combined = None
    total_build = 0.0
    total_prop = 0.0
    term_summaries = []
    for coeff, level_x, level_y in smolyak_fedvr_terms(level):
        elements = (
            fedvr_elements_from_level(level_x),
            fedvr_elements_from_level(level_y),
        )
        t0 = time.perf_counter()
        H, points, weights = build_fedvr_ldr(domain, elements, n_lobatto, mass)
        build_time = time.perf_counter() - t0
        energies, vectors = adiabatic_data(points, gauge=True)
        overlap = electronic_overlap(vectors)
        psi0 = sine_reference_packet(points, overlap, center, width, state, weights=weights)

        t0 = time.perf_counter()
        states = propagate_sine_ldr(H, psi0, times)
        prop_time = time.perf_counter() - t0
        pops = sine_electronic_populations(states, points)

        if combined is None:
            combined = coeff * pops
        else:
            combined += coeff * pops
        total_build += build_time
        total_prop += prop_time
        term_summaries.append(
            {
                "coeff": coeff,
                "levels": (level_x, level_y),
                "elements": elements,
                "points": len(points),
                "dim": H.shape[0],
                "nnz": H.nnz,
                "build_time": build_time,
                "prop_time": prop_time,
            }
        )
    return combined, term_summaries, total_build, total_prop


def build_sparse_overlap_matrix(spatial, overlap):
    spatial = spatial.tocoo()
    ngrid, nstates = overlap.shape[:2]
    rows, cols, data = [], [], []
    for i, j, value in zip(spatial.row, spatial.col, spatial.data):
        block = value * overlap[i, :, j, :]
        nz_a, nz_b = np.nonzero(np.abs(block) > 1.0e-14)
        rows.extend((i * nstates + nz_a).tolist())
        cols.extend((j * nstates + nz_b).tolist())
        data.extend(block[nz_a, nz_b].tolist())
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(ngrid * nstates, ngrid * nstates))
    return 0.5 * (matrix + matrix.getH())


def build_sg_ldr(domain, level, mass):
    sg = SparseGridLDR(
        ndim=2,
        level=level,
        domain=(domain, domain),
        mass=np.array([mass, mass], dtype=float),
        index_rule="smolyak",
    )
    energies, vectors = adiabatic_data(sg.nodes, gauge=True)
    overlap = electronic_overlap(vectors)

    S = sg.build_overlap()
    T = sg.build_kinetic()
    B = build_sparse_overlap_matrix(S, overlap)
    kinetic = build_sparse_overlap_matrix(T, overlap)

    rows, cols, data = [], [], []
    coo = S.tocoo()
    for i, j, sij in zip(coo.row, coo.col, coo.data):
        block = 0.5 * (energies[i, :, None] + energies[j, None, :])
        block = sij * block * overlap[i, :, j, :]
        nz_a, nz_b = np.nonzero(np.abs(block) > 1.0e-14)
        rows.extend((i * 2 + nz_a).tolist())
        cols.extend((j * 2 + nz_b).tolist())
        data.extend(block[nz_a, nz_b].tolist())
    potential = sp.csr_matrix((data, (rows, cols)), shape=B.shape)
    H = kinetic + 0.5 * (potential + potential.getH())
    return sg, H, B, overlap


def sine_reference_packet(points, overlap, center, width, state, weights=None):
    dr = points - np.asarray(center, dtype=float)
    envelope = np.exp(-width * np.sum(dr * dr, axis=1))
    iref = int(np.argmin(np.sum((points - np.asarray(center)) ** 2, axis=1)))
    if weights is not None:
        envelope = np.sqrt(np.asarray(weights, dtype=float)) * envelope
    psi = envelope[:, None] * overlap[:, :, iref, state]
    psi = psi.reshape(-1).astype(complex)
    return psi / np.linalg.norm(psi)


def sg_reference_packet(sg, B, overlap, center, width, state):
    center = np.asarray(center, dtype=float)
    dr = sg.nodes - center
    envelope = np.exp(-width * np.sum(dr * dr, axis=1))
    iref = int(np.argmin(np.sum((sg.nodes - center) ** 2, axis=1)))
    values = envelope[:, None] * overlap[:, :, iref, state]
    coeff = sg.nodal_values_to_coefficients(values).reshape(-1).astype(complex)
    coeff /= np.sqrt(np.vdot(coeff, B @ coeff).real)
    return coeff


def sine_electronic_populations(states, points):
    pops = np.zeros((len(states), 3), dtype=float)
    for i, state in enumerate(states):
        psi = state.reshape(len(points), 2)
        epop = np.sum(np.abs(psi) ** 2, axis=0)
        pops[i] = (epop[0], epop[1], epop.sum())
    return pops


def sg_electronic_populations(coeffs, sg, B, order):
    points, weights = sg.quadrature_points(order=order, cellwise=True)
    phi = sg.interpolation_matrix(points)
    _, q_vectors = adiabatic_data(points, gauge=True)
    _, node_vectors = adiabatic_data(sg.nodes, gauge=True)
    local_overlap = np.einsum("qca,icb->qaib", q_vectors.conj(), node_vectors)
    pops = np.zeros((len(coeffs), 3), dtype=float)
    for i, coeff in enumerate(coeffs):
        total = np.vdot(coeff, B @ coeff).real
        coeff_matrix = coeff.reshape(sg.npts, 2)
        amplitudes = np.einsum("qi,ib,qaib->qa", phi, coeff_matrix, local_overlap)
        epop = np.sum(weights[:, None] * np.abs(amplitudes) ** 2, axis=0).real
        pops[i] = (epop[0], epop[1], total)
    return pops


def propagate_sine_ldr(H, psi0, times):
    return sla.expm_multiply((-1j) * H, psi0, start=times[0], stop=times[-1], num=len(times))


def propagate_sg(H, B, coeff0, times):
    evals, evecs = la.eigh(H.toarray(), B.toarray())
    amplitudes = evecs.conj().T @ (B @ coeff0)
    states = np.empty((len(times), len(coeff0)), dtype=complex)
    for i, time in enumerate(times):
        states[i] = evecs @ (np.exp(-1j * evals * time) * amplitudes)
    return evals, states


def plot_populations(
    times,
    sine_pops,
    sg_pops,
    outpath,
    fedvr_pops=None,
    sparse_fedvr_pops=None,
    smolyak_fedvr_pops=None,
    direct_sparse_fedvr_pops=None,
):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True, constrained_layout=True)
    axes[0].plot(times, sine_pops[:, 0], color="tab:blue", lw=2, label="sine DVR-LDR: state 0")
    axes[0].plot(times, sine_pops[:, 1], color="tab:blue", lw=1.8, ls="--", label="sine DVR-LDR: state 1")
    axes[0].plot(times, sg_pops[:, 0], color="tab:green", lw=2, label="SG-LDR: state 0")
    axes[0].plot(times, sg_pops[:, 1], color="tab:green", lw=1.8, ls="--", label="SG-LDR: state 1")
    if fedvr_pops is not None:
        axes[0].plot(times, fedvr_pops[:, 0], color="tab:orange", lw=2, label="FE-DVR LDR: state 0")
        axes[0].plot(times, fedvr_pops[:, 1], color="tab:orange", lw=1.8, ls="--", label="FE-DVR LDR: state 1")
    if sparse_fedvr_pops is not None:
        axes[0].plot(
            times,
            sparse_fedvr_pops[:, 0],
            color="tab:red",
            lw=2,
            label="sparse FE-DVR LDR: state 0",
        )
        axes[0].plot(
            times,
            sparse_fedvr_pops[:, 1],
            color="tab:red",
            lw=1.8,
            ls="--",
            label="sparse FE-DVR LDR: state 1",
        )
    if smolyak_fedvr_pops is not None:
        axes[0].plot(
            times,
            smolyak_fedvr_pops[:, 0],
            color="tab:cyan",
            lw=2,
            label="Smolyak FE-DVR LDR: state 0",
        )
        axes[0].plot(
            times,
            smolyak_fedvr_pops[:, 1],
            color="tab:cyan",
            lw=1.8,
            ls="--",
            label="Smolyak FE-DVR LDR: state 1",
        )
    if direct_sparse_fedvr_pops is not None:
        axes[0].plot(
            times,
            direct_sparse_fedvr_pops[:, 0],
            color="tab:purple",
            lw=2,
            label="direct sparse FE-DVR LDR: state 0",
        )
        axes[0].plot(
            times,
            direct_sparse_fedvr_pops[:, 1],
            color="tab:purple",
            lw=1.8,
            ls="--",
            label="direct sparse FE-DVR LDR: state 1",
        )
    axes[0].set_ylabel("electronic population")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    axes[1].plot(times, sine_pops[:, 2], color="tab:blue", lw=1.8, label="sine norm")
    axes[1].plot(times, sg_pops[:, 2], color="tab:green", lw=1.8, label="SG norm")
    if fedvr_pops is not None:
        axes[1].plot(times, fedvr_pops[:, 2], color="tab:orange", lw=1.8, label="FE-DVR norm")
    if sparse_fedvr_pops is not None:
        axes[1].plot(
            times,
            sparse_fedvr_pops[:, 2],
            color="tab:red",
            lw=1.8,
            label="sparse FE-DVR norm",
        )
    if smolyak_fedvr_pops is not None:
        axes[1].plot(
            times,
            smolyak_fedvr_pops[:, 2],
            color="tab:cyan",
            lw=1.8,
            label="Smolyak FE-DVR norm",
        )
    if direct_sparse_fedvr_pops is not None:
        axes[1].plot(
            times,
            direct_sparse_fedvr_pops[:, 2],
            color="tab:purple",
            lw=1.8,
            label="direct sparse FE-DVR norm",
        )
    axes[1].set_xlabel("time / a.u.")
    axes[1].set_ylabel("norm")
    axes[1].set_ylim(0.995, 1.005)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_grid(sg, sine_points, outpath):
    fig, ax = plt.subplots(figsize=(4.4, 4.2), constrained_layout=True)
    ax.scatter(sine_points[:, 0], sine_points[:, 1], s=5, color="0.82", label="sine DVR")
    ax.scatter(sg.nodes[:, 0], sg.nodes[:, 1], s=16, color="tab:green", label="Smolyak SG")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_fedvr_pruning(full_points, active, outpath):
    fig, ax = plt.subplots(figsize=(4.4, 4.2), constrained_layout=True)
    ax.scatter(full_points[:, 0], full_points[:, 1], s=10, color="0.84", label="pruned")
    ax.scatter(
        full_points[active, 0],
        full_points[active, 1],
        s=18,
        color="tab:red",
        label="active sparse FE-DVR",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_direct_sparse_fedvr(points_full, active, outpath):
    fig, ax = plt.subplots(figsize=(4.4, 4.2), constrained_layout=True)
    ax.scatter(points_full[:, 0], points_full[:, 1], s=5, color="0.86", label="finest FE-DVR")
    ax.scatter(
        points_full[active, 0],
        points_full[active, 1],
        s=14,
        color="tab:purple",
        label="direct sparse FE-DVR",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", type=float, nargs=2, default=(-4.0, 4.0))
    parser.add_argument("--sine-npts", type=int, default=28)
    parser.add_argument("--fedvr-elements", type=int, default=5)
    parser.add_argument("--fedvr-lobatto", type=int, default=5)
    parser.add_argument("--sparse-fedvr-ymax", type=float, default=2.2)
    parser.add_argument("--sparse-fedvr-packet-radius", type=float, default=0.9)
    parser.add_argument("--direct-sparse-fedvr-max-level", type=int, default=6)
    parser.add_argument("--direct-sparse-fedvr-level", type=int, default=7)
    parser.add_argument("--direct-sparse-fedvr-closure", type=int, default=0)
    parser.add_argument("--sg-level", type=int, default=6)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--center", type=float, nargs=2, default=(-1.9, 0.0))
    parser.add_argument("--width", type=float, default=1.7)
    parser.add_argument("--initial-state", type=int, default=0)
    parser.add_argument("--tmax", type=float, default=80.0)
    parser.add_argument("--nt", type=int, default=160)
    parser.add_argument("--quad-order", type=int, default=2)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name("model_sparse_grid_overlap_ldr_vs_sine"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    domain = tuple(args.domain)
    times = np.linspace(0.0, args.tmax, args.nt + 1)

    t0 = time.perf_counter()
    H_sine, sine_points, sine_overlap = build_sine_ldr(domain, args.sine_npts, args.mass)
    sine_build = time.perf_counter() - t0
    psi0 = sine_reference_packet(
        sine_points,
        sine_overlap,
        args.center,
        args.width,
        args.initial_state,
    )

    t0 = time.perf_counter()
    sine_states = propagate_sine_ldr(H_sine, psi0, times)
    sine_prop = time.perf_counter() - t0
    sine_pops = sine_electronic_populations(sine_states, sine_points)

    t0 = time.perf_counter()
    H_fedvr, fedvr_points, fedvr_weights = build_fedvr_ldr(
        domain,
        args.fedvr_elements,
        args.fedvr_lobatto,
        args.mass,
    )
    fedvr_build = time.perf_counter() - t0
    fedvr_energies, fedvr_vectors = adiabatic_data(fedvr_points, gauge=True)
    fedvr_overlap = electronic_overlap(fedvr_vectors)
    fedvr_psi0 = sine_reference_packet(
        fedvr_points,
        fedvr_overlap,
        args.center,
        args.width,
        args.initial_state,
        weights=fedvr_weights,
    )

    t0 = time.perf_counter()
    fedvr_states = propagate_sine_ldr(H_fedvr, fedvr_psi0, times)
    fedvr_prop = time.perf_counter() - t0
    fedvr_pops = sine_electronic_populations(fedvr_states, fedvr_points)

    t0 = time.perf_counter()
    H_sparse_fedvr, sparse_fedvr_points, sparse_fedvr_weights, sparse_fedvr_active = (
        build_sparse_fedvr_ldr(
            domain,
            args.fedvr_elements,
            args.fedvr_lobatto,
            args.mass,
            args.center,
            args.sparse_fedvr_ymax,
            args.sparse_fedvr_packet_radius,
        )
    )
    sparse_fedvr_build = time.perf_counter() - t0
    sparse_fedvr_energies, sparse_fedvr_vectors = adiabatic_data(sparse_fedvr_points, gauge=True)
    sparse_fedvr_overlap = electronic_overlap(sparse_fedvr_vectors)
    sparse_fedvr_psi0 = sine_reference_packet(
        sparse_fedvr_points,
        sparse_fedvr_overlap,
        args.center,
        args.width,
        args.initial_state,
        weights=sparse_fedvr_weights,
    )

    t0 = time.perf_counter()
    sparse_fedvr_states = propagate_sine_ldr(H_sparse_fedvr, sparse_fedvr_psi0, times)
    sparse_fedvr_prop = time.perf_counter() - t0
    sparse_fedvr_pops = sine_electronic_populations(sparse_fedvr_states, sparse_fedvr_points)

    t0 = time.perf_counter()
    (
        H_direct_sparse_fedvr,
        direct_sparse_fedvr_points,
        direct_sparse_fedvr_weights,
        direct_sparse_fedvr_active,
        direct_sparse_fedvr_full_points,
    ) = build_direct_sparse_fedvr_ldr(
        domain,
        args.direct_sparse_fedvr_max_level,
        args.direct_sparse_fedvr_level,
        args.mass,
        closure_steps=args.direct_sparse_fedvr_closure,
    )
    direct_sparse_fedvr_build = time.perf_counter() - t0
    direct_sparse_fedvr_energies, direct_sparse_fedvr_vectors = adiabatic_data(
        direct_sparse_fedvr_points,
        gauge=True,
    )
    direct_sparse_fedvr_overlap = electronic_overlap(direct_sparse_fedvr_vectors)
    direct_sparse_fedvr_psi0 = sine_reference_packet(
        direct_sparse_fedvr_points,
        direct_sparse_fedvr_overlap,
        args.center,
        args.width,
        args.initial_state,
        weights=direct_sparse_fedvr_weights,
    )

    t0 = time.perf_counter()
    direct_sparse_fedvr_states = propagate_sine_ldr(
        H_direct_sparse_fedvr,
        direct_sparse_fedvr_psi0,
        times,
    )
    direct_sparse_fedvr_prop = time.perf_counter() - t0
    direct_sparse_fedvr_pops = sine_electronic_populations(
        direct_sparse_fedvr_states,
        direct_sparse_fedvr_points,
    )

    t0 = time.perf_counter()
    sg, H_sg, B_sg, overlap = build_sg_ldr(domain, args.sg_level, args.mass)
    sg_build = time.perf_counter() - t0
    coeff0 = sg_reference_packet(sg, B_sg, overlap, args.center, args.width, args.initial_state)

    t0 = time.perf_counter()
    evals_sg, sg_states = propagate_sg(H_sg, B_sg, coeff0, times)
    sg_prop = time.perf_counter() - t0
    sg_pops = sg_electronic_populations(sg_states, sg, B_sg, args.quad_order)

    pop_path = args.outdir / (
        f"model_sine{args.sine_npts}_sg_l{args.sg_level}_"
        f"fedvr_e{args.fedvr_elements}p{args.fedvr_lobatto}_"
        f"sfey{args.sparse_fedvr_ymax:g}_"
        f"dsfe_l{args.direct_sparse_fedvr_level}_"
        f"c{args.direct_sparse_fedvr_closure}_"
        f"t{args.tmax:g}_overlap_electronic_populations.png"
    )
    grid_path = args.outdir / f"model_sine{args.sine_npts}_sg_l{args.sg_level}_points.png"
    sparse_fedvr_grid_path = args.outdir / (
        f"model_sparse_fedvr_e{args.fedvr_elements}p{args.fedvr_lobatto}_"
        f"y{args.sparse_fedvr_ymax:g}_points.png"
    )
    direct_sparse_fedvr_grid_path = args.outdir / (
        f"model_direct_sparse_fedvr_l{args.direct_sparse_fedvr_level}_"
        f"max{args.direct_sparse_fedvr_max_level}_"
        f"c{args.direct_sparse_fedvr_closure}_points.png"
    )
    plot_populations(
        times,
        sine_pops,
        sg_pops,
        pop_path,
        fedvr_pops=fedvr_pops,
        sparse_fedvr_pops=sparse_fedvr_pops,
        direct_sparse_fedvr_pops=direct_sparse_fedvr_pops,
    )
    plot_grid(sg, sine_points, grid_path)
    plot_fedvr_pruning(fedvr_points, sparse_fedvr_active, sparse_fedvr_grid_path)
    plot_direct_sparse_fedvr(
        direct_sparse_fedvr_full_points,
        direct_sparse_fedvr_active,
        direct_sparse_fedvr_grid_path,
    )

    print(f"[size] sine DVR points={len(sine_points)}, dim={H_sine.shape[0]}, H nnz={H_sine.nnz}")
    print(f"[size] FE-DVR points={len(fedvr_points)}, dim={H_fedvr.shape[0]}, H nnz={H_fedvr.nnz}")
    print(
        f"[size] sparse FE-DVR points={len(sparse_fedvr_points)}, "
        f"dim={H_sparse_fedvr.shape[0]}, H nnz={H_sparse_fedvr.nnz}"
    )
    print(
        f"[size] direct sparse FE-DVR points={len(direct_sparse_fedvr_points)}, "
        f"dim={H_direct_sparse_fedvr.shape[0]}, H nnz={H_direct_sparse_fedvr.nnz}"
    )
    print(f"[size] SG-LDR points={sg.npts}, dim={H_sg.shape[0]}, H nnz={H_sg.nnz}, S nnz={B_sg.nnz}")
    print("[initial] sine", np.array2string(sine_pops[0], precision=8))
    print("[initial] FE-DVR", np.array2string(fedvr_pops[0], precision=8))
    print("[initial] sparse FE-DVR", np.array2string(sparse_fedvr_pops[0], precision=8))
    print("[initial] direct sparse FE-DVR", np.array2string(direct_sparse_fedvr_pops[0], precision=8))
    print("[initial] SG-LDR", np.array2string(sg_pops[0], precision=8))
    print("[final] sine", np.array2string(sine_pops[-1], precision=8))
    print("[final] FE-DVR", np.array2string(fedvr_pops[-1], precision=8))
    print("[final] sparse FE-DVR", np.array2string(sparse_fedvr_pops[-1], precision=8))
    print("[final] direct sparse FE-DVR", np.array2string(direct_sparse_fedvr_pops[-1], precision=8))
    print("[final] SG-LDR", np.array2string(sg_pops[-1], precision=8))
    print(f"[timing] sine build {sine_build:.6f} s")
    print(f"[timing] sine propagate {sine_prop:.6f} s")
    print(f"[timing] FE-DVR build {fedvr_build:.6f} s")
    print(f"[timing] FE-DVR propagate {fedvr_prop:.6f} s")
    print(f"[timing] sparse FE-DVR build {sparse_fedvr_build:.6f} s")
    print(f"[timing] sparse FE-DVR propagate {sparse_fedvr_prop:.6f} s")
    print(f"[timing] direct sparse FE-DVR build {direct_sparse_fedvr_build:.6f} s")
    print(f"[timing] direct sparse FE-DVR propagate {direct_sparse_fedvr_prop:.6f} s")
    print(f"[timing] SG build {sg_build:.6f} s")
    print(f"[timing] SG generalized propagation {sg_prop:.6f} s")
    print(f"[levels] SG first 8 = {np.array2string(evals_sg[:8], precision=8)}")
    print(f"[plot] {pop_path}")
    print(f"[plot] {grid_path}")
    print(f"[plot] {sparse_fedvr_grid_path}")
    print(f"[plot] {direct_sparse_fedvr_grid_path}")


if __name__ == "__main__":
    main()
