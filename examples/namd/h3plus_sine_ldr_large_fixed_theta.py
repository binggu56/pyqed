#!/usr/bin/env python3
"""Larger fixed-theta H3+ sine-DVR LDR benchmark.

This script is intentionally narrow: it builds a larger 2D sine-DVR reference
for the H3+ fixed-theta test while avoiding the dense all-pairs LDR overlap
construction.  The kinetic operator is dense in the sine DVR, but the
electronic overlap blocks are approximated by nearest-neighbor linked products.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.dvr import FEDVR, SineDVR
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casci import CASCI
from pyqed.units import amu2au, au2fs

from h3plus_fedvr_fixed_theta import (
    fixed_theta_stretch_kinetic,
    h3plus_body_frame,
    initial_packet,
    _grid_weights,
)


def clean_float(value):
    return f"{float(value):.3f}".replace("-", "m").replace(".", "p")


def run_electronic_point(task):
    (
        index,
        r1,
        r2,
        theta,
        backend,
        basis,
        nstates,
        ncas,
        nelecas,
        scf_tol,
        max_cycle,
        damping,
    ) = task
    mol = Molecule(
        atom=h3plus_body_frame(float(r1), float(r2), float(theta)),
        basis=basis,
        charge=1,
        spin=0,
        unit="bohr",
    )

    if backend == "am1-meci":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pyqed.qchem.semiempirical.am1 import RAM1

        mf = RAM1(mol).run(
            conv_tol=scf_tol,
            max_cycle=max_cycle,
            verbose=0,
            damping=damping,
        )
        ci = mf.MECI(nstates=nstates, ncas=ncas).run()
    elif backend == "rhf-casci":
        mol.build()
        mf = mol.RHF(verbose=0).run(max_cycle=max_cycle)
        ci = CASCI(mf, ncas=ncas, nelecas=nelecas, verbose=0).run(nstates=nstates)
    else:
        raise ValueError(f"Unknown backend {backend!r}.")

    energies = getattr(ci, "e_tot", getattr(ci, "e", None))
    if energies is None:
        raise AttributeError("Electronic object has neither e_tot nor e.")
    return index, np.asarray(energies, dtype=float)[:nstates], ci


def scan_or_load(r1_dvr, r2_dvr, theta, args, cache, backend):
    if cache.exists() and not args.force_scan:
        data = np.load(cache, allow_pickle=True)
        if (
            np.allclose(data["r1"], r1_dvr.x)
            and np.allclose(data["r2"], r2_dvr.x)
            and np.isclose(float(data["theta"]), theta)
            and data["apes"].shape[-1] >= args.nstates
            and str(data.get("backend", backend)) == backend
        ):
            print(f"[scan] loaded {cache}")
            return data["apes"][..., : args.nstates], data["objects"]
        print(f"[scan] ignoring incompatible cache {cache}")

    tasks = []
    for i, r1 in enumerate(r1_dvr.x):
        for j, r2 in enumerate(r2_dvr.x):
            tasks.append(
                (
                    (i, j),
                    float(r1),
                    float(r2),
                    float(theta),
                    backend,
                    args.basis,
                    args.nstates,
                    args.ncas,
                    args.nelecas,
                    args.scf_tol,
                    args.max_cycle,
                    args.damping,
                )
            )

    apes = np.zeros((r1_dvr.npts, r2_dvr.npts, args.nstates), dtype=float)
    objects = np.empty((r1_dvr.npts, r2_dvr.npts), dtype=object)
    t0 = time.perf_counter()
    done = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_electronic_point, task) for task in tasks]
            for future in as_completed(futures):
                (i, j), energies, obj = future.result()
                apes[i, j] = energies
                objects[i, j] = obj
                done += 1
                if done % args.progress_every == 0 or done == len(tasks):
                    print(f"[scan] {done}/{len(tasks)}")
    else:
        for task in tasks:
            (i, j), energies, obj = run_electronic_point(task)
            apes[i, j] = energies
            objects[i, j] = obj
            done += 1
            if done % args.progress_every == 0 or done == len(tasks):
                print(f"[scan] {done}/{len(tasks)}")

    elapsed = time.perf_counter() - t0
    np.savez_compressed(
        cache,
        apes=apes,
        objects=objects,
        r1=r1_dvr.x,
        r2=r2_dvr.x,
        theta=theta,
        backend=backend,
        basis=args.basis,
        ncas=args.ncas,
        nelecas=args.nelecas,
    )
    print(f"[scan] saved {cache}")
    print(f"[scan] time={elapsed:.3f} s")
    return apes, objects


def electronic_overlap(left, right, nstates):
    if left is right:
        return np.eye(nstates, dtype=complex)
    if hasattr(left, "wavefunction_overlap"):
        block = np.asarray(left.wavefunction_overlap(right), dtype=complex)
    elif hasattr(left, "overlap"):
        block = np.asarray(left.overlap(right), dtype=complex)
    else:
        raise TypeError("Electronic object does not expose an overlap method.")
    return block[:nstates, :nstates]


def build_neighbor_links(objects, nstates):
    nx, ny = objects.shape
    x_links = np.empty((nx - 1, ny), dtype=object)
    y_links = np.empty((nx, ny - 1), dtype=object)
    count = 0
    t0 = time.perf_counter()
    for i in range(nx - 1):
        for j in range(ny):
            x_links[i, j] = electronic_overlap(objects[i, j], objects[i + 1, j], nstates)
            count += 1
    for i in range(nx):
        for j in range(ny - 1):
            y_links[i, j] = electronic_overlap(objects[i, j], objects[i, j + 1], nstates)
            count += 1
    return x_links, y_links, count, time.perf_counter() - t0


def linked_overlap(left, right, shape, x_links, y_links, nstates):
    if left == right:
        return np.eye(nstates, dtype=complex)

    start = np.array(np.unravel_index(left, shape), dtype=int)
    target = np.array(np.unravel_index(right, shape), dtype=int)

    def step(current, axis, direction):
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
                block = block @ step(current, axis, direction)
                current[axis] += direction
        return block

    return 0.5 * (path_product((0, 1)) + path_product((1, 0)))


def cap_profile_1d(dvr, width_min, width_max, strength, order):
    x = np.asarray(dvr.x, dtype=float)
    cap = np.zeros_like(x)
    if strength <= 0.0:
        return cap
    if width_min > 0.0:
        start = float(dvr.xmin) + float(width_min)
        mask = x < start
        cap[mask] += strength * ((start - x[mask]) / float(width_min)) ** order
    if width_max > 0.0:
        start = float(dvr.xmax) - float(width_max)
        mask = x > start
        cap[mask] += strength * ((x[mask] - start) / float(width_max)) ** order
    return cap


def cap_profile_2d(r1_dvr, r2_dvr, args):
    cap1 = cap_profile_1d(
        r1_dvr,
        args.cap_width_min,
        args.cap_width_max,
        args.cap_strength,
        args.cap_order,
    )
    cap2 = cap_profile_1d(
        r2_dvr,
        args.cap_width_min,
        args.cap_width_max,
        args.cap_strength,
        args.cap_order,
    )
    return cap1[:, None] + cap2[None, :]


def make_dvr(args):
    if args.grid == "sine":
        return (
            SineDVR(args.r_min, args.r_max, args.npts),
            SineDVR(args.r_min, args.r_max, args.npts),
        )
    if args.grid == "fedvr":
        return (
            FEDVR(args.r_min, args.r_max, args.n_elements, args.n_lobatto),
            FEDVR(args.r_min, args.r_max, args.n_elements, args.n_lobatto),
        )
    raise ValueError(f"unknown grid {args.grid!r}")


def grid_tag(args, r1_dvr):
    if args.grid == "sine":
        return f"sine{r1_dvr.npts}"
    return f"fedvr_e{args.n_elements}_p{args.n_lobatto}_n{r1_dvr.npts}"


def build_linked_hamiltonian(r1_dvr, r2_dvr, theta, apes, objects, args):
    proton_mass = 1.00782503223 * amu2au
    masses_au = np.array([proton_mass, proton_mass, proton_mass])
    kinetic = fixed_theta_stretch_kinetic(r1_dvr, r2_dvr, theta, masses_au).tocoo()
    ngrid = r1_dvr.npts * r2_dvr.npts
    dim = ngrid * args.nstates
    x_links, y_links, link_count, link_time = build_neighbor_links(objects, args.nstates)

    rows = []
    cols = []
    data = []
    t0 = time.perf_counter()
    kept_edges = 0
    shape = objects.shape
    for i, j, value in zip(kinetic.row, kinetic.col, kinetic.data):
        if abs(value) <= args.kinetic_threshold:
            continue
        kept_edges += 1
        block = value * linked_overlap(int(i), int(j), shape, x_links, y_links, args.nstates)
        nz_a, nz_b = np.nonzero(np.abs(block) > args.block_threshold)
        rows.extend((int(i) * args.nstates + nz_a).tolist())
        cols.extend((int(j) * args.nstates + nz_b).tolist())
        data.extend(block[nz_a, nz_b].tolist())
    path_time = time.perf_counter() - t0

    kinetic_ldr = sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))
    kinetic_ldr = 0.5 * (kinetic_ldr + kinetic_ldr.getH())
    cap = cap_profile_2d(r1_dvr, r2_dvr, args)
    potential_diag = apes.reshape(-1).astype(complex)
    if args.cap_strength > 0.0:
        potential_diag = potential_diag - 1j * np.repeat(cap.reshape(-1), args.nstates)
    potential = sp.diags(potential_diag, format="csr", dtype=complex)
    return kinetic_ldr + potential, {
        "kinetic_nnz": int(kinetic.nnz),
        "kept_edges": int(kept_edges),
        "link_count": int(link_count),
        "link_time": float(link_time),
        "path_time": float(path_time),
        "cap_max": float(cap.max()),
    }


def initial_electronic_projection(objects, r1_dvr, r2_dvr, args):
    if args.initial_electronic_projection == "local":
        projector = np.zeros((r1_dvr.npts, r2_dvr.npts, args.nstates), dtype=complex)
        projector[:, :, args.initial_state] = 1.0
        return projector

    if args.initial_electronic_projection != "reference-overlap":
        raise ValueError(
            "initial_electronic_projection must be 'local' or 'reference-overlap'."
        )

    default_ref_r1 = getattr(args, "_fc_reference_r1", args.packet_r1)
    default_ref_r2 = getattr(args, "_fc_reference_r2", args.packet_r2)
    ref_r1 = args.reference_r1 if args.reference_r1 is not None else default_ref_r1
    ref_r2 = args.reference_r2 if args.reference_r2 is not None else default_ref_r2
    iref = int(np.argmin(np.abs(r1_dvr.x - ref_r1)))
    jref = int(np.argmin(np.abs(r2_dvr.x - ref_r2)))
    ref_obj = objects[iref, jref]

    projector = np.zeros((r1_dvr.npts, r2_dvr.npts, args.nstates), dtype=complex)
    for i in range(r1_dvr.npts):
        for j in range(r2_dvr.npts):
            block = electronic_overlap(objects[i, j], ref_obj, args.nstates)
            projector[i, j] = block[:, args.initial_state]

    print(
        "[initial electronic projection] "
        f"reference-overlap from grid ({iref}, {jref}) "
        f"r1={r1_dvr.x[iref]:.6f}, r2={r2_dvr.x[jref]:.6f}, "
        f"state=S{args.initial_state}"
    )
    return projector


def fc_ground_packet(r1_dvr, r2_dvr, theta, apes, args):
    proton_mass = 1.00782503223 * amu2au
    masses_au = np.array([proton_mass, proton_mass, proton_mass])
    kinetic = fixed_theta_stretch_kinetic(r1_dvr, r2_dvr, theta, masses_au)
    potential = sp.diags(apes[:, :, args.fc_surface].reshape(-1), format="csr")
    h0 = kinetic + potential
    if h0.shape[0] <= 512:
        evals, evecs = np.linalg.eigh(h0.toarray())
        energy = float(evals[0].real)
        scalar = np.asarray(evecs[:, 0], dtype=complex)
    else:
        evals, evecs = eigsh(h0, k=1, which="SA")
        energy = float(evals[0].real)
        scalar = np.asarray(evecs[:, 0], dtype=complex)

    peak = int(np.argmax(np.abs(scalar) ** 2))
    ipeak, jpeak = np.unravel_index(peak, (r1_dvr.npts, r2_dvr.npts))
    args._fc_reference_r1 = float(r1_dvr.x[ipeak])
    args._fc_reference_r2 = float(r2_dvr.x[jpeak])
    if scalar[peak].real < 0:
        scalar *= -1
    scalar = scalar / np.linalg.norm(scalar)
    print(
        "[initial nuclear packet] FC-like ground state on "
        f"S{args.fc_surface}: E={energy:.10f} Eh, "
        f"peak grid ({ipeak}, {jpeak}) "
        f"r1={r1_dvr.x[ipeak]:.6f}, r2={r2_dvr.x[jpeak]:.6f}"
    )
    return scalar


def initial_nuclear_packet(r1_dvr, r2_dvr, theta, apes, args):
    if args.initial_nuclear == "gaussian":
        return initial_packet(
            r1_dvr,
            r2_dvr,
            (args.packet_r1, args.packet_r2),
            args.packet_width,
        )
    if args.initial_nuclear == "fc-ground":
        return fc_ground_packet(r1_dvr, r2_dvr, theta, apes, args)
    raise ValueError(f"unknown initial_nuclear {args.initial_nuclear!r}")


def initial_state(r1_dvr, r2_dvr, theta, apes, args, objects=None):
    scalar = initial_nuclear_packet(r1_dvr, r2_dvr, theta, apes, args)
    if objects is None:
        projector = np.zeros((r1_dvr.npts, r2_dvr.npts, args.nstates), dtype=complex)
        projector[:, :, args.initial_state] = 1.0
    else:
        projector = initial_electronic_projection(objects, r1_dvr, r2_dvr, args)
    psi = scalar.reshape(r1_dvr.npts, r2_dvr.npts)[:, :, None] * projector
    psi = psi.reshape(-1)
    return psi / np.linalg.norm(psi)


def propagate(H, psi0, times_fs, nstates):
    t0 = time.perf_counter()
    states = expm_multiply(
        -1j * H,
        psi0,
        start=float(times_fs[0] / au2fs),
        stop=float(times_fs[-1] / au2fs),
        num=len(times_fs),
        traceA=-1j * H.diagonal().sum(),
    )
    elapsed = time.perf_counter() - t0
    psi = states.reshape(len(times_fs), -1, nstates)
    populations = np.sum(np.abs(psi) ** 2, axis=1)
    norms = np.sum(populations, axis=1)
    return states, populations, norms, elapsed


def cell_edges(dvr):
    centers = np.asarray(dvr.x, dtype=float)
    edges = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = float(dvr.xmin)
    edges[-1] = float(dvr.xmax)
    return edges


def plot_populations(series, outpath):
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    state_styles = {0: ":", 1: "-", 2: "--"}
    state_colors = {0: "tab:gray", 1: "tab:blue", 2: "tab:orange"}
    backend_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    use_backend_colors = len(series) > 1
    for series_index, (label, times, pops, alpha) in enumerate(series):
        for state in range(pops.shape[1]):
            color = (
                backend_colors[series_index % len(backend_colors)]
                if use_backend_colors
                else state_colors.get(state, "k")
            )
            linestyle = state_styles.get(state, "-.")
            ax.plot(
                times,
                pops[:, state],
                color=color,
                ls=linestyle,
                lw=2.0 if alpha >= 0.95 else 1.4,
                alpha=alpha,
                label=f"{label} S{state}",
            )
    ax.set_xlabel("time / fs")
    ax.set_ylabel("population")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_wavepackets(times_fs, states, populations, r1_dvr, r2_dvr, nstates, snapshot_times, outpath):
    selected = [int(np.argmin(np.abs(times_fs - t))) for t in snapshot_times if t <= times_fs[-1]]
    selected = list(dict.fromkeys(selected))
    psi = states.reshape(len(times_fs), r1_dvr.npts, r2_dvr.npts, nstates)
    r1_edges = cell_edges(r1_dvr)
    r2_edges = cell_edges(r2_dvr)
    r2_mesh, r1_mesh = np.meshgrid(r2_edges, r1_edges, indexing="xy")
    w1 = _grid_weights(r1_dvr)[:, None]
    w2 = _grid_weights(r2_dvr)[None, :]
    cell_area = w1 * w2
    labels = [f"S{state}" for state in range(nstates)]
    cmap = "magma"
    global_vmax = float(np.max(np.abs(psi) ** 2 / cell_area[None, :, :, None]))
    if global_vmax <= 0.0:
        global_vmax = 1.0

    fig, axes = plt.subplots(
        nstates,
        len(selected),
        figsize=(2.75 * len(selected), 2.45 * nstates),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    last = None
    for state in range(nstates):
        for col, idx in enumerate(selected):
            ax = axes[state, col]
            density = np.abs(psi[idx, :, :, state]) ** 2 / cell_area
            last = ax.pcolormesh(
                r2_mesh,
                r1_mesh,
                density,
                shading="flat",
                cmap=cmap,
                vmin=0.0,
                vmax=global_vmax,
            )
            ax.scatter(
                np.tile(r2_dvr.x, r1_dvr.npts),
                np.repeat(r1_dvr.x, r2_dvr.npts),
                s=2.0,
                color="white",
                alpha=0.20,
                linewidths=0,
            )
            if state == 0:
                ax.set_title(f"{times_fs[idx]:g} fs")
            if col == 0:
                ax.set_ylabel(f"{labels[state]}\nr1 / bohr")
            if state == nstates - 1:
                ax.set_xlabel("r2 / bohr")
            ax.text(
                0.04,
                0.93,
                f"P={populations[idx, state]:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 1.8},
            )
            ax.set_aspect("equal", adjustable="box")
    if last is not None:
        fig.colorbar(last, ax=axes, shrink=0.84, label="DVR cell density")
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def run_backend(backend, args):
    theta = np.deg2rad(args.theta_deg)
    r1_dvr, r2_dvr = make_dvr(args)
    basis_tag = grid_tag(args, r1_dvr)
    domain_tag = (
        f"rmin{clean_float(args.r_min)}_rmax{clean_float(args.r_max)}"
    )
    cache = args.outdir / (
        f"scan_{basis_tag}x{basis_tag}_{domain_tag}_theta{args.theta_deg:g}_"
        f"{backend}_basis{args.basis}_cas{args.ncas}e{args.nelecas}.npz"
    )

    apes, objects = scan_or_load(r1_dvr, r2_dvr, theta, args, cache, backend)
    t0 = time.perf_counter()
    H, meta = build_linked_hamiltonian(r1_dvr, r2_dvr, theta, apes, objects, args)
    build_time = time.perf_counter() - t0
    times_fs = np.linspace(0.0, args.tmax_fs, args.nt)
    psi0 = initial_state(r1_dvr, r2_dvr, theta, apes, args, objects=objects)
    states, populations, norms, prop_time = propagate(H, psi0, times_fs, args.nstates)

    projection_tag = args.initial_electronic_projection.replace("-", "_")
    packet_tag = (
        f"r{clean_float(args.packet_r1)}_{clean_float(args.packet_r2)}_"
        f"w{clean_float(args.packet_width)}_t{clean_float(args.tmax_fs)}"
    )
    nuclear_tag = args.initial_nuclear.replace("-", "_")
    if args.initial_nuclear == "fc-ground":
        nuclear_tag = f"{nuclear_tag}_s{args.fc_surface}"
    cap_tag = "capoff"
    if args.cap_strength > 0.0:
        cap_tag = (
            f"cap{clean_float(args.cap_strength)}_"
            f"lo{clean_float(args.cap_width_min)}_hi{clean_float(args.cap_width_max)}_"
            f"o{int(args.cap_order)}"
        )
    tag = (
        f"{backend}_{basis_tag}_{domain_tag}_linked_ldr_"
        f"{projection_tag}_{nuclear_tag}_{packet_tag}_{cap_tag}"
    )
    out = args.outdir / f"h3plus_{tag}_populations.npz"
    np.savez_compressed(
        out,
        times_fs=times_fs,
        populations=populations,
        norms=norms,
        states=states.reshape(len(times_fs), r1_dvr.npts, r2_dvr.npts, args.nstates),
        r1=r1_dvr.x,
        r2=r2_dvr.x,
        apes=apes,
        backend=backend,
        basis=args.basis,
        ncas=args.ncas,
        nelecas=args.nelecas,
        initial_electronic_projection=args.initial_electronic_projection,
        initial_nuclear=args.initial_nuclear,
        fc_surface=args.fc_surface,
        initial_state=args.initial_state,
        theta=theta,
        cap_strength=args.cap_strength,
        cap_width_min=args.cap_width_min,
        cap_width_max=args.cap_width_max,
        cap_order=args.cap_order,
        cap_profile=cap_profile_2d(r1_dvr, r2_dvr, args),
        h_nnz=H.nnz,
        build_time=build_time,
        propagation_time=prop_time,
        **meta,
    )

    png = args.outdir / f"h3plus_{tag}_populations.png"
    plot_populations([(backend, times_fs, populations, 1.0)], png)
    wavepacket_png = args.outdir / f"h3plus_{tag}_wavepackets.png"
    plot_wavepackets(
        times_fs,
        states,
        populations,
        r1_dvr,
        r2_dvr,
        args.nstates,
        (0.0, 1.0, 3.0, 10.0, 30.0),
        wavepacket_png,
    )

    print(f"[{backend}]")
    print(
        f"[size] grid={args.grid}, npts={r1_dvr.npts}x{r2_dvr.npts}, "
        f"dim={H.shape[0]}, H nnz={H.nnz}"
    )
    print(
        "[overlap] neighbor links={link_count}, kinetic edges={kept_edges}, "
        "link_time={link_time:.3f} s, path_time={path_time:.3f} s".format(**meta)
    )
    print(f"[timing] H build={build_time:.3f} s")
    print(f"[timing] expm_multiply={prop_time:.3f} s")
    if args.cap_strength > 0.0:
        print(
            "[CAP] strength={:.6g} Eh, width_min={:.3f}, width_max={:.3f}, "
            "order={}, max={:.6g} Eh".format(
                args.cap_strength,
                args.cap_width_min,
                args.cap_width_max,
                args.cap_order,
                meta["cap_max"],
            )
        )
    print(f"[norm] min={norms.min():.12f}, max={norms.max():.12f}")
    for target in (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0):
        if target > times_fs[-1]:
            continue
        idx = int(np.argmin(np.abs(times_fs - target)))
        print(f"[snapshot] t={times_fs[idx]:.2f} fs {np.array2string(populations[idx], precision=8)}")
    print(f"[final] t={times_fs[-1]:.2f} fs {np.array2string(populations[-1], precision=8)}")
    print(f"[max S1] {float(populations[:, 1].max()):.8e} at {times_fs[int(populations[:, 1].argmax())]:.2f} fs")
    print(f"[data] {out}")
    print(f"[plot] {png}")
    print(f"[wavepackets] {wavepacket_png}")
    return {
        "backend": backend,
        "times_fs": times_fs,
        "populations": populations,
        "norms": norms,
        "data": out,
        "plot": png,
        "wavepacket_plot": wavepacket_png,
        "meta": meta,
        "build_time": build_time,
        "propagation_time": prop_time,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=("sine", "fedvr"), default="sine")
    parser.add_argument("--npts", type=int, default=21)
    parser.add_argument("--n-elements", type=int, default=8)
    parser.add_argument("--n-lobatto", type=int, default=5)
    parser.add_argument("--r-min", type=float, default=0.90)
    parser.add_argument("--r-max", type=float, default=3.20)
    parser.add_argument("--theta-deg", type=float, default=60.0)
    parser.add_argument("--backend", choices=("am1-meci", "rhf-casci", "both"), default="am1-meci")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--ncas", type=int, default=3)
    parser.add_argument("--nelecas", type=int, default=2)
    parser.add_argument("--initial-state", type=int, default=1)
    parser.add_argument(
        "--initial-electronic-projection",
        choices=("local", "reference-overlap"),
        default="local",
        help=(
            "Use 'local' to put amplitude on the same adiabatic state index at "
            "every grid point, or 'reference-overlap' to project one reference "
            "electronic state onto each local adiabatic basis using overlaps."
        ),
    )
    parser.add_argument("--reference-r1", type=float, default=None)
    parser.add_argument("--reference-r2", type=float, default=None)
    parser.add_argument(
        "--initial-nuclear",
        choices=("gaussian", "fc-ground"),
        default="gaussian",
        help=(
            "Use the analytic Gaussian packet or an FC-like vibrational ground "
            "state on --fc-surface."
        ),
    )
    parser.add_argument(
        "--fc-surface",
        type=int,
        default=0,
        help="Adiabatic surface used to compute the FC-like nuclear ground state.",
    )
    parser.add_argument("--packet-r1", type=float, default=1.36)
    parser.add_argument("--packet-r2", type=float, default=2.28)
    parser.add_argument("--packet-width", type=float, default=5.0)
    parser.add_argument("--tmax-fs", type=float, default=3.0)
    parser.add_argument("--nt", type=int, default=151)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--scf-tol", type=float, default=1.0e-9)
    parser.add_argument("--max-cycle", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--kinetic-threshold", type=float, default=1.0e-13)
    parser.add_argument("--block-threshold", type=float, default=1.0e-14)
    parser.add_argument(
        "--cap-strength",
        type=float,
        default=0.0,
        help="CAP strength in Hartree. The Hamiltonian uses H - i W, with W >= 0.",
    )
    parser.add_argument(
        "--cap-width-min",
        type=float,
        default=0.0,
        help="Width in bohr of the lower-boundary CAP on both r1 and r2.",
    )
    parser.add_argument(
        "--cap-width-max",
        type=float,
        default=0.0,
        help="Width in bohr of the upper-boundary CAP on both r1 and r2.",
    )
    parser.add_argument("--cap-order", type=int, default=2)
    parser.add_argument("--force-scan", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/private/tmp/h3plus_am1_meci_fedvr_chebyshev_benchmark"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    backends = ("rhf-casci", "am1-meci") if args.backend == "both" else (args.backend,)
    results = [run_backend(backend, args) for backend in backends]

    if len(results) > 1:
        series = [
            (result["backend"], result["times_fs"], result["populations"], 1.0)
            for result in results
        ]
        compare_png = args.outdir / (
            f"h3plus_rhf_casci_vs_am1_meci_sine{args.npts}_linked_ldr_"
            f"{args.initial_electronic_projection.replace('-', '_')}_"
            f"r{args.packet_r1:.3f}_{args.packet_r2:.3f}_"
            f"w{args.packet_width:.3f}_t{args.tmax_fs:.3f}_populations.png"
        )
        compare_png = Path(
            str(compare_png).replace("-", "m").replace(".", "p", str(compare_png).count(".") - 1)
        )
        plot_populations(series, compare_png)
        ref = results[0]
        for result in results[1:]:
            interp = np.column_stack(
                [
                    np.interp(ref["times_fs"], result["times_fs"], result["populations"][:, state])
                    for state in range(result["populations"].shape[1])
                ]
            )
            diff = float(np.max(np.abs(interp - ref["populations"])))
            print(
                f"[compare] {result['backend']} vs {ref['backend']} "
                f"max_abs_population_diff={diff:.8f}"
            )
        print(f"[compare plot] {compare_png}")


if __name__ == "__main__":
    main()
