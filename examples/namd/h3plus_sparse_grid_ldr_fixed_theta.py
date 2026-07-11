#!/usr/bin/env python3
"""Compare direct sparse-grid LDR against FE-DVR for fixed-angle H3+."""

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
import scipy.sparse as sp
import scipy.linalg as la
import scipy.sparse.linalg as sla

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.dvr import FEDVR
from pyqed.qchem import CASCI, Molecule
from pyqed.qchem.mcscf.casci import overlap as casci_overlap
from pyqed.smolyak.interpolator import SparseInterpolator
from pyqed.smolyak.sg import SparseGridLDR
from pyqed.units import amu2au, au2fs

from h3plus_fedvr_fixed_theta import (
    HARTREE_TO_EV,
    casci_energy,
    fixed_theta_stretch_kinetic,
    h3plus_body_frame,
    initial_packet,
    region_masks,
    region_population,
    run_basis,
    solve_ground_surface,
)


def normalize_electronic_method(method):
    method = str(method).lower().replace("_", "-")
    aliases = {
        "am1": "am1-meci",
        "meci": "am1-meci",
        "am1/meci": "am1-meci",
        "cas": "casci",
    }
    return aliases.get(method, method)


class FEArgs:
    def __init__(self, args):
        self.basis = args.basis
        self.ncas = args.ncas
        self.nelecas = args.nelecas
        self.nstates = args.nstates
        self.worker_threads = args.worker_threads
        self.force = args.force
        self.nlevels = args.nlevels


def fixed_theta_g_matrix(theta):
    proton_mass = 1.00782503223 * amu2au
    g11 = 2.0 / proton_mass
    g22 = 2.0 / proton_mass
    g12 = np.cos(theta) / proton_mass
    return np.array([[g11, g12], [g12, g22]], dtype=float)


def scan_sparse_grid_apes(sg, theta, args):
    apes = np.zeros((sg.npts, args.nstates), dtype=float)
    total = sg.npts
    t0 = time.perf_counter()
    for i, (r1, r2) in enumerate(sg.nodes, start=1):
        apes[i - 1] = casci_energy(
            r1,
            r2,
            theta,
            args.basis,
            args.ncas,
            args.nelecas,
            args.nstates,
        )
        print(
            f"[sg scan] {i:3d}/{total}: r1={r1:.6f} r2={r2:.6f} "
            f"E0={apes[i - 1, 0]:.10f}"
        )
    print(f"[sg scan] completed in {time.perf_counter() - t0:.2f} s")
    return apes


def run_casci_object(r1, r2, theta, args):
    mol = Molecule(
        atom=h3plus_body_frame(r1, r2, theta),
        basis=args.basis,
        charge=1,
        spin=0,
        unit="bohr",
    )
    mol.build()
    mf = mol.RHF(verbose=0).run(max_cycle=80)
    return CASCI(
        mf,
        ncas=args.ncas,
        nelecas=args.nelecas,
        verbose=0,
    ).run(nstates=args.nstates)


def run_electronic_object(r1, r2, theta, args):
    method = normalize_electronic_method(args.electronic_method)
    if method == "casci":
        return run_casci_object(r1, r2, theta, args)
    if method != "am1-meci":
        raise ValueError("--electronic-method must be 'casci' or 'am1/meci'.")

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


def electronic_state_overlap(left, right):
    if hasattr(left, "wavefunction_overlap"):
        return left.wavefunction_overlap(right)
    return casci_overlap(left, right)


def scan_fe_casci_grid(r1_dvr, r2_dvr, theta, args):
    apes = np.zeros((r1_dvr.npts, r2_dvr.npts, args.nstates), dtype=float)
    mc_grid = np.empty((r1_dvr.npts, r2_dvr.npts), dtype=object)
    total = r1_dvr.npts * r2_dvr.npts
    count = 0
    t0 = time.perf_counter()
    for i, r1 in enumerate(r1_dvr.x):
        for j, r2 in enumerate(r2_dvr.x):
            count += 1
            mc = run_casci_object(r1, r2, theta, args)
            mc_grid[i, j] = mc
            apes[i, j] = np.asarray(mc.e_tot[: args.nstates], dtype=float)
            print(
                f"[ldr scan] {count:3d}/{total}: r1={r1:.6f} r2={r2:.6f} "
                f"E0={apes[i, j, 0]:.10f}"
            )
    print(f"[ldr scan] completed in {time.perf_counter() - t0:.2f} s")
    return apes, mc_grid


def build_full_casci_overlap(mc_grid, nstates):
    flat = mc_grid.reshape(-1)
    ng = len(flat)
    A = np.zeros((ng, nstates, ng, nstates), dtype=complex)
    eye = np.eye(nstates, dtype=complex)
    t0 = time.perf_counter()
    for i in range(ng):
        A[i, :, i, :] = eye
        for j in range(i + 1, ng):
            block = np.asarray(casci_overlap(flat[i], flat[j]), dtype=complex)
            if block.shape != (nstates, nstates):
                raise ValueError(f"CASCI overlap block shape {block.shape} != {(nstates, nstates)}")
            A[i, :, j, :] = block
            A[j, :, i, :] = block.conj().T
    print(f"[ldr overlap] full A built in {time.perf_counter() - t0:.2f} s")
    return A


def build_ldr_hamiltonian(r1_dvr, r2_dvr, theta, apes, overlap_matrix):
    proton_mass = 1.00782503223 * amu2au
    masses_au = np.array([proton_mass, proton_mass, proton_mass])
    T = fixed_theta_stretch_kinetic(r1_dvr, r2_dvr, theta, masses_au).tocoo()
    ng, nstates = r1_dvr.npts * r2_dvr.npts, apes.shape[-1]
    A = overlap_matrix.reshape(ng, nstates, ng, nstates)

    rows, cols, data = [], [], []
    for i, j, tij in zip(T.row, T.col, T.data):
        block = tij * A[i, :, j, :]
        nz_a, nz_b = np.nonzero(np.abs(block) > 1e-14)
        rows.extend((i * nstates + nz_a).tolist())
        cols.extend((j * nstates + nz_b).tolist())
        data.extend(block[nz_a, nz_b].tolist())

    dim = ng * nstates
    kinetic = sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))
    kinetic = 0.5 * (kinetic + kinetic.getH())
    potential = sp.diags(apes.reshape(-1), format="csr", dtype=complex)
    return kinetic + potential


def project_reference_packet(r1_dvr, r2_dvr, overlap_matrix, state, center, width):
    nstates = overlap_matrix.shape[1]
    r1 = np.asarray(r1_dvr.x, dtype=float)[:, None]
    r2 = np.asarray(r2_dvr.x, dtype=float)[None, :]
    w1 = np.asarray(r1_dvr.w, dtype=float)[:, None]
    w2 = np.asarray(r2_dvr.w, dtype=float)[None, :]
    r10, r20 = center
    envelope = np.exp(-width * ((r1 - r10) ** 2 + (r2 - r20) ** 2))
    iref = int(np.argmin(np.abs(r1_dvr.x - r10)))
    jref = int(np.argmin(np.abs(r2_dvr.x - r20)))
    ref_flat = np.ravel_multi_index((iref, jref), (r1_dvr.npts, r2_dvr.npts))
    A = overlap_matrix.reshape(r1_dvr.npts * r2_dvr.npts, nstates, -1, nstates)

    coeff = np.zeros((r1_dvr.npts * r2_dvr.npts, nstates), dtype=complex)
    weights = np.sqrt(w1 * w2).reshape(-1)
    for flat in range(coeff.shape[0]):
        coeff[flat] = envelope.reshape(-1)[flat] * weights[flat] * A[flat, :, ref_flat, state]
    coeff = coeff.reshape(-1)
    norm = np.linalg.norm(coeff)
    if norm == 0.0:
        raise ValueError("Reference-projected LDR packet has zero norm.")
    return coeff / norm


def local_state_packet(r1_dvr, r2_dvr, state, nstates, center, width):
    scalar = initial_packet(r1_dvr, r2_dvr, center, width)
    coeff = np.zeros((scalar.size, nstates), dtype=complex)
    coeff[:, state] = scalar
    return coeff.reshape(-1)


def electronic_populations(coeff, nstates):
    return np.sum(np.abs(coeff.reshape(-1, nstates)) ** 2, axis=0)


def region_total_populations(coeff, r1_dvr, r2_dvr, nstates):
    density = np.sum(np.abs(coeff.reshape(-1, nstates)) ** 2, axis=1)
    lower, upper, bridge = region_masks(r1_dvr, r2_dvr)
    return np.array(
        [density[lower].sum(), density[upper].sum(), density[bridge].sum(), density.sum()],
        dtype=float,
    )


def propagate_ldr_populations(result, center, width, state, project_reference, times_fs):
    r1_dvr, r2_dvr = result["basis"]
    nstates = result["apes"].shape[-1]
    if project_reference:
        psi = project_reference_packet(
            r1_dvr,
            r2_dvr,
            result["overlap_matrix"],
            state,
            center,
            width,
        )
    else:
        psi = local_state_packet(r1_dvr, r2_dvr, state, nstates, center, width)

    evals, evecs = la.eigh(result["H"].toarray())
    amps = evecs.conj().T @ psi
    epops = np.zeros((len(times_fs), nstates), dtype=float)
    rpops = np.zeros((len(times_fs), 4), dtype=float)
    for i, time_fs in enumerate(times_fs):
        coeff = evecs @ (np.exp(-1j * evals * time_fs / au2fs) * amps)
        epops[i] = electronic_populations(coeff, nstates)
        rpops[i] = region_total_populations(coeff, r1_dvr, r2_dvr, nstates)
    return epops, rpops


def run_full_overlap_ldr(args, theta):
    r1_dvr = FEDVR(args.r_min, args.r_max, args.n_elements, args.n_lobatto)
    r2_dvr = FEDVR(args.r_min, args.r_max, args.n_elements, args.n_lobatto)
    cache = args.outdir / (
        f"h3plus_full_overlap_ldr_theta{args.theta_deg:.1f}_"
        f"r{args.r_min:.2f}_{args.r_max:.2f}_"
        f"e{args.n_elements}_p{args.n_lobatto}_s{args.nstates}.npz"
    )

    t0 = time.perf_counter()
    if cache.exists() and not args.force:
        data = np.load(cache)
        apes = data["apes"]
        overlap_matrix = data["overlap_matrix"]
        scan_time = time.perf_counter() - t0
        print(f"[cache] loaded {cache}")
    else:
        apes, mc_grid = scan_fe_casci_grid(r1_dvr, r2_dvr, theta, args)
        overlap_matrix = build_full_casci_overlap(mc_grid, args.nstates)
        np.savez(
            cache,
            apes=apes,
            overlap_matrix=overlap_matrix,
            r1=r1_dvr.x,
            r2=r2_dvr.x,
            theta=theta,
        )
        scan_time = time.perf_counter() - t0
        print(f"[cache] saved {cache}")

    t0 = time.perf_counter()
    H = build_ldr_hamiltonian(r1_dvr, r2_dvr, theta, apes, overlap_matrix)
    build_time = time.perf_counter() - t0

    return {
        "basis": (r1_dvr, r2_dvr),
        "apes": apes,
        "overlap_matrix": overlap_matrix,
        "H": H,
        "scan_time": scan_time,
        "build_time": build_time,
        "eig_time": 0.0,
        "cache": cache,
    }


def make_sparse_grid(args, theta):
    index_rule = "smolyak" if args.sg_index_rule == "adaptive-diagonal" else args.sg_index_rule
    sg = SparseGridLDR(
        ndim=2,
        level=args.sg_level,
        domain=((args.r_min, args.r_max), (args.r_min, args.r_max)),
        g_matrix=fixed_theta_g_matrix(theta),
        index_rule=index_rule,
    )
    if args.sg_index_rule == "adaptive-diagonal":
        packet = np.array([args.packet_r1, args.packet_r2], dtype=float)
        mirror = packet[::-1]
        diagonal_width = float(args.sg_diagonal_width)
        packet_radius = float(args.sg_packet_radius)

        def refine_predicate(point):
            return (
                abs(point[0] - point[1]) <= diagonal_width
                or np.linalg.norm(point - packet) <= packet_radius
                or np.linalg.norm(point - mirror) <= packet_radius
            )

        sg.refine_tensor_region(level=args.sg_level, predicate=refine_predicate)
        print(
            f"[sg refine] adaptive-diagonal basis: {sg.npts} functions "
            f"(diag width={diagonal_width}, packet radius={packet_radius})"
        )
    return sg


def sparse_grid_refine_tag(args):
    if args.sg_index_rule != "adaptive-diagonal":
        return ""
    return (
        f"_diag{args.sg_diagonal_width:.2f}"
        f"_pr{args.sg_packet_radius:.2f}"
    )


def run_sparse_grid(args, theta):
    refine_tag = sparse_grid_refine_tag(args)
    cache = args.outdir / (
        f"h3plus_sparse_grid_theta{args.theta_deg:.1f}_"
        f"r{args.r_min:.2f}_{args.r_max:.2f}_l{args.sg_level}_"
        f"{args.sg_index_rule}{refine_tag}.npz"
    )
    sg = make_sparse_grid(args, theta)

    t0 = time.perf_counter()
    if cache.exists() and not args.force:
        data = np.load(cache)
        apes = data["apes"]
        if apes.shape[0] != sg.npts:
            raise ValueError(
                f"Cached SG APES has {apes.shape[0]} points, expected {sg.npts}; "
                "rerun with --force."
            )
        scan_time = time.perf_counter() - t0
        print(f"[cache] loaded {cache}")
    else:
        apes = scan_sparse_grid_apes(sg, theta, args)
        np.savez(
            cache,
            apes=apes,
            nodes=sg.nodes,
            theta=theta,
            index_rule=np.asarray(args.sg_index_rule),
            basis_indices=np.asarray(sg.basis_indices, dtype=int),
        )
        scan_time = time.perf_counter() - t0
        print(f"[cache] saved {cache}")

    t0 = time.perf_counter()
    H = sg.build_hamiltonian(
        apes[:, 0],
        quadrature_order=args.potential_quad_order,
    )
    S = sg.overlap()
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    evals, evecs = sg.solve(
        apes[:, 0],
        nstates=args.nlevels,
        quadrature_order=args.potential_quad_order,
    )
    eig_time = time.perf_counter() - t0

    return {
        "basis": sg,
        "apes": apes,
        "H": H,
        "S": S,
        "levels": evals,
        "evecs": evecs,
        "scan_time": scan_time,
        "build_time": build_time,
        "eig_time": eig_time,
        "cache": cache,
    }


def scan_sparse_grid_electronic_objects(sg, theta, args):
    apes = np.zeros((sg.npts, args.nstates), dtype=float)
    objects = np.empty(sg.npts, dtype=object)
    total = sg.npts
    t0 = time.perf_counter()
    for i, (r1, r2) in enumerate(sg.nodes, start=1):
        mc = run_electronic_object(r1, r2, theta, args)
        objects[i - 1] = mc
        energies = getattr(mc, "e_tot", getattr(mc, "e", None))
        if energies is None:
            raise AttributeError("Electronic object has neither e_tot nor e energies.")
        apes[i - 1] = np.asarray(energies, dtype=float)[: args.nstates]
        print(
            f"[sg ldr scan] {i:3d}/{total}: r1={r1:.6f} r2={r2:.6f} "
            f"E0={apes[i - 1, 0]:.10f}"
        )
    print(f"[sg ldr scan] completed in {time.perf_counter() - t0:.2f} s")
    return apes, objects


def build_sparse_grid_full_overlap(objects, nstates):
    ng = len(objects)
    A = np.zeros((ng, nstates, ng, nstates), dtype=complex)
    eye = np.eye(nstates, dtype=complex)
    t0 = time.perf_counter()
    for i in range(ng):
        A[i, :, i, :] = eye
        for j in range(i + 1, ng):
            block = np.asarray(electronic_state_overlap(objects[i], objects[j]), dtype=complex)
            if block.shape != (nstates, nstates):
                raise ValueError(f"Electronic overlap block shape {block.shape} != {(nstates, nstates)}")
            A[i, :, j, :] = block
            A[j, :, i, :] = block.conj().T
    print(f"[sg ldr overlap] full A built in {time.perf_counter() - t0:.2f} s")
    return A


def kron_with_electronic_overlap(spatial, overlap_matrix):
    spatial = spatial.tocoo()
    ng, nstates = overlap_matrix.shape[:2]
    rows, cols, data = [], [], []
    for i, j, value in zip(spatial.row, spatial.col, spatial.data):
        block = value * overlap_matrix[i, :, j, :]
        nz_a, nz_b = np.nonzero(np.abs(block) > 1e-14)
        rows.extend((i * nstates + nz_a).tolist())
        cols.extend((j * nstates + nz_b).tolist())
        data.extend(block[nz_a, nz_b].tolist())
    shape = (ng * nstates, ng * nstates)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=shape)
    return 0.5 * (matrix + matrix.getH())


def build_sparse_grid_overlap_ldr_matrices(sg, apes, overlap_matrix):
    if sg.S is None:
        sg.build_overlap()
    if sg.T is None:
        sg.build_kinetic()
    nstates = apes.shape[1]
    ng = sg.npts

    B = kron_with_electronic_overlap(sg.S, overlap_matrix)
    kinetic = kron_with_electronic_overlap(sg.T, overlap_matrix)

    rows, cols, data = [], [], []
    for i, j, sij in zip(sg.S.tocoo().row, sg.S.tocoo().col, sg.S.tocoo().data):
        block = 0.5 * (apes[i, :, None] + apes[j, None, :])
        block = sij * block * overlap_matrix[i, :, j, :]
        nz_a, nz_b = np.nonzero(np.abs(block) > 1e-14)
        rows.extend((i * nstates + nz_a).tolist())
        cols.extend((j * nstates + nz_b).tolist())
        data.extend(block[nz_a, nz_b].tolist())
    shape = (ng * nstates, ng * nstates)
    potential = sp.csr_matrix((data, (rows, cols)), shape=shape)
    potential = 0.5 * (potential + potential.getH())
    return kinetic + potential, B


def run_sparse_grid_full_overlap_ldr(args, theta):
    method = normalize_electronic_method(args.electronic_method)
    refine_tag = sparse_grid_refine_tag(args)
    cache = args.outdir / (
        f"h3plus_sg_full_overlap_ldr_{method}_theta{args.theta_deg:.1f}_"
        f"r{args.r_min:.2f}_{args.r_max:.2f}_l{args.sg_level}_"
        f"{args.sg_index_rule}{refine_tag}_s{args.nstates}.npz"
    )
    sg = make_sparse_grid(args, theta)

    t0 = time.perf_counter()
    if cache.exists() and not args.force:
        data = np.load(cache)
        apes = data["apes"]
        overlap_matrix = data["overlap_matrix"]
        if apes.shape[0] != sg.npts:
            raise ValueError(
                f"Cached SG full-overlap APES has {apes.shape[0]} points, "
                f"expected {sg.npts}; rerun with --force."
            )
        scan_time = time.perf_counter() - t0
        print(f"[cache] loaded {cache}")
    else:
        apes, objects = scan_sparse_grid_electronic_objects(sg, theta, args)
        overlap_matrix = build_sparse_grid_full_overlap(objects, args.nstates)
        np.savez(
            cache,
            apes=apes,
            overlap_matrix=overlap_matrix,
            nodes=sg.nodes,
            theta=theta,
            electronic_method=np.asarray(method),
            basis_indices=np.asarray(sg.basis_indices, dtype=int),
        )
        scan_time = time.perf_counter() - t0
        print(f"[cache] saved {cache}")

    t0 = time.perf_counter()
    H, B = build_sparse_grid_overlap_ldr_matrices(sg, apes, overlap_matrix)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    evals, evecs = la.eigh(H.toarray(), B.toarray())
    eig_time = time.perf_counter() - t0

    return {
        "basis": sg,
        "apes": apes,
        "overlap_matrix": overlap_matrix,
        "H": H,
        "S": B,
        "levels": evals,
        "evecs": evecs,
        "scan_time": scan_time,
        "build_time": build_time,
        "eig_time": eig_time,
        "cache": cache,
    }


def run_fedvr(args, theta):
    r1_dvr = FEDVR(args.r_min, args.r_max, args.n_elements, args.n_lobatto)
    r2_dvr = FEDVR(args.r_min, args.r_max, args.n_elements, args.n_lobatto)
    cache = args.outdir / (
        f"h3plus_fedvr_theta{args.theta_deg:.1f}_"
        f"r{args.r_min:.2f}_{args.r_max:.2f}_"
        f"e{args.n_elements}_p{args.n_lobatto}.npz"
    )

    fe_args = FEArgs(args)
    t0 = time.perf_counter()
    apes, _, _, _ = run_basis(
        "FE-DVR",
        r1_dvr,
        r2_dvr,
        theta,
        fe_args,
        cache,
        cache.with_suffix(".png"),
    )
    scan_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    levels, kinetic, H = solve_ground_surface(
        r1_dvr,
        r2_dvr,
        theta,
        apes,
        min(args.nlevels, r1_dvr.npts * r2_dvr.npts - 1),
    )
    build_eig_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    dense_evals, dense_evecs = la.eigh(H.toarray())
    dense_eig_time = time.perf_counter() - t0

    return {
        "basis": (r1_dvr, r2_dvr),
        "apes": apes,
        "H": H,
        "levels": levels,
        "dense_levels": dense_evals,
        "dense_evecs": dense_evecs,
        "kinetic": kinetic,
        "scan_time": scan_time,
        "build_time": build_eig_time - dense_eig_time,
        "eig_time": dense_eig_time,
        "cache": cache,
    }


def _tensor_points(r1_dvr, r2_dvr):
    r1, r2 = np.meshgrid(r1_dvr.x, r2_dvr.x, indexing="ij")
    return np.column_stack([r1.reshape(-1), r2.reshape(-1)])


def run_sparse_apes_on_fedvr(args, theta, fe):
    r1_dvr, r2_dvr = fe["basis"]
    target = _tensor_points(r1_dvr, r2_dvr)
    cache = args.outdir / (
        f"h3plus_{args.interp_type.lower()}_sparse_apes_theta{args.theta_deg:.1f}_"
        f"r{args.r_min:.2f}_{args.r_max:.2f}_"
        f"l{args.interp_level}_fe_e{args.n_elements}_p{args.n_lobatto}.npz"
    )

    t0 = time.perf_counter()
    if cache.exists() and not args.force:
        data = np.load(cache)
        apes = data["apes"]
        sample_count = int(data["sample_count"])
        scan_time = time.perf_counter() - t0
        print(f"[cache] loaded {cache}")
    else:
        samples = {}

        def sparse_apes(points):
            out = np.empty(len(points), dtype=float)
            for i, (r1, r2) in enumerate(points):
                key = (round(float(r1), 13), round(float(r2), 13))
                if key not in samples:
                    energy = casci_energy(
                        r1,
                        r2,
                        theta,
                        args.basis,
                        args.ncas,
                        args.nelecas,
                        args.nstates,
                    )[0]
                    samples[key] = energy
                    print(
                        f"[{args.interp_type} scan] {len(samples):3d}: "
                        f"r1={r1:.6f} r2={r2:.6f} E0={energy:.10f}"
                    )
                out[i] = samples[key]
            return out

        interval = np.array(
            [[args.r_min, args.r_min], [args.r_max, args.r_max]],
            dtype=float,
        )
        interpolator = SparseInterpolator(
            args.interp_level,
            2,
            interpolation_type=args.interp_type,
            interpolation_interval=interval,
            tol=0.0,
        )
        interpolated = interpolator.fit(sparse_apes, target)
        apes = np.zeros_like(fe["apes"])
        apes[:, :, 0] = interpolated.reshape(r1_dvr.npts, r2_dvr.npts)
        sample_count = len(samples)
        np.savez(
            cache,
            apes=apes,
            sample_count=sample_count,
            theta=theta,
            target=target,
        )
        scan_time = time.perf_counter() - t0
        print(f"[cache] saved {cache}")

    t0 = time.perf_counter()
    levels, kinetic, H = solve_ground_surface(
        r1_dvr,
        r2_dvr,
        theta,
        apes,
        min(args.nlevels, r1_dvr.npts * r2_dvr.npts - 1),
    )
    build_eig_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    dense_evals, dense_evecs = la.eigh(H.toarray())
    dense_eig_time = time.perf_counter() - t0

    return {
        "basis": (r1_dvr, r2_dvr),
        "apes": apes,
        "H": H,
        "levels": levels,
        "dense_levels": dense_evals,
        "dense_evecs": dense_evecs,
        "kinetic": kinetic,
        "scan_time": scan_time,
        "build_time": build_eig_time - dense_eig_time,
        "eig_time": dense_eig_time,
        "cache": cache,
        "sample_count": sample_count,
    }


def project_fe_packet(result, center, width, nlevels):
    r1_dvr, r2_dvr = result["basis"]
    psi = initial_packet(r1_dvr, r2_dvr, center, width)
    evecs = result["dense_evecs"][:, :nlevels]
    coeff = evecs @ (evecs.conj().T @ psi)
    coeff /= np.linalg.norm(coeff)
    weights = np.abs(evecs.conj().T @ psi) ** 2
    return coeff, weights.sum()


def project_sg_packet(result, center, width, nlevels, projection_order):
    sg = result["basis"]
    center = np.asarray(center, dtype=float)

    def packet(points):
        dr = points - center
        return np.exp(-width * np.sum(dr * dr, axis=1))

    coeff = sg.l2_project(packet, order=projection_order)
    S = result["S"]
    coeff = coeff.astype(complex)
    coeff /= np.sqrt(np.vdot(coeff, S @ coeff).real)
    evecs = result["evecs"][:, :nlevels]
    projected = evecs @ (evecs.conj().T @ (S @ coeff))
    projected /= np.sqrt(np.vdot(projected, S @ projected).real)
    weights = np.abs(evecs.conj().T @ (S @ coeff)) ** 2
    return projected, weights.sum()


def project_sg_reference_packet(result, center, width, state, nlevels):
    sg = result["basis"]
    nstates = result["apes"].shape[1]
    center = np.asarray(center, dtype=float)
    dr = sg.nodes - center
    envelope = np.exp(-width * np.sum(dr * dr, axis=1))
    iref = int(np.argmin(np.sum((sg.nodes - center) ** 2, axis=1)))
    values = envelope[:, None] * result["overlap_matrix"][:, :, iref, state]
    coeff = sg.nodal_values_to_coefficients(values).reshape(-1).astype(complex)

    B = result["S"]
    norm = np.vdot(coeff, B @ coeff).real
    if norm <= 0.0:
        raise ValueError("Reference-projected SG-LDR packet has non-positive norm.")
    coeff /= np.sqrt(norm)

    evecs = result["evecs"][:, :nlevels]
    amplitudes = evecs.conj().T @ (B @ coeff)
    projected = evecs @ amplitudes
    projected /= np.sqrt(np.vdot(projected, B @ projected).real)
    weights = np.abs(amplitudes) ** 2
    return projected, weights.sum()


def sg_overlap_ldr_populations_over_time(result, coeff0, nlevels, times_fs, quad_order):
    sg = result["basis"]
    nstates = result["apes"].shape[1]
    evals = result["levels"][:nlevels]
    evecs = result["evecs"][:, :nlevels]
    B = result["S"]
    lower_spatial = sp.csr_matrix(regional_overlap_matrices(sg, quad_order))
    lower = kron_with_electronic_overlap(lower_spatial, result["overlap_matrix"])
    upper = B - lower
    amplitudes = evecs.conj().T @ (B @ coeff0)

    epops = np.zeros((len(times_fs), nstates), dtype=float)
    rpops = np.zeros((len(times_fs), 4), dtype=float)
    state_slices = [np.arange(sg.npts) * nstates + state for state in range(nstates)]
    state_metrics = [B[rows][:, rows].tocsr() for rows in state_slices]
    for i, time_fs in enumerate(times_fs):
        coeff = evecs @ (np.exp(-1j * evals * time_fs / au2fs) * amplitudes)
        total = np.vdot(coeff, B @ coeff).real
        lower_pop = np.vdot(coeff, lower @ coeff).real
        upper_pop = np.vdot(coeff, upper @ coeff).real
        rpops[i] = (lower_pop, upper_pop, 0.0, total)
        for state, (rows, block) in enumerate(zip(state_slices, state_metrics)):
            vec = coeff[rows]
            epops[i, state] = np.vdot(vec, block @ vec).real
    return epops, rpops


def fe_populations_over_time(result, coeff0, nlevels, times_fs):
    r1_dvr, r2_dvr = result["basis"]
    lower, upper, bridge = region_masks(r1_dvr, r2_dvr)
    evals = result["dense_levels"][:nlevels]
    evecs = result["dense_evecs"][:, :nlevels]
    amplitudes = evecs.conj().T @ coeff0
    pops = np.zeros((len(times_fs), 4), dtype=float)
    for i, time_fs in enumerate(times_fs):
        coeff = evecs @ (np.exp(-1j * evals * time_fs / args_au2fs()) * amplitudes)
        pops[i] = region_population(coeff, lower, upper, bridge)
    return pops


def args_au2fs():
    from pyqed.units import au2fs

    return au2fs


def regional_overlap_matrices(sg, order):
    points, weights = sg.quadrature_points(order=order, cellwise=True)
    phi = sg.interpolation_matrix(points)
    lower = points[:, 0] < points[:, 1]
    weighted_phi = phi * weights[:, None] * lower[:, None]
    return weighted_phi.T @ phi


def sg_populations_over_time(result, coeff0, nlevels, times_fs, quad_order):
    sg = result["basis"]
    evals = result["levels"][:nlevels]
    evecs = result["evecs"][:, :nlevels]
    S = result["S"]
    lower_m = regional_overlap_matrices(sg, quad_order)
    amplitudes = evecs.conj().T @ (S @ coeff0)
    pops = np.zeros((len(times_fs), 4), dtype=float)
    for i, time_fs in enumerate(times_fs):
        coeff = evecs @ (np.exp(-1j * evals * time_fs / args_au2fs()) * amplitudes)
        lower = np.vdot(coeff, lower_m @ coeff).real
        total = np.vdot(coeff, S @ coeff).real
        pops[i] = (lower, total - lower, 0.0, total)
    return pops


def plot_population_comparison(times, fe_pops, sg_pops, outpath, interp_pops=None):
    fig, (ax_pop, ax_norm) = plt.subplots(
        2,
        1,
        figsize=(6.2, 5.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_pop.plot(times, fe_pops[:, 0], color="tab:blue", lw=2, label="FE-DVR: r1 < r2")
    ax_pop.plot(times, fe_pops[:, 1], color="tab:blue", lw=1.8, ls="--", label="FE-DVR: r1 > r2")
    ax_pop.plot(times, sg_pops[:, 0], color="tab:green", lw=2, label="SG-LDR: r1 < r2")
    ax_pop.plot(times, sg_pops[:, 1], color="tab:green", lw=1.8, ls="--", label="SG-LDR: r1 > r2")
    ax_norm.plot(times, fe_pops[:, 3], color="tab:blue", lw=1.8, label="FE-DVR")
    ax_norm.plot(times, sg_pops[:, 3], color="tab:green", lw=1.8, label="SG-LDR")
    if interp_pops is not None:
        ax_pop.plot(times, interp_pops[:, 0], color="tab:orange", lw=2, label="Sparse APES/FE: r1 < r2")
        ax_pop.plot(
            times,
            interp_pops[:, 1],
            color="tab:orange",
            lw=1.8,
            ls="--",
            label="Sparse APES/FE: r1 > r2",
        )
        ax_norm.plot(times, interp_pops[:, 3], color="tab:orange", lw=1.8, label="Sparse APES/FE")
    ax_pop.set_ylabel("population")
    ax_pop.set_ylim(-0.03, 1.03)
    ax_pop.legend(frameon=False, ncol=2, fontsize=8)
    ax_norm.set_xlabel("time / fs")
    ax_norm.set_ylabel("norm")
    ax_norm.set_ylim(0.995, 1.005)
    ax_norm.legend(frameon=False, fontsize=8)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_ldr_populations(times, electronic_pops, region_pops, outpath):
    fig, (ax_el, ax_reg, ax_norm) = plt.subplots(
        3,
        1,
        figsize=(6.2, 6.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 2, 1]},
    )
    for state in range(electronic_pops.shape[1]):
        ax_el.plot(times, electronic_pops[:, state], lw=2, label=f"state {state}")
    ax_reg.plot(times, region_pops[:, 0], color="tab:blue", lw=2, label="r1 < r2")
    ax_reg.plot(times, region_pops[:, 1], color="tab:orange", lw=2, ls="--", label="r1 > r2")
    ax_reg.plot(times, region_pops[:, 2], color="tab:green", lw=1.6, ls=":", label="diagonal")
    ax_norm.plot(times, region_pops[:, 3], color="0.2", lw=1.8)
    ax_el.set_ylabel("electronic pop.")
    ax_reg.set_ylabel("region pop.")
    ax_norm.set_ylabel("norm")
    ax_norm.set_xlabel("time / fs")
    ax_el.set_ylim(-0.03, 1.03)
    ax_reg.set_ylim(-0.03, 1.03)
    ax_norm.set_ylim(0.995, 1.005)
    ax_el.legend(frameon=False, fontsize=8, ncol=2)
    ax_reg.legend(frameon=False, fontsize=8, ncol=3)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def print_timing(label, result, prop_time):
    print(f"\n[{label} timing]")
    print(f"scan/load       {result['scan_time']:.6f} s")
    print(f"H/S build       {result['build_time']:.6f} s")
    print(f"eigensolve      {result['eig_time']:.6f} s")
    print(f"population prop {prop_time:.6f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument(
        "--electronic-method",
        choices=("casci", "am1/meci", "am1", "meci"),
        default="casci",
    )
    parser.add_argument("--ncas", type=int, default=3)
    parser.add_argument("--nelecas", type=int, default=2)
    parser.add_argument("--nstates", type=int, default=1)
    parser.add_argument("--scf-tol", type=float, default=1.0e-9)
    parser.add_argument("--max-cycle", type=int, default=120)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--r-min", type=float, default=0.90)
    parser.add_argument("--r-max", type=float, default=3.20)
    parser.add_argument("--theta-deg", type=float, default=60.0)
    parser.add_argument("--n-elements", type=int, default=5)
    parser.add_argument("--n-lobatto", type=int, default=4)
    parser.add_argument("--sg-level", type=int, default=5)
    parser.add_argument(
        "--sg-index-rule",
        choices=("smolyak", "tensor", "adaptive-diagonal"),
        default="adaptive-diagonal",
    )
    parser.add_argument("--sg-diagonal-width", type=float, default=0.30)
    parser.add_argument("--sg-packet-radius", type=float, default=0.35)
    parser.add_argument("--interp-level", type=int, default=5)
    parser.add_argument("--interp-type", choices=("CH", "CC"), default="CH")
    parser.add_argument("--nlevels", type=int, default=16)
    parser.add_argument("--packet-project-levels", type=int, default=8)
    parser.add_argument("--packet-r1", type=float, default=1.36)
    parser.add_argument("--packet-r2", type=float, default=2.28)
    parser.add_argument("--packet-width", type=float, default=5.0)
    parser.add_argument("--nt", type=int, default=200)
    parser.add_argument("--dt-fs", type=float, default=0.05)
    parser.add_argument("--nout", type=int, default=2)
    parser.add_argument("--quad-order", type=int, default=2)
    parser.add_argument("--potential-quad-order", type=int, default=2)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-full-overlap-ldr", action="store_true")
    parser.add_argument("--run-sg-full-overlap-ldr", action="store_true")
    parser.add_argument("--ldr-initial-state", type=int, default=0)
    parser.add_argument(
        "--no-ldr-project-reference-state",
        action="store_true",
        help="Use a local adiabatic state packet instead of projecting a reference state with A.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name("h3plus_sparse_grid_ldr_fixed_theta"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    theta = np.deg2rad(args.theta_deg)
    center = (args.packet_r1, args.packet_r2)
    times = np.arange(args.nt // args.nout + 1) * args.nout * args.dt_fs

    fe = run_fedvr(args, theta)
    sg = run_sparse_grid(args, theta)
    interp = run_sparse_apes_on_fedvr(args, theta, fe)
    ldr = run_full_overlap_ldr(args, theta) if args.run_full_overlap_ldr else None
    sg_ldr = run_sparse_grid_full_overlap_ldr(args, theta) if args.run_sg_full_overlap_ldr else None

    ncompare = min(args.nlevels, len(fe["levels"]), len(sg["levels"]))
    print("\n[levels] sparse-grid LDR minus FE-DVR")
    print(
        "[levels] relative eV =",
        np.array2string(
            (sg["levels"][:ncompare] - fe["dense_levels"][:ncompare]) * HARTREE_TO_EV,
            precision=8,
        ),
    )
    print(f"[size] FE-DVR points={fe['H'].shape[0]}, H nnz={fe['H'].nnz}")
    print(
        f"[size] SG-LDR basis={sg['basis'].npts} "
        f"({args.sg_index_rule}), H nnz={sg['H'].nnz}, S nnz={sg['S'].nnz}"
    )
    print(
        f"[size] {args.interp_type} sparse APES samples={interp['sample_count']}, "
        f"FE H nnz={interp['H'].nnz}"
    )

    t0 = time.perf_counter()
    fe_coeff, fe_weight = project_fe_packet(
        fe,
        center,
        args.packet_width,
        args.packet_project_levels,
    )
    fe_pops = fe_populations_over_time(
        fe,
        fe_coeff,
        args.packet_project_levels,
        times,
    )
    fe_prop_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sg_coeff, sg_weight = project_sg_packet(
        sg,
        center,
        args.packet_width,
        args.packet_project_levels,
        args.quad_order,
    )
    sg_pops = sg_populations_over_time(
        sg,
        sg_coeff,
        args.packet_project_levels,
        times,
        args.quad_order,
    )
    sg_prop_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    interp_coeff, interp_weight = project_fe_packet(
        interp,
        center,
        args.packet_width,
        args.packet_project_levels,
    )
    interp_pops = fe_populations_over_time(
        interp,
        interp_coeff,
        args.packet_project_levels,
        times,
    )
    interp_prop_time = time.perf_counter() - t0

    print(f"\n[projection] FE-DVR weight={fe_weight:.6f}")
    print(f"[projection] SG-LDR weight={sg_weight:.6f}")
    print(f"[projection] sparse APES/FE weight={interp_weight:.6f}")
    print(
        "[initial] FE-DVR",
        np.array2string(fe_pops[0], precision=8),
    )
    print(
        "[initial] SG-LDR",
        np.array2string(sg_pops[0], precision=8),
    )
    print("[final] FE-DVR", np.array2string(fe_pops[-1], precision=8))
    print("[final] SG-LDR", np.array2string(sg_pops[-1], precision=8))
    print("[final] sparse APES/FE", np.array2string(interp_pops[-1], precision=8))

    print_timing("FE-DVR", fe, fe_prop_time)
    print_timing("SG-LDR", sg, sg_prop_time)
    print_timing(f"{args.interp_type} sparse APES/FE", interp, interp_prop_time)

    if ldr is not None:
        if args.nstates < 1:
            raise ValueError("--nstates must be positive for full-overlap LDR.")
        if not 0 <= args.ldr_initial_state < args.nstates:
            raise ValueError("--ldr-initial-state must be in [0, nstates).")
        t0 = time.perf_counter()
        ldr_epops, ldr_rpops = propagate_ldr_populations(
            ldr,
            center,
            args.packet_width,
            args.ldr_initial_state,
            not args.no_ldr_project_reference_state,
            times,
        )
        ldr_prop_time = time.perf_counter() - t0
        print("\n[full overlap LDR]")
        print(f"[size] H dim={ldr['H'].shape[0]}, H nnz={ldr['H'].nnz}")
        print("[initial electronic]", np.array2string(ldr_epops[0], precision=8))
        print("[final electronic]", np.array2string(ldr_epops[-1], precision=8))
        print("[initial region]", np.array2string(ldr_rpops[0], precision=8))
        print("[final region]", np.array2string(ldr_rpops[-1], precision=8))
        print_timing("full overlap LDR", ldr, ldr_prop_time)

        ldr_plot = args.outdir / (
            f"h3plus_full_overlap_ldr_e{args.n_elements}p{args.n_lobatto}_"
            f"s{args.nstates}_populations.png"
        )
        plot_ldr_populations(times, ldr_epops, ldr_rpops, ldr_plot)
        print(f"[plot] {ldr_plot}")

    if sg_ldr is not None:
        if args.nstates < 1:
            raise ValueError("--nstates must be positive for SG full-overlap LDR.")
        if not 0 <= args.ldr_initial_state < args.nstates:
            raise ValueError("--ldr-initial-state must be in [0, nstates).")
        t0 = time.perf_counter()
        sg_ldr_coeff, sg_ldr_weight = project_sg_reference_packet(
            sg_ldr,
            center,
            args.packet_width,
            args.ldr_initial_state,
            args.packet_project_levels,
        )
        sg_ldr_epops, sg_ldr_rpops = sg_overlap_ldr_populations_over_time(
            sg_ldr,
            sg_ldr_coeff,
            args.packet_project_levels,
            times,
            args.quad_order,
        )
        sg_ldr_prop_time = time.perf_counter() - t0
        print("\n[SG full overlap LDR]")
        print(f"[projection] SG full-overlap weight={sg_ldr_weight:.6f}")
        print(f"[size] H dim={sg_ldr['H'].shape[0]}, H nnz={sg_ldr['H'].nnz}, S nnz={sg_ldr['S'].nnz}")
        print("[initial electronic]", np.array2string(sg_ldr_epops[0], precision=8))
        print("[final electronic]", np.array2string(sg_ldr_epops[-1], precision=8))
        print("[initial region]", np.array2string(sg_ldr_rpops[0], precision=8))
        print("[final region]", np.array2string(sg_ldr_rpops[-1], precision=8))
        print_timing("SG full overlap LDR", sg_ldr, sg_ldr_prop_time)

        sg_ldr_plot = args.outdir / (
            f"h3plus_sg_full_overlap_ldr_{normalize_electronic_method(args.electronic_method)}_"
            f"l{args.sg_level}_{args.sg_index_rule}_s{args.nstates}_populations.png"
        )
        plot_ldr_populations(times, sg_ldr_epops, sg_ldr_rpops, sg_ldr_plot)
        print(f"[plot] {sg_ldr_plot}")

    plot_path = args.outdir / (
        f"h3plus_fe_e{args.n_elements}p{args.n_lobatto}_"
        f"sg_l{args.sg_level}_{args.sg_index_rule}_"
        f"proj{args.packet_project_levels}_populations.png"
    )
    plot_population_comparison(times, fe_pops, sg_pops, plot_path, interp_pops=interp_pops)
    print(f"[plot] {plot_path}")


if __name__ == "__main__":
    main()
