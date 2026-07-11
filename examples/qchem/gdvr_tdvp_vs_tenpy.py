#!/usr/bin/env python3
"""Benchmark pyqed GDVR TDVP against TeNPy TDVP on static H2 GDVR MPOs."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import statistics
import time
import types
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from pyqed.mps.decompose import tt_to_tensor
from pyqed.mps.mps import _mpo_to_dense_operator
from pyqed.mps.tdmps import TDMPS
from pyqed.qchem.gdvr import AtomicChain, TDDMRG
from pyqed.qchem.gdvr.tddmrg import (
    GDVRSpatialDensityPhase,
    GDVRSpatialHybridDensityPhase,
    GDVRSpatialOneBodyRotation,
    GDVRSpatialPronyDensityPhase,
    GDVRSpatialSVDDensityPhase,
)

try:
    from tenpy.algorithms import tdvp as tenpy_tdvp
    from tenpy.linalg import np_conserved as npc
    from tenpy.linalg.charges import LegCharge
    from tenpy.networks.mpo import MPO as TenpyMPO
    from tenpy.networks.mps import MPS as TenpyMPS
    from tenpy.networks.site import Site

    HAVE_TENPY = True
except Exception as exc:  # pragma: no cover - optional dependency
    TENPY_IMPORT_ERROR = exc
    HAVE_TENPY = False


def _state_vector(psi):
    vec = np.asarray(tt_to_tensor(psi.factors), dtype=complex).reshape(-1)
    norm = np.linalg.norm(vec)
    return vec if norm == 0.0 else vec / norm


def _tenpy_state_vector(psi, length, phys_dim):
    theta = psi.get_theta(0, length).to_ndarray()
    tensor = theta.reshape((1,) + (phys_dim,) * length + (1,))[0, ..., 0]
    vec = np.asarray(tensor, dtype=complex).reshape(-1)
    norm = np.linalg.norm(vec)
    return vec if norm == 0.0 else vec / norm


def _state_error(reference, state):
    overlap = min(1.0, float(abs(np.vdot(reference, state))))
    return overlap, float(np.sqrt(max(0.0, 2.0 - 2.0 * overlap)))


def _summary(values):
    values = [float(v) for v in values]
    return {
        "min": min(values),
        "median": float(statistics.median(values)),
        "max": max(values),
        "samples": values,
    }


def _json_ready(obj):
    if isinstance(obj, complex):
        return {"real": float(np.real(obj)), "imag": float(np.imag(obj))}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {
                "real": np.real(obj).tolist(),
                "imag": np.imag(obj).tolist(),
            }
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items() if k != "state"}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def _build_h2_gdvr(nz, args):
    z0 = 0.5 * float(args.h2_bond)
    mol = AtomicChain(["H", "H"], coords=[(0.0, 0.0, -z0), (0.0, 0.0, z0)])
    mol.build(Lz=args.lz, Nz=int(nz), M=args.m, verbose=False)
    mf = mol.RHF().run(conv=1.0e-8, max_iter=80, verbose=False)
    td = TDDMRG(
        mf,
        D=args.bond,
        td_bond_dim=args.bond,
        init_guess="hf",
    ).build()
    return mf, td, td._get_td_hamiltonian(), td._default_initial_state()


def _pyqed_mpo_to_tenpy(H, phys_dim, labels):
    length = len(H.factors)
    site = Site(LegCharge.from_trivial(phys_dim), state_labels=labels, sort_charge=False)
    sites = [site] * length
    Ws = [
        npc.Array.from_ndarray_trivial(
            np.asarray(W, dtype=complex),
            labels=["wL", "wR", "p", "p*"],
        )
        for W in H.factors
    ]
    mpo = TenpyMPO(
        sites,
        Ws,
        bc="finite",
        IdL=[0] * (length + 1),
        IdR=[0] * (length + 1),
        mps_unit_cell_width=length,
    )
    model = types.SimpleNamespace(lat=types.SimpleNamespace(bc_MPS="finite"), H_MPO=mpo)
    return model, sites, mpo


def _pyqed_mps_to_tenpy(psi, sites, length):
    Bflat = [np.asarray(B, dtype=complex).transpose(1, 0, 2) for B in psi.factors]
    return TenpyMPS.from_Bflat(
        sites,
        Bflat,
        bc="finite",
        dtype=complex,
        permute=False,
        unit_cell_width=length,
    )


def _run_pyqed(H, psi0, args, *, return_state):
    solver = TDMPS(H, D=args.bond)
    stdout = io.StringIO()
    e_ops = [H] if args.measure_observables else []
    start = time.perf_counter()
    with contextlib.redirect_stdout(stdout):
        solver.run(
            psi0,
            dt=args.dt,
            steps=args.steps,
            e_ops=e_ops,
            integrator=args.integrator,
            krylov_dim=args.krylov_dim,
            krylov_tol=args.krylov_tol,
            krylov_method=args.krylov_method,
            sparse_threshold=args.sparse_threshold,
            sparse_vectorized=not args.no_sparse_vectorized,
            reuse_tdvp_engine=not args.no_reuse_tdvp_engine,
            canonicalize_each_step=args.canonicalize_each_step,
            measure_observables=args.measure_observables,
            track_energy=args.track_energy,
            progress=False,
        )
    elapsed = time.perf_counter() - start
    norm2 = float(np.real(solver.final_state.norm()))
    state = _state_vector(solver.final_state) if return_state else None
    return {
        "time_s": elapsed,
        "state": state,
        "energy": complex(solver.observables[-1, 0]) if args.measure_observables else np.nan,
        "max_chi": int(max(solver.final_state.bond_orders())),
        "norm_error": float(abs(np.sqrt(max(norm2, 0.0)) - 1.0)),
        "truncation_error": float(np.nanmax(solver.tdvp_truncation_errors))
        if solver.tdvp_truncation_errors is not None
        else 0.0,
    }


def _run_pyqed_split(td, psi0, args, *, return_state):
    mol = td.gdvr_mf.mol
    nsites = int(mol.shapes["size"])
    fit_rank = max(1, min(int(args.split_prony_rank), max(1, nsites - 2)))
    svd_rank = max(1, min(int(args.split_svd_rank), max(1, nsites - 1)))
    if args.split_density_method == "exact":
        density_half = GDVRSpatialDensityPhase(mol, 0.5 * args.dt)
    elif args.split_density_method == "prony":
        density_half = GDVRSpatialPronyDensityPhase(
            mol,
            0.5 * args.dt,
            rank=fit_rank,
            residual_rank=args.split_prony_residual_rank,
        )
    elif args.split_density_method == "svd":
        density_half = GDVRSpatialSVDDensityPhase(
            mol,
            0.5 * args.dt,
            rank=svd_rank,
        )
    elif args.split_density_method == "hybrid":
        density_half = GDVRSpatialHybridDensityPhase(
            mol,
            0.5 * args.dt,
            prony_rank=fit_rank,
            residual_rank=args.split_hybrid_residual_rank,
        )
    else:
        raise ValueError("Unsupported split density method.")
    one_body = GDVRSpatialOneBodyRotation(np.asarray(mol.hcore, dtype=complex), args.dt)
    psi = psi0.copy()
    start = time.perf_counter()
    for _ in range(args.steps):
        density_kwargs = {"max_bond": args.bond}
        if args.split_density_method != "exact":
            density_kwargs.update(
                {
                    "integrator": args.integrator,
                    "krylov_dim": args.krylov_dim,
                    "krylov_tol": args.krylov_tol,
                    "krylov_method": args.krylov_method,
                    "sparse_threshold": args.sparse_threshold,
                    "sparse_vectorized": not args.no_sparse_vectorized,
                    "reuse_tdvp_engine": not args.no_reuse_tdvp_engine,
                    "canonicalize_each_step": args.canonicalize_each_step,
                }
            )
        psi = density_half.apply(psi, **density_kwargs)
        psi = psi.compress(args.bond).normalize()
        psi = one_body.apply(psi, max_bond=args.bond)
        psi = psi.compress(args.bond).normalize()
        psi = density_half.apply(psi, **density_kwargs)
        psi = psi.compress(args.bond).normalize()
    elapsed = time.perf_counter() - start
    norm2 = float(np.real(psi.norm()))
    return {
        "time_s": elapsed,
        "state": _state_vector(psi) if return_state else None,
        "energy": np.nan,
        "max_chi": int(max(psi.bond_orders())),
        "norm_error": float(abs(np.sqrt(max(norm2, 0.0)) - 1.0)),
        "truncation_error": 0.0,
        "density_fit": getattr(density_half, "fit_info", None),
    }


def _run_tenpy(H, psi0, args, *, return_state):
    if not HAVE_TENPY:
        raise RuntimeError(f"TeNPy is not available: {TENPY_IMPORT_ERROR!r}")

    length = len(H.factors)
    phys_dim = int(np.asarray(H.factors[0]).shape[2])
    labels = ["empty", "up", "down", "double"] if phys_dim == 4 else [str(i) for i in range(phys_dim)]
    model, sites, mpo = _pyqed_mpo_to_tenpy(H, phys_dim, labels)
    psi = _pyqed_mps_to_tenpy(psi0, sites, length)
    engine_cls = {
        "tdvp": tenpy_tdvp.SingleSiteTDVPEngine,
        "tdvp2": tenpy_tdvp.TwoSiteTDVPEngine,
    }[args.integrator]
    options = {
        "dt": args.dt,
        "N_steps": args.steps,
        "preserve_norm": True,
        "trunc_params": {
            "chi_max": args.bond,
            "svd_min": 1.0e-15,
        },
        "lanczos_params": {
            "N_max": args.krylov_dim,
            "P_tol": args.krylov_tol,
        },
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module=r"tenpy\.")
        start = time.perf_counter()
        engine = engine_cls(psi, model, options)
        engine.run()
        elapsed = time.perf_counter() - start
    state = _tenpy_state_vector(psi, length, phys_dim) if return_state else None
    return {
        "time_s": elapsed,
        "state": state,
        "energy": complex(mpo.expectation_value(psi)),
        "max_chi": int(max(psi.chi)) if psi.chi else 1,
        "norm_error": float(abs(getattr(psi, "norm", np.nan) - 1.0)),
        "truncation_error": str(engine.trunc_err),
    }


def _exact_result(H, psi0, args):
    dim = math.prod(int(np.asarray(W).shape[2]) for W in H.factors)
    if dim > args.exact_max_dim:
        return None
    h_dense = _mpo_to_dense_operator(H)
    vec0 = _state_vector(psi0)
    evolved = expm(-1j * args.dt * args.steps * h_dense) @ vec0
    evolved /= np.linalg.norm(evolved)
    return evolved, np.vdot(evolved, h_dense @ evolved)


def run_case(nz, args):
    mf, td, H, psi0 = _build_h2_gdvr(nz, args)
    dim = math.prod(int(np.asarray(W).shape[2]) for W in H.factors)
    return_state = dim <= args.state_overlap_max_dim
    exact = _exact_result(H, psi0, args)
    pyqed_runs = [_run_pyqed(H, psi0, args, return_state=return_state) for _ in range(args.repeats)]
    tenpy_runs = [_run_tenpy(H, psi0, args, return_state=return_state) for _ in range(args.repeats)]
    split_runs = []
    if args.include_split_diagonal:
        split_runs = [_run_pyqed_split(td, psi0, args, return_state=return_state) for _ in range(args.repeats)]

    pyqed_best = min(pyqed_runs, key=lambda row: row["time_s"])
    tenpy_best = min(tenpy_runs, key=lambda row: row["time_s"])
    if return_state:
        pyqed_tenpy_overlap, pyqed_tenpy_state_error = _state_error(pyqed_best["state"], tenpy_best["state"])
    else:
        pyqed_tenpy_overlap = np.nan
        pyqed_tenpy_state_error = np.nan
    row = {
        "nz": int(nz),
        "sites": len(H.factors),
        "phys_dim": int(np.asarray(H.factors[0]).shape[2]),
        "hilbert_dim": dim,
        "state_overlap_computed": bool(return_state),
        "mpo_max_bond": int(max(np.asarray(W).shape[1] for W in H.factors)),
        "gdvr_terms": int(td._active_integral_build_info["symbolic_terms"]),
        "rhf_energy": float(mf.e_tot),
        "dt": args.dt,
        "steps": args.steps,
        "bond": args.bond,
        "integrator": args.integrator,
        "krylov_dim": args.krylov_dim,
        "krylov_tol": args.krylov_tol,
        "krylov_method": args.krylov_method,
        "pyqed": {**pyqed_best, "times": _summary([run["time_s"] for run in pyqed_runs])},
        "tenpy": {**tenpy_best, "times": _summary([run["time_s"] for run in tenpy_runs])},
        "pyqed_tenpy_overlap": pyqed_tenpy_overlap,
        "pyqed_tenpy_state_error": pyqed_tenpy_state_error,
    }
    if split_runs:
        split_best = min(split_runs, key=lambda row: row["time_s"])
        row["pyqed_split_diagonal"] = {
            **split_best,
            "times": _summary([run["time_s"] for run in split_runs]),
        }
        if return_state:
            row["split_vs_pyqed_overlap"], row["split_vs_pyqed_state_error"] = _state_error(
                pyqed_best["state"],
                split_best["state"],
            )
    if exact is not None:
        exact_state, exact_energy = exact
        row["exact_energy"] = exact_energy
        if return_state:
            row["pyqed_exact_overlap"], row["pyqed_exact_state_error"] = _state_error(
                exact_state,
                pyqed_best["state"],
            )
            row["tenpy_exact_overlap"], row["tenpy_exact_state_error"] = _state_error(
                exact_state,
                tenpy_best["state"],
            )
    return row


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nz", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--lz", type=float, default=4.0)
    parser.add_argument("--h2-bond", type=float, default=1.4)
    parser.add_argument("--bond", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--integrator", choices=("tdvp", "tdvp2"), default="tdvp2")
    parser.add_argument("--krylov-dim", type=int, default=8)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-12)
    parser.add_argument("--krylov-method", choices=("lanczos", "arnoldi"), default="lanczos")
    parser.add_argument("--sparse-threshold", type=float, default=0.0)
    parser.add_argument("--no-sparse-vectorized", action="store_true")
    parser.add_argument("--no-reuse-tdvp-engine", action="store_true")
    parser.add_argument("--canonicalize-each-step", action="store_true")
    parser.add_argument("--measure-observables", action="store_true")
    parser.add_argument("--track-energy", action="store_true")
    parser.add_argument("--include-split-diagonal", action="store_true")
    parser.add_argument("--split-density-method", choices=("exact", "prony", "svd", "hybrid"), default="prony")
    parser.add_argument("--split-prony-rank", type=int, default=8)
    parser.add_argument("--split-prony-residual-rank", type=int, default=0)
    parser.add_argument("--split-svd-rank", type=int, default=8)
    parser.add_argument("--split-hybrid-residual-rank", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--exact-max-dim", type=int, default=2048)
    parser.add_argument("--state-overlap-max-dim", type=int, default=1_500_000)
    parser.add_argument("--out", default="/private/tmp/gdvr_tdvp_vs_tenpy.json")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = [run_case(nz, args) for nz in args.nz]
    payload = {
        "benchmark": "gdvr_tdvp_vs_tenpy_static_h2",
        "tenpy_available": HAVE_TENPY,
        "cases": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")

    print(
        "Nz  MPO_D  pyqed_med_s  tenpy_med_s  split_med_s  speedup(pyqed/tenpy)  "
        "|pyqed-tenpy|  pyqed_exact_err  tenpy_exact_err"
    )
    for row in rows:
        py_med = row["pyqed"]["times"]["median"]
        te_med = row["tenpy"]["times"]["median"]
        split = row.get("pyqed_split_diagonal")
        split_med = split["times"]["median"] if split is not None else np.nan
        ratio = py_med / te_med if te_med else np.nan
        print(
            f"{row['nz']:2d} {row['mpo_max_bond']:6d} {py_med:12.6f} {te_med:12.6f} "
            f"{split_med:12.6f} "
            f"{ratio:20.6f} {row['pyqed_tenpy_state_error']:14.6e} "
            f"{row.get('pyqed_exact_state_error', np.nan):15.6e} "
            f"{row.get('tenpy_exact_state_error', np.nan):15.6e}"
        )
    print(f"Saved JSON: {out}")


if __name__ == "__main__":
    main()
