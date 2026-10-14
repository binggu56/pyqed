"""Eight-thread molecular AC benchmark reusing verified TDI RHF/RI checkpoints."""

import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from benchmarks.tdi_window_convergence import input_data, restore_rhf
from pyqed.qchem import Molecule
from pyqed.gw.gw import GW

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--nw", type=int, default=100)
p.add_argument("--ac", choices=("pade", "twopole"), default="pade")
p.add_argument("--pade-npts", type=int, default=18)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--orbs", type=int, nargs="+")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
if (a.output / "report.json").exists():
    raise FileExistsError("Choose a fresh output directory")
os.environ["MPLCONFIGDIR"] = "/private/tmp/pyqed-mpl"
source = a.checkpoint / "input_script_snapshot.py"
if (
    hashlib.sha256(source.read_bytes()).hexdigest()
    != (a.checkpoint / "input_sha256.txt").read_text().strip()
):
    raise ValueError("Input checkpoint provenance mismatch")
xyz, cfg = input_data(source)
rows = [row.split() for row in xyz.strip().splitlines()[2:]]
mol = Molecule(
    atom=[(r[0], list(map(float, r[1:]))) for r in rows],
    basis=cfg["basis"],
    unit="angstrom",
    charge=0,
    spin=0,
)
report = dict(
    status="running",
    method=f"molecular G0W0 {a.ac} AC",
    threads=a.threads,
    nw=a.nw,
    checkpoint=str(a.checkpoint),
    input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
)
started = time.perf_counter()


def save():
    report["elapsed_seconds"] = time.perf_counter() - started
    report["peak_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    (a.output / "report.json").write_text(json.dumps(report, indent=2))


def progress(gw, index):
    report["ac_info"] = gw.ac_info
    report["completed_frequencies"] = index + 1
    np.savez(
        a.output / "sigma_checkpoint.npz",
        sigma=gw.ac_sigma,
        omega=gw.ac_omega,
        mo_energy=mf.mo_energy,
        completed=index + 1,
    )
    save()
    print(
        json.dumps(
            dict(
                frequency=index + 1,
                total=a.nw,
                elapsed_seconds=report["elapsed_seconds"],
                timing=gw.ac_info["frequencies"][-1],
            )
        ),
        flush=True,
    )


with threadpool_limits(limits=a.threads, user_api="blas"):
    t = time.perf_counter()
    mf = restore_rhf(mol, a.checkpoint, 4000)
    factors = np.load(a.checkpoint / "MO_RI_factors.npy", mmap_mode="r")
    gw = GW(
        mf, freq_int="ac", screening="TDH", threads=a.threads, mo_pair_factors=factors
    )
    report["restore_and_gw_setup_seconds"] = time.perf_counter() - t
    report["threadpools"] = threadpool_info()
    report.update(nmo=len(mf.mo_energy), nocc=int(mol.nelec) // 2, rank=len(factors))
    print(
        json.dumps({k: v for k, v in report.items() if k != "threadpools"}), flush=True
    )
    save()
    t = time.perf_counter()
    try:
        gw.run(
            nw=a.nw,
            ac=a.ac,
            pade_npts=a.pade_npts,
            orbs=a.orbs,
            transition_factors=np.load(
                a.checkpoint / "screening_cache/occupied_virtual_factors.npy",
                mmap_mode="r",
            ),
            callback=progress,
        )
        report["status"] = "complete"
        report["qp_orbitals"] = gw.orbital_indices.tolist()
        report["screening_orbitals"] = len(mf.mo_energy)
        report["qp_gap_eV"] = float(
            (gw.e_qp[gw.nocc // 2] - gw.e_qp[gw.nocc // 2 - 1]) * 27.211386245988
        )
        report["qp_residual_max_Ha"] = float(np.nanmax(gw.qp_residuals))
        report["qp_weight_range"] = [
            float(np.nanmin(gw.qp_weights)),
            float(np.nanmax(gw.qp_weights)),
        ]
        np.savez(
            a.output / "qp.npz",
            hf=mf.mo_energy,
            qp=gw.e_qp,
            weights=gw.qp_weights,
            residuals=gw.qp_residuals,
            coefficients=gw.ac_coeff,
            sigma=gw.ac_sigma,
            omega=gw.ac_omega,
        )
    except BaseException as exc:
        report["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        report["error"] = repr(exc)
        raise
    finally:
        report["gw_seconds"] = time.perf_counter() - t
        report["ac_info"] = gw.ac_info
        save()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        times = gw.ac_info["frequencies"]
        for key in ("dielectric_seconds", "inverse_seconds", "projection_seconds"):
            axs[0].plot(
                np.arange(len(times)) + 1,
                [r[key] for r in times],
                label=key.replace("_seconds", ""),
            )
        axs[0].set(xlabel="Quadrature point", ylabel="Wall time (s)")
        axs[0].legend()
        if gw.e_qp is not None:
            axs[1].plot(
                mf.mo_energy * 27.211386245988,
                (gw.e_qp - mf.mo_energy) * 27.211386245988,
                ".",
            )
            axs[1].set(xlabel="HF energy (eV)", ylabel="AC QP correction (eV)")
        fig.suptitle(
            f"TDI / 6-31G: {a.ac} AC, {a.nw} points, {a.threads} threads\n"
            f"{len(a.orbs) if a.orbs else len(mf.mo_energy)} QP targets; all orbitals retained in screening"
        )
        fig.savefig(a.output / "TDI_AC_benchmark.png", dpi=160)
        print(json.dumps(report), flush=True)
