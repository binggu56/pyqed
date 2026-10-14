"""Full-space TDI BSE using saved PySCF Padé QPs and local RHF/RI data.

Adapts the static singlet BSE of Bruneval, Hamed and Neaton, JCP 142,
244101 (2015), doi:10.1063/1.4922489. Uses all QP gaps in direct-RPA
screening and the full resonant/antiresonant BSE. Local orbitals differ
slightly from the PySCF reference; this is not a bitwise reproduction.
The existing PyQED harmonic-Davidson solver differs from PySCF's solver.
No SCF, integral generation or GW integration is performed.
"""

import os

for name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ["MPLCONFIGDIR"] = "/private/tmp/pyqed-mpl"
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import re
import resource
import time
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
from benchmarks.tdi_window_convergence import input_data, restore_rhf, save_array
from pyqed.qchem import Molecule
from pyqed.gw.gw import GW
from pyqed.gw.bse import BSE
from pyqed.gw.contour import ContourDeformation
from pyqed.gw.screening import StaticScreening


def read_qp_log(path, nmo):
    text = path.read_text()
    begin = re.findall(r"^.*QP_BEGIN orbital=(\d+)/(\d+) mf_Eh=([-\d.]+)$", text, re.M)
    end = re.findall(
        r"^.*QP_END orbital=(\d+)/(\d+) status=completed qp_Eh=([-\d.]+) residual_Eh=([\deE.+-]+)",
        text,
        re.M,
    )
    if len(begin) != nmo or len(end) != nmo:
        raise ValueError("Require exactly one completed QP record per orbital")
    for rows in (begin, end):
        if [int(r[0]) for r in rows] != list(range(1, nmo + 1)) or any(
            int(r[1]) != nmo for r in rows
        ):
            raise ValueError("QP orbital ordering/count mismatch")
    qp = np.array([float(r[2]) for r in end])
    hf = np.array([float(r[2]) for r in begin])
    residual = np.array([float(r[3]) for r in end])
    if not np.isfinite(qp).all() or not np.all(residual <= 1e-6):
        raise ValueError("Invalid QP roots")
    return qp, hf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--qp-log", type=Path, required=True)
    p.add_argument("--pyscf-script", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = dict(
        status="running",
        threads=8,
        nroots=7,
        screening="QP-gap static direct RPA",
        full_space=True,
        tda=False,
        gw_rerun=False,
        scf_rerun=False,
        max_space=192,
        max_cycle=100,
        tolerance=1e-6,
        events=[],
    )

    def event(stage, **data):
        row = dict(stage=stage, elapsed_seconds=time.perf_counter() - started, **data)
        report["events"].append(row)
        report["peak_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        (a.output / "report.json").write_text(
            json.dumps(
                report,
                indent=2,
                default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
            )
        )
        print(json.dumps(row), flush=True)

    checkpoint = a.checkpoint
    source = checkpoint / "input_script_snapshot.py"
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != (checkpoint / "input_sha256.txt").read_text().strip()
    ):
        raise ValueError("Local input provenance mismatch")
    xyz, cfg = input_data(source)
    values = {}
    for node in ast.parse(a.pyscf_script.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "FC_XYZ":
                    values[target.id] = ast.literal_eval(node.value)
    if xyz.strip().splitlines()[2:] != values["FC_XYZ"].strip().splitlines()[2:]:
        raise ValueError("PySCF/local molecular geometry mismatch")
    rows = [r.split() for r in xyz.strip().splitlines()[2:]]
    mol = Molecule(
        atom=[(r[0], list(map(float, r[1:]))) for r in rows],
        basis=cfg["basis"],
        unit="angstrom",
        charge=0,
        spin=0,
    )
    with np.load(checkpoint / "rhf.npz") as saved:
        hf = saved["mo_energy"].copy()
    qp, reference_hf = read_qp_log(a.qp_log, len(hf))
    delta = float(np.max(abs(hf - reference_hf)))
    if delta > 1e-5:
        raise ValueError("Local/PySCF HF orbital energy mismatch exceeds 1e-5 Ha")
    report.update(
        qp_source=str(a.qp_log),
        qp_source_sha256=hashlib.sha256(a.qp_log.read_bytes()).hexdigest(),
        checkpoint=str(checkpoint),
        hf_max_difference_Ha=delta,
        provenance_note="All 798 QPs from PySCF log (12 decimal Ha); local saved RHF orbitals and RI factors",
    )
    np.savez(a.output / "qp_input.npz", qp=qp, hf_local=hf, hf_pyscf=reference_hf)
    event("restore_begin")
    with threadpool_limits(limits=8, user_api="blas"):
        mf = restore_rhf(mol, checkpoint, 6000)
        factors = np.load(checkpoint / "MO_RI_factors.npy", mmap_mode="r")
        gw = GW(mf, freq_int="ac", threads=8, mo_pair_factors=factors)
        gw.e_qp = qp
        signature = json.loads(
            (checkpoint / "screening_cache/reference.json").read_text()
        )
        for key, data in [("energy_sha256", hf), ("coefficient_sha256", mf.mo_coeff)]:
            if (
                signature[key]
                != hashlib.sha256(np.asarray(data, dtype=float).tobytes()).hexdigest()
            ):
                raise ValueError("RI screening cache provenance mismatch")
        no = gw.nocc // 2
        report.update(
            nocc=no, nvir=len(qp) - no, dimension=no * (len(qp) - no), rank=len(factors)
        )
        n = 2 * report["dimension"]
        cap = report["max_space"]
        estimate = 16 * (6 * n * cap + 12 * n * 7 + 12 * cap * cap + 10 * n)
        report["davidson_estimated_workspace_GB"] = estimate / 1e9
        if estimate > mf.max_memory * 1e6:
            raise ValueError("Davidson workspace exceeds the configured memory budget")
        event("restore_complete")
        # Use the existing blocked dielectric with an eight-thread BLAS team.
        gw.threads = 1
        response = ContourDeformation(
            gw,
            qp,
            [no - 1, no],
            nw=8,
            prepare=False,
            transition_factors=np.load(
                checkpoint / "screening_cache/occupied_virtual_factors.npy",
                mmap_mode="r",
            ),
        )
        gw.threads = 8
        identity = dict(qp_sha256=hashlib.sha256(qp.tobytes()).hexdigest(), **signature)
        cache = a.output / "screening_reference.json"
        if cache.exists() and json.loads(cache.read_text()) != identity:
            raise ValueError("Output screening cache does not match input")
        cache.write_text(json.dumps(identity, indent=2))
        chol_path = a.output / "qp_static_cholesky.npy"
        if not chol_path.exists():
            event("qp_static_screening_begin")
            matrix = response.dielectric(0.0, imaginary=True)
            chol = la.cho_factor(
                matrix, lower=True, overwrite_a=True, check_finite=False
            )
            save_array(chol_path, chol[0])
            del matrix, chol
        static = StaticScreening(
            factors, (np.load(chol_path, mmap_mode="r"), True), 6000
        )
        event("qp_static_screening_ready")
        for name, left, right in [
            ("occupied", slice(None, no), slice(None, no)),
            ("ov", slice(None, no), slice(no, None)),
        ]:
            target = a.output / f"qp_screened_{name}.npy"
            if not target.exists():
                event("screened_pairs_begin", block=name)
                transformed = static._transform(left, right)
                save_array(target, transformed)
                del transformed
            setattr(static, "_" + name, np.load(target, mmap_mode="r"))
            event("screened_pairs_ready", block=name)
        bse = BSE(gw, threads=8)
        # Explicit QP-screened reference; never relabel it as HF screening.
        bse._static_screening = static
    module = importlib.import_module("pyqed.gw.bse")
    original = module._bse_full_matmat
    columns = 0

    def timed_action(obj, vectors, batch_columns):
        nonlocal columns
        t = time.perf_counter()
        result = original(obj, vectors, batch_columns)
        columns += vectors.shape[1]
        event(
            "bse_action",
            columns=columns,
            threading_info=dict(obj.threading_info or {}),
            block_columns=vectors.shape[1],
            seconds=time.perf_counter() - t,
        )
        return result

    module._bse_full_matmat = timed_action
    event("bse_begin")
    t = time.perf_counter()
    try:
        with threadpool_limits(limits=1, user_api="blas"):
            bse.run(
                nroots=7,
                eigensolver="davidson",
                max_space=192,
                max_cycle=100,
                tol=1e-6,
                return_info=True,
                return_vectors=True,
            )
        report.update(
            status="complete",
            solve_seconds=time.perf_counter() - t,
            info=bse.info,
            threading_info=bse.threading_info,
            energies_eV=(bse.e * 27.211386245988).tolist(),
        )
        np.savez(a.output / "bse.npz", energies=bse.e, xy=bse.xy, qp=qp)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        reference = [
            2.554626533,
            2.659181224,
            3.822950365,
            3.856573874,
            4.150815852,
            4.150816949,
            4.175459452,
        ]
        axes[0].plot(
            range(1, 8),
            report["energies_eV"],
            "o-",
            label="PyQED; saved PySCF Padé QPs",
        )
        axes[0].plot(range(1, 8), reference, "x--", label="PySCF supplied log")
        axes[0].set(xlabel="Root", ylabel="Energy (eV)")
        axes[0].legend(fontsize=8)
        axes[1].semilogy(range(1, 8), bse.info["residual_norms"], "o-")
        axes[1].axhline(1e-6, color="gray", linestyle="--")
        axes[1].set(xlabel="Root", ylabel="Residual norm (Ha)")
        fig.suptitle("TDI full-space singlet BSE, QP-gap screening, 8 threads")
        fig.savefig(a.output / "TDI_full_BSE_Pade.png", dpi=160)
        event("complete")
    except Exception as exc:
        report.update(status="failed", error=repr(exc), info=bse.info)
        event("failed")
        raise
    finally:
        module._bse_full_matmat = original


if __name__ == "__main__":
    main()
