"""Local selected-QP contour pilot; complete RI screening, coarse nw=8.
Uses PyQED's adapted inverse-Casida Lanczos response reuse by default;
see LanczosResponse for references, error bounds and near-pole limitations.
BSE operator timings use HF gaps outside the selected QP set, not physical
BSE excitation results. Re-run with --nw 64 for the original quadrature.
"""

import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[key] = "1"
from pathlib import Path

import argparse, json, time, contextlib, resource, importlib, hashlib, logging
from benchmarks.tdi_window_convergence import input_data, save_array, restore_rhf
import numpy as np
from threadpoolctl import threadpool_limits
from pyqed.qchem import Molecule
from pyqed.qchem.hf.rhf import RHF
from pyqed.gw.gw import GW
from pyqed.gw.bse import BSE, _bse_tda_matmat, _bse_full_matmat
import scipy.linalg as la

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input-script", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--preparation-threads", type=int, default=8)
p.add_argument("--screening-blas-threads", type=int, default=8)
p.add_argument("--response-cache-mb", type=float, default=1000)
p.add_argument("--nw", type=int, default=8)
p.add_argument("--max-memory-mb", type=float, default=8000)
p.add_argument("--orbs", type=int, nargs="+")
p.add_argument("--full-gw", action="store_true")
p.add_argument("--resume", action="store_true")
p.add_argument("--residue-solver", choices=["gmres", "lanczos"], default="lanczos")
a = p.parse_args()
if (
    min(a.threads, a.preparation_threads, a.screening_blas_threads) < 1
    or a.nw < 8
    or min(a.max_memory_mb, a.response_cache_mb) <= 0
):
    p.error("Require positive threads/memory and nw >= 8")
ROOT = a.output.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
if (ROOT / "report.json").exists() and not a.resume:
    raise FileExistsError("Use a new output directory or --resume")
(ROOT / "scratch").mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(ROOT / "scratch")
os.environ["MPLCONFIGDIR"] = str(ROOT / "scratch" / "mpl")
import tempfile

tempfile.tempdir = str(ROOT / "scratch")
logging.basicConfig(level=logging.INFO)
source = a.input_script.resolve()
xyz, config = input_data(source)
settings = dict(FC_XYZ=xyz, DEFAULT_CONFIG=config)
signature = hashlib.sha256(source.read_bytes()).hexdigest()
provenance = ROOT / "input_sha256.txt"
if a.resume and (
    not provenance.exists() or provenance.read_text().strip() != signature
):
    raise ValueError("Resume requires matching input provenance")
provenance.write_text(signature + "\n")
(ROOT / "input_script_snapshot.py").write_bytes(source.read_bytes())
(ROOT / "input.xyz").write_text(settings["FC_XYZ"])
(ROOT / "pilot.pid").write_text(str(os.getpid()))
report = dict(
    settings=dict(
        settings["DEFAULT_CONFIG"],
        nw=a.nw,
        max_memory_mb=a.max_memory_mb,
        threads=a.threads,
        preparation_threads=a.preparation_threads,
        screening_blas_threads=a.screening_blas_threads,
        response_cache_mb=a.response_cache_mb,
        residue_solver=a.residue_solver,
        screening_tol=1e-9,
    ),
    stages={},
    calls={},
    status="running",
)


def emit(event, **kw):
    row = dict(event=event, time=time.time(), **kw)
    print(
        json.dumps(row, default=lambda x: x.item() if hasattr(x, "item") else str(x)),
        flush=True,
    )
    with (ROOT / "events.jsonl").open("a") as f:
        f.write(
            json.dumps(
                row, default=lambda x: x.item() if hasattr(x, "item") else str(x)
            )
            + "\n"
        )
    (ROOT / "report.json").write_text(
        json.dumps(
            report,
            indent=2,
            default=lambda x: x.item() if hasattr(x, "item") else str(x),
        )
    )


@contextlib.contextmanager
def stage(name):
    report["stage"] = name
    emit("stage_start", stage=name)
    t = time.perf_counter()
    try:
        preparation = name in (
            "RI build",
            "RHF",
            "AO to MO",
            "BSE initialization and static caches",
        )
        with threadpool_limits(
            limits=a.preparation_threads if preparation else 1, user_api="blas"
        ):
            yield
    finally:
        report["stages"][name] = time.perf_counter() - t
        emit(
            "stage_end",
            stage=name,
            seconds=report["stages"][name],
            peak_rss_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
        )


def instrument(cls, name):
    fn = getattr(cls, name)

    def wrapped(self, *args, **kwargs):
        t = time.perf_counter()
        try:
            if name == "screened_elements":
                with threadpool_limits(
                    limits=a.screening_blas_threads, user_api="blas"
                ):
                    return fn(self, *args, **kwargs)
            return fn(self, *args, **kwargs)
        finally:
            dt = time.perf_counter() - t
            row = report["calls"].setdefault(name, [])
            if len(row) < 20000:
                row.append(dt)
            stats = report.setdefault("call_stats", {}).setdefault(
                name, dict(count=0, seconds=0.0)
            )
            stats["count"] += 1
            stats["seconds"] += dt
            if name in ("dielectric", "screened_elements"):
                emit("kernel", kernel=name, seconds=dt, count=len(row))

    setattr(cls, name, wrapped)


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    names = list(report["stages"])
    axs[0].barh(names, [report["stages"][k] for k in names])
    axs[0].set_xlabel("Measured wall time (s)")
    axs[0].set_xscale("log")
    if "qp" in report:
        q = report["qp"]
        axs[1].plot(
            q["orbitals"], np.array(q["energies"]) - np.array(q["hf_energies"]), "o-"
        )
        axs[1].set(
            xlabel="Spatial orbital (zero based)", ylabel="QP minus HF energy (Ha)"
        )
    fig.suptitle("TDI NCH3 FC / 6-31G: local contour GW pilot, nw=" + str(a.nw))
    fig.savefig(ROOT / "TDI_NCH3_FC_contour_pilot.png", dpi=170)


try:
    with threadpool_limits(limits=1):
        rows = [x.split() for x in settings["FC_XYZ"].strip().splitlines()[2:]]
        if a.resume:
            mol = Molecule(
                atom=[(r[0], list(map(float, r[1:]))) for r in rows],
                basis=config["basis"],
                unit="angstrom",
                charge=0,
                spin=0,
            )
            mol.max_memory = a.max_memory_mb
            with stage("Restore RHF"):
                mf = restore_rhf(mol, ROOT, a.max_memory_mb)
        else:
            with stage("RI build"):
                mol = Molecule(
                    atom=[(r[0], list(map(float, r[1:]))) for r in rows],
                    basis=config["basis"],
                    unit="angstrom",
                    charge=0,
                    spin=0,
                )
                mol.max_memory = a.max_memory_mb
                cfg = settings["DEFAULT_CONFIG"]
                keys = (
                    "auxbasis",
                    "ri_metric_tol",
                    "ri_metric_solver",
                    "ri_purpose",
                    "ri_storage",
                    "ri_tensor_backend",
                    "ri_screen_tol",
                    "ri_cache",
                    "one_electron_cache",
                )
                mol.workers = a.threads
                mol.build(
                    eri="ri", options={k: cfg[k] for k in keys}, workers=a.threads
                )
                save_array(ROOT / "AO_RI_factors.npy", mol.eri_factors.pair_factors)
            with stage("RHF"):
                mf = RHF(mol)
                mf.max_memory = a.max_memory_mb
                mf.run(verbose=2, tol=1e-8, max_cycle=100)
                np.savez(
                    ROOT / "rhf.npz",
                    mo_coeff=mf.mo_coeff,
                    mo_occ=mf.mo_occ,
                    mo_energy=mf.mo_energy,
                    e_hf=mf.e_tot,
                    converged=mf.converged,
                    ao_overlap=mf.get_ovlp(),
                    hcore=mf.get_hcore(),
                )
                if not mf.converged:
                    raise RuntimeError("RHF did not converge")
        with stage("AO to MO"):
            saved_mo = ROOT / "MO_RI_factors.npy"
            reuse = (
                np.load(saved_mo, mmap_mode="r")
                if a.resume and saved_mo.exists()
                else None
            )
            gw = GW(
                mf,
                screening="TDH",
                eta=config["eta"],
                freq_int="contour",
                threads=a.threads,
                mo_pair_factors=reuse,
            )
            if reuse is None:
                save_array(saved_mo, gw._pair_factors)
                gw._pair_factors = np.load(saved_mo, mmap_mode="r")
            report.update(
                nmo=len(mf.mo_energy),
                nocc=int(mol.nelec) // 2,
                rank=len(gw._pair_factors),
                mo_disk_backed=isinstance(gw._pair_factors, np.memmap),
            )
        c = importlib.import_module("pyqed.gw.contour")
        for name in ("dielectric", "screened_elements", "evaluate", "residue"):
            instrument(c.ContourDeformation, name)
        occ = int(mol.nelec) // 2
        orbs = (
            list(range(len(mf.mo_energy)))
            if a.full_gw
            else a.orbs
            or sorted(
                set(
                    [
                        max(0, occ - 21),
                        occ - 1,
                        occ,
                        min(len(mf.mo_energy) - 1, occ + 20),
                    ]
                )
            )
        )
        controls = dict(
            nw=a.nw,
            quadrature_scale=0.5,
            residue_solver=a.residue_solver,
            screening_tol=1e-9,
            screening_maxiter=300,
        )
        folder = ROOT / "screening_cache"
        folder.mkdir(exist_ok=True)
        signature = dict(
            energy_sha256=hashlib.sha256(mf.mo_energy.tobytes()).hexdigest(),
            coefficient_sha256=hashlib.sha256(mf.mo_coeff.tobytes()).hexdigest(),
            eta=gw.eta,
            nw=a.nw,
            quadrature_scale=0.5,
        )
        metadata = folder / "reference.json"
        if metadata.exists() and json.loads(metadata.read_text()) != signature:
            raise ValueError("Screening cache differs from this RHF reference/grid")
        metadata.write_text(json.dumps(signature, indent=2))
        key = hashlib.sha256(np.asarray(orbs, dtype=np.int64).tobytes()).hexdigest()[
            :16
        ]
        with stage("imaginary screening"):
            pair_path = folder / "occupied_virtual_factors.npy"
            pairs = np.load(pair_path, mmap_mode="r") if pair_path.exists() else None
            contour = c.ContourDeformation(
                gw,
                mf.mo_energy,
                orbs,
                prepare=False,
                transition_factors=pairs,
                **controls,
            )
            contour.response_cache_bytes = int(a.response_cache_mb * 1e6)
            if pairs is None:
                save_array(pair_path, contour.pairs)
                contour.pairs = np.load(pair_path, mmap_mode="r")
            target_path = folder / f"QP_factors_{key}.npy"
            if not target_path.exists():
                targets = np.lib.format.open_memmap(
                    target_path.with_suffix(".tmp"),
                    mode="w+",
                    dtype=float,
                    shape=(len(contour.factors), len(orbs), len(mf.mo_energy)),
                )
                for first in range(0, len(contour.factors), 64):
                    targets[first : first + 64] = contour.factors[first : first + 64][
                        :, orbs, :
                    ]
                targets.flush()
                del targets
                target_path.with_suffix(".tmp").replace(target_path)
            contour.target_factors = np.load(target_path, mmap_mode="r")
            for index, frequency in enumerate(np.r_[0.0, contour.freqs]):
                cp, ep = (
                    folder / f"cholesky_{index}.npy",
                    folder / f"elements_{index}_{key}.npy",
                )
                if cp.exists():
                    chol = (np.load(cp, mmap_mode="r"), True)
                else:
                    chol = la.cho_factor(
                        contour.dielectric(frequency, imaginary=True),
                        lower=True,
                        overwrite_a=True,
                        check_finite=False,
                    )
                    save_array(cp, chol[0])
                elements = (
                    np.load(ep)
                    if ep.exists()
                    else contour.screened_elements(chol, orbs)
                )
                if not ep.exists():
                    save_array(ep, elements)
                contour.set_screening_frequency(
                    index,
                    elements,
                    chol if index == 0 else None,
                    max_memory=a.max_memory_mb,
                )
                emit("screening_frequency_complete", index=index)
            gw._contour = contour
        report["qp_timings"] = []
        with stage("selected contour GW"):
            for orbital in sorted(orbs, key=lambda p: abs(p - (occ - 0.5))):
                emit("QP_start", orbital=int(orbital))
                before = time.perf_counter()
                gw.run(orbs=[int(orbital)], reuse_screening=True, tol=1e-9, **controls)
                report["qp_timings"].append(
                    dict(orbital=int(orbital), seconds=time.perf_counter() - before)
                )
                np.savez(
                    ROOT / "gw_partial.npz",
                    e_qp=gw.e_qp,
                    weights=gw.qp_weights,
                    residuals=gw.qp_residuals,
                )
                emit("QP_complete", **report["qp_timings"][-1])
            report["qp"] = dict(
                orbitals=orbs,
                energies=gw.e_qp[orbs].tolist(),
                hf_energies=mf.mo_energy[orbs].tolist(),
                weights=gw.qp_weights[orbs].tolist(),
                residuals=gw.qp_residuals[orbs].tolist(),
            )
            report["screening_info"] = gw._contour.screening_info
            report["threading_info"] = gw.threading_info
            np.savez(
                ROOT / "gw_selected.npz",
                e_qp=gw.e_qp,
                weights=gw.qp_weights,
                residuals=gw.qp_residuals,
                orbitals=orbs,
            )
        with stage("BSE initialization and static caches"):
            # Timing proxy only: uncomputed QP gaps use the HF reference.
            gw.e_qp = np.where(np.isfinite(gw.e_qp), gw.e_qp, mf.mo_energy)
            bse = BSE(gw, threads=a.threads)
            bse._M = bse._static_screening
            bse._static_screening.static_occupied(occ)
            bse._static_screening.static_ov(occ)
        rng = np.random.default_rng(20261013)
        dim = occ * (len(mf.mo_energy) - occ)
        report["bse_action_timings"] = {}
        for label, fn, size in [
            ("TDA", _bse_tda_matmat, dim),
            ("Full BSE", _bse_full_matmat, 2 * dim),
        ]:
            for width in (1, 8):
                with stage(label + " action batch " + str(width)):
                    vectors = rng.standard_normal((size, width))
                    t = time.perf_counter()
                    out = fn(bse, vectors, width)
                    dt = time.perf_counter() - t
                    if not np.all(np.isfinite(out)):
                        raise RuntimeError("Nonfinite BSE action")
                    report["bse_action_timings"][label + " batch " + str(width)] = dt
                    del out, vectors
        report["status"] = "complete"
        plot()
        emit("complete")
except BaseException as exc:
    report.update(status="failed", error=repr(exc))
    plot()
    emit("failed", error=repr(exc))
    raise
