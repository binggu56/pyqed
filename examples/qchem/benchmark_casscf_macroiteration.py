"""Run one real SA-CASSCF macroiteration and preserve resource diagnostics.

Default: benzene/aug-cc-pVDZ, six singlets, CAS(8,8), frontier active orbitals.
This measures the optimizer; the resulting iterate is not a converged
spectroscopic model. Run through monitor_local_calculation.py for RSS sampling.
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
os.environ["PYQED_LIGHTWEIGHT_IMPORTS"] = "1"

import argparse
from contextlib import contextmanager
import ctypes
import functools
import hashlib
import json
import logging
from pathlib import Path
import resource
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
from pyqed import num_workers
from pyqed.qchem import Molecule
from pyqed.qchem.hf.rhf import RHF
from pyqed.qchem.mcscf import casscf
from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed.qchem.mcscf.ri_orbitals import OrbitalRIFactors
from examples.qchem.validate_molecular_symmetry import geometry


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
        )
        + "\n"
    )
    temporary.replace(path)


def disk_counters():
    """macOS process disk counters from the SDK's rusage_info_v2 layout."""
    if sys.platform != "darwin":
        return {}
    fields = (
        "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size "
        "resident_size phys_footprint proc_start_abstime proc_exit_abstime "
        "child_user_time child_system_time child_pkg_idle_wkups child_interrupt_wkups "
        "child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten"
    ).split()

    class Usage(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
            (name, ctypes.c_uint64) for name in fields
        ]

    usage = Usage()
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    library.proc_pid_rusage.restype = ctypes.c_int
    if library.proc_pid_rusage(os.getpid(), 2, ctypes.byref(usage)) != 0:
        return dict(unavailable_errno=ctypes.get_errno())
    return dict(
        read_bytes=usage.diskio_bytesread,
        written_bytes=usage.diskio_byteswritten,
        pageins=usage.pageins,
    )


class MacroComplete(Exception):
    pass


def plot(out):
    os.environ.setdefault("MPLCONFIGDIR", str(out / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((out / "report.json").read_text())
    history = json.loads((out / "casscf_micro_history.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    phases = report["phases"]
    axes[0].bar(list(phases), [r["seconds"] for r in phases.values()], color="#0284c7")
    axes[0].set(ylabel="Wall time (s)", title="Measured stages")
    axes[1].plot([r["micro"] for r in history], [r["energy"] for r in history], "o-")
    axes[1].set(xlabel="Microiteration", ylabel="State-average energy (Ha)")
    axes[1].ticklabel_format(useOffset=False, axis="y")
    axes[2].semilogy(
        [r["micro"] for r in history], [r["gradient_norm"] for r in history], "o-"
    )
    axes[2].set(xlabel="Microiteration", ylabel="Orbital gradient norm")
    fig.suptitle(report["description"] + " — first macroiteration, not converged")
    fig.savefig(out / "casscf_macroiteration_diagnostics.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.plot_only:
        plot(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out / "provenance.json").exists():
        raise FileExistsError("Use a fresh benchmark directory")
    scratch = out / "scratch"
    scratch.mkdir()
    os.environ["PYQED_SCRATCH_DIR"] = str(scratch)
    os.environ.setdefault("MPLCONFIGDIR", str(out / "matplotlib"))
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    atom, unit, *_ = geometry("benzene")
    basis, auxbasis, ncas, nelecas, roots = "aug-cc-pvdz", "aug-cc-pvdz-rifit", 8, 8, 6
    if args.smoke:
        atom, unit = "Li 0 0 0; H 0 0 1.6", "angstrom"
        basis, auxbasis, ncas, nelecas, roots = (
            "sto-3g",
            "def2-universal-jkfit",
            2,
            2,
            2,
        )
    description = (
        f'{"LiH" if args.smoke else "Benzene"}/{basis} SA{roots}-CAS({nelecas},{ncas})'
    )
    files = [
        Path(__file__),
        Path(casscf.__file__),
        Path(sys.modules[OrbitalRIFactors.__module__].__file__),
        Path(sys.modules[CASCI.__module__].__file__),
        Path(sys.modules[RHF.__module__].__file__),
    ]
    save_json(
        out / "provenance.json",
        dict(
            description=description,
            geometry=atom,
            unit=unit,
            basis=basis,
            auxbasis=auxbasis,
            ncas=ncas,
            nelecas=nelecas,
            roots=roots,
            workers=num_workers(),
            blas_threads=1,
            python=sys.executable,
            sources={
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
            figure_title=description + " — one macroiteration",
            memory_figure="casscf_macroiteration_memory.png",
        ),
    )
    phases, profiles = {}, {}
    started, io_start = time.perf_counter(), disk_counters()

    @contextmanager
    def stage(name):
        before, io_before = time.perf_counter(), disk_counters()
        save_json(out / "stage.json", dict(stage=name, status="running"))
        print(name + "_BEGIN", flush=True)
        try:
            yield
        finally:
            phases[name] = dict(
                seconds=time.perf_counter() - before,
                io_before=io_before,
                io_after=disk_counters(),
            )
            save_json(out / "phases.json", phases)
            print(name + "_END " + json.dumps(phases[name]), flush=True)

    def record(name, seconds):
        row = profiles.setdefault(name, dict(calls=0, seconds=0.0))
        row["calls"] += 1
        row["seconds"] += seconds
        save_json(out / "profile.json", profiles)

    saved = {}
    for name in ("active_integrals", "generalized_fock", "hessian_action"):
        original = getattr(OrbitalRIFactors, name)
        saved[name] = original

        def instrument(function, label):
            @functools.wraps(function)
            def wrapped(*a, **kw):
                before = time.perf_counter()
                try:
                    return function(*a, **kw)
                finally:
                    record(label, time.perf_counter() - before)

            return wrapped

        setattr(OrbitalRIFactors, name, instrument(original, name))

    class SingletCASCI(CASCI):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.direct_ci_auto_spin0 = False
            self.spin_root_cushion = 24
            self.spin_selection_tol = 1e-7
            self.direct_ci_eigensolver = "davidson"
            self.direct_ci_dense_fallback_ndets = 0
            self.direct_ci_max_cycle = 200
            self.direct_ci_max_subspace = 160
            self.direct_ci_residual_tol = 1e-9
            self.direct_ci_workers = num_workers()

        def run(self, *a, **kw):
            before = time.perf_counter()
            try:
                result = super().run(*a, **kw)
                spins = [self.spin_square(i) for i in range(len(self.ci))]
                if (
                    not self.converged
                    or len(spins) != roots
                    or np.max(np.abs(spins)) > 1e-6
                ):
                    raise RuntimeError("CI roots are not converged singlets")
                return result
            finally:
                record("ci_solve_including_integrals", time.perf_counter() - before)

    original_ci, mc = casscf.CASCI, None
    casscf.CASCI = SingletCASCI
    try:
        with threadpool_limits(1):
            mol = Molecule(atom=atom, unit=unit, basis=basis, charge=0, spin=0)
            with stage("RI"):
                mol.build(
                    eri="ri",
                    auxbasis=auxbasis,
                    workers=num_workers(),
                    options=dict(
                        ri_storage="packed",
                        ri_cache=True,
                        ri_cache_dir=str(out / "ri_cache"),
                    ),
                )
                save_json(
                    out / "integrals.json",
                    dict(
                        nao=mol.nao,
                        naux=len(mol.eri_factors),
                        packed_bytes=mol.eri_factors.pair_factors.nbytes,
                        build_info=mol._builtin_build_info,
                    ),
                )
            mf = RHF(mol)
            mf.chkfile = str(out / "rhf.chk")
            with stage("RHF"):
                mf.run(tol=1e-10, max_cycle=150, verbose=2)
                assert mf.converged
                mf.save(str(out / "rhf.npz"))
            ncore = (mol.nelec - nelecas) // 2
            active = list(range(ncore, ncore + ncas))
            assert np.sum(mf.mo_occ[active]) == nelecas
            mc = casscf.SecondOrderCASSCF(
                mf,
                ncas=ncas,
                nelecas=nelecas,
                ms2=0,
                multiplicity=1,
                max_cycle=1,
                max_micro_cycle=1 if args.smoke else 8,
                conv_tol=1e-8,
                conv_tol_grad=1e-4,
                conv_tol_grad_relaxed=1e-4,
                conv_tol_step=1e-4,
                max_step=0.1,
                step_size=0.25,
                level_shift=0.0005,
                optimizer="AH",
                coupling="qn",
                ah_hessian="analytic",
                micro_ci_mode="full",
                ah_pspace_size=0,
                ah_max_cycle=30,
                ah_max_subspace=40,
                auto_active_restarts=False,
                diis=False,
                ci_method="direct_ci",
                use_cholesky=True,
                trial_memory_mb=384,
                verbose=1,
            )
            mc.state_average([1 / roots] * roots)
            initial = mc.reorder_mo_for_active_orbitals(mf.mo_coeff, active)
            summary = {}

            def checkpoint(state):
                ci = state["casci"]
                spins = np.asarray([ci.spin_square(i) for i in range(roots)])
                coeff, vectors = state["mo_coeff"], np.asarray(ci.ci)
                overlap = mf.get_ovlp()
                singular = np.linalg.svd(
                    initial[:, active].T @ overlap @ coeff[:, active], compute_uv=False
                )
                np.testing.assert_allclose(
                    coeff.T @ overlap @ coeff, np.eye(coeff.shape[1]), atol=1e-7
                )
                np.testing.assert_allclose(
                    vectors @ vectors.T, np.eye(roots), atol=1e-7
                )
                if not ci.converged or np.max(abs(spins)) > 1e-6 or min(singular) < 0.5:
                    raise RuntimeError(
                        "Macroiterate CI/spin/active-overlap checks failed"
                    )
                summary.update(
                    energies_Ha=ci.e_tot,
                    spin_square=spins,
                    active_overlap=singular,
                    diagnostics=state["diagnostics"],
                    ci_converged=True,
                    casscf_converged=False,
                    initial_active_mos_1based=np.asarray(active) + 1,
                )
                np.savez(
                    out / "one_macroiteration.npz",
                    mo_coeff=coeff,
                    ci=vectors,
                    energies=ci.e_tot,
                    spin_square=spins,
                    active_overlap=singular,
                    initial_mo_coeff=initial,
                    ao_overlap=overlap,
                )
                raise MacroComplete

            with stage("CASSCF"):
                try:
                    mc.run(
                        nstates=roots,
                        mo_coeff=mf.mo_coeff,
                        active_orbitals=active,
                        macro_callback=checkpoint,
                    )
                except MacroComplete:
                    pass
            if not summary:
                raise RuntimeError("No macroiteration checkpoint was produced")
            report = dict(
                description=description,
                status="complete",
                benchmark_only=True,
                elapsed_s=time.perf_counter() - started,
                phases=phases,
                profile=profiles,
                os_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (2**30 if sys.platform == "darwin" else 2**20),
                io_before=io_start,
                io_after=disk_counters(),
                **summary,
            )
            save_json(out / "report.json", report)
            save_json(out / "stage.json", dict(stage="DONE", status="complete"))
    finally:
        casscf.CASCI = original_ci
        for name, original in saved.items():
            setattr(OrbitalRIFactors, name, original)
        if mc is not None:
            save_json(out / "casscf_history.json", mc.history)
            save_json(out / "casscf_micro_history.json", mc.micro_history)
    plot(out)
    print("BENCHMARK_COMPLETE " + str(out), flush=True)


if __name__ == "__main__":
    main()
