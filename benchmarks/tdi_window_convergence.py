"""TDI FC restricted QP/BSE windows with complete RI direct-RPA screening.

The input Python file is parsed as data, never imported or executed. All
screening/intermediate orbitals, basis, quadrature, eta and root tolerances
are retained. Only the neutral excitation space is projected onto frontier
windows; compare successive energies and seven-state subspaces. This adapts
the molecular static BSE workflow of Bruneval, Hamed and Neaton, JCP 142,
244101 (2015), doi:10.1063/1.4922489; window stability is a convergence
diagnostic, not a rigorous complete-space error bound. Outputs stay outside
the repository. Use PYTHONPATH=. and single BLAS/OpenMP threads.
"""

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback


from pyqed.parallel import num_workers
def save_array(path, array):
    import numpy as np
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array)
    temporary.replace(path)


def restore_rhf(mol, out, max_memory):
    """Restore the native RHF reference and packed read-only AO RI factors."""
    import numpy as np
    from pyqed.qchem.hf.rhf import RHF
    from pyqed.qchem.basis import PackedRIFactors
    from pyqed.gw.gw import _make_rhf_dm
    with np.load(out/'rhf.npz', allow_pickle=False) as saved:
        if not bool(saved['converged']):
            raise ValueError('Saved RHF is not converged')
        mol.nao = mol.nmo = saved['mo_coeff'].shape[0]
        mol._overlap, mol.hcore = saved['ao_overlap'].copy(), saved['hcore'].copy()
        mol.eri_factors = PackedRIFactors(np.load(out/'AO_RI_factors.npy', mmap_mode='r'), mol.nao)
        mf = RHF(mol)
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = [saved[k].copy() for k in ['mo_coeff','mo_occ','mo_energy']]
        mf.e_tot, mf.converged = float(saved['e_hf']), True
        mf.hcore, mf.eri_factors = mol.hcore, mol.eri_factors
        mf.dm = _make_rhf_dm(mf.mo_coeff, mf.nocc)
        mf.max_memory = max_memory
        return mf


def parallel_qp_orbitals(gw, orbs, controls, tol, workers):
    # Keep numerical imports after the driver's environment configuration.
    from pyqed.gw.contour import parallel_qp_orbitals as solve
    return solve(gw, orbs, controls, tol, workers)


def initialize_screening_worker(folder, energy, gaps, eta, block, orbs, target_key):
    import numpy as np
    from pyqed.gw.contour import ContourDeformation
    global frequency_worker, frequency_folder, frequency_target_key
    frequency_folder = Path(folder)
    frequency_worker = object.__new__(ContourDeformation)
    frequency_worker.factors = np.load(frequency_folder/"MO_RI_factors.npy", mmap_mode="r")
    frequency_worker.pairs = np.load(frequency_folder/"occupied_virtual_factors.npy", mmap_mode="r")
    frequency_worker.energy, frequency_worker.gaps = energy, gaps
    frequency_worker.eta, frequency_worker.block, frequency_worker.orbs = eta, block, orbs
    frequency_worker.columns = {int(p): j for j, p in enumerate(orbs)}
    frequency_worker.target_factors = np.load(frequency_folder/f"QP_factors_{target_key}.npy", mmap_mode="r")
    frequency_target_key = target_key


def screening_frequency(task):
    import numpy as np
    import scipy.linalg as la
    index, frequency = task
    start = time.perf_counter()
    factor_path = frequency_folder/f"cholesky_{index}.npy"
    if factor_path.exists():
        chol = (np.load(factor_path, mmap_mode="r"), True)
    else:
        chol = la.cho_factor(frequency_worker.dielectric(frequency, imaginary=True),
                             lower=True, overwrite_a=True, check_finite=False)
        save_array(factor_path, chol[0])
    elements = frequency_worker.screened_elements(chol, frequency_worker.orbs)
    save_array(frequency_folder/f"elements_{index}_{frequency_target_key}.npy", elements)
    return index, time.perf_counter()-start


def prepare_screening(gw, targets, controls, args, record):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import numpy as np
    from pyqed.gw.contour import ContourDeformation
    folder = args.output.resolve()/"screening_cache"
    folder.mkdir(exist_ok=True)
    signature = dict(energy_sha256=hashlib.sha256(gw.e_mf.tobytes()).hexdigest(),
                     coefficient_sha256=hashlib.sha256(gw.mo_coeff.tobytes()).hexdigest(),
                     eta=gw.eta, nw=controls["nw"], quadrature_scale=controls["quadrature_scale"])
    target_key = hashlib.sha256(targets.tobytes()).hexdigest()[:16]
    metadata = folder/"reference.json"
    if metadata.exists() and json.loads(metadata.read_text()) != signature:
        raise ValueError("Saved screening cache does not match this reference/window/grid")
    if not metadata.exists():
        save_array(folder/"MO_RI_factors.npy", gw._pair_factors)
        metadata.write_text(json.dumps(signature, indent=2)+"\n")
    gw._pair_factors = np.load(folder/"MO_RI_factors.npy", mmap_mode="r")
    pair_path = folder/"occupied_virtual_factors.npy"
    transitions = np.load(pair_path, mmap_mode="r") if pair_path.exists() else None
    contour = ContourDeformation(gw, gw.e_mf[::2], targets, prepare=False,
                                transition_factors=transitions, **controls)
    target_path = folder/f"QP_factors_{target_key}.npy"
    if not target_path.exists():
        temporary = target_path.with_name(target_path.name+".tmp")
        selected = np.lib.format.open_memmap(temporary, mode="w+", dtype=float,
                    shape=(len(contour.factors), len(targets), len(contour.energy)))
        for first in range(0, len(contour.factors), 64):
            selected[first:first+64] = contour.factors[first:first+64, :, targets].transpose(0, 2, 1)
        selected.flush()
        del selected
        temporary.replace(target_path)
    contour.target_factors = np.load(target_path, mmap_mode="r")
    if contour.target_factors.shape != (len(contour.factors), len(targets), len(contour.energy)):
        raise ValueError("Saved target factors do not retain all intermediate orbitals")
    if not (folder/"occupied_virtual_factors.npy").exists():
        save_array(folder/"occupied_virtual_factors.npy", contour.pairs)
    contour.pairs = np.load(folder/"occupied_virtual_factors.npy", mmap_mode="r")
    if contour.pairs.shape != (len(contour.factors), len(contour.gaps)):
        raise ValueError("Saved full transition factors have invalid dimensions")
    frequencies = np.r_[0., contour.freqs]
    pending = []

    def collect(index):
        chol = (np.load(folder/"cholesky_0.npy", mmap_mode="r"), True) if index == 0 else None
        contour.set_screening_frequency(index, np.load(folder/f"elements_{index}_{target_key}.npy"),
                                         cholesky=chol, max_memory=args.max_memory_mb)

    for index, frequency in enumerate(frequencies):
        if (folder/f"elements_{index}_{target_key}.npy").exists() and (folder/f"cholesky_{index}.npy").exists():
            collect(index)
        else:
            pending.append((index, float(frequency)))
    record("screening_tasks", frequencies_total=len(frequencies), frequencies_cached=len(frequencies)-len(pending),
           screening_workers=args.screening_workers, full_transition_count=len(contour.gaps),
           auxiliary_rank=len(contour.factors))
    if pending:
        with ProcessPoolExecutor(max_workers=args.screening_workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_screening_worker, initargs=(str(folder), contour.energy, contour.gaps,
                    contour.eta, contour.block, contour.orbs, target_key)) as pool:
            futures = [pool.submit(screening_frequency, task) for task in pending]
            for future in as_completed(futures):
                index, duration = future.result()
                collect(index)
                record("screening_frequency_complete", frequency_index=index,
                       frequencies_done=len(contour._prepared_frequencies), frequencies_total=len(frequencies),
                       wall_s=duration)
    if not contour.prepared:
        raise RuntimeError("Not all screening frequencies were collected")
    return contour


def input_data(path):
    tree = ast.parse(path.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("FC_XYZ", "DEFAULT_CONFIG"):
                    values[target.id] = ast.literal_eval(node.value)
    return values["FC_XYZ"], values["DEFAULT_CONFIG"]


def plot_history(out, history, threshold, label="TDI FC"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    sizes = [h["occupied"] for h in history]
    energy = np.array([h["energies_ev"] for h in history])
    for root in range(energy.shape[1]):
        axes[0].plot(sizes, energy[:, root], "o-", label=f"S{root+1}")
    axes[0].set(xlabel="Occupied / virtual orbitals per window", ylabel="Excitation energy (eV)")
    axes[0].legend(ncol=2, frameon=False)
    if len(history) > 1:
        axes[1].plot(sizes[1:], [h["max_change_mev"] for h in history[1:]], "o-")
    axes[1].axhline(threshold, color="0.5", linestyle="--", label=f"{threshold:g} meV target")
    axes[1].set(xlabel="Occupied / virtual orbitals per window", ylabel="Largest change among seven roots (meV)")
    axes[1].legend(frameon=False)
    fig.suptitle(f"{label} — orbital-window BSE, complete reference screening")
    prefix = "".join(c if c.isalnum() else "_" for c in label)
    fig.savefig(out/f"{prefix}_BSE_window_convergence.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-script", type=Path, required=True)
    parser.add_argument("--label", default="TDI FC")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, nargs="+", default=[8, 12, 20, 32, 48])
    parser.add_argument("--threshold-mev", type=float, default=10.)
    parser.add_argument("--max-memory-mb", type=float, default=6000.)
    parser.add_argument("--residue-solver", choices=["direct", "gmres", "lanczos"], default="lanczos",
                        help="Contour residue solver (default: lanczos, validated on TDI 8/8)")
    parser.add_argument("--resume", action="store_true", help="Reuse verified AO RI and converged RHF checkpoints")
    parser.add_argument("--screening-workers", type=int, default=num_workers(),
                        help="Independent frequencies sharing read-only disk-backed factors; one BLAS thread each")
    parser.add_argument("--qp-workers", type=int, default=1,
                        help="Independent orbital solves sharing complete screening; one BLAS thread each")
    args = parser.parse_args()
    if (not args.windows or any(v < 4 for v in args.windows)
            or args.windows != sorted(set(args.windows)) or args.threshold_mev <= 0
            or args.screening_workers < 1 or args.qp_workers < 1):
        parser.error("Require strictly growing windows of at least four and positive threshold")
    out = args.output.resolve()
    if args.resume:
        if not (out/"rhf.npz").is_file() or not (out/"AO_RI_factors.npy").is_file():
            parser.error("Resume requires both converged RHF and AO RI checkpoints")
    else:
        out.mkdir(parents=True, exist_ok=False)
    scratch = out/"scratch"
    scratch.mkdir(exist_ok=args.resume)
    os.environ.update(TMPDIR=str(scratch), PYSCF_TMPDIR=str(scratch))
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = "1"
    import numpy as np
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF
    from pyqed.qchem.basis import PackedRIFactors
    started = time.perf_counter()
    state = dict(status="running", stage="starting", pid=os.getpid(), history=[])
    lock, stop = threading.Lock(), threading.Event()
    def json_default(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(f"Unsupported JSON value: {type(value).__name__}")

    def record(event, **data):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        with lock:
            row = dict(event=event, elapsed_s=time.perf_counter()-started, stage=state["stage"],
                       cpu_s=usage.ru_utime+usage.ru_stime,
                       peak_rss_bytes=usage.ru_maxrss*(1 if sys.platform == "darwin" else 1024), **data)
            print(json.dumps(row, default=json_default), flush=True)
            with (out/"events.jsonl").open("a") as stream:
                stream.write(json.dumps(row, default=json_default)+"\n")
            status = dict(state, elapsed_s=row["elapsed_s"], cpu_s=row["cpu_s"],
                          peak_rss_bytes=row["peak_rss_bytes"])
            tmp = out/"status.tmp"
            tmp.write_text(json.dumps(status, indent=2, default=json_default)+"\n")
            tmp.replace(out/"status.json")

    def heartbeat():
        while not stop.wait(30):
            record("heartbeat")
    threading.Thread(target=heartbeat, daemon=True).start()

    @contextmanager
    def stage(name):
        state["stage"] = name
        before = time.perf_counter()
        record("stage_start")
        yield
        record("stage_complete", wall_s=time.perf_counter()-before)

    xyz, cfg = input_data(args.input_script)
    cfg["max_memory_mb"] = args.max_memory_mb
    (out/"input.xyz").write_text(xyz)
    source = Path(__file__).resolve().parents[1]
    provenance = dict(input_sha256=hashlib.sha256(args.input_script.read_bytes()).hexdigest(),
        settings=cfg, windows=args.windows, threshold_mev=args.threshold_mev, label=args.label,
        screening="complete occupied-virtual RI direct RPA", residue_solver=args.residue_solver,
        qp_workers=args.qp_workers,
        source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [source/"pyqed/gw/gw.py", source/"pyqed/gw/contour.py", source/"pyqed/gw/bse.py",
                      source/"pyqed/gw/lanczos_response.py"]})
    if args.resume:
        previous = json.loads((out/"provenance.json").read_text())
        if previous["input_sha256"] != provenance["input_sha256"] or previous["settings"] != cfg:
            raise ValueError("Resume input/settings differ from the saved reference")
        (out/"preparation_provenance.json").write_text(json.dumps(previous, indent=2)+"\n")
        provenance["reused_converged_rhf"] = True
    (out/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    try:
        mol = Molecule(atom="\n".join(xyz.strip().splitlines()[2:]), unit="angstrom",
                       basis=cfg["basis"], charge=0, spin=0)
        mol.max_memory = args.max_memory_mb
        if args.resume:
            with stage("restore_RHF"):
                mf = restore_rhf(mol, out, args.max_memory_mb)
        else:
            with stage("RI_integrals"):
                keys = ["auxbasis", "ri_metric_tol", "ri_metric_solver", "ri_purpose", "ri_storage",
                        "ri_tensor_backend", "ri_screen_tol", "ri_cache", "one_electron_cache"]
                mol.build(eri="ri", options=dict(workers=1,
                                                **{key: cfg[key] for key in keys}))
                save_array(out/"AO_RI_factors.npy", mol.eri_factors.pair_factors)
                mol.eri_factors = PackedRIFactors(np.load(out/"AO_RI_factors.npy", mmap_mode="r"), mol.nao)
                record("integral_sizes", nmo=mol.nao, auxiliary_rank=len(mol.eri_factors))
            with stage("RHF"):
                mf = RHF(mol)
                mf.max_memory = args.max_memory_mb
                mf.run(verbose=1, tol=cfg["scf_tol"], max_cycle=cfg["scf_max_cycle"])
                np.savez(out/"rhf.npz", mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ, mo_energy=mf.mo_energy,
                         e_hf=mf.e_tot, ao_overlap=mf.get_ovlp(), hcore=mf.get_hcore(), converged=mf.converged)
                if not mf.converged:
                    raise RuntimeError("TDI RHF did not converge")
        no, nmo = int(np.count_nonzero(mf.mo_occ)), len(mf.mo_energy)
        if args.windows[-1] > min(no, nmo-no):
            raise ValueError("Requested window exceeds occupied/virtual space")
        with stage("RHF_diagnostics"):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
            selected = np.arange(max(0, no-30), min(nmo, no+30))
            ax.plot(selected-no, mf.mo_energy[selected]*27.211386245988, "o", ms=3)
            ax.axvline(-.5, color="0.5", linestyle="--")
            ax.set(xlabel="Orbital index relative to LUMO", ylabel="RHF energy (eV)",
                   title=f"{args.label} — frontier orbital spectrum")
            prefix = "".join(c if c.isalnum() else "_" for c in args.label)
            fig.savefig(out/f"{prefix}_RHF_frontier_spectrum.png", dpi=160)
            plt.close(fig)
        from pyqed.gw.gw import GW
        from pyqed.gw.contour import ContourDeformation
        from pyqed.gw.bse import BSE
        controls = dict(nw=cfg["nw"], quadrature_scale=cfg["quadrature_scale"],
                        residue_solver=args.residue_solver, screening_tol=1e-12)
        with stage("GW_orbital_transformation"):
            folder = out/"screening_cache"
            factors = None
            if args.resume and (folder/"reference.json").exists():
                expected = dict(energy_sha256=hashlib.sha256(np.repeat(mf.mo_energy, 2).tobytes()).hexdigest(),
                    coefficient_sha256=hashlib.sha256(mf.mo_coeff.tobytes()).hexdigest(),
                    eta=cfg["eta"], nw=controls["nw"], quadrature_scale=controls["quadrature_scale"])
                if json.loads((folder/"reference.json").read_text()) != expected:
                    raise ValueError('Saved MO factors have different reference/screening settings')
                factors = np.load(folder/"MO_RI_factors.npy", mmap_mode="r")
            gw = GW(mf, screening=cfg["screening"], eta=cfg["eta"], freq_int="contour",
                    mo_pair_factors=factors)
        if args.resume and (out/"gw_partial.npz").exists():
            with np.load(out/"gw_partial.npz") as saved:
                for key in ["e_qp", "qp_weights", "qp_residuals", "orbital_indices"]:
                    setattr(gw, key, saved[key].copy())
        previous_subspace = None
        stable = 0
        for size in args.windows:
            active = np.arange(no-size, no+size)
            with stage(f"complete_screening_window_{size}_{size}"):
                gw._contour = prepare_screening(gw, active, controls, args, record)
            completed = set([] if gw.e_qp is None else np.flatnonzero(np.isfinite(gw.e_qp)))
            pending = [int(p) for p in sorted(active, key=lambda p: abs(p-(no-.5))) if p not in completed]
            def checkpoint_qp(p, info, **timing):
                temporary = out/'gw_partial.npz.tmp'
                with temporary.open('wb') as stream:
                    np.savez(stream, e_qp=gw.e_qp, qp_weights=gw.qp_weights,
                             qp_residuals=gw.qp_residuals, orbital_indices=gw.orbital_indices)
                temporary.replace(out/'gw_partial.npz')
                record("qp_complete", orbital=p, energy_ha=float(gw.e_qp[p]),
                       screening_info=info, **timing)
            if args.qp_workers > 1 and pending:
                state.update(qp_pending=pending.copy(), qp_completed=[], qp_workers=args.qp_workers)
                with stage(f"QP_parallel_window_{size}_{size}"):
                    for p, energy, weight, residual, info, duration in parallel_qp_orbitals(
                            gw, pending, controls, cfg['qp_solver_tol'], args.qp_workers):
                        if gw.e_qp is None:
                            gw.e_qp = np.full(nmo, np.nan)
                            gw.qp_weights = np.full(nmo, np.nan)
                            gw.qp_residuals = np.full(nmo, np.nan)
                        gw.e_qp[p], gw.qp_weights[p], gw.qp_residuals[p] = energy, weight, residual
                        gw.orbital_indices = np.array([p for p in active if np.isfinite(gw.e_qp[p])])
                        state['qp_pending'].remove(p)
                        state['qp_completed'].append(p)
                        checkpoint_qp(p, info, worker_wall_s=duration)
                    gw.converged, gw.method = True, 'g0w0'
            else:
                for p in pending:
                    with stage(f"QP_orbital_{p}"):
                        gw.run(orbs=[p], reuse_screening=True, tol=cfg["qp_solver_tol"], **controls)
                        checkpoint_qp(p, gw._contour.screening_info.copy())
            with stage(f"BSE_window_{size}_{size}"):
                bse = BSE(gw, orbital_window=(size, size))
                roots = 9
                while True:
                    bse.run(nroots=roots, low_rank=True, tol=cfg["response_solver_tol"],
                            max_cycle=cfg["bse_max_cycle"], max_space=max(cfg["bse_max_space"], 2*roots+8),
                            eigensolver=cfg["bse_eigensolver"], return_vectors=True)
                    # Retain a complete near-degenerate cluster at the S7 boundary.
                    tracked = int(np.count_nonzero(bse.e <= bse.e[6]+args.threshold_mev/1000/27.211386245988))
                    if tracked < roots:
                        break
                    if roots >= size*size-1:
                        raise RuntimeError("No excitation headroom above the seventh-state cluster")
                    roots = min(roots+2, size*size-1)
                if not bse.info["converged"]:
                    raise RuntimeError("BSE eigenvectors did not pass residual checks")
                energy = bse.e[:7]*27.211386245988
                row = dict(occupied=size, virtual=size, energies_ev=energy.tolist(),
                           solver_info=bse.info, max_change_mev=None, min_subspace_overlap=None,
                           tracked_subspace_roots=tracked)
                destinations = (np.arange(no-size, no)[:, None]*(nmo-no)
                                + np.arange(size)[None, :]).ravel()
                embedded = np.zeros((2*no*(nmo-no), tracked))
                embedded[destinations] = bse.x[:, :tracked]
                embedded[no*(nmo-no)+destinations] = bse.y[:, :tracked]
                subspace = np.linalg.qr(embedded, mode="reduced")[0]
                if previous_subspace is not None:
                    row["max_change_mev"] = float(np.max(abs(energy-np.array(state["history"][-1]["energies_ev"]))))*1000
                    row["min_subspace_overlap"] = (float(np.min(np.linalg.svd(
                        previous_subspace.T@subspace, compute_uv=False))**2)
                        if previous_subspace.shape[1] == tracked else 0.)
                    stable = stable+1 if (row["max_change_mev"] <= args.threshold_mev
                                          and row["min_subspace_overlap"] >= .95) else 0
                previous_subspace = subspace
                np.savez(out/f"BSE_window_{size}_{size}.npz", e=bse.e, x=bse.x, y=bse.y,
                         orbital_indices=bse.orbital_indices)
                state["history"].append(row)
                record("window_complete", occupied=size, virtual=size, energies_ev=energy.tolist(),
                       max_change_mev=row["max_change_mev"], min_subspace_overlap=row["min_subspace_overlap"])
                plot_history(out, state["history"], args.threshold_mev, args.label)
                (out/"window_history.json").write_text(json.dumps(state["history"], indent=2,
                                                               default=json_default)+"\n")
                if stable >= 2:
                    break
        state.update(status="complete", stage="complete", window_converged=stable >= 2)
        record("study_complete", window_converged=stable >= 2)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        record("study_failed", error=repr(error))
        traceback.print_exc()
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    main()
