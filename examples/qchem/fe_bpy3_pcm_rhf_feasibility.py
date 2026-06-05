"""PCM-RHF feasibility check for repaired [Fe(bpy)3]2+.

This driver deliberately separates the questions:

1. Can native PyQED RI-RHF converge for the full complex?
2. Can native PyQED PCM build the cavity and reaction-field potential for that
   density at a chosen surface resolution?
3. Optionally, can native PyQED run self-consistent PCM-RHF with that reaction
   field in the SCF Fock matrix?

By default the driver reports a frozen-density PCM correction and one-shot
solvent-field orbital diagnostics.  Pass ``--self-consistent-pcm`` to run the
new native self-consistent PCM-RHF path after the gas-phase reference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.qchem import Molecule, RHF
from pyqed.qchem.hf.rhf import _generalized_eigh
from pyqed.qchem.solvent.pcm import PCM
from pyqed.units import au2ev


def _load_fe_helpers():
    helper = Path(__file__).with_name("fe_bpy3_pyscf_ri_feasibility.py")
    spec = importlib.util.spec_from_file_location("fe_bpy3_pyscf_ri_feasibility", helper)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def atom_string(atoms):
    return "; ".join(f"{sym} {x:.10f} {y:.10f} {z:.10f}" for sym, x, y, z in atoms)


def timed(label, func):
    print(f"{label} ...", flush=True)
    t0 = time.perf_counter()
    out = func()
    seconds = time.perf_counter() - t0
    print(f"{label} finished in {seconds:.2f} s", flush=True)
    return out, seconds


def frontier(mo_energy, mo_occ, n=8):
    occ_idx = np.where(np.asarray(mo_occ) > 0.0)[0]
    homo = int(occ_idx[-1]) if occ_idx.size else -1
    lo = max(0, homo - n)
    hi = min(len(mo_energy), homo + n + 2)
    return homo, [
        {
            "index": int(i),
            "occ": float(mo_occ[i]),
            "energy_au": float(mo_energy[i]),
            "energy_ev": float(mo_energy[i] * au2ev),
        }
        for i in range(lo, hi)
    ]


def mulliken_atom_population(mol, dm, atom_index):
    labels = mol.ao_labels()
    idx = [i for i, label in enumerate(labels) if int(str(label).split()[0]) == int(atom_index)]
    ps = np.asarray(dm) @ np.asarray(mol.overlap)
    return float(np.trace(ps[np.ix_(idx, idx)]).real)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--charge", type=int, default=2)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--driver", default="builtin")
    parser.add_argument("--eri", choices=("auto", "dense", "s4", "s8", "direct", "factors", "ri"), default="ri")
    parser.add_argument("--auxbasis", default="def2-svp-rifit")
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eri-workers", type=int, default=None)
    parser.add_argument("--ri-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ri-cache-dir", type=Path, default=None)
    parser.add_argument("--ri-storage", choices=("auto", "packed", "full"), default="packed")
    parser.add_argument("--ri-tensor-backend", choices=("auto", "cython", "python", "native"), default="native")
    parser.add_argument("--max-cycle", type=int, default=120)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--conv-tol-grad", type=float, default=1e-6)
    parser.add_argument("--damping", type=float, default=0.2)
    parser.add_argument("--level-shift", type=float, default=0.5)
    parser.add_argument("--scf-diis", choices=("cdiis", "ediis", "adiis", "hybrid"), default="cdiis")
    parser.add_argument("--diis-space", type=int, default=12)
    parser.add_argument("--init-guess", default="minao")
    parser.add_argument("--pcm-eps", type=float, default=35.688)
    parser.add_argument("--pcm-method", default="C-PCM")
    parser.add_argument("--lebedev-order", type=int, default=3)
    parser.add_argument("--pcm-max-memory", type=float, default=4000.0)
    parser.add_argument("--pcm-integral-backend", choices=("auto", "native", "pyscf"), default="native")
    parser.add_argument("--self-consistent-pcm", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verbose", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    helper = _load_fe_helpers()
    atoms = helper.load_xyz(args.xyz) if args.xyz else helper.generated_fe_bpy3()

    print("Model:", "XYZ input" if args.xyz else "repaired generated [Fe(bpy)3]2+")
    print(f"Atoms={len(atoms)}, charge={args.charge}, spin={args.spin}, basis={args.basis}")
    print(f"PCM: method={args.pcm_method}, eps={args.pcm_eps}, lebedev_order={args.lebedev_order}")

    mol = Molecule(
        atom=atom_string(atoms),
        unit="angstrom",
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
    )
    build_options = {
        "parallel": args.parallel,
        "eri_workers": args.eri_workers,
        "ri_cache": args.ri_cache,
        "ri_cache_dir": None if args.ri_cache_dir is None else str(args.ri_cache_dir),
        "ri_storage": args.ri_storage,
        "ri_tensor_backend": args.ri_tensor_backend,
    }
    build_options = {k: v for k, v in build_options.items() if v is not None}

    _, build_seconds = timed(
        "PyQED integral build",
        lambda: mol.build(
            driver=args.driver,
            eri=args.eri,
            auxbasis=args.auxbasis,
            options=build_options,
        ),
    )
    print(f"Electrons={mol.nelec}, AOs={mol.nao}")
    build_info = getattr(mol, "_builtin_build_info", {}) or {}
    ri_info = dict(build_info.get("ri", {}) or {})
    if getattr(mol, "eri_factors", None) is not None:
        print(f"RI/low-rank factors: rank={mol.eri_factors.shape[0]}")
    if ri_info:
        print(
            "RI diagnostics: "
            f"builder={ri_info.get('tensor_builder')}, "
            f"workers={ri_info.get('workers')}, "
            f"cache_hit={ri_info.get('cache_hit')}"
        )

    mf, rhf_seconds = timed(
        "Gas PyQED RHF",
        lambda: RHF(mol).run(
            max_cycle=args.max_cycle,
            tol=args.tol,
            conv_tol_grad=args.conv_tol_grad,
            damping=args.damping,
            level_shift=args.level_shift,
            scf_diis=args.scf_diis,
            diis_space=args.diis_space,
            init_guess=args.init_guess,
            verbose=args.verbose,
        ),
    )
    print(f"Gas RHF converged={bool(getattr(mf, 'converged', False))}")
    print(f"E_gas = {mf.e_tot:.12f} Ha")

    def configured_pcm():
        obj = PCM(mol)
        obj.eps = args.pcm_eps
        obj.method = args.pcm_method
        obj.lebedev_order = args.lebedev_order
        obj.max_memory = args.pcm_max_memory
        obj.integral_backend = args.pcm_integral_backend
        obj.verbose = args.verbose
        return obj

    pcm = configured_pcm()
    (e_pcm, v_pcm), pcm_seconds = timed("Frozen-density PCM kernel", lambda: pcm.kernel(mf.dm))

    tr_vdm = float(np.einsum("ij,ji->", v_pcm, mf.dm).real)
    e_frozen = float(mf.e_tot + e_pcm)
    fock_gas = mf.get_fock(mf.dm)
    mo_energy_pcm_1shot, _mo_coeff_pcm = _generalized_eigh(fock_gas + v_pcm, mol.overlap)
    homo, gas_frontier = frontier(mf.mo_energy, mf.mo_occ)
    _homo2, pcm_frontier = frontier(mo_energy_pcm_1shot, mf.mo_occ)
    fe_pop = mulliken_atom_population(mol, mf.dm, 0)

    ngrids = int(len(pcm.surface["grid_coords"]))
    q_norm = float(np.linalg.norm(pcm._intermediates.get("q_sym", pcm._intermediates.get("q"))))
    v_norm = float(np.linalg.norm(v_pcm))
    print(f"PCM grids={ngrids}, E_pcm={e_pcm:.12f} Ha, E_gas+pcm={e_frozen:.12f} Ha")
    print(f"Tr[V_pcm D]={tr_vdm:.12f} Ha, ||V_pcm||={v_norm:.6e}, ||q||={q_norm:.6e}")
    print(f"Fe Mulliken population={fe_pop:.6f}")
    print(
        "One-shot frontier shifts/eV: "
        f"HOMO {pcm_frontier[8]['energy_ev'] - gas_frontier[8]['energy_ev']:.6f}, "
        f"LUMO {pcm_frontier[9]['energy_ev'] - gas_frontier[9]['energy_ev']:.6f}"
    )

    pcm_scf_summary = None
    if args.self_consistent_pcm:
        pcm_scf = configured_pcm()
        mf_pcm, pcm_scf_seconds = timed(
            "Self-consistent PCM-RHF",
            lambda: RHF(mol).run(
                dm0=mf.dm,
                with_solvent=pcm_scf,
                max_cycle=args.max_cycle,
                tol=args.tol,
                conv_tol_grad=args.conv_tol_grad,
                damping=args.damping,
                level_shift=args.level_shift,
                scf_diis=args.scf_diis,
                diis_space=args.diis_space,
                init_guess="dm",
                verbose=args.verbose,
            ),
        )
        print(f"PCM-RHF converged={bool(getattr(mf_pcm, 'converged', False))}")
        print(f"E_pcm_scf = {mf_pcm.e_tot:.12f} Ha")
        print(
            "PCM-SCF diagnostics: "
            f"E_solvent={mf_pcm.scf_info.get('solvent_energy', 0.0):.12f} Ha, "
            f"||V_solvent||={mf_pcm.scf_info.get('solvent_potential_norm', 0.0):.6e}"
        )
        pcm_scf_summary = {
            "seconds": float(pcm_scf_seconds),
            "converged": bool(getattr(mf_pcm, "converged", False)),
            "energy_ha": float(mf_pcm.e_tot),
            "energy_shift_from_gas_ha": float(mf_pcm.e_tot - mf.e_tot),
            "scf_info": dict(getattr(mf_pcm, "scf_info", {}) or {}),
            "pcm": {
                "ngrids": int(len(pcm_scf.surface["grid_coords"])),
                "integral_backend": pcm_scf._intermediates.get("integral_backend"),
                "energy_ha": float(getattr(pcm_scf, "e", 0.0)),
                "v_norm": float(np.linalg.norm(getattr(pcm_scf, "v", 0.0))),
            },
        }

    summary = {
        "model": "xyz" if args.xyz else "generated_fe_bpy3_repaired",
        "natom": len(atoms),
        "charge": args.charge,
        "spin": args.spin,
        "basis": args.basis,
        "driver": args.driver,
        "eri": args.eri,
        "auxbasis": args.auxbasis,
        "build_options": build_options,
        "nelectron": int(mol.nelec),
        "nao": int(mol.nao),
        "build_seconds": float(build_seconds),
        "rhf_seconds": float(rhf_seconds),
        "rhf_converged": bool(getattr(mf, "converged", False)),
        "gas_energy_ha": float(mf.e_tot),
        "scf_info": dict(getattr(mf, "scf_info", {}) or {}),
        "ri_rank": None if getattr(mol, "eri_factors", None) is None else int(mol.eri_factors.shape[0]),
        "ri_info": ri_info,
        "pcm": {
            "method": args.pcm_method,
            "eps": float(args.pcm_eps),
            "lebedev_order": int(args.lebedev_order),
            "ngrids": ngrids,
            "integral_backend": pcm._intermediates.get("integral_backend"),
            "seconds": float(pcm_seconds),
            "energy_ha": float(e_pcm),
            "gas_plus_pcm_energy_ha": e_frozen,
            "trace_vdm_ha": tr_vdm,
            "v_norm": v_norm,
            "q_norm": q_norm,
        },
        "pcm_scf": pcm_scf_summary,
        "fe_mulliken_population": fe_pop,
        "homo": int(homo),
        "gas_frontier": gas_frontier,
        "pcm_one_shot_frontier": pcm_frontier,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
