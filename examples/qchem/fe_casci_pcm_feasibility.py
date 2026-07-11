"""Fe(II) CASCI/PCM feasibility pilot.

This is a deliberately small first check for spin-crossover workflows.  The
default model is a minimal octahedral FeN6 proxy; the optional [Fe(NH3)6]2+
model is still not intended to replace a production [Fe(bpy)3]2+ active-space
study.  The goal is to verify that the
native PyQED RHF -> CASCI -> CASCI+PCM path can run spin-sector probes and to
make the computational bottlenecks visible early.

Examples
--------
    python examples/qchem/fe_casci_pcm_feasibility.py --model fe-n6 --pcm-cycles 1
    python examples/qchem/fe_casci_pcm_feasibility.py --spins 0,2,4 --nstates 2
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

from pyqed.qchem import CASCI, Molecule, RHF
from pyqed.units import au2ev


def _perpendicular_frame(axis):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, trial)) > 0.8:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = trial - np.dot(trial, axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2


def _atom_string(atoms):
    return "; ".join(f"{sym} {x:.8f} {y:.8f} {z:.8f}" for sym, x, y, z in atoms)


def fe_n6_atoms(fe_n=2.05):
    atoms = [("Fe", 0.0, 0.0, 0.0)]
    axes = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )
    for axis in axes:
        pos = fe_n * np.asarray(axis)
        atoms.append(("N", *pos))
    return atoms


def fe_nh3_6_atoms(fe_n=2.05, n_h=1.02):
    atoms = fe_n6_atoms(fe_n=fe_n)
    axes = [np.asarray(atom[1:], dtype=float) / fe_n for atom in atoms[1:]]

    # Three N-H bonds on the ligand side of each N, with near-ammonia H-N-H angles.
    cos_theta = 0.372
    sin_theta = float(np.sqrt(1.0 - cos_theta * cos_theta))
    azimuths = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
    for axis in axes:
        n_pos = fe_n * axis
        e1, e2 = _perpendicular_frame(axis)
        for phi in azimuths:
            h_dir = (
                cos_theta * axis
                + sin_theta * np.cos(phi) * e1
                + sin_theta * np.sin(phi) * e2
            )
            h_pos = n_pos + n_h * h_dir
            atoms.append(("H", *h_pos))
    return atoms


def fe_bpy3_atoms():
    helper = Path(__file__).with_name("fe_bpy3_pyscf_ri_feasibility.py")
    spec = importlib.util.spec_from_file_location("fe_bpy3_pyscf_ri_feasibility", helper)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.generated_fe_bpy3()


def make_atoms(model):
    if model == "fe-n6":
        return fe_n6_atoms(), "minimal FeN6 ligand-field proxy"
    if model == "fe-nh3-6":
        return fe_nh3_6_atoms(), "[Fe(NH3)6]2+ proxy"
    if model == "fe-bpy3":
        return fe_bpy3_atoms(), "repaired generated [Fe(bpy)3]2+"
    raise ValueError(f"Unknown model '{model}'.")


def parse_spins(text):
    spins = []
    for item in text.split(","):
        item = item.strip()
        if item:
            spins.append(int(item))
    if not spins:
        raise ValueError("--spins must contain at least one integer")
    return spins


def parse_active_orbitals(text):
    if text is None or str(text).strip() == "":
        return None
    active = tuple(int(item.strip()) for item in str(text).split(",") if item.strip())
    if not active:
        raise ValueError("--active-orbitals must contain at least one index.")
    return active


def timed(label, func):
    print(f"{label} ...", flush=True)
    t0 = time.perf_counter()
    out = func()
    dt = time.perf_counter() - t0
    print(f"{label} finished in {dt:.2f} s", flush=True)
    return out, dt


def _fe_3d_ao_indices(mol, atom_index=0):
    idx = []
    for ao_idx, label in enumerate(mol.ao_labels()):
        parts = str(label).split()
        if len(parts) < 3:
            continue
        try:
            atom = int(parts[0])
        except ValueError:
            continue
        shell = parts[-1].lower()
        if atom == int(atom_index) and "d" in shell:
            idx.append(ao_idx)
    if not idx:
        raise ValueError("No Fe 3d-like AO labels found. Check AO labels and atom ordering.")
    return np.asarray(idx, dtype=int)


def _mo_group_mulliken_weights(mf, ao_indices):
    coeff = np.asarray(mf.mo_coeff)
    overlap = np.asarray(mf.mol.overlap)
    scoeff = overlap @ coeff
    return np.einsum("pi,pi->i", coeff[ao_indices, :], scoeff[ao_indices, :], optimize=True).real


def select_fe_3d_orbitals(mf, ncas, atom_index=0, window=None):
    scores = _mo_group_mulliken_weights(mf, _fe_3d_ao_indices(mf.mol, atom_index=atom_index))
    if window is None:
        candidates = np.arange(scores.size)
    else:
        lo, hi = window
        candidates = np.arange(max(0, int(lo)), min(scores.size, int(hi) + 1))
    if candidates.size < ncas:
        raise ValueError("Fe 3d selection window has fewer orbitals than ncas.")
    ranked = candidates[np.argsort(scores[candidates])[::-1]]
    active = tuple(sorted(int(i) for i in ranked[:ncas]))
    diagnostics = [
        {
            "index": int(i),
            "energy_ev": float(mf.mo_energy[i] * au2ev),
            "occ": float(mf.mo_occ[i]),
            "fe_3d_weight": float(scores[i]),
        }
        for i in ranked[: max(ncas, min(12, ranked.size))]
    ]
    return active, diagnostics


def reorder_mo_for_active_orbitals(mf, active_orbitals, ncas, nelecas):
    active = [int(idx) for idx in active_orbitals]
    if len(active) != int(ncas):
        raise ValueError(f"active_orbitals must contain exactly ncas={ncas} entries.")
    if len(set(active)) != len(active):
        raise ValueError("active_orbitals contains duplicate indices.")
    mo_coeff = np.asarray(mf.mo_coeff)
    nmo = mo_coeff.shape[1]
    if min(active) < 0 or max(active) >= nmo:
        raise ValueError("active_orbitals contains an out-of-range MO index.")
    ncore2 = int(mf.nelec) - int(nelecas)
    if ncore2 < 0 or ncore2 % 2:
        raise ValueError("Inconsistent reference/active electron counts.")
    ncore = ncore2 // 2
    active_set = set(active)
    rest = [idx for idx in range(nmo) if idx not in active_set]
    order = rest[:ncore] + active + rest[ncore:]
    return mo_coeff[:, order], order


def run_casci_sector(mf, args, spin, with_pcm, mo_coeff=None):
    mc = CASCI(mf, ncas=args.ncas, nelecas=args.nelecas, spin=spin, verbose=args.verbose)
    if with_pcm:
        mc = mc.PCM(
            eps=args.pcm_eps,
            method=args.pcm_method,
            max_cycle=args.pcm_cycles,
            conv_tol=args.pcm_conv_tol,
            lebedev_order=args.lebedev_order,
            state_id=args.pcm_state_id,
            state_average=args.pcm_state_average,
        )
    result, seconds = timed(
        f"{'PCM ' if with_pcm else 'gas '}CAS({args.nelecas},{args.ncas}) spin={spin}",
        lambda: mc.run(
            nstates=args.nstates,
            mo_coeff=mo_coeff,
            method=args.casci_method,
            use_cholesky=False if args.dense_casci_integrals else None,
        ),
    )
    return {
        "spin": spin,
        "energies_au": np.asarray(result.e_tot, dtype=float).tolist(),
        "seconds": seconds,
    }


def print_table(label, rows):
    e0 = min(row["energies_au"][0] for row in rows)
    print(f"\n{label}")
    print("spin     E0/au             gap/eV      root gaps/eV")
    for row in rows:
        energies = np.asarray(row["energies_au"], dtype=float)
        root_gaps = (energies - energies[0]) * au2ev
        gap = (energies[0] - e0) * au2ev
        roots = " ".join(f"{x:9.4f}" for x in root_gaps)
        print(f"{row['spin']:4d} {energies[0]:17.10f} {gap:11.4f}   {roots}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("fe-nh3-6", "fe-n6", "fe-bpy3"), default="fe-n6")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--charge", type=int, default=2)
    parser.add_argument("--reference-spin", type=int, default=0)
    parser.add_argument("--driver", default="builtin")
    parser.add_argument(
        "--eri",
        choices=("auto", "dense", "s4", "s8", "direct", "factors", "ri"),
        default="ri",
        help="AO ERI representation. Use ri for the Fe feasibility path.",
    )
    parser.add_argument(
        "--auxbasis",
        default=None,
        help="Auxiliary basis for --eri ri. Defaults to the bundled RI/J-fit partner.",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable native integral parallelism where available.",
    )
    parser.add_argument("--eri-workers", type=int, default=None)
    parser.add_argument(
        "--ri-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache native RI factors by geometry/basis/auxbasis/options.",
    )
    parser.add_argument("--ri-cache-dir", type=Path, default=None)
    parser.add_argument("--ri-storage", choices=("auto", "packed", "full"), default="packed")
    parser.add_argument(
        "--ri-tensor-backend",
        choices=("auto", "cython", "python", "native"),
        default="native",
    )
    parser.add_argument("--ri-screen-tol", type=float, default=None)
    parser.add_argument("--scf-max-cycle", type=int, default=80)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-8)
    parser.add_argument("--scf-conv-tol-grad", type=float, default=1e-6)
    parser.add_argument("--scf-damping", type=float, default=0.2)
    parser.add_argument("--scf-level-shift", type=float, default=0.5)
    parser.add_argument("--scf-diis", choices=("cdiis", "ediis", "adiis", "hybrid"), default="cdiis")
    parser.add_argument("--scf-diis-space", type=int, default=12)
    parser.add_argument("--scf-init-guess", default="minao")
    parser.add_argument("--ncas", type=int, default=5)
    parser.add_argument("--nelecas", type=int, default=6)
    parser.add_argument(
        "--active-space",
        choices=("frontier", "fe-3d"),
        default="frontier",
        help="How to choose the active orbitals before CASCI.",
    )
    parser.add_argument(
        "--active-orbitals",
        type=parse_active_orbitals,
        default=None,
        help="Comma-separated zero-based RHF MO indices. Overrides --active-space.",
    )
    parser.add_argument(
        "--active-window",
        default=None,
        help="Optional inclusive MO index window lo:hi for automatic Fe-3d selection.",
    )
    parser.add_argument("--spins", default="0,2,4")
    parser.add_argument("--nstates", type=int, default=1)
    parser.add_argument("--casci-method", choices=("direct_ci", "ci", "jw"), default="direct_ci")
    parser.add_argument(
        "--dense-casci-integrals",
        action="store_true",
        help="Force dense active-space ERIs instead of reusing RI/Cholesky factors.",
    )
    parser.add_argument("--skip-gas", action="store_true")
    parser.add_argument("--skip-pcm", action="store_true")
    parser.add_argument("--pcm-eps", type=float, default=35.688, help="Acetonitrile dielectric.")
    parser.add_argument("--pcm-method", default="C-PCM")
    parser.add_argument("--pcm-cycles", type=int, default=3)
    parser.add_argument("--pcm-conv-tol", type=float, default=1e-6)
    parser.add_argument("--pcm-state-id", type=int, default=0)
    parser.add_argument("--pcm-state-average", action="store_true")
    parser.add_argument("--lebedev-order", type=int, default=17)
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    parser.add_argument("--verbose", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    spins = parse_spins(args.spins)
    atoms, model_label = make_atoms(args.model)

    print(f"Model: {model_label}")
    print(f"Atoms: {len(atoms)}, charge={args.charge}, reference spin={args.reference_spin}")
    print(
        f"Basis={args.basis}, active space=CAS({args.nelecas},{args.ncas}), "
        f"spin sectors={spins}"
    )
    if args.model == "fe-n6":
        print("Note: fe-n6 is only a wiring/speed proxy; use fe-nh3-6 or fe-bpy3 before chemistry claims.")
    if args.model == "fe-bpy3":
        print("Note: this generated Fe(bpy)3 geometry is for feasibility; use an optimized structure for production.")

    mol = Molecule(
        atom=_atom_string(atoms),
        unit="angstrom",
        basis=args.basis,
        charge=args.charge,
        spin=args.reference_spin,
    )
    build_options = {
        "parallel": args.parallel,
        "eri_workers": args.eri_workers,
        "ri_cache": args.ri_cache,
        "ri_cache_dir": None if args.ri_cache_dir is None else str(args.ri_cache_dir),
        "ri_storage": args.ri_storage,
        "ri_tensor_backend": args.ri_tensor_backend,
        "ri_screen_tol": args.ri_screen_tol,
    }
    build_options = {key: value for key, value in build_options.items() if value is not None}

    _, build_seconds = timed(
        "Integral build",
        lambda: mol.build(
            driver=args.driver,
            eri=args.eri,
            auxbasis=args.auxbasis,
            options=build_options,
        ),
    )
    print(f"Electrons={mol.nelec}, AOs={mol.nao}")
    eri_factors = getattr(mol, "eri_factors", None)
    ri_info = {}
    build_info = getattr(mol, "_builtin_build_info", None)
    if isinstance(build_info, dict):
        ri_info = dict(build_info.get("ri", {}) or {})
    if eri_factors is not None:
        print(f"RI/low-rank factors: rank={eri_factors.shape[0]}")
        if ri_info:
            print(
                "RI diagnostics: "
                f"builder={ri_info.get('tensor_builder')}, "
                f"workers={ri_info.get('workers')}, "
                f"cache_hit={ri_info.get('cache_hit')}"
            )

    mf, scf_seconds = timed(
        "RHF",
        lambda: RHF(mol).run(
            max_cycle=args.scf_max_cycle,
            tol=args.scf_conv_tol,
            conv_tol_grad=args.scf_conv_tol_grad,
            damping=args.scf_damping,
            level_shift=args.scf_level_shift,
            scf_diis=args.scf_diis,
            diis_space=args.scf_diis_space,
            init_guess=args.scf_init_guess,
            verbose=args.verbose,
        ),
    )
    print(f"RHF converged={getattr(mf, 'converged', None)}, E={mf.e_tot:.10f} au")

    ncore = (mf.nelec - args.nelecas) // 2
    active_orbitals = args.active_orbitals
    active_diagnostics = None
    if args.active_window is not None:
        lo, hi = (int(x.strip()) for x in str(args.active_window).split(":", 1))
        active_window = (lo, hi)
    else:
        active_window = None
    if active_orbitals is None and args.active_space == "fe-3d":
        active_orbitals, active_diagnostics = select_fe_3d_orbitals(
            mf,
            args.ncas,
            atom_index=0,
            window=active_window,
        )

    mo_coeff_cas = None
    mo_order = None
    if active_orbitals is not None:
        mo_coeff_cas, mo_order = reorder_mo_for_active_orbitals(
            mf,
            active_orbitals,
            args.ncas,
            args.nelecas,
        )
        active_indices = list(active_orbitals)
        active = np.asarray(mf.mo_energy[active_indices], dtype=float) * au2ev
        print(f"Selected active RHF MO indices: {active_indices}")
        if active_diagnostics is not None:
            print("Top Fe-3d candidates:")
            for row in active_diagnostics:
                print(
                    "  MO {index:4d}  occ={occ:4.1f}  eps={energy_ev:9.4f} eV  "
                    "Fe3d={fe_3d_weight:8.4f}".format(**row)
                )
    else:
        active_indices = list(range(ncore, ncore + args.ncas))
        active = np.asarray(mf.mo_energy[ncore : ncore + args.ncas], dtype=float) * au2ev
    if active_orbitals is None:
        print(
            f"Default active MO window: ncore={ncore}, indices={ncore}..{ncore + args.ncas - 1}"
        )
    else:
        print(
            f"CAS block after reordering: ncore={ncore}, active positions={ncore}..{ncore + args.ncas - 1}"
        )
    print("Active MO energies/eV:", " ".join(f"{x:9.4f}" for x in active))

    results = {
        "model": args.model,
        "basis": args.basis,
        "eri": args.eri,
        "auxbasis": args.auxbasis,
        "build_options": build_options,
        "charge": args.charge,
        "reference_spin": args.reference_spin,
        "natom": len(atoms),
        "nelec": int(mol.nelec),
        "nao": int(mol.nao),
        "ncas": args.ncas,
        "nelecas": args.nelecas,
        "spins": spins,
        "build_seconds": build_seconds,
        "scf_seconds": scf_seconds,
        "rhf_converged": bool(getattr(mf, "converged", False)),
        "rhf_energy_au": float(mf.e_tot),
        "ri_rank": None if eri_factors is None else int(eri_factors.shape[0]),
        "ri_info": ri_info,
        "ncore": int(ncore),
        "active_space": args.active_space,
        "active_orbitals": [int(i) for i in active_indices],
        "mo_reorder": None if mo_order is None else [int(i) for i in mo_order],
        "active_diagnostics": active_diagnostics,
        "active_mo_energies_ev": active.tolist(),
    }

    if not args.skip_gas:
        gas_rows = [run_casci_sector(mf, args, spin, with_pcm=False, mo_coeff=mo_coeff_cas) for spin in spins]
        results["gas"] = gas_rows
        print_table("Gas CASCI spin-sector scan", gas_rows)

    if not args.skip_pcm:
        pcm_rows = [run_casci_sector(mf, args, spin, with_pcm=True, mo_coeff=mo_coeff_cas) for spin in spins]
        results["pcm"] = pcm_rows
        print_table("PCM CASCI spin-sector scan", pcm_rows)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
