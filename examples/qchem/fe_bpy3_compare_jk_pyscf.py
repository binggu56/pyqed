"""Same-density PyQED/PySCF J/K comparison for [Fe(bpy)3]2+.

This optional diagnostic keeps PySCF out of the PyQED library.  It converges a
PySCF DF-RHF reference, reorders the PySCF AO density into PyQED's native AO
order, and compares one-electron matrices, J/K matrices, and RHF energies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.qchem import Molecule, RHF
from pyqed.qchem.hf.rhf import get_jk


def _load_fe_helpers():
    helper = Path(__file__).with_name("fe_bpy3_pyscf_ri_feasibility.py")
    spec = importlib.util.spec_from_file_location("fe_bpy3_pyscf_ri_feasibility", helper)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def atom_string(atoms):
    return "; ".join(f"{sym} {x:.10f} {y:.10f} {z:.10f}" for sym, x, y, z in atoms)


def atom_string_from_coords(symbols, coords):
    return "; ".join(
        f"{sym} {float(x):.15f} {float(y):.15f} {float(z):.15f}"
        for sym, (x, y, z) in zip(symbols, coords)
    )


def parse_gbs_for_pyscf(path):
    text = Path(path).read_text()
    data = re.split(r"\n\s*(\w[\w]?)\s+\w+\s*\n", text)
    if data and "\n" in data[0]:
        data = data[1:]

    lmap = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7}
    out = {}
    for atom, shells in zip(data[::2], data[1::2]):
        shell_parts = re.split(r"\n?\s*(\w+)\s+\w+\s+\w+\.\w+\s*\n", shells)[1:]
        atom_shells = []
        for angmom, body in zip(shell_parts[::2], shell_parts[1::2]):
            ams = [lmap[ch.lower()] for ch in angmom]
            rows = []
            for line in body.splitlines():
                match = re.search(
                    r"^\s*([0-9.DE+\-]+)\s+((?:(?:[0-9.DE+\-]+)\s+)*(?:[0-9.DE+\-]+))\s*$",
                    line,
                )
                if match is None:
                    continue
                exp = float(match.group(1).lower().replace("d", "e"))
                coeffs = [float(x.lower().replace("d", "e")) for x in match.group(2).split()]
                rows.append((exp, coeffs))
            for idx, lval in enumerate(ams):
                shell = [lval]
                for exp, coeffs in rows:
                    shell.append([exp, coeffs[idx]])
                atom_shells.append(shell)
        if atom_shells:
            out[atom] = atom_shells
    return out


def _norm_label(label):
    fields = str(label).split()
    if len(fields) < 3:
        return str(label).strip()
    atom_idx, symbol, orb = fields[:3]
    for old, new in (("dx2", "dxx"), ("dy2", "dyy"), ("dz2", "dzz")):
        orb = orb.replace(old, new)
    return f"{int(atom_idx)} {symbol} {orb}"


def ao_permutation(pyscf_labels, pyqed_labels):
    pyscf_map = {}
    for idx, label in enumerate(pyscf_labels):
        key = _norm_label(label)
        if key in pyscf_map:
            raise ValueError(f"Duplicate PySCF AO label after normalization: {key}")
        pyscf_map[key] = idx

    permutation = []
    missing = []
    for label in pyqed_labels:
        key = _norm_label(label)
        if key not in pyscf_map:
            missing.append(key)
        else:
            permutation.append(pyscf_map[key])
    if missing:
        raise ValueError(f"Could not map {len(missing)} PyQED AO labels to PySCF labels: {missing[:8]}")
    return np.asarray(permutation, dtype=int)


def matrix_stats(a, b):
    diff = np.asarray(a) - np.asarray(b)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "fro": float(np.linalg.norm(diff)),
        "rms": float(np.linalg.norm(diff) / np.sqrt(diff.size)),
    }


def rhf_energy(dm, hcore, vj, vk, enuc):
    vhf = vj - 0.5 * vk
    e1 = np.einsum("ij,ji->", hcore, dm).real
    e2 = 0.5 * np.einsum("ij,ji->", vhf, dm).real
    return float(e1 + e2 + enuc), float(e1), float(e2)


def transform_to_pyqed_normalization(matrix, scale):
    return scale[:, None] * np.asarray(matrix, dtype=float) * scale[None, :]


def transform_density_to_pyqed_normalization(dm, scale):
    return np.asarray(dm, dtype=float) / scale[:, None] / scale[None, :]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--auxbasis", default="def2-svp-rifit")
    parser.add_argument("--charge", type=int, default=2)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--max-cycle", type=int, default=160)
    parser.add_argument("--conv-tol", type=float, default=1e-8)
    parser.add_argument("--level-shift", type=float, default=2.0)
    parser.add_argument("--damping", type=float, default=0.4)
    parser.add_argument("--diis-space", type=int, default=8)
    parser.add_argument("--init-guess", default="minao")
    parser.add_argument("--ri-cache-dir", type=Path, default=Path("/private/tmp/pyqed-ri-cache-shelltriplet-threaded-cold-v2"))
    parser.add_argument("--eri-workers", type=int, default=6)
    parser.add_argument(
        "--run-pyqed-rhf",
        action="store_true",
        help="Continue PyQED RHF from the transformed PySCF density.",
    )
    parser.add_argument("--pyqed-rhf-max-cycle", type=int, default=12)
    parser.add_argument("--pyqed-rhf-tol", type=float, default=1e-8)
    parser.add_argument("--pyqed-rhf-conv-tol-grad", type=float, default=1e-6)
    parser.add_argument("--pyqed-rhf-damping", type=float, default=0.0)
    parser.add_argument("--pyqed-rhf-level-shift", type=float, default=0.0)
    parser.add_argument(
        "--pyqed-rhf-scf-diis",
        choices=("cdiis", "ediis", "adiis", "hybrid"),
        default="cdiis",
    )
    parser.add_argument("--out", type=Path, default=Path("/private/tmp/pyqed_fe_bpy3_same_density_jk_compare.json"))
    parser.add_argument("--verbose", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    from pyscf import gto, scf

    helper = _load_fe_helpers()
    atoms = helper.load_xyz(args.xyz) if args.xyz else helper.generated_fe_bpy3()
    atom = atom_string(atoms)
    symbols = [sym for sym, *_ in atoms]

    aux_path = ROOT / "pyqed" / "qchem" / "basis_set" / f"{args.auxbasis}.1.gbs"
    if not aux_path.exists():
        raise FileNotFoundError(aux_path)
    auxbasis = parse_gbs_for_pyscf(aux_path)

    print("Building PyQED native RI ...", flush=True)
    mol = Molecule(atom=atom, unit="angstrom", basis=args.basis, charge=args.charge, spin=args.spin)
    t0 = time.perf_counter()
    mol.build(
        driver="builtin",
        eri="ri",
        auxbasis=args.auxbasis,
        options={
            "ri_tensor_backend": "native",
            "ri_storage": "packed",
            "ri_cache": True,
            "ri_cache_dir": str(args.ri_cache_dir),
            "parallel": True,
            "eri_workers": args.eri_workers,
            "one_electron_cache": True,
        },
    )
    pyqed_build_seconds = time.perf_counter() - t0

    print("Building PySCF reference ...", flush=True)
    pyscf_atom = atom_string_from_coords(symbols, mol.atom_coords())
    pmol = gto.M(
        atom=pyscf_atom,
        unit="Bohr",
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        cart=True,
        verbose=args.verbose,
    )
    pmf = scf.RHF(pmol).density_fit(auxbasis=auxbasis)
    pmf.max_cycle = args.max_cycle
    pmf.conv_tol = args.conv_tol
    pmf.level_shift = args.level_shift
    pmf.damp = args.damping
    pmf.diis_space = args.diis_space
    pmf.init_guess = args.init_guess
    pmf.verbose = args.verbose
    t0 = time.perf_counter()
    pyscf_energy = pmf.kernel()
    pyscf_seconds = time.perf_counter() - t0
    dm_pyscf = np.asarray(pmf.make_rdm1(), dtype=float)
    s_pyscf = pmol.intor_symmetric("int1e_ovlp")
    h_pyscf = pmf.get_hcore()
    vj_pyscf, vk_pyscf = pmf.get_jk(dm=dm_pyscf)

    perm = ao_permutation(pmol.ao_labels(), mol.ao_labels())
    dm_ref_raw = dm_pyscf[np.ix_(perm, perm)]
    s_ref_raw = s_pyscf[np.ix_(perm, perm)]
    h_ref_raw = h_pyscf[np.ix_(perm, perm)]
    vj_ref_raw = vj_pyscf[np.ix_(perm, perm)]
    vk_ref_raw = vk_pyscf[np.ix_(perm, perm)]

    # PySCF's cartesian d functions are not normalized the same way as PyQED's
    # AO basis.  Use the diagonal overlap ratio after label permutation to
    # compare matrices and densities in PyQED's normalized AO representation.
    ao_scale = np.sqrt(np.diag(mol.overlap) / np.diag(s_ref_raw))
    dm = transform_density_to_pyqed_normalization(dm_ref_raw, ao_scale)
    s_ref = transform_to_pyqed_normalization(s_ref_raw, ao_scale)
    h_ref = transform_to_pyqed_normalization(h_ref_raw, ao_scale)
    vj_ref = transform_to_pyqed_normalization(vj_ref_raw, ao_scale)
    vk_ref = transform_to_pyqed_normalization(vk_ref_raw, ao_scale)

    t0 = time.perf_counter()
    vj_pyqed, vk_pyqed = get_jk(mol, dm, eri_factors=mol.eri_factors)
    pyqed_jk_seconds = time.perf_counter() - t0

    enuc_pyscf = float(pmol.energy_nuc())
    enuc_pyqed = float(mol.energy_nuc())
    e_pyscf_formula, e1_pyscf, e2_pyscf = rhf_energy(dm, h_ref, vj_ref, vk_ref, enuc_pyscf)
    e_pyqed_formula, e1_pyqed, e2_pyqed = rhf_energy(dm, mol.hcore, vj_pyqed, vk_pyqed, enuc_pyqed)

    summary = {
        "natom": len(atoms),
        "basis": args.basis,
        "auxbasis": args.auxbasis,
        "charge": args.charge,
        "spin": args.spin,
        "coordinate_reference": "PyQED Bohr coordinates",
        "nao_pyscf": int(pmol.nao_nr()),
        "nao_pyqed": int(mol.nao),
        "pyscf_converged": bool(pmf.converged),
        "pyscf_kernel_energy_ha": float(pyscf_energy),
        "pyscf_formula_energy_ha": e_pyscf_formula,
        "pyqed_formula_energy_ha": e_pyqed_formula,
        "energy_delta_pyqed_minus_pyscf_ha": float(e_pyqed_formula - e_pyscf_formula),
        "one_electron_energy_delta_ha": float(e1_pyqed - e1_pyscf),
        "two_electron_energy_delta_ha": float(e2_pyqed - e2_pyscf),
        "nuclear_energy_delta_ha": float(enuc_pyqed - enuc_pyscf),
        "pyscf_seconds": float(pyscf_seconds),
        "pyqed_build_seconds": float(pyqed_build_seconds),
        "pyqed_jk_seconds": float(pyqed_jk_seconds),
        "ao_normalization_scale_min": float(np.min(ao_scale)),
        "ao_normalization_scale_max": float(np.max(ao_scale)),
        "overlap_delta_before_normalization": matrix_stats(mol.overlap, s_ref_raw),
        "overlap_delta": matrix_stats(mol.overlap, s_ref),
        "hcore_delta": matrix_stats(mol.hcore, h_ref),
        "vj_delta": matrix_stats(vj_pyqed, vj_ref),
        "vk_delta": matrix_stats(vk_pyqed, vk_ref),
        "vhf_delta": matrix_stats(vj_pyqed - 0.5 * vk_pyqed, vj_ref - 0.5 * vk_ref),
        "dm_trace_overlap_pyqed": float(np.einsum("ij,ji->", dm, mol.overlap).real),
        "dm_trace_overlap_pyscf": float(np.einsum("ij,ji->", dm, s_ref).real),
        "ri_info": dict((getattr(mol, "_builtin_build_info", {}) or {}).get("ri", {}) or {}),
    }

    if args.run_pyqed_rhf:
        print("Running PyQED RHF from PySCF density ...", flush=True)
        t0 = time.perf_counter()
        mf = RHF(mol).run(
            dm0=dm,
            init_guess="dm",
            max_cycle=args.pyqed_rhf_max_cycle,
            tol=args.pyqed_rhf_tol,
            conv_tol_grad=args.pyqed_rhf_conv_tol_grad,
            damping=args.pyqed_rhf_damping,
            level_shift=args.pyqed_rhf_level_shift,
            scf_diis=args.pyqed_rhf_scf_diis,
            verbose=args.verbose,
        )
        pyqed_rhf_seconds = time.perf_counter() - t0
        dm_final = np.asarray(mf.dm, dtype=float)
        summary["pyqed_rhf_from_pyscf_density"] = {
            "seconds": float(pyqed_rhf_seconds),
            "max_cycle": int(args.pyqed_rhf_max_cycle),
            "tol": float(args.pyqed_rhf_tol),
            "conv_tol_grad": float(args.pyqed_rhf_conv_tol_grad),
            "damping": float(args.pyqed_rhf_damping),
            "level_shift": float(args.pyqed_rhf_level_shift),
            "scf_diis": args.pyqed_rhf_scf_diis,
            "converged": bool(getattr(mf, "converged", False)),
            "energy_ha": float(mf.e_tot),
            "energy_delta_from_same_density_pyqed_ha": float(mf.e_tot - e_pyqed_formula),
            "energy_delta_from_pyscf_ha": float(mf.e_tot - e_pyscf_formula),
            "density_delta_from_pyscf_density_fro": float(np.linalg.norm(dm_final - dm)),
            "density_delta_from_pyscf_density_rms": float(
                np.linalg.norm(dm_final - dm) / np.sqrt(dm.size)
            ),
            "electron_count": float(np.einsum("ij,ji->", dm_final, mol.overlap).real),
            "scf_info": dict(getattr(mf, "scf_info", {}) or {}),
        }

    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
