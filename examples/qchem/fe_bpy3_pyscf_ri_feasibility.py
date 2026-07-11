"""Optional PySCF RI/DF-SCF feasibility check for full [Fe(bpy)3]2+.

This script intentionally does not import pyqed.  It is a standalone reference
driver for testing whether the full spin-crossover-size complex is practical
with PySCF density fitting before wiring any production PyQED workflow.

The built-in geometry is an idealized [Fe(bpy)3]2+ model for timing and
convergence tests.  Use ``--xyz`` with an optimized structure before drawing
chemical conclusions.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def _unit(vec):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("zero vector")
    return vec / norm


def _rotate_in_plane(vec, normal, theta):
    vec = np.asarray(vec, dtype=float)
    normal = _unit(normal)
    return (
        vec * math.cos(theta)
        + np.cross(normal, vec) * math.sin(theta)
        + normal * np.dot(normal, vec) * (1.0 - math.cos(theta))
    )


def _pyridine_from_edge(n_pos, connector_pos, other_n_pos, normal, ring_radius=1.39, ch=1.08):
    """Build a pyridine ring with N-C2 fixed and one missing H on C2."""
    n_pos = np.asarray(n_pos, dtype=float)
    connector_pos = np.asarray(connector_pos, dtype=float)
    other_n_pos = np.asarray(other_n_pos, dtype=float)
    normal = _unit(normal)
    edge = connector_pos - n_pos
    side = np.linalg.norm(edge)
    if side <= 0.0 or side > 2.0 * ring_radius:
        raise ValueError("invalid pyridine N-C edge")

    midpoint = 0.5 * (n_pos + connector_pos)
    height = math.sqrt(max(ring_radius * ring_radius - 0.25 * side * side, 0.0))
    edge_unit = edge / side
    perp = _unit(np.cross(normal, edge_unit))
    centers = (midpoint + height * perp, midpoint - height * perp)
    center = max(centers, key=lambda point: np.linalg.norm(point - other_n_pos))

    vertices = []
    radial0 = n_pos - center
    for idx in range(6):
        vertices.append(center + _rotate_in_plane(radial0, normal, math.radians(60.0 * idx)))

    # Choose the orientation where vertex 1 is closest to the requested C2 atom.
    if np.linalg.norm(vertices[1] - connector_pos) > np.linalg.norm(vertices[-1] - connector_pos):
        vertices = [vertices[0]] + list(reversed(vertices[1:]))

    atoms = [("N", *vertices[0])]
    for idx in range(1, 6):
        atoms.append(("C", *vertices[idx]))
        if idx != 1:
            h_pos = vertices[idx] + ch * _unit(vertices[idx] - center)
            atoms.append(("H", *h_pos))
    return atoms


def _bpy_atoms(n1, n2, normal, ring_radius=1.39, inter_ring_cc=1.47, ch=1.08):
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    normal = _unit(normal)
    axis = _unit(n2 - n1)
    lateral = _unit(np.cross(normal, axis))
    nn_distance = np.linalg.norm(n2 - n1)
    along = 0.5 * (nn_distance - inter_ring_cc)
    if along <= 0.0 or along >= ring_radius:
        raise ValueError("Fe-N scaffold is incompatible with the requested bpy geometry")
    offset = math.sqrt(ring_radius * ring_radius - along * along)
    c1 = n1 + along * axis + offset * lateral
    c2 = n2 - along * axis + offset * lateral
    return (
        _pyridine_from_edge(n1, c1, n2, normal, ring_radius=ring_radius, ch=ch)
        + _pyridine_from_edge(n2, c2, n1, -normal, ring_radius=ring_radius, ch=ch)
    )


def generated_fe_bpy3(fe_n=2.00):
    """Return a rough 61-atom [Fe(bpy)3]2+ structure in Angstrom."""
    axes = {
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "+z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    # Three cis N,N chelates that cover all six octahedral vertices.
    pairs = (("+x", "+y"), ("-x", "+z"), ("-y", "-z"))
    atoms = [("Fe", 0.0, 0.0, 0.0)]
    for left, right in pairs:
        u = axes[left]
        v = axes[right]
        normal = -np.cross(u, v)
        n1 = fe_n * u
        n2 = fe_n * v
        atoms.extend(_bpy_atoms(n1, n2, normal))
    return atoms


def atom_string(atoms):
    return "; ".join(f"{sym} {x:.8f} {y:.8f} {z:.8f}" for sym, x, y, z in atoms)


def load_xyz(path):
    lines = Path(path).read_text().splitlines()
    try:
        natom = int(lines[0].strip())
        body = lines[2 : 2 + natom]
    except Exception:
        body = lines
    atoms = []
    for line in body:
        fields = line.split()
        if len(fields) < 4 or fields[0].startswith("#"):
            continue
        atoms.append((fields[0], float(fields[1]), float(fields[2]), float(fields[3])))
    if not atoms:
        raise ValueError(f"No atoms found in {path}")
    return atoms


def make_mf(mol, args):
    from pyscf import scf

    reference = args.reference
    if reference == "auto":
        reference = "rhf" if mol.spin == 0 else "rohf"

    if reference == "rhf":
        mf = scf.RHF(mol)
    elif reference == "rohf":
        mf = scf.ROHF(mol)
    elif reference == "uhf":
        mf = scf.UHF(mol)
    else:
        raise ValueError(f"Unknown reference {reference!r}")

    mf = mf.density_fit(auxbasis=args.auxbasis)
    mf.max_cycle = args.max_cycle
    mf.conv_tol = args.conv_tol
    mf.level_shift = args.level_shift
    mf.damp = args.damping
    mf.diis_space = args.diis_space
    mf.init_guess = args.init_guess
    mf.verbose = args.verbose
    if args.chkfile:
        mf.chkfile = str(args.chkfile)

    if args.pcm:
        from pyscf import solvent

        mf = solvent.PCM(mf)
        mf.with_solvent.eps = args.pcm_eps
        mf.with_solvent.method = args.pcm_method
        mf.with_solvent.lebedev_order = args.pcm_lebedev_order
        mf.with_solvent.verbose = args.verbose

    if args.newton:
        mf = scf.newton(mf)
        mf.max_cycle = args.max_cycle
    return mf


def print_frontier(mf, n=12):
    mo_energy = np.asarray(mf.mo_energy)
    if mo_energy.ndim == 2:
        mo_energy = mo_energy[0]
    occ = np.asarray(mf.mo_occ)
    if occ.ndim == 2:
        occ = occ[0]
    occ_idx = np.where(occ > 0)[0]
    homo = int(occ_idx[-1]) if occ_idx.size else -1
    start = max(0, homo - n)
    stop = min(mo_energy.size, homo + n + 2)
    print("\nFrontier orbital energies")
    print("idx      occ       eps/eV")
    for idx in range(start, stop):
        marker = "H" if idx == homo else ("L" if idx == homo + 1 else " ")
        print(f"{idx:4d}{marker} {occ[idx]:8.3f} {mo_energy[idx] * 27.211386245988:12.5f}")
    return homo


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path, help="Optional real [Fe(bpy)3]2+ XYZ geometry.")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--cart", action="store_true", help="Use cartesian Gaussian functions.")
    parser.add_argument("--auxbasis", default=None)
    parser.add_argument("--charge", type=int, default=2)
    parser.add_argument("--spin", type=int, default=0, help="2S, so singlet=0, triplet=2, quintet=4.")
    parser.add_argument("--reference", choices=("auto", "rhf", "rohf", "uhf"), default="auto")
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--conv-tol", type=float, default=1e-7)
    parser.add_argument("--level-shift", type=float, default=0.5)
    parser.add_argument("--damping", type=float, default=0.2)
    parser.add_argument("--diis-space", type=int, default=12)
    parser.add_argument("--init-guess", default="minao")
    parser.add_argument("--newton", action="store_true")
    parser.add_argument("--pcm", action="store_true")
    parser.add_argument("--pcm-eps", type=float, default=35.688)
    parser.add_argument("--pcm-method", default="C-PCM")
    parser.add_argument("--pcm-lebedev-order", type=int, default=29)
    parser.add_argument("--chkfile", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verbose", type=int, default=4)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    from pyscf import gto

    atoms = load_xyz(args.xyz) if args.xyz else generated_fe_bpy3()
    print("Model:", "XYZ input" if args.xyz else "generated idealized [Fe(bpy)3]2+")
    print(f"Atoms={len(atoms)}, charge={args.charge}, spin={args.spin}, basis={args.basis}")

    mol = gto.M(
        atom=atom_string(atoms),
        unit="Angstrom",
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        cart=args.cart,
        verbose=args.verbose,
    )
    print(f"Electrons={mol.nelectron}, AOs={mol.nao_nr()}")

    mf = make_mf(mol, args)
    t0 = time.perf_counter()
    energy = mf.kernel()
    seconds = time.perf_counter() - t0
    print(f"\nSCF finished in {seconds:.2f} s")
    print(f"converged={bool(mf.converged)}")
    print(f"E = {energy:.12f} Ha")
    homo = print_frontier(mf)

    summary = {
        "model": "xyz" if args.xyz else "generated_fe_bpy3",
        "natom": len(atoms),
        "charge": args.charge,
        "spin": args.spin,
        "basis": args.basis,
        "cart": bool(args.cart),
        "auxbasis": args.auxbasis,
        "reference": args.reference,
        "pcm": bool(args.pcm),
        "pcm_eps": float(args.pcm_eps) if args.pcm else None,
        "pcm_method": args.pcm_method if args.pcm else None,
        "pcm_lebedev_order": int(args.pcm_lebedev_order) if args.pcm else None,
        "pcm_solvent_energy_ha": (
            float(getattr(mf.with_solvent, "e", 0.0)) if args.pcm else None
        ),
        "pcm_v_norm": (
            float(np.linalg.norm(getattr(mf.with_solvent, "v", 0.0))) if args.pcm else None
        ),
        "pcm_ngrids": (
            int(len(mf.with_solvent.surface["grid_coords"])) if args.pcm else None
        ),
        "nelectron": int(mol.nelectron),
        "nao": int(mol.nao_nr()),
        "converged": bool(mf.converged),
        "energy_ha": float(energy),
        "seconds": float(seconds),
        "homo": int(homo),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
