"""Benchmark PyQED AM1 heats of formation against MOPAC.

This is a single-point benchmark at fixed Cartesian geometries.  It is intended
as a debugging harness for the PyQED AM1 implementation, not as a published
thermochemistry benchmark.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pyqed.qchem import Molecule
    from pyqed.qchem.semiempirical.am1 import RAM1


def find_mopac():
    env_path = os.environ.get("MOPAC")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
    found = shutil.which("mopac")
    if found:
        return Path(found)
    legacy = Path("/opt/anaconda3/bin/mopac")
    if legacy.exists():
        return legacy
    return None


MOPAC = find_mopac()


GEOMETRIES = {
    "H2": [("H", (0.0, 0.0, 0.0)), ("H", (0.74, 0.0, 0.0))],
    "CH4": [
        ("C", (0.0, 0.0, 0.0)),
        ("H", (0.6276, 0.6276, 0.6276)),
        ("H", (0.6276, -0.6276, -0.6276)),
        ("H", (-0.6276, 0.6276, -0.6276)),
        ("H", (-0.6276, -0.6276, 0.6276)),
    ],
    "NH3": [
        ("N", (0.0, 0.0, 0.0)),
        ("H", (0.940, 0.0, 0.363)),
        ("H", (-0.470, 0.814, 0.363)),
        ("H", (-0.470, -0.814, 0.363)),
    ],
    "H2O": [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.75695, 0.0, 0.58588)),
        ("H", (-0.75695, 0.0, 0.58588)),
    ],
    "HF": [("H", (0.0, 0.0, 0.0)), ("F", (0.9168, 0.0, 0.0))],
    "CO": [("C", (0.0, 0.0, 0.0)), ("O", (1.128, 0.0, 0.0))],
    "CO2": [
        ("O", (-1.16, 0.0, 0.0)),
        ("C", (0.0, 0.0, 0.0)),
        ("O", (1.16, 0.0, 0.0)),
    ],
    "C2H2": [
        ("H", (-1.6615, 0.0, 0.0)),
        ("C", (-0.6015, 0.0, 0.0)),
        ("C", (0.6015, 0.0, 0.0)),
        ("H", (1.6615, 0.0, 0.0)),
    ],
}


def atom_string(geometry):
    return "; ".join(f"{sym} {x:.10f} {y:.10f} {z:.10f}" for sym, (x, y, z) in geometry)


def mopac_input(geometry):
    lines = ["AM1 1SCF PRECISE XYZ", "PyQED AM1 benchmark", ""]
    for sym, (x, y, z) in geometry:
        lines.append(f"{sym} {x:.10f} 0 {y:.10f} 0 {z:.10f} 0")
    return "\n".join(lines) + "\n"


def run_mopac(name, geometry):
    if MOPAC is None:
        raise FileNotFoundError("MOPAC executable not found. Set $MOPAC or add mopac to PATH.")
    with tempfile.TemporaryDirectory(prefix="pyqed-am1-", dir="/private/tmp") as tmp:
        path = Path(tmp) / f"{name.lower()}.mop"
        path.write_text(mopac_input(geometry), encoding="utf-8")
        subprocess.run([str(MOPAC), str(path)], cwd=tmp, check=True, capture_output=True, text=True)
        out = path.with_suffix(".out").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"FINAL HEAT OF FORMATION\s*=\s*([-+0-9.]+)\s+KCAL/MOL", out)
    if not match:
        raise RuntimeError(f"Could not parse MOPAC heat of formation for {name}.")
    return float(match.group(1))


def run_pyqed(name, geometry):
    mol = Molecule(atom=atom_string(geometry), unit="Angstrom")
    mf = RAM1(mol).run(conv_tol=1e-8, verbose=0)
    return float(mf.e_heat_formation)


def main():
    rows = []
    for name, geometry in GEOMETRIES.items():
        mopac_heat = run_mopac(name, geometry)
        pyqed_heat = run_pyqed(name, geometry)
        rows.append((name, pyqed_heat, mopac_heat, pyqed_heat - mopac_heat))

    header = f"{'molecule':<8} {'PyQED AM1':>14} {'MOPAC AM1':>14} {'diff':>14}"
    print(header)
    print("-" * len(header))
    for name, pyqed_heat, mopac_heat, diff in rows:
        print(f"{name:<8} {pyqed_heat:>14.6f} {mopac_heat:>14.6f} {diff:>14.6f}")
    mae = sum(abs(row[3]) for row in rows) / len(rows)
    print()
    print(f"MAE = {mae:.6f} kcal/mol over {len(rows)} fixed-geometry single points")


if __name__ == "__main__":
    main()
