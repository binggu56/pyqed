"""Compare native PyQED OM2 heats of formation against G2 SI targets.

The published G2 table reports heats of formation, while the native OM2 driver
returns total energies.  Exact OM2 heats require the MNDO atomic reference
energy table, which is not implemented here yet.  This script therefore fits
simple atom-reference offsets on hydrides, then tests transferability on other
closed-shell G2 molecules.  Large validation residuals mean the current native
Hamiltonian is not yet quantitatively equivalent to published OM2.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.qchem.semiempirical import OM2, published_om2_molecule_benchmarks
from pyqed.units import au2kcalmol


KCAL_PER_HARTREE = au2kcalmol
ELEMENTS = ("H", "C", "N", "O", "F")


def bent(symbol_center, symbol_outer, r, angle_deg):
    angle = math.radians(angle_deg)
    half = 0.5 * angle
    x = r * math.sin(half)
    z = r * math.cos(half)
    return [
        (symbol_center, (0.0, 0.0, 0.0)),
        (symbol_outer, (x, 0.0, z)),
        (symbol_outer, (-x, 0.0, z)),
    ]


def trigonal_pyramid(center, outer, r, angle_deg):
    angle = math.radians(angle_deg)
    zfrac2 = max((math.cos(angle) + 0.5) / 1.5, 0.0)
    z = r * math.sqrt(zfrac2)
    rho = math.sqrt(max(r * r - z * z, 0.0))
    return [
        (center, (0.0, 0.0, 0.0)),
        (outer, (rho, 0.0, z)),
        (outer, (-0.5 * rho, math.sqrt(3.0) * 0.5 * rho, z)),
        (outer, (-0.5 * rho, -math.sqrt(3.0) * 0.5 * rho, z)),
    ]


def tetrahedral(center, outer, r):
    a = r / math.sqrt(3.0)
    return [
        (center, (0.0, 0.0, 0.0)),
        (outer, (a, a, a)),
        (outer, (a, -a, -a)),
        (outer, (-a, a, -a)),
        (outer, (-a, -a, a)),
    ]


GEOMETRIES = {
    "singlet methylene (CH2)": bent("C", "H", 1.11, 102.4),
    "methane (CH4)": tetrahedral("C", "H", 1.087),
    "ammonia (NH3)": trigonal_pyramid("N", "H", 1.012, 106.7),
    "water (H2O)": bent("O", "H", 0.9584, 104.45),
    "hydrogen fluoride (HF)": [("H", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 0.9168))],
    "acetylene (C2H2)": [
        ("H", (-1.6615, 0.0, 0.0)),
        ("C", (-0.6015, 0.0, 0.0)),
        ("C", (0.6015, 0.0, 0.0)),
        ("H", (1.6615, 0.0, 0.0)),
    ],
    "ethylene (C2H4)": [
        ("C", (-0.6695, 0.0, 0.0)),
        ("C", (0.6695, 0.0, 0.0)),
        ("H", (-1.212, 0.941, 0.0)),
        ("H", (-1.212, -0.941, 0.0)),
        ("H", (1.212, 0.941, 0.0)),
        ("H", (1.212, -0.941, 0.0)),
    ],
    "hydrogen cyanide (HCN)": [
        ("H", (-1.065, 0.0, 0.0)),
        ("C", (0.0, 0.0, 0.0)),
        ("N", (1.153, 0.0, 0.0)),
    ],
    "carbon monoxide (CO)": [("C", (0.0, 0.0, 0.0)), ("O", (1.128, 0.0, 0.0))],
    "formaldehyde (H2C=O)": [
        ("C", (0.0, 0.0, 0.0)),
        ("O", (1.208, 0.0, 0.0)),
        ("H", (-0.587, 0.932, 0.0)),
        ("H", (-0.587, -0.932, 0.0)),
    ],
    "N2 molecule": [("N", (0.0, 0.0, 0.0)), ("N", (1.098, 0.0, 0.0))],
    "F2 molecule": [("F", (0.0, 0.0, 0.0)), ("F", (1.412, 0.0, 0.0))],
    "carbon dioxide (CO2)": [
        ("O", (-1.16, 0.0, 0.0)),
        ("C", (0.0, 0.0, 0.0)),
        ("O", (1.16, 0.0, 0.0)),
    ],
    "CF4": tetrahedral("C", "F", 1.323),
    "N2O": [
        ("N", (-1.13, 0.0, 0.0)),
        ("N", (0.0, 0.0, 0.0)),
        ("O", (1.19, 0.0, 0.0)),
    ],
    "NF3": trigonal_pyramid("N", "F", 1.371, 102.5),
    "O3 (ozone)": bent("O", "O", 1.278, 116.8),
}


CALIBRATION = (
    "singlet methylene (CH2)",
    "methane (CH4)",
    "ammonia (NH3)",
    "water (H2O)",
    "hydrogen fluoride (HF)",
)


def atom_spec(geometry):
    return "; ".join(f"{sym} {x:.10f} {y:.10f} {z:.10f}" for sym, (x, y, z) in geometry)


def atom_counts(geometry):
    counts = Counter(sym for sym, _ in geometry)
    return np.array([counts[element] for element in ELEMENTS], dtype=float)


def run_native_om2(name, geometry, *, orthogonalization_correction=False):
    mf = OM2(
        atom=atom_spec(geometry),
        unit="angstrom",
        orthogonalization_correction=orthogonalization_correction,
    ).run()
    return float(mf.e_tot)


def fit_offsets(rows):
    x = []
    y = []
    for row in rows:
        x.append(row["counts"])
        y.append(row["target_om2"] - row["e_tot"] * KCAL_PER_HARTREE)
    offsets, *_ = np.linalg.lstsq(np.vstack(x), np.asarray(y), rcond=None)
    return offsets


def build_rows(*, orthogonalization_correction):
    records = {
        rec.name: rec
        for rec in published_om2_molecule_benchmarks("G2-CHNOF")
        if rec.name in GEOMETRIES
    }
    rows = []
    for name, geometry in GEOMETRIES.items():
        rec = records[name]
        e_tot = run_native_om2(name, geometry, orthogonalization_correction=orthogonalization_correction)
        rows.append(
            {
                "name": name,
                "counts": atom_counts(geometry),
                "e_tot": e_tot,
                "target_ref": rec.reference,
                "target_om2": rec.om2,
                "split": "fit" if name in CALIBRATION else "test",
            }
        )
    return rows


def summarize(rows):
    fit_rows = [row for row in rows if row["split"] == "fit"]
    offsets = fit_offsets(fit_rows)
    for row in rows:
        row["pyqed_heat"] = row["e_tot"] * KCAL_PER_HARTREE + float(row["counts"] @ offsets)
        row["err_vs_published_om2"] = row["pyqed_heat"] - row["target_om2"]
        row["err_vs_ref"] = row["pyqed_heat"] - row["target_ref"]
    test_err = np.array([row["err_vs_published_om2"] for row in rows if row["split"] == "test"])
    return offsets, test_err


def print_table(title, rows, offsets, test_err):
    print(title)
    print("Atom-reference offsets fitted to published OM2 hydride heats")
    for element, offset in zip(ELEMENTS, offsets):
        print(f"  {element:2s}: {offset:12.3f} kcal/mol")
    print()
    print("Native PyQED OM2 heat-of-formation diagnostic")
    header = f"{'split':<5} {'molecule':<28} {'E_tot/Eh':>13} {'PyQED':>10} {'OM2 SI':>10} {'err':>10} {'ref':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['split']:<5} {row['name']:<28} {row['e_tot']:>13.6f} "
            f"{row['pyqed_heat']:>10.2f} {row['target_om2']:>10.2f} "
            f"{row['err_vs_published_om2']:>10.2f} {row['target_ref']:>10.2f}"
        )
    print()
    print(
        "Validation MAE vs published OM2 SI = "
        f"{np.mean(np.abs(test_err)):.2f} kcal/mol over {len(test_err)} molecules"
    )


def main():
    rows = build_rows(orthogonalization_correction=False)
    offsets, test_err = summarize(rows)
    print_table("Native PyQED OM2 compact-NDDO diagnostic", rows, offsets, test_err)
    print()

    corrected_rows = build_rows(orthogonalization_correction=True)
    _, corrected_test_err = summarize(corrected_rows)
    print(
        "Approximate orthogonalization-correction MAE = "
        f"{np.mean(np.abs(corrected_test_err)):.2f} kcal/mol"
    )
    print("Caveat: geometries are compact literature-like guesses, not OM2-optimized G2 geometries.")


if __name__ == "__main__":
    main()
