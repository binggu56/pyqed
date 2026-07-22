"""Static isotropic IAM UED and mode-resolved CHD difference signals."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import atomic_mass, hbar, physical_constants, speed_of_light

from pyqed.ued.ued import electron_atomic_form_factor


GEOMETRY = Path("chd_c2_casscf44_augccpvdz.xyz")
MODE_DATA = Path("chd_c2_b3lyp_augccpvdz_normal_modes.npz")
PROJECTION_DATA = Path("chd_c2_3px_3py_mode_projection.npz")
OUTPUT_PREFIX = Path("chd_c2_static_mode_ued")
SELECTED_MODES = (1, 5, 13, 27)
# Paper molecular frame: 3px is out of the conjugated plane; 3py is the
# corresponding in-plane B-symmetry state. Labels are not XYZ Cartesian axes.
STATE_WEIGHTS = {"3px": 1.0 / 1.8, "3py": 0.8 / 1.8}
HARTREE_J = physical_constants["Hartree energy"][0]
BOHR_M = physical_constants["Bohr radius"][0]
BOHR_ANGSTROM = BOHR_M * 1.0e10


def read_xyz(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    natom = int(lines[0])
    fields = [line.split() for line in lines[2 : 2 + natom]]
    symbols = [row[0] for row in fields]
    coords = np.asarray([[float(x) for x in row[1:4]] for row in fields])
    return symbols, coords


def isotropic_iam(symbols, coords, s):
    factors = np.vstack(
        [
            electron_atomic_form_factor(symbol, s, q_unit="angstrom^-1")
            for symbol in symbols
        ]
    )
    i_atomic = np.sum(factors**2, axis=0)
    i_molecular = np.zeros_like(s)
    for left in range(len(symbols)):
        for right in range(left + 1, len(symbols)):
            distance = np.linalg.norm(coords[right] - coords[left])
            i_molecular += (
                2.0
                * factors[left]
                * factors[right]
                * np.sinc(s * distance / np.pi)
            )
    i_total = i_atomic + i_molecular
    sm = s * i_molecular / i_atomic
    return i_atomic, i_molecular, i_total, sm


def displacement_from_force(force_q_au, frequency_cm1):
    force_si = force_q_au * HARTREE_J / (BOHR_M * np.sqrt(atomic_mass))
    omega_si = 2.0 * np.pi * speed_of_light * 100.0 * frequency_cm1
    delta_q_si = force_si / omega_si**2
    return delta_q_si / (BOHR_M * np.sqrt(atomic_mass))


def write_xyz(path, symbols, coords, comment):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(symbols)}\n{comment}\n")
        for symbol, xyz in zip(symbols, coords):
            handle.write(
                f"{symbol:2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}\n"
            )


def main():
    symbols, coords = read_xyz(GEOMETRY)
    modes = np.load(MODE_DATA)
    projected = np.load(PROJECTION_DATA)
    s = np.linspace(0.05, 12.0, 1200)
    i_atomic, i_molecular, i_total, sm = isotropic_iam(symbols, coords, s)

    mode_results = {}
    for mode_number in SELECTED_MODES:
        index = mode_number - 1
        frequency = float(modes["frequencies_cm1"][index])
        state_intensities = []
        state_sms = []
        for state, weight in STATE_WEIGHTS.items():
            force = float(projected[f"force_projection_{state}_au"][index])
            delta_q = displacement_from_force(force, frequency)
            # A vertically launched displaced-harmonic packet reaches 2 Delta Q.
            displaced = coords + 2.0 * delta_q * modes["normal_modes"][index] * BOHR_ANGSTROM
            _, _, displaced_total, displaced_sm = isotropic_iam(symbols, displaced, s)
            state_intensities.append(weight * displaced_total)
            state_sms.append(weight * displaced_sm)
            write_xyz(
                Path(f"{OUTPUT_PREFIX}_mode{mode_number}_{state}_max.xyz"),
                symbols,
                displaced,
                f"Mode {mode_number}, {state}, maximum harmonic excursion; "
                f"frequency={frequency:.6f} cm^-1",
            )
        population_intensity = np.sum(state_intensities, axis=0)
        population_sm = np.sum(state_sms, axis=0)
        mode_results[mode_number] = {
            "frequency": frequency,
            "delta_intensity_percent": 100.0 * (population_intensity - i_total) / i_total,
            "delta_sm": population_sm - sm,
        }

    fields = ["s_angstrom-1", "I_atomic", "I_molecular", "I_total", "sM"]
    for mode_number in SELECTED_MODES:
        fields.extend(
            [
                f"mode{mode_number}_delta_I_percent",
                f"mode{mode_number}_delta_sM",
            ]
        )
    with Path(f"{OUTPUT_PREFIX}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for q_index, q in enumerate(s):
            row = {
                "s_angstrom-1": q,
                "I_atomic": i_atomic[q_index],
                "I_molecular": i_molecular[q_index],
                "I_total": i_total[q_index],
                "sM": sm[q_index],
            }
            for mode_number, result in mode_results.items():
                row[f"mode{mode_number}_delta_I_percent"] = result[
                    "delta_intensity_percent"
                ][q_index]
                row[f"mode{mode_number}_delta_sM"] = result["delta_sm"][q_index]
            writer.writerow(row)

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 6.4), sharex=True)
    axes[0].plot(s, sm, color="black", lw=1.25)
    axes[0].axhline(0.0, color="0.7", lw=0.7)
    axes[0].set_ylabel(r"$sM(s)$")
    axes[0].text(-0.10, 1.02, "a", transform=axes[0].transAxes, fontweight="bold")

    for color, mode_number in zip(colors, SELECTED_MODES):
        result = mode_results[mode_number]
        axes[1].plot(
            s,
            result["delta_intensity_percent"],
            color=color,
            lw=1.2,
            label=rf"mode {mode_number}: {result['frequency']:.0f} cm$^{{-1}}$",
        )
    axes[1].axhline(0.0, color="0.7", lw=0.7)
    axes[1].set(xlabel=r"$s$ ($\mathrm{\AA}^{-1}$)", ylabel=r"$\Delta I/I_0$ (%)")
    axes[1].text(-0.10, 1.02, "b", transform=axes[1].transAxes, fontweight="bold")
    axes[1].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    for axis in axes:
        axis.set_xlim(0.0, 12.0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.21, hspace=0.08)
    fig.savefig(f"{OUTPUT_PREFIX}.pdf")
    fig.savefig(f"{OUTPUT_PREFIX}.png", dpi=350)
    plt.close(fig)

    print(f"csv={OUTPUT_PREFIX}.csv")
    print(f"pdf={OUTPUT_PREFIX}.pdf")
    print(f"png={OUTPUT_PREFIX}.png")
    for mode_number, result in mode_results.items():
        peak = np.max(np.abs(result["delta_intensity_percent"]))
        peak_s = s[np.argmax(np.abs(result["delta_intensity_percent"]))]
        print(
            f"mode {mode_number}: {result['frequency']:.3f} cm^-1, "
            f"max |dI/I|={peak:.4f}% at s={peak_s:.3f} A^-1"
        )


if __name__ == "__main__":
    main()
