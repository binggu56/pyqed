"""Rotationally averaged CHD UED from explicit SA-CASSCF electron densities."""

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from pyscf import fci, gto, mcscf, scf
from pyscf.gto import ft_ao

from pyqed import au2angstrom
from pyqed.ued.ued import electron_atomic_form_factor

QCHEM_EXAMPLES = Path(__file__).resolve().parents[1] / "qchem"
if str(QCHEM_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(QCHEM_EXAMPLES))

from chd_sa_casscf48_aug_rydberg import RYDBERG_BASIS, read_xyz


GEOMETRY = Path("chd_c2_casscf44_augccpvdz.xyz")
SA_DATA = Path("chd_c2_sa_casscf48_aug_rydberg.json")
SA_MO = Path("chd_c2_sa_casscf48_aug_rydberg_mo.npy")
OUTPUT_PREFIX = Path("chd_c2_casscf_real_density_ued")
# Paper molecular frame: x is out of the conjugated plane, z is the C2 axis,
# and y is the remaining in-plane direction. These are not XYZ Cartesian axes.
ROOTS = {"ground": 0, "3px": 2, "3py": 3}
MIXTURE = {"3px": 1.0 / 1.8, "3py": 0.8 / 1.8}


def fibonacci_sphere(npoints):
    index = np.arange(npoints, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / npoints
    phi = np.pi * (3.0 - np.sqrt(5.0)) * index
    radius = np.sqrt(1.0 - z * z)
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def state_density(mc, state):
    core = mc.mo_coeff[:, : mc.ncore]
    active = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
    active_dm = fci.direct_spin0.make_rdm1(
        mc.ci[state], mc.ncas, mc.nelecas
    )
    return 2.0 * core @ core.T + active @ active_dm @ active.T


def amplitudes_in_batches(mol, densities, q_vectors, batch_size=64):
    charges = mol.atom_charges().astype(float)
    coords = mol.atom_coords()
    output = {
        label: {
            "sigma_nuc": np.empty(len(q_vectors), dtype=complex),
            "sigma_el": np.empty(len(q_vectors), dtype=complex),
        }
        for label in densities
    }
    for start in range(0, len(q_vectors), batch_size):
        stop = min(start + batch_size, len(q_vectors))
        q = q_vectors[start:stop]
        nuclear = np.einsum(
            "a,qa->q", charges, np.exp(-1j * np.einsum("ax,qx->qa", coords, q))
        )
        ao_pair = ft_ao.ft_aopair(mol, q, aosym="s1", return_complex=True)
        for label, density in densities.items():
            electronic = np.einsum("mn,qmn->q", density, ao_pair, optimize=True)
            output[label]["sigma_nuc"][start:stop] = nuclear
            output[label]["sigma_el"][start:stop] = -electronic
        if stop % 1024 < batch_size or stop == len(q_vectors):
            print(f"Fourier points {stop}/{len(q_vectors)}", flush=True)
    return output


def rotational_average(amplitudes, nq, ndir, s_bohr):
    result = {}
    born = 4.0 / s_bohr**4
    for label, values in amplitudes.items():
        nuc = values["sigma_nuc"].reshape(nq, ndir)
        elec = values["sigma_el"].reshape(nq, ndir)
        total = nuc + elec
        result[label] = {
            "I_nuc": np.mean(np.abs(nuc) ** 2, axis=1),
            "I_el": np.mean(np.abs(elec) ** 2, axis=1),
            "I_cross": np.mean(2.0 * np.real(nuc * np.conj(elec)), axis=1),
            "I_total": np.mean(np.abs(total) ** 2, axis=1),
        }
        result[label]["I_born"] = result[label]["I_total"] * born
    return result


def main():
    metadata = json.loads(SA_DATA.read_text(encoding="utf-8"))
    atoms = read_xyz(GEOMETRY)
    mol = gto.M(
        atom=atoms + [("X", tuple(metadata["rydberg_center_angstrom"]))],
        basis={"C": "aug-cc-pvdz", "H": "aug-cc-pvdz", "X": RYDBERG_BASIS},
        unit="Angstrom",
        charge=0,
        spin=0,
        symmetry="C2",
        verbose=4,
        output=f"{OUTPUT_PREFIX}.log",
        max_memory=6000,
    )
    mf = scf.RHF(mol).density_fit().run(conv_tol=1.0e-10)
    mc = mcscf.CASSCF(mf, 8, 4).density_fit()
    mc.fcisolver = fci.direct_spin0.FCI(mol)
    mc.conv_tol = 1.0e-8
    mc.max_cycle_macro = 100
    mc = mc.state_average_([1.0 / 7.0] * 7)
    mc.kernel(np.load(SA_MO))
    if not mc.converged:
        raise RuntimeError("SA-CASSCF did not converge")

    densities = {label: state_density(mc, root) for label, root in ROOTS.items()}
    electron_counts = {
        label: float(np.einsum("ij,ji->", density, mf.get_ovlp()))
        for label, density in densities.items()
    }
    s_angstrom = np.linspace(0.15, 8.0, 240)
    s_bohr = s_angstrom * au2angstrom
    directions = fibonacci_sphere(38)
    q_vectors = (s_bohr[:, None, None] * directions[None, :, :]).reshape(-1, 3)
    amplitudes = amplitudes_in_batches(mol, densities, q_vectors)
    averaged = rotational_average(amplitudes, len(s_angstrom), len(directions), s_bohr)

    mixture_born = sum(MIXTURE[state] * averaged[state]["I_born"] for state in MIXTURE)
    ground_born = averaged["ground"]["I_born"]
    difference = {
        state: 100.0 * (averaged[state]["I_born"] - ground_born) / ground_born
        for state in ("3px", "3py")
    }
    difference["mixture"] = 100.0 * (mixture_born - ground_born) / ground_born

    # Conventional modified molecular signal. electron_atomic_form_factor is
    # proportional to (Z-f_x)/s^2, so multiplying by s^2 gives the neutral-atom
    # charge amplitude in the same normalization as sigma_nuc + sigma_el.
    symbols = [atom[0] for atom in atoms]
    atomic_charge_amplitudes = np.vstack(
        [
            electron_atomic_form_factor(symbol, s_angstrom, q_unit="angstrom^-1")
            * s_angstrom**2
            for symbol in symbols
        ]
    )
    i_atomic_charge = np.sum(atomic_charge_amplitudes**2, axis=0)
    sm = {
        state: s_angstrom
        * (averaged[state]["I_total"] - i_atomic_charge)
        / i_atomic_charge
        for state in ROOTS
    }
    mixture_total = sum(
        MIXTURE[state] * averaged[state]["I_total"] for state in MIXTURE
    )
    sm["mixture"] = s_angstrom * (mixture_total - i_atomic_charge) / i_atomic_charge
    delta_sm = {
        state: sm[state] - sm["ground"] for state in ("3px", "3py", "mixture")
    }

    fields = ["s_angstrom-1"]
    for state in ROOTS:
        fields.extend(
            [
                f"{state}_I_nuclear",
                f"{state}_I_electronic",
                f"{state}_I_interference",
                f"{state}_I_total",
                f"{state}_I_born",
            ]
        )
    fields.extend(
        [
            "I_atomic_charge_background",
            "ground_sM",
            "3px_sM",
            "3py_sM",
            "mixture_sM",
            "3px_delta_sM",
            "3py_delta_sM",
            "mixture_delta_sM",
            "3px_delta_percent",
            "3py_delta_percent",
            "mixture_delta_percent",
        ]
    )
    with Path(f"{OUTPUT_PREFIX}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, s_value in enumerate(s_angstrom):
            row = {"s_angstrom-1": s_value}
            for state in ROOTS:
                row.update(
                    {
                        f"{state}_I_nuclear": averaged[state]["I_nuc"][i],
                        f"{state}_I_electronic": averaged[state]["I_el"][i],
                        f"{state}_I_interference": averaged[state]["I_cross"][i],
                        f"{state}_I_total": averaged[state]["I_total"][i],
                        f"{state}_I_born": averaged[state]["I_born"][i],
                    }
                )
            row.update(
                {
                    "I_atomic_charge_background": i_atomic_charge[i],
                    "ground_sM": sm["ground"][i],
                    "3px_sM": sm["3px"][i],
                    "3py_sM": sm["3py"][i],
                    "mixture_sM": sm["mixture"][i],
                    "3px_delta_sM": delta_sm["3px"][i],
                    "3py_delta_sM": delta_sm["3py"][i],
                    "mixture_delta_sM": delta_sm["mixture"][i],
                    "3px_delta_percent": difference["3px"][i],
                    "3py_delta_percent": difference["3py"][i],
                    "mixture_delta_percent": difference["mixture"][i],
                }
            )
            writer.writerow(row)

    fig, axes = plt.subplots(2, 1, figsize=(7.1, 6.2), sharex=True)
    axes[0].plot(s_angstrom, sm["ground"], color="black", lw=1.3, label="ground")
    axes[0].plot(s_angstrom, sm["3px"], color="#0072B2", lw=1.1, label=r"$3p_x$")
    axes[0].plot(s_angstrom, sm["3py"], color="#D55E00", lw=1.1, ls="--", label=r"$3p_y$")
    axes[0].axhline(0.0, color="0.7", lw=0.7)
    axes[0].set_ylabel(r"$sM(s)$")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].text(-0.10, 1.02, "a", transform=axes[0].transAxes, fontweight="bold")
    axes[1].plot(s_angstrom, delta_sm["3px"], color="#0072B2", lw=1.1, label=r"$3p_x-X$")
    axes[1].plot(s_angstrom, delta_sm["3py"], color="#D55E00", lw=1.1, ls="--", label=r"$3p_y-X$")
    axes[1].plot(s_angstrom, delta_sm["mixture"], color="#009E73", lw=1.5, label="1:0.8 mixture")
    axes[1].axhline(0.0, color="0.7", lw=0.7)
    axes[1].set(xlabel=r"$s$ ($\mathrm{\AA}^{-1}$)", ylabel=r"electronic $\Delta sM(s)$")
    axes[1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    axes[1].text(-0.10, 1.02, "b", transform=axes[1].transAxes, fontweight="bold")
    for axis in axes:
        axis.set_xlim(s_angstrom[0], s_angstrom[-1])
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.20, hspace=0.10)
    fig.savefig(f"{OUTPUT_PREFIX}.pdf")
    fig.savefig(f"{OUTPUT_PREFIX}.png", dpi=350)
    plt.close(fig)

    summary = {
        "method": "SA-CASSCF(4,8)/aug-cc-pVDZ + Kaufmann 3s/3p explicit density",
        "geometry": str(GEOMETRY),
        "state_label_convention": {
            "source": "Yong/Ruddock CHD 200 nm papers",
            "x": "normal to the conjugated-carbon plane (out of plane)",
            "z": "molecular C2 axis",
            "y": "in-plane axis completing a right-handed x,y,z frame",
            "warning": "molecular-frame labels, not stored-XYZ Cartesian axes",
        },
        "roots": ROOTS,
        "population_ratio": "3px:3py = 1:0.8",
        "orientation_directions": len(directions),
        "electron_counts": electron_counts,
        "mixture_peak_abs_delta_percent": float(np.max(np.abs(difference["mixture"]))),
        "mixture_peak_s_angstrom-1": float(s_angstrom[np.argmax(np.abs(difference["mixture"]))]),
        "mixture_peak_abs_delta_sM": float(np.max(np.abs(delta_sm["mixture"]))),
        "mixture_peak_delta_sM_s_angstrom-1": float(
            s_angstrom[np.argmax(np.abs(delta_sm["mixture"]))]
        ),
    }
    Path(f"{OUTPUT_PREFIX}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
