"""Analytic paper-frame CHD 3px/3py off-diagonal vibronic couplings."""

import json
from pathlib import Path

import numpy as np
from pyscf import fci, gto, mcscf, scf
from pyscf.nac import sacasscf as sacasscf_nac

from chd_sa_casscf48_aug_rydberg import RYDBERG_BASIS, read_xyz


GEOMETRY = Path("chd_c2_casscf44_augccpvdz.xyz")
MODE_DATA = Path("chd_c2_b3lyp_augccpvdz_normal_modes.npz")
SA_DATA = Path("chd_c2_sa_casscf48_aug_rydberg.json")
SA_MO = Path("chd_c2_sa_casscf48_aug_rydberg_mo.npy")
OUTPUT_PREFIX = Path("chd_c2_3px_3py_offdiagonal_vibronic")
ROOT_X = 2
ROOT_Y = 3
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_CM1 = 219474.6313632


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
        raise RuntimeError("restored SA-CASSCF calculation did not converge")

    driver = sacasscf_nac.NonAdiabaticCouplings(mc)
    # With mult_ediff=True the returned quantity is <bra|dH/dR|ket>, not
    # <bra|d/dR|ket>. Full (non-ETF) terms are appropriate for an internal
    # molecular vibronic Hamiltonian.
    h_xy = driver.kernel(state=(ROOT_Y, ROOT_X), mult_ediff=True, use_etfs=False)
    h_yx = driver.kernel(state=(ROOT_X, ROOT_Y), mult_ediff=True, use_etfs=False)
    h_symmetric = 0.5 * (h_xy + h_yx)
    h_antisymmetric = 0.5 * (h_xy - h_yx)

    with np.load(MODE_DATA) as mode_data:
        modes = mode_data["normal_modes"]
        frequencies = mode_data["frequencies_cm1"]
    # The floating Rydberg center is not a physical vibrational coordinate.
    lambda_xy = np.einsum("kax,ax->k", modes, h_symmetric[:14], optimize=True)
    lambda_xy_forward = np.einsum("kax,ax->k", modes, h_xy[:14], optimize=True)
    lambda_xy_reverse = np.einsum("kax,ax->k", modes, h_yx[:14], optimize=True)

    ordering = np.argsort(np.abs(lambda_xy))[::-1]
    selected = [5, 8, 13, 27]
    result = {
        "method": "analytic seven-state SA-CASSCF(4,8) Hamiltonian derivative coupling",
        "basis": "aug-cc-pVDZ + Kaufmann 3s/3p",
        "state_labels": "paper frame: root 2 = 3px out of plane; root 3 = 3py in plane",
        "root_pair": [ROOT_X, ROOT_Y],
        "sa_casscf_gap_ev": float((mc.e_states[ROOT_Y] - mc.e_states[ROOT_X]) * HARTREE_TO_EV),
        "multiplied_by_energy_difference": True,
        "use_electron_translation_factors": False,
        "floating_center_fixed": True,
        "cartesian_hermiticity_rms_Eh_per_bohr": float(np.sqrt(np.mean(h_antisymmetric[:14] ** 2))),
        "cartesian_hermiticity_max_Eh_per_bohr": float(np.max(np.abs(h_antisymmetric[:14]))),
        "ghost_symmetric_coupling_norm_Eh_per_bohr": float(np.linalg.norm(h_symmetric[14])),
        "selected_modes": {
            str(mode): {
                "frequency_cm-1": float(frequencies[mode - 1]),
                "lambda_Eh_per_bohr_sqrtamu": float(lambda_xy[mode - 1]),
                "lambda_cm-1_per_bohr_sqrtamu": float(lambda_xy[mode - 1] * HARTREE_TO_CM1),
                "forward": float(lambda_xy_forward[mode - 1]),
                "reverse": float(lambda_xy_reverse[mode - 1]),
            }
            for mode in selected
        },
        "top_modes_by_abs_lambda": [
            {
                "mode": int(index + 1),
                "frequency_cm-1": float(frequencies[index]),
                "lambda_Eh_per_bohr_sqrtamu": float(lambda_xy[index]),
                "abs_lambda_Eh_per_bohr_sqrtamu": float(abs(lambda_xy[index])),
            }
            for index in ordering[:15]
        ],
        "all_lambda_Eh_per_bohr_sqrtamu": lambda_xy.tolist(),
        "all_lambda_forward": lambda_xy_forward.tolist(),
        "all_lambda_reverse": lambda_xy_reverse.tolist(),
    }
    np.savez_compressed(
        f"{OUTPUT_PREFIX}.npz",
        h_xy_cartesian=h_xy,
        h_yx_cartesian=h_yx,
        h_symmetric_cartesian=h_symmetric,
        lambda_xy=lambda_xy,
        lambda_xy_forward=lambda_xy_forward,
        lambda_xy_reverse=lambda_xy_reverse,
        frequencies_cm1=frequencies,
    )
    Path(f"{OUTPUT_PREFIX}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
