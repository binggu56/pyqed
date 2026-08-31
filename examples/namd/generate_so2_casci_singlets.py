#!/usr/bin/env python3
"""Generate spin-fixed SO2 CASCI singlet energies and wavefunction overlaps."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import au2ev
from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.casci import CASCI, overlap as casci_overlap


def electronic_metadata(options):
    """Return the electronic-model signature stored with every SO2 dataset."""
    return {
        "basis": np.asarray(str(options.basis)),
        "nstates": np.asarray(int(options.nstates)),
        "ncas": np.asarray(int(options.ncas)),
        "nelecas": np.asarray(int(options.nelecas)),
        "ms2": np.asarray(0),
        "multiplicity": np.asarray(1),
        "spin_root_cushion": np.asarray(int(options.spin_root_cushion)),
        "coordinate_frame": np.asarray("bond-bisector-c2x"),
    }


def validate_electronic_metadata(archive, options, *, label="SO2 dataset"):
    """Reject caches generated with a different electronic Hamiltonian."""
    expected = electronic_metadata(options)
    missing = [name for name in expected if name not in archive]
    if missing:
        raise ValueError(
            f"{label} lacks electronic metadata {missing}; regenerate it"
        )
    mismatches = []
    for name, value in expected.items():
        actual = np.asarray(archive[name]).item()
        wanted = np.asarray(value).item()
        if name == "basis":
            equal = str(actual).lower() == str(wanted).lower()
        else:
            equal = actual == wanted
        if not equal:
            mismatches.append(f"{name}={actual!r} (requested {wanted!r})")
    if mismatches:
        raise ValueError(f"{label} electronic model differs: " + ", ".join(mismatches))


def require_spin_pure_singlets(spin_square, *, tolerance=1.0e-7):
    """Require all selected CASCI roots to satisfy the singlet constraint."""
    maximum = float(np.max(np.abs(np.asarray(spin_square, dtype=float))))
    if not np.isfinite(maximum) or maximum > float(tolerance):
        raise RuntimeError(
            "Spin-pure singlet selection failed: "
            f"max |<S^2>|={maximum:.3e} > {float(tolerance):.3e}"
        )
    return maximum


def geometry(r1, r2, theta):
    """Return the SO2 geometry in a bond-bisector body-fixed frame."""

    half = 0.5 * float(theta)
    return np.asarray(
        [
            [r1 * np.cos(half), r1 * np.sin(half), 0.0],
            [0.0, 0.0, 0.0],
            [r2 * np.cos(half), -r2 * np.sin(half), 0.0],
        ]
    )


def ao_diagonal_symmetry_operator(molecule, signs, *, tolerance=1.0e-10):
    """Return an AO representation for a diagonal Cartesian point operation."""

    def same_array(left, right):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        return left.shape == right.shape and np.allclose(
            left, right, atol=tolerance, rtol=0.0
        )

    basis = list(molecule._bas)
    if len(basis) != int(molecule.nao):
        raise NotImplementedError(
            "SO2 point-group lifting currently requires native Cartesian AOs"
        )
    signs = np.asarray(signs, dtype=float)
    if signs.shape != (3,) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("Cartesian symmetry signs must contain three values +/-1")
    operator = np.zeros((len(basis), len(basis)))
    for source, basis_fn in enumerate(basis):
        shell = tuple(int(value) for value in basis_fn.shell)
        target_origin = signs * np.asarray(basis_fn.origin, dtype=float)
        exponents = np.asarray(basis_fn.exps, dtype=float)
        coefficients = np.asarray(basis_fn.coefs, dtype=float)
        matches = [
            target
            for target, candidate in enumerate(basis)
            if tuple(int(value) for value in candidate.shell) == shell
            and np.allclose(candidate.origin, target_origin, atol=tolerance, rtol=0.0)
            and same_array(candidate.exps, exponents)
            and same_array(candidate.coefs, coefficients)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"could not uniquely map AO {source} under the point operation"
            )
        operator[matches[0], source] = float(
            np.prod(signs ** np.asarray(shell))
        )
    return operator


def ao_c2x_exchange_operator(molecule, *, tolerance=1.0e-10):
    r"""Return the AO representation of $C_2(x):(x,y,z)\mapsto(x,-y,-z)$."""

    return ao_diagonal_symmetry_operator(
        molecule, (1.0, -1.0, -1.0), tolerance=tolerance
    )


def electronic_symmetry_representation(model, signs, *, tolerance=1.0e-5):
    """Lift one Cartesian point operation into the selected CASCI manifold."""

    ao_operator = ao_diagonal_symmetry_operator(model.mol, signs)
    ao_metric = np.asarray(model.mf.get_ovlp())
    orbitals = np.asarray(model.mo_coeff)
    mo_operator = orbitals.conj().T @ ao_metric @ ao_operator @ orbitals
    raw = casci_overlap(model.frame(), model.frame(), s=mo_operator)
    raw = 0.5 * (raw + raw.conj().T)
    diagonal = np.real(np.diag(raw))
    parities = np.where(diagonal >= 0.0, 1.0, -1.0)
    representation = np.diag(parities).astype(complex)
    off_diagonal = raw - np.diag(np.diag(raw))
    diagonal_defect = float(np.max(np.abs(np.abs(diagonal) - 1.0)))
    off_diagonal_max = float(np.max(np.abs(off_diagonal)))
    if diagonal_defect > float(tolerance) or off_diagonal_max > float(tolerance):
        raise RuntimeError(
            "CASCI roots are not resolved into one-dimensional symmetry sectors: "
            f"diagonal defect={diagonal_defect:.3e}, off-diagonal={off_diagonal_max:.3e}"
        )
    ao_metric_defect = float(
        np.linalg.norm(ao_operator.conj().T @ ao_metric @ ao_operator - ao_metric)
    )
    return representation, raw, {
        "ao_metric_defect": ao_metric_defect,
        "state_involution_defect": float(
            np.linalg.norm(raw @ raw - np.eye(len(raw)))
        ),
        "state_off_diagonal_max": off_diagonal_max,
        "state_diagonal_defect": diagonal_defect,
    }


def electronic_exchange_representation(model, *, tolerance=1.0e-5):
    r"""Lift oxygen exchange $C_2(x)$ into the selected CASCI manifold."""

    return electronic_symmetry_representation(
        model, (1.0, -1.0, -1.0), tolerance=tolerance
    )


def so2_point_group_representations(r, theta, options):
    """Return all four $C_{2v}$ representations in the CASCI state manifold."""

    model = electronic_structure(float(r), float(r), float(theta), options)
    operations = {
        "E": (1.0, 1.0, 1.0),
        "C2(x)": (1.0, -1.0, -1.0),
        "sigma_xy": (1.0, 1.0, -1.0),
        "sigma_xz": (1.0, -1.0, 1.0),
    }
    representations = []
    raw = []
    diagnostics = []
    for signs in operations.values():
        representation, lifted, info = electronic_symmetry_representation(
            model, signs
        )
        representations.append(representation)
        raw.append(lifted)
        diagnostics.append(info)
    representations = np.asarray(representations)
    group_defect = float(
        np.linalg.norm(representations[1] @ representations[2] - representations[3])
    )
    return tuple(operations), representations, np.asarray(raw), {
        "operations": diagnostics,
        "generator_product_defect": group_defect,
    }


def so2_exchange_representation(r, theta, options):
    """Run one symmetric SO2 calculation and return its exchange sectors."""

    _names, representations, raw, diagnostics = so2_point_group_representations(
        r, theta, options
    )
    return representations[1], raw[1], diagnostics["operations"][1]


def electronic_structure(r1, r2, theta, args):
    xyz = geometry(float(r1), float(r2), float(theta))
    molecule = Molecule(
        atom=[
            [symbol, tuple(position)]
            for symbol, position in zip(("O", "S", "O"), xyz)
        ],
        charge=0,
        spin=0,
        unit="bohr",
        basis=args.basis,
    )
    molecule.build(eri="dense")
    reference = RHF(molecule).run(
        tol=args.scf_tol,
        max_cycle=args.max_cycle,
        verbose=0,
    )
    if not reference.converged:
        raise RuntimeError("RHF did not converge")
    casci = CASCI(
        reference,
        ncas=args.ncas,
        nelecas=args.nelecas,
        ms2=0,
        multiplicity=1,
    )
    casci.spin_root_cushion = args.spin_root_cushion
    return casci.run(nstates=args.nstates, method="direct_ci")


def generate(args):
    r1 = np.linspace(args.r_min, args.r_max, args.n_r)
    r2 = np.linspace(args.r_min, args.r_max, args.n_r)
    theta = np.deg2rad(
        np.linspace(args.theta_min_deg, args.theta_max_deg, args.n_theta)
    )
    shape = (len(r1), len(r2), len(theta))
    models = np.empty(shape, dtype=object)
    energies = np.empty((*shape, args.nstates))
    spin_square = np.empty_like(energies)
    started = time.perf_counter()
    total = int(np.prod(shape))
    for count, index in enumerate(np.ndindex(shape), start=1):
        model = electronic_structure(
            r1[index[0]], r2[index[1]], theta[index[2]], args
        )
        models[index] = model.frame()
        energies[index] = np.asarray(model.e_tot)
        spin_square[index] = [model.spin_square(state) for state in range(args.nstates)]
        if count == 1 or count % args.progress_every == 0 or count == total:
            print(
                f"[CASCI] {count}/{total}, E0={energies[index][0]:.10f} Eh, "
                f"max |S2|={np.max(np.abs(spin_square[index])):.2e}, "
                f"elapsed={time.perf_counter() - started:.1f} s",
                flush=True,
            )

    links = []
    for axis in range(3):
        edge_shape = list(shape)
        edge_shape[axis] -= 1
        values = np.empty((*edge_shape, args.nstates, args.nstates), dtype=complex)
        for left in np.ndindex(tuple(edge_shape)):
            right = list(left)
            right[axis] += 1
            values[left] = models[left].overlap(models[tuple(right)])
        links.append(values)
    return (r1, r2, theta), energies, spin_square, tuple(links)


def plot_dataset(grids, energies, spin_square, links, filename):
    r1, _r2, theta = grids
    center = tuple(len(grid) // 2 for grid in grids)
    relative = (energies - energies[..., :1].min()) * au2ev
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), constrained_layout=True)
    for state in range(energies.shape[-1]):
        axes[0].plot(
            r1,
            relative[:, center[1], center[2], state],
            marker="o",
            label=f"S{state}",
        )
        axes[1].plot(
            np.rad2deg(theta),
            relative[center[0], center[1], :, state],
            marker="o",
        )
    axes[0].set(xlabel=r"$r_1$ (bohr)", ylabel="Relative energy (eV)")
    axes[1].set(xlabel=r"$\theta$ (degree)", ylabel="Relative energy (eV)")
    axes[0].legend(frameon=False)
    for axis, values in enumerate(links):
        singular_values = np.linalg.svd(values, compute_uv=False)
        axes[2].hist(
            singular_values.ravel(), bins=24, histtype="step", label=f"axis {axis}"
        )
    axes[2].set(xlabel="Link singular value", ylabel="Count")
    axes[2].legend(frameon=False)
    axes[2].text(
        0.98,
        0.04,
        rf"max $|\langle S^2\rangle|={np.max(np.abs(spin_square)):.1e}$",
        transform=axes[2].transAxes,
        ha="right",
    )
    for label, axis in zip("abc", axes):
        axis.text(
            0.02,
            0.98,
            label,
            transform=axis.transAxes,
            va="top",
            fontweight="bold",
        )
        axis.spines[["top", "right"]].set_visible(False)
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(filename, dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-r", type=int, default=5)
    parser.add_argument("--n-theta", type=int, default=5)
    parser.add_argument("--r-min", type=float, default=2.68)
    parser.add_argument("--r-max", type=float, default=2.92)
    parser.add_argument("--theta-min-deg", type=float, default=110.0)
    parser.add_argument("--theta-max-deg", type=float, default=130.0)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--ncas", type=int, default=6)
    parser.add_argument("--nelecas", type=int, default=6)
    parser.add_argument("--spin-root-cushion", type=int, default=8)
    parser.add_argument("--scf-tol", type=float, default=1.0e-10)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/so2_casci_singlet_5x5x5.npz"),
    )
    args = parser.parse_args()
    grids, energies, spin_square, links = generate(args)
    require_spin_pure_singlets(spin_square)
    point_group_names, point_group, point_group_raw, point_group_diagnostics = (
        so2_point_group_representations(
        0.5 * (args.r_min + args.r_max),
        np.deg2rad(0.5 * (args.theta_min_deg + args.theta_max_deg)),
        args,
        )
    )
    exchange = point_group[1]
    exchange_raw = point_group_raw[1]
    exchange_diagnostics = point_group_diagnostics["operations"][1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        r1=grids[0],
        r2=grids[1],
        theta=grids[2],
        energies=energies,
        spin_square=spin_square,
        links_0=links[0],
        links_1=links[1],
        links_2=links[2],
        exchange_representation=exchange,
        exchange_raw=exchange_raw,
        exchange_ao_metric_defect=np.asarray(
            exchange_diagnostics["ao_metric_defect"]
        ),
        exchange_state_involution_defect=np.asarray(
            exchange_diagnostics["state_involution_defect"]
        ),
        exchange_state_off_diagonal_max=np.asarray(
            exchange_diagnostics["state_off_diagonal_max"]
        ),
        exchange_state_diagonal_defect=np.asarray(
            exchange_diagnostics["state_diagonal_defect"]
        ),
        point_group_names=np.asarray(point_group_names),
        point_group_representations=point_group,
        point_group_raw=point_group_raw,
        point_group_ao_metric_defects=np.asarray([
            value["ao_metric_defect"]
            for value in point_group_diagnostics["operations"]
        ]),
        point_group_state_involution_defects=np.asarray([
            value["state_involution_defect"]
            for value in point_group_diagnostics["operations"]
        ]),
        point_group_state_off_diagonal_max=np.asarray([
            value["state_off_diagonal_max"]
            for value in point_group_diagnostics["operations"]
        ]),
        point_group_generator_product_defect=np.asarray(
            point_group_diagnostics["generator_product_defect"]
        ),
        source=np.asarray(
            f"SO2 native {args.basis} CASCI({args.nelecas}e,{args.ncas}o), "
            f"Ms=0, multiplicity=1"
        ),
        **electronic_metadata(args),
    )
    figure = args.output.with_suffix(".png")
    plot_dataset(grids, energies, spin_square, links, figure)
    print(f"dataset: {args.output}")
    print(f"figure: {figure}")


if __name__ == "__main__":
    main()
