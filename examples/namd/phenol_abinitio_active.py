#!/usr/bin/env python3
"""Restartable multistate phenol data generation and active MACE fitting.

The electronic-structure demonstration uses RHF/TDA because it is affordable
for a local smoke calculation.  The dataset records Cartesian geometries,
state energies, signed cross-geometry state overlaps, partial charges, aligned
Hamiltonians, and QCSchema-compatible molecule records.  The same container
can be populated by a higher-level CASCI/CASSCF provider later.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, gto, scf, tdscf
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from pyqed.ml import MACE
from pyqed.models.phenol_coordinates import (
    PHENOL_MASSES,
    PHENOL_SPECIES,
    PhenolReactiveChart,
    select_phenol_active_modes,
)
from pyqed.qchem.dft.hessian import analyze_cartesian_hessian
from pyqed.units import au2angstrom, au2ev


def symmetric_basis(nstates):
    basis = []
    for state in range(int(nstates)):
        value = np.zeros((nstates, nstates))
        value[state, state] = 1.0
        basis.append(value)
    for left in range(int(nstates)):
        for right in range(left + 1, int(nstates)):
            value = np.zeros((nstates, nstates))
            value[left, right] = value[right, left] = 1.0
            basis.append(value)
    return np.asarray(basis)


def symmetric_coefficients(matrices):
    matrices = np.asarray(matrices)
    output = [matrices[:, state, state].real for state in range(matrices.shape[-1])]
    for left in range(matrices.shape[-1]):
        for right in range(left + 1, matrices.shape[-1]):
            output.append(matrices[:, left, right].real)
    return np.column_stack(output)


def sampling_bounds(chart):
    bounds = chart.default_bounds.copy()
    bounds[0] = (0.88, 1.55)
    bounds[1] = (-0.55, 0.55)
    bounds[2] = (np.deg2rad(99.0), np.deg2rad(119.0))
    bounds[3] = (-0.10, 0.10)
    bounds[4] = (-0.08, 0.08)
    return bounds


def hessian_modes(args):
    """Calculate normal modes and select Wilson 16a/8a directions."""

    args.output.mkdir(parents=True, exist_ok=True)
    template_chart = PhenolReactiveChart()
    geometry = template_chart.geometry(template_chart.equilibrium)
    mol = gto.M(
        atom=list(zip(PHENOL_SPECIES, geometry)), basis=args.basis,
        unit="Angstrom", charge=0, spin=0, symmetry=False, verbose=0,
    )
    if args.method.lower() == "rhf":
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = str(args.method)
        mf.grids.level = int(args.grid_level)
    mf.run(conv_tol=1.0e-9)
    if not mf.converged:
        raise RuntimeError("reference failed before the phenol Hessian calculation")
    raw = np.asarray(mf.Hessian().kernel())
    cartesian = raw.transpose(0, 2, 1, 3).reshape(3 * mol.natm, 3 * mol.natm)
    analysis = analyze_cartesian_hessian(
        cartesian, mol.atom_coords(), mol.atom_mass_list(),
        remove_translation_rotation=True,
    )
    frequencies = np.asarray(analysis["freq_cm1"], dtype=float)
    modes = np.asarray(analysis["modes"], dtype=float)
    # Normalize to the mass metric used by PhenolReactiveChart.
    modes /= np.sqrt(
        np.einsum("kia,kia,i->k", modes, modes, PHENOL_MASSES)
    )[:, None, None]
    selected_modes, selection = select_phenol_active_modes(frequencies, modes)
    selected = [item["index"] for item in selection]
    output = args.output / "phenol_hessian_modes.npz"
    np.savez(
        output, modes=selected_modes, selected_indices=np.asarray(selected),
        selected_frequencies_cm1=frequencies[selected],
        frequencies_cm1=frequencies, all_modes=modes,
        equilibrium_geometry_angstrom=geometry, basis=np.asarray(args.basis),
        method=np.asarray(args.method),
        labels=np.asarray(("16a", "8a")),
        reflection_parities=np.asarray((-1, 1)),
    )
    figure, panel = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    panel.vlines(np.arange(len(frequencies)), 0.0, frequencies, color="#777777", lw=1)
    panel.scatter(selected, frequencies[selected], color=("#d55e00", "#0072b2"), zorder=3)
    panel.set(
        xlabel="vibrational mode index",
        ylabel=r"frequency (cm$^{-1}$)",
        title=f"{args.method}/{args.basis} phenol modes",
    )
    panel.spines[["top", "right"]].set_visible(False)
    figure.savefig(args.output / "phenol_hessian_modes.png", dpi=240)
    plt.close(figure)
    summary = {
        "method": args.method, "basis": args.basis,
        "selected_indices": list(map(int, selected)),
        "selected_frequencies_cm1": frequencies[selected].tolist(),
        "selection": selection,
        "output": str(output),
    }
    (args.output / "phenol_hessian_modes.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def sobol_coordinates(count, bounds, seed):
    count = int(count)
    if count < 1:
        raise ValueError("sample count must be positive")
    power = int(np.ceil(np.log2(max(count - 1, 1))))
    unit = qmc.Sobol(len(bounds), scramble=True, seed=int(seed)).random_base2(power)
    unit = unit[: max(count - 1, 0)]
    values = bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])
    equilibrium = PhenolReactiveChart().equilibrium[None, :]
    return np.vstack((equilibrium, values))[:count]


def _excited_density(mf, amplitude):
    amplitude = np.asarray(amplitude, dtype=float)
    amplitude /= np.linalg.norm(amplitude)
    occupied = mf.mo_occ > 0
    virtual = mf.mo_occ == 0
    density = np.diag(mf.mo_occ.copy())
    density[np.ix_(occupied, occupied)] -= amplitude @ amplitude.T
    density[np.ix_(virtual, virtual)] += amplitude.T @ amplitude
    return mf.mo_coeff @ density @ mf.mo_coeff.T


def calculate_point(coordinate, chart, basis, nstates, gradients=False):
    geometry = chart.geometry(coordinate)
    mol = gto.M(
        atom=list(zip(PHENOL_SPECIES, geometry)),
        basis=str(basis), unit="Angstrom", charge=0, spin=0,
        symmetry=False, verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-9
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        mf = mf.newton().run()
    if not mf.converged:
        raise RuntimeError("phenol RHF did not converge")
    tda = tdscf.TDA(mf)
    tda.nstates = int(nstates)
    tda.conv_tol = 1.0e-7
    tda.kernel()
    if not np.all(tda.converged):
        raise RuntimeError("phenol TDA did not converge")
    amplitudes = np.asarray([xy[0] for xy in tda.xy], dtype=float)
    lowdin = []
    for amplitude in amplitudes:
        density = _excited_density(mf, amplitude)
        _population, charges = scf.hf.mulliken_pop_meta_lowdin_ao(
            mol, density, verbose=0
        )
        lowdin.append(charges)
    excited_gradients = None
    if gradients:
        gradient_driver = tda.nuc_grad_method()
        excited_gradients = np.asarray(
            [gradient_driver.kernel(xy=xy) for xy in tda.xy]
        )
    return {
        "coordinate": np.asarray(coordinate),
        "geometry": geometry,
        "scf_energy": np.asarray(mf.e_tot),
        "energies": mf.e_tot + np.asarray(tda.e),
        "amplitudes": amplitudes,
        "mo_coeff": np.asarray(mf.mo_coeff),
        "mo_occ": np.asarray(mf.mo_occ),
        "lowdin_charges": np.asarray(lowdin),
        "gradients": excited_gradients,
    }


def save_record(path, record):
    values = {key: value for key, value in record.items() if value is not None}
    np.savez(path, **values)


def load_record(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def overlap_matrix(left, right, basis):
    left_mol = gto.M(
        atom=list(zip(PHENOL_SPECIES, left["geometry"])), basis=basis,
        unit="Angstrom", charge=0, spin=0, symmetry=False, verbose=0,
    )
    right_mol = gto.M(
        atom=list(zip(PHENOL_SPECIES, right["geometry"])), basis=basis,
        unit="Angstrom", charge=0, spin=0, symmetry=False, verbose=0,
    )
    ao = gto.intor_cross("int1e_ovlp", left_mol, right_mol)
    left_occ = left["mo_occ"] > 0
    right_occ = right["mo_occ"] > 0
    occupied = left["mo_coeff"][:, left_occ].T @ ao @ right["mo_coeff"][:, right_occ]
    virtual = left["mo_coeff"][:, ~left_occ].T @ ao @ right["mo_coeff"][:, ~right_occ]
    return np.einsum(
        "sia,ij,tjb,ab->st",
        left["amplitudes"], occupied, right["amplitudes"], virtual,
        optimize=True,
    )


def sparse_graph(coordinates, neighbors=3):
    coordinates = np.asarray(coordinates)
    scale = np.ptp(coordinates, axis=0)
    scale[scale < 1.0e-12] = 1.0
    distance = cdist(coordinates / scale, coordinates / scale)
    finite = distance.copy()
    np.fill_diagonal(finite, 0.0)
    tree = minimum_spanning_tree(finite).tocoo()
    pairs = {tuple(sorted((int(i), int(j)))) for i, j in zip(tree.row, tree.col)}
    np.fill_diagonal(distance, np.inf)
    count = min(max(int(neighbors), 1), len(coordinates) - 1)
    for left in range(len(coordinates)):
        for right in np.argpartition(distance[left], count - 1)[:count]:
            pairs.add(tuple(sorted((left, int(right)))))
    pairs = np.asarray(sorted(pairs), dtype=int)
    lengths = np.linalg.norm(
        coordinates[pairs[:, 0]] / scale - coordinates[pairs[:, 1]] / scale,
        axis=1,
    )
    return pairs, lengths


def polar_unitary(matrix):
    left, _singular, right_h = np.linalg.svd(matrix)
    return left @ right_h


def align_hamiltonians(energies, pairs, overlaps, lengths):
    npoints, nstates = energies.shape
    gauges = np.zeros((npoints, nstates, nstates), dtype=complex)
    gauges[0] = np.eye(nstates)
    adjacency = [[] for _ in range(npoints)]
    for edge, ((left, right), length) in enumerate(zip(pairs, lengths)):
        adjacency[left].append((length, right, edge, False))
        adjacency[right].append((length, left, edge, True))
    queue = [0]
    visited = {0}
    residuals = []
    while queue:
        left = queue.pop(0)
        for _length, right, edge, reverse in sorted(adjacency[left]):
            if right in visited:
                continue
            link = overlaps[edge].conj().T if reverse else overlaps[edge]
            transformed = gauges[left].conj().T @ link
            gauges[right] = polar_unitary(transformed).conj().T
            residuals.append(np.linalg.norm(gauges[left].conj().T @ link @ gauges[right] - np.eye(nstates)))
            visited.add(right)
            queue.append(right)
    if len(visited) != npoints:
        raise RuntimeError("overlap graph is disconnected")
    shift = float(np.min(energies[0]))
    matrices = np.einsum(
        "nia,ni,nib->nab", gauges.conj(), energies - shift, gauges, optimize=True
    )
    matrices = 0.5 * (matrices + matrices.conj().swapaxes(-1, -2))
    return matrices, gauges, shift, np.asarray(residuals)


def write_qcschema(path, coordinates, geometries, energies, basis, method):
    bohr = geometries / au2angstrom
    with Path(path).open("w") as stream:
        for index, (coordinate, geometry, levels) in enumerate(
            zip(coordinates, bohr, energies)
        ):
            record = {
                "schema_name": "qcschema_molecule",
                "schema_version": 2,
                "symbols": list(PHENOL_SPECIES),
                "geometry": geometry.reshape(-1).tolist(),
                "molecular_charge": 0.0,
                "molecular_multiplicity": 1,
                "extras": {
                    "pyqed_coordinate": coordinate.tolist(),
                    "pyqed_multistate": {
                        "method": method,
                        "basis": basis,
                        "state_energies_hartree": levels.tolist(),
                        "record_index": index,
                    },
                },
            }
            stream.write(json.dumps(record) + "\n")


def generate(args):
    modes = None
    if args.modes is not None:
        with np.load(args.modes, allow_pickle=False) as archive:
            modes = np.asarray(archive["modes"])
    chart = PhenolReactiveChart(modes=modes)
    if args.coordinates is None:
        coordinates = sobol_coordinates(args.samples, sampling_bounds(chart), args.seed)
        if args.local_scale != 1.0:
            coordinates = chart.equilibrium + float(args.local_scale) * (
                coordinates - chart.equilibrium
            )
    else:
        with np.load(args.coordinates, allow_pickle=False) as archive:
            coordinates = np.asarray(archive["coordinates"])
    args.output.mkdir(parents=True, exist_ok=True)
    record_dir = args.output / "records"
    record_dir.mkdir(exist_ok=True)
    records = []
    for index, coordinate in enumerate(coordinates):
        path = record_dir / f"point_{index:05d}.npz"
        if path.exists():
            record = load_record(path)
            if not np.allclose(record["coordinate"], coordinate):
                raise ValueError(f"cached coordinate mismatch in {path}")
        else:
            print(f"[RHF/TDA] {index + 1}/{len(coordinates)}", flush=True)
            record = calculate_point(
                coordinate, chart, args.basis, args.nstates, args.gradients
            )
            save_record(path, record)
        records.append(record)
    pairs, lengths = sparse_graph(coordinates, args.neighbors)
    overlaps = np.asarray(
        [overlap_matrix(records[left], records[right], args.basis) for left, right in pairs]
    )
    energies = np.asarray([record["energies"] for record in records])
    matrices, gauges, shift, residuals = align_hamiltonians(
        energies, pairs, overlaps, lengths
    )
    geometries = np.asarray([record["geometry"] for record in records])
    output = args.output / "phenol_multistate_abinitio.npz"
    payload = {
        "coordinates": coordinates,
        "coordinate_names": np.asarray(chart.names),
        "coordinate_bounds": sampling_bounds(chart),
        "geometries": geometries,
        "species": np.asarray(PHENOL_SPECIES),
        "energies": energies,
        "energy_shift": np.asarray(shift),
        "aligned_hamiltonians": matrices,
        "gauges": gauges,
        "overlap_pairs": pairs,
        "overlap_values": overlaps,
        "overlap_lengths": lengths,
        "lowdin_charges": np.asarray([record["lowdin_charges"] for record in records]),
        "normal_modes": chart.modes,
        "method": np.asarray("RHF/TDA"),
        "basis": np.asarray(args.basis),
    }
    if args.gradients:
        payload["gradients"] = np.asarray([record["gradients"] for record in records])
    np.savez(output, **payload)
    schema = args.output / "phenol_multistate_qcschema.jsonl"
    write_qcschema(schema, coordinates, geometries, energies, args.basis, "RHF/TDA")
    summary = {
        "samples": len(coordinates), "states": args.nstates,
        "method": f"RHF/TDA/{args.basis}", "overlap_edges": len(pairs),
        "maximum_tree_alignment_residual": float(np.max(residuals, initial=0.0)),
        "minimum_overlap_singular_value": float(np.min(np.linalg.svd(overlaps, compute_uv=False))),
        "dataset": str(output), "qcschema": str(schema),
    }
    (args.output / "phenol_multistate_abinitio.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def _fit_one(data, chart, args, seed):
    coordinates = data["coordinates"]
    bounds = data["coordinate_bounds"]
    axes = tuple(np.linspace(lower, upper, 3) for lower, upper in bounds)
    nstates = data["aligned_hamiltonians"].shape[-1]
    fit = MACE(
        axes, PHENOL_SPECIES, chart.geometry, nstates,
        chart_features=True, geometry_units="angstrom",
        channels=args.channels, max_ell=2, interactions=2, correlation=2,
        radial_basis=args.radial_basis,
        radial_mlp=(args.width, args.width), cutoff=args.cutoff,
    )
    basis = symmetric_basis(nstates)
    fit.fit_basis_h(
        coordinates,
        symmetric_coefficients(data["aligned_hamiltonians"]),
        basis,
        hidden=(args.width, args.width), epochs=args.epochs,
        learning_rate=args.learning_rate, seed=seed,
    )
    return fit


def fit_and_propose(args):
    with np.load(args.dataset, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    chart = PhenolReactiveChart(modes=data["normal_modes"])
    fits = [_fit_one(data, chart, args, args.seed + 101 * member) for member in range(args.ensemble)]
    predictions = np.asarray([fit.neural_energy.predict(data["coordinates"]) for fit in fits])
    mean = np.mean(predictions, axis=0)
    error = mean - data["aligned_hamiltonians"]
    rmse = float(np.sqrt(np.mean(np.abs(error) ** 2)) * au2ev * 1000.0)
    checkpoint = args.output / "phenol_abinitio_mace.pt"
    args.output.mkdir(parents=True, exist_ok=True)
    fits[0].save(checkpoint)
    bounds = data["coordinate_bounds"]
    pool = sobol_coordinates(args.pool, bounds, args.seed + 5003)
    ensemble = np.asarray([fit.neural_energy.predict(pool) for fit in fits])
    uncertainty = np.sqrt(np.mean(np.abs(ensemble - np.mean(ensemble, axis=0)) ** 2, axis=(0, 2, 3)))
    levels = np.linalg.eigvalsh(np.mean(ensemble, axis=0))
    gap = np.min(np.diff(levels, axis=1), axis=1)
    scaled = (pool - chart.equilibrium) / np.maximum(bounds[:, 1] - bounds[:, 0], 1.0e-12)
    dynamical_weight = np.exp(-2.0 * np.sum(scaled[:, [0, 2, 3, 4]] ** 2, axis=1))
    acquisition = uncertainty * dynamical_weight / np.maximum(gap, 2.0e-3)
    # Greedy exclusion keeps proposed calculations geometrically distinct.
    selected = []
    normalized = (pool - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    for candidate in np.argsort(acquisition)[::-1]:
        if not selected or np.min(cdist(normalized[[candidate]], normalized[selected])) > args.exclusion:
            selected.append(int(candidate))
        if len(selected) == args.proposals:
            break
    proposal_file = args.output / "phenol_active_proposals.npz"
    np.savez(
        proposal_file, coordinates=pool[selected], acquisition=acquisition[selected],
        uncertainty_hartree=uncertainty[selected], minimum_gap_hartree=gap[selected],
    )
    figure, panels = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    image = panels[0].scatter(pool[:, 0], np.rad2deg(pool[:, 1]), c=acquisition, s=9, cmap="magma")
    panels[0].scatter(pool[selected, 0], np.rad2deg(pool[selected, 1]), facecolors="none", edgecolors="cyan", s=45)
    figure.colorbar(image, ax=panels[0], label="acquisition score")
    panels[0].set(xlabel=r"$R_{OH}$ (angstrom)", ylabel=r"$\phi$ (degree)")
    panels[1].semilogy(fits[0].history, color="#0072b2")
    panels[1].set(xlabel="epoch", ylabel="normalized loss", title=f"training RMSE = {rmse:.1f} meV")
    figure.savefig(args.output / "phenol_abinitio_active.png", dpi=240)
    plt.close(figure)
    summary = {
        "training_samples": len(data["coordinates"]), "ensemble": len(fits),
        "training_matrix_rmse_mev": rmse, "proposals": len(selected),
        "checkpoint": str(checkpoint), "proposal_file": str(proposal_file),
    }
    (args.output / "phenol_abinitio_active.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    modes_parser = commands.add_parser("modes")
    modes_parser.add_argument("--method", default="b3lyp")
    modes_parser.add_argument("--basis", default="6-31+g*")
    modes_parser.add_argument("--grid-level", type=int, default=3)
    modes_parser.add_argument("--output", type=Path, default=Path("/private/tmp/phenol_hessian_modes"))
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--samples", type=int, default=8)
    generate_parser.add_argument("--coordinates", type=Path)
    generate_parser.add_argument("--basis", default="sto-3g")
    generate_parser.add_argument("--modes", type=Path)
    generate_parser.add_argument("--nstates", type=int, default=3)
    generate_parser.add_argument("--neighbors", type=int, default=3)
    generate_parser.add_argument(
        "--local-scale", type=float, default=1.0,
        help="contract the sampled chart about equilibrium; useful for overlap smoke tests",
    )
    generate_parser.add_argument("--gradients", action="store_true")
    generate_parser.add_argument("--seed", type=int, default=41)
    generate_parser.add_argument("--output", type=Path, default=Path("/private/tmp/phenol_abinitio_active"))
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("dataset", type=Path)
    fit_parser.add_argument("--epochs", type=int, default=500)
    fit_parser.add_argument("--ensemble", type=int, default=3)
    fit_parser.add_argument("--pool", type=int, default=2048)
    fit_parser.add_argument("--proposals", type=int, default=32)
    fit_parser.add_argument("--exclusion", type=float, default=0.08)
    fit_parser.add_argument("--channels", type=int, default=12)
    fit_parser.add_argument("--width", type=int, default=48)
    fit_parser.add_argument("--radial-basis", type=int, default=12)
    fit_parser.add_argument("--cutoff", type=float, default=4.5)
    fit_parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    fit_parser.add_argument("--seed", type=int, default=41)
    fit_parser.add_argument("--output", type=Path, default=Path("/private/tmp/phenol_abinitio_active_fit"))
    args = parser.parse_args()
    if args.command == "modes":
        hessian_modes(args)
    elif args.command == "generate":
        generate(args)
    else:
        fit_and_propose(args)


if __name__ == "__main__":
    main()
