"""Converge zone-boundary LiH exciton-phonon coupling and embedded spectrum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.pbc.gw import (
    ExcitonPhononChannel,
    ExcitonPhononContinuum,
    KPointTransitionSpace,
    analytic_tda_electron_phonon_coupling,
    commensurate_gdf_screened_tda_kernel_derivative,
    periodic_tda_operator,
    validate_commensurate_gdf_screened_tda_kernel_derivative,
)
from pyqed.qchem.pbc import (
    Cell,
    commensurate_gdf_q_derivative,
    gdf_q_derivative,
)
from pyqed.units import au2ev, au2mev


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compact_lih_basis():
    """Return a localized three-function all-electron benchmark basis."""

    return {
        "Li": [
            (0, np.asarray([3.2]), np.asarray([[1.0]])),
            (0, np.asarray([0.9]), np.asarray([[1.0]])),
        ],
        "H": [(0, np.asarray([1.2]), np.asarray([[1.0]]))],
    }


def _rocksalt_lih(lattice_constant, basis):
    half = 0.5 * float(lattice_constant)
    lattice = np.asarray(
        [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]],
        dtype=float,
    )
    basis_data = _compact_lih_basis() if basis == "compact" else basis
    return Cell(
        atom=[("Li", (0.0, 0.0, 0.0)), ("H", (half, half, half))],
        a=lattice,
        basis=basis_data,
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()


def _zone_boundary_q(space):
    qpoints = np.asarray(space.qpts, dtype=float)
    norms = np.linalg.norm(qpoints, axis=1)
    candidates = []
    for index in np.flatnonzero(norms > 1.0e-10):
        if space.find_qpoint_index(-qpoints[index]) == int(index):
            candidates.append(int(index))
    if not candidates:
        raise ValueError(
            "The zone-boundary benchmark requires an even k-point mesh."
        )
    return min(candidates, key=lambda index: norms[index])


def _solve_case(
    args,
    nk,
    recip_cut,
    *,
    build_spectrum=False,
    keep_objects=False,
):
    started = time.perf_counter()
    cell = _rocksalt_lih(args.lattice_constant, args.basis)
    mean_field = cell.KRHF(
        nk=(int(nk), 1, 1),
        eta=0.7,
        real_cut=args.pair_cut,
        pair_cut=args.pair_cut,
        recip_cut=int(recip_cut),
        one_body_nuclear_cut=1,
        jk_builder="gdf",
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    ).density_fit(
        auxbasis=args.auxbasis,
        reciprocal_kernel="full",
        recip_cut=int(recip_cut),
        pair_cut=args.pair_cut,
        pair_screen_tol=0.0,
        metric_tol=1.0e-12,
        storage="memory",
    ).run(max_cycle=80, conv_tol=1.0e-11, conv_tol_dm=1.0e-9)
    if not mean_field.converged:
        raise RuntimeError(f"LiH KRHF did not converge for Nk={nk}")
    scf_seconds = time.perf_counter() - started

    space = KPointTransitionSpace(mean_field, qpts="mesh")
    zero_q_index = space.find_qpoint_index(np.zeros(3))
    phonon_q_index = _zone_boundary_q(space)
    qpoint = np.asarray(space.qpts[phonon_q_index], dtype=float)
    source_operator = periodic_tda_operator(
        space,
        q_index=zero_q_index,
        direct_scale=2.0,
        exchange_scale=1.0,
        screened_exchange_scale=1.0,
        coulomb_component="gdf",
    )
    target_q_index = space.find_qpoint_index(
        np.asarray(space.qpts[zero_q_index]) + qpoint
    )
    target_operator = periodic_tda_operator(
        space,
        q_index=target_q_index,
        direct_scale=2.0,
        exchange_scale=1.0,
        screened_exchange_scale=1.0,
        coulomb_component="gdf",
    )

    derivative_started = time.perf_counter()
    q_derivative = gdf_q_derivative(
        mean_field,
        qpoint,
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        cphf_tol=1.0e-9,
    )
    kernel_derivative = commensurate_gdf_screened_tda_kernel_derivative(
        source_operator,
        q_derivative,
    )
    coupling = analytic_tda_electron_phonon_coupling(
        space,
        zero_q_index,
        phonon_q_index,
        args.frequency,
        q_derivative.fock_derivative,
        overlap_derivative=q_derivative.overlap_derivative,
        kernel_derivative=kernel_derivative,
        branch=args.branch,
    ).validate_momentum(space)
    derivative_seconds = time.perf_counter() - derivative_started

    nroots = min(args.nroots, source_operator.shape[0] - 1)
    source = source_operator.eigensolve(nroots=nroots, tol=1.0e-9)
    target = target_operator.eigensolve(nroots=nroots, tol=1.0e-9)
    exciton_couplings = coupling.between(target.vectors, source.vectors)
    factor_info = dict(q_derivative.gdf_q_derivative_factors.info)

    result = {
        "nk": int(nk),
        "kmesh": [int(nk), 1, 1],
        "recip_cut": int(recip_cut),
        "pair_cut": int(args.pair_cut),
        "scf_energy": float(mean_field.e_tot),
        "qpoint_cartesian": qpoint.tolist(),
        "source_q_index": int(zero_q_index),
        "target_q_index": int(target_q_index),
        "source_exciton_ev": (source.e * au2ev).tolist(),
        "target_exciton_ev": (target.e * au2ev).tolist(),
        "exciton_phonon_coupling_mev_real": (
            exciton_couplings.real * au2mev
        ).tolist(),
        "exciton_phonon_coupling_mev_imag": (
            exciton_couplings.imag * au2mev
        ).tolist(),
        "exciton_phonon_coupling_mev_abs": (
            np.abs(exciton_couplings) * au2mev
        ).tolist(),
        "maximum_coupling_mev": float(np.max(np.abs(exciton_couplings)) * au2mev),
        "one_body_derivative_norm": float(
            np.linalg.norm(coupling.electron_phonon_derivative.one_body.toarray())
        ),
        "kernel_derivative_norm": float(np.linalg.norm(kernel_derivative)),
        "q_derivative_info": _jsonable(q_derivative.info),
        "q_factor_info": _jsonable(factor_info),
        "scf_seconds": float(scf_seconds),
        "derivative_seconds": float(derivative_seconds),
        "total_seconds": float(time.perf_counter() - started),
    }

    if build_spectrum:
        channel = ExcitonPhononChannel.thermal_from_coupling(
            coupling,
            target_operator,
            source.vectors,
            temperature=args.temperature,
            excluded_vectors=target.vectors,
            solver_tol=1.0e-9,
        )
        continuum = channel.continuum
        phonon_continuum = ExcitonPhononContinuum((channel,))
        lower = min(float(np.min(source.e)), float(np.min(target.e) - args.frequency))
        upper = max(float(np.max(source.e)), float(np.max(target.e) + args.frequency))
        padding = 0.08 / au2ev
        energies = np.linspace(lower - padding, upper + padding, args.spectrum_points)
        embedding = phonon_continuum.run_spectrum(
            np.diag(source.e),
            energies,
            eta=args.eta_ev / au2ev,
        )
        result["temperature_kelvin"] = float(args.temperature)
        result["bose_occupation"] = float(channel.occupation)
        result["spectrum_energy_ev"] = (energies * au2ev).tolist()
        result["spectral_density"] = embedding.spectral_density.tolist()
        result["hybridization_ev"] = (
            embedding.hybridization_trace * au2ev
        ).tolist()
        result["continuum_dimension"] = int(continuum.ncontinuum)
        result["spectrum_success"] = bool(embedding.success)
    if keep_objects:
        validation_derivative = commensurate_gdf_q_derivative(
            mean_field,
            qpoint,
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            cphf_tol=1.0e-9,
        )
        result["reference_residuals"] = {
            name: float(value)
            for name, value in validation_derivative.info[
                "reference_residuals"
            ].items()
        }
        result["largest_reference_residual"] = max(
            value
            for name, value in result["reference_residuals"].items()
            if name.endswith("_relative")
        )
        result["objects"] = (source_operator, validation_derivative)
    return result


def _plot(results, output):
    output = Path(output).expanduser().resolve()
    final = next(row for row in results if "spectrum_energy_ev" in row)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    markers = ("o", "s", "^", "D")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "savefig.dpi": 360,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), constrained_layout=True)

    for index, nk in enumerate(sorted({row["nk"] for row in results})):
        rows = [row for row in results if row["nk"] == nk]
        axes[0, 0].plot(
            [row["recip_cut"] for row in rows],
            [row["maximum_coupling_mev"] for row in rows],
            marker=markers[index],
            color=colors[index],
            label=rf"$N_k={nk}$",
        )
    axes[0, 0].set(
        xlabel="GDF reciprocal cutoff",
        ylabel=r"$\max_{SS'}|g_{SS'\nu}|$ (meV)",
        title="a  Zone-boundary coupling convergence",
        xticks=sorted({row["recip_cut"] for row in results}),
    )
    axes[0, 0].legend(frameon=False)

    converged = {}
    for row in results:
        if row["nk"] not in converged or row["recip_cut"] > converged[row["nk"]]["recip_cut"]:
            converged[row["nk"]] = row
    nks = sorted(converged)
    axes[0, 1].plot(
        nks,
        [converged[nk]["source_exciton_ev"][0] for nk in nks],
        "o-",
        color=colors[0],
        label=r"source $S=0$",
    )
    axes[0, 1].plot(
        nks,
        [converged[nk]["target_exciton_ev"][0] for nk in nks],
        "s--",
        color=colors[1],
        label=r"target $S'=0$",
    )
    axes[0, 1].set(
        xlabel=r"$N_k$ in the $N_k\times1\times1$ mesh",
        ylabel="TDA energy (eV)",
        title="b  Exciton mesh trend",
        xticks=nks,
    )
    axes[0, 1].legend(frameon=False)

    energies = np.asarray(final["spectrum_energy_ev"])
    spectrum = np.asarray(final["spectral_density"])
    spectrum /= max(float(np.max(spectrum)), np.finfo(float).tiny)
    axes[1, 0].plot(energies, spectrum, color=colors[0])
    for energy in final["source_exciton_ev"]:
        axes[1, 0].axvline(energy, color="0.45", lw=0.8, ls="--")
    axes[1, 0].set(
        xlabel="Energy (eV)",
        ylabel="Normalized active spectrum",
        title=rf"c  Finite-$T$ Feshbach spectrum ({final['temperature_kelvin']:.0f} K)",
    )

    axes[1, 1].plot(
        nks,
        [converged[nk]["derivative_seconds"] for nk in nks],
        "o-",
        color=colors[2],
        label="q derivative + kernel",
    )
    axes[1, 1].plot(
        nks,
        [converged[nk]["scf_seconds"] for nk in nks],
        "s--",
        color=colors[3],
        label="GDF-KRHF",
    )
    axes[1, 1].set(
        xlabel=r"$N_k$ in the $N_k\times1\times1$ mesh",
        ylabel="Wall time (s)",
        title="d  Cost",
        xticks=nks,
    )
    axes[1, 1].legend(frameon=False)

    for axis in axes.reshape(-1):
        axis.grid(alpha=0.2, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def run(args):
    cases = [
        (nk, cutoff)
        for nk in args.meshes
        for cutoff in args.recip_cuts
    ]
    final_case = max(cases)
    validation_mesh = (
        args.validation_mesh
        if args.validation_mesh in args.meshes
        else max(args.meshes)
    )
    results = []
    for nk, cutoff in cases:
        print(f"LiH Nk={nk}, recip_cut={cutoff}", flush=True)
        result = _solve_case(
            args,
            nk,
            cutoff,
            build_spectrum=(nk, cutoff) == final_case,
            keep_objects=False,
        )
        results.append(result)

    validation = None
    if not args.skip_validation:
        print(
            "LiH validation "
            f"Nk={validation_mesh}, recip_cut={args.validation_recip_cut}",
            flush=True,
        )
        validation_case = _solve_case(
            args,
            validation_mesh,
            args.validation_recip_cut,
            build_spectrum=False,
            keep_objects=True,
        )
        source_operator, q_derivative = validation_case.pop("objects")
        checked = validate_commensurate_gdf_screened_tda_kernel_derivative(
            source_operator,
            q_derivative,
            steps=(args.validation_step,),
            representation_tol=args.representation_tol,
        )
        validation = {
            "step": float(args.validation_step),
            "total_relative_error": float(checked["relative_error"][0]),
            "bare_relative_error": float(
                checked["component_errors"]["bare"]["relative"][0]
            ),
            "screened_relative_error": float(
                checked["component_errors"]["screened"]["relative"][0]
            ),
            "largest_reference_residual": float(
                checked["largest_reference_residual"]
            ),
            "nk": int(validation_mesh),
            "recip_cut": int(args.validation_recip_cut),
        }

    _plot(results, args.figure)
    payload = {
        "system": "rocksalt LiH",
        "basis": str(args.basis),
        "auxbasis": str(args.auxbasis),
        "mesh_family": "Nk x 1 x 1",
        "qpoint_policy": "fixed self-opposite zone boundary",
        "frequency_hartree": float(args.frequency),
        "frequency_ev": float(args.frequency * au2ev),
        "mode": [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        "temperature_kelvin": float(args.temperature),
        "cases": results,
        "unprojected_supercell_validation": validation,
        "figure": str(Path(args.figure).expanduser().resolve()),
        "pdf": str(Path(args.figure).expanduser().resolve().with_suffix(".pdf")),
        "fidelity": (
            "direct primitive-cell full-reciprocal one-body, GDF, and CPHF "
            "response; static direct-RPA TDA kernel; commensurate displaced "
            "validation; supplied phonon mode/frequency; one-phonon "
            "Fan/Feshbach"
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": len(results),
                "output": str(output),
                "figure": payload["figure"],
                "pdf": payload["pdf"],
                "validation": validation,
            },
            indent=2,
        )
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", type=int, nargs="+", default=(2, 4, 6))
    parser.add_argument("--recip-cuts", type=int, nargs="+", default=(2, 3, 4))
    parser.add_argument("--pair-cut", type=int, default=2)
    parser.add_argument("--lattice-constant", type=float, default=7.72)
    parser.add_argument("--basis", default="compact")
    parser.add_argument("--auxbasis", default="sto-3g")
    parser.add_argument("--frequency", type=float, default=0.008)
    parser.add_argument("--branch", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--nroots", type=int, default=2)
    parser.add_argument("--eta-ev", type=float, default=0.04)
    parser.add_argument("--spectrum-points", type=int, default=240)
    parser.add_argument("--validation-step", type=float, default=1.0e-3)
    parser.add_argument("--validation-mesh", type=int, default=4)
    parser.add_argument("--validation-recip-cut", type=int, default=9)
    parser.add_argument("--representation-tol", type=float, default=1.0e-7)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_lih_exciton_phonon_convergence.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("/private/tmp/pbc_lih_exciton_phonon_convergence.png"),
    )
    args = parser.parse_args()
    if min(args.meshes) < 2 or min(args.recip_cuts) < 1:
        parser.error("meshes must be >=2 and reciprocal cutoffs must be positive")
    if any(mesh % 2 for mesh in args.meshes):
        parser.error("the fixed zone-boundary benchmark requires even meshes")
    if args.validation_recip_cut < 1:
        parser.error("validation reciprocal cutoff must be positive")
    run(args)


if __name__ == "__main__":
    main()
