#!/usr/bin/env python3
"""Run cached ab initio SO2 through direct-link FTT and TTLDR dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from examples.ldr.so2_casci_cgldr import infer_sine_domain
from pyqed.dvr import SineDVR
from pyqed.ldr import AbInitioFit
from pyqed.ldr.ttfit import LinkPath, field_mpo, grid_links
from pyqed.mps.decompose import decompose
from pyqed.mps.mps import MPO
from pyqed.namd.triatomic import Triatom
from pyqed.namd.ttldr import TTLDR
from pyqed.units import au2fs


DEFAULT_FIXTURE = Path("/private/tmp/so2_casci6e6o_singlet_5x5x5.npz")
DEFAULT_ALIGNED_GAUGE = Path(
    "/private/tmp/so2_cas6e6o_631gstar_procrustes_gauge_9x9x9/"
    "procrustes_gauge.npz"
)
DEFAULT_ALIGNED_GRIDS = Path("/private/tmp/so2_9x9x9_grids.npz")


class CachedSO2:
    """Index-addressable CASCI energies and nearest-neighbor overlaps."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.grids = tuple(
                np.asarray(archive[name], dtype=float)
                for name in ("r1", "r2", "theta")
            )
            self.energies = np.asarray(archive["energies"], dtype=float)
            self.spin_square = np.asarray(archive["spin_square"], dtype=float)
            arrays = tuple(np.asarray(archive[f"links_{axis}"]) for axis in range(3))
            self.metadata = {
                key: np.asarray(archive[key]).item()
                for key in ("source", "basis", "ncas", "nelecas", "multiplicity")
            }
        self.shape = self.energies.shape[:-1]
        self.nstates = self.energies.shape[-1]
        self.links = {
            (axis, index): values[index]
            for axis, values in enumerate(arrays)
            for index in np.ndindex(values.shape[:-2])
        }
        self.path = LinkPath(self.shape, self.nstates, self.links)
        self.calls = []
        self.labels = ("r1", "r2", "theta")
        self.energy_shift = None
        self.coordinate_system = "valence"

    def build(self, index):
        index = tuple(map(int, index))
        self.calls.append(index)
        return index, self.energies[index]

    def overlap(self, left, right):
        return self.path.between(tuple(left), tuple(right))


class CachedAlignedSO2:
    """Replay a Procrustes-aligned SO2 cache as raw electronic frames."""

    def __init__(self, path, gauge_path, grids_path):
        with np.load(path, allow_pickle=False) as archive:
            local = np.asarray(archive["energy"], dtype=complex)
            aligned_links = tuple(
                np.asarray(archive[f"link_{axis}"], dtype=complex)
                for axis in range(3)
            )
        with np.load(gauge_path, allow_pickle=False) as archive:
            self.gauge = np.asarray(archive["gauge"], dtype=complex)
            self.positive = np.asarray(archive["positive"], dtype=complex)
            self.anchor = tuple(map(int, archive["center"]))
        with np.load(grids_path, allow_pickle=False) as archive:
            self.grids = tuple(
                np.asarray(archive[name], dtype=float)
                for name in ("qs", "theta", "qa")
            )
        self.shape = local.shape[:-2]
        self.nstates = local.shape[-1]
        expected = (*self.shape, self.nstates, self.nstates)
        if self.gauge.shape != expected or self.positive.shape != expected:
            raise ValueError("aligned energy and Procrustes caches have different grids")
        if tuple(map(len, self.grids)) != self.shape:
            raise ValueError("coordinate grids do not match the aligned field cache")

        self.energies = np.linalg.eigvalsh(local)
        self.spin_square = np.zeros_like(self.energies)
        self.labels = ("qs", "theta", "qa")
        self.coordinate_system = "qs-theta-qa"
        self.energy_shift = 0.0
        self.metadata = {
            "source": "cached spin-pure SO2 Procrustes fields",
            "basis": "6-31G*",
            "ncas": 6,
            "nelecas": 6,
            "multiplicity": 1,
        }
        self._direct = self.gauge @ self.positive
        self.links = {}
        for axis, values in enumerate(aligned_links):
            for index in np.ndindex(values.shape[:-2]):
                right = list(index)
                right[axis] += 1
                right = tuple(right)
                self.links[(axis, index)] = (
                    self.gauge[index]
                    @ values[index]
                    @ self.gauge[right].conj().T
                )
        self.path = LinkPath(self.shape, self.nstates, self.links)
        self.calls = []

    def build(self, index):
        index = tuple(map(int, index))
        self.calls.append(index)
        return index, self.energies[index]

    def overlap(self, left, right):
        left = tuple(map(int, left))
        right = tuple(map(int, right))
        if left == right:
            return np.eye(self.nstates, dtype=complex)
        if right == self.anchor:
            return self._direct[left]
        if left == self.anchor:
            return self._direct[right].conj().T
        delta = np.asarray(right) - left
        active = np.flatnonzero(delta)
        if len(active) == 1 and abs(delta[active[0]]) == 1:
            axis = int(active[0])
            if delta[axis] > 0:
                return self.links[(axis, left)]
            return self.links[(axis, right)].conj().T
        return self.path.between(left, right)


def so2_geometry(r1, r2, theta):
    return [
        ["O", (float(r1), 0.0, 0.0)],
        ["S", (0.0, 0.0, 0.0)],
        ["O", (float(r2) * np.cos(theta), float(r2) * np.sin(theta), 0.0)],
    ]


def valence_keo(grids, nstates):
    center = tuple(grid[len(grid) // 2] for grid in grids)
    molecule = Triatom(
        so2_geometry(*center),
        basis="sto-3g",
        nstates=nstates,
        charge=0,
        spin=0,
        unit="bohr",
        coordinates="valence",
        dvr_type=("sine", "sine", "sine"),
    )
    molecule.set_dvr(
        domains=[infer_sine_domain(grid) for grid in grids],
        npts=[len(grid) for grid in grids],
        dvr_type=("sine", "sine", "sine"),
    )
    for expected, actual in zip(grids, molecule.x):
        np.testing.assert_allclose(actual, expected, atol=2.0e-12, rtol=0.0)
    return tuple(molecule.buildK_product_terms(symmetrize=True))


def transformed_valence_keo(grids, nstates, *, svd_tol=0.0):
    qs, theta, qa = grids
    center = tuple(grid[len(grid) // 2] for grid in grids)
    r1 = (center[0] + center[2]) / np.sqrt(2.0)
    r2 = (center[0] - center[2]) / np.sqrt(2.0)
    molecule = Triatom(
        so2_geometry(r1, r2, center[1]),
        basis="sto-3g",
        nstates=nstates,
        charge=0,
        spin=0,
        unit="bohr",
        coordinates="valence",
    )
    axes = (
        SineDVR(*infer_sine_domain(qs), len(qs)),
        SineDVR(*infer_sine_domain(theta), len(theta)),
        SineDVR(*infer_sine_domain(qa), len(qa)),
    )
    for expected, axis in zip(grids, axes):
        np.testing.assert_allclose(axis.x, expected, atol=2.0e-12, rtol=0.0)
    return tuple(
        molecule.buildK_qsqa_terms(
            axes,
            symmetrize=True,
            svd_tol=float(svd_tol),
        )
    )


def dense_keo(terms):
    matrix = None
    for term in terms:
        _label, coefficient, *factors = term
        value = np.asarray(factors[0])
        for factor in factors[1:]:
            value = np.kron(value, np.asarray(factor))
        value = coefficient * value
        matrix = value if matrix is None else matrix + value
    return 0.5 * (matrix + matrix.conj().T)


def dense_ldr(kinetic, energy, links):
    shape = energy.shape[:-2]
    nstates = energy.shape[-1]
    indices = tuple(np.ndindex(shape))
    path = LinkPath(shape, nstates, links)
    overlap = np.asarray(
        [[path.between(left, right) for right in indices] for left in indices]
    )
    matrix = (
        kinetic[:, None, :, None]
        * overlap.transpose(0, 2, 1, 3)
    ).reshape(len(indices) * nstates, len(indices) * nstates)
    for point, block in enumerate(energy.reshape(-1, nstates, nstates)):
        section = slice(point * nstates, (point + 1) * nstates)
        matrix[section, section] += block
    return 0.5 * (matrix + matrix.conj().T)


def matrix_field_mpo(values, rank):
    values = np.asarray(values, dtype=complex)
    shape = values.shape[:-2]
    nstates = values.shape[-1]
    cores = {
        (alpha, beta): decompose(values[..., alpha, beta], rank=int(rank))
        for alpha in range(nstates)
        for beta in range(nstates)
    }
    operator = field_mpo(cores, shape, nstates)
    return 0.5 * (operator + operator.adjoint())


def state_projector(state):
    factors = []
    for site in range(state.L):
        tensor = np.asarray(state._get_std_B(site))
        left, physical, right = tensor.shape
        factor = np.einsum(
            "apr,bqs->abrspq", tensor, tensor.conj(), optimize=True
        ).reshape(left * left, right * right, physical, physical)
        factors.append(factor)
    return MPO(factors)


def propagate(hamiltonian, initial, times):
    values, vectors = eigh(hamiltonian, overwrite_a=False, check_finite=False)
    coefficients = vectors.conj().T @ initial
    phases = np.exp(-1j * np.outer(times, values))
    return (phases * coefficients[None, :]) @ vectors.T


def expectations(states, operators):
    return np.column_stack(
        [
            np.einsum("ti,ij,tj->t", states.conj(), operator, states).real
            for operator in operators
        ]
    )


def relative(left, right):
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-15))


def plot_results(path, times_fs, exact, fitted, tensor, fit_errors, coordinate_labels):
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 5.8), constrained_layout=True)
    axes = axes.ravel()

    labels = (r"$\bar E$",) + tuple(
        rf"$\bar L_{{{label}}}$" for label in coordinate_labels
    )
    axes[0].bar(np.arange(4), 100.0 * np.asarray(fit_errors), color=("#555555", *colors))
    axes[0].set(xticks=np.arange(4), xticklabels=labels, ylabel="Relative error (%)")

    for state, color in enumerate(colors):
        axes[1].plot(times_fs, exact["populations"][:, state], color=color, lw=1.5, label=rf"$P_{state}$")
        axes[1].plot(times_fs, fitted["populations"][:, state], color=color, lw=1.1, ls=":")
        axes[1].plot(times_fs, tensor["populations"][:, state], color=color, lw=1.1, ls="--")
    axes[1].set(xlabel="Time (fs)", ylabel="Adiabatic population", ylim=(-0.02, 1.02))
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    axes[1].text(0.04, 0.72, "solid: exact grid\ndotted: fitted dense\ndashed: TTLDR", transform=axes[1].transAxes, ha="left", va="center", fontsize=7.5)

    axes[2].plot(times_fs, exact["autocorrelation"], color="#222222", lw=1.5, label="Exact grid")
    axes[2].plot(times_fs, fitted["autocorrelation"], color="#0072B2", lw=1.2, ls=":", label="Fitted dense")
    axes[2].plot(times_fs, tensor["autocorrelation"], color="#D55E00", lw=1.2, ls="--", label="TTLDR")
    axes[2].set(xlabel="Time (fs)", ylabel=r"$|C(t)|$", ylim=(-0.02, 1.02))
    axes[2].legend(frameon=False, fontsize=8)

    fit_deviation = np.max(np.abs(fitted["populations"] - exact["populations"]), axis=1)
    tdvp_deviation = np.max(np.abs(tensor["populations"] - fitted["populations"]), axis=1)
    axes[3].semilogy(times_fs, np.maximum(fit_deviation, 1.0e-12), color="#0072B2", label="FTT Hamiltonian")
    axes[3].semilogy(times_fs, np.maximum(tdvp_deviation, 1.0e-12), color="#D55E00", label="TDVP")
    axes[3].set(xlabel="Time (fs)", ylabel="Max. population error")
    axes[3].legend(frameon=False, fontsize=8)

    for label, axis in zip("abcd", axes):
        axis.text(0.02, 0.98, label, transform=axis.transAxes, va="top", fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", linewidth=0.6)
    figure.savefig(path, dpi=350)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--aligned-gauge", type=Path, default=DEFAULT_ALIGNED_GAUGE)
    parser.add_argument("--aligned-grids", type=Path, default=DEFAULT_ALIGNED_GRIDS)
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/so2_abinitio_ftt_ttldr"))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--validation", type=int, default=128)
    parser.add_argument(
        "--sampler",
        choices=("cross", "block-cross", "sparse", "cur"),
        default="cross",
    )
    parser.add_argument("--initial", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sparse-sequence", choices=("halton", "random"), default="halton")
    parser.add_argument("--cur-axis", type=int, default=1)
    parser.add_argument("--cur-slabs", type=int, default=4)
    parser.add_argument("--cur-probes", type=int)
    parser.add_argument("--fit-sweeps", type=int, default=12)
    parser.add_argument("--fit-rtol", type=float, default=1.0e-4)
    parser.add_argument("--overlap-rank", type=int, default=32)
    parser.add_argument("--operator-rank", type=int, default=64)
    parser.add_argument("--kinetic-svd-tol", type=float, default=1.0e-10)
    parser.add_argument(
        "--propagation-backend", choices=("sum", "combined"), default="sum"
    )
    parser.add_argument("--combined-rank", type=int, default=64)
    parser.add_argument("--state-rank", type=int, default=48)
    parser.add_argument("--time-fs", type=float, default=2.0)
    parser.add_argument("--dt-fs", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    with np.load(args.fixture, allow_pickle=False) as archive:
        aligned_fixture = "energy" in archive and "link_0" in archive
    data = (
        CachedAlignedSO2(args.fixture, args.aligned_gauge, args.aligned_grids)
        if aligned_fixture
        else CachedSO2(args.fixture)
    )
    if np.max(np.abs(data.spin_square)) > 1.0e-8:
        raise RuntimeError("the cached CASCI roots are not spin-pure singlets")
    anchor = getattr(data, "anchor", tuple(size // 2 for size in data.shape))
    indices = tuple(np.ndindex(data.shape))

    fit_started = time.perf_counter()
    fields = args.output_dir / "fields"
    with AbInitioFit(
        data.grids,
        data.nstates,
        data.build,
        anchor=anchor,
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=data.overlap,
        cache=args.output_dir / "point_cache",
        energy_shift=data.energy_shift,
    ) as fit:
        fit.run(
            sampler=args.sampler,
            representation="links",
            rank=args.rank,
            energy_rank=args.rank,
            link_rank=args.rank,
            degrees=args.degree,
            sweeps=args.sweeps,
            rtol=args.fit_rtol if args.sampler == "sparse" else 1.0e-10,
            validation=args.validation,
            seed=args.seed,
            start_rank=2,
            kick_rank=2,
            initial=args.initial,
            rounds=args.rounds,
            sparse_sequence=args.sparse_sequence,
            cur_axis=args.cur_axis,
            cur_slabs=args.cur_slabs,
            cur_probes=args.cur_probes,
            fit_sweeps=args.fit_sweeps,
        )
        sampling = fit.stats
        exact_energy = fit.oracle.hamiltonian_many(indices).reshape(
            *data.shape, data.nstates, data.nstates
        )
        exact_links = {}
        for axis, size in enumerate(data.shape):
            edge_shape = list(data.shape)
            edge_shape[axis] -= 1
            left_indices = tuple(np.ndindex(tuple(edge_shape)))
            pairs = []
            for left in left_indices:
                right = list(left)
                right[axis] += 1
                pairs.append((left, tuple(right)))
            values = fit.oracle.overlap_many(pairs)
            exact_links.update(
                {(axis, left): value for left, value in zip(left_indices, values)}
            )
        fit.save(fields, labels=data.labels, metadata=data.metadata)
        energy_shift = fit.energy_shift
    fit_seconds = time.perf_counter() - fit_started

    fitted = AbInitioFit.load(fields)
    try:
        terms = (
            transformed_valence_keo(
                data.grids,
                data.nstates,
                svd_tol=args.kinetic_svd_tol,
            )
            if data.coordinate_system == "qs-theta-qa"
            else valence_keo(data.grids, data.nstates)
        )
        build_started = time.perf_counter()
        driver = TTLDR.from_fit(
            fitted,
            keo=terms,
            overlap_rank=args.overlap_rank,
            operator_rank=args.operator_rank,
            fitted_kinetic_backend="link-mpo",
            seed=args.seed,
        )
        if args.propagation_backend == "combined":
            combined = driver.hamiltonian.compress_hermitian(args.combined_rank)
            driver._hamiltonian = combined
            driver.components = (combined,)
        ttldr_build_seconds = time.perf_counter() - build_started

        mesh = np.meshgrid(*data.grids, indexing="ij")
        coordinates = np.stack([value.reshape(-1) for value in mesh], axis=1)
        fitted_energy = fitted.energy.predict(coordinates).reshape(exact_energy.shape)
        fitted_links = grid_links(fitted.links, data.grids)
        energy_error = relative(fitted_energy, exact_energy)
        link_errors = []
        for axis in range(3):
            keys = [key for key in exact_links if key[0] == axis]
            link_errors.append(
                relative(
                    np.asarray([fitted_links[key] for key in keys]),
                    np.asarray([exact_links[key] for key in keys]),
                )
            )

        kinetic = dense_keo(terms)
        exact_hamiltonian = dense_ldr(kinetic, exact_energy, exact_links)
        fitted_hamiltonian = driver.hamiltonian.to_dense()
        hamiltonian_error = relative(fitted_hamiltonian, exact_hamiltonian)

        packet_widths = (
            (0.105, np.deg2rad(6.0), 0.075)
            if data.coordinate_system == "qs-theta-qa"
            else (0.075, 0.075, np.deg2rad(6.0))
        )
        nuclear = np.ones(data.shape, dtype=complex)
        for axis, (grid, width) in enumerate(zip(data.grids, packet_widths)):
            shape = [1, 1, 1]
            shape[axis] = len(grid)
            nuclear *= np.exp(-0.5 * ((grid.reshape(shape) - grid[anchor[axis]]) / width) ** 2)
        predicted_values, predicted_vectors = np.linalg.eigh(fitted_energy)
        del predicted_values
        selected = predicted_vectors[..., :, 2].copy()
        anchor_vector = selected[anchor]
        overlaps = np.einsum("a,...a->...", anchor_vector.conj(), selected)
        phase = np.ones(data.shape, dtype=complex)
        regular = np.abs(overlaps) > 1.0e-12
        phase[regular] = overlaps[regular].conj() / np.abs(overlaps[regular])
        initial = nuclear[..., None] * selected * phase[..., None]
        initial = initial.reshape(-1)
        initial /= np.linalg.norm(initial)
        mps = driver.state(initial.reshape(driver.dims), max_rank=args.state_rank, physical=False)

        projector_fields = []
        for state in range(data.nstates):
            vectors = predicted_vectors[..., :, state]
            projector_fields.append(
                np.einsum("...a,...b->...ab", vectors, vectors.conj(), optimize=True)
            )
        projectors = tuple(
            matrix_field_mpo(values, rank=args.state_rank) for values in projector_fields
        )
        survival = state_projector(mps)
        observable_matrices = tuple(projector.to_dense() for projector in projectors)

        steps = int(round(args.time_fs / args.dt_fs))
        propagation_started = time.perf_counter()
        driver.run(
            mps,
            dt=args.dt_fs / au2fs,
            steps=steps,
            interval=1,
            max_bond=args.state_rank,
            cutoff=1.0e-12,
            krylov_dim=16,
            krylov_tol=1.0e-11,
            progress=False,
            e_ops=(*projectors, survival),
        )
        tdvp_seconds = time.perf_counter() - propagation_started
        times = np.asarray(driver.times)
        times_fs = times * au2fs

        exact_states = propagate(exact_hamiltonian, initial, times)
        fitted_states = propagate(fitted_hamiltonian, initial, times)
        exact_populations = expectations(exact_states, observable_matrices)
        fitted_populations = expectations(fitted_states, observable_matrices)
        tensor_populations = np.asarray(driver.populations[:, : data.nstates])
        exact_auto = np.abs(exact_states @ initial.conj())
        fitted_auto = np.abs(fitted_states @ initial.conj())
        tensor_auto = np.sqrt(np.maximum(driver.populations[:, -1], 0.0))

        exact_result = {"populations": exact_populations, "autocorrelation": exact_auto}
        fitted_result = {"populations": fitted_populations, "autocorrelation": fitted_auto}
        tensor_result = {"populations": tensor_populations, "autocorrelation": tensor_auto}
        figure_path = args.output_dir / "so2_abinitio_ftt_ttldr.png"
        plot_results(
            figure_path,
            times_fs,
            exact_result,
            fitted_result,
            tensor_result,
            (energy_error, *link_errors),
            data.labels,
        )

        tensor_final = driver.dense(driver.final_state, physical=False).reshape(-1)
        fidelity = abs(np.vdot(fitted_states[-1], tensor_final)) ** 2
        fidelity /= np.vdot(fitted_states[-1], fitted_states[-1]).real
        fidelity /= np.vdot(tensor_final, tensor_final).real
        summary = {
            "method": "spin-pure SO2 CASCI(6e,6o)/6-31G* + direct-link FunctionalTT + TTLDR",
            "coordinates": list(data.labels),
            "kinetic_model": "coordinate-matched sine-DVR transformed valence KEO",
            "kinetic_svd_tolerance": args.kinetic_svd_tol,
            "sampler": args.sampler,
            "propagation_backend": args.propagation_backend,
            "fixture": str(args.fixture),
            "grid": list(data.shape),
            "nstates": data.nstates,
            "energy_shift_Eh": energy_shift,
            "spin_square_max_abs": float(np.max(np.abs(data.spin_square))),
            "fit": {
                "energy_relative_error": energy_error,
                "link_relative_error": {
                    label: error for label, error in zip(data.labels, link_errors)
                },
                "sampled_geometries": sampling["fit"]["unique_geometries"],
                "new_electronic_structure_calls": sampling["fit"]["quantum_chemistry_calls"],
                "disk_cache_restores": sampling["fit"]["disk_cache_restores"],
                "sampling": sampling,
            },
            "hamiltonian_relative_error": hamiltonian_error,
            "maximum_fitted_population_error_vs_exact": float(
                np.max(np.abs(fitted_populations - exact_populations))
            ),
            "maximum_ttldr_population_error_vs_fitted_dense": float(
                np.max(np.abs(tensor_populations - fitted_populations))
            ),
            "maximum_ttldr_population_error_vs_exact": float(
                np.max(np.abs(tensor_populations - exact_populations))
            ),
            "final_ttldr_fidelity_vs_fitted_dense": float(fidelity),
            "maximum_ttldr_norm_error": float(np.max(np.abs(np.asarray(driver.norms) - 1.0))),
            "operator_ranks": driver.operator_ranks,
            "timings_seconds": {
                "abinitio_fit": fit_seconds,
                "ttldr_build": ttldr_build_seconds,
                "ttldr_tdvp": tdvp_seconds,
                "total": time.perf_counter() - total_started,
            },
            "output": {
                "fields": str(fields),
                "figure": str(figure_path),
            },
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez(
            args.output_dir / "dynamics.npz",
            times_fs=times_fs,
            exact_populations=exact_populations,
            fitted_dense_populations=fitted_populations,
            ttldr_populations=tensor_populations,
            exact_autocorrelation=exact_auto,
            fitted_dense_autocorrelation=fitted_auto,
            ttldr_autocorrelation=tensor_auto,
            ttldr_norms=driver.norms,
        )
        print(json.dumps(summary, indent=2), flush=True)
        print(f"figure: {figure_path}", flush=True)
    finally:
        fitted.close()


if __name__ == "__main__":
    main()
