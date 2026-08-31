#!/usr/bin/env python3
"""Staged phenol photodissociation benchmark: analytic 2D -> MACE -> FTT -> TTLDR.

Stage 1 uses the published three-state ``(R_OH, phi_CCOH)`` diabatic model in
``pyqed.models.phenol`` as an exact reference.  It validates the phenol
coordinate map and tensor-dynamics plumbing before replacing the analytic
matrix by ab initio energies and overlap-derived endpoint features in later
stages.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.stats import qmc

from pyqed.dvr import ExponentialDVR, SineDVR
from pyqed.ml import MACE
from pyqed.models.phenol import Phenol3D, dpes1
from pyqed.mps.functional import FunctionalTT
from pyqed.namd.ttldr import TTLDR
from pyqed.units import au2angstrom, au2ev, au2fs


SPECIES = ("C",) * 6 + ("O", "H") + ("H",) * 5
STATE_LABELS = (r"$\pi\pi$", r"$\pi\sigma^*$", r"$\pi\pi^*$")
STATE_COLORS = ("#3b3b3b", "#d55e00", "#0072b2")
PHENOL_MATRIX_BASIS = np.asarray([
    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
])


def phenol_geometry(coordinate, *, coh_angle_deg=108.8):
    """Return a 13-atom phenol geometry for ``(R_OH [angstrom], phi [rad])``.

    The phenoxyl fragment is fixed in the xy plane.  ``phi=0`` places the OH
    hydrogen in that plane, and positive phi moves it toward positive z.
    """

    r_oh, phi = map(float, coordinate)
    r_cc, r_co, r_ch = 1.394, 1.360, 1.084
    carbons = np.asarray([
        r_cc * np.asarray((np.cos(index * np.pi / 3.0), np.sin(index * np.pi / 3.0), 0.0))
        for index in range(6)
    ])
    oxygen = carbons[0] + np.asarray((r_co, 0.0, 0.0))
    theta = np.deg2rad(float(coh_angle_deg))
    oh_direction = (
        np.cos(theta) * np.asarray((-1.0, 0.0, 0.0))
        + np.sin(theta)
        * (
            np.cos(phi) * np.asarray((0.0, 1.0, 0.0))
            + np.sin(phi) * np.asarray((0.0, 0.0, 1.0))
        )
    )
    hydroxyl_h = oxygen + r_oh * oh_direction
    ring_h = np.asarray([
        carbons[index] + r_ch * carbons[index] / np.linalg.norm(carbons[index])
        for index in range(1, 6)
    ])
    return np.vstack((carbons, oxygen, hydroxyl_h, ring_h))


def reference_dpem(coordinates):
    """Evaluate the analytic three-state diabatic matrix in Hartree."""

    values = np.asarray(coordinates, dtype=float)
    one_point = values.ndim == 1
    values = np.atleast_2d(values)
    matrices = np.asarray([
        dpes1(r_angstrom / au2angstrom, phi)
        for r_angstrom, phi in values
    ])
    return matrices[0] if one_point else matrices


def parity_reduce(coordinates, matrices, *, planar_probe=1.0e-4):
    r"""Factor the odd diabatic couplings as $H_{ij}=\sin\phi\,C_{ij}$."""

    coordinates = np.atleast_2d(np.asarray(coordinates, dtype=float))
    values = np.asarray(matrices, dtype=complex).copy()
    sine = np.sin(coordinates[:, 1])
    regular = np.abs(sine) > 1.0e-8
    for left, right in ((0, 1), (1, 2)):
        values[regular, left, right] /= sine[regular]
        values[regular, right, left] /= sine[regular]
        if np.any(~regular):
            probes = coordinates[~regular].copy()
            probes[:, 1] = float(planar_probe)
            probe_values = reference_dpem(probes)[:, left, right] / np.sin(planar_probe)
            values[~regular, left, right] = probe_values
            values[~regular, right, left] = probe_values.conj()
    return values


def parity_expand(coordinates, coefficients):
    """Reconstruct the physical phenol diabatic matrix from even coefficients."""

    coordinates = np.atleast_2d(np.asarray(coordinates, dtype=float))
    values = np.asarray(coefficients, dtype=complex).copy()
    sine = np.sin(coordinates[:, 1])
    for left, right in ((0, 1), (1, 2)):
        values[:, left, right] *= sine
        values[:, right, left] *= sine
    return 0.5 * (values + values.conj().swapaxes(-1, -2))


def independent_coefficients(matrices):
    """Extract the five allowed real coefficients from a reduced DPEM."""

    values = np.asarray(matrices)
    return np.column_stack(
        (
            values[:, 0, 0].real,
            values[:, 1, 1].real,
            values[:, 2, 2].real,
            values[:, 0, 1].real,
            values[:, 1, 2].real,
        )
    )


def product_coordinates(axes):
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([value.reshape(-1) for value in mesh], axis=1)


def sobol_coordinates(count, bounds, seed):
    """Return a nested Sobol prefix in the two-coordinate chart."""

    count = int(count)
    if count < 1:
        raise ValueError("sample count must be positive")
    power = int(np.ceil(np.log2(count)))
    unit = qmc.Sobol(2, scramble=True, seed=int(seed)).random_base2(power)[:count]
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    return lower + unit * (upper - lower)


def reactive_training_coordinates(count, bounds, seed):
    """Mix uniform and short-bond-biased nested low-discrepancy samples."""

    count = int(count)
    uniform_count = count // 2
    uniform = sobol_coordinates(uniform_count, bounds, seed)
    reactive = sobol_coordinates(count - uniform_count, bounds, seed + 1)
    lower, upper = map(float, bounds[0])
    fraction = (reactive[:, 0] - lower) / (upper - lower)
    reactive[:, 0] = lower + (upper - lower) * fraction**2
    return np.vstack((uniform, reactive))


def reflection_paired_training_coordinates(count, bounds, seed, *, planar_points=17):
    """Build an exactly ``phi -> -phi`` paired design including the planar cut."""

    count = int(count)
    half = max(1, count // 2)
    positive_bounds = (bounds[0], (0.0, max(abs(bounds[1][0]), abs(bounds[1][1]))))
    positive = reactive_training_coordinates(half, positive_bounds, seed)
    negative = positive.copy()
    negative[:, 1] *= -1.0
    phase = np.linspace(0.0, np.pi, int(planar_points))
    lower, upper = bounds[0]
    radial = lower + 0.5 * (upper - lower) * (1.0 - np.cos(phase))
    planar = np.column_stack((radial, np.zeros_like(radial)))
    return np.vstack((positive, negative, planar))


def build_dvrs(nr, ntorsion, r_bounds=(0.72, 4.5)):
    probe = Phenol3D([Phenol3D.r_eq], [Phenol3D.theta_eq], [0.0])
    r_dvr = SineDVR(
        float(r_bounds[0]) / au2angstrom,
        float(r_bounds[1]) / au2angstrom,
        int(nr),
        mass=probe.radial_mass,
    )
    torsion_dvr = ExponentialDVR(
        npts=int(ntorsion), L=2.0 * np.pi, x0=np.pi / int(ntorsion),
        mass=probe.torsional_inertia,
    )
    axes = (r_dvr.x * au2angstrom, torsion_dvr.x.copy())
    return axes, (r_dvr, torsion_dvr)


def kinetic_terms(dvrs):
    r_dvr, torsion_dvr = dvrs
    identity_r = np.eye(r_dvr.npts)
    identity_phi = np.eye(torsion_dvr.npts)
    return (
        (1.0, (r_dvr.t(), identity_phi)),
        (1.0, (identity_r, torsion_dvr.t())),
    )


def identity_link_samples(coordinates, nstates=3):
    identity = np.eye(int(nstates), dtype=complex)
    values = np.broadcast_to(identity, (len(coordinates), *identity.shape)).copy()
    return tuple((coordinates, values) for _axis in range(2))


def exact_fit(axes, energy, *, rank=24, degree=12):
    """Build an exact-grid FTT object with identity electronic links."""

    bounds = tuple((float(axis[0]), float(axis[-1])) for axis in axes)
    degrees = tuple(min(int(degree), len(axis) - 1) for axis in axes)
    energy_model = FunctionalTT(
        degrees=degrees,
        rank=int(rank),
        bounds=bounds,
        normalization="frobenius",
        hermitian=True,
    ).fit_grid(axes, energy)
    links = []
    for active in range(2):
        edge_axes = list(axes)
        edge_axes[active] = 0.5 * (edge_axes[active][:-1] + edge_axes[active][1:])
        shape = tuple(len(axis) for axis in edge_axes)
        values = np.broadcast_to(np.eye(3), (*shape, 3, 3)).copy()
        links.append(
            FunctionalTT(
                degrees=tuple(min(int(degree), len(axis) - 1) for axis in edge_axes),
                rank=int(rank),
                bounds=tuple((float(axis[0]), float(axis[-1])) for axis in edge_axes),
                normalization="frobenius",
                hermitian=False,
            ).fit_grid(tuple(edge_axes), values)
        )
    return SimpleNamespace(
        success=True,
        grids=tuple(axes),
        energy=energy_model,
        links=tuple(links),
        feature=None,
    )


def initial_packet(axes, *, bright_state=2, sigma_r=0.075, sigma_phi=0.22):
    r, phi = np.meshgrid(*axes, indexing="ij")
    r_eq = Phenol3D.r_eq * au2angstrom
    wrapped_phi = np.angle(np.exp(1j * phi))
    nuclear = np.exp(-0.25 * ((r - r_eq) / float(sigma_r)) ** 2)
    nuclear *= np.exp(-0.25 * (wrapped_phi / float(sigma_phi)) ** 2)
    state = np.zeros((*r.shape, 3), dtype=complex)
    state[..., int(bright_state)] = nuclear
    return state / np.linalg.norm(state)


def dense_reference_hamiltonian(energy, dvrs):
    r_dvr, torsion_dvr = dvrs
    nuclear = sp.kron(r_dvr.t(), sp.eye(torsion_dvr.npts), format="csr")
    nuclear += sp.kron(sp.eye(r_dvr.npts), torsion_dvr.t(), format="csr")
    kinetic = sp.kron(nuclear, sp.eye(3), format="csr")
    potential = sp.block_diag(energy.reshape(-1, 3, 3), format="csr")
    return kinetic + potential


def propagate(hamiltonian, initial, times_au):
    return expm_multiply(
        -1j * hamiltonian,
        initial.reshape(-1),
        start=float(times_au[0]),
        stop=float(times_au[-1]),
        num=len(times_au),
        endpoint=True,
    ).reshape(len(times_au), *initial.shape)


def observables(states, axes, *, dissociation=2.5):
    density = np.sum(np.abs(states) ** 2, axis=-1)
    populations = np.sum(np.abs(states) ** 2, axis=(1, 2))
    radial_density = density.sum(axis=2)
    r_mean = np.einsum("tr,r->t", radial_density, axes[0], optimize=True)
    dissociated = radial_density[:, axes[0] >= float(dissociation)].sum(axis=1)
    return {
        "populations": populations,
        "radial_density": radial_density,
        "r_mean": r_mean,
        "dissociation": dissociated,
    }


def field_metrics(predicted, reference):
    difference = np.asarray(predicted) - np.asarray(reference)
    point_error = np.linalg.norm(difference, axis=(-2, -1))
    return {
        "matrix_rmse_meV": float(np.sqrt(np.mean(point_error**2)) * au2ev * 1.0e3),
        "matrix_max_meV": float(np.max(point_error) * au2ev * 1.0e3),
        "relative_frobenius_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference), np.finfo(float).tiny)
        ),
    }


def plot_fit(axes, reference, predicted, history, output):
    reference_levels = np.linalg.eigvalsh(reference) * au2ev
    predicted_levels = np.linalg.eigvalsh(predicted) * au2ev
    phi_zero = int(np.argmin(np.abs(axes[1])))
    error = np.linalg.norm(predicted - reference, axis=(-2, -1)) * au2ev * 1.0e3
    figure, panels = plt.subplots(1, 3, figsize=(12.2, 3.55), constrained_layout=True)
    panels[0].plot(np.maximum(history, 1.0e-16), color="#0072b2")
    panels[0].set(xlabel="epoch", ylabel="MACE loss", yscale="log", title="Training")
    for state, color in enumerate(STATE_COLORS):
        panels[1].plot(
            axes[0], reference_levels[:, phi_zero, state], color=color, lw=2.0,
            label=f"Reference {STATE_LABELS[state]}",
        )
        panels[1].plot(
            axes[0], predicted_levels[:, phi_zero, state], color=color, lw=1.25,
            ls="--", label=f"This work {STATE_LABELS[state]}",
        )
    panels[1].set(
        xlabel=r"$R_{\mathrm{OH}}$ ($\AA$)", ylabel="energy (eV)",
        title=rf"Planar cut, $\phi={np.rad2deg(axes[1][phi_zero]):.1f}^\circ$",
    )
    panels[1].legend(frameon=False, fontsize=7, ncol=2)
    image = panels[2].pcolormesh(
        np.rad2deg(axes[1]), axes[0], error, shading="auto", cmap="magma",
    )
    panels[2].set(
        xlabel=r"$\phi_{\mathrm{CCOH}}$ (degree)",
        ylabel=r"$R_{\mathrm{OH}}$ ($\AA$)",
        title="MACE matrix error",
    )
    figure.colorbar(image, ax=panels[2], label="meV")
    for panel in panels:
        panel.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_dynamics(times_fs, axes, reference, predicted, tt_populations, output):
    reference_obs = observables(reference, axes)
    predicted_obs = observables(predicted, axes)
    figure, panels = plt.subplots(1, 3, figsize=(12.2, 3.55), constrained_layout=True)
    for state, color in enumerate(STATE_COLORS):
        panels[0].plot(
            times_fs, reference_obs["populations"][:, state], color=color, lw=2.0,
            label=f"Reference {STATE_LABELS[state]}",
        )
        panels[0].plot(
            times_fs, predicted_obs["populations"][:, state], color=color,
            lw=1.25, ls="--", label=f"This work {STATE_LABELS[state]}",
        )
        panels[0].scatter(
            times_fs, tt_populations[:, state], color=color, s=7, alpha=0.5,
        )
    panels[0].set(xlabel="time (fs)", ylabel="diabatic population", ylim=(-0.02, 1.02))
    panels[0].legend(frameon=False, fontsize=7, ncol=2)
    panels[1].plot(
        times_fs, reference_obs["r_mean"], color="#222222", lw=2.0,
        label="Reference",
    )
    panels[1].plot(
        times_fs, predicted_obs["r_mean"], color="#0072b2", lw=1.25,
        ls="--", label="This work",
    )
    panels[1].plot(
        times_fs, reference_obs["dissociation"], color="#d55e00", lw=1.5,
        label=r"Reference $P(R>2.5\,\AA)$",
    )
    panels[1].set(xlabel="time (fs)", ylabel=r"$\langle R_{\mathrm{OH}}\rangle$ ($\AA$) / yield")
    panels[1].legend(frameon=False, fontsize=8)
    image = panels[2].pcolormesh(
        times_fs, axes[0], predicted_obs["radial_density"].T,
        shading="auto", cmap="magma",
    )
    panels[2].axhline(2.5, color="white", lw=0.9, ls="--")
    panels[2].set(
        xlabel="time (fs)", ylabel=r"$R_{\mathrm{OH}}$ ($\AA$)",
        title="This work: radial probability",
    )
    figure.colorbar(image, ax=panels[2], label="probability")
    for panel in panels:
        panel.spines[["top", "right"]].set_visible(False)
    figure.savefig(output, dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run(args):
    axes, dvrs = build_dvrs(args.nr, args.ntorsion, (args.rmin, args.rmax))
    coordinates = product_coordinates(axes)
    reference = reference_dpem(coordinates).reshape(*map(len, axes), 3, 3)
    bounds = tuple((float(axis[0]), float(axis[-1])) for axis in axes)
    supplied_training = getattr(args, "training_coordinates", None)
    if supplied_training is None:
        training_coordinates = reflection_paired_training_coordinates(
            args.samples, bounds, args.seed
        )
        equilibrium = np.asarray([[Phenol3D.r_eq * au2angstrom, 0.0]])
        training_coordinates = np.vstack((equilibrium, training_coordinates))
    else:
        training_coordinates = np.asarray(supplied_training, dtype=float)
        if training_coordinates.ndim != 2 or training_coordinates.shape[1] != 2:
            raise ValueError("training_coordinates must have shape (nsamples, 2)")
    training_energy = reference_dpem(training_coordinates)
    training_reduced = parity_reduce(training_coordinates, training_energy)
    fit = MACE(
        axes,
        SPECIES,
        phenol_geometry,
        3,
        # The reactive chart is known in this staged benchmark.  Supplying it
        # alongside the atomistic MACE invariants avoids asking message passing
        # to rediscover the breaking O--H distance and C--C--O--H torsion.
        chart_features=True,
        geometry_units="angstrom",
        channels=args.channels,
        max_ell=2,
        interactions=2,
        correlation=2,
        radial_basis=args.radial_basis,
        radial_mlp=(args.head_width, args.head_width),
        cutoff=args.cutoff,
    )
    if args.target == "parity":
        fit.fit_basis_h(
            training_coordinates,
            independent_coefficients(training_reduced),
            PHENOL_MATRIX_BASIS,
            hidden=(args.head_width, args.head_width),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    else:
        fit.fit(
            (training_coordinates, training_energy),
            identity_link_samples(training_coordinates),
            hidden=(args.head_width, args.head_width),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            energy_weight=1.0,
            link_weight=0.1,
            seed=args.seed,
            distill=True,
            tt_rank=args.tt_rank,
            tt_degree=args.tt_degree,
        )
    if args.target == "parity":
        neural_coefficients = fit.neural_energy.predict(coordinates)
        neural_physical = parity_expand(coordinates, neural_coefficients).reshape(reference.shape)
        distilled = exact_fit(
            axes, neural_physical, rank=args.tt_rank, degree=args.tt_degree
        )
        fit.energy = distilled.energy
        fit.links = distilled.links
        fit.feature = None
        fitted_physical = fit.energy.predict(coordinates).reshape(reference.shape)
        scale = max(float(np.linalg.norm(neural_physical)), np.finfo(float).tiny)
        fit.info["distillation"] = {
            "energy_relative_error": float(
                np.linalg.norm(fitted_physical - neural_physical) / scale
            ),
            "link_relative_errors": (0.0, 0.0),
            "rank": int(args.tt_rank),
            "degree": int(args.tt_degree),
            "physical_reconstruction": "odd couplings = sin(phi) * even coefficients",
        }
    predicted = fit.energy.predict(coordinates).reshape(reference.shape)

    validation_coordinates = sobol_coordinates(
        args.validation_samples, bounds, args.seed + 1009
    )
    validation_reference = reference_dpem(validation_coordinates)
    validation_raw = fit.neural_energy.predict(validation_coordinates)
    validation_predicted = (
        parity_expand(validation_coordinates, validation_raw)
        if args.target == "parity"
        else validation_raw
    )
    metrics = {
        "stage": 1,
        "coordinates": ["R_OH_angstrom", "phi_CCOH_radian"],
        "target_representation": args.target,
        "training_geometries": int(len(training_coordinates)),
        "target_grid_geometries": int(np.prod(tuple(map(len, axes)))),
        "offgrid_validation": field_metrics(validation_predicted, validation_reference),
        "target_grid_ftt": field_metrics(predicted, reference),
        "mace_initial_loss": float(fit.history[0]),
        "mace_final_loss": float(fit.history[-1]),
        "distillation": fit.info["distillation"],
    }

    keo = kinetic_terms(dvrs)
    driver = TTLDR.from_fit(
        fit,
        keo=keo,
        overlap_rank=args.overlap_rank,
        potential_rank=args.potential_rank,
        operator_rank=args.operator_rank,
    )
    exact = exact_fit(axes, reference, rank=max(args.tt_rank, 24), degree=args.tt_degree)
    reference_driver = TTLDR.from_fit(
        exact,
        keo=keo,
        overlap_rank=max(args.overlap_rank, 16),
        potential_rank=None,
        operator_rank=None,
    )
    reference_h = dense_reference_hamiltonian(reference, dvrs)
    exact_ftt_h = reference_driver.hamiltonian.to_dense()
    predicted_h = driver.hamiltonian.to_dense()
    reference_dense = reference_h.toarray()
    metrics["exact_ftt_hamiltonian_relative_error"] = float(
        np.linalg.norm(exact_ftt_h - reference_dense) / np.linalg.norm(reference_dense)
    )
    metrics["mace_ftt_hamiltonian_relative_error"] = float(
        np.linalg.norm(predicted_h - reference_dense) / np.linalg.norm(reference_dense)
    )
    metrics["mace_ftt_hermiticity_error"] = float(
        np.linalg.norm(predicted_h - predicted_h.conj().T)
        / max(np.linalg.norm(predicted_h), np.finfo(float).tiny)
    )

    initial = initial_packet(axes, bright_state=args.bright_state)
    times_fs = np.linspace(0.0, args.tmax_fs, args.steps + 1)
    times_au = times_fs / au2fs
    reference_states = propagate(reference_h, initial, times_au)
    predicted_states = propagate(predicted_h, initial, times_au)
    state = driver.state(initial, max_rank=args.state_rank)
    driver.run(
        state,
        dt=float(times_au[1] - times_au[0]),
        steps=args.steps,
        interval=1,
        max_bond=args.state_rank,
        integrator="tdvp2",
        cutoff=1.0e-11,
        progress=False,
        e_ops=driver.projectors(),
    )
    tt_final = driver.dense(driver.final_state)
    normalized_reference = reference_states[-1] / np.linalg.norm(reference_states[-1])
    normalized_predicted = predicted_states[-1] / np.linalg.norm(predicted_states[-1])
    normalized_tt = tt_final / np.linalg.norm(tt_final)
    metrics.update(
        mace_ftt_final_fidelity=float(
            abs(np.vdot(normalized_reference.reshape(-1), normalized_predicted.reshape(-1))) ** 2
        ),
        ttldr_final_fidelity_to_reference=float(
            abs(np.vdot(normalized_reference.reshape(-1), normalized_tt.reshape(-1))) ** 2
        ),
        ttldr_final_fidelity_to_predicted_dense=float(
            abs(np.vdot(normalized_predicted.reshape(-1), normalized_tt.reshape(-1))) ** 2
        ),
        maximum_norm_drift=float(np.max(np.abs(driver.norms - 1.0))),
        operator_ranks=list(map(int, driver.operator_ranks)),
        final_state_ranks=list(map(int, driver.final_state.bond_orders())),
    )
    return {
        "axes": axes,
        "fit": fit,
        "reference": reference,
        "predicted": predicted,
        "times_fs": times_fs,
        "reference_states": reference_states,
        "predicted_states": predicted_states,
        "tt_populations": driver.populations,
        "tt_final": tt_final,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nr", type=int, default=29)
    parser.add_argument("--ntorsion", type=int, default=15)
    parser.add_argument("--rmin", type=float, default=0.82)
    parser.add_argument("--rmax", type=float, default=3.5)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--head-width", type=int, default=48)
    parser.add_argument("--radial-basis", type=int, default=12)
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--target", choices=("parity", "direct"), default="parity")
    parser.add_argument("--tt-rank", type=int, default=16)
    parser.add_argument("--tt-degree", type=int, default=16)
    parser.add_argument("--overlap-rank", type=int, default=12)
    parser.add_argument("--potential-rank", type=int, default=20)
    parser.add_argument("--operator-rank", type=int, default=48)
    parser.add_argument("--state-rank", type=int, default=32)
    parser.add_argument("--bright-state", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--tmax-fs", type=float, default=40.0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/private/tmp/phenol_stage1_mace_ftt_ttldr"),
    )
    args = parser.parse_args()
    if args.nr < 5 or args.ntorsion < 5:
        raise ValueError("stage 1 requires at least five DVR points per coordinate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run(args)
    stem = args.output_dir / "phenol_stage1_mace_ftt_ttldr"
    result["fit"].save(stem.with_suffix(".pt"))
    plot_fit(
        result["axes"], result["reference"], result["predicted"],
        result["fit"].history, stem.with_name(stem.name + "_fit.png"),
    )
    plot_dynamics(
        result["times_fs"], result["axes"], result["reference_states"],
        result["predicted_states"], result["tt_populations"],
        stem.with_name(stem.name + "_dynamics.png"),
    )
    np.savez_compressed(
        stem.with_suffix(".npz"),
        r_angstrom=result["axes"][0],
        phi_radian=result["axes"][1],
        reference_dpem=result["reference"],
        mace_ftt_dpem=result["predicted"],
        times_fs=result["times_fs"],
        reference_states=result["reference_states"],
        mace_ftt_states=result["predicted_states"],
        ttldr_populations=result["tt_populations"],
        ttldr_final=result["tt_final"],
    )
    stem.with_suffix(".json").write_text(
        json.dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"[fit figure] {stem.with_name(stem.name + '_fit.png')}")
    print(f"[dynamics figure] {stem.with_name(stem.name + '_dynamics.png')}")
    print(f"[data] {stem.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
