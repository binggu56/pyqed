#!/usr/bin/env python3
"""Three-dimensional H3+ CASCI -> MACE -> FunctionalTT -> TTLDR benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import expm_multiply

from examples.ldr.h3plus_rectilinear_cgldr import h3plus_geometry
from examples.namd.h3plus_casci_positive_link_fit import (
    HARTREE_TO_EV,
    TARGET_STATES,
    error_metrics,
    selected_blocks,
)
from pyqed.ldr.overlap import procrustes
from pyqed.ml import MACE
from pyqed.mps.functional import FunctionalTT
from pyqed.namd.ttldr import TTLDR
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casci import CASCI, overlap
from pyqed.units import au2fs, proton_mass


COORDINATE_NAMES = ("Qs", "Qx", "Qy")


def geometry(coordinate):
    qs, qx, qy = map(float, coordinate)
    return h3plus_geometry(
        {"Qs": qs, "Qx": qx, "Qy": qy}, symmetry_breaking_offset=0.0
    )


def electronic_point(coordinate, *, basis="sto-3g", nroots=6):
    molecule = Molecule(
        atom=[["H", *xyz] for xyz in geometry(coordinate)],
        unit="bohr",
        basis=basis,
        charge=1,
        spin=0,
    )
    molecule.build()
    mean_field = molecule.RHF().run()
    return CASCI(mean_field, ncas=3, nelecas=2).run(nstates=int(nroots))


def generate_cache(axes, *, basis="sto-3g", nroots=6, reference_coordinate=None):
    axes = tuple(np.asarray(axis, dtype=float) for axis in axes)
    shape = tuple(len(axis) for axis in axes)
    points = np.empty(shape, dtype=object)
    energies = np.empty((*shape, int(nroots)))
    completed = 0
    total = int(np.prod(shape))
    for index in np.ndindex(shape):
        coordinate = tuple(axes[axis][index[axis]] for axis in range(3))
        point = electronic_point(coordinate, basis=basis, nroots=nroots)
        points[index] = point
        energies[index] = np.asarray(point.e_tot, dtype=float)
        completed += 1
        if completed == 1 or completed % max(1, total // 10) == 0:
            print(f"[electronic] {completed}/{total}", flush=True)

    anchor = tuple(int(np.argmin(np.abs(axis))) for axis in axes)
    if reference_coordinate is None:
        reference_point = points[anchor]
        anchor_coordinate = tuple(axes[axis][anchor[axis]] for axis in range(3))
        stored_anchor = anchor
    else:
        anchor_coordinate = tuple(map(float, reference_coordinate))
        reference_point = electronic_point(
            anchor_coordinate, basis=basis, nroots=nroots
        )
        stored_anchor = (-1, -1, -1)

    reference_links = np.empty((*shape, nroots, nroots), dtype=complex)
    for index in np.ndindex(shape):
        reference_links[index] = overlap(points[index], reference_point)

    links = []
    for axis in range(3):
        edge_shape = list(shape)
        edge_shape[axis] -= 1
        values = np.empty((*edge_shape, nroots, nroots), dtype=complex)
        for left in np.ndindex(*edge_shape):
            right = list(left)
            right[axis] += 1
            values[left] = overlap(points[left], points[tuple(right)])
        links.append(values)
    return {
        **{f"grid_{axis}": value for axis, value in enumerate(axes)},
        "energies": energies,
        "reference_links": reference_links,
        **{f"links_{axis}": value for axis, value in enumerate(links)},
        "anchor": np.asarray(stored_anchor, dtype=int),
        "anchor_coordinate": np.asarray(anchor_coordinate, dtype=float),
        "basis": np.asarray(basis),
        "nroots": np.asarray(nroots),
        "target_states": np.asarray(TARGET_STATES),
        "coordinate_names": np.asarray(COORDINATE_NAMES),
    }


def save_cache(filename, data):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez(filename, **data)


def load_cache(filename):
    with np.load(filename) as archive:
        data = {key: archive[key] for key in archive.files}
    if tuple(np.asarray(data["target_states"], dtype=int)) != TARGET_STATES:
        raise ValueError("cache does not retain the H3+ S1/S2 pair")
    return data


def axes_from_cache(data):
    return tuple(np.asarray(data[f"grid_{axis}"], dtype=float) for axis in range(3))


def anchor_aligned_fields(data, *, energy_shift=None):
    axes = axes_from_cache(data)
    shape = tuple(len(axis) for axis in axes)
    reference = selected_blocks(data["reference_links"])
    gauges, reference_positive, reference_singular = procrustes(reference)
    energies = np.asarray(data["energies"], dtype=float)
    if energy_shift is None:
        anchor = tuple(np.asarray(data["anchor"], dtype=int))
        if any(index < 0 for index in anchor):
            raise ValueError("an external anchor requires the training energy shift")
        energy_shift = float(np.mean(energies[anchor][list(TARGET_STATES)]))
    target_energy = energies[..., list(TARGET_STATES)] - float(energy_shift)
    hamiltonian = np.einsum(
        "...ia,...i,...ib->...ab",
        gauges.conj(), target_energy, gauges, optimize=True,
    )
    hamiltonian = 0.5 * (
        hamiltonian + hamiltonian.swapaxes(-1, -2).conj()
    )

    aligned_links = []
    for axis in range(3):
        raw = selected_blocks(data[f"links_{axis}"])
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        aligned_links.append(
            np.einsum(
                "...ia,...ij,...jb->...ab",
                gauges[tuple(left)].conj(),
                raw,
                gauges[tuple(right)],
                optimize=True,
            )
        )
    return {
        "axes": axes,
        "hamiltonian": hamiltonian,
        "links": tuple(aligned_links),
        "gauges": gauges,
        "reference_positive": reference_positive,
        "reference_singular": reference_singular,
        "energy_shift": float(energy_shift),
        "shape": shape,
    }


def product_coordinates(axes):
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([value.reshape(-1) for value in mesh], axis=1)


def edge_coordinates(axes, axis):
    values = list(axes)
    values[axis] = 0.5 * (values[axis][:-1] + values[axis][1:])
    return product_coordinates(tuple(values))


def link_samples(fields):
    shape = fields["shape"]
    pairs = []
    values = []
    for axis, links in enumerate(fields["links"]):
        for left in np.ndindex(links.shape[:-2]):
            right = list(left)
            right[axis] += 1
            pairs.append(
                (np.ravel_multi_index(left, shape), np.ravel_multi_index(right, shape))
            )
            values.append(links[left])
    return np.asarray(pairs, dtype=int), np.asarray(values, dtype=complex)


def infer_reflection_representation(fields, *, samples=8193):
    """Resolve the arbitrary orientation of the degenerate CASCI E' pair."""

    hamiltonian = np.asarray(fields["hamiltonian"])
    reflected = hamiltonian[:, :, ::-1]
    identity = np.eye(2)
    traceless = hamiltonian - 0.5 * np.trace(
        hamiltonian, axis1=-2, axis2=-1
    )[..., None, None] * identity
    reflected = reflected - 0.5 * np.trace(
        reflected, axis1=-2, axis2=-1
    )[..., None, None] * identity
    def objective(angle):
        cosine, sine = np.cos(2.0 * angle), np.sin(2.0 * angle)
        representation = np.asarray([[cosine, sine], [sine, -cosine]])
        predicted = representation @ traceless @ representation
        return float(np.linalg.norm(predicted - reflected))

    best = None
    for angle in np.linspace(0.0, np.pi, int(samples), endpoint=False):
        error = objective(angle)
        if best is None or error < best[0]:
            best = (error, angle)
    spacing = np.pi / int(samples)
    refined = minimize_scalar(
        objective,
        bounds=(best[1] - spacing, best[1] + spacing),
        method="bounded",
        options={"xatol": 1.0e-14},
    )
    angle = float(refined.x)
    cosine, sine = np.cos(2.0 * angle), np.sin(2.0 * angle)
    representation = np.asarray([[cosine, sine], [sine, -cosine]])
    scale = max(float(np.linalg.norm(reflected)), np.finfo(float).tiny)
    return representation, float(refined.fun) / scale


def infer_rotation_representation(fields):
    """Choose the CASCI E' handedness from the two possible C3 generators."""

    axes = fields["axes"]
    hamiltonian = np.asarray(fields["hamiltonian"])
    coordinates = product_coordinates(axes)
    angle = 2.0 * np.pi / 3.0
    coordinate_rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    transformed = coordinates.copy()
    transformed[:, 1:] = coordinates[:, 1:] @ coordinate_rotation.T
    reference = RegularGridInterpolator(
        axes, hamiltonian, bounds_error=False, fill_value=np.nan
    )(transformed)
    mask = np.isfinite(reference[..., 0, 0])
    values = hamiltonian.reshape(-1, 2, 2)[mask]
    reference = reference[mask]
    identity = np.eye(2)
    traceless = reference - 0.5 * np.trace(
        reference, axis1=-2, axis2=-1
    )[..., None, None] * identity
    best = None
    for sign in (1.0, -1.0):
        electronic_angle = sign * angle
        representation = np.asarray(
            [
                [np.cos(electronic_angle), -np.sin(electronic_angle)],
                [np.sin(electronic_angle), np.cos(electronic_angle)],
            ]
        )
        error = float(np.linalg.norm(representation @ values @ representation.T - reference))
        if best is None or error < best[0]:
            best = (error, representation)
    scale = max(float(np.linalg.norm(traceless)), np.finfo(float).tiny)
    return best[1], best[0] / scale


def h3plus_s3_group(feature_rank, reflection, rotation=None):
    """Return aligned D3 actions on (Qs,Qx,Qy), the E' pair, and latent Y."""

    feature_rank = int(feature_rank)
    if feature_rank < 2 or feature_rank % 2:
        raise ValueError("the H3+ S3 feature rank must be a positive multiple of two")
    angle = 2.0 * np.pi / 3.0
    rotation2 = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    coordinate_rotation = np.eye(3)
    coordinate_rotation[1:, 1:] = rotation2
    coordinate_reflection = np.diag([1.0, 1.0, -1.0])
    electronic_rotation = (
        rotation2.astype(complex)
        if rotation is None
        else np.asarray(rotation, dtype=complex)
    )
    electronic_reflection = np.asarray(reflection, dtype=complex)
    coordinates = []
    electronic = []
    for power in range(3):
        coordinates.append(np.linalg.matrix_power(coordinate_rotation, power))
        electronic.append(np.linalg.matrix_power(electronic_rotation, power))
    for power in range(3):
        coordinates.append(
            coordinate_reflection @ np.linalg.matrix_power(coordinate_rotation, power)
        )
        electronic.append(
            electronic_reflection @ np.linalg.matrix_power(electronic_rotation, power)
        )
    electronic = np.asarray(electronic)
    ambient = np.asarray(
        [np.kron(np.eye(feature_rank // 2), value) for value in electronic]
    )
    return {
        "coordinate_representations": np.asarray(coordinates),
        "electronic_representations": electronic,
        "ambient_representations": ambient,
        "tolerance": 2.0e-7,
    }


def field_errors(predicted, reference, *, energy=False):
    mask = np.ones(reference.shape[:-2], dtype=bool)
    return error_metrics(
        np.asarray(predicted).reshape(-1, 2, 2),
        np.asarray(reference).reshape(-1, 2, 2),
        mask,
        scale=HARTREE_TO_EV if energy else 1.0,
    )


def evaluate_mace(fit, fields):
    axes = fields["axes"]
    energy = fit.neural_energy.predict(product_coordinates(axes)).reshape(
        fields["hamiltonian"].shape
    )
    if fit.neural_feature is not None:
        feature = fit.neural_feature.predict(product_coordinates(axes)).reshape(
            *fields["shape"], fit.feature_rank, fit.nstates
        )
        links = []
        for axis, reference in enumerate(fields["links"]):
            left = [slice(None)] * 3
            right = [slice(None)] * 3
            left[axis] = slice(None, -1)
            right[axis] = slice(1, None)
            links.append(
                feature[tuple(left)].conj().swapaxes(-1, -2)
                @ feature[tuple(right)]
            )
        return energy, tuple(links)
    links = []
    for axis, reference in enumerate(fields["links"]):
        links.append(
            fit.neural_links[axis].predict(edge_coordinates(axes, axis)).reshape(
                reference.shape
            )
        )
    return energy, tuple(links)


def symmetry_errors(energy_model, feature_model, group, coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    reference_energy = energy_model.predict(coordinates)
    reference_feature = feature_model.predict(coordinates)
    energy_errors = []
    feature_errors = []
    for coordinate_action, electronic, ambient in zip(
        group["coordinate_representations"],
        group["electronic_representations"],
        group["ambient_representations"],
    ):
        transformed = coordinates @ coordinate_action.T
        actual_energy = energy_model.predict(transformed)
        expected_energy = electronic @ reference_energy @ electronic.conj().T
        actual_feature = feature_model.predict(transformed)
        expected_feature = ambient @ reference_feature @ electronic.conj().T
        energy_errors.append(np.linalg.norm(actual_energy - expected_energy))
        feature_errors.append(np.linalg.norm(actual_feature - expected_feature))
    return {
        "energy_relative_error": float(
            max(energy_errors) / max(np.linalg.norm(reference_energy), 1.0e-15)
        ),
        "feature_relative_error": float(
            max(feature_errors) / max(np.linalg.norm(reference_feature), 1.0e-15)
        ),
    }


def align_external_anchor_sign(fields, predicted_energy):
    """Resolve the constant O(2) gauge of a separately rebuilt degenerate anchor."""

    def objective(angle, parity):
        cosine, sine = np.cos(angle), np.sin(angle)
        transform = np.asarray([[cosine, -sine], [sine, cosine]])
        if parity < 0:
            transform = transform @ np.diag([1.0, -1.0])
        energy = np.einsum(
            "ia,...ij,jb->...ab",
            transform.conj(), fields["hamiltonian"], transform,
            optimize=True,
        )
        return float(np.linalg.norm(np.asarray(predicted_energy) - energy))

    best = None
    samples = 4096
    spacing = 2.0 * np.pi / samples
    for parity in (1.0, -1.0):
        errors = [objective(angle, parity) for angle in np.arange(samples) * spacing]
        index = int(np.argmin(errors))
        center = index * spacing
        refined = minimize_scalar(
            lambda angle: objective(angle, parity),
            bounds=(center - spacing, center + spacing),
            method="bounded",
            options={"xatol": 1.0e-14},
        )
        if best is None or refined.fun < best[0]:
            best = (float(refined.fun), float(refined.x), parity)
    cosine, sine = np.cos(best[1]), np.sin(best[1])
    transform = np.asarray([[cosine, -sine], [sine, cosine]])
    if best[2] < 0:
        transform = transform @ np.diag([1.0, -1.0])
    energy = np.einsum(
        "ia,...ij,jb->...ab",
        transform.conj(), fields["hamiltonian"], transform,
        optimize=True,
    )
    links = tuple(
        np.einsum(
            "ia,...ij,jb->...ab",
            transform.conj(), value, transform,
            optimize=True,
        )
        for value in fields["links"]
    )
    aligned = dict(fields)
    aligned["hamiltonian"] = energy
    aligned["links"] = links
    aligned["external_anchor_relative_sign"] = float(best[2])
    aligned["external_anchor_gauge_angle"] = float(best[1])
    aligned["external_anchor_gauge"] = transform
    return aligned


def exact_fit(axes, energy, links, *, rank=32):
    def field(field_axes, values, hermitian):
        return FunctionalTT(
            degrees=tuple(len(axis) - 1 for axis in field_axes),
            rank=int(rank),
            bounds=tuple((float(axis[0]), float(axis[-1])) for axis in field_axes),
            normalization="frobenius",
            hermitian=hermitian,
        ).fit_grid(field_axes, values)

    models = []
    for axis, values in enumerate(links):
        link_axes = list(axes)
        link_axes[axis] = 0.5 * (link_axes[axis][:-1] + link_axes[axis][1:])
        models.append(field(tuple(link_axes), values, False))
    return SimpleNamespace(
        success=True,
        grids=tuple(axes),
        energy=field(tuple(axes), energy, True),
        links=tuple(models),
        feature=None,
    )


def kinetic_terms(axes, *, mass=proton_mass):
    identities = tuple(np.eye(len(axis)) for axis in axes)
    terms = []
    for active, axis in enumerate(axes):
        spacing = float(np.mean(np.diff(axis)))
        laplacian = np.diag(np.full(len(axis), 2.0))
        laplacian += np.diag(np.full(len(axis) - 1, -1.0), 1)
        laplacian += np.diag(np.full(len(axis) - 1, -1.0), -1)
        factors = list(identities)
        factors[active] = laplacian / (2.0 * float(mass) * spacing**2)
        terms.append((1.0, tuple(factors)))
    return tuple(terms)


def initial_packet(axes, energy, *, momentum=8.0, electronic_state=1):
    """Return a Gaussian packet entirely on local adiabatic S1 or S2."""

    electronic_state = int(electronic_state)
    if electronic_state not in (0, 1):
        raise ValueError("electronic_state must be 0 (S1) or 1 (S2)")
    mesh = np.meshgrid(*axes, indexing="ij")
    center = (-0.03, -0.06, 0.0)
    width = (0.055, 0.045, 0.045)
    nuclear = np.ones(tuple(len(axis) for axis in axes), dtype=complex)
    for coordinate, origin, sigma in zip(mesh, center, width):
        nuclear *= np.exp(-0.25 * ((coordinate - origin) / sigma) ** 2)
    nuclear *= np.exp(1j * float(momentum) * (mesh[1] - center[1]))
    _levels, vectors = np.linalg.eigh(energy)
    state = nuclear[..., None] * vectors[..., :, electronic_state]
    return state / np.linalg.norm(state)


def populations(states):
    states = np.asarray(states)
    return np.sum(np.abs(states.reshape(len(states), -1, 2)) ** 2, axis=1)


def adiabatic_populations(states, energy):
    """Project aligned-frame states onto the local adiabatic S1/S2 fields."""

    states = np.asarray(states)
    _levels, vectors = np.linalg.eigh(np.asarray(energy))
    coefficients = np.einsum(
        "...ai,t...a->t...i", vectors.conj(), states, optimize=True
    )
    return np.sum(
        np.abs(coefficients) ** 2,
        axis=tuple(range(1, coefficients.ndim - 1)),
    )


def nuclear_marginal(state, axis):
    density = np.sum(np.abs(np.asarray(state)) ** 2, axis=-1)
    summed = tuple(index for index in range(3) if index != int(axis))
    return density.sum(axis=summed)


def trajectory_observables(states, axes, energy):
    """Return nuclear moments and the reduced local-adiabatic density matrix."""

    states = np.asarray(states)
    density = np.sum(np.abs(states) ** 2, axis=-1)
    norm = density.sum(axis=tuple(range(1, density.ndim)))
    mesh = np.meshgrid(*axes, indexing="ij")
    means = []
    widths = []
    for coordinate in mesh:
        mean = np.einsum("t..., ...->t", density, coordinate, optimize=True) / norm
        second = (
            np.einsum("t..., ...->t", density, coordinate**2, optimize=True)
            / norm
        )
        means.append(mean)
        widths.append(np.sqrt(np.maximum(second - mean**2, 0.0)))

    _levels, vectors = np.linalg.eigh(np.asarray(energy))
    coefficients = np.einsum(
        "...ai,t...a->t...i", vectors.conj(), states, optimize=True
    )
    electronic_density = np.einsum(
        "t...i,t...j->tij", coefficients, coefficients.conj(), optimize=True
    )
    electronic_density /= np.trace(electronic_density, axis1=1, axis2=2)[:, None, None]
    purity = np.einsum(
        "tij,tji->t", electronic_density, electronic_density, optimize=True
    ).real
    initial = states[0].reshape(-1)
    autocorrelation = np.abs(
        np.einsum(
            "i,ti->t", initial.conj(), states.reshape(len(states), -1),
            optimize=True,
        )
    ) ** 2
    autocorrelation /= np.vdot(initial, initial).real * np.sum(
        np.abs(states.reshape(len(states), -1)) ** 2, axis=1
    )
    return {
        "coordinate_means": np.stack(means, axis=1),
        "coordinate_widths": np.stack(widths, axis=1),
        "electronic_density": electronic_density,
        "electronic_coherence": np.abs(electronic_density[:, 0, 1]),
        "electronic_purity": purity,
        "autocorrelation": autocorrelation,
    }


def run_dynamics(
    fit, fields, *, dt_fs, steps, state_rank, operator_rank, initial_state=1
):
    axes = fields["axes"]
    terms = kinetic_terms(axes)
    predicted_driver = TTLDR.from_fit(
        fit,
        keo=terms,
        overlap_rank=16,
        operator_rank=int(operator_rank),
        potential_rank=24,
    )
    reference_model = exact_fit(
        axes, fields["hamiltonian"], fields["links"], rank=32
    )
    reference_driver = TTLDR.from_fit(
        reference_model,
        keo=terms,
        overlap_rank=24,
        operator_rank=None,
        potential_rank=None,
    )
    predicted_h = predicted_driver.hamiltonian.to_dense()
    reference_h = reference_driver.hamiltonian.to_dense()
    initial = initial_packet(
        axes, fields["hamiltonian"], electronic_state=initial_state
    )
    initial_vector = initial.reshape(-1)
    dt = float(dt_fs) / au2fs
    times = np.linspace(0.0, int(steps) * dt, int(steps) + 1)
    reference_states = expm_multiply(
        -1j * reference_h,
        initial_vector,
        start=times[0],
        stop=times[-1],
        num=len(times),
        endpoint=True,
    ).reshape(len(times), *initial.shape)
    predicted_states = expm_multiply(
        -1j * predicted_h,
        initial_vector,
        start=times[0],
        stop=times[-1],
        num=len(times),
        endpoint=True,
    ).reshape(len(times), *initial.shape)

    state = predicted_driver.state(initial, max_rank=int(state_rank))
    predicted_driver.run(
        state,
        dt=dt,
        steps=int(steps),
        interval=1,
        max_bond=int(state_rank),
        integrator="tdvp2",
        cutoff=1.0e-12,
        progress=False,
        e_ops=predicted_driver.projectors(),
    )
    tt_final = predicted_driver.dense(predicted_driver.final_state)
    reference_final = reference_states[-1]
    predicted_final = predicted_states[-1]
    reference_final /= np.linalg.norm(reference_final)
    predicted_final /= np.linalg.norm(predicted_final)
    tt_final /= np.linalg.norm(tt_final)
    reference_observables = trajectory_observables(
        reference_states, axes, fields["hamiltonian"]
    )
    predicted_observables = trajectory_observables(
        predicted_states, axes, fields["hamiltonian"]
    )
    return {
        "driver": predicted_driver,
        "times_fs": times * au2fs,
        "reference_states": reference_states,
        "predicted_states": predicted_states,
        "tt_final": tt_final,
        "reference_populations": populations(reference_states),
        "predicted_populations": populations(predicted_states),
        "tt_populations": predicted_driver.populations,
        "reference_adiabatic_populations": adiabatic_populations(
            reference_states, fields["hamiltonian"]
        ),
        "predicted_adiabatic_populations": adiabatic_populations(
            predicted_states, fields["hamiltonian"]
        ),
        "tt_final_adiabatic_populations": adiabatic_populations(
            tt_final[None, ...], fields["hamiltonian"]
        )[0],
        "reference_observables": reference_observables,
        "predicted_observables": predicted_observables,
        "initial_electronic_state": int(initial_state),
        "hamiltonian_relative_error": float(
            np.linalg.norm(predicted_h - reference_h) / np.linalg.norm(reference_h)
        ),
        "mace_ftt_final_fidelity": float(
            abs(np.vdot(reference_final.reshape(-1), predicted_final.reshape(-1))) ** 2
        ),
        "ttldr_final_fidelity_to_reference": float(
            abs(np.vdot(reference_final.reshape(-1), tt_final.reshape(-1))) ** 2
        ),
        "ttldr_final_fidelity_to_predicted_dense": float(
            abs(np.vdot(predicted_final.reshape(-1), tt_final.reshape(-1))) ** 2
        ),
        "maximum_ttldr_density_error": float(
            np.max(np.abs(np.abs(tt_final) ** 2 - np.abs(predicted_final) ** 2))
        ),
    }


def plot_results(fit, fields, validation, validation_prediction, dynamics, output):
    axes = fields["axes"]
    validation_axes = validation["axes"]
    validation_energy, validation_links = validation_prediction
    center = len(axes[0]) // 2
    reference_levels = np.linalg.eigvalsh(fields["hamiltonian"])
    mace_energy, mace_links = evaluate_mace(fit, fields)
    mace_levels = np.linalg.eigvalsh(mace_energy)
    gap = (reference_levels[..., 1] - reference_levels[..., 0]) * HARTREE_TO_EV
    mace_gap = (mace_levels[..., 1] - mace_levels[..., 0]) * HARTREE_TO_EV
    energy_error = np.linalg.norm(
        validation_energy - validation["hamiltonian"], axis=(-2, -1)
    ) * HARTREE_TO_EV * 1.0e3
    link_errors = [
        np.linalg.norm(predicted - reference, axis=(-2, -1)).reshape(-1)
        for predicted, reference in zip(validation_links, validation["links"])
    ]

    figure, panels = plt.subplots(2, 3, figsize=(12.2, 7.0), constrained_layout=True)
    panels[0, 0].plot(np.maximum(fit.history, 1.0e-16), color="#0072B2")
    panels[0, 0].set_yscale("log")
    panels[0, 0].set(
        xlabel="Epoch", ylabel="MACE loss", title=r"$S_3$-equivariant MACE-$Y$ training"
    )
    image = panels[0, 1].pcolormesh(
        axes[1], axes[2], gap[center].T, shading="auto", cmap="magma"
    )
    panels[0, 1].contour(
        axes[1], axes[2], mace_gap[center].T,
        levels=7, colors="white", linewidths=0.65,
    )
    panels[0, 1].set(
        xlabel=r"$Q_x$ (bohr)", ylabel=r"$Q_y$ (bohr)",
        title=r"$Q_s=0$: gap reference, MACE contours",
    )
    figure.colorbar(image, ax=panels[0, 1], label="eV")
    validation_center = len(validation_axes[0]) // 2
    image = panels[0, 2].pcolormesh(
        validation_axes[1], validation_axes[2],
        energy_error[validation_center].T, shading="auto", cmap="viridis",
    )
    panels[0, 2].set(
        xlabel=r"$Q_x$ (bohr)", ylabel=r"$Q_y$ (bohr)",
        title="Off-grid MACE matrix error",
    )
    figure.colorbar(image, ax=panels[0, 2], label="meV")

    panels[1, 0].boxplot(
        [np.maximum(values, 1.0e-16) for values in link_errors],
        tick_labels=(r"$L_s$", r"$L_x$", r"$L_y$"),
        showfliers=True,
    )
    panels[1, 0].set_yscale("log")
    panels[1, 0].set(
        ylabel=r"Off-grid $\|\Delta L\|_F$", title=r"$Y^\dagger(R)Y(R')$ validation"
    )
    colors = ("#0072B2", "#D55E00")
    for state, color in enumerate(colors):
        panels[1, 1].plot(
            dynamics["times_fs"], dynamics["reference_populations"][:, state],
            color=color, lw=1.8, label=rf"reference $P_{state}$",
        )
        panels[1, 1].plot(
            dynamics["times_fs"], dynamics["predicted_populations"][:, state],
            color=color, lw=1.2, ls="--", label=rf"MACE+$Y$+FTT $P_{state}$",
        )
        panels[1, 1].scatter(
            dynamics["times_fs"], dynamics["tt_populations"][:, state],
            color=color, s=7, alpha=0.6,
        )
    panels[1, 1].set(
        xlabel="Time (fs)", ylabel="Aligned-frame population",
        ylim=(-0.02, 1.02), title="TTLDR dynamics (dots)",
    )
    panels[1, 1].legend(frameon=False, fontsize=6, ncol=2)
    qx = axes[1]
    panels[1, 2].plot(
        qx, nuclear_marginal(dynamics["reference_states"][-1], 1),
        color="black", lw=1.8, label="reference",
    )
    panels[1, 2].plot(
        qx, nuclear_marginal(dynamics["predicted_states"][-1], 1),
        color="#D55E00", ls="--", lw=1.4, label=r"MACE+$Y$+FTT dense",
    )
    panels[1, 2].scatter(
        qx, nuclear_marginal(dynamics["tt_final"], 1),
        color="#0072B2", s=18, label="TTLDR",
    )
    panels[1, 2].set(
        xlabel=r"$Q_x$ (bohr)", ylabel="Final marginal probability",
        title="Final nuclear packet",
    )
    panels[1, 2].legend(frameon=False, fontsize=7)
    for label, panel in zip("abcdef", panels.flat):
        panel.text(0.02, 0.98, label, transform=panel.transAxes, va="top", fontweight="bold")
        panel.spines[["top", "right"]].set_visible(False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=320)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)

    rank_figure, rank_axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    models = (fit.energy, fit.feature)
    labels = (r"$\bar E$", r"$Y$")
    width = 0.18
    for index, (model, label) in enumerate(zip(models, labels)):
        values = np.asarray(model.ranks_[1:-1], dtype=float)
        rank_axes[0].bar(np.arange(len(values)) + width * index, values, width, label=label)
    rank_axes[0].set(
        xticks=np.arange(2) + 0.5 * width,
        xticklabels=("bond 1", "bond 2"),
        ylabel="FunctionalTT rank", title="Distilled field ranks",
    )
    rank_axes[0].legend(frameon=False, fontsize=7)
    driver = dynamics["driver"]
    component_ranks = [component.bond_orders() for component in driver.components]
    for index, values in enumerate(component_ranks):
        bonds = np.arange(1, len(values) + 1)
        rank_axes[1].plot(bonds, values, "o-", ms=3, label=f"component {index}")
    combined_ranks = driver.hamiltonian.bond_orders()
    rank_axes[1].plot(
        np.arange(1, len(combined_ranks) + 1), combined_ranks,
        "o--", color="black", ms=3, lw=1.2, label="summed MPO",
    )
    rank_axes[1].set(
        xticks=np.arange(1, len(combined_ranks) + 1),
        xlabel="MPO bond", ylabel="bond dimension", title="TTLDR operator components",
    )
    rank_axes[1].legend(frameon=False, fontsize=6)
    for panel in rank_axes:
        panel.spines[["top", "right"]].set_visible(False)
    rank_output = output.with_name(output.stem + "_ranks.png")
    rank_figure.savefig(rank_output, dpi=320)
    rank_figure.savefig(rank_output.with_suffix(".pdf"))
    plt.close(rank_figure)
    return rank_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npts", type=int, default=5)
    parser.add_argument("--qmin", type=float, default=-0.12)
    parser.add_argument("--qmax", type=float, default=0.12)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--head-width", type=int, default=32)
    parser.add_argument("--tt-rank", type=int, default=8)
    parser.add_argument("--tt-degree", type=int, default=4)
    parser.add_argument("--feature-rank", type=int, default=6)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--dt-fs", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--state-rank", type=int, default=24)
    parser.add_argument("--operator-rank", type=int, default=64)
    parser.add_argument(
        "--cache", type=Path,
        default=Path("/private/tmp/h3plus_centered_s3_casci_s1s2_5x5x5.npz"),
    )
    parser.add_argument(
        "--validation-cache", type=Path,
        default=Path("/private/tmp/h3plus_centered_s3_casci_s1s2_offgrid_4x4x4.npz"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/private/tmp/h3plus_3d_mace_ftt_ttldr"),
    )
    args = parser.parse_args()
    if args.npts < 3 or args.npts % 2 == 0:
        raise ValueError("npts must be odd and at least three so the S3-fixed anchor is sampled")
    axes = tuple(np.linspace(args.qmin, args.qmax, args.npts) for _ in range(3))
    if args.cache.exists() and not args.force:
        data = load_cache(args.cache)
        for cached, requested in zip(axes_from_cache(data), axes):
            np.testing.assert_allclose(cached, requested)
        print(f"[cache] loaded {args.cache}", flush=True)
    else:
        data = generate_cache(axes, basis=args.basis, nroots=6)
        save_cache(args.cache, data)
        print(f"[cache] saved {args.cache}", flush=True)
    fields = anchor_aligned_fields(data)
    reflection, reflection_residual = infer_reflection_representation(fields)
    rotation, rotation_residual = infer_rotation_representation(fields)
    finite_group = h3plus_s3_group(args.feature_rank, reflection, rotation)
    coordinates = product_coordinates(fields["axes"])
    pairs, sampled_links = link_samples(fields)

    fit = MACE(
        fields["axes"],
        ("H", "H", "H"),
        geometry,
        2,
        chart_features=True,
        geometry_units="bohr",
        channels=args.channels,
        max_ell=2,
        interactions=2,
        correlation=2,
        radial_basis=6,
        radial_mlp=(args.head_width, args.head_width),
        cutoff=4.0,
    ).fit_y(
        (coordinates, fields["hamiltonian"].reshape(-1, 2, 2)),
        coordinates,
        pairs,
        sampled_links,
        feature_rank=args.feature_rank,
        feature_objective="links-only",
        ambient_representation="full",
        energy_representation="direct",
        finite_group=finite_group,
        hidden=(args.head_width, args.head_width),
        epochs=args.epochs,
        learning_rate=2.0e-3,
        weight_decay=1.0e-8,
        frame_fraction=0.35,
        ambient_fraction=0.20,
        smoothness=1.0e-5,
        seed=args.seed,
        distill=True,
        tt_rank=args.tt_rank,
        tt_degree=args.tt_degree,
    )

    centers = tuple(0.5 * (axis[:-1] + axis[1:]) for axis in axes)
    anchor_coordinate = tuple(np.asarray(data["anchor_coordinate"], dtype=float))
    if args.validation_cache.exists() and not args.force:
        validation_data = load_cache(args.validation_cache)
        for cached, requested in zip(axes_from_cache(validation_data), centers):
            np.testing.assert_allclose(cached, requested)
        np.testing.assert_allclose(validation_data["anchor_coordinate"], anchor_coordinate)
        print(f"[cache] loaded {args.validation_cache}", flush=True)
    else:
        validation_data = generate_cache(
            centers,
            basis=args.basis,
            nroots=6,
            reference_coordinate=anchor_coordinate,
        )
        save_cache(args.validation_cache, validation_data)
        print(f"[cache] saved {args.validation_cache}", flush=True)
    validation = anchor_aligned_fields(
        validation_data, energy_shift=fields["energy_shift"]
    )
    training_prediction = evaluate_mace(fit, fields)
    validation_prediction = evaluate_mace(fit, validation)
    validation = align_external_anchor_sign(validation, validation_prediction[0])
    radius = 0.65 * min(abs(args.qmin), abs(args.qmax))
    angles = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    symmetry_coordinates = np.asarray(
        [
            (qs, radius * np.cos(angle), radius * np.sin(angle))
            for qs in (-0.5 * radius, 0.0, 0.5 * radius)
            for angle in angles
        ]
    )
    metrics = {
        "mace_training_energy": field_errors(
            training_prediction[0], fields["hamiltonian"], energy=True
        ),
        "mace_offgrid_energy": field_errors(
            validation_prediction[0], validation["hamiltonian"], energy=True
        ),
        "mace_training_links": [
            field_errors(predicted, reference)
            for predicted, reference in zip(training_prediction[1], fields["links"])
        ],
        "mace_offgrid_links": [
            field_errors(predicted, reference)
            for predicted, reference in zip(validation_prediction[1], validation["links"])
        ],
        "minimum_target_neighbor_singular_value": float(min(
            np.min(np.linalg.svd(selected_blocks(data[f"links_{axis}"]), compute_uv=False))
            for axis in range(3)
        )),
        "minimum_reference_singular_value": float(np.min(fields["reference_singular"])),
        "minimum_gap_to_S0_eV": float(
            np.min(data["energies"][..., 1] - data["energies"][..., 0]) * HARTREE_TO_EV
        ),
        "minimum_gap_to_S3_eV": float(
            np.min(data["energies"][..., 3] - data["energies"][..., 2]) * HARTREE_TO_EV
        ),
        "mace_final_loss": float(fit.history[-1]),
        "ab_initio_reflection_covariance_residual": float(reflection_residual),
        "ab_initio_rotation_covariance_interpolation_residual": float(
            rotation_residual
        ),
        "neural_s3_covariance": symmetry_errors(
            fit.neural_energy,
            fit.neural_feature,
            finite_group,
            symmetry_coordinates,
        ),
        "ftt_s3_covariance": symmetry_errors(
            fit.energy,
            fit.feature,
            finite_group,
            symmetry_coordinates,
        ),
        "validation_external_anchor_relative_sign": float(
            validation["external_anchor_relative_sign"]
        ),
        "validation_external_anchor_gauge_angle": float(
            validation["external_anchor_gauge_angle"]
        ),
        "validation_external_anchor_gauge": np.asarray(
            validation["external_anchor_gauge"]
        ).real.tolist(),
        "ftt_distillation": fit.info["distillation"],
    }
    dynamics = run_dynamics(
        fit,
        fields,
        dt_fs=args.dt_fs,
        steps=args.steps,
        state_rank=args.state_rank,
        operator_rank=args.operator_rank,
    )
    for key in (
        "hamiltonian_relative_error",
        "mace_ftt_final_fidelity",
        "ttldr_final_fidelity_to_reference",
        "ttldr_final_fidelity_to_predicted_dense",
        "maximum_ttldr_density_error",
    ):
        metrics[key] = dynamics[key]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "h3plus_3d_mace_ftt_ttldr"
    fit.save(stem.with_suffix(".pt"))
    np.savez(
        stem.with_suffix(".npz"),
        grids=np.asarray(fields["axes"]),
        reference_energy=fields["hamiltonian"],
        mace_energy=training_prediction[0],
        times_fs=dynamics["times_fs"],
        reference_populations=dynamics["reference_populations"],
        predicted_populations=dynamics["predicted_populations"],
        ttldr_populations=dynamics["tt_populations"],
        reference_final=dynamics["reference_states"][-1],
        predicted_final=dynamics["predicted_states"][-1],
        ttldr_final=dynamics["tt_final"],
        reference_coordinate_means=dynamics["reference_observables"][
            "coordinate_means"
        ],
        predicted_coordinate_means=dynamics["predicted_observables"][
            "coordinate_means"
        ],
        reference_coordinate_widths=dynamics["reference_observables"][
            "coordinate_widths"
        ],
        predicted_coordinate_widths=dynamics["predicted_observables"][
            "coordinate_widths"
        ],
        reference_electronic_density=dynamics["reference_observables"][
            "electronic_density"
        ],
        predicted_electronic_density=dynamics["predicted_observables"][
            "electronic_density"
        ],
        reference_autocorrelation=dynamics["reference_observables"][
            "autocorrelation"
        ],
        predicted_autocorrelation=dynamics["predicted_observables"][
            "autocorrelation"
        ],
    )
    metrics_path = stem.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    figure_path = stem.with_suffix(".png")
    rank_path = plot_results(
        fit, fields, validation, validation_prediction, dynamics, figure_path
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"figure: {figure_path}", flush=True)
    print(f"ranks: {rank_path}", flush=True)
    print(f"checkpoint: {stem.with_suffix('.pt')}", flush=True)


if __name__ == "__main__":
    main()
