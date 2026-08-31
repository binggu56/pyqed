#!/usr/bin/env python3
"""Benchmark 3D H3+ FCI/aug-cc-pVDZ TNLDR against direct-product LDR."""

import json
from pathlib import Path
from time import perf_counter

from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

from pyqed.dvr import DVR, SineDVR
from pyqed.ldr import AbInitioFit, Coord, keo
from pyqed.ml import MACE
from pyqed.namd import TNLDR
from pyqed.qchem import Molecule
from pyqed.units import au2ev, au2fs


stem = "h3plus_fci_augccpvdz_3d_s3_mace_ftt_vs_direct_7x7x7_20fs"
output = Path("/private/tmp") / stem
output.mkdir(parents=True, exist_ok=True)
tmax_fs = 20.0
dt_fs = 0.02
bond_length = 1.65


def geometry(q):
    """Map breathing and two branching coordinates to Cartesian H3+."""
    root3 = jnp.sqrt(3.0)
    triangle = jnp.asarray(
        ((-0.5, -0.5 / root3, 0.0),
         (0.5, -0.5 / root3, 0.0),
         (0.0, 1.0 / root3, 0.0))
    )
    stretch = triangle.at[:, :2].set(
        triangle[:, :2] @ jnp.diag(jnp.asarray((1.0, -1.0)))
    )
    shear = triangle.at[:, :2].set(
        triangle[:, :2] @ jnp.asarray(((0.0, 1.0), (1.0, 0.0)))
    )
    qs, qx, qy = q
    qb = -0.20 + qs
    return (
        (bond_length + qb) * triangle
        + qx * stretch
        + qy * shear
    )


def mace_geometry(q):
    root3 = np.sqrt(3.0)
    triangle = np.asarray(
        ((-0.5, -0.5 / root3, 0.0),
         (0.5, -0.5 / root3, 0.0),
         (0.0, 1.0 / root3, 0.0))
    )
    stretch = triangle.copy()
    stretch[:, :2] = triangle[:, :2] @ np.diag((1.0, -1.0))
    shear = triangle.copy()
    shear[:, :2] = triangle[:, :2] @ np.asarray(((0.0, 1.0), (1.0, 0.0)))
    qs, qx, qy = q
    return (bond_length - 0.20 + qs) * triangle + qx * stretch + qy * shear


bounds = ((-0.05, 0.05), (-0.12, 0.12), (-0.12, 0.12))
grid = DVR.from_axes(
    tuple(SineDVR(lower, upper, 7) for lower, upper in bounds),
    names=("Qs", "Qx", "Qy"),
)
coord = Coord(to_cartesian=geometry, bounds=bounds)
reference_geometry = np.asarray(geometry((0.0, 0.0, 0.0)))
mol = Molecule(
    atom=list(zip(("H", "H", "H"), reference_geometry)),
    charge=1,
    spin=0,
    unit="bohr",
    basis="aug-cc-pvdz",
).build(eri="dense")
mf = mol.RHF().run()
mc = mol.casci(mol.nao, 2, nstates=3, mf=mf).run(nstates=3)

started = perf_counter()
sampler = AbInitioFit(
    mc,
    coord=coord,
    states=(1, 2),
    fit_options={"degrees": (6, 10, 10), "rank": 64},
    database=output / "electronic.sqlite",
    workers=1,
    progress=True,
)


def group_coordinates():
    angle = 2.0 * np.pi / 3.0
    rotation = np.eye(3)
    rotation[1:, 1:] = (
        (np.cos(angle), -np.sin(angle)),
        (np.sin(angle), np.cos(angle)),
    )
    reflection = np.diag((1.0, 1.0, -1.0))
    return np.asarray(
        [np.linalg.matrix_power(rotation, power) for power in range(3)]
        + [
            reflection @ np.linalg.matrix_power(rotation, power)
            for power in range(3)
        ]
    )


def infer_s3(orbit_hamiltonians, feature_rank=6):
    identity = np.eye(2)

    def traceless(value):
        return value - 0.5 * np.trace(value) * identity

    base = orbit_hamiltonians[0]
    rotated = orbit_hamiltonians[1]
    reflected = orbit_hamiltonians[3]

    def objective(angle):
        cosine, sine = np.cos(2.0 * angle), np.sin(2.0 * angle)
        representation = np.asarray(((cosine, sine), (sine, -cosine)))
        return np.linalg.norm(
            representation @ traceless(base) @ representation
            - traceless(reflected)
        )

    angles = np.linspace(0.0, np.pi, 2048, endpoint=False)
    center = angles[int(np.argmin([objective(value) for value in angles]))]
    spacing = np.pi / len(angles)
    result = minimize_scalar(
        objective,
        bounds=(center - spacing, center + spacing),
        method="bounded",
    )
    cosine, sine = np.cos(2.0 * result.x), np.sin(2.0 * result.x)
    electronic_reflection = np.asarray(((cosine, sine), (sine, -cosine)))
    angle = 2.0 * np.pi / 3.0
    rotations = []
    for sign in (1.0, -1.0):
        value = sign * angle
        representation = np.asarray(
            ((np.cos(value), -np.sin(value)),
             (np.sin(value), np.cos(value)))
        )
        rotations.append(
            (np.linalg.norm(representation @ base @ representation.T - rotated),
             representation)
        )
    electronic_rotation = min(rotations, key=lambda item: item[0])[1]
    electronic = np.asarray(
        [np.linalg.matrix_power(electronic_rotation, power) for power in range(3)]
        + [
            electronic_reflection
            @ np.linalg.matrix_power(electronic_rotation, power)
            for power in range(3)
        ]
    )
    return {
        "coordinate_representations": group_coordinates(),
        "electronic_representations": electronic,
        "ambient_representations": np.asarray(
            [np.kron(np.eye(feature_rank // 2), value) for value in electronic]
        ),
        "tolerance": 2.0e-7,
    }


def graph_pairs(coordinates, neighbors=3):
    coordinates = np.asarray(coordinates)
    scale = np.ptp(coordinates, axis=0)
    scale[scale < 1.0e-12] = 1.0
    distances = cdist(coordinates / scale, coordinates / scale)
    tree = minimum_spanning_tree(distances).tocoo()
    pairs = {
        tuple(sorted((int(left), int(right))))
        for left, right in zip(tree.row, tree.col)
    }
    np.fill_diagonal(distances, np.inf)
    for left in range(len(coordinates)):
        nearest = np.argpartition(distances[left], neighbors - 1)[:neighbors]
        pairs.update(tuple(sorted((left, int(right)))) for right in nearest)
    return np.asarray(sorted(pairs), dtype=int)


def random_coordinates(random, count):
    qs = random.uniform(bounds[0][0], bounds[0][1], count)
    radius = 0.105 * np.sqrt(random.random(count))
    angle = random.uniform(0.0, np.pi / 3.0, count)
    return np.column_stack((qs, radius * np.cos(angle), radius * np.sin(angle)))


def train_mace(
    coordinates, finite_group, previous=None, epochs=400, energy_weight=50.0
):
    pairs = graph_pairs(coordinates)
    fields = sampler.continuous_fields(coordinates, pairs)
    fit = MACE(
        grid.x,
        ("H", "H", "H"),
        mace_geometry,
        2,
        chart_features=True,
        chart_bounds=bounds,
        geometry_units="bohr",
        channels=8,
        max_ell=2,
        interactions=2,
        correlation=2,
        radial_basis=6,
        radial_mlp=(32, 32),
        cutoff=4.0,
    ).fit_y(
        (coordinates, fields["hamiltonians"]),
        coordinates,
        pairs,
        fields["links"],
        feature_rank=6,
        feature_objective="links-only",
        ambient_representation="full",
        energy_representation="direct",
        finite_group=finite_group,
        hidden=(32, 32),
        epochs=epochs,
        learning_rate=2.0e-3 if previous is None else 6.0e-4,
        weight_decay=1.0e-8,
        frame_fraction=0.35 if previous is None else 0.0,
        ambient_fraction=0.20 if previous is None else 0.0,
        smoothness=1.0e-5,
        energy_weight=energy_weight,
        initial_fit=previous,
        seed=7,
        distill=False,
    )
    return fit, fields


def assess(fit, fields):
    predicted_h = fit.neural_energy.predict(fields["coordinates"])
    feature = fit.neural_feature.predict(fields["coordinates"])
    pairs = fields["pairs"]
    predicted_links = (
        feature[pairs[:, 0]].conj().swapaxes(-1, -2)
        @ feature[pairs[:, 1]]
    )
    h_errors = np.linalg.norm(
        predicted_h - fields["hamiltonians"], axis=(-2, -1)
    )
    return {
        "maximum_hamiltonian_error": float(np.max(h_errors)),
        "rms_hamiltonian_error": float(np.sqrt(np.mean(h_errors**2))),
        "relative_hamiltonian_error": float(
            np.linalg.norm(predicted_h - fields["hamiltonians"])
            / np.linalg.norm(fields["hamiltonians"])
        ),
        "relative_link_error": float(
            np.linalg.norm(predicted_links - fields["links"])
            / np.linalg.norm(fields["links"])
        ),
    }


def covariance_error(energy, feature, finite_group, coordinates):
    reference_h = energy.predict(coordinates)
    reference_y = feature.predict(coordinates)
    h_errors = []
    y_errors = []
    for coordinate_action, electronic, ambient in zip(
        finite_group["coordinate_representations"],
        finite_group["electronic_representations"],
        finite_group["ambient_representations"],
    ):
        transformed = coordinates @ coordinate_action.T
        h_errors.append(
            np.linalg.norm(
                energy.predict(transformed)
                - electronic @ reference_h @ electronic.conj().T
            )
        )
        y_errors.append(
            np.linalg.norm(
                feature.predict(transformed)
                - ambient @ reference_y @ electronic.conj().T
            )
        )
    return {
        "hamiltonian_relative_error": float(
            max(h_errors) / np.linalg.norm(reference_h)
        ),
        "feature_relative_error": float(
            max(y_errors) / np.linalg.norm(reference_y)
        ),
    }


def raw_covariance_error(sampler, finite_group, coordinates):
    reference = sampler.continuous_fields(coordinates)["hamiltonians"]
    matrix_errors = []
    spectral_errors = []
    for coordinate_action, electronic in zip(
        finite_group["coordinate_representations"],
        finite_group["electronic_representations"],
    ):
        transformed = sampler.continuous_fields(
            coordinates @ coordinate_action.T
        )["hamiltonians"]
        covariant = electronic @ reference @ electronic.conj().T
        matrix_errors.append(np.linalg.norm(transformed - covariant))
        spectral_errors.append(
            np.linalg.norm(
                np.linalg.eigvalsh(transformed)
                - np.linalg.eigvalsh(reference)
            )
        )
    return {
        "hamiltonian_relative_error": float(
            max(matrix_errors) / np.linalg.norm(reference)
        ),
        "spectrum_relative_error": float(
            max(spectral_errors) / np.linalg.norm(np.linalg.eigvalsh(reference))
        ),
    }


random = np.random.default_rng(7)
coordinate_group = group_coordinates()
calibration_base = np.asarray((0.0, 0.048, 0.019))
calibration = calibration_base @ coordinate_group.transpose(0, 2, 1)
base_mesh = np.stack(
    np.meshgrid(*sampler.grids, indexing="ij"), axis=-1
).reshape(-1, 3)
dynamics_coordinates = np.stack(
    np.meshgrid(*grid.x, indexing="ij"), axis=-1
).reshape(-1, coord.ndim)
selected = random.choice(len(base_mesh), size=72, replace=False)
training_coordinates = np.unique(
    np.vstack((np.zeros((1, 3)), calibration, base_mesh[selected])), axis=0
)
calibration_fields = sampler.continuous_fields(calibration)
finite_group = infer_s3(calibration_fields["hamiltonians"])

validation_coordinates = random_coordinates(random, 64)
validation_pairs = graph_pairs(validation_coordinates)
validation_fields = sampler.continuous_fields(
    validation_coordinates, validation_pairs
)
candidate_pools = [random_coordinates(random, 64) for _ in range(6)]
checkpoint = output / "adaptive_mace_y.pt"
history_path = output / "adaptive_history.json"
if checkpoint.is_file() and history_path.is_file():
    fit = MACE.load(checkpoint, mace_geometry)
    adaptive_history = json.loads(history_path.read_text())
else:
    fit, training_fields = train_mace(training_coordinates, finite_group)
    adaptive_history = [{"round": 0, **assess(fit, validation_fields)}]
    print(f"adaptive MACE round 0: {adaptive_history[-1]}", flush=True)
    for adaptive_round in range(1, 7):
        if (
            adaptive_history[-1]["maximum_hamiltonian_error"] <= 2.0e-4
            and adaptive_history[-1]["relative_link_error"] <= 1.0e-2
        ):
            break
        candidates = candidate_pools[adaptive_round - 1]
        candidate_fields = sampler.continuous_fields(candidates)
        candidate_error = np.linalg.norm(
            fit.neural_energy.predict(candidates)
            - candidate_fields["hamiltonians"],
            axis=(-2, -1),
        )
        worst = np.argsort(candidate_error)[-16:]
        training_coordinates = np.vstack((training_coordinates, candidates[worst]))
        fit, training_fields = train_mace(
            training_coordinates, finite_group, previous=fit, epochs=250
        )
        adaptive_history.append(
            {
                "round": adaptive_round,
                "acquisition_maximum_hamiltonian_error": float(
                    np.max(candidate_error)
                ),
                **assess(fit, validation_fields),
            }
        )
        print(
            f"adaptive MACE round {adaptive_round}: {adaptive_history[-1]}",
            flush=True,
        )
    fit.distill_y(rank=48, degree=6, method="grid", seed=7)
    checkpoint = fit.save(checkpoint)
    history_path.write_text(json.dumps(adaptive_history, indent=2) + "\n")
grid_fields = sampler.continuous_fields(
    dynamics_coordinates, graph_pairs(dynamics_coordinates)
)
unseen_grid_assessment = assess(fit, grid_fields)
fitting_grid_snapshot = grid_fields["hamiltonians"]
adaptive_info = {
    "architecture": "ab-initio-database -> S3-MACE-(H,Y) -> FTT -> TNLDR",
    "validation_is_independent": True,
    "dynamics_grid_used_for_training": False,
    "validation_points": len(validation_coordinates),
    "unseen_dynamics_grid_points": len(dynamics_coordinates),
    "unseen_dynamics_grid": unseen_grid_assessment,
    "training_points": int(fit.info["energy_samples"]),
    "rounds": len(adaptive_history) - 1,
    "converged": bool(
        adaptive_history[-1]["maximum_hamiltonian_error"] <= 2.0e-4
        and adaptive_history[-1]["relative_link_error"] <= 1.0e-2
    ),
    "history": adaptive_history,
}
symmetry_random = np.random.default_rng(91)
symmetry_radius = 0.8 * min(
    abs(float(fit.grids[1][0])), abs(float(fit.grids[1][-1])),
    abs(float(fit.grids[2][0])), abs(float(fit.grids[2][-1])),
)
symmetry_angles = symmetry_random.uniform(0.0, 2.0 * np.pi, 12)
symmetry_radii = symmetry_radius * np.sqrt(symmetry_random.random(12))
symmetry_probes = np.column_stack(
    (
        symmetry_random.uniform(fit.grids[0][0], fit.grids[0][-1], 12),
        symmetry_radii * np.cos(symmetry_angles),
        symmetry_radii * np.sin(symmetry_angles),
    )
)
raw_target_covariance = raw_covariance_error(
    sampler, finite_group, symmetry_probes
)
mace_covariance = covariance_error(
    fit.neural_energy, fit.neural_feature, finite_group, symmetry_probes
)
ftt_covariance = covariance_error(
    fit.energy, fit.feature, finite_group, symmetry_probes
)
nuclear_keo = keo.podolsky().bind(
    coord,
    grid=grid,
    molecule=mol,
)
direct = sampler.direct_product(
    grid,
    keo=nuclear_keo,
    workers=1,
    progress=True,
    energy_shift=sampler.energy_shift,
)
coordinates = dynamics_coordinates
frame_blocks = np.swapaxes(
    direct.procrustes_gauges.conj(), -1, -2
)
flat_frames = frame_blocks.reshape(-1, direct.nstates, direct.nstates)
exact_blocks = np.einsum(
    "nai,ni,nbi->nab",
    flat_frames,
    direct.energies.reshape(-1, direct.nstates),
    flat_frames.conj(),
    optimize=True,
)
raw_grid_blocks = sampler.continuous_fields(coordinates)["hamiltonians"]
raw_grid_blocks_with_pairs = sampler.continuous_fields(
    coordinates, graph_pairs(coordinates)
)["hamiltonians"]
neural_grid_blocks = np.asarray(fit.neural_energy.predict(coordinates))
ftt_grid_blocks_before_build = np.asarray(fit.energy.predict(coordinates))
tnldr = TNLDR(
    fit,
    grid=grid,
    coord=coord,
    keo=nuclear_keo,
    overlap_rank=32,
    operator_rank=128,
).build()
build_seconds = perf_counter() - started

fitted_blocks = np.asarray(fit.energy.predict(coordinates))
fitted_energies, fitted_vectors = np.linalg.eigh(fitted_blocks)
exact_energies = np.sort(direct.energies.reshape(-1, direct.nstates), axis=-1)
feature_values = np.asarray(fit.feature.predict(coordinates))
point_ids = {point: index for index, point in enumerate(np.ndindex(grid.shape))}
exact_links = []
fitted_links = []
for left in np.ndindex(grid.shape):
    for axis, size in enumerate(grid.shape):
        if left[axis] + 1 >= size:
            continue
        right = list(left)
        right[axis] += 1
        right = tuple(right)
        exact_links.append(direct.links[(axis, left)])
        fitted_links.append(
            feature_values[point_ids[left]].conj().T
            @ feature_values[point_ids[right]]
        )
link_singular_value_error = np.abs(
    np.linalg.svd(np.asarray(fitted_links), compute_uv=False)
    - np.linalg.svd(np.asarray(exact_links), compute_uv=False)
)

center = np.asarray((0.0, -0.08, 0.0))
sigma = np.asarray((0.025, 0.03, 0.03))
momentum = np.asarray((0.0, 0.70, 0.0))
factors = tuple(
    np.where(
        np.abs(axis - value) <= 3.0 * width,
        np.exp(-0.25 * ((axis - value) / width) ** 2 + 1j * kick * axis),
        0.0,
    )
    for axis, value, width, kick in zip(grid.x, center, sigma, momentum)
)
envelope = np.einsum("i,j,k->ijk", *factors)
packet_anchor = tuple(
    int(np.argmin(np.abs(axis - value)))
    for axis, value in zip(grid.x, center)
)
direct_packet = direct.wavepacket(
    envelope,
    state=1,
    anchor=packet_anchor,
    support_threshold=1.0e-12,
)
energy_matrix_relative_error = np.linalg.norm(
    fitted_blocks - exact_blocks
) / np.linalg.norm(exact_blocks)
raw_grid_gauge_error = np.linalg.norm(
    raw_grid_blocks - exact_blocks
) / np.linalg.norm(exact_blocks)
raw_grid_batch_consistency_error = np.linalg.norm(
    raw_grid_blocks_with_pairs - raw_grid_blocks
) / np.linalg.norm(raw_grid_blocks)
raw_grid_target_drift_error = np.linalg.norm(
    raw_grid_blocks - fitting_grid_snapshot
) / np.linalg.norm(fitting_grid_snapshot)
mace_grid_hamiltonian_error = np.linalg.norm(
    fitted_blocks - raw_grid_blocks
) / np.linalg.norm(raw_grid_blocks)
neural_grid_hamiltonian_error = np.linalg.norm(
    neural_grid_blocks - raw_grid_blocks
) / np.linalg.norm(raw_grid_blocks)
ftt_grid_distillation_error = np.linalg.norm(
    ftt_grid_blocks_before_build - neural_grid_blocks
) / np.linalg.norm(neural_grid_blocks)
ftt_build_mutation_error = np.linalg.norm(
    fitted_blocks - ftt_grid_blocks_before_build
) / np.linalg.norm(ftt_grid_blocks_before_build)
projector_errors = []
direct_orders = np.argsort(
    direct.energies.reshape(-1, direct.nstates), axis=-1
)
for state in range(direct.nstates):
    exact_vectors = np.take_along_axis(
        flat_frames,
        np.broadcast_to(
            direct_orders[:, state, None, None],
            (direct.ngrid, direct.nstates, 1),
        ),
        axis=-1,
    )[..., 0]
    exact_projector = np.einsum(
        "na,nb->nab", exact_vectors, exact_vectors.conj(), optimize=True
    )
    fitted_vector = fitted_vectors[:, :, state]
    fitted_projector = np.einsum(
        "na,nb->nab", fitted_vector, fitted_vector.conj(), optimize=True
    )
    projector_errors.append(
        np.linalg.norm(fitted_projector - exact_projector)
        / np.linalg.norm(exact_projector)
    )
working_packet = np.einsum(
    "...ai,...i->...a", frame_blocks, direct_packet, optimize=True
)
tn_packet = tnldr.state(
    working_packet,
    max_rank=48,
    physical=False,
)
state_compression_error = np.linalg.norm(
    tnldr.dense(tn_packet, physical=False) - working_packet
)
projector_1, projector_1_info = tnldr.adiabatic_projector(
    1,
    method="dense",
    max_rank=None,
)
projector_0, projector_info = tnldr.adiabatic_projector(
    0,
    method="dense",
    max_rank=None,
)

frame = np.zeros((direct.size, direct.size), dtype=complex)
for point, block in enumerate(frame_blocks.reshape(-1, direct.nstates, direct.nstates)):
    indices = slice(point * direct.nstates, (point + 1) * direct.nstates)
    frame[indices, indices] = block
direct_hamiltonian = direct.hamiltonian(matrix_free=False)
tnldr_hamiltonian = tnldr.hamiltonian.to_dense()
transformed_direct_hamiltonian = frame @ direct_hamiltonian @ frame.conj().T
hamiltonian_relative_error = np.linalg.norm(
    tnldr_hamiltonian - transformed_direct_hamiltonian
) / np.linalg.norm(transformed_direct_hamiltonian)
spectral_error = np.max(
    np.abs(
        np.linalg.eigvalsh(tnldr_hamiltonian)
        - np.linalg.eigvalsh(direct_hamiltonian)
    )
)
aligned_exact_links = []
for left in np.ndindex(grid.shape):
    for axis, size in enumerate(grid.shape):
        if left[axis] + 1 >= size:
            continue
        right = list(left)
        right[axis] += 1
        right = tuple(right)
        aligned_exact_links.append(
            flat_frames[point_ids[left]]
            @ direct.links[(axis, left)]
            @ flat_frames[point_ids[right]].conj().T
        )
nearest_link_relative_error = np.linalg.norm(
    np.asarray(fitted_links) - np.asarray(aligned_exact_links)
) / np.linalg.norm(aligned_exact_links)

dt = dt_fs / au2fs
nsteps = round(tmax_fs / dt_fs)
started = perf_counter()
direct.run(direct_packet, dt=dt, nsteps=nsteps, matrix_free=False)
direct_seconds = perf_counter() - started
direct_order = np.argsort(direct.energies, axis=-1)
direct_populations = np.sum(
    np.take_along_axis(
        np.abs(direct.states) ** 2,
        direct_order[None, ...],
        axis=-1,
    ),
    axis=(1, 2, 3),
)

started = perf_counter()
tnldr.run(
    tn_packet,
    dt=dt,
    steps=nsteps,
    interval=1,
    max_bond=96,
    integrator="tdvp2",
    e_ops=(projector_0, projector_1),
    progress=False,
)
tn_seconds = perf_counter() - started

time_fs = direct.times * au2fs
population_error = np.abs(tnldr.populations - direct_populations)
direct_marginal = np.sum(np.abs(direct.states[-1]) ** 2, axis=(1, 2, 3))
tn_dense = tnldr.dense(tnldr.final_state)
tn_marginal = np.sum(np.abs(tn_dense) ** 2, axis=(1, 2, 3))
direct_marginal /= direct_marginal.sum()
tn_marginal /= tn_marginal.sum()
summary = {
    "electronic_reference": {
        "method": "full CI on RHF orbitals",
        "basis": "aug-cc-pVDZ",
        "electrons": 2,
        "spatial_orbitals": int(mol.nao),
        "roots": 3,
        "selected_zero_based_roots": [1, 2],
    },
    "grid": list(grid.shape),
    "coordinate_names": list(grid.names),
    "bounds": [list(interval) for interval in bounds],
    "tmax_fs": tmax_fs,
    "dt_fs": dt_fs,
    "steps": nsteps,
    "electronic_states": [1, 2],
    "initial_physical_state": 2,
    "initial_center": center.tolist(),
    "initial_sigma": sigma.tolist(),
    "initial_momentum": momentum.tolist(),
    "initial_anchor": list(packet_anchor),
    "direct_dimension": direct.size,
    "adaptive_mace": adaptive_info,
    "fit_feature_rank": int(fit.feature_rank),
    "fit_backend": fit.info["backend"],
    "ftt_distillation": fit.info["distillation"],
    "mace_checkpoint": str(checkpoint),
    "link_mpo_rank": int(tnldr.overlap_rank),
    "fit_gauge": "anchor-procrustes",
    "symmetry": "exact-S3-finite-group-projection",
    "raw_target_s3_covariance": raw_target_covariance,
    "mace_s3_covariance": mace_covariance,
    "ftt_s3_covariance": ftt_covariance,
    "unitarize_links": False,
    "direct_geometries": int(direct.direct_product_info["geometries"]),
    "direct_overlap_pairs": int(direct.direct_product_info["overlap_pairs"]),
    "direct_overlap_representation": direct.direct_product_info[
        "overlap_representation"
    ],
    "direct_gauge": direct.direct_product_info["gauge"],
    "direct_database_hits": int(direct.direct_product_info["database_hits"]),
    "direct_database_writes": int(direct.direct_product_info["database_writes"]),
    "database_records": int(direct.database_info["records"]),
    "max_population_error": float(np.max(population_error)),
    "final_population_error": population_error[-1].tolist(),
    "max_fitted_pes_error_hartree": float(
        np.max(np.abs(fitted_energies - exact_energies))
    ),
    "energy_matrix_relative_error": float(energy_matrix_relative_error),
    "raw_grid_gauge_relative_error": float(raw_grid_gauge_error),
    "raw_grid_batch_consistency_relative_error": float(
        raw_grid_batch_consistency_error
    ),
    "raw_grid_target_drift_relative_error": float(
        raw_grid_target_drift_error
    ),
    "mace_grid_hamiltonian_relative_error": float(
        mace_grid_hamiltonian_error
    ),
    "neural_grid_hamiltonian_relative_error": float(
        neural_grid_hamiltonian_error
    ),
    "ftt_grid_distillation_relative_error": float(
        ftt_grid_distillation_error
    ),
    "ftt_build_mutation_relative_error": float(ftt_build_mutation_error),
    "adiabatic_projector_relative_errors": [
        float(value) for value in projector_errors
    ],
    "rms_link_singular_value_error": float(
        np.sqrt(np.mean(link_singular_value_error**2))
    ),
    "max_link_singular_value_error": float(np.max(link_singular_value_error)),
    "nearest_link_relative_error": float(nearest_link_relative_error),
    "hamiltonian_relative_error": float(hamiltonian_relative_error),
    "max_spectral_error_hartree": float(spectral_error),
    "final_breathing_marginal_l1_error": float(
        np.sum(np.abs(tn_marginal - direct_marginal))
    ),
    "max_direct_norm_error": float(np.max(np.abs(direct.norm - 1.0))),
    "max_tnldr_norm_error": float(np.max(np.abs(tnldr.norms - 1.0))),
    "build_seconds": build_seconds,
    "direct_seconds": direct_seconds,
    "tnldr_seconds": tn_seconds,
    "initial_state_compression_error": float(state_compression_error),
    "projector_1_validation_error": float(
        projector_1_info["validation_error"]
    ),
    "projector_validation_error": float(projector_info["validation_error"]),
    "database": str(direct.database_path),
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
    }
)
figure, panels = plt.subplots(1, 4, figsize=(12.2, 3.0), constrained_layout=True)
colors = ("#0072B2", "#D55E00")
for state, color in enumerate(colors):
    physical = state + 1
    panels[0].plot(
        time_fs,
        direct_populations[:, state],
        color=color,
        label=fr"Direct $S_{physical}$",
    )
    panels[0].plot(
        time_fs,
        tnldr.populations[:, state],
        "--",
        color=color,
        label=fr"TNLDR $S_{physical}$",
    )
panels[0].set(
    xlabel="Time (fs)",
    ylabel="Adiabatic population",
    ylim=(-0.025, 1.025),
)
panels[0].legend(ncol=2, frameon=False)
panels[1].semilogy(
    time_fs,
    np.maximum(np.max(population_error, axis=1), 1.0e-16),
    color="#7A3E9D",
)
panels[1].set(xlabel="Time (fs)", ylabel="Maximum population error")
panels[2].plot(grid.x[0], direct_marginal, "o-", color="#009E73", label="Direct")
panels[2].plot(grid.x[0], tn_marginal, "s--", color="#CC79A7", label="TNLDR")
panels[2].set(xlabel=r"Breathing coordinate $Q_b$ (bohr)", ylabel="Final marginal")
panels[2].legend(frameon=False)
adaptive_rounds = [record["round"] for record in adaptive_history]
adaptive_errors = [
    record["maximum_hamiltonian_error"]
    for record in adaptive_history
]
panels[3].semilogy(adaptive_rounds, adaptive_errors, "o-", color="#E69F00")
panels[3].axhline(
    2.0e-4,
    color="0.35",
    linestyle=":",
    label="absolute tolerance",
)
panels[3].set(
    xlabel="Adaptive round",
    ylabel=r"Held-out $\max\|\Delta \bar H\|_F$ (Ha)",
    xticks=adaptive_rounds,
)
panels[3].legend(frameon=False)
for label, panel in zip(("a", "b", "c", "d"), panels):
    panel.grid(alpha=0.2, linewidth=0.6)
    panel.text(-0.16, 1.04, label, transform=panel.transAxes, fontweight="bold")

figure_path = output / stem
figure.savefig(figure_path.with_suffix(".pdf"))
figure.savefig(figure_path.with_suffix(".png"), dpi=300)
np.savez(
    output / f"{stem}.npz",
    times=direct.times,
    direct_populations=direct_populations,
    tnldr_populations=tnldr.populations,
    direct_norms=direct.norm,
    tnldr_norms=tnldr.norms,
    breathing_axis=grid.x[0],
    direct_breathing_marginal=direct_marginal,
    tnldr_breathing_marginal=tn_marginal,
)

cut_axis = np.linspace(fit.grids[1][0], fit.grids[1][-1], 21)
cut_coordinates = np.column_stack(
    (np.zeros_like(cut_axis), cut_axis, np.zeros_like(cut_axis))
)
raw_cut = sampler.continuous_fields(cut_coordinates)["hamiltonians"]
dense_axis = np.linspace(cut_axis[0], cut_axis[-1], 401)
dense_coordinates = np.column_stack(
    (np.zeros_like(dense_axis), dense_axis, np.zeros_like(dense_axis))
)
raw_levels = np.linalg.eigvalsh(raw_cut)
mace_levels = np.linalg.eigvalsh(
    fit.neural_energy.predict(dense_coordinates)
)
ftt_levels = np.linalg.eigvalsh(fit.energy.predict(dense_coordinates))
mace_at_raw = np.linalg.eigvalsh(
    fit.neural_energy.predict(cut_coordinates)
)
ftt_at_raw = np.linalg.eigvalsh(fit.energy.predict(cut_coordinates))
level_origin = float(np.min(raw_levels))

cut_figure, cut_panels = plt.subplots(
    1, 2, figsize=(7.4, 3.1), constrained_layout=True
)
for state, color in enumerate(colors):
    physical = state + 1
    cut_panels[0].plot(
        dense_axis,
        (mace_levels[:, state] - level_origin) * au2ev,
        color=color,
        label=fr"MACE $S_{physical}$",
    )
    cut_panels[0].plot(
        dense_axis,
        (ftt_levels[:, state] - level_origin) * au2ev,
        color=color,
        linestyle="--",
        label=fr"FTT $S_{physical}$",
    )
    cut_panels[0].scatter(
        cut_axis,
        (raw_levels[:, state] - level_origin) * au2ev,
        facecolors="white",
        edgecolors=color,
        linewidths=0.9,
        s=22,
        zorder=3,
        label=fr"FCI $S_{physical}$",
    )
    cut_panels[1].plot(
        cut_axis,
        (mace_at_raw[:, state] - raw_levels[:, state]) * au2ev * 1.0e3,
        "o-",
        color=color,
        markersize=3.2,
        label=fr"MACE $S_{physical}$",
    )
    cut_panels[1].plot(
        cut_axis,
        (ftt_at_raw[:, state] - raw_levels[:, state]) * au2ev * 1.0e3,
        "s--",
        color=color,
        markersize=3.0,
        label=fr"FTT $S_{physical}$",
    )
cut_panels[0].set(
    xlabel=r"Branching coordinate $Q_x$ (bohr)",
    ylabel="Energy relative to cut minimum (eV)",
)
cut_panels[1].axhline(0.0, color="0.4", linewidth=0.8)
cut_panels[1].set(
    xlabel=r"Branching coordinate $Q_x$ (bohr)",
    ylabel="Fitted $-$ raw energy (meV)",
)
cut_panels[0].legend(
    ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(1.04, 1.23)
)
for label, panel in zip(("a", "b"), cut_panels):
    panel.grid(alpha=0.2, linewidth=0.6)
    panel.text(-0.16, 1.04, label, transform=panel.transAxes, fontweight="bold")
cut_path = output / f"{stem}_pes_cut_qs0_qy0"
cut_figure.savefig(cut_path.with_suffix(".pdf"))
cut_figure.savefig(cut_path.with_suffix(".png"), dpi=350)
np.savez(
    cut_path.with_suffix(".npz"),
    raw_axis=cut_axis,
    raw_hamiltonian=raw_cut,
    raw_levels=raw_levels,
    dense_axis=dense_axis,
    mace_levels=mace_levels,
    ftt_levels=ftt_levels,
    level_origin=level_origin,
)
summary["pes_cut"] = {
    "fixed_coordinates": {"Qs": 0.0, "Qy": 0.0},
    "raw_points": len(cut_axis),
    "maximum_mace_error_meV": float(
        np.max(np.abs(mace_at_raw - raw_levels)) * au2ev * 1.0e3
    ),
    "maximum_ftt_error_meV": float(
        np.max(np.abs(ftt_at_raw - raw_levels)) * au2ev * 1.0e3
    ),
}
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

print(json.dumps(summary, indent=2))
print(f"figure: {figure_path.with_suffix('.pdf')}")
print(f"figure: {figure_path.with_suffix('.png')}")
print(f"PES cut: {cut_path.with_suffix('.pdf')}")
print(f"PES cut: {cut_path.with_suffix('.png')}")
