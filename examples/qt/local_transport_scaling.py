"""Scaling benchmark for local and learned trajectory-transport potentials."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qt import (
    InvariantNeuralTransportPotential,
    ProjectedTwoParticleJastrow1D,
    SharedRadialTransportBasis,
    select_three_body_features,
    weak_poisson_objective,
)


def pair_jastrow_samples(
    particles, *, samples=512, dimension=2, seed=0, steps=80
):
    """Metropolis samples from a trapped pair-Jastrow density."""
    rng = np.random.default_rng(seed)
    coordinates = rng.normal(scale=0.65, size=(samples, particles, dimension))

    def log_density(points):
        confinement = -1.1 * np.sum(points**2, axis=(1, 2))
        correlation = np.zeros(points.shape[0])
        for particle_i in range(particles):
            for particle_j in range(particle_i + 1, particles):
                separation2 = np.sum(
                    (points[:, particle_i] - points[:, particle_j]) ** 2,
                    axis=1,
                )
                correlation -= 0.55 * np.exp(-separation2 / (2.0 * 0.8**2))
        return confinement + 2.0 * correlation

    current_log_density = log_density(coordinates)
    for _ in range(steps):
        proposal = coordinates + rng.normal(scale=0.18, size=coordinates.shape)
        proposal_log_density = log_density(proposal)
        accept = np.log(rng.random(samples)) < (
            proposal_log_density - current_log_density
        )
        coordinates[accept] = proposal[accept]
        current_log_density[accept] = proposal_log_density[accept]
    return coordinates


def pair_jastrow_scores(coordinates):
    """Centered log-density scores for trap and pair parameters."""
    samples, particles, _ = coordinates.shape
    trap = -np.sum(coordinates**2, axis=(1, 2))
    pair = np.zeros(samples)
    for particle_i in range(particles):
        for particle_j in range(particle_i + 1, particles):
            separation2 = np.sum(
                (coordinates[:, particle_i] - coordinates[:, particle_j]) ** 2,
                axis=1,
            )
            pair -= 2.0 * np.exp(-separation2 / (2.0 * 0.8**2))
    scores = np.column_stack((trap, pair))
    return scores - np.mean(scores, axis=0)


def equivariance_error(model, coordinates):
    """Measure neural rotation and permutation covariance."""
    angle = 0.61
    rotation = np.array(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    permutation = np.arange(coordinates.shape[1])[::-1]
    values, gradients = model.values_and_gradients(coordinates)
    permuted_values, permuted_gradients = model.values_and_gradients(
        coordinates[:, permutation]
    )
    rotated = np.einsum("ij,npj->npi", rotation, coordinates)
    rotated_values, rotated_gradients = model.values_and_gradients(rotated)
    expected_rotated = np.einsum("ij,npjf->npif", rotation, gradients)
    return max(
        np.max(np.abs(permuted_values - values)),
        np.max(np.abs(permuted_gradients - gradients[:, permutation])),
        np.max(np.abs(rotated_values - values)),
        np.max(np.abs(rotated_gradients - expected_rotated)),
    )


def main(output_dir="/private/tmp/pyqed_qt_local_transport"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    basis = SharedRadialTransportBasis()
    particle_counts = np.arange(3, 9)
    elapsed, objectives, feature_counts = [], [], []
    samples_by_size = {}
    for particles in particle_counts:
        coordinates = pair_jastrow_samples(
            int(particles), samples=512, seed=100 + int(particles)
        )
        scores = pair_jastrow_scores(coordinates)
        samples_by_size[int(particles)] = (coordinates, scores)
        start = perf_counter()
        values, gradients, _ = basis.values_and_gradients(coordinates)
        objective, _, _ = weak_poisson_objective(values, gradients, scores)
        elapsed.append(perf_counter() - start)
        objectives.append(objective)
        feature_counts.append(values.shape[1])

    three_body_coordinates = samples_by_size[4][0]
    base_scores = samples_by_size[4][1]
    three_body_probe = SharedRadialTransportBasis(
        one_body_centers=(),
        pair_centers=(),
        three_body_centers=(0.8,),
    )
    three_body_values, _, _ = three_body_probe.values_and_gradients(
        three_body_coordinates
    )
    three_body_score = three_body_values - np.mean(three_body_values, axis=0)
    augmented_scores = np.column_stack((base_scores, three_body_score))
    selected_basis, selection_history = select_three_body_features(
        three_body_coordinates,
        augmented_scores,
        basis,
        (0.4, 0.8, 1.2, 1.6),
        max_features=2,
        minimum_relative_improvement=1.0e-3,
    )

    neural_coordinates, neural_scores = samples_by_size[5]
    neural = InvariantNeuralTransportPotential(
        neural_scores.shape[1], hidden_width=12, seed=31
    ).fit(
        neural_coordinates[:256],
        neural_scores[:256],
        steps=140,
        learning_rate=2.0e-3,
    )
    symmetry_error = equivariance_error(neural, neural_coordinates[:24])

    trajectory_solver = ProjectedTwoParticleJastrow1D(
        ntraj=512, seed=37, ngrid=81, transport_basis="neural"
    )
    trajectory_cloud = trajectory_solver.sample_initial(np.log((1.2, 0.5, 0.5)))
    trajectory_theta, _ = trajectory_solver.reconstruct_parameters(
        trajectory_cloud
    )
    trajectory_solver.train_neural_transport(
        trajectory_cloud,
        trajectory_theta,
        steps=100,
        learning_rate=2.0e-3,
    )
    _, neural_tangents, neural_metric, neural_diagnostics = (
        trajectory_solver.constrained_continuity_lift(
            trajectory_cloud, trajectory_theta
        )
    )
    neural_state = trajectory_solver._sample_state(
        trajectory_theta,
        trajectory_cloud,
        tangent_data=(neural_tangents, neural_metric, neural_diagnostics),
    )
    neural_lift_error = np.max(
        np.abs(neural_diagnostics["lift_identity"] - np.eye(3))
    )
    neural_force_gap = np.max(
        np.abs(neural_state["force"] + neural_state["gradient"])
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)
    axes[0].plot(particle_counts, feature_counts, "o-", label="shared features")
    axes[0].plot(
        particle_counts,
        particle_counts * (particle_counts - 1) / 2,
        "s--",
        label="interacting pairs",
    )
    axes[0].set(
        xlabel="particles",
        ylabel="count",
        title="Fixed representation size",
    )
    axes[0].legend()

    axes[1].plot(particle_counts, 1.0e3 * np.asarray(elapsed), "o-")
    axes[1].set(
        xlabel="particles",
        ylabel="assembly time (ms)",
        title="512-configuration MC assembly",
    )

    loss = neural.loss_history
    axes[2].semilogy(
        np.arange(loss.size), loss - np.min(loss) + 1.0e-8, label="neural loss excess"
    )
    axes[2].set(
        xlabel="training step",
        ylabel="loss minus best loss",
        title="Learned invariant potential",
    )
    axes[2].legend()
    scaling_path = output_dir / "local_transport_scaling.png"
    fig.savefig(scaling_path, dpi=180)
    plt.close(fig)

    selected_centers = selected_basis.three_body_centers
    improvements = [item["relative_improvement"] for item in selection_history]
    print(f"particle counts:          {particle_counts}")
    print(f"shared feature counts:    {feature_counts}")
    print(f"assembly time (ms):       {1.0e3 * np.asarray(elapsed)}")
    print(f"weak objectives:          {np.asarray(objectives)}")
    print(f"selected 3-body centers:  {selected_centers}")
    print(f"relative improvements:    {improvements}")
    print(f"neural initial/final loss:{loss[0]:.6e} {loss[-1]:.6e}")
    print(f"neural symmetry error:    {symmetry_error:.3e}")
    print(f"neural KKT rank/condition:{neural_diagnostics['retained_basis_rank']} "
          f"{neural_diagnostics['constraint_condition']:.3e}")
    print(f"neural JU-I error:        {neural_lift_error:.3e}")
    print(f"neural force-gradient gap:{neural_force_gap:.3e}")
    print(f"scaling figure:           {scaling_path}")
    return {
        "particle_counts": particle_counts,
        "feature_counts": np.asarray(feature_counts),
        "elapsed": np.asarray(elapsed),
        "selected_basis": selected_basis,
        "neural": neural,
        "symmetry_error": symmetry_error,
        "figure": scaling_path,
    }


if __name__ == "__main__":
    main()
