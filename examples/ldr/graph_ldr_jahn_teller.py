"""Validate GraphLDR on a two-state conical-intersection model.

The example compares Cartesian and polar finite-volume graphs for the full
two-state problem, then verifies the Berry phase and quantum-metric potential
on a single-surface annulus.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.sparse.linalg import eigsh

from pyqed.ldr import FEMLDR, GraphLDR, GraphMesh, LDR, TriangularMesh


def diabatic_potential(points, *, omega=1.0, kappa=1.0, coupling=1.0):
    """Return the dimensionless two-state E x e Jahn--Teller potential."""

    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    harmonic = 0.5 * omega**2 * (x**2 + y**2)
    potential = np.zeros((len(points), 2, 2), dtype=float)
    potential[:, 0, 0] = harmonic + kappa * x
    potential[:, 1, 1] = harmonic - kappa * x
    potential[:, 0, 1] = potential[:, 1, 0] = coupling * y
    return potential


def lowest_energies(solver, count=6):
    """Return the lowest eigenvalues of a static GraphLDR Hamiltonian."""

    values = eigsh(
        solver.hamiltonian(sparse=True),
        k=int(count),
        which="SA",
        return_eigenvectors=False,
    )
    return np.sort(values.real)


def full_two_state_comparison(
    *,
    extent=5.0,
    ncart=21,
    nr=13,
    ntheta=24,
    count=6,
):
    """Compare low vibronic energies on Cartesian and polar graphs."""

    axis = np.linspace(-extent, extent, int(ncart))
    cartesian_mesh = GraphMesh.rectilinear(axis, axis)
    polar_mesh = GraphMesh.polar(np.linspace(0.0, extent, int(nr)), int(ntheta))
    cartesian = GraphLDR(cartesian_mesh, 2).set_diabatic(
        diabatic_potential(cartesian_mesh.nodes)
    )
    polar = GraphLDR(polar_mesh, 2).set_diabatic(
        diabatic_potential(polar_mesh.nodes)
    )
    cartesian_energies = lowest_energies(cartesian, count)
    polar_energies = lowest_energies(polar, count)
    return {
        "cartesian_mesh": cartesian_mesh,
        "polar_mesh": polar_mesh,
        "cartesian_energies": cartesian_energies,
        "polar_energies": polar_energies,
        "max_energy_difference": float(
            np.max(np.abs(cartesian_energies - polar_energies))
        ),
    }


def lower_surface_annulus(*, ntheta=48, nr=14, rmin=0.7, rmax=4.5, count=4):
    """Compare raw overlaps with Berry links plus the analytic QGT potential."""

    mesh = GraphMesh.polar(np.linspace(rmin, rmax, int(nr)), int(ntheta))
    energies, frames = np.linalg.eigh(diabatic_potential(mesh.nodes))
    overlaps = np.asarray(
        [np.vdot(frames[left, :, 0], frames[right, :, 0]) for left, right in mesh.edges]
    )[:, None, None]
    raw = GraphLDR(
        mesh,
        1,
        energies=energies[:, [0]],
        overlaps=overlaps,
    )

    unitary_links = overlaps / np.abs(overlaps)
    radius_squared = np.sum(mesh.nodes**2, axis=1)
    quantum_metric_potential = 1.0 / (8.0 * radius_squared)
    explicit_qgt = GraphLDR(
        mesh,
        1,
        energies=(energies[:, 0] + quantum_metric_potential)[:, None],
        overlaps=unitary_links,
    )

    wilson_loop = 1.0 + 0.0j
    for angular in range(int(ntheta)):
        neighbor = (angular + 1) % int(ntheta)
        link = raw.overlap_block(angular, neighbor)[0, 0]
        wilson_loop *= link / abs(link)
    raw_energies = lowest_energies(raw, count)
    explicit_energies = lowest_energies(explicit_qgt, count)
    return {
        "mesh": mesh,
        "raw": raw,
        "explicit_qgt": explicit_qgt,
        "wilson_loop": wilson_loop,
        "raw_energies": raw_energies,
        "explicit_qgt_energies": explicit_energies,
        "max_energy_difference": float(
            np.max(np.abs(raw_energies - explicit_energies))
        ),
    }


def qgt_convergence(ntheta_values=(12, 16, 24, 32, 48, 64)):
    """Return raw-overlap versus explicit-QGT spectral convergence."""

    values = np.asarray(tuple(int(value) for value in ntheta_values), dtype=int)
    differences = np.asarray(
        [lower_surface_annulus(ntheta=value)["max_energy_difference"] for value in values]
    )
    return values, differences


def wavepacket_dynamics(
    *,
    extent=5.0,
    nr=21,
    ntheta=48,
    center=(-2.2, 0.0),
    width=0.45,
    momentum=(2.5, 0.0),
    dt=0.02,
    nsteps=250,
    nout=10,
    mesh_method="finite-volume",
):
    """Launch a diabatic Gaussian packet through the conical intersection."""

    radii = np.linspace(0.0, extent, int(nr))
    if mesh_method == "finite-volume":
        mesh = GraphMesh.polar(radii, int(ntheta))
    elif mesh_method == "fem":
        mesh = GraphMesh.polar_fem(radii, int(ntheta))
    else:
        raise ValueError("mesh_method must be 'finite-volume' or 'fem'")
    solver = GraphLDR(mesh, 2).set_diabatic(diabatic_potential(mesh.nodes))
    center = np.asarray(center, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    displacement = mesh.nodes - center
    envelope = np.sqrt(mesh.volumes) * np.exp(
        -np.sum(displacement**2, axis=1) / (4.0 * width**2)
        + 1j * (mesh.nodes @ momentum)
    )

    diabatic_vector = np.array([1.0, 0.0], dtype=complex)
    local_electronic = np.einsum(
        "mab,a->mb",
        solver.frames.conj(),
        diabatic_vector,
        optimize=True,
    )
    initial = envelope[:, None] * local_electronic
    initial /= np.linalg.norm(initial)
    solver.run(
        initial,
        dt=dt,
        nsteps=nsteps,
        nout=nout,
        matrix_free=False,
    )

    adiabatic_populations = np.sum(np.abs(solver.states) ** 2, axis=1)
    diabatic_states = np.einsum(
        "mab,tmb->tma",
        solver.frames,
        solver.states,
        optimize=True,
    )
    diabatic_populations = np.sum(np.abs(diabatic_states) ** 2, axis=1)
    node_probabilities = np.sum(np.abs(solver.states) ** 2, axis=2)
    physical_density = node_probabilities / mesh.volumes
    coordinate_means = node_probabilities @ mesh.nodes
    return {
        "mesh": mesh,
        "solver": solver,
        "adiabatic_populations": adiabatic_populations,
        "diabatic_populations": diabatic_populations,
        "physical_density": physical_density,
        "coordinate_means": coordinate_means,
        "max_norm_error": float(np.max(np.abs(solver.norm - 1.0))),
    }


def full_ldr_dynamics(
    *,
    extent=5.0,
    ncart=31,
    center=(-2.2, 0.0),
    width=0.45,
    momentum=(2.5, 0.0),
    dt=0.02,
    nsteps=250,
    nout=10,
):
    """Run the same packet with the product-DVR full-overlap LDR solver."""

    solver = LDR.from_domains(
        ((-extent, extent), (-extent, extent)),
        (int(ncart), int(ncart)),
        2,
        names=("x", "y"),
    )
    potential = diabatic_potential(solver.points).reshape(
        *solver.shape,
        2,
        2,
    )
    solver.set_diabatic(potential, representation="full")

    center = np.asarray(center, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    displacement = solver.points - center
    envelope = np.exp(
        -np.sum(displacement**2, axis=1) / (4.0 * width**2)
        + 1j * (solver.points @ momentum)
    ).reshape(solver.shape)
    diabatic_vector = np.array([1.0, 0.0], dtype=complex)
    local_electronic = np.einsum(
        "...ab,a->...b",
        solver.frames.conj(),
        diabatic_vector,
        optimize=True,
    )
    initial = envelope[..., None] * local_electronic
    initial /= np.linalg.norm(initial)
    solver.run(
        initial,
        dt=dt,
        nsteps=nsteps,
        nout=nout,
        matrix_free=True,
    )

    states = solver.states.reshape(len(solver.times), solver.ngrid, 2)
    frames = solver.frames.reshape(solver.ngrid, 2, 2)
    adiabatic_populations = np.sum(np.abs(states) ** 2, axis=1)
    diabatic_states = np.einsum(
        "mab,tmb->tma",
        frames,
        states,
        optimize=True,
    )
    diabatic_populations = np.sum(np.abs(diabatic_states) ** 2, axis=1)
    node_probabilities = np.sum(np.abs(states) ** 2, axis=2)
    coordinate_means = node_probabilities @ solver.points
    return {
        "solver": solver,
        "adiabatic_populations": adiabatic_populations,
        "diabatic_populations": diabatic_populations,
        "coordinate_means": coordinate_means,
        "max_norm_error": float(np.max(np.abs(solver.norm - 1.0))),
    }


def fourth_order_graph_dynamics(
    *,
    extent=5.0,
    ncart=31,
    center=(-2.2, 0.0),
    width=0.45,
    momentum=(2.5, 0.0),
    dt=0.02,
    nsteps=250,
    nout=10,
):
    """Run the packet with a fourth-order two-hop Cartesian graph KEO."""

    spacing = 2.0 * extent / (int(ncart) + 1)
    axis = -extent + spacing * np.arange(1, int(ncart) + 1)
    mesh = GraphMesh.rectilinear_fourth_order(axis, axis)
    solver = GraphLDR(mesh, 2).set_diabatic(diabatic_potential(mesh.nodes))
    center = np.asarray(center, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    displacement = mesh.nodes - center
    envelope = np.sqrt(mesh.volumes) * np.exp(
        -np.sum(displacement**2, axis=1) / (4.0 * width**2)
        + 1j * (mesh.nodes @ momentum)
    )
    local_electronic = np.einsum(
        "mab,a->mb",
        solver.frames.conj(),
        np.array([1.0, 0.0], dtype=complex),
        optimize=True,
    )
    initial = envelope[:, None] * local_electronic
    initial /= np.linalg.norm(initial)
    solver.run(
        initial,
        dt=dt,
        nsteps=nsteps,
        nout=nout,
        matrix_free=False,
    )

    adiabatic_populations = np.sum(np.abs(solver.states) ** 2, axis=1)
    diabatic_states = np.einsum(
        "mab,tmb->tma",
        solver.frames,
        solver.states,
        optimize=True,
    )
    diabatic_populations = np.sum(np.abs(diabatic_states) ** 2, axis=1)
    node_probabilities = np.sum(np.abs(solver.states) ** 2, axis=2)
    return {
        "mesh": mesh,
        "solver": solver,
        "adiabatic_populations": adiabatic_populations,
        "diabatic_populations": diabatic_populations,
        "coordinate_means": node_probabilities @ mesh.nodes,
        "max_norm_error": float(np.max(np.abs(solver.norm - 1.0))),
    }


def fem_ldr_dynamics(
    *,
    extent=5.0,
    nr=11,
    ntheta=24,
    order=2,
    center=(-2.2, 0.0),
    width=0.45,
    momentum=(2.5, 0.0),
    dt=0.02,
    nsteps=250,
    nout=10,
    geometry="polar",
    ncells=16,
):
    """Run connection-dressed finite-element LDR on triangular elements."""

    if geometry == "polar":
        mesh = TriangularMesh.polar(
            np.linspace(0.0, extent, int(nr)),
            int(ntheta),
            order=order,
        )
    elif geometry == "cartesian":
        mesh = TriangularMesh.cartesian(
            ((-extent, extent), (-extent, extent)),
            int(ncells),
            order=order,
        )
    else:
        raise ValueError("geometry must be 'polar' or 'cartesian'")
    return _run_fem_mesh_dynamics(
        mesh,
        center=center,
        width=width,
        momentum=momentum,
        dt=dt,
        nsteps=nsteps,
        nout=nout,
    )


def _run_fem_mesh_dynamics(
    mesh,
    *,
    center=(-2.2, 0.0),
    width=0.45,
    momentum=(2.5, 0.0),
    dt=0.02,
    nsteps=250,
    nout=10,
):
    """Run the finite-element wavepacket benchmark on a supplied mesh."""

    solver = FEMLDR(mesh, 2).set_diabatic(diabatic_potential(mesh.nodes))
    center = np.asarray(center, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    displacement = mesh.nodes - center
    envelope = np.exp(
        -np.sum(displacement**2, axis=1) / (4.0 * width**2)
        + 1j * (mesh.nodes @ momentum)
    )
    local_electronic = np.einsum(
        "mab,a->mb",
        solver.frames.conj(),
        np.array([1.0, 0.0], dtype=complex),
        optimize=True,
    )
    initial = solver.normalize(envelope[:, None] * local_electronic)
    solver.run(initial, dt=dt, nsteps=nsteps, nout=nout)
    return {
        "mesh": mesh,
        "solver": solver,
        "adiabatic_populations": solver.adiabatic_populations(),
        "diabatic_populations": solver.diabatic_populations(),
        "coordinate_means": solver.coordinate_means(),
        "max_norm_error": float(np.max(np.abs(solver.norm - 1.0))),
    }


def adaptive_fem_ldr_dynamics(
    *,
    extent=5.0,
    nr=6,
    ntheta=12,
    cycles=1,
    theta=0.60,
    max_fraction=0.35,
    electronic_weight=0.25,
    target_nodes=None,
    center=(-2.2, 0.0),
    width=0.45,
    momentum=(2.5, 0.0),
    dt=0.02,
    nsteps=250,
    nout=10,
):
    """Adapt P2 elements from residual pilot dynamics, then rerun."""

    mesh = TriangularMesh.polar(
        np.linspace(0.0, float(extent), int(nr)),
        int(ntheta),
        order=2,
    )
    cycles = int(cycles)
    if cycles < 1:
        raise ValueError("cycles must be positive")
    initial_nodes = mesh.size
    history = []
    for cycle in range(cycles):
        pilot = _run_fem_mesh_dynamics(
            mesh,
            center=center,
            width=width,
            momentum=momentum,
            dt=dt,
            nsteps=nsteps,
            nout=1,
        )
        solver = pilot["solver"]
        nuclear = solver.residual_indicators()
        electronic = solver.projector_indicators()
        nuclear_scaled = nuclear / max(float(np.max(nuclear)), np.finfo(float).tiny)
        electronic_scaled = electronic / max(
            float(np.max(electronic)),
            np.finfo(float).tiny,
        )
        combined = (
            (1.0 - float(electronic_weight)) * nuclear_scaled
            + float(electronic_weight) * electronic_scaled
        )
        if target_nodes is None:
            marked = mesh.dorfler_mark(
                combined,
                theta=theta,
                max_fraction=max_fraction,
            )
            refined = mesh.refine(marked)
        else:
            stage_target = round(
                initial_nodes
                + (cycle + 1) * (int(target_nodes) - initial_nodes) / cycles
            )
            refined, marked = mesh.refine_to_size(combined, stage_target)
        history.append(
            {
                "mesh": mesh,
                "residual_indicator": nuclear,
                "electronic_indicator": electronic,
                "combined_indicator": combined,
                "marked": marked,
                "stage_target_nodes": (
                    None if target_nodes is None else stage_target
                ),
            }
        )
        if refined is mesh:
            break
        mesh = refined

    result = _run_fem_mesh_dynamics(
        mesh,
        center=center,
        width=width,
        momentum=momentum,
        dt=dt,
        nsteps=nsteps,
        nout=nout,
    )
    result["adaptation_history"] = history
    return result


def plot_dynamics(result, filename):
    """Plot electronic populations, packet motion, and density snapshots."""

    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    solver = result["solver"]
    times = solver.times
    means = result["coordinate_means"]
    adiabatic = result["adiabatic_populations"]
    diabatic = result["diabatic_populations"]
    first_passage = np.argmin(np.where(times <= 2.0, np.abs(means[:, 0]), np.inf))
    max_transfer = np.argmax(adiabatic[:, 1])
    snapshot_indices = (0, first_passage, max_transfer)

    figure, axes = plt.subplot_mosaic(
        [["populations", "populations", "motion"], ["initial", "crossing", "outgoing"]],
        figsize=(12.0, 7.0),
        constrained_layout=True,
    )
    population_axis = axes["populations"]
    population_axis.plot(times, adiabatic[:, 0], label="lower adiabatic")
    population_axis.plot(times, adiabatic[:, 1], label="upper adiabatic")
    population_axis.plot(times, diabatic[:, 0], "--", label="diabatic 1")
    population_axis.plot(times, diabatic[:, 1], "--", label="diabatic 2")
    population_axis.set(
        xlabel="time",
        ylabel="population",
        ylim=(-0.02, 1.02),
        title="Electronic population transfer",
    )
    population_axis.legend(frameon=False, ncol=2)
    population_axis.grid(alpha=0.2)

    motion_axis = axes["motion"]
    motion_axis.plot(times, means[:, 0], label=r"$\langle x\rangle$")
    motion_axis.axhline(0.0, color="0.5", linewidth=0.8)
    for index in snapshot_indices:
        motion_axis.plot(times[index], means[index, 0], "o")
    motion_axis.set(
        xlabel="time",
        ylabel=r"$\langle x\rangle$",
        title="Passage through the intersection",
    )
    motion_axis.grid(alpha=0.2)

    mesh = result["mesh"]
    triangulation = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1])
    density = result["physical_density"]
    maximum = np.max(density[list(snapshot_indices)])
    image = None
    for name, index in zip(("initial", "crossing", "outgoing"), snapshot_indices):
        axis = axes[name]
        image = axis.tripcolor(
            triangulation,
            density[index],
            shading="gouraud",
            vmin=0.0,
            vmax=maximum,
        )
        axis.plot(0.0, 0.0, "+", color="white", markersize=8, markeredgewidth=1.4)
        axis.set(
            xlabel="x",
            ylabel="y",
            title=f"t = {times[index]:.2f}",
            xlim=(-4.0, 4.0),
            ylim=(-3.2, 3.2),
        )
        axis.set_aspect("equal")
    figure.colorbar(image, ax=[axes[name] for name in ("initial", "crossing", "outgoing")], label="nuclear density")
    figure.suptitle(
        "GraphLDR wavepacket dynamics through a conical intersection\n"
        f"maximum norm error = {result['max_norm_error']:.1e}"
    )
    figure.savefig(filename, dpi=180)
    plt.close(figure)
    return filename


def plot_results(full, annulus, convergence, filename):
    """Plot graph layouts, spectra, Berry phase, and QGT convergence."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    cartesian = full["cartesian_mesh"].nodes
    polar = full["polar_mesh"].nodes
    axes[0, 0].scatter(
        cartesian[:, 0],
        cartesian[:, 1],
        s=7,
        alpha=0.35,
        label=f"Cartesian ({len(cartesian)})",
    )
    axes[0, 0].scatter(
        polar[:, 0],
        polar[:, 1],
        s=8,
        alpha=0.75,
        label=f"Polar ({len(polar)})",
    )
    axes[0, 0].set(xlabel="x", ylabel="y", title="Nuclear graph nodes")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].legend(frameon=False)

    level = np.arange(len(full["cartesian_energies"]))
    axes[0, 1].plot(
        level,
        full["cartesian_energies"],
        "o-",
        label="Cartesian",
    )
    axes[0, 1].plot(level, full["polar_energies"], "s--", label="Polar")
    axes[0, 1].set(
        xlabel="level index",
        ylabel="energy",
        title="Full two-state vibronic spectrum",
    )
    axes[0, 1].legend(frameon=False)

    ntheta, differences = convergence
    axes[1, 0].loglog(ntheta, differences, "o-")
    axes[1, 0].set(
        xlabel=r"angular nodes $N_\theta$",
        ylabel="maximum energy difference",
        title="Raw overlaps approach explicit QGT",
    )
    axes[1, 0].grid(alpha=0.25, which="both")

    annulus_level = np.arange(len(annulus["raw_energies"]))
    axes[1, 1].plot(
        annulus_level,
        annulus["raw_energies"],
        "o-",
        label="raw overlap",
    )
    axes[1, 1].plot(
        annulus_level,
        annulus["explicit_qgt_energies"],
        "s--",
        label=r"unitary link + $\hbar^2/(8r^2)$",
    )
    axes[1, 1].set(
        xlabel="level index",
        ylabel="energy",
        title=f"Single surface: Wilson loop = {annulus['wilson_loop'].real:.0f}",
    )
    axes[1, 1].legend(frameon=False)

    figure.suptitle(r"GraphLDR validation on the $E\otimes e$ Jahn--Teller model")
    figure.savefig(filename, dpi=180)
    plt.close(figure)
    return filename


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncart", type=int, default=21)
    parser.add_argument("--nr", type=int, default=13)
    parser.add_argument("--ntheta", type=int, default=24)
    parser.add_argument("--annulus-ntheta", type=int, default=48)
    parser.add_argument("--plot", metavar="FILE")
    parser.add_argument("--dynamics", action="store_true")
    parser.add_argument("--dynamics-plot", metavar="FILE")
    parser.add_argument("--full-ldr-reference", action="store_true")
    args = parser.parse_args(argv)

    full = full_two_state_comparison(
        ncart=args.ncart,
        nr=args.nr,
        ntheta=args.ntheta,
    )
    annulus = lower_surface_annulus(ntheta=args.annulus_ntheta)
    print("Cartesian low energies:", full["cartesian_energies"])
    print("Polar low energies:    ", full["polar_energies"])
    print("Maximum difference:    ", full["max_energy_difference"])
    print("Annulus Wilson loop:   ", annulus["wilson_loop"])
    print("Raw-overlap energies:  ", annulus["raw_energies"])
    print("Explicit-QGT energies: ", annulus["explicit_qgt_energies"])
    print("Maximum QGT difference:", annulus["max_energy_difference"])
    if args.plot:
        convergence = qgt_convergence()
        plot_results(full, annulus, convergence, args.plot)
        print("Plot saved to:         ", args.plot)
    if args.dynamics or args.dynamics_plot:
        dynamics = wavepacket_dynamics()
        adiabatic = dynamics["adiabatic_populations"]
        diabatic = dynamics["diabatic_populations"]
        print("Maximum norm error:    ", dynamics["max_norm_error"])
        print("Maximum upper-surface: ", np.max(adiabatic[:, 1]))
        print("Final adiabatic pops:  ", adiabatic[-1])
        print("Final diabatic pops:   ", diabatic[-1])
        if args.dynamics_plot:
            plot_dynamics(dynamics, args.dynamics_plot)
            print("Dynamics plot saved to:", args.dynamics_plot)
        if args.full_ldr_reference:
            reference = full_ldr_dynamics()
            difference = (
                dynamics["adiabatic_populations"]
                - reference["adiabatic_populations"]
            )
            print("Full-LDR grid points:  ", reference["solver"].ngrid)
            print("Graph/full-LDR RMSE:   ", np.sqrt(np.mean(difference**2)))
            print(
                "Full-LDR final pops:   ",
                reference["adiabatic_populations"][-1],
            )


if __name__ == "__main__":
    main()
