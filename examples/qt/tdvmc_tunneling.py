"""Test continuity-corrected moving quadrature for real-time TDVMC."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qt import ComplexJastrowTDVMC1D, split_operator_step


def rk4_tdvmc_bohmian_step(model, theta, trajectories, dt):
    """Advance grid-quadrature TDVMC and its passive Bohmian paths together."""

    def rhs(current_theta, current_x):
        theta_dot, _ = model.tdvp_velocity(current_theta)
        return theta_dot, model.bohmian_velocity(current_x, current_theta)

    kt1, kx1 = rhs(theta, trajectories)
    kt2, kx2 = rhs(theta + 0.5 * dt * kt1, trajectories + 0.5 * dt * kx1)
    kt3, kx3 = rhs(theta + 0.5 * dt * kt2, trajectories + 0.5 * dt * kx2)
    kt4, kx4 = rhs(theta + dt * kt3, trajectories + dt * kx3)
    return (
        theta + (dt / 6.0) * (kt1 + 2.0 * kt2 + 2.0 * kt3 + kt4),
        trajectories + (dt / 6.0) * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4),
    )


def rk4_fixed_quantile_step(model, theta, quantiles, dt):
    """Advance TDVMC using only its fixed-weight continuity trajectories."""

    def rhs(current_theta):
        points = model.quantile_trajectories(current_theta, quantiles)
        return model.tdvp_velocity(current_theta, points)[0]

    k1 = rhs(theta)
    k2 = rhs(theta + 0.5 * dt * k1)
    k3 = rhs(theta + 0.5 * dt * k2)
    k4 = rhs(theta + dt * k3)
    return theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def spectral_energy(psi, potential, dx, mass=1.0, hbar=1.0):
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(psi.size, d=dx)
    derivative = np.fft.ifft(1j * wave_numbers * np.fft.fft(psi))
    kinetic = hbar**2 * np.trapezoid(np.abs(derivative) ** 2, dx=dx) / (2.0 * mass)
    potential_energy = np.trapezoid(potential * np.abs(psi) ** 2, dx=dx)
    return float(np.real(kinetic + potential_energy))


def exact_current(psi, dx, mass=1.0, hbar=1.0):
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(psi.size, d=dx)
    derivative = np.fft.ifft(1j * wave_numbers * np.fft.fft(psi))
    return hbar * np.imag(np.conj(psi) * derivative) / mass


def main(
    output_dir="/private/tmp/pyqed_qt_tdvmc_tunneling",
    *,
    time_step=0.004,
    final_time=6.0,
    ntraj=1024,
    record_every=10,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ComplexJastrowTDVMC1D(
        ngrid=1601,
        centers=(-2.4, -1.2, 0.0, 1.2, 2.4),
        metric_shift=1.0e-3,
    )
    grid = model.grid
    theta_grid = model.initial_parameters()
    theta_corrected = theta_grid.copy()
    quantiles = (np.arange(ntraj) + 0.5) / ntraj
    corrected_x = model.quantile_trajectories(theta_corrected, quantiles)
    raw_x = corrected_x.copy()
    exact_psi = model.wavefunction(theta_grid)
    random = np.random.default_rng(41)

    records = {
        key: []
        for key in (
            "time",
            "exact_energy",
            "tdvmc_energy",
            "corrected_energy",
            "corrected_sample_energy",
            "raw_sample_energy",
            "fresh_mc_energy",
            "density_error",
            "corrected_density_error",
            "exact_left",
            "tdvmc_left",
            "corrected_left",
            "corrected_empirical_left",
            "raw_empirical_left",
            "corrected_wasserstein",
            "raw_wasserstein",
            "corrected_parameter_error",
            "raw_parameter_error",
            "fresh_mc_parameter_error",
            "continuity_residual",
            "velocity_correction",
            "exact_barrier_current",
            "tdvmc_barrier_current",
            "corrected_transport_current",
        )
    }
    snapshots = []

    steps = int(round(final_time / time_step))
    for step in range(steps + 1):
        if step % record_every == 0 or step == steps:
            time = step * time_step
            exact_rho = np.abs(exact_psi) ** 2
            rho = model.density(theta_grid)
            corrected_rho = model.density(theta_corrected)
            audit_x = model.quantile_trajectories(theta_grid, quantiles)
            corrected_x = model.quantile_trajectories(theta_corrected, quantiles)
            target_raw_x = audit_x
            fresh_x = model.quantile_trajectories(
                theta_grid, random.random(ntraj)
            )
            dot_grid, grid_diagnostics = model.tdvp_velocity(theta_grid)
            dot_audit, corrected_diagnostics = model.tdvp_velocity(
                theta_grid, audit_x
            )
            dot_corrected, _ = model.tdvp_velocity(
                theta_corrected, corrected_x
            )
            _, corrected_grid_diagnostics = model.tdvp_velocity(theta_corrected)
            dot_raw, raw_diagnostics = model.tdvp_velocity(theta_grid, raw_x)
            dot_fresh, fresh_diagnostics = model.tdvp_velocity(theta_grid, fresh_x)
            continuity = model.continuity_state(theta_corrected, dot_corrected)
            current_exact = exact_current(exact_psi, model.dx, model.mass, model.hbar)
            bohmian_velocity = model.bohmian_velocity(grid, theta_grid)
            barrier_index = int(np.argmin(np.abs(grid)))
            left = grid <= 0.0
            velocity_scale = max(float(np.linalg.norm(dot_grid)), 1.0e-12)

            records["time"].append(time)
            records["exact_energy"].append(
                spectral_energy(exact_psi, model.potential_grid, model.dx)
            )
            records["tdvmc_energy"].append(grid_diagnostics["energy"])
            records["corrected_energy"].append(
                corrected_grid_diagnostics["energy"]
            )
            records["corrected_sample_energy"].append(corrected_diagnostics["energy"])
            records["raw_sample_energy"].append(raw_diagnostics["energy"])
            records["fresh_mc_energy"].append(fresh_diagnostics["energy"])
            records["density_error"].append(np.trapezoid(np.abs(rho - exact_rho), grid))
            records["corrected_density_error"].append(
                np.trapezoid(np.abs(corrected_rho - exact_rho), grid)
            )
            records["exact_left"].append(np.trapezoid(exact_rho[left], grid[left]))
            records["tdvmc_left"].append(np.trapezoid(rho[left], grid[left]))
            records["corrected_left"].append(
                np.trapezoid(corrected_rho[left], grid[left])
            )
            records["corrected_empirical_left"].append(np.mean(corrected_x <= 0.0))
            records["raw_empirical_left"].append(np.mean(raw_x <= 0.0))
            records["corrected_wasserstein"].append(
                np.sqrt(
                    np.mean(
                        (
                            corrected_x
                            - model.quantile_trajectories(
                                theta_corrected, quantiles
                            )
                        )
                        ** 2
                    )
                )
            )
            records["raw_wasserstein"].append(
                np.sqrt(np.mean((np.sort(raw_x) - target_raw_x) ** 2))
            )
            records["corrected_parameter_error"].append(
                np.linalg.norm(dot_audit - dot_grid) / velocity_scale
            )
            records["raw_parameter_error"].append(
                np.linalg.norm(dot_raw - dot_grid) / velocity_scale
            )
            records["fresh_mc_parameter_error"].append(
                np.linalg.norm(dot_fresh - dot_grid) / velocity_scale
            )
            records["continuity_residual"].append(
                continuity["continuity_residual_rms"]
            )
            records["velocity_correction"].append(
                continuity["velocity_correction_rms"]
            )
            records["exact_barrier_current"].append(current_exact[barrier_index])
            records["tdvmc_barrier_current"].append(
                rho[barrier_index] * bohmian_velocity[barrier_index]
            )
            records["corrected_transport_current"].append(
                corrected_rho[barrier_index]
                * continuity["tangent_velocity"][barrier_index]
            )
            if len(snapshots) < 4 and step >= round(len(snapshots) * steps / 3):
                snapshots.append(
                    (
                        time,
                        exact_rho.copy(),
                        rho.copy(),
                        corrected_rho.copy(),
                        corrected_x.copy(),
                        raw_x.copy(),
                    )
                )

        if step == steps:
            break

        theta_grid, raw_x = rk4_tdvmc_bohmian_step(
            model, theta_grid, raw_x, time_step
        )
        theta_corrected = rk4_fixed_quantile_step(
            model, theta_corrected, quantiles, time_step
        )
        exact_psi = split_operator_step(
            exact_psi,
            model.potential_grid,
            model.dx,
            time_step,
            mass=model.mass,
            hbar=model.hbar,
        )
        exact_psi /= np.sqrt(np.trapezoid(np.abs(exact_psi) ** 2, grid))

    history = {key: np.asarray(value) for key, value in records.items()}

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    final = snapshots[-1]
    axes[0, 0].plot(grid, final[1], color="black", linewidth=2.0, label="exact")
    axes[0, 0].plot(grid, final[2], "--", linewidth=2.0, label="grid Jastrow TDVMC")
    axes[0, 0].plot(grid, final[3], ":", linewidth=2.0, label="trajectory TDVMC")
    axes[0, 0].scatter(
        final[4], np.zeros_like(final[4]), marker="|", s=45, alpha=0.45,
        label="corrected paths",
    )
    axes[0, 0].set(xlabel="$x$", ylabel=r"$|\psi|^2$", title=f"Density at $t={final_time:g}$")
    axes[0, 0].legend()

    axes[0, 1].plot(history["time"], history["exact_left"], color="black", label="exact")
    axes[0, 1].plot(history["time"], history["tdvmc_left"], "--", label="Jastrow TDVMC")
    axes[0, 1].plot(history["time"], history["corrected_left"], ":", label="trajectory TDVMC density")
    axes[0, 1].plot(history["time"], history["corrected_empirical_left"], label="corrected paths")
    axes[0, 1].plot(history["time"], history["raw_empirical_left"], label="raw Bohmian paths")
    axes[0, 1].set(xlabel="time", ylabel=r"$P(x<0)$", title="Barrier transfer")
    axes[0, 1].legend()

    axes[1, 0].semilogy(
        history["time"], np.maximum(history["corrected_wasserstein"], 1e-12),
        label="corrected",
    )
    axes[1, 0].semilogy(
        history["time"], np.maximum(history["raw_wasserstein"], 1e-12),
        label="raw Bohmian",
    )
    axes[1, 0].set(xlabel="time", ylabel=r"$W_2$ to $|\psi_\theta|^2$", title="Quantum-equilibrium drift")
    axes[1, 0].legend()

    axes[1, 1].plot(history["time"], history["density_error"], label="grid TDVMC")
    axes[1, 1].plot(history["time"], history["corrected_density_error"], label="trajectory TDVMC")
    axes[1, 1].set(
        xlabel="time", ylabel=r"$\int|\rho_\theta-\rho_{\rm exact}|dx$",
        title="Finite-Jastrow dynamical error",
    )
    axes[1, 1].legend()
    density_path = output_dir / "tdvmc_tunneling_density.png"
    fig.savefig(density_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    axes[0, 0].plot(history["time"], history["exact_energy"], color="black", label="exact")
    axes[0, 0].plot(history["time"], history["tdvmc_energy"], "--", label="grid TDVMC")
    axes[0, 0].plot(history["time"], history["corrected_energy"], ":", label="trajectory TDVMC")
    axes[0, 0].plot(history["time"], history["corrected_sample_energy"], label="corrected quadrature")
    axes[0, 0].plot(history["time"], history["raw_sample_energy"], label="raw quadrature")
    axes[0, 0].set(xlabel="time", ylabel="energy", title="Energy estimates")
    axes[0, 0].legend()

    axes[0, 1].semilogy(history["time"], history["corrected_parameter_error"], label="corrected fixed paths")
    axes[0, 1].semilogy(history["time"], history["raw_parameter_error"], label="raw Bohmian paths")
    axes[0, 1].semilogy(history["time"], history["fresh_mc_parameter_error"], alpha=0.7, label="fresh ideal MC")
    axes[0, 1].set(xlabel="time", ylabel=r"relative error in $\dot\theta$", title="TDVMC sampling quality")
    axes[0, 1].legend()

    axes[1, 0].semilogy(history["time"], history["continuity_residual"], label="continuity residual")
    axes[1, 0].semilogy(history["time"], history["velocity_correction"], label=r"$\|\delta v\|_\rho$")
    axes[1, 0].set(xlabel="time", ylabel="RMS", title="Required non-Bohmian correction")
    axes[1, 0].legend()

    axes[1, 1].plot(history["time"], history["exact_barrier_current"], color="black", label="exact physical")
    axes[1, 1].plot(history["time"], history["tdvmc_barrier_current"], "--", label="TDVMC physical")
    axes[1, 1].plot(history["time"], history["corrected_transport_current"], label="corrected transport")
    axes[1, 1].set(xlabel="time", ylabel=r"$j(0,t)$", title="Barrier current")
    axes[1, 1].legend()
    diagnostics_path = output_dir / "tdvmc_tunneling_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=180)
    plt.close(fig)

    np.savez(output_dir / "tdvmc_tunneling_history.npz", **history)
    print(f"initial/final exact energy:       {history['exact_energy'][0]:.10f} {history['exact_energy'][-1]:.10f}")
    print(f"initial/final TDVMC energy:       {history['tdvmc_energy'][0]:.10f} {history['tdvmc_energy'][-1]:.10f}")
    print(f"initial/final trajectory energy:  {history['corrected_energy'][0]:.10f} {history['corrected_energy'][-1]:.10f}")
    print(f"final grid/trajectory density L1: {history['density_error'][-1]:.6f} {history['corrected_density_error'][-1]:.6f}")
    print(f"final exact/TDVMC left population:{history['exact_left'][-1]:.6f} {history['tdvmc_left'][-1]:.6f}")
    print(f"final trajectory-density P_L:     {history['corrected_left'][-1]:.6f}")
    print(f"final corrected/raw empirical P_L:{history['corrected_empirical_left'][-1]:.6f} {history['raw_empirical_left'][-1]:.6f}")
    print(f"final corrected/raw trajectory W2:{history['corrected_wasserstein'][-1]:.6e} {history['raw_wasserstein'][-1]:.6e}")
    print(f"mean corrected/raw/fresh theta error: {np.mean(history['corrected_parameter_error']):.4e} {np.mean(history['raw_parameter_error']):.4e} {np.mean(history['fresh_mc_parameter_error']):.4e}")
    print(f"max continuity residual:          {np.max(history['continuity_residual']):.6e}")
    print(f"max velocity correction RMS:      {np.max(history['velocity_correction']):.6e}")
    print(f"density figure:                   {density_path}")
    print(f"diagnostics figure:               {diagnostics_path}")
    model.theta = theta_corrected
    model.trajectories = corrected_x
    model.history = history
    model.success = np.all(np.isfinite(history["tdvmc_energy"])) and np.all(
        np.isfinite(history["corrected_energy"])
    )
    model.message = "propagation completed" if model.success else "non-finite propagation"
    return model


if __name__ == "__main__":
    main()
