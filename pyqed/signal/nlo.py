"""Second-order nonlinear optical response functions.

The routines here evaluate the resonant sum-over-states expression for
sum-frequency generation (SFG) and its degenerate second-harmonic generation
(SHG) limit.  Energies and frequencies are expected to be in the same units.
"""

from __future__ import annotations

import numpy as np


def _validate_states(energies, dipole, ground, states):
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 1:
        raise ValueError("energies must be a one-dimensional array")

    dipole = np.asarray(dipole, dtype=complex)
    if dipole.ndim == 2:
        if dipole.shape != (energies.size, energies.size):
            raise ValueError("scalar dipole must have shape (nstates, nstates)")
    elif dipole.ndim == 3:
        if dipole.shape[1:] != (energies.size, energies.size):
            raise ValueError("vector dipole must have shape (npol, nstates, nstates)")
    else:
        raise ValueError("dipole must have shape (nstates, nstates) or (npol, nstates, nstates)")

    ground = int(ground)
    if ground < 0 or ground >= energies.size:
        raise ValueError("ground must index one of the states")

    if states is None:
        states = np.arange(energies.size)
    else:
        states = np.asarray(states, dtype=int)
        if states.ndim != 1:
            raise ValueError("states must be a one-dimensional sequence of indices")
        if np.any(states < 0) or np.any(states >= energies.size):
            raise ValueError("states contains an index outside the energy array")

    return energies - energies[ground], dipole, ground, states


def _state_widths(gamma, nstates):
    widths = np.asarray(gamma, dtype=float)
    if widths.ndim == 0:
        return np.full(nstates, float(widths))
    if widths.shape != (nstates,):
        raise ValueError("gamma must be a scalar or one value per state")
    return widths


def _sfg_scalar(energies, dipole, omega1, omega2, gamma, ground, states, prefactor):
    omega_sigma = omega1 + omega2
    chi = np.zeros(np.broadcast_shapes(np.shape(omega1), np.shape(omega2)), dtype=complex)

    for n in states:
        dn = omega_sigma - energies[n] + 1j * gamma[n]
        for m in states:
            strength = dipole[ground, n] * dipole[n, m] * dipole[m, ground]
            if strength == 0.0:
                continue
            d1 = omega1 - energies[m] + 1j * gamma[m]
            d2 = omega2 - energies[m] + 1j * gamma[m]
            chi += strength * (1.0 / (dn * d1) + 1.0 / (dn * d2))

    return prefactor * chi


def _sfg_tensor(energies, dipole, omega1, omega2, gamma, ground, states, prefactor):
    omega_sigma = omega1 + omega2
    freq_shape = np.broadcast_shapes(np.shape(omega1), np.shape(omega2))
    npol = dipole.shape[0]
    chi = np.zeros((npol, npol, npol) + freq_shape, dtype=complex)

    for i in range(npol):
        for j in range(npol):
            for k in range(npol):
                component = np.zeros(freq_shape, dtype=complex)
                for n in states:
                    dn = omega_sigma - energies[n] + 1j * gamma[n]
                    for m in states:
                        strength1 = (
                            dipole[i, ground, n]
                            * dipole[j, n, m]
                            * dipole[k, m, ground]
                        )
                        strength2 = (
                            dipole[i, ground, n]
                            * dipole[k, n, m]
                            * dipole[j, m, ground]
                        )
                        if strength1 == 0.0 and strength2 == 0.0:
                            continue
                        d1 = omega1 - energies[m] + 1j * gamma[m]
                        d2 = omega2 - energies[m] + 1j * gamma[m]
                        component += strength1 / (dn * d1)
                        component += strength2 / (dn * d2)
                chi[i, j, k] = prefactor * component

    return chi


def sum_frequency_generation(
    energies,
    dipole,
    omega1,
    omega2,
    gamma=0.0,
    ground=0,
    states=None,
    prefactor=1.0,
):
    """Return the resonant second-order SFG susceptibility.

    The evaluated response is
    ``chi2(-omega1-omega2; omega1, omega2)``.  For scalar dipoles the return
    value has the broadcast shape of ``omega1`` and ``omega2``.  For Cartesian
    dipoles with shape ``(npol, nstates, nstates)``, the return value has shape
    ``(npol, npol, npol, ...)`` where the first index is the emitted
    polarization and the next two indices correspond to ``omega1`` and
    ``omega2``.

    Parameters
    ----------
    energies : array_like, shape (nstates,)
        State energies.  The ground-state energy is subtracted internally.
    dipole : array_like
        Scalar dipoles ``mu[a, b]`` or vector dipoles ``mu[p, a, b]`` in the
        energy eigenbasis.
    omega1, omega2 : float or array_like
        Incoming frequencies.
    gamma : float or array_like, optional
        Homogeneous linewidth/dephasing parameter for each state.
    ground : int, optional
        Initial state index.
    states : sequence of int, optional
        Intermediate states to include.  By default all states are included.
    prefactor : complex, optional
        Multiplicative factor for unit conventions, densities, or constants.
    """

    energies, dipole, ground, states = _validate_states(energies, dipole, ground, states)
    gamma = _state_widths(gamma, energies.size)
    omega1, omega2 = np.broadcast_arrays(np.asarray(omega1), np.asarray(omega2))

    if dipole.ndim == 2:
        return _sfg_scalar(energies, dipole, omega1, omega2, gamma, ground, states, prefactor)
    return _sfg_tensor(energies, dipole, omega1, omega2, gamma, ground, states, prefactor)


def second_harmonic_generation(
    energies,
    dipole,
    omega,
    gamma=0.0,
    ground=0,
    states=None,
    prefactor=1.0,
):
    """Return the SHG susceptibility ``chi2(-2 omega; omega, omega)``."""

    return sum_frequency_generation(
        energies,
        dipole,
        omega,
        omega,
        gamma=gamma,
        ground=ground,
        states=states,
        prefactor=prefactor,
    )


def phase_matching_sinc(delta_k, length):
    """Return the complex plane-wave phase-matching amplitude.

    The result is ``exp(i delta_k L / 2) sinc(delta_k L / 2)``, with NumPy's
    normalized ``sinc`` convention handled internally.
    """

    phase = np.asarray(delta_k) * length / 2.0
    return np.exp(1j * phase) * np.sinc(phase / np.pi)


def intensity(chi2, phase_matching=1.0):
    """Return an uncalibrated SFG/SHG intensity proportional to ``|chi2|^2``."""

    return np.abs(np.asarray(chi2) * phase_matching) ** 2


quadratic_susceptibility = sum_frequency_generation
sfg = sum_frequency_generation
shg = second_harmonic_generation


__all__ = [
    "intensity",
    "phase_matching_sinc",
    "quadratic_susceptibility",
    "second_harmonic_generation",
    "sfg",
    "shg",
    "sum_frequency_generation",
]
