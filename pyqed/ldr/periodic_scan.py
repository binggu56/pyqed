"""Half-filled electronic scans for periodic SSH-Holstein normal modes."""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from .periodic import PeriodicSSHHolsteinMomentumGQD


def real_normal_modes(ncells):
    r"""Return an orthonormal real Fourier basis for ``ncells`` coordinates.

    The profiles obey

    .. math::

       \frac{1}{N}\sum_R f_{\lambda R}f_{\mu R}=\delta_{\lambda\mu}.
    """

    ncells = int(ncells)
    if ncells < 3:
        raise ValueError("ncells must be at least 3.")
    cells = np.arange(ncells, dtype=float)
    modes = [
        {
            "name": "q0",
            "q_index": 0,
            "qpoint": 0.0,
            "component": "uniform",
            "profile": np.ones(ncells),
        }
    ]
    for q_index in range(1, (ncells - 1) // 2 + 1):
        qpoint = 2.0 * np.pi * q_index / ncells
        modes.extend(
            (
                {
                    "name": f"q{q_index}_cos",
                    "q_index": q_index,
                    "qpoint": qpoint,
                    "component": "cosine",
                    "profile": np.sqrt(2.0) * np.cos(qpoint * cells),
                },
                {
                    "name": f"q{q_index}_sin",
                    "q_index": q_index,
                    "qpoint": qpoint,
                    "component": "sine",
                    "profile": np.sqrt(2.0) * np.sin(qpoint * cells),
                },
            )
        )
    if ncells % 2 == 0:
        modes.append(
            {
                "name": "qpi",
                "q_index": ncells // 2,
                "qpoint": np.pi,
                "component": "alternating",
                "profile": (-1.0) ** cells,
            }
        )
    return tuple(modes)


class PeriodicSSHHolsteinHalfFilledScan:
    """Independent real-mode scans of a spinless half-filled supercell."""

    def __init__(
        self,
        *,
        ncells=4,
        hopping=0.6,
        dimerization=0.08,
        ssh_coupling=0.22,
        sublattice_bias=0.12,
        holstein_coupling=0.06,
        phonon_frequency=0.12,
    ):
        self.model = PeriodicSSHHolsteinMomentumGQD(
            ncells=ncells,
            q_index=1,
            hopping=hopping,
            dimerization=dimerization,
            ssh_coupling=ssh_coupling,
            sublattice_bias=sublattice_bias,
            holstein_coupling=holstein_coupling,
            phonon_frequency=phonon_frequency,
        )
        self.ncells = self.model.ncells
        self.norbitals = self.model.nstates
        self.nelectrons = self.ncells
        self.ndeterminants = comb(self.norbitals, self.nelectrons)
        self.modes = real_normal_modes(self.ncells)
        self.mode_names = tuple(mode["name"] for mode in self.modes)
        self.mode_qpoints = np.asarray(
            [mode["qpoint"] for mode in self.modes]
        )
        self.mode_profiles = np.asarray(
            [mode["profile"] for mode in self.modes]
        )
        gram = self.mode_profiles @ self.mode_profiles.T / self.ncells
        self.mode_orthogonality_error = float(
            np.max(np.abs(gram - np.eye(self.ncells)))
        )

        self.coordinates = None
        self.determinant_occupations = None
        self.one_particle_energies = None
        self.orbital_momentum_weights = None
        self.electronic_ground_energies = None
        self.fundamental_gaps = None
        self.single_excitation_energies = None
        self.many_body_energies = None
        self.vibronic_surfaces = None
        self.excitation_energies = None
        self.determinant_order = None
        self.minimum_gaps = None
        self.cosine_sine_spectrum_error = None
        self.success = False
        self.message = "not run"

    def scan(
        self,
        coordinates=None,
        *,
        nstates=None,
        max_determinants=200_000,
    ):
        """Scan every independent real normal mode with all others at zero."""

        if coordinates is None:
            coordinates = np.linspace(-3.0, 3.0, 121)
        coordinates = np.asarray(coordinates, dtype=float)
        if (
            coordinates.ndim != 1
            or coordinates.size < 2
            or not np.all(np.isfinite(coordinates))
            or np.any(np.diff(coordinates) <= 0.0)
        ):
            raise ValueError("coordinates must be finite and strictly increasing.")
        max_determinants = int(max_determinants)
        if self.ndeterminants > max_determinants:
            raise ValueError(
                f"The {self.ndeterminants} determinants exceed "
                f"max_determinants={max_determinants}."
            )
        if nstates is None:
            nstates = self.ndeterminants
        nstates = int(nstates)
        if not 0 < nstates <= self.ndeterminants:
            raise ValueError(
                f"nstates must be in [1, {self.ndeterminants}]."
            )

        determinant_occupations = np.zeros(
            (self.ndeterminants, self.norbitals),
            dtype=float,
        )
        for index, occupied in enumerate(
            combinations(range(self.norbitals), self.nelectrons)
        ):
            determinant_occupations[index, occupied] = 1.0

        one_particle = []
        momentum_weights = []
        many_body = []
        surfaces = []
        excitation = []
        orders = []
        single_excitation = []
        harmonic = 0.5 * self.model.phonon_frequency**2 * coordinates**2
        for mode in self.modes:
            hamiltonian = self.model.electronic_hamiltonian_for_profile(
                coordinates,
                mode["profile"],
            )
            orbital_energies, orbital_frames = np.linalg.eigh(hamiltonian)
            reshaped_frames = orbital_frames.reshape(
                coordinates.size,
                self.ncells,
                2,
                self.norbitals,
            )
            weights = np.sum(np.abs(reshaped_frames) ** 2, axis=2).transpose(
                0,
                2,
                1,
            )
            determinant_energies = orbital_energies @ determinant_occupations.T
            order = np.argsort(determinant_energies, axis=1)[:, :nstates]
            sorted_energies = np.take_along_axis(
                determinant_energies,
                order,
                axis=1,
            )
            occupied = orbital_energies[:, : self.nelectrons]
            virtual = orbital_energies[:, self.nelectrons :]

            one_particle.append(orbital_energies)
            momentum_weights.append(weights)
            many_body.append(sorted_energies)
            surfaces.append(sorted_energies + harmonic[:, None])
            excitation.append(sorted_energies - sorted_energies[:, :1])
            orders.append(order)
            single_excitation.append(
                virtual[:, None, :] - occupied[:, :, None]
            )

        self.coordinates = coordinates
        self.determinant_occupations = determinant_occupations
        self.one_particle_energies = np.asarray(one_particle)
        self.orbital_momentum_weights = np.asarray(momentum_weights)
        self.many_body_energies = np.asarray(many_body)
        self.vibronic_surfaces = np.asarray(surfaces)
        self.excitation_energies = np.asarray(excitation)
        self.determinant_order = np.asarray(orders)
        self.single_excitation_energies = np.asarray(single_excitation)
        self.electronic_ground_energies = self.many_body_energies[:, :, 0]
        self.fundamental_gaps = (
            self.one_particle_energies[:, :, self.nelectrons]
            - self.one_particle_energies[:, :, self.nelectrons - 1]
        )
        self.minimum_gaps = np.min(self.fundamental_gaps, axis=1)

        paired_errors = []
        for left, mode in enumerate(self.modes):
            if mode["component"] != "cosine":
                continue
            for right, candidate in enumerate(self.modes):
                if (
                    candidate["q_index"] == mode["q_index"]
                    and candidate["component"] == "sine"
                ):
                    paired_errors.append(
                        np.max(
                            np.abs(
                                self.one_particle_energies[left]
                                - self.one_particle_energies[right]
                            )
                        )
                    )
        self.cosine_sine_spectrum_error = float(
            max(paired_errors, default=0.0)
        )
        self.success = True
        self.message = "completed independent real-mode half-filled scans"
        return self


__all__ = [
    "PeriodicSSHHolsteinHalfFilledScan",
    "real_normal_modes",
]
