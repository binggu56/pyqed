"""Dynamical compact-U(1) Schwinger model with a Wilson-dressed DVR kernel."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pyqed.dvr import ExponentialDVR


def _apply_one_body(bits: int, destination: int, source: int):
    """Apply ``c_destination^dagger c_source`` to a bit determinant."""
    if not (bits >> source) & 1 or (bits >> destination) & 1:
        return None
    source_sign = -1 if (bits & ((1 << source) - 1)).bit_count() % 2 else 1
    intermediate = bits ^ (1 << source)
    destination_sign = (
        -1
        if (intermediate & ((1 << destination) - 1)).bit_count() % 2
        else 1
    )
    return intermediate | (1 << destination), source_sign * destination_sign


class QuantumSchwingerDVR:
    r"""Exact physical-basis Hamiltonian for a compact quantum link field.

    There are two Dirac orbitals per periodic DVR point and one integer electric
    flux ``L_n`` per link.  The normal-ordered charge is

    .. math::

        q_n = n_{n,0} + n_{n,1} - 1,

    and physical basis states obey the convention

    .. math::

        L_{n-1} - L_n = q_n.

    The link convention is ``U_n=exp(-i theta_n)``, so ``U_n`` lowers ``L_n``.
    Electric flux is truncated to ``[-flux_cutoff, flux_cutoff]``.  The
    nonlocal fermion hopping uses the odd-grid Fourier derivative and the
    shortest Wilson string between every pair of sites.

    This exact-diagonalization representation is intended as a small-system
    reference.  Its many-body Hilbert space is exponential even though the
    underlying Wilson-DVR spatial kernel admits the ``O(N log N)`` FFT
    factorization implemented by :class:`pyqed.lgt.WilsonFourierDVR`.
    """

    def __init__(
        self,
        npts: int,
        length: float,
        *,
        coupling: float = 1.0,
        mass: float = 0.0,
        flux_cutoff: int = 2,
    ):
        if int(npts) < 3 or int(npts) % 2 == 0:
            raise ValueError("npts must be an odd integer of at least three")
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("length must be positive and finite")
        if not np.isfinite(coupling) or coupling <= 0.0:
            raise ValueError("coupling must be positive and finite")
        if int(flux_cutoff) < 1:
            raise ValueError("flux_cutoff must be a positive integer")

        self.npts = int(npts)
        self.length = float(length)
        self.spacing = self.length / self.npts
        self.coupling = float(coupling)
        self.mass = float(mass)
        self.flux_cutoff = int(flux_cutoff)
        self.norbitals = 2 * self.npts
        self.derivative = ExponentialDVR(
            npts=self.npts, L=self.length
        ).derivative()

        self.basis_bits, self.basis_flux = self._build_physical_basis()
        self.index = {
            (int(bits), *map(int, flux)): position
            for position, (bits, flux) in enumerate(
                zip(self.basis_bits, self.basis_flux)
            )
        }
        self.dimension = len(self.basis_bits)
        self.hamiltonian = None
        self.vector_operator = None
        self.scalar_operator = None
        self.energies = None
        self.states = None
        self.vector_strengths = None
        self.scalar_strengths = None
        self.vacuum_dimension = None
        self.vector_excitation_energy = None
        self.vector_momentum = 2.0 * np.pi / self.length
        self.vector_gap = None
        self.scalar_gap = None
        self.vector_level = None
        self.scalar_level = None

    def charges(self, bits: int):
        return np.asarray(
            [
                ((bits >> (2 * site)) & 1)
                + ((bits >> (2 * site + 1)) & 1)
                - 1
                for site in range(self.npts)
            ],
            dtype=int,
        )

    def gauss_law(self, bits: int, flux):
        flux = np.asarray(flux, dtype=int)
        return np.roll(flux, 1) - flux - self.charges(bits)

    def _build_physical_basis(self):
        bits_list = []
        flux_list = []
        # Charge neutrality fixes half filling: N fermions in 2N orbitals.
        for occupied in combinations(range(self.norbitals), self.npts):
            bits = sum(1 << orbital for orbital in occupied)
            charge = self.charges(bits)
            cumulative = np.cumsum(charge)
            for boundary_flux in range(-self.flux_cutoff, self.flux_cutoff + 1):
                flux = boundary_flux - cumulative
                if np.all(np.abs(flux) <= self.flux_cutoff):
                    bits_list.append(bits)
                    flux_list.append(flux)
        return np.asarray(bits_list, dtype=np.int64), np.asarray(flux_list, dtype=np.int16)

    def _signed(self, displacement: int):
        half = self.npts // 2
        return (int(displacement) + half) % self.npts - half

    def _transport_flux(self, flux, destination_site: int, source_site: int):
        """Apply the shortest Wilson string transporting source to destination."""
        shifted = np.asarray(flux, dtype=int).copy()
        source_displacement = self._signed(source_site - destination_site)
        if source_displacement > 0:
            # U_destination ... U_source-1 lowers flux on the forward path.
            for step in range(source_displacement):
                shifted[(destination_site + step) % self.npts] -= 1
        elif source_displacement < 0:
            # The reverse path contains U^dagger and raises the traversed flux.
            for step in range(-source_displacement):
                shifted[(source_site + step) % self.npts] += 1
        return shifted

    def build_hamiltonian(self):
        rows = []
        columns = []
        values = []
        electric_prefactor = 0.5 * self.coupling**2 * self.spacing

        for column, (bits_raw, flux_raw) in enumerate(
            zip(self.basis_bits, self.basis_flux)
        ):
            bits = int(bits_raw)
            flux = np.asarray(flux_raw, dtype=int)
            mass_value = 0.0
            for site in range(self.npts):
                mass_value += (
                    ((bits >> (2 * site)) & 1)
                    - ((bits >> (2 * site + 1)) & 1)
                )
            diagonal = electric_prefactor * float(flux @ flux)
            diagonal += self.mass * mass_value
            rows.append(column)
            columns.append(column)
            values.append(diagonal)

            for destination_site in range(self.npts):
                for source_site in range(self.npts):
                    derivative = self.derivative[destination_site, source_site]
                    if abs(derivative) < 1.0e-15:
                        continue
                    transported_flux = self._transport_flux(
                        flux, destination_site, source_site
                    )
                    inside_cutoff = np.all(
                        np.abs(transported_flux) <= self.flux_cutoff
                    )
                    for destination_spin, source_spin in ((0, 1), (1, 0)):
                        destination = 2 * destination_site + destination_spin
                        source = 2 * source_site + source_spin
                        result = _apply_one_body(bits, destination, source)
                        if result is None:
                            continue
                        new_bits, sign = result
                        key = (new_bits, *map(int, transported_flux))
                        row = self.index.get(key)
                        if row is None:
                            if inside_cutoff:
                                raise RuntimeError(
                                    "Wilson hop did not preserve the Gauss-law basis"
                                )
                            continue
                        rows.append(row)
                        columns.append(column)
                        values.append(-1j * derivative * sign)

        hamiltonian = sp.coo_matrix(
            (values, (rows, columns)),
            shape=(self.dimension, self.dimension),
            dtype=complex,
        ).tocsr()
        hamiltonian.sum_duplicates()
        self.hamiltonian = hamiltonian
        return hamiltonian

    def build_channel_operators(self):
        scalar_diagonal = np.zeros(self.dimension)
        vector_diagonal = np.zeros(self.dimension)
        density_phase = np.cos(
            2.0 * np.pi * np.arange(self.npts) / self.npts
        )
        for column, (bits_raw, flux) in enumerate(
            zip(self.basis_bits, self.basis_flux)
        ):
            bits = int(bits_raw)
            charge = self.charges(bits)
            vector_diagonal[column] = density_phase @ charge
            for site in range(self.npts):
                scalar_diagonal[column] += (
                    ((bits >> (2 * site)) & 1)
                    - ((bits >> (2 * site + 1)) & 1)
                )

        self.scalar_operator = sp.diags(scalar_diagonal, format="csr")
        # The lowest nonzero Fourier component of j^0 is a gauge-invariant
        # vector-channel interpolator.  Its excitation energy obeys
        # E(k)^2 = M_V^2 + k^2 in the continuum.
        self.vector_operator = sp.diags(vector_diagonal, format="csr")
        return self.vector_operator, self.scalar_operator

    @staticmethod
    def _channel_level(
        strengths, first_excited: int, relative_tolerance=1.0e-8
    ):
        excited = np.asarray(strengths[first_excited:])
        if excited.size == 0 or np.max(excited) <= 0.0:
            return None
        threshold = max(1.0e-13, relative_tolerance * float(np.max(excited)))
        candidates = np.flatnonzero(excited > threshold)
        return (
            None
            if candidates.size == 0
            else int(candidates[0] + first_excited)
        )

    def run(self, nroots: int = 24, tolerance: float = 1.0e-10):
        if self.hamiltonian is None:
            self.build_hamiltonian()
        if self.vector_operator is None:
            self.build_channel_operators()

        nroots = min(int(nroots), self.dimension)
        if nroots == self.dimension or self.dimension <= 192:
            energies, states = scipy.linalg.eigh(
                self.hamiltonian.toarray(), check_finite=False
            )
            energies = energies[:nroots]
            states = states[:, :nroots]
        else:
            energies, states = spla.eigsh(
                self.hamiltonian,
                k=nroots,
                which="SA",
                tol=tolerance,
            )
            order = np.argsort(energies)
            energies = energies[order]
            states = states[:, order]

        vacuum_tolerance = max(tolerance, 1.0e-9)
        self.vacuum_dimension = int(
            np.count_nonzero(np.abs(energies - energies[0]) <= vacuum_tolerance)
        )
        vacua = states[:, : self.vacuum_dimension]
        vector_sources = self.vector_operator @ vacua
        scalar_sources = self.scalar_operator @ vacua
        self.vector_strengths = np.sum(
            np.abs(states.conj().T @ vector_sources) ** 2, axis=1
        ) / self.vacuum_dimension
        self.scalar_strengths = np.sum(
            np.abs(states.conj().T @ scalar_sources) ** 2, axis=1
        ) / self.vacuum_dimension
        self.energies = np.asarray(energies)
        self.states = states
        self.vector_level = self._channel_level(
            self.vector_strengths, self.vacuum_dimension
        )
        self.scalar_level = self._channel_level(
            self.scalar_strengths, self.vacuum_dimension
        )
        self.vector_excitation_energy = (
            None
            if self.vector_level is None
            else float(energies[self.vector_level] - energies[0])
        )
        self.vector_gap = (
            None
            if self.vector_excitation_energy is None
            else float(
                np.sqrt(
                    max(
                        self.vector_excitation_energy**2
                        - self.vector_momentum**2,
                        0.0,
                    )
                )
            )
        )
        self.scalar_gap = (
            None
            if self.scalar_level is None
            else float(energies[self.scalar_level] - energies[0])
        )
        return self
