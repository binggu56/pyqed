"""Circular dichroism from a CASCI backend."""

from dataclasses import dataclass

import numpy as np
from opt_einsum import contract

from pyqed.units import au2ev
from pyqed.qchem.mcscf.casci import (
    _is_uhf_reference,
    _normalize_spin_1e_operator,
    _transform_1e_operator_ao_to_mo,
    contract_with_tdm1,
)


@dataclass
class CDResult:
    """CASCI circular-dichroism transition data."""

    ground: int
    states: np.ndarray
    excitation_energies: np.ndarray
    electric_dipoles: np.ndarray
    magnetic_dipoles: np.ndarray
    rotatory_strengths: np.ndarray
    oscillator_strengths: np.ndarray
    origin: np.ndarray


class CD:
    """
    Circular dichroism using CASCI transition-density matrices.

    Parameters
    ----------
    backend
        A completed ``pyqed.qchem.CASCI`` calculation.  Call ``run(nstates=...)``
        on CASCI before building CD transition data.
    origin
        Common gauge origin in bohr.  Defaults to the nuclear charge center,
        matching the builtin magnetic-dipole convention.
    """

    def __init__(self, backend, origin=None):
        self.backend = backend
        self.origin = self._resolve_origin(origin)
        self.result = None

    def _resolve_origin(self, origin):
        if origin is not None:
            origin = np.asarray(origin, dtype=float)
            if origin.shape != (3,):
                raise ValueError("origin must be a length-3 Cartesian vector.")
            return origin

        mol = self.backend.mol
        if hasattr(mol, "nuc_charge_center"):
            return np.asarray(mol.nuc_charge_center(), dtype=float)
        atom_coords = np.asarray(mol.atom_coords(), dtype=float)
        atom_charges = np.asarray(mol.atom_charges(), dtype=float)
        return contract("z,zx->x", atom_charges, atom_coords) / atom_charges.sum()

    def _check_backend(self):
        if getattr(self.backend, "ci", None) is None or getattr(self.backend, "e_tot", None) is None:
            raise ValueError("Run the CASCI backend before computing CD data.")
        if getattr(self.backend, "SC1", None) is None:
            raise ValueError("CASCI backend is missing Slater-Condon one-body data.")

    def electric_dipole_operator(self, basis="mo"):
        """Electronic electric-dipole operator, ``mu = -r``, in AO or MO basis."""
        op = -np.asarray(self.backend.mol.moment_integral(center=self.origin), dtype=float)
        return self._operator_basis(op, basis)

    def magnetic_dipole_operator(self, basis="mo"):
        """
        Magnetic-dipole operator in AO or MO basis.

        The returned real matrix follows the standard orbital magnetic-dipole
        convention used for length-gauge CD, ``-0.5 * (r - origin) x grad``.
        """
        op = np.asarray(
            self.backend.mol.magnetic_dipole_integral(center=self.origin, convention="cd"),
            dtype=float,
        )
        return self._operator_basis(op, basis)

    def _operator_basis(self, op, basis):
        key = str(basis).lower()
        if key == "ao":
            return op
        if key != "mo":
            raise ValueError("basis must be 'ao' or 'mo'.")

        return np.asarray([
            _transform_1e_operator_ao_to_mo(component, self.backend.mo_coeff)
            for component in op
        ], dtype=object if _is_uhf_reference(self.backend.mo_coeff) else float)

    def _active_operator(self, op):
        ncore = int(self.backend.ncore)
        ncas = int(self.backend.ncas)
        h1a, h1b = _normalize_spin_1e_operator(op)
        active = (
            h1a[ncore:ncore + ncas, ncore:ncore + ncas],
            h1b[ncore:ncore + ncas, ncore:ncore + ncas],
        )
        if active[0] is active[1] or np.array_equal(active[0], active[1]):
            return active[0]
        return active

    def _transition_vector(self, operators_mo, bra, ket):
        values = []
        for op in operators_mo:
            values.append(
                contract_with_tdm1(
                    self.backend.ci[bra],
                    self.backend.ci[ket],
                    self.backend.binary,
                    self.backend.SC1,
                    self._active_operator(op),
                )
            )
        return np.asarray(values)

    def run(self, ground=0, states=None):
        """
        Compute transition moments and rotatory strengths from one ground state.

        Returns
        -------
        CDResult
            Energies are in hartree and rotatory strengths are in atomic units.
        """
        self._check_backend()
        e_tot = np.asarray(self.backend.e_tot, dtype=float)
        ground = int(ground)
        if ground < 0 or ground >= e_tot.size:
            raise IndexError("ground state index is out of range.")

        if states is None:
            states = [idx for idx in range(e_tot.size) if idx != ground]
        states = np.asarray(states, dtype=int)
        if np.any(states < 0) or np.any(states >= e_tot.size):
            raise IndexError("One or more target state indices are out of range.")
        if np.any(states == ground):
            raise ValueError("CD target states must not include the ground state.")

        electric_op = self.electric_dipole_operator(basis="mo")
        magnetic_op = self.magnetic_dipole_operator(basis="mo")

        electric = np.asarray([
            self._transition_vector(electric_op, state, ground)
            for state in states
        ])
        magnetic = np.asarray([
            self._transition_vector(magnetic_op, state, ground)
            for state in states
        ])

        excitation_energies = e_tot[states] - e_tot[ground]
        rotatory = -np.real(np.einsum("nx,nx->n", electric, magnetic.conj(), optimize=True))
        oscillator = (2.0 / 3.0) * excitation_energies * np.einsum(
            "nx,nx->n", electric, electric.conj(), optimize=True
        ).real

        self.result = CDResult(
            ground=ground,
            states=states,
            excitation_energies=excitation_energies,
            electric_dipoles=electric,
            magnetic_dipoles=magnetic,
            rotatory_strengths=rotatory,
            oscillator_strengths=oscillator,
            origin=self.origin.copy(),
        )
        return self.result

    def spectrum(self, x=None, width=0.1, units="ev", lineshape="gaussian", result=None):
        """
        Broaden the CD stick spectrum.

        Parameters
        ----------
        x
            Energy grid.  If omitted, a grid is generated around the transitions.
        width
            Gaussian sigma or Lorentzian half width at half maximum in ``units``.
        units
            ``'ev'`` or ``'au'``/``'hartree'``.
        lineshape
            ``'gaussian'`` or ``'lorentzian'``.
        """
        result = self.result if result is None else result
        if result is None:
            result = self.run()

        unit_key = str(units).lower()
        if unit_key in {"ev", "electronvolt", "electronvolts"}:
            scale = au2ev
        elif unit_key in {"au", "hartree", "ha"}:
            scale = 1.0
        else:
            raise ValueError("units must be 'ev' or 'au'.")

        centers = np.asarray(result.excitation_energies, dtype=float) * scale
        strengths = np.asarray(result.rotatory_strengths, dtype=float)
        width = float(width)
        if width <= 0.0:
            raise ValueError("width must be positive.")

        if x is None:
            lo = max(0.0, float(np.min(centers) - 8.0 * width))
            hi = float(np.max(centers) + 8.0 * width)
            x = np.linspace(lo, hi, 1000)
        else:
            x = np.asarray(x, dtype=float)

        signal = np.zeros_like(x, dtype=float)
        shape = str(lineshape).lower()
        for center, strength in zip(centers, strengths):
            if shape in {"gaussian", "gauss"}:
                line = np.exp(-0.5 * ((x - center) / width) ** 2) / (width * np.sqrt(2.0 * np.pi))
            elif shape in {"lorentzian", "lorentz"}:
                line = (width / np.pi) / ((x - center) ** 2 + width ** 2)
            else:
                raise ValueError("lineshape must be 'gaussian' or 'lorentzian'.")
            signal += strength * line
        return x, signal

    def plot(self, x=None, width=0.1, units="ev", lineshape="gaussian", ax=None, **kwargs):
        """Plot a broadened CD spectrum and return ``(ax, x, signal)``."""
        import matplotlib.pyplot as plt

        x, signal = self.spectrum(x=x, width=width, units=units, lineshape=lineshape)
        if ax is None:
            _, ax = plt.subplots()
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        ax.plot(x, signal, **kwargs)
        ax.set_xlabel("Energy (eV)" if str(units).lower().startswith("ev") else "Energy (hartree)")
        ax.set_ylabel("CD intensity (arb.)")
        return ax, x, signal
