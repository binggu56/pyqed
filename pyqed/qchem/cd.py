"""Circular dichroism from CASCI and TDDFT/TDA backends."""

from dataclasses import dataclass, fields

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
    """Circular-dichroism transition data."""

    ground: int
    states: np.ndarray
    excitation_energies: np.ndarray
    electric_dipoles: np.ndarray
    magnetic_dipoles: np.ndarray
    rotatory_strengths: np.ndarray
    oscillator_strengths: np.ndarray
    origin: np.ndarray
    solvent_response_energies: np.ndarray = None
    solvent_response_corrections: np.ndarray = None
    solvent_response_model: str = None
    solvent_response_eps: float = None
    solvent_response_matrix: np.ndarray = None
    solvent_response_vectors: np.ndarray = None
    solvent_response_electric_dipoles: np.ndarray = None
    solvent_response_magnetic_dipoles: np.ndarray = None
    solvent_response_rotatory_strengths: np.ndarray = None
    solvent_response_oscillator_strengths: np.ndarray = None


class CD:
    """
    Circular dichroism using CASCI or TDDFT/TDA transition data.

    Parameters
    ----------
    backend
        A completed ``pyqed.qchem.CASCI`` calculation or a completed native
        ``TDA``/``TDDFT`` calculation.  Call ``run(...)`` before building CD
        transition data.
    origin
        Common gauge origin in bohr.  Defaults to the nuclear charge center,
        matching the builtin magnetic-dipole convention.
    """

    def __init__(self, backend, origin=None):
        self.backend = backend
        self.origin = self._resolve_origin(origin)
        self.result = None

    def _store_result(self, result):
        self.result = result
        for field in fields(result):
            setattr(self, field.name, getattr(result, field.name))
        return result

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

    def _is_td_backend(self):
        return (
            hasattr(self.backend, "transition_dipole")
            and hasattr(self.backend, "transition_magnetic_dipole")
            and hasattr(self.backend, "xy")
            and hasattr(self.backend, "e")
        )

    def _check_td_backend(self):
        if getattr(self.backend, "e", None) is None or getattr(self.backend, "xy", None) is None:
            raise ValueError("Run the TDA/TDDFT backend before computing CD data.")

    def _td_state_labels(self, states):
        nstates = np.asarray(self.backend.e, dtype=float).size
        if states is None:
            labels = np.arange(1, nstates + 1, dtype=int)
        else:
            labels = np.atleast_1d(np.asarray(states, dtype=int))
        if labels.size == 0:
            raise ValueError(
                "CD target states must contain at least one excited state; "
                "run TDDFT/TDA with nstates > 0 or pass target state labels."
            )
        if np.any(labels < 1) or np.any(labels > nstates):
            raise IndexError("One or more TDDFT/TDA target state labels are out of range.")
        return labels, labels - 1

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

    def _full_mo_transition_density(self, bra, ket):
        if _is_uhf_reference(self.backend.mo_coeff):
            raise NotImplementedError(
                "LR-PCM CD corrections currently support restricted CASCI references only."
            )
        ncore = int(self.backend.ncore)
        ncas = int(self.backend.ncas)
        nmo = int(self.backend.mf.nmo)
        dm_mo = np.zeros((nmo, nmo), dtype=float)
        dm_mo[ncore:ncore + ncas, ncore:ncore + ncas] = self.backend.make_tdm1(bra, ket)
        return dm_mo

    def _mo_density_to_ao(self, dm_mo):
        coeff = np.asarray(self.backend.mo_coeff)
        return coeff @ dm_mo @ coeff.conj().T

    def _make_fast_solvent(self, eps):
        try:
            from pyqed.qchem.solvent.pcm import PCM
        except (ImportError, OSError) as exc:
            raise RuntimeError("LR-PCM CD corrections require the PCM solvent module.") from exc

        solvent = PCM(self.backend.mol)
        reference = getattr(self.backend, "with_solvent", None)
        if reference is not None:
            for key in (
                "method",
                "vdw_scale",
                "r_probe",
                "radii_table",
                "lebedev_order",
                "max_memory",
                "verbose",
            ):
                if hasattr(reference, key):
                    setattr(solvent, key, getattr(reference, key))
        solvent.eps = float(eps)
        solvent.equilibrium_solvation = False
        return solvent

    def _lr_pcm_response_matrix(self, states, ground, eps=1.78):
        solvent = self._make_fast_solvent(eps)
        tdms = []
        response_potentials = []
        for state in states:
            tdm_mo = self._full_mo_transition_density(int(state), ground)
            tdm_ao = self._mo_density_to_ao(tdm_mo)
            tdms.append(tdm_ao)
            response_potentials.append(solvent._B_dot_x(tdm_ao))

        nstates = len(tdms)
        response_matrix = np.empty((nstates, nstates), dtype=float)
        for i, tdm_i in enumerate(tdms):
            for j, v_j in enumerate(response_potentials):
                response_matrix[i, j] = np.einsum("ij,ji->", v_j, tdm_i, optimize=True).real
        return 0.5 * (response_matrix + response_matrix.T)

    @staticmethod
    def _rotate_transition_vectors(values, vectors):
        return np.einsum("sa,sx->ax", vectors, values, optimize=True)

    def _run_td(self, ground=0, states=None, solvent_response=None):
        if int(ground) != 0:
            raise ValueError("TDDFT/TDA CD uses the electronic ground state; ground must be 0.")
        if solvent_response is not None and str(solvent_response).lower() not in {"none", "false"}:
            raise ValueError(
                "For TDDFT/TDA, attach PCM to the TD object itself with "
                "td.PCM().run(...); CD will then use the PCM-corrected TD states."
            )

        self._check_td_backend()
        state_labels, state_idx = self._td_state_labels(states)
        excitation_energies = np.asarray(self.backend.e, dtype=float)[state_idx]

        electric = -np.asarray(
            self.backend.transition_dipole(center=self.origin),
            dtype=float,
        )[state_idx]
        magnetic = np.asarray(
            self.backend.transition_magnetic_dipole(center=self.origin, convention="cd"),
            dtype=float,
        )[state_idx]

        rotatory = -np.real(np.einsum("nx,nx->n", electric, magnetic.conj(), optimize=True))
        oscillator = (2.0 / 3.0) * excitation_energies * np.einsum(
            "nx,nx->n", electric, electric.conj(), optimize=True
        ).real

        result = CDResult(
            ground=0,
            states=state_labels,
            excitation_energies=excitation_energies,
            electric_dipoles=electric,
            magnetic_dipoles=magnetic,
            rotatory_strengths=rotatory,
            oscillator_strengths=oscillator,
            origin=self.origin.copy(),
        )
        return self._store_result(result)

    def run(self, ground=0, states=None, solvent_response=None, solvent_response_eps=1.78):
        """
        Compute transition moments and rotatory strengths from one ground state.

        Parameters
        ----------
        solvent_response
            If ``'lr_pcm'``/``'lr'`` is requested, compute perturbative
            non-equilibrium LR-PCM corrections to the vertical excitation
            energies using the CASCI transition densities and an optical
            dielectric ``solvent_response_eps``.
        solvent_response_eps
            Fast-solvent dielectric used for LR-PCM.  The common default 1.78
            mirrors the optical-dielectric convention used in vertical
            non-equilibrium PCM spectra.

        Returns
        -------
        CDResult
            Energies are in hartree and rotatory strengths are in atomic units.
        """
        if self._is_td_backend():
            return self._run_td(
                ground=ground,
                states=states,
                solvent_response=solvent_response,
            )

        self._check_backend()
        e_tot = np.asarray(self.backend.e_tot, dtype=float)
        ground = int(ground)
        if ground < 0 or ground >= e_tot.size:
            raise IndexError("ground state index is out of range.")

        if states is None:
            states = [idx for idx in range(e_tot.size) if idx != ground]
        states = np.atleast_1d(np.asarray(states, dtype=int))
        if states.size == 0:
            raise ValueError(
                "CD target states must contain at least one excited state; "
                "run CASCI with nstates > 1 or pass target state indices."
            )
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

        solvent_response_model = None
        solvent_response_corrections = None
        solvent_response_energies = None
        solvent_response_eps_used = None
        solvent_response_matrix = None
        solvent_response_vectors = None
        solvent_response_electric = None
        solvent_response_magnetic = None
        solvent_response_rotatory = None
        solvent_response_oscillator = None
        if solvent_response is not None:
            model = str(solvent_response).lower()
            if model not in {"lr", "lr_pcm", "lr-pcm", "none", "false"}:
                raise ValueError("solvent_response must be None or 'lr_pcm'.")
            if model not in {"none", "false"}:
                solvent_response_model = "lr_pcm"
                solvent_response_eps_used = float(solvent_response_eps)
                solvent_response_matrix = self._lr_pcm_response_matrix(
                    states,
                    ground,
                    eps=solvent_response_eps_used,
                )
                h_response = np.diag(excitation_energies) + solvent_response_matrix
                solvent_response_energies, solvent_response_vectors = np.linalg.eigh(h_response)
                solvent_response_corrections = solvent_response_energies - excitation_energies
                solvent_response_electric = self._rotate_transition_vectors(
                    electric,
                    solvent_response_vectors,
                )
                solvent_response_magnetic = self._rotate_transition_vectors(
                    magnetic,
                    solvent_response_vectors,
                )
                solvent_response_rotatory = -np.real(np.einsum(
                    "nx,nx->n",
                    solvent_response_electric,
                    solvent_response_magnetic.conj(),
                    optimize=True,
                ))
                solvent_response_oscillator = (2.0 / 3.0) * solvent_response_energies * np.einsum(
                    "nx,nx->n",
                    solvent_response_electric,
                    solvent_response_electric.conj(),
                    optimize=True,
                ).real

        result = CDResult(
            ground=ground,
            states=states,
            excitation_energies=excitation_energies,
            electric_dipoles=electric,
            magnetic_dipoles=magnetic,
            rotatory_strengths=rotatory,
            oscillator_strengths=oscillator,
            origin=self.origin.copy(),
            solvent_response_energies=solvent_response_energies,
            solvent_response_corrections=solvent_response_corrections,
            solvent_response_model=solvent_response_model,
            solvent_response_eps=solvent_response_eps_used,
            solvent_response_matrix=solvent_response_matrix,
            solvent_response_vectors=solvent_response_vectors,
            solvent_response_electric_dipoles=solvent_response_electric,
            solvent_response_magnetic_dipoles=solvent_response_magnetic,
            solvent_response_rotatory_strengths=solvent_response_rotatory,
            solvent_response_oscillator_strengths=solvent_response_oscillator,
        )
        return self._store_result(result)

    def spectrum(
        self,
        x=None,
        width=0.1,
        units="ev",
        lineshape="gaussian",
        result=None,
        energy_source="raw",
    ):
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
        energy_source
            ``'raw'`` uses CASCI excitation energies. ``'lr_pcm'`` uses
            perturbative LR-PCM corrected excitation energies from
            ``run(solvent_response='lr_pcm')``.
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

        source_key = str(energy_source).lower()
        if source_key in {"raw", "vertical", "casci"}:
            energies = result.excitation_energies
            strengths = result.rotatory_strengths
        elif source_key in {"lr", "lr_pcm", "lr-pcm", "solvent", "solvent_response", "corrected"}:
            if result.solvent_response_energies is None:
                raise ValueError(
                    "LR-PCM corrected energies are unavailable; call "
                    "run(solvent_response='lr_pcm') first."
                )
            energies = result.solvent_response_energies
            strengths = (
                result.solvent_response_rotatory_strengths
                if result.solvent_response_rotatory_strengths is not None
                else result.rotatory_strengths
            )
        else:
            raise ValueError("energy_source must be 'raw' or 'lr_pcm'.")

        centers = np.asarray(energies, dtype=float) * scale
        strengths = np.asarray(strengths, dtype=float)
        if centers.size == 0:
            raise ValueError("Cannot broaden a CD spectrum with no transition energies.")
        width = float(width)
        if width <= 0.0:
            raise ValueError("width must be positive.")

        shape = str(lineshape).lower()
        if shape not in {"gaussian", "gauss", "lorentzian", "lorentz"}:
            raise ValueError("lineshape must be 'gaussian' or 'lorentzian'.")

        if x is None:
            lo = max(0.0, float(np.min(centers) - 8.0 * width))
            hi = float(np.max(centers) + 8.0 * width)
            x = np.linspace(lo, hi, 1000)
        else:
            x = np.asarray(x, dtype=float)

        signal = np.zeros_like(x, dtype=float)
        for center, strength in zip(centers, strengths):
            if shape in {"gaussian", "gauss"}:
                line = np.exp(-0.5 * ((x - center) / width) ** 2) / (width * np.sqrt(2.0 * np.pi))
            else:
                line = (width / np.pi) / ((x - center) ** 2 + width ** 2)
            signal += strength * line
        return x, signal

    def plot(
        self,
        x=None,
        width=0.1,
        units="ev",
        lineshape="gaussian",
        ax=None,
        energy_source="raw",
        **kwargs,
    ):
        """Plot a broadened CD spectrum and return ``(ax, x, signal)``."""
        import matplotlib.pyplot as plt

        x, signal = self.spectrum(
            x=x,
            width=width,
            units=units,
            lineshape=lineshape,
            energy_source=energy_source,
        )
        if ax is None:
            _, ax = plt.subplots()
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        ax.plot(x, signal, **kwargs)
        ax.set_xlabel("Energy (eV)" if str(units).lower().startswith("ev") else "Energy (hartree)")
        ax.set_ylabel("CD intensity (arb.)")
        return ax, x, signal
