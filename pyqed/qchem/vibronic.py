"""Ab-initio ingredients for linear vibronic-coupling Hamiltonians.

The central object is :class:`LVC`.  Its linear electronic-mode terms are stored
as a single tensor returned by ``vibronic_couplings()``:

``V[a, b, m]``
    Linear coefficient multiplying normal coordinate ``Q_m`` in electronic
    matrix element ``H_ab``.  Diagonal entries are ``dE_a / dQ_m`` and
    off-diagonal entries are ``<a|dH/dQ_m|b>``.

The normal-mode vectors are assumed to be Cartesian displacements per unit
normal coordinate, i.e. ``dR / dQ`` with shape ``(nmodes, natom, 3)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


@dataclass(frozen=True)
class LVC:
    """Linear vibronic-coupling model.

    Parameters
    ----------
    reference_energies
        Electronic energies at the reference geometry, shape ``(nstates,)``.
    mode_frequencies
        Harmonic frequencies for the selected normal modes, shape ``(nmodes,)``.
    couplings
        Linear vibronic coefficients ``V[a, b, m]``, shape
        ``(nstates, nstates, nmodes)``.
    normal_modes
        Optional Cartesian normal-mode vectors ``dR/dQ``.
    mode_ids
        Optional external 1-based normal-mode labels, useful when importing
        SHARC ``LVC.template`` files that keep the original mode numbering.
    reference_geometry
        Optional reference Cartesian geometry.
    """

    reference_energies: np.ndarray
    mode_frequencies: np.ndarray
    couplings: np.ndarray
    normal_modes: np.ndarray | None = None
    mode_ids: np.ndarray | None = None
    reference_geometry: np.ndarray | None = None

    @classmethod
    def from_casci(
        cls,
        mc,
        modes,
        frequencies,
        state_ids=None,
        mode_ids=None,
        reference_geometry=None,
        return_quadratic=False,
    ):
        """Build an ``LVC`` model from a completed CASCI calculation.

        ``modes`` must contain Cartesian displacement coefficients for the
        chosen normal coordinates, with shape ``(nmodes, natom, 3)`` or an
        equivalent flattened form accepted by ``mc.vibronic_couplings()``.
        """

        f, g = mc.vibronic_couplings(state_ids=state_ids, modes=modes)
        f = np.real_if_close(f)
        g = np.real_if_close(g)
        if state_ids is None:
            state_ids = tuple(range(f.shape[0]))
        state_ids = np.asarray(state_ids, dtype=int)
        energies = np.asarray(mc.e_tot, dtype=float)[state_ids]

        if reference_geometry is None and hasattr(mc, "mol"):
            reference_geometry = np.asarray(mc.mol.atom_coords(), dtype=float)

        model = cls(
            reference_energies=energies,
            mode_frequencies=frequencies,
            couplings=f,
            normal_modes=modes,
            mode_ids=mode_ids,
            reference_geometry=reference_geometry,
        )
        if return_quadratic:
            return model, g
        return model

    def __post_init__(self):
        object.__setattr__(self, "reference_energies", _as_1d(self.reference_energies, "reference_energies"))
        object.__setattr__(self, "mode_frequencies", _as_1d(self.mode_frequencies, "mode_frequencies"))
        object.__setattr__(self, "couplings", np.asarray(self.couplings, dtype=float))
        if self.normal_modes is not None:
            object.__setattr__(self, "normal_modes", np.asarray(self.normal_modes, dtype=float))
        if self.mode_ids is not None:
            object.__setattr__(self, "mode_ids", np.asarray(self.mode_ids, dtype=int))
        if self.reference_geometry is not None:
            object.__setattr__(
                self,
                "reference_geometry",
                np.asarray(self.reference_geometry, dtype=float),
            )
        self._validate_shapes()

    @property
    def nstates(self) -> int:
        return int(self.reference_energies.size)

    @property
    def nmodes(self) -> int:
        return int(self.mode_frequencies.size)

    @property
    def state_forces(self) -> np.ndarray:
        """Return diagonal force coefficients, ``-dE_state/dQ_mode``."""

        return -np.diagonal(self.couplings, axis1=0, axis2=1).T

    def vibronic_couplings(self, copy=True) -> np.ndarray:
        """Return the full linear vibronic-coupling tensor ``V[a, b, m]``."""

        return self.couplings.copy() if copy else self.couplings

    def electronic_hamiltonian(self, q: ArrayLike, include_harmonic=False) -> np.ndarray:
        """Evaluate the electronic vibronic matrix at normal coordinates ``q``.

        ``include_harmonic=True`` adds the common harmonic nuclear energy
        ``0.5 * sum_m omega_m**2 * q_m**2`` to every diagonal state.
        """

        q = _as_1d(q, "q")
        if q.size != self.nmodes:
            raise ValueError(f"q must have shape ({self.nmodes},), got {q.shape}.")

        h = np.diag(self.reference_energies).astype(float)
        h += np.einsum("abm,m->ab", self.couplings, q, optimize=True)
        if include_harmonic:
            h[np.diag_indices(self.nstates)] += 0.5 * np.dot(self.mode_frequencies**2, q**2)
        return h

    def adiabatic_energies(self, q: ArrayLike, include_harmonic=False) -> np.ndarray:
        """Return eigenvalues of the vibronic matrix at ``q``."""

        return np.linalg.eigvalsh(self.electronic_hamiltonian(q, include_harmonic=include_harmonic))

    def _validate_shapes(self):
        nstates = self.reference_energies.size
        nmodes = self.mode_frequencies.size
        if self.couplings.shape != (nstates, nstates, nmodes):
            raise ValueError(
                "couplings must have shape "
                f"({nstates}, {nstates}, {nmodes}), got {self.couplings.shape}."
            )
        if not np.allclose(self.couplings, self.couplings.swapaxes(0, 1)):
            raise ValueError("couplings must be symmetric in the state indices.")
        if self.normal_modes is not None and self.normal_modes.shape[0] != nmodes:
            raise ValueError(
                f"normal_modes must have {nmodes} modes in axis 0, got {self.normal_modes.shape}."
            )
        if self.mode_ids is not None:
            if self.mode_ids.shape != (nmodes,):
                raise ValueError(f"mode_ids must have shape ({nmodes},), got {self.mode_ids.shape}.")
            if len(set(self.mode_ids.tolist())) != nmodes:
                raise ValueError("mode_ids must be unique.")


def build_lvc(
    reference_energies,
    mode_frequencies,
    normal_modes,
    state_gradients,
    derivative_couplings=None,
    mode_derivative_couplings=None,
    vibronic_couplings=None,
    mode_ids=None,
    reference_geometry=None,
):
    """Build an :class:`LVC` model from ab-initio gradients and couplings.

    The diagonal tensor entries come from projected state gradients:

    ``V[a, a, m] = dE_a / dQ_m``

    The off-diagonal entries are estimated from derivative couplings:

    ``V[a, b, m] = (E_b - E_a) <psi_a | d/dQ_m | psi_b>``
    """

    reference_energies = _as_1d(reference_energies, "reference_energies")
    mode_frequencies = _as_1d(mode_frequencies, "mode_frequencies")
    normal_modes = np.asarray(normal_modes, dtype=float)
    state_gradients = np.asarray(state_gradients, dtype=float)

    diagonal = project_cartesian_to_modes(state_gradients, normal_modes)
    if diagonal.shape != (reference_energies.size, mode_frequencies.size):
        raise ValueError(
            "Projected state gradients must have shape "
            f"({reference_energies.size}, {mode_frequencies.size}), got {diagonal.shape}."
        )

    if vibronic_couplings is None:
        couplings = vibronic_couplings_from_derivative_couplings(
            reference_energies,
            normal_modes=normal_modes,
            derivative_couplings=derivative_couplings,
            mode_derivative_couplings=mode_derivative_couplings,
        )
    else:
        couplings = np.asarray(vibronic_couplings, dtype=float).copy()

    for state in range(reference_energies.size):
        couplings[state, state] = diagonal[state]

    return LVC(
        reference_energies=reference_energies,
        mode_frequencies=mode_frequencies,
        couplings=couplings,
        normal_modes=normal_modes,
        mode_ids=mode_ids,
        reference_geometry=reference_geometry,
    )


def build_linear_vibronic_model(*args, **kwargs):
    """Backward-compatible alias for :func:`build_lvc`."""

    return build_lvc(*args, **kwargs)


def project_cartesian_to_modes(cartesian_values, normal_modes):
    """Project Cartesian derivatives onto normal coordinates.

    ``cartesian_values`` may have leading dimensions, followed by
    ``(natom, 3)``.  The result has the same leading dimensions followed by
    ``(nmodes,)``.
    """

    cartesian_values = np.asarray(cartesian_values, dtype=float)
    normal_modes = np.asarray(normal_modes, dtype=float)
    if normal_modes.ndim != 3 or normal_modes.shape[-1] != 3:
        raise ValueError("normal_modes must have shape (nmodes, natom, 3).")
    if cartesian_values.shape[-2:] != normal_modes.shape[1:]:
        raise ValueError(
            "cartesian_values trailing shape must match normal_modes atom axes: "
            f"{cartesian_values.shape[-2:]} != {normal_modes.shape[1:]}."
        )
    return np.einsum("...Ax,mAx->...m", cartesian_values, normal_modes, optimize=True)


def vibronic_couplings_from_derivative_couplings(
    reference_energies,
    normal_modes=None,
    derivative_couplings=None,
    mode_derivative_couplings=None,
):
    """Convert derivative couplings into off-diagonal linear coefficients.

    Provide either Cartesian ``derivative_couplings`` with shape
    ``(nstates, nstates, natom, 3)`` plus ``normal_modes``, or already
    projected ``mode_derivative_couplings`` with shape
    ``(nstates, nstates, nmodes)``.  Diagonal entries are returned as zero
    because they are supplied by projected state gradients in :func:`build_lvc`.
    """

    reference_energies = _as_1d(reference_energies, "reference_energies")
    nstates = reference_energies.size

    if mode_derivative_couplings is None:
        if derivative_couplings is None:
            raise ValueError(
                "Provide derivative_couplings or mode_derivative_couplings "
                "to compute vibronic couplings."
            )
        if normal_modes is None:
            raise ValueError("normal_modes are required with Cartesian derivative_couplings.")
        mode_derivative_couplings = project_cartesian_to_modes(derivative_couplings, normal_modes)
    else:
        mode_derivative_couplings = np.asarray(mode_derivative_couplings, dtype=float)

    if mode_derivative_couplings.ndim != 3 or mode_derivative_couplings.shape[:2] != (
        nstates,
        nstates,
    ):
        raise ValueError(
            "mode_derivative_couplings must have shape "
            f"({nstates}, {nstates}, nmodes), got {mode_derivative_couplings.shape}."
        )

    nmodes = mode_derivative_couplings.shape[2]
    couplings = np.zeros((nstates, nstates, nmodes), dtype=float)
    for i in range(nstates):
        for j in range(i + 1, nstates):
            value = (reference_energies[j] - reference_energies[i]) * mode_derivative_couplings[i, j]
            couplings[i, j] = value
            couplings[j, i] = value
    return couplings


def mode_derivative_couplings_from_overlaps(overlaps_minus, overlaps_plus, step):
    """Estimate mode derivative couplings from wavefunction overlaps.

    The expected input shape is ``(nmodes, nstates, nstates)`` for overlaps
    between reference states and states displaced by ``-step``/``+step`` along
    each mode.  The returned array has shape ``(nstates, nstates, nmodes)``.
    """

    overlaps_minus = np.asarray(overlaps_minus, dtype=float)
    overlaps_plus = np.asarray(overlaps_plus, dtype=float)
    if overlaps_minus.shape != overlaps_plus.shape:
        raise ValueError("overlaps_minus and overlaps_plus must have identical shapes.")
    if overlaps_plus.ndim != 3:
        raise ValueError("overlaps must have shape (nmodes, nstates, nstates).")
    if step <= 0:
        raise ValueError("step must be positive.")

    dqm = (overlaps_plus - overlaps_minus) / (2.0 * step)
    return np.moveaxis(dqm, 0, -1)


def load_sharc_lvc_template(
    path,
    mode_frequencies=None,
    normal_modes=None,
    reference_geometry=None,
):
    """Load a classic SHARC ``LVC.template`` file as an :class:`LVC` model.

    The reader imports the ``epsilon``, ``kappa``, and ``lambda`` sections.  It
    ignores SOC, dipole, quadratic, and multipolar-fit sections.
    """

    text = Path(path).read_text()
    return lvc_from_sharc_template(
        text,
        mode_frequencies=mode_frequencies,
        normal_modes=normal_modes,
        reference_geometry=reference_geometry,
    )


def lvc_from_sharc_template(
    text,
    mode_frequencies=None,
    normal_modes=None,
    reference_geometry=None,
):
    """Parse SHARC ``LVC.template`` text into an :class:`LVC` instance."""

    sections = _parse_sharc_lvc_sections(text)
    epsilon_rows = sections.get("epsilon", [])
    if not epsilon_rows:
        raise ValueError("SHARC LVC template is missing an epsilon section.")

    state_keys = [(int(row[0]), int(row[1])) for row in epsilon_rows]
    state_index = {key: idx for idx, key in enumerate(state_keys)}
    energies = np.array([float(row[2]) for row in epsilon_rows], dtype=float)

    mode_ids = _sharc_mode_ids(sections)
    mode_index = {mode_id: idx for idx, mode_id in enumerate(mode_ids)}
    frequencies = _select_mode_frequencies(mode_frequencies, mode_ids)
    couplings = np.zeros((len(state_keys), len(state_keys), len(mode_ids)), dtype=float)

    for row in sections.get("kappa", []):
        key = (int(row[0]), int(row[1]))
        state = _require_sharc_state(key, state_index)
        mode = mode_index[int(row[2])]
        couplings[state, state, mode] = float(row[3])

    for row in sections.get("lambda", []):
        left_key = (int(row[0]), int(row[1]))
        right_key = (int(row[0]), int(row[2]))
        left = _require_sharc_state(left_key, state_index)
        right = _require_sharc_state(right_key, state_index)
        mode = mode_index[int(row[3])]
        value = float(row[4])
        couplings[left, right, mode] = value
        couplings[right, left, mode] = value

    return LVC(
        reference_energies=energies,
        mode_frequencies=frequencies,
        couplings=couplings,
        normal_modes=normal_modes,
        mode_ids=mode_ids,
        reference_geometry=reference_geometry,
    )


def compare_lvc_to_sharc(model, sharc_template, atol=1e-10, rtol=1e-8):
    """Compare a PyQED :class:`LVC` model against a SHARC ``LVC.template``.

    ``sharc_template`` may be an :class:`LVC` instance, a path, or raw template
    text.  The returned dictionary contains maximum absolute differences and a
    boolean ``passed`` flag.
    """

    if isinstance(sharc_template, LVC):
        reference = sharc_template
    else:
        template_text = str(sharc_template)
        if "\n" in template_text:
            reference = lvc_from_sharc_template(template_text, mode_frequencies=model.mode_frequencies)
        else:
            reference = load_sharc_lvc_template(template_text, mode_frequencies=model.mode_frequencies)

    energies = np.asarray(model.reference_energies, dtype=float)
    ref_energies = np.asarray(reference.reference_energies, dtype=float)
    couplings = np.asarray(model.vibronic_couplings(), dtype=float)
    ref_couplings = np.asarray(reference.vibronic_couplings(), dtype=float)

    if energies.shape != ref_energies.shape:
        raise ValueError(f"Energy shapes differ: {energies.shape} != {ref_energies.shape}.")
    if couplings.shape != ref_couplings.shape:
        raise ValueError(f"Coupling shapes differ: {couplings.shape} != {ref_couplings.shape}.")
    if model.mode_ids is not None and reference.mode_ids is not None:
        if not np.array_equal(model.mode_ids, reference.mode_ids):
            raise ValueError("Mode ids differ between the PyQED and SHARC models.")

    energy_error = energies - ref_energies
    coupling_error = couplings - ref_couplings
    max_energy_error = float(np.max(np.abs(energy_error))) if energy_error.size else 0.0
    max_coupling_error = float(np.max(np.abs(coupling_error))) if coupling_error.size else 0.0

    return {
        "passed": bool(
            np.allclose(energies, ref_energies, atol=atol, rtol=rtol)
            and np.allclose(couplings, ref_couplings, atol=atol, rtol=rtol)
        ),
        "max_energy_error": max_energy_error,
        "max_coupling_error": max_coupling_error,
        "energy_error": energy_error,
        "coupling_error": coupling_error,
    }


def _parse_sharc_lvc_sections(text):
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)

    sections = {}
    i = 0
    known = {"epsilon", "kappa", "lambda"}
    while i < len(lines):
        key = lines[i].lower()
        if key not in known:
            i += 1
            continue
        if i + 1 >= len(lines):
            raise ValueError(f"SHARC section {key!r} is missing its row count.")
        try:
            nrows = int(lines[i + 1].split()[0])
        except ValueError as exc:
            raise ValueError(f"SHARC section {key!r} has a non-integer row count.") from exc
        start = i + 2
        stop = start + nrows
        if stop > len(lines):
            raise ValueError(f"SHARC section {key!r} declares {nrows} rows but is truncated.")
        sections[key] = [lines[j].split() for j in range(start, stop)]
        i = stop
    return sections


def _sharc_mode_ids(sections):
    ids = []
    for row in sections.get("kappa", []):
        ids.append(int(row[2]))
    for row in sections.get("lambda", []):
        ids.append(int(row[3]))
    return np.array(sorted(set(ids)), dtype=int)


def _select_mode_frequencies(mode_frequencies, mode_ids):
    if mode_frequencies is None:
        return np.zeros(len(mode_ids), dtype=float)

    mode_frequencies = _as_1d(mode_frequencies, "mode_frequencies")
    if mode_frequencies.size == len(mode_ids):
        return mode_frequencies
    if mode_ids.size and mode_frequencies.size >= int(np.max(mode_ids)):
        return mode_frequencies[mode_ids - 1]
    raise ValueError(
        "mode_frequencies must either match the number of SHARC modes or contain "
        "all 1-based SHARC mode ids."
    )


def _require_sharc_state(key, state_index):
    try:
        return state_index[key]
    except KeyError as exc:
        raise ValueError(f"SHARC coupling references state {key}, absent from epsilon.") from exc


def _as_1d(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}.")
    return arr


__all__ = [
    "LVC",
    "build_linear_vibronic_model",
    "build_lvc",
    "compare_lvc_to_sharc",
    "load_sharc_lvc_template",
    "lvc_from_sharc_template",
    "mode_derivative_couplings_from_overlaps",
    "project_cartesian_to_modes",
    "vibronic_couplings_from_derivative_couplings",
]
