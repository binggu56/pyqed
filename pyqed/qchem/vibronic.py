"""Ab-initio ingredients for linear vibronic-coupling Hamiltonians.

The central objects are :class:`LVC` and :class:`QVC`.  Their electronic-mode
coefficients are stored as ``linear_couplings`` and ``quadratic_couplings``:

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
    E
        Electronic energies at the reference geometry, shape ``(nstates,)``.
    omega
        Harmonic frequencies for the selected normal modes, shape ``(nmodes,)``.
    linear_couplings
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

    E: np.ndarray
    omega: np.ndarray
    linear_couplings: np.ndarray
    normal_modes: np.ndarray | None = None
    mode_ids: np.ndarray | None = None
    reference_geometry: np.ndarray | None = None

    @classmethod
    def from_casci(
        cls,
        mc,
        modes,
        omega,
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
            E=energies,
            omega=omega,
            linear_couplings=f,
            normal_modes=modes,
            mode_ids=mode_ids,
            reference_geometry=reference_geometry,
        )
        if return_quadratic:
            return model, g
        return model

    def __post_init__(self):
        object.__setattr__(self, "E", _as_1d(self.E, "E"))
        object.__setattr__(self, "omega", _as_1d(self.omega, "omega"))
        object.__setattr__(
            self,
            "linear_couplings",
            np.asarray(self.linear_couplings, dtype=float),
        )
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
        return int(self.E.size)

    @property
    def nmodes(self) -> int:
        return int(self.omega.size)

    @property
    def state_forces(self) -> np.ndarray:
        """Return diagonal force coefficients, ``-dE_state/dQ_mode``."""

        return -np.diagonal(self.linear_couplings, axis1=0, axis2=1).T

    def __call__(self, q: ArrayLike) -> np.ndarray:
        """Evaluate the electronic Hamiltonian without the common harmonic term."""

        return self.electronic_hamiltonian(q)

    def electronic_hamiltonian(self, q: ArrayLike, include_harmonic=False) -> np.ndarray:
        """Evaluate the electronic vibronic matrix at normal coordinates ``q``.

        ``include_harmonic=True`` adds the common harmonic nuclear energy
        ``0.5 * sum_m omega_m * q_m**2`` to every diagonal state.
        """

        q = _as_1d(q, "q")
        if q.size != self.nmodes:
            raise ValueError(f"q must have shape ({self.nmodes},), got {q.shape}.")

        h = np.diag(self.E).astype(float)
        h += np.einsum(
            "abm,m->ab", self.linear_couplings, q, optimize=True
        )
        if include_harmonic:
            h[np.diag_indices(self.nstates)] += 0.5 * np.dot(
                self.omega, q**2
            )
        return h

    def adiabatic_energies(self, q: ArrayLike, include_harmonic=False) -> np.ndarray:
        """Return eigenvalues of the vibronic matrix at ``q``."""

        return np.linalg.eigvalsh(self.electronic_hamiltonian(q, include_harmonic=include_harmonic))

    def _validate_shapes(self):
        nstates = self.E.size
        nmodes = self.omega.size
        if self.linear_couplings.shape != (nstates, nstates, nmodes):
            raise ValueError(
                "linear_couplings must have shape "
                f"({nstates}, {nstates}, {nmodes}), "
                f"got {self.linear_couplings.shape}."
            )
        if not np.allclose(
            self.linear_couplings,
            self.linear_couplings.swapaxes(0, 1),
        ):
            raise ValueError(
                "linear_couplings must be symmetric in the state indices."
            )
        if self.normal_modes is not None and self.normal_modes.shape[0] != nmodes:
            raise ValueError(
                f"normal_modes must have {nmodes} modes in axis 0, got {self.normal_modes.shape}."
            )
        if self.mode_ids is not None:
            if self.mode_ids.shape != (nmodes,):
                raise ValueError(f"mode_ids must have shape ({nmodes},), got {self.mode_ids.shape}.")
            if len(set(self.mode_ids.tolist())) != nmodes:
                raise ValueError("mode_ids must be unique.")

    def TDDMRG(
        self,
        nbas=10,
        D=40,
        *,
        basis="fock",
        dvrs=None,
        domains=None,
        include_harmonic=True,
        step=1.0,
        potential_tol=0.0,
        mpo_tol=1.0e-12,
    ):
        """Return a ``TDMPS`` driver for this vibronic Hamiltonian.

        The default Fock representation uses dimensionless normal coordinates.
        For a sine-DVR representation, set ``basis="dvr"`` and provide
        ``domains`` as one ``(minimum, maximum)`` pair per mode.  ``nbas`` then
        controls the number of DVR points.  Alternatively, pass explicit
        ``dvrs`` for custom coordinate bases.
        """

        from pyqed.mps.lvc import (
            dvr_potential_mpo,
            fock_hamiltonian_mpo,
            full_hamiltonian_mpo,
        )
        from pyqed.mps.tdmps import TDMPS

        basis = str(basis).lower()
        if basis == "fock":
            hamiltonian = fock_hamiltonian_mpo(
                self,
                nbas=nbas,
                include_harmonic=include_harmonic,
                tol=mpo_tol,
            )
            driver = TDMPS(hamiltonian, D=D)
            driver.model = self
            driver.basis = "fock"
            driver.nbas = np.broadcast_to(nbas, (self.nmodes,)).astype(int)
            return driver
        if basis != "dvr":
            raise ValueError("basis must be 'fock' or 'dvr'.")
        if dvrs is not None and domains is not None:
            raise ValueError("Provide either dvrs or domains, not both.")
        if dvrs is None:
            if domains is None:
                raise ValueError(
                    "domains or explicit dvrs must be provided when "
                    "basis='dvr'."
                )
            from pyqed.dvr.dvr_1d import SineDVR

            domains = np.asarray(domains, dtype=float)
            if domains.shape == (2,):
                domains = np.broadcast_to(domains, (self.nmodes, 2))
            if domains.shape != (self.nmodes, 2):
                raise ValueError(
                    "domains must have shape "
                    f"({self.nmodes}, 2), got {domains.shape}."
                )
            if np.any(~np.isfinite(domains)) or np.any(
                domains[:, 1] <= domains[:, 0]
            ):
                raise ValueError(
                    "Every DVR domain must contain finite increasing bounds."
                )
            counts = np.broadcast_to(nbas, (self.nmodes,)).astype(int)
            if np.any(counts <= 0):
                raise ValueError("Every DVR basis size must be positive.")
            if np.any(self.omega <= 0.0):
                raise ValueError(
                    "Automatic dimensionless normal-mode DVRs require "
                    "positive omega."
                )
            dvrs = [
                SineDVR(
                    xmin=lower,
                    xmax=upper,
                    npts=count,
                    mass=1.0 / frequency,
                )
                for (lower, upper), count, frequency in zip(
                    domains, counts, self.omega
                )
            ]

        dvrs = list(dvrs)
        if len(dvrs) != self.nmodes:
            raise ValueError(
                f"Expected {self.nmodes} DVRs, got {len(dvrs)}."
            )

        grids = []
        for mode, dvr in enumerate(dvrs):
            if not hasattr(dvr, "x"):
                raise TypeError(
                    f"DVR {mode} does not expose a coordinate grid 'x'."
                )
            grids.append(np.asarray(dvr.x, dtype=float))

        potential = dvr_potential_mpo(
            lambda q: self.electronic_hamiltonian(
                q, include_harmonic=include_harmonic
            ),
            grids,
            step=step,
            tol=potential_tol,
            mpo_tol=mpo_tol,
        )
        hamiltonian = full_hamiltonian_mpo(potential, dvrs)
        driver = TDMPS(hamiltonian, D=D)
        driver.model = self
        driver.basis = "dvr"
        driver.potential = potential
        driver.dvrs = dvrs
        driver.grids = grids
        driver.nbas = np.asarray([grid.size for grid in grids], dtype=int)
        return driver


@dataclass(frozen=True, kw_only=True)
class QVC(LVC):
    r"""Quadratic vibronic-coupling model.

    ``quadratic_couplings[a, b, m, n]`` stores the electronic-space Hessian

    .. math::

        B_{ab,mn}
        =
        \left.
        \frac{\partial^2 H_{ab}}
        {\partial Q_m\partial Q_n}
        \right|_{\mathbf Q=0}.

    The electronic Hamiltonian is evaluated using

    .. math::

        \mathbf H(\mathbf Q)
        =
        \mathbf H_0
        + \sum_m \mathbf A_m Q_m
        + \frac{1}{2}\sum_{mn}\mathbf B_{mn}Q_mQ_n.

    Consequently, mixed-mode entries may be supplied symmetrically as
    ``B[:, :, m, n] == B[:, :, n, m]`` without double-counting.
    """

    quadratic_couplings: np.ndarray

    @classmethod
    def from_casci(
        cls,
        mc,
        modes,
        omega,
        state_ids=None,
        mode_ids=None,
        reference_geometry=None,
    ) -> "QVC":
        """Build a ``QVC`` model from CASCI first and second derivatives."""

        linear, quadratic = LVC.from_casci(
            mc,
            modes=modes,
            omega=omega,
            state_ids=state_ids,
            mode_ids=mode_ids,
            reference_geometry=reference_geometry,
            return_quadratic=True,
        )
        return cls.from_lvc(linear, quadratic)

    def __post_init__(self):
        super().__post_init__()
        quadratic = np.asarray(self.quadratic_couplings, dtype=float)
        object.__setattr__(self, "quadratic_couplings", quadratic)

        expected = (self.nstates, self.nstates, self.nmodes, self.nmodes)
        if quadratic.shape != expected:
            raise ValueError(
                f"quadratic_couplings must have shape {expected}, "
                f"got {quadratic.shape}."
            )
        if not np.allclose(quadratic, quadratic.swapaxes(0, 1)):
            raise ValueError(
                "quadratic_couplings must be symmetric in the state indices."
            )
        if not np.allclose(quadratic, quadratic.swapaxes(2, 3)):
            raise ValueError(
                "quadratic_couplings must be symmetric in the mode indices."
            )

    @classmethod
    def from_lvc(cls, model: LVC, quadratic_couplings) -> "QVC":
        """Promote an :class:`LVC` model by adding its electronic Hessian."""

        return cls(
            E=model.E,
            omega=model.omega,
            linear_couplings=model.linear_couplings,
            normal_modes=model.normal_modes,
            mode_ids=model.mode_ids,
            reference_geometry=model.reference_geometry,
            quadratic_couplings=quadratic_couplings,
        )

    def electronic_hamiltonian(
        self,
        q: ArrayLike,
        include_harmonic=False,
    ) -> np.ndarray:
        """Evaluate the quadratic electronic vibronic Hamiltonian."""

        q = _as_1d(q, "q")
        if q.size != self.nmodes:
            raise ValueError(f"q must have shape ({self.nmodes},), got {q.shape}.")

        h = super().electronic_hamiltonian(q, include_harmonic=False)
        h += 0.5 * np.einsum(
            "abmn,m,n->ab",
            self.quadratic_couplings,
            q,
            q,
            optimize=True,
        )
        if include_harmonic:
            h[np.diag_indices(self.nstates)] += 0.5 * np.dot(
                self.omega, q**2
            )
        return h


def build_lvc(
    E,
    omega,
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

    E = _as_1d(E, "E")
    omega = _as_1d(omega, "omega")
    normal_modes = np.asarray(normal_modes, dtype=float)
    state_gradients = np.asarray(state_gradients, dtype=float)

    diagonal = project_cartesian_to_modes(state_gradients, normal_modes)
    if diagonal.shape != (E.size, omega.size):
        raise ValueError(
            "Projected state gradients must have shape "
            f"({E.size}, {omega.size}), got {diagonal.shape}."
        )

    if vibronic_couplings is None:
        couplings = vibronic_couplings_from_derivative_couplings(
            E,
            normal_modes=normal_modes,
            derivative_couplings=derivative_couplings,
            mode_derivative_couplings=mode_derivative_couplings,
        )
    else:
        couplings = np.asarray(vibronic_couplings, dtype=float).copy()

    for state in range(E.size):
        couplings[state, state] = diagonal[state]

    return LVC(
        E=E,
        omega=omega,
        linear_couplings=couplings,
        normal_modes=normal_modes,
        mode_ids=mode_ids,
        reference_geometry=reference_geometry,
    )

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
    E,
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

    E = _as_1d(E, "E")
    nstates = E.size

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
            value = (E[j] - E[i]) * mode_derivative_couplings[i, j]
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
    omega=None,
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
        omega=omega,
        normal_modes=normal_modes,
        reference_geometry=reference_geometry,
    )


def lvc_from_sharc_template(
    text,
    omega=None,
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
    frequencies = _select_omega(omega, mode_ids)
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
        E=energies,
        omega=frequencies,
        linear_couplings=couplings,
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
            reference = lvc_from_sharc_template(
                template_text, omega=model.omega
            )
        else:
            reference = load_sharc_lvc_template(
                template_text, omega=model.omega
            )

    energies = np.asarray(model.E, dtype=float)
    ref_energies = np.asarray(reference.E, dtype=float)
    couplings = np.asarray(model.linear_couplings, dtype=float)
    ref_couplings = np.asarray(reference.linear_couplings, dtype=float)

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


def _select_omega(omega, mode_ids):
    if omega is None:
        return np.zeros(len(mode_ids), dtype=float)

    omega = _as_1d(omega, "omega")
    if omega.size == len(mode_ids):
        return omega
    if mode_ids.size and omega.size >= int(np.max(mode_ids)):
        return omega[mode_ids - 1]
    raise ValueError(
        "omega must either match the number of SHARC modes or contain all "
        "1-based SHARC mode ids."
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
    "QVC",
    "build_lvc",
    "compare_lvc_to_sharc",
    "load_sharc_lvc_template",
    "lvc_from_sharc_template",
    "mode_derivative_couplings_from_overlaps",
    "project_cartesian_to_modes",
    "vibronic_couplings_from_derivative_couplings",
]
