"""Finite-basis MCTDHF propagation with moving active spatial orbitals."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature

import numpy as np
from scipy.linalg import eigh, expm
from scipy.sparse.linalg import LinearOperator, eigsh, expm_multiply

from pyqed.qchem.ci.fci import CI_H, SlaterCondon, get_fci_combos
from pyqed.qchem.mcscf.direct_ci import build_direct_connectivity
from pyqed.qchem.mcscf.casci import (
    _ci_to_spin_string_matrix,
    _spin_string_links,
    _unique_rows_first,
    make_rdm1,
    make_rdm2,
    transform_spatial_eri_to_mo,
)
from pyqed.qchem.tdcasci import _axis_to_index, _field_vector, _window_values


def _electron_pair(nelec, spin=0):
    if isinstance(nelec, (tuple, list, np.ndarray)):
        if len(nelec) != 2:
            raise ValueError("nelec must be an integer or a length-2 pair.")
        na, nb = int(nelec[0]), int(nelec[1])
    else:
        ne = int(nelec)
        spin = int(spin)
        if (ne + spin) % 2:
            raise ValueError("nelec and spin give noninteger alpha/beta counts.")
        na = (ne + spin) // 2
        nb = ne - na
    if na < 0 or nb < 0:
        raise ValueError("electron counts must be nonnegative.")
    return na, nb


def _metric_powers(overlap, thresh=1.0e-12):
    eig, vec = eigh(0.5 * (overlap + overlap.conj().T))
    if np.min(eig) < thresh:
        raise np.linalg.LinAlgError("AO overlap matrix is singular.")
    s_half = (vec * np.sqrt(eig)) @ vec.conj().T
    s_inv_half = (vec * (1.0 / np.sqrt(eig))) @ vec.conj().T
    return s_half, s_inv_half


def _string_transform_matrix(orbital_transform, occ_strings, dtype):
    occ = [np.flatnonzero(row) for row in np.asarray(occ_strings, dtype=np.int8)]
    out = np.empty((len(occ), len(occ)), dtype=dtype)
    for i, bra in enumerate(occ):
        for j, ket in enumerate(occ):
            if len(bra) == 0:
                out[i, j] = 1.0
            elif len(bra) == 1:
                out[i, j] = orbital_transform[bra[0], ket[0]]
            else:
                out[i, j] = np.linalg.det(orbital_transform[np.ix_(bra, ket)])
    return out


def _call_with_supported_keywords(method, /, **kwargs):
    try:
        sig = signature(method)
    except (TypeError, ValueError):
        return method(**kwargs)

    params = sig.parameters
    if any(param.kind == Parameter.VAR_KEYWORD for param in params.values()):
        return method(**kwargs)

    filtered = {
        name: value
        for name, value in kwargs.items()
        if name in params
        and params[name].kind
        in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
    }
    return method(**filtered)


@dataclass
class MCTDHFTrajectory:
    """Stored MCTDHF trajectory."""

    times: np.ndarray
    ci: np.ndarray | None
    orbitals: np.ndarray | None
    energies: np.ndarray
    electronic_energies: np.ndarray
    norms: np.ndarray
    orbital_errors: np.ndarray
    active_gauge_errors: np.ndarray
    core_overlap_errors: np.ndarray
    natural_occupations: np.ndarray
    natural_occupation_trace_errors: np.ndarray
    autocorrelation: np.ndarray
    dipoles: np.ndarray
    fields: np.ndarray

    @property
    def final_time(self):
        """Return the last saved propagation time."""
        return self.times[-1]

    @property
    def final_ci(self):
        """Return the last saved CI/provider state."""
        if self.ci is None:
            raise ValueError("CI states were not stored for this trajectory.")
        return self.ci[-1]

    @property
    def final_orbitals(self):
        """Return the last saved active orbitals."""
        if self.orbitals is None:
            raise ValueError("Orbitals were not stored for this trajectory.")
        return self.orbitals[-1]

    def dipole_spectrum(self, axis="z", window="hann", subtract_mean=True):
        """Return angular frequencies and dipole power spectrum."""
        idx = _axis_to_index(axis)
        signal = np.asarray(self.dipoles[:, idx], dtype=float)
        if subtract_mean:
            signal = signal - np.mean(signal)
        dt = float(self.times[1] - self.times[0]) if self.times.size > 1 else 1.0
        win = _window_values(window, signal.size)
        response = np.fft.rfft(signal * win)
        omega = 2.0 * np.pi * np.fft.rfftfreq(signal.size, d=dt)
        return omega, np.abs(response) ** 2

    def autocorrelation_spectrum(self, window="hann"):
        """Return angular frequencies and autocorrelation spectrum."""
        signal = np.asarray(self.autocorrelation, dtype=complex)
        dt = float(self.times[1] - self.times[0]) if self.times.size > 1 else 1.0
        win = _window_values(window, signal.size)
        response = np.fft.fft(signal * win)
        omega = 2.0 * np.pi * np.fft.fftfreq(signal.size, d=dt)
        order = np.argsort(omega)
        return omega[order], np.abs(response[order])


class DenseCIDensityProvider:
    """Density contractions for dense determinant-space CI coefficients."""

    def __init__(self, driver):
        self.driver = driver

    def make_rdm1(self, ci):
        driver = self.driver
        return make_rdm1(np.asarray(ci, dtype=complex), driver.binary, driver.SC1)

    def make_rdm2(self, ci):
        driver = self.driver
        return make_rdm2(np.asarray(ci, dtype=complex), driver.binary, driver.SC1, driver.SC2)

    def contract_rdm2_eri_full(self, ci, eri_full):
        """
        Contract the spin-traced 2-RDM with mixed AO/active ERIs.

        Returns ``K[x, o] = sum_{q r s} Gamma[o, q, r, s] eri[x, q, r, s]``
        without materializing the full ``Gamma`` tensor.
        """
        driver = self.driver
        ci = np.asarray(ci, dtype=complex)
        eri_full = np.asarray(eri_full, dtype=complex)
        if eri_full.shape != (driver.nao, driver.norb, driver.norb, driver.norb):
            raise ValueError(
                "eri_full must have shape (nao, norb, norb, norb); "
                f"got {eri_full.shape}."
            )

        alpha, beta, coeff = _ci_to_spin_string_matrix(ci, driver.binary)
        alpha_one, alpha_two = _spin_string_links(alpha)
        beta_one, beta_two = _spin_string_links(beta)
        cbra = coeff.conj()
        out = np.zeros((driver.nao, driver.norb), dtype=np.result_type(ci, eri_full, complex))

        def add_same_spin(links, overlap):
            p, q, r, s, bra, ket, phase = links
            if len(p) == 0:
                return
            coeff_link = phase * overlap[bra, ket]
            values = coeff_link[:, None] * eri_full[:, q, r, s].T
            np.add.at(out.T, p, values)

        def add_opposite_spin(first_links, second_links, left, right):
            pa, qa, bra_a, ket_a, phase_a = first_links
            rb, sb, bra_b, ket_b, phase_b = second_links
            if len(pa) == 0 or len(rb) == 0:
                return
            for la in range(len(pa)):
                coeff_link = (
                    phase_a[la]
                    * phase_b
                    * left[bra_a[la], bra_b]
                    * right[ket_a[la], ket_b]
                )
                out[:, pa[la]] += np.einsum(
                    "l,xl->x",
                    coeff_link,
                    eri_full[:, qa[la], rb, sb],
                    optimize=True,
                )

        add_same_spin(alpha_two, cbra @ coeff.T)
        add_same_spin(beta_two, cbra.T @ coeff)
        add_opposite_spin(alpha_one, beta_one, cbra, coeff)
        add_opposite_spin(beta_one, alpha_one, cbra.T, coeff.T)
        return out

    def contract_rdm2_eri_full_reference(self, ci, eri_full):
        """Reference contraction that explicitly materializes the active RDM2."""
        dm2 = self.make_rdm2(ci)
        return np.einsum("oqrs,xqrs->xo", dm2, eri_full, optimize=True)


class RDM12DensityProvider:
    """
    Density provider backed by an object exposing active-space RDM1/RDM2 APIs.

    This is the adapter for external wavefunction solvers.  The wrapped object
    is expected to provide spatial active-space density matrices with the same
    conventions as CASCI, e.g. DMRG ``make_rdm1(..., spatial=True)`` and
    ``make_rdm2(..., spatial=True)``.
    """

    def __init__(
        self,
        backend,
        *,
        state_id=0,
        spatial=True,
        with_core=False,
        rdm1_method="make_rdm1",
        rdm2_method="make_rdm2",
    ):
        self.backend = backend
        self.state_id = int(state_id)
        self.spatial = bool(spatial)
        self.with_core = bool(with_core)
        self.rdm1_method = rdm1_method
        self.rdm2_method = rdm2_method
        self.driver = None

    _optional_backend_hooks = frozenset(
        {
            "ci_vector",
            "electronic_energy",
            "kick_ci",
            "norm",
            "overlap",
            "propagate_ci",
            "rotate_ci_for_orbital_rotation",
            "sigma_vector",
        }
    )

    def bind(self, driver):
        self.driver = driver
        return self

    def __getattr__(self, name):
        if name not in self._optional_backend_hooks:
            raise AttributeError(name)
        method = getattr(self.backend, name, None)
        if method is None or not callable(method):
            raise AttributeError(name)

        def wrapper(**kwargs):
            call_kwargs = {
                "state_id": self.state_id,
                "spatial": self.spatial,
                "with_core": self.with_core,
                "driver": self.driver,
            }
            call_kwargs.update(kwargs)
            return _call_with_supported_keywords(
                method,
                **call_kwargs,
            )

        return wrapper

    def _call_density_method(self, method_name, ci=None):
        method = getattr(self.backend, method_name)
        try:
            return _call_with_supported_keywords(
                method,
                ci=ci,
                state=ci,
                state_id=self.state_id,
                spatial=self.spatial,
                with_core=self.with_core,
                driver=self.driver,
            )
        except TypeError as exc:
            if "unexpected keyword" not in str(exc) and "positional" not in str(exc):
                raise
            return method(self.state_id, self.spatial, self.with_core)

    def _validate_shape(self, array, expected_shape, name):
        arr = np.asarray(array, dtype=complex)
        if self.driver is not None and arr.shape != expected_shape:
            raise ValueError(f"{name} has shape {arr.shape}; expected {expected_shape}.")
        return arr

    def make_rdm1(self, ci=None):
        dm1 = self._call_density_method(self.rdm1_method, ci=ci)
        if self.driver is None:
            return np.asarray(dm1, dtype=complex)
        return self._validate_shape(dm1, (self.driver.norb, self.driver.norb), "RDM1")

    def make_rdm2(self, ci=None):
        dm2 = self._call_density_method(self.rdm2_method, ci=ci)
        if self.driver is None:
            return np.asarray(dm2, dtype=complex)
        shape = (self.driver.norb, self.driver.norb, self.driver.norb, self.driver.norb)
        return self._validate_shape(dm2, shape, "RDM2")

    def contract_rdm2_eri_full(self, ci, eri_full):
        dm2 = self.make_rdm2(ci)
        eri_full = np.asarray(eri_full, dtype=complex)
        return np.einsum("oqrs,xqrs->xo", dm2, eri_full, optimize=True)

    def contract_rdm2_eri_full_reference(self, ci, eri_full):
        return self.contract_rdm2_eri_full(ci, eri_full)


class DMRGDensityProvider(RDM12DensityProvider):
    """Density provider for PyQED DMRG/QCDMRG objects."""

    def __init__(self, dmrg, *, state_id=0, spatial=True, with_core=False):
        super().__init__(
            dmrg,
            state_id=state_id,
            spatial=spatial,
            with_core=with_core,
        )


class MCTDHF:
    """
    Multiconfiguration time-dependent Hartree-Fock in a finite AO basis.

    The wavefunction is expanded in ``CAS(nelec, norb)`` while the ``norb``
    active spatial orbitals move in the parent AO basis.  The orbital equation
    uses the standard active-space 1-RDM inverse, spin-traced 2-RDM, and a
    virtual-space projector.  The CI propagation supports a dense reference
    backend and a matrix-free Krylov backend based on compact Slater-Condon
    connectivity.
    """

    def __init__(
        self,
        mf,
        norb=None,
        nelec=None,
        *,
        ncas=None,
        nelecas=None,
        spin=0,
        mo_coeff=None,
        active_orbitals=None,
        core_orbitals=None,
        initial_ci=None,
        field=None,
        interaction_ao=None,
        h1_ao=None,
        orbital_mode="mctdhf",
        orbital_integrator="midpoint",
        ci_backend="auto",
        dense_ci_threshold=512,
        density_provider=None,
        rdm_rcond=1.0e-10,
        use_cholesky=None,
    ):
        if norb is None:
            norb = ncas
        if nelec is None:
            nelec = nelecas
        if norb is None or nelec is None:
            raise ValueError("Set norb/ncas and nelec/nelecas.")
        if getattr(mf, "mo_coeff", None) is None:
            raise ValueError("Run HF before starting MCTDHF.")

        self.mf = mf
        self.norb = int(norb)
        self.ncas = self.norb
        self.na, self.nb = _electron_pair(nelec, spin=spin)
        self.nelec = (self.na, self.nb)
        self.active_orbitals = active_orbitals
        self.initial_ci = None if initial_ci is None else np.asarray(initial_ci, dtype=complex).reshape(-1)
        self.core_orbitals = None
        self.core_density_ao = None
        self.core_veff_ao = None
        self.e_core = 0.0
        self.field = field
        self.interaction_ao = interaction_ao
        self.h1_ao = h1_ao
        self.orbital_mode = str(orbital_mode).lower()
        self.orbital_integrator = str(orbital_integrator).lower()
        self.ci_backend = str(ci_backend).lower()
        self.dense_ci_threshold = int(dense_ci_threshold)
        self.density_provider = density_provider
        self.rdm_rcond = float(rdm_rcond)
        self.use_cholesky = use_cholesky

        self.overlap = np.asarray(mf.get_ovlp(), dtype=complex)
        self.s_half, self.s_inv_half = _metric_powers(self.overlap)
        self.nao = int(self.overlap.shape[0])
        self.e_nuc = float(mf.energy_nuc())
        if self.norb > self.nao:
            raise ValueError("norb must not exceed the AO basis dimension.")
        if self.norb < self.na or self.norb < self.nb:
            raise ValueError("norb must be at least max(nalpha, nbeta).")

        if core_orbitals is not None:
            core = np.asarray(core_orbitals, dtype=complex)
            if core.ndim != 2 or core.shape[0] != self.nao:
                raise ValueError("core_orbitals must have shape (nao, ncore).")
            if core.shape[1] > 0:
                self.core_orbitals = self.orthonormalize_orbitals(core)
                self.core_density_ao = 2.0 * self.core_orbitals @ self.core_orbitals.conj().T
                hcore = np.asarray(mf.get_hcore(), dtype=complex)
                corevhf = np.asarray(mf.get_veff(self.core_density_ao), dtype=complex)
                self.core_veff_ao = corevhf
                self.e_core = (
                    np.einsum("ij,ji->", self.core_density_ao, hcore, optimize=True).real
                    + 0.5
                    * np.einsum("ij,ji->", self.core_density_ao, corevhf, optimize=True).real
                )

        self.orbitals0 = self.orthonormalize_orbitals(
            self._initial_orbitals(mo_coeff=mo_coeff, active_orbitals=active_orbitals)
        )
        if self.core_orbitals is not None and self.core_active_overlap_error(self.orbitals0) > 1.0e-8:
            raise ValueError("core_orbitals must be S-orthogonal to the active orbitals.")
        occ = np.zeros((2, self.norb), dtype=np.int8)
        occ[0, : self.na] = 1
        occ[1, : self.nb] = 1
        self.binary = get_fci_combos(mo_occ=occ)
        self.SC1, self.SC2 = SlaterCondon(self.binary)
        self.ndet = int(self.binary.shape[0])
        self.alpha_strings = _unique_rows_first(self.binary[:, 0, :])
        self.beta_strings = _unique_rows_first(self.binary[:, 1, :])
        self.nalpha_strings = int(self.alpha_strings.shape[0])
        self.nbeta_strings = int(self.beta_strings.shape[0])
        self.direct_connectivity = build_direct_connectivity(self.binary)
        if self.initial_ci is not None and self.initial_ci.shape[0] != self.ndet:
            raise ValueError(
                f"initial_ci length {self.initial_ci.shape[0]} does not match ndet={self.ndet}."
            )

        if self.ci_backend not in {"auto", "dense", "krylov"}:
            raise ValueError("ci_backend must be 'auto', 'dense', or 'krylov'.")
        if self.orbital_integrator not in {"midpoint", "rk4"}:
            raise ValueError("orbital_integrator must be 'midpoint' or 'rk4'.")
        if self.density_provider is None:
            self.density_provider = DenseCIDensityProvider(self)
        elif hasattr(self.density_provider, "bind"):
            self.density_provider.bind(self)

    @classmethod
    def from_casci(cls, casci, *, state_id=0, **kwargs):
        """Create an MCTDHF driver seeded from a completed CASCI calculation."""
        if getattr(casci, "ci", None) is None:
            raise ValueError("Run CASCI before constructing MCTDHF from it.")
        if getattr(casci, "mo_cas", None) is None:
            raise ValueError("CASCI active orbitals are unavailable.")
        state_id = int(state_id)
        if state_id < 0 or state_id >= len(casci.ci):
            raise ValueError("state_id is out of range for casci.ci.")
        if "norb" in kwargs or "ncas" in kwargs:
            raise ValueError("Do not pass norb/ncas to from_casci(); they come from CASCI.")
        if "nelec" in kwargs or "nelecas" in kwargs:
            raise ValueError("Do not pass nelec/nelecas to from_casci(); they come from CASCI.")
        if "mo_coeff" in kwargs:
            raise ValueError("Do not pass mo_coeff to from_casci(); it uses casci.mo_cas.")
        if "initial_ci" in kwargs:
            raise ValueError("Do not pass initial_ci to from_casci(); it uses casci.ci[state_id].")
        return cls(
            casci.mf,
            norb=int(casci.ncas),
            nelec=getattr(casci, "nelecas_spin", getattr(casci, "nelecas")),
            mo_coeff=np.asarray(casci.mo_cas),
            core_orbitals=np.asarray(casci.mo_core) if int(getattr(casci, "ncore", 0)) else None,
            initial_ci=np.asarray(casci.ci[state_id], dtype=complex),
            **kwargs,
        )

    def _active_orbital_indices(self, active_orbitals, nmo):
        if active_orbitals is None:
            return tuple(range(self.norb))
        if isinstance(active_orbitals, slice):
            indices = tuple(range(int(nmo))[active_orbitals])
        elif np.isscalar(active_orbitals):
            start = int(active_orbitals)
            indices = tuple(range(start, start + self.norb))
        else:
            indices = tuple(int(i) for i in active_orbitals)
        if len(indices) != self.norb:
            raise ValueError(
                f"active_orbitals must contain exactly norb={self.norb} entries."
            )
        if len(set(indices)) != len(indices):
            raise ValueError("active_orbitals contains duplicate indices.")
        if min(indices) < 0 or max(indices) >= int(nmo):
            raise ValueError("active_orbitals contains an out-of-range MO index.")
        return indices

    def _initial_orbitals(self, mo_coeff=None, active_orbitals=None):
        coeff = self.mf.mo_coeff if mo_coeff is None else mo_coeff
        if isinstance(coeff, (tuple, list)):
            coeff = coeff[0]
        coeff = np.asarray(coeff, dtype=complex)
        if coeff.ndim != 2 or coeff.shape[0] != self.nao:
            raise ValueError("mo_coeff must have shape (nao, nmo).")
        indices = self._active_orbital_indices(active_orbitals, coeff.shape[1])
        self.active_orbitals = indices
        return coeff[:, indices]

    def orthonormalize_orbitals(self, orbitals):
        """Symmetrically S-orthonormalize AO orbital coefficients."""
        c = np.asarray(orbitals, dtype=complex)
        metric = c.conj().T @ self.overlap @ c
        eig, vec = eigh(0.5 * (metric + metric.conj().T))
        if np.min(eig) < 1.0e-12:
            raise np.linalg.LinAlgError("Active orbital overlap is singular.")
        inv_half = (vec * (1.0 / np.sqrt(eig))) @ vec.conj().T
        return c @ inv_half

    def orbital_error(self, orbitals):
        metric = orbitals.conj().T @ self.overlap @ orbitals
        return float(np.linalg.norm(metric - np.eye(self.norb)))

    def core_active_overlap_error(self, orbitals):
        if self.core_orbitals is None:
            return 0.0
        return float(np.linalg.norm(self.core_orbitals.conj().T @ self.overlap @ orbitals))

    def active_gauge_error(self, old_orbitals, new_orbitals):
        """Measure the anti-Hermitian active rotation between two frames."""
        sact = old_orbitals.conj().T @ self.overlap @ new_orbitals
        return float(np.linalg.norm(sact - sact.conj().T))

    def project_out_subspace(self, vectors, basis):
        """Project ``vectors`` out of the column span of ``basis``."""
        q = np.asarray(basis, dtype=complex)
        if q.size == 0:
            return vectors
        gram = q.conj().T @ q
        coeff = np.linalg.pinv(gram, rcond=self.rdm_rcond) @ (q.conj().T @ vectors)
        return vectors - q @ coeff

    def field_vector(self, time, field=None):
        return _field_vector(self.field if field is None else field, time)

    def _h1_source_value(self, time, h1_ao=None):
        source = self.h1_ao if h1_ao is None else h1_ao
        if source is None:
            return None
        return source(time) if callable(source) else source

    def _interaction_operator_ao(self):
        if self.interaction_ao is not None:
            return np.asarray(self.interaction_ao, dtype=complex)
        if hasattr(self.mf, "dipole"):
            return np.asarray(self.mf.dipole(basis="ao"), dtype=complex)
        return None

    def one_body_ao(self, time, field=None, h1_ao=None):
        h = np.asarray(self.mf.get_hcore(), dtype=complex)
        f = self.field_vector(time, field=field)
        op = self._interaction_operator_ao()
        if op is not None and np.any(f):
            h = h - np.einsum("x,xij->ij", f, op, optimize=True)
        extra = self._h1_source_value(time, h1_ao=h1_ao)
        if extra is not None:
            h = h + np.asarray(extra, dtype=complex)
        return 0.5 * (h + h.conj().T)

    def core_one_body_energy(self, time=0.0, field=None, h1_ao=None):
        if self.core_density_ao is None:
            return 0.0
        h_ao = self.one_body_ao(time, field=field, h1_ao=h1_ao)
        h0 = np.asarray(self.mf.get_hcore(), dtype=complex)
        return np.einsum("ij,ji->", self.core_density_ao, h_ao - h0, optimize=True)

    def effective_one_body_ao(self, time=0.0, field=None, h1_ao=None):
        h = self.one_body_ao(time, field=field, h1_ao=h1_ao)
        if self.core_veff_ao is not None:
            h = h + self.core_veff_ao
        return 0.5 * (h + h.conj().T)

    def active_integrals(self, orbitals, time=0.0, field=None, h1_ao=None):
        c = np.asarray(orbitals, dtype=complex)
        h_ao = self.effective_one_body_ao(time, field=field, h1_ao=h1_ao)
        h1 = c.conj().T @ h_ao @ c
        eri = transform_spatial_eri_to_mo(
            self.mf,
            c,
            use_cholesky=self.use_cholesky,
        )
        return 0.5 * (h1 + h1.conj().T), eri

    def _spin_block_integrals(self, h1, eri):
        same = eri - eri.swapaxes(1, 3)
        h1_blocks = np.asarray([h1, h1])
        h2_blocks = np.stack(
            (
                np.stack((same, eri)),
                np.stack((eri, same)),
            )
        )
        return h1_blocks, h2_blocks

    def one_body_ci_matrix_from_active(self, op_active):
        """Dense determinant-space matrix for a spin-independent active operator."""
        op = np.asarray(op_active, dtype=complex)
        if op.shape != (self.norb, self.norb):
            raise ValueError(f"op_active must have shape ({self.norb}, {self.norb}).")
        h1_blocks = np.asarray([op, op])
        h2_blocks = np.zeros(
            (2, 2, self.norb, self.norb, self.norb, self.norb),
            dtype=complex,
        )
        mat = CI_H(self.binary, h1_blocks, h2_blocks, self.SC1, self.SC2)
        return 0.5 * (mat + mat.conj().T)

    def one_body_ci_matrix_ao(self, operator_ao, orbitals=None):
        """Dense determinant-space matrix for an AO one-body operator."""
        if orbitals is None:
            orbitals = self.orbitals0
        c = np.asarray(orbitals, dtype=complex)
        op = np.asarray(operator_ao, dtype=complex)
        if op.ndim != 2 or op.shape != (self.nao, self.nao):
            raise ValueError(f"operator_ao must have shape ({self.nao}, {self.nao}).")
        op_active = c.conj().T @ op @ c
        return self.one_body_ci_matrix_from_active(0.5 * (op_active + op_active.conj().T))

    def kick_ci(self, ci, strength=1.0e-4, axis="x", orbitals=None):
        """Apply an impulsive one-body interaction to a CI state."""
        if orbitals is None:
            orbitals = self.orbitals0
        hook = self._provider_hook("kick_ci")
        if hook is not None:
            op = self._interaction_operator_ao()
            idx = _axis_to_index(axis)
            op_axis = None if op is None else np.asarray(op, dtype=complex)[idx]
            return _call_with_supported_keywords(
                hook,
                ci=ci,
                strength=float(strength),
                axis=axis,
                orbitals=orbitals,
                operator_ao=op_axis,
                driver=self,
            )
        if not self._is_dense_ci_vector(ci):
            raise NotImplementedError("External CI state requires density_provider.kick_ci(...).")
        op = self._interaction_operator_ao()
        if op is None:
            raise ValueError("No interaction operator is available for an impulsive kick.")
        mat = self.one_body_ci_matrix_ao(np.asarray(op, dtype=complex)[_axis_to_index(axis)], orbitals=orbitals)
        kicked = expm(-1j * float(strength) * mat) @ np.asarray(ci, dtype=complex)
        return kicked / np.linalg.norm(kicked)

    def _compact_integrals(self, h1, eri):
        same = eri - eri.swapaxes(1, 3)
        return h1, same, eri

    def _resolved_ci_backend(self, backend=None):
        key = self.ci_backend if backend is None else str(backend).lower()
        if key == "auto":
            return "dense" if self.ndet <= self.dense_ci_threshold else "krylov"
        if key not in {"dense", "krylov"}:
            raise ValueError("CI backend must be 'auto', 'dense', or 'krylov'.")
        return key

    def _provider_hook(self, name):
        method = getattr(self.density_provider, name, None)
        if method is None or not callable(method):
            return None
        owner = getattr(method, "__self__", None)
        if owner is self:
            return None
        return method

    def _call_provider_hook(self, name, /, **kwargs):
        method = self._provider_hook(name)
        if method is None:
            return None
        return _call_with_supported_keywords(method, driver=self, **kwargs)

    def _is_dense_ci_vector(self, ci):
        arr = np.asarray(ci)
        return arr.shape == (self.ndet,) and arr.dtype != object

    def ci_norm(self, ci):
        hook = self._provider_hook("norm")
        if hook is not None:
            return float(_call_with_supported_keywords(hook, ci=ci, driver=self).real)
        if not self._is_dense_ci_vector(ci):
            raise NotImplementedError(
                "External CI state requires density_provider.norm(ci=...)."
            )
        return float(np.vdot(np.asarray(ci, dtype=complex), np.asarray(ci, dtype=complex)).real)

    def state_overlap(self, bra, ket, bra_orbitals=None, ket_orbitals=None):
        """Return the many-electron overlap between two MCTDHF states."""
        hook = self._provider_hook("overlap")
        if hook is not None:
            return _call_with_supported_keywords(
                hook,
                bra=bra,
                ket=ket,
                bra_orbitals=bra_orbitals,
                ket_orbitals=ket_orbitals,
                driver=self,
            )
        if not (self._is_dense_ci_vector(bra) and self._is_dense_ci_vector(ket)):
            raise NotImplementedError(
                "External CI state requires density_provider.overlap(bra=..., ket=...)."
            )
        if bra_orbitals is None and ket_orbitals is None:
            return np.vdot(np.asarray(bra, dtype=complex), np.asarray(ket, dtype=complex))
        if bra_orbitals is None or ket_orbitals is None:
            raise ValueError("Set both bra_orbitals and ket_orbitals for orbital-frame overlap.")
        return self.dense_state_overlap(
            bra,
            ket,
            bra_orbitals=bra_orbitals,
            ket_orbitals=ket_orbitals,
        )

    def dense_state_overlap(self, bra, ket, *, bra_orbitals, ket_orbitals):
        """Dense determinant overlap including different active orbital frames."""
        sact = np.asarray(bra_orbitals, dtype=complex).conj().T @ self.overlap @ np.asarray(
            ket_orbitals,
            dtype=complex,
        )
        wa = _string_transform_matrix(sact, self.alpha_strings, complex)
        wb = _string_transform_matrix(sact, self.beta_strings, complex)
        cbra = self._ci_tensor(bra)
        cket = self._ci_tensor(ket)
        return np.einsum("ab,ac,bd,cd->", cbra.conj(), wa, wb, cket, optimize=True)

    def hamiltonian_matrix(self, orbitals=None, time=0.0, field=None, h1_ao=None):
        """Dense CAS Hamiltonian for the current active orbitals."""
        if orbitals is None:
            orbitals = self.orbitals0
        h1, eri = self.active_integrals(
            orbitals,
            time=time,
            field=field,
            h1_ao=h1_ao,
        )
        h1_blocks, h2_blocks = self._spin_block_integrals(h1, eri)
        h = CI_H(self.binary, h1_blocks, h2_blocks, self.SC1, self.SC2)
        return 0.5 * (h + h.conj().T)

    def _direct_ci_diagonal(self, h1, eri_same, eri_cross):
        occ_a = self.binary[:, 0, :]
        occ_b = self.binary[:, 1, :]
        hdiag = np.diag(h1)
        diag = np.einsum("Ip,p->I", occ_a + occ_b, hdiag, optimize=True)
        eri_ppqq_same = np.einsum("ppqq->pq", eri_same, optimize=True)
        eri_ppqq_cross = np.einsum("ppqq->pq", eri_cross, optimize=True)
        diag = diag + 0.5 * np.einsum("Ip,Iq,pq->I", occ_a, occ_a, eri_ppqq_same, optimize=True)
        diag = diag + 0.5 * np.einsum("Ip,Iq,pq->I", occ_b, occ_b, eri_ppqq_same, optimize=True)
        diag = diag + 0.5 * np.einsum("Ip,Iq,pq->I", occ_a, occ_b, eri_ppqq_cross, optimize=True)
        diag = diag + 0.5 * np.einsum("Ip,Iq,pq->I", occ_b, occ_a, eri_ppqq_cross, optimize=True)
        return diag

    def direct_sigma_from_integrals(self, ci, h1, eri):
        """
        Matrix-free compact CI Hamiltonian action for current active integrals.

        This mirrors the direct-CI compact Slater-Condon action but keeps a
        complex-safe NumPy implementation for MCTDHF's complex moving orbitals.
        """
        c = np.asarray(ci, dtype=complex).reshape(-1)
        if c.shape[0] != self.ndet:
            raise ValueError(f"CI vector length {c.shape[0]} does not match ndet={self.ndet}.")
        h1, eri_same, eri_cross = self._compact_integrals(h1, eri)
        conn = self.direct_connectivity
        sigma = self._direct_ci_diagonal(h1, eri_same, eri_cross) * c

        def add_singles(I, J, p, q, phase, spin):
            if len(I) == 0:
                return
            vals = np.empty(len(I), dtype=complex)
            occ_same = self.binary[J, spin, :]
            occ_cross = self.binary[J, 1 - spin, :]
            for k in range(len(I)):
                pk = p[k]
                qk = q[k]
                val = h1[pk, qk]
                for r in np.flatnonzero(occ_same[k]):
                    if r != qk:
                        val += eri_same[pk, qk, r, r]
                for r in np.flatnonzero(occ_cross[k]):
                    val += eri_cross[pk, qk, r, r]
                vals[k] = -phase[k] * val * c[J[k]]
            np.add.at(sigma, I, vals)

        def add_doubles(I, J, p, q, r, s, phase, tensor):
            if len(I) == 0:
                return
            vals = phase * tensor[p, q, r, s] * c[J]
            np.add.at(sigma, I, vals)

        add_singles(conn.I_A, conn.J_A, conn.p_A, conn.q_A, conn.phase_A, 0)
        add_singles(conn.I_B, conn.J_B, conn.p_B, conn.q_B, conn.phase_B, 1)
        add_doubles(conn.I_AA, conn.J_AA, conn.p_AA, conn.q_AA, conn.r_AA, conn.s_AA, conn.phase_AA, eri_same)
        add_doubles(conn.I_BB, conn.J_BB, conn.p_BB, conn.q_BB, conn.r_BB, conn.s_BB, conn.phase_BB, eri_same)
        add_doubles(conn.I_AB, conn.J_AB, conn.p_AB, conn.q_AB, conn.r_AB, conn.s_AB, conn.phase_AB, eri_cross)
        return sigma

    def sigma_vector(self, ci, orbitals=None, time=0.0, field=None, h1_ao=None, backend=None):
        """Apply the active CAS Hamiltonian to a CI vector."""
        if orbitals is None:
            orbitals = self.orbitals0
        hook = self._provider_hook("sigma_vector")
        if hook is not None:
            h1, eri = self.active_integrals(
                orbitals,
                time=time,
                field=field,
                h1_ao=h1_ao,
            )
            return _call_with_supported_keywords(
                hook,
                ci=ci,
                orbitals=orbitals,
                time=time,
                field=field,
                h1_ao=h1_ao,
                h1=h1,
                eri=eri,
                driver=self,
            )
        if self._resolved_ci_backend(backend) == "dense":
            return self.hamiltonian_matrix(
                orbitals=orbitals,
                time=time,
                field=field,
                h1_ao=h1_ao,
            ) @ np.asarray(ci, dtype=complex)
        h1, eri = self.active_integrals(
            orbitals,
            time=time,
            field=field,
            h1_ao=h1_ao,
        )
        return self.direct_sigma_from_integrals(ci, h1, eri)

    def ci_linear_operator(self, orbitals=None, time=0.0, field=None, h1_ao=None):
        """Return a matrix-free active-space Hamiltonian LinearOperator."""
        if orbitals is None:
            orbitals = self.orbitals0
        h1, eri = self.active_integrals(
            orbitals,
            time=time,
            field=field,
            h1_ao=h1_ao,
        )
        h1_compact, eri_same, eri_cross = self._compact_integrals(h1, eri)
        trace_h = np.sum(self._direct_ci_diagonal(h1_compact, eri_same, eri_cross))

        def matvec(vec):
            return self.direct_sigma_from_integrals(np.asarray(vec).reshape(-1), h1, eri)

        def matmat(mat):
            arr = np.asarray(mat)
            return np.column_stack([matvec(arr[:, j]) for j in range(arr.shape[1])])

        op = LinearOperator(
            (self.ndet, self.ndet),
            matvec=matvec,
            rmatvec=matvec,
            matmat=matmat,
            rmatmat=matmat,
            dtype=complex,
        )
        op.trace_h = trace_h
        return op

    def propagate_ci(self, ci, orbitals=None, time=0.0, dt=0.0, field=None, h1_ao=None, backend=None):
        """Propagate CI coefficients under fixed active orbitals for one step."""
        if orbitals is None:
            orbitals = self.orbitals0
        hook = self._provider_hook("propagate_ci")
        if hook is not None:
            h1, eri = self.active_integrals(
                orbitals,
                time=time,
                field=field,
                h1_ao=h1_ao,
            )
            return _call_with_supported_keywords(
                hook,
                ci=ci,
                orbitals=orbitals,
                time=time,
                dt=float(dt),
                field=field,
                h1_ao=h1_ao,
                h1=h1,
                eri=eri,
                driver=self,
            )
        c = np.asarray(ci, dtype=complex)
        if self._resolved_ci_backend(backend) == "dense":
            h = self.hamiltonian_matrix(
                orbitals=orbitals,
                time=time,
                field=field,
                h1_ao=h1_ao,
            )
            return expm(-1j * float(dt) * h) @ c
        h_op = self.ci_linear_operator(
            orbitals=orbitals,
            time=time,
            field=field,
            h1_ao=h1_ao,
        )
        trace_h = getattr(h_op, "trace_h", 0.0)
        return expm_multiply((-1j * float(dt)) * h_op, c, traceA=(-1j * float(dt)) * trace_h)

    def ci_eigenstates(
        self,
        nstates=1,
        orbitals=None,
        time=0.0,
        field=None,
        h1_ao=None,
        backend=None,
        tol=1.0e-10,
        maxiter=None,
    ):
        """Return lowest active-space CI eigenvalues and eigenvectors."""
        nstates = int(nstates)
        if nstates < 1 or nstates > self.ndet:
            raise ValueError(f"nstates must be between 1 and ndet={self.ndet}.")
        key = self._resolved_ci_backend(backend)
        if key == "dense" or nstates >= self.ndet:
            h = self.hamiltonian_matrix(
                orbitals=orbitals,
                time=time,
                field=field,
                h1_ao=h1_ao,
            )
            evals, vecs = eigh(h)
            return evals[:nstates].real, vecs[:, :nstates]

        if self.ndet <= 2 or nstates >= self.ndet - 1:
            h = self.hamiltonian_matrix(
                orbitals=orbitals,
                time=time,
                field=field,
                h1_ao=h1_ao,
            )
            evals, vecs = eigh(h)
            return evals[:nstates].real, vecs[:, :nstates]

        h_op = self.ci_linear_operator(
            orbitals=orbitals,
            time=time,
            field=field,
            h1_ao=h1_ao,
        )
        evals, vecs = eigsh(
            h_op,
            k=nstates,
            which="SA",
            tol=float(tol),
            maxiter=maxiter,
        )
        order = np.argsort(evals.real)
        return evals[order].real, vecs[:, order]

    def ci_vector(self, ci0=None, orbitals=None, time=0.0, field=None, h1_ao=None, backend=None):
        if orbitals is None:
            orbitals = self.orbitals0
        hook = self._provider_hook("ci_vector")
        if hook is not None:
            h1, eri = self.active_integrals(
                orbitals,
                time=time,
                field=field,
                h1_ao=h1_ao,
            )
            return _call_with_supported_keywords(
                hook,
                ci0=ci0,
                orbitals=orbitals,
                time=time,
                field=field,
                h1_ao=h1_ao,
                h1=h1,
                eri=eri,
                driver=self,
            )
        if ci0 is None:
            ci0 = self.initial_ci if self.initial_ci is not None else 0
        if np.isscalar(ci0):
            _evals, vecs = self.ci_eigenstates(
                nstates=int(ci0) + 1,
                orbitals=orbitals,
                time=time,
                field=field,
                h1_ao=h1_ao,
                backend=backend,
            )
            c = vecs[:, int(ci0)]
        else:
            c = np.asarray(ci0, dtype=complex).reshape(-1).copy()
        if c.shape[0] != self.ndet:
            raise ValueError(f"CI vector length {c.shape[0]} does not match ndet={self.ndet}.")
        norm = np.linalg.norm(c)
        if norm == 0.0:
            raise ValueError("Initial CI vector has zero norm.")
        return c / norm

    def make_rdm1(self, ci):
        return self.density_provider.make_rdm1(ci)

    def natural_occupations(self, ci):
        """Return active-space natural occupations sorted descending."""
        dm = self.make_rdm1(ci)
        dm1 = 0.5 * (dm + dm.conj().T)
        occ = np.linalg.eigvalsh(dm1).real
        return occ[::-1]

    def make_rdm2(self, ci):
        if not hasattr(self.density_provider, "make_rdm2"):
            raise NotImplementedError("density_provider does not provide make_rdm2().")
        return self.density_provider.make_rdm2(ci)

    def contract_rdm2_eri_full(self, ci, eri_full):
        return self.density_provider.contract_rdm2_eri_full(ci, eri_full)

    def contract_rdm2_eri_full_reference(self, ci, eri_full):
        """Reference contraction that explicitly materializes the active RDM2."""
        if hasattr(self.density_provider, "contract_rdm2_eri_full_reference"):
            return self.density_provider.contract_rdm2_eri_full_reference(ci, eri_full)
        dm2 = self.make_rdm2(ci)
        return np.einsum("oqrs,xqrs->xo", dm2, eri_full, optimize=True)

    def make_rdm1_ao(self, ci, orbitals=None):
        """Return the spin-traced AO density matrix for the active electrons."""
        if orbitals is None:
            orbitals = self.orbitals0
        dm1 = self.make_rdm1(ci)
        c = np.asarray(orbitals, dtype=complex)
        return c @ dm1.T @ c.conj().T

    def make_total_rdm1_ao(self, ci, orbitals=None):
        """Return spin-traced AO density including frozen-core electrons."""
        dm = self.make_rdm1_ao(ci, orbitals=orbitals)
        if self.core_density_ao is not None:
            dm = dm + self.core_density_ao
        return dm

    def one_body_expectation_ao(self, ci, operator_ao, orbitals=None):
        """Contract an AO one-body operator with the MCTDHF 1-RDM."""
        dm_ao = self.make_total_rdm1_ao(ci, orbitals=orbitals)
        op = np.asarray(operator_ao, dtype=complex)
        return np.einsum("ij,ji->", op, dm_ao, optimize=True)

    def dipole_moment(self, ci, orbitals=None):
        """Electronic dipole expectation value from the moving-orbital density."""
        op = self._interaction_operator_ao()
        if op is None:
            return np.zeros(3, dtype=float)
        dm_ao = self.make_total_rdm1_ao(ci, orbitals=orbitals)
        value = np.einsum("xij,ji->x", op, dm_ao, optimize=True)
        return value.real

    def _ci_tensor(self, ci):
        ci = np.asarray(ci, dtype=complex).reshape(-1)
        if ci.shape[0] != self.ndet:
            raise ValueError(f"CI vector length {ci.shape[0]} does not match ndet={self.ndet}.")
        return ci.reshape(self.nalpha_strings, self.nbeta_strings)

    def rotate_ci_for_orbital_rotation(self, ci, orbital_rotation):
        """
        Transform CI coefficients after ``C_new = C_old @ orbital_rotation``.

        The returned coefficients represent the same many-electron state in the
        rotated active orbital basis.
        """
        rot = np.asarray(orbital_rotation, dtype=complex)
        if rot.shape != (self.norb, self.norb):
            raise ValueError("orbital_rotation must have shape (norb, norb).")
        wa = _string_transform_matrix(rot, self.alpha_strings, complex)
        wb = _string_transform_matrix(rot, self.beta_strings, complex)
        coeff = self._ci_tensor(ci)
        transformed = np.linalg.solve(wa, coeff)
        transformed = np.linalg.solve(wb, transformed.T).T
        out = transformed.reshape(self.ndet)
        norm = np.linalg.norm(out)
        if norm == 0.0:
            raise np.linalg.LinAlgError("Orbital rotation produced a zero CI vector.")
        return out / norm

    def align_orbital_gauge(self, old_orbitals, new_orbitals, ci):
        """
        Remove numerical active-active rotation from ``new_orbitals``.

        MCTDHF is propagated in the gauge where active orbitals have no
        active-space time derivative.  After a finite step and metric
        reorthonormalization, this routine applies the closest active-space
        unitary that makes the overlap with the previous frame Hermitian
        positive and transforms the CI vector inversely.
        """
        sact = old_orbitals.conj().T @ self.overlap @ new_orbitals
        u, _sigma, vh = np.linalg.svd(sact, full_matrices=False)
        polar = u @ vh
        rotation = polar.conj().T
        aligned_orbitals = new_orbitals @ rotation
        if self._is_dense_ci_vector(ci):
            aligned_ci = self.rotate_ci_for_orbital_rotation(ci, rotation)
        else:
            hook = self._provider_hook("rotate_ci_for_orbital_rotation")
            if hook is None:
                raise NotImplementedError(
                    "External CI state with moving orbitals requires "
                    "density_provider.rotate_ci_for_orbital_rotation(...)."
                )
            aligned_ci = _call_with_supported_keywords(
                hook,
                ci=ci,
                orbital_rotation=rotation,
                old_orbitals=old_orbitals,
                new_orbitals=new_orbitals,
                aligned_orbitals=aligned_orbitals,
                driver=self,
            )
        return aligned_ci, aligned_orbitals, rotation

    def electronic_energy(self, ci, orbitals=None, time=0.0, field=None, h1_ao=None):
        if orbitals is None:
            orbitals = self.orbitals0
        hook = self._provider_hook("electronic_energy")
        if hook is not None:
            h1, eri = self.active_integrals(
                orbitals,
                time=time,
                field=field,
                h1_ao=h1_ao,
            )
            return _call_with_supported_keywords(
                hook,
                ci=ci,
                orbitals=orbitals,
                time=time,
                field=field,
                h1_ao=h1_ao,
                h1=h1,
                eri=eri,
                driver=self,
            )
        if not self._is_dense_ci_vector(ci):
            raise NotImplementedError(
                "External CI state requires density_provider.electronic_energy(ci=...)."
            )
        sigma = self.sigma_vector(
            ci,
            orbitals=orbitals,
            time=time,
            field=field,
            h1_ao=h1_ao,
        )
        c = np.asarray(ci, dtype=complex)
        return np.vdot(c, sigma)

    def energy(self, ci, orbitals=None, time=0.0, field=None, h1_ao=None):
        return self.electronic_energy(
            ci,
            orbitals=orbitals,
            time=time,
            field=field,
            h1_ao=h1_ao,
        ) + (
            self.e_nuc
            + self.e_core
            + self.core_one_body_energy(time=time, field=field, h1_ao=h1_ao).real
        ) * self.ci_norm(ci)

    def orbital_rhs(self, orbitals, ci, time=0.0, field=None, h1_ao=None):
        """Return ``dC/dt`` for active AO orbital coefficients."""
        if self.orbital_mode in {"frozen", "tdcasci", "fixed"}:
            return np.zeros_like(orbitals, dtype=complex)
        if self.orbital_mode not in {"mctdhf", "projected", "projected_rdm"}:
            raise ValueError("orbital_mode must be 'mctdhf' or 'frozen'.")

        c = np.asarray(orbitals, dtype=complex)
        y = self.s_half @ c
        h_ao = self.effective_one_body_ao(time, field=field, h1_ao=h1_ao)
        h_full = self.s_inv_half.conj().T @ h_ao @ c
        eri_full = transform_spatial_eri_to_mo(
            self.mf,
            self.s_inv_half,
            c,
            c,
            c,
            use_cholesky=self.use_cholesky,
        )
        dm1 = self.make_rdm1(ci)
        rho_inv = np.linalg.pinv(dm1, rcond=self.rdm_rcond)
        two_body = self.contract_rdm2_eri_full(ci, eri_full)
        mean_field = h_full + two_body @ rho_inv.T
        if self.core_orbitals is None:
            projector_basis = y
        else:
            projector_basis = np.column_stack((self.s_half @ self.core_orbitals, y))
        projected = self.project_out_subspace(mean_field, projector_basis)
        return -1j * (self.s_inv_half @ projected)

    def _step_midpoint(self, c0, o0, time, dt, field=None, h1_ao=None):
        c_mid = self.propagate_ci(
            c0,
            orbitals=o0,
            time=time,
            dt=0.5 * dt,
            field=field,
            h1_ao=h1_ao,
        )
        o_mid = self.orthonormalize_orbitals(
            o0 + 0.5 * dt * self.orbital_rhs(o0, c0, time=time, field=field, h1_ao=h1_ao)
        )

        t_mid = float(time) + 0.5 * dt
        c1 = self.propagate_ci(
            c0,
            orbitals=o_mid,
            time=t_mid,
            dt=dt,
            field=field,
            h1_ao=h1_ao,
        )
        o1 = self.orthonormalize_orbitals(
            o0 + dt * self.orbital_rhs(o_mid, c_mid, time=t_mid, field=field, h1_ao=h1_ao)
        )
        if self._is_dense_ci_vector(c1):
            c1 = c1 / np.linalg.norm(c1)
        if self.orbital_mode not in {"frozen", "tdcasci", "fixed"}:
            c1, o1, _rotation = self.align_orbital_gauge(o0, o1, c1)
        return c1, o1

    def _step_rk4(self, c0, o0, time, dt, field=None, h1_ao=None):
        t0 = float(time)
        t_mid = t0 + 0.5 * dt
        t1 = t0 + dt

        k1 = self.orbital_rhs(o0, c0, time=t0, field=field, h1_ao=h1_ao)
        c_half_1 = self.propagate_ci(
            c0,
            orbitals=o0,
            time=t0,
            dt=0.5 * dt,
            field=field,
            h1_ao=h1_ao,
        )
        o_half_1 = self.orthonormalize_orbitals(o0 + 0.5 * dt * k1)

        k2 = self.orbital_rhs(o_half_1, c_half_1, time=t_mid, field=field, h1_ao=h1_ao)
        c_half_2 = self.propagate_ci(
            c0,
            orbitals=o_half_1,
            time=t_mid,
            dt=0.5 * dt,
            field=field,
            h1_ao=h1_ao,
        )
        o_half_2 = self.orthonormalize_orbitals(o0 + 0.5 * dt * k2)

        k3 = self.orbital_rhs(o_half_2, c_half_2, time=t_mid, field=field, h1_ao=h1_ao)
        c1 = self.propagate_ci(
            c0,
            orbitals=o_half_2,
            time=t_mid,
            dt=dt,
            field=field,
            h1_ao=h1_ao,
        )
        o_full = self.orthonormalize_orbitals(o0 + dt * k3)

        k4 = self.orbital_rhs(o_full, c1, time=t1, field=field, h1_ao=h1_ao)
        o1 = self.orthonormalize_orbitals(o0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        if self._is_dense_ci_vector(c1):
            c1 = c1 / np.linalg.norm(c1)
        c1, o1, _rotation = self.align_orbital_gauge(o0, o1, c1)
        return c1, o1

    def step(self, ci, orbitals, time, dt, field=None, h1_ao=None):
        """Advance CI and active orbitals by one configured time step."""
        dt = float(dt)
        c0 = np.asarray(ci, dtype=complex) if self._is_dense_ci_vector(ci) else ci
        o0 = np.asarray(orbitals, dtype=complex)
        if self.orbital_mode in {"frozen", "tdcasci", "fixed"}:
            c1 = self.propagate_ci(
                c0,
                orbitals=o0,
                time=time,
                dt=dt,
                field=field,
                h1_ao=h1_ao,
            )
            if self._is_dense_ci_vector(c1):
                c1 = c1 / np.linalg.norm(c1)
            return c1, o0

        if self.orbital_integrator == "midpoint":
            return self._step_midpoint(c0, o0, time, dt, field=field, h1_ao=h1_ao)
        if self.orbital_integrator == "rk4":
            return self._step_rk4(c0, o0, time, dt, field=field, h1_ao=h1_ao)
        raise ValueError("orbital_integrator must be 'midpoint' or 'rk4'.")

    def run(
        self,
        dt,
        nsteps,
        ci0=None,
        orbitals0=None,
        field=None,
        h1_ao=None,
        t0=0.0,
        kick=None,
        save_every=1,
        store_ci=True,
        store_orbitals=True,
    ):
        """Propagate the MCTDHF state for ``nsteps`` time steps."""
        dt = float(dt)
        t0 = float(t0)
        if not np.isfinite(dt):
            raise ValueError("dt must be finite.")
        if not np.isfinite(t0):
            raise ValueError("t0 must be finite.")
        try:
            nsteps_float = float(nsteps)
        except (TypeError, ValueError) as exc:
            raise ValueError("nsteps must be a nonnegative integer.") from exc
        if not np.isfinite(nsteps_float) or not nsteps_float.is_integer():
            raise ValueError("nsteps must be a nonnegative integer.")
        nsteps = int(nsteps_float)
        if nsteps < 0:
            raise ValueError("nsteps must be a nonnegative integer.")
        try:
            save_every_float = float(save_every)
        except (TypeError, ValueError) as exc:
            raise ValueError("save_every must be a positive integer.") from exc
        if not np.isfinite(save_every_float) or not save_every_float.is_integer():
            raise ValueError("save_every must be a positive integer.")
        save_every = int(save_every_float)
        if save_every < 1:
            raise ValueError("save_every must be a positive integer.")

        full_times = t0 + dt * np.arange(nsteps + 1, dtype=float)
        save_steps = list(range(0, nsteps + 1, save_every))
        if save_steps[-1] != nsteps:
            save_steps.append(nsteps)
        save_set = set(save_steps)
        times = full_times[np.asarray(save_steps, dtype=int)]
        orbitals = self.orbitals0 if orbitals0 is None else self.orthonormalize_orbitals(orbitals0)
        ci = self.ci_vector(
            ci0,
            orbitals=orbitals,
            time=t0,
            field=field,
            h1_ao=h1_ao,
            backend=self.ci_backend,
        )
        if kick is not None:
            ci = self.kick_ci(ci, orbitals=orbitals, **kick)
        ci_reference = np.asarray(ci, dtype=complex).copy() if self._is_dense_ci_vector(ci) else ci
        orbital_reference = np.asarray(orbitals, dtype=complex).copy()

        dense_ci_history = self._is_dense_ci_vector(ci)
        ci_hist = None
        if store_ci and dense_ci_history:
            ci_hist = np.zeros((times.size, self.ndet), dtype=complex)
        elif store_ci:
            ci_hist = np.empty(times.size, dtype=object)
        orb_hist = None
        if store_orbitals:
            orb_hist = np.zeros((times.size, self.nao, self.norb), dtype=complex)
        energies = np.zeros(times.size, dtype=float)
        electronic = np.zeros(times.size, dtype=float)
        norms = np.zeros(times.size, dtype=float)
        orbital_errors = np.zeros(times.size, dtype=float)
        active_gauge_errors = np.zeros(times.size, dtype=float)
        core_overlap_errors = np.zeros(times.size, dtype=float)
        natural_occupations = np.zeros((times.size, self.norb), dtype=float)
        natural_occupation_trace_errors = np.zeros(times.size, dtype=float)
        autocorrelation = np.zeros(times.size, dtype=complex)
        dipoles = np.zeros((times.size, 3), dtype=float)
        fields = np.zeros((times.size, 3), dtype=float)
        previous_saved_orbitals = None
        isave = 0

        for istep, time in enumerate(full_times):
            if istep in save_set:
                if store_ci:
                    if dense_ci_history:
                        if not self._is_dense_ci_vector(ci):
                            raise TypeError("CI provider changed state type during propagation.")
                        ci_hist[isave] = ci
                    else:
                        ci_hist[isave] = ci
                if store_orbitals:
                    orb_hist[isave] = orbitals
                electronic[isave] = self.electronic_energy(
                    ci,
                    orbitals=orbitals,
                    time=time,
                    field=field,
                    h1_ao=h1_ao,
                ).real
                norms[isave] = self.ci_norm(ci)
                try:
                    autocorrelation[isave] = self.state_overlap(
                        ci_reference,
                        ci,
                        bra_orbitals=orbital_reference,
                        ket_orbitals=orbitals,
                    )
                except NotImplementedError:
                    autocorrelation[isave] = np.nan + 1j * np.nan
                energies[isave] = self.energy(
                    ci,
                    orbitals=orbitals,
                    time=time,
                    field=field,
                    h1_ao=h1_ao,
                ).real
                orbital_errors[isave] = self.orbital_error(orbitals)
                core_overlap_errors[isave] = self.core_active_overlap_error(orbitals)
                natural_occupations[isave] = self.natural_occupations(ci)
                natural_occupation_trace_errors[isave] = (
                    np.sum(natural_occupations[isave]) - float(self.na + self.nb)
                )
                if previous_saved_orbitals is not None:
                    active_gauge_errors[isave] = self.active_gauge_error(previous_saved_orbitals, orbitals)
                dipoles[isave] = self.dipole_moment(ci, orbitals=orbitals)
                fields[isave] = self.field_vector(time, field=field)
                previous_saved_orbitals = orbitals
                isave += 1
            if istep < nsteps:
                ci, orbitals = self.step(
                    ci,
                    orbitals,
                    time,
                    dt,
                    field=field,
                    h1_ao=h1_ao,
                )

        return MCTDHFTrajectory(
            times=times,
            ci=ci_hist,
            orbitals=orb_hist,
            energies=energies,
            electronic_energies=electronic,
            norms=norms,
            orbital_errors=orbital_errors,
            active_gauge_errors=active_gauge_errors,
            core_overlap_errors=core_overlap_errors,
            natural_occupations=natural_occupations,
            natural_occupation_trace_errors=natural_occupation_trace_errors,
            autocorrelation=autocorrelation,
            dipoles=dipoles,
            fields=fields,
        )


__all__ = [
    "DenseCIDensityProvider",
    "DMRGDensityProvider",
    "MCTDHF",
    "MCTDHFTrajectory",
    "RDM12DensityProvider",
]
