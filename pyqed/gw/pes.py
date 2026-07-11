"""Photoelectron spectra from GW quasiparticle orbitals.

This module provides a first ab initio PES workflow: peak positions come from
GW quasiparticle energies, while intensities are computed with analytic
Gaussian plane-wave dipole integrals.
"""

from dataclasses import dataclass, fields

import numpy as np

from pyqed.qchem.basis import ContractedGaussian
from pyqed.units import au2ev, fine_structure


@dataclass
class PESResult:
    """Photoelectron stick-spectrum data."""

    orbitals: np.ndarray
    binding_energies: np.ndarray
    kinetic_energies: np.ndarray
    intensities: np.ndarray
    qp_weights: np.ndarray
    transition_moments: np.ndarray
    photon_energy: float
    origin: np.ndarray
    units: str = "au"
    continuum: str = "plane_wave"
    gauge: str = "length"
    direction: np.ndarray = None
    polarization: np.ndarray = None
    averaging: str = "orientation"
    intensity_kind: str = "matrix_element"
    intensity_units: str = "arb."
    cross_section_prefactors: np.ndarray = None
    dyson_kind: str = "mo"


@dataclass
class PESSpectralResult:
    """Orbital-resolved GW spectral function and optional PES signal."""

    orbitals: np.ndarray
    omega: np.ndarray
    binding_energies: np.ndarray
    spectral_function: np.ndarray
    spectral_matrix: np.ndarray = None
    signal: np.ndarray = None
    photon_energy: float = None
    origin: np.ndarray = None
    direction: np.ndarray = None
    polarization: np.ndarray = None
    units: str = "au"
    continuum: str = "plane_wave"
    gauge: str = "length"
    averaging: str = "orientation"
    intensity_kind: str = "spectral_function"
    intensity_units: str = "1/Ha"
    cross_section_prefactors: np.ndarray = None
    approximation: str = "diagonal"

    def peaks(
        self,
        source="signal",
        units="ev",
        threshold_rel=0.05,
        min_distance=1,
        max_peaks=None,
    ):
        """Find local maxima in the total PES signal or orbital spectral functions."""
        return spectral_peaks(
            self,
            source=source,
            units=units,
            threshold_rel=threshold_rel,
            min_distance=min_distance,
            max_peaks=max_peaks,
        )


@dataclass
class PESPeakResult:
    """Peak positions and heights extracted from a spectral PES result."""

    orbitals: np.ndarray
    binding_energies: np.ndarray
    intensities: np.ndarray
    indices: np.ndarray
    source: str = "signal"
    units: str = "ev"


@dataclass
class DysonOrbitalResult:
    """GW Dyson orbital from a matrix quasiparticle equation."""

    orbital: int
    energy: float
    qp_weight: float
    mo_coefficients: np.ndarray
    ao_coefficients: np.ndarray
    orbital_space: np.ndarray
    residual: float
    converged: bool
    niter: int
    spin: str = "alpha"
    dyson_kind: str = "matrix"


def _unit_scale(units):
    key = str(units).lower()
    if key in {"au", "hartree", "ha"}:
        return 1.0
    if key in {"ev", "electronvolt", "electronvolts"}:
        return au2ev
    raise ValueError("units must be 'au' or 'ev'.")


def _as_au(value, units):
    return float(value) / _unit_scale(units)


def _normalize(vec):
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return vec / norm


def nuclear_center_of_mass(mol):
    """Return the nuclear center of mass in bohr-like molecule coordinates."""
    if hasattr(mol, "center_of_mass"):
        return np.asarray(mol.center_of_mass(), dtype=float)
    if hasattr(mol, "atom_mass_list") and hasattr(mol, "atom_coords"):
        masses = np.asarray(mol.atom_mass_list(), dtype=float)
        coords = np.asarray(mol.atom_coords(), dtype=float)
        return np.einsum("a,ax->x", masses, coords, optimize=True) / masses.sum()
    if hasattr(mol, "nuc_charge_center"):
        return np.asarray(mol.nuc_charge_center(), dtype=float)
    return np.zeros(3)


def _fibonacci_sphere(npoints):
    npoints = int(npoints)
    if npoints <= 0:
        raise ValueError("npoints must be positive.")
    if npoints == 1:
        return np.array([[0.0, 0.0, 1.0]])
    idx = np.arange(npoints, dtype=float)
    z = 1.0 - 2.0 * (idx + 0.5) / npoints
    radius = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = np.pi * (3.0 - np.sqrt(5.0)) * idx
    return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))


def _gaussian_fourier_moments_1d(alpha, k, max_order):
    """Return J_n(k) = int u^n exp(-alpha u^2) exp(-i k u) du."""
    alpha = float(alpha)
    k = float(k)
    max_order = int(max_order)
    out = np.empty(max_order + 1, dtype=np.complex128)
    out[0] = np.sqrt(np.pi / alpha) * np.exp(-(k * k) / (4.0 * alpha))
    if max_order == 0:
        return out
    out[1] = -1j * k * out[0] / (2.0 * alpha)
    for n in range(1, max_order):
        out[n + 1] = (n * out[n - 1] - 1j * k * out[n]) / (2.0 * alpha)
    return out


def _primitive_plane_wave_dipole(alpha, shell, center, kvec, origin):
    shell = tuple(int(x) for x in shell)
    center = np.asarray(center, dtype=float)
    kvec = np.asarray(kvec, dtype=float)
    origin = np.asarray(origin, dtype=float)
    lx, ly, lz = shell

    mx = _gaussian_fourier_moments_1d(alpha, kvec[0], lx + 1)
    my = _gaussian_fourier_moments_1d(alpha, kvec[1], ly + 1)
    mz = _gaussian_fourier_moments_1d(alpha, kvec[2], lz + 1)
    phase = np.exp(-1j * float(np.dot(kvec, center)))

    base = phase * mx[lx] * my[ly] * mz[lz]
    dip = np.empty(3, dtype=np.complex128)
    dip[0] = (center[0] - origin[0]) * base + phase * mx[lx + 1] * my[ly] * mz[lz]
    dip[1] = (center[1] - origin[1]) * base + phase * mx[lx] * my[ly + 1] * mz[lz]
    dip[2] = (center[2] - origin[2]) * base + phase * mx[lx] * my[ly] * mz[lz + 1]
    return dip


def ao_plane_wave_dipoles(mol, kvec, origin=None):
    """
    Analytic AO integrals ``<exp(i k.r)|(r-origin)|AO_mu>``.

    Returns an array with shape ``(nao, 3)`` in the molecule's AO basis.
    """
    if origin is None:
        origin = nuclear_center_of_mass(mol)
    origin = np.asarray(origin, dtype=float)

    if not hasattr(mol, "_cart_basis"):
        raise ValueError("Analytic PES intensities require a native PyQED molecule.")
    basis, transform = mol._cart_basis()
    if not all(isinstance(fn, ContractedGaussian) for fn in basis):
        raise ValueError("Analytic PES intensities require a ContractedGaussian AO basis.")

    cart_dipoles = np.zeros((len(basis), 3), dtype=np.complex128)
    for ao, fn in enumerate(basis):
        for alpha, weight in zip(fn.exps, fn.prim_weights):
            cart_dipoles[ao] += weight * _primitive_plane_wave_dipole(
                alpha,
                fn.shell,
                fn.origin,
                kvec,
                origin,
            )

    if transform is None:
        return cart_dipoles
    return np.einsum("ps,px->sx", transform, cart_dipoles, optimize=True)


def _mo_coefficients_for_gw(gw):
    coeff = getattr(gw, "mo_coeff", None)
    if coeff is not None:
        return np.asarray(coeff, dtype=float)
    coeff = getattr(gw._scf, "mo_coeff", None)
    if coeff is None:
        raise ValueError("GW PES requires molecular orbital coefficients.")
    return np.asarray(coeff, dtype=float)


def _spin_offset(spin):
    key = str(spin).lower()
    if key in {"alpha", "a", "up", "0"}:
        return 0
    if key in {"beta", "b", "down", "1"}:
        return 1
    raise ValueError("spin must be 'alpha' or 'beta'.")


def _orbital_space(gw, orbital, orbital_space=None):
    nmo = int(gw.nso // 2)
    if orbital_space is None:
        orbital_space = np.arange(nmo, dtype=int)
    else:
        orbital_space = np.atleast_1d(np.asarray(orbital_space, dtype=int))
    if orbital_space.ndim != 1:
        raise ValueError("orbital_space must be one-dimensional.")
    if np.any(orbital_space < 0) or np.any(orbital_space >= nmo):
        raise IndexError("One or more orbital_space indices are out of range.")
    if int(orbital) not in set(int(item) for item in orbital_space):
        raise ValueError("orbital must be included in orbital_space.")
    return orbital_space


def _self_energy_blocks(gw, omega, e_rpa, t_rpa, orbital_space, spin="alpha"):
    offset = _spin_offset(spin)
    spin_orbitals = 2 * np.asarray(orbital_space, dtype=int) + offset
    nspace = len(spin_orbitals)
    sigma_c = np.zeros((nspace, nspace), dtype=np.complex128)
    sigma_x = np.zeros_like(sigma_c)
    v_mf = np.zeros_like(sigma_c)
    for row, p in enumerate(spin_orbitals):
        for col, q in enumerate(spin_orbitals):
            sigma_c[row, col], sigma_x[row, col] = gw.sigma(
                int(p),
                int(q),
                float(omega),
                e_rpa,
                t_rpa,
            )
            v_mf[row, col] = gw.v_mf[int(p), int(q)]
    return sigma_c, sigma_x, v_mf


def _dyson_effective_matrix(gw, omega, e_rpa, t_rpa, orbital_space, spin="alpha"):
    sigma_c, sigma_x, v_mf = _self_energy_blocks(
        gw,
        omega,
        e_rpa,
        t_rpa,
        orbital_space,
        spin=spin,
    )
    offset = _spin_offset(spin)
    spin_orbitals = 2 * np.asarray(orbital_space, dtype=int) + offset
    h_eff = np.diag(np.asarray(gw.e_mf, dtype=float)[spin_orbitals])
    h_eff = h_eff + (sigma_c + sigma_x - v_mf).real
    return 0.5 * (h_eff + h_eff.T)


def _select_eigenpair(matrix, reference):
    evals, evecs = np.linalg.eigh(matrix)
    overlaps = np.abs(evecs.T.conjugate() @ reference)
    idx = int(np.argmax(overlaps))
    vec = evecs[:, idx]
    phase_idx = int(np.argmax(np.abs(vec)))
    if vec[phase_idx] < 0.0:
        vec = -vec
    return float(evals[idx]), vec


def dyson_orbital(
    gw,
    orbital,
    orbital_space=None,
    spin="alpha",
    e_rpa=None,
    t_rpa=None,
    initial_energy=None,
    delta=1.0e-4,
    conv_tol=1.0e-8,
    max_cycle=30,
):
    """
    Solve a matrix GW quasiparticle equation and return the Dyson orbital.

    The equation solved in the selected spatial-MO subspace is
    ``[eps + Sigma(E) - v_mf] c = E c``.  The returned AO coefficients include
    the pole residue, ``sqrt(Z) * C_MO c``, with
    ``Z = [c.T (I - dSigma/domega) c]^-1``.
    """
    if gw.e_qp is None:
        raise ValueError("Run GW before requesting a Dyson orbital.")
    orbital = int(orbital)
    orbital_space = _orbital_space(gw, orbital, orbital_space=orbital_space)
    if e_rpa is None or t_rpa is None:
        e_rpa, t_rpa = gw.rpa(method=gw.screening)

    pos = int(np.where(orbital_space == orbital)[0][0])
    reference = np.zeros(len(orbital_space), dtype=float)
    reference[pos] = 1.0
    omega = float(gw.e_qp[orbital] if initial_energy is None else initial_energy)
    delta = float(delta)
    if delta <= 0.0:
        raise ValueError("delta must be positive.")
    max_cycle = int(max_cycle)
    if max_cycle <= 0:
        raise ValueError("max_cycle must be positive.")

    converged = False
    residual = np.inf
    coeff = reference.copy()
    for cycle in range(1, max_cycle + 1):
        h_eff = _dyson_effective_matrix(gw, omega, e_rpa, t_rpa, orbital_space, spin=spin)
        eig, coeff = _select_eigenpair(h_eff, reference)
        residual = eig - omega
        if abs(residual) < conv_tol:
            converged = True
            break

        h_plus = _dyson_effective_matrix(
            gw,
            omega + delta,
            e_rpa,
            t_rpa,
            orbital_space,
            spin=spin,
        )
        h_minus = _dyson_effective_matrix(
            gw,
            omega - delta,
            e_rpa,
            t_rpa,
            orbital_space,
            spin=spin,
        )
        dh = (h_plus - h_minus) / (2.0 * delta)
        slope = float(coeff.T @ dh @ coeff)
        jac = slope - 1.0
        if abs(jac) < 1.0e-12:
            break
        step = residual / jac
        if abs(step) > 0.5:
            step = np.sign(step) * 0.5
        omega -= step

    h_eff = _dyson_effective_matrix(gw, omega, e_rpa, t_rpa, orbital_space, spin=spin)
    eig, coeff = _select_eigenpair(h_eff, reference)
    residual = eig - omega
    if abs(residual) < conv_tol:
        converged = True

    h_plus = _dyson_effective_matrix(
        gw,
        omega + delta,
        e_rpa,
        t_rpa,
        orbital_space,
        spin=spin,
    )
    h_minus = _dyson_effective_matrix(
        gw,
        omega - delta,
        e_rpa,
        t_rpa,
        orbital_space,
        spin=spin,
    )
    dh = (h_plus - h_minus) / (2.0 * delta)
    residue_denominator = float(coeff.T @ (np.eye(len(orbital_space)) - dh) @ coeff)
    qp_weight = 1.0 / residue_denominator
    amplitude = np.sqrt(qp_weight + 0j)
    mo_coeff = _mo_coefficients_for_gw(gw)
    ao_coefficients = amplitude * (mo_coeff[:, orbital_space] @ coeff)

    if np.linalg.norm(ao_coefficients.imag) < 1.0e-12:
        ao_coefficients = ao_coefficients.real
    mo_dyson_coefficients = amplitude * coeff
    if np.linalg.norm(np.asarray(mo_dyson_coefficients).imag) < 1.0e-12:
        mo_dyson_coefficients = mo_dyson_coefficients.real

    return DysonOrbitalResult(
        orbital=orbital,
        energy=float(omega),
        qp_weight=float(qp_weight),
        mo_coefficients=mo_dyson_coefficients,
        ao_coefficients=ao_coefficients,
        orbital_space=orbital_space.copy(),
        residual=float(residual),
        converged=bool(converged),
        niter=cycle,
        spin=str(spin).lower(),
    )


def quasiparticle_weights(gw, orbitals=None, delta=1.0e-4):
    """
    Finite-difference GW quasiparticle pole strengths.

    ``Z_i = [1 - d Re Sigma_i(omega) / d omega]^-1`` evaluated at the GW
    quasiparticle energy. The diagonal spin-up self-energy is used for the
    corresponding spatial orbital.
    """
    if gw.e_qp is None:
        raise ValueError("Run GW before requesting quasiparticle weights.")
    if orbitals is None:
        orbitals = np.arange(len(gw.e_qp), dtype=int)
    orbitals = np.atleast_1d(np.asarray(orbitals, dtype=int))
    if np.any(orbitals < 0) or np.any(orbitals >= len(gw.e_qp)):
        raise IndexError("One or more orbital indices are out of range.")

    e_rpa, t_rpa = gw.rpa(method=gw.screening)
    delta = float(delta)
    if delta <= 0.0:
        raise ValueError("delta must be positive.")

    weights = np.empty(len(orbitals), dtype=float)
    for pos, orbital in enumerate(orbitals):
        p = 2 * int(orbital)
        omega = float(gw.e_qp[int(orbital)])
        sigma_plus = gw.sigma(p, p, omega + delta, e_rpa, t_rpa)[0]
        sigma_minus = gw.sigma(p, p, omega - delta, e_rpa, t_rpa)[0]
        derivative = ((sigma_plus - sigma_minus) / (2.0 * delta)).real
        weights[pos] = 1.0 / (1.0 - derivative)
    return weights


def _qp_weights(gw, orbitals, qp_weight=None):
    if qp_weight is None:
        return np.ones(len(orbitals), dtype=float)
    if isinstance(qp_weight, str):
        key = qp_weight.lower().replace("-", "_")
        if key in {"gw", "z", "qp", "quasiparticle", "finite_difference"}:
            return quasiparticle_weights(gw, orbitals=orbitals)
        if key in {"none", "unit", "ones"}:
            return np.ones(len(orbitals), dtype=float)
        raise ValueError("qp_weight must be None, numeric, array-like, or 'gw'.")
    if np.isscalar(qp_weight):
        return np.full(len(orbitals), float(qp_weight))
    qp_weight = np.asarray(qp_weight, dtype=float)
    if qp_weight.shape[0] == len(orbitals):
        return qp_weight.copy()
    return qp_weight[np.asarray(orbitals, dtype=int)]


def _dyson_key(dyson):
    key = str(dyson).lower().replace("-", "_")
    aliases = {
        "mo": "mo",
        "orbital": "mo",
        "canonical": "mo",
        "hf": "mo",
        "qp": "qp",
        "gw": "qp",
        "quasiparticle": "qp",
        "renormalized": "qp",
        "renormalised": "qp",
        "matrix": "matrix",
        "full": "matrix",
        "dyson": "matrix",
        "full_matrix": "matrix",
        "matrix_qp": "matrix",
    }
    if key not in aliases:
        raise ValueError("dyson must be 'mo', 'qp', or 'matrix'.")
    return aliases[key]


def _dyson_amplitudes(gw, orbitals, dyson="mo", qp_weight=None):
    key = _dyson_key(dyson)
    if key == "matrix":
        if qp_weight is not None:
            raise ValueError("dyson='matrix' already includes sqrt(Z); do not also pass qp_weight.")
        return key, None, np.ones(len(orbitals), dtype=float), np.ones(len(orbitals), dtype=float)
    if key == "mo":
        weights = _qp_weights(gw, orbitals, qp_weight=qp_weight)
        return key, np.ones(len(orbitals), dtype=float), weights, weights

    if qp_weight is not None:
        raise ValueError("dyson='qp' already includes sqrt(Z); do not also pass qp_weight.")
    weights = quasiparticle_weights(gw, orbitals=orbitals)
    if np.any(weights < 0.0):
        raise ValueError("Cannot build qp Dyson amplitudes from negative quasiparticle weights.")
    amplitudes = np.sqrt(weights)
    intensity_weights = np.ones(len(orbitals), dtype=float)
    return key, amplitudes, intensity_weights, weights


def _transverse_polarization_average(dipole, direction):
    direction = _normalize(direction)
    transverse_norm = np.vdot(dipole, dipole).real - abs(np.dot(direction, dipole)) ** 2
    return 0.5 * max(0.0, float(transverse_norm))


def _intensity_key(intensity):
    key = str(intensity).lower().replace("-", "_")
    aliases = {
        "matrix": "matrix_element",
        "matrix_element": "matrix_element",
        "matrix_elements": "matrix_element",
        "me": "matrix_element",
        "strength": "matrix_element",
        "cross_section": "cross_section",
        "cross": "cross_section",
        "sigma": "cross_section",
        "normalized": "normalized",
        "normalised": "normalized",
        "norm": "normalized",
    }
    if key not in aliases:
        raise ValueError("intensity must be 'matrix_element', 'cross_section', or 'normalized'.")
    return aliases[key]


def _spectral_approx_key(approx):
    key = str(approx).lower().replace("-", "_")
    aliases = {
        "diag": "diagonal",
        "diagonal": "diagonal",
        "orbital": "diagonal",
        "channel": "diagonal",
        "matrix": "matrix",
        "full": "matrix",
        "full_matrix": "matrix",
    }
    if key not in aliases:
        raise ValueError("approx must be 'diagonal' or 'matrix'.")
    return aliases[key]


def _cross_section_prefactors(photon_energy, kinetic_energies, angular_integrated=False):
    kinetic_energies = np.asarray(kinetic_energies, dtype=float)
    electron_momenta = np.sqrt(2.0 * np.clip(kinetic_energies, 0.0, None))
    prefactors = 4.0 * np.pi**2 * fine_structure * float(photon_energy) * electron_momenta
    if angular_integrated:
        prefactors = 4.0 * np.pi * prefactors
    return prefactors


def _apply_intensity_convention(
    matrix_strengths,
    photon_energy,
    kinetic_energies,
    intensity,
    angular_integrated=False,
):
    key = _intensity_key(intensity)
    matrix_strengths = np.asarray(matrix_strengths, dtype=float)
    if key == "matrix_element":
        return matrix_strengths, key, "arb.", np.ones_like(matrix_strengths)
    if key == "cross_section":
        prefactors = _cross_section_prefactors(
            photon_energy,
            kinetic_energies,
            angular_integrated=angular_integrated,
        )
        units = "a0^2" if angular_integrated else "a0^2/sr"
        return prefactors * matrix_strengths, key, units, prefactors

    scale = float(np.max(np.abs(matrix_strengths))) if matrix_strengths.size else 0.0
    normalized = matrix_strengths / scale if scale > 0.0 else matrix_strengths.copy()
    return normalized, key, "normalized", np.ones_like(matrix_strengths)


def _energy_grid_from_inputs(
    omega_grid=None,
    binding_grid=None,
    units="au",
    npoints=1000,
    binding_range=None,
    default_binding_max=2.0,
):
    if omega_grid is not None and binding_grid is not None:
        raise ValueError("Provide only one of omega_grid or binding_grid.")
    if omega_grid is not None:
        omega = np.asarray(omega_grid, dtype=float) / _unit_scale(units)
        return omega
    if binding_grid is not None:
        binding = np.asarray(binding_grid, dtype=float) / _unit_scale(units)
        return -binding
    if binding_range is None:
        binding_min = 0.0
        binding_max = float(default_binding_max)
    else:
        binding_min, binding_max = np.asarray(binding_range, dtype=float) / _unit_scale(units)
    if binding_max <= binding_min:
        raise ValueError("binding_range must be increasing.")
    return -np.linspace(binding_min, binding_max, int(npoints))


def _with_temporary_eta(gw, eta):
    class _EtaContext:
        def __enter__(self_inner):
            self_inner.old_eta = gw.eta
            if eta is not None:
                gw.eta = float(eta)
            return gw

        def __exit__(self_inner, exc_type, exc, tb):
            gw.eta = self_inner.old_eta
            return False

    return _EtaContext()


def _local_peak_indices(values, threshold_rel=0.05, min_distance=1):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if values.size == 0:
        return np.array([], dtype=int)

    vmax = float(np.max(values))
    if vmax <= 0.0:
        return np.array([], dtype=int)
    threshold = float(threshold_rel) * vmax
    min_distance = max(1, int(min_distance))

    candidates = []
    for idx, value in enumerate(values):
        left = values[idx - 1] if idx > 0 else -np.inf
        right = values[idx + 1] if idx + 1 < values.size else -np.inf
        if value >= threshold and value >= left and value >= right and (value > left or value > right):
            candidates.append(idx)

    selected = []
    for idx in sorted(candidates, key=lambda item: values[item], reverse=True):
        if all(abs(idx - prev) >= min_distance for prev in selected):
            selected.append(idx)
    return np.asarray(selected, dtype=int)


def spectral_peaks(
    result,
    source="signal",
    units="ev",
    threshold_rel=0.05,
    min_distance=1,
    max_peaks=None,
):
    """Find peaks in ``PESSpectralResult`` data."""
    key = str(source).lower().replace("-", "_")
    if key in {"signal", "pes", "total"}:
        if result.signal is None:
            raise ValueError("source='signal' requires a result with a PES signal.")
        traces = np.asarray(result.signal, dtype=float)[None, :]
        orbitals = np.array([-1], dtype=int)
        source_name = "signal"
    elif key in {"spectral", "spectral_function", "orbital"}:
        traces = np.asarray(result.spectral_function, dtype=float)
        orbitals = np.asarray(result.orbitals, dtype=int)
        source_name = "spectral_function"
    else:
        raise ValueError("source must be 'signal' or 'spectral_function'.")

    binding = np.asarray(result.binding_energies, dtype=float) * _unit_scale(units)
    peak_rows = []
    for row, orbital in zip(traces, orbitals):
        for idx in _local_peak_indices(row, threshold_rel=threshold_rel, min_distance=min_distance):
            peak_rows.append((int(orbital), float(binding[idx]), float(row[idx]), int(idx)))

    peak_rows.sort(key=lambda item: item[2], reverse=True)
    if max_peaks is not None:
        peak_rows = peak_rows[: int(max_peaks)]

    if not peak_rows:
        return PESPeakResult(
            orbitals=np.array([], dtype=int),
            binding_energies=np.array([], dtype=float),
            intensities=np.array([], dtype=float),
            indices=np.array([], dtype=int),
            source=source_name,
            units=units,
        )

    return PESPeakResult(
        orbitals=np.asarray([row[0] for row in peak_rows], dtype=int),
        binding_energies=np.asarray([row[1] for row in peak_rows], dtype=float),
        intensities=np.asarray([row[2] for row in peak_rows], dtype=float),
        indices=np.asarray([row[3] for row in peak_rows], dtype=int),
        source=source_name,
        units=units,
    )


class PES:
    """Plane-wave photoelectron spectrum from a GW object."""

    def __init__(self, gw, photon_energy=None, units="ev", origin=None):
        self.gw = gw
        self.photon_energy = None if photon_energy is None else _as_au(photon_energy, units)
        self.units = units
        self.origin = None if origin is None else np.asarray(origin, dtype=float)
        if self.origin is not None and self.origin.shape != (3,):
            raise ValueError("origin must be a length-3 Cartesian vector.")
        self.result = None

    def _store_result(self, result):
        self.result = result
        for field in fields(result):
            setattr(self, field.name, getattr(result, field.name))
        return result

    def _default_origin(self):
        mol = self.gw.mol
        if self.origin is not None:
            return self.origin
        return nuclear_center_of_mass(mol)

    def _occupied_orbitals(self, orbitals=None):
        nocc = int(self.gw.nocc // 2)
        if orbitals is None:
            return np.arange(nocc, dtype=int)
        orbitals = np.atleast_1d(np.asarray(orbitals, dtype=int))
        if np.any(orbitals < 0) or np.any(orbitals >= len(self.gw.e_qp)):
            raise IndexError("One or more orbital indices are out of range.")
        return orbitals

    def _transition_dipole_from_coefficients(self, coefficients, kinetic_energy, direction, origin=None):
        kinetic_energy = float(kinetic_energy)
        if kinetic_energy <= 0.0:
            return np.zeros(3, dtype=np.complex128)
        direction = _normalize(direction)
        kvec = np.sqrt(2.0 * kinetic_energy) * direction
        ao_dipoles = ao_plane_wave_dipoles(
            self.gw.mol,
            kvec,
            origin=self._default_origin() if origin is None else origin,
        )
        coefficients = np.asarray(coefficients)
        return np.einsum("u,ux->x", coefficients, ao_dipoles, optimize=True)

    def _mo_transition_dipoles(self, orbitals, kinetic_energy, direction, origin=None):
        kinetic_energy = float(kinetic_energy)
        orbitals = np.atleast_1d(np.asarray(orbitals, dtype=int))
        if kinetic_energy <= 0.0:
            return np.zeros((len(orbitals), 3), dtype=np.complex128)
        direction = _normalize(direction)
        kvec = np.sqrt(2.0 * kinetic_energy) * direction
        ao_dipoles = ao_plane_wave_dipoles(
            self.gw.mol,
            kvec,
            origin=self._default_origin() if origin is None else origin,
        )
        mo_coeff = _mo_coefficients_for_gw(self.gw)[:, orbitals]
        return np.einsum("up,ux->px", mo_coeff, ao_dipoles, optimize=True)

    def transition_dipole(self, orbital, kinetic_energy, direction, origin=None, dyson="mo"):
        """Vector ``<exp(i k.r)|(r-origin)|Dyson>`` for one plane-wave direction."""
        mo_coeff = _mo_coefficients_for_gw(self.gw)
        key = _dyson_key(dyson)
        if key == "matrix":
            coefficients = self.gw.dyson_orbital(int(orbital)).ao_coefficients
            return self._transition_dipole_from_coefficients(
                coefficients,
                kinetic_energy,
                direction,
                origin=origin,
            )

        coefficients = mo_coeff[:, int(orbital)]
        if key == "qp":
            z = quasiparticle_weights(self.gw, orbitals=[int(orbital)])[0]
            if z < 0.0:
                raise ValueError("Cannot build qp Dyson amplitudes from negative quasiparticle weights.")
            coefficients = np.sqrt(z) * coefficients
        return self._transition_dipole_from_coefficients(
            coefficients,
            kinetic_energy,
            direction,
            origin=origin,
        )

    def transition_moment(self, orbital, kinetic_energy, direction, polarization, origin=None, dyson="mo"):
        """Analytic plane-wave length-gauge transition moment for one MO."""
        polarization = np.asarray(polarization, dtype=float)
        mo_dipole = self.transition_dipole(
            orbital,
            kinetic_energy,
            direction,
            origin=origin,
            dyson=dyson,
        )
        return np.dot(polarization, mo_dipole)

    def orientation_averaged_intensity(
        self,
        orbital,
        kinetic_energy,
        ndirections=50,
        qp_weight=1.0,
        origin=None,
        dyson_coefficients=None,
    ):
        """Average over emission directions with exact transverse polarization average."""
        kinetic_energy = float(kinetic_energy)
        if kinetic_energy <= 0.0:
            return 0.0, 0.0j
        total = 0.0
        moment_accum = 0.0j
        directions = _fibonacci_sphere(ndirections)
        if dyson_coefficients is None:
            mo_coeff = _mo_coefficients_for_gw(self.gw)
            dyson_coefficients = mo_coeff[:, int(orbital)]
        for direction in directions:
            direction = _normalize(direction)
            mo_dipole = self._transition_dipole_from_coefficients(
                dyson_coefficients,
                kinetic_energy,
                direction,
                origin=origin,
            )
            intensity = _transverse_polarization_average(mo_dipole, direction)
            total += intensity
            moment_accum += np.sqrt(intensity)
        nsample = len(directions)
        return float(qp_weight) * total / nsample, moment_accum / nsample

    def arpes(
        self,
        photon_energy=None,
        units=None,
        orbitals=None,
        direction=(0.0, 0.0, 1.0),
        polarization=None,
        qp_weight=None,
        intensity="matrix_element",
        dyson="mo",
    ):
        """
        Compute angle-resolved PES sticks for one photoelectron direction.

        If ``polarization`` is ``None``, the intensity is averaged analytically
        over the two polarizations transverse to ``direction``. Otherwise the
        fixed-polarization signal ``|e.D(k)|^2`` is used.
        """
        if self.gw.e_qp is None:
            raise ValueError("Run GW before computing a photoelectron spectrum.")
        units = self.units if units is None else units
        photon_energy = self.photon_energy if photon_energy is None else _as_au(photon_energy, units)
        if photon_energy is None:
            raise ValueError("photon_energy is required for photoelectron intensities.")

        direction = _normalize(direction)
        if polarization is None:
            pol = None
            averaging = "angle_resolved_transverse_polarization"
        else:
            pol = _normalize(polarization)
            averaging = "angle_resolved_fixed_polarization"

        orbitals = self._occupied_orbitals(orbitals)
        binding = -np.asarray(self.gw.e_qp, dtype=float)[orbitals]
        kinetic = photon_energy - binding
        dyson_kind, dyson_amplitudes, intensity_weights, weights = _dyson_amplitudes(
            self.gw,
            orbitals,
            dyson=dyson,
            qp_weight=qp_weight,
        )

        matrix_strengths = np.zeros(len(orbitals), dtype=float)
        moments = np.zeros(len(orbitals), dtype=np.complex128)
        origin = self._default_origin()
        mo_coeff = _mo_coefficients_for_gw(self.gw)
        dyson_results = {}
        if dyson_kind == "matrix":
            e_rpa, t_rpa = self.gw.rpa(method=self.gw.screening)
            for orbital in orbitals:
                result = self.gw.dyson_orbital(int(orbital), e_rpa=e_rpa, t_rpa=t_rpa)
                dyson_results[int(orbital)] = result
                weights[np.where(orbitals == orbital)[0][0]] = result.qp_weight
        for pos, orbital in enumerate(orbitals):
            if dyson_kind == "matrix":
                dyson_coefficients = dyson_results[int(orbital)].ao_coefficients
            else:
                dyson_coefficients = dyson_amplitudes[pos] * mo_coeff[:, int(orbital)]
            dipole = self._transition_dipole_from_coefficients(
                dyson_coefficients,
                kinetic[pos],
                direction,
                origin=origin,
            )
            if pol is None:
                strength = _transverse_polarization_average(dipole, direction)
                moment = np.sqrt(strength)
            else:
                moment = np.dot(pol, dipole)
                strength = abs(moment) ** 2
            matrix_strengths[pos] = intensity_weights[pos] * float(strength)
            moments[pos] = moment

        intensities, intensity_kind, intensity_units, prefactors = _apply_intensity_convention(
            matrix_strengths,
            photon_energy,
            kinetic,
            intensity,
            angular_integrated=False,
        )

        result = PESResult(
            orbitals=orbitals,
            binding_energies=binding,
            kinetic_energies=kinetic,
            intensities=intensities,
            qp_weights=weights,
            transition_moments=moments,
            photon_energy=float(photon_energy),
            origin=origin.copy(),
            direction=direction.copy(),
            polarization=None if pol is None else pol.copy(),
            averaging=averaging,
            intensity_kind=intensity_kind,
            intensity_units=intensity_units,
            cross_section_prefactors=prefactors,
            dyson_kind=dyson_kind,
        )
        return self._store_result(result)

    def run(
        self,
        photon_energy=None,
        units=None,
        orbitals=None,
        ndirections=50,
        qp_weight=None,
        intensity="matrix_element",
        dyson="mo",
    ):
        """Compute PES sticks for occupied GW orbitals."""
        if self.gw.e_qp is None:
            raise ValueError("Run GW before computing a photoelectron spectrum.")
        units = self.units if units is None else units
        photon_energy = self.photon_energy if photon_energy is None else _as_au(photon_energy, units)
        if photon_energy is None:
            raise ValueError("photon_energy is required for photoelectron intensities.")

        orbitals = self._occupied_orbitals(orbitals)
        binding = -np.asarray(self.gw.e_qp, dtype=float)[orbitals]
        kinetic = photon_energy - binding
        dyson_kind, dyson_amplitudes, intensity_weights, weights = _dyson_amplitudes(
            self.gw,
            orbitals,
            dyson=dyson,
            qp_weight=qp_weight,
        )

        matrix_strengths = np.zeros(len(orbitals), dtype=float)
        moments = np.zeros(len(orbitals), dtype=np.complex128)
        origin = self._default_origin()
        mo_coeff = _mo_coefficients_for_gw(self.gw)
        dyson_results = {}
        if dyson_kind == "matrix":
            e_rpa, t_rpa = self.gw.rpa(method=self.gw.screening)
            for orbital in orbitals:
                result = self.gw.dyson_orbital(int(orbital), e_rpa=e_rpa, t_rpa=t_rpa)
                dyson_results[int(orbital)] = result
                weights[np.where(orbitals == orbital)[0][0]] = result.qp_weight
        for pos, orbital in enumerate(orbitals):
            if dyson_kind == "matrix":
                dyson_coefficients = dyson_results[int(orbital)].ao_coefficients
            else:
                dyson_coefficients = dyson_amplitudes[pos] * mo_coeff[:, int(orbital)]
            matrix_strengths[pos], moments[pos] = self.orientation_averaged_intensity(
                orbital,
                kinetic[pos],
                ndirections=ndirections,
                qp_weight=intensity_weights[pos],
                origin=origin,
                dyson_coefficients=dyson_coefficients,
            )

        intensities, intensity_kind, intensity_units, prefactors = _apply_intensity_convention(
            matrix_strengths,
            photon_energy,
            kinetic,
            intensity,
            angular_integrated=True,
        )

        result = PESResult(
            orbitals=orbitals,
            binding_energies=binding,
            kinetic_energies=kinetic,
            intensities=intensities,
            qp_weights=weights,
            transition_moments=moments,
            photon_energy=float(photon_energy),
            origin=origin.copy(),
            intensity_kind=intensity_kind,
            intensity_units=intensity_units,
            cross_section_prefactors=prefactors,
            dyson_kind=dyson_kind,
        )
        return self._store_result(result)

    def spectral_function(
        self,
        omega_grid=None,
        binding_grid=None,
        units="ev",
        orbitals=None,
        npoints=1000,
        binding_range=None,
        eta=None,
    ):
        """
        Compute diagonal GW spectral functions ``A_p(omega)`` on an energy grid.

        The returned ``omega`` and ``binding_energies`` are stored in atomic
        units. Use ``units`` only to interpret user-provided grids/ranges.
        """
        if self.gw.e_qp is None:
            raise ValueError("Run GW before computing a spectral function.")
        orbitals = self._occupied_orbitals(orbitals)
        occupied_binding = -np.asarray(self.gw.e_qp, dtype=float)[orbitals]
        default_binding_max = max(2.0, float(np.max(occupied_binding) + 1.0))
        omega = _energy_grid_from_inputs(
            omega_grid=omega_grid,
            binding_grid=binding_grid,
            units=units,
            npoints=npoints,
            binding_range=binding_range,
            default_binding_max=default_binding_max,
        )
        if omega.ndim != 1:
            raise ValueError("Energy grid must be one-dimensional.")

        with _with_temporary_eta(self.gw, eta):
            e_rpa, t_rpa = self.gw.rpa(method=self.gw.screening)
            spectral = np.zeros((len(orbitals), len(omega)), dtype=float)
            for pos, orbital in enumerate(orbitals):
                p = 2 * int(orbital)
                sigma_c, sigma_x = self.gw.sigma(p, p, omega, e_rpa, t_rpa)
                sigma_c = np.asarray(sigma_c, dtype=np.complex128)
                sigma_x = np.asarray(sigma_x, dtype=np.complex128)
                sign_occ = 1.0 if int(orbital) < int(self.gw.nocc // 2) else -1.0
                denominator = (
                    omega
                    - self.gw.e_mf[p]
                    - 1j * self.gw.eta * sign_occ
                    - (sigma_c + sigma_x - self.gw.v_mf[p, p])
                )
                green = 1.0 / denominator
                spectral[pos] = np.clip(sign_occ * green.imag / np.pi, 0.0, None)

        return PESSpectralResult(
            orbitals=orbitals,
            omega=omega,
            binding_energies=-omega,
            spectral_function=spectral,
            origin=self._default_origin().copy(),
        )

    def spectral_matrix(
        self,
        omega_grid=None,
        binding_grid=None,
        units="ev",
        orbitals=None,
        npoints=1000,
        binding_range=None,
        eta=None,
        spin="alpha",
    ):
        """
        Compute the full MO-subspace GW spectral matrix ``A_pq(omega)``.

        The matrix is formed from ``G(omega) = [omega - eps - Sigma(omega)
        + v_mf]^{-1}`` in the selected spatial-orbital subspace.  The returned
        ``spectral_matrix`` has shape ``(norb, norb, nomega)``.
        """
        if self.gw.e_qp is None:
            raise ValueError("Run GW before computing a spectral matrix.")
        orbitals = self._occupied_orbitals(orbitals)
        occupied_binding = -np.asarray(self.gw.e_qp, dtype=float)[orbitals]
        default_binding_max = max(2.0, float(np.max(occupied_binding) + 1.0))
        omega = _energy_grid_from_inputs(
            omega_grid=omega_grid,
            binding_grid=binding_grid,
            units=units,
            npoints=npoints,
            binding_range=binding_range,
            default_binding_max=default_binding_max,
        )
        if omega.ndim != 1:
            raise ValueError("Energy grid must be one-dimensional.")

        signs = np.where(orbitals < int(self.gw.nocc // 2), 1.0, -1.0)
        if not np.all(signs == signs[0]):
            raise ValueError("spectral_matrix requires all orbitals to share occupied/virtual character.")
        sign = float(signs[0])
        offset = _spin_offset(spin)
        spin_orbitals = 2 * np.asarray(orbitals, dtype=int) + offset
        eps = np.asarray(self.gw.e_mf, dtype=float)[spin_orbitals]

        with _with_temporary_eta(self.gw, eta):
            e_rpa, t_rpa = self.gw.rpa(method=self.gw.screening)
            spectral_matrix = np.zeros((len(orbitals), len(orbitals), len(omega)), dtype=np.complex128)
            ident = np.eye(len(orbitals), dtype=np.complex128)
            for epos, energy in enumerate(omega):
                sigma_c, sigma_x, v_mf = _self_energy_blocks(
                    self.gw,
                    energy,
                    e_rpa,
                    t_rpa,
                    orbitals,
                    spin=spin,
                )
                denominator = (
                    (energy - 1j * self.gw.eta * sign) * ident
                    - np.diag(eps)
                    - (sigma_c + sigma_x - v_mf)
                )
                green = np.linalg.inv(denominator)
                spectral_matrix[:, :, epos] = sign * (green - green.T.conjugate()) / (2j * np.pi)

        spectral_function = np.real(np.diagonal(spectral_matrix, axis1=0, axis2=1).T)
        spectral_function = np.clip(spectral_function, 0.0, None)
        if np.linalg.norm(spectral_matrix.imag) < 1.0e-12:
            spectral_matrix = spectral_matrix.real

        return PESSpectralResult(
            orbitals=orbitals,
            omega=omega,
            binding_energies=-omega,
            spectral_function=spectral_function,
            spectral_matrix=spectral_matrix,
            origin=self._default_origin().copy(),
            approximation="matrix",
        )

    def _spectral_pes_signal_diagonal(
        self,
        spec,
        photon_energy,
        origin,
        ndirections,
        direction=None,
        polarization=None,
    ):
        kinetic = photon_energy - spec.binding_energies
        matrix_signal = np.zeros_like(spec.spectral_function)
        directions = [_normalize(direction)] if direction is not None else _fibonacci_sphere(ndirections)
        pol = None if polarization is None else _normalize(polarization)

        for epos, kinetic_energy in enumerate(kinetic):
            if kinetic_energy <= 0.0:
                continue
            for sample_direction in directions:
                dipoles = self._mo_transition_dipoles(
                    spec.orbitals,
                    kinetic_energy,
                    sample_direction,
                    origin=origin,
                )
                if pol is None:
                    projection = np.eye(3) - np.outer(sample_direction, sample_direction)
                    strengths = 0.5 * np.real(
                        np.einsum("px,xy,py->p", dipoles.conjugate(), projection, dipoles, optimize=True)
                    )
                else:
                    amplitudes = dipoles @ pol
                    strengths = np.abs(amplitudes) ** 2
                matrix_signal[:, epos] += spec.spectral_function[:, epos] * strengths

        matrix_signal /= len(directions)
        return np.sum(matrix_signal, axis=0)

    def _spectral_pes_signal_matrix(
        self,
        spec,
        photon_energy,
        origin,
        ndirections,
        direction=None,
        polarization=None,
    ):
        if spec.spectral_matrix is None:
            raise ValueError("approx='matrix' requires a spectral matrix.")
        kinetic = photon_energy - spec.binding_energies
        signal = np.zeros_like(kinetic, dtype=float)
        directions = [_normalize(direction)] if direction is not None else _fibonacci_sphere(ndirections)
        pol = None if polarization is None else _normalize(polarization)

        for epos, kinetic_energy in enumerate(kinetic):
            if kinetic_energy <= 0.0:
                continue
            spectral_matrix = np.asarray(spec.spectral_matrix[:, :, epos], dtype=np.complex128)
            for sample_direction in directions:
                dipoles = self._mo_transition_dipoles(
                    spec.orbitals,
                    kinetic_energy,
                    sample_direction,
                    origin=origin,
                )
                if pol is None:
                    projection = np.eye(3) - np.outer(sample_direction, sample_direction)
                    value = 0.0
                    for axis_a in range(3):
                        for axis_b in range(3):
                            if projection[axis_a, axis_b] == 0.0:
                                continue
                            value += projection[axis_a, axis_b] * np.vdot(
                                dipoles[:, axis_a],
                                spectral_matrix @ dipoles[:, axis_b],
                            ).real
                    signal[epos] += 0.5 * value
                else:
                    amplitudes = dipoles @ pol
                    signal[epos] += np.vdot(amplitudes, spectral_matrix @ amplitudes).real

        signal /= len(directions)
        return np.clip(signal, 0.0, None)

    def spectral_pes(
        self,
        photon_energy=None,
        omega_grid=None,
        binding_grid=None,
        units=None,
        orbitals=None,
        npoints=400,
        binding_range=None,
        eta=None,
        ndirections=50,
        direction=None,
        polarization=None,
        intensity="matrix_element",
        approx="diagonal",
    ):
        """
        Compute a full spectral PES signal from the GW spectral function.

        ``approx='diagonal'`` evaluates ``sum_p |d_p|^2 A_pp``.  ``approx='matrix'``
        builds the full spectral matrix and evaluates ``d^dagger A d`` in the
        selected orbital subspace.
        """
        if self.gw.e_qp is None:
            raise ValueError("Run GW before computing a spectral PES.")
        units = self.units if units is None else units
        photon_energy = self.photon_energy if photon_energy is None else _as_au(photon_energy, units)
        if photon_energy is None:
            raise ValueError("photon_energy is required for photoelectron intensities.")
        if direction is None and polarization is not None:
            raise ValueError("polarization requires an explicit photoelectron direction.")

        approx_key = _spectral_approx_key(approx)
        spec_fn = self.spectral_matrix if approx_key == "matrix" else self.spectral_function
        spec = spec_fn(
            omega_grid=omega_grid,
            binding_grid=binding_grid,
            units=units,
            orbitals=orbitals,
            npoints=npoints,
            binding_range=binding_range,
            eta=eta,
        )
        origin = self._default_origin()
        kinetic = photon_energy - spec.binding_energies
        if approx_key == "matrix":
            signal = self._spectral_pes_signal_matrix(
                spec,
                photon_energy,
                origin,
                ndirections,
                direction=direction,
                polarization=polarization,
            )
        else:
            signal = self._spectral_pes_signal_diagonal(
                spec,
                photon_energy,
                origin,
                ndirections,
                direction=direction,
                polarization=polarization,
            )

        angular_integrated = direction is None
        signal, intensity_kind, intensity_units, prefactors = _apply_intensity_convention(
            signal,
            photon_energy,
            kinetic,
            intensity,
            angular_integrated=angular_integrated,
        )
        averaging = (
            f"orientation_spectral_{approx_key}"
            if direction is None
            else f"angle_resolved_spectral_{approx_key}"
        )

        return PESSpectralResult(
            orbitals=spec.orbitals,
            omega=spec.omega,
            binding_energies=spec.binding_energies,
            spectral_function=spec.spectral_function,
            spectral_matrix=spec.spectral_matrix,
            signal=signal,
            photon_energy=float(photon_energy),
            origin=origin.copy(),
            direction=None if direction is None else np.asarray(direction, dtype=float).copy(),
            polarization=None if polarization is None else np.asarray(polarization, dtype=float).copy(),
            averaging=averaging,
            intensity_kind=intensity_kind,
            intensity_units=intensity_units,
            cross_section_prefactors=prefactors,
            approximation=approx_key,
        )

    def satellite_spectrum(self, *args, **kwargs):
        """Compatibility alias for :meth:`spectral_pes`."""
        kwargs.setdefault("approx", "diagonal")
        result = self.spectral_pes(*args, **kwargs)
        if result.averaging == "orientation_spectral_diagonal":
            result.averaging = "orientation_spectral_function"
        return result

    def spectrum(self, x=None, width=0.2, units="ev", lineshape="gaussian", result=None):
        """Broaden the PES stick spectrum over binding energy."""
        result = self.result if result is None else result
        if result is None:
            result = self.run()
        scale = _unit_scale(units)
        centers = np.asarray(result.binding_energies, dtype=float) * scale
        strengths = np.asarray(result.intensities, dtype=float)
        if centers.size == 0:
            raise ValueError("Cannot broaden an empty PES.")
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

    def plot(self, x=None, width=0.2, units="ev", lineshape="gaussian", ax=None, **kwargs):
        """Plot the broadened PES and return ``(ax, x, signal)``."""
        import matplotlib.pyplot as plt

        x, signal = self.spectrum(x=x, width=width, units=units, lineshape=lineshape)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(x, signal, **kwargs)
        if str(units).lower().startswith("ev"):
            ax.set_xlabel("Binding energy (eV)")
        else:
            ax.set_xlabel("Binding energy (hartree)")
        ylabel = "PES intensity"
        if self.result is not None:
            ylabel += f" ({self.result.intensity_units})"
        ax.set_ylabel(ylabel)
        return ax, x, signal
