"""Photoemission spectral functions from periodic diagonal GW."""

from dataclasses import dataclass

import numpy as np

from pyqed.qchem.fourier import gaussian_basis_ft_batch
from pyqed.units import au2ev, kelvin

from .response import KPointTransitionSpace
from .self_energy import (
    DiagonalSelfEnergyCache,
    _target_band_pairs,
    diagonal_correlation_self_energy,
)


@dataclass
class PeriodicPESPeakResult:
    """Peak positions extracted from a periodic GW spectral function."""

    targets: np.ndarray
    binding_energies: np.ndarray
    intensities: np.ndarray
    indices: np.ndarray
    source: str = "signal"
    units: str = "ev"

    @property
    def k_indices(self):
        return self.targets[:, 0]

    @property
    def band_indices(self):
        return self.targets[:, 1]


@dataclass
class PeriodicSpectralFunctionResult:
    """Band- and k-resolved diagonal periodic GW spectral function."""

    targets: np.ndarray
    kpoints: np.ndarray
    omega: np.ndarray
    binding_energies: np.ndarray
    energy_reference: float
    energy_reference_label: str
    spectral_function: np.ndarray
    sigma_c: np.ndarray
    green_function: np.ndarray
    occupations: np.ndarray
    target_weights: np.ndarray
    signal: np.ndarray
    eta: float
    broadening: float
    info: dict
    units: str = "au"
    intensity_kind: str = "spectral_function"
    intensity_units: str = "1/Ha"

    @property
    def k_indices(self):
        return self.targets[:, 0]

    @property
    def band_indices(self):
        return self.targets[:, 1]

    def peaks(
        self,
        source="signal",
        units="ev",
        threshold_rel=0.05,
        min_distance=1,
        max_peaks=None,
    ):
        """Find local maxima in the integrated signal or target spectra."""

        return periodic_spectral_peaks(
            self,
            source=source,
            units=units,
            threshold_rel=threshold_rel,
            min_distance=min_distance,
            max_peaks=max_peaks,
        )


@dataclass
class PeriodicPhotoemissionResult:
    """Matrix-element-weighted periodic GW photoemission spectrum."""

    spectral_result: PeriodicSpectralFunctionResult
    binding_energies: np.ndarray
    binding_energies_fermi: np.ndarray
    kinetic_energies: np.ndarray
    final_momenta: np.ndarray
    matrix_elements: np.ndarray
    matrix_strengths: np.ndarray
    momentum_weights: np.ndarray
    fermi_factors: np.ndarray
    target_intensity: np.ndarray
    raw_signal: np.ndarray
    signal: np.ndarray
    photon_energy: float
    work_function: float
    inner_potential: float
    temperature: float
    energy_resolution: float
    direction: np.ndarray
    polarization: object
    surface_normal: np.ndarray
    info: dict
    units: str = "au"
    intensity_kind: str = "plane_wave_velocity_gauge"
    intensity_units: str = "arb."

    @property
    def targets(self):
        return self.spectral_result.targets

    @property
    def k_indices(self):
        return self.targets[:, 0]

    @property
    def band_indices(self):
        return self.targets[:, 1]

    def peaks(
        self,
        source="signal",
        units="ev",
        threshold_rel=0.05,
        min_distance=1,
        max_peaks=None,
    ):
        """Find peaks in the detector-broadened or target-resolved intensity."""

        return periodic_photoemission_peaks(
            self,
            source=source,
            units=units,
            threshold_rel=threshold_rel,
            min_distance=min_distance,
            max_peaks=max_peaks,
        )


def _unit_scale(units):
    key = str(units).strip().lower()
    if key in {"au", "ha", "hartree"}:
        return 1.0
    if key in {"ev", "electronvolt", "electronvolts"}:
        return float(au2ev)
    raise ValueError("units must be 'au' or 'ev'.")


def _qp_energy_table(ref, e_qp):
    if e_qp is None:
        return np.asarray(ref.mo_energy, dtype=float)
    table = np.asarray(e_qp, dtype=float)
    if table.ndim == 1 and ref.nkpts == 1:
        table = table.reshape(1, -1)
    if table.shape != ref.mo_energy.shape:
        raise ValueError(
            "e_qp must have shape matching mo_energy "
            f"{ref.mo_energy.shape}; got {table.shape}."
        )
    if np.any(~np.isfinite(table)):
        raise ValueError("e_qp must contain only finite values.")
    return table


def _spectral_targets(ref, bands, occupied_only):
    pairs, _mask, normalized_bands = _target_band_pairs(ref, bands)
    if occupied_only:
        pairs = tuple(
            (k_index, band_index)
            for k_index, band_index in pairs
            if ref.mo_occ[k_index, band_index] >= 2.0 - ref.occupation_tol
        )
    if not pairs:
        raise ValueError("No bands remain in the requested spectral target set.")

    occupations = np.asarray(
        [ref.mo_occ[k_index, band_index] for k_index, band_index in pairs],
        dtype=float,
    )
    fractional = (occupations > ref.occupation_tol) & (
        occupations < 2.0 - ref.occupation_tol
    )
    if np.any(fractional):
        raise NotImplementedError(
            "Fractional occupations are not yet supported by periodic GW spectra."
        )
    return pairs, occupations, normalized_bands


def _energy_reference(ref, e_qp, value, units):
    if not isinstance(value, str):
        reference = float(value) / _unit_scale(units)
        if not np.isfinite(reference):
            raise ValueError("energy_reference must be finite.")
        return reference, "explicit"

    key = value.strip().lower().replace("-", "_")
    occupied = ref.mo_occ >= 2.0 - ref.occupation_tol
    virtual = ref.mo_occ <= ref.occupation_tol
    if key in {"zero", "vacuum", "absolute"}:
        return 0.0, "zero"
    if not np.any(occupied):
        raise ValueError("The reference has no occupied bands.")
    vbm = float(np.max(e_qp[occupied]))
    if key in {"vbm", "valence", "valence_band_maximum"}:
        return vbm, "vbm"
    if key in {"fermi", "fermi_level", "midgap"}:
        if np.any(virtual):
            cbm = float(np.min(e_qp[virtual]))
            if cbm > vbm:
                return 0.5 * (vbm + cbm), "fermi_midgap"
        return vbm, "fermi"
    raise ValueError(
        "energy_reference must be 'vbm', 'fermi', 'zero', or a numeric value."
    )


def _energy_grid(
    targets,
    e_qp,
    energy_reference,
    omega_grid,
    binding_grid,
    units,
    npoints,
    binding_range,
):
    if omega_grid is not None and binding_grid is not None:
        raise ValueError("Provide only one of omega_grid or binding_grid.")
    scale = _unit_scale(units)
    if omega_grid is not None:
        omega = np.asarray(omega_grid, dtype=float) / scale
    elif binding_grid is not None:
        binding = np.asarray(binding_grid, dtype=float) / scale
        omega = energy_reference - binding
    else:
        npoints = int(npoints)
        if npoints < 2:
            raise ValueError("npoints must be at least two.")
        if binding_range is None:
            target_binding = np.asarray(
                [
                    energy_reference - e_qp[k_index, band_index]
                    for k_index, band_index in targets
                ],
                dtype=float,
            )
            margin = 5.0 / float(au2ev)
            binding_min = max(0.0, float(np.min(target_binding) - margin))
            binding_max = max(
                binding_min + 10.0 / float(au2ev),
                float(np.max(target_binding) + margin),
            )
        else:
            limits = np.asarray(binding_range, dtype=float)
            if limits.shape != (2,):
                raise ValueError("binding_range must contain two values.")
            binding_min, binding_max = limits / scale
        if binding_max <= binding_min:
            raise ValueError("binding_range must be increasing.")
        binding = np.linspace(binding_min, binding_max, npoints)
        omega = energy_reference - binding

    if omega.ndim != 1:
        raise ValueError("The energy grid must be one-dimensional.")
    if omega.size < 2:
        raise ValueError("The energy grid must contain at least two points.")
    if np.any(~np.isfinite(omega)):
        raise ValueError("The energy grid must contain only finite values.")
    return omega, energy_reference - omega


def _local_peak_indices(values, threshold_rel=0.05, min_distance=1):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if values.size == 0 or float(np.max(values)) <= 0.0:
        return np.array([], dtype=int)

    threshold = float(threshold_rel) * float(np.max(values))
    min_distance = max(1, int(min_distance))
    candidates = []
    for index, value in enumerate(values):
        left = values[index - 1] if index > 0 else -np.inf
        right = values[index + 1] if index + 1 < values.size else -np.inf
        if value >= threshold and value >= left and value >= right and (
            value > left or value > right
        ):
            candidates.append(index)

    selected = []
    for index in sorted(candidates, key=lambda item: values[item], reverse=True):
        if all(abs(index - previous) >= min_distance for previous in selected):
            selected.append(index)
    return np.asarray(selected, dtype=int)


def periodic_spectral_peaks(
    result,
    source="signal",
    units="ev",
    threshold_rel=0.05,
    min_distance=1,
    max_peaks=None,
):
    """Find peaks in a :class:`PeriodicSpectralFunctionResult`."""

    key = str(source).strip().lower().replace("-", "_")
    if key in {"signal", "pes", "total", "integrated"}:
        traces = np.asarray(result.signal, dtype=float)[None, :]
        targets = np.asarray([[-1, -1]], dtype=int)
        source_name = "signal"
    elif key in {"spectral", "spectral_function", "band", "target"}:
        traces = np.asarray(result.spectral_function, dtype=float)
        targets = np.asarray(result.targets, dtype=int)
        source_name = "spectral_function"
    else:
        raise ValueError("source must be 'signal' or 'spectral_function'.")

    binding = np.asarray(result.binding_energies, dtype=float) * _unit_scale(units)
    rows = []
    for trace, target in zip(traces, targets):
        indices = _local_peak_indices(
            trace,
            threshold_rel=threshold_rel,
            min_distance=min_distance,
        )
        for index in indices:
            rows.append(
                (
                    np.asarray(target, dtype=int),
                    float(binding[index]),
                    float(trace[index]),
                    int(index),
                )
            )
    rows.sort(key=lambda row: row[2], reverse=True)
    if max_peaks is not None:
        rows = rows[: int(max_peaks)]

    return PeriodicPESPeakResult(
        targets=np.asarray([row[0] for row in rows], dtype=int).reshape(-1, 2),
        binding_energies=np.asarray([row[1] for row in rows], dtype=float),
        intensities=np.asarray([row[2] for row in rows], dtype=float),
        indices=np.asarray([row[3] for row in rows], dtype=int),
        source=source_name,
        units=units,
    )


def periodic_spectral_function(
    space,
    omega_grid=None,
    binding_grid=None,
    units="ev",
    bands=None,
    occupied_only=True,
    npoints=1000,
    binding_range=None,
    energy_reference="vbm",
    e_qp=None,
    q_indices=None,
    eta=1.0e-2,
    broadening=None,
    direct_scale=2.0,
    coulomb_component="reciprocal_ewald_lr",
    g2_tol=1.0e-16,
    thresh=1.0e-10,
    energy_table=None,
    cache=None,
    intermediate_bands=None,
    finite_size_correction=False,
    finite_size_q_magnitude=1.0e-3,
    finite_size_q_direction=(1.0, 0.0, 0.0),
    finite_size_head_method="auto",
    spin_degeneracy=2.0,
):
    r"""Compute diagonal periodic GW photoemission spectral functions.

    The current periodic GW implementation uses a Hartree-Fock reference, so
    the static exchange contribution is already present in the reference
    eigenvalue. For target ``(n, k)`` the time-ordered diagonal Green function
    is formed from the exact RPA-pole correlation self-energy as

    ``G = 1 / (omega - eps_nk - i s eta - Sigma_c(omega))``,

    where ``s=+1`` for an occupied band and ``s=-1`` for a virtual band. The
    corresponding positive branch is ``A = s Im(G) / pi``. ``eta`` and
    ``broadening`` are always specified in Hartree; ``units`` applies to the
    user-provided energy axes and numeric energy reference.

    ``bands`` follows the periodic GW target convention: a one-dimensional
    band list applies at every k point, while a dictionary maps selected k
    indices to band lists. By default only occupied targets are retained.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space)
    ref = space.reference
    e_qp = _qp_energy_table(ref, e_qp)
    targets, occupations, normalized_bands = _spectral_targets(
        ref,
        bands,
        bool(occupied_only),
    )
    reference_energy, reference_label = _energy_reference(
        ref,
        e_qp,
        energy_reference,
        units,
    )
    omega, binding = _energy_grid(
        targets,
        e_qp,
        reference_energy,
        omega_grid,
        binding_grid,
        units,
        npoints,
        binding_range,
    )

    eta = float(eta)
    if eta <= 0.0 or not np.isfinite(eta):
        raise ValueError("eta must be a positive finite value in Hartree.")
    if broadening is None:
        broadening = eta
    broadening = float(broadening)
    if broadening <= 0.0 or not np.isfinite(broadening):
        raise ValueError("broadening must be a positive finite value in Hartree.")
    spin_degeneracy = float(spin_degeneracy)
    if spin_degeneracy < 0.0 or not np.isfinite(spin_degeneracy):
        raise ValueError("spin_degeneracy must be a nonnegative finite value.")
    if cache is None:
        cache = DiagonalSelfEnergyCache()

    ntarget = len(targets)
    sigma_c = np.empty((ntarget, len(omega)), dtype=np.complex128)
    green = np.empty_like(sigma_c)
    spectral = np.empty((ntarget, len(omega)), dtype=float)
    normalized_q_indices = None
    finite_size_methods = set()
    for position, ((k_index, band_index), occupation) in enumerate(
        zip(targets, occupations)
    ):
        self_energy = diagonal_correlation_self_energy(
            space,
            k_index=k_index,
            band_index=band_index,
            omega=omega,
            q_indices=q_indices,
            eta=eta,
            direct_scale=direct_scale,
            coulomb_component=coulomb_component,
            g2_tol=g2_tol,
            thresh=thresh,
            energy_table=energy_table,
            cache=cache,
            intermediate_bands=intermediate_bands,
            finite_size_correction=finite_size_correction,
            finite_size_q_magnitude=finite_size_q_magnitude,
            finite_size_q_direction=finite_size_q_direction,
            finite_size_head_method=finite_size_head_method,
        )
        normalized_q_indices = np.asarray(self_energy.q_indices, dtype=int)
        if self_energy.finite_size_method is not None:
            finite_size_methods.add(str(self_energy.finite_size_method))
        sigma_c[position] = np.asarray(self_energy.sigma_c, dtype=np.complex128)
        branch_sign = 1.0 if occupation >= 2.0 - ref.occupation_tol else -1.0
        denominator = (
            omega
            - float(ref.mo_energy[k_index, band_index])
            - 1j * branch_sign * broadening
            - sigma_c[position]
        )
        green[position] = 1.0 / denominator
        spectral[position] = np.clip(
            branch_sign * green[position].imag / np.pi,
            0.0,
            None,
        )

    target_weights = np.full(
        ntarget,
        spin_degeneracy / float(ref.nkpts),
        dtype=float,
    )
    signal = np.einsum("t,tw->w", target_weights, spectral, optimize=True)
    target_array = np.asarray(targets, dtype=int).reshape(-1, 2)
    kpoints = np.asarray(ref.kpts[target_array[:, 0]], dtype=float)
    unique_k = np.unique(target_array[:, 0])
    return PeriodicSpectralFunctionResult(
        targets=target_array,
        kpoints=kpoints,
        omega=omega,
        binding_energies=binding,
        energy_reference=float(reference_energy),
        energy_reference_label=reference_label,
        spectral_function=spectral,
        sigma_c=sigma_c,
        green_function=green,
        occupations=occupations,
        target_weights=target_weights,
        signal=signal,
        eta=eta,
        broadening=broadening,
        info={
            "backend": "kpoint_diagonal_exact_pole_spectral_function",
            "pbc": True,
            "frequency_integration": "poles",
            "nkpts": int(ref.nkpts),
            "nband": int(ref.nband),
            "ntarget": int(ntarget),
            "bands": normalized_bands,
            "occupied_only": bool(occupied_only),
            "q_indices": normalized_q_indices,
            "coulomb_component": self_energy.coulomb_component,
            "direct_scale": float(direct_scale),
            "g2_tol": float(g2_tol),
            "thresh": float(thresh),
            "finite_size_correction": bool(finite_size_correction),
            "finite_size_methods": tuple(sorted(finite_size_methods)),
            "spin_degeneracy": spin_degeneracy,
            "complete_k_mesh": bool(len(unique_k) == ref.nkpts),
            "cache_sizes": cache.sizes(),
        },
    )


def _normalize_real_vector(vector, name):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite length-3 vector.")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero.")
    return vector / norm


def _normalize_polarization(polarization):
    if polarization is None:
        return None
    polarization = np.asarray(polarization, dtype=np.complex128)
    if polarization.shape != (3,) or np.any(~np.isfinite(polarization)):
        raise ValueError("polarization must be a finite length-3 vector.")
    norm = float(np.sqrt(np.vdot(polarization, polarization).real))
    if norm <= 0.0:
        raise ValueError("polarization must be nonzero.")
    return polarization / norm


def periodic_plane_wave_orbital_ft(reference, final_momenta, targets):
    """Return cell-normalized Bloch-orbital Fourier amplitudes."""

    final_momenta = np.asarray(final_momenta, dtype=float)
    if final_momenta.ndim != 2 or final_momenta.shape[1] != 3:
        raise ValueError("final_momenta must have shape (nenergy, 3).")
    targets = np.asarray(targets, dtype=int)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targets must have shape (ntarget, 2).")
    if np.any(targets[:, 0] < 0) or np.any(targets[:, 0] >= reference.nkpts):
        raise IndexError("targets contains an out-of-range k-point index.")
    if np.any(targets[:, 1] < 0) or np.any(targets[:, 1] >= reference.nband):
        raise IndexError("targets contains an out-of-range band index.")

    molecule = reference.cell.unit_molecule
    cartesian_basis, transform = molecule._cart_basis()
    ao_ft = gaussian_basis_ft_batch(
        cartesian_basis,
        final_momenta,
        transform=transform,
    )
    volume = abs(float(np.linalg.det(reference.cell.lattice_vectors)))
    if volume <= 0.0:
        raise ValueError("The periodic cell volume must be positive.")
    amplitudes = np.empty((len(targets), len(final_momenta)), dtype=np.complex128)
    for position, (k_index, band_index) in enumerate(targets):
        amplitudes[position] = (
            ao_ft @ reference.mo_coeff[int(k_index), :, int(band_index)]
        ) / np.sqrt(volume)
    return amplitudes


def periodic_plane_wave_velocity_matrix_elements(
    reference,
    final_momenta,
    targets,
    polarization=None,
):
    """Return plane-wave velocity-gauge matrix elements for Bloch targets."""

    final_momenta = np.asarray(final_momenta, dtype=float)
    orbital_ft = periodic_plane_wave_orbital_ft(
        reference,
        final_momenta,
        targets,
    )
    polarization = _normalize_polarization(polarization)
    if polarization is None:
        longitudinal = np.sqrt(
            np.einsum("wi,wi->w", final_momenta, final_momenta, optimize=True)
            / 3.0
        )
    else:
        longitudinal = final_momenta @ polarization
    return orbital_ft * longitudinal[None, :]


def _free_electron_final_momenta(
    kinetic_energy,
    direction,
    surface_normal,
    inner_potential,
):
    kinetic_energy = np.asarray(kinetic_energy, dtype=float)
    available = kinetic_energy > 0.0
    vacuum_magnitude = np.sqrt(2.0 * np.clip(kinetic_energy, 0.0, None))
    vacuum_momenta = vacuum_magnitude[:, None] * direction[None, :]
    normal_component = vacuum_momenta @ surface_normal
    parallel = vacuum_momenta - normal_component[:, None] * surface_normal[None, :]
    sign = 1.0 if float(np.dot(direction, surface_normal)) >= 0.0 else -1.0
    crystal_normal = sign * np.sqrt(
        np.clip(normal_component * normal_component + 2.0 * inner_potential, 0.0, None)
    )
    final_momenta = parallel + crystal_normal[:, None] * surface_normal[None, :]
    final_momenta[~available] = 0.0
    return final_momenta, available


def _reciprocal_shell(reference, shell):
    shell = int(shell)
    if shell < 0:
        raise ValueError("reciprocal_shell must be non-negative.")
    values = np.arange(-shell, shell + 1, dtype=int)
    gx, gy, gz = np.meshgrid(values, values, values, indexing="ij")
    coefficients = np.column_stack((gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)))
    return coefficients @ np.asarray(reference.reciprocal_vectors, dtype=float)


def _surface_momentum_weights(
    reference,
    final_momenta,
    targets,
    surface_normal,
    broadening,
    reciprocal_shell,
):
    if broadening is None:
        return np.ones((len(targets), len(final_momenta)), dtype=float)
    broadening = float(broadening)
    if broadening <= 0.0 or not np.isfinite(broadening):
        raise ValueError("momentum_broadening must be positive in bohr^-1.")
    projector = np.eye(3) - np.outer(surface_normal, surface_normal)
    reciprocal = _reciprocal_shell(reference, reciprocal_shell)
    weights = np.empty((len(targets), len(final_momenta)), dtype=float)
    for position, (k_index, _band_index) in enumerate(targets):
        centers = np.asarray(reference.kpts[int(k_index)], dtype=float) + reciprocal
        mismatch = final_momenta[:, None, :] - centers[None, :, :]
        mismatch_parallel = mismatch @ projector
        distance2 = np.min(
            np.einsum(
                "wgi,wgi->wg",
                mismatch_parallel,
                mismatch_parallel,
                optimize=True,
            ),
            axis=1,
        )
        weights[position] = np.exp(-0.5 * distance2 / (broadening * broadening))
    return weights


def _fermi_factors(binding_energy, temperature):
    binding_energy = np.asarray(binding_energy, dtype=float)
    temperature = float(temperature)
    if temperature < 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be a nonnegative finite value in kelvin.")
    if temperature == 0.0:
        factors = np.where(binding_energy > 0.0, 1.0, 0.0)
        factors[np.isclose(binding_energy, 0.0, atol=1.0e-14)] = 0.5
        return factors
    exponent = np.clip(-binding_energy / (temperature * float(kelvin)), -700.0, 700.0)
    return 1.0 / (np.exp(exponent) + 1.0)


def _gaussian_detector_convolution(values, energy_grid, fwhm):
    values = np.asarray(values, dtype=float)
    energy_grid = np.asarray(energy_grid, dtype=float)
    fwhm = float(fwhm)
    if fwhm <= 0.0:
        return values.copy()
    spacing = np.diff(energy_grid)
    if spacing.size == 0 or not np.allclose(spacing, spacing[0], rtol=1.0e-7, atol=1.0e-12):
        raise ValueError("Detector broadening requires a uniformly spaced energy grid.")
    sigma_energy = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_points = sigma_energy / abs(float(spacing[0]))
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(values, sigma_points, axis=-1, mode="nearest")


def periodic_photoemission_peaks(
    result,
    source="signal",
    units="ev",
    threshold_rel=0.05,
    min_distance=1,
    max_peaks=None,
):
    """Find peaks in a matrix-element-weighted periodic PES result."""

    key = str(source).strip().lower().replace("-", "_")
    if key in {"signal", "pes", "total", "integrated"}:
        traces = np.asarray(result.signal, dtype=float)[None, :]
        targets = np.asarray([[-1, -1]], dtype=int)
        source_name = "signal"
    elif key in {"target", "targets", "band", "target_intensity"}:
        traces = np.asarray(result.target_intensity, dtype=float)
        targets = np.asarray(result.targets, dtype=int)
        source_name = "target_intensity"
    else:
        raise ValueError("source must be 'signal' or 'target_intensity'.")

    binding = np.asarray(result.binding_energies_fermi, dtype=float) * _unit_scale(units)
    rows = []
    for trace, target in zip(traces, targets):
        for index in _local_peak_indices(
            trace,
            threshold_rel=threshold_rel,
            min_distance=min_distance,
        ):
            rows.append(
                (
                    np.asarray(target, dtype=int),
                    float(binding[index]),
                    float(trace[index]),
                    int(index),
                )
            )
    rows.sort(key=lambda row: row[2], reverse=True)
    if max_peaks is not None:
        rows = rows[: int(max_peaks)]
    return PeriodicPESPeakResult(
        targets=np.asarray([row[0] for row in rows], dtype=int).reshape(-1, 2),
        binding_energies=np.asarray([row[1] for row in rows], dtype=float),
        intensities=np.asarray([row[2] for row in rows], dtype=float),
        indices=np.asarray([row[3] for row in rows], dtype=int),
        source=source_name,
        units=units,
    )


def periodic_photoemission_spectrum(
    reference,
    spectral_result,
    photon_energy,
    work_function,
    units="ev",
    direction=(0.0, 0.0, 1.0),
    polarization=None,
    surface_normal=(0.0, 0.0, 1.0),
    inner_potential=0.0,
    temperature=0.0,
    energy_resolution=0.0,
    binding_offset=None,
    momentum_broadening=None,
    reciprocal_shell=1,
):
    r"""Compute a matrix-element-weighted free-electron-final-state PES.

    The detector kinetic energy is ``hnu - work_function - binding_energy``.
    Initial-state spectral weight is multiplied by a plane-wave velocity-gauge
    matrix element, optional surface-parallel momentum matching, and the Fermi
    occupation before Gaussian detector broadening. Energies follow ``units``;
    momentum broadening is specified in bohr^-1 and temperature in kelvin.
    """

    if not isinstance(spectral_result, PeriodicSpectralFunctionResult):
        raise TypeError("spectral_result must be a PeriodicSpectralFunctionResult.")
    direction = _normalize_real_vector(direction, "direction")
    surface_normal = _normalize_real_vector(surface_normal, "surface_normal")
    polarization = _normalize_polarization(polarization)
    scale = _unit_scale(units)
    photon_energy = float(photon_energy) / scale
    work_function = float(work_function) / scale
    inner_potential = float(inner_potential) / scale
    energy_resolution = float(energy_resolution) / scale
    if photon_energy <= 0.0 or not np.isfinite(photon_energy):
        raise ValueError("photon_energy must be positive.")
    if work_function < 0.0 or not np.isfinite(work_function):
        raise ValueError("work_function must be nonnegative.")
    if inner_potential < 0.0 or not np.isfinite(inner_potential):
        raise ValueError("inner_potential must be nonnegative.")
    if energy_resolution < 0.0 or not np.isfinite(energy_resolution):
        raise ValueError("energy_resolution must be nonnegative.")

    if binding_offset is None:
        if not str(spectral_result.energy_reference_label).startswith("fermi"):
            raise ValueError(
                "Experimental PES requires a Fermi-referenced spectral result or "
                "an explicit binding_offset."
            )
        binding_offset_au = 0.0
    else:
        binding_offset_au = float(binding_offset) / scale
    binding = np.asarray(spectral_result.binding_energies, dtype=float)
    binding_fermi = binding + binding_offset_au
    kinetic_energy = photon_energy - work_function - binding_fermi
    final_momenta, available = _free_electron_final_momenta(
        kinetic_energy,
        direction,
        surface_normal,
        inner_potential,
    )
    matrix_elements = periodic_plane_wave_velocity_matrix_elements(
        reference,
        final_momenta,
        spectral_result.targets,
        polarization=polarization,
    )
    matrix_strengths = np.abs(matrix_elements) ** 2
    momentum_weights = _surface_momentum_weights(
        reference,
        final_momenta,
        spectral_result.targets,
        surface_normal,
        momentum_broadening,
        reciprocal_shell,
    )
    momentum_weights[:, ~available] = 0.0
    fermi_factors = _fermi_factors(binding_fermi, temperature)
    target_raw = (
        np.asarray(spectral_result.spectral_function, dtype=float)
        * matrix_strengths
        * momentum_weights
        * fermi_factors[None, :]
    )
    target_intensity = _gaussian_detector_convolution(
        target_raw,
        binding_fermi,
        energy_resolution,
    )
    raw_signal = np.einsum(
        "t,tw->w",
        spectral_result.target_weights,
        target_raw,
        optimize=True,
    )
    signal = np.einsum(
        "t,tw->w",
        spectral_result.target_weights,
        target_intensity,
        optimize=True,
    )
    return PeriodicPhotoemissionResult(
        spectral_result=spectral_result,
        binding_energies=binding.copy(),
        binding_energies_fermi=binding_fermi,
        kinetic_energies=kinetic_energy,
        final_momenta=final_momenta,
        matrix_elements=matrix_elements,
        matrix_strengths=matrix_strengths,
        momentum_weights=momentum_weights,
        fermi_factors=fermi_factors,
        target_intensity=target_intensity,
        raw_signal=raw_signal,
        signal=signal,
        photon_energy=photon_energy,
        work_function=work_function,
        inner_potential=inner_potential,
        temperature=float(temperature),
        energy_resolution=energy_resolution,
        direction=direction,
        polarization=None if polarization is None else polarization.copy(),
        surface_normal=surface_normal,
        info={
            "backend": "periodic_gw_plane_wave_photoemission",
            "pbc": True,
            "initial_state": "diagonal_exact_pole_gw",
            "final_state": "free_electron_plane_wave",
            "gauge": "velocity",
            "matrix_element": "gaussian_ao_fourier",
            "polarization_average": bool(polarization is None),
            "momentum_conservation": (
                "surface_parallel_gaussian"
                if momentum_broadening is not None
                else "not_applied"
            ),
            "momentum_broadening_bohr_inverse": (
                None if momentum_broadening is None else float(momentum_broadening)
            ),
            "reciprocal_shell": int(reciprocal_shell),
            "available_energy_points": int(np.count_nonzero(available)),
            "energy_points": int(len(binding)),
            "binding_offset_ha": float(binding_offset_au),
            "experimental_effects": (
                "work_function",
                "fermi_occupation",
                "energy_resolution",
                "surface_parallel_momentum",
            ),
            "missing_effects": (
                "one_step_surface_final_state",
                "inelastic_mean_free_path",
                "extrinsic_losses",
                "detector_acceptance",
                "absolute_cross_section",
            ),
        },
    )


__all__ = [
    "PeriodicPESPeakResult",
    "PeriodicPhotoemissionResult",
    "PeriodicSpectralFunctionResult",
    "periodic_photoemission_peaks",
    "periodic_photoemission_spectrum",
    "periodic_plane_wave_orbital_ft",
    "periodic_plane_wave_velocity_matrix_elements",
    "periodic_spectral_function",
    "periodic_spectral_peaks",
]
