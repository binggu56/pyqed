"""Optical observables from q=0 periodic BSE and TDA roots."""

from dataclasses import dataclass

import numpy as np

from .finite_size import bloch_ao_gradient_matrices, cell_volume
from .response import KPointTransitionSpace


_HARTREE_TO_EV = 27.211386245988


@dataclass
class PeriodicBSEOpticalResult:
    """Polarization-resolved optical spectrum assembled from BSE roots."""

    energy_grid: np.ndarray
    excitation_energies: np.ndarray
    dielectric_imag: np.ndarray
    dielectric_tensor_imag: np.ndarray
    oscillator_strengths: np.ndarray
    line_strengths: np.ndarray
    transition_velocity: np.ndarray
    transition_dipoles: np.ndarray
    exciton_dipoles: np.ndarray
    polarization: np.ndarray | None
    broadening: float
    units: str
    lineshape: str
    metric: str
    q_index: int
    info: dict

    @property
    def signal(self):
        return self.dielectric_imag


@dataclass
class PeriodicBSEHaydockResult:
    """Optical spectrum from matrix-free Hermitian Lanczos recursion."""

    energy_grid: np.ndarray
    dielectric_imag: np.ndarray
    spectral_density: np.ndarray
    transition_velocity: np.ndarray
    transition_dipoles: np.ndarray
    polarization: np.ndarray | None
    lanczos_alpha: tuple
    lanczos_beta: tuple
    starting_norms: np.ndarray
    broadening: float
    units: str
    q_index: int
    info: dict

    @property
    def signal(self):
        return self.dielectric_imag


def _unit_scale(units):
    key = str(units).strip().lower()
    if key in {"ha", "hartree", "hartrees", "au", "a.u."}:
        return 1.0, "hartree"
    if key in {"ev", "electronvolt", "electronvolts"}:
        return _HARTREE_TO_EV, "ev"
    raise ValueError("units must be 'hartree' or 'ev'.")


def _normalize_polarization(polarization):
    if polarization is None:
        return None
    if isinstance(polarization, str):
        key = polarization.strip().lower()
        axes = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }
        try:
            polarization = axes[key]
        except KeyError as exc:
            raise ValueError("polarization string must be 'x', 'y', or 'z'.") from exc
    vector = np.asarray(polarization, dtype=np.complex128)
    if vector.shape != (3,):
        raise ValueError("polarization must be a length-3 vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("polarization must contain finite values.")
    norm = float(np.sqrt(np.vdot(vector, vector).real))
    if norm <= 0.0:
        raise ValueError("polarization must be nonzero.")
    return vector / norm


def _require_optical_q(space, q_index, tol=1.0e-10):
    q_index = space.normalize_q_index(q_index)
    qvec = np.asarray(space.qpts[q_index], dtype=float)
    if np.linalg.norm(qvec) > float(tol):
        raise ValueError("Optical BSE absorption requires the q=0 transition block.")
    transitions = space.transitions(q_index)
    if any(tr.k_index != tr.kq_index for tr in transitions):
        raise ValueError("The q=0 optical block must contain vertical k-to-k transitions.")
    return q_index, transitions


def periodic_transition_velocity_matrix_elements(space, q_index=0):
    """Return native canonical-momentum matrix elements for q=0 transitions.

    The returned row ``t`` is ``<v,k|-i grad|c,k>`` for the transition order
    stored by :class:`KPointTransitionSpace`. This all-electron Gaussian path
    does not include nonlocal pseudopotential commutator corrections.
    """

    if not isinstance(space, KPointTransitionSpace):
        space = KPointTransitionSpace(space, qpts="optical")
    q_index, transitions = _require_optical_q(space, q_index)
    reference = space.reference
    mf = reference._pbc_mf
    velocity = np.empty((len(transitions), 3), dtype=np.complex128)
    momentum_ao = {}
    for row, transition in enumerate(transitions):
        k_index = int(transition.k_index)
        if k_index not in momentum_ao:
            gradient = bloch_ao_gradient_matrices(mf, reference.kpts[k_index])
            momentum_ao[k_index] = -1j * gradient
        c_occ = reference.mo_coeff[k_index, :, int(transition.occ_band)]
        c_vir = reference.mo_coeff[k_index, :, int(transition.vir_band)]
        velocity[row] = np.einsum(
            "p,xpq,q->x",
            c_occ.conj(),
            momentum_ao[k_index],
            c_vir,
            optimize=True,
        )
    if not np.all(np.isfinite(velocity)):
        raise FloatingPointError("Periodic transition velocity contains non-finite values.")
    return velocity


def _root_amplitudes(result):
    vectors = result.vectors
    if vectors is None:
        raise ValueError("Optical absorption requires BSE vectors; use return_vectors=True.")
    vectors = np.asarray(vectors, dtype=np.complex128)
    ntrans = len(result.block.transition_energy)
    nroots = len(result.e)
    if result.metric == "tda":
        expected = (ntrans, nroots)
        if vectors.shape != expected:
            raise ValueError(f"TDA vectors must have shape {expected}; got {vectors.shape}.")
        return vectors
    if result.metric == "full":
        expected = (2 * ntrans, nroots)
        if vectors.shape != expected:
            raise ValueError(f"Full BSE vectors must have shape {expected}; got {vectors.shape}.")
        return vectors[:ntrans] + vectors[ntrans:]
    raise ValueError("BSE result metric must be 'tda' or 'full'.")


def _line_profile(grid, roots, width, lineshape):
    delta = grid[None, :] - roots[:, None]
    key = str(lineshape).strip().lower()
    if key in {"lorentz", "lorentzian"}:
        profile = width / (np.pi * (delta * delta + width * width))
        return profile, "lorentzian"
    if key in {"gauss", "gaussian"}:
        profile = np.exp(-0.5 * (delta / width) ** 2)
        profile /= np.sqrt(2.0 * np.pi) * width
        return profile, "gaussian"
    raise ValueError("lineshape must be 'lorentzian' or 'gaussian'.")


def periodic_bse_absorption(
    result,
    energy_grid=None,
    polarization=None,
    broadening=0.1,
    units="ev",
    lineshape="lorentzian",
    transition_velocity=None,
    npoints=2001,
):
    """Build oscillator strengths and the imaginary dielectric spectrum.

    ``broadening`` is the Lorentzian half width at half maximum or the
    Gaussian standard deviation, expressed in ``units``. A missing
    ``polarization`` requests the Cartesian isotropic average. Complex
    polarization vectors are accepted for circularly polarized light.
    """

    if not hasattr(result, "space") or not hasattr(result, "block"):
        raise TypeError("periodic_bse_absorption expects a PeriodicBSEResult.")
    space = result.space
    q_index, transitions = _require_optical_q(space, result.block.q_index)
    roots = np.asarray(result.e, dtype=float)
    if roots.ndim != 1 or roots.size == 0:
        raise ValueError("Optical absorption requires at least one BSE root.")
    if np.any(~np.isfinite(roots)) or np.any(roots <= 0.0):
        raise ValueError("Optical absorption requires positive finite BSE roots.")

    gaps = np.asarray(result.block.transition_energy, dtype=float)
    if gaps.shape != (len(transitions),):
        raise ValueError("BSE transition energies do not match the transition table.")
    if np.any(~np.isfinite(gaps)) or np.any(gaps <= 0.0):
        raise ValueError("Optical length-gauge conversion requires positive transition gaps.")

    if transition_velocity is None:
        transition_velocity = periodic_transition_velocity_matrix_elements(
            space,
            q_index=q_index,
        )
        velocity_backend = "builtin_gaussian_gradient"
    else:
        transition_velocity = np.asarray(transition_velocity, dtype=np.complex128)
        velocity_backend = "supplied"
    expected_velocity_shape = (len(transitions), 3)
    if transition_velocity.shape != expected_velocity_shape:
        raise ValueError(
            "transition_velocity must have shape "
            f"{expected_velocity_shape}; got {transition_velocity.shape}."
        )
    if not np.all(np.isfinite(transition_velocity)):
        raise ValueError("transition_velocity must contain finite values.")

    transition_dipoles = 1j * transition_velocity / gaps[:, None]
    amplitudes = _root_amplitudes(result)
    weights = np.asarray(result.block.transition_weights, dtype=float)
    if weights.shape != gaps.shape or np.any(weights < 0.0):
        raise ValueError("BSE transition weights are invalid.")
    weighted_dipoles = np.sqrt(2.0 * weights)[:, None] * transition_dipoles
    exciton_dipoles = np.einsum(
        "tr,tx->rx",
        amplitudes,
        weighted_dipoles,
        optimize=True,
    )

    polarization = _normalize_polarization(polarization)
    if polarization is None:
        line_strengths = np.sum(abs(exciton_dipoles) ** 2, axis=1) / 3.0
    else:
        projected = np.einsum(
            "x,rx->r",
            polarization.conj(),
            exciton_dipoles,
            optimize=True,
        )
        line_strengths = abs(projected) ** 2
    oscillator_strengths = 2.0 * roots * line_strengths

    scale, canonical_units = _unit_scale(units)
    broadening = float(broadening)
    if not np.isfinite(broadening) or broadening <= 0.0:
        raise ValueError("broadening must be a positive finite value.")
    width_au = broadening / scale
    if energy_grid is None:
        try:
            npoints = int(npoints)
        except (TypeError, ValueError) as exc:
            raise TypeError("npoints must be an integer.") from exc
        if npoints < 2:
            raise ValueError("npoints must be at least 2.")
        lower = max(0.0, float(np.min(roots) - 8.0 * width_au))
        upper = float(np.max(roots) + 8.0 * width_au)
        grid_au = np.linspace(lower, upper, npoints)
        energy_grid = grid_au * scale
    else:
        energy_grid = np.asarray(energy_grid, dtype=float)
        if energy_grid.ndim != 1 or energy_grid.size < 2:
            raise ValueError(
                "energy_grid must be a one-dimensional array with at least 2 points."
            )
        if np.any(~np.isfinite(energy_grid)) or np.any(np.diff(energy_grid) <= 0.0):
            raise ValueError("energy_grid must be finite and strictly increasing.")
        grid_au = energy_grid / scale

    profiles, canonical_lineshape = _line_profile(
        grid_au,
        roots,
        width_au,
        lineshape,
    )
    dipole_outer = np.einsum(
        "ri,rj->rij",
        exciton_dipoles,
        exciton_dipoles.conj(),
        optimize=True,
    )
    volume = cell_volume(space.reference)
    dielectric_tensor = (4.0 * np.pi**2 / volume) * np.einsum(
        "rij,rw->ijw",
        dipole_outer,
        profiles,
        optimize=True,
    )
    if polarization is None:
        dielectric_imag = np.trace(dielectric_tensor, axis1=0, axis2=1).real / 3.0
    else:
        dielectric_imag = np.einsum(
            "i,ijw,j->w",
            polarization.conj(),
            dielectric_tensor,
            polarization,
            optimize=True,
        ).real

    return PeriodicBSEOpticalResult(
        energy_grid=np.asarray(energy_grid, dtype=float),
        excitation_energies=roots * scale,
        dielectric_imag=np.asarray(dielectric_imag, dtype=float),
        dielectric_tensor_imag=dielectric_tensor,
        oscillator_strengths=np.asarray(oscillator_strengths, dtype=float),
        line_strengths=np.asarray(line_strengths, dtype=float),
        transition_velocity=transition_velocity,
        transition_dipoles=transition_dipoles,
        exciton_dipoles=exciton_dipoles,
        polarization=polarization,
        broadening=broadening,
        units=canonical_units,
        lineshape=canonical_lineshape,
        metric=result.metric,
        q_index=q_index,
        info={
            "backend": "periodic_bse_optics",
            "pbc": True,
            "q_index": q_index,
            "metric": result.metric,
            "nroots": int(len(roots)),
            "ntransitions": int(len(transitions)),
            "velocity_backend": velocity_backend,
            "spin_degeneracy": 2,
            "kpoint_quadrature": "symmetric_sqrt_weights",
            "polarization": "isotropic" if polarization is None else "resolved",
            "broadening_convention": (
                "hwhm" if canonical_lineshape == "lorentzian" else "standard_deviation"
            ),
            "cell_volume": volume,
            "converged": bool(result.info.get("converged", False)),
        },
    )


def _lanczos_coefficients(operator, start, niter, tol, reorthogonalize):
    start = np.asarray(start, dtype=np.complex128)
    norm = float(np.linalg.norm(start))
    if norm <= tol:
        return np.zeros(0), np.zeros(0), norm

    q = start / norm
    previous = np.zeros_like(q)
    previous_beta = 0.0
    basis = []
    alpha = []
    beta = []
    for iteration in range(min(int(niter), len(start))):
        if reorthogonalize:
            basis.append(q.copy())
        action = operator.matvec(q)
        diagonal = np.vdot(q, action)
        if abs(diagonal.imag) > 100.0 * tol * max(1.0, abs(diagonal.real)):
            raise np.linalg.LinAlgError("The matrix-free TDA operator is not Hermitian.")
        alpha.append(float(diagonal.real))
        residual = action - diagonal.real * q - previous_beta * previous
        if reorthogonalize:
            for _pass in range(2):
                for vector in basis:
                    residual -= vector * np.vdot(vector, residual)
        if iteration + 1 >= min(int(niter), len(start)):
            break
        next_beta = float(np.linalg.norm(residual))
        if next_beta <= tol:
            break
        beta.append(next_beta)
        previous, q = q, residual / next_beta
        previous_beta = next_beta
    return np.asarray(alpha), np.asarray(beta), norm


def _continued_fraction_density(grid, width, alpha, beta, norm):
    if len(alpha) == 0 or norm == 0.0:
        return np.zeros_like(grid, dtype=float)
    frequency = np.asarray(grid, dtype=float) + 1j * float(width)
    denominator = frequency - alpha[-1]
    for index in range(len(alpha) - 2, -1, -1):
        denominator = frequency - alpha[index] - beta[index] ** 2 / denominator
    response = norm * norm / denominator
    return -response.imag / np.pi


def periodic_tda_haydock(
    operator,
    energy_grid=None,
    polarization=None,
    broadening=0.1,
    units="ev",
    transition_velocity=None,
    niter=100,
    tol=1.0e-12,
    reorthogonalize=True,
    npoints=2001,
):
    """Evaluate q=0 TDA absorption with a matrix-free Haydock recursion."""

    if not hasattr(operator, "matvec") or not hasattr(operator, "space"):
        raise TypeError("periodic_tda_haydock expects a PeriodicTDAOperator.")
    if operator.shape[0] == 0:
        raise ValueError("Haydock absorption requires a nonempty transition space.")
    try:
        niter = int(niter)
    except (TypeError, ValueError) as exc:
        raise TypeError("niter must be an integer.") from exc
    if niter < 1:
        raise ValueError("niter must be positive.")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive finite value.")

    scale, canonical_units = _unit_scale(units)
    broadening = float(broadening)
    if not np.isfinite(broadening) or broadening <= 0.0:
        raise ValueError("broadening must be a positive finite value.")
    width_au = broadening / scale
    if energy_grid is None:
        try:
            npoints = int(npoints)
        except (TypeError, ValueError) as exc:
            raise TypeError("npoints must be an integer.") from exc
        if npoints < 2:
            raise ValueError("npoints must be at least 2.")
        lower = max(0.0, float(np.min(operator.diagonal) - 8.0 * width_au))
        upper = float(np.max(operator.diagonal) + 8.0 * width_au)
        grid_au = np.linspace(lower, upper, npoints)
        energy_grid = grid_au * scale
    else:
        energy_grid = np.asarray(energy_grid, dtype=float)
        if energy_grid.ndim != 1 or energy_grid.size < 2:
            raise ValueError("energy_grid must be a one-dimensional array with at least 2 points.")
        if np.any(~np.isfinite(energy_grid)) or np.any(np.diff(energy_grid) <= 0.0):
            raise ValueError("energy_grid must be finite and strictly increasing.")
        grid_au = energy_grid / scale

    transitions = operator.space.transitions(operator.q_index)
    gaps = np.asarray(operator.transition_energy, dtype=float)
    if np.any(gaps <= 0.0):
        raise ValueError("Haydock length-gauge conversion requires positive transition gaps.")
    if transition_velocity is None:
        transition_velocity = periodic_transition_velocity_matrix_elements(
            operator.space,
            q_index=operator.q_index,
        )
        velocity_backend = "builtin_gaussian_gradient"
    else:
        transition_velocity = np.asarray(transition_velocity, dtype=np.complex128)
        velocity_backend = "supplied"
    expected_shape = (len(transitions), 3)
    if transition_velocity.shape != expected_shape:
        raise ValueError(
            f"transition_velocity must have shape {expected_shape}; "
            f"got {transition_velocity.shape}."
        )
    transition_dipoles = 1j * transition_velocity / gaps[:, None]
    weighted_dipoles = (
        np.sqrt(2.0 * operator.transition_weights)[:, None] * transition_dipoles
    )

    polarization = _normalize_polarization(polarization)
    if polarization is None:
        projections = tuple(weighted_dipoles[:, axis] for axis in range(3))
        average = 1.0 / 3.0
    else:
        projections = (
            np.einsum(
                "x,tx->t",
                polarization.conj(),
                weighted_dipoles,
                optimize=True,
            ),
        )
        average = 1.0

    densities = []
    alphas = []
    betas = []
    norms = []
    for projection in projections:
        alpha, beta, norm = _lanczos_coefficients(
            operator,
            projection.conj(),
            niter=niter,
            tol=tol,
            reorthogonalize=bool(reorthogonalize),
        )
        densities.append(
            _continued_fraction_density(grid_au, width_au, alpha, beta, norm)
        )
        alphas.append(alpha)
        betas.append(beta)
        norms.append(norm)
    spectral_density = average * np.sum(densities, axis=0)
    volume = cell_volume(operator.space.reference)
    dielectric_imag = (4.0 * np.pi**2 / volume) * spectral_density
    iterations = tuple(int(len(alpha)) for alpha in alphas)
    krylov_limit = min(int(niter), int(operator.shape[0]))
    krylov_complete = all(
        count == operator.shape[0] or count < krylov_limit
        for count in iterations
    )

    return PeriodicBSEHaydockResult(
        energy_grid=np.asarray(energy_grid, dtype=float),
        dielectric_imag=np.asarray(dielectric_imag, dtype=float),
        spectral_density=np.asarray(spectral_density, dtype=float),
        transition_velocity=transition_velocity,
        transition_dipoles=transition_dipoles,
        polarization=polarization,
        lanczos_alpha=tuple(alphas),
        lanczos_beta=tuple(betas),
        starting_norms=np.asarray(norms, dtype=float),
        broadening=broadening,
        units=canonical_units,
        q_index=int(operator.q_index),
        info={
            "backend": "periodic_bse_haydock",
            "solver": "hermitian_lanczos_continued_fraction",
            "pbc": True,
            "q_index": int(operator.q_index),
            "dimension": int(operator.shape[0]),
            "iterations": iterations,
            "requested_iterations": int(niter),
            "krylov_complete": bool(krylov_complete),
            "reorthogonalize": bool(reorthogonalize),
            "velocity_backend": velocity_backend,
            "spin_degeneracy": 2,
            "polarization": "isotropic" if polarization is None else "resolved",
            "broadening_convention": "hwhm",
            "cell_volume": volume,
            "operator": dict(operator.info),
            "converged": bool(krylov_complete),
        },
    )

__all__ = [
    "PeriodicBSEHaydockResult",
    "PeriodicBSEOpticalResult",
    "periodic_bse_absorption",
    "periodic_tda_haydock",
    "periodic_transition_velocity_matrix_elements",
]
