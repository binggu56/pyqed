"""Molecular imaginary-axis G0W0 and two-pole/Padé continuation."""

import copy
import time
import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares, newton
from threadpoolctl import threadpool_limits


def _twopole(freq, coeff):
    cf = coeff[:5] + 1.0j * coeff[5:]
    return cf[0] + cf[1] / (freq + cf[3]) + cf[2] / (freq + cf[4])


def _twopole_fit_residual(coeff, omega, sigma):
    residual = _twopole(omega, coeff) - sigma
    residual = np.array(residual, copy=True)
    if residual.size:
        residual[0] /= 0.01
    return np.asarray([residual.real, residual.imag]).reshape(-1)


def _fit_twopole_diag(sigma, omega, orbitals, nocc):
    sigma = np.asarray(sigma, dtype=np.complex128)
    omega = np.asarray(omega, dtype=np.complex128)
    coeff = np.zeros((10, sigma.shape[0]), dtype=float)
    for pos, orbital in enumerate(orbitals):
        if int(orbital) < int(nocc):
            guess = np.asarray([0, 1, 1, 1, -1, 0, 0, 0, -1.0, -0.5], dtype=float)
        else:
            guess = np.asarray([0, 1, 1, 1, -1, 0, 0, 0, 1.0, 0.5], dtype=float)
        result = least_squares(
            _twopole_fit_residual,
            guess,
            jac="3-point",
            method="trf",
            xtol=1.0e-10,
            gtol=1.0e-10,
            max_nfev=1000,
            args=(omega[pos], sigma[pos]),
        )
        coeff[:, pos] = result.x
    return coeff


def _ac_frequency_grid(freqs, weights, iw_cutoff):
    if iw_cutoff is None:
        nw_sigma = len(freqs) + 1
    else:
        nw_sigma = int(np.sum(np.asarray(freqs) < float(iw_cutoff))) + 1
    omega_occ = np.zeros(nw_sigma, dtype=np.complex128)
    omega_vir = np.zeros(nw_sigma, dtype=np.complex128)
    omega_occ[1:] = -1.0j * freqs[: nw_sigma - 1]
    omega_vir[1:] = 1.0j * freqs[: nw_sigma - 1]
    return nw_sigma, omega_occ, omega_vir


def kernel(
    gw,
    mo_energy,
    mo_coeff,
    orbs=None,
    nw=100,
    iw_cutoff=5.0,
    quadrature_scale=0.5,
    linearized=False,
    tol=1e-6,
    maxiter=100,
    transition_factors=None,
    callback=None,
    ac="pade",
    pade_npts=18,
    pade_step_ratio=2.0 / 3.0,
    stability_tol=1e-4,
    on_unstable="raise",
):
    """Diagonal molecular G0W0 with two-pole or Padé analytic continuation.

    Adaptation of imaginary-axis RI-GW in Ren et al., New J. Phys. 14,
    053020 (2012), doi:10.1088/1367-2630/14/5/053020, and Zhu and Chan,
    JCTC 17, 727–741 (2021), doi:10.1021/acs.jctc.0c00704. Reuses PyQED's
    periodic two-pole least-squares fit and molecular direct-RPA dielectric.
    Real, gapped restricted references and diagonal one-shot TDH only. All
    intermediate orbitals remain. Both continuation fits are approximations, not
    exact contour integration; fit errors and quadrature must be checked.
    Newton roots are not guaranteed to follow the contour continuation branch.
    Failed/nonpositive-weight roots raise; unrequested orbitals remain NaN.
    ``threads`` controls BLAS teams, with no nested orbital workers. Scratch
    is blocked; max_memory is not a global RSS cap. Projection reuses product
    buffers and hoists target-factor packing when the scratch budget permits.
    The static Cholesky factor is retained for BSE. callback(gw, index)
    receives each completed frequency.
    Grid/root parameters match PySCF: 100 integration points, 18 Padé points, spacing
    ratio 2/3, 5 Ha fitting cutoff, nonlinear roots with tolerance 1e-6 Ha
    and at most 100 iterations. ``ac="pade"`` selects the continued-fraction method documented in
    :func:`refit`; ``ac="twopole"`` retains the shared periodic fit.
    Padé roots are checked for point-count sensitivity by default.
    ``on_unstable="contour"`` explicitly repairs rejected targets with direct
    contour deformation; the default ``"raise"`` rejects them without hidden
    extra computation. See :func:`repair_continuation` for cost and limitations.
    """
    from .contour import ContourDeformation
    from .screening import StaticScreening
    from .gw import _set_rhf_orbitals, _sigma_x_diagonal

    if on_unstable not in ("raise", "contour"):
        raise ValueError("on_unstable must be raise or contour")
    if stability_tol is not None and (isinstance(stability_tol, (bool, np.bool_))
            or not np.isfinite(stability_tol) or stability_tol <= 0):
        raise ValueError("stability_tol must be positive and finite, or None")
    if tol <= 0 or not np.isfinite(tol) or int(maxiter) != maxiter or maxiter < 1:
        raise ValueError("Require positive tolerance and integer maxiter")
    if iw_cutoff is not None and (not np.isfinite(iw_cutoff) or iw_cutoff <= 0):
        raise ValueError("iw_cutoff must be positive or None")
    if not np.array_equal(mo_coeff, gw.mo_coeff):
        _set_rhf_orbitals(gw, mo_energy, mo_coeff)
    energy = np.asarray(mo_energy, dtype=float)
    selected = list(range(len(energy))) if orbs is None else list(orbs)
    # Reuse the validated dielectric/grid machinery, selecting BLAS contractions.
    reference = copy.copy(gw)
    reference.threads = 1
    response = ContourDeformation(
        reference,
        energy,
        selected,
        nw,
        quadrature_scale,
        prepare=False,
        transition_factors=transition_factors,
    )
    nocc, nmo, rank = response.nocc, len(energy), len(response.factors)
    orbs = response.orbs
    ef = 0.5 * (energy[nocc - 1] + energy[nocc])
    nsigma, occ_grid, vir_grid = _ac_frequency_grid(
        response.freqs, response.weights, iw_cutoff
    )
    if ac not in ("twopole", "pade"):
        raise ValueError("ac must be twopole or pade")
    if ac == "pade":
        _pade_indices(nsigma, pade_npts, pade_step_ratio)
    if ac == "twopole" and nsigma < 6:
        raise ValueError("Two-pole AC requires at least six evaluation frequencies")
    omega = np.where((orbs < nocc)[:, None], occ_grid, vir_grid)
    gw.ac_sigma = np.zeros(omega.shape, dtype=complex)
    gw.ac_omega = omega
    gw.ac_info = dict(
        method=ac,
        nw=int(nw),
        iw_cutoff=iw_cutoff,
        threads=getattr(gw, "threads", 1),
        fermi_level=float(ef),
        frequencies=[],
        linearized=bool(linearized),
    )
    # Bound pair-block copies and products; no full four-index tensor.
    batch = max(
        1, min(len(orbs), int(max(1.0, gw.max_memory) * 1e6 / (32 * rank * nmo)))
    )
    # Retain a view for consecutive targets; pack arbitrary selections once
    # only when the extra storage fits the scratch budget. Low-memory runs
    # gather each block into one reusable buffer instead.
    t = time.perf_counter()
    consecutive = np.all(np.diff(orbs) == 1)
    bytes_per_target = response.factors.dtype.itemsize * rank * nmo
    packed_bytes = bytes_per_target * len(orbs)
    rhs_all = None
    if response.factors.flags.c_contiguous and consecutive:
        rhs_all = response.factors.reshape(rank, -1)[
            :, orbs[0] * nmo : (orbs[-1] + 1) * nmo
        ]
        packed_bytes = 0
    elif packed_bytes <= max(1.0, gw.max_memory) * 1e6 / 4:
        rhs_all = np.empty((rank, len(orbs), nmo), dtype=response.factors.dtype)
        for j, p in enumerate(orbs):
            rhs_all[:, j] = response.factors[:, p]
        rhs_all = rhs_all.reshape(rank, -1)
    else:
        packed_bytes = bytes_per_target * batch
    packed = (
        np.empty((rank, batch, nmo), dtype=response.factors.dtype)
        if rhs_all is None else None
    )
    product_buffer = np.empty((rank, batch * nmo))
    element_buffer = np.empty((batch, nmo))
    offsets = [grid[None, :] + ef - energy[:, None] for grid in (occ_grid, vir_grid)]
    blocks = [
        (first, min(first + batch, len(orbs))) for first in range(0, len(orbs), batch)
    ]
    masks = [
        (orbs[first:last] < nocc, orbs[first:last] >= nocc) for first, last in blocks
    ]
    gw.ac_info["projection_setup_seconds"] = time.perf_counter() - t
    gw.ac_info["projection_packed_bytes"] = packed_bytes
    gw._contour = None
    gw._M = gw._charge_screening = None
    with threadpool_limits(limits=getattr(gw, "threads", 1), user_api="blas"):
        t = time.perf_counter()
        chol = la.cho_factor(
            response.dielectric(0.0, imaginary=True),
            lower=True,
            overwrite_a=True,
            check_finite=False,
        )
        response.static_screening = StaticScreening(
            response.factors, chol, gw.max_memory
        )
        gw._ac_screening = response
        gw.ac_info["static_seconds"] = time.perf_counter() - t
        for index, (frequency, weight) in enumerate(
            zip(response.freqs, response.weights)
        ):
            t = time.perf_counter()
            dielectric = response.dielectric(frequency, imaginary=True)
            polarization_seconds = time.perf_counter() - t
            t = time.perf_counter()
            fact = la.cho_factor(
                dielectric, lower=True, overwrite_a=True, check_finite=False
            )
            wc = la.cho_solve(fact, np.eye(rank), overwrite_b=True, check_finite=False)
            wc.flat[:: rank + 1] -= 1.0
            inversion_seconds = time.perf_counter() - t
            t = time.perf_counter()
            branches = [
                weight * x / (x * x + frequency * frequency) / np.pi for x in offsets
            ]
            for (first, last), block_masks in zip(blocks, masks):
                count = last - first
                if rhs_all is None:
                    # Copy slices directly: np.take may materialize the
                    # entire input when factors are not C-contiguous.
                    for j, p in enumerate(orbs[first:last]):
                        packed[:, j] = response.factors[:, p]
                    rhs = packed[:, :count].reshape(rank, count * nmo)
                else:
                    rhs = rhs_all[:, first * nmo : last * nmo]
                product = product_buffer[:, :count * nmo]
                np.matmul(wc, rhs, out=product)
                elements = element_buffer[:count]
                np.einsum("ij,ij->j", rhs, product, out=elements.reshape(-1))
                for mask, branch in zip(block_masks, branches):
                    gw.ac_sigma[first:last][mask] -= elements[mask] @ branch
            row = dict(
                index=index,
                dielectric_seconds=polarization_seconds,
                inverse_seconds=inversion_seconds,
                projection_seconds=time.perf_counter() - t,
            )
            gw.ac_info["frequencies"].append(row)
            if callback is not None:
                callback(gw, index)
        try:
            return refit(
                gw,
                energy,
                orbs=orbs,
                static=_sigma_x_diagonal(gw, orbs) - gw.v_mf[2 * orbs, 2 * orbs],
                ac=ac,
                pade_npts=pade_npts,
                pade_step_ratio=pade_step_ratio,
                linearized=linearized,
                tol=tol,
                maxiter=maxiter,
                stability_tol=stability_tol,
            )
        except RuntimeError:
            if on_unstable != "contour" or not gw.ac_info.get("stability", {}).get("unstable_orbitals"):
                raise
            return repair_continuation(gw, energy, nw=nw,
                                       quadrature_scale=quadrature_scale, tol=tol)



def _pade_indices(size, npts, step_ratio):
    if (
        isinstance(npts, (bool, np.bool_))
        or int(npts) != npts
        or npts < 2
        or npts % 2
        or npts >= size
    ):
        raise ValueError(
            "pade_npts must be even, at least two and less than the frequency count"
        )
    if not np.isfinite(step_ratio) or step_ratio <= 0:
        raise ValueError("pade_step_ratio must be finite and positive")
    steps = np.linspace(1.0, step_ratio, int(npts))
    positions = np.cumsum(steps * size / steps.sum())
    indices = np.rint(positions + 1 - positions[0]).astype(int)
    if len(np.unique(indices)) != npts or indices[0] < 0 or indices[-1] >= size:
        raise ValueError(
            "Padé point selection produced duplicate or out-of-range indices"
        )
    return indices


def _pade_coefficients(values, nodes):
    """Reciprocal differences for a scalar continued-fraction interpolant."""
    coeff = np.array(values, dtype=complex, copy=True)
    nodes = np.asarray(nodes, dtype=complex)
    if (
        coeff.ndim != 1
        or nodes.shape != coeff.shape
        or len(np.unique(nodes)) != len(nodes)
    ):
        raise ValueError("Padé needs matching vectors and distinct interpolation nodes")
    if not np.all(np.isfinite(coeff)) or not np.all(np.isfinite(nodes)):
        raise ValueError("Padé input must be finite")
    for i in range(1, len(coeff)):
        difference = coeff[i - 1] - coeff[i:]
        if np.all(difference == 0):
            coeff[i:] = 0
            break
        if coeff[i - 1] == 0:
            raise np.linalg.LinAlgError(
                "Singular Padé reciprocal differences; change selected points"
            )
        denominator = (nodes[i:] - nodes[i - 1]) * coeff[i:]
        if np.any(denominator == 0):
            raise np.linalg.LinAlgError(
                "Singular Padé reciprocal differences; change selected points"
            )
        coeff[i:] = difference / denominator
        if not np.all(np.isfinite(coeff[i:])):
            raise np.linalg.LinAlgError("Nonfinite Padé reciprocal differences")
    return coeff


def _pade(z, coeff, nodes):
    """Evaluate a continued fraction and its analytic derivative."""
    z = np.asarray(z, dtype=complex)
    tail, derivative = np.zeros_like(z), np.zeros_like(z)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        for i in range(len(coeff) - 1, 0, -1):
            denominator = 1 + tail
            numerator = coeff[i] * (z - nodes[i - 1])
            derivative = (
                coeff[i] / denominator - numerator * derivative / denominator**2
            )
            tail = numerator / denominator
        value = coeff[0] / (1 + tail)
        derivative = -value * derivative / (1 + tail)
    return value, derivative


def evaluate_fit(gw, position, frequency):
    """Evaluate correlation self-energy and derivative at absolute energy."""
    repaired = gw.ac_info.get("contour_orbitals", [])
    if repaired and int(gw.orbital_indices[position]) in repaired:
        orbital = int(gw.orbital_indices[position])
        frequencies = np.asarray(frequency)
        if np.any(frequencies.imag != 0):
            raise ValueError("Contour-repaired self-energies require real frequencies")
        values = np.asarray([gw._ac_contour.evaluate(orbital, float(w.real))
                             for w in frequencies.reshape(-1)])
        return tuple(values[:, j].reshape(frequencies.shape) for j in (0, 1))
    z = np.asarray(frequency) - gw.ac_info["fermi_level"]
    coeff = gw.ac_coeff[:, position]
    if gw.ac_info["method"] == "pade":
        return _pade(z, coeff, gw.ac_nodes[position])
    cf = coeff[:5] + 1j * coeff[5:]
    return _twopole(z, coeff), -cf[1] / (z + cf[3]) ** 2 - cf[2] / (z + cf[4]) ** 2


def refit(
    gw,
    mo_energy,
    *,
    orbs=None,
    static=None,
    ac="pade",
    pade_npts=18,
    pade_step_ratio=2.0 / 3.0,
    linearized=False,
    tol=1e-6,
    maxiter=100,
    stability_tol=1e-4,
):
    """Refit saved ``ac_sigma``/``ac_omega`` and solve QP roots without screening.

    Padé uses reciprocal-difference continued fractions following H. J.
    Vidberg and J. W. Serene, J. Low Temp. Phys. 29, 179–192 (1977),
    doi:10.1007/BF00655090. This adapts the method to diagonal molecular GW,
    with PySCF-style nonuniform selection (18 points, step ratio 2/3) and
    analytic differentiation of the finite continued fraction. It is not a
    noise-regularized or causality-constrained fit. Spurious poles and point
    sensitivity remain possible; small interpolation/root residuals do not
    bound real-axis error. By default, neighboring point counts (n-2, n+2)
    must yield positive-weight roots within ``stability_tol=1e-4`` Ha.
    Failure raises and invalidates the affected energies. This sensitivity
    check is a rejection criterion, not a real-axis accuracy certificate.
    Use contour deformation for rejected states. ``stability_tol=None``
    explicitly requests unchecked continuation; diagnostics record this.
    Test quadrature and independent contour references even after passing.
    Two-pole fitting reuses the periodic helper unchanged.

    Frequencies are relative to ``ac_info['fermi_level']``; ``static`` is
    Sigma_x minus the reference exchange-correlation diagonal for each target.
    It is cached as ``ac_static``. Inputs must share the same reference/grid
    and orbital ordering. This function never accesses integrals or screening.
    Fields ``e_qp``, ``qp_weights``, ``qp_residuals``, ``ac_coeff``, ``ac_nodes``
    and fit diagnostics are updated for this fit. A failed fit leaves
    ``converged=False`` and may retain partial diagnostics.
    """
    started = time.perf_counter()
    gw.converged = False
    if ac not in ("twopole", "pade"):
        raise ValueError("ac must be twopole or pade")
    if not np.isfinite(tol) or tol <= 0 or int(maxiter) != maxiter or maxiter < 1:
        raise ValueError("Require positive tolerance and integer maxiter")
    if stability_tol is not None and (isinstance(stability_tol, (bool, np.bool_))
            or not np.isfinite(stability_tol) or stability_tol <= 0):
        raise ValueError("stability_tol must be positive and finite, or None")
    energy = np.asarray(mo_energy, dtype=float)
    ids = np.asarray(gw.orbital_indices if orbs is None else orbs)
    sigma, omega = np.asarray(gw.ac_sigma), np.asarray(gw.ac_omega)
    static = np.asarray(gw.ac_static if static is None else static, dtype=float)
    if (
        energy.ndim != 1
        or ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or len(ids) == 0
        or len(np.unique(ids)) != len(ids)
        or np.any(ids < 0)
        or np.any(ids >= len(energy))
        or sigma.ndim != 2
        or sigma.shape != omega.shape
        or len(sigma) != len(ids)
        or static.shape != (len(ids),)
        or not np.all(np.isfinite(static))
        or not np.all(np.isfinite(energy))
        or not np.all(np.isfinite(sigma))
        or not np.all(np.isfinite(omega))
        or not np.isfinite(gw.ac_info["fermi_level"])
    ):
        raise ValueError(
            "Inconsistent or nonfinite saved AC data, targets or static shifts"
        )
    gw.e_qp = np.full(len(energy), np.nan)
    gw.qp_weights = np.full(len(energy), np.nan)
    gw.qp_residuals = np.full(len(energy), np.nan)
    gw.orbital_indices = ids.copy()
    gw.ac_info.pop("stability", None)
    gw.ac_info.pop("contour_orbitals", None)
    gw.ac_info.pop("contour_repair", None)
    gw._ac_contour = None
    if ac == "pade":
        indices = _pade_indices(sigma.shape[1], pade_npts, pade_step_ratio)
        nodes = omega[:, indices].copy()
        coeff = np.stack(
            [_pade_coefficients(row[indices], z) for row, z in zip(sigma, nodes)],
            axis=1,
        )
    else:
        if sigma.shape[1] < 6:
            raise ValueError("Two-pole AC requires at least six evaluation frequencies")
        indices = np.arange(sigma.shape[1])
        nodes = None
        coeff = _fit_twopole_diag(sigma, omega, ids, gw.nocc // 2)
    gw.ac_coeff, gw.ac_nodes, gw.ac_static = coeff, nodes, static.copy()
    gw.ac_info.update(
        method=ac,
        linearized=bool(linearized),
        fit_indices=indices.tolist(),
        pade_npts=int(pade_npts) if ac == "pade" else None,
        pade_step_ratio=float(pade_step_ratio) if ac == "pade" else None,
    )
    ef = gw.ac_info["fermi_level"]
    fitted = np.asarray(
        [evaluate_fit(gw, j, omega[j] + ef)[0] for j in range(len(ids))]
    )
    errors = abs(fitted - sigma)
    held_out = np.ones(sigma.shape[1], dtype=bool)
    held_out[indices] = False
    gw.ac_info["fit_max_errors"] = np.max(errors, axis=1).tolist()
    gw.ac_info["held_out_max_errors"] = (
        np.max(errors[:, held_out], axis=1).tolist() if np.any(held_out) else None
    )
    gw.orbital_indices = ids.copy()
    for j, p in enumerate(ids):
        root, z, residual = _fit_root(
            lambda w: evaluate_fit(gw, j, w), energy[p], static[j],
            linearized, tol, maxiter)
        gw.e_qp[p], gw.qp_weights[p], gw.qp_residuals[p] = root, z, residual
    gw.ac_info["stability"] = dict(checked=False, tolerance_Ha=stability_tol)
    if ac == "pade" and stability_tol is not None:
        counts = [n for n in (pade_npts-2, pade_npts+2)
                  if 2 <= n < sigma.shape[1] and n % 2 == 0]
        if not counts:
            gw.e_qp[ids] = np.nan
            raise RuntimeError("Padé stability check needs more frequency samples")
        alternatives = np.full((len(counts), len(ids)), np.nan)
        for k, count in enumerate(counts):
            selected = _pade_indices(sigma.shape[1], count, pade_step_ratio)
            for j, p in enumerate(ids):
                try:
                    nodes = omega[j, selected]
                    coeff = _pade_coefficients(sigma[j, selected], nodes)
                    fit = lambda w: _pade(w-ef, coeff, nodes)
                    alternatives[k, j] = _fit_root(
                        fit, energy[p], static[j], linearized, tol, maxiter)[0]
                except (RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                    pass
        spread = np.max(abs(alternatives-gw.e_qp[ids]), axis=0)
        bad = ~np.isfinite(spread) | (spread > stability_tol)
        gw.ac_info["stability"] = dict(checked=True, tolerance_Ha=stability_tol,
            point_counts=counts, candidate_energies=gw.e_qp[ids].tolist(),
            max_changes_Ha=[float(x) if np.isfinite(x) else None for x in spread],
            unstable_orbitals=ids[bad].tolist(), passed=not np.any(bad))
        if np.any(bad):
            gw.e_qp[ids[bad]] = np.nan
            gw.qp_weights[ids[bad]] = np.nan
            gw.qp_residuals[ids[bad]] = np.nan
            gw.ac_info["fit_and_root_seconds"] = time.perf_counter()-started
            raise RuntimeError(
                f"Unstable Padé continuation for orbitals {ids[bad].tolist()}; "
                "use contour_deformation for these states. A small QP root "
                "residual does not establish continuation accuracy. "
                "stability_tol=None explicitly disables this check.")
    gw.ac_info["fit_and_root_seconds"] = time.perf_counter() - started
    gw.converged = True
    return gw.e_qp


def _fit_root(evaluate, energy, static, linearized, tol, maxiter):
    """Solve and validate one fitted diagonal Dyson equation."""
    correction = lambda w: evaluate(w)[0].real + static
    derivative = lambda w: evaluate(w)[1].real
    if linearized:
        root = energy + correction(energy)/(1-derivative(energy))
    else:
        root = newton(lambda w: w-energy-correction(w), energy,
                      fprime=lambda w: 1-derivative(w), tol=tol, maxiter=int(maxiter))
    z = 1/(1-derivative(energy if linearized else root))
    residual = abs(root-energy-correction(root))
    if (not np.isfinite(root) or not np.isfinite(z) or z <= 0
            or (not linearized and residual > max(tol, 1e-10))):
        raise RuntimeError("AC quasiparticle root failed")
    return root, z, residual


def repair_continuation(gw, mo_energy, *, nw=100, quadrature_scale=0.5, tol=1e-7,
                        eta=1e-6, accuracy_tol=1e-5, max_refinements=3):
    """Replace rejected Padé targets with direct contour-deformation QPs.

    Uses the complete RI direct-RPA screening and positive-weight coupling
    continuation of :func:`pyqed.gw.contour.kernel` (Zhu and Chan, JCTC 17,
    727–741 (2021), doi:10.1021/acs.jctc.0c00704). This hybrid adapts that
    formulation by retaining accepted AC roots and replacing rejected targets
    only. All intermediate orbitals remain. No fallback to HF or unvalidated
    Padé occurs. Repairs use a contour-specific eta (not the unrelated AC
    broadening), double the quadrature and reduce eta tenfold independently.
    Both QP changes must be below accuracy_tol (Ha); otherwise refinement
    continues or raises, leaving rejected energies unset. This is a numerical
    convergence check, not a proof of the dominant spectral branch. Many
    deep/core or high virtual targets can be expensive.
    Repaired roots with spectral weight below 0.01 trigger a warning and
    are listed as weak_root_orbitals in the repair audit. This diagnostic
    threshold does not select a dominant peak or alter numerical convergence.
    Subsequent self-energy evaluations dispatch to the same contour model.
    """
    import warnings
    from .contour import ContourDeformation, kernel as contour_kernel
    from .gw import _sigma_x_diagonal
    for name, value in (("accuracy_tol", accuracy_tol), ("tol", tol)):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if isinstance(eta, (bool, np.bool_)) or not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    if (isinstance(max_refinements, (bool, np.bool_))
            or not isinstance(max_refinements, (int, np.integer)) or max_refinements < 1):
        raise ValueError("max_refinements must be a positive integer")
    targets = gw.ac_info.get("stability", {}).get("unstable_orbitals", [])
    if not targets:
        raise ValueError("No rejected Padé targets to repair")
    gw.converged = False
    positions = [list(gw.orbital_indices).index(p) for p in targets]
    physical_static = (_sigma_x_diagonal(gw, targets)
                       - gw.v_mf[2*np.asarray(targets), 2*np.asarray(targets)])
    if not np.allclose(gw.ac_static[positions], physical_static, atol=1e-10, rtol=0):
        raise ValueError("Contour repair requires the same physical static self-energy")
    warnings.warn(f"Replacing unstable Padé targets {targets} with direct contour "
                  "deformation; this may be substantially more expensive.", RuntimeWarning,
                  stacklevel=2)
    gw.e_qp[targets] = np.nan
    gw.qp_weights[targets] = np.nan
    gw.qp_residuals[targets] = np.nan
    gw.ac_info.pop("contour_orbitals", None)
    gw._ac_contour = None
    reference = copy.copy(gw)
    reference.e_qp = None
    reference.eta = float(eta)
    started = time.perf_counter()
    response = getattr(gw, '_ac_screening', None)
    history = []
    audit = dict(passed=False, accuracy_tol_Ha=float(accuracy_tol), history=history)
    gw.ac_info['contour_repair'] = audit

    def solve(points, previous=None):
        # AC owns a multithreaded BLAS context; contour uses OpenMP instead.
        with threadpool_limits(limits=1 if getattr(reference, 'threads', 1) > 1 else None,
                               user_api='blas'):
            # Imaginary-axis screening is independent of the real-axis regulator.
            if previous is None:
                reference._contour = ContourDeformation(
                    reference, mo_energy, targets, points, quadrature_scale,
                    transition_factors=None if response is None else response.pairs)
            else:
                reference._contour = copy.copy(previous)
                reference._contour.eta = reference.eta
            reference.e_qp = None
            return contour_kernel(reference, mo_energy, gw.mo_coeff, orbs=targets,
                                  nw=points, quadrature_scale=quadrature_scale,
                                  tol=min(tol, accuracy_tol*.01), reuse_screening=True)
    corrected = solve(nw)
    for _ in range(max_refinements):
        refined = solve(2*nw)
        quadrature_change = np.abs(refined[targets]-corrected[targets])
        reference.eta *= .1
        corrected = solve(2*nw, reference._contour)
        eta_change = np.abs(corrected[targets]-refined[targets])
        history.append(dict(nw=int(2*nw), eta=reference.eta,
                            quadrature_changes_Ha=quadrature_change.tolist(),
                            eta_changes_Ha=eta_change.tolist(),
                            energies_Ha=corrected[targets].tolist()))
        nw *= 2
        if (np.all(np.isfinite(corrected[targets]))
                and np.all(quadrature_change <= accuracy_tol)
                and np.all(eta_change <= accuracy_tol)):
            break
    else:
        audit['seconds'] = time.perf_counter()-started
        raise RuntimeError("Contour repair failed quadrature/eta convergence; "
                           "rejected quasiparticle energies remain unset")
    gw.e_qp[targets] = corrected[targets]
    gw.qp_weights[targets] = reference.qp_weights[targets]
    gw.qp_residuals[targets] = reference.qp_residuals[targets]
    gw._ac_contour = reference._contour
    gw.ac_info['contour_orbitals'] = list(targets)
    audit.update(passed=True, seconds=time.perf_counter()-started,
                 **reference.contour_info)
    weak = np.asarray(targets)[gw.qp_weights[targets] < .01].tolist()
    audit.update(weak_root_orbitals=weak, weak_root_weight_threshold=.01)
    gw.converged = bool(np.all(np.isfinite(gw.e_qp[gw.orbital_indices])))
    if weak:
        warnings.warn(f"Contour repair found Dyson roots with spectral weight below "
                      f"0.01 for orbitals {weak}; numerical convergence does not "
                      "identify the dominant quasiparticle peak. Inspect qp_weights "
                      "and the root-selection policy.", RuntimeWarning, stacklevel=2)
    return gw.e_qp
