"""Native Goedecker-Teter-Hutter pseudopotential helpers."""

from dataclasses import dataclass
from functools import lru_cache
import math
import re

import numpy as np
from scipy.special import sph_harm_y

from pyqed.qchem.basis import _shell, overlap


def _canonical_symbol(symbol):
    text = "".join(re.findall(r"[A-Za-z]+", str(symbol)))
    if not text:
        raise ValueError(f"Invalid element symbol {symbol!r}.")
    return text[0].upper() + text[1:].lower()


@dataclass(frozen=True)
class GTHProjector:
    radius: float
    coupling: np.ndarray

    @property
    def nproj(self):
        return int(self.coupling.shape[0])


@dataclass(frozen=True)
class GTHPseudo:
    """Normalized scalar-relativistic GTH/HGH pseudopotential parameters."""

    symbol: str
    ionic_charge: float
    valence_configuration: tuple
    local_radius: float
    local_coefficients: tuple
    projectors: tuple
    name: str | None = None

    def local_fourier(self, gvecs):
        """Return the positive GTH local kernel used with a leading minus sign."""
        gvecs = np.asarray(gvecs, dtype=float)
        if gvecs.ndim != 2 or gvecs.shape[1] != 3:
            raise ValueError("gvecs must have shape (ng, 3).")
        g2 = np.einsum("gi,gi->g", gvecs, gvecs)
        radius = float(self.local_radius)
        reduced = g2 * radius * radius
        damping = np.exp(-0.5 * reduced)

        values = np.empty_like(g2)
        nonzero = g2 > 1.0e-16
        values[nonzero] = (
            4.0 * np.pi * float(self.ionic_charge)
            * damping[nonzero] / g2[nonzero]
        )
        values[~nonzero] = -2.0 * np.pi * float(self.ionic_charge) * radius**2

        coeff = np.zeros_like(g2)
        local = self.local_coefficients
        if len(local) >= 1:
            coeff += local[0]
        if len(local) >= 2:
            coeff += local[1] * (3.0 - reduced)
        if len(local) >= 3:
            coeff += local[2] * (15.0 - 10.0 * reduced + reduced**2)
        if len(local) >= 4:
            coeff += local[3] * (
                105.0 - 105.0 * reduced + 21.0 * reduced**2 - reduced**3
            )
        values -= (2.0 * np.pi) ** 1.5 * radius**3 * damping * coeff
        return values


def _normalize_raw_gth(symbol, raw, name=None):
    if not isinstance(raw, (list, tuple)) or len(raw) < 5:
        raise ValueError(f"Invalid GTH pseudopotential data for {symbol!r}.")
    ionic_parts, radius, nlocal, coefficients, ntypes = raw[:5]
    valence_configuration = tuple(float(x) for x in ionic_parts)
    ionic_charge = float(np.sum(np.asarray(valence_configuration, dtype=float)))
    radius = float(radius)
    if ionic_charge <= 0.0 or radius <= 0.0:
        raise ValueError("GTH ionic charges and local radii must be positive.")

    nlocal = int(nlocal)
    local = tuple(float(x) for x in coefficients[:nlocal])
    if nlocal < 0 or nlocal > 4 or len(local) != nlocal:
        raise ValueError("GTH local potentials support zero to four coefficients.")

    ntypes = int(ntypes)
    if len(raw) < 5 + ntypes:
        raise ValueError("GTH nonlocal projector data are incomplete.")
    projectors = []
    for angular_momentum, block in enumerate(raw[5:5 + ntypes]):
        proj_radius, nproj, coupling = block
        nproj = int(nproj)
        matrix = np.asarray(coupling, dtype=float)
        if nproj == 0:
            matrix = np.zeros((0, 0), dtype=float)
        if matrix.shape != (nproj, nproj):
            raise ValueError(
                f"Invalid l={angular_momentum} GTH coupling matrix for {symbol!r}."
            )
        matrix = np.asarray(0.5 * (matrix + matrix.T), dtype=float)
        matrix.setflags(write=False)
        projectors.append(GTHProjector(float(proj_radius), matrix))

    return GTHPseudo(
        symbol=_canonical_symbol(symbol),
        ionic_charge=ionic_charge,
        valence_configuration=valence_configuration,
        local_radius=radius,
        local_coefficients=local,
        projectors=tuple(projectors),
        name=None if name is None else str(name),
    )


def _load_named_gth(name, symbol):
    try:
        from pyscf.pbc.gto import pseudo as pyscf_pseudo
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Named GTH pseudopotentials require PySCF as a data loader. "
            "Pass an explicit PySCF-format pseudo dictionary to run without PySCF."
        ) from exc
    return pyscf_pseudo.load(str(name), _canonical_symbol(symbol))


def load_gth_pseudos(specification, symbols):
    """Normalize a named or explicit PySCF-format GTH pseudopotential mapping."""
    if specification is None:
        return {}
    symbols = tuple(dict.fromkeys(_canonical_symbol(sym) for sym in symbols))
    if isinstance(specification, str):
        return {
            sym: _normalize_raw_gth(
                sym,
                _load_named_gth(specification, sym),
                name=specification,
            )
            for sym in symbols
        }
    if not hasattr(specification, "items"):
        raise TypeError("pseudo must be a GTH name or an element-to-pseudo mapping.")

    raw_mapping = {
        _canonical_symbol(symbol): value
        for symbol, value in specification.items()
    }
    result = {}
    for symbol in symbols:
        if symbol not in raw_mapping:
            continue
        raw = raw_mapping[symbol]
        name = raw if isinstance(raw, str) else None
        if isinstance(raw, str):
            raw = _load_named_gth(raw, symbol)
        if isinstance(raw, GTHPseudo):
            result[symbol] = raw
        else:
            result[symbol] = _normalize_raw_gth(symbol, raw, name=name)
    return result


@lru_cache(maxsize=128)
def _solid_harmonic_coefficients(angular_momentum, magnetic_number):
    """Cartesian coefficients of the normalized complex solid harmonic."""
    angular_momentum = int(angular_momentum)
    magnetic_number = int(magnetic_number)
    monomials = tuple(_shell(angular_momentum))
    rng = np.random.default_rng(7919 + 101 * angular_momentum + magnetic_number)
    points = rng.normal(size=(max(32, 4 * len(monomials)), 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    x, y, z = points.T
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    design = np.asarray([
        [point[0] ** a * point[1] ** b * point[2] ** c for a, b, c in monomials]
        for point in points
    ])
    target = sph_harm_y(
        angular_momentum,
        magnetic_number,
        theta,
        phi,
    )
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = np.max(np.abs(design @ coefficients - target))
    if residual > 1.0e-11:
        raise RuntimeError("Failed to construct Cartesian solid-harmonic coefficients.")
    return tuple(
        (monomial, complex(coefficient))
        for monomial, coefficient in zip(monomials, coefficients)
        if abs(coefficient) > 1.0e-13
    )


@lru_cache(maxsize=16)
def _radial_power_terms(power):
    """Expansion coefficients of (x^2 + y^2 + z^2)^power."""
    power = int(power)
    terms = []
    for ix in range(power + 1):
        for iy in range(power - ix + 1):
            iz = power - ix - iy
            coefficient = (
                math.factorial(power)
                / (math.factorial(ix) * math.factorial(iy) * math.factorial(iz))
            )
            terms.append((ix, iy, iz, float(coefficient)))
    return tuple(terms)


def projector_overlap(basis_function, center, angular_momentum, magnetic_number,
                      projector_index, radius):
    """Return ``<p_i^l Y_lm | chi>`` using exact Gaussian moments."""
    l = int(angular_momentum)
    i = int(projector_index)
    radius = float(radius)
    alpha = 0.5 / (radius * radius)
    one_based = i + 1
    normalization = np.sqrt(2.0) / (
        radius ** (l + 0.5 * (4 * one_based - 1))
        * np.sqrt(math.gamma(l + 0.5 * (4 * one_based - 1)))
    )

    value = 0.0j
    for shell, solid_coefficient in _solid_harmonic_coefficients(l, int(magnetic_number)):
        for ix, iy, iz, radial_coefficient in _radial_power_terms(i):
            projector_shell = (
                shell[0] + 2 * ix,
                shell[1] + 2 * iy,
                shell[2] + 2 * iz,
            )
            coefficient = normalization * solid_coefficient.conjugate() * radial_coefficient
            for exponent, weight in zip(
                basis_function.exps,
                basis_function.prim_weights,
            ):
                value += coefficient * weight * overlap(
                    float(exponent),
                    basis_function.shell,
                    basis_function.origin,
                    alpha,
                    projector_shell,
                    center,
                )
    return value


@lru_cache(maxsize=262144)
def _primitive_three_gaussian_overlap(
    exponent_a, shell_a, center_a,
    exponent_b, shell_b, center_b,
    exponent_c, shell_c, center_c,
):
    exponents = (float(exponent_a), float(exponent_b), float(exponent_c))
    centers = tuple(np.asarray(center, dtype=float) for center in (center_a, center_b, center_c))
    total_exponent = sum(exponents)
    product_center = sum(
        exponent * center for exponent, center in zip(exponents, centers)
    ) / total_exponent
    prefactor = np.exp(-sum(
        exponent * float(np.dot(center, center))
        for exponent, center in zip(exponents, centers)
    ) + total_exponent * float(np.dot(product_center, product_center)))

    value = float(prefactor)
    for axis in range(3):
        moment = 0.0
        degrees = tuple(int(shell[axis]) for shell in (shell_a, shell_b, shell_c))
        displacements = tuple(
            float(product_center[axis] - center[axis]) for center in centers
        )
        for power_a in range(degrees[0] + 1):
            coeff_a = (
                math.comb(degrees[0], power_a)
                * displacements[0] ** (degrees[0] - power_a)
            )
            for power_b in range(degrees[1] + 1):
                coeff_ab = coeff_a * (
                    math.comb(degrees[1], power_b)
                    * displacements[1] ** (degrees[1] - power_b)
                )
                for power_c in range(degrees[2] + 1):
                    degree = power_a + power_b + power_c
                    if degree % 2:
                        continue
                    coefficient = coeff_ab * (
                        math.comb(degrees[2], power_c)
                        * displacements[2] ** (degrees[2] - power_c)
                    )
                    moment += coefficient * (
                        math.gamma(0.5 * (degree + 1))
                        / total_exponent ** (0.5 * (degree + 1))
                    )
        value *= moment
    return float(value)


def local_gaussian_overlap(left, right, center, pseudo):
    """Integrate the Gaussian-polynomial part of a GTH local potential."""
    if not pseudo.local_coefficients:
        return 0.0
    radius = float(pseudo.local_radius)
    exponent_c = 0.5 / (radius * radius)
    center_array = np.asarray(center, dtype=float)
    left_center = tuple(float(x) for x in np.asarray(left.origin) - center_array)
    right_center = tuple(float(x) for x in np.asarray(right.origin) - center_array)
    center = (0.0, 0.0, 0.0)
    value = 0.0
    for radial_power, local_coefficient in enumerate(pseudo.local_coefficients):
        if local_coefficient == 0.0:
            continue
        scale = float(local_coefficient) / radius ** (2 * radial_power)
        for ix, iy, iz, radial_coefficient in _radial_power_terms(radial_power):
            shell_c = (2 * ix, 2 * iy, 2 * iz)
            for exponent_a, weight_a in zip(left.exps, left.prim_weights):
                for exponent_b, weight_b in zip(right.exps, right.prim_weights):
                    value += (
                        scale
                        * radial_coefficient
                        * float(weight_a)
                        * float(weight_b)
                        * _primitive_three_gaussian_overlap(
                            float(exponent_a),
                            tuple(left.shell),
                            left_center,
                            float(exponent_b),
                            tuple(right.shell),
                            right_center,
                            exponent_c,
                            shell_c,
                            center,
                        )
                    )
    return float(value)


__all__ = [
    "GTHProjector",
    "GTHPseudo",
    "load_gth_pseudos",
    "local_gaussian_overlap",
    "projector_overlap",
]
