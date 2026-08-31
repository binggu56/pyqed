"""Reactive-coordinate charts for multidimensional phenol models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqed.models.phenol import Phenol3D, dpes1
from pyqed.units import au2amu, au2angstrom, au2ev, wavenumber2hartree


PHENOL_SPECIES = ("C",) * 6 + ("O", "H") + ("H",) * 5
PHENOL_MASSES = np.asarray((12.011,) * 6 + (15.999, 1.008) + (1.008,) * 5)


def _base_geometry(r_oh, torsion, bend):
    r_cc, r_co, r_ch = 1.394, 1.360, 1.084
    phase = np.arange(6) * np.pi / 3.0
    carbons = r_cc * np.column_stack((np.cos(phase), np.sin(phase), np.zeros(6)))
    oxygen = carbons[0] + np.asarray((r_co, 0.0, 0.0))
    direction = (
        np.cos(bend) * np.asarray((-1.0, 0.0, 0.0))
        + np.sin(bend)
        * np.asarray((0.0, np.cos(torsion), np.sin(torsion)))
    )
    hydroxyl_h = oxygen + float(r_oh) * direction
    ring_h = np.asarray(
        [
            carbons[index]
            + r_ch * carbons[index] / np.linalg.norm(carbons[index])
            for index in range(1, 6)
        ]
    )
    return np.vstack((carbons, oxygen, hydroxyl_h, ring_h))


def _remove_rigid_motion(vector, geometry, masses=PHENOL_MASSES):
    """Project a Cartesian displacement away from translations and rotations."""

    geometry = np.asarray(geometry, dtype=float)
    vector = np.asarray(vector, dtype=float)
    center = np.average(geometry, axis=0, weights=masses)
    centered = geometry - center
    rigid = []
    for axis in np.eye(3):
        rigid.append(np.broadcast_to(axis, geometry.shape).copy())
    for axis in np.eye(3):
        rigid.append(np.cross(np.broadcast_to(axis, geometry.shape), centered))
    weight = np.sqrt(masses)[:, None]
    basis = []
    for candidate in rigid:
        candidate = (candidate * weight).reshape(-1)
        for previous in basis:
            candidate -= np.dot(previous, candidate) * previous
        norm = np.linalg.norm(candidate)
        if norm > 1.0e-12:
            basis.append(candidate / norm)
    weighted = (vector * weight).reshape(-1)
    for candidate in basis:
        weighted -= np.dot(candidate, weighted) * candidate
    return weighted.reshape(vector.shape) / weight


def _mass_orthonormalize(vectors, geometry):
    modes = []
    for vector in np.asarray(vectors, dtype=float):
        candidate = _remove_rigid_motion(vector, geometry)
        for previous in modes:
            projection = np.einsum(
                "ia,ia,i->", candidate, previous, PHENOL_MASSES
            )
            candidate -= projection * previous
        norm = np.sqrt(
            np.einsum("ia,ia,i->", candidate, candidate, PHENOL_MASSES)
        )
        if norm < 1.0e-12:
            raise ValueError("phenol mode templates are linearly dependent")
        modes.append(candidate / norm)
    return np.asarray(modes)


def phenol_template_modes():
    """Return symmetry-adapted templates for phenol ``16a`` and ``8a``.

    The templates identify the desired Hessian modes reproducibly.  They are
    also useful in lightweight workflow tests, but production calculations
    should replace them with the Hessian modes returned by
    :func:`select_phenol_active_modes`.
    """

    equilibrium = _base_geometry(
        Phenol3D.r_eq * au2angstrom, 0.0, Phenol3D.theta_eq
    )
    # Wilson 16a: low-frequency out-of-plane phenoxyl ring deformation.  The
    # attached ring hydrogens follow their carbon, while O--H motion is kept
    # out of the template so it cannot collapse onto the explicit OH torsion.
    mode_16a = np.zeros_like(equilibrium)
    pattern = np.asarray((0.0, 1.0, -1.0, 0.0, 1.0, -1.0))
    mode_16a[:6, 2] = pattern
    mode_16a[8:, 2] = pattern[1:]

    # Wilson 8a: high-frequency in-plane ring stretch.  A twofold radial
    # modulation separates it from the totally symmetric breathing motion.
    mode_8a = np.zeros_like(equilibrium)
    radial = equilibrium[:6].copy()
    radial[:, 2] = 0.0
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    modulation = np.cos(2.0 * np.arange(6) * np.pi / 3.0)
    mode_8a[:6] = modulation[:, None] * radial
    mode_8a[8:] = modulation[1:, None] * radial[1:]
    return _mass_orthonormalize((mode_16a, mode_8a), equilibrium)


def mode_reflection_parity(mode):
    """Return the mass-metric expectation value of reflection in the ring plane."""

    mode = np.asarray(mode, dtype=float)
    if mode.shape != (len(PHENOL_SPECIES), 3):
        raise ValueError("phenol mode must have shape (13, 3)")
    reflected = mode @ np.diag((1.0, 1.0, -1.0))
    norm = np.einsum("ia,ia,i->", mode, mode, PHENOL_MASSES)
    if norm < 1.0e-14:
        raise ValueError("phenol mode has zero mass norm")
    return float(
        np.einsum("ia,ia,i->", mode, reflected, PHENOL_MASSES) / norm
    )


def select_phenol_active_modes(
    frequencies_cm1,
    modes,
    *,
    targets_cm1=(249.92, 1691.34),
):
    """Select and phase the Wilson ``16a`` and ``8a`` Hessian modes.

    Selection combines reflection symmetry, proximity to the spectroscopic
    reference frequencies, and overlap with deterministic Wilson templates.
    The returned order is ``(16a, 8a)`` with reflection parities ``(-1,+1)``.
    """

    frequencies = np.asarray(frequencies_cm1, dtype=float)
    modes = np.asarray(modes, dtype=float)
    if modes.shape != (len(frequencies), len(PHENOL_SPECIES), 3):
        raise ValueError("modes and frequencies have incompatible shapes")
    equilibrium = _base_geometry(
        Phenol3D.r_eq * au2angstrom, 0.0, Phenol3D.theta_eq
    )
    normalized = []
    for mode in modes:
        candidate = _remove_rigid_motion(mode, equilibrium)
        norm = np.sqrt(
            np.einsum("ia,ia,i->", candidate, candidate, PHENOL_MASSES)
        )
        normalized.append(candidate / norm if norm > 1.0e-12 else candidate)
    normalized = np.asarray(normalized)
    templates = phenol_template_modes()
    parities = np.asarray([mode_reflection_parity(mode) for mode in normalized])
    selected = []
    diagnostics = []
    for label, target, required_parity, template in zip(
        ("16a", "8a"), targets_cm1, (-1.0, 1.0), templates
    ):
        candidates = np.flatnonzero(
            (frequencies > 20.0)
            & (required_parity * parities > 0.75)
            & ~np.isin(np.arange(len(frequencies)), selected)
        )
        if not len(candidates):
            raise ValueError(f"no Hessian mode has the required {label} symmetry")
        overlaps = np.asarray(
            [
                abs(np.einsum("ia,ia,i->", normalized[index], template, PHENOL_MASSES))
                for index in candidates
            ]
        )
        frequency_score = np.exp(
            -0.5 * (np.log(frequencies[candidates] / float(target)) / 0.28) ** 2
        )
        score = 0.65 * overlaps + 0.35 * frequency_score
        index = int(candidates[np.argmax(score)])
        phase = np.einsum(
            "ia,ia,i->", normalized[index], template, PHENOL_MASSES
        )
        if phase < 0.0:
            normalized[index] *= -1.0
        selected.append(index)
        diagnostics.append(
            {
                "label": label,
                "index": index,
                "frequency_cm1": float(frequencies[index]),
                "reflection_parity": float(parities[index]),
                "template_overlap": float(overlaps[np.argmax(score)]),
                "selection_score": float(np.max(score)),
            }
        )
    reflection = np.diag((1.0, 1.0, -1.0))
    projected = np.asarray(
        [
            0.5 * (mode + parity * (mode @ reflection.T))
            for mode, parity in zip(normalized[selected], (-1.0, 1.0))
        ]
    )
    active = _mass_orthonormalize(projected, equilibrium)
    return active, tuple(diagnostics)


@dataclass(frozen=True)
class PhenolReactiveChart:
    r"""Five-dimensional chart $(R_{OH},\phi,\theta,Q_{16a},Q_{8a})$.

    Distances and Cartesian normal coordinates are in Angstrom and angles are
    in radians.  The normal-mode vectors are mass-orthonormal Cartesian
    displacements, so each scalar amplitude has an unambiguous scale.
    """

    modes: np.ndarray | None = None

    names = (
        "R_OH_angstrom",
        "phi_CCOH_radian",
        "theta_COH_radian",
        "Q_16a_angstrom_sqrt_amu",
        "Q_8a_angstrom_sqrt_amu",
    )
    default_bounds = np.asarray(
        (
            (0.82, 3.50),
            (-np.pi, np.pi),
            (np.deg2rad(94.0), np.deg2rad(124.0)),
            (-0.75, 0.75),
            (-0.30, 0.30),
        )
    )

    def __post_init__(self):
        modes = phenol_template_modes() if self.modes is None else np.asarray(self.modes)
        if modes.shape != (2, len(PHENOL_SPECIES), 3):
            raise ValueError("modes must have shape (2, 13, 3)")
        gram = np.einsum("kia,lia,i->kl", modes, modes, PHENOL_MASSES, optimize=True)
        if not np.allclose(gram, np.eye(2), atol=1.0e-8):
            raise ValueError("modes must be mass-orthonormal")
        object.__setattr__(self, "modes", modes.copy())

    @property
    def equilibrium(self):
        return np.asarray(
            (Phenol3D.r_eq * au2angstrom, 0.0, Phenol3D.theta_eq, 0.0, 0.0)
        )

    def geometry(self, coordinate):
        coordinate = np.asarray(coordinate, dtype=float)
        if coordinate.shape != (5,):
            raise ValueError("phenol coordinate must contain five values")
        r_oh, torsion, bend, q_16a, q_8a = coordinate
        geometry = _base_geometry(r_oh, torsion, bend)
        geometry += q_16a * self.modes[0] + q_8a * self.modes[1]
        return geometry

    def geometries(self, coordinates):
        return np.asarray([self.geometry(point) for point in coordinates])

    def with_modes(self, modes):
        return type(self)(modes=np.asarray(modes, dtype=float))

    @staticmethod
    def coordinate_to_atomic(coordinate):
        """Convert the public 5D chart to atomic mass-weighted coordinates."""

        coordinate = np.asarray(coordinate, dtype=float)
        if coordinate.shape != (5,):
            raise ValueError("phenol coordinate must contain five values")
        scale = np.asarray(
            (
                1.0 / au2angstrom,
                1.0,
                1.0,
                1.0 / (au2angstrom * np.sqrt(au2amu)),
                1.0 / (au2angstrom * np.sqrt(au2amu)),
            )
        )
        return coordinate * scale

    @staticmethod
    def coordinate_from_atomic(coordinate):
        """Convert atomic mass-weighted coordinates to the public 5D chart."""

        coordinate = np.asarray(coordinate, dtype=float)
        if coordinate.shape != (5,):
            raise ValueError("phenol coordinate must contain five values")
        scale = np.asarray(
            (
                au2angstrom,
                1.0,
                1.0,
                au2angstrom * np.sqrt(au2amu),
                au2angstrom * np.sqrt(au2amu),
            )
        )
        return coordinate * scale

    def jax_map(self):
        """Return a differentiable atomic-coordinate-to-Cartesian map."""

        import jax
        from jax import numpy as jnp

        jax.config.update("jax_enable_x64", True)
        modes = jnp.asarray(self.modes * np.sqrt(au2amu))
        r_cc = 1.394 / au2angstrom
        r_co = 1.360 / au2angstrom
        r_ch = 1.084 / au2angstrom
        phase = jnp.arange(6) * jnp.pi / 3.0

        def geometry(coordinate):
            r_oh, torsion, bend, q_16a, q_8a = coordinate
            carbons = r_cc * jnp.column_stack(
                (jnp.cos(phase), jnp.sin(phase), jnp.zeros(6))
            )
            oxygen = carbons[0] + jnp.asarray((r_co, 0.0, 0.0))
            direction = (
                jnp.cos(bend) * jnp.asarray((-1.0, 0.0, 0.0))
                + jnp.sin(bend)
                * jnp.stack((0.0 * torsion, jnp.cos(torsion), jnp.sin(torsion)))
            )
            hydroxyl_h = oxygen + r_oh * direction
            radial = carbons[1:] / jnp.linalg.norm(carbons[1:], axis=1)[:, None]
            ring_h = carbons[1:] + r_ch * radial
            base = jnp.vstack((carbons, oxygen, hydroxyl_h, ring_h))
            return base + q_16a * modes[0] + q_8a * modes[1]

        return geometry

    def model_dpem(self, coordinate):
        """Five-dimensional extension of the published two-coordinate DPEM.

        The equilibrium-bend, zero-normal-mode cut is exactly ``dpes1``.  The
        added state-specific harmonic terms are a workflow benchmark, not an
        ab-initio phenol surface.
        """

        r_oh, torsion, bend, q_16a, q_8a = map(float, coordinate)
        matrix = np.asarray(dpes1(r_oh / au2angstrom, torsion), dtype=float)
        bend_frequency = np.asarray((1200.0, 900.0, 700.0)) * wavenumber2hartree
        bend_shift = np.deg2rad(np.asarray((0.0, 5.0, -7.0)))
        bend_delta = bend - Phenol3D.theta_eq
        bend_inertia = Phenol3D(
            [Phenol3D.r_eq], [Phenol3D.theta_eq], [0.0]
        ).bend_inertia
        matrix[np.diag_indices(3)] += bend_inertia * bend_frequency**2 * (
            0.5 * bend_delta**2 - bend_shift * bend_delta
        )
        # State-dependent normal-coordinate wells in Hartree.  The curvatures
        # are expressed directly in this reduced mass-weighted chart.
        frequencies = np.asarray(((850.0, 610.0), (720.0, 540.0), (930.0, 460.0)))
        curvature = (frequencies * wavenumber2hartree) ** 2 * 1.2e4
        shifts = np.asarray(((0.00, 0.00), (0.055, -0.035), (-0.040, 0.045)))
        q = np.asarray((q_16a, q_8a))
        matrix[np.diag_indices(3)] += np.sum(
            0.5 * curvature * (q[None, :] - shifts) ** 2
            - 0.5 * curvature * shifts**2,
            axis=1,
        )
        matrix[0, 2] = matrix[2, 0] = (
            0.035 / au2ev * q_16a / 0.75
        )
        return matrix
