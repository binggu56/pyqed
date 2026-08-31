"""Adiabatically adjusting principal-axis coordinates for triatoms."""

from __future__ import annotations

import numpy as np


class APH:
    r"""Principal-axis Smith-Whitten coordinates used by APH scattering.

    The internal coordinates are ``(rho, theta, phi)`` with
    ``rho > 0``, ``0 <= theta <= pi/2``, and periodic ``phi``.  In the
    standard APH convention, ``theta=0`` is the equilateral north pole and
    ``theta=pi/2`` is the collinear equator.  The selected
    ``A + BC`` partition fixes the reference Jacobi vectors; for equal masses,
    shifts of ``phi`` by ``pi/3`` traverse the atom-permuted arrangement
    sectors.
    """

    coordinate_labels = ("rho", "theta", "phi")
    domains = ((0.0, np.inf), (0.0, 0.5 * np.pi), (0.0, 2.0 * np.pi))

    def __init__(self, atoms, masses, jacobi_atoms=(0, (1, 2))):
        atoms = tuple(str(atom).strip() for atom in atoms)
        if len(atoms) != 3 or any(not atom for atom in atoms):
            raise ValueError("atoms must contain three nonempty labels")
        masses = np.asarray(masses, dtype=float)
        if masses.shape != (3,):
            raise ValueError("APH coordinates require exactly three atomic masses")
        if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
            raise ValueError("masses must be positive and finite")

        self.atoms = atoms
        self.masses = masses
        self.natoms = 3
        self.jacobi_atoms = self._normalize_partition(jacobi_atoms)

        atom_a, atom_b, atom_c = self.jacobi_atoms
        mass_a, mass_b, mass_c = masses[[atom_a, atom_b, atom_c]]
        mass_bc = mass_b + mass_c
        total_mass = mass_a + mass_bc
        self.mu_r = mass_b * mass_c / mass_bc
        self.mu_R = mass_a * mass_bc / total_mass
        self.mu = np.sqrt(mass_a * mass_b * mass_c / total_mass)

    @staticmethod
    def _normalize_partition(jacobi_atoms):
        if not isinstance(jacobi_atoms, (tuple, list)):
            raise TypeError("jacobi_atoms must be (A, (B, C))")
        if len(jacobi_atoms) == 2:
            atom_a, pair = jacobi_atoms
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("jacobi_atoms must be (A, (B, C))")
            atom_b, atom_c = pair
        elif len(jacobi_atoms) == 3:
            atom_a, atom_b, atom_c = jacobi_atoms
        else:
            raise ValueError("jacobi_atoms must be (A, (B, C))")
        partition = tuple(int(atom) for atom in (atom_a, atom_b, atom_c))
        if sorted(partition) != [0, 1, 2]:
            raise ValueError("jacobi_atoms must contain each atom index exactly once")
        return partition

    def scaled_jacobi(self, coordinates, *, module=np):
        """Return the two mass-scaled Jacobi vectors in principal axes."""
        q = module.asarray(coordinates)
        if q.ndim != 1 or q.shape[0] != 3:
            raise ValueError("APH coordinates must have shape (3,)")
        rho, theta, phi = q
        principal_angle = 0.25 * np.pi - 0.5 * theta
        zero = 0.0 * rho
        x = module.stack(
            (
                rho * module.cos(principal_angle) * module.cos(phi),
                -rho * module.sin(principal_angle) * module.sin(phi),
                zero,
            )
        )
        y = module.stack(
            (
                rho * module.cos(principal_angle) * module.sin(phi),
                rho * module.sin(principal_angle) * module.cos(phi),
                zero,
            )
        )
        return module.stack((x, y))

    def cartesian(self, coordinates, *, module=np):
        """Map ``(rho, theta, phi)`` to a center-of-mass Cartesian geometry."""
        x, y = self.scaled_jacobi(coordinates, module=module)
        atom_a, atom_b, atom_c = self.jacobi_atoms
        mass_a, mass_b, mass_c = self.masses[[atom_a, atom_b, atom_c]]
        mass_bc = mass_b + mass_c
        total_mass = mass_a + mass_bc

        bond = x / module.sqrt(self.mu_r / self.mu)
        separation = y / module.sqrt(self.mu_R / self.mu)
        center_bc = -(mass_a / total_mass) * separation

        positions = [None, None, None]
        positions[atom_a] = center_bc + separation
        positions[atom_b] = center_bc - (mass_c / mass_bc) * bond
        positions[atom_c] = center_bc + (mass_b / mass_bc) * bond
        return module.stack(positions)

    def numpy_map(self):
        """Return a NumPy coordinate-to-Cartesian callable."""
        return lambda coordinates: self.cartesian(coordinates, module=np)

    def geometry(self, coordinates):
        """Return ``(atom, position)`` pairs for an electronic calculation."""
        positions = self.cartesian(coordinates)
        return tuple(
            (atom, np.array(position, copy=True))
            for atom, position in zip(self.atoms, positions)
        )

    def jax_map(self):
        """Return a JAX-differentiable coordinate-to-Cartesian callable."""
        from jax import numpy as jnp

        return lambda coordinates: self.cartesian(coordinates, module=jnp)

    def scaled_jacobi_from_cartesian(self, geometry):
        """Return mass-scaled Jacobi vectors from a Cartesian geometry."""
        geometry = np.asarray(geometry, dtype=float)
        if geometry.shape != (3, 3) or np.any(~np.isfinite(geometry)):
            raise ValueError("geometry must be a finite array with shape (3, 3)")
        atom_a, atom_b, atom_c = self.jacobi_atoms
        mass_a, mass_b, mass_c = self.masses[[atom_a, atom_b, atom_c]]
        mass_bc = mass_b + mass_c
        center_bc = (
            mass_b * geometry[atom_b] + mass_c * geometry[atom_c]
        ) / mass_bc
        bond = geometry[atom_c] - geometry[atom_b]
        separation = geometry[atom_a] - center_bc
        x = np.sqrt(self.mu_r / self.mu) * bond
        y = np.sqrt(self.mu_R / self.mu) * separation
        return np.stack((x, y))

    def from_cartesian(self, geometry):
        """Recover APH coordinates, up to the discrete principal-axis gauge."""
        x, y = self.scaled_jacobi_from_cartesian(geometry)
        rho = float(np.sqrt(np.dot(x, x) + np.dot(y, y)))
        if rho == 0.0:
            raise ValueError("APH coordinates are undefined at rho=0")

        plane_normal = np.cross(x, y)
        normal_norm = np.linalg.norm(plane_normal)
        if normal_norm <= 1.0e-12 * rho**2:
            raise ValueError("phi is undefined for a collinear geometry")
        axis_z = plane_normal / normal_norm

        dyadic = np.outer(x, x) + np.outer(y, y)
        eigenvalues, eigenvectors = np.linalg.eigh(dyadic)
        large = max(float(eigenvalues[-1]), 0.0)
        small = max(float(eigenvalues[-2]), 0.0)
        if abs(large - small) <= 1.0e-12 * rho**2:
            raise ValueError("principal axes are undefined at theta=0")
        axis_x = eigenvectors[:, -1]
        pivot = int(np.argmax(np.abs(axis_x)))
        if axis_x[pivot] < 0.0:
            axis_x = -axis_x
        principal_angle = float(np.arctan2(np.sqrt(small), np.sqrt(large)))
        theta = 0.5 * np.pi - 2.0 * principal_angle
        phi = float(
            np.mod(
                np.arctan2(np.dot(y, axis_x), np.dot(x, axis_x)),
                2.0 * np.pi,
            )
        )
        return np.array((rho, theta, phi))

    def hyperradius(self, geometry):
        """Evaluate the mass-scaled hyperradius of a Cartesian geometry."""
        vectors = self.scaled_jacobi_from_cartesian(geometry)
        return float(np.linalg.norm(vectors))

    def pair_distances(self, coordinates):
        """Return ``(r01, r02, r12)`` for one APH point."""
        geometry = self.cartesian(coordinates)
        return np.array(
            (
                np.linalg.norm(geometry[0] - geometry[1]),
                np.linalg.norm(geometry[0] - geometry[2]),
                np.linalg.norm(geometry[1] - geometry[2]),
            )
        )

    def metric(self, dvrs):
        """Sample the exact vibrational metric and Podolsky potential."""
        from pyqed.namd.polyspherical import sample_metric

        if len(tuple(dvrs)) != 3:
            raise ValueError("APH requires DVR axes for (rho, theta, phi)")
        return sample_metric(tuple(dvrs), self.masses, self.jax_map())

    def mpo(self, dvrs, *, return_fields=False, **kwargs):
        """Build the exact $J=0$ APH kinetic-energy operator as an MPO."""
        from pyqed.namd.polyspherical import metric_keo_mpo

        dvrs = tuple(dvrs)
        metric, pseudopotential = self.metric(dvrs)
        operator = metric_keo_mpo(
            dvrs, metric, pseudopotential, **kwargs
        )
        if return_fields:
            return operator, metric, pseudopotential
        return operator

    def matrix(self, dvrs, *, return_fields=False, **kwargs):
        """Build the exact $J=0$ APH kinetic-energy matrix."""
        value = self.mpo(dvrs, return_fields=return_fields, **kwargs)
        if return_fields:
            operator, metric, pseudopotential = value
            return operator.to_dense(), metric, pseudopotential
        return value.to_dense()

    def angular_hamiltonian(
        self,
        rho,
        dvrs,
        potential,
        *,
        return_fields=False,
        **kwargs,
    ):
        r"""Build the fixed-$\rho$ APH angular Hamiltonian $H_\Omega(\rho)$."""
        import jax
        from jax import numpy as jnp

        from pyqed.namd.keo import Gmat, pseudo
        from pyqed.namd.polyspherical import metric_keo_mpo

        dvrs = tuple(dvrs)
        if len(dvrs) != 2:
            raise ValueError("angular Hamiltonian requires (theta, phi) DVRs")
        shape = tuple(int(dvr.npts) for dvr in dvrs)
        mesh = jnp.meshgrid(*(dvr.x for dvr in dvrs), indexing="ij")
        angular = jnp.stack([axis.reshape(-1) for axis in mesh], axis=1)
        points = jnp.column_stack(
            (jnp.full(angular.shape[0], float(rho)), angular)
        )
        masses = jnp.asarray(self.masses)
        coordinate_map = self.jax_map()
        full_metric = jax.vmap(Gmat, in_axes=(0, None, None))(
            points, masses, coordinate_map
        )
        metric = np.asarray(full_metric[:, 1:3, 1:3]).reshape(*shape, 2, 2)
        pseudopotential = np.asarray(
            jax.vmap(pseudo, in_axes=(0, None, None))(
                points, masses, coordinate_map
            )
        ).reshape(shape)
        operator = metric_keo_mpo(
            dvrs, metric, pseudopotential, **kwargs
        ).to_dense()
        potential = np.asarray(potential)
        if potential.shape != shape:
            raise ValueError(
                f"potential shape {potential.shape} != angular shape {shape}"
            )
        operator = operator + np.diag(potential.reshape(-1))
        if return_fields:
            return operator, metric, pseudopotential
        return operator
