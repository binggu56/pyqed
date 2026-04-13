#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical integration helpers for AO-based DFT.
"""

import numpy as np
from gbasis.evals.eval import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis
from periodictable import elements
from scipy.integrate._lebedev import get_lebedev_sphere


BOHR_TO_ANGSTROM = 0.529177249
DEFAULT_N_RADIAL = 50
DEFAULT_N_ANGULAR = 110


class AOGrid:
    """
    Real-space grid carrying AO values for numerical XC integration.

    Parameters
    ----------
    ao : ndarray, shape (ngrids, nao)
        AO basis values on grid points.
    weights : ndarray, shape (ngrids,)
        Quadrature weights.
    coords : ndarray, optional
        Grid coordinates with shape (ngrids, 3).
    """

    def __init__(self, ao, weights, coords=None, ao_grad=None):
        self.ao = np.asarray(ao, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.coords = None if coords is None else np.asarray(coords, dtype=float)
        self.ao_grad = None if ao_grad is None else np.asarray(ao_grad, dtype=float)

        if self.ao.ndim != 2:
            raise ValueError("ao must be a 2D array with shape (ngrids, nao).")

        if self.weights.ndim != 1:
            raise ValueError("weights must be a 1D array with shape (ngrids,).")

        if self.ao.shape[0] != self.weights.shape[0]:
            raise ValueError("ao and weights must use the same number of grid points.")

        if self.coords is not None:
            if self.coords.ndim != 2 or self.coords.shape[1] != 3:
                raise ValueError("coords must have shape (ngrids, 3).")
            if self.coords.shape[0] != self.weights.shape[0]:
                raise ValueError("coords and weights must use the same number of grid points.")

        if self.ao_grad is not None:
            if self.ao_grad.ndim != 3 or self.ao_grad.shape[0] != 3:
                raise ValueError("ao_grad must have shape (3, ngrids, nao).")
            if self.ao_grad.shape[1] != self.weights.shape[0]:
                raise ValueError("ao_grad and weights must use the same number of grid points.")
            if self.ao_grad.shape[2] != self.ao.shape[1]:
                raise ValueError("ao_grad and ao must use the same number of AOs.")

    @classmethod
    def from_molecule(cls, mol, coords, weights, screen_basis=True, tol_screen=1e-8,
                      with_grad=False):
        """
        Build an AOGrid from a ``pyqed.qchem.Molecule`` with ``mol._bas``.
        """
        if getattr(mol, '_bas', None) is None:
            raise ValueError("mol._bas is not available. Call mol.build(driver='gbasis') first.")

        coords = np.asarray(coords, dtype=float)
        weights = np.asarray(weights, dtype=float)
        ao = evaluate_basis(
            mol._bas,
            coords,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        ).T
        ao_grad = None
        if with_grad:
            ao_grad = np.stack([
                evaluate_deriv_basis(
                    mol._bas,
                    coords,
                    np.array([1, 0, 0]),
                    screen_basis=screen_basis,
                    tol_screen=tol_screen,
                ).T,
                evaluate_deriv_basis(
                    mol._bas,
                    coords,
                    np.array([0, 1, 0]),
                    screen_basis=screen_basis,
                    tol_screen=tol_screen,
                ).T,
                evaluate_deriv_basis(
                    mol._bas,
                    coords,
                    np.array([0, 0, 1]),
                    screen_basis=screen_basis,
                    tol_screen=tol_screen,
                ).T,
            ], axis=0)
        return cls(ao=ao, weights=weights, coords=coords, ao_grad=ao_grad)

    @classmethod
    def atom_centered(cls, mol, n_radial=DEFAULT_N_RADIAL,
                      n_angular=DEFAULT_N_ANGULAR, radial_scale=1.0,
                      angular_grid='lebedev', radial_grid='treutler_ahlrichs',
                      screen_basis=True, tol_screen=1e-8, with_grad=False):
        """
        Build a simple atom-centered quadrature grid with Becke-style partitioning.

        Notes
        -----
        This is a lightweight implementation intended for AO-based DFT prototyping.
        It uses
        - a product radial-angular grid for each atom;
        - Treutler-Ahlrichs radial nodes by default;
        - Lebedev angular points by default;
        - Becke-style smooth partition weights without hetero-atomic radius shifts.

        The default size is chosen to be noticeably more accurate than the
        original prototype settings while staying lightweight for small
        molecules.
        """
        coords, weights = atom_centered_grid(
            mol,
            n_radial=n_radial,
            n_angular=n_angular,
            radial_scale=radial_scale,
            angular_grid=angular_grid,
            radial_grid=radial_grid,
        )
        return cls.from_molecule(
            mol,
            coords,
            weights,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
            with_grad=with_grad,
        )

    def attach_gradients(self, mol, screen_basis=True, tol_screen=1e-8):
        """
        Populate AO gradients in place using the stored grid coordinates.
        """
        if self.coords is None:
            raise ValueError("Grid coordinates are required to build AO gradients.")
        if getattr(mol, '_bas', None) is None:
            raise ValueError("mol._bas is not available. Call mol.build(driver='gbasis') first.")

        self.ao_grad = np.stack([
            evaluate_deriv_basis(
                mol._bas,
                self.coords,
                np.array([1, 0, 0]),
                screen_basis=screen_basis,
                tol_screen=tol_screen,
            ).T,
            evaluate_deriv_basis(
                mol._bas,
                self.coords,
                np.array([0, 1, 0]),
                screen_basis=screen_basis,
                tol_screen=tol_screen,
            ).T,
            evaluate_deriv_basis(
                mol._bas,
                self.coords,
                np.array([0, 0, 1]),
                screen_basis=screen_basis,
                tol_screen=tol_screen,
            ).T,
        ], axis=0)
        return self

    @property
    def ngrids(self):
        return self.weights.size

    @property
    def nao(self):
        return self.ao.shape[1]


def density_on_grid(dm, ao):
    """
    Electron density on a numerical grid.
    """
    return np.einsum('gu,uv,gv->g', ao, dm, ao, optimize=True).real


def density_gradient_on_grid(dm, ao, ao_grad):
    """
    Density gradient on a numerical grid.
    """
    term1 = np.einsum('kgu,uv,gv->kg', ao_grad, dm, ao, optimize=True)
    term2 = np.einsum('gu,uv,kgv->kg', ao, dm, ao_grad, optimize=True)
    return (term1 + term2).real


def build_local_potential_matrix(values, weights, ao):
    """
    Build AO matrix for a local potential V(r).
    """
    return np.einsum('g,gu,gv->uv', weights * values, ao, ao, optimize=True)


def build_gga_potential_matrix(vrho, vsigma, rho_grad, weights, ao, ao_grad):
    """
    Build AO matrix for a restricted GGA potential.
    """
    mat = build_local_potential_matrix(vrho, weights, ao)
    weighted_grad = 2.0 * rho_grad * (weights * vsigma)[None, :]
    mat += np.einsum('kg,gu,kgv->uv', weighted_grad, ao, ao_grad, optimize=True)
    mat += np.einsum('kg,kgu,gv->uv', weighted_grad, ao_grad, ao, optimize=True)
    return mat


def xc_energy_from_grid(rho, eps_xc, weights):
    """
    Exchange-correlation energy from density on the integration grid.
    """
    return np.dot(weights, rho * eps_xc).real


def cartesian_box_grid(xlim, ylim, zlim, nx, ny=None, nz=None):
    """
    Uniform Cartesian quadrature on a rectangular box.

    Returns
    -------
    coords : ndarray, shape (ngrids, 3)
    weights : ndarray, shape (ngrids,)
    """
    if ny is None:
        ny = nx
    if nz is None:
        nz = nx

    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    z = np.linspace(zlim[0], zlim[1], nz)
    dx = x[1] - x[0] if nx > 1 else (xlim[1] - xlim[0])
    dy = y[1] - y[0] if ny > 1 else (ylim[1] - ylim[0])
    dz = z[1] - z[0] if nz > 1 else (zlim[1] - zlim[0])

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    weights = np.full(coords.shape[0], dx * dy * dz)
    return coords, weights


def atom_centered_grid(mol, n_radial=DEFAULT_N_RADIAL,
                       n_angular=DEFAULT_N_ANGULAR, radial_scale=1.0,
                       angular_grid='lebedev', radial_grid='treutler_ahlrichs'):
    """
    Atom-centered quadrature coordinates and weights for a molecule.

    Parameters
    ----------
    mol : pyqed.qchem.Molecule-like
        Must provide ``atom_coords()``, ``atom_symbols()``, and ``natom``.
    n_radial : int
        Number of radial shells per atom.
    n_angular : int
        Number of angular points per shell.
    radial_scale : float
        Multiplier applied to the element covalent radius.
    radial_grid : str
        Radial quadrature rule. Supported: ``'treutler_ahlrichs'``,
        ``'mura_knowles'``, ``'rational'``.
    """
    atom_coords = np.asarray(mol.atom_coords(), dtype=float)
    atom_symbols = mol.atom_symbols()
    natom = len(atom_symbols)

    if angular_grid == 'lebedev':
        directions, angular_weights = lebedev_sphere(n_angular)
    elif angular_grid == 'fibonacci':
        directions, angular_weights = fibonacci_sphere(n_angular)
    else:
        raise ValueError("angular_grid must be 'lebedev' or 'fibonacci'.")

    all_coords = []
    all_local_weights = []
    owners = []

    for a, symbol in enumerate(atom_symbols):
        scale = radial_scale * default_atomic_radius(symbol)
        radial_points, radial_weights = radial_grid_points(
            n_radial,
            scale=scale,
            scheme=radial_grid,
        )

        atom_grid = atom_coords[a] + radial_points[:, None, None] * directions[None, :, :]
        atom_weights = radial_weights[:, None] * angular_weights[None, :]

        all_coords.append(atom_grid.reshape(-1, 3))
        all_local_weights.append(atom_weights.ravel())
        owners.append(np.full(n_radial * n_angular, a, dtype=int))

    coords = np.vstack(all_coords)
    local_weights = np.concatenate(all_local_weights)
    owners = np.concatenate(owners)

    partition = becke_partition(coords, atom_coords)
    weights = local_weights * partition[owners, np.arange(coords.shape[0])]

    return coords, weights


def default_atomic_radius(symbol):
    """
    Default per-atom radial scale in bohr based on the covalent radius.
    """
    elem = elements.isotope(symbol)
    radius_angstrom = getattr(elem, 'covalent_radius', None)
    if radius_angstrom is None:
        radius_angstrom = 0.7
    return radius_angstrom / BOHR_TO_ANGSTROM


def radial_grid_points(n_radial, scale=1.0, scheme='treutler_ahlrichs'):
    """
    Semi-infinite radial quadrature on [0, inf).

    Parameters
    ----------
    n_radial : int
        Number of radial nodes.
    scale : float
        Atomic scale in bohr.
    scheme : str
        ``'treutler_ahlrichs'`` for a logarithmic M4 grid, ``'mura_knowles'``
        for a log-cubic grid, or ``'rational'`` for the original algebraic map.
    """
    if scheme == 'treutler_ahlrichs':
        step = np.pi / (n_radial + 1.0)
        angle = np.arange(1, n_radial + 1, dtype=float) * step
        x = np.cos(angle)
        ln2 = scale / np.log(2.0)

        r = -ln2 * (1.0 + x) ** 0.6 * np.log((1.0 - x) / 2.0)
        dr = step * np.sin(angle) * ln2 * (1.0 + x) ** 0.6
        dr *= (-0.6 / (1.0 + x) * np.log((1.0 - x) / 2.0) + 1.0 / (1.0 - x))

        r = r[::-1]
        weights = dr[::-1] * r ** 2
    elif scheme == 'mura_knowles':
        x, wx = np.polynomial.legendre.leggauss(n_radial)
        t = np.clip(0.5 * (x + 1.0), 1e-14, 1.0 - 1e-14)
        wt = 0.5 * wx

        # Logarithmic radial map with denser sampling near the nuclei and a
        # smoother description of the asymptotic tail.
        r = -scale * np.log(1.0 - t ** 3)
        dr_dt = scale * (3.0 * t ** 2) / (1.0 - t ** 3)
        weights = wt * dr_dt * r ** 2
    elif scheme == 'rational':
        x, wx = np.polynomial.legendre.leggauss(n_radial)
        t = np.clip(0.5 * (x + 1.0), 1e-14, 1.0 - 1e-14)
        wt = 0.5 * wx

        r = scale * t / (1.0 - t)
        dr_dt = scale / (1.0 - t) ** 2
        weights = wt * dr_dt * r ** 2
    else:
        raise ValueError(
            "radial_grid must be 'treutler_ahlrichs', 'mura_knowles', or 'rational'."
        )
    return r, weights


def fibonacci_sphere(n_angular):
    """
    Equal-weight angular quadrature points on the unit sphere.
    """
    idx = np.arange(n_angular, dtype=float)
    z = 1.0 - 2.0 * (idx + 0.5) / n_angular
    rho = np.sqrt(np.clip(1.0 - z ** 2, 0.0, None))
    phi = np.pi * (3.0 - np.sqrt(5.0)) * idx

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    directions = np.column_stack((x, y, z))
    weights = np.full(n_angular, 4.0 * np.pi / n_angular)
    return directions, weights


def lebedev_sphere(n_angular):
    """
    True Lebedev angular quadrature on the unit sphere.

    Parameters
    ----------
    n_angular : int
        Number of Lebedev points, e.g. 6, 14, 26, 38, 50, 74, ...
    """
    try:
        leb = get_lebedev_sphere(n_angular)
    except Exception as exc:
        raise ValueError(
            "Unsupported Lebedev grid size. "
            "Choose one of 6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, "
            "230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, "
            "2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810."
        ) from exc
    directions = np.column_stack((leb.x, leb.y, leb.z))
    weights = np.asarray(leb.w, dtype=float)
    return directions, weights


def becke_partition(coords, atom_coords):
    """
    Smooth partition of unity over atoms using the standard Becke switching form.
    """
    coords = np.asarray(coords, dtype=float)
    atom_coords = np.asarray(atom_coords, dtype=float)

    natom = atom_coords.shape[0]
    npts = coords.shape[0]
    p = np.ones((natom, npts), dtype=float)

    dist = np.linalg.norm(coords[None, :, :] - atom_coords[:, None, :], axis=2)

    for a in range(natom):
        for b in range(a + 1, natom):
            rab_vec = atom_coords[a] - atom_coords[b]
            rab = np.linalg.norm(rab_vec)
            if rab < 1e-14:
                continue

            mu = (dist[a] - dist[b]) / rab
            g = becke_switch(mu)
            s_ab = 0.5 * (1.0 - g)
            p[a] *= s_ab
            p[b] *= (1.0 - s_ab)

    denom = p.sum(axis=0)
    denom[denom == 0] = 1.0
    return p / denom


def becke_switch(mu):
    """
    Repeated Becke polynomial smoothing.
    """
    x = np.clip(np.asarray(mu, dtype=float), -1.0, 1.0)
    for _ in range(3):
        x = 0.5 * x * (3.0 - x ** 2)
    return x
