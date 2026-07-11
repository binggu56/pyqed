#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical integration helpers for AO-based DFT.
"""

import numpy as np
from periodictable import elements
from scipy.integrate._lebedev import get_lebedev_sphere

from pyqed.qchem.basis import ContractedGaussian


BOHR_TO_ANGSTROM = 0.529177249
DEFAULT_N_RADIAL = 50
DEFAULT_N_ANGULAR = 110


def _evaluate_basis_compat(basis, coords, screen_basis=True, tol_screen=1e-8):
    """
    Compatibility wrapper for different ``gbasis`` evaluator signatures.
    """
    from gbasis.evals.eval import evaluate_basis

    try:
        return evaluate_basis(
            basis,
            coords,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )
    except TypeError:
        return evaluate_basis(basis, coords)


def _evaluate_deriv_basis_compat(basis, coords, orders, screen_basis=True, tol_screen=1e-8):
    """
    Compatibility wrapper for different ``gbasis`` derivative-evaluator signatures.
    """
    from gbasis.evals.eval_deriv import evaluate_deriv_basis

    try:
        return evaluate_deriv_basis(
            basis,
            coords,
            orders,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )
    except TypeError:
        return evaluate_deriv_basis(basis, coords, orders)


def _is_native_basis(basis):
    return basis is not None and len(basis) > 0 and all(
        isinstance(fn, ContractedGaussian) for fn in basis
    )


def _basis_and_transform(mol):
    basis = getattr(mol, '_bas_cart', None)
    transform = getattr(mol, '_ao_cart2sph', None)
    if basis is None:
        basis = getattr(mol, '_bas', None)
        transform = None
    if basis is None:
        raise ValueError("mol._bas is not available. Build the molecule before making an AO grid.")
    return basis, transform


def _pow_nonnegative(values, power):
    if power < 0:
        return np.zeros_like(values, dtype=float)
    return np.power(values, power)


def _gaussian_axis_derivative_poly(values, exponents, power, order):
    values = np.asarray(values, dtype=float)[None, :]
    exponents = np.asarray(exponents, dtype=float)[:, None]
    power = int(power)
    if order == 0:
        poly = _pow_nonnegative(values, power)
    elif order == 1:
        poly = (
            power * _pow_nonnegative(values, power - 1)
            - 2.0 * exponents * _pow_nonnegative(values, power + 1)
        )
    elif order == 2:
        poly = (
            power * (power - 1) * _pow_nonnegative(values, power - 2)
            - 2.0 * exponents * (2 * power + 1) * _pow_nonnegative(values, power)
            + 4.0 * exponents * exponents * _pow_nonnegative(values, power + 2)
        )
    else:
        raise NotImplementedError("Native AO grid derivatives are implemented up to second order.")
    return poly


def _evaluate_native_basis_derivative(basis, coords, orders):
    return _evaluate_native_basis_derivatives(basis, coords, [orders])[0]


def _evaluate_native_basis_derivatives(basis, coords, orders_list):
    coords = np.asarray(coords, dtype=float)
    orders_list = [tuple(int(order) for order in orders) for orders in orders_list]
    for orders in orders_list:
        if len(orders) != 3 or any(order < 0 for order in orders):
            raise ValueError("orders must be a length-3 non-negative derivative order.")
        if sum(orders) > 2:
            raise NotImplementedError("Native AO grid derivatives are implemented up to second order.")

    outputs = [
        np.empty((len(basis), coords.shape[0]), dtype=float)
        for _orders in orders_list
    ]
    for idx, fn in enumerate(basis):
        rel = coords - np.asarray(fn.origin, dtype=float)
        shell = tuple(int(v) for v in fn.shell)
        exponents = np.asarray(fn.exps, dtype=float)
        radial = np.exp(
            -exponents[:, None] * np.einsum('gi,gi->g', rel, rel)[None, :]
        )
        axis_poly = {}
        for axis in range(3):
            for order in {orders[axis] for orders in orders_list}:
                axis_poly[axis, order] = _gaussian_axis_derivative_poly(
                    rel[:, axis],
                    exponents,
                    shell[axis],
                    order,
                )

        weights = np.asarray(fn.prim_weights, dtype=float)
        for out, orders in zip(outputs, orders_list):
            primitive_values = radial.copy()
            for axis in range(3):
                primitive_values *= axis_poly[axis, orders[axis]]
            out[idx] = weights @ primitive_values
    return outputs


def _transform_native_ao(values, transform):
    if transform is None:
        return np.asarray(values, dtype=float)
    return np.asarray(transform, dtype=float).T @ np.asarray(values, dtype=float)


def _evaluate_ao(mol, coords, screen_basis=True, tol_screen=1e-8):
    return _evaluate_ao_derivatives(
        mol,
        coords,
        [(0, 0, 0)],
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )[0]


def _evaluate_ao_derivative(mol, coords, orders, screen_basis=True, tol_screen=1e-8):
    return _evaluate_ao_derivatives(
        mol,
        coords,
        [orders],
        screen_basis=screen_basis,
        tol_screen=tol_screen,
    )[0]


def _evaluate_ao_derivatives(mol, coords, orders_list, screen_basis=True, tol_screen=1e-8):
    basis, transform = _basis_and_transform(mol)
    if _is_native_basis(basis):
        return [
            _transform_native_ao(values, transform).T
            for values in _evaluate_native_basis_derivatives(basis, coords, orders_list)
        ]

    out = []
    for orders in orders_list:
        if tuple(int(order) for order in orders) == (0, 0, 0):
            values = _evaluate_basis_compat(
                basis,
                coords,
                screen_basis=screen_basis,
                tol_screen=tol_screen,
            ).T
        else:
            values = _evaluate_deriv_basis_compat(
                basis,
                coords,
                np.asarray(orders, dtype=int),
                screen_basis=screen_basis,
                tol_screen=tol_screen,
            ).T
        out.append(values)
    return out


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

    def __init__(self, ao, weights, coords=None, ao_grad=None, ao_hess=None,
                 owners=None, local_weights=None, kind='custom',
                 moves_with_atoms=False, settings=None):
        self.ao = np.asarray(ao, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.coords = None if coords is None else np.asarray(coords, dtype=float)
        self.ao_grad = None if ao_grad is None else np.asarray(ao_grad, dtype=float)
        self.ao_hess = None if ao_hess is None else np.asarray(ao_hess, dtype=float)
        self.owners = None if owners is None else np.asarray(owners, dtype=int)
        self.local_weights = (
            None if local_weights is None else np.asarray(local_weights, dtype=float)
        )
        self.kind = kind
        self.moves_with_atoms = bool(moves_with_atoms)
        self.settings = {} if settings is None else dict(settings)

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

        if self.ao_hess is not None:
            if self.ao_hess.ndim != 4 or self.ao_hess.shape[0] != 3 or self.ao_hess.shape[1] != 3:
                raise ValueError("ao_hess must have shape (3, 3, ngrids, nao).")
            if self.ao_hess.shape[2] != self.weights.shape[0]:
                raise ValueError("ao_hess and weights must use the same number of grid points.")
            if self.ao_hess.shape[3] != self.ao.shape[1]:
                raise ValueError("ao_hess and ao must use the same number of AOs.")

        if self.owners is not None:
            if self.owners.ndim != 1 or self.owners.shape[0] != self.weights.shape[0]:
                raise ValueError("owners must have shape (ngrids,).")

        if self.local_weights is not None:
            if self.local_weights.ndim != 1 or self.local_weights.shape[0] != self.weights.shape[0]:
                raise ValueError("local_weights must have shape (ngrids,).")

    @classmethod
    def from_molecule(cls, mol, coords, weights, screen_basis=True, tol_screen=1e-8,
                      with_grad=False, with_hess=False, kind='custom',
                      moves_with_atoms=False, owners=None, local_weights=None,
                      settings=None):
        """
        Build an AOGrid from a ``pyqed.qchem.Molecule`` with an attached AO basis.
        """
        coords = np.asarray(coords, dtype=float)
        weights = np.asarray(weights, dtype=float)
        requested_orders = [(0, 0, 0)]
        grad_orders = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        if with_grad:
            requested_orders.extend(grad_orders)
        hess_orders = (
            ((0, 0), (2, 0, 0)),
            ((0, 1), (1, 1, 0)),
            ((0, 2), (1, 0, 1)),
            ((1, 1), (0, 2, 0)),
            ((1, 2), (0, 1, 1)),
            ((2, 2), (0, 0, 2)),
        )
        if with_hess:
            requested_orders.extend(orders for _indices, orders in hess_orders)

        evaluated = _evaluate_ao_derivatives(
            mol,
            coords,
            requested_orders,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )
        values_by_order = dict(zip(requested_orders, evaluated))
        ao = values_by_order[(0, 0, 0)]
        ao_grad = None
        if with_grad:
            ao_grad = np.stack([values_by_order[orders] for orders in grad_orders], axis=0)
        ao_hess = None
        if with_hess:
            ao_hess = np.empty((3, 3, coords.shape[0], ao.shape[1]), dtype=float)
            for (i, j), orders in hess_orders:
                values = values_by_order[orders]
                ao_hess[i, j] = values
                ao_hess[j, i] = values
        return cls(
            ao=ao,
            weights=weights,
            coords=coords,
            ao_grad=ao_grad,
            ao_hess=ao_hess,
            owners=owners,
            local_weights=local_weights,
            kind=kind,
            moves_with_atoms=moves_with_atoms,
            settings=settings,
        )

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
        coords, weights, owners, local_weights = atom_centered_grid(
            mol,
            n_radial=n_radial,
            n_angular=n_angular,
            radial_scale=radial_scale,
            angular_grid=angular_grid,
            radial_grid=radial_grid,
            return_metadata=True,
        )
        return cls.from_molecule(
            mol,
            coords,
            weights,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
            with_grad=with_grad,
            kind='atom_centered',
            moves_with_atoms=True,
            owners=owners,
            local_weights=local_weights,
            settings={
                'n_radial': n_radial,
                'n_angular': n_angular,
                'radial_scale': radial_scale,
                'angular_grid': angular_grid,
                'radial_grid': radial_grid,
                'screen_basis': screen_basis,
                'tol_screen': tol_screen,
                'with_grad': with_grad,
            },
        )

    def attach_gradients(self, mol, screen_basis=True, tol_screen=1e-8):
        """
        Populate AO gradients in place using the stored grid coordinates.
        """
        if self.coords is None:
            raise ValueError("Grid coordinates are required to build AO gradients.")

        grad_orders = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.ao_grad = np.stack(
            _evaluate_ao_derivatives(
                mol,
                self.coords,
                grad_orders,
                screen_basis=screen_basis,
                tol_screen=tol_screen,
            ),
            axis=0,
        )
        return self

    def attach_hessians(self, mol, screen_basis=True, tol_screen=1e-8):
        """
        Populate AO Hessians in place using the stored grid coordinates.
        """
        if self.coords is None:
            raise ValueError("Grid coordinates are required to build AO Hessians.")

        self.ao_hess = np.empty((3, 3, self.coords.shape[0], self.ao.shape[1]), dtype=float)
        hess_orders = (
            ((0, 0), (2, 0, 0)),
            ((0, 1), (1, 1, 0)),
            ((0, 2), (1, 0, 1)),
            ((1, 1), (0, 2, 0)),
            ((1, 2), (0, 1, 1)),
            ((2, 2), (0, 0, 2)),
        )
        evaluated = _evaluate_ao_derivatives(
            mol,
            self.coords,
            [orders for _indices, orders in hess_orders],
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )
        for ((i, j), _orders), values in zip(hess_orders, evaluated):
            self.ao_hess[i, j] = values
            self.ao_hess[j, i] = values
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


def density_hessian_on_grid(dm, ao, ao_grad, ao_hess):
    """
    Density Hessian on a numerical grid.
    """
    term1 = np.einsum('klgu,uv,gv->klg', ao_hess, dm, ao, optimize=True)
    term2 = np.einsum('kgu,uv,lgv->klg', ao_grad, dm, ao_grad, optimize=True)
    term3 = np.einsum('lgu,uv,kgv->klg', ao_grad, dm, ao_grad, optimize=True)
    term4 = np.einsum('gu,uv,klgv->klg', ao, dm, ao_hess, optimize=True)
    return (term1 + term2 + term3 + term4).real


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
                       angular_grid='lebedev', radial_grid='treutler_ahlrichs',
                       return_metadata=False):
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

    if return_metadata:
        return coords, weights, owners, local_weights
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


def becke_weight_response(coords, atom_coords, owners, local_weights):
    """
    Derivative of atom-centered Becke weights with respect to nuclear positions.

    Returns
    -------
    dweights : ndarray, shape (natom, npts, 3)
        ``dweights[a, g, k] = d w_g / d R_{a,k}``.
    """
    coords = np.asarray(coords, dtype=float)
    atom_coords = np.asarray(atom_coords, dtype=float)
    owners = np.asarray(owners, dtype=int)
    local_weights = np.asarray(local_weights, dtype=float)

    natom = atom_coords.shape[0]
    npts = coords.shape[0]
    if owners.shape != (npts,):
        raise ValueError("owners must have shape (npts,).")
    if local_weights.shape != (npts,):
        raise ValueError("local_weights must have shape (npts,).")

    dweights = np.zeros((natom, npts, 3), dtype=float)
    point_owner_mask = np.eye(natom, dtype=float)[owners]

    diff = coords[None, :, :] - atom_coords[:, None, :]
    dist = np.linalg.norm(diff, axis=2)
    unit = np.zeros_like(diff)
    mask = dist > 1e-14
    unit[mask] = diff[mask] / dist[mask, None]

    for target in range(natom):
        p = np.ones((natom, npts), dtype=float)
        dp = np.zeros((natom, npts, 3), dtype=float)
        owner_move = point_owner_mask[:, target][:, None]

        for a in range(natom):
            for b in range(a + 1, natom):
                rab_vec = atom_coords[a] - atom_coords[b]
                rab = np.linalg.norm(rab_vec)
                if rab < 1e-14:
                    continue

                mu = (dist[a] - dist[b]) / rab
                g, gp = becke_switch_with_derivative(mu)
                s = 0.5 * (1.0 - g)

                dda = (owner_move - float(a == target)) * unit[a]
                ddb = (owner_move - float(b == target)) * unit[b]
                drab = (float(a == target) - float(b == target)) * (rab_vec / rab)
                dmu = (dda - ddb - mu[:, None] * drab[None, :]) / rab
                ds = -0.5 * gp[:, None] * dmu

                pa = p[a].copy()
                pb = p[b].copy()
                dpa = dp[a].copy()
                dpb = dp[b].copy()

                p[a] = pa * s
                dp[a] = dpa * s[:, None] + pa[:, None] * ds

                one_minus_s = 1.0 - s
                p[b] = pb * one_minus_s
                dp[b] = dpb * one_minus_s[:, None] - pb[:, None] * ds

        denom = p.sum(axis=0)
        denom[denom == 0] = 1.0
        ddenom = dp.sum(axis=0)

        owner_p = p[owners, np.arange(npts)]
        owner_dp = dp[owners, np.arange(npts), :]
        owner_partition_grad = (
            owner_dp * denom[:, None] - owner_p[:, None] * ddenom
        ) / (denom[:, None] ** 2)
        dweights[target] = local_weights[:, None] * owner_partition_grad

    return dweights


def becke_switch(mu):
    """
    Repeated Becke polynomial smoothing.
    """
    x = np.clip(np.asarray(mu, dtype=float), -1.0, 1.0)
    for _ in range(3):
        x = 0.5 * x * (3.0 - x ** 2)
    return x


def becke_switch_with_derivative(mu):
    """
    Repeated Becke polynomial smoothing and its derivative.
    """
    mu = np.asarray(mu, dtype=float)
    x = np.clip(mu, -1.0, 1.0)
    dx_dmu = np.where((mu >= -1.0) & (mu <= 1.0), 1.0, 0.0)
    for _ in range(3):
        fprime = 1.5 * (1.0 - x ** 2)
        x = 0.5 * x * (3.0 - x ** 2)
        dx_dmu = fprime * dx_dmu
    return x, dx_dmu
