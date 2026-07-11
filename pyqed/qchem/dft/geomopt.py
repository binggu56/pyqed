#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry optimization for native AO-based RKS.
"""

import os
import tempfile
import uuid
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

from pyqed.qchem.mol import Molecule

from .grid import AOGrid, BOHR_TO_ANGSTROM
from .hessian import analyze_cartesian_hessian
from .xc import needs_gradients


@dataclass
class GeometryOptimizationResult:
    mf: object
    result: object
    coords: np.ndarray
    energy: float
    gradient: np.ndarray
    trajectory: list
    backend: str
    approximate_hessian: np.ndarray | None = None
    approximate_inverse_hessian: np.ndarray | None = None
    exact_hessian: np.ndarray | None = None

    def hessian(self, exact=False, inverse=False):
        """
        Return an optimizer Hessian in Cartesian coordinates.

        Parameters
        ----------
        exact : bool
            Return the exact final Cartesian Hessian when available.
        inverse : bool
            Return the approximate inverse Hessian when available.
        """
        if exact and inverse:
            raise ValueError("exact and inverse cannot both be True.")
        if exact:
            if self.exact_hessian is None:
                raise ValueError("No exact final Hessian is available for this optimization.")
            return self.exact_hessian
        if inverse:
            if self.approximate_inverse_hessian is None:
                raise ValueError("No approximate inverse Hessian is available for this optimization.")
            return self.approximate_inverse_hessian
        if self.approximate_hessian is None:
            raise ValueError("No approximate Hessian is available for this optimization.")
        return self.approximate_hessian

    def vibrational_analysis(
        self,
        exact=False,
        remove_translation_rotation=True,
        negative_imaginary=True,
        zero_tol=1e-7,
    ):
        """
        Analyze a Cartesian Hessian into vibrational frequencies and normal modes.
        """
        return analyze_cartesian_hessian(
            self.hessian(exact=exact),
            self.coords,
            self.mf.mol.atom_mass_list(),
            remove_translation_rotation=remove_translation_rotation,
            negative_imaginary=negative_imaginary,
            zero_tol=zero_tol,
        )

    def frequencies(self, exact=False, unit='cm^-1', **kwargs):
        """
        Convenience accessor for vibrational frequencies.
        """
        data = self.vibrational_analysis(exact=exact, **kwargs)
        unit = unit.lower()
        if unit in ('cm^-1', 'cm-1', 'wavenumber', 'wavenumbers'):
            return data['freq_cm1']
        if unit in ('au', 'a.u.', 'hartree'):
            return data['freq_au']
        raise ValueError("unit must be 'cm^-1' or 'au'.")


def _copy_molecule(mol):
    return Molecule(
        atom=deepcopy(mol.atom),
        charge=mol.charge,
        spin=mol.spin,
        basis=mol.basis,
        unit='bohr',
    )


def _build_grid(mf, mol):
    grid = getattr(mf, 'grid', None)
    if grid is None:
        return AOGrid.atom_centered(mol, with_grad=needs_gradients(mf.xc))

    if getattr(grid, 'kind', None) != 'atom_centered':
        raise NotImplementedError(
            "Native geometry optimization currently supports only atom-centered grids."
        )

    settings = dict(getattr(grid, 'settings', {}))
    settings.setdefault('with_grad', needs_gradients(mf.xc))
    return AOGrid.atom_centered(mol, **settings)


def _approximate_hessian_from_inverse(inv_hessian):
    if inv_hessian is None:
        return None, None
    mat = np.asarray(inv_hessian, dtype=float)
    if mat.ndim != 2:
        return None, None
    return np.linalg.pinv(mat), mat

def _evaluate_geometry(mf, mol_template, coords, trajectory, callback):
    mol = _copy_molecule(mol_template)
    mol.set_geom(coords)
    mol.build(driver='gbasis')

    grid = _build_grid(mf, mol)
    step_mf = mf.__class__(mol, grid=grid, xc=mf.xc, init_guess=mf.init_guess)
    step_mf.max_cycle = mf.max_cycle
    step_mf.conv_tol = mf.conv_tol
    step_mf.damping = mf.damping
    step_mf.verbose = mf.verbose
    step_mf.run()

    grad = step_mf.nuc_grad_method().run()
    record = {
        'coords': np.asarray(coords, dtype=float).copy(),
        'energy': float(step_mf.e_tot),
        'gradient': np.asarray(grad, dtype=float).copy(),
        'mf': step_mf,
    }
    trajectory.append(record)

    if callback is not None:
        callback(record['coords'], record['energy'], record['gradient'], step_mf)

    return record


def _optimize_geometry_scipy(mf, method='BFGS', maxiter=50, gtol=1e-3, callback=None):
    trajectory = []
    mol_template = _copy_molecule(mf.mol)
    x0 = mol_template.atom_coords().reshape(-1)
    cache = {'x': None, 'value': None}
    last = {'record': None}

    def evaluate(x):
        x = np.asarray(x, dtype=float)
        if cache['x'] is not None and np.allclose(x, cache['x']):
            return cache['value']

        coords = np.asarray(x, dtype=float).reshape(mol_template.natom, 3)
        record = _evaluate_geometry(mf, mol_template, coords, trajectory, callback)
        flat_grad = record['gradient'].reshape(-1)

        last['record'] = record
        cache['x'] = x.copy()
        cache['value'] = (record['energy'], flat_grad)
        return cache['value']

    result = minimize(
        fun=lambda x: evaluate(x)[0],
        x0=x0,
        jac=lambda x: evaluate(x)[1],
        method=method,
        options={'gtol': gtol, 'maxiter': maxiter},
    )

    if last['record'] is None or not np.allclose(result.x, last['record']['coords'].reshape(-1)):
        evaluate(result.x)

    final = last['record']
    approx_hessian, approx_inv_hessian = _approximate_hessian_from_inverse(
        getattr(result, 'hess_inv', None)
    )
    return GeometryOptimizationResult(
        mf=final['mf'],
        result=result,
        coords=final['coords'],
        energy=final['energy'],
        gradient=final['gradient'],
        trajectory=trajectory,
        backend='scipy',
        approximate_hessian=approx_hessian,
        approximate_inverse_hessian=approx_inv_hessian,
    )


def _load_geometric():
    try:
        import geometric
        import geometric.molecule
        from geometric import engine
        from geometric.errors import GeomOptNotConvergedError
    except ImportError as exc:
        raise ImportError(
            "geomeTRIC is not installed. Install it with `pip install geometric` "
            "to use backend='geometric'."
        ) from exc

    try:
        from geometric import internal, nifty, optimize
        internal.ang2bohr = optimize.ang2bohr = nifty.ang2bohr = 1.0 / BOHR_TO_ANGSTROM
        engine.bohr2ang = internal.bohr2ang = geometric.molecule.bohr2ang = (
            nifty.bohr2ang
        ) = optimize.bohr2ang = BOHR_TO_ANGSTROM
    except Exception:
        pass

    return geometric, engine, GeomOptNotConvergedError


def _write_quiet_log_ini(dirname):
    path = os.path.join(dirname, 'geometric_quiet.ini')
    logfile = os.path.join(dirname, 'geometric.log')
    with open(path, 'w', encoding='ascii') as fh:
        fh.write(
            "[loggers]\n"
            "keys=root\n\n"
            "[handlers]\n"
            "keys=file_handler\n\n"
            "[formatters]\n"
            "keys=formatter\n\n"
            "[logger_root]\n"
            "level=WARNING\n"
            "handlers=file_handler\n\n"
            "[handler_file_handler]\n"
            "class=logging.FileHandler\n"
            "level=WARNING\n"
            "formatter=formatter\n"
            f"args=(r'{logfile}', 'w')\n\n"
            "[formatter_formatter]\n"
            "format=%(message)s\n"
        )
    return path


def _optimize_geometry_geometric(mf, maxiter=50, callback=None, **kwargs):
    geometric, engine_module, GeomOptNotConvergedError = _load_geometric()

    trajectory = []
    mol_template = _copy_molecule(mf.mol)
    last = {'record': None}

    class PyQEDGeometricEngine(engine_module.Engine):
        def __init__(self):
            molecule = geometric.molecule.Molecule()
            molecule.elem = [mol_template.atom_symbol(i) for i in range(mol_template.natom)]
            molecule.xyzs = [mol_template.atom_coords() * BOHR_TO_ANGSTROM]
            super().__init__(molecule)
            self.cycle = 0

        def calc_new(self, coords, dirname):
            self.cycle += 1
            coords = np.asarray(coords, dtype=float).reshape(mol_template.natom, 3)
            record = _evaluate_geometry(mf, mol_template, coords, trajectory, callback)
            last['record'] = record
            return {
                'energy': record['energy'],
                'gradient': record['gradient'].reshape(-1),
            }

    optimizer_engine = PyQEDGeometricEngine()
    kwargs = dict(kwargs)
    kwargs['maxiter'] = maxiter

    with tempfile.TemporaryDirectory() as tmpdir:
        input_stub = os.path.join(tmpdir, str(uuid.uuid4()))
        approx_hessian_file = kwargs.get('write_cart_hess')
        if approx_hessian_file is None:
            approx_hessian_file = os.path.join(tmpdir, 'approx_hessian.txt')
            kwargs['write_cart_hess'] = approx_hessian_file
        if 'verbose' not in kwargs:
            kwargs['verbose'] = 0
        if 'logIni' not in kwargs:
            kwargs['logIni'] = _write_quiet_log_ini(tmpdir)
        try:
            optimize_result = geometric.optimize.run_optimizer(
                customengine=optimizer_engine,
                input=input_stub,
                **kwargs,
            )
            success = True
            message = 'Optimization converged.'
        except GeomOptNotConvergedError as exc:
            optimize_result = exc
            success = False
            message = f'Optimization did not converge in {maxiter} steps.'

        approx_hessian = None
        if os.path.exists(approx_hessian_file):
            approx_hessian = np.loadtxt(approx_hessian_file)

        exact_hessian = None
        exact_hessian_file = os.path.join(f"{input_stub}.tmp", 'hessian', 'hessian.txt')
        if os.path.exists(exact_hessian_file):
            exact_hessian = np.loadtxt(exact_hessian_file)

    final = last['record']
    if final is None:
        raise RuntimeError("geomeTRIC did not return any optimization step.")

    result = SimpleNamespace(
        success=success,
        message=message,
        niter=optimizer_engine.cycle,
        raw=optimize_result,
    )
    return GeometryOptimizationResult(
        mf=final['mf'],
        result=result,
        coords=final['coords'],
        energy=final['energy'],
        gradient=final['gradient'],
        trajectory=trajectory,
        backend='geometric',
        approximate_hessian=approx_hessian,
        exact_hessian=exact_hessian,
    )


def optimize_geometry(
    mf,
    backend='scipy',
    method='BFGS',
    maxiter=50,
    gtol=1e-3,
    callback=None,
    **kwargs,
):
    """
    Optimize the nuclear geometry using native RKS energies and gradients.

    Parameters
    ----------
    mf : pyqed.qchem.dft.RKS
        Reference calculation carrying molecule, XC functional, and SCF options.
    backend : str
        ``'scipy'`` for Cartesian BFGS or ``'geometric'`` for geomeTRIC.
    method : str
        Optimization method passed to ``scipy.optimize.minimize`` for the
        ``'scipy'`` backend.
    maxiter : int
        Maximum geometry steps.
    gtol : float
        Convergence target on the Cartesian gradient norm for the ``'scipy'``
        backend.
    callback : callable, optional
        Callback receiving ``(coords, energy, gradient, mf_step)`` after each
        energy/gradient evaluation.
    """
    backend = backend.lower()
    if backend == 'scipy':
        return _optimize_geometry_scipy(
            mf,
            method=method,
            maxiter=maxiter,
            gtol=gtol,
            callback=callback,
        )
    if backend == 'geometric':
        return _optimize_geometry_geometric(
            mf,
            maxiter=maxiter,
            callback=callback,
            **kwargs,
        )
    raise ValueError("backend must be either 'scipy' or 'geometric'.")
