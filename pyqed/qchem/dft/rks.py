#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AO-based restricted Kohn-Sham DFT.
"""

import numpy as np

from .grid import AOGrid
from .scf import build_xc, get_j, get_k, make_rdm1, run_rks, ks_energy
from .xc import hybrid_coeff, needs_gradients


class RKS:
    """
    Restricted Kohn-Sham DFT with AO integrals and numerical XC integration.

    Notes
    -----
    This implementation is intentionally lightweight:
    - it does not depend on PySCF;
    - it assumes ``mol`` already carries ``overlap``, ``hcore``, ``eri``,
      ``nao``, ``nelec``, and optionally ``e_nuc``;
    - it currently supports ``xc='lda'``/``'lda_x'``,
      ``xc='vwn'``/``'lda_c_vwn'``, ``xc='svwn'``/``'lda,vwn'``, and
      ``xc='b3lyp'``;
    - if ``grid`` is omitted, a default atom-centered grid is generated.
    """

    def __init__(self, mol, grid=None, xc='lda_x', init_guess='hcore'):
        self.mol = mol
        if grid is None:
            self.grid = AOGrid.atom_centered(mol, with_grad=needs_gradients(xc))
        else:
            self.grid = grid
        self.xc = xc
        self.init_guess = init_guess
        self.max_cycle = 50
        self.conv_tol = 1e-8
        self.damping = 0.25
        self.verbose = 0

        self.e_tot = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.dm = None
        self.hcore = None
        self.j = None
        self.k = None
        self.vxc = None
        self.fock = None
        self.rho = None
        self.exc = None
        self.converged = False
        self.hyb = hybrid_coeff(xc)

        self.nocc = self.mol.nelec // 2
        self.nao = self.mol.nao
        self.nmo = self.mol.nao
        self.nelec = self.mol.nelec
        self._ensure_grid_for_xc(self.xc)

    def _ensure_grid_for_xc(self, xc):
        if not needs_gradients(xc):
            return

        if getattr(self.grid, 'ao_grad', None) is not None:
            return

        if getattr(self.grid, 'coords', None) is not None:
            self.grid.attach_gradients(self.mol)
            return

        # Some lightweight grids only carry AO values and weights.  GGA and
        # hybrid-GGA functionals need AO gradients, so rebuild a default
        # atom-centered grid that includes them.
        self.grid = AOGrid.atom_centered(self.mol, with_grad=True)

    def make_rdm1(self):
        if self.mo_coeff is None or self.mo_occ is None:
            raise ValueError("No converged orbitals are available yet.")
        return make_rdm1(self.mo_coeff, self.mo_occ)

    def get_hcore(self):
        return self.mol.hcore

    def get_ovlp(self):
        return self.mol.overlap

    def energy_nuc(self):
        return self.mol.energy_nuc()

    def nuc_grad_method(self):
        from .grad import Gradients
        return Gradients(self)

    def optimize_geometry(
        self,
        backend='scipy',
        method='BFGS',
        maxiter=50,
        gtol=1e-3,
        callback=None,
        **kwargs,
    ):
        from .geomopt import optimize_geometry
        return optimize_geometry(
            self,
            backend=backend,
            method=method,
            maxiter=maxiter,
            gtol=gtol,
            callback=callback,
            **kwargs,
        )

    def get_j(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        return get_j(self.mol, dm)

    def get_veff(self, dm=None):
        self._ensure_grid_for_xc(self.xc)
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        _, _, vxc_mat = build_xc(dm, self.grid, xc=self.xc)
        veff = self.get_j(dm) + vxc_mat
        hyb = hybrid_coeff(self.xc)
        if hyb != 0.0:
            veff = veff - 0.5 * hyb * get_k(self.mol, dm)
        return veff

    def get_fock(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        return self.get_hcore() + self.get_veff(dm)

    def energy_elec(self, dm=None):
        self._ensure_grid_for_xc(self.xc)
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        j = get_j(self.mol, dm)
        k = get_k(self.mol, dm) if hybrid_coeff(self.xc) != 0.0 else None
        _, exc, vxc_mat = build_xc(dm, self.grid, xc=self.xc)
        return ks_energy(
            dm,
            self.get_hcore(),
            j,
            exc,
            vxc_mat,
            e_nuc=0.0,
            k=k,
            hyb=hybrid_coeff(self.xc),
        )

    def run(self, dm0=None, **kwargs):
        xc = kwargs.get('xc', self.xc)
        self._ensure_grid_for_xc(xc)
        out = run_rks(
            self.mol,
            self.grid,
            dm0=dm0,
            init_guess=kwargs.get('init_guess', self.init_guess),
            xc=xc,
            max_cycle=kwargs.get('max_cycle', self.max_cycle),
            conv_tol=kwargs.get('conv_tol', self.conv_tol),
            damping=kwargs.get('damping', self.damping),
            verbose=kwargs.get('verbose', self.verbose),
        )

        self.xc = xc
        self.converged = out['converged']
        self.e_tot = out['e_tot']
        self.mo_energy = out['mo_energy']
        self.mo_coeff = out['mo_coeff']
        self.mo_occ = out['mo_occ']
        self.dm = out['dm']
        self.hcore = out['hcore']
        self.j = out['j']
        self.k = out['k']
        self.vxc = out['vxc']
        self.fock = out['fock']
        self.rho = out['rho']
        self.exc = out['exc']
        self.hyb = out['hyb']
        self.grid = out['grid']
        return self

    def RTTDDFT(self, interaction_ao=None, field=None, **kwargs):
        """
        Convenience constructor for real-time TDDFT propagation.
        """
        from pyqed.qchem.rttddft import RTTDDFT
        return RTTDDFT(self, interaction_ao=interaction_ao, field=field, **kwargs)

    def TDA(self):
        """
        Convenience constructor for linear-response TDA.
        """
        from pyqed.qchem.lrtddft import TDA
        return TDA(self)

    def TDDFT(self):
        """
        Convenience constructor for linear-response TDDFT.
        """
        from pyqed.qchem.lrtddft import TDDFT
        return TDDFT(self)
