#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native nuclear gradients for AO-based RKS.
"""

import numpy as np

from pyqed.qchem.mol import grad_nuc

from .grid import (
    becke_weight_response,
    density_gradient_on_grid,
    density_hessian_on_grid,
    density_on_grid,
)
from .xc import eval_xc, hybrid_coeff, xc_type


class Gradients:
    """
    Analytic nuclear gradients for native ``pyqed.qchem.dft.RKS`` objects.

    Notes
    -----
    Native frozen-grid LDA and pure-GGA paths are implemented. More general cases still
    need additional machinery:
    - moving-grid response for atom-centered quadratures;
    - hybrid exchange response for hybrid functionals.
    """

    def __init__(self, mf):
        self.base = mf
        self.mol = mf.mol
        self.grid = mf.grid
        self.de = None
        self.de_elec = None
        self.de_nuc = None
        self._cbasis = None

    def dump_flags(self):
        return self

    def _require_scf(self):
        if getattr(self.base, 'dm', None) is None:
            raise ValueError(
                "Run the RKS calculation before requesting nuclear gradients."
            )

    def _unsupported_reasons(self):
        reasons = [
            "AO derivative integrals require a local libcint-compatible backend.",
        ]

        if xc_type(self.base.xc) not in ('LDA', 'GGA'):
            reasons.append(
                f"XC functional '{self.base.xc}' needs native {xc_type(self.base.xc)} response terms."
            )

        hyb = hybrid_coeff(self.base.xc)
        if hyb != 0.0:
            reasons.append(
                f"XC functional '{self.base.xc}' uses exact exchange (hyb={hyb:.6f}), "
                "so exchange-gradient terms are also required."
            )

        if getattr(self.grid, 'moves_with_atoms', False):
            if getattr(self.grid, 'owners', None) is None:
                reasons.append(
                    "moving grids need per-point owner metadata to evaluate quadrature response."
                )
            if getattr(self.grid, 'local_weights', None) is None:
                reasons.append(
                    "moving grids need local quadrature weights to evaluate Becke-weight response."
                )

        return reasons

    def _gradient_supported(self):
        return (
            hybrid_coeff(self.base.xc) == 0.0
            and xc_type(self.base.xc) in ('LDA', 'GGA')
            and (
                not getattr(self.grid, 'moves_with_atoms', False)
                or (
                    getattr(self.grid, 'owners', None) is not None
                    and getattr(self.grid, 'local_weights', None) is not None
                )
            )
        )

    def _ensure_ao_grad(self):
        if getattr(self.grid, 'ao_grad', None) is None:
            if getattr(self.grid, 'coords', None) is None:
                raise ValueError(
                    "Grid coordinates are required to build AO gradients for nuclear gradients."
                )
            self.grid.attach_gradients(self.mol)

    def _ensure_ao_hess(self):
        if getattr(self.grid, 'ao_hess', None) is None:
            if getattr(self.grid, 'coords', None) is None:
                raise ValueError(
                    "Grid coordinates are required to build AO Hessians for GGA nuclear gradients."
                )
            self.grid.attach_hessians(self.mol)

    def _build_cbasis(self):
        if self._cbasis is not None:
            return self._cbasis

        try:
            from pyqed.qchem._libcint import CBasis1e
        except Exception as exc:
            raise NotImplementedError(
                "Native analytic RKS gradients require the local libcint wrapper "
                "and a libcint-compatible shared library."
            ) from exc

        if getattr(self.mol, '_bas', None) is None:
            raise ValueError(
                "mol._bas is not available. Build the molecule with driver='gbasis' first."
            )

        coord_type = getattr(self.mol._bas[0], 'coord_type', 'spherical')
        self._cbasis = CBasis1e(
            self.mol._bas,
            self.mol.atom_symbols(),
            self.mol.atom_coords(),
            coord_type=coord_type,
        )
        return self._cbasis

    def _move_comp_axis_first(self, arr):
        arr = np.asarray(arr)
        if arr.ndim == 2:
            return arr[None, ...]
        return np.moveaxis(arr, -1, 0)

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None:
            mo_energy = self.base.mo_energy
        if mo_coeff is None:
            mo_coeff = self.base.mo_coeff
        if mo_occ is None:
            mo_occ = self.base.mo_occ

        mo0 = mo_coeff[:, mo_occ > 0]
        mo0e = mo0 * (mo_energy[mo_occ > 0] * mo_occ[mo_occ > 0])
        return np.dot(mo0e, mo0.T.conj())

    def get_ovlp(self):
        cbas = self._build_cbasis()
        ipovlp = self._move_comp_axis_first(
            cbas.int1e('int1e_ipovlp', components=(3,), hermi=False)
        )
        return -ipovlp

    def hcore_generator(self):
        cbas = self._build_cbasis()
        ipkin = self._move_comp_axis_first(
            cbas.int1e('int1e_ipkin', components=(3,), hermi=False)
        )
        ipnuc = self._move_comp_axis_first(
            cbas.int1e('int1e_ipnuc', components=(3,), hermi=False)
        )
        h1 = -(ipkin + ipnuc)

        def hcore_deriv(atm_id):
            p0, p1 = cbas.ao_slice_by_atom(atm_id)
            vrinv = self._move_comp_axis_first(
                cbas.int1e(
                    'int1e_iprinv',
                    components=(3,),
                    inv_origin=self.mol.atom_coord(atm_id),
                    hermi=False,
                )
            )
            vrinv *= -self.mol.atom_charge(atm_id)
            vrinv[:, p0:p1, :] += h1[:, p0:p1, :]
            return vrinv + vrinv.transpose(0, 2, 1)

        return hcore_deriv

    def _hartree_potential_on_grid(self, dm):
        cbas = self._build_cbasis()
        coords = np.asarray(self.grid.coords, dtype=float)
        v_h = np.zeros(coords.shape[0], dtype=float)
        for g, coord in enumerate(coords):
            rinv = cbas.int1e('int1e_rinv', inv_origin=coord)
            v_h[g] = np.einsum('uv,vu->', dm, rinv).real
        return v_h

    def get_jk(self, dm=None):
        if dm is None:
            dm = self.base.dm

        cbas = self._build_cbasis()
        try:
            from pyscf.scf import _vhf
        except Exception as exc:
            raise NotImplementedError(
                "Native analytic RKS gradients require a local libcvhf backend "
                "to build two-electron derivative contractions."
            ) from exc

        intor = 'int2e_ip1' + cbas._suffix
        vj, vk = _vhf.direct_mapdm(
            intor,
            's2kl',
            ('lk->s1ij', 'jk->s1il'),
            np.asarray(dm),
            3,
            cbas.atm,
            cbas.bas,
            cbas.env,
        )
        return -np.asarray(vj), -np.asarray(vk)

    def get_veff(self, dm=None):
        if dm is None:
            dm = self.base.dm
        self._ensure_ao_grad()

        rho = density_on_grid(dm, self.grid.ao)
        vj, _ = self.get_jk(dm)
        if xc_type(self.base.xc) == 'LDA':
            _, v_xc = eval_xc(rho, xc=self.base.xc)
            v_eff = np.asarray(v_xc, dtype=float)
            vxc = -np.einsum(
                'g,kgu,gv->kuv',
                self.grid.weights * v_eff,
                self.grid.ao_grad,
                self.grid.ao,
                optimize=True,
            )
        else:
            self._ensure_ao_hess()
            rho_grad = density_gradient_on_grid(dm, self.grid.ao, self.grid.ao_grad)
            _, (vrho, vsigma) = eval_xc(rho, xc=self.base.xc, grad_rho=rho_grad)
            weighted_vrho = self.grid.weights * np.asarray(vrho, dtype=float)
            weighted_gga = 2.0 * self.grid.weights * np.asarray(vsigma, dtype=float)

            vxc = -np.einsum(
                'g,kgu,gv->kuv',
                weighted_vrho,
                self.grid.ao_grad,
                self.grid.ao,
                optimize=True,
            )
            vxc -= np.einsum(
                'g,lg,lkgu,gv->kuv',
                weighted_gga,
                rho_grad,
                self.grid.ao_hess,
                self.grid.ao,
                optimize=True,
            )
            vxc -= np.einsum(
                'g,lg,kgu,lgv->kuv',
                weighted_gga,
                rho_grad,
                self.grid.ao_grad,
                self.grid.ao_grad,
                optimize=True,
            )
        return vj + vxc

    def get_xc_grid_response(self, dm=None, atmlst=None):
        if dm is None:
            dm = self.base.dm

        natm = self.mol.natom
        if atmlst is None:
            atmlst = tuple(range(natm))
        else:
            atmlst = tuple(atmlst)

        if not getattr(self.grid, 'moves_with_atoms', False):
            return np.zeros((len(atmlst), 3), dtype=float)

        if getattr(self.grid, 'owners', None) is None or getattr(self.grid, 'local_weights', None) is None:
            raise NotImplementedError(
                "Moving-grid XC response requires grid owner and local-weight metadata."
            )

        self._ensure_ao_grad()
        rho = density_on_grid(dm, self.grid.ao)
        rho_grad = density_gradient_on_grid(dm, self.grid.ao, self.grid.ao_grad)
        weights = np.asarray(self.grid.weights, dtype=float)
        owners = np.asarray(self.grid.owners, dtype=int)
        dweights = becke_weight_response(
            self.grid.coords,
            self.mol.atom_coords(),
            owners,
            self.grid.local_weights,
        )

        if xc_type(self.base.xc) == 'LDA':
            eps_xc, v_xc = eval_xc(rho, xc=self.base.xc)
            point_response = np.asarray(v_xc, dtype=float)[None, :] * rho_grad
        else:
            self._ensure_ao_hess()
            rho_hess = density_hessian_on_grid(
                dm,
                self.grid.ao,
                self.grid.ao_grad,
                self.grid.ao_hess,
            )
            eps_xc, (vrho, vsigma) = eval_xc(rho, xc=self.base.xc, grad_rho=rho_grad)
            point_response = np.asarray(vrho, dtype=float)[None, :] * rho_grad
            point_response += 2.0 * np.asarray(vsigma, dtype=float)[None, :] * np.einsum(
                'lg,mlg->mg',
                rho_grad,
                rho_hess,
                optimize=True,
            )

        exc_density = rho * np.asarray(eps_xc, dtype=float)
        owner_masks = np.eye(natm, dtype=float)[owners].T

        de = np.zeros((len(atmlst), 3), dtype=float)
        for k, ia in enumerate(atmlst):
            de[k] += np.einsum(
                'gx,g->x',
                dweights[ia],
                exc_density,
                optimize=True,
            ).real
            de[k] += np.einsum(
                'g,xg->x',
                owner_masks[ia] * weights,
                point_response,
                optimize=True,
            ).real
        return de

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        self._require_scf()
        if not self._gradient_supported():
            reasons = self._unsupported_reasons()
            msg = "Native analytic RKS electronic gradients are not implemented yet"
            if reasons:
                msg += ": " + " ".join(reasons)
            raise NotImplementedError(msg)

        dm0 = self.base.make_rdm1()
        dme0 = self.make_rdm1e(mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)
        hcore_deriv = self.hcore_generator()
        s1 = self.get_ovlp()
        vhf = self.get_veff(dm0)

        cbas = self._build_cbasis()
        natm = self.mol.natom
        if atmlst is None:
            atmlst = tuple(range(natm))
        else:
            atmlst = tuple(atmlst)

        de = np.zeros((len(atmlst), 3), dtype=float)
        for k, ia in enumerate(atmlst):
            p0, p1 = cbas.ao_slice_by_atom(ia)
            h1ao = hcore_deriv(ia)
            de[k] += np.einsum('xij,ij->x', h1ao, dm0, optimize=True).real
            de[k] += 2.0 * np.einsum(
                'xij,ij->x', vhf[:, p0:p1], dm0[p0:p1], optimize=True
            ).real
            de[k] -= 2.0 * np.einsum(
                'xij,ij->x', s1[:, p0:p1], dme0[p0:p1], optimize=True
            ).real
        de += self.get_xc_grid_response(dm=dm0, atmlst=atmlst)
        return de

    def electronic(self, atmlst=None):
        self._require_scf()
        return self.grad_elec(atmlst=atmlst)

    def nuclear(self, atmlst=None):
        return grad_nuc(self.mol, atmlst=atmlst)

    def run(self, atmlst=None):
        self.de_nuc = self.nuclear(atmlst=atmlst)
        self.de_elec = self.electronic(atmlst=atmlst)
        self.de = self.de_nuc + self.de_elec
        return self.de

    def kernel(self, atmlst=None):
        """
        Backward-compatible alias for ``run()``.
        """
        return self.run(atmlst=atmlst)

    def components(self, atmlst=None):
        """
        Return the gradient components that are already available.

        This helper is intentionally separate from ``kernel`` so callers do not
        accidentally confuse the exact nuclear-repulsion contribution with the
        full molecular gradient.
        """
        self.de_nuc = self.nuclear(atmlst=atmlst)
        de_elec = None
        if self._gradient_supported():
            try:
                de_elec = self.electronic(atmlst=atmlst)
            except Exception:
                de_elec = None
        return {
            'nuclear': self.de_nuc,
            'electronic': de_elec,
        }
