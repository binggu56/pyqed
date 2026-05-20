#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Point-charge electrostatic embedding for PyQED qchem methods."""

import numpy as np

from pyqed.qchem.basis import point_charge
from pyqed.qchem.basis_derivatives import (
    _atom_ids_for_basis,
    _axis_order,
    _contracted_one_deriv,
    eri_derivative_veff_scalar,
    one_electron_derivatives,
)
from pyqed.qchem.mol import grad_nuc


def embed_point_charges(mf, coords, charges, **kwargs):
    """Return an SCF-like object embedded in external point charges.

    Parameters
    ----------
    mf
        A PyQED mean-field object with ``mol`` and ``run`` attributes.
    coords
        External point-charge coordinates in Bohr, shape ``(ncharge, 3)``.
    charges
        External point charges in electron-charge units, shape ``(ncharge,)``.

    Notes
    -----
    The SCF one-electron Hamiltonian is augmented by
    ``-sum_A q_A / |r - R_A|``.  The nuclear-point-charge interaction is added
    to the reported total energy after each SCF call.
    """
    if _is_post_scf_method(mf):
        return PointChargeEmbeddedPostSCF(mf, coords, charges, **kwargs)
    return PointChargeEmbeddedSCF(mf, coords, charges, **kwargs)


class PointChargeEmbeddedSCF:
    """SCF wrapper with external point-charge electrostatic embedding."""

    def __init__(
        self,
        mf,
        coords,
        charges,
        fd_step=1e-4,
        analytic_qm_gradients=True,
        analytic_point_charge_forces=True,
        build_driver=None,
        run_kwargs=None,
        reference_run_kwargs=None,
    ):
        self.base = mf
        self.mf = mf
        self.mol = mf.mol
        self.coords = _as_coords(coords)
        self.charges = _as_charges(charges, len(self.coords))
        self.fd_step = float(fd_step)
        self.analytic_qm_gradients = bool(analytic_qm_gradients)
        self.analytic_point_charge_forces = bool(analytic_point_charge_forces)
        self.build_driver = build_driver or getattr(self.mol, "_build_driver", None) or "builtin"
        self.run_kwargs = {} if run_kwargs is None else dict(run_kwargs)
        self.reference_run_kwargs = (
            {} if reference_run_kwargs is None else dict(reference_run_kwargs)
        )
        self.e_ext_nuc = None
        self.e_tot = None

    def __getattr__(self, name):
        return getattr(self.mf, name)

    def run(self, **kwargs):
        """Run the embedded SCF calculation at the current geometry."""
        run_kwargs = dict(self.run_kwargs)
        run_kwargs.update(kwargs)
        energy = self._run_at(
            self.mol.atom_coords(),
            self.coords,
            run_kwargs=run_kwargs,
            update_self=True,
        )
        self.e_tot = energy
        return self

    def kernel(self, **kwargs):
        """Alias returning the embedded total energy."""
        self.run(**kwargs)
        return self.e_tot

    def energy_and_gradients(self, fd_step=None, run_kwargs=None):
        """Return ``(energy, qm_gradient, point_charge_forces)``.

        ``qm_gradient`` is ``dE/dR`` for the QM nuclei.  Point-charge values are
        returned as forces, i.e. ``-dE/dR_charge``.
        """
        step = self.fd_step if fd_step is None else float(fd_step)
        if step <= 0.0:
            raise ValueError("fd_step must be positive.")

        eval_kwargs = dict(self.run_kwargs)
        if run_kwargs is not None:
            eval_kwargs.update(run_kwargs)

        qm0 = np.asarray(self.mol.atom_coords(), dtype=float)
        pc0 = self.coords.copy()
        energy = self._run_at(qm0, pc0, run_kwargs=eval_kwargs, update_self=True)
        central_dm = self.mf.make_rdm1()

        if self.analytic_qm_gradients and _supports_embedded_rhf_gradient(self.mf):
            qm_grad = embedded_rhf_gradient(self.mf, pc0, self.charges)
        elif self.analytic_qm_gradients and _supports_embedded_rks_gradient(self.mf):
            qm_grad = embedded_rks_gradient(self.mf, pc0, self.charges)
        else:
            qm_grad = self._finite_difference_qm_gradient(qm0, pc0, step, eval_kwargs)

        if self.analytic_point_charge_forces and _supports_native_point_charge_forces(self.mol):
            point_forces = point_charge_forces(
                self.mol,
                central_dm,
                pc0,
                self.charges,
            )
        else:
            point_forces = self._finite_difference_point_charge_forces(
                qm0,
                pc0,
                step,
                eval_kwargs,
            )

        self._run_at(qm0, pc0, run_kwargs=eval_kwargs, update_self=True)
        return energy, qm_grad, point_forces

    def _finite_difference_qm_gradient(self, qm_coords, pc_coords, step, run_kwargs):
        qm_grad = np.zeros_like(qm_coords)
        for atom in range(qm_coords.shape[0]):
            for axis in range(3):
                plus = qm_coords.copy()
                minus = qm_coords.copy()
                plus[atom, axis] += step
                minus[atom, axis] -= step
                e_plus = self._run_at(plus, pc_coords, run_kwargs=run_kwargs)
                e_minus = self._run_at(minus, pc_coords, run_kwargs=run_kwargs)
                qm_grad[atom, axis] = (e_plus - e_minus) / (2.0 * step)
        return qm_grad

    def _finite_difference_point_charge_forces(self, qm_coords, pc_coords, step, run_kwargs):
        pc_grad = np.zeros_like(pc_coords)
        for charge_index in range(pc_coords.shape[0]):
            for axis in range(3):
                plus = pc_coords.copy()
                minus = pc_coords.copy()
                plus[charge_index, axis] += step
                minus[charge_index, axis] -= step
                e_plus = self._run_at(qm_coords, plus, run_kwargs=run_kwargs)
                e_minus = self._run_at(qm_coords, minus, run_kwargs=run_kwargs)
                pc_grad[charge_index, axis] = (e_plus - e_minus) / (2.0 * step)
        return -pc_grad

    def _run_at(self, qm_coords, pc_coords, run_kwargs=None, update_self=False):
        qm_coords = _as_coords(qm_coords)
        pc_coords = _as_coords(pc_coords)
        if qm_coords.shape[0] != self.mol.natom:
            raise ValueError(
                f"qm_coords has {qm_coords.shape[0]} atoms, expected {self.mol.natom}."
            )
        if pc_coords.shape[0] != self.charges.shape[0]:
            raise ValueError(
                f"pc_coords has {pc_coords.shape[0]} charges, expected {self.charges.shape[0]}."
            )

        self.mol.set_geom(qm_coords)
        self.mol.build(driver=self.build_driver)
        hcore0 = np.asarray(self.mol.hcore, dtype=float)
        self.mol.hcore = hcore0 + point_charge_hcore(self.mol, pc_coords, self.charges)

        kwargs = {} if run_kwargs is None else dict(run_kwargs)
        self.mf.run(**kwargs)
        e_ext_nuc = nuclear_point_charge_energy(self.mol, pc_coords, self.charges)
        energy = float(np.real(self.mf.e_tot + e_ext_nuc))
        self.mf.e_tot = energy

        if update_self:
            self.coords = pc_coords.copy()
            self.e_ext_nuc = e_ext_nuc
            self.e_tot = energy
        return energy


class PointChargeEmbeddedPostSCF:
    """Post-SCF wrapper with point-charge embedding of the reference method."""

    def __init__(
        self,
        method,
        coords,
        charges,
        fd_step=1e-4,
        analytic_qm_gradients=False,
        analytic_point_charge_forces=False,
        build_driver=None,
        run_kwargs=None,
        reference_run_kwargs=None,
    ):
        self.base = method
        self.method = method
        self.mf = method
        self.reference = method.mf
        self.mol = method.mol
        self.coords = _as_coords(coords)
        self.charges = _as_charges(charges, len(self.coords))
        self.fd_step = float(fd_step)
        self.analytic_qm_gradients = bool(analytic_qm_gradients)
        self.analytic_point_charge_forces = bool(analytic_point_charge_forces)
        self.build_driver = build_driver or getattr(self.mol, "_build_driver", None) or "builtin"
        self.run_kwargs = {} if run_kwargs is None else dict(run_kwargs)
        self.reference_run_kwargs = (
            {} if reference_run_kwargs is None else dict(reference_run_kwargs)
        )
        self.e_ext_nuc = None
        self.e_tot = None

    def __getattr__(self, name):
        return getattr(self.method, name)

    def run(self, **kwargs):
        run_kwargs = dict(self.run_kwargs)
        run_kwargs.update(kwargs)
        energy = self._run_at(
            self.mol.atom_coords(),
            self.coords,
            run_kwargs=run_kwargs,
            update_self=True,
        )
        self.e_tot = energy
        return self

    def kernel(self, **kwargs):
        self.run(**kwargs)
        return self.e_tot

    def _run_at(self, qm_coords, pc_coords, run_kwargs=None, update_self=False):
        qm_coords = _as_coords(qm_coords)
        pc_coords = _as_coords(pc_coords)
        if qm_coords.shape[0] != self.mol.natom:
            raise ValueError(
                f"qm_coords has {qm_coords.shape[0]} atoms, expected {self.mol.natom}."
            )
        if pc_coords.shape[0] != self.charges.shape[0]:
            raise ValueError(
                f"pc_coords has {pc_coords.shape[0]} charges, expected {self.charges.shape[0]}."
            )

        self.mol.set_geom(qm_coords)
        self.mol.build(driver=self.build_driver)
        hcore0 = np.asarray(self.mol.hcore, dtype=float)
        self.mol.hcore = hcore0 + point_charge_hcore(self.mol, pc_coords, self.charges)

        self.reference.run(**self.reference_run_kwargs)
        kwargs = {} if run_kwargs is None else dict(run_kwargs)
        self.method.run(**kwargs)
        e_ext_nuc = nuclear_point_charge_energy(self.mol, pc_coords, self.charges)
        energy = np.asarray(self.method.e_tot, dtype=float) + e_ext_nuc
        self.method.e_tot = energy

        if update_self:
            self.coords = pc_coords.copy()
            self.e_ext_nuc = e_ext_nuc
            self.e_tot = energy
        return energy


def point_charge_hcore(mol, coords, charges):
    """One-electron potential matrix from external point charges."""
    coords = _as_coords(coords)
    charges = _as_charges(charges, len(coords))

    if _has_gbasis_basis(mol):
        from gbasis.integrals.point_charge import point_charge_integral

        return np.sum(point_charge_integral(mol._bas, coords, charges), axis=-1)

    basis, transform = _basis_and_transform(mol)
    nao = len(basis)
    vext_cart = np.zeros((nao, nao), dtype=float)
    for i, bra in enumerate(basis):
        for j, ket in enumerate(basis[: i + 1]):
            value = 0.0
            for coord, charge in zip(coords, charges):
                value -= float(charge) * point_charge(bra, ket, coord)
            vext_cart[i, j] = value
            vext_cart[j, i] = value
    return _transform_one(vext_cart, transform)


def point_charge_hcore_derivatives(mol, coords, charges):
    """QM nuclear derivatives of the external point-charge hcore."""
    coords = _as_coords(coords)
    charges = _as_charges(charges, len(coords))

    basis, transform = _basis_and_transform(mol)
    qm_coords = np.asarray(mol.atom_coords(), dtype=float)
    atom_ids = _atom_ids_for_basis(basis, qm_coords)
    natm = qm_coords.shape[0]
    nao = len(basis)
    deriv_cart = np.zeros((natm, 3, nao, nao), dtype=float)

    for i, bra in enumerate(basis):
        atom_i = atom_ids[i]
        for j, ket in enumerate(basis[: i + 1]):
            atom_j = atom_ids[j]
            for coord, charge in zip(coords, charges):
                for axis in range(3):
                    order = _axis_order(axis)
                    bra_value = -float(charge) * _contracted_one_deriv(
                        bra,
                        ket,
                        "nuclear",
                        order_a=order,
                        center=coord,
                    )
                    _add_symmetric_derivative(deriv_cart, atom_i, axis, i, j, bra_value)
                    ket_value = -float(charge) * _contracted_one_deriv(
                        bra,
                        ket,
                        "nuclear",
                        order_b=order,
                        center=coord,
                    )
                    _add_symmetric_derivative(deriv_cart, atom_j, axis, i, j, ket_value)
    return _transform_one(deriv_cart, transform)


def embedded_rhf_gradient(mf, coords, charges):
    """Analytic QM nuclear gradient for native RHF with point-charge embedding."""
    mol = mf.mol
    coords = _as_coords(coords)
    charges = _as_charges(charges, len(coords))
    dm = np.asarray(mf.make_rdm1(), dtype=float)
    mo_coeff = np.asarray(mf.mo_coeff, dtype=float)
    mo_occ = np.asarray(mf.mo_occ, dtype=float)
    mo_energy = np.asarray(mf.mo_energy, dtype=float)
    occidx = mo_occ > 0
    cocc = mo_coeff[:, occidx]
    weighted_dm = np.einsum(
        "pi,qi,i->pq",
        cocc,
        cocc,
        mo_energy[occidx] * mo_occ[occidx],
        optimize=True,
    )

    s1 = one_electron_derivatives(mol, "overlap", order=1)
    h1 = one_electron_derivatives(mol, "hcore", order=1)
    h1 = h1 + point_charge_hcore_derivatives(mol, coords, charges)
    g1 = eri_derivative_veff_scalar(mol, dm, dm, order=1).reshape(mol.natom, 3)

    gradient = np.einsum("Axpq,qp->Ax", h1, dm, optimize=True)
    gradient += 0.5 * g1
    gradient -= np.einsum("Axpq,qp->Ax", s1, weighted_dm, optimize=True)
    gradient += grad_nuc(mol)
    gradient += nuclear_point_charge_gradient(mol, coords, charges)
    return np.asarray(np.real(gradient), dtype=float)


def embedded_rks_gradient(mf, coords, charges):
    """Analytic QM nuclear gradient for native RKS with point-charge embedding."""
    coords = _as_coords(coords)
    charges = _as_charges(charges, len(coords))
    grad = mf.nuc_grad_method()
    base_hcore_generator = grad.hcore_generator

    def hcore_generator():
        base = base_hcore_generator()
        cbas = grad._build_cbasis()
        external_terms = [
            float(charge)
            * grad._move_comp_axis_first(
                cbas.int1e(
                    "int1e_iprinv",
                    components=(3,),
                    inv_origin=coord,
                    hermi=False,
                )
            )
            for coord, charge in zip(coords, charges)
        ]

        def hcore_deriv(atom_id):
            out = np.array(base(atom_id), copy=True)
            p0, p1 = cbas.ao_slice_by_atom(atom_id)
            ext = np.zeros_like(out)
            for term in external_terms:
                ext[:, p0:p1, :] += term[:, p0:p1, :]
            return out + ext + ext.transpose(0, 2, 1)

        return hcore_deriv

    grad.hcore_generator = hcore_generator
    base_nuclear = grad.nuclear

    def nuclear(atmlst=None):
        out = base_nuclear(atmlst=atmlst)
        ext = nuclear_point_charge_gradient(mf.mol, coords, charges)
        if atmlst is not None:
            ext = ext[atmlst]
        return out + ext

    grad.nuclear = nuclear
    return np.asarray(np.real(grad.run()), dtype=float)


def nuclear_point_charge_energy(mol, coords, charges):
    """Classical interaction energy between QM nuclei and point charges."""
    coords = _as_coords(coords)
    charges = _as_charges(charges, len(coords))
    qm_coords = np.asarray(mol.atom_coords(), dtype=float)
    qm_charges = np.asarray(mol.atom_charges(), dtype=float)

    energy = 0.0
    for nuclear_coord, nuclear_charge in zip(qm_coords, qm_charges):
        deltas = nuclear_coord - coords
        distances = np.linalg.norm(deltas, axis=1)
        if np.any(distances == 0.0):
            raise ValueError("A point charge lies on a QM nucleus.")
        energy += float(np.sum(nuclear_charge * charges / distances))
    return energy


def nuclear_point_charge_gradient(mol, coords, charges):
    """QM nuclear gradient of the nuclear-point-charge interaction."""
    coords = _as_coords(coords)
    charges = _as_charges(charges, len(coords))
    qm_coords = np.asarray(mol.atom_coords(), dtype=float)
    qm_charges = np.asarray(mol.atom_charges(), dtype=float)

    gradient = np.zeros_like(qm_coords)
    for atom, (nuclear_coord, nuclear_charge) in enumerate(zip(qm_coords, qm_charges)):
        deltas = nuclear_coord - coords
        distances = np.linalg.norm(deltas, axis=1)
        if np.any(distances == 0.0):
            raise ValueError("A point charge lies on a QM nucleus.")
        gradient[atom] -= np.einsum(
            "a,ax,a->x",
            float(nuclear_charge) * charges,
            deltas,
            distances ** -3,
            optimize=True,
        )
    return gradient


def point_charge_forces(mol, dm, coords, charges):
    """Analytic forces on external point charges.

    The returned array has shape ``(ncharge, 3)`` and includes the force from
    both the embedded electron density and the QM nuclei.
    """
    coords = _as_coords(coords)
    charges = _as_charges(charges, len(coords))
    dm = np.asarray(dm, dtype=float)

    basis, transform = _basis_and_transform(mol)
    dm_cart = _transform_density_to_cart(dm, transform)
    qm_coords = np.asarray(mol.atom_coords(), dtype=float)
    qm_charges = np.asarray(mol.atom_charges(), dtype=float)
    forces = np.zeros_like(coords)
    for charge_index, (coord, charge) in enumerate(zip(coords, charges)):
        grad_elec = np.zeros(3, dtype=float)
        for i, bra in enumerate(basis):
            for j, ket in enumerate(basis):
                weight = dm_cart[j, i]
                if weight == 0.0:
                    continue
                grad_elec += (
                    -float(charge)
                    * weight
                    * _point_charge_integral_center_gradient(bra, ket, coord)
                )

        deltas = qm_coords - coord
        distances = np.linalg.norm(deltas, axis=1)
        if np.any(distances == 0.0):
            raise ValueError("A point charge lies on a QM nucleus.")
        grad_nuc = np.einsum(
            "a,ax,a->x",
            qm_charges * float(charge),
            deltas,
            distances ** -3,
            optimize=True,
        )
        forces[charge_index] = -(grad_elec + grad_nuc)
    return forces


def _basis_and_transform(mol):
    from pyqed.qchem.basis import ContractedGaussian

    basis = getattr(mol, "_bas_cart", None)
    transform = getattr(mol, "_ao_cart2sph", None)
    if basis is None:
        basis = getattr(mol, "_bas", None)
        transform = None
    if basis is None or not all(isinstance(fn, ContractedGaussian) for fn in basis):
        raise ValueError("Build the molecule with driver='builtin' before point-charge embedding.")
    return tuple(basis), transform


def _has_gbasis_basis(mol):
    basis = getattr(mol, "_bas", None)
    if not basis:
        return False
    return basis[0].__class__.__name__ == "GeneralizedContractionShell"


def _is_post_scf_method(method):
    return (
        hasattr(method, "mf")
        and hasattr(method, "ncas")
        and hasattr(method, "run")
        and getattr(method, "mf", None) is not method
    )


def _supports_embedded_rhf_gradient(mf):
    return mf.__class__.__name__ == "RHF" and hasattr(mf, "mo_occ")


def _supports_embedded_rks_gradient(mf):
    return mf.__class__.__name__ == "RKS" and hasattr(mf, "nuc_grad_method")


def _supports_native_point_charge_forces(mol):
    basis = getattr(mol, "_bas_cart", None)
    if basis is None:
        basis = getattr(mol, "_bas", None)
    if not basis:
        return False
    from pyqed.qchem.basis import ContractedGaussian

    return all(isinstance(fn, ContractedGaussian) for fn in basis)


def _transform_one(matrix, transform):
    if transform is None:
        return matrix
    return np.einsum("pi,...pq,qj->...ij", transform, matrix, transform, optimize=True)


def _transform_density_to_cart(dm, transform):
    dm = np.asarray(dm, dtype=float)
    if transform is None:
        return dm
    return np.einsum("pi,ij,qj->pq", transform, dm, transform, optimize=True)


def _point_charge_integral_center_gradient(bra, ket, center):
    gradient = np.zeros(3, dtype=float)
    for axis in range(3):
        order = _axis_order(axis)
        gradient[axis] = -(
            _contracted_one_deriv(bra, ket, "nuclear", order_a=order, center=center)
            + _contracted_one_deriv(bra, ket, "nuclear", order_b=order, center=center)
        )
    return gradient


def _add_symmetric_derivative(deriv, atom, axis, i, j, value):
    deriv[atom, axis, i, j] += value
    if i != j:
        deriv[atom, axis, j, i] += value


def _as_coords(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(1, 3)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3).")
    return coords


def _as_charges(charges, expected):
    charges = np.asarray(charges, dtype=float).reshape(-1)
    if charges.shape != (expected,):
        raise ValueError(f"charges must have shape ({expected},).")
    return charges
