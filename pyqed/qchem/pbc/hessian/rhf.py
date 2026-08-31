"""CPHF-relaxed fixed-cell Hessian for Gamma-point periodic KRHF."""

from __future__ import annotations

import time

import numpy as np

from pyqed.qchem.pbc.cell import Cell
from pyqed.qchem.pbc.ewald import ewald_nuclear_gradient, ewald_nuclear_hessian
from pyqed.units import amu_to_au, au2wavenumber


class KRHFHessian:
    """Build the relaxed Gamma-point KRHF Cartesian Hessian.

    Nuclear derivatives, electronic integral derivatives, and the CPHF density
    response are analytic for all-electron reciprocal J/K references.  Other
    supported references retain a central difference of analytic first
    derivatives without running displaced SCF calculations.
    """

    def __init__(self, mean_field):
        self.base = mean_field
        self.cell = mean_field.cell
        self.coords = np.asarray(self.cell._atom_coords, dtype=float)
        self.hess = None
        self.raw_hess = None
        self.hess4 = None
        self.response = None
        self.first_order_density = None
        self.first_order_energy_weighted_density = None
        self.explicit_second = None
        self.response_hessian = None
        self.nuclear_hessian = None
        self.dynamical_matrix = None
        self.eigenvectors = None
        self.freq_au = None
        self.freq_cm1 = None
        self.second_derivative_backend = None
        self.seconds = None
        self.success = False
        self.message = "not run"

    @property
    def natom(self):
        return len(self.coords)

    @property
    def npert(self):
        return 3 * self.natom

    def _validate(self):
        mf = self.base
        if not getattr(mf, "converged", False):
            raise RuntimeError("Run and converge KRHF before requesting its Hessian.")
        if int(self.cell.dimension) != 3:
            raise NotImplementedError("Periodic KRHF Hessians require dimension=3.")
        if int(mf.nkpts) != 1 or np.linalg.norm(np.asarray(mf.kpts[0])) > 1.0e-12:
            raise NotImplementedError(
                "Periodic KRHF Hessians currently support the Gamma point only."
            )
        if str(mf.jk_builder) not in ("reciprocal", "ewald"):
            raise NotImplementedError(
                "Periodic KRHF Hessians currently require a reciprocal or Ewald J/K builder."
            )

    def _mean_field_at(self, coords):
        source = self.base
        cell = Cell(
            atom=[
                (str(symbol), tuple(position))
                for symbol, position in zip(self.cell._atom_symbols, coords)
            ],
            a=np.asarray(self.cell.lattice_vectors, dtype=float),
            basis=self.cell.basis,
            unit="bohr",
            charge=self.cell.charge,
            spin=self.cell.spin,
            dimension=self.cell.dimension,
            vacuum=self.cell.vacuum,
            low_dim_ft_type=self.cell.low_dim_ft_type,
            integral_options=self.cell.integral_options,
            pseudo=self.cell.pseudo,
        ).build()
        mean_field = cell.KRHF(
            kpts=np.asarray(source.kpts, dtype=float),
            eta=source.eta,
            real_cut=source.real_cut,
            recip_cut=source.recip_cut,
            recip_precision=source.recip_precision,
            recip_max_cut=source.recip_max_cut,
            mesh=source.mesh,
            damping=source.damping,
            nuclear_background=source.nuclear_background,
            eri_screen_tol=source.eri_screen_tol,
            jk_builder=source.jk_builder,
            pair_cut=source.pair_cut,
            pair_ft_screen_tol=source.pair_ft_screen_tol,
            occupation_mode=source.occupation_mode,
            occupation_tol=source.occupation_tol,
            pseudo_cut=source.pseudo_cut,
            pseudo_local_screen_tol=source.pseudo_local_screen_tol,
            one_body_screen_tol=source.one_body_screen_tol,
            one_body_nuclear_cut=source.one_body_nuclear_cut,
            one_body_workers=source.one_body_workers,
            diis=source.diis,
            diis_space=source.diis_space,
            diis_start_cycle=source.diis_start_cycle,
        )
        mean_field._build_integrals()
        return mean_field

    def _integral_derivatives_at(self, coords, dm):
        mean_field = self._mean_field_at(coords)
        gradient = mean_field.nuc_grad_method()
        return gradient.explicit_integral_derivatives(
            dm,
            require_scf=False,
        )

    def _energy_weighted_density_response(self, response):
        mf = self.base
        mo_coeff = np.asarray(mf.mo_coeff, dtype=np.complex128)
        mo_occ = np.asarray(mf.mo_occ, dtype=float)
        mo_energy = np.asarray(mf.mo_energy, dtype=float)
        occupied = mo_occ > 1.0e-10
        weights = mo_occ[occupied]
        if np.any(np.abs(weights - 2.0) > 1.0e-10):
            raise NotImplementedError(
                "Periodic KRHF Hessians require integer closed-shell occupations."
            )
        cocc = mo_coeff[:, occupied]
        c1 = np.asarray(response.mo_coeff1, dtype=np.complex128)
        e1 = np.asarray(response.mo_e1, dtype=np.complex128)
        weighted_energy = weights * mo_energy[occupied]
        orbital = np.einsum(
            "xpi,i,qi->xpq",
            c1,
            weighted_energy,
            cocc.conj(),
            optimize=True,
        )
        orbital += orbital.conj().transpose(0, 2, 1)
        energy = np.einsum(
            "pi,xij,qj->xpq",
            cocc,
            2.0 * e1,
            cocc.conj(),
            optimize=True,
        )
        return 0.5 * (
            orbital + energy + (orbital + energy).conj().transpose(0, 2, 1)
        )

    def _analytic_explicit_second(self, gradient, dm0, w0):
        nao = int(self.cell.nao)
        npert = self.npert
        s2, h2 = gradient.one_electron_second_derivatives()
        s2 = s2.reshape(npert, npert, nao, nao)
        h2 = h2.reshape(npert, npert, nao, nao)
        veff2_scalar = gradient.reciprocal_veff_second_scalar(dm0).reshape(
            npert,
            npert,
        )
        explicit_second = (
            np.einsum("pq,xyqp->xy", dm0, h2, optimize=True)
            + 0.5 * veff2_scalar
            - np.einsum("pq,xyqp->xy", w0, s2, optimize=True)
        ).real
        nuclear_hessian = ewald_nuclear_hessian(
            self.cell.ionic_charges,
            self.coords,
            np.asarray(self.cell.lattice_vectors, dtype=float),
            eta=self.base.eta,
            real_cut=self.base.real_cut,
            recip_cut=self.base.recip_cut,
        ).reshape(npert, npert)
        return explicit_second, nuclear_hessian

    def _finite_difference_explicit_second(self, dm0, w0, step):
        nao = int(self.cell.nao)
        npert = self.npert
        explicit_second = np.zeros((npert, npert), dtype=float)
        nuclear_hessian = np.zeros_like(explicit_second)
        charges = self.cell.ionic_charges
        lattice = np.asarray(self.cell.lattice_vectors, dtype=float)
        for perturbation in range(npert):
            atom, axis = divmod(perturbation, 3)
            plus_coords = np.array(self.coords, copy=True)
            minus_coords = np.array(self.coords, copy=True)
            plus_coords[atom, axis] += step
            minus_coords[atom, axis] -= step
            s1_plus, h1_plus, veff1_plus = self._integral_derivatives_at(
                plus_coords,
                dm0,
            )
            s1_minus, h1_minus, veff1_minus = self._integral_derivatives_at(
                minus_coords,
                dm0,
            )
            ds1 = (s1_plus - s1_minus).reshape(npert, nao, nao) / (2.0 * step)
            dh1 = (h1_plus - h1_minus).reshape(npert, nao, nao) / (2.0 * step)
            dveff1 = (
                (veff1_plus - veff1_minus).reshape(npert, nao, nao)
                / (2.0 * step)
            )
            explicit_second[:, perturbation] = (
                np.einsum("pq,xqp->x", dm0, dh1, optimize=True)
                + 0.5 * np.einsum("pq,xqp->x", dm0, dveff1, optimize=True)
                - np.einsum("pq,xqp->x", w0, ds1, optimize=True)
            ).real

            nuclear_plus = ewald_nuclear_gradient(
                charges,
                plus_coords,
                lattice,
                eta=self.base.eta,
                real_cut=self.base.real_cut,
                recip_cut=self.base.recip_cut,
            )
            nuclear_minus = ewald_nuclear_gradient(
                charges,
                minus_coords,
                lattice,
                eta=self.base.eta,
                real_cut=self.base.real_cut,
                recip_cut=self.base.recip_cut,
            )
            nuclear_hessian[:, perturbation] = (
                (nuclear_plus - nuclear_minus).reshape(-1) / (2.0 * step)
            )
        return explicit_second, nuclear_hessian

    @staticmethod
    def _enforce_acoustic_sum_rule(hessian, natom):
        hess4 = np.asarray(hessian, dtype=float).reshape(natom, 3, natom, 3)
        hess4 = np.array(hess4, copy=True)
        for _iteration in range(3):
            residual = np.sum(hess4, axis=2)
            for atom in range(natom):
                hess4[atom, :, atom, :] -= residual[atom]
            hess4 = 0.5 * (hess4 + hess4.transpose(2, 3, 0, 1))
        residual = np.sum(hess4, axis=2)
        for atom in range(natom):
            hess4[atom, :, atom, :] -= residual[atom]
        return hess4.reshape(3 * natom, 3 * natom)

    def kernel(
        self,
        *,
        step=2.0e-4,
        cphf_tol=1.0e-10,
        second_derivative_backend="auto",
        symmetrize=True,
        enforce_acoustic_sum_rule=True,
    ):
        self._validate()
        backend = str(second_derivative_backend).strip().lower()
        if backend not in ("auto", "analytic", "finite_difference"):
            raise ValueError(
                "second_derivative_backend must be 'auto', 'analytic', or "
                "'finite_difference'."
            )
        analytic_supported = (
            str(self.base.jk_builder) == "reciprocal" and not self.cell.has_pseudo
        )
        if backend == "auto":
            backend = "analytic" if analytic_supported else "finite_difference"
        if backend == "analytic" and not analytic_supported:
            raise NotImplementedError(
                "Analytic periodic KRHF second derivatives currently require an "
                "all-electron reciprocal J/K reference."
            )
        if backend == "finite_difference":
            step = float(step)
            if not np.isfinite(step) or step <= 0.0:
                raise ValueError("step must be a positive finite distance in Bohr.")
        self.second_derivative_backend = backend
        started = time.perf_counter()
        mf = self.base
        nao = int(self.cell.nao)
        npert = self.npert
        dm0 = np.asarray(mf.make_rdm1(), dtype=np.complex128)

        gradient = mf.nuc_grad_method()
        s1, h1, veff1 = gradient.explicit_integral_derivatives(dm0)
        s1 = s1.reshape(npert, nao, nao)
        h1 = h1.reshape(npert, nao, nao)
        veff1 = veff1.reshape(npert, nao, nao)
        f1 = h1 + veff1
        response = mf.response().kernel(f1, s1=s1, tol=cphf_tol)
        dm1 = np.asarray(response.dm1, dtype=np.complex128)
        w1 = self._energy_weighted_density_response(response)

        mo_coeff = np.asarray(mf.mo_coeff, dtype=np.complex128)
        mo_occ = np.asarray(mf.mo_occ, dtype=float)
        occupied = mo_occ > 1.0e-10
        cocc = mo_coeff[:, occupied]
        w0 = np.einsum(
            "pi,qi,i->pq",
            cocc,
            cocc.conj(),
            mo_occ[occupied] * np.asarray(mf.mo_energy)[occupied],
            optimize=True,
        )

        if backend == "analytic":
            explicit_second, nuclear_hessian = self._analytic_explicit_second(
                gradient,
                dm0,
                w0,
            )
        else:
            explicit_second, nuclear_hessian = (
                self._finite_difference_explicit_second(dm0, w0, step)
            )

        veff_dm1 = gradient.effective_potential_derivatives_many(
            dm1,
            s1=s1,
        ).reshape(npert, npert, nao, nao)
        response_hessian = np.zeros_like(explicit_second)
        for x in range(npert):
            for y in range(npert):
                response_hessian[x, y] = (
                    np.einsum("pq,qp->", dm1[y], h1[x], optimize=True)
                    + 0.5
                    * np.einsum("pq,qp->", dm1[y], veff1[x], optimize=True)
                    + 0.5
                    * np.einsum("pq,qp->", dm0, veff_dm1[y, x], optimize=True)
                    - np.einsum("pq,qp->", w1[y], s1[x], optimize=True)
                ).real

        hessian = explicit_second + response_hessian + nuclear_hessian
        self.raw_hess = np.array(hessian, copy=True)
        if symmetrize:
            hessian = 0.5 * (hessian + hessian.T)
        if enforce_acoustic_sum_rule:
            hessian = self._enforce_acoustic_sum_rule(hessian, self.natom)

        self.hess = np.asarray(hessian, dtype=float)
        self.hess4 = self.hess.reshape(self.natom, 3, self.natom, 3)
        self.response = response
        self.first_order_density = dm1.reshape(self.natom, 3, nao, nao)
        self.first_order_energy_weighted_density = w1.reshape(
            self.natom,
            3,
            nao,
            nao,
        )
        self.explicit_second = explicit_second
        self.response_hessian = response_hessian
        self.nuclear_hessian = nuclear_hessian
        self.seconds = float(time.perf_counter() - started)
        self.success = True
        self.message = (
            "CPHF-relaxed Gamma-point Hessian built with "
            f"{self.second_derivative_backend} second derivatives"
        )
        return self.hess

    run = kernel

    @property
    def acoustic_sum_rule_residual(self):
        if self.hess4 is None:
            raise RuntimeError("Run the Hessian calculation first.")
        return float(np.max(np.abs(np.sum(self.hess4, axis=2))))

    def frequencies(self, *, units="cm-1", return_eigenvectors=False):
        if self.hess is None:
            raise RuntimeError("Run the Hessian calculation first.")
        masses = np.asarray(self.cell.unit_molecule.atom_mass_list(), dtype=float)
        masses = np.repeat(masses * amu_to_au, 3)
        dynamical = self.hess / np.sqrt(np.outer(masses, masses))
        dynamical = 0.5 * (dynamical + dynamical.T)
        eigenvalues, eigenvectors = np.linalg.eigh(dynamical)
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
        self.dynamical_matrix = dynamical
        self.eigenvectors = eigenvectors
        self.freq_au = np.asarray(frequencies, dtype=float)
        self.freq_cm1 = self.freq_au * au2wavenumber
        unit_key = str(units).strip().lower().replace("^", "")
        if unit_key in ("cm-1", "cm1", "wavenumber", "wavenumbers"):
            values = self.freq_cm1
        elif unit_key in ("au", "a.u.", "hartree"):
            values = self.freq_au
        else:
            raise ValueError("units must be 'au' or 'cm-1'.")
        if return_eigenvectors:
            return np.array(values, copy=True), eigenvectors
        return np.array(values, copy=True)


__all__ = ["KRHFHessian"]
