# -*- coding: utf-8 -*-
"""Small finite-basis imaginary-axis self-consistent GW prototype.

This module is intentionally conservative: it is a dense reference prototype
for tiny molecular bases, not a production scGW engine.  It keeps the full
matrix Green's function on a symmetric imaginary-frequency grid and iterates
the skeleton GW equations

    P = -G G,     W = v + v P W,     Sigma_c = -G (W - v)

followed by a Dyson update.  ``run(update_screening=False)`` gives the
corresponding scGW0 loop, where the screened interaction is fixed from the
initial Green's function.  The implementation is useful for testing data
structures, convergence controls, and future total-energy work.
"""

from __future__ import annotations

import numpy as np

<<<<<<< HEAD
from pyqed.gw.gw import (
    GW,
    _get_hcore_ao,
    _reference_total_energy,
    _spin_matrix_from_spatial,
    _spin_pair_factors,
)
=======
from pyqed.gw.gw import GW, _reference_total_energy, _spin_pair_factors
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4


def _symmetric_frequency_grid(nfreq=17, wmax=20.0):
    if nfreq < 3:
        raise ValueError("nfreq must be at least 3.")
    if nfreq % 2 == 0:
        raise ValueError("nfreq must be odd so the grid contains zero frequency.")
    omega = np.linspace(-float(wmax), float(wmax), int(nfreq))
    weights = np.empty_like(omega)
    step = omega[1] - omega[0]
    weights[:] = step / (2.0 * np.pi)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return omega, weights


<<<<<<< HEAD
def _tangent_frequency_grid(nfreq=17, scale=10.0):
    if nfreq < 3:
        raise ValueError("nfreq must be at least 3.")
    if nfreq % 2 == 0:
        raise ValueError("nfreq must be odd so the grid contains zero frequency.")
    x, wx = np.polynomial.legendre.leggauss(int(nfreq))
    angle = 0.5 * np.pi * x
    omega = float(scale) * np.tan(angle)
    weights = wx * float(scale) * 0.25 / (np.cos(angle) ** 2)
    return omega, weights


def _imaginary_frequency_grid(nfreq=17, wmax=20.0, kind="linear"):
    kind = str(kind).lower()
    if kind in {"linear", "uniform"}:
        return _symmetric_frequency_grid(nfreq=nfreq, wmax=wmax)
    if kind in {"tangent", "tan", "mapped"}:
        return _tangent_frequency_grid(nfreq=nfreq, scale=wmax)
    raise ValueError("grid must be 'linear' or 'tangent'.")


=======
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
def _fermionic_matsubara_grid(nfreq=129, beta=200.0):
    if nfreq < 3:
        raise ValueError("density_nfreq must be at least 3.")
    if nfreq % 2 == 0:
        raise ValueError("density_nfreq must be odd for a symmetric Matsubara grid.")
    half = nfreq // 2
    n = np.arange(-half, half + 1)
    return (2 * n + 1) * np.pi / float(beta)


def _interp_matrix_grid(grid, values, x):
    values = np.asarray(values)
    if x <= grid[0] or x >= grid[-1]:
        # Keep the finite-grid prototype stable by dropping convolution tails.
        return np.zeros(values.shape[1:], dtype=values.dtype)
    flat = values.reshape(values.shape[0], -1)
    out = np.empty(flat.shape[1], dtype=values.dtype)
    for col in range(flat.shape[1]):
        out[col] = (
            np.interp(x, grid, flat[:, col].real)
            + 1j * np.interp(x, grid, flat[:, col].imag)
        )
    return out.reshape(values.shape[1:])


<<<<<<< HEAD
def _interp_green_grid(grid, values, x, mu, static, eta=0.0):
    """Interpolate ``G(iw)`` and use its large-frequency tail off grid.

    The skeleton convolutions need shifted Green's functions ``G(iw +/- inu)``.
    Returning zero outside the finite grid creates a hard cutoff and is the
    dominant source of grid sensitivity.  For off-grid points we use the first
    two terms of the large-frequency expansion

        G(z) = z^{-1} I + z^{-2} H + O(z^{-3}),   z = mu + i w,

    where ``H`` is the current static Dyson Hamiltonian.
    """
    values = np.asarray(values)
    if grid[0] < x < grid[-1]:
        return _interp_matrix_grid(grid, values, x)
    eye = np.eye(values.shape[1], dtype=np.complex128)
    sign = 0.0 if x == 0.0 else np.sign(x)
    z = complex(mu, x + eta * sign)
    return eye / z + np.asarray(static, dtype=np.complex128) / (z * z)


=======
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
class SCGW:
    """Dense imaginary-axis self-consistent GW prototype.

    Parameters
    ----------
    mf
        Restricted mean-field object.
    nfreq
        Odd number of imaginary-frequency grid points.
    wmax
        Maximum absolute imaginary frequency in Hartree.
    eta
        Small real stabilizer added to Dyson denominators.
    screening
        Screening label passed through the existing :class:`GW` reference
        builder.
    """

    def __init__(
        self,
        mf,
        nfreq=17,
        wmax=20.0,
        eta=1e-8,
        screening="TDH",
        beta=200.0,
        adjust_mu=True,
        target_nelec=None,
        density_nfreq=129,
<<<<<<< HEAD
        grid="tangent",
=======
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
    ):
        self._scf = mf
        self.mol = mf.mol
        self.nfreq = int(nfreq)
        self.wmax = float(wmax)
        self.eta = float(eta)
        self.screening = screening
        self.beta = float(beta)
        self.adjust_mu = bool(adjust_mu)
        self.density_nfreq = int(density_nfreq)
<<<<<<< HEAD
        self.grid = str(grid).lower()
=======
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4

        # Reuse the existing GW builder for spin-orbital MO integrals,
        # HF eigenvalues, and the static mean-field potential.
        ref = GW(mf, screening=screening, eta=max(eta, 1e-8))
        self._ref = ref

        self.nso = ref.nso
        self.nocc = ref.nocc
        self.target_nelec = float(self.nocc if target_nelec is None else target_nelec)
        self.e_mf = np.asarray(ref.e_mf, dtype=float)
        self.v_mf = np.asarray(ref.v_mf, dtype=float)
<<<<<<< HEAD
        hcore_spatial = ref.mo_coeff.T @ _get_hcore_ao(ref) @ ref.mo_coeff
        self.hcore = _spin_matrix_from_spatial(hcore_spatial)
=======
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
        self.backend = "factorized" if ref.eri is None else "dense"
        self.pair_factors = None
        if self.backend == "factorized":
            self.pair_factors = np.asarray(_spin_pair_factors(ref), dtype=float)
            self.eri = None
        else:
            self.eri = np.asarray(ref.eri, dtype=float)
<<<<<<< HEAD
        self.omega, self.weights = _imaginary_frequency_grid(nfreq, wmax, kind=self.grid)
=======
        self.omega, self.weights = _symmetric_frequency_grid(nfreq, wmax)
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
        self.density_omega = _fermionic_matsubara_grid(density_nfreq, beta)
        self.zero_index = self.nfreq // 2

        self.mu = self._initial_mu()
        self.h0 = np.diag(self.e_mf) - self.v_mf
        self.Sigma_x = self._exchange_self_energy()
        self.sigma_x = self.Sigma_x
        self.v_pair = (
            self.eri.reshape(self.nso * self.nso, self.nso * self.nso)
            if self.eri is not None
            else None
        )

        self.G = None
        self.P = None
        self.W = None
        self.Sigma_c = np.zeros((self.nfreq, self.nso, self.nso), dtype=np.complex128)
        self.history = []
        self.mu_history = []
        self.converged = False
        self.density_matrix = None
        self.nelec = None
        self.e = None
        self.e_qp = None
        self.e_tot = None
<<<<<<< HEAD
        self.e_tot_gm = None
        self.e_tot_lw = None
        self.energy_components = None
        self.e_scf = _reference_total_energy(ref)
        self.e_nuc = self._nuclear_repulsion_energy()
=======
        self.e_scf = _reference_total_energy(ref)
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
        self.info = None
        self.W0 = None

    def _initial_mu(self):
        homo = self.e_mf[self.nocc - 1]
        lumo = self.e_mf[self.nocc]
        return 0.5 * (homo + lumo)

<<<<<<< HEAD
    def _nuclear_repulsion_energy(self):
        if hasattr(self._scf, "energy_nuc"):
            return float(self._scf.energy_nuc())
        if hasattr(self.mol, "energy_nuc"):
            return float(self.mol.energy_nuc())
        if hasattr(self.mol, "nuclear_repulsion"):
            return float(self.mol.nuclear_repulsion())
        return 0.0

=======
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
    def _exchange_self_energy(self, density_matrix=None):
        if density_matrix is None:
            density_matrix = np.zeros((self.nso, self.nso))
            density_matrix[:self.nocc, :self.nocc] = np.eye(self.nocc)
        density_matrix = np.asarray(density_matrix, dtype=float)
        if self.backend == "factorized":
            sigma_x = -np.einsum(
                "Ppr,rs,Psq->pq",
                self.pair_factors,
                density_matrix,
                self.pair_factors,
                optimize=True,
            )
        else:
            sigma_x = -np.einsum("prsq,rs->pq", self.eri, density_matrix, optimize=True)
        return 0.5 * (sigma_x + sigma_x.T)

<<<<<<< HEAD
    def _hartree_energy(self, density_matrix):
        density_matrix = np.asarray(density_matrix)
        if self.backend == "factorized":
            rho_aux = np.einsum("Ppq,qp->P", self.pair_factors, density_matrix, optimize=True)
            return 0.5 * float(np.vdot(rho_aux, rho_aux).real)
        return 0.5 * float(
            np.einsum(
                "pqrs,qp,sr->",
                self.eri,
                density_matrix,
                density_matrix,
                optimize=True,
            ).real
        )

    def _matsubara_sigma_g_trace(self, sigma_c=None, mu=None, sigma_x=None):
        if sigma_c is None:
            sigma_c = self.Sigma_c
        if mu is None:
            mu = self.mu
        if sigma_x is None:
            sigma_x = self.Sigma_x
        green = self._dyson_green_on_grid(sigma_c, self.density_omega, mu=mu, sigma_x=sigma_x)
        trace = 0.0j
        for iw, w in enumerate(self.density_omega):
            sigma_w = _interp_matrix_grid(self.omega, sigma_c, w)
            trace += np.trace(sigma_w @ green[iw])
        return float((trace / self.beta).real)

    def galitskii_migdal_total_energy(self, sigma_c=None, mu=None, sigma_x=None, density_matrix=None):
        """Return the Galitskii-Migdal molecular total energy and components.

        The energy is evaluated as

            E = Tr[h gamma] + E_H[gamma] + 1/2 Tr[Sigma_x gamma]
                + 1/(2 beta) sum_n Tr[Sigma_c(iw_n) G(iw_n)] + E_nuc.

        ``Sigma_c`` and ``G`` are sampled on the fermionic Matsubara grid used
        for the density matrix.
        """
        if sigma_c is None:
            sigma_c = self.Sigma_c
        if mu is None:
            mu = self.mu
        if sigma_x is None:
            sigma_x = self.Sigma_x
        if density_matrix is None:
            density_matrix = self.make_density_matrix(
                sigma_c=sigma_c,
                mu=mu,
                method="green",
                sigma_x=sigma_x,
            )

        e_one = float(np.einsum("pq,qp->", self.hcore, density_matrix, optimize=True).real)
        e_h = self._hartree_energy(density_matrix)
        e_x = 0.5 * float(np.einsum("pq,qp->", sigma_x, density_matrix, optimize=True).real)
        e_c = 0.5 * self._matsubara_sigma_g_trace(
            sigma_c=sigma_c,
            mu=mu,
            sigma_x=sigma_x,
        )
        e_elec = e_one + e_h + e_x + e_c
        e_tot = e_elec + self.e_nuc
        return {
            "method": "galitskii_migdal",
            "e_tot": float(e_tot),
            "e_elec": float(e_elec),
            "e_nuc": float(self.e_nuc),
            "e_one": float(e_one),
            "e_hartree": float(e_h),
            "e_exchange": float(e_x),
            "e_correlation": float(e_c),
            "sigma_g_correlation": float(2.0 * e_c),
            "beta": self.beta,
            "density_nfreq": self.density_nfreq,
        }

    def luttinger_ward_total_energy(self, sigma_c=None, mu=None, sigma_x=None, density_matrix=None):
        """Return the GW Luttinger-Ward stationary energy diagnostic.

        For a self-consistent, Phi-derivable GW solution the Luttinger-Ward
        interaction functional has ``Phi_c^GW = 1/2 Tr[Sigma_c G]`` for the
        ring skeleton used here, so the stationary internal energy is identical
        to the Galitskii-Migdal value.  We return the same total with the
        correlation functional labelled explicitly.
        """
        components = self.galitskii_migdal_total_energy(
            sigma_c=sigma_c,
            mu=mu,
            sigma_x=sigma_x,
            density_matrix=density_matrix,
        )
        components = dict(components)
        components["method"] = "luttinger_ward_gw_stationary"
        components["phi_correlation"] = components["e_correlation"]
        components["stationarity_residual"] = (
            max(
                self.history[-1]["sigma_delta"],
                self.history[-1]["sigma_x_delta"],
                self.history[-1]["green_delta"],
            )
            if self.history
            else None
        )
        return components

=======
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
    def _dyson_green_on_grid(self, sigma_c, omega, mu=None, sigma_x=None):
        if mu is None:
            mu = self.mu
        if sigma_x is None:
            sigma_x = self.Sigma_x
        eye = np.eye(self.nso)
        omega = np.asarray(omega, dtype=float)
        green = np.empty((len(omega), self.nso, self.nso), dtype=np.complex128)
        static = self.h0 + sigma_x
        for iw, w in enumerate(omega):
            sigma_w = _interp_matrix_grid(self.omega, sigma_c, w)
            mat = (mu + 1j * w + 1j * self.eta * np.sign(w)) * eye
            mat = mat - static - sigma_w
            green[iw] = np.linalg.inv(mat)
        return green

    def _dyson_green(self, sigma_c, mu=None, sigma_x=None):
        return self._dyson_green_on_grid(sigma_c, self.omega, mu=mu, sigma_x=sigma_x)

    def _static_hamiltonian(self, sigma_c=None, sigma_x=None):
        if sigma_c is None:
            sigma_c = self.Sigma_c
        if sigma_x is None:
            sigma_x = self.Sigma_x
        sigma0 = np.asarray(sigma_c)[self.zero_index].real
        heff = self.h0 + sigma_x + 0.5 * (sigma0 + sigma0.T)
        return 0.5 * (heff + heff.T)

    def _fermi_occupations(self, energies, mu):
        x = np.clip(self.beta * (np.asarray(energies, dtype=float) - mu), -80.0, 80.0)
        return 1.0 / (np.exp(x) + 1.0)

    def _particle_number_for_mu(self, energies, mu):
        return float(np.sum(self._fermi_occupations(energies, mu)))

    def _solve_mu(self, sigma_c, guess=None, sigma_x=None):
        if not self.adjust_mu:
            return self.mu if guess is None else float(guess)

        energies = np.linalg.eigvalsh(self._static_hamiltonian(sigma_c, sigma_x=sigma_x))
        margin = max(10.0 / max(self.beta, 1.0), 10.0 * self.eta, 1.0)
        lo = float(np.min(energies) - margin)
        hi = float(np.max(energies) + margin)

        nlo = self.particle_number(sigma_c=sigma_c, mu=lo, method="green", sigma_x=sigma_x)
        nhi = self.particle_number(sigma_c=sigma_c, mu=hi, method="green", sigma_x=sigma_x)
<<<<<<< HEAD
        for _ in range(20):
            if nlo <= self.target_nelec <= nhi:
                break
            margin *= 2.0
            lo = float(np.min(energies) - margin)
            hi = float(np.max(energies) + margin)
            nlo = self.particle_number(sigma_c=sigma_c, mu=lo, method="green", sigma_x=sigma_x)
            nhi = self.particle_number(sigma_c=sigma_c, mu=hi, method="green", sigma_x=sigma_x)
=======
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
        if not (nlo <= self.target_nelec <= nhi):
            # Fall back to the midpoint of the frontier static eigenvalues.
            idx = int(np.clip(round(self.target_nelec), 1, len(energies) - 1))
            return float(0.5 * (energies[idx - 1] + energies[idx]))

        for _ in range(100):
            mid = 0.5 * (lo + hi)
            nmid = self.particle_number(sigma_c=sigma_c, mu=mid, method="green", sigma_x=sigma_x)
            if abs(nmid - self.target_nelec) < 1e-10:
                return float(mid)
            if nmid < self.target_nelec:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    def _static_density_matrix(self, sigma_c=None, mu=None, sigma_x=None):
        if sigma_c is None:
            sigma_c = self.Sigma_c
        if mu is None:
            mu = self.mu
        energies, coeff = np.linalg.eigh(self._static_hamiltonian(sigma_c, sigma_x=sigma_x))
        occ = self._fermi_occupations(energies, mu)
        dm = (coeff * occ[None, :]) @ coeff.T
        return 0.5 * (dm + dm.T)

    def _green_density_matrix(self, sigma_c=None, mu=None, sigma_x=None):
        if sigma_c is None:
            sigma_c = self.Sigma_c
        if mu is None:
            mu = self.mu
        green = self._dyson_green_on_grid(sigma_c, self.density_omega, mu=mu, sigma_x=sigma_x)
        eye = np.eye(self.nso, dtype=np.complex128)
        corr = np.zeros((self.nso, self.nso), dtype=np.complex128)
        for iw, w in enumerate(self.density_omega):
            corr += green[iw] - eye / (1j * w)
        dm = 0.5 * eye + corr / self.beta
        dm = dm.real
        return 0.5 * (dm + dm.T)

    def make_density_matrix(self, sigma_c=None, mu=None, method="green", sigma_x=None):
        """Return the density matrix.

        ``method="green"`` uses a tail-corrected fermionic Matsubara sum of
        the interacting Green's function.  ``method="static"`` uses Fermi
        occupations of ``h0 + Sigma_x + Re Sigma_c(i0)`` and is kept as a
        diagnostic fallback.
        """
        method = str(method).lower()
        if method in ("green", "g", "matsubara"):
            return self._green_density_matrix(sigma_c=sigma_c, mu=mu, sigma_x=sigma_x)
        if method in ("static", "fermi"):
            return self._static_density_matrix(sigma_c=sigma_c, mu=mu, sigma_x=sigma_x)
        raise ValueError("method must be 'green' or 'static'.")

    def particle_number(self, sigma_c=None, mu=None, method="green", sigma_x=None):
        """Return the electron count from the selected density matrix."""
        dm = self.make_density_matrix(sigma_c=sigma_c, mu=mu, method=method, sigma_x=sigma_x)
        return float(np.trace(dm))

<<<<<<< HEAD
    def _polarizability(self, green, mu=None, sigma_x=None):
        if mu is None:
            mu = self.mu
        if sigma_x is None:
            sigma_x = self.Sigma_x
        static = self.h0 + sigma_x
=======
    def _polarizability(self, green):
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
        if self.backend == "factorized":
            naux = self.pair_factors.shape[0]
            pol = np.zeros((self.nfreq, naux, naux), dtype=np.complex128)
            for inu, nu in enumerate(self.omega):
                acc = np.zeros((naux, naux), dtype=np.complex128)
                for iw, w in enumerate(self.omega):
<<<<<<< HEAD
                    g_shift = _interp_green_grid(
                        self.omega,
                        green,
                        w + nu,
                        mu=mu,
                        static=static,
                        eta=self.eta,
                    )
=======
                    g_shift = _interp_matrix_grid(self.omega, green, w + nu)
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
                    g = green[iw]
                    acc -= self.weights[iw] * np.einsum(
                        "Ppq,pr,Qrs,sq->PQ",
                        self.pair_factors,
                        g_shift,
                        self.pair_factors,
                        g,
                        optimize=True,
                    )
                pol[inu] = acc
            return pol

        dim = self.nso * self.nso
        pol = np.zeros((self.nfreq, dim, dim), dtype=np.complex128)
        for inu, nu in enumerate(self.omega):
            acc = np.zeros((self.nso, self.nso, self.nso, self.nso), dtype=np.complex128)
            for iw, w in enumerate(self.omega):
<<<<<<< HEAD
                g_shift = _interp_green_grid(
                    self.omega,
                    green,
                    w + nu,
                    mu=mu,
                    static=static,
                    eta=self.eta,
                )
=======
                g_shift = _interp_matrix_grid(self.omega, green, w + nu)
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
                g = green[iw]
                acc -= self.weights[iw] * np.einsum("pr,sq->pqrs", g_shift, g, optimize=True)
            pol[inu] = acc.reshape(dim, dim)
        return pol

    def _screened_interaction(self, pol):
        dim = pol.shape[1]
        eye = np.eye(dim)
        screened = np.empty_like(pol)
        for inu in range(self.nfreq):
            if self.backend == "factorized":
                screened[inu] = np.linalg.solve(eye - pol[inu], eye)
            else:
                screened[inu] = np.linalg.solve(eye - self.v_pair @ pol[inu], self.v_pair)
        return screened

<<<<<<< HEAD
    def _correlation_self_energy(self, green, screened, mu=None, sigma_x=None):
        if mu is None:
            mu = self.mu
        if sigma_x is None:
            sigma_x = self.Sigma_x
        static = self.h0 + sigma_x
=======
    def _correlation_self_energy(self, green, screened):
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
        if self.backend == "factorized":
            eye_aux = np.eye(self.pair_factors.shape[0], dtype=np.complex128)
            w_corr = screened - eye_aux[None, :, :]
        else:
            w_corr = screened - self.v_pair[None, :, :]
        sigma = np.zeros_like(green)
        for iw, w in enumerate(self.omega):
            acc = np.zeros((self.nso, self.nso), dtype=np.complex128)
            for inu, nu in enumerate(self.omega):
<<<<<<< HEAD
                g_shift = _interp_green_grid(
                    self.omega,
                    green,
                    w - nu,
                    mu=mu,
                    static=static,
                    eta=self.eta,
                )
=======
                g_shift = _interp_matrix_grid(self.omega, green, w - nu)
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
                if self.backend == "factorized":
                    acc -= self.weights[inu] * np.einsum(
                        "rs,Ppr,PQ,Qqs->pq",
                        g_shift,
                        self.pair_factors,
                        w_corr[inu],
                        self.pair_factors,
                        optimize=True,
                    )
                else:
                    wc4 = w_corr[inu].reshape(self.nso, self.nso, self.nso, self.nso)
                    acc -= self.weights[inu] * np.einsum("rs,prqs->pq", g_shift, wc4, optimize=True)
            sigma[iw] = acc
        return sigma

    def quasiparticle_estimate(self):
        """Return a Hermitian static estimate from ``Sigma_c(iw=0)``.

        This is not analytic continuation.  It is only a stable diagnostic for
        the prototype self-consistency loop.
        """
        eso = np.linalg.eigvalsh(self._static_hamiltonian())
        return 0.5 * (eso[0::2] + eso[1::2])

    def run(
        self,
        max_cycle=20,
        conv_tol=1e-6,
        damping=0.2,
        update_screening=True,
        update_exchange=True,
        verbose=0,
    ):
        """Run scGW or scGW0 self-consistency.

        ``update_screening=True`` recomputes ``P`` and ``W`` every cycle
        (full scGW).  ``False`` fixes ``W`` from the initial Green's function
        (scGW0).  ``update_exchange`` controls whether the bare-exchange part
        is rebuilt from the current Green's-function density matrix.
        """
        if not (0.0 < damping <= 1.0):
            raise ValueError("damping must be in the interval (0, 1].")

        sigma = np.array(self.Sigma_c, copy=True)
        sigma_x = np.array(self.Sigma_x, copy=True)
        self.mu = self._solve_mu(sigma, self.mu, sigma_x=sigma_x)
        green = self._dyson_green(sigma, self.mu, sigma_x=sigma_x)
        self.history = []
        self.mu_history = [self.mu]
        fixed_screened = None
        if not update_screening:
<<<<<<< HEAD
            pol0 = self._polarizability(green, mu=self.mu, sigma_x=sigma_x)
=======
            pol0 = self._polarizability(green)
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
            fixed_screened = self._screened_interaction(pol0)
            self.W0 = fixed_screened.copy()

        for cycle in range(1, int(max_cycle) + 1):
            if update_screening:
<<<<<<< HEAD
                pol = self._polarizability(green, mu=self.mu, sigma_x=sigma_x)
=======
                pol = self._polarizability(green)
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
                screened = self._screened_interaction(pol)
            else:
                pol = None
                screened = fixed_screened
<<<<<<< HEAD
            sigma_new = self._correlation_self_energy(
                green,
                screened,
                mu=self.mu,
                sigma_x=sigma_x,
            )
=======
            sigma_new = self._correlation_self_energy(green, screened)
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
            mixed_sigma = (1.0 - damping) * sigma + damping * sigma_new
            if update_exchange:
                trial_dm = self.make_density_matrix(
                    sigma_c=mixed_sigma,
                    mu=self.mu,
                    method="green",
                    sigma_x=sigma_x,
                )
                sigma_x_new = self._exchange_self_energy(trial_dm)
                mixed_sigma_x = (1.0 - damping) * sigma_x + damping * sigma_x_new
            else:
                mixed_sigma_x = sigma_x

            new_mu = self._solve_mu(mixed_sigma, self.mu, sigma_x=mixed_sigma_x)
            new_green = self._dyson_green(mixed_sigma, new_mu, sigma_x=mixed_sigma_x)

            sigma_delta = float(np.max(np.abs(mixed_sigma - sigma)))
            sigma_x_delta = float(np.max(np.abs(mixed_sigma_x - sigma_x)))
            green_delta = float(np.max(np.abs(new_green - green)))
            nelec = self.particle_number(
                sigma_c=mixed_sigma,
                mu=new_mu,
                method="green",
                sigma_x=mixed_sigma_x,
            )
            self.history.append({
                "cycle": cycle,
                "sigma_delta": sigma_delta,
                "sigma_x_delta": sigma_x_delta,
                "green_delta": green_delta,
                "mu": new_mu,
                "nelec": nelec,
                "update_screening": bool(update_screening),
                "update_exchange": bool(update_exchange),
            })
            if verbose:
                print(
                    f"scGW cycle {cycle}: max |dSigma| = {sigma_delta:.6e}, "
                    f"max |dSigma_x| = {sigma_x_delta:.6e}, "
                    f"max |dG| = {green_delta:.6e}, "
                    f"mu = {new_mu:.10f}, N = {nelec:.10f}"
                )

            sigma = mixed_sigma
            sigma_x = mixed_sigma_x
            green = new_green
            self.mu = new_mu
            self.mu_history.append(self.mu)
            if max(sigma_delta, sigma_x_delta, green_delta) < conv_tol:
                self.converged = True
                break
        else:
            self.converged = False

        self.Sigma_c = sigma
        self.Sigma_x = sigma_x
        self.sigma_x = self.Sigma_x
        self.G = green
<<<<<<< HEAD
        self.P = self._polarizability(green, mu=self.mu, sigma_x=self.Sigma_x)
=======
        self.P = self._polarizability(green)
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
        self.W = self._screened_interaction(self.P) if update_screening else fixed_screened
        self.density_matrix = self.make_density_matrix(method="green", sigma_x=self.Sigma_x)
        self.nelec = float(np.trace(self.density_matrix))
        self.e_qp = self.quasiparticle_estimate()
        self.e = self.e_qp
<<<<<<< HEAD
        self.energy_components = self.galitskii_migdal_total_energy(
            density_matrix=self.density_matrix,
        )
        self.e_tot_gm = self.energy_components["e_tot"]
        lw_components = self.luttinger_ward_total_energy(density_matrix=self.density_matrix)
        self.e_tot_lw = lw_components["e_tot"]
        self.e_tot = self.e_tot_gm
=======
        self.e_tot = None
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
        self.info = {
            "method": (
                "scgw_imaginary_axis_prototype"
                if update_screening
                else "scgw0_imaginary_axis_prototype"
            ),
            "converged": self.converged,
            "nfreq": self.nfreq,
            "wmax": self.wmax,
<<<<<<< HEAD
            "grid": self.grid,
=======
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
            "beta": self.beta,
            "mu": self.mu,
            "target_nelec": self.target_nelec,
            "nelec": self.nelec,
            "adjust_mu": self.adjust_mu,
            "update_screening": bool(update_screening),
            "update_exchange": bool(update_exchange),
            "density_nfreq": self.density_nfreq,
            "backend": self.backend,
            "density_matrix": "tail_corrected_matsubara_green_function",
<<<<<<< HEAD
            "total_energy": "galitskii_migdal",
            "e_tot_gm": self.e_tot_gm,
            "e_tot_lw": self.e_tot_lw,
=======
            "total_energy": "not_implemented",
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
        }
        return self

    def scgw0(self, **kwargs):
        kwargs["update_screening"] = False
        return self.run(**kwargs)

    def scgw(self, **kwargs):
        kwargs["update_screening"] = True
        return self.run(**kwargs)
<<<<<<< HEAD


def frequency_convergence(
    mf,
    nfreq_values=(7, 9, 11),
    wmax=20.0,
    method="scgw0",
    eta=1e-8,
    screening="TDH",
    beta=200.0,
    density_nfreq=129,
    grid="tangent",
    run_kwargs=None,
    energy_tol=1e-5,
    qp_tol=1e-4,
):
    """Run a small scGW/scGW0 frequency-grid convergence scan.

    Returns a list of dictionaries with total energies, diagnostic QP
    estimates, electron counts, changes relative to the previous grid, and
    ``grid_converged`` flags.  This is the practical validation tool for the
    finite-grid prototype until an external fully self-consistent GW reference
    is available.
    """
    run_kwargs = {} if run_kwargs is None else dict(run_kwargs)
    mode = str(method).lower().replace("-", "")
    if mode not in {"scgw0", "scgw"}:
        raise ValueError("method must be 'scgw0' or 'scgw'.")

    if np.ndim(wmax) == 0:
        wmax_values = [float(wmax)] * len(tuple(nfreq_values))
    else:
        wmax_values = [float(value) for value in wmax]
        if len(wmax_values) != len(tuple(nfreq_values)):
            raise ValueError("wmax sequence must have the same length as nfreq_values.")

    rows = []
    prev = None
    for nfreq, wmax_i in zip(tuple(nfreq_values), wmax_values):
        calc = SCGW(
            mf,
            nfreq=int(nfreq),
            wmax=wmax_i,
            eta=eta,
            screening=screening,
            beta=beta,
            density_nfreq=density_nfreq,
            grid=grid,
        )
        if mode == "scgw0":
            calc.scgw0(**run_kwargs)
        else:
            calc.scgw(**run_kwargs)

        row = {
            "method": mode,
            "nfreq": int(nfreq),
            "wmax": wmax_i,
            "grid": str(grid).lower(),
            "e_tot": float(calc.e_tot),
            "e_tot_gm": float(calc.e_tot_gm),
            "e_tot_lw": float(calc.e_tot_lw),
            "e_correlation": float(calc.energy_components["e_correlation"]),
            "mu": float(calc.mu),
            "nelec": float(calc.nelec),
            "e_qp": np.asarray(calc.e_qp, dtype=float).copy(),
            "converged": bool(calc.converged),
            "ncycle": len(calc.history),
            "backend": calc.backend,
        }
        if prev is None:
            row["delta_e_tot"] = None
            row["delta_qp_max"] = None
            row["grid_converged"] = False
        else:
            row["delta_e_tot"] = row["e_tot"] - prev["e_tot"]
            row["delta_qp_max"] = float(np.max(np.abs(row["e_qp"] - prev["e_qp"])))
            row["grid_converged"] = (
                abs(row["delta_e_tot"]) < float(energy_tol)
                and row["delta_qp_max"] < float(qp_tol)
            )
        row["energy_tol"] = float(energy_tol)
        row["qp_tol"] = float(qp_tol)
        row["reliable"] = bool(row["converged"] and row["grid_converged"])
        rows.append(row)
        prev = row
    return rows
=======
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
