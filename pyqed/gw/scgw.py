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

from pyqed.gw.gw import GW, _reference_total_energy, _spin_pair_factors


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

        # Reuse the existing GW builder for spin-orbital MO integrals,
        # HF eigenvalues, and the static mean-field potential.
        ref = GW(mf, screening=screening, eta=max(eta, 1e-8))
        self._ref = ref

        self.nso = ref.nso
        self.nocc = ref.nocc
        self.target_nelec = float(self.nocc if target_nelec is None else target_nelec)
        self.e_mf = np.asarray(ref.e_mf, dtype=float)
        self.v_mf = np.asarray(ref.v_mf, dtype=float)
        self.backend = "factorized" if ref.eri is None else "dense"
        self.pair_factors = None
        if self.backend == "factorized":
            self.pair_factors = np.asarray(_spin_pair_factors(ref), dtype=float)
            self.eri = None
        else:
            self.eri = np.asarray(ref.eri, dtype=float)
        self.omega, self.weights = _symmetric_frequency_grid(nfreq, wmax)
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
        self.e_scf = _reference_total_energy(ref)
        self.info = None
        self.W0 = None

    def _initial_mu(self):
        homo = self.e_mf[self.nocc - 1]
        lumo = self.e_mf[self.nocc]
        return 0.5 * (homo + lumo)

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

    def _polarizability(self, green):
        if self.backend == "factorized":
            naux = self.pair_factors.shape[0]
            pol = np.zeros((self.nfreq, naux, naux), dtype=np.complex128)
            for inu, nu in enumerate(self.omega):
                acc = np.zeros((naux, naux), dtype=np.complex128)
                for iw, w in enumerate(self.omega):
                    g_shift = _interp_matrix_grid(self.omega, green, w + nu)
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
                g_shift = _interp_matrix_grid(self.omega, green, w + nu)
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

    def _correlation_self_energy(self, green, screened):
        if self.backend == "factorized":
            eye_aux = np.eye(self.pair_factors.shape[0], dtype=np.complex128)
            w_corr = screened - eye_aux[None, :, :]
        else:
            w_corr = screened - self.v_pair[None, :, :]
        sigma = np.zeros_like(green)
        for iw, w in enumerate(self.omega):
            acc = np.zeros((self.nso, self.nso), dtype=np.complex128)
            for inu, nu in enumerate(self.omega):
                g_shift = _interp_matrix_grid(self.omega, green, w - nu)
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
            pol0 = self._polarizability(green)
            fixed_screened = self._screened_interaction(pol0)
            self.W0 = fixed_screened.copy()

        for cycle in range(1, int(max_cycle) + 1):
            if update_screening:
                pol = self._polarizability(green)
                screened = self._screened_interaction(pol)
            else:
                pol = None
                screened = fixed_screened
            sigma_new = self._correlation_self_energy(green, screened)
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
        self.P = self._polarizability(green)
        self.W = self._screened_interaction(self.P) if update_screening else fixed_screened
        self.density_matrix = self.make_density_matrix(method="green", sigma_x=self.Sigma_x)
        self.nelec = float(np.trace(self.density_matrix))
        self.e_qp = self.quasiparticle_estimate()
        self.e = self.e_qp
        self.e_tot = None
        self.info = {
            "method": (
                "scgw_imaginary_axis_prototype"
                if update_screening
                else "scgw0_imaginary_axis_prototype"
            ),
            "converged": self.converged,
            "nfreq": self.nfreq,
            "wmax": self.wmax,
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
            "total_energy": "not_implemented",
        }
        return self

    def scgw0(self, **kwargs):
        kwargs["update_screening"] = False
        return self.run(**kwargs)

    def scgw(self, **kwargs):
        kwargs["update_screening"] = True
        return self.run(**kwargs)
