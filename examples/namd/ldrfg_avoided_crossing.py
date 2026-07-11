"""Small analytic avoided-crossing model for LDRFG.

The model has one LDR/DVR coordinate ``x`` and one frozen-Gaussian coordinate
``q``.  The electronic Hamiltonian at mixed geometry ``(x, q)`` is

    h_el(x, q) = [[z, delta], [delta, -z]]
    z = a_x x + a_q q

plus an optional scalar harmonic confinement added to both adiabatic surfaces.
The adiabatic states are real analytic vectors, which makes this a useful
smoke/validation model for the LDRFG equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.namd import LDRFG


@dataclass
class AvoidedCrossingLDRFGModel:
    x_grid: np.ndarray
    kinetic_x: np.ndarray
    mass_y: float = 1.0
    a_x: float = 1.0
    a_q: float = 0.8
    delta: float = 0.2
    k_x: float = 0.02
    k_q: float = 0.03
    gamma_y: float = 1.0

    def z(self, q):
        q = float(np.asarray(q)[0])
        return self.a_x * self.x_grid + self.a_q * q

    def theta(self, q):
        return 0.5 * np.arctan2(self.delta, self.z(q))

    def dtheta_dq(self, q):
        z = self.z(q)
        return -0.5 * self.delta * self.a_q / (z * z + self.delta * self.delta)

    def electronic_vectors(self, q):
        """Return adiabatic electronic vectors U[n, diabatic, adiabatic]."""
        theta = self.theta(q)
        c = np.cos(theta)
        s = np.sin(theta)
        vecs = np.empty((self.x_grid.size, 2, 2), dtype=float)
        vecs[:, :, 0] = np.stack((-s, c), axis=-1)
        vecs[:, :, 1] = np.stack((c, s), axis=-1)
        return vecs

    def electronic_vector_gradients(self, q):
        """Return dU/dq with shape U_q[n, diabatic, adiabatic]."""
        theta = self.theta(q)
        dtheta = self.dtheta_dq(q)
        c = np.cos(theta)
        s = np.sin(theta)
        grads = np.empty((self.x_grid.size, 2, 2), dtype=float)
        grads[:, :, 0] = np.stack((-c * dtheta, -s * dtheta), axis=-1)
        grads[:, :, 1] = np.stack((-s * dtheta, c * dtheta), axis=-1)
        return grads

    def energies(self, q):
        q0 = float(np.asarray(q)[0])
        z = self.z(q)
        rho = np.sqrt(z * z + self.delta * self.delta)
        scalar = 0.5 * self.k_x * self.x_grid**2 + 0.5 * self.k_q * q0**2
        return np.stack((scalar - rho, scalar + rho), axis=-1)

    def grad_energies(self, q):
        q0 = float(np.asarray(q)[0])
        z = self.z(q)
        rho = np.sqrt(z * z + self.delta * self.delta)
        drho = self.a_q * z / rho
        dscalar = self.k_q * q0
        return np.stack((dscalar - drho, dscalar + drho), axis=-1)[None, :, :]

    def overlap(self, q):
        vecs = self.electronic_vectors(q)
        return np.einsum("mdb,nda->mbna", vecs, vecs)

    def grad_overlap(self, q):
        vecs = self.electronic_vectors(q)
        dvecs = self.electronic_vector_gradients(q)
        grad = np.einsum("mdb,nda->mbna", dvecs, vecs)
        grad += np.einsum("mdb,nda->mbna", vecs, dvecs)
        return grad[None, :, :, :, :]

    def berry(self, q):
        vecs = self.electronic_vectors(q)
        dvecs = self.electronic_vector_gradients(q)
        local = np.einsum("ndb,nda->nba", vecs, dvecs)
        berry = np.zeros((1, self.x_grid.size, 2, self.x_grid.size, 2), dtype=float)
        for n in range(self.x_grid.size):
            berry[0, n, :, n, :] = local[n]
        return berry

    def solver(self, include_berry=True):
        berry = self.berry if include_berry else None
        return LDRFG(
            self.kinetic_x,
            masses_y=[self.mass_y],
            energies=self.energies,
            overlap=self.overlap,
            grad_energies=self.grad_energies,
            grad_overlap=self.grad_overlap,
            berry=berry,
            gamma=np.array([[self.gamma_y]]),
        )


def finite_difference_force(solver, c, q, p, eps=1.0e-5):
    q = np.asarray(q, dtype=float)
    qp = q.copy()
    qm = q.copy()
    qp[0] += eps
    qm[0] -= eps
    ep = solver.energy(c, qp, p).real
    em = solver.energy(c, qm, p).real
    return -(ep - em) / (2.0 * eps)


def second_derivative_kinetic(npts, dx, mass=1.0):
    kinetic = np.diag(np.full(npts, 1.0 / (mass * dx * dx)))
    kinetic += np.diag(np.full(npts - 1, -0.5 / (mass * dx * dx)), k=1)
    kinetic += np.diag(np.full(npts - 1, -0.5 / (mass * dx * dx)), k=-1)
    return kinetic


def first_derivative_momentum(npts, dx):
    deriv = np.diag(np.full(npts - 1, 0.5 / dx), k=1)
    deriv += np.diag(np.full(npts - 1, -0.5 / dx), k=-1)
    return -1j * deriv


def frozen_gaussian_on_grid(q_grid, q0, p0, gamma):
    g = np.exp(-0.5 * gamma * (q_grid - q0) ** 2 + 1j * p0 * (q_grid - q0))
    return g / np.linalg.norm(g)


def exact_initial_state(model, q_grid, c, q0, p0):
    vecs = model.electronic_vectors([q0])
    gq = frozen_gaussian_on_grid(q_grid, q0, p0, model.gamma_y)
    psi = np.zeros((model.x_grid.size, q_grid.size, 2), dtype=complex)
    for n in range(model.x_grid.size):
        diabatic_amplitude = vecs[n] @ c[n]
        psi[n, :, :] = gq[:, None] * diabatic_amplitude[None, :]
    psi /= np.sqrt(np.vdot(psi.ravel(), psi.ravel()))
    return psi


def exact_diabatic_hamiltonian(model, q_grid):
    nx = model.x_grid.size
    nq = q_grid.size
    tx = sp.csr_matrix(model.kinetic_x)
    dq = q_grid[1] - q_grid[0]
    tq = sp.csr_matrix(second_derivative_kinetic(nq, dq, mass=model.mass_y))
    ix = sp.eye(nx, format="csr")
    iq = sp.eye(nq, format="csr")
    iel = sp.eye(2, format="csr")

    h = sp.kron(sp.kron(tx, iq), iel, format="csr")
    h += sp.kron(sp.kron(ix, tq), iel, format="csr")

    rows = []
    cols = []
    data = []
    for n, x in enumerate(model.x_grid):
        for j, q in enumerate(q_grid):
            z = model.a_x * x + model.a_q * q
            scalar = 0.5 * model.k_x * x * x + 0.5 * model.k_q * q * q
            base = (n * nq + j) * 2
            rows.extend([base, base + 1, base, base + 1])
            cols.extend([base, base + 1, base + 1, base])
            data.extend([scalar + z, scalar - z, model.delta, model.delta])

    v = sp.coo_matrix((data, (rows, cols)), shape=(nx * nq * 2, nx * nq * 2)).tocsr()
    return h + v


def exact_observables(model, q_grid, psi):
    nx = model.x_grid.size
    nq = q_grid.size
    psi = psi.reshape(nx, nq, 2)
    prob_q = np.sum(np.abs(psi) ** 2, axis=(0, 2))
    q_mean = float(np.sum(prob_q * q_grid))

    dq = q_grid[1] - q_grid[0]
    pmat = first_derivative_momentum(nq, dq)
    p_mean = np.einsum("nqa,qk,nka->", psi.conj(), pmat, psi)

    pop_ad = np.zeros(2)
    for j, q in enumerate(q_grid):
        vecs = model.electronic_vectors([q])
        amp = np.einsum("nda,nd->na", vecs, psi[:, j, :])
        pop_ad += np.sum(np.abs(amp) ** 2, axis=0)

    return {
        "q": q_mean,
        "p": float(np.real_if_close(p_mean)),
        "pop_ad": pop_ad,
    }


def exact_reference(model, q_grid, c0, q0, p0, times):
    h = exact_diabatic_hamiltonian(model, q_grid)
    psi0 = exact_initial_state(model, q_grid, c0, q0, p0)
    states = expm_multiply(
        -1j * h,
        psi0.ravel(),
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
    )
    energies = np.array([np.vdot(psi, h @ psi).real for psi in states])
    obs = [exact_observables(model, q_grid, psi) for psi in states]
    return {
        "times": np.asarray(times),
        "q": np.array([item["q"] for item in obs]),
        "p": np.array([item["p"] for item in obs]),
        "pop_ad": np.array([item["pop_ad"] for item in obs]),
        "energy": energies,
    }


def initial_ldrfg_state(x, q0=-1.0, p0=1.5, state=0):
    c = np.zeros((x.size, 2), dtype=complex)
    envelope = np.exp(-0.5 * ((x + 1.0) / 0.7) ** 2)
    c[:, state] = envelope / np.linalg.norm(envelope)
    return c, np.array([q0], dtype=float), np.array([p0], dtype=float)


def run_demo(nsteps=200, dt=0.02, initial_state=0):
    x = np.linspace(-3.0, 3.0, 9)
    dx = x[1] - x[0]
    kinetic_x = second_derivative_kinetic(x.size, dx)

    model = AvoidedCrossingLDRFGModel(x, kinetic_x, mass_y=5.0)
    solver = model.solver()
    c, q, p = initial_ldrfg_state(x, state=initial_state)

    times = [0.0]
    qs = [q.copy()]
    ps = [p.copy()]
    pops = [[np.sum(np.abs(c[:, 0]) ** 2), np.sum(np.abs(c[:, 1]) ** 2)]]
    energies = [solver.energy(c, q, p).real]

    for step in range(nsteps):
        c, q, p = solver.step_rk4(c, q, p, dt)
        c /= np.sqrt(np.vdot(c.ravel(), c.ravel()))
        times.append((step + 1) * dt)
        qs.append(q.copy())
        ps.append(p.copy())
        pops.append([np.sum(np.abs(c[:, 0]) ** 2), np.sum(np.abs(c[:, 1]) ** 2)])
        energies.append(solver.energy(c, q, p).real)

    return {
        "times": np.asarray(times),
        "q": np.asarray(qs)[:, 0],
        "p": np.asarray(ps)[:, 0],
        "pop_ad": np.asarray(pops),
        "energy": np.asarray(energies),
    }


def compare_to_exact(nsteps=80, dt=0.01, nq=81, q_domain=(-6.0, 6.0), initial_state=0):
    x = np.linspace(-3.0, 3.0, 9)
    dx = x[1] - x[0]
    kinetic_x = second_derivative_kinetic(x.size, dx)
    model = AvoidedCrossingLDRFGModel(x, kinetic_x, mass_y=5.0)
    solver = model.solver()
    c, q, p = initial_ldrfg_state(x, state=initial_state)

    times = [0.0]
    ldrfg_q = [q[0]]
    ldrfg_p = [p[0]]
    ldrfg_pop = [[np.sum(np.abs(c[:, 0]) ** 2), np.sum(np.abs(c[:, 1]) ** 2)]]
    ldrfg_energy = [solver.energy(c, q, p).real]

    c0 = c.copy()
    q0 = float(q[0])
    p0 = float(p[0])

    for step in range(nsteps):
        c, q, p = solver.step_rk4(c, q, p, dt)
        c /= np.sqrt(np.vdot(c.ravel(), c.ravel()))
        times.append((step + 1) * dt)
        ldrfg_q.append(q[0])
        ldrfg_p.append(p[0])
        ldrfg_pop.append([np.sum(np.abs(c[:, 0]) ** 2), np.sum(np.abs(c[:, 1]) ** 2)])
        ldrfg_energy.append(solver.energy(c, q, p).real)

    q_grid = np.linspace(float(q_domain[0]), float(q_domain[1]), nq)
    exact = exact_reference(model, q_grid, c0, q0, p0, np.asarray(times))
    ldrfg = {
        "times": np.asarray(times),
        "q": np.asarray(ldrfg_q),
        "p": np.asarray(ldrfg_p),
        "pop_ad": np.asarray(ldrfg_pop),
        "energy": np.asarray(ldrfg_energy),
    }
    diff = {
        "q_rms": float(np.sqrt(np.mean((ldrfg["q"] - exact["q"]) ** 2))),
        "p_rms": float(np.sqrt(np.mean((ldrfg["p"] - exact["p"]) ** 2))),
        "pop_ad_rms": float(np.sqrt(np.mean((ldrfg["pop_ad"] - exact["pop_ad"]) ** 2))),
        "ldrfg_energy_drift": float(ldrfg["energy"][-1] - ldrfg["energy"][0]),
        "exact_energy_drift": float(exact["energy"][-1] - exact["energy"][0]),
    }
    return {"ldrfg": ldrfg, "exact": exact, "diff": diff}


def plot_comparison(result, filename="ldrfg_avoided_crossing_comparison.png"):
    import matplotlib.pyplot as plt

    times = result["ldrfg"]["times"]
    ldrfg = result["ldrfg"]
    exact = result["exact"]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(times, exact["q"], "k-", lw=2.0, label="exact")
    ax.plot(times, ldrfg["q"], "C0--", lw=2.0, label="LDRFG")
    ax.set_xlabel("time / a.u.")
    ax.set_ylabel(r"$\langle q\rangle$")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(times, exact["p"], "k-", lw=2.0, label="exact")
    ax.plot(times, ldrfg["p"], "C1--", lw=2.0, label="LDRFG")
    ax.set_xlabel("time / a.u.")
    ax.set_ylabel(r"$\langle p_q\rangle$")

    ax = axes[1, 0]
    ax.plot(times, exact["pop_ad"][:, 0], "k-", lw=2.0, label="exact S0")
    ax.plot(times, exact["pop_ad"][:, 1], "0.45", lw=2.0, label="exact S1")
    ax.plot(times, ldrfg["pop_ad"][:, 0], "C2--", lw=2.0, label="LDRFG S0")
    ax.plot(times, ldrfg["pop_ad"][:, 1], "C3--", lw=2.0, label="LDRFG S1")
    ax.set_xlabel("time / a.u.")
    ax.set_ylabel("adiabatic population")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[1, 1]
    ax.plot(times, exact["energy"] - exact["energy"][0], "k-", lw=2.0, label="exact")
    ax.plot(times, ldrfg["energy"] - ldrfg["energy"][0], "C4--", lw=2.0, label="LDRFG")
    ax.set_xlabel("time / a.u.")
    ax.set_ylabel(r"$E(t)-E(0)$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    fig.suptitle("LDRFG vs exact avoided-crossing dynamics")
    fig.savefig(filename, dpi=180)
    plt.close(fig)
    return filename


def plot_population_dynamics(result, filename="ldrfg_excited_population_dynamics.png"):
    import matplotlib.pyplot as plt

    times = result["ldrfg"]["times"]
    ldrfg = result["ldrfg"]["pop_ad"]
    exact = result["exact"]["pop_ad"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(times, exact[:, 0], "k-", lw=2.2, label="exact S0")
    ax.plot(times, exact[:, 1], color="0.45", lw=2.2, label="exact S1")
    ax.plot(times, ldrfg[:, 0], "C2--", lw=2.2, label="LDRFG S0")
    ax.plot(times, ldrfg[:, 1], "C3--", lw=2.2, label="LDRFG S1")
    ax.set_xlabel("time / a.u.")
    ax.set_ylabel("adiabatic population")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Excited-state avoided-crossing population dynamics")
    fig.savefig(filename, dpi=180)
    plt.close(fig)
    return filename


if __name__ == "__main__":
    result = compare_to_exact()
    print("RMS errors:", result["diff"])
    print("Final LDRFG q, p:", result["ldrfg"]["q"][-1], result["ldrfg"]["p"][-1])
    print("Final exact q, p:", result["exact"]["q"][-1], result["exact"]["p"][-1])
    print("Saved plot:", plot_comparison(result))
