import numpy as np
import logging
import os

from pyqed.mps.tdmps import TDMPS
from pyqed.mps.mps import MPO, symmetric_to_dense
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo
from pyqed.qchem.gdvr.gdvr_dmrg import run_gdvr_dmrg, get_jw_term_robust

logger = logging.getLogger(__name__)


class GDVRTDMPS(TDMPS):
    """
    Debug-first split propagator for GDVR TD-MPS.

    Current split:
        U_diag(dt/2) -> U_hop(dt) -> U_diag(dt/2)

    where U_diag contains:
        - diagonal one-body term t_ii n_i
        - laser term z_i E(t) n_i
        - optional constant energy shift

    IMPORTANT:
        This version does NOT yet apply the exact density-density phase
        exp[-i dt sum_{i<j} V_ij n_i n_j].
        Keep that OFF until particle number is confirmed stable.
    """

    def __init__(self, mol, Lz, Nz, basis_cfg, e_field_func, D=40, abelian_symmetry=True):
        super().__init__(H_mpo=None, D=D)
        self.mol = mol
        self.Lz = Lz
        self.Nz = Nz
        self.basis_cfg = basis_cfg
        self.e_field_func = e_field_func
        self.D = D
        self.abelian_symmetry = abelian_symmetry

        self.psi_gs = None
        self.z_grid = None
        self.site_qn_maps = None

        self.H_hop_mpo = None
        self.t_ii = None
        self.V_coul = None
        self.E_gs = 0.0

    def run_dmrg(self, **dmrg_kwargs):
        logger.info("Phase 1: Static GDVR-DMRG")

        E_gs, solver, z_grid, site_qn_maps, _ = run_gdvr_dmrg(
            self.mol,
            self.Lz,
            self.Nz,
            self.basis_cfg,
            abelian_symmetry=self.abelian_symmetry,
            **dmrg_kwargs
        )

        self.E_gs = E_gs
        self.z_grid = z_grid
        self.site_qn_maps = site_qn_maps

        # Convert symmetric GS to dense MPS for TD evolution.
        # Do NOT manually pad physical dimension; that can corrupt occupations.
        if self.abelian_symmetry:
            logger.info("Converting symmetric ground state to dense MPS")
            self.psi_gs = symmetric_to_dense(solver.ground_state)
        else:
            self.psi_gs = solver.ground_state

        # Store diagonal one-body and density-density data.
        self.t_ii = np.diag(solver.Hcore).copy()
        self.V_coul = np.array(solver.V_coul, copy=True)

        # Build hopping-only MPO from off-diagonal Hcore.
        basis = [BasisSimpleElectron(k) for k in range(2 * self.Nz)]
        hop_terms = []

        rows, cols = np.nonzero(np.abs(solver.Hcore) > 1e-12)
        for i, j in zip(rows, cols):
            if i == j:
                continue

            val = solver.Hcore[i, j]

            # spin-up block
            hop_terms.append(
                get_jw_term_robust([r"a^\dagger", "a"], [2 * i, 2 * j], val)
            )
            # spin-down block
            hop_terms.append(
                get_jw_term_robust([r"a^\dagger", "a"], [2 * i + 1, 2 * j + 1], val)
            )

        mpo_obj = Mpo(Model(basis=basis, ham_terms=hop_terms), algo="qr")
        self.H_hop_mpo = MPO([w.transpose(0, 3, 1, 2) for w in mpo_obj.matrices])

        # TDMPS base class uses self.H to build exp(-i H dt)
        self.H = self.H_hop_mpo
        return self.E_gs

    def _renormalize_in_place(self, psi):
        n = np.abs(np.atleast_1d(psi.norm())[0])
        if n > 1e-16:
            psi.factors[0] /= n
        return psi

    def _apply_one_body_phases(self, psi, dt, time):
        """
        Apply diagonal one-body phases in-place:
            exp[-i dt sum_i eps_i(t) n_i]

        with
            eps_i(t) = t_ii[z(i)] + z_i E(t) - E_gs / L_sites
        """
        E_t = self.e_field_func(time)
        L_sites = 2 * self.Nz
        shift = self.E_gs / L_sites

        for so in range(L_sites):
            z_idx = so // 2
            z_val = self.z_grid[z_idx]
            phase = (self.t_ii[z_idx] + z_val * E_t - shift) * dt

            # physical basis is |0>, |1>
            psi.factors[so][:, 1, :] *= np.exp(-1j * phase)

        return psi

    def step(self, psi):
        """
        Apply the hopping propagator for one full step, then compress.
        """
        psi = self.U @ psi
        psi = psi.compress(self.D)
        psi = self._renormalize_in_place(psi)
        return psi

    def _measure_site_populations(self, psi):
        site_rdms = psi._calc_local_site_rdms()
        return np.array([np.real(site_rdms[j][1, 1]) for j in range(2 * self.Nz)])

    def run(self, dt, steps, interval=10, save_dir="dynamics_results"):
        if self.psi_gs is None:
            raise RuntimeError("Call run_dmrg() before run().")

        os.makedirs(save_dir, exist_ok=True)

        psi = self.psi_gs
        current_time = 0.0

        # Build hopping propagator U_hop(dt)
        self.build_propagator(dt, order=2)

        # Build cached interaction half-step gates U_V(dt/2)
        self._prepare_density_gates(dt)

        n_save = steps // interval
        times = np.zeros(n_save)
        spatial_densities = np.zeros((n_save, self.Nz))
        total_electrons = np.zeros(n_save)

        logger.info(f"Starting split TD-MPS with U_1b + U_V + U_hop, dt={dt}, steps={steps}")

        for step_idx in range(steps):
            # Strang split:
            # U_1b(dt/2) -> U_V(dt/2) -> U_hop(dt) -> U_V(dt/2) -> U_1b(dt/2)

            self._apply_one_body_phases(psi, dt / 2, current_time)
            psi = self._apply_density_density_gates(psi)

            psi = self.step(psi)
            current_time += dt

            psi = self._apply_density_density_gates(psi)
            self._apply_one_body_phases(psi, dt / 2, current_time)

            psi.right_canonicalize()
            psi = self._renormalize_in_place(psi)

            if (step_idx + 1) % interval == 0:
                obs_idx = (step_idx + 1) // interval - 1
                site_pops = self._measure_site_populations(psi)
                rho_z = site_pops[0::2] + site_pops[1::2]

                times[obs_idx] = current_time
                spatial_densities[obs_idx] = rho_z
                total_electrons[obs_idx] = np.sum(site_pops)

                logger.info(
                    f"Step {step_idx + 1} (t={current_time:.6f}): "
                    f"Total Electrons = {total_electrons[obs_idx]:.10f}"
                )

        self.psi_final = psi
        self.times = times
        self.densities = spatial_densities
        self.total_electrons = total_electrons

        np.savez(
            os.path.join(save_dir, "density_evolution.npz"),
            times=times,
            densities=spatial_densities,
            total_electrons=total_electrons,
            z_grid=self.z_grid,
        )

        return self

    def _renormalize_in_place(self, psi):
        nrm = np.abs(np.atleast_1d(psi.norm())[0])
        if nrm > 1e-16:
            psi.factors[0] /= nrm
        return psi


    def _build_density_term_list(self, tol=1e-12):
        """
        Expand spatial V_coul into spin-orbital density-density terms
        using the SAME convention as gdvr_dmrg.py.

        Static code convention:
        - on-site: V[i,i] * n_{2i} n_{2i+1}
        - off-site unique pair i<k:
                V[i,k] * (n_up-up + n_dn-dn + n_up-dn + n_dn-up)
        because gdvr_dmrg.py loops over both (i,k) and (k,i) and uses 0.5*V each time.
        """
        terms = []
        V = np.asarray(self.V_coul)
        Nz = self.Nz

        for i in range(Nz):
            # on-site opposite-spin term
            if abs(V[i, i]) > tol:
                terms.append((2 * i, 2 * i + 1, float(V[i, i])))

            # off-site spatial pairs
            for k in range(i + 1, Nz):
                vik = V[i, k]
                if abs(vik) <= tol:
                    continue

                # four spin-channel products, each with coefficient V[i,k]
                terms.append((2 * i,     2 * k,     float(vik)))
                terms.append((2 * i + 1, 2 * k + 1, float(vik)))
                terms.append((2 * i,     2 * k + 1, float(vik)))
                terms.append((2 * i + 1, 2 * k,     float(vik)))

        # Deterministic order; shorter-range first is a reasonable numerical default
        terms.sort(key=lambda x: (abs(x[1] - x[0]), x[0], x[1]))
        return terms


    def _build_pair_phase_mpo(self, p, q, phi):
        """
        Exact bond-dimension-2 MPO for exp(-i phi n_p n_q)
        on a spin-orbital chain with local basis |0>, |1>.

        Since n^2 = n,
            exp(-i phi n_p n_q) = I + (exp(-i phi) - 1) n_p n_q.
        """
        if p == q:
            raise ValueError("Pair-phase MPO requires p != q.")
        if p > q:
            p, q = q, p

        L = 2 * self.Nz
        alpha = np.exp(-1j * phi) - 1.0

        I2 = np.eye(2, dtype=complex)
        n_op = np.array([[0.0, 0.0],
                        [0.0, 1.0]], dtype=complex)

        mpo = []

        for site in range(L):
            # full internal tensor with bond dim 2
            W = np.zeros((2, 2, 2, 2), dtype=complex)

            # generic identity propagation
            W[0, 0, :, :] = I2
            W[1, 1, :, :] = I2

            # source site p injects alpha * n into the "active" channel
            if site == p:
                W[0, 1, :, :] = alpha * n_op

            # sink site q closes the channel with n
            if site == q:
                W[1, 0, :, :] = n_op

            # boundary reduction: select left state 0 and right state 0
            if site == 0:
                W = W[0:1, :, :, :]
            if site == L - 1:
                W = W[:, 0:1, :, :]

            mpo.append(W)

        return MPO(mpo)


    def _prepare_density_gates(self, dt, tol=1e-12):
        """
        Cache half-step interaction gates U_V(dt/2).
        """
        tau = 0.5 * dt
        self._density_terms = self._build_density_term_list(tol=tol)
        self._density_gate_mpos = []

        for p, q, coeff in self._density_terms:
            phi = tau * coeff
            gate = self._build_pair_phase_mpo(p, q, phi)
            self._density_gate_mpos.append((p, q, coeff, gate))

        logger.info(f"Prepared {len(self._density_gate_mpos)} density-density half-step gates.")


    def _apply_density_density_gates(self, psi):
        """
        Apply the cached half-step U_V(dt/2) gates sequentially.

        Since all n_i n_j commute, this ordering is exact for the interaction piece.
        Compression introduces only numerical truncation error.
        """
        for p, q, coeff, gate in self._density_gate_mpos:
            psi = gate @ psi
            psi = psi.compress(self.D)
            psi = self._renormalize_in_place(psi)
        return psi


    def _measure_site_populations(self, psi):
        site_rdms = psi._calc_local_site_rdms()
        return np.array([np.real(site_rdms[j][1, 1]) for j in range(2 * self.Nz)])

if __name__ == "__main__":
    from pyqed.qchem.gdvr.gdvr_mean_field import Molecule
    S_EXPS = [18.73113696, 2.825394365, 0.6401216923, 0.1612777588]
    basis_cfg = {'s': S_EXPS}
    charges = [1.0] * 4
    coords = [[0.0, 0.0, -3.6], [0.0, 0.0, -0.91], [0.0, 0.0, 0.91], [0.0, 0.0, 3.6]]
    mol = Molecule(charges, coords, nelec=4, spin=0)

    def laser(t): return 0.005 * np.cos(0.1 * t)

    td_solver = GDVRTDMPS(mol=mol, Lz=6.0, Nz=16, basis_cfg=basis_cfg, 
                          e_field_func=laser, D=30)

    E_gs = td_solver.run_dmrg(pre_opt_cycles=10, dmrg_bond_dim=30)
    print(f"Ground state found: {E_gs:.10f} Ha")

    td_solver.run(dt=0.005, steps=100, interval=1)