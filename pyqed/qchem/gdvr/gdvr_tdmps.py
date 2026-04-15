import numpy as np
import logging
from pyqed.mps.tdmps import TDMPS
from pyqed.mps.mps import MPO, expect_mps, symmetric_to_dense
from pyqed.qchem.gdvr.gdvr_dmrg import run_gdvr_dmrg
import os
logger = logging.getLogger(__name__)

class GDVRTDMPS(TDMPS):
    def __init__(self, mol, Lz, Nz, basis_cfg, e_field_func, D=40, abelian_symmetry=True):
        """
        Integrated Time-Dependent solver for GDVR systems.
        """
        self.mol = mol
        self.Lz = Lz
        self.Nz = Nz
        self.basis_cfg = basis_cfg
        self.e_field_func = e_field_func
        self.D = D
        self.abelian_symmetry = abelian_symmetry

        # Placeholders for outputs from the static DMRG run
        self.dmrg_obj = None
        self.z_grid = None
        self.site_qn_maps = None
        self.psi_gs = None

    def run_dmrg(self, **dmrg_kwargs):
        """
        Executes the static GDVR-DMRG to find the ground state and build the Hamiltonian.
        """
        logger.info("Running static GDVR-DMRG to establish ground state...")
        
        # Unpack the dense Hamiltonian (mpo_dmrg) from the modified run_gdvr_dmrg
        E_gs, solver, z, site_qn_maps, dense_H_mpo = run_gdvr_dmrg(
            mol=self.mol, Lz=self.Lz, Nz=self.Nz, basis_cfg=self.basis_cfg,
            abelian_symmetry=self.abelian_symmetry,
            **dmrg_kwargs
        )
        
        self.dmrg_obj = solver
        self.z_grid = z
        self.site_qn_maps = site_qn_maps
        
        # Convert the Symmetric Ground State to a Dense MPS
        if self.abelian_symmetry:
            logger.info("Converting symmetric ground state to dense MPS for time evolution...")
            self.psi_gs = symmetric_to_dense(self.dmrg_obj.ground_state)
        else:
            self.psi_gs = self.dmrg_obj.ground_state 
        
        self.H_meas = MPO(dense_H_mpo) if isinstance(dense_H_mpo, list) else dense_H_mpo
        
        # 2. Build the Energy-Shifted Hamiltonian for Time Evolution
        logger.info(f"Shifting TD Hamiltonian by ground state energy: {E_gs}")
        L = len(self.H_meas.factors)
        I_factors = []
        for i in range(L):
            d = self.H_meas.factors[i].shape[2]
            W = np.zeros((1, 1, d, d), dtype=complex)
            for j in range(d): 
                W[0, 0, j, j] = 1.0
            
            # Apply the -E_gs shift to the very first site
            if i == 0: 
                W *= -E_gs  
            I_factors.append(W)
            
        I_mpo = MPO(I_factors)
        H_shifted = self.H_meas + I_mpo
        
        # Compress it back down to prevent bond dimension growth
        H_shifted = H_shifted.compress(max(self.H_meas.bond_orders()))
        
        # Initialize the TDMPS engine using the SHIFTED Hamiltonian
        super().__init__(H_mpo=H_shifted, D=self.D)
        
        return E_gs

    def _get_U_TD_mpo(self, time, delta_t):
        """
        Builds the D=1 Dense MPO for the time-dependent laser field at a specific time.
        """
        E_t = self.e_field_func(time)
        mpo_tensors = []
        
        for i in range(2 * self.Nz):
            z_i = self.z_grid[i // 2]
            
            phase = z_i * E_t * delta_t
            
            # spin orbital weare using
            U_local = np.diag([
                1.0,                   # empty 
                np.exp(-1j * phase)    # occupied
            ])
            
            # Shape (Left, Right, Out, In)
            W = np.zeros((1, 1, 2, 2), dtype=complex)
            W[0, 0, :, :] = U_local
            mpo_tensors.append(W)

        return MPO(mpo_tensors)


    def run(self, dt, steps, e_ops=[], interval=10, flush_interval=10,
            save_dir="dynamics_data", order=2):
        """
        Executes Strang-split dense time-evolution.

        Measures local density EVERY step.
        Measures dipole EVERY step.
        Measures global e_ops every `interval` steps.
        Dynamically saves to disk every `flush_interval` steps to prevent data loss.
        """
        if self.psi_gs is None:
            raise RuntimeError("You must call run_dmrg() before calling run()!")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Data will be dynamically flushed to: {save_dir}/")

        self.build_propagator(dt, order=order)

        logger.info(f"Starting TD-DMRG: {steps} steps, dt={dt}...")

        # Measurements are taken AFTER each full step
        self.times = (np.arange(steps) + 1) * dt

        # Track density and dipole every step
        spatial_densities = np.zeros((steps, self.Nz), dtype=float)
        dipoles = np.zeros(steps, dtype=float)

        # Track energy / other observables every interval
        num_obs_steps = steps // interval
        obs_times = np.zeros(num_obs_steps, dtype=float)
        observables = np.zeros((num_obs_steps, len(e_ops)), dtype=complex)

        psi = self.psi_gs.copy()
        current_time = 0.0

        for step_idx in range(steps):
            U_TD_half_start = self._get_U_TD_mpo(current_time, dt / 2)
            psi = U_TD_half_start @ psi

            psi = self.step(psi)

            current_time += dt

            U_TD_half_end = self._get_U_TD_mpo(current_time, dt / 2)
            psi = U_TD_half_end @ psi

            # Normalize explicitly
            current_norm = psi.norm()
            psi.factors[0] = psi.factors[0] / current_norm

            # local density every step 
            site_rdms = psi._calc_local_site_rdms()
            site_pops = np.array(
                [np.real(site_rdms[j][1, 1]) for j in range(2 * self.Nz)],
                dtype=float
            )

            # Sum spin-up + spin-down at each z-slice
            slice_density = site_pops[0::2] + site_pops[1::2]
            spatial_densities[step_idx] = slice_density

            total_electrons = np.sum(site_pops)

            # dipole every step
            dipole_t = float(np.dot(self.z_grid, slice_density))
            dipoles[step_idx] = dipole_t

            logger.info(
                f"Step {step_idx + 1} (t={current_time:.3f}): "
                f"Total Electrons = {total_electrons:.8f}, "
                f"Dipole = {dipole_t:.12f}"
            )

            # global observables every interval
            if (step_idx + 1) % interval == 0:
                obs_idx = (step_idx + 1) // interval - 1
                obs_times[obs_idx] = current_time
                observables[obs_idx] = [
                    expect_mps(psi.factors, e.factors) for e in e_ops
                ]

                if len(e_ops) > 0:
                    obs_str = ", ".join(
                        [f"Obs[{k}]={observables[obs_idx, k]}" for k in range(len(e_ops))]
                    )
                    logger.info(f"  Observable snapshot: {obs_str}")

            # flush to disk
            if save_dir and (step_idx + 1) % flush_interval == 0:
                data_file = os.path.join(save_dir, "density_evolution.npz")
                np.savez(
                    data_file,
                    times=self.times[:step_idx + 1],
                    densities=spatial_densities[:step_idx + 1],
                    dipoles=dipoles[:step_idx + 1],
                    z_grid=self.z_grid,
                    obs_times=obs_times[:(step_idx + 1) // interval],
                    observables=observables[:(step_idx + 1) // interval],
                )

                state_file = os.path.join(save_dir, "psi_latest.npz")
                np.savez_compressed(state_file, *psi.factors)

                logger.info(
                    f"  -> Flushed to disk at step {step_idx + 1} (t={current_time:.3f})"
                )

        self.observables = observables
        self.densities = spatial_densities
        self.dipoles = dipoles
        self.final_state = psi
        return self

def diagnose_dipole_coupling(H1, z):
    Z = np.diag(z)
    comm = Z @ H1 - H1 @ Z
    print("=" * 60)
    print("Dipole-coupling diagnostic")
    print("-" * 60)
    print(f"||[Z,H1]||            : {np.linalg.norm(comm):.6e}")
    print(f"max |[Z,H1]_ij|       : {np.max(np.abs(comm)):.6e}")
    print("=" * 60)
if __name__ == "__main__":
    from pyqed.qchem.gdvr.gdvr_mean_field import Molecule
    S_EXPS = [18.73113696, 2.825394365, 0.6401216923, 0.1612777588]
    basis_cfg = {'s': S_EXPS}
    charges = [1.0]*2
    # coords = [[0.0, 0.0, -3.6], [0.0, 0.0, -0.91], [0.0, 0.0, 0.91], [0.0, 0.0, 3.6]]
    coords = [[0.0, 0.0, -2],[0.0, 0.0, 2]]
    mol = Molecule(charges, coords, nelec=2, spin = 0)

    def electric_field(t):
        # return 0
        return 2* np.sin(0.5 * t)

    td_solver = GDVRTDMPS(
        mol=mol, Lz=6.0, Nz=16, basis_cfg=basis_cfg, 
        e_field_func=electric_field, D=30
    )

    E_gs = td_solver.run_dmrg(pre_opt_cycles=10, dmrg_bond_dim=30)
    print(f"Ground state found: {E_gs}")
    H1 = td_solver.dmrg_obj.Hcore
    diagnose_dipole_coupling(H1, td_solver.z_grid)
    td_solver.run(dt=0.01, steps=100, e_ops=[td_solver.H], interval=2)