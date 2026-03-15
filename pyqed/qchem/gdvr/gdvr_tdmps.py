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
        
        # Initialize the TDMPS engine using the dense Hamiltonian
        H_mpo = MPO(dense_H_mpo) if isinstance(dense_H_mpo, list) else dense_H_mpo
        super().__init__(H_mpo=H_mpo, D=self.D)
        
        return E_gs

    def _get_U_TD_mpo(self, time, delta_t):
        """
        Builds the D=1 Dense MPO for the time-dependent laser field at a specific time.
        """
        E_t = self.e_field_func(time)
        mpo_tensors = []
        
        for i in range(2 * self.Nz):
            z_i = self.z_grid[i // 2]
            
            # Phase for a single electron at this site
            phase = z_i * E_t * delta_t
            
            # Local unitary for d=2 spin-orbital basis
            U_local = np.diag([
                1.0,                   # Empty state
                np.exp(-1j * phase)    # Occupied state
            ])
            
            # Shape (LeftBond, RightBond, PhysOut, PhysIn)
            W = np.zeros((1, 1, 2, 2), dtype=complex)
            W[0, 0, :, :] = U_local
            mpo_tensors.append(W)

        return MPO(mpo_tensors)

    def run(self, dt, steps, e_ops=[], interval=10, flush_interval=10, save_dir="dynamics_data"):
        """
        Executes Strang-split dense time-evolution.
        
        Measures local density EVERY step.
        Measures global e_ops every `interval` steps.
        Dynamically saves to disk every `flush_interval` steps to prevent data loss.
        """
        if self.psi_gs is None:
            raise RuntimeError("You must call run_dmrg() before calling run()!")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Data will be dynamically flushed to: {save_dir}/")

        self.build_propagator(dt)
        
        logger.info(f"Starting TD-DMRG: {steps} steps, dt={dt}...")
        
        # Track time and density every step
        self.times = np.arange(steps) * dt
        spatial_densities = np.zeros((steps, self.Nz))
        
        # Track energy every interval step
        num_obs_steps = steps // interval
        obs_times = np.zeros(num_obs_steps)
        observables = np.zeros((num_obs_steps, len(e_ops)), dtype=complex)
        
        n_op = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
            
        psi = self.psi_gs
        current_time = 0.0

        for step_idx in range(steps):
            # Half-step TD
            U_TD_half_start = self._get_U_TD_mpo(current_time, dt / 2)
            psi = U_TD_half_start @ psi 
            # Full-step Static with SVD
            psi = self.step(psi) 
            current_time += dt 
            # Half-step TD again
            U_TD_half_end = self._get_U_TD_mpo(current_time, dt / 2)
            psi = U_TD_half_end @ psi
            current_norm = psi.norm() 
            # print(current_norm)
            psi.factors[0] = psi.factors[0] / current_norm
            
            # measure Local Density Every Step
            site_rdms = psi._calc_local_site_rdms()
            site_pops = np.array([np.real(site_rdms[j][1, 1]) for j in range(2 * self.Nz)])
            spatial_densities[step_idx] = site_pops[0::2] + site_pops[1::2]

            # measure energy every 'interval'
            if (step_idx + 1) % interval == 0:
                obs_idx = (step_idx + 1) // interval - 1
                obs_times[obs_idx] = current_time
                observables[obs_idx] = [expect_mps(psi.factors, e.factors) for e in e_ops]

            if save_dir and (step_idx + 1) % flush_interval == 0:
                # Update the density data file (Overwrites single file)
                data_file = os.path.join(save_dir, "density_evolution.npz")
                np.savez(data_file, 
                         times=self.times[:step_idx+1], 
                         densities=spatial_densities[:step_idx+1],
                         z_grid=self.z_grid,
                         obs_times=obs_times[:(step_idx + 1) // interval],
                         observables=observables[:(step_idx + 1) // interval])
                
                # Overwrite the latest wavefunction backup
                state_file = os.path.join(save_dir, "psi_latest.npz")
                np.savez_compressed(state_file, *psi.factors)
                
                logger.info(f"  -> Flushed to disk at step {step_idx + 1} (t={current_time:.3f})")
            
        self.observables = observables
        self.densities = spatial_densities
        return self
if __name__ == "__main__":
    from pyqed.qchem.gdvr.gdvr_mean_field import Molecule
    S_EXPS = [18.73113696, 2.825394365, 0.6401216923, 0.1612777588]
    basis_cfg = {'s': S_EXPS}
    charges = [1.0]*4
    coords = [[0.0, 0.0, -3.6], [0.0, 0.0, -0.91], [0.0, 0.0, 0.91], [0.0, 0.0, 3.6]]
    mol = Molecule(charges, coords, nelec=4, spin = 0)

    def electric_field(t):
        # return 0
        return 0.5 * np.cos(0.1 * t)

    td_solver = GDVRTDMPS(
        mol=mol, Lz=6.0, Nz=32, basis_cfg=basis_cfg, 
        e_field_func=electric_field, D=20
    )

    E_gs = td_solver.run_dmrg(pre_opt_cycles=10, dmrg_bond_dim=20)
    print(f"Ground state found: {E_gs}")

    td_solver.run(dt=0.01, steps=1000, e_ops=[td_solver.H], interval=10)