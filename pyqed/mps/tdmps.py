import numpy as np
from pyqed.mps.mps import MPS, MPO, expmpo, apply_mpo, expect_mps
import logging


class TDMPS:
    def __init__(self, H_mpo, D=40):
        """
        Time-Dependent MPS Solver (Layout Agnostic).

        Args:
            psi0 (MPS): Initial state, MPS class object
            H_mpo (MPO): Hamiltonian. (Lv, Rv, P_oout, P_in)
            dt (complex): Time step.
            bond_dim (int): Max bond dimension.
        """

        # self.psi0 = psi0
        self.H = H_mpo
        # self.dt = dt
        self.bond_dim = self.D = D
        # self.order = order
        # self.scale = scale
        # self.time = 0.0
        # self.U = self._construct_propagator() # Propagator

        # DO NOT CHANGE
        self.U = None
        self.observables = None
        

    def build_propagator(self, dt, order=2, scale=0):
        """
        Construct the MPO of the short-time propagator
        .. math::

            U = exp(-i H  \Delta t)
        

        Parameters
        ----------
        D : TYPE, optional
            maximal bond dimension for U. The default is 40.
        order : TYPE, optional
            DESCRIPTION. The default is 2.
        scale : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        logging.info(f"Build propagator (dt={dt}, order={order})...")
        constant = -1j * dt

        self.U = expmpo(self.H, constant=constant, D=self.D,
                        method='taylor', order=order, scale=scale)

        return self.U

    def step(self, psi):
        """
        Evolve system by one step dt.
        """
        
        # Apply MPO (Returns tensors in ['lv', 'p', 'rv'] layout)
        # psi = propagate(self.U.factors, psi)
        
        psi = self.U @ psi
        return psi.compress(self.D).normalize()


    def fast_run(self):
        pass

    def run(self, psi0, dt, steps, e_ops=[], interval=1):
        """
        Run time evolution.

        Parameters
        ----------
        steps : TYPE
            DESCRIPTION.
        e_ops : list, optional
            list of MPOs for observables. The default is [].
        interval : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if not isinstance(psi0, MPS):
            raise TypeError("Initialize state is not an MPS object.")
            
        # dt = self.dt 
        
        self.build_propagator(dt)

        print(f"Starting time-evolution for {steps} steps with dt = {dt}...")
        self.times = np.arange(0, steps, interval) * dt
        
        # if e_ops:
        observables = np.zeros((len(self.times), len(e_ops)))
            
        psi = psi0
        for i in range(steps//interval):
            for k in range(interval):
                
                psi = self.step(psi)

            # compute observables

            # Measure Observables
            # psi_std = [self.psi._get_std_B(k) for k in range(self.psi.L)]

            # note the first observable is computed at t = dt
            observables[i] = [expect_mps(psi.factors, e.factors) for e in e_ops]

            
            
            # if (i + 1) % 10 == 0:
            #     # Print Energy
            #     e_str = f", Obs[0]={np.real(results['obs'][0][-1]):.6f}" if observables else ""
            #     print(f"Step {i+1}/{steps}, Time={self.time:.4f}{e_str}")
        self.observables = observables
        
        return self

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pyqed.models.heisenberg import Heisenberg

    mol = Heisenberg(L=10)
    H = mol.build_H_mpo()
    neel = mol.build_neel_state()
    
    dt = 0.01 
    steps = 10

    # Initialize TDMPS Solver
    td = TDMPS(H, D=40)
    td.run(neel, dt, steps, e_ops=[H])
    

    # # Plot if you wish
    # times = results['time']
    # energy = np.real(results['obs'][0])
    # norms = results['norm_check']

    # print("\nSimulation Complete.")
    # print(f"Final Energy: {energy[-1]:.6f}")
    # print(f"Energy Conservation Error: {np.max(np.abs(energy - energy[0])):.2e}")
