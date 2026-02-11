import numpy as np
from pyqed.mps.mps import MPS, MPO, expect_mps, expmpo, apply_mpo
import logging

# def propagate(w_list, psi_mps):
#     """
#     one step for psi = U @ psi done in tensor
#     Contracts MPO (U) with MPS (psi)

#     Args:
#         w_list: List of MPO tensors. shape: (Left, Right, Phys_Out, Phys_In) TODO: also add index label for MPO
#         psi_mps: MPS object (handles its own internal layout).

#     Returns:
#         list: New tensors in standard (Left, Phys, Right) layout.
#     """
#     L = psi_mps.L
#     if L != len(w_list):
#         raise ValueError(f"MPO length ({len(w_list)}) and MPS length ({L}) mismatch.")

#     psi_new = []

#     for i in range(L):
#         W = w_list[i] # Shape: (wL, wR, pOut, pIn)
#         B = psi_mps._get_std_B(i) # MPS to (Left, Phys, Right)

#         # psi = U @ psi
#         # B: (bL, pIn, bR)
#         # W: (wL, wR, pOut, pIn)
#         # Contract B[Phys] (axis 1) with W[PhysIn] (axis 3)
#         T = np.tensordot(B, W, axes=(1, 3))

#         # Result T: (bL, bR, wL, wR, pOut)
#         # rearrange to: (NewLeft, NewPhys, NewRight)
#         # NewLeft  = (bL, wL) -> Indices (0, 2)
#         # NewPhys  = (pOut)   -> Index   (4)
#         # NewRight = (bR, wR) -> Indices (1, 3)
#         # Transpose: (0, 2, 4, 1, 3)
#         T = T.transpose(0, 2, 4, 1, 3)

#         # Fuse Bonds
#         s = T.shape
#         dim_L = s[0] * s[1]
#         dim_P = s[2]
#         dim_R = s[3] * s[4]
#         T_flat = T.reshape(dim_L, dim_P, dim_R)
#         psi_new.append(T_flat)
#     return MPS(psi_new)

class TDMPS:
    def __init__(self, psi0, H_mpo, dt, D=40):
        """
        Time-Dependent MPS Solver (Layout Agnostic).

        Args:
            psi0 (MPS): Initial state, MPS class object
            H_mpo (MPO): Hamiltonian. (Lv, Rv, P_oout, P_in)
            dt (complex): Time step.
            bond_dim (int): Max bond dimension.
        """
        if not isinstance(psi0, MPS):
            raise TypeError("initialize psi0 as an MPS class object.")
        self.psi0 = psi0
        self.H = H_mpo
        self.dt = dt
        self.bond_dim = self.D = D
        # self.order = order
        # self.scale = scale
        # self.time = 0.0
        # self.U = self._construct_propagator() # Propagator

        # DO NOT CHANGE
        self.U = None
        

    def build_propagator(self, order=2, scale=0):
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

        logging.info(f"Build propagator (dt={self.dt}, order={order})...")
        constant = -1j * self.dt

        self.U = expmpo(self.H, constant=constant, D=self.D,
                        method='taylor', order=order, scale=scale)

        return self.U


    # def normalize(self, psi):
    #     """
    #     Re-normalizes the MPS.
    #     """
    #     norm = psi.norm()
    #     if norm > 1e-12:
    #         psi.Bs[0] /= np.sqrt(norm)
    #     return norm

    def step(self, psi):
        """
        Evolve system by one step dt.
        """
        
        # Apply MPO (Returns tensors in ['lv', 'p', 'rv'] layout)
        # psi = propagate(self.U.factors, psi)
        
        psi = self.U @ psi

        # Create MPS object for Compression
        return psi.compress(self.D).normalize()

        # # Normalize, update time
        # self.normalize(psi)

        # step_mag = abs(self.dt) if np.isreal(self.dt) else abs(self.dt.imag)
        # self.time += step_mag
        # return psi
    def fast_run(self):
        pass

    def run(self, steps, e_ops=[], interval=1):
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
        dt = self.dt 
        
        # results = {
        #     'time': [],
        #     'norm_check': [],
        #     'obs': [[] for _ in observables]
        # }
        if self.U is None:
            self.build_propagator()

        print(f"Starting Evolution for {steps} steps...")
        self.times = np.arange(0, steps, interval) * dt

        observables = np.zeros((len(e_ops), len(self.times)))
        psi = self.psi0
        for i in range(steps//interval):
            for k in range(interval):
                
                psi = self.step(psi)

            # compute observables


            # results['time'].append(self.time)
            # results['norm_check'].append(self.compute_norm())

            # Measure Observables
            # psi_std = [self.psi._get_std_B(k) for k in range(self.psi.L)]

            # for idx, op in enumerate(e_ops):
            #     val = expect_mps(psi, op.factors)
            #     results['obs'][idx].append(val)

            
            
            # if (i + 1) % 10 == 0:
            #     # Print Energy
            #     e_str = f", Obs[0]={np.real(results['obs'][0][-1]):.6f}" if observables else ""
            #     print(f"Step {i+1}/{steps}, Time={self.time:.4f}{e_str}")
        return self

if __name__ == "__main__":
    import matplotlib.pyplot as plt


    # below is an example using TDMPS with heisenberg chain
    # helpers here are just dense MPS and MPO builderr
    def build_heisenberg_mpo(N):
        """
        Constructs the Heisenberg Hamiltonian MPO for N sites.
        """
        # d = 2
        I = np.identity(2)
        Z = np.zeros((2, 2))
        Sz = np.array([[0.5, 0], [0, -0.5]])
        Sp = np.array([[0, 0], [1, 0]])
        Sm = np.array([[0, 1], [0, 0]])

        # Define the bulk MPO tensor W (5x5 matrix of operators)
        # Rows/Cols: I, Sz, Sp, Sm, Hamiltonian-accumulator
        W = np.array([[I, Sz, 0.5*Sp, 0.5*Sm, Z],
                    [Z, Z,  Z,      Z,      Sz],
                    [Z, Z,  Z,      Z,      Sm],
                    [Z, Z,  Z,      Z,      Sp],
                    [Z, Z,  Z,      Z,      I]])

        # Boundary vectors
        Wfirst = np.array([[I, Sz, 0.5*Sp, 0.5*Sm, Z]]) # 1x5
        Wlast = np.array([[Z], [Sz], [Sm], [Sp], [I]])  # 5x1

        # Construct full list
        H_factors = [Wfirst] + ([W] * (N - 2)) + [Wlast]
        return MPO(H_factors)

    def build_neel_state(N):
        """
        Builds a Neel state |up down up down ...>
        Layout: (Left, Phys, Right) -> labels=['lv', 'p', 'rv']
        """
        factors = []
        for i in range(N):
            # MPS Shape (L,P,R)
            B = np.zeros((1, 2, 1))
            if i % 2 == 0:
                B[0, 0, 0] = 1.0 # Up
            else:
                B[0, 1, 0] = 1.0 # Down
            factors.append(B)
        return MPS(factors, labels=['lv', 'p', 'rv'])

    def build_ferom_state(N):
        """
        Builds a ferromagnetic state |up up up up ...>
        Layout: (Left, Phys, Right) -> labels=['lv', 'p', 'rv']
        """
        factors = []
        for i in range(N):
            # Shape (Left=1, Phys=2, Right=1)
            B = np.zeros((1, 2, 1))
            B[0, 0, 0] = 1.0 # Up
            factors.append(B)
        return MPS(factors, labels=['lv', 'p', 'rv'])

    N = 10
    dt = 0.01 - 0.0j
    # dt = -0.1j
    steps = 10
    bond_dim = 50

    print(f"Initializing Heisenberg Chain (N={N})...")
    H_mpo = build_heisenberg_mpo(N)
    psi0 = build_neel_state(N)
    # psi0 = build_ferom_state(N)

    # Initialize TDMPS Solver
    solver = TDMPS(psi0, H_mpo, dt)

    # Run Evolution
    results = solver.run(steps, e_ops=[H_mpo])

    # # Plot if you wish
    # times = results['time']
    # energy = np.real(results['obs'][0])
    # norms = results['norm_check']

    # print("\nSimulation Complete.")
    # print(f"Final Energy: {energy[-1]:.6f}")
    # print(f"Energy Conservation Error: {np.max(np.abs(energy - energy[0])):.2e}")

    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 2, 1)
    # plt.plot(times, energy, 'b.-')
    # plt.title('Total Energy <H>(t)')
    # plt.xlabel('Time')
    # plt.ylabel('Energy')
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # plt.plot(times, norms, 'r--')
    # plt.title('Norm <psi|psi>')
    # plt.xlabel('Time')
    # plt.ylim(0.99, 1.01)
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()