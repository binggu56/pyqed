import numpy as np
from mps import MPS, MPO, expect_mps
from pyqed.mps.decompose import compress 

def apply_mpo_fixed(w_list, b_list, chi_max):
    """
    Correct implementation of MPO applied to MPS.
    Strictly assumes and maintains (Left, Physical, Right) layout.
    """
    L = len(b_list)
    if L != len(w_list):
        raise ValueError("MPO and MPS lengths do not match.")
        
    raw_factors = []
    
    for i in range(L):
        W = w_list[i] # (wL, wR, pOut, pIn)
        B = b_list[i] # (bL, pIn, bR)
        
        # Contract Physical indices: B[1] (pIn) with W[3] (pIn)
        T = np.tensordot(B, W, axes=(1, 3))
        
        # Current indices: [bL, bR, wL, wR, pOut]
        # Target order for compress: (bL, wL) as NewLeft, pOut as NewPhys, (bR, wR) as NewRight
        # Target transpose: bL(0), wL(2), pOut(4), bR(1), wR(3)
        T = T.transpose(0, 2, 4, 1, 3)
        
        # Reshape to (NewLeft, NewPhys, NewRight)
        s = T.shape
        new_left = s[0] * s[1]
        new_phys = s[2]
        new_right = s[3] * s[4]
        
        T_flat = T.reshape(new_left, new_phys, new_right)
        raw_factors.append(T_flat)
        
    res = compress(raw_factors, chi_max)
    if isinstance(res, tuple):
        return res[0]
    return res

class TDDMRG:
    def __init__(self, psi0, H_mpo, dt, bond_dim=100, order=4, scale=0):
        """
        Generic Time-Dependent DMRG Solver using MPO Time Evolution.
        Enforces (Left, Physical, Right) layout for internal storage.
        """
        # --- 1. Fix Layout of psi0 to (Left, Physical, Right) ---
        fixed_factors = []
        for B in psi0.factors:
            if B.ndim == 3:
                s = B.shape
                # Detect if input is (Phys, Left, Right) -> (d, 1, 1)
                if s[1] == 1 and s[2] == 1 and s[0] > 1:
                    B = B.transpose(1, 0, 2)
            fixed_factors.append(B)
        
        self.psi = MPS(fixed_factors)
        self.H = H_mpo
        self.dt = dt
        self.bond_dim = bond_dim
        self.order = order
        self.scale = scale
        self.time = 0.0
        
        self.U = self._construct_propagator()

    def _construct_propagator(self):
        print(f"Constructing MPO Propagator (dt={self.dt}, order={self.order})...")
        
        constant = -1j * self.dt

        return self.H.exponential(
            constant=constant, 
            D=self.bond_dim, 
            method='taylor', 
            order=self.order, 
            scale=self.scale
        )

    def compute_norm(self):
        """
        Calculates <psi|psi> using internal layout (Left, Phys, Right).
        """
        val = np.ones((1, 1)) 
        for B in self.psi.factors:
            # B: (chi_L, d, chi_R)
            # Contract val (a, b) with B (b, c, d) -> T(a, c, d)
            T = np.tensordot(val, B, axes=(1, 0))
            # Contract T with B* (a, c, e) on Left(a) and Phys(c)
            val = np.tensordot(T, B.conj(), axes=([0, 1], [0, 1]))
        return np.sqrt(np.abs(val[0, 0]))

    def normalize_psi(self):
        norm = self.compute_norm()
        if norm > 1e-12:
            self.psi.factors[0] /= norm
        return norm

    def step(self):
        """
        Evolve the system forward by one step dt.
        """
        new_factors = apply_mpo_fixed(self.U.factors, self.psi.factors, self.bond_dim)
        self.psi = MPS(new_factors)
        self.normalize_psi()
        
        step_mag = abs(self.dt) if np.isreal(self.dt) else abs(self.dt.imag)
        self.time += step_mag

    def evolve(self, steps, observables=[]):
        """
        Run the evolution loop.
        """
        results = {
            'time': [],
            'norm_check': [],
            'obs': [[] for _ in observables]
        }

        print(f"Starting evolution for {steps} steps...")
        for i in range(steps):
            results['time'].append(self.time)
            results['norm_check'].append(self.compute_norm())
            
            # --- FIX: Convert to Legacy Layout for expect_mps ---
            # expect_mps in mps.py expects (Physical, Left, Right).
            # We currently store (Left, Physical, Right).
            # We create a temporary transposed list for measurement.
            legacy_psi = [B.transpose(1, 0, 2) for B in self.psi.factors]
            
            for idx, op in enumerate(observables):
                # Pass the legacy format to the legacy function
                val = expect_mps(legacy_psi, op.factors, legacy_psi)
                results['obs'][idx].append(val)
            
            self.step()
            
            if (i + 1) % 10 == 0:
                print(f"Step {i+1}/{steps}, Time={self.time:.4f}")
                
        return results
    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    # from pyqed.mps.mps import MPO, MPS
    # from pyqed.mps.tddmrg import TDDMRG

    def build_heisenberg_model(N):
        """
        Constructs the Heisenberg Hamiltonian MPO for N sites.
        """
        d = 2
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

    def build_initial_state(N):
        """
        Builds a Neel state |up down up down ...>
        """
        d = 2
        A_up = np.zeros((d, 1, 1)); A_up[0, 0, 0] = 1.0
        A_dn = np.zeros((d, 1, 1)); A_dn[1, 0, 0] = 1.0
        
        factors = [A_up, A_dn] * (N // 2)
        if N % 2 == 1: factors.append(A_up)
        
        return MPS(factors)


    N = 10
    dt = 0.05
    dt = -0.1j
    steps = 500
    bond_dim = 40
    
    print(f"Initializing Heisenberg Chain (N={N})...")
    H_mpo = build_heisenberg_model(N)
    psi0 = build_initial_state(N)

    # Initialize TDDMRG Solver 
    # We use order=2 for speed in this demo, order=4 is better for precision
    solver = TDDMRG(psi0, H_mpo, dt, bond_dim=bond_dim, order=2)

    # Run Evolution
    # We measure the total energy <H> as a sanity check. 
    # For time-independent H, energy should be conserved.
    results = solver.evolve(steps=steps, observables=[H_mpo])

    # Display Results
    times = results['time']
    energy = np.real(results['obs'][0])
    norms = results['norm_check']

    print("\nSimulation Complete.")
    print(f"Final Energy: {energy[-1]:.6f}")
    print(f"Energy conservation error: {np.max(np.abs(energy - energy[0])):.2e}")

    # Optional: Plotting
    try:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(times, energy, 'b.-')
        plt.title('Total Energy <H>(t)')
        plt.xlabel('Time')
        plt.ylabel('Energy')

        plt.subplot(1, 2, 2)
        plt.plot(times, norms, 'r--')
        plt.title('Norm <psi|psi>')
        plt.xlabel('Time')
        plt.ylim(0.99, 1.01)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not found, skipping plot.")