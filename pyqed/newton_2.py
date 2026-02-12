import numpy as np
import scipy.linalg as la

# =============================================================================
# 1. Physics Module (Hubbard Model)
#    - Responsibilities: H_core, V_int, Energy, Gradient, Energy-Hessian
#    - Knows NOTHING about Newton's method or Optimization logic.
# =============================================================================
class HubbardModel:
    """
    Physics engine for the 1D Hubbard Model.
    Provides Energy, Gradient, and Hessian (Energy curvature) for a given U.
    """
    def __init__(self, n_sites, n_electrons, U_int=4.0, t_hop=1.0):
        self.n = n_sites
        self.p = n_electrons // 2
        
        # --- 1. Static Hamiltonian Tensors (Calculated once) ---
        # Kinetic (h): Shape (n, n)
        self.h = np.zeros((self.n, self.n))
        for i in range(self.n - 1):
            self.h[i, i+1] = -t_hop
            self.h[i+1, i] = -t_hop
            
        # Repulsion (V): Shape (n, n, n, n)
        self.V = np.zeros((self.n, self.n, self.n, self.n))
        for i in range(self.n):
            self.V[i, i, i, i] = U_int
            
        # --- 2. Static Density Matrices (RHF) ---
        self.gamma = 2.0 * np.eye(self.p)
        self.Gamma = np.zeros((self.p, self.p, self.p, self.p))
        for a in range(self.p):
            for b in range(self.p):
                self.Gamma[a, b, a, b] += 4.0 
                self.Gamma[a, b, b, a] -= 2.0

    def get_energy(self, U):
        """Returns scalar Energy E(U)."""
        E1 = np.einsum('pq, pa, qb, ab ->', self.h, U, U, self.gamma)
        V_mo = np.einsum('pqrs, pa, qb, rc, sd -> abcd', self.V, U, U, U, U)
        E2 = 0.5 * np.einsum('abcd, acdb ->', V_mo, self.Gamma)
        return E1 + E2

    def get_gradient(self, U):
        """Returns Energy Gradient (n x p)."""
        grad_1 = 2.0 * np.einsum('pq, qb, ab -> pa', self.h, U, self.gamma)
        grad_2 = 2.0 * np.einsum('pqrs, qb, rc, sd, acdb -> pa', 
                                 self.V, U, U, U, self.Gamma)
        return grad_1 + grad_2

    def get_energy_hessian(self, U):
        """
        Returns the Energy Hessian d^2E/dU^2 (dim x dim matrix).
        Uses Finite Difference on the Analytic Gradient for robustness.
        """
        n, p = U.shape
        dim = n * p
        epsilon = 1e-5
        
        H_energy = np.zeros((dim, dim))
        u_flat = U.flatten()
        
        # Base gradient
        g0 = self.get_gradient(U).flatten()
        
        # Compute Jacobian column by column
        for i in range(dim):
            u_perturb = u_flat.copy()
            u_perturb[i] += epsilon
            
            # Recalculate gradient at perturbed point
            U_temp = u_perturb.reshape(n, p)
            g_perturb = self.get_gradient(U_temp).flatten()
            
            # Finite Difference
            H_energy[:, i] = (g_perturb - g0) / epsilon
            
        return H_energy

# =============================================================================
# 2. Math Module (Newton Engine)
#    - Responsibilities: KKT System, Line Search, Retraction, Convergence
#    - Knows NOTHING about Electrons, Hamiltonians, or Physics.
# =============================================================================
class UnitaryNewtonSolver:
    """
    Generic Newton-Raphson solver for minimizing f(U) subject to U^T U = I.
    """
    def __init__(self, max_iter=15, tol=1e-7):
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, U_init, func_E, func_Grad, func_Hess):
        """
        Args:
            U_init: Initial guess (n x p).
            func_E: Callable returning float (Energy).
            func_Grad: Callable returning (n x p) array (Gradient).
            func_Hess: Callable returning (dim x dim) array (Energy Hessian).
        """
        U = U_init.copy()
        print('U shape',U.shape)
        n, p = U.shape
        print('{},{}'.format(n,p))
        dim = n * p
        
        print(f"--- Starting Newton Engine (n={n}, p={p}) ---")
        print(f"Initial Value: {func_E(U):.8f}\n")
        
        # Initialize Multipliers (Lambda)
        Lambda = np.zeros((p, p))
        
        for step in range(1, self.max_iter + 1):
            # 1. Get Physics Data (Generic calls)
            E_curr = func_E(U)
            Grad = func_Grad(U)
            print('grad', Grad.shape)
            H_Energy = func_Hess(U)
            
            # 2. Update Lagrange Multipliers (Projection)
            # Lambda ~ U.T @ Grad (Symmetrized)
            L_proxy = U.T @ Grad
            Lambda = 0.5 * (L_proxy + L_proxy.T)
            
            # 3. Build Lagrangian Hessian
            # H_Lag = H_Energy - H_Constraint
            # H_Constraint = 2 * Lambda (x) I_n (Manifold curvature)
            H_Constraint = 2.0 * np.kron(Lambda, np.eye(n))
            H_Total = H_Energy - H_Constraint
            
            # 4. Build KKT Constraints (Matrix B)
            # Enforces: U.T * dU + dU.T * U = 0
            num_cons = p * (p + 1) // 2
            B = np.zeros((num_cons, dim))
            idx = 0
            for j in range(p):
                for i in range(j + 1):
                    # Place U_j at block i, U_i at block j
                    B[idx, i*n : (i+1)*n] += U[:, j]
                    B[idx, j*n : (j+1)*n] += U[:, i]
                    idx += 1
            
            # 5. Assemble Full KKT System
            # [ H   B.T ] [ dU ] = [ -g ]
            # [ B    0  ] [ dL ]   [  0 ]
            zeros_block = np.zeros((num_cons, num_cons))
            top = np.hstack([H_Total, B.T])
            bot = np.hstack([B, zeros_block])
            KKT = np.vstack([top, bot])
            
            rhs = np.concatenate([-Grad.flatten(), np.zeros(num_cons)])
            
            # 6. Solve Linear System
            try:
                sol = la.solve(KKT, rhs)
                delta_U = sol[:dim].reshape(n, p)
            except la.LinAlgError:
                print("  [!] Singular Matrix. Adding Regularization.")
                KKT[0:dim, 0:dim] += 1e-3 * np.eye(dim)
                sol = la.solve(KKT, rhs)
                delta_U = sol[:dim].reshape(n, p)
            
            # 7. Line Search & Retraction
            step_norm = la.norm(delta_U)
            alpha = 1.0
            
            for _ in range(5):
                # Retract: U_new = QR(U + alpha*dU)
                U_trial_raw = U + alpha * delta_U
                U_trial, _ = la.qr(U_trial_raw, mode='economic')
                
                # Check Descent
                E_trial = func_E(U_trial)
                if E_trial < E_curr + 1e-9:
                    U = U_trial
                    E_curr = E_trial
                    break
                alpha *= 0.5
                
            print(f"Step {step}: Value = {E_curr:.8f} | |Step| = {step_norm:.6e} | Alpha = {alpha}")
            
            if step_norm < self.tol:
                print("--> Converged!")
                break
                
        return U, E_curr

# =============================================================================
#  Main
# =============================================================================
if __name__ == "__main__":
    # hubbard model
    N_SITES = 4
    N_ELEC = 3 
    U_INT = 6.0
    
    # 1. Instantiate Model
    print("1. Initializing Hubbard Model...")
    model = HubbardModel(n_sites=N_SITES, n_electrons=N_ELEC, U_int=U_INT)
    
    # 2. randomarized Initial Guess
    print("2. Generating Initial Guess (Huckel)...")
    evals, evecs = la.eigh(model.h)
    p = model.p
    U_guess = evecs[:, :p]
    # Add noise to break symmetry
    U_guess += np.random.rand(N_SITES, p) * 0.1
    U_guess, _ = la.qr(U_guess, mode='economic')

    # --- B. Feed Stuff into Newton Engine ---
    print("3. Feeding functions into Newton Engine...")
    engine = UnitaryNewtonSolver(max_iter=20, tol=1e-7)
    

    
    # We pass the Model's bound methods as the "Physics Functions"
    final_U, final_E = engine.solve(
        U_init=U_guess, 
        func_E=model.get_energy,        # The Engine calls this to get E
        func_Grad=model.get_gradient,   # The Engine calls this to get Gradient
        func_Hess=model.get_energy_hessian # The Engine calls this to get H_E
    )
    
    # --- C. Verification Patch ---
    print("\n" + "="*40)
    print("       FINAL RESULT VERIFICATION       ")
    print("="*40)
    print(f"Final Energy: {final_E:.8f}")
    
    # Check Orthogonality
    overlap = final_U.T @ final_U
    ortho_err = la.norm(overlap - np.eye(p))
    print(f"Orthogonality Error: {ortho_err:.2e}")
    
    if ortho_err < 1e-10:
        print(">> Status: VALID (Orthonormal)")
    else:
        print(">> Status: INVALID (Constraint Broken)")