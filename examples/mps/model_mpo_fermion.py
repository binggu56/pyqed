"""


Key Feature:
------------
The 'BasisSimpleElectron' has 'is_electron = True'. 
The MPO class will automatically insert 'sigma_z' gates (Jordan-Wigner strings)
to preserve fermionic anti-commutation relations.
"""

import logging
import numpy as np
from pyqed.operator_mpo.model import Model
from pyqed.operator_mpo.operator import Op
from pyqed.operator_mpo.basis import BasisSimpleElectron
from pyqed.operator_mpo.model_mpo import ModelMPO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def tight_binding(nsite, t, v):
    """
    ModelMPO Example: Fermionic Hopping Model (Tight-Binding)

    Hamiltonian Definition:
    -----------------------
    The Hamiltonian represents spinless fermions hopping on a 1D chain with 
    disordered on-site potentials.

        H = -t * Σ (c†_i * c_{i+1} + c†_{i+1} * c_i)  +  Σ (v_i * n_i)
                 i                                     i

    Where:
      - t       : Hopping amplitude
      - v_i     : Random on-site potential
      - c†, c   : Fermionic creation/annihilation operators (a^dag, a)
      - n_i     : Number operator (n)

    Parameters
    ----------
    nsite : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    logger.info(f"--- Building MPO for Fermionic Chain (N={nsite}) ---")

    # ------------------------------------------------------------------
    # 2. Build Hamiltonian Terms
    # ------------------------------------------------------------------
    ham_terms = []
    
    # Part A: Nearest Neighbor Hopping
    # We use the explicit product syntax to ensure 'op_mat' receives 
    # exactly "a^dagger" and "a" as defined in your class.
    for i in range(nsite - 1):
        # Forward hopping: -t * (a^dag_i * a_{i+1})
        term_fwd = Op(r"a^\dagger", i) * Op("a", i+1) * (-t)
        ham_terms.append(term_fwd)
        
        # Backward hopping (h.c.): -t * (a^dag_{i+1} * a_i)
        # Note: (a^dag_i a_{i+1})^dagger = a^dag_{i+1} a_i
        term_bwd = Op(r"a^\dagger", i+1) * Op("a", i) * (-t)
        ham_terms.append(term_bwd)
    
    # Part B: On-site Random Potential
    # We use "n" since your BasisSimpleElectron supports it directly.
    for i in range(nsite):
        v_i = v[i]
        term_on_site = Op("n", i, factor=v_i)
        ham_terms.append(term_on_site)

    # ------------------------------------------------------------------
    # 3. Initialize Basis and Model
    # ------------------------------------------------------------------
    # Use the custom BasisSimpleElectron that includes JW logic (is_electron=True)
    basis = [BasisSimpleElectron(dof=i) for i in range(nsite)]
    
    model = Model(basis=basis, ham_terms=ham_terms)
    
    # ------------------------------------------------------------------
    # 4. Generate MPO
    # ------------------------------------------------------------------
    logger.info("Generating Fermionic MPO (Automatic Jordan-Wigner)...")
    
    # 'algo="qr"' creates the MPO using QR decomposition
    mpo = ModelMPO(model, algo="qr")
    
    return mpo
    
    # logger.info("MPO Successfully Created.")

    # logger.info("\n--- Inspection ---")
    # for site_idx, W in enumerate(mpo):
    #     logger.info(f"Site {site_idx}: Tensor Shape {W.shape}")

    # # Dense Verification (Small Systems Only)
    # logger.info("\n--- Converting to Dense Matrix (Verification) ---")
    
    # H_dense = mpo.to_dense()
    # logger.info(f"Dense Hamiltonian Shape: {H_dense.shape}")
    
    # # Check Ground State Energy
    # eigvals = np.linalg.eigvalsh(H_dense)
    # logger.info(f"Ground State Energy:   {eigvals[0]:.6f}")
    # logger.info(f"First Excited Energy:  {eigvals[1]:.6f}")

if __name__ == "__main__":
    
    # ------------------------------------------------------------------
    # 1. Define Physics Parameters
    # ------------------------------------------------------------------
    nsite = 8       # System size
    t = 1.0         # Hopping strength
    
    # Generate random potentials v_i
    np.random.seed(42)
    v = np.random.rand(nsite) * 0.5
    
    H = tight_binding(nsite, t, v)
    
    print(H)
    
