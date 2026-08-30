"""
ModelMPO Example: Bosonic Tight-Binding Chain

Hamiltonian Definition:
-----------------------
The Hamiltonian being simulated is a 1D Bosonic chain with nearest-neighbor hopping:

    H = Σ [ω * n_i]  +  Σ [J * (b†_i * b_{i+1} + b_i * b†_{i+1})]
         i               i

Where:
  - i       : Site index
  - ω (omega): On-site oscillator frequency
  - J (coup): Hopping/Coupling strength
  - n_i     : Number operator (b†_i * b_i)
  - b†, b   : Creation and Annihilation operators
"""
import logging
from pyqed.operator_mpo.model import Model
from pyqed.operator_mpo.operator import Op
from pyqed.operator_mpo.basis import BasisSHO
from pyqed.operator_mpo.model_mpo import ModelMPO
import numpy as np
# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_auto_mpo_example(nsite, nbas, omega, coupling):


    logger.info(f"--- Building MPO for Bosonic Chain (N={nsite}) ---")

    # ------------------------------------------------------------------
    # 2. Build Hamiltonian Terms (Symbolic)
    # ------------------------------------------------------------------
    ham_terms = []

    # Term A: On-site Energy (omega * n)
    # Op(operator_string, site_index, factor)
    for i in range(nsite):
        ham_terms.append(Op(r"b^\dagger b", i, factor=omega))

    # Term B: Interaction / Hopping (b^dag_i * b_{i+1} + h.c.)
    # Note: We multiply Op objects to create multi-site terms
    for i in range(nsite - 1):
        # Forward hopping
        term_fwd = Op(r"b^\dagger", i) * Op("b", i+1) * coupling
        ham_terms.append(term_fwd)

        # Backward hopping (Hermitian conjugate)
        term_bwd = Op("b", i) * Op(r"b^\dagger", i+1) * coupling
        ham_terms.append(term_bwd)

    # ------------------------------------------------------------------
    # 3. Initialize Basis and Model
    # ------------------------------------------------------------------
    # Create the local basis for each site (Simple Harmonic Oscillator)
    basis = [BasisSHO(dof=i, omega=omega, nbas=nbas) for i in range(nsite)]

    # The Model class bundles the basis and the symbolic terms
    model = Model(basis=basis, ham_terms=ham_terms)

    # ------------------------------------------------------------------
    # 4. Generate MPO, which is automatic.
    # ------------------------------------------------------------------
    logger.info("Generating MPO from symbolic model...")

    # create the MPO, and using QR decomposition
    mpo = ModelMPO(model, algo="qr")

    # logger.info("MPO Successfully Created.")

    # # ------------------------------------------------------------------
    # # 5. DISCUSSION: How to USE the MPO
    # # ------------------------------------------------------------------
    # # A. IT BEHAVES LIKE A LIST
    # # The 'mpo' object is iterable. You can access individual tensors by index.
    # # Each tensor 'W' has 4 indices: (Bond_Left, Phys_Row, Phys_Col, Bond_Right)
    # # ------------------------------------------------------------------
    # for site_idx, W in enumerate(mpo):
    #     logger.info(f"Site {site_idx}: Tensor Shape {W.shape}")

    # # B. DEBUGGING / SMALL SYSTEM VERIFICATION
    # # If the system is small, you can convert the
    # # MPO back to a dense matrix to verify energy eigenvalues or inspect terms.
    # # ------------------------------------------------------------------
    # logger.info("\n--- Converting to Dense Matrix (Verification) ---")

    # # ModelMPO.to_dense() contracts all tensors: W0 - W1 - W2 ...
    # H_dense = mpo.to_dense()
    # logger.info(f"Dense Hamiltonian Shape: {H_dense.shape}")

    # # Simple check: Calculate Ground State Energy
    # eigvals = np.linalg.eigvalsh(H_dense)
    # logger.info(f"Ground State Energy:   {eigvals[0]:.6f}")
    # logger.info(f"First Excited Energy:  {eigvals[1]:.6f}")

    return mpo



if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1. Define Physics Parameters
    # ------------------------------------------------------------------
    nsite = 4         # Number of sites
    nbas = 4          # Max number of bosons per site (dimension of local space)
    omega = 1.0       # Frequency
    coupling = 0.2    # Hopping strength

    run_auto_mpo_example(nsite, nbas, omega, coupling)
