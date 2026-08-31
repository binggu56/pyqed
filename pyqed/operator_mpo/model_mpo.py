import logging
from typing import List, Union
import numpy as np

from .backend import (
    backend,
    npseed,
    randomseed,
    xpseed,
    get_git_commit_hash,
    OE_BACKEND
)

from .compiler import construct_symbolic_mpo, _terms_to_table, symbolic_mo_to_numeric_mo, swap_site
from .model import Model, HolsteinModel
from .operator import Op
from .utils import Quantity

logger = logging.getLogger(__name__)


class ModelMPO:
    """
    A lightweight MPO wrapper compatible with your PyTorch script.
    """

    # Static flag to ensure we only log the backend info once,
    # even if multiple MPOs are created.
    _backend_logged = False

    def __init__(self, model: Model = None, terms: Union[Op, List[Op]] = None, offset: Quantity = Quantity(0), algo="qr"):

        # Use Backend Setup (Hardware & Random number Seeds & 32/64 bits. Default: cpu+fixed seed+ 64 bits)
        if not ModelMPO._backend_logged:
            # select cpu/gpu. CuPy not avaivable in macOS so not tested yet
            if OE_BACKEND == "numpy":
                logger.info("Use NumPy as backend")
                logger.info(f"numpy random seed is {npseed}")
            else:
                logger.info("Use CuPy as backend")
                logger.info(f"cupy random seed is {xpseed}")

            logger.info(f"random seed is {randomseed}")
            logger.info(f"Git Commit Hash: {get_git_commit_hash()}")

            # select 32/64 bits. default is 64 bits.
            if backend.is_32bits:
                logger.info("use 32 bits")
            else:
                logger.info("use 64 bits")

            ModelMPO._backend_logged = True

        # Standard MPO Initialization
        self.matrices = []

        if model is None:
            return

        # Handle Offset (Energy shift)
        if not isinstance(offset, Quantity):
            offset = Quantity(offset)
        self.offset = offset.as_au()

        # Prepare Hamiltonian Terms
        if terms is None:
            terms = model.ham_terms
        elif isinstance(terms, Op):
            terms = [terms]

        if len(terms) == 0:
            raise ValueError("Terms contain nothing.")

        # Print out Operator Logs
        logger.info(f"# of operator terms: {len(terms)}")
        logger.info(f"symbolic mpo algorithm: {algo}")
        logger.info(f"Input operator terms: {len(terms)}")

        # Convert Terms to Lookup Table
        table, primary_ops, factor = _terms_to_table(model, terms, -self.offset)
        self.dtype = factor.dtype

        # Print out "After combination" Log, give maximum bond dimension
        logger.info(f"After combination of the same terms: {len(table)}")

        # Construct the exact MPO extracted from symbolic MPO construction using the bipartite graph theory
        self.symbolic_mpo, _, _, _, _, _ = \
            construct_symbolic_mpo(table, primary_ops, factor, algo=algo)

        self.model = model

        # Generate Numeric Matrices
        assert model.basis is not None
        for impo, mo in enumerate(self.symbolic_mpo):
            mo_mat = symbolic_mo_to_numeric_mo(model.basis[impo], mo, self.dtype)
            self.matrices.append(mo_mat)


    def to_dense(self, check_size=True):
        """
        Contract the MPO tensors into a full dense Hamiltonian matrix.
        Universal for all models (Fermion/Boson/Spin).
        This could be used for ensuring the constructed MPO is the wanted one, for example compare energy with Exact Diagonalization.
        """
        # 1. Safety Check: Estimate size before contracting
        # Total dimension = product of all physical dimensions
        # We look at the 2nd index (Physical Up) of each tensor
        dims = [m.shape[1] for m in self.matrices]
        total_dim = np.prod(dims)

        if check_size and total_dim > 20000:
            raise ValueError(f"MPO is too large to convert to dense! "
                             f"Dim={total_dim} > 20000. "
                             f"Set check_size=False to force.")

        # 2. Contraction Logic (Standard)
        # Start with a 1x1x1 tensor
        accum = np.ones((1, 1, 1), dtype=self.dtype)

        for W in self.matrices:
            # W shape: (Bond_Left, Phys_Up, Phys_Down, Bond_Right)
            # Accum shape: (Rows, Cols, Bond_Left)

            # Contract: Accum[..., Bond_Left] * W[Bond_Left, ...]
            temp = np.tensordot(accum, W, axes=([-1], [0]))

            # Current shape: (Rows_accum, Cols_accum, Phys_Up, Phys_Down, Bond_Right)
            # We want:       (Rows_accum * Phys_Up, Cols_accum * Phys_Down, Bond_Right)

            # Transpose to group Row indices and Col indices
            temp = np.transpose(temp, (0, 2, 1, 3, 4))

            r_old, p_up, c_old, p_down, b_right = temp.shape

            accum = temp.reshape(r_old * p_up, c_old * p_down, b_right)

        # 3. Squeeze to remove the dummy bonds (1 at start, 1 at end)
        return accum.squeeze()



    def to_dense_subspace(self, n_particles):
        """
        Generates the dense Hamiltonian matrix projected into the subspace
        with a fixed number of particles.

        Args:
            n_particles (int): The target number of electrons (e.g., 4 for LiH).

        Returns:
            np.ndarray: A square matrix of shape (Dim_Sub, Dim_Sub).
        """
        import itertools

        # 1. Get System Size
        L = len(self.matrices)

        # 2. Generate Basis States (Bitstrings with sum = n_particles)
        # We use 0 for Empty, 1 for Occupied (matching BasisSimpleElectron)
        # Result is shape (B, L)
        basis_tuples = [
            seq for seq in itertools.product([0, 1], repeat=L)
            if sum(seq) == n_particles
        ]

        if not basis_tuples:
            raise ValueError(f"No states found with {n_particles} particles!")

        # Convert to numpy for boolean indexing
        # Shape: (Basis_Size, L)
        basis = np.array(basis_tuples, dtype=int)
        dim_sub = len(basis)

        # 3. Initialize Accumulator
        # Shape: (Rows, Cols, Bond_Dim)
        # We start with the trivial left boundary (1.0)
        # Broadcasted to all pairs of basis states (Dim_Sub x Dim_Sub)
        accum = np.zeros((dim_sub, dim_sub, 1), dtype=self.dtype)
        accum[:, :, 0] = 1.0

        # 4. Sweep through the MPO sites
        for site_idx, W in enumerate(self.matrices):
            # W shape: (Bond_Left, Phys_Row, Phys_Col, Bond_Right)
            # Physical indices are 0 (Empty) and 1 (Occupied)

            # Slice W into the 4 possible operators:
            # <0|W|0>, <0|W|1>, <1|W|0>, <1|W|1>
            # Each slice is a matrix (Bond_Left, Bond_Right)
            W00 = W[:, 0, 0, :]
            W01 = W[:, 0, 1, :]
            W10 = W[:, 1, 0, :]
            W11 = W[:, 1, 1, :]

            # Get the physical state of the basis at this site
            # 'states' is a vector of length Dim_Sub (0s and 1s)
            states = basis[:, site_idx]

            # Find indices where state is 0 or 1
            idx0 = np.where(states == 0)[0]
            idx1 = np.where(states == 1)[0]

            # We need to update 'accum' for all 4 block combinations of (Row, Col)
            # We create a new accumulator for the next bond
            new_accum = np.zeros((dim_sub, dim_sub, W.shape[-1]), dtype=self.dtype)

            # Block 0->0: Row state is 0, Col state is 0
            # We take the subset of 'accum' corresponding to idx0 x idx0
            # And multiply by the W00 transition matrix
            if len(idx0) > 0:
                # Use broadcasting or tensordot.
                # accum[idx0][:, idx0] is (N0, N0, D_L)
                # W00 is (D_L, D_R)
                # Result is (N0, N0, D_R)
                new_accum[np.ix_(idx0, idx0)] += np.dot(accum[np.ix_(idx0, idx0)], W00)

            # Block 0->1
            if len(idx0) > 0 and len(idx1) > 0:
                new_accum[np.ix_(idx0, idx1)] += np.dot(accum[np.ix_(idx0, idx1)], W01)

            # Block 1->0
            if len(idx1) > 0 and len(idx0) > 0:
                new_accum[np.ix_(idx1, idx0)] += np.dot(accum[np.ix_(idx1, idx0)], W10)

            # Block 1->1
            if len(idx1) > 0:
                new_accum[np.ix_(idx1, idx1)] += np.dot(accum[np.ix_(idx1, idx1)], W11)

            accum = new_accum

        # 5. Squeeze the final dummy bond
        return accum.squeeze()

    # Compatibility Methods, MPO would be treated just like lists
    def __iter__(self):
        return iter(self.matrices)

    def __getitem__(self, item):
        return self.matrices[item]

    def __len__(self):
        return len(self.matrices)
