import numpy as np
import logging
import time
from pyqed import Molecule
from pyqed.mps.mps import DMRG
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo

# log info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


#  Fermionic Logic patch adding JW chain
def get_jw_term_robust(op_str_list, indices, factor):
    """
    Constructs a fermionic term with explicit Jordan-Wigner strings (sigma_z)
    and correct sign handling (parity).
    """
    # 1. Canonical Sort: Sort operators by site index
    chain = list(zip(indices, op_str_list))
    n = len(chain)
    swaps = 0
    for i in range(n):
        for j in range(0, n-i-1):
            if chain[j][0] > chain[j+1][0]:
                chain[j], chain[j+1] = chain[j+1], chain[j]
                swaps += 1

    sorted_indices = [x[0] for x in chain]
    sorted_ops = [x[1] for x in chain]

    final_indices = []
    final_ops_str = []
    parity = 0
    extra_sign = 1

    # 2. Insert sigma_z filling (Jordan-Wigner String)
    for k in range(n):
        site = sorted_indices[k]
        op_sym = sorted_ops[k]

        # Fill gap between previous site and current site with Z
        if k > 0:
            prev_site = sorted_indices[k-1]
            if parity % 2 == 1:
                for z_site in range(prev_site + 1, site):
                    final_indices.append(z_site)
                    final_ops_str.append("sigma_z")

        # 3. Handle Creation/Annihilation Phase
        # If we are applying 'a' and there are an odd number of operators to the right, flip sign
        ops_to_right = n - 1 - k
        if (op_sym == "a") and (ops_to_right % 2 == 1):
            extra_sign *= -1

        final_indices.append(site)
        final_ops_str.append(op_sym)
        parity += 1

    final_op_string = " ".join(final_ops_str)
    return Op(final_op_string, final_indices, factor=factor * ((-1) ** swaps) * extra_sign)

# initial guess from hf but with added noise to prevenr stuck in hf product state, it happens sometimes
def get_noisy_hf_guess(n_elec, n_spin, noise=1e-3):
    """
    Creates an MPS guess based on filling the first N_elec spin-orbitals,
    but adds small noise to prevent the solver from getting stuck in the HF state.
    """
    d = 2
    mps_guess = []
    filled_count = 0

    for i in range(n_spin):
        vec = np.zeros((d, 1, 1))
        if filled_count < n_elec:
            vec[1, 0, 0] = 1.0; filled_count += 1
        else:
            vec[0, 0, 0] = 1.0

        # Add Noise
        vec += (np.random.rand(d, 1, 1) - 0.5) * noise
        vec /= np.linalg.norm(vec)
        mps_guess.append(vec)

    return mps_guess


#  dmrg mpo generator
def qc_dmrg_mpo(mf):

    # 1. Extract Integrals & dims
    mol = mf.mol
    h1 = mf.get_hcore_mo()
    eri = mf.get_eri_mo(notation='chem') # (pq|rs)

    n_spatial = mol.nao
    n_spin = 2 * n_spatial
    print(f"  System: {n_spatial} spatial orbitals, {n_spin} spin-orbitals")

    # 2. Build Hamiltonian (Using Robust JW Builder)
    print("  Building Hamiltonian MPO...")
    ham_terms = []
    cutoff = 1e-10

    # --- One-Body Terms: h_pq a+_p a_q ---
    for p in range(n_spatial):
        for q in range(n_spatial):
            val = h1[p, q]
            if abs(val) > cutoff:
                # Spin Up (Indices 2p, 2q)
                ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*p, 2*q], val))
                # Spin Down (Indices 2p+1, 2q+1)
                ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*p+1, 2*q+1], val))

    # --- Two-Body Terms: 0.5 * (pq|rs) a+_p a+_r a_s a_q ---
    for p in range(n_spatial):
        for q in range(n_spatial):
            for r in range(n_spatial):
                for s in range(n_spatial):
                    val = 0.5 * eri[p, q, r, s]
                    if abs(val) < cutoff: continue

                    # p,r creation; s,q annihilation

                    # Same Spin (Pauli Exclusion p!=r)
                    if p != r and s != q:
                        # Up-Up
                        ham_terms.append(get_jw_term_robust(
                            [r"a^\dagger", r"a^\dagger", "a", "a"],
                            [2*p, 2*r, 2*s, 2*q], val
                        ))
                        # Dn-Dn
                        ham_terms.append(get_jw_term_robust(
                            [r"a^\dagger", r"a^\dagger", "a", "a"],
                            [2*p+1, 2*r+1, 2*s+1, 2*q+1], val
                        ))

                    # Mixed Spin (No Pauli restriction on spatial indices)
                    # Up-Dn (p Up, r Dn, s Dn, q Up)
                    ham_terms.append(get_jw_term_robust(
                        [r"a^\dagger", r"a^\dagger", "a", "a"],
                        [2*p, 2*r+1, 2*s+1, 2*q], val
                    ))
                    # Dn-Up (p Dn, r Up, s Up, q Dn)
                    ham_terms.append(get_jw_term_robust(
                        [r"a^\dagger", r"a^\dagger", "a", "a"],
                        [2*p+1, 2*r, 2*s, 2*q+1], val
                    ))

    # 3. Generate MPO
    basis_sites = [BasisSimpleElectron(i) for i in range(n_spin)]
    model = Model(basis=basis_sites, ham_terms=ham_terms)
    mpo = Mpo(model, algo="qr")

    # get it transposed for solver in PyQED: (L, R, P, P) -> (L, P, R, P)
    mpo_dmrg = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]


    return mpo_dmrg



# main
if __name__ == "__main__":


    mol = Molecule(atom = [
        ['H' , (0. , 0. , 0)],
        ['H' , (0. , 0. , 4)]])
        # ['H' , (0. , 0. , 0)],
        # ['Li' , (0. , 0. , 1.4)]])

    mol.basis = '6311g'
    mol.build()
    mf = mol.RHF().run()
    print(f"RHF Energy: {mf.e_tot:.8f} Ha")

    # DMRG Parameters
    BOND_DIM = 16
    N_SWEEPS = 20
    Initial_guess_NOISE    = 1e-3

    # get mpo and mps initial guess
    mpo_dmrg = qc_dmrg_mpo(mf)
    mps_guess = get_noisy_hf_guess(mol.nelec, 2*mol.nao, noise=Initial_guess_NOISE)


    t0 = time.time()
    # run dmrg!
    print(f"  Starting Sweeps (D={BOND_DIM})...")
    solver = DMRG(mpo_dmrg, D=BOND_DIM, init_guess=mps_guess)
    solver.run()

    # 6. Report result
    e_dmrg_total = solver.e_tot + mf.energy_nuc()

    print("RESULTS")
    print(f"  RHF Energy:         {mf.e_tot:.8f} Ha")
    print(f"  DMRG Total Energy:  {e_dmrg_total:.8f} Ha")
    print(f"  Correlation Energy: {e_dmrg_total - mf.e_tot:.8f} Ha")
    print(f"  Time:               {time.time()-t0:.2f} s")




    # ###### debug use, useful if wish to compare with full ci (not sure where the full ci in pyqed is)
    # from pyscf import gto, scf, ao2mo, fci
    # mol = gto.M(atom='H 0 0 0; H 0 0 4', unit='Bohr', basis='6-31g', verbose=0)
    # mol = gto.M(atom='H 0 0 0; Li 0 0 1.4', unit='Bohr', basis='6-31g', verbose=0)
    # mf = scf.RHF(mol).run()
    # fci_energy = fci.FCI(mol, mf.mo_coeff).kernel()[0]
    # print("-" * 60)
    # print(f"PySCF Full CI:    {fci_energy:.10f} Ha")