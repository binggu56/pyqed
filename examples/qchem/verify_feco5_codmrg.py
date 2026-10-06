"""Verify saved Fe(CO)5 CO orbitals and reduced MPS without another solve."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import numpy as np
from pyqed.mps.nonabelian._su2_kernel import SU2MovingEnvironment
from pyqed.optimize import OrbitalContractionPlan
from pyqed.qchem.dmrg import QCDMRG
from pyqed.qchem.mcscf.casci import _get_mf_cholesky_factors, mo_pair_factors
from pyqed.qchem.mcscf.cocas import _gn_details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--orbitals', type=Path)
    parser.add_argument('--expected-energy', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directory = args.directory
    with (directory/'rhf.pkl').open('rb') as handle:
        mf = pickle.load(handle)
    with (args.state or directory/'mps_state_final.pkl').open('rb') as handle:
        state = pickle.load(handle)
    coefficients = np.load(args.orbitals or directory/'orbitals_final.npy')
    expected = args.expected_energy
    if expected is None:
        expected = json.loads((directory/'result.json').read_text())['energy_hartree']

    # NPDMs depend only on the installed reduced MPS and local fermion operators.
    # The zero integral owner is used solely to contract densities; no Hamiltonian
    # solve or determinant-space expansion is performed in this verification.
    owner = SU2MovingEnvironment(np.zeros((12, 12)), np.zeros((12,)*4), 12, two_s=0)
    solver = QCDMRG(mf, ncas=12, nelecas=12, D=100, symmetry='su2', integral_backend='ri')
    solver.dmrg = SimpleNamespace(ground_state=state, states=[state])
    solver._su2_runtime = SimpleNamespace(moving_environment=owner)
    dm1, dm2 = solver.make_rdm12(0, spatial=True, with_core=True)
    core = solver.ncore
    active_dm1, active_dm2 = dm1[core:, core:], dm2[core:, core:, core:, core:]
    overlap = mf.get_ovlp()
    reference = mf.mo_coeff
    orbitals = reference.T @ overlap @ coefficients
    h1 = mf.get_hcore_mo()
    factors = mo_pair_factors(_get_mf_cholesky_factors(mf), reference)
    plan = OrbitalContractionPlan(h1, factors, orbitals.shape, dm1.shape, dm2.shape, ncore=core)
    energy = float(plan.energy(orbitals, h1, factors, dm1, dm2) + mf.mol.energy_nuc())
    gradient = _gn_details(orbitals, h1, factors, dm1, dm2, plan,
                           ncore=core, ncas=12, active_active=True)
    report = dict(
        energy_hartree=energy, reported_energy_hartree=expected,
        energy_difference_hartree=energy-expected, orbital_gradient=gradient,
        orbital_gradient_converged=gradient['total'] < 1e-4,
        orbital_orthogonality_max_error=float(np.max(np.abs(coefficients.T @ overlap @ coefficients-np.eye(coefficients.shape[1])))),
        active_electrons=float(np.trace(active_dm1)),
        active_pair_trace=float(np.einsum('pprr->', active_dm2)),
        rdm_contraction_max_error=float(np.max(np.abs(np.einsum('pqrr->pq', active_dm2)-11*active_dm1))),
        maximum_per_sector_dimension=max(int(basis.dims[sector]) for basis in
            (state.bond_basis(bond) for bond in range(len(state)-1)) for sector in basis.sectors),
        target_sector=repr(state.target_sector), npdm=solver.spatial_rdm_diagnostics,
    )
    output = args.output or directory/'verification.json'
    output.write_text(json.dumps(report, indent=2))
    np.savez(output.with_suffix('.npz'), rdm1_active=active_dm1, rdm2_active=active_dm2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
