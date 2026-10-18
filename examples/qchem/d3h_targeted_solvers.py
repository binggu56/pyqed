"""Compare full D3h multiplets in targeted CI, SU2-LETTA, and DMRG.

CI is the explicit small-system reference. Tensor-network paths use reduced
operators and a structural spatial group leg, without determinant projection.
Run from the repository root with PYTHONPATH=. and BLAS threads at one.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.casci import CASCI
from pyqed.qchem.mcscf.direct_ci import transform_active_space_spatial_integrals
from pyqed.qchem.dmrg.dmrg import DMRG
from pyqed.qchem.symmetry import mo_symmetry_representations
from pyqed.letta import SU2LETTA
from pyqed.mps.nonabelian.environment import contract_chain_transition
from pyqed.mps.nonabelian.orbital_transform import apply_spatial_orbital_transform


def covariance_errors(solver):
    if hasattr(solver, 'symmetry_diagnostics'):
        return dict(solver.symmetry_diagnostics(), history=[
            {'cycle': item['cycle'], 'energy': item['energy']} for item in solver.history])
    action = solver.spatial_symmetry
    states = solver.states
    gram = np.array([[contract_chain_transition(a.tensors, solver.identity, b.tensors)
                      for b in states] for a in states])
    orthogonality = float(np.max(np.abs(gram - np.eye(len(states)))))
    covariance = 0.0
    for op, matrix in zip(action.names, action.matrices):
        for row, state in enumerate(states):
            transformed = apply_spatial_orbital_transform(state, matrix, inverse=False,
                cutoff=0, max_bond=None, orbital_blocks=action.orbital_blocks)
            values = np.array([contract_chain_transition(a.tensors, solver.identity, transformed.tensors)
                               for a in states])
            covariance = max(covariance, float(np.max(np.abs(values - action.irrep_matrices[op][:, row]))))
    row_energies = [float(contract_chain_transition(s.tensors, solver.mpo, s.tensors).real + solver.ecore)
                    for s in states]
    return {'orthogonality_error': orthogonality, 'covariance_error': covariance,
            'partner_energies_hartree': row_energies,
            'history': [{'cycle': item['cycle'], 'energy': item['energy']} for item in solver.history]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('/private/tmp/pyqed-d3h-targeted'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mol = Molecule(atom='H 1 0 0; H -0.5 0.8660254037844386 0; H -0.5 -0.8660254037844386 0',
                   unit='bohr', basis='sto-3g', charge=1, symmetry='D3h').build()
    mf = RHF(mol).run(verbose=0)
    h1, eri, ecore = transform_active_space_spatial_integrals(mf, mf.mo_coeff, ncas=3, ncore=0)
    reps = mo_symmetry_representations(mol, mf.mo_coeff)
    results = []
    for irrep, spin in (("A1'", 0), ("E'", 0), ("A2'", 2)):
        reference = CASCI(mf, ncas=3, nelecas=2, multiplicity=spin+1).run(wfnsym=irrep)
        dmrg = DMRG(mf, ncas=3, nelecas=2, D=1, symmetry='su2', spin=spin, tol=1e-10, verbose=0)
        ds = dmrg.run(wfnsym=irrep, nsweeps=4)
        ls = SU2LETTA.from_integrals(h1, eri, nelec=2, spin=spin, D=1, graph='nn',
            ecore=ecore, seed=11, orbital_representations=reps, wfnsym=irrep).run(nsweeps=4, tol=1e-10)
        row = {'irrep': irrep, 'two_s': spin, 'ci_energy_hartree': reference.e_tot.tolist(),
               'dmrg': covariance_errors(ds), 'letta': covariance_errors(ls)}
        results.append(row)
        print(json.dumps({'irrep': irrep, 'ci': reference.e_tot.tolist(),
                          'dmrg': ds.e_tot.tolist(), 'letta': ls.e_tot.tolist()}), flush=True)
        ds.close()
        ls.close()
    data = {'molecule': 'H3+', 'basis': 'STO-3G', 'geometry_bohr': mol.atom_coords().tolist(),
            'seed_D': 1, 'results': results}
    (args.output_dir / 'h3plus_d3h_targeted_solvers.json').write_text(json.dumps(data, indent=2))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    methods = [('ci', '#3d4852', -0.16), ('dmrg', '#2472ac', 0), ('letta', '#d47a26', 0.16)]
    for method, color, shift in methods:
        for i, row in enumerate(results):
            energy = row['ci_energy_hartree'] if method == 'ci' else row[method]['partner_energies_hartree']
            x = i + shift + np.linspace(-0.025, 0.025, len(energy))
            axes[0].scatter(x, energy, c=color, s=45, marker={'ci':'_', 'dmrg':'o', 'letta':'x'}[method],
                            label={'ci':'CI reference', 'dmrg':'DMRG, D=1', 'letta':'LETTA, D=1'}[method] if i == 0 else None)
    for method, color, _ in methods[1:]:
        errors = [max(1e-16, row[method]['covariance_error']) for row in results]
        axes[1].plot(range(3), errors, 'o-', color=color, label=method.upper())
    axes[0].set(ylabel='Total energy (hartree)', title='Targeted spatial multiplets')
    axes[1].set(ylabel='Maximum partner covariance error', yscale='log', title='All twelve group operations')
    for ax in axes:
        ax.set(xticks=range(3), xticklabels=["$A_1'$ (singlet)", "$E'$ (singlet)", "$A_2'$ (triplet)"])
        ax.grid(axis='y', alpha=0.18)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(r'$\mathrm{H}_3^+$ / STO-3G: full $D_{3h}$ targeting', fontsize=13)
    for extension in ('png', 'pdf'):
        fig.savefig(args.output_dir / f'h3plus_d3h_targeted_solvers.{extension}', dpi=180)
    plt.close(fig)
    print('Outputs:', args.output_dir)


if __name__ == '__main__':
    main()
