"""Bounded ISD/L-BFGS inner comparison and one reduced SU(2) macro evaluation."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('PYQED_SU2_DISABLE_OUTPUT_FUSION', '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/feco5-matplotlib')
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
import numpy as np
from pyqed.optimize import OrbitalContractionPlan, minimize, orbital_gap_preconditioner
from pyqed.qchem.dmrg import QCDMRG
from pyqed.qchem.mcscf.casci import _get_mf_cholesky_factors, mo_pair_factors
from pyqed.qchem.mcscf.cocas import _physical_orbital_gradient, _gn_details
from pyqed.mps.nonabelian._su2_kernel import SU2MovingEnvironment


def plot(output, reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    baseline = reports['ISD']['history'][0]['energy']
    for method, report in reports.items():
        rows = report['history']
        for axis, xkey in ((axes[0], 'iteration'), (axes[1], 'seconds')):
            axis.plot([r[xkey] for r in rows], [(r['energy']-baseline)*1e6 for r in rows], label=method)
        axes[2].semilogy([r['iteration'] for r in rows], [r['physical_gradient'] for r in rows], label=method)
    axes[0].set(xlabel='Inner iteration', ylabel='Frozen-RDM energy change (µHartree)')
    axes[1].set(xlabel='Inner wall time (s)', ylabel='Frozen-RDM energy change (µHartree)')
    axes[2].set(xlabel='Inner iteration', ylabel='Physical orbital gradient norm')
    for axis in axes:
        axis.legend()
    fig.suptitle('Fe(CO)₅ · def2-SVP · CAS(12,12) · D=100 · bounded inner comparison')
    fig.savefig(output/'feco5_isd_quick_test.png', dpi=180)
    fig.savefig(output/'feco5_isd_quick_test.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Paired checkpoint stem, without suffix.')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--gap-floor', type=float, help='Also test ISD with positive orbital-gap scaling.')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source_hashes = {str(Path(module.__file__).resolve()): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                     for module in (__import__('pyqed.optimize', fromlist=['']), __import__('pyqed.qchem.mcscf.cocas', fromlist=['']))}
    with args.reference.open('rb') as handle:
        mf = pickle.load(handle)
    with Path(str(args.checkpoint)+'_state.pkl').open('rb') as handle:
        state = pickle.load(handle)
    coefficients = np.load(str(args.checkpoint)+'_orbitals.npy')
    solver = QCDMRG(mf, ncas=12, nelecas=12, D=100, symmetry='su2', integral_backend='ri', init_guess=state)
    # Reduced NPDM operator owner only; the zero Hamiltonian is never solved.
    owner = SU2MovingEnvironment(np.zeros((12,12)), np.zeros((12,)*4), 12, two_s=0)
    solver.dmrg = SimpleNamespace(ground_state=state, states=[state])
    solver._su2_runtime = SimpleNamespace(moving_environment=owner)
    d1, d2 = solver.make_rdm12(0, spatial=True, with_core=True)
    core = solver.ncore
    h1 = mf.get_hcore_mo()
    factors = mo_pair_factors(_get_mf_cholesky_factors(mf), mf.mo_coeff)
    u0 = mf.mo_coeff.T @ mf.get_ovlp() @ coefficients
    plan = OrbitalContractionPlan(h1, factors, u0.shape, d1.shape, d2.shape, ncore=core)
    initial_energy = float(plan.energy(u0, h1, factors, d1, d2)+mf.mol.energy_nuc())
    reports, candidates = {}, {}
    methods = ('ISD', 'LBFGS') if args.gap_floor is None else ('ISD', 'LBFGS', 'ISD-gap')
    preconditioner = (None if args.gap_floor is None else orbital_gap_preconditioner(
        mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff, args.gap_floor))
    for method in methods:
        rows = []
        inner_start = time.perf_counter()

        def recorded_gradient(u, *values):
            g = plan.gradient(u, *values)
            rows.append(dict(iteration=len(rows), seconds=time.perf_counter()-inner_start,
                             energy=float(plan.energy(u, *values)),
                             physical_gradient=float(np.linalg.norm(_physical_orbital_gradient(u, g, core, 12, active_active=True)))))
            return g

        candidate, energy = minimize(plan.energy, u0, args=(h1, factors, d1, d2),
                                    gradient_fn=recorded_gradient, algorithm='ISD' if method=='ISD-gap' else method,
                                    isd_preconditioner=preconditioner if method=='ISD-gap' else None,
                                    tau=1., epsilon=1e-5, max_iterations=args.steps, history_size=30)
        candidates[method] = candidate
        reports[method] = dict(history=rows, seconds=time.perf_counter()-inner_start,
                               iterations=len(rows)-1, frozen_energy_hartree=float(energy+mf.mol.energy_nuc()))
        print(method, {k:v for k,v in reports[method].items() if k!='history'}, flush=True)
    plot(output, reports)
    trial_method = 'ISD' if args.gap_floor is None else 'ISD-gap'
    candidate_coefficients = mf.mo_coeff @ candidates[trial_method]
    np.save(output/'orbitals_isd_trial.npy', candidate_coefficients)
    trial = QCDMRG(mf, ncas=12, nelecas=12, D=100, symmetry='su2', integral_backend='ri', init_guess=state, verbose=2)
    trial.spatial_rdm2_algorithm = 'npdm'
    solve_start = time.perf_counter()
    trial.run(mo_coeff=candidate_coefficients, nsweeps=4, conv_tol=1e-7,
              su2_kernel_backend='cpp', max_bond_mode='per_sector', cutoff=1e-7,
              davidson_tol=1e-10, davidson_max_iter=150,
              local_solver_kwargs=dict(max_space=96, tol_residual=1e-9, workspace_budget_bytes=512*1024**2))
    solve_seconds = time.perf_counter()-solve_start
    final_d1, final_d2 = trial.make_rdm12(0, spatial=True, with_core=True)
    gradient = _gn_details(candidates[trial_method], h1, factors, final_d1, final_d2, plan,
                           ncore=core, ncas=12, active_active=True)
    final_state = trial.export_ground_state()
    with (output/'mps_state_isd_trial.pkl').open('wb') as handle:
        pickle.dump(final_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez(output/'rdm_active_initial.npz', rdm1=d1[core:,core:], rdm2=d2[core:,core:,core:,core:])
    result = dict(checkpoint=str(args.checkpoint), steps=args.steps, trial_optimizer=trial_method,
                  gap_floor=args.gap_floor, initial_energy_hartree=initial_energy,
                  inner=reports, isd_trial_energy_hartree=float(trial.e_tot),
                  isd_trial_energy_change_hartree=float(trial.e_tot)-initial_energy,
                  isd_trial_gradient=gradient, dmrg_converged=bool(trial.dmrg.converged),
                  dmrg_seconds=solve_seconds, total_seconds=time.perf_counter()-started,
                  full_co_convergence_test=False,
                  source_hashes=source_hashes)
    (output/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='inner'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
