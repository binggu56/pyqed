"""Paired RI SA6 singlet CAS(4,4) memory regression, from a shared RHF snapshot.

The dimer is an idealized cofacial pentacene pair, 3.5 angstrom separation
and 1.5 angstrom long-axis slip. It is a software benchmark, not an optimized
geometry or a basis-converged prediction. Run heavy phases through Slurm.
Each solve must use its own process and the intended source via PYTHONPATH.
"""
import argparse
import hashlib
import json
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, default=lambda x: x.tolist()
        if isinstance(x, np.ndarray) else str(x)) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(system):
    if system == 'small':
        return '; '.join(f'H {i*.9} {(.07 if i % 2 else 0)} 0' for i in range(6))
    from benchmark_gw_tda_memory import geometry as acene
    monomer = acene(5)
    second = []
    for atom in monomer.split(';'):
        symbol, x, y, z = atom.split()
        second.append(f'{symbol} {float(x)+1.5} {y} {float(z)+3.5}')
    return monomer + '; ' + '; '.join(second)


def molecule(config):
    from pyqed.qchem import Molecule
    return Molecule(atom=config['atom'], basis=config['basis'],
        unit='angstrom').build(eri='ri', auxbasis=config['auxbasis'])


def prepare(args):
    config = dict(system=args.system, atom=geometry(args.system), basis=args.basis,
        auxbasis='def2-universal-jkfit', ncas=4, nelecas=4, nstates=6,
        multiplicity=1, weights=[1/6]*6, conv_tol=1e-7,
        conv_tol_grad=1e-4, conv_tol_grad_relaxed=1e-4, max_cycle=50,
        coupling='qn', optimizer='AH', ah_hessian='analytic',
        ci_method='ci',
        integral_route='full_factor', hessian_route='cblas_workspace')
    args.data.mkdir(parents=True, exist_ok=True)
    if (args.data/'input.json').exists():
        raise FileExistsError('Use a fresh data directory to preserve provenance')
    started = time.perf_counter()
    mol = molecule(config)
    mf = mol.RHF(verbose=1).run(tol=1e-10, stability_analysis=False)
    assert mf.converged
    mf.save(args.data/'rhf.npz')
    config.update(nmo=mf.nmo, nelec=mol.nelec, rhf_sha256=sha(args.data/'rhf.npz'),
        preparation_seconds=time.perf_counter()-started)
    dump(args.data/'input.json', config)


def solve(args):
    import psutil
    from pyqed.qchem.mcscf import casscf, orbopt
    from pyqed.qchem import _casscf_cpp
    np.random.seed(20261028)
    caps = _casscf_cpp.native_capabilities()
    assert caps['cblas'] and caps['factor_hessian_workspace'], caps
    config = json.loads((args.data/'input.json').read_text())
    assert sha(args.data/'rhf.npz') == config['rhf_sha256']
    started = time.perf_counter()
    mol = molecule(config)
    mf = mol.RHF()
    with np.load(args.data/'rhf.npz', allow_pickle=False) as saved:
        for name in ('mo_coeff', 'mo_energy', 'mo_occ'):
            setattr(mf, name, saved[name].copy())
        mf.e_tot = float(saved['e_tot'])
    mf.converged = True
    mf.e_nuc = mol.energy_nuc()
    mf.eri_factors = mol.eri_factors
    mf.cholesky_jk = True
    mf.dm = (mf.mo_coeff * mf.mo_occ) @ mf.mo_coeff.T
    names = ('ncas', 'nelecas', 'multiplicity', 'conv_tol', 'conv_tol_grad',
        'conv_tol_grad_relaxed', 'max_cycle', 'coupling', 'optimizer', 'ah_hessian', 'ci_method')
    mc = casscf.SecondOrderCASSCF(mf, **{name: config[name] for name in names}, verbose=1)
    mc.state_average(config['weights'])
    # Exercise the reported full-factor/workspace route in both versions,
    # including the small reference which otherwise precontracts its Hessian.
    mc._factor_block_source = lambda *args: None
    casscf.create_factor_response_action = lambda *args: None
    # Count actual CBLAS workspace applications; fail on any NumPy fallback.
    counts = dict(created=0, applied=0)
    workspace_details = []
    create, apply = orbopt.create_factor_hessian_workspace, orbopt.apply_factor_hessian_workspace
    def checked_create(*a, **kw):
        result = create(*a, **kw)
        assert result is not None, 'Hessian workspace creation failed'
        counts['created'] += 1
        if hasattr(_casscf_cpp,'factor_hessian_workspace_info'):
            info = _casscf_cpp.factor_hessian_workspace_info(result)
            workspace_details.append(info)
            if args.require_packed:
                assert info.get('packed_pairs'), 'Packed CBLAS factor storage was not used'
        return result
    def checked_apply(*a, **kw):
        result = apply(*a, **kw)
        assert result is not None, 'Hessian workspace application fell back'
        counts['applied'] += 1
        return result
    casscf.create_factor_hessian_workspace = checked_create
    casscf.apply_factor_hessian_workspace = checked_apply
    orbopt.create_factor_hessian_workspace = checked_create
    orbopt.apply_factor_hessian_workspace = checked_apply
    if args.require_packed:
        from pyqed.qchem.basis import PackedRIFactors
        def forbidden_expansion(*a,**kw):
            raise AssertionError('Full expansion of packed factors during CASSCF')
        PackedRIFactors.__array__ = forbidden_expansion
    samples, stop = [], threading.Event()
    process = psutil.Process()
    solve_start = time.perf_counter()
    def sample():
        while not stop.is_set():
            samples.append([time.perf_counter()-solve_start, process.memory_info().rss/2**30])
            stop.wait(.1)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    failure = None
    try:
        mc.run(nstates=6, mo_coeff=mf.mo_coeff.copy())
        assert counts['applied'] > 0, 'Requested factor Hessian path was not exercised'
        np.testing.assert_allclose(mc.spin_square(), 0, atol=1e-8, rtol=0)
    except Exception as error:
        failure = repr(error)
        raise
    finally:
        stop.set()
        thread.join()
        report = dict(variant=args.variant, config=config, source=str(casscf.__file__),
            source_sha256=sha(casscf.__file__), script_sha256=sha(__file__),
            extension_sha256=sha(_casscf_cpp.__file__), capabilities=caps,
            workspace_calls=counts, workspace_info=workspace_details,
            converged=bool(mc.converged), failure=failure,
            casscf_seconds=time.perf_counter()-solve_start,
            total_seconds=time.perf_counter()-started,
            peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (2**30 if sys.platform == 'darwin' else 2**20),
            sampled_casscf_peak_rss_gib=max((s[1] for s in samples), default=0),
            rss_samples=samples, history=mc.history, micro_history=mc.micro_history,
            energies=getattr(mc, 'e_tot', None))
        if mc.converged:
            report['spin_square'] = mc.spin_square()
        dump(args.data/f'{args.variant}.json', report)


def compare(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads((args.data/f'{v}.json').read_text()) for v in ('old', 'new')]
    for row in rows:
        assert row['converged'] and row['failure'] is None
        assert row['workspace_calls']['applied'] > 0
        np.testing.assert_allclose(row['spin_square'], 0, atol=1e-8, rtol=0)
    assert rows[0]['config'] == rows[1]['config']
    assert rows[0]['extension_sha256'] == rows[1]['extension_sha256']
    delta = np.asarray(rows[1]['energies'])-rows[0]['energies']
    report = dict(max_energy_change_hartree=float(max(abs(delta))),
        energy_changes_hartree=delta, old_rss_gib=rows[0]['peak_process_rss_gib'],
        new_rss_gib=rows[1]['peak_process_rss_gib'],
        old_casscf_seconds=rows[0]['casscf_seconds'], new_casscf_seconds=rows[1]['casscf_seconds'])
    dump(args.data/'comparison.json', report)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    for row in rows:
        t, rss = np.asarray(row['rss_samples']).T
        axes[0].plot(t, rss, label=row['variant'])
    axes[0].set(xlabel='SA-CASSCF elapsed (s)', ylabel='Process RSS (GiB)')
    axes[0].legend()
    axes[1].bar(['old', 'new'], [r['casscf_seconds'] for r in rows])
    axes[1].set(ylabel='SA-CASSCF time (s)')
    axes[2].plot(range(1, 7), delta, 'o-')
    axes[2].set(xlabel='Singlet root', ylabel='New − old energy (hartree)')
    fig.suptitle(f"{rows[0]['config']['system']}: RI SA6 singlet CAS(4e,4o), {rows[0]['config']['basis']}")
    fig.savefig(args.data/'casscf_memory_comparison.png', dpi=160)
    np.testing.assert_allclose(delta, 0, atol=1e-7, rtol=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'solve', 'compare'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--system', choices=['small', 'pentacene-dimer'], default='pentacene-dimer')
    parser.add_argument('--basis', default='sto-3g')
    parser.add_argument('--variant', choices=['old', 'new'], default='new')
    parser.add_argument('--require-packed', action='store_true')
    args = parser.parse_args()
    globals()[args.phase](args)
