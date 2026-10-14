"""Profile complete contour screening and HOMO/LUMO self-energy at fixed controls.

Water/cc-pVDZ provides an end-to-end numerical check; larger synthetic RI
matrices isolate contraction scaling, not molecular runtime. BLAS=1 and
outer workers=1. Use --baseline to compare saved builds in the same process.
"""
import argparse
import contextlib
import cProfile
import importlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from benchmarks.benchmark_bse_molecules import CASES
from pyqed.qchem import Molecule
from pyqed.gw.gw import GW

c = importlib.import_module('pyqed.gw.contour')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--baseline', type=Path, help='Saved baseline extension for interleaved comparison.')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    kernels = {'optimized': c._contour_contractions}
    if args.baseline:
        spec = importlib.util.spec_from_file_location('_baseline._contour_contractions', args.baseline)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        kernels = {'baseline': baseline, **kernels}
    configurations = [(variant, threads) for threads in (1, 4) for variant in kernels]
    report = dict(case='water/cc-pVDZ', nw=32, eta=1e-6, threads=[1, 4], rows=[], kernels=[])
    with threadpool_limits(limits=1):
        with (args.output/'preparation.log').open('w') as log, contextlib.redirect_stdout(log):
            mol = Molecule(atom=CASES['H2O'], basis='cc-pvdz', unit='angstrom').build(
                eri='cd', options={'low_rank_tol': 1e-10, 'eri_screen_tol': 0., 'workers': 1})
            mf = mol.RHF().run(tol=1e-11, max_cycle=100, verbose=0)
            assert mf.converged
            template = GW(mf, freq_int='cd', eta=1e-6, threads=1)
        no = int(template.nocc//2)
        selected = [no-1, no]
        report.update(nao=int(mol.nao), naux=len(template._pair_factors), selected=selected,
                      libraries=threadpool_info())
        original = {name: getattr(c.ContourDeformation, name) for name in
                    ('__init__', 'dielectric', 'screened_elements', 'evaluate', 'residue', '_pair_projection')}
        for repeat in range(args.repeats):
            for variant, threads in (configurations if repeat%2 == 0 else configurations[::-1]):
                c._contour_contractions = kernels[variant]
                times = {name: 0. for name in original}
                calls = {name: 0 for name in original}
                def timed(name):
                    def call(self, *a, **kw):
                        start = perf_counter()
                        try:
                            return original[name](self, *a, **kw)
                        finally:
                            times[name] += perf_counter()-start
                            calls[name] += 1
                    return call
                profile = cProfile.Profile()
                gw = GW(mf, freq_int='cd', eta=1e-6, threads=threads,
                        mo_pair_factors=template._pair_factors)
                with contextlib.ExitStack() as stack:
                    for name in original:
                        stack.enter_context(patch.object(c.ContourDeformation, name, timed(name)))
                    log = stack.enter_context((args.output/f'gw-{variant}-{threads}-{repeat}.log').open('w'))
                    stack.enter_context(contextlib.redirect_stdout(log))
                    start = perf_counter()
                    gw.run(orbs=selected, nw=32, residue_solver='direct', screening_tol=1e-11)
                    seconds = perf_counter()-start
                sigma = [[gw._contour.evaluate(int(p), float(mf.mo_energy[p])+shift)
                          for shift in (-.037, 0., .037)] for p in selected]
                sigma = np.asarray(sigma, dtype=complex)
                row = dict(variant=variant, threads=threads, repeat=repeat, seconds=seconds, inclusive_seconds=times,
                           sigma_real=sigma.real.tolist(), sigma_imag=sigma.imag.tolist(),
                           calls=calls, qp=gw.e_qp[selected].tolist(), weights=gw.qp_weights[selected].tolist(),
                           residuals=gw.qp_residuals[selected].tolist(), threading=gw.threading_info)
                assert max(row['residuals']) < 1e-9
                report['rows'].append(row)
                print(json.dumps(row), flush=True)
        c._contour_contractions = kernels['optimized']
        # Separate profiled run: profiler overhead is excluded from timed repetitions.
        gw = GW(mf, freq_int='cd', eta=1e-6, threads=4, mo_pair_factors=template._pair_factors)
        with (args.output/'profile.log').open('w') as log, contextlib.redirect_stdout(log):
            profile.runcall(gw.run, orbs=selected, nw=32, residue_solver='direct', screening_tol=1e-11)
        profile.dump_stats(str(args.output/'gw-4.prof'))
        kernel = c._contour_contractions
        rng = np.random.default_rng(651)
        for rank, transitions in [(256, 2048), (512, 8192)]:
            pairs = rng.normal(scale=.005, size=(rank, transitions+3))[:, :transitions]
            weights = -rng.uniform(.1, 2., transitions)
            complex_weights = weights + 1j*rng.normal(scale=.1, size=transitions)
            vector = rng.normal(size=rank)+1j*rng.normal(size=rank)
            for variant, kernel in kernels.items():
                for name, fn in [('imaginary_dielectric', lambda t: kernel.imaginary_dielectric(pairs, weights, t)),
                                 ('dielectric', lambda t: kernel.dielectric(pairs, complex_weights, t)),
                                 ('action', lambda t: kernel.action(pairs, complex_weights, vector, t)),
                                 ('project', lambda t: kernel.project(pairs, vector, t))]:
                    reference = fn(1)[0]
                    np.save(args.output/f'{variant}-{name}-{rank}.npy', reference)
                    for threads in (1, 4):
                        fn(threads)
                        durations=[]
                        for _ in range(args.repeats):
                            start=perf_counter(); result, used=fn(threads); durations.append(perf_counter()-start)
                        error=float(np.max(np.abs(result-reference)))
                        np.testing.assert_allclose(result, reference, atol=1e-11, rtol=1e-11)
                        report['kernels'].append(dict(variant=variant, kernel=name, rank=rank, transitions=transitions,
                            threads=threads, used=used, seconds=durations, max_thread_difference=error))
        (args.output/'results.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
