"""Compare molecular full-BSE solvers on synthetic symmetric factor kernels.

Uses production screened actions; small dense matrices are validation only.
This is not an ab initio molecular benchmark. Limit all BLAS/OpenMP threads
to one, run with PYTHONPATH=., and place --output outside the repository.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pyqed.gw.bse import solve_bse, _bse_full_matvec


def make_model(nocc, nvir):
    rng = np.random.default_rng(128+nocc)
    n = nocc+nvir
    factors = rng.normal(size=(16, n, n))*.015
    factors = (factors+factors.transpose(0, 2, 1))/2
    m = rng.normal(size=(n, n, 12))*.006
    m = (m+m.transpose(1, 0, 2))/2
    return SimpleNamespace(nocc=nocc, nso=n, e_qp=None,
        e_mf=np.r_[np.linspace(-1., -.4, nocc), np.linspace(.2, 2., nvir)],
        _pair_factors=factors, eri=None, _M=m, e_rpa=np.linspace(.5, 2., 12))


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    methods = ['arpack', 'davidson', 'davidson_batched']
    labels = ['ARPACK', 'Davidson', 'Davidson batched']
    for i, (method, label) in enumerate(zip(methods, labels)):
        selected = [r for r in rows if r['method'] == method]
        x = np.arange(len(selected))+(i-1)*.25
        axes[0].bar(x, [r['seconds'] for r in selected], .25, label=label)
        axes[1].bar(x, [r['operator_columns'] for r in selected], .25)
    for ax in axes:
        ax.set_xticks(np.arange(len(rows)//3), [str(r['dimension']) for r in rows if r['method'] == 'arpack'])
        ax.set_xlabel('Full BSE dimension')
    axes[0].set_ylabel('Median solve time (s)')
    axes[1].set_ylabel('Median operator columns\n(including validation)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=3)
    fig.suptitle('Synthetic factorized molecular BSE kernels: 5 roots, single thread')
    for ext in ['png', 'pdf']:
        fig.savefig(output/f'bse_nonsymmetric_comparison.{ext}', dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(json.loads((args.output/'report.json').read_text()), args.output)
        return
    rows = []
    for nocc, nvir in [(4, 16), (8, 24)]:
        model = make_model(nocc, nvir)
        n = 2*nocc*nvir
        h = np.column_stack([_bse_full_matvec(model, v) for v in np.eye(n)])
        reference = np.sort(np.linalg.eigvals(h).real)[n//2:n//2+5]
        def run(method):
            start = time.perf_counter()
            w, v, info = solve_bse(model, nroots=5, tol=1e-9, max_cycle=300,
                eigensolver='arpack' if method == 'arpack' else 'davidson',
                batch_columns=8 if method == 'davidson_batched' else None, return_info=True)
            seconds = time.perf_counter()-start
            error = float(max(abs(w-reference)))
            residual = float(max(np.linalg.norm(h@v-v*w, axis=0)))
            assert error < 1e-8 and residual < 1e-9 and info['converged']
            return dict(seconds=seconds, operator_columns=info['operator_columns'],
                        max_residual=residual, eigenvalue_error=error, metric_error=info['metric_error'])
        methods = ['arpack', 'davidson', 'davidson_batched']
        for method in methods:
            run(method)
        runs = {method: [] for method in methods}
        for repeat in range(args.repeats):
            for method in np.random.default_rng(501+repeat).permutation(methods):
                runs[method].append(run(method))
        for method in methods:
            samples = runs[method]
            row = dict(method=method, dimension=n, samples=samples,
                seconds=float(np.median([r['seconds'] for r in samples])),
                operator_columns=float(np.median([r['operator_columns'] for r in samples])))
            rows.append(row)
            print(n, method, row['seconds'], row['operator_columns'], flush=True)
        (args.output/'report.json').write_text(json.dumps(rows, indent=2))
    plot(rows, args.output)


if __name__ == '__main__':
    main()
