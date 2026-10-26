"""Validate molecular Hessian columns against independently rebuilt gradients.

Run after pyrazine_adjoint_gradients.py --quantity hessian --method adjoint.
Displaced gradients use the saved system and basis with the same reference
settings. Processes are isolated unless --reuse-process is requested. CD is rebuilt at each geometry; rank/pivot changes are reported,
not hidden. Two central-difference steps diagnose truncation and rebuild noise.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
import subprocess
import sys
from pathlib import Path
import numpy as np


def point_name(column, step, sign):
    return f'fd_c{column}_h{step:g}_{"plus" if sign == 1 else "minus"}'


def full_diagnostics(out):
    h=np.load(out/'adjoint_hessian.npy')[0]
    g=np.load(out/'adjoint_gradient.npy')[0]
    run=json.loads((out/'adjoint.json').read_text())
    translations=np.tile(np.eye(3),(len(h)//3,1))
    data=dict(max_asymmetry=float(np.max(abs(h-h.T))),
        max_translation_residual=float(np.max(abs(h@translations))),
        total_gradient=(g@translations).tolist())
    (out/'hessian_diagnostics.json').write_text(json.dumps(data,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    lim=np.max(abs(h))
    im=axs[0].imshow(h,cmap='RdBu_r',vmin=-lim,vmax=lim)
    fig.colorbar(im,ax=axs[0],label='Hartree/bohr²')
    axs[0].set(title='a  Full analytic Hessian',xlabel='Coordinate',ylabel='Coordinate')
    im=axs[1].imshow(h-h.T,cmap='RdBu_r')
    fig.colorbar(im,ax=axs[1],label='Hartree/bohr²')
    axs[1].set(title='b  H − Hᵀ (unsymmetrized)',xlabel='Coordinate',ylabel='Coordinate')
    stages=run['derivative_info']['timings']
    bars=axs[2].bar(['First','Reverse','Curvature'],np.array(list(stages.values()))/60,
        color=['#0072B2','#D55E00','#009E73'])
    axs[2].bar_label(bars,fmt='%.1f',padding=3)
    axs[2].set(title='c  Hessian stage timings',ylabel='Wall time (min)')
    axs[2].margins(y=.2)
    fig.suptitle(f"Pyrazine/cc-pVDZ, 30 coordinates: {run['seconds']/60:.2f} min, peak RSS {run['peak_rss_bytes']/2**30:.3f} GiB")
    for ext in ('png','pdf'):fig.savefig(out/f'pyrazine_full_hessian_diagnostics.{ext}',dpi=300)
    return data


def summarize(out, columns, steps):
    h = np.load(out/'adjoint_hessian.npy')[0]
    g = np.load(out/'adjoint_gradient.npy')[0]
    run = json.loads((out/'adjoint.json').read_text())
    reference = np.load(out/'adjoint_reference.npz')
    selected = run.get('columns',list(range(h.shape[1])))
    block = h[selected,:]
    rectangular = h.shape[0] != h.shape[1]
    translations = np.tile(np.eye(3), (h.shape[0]//3, 1))
    report = dict(hessian_run=run, columns=columns, analytic_columns=selected, steps_bohr=steps,
        max_asymmetry=float(np.max(abs(block-block.T))),
        max_translation_residual=float(np.max(abs(h.T@translations))),
        total_gradient=(g@translations).tolist(), validation=[])
    errors = []
    for column in columns:
        estimates = []
        for step in steps:
            dirs = [out/point_name(column, step, sign) for sign in (1, -1)]
            records = [json.loads((d/'adjoint.json').read_text()) for d in dirs]
            if not all(r['success'] for r in records):
                raise RuntimeError('Displaced gradient failed')
            refs = [np.load(d/'adjoint_reference.npz') for d in dirs]
            for sign, displaced in zip((1, -1), refs):
                expected = reference['coords'].copy()
                expected.flat[column] += sign*step
                np.testing.assert_allclose(displaced['coords'], expected, atol=1e-13, rtol=0)
            gradients = [np.load(d/'adjoint_gradient.npy')[0] for d in dirs]
            estimate = (gradients[0]-gradients[1])/(2*step)
            estimates.append(estimate)
            delta = estimate-h[:len(estimate), selected.index(column)]
            errors.append(delta)
            pivots = [np.load(d/'adjoint_cd_pivots.npy') for d in dirs]
            report['validation'].append(dict(column=column,step_bohr=step,
                max_error=float(np.max(abs(delta))),rms_error=float(np.sqrt(np.mean(delta**2))),
                gradient_rows=len(estimate),
                fixed_pivots=records[0].get('fixed_pivots'),
                max_qp_changes=[float(np.max(abs(r['qp']-reference['qp']))) for r in refs],
                ranks=[r['rank'] for r in records],
                plus_minus_pivots_equal=bool(np.array_equal(*pivots)),
                excitation_energies=[r['excitation'].tolist() for r in refs],
                minimum_excitation_gap=float(min(np.min(np.diff(r['excitation'])) for r in refs)),
                gradient_wall_seconds=[r['seconds'] for r in records],
                reference_seconds=[r['reference_seconds'] for r in records]))
        if len(steps) == 2:
            ratio = steps[0]/steps[1]
            extrapolated = (ratio**2*estimates[1]-estimates[0])/(ratio**2-1)
            report.setdefault('step_convergence', []).append(dict(column=column,
                max_step_change=float(np.max(abs(estimates[1]-estimates[0]))),
                richardson_max_error=float(np.max(abs(extrapolated-h[:len(extrapolated),selected.index(column)])))))
    finest = min(steps)
    report['finest_step_max_error'] = max(v['max_error'] for v in report['validation'] if v['step_bohr'] == finest)
    report['absolute_validation_tolerance'] = 1e-5
    report['validation_passed'] = report['finest_step_max_error'] < report['absolute_validation_tolerance']
    np.save(out/'finite_difference_errors.npy',np.array(errors))
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,2,figsize=(10,7),layout='constrained')
    lim = np.max(abs(h))
    im=axs[0,0].imshow(h,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto' if rectangular else 'equal')
    fig.colorbar(im,ax=axs[0,0],label='Hartree/bohr²')
    axs[0,0].set(title='a  Selected analytic columns' if rectangular else 'a  Full analytic Hessian',xlabel='Coordinate',ylabel='Coordinate')
    if rectangular:
        axs[0,0].set_xticks(range(len(selected)),selected)
    colors=['#0072B2','#D55E00','#009E73']
    for index,column in enumerate(columns):
        entries=sorted((v for v in report['validation'] if v['column']==column),
                       key=lambda v:v['step_bohr'])
        axs[0,1].loglog([v['step_bohr'] for v in entries],
                       [max(v['max_error'],1e-16) for v in entries],
                       marker=['o','s','^'][index//3%3],color=colors[index%3],
                       label=f'Column {column}')
    axs[0,1].axhline(report['absolute_validation_tolerance'],color='0.4',
                    linestyle='--',linewidth=1,label='Tolerance')
    axs[0,1].set(title='b  Finite-difference convergence',xlabel='Step (bohr)',
                ylabel='Max column error (Hartree/bohr²)')
    axs[0,1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3)
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    axs[0,1].xaxis.set_major_locator(FixedLocator(sorted(steps)))
    axs[0,1].xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
    axs[0,1].xaxis.set_minor_locator(NullLocator())
    timings=run['derivative_info']['timings']
    bars=axs[1,0].bar(['First response','Reverse response','Integral curvature'],
                     list(timings.values()),color=colors)
    axs[1,0].bar_label(bars,fmt='%.1f',padding=3)
    axs[1,0].set(title='c  Hessian stage timings',ylabel='Wall time (s)')
    axs[1,0].margins(y=.2);axs[1,0].tick_params(axis='x',labelsize=8)
    axs[1,1].axis('off')
    axs[1,1].text(0,1,'d  Benchmark and invariance checks',va='top',weight='bold')
    lines=[f"Analytic Hessian ({h.shape[0]} × {h.shape[1]}): {run['seconds']:.1f} s",f"Sampled peak RSS: {run['peak_rss_bytes']/2**30:.3f} GiB",
        f"Incremental peak: {run['incremental_peak_bytes']/2**30:.3f} GiB",
        f"Selected-block asymmetry: {report['max_asymmetry']:.2e}",
        f"Max translational residual: {report['max_translation_residual']:.2e}",
        'State 0 = first excitation; total RHF + excitation',
        'Idealized geometry; not an optimized minimum']
    axs[1,1].text(0,.87,'\n\n'.join(lines),va='top',fontsize=9)
    fig.suptitle(f"{run['system'].title()}/{run['basis']}: {h.shape[0]} × {h.shape[1]} Z-vector Hessian")
    for ax in axs.flat: ax.spines[['top','right']].set_visible(False)
    scope = 'selected' if rectangular else 'full'
    for ext in ('png','pdf'):fig.savefig(out/f"{run['system']}_{scope}_hessian_validation.{ext}",dpi=300)
    print(json.dumps({k:v for k,v in report.items() if k!='hessian_run'},indent=2))


def scan_roots(out):
    """Probe selection continuity without further derivative calculations."""
    reference=np.load(out/'adjoint_reference.npz')
    records=[]
    for step in (0.,1e-5,1e-6):
        dest=out/f'root_scan_x_{step:g}'
        dest.mkdir(exist_ok=True)
        if not (dest/'reference_only.json').exists():
            with (dest/'run.log').open('w') as stream:
                subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('pyrazine_adjoint_gradients.py')),
                    '--method','adjoint','--output',str(dest),'--reference-only',
                    '--displacement-coordinate','0','--displacement='+str(step)],
                    stdout=stream,stderr=subprocess.STDOUT,check=True)
        point=np.load(dest/'adjoint_reference.npz')
        selection=np.load(dest/'adjoint_qp_selection.npz')
        change=point['qp']-reference['qp']
        record=dict(step_bohr=step,max_qp_change=float(np.max(abs(change))),
            largest_changed_orbital=int(np.argmax(abs(change))),
            orbital94_root=float(selection['roots'][94]),
            orbital94_weight=float(selection['weights'][94]),
            excitation=point['excitation'].tolist(),
            rank=json.loads((dest/'reference_only.json').read_text())['rank'])
        records.append(record)
        print(record,flush=True)
    (out/'root_selection_scan.json').write_text(json.dumps(records,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--columns',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--steps',type=float,nargs='+',default=[1e-3,5e-4])
    p.add_argument('--gradient-directions',type=int,default=0)
    p.add_argument('--fixed-pivots',type=Path)
    p.add_argument('--reuse-process',action='store_true',help='Avoid import overhead for small-system gradients; per-gradient RSS is then not isolated.')
    p.add_argument('--summarize',action='store_true')
    p.add_argument('--hessian-only',action='store_true')
    p.add_argument('--scan-roots',action='store_true')
    a=p.parse_args()
    if a.scan_roots:
        scan_roots(a.output)
        return
    if a.hessian_only:
        print(json.dumps(full_diagnostics(a.output),indent=2))
        return
    if not a.summarize:
        reference_run=json.loads((a.output/'adjoint.json').read_text())
        for column in a.columns:
            for step in a.steps:
                for sign in (1,-1):
                    dest=a.output/point_name(column,step,sign)
                    dest.mkdir(parents=True,exist_ok=True)
                    record=dest/'adjoint.json'
                    if record.exists():
                        saved=json.loads(record.read_text())
                        if saved.get('success'):
                            fixed=None if a.fixed_pivots is None else str(a.fixed_pivots)
                            count=a.gradient_directions or saved['coordinates']
                            controls = {'scf_tol': 1e-11, 'scf_gradient_tol': None,
                                        'qp_tol': None, 'dense_bse': False, 'bse_tol': 1e-9}
                            changed = any(saved.get(k, default) != reference_run.get(k, default)
                                          for k, default in controls.items())
                            changed |= any(saved[k] != reference_run[k] for k in ('system', 'basis'))
                            if changed or saved.get('fixed_pivots') != fixed or len(saved['directions']) != count:
                                raise ValueError('Existing gradient uses different controls; choose a fresh output directory.')
                            continue
                    print(f'Gradient column {column}, displacement {sign*step:g} bohr',flush=True)
                    if a.reuse_process:
                        from pyrazine_adjoint_gradients import worker
                        options=argparse.Namespace(output=dest,method='adjoint',quantity='gradient',
                            system=reference_run['system'],basis=reference_run['basis'],
                            dense_bse=reference_run.get('dense_bse',False),bse_tol=reference_run.get('bse_tol',1e-9),
                            qp_tol=reference_run.get('qp_tol'),scf_tol=reference_run.get('scf_tol',1e-11),
                            scf_gradient_tol=reference_run.get('scf_gradient_tol'),
                            displacement_coordinate=column,displacement=sign*step,
                            directions=a.gradient_directions,memory=reference_run['max_memory_MB'],
                            fixed_pivots=a.fixed_pivots,reference_only=False,reuse_process=True)
                        with (dest/'run.log').open('w') as stream, redirect_stdout(stream), redirect_stderr(stream):
                            worker(options)
                        continue
                    with (dest/'run.log').open('w') as stream:
                        subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('pyrazine_adjoint_gradients.py')),
                            '--quantity','gradient','--method','adjoint','--output',str(dest),
                            '--system',reference_run['system'],'--basis',reference_run['basis'],
                            '--displacement-coordinate',str(column),'--displacement='+str(sign*step),
                            '--directions',str(a.gradient_directions)]
                            +(['--dense-bse'] if reference_run.get('dense_bse',False) else [])
                            +['--bse-tol',str(reference_run.get('bse_tol',1e-9))]
                            +([] if reference_run.get('qp_tol') is None else ['--qp-tol',str(reference_run['qp_tol'])])
                            +['--scf-tol',str(reference_run.get('scf_tol',1e-11))]
                            +([] if reference_run.get('scf_gradient_tol') is None else
                              ['--scf-gradient-tol',str(reference_run['scf_gradient_tol'])])
                            +([] if a.fixed_pivots is None else ['--fixed-pivots',str(a.fixed_pivots)]),
                            stdout=stream,stderr=subprocess.STDOUT,check=True)
    summarize(a.output,a.columns,a.steps)


if __name__=='__main__':main()
