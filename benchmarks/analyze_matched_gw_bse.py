"""Diagnose saved fits from gw_bse_matched_scaling.py without repeating GW.

Usage: PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python
benchmarks/analyze_matched_gw_bse.py /private/tmp/anthracene-gw-bse-matched
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import newton
from pyqed.gw.ac import _pade
from pyscf.gw.utils.ac_grid import pade_thiele_ndarray, thiele_ndarray

EV = 27.211386245988


def corrected_pade(freqs, nodes, coeff):
    """Explicit benchmark adapter for PySCF 2.14's duplicated-tail bug.

    Reuses PyQED's tested continued-fraction evaluator. This is a corrected
    reference, not stock PySCF; callers must label it accordingly.
    """
    freqs, coeff = np.asarray(freqs), np.asarray(coeff)
    if freqs.ndim != 1 or np.ndim(nodes) != 1 or len(coeff) != len(nodes):
        raise ValueError("Inconsistent continued-fraction shapes")
    shape = (len(freqs),)+(1,)*(coeff.ndim-1)
    return _pade(freqs.reshape(shape), coeff, nodes)[0]


def analyze(output):
    hf = np.load(output/'orbitals.npz')['energy']
    nocc = json.loads((output/'input.json').read_text())['nocc']
    q = np.load(output/'PyQED-4-1-fit.npz')
    p = np.load(output/'PySCF-4-1-fit.npz')
    q_qp = np.load(output/'PyQED-4-1.npz')['qp']
    p_qp = np.load(output/'PySCF-4-1.npz')['qp']
    sigma = p['sigma'].copy()
    sigma[:nocc] = sigma[:nocc].conj()
    frequencies = np.broadcast_to(p['omega']-q['ef'], q['omega'].shape).copy()
    frequencies[:nocc] = frequencies[:nocc].conj()
    def q_fit(i, w):
        return _pade(w-q['ef'], q['coeff'][:, i], q['nodes'][i])
    def p_corrected_fit(i, w):
        return _pade(w, p['coeff'][:, i], p['omega'][p['indices']])
    def p_stock_fit(i, w):
        return pade_thiele_ndarray(np.asarray([w],dtype=complex),p['omega'][p['indices']],p['coeff'][:,i])[0]
    corrected_qp=np.full_like(hf,np.nan)
    corrected_failures={}
    for i in range(len(hf)):
        try:
            corrected_qp[i]=newton(lambda w:w-hf[i]-p_corrected_fit(i,w)[0].real-p['static'][i],
                                    hf[i],tol=1e-7,maxiter=100)
        except (RuntimeError,FloatingPointError) as err:corrected_failures[i]=str(err)
    corrected_weights=np.array([1/(1-p_corrected_fit(i,corrected_qp[i])[1].real)
                                if np.isfinite(corrected_qp[i]) else np.nan for i in range(len(hf))])
    if np.all(np.isfinite(corrected_qp)) and np.all(corrected_weights>0):
        np.savez(output/'PySCF-4-corrected-pade-qp.npz',qp=corrected_qp)
    else:
        (output/'PySCF-4-corrected-pade-qp.npz').unlink(missing_ok=True)
    nodes=np.array([1j,2j]);values=1/(1+nodes)
    toy_coeff=thiele_ndarray(values,nodes)
    toy_stock=pade_thiele_ndarray(nodes,nodes,toy_coeff)
    toy_corrected=_pade(nodes,toy_coeff,nodes)[0]
    np.testing.assert_allclose(toy_corrected,values,atol=1e-15)
    selected_values=p['sigma'][:,p['indices']].T
    stock_selected=pade_thiele_ndarray(p['omega'][p['indices']],p['omega'][p['indices']],p['coeff'])
    corrected_selected=np.column_stack([p_corrected_fit(i,p['omega'][p['indices']])[0] for i in range(len(hf))])
    report = dict(
        pade_two_point_stock_error=float(np.max(abs(toy_stock-values))),
        pade_two_point_corrected_error=float(np.max(abs(toy_corrected-values))),
        pyscf_selected_node_interpolation_error_Ha=float(np.max(abs(stock_selected-selected_values))),
        corrected_selected_node_interpolation_error_Ha=float(np.max(abs(corrected_selected-selected_values))),
        corrected_qp_failures=corrected_failures,
        corrected_nonpositive_weight_orbitals=np.flatnonzero(corrected_weights<=0),
        corrected_qp_min_weight=float(np.nanmin(corrected_weights)),
        corrected_qp_max_difference_from_PyQED_eV=float(np.nanmax(abs(corrected_qp-q_qp))*EV),
        corrected_qp_difference_from_PyQED_eV=(corrected_qp-q_qp)*EV,

        imaginary_sigma_max_error_Ha=float(np.max(abs(q['sigma']-sigma))),
        imaginary_frequency_max_error_Ha=float(np.max(abs(q['omega']-frequencies))),
        static_shift_max_error_Ha=float(np.max(abs(q['static']-p['static']))),
        qp_difference_eV=(q_qp-p_qp)*EV,
        frontier_qp_difference_eV=(q_qp-p_qp)[nocc-3:nocc+3]*EV,
        cross_fit_residual_Ha=[float(abs(p_qp[i]-hf[i]-q_fit(i,p_qp[i])[0].real-q['static'][i])) for i in range(len(hf))],
        fit_and_root_diagnostics=[])
    for i in np.argsort(abs(q_qp-p_qp))[-12:]:
        row = dict(orbital=int(i),hf_eV=float(hf[i]*EV),
                   qp_difference_eV=float((q_qp[i]-p_qp[i])*EV),
                   sigma_at_hf_difference_eV=float((q_fit(i,hf[i])[0]-p_stock_fit(i,hf[i])).real*EV))
        for name, fit, shift, derivative in [('PyQED_secant',q_fit,q['static'][i],False),
                                            ('PySCF_corrected_evaluator_Newton',p_corrected_fit,p['static'][i],True)]:
            try:
                root = newton(lambda w:w-hf[i]-fit(i,w)[0].real-shift,hf[i],
                              fprime=(lambda w:1-fit(i,w)[1].real) if derivative else None,
                              tol=1e-7,maxiter=100)
                row[name+'_eV']=float(root*EV)
            except (RuntimeError,FloatingPointError) as err:
                row[name+'_failure']=str(err)
        row['PySCF_corrected_evaluator_secant_eV']=float(corrected_qp[i]*EV)
        row['PyQED_default_eV']=float(q_qp[i]*EV)
        row['PySCF_default_eV']=float(p_qp[i]*EV)
        report['fit_and_root_diagnostics'].append(row)
    assert report['imaginary_sigma_max_error_Ha']<1e-12
    assert report['static_shift_max_error_Ha']<1e-10
    controls=[json.loads(p.read_text()) for p in output.glob('Py*-control.json')]
    if controls:
        assert len(controls)==8
        assert all(r['success'] for r in controls)
        report['control_excitation_spread_eV']=np.ptp([r['excitation_eV'] for r in controls],axis=0)
        report['control_max_residual_Ha']=max(max(r['physical_residuals']) for r in controls)
    (output/'fit_diagnostics.json').write_text(json.dumps(report,indent=2,default=lambda x:x.tolist())+'\n')
    plot_diagnostics(output, report, nocc)
    print(json.dumps({k:v for k,v in report.items() if k not in ['fit_and_root_diagnostics','cross_fit_residual_Ha','qp_difference_eV','corrected_qp_difference_from_PyQED_eV']},indent=2,default=lambda x:x.tolist()))
    return report


def plot_diagnostics(output, report, nocc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    delta=np.asarray(report['qp_difference_eV'])
    axes[0].plot(np.arange(len(delta)),delta,'.',color='#0072B2')
    axes[0].axvline(nocc-.5,color='gray',ls='--',label='HOMO / LUMO boundary')
    axes[0].axhline(0,color='gray',lw=.6)
    axes[0].set(xlabel='Orbital index (zero-based)',ylabel='PyQED − PySCF QP energy (eV)',
                title='Independent GW, four threads')
    axes[0].legend(fontsize=8)
    controls=[json.loads(p.read_text()) for p in output.glob('Py*-control.json')]
    if controls:
        reference=next(r for r in controls if r['solver']=='PySCF' and r['threads']==1)
        for r in sorted(controls,key=lambda r:(r['solver'],r['threads'])):
            d=(np.asarray(r['excitation_eV'])-reference['excitation_eV'])*1e6
            axes[1].plot(range(1,len(d)+1),d,'o-',label=f"{r['solver']}, {r['threads']}")
        axes[1].legend(fontsize=7,ncol=2)
    axes[1].set(xlabel='Singlet root',ylabel='Difference from PySCF / 1 thread (µeV)',
                title='BSE control: identical QPs and screening',xticks=range(1,6))
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Anthracene / cc-pVDZ: separate GW continuation errors from BSE errors')
    fig.savefig(output/'anthracene_gw_bse_accuracy_diagnostics.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    if args.plot_only:
        report=json.loads((args.output/'fit_diagnostics.json').read_text())
        nocc=json.loads((args.output/'input.json').read_text())['nocc']
        plot_diagnostics(args.output,report,nocc)
    else:analyze(args.output)
