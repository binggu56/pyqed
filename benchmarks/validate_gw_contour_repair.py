"""Small independent spectral validation of the checked AC-to-contour repair.

The dense screening eigensolve is an explicit small-molecule reference only.
Run with PYTHONPATH=. and single-threaded BLAS/OpenMP.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from pyscf import gto, scf, df, lib
from pyqed.qchem.basis import PackedRIFactors
from pyqed.gw.gw import GW
from pyqed.gw.ac import refit, repair_continuation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lib.num_threads(1)
    mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.chkfile = None
    mf.conv_tol = 1e-12
    mf.kernel()
    fitted = df.DF(mol).build()
    mol.eri_factors = PackedRIFactors(np.concatenate(list(fitted.loop())), mol.nao)
    ac = GW(mf, freq_int='ac').run(nw=64, stability_tol=None)
    old = ac.e_qp.copy()
    try:
        refit(ac, mf.mo_energy)
    except RuntimeError:
        pass
    targets = ac.ac_info['stability']['unstable_orbitals']
    if not targets:
        raise RuntimeError('Validation requires a rejected continuation target')
    factors = ac._pair_factors
    previous = GW(mf, freq_int='contour', mo_pair_factors=factors,
                  eta=ac.eta).run(nw=64, orbs=targets, tol=1e-9)
    repair_continuation(ac, mf.mo_energy, nw=64, tol=1e-9, accuracy_tol=1e-7)
    exact = GW(mf, freq_int='exact', mo_pair_factors=factors, eta=0).run()
    from pyscf.gw import gw_cd
    reference = gw_cd.GWCD(mf)
    reference.Lpq = factors
    reference.orbs = list(range(len(mf.mo_energy)))
    reference.eta = 1e-8
    reference.nw = 256
    reference.vhf_df = True
    reference.qpe_tol = 1e-10
    success, converged, reference_qp = gw_cd.kernel(reference)
    if not success or not np.all(converged[targets]):
        raise RuntimeError('PySCF contour reference did not converge')
    report = dict(pyscf_contour_Ha=reference_qp[targets].tolist(),
                  pyscf_difference_eV=((ac.e_qp[targets]-reference_qp[targets])*27.211386245988).tolist(),
                  molecule='water', basis='sto-3g', targets=targets,
                  unchecked_ac_Ha=old.tolist(),
                  old_contour_Ha=previous.e_qp[targets].tolist(),
                  repaired_Ha=ac.e_qp[targets].tolist(),
                  exact_Ha=exact.e_qp[targets].tolist(),
                  repaired_errors_eV=((ac.e_qp[targets]-exact.e_qp[targets])*27.211386245988).tolist(),
                  qp_weights=ac.qp_weights[targets].tolist(),
                  qp_residuals_Ha=ac.qp_residuals[targets].tolist(),
                  audit=ac.ac_info['contour_repair'])
    (args.output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    values = [np.max(abs(old[targets]-exact.e_qp[targets])),
              np.max(abs(previous.e_qp[targets]-exact.e_qp[targets])),
              np.max(abs(ac.e_qp[targets]-exact.e_qp[targets]))]
    axes[0].bar(['Unchecked Padé', 'Old contour\nη=0.01 Ha', 'Checked contour'],
                np.maximum(values, 1e-16)*27.211386245988)
    axes[0].set(yscale='log', ylabel='Maximum error vs spectral reference (eV)',
                title='Rejected water core state')
    history = report['audit']['history']
    for key, label in [('quadrature_changes_Ha', 'Quadrature'), ('eta_changes_Ha', 'Broadening')]:
        axes[1].semilogy(range(1,len(history)+1),
            [max(max(h[key]),1e-16)*27.211386245988 for h in history], 'o-', label=label)
    axes[1].axhline(report['audit']['accuracy_tol_Ha']*27.211386245988, ls='--',color='gray')
    axes[1].set(xlabel='Refinement round', ylabel='Energy change (eV)', title='Independent refinement checks')
    axes[1].legend()
    axes[1].set_xticks(range(1,len(history)+1))
    axes[1].text(.5, .5, 'Zero changes plotted at a numerical floor',
                 transform=axes[1].transAxes, ha='center', fontsize=8)
    fig.savefig(args.output/'water_gw_checked_repair.png', dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    if max(abs(np.asarray(report['pyscf_difference_eV']))) > 1e-5:
        raise RuntimeError('Repaired roots disagree with independent PySCF contour GW')
    if max(abs(np.asarray(report['repaired_errors_eV']))) > 1e-5:
        raise RuntimeError('Repaired roots disagree with the independent spectral reference')


if __name__ == '__main__':
    main()
