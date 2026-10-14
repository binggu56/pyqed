"""Small reproducible default Padé GW/QP-gap BSE check against PySCF 2.14.

H2/6-31G uses a small three-pair space; explicitly request one BSE root. Other numerical controls use the defaults. This is a regression
reference, not a chemically converged spectrum or a timing benchmark.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from pyscf import gto, scf, lib
from pyscf.gw import gw_ac, bse as pbse
from threadpoolctl import threadpool_limits
from pyqed.gw.gw import GW
from pyqed.gw.bse import BSE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lib.num_threads(1)
    with threadpool_limits(limits=1):
        mol=gto.M(atom='H 0 0 0; H 0 0 .74',basis='6-31g',verbose=0)
        mf=scf.RHF(mol);mf.chkfile=None;mf.conv_tol=1e-12;mf.kernel()
        q=GW(mf).run();ref=gw_ac.GWAC(mf);ref.kernel()
        b=BSE(q).run(nroots=1,return_vectors=True)
        other=pbse.BSE(ref);other.nroot=1;other.init_ntri=1;other.kernel('s')
        values=np.r_[q.e_qp[:2],b.e]*27.211386245988
        expected=np.r_[ref.mo_energy[:2],other.exci]*27.211386245988
        np.testing.assert_allclose(values,expected,atol=3e-5,rtol=0)
        result=dict(molecule='H2',basis='6-31G',screening='QP gaps',
            pyqed_eV=values.tolist(),pyscf_eV=expected.tolist(),difference_eV=(values-expected).tolist(),
            stability=q.ac_info['stability'],bse_converged=b.info['converged'])
        (a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    x=np.arange(3)
    axes[0].bar(x-.18,values,.36,label='PyQED')
    axes[0].bar(x+.18,expected,.36,label='PySCF 2.14')
    axes[0].set(ylabel='Energy (eV)',title='GW energies and BSE excitation')
    axes[0].legend()
    axes[1].bar(x,(values-expected)*1e6)
    axes[1].set(ylabel='PyQED − PySCF (µeV)',title='Difference')
    for ax in axes:ax.set(xticks=x,xticklabels=['GW HOMO','GW LUMO','BSE S1']);ax.grid(axis='y',alpha=.2)
    fig.suptitle('H₂ / 6-31G: default Padé GW and QP-gap BSE\nPadé stability checks enabled; lowest BSE root')
    fig.savefig(a.output/'h2_gw_bse_defaults.png',dpi=160)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
