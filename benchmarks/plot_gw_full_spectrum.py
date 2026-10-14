"""Reproduce full-orbital contour validation and satellite-weight diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = json.loads((args.input/'comparison.json').read_text())
    records = report['orbitals']
    ids = np.array(sorted(map(int,records)))
    energy = np.array([records[str(p)].get('native',{}).get('energy',np.nan) for p in ids])
    reference = np.array([records[str(p)]['stages'][-1]['energy']
                          if records[str(p)].get('stages') else np.nan for p in ids])
    weight = np.array([records[str(p)].get('native',{}).get('weight',np.nan) for p in ids])
    accepted = np.array([records[str(p)]['accepted'] for p in ids])
    args.output.mkdir(parents=True,exist_ok=True)
    fig,axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,mask,title in [(axes[0,0],ids<18,'Core orbitals'),
                          (axes[0,1],ids>=18,'Valence and virtual orbitals')]:
        ax.plot(ids[mask],energy[mask]*27.211386245988,'o',ms=3,label='PyQED full spectral GW')
        ax.plot(ids[mask],reference[mask]*27.211386245988,'+',ms=5,label='PySCF contour equations')
        ax.set(title=title,xlabel='Spatial MO index (zero based)',ylabel='Selected root energy (eV)')
    axes[0,0].legend(fontsize=8)
    error = abs(energy-reference)*27.211386245988
    axes[1,0].semilogy(ids,np.maximum(error,1e-14),'.',color='#2878a5')
    axes[1,0].axhline(report['energy_tolerance_Ha']*27.211386245988,
                      ls='--',color='black',label='Acceptance tolerance')
    axes[1,0].set(xlabel='Spatial MO index',ylabel='Absolute matched-root difference (eV)')
    axes[1,0].set_title(r'Zero differences shown at $10^{-14}$ eV',fontsize=10)
    axes[1,0].legend(fontsize=8)
    axes[1,1].semilogy(ids,weight,'.',color='#2878a5')
    axes[1,1].axhline(.01,ls='--',color='gray',label='Weak-root warning threshold')
    axes[1,1].legend(fontsize=8,loc='lower left')
    if 239 in ids:
        i = int(np.flatnonzero(ids==239)[0])
        axes[1,1].plot(239,weight[i],'o',color='#c34f31')
        axes[1,1].annotate(f'239: Z = {weight[i]:.3g}',(239,weight[i]),
                            xytext=(-90,25),textcoords='offset points',
                            arrowprops=dict(arrowstyle='->'))
    axes[1,1].set(xlabel='Spatial MO index',ylabel='Selected-root spectral weight Z')
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.suptitle(f'Tetracene: {accepted.sum()}/{len(ids)} matched roots accepted\n'
                 'Full screening; local root selection; not a dominant-peak assignment',fontsize=12)
    fig.savefig(args.output/'tetracene_full_gw_contour_validation.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
