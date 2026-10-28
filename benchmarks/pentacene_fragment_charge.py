"""Reproduce six-state fragment moments directly from the saved active RDMs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from pyqed.qchem.mcscf.fragment_charge import fragment_charge_from_rdms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    data = np.load(args.checkpoint)
    groups = [np.flatnonzero(data['ao_atom_indices']<36),np.flatnonzero(data['ao_atom_indices']>=36)]
    start = time.perf_counter()
    rows = []
    for state,(d1,d2) in enumerate(zip(data['dm1_active'],data['dm2_active'])):
        result = fragment_charge_from_rdms(d1,d2,data['overlap'],data['mo_coeff'],groups,
            ncore=int(data['ncore']),nuclear_charges=[146,146])
        np.testing.assert_allclose(result['covariance'].sum(axis=0),0,atol=1e-8)
        rows.append(dict(state=state,energy=float(data['energies'][state]),
            charge_A=float(result['charge_mean'][0]),charge_B=float(result['charge_mean'][1]),
            variance_A=float(result['variance'][0]),variance_B=float(result['variance'][1]),
            covariance_AB=float(result['covariance'][0,1])))
    elapsed = time.perf_counter()-start
    import pyqed.qchem.mcscf.fragment_charge as module
    report = dict(partition='symmetric Lowdin AO',fragments=[list(range(36)),list(range(36,72))],
        checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        seconds=elapsed,states=rows,
        interpretation='Variance is basis dependent and is not an exact CT probability.')
    (args.output/'fragment_charge.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.output/'fragment_charge.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(7,4),layout='constrained')
    x = np.arange(len(rows))
    ax.bar(x-.16,[r['variance_A'] for r in rows],width=.32,label='Monomer A')
    ax.bar(x+.16,[r['variance_B'] for r in rows],width=.32,label='Monomer B')
    ax.axhline(1,color='gray',ls='--',lw=1,label='Ideal symmetric single CT')
    ax.set(xticks=x,xticklabels=[f'S{i}' for i in x],ylabel=r'$\mathrm{Var}(Q_A),\ \mathrm{Var}(Q_B)\ [e^2]$',
        title='Pentacene dimer · SA6-CAS(4e,4o)\nSymmetric Löwdin fragment charge fluctuations')
    ax.legend(frameon=False); fig.savefig(args.output/'pentacene_fragment_charge.png',dpi=180)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
