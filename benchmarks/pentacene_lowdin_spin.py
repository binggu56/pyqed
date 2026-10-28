"""Reanalyze saved pentacene roots with the common Lowdin spin/charge partition."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from pyqed.qchem.mcscf.local_spin import local_spin, local_spin_from_rdms, _lowdin_coefficients
from pyqed.qchem.mcscf.fragment_charge import fragment_charge_from_rdms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    d = np.load(args.checkpoint)
    core,active = int(d['ncore']),int(d['ncas'])
    mc = SimpleNamespace(converged=True,ci=d['ci'],binary=d['binary'],ncore=core,ncas=active,
        mo_coeff=d['mo_coeff'],mol=SimpleNamespace(overlap=d['overlap']),mf=SimpleNamespace())
    groups = [np.flatnonzero(d['ao_atom_indices']<36),np.flatnonzero(d['ao_atom_indices']>=36)]
    t = _lowdin_coefficients(d['overlap'],d['mo_coeff'][:,:core+active])
    rows = []
    start = time.perf_counter()
    for state in range(6):
        row = dict(state=state)
        for name,group in zip(('A','B'),groups):
            row['spin_'+name] = local_spin(mc,aolst=group,state_id=state)
            expected = local_spin_from_rdms(d['dm1_active'][state],d['dm2_active'][state],t[group].T@t[group],ncore=core)
            np.testing.assert_allclose(row['spin_'+name],expected,atol=1e-10)
        row['total_spin'] = local_spin(mc,state_id=state)
        np.testing.assert_allclose(row['total_spin'],d['spin_square'][state],atol=1e-9)
        rows.append(row)
    elapsed = time.perf_counter()-start
    for state,row in enumerate(rows):
        charges = fragment_charge_from_rdms(d['dm1_active'][state],d['dm2_active'][state],d['overlap'],d['mo_coeff'],groups,ncore=core,nuclear_charges=[146,146])
        row.update(variance_A=float(charges['variance'][0]),variance_B=float(charges['variance'][1]))
    import importlib
    module = importlib.import_module('pyqed.qchem.mcscf.local_spin')
    report = dict(partition='symmetric Lowdin AO',checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),spin_seconds=elapsed,states=rows)
    (args.output/'lowdin_spin.json').write_text(json.dumps(report,indent=2)+'\n')
    with (args.output/'lowdin_spin.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key,label in zip(axes,('spin','variance'),(r'$\langle S_A^2\rangle,\langle S_B^2\rangle\ [\hbar^2]$',r'$\mathrm{Var}(Q_A),\mathrm{Var}(Q_B)\ [e^2]$')):
        for offset,name in zip((-.17,.17),('A','B')):
            ax.bar(np.arange(6)+offset,[r[key+'_'+name] for r in rows],width=.34,label='Monomer '+name)
        ax.set(xticks=np.arange(6),xticklabels=[f'S{i}' for i in range(6)],ylabel=label)
    axes[0].legend(frameon=False)
    fig.suptitle('Pentacene dimer · SA6-CAS(4e,4o) · common Löwdin partition')
    fig.savefig(args.output/'pentacene_lowdin_spin_charge.png',dpi=180)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
