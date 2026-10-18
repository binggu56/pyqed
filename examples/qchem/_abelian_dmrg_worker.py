"""Isolated native-integral PyQED MPS worker for the block2 comparison."""
import argparse
import json
import os
import sys
from pathlib import Path
from examples.qchem.benchmark_abelian_dmrg_scaling import model, calculate


def load_shared_state(path, group=None):
    """Read the block2 benchmark's left-coupled reduced singlet tensors."""
    import numpy as np
    from pyqed.symmetry import IrrepTensor
    from pyqed.mps.su2 import SpinChargeSector, SU2Irrep
    from pyqed.mps.nonabelian import MPS
    codes = {0:0}
    if group is not None:
        from pyscf import symm
        codes = {symm.irrep_name2id(group.name, name):
                 sum((int(c)<0)<<i for i,c in enumerate(chars))
                 for name,chars in group.character_table.items()}
    with np.load(path, allow_pickle=False) as packed:
        labels = json.loads(str(packed['labels']))
        sites = []
        for i in range(max(item[0] for item in labels)+1):
            blocks = {tuple(SpinChargeSector(n, SU2Irrep(s), point_group=codes[g])
                            for n,s,g in qs): packed[f'b{j}'].copy()
                      for j,(site,qs) in enumerate(labels) if site == i}
            qns = []
            for axis in range(3):
                dims = {}
                for key, value in blocks.items():
                    sector, dim = key[axis], value.shape[axis]
                    if sector in dims and dims[sector] != dim:
                        raise ValueError('Inconsistent shared-state sector multiplicity.')
                    dims[sector] = dim
                qns.append([sector for sector,dim in dims.items() for _ in range(dim)])
            sites.append(IrrepTensor(data=blocks, qns=qns, dirs=[-1,1,1],
                                     metadata={'physical_basis':'fully_reduced_su2'}))
    return MPS.from_tensors(sites, target_sector=sites[-1].qns[-1][0])


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--norb', type=int, required=True)
    parser.add_argument('--D', type=int, required=True)
    parser.add_argument('--method', choices=('off', 'reduced'), required=True)
    parser.add_argument('--tolerance', type=float, required=True)
    parser.add_argument('--nsweeps', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shared-state', type=Path)
    parser.add_argument('--fixed-sweeps', action='store_true')
    args = parser.parse_args()
    mf, coeff = model(args.norb)
    state = load_shared_state(args.shared_state,
        mf.mol.symmetry_info.group if args.method == 'reduced' else None) if args.shared_state else None
    row = calculate(mf, coeff, args.norb, args.method, args.D, args.tolerance,
                    nsweeps=args.nsweeps, verbose=1, diagnose_symmetry=True,
                    initial_guess=state, fixed_sweeps=args.fixed_sweeps)
    row.update(backend='pyqed-mps-su2', isolated=True)
    row['contraction_settings'] = {key:os.environ.get(key) for key in (
        'PYQED_SU2_OPERATOR_PANEL_CACHE_BYTES', 'PYQED_SU2_OUTPUT_PANEL_CACHE_BYTES',
        'PYQED_SU2_PREPARATION_CACHE_BYTES', 'PYQED_SU2_ENVIRONMENT_CACHE_BYTES',
        'PYQED_SU2_DIRECT_PANEL_SUM', 'PYQED_SU2_DIRECT_BLOCK_BATCH',
        'PYQED_SU2_DIRECT_PRODUCT_CACHE_BYTES')}
    args.output.write_text(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
