"""Audit saved AC roots against direct full-frequency contour self-energies."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from pyscf import lib
from threadpoolctl import threadpool_limits
from gw_bse_matched_scaling import restore
from pyqed.gw.gw import GW
from pyqed.gw.contour import ContourDeformation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--orbs', type=int, nargs='+', required=True)
    p.add_argument('--eta', type=float, default=1e-3)
    p.add_argument('--nw', type=int, default=100)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--blas-contractions', action='store_true')
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mf, _ = restore(args.input)
    qp = np.load(args.input/'PyQED-8-1.npz')['qp']
    static = np.load(args.input/'PyQED-8-1-fit.npz')['static']
    lib.num_threads(args.threads)
    rows = []
    with threadpool_limits(limits=args.threads, user_api='blas'):
        gw = GW(mf, freq_int='ac', threads=args.threads)
        gw.eta = args.eta
        if args.blas_contractions:
            gw.threads = 1
        start = time.perf_counter()
        contour = ContourDeformation(gw, mf.mo_energy, args.orbs, nw=args.nw)
        print('Prepared contour screening in',time.perf_counter()-start,'seconds',flush=True)
        for orbital in args.orbs:
            start = time.perf_counter()
            value, derivative = contour.evaluate(orbital, qp[orbital])
            row = dict(orbital=orbital, qp=float(qp[orbital]), eta=args.eta,
                residual=float(qp[orbital]-mf.mo_energy[orbital]-value.real-static[orbital]),
                sigma_real=float(value.real), slope=float(derivative.real),
                nw=args.nw, threads=args.threads, blas_contractions=args.blas_contractions,
                seconds=time.perf_counter()-start)
            rows.append(row)
            args.output.write_text(json.dumps(rows, indent=2)+'\n')
            print(row, flush=True)


if __name__=='__main__':
    main()
