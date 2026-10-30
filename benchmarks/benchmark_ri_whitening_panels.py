"""Bounded whitening panels with TDI strides; no molecular integral rebuild.

Use separate processes for before/after variants, with BLAS threads set to one.
The anonymous mapping reserves 40 GiB of address space but touches only panels.
"""
import argparse
import json
import mmap
from pathlib import Path
import resource
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
from pyqed import num_workers
from pyqed.qchem import basis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=6430)
    parser.add_argument('--stride', type=int, default=843051)
    parser.add_argument('--widths', type=int, nargs='+', default=[326, 1536])
    args = parser.parse_args()
    rng = np.random.default_rng(45)
    lower = np.empty((args.rank, args.rank), order='F')
    rng.standard_normal(out=lower)
    lower *= 1e-4
    for j in range(args.rank):
        lower[:j, j] = 0
        lower[j, j] = 2.
    records = []
    with threadpool_limits(1):
        for width in args.widths:
            block = rng.normal(size=(args.rank, width))
            size = ((args.rank-1)*args.stride+width)*8
            with mmap.mmap(-1, size) as storage:
                out = np.ndarray((args.rank, width), buffer=storage,
                    strides=(args.stride*8, 8), dtype=float)
                times = []
                for repeat in range(3):
                    start = time.perf_counter()
                    basis._whiten_ri_block(lower, block, 'cholesky', out=out, workers=num_workers())
                    times.append(time.perf_counter()-start)
                from scipy.linalg import solve_triangular
                reference = solve_triangular(lower, block[:, :3], lower=True)
                np.testing.assert_allclose(out[:, :3], reference, atol=2e-12, rtol=2e-12)
                records.append(dict(width=width, seconds=times, median_seconds=float(np.median(times))))
                del out
    report = dict(rank=args.rank, stride=args.stride, workers=num_workers(), panels=records,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
            (2**20 if sys.platform == 'darwin' else 2**10))
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
