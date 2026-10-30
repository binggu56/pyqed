"""Measure occupied RI J/K on TDI-sized panels, without a molecular rebuild.

BLAS must use one thread. Reference source, if supplied, defines
_contract_jk_ri_occ_packed and is only an explicitly selected timing baseline.
"""
import argparse
import importlib.util
import json
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
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--extension', type=Path)
    parser.add_argument('--auxiliary', type=int, default=16)
    args = parser.parse_args()
    if args.extension:
        spec = importlib.util.spec_from_file_location('pyqed.qchem._integrals_cpp', args.extension)
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        basis._integrals_cpp = extension
    function = basis._contract_jk_ri_occ_packed
    if args.reference:
        namespace = dict(basis.__dict__)
        exec(args.reference.read_text(), namespace)
        function = namespace['_contract_jk_ri_occ_packed']
    rng = np.random.default_rng(12)
    n, o, p = 1298, 271, args.auxiliary
    factors = rng.normal(size=(p, n*(n+1)//2))
    coeff = rng.normal(size=(n, o))
    occ = np.full(o, 2.)
    times = []
    with threadpool_limits(1):
        for repeat in range(3):
            start = time.perf_counter()
            j, k = function(factors, coeff, occ, n, workers=num_workers())
            times.append(time.perf_counter()-start)
    report = dict(nao=n, nocc=o, naux=p, workers=num_workers(), seconds=times,
        median_seconds=float(np.median(times)),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
            (2**20 if sys.platform == 'darwin' else 2**10))
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
