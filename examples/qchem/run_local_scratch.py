"""Run a calculation with scratch and relative outputs outside OneDrive.

Usage: PYTHONPATH=. python examples/qchem/run_local_scratch.py /path/to/calculation.py
"""
import argparse
import os
from pathlib import Path
import runpy
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('script', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = args.script.resolve(strict=True)
    scratch = Path(os.environ.get('PYQED_SCRATCH_DIR', '/private/tmp/pyqed-runs')).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['PYQED_SCRATCH_DIR'] = str(scratch)
    os.environ['TMPDIR'] = str(scratch)
    tempfile.tempdir = str(scratch)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    # Keep repository imports available after moving the working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *args.arguments]
    os.chdir(scratch)
    print(f'LOCAL_SCRATCH {scratch}', flush=True)
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
