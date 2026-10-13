TDI GW/BSE — 16 threads
======================

The script reads the included tdi.xyz and runs density-fitted RHF, full-space
Padé G0W0, and ten singlet full-BSE excitations with QP-gap screening.
It uses 6-31G / cc-pVDZ-JKFIT, all 798 orbitals, and the default GW/BSE
convergence thresholds. Stability checks remain enabled: rejected GW roots
stop the calculation rather than silently falling back or restricting the space.

On the cluster, activate an environment with PySCF 2.14 and matplotlib, then
build the current PyQED from the repository root::

    PYQED_GW_OPENMP=required PYQED_BSE_OPENMP=required PYQED_CONTOUR_OPENMP=required python -m pip install -e .

Submit from the repository root (replace the scratch path and supply your
cluster's partition/account options as needed)::

    export TDI_RUN_ROOT=/scratch/your-user/pyqed-runs
    sbatch examples/qchem/tdi_gw_bse/submit.sbatch

Set THREADS and OUTPUT at the top of run.py. THREADS defaults to 16;
OUTPUT defaults to tdi-results beneath the working directory (the job scratch
directory when using the launcher). An absolute OUTPUT path is also accepted.

Or run inside an existing 16-CPU allocation::

    PYTHONPATH=. python examples/qchem/tdi_gw_bse/run.py

The output directory must be empty. The launcher requests 128 GB RAM and
24 hours; allow at least 150 GB scratch. The 64 GB working-array budget in
run.py is not a hard memory cap, and the time request is not a runtime estimate.
No automatic restart is implemented. TDI_PYTHON can select an interpreter;
TDI_REPO can override the source path. Rebuild extensions on Linux rather
than copying macOS binaries.

Outputs: scf.chk and df.h5 preserve the reference; gw.npz stores available
GW/continuation arrays even if GW fails; bse.npz contains converged excitation
energies in eV and X/Y amplitudes. gw_bse_diagnostics.png plots the converged
results. Failures exit nonzero and appear in the Slurm log. See
../../../docs/source/gw_bse.rst for method references and fidelity.

Geometry: 112-atom TDI FC frame 906, q38=0, q232=0, in Angstrom, extracted
from the saved 2026-10-12 input. Original input SHA256:
0f81ce7373b925632c381ae1e0e3a312bed34594bec7328313b5f1c3350271b0.
