Installation
============

Use an isolated Python environment and keep the documentation version aligned
with the installed PyQED release or Git commit.

Published release
-----------------

Create and activate a virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip

On Windows PowerShell, activate it with ``.venv\Scripts\Activate.ps1``.
Then install the release from PyPI:

.. code-block:: bash

   python -m pip install pyqed

The PyPI release can lag behind the development branch.  Before following a
``latest`` documentation page, check which distribution was installed:

.. code-block:: bash

   python -c "from importlib.metadata import version; print(version('pyqed'))"

Current development tree
------------------------

To work from source:

.. code-block:: bash

   git clone https://github.com/binggu56/pyqed.git
   cd pyqed
   python -m pip install -e .

The default build uses the tested pure-Python fallbacks.  To compile the
optional native accelerators, install from a source checkout with a supported
C/C++ toolchain and opt in explicitly:

.. code-block:: bash

   PYQED_BUILD_EXTENSIONS=1 python -m pip install .

Native integral extensions use link-time optimization by default. Set
``PYQED_LTO=0`` only when diagnosing a toolchain that cannot link LTO objects.
For a machine-local build, ``PYQED_NATIVE_CPU=1`` also enables the compiler's
host-specific SIMD instruction set; do not use that option for redistributable
wheels.

The native DMRG Davidson backend detects OpenMP automatically.  Linux builds
normally use ``-fopenmp``.  On macOS, install Homebrew ``libomp`` or provide a
runtime with ``PYQED_OPENMP_PREFIX``.  Automatic discovery prefers Homebrew
over the active Conda environment because an older Conda runtime can be placed
first on the Python extension search path.  Both the install-time SU(2) kernel
and runtime-built dense Davidson kernel link the selected runtime by its full
path so they cannot silently select different ``libomp.dylib`` files.
``PYQED_MPS_OPENMP=0`` disables OpenMP, while ``PYQED_MPS_OPENMP=1`` makes a
missing runtime a build error.  Runtime thread selection is explicit through
``DMRG(..., n_threads=N)`` rather than solely through ``OMP_NUM_THREADS``.

For a redistributable macOS wheel, build against a ``libomp`` whose deployment
target covers the supported macOS versions and bundle or repair that runtime
as part of the wheel.  A machine-local Homebrew path is intentionally local to
that build host.

Record the commit for any research result:

.. code-block:: bash

   git rev-parse HEAD
   git status --short

The second command records whether local modifications were present.

Verify the native path
----------------------

Run this small calculation after installation:

.. code-block:: bash

   python -c "from pyqed.qchem import Molecule; m=Molecule(atom='H 0 0 0; H 0 0 0.74', unit='angstrom', basis='sto-3g'); m.build(eri='auto'); x=m.RHF().run(); print(x.converged, x.e_tot)"

The command should finish with a converged flag and a finite total energy.  If
it fails, include the command and full output in a support request.

Optional dependencies
---------------------

The base installation does not enable every example or backend.  A method page
identifies its optional requirements where known.  Common categories include:

* PySCF for independent comparisons and selected external-backend workflows;
* plotting or visualization packages for figure-generating examples;
* electronic-structure or molecular-dynamics programs used by comparison
  scripts; and
* a C/C++ toolchain for optional accelerated extensions when building from
  source.

Do not install every optional package merely to run the quickstart.  Select a
workflow first, then add only its dependencies.

Documentation environment
-------------------------

Build the documentation from a source checkout with:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -W --keep-going -b html docs/source /tmp/pyqed-docs

Troubleshooting checklist
-------------------------

When an import or calculation fails, record:

* ``python --version``;
* ``python -m pip show pyqed``;
* the operating system and architecture;
* the exact example or input;
* optional backend and dependency versions; and
* the complete traceback.

See :doc:`support` for the issue template and :doc:`capabilities` for known
maturity boundaries.
