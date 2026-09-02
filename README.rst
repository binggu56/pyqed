PyQED
=====

.. image:: https://img.shields.io/pypi/v/pyqed.svg
   :target: https://pypi.org/project/pyqed/
   :alt: PyPI release

.. image:: https://readthedocs.org/projects/pyqed/badge/?version=latest
   :target: https://docs.pyqed.org/
   :alt: Documentation status

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/binggu56/pyqed/blob/main/LICENSE
   :alt: MIT license

PyQED is open-source research software for light--matter interactions,
quantum dynamics, spectroscopy, open quantum systems, and electronic-structure
workflows.  The project combines reusable Python APIs with executable research
examples and validation paths.

* Project website: https://pyqed.org/
* Documentation: https://docs.pyqed.org/
* Source: https://github.com/binggu56/pyqed
* Issue tracker: https://github.com/binggu56/pyqed/issues

Project status
--------------

PyQED is active research software.  Interfaces and numerical workflows have
different maturity levels, and some are experimental.  Consult the
`capability status <https://docs.pyqed.org/en/latest/capabilities.html>`_ before
selecting a method for production work.  A passing example or benchmark is
evidence for the documented case, not a guarantee for every model or
parameter regime.

Installation
------------

Install a published release from PyPI:

.. code-block:: bash

   python -m pip install pyqed

For the current ``bg`` development branch:

.. code-block:: bash

   git clone --branch bg https://github.com/binggu56/pyqed.git
   cd pyqed
   python -m pip install -e .

Published releases provide platform wheels containing the production qchem
accelerators. Source installs build the same required accelerators by default
and therefore need a supported C/C++ toolchain. Verify an installation with:

.. code-block:: bash

   python -m pyqed.qchem.check_install

The slow pure-Python integral implementation is an explicit reference/debug
mode; see the installation guide for its source-build and runtime settings.

Use documentation from the same release or commit as the installed code.  See
the `installation guide <https://docs.pyqed.org/en/latest/installation.html>`_
for verification and optional-dependency guidance.

Five-minute calculation
-----------------------

This native, small-basis RHF calculation does not require PySCF:

.. code-block:: python

   from pyqed.qchem import Molecule

   mol = Molecule(
       atom="H 0 0 0; H 0 0 0.74",
       unit="angstrom",
       basis="sto-3g",
   )
   mol.build(eri="auto")
   mf = mol.RHF().run()

   print("converged:", mf.converged)
   print("RHF energy:", mf.e_tot)

Continue with the `quickstart
<https://docs.pyqed.org/en/latest/quickstart.html>`_ and the `examples gallery
<https://docs.pyqed.org/en/latest/examples.html>`_.

Lattice-gauge pilot
-------------------

``pyqed.lgt`` contains a one-dimensional Schwinger-model pilot with compact
quantum links and Wilson-line-dressed Fourier-DVR hopping.  The alternating
matter/link MPS represents every local Gauss law as a component of one
additive vector quantum number, so the symmetric DMRG calculation needs no
finite penalty.  Reproduce the ED/MPS comparison and scaling figures with:

.. code-block:: bash

   PYTHONPATH=. python examples/lgt/alternating_wilson_dvr_mps.py

The channel-targeted real-time calculation extracts ``M_V`` and ``M_S``
without a multiroot state-averaged sweep:

.. code-block:: bash

   PYTHONPATH=. python examples/lgt/channel_targeted_mv_ms_mps.py

The Gauss-resolved MPO is rounded sector by sector before DMRG/TDVP, avoiding
the redundant sum-of-products bond without mixing gauge sectors.  For the
larger ``N=7`` physical-ED flux-convergence reference, run:

.. code-block:: bash

   PYTHONPATH=. python examples/lgt/dynamical_schwinger_dvr.py --npts 7

Development
-----------

Run focused tests from the repository root:

.. code-block:: bash

   PYTHONPATH=. python -m pytest tests/test_rhf.py -q

Build the documentation with warnings treated as errors:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -W --keep-going -b html docs/source /tmp/pyqed-docs

Read `CONTRIBUTING.md
<https://github.com/binggu56/pyqed/blob/main/CONTRIBUTING.md>`_ before proposing
changes.  Scientific changes should include a focused test or reproducible
benchmark and document units, conventions, dependencies, and references.

Citing PyQED
------------

Use the metadata in ``CITATION.cff`` and record the exact PyQED release or Git
commit used.  No project DOI is asserted until one appears in an archived
release.  See the `citation guide
<https://docs.pyqed.org/en/latest/citing.html>`_ for a reproducibility
checklist.

License
-------

PyQED is distributed under the MIT License.  See ``LICENSE``.
