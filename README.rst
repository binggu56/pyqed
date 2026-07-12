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

For the current development tree:

.. code-block:: bash

   git clone https://github.com/binggu56/pyqed.git
   cd pyqed
   python -m pip install -e .

Standard installations use the tested pure-Python fallbacks.  To compile the
optional native accelerators from a source checkout, use a supported C/C++
toolchain and opt in explicitly:

.. code-block:: bash

   PYQED_BUILD_EXTENSIONS=1 python -m pip install .

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
   mol.build(driver="builtin", eri="auto")
   mf = mol.RHF().run()

   print("converged:", mf.converged)
   print("RHF energy:", mf.e_tot)

Continue with the `quickstart
<https://docs.pyqed.org/en/latest/quickstart.html>`_ and the `examples gallery
<https://docs.pyqed.org/en/latest/examples.html>`_.

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
commit used.  The archived software has two DOI forms:

* all PyQED versions: https://doi.org/10.5281/zenodo.21316543;
* exact v0.2.0 archive: https://doi.org/10.5281/zenodo.21316544.

Use the version-specific DOI when citing v0.2.0, and use the all-versions DOI
when you want a citation that resolves to the latest archived release.  Also
cite the project paper:

  Yujuan Xie, Xiaotong Zhu, and Bing Gu, “PyQED: A Python Framework for
  *Ab Initio* Geometric Quantum Dynamics,” *Chinese Journal of Chemical
  Physics* (2026), https://doi.org/10.1063/1674-0068/cjcp2510161.

The article DOI is distinct from both software-archive DOIs.  See the
`citation guide <https://docs.pyqed.org/en/latest/citing.html>`_ for
method-citation and reproducibility guidance.

License
-------

PyQED is distributed under the MIT License.  See ``LICENSE``.
