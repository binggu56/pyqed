PyQED documentation
===================

PyQED is open-source research software for light--matter interactions,
quantum dynamics, spectroscopy, open quantum systems, and electronic-structure
workflows.  These pages are organized around tasks: install the package, run a
small calculation, choose a method, inspect its evidence, and reproduce a
result.

.. important::

   PyQED is active research software.  APIs and numerical paths have different
   maturity levels.  Check :doc:`capabilities` and the limitations on each
   method page before using a workflow in production research.

Start here
----------

* :doc:`installation` -- create an isolated environment and verify it.
* :doc:`quickstart` -- run a native H2 restricted Hartree--Fock calculation.
* :doc:`tutorials` -- follow a task-oriented learning path.
* :doc:`examples` -- find an executable repository example.
* :doc:`capabilities` -- understand Beta and Experimental status.
* :doc:`benchmarks` -- reproduce validation and performance evidence.
* :doc:`citing` -- cite the exact code and method used.

Method areas
------------

The development tree contains workflows for quantum chemistry, discrete and
finite-element variable representations, nonadiabatic and geometric dynamics,
open-system dynamics, spectroscopy, Floquet models, light--matter coupling,
and tensor-network methods.  The presence of a module does not by itself mean
that every configuration is supported; the capability table links each area
to its current documentation, examples, and tests.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   tutorials
   examples

.. toctree::
   :maxdepth: 2
   :caption: Concepts and user guides

   theory
   guide/guide
   backends
   qchem
   hf_analysis
   mp2_comp2
   gw_bse
   tddft_ehrenfest
   mps
   dvr
   geometric_quantum_dynamics
   pyqed.floquet
   pyqed.models
   pyqed.namd
   pyqed.polariton
   heom

.. toctree::
   :maxdepth: 2
   :caption: API and implementation

   api
   qchem_architecture
   nonabelian_dmrg_design

.. toctree::
   :maxdepth: 2
   :caption: Evidence and citation

   capabilities
   benchmarks
   citing

.. toctree::
   :maxdepth: 2
   :caption: Develop and get help

   development
   support
   developers

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
