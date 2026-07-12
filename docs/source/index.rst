.. meta::
   :description: Official PyQED documentation for quantum chemistry, nonadiabatic dynamics, open quantum systems, spectroscopy, and tensor-network methods.

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
* :doc:`guide/guide` -- choose a scientific task from the complete user guide.
* :doc:`tutorials` -- follow a task-oriented learning path.
* :doc:`examples` -- find an executable repository example.
* :doc:`capabilities` -- understand Beta and Experimental status.
* :doc:`benchmarks` -- reproduce validation and performance evidence.
* :doc:`citing` -- cite the exact code and method used.

User guide
----------

The :doc:`PyQED user guide <guide/guide>` is the main map of the documentation.
It groups existing material into foundations, electronic structure, quantum
dynamics, open systems, light--matter models, and tensor networks.  Start with
:doc:`how PyQED calculations work <guide/core_workflow>` if you are moving
between method families: it explains the shared model--build--solve--validate
workflow without making you learn every module first.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   guide/guide
   tutorials
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference and evidence

   api
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
