API entry points
================

PyQED's API documentation is being curated around supported workflows.  This
page maps common tasks to the corresponding modules and guides; it is not a
promise that every public Python name is stable.

Core objects and operators
--------------------------

The top-level :mod:`pyqed` package exposes lightweight utilities and lazily
loads selected objects.  Common numerical and operator helpers live in
:mod:`pyqed.phys`.  Inspect the source and focused tests for behavior not yet
covered by a guide.

Electronic structure
--------------------

* :mod:`pyqed.qchem` -- molecule construction and solver entry points.
* :mod:`pyqed.qchem.hf` -- Hartree--Fock solvers and analysis.
* :mod:`pyqed.qchem.mp` -- perturbation-theory workflows.
* :mod:`pyqed.qchem.mcscf` -- active-space and orbital-optimization workflows.
* :mod:`pyqed.gw` -- molecular GW and BSE-family workflows.
* :mod:`pyqed.qchem.pbc` -- periodic electronic-structure paths.

Start with :doc:`qchem`, :doc:`hf_analysis`, :doc:`mp2_comp2`,
:doc:`guide/guide_qchem_mcscf`, and :doc:`gw_bse` rather than constructing
objects from signatures alone.

Dynamics and representations
----------------------------

* :mod:`pyqed.dvr` -- discrete and finite-element variable representations.
* :mod:`pyqed.namd` -- nonadiabatic and locally diabatic dynamics.
* :mod:`pyqed.HEOM` -- hierarchical open-system solvers.
* :mod:`pyqed.floquet` -- periodically driven models.
* :mod:`pyqed.polariton` -- light--matter model workflows.
* :mod:`pyqed.mps` -- matrix-product-state and related research methods.
* :mod:`pyqed.protein` -- experimental protein-analysis helpers.

API stability
-------------

An importable name is not automatically a stable public interface.  The
:doc:`capabilities` labels apply to documented workflows, while individual
experimental functions may change without a deprecation period.  A future
stable API will be listed explicitly rather than inferred from module
visibility.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`

.. toctree::
   :hidden:

   pyqed.protein
