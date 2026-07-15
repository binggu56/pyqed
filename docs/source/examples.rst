Examples gallery
================

The repository contains many scripts under ``examples/``.  This page puts the
smallest maintained entry points first and labels larger files as research
workflows.  A filename in the inventory does not imply that every option is
validated; consult :doc:`capabilities` before using a result as evidence.

Run source-tree examples from the repository root with ``PYTHONPATH=.``.  The
code shown here is included directly from the tracked files, so the web guide
and executable examples stay synchronized.

Four verified starting points
------------------------------

Native H2 restricted Hartree--Fock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use for:** checking an installation and learning the molecular
model--build--solve pattern.  **Requirements:** the base PyQED installation.
**Typical runtime:** less than a minute on a laptop.

.. literalinclude:: ../../examples/quickstart.py
   :language: python
   :linenos:

Run it with:

.. code-block:: bash

   PYTHONPATH=. python examples/quickstart.py

The final line is approximately ``RHF energy: -1.116759307396 Eh``.  See
:doc:`quickstart` for the explanation and :doc:`qchem` before changing the
molecule or electronic-structure method.

H2O harmonic normal-mode viewer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use for:** calculating a native RHF Hessian and animating all three water
normal modes. **Requirements:** the base PyQED installation; opening the
interactive scene requires a browser that can reach ``https://pyqed.org``.
**Typical runtime:** a few seconds on a laptop. PySCF is not required.

.. literalinclude:: ../../examples/qchem/h2o_normal_modes_viewer.py
   :language: python
   :linenos:

Run it with:

.. code-block:: bash

   PYTHONPATH=. python examples/qchem/h2o_normal_modes_viewer.py

For a terminal or continuous-integration smoke run, add ``--no-browser``. The
example prints the signed harmonic frequencies and still validates the viewer
scene without opening a window. It uses a geometry optimized at the same
RHF/STO-3G level. These frequencies illustrate the API; they are not benchmark
spectroscopic predictions. See :doc:`hf_analysis` for the mode-animation
conventions and Hessian backend constraints.

Sine-DVR harmonic oscillator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use for:** learning grid construction, local potentials, and a known
analytic spectrum.  **Requirements:** the base PyQED installation.
**Typical runtime:** a few seconds.

.. literalinclude:: ../../examples/dvr/sine_harmonic_oscillator.py
   :language: python
   :linenos:

Run it with:

.. code-block:: bash

   PYTHONPATH=. python examples/dvr/sine_harmonic_oscillator.py

Expected output is ``[0.5 1.5 2.5 3.5]``.  The :doc:`dvr` guide explains why
both the box and number of points must be converged for a new potential.

Compact HEOM spin--boson dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use for:** exercising the current one-exponential Drude--Lorentz HEOM path.
**Requirements:** the base installation for this compact solver; the
``pyqed[heom]`` extra is needed for separate DEOM/structured-bath helpers.
**Typical runtime:** less than a minute.

.. literalinclude:: ../../examples/heom_compact.py
   :language: python
   :linenos:

Run it with:

.. code-block:: bash

   PYTHONPATH=. python examples/heom_compact.py

The final line is ``Final <sigma_z>: -0.96907844``.  Read :doc:`heom` before
interpreting the parameter called ``temperature`` or changing the time step,
bath decomposition, or hierarchy depth.

Choose a larger workflow
------------------------

The scripts below are useful next destinations, but many have optional
dependencies, longer runtimes, model-specific inputs, or research interfaces.
Inspect their imports, parameters, data files, and output paths before running
them.

Quantum chemistry
~~~~~~~~~~~~~~~~~

* ``examples/qchem/sa_casscf_factor.py`` -- native factorized,
  state-averaged CASSCF on a small H4 model.
* ``examples/qchem/casscf_factor_vs_dense.py`` -- compare dense and factorized
  CASSCF paths.
* ``examples/qchem/comp2_h2o.py`` -- COMP2 on water.
* ``examples/qchem/gw_qsgw.py`` -- G0W0, eigenvalue-self-consistent GW, and
  qsGW on H2.
* ``examples/qchem/rttdhf_h2_kick_spectrum.py`` -- real-time TDHF kick
  spectrum for H2.

Read :doc:`qchem`, :doc:`backends`, and the relevant method page before
scaling these calculations.

Grid and nonadiabatic dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``examples/dvr/fedvr_vs_sine_quartic.py`` -- compare two grid
  representations for a quartic potential.
* ``examples/dvr/gwp_sddvr_2d_independent_ho.py`` -- Gaussian-wavepacket
  SD-DVR for a two-dimensional oscillator.
* ``examples/dvr/shin_metiu.py`` -- an early Shin--Metiu research script; it
  is not maintained as a first-run example, so inspect and adapt it rather
  than expecting a turnkey smoke test.
* ``examples/namd/ehrenfest.py`` -- model Ehrenfest dynamics.
* ``examples/namd/ldrfg_avoided_crossing.py`` -- an avoided-crossing LDRFG
  workflow.
* ``examples/namd/abinitio_ehrenfest_pyscf.py`` -- an ab initio workflow with
  an optional PySCF backend.

Continue with :doc:`geometric_quantum_dynamics` and :doc:`pyqed.namd`.

Open systems and spectroscopy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``examples/heom.py`` -- the longer historical HEOM exploration script.
* ``examples/deom.py`` -- a dissipaton-equation-of-motion research workflow.
* ``examples/redfield.py`` -- Redfield dynamics.
* ``examples/signals/absorption.py`` -- an absorption-signal calculation.
* ``examples/2DES.py`` -- a larger two-dimensional spectroscopy workflow.

See :doc:`guide/guide_open_dynamics`, :doc:`heom`, and
:doc:`guide/guide_spectroscopy` for the governing assumptions.

Floquet, light--matter, and tensor networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``examples/floquet/two_level_system.py`` -- a driven two-level model.
* ``examples/floquet/RiceMele.py`` -- a Rice--Mele model.
* ``examples/floquet/Floquet_topological_phase_diagram.py`` -- a phase-diagram
  workflow.
* ``examples/test_cavity.py`` -- a cavity-QED smoke script.
* ``examples/qchem/tddmrg_h2_threeway_compare.py`` -- a time-dependent DMRG
  comparison.

Read :doc:`pyqed.floquet`, :doc:`pyqed.polariton`, and :doc:`mps` before
adapting them.

Turn an example into evidence
-----------------------------

Before publishing a result derived from an example:

* pin the PyQED release or full Git commit and record whether the tree changed;
* preserve the exact input, random seed, units, basis or grid, backend, and
  solver tolerances;
* establish convergence with respect to every relevant numerical control;
* compare a small case with an analytic or independent reference; and
* use the :doc:`benchmarks` manifest for validation or performance claims.

If an example fails, report the smallest reproducible command using the
:doc:`support` checklist.
