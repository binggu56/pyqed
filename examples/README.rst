PyQED examples
==============

The examples are executable programs, not test fixtures or output folders.  Start
with the small, deterministic programs below, then move to the subsystem folders
for larger calculations.

Good first examples
-------------------

* ``quickstart.py`` — native H2 restricted Hartree--Fock calculation.
* ``dvr/fedvr_harmonic_oscillator.py`` — one-dimensional FEDVR eigenproblem.
* ``namd/ehrenfest_histories.py`` — Shin--Metiu Ehrenfest dynamics.
* ``signals/absorption.py`` — absorption spectrum of a four-level model.

Run an example from the repository root so it imports the working tree::

   PYTHONPATH=. python examples/quickstart.py

Output discipline
-----------------

Examples should print a compact numerical summary and, when practical, state the
expected result in their module docstring.  Generated figures, checkpoints, and
large arrays belong in ``examples/_outputs/`` (ignored by Git) or in an explicit
directory outside the repository, such as ``/private/tmp/pyqed-runs``.  Do not
write generated data next to source files.

Contributing an example
-----------------------

Use a descriptive, machine-independent filename and keep the default calculation
small enough for a modest personal computer.  Prefer deterministic inputs, avoid
commented-out alternative programs, list optional dependencies near the top, and
link heavier research workflows from documentation rather than making them the
first example users encounter.
