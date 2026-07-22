Nonadiabatic Molecular Dynamics
===============================

The :mod:`pyqed.namd` package contains adiabatic and diabatic wavepacket
dynamics utilities.

Main source modules:

* ``pyqed.namd.adiabatic``
* ``pyqed.namd.diabatic``
* ``pyqed.namd.eckart``
* ``pyqed.namd.gmat``
* ``pyqed.namd.liquid_ldr``

Liquid-phase LDR helpers live in ``pyqed.namd.liquid_ldr``.  The module
contains analytic liquid-driven LDR propagation, solvent-embedded CASCI LDR
snapshots, Berry/no-Berry geometric diagnostics, hot-spot ranking, convergence
checks, and readiness gates.  See :doc:`geometric_quantum_dynamics` for the
end-to-end workflow and example commands.

This page is static because several legacy NAMD helper files are not currently
valid importable Python modules, which makes autodoc unsuitable for RTD.
