Benchmarks and validation
=========================

PyQED distinguishes three kinds of evidence:

Correctness regression
   A small deterministic check that catches a previously observed error or
   enforces an invariant.

Independent validation
   A comparison with an analytic result, independently implemented algorithm,
   external program, or deliberately small dense reference.

Performance benchmark
   A timing or memory measurement tied to a fully described workload and
   environment.

A correctness test is not automatically a performance benchmark, and agreement
for one molecule or model is not a universal accuracy claim.

Reproducibility contract
------------------------

Every published benchmark result should record:

* PyQED version, Git commit, and dirty-tree state;
* exact command and input artifacts;
* Python, dependency, compiler, and optional-backend versions;
* operating system, architecture, CPU, memory, BLAS, and thread settings;
* system geometry/model, basis/grid, charge/spin, solver options, and tolerances;
* warm-up and repeat policy for timing;
* metric units and aggregation statistic; and
* reference implementation, reference version, comparison quantity, and
  acceptance tolerance for validation.

The top-level ``benchmarks/`` directory contains a machine-readable catalog,
JSON Schema, an unpopulated manifest template, and a review template.  Replace
every ``REPLACE_ME`` value before treating a manifest as evidence.

Existing benchmark entry points
-------------------------------

These tracked scripts are useful starting points; their names do not make their
output a release claim until a completed manifest and artifacts are reviewed.

.. list-table::
   :header-rows: 1
   :widths: 28 40 32

   * - Purpose
     - Script
     - Comparison scope
   * - Integral-representation comparison
     - ``examples/qchem/casscf_factor_vs_dense.py``
     - Dense versus factorized CASSCF on one small H4 case
   * - Native spherical integral consumers
     - ``examples/qchem/spherical_consumer_validation.py``
     - Dense-reference accuracy, direct-J/K timing against PySCF, Cartesian
       round-trip timing, RI storage, screening, and native worker scaling
   * - Relativistic mean-field validation
     - ``examples/qchem/benchmark_x2c_rhf_vs_pyscf.py``
     - Selected X2C/RHF cases against PySCF
   * - Solvent mean-field validation
     - ``examples/qchem/benchmark_pcm_rhf_pyscf.py``
     - Selected PCM/RHF cases against PySCF
   * - Spin--orbit validation
     - ``examples/qchem/benchmark_soc_vs_pyscf.py``
     - Selected spin--orbit quantities against PySCF
   * - Molecular GW PES comparison
     - ``examples/qchem/gw/benchmark_pes_molgw.py``
     - Selected molecular GW quantities against MOLGW
   * - Non-Abelian readiness diagnostics
     - ``examples/qchem/benchmark_su2_readiness.py``
     - Component/reduced-path diagnostics; not a general solver claim
   * - Fully contracted CASPT2 validation
     - ``benchmarks/caspt2_openmolcas.py``
     - Matched CASSCF/zero-IPEA comparison with OpenMolcas when available;
       explicitly correlates all inactive orbitals and always writes JSON
       diagnostics and a contraction figure

Run locally
-----------

Run from the repository root and control numerical thread fan-out:

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
   VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
   PYTHONPATH=. python examples/qchem/casscf_factor_vs_dense.py

Store generated caches and heavy outputs outside the repository where
practical.  Preserve the raw output alongside the completed manifest rather
than copying only a rounded value into a table.

Publishing policy
-----------------

A result may appear in a release note or project website only when its command,
manifest, and referenced artifacts are available and the stated scope matches
the evidence.  If hardware, dependency, or input fields are unknown, label the
result preliminary instead of filling them with assumptions.
