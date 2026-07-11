Development and contribution
============================

The project-level contribution policy lives in `CONTRIBUTING.md
<https://github.com/binggu56/pyqed/blob/main/CONTRIBUTING.md>`_.  It covers local
setup, scientific evidence, benchmark provenance, and the pull-request
checklist.

Choose a contribution
---------------------

Good first contributions are focused documentation corrections, a minimized
bug reproducer, a deterministic regression test, or an example that exercises
one already implemented path.  Discuss broad API changes, dependencies, or
new numerical conventions in an issue before implementation.

Local validation
----------------

Run the smallest relevant test selection:

.. code-block:: bash

   PYTHONPATH=. python -m pytest tests/test_rhf.py -q

For numerical work, restrict BLAS/OpenMP fan-out unless parallel behavior is
the subject of the test:

.. code-block:: bash

   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
   VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
   PYTHONPATH=. python -m pytest tests/test_rhf.py -q

Build documentation with warnings treated as errors:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -W --keep-going -b html docs/source /tmp/pyqed-docs

Project policies
----------------

* `Governance <https://github.com/binggu56/pyqed/blob/main/GOVERNANCE.md>`_
* `Roadmap <https://github.com/binggu56/pyqed/blob/main/ROADMAP.md>`_
* `Code of Conduct
  <https://github.com/binggu56/pyqed/blob/main/CODE_OF_CONDUCT.md>`_
* `Security policy
  <https://github.com/binggu56/pyqed/blob/main/SECURITY.md>`_
* :doc:`capabilities` and :doc:`benchmarks`

Release changes should update ``HISTORY.rst`` and keep package, documentation,
and citation metadata consistent.  Package-index publication happens only
after the release gates pass.
