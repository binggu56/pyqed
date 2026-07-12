Citing PyQED
============

Software citation and method citation serve different purposes.  Cite the
exact PyQED artifact used, and separately cite the primary references for the
scientific methods used in the calculation.

Software citation
-----------------

The repository provides ``CITATION.cff``.  GitHub's **Cite this repository**
control can render that metadata.  A reproducible citation should include:

* the authors listed in ``CITATION.cff``;
* the software title, PyQED;
* the exact released version, or the full Git commit for unreleased work;
* the repository URL, https://github.com/binggu56/pyqed; and
* the access or release year required by the target citation style.

Project paper
-------------

In addition to the exact software release or commit, cite the PyQED project
paper:

  Yujuan Xie, Xiaotong Zhu, and Bing Gu, “PyQED: A Python Framework for
  *Ab Initio* Geometric Quantum Dynamics,” *Chinese Journal of Chemical
  Physics* (2026), https://doi.org/10.1063/1674-0068/cjcp2510161.

The DOI above identifies the journal article, not an archived software
release.  No software DOI is asserted in the repository at present.  If a
future release is deposited in an archive, use the DOI shown by that specific
archived release; do not reuse the article DOI.

Method citations
----------------

The software citation does not replace citations for RHF, CASSCF, HEOM, DVR,
GW/BSE, LDR, tensor-network, or other scientific methods.  Consult the method
guide, example header, and linked method documentation for the references relevant
to the exact algorithm and approximation.  When a page does not yet state the
method reference clearly, open an issue rather than guessing.

Reproducibility record
----------------------

Archive with a publication:

* ``python -m pip show pyqed`` output or the Git commit;
* whether the source tree had local modifications;
* exact input files and random seeds;
* Python and dependency versions;
* units, model/geometry, basis or grid, boundary conditions, and spin/symmetry;
* solver options and convergence tolerances; and
* a completed :doc:`benchmarks` manifest for quantitative validation or
  performance claims.

Contributors and derivative works
---------------------------------

Preserve the MIT license notice when redistributing covered source.  Credit
individual contributors and external programs whose work materially supports
a result; repository authorship is not a substitute for those acknowledgments.
