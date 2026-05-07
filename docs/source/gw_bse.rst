GW and BSE
==========

PyQED provides a native dense molecular GW/BSE workflow in :mod:`pyqed.gw`.
The implementation is designed as a transparent reference backend for small
and medium molecules: it is useful for comparing against PySCF and MOLGW,
testing new approximations, and building neutral excited-state potential
energy surfaces.

The recommended workflow is:

.. code-block:: python

   mf = RHF(mol).run()
   gw = GW(mf).run()
   bse = BSE(gw).run(nroots=5)

Mean-field, GW, and BSE have distinct roles:

* ``RHF`` builds the closed-shell reference, orbitals, and SCF total energy.
* ``GW`` computes quasiparticle energies and screening information.
* ``BSE`` computes neutral excitation energies from the GW/RPA reference.

Basic Example
-------------

.. code-block:: python

   from pyqed.qchem import Molecule
   from pyqed.qchem.hf import RHF
   from pyqed.gw.gw import GW
   from pyqed.gw.bse import BSE, TDA

   mol = Molecule(
       atom="H 0 0 0; H 0 0 0.74",
       basis="sto-3g",
       unit="angstrom",
   )
   mol.build(driver="builtin", eri="dense")

   mf = RHF(mol).run()
   gw = GW(mf, screening="TDH", eta=1e-3).run()

   bse = BSE(gw).run(nroots=3)
   tda = TDA(gw).run(nroots=3)

   print("SCF total energy:", mf.e_tot)
   print("GW quasiparticle energies:", gw.e_qp)
   print("BSE excitation energies:", bse.e)
   print("TDA excitation energies:", tda.e)

The :class:`~pyqed.gw.gw.GW` object stores quasiparticle energies in
``gw.e_qp``.  The older name ``gw.egw`` is kept as a compatibility alias.
For GW only, ``gw.e`` mirrors ``gw.e_qp``.  For BSE and TDA, ``bse.e`` and
``tda.e`` are excitation energies, so quasiparticle input energies live in
``bse.e_qp`` and ``tda.e_qp``.

GW Flavors
----------

The main entry point is :class:`pyqed.gw.gw.GW`.  It currently supports
restricted closed-shell references and dense/factorized molecular integrals.

Available methods include:

* ``GW(mf).run(method="g0w0")`` or ``GW(mf).g0w0()`` for one-shot GW.
* ``GW(mf).evgw(update_screening=False)`` for eigenvalue-only ``GnW0``.
* ``GW(mf).evgw(update_screening=True)`` for eigenvalue-only ``GnWn``.
* ``GW(mf).qsgw()`` for a dense quasiparticle self-consistent reference path.
* ``SCGW(mf).run()`` for an experimental dense imaginary-axis scGW prototype.

``GW.run()`` returns the GW object so that downstream code can pass it directly
to BSE.  It still behaves like the quasiparticle-energy array in common NumPy
contexts:

.. code-block:: python

   gw = GW(mf).run()

   qp = gw.e_qp
   qp_array = np.asarray(gw)
   homo = gw[nocc - 1]
   qp_ev = gw * 27.211386245988

Experimental scGW Prototype
---------------------------

``pyqed.gw.scgw.SCGW`` is a small finite-basis imaginary-axis prototype for
self-consistent GW.  It stores full matrix Green's functions, polarizabilities,
screened interactions, and correlation self-energies on a symmetric imaginary
frequency grid.  Two modes are available:

* ``scgw0``: update ``G`` and ``Sigma`` while keeping the initial screened
  interaction ``W0`` fixed.
* ``scgw``: update ``G``, ``P``, ``W``, and ``Sigma`` every macroiteration.

Both modes can optionally rebuild the bare-exchange part from the current
Green's-function density matrix.

.. code-block:: python

   from pyqed.gw.scgw import SCGW

   scgw0 = SCGW(mf, nfreq=17, wmax=20.0).scgw0(
       max_cycle=20,
       conv_tol=1e-6,
       damping=0.2,
   )

   scgw = SCGW(mf, nfreq=17, wmax=20.0).scgw(
       max_cycle=20,
       conv_tol=1e-6,
       damping=0.2,
   )

   print(scgw.converged)
   print(scgw.mu, scgw.nelec)
   print(scgw.e_qp)       # static imaginary-axis diagnostic estimate
   print(scgw.G.shape)    # (nfreq, nso, nso)

The same functionality is also exposed through the normal GW driver:

.. code-block:: python

   from pyqed.gw.gw import GW

   gw0 = GW(mf).scgw0(nfreq=17, wmax=20.0, max_cycle=20)
   gw = GW(mf).scgw(nfreq=17, wmax=20.0, max_cycle=20)

   print(gw0.scgw_result.W0 is not None)
   print(gw.scgw_result.info["update_screening"])

When the mean-field object carries ``mol.eri_factors``/``mf.eri_factors``,
the prototype keeps ``P`` and ``W`` in the auxiliary factor space instead of
expanding the four-index ERI tensor.  Dense integrals are still supported for
small reference calculations.

This is not yet a production scGW implementation.  In particular, analytic
continuation and Galitskii-Migdal/Luttinger-Ward total energies are still
future work.  The current chemical-potential control fixes the electron count
with a tail-corrected Matsubara sum of the interacting Green's function.

scGW Theory
-----------

Self-consistent GW solves Hedin's equations with the vertex set to one,
``Gamma = 1``.  In an orthonormal molecular-orbital basis, the central Dyson
equation is

.. math::

   G(i\omega_n) =
   \left[
     (\mu + i\omega_n) I - h_0 - \Sigma_x - \Sigma_c(i\omega_n)
   \right]^{-1}.

Here ``h0`` is the one-particle Hamiltonian with the mean-field potential
removed, ``Sigma_x`` is the static exchange self-energy, and ``Sigma_c`` is the
dynamic correlation self-energy.

The independent-particle polarizability is built from the interacting Green's
function:

.. math::

   P_{pq,rs}(i\nu_m)
   =
   -\frac{1}{\beta}
   \sum_n
   G_{pr}(i\omega_n + i\nu_m)
   G_{sq}(i\omega_n).

The screened Coulomb interaction follows a Dyson-like equation:

.. math::

   W(i\nu_m) = v + v P(i\nu_m) W(i\nu_m)
             = \left[1 - vP(i\nu_m)\right]^{-1} v.

The GW correlation self-energy is then

.. math::

   \Sigma^c_{pq}(i\omega_n)
   =
   -\frac{1}{\beta}
   \sum_m
   G_{rs}(i\omega_n - i\nu_m)
   W^c_{pr,qs}(i\nu_m),

where ``W_c = W - v``.  The total self-energy is

.. math::

   \Sigma(i\omega_n) = \Sigma_x + \Sigma_c(i\omega_n).

The chemical potential is adjusted to conserve particle number.  The density
matrix is obtained from the interacting Green's function using a high-frequency
tail correction:

.. math::

   \gamma =
   \frac{1}{2} I
   +
   \frac{1}{\beta}
   \sum_n
   \left[
     G(i\omega_n) - \frac{I}{i\omega_n}
   \right],

and ``mu`` is solved so that

.. math::

   N = \mathrm{Tr}\,\gamma.

The current PyQED prototype uses dense tensors and numerical interpolation on
finite imaginary-frequency grids.  It is therefore a reference implementation
for algorithm development, not yet a high-accuracy production scGW solver.

BSE and TDA
-----------

The preferred BSE API takes a completed GW object:

.. code-block:: python

   gw = GW(mf).run()
   bse = BSE(gw).run(nroots=5)
   tda = TDA(gw).run(nroots=5)

Equivalently, use the convenience constructors on the GW object:

.. code-block:: python

   bse = gw.bse().run(nroots=5)
   tda = gw.tda().run(nroots=5)

``BSE`` solves the full Bethe-Salpeter eigenproblem and stores stacked
``X/Y`` amplitudes in ``bse.xy``.  The views ``bse.x`` and ``bse.y`` return
the excitation and de-excitation blocks.  ``TDA`` solves the Tamm-Dancoff
approximation and stores amplitudes in ``tda.x`` only.

For direct BSE calculations that follow the common MOLGW convention of using
HF/gKS orbital-energy differences instead of prior GW quasiparticle energies,
set ``use_qp=False``:

.. code-block:: python

   bse = BSE(gw).run(nroots=5, use_qp=False)
   tda = TDA(gw).run(nroots=5, use_qp=False)

Potential Energy Surfaces
-------------------------

For a neutral excited-state PES from BSE, use a consistent ground-state
reference at every geometry.  The practical default is:

.. math::

   E_0(R) = E_\mathrm{SCF}(R)

.. math::

   E_n(R) = E_\mathrm{SCF}(R) + \Omega_n^\mathrm{BSE}(R)

where ``mf.e_tot`` is the SCF total energy and ``bse.e[n]`` is the neutral BSE
excitation energy.

If an RPA-correlated ground-state reference is desired, use the same offset
for all excited states:

.. math::

   E_0(R) = E_\mathrm{SCF}(R) + E_c^\mathrm{RPA}(R)

.. math::

   E_n(R) = E_0(R) + \Omega_n^\mathrm{BSE}(R)

PyQED exposes this as:

.. code-block:: python

   gw = GW(mf).run()
   e0_rpa = gw.total_energy(method="rpa")
   bse = BSE(gw).run(nroots=3)
   excited_pes = e0_rpa + bse.e

The quasiparticle energies ``gw.e_qp`` should not be used directly as neutral
ground-state or excited-state PES energies; they correspond to charged
addition/removal quasiparticle levels.

Scanner Interface
-----------------

``BSE`` and ``TDA`` provide an ``as_scanner()`` helper for PES scans.  The
default scanner return value is ``[E0, E0 + Omega_1, ...]`` using the SCF
ground-state reference:

.. code-block:: python

   gw = GW(mf).run()
   bse = BSE(gw).run(nroots=3)

   scanner = bse.as_scanner(nroots=3)
   energies = scanner(new_coords)

   e0 = energies[0]
   excited = energies[1:]

For excitation energies only:

.. code-block:: python

   omega_scanner = bse.as_scanner(nroots=3, energy="excitation")
   omega = omega_scanner(new_coords)

For an RPA-shifted PES:

.. code-block:: python

   rpa_scanner = bse.as_scanner(nroots=3, energy="rpa")
   energies = rpa_scanner(new_coords)

After each call, the scanner stores the latest objects as ``scanner.mf``,
``scanner.gw``, and ``scanner.bse``.

Wavefunction Overlaps
---------------------

BSE and TDA objects can compute overlaps between excitation vectors at
different geometries:

.. code-block:: python

   gw1 = GW(mf1).run()
   gw2 = GW(mf2).run()

   tda1 = TDA(gw1).run(nroots=3, return_vectors=True)
   tda2 = TDA(gw2).run(nroots=3, return_vectors=True)
   overlap_tda = tda1.wavefunction_overlap(tda2)

   bse1 = BSE(gw1).run(nroots=3, return_vectors=True)
   bse2 = BSE(gw2).run(nroots=3, return_vectors=True)
   overlap_bse = bse1.wavefunction_overlap(bse2)

This is useful for following states along a PES and diagnosing state flips.

Integral Backends
-----------------

GW/BSE can use native dense integrals or factorized/RI inputs from the
mean-field reference:

.. code-block:: python

   mol.build(driver="builtin", eri="ri", auxbasis="cc-pvdz-rifit")
   mf = RHF(mol).run(cholesky_jk=True)
   gw = GW(mf).run()
   bse = BSE(gw).run(nroots=5)

When available, AO Cholesky or RI factors are transformed to MO pair factors.
This avoids storing the full four-index MO tensor in the GW self-energy and
low-rank BSE/TDA paths.  The dense reference solvers are still intended for
small and medium molecules.

Validation Notes
----------------

The GW/BSE smoke tests currently cover:

* ``G0W0`` against PySCF exact-frequency GW.
* ``GnW0``, ``GnWn``, and ``qsGW`` against MOLGW reference data.
* dense and factorized integral consistency.
* dense and low-rank BSE/TDA consistency.
* same-geometry BSE/TDA overlap identities.

Related Examples
----------------

* ``examples/qchem/gw_qsgw.py``
* ``examples/qchem/gw_bse_ri.py``
* ``examples/qchem/gw/wavefunction_overlap.py``
* :doc:`qchem`
* :doc:`examples`
