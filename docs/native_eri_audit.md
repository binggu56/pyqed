# Native ERI Audit

This note classifies current `mol.eri` / `mf.eri` consumers by whether they
truly require a dense AO four-index ERI tensor, already support low-rank
factors, or can be migrated with a local refactor.

## Summary

Dense AO ERIs are not required everywhere.

The current codebase already has one complete factorized path:

- `pyqed/qchem/hf/rhf.py`
- `pyqed/qchem/mcscf/casci.py`
- `pyqed/qchem/mcscf/direct_ci.py`

The main remaining dense-ERI requirements are:

- RHF/UHF convenience methods that explicitly return full AO/MO ERI tensors
- AO J/K implementations outside the RHF low-rank path
- CIS / LR-TDHF-style code that contracts `(ov|ov)` and `(oo|vv)` blocks from `mol.eri`
- older FCI / DMRG / MPS model builders that take a full 4-index tensor as their primary Hamiltonian input

## Categories

### Already factor-capable

These paths already accept `eri_factors` or can work through the AO Cholesky
representation instead of a dense AO ERI tensor.

- `pyqed/qchem/hf/rhf.py:354-373`
  `RHF.get_veff()`, `get_j()`, and `get_jk()` already dispatch through
  `eri_factors`.
- `pyqed/qchem/hf/rhf.py:825-858`
  `get_jk()` implements factorized J/K directly.
- `pyqed/qchem/hf/rhf.py:751-822`
  `get_or_build_low_rank_eri_factors()` caches AO factors and now accepts
  factor-only native builds.
- `pyqed/qchem/mcscf/casci.py:217-280`
  `transform_spatial_eri_to_mo(..., use_cholesky=True)` transforms AO factors
  to MO pair factors and assembles the required MO ERIs without `mf.eri`.
- `pyqed/qchem/mcscf/casci.py:637-701`
  `CASCI.get_SO_matrix()` already supports `use_cholesky=True`.
- `pyqed/qchem/mcscf/direct_ci.py:280-305`
  `transform_active_space_spatial_integrals()` already supports `use_cholesky=True`.
- `pyqed/qchem/mcscf/direct_ci.py:1296-1332`
  `direct_ci.CASCI.get_SO_matrix()` already supports `use_cholesky=True`.

### Dense AO ERI required as written

These functions or modules directly contract a full AO ERI tensor and do not
currently have a factorized alternative.

- `pyqed/qchem/hf/uhf.py:169-206`
  UHF AO J/K and AO->MO transforms use `mol.eri` directly.
- `pyqed/qchem/dft/scf.py:31-42`
  AO DFT J/K helpers use direct contractions against `mol.eri`.
- `pyqed/qchem/lrtddft.py:67-90`
  Builds `eri_iajb` / `eri_ijab` from the full AO ERI tensor.
- `pyqed/qchem/ci/cis.py:299-307`
  Builds `(ov|ov)` and `(oo|vv)` from `mf.mol.eri`.
- `pyqed/qchem/ci/fci.py:48-110`
  Uses `mf.mol.eri` as the AO ERI source before AO->MO transformation.
- `pyqed/qchem/mcscf/casci.py:728-730`
  `CASCI.qubitization()` still uses `self.mf.eri`.
- `pyqed/qchem/mcscf/direct_ci.py:1472-1474`
  `direct_ci.CASCI.qubitization()` still uses `self.mf.eri`.
- `pyqed/qchem/dmrg/dmrg.py:635-638`
  DMRG Hamiltonian build takes a dense 4-index tensor.
- `pyqed/mps/fermion.py:161-168`, `395-405`, `663-668`
  MPS fermion model stores and slices a dense 4-index tensor.

### Dense full ERI only for convenience APIs

These are not fundamental algorithmic blockers, but they currently expose
full tensors by contract.

- `pyqed/qchem/hf/rhf.py:174-198`
  `RHF.get_eri()` returns full AO or MO ERIs.
- `pyqed/qchem/hf/rhf.py:200-236`
  `RHF.get_eri_so()` materializes a full spin-orbital tensor.
- `pyqed/qchem/hf/rhf.py:406-433`
  `RHF.get_eri_mo()` does a dense AO->MO transform from `self.eri`.

These methods should remain available, but they do not need to be on the
critical path for SCF/CASSCF.

### Not AO-dense-builtin blockers

These modules use full 4-index tensors, but not specifically because the
builtin AO builder must provide `mol.eri`.

- `pyqed/qchem/tdscf/tdhf.py:151-161`
  Uses PySCF `ao2mo` to form full spin-orbital ERIs.
- `pyqed/gw/gw.py:286-306`
  Uses PySCF `ao2mo` to form full MO/spin-orbital ERIs.
- `pyqed/gw/bse.py:68-75`
  Same pattern as `gw.py`.

These are downstream full-integral consumers, but they are not blocked on
the native `mol.build()` path producing `mol.eri` specifically.

### Out of scope for the AO builtin ERI discussion

These files use `eri`, but not as a molecular AO four-index electron-repulsion
tensor in the standard Gaussian AO sense.

- `pyqed/qchem/dvr/*`
- `pyqed/qchem/gdvr/*`

Those `eri` objects are DVR/GDVR interaction matrices or reshaped lattice-like
objects, not the same API question as builtin AO `mol.eri`.

## Recommended refactor order

### Priority 1

Remove dense AO ERI from the main SCF/CASSCF critical path completely.

- Keep using factorized J/K in `pyqed/qchem/hf/rhf.py`
- Prefer `use_cholesky=True` paths in CASCI/CASSCF/direct-CI entry points
- Avoid calling convenience methods that force `self.eri` materialization

### Priority 2

Add factorized AO->MO block builders for response / CI code.

- `pyqed/qchem/lrtddft.py`
- `pyqed/qchem/ci/cis.py`
- `pyqed/qchem/ci/fci.py`

These modules mostly need specific MO blocks such as `(ov|ov)` or `(oo|vv)`,
which can be formed from transformed pair factors without ever building the
full AO ERI tensor.

### Priority 3

Refactor optional full-tensor helpers to compute on demand from factors.

- `RHF.get_eri_mo()`
- `RHF.get_eri_so()`
- `UHF.get_eri_mo()`

These can keep their public behavior while using `eri_factors` internally when
available.

### Priority 4

Rework old DMRG / MPS / qubitization code only if those workflows matter.

- `pyqed/qchem/dmrg/dmrg.py`
- `pyqed/mps/fermion.py`
- `pyqed/qchem/mcscf/*qubitization*`

Those APIs are currently tensor-centric by construction, so migrating them is a
larger design change than the SCF / CASCI / direct-CI work.

## Practical conclusion

The codebase does not need dense AO ERIs globally.

What it needs is a cleaner split:

- dense ERI only for tensor-centric legacy consumers
- factorized ERI as the default for SCF and active-space workflows
- on-demand dense reconstruction only at API boundaries that explicitly ask for it
