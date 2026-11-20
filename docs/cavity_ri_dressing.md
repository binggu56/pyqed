# Fixed-photon-coordinate RI-CASSCF

`Polariton.dressed_states` accepts molecules built with `eri="ri"` as well
as packed `eri="s8"`. Use `use_cholesky=True` in a CASSCF factory for the
factorized orbital optimization and CI transformations. The solver option
name covers RI factors as well as Cholesky factors.

The length-gauge, single-mode, long-wavelength Pauli–Fierz dressing follows
Rokaj et al., J. Phys. B **51**, 034005 (2018),
https://doi.org/10.1088/1361-6455/aa9c99. This is a fixed-coordinate adaptation:
photon kinetic energy is omitted from electronic surfaces; the photon
harmonic potential is returned separately. Nuclear dipole scalar terms are
included in the returned electronic energy offset.

The molecular Coulomb integrals retain the auxiliary-basis RI approximation.
The two-electron dipole self-energy is appended as one exact factor
`coupling * position`, in the same full or packed-pair storage as the RI
factors. No four-index AO tensor is constructed, and the input molecule is
unchanged. RI accuracy depends on the auxiliary basis; agreement with exact
Coulomb-integral CASSCF is not guaranteed.

For a photon-coordinate scan, `dressed_states` prepares the coordinate-independent
position integrals, dipole self-energy, and augmented RI factors once. Each
coordinate receives a separate one-electron Hamiltonian while sharing the
unchanged two-electron storage. Orbital optimization and state averaging still
run independently at every coordinate; this does not introduce an additional
Hamiltonian approximation.

## Restarting the LiF photon DVR

`examples/qchem/lif_photon_dvr_restart.py --checkpoints INPUT --output OUTPUT`
rechecks saved SA4 orbitals and CI vectors without solving an intermediate
RHF problem. `INPUT/scan.npz` supplies dimensionless `x`; `state_XX.npz`
contains the corresponding PyQED-AO `mo_coeff` and root-by-determinant `ci`.
Orbitals must be orthonormal in the current PyQED AO basis.
The restart is specific to the documented LiF geometry, aug-cc-pVQZ orbital
basis, def2-universal-jkfit auxiliary basis, and cavity parameters. Do not
reuse checkpoints from a different model. No RHF convergence is claimed.

The sine-DVR box is `[-3.25, 3.25]/sqrt(omega)` with 25 interior nodes and
unit photon mass. Each node is independently reoptimized to an orbital
gradient tolerance of 1e-5. `ldr_inputs.npz` stores physical `Q`, electronic
energies (nuclear offset included), separate photon potential, and full
pairwise electronic overlaps in DVR order. Add the photon potential once
when forming the LDR potential.
