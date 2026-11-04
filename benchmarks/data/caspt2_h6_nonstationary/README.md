Fixed-orbital OpenMolcas regression for the Si/Sijr coupling convention.

The decks use OpenMolcas 26.06, C1, H6/STO-3G CAS(2,2), no frozen orbitals,
zero IPEA, and imaginary shifts 0 and 0.25 hartree. Run either deck with
`pymolcas` from this directory. `orbitals.RasOrb` contains the fixed orbitals.

To reconstruct them in PyQED, solve RHF at the coordinates in the decks,
make each MO's largest AO coefficient positive, then rotate columns (0,5)
by `[[cos(0.1),-sin(0.1)],[sin(0.1),cos(0.1)]]` before CASCI. This deliberately
leaves a nonzero inactive/virtual Fock block, unlike a stationary RHF gate.

The independent reference energy is -2.7600130349 hartree. OpenMolcas E2 is
-0.0717435142 (eta=0) and -0.0717229841 (eta=0.25) hartree. The retained rank
is 72. `tests/test_perturber_templates.py` checks these values without requiring
OpenMolcas at test time. The direct projected-Fock equations differ by about
2.5 microhartree; the explicit `ic_convention="molcas"` reproduces the decks.
