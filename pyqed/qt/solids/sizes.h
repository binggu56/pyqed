!     QSATS version 1.0 (3 March 2011)

!     file name: sizes.h

! --- number of replicas or beads in the VPI polymer chain.

      parameter (NREPS=430)

! --- number of atoms in the system.

      parameter (NATOMS=180)

! --- various multiples of NATOMS.

      parameter (NATOM3=NATOMS*3)
      parameter (NATOM6=NATOMS*6)
      parameter (NATOM7=NATOMS*7)

! --- number of points on the interatomic potential energy curve, for
!     linear interpolation of the potential energy function.

      parameter (NVBINS=20000)

! --- "radius" of the interacting-pair region, in multiples of the
!     nearest-neighbor distance.

      parameter (RATIO=2.05)

! --- number of interacting pairs for each atom.

      parameter (NIP=56)

! --- total number of interacting pairs in the simulation box.

      parameter (NPAIRS=NATOMS*NIP)
      parameter (NPAIRS2=NATOMS*NIP/2)
