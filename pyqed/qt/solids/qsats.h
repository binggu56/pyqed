c     QSATS version 1.0 (3 March 2011)

c     file name: qsats.h

c ----------------------------------------------------------------------

c     this file contains the common blocks used by QSATS.

c ----------------------------------------------------------------------

c --- parameters and counters for the VPI simulation.

      common /monte/  zm, tau, den, scale, amass, ztacc, ztrej,
     +                nph2, nloop, nprint, nacc, nrej, irrst, idebug

c --- trial wave function parameters.

      common /psitri/ aa, bb

c --- random number generator variables.

      double precision zm1, zm2, rm1, rm2, rscale, rstatv

      common /moduli/ zm1, zm2, rm1, rm2

      common /rancm1/ rscale

      common /rancm2/ rstatv(8, NREPS)

c --- potential energy lookup table.

      common /potcom/ v(2, NVBINS)

c --- VPI replicas and atomic masses.

      common /vpi/    path(NATOM3, NREPS),
     +                pathnu(NATOM3, NREPS),
     +                zmass(NATOM3)

c --- filenames.

      character*16 spfile, svfile, ltfile

      common /files/  spfile, svfile, ltfile

c --- description of the crystal lattice.

      common /crystl/ xtal(NATOMS, 3)

      common /box/    xlen, ylen, zlen, dxmax, dymax, dzmax

c --- arrays dealing with interacting pairs.

      common /vpairs/ vpvec(3, NPAIRS),
     +                vpvec2(3, NPAIRS2),
     +                ivpair(2, NPAIRS),
     +                ivpair2(2, NPAIRS2),
     +                ipairs(NIP, NATOMS),
     +                npair(NATOMS),
     +                nvpair,nvpair2

c --- counters to monitor load balancing.

      common /parcom/ iwork(60)

