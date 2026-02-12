c     QSATS version 1.0 (3 March 2011)

c     file name: rpsend.f

c ----------------------------------------------------------------------

c     this is the subroutine that sends a replica to a child process.

c     errchk is a subroutine called after every MPI subroutine that
c     checks the MPI error code and reports any errors.

c ----------------------------------------------------------------------

      subroutine rpsend(loop, nrep, ntask, MPI_R)
      
      implicit double precision (a-h, o-z)

      include 'mpif.h'

      include 'sizes.h'
      
      include 'qsats.h'

      dimension replic(NATOM6)

      dimension rstate(8)

      parameter (half=0.5d0)
      
      if (nrep.lt.1.or.nrep.gt.NREPS) then
         write (6, *) 'bad replica number in rpsend: nrep = ', nrep
      end if

c --- pack together the old and new atomic positions.

      do i=1, NATOM3
         replic(i)=path(i, nrep)
      end do

c --- this computes the provisional new atomic positions (excluding
c     the gaussian displacements that will be computed by the child
c     process).

      if (nrep.eq.1) then

         do i=1, NATOM3
            replic(i+NATOM3)=path(i, nrep+1)
         end do

      else if (nrep.eq.NREPS) then

         do i=1, NATOM3
            replic(i+NATOM3)=path(i, nrep-1)
         end do

      else

c ------ for interior replicas, the provisional positions are the
c        average of the positions in the two adjacent replicas.

         do i=1, NATOM3
            replic(i+NATOM3)=
     +         half*(path(i, nrep-1)+path(i, nrep+1))
         end do

      end if

c     if (nrep.le.2) then
c        write (60, 9595) loop, 0, 'P', nrep, replic(NATOM3+1)
c595     format (i7, '.', i2.2, 1x, a1, 1x, i2, 2(1x, f15.8))
c        call flush(60)
c     end if

c --- send the loop number to the child process.

      call MPI_SEND(loop,
     +              1,
     +              MPI_INTEGER,
     +              ntask,
     +              0207,
     +              MPI_COMM_WORLD,
     +              ierr)

      call errchk(0, ierr, 100207)

c --- send the coordinates to the child process.

      call MPI_SEND(replic,
     +              NATOM6,
     +              MPI_R,
     +              ntask,
     +              0205,
     +              MPI_COMM_WORLD,
     +              ierr)

      call errchk(0, ierr, 100205)

c --- send the random number generator state vector to the child
c     process.

      do i=1, 8
         rstate(i)=rstatv(i, nrep)
      end do

      call MPI_SEND(rstate,
     +              8,
     +              MPI_DOUBLE_PRECISION,
     +              ntask,
     +              0206,
     +              MPI_COMM_WORLD,
     +              ierr)

      call errchk(0, ierr, 100206)

      return
      end
