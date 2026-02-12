c     QSATS version 1.0 (3 March 2011)

c     file name: main.f

c ----------------------------------------------------------------------

c     this is the main program; it simply detects whether this is the
c     parent node (task id 0) or a child node (task id > 0) and starts
c     the parent or child process, respectively.

c ----------------------------------------------------------------------

      program main

      implicit double precision (a-h, o-z)

      include 'sizes.h'

      include 'qsats.h'

      include 'mpif.h'

      if (myid.eq.0) then

      end if

      MPI_R=MPI_DOUBLE_PRECISION

      call MPI_INIT(ierr)

      call errchk(-1, ierr, 1000)

      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)

      call errchk(-1, ierr, 1001)

      if (mod(NREPS, 2).ne.0) then

         write (6, *) '# of replicas = ', NREPS
         write (6, *) 'this compile-time parameter should be even'
         write (6, *) 'QSATS terminating'

         call MPI_FINALIZE(ierr)

         call errchk(-1, ierr, 1002)

      else

         call rcheck(myid)

         if (myid.eq.0) then

            write (6, 6000)
6000        format ('this is QSATS version 1.0 (3 March 2011)'//,
     +              'QSATS beginning execution'/)

            call tstamp

            write (6, 6001) NREPS, NATOMS, NATOM3, NATOM6, NATOM7,
     +                      NVBINS, RATIO, NIP, NPAIRS
6001        format ('compile-time parameters:'//,
     +              'NREPS =  ', i6/,
     +              'NATOMS = ', i6/,
     +              'NATOM3 = ', i6/,
     +              'NATOM6 = ', i6/,
     +              'NATOM7 = ', i6/,
     +              'NVBINS = ', i6/,
     +              'RATIO  = ', f6.4/,
     +              'NIP    = ', i6/,
     +              'NPAIRS = ', i6/)

            call parent(ierror)

         end if

         if (myid.gt.0) call child(MPI_R)

         call MPI_FINALIZE(ierr)

         call errchk(-1, ierr, 1003)

         if (myid.eq.0.and.ierror.eq.0) then
            write (6, 6002)
6002        format ('normal termination of QSATS')
         end if

         if (myid.eq.0.and.ierror.ne.0) then
            write (6, 6003)
6003        format ('abnormal termination of QSATS')
         end if

      end if

      stop
      end

c ----------------------------------------------------------------------

c     errchk is a subroutine called after every MPI subroutine that
c     checks the MPI error code and halts the program if there is an
c     error.

c     it prints out the task id, the error id, and a checkpoint code,
c     which is keyed to the message type that is being sent or received
c     at that point in the simulation.

c ----------------------------------------------------------------------

      subroutine errchk(itask, ierr, ichkpt)

      if (ierr.ne.0) then

         write (6, 6000) itask, ierr, ichkpt
6000     format ('task ', i4, ' reports ierr ', i4,
     +           ' at checkpoint ', i4)

         call quit

      end if

      return
      end

c ----------------------------------------------------------------------

c     quit is a subroutine used to terminate MPI execution if there is
c     a non-MPI error that forces the program to terminate.

c ----------------------------------------------------------------------

      subroutine quit

      call MPI_ABORT(MPI_COMM_WORLD, ierr)

      if (ierr.ne.0) then
         write (6, 6000) ierr
6000     format ('MPI_ABORT experiences ierr ', i4)
      end if

      stop
      end

