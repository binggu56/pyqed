c     QSATS version 1.0 (3 March 2011)

c     file name: input.f

c ----------------------------------------------------------------------

c     this inputs the names of various I/O files and also reads in the
c     parameters for the simulation.

c ----------------------------------------------------------------------

      subroutine input

      implicit double precision (a-h, o-z)

      include 'sizes.h'

      include 'qsats.h'

      character*8 inword

c --- read in filenames.

      read (5, 5000, err=922) spfile
5000  format (a16)
      read (5, 5000, err=923) svfile
      read (5, 5000, err=924) ltfile

c --- set debug level.

      read (5, 5001, err=931) inword
5001  format (a8)

      if (inword.eq.'NONE') then

         idebug=0

      else if (inword.eq.'MINIMAL') then

         idebug=1

      else if (inword.eq.'LOW') then

         idebug=2

      else if (inword.eq.'MEDIUM') then

         idebug=3

      else if (inword.eq.'HIGH') then

         idebug=4

      else

         write (6, *) 'invalid debug level'

      end if

c --- read in the simulation parameters.

      read (5, *, err=901) tau
      read (5, *, err=902) den
      read (5, *, err=903) amass
      read (5, *, err=904) aa
      read (5, *, err=905) bb
      read (5, *, err=906) nloop
      read (5, *, err=907) nprint

      write (6, 6000) NATOMS, NREPS
6000  format ('REPEATING input parameters'//,
     +        'atom count    = ', i6/,
     +        'replica count = ', i6/)

      write (6, 6001) tau, den, amass, aa, bb
6001  format ('tau             = ', f14.7, ' au time'/,
     +        'density         = ', f14.7, ' atoms per cubic bohr'/,
     +        'atomic mass     = ', f14.7, ' electron masses'/,
     +        'alpha parameter = ', f14.7, ' bohr**(-2)'/,
     +        'B parameter     = ', f14.7, ' bohr'/)

      write (6, 6002) nloop, nprint
6002  format ('number of simulation steps = ', i8/,
     +        'snapshot interval          = ', i8/)

      return

901   write (6, *) 'error reading time step value'
      goto 999

902   write (6, *) 'error reading density value'
      goto 999

903   write (6, *) 'error reading atomic mass value'
      goto 999

904   write (6, *) 'error reading aa value'
      goto 999

905   write (6, *) 'error reading bb value'
      goto 999

906   write (6, *) 'error reading nloop value'
      goto 999

907   write (6, *) 'error reading nprint value'
      goto 999

921   write (6, *) 'error reading RNG file name'
      goto 999

922   write (6, *) 'error reading snapshot file name'
      goto 999

923   write (6, *) 'error reading save file name'
      goto 999

924   write (6, *) 'error reading lattice file name'
      goto 999

931   write (6, *) 'error reading debug level'
      goto 999

932   write (6, *) 'error reading RNG initialization mode'
      goto 999

999   call quit

      return
      end

