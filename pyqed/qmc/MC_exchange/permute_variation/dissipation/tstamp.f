c     QSATS version 1.0 (3 March 2011)

c     file name: tstamp.master

c ----------------------------------------------------------------------

c     this subroutine is used to provide a time-stamp for the most
c     recent compilation of the code.

c ----------------------------------------------------------------------

      subroutine tstamp

      write (6, 6000)
6000  format ('this code was compiled on ',
     +        'MACHNAME'/,
     +        'compilation date = '
     +        'COMPDATE'/)

      return
      end

