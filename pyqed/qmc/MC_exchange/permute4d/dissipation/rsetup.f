c     QSATS version 1.0 (3 March 2011)

c     file name: rsetup.f

c ----------------------------------------------------------------------

c     this subroutine initializes the pseudo random number generators
c     for the replicas.  it also initializes the value of the rscale
c     variable, which is needed to convert integer pseudo random
c     numbers, which are the raw output of the generators, to floating
c     point pseudo random numbers.

c ----------------------------------------------------------------------

      subroutine rsetup

      implicit double precision (a-h, o-z)

      include 'sizes.h'

      include 'qsats.h'

      dimension rseed(6)

      rscale=1.0d0/4294967088.0d0

      write (6, 6000) 
6000  format ('INITIALIZING random number seeds'/)

      do i=1, 6
         rseed(i)=12345.0d0
      end do

      do i=1, NREPS

         do j=1, 6
            rstatv(j, i)=rseed(j)
         end do

         rstatv(7, i)=-1.0d0
         rstatv(8, i)=0.0d0

         call rskip(rseed)

      end do

      if (idebug.ge.3) then

         write (6, 6001)
6001     format  ('rstatv(1) values:'/)

         do i=1, NREPS
            write (6, 6100) i, rstatv(1, i)
6100        format (i5, 1x, f20.1)
         end do

         write (6, *) ''

      end if

      return
      end
