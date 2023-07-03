c     QSATS version 1.0 (3 March 2011)

c     file name: cmrg.f

c ----------------------------------------------------------------------

c     this is the subroutine that checks the double precision arithmetic
c     that the pseudo random number generator depends on.

c     both parent and child processes execute this subroutine, but only
c     the parent process reports its findings to standard output.

c ----------------------------------------------------------------------

      subroutine rcheck(myid)

      implicit none

      integer i, j, myid

      double precision rstate(3), z, u, check(16), xx1, xx2, xx3

      double precision zm1, zm2, rm1, rm2

      common /moduli/ zm1, zm2, rm1, rm2

c --- both parent and child processes need to initialize these variables
c     for the pRNG to work.

      zm1=4294967087.0d0
      zm2=4294944443.0d0
      rm1=1.0d0/zm1
      rm2=1.0d0/zm2

      if (myid.gt.0) return

c --- the rest of the subroutine checks the implementation of double
c     precision floating point arithmetic in a variety of ways.

      if (myid.eq.0) write (6, 6000)
6000  format ('checking double precision FP representation for the ',
     +        'MRG32k3a CMRG...'/)

      j=0

      check(1)=5.0d0
      check(2)=4.0d0
      check(3)=7.0d0
      check(4)=5.0d0
      check(5)=5.0d0
      check(6)=0.0d0
      check(7)=1.0d0
      check(8)=4.0d0
      check(9)=4.0d0
      check(10)=9.0d0
      check(11)=9.0d0
      check(12)=3.0d0
      check(13)=5.0d0
      check(14)=5.0d0
      check(15)=7.0d0
      check(16)=6.0d0

      z=2.0d0

      do i=1, 52
         u=z/2.0d0*3.0d0+1.0d0
         z=z*2.0d0
      end do

      do i=1, 16

         z=dmod(u, 10.0d0)

         if (z.ne.check(i)) then

            j=1

            if (myid.eq.0) write (6, 6011) i, z, check(i)
6011        format ('decimation of large floating point number ',
     +              'fails at i = ', i2/,
     +              'result of decimation is ', f3.1,
     +              ' but should be ', f3.1/)

         end if

         u=u-z
         u=u/10.0d0

      end do

      if (myid.eq.0.and.j.eq.0) write (6, 6100)
6100  format ('double precision FP representation is OK'/)

      if (myid.eq.0.and.j.ne.0) write (6, 6110)
6110  format ('problem with double precision FP representation'/)

      if (myid.eq.0) write (6, 6200)
6200  format ('checking CMRG cycle-skipping arithmetic...'/)

      j=0

      rstate(1)=4294967086.0d0
      rstate(2)=4294967086.0d0
      rstate(3)=4294967086.0d0

      call mulmod(rstate(1), 2427906178.0d0, 4294967087.0d0, xx1)
      call mulmod(rstate(2), 3580155704.0d0, 4294967087.0d0, xx2)
      call mulmod(rstate(3),  949770784.0d0, 4294967087.0d0, xx3)

      u=dmod(xx1+xx2+xx3, 4294967087.0d0)

      check(1)=8.0d0
      check(2)=0.0d0
      check(3)=5.0d0
      check(4)=1.0d0
      check(5)=0.0d0
      check(6)=1.0d0
      check(7)=2.0d0
      check(8)=3.0d0
      check(9)=6.0d0
      check(10)=1.0d0

      do i=1, 10

         z=dmod(u, 10.0d0)

         if (z.ne.check(i)) then

            j=1

            if (myid.eq.0) write (6, 6211) i, z, check(i)
6211        format ('decimation of cycle-skipping product ',
     +              'fails at i = ', i2/,
     +              'result of decimation is ', f3.1,
     +              ' but should be ', f3.1/)

         end if

         u=u-z
         u=u/10.0d0

      end do

      if (myid.eq.0.and.j.eq.0) write (6, 6300)
6300  format ('cycle-skipping arithmetic is OK'/)

      if (myid.eq.0.and.j.ne.0) write (6, 6310)
6310  format ('problem with cycle-skipping arithmetic'/)

      return
      end

c ----------------------------------------------------------------------

c     this subroutine generates one uniform pseudo random number and
c     advances the state vector of the random number generator.

c ----------------------------------------------------------------------

      subroutine rstep(rstate, z, rscale)

      implicit none

      integer i

      double precision rstate(8), xx1, z, rscale

      double precision zm1, zm2, rm1, rm2

      common /moduli/ zm1, zm2, rm1, rm2

      xx1 = 1403580.0d0 * rstate(2) - 810728.0d0 * rstate(1)

      rstate(1) = rstate(2)
      rstate(2) = rstate(3)
      rstate(3) = (xx1 - idint(xx1*rm1)*zm1)

      if (rstate(3).lt.0.0d0) rstate(3) = rstate(3) + 4294967087.0d0

      xx1 = 527612.0d0 * rstate(6) - 1370589.0d0 * rstate(4)

      rstate(4) = rstate(5)
      rstate(5) = rstate(6)
      rstate(6) = (xx1 - idint(xx1*rm2)*zm2)

      if (rstate(6).lt.0.0d0) rstate(6) = rstate(6) + 4294944443.0d0

      xx1=rstate(3)-rstate(6)

      z = (xx1 - idint(xx1*rm1)*zm1)

      if (z.le.0.0d0) z=z+4294967087.0d0

      z=z*rscale

      return
      end

c ----------------------------------------------------------------------

c     this subroutine skips to the next stream for the pseudo random
c     number generator.

c ----------------------------------------------------------------------

      subroutine rskip(rstate)

      implicit none

      double precision rstate(6), xx1, xx2, xx3, yy1, yy2, yy3

      call mulmod(rstate(1), 2427906178.0d0, 4294967087.0d0, xx1)
      call mulmod(rstate(2), 3580155704.0d0, 4294967087.0d0, xx2)
      call mulmod(rstate(3),  949770784.0d0, 4294967087.0d0, xx3)

      yy1=dmod(xx1+xx2+xx3, 4294967087.0d0)

      call mulmod(rstate(1),  226153695.0d0, 4294967087.0d0, xx1)
      call mulmod(rstate(2), 1230515664.0d0, 4294967087.0d0, xx2)
      call mulmod(rstate(3), 3580155704.0d0, 4294967087.0d0, xx3)

      yy2=dmod(xx1+xx2+xx3, 4294967087.0d0)

      call mulmod(rstate(1), 1988835001.0d0, 4294967087.0d0, xx1)
      call mulmod(rstate(2),  986791581.0d0, 4294967087.0d0, xx2)
      call mulmod(rstate(3), 1230515664.0d0, 4294967087.0d0, xx3)

      yy3=dmod(xx1+xx2+xx3, 4294967087.0d0)

      rstate(1) = yy1
      rstate(2) = yy2
      rstate(3) = yy3

      call mulmod(rstate(4), 1464411153.0d0, 4294944443.0d0, xx1)
      call mulmod(rstate(5),  277697599.0d0, 4294944443.0d0, xx2)
      call mulmod(rstate(6), 1610723613.0d0, 4294944443.0d0, xx3)

      yy1=dmod(xx1+xx2+xx3, 4294944443.0d0)

      call mulmod(rstate(4),   32183930.0d0, 4294944443.0d0, xx1)
      call mulmod(rstate(5), 1464411153.0d0, 4294944443.0d0, xx2)
      call mulmod(rstate(6), 1022607788.0d0, 4294944443.0d0, xx3)

      yy2=dmod(xx1+xx2+xx3, 4294944443.0d0)

      call mulmod(rstate(4), 2824425944.0d0, 4294944443.0d0, xx1)
      call mulmod(rstate(5),   32183930.0d0, 4294944443.0d0, xx2)
      call mulmod(rstate(6), 2093834863.0d0, 4294944443.0d0, xx3)

      yy3=dmod(xx1+xx2+xx3, 4294944443.0d0)

      rstate(4) = yy1
      rstate(5) = yy2
      rstate(6) = yy3

      return
      end

c ----------------------------------------------------------------------

c     this subroutine uses the Box-Muller algorithm to convert uniform
c     pseudo random numbers into Gaussian deviates.

c ----------------------------------------------------------------------

      subroutine gstep(rstate, z, rscale)

      implicit none

      double precision rstate(8), z, rscale, gsave, r, u, twopi

c --- this is (not surprisingly) two times pi.

      parameter (twopi=6.2831853071795864770d0)

      if (rstate(7).gt.0.0d0) then

         z=rstate(8)
         rstate(7)=-1.0d0

      else

         call rstep(rstate, z, rscale)

         r=sqrt(-2.0d0*dlog(z))

         call rstep(rstate, z, rscale)

         rstate(8)=r*sin(twopi*z)

         z=r*cos(twopi*z)

         rstate(7)=1.0d0

      end if

      return
      end

c ----------------------------------------------------------------------

c     this subroutine computes (a * s) mod m without overflow using an
c     algorithm presented in the online extended version of P. L'Ecuyer
c     et al., Operations Research, volume 50, p. 1073 (2002).

c ----------------------------------------------------------------------

      subroutine mulmod(a, s, zm, z)

      implicit none

      double precision a, s, zm, z, u, abig, asmall, two17

c --- this is 2 raised to the power 17.

      parameter (two17=131072.0d0)

      asmall=dmod(a, two17)
      abig=(a-asmall)/two17

      u=abig*s

      u=dmod(u, zm)*two17+asmall*s

      z=dmod(u, zm)

      return
      end
