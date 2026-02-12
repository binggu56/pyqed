c     QSATS version 1.0 (3 March 2011)

c     file name: vinit.f

c ----------------------------------------------------------------------

c     this subroutine sets up the arrays for linear interpolation of
c     the potential energy function (as a function of the squared
c     interatomic distance).

c ----------------------------------------------------------------------

      subroutine vinit(r2min, bin)

      implicit double precision (a-h, o-z)

      parameter (one=1.0d0)

c --- hartree to kelvin conversion factor

      parameter (hart=315774.65d0)

      include 'sizes.h'

      include 'qsats.h'
!     original
      r2min=9.0d0 
!      r2min = 4.d0
      bin=0.05d0

      write (6, 6000) sqrt(r2min), bin
6000  format ('DEFINING potential energy grid parameters'//,
     +        'minimum R      = ', f10.5, ' bohr'/,
     +        'R**2 bin size  = ', f10.5, ' bohr**2'/)

      write (6, 6020)
6020  format ('using HFD-B(He) potential energy curve'/, '[R.A. ',
     +        'Aziz et al., Mol. Phys. vol. 61, p. 1487 (1987)]'/)

c --- evaluate the potential energy function at grid points in R**2.

      do i=1, NVBINS
         r2=dble(i-1)*bin+r2min
         v(1, i)=hfdbhe(r2)
      end do

c --- compute the slopes of the line segments connecting the grid
c     points.

      do i=1, NVBINS-1
         v(2, i)=(v(1, i+1)-v(1, i))/bin
      end do        
!      do i=2,NVBINS-1
!         v(3, i)=(v(1, i+1)+v(1, i-1)-2d0*v(1,i))/bin**2
!      end do

c --- debugging output.

      if (idebug.ge.3) then

         vmin=v(1, 1)

         do i=1, NVBINS
            if (v(1, i).lt.vmin) then
               vmin=v(1, i)
               r2=dble(i-1)*bin+r2min
            end if
         end do

         vmin=vmin*hart

         write (6, 6100) vmin, r2, sqrt(r2)
6100     format ('minimum is ', f12.5, ' K at R**2 = ', f10.5,
     +           ' bohr**2 or R = ', f10.5, ' bohr'/)

      end if

      return
      end

c ----------------------------------------------------------------------

c     this function computes the He-He potential energy as a function
c     of the squared interatomic distance.

c     the potential energy function is described in detail by R.A. Aziz
c     et al., Molecular Physics, volume 61, p. 1487 (1987).

c ----------------------------------------------------------------------

      double precision function hfdbhe(r2)

      implicit real*8 (a-h, o-z)

c --- parameters for the HFD(B) He-He potential energy function

      parameter (astar=1.8443101d5)
      parameter (alstar=10.43329537d0)
      parameter (bestar=-2.27965105d0)
      parameter (d=1.4826d0)
      parameter (c6=1.36745214d0)
      parameter (c8=0.42123807d0)
      parameter (c10=0.17473318d0)

      parameter (rm=5.59926d0)
      parameter (eps=10.948d0)

c --- hartree to kelvin conversion factor

      parameter (hart=315774.65d0)

      r=sqrt(r2)

      x=r/rm

      vstar=astar*exp(-alstar*x+bestar*x**2)

      vd=c6/x**6+c8/x**8+c10/x**10

      if (x.lt.d) vd=vd*exp(-(d/x-1.0d0)**2)

      hfdbhe=(vstar-vd)*eps/hart

      return
      end
