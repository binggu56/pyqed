      subroutine long_force(q, vlong, dvl)

      implicit real*8(a-h,o-z)

      include 'sizes.h'

      include 'qsats.h'
      
      common /bincom/ bin, binvrs, r2min


      real*8, intent(in)  :: q(NATOM3)
      real*8, intent(out) :: vlong, dvl(NATOM3)

      real*8 :: a(10)

      h2k = 3.157746d5

      a(1) = -0.189998e+01
      a(2) =  -0.421551E-02
      a(3) = 0.177780E-03
      a(4) = 0.603591E-04
      a(5) =  -0.417010E-01
      a(6) =  -0.414189E-01  
      a(7) = -0.341826E-01   
      a(8) = 0.363379E-03  
      a(9) = -0.124138E-04   
      a(10) = 0.224261E-05

      vlong = 0d0
      do i=1,NATOMS
        x = q(3*i-2)
        y = q(3*i-1)
        z = q(3*i)

        vlong = vlong + a(1) + 
     +           a(2)*q(3*i-2)          + 
     +           a(3)*q(3*i-1)          +   
     +           a(4)*q(3*i)            +
     +           a(5)*q(3*i-2)**2       +
     +           a(6)*q(3*i-1)**2       +
     +           a(7)*q(3*i)**2         + 
     +           a(8)*q(3*i-2)*q(3*i-1) +
     +           a(9)*q(3*i-1)*q(3*i)   +   
     +           a(10)*q(3*i-2)*q(3*i) 

!        vlong = vlong  - 0.147941E+01           -
!     +           0.278253E-02*q(3*i-2)          + 
!     +           0.112328E-03*q(3*i-1)          +   
!     +           0.382942E-04*q(3*i)            -
!     +           0.290524E-01*q(3*i-2)**2       -
!     +           0.288602E-01*q(3*i-1)**2       -
!     +           0.237969E-01*q(3*i)**2         + 
!     +           0.239032E-03*q(3*i-2)*q(3*i-1) -
!     +           0.780090E-05*q(3*i-1)*q(3*i)   +   
!     +           0.103676E-05*q(3*i-2)*q(3*i)


        dvl(3*i-2) = a(2)+2d0*a(5)*x + a(8)*y + a(10)*z 
        dvl(3*i-1) = a(3)+2d0*a(6)*y + a(8)*x + a(9)*z
        dvl(3*i)   = a(4)+2d0*a(7)*z + a(9)*y + a(10)*x
      enddo

c --- energy units conversion K ---> Hartree

      vlong = vlong/h2k
      dvl = dvl/h2k

      return
      end subroutine

