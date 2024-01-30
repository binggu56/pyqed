      subroutine long_force(q, vlong, dvl)

      implicit real*8(a-h,o-z)

      include 'sizes.h'

      include 'qsats.h'
      
      common /bincom/ bin, binvrs, r2min


      real*8, intent(in)  :: q(NATOM3)
      real*8, intent(out) :: vlong, dvl(NATOM3)

      h2k = 3.157746d5

      vlong = 0d0
      do i=1,NATOMS
        x = q(3*i-2)
        y = q(3*i-1)
        z = q(3*i)

        vlong = vlong  - 0.147941E+01           -
     +           0.278253E-02*q(3*i-2)          + 
     +           0.112328E-03*q(3*i-1)          +   
     +           0.382942E-04*q(3*i)            -
     +           0.290524E-01*q(3*i-2)**2       -
     +           0.288602E-01*q(3*i-1)**2       -
     +           0.237969E-01*q(3*i)**2         + 
     +           0.239032E-03*q(3*i-2)*q(3*i-1) -
     +           0.780090E-05*q(3*i-1)*q(3*i)   +   
     +           0.103676E-05*q(3*i-2)*q(3*i) 

        dvl(3*i-2) = -0.278253e-2-0.581048e-1*x  + 0.239032e-3*y +
     +                0.103676e-5*z 
        dvl(3*i-1) =  0.112328e-3-0.577204e-1*y  + 0.239032e-3*x - 
     +                0.780090e-5*z
        dvl(3*i)    =  0.382942e-4-0.475938e-1*z - 0.780090e-5*y +
     +                0.103676e-5*x
      enddo

c --- energy units conversion K ---> Hartree

      vlong = vlong/h2k
      dvl = dvl/h2k

      return
      end subroutine

