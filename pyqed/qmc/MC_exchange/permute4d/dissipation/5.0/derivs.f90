
      subroutine derivs(ndim,ntraj,x,dv,pe)
      use model, only : R0 

      implicit double precision (a-h, o-z)

      real*8, dimension(ndim,ntraj) :: x,dv 
      real*8, dimension(ntraj) :: pe(ntraj)


      akt = 16d0
      akv = 1d0  

      do i=1,ntraj 
!        do j=1,ndim 
         xcm = (x(1,i) + x(2,i) + x(3,i) + x(4,i))/4d0

          dv(1,i) = akt*xcm/4d0 - akv*(x(2,i)-x(1,i)-R0) - &
                    (x(3,i)-x(1,i)-R0)*akv - akv*(x(4,i)-x(1,i)-R0)  

          dv(2,i) = akt*xcm/4d0 + akv*(x(2,i)-x(1,i)-R0)  - &
                    (x(3,i)-x(2,i)-R0)*akv - akv*(x(4,i)-x(2,i)-R0)  
          
          dv(3,i) = akt*xcm/4d0 + akv*(x(3,i)-x(2,i)-R0) + & 
                    (x(3,i)-x(1,i)-R0)*akv - akv*(x(4,i)-x(3,i)-R0) 
      
         dv(4,i) =  akt*xcm/4d0 + akv*(x(4,i)-x(3,i)-R0) +  & 
                  akv*(x(4,i)-x(2,i)-R0) + akv*(x(4,i)-x(1,i)-R0)

          pe(i) = akt*xcm**2/2d0 + akv*((x(2,i)-x(1,i)-R0)**2/2d0 + & 
              (x(3,i)-x(2,i)-R0)**2/2d0 + (x(3,i)-x(1,i)-R0)**2/2d0 + & 
              (x(4,i) - x(1,i) - R0)**2/2d0 + (x(4,i)-x(2,i)-R0)**2/2d0 + & 
              (x(4,i)-x(3,i)-R0)**2/2d0) - R0**2/2d0
!        enddo 
      enddo 

      return
      end subroutine

