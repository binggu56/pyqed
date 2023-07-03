c ----------------------------------------------------------------------

c     this subroutine computes the local energy and potential energy
c     of a configuration.

c ----------------------------------------------------------------------

      subroutine local(q, vloc, dv)

      implicit double precision (a-h, o-z)

      include 'sizes.h'

      include 'qsats.h'
      
      real*8, intent(out) :: vloc

      common /bincom/ bin, binvrs, r2min

      dimension q(NATOM3), dv(NATOM3)

! --- loop over all interacting pairs.
! --- potential for one configuration

      vloc = 0.0d0

      do n = 1, nvpair2

         i = ivpair2(1, n)
         j = ivpair2(2, n)

         dx = -((q(3*j-2))+vpvec2(1, n)+(-q(3*i-2)))
         dy = -((q(3*j-1))+vpvec2(2, n)+(-q(3*i-1)))
         dz = -((q(3*j))  +vpvec2(3, n)+(-q(3*i))  )

         r2 = dx*dx+dy*dy+dz*dz
         r = dsqrt(r2)

         ibin = int((r2-r2min)*binvrs)+1

         if (ibin.gt.0) then

            dr=(r2-r2min)-bin*dble(ibin-1)
            vloc=vloc+v(1, ibin)+v(2, ibin)*dr

         else

            vloc=vloc+v(1, 1)+(r2-r2min)*v(2,1)

         end if
         
      end do
        
! ----- classical force for each DoF (NATOM3)
      dv = 0d0

      do n=1, nvpair2

        i = ivpair2(1,n)
        j = ivpair2(2,n)
         
        dx=-((q(3*j-2))+vpvec2(1, n)+(-q(3*i-2)))
        dy=-((q(3*j-1))+vpvec2(2, n)+(-q(3*i-1)))
        dz=-((q(3*j))  +vpvec2(3, n)+(-q(3*i))  )

        r2=dx*dx+dy*dy+dz*dz
        r = dsqrt(r2)

        ibin=int((r2-r2min)*binvrs)+1
          

        if (ibin.gt.1) then
           dr = (r2-r2min)-bin*dble(ibin-1)
           z = v(2,ibin)+(v(2,ibin+1)-v(2,ibin-1))/2d0*dr*binvrs
      
           dv(3*i-2) = dv(3*i-2)+z*dx*2d0
           dv(3*i-1) = dv(3*i-1)+z*dy*2d0
           dv(3*i)   = dv(3*i)  +z*dz*2d0

           dv(3*j-2) = dv(3*j-2)-z*dx*2d0
           dv(3*j-1) = dv(3*j-1)-z*dy*2d0
           dv(3*j)   = dv(3*j)  -z*dz*2d0

        else
!           dr = (r2-r2min)-bin*dble(ibin-1)
           z = v(2,1)

!           z = 0d0
           dv(3*i-2) = dv(3*i-2)+z*dx*2d0
           dv(3*i-1) = dv(3*i-1)+z*dy*2d0
           dv(3*i)   = dv(3*i)  +z*dz*2d0

           dv(3*j-2) = dv(3*j-2)-z*dx*2d0
           dv(3*j-1) = dv(3*j-1)-z*dy*2d0
           dv(3*j)   = dv(3*j)  -z*dz*2d0
!           write(*,*) "Out of range of r2."
         end if

       enddo         
      



      return
      end subroutine

