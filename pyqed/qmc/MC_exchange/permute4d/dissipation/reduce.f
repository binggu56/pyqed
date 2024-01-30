c ---- reduce the double-counting of nvpairs

      subroutine reduce()

      implicit double precision (a-h,o-z)

      include 'sizes.h'
      include 'qsats.h'
       
!      integer*4, intent(out) :: nvpair2
!      character*20  DES

      common /bincom/ bin, binvrs, r2min
      
!      print *,DES
      nvpair2 = 0

      do n=1,nvpair
        
        i = ivpair(1,n)
        j = ivpair(2,n)

        if(i < j) then

          nvpair2 = nvpair2+1

          ivpair2(1,nvpair2) = i
          ivpair2(2,nvpair2) = j
 
          vpvec2(1, nvpair2) = vpvec(1,n)
          vpvec2(2, nvpair2) = vpvec(2,n)
          vpvec2(3, nvpair2) = vpvec(3,n)

        endif
!        write(*,*) 'nvpair2=',nvpair2
      enddo

      return
      end subroutine

        
