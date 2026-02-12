
      module sci

        implicit real*8(a-h,o-z)

        real*8, public, parameter  :: pi = 4.d0*atan(1.d0)

        complex*16, public, parameter :: im=(0d0,1d0)

        real (kind = 8), parameter :: half = 0.5d0
        real (kind = 8), parameter :: one = 1d0
        real (kind = 8), parameter :: au2k = 3.2d5


        integer*4 :: nb ! number of basis in the linear fitting
      end module

! --- parameters for Monte Carlo integration 
module Monte

      implicit none

      real (kind = 8)  :: eSum, eSqdSum, envSum, enkSum, envpSum, enkpSum, rDpSum 
      integer*4 ::  nAccept

      integer (kind = 4), parameter :: nx = 400 ! only used for one-dimensional problem 

      integer (kind = 8) idum
      real (kind = 8) xmin, xmax, dx 
      real (kind = 8) delta  
       
      integer (kind = 8), parameter :: ndim = 2 

end module  

! --- parameters associated with AQP calculation 
      
      module AQP 
      
      use Monte, only : ndim 
      
      implicit none 
      
      real (kind = 8), dimension(ndim,ndim) :: rb 
      real (kind = 8), dimension(ndim) :: ra, rc, rd 
      
      save 
      end module 
      
