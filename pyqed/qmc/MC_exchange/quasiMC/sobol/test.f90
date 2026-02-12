program main

!*****************************************************************************80
!
!! MAIN is the main program for SOBOL_PRB.
!
!  Discussion:
!
!    SOBOL_PRB tests the SOBOL library.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license. 
!
!  Modified:
!
!    16 January 2010
!
!  Author:
!
!    John Burkardt

implicit real (kind = 8)(a-h, o-z) 


  integer ( kind = 8 ), parameter :: ndim = 4 

  integer ( kind = 8 ) dim_num
  integer ( kind = 8 ), dimension ( 3 ) :: dim_num_test = (/ 100, 500, 1000 /)
  integer ( kind = 8 ) i
  real ( kind = 8 ), allocatable, dimension ( : ) :: r
  integer ( kind = 8 ), dimension(ndim) ::  seed
  integer ( kind = 8 ) seed_in
  integer ( kind = 8 ) seed_out
  integer ( kind = 8 ) test
      real*8 gasdev 
      real*8, dimension(ndim) :: q,x 

  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) 'TEST13'
  write ( *, '(a)' ) '  I8_SOBOL computes the next element of a Sobol sequence.'
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) '  In this test, we get a few samples at high dimension.'
  write ( *, '(a)' ) '  We only print the first and last 2 entries of each'
  write ( *, '(a)' ) '  sample.'

    dim_num = 2 

      open(100, file = 'random.dat') 
    

    write ( *, '(a)' ) ' '
    write ( *, '(a,i8)' ) '  Using dimension    ', ndim 

    seed = (/1,2,3,4/) 

      seed_in = 0 

      pi = 4.0*atan(1.0) 


!      ns = 100000 ! sampling points 

      print *, 'how many sampling points you want?'
      read(5,*) ns  
 
      z0 = 0d0 
      q = (/0d0,0d0,0d0,0d0/) 

      anm = (1d0/2d0/pi)**2 

    do i = 0, ns 

!      do j=1,ndim 
!        x(j) = gasdev(seed_in) + q(j) 
!      enddo 

      call i8_sobol(ndim, seed_in, x) 
      x = 12.0*(x-0.5d0)

     
      z0 = z0 + x(1)**2*x(2)**2*exp(-x(1)**2/2.0-x(2)**2/2d0-x(3)**2/2d0-x(4)**2/2d0) 
    enddo 


    write ( 6, * )  'integration  = ', z0*anm/dble(ns)*12.0**4






  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) '  Normal end of execution.'
  write ( *, '(a)' ) ' '
  call timestamp ( )

1000 format(2(f14.6,1x))
  stop
end
