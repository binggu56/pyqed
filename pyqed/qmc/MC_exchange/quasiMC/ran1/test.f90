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




  integer ( kind = 8 ), parameter :: test_num = 3

  integer ( kind = 8 ) dim_num
  integer ( kind = 8 ), dimension ( 3 ) :: dim_num_test = (/ 100, 500, 1000 /)
  integer ( kind = 8 ) i
  real ( kind = 8 ), allocatable, dimension ( : ) :: r
  integer ( kind = 8 ) seed
  integer ( kind = 8 ) seed_in
  integer ( kind = 8 ) seed_out
  integer ( kind = 8 ) test
      real*8 gasdev 

  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) 'TEST13'
  write ( *, '(a)' ) '  I8_SOBOL computes the next element of a Sobol sequence.'
  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) '  In this test, we get a few samples at high dimension.'
  write ( *, '(a)' ) '  We only print the first and last 2 entries of each'
  write ( *, '(a)' ) '  sample.'

    dim_num = 2 

      open(100, file = 'normal.dat') 
    
    allocate ( r(1:dim_num) )

    write ( *, '(a)' ) ' '
    write ( *, '(a,i8)' ) '  Using dimension DIM_NUM =   ', dim_num

    seed = 0

    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) '      Seed      Seed      I8_SOBOL'
    write ( *, '(a)' ) '        In       Out   (First 2, Last 2)'
    write ( *, '(a)' ) ' '

    do i = 0, 1000

      seed_in = seed

      r(1) = gasdev(seed) 
      r(2) = gasdev(seed) 

      seed_out = seed

      write ( *, '(2x,i16,2x,i16,2x,4f12.6)' ) &
        seed_in, seed_out, r(1:2) 


      write ( 100, 1000 )  r(1:2)

    end do

    deallocate ( r )



  write ( *, '(a)' ) ' '
  write ( *, '(a)' ) 'SOBOL_PRB'
  write ( *, '(a)' ) '  Normal end of execution.'
  write ( *, '(a)' ) ' '
!  call timestamp ( )

1000 format(2(f14.6,1x))
  stop
end
