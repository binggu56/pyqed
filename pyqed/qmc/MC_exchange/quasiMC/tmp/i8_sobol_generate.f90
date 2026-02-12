subroutine i8_sobol_generate ( m, n, skip, r )

!*****************************************************************************80
!
!! I8_SOBOL_GENERATE generates a Sobol dataset.
!
!  Discussion:
!
!    Note that the internal variable SEED is of type integer ( kind = 8 ).
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    12 December 2009
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) M, the spatial dimension.
!
!    Input, integer N, ( kind = 4 ) the number of points to generate.
!
!    Input, integer ( kind = 4 ) SKIP, the number of initial points to skip.
!
!    Output, real ( kind = 8 ) R(M,N), the points.
!
  implicit none

  integer ( kind = 4 ) m
  integer ( kind = 4 ) n

  integer ( kind = 4 ) j
  real ( kind = 8 ), dimension ( m, n ) :: r
  integer ( kind = 8 ) seed
  integer ( kind = 4 ) skip

  seed = skip

  do j = 1, n
    call i8_sobol ( m, seed, r(1:m,j) )
  end do

  return
end
