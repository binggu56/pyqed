subroutine i4_sobol_generate ( m, n, skip, r )

!*****************************************************************************80
!
!! I4_SOBOL_GENERATE generates a Sobol dataset.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    17 January 2005
!
!  Author:
!
!    John Burkardt
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) M, the spatial dimension.
!
!    Input, integer ( kind = 4 ) N, the number of points to generate.
!
!    Input, integer ( kind = 4 ) SKIP, the number of initial points to skip.
!
!    Output, real ( kind = 4 ) R(M,N), the points.
!
  implicit none

  integer ( kind = 4 ) m
  integer ( kind = 4 ) n

  integer ( kind = 4 ) j
  real ( kind = 4 ), dimension ( m, n ) :: r
  integer ( kind = 4 ) seed
  integer ( kind = 4 ) skip

  do j = 1, n
    seed = skip + j - 1
    call i4_sobol ( m, seed, r(1:m,j) )
  end do

  return
end
