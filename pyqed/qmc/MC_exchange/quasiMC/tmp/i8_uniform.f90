function i8_uniform ( a, b, seed )

!*****************************************************************************80
!
!! I8_UNIFORM returns a scaled pseudorandom I8.
!
!  Discussion:
!
!    An I8 is an integer ( kind = 8 ) value.
!
!    Note that ALL integer variables in this routine are
!    of type integer ( kind = 8 )!
!
!    The input arguments to this function should NOT be constants; they should
!    be variables of type integer ( kind = 8 )!
!
!    The pseudorandom number should be uniformly distributed
!    between A and B.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    12 November 2006
!
!  Author:
!
!    John Burkardt
!
!  Reference:
!
!    Paul Bratley, Bennett Fox, Linus Schrage,
!    A Guide to Simulation,
!    Springer Verlag, pages 201-202, 1983.
!
!    Pierre L'Ecuyer,
!    Random Number Generation,
!    in Handbook of Simulation,
!    edited by Jerry Banks,
!    Wiley Interscience, page 95, 1998.
!
!    Bennett Fox,
!    Algorithm 647:
!    Implementation and Relative Efficiency of Quasirandom
!    Sequence Generators,
!    ACM Transactions on Mathematical Software,
!    Volume 12, Number 4, pages 362-376, 1986.
!
!    Peter Lewis, Allen Goodman, James Miller
!    A Pseudo-Random Number Generator for the System/360,
!    IBM Systems Journal,
!    Volume 8, pages 136-143, 1969.
!
!  Parameters:
!
!    Input, integer ( kind = 8 ) A, B, the limits of the interval.
!
!    Input/output, integer ( kind = 8 ) SEED, the "seed" value, which
!    should NOT be 0.  On output, SEED has been updated.
!
!    Output, integer ( kind = 8 ) I8_UNIFORM, a number between A and B.
!
  implicit none

  integer ( kind = 8 ) a
  integer ( kind = 8 ) b
  integer ( kind = 8 ) i8_uniform
  real ( kind = 8 ) r
  real ( kind = 8 ) r8i8_uniform_01
  integer ( kind = 8 ) seed
  integer ( kind = 8 ) value

  if ( seed == 0 ) then
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) 'I8_UNIFORM - Fatal error!'
    write ( *, '(a)' ) '  Input value of SEED = 0.'
    stop
  end if

  r = r8i8_uniform_01 ( seed )
!
!  Scale R to lie between A-0.5 and B+0.5.
!
  r = ( 1.0D+00 - r ) * ( real ( min ( a, b ), kind = 8 ) - 0.5D+00 ) &
    +             r   * ( real ( max ( a, b ), kind = 8 ) + 0.5D+00 )
!
!  Use rounding to convert R to an integer between A and B.
!
  value = nint ( r, kind = 8 )

  value = max ( value, min ( a, b ) )
  value = min ( value, max ( a, b ) )

  i8_uniform = value

  return
end
