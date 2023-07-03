! --- compute the local energy

    subroutine local(x,z,env,enk,rDp)

    use sci, only : pi

    use Monte
    
    use AQP 
    
    implicit real*8(a-h, o-z)
    
    real*8, dimension(ndim) :: x, am, dp 
    
    am = 8.d0

    x1 = x(1) 
    x2 = x(2) 
    
    ni = 1 
    nj = 2 
    
    aij = ra(ni) - ra(nj)
    cij = rc(ni) - rc(nj)
    dij = rd(ni) - rd(nj)
    
    z = aij*(x(2)-x(1)) + cij * (x2**3 - x1**3)/3d0 + dij * (x2**4-x1**4)/4d0 + &
        (rb(1,1) - rb(2,2)) * (x2**2-x1**2)/2d0
    
    if(ndim > 2) then
        do k=3,ndim             
            z = z + (rb(1,k) - rb(2,k))*(x(2)-x(1))*x(k)
        enddo 
    endif 
    
    z = exp(z) 
    
    rDp = 0d0 
    
    ! potential energy 

    call potential(x,env)

     dp(1) = -aij - cij*x1**2- dij*x1**3 - (rb(1,1)-rb(2,2))*x(1)

     dp(2) =  aij + cij*x2**2 + dij*x2**3 + (rb(1,1)-rb(2,2))*x(2) 

    if(ndim .gt. 2) then 
        do k=3,ndim
            dp(k) = (rb(1,k)-rb(2,k))*(x(2)-x(1)) 
        enddo 
     endif 
    
    ! kinetic energy
    enk = 0d0 
     
    do j=1,ndim 
      rj = 0d0 
      rj = rj + ra(j) + rc(j)*x(j)**2 + rd(j)*x(j)**3
      do k=1,ndim 
        rj = rj + rb(j,k)*x(k) 
      enddo
      
      enk = enk + rj**2/2d0/am(j) 

      rDp = rDp + rj*dp(j)/2d0/am(j) ! r_mu * \grad_mu poly
    
    enddo  


    return 
    end subroutine
