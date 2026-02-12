! ---- Monte Carlo integration

    program MonteCarlo
    
    use sci   
    use Monte 
    use AQP 
    
    implicit real*8(a-h, o-z)
    
    integer (kind = 8) adjustInterval     
    integer (kind = 8) thermSteps
    
    real*8, dimension(ndim) :: x 
    
    real*8 gasdev  

    call map() 
    

! --- Monte Carlo parameters 

    MCSteps = 2000000 

    print *, 'Monte Carlo for Gaussian integration'
    print *, '-----------------------------------'

    !xmax = 6d0
    !xmin = -6d0
    !dx = (xmax-xmin)/nx
    
    delta = 1.0 ! jump size 
    idum = 11 
    
! --- initialize the random walkers 
    call random_number(x) 
    
    print *,'initial sampling point', x 
    
    !rho = 0d0

! perform 20% of MCSteps as thermalization steps
! and adjust step size so acceptance ratio ~50%

    thermSteps = int(0.2 * MCSteps)
    adjustInterval = int(0.1 * thermSteps) + 1
    nAccept = 0
    
    print *, ' Performing thermalization steps ...', thermSteps

    do i=1, thermSteps
    
        call MetropolisStep(x)    

        if(mod(i+1, adjustInterval) == 0) then
        
            delta = delta * dble(nAccept) / (0.5 * dble(adjustInterval))
            nAccept = 0
            
        endif
        
    enddo 
    
    print *, ' Adjusted Gaussian step size ', delta


! --- production steps 

    nAccept = 0  ! accumulator for number of accepted steps
    eSum = 0.
    eSqdSum = 0.
    envSum = 0.0 
    enkSum = 0d0 
    envpSum = 0.0 
    enkpSum = 0.0 
    rDpSum = 0.0 
    
    nAccept = 0 
    print *, " Performing production steps ...", MCSteps

    do i=1, MCSteps
    !MonteCarloStep()
        call MetropolisStep(x)       
    enddo


! --- compute and print energy

    pAve = eSum / MCSteps
    
    envAve = envSum / MCSteps
    
    enkAve = enkSum / MCSteps
    
    rDpAve = rDpSum / MCSteps 

    eVar = eSqdSum / MCSteps - eAve * eAve

    error = sqrt(eVar / MCSteps)

    write(6, 1012)  pAve, error, envAve, enkAve, enkAve+envAve  
1012 format('Permutation overlap = ',f14.8, ' +/- ', f14.8/, & 
             'Pontential energy = ', f14.7/, & 
             'Kinenticl energy = ',f14.7/, & 
             'Total energy = ',f14.7)

    envSym = (envAve+envpAve)/(1d0+pAve)
    enkSym = (enkAve+enkpAve+rDpAve)/(1d0+pAve)
    
    write(6, 1013) eVar, envSym, envSym-envAve, enkSym, enkSym-enkAve, enkSym+envSym  
1013 format('Variance = ',f14.8/  & 
            'Corrected potential energy = ',f14.7/,  'Correction = ', f14.7/ &  
            'Corrected kinentic energy = ',f14.7/, 'Correction = ',f14.7/, & 
            'Corrected total energy = ',f14.7)      


! --- write wave function squared in file
    !open(100, file = "psiSqd.data")


!    psiNorm = sum(rho) * dx
!    do i=1,nx
!        z = xmin + i * dx
!        write(100, *) z, rho(i) / psiNorm
!    enddo

    stop 
    end program 
 ! ---------------------------

    subroutine prob(y, x, a)
    
    use Monte 
    use AQP
    
    implicit real*8(a-h, o-z)
    
    real*8, dimension(ndim) :: x,y 

! --- trial function is exp(-alpha1*x1^2-alpha2*x2^2)
! --- compute the ratio of rho(x') / rho(x)

    ax = 1.0
    ay = 1.0
    
    z0 = psiSqd(x) 
    z1 = psiSqd(y)  
   
    a = z1/z0 
    
    end subroutine

! --- compute the local energy

    subroutine local(x,z,env,enk,rDp)

    use sci, only : pi

    use Monte, only : ndim 
    
    use AQP 
    
    implicit real*8(a-h, o-z)
    
    real*8, dimension(ndim) :: x, am, dp 
    
    am = 4.d0

    x1 = x(1) 
    x2 = x(2) 
    
    ni = 1 
    nj = 2 
    
    aij = ra(ni)-ra(nj)
    cij = rc(ni)-rc(nj)
    dij = rd(ni)-rd(nj)
    
    z = aij*(x(2)-x(1)) + cij * (x2**3 - x1**3)/3d0 + dij * (x2**4-x1**4)/4d0 + &
        (rb(1,1) - rb(2,2)) * (x2**2-x1**2)/2d0
    
    if(ndim .gt. 2) then 
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

      rDp = rDp + rj*dp(j) ! r_mu * \grad_mu poly 
    
    enddo  

    return 
    end subroutine


! --- potential energy     
    subroutine potential(x,env)
    

    use Monte, only : ndim  
    implicit real*8(a-h,o-z)
    
    real*8, dimension(ndim) :: x 
    
    R0 = 1.5d0 
    env = (x(1)+x(2))**2/2d0 + (abs(x(1)-x(2))-R0)**2/2d0

    return 
    end subroutine 

! ---------------------------------
! --- Metropolis Scheme
! ---------------------------------
    subroutine MetropolisStep(x)

    use Monte 
   
    
    implicit real*8(a-h, o-z)
    
    real*8, dimension(ndim) :: x,y  
    
    real*8 gasdev 
    !real*8 rho(nx) 
    
    ! chose a walker at random
    ! n = int(np.random.rand() * N)

    ! make a trial move
    do j=1,ndim 
        y(j) = x(j) + delta * gasdev(idum)
    enddo 

    ! Metropolis test
    call prob(y,x,a)
    
    if (a > rand(0)) then
        
        x = y
        nAccept = nAccept + 1
    endif

    ! accumulate local functions
    call local(x,p,env,enk,rDp)
    eSum = eSum + p
    eSqdSum = eSqdSum +  p * p
    envSum = envSum + env ! <V> 
    enkSum = enkSum + enk ! <K> 
    envpSum = envpSum + p*env ! <VP> 
    enkpSum = enkpSum + p*enk ! <KP> 
    rDpSum = rDpSum + rDp  ! < dpsi/psi * dPoly * exp(Poly) > 
    
    
    !i = int((x - xmin) / dx)
    !if(i .ge. 0 .and. i < nx) then
    !    rho(i) = rho(i) + 1
    !endif

    return
    end subroutine
    
! --- compute psiSqd from the linear-fitting coefficients of r 

    double precision function psiSqd(x)
    
    use Monte, only: ndim 
    
    use AQP 
    
    implicit real*8(a-h, o-z)  
    
    real*8, dimension(ndim) :: x 
      
      z = 0d0 
      do j=1,ndim 
        z = z + ra(j)*x(j) + rb(j,j)/2d0 * x(j)**2 + rc(j)/3d0 * x(j)**3 + rd(j)/4d0 * x(j)**4 
      enddo 
      do j=1,ndim 
        do k=j+1,ndim 
          z = z + rb(j,k)*x(j)*x(k) 
        enddo 
       enddo 
      
      psiSqd = exp(2.0 * z) 
      
      return  

    end function 


! --- map fitting coefficients cr,cr2 to {a,b,c,d}, which are coefficients written in 
!     non-displaced Cartesian coordinates
    subroutine map() 
    use Monte, only: ndim
    use AQP  
    
    implicit real*8(a-h, o-z) 
     
    real (kind = 8 ) cr(ndim+1,ndim),cr2(4,ndim)

    real (kind = 8), dimension(ndim,ndim) :: beta, delta
    real (kind = 8), dimension(ndim) :: alfa, gama, rho

    real (kind = 8), dimension(ndim) :: q
    real (kind = 8), dimension(ndim) :: dp

! --- choose pair of atoms to exchange
      
      ni = 1
      nj = 2

! --- read fitting coefficients to get analytical wavefunction

      open(11, file = 'linearFit.data')
      do k=1,ndim+1
        read(11,*) (cr(k,j),j=1,ndim)
      enddo

      open(12, file = 'cubicFit.data')
      do j=1,4
        read(12,*) (cr2(j,k),k=1,ndim)
      enddo

      close(11)
      close(12)

      write(6,*) 'Finish reading files.'


! -- no need to translate coordinates
!      do i=1,ntraj
!      do j=1,ndim
!        x(j,i) = u(j,i) + xtal(j,1)
!      enddo
!      enddo

      delta = 0d0
      do k=1,ndim
        delta(k,k) = 1d0
      enddo


      do k=1,ndim
        alfa(k) = cr(ndim+1,k)+cr2(4,k)
        gama(k) = cr2(2,k)
        rho(k) = cr2(3,k)
      enddo

      do k=1,ndim
      do j=1,ndim
        beta(j,k) = cr(k,j)
      enddo
      enddo

      do k=1,ndim
        beta(k,k) = beta(k,k) + cr2(1,k)
      enddo

! --- q(ndim) optimal Cartesian coordinates

      ra = 0d0

    q = 0d0 ! no translation, use when the coordinates used in trajectory code is displaced-Cartesian 
    
      do k=1,ndim
      do m = 1,ndim
        ra(k) = ra(k)- beta(k,m)*q(m)
      enddo
      enddo

      do k=1,ndim
        ra(k) = ra(k) + alfa(k) + gama(k)*q(k)**2
        rc(k) = gama(k) - 3d0*rho(k)**2*q(k)
        rd(k) =  rho(k)
      enddo

      do k=1,ndim
      do m=1,ndim
        rb(k,m) = beta(k,m)
      enddo
      enddo

      do k=1,ndim
        rb(k,k) = rb(k,k) - 2d0*gama(k)*q(k) + 3d0*rho(k)*q(k)**2
      enddo


    return 
    end subroutine 


