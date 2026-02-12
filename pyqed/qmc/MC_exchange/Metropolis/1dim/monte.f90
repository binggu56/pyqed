! ---- Monte Carlo integration

    program MonteCarlo
    use sci 
    use common 
    use Monte 

    implicit real*8(a-h, o-z)

    
    real*8, dimension(nx) :: rho 
    
    integer (kind = 4) adjustInterval 
    
    integer (kind = 4) thermSteps
    
    real*8 gasdev, ran1 

    MCSteps = 10000



!#def oneMonteCarloStep():
!
!    # perform N Metropolis steps
!    #for i in range(N):
!     #   MetropolisStep()



    write(6, *) 'Monte Carlo for Gaussian integration '

    print *, '---------------------------------'
    print *, " Enter number of Monte Carlo steps: ", MCSteps 

    xmax = 6d0
    xmin = -6d0
    dx = (xmax-xmin)/nx
    
    delta = 1.0 ! jump size 
    idum = 54 
    
!    """initialize the random walkers"""
    x = rand(0)
    
    rho = 0d0

! perform 20% of MCSteps as thermalization steps
! and adjust step size so acceptance ratio ~50%

    thermSteps = int(0.2 * MCSteps)
    adjustInterval = int(0.1 * thermSteps) + 1
    nAccept = 0
    print *, ' Performing thermalization steps ...', thermSteps

    do i=1, thermSteps
    !oneMonteCarloStep()
        
        call MetropolisStep(x, rho)    

        if(mod(i+1, adjustInterval) == 0) then
        
            delta = delta * dble(nAccept) / (0.5 * dble(adjustInterval))
            nAccept = 0
            
        endif
    enddo 
    
    print *, ' Adjusted Gaussian step size ', delta



    nAccept = 0  ! accumulator for number of accepted steps
    eSum = 0.
    eSqdSum = 0.

    nAccept = 0;
    print *, " Performing production steps ...", MCSteps



    do i=1, MCSteps
    !MonteCarloStep()
        call MetropolisStep(x, rho)
        
    enddo


! compute and print energy

    eAve = eSum / MCSteps

    eVar = eSqdSum / MCSteps - eAve * eAve

    error = sqrt(eVar / MCSteps)

    write(6, 1012)  eAve, error
1012 format('Expectation = ',f14.8, ' +/- ', f14.8)

    write(6, 1013) eVar
1013 format('Variance = ',f14.8) 


! write wave function squared in file
    open(100, file = "psiSqd.data")


    psiNorm = sum(rho) * dx
    do i=1,nx
        z = xmin + i * dx
        write(100, *) z, rho(i) / psiNorm
    enddo

    stop 
    end program 
 ! ---------------------------

    subroutine prob(y, x, a)
    implicit real*8(a-h, o-z)

!    trial function is exp(-alpha*x^2)
!    compute the ratio of rho(x') / rho(x)

    alpha = 1.0
    a = exp(- alpha/2. * (y-2)**2 + alpha * (x-2)**2/2)
    end subroutine

! --- compute the local energy

    subroutine local(x,z)

    use sci, only : pi

    implicit real*8(a-h, o-z)

    z =  sqrt(1./2./pi) * exp(-1./2. * (x+2)**2)

    end subroutine

! --- Metropolis Scheme

    subroutine MetropolisStep(x, rho)

    use Monte, only : idum, xmin, xmax, dx, delta 
    use common, only : eSum, eSqdSum, nAccept, nx 
    
    implicit real*8(a-h, o-z)
    
    real*8 gasdev 
    real*8 rho(nx) 
    
    
    

    ! chose a walker at random
    ! n = int(np.random.rand() * N)

    ! make a trial move
    y = x + delta * gasdev(idum)

    ! Metropolis test
    call prob(y,x,a)
    
    if (a > rand(0)) then
        
        x = y
        nAccept = nAccept + 1
    endif

    ! accumulate local functions
    call local(x,e)
    eSum = eSum + e
    eSqdSum = eSqdSum +  e * e
    i = int((x - xmin) / dx)
    if(i .ge. 0 .and. i < nx) then
        rho(i) = rho(i) + 1
    endif

    return
    end subroutine

