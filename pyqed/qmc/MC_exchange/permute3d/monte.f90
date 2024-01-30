! ---- Monte Carlo integration

! --- To use the program, a couple of things has to be changed:
!      1. potential : the potential subroutine
!      2. map: the directory where the data is saved
!      3. mod.f90 : parameters like dimensionality and number of trajectories
!      4. local.f90 : particle mass

! -------------------------------

    program MonteCarlo
    
    use sci   
    use Monte 
    use AQP 
    use model 
    
    implicit real*8(a-h, o-z)
    
    integer (kind = 8) adjustInterval     
    integer (kind = 8) thermSteps
    
    real*8, dimension(Ndim, NWalkers) :: x 
    
    real*8 gasdev  

    call map() 
    

! --- Monte Carlo parameters 



    print *, 'Monte Carlo for Gaussian integration'
    print *, '-----------------------------------'


    !xmax = 6d0
    !xmin = -6d0
    !dx = (xmax-xmin)/nx
    
    delta = 1.0 ! jump size 
    idum = 11 
    
    print *,'interparticle distance ', R0
    print *,' ' 
     
! --- initialize the random walkers 
    call random_number(x) 
    
    do k=1,NWalkers 
        print *,'initial walker ', k, ' = ', x(1:Ndim,k)  
    enddo 
   
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
    
    pSum = 0.0 
    pSqdSum = 0.0 
    
    envSum = 0.0 
    enkSum = 0d0 
    envpSum = 0.0 
    enkpSum = 0.0 
    rDpSum = 0.0 ! r grad p exp(p) 
    
    eSqdSum = 0.0 ! energy without symmetrization 
    hSqdSum = 0.0 ! energy for symmetrized wavefunction 
    
     
    
    nAccept = 0 
    print *, " Performing production steps ...", MCSteps

    do i=1, MCSteps
        call OneMetropolisStep(x)       
    enddo


! --- compute and print energy

    pAve = pSum / MCSteps / NWalkers 
    
    envAve = envSum / MCSteps / NWalkers 
    
    enkAve = enkSum / MCSteps / NWalkers 
    
    eAve = envAve + enkAve 
    
    eVar = eSqdSum / MCSteps - eAve * eAve 
    
    rDpAve = rDpSum / MCSteps / NWalkers 

    pVar = pSqdSum / MCSteps / NWalkers  - pAve * pAve
     

    pError = sqrt(pVar) / sqrt(dble(NWalkers) * dble(MCSteps))
    
    NN = 2 ! nearest-neighbour permutation 
    
    ! symmetrized energy 
    
    enkpAve = enkpSum / NWalkers / MCSteps
    envpAve = envpSum / NWalkers / MCSteps

    envSym = (envAve + NN*envpAve) / (1d0 + NN*pAve)

    enkSym = (enkAve + NN*(enkpAve+rDpAve)) / (1d0+ NN*pAve)

    GSE = envSym + enkSym

    hSum = enkSum + NN*enkpSum + envSum + NN*envpSum + NN*rDpSum

    hAve = hSum / NWalkers / MCSteps 
    
    hVar = hSqdSum / MCSteps / NWalkers - hAve * hAve
    
    hError = sqrt(hVar) / sqrt(dble(NWalkers) * dble(MCSteps))
    
    ! total error for the ground-state energy



    GSEError = hError/(1d0+ NN*pAve) + hAve*NN*pError/(1d0 + NN*pAve)**2   

    write(6, 1012)  pAve, pError, envAve, enkAve, eAve   
1012 format('Permutation overlap = ',f14.8, ' +/- ', f14.7//,   & 
             'Pontential energy = ', f14.7/,    & 
             'Kinentic energy = ',f14.7/,      & 
             'Total energy = ',f14.7)



    write(6, 1013) envSym, envSym-envAve, enkSym, enkSym-enkAve, GSE, GSEError
1013 format('Corrected potential energy = ',f14.7/,  'Correction = ', f14.7/ &  
            'Corrected kinentic energy = ',f14.7/, 'Correction = ',f14.7/, & 
            'Corrected ground-state energy = ',f14.7, ' +/- ', f14.7)      


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




! --- potential energy     
    subroutine potential(x,env)
    
    use Monte, only : ndim
    use model, only : R0 
      
    implicit real*8(a-h,o-z)
    
    real*8, dimension(ndim) :: x 
    
    akt = 10d0
    akv = 1d0  

    xcm = (x(1)+x(2)+x(3))/3d0
 
    env = akt*xcm**2/2d0 + akv*((abs(x(1)-x(2))-R0)**2/2d0 + & 
      (abs(x(1)-x(3))-R0)**2/2d0 + (abs(x(2)-x(3))-R0)**2/2d0) - R0**2/6d0

    return 
    end subroutine 

! ---------------------------------
! --- Metropolis Scheme
! ---------------------------------

subroutine OneMetropolisStep(x)
    use Monte 
    
    implicit real*8(a-h, o-z)
    
    real*8, dimension(Ndim, NWalkers) :: x 
    
    do i=1, NWalkers 
        call MetropolisStep(x) 
    enddo
end subroutine 

subroutine MetropolisStep(x)

    use Monte 
   
    
    implicit real*8(a-h, o-z)
    
    real*8, dimension(ndim) :: y,z
    real*8, dimension(Ndim, NWalkers) :: x  
    
    real*8 gasdev 
    !real*8 rho(nx) 
    
    ! chose a walker at random
    k = int(ran1(idum) * NWalkers) + 1


    ! make a trial move
    do j=1,ndim 
        y(j) = x(j,k) + delta * gasdev(idum)
    enddo 

    ! Metropolis test
    do j=1,Ndim
        z(j) = x(j,k)
    enddo 
    call prob(y,z,a)
    
    if (a > ran1(idum)) then

        do j=1,Ndim
           x(j,k) = y(j) 
        enddo 
        nAccept = nAccept + 1

    endif

    ! accumulate local functions
    call local(z,p,env,enk,rDp)
    
    pSum = pSum + p  ! p represents exp(Q) 

    pSqdSum = pSqdSum +  p * p
    
    envSum = envSum + env ! <V> 
    enkSum = enkSum + enk ! <K> 

    envpSum = envpSum + p*env ! <VP> 
    enkpSum = enkpSum + p*enk ! <KP> 
    rDpSum = rDpSum + rDp*p  ! < dpsi/psi * dPoly * exp(Poly) >
    
    hSqdSum = hSqdSum + ( (enk+env)*(1d0+p) + rDp*p )**2

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
    use Monte
    use AQP  
    
    implicit real*8(a-h, o-z) 
     
    real (kind = 8 ) cr(ndim+1,ndim), cr2(4,ndim)

    real (kind = 8), dimension(ndim,ndim) :: beta
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


    print *,'cr =', cr
    print *, 'cr2 = ',cr2
! -- no need to translate coordinates
!      do i=1,ntraj
!      do j=1,ndim
!        x(j,i) = u(j,i) + xtal(j,1)
!      enddo
!      enddo



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


