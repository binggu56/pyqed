      program eloc
      USE AQP 
      use model 

      implicit double precision (a-h, o-z)


!------------trajectory initial allocate------------------------
      real*8, dimension(:,:), allocatable :: x,p,du,ap,ar,rp,fr
      real*8, dimension(:), allocatable :: p0,w,alpha,x0,pe,ke,u,cf,am 
      real*8, dimension(:,:),allocatable :: dv 

      real*8 :: gasdev
      
      integer*4, dimension(:), allocatable :: idum

      parameter (half=0.5d0)
      parameter (one=1.0d0)

      open(110,file='energy.dat')
      open(101,file='x.dat')
      open(102,file='traj.dat')
      open(103,file='qpot1.dat')
      open(104,file='qpot2.dat') 
      open(105,file='xy.dat') 
!------read parameters from IN----------------------      
      open(10,file='IN')

      read(10,*) Ntraj
      read(10,*) ndim 
      read(10,*) kmax,dt
      read(10,*) a0
      read(10,*) am0 
      read(10,*) cf0
      read(10,*) R0 

      close(10)

!-----------------------------------------------------
      allocate(pe(Ntraj),ke(Ntraj),x(Ndim,Ntraj),              &
              p(Ndim,Ntraj),u(Ntraj),du(Ndim,Ntraj),w(Ntraj),  & 
              ap(ndim,ntraj),ar(ndim,ntraj),rp(ndim,ntraj),fr(ndim,ntraj))

      allocate(dv(ndim,ntraj))

      allocate(p0(Ndim),alpha(Ndim),x0(Ndim),cf(Ndim),idum(ndim),am(ndim)) 

      
      dt2 = dt/2d0
      t   = 0d0
      pow = 4d0

      do i=1,Ndim
        am(i) = am0
        p0(i) = 0d0
        alpha(i) = a0
        cf(i) = cf0
      enddo

      x0(1) = -2d0*R0/3d0 
      x0(2) = 0.0 
      x0(3) = 2d0*R0/3d0 

      call seed(idum,Ndim)

      write(*,1010) Ntraj,Ndim,kmax,dt,cf(1),am
1010  format('Initial Conditions'//,  &
            'Ntraj   = ' , i6/,       &
            'Ndim    = ' , i6/,       &
            'kmax    = ' , i6/,       &
            'dt      = ' , f8.4/,     &
            'fric    = ' , f8.4/,     &
            'Mass    = ' , 3e14.6/)

      write(*,1011) alpha,x0,p0, R0 
1011  format('Initial Wavepacket'//,  &
            'alpha0   = ' , 3f8.4/,    &
            'center   = ' , 3f8.4/,    &
            'momentum = ' , 3f8.4//     &
            'interparticle distance = ', f8.4/)


! --- initial Lagrangian grid points (evolve with time)
      do i=1,Ntraj
        do j=1,Ndim
1100      x(j,i)=gasdev(idum(j))
          x(j,i)=x(j,i)/dsqrt(4d0*alpha(j))+x0(j)
          if((x(j,i)-x0(j))**2 .gt. pow/2d0/alpha(j)) goto 1100
        enddo
      enddo



! --- initial momentum for QTs and weights

      do i=1,Ntraj
        do j=1,Ndim
            p(j,i) = p0(j)
            rp(j,i) = -2d0*alpha(j)*(x(j,i)-x0(j))
        enddo
        w(i) = 1d0/dble(ntraj)
      enddo

!---- expectation value of x(1,ntraj)--------
      av = 0d0
      do i=1,ntraj
        av = av+w(i)*x(2,i)
      enddo
      write(*,6008) av
6008  format('Expectation value of x(2,ntraj) = ', f10.6)


! --- trajectories propagation	
      do 10 k=1,kmax


        t=t+dt

        call derivs(ndim,ntraj,x,dv,pe)
        call fit(k,ndim,ntraj,am,x,p,rp,w,u,du,ap,fr)

! ----- half-step increments of momenta & full step increment of positions
        do i=1,ntraj 
          do j=1,Ndim
              p(j,i)=p(j,i)+(-dv(j,i)-du(j,i)-cf(j)*p(j,i))*dt
              x(j,i)=x(j,i)+ap(j,i)*dt/am(j)
              rp(j,i)=rp(j,i)+fr(j,i)*dt
          enddo   
        enddo 

! ----- half-step increments of momenta
!        do i=1,Ntraj
!
!          do m=1,NATOM3
!            q(m) = x(m,i)
!          enddo
!
!          call local(q,vloc,dv)
!          call long_force(q,vlong,dvl)
!
!          ev(i) = vloc+vlong
!
!          do j=1,Ndim
!            p(j,i)=p(j,i)+(-dv(j)-dvl(j)+qf(j,i)-cf(j)*p(j,i))*dt2
!          enddo
          
!        enddo

!-------update potential, kinetic, and total energy
          do i=1,Ntraj
            ke(i) = 0d0
            do j=1,Ndim
              ke(i)=p(j,i)**2/(2d0*am(j))+ke(i)
            enddo
          enddo

          write(101,1000) t,(x(1,i),i=1,20)
          

        call aver(Ntraj,w,pe,po)
        call aver(Ntraj,w,ke,enk)
        call aver(Ntraj,w,u,qu)

        tot = po+enk+qu
      
!      if(k == 100) then 
!        do i=1,ntraj 
!          write(103,1000) x(1,i),u(i)
!        enddo 
!
!      elseif(k == 400) then 
!
!        do i=1,ntraj 
!          write(104,1000) x(1,i),u(i)
!        enddo 
!        
!      endif 

        write(110,1000) t,enk,po,qu,tot
        call flush(110)

!      open(106,file='temp.dat')
!      if (mod(kt,1000) .eq. 0) then
!        do i=1,Ntraj
!          do j=1,Ndim
!            write(106,1000) x(j,i),p(j,i)
!            call flush(106)
!          enddo
!        enddo
!      endif

!      close(106)

10    enddo


      write(*,1020) tot 
1020  format('Total Energy =', f10.5/ , & 
             'MISSION COMPLETE.')

      do i=1,ntraj 
        write(105,1000) x(1,i),x(2,i),x(3,i) 
      enddo 
      
!       convert to eV 
!          vloc = vloc*27.211d0
1000  format(100(e14.7,1x))
      
      end program

! -----------------------------------
      subroutine quit

      write (6, *) 'termination via subroutine quit'

      stop
      end subroutine
! ---------------------------------------------------------------
!     random number seeds
! ---------------------------------------------------------------
      subroutine seed(idum,Ndim)
      implicit real*8(a-h,o-z)
      integer*4, intent(IN) :: Ndim
      integer*4, intent(OUT) :: idum(Ndim)

      do i=1,Ndim
        idum(i) = 5 + i
      enddo

      return
      end subroutine
! ---------------------------------------------------------------
!     average over trajectories
! ---------------------------------------------------------------
      subroutine aver(Ntraj,w,x,y)
      implicit real*8(a-h,o-z)
      real*8 :: x(Ntraj), w(Ntraj)

      y = 0d0

      do i=1,Ntraj
        y = y+x(i)*w(i)
      enddo
     
      return
      end subroutine
! ----------------------------------------------------------------
