
c     QSATS version 1.0 (3 March 2011)

c     file name: eloc.f

c ----------------------------------------------------------------------

c     this computes the total energy and the expectation value of the
c     potential energy from the snapshots recorded by QSATS.

c ----------------------------------------------------------------------

      program eloc

      implicit double precision (a-h, o-z)

      include 'sizes.h'

      include 'qsats.h'
      
      common /bincom/ bin, binvrs, r2min

      real*8,    dimension(:,:), allocatable :: x,p,qf
      real*8,    dimension(:),   allocatable :: am,p0,w,alpha,
     +                                          x0,ke,qp,cf
      real*8,    dimension(:),   allocatable :: ev

      character*4 :: atom(NATOMS)
      real :: gasdev    
      real*8 :: dvl(NATOM3)

c --- this common block is used to enable interpolation in the potential
c     energy lookup table in the subroutine local below.



      dimension q(NATOM3), vtavg(NREPS), vtavg2(NREPS),
     +          etavg(NREPS), etavg2(NREPS),dv(NATOM3)
      dimension idum(NATOM3)

      parameter (half=0.5d0)
      parameter (one=1.0d0)


      open(100, file='energy.dat')
      open(101, file='xoutput')
      open(102, file='traj4')
      open(103, file='pot.dat')
      open(104, file='force.dat')
      open(105, file='pes.dat')
      open(106, file='temp.dat')

      open(10,  file='IN')
      read(10, *) Ntraj
      read(10, *) kmax,dt
      read(10, *) cf0
      read(10, *) a0
      close(10)

      Ndim = NATOM3

c --- initialization.

!      call tstamp

      write (6, 6001) NREPS, NATOMS, NATOM3, NATOM6, NATOM7,
     +                NVBINS, RATIO, NIP, NPAIRS
6001  format ('compile-time parameters:'//,
     +        'NREPS  = ', i6/,
     +        'NATOMS = ', i6/,
     +        'NATOM3 = ', i6/,
     +        'NATOM6 = ', i6/,
     +        'NATOM7 = ', i6/,
     +        'NVBINS = ', i6/,
     +        'RATIO  = ', f6.4/,
     +        'NIP    = ', i6/,
     +        'NPAIRS = ', i6/)

!      call input
      den = 4.61421d-3

      call vinit(r2min, bin)
!-----------------------------------------------------
      allocate(ke(Ntraj),
     &         x(Ndim,Ntraj),p(Ndim,Ntraj),
     &         qp(Ntraj),qf(Ndim,Ntraj),w(Ntraj))
      allocate(ev(Ntraj))

      allocate(p0(Ndim),alpha(Ndim),am(Ndim),
     &           x0(Ndim),cf(Ndim)) 
     
      binvrs=one/bin

c --- read crystal lattice points.
      ltfile = 'ltfile'
      write (6, 6200) ltfile
6200  format ('READING crystal lattice from ', a16/)

      open (8, file=ltfile, status='old')

      read (8, *) nlpts

      if (nlpts.ne.NATOMS) then
         write (6, *) 'ERROR: number of atoms in lattice file = ', nlpts
         write (6, *) 'number of atoms in source code = ', NATOMS
         stop
      end if

c --- read the edge lengths of the supercell.

      read (8, *) xlen, ylen, zlen

c --- compute a distance scaling factor.

      den0=dble(NATOMS)/(xlen*ylen*zlen)

c --- scale is a distance scaling factor, computed from the atomic
c     number density specified by the user.

      scale=exp(dlog(den/den0)/3.0d0)

      write (6, 6300) scale
6300  format ('supercell scaling factor computed from density = ',
     +        f12.8/)

      xlen=xlen/scale
      ylen=ylen/scale
      zlen=zlen/scale

      write (6, 6310) xlen, ylen, zlen
6310  format ('supercell edge lengths [bohr]         = ', 3f10.5/)

      dxmax=half*xlen
      dymax=half*ylen
      dzmax=half*zlen

      do i=1, NATOMS

         read (8, *) xtal(i, 1), xtal(i, 2), xtal(i, 3)

         xtal(i, 1)=xtal(i, 1)/scale
         xtal(i, 2)=xtal(i, 2)/scale
         xtal(i, 3)=xtal(i, 3)/scale

      end do

      close (8)

      write (6, 6320) xtal(NATOMS, 1), xtal(NATOMS, 2),
     +                xtal(NATOMS, 3)
6320  format ('final lattice point [bohr]            = ', 3f10.5/)

c --- this variable helps us remember the nearest-neighbor distance.

      rnnmin=-1.0d0

      do j=2, NATOMS

         dx=xtal(j, 1)-xtal(1, 1)
         dy=xtal(j, 2)-xtal(1, 2)
         dz=xtal(j, 3)-xtal(1, 3)

c ------ this sequence of if-then-else statements enforces the
c        minimum image convention.

         if (dx.gt.dxmax) then
            dx=dx-xlen
         else if (dx.lt.-dxmax) then
            dx=dx+xlen
         end if

         if (dy.gt.dymax) then
            dy=dy-ylen
         else if (dy.lt.-dymax) then
            dy=dy+ylen
         end if

         if (dz.gt.dzmax) then
            dz=dz-zlen
         else if (dz.lt.-dzmax) then
            dz=dz+zlen
         end if

         r=sqrt(dx*dx+dy*dy+dz*dz)

         if (r.lt.rnnmin.or.rnnmin.le.0.0d0) rnnmin=r

      end do

      write (6, 6330) rnnmin
6330  format ('nearest neighbor (NN) distance [bohr] = ', f10.5/)

      write (6, 6340) xlen/rnnmin, ylen/rnnmin, zlen/rnnmin
6340  format ('supercell edge lengths [NN distances] = ', 3f10.5/)

c --- compute interacting pairs.

      do i=1, NATOMS
         npair(i)=0
      end do

      nvpair=0

      do i=1, NATOMS
      do j=1, NATOMS

         if (j.ne.i) then

            dx=xtal(j, 1)-xtal(i, 1)
            dy=xtal(j, 2)-xtal(i, 2)
            dz=xtal(j, 3)-xtal(i, 3)

c --------- this sequence of if-then-else statements enforces the
c           minimum image convention.

            if (dx.gt.dxmax) then
               dx=dx-xlen
            else if (dx.lt.-dxmax) then
               dx=dx+xlen
            end if

            if (dy.gt.dymax) then
               dy=dy-ylen
            else if (dy.lt.-dymax) then
               dy=dy+ylen
            end if

            if (dz.gt.dzmax) then
               dz=dz-zlen
            else if (dz.lt.-dzmax) then
               dz=dz+zlen
            end if

            r2=dx*dx+dy*dy+dz*dz

            r=sqrt(r2)

c --------- interacting pairs are those for which r is less than a
c           certain cutoff amount. 

            if (r/rnnmin.lt.RATIO) then

               nvpair=nvpair+1

               ivpair(1, nvpair)=i
               ivpair(2, nvpair)=j

               vpvec(1, nvpair)=dx
               vpvec(2, nvpair)=dy
               vpvec(3, nvpair)=dz

               npair(i)=npair(i)+1

               ipairs(npair(i), i)=nvpair

            end if

         end if

      end do
      end do

      write (6, 6400) npair(1), nvpair
6400  format ('atom 1 interacts with ', i3, ' other atoms'//,
     +        'total number of interacting pairs = ', i6/)

c --- initialization.

      loop=0
!      do k=1, NREPS
!         vtavg(k)=0.0d0
!         etavg(k)=0.0d0
!         vtavg2(k)=0.0d0
!         etavg2(k)=0.0d0
!      end do

!      open (10, file=spfile, form='unformatted')

c --- this loops reads the snapshots saved by QSATS.

!300   loop=loop+1

!     do k=1, NREPS, 11

!        read (10, end=600) (path(i, k), i=1, NATOM3)

!------reduce the number of pairs-------------
      call reduce('REDUCE')

      print *,'nvpair2=',nvpair2
      print *,ivpair2(1,1),ivpair2(2,1)

c ------ compute the local energy and the potential energy.
      
      dt2 = dt/2d0
      t   = 0d0
      p0  = 0d0
      cf  = cf0
      pow = 6d0
      do i=1,Ndim
        am(i) = 4d0*1836.15d0
        x0(i) = 0.0d0
        alpha(i) = a0
      enddo

!      alpha = 0.5d0

      call seed(idum,Ndim)

      write(*,1010) Ntraj,Ndim,kmax,dt,cf0,am(1)
1010  format('Initial Conditions'//,
     +       'Ntraj   = ' , i6/, 
     +       'Ndim    = ' , i6/,
     +       'kmax    = ' , i6/,
     +       'dt      = ' , f6.4/,
     +       'fric    = ' , f6.4/,
     +       'Mass    = ' , e14.6/)

      write(*,1011) alpha(1),x0(1),p0(1)
1011  format('Initial Wavepacket'//,
     +       'alpha0   = ' , f6.4/,
     +       'center   = ' , f6.4/,
     +       'momentum = ' , f6.4/)



!-------reading temp file--------------------
      open(11, file='temp.dat')
      ic = 0
      do i=1,Ntraj
        ic = ic+1
        do j=1,NATOM3
          read(11,1000) x(j,i),p(j,i)
        enddo
      enddo
      close(11)

      if(ic .ne. ntraj) then
        write(*,*) 'Reading file error.'
        stop
      endif



! --- initial Lagrangian grid points (evolve with time)
!      do i=1,Ntraj
!        do j=1,Ndim
!1100      x(j,i)=gasdev(idum(j))
!          x(j,i)=x(j,i)/sqrt(4d0*alpha(j))+x0(j)
!          if((x(j,i)-x0(j))**2 .gt. pow/2d0/alpha(j)) goto 1100
!        enddo
!      enddo

! --- initial momentum for QTs and weights
      do i=1,Ntraj
!        do j=1,Ndim
!            p(j,i)=p0(j)
!        enddo
        w(i) = 1d0/Ntraj
      enddo
c --- force at time 0 --------------

      call qpot(am,x,w,ndim,ntraj,qp,qf)

! --- trajectories propagation	
      do 10 kt=1,kmax

        t=t+dt

! ----- half-step increments of momenta & full step increment of positions
        do i=1,Ntraj
          do m=1,NATOM3
            q(m) = x(m,i)
          enddo

          call local(q,vloc,dv)
          call long_force(q,vlong,dvl)

          do j=1,Ndim
              p(j,i)=p(j,i)+(-dv(j)-dvl(j)+qf(j,i)-cf(j)*p(j,i))*dt2
              x(j,i)=x(j,i)+p(j,i)*dt/am(j)
          enddo   
        enddo 
    
        call qpot(am,x,w,Ndim,Ntraj,qp,qf)

! ----- half-step increments of momenta
        do i=1,Ntraj

          do m=1,NATOM3
            q(m) = x(m,i)
          enddo

          call local(q,vloc,dv)
          call long_force(q,vlong,dvl)

          ev(i) = vloc+vlong

          do j=1,Ndim
            p(j,i)=p(j,i)+(-dv(j)-dvl(j)+qf(j,i)-cf(j)*p(j,i))*dt2
          enddo
          
        enddo
!-------update potential, kinetic, and total energy
          do i=1, Ntraj
            ke(i) = 0d0
            do j=1,Ndim
              ke(i)=p(j,i)**2/(2d0*am(j))+ke(i)
            enddo
          enddo

          write(101,1000) t,(x(1,i),i=1,20)
          write(102,1000) t,(x(j,4),j=1,20)

        call aver(Ntraj,w,ev,po)
        call aver(Ntraj,w,ke,enk)
        call aver(Ntraj,w,qp,qu)

        tot = po+enk+qu

        do i=1,ntraj
          write(105,1000) dble(i),ev(i)
        enddo

        write(100,1000) t,po,enk,qu,tot
        call flush(100)

c --- store temporary data
      if (mod(kt,100) .eq. 0) then
        do i=1,Ntraj
          do j=1,Ndim
            write(106,1000) x(j,i),p(j,i)
            call flush(106)
          enddo
        enddo
      endif

10    enddo


      write(*,1020) tot 
1020  format('Total Energy =', f10.5/ ,
     +       'MISSION COMPLETE.')

      
!          call local(q,vloc,dv)

!       convert to eV 
!          vloc = vloc*27.211d0
c ------ convert to kelvin per atom.

!         tloc=tloc/(3.1668513d-6*dble(NATOMS))
!          vloc=vloc/(3.1668513d-6*dble(NATOMS))

c ------ accumulate the results.

!         vtavg(k)=vtavg(k)+vloc
!         vtavg2(k)=vtavg2(k)+(vloc)**2
!         etavg(k)=etavg(k)+tloc+vloc
!         etavg2(k)=etavg2(k)+(tloc+vloc)**2

!350      continue

!      end do

!      goto 300

c --- account for overshooting.

!600   loop=loop-1

!      write (6, 6600) loop
!6600  format ('number of snapshots = ', i6/)

c --- compute the averages and standard deviations.

!      do k=1, NREPS, 11
!
!      vtavg(k)=vtavg(k)/dble(loop)
!      vtavg2(k)=vtavg2(k)/dble(loop)
!      etavg(k)=etavg(k)/dble(loop)
!      etavg2(k)=etavg2(k)/dble(loop)
!
!      vsd=sqrt(vtavg2(k)-vtavg(k)**2)
!      esd=sqrt(etavg2(k)-etavg(k)**2)
!
!      write (6, 6610) k, 'VAVG = ', vtavg(k)
!6610  format ('replica ', i3, 1x, a7, f10.5, ' Kelvin')
!          write (100,1000) q(1),vloc
!       end do
!1001    format('potential at equilibrim configuration [K/atom] =',
!     &         f12.7)
!
!      write (6, 6610) k, 'V SD = ', vsd
!
!      write (6, 6610) k, 'EAVG = ', etavg(k)
!
!      write (6, 6610) k, 'E SD = ', esd
!
!      end do
1000  format(20(e14.7,1x))
      return
      end program


c ----------------------------------------------------------------------

c     quit is a subroutine used to terminate execution if there is
c     an error.

c     it is needed here because the subroutine that reads the parameters
c     (subroutine input) may call it.

c ----------------------------------------------------------------------

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
