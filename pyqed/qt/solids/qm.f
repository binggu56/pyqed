! ------------------------------------------------------------------
!     This computes the ground state energy and wavefunction and 
!     expectation values using quantum trajectory dynamics with 
!     friction 
! 
!     Bing Gu 
!     May 5, 2016 
! ----------------------------------------------------------------------

      program qm 

! --- common parameter modules 
      use cdat

      implicit double precision (a-h, o-z)

      include 'sizes.h'

      include 'qsats.h'

      include 'mpif.h'
      
      common /bincom/ bin, binvrs, r2min

!------------trajectory initial allocate------------------------

      real*8, dimension(NATOM3)         :: p0,alpha,x0,cf

      real*8, allocatable, dimension(:)   :: w,pe,ke,wp

      real*8, allocatable, dimension(:,:) :: x,p,ap,ar,rp,du,fr

      real*8, allocatable, dimension(:,:) :: ap_proc,du_proc,fr_proc,
     +                                       x_proc,p_proc,rp_proc,
     +       s1,cpp,crp,s1p,ar_proc,s2p, cp2,cr2, s2_sum

      real*8, allocatable, dimension(:,:) :: cp,cr,cp2_proc,cr2_proc,cpr

      real*8 :: gasdev

! --- arrays for pdf 

      real*8, allocatable, dimension(:)   :: gr,p_gr 

!------------------------------------------------------
      character*4 :: atom(NATOMS)
      real*8      :: am(NATOM3)
      integer*4   :: ndim,kmax
      integer     :: idum(NATOM3)

      integer     :: myid, ierr, numprocs, root, tag

      parameter (half=0.5d0)
      parameter (one=1.0d0)

      tag = 1

! --- initialize mpi environment

      call mpi_init(ierr)

      call mpi_comm_rank(mpi_comm_world, myid, ierr)

      call mpi_comm_size(mpi_comm_world, numprocs, ierr)

! --- define root processor 

      root = 0

! --- open files to restore data in root processor 

      if (myid == root) then

            open(11, file='en.dat',   status='new', action='write')
            open(12, file='traj.dat', status='new', action='write')
            open(113,file='pdf.dat',  status='new', action='write')


!---- read variables from IN

            open(10,file='IN', status='old', action='read')

            read(10,*) ntraj
            read(10,*) kmax,dt
            read(10,*) am0
            read(10,*) a0
            read(10,*) cf0
            read(10,*) iread
            read(10,*) NBIN_PDF

      close(10)

      write(*,6002) 
6002  format('------------------------------'/, 
     +       '    MPI VERSION of QTM_FRIC'/,
     +       '------------------------------'/)

      write (6, 6001) NREPS, NATOMS, NATOM3, NATOM6, NATOM7,
     +                NVBINS, RATIO, NIP, NPAIRS
6001  format ('compile-time parameters:'//,  
     +       'NREPS  = ', i6/,               
     +       'NATOMS = ', i6/,               
     +       'NATOM3 = ', i6/,               
     +       'NATOM6 = ', i6/,               
     +       'NATOM7 = ', i6/,               
     +       'NVBINS = ', i6/,               
     +       'RATIO  = ', f8.4/,              
     +       'NIP    = ', i6/,               
     +       'NPAIRS = ', i6/)    
      
      endif   
!    master work end


! --- passing input parameters to other processors 

      call mpi_bcast(ntraj,1,mpi_integer,root,
     +               mpi_comm_world,ierr)

      call mpi_bcast(dt,1,mpi_double_precision,root,
     +               mpi_comm_world,ierr)

      call mpi_bcast(a0,1,mpi_double_precision,root,
     +               mpi_comm_world,ierr)

      call mpi_bcast(am0,1,mpi_double_precision,root,
     +               mpi_comm_world,ierr)

      call mpi_bcast(cf0,1,mpi_double_precision,root,
     +               mpi_comm_world,ierr)

      call mpi_bcast(kmax,1,mpi_integer,root,mpi_comm_world,ierr)

! --- number of DOF, 3 * NATOMS

      Ndim = NATOM3

! --- density of atomic solid 

      den = 5.231d-3 
!      den = 4.61421d-3 ( RJ Hinde )

! --- number of trajectories per processor  

      ntraj_proc = ntraj/numprocs

! --- initialization for the potential energy surface computation 

      call vinit(r2min, bin)

      if(myid == root) then

      write (*, 6000) sqrt(r2min), bin

6000  format ('DEFINING potential energy grid parameters'//,
     +        'minimum R      = ', f10.5, ' bohr'/,
     +        'R**2 bin size  = ', f10.5, ' bohr**2'/)

      write (*, 6020)
6020  format ('using HFD-B(He) potential energy curve'/, '[R.A. ',
     +        'Aziz et al., Mol. Phys. vol. 61, p. 1487 (1987)]'/)

      endif   ! root print
      
      binvrs = one/bin

! --- read crystal lattice points.

      ltfile = 'lattice-file-180'

      if (myid == 0) then

      write (6, 6200) ltfile
6200  format ('READING crystal lattice from ', a16/)
      
      endif

      open (8, file=ltfile, status='old')

      read (8, *) nlpts

      if(myid == root) then

      if (nlpts.ne.NATOMS) then
         write (6, *) 'ERROR: number of atoms in lattice file = ', 
     +                 nlpts
         write (6, *) 'number of atoms in source code = ', NATOMS
         stop
      end if

      endif

! --- read the edge lengths of the supercell.
      
      read (8, *) xlen, ylen, zlen

! --- compute a distance scaling factor, the scaling factor is related
! --- to the density 

      den0 = dble(NATOMS)/(xlen*ylen*zlen)

! --- scale is a distance scaling factor, computed from the atomic
!     number density specified by the user.

      scale = exp(dlog(den/den0)/3.0d0)

      xlen=xlen/scale
      ylen=ylen/scale
      zlen=zlen/scale


      dxmax=half*xlen
      dymax=half*ylen
      dzmax=half*zlen


! --- lattice configuration read from ltfile 

      do i=1, NATOMS

         read (8, *) xtal(i, 1), xtal(i, 2), xtal(i, 3)

         xtal(i, 1)=xtal(i, 1)/scale
         xtal(i, 2)=xtal(i, 2)/scale
         xtal(i, 3)=xtal(i, 3)/scale

      end do

      close (8)
      
      if (myid == 0) then

      write (6, 6300) scale
6300  format ('supercell scaling factor computed from density = ',
     +         f12.8/)

      write (6, 6310) xlen, ylen, zlen
6310  format ('supercell edge lengths [bohr]         = ', 3f10.5/)

      write (6, 6320) xtal(NATOMS, 1), xtal(NATOMS, 2),
     +                xtal(NATOMS, 3)
6320  format ('final lattice point [bohr]            = ', 3f10.5/)
      
      endif

! --- this variable helps us remember the nearest-neighbor distance.

      rnnmin=-1.0d0

      do j=2, NATOMS

         dx=xtal(j, 1)-xtal(1, 1)
         dy=xtal(j, 2)-xtal(1, 2)
         dz=xtal(j, 3)-xtal(1, 3)

! ------ this sequence of if-then-else statements enforces the
!        minimum image convention. (Periodic Boundary Conditions)

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

         r = dsqrt(dx*dx+dy*dy+dz*dz)

         if (r.lt.rnnmin.or.rnnmin.le.0.0d0) rnnmin=r

      end do

      if(myid == 0) then

            write (6, 6330) rnnmin
6330        format ('nearest neighbor (NN) distance [bohr] = ', f10.5/)

            write (6, 6340) xlen/rnnmin, ylen/rnnmin, zlen/rnnmin
6340        format ('supercell edge lengths [NN distances] = ', 3f10.5/)

      endif

! --- compute interacting pairs.

      do i=1, NATOMS
            npair(i)=0
      end do

      nvpair=0

      do i = 1, NATOMS
      do j = i+1, NATOMS ! eliminate overcounting 

         if (j.ne.i) then

            dx=xtal(j, 1)-xtal(i, 1)
            dy=xtal(j, 2)-xtal(i, 2)
            dz=xtal(j, 3)-xtal(i, 3)

! --------- this sequence of if-then-else statements enforces the
!           minimum image convention.

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

! --------- interacting pairs are those for which r is less than a
!           certain cutoff amount. 

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
      
      if (myid == 0) then
            write (6, 6400) npair(1), nvpair
6400        format ('atom 1 interacts with ', i3, ' other atoms'//, 
     +              'total number of interacting pairs = ', i6/)
      endif

! --- allocate local arrays x,p,rp for root only (only those arrays only
!     exist at root proc) 

      if (myid == root) then

            allocate(s1(ndim+1,ndim+1),pe(ntraj),ke(ntraj),
     +               x(ndim,ntraj), p(ndim,ntraj),
     +               ap(ndim,ntraj),ar(ndim,ntraj),
     +               rp(ndim,ntraj),w(ntraj),s2_sum(16,ndim))
      endif

! --- allocate arrays for all processors 

        allocate(cp2(4,ndim),cr2(4,ndim))

        allocate(wp(ntraj_proc),cpr(ndim+1,2*ndim), s2p(16,ndim))

        allocate(fr_proc(ndim,ntraj_proc),du_proc(ndim,ntraj_proc),
     +         ap_proc(ndim,ntraj_proc),x_proc(ndim,ntraj_proc),
     +         p_proc(ndim,ntraj_proc),rp_proc(ndim,ntraj_proc),
     +         ar_proc(ndim,ntraj_proc))
      
        allocate(s1p(ndim+1,ndim+1),cpp(ndim+1,ndim),crp(ndim+1,ndim),
     +         cp2_proc(4,ndim),cr2_proc(4,ndim),
     +         cp(ndim+1,ndim),cr(ndim+1,ndim))

! --- initial parameters 
! --- dt : time interval 
! --- a0, x0, p0 : real,  initial wavefunction ~ exp(a0*(x-x0)^2 + I*p0*(x-x0))
! --- cf : array(Ndim), friction constant (would be different for each DOF), usually
!          set as constant cf0 
      
      dt2 = dt/2d0
      t   = 0d0

      pow = 6d0 ! cutoff for initial sampling 

      do i=1,Ndim
        am(i) = am0*1836.15d0
        x0(i) = 0.0d0
        p0(i) = 0d0
        alpha(i) = a0
        cf(i) = cf0
      enddo

      if(myid == root) then

! --- s2p : basis overlap matrix in each processor 
! --- s2_sum : sum of s2p 

            s2p = 0d0 
            s2_sum = 0d0 

        w = 1d0/dble(ntraj)

! ----- initial setup for trajectories

        write(*,6005) numprocs, ntraj_proc        
6005    format('Num of cores                 =', i6/,
     +         'Num of trajectories per core =', i6)

! ----- check number of trajectories

        if (mod(ntraj,numprocs) .ne. 0) then
          
          write(*,6004) mod(ntraj,numprocs)
6004      format('Bad number of trajectories.'/, 
     +           'The remainder is',i6/)
          stop
        endif

! --- intial print 

      write(*,1010) Ntraj,Ndim,kmax,dt,cf(1),am(1)
1010  format('Initial Conditions'//,
     +       'Ntraj   = ' , i6/,                          
     +       'Ndim    = ' , i6/,                         
     +       'kmax    = ' , i6/,  'dt      = ' , f8.4/,  
     +       'fric    = ' , f8.4/,'Mass    = ' , e14.6/) 

      write(*,1011) alpha(1),x0(1),p0(1)
1011  format('Initial Wavepacket'//,'alpha0   = ' , f8.4/,  
     +       'center   = ' , f8.4/,'momentum = ' , f8.4/)

! --- intial {x,p,r}, sampling, root

      if(iread == 0) then
      
      call seed(idum,Ndim)

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
      enddo
      
! --- continue job from last checkpoint
      elseif(iread == 1) then

      open(11, file = 'temp.dat')

      do i=1,ntraj
        do j=1,ndim
          read(11,*) x(j,i),p(j,i),rp(j,i)
        enddo
      enddo

      close(11)

      endif

!---- expectation value of x(1,ntraj)--------

      av = 0d0
      do i=1,ntraj
        av = av+w(i)*x(1,i)
      enddo
      write(*,6008) av
6008  format('Expectation value of x(1,ntraj) = ', f10.6)

      endif ! root work end 

      time = mpi_wtime()

! --- send required info {cf,x,p,r,w} for slave nodes 
      call mpi_scatter(x,ndim*ntraj_proc,mpi_double_precision,
     +                 x_proc,ndim*ntraj_proc,mpi_double_precision,
     +                 root,mpi_comm_world,ierr)

      call mpi_scatter(p,ndim*ntraj_proc,mpi_double_precision,
     +                 p_proc,ndim*ntraj_proc,mpi_double_precision,
     +                 root,mpi_comm_world,ierr)

      call mpi_scatter(rp,ndim*ntraj_proc,mpi_double_precision,
     +                 rp_proc,ndim*ntraj_proc,mpi_double_precision,
     +                 root,mpi_comm_world,ierr)

      call mpi_scatter(w,ntraj_proc,mpi_double_precision,
     +                 wp,ntraj_proc,mpi_double_precision,
     +                 root,mpi_comm_world,ierr)

      if(myid == root) then

        time = mpi_wtime() - time

        write(*,6678) time

6678    format('Scatter work finished. Time for initial scattering',
     +         f12.6/)
        
        write(*,6679) 

6679    format('Now propagate quantum trajectories...')
      endif

! --- bin for pair distribution function 

!      NBIN_PDF = 400 

! --- allocate arrays for pdf 

      if(rank == 0) allocate(gr(NBIN_PDF))

      allocate(p_gr(NBIN_PDF))
      

! --- trajectories propagation      

      do 10 kt=1,kmax

        t=t+dt
 
        time = mpi_wtime()

! ----- root, compute quantum force x(Ndim,Ntraj),p,r

        call prefit(ntraj_proc,ndim,wp,x_proc,p_proc,rp_proc,
     +              s1p,cpp,crp)

        call mpi_reduce(s1p,s1,(ndim+1)*(ndim+1),MPI_DOUBLE_PRECISION,
     +                  MPI_SUM,root,MPI_COMM_WORLD,ierr)
      
        call mpi_reduce(cpp,cp,(ndim+1)*ndim,MPI_DOUBLE_PRECISION,
     +                  MPI_SUM,root,MPI_COMM_WORLD,ierr)
      
        call mpi_reduce(crp,cr,(ndim+1)*ndim,MPI_DOUBLE_PRECISION,
     +                  MPI_SUM,root,MPI_COMM_WORLD,ierr)
        
        time = mpi_wtime() - time

! ----- linear fitting for the first step
        if (myid == root) then

!           write(*,6688) time
!6688       format('time to collect matrix elements',f12.6/)

!          time = mpi_wtime()

          call fit(ntraj,ndim,cp,cr,s1,am,w)

!          time = mpi_wtime()-time

!          write(*,6680) time

!6680      format('time for linear fit at root', f12.6/)
        endif

!        call MPI_BCAST(cpr,(ndim+1)*2*ndim,MPI_DOUBLE_PRECISION,root,
!     +       MPI_COMM_WORLD,ierr)

        call mpi_bcast(cp,(ndim+1)*ndim,mpi_double_precision,root,
     +                   mpi_comm_world,ierr)

        call mpi_bcast(cr,(ndim+1)*ndim,mpi_double_precision,root,
     +                   mpi_comm_world,ierr)

! ----  get approximate {p,r}, slave processors  
        
        call aver(ndim,ntraj_proc,ntraj,x_proc,cp,cr,ap_proc,ar_proc)

! ----- collect approximated {ap,ar}
      
!        call MPI_GATHER(ap_proc,ndim*ntraj_proc,mpi_double_precision,ap,
!     +                  ndim*ntraj_proc,mpi_double_precision,ROOT,
!     +                  MPI_COMM_WORLD,ierr)
!
!        call MPI_GATHER(ar_proc,ndim*ntraj_proc,mpi_double_precision,ar,
!     +                  ndim*ntraj_proc,mpi_double_precision,ROOT,
!     +                  MPI_COMM_WORLD,ierr)
!
! ---- compute averages of S2 = f X f, f = (1,x,x^2,x^3)

        call  aver_proc(ndim,ntraj_proc,wp,cp2_proc,cr2_proc,s2p,
     +                  x_proc,p_proc,rp_proc,ap_proc,ar_proc) 

        call mpi_barrier(mpi_comm_world,ierr)
        
        call mpi_reduce(cp2_proc,cp2,4*ndim,
     +       MPI_DOUBLE_PRECISION,mpi_sum,root,MPI_COMM_WORLD,ierr)

        call mpi_reduce(cr2_proc,cr2,4*ndim,
     +       MPI_DOUBLE_PRECISION,mpi_sum,root,MPI_COMM_WORLD,ierr)

        call mpi_reduce(s2p,s2_sum,16*ndim,
     +       MPI_DOUBLE_PRECISION,mpi_sum,root,MPI_COMM_WORLD,ierr)

! ----  do second fitting, root

        if(myid == root) then

!          time = mpi_wtime()-time
!          write(*,6689) time
!6689      format('time to gather approximated p,r',f12.6/)
!          time = mpi_wtime()

! ------- most time consuming part 

          call fit2(ndim,ntraj,w,cp2,cr2,s2_sum)

!          time = mpi_wtime() - time
!          write(*,6691) time
!6691      format('time to do second fit at root',f12.6/)

        endif

!        call MPI_BARRIER(mpi_comm_world,ierr)

        call mpi_bcast(cp2,ndim*4,mpi_double_precision,root,
     +                   mpi_comm_world,ierr)

        call mpi_bcast(cr2,ndim*4,mpi_double_precision,root,
     +                   mpi_comm_world,ierr)

        call mpi_barrier(mpi_comm_world,ierr)

        call comp(am,wp,ntraj_proc,ndim,eup,cp,cr,cp2,cr2,
     +            x_proc,ap_proc,du_proc,fr_proc)

        
! ------- propagate trajectory in each proc for one time step
          call traj(myid,dt,ndim,ntraj_proc,cf,am,x_proc,p_proc,
     +            rp_proc,ap_proc,wp,du_proc,fr_proc,proc_po,enk_proc)


! ------- if last step, compute pair distribution function 

          if(kt == kmax) then 

            call pdf(ndim,ntraj_proc,NBIN_PDF,wp,x_proc,p_gr)

          endif  

          call mpi_reduce(p_gr,gr,NBIN_PDF,MPI_DOUBLE_PRECISION,
     +         mpi_sum,root,MPI_COMM_WORLD,ierr)

! ----- set values to 0 to do mpi_reduce to get the full {x(ndim,ntraj),p,r} matrix
! ----- collect info from other nodes {x,p,r}, then compute quantum potential, by root
!        time = mpi_wtime()
!
!        call MPI_GATHER(x_proc,ndim*ntraj_proc,mpi_double_precision,x,
!     +                  ndim*ntraj_proc,mpi_double_precision,ROOT,
!     +                  MPI_COMM_WORLD,ierr)
!
!        call MPI_GATHER(p_proc,ndim*ntraj_proc,mpi_double_precision,p,
!     +                  ndim*ntraj_proc,mpi_double_precision,ROOT,
!     +                  MPI_COMM_WORLD,ierr)
!
!        call MPI_GATHER(rp_proc,ndim*ntraj_proc,mpi_double_precision,rp,
!     +                  ndim*ntraj_proc,mpi_double_precision,ROOT,
!     +                  MPI_COMM_WORLD,ierr)
!
! ----- root, gather energy components  

        call mpi_reduce(proc_po,po,1,MPI_DOUBLE_PRECISION,MPI_SUM,
     +                  root,MPI_COMM_WORLD,ierr)

        call mpi_reduce(eup,qu,1,MPI_DOUBLE_PRECISION,MPI_SUM,
     +                  root,MPI_COMM_WORLD,ierr)

        call mpi_reduce(enk_proc,enk,1,MPI_DOUBLE_PRECISION,MPI_SUM,
     +                  root,MPI_COMM_WORLD,ierr)


! ----- write data to files 
        if(myid == root) then
          
          tot = po+enk+qu

          write(11,1000) t,enk,po,qu,tot
          call flush(11)

          write(12,1000) t,(x(1,i),i=1,20),(p(1,i),i=1,20) 

          do i=1,NBIN_PDF
            write(113,1000) i*bin_pdf, gr(i)
          enddo 

        endif

10    enddo


! --- record final data 
! --- transfer data to output 
      call MPI_GATHER(x_proc,ndim*ntraj_proc,mpi_double_precision,x,
     +                ndim*ntraj_proc,mpi_double_precision,ROOT,
     +                MPI_COMM_WORLD,ierr)

      call MPI_GATHER(p_proc,ndim*ntraj_proc,mpi_double_precision,p,
     +                ndim*ntraj_proc,mpi_double_precision,ROOT,
     +                MPI_COMM_WORLD,ierr)

      call MPI_GATHER(rp_proc,ndim*ntraj_proc,mpi_double_precision,rp,
     +                ndim*ntraj_proc,mpi_double_precision,ROOT,
     +                MPI_COMM_WORLD,ierr)

      if (myid == root) then

      open(13,file='temp.dat',action='write')
      
        do i=1,Ntraj
          do j=1,Ndim
            write(13,1000) x(j,i),p(j,i),rp(j,i)
          enddo
        enddo

      close(13)

      close(11)
      close(12)


! --- deallocate arrays
      deallocate(fr_proc,du_proc,ap_proc,x_proc,p_proc,rp_proc)


      write(*,1020) tot 
1020  format('E(total) at final time step =', f10.5/ ,
     +       'MISSION COMPLETE.')
      
      endif ! root
      
!          call local(q,vloc,dv)

!       convert to eV 
!          vloc = vloc*27.211d0
! ------ convert to kelvin per atom.

!         tloc=tloc/(3.1668513d-6*dble(NATOMS))
!          vloc=vloc/(3.1668513d-6*dble(NATOMS))

! ------ accumulate the results.

!         vtavg(k)=vtavg(k)+vloc
!         vtavg2(k)=vtavg2(k)+(vloc)**2
!         etavg(k)=etavg(k)+tloc+vloc
!         etavg2(k)=etavg2(k)+(tloc+vloc)**2

!350      continue

!      end do

!      goto 300

! --- account for overshooting.

!600   loop=loop-1

!      write (6, 6600) loop
!6600  format ('number of snapshots = ', i6/)

! --- compute the averages and standard deviations.

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

      call mpi_finalize(ierr)

1000  format(20(e14.7,1x))
      
      stop
      end program


! ----------------------------------------------------------------------

!     quit is a subroutine used to terminate execution if there is
!     an error.

!     it is needed here because the subroutine that reads the parameters
!     (subroutine input) may call it.

! ----------------------------------------------------------------------

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
! ----------------------------------------------------------------
! --- propagation of trajs for one time step
! --- each processor have part of the full matrix {x,p,r}, 
!     of size (NDIM,NTRAJ_PROC), run independently
! ----------------------------------------------------------------
      subroutine traj(myid,dt,ndim,ntraj_proc,cf,am,x_proc,p_proc,
     +                rp_proc,ap_proc,wp,
     +                du_proc,fr_proc,proc_po,enk_proc)

      use cdat

      implicit real*8(a-h, o-z)

      include 'sizes.h'

      include 'qsats.h'
      
      integer*4,intent(in)    :: myid,ntraj_proc,ndim
      real*8,   intent(in)    :: dt
      real*8, intent(in), dimension(ntraj_proc)  :: wp
      real*8, intent(in), dimension(ndim) :: am,cf
      real*8, intent(inout), dimension(ndim,ntraj_proc) :: x_proc,
     +       p_proc,rp_proc

      real*8, dimension(ndim,ntraj_proc) :: du_proc,fr_proc,ap_proc
      real*8, dimension(ndim)       :: dv,dvl

      real*8, dimension(ntraj_proc)  :: pe

      real*8 :: q(NATOM3)

      dt2 = dt/2d0
      pe = 0d0

! ----- half-step increments of momenta & full step increment of positions
!      do i = myid*ntraj_proc+1,(myid+1)*ntraj_proc
      do i=1,ntraj_proc
        
        do m=1,NATOM3
          q(m) = x_proc(m,i)
        enddo
        
!       force and long-range force

        call local(q,vloc,dv)
        call long_force(q,vlong,dvl) 

        pe(i) = vloc+vlong

        do j=1,Ndim

            p_proc(j,i) = p_proc(j,i)+ ( -dv(j)-dvl(j)-
     +                    du_proc(j,i) - cf(j)*p_proc(j,i)/am(j) ) * dt

            x_proc(j,i) = x_proc(j,i) + ap_proc(j,i)*dt/am(j)

            rp_proc(j,i) = rp_proc(j,i) + fr_proc(j,i)*dt

        enddo

      enddo

! --- update potential, kinetic, and total energy each proc
      
      enk_proc = 0d0 
      
      do i=1,ntraj_proc 
        do j=1,ndim
          enk_proc = enk_proc + p_proc(j,i)**2/(2d0*am(j))*wp(i)
        enddo 
      enddo 
      
      proc_po = 0d0

      do i = 1,Ntraj_proc
        proc_po = pe(i)*wp(i)+proc_po
      enddo

      return
      end subroutine

!--------------------------------------

!     distribute each processor with part of the work to get S=f*f

!-----------------------------------------------------------

      subroutine prefit(ntraj_proc,ndim,wp,x_proc,p_proc,rp_proc,
     +           s1,cp,cr)

      use cdat

      implicit real*8(a-h,o-z)

      integer*4, intent(in) :: ntraj_proc,ndim


      real*8, intent(in), dimension(ndim,ntraj_proc) :: x_proc,
     +                                                p_proc,rp_proc
      real*8, intent(out), dimension(ndim+1,ndim+1) :: s1

      real*8 :: f(ndim+1),wp(ntraj_proc),cp(ndim+1,ndim),cr(ndim+1,ndim)

      s1 = 0d0
      cp = 0d0
      cr = 0d0

      do i=1,ntraj_proc

        call basis(ndim,ntraj_proc,i,x_proc,f)
        
        do k2 = 1,ndim+1
          do k1=1,k2
            s1(k1,k2) = s1(k1,k2)+f(k1)*f(k2)*wp(i)
          enddo
        enddo
      enddo

      do i=1,ntraj_proc
        
        call basis(ndim,ntraj_proc,i,x_proc,f)
        
        do k=1,ndim
          do j=1,ndim+1
            cp(j,k) = cp(j,k)+f(j)*p_proc(k,i)*wp(i)
            cr(j,k) = cr(j,k)+f(j)*rp_proc(k,i)*wp(i)
          enddo
        enddo
      enddo

      do k2=1,ndim+1
        do k1=1,k2
          s1(k2,k1) = s1(k1,k2)
        enddo
      enddo


      return
      end subroutine
