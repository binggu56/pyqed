c     QSATS version 1.0 (3 March 2011)

c     file name: parent.f

c ----------------------------------------------------------------------

c     this is the parent process that runs on node 0.

c     errchk is a subroutine called after every MPI subroutine that
c     checks the MPI error code and reports any errors.

c ----------------------------------------------------------------------

      subroutine parent(ierror)

      implicit double precision (a-h, o-z)

      include 'sizes.h'

      include 'qsats.h'

      include 'mpif.h'

      dimension istat(MPI_STATUS_SIZE)

      dimension imsg(9), fmsg(6)

      dimension isent(NREPS), ikeep(NATOMS), replic(NATOM7)

      dimension rstate(8)

      parameter (half=0.5d0)
      parameter (two=2.0d0)
      parameter (one=1.0d0)

c ======================================================================
c     PART ONE: INITIALIZATION
c ======================================================================

      ierror=0

c --- read input file.

      call input

      write (6, 6100) ltfile, spfile, svfile
6100  format ('lattice file name  = ', a16/,
     +        'snapshot file name = ', a16/,
     +        'save file name     = ', a16/)

      if (idebug.eq.0) write (6, 6110) idebug, 'NONE'
      if (idebug.eq.1) write (6, 6110) idebug, 'MINIMAL'
      if (idebug.eq.2) write (6, 6110) idebug, 'LOW'
      if (idebug.eq.3) write (6, 6110) idebug, 'MEDIUM'
      if (idebug.eq.4) write (6, 6110) idebug, 'HIGH'

6110  format ('debug level = ', i1,' or ', a8/)

c --- read the potential energy curve.

      call vinit(r2min, bin)

c --- read crystal lattice points.

      write (6, 6200) ltfile
6200  format ('READING crystal lattice from ', a16/)

      open (8, file=ltfile, status='old', err=901)

      read (8, *, err=902) nlpts

      if (nlpts.ne.NATOMS) then
         write (6, *) 'ERROR: number of atoms in lattice file = ', nlpts
         write (6, *) 'number of atoms in source code = ', NATOMS
         call quit
      end if

c --- read the edge lengths of the supercell.

      read (8, *, err=903) xlen, ylen, zlen

      den0=dble(NATOMS)/(xlen*ylen*zlen)

c --- compute a distance scaling factor.

      scale=exp(dlog(den/den0)/3.0d0)

      write (6, 6300) scale
6300  format ('supercell scaling factor computed from density = ',
     +        f12.8/)

c --- scale is a distance scaling factor, computed from the atomic
c     number density specified by the user.

      xlen=xlen/scale
      ylen=ylen/scale
      zlen=zlen/scale
      
      dxmax=half*xlen
      dymax=half*ylen
      dzmax=half*zlen
      
      do i=1, NATOMS
         
         read (8, *, err=904) xtal(i, 1), xtal(i, 2), xtal(i, 3)
      
         xtal(i, 1)=xtal(i, 1)/scale
         xtal(i, 2)=xtal(i, 2)/scale
         xtal(i, 3)=xtal(i, 3)/scale
      
      end do

      close (8)

c --- this helps us remember the nearest-neighbor distance.

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

      write (6, 6310) rnnmin
6310  format ('nearest neighbor (NN) distance [bohr] = ', f10.5/)

      write (6, 6320) xtal(NATOMS, 1), xtal(NATOMS, 2),
     +                xtal(NATOMS, 3)
6320  format ('final lattice point [bohr]            = ', 3f10.5/)

      write (6, 6330) xlen, ylen, zlen
6330  format ('supercell edge lengths [bohr]         = ', 3f10.5/)
      
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
     +        'total number of interacting pairs = ', i6)

      if (idebug.ge.2) then

         write (6, 6401)
6401     format (/'interaction pair vectors for atom 1 ',
     +            '[NN distances]:'/)

         do i=1, npair(1)
            ip=ipairs(i, 1)
            d=sqrt(vpvec(1, ip)**2+vpvec(2, ip)**2+vpvec(3, ip)**2)/
     +        rnnmin
            write (6, 6410) ip, ivpair(2, ip), vpvec(1, ip)/rnnmin,
     +                      vpvec(2, ip)/rnnmin, vpvec(3, ip)/rnnmin, d
6410        format ('vector # ', i3, ' to atom ', i4, ': ',
     +              3(1x, f9.5), ' length = ', f8.5)
         end do

      end if

c --- set the displacement vectors for all replicas to zero.

      write (6, 6500)
6500  format (/'SETTING initial configuration to zero'/)

      do j=1, NREPS
      do i=1, NATOM3
         path(i, j)=0.0
      end do
      end do

c --- initialize random number generator.

      call rsetup

c --- now see if there is an old set of displacement vectors from a
c     previous run.  if not, jump head to line 200.

      open (8, file=svfile, form='unformatted', status='old', err=200)

      write (6, 6510) svfile
6510  format ('READING initial configuration from ', a16/)

      do j=1, NREPS

         read (8) (rstatv(i, j), i=1, 8)
         read (8) (path(i, j), i=1, NATOM3)

      end do

      close (8)

200   if (idebug.ge.3) then

         write (6, 6170)
6170     format ('x(1) and rstatv(1) values for each replica:'/)

         do j=1, NREPS

            write (6, 6180) j, path(1, j), rstatv(1, j)
6180        format (i5, 1x, f15.9, 1x, f20.1)

         end do

         write (6, *) ''

      end if

c --- this is the output file where snapshots of the replicas will be
c     stored for analysis by another program.

      open (10, file=spfile, form='unformatted')

c --- initialize MPI.

      MPI_R=MPI_DOUBLE_PRECISION

      call MPI_COMM_SIZE(MPI_COMM_WORLD, ntasks, ierr)

      call errchk(0, ierr, 100000)

      write (6, 6600) ntasks-1
6600  format ('number of child processes = ', i3/)

      if (ntasks-1.gt.60) then

         write (6, 6610)
6610     format ('too many child processes; expand the iwork array.'/
     +           'also note that write statements for HIGH '
     +           'debugging level may fail on some systems.')

         call quit

      end if

c --- this array just counts how evenly the workload was spread among
c     the child processes.

      do i=1, ntasks-1
         iwork(i)=0
      end do

c --- broadcast integer constants to all child processes.

      imsg(1)=NATOMS
      imsg(2)=NATOM3
      imsg(3)=NATOM6
      imsg(4)=NATOM7
      imsg(5)=NREPS
      imsg(6)=NIP
      imsg(7)=NPAIRS
      imsg(8)=NVBINS
      imsg(9)=idebug

      do itask=1, ntasks-1

         call MPI_SEND(imsg,
     +                 9,
     +                 MPI_INTEGER,
     +                 itask,
     +                 0101,
     +                 MPI_COMM_WORLD,
     +                 ierr)

         call errchk(0, ierr, 100101)

      end do

      if (idebug.gt.0) open (9, file='debug.log')

      if (idebug.eq.1) write (9, 6110) idebug, 'MINIMAL'
      if (idebug.eq.2) write (9, 6110) idebug, 'LOW'
      if (idebug.eq.3) write (9, 6110) idebug, 'MEDIUM'
      if (idebug.eq.4) write (9, 6110) idebug, 'HIGH'

c --- broadcast floating-point constants to all child processes.

      fmsg(1)=tau
      fmsg(2)=bin
      fmsg(3)=r2min
      fmsg(4)=amass
      fmsg(5)=aa
      fmsg(6)=bb

      do itask=1, ntasks-1

         call MPI_SEND(fmsg,
     +                 6,
     +                 MPI_R,
     +                 itask,
     +                 0102,
     +                 MPI_COMM_WORLD,
     +                 ierr)

         call errchk(0, ierr, 100102)

      end do

c --- broadcast the interacting-pair vectors to all child processes.

      do itask=1, ntasks-1

         call MPI_SEND(vpvec,
     +                 3*NPAIRS,
     +                 MPI_R,
     +                 itask,
     +                 0103,
     +                 MPI_COMM_WORLD,
     +                 ierr)

         call errchk(0, ierr, 100103)

      end do

c --- broadcast the list of atom id numbers for the interacting pairs
c     to all child processes.

      do itask=1, ntasks-1

         call MPI_SEND(ivpair,
     +                 2*NPAIRS,
     +                 MPI_INTEGER,
     +                 itask,
     +                 0104,
     +                 MPI_COMM_WORLD,
     +                 ierr)

         call errchk(0, ierr, 100104)

      end do

c --- broadcast the size of each stencil to all child processes.  all
c     stencils should be the same size, but we treat this as a variable.

      do itask=1, ntasks-1

         call MPI_SEND(npair,
     +                 NATOMS,
     +                 MPI_INTEGER,
     +                 itask,
     +                 0105,
     +                 MPI_COMM_WORLD,
     +                 ierr)

         call errchk(0, ierr, 100105)

      end do

c --- broadcast the list of interacting pair id numbers that define the
c     stencils to all child processes.

      do itask=1, ntasks-1

         call MPI_SEND(ipairs,
     +                 NIP*NATOMS,
     +                 MPI_INTEGER,
     +                 itask,
     +                 0106,
     +                 MPI_COMM_WORLD,
     +                 ierr)

         call errchk(0, ierr, 100106)

      end do

c --- broadcast the potential energy curve V(R) to all child processes.

      do itask=1, ntasks-1

         call MPI_SEND(v,
     +                 2*NVBINS,
     +                 MPI_R,
     +                 itask,
     +                 0107,
     +                 MPI_COMM_WORLD,
     +                 ierr)

         call errchk(0, ierr, 100107)

      end do

      if (idebug.gt.0) write (9, *) 'end parent PART ONE'
      if (idebug.gt.0) write (9, *) ''

c ======================================================================
c     PART TWO: PERFORMING THE SIMULATION
c ======================================================================

c --- initialization of various progress counters.

c --- this is how many iterations we have done.

      loop=0

c --- these tell us about the acceptance ratio for the atom moves.

      ztacc=0.0d0
      ztrej=0.0d0

      ztacc0=0.0d0
      ztrej0=0.0d0

300   loop=loop+1

c --- these counters make sure that we don't lose a replica somewhere in
c     the ether. we use them to count how many replicas have been sent and
c     received.

      nsent=0
      nrcvd=0

c --- this is a list of flags that are zero for replicas that haven't yet
c     been sent to a child for processing, positive for replicas that have
c     been sent, and negative for replicas that have been processed and
c     returned to the parent.

c     isent(n) is set to the (positive) task id of the receiving child
c     process when a replica is sent.  this is basically leaving a trail
c     of crumbs so that we can track down the replicas and ask the children
c     to return them to us.

      do nrep=1, NREPS
         isent(nrep)=0
      end do

c --- first do all odd replicas.

      call oddrep(loop, nsent, nrcvd, MPI_R)

c --- then do all even replicas.

      call evnrep(loop, nsent, nrcvd, MPI_R)

c --- check for lost replicas.

      if (nsent.ne.NREPS.or.nrcvd.ne.NREPS) then
         write (6, *) 'replicas have been lost!'
         write (6, *) 'nsent = ', nsent
         write (6, *) 'nrcvd = ', nrcvd
         ierror=1
      end if

c --- take a snapshot every so often.

      if (mod(loop, nprint).eq.0) then

         zacc=ztacc-ztacc0
         zrej=ztrej-ztrej0

         ztacc0=ztacc
         ztrej0=ztrej

         if (idebug.gt.0) then
            write (9, 9400) zacc, zrej, 100.0d0*zacc/(zacc+zrej)
9400        format ('accepted = ', f11.0, 1x,
     +              'rejected = ', f11.0, 3x, 
     +              '% accepted = ', f6.2)
            call flush(9)
         end if

c ------ we only actually take snapshots of every 11th replica.

         do k=1, NREPS, 11
            write (10) (path(i, k), i=1, NATOM3)
         end do

      end if

c --- do the next loop if needed.

      if (loop.lt.nloop) goto 300

c --- otherwise save a checkpoint file.

      write (6, 6810) svfile
6810  format ('SAVING final configuration to ', a16/)

      open (8, file=svfile, form='unformatted')

      do k=1, NREPS
         write (8) (rstatv(i, k), i=1, 8)
         write (8) (path(i, k), i=1, NATOM3)
      end do

      if (idebug.ge.3) then

         write (6, 6170)

         do k=1, NREPS
            write (6, 6180) k, path(1, k), rstatv(1, k)
         end do

         write (6, *) ''

      end if

      close (8)

      close (10)

      if (idebug.gt.0) then
         write (9, *) ''
         write (9, *) 'QSATS is done!'
         write (9, *) ''
      end if

c --- show how much work every child did.

      if (idebug.gt.0) then
         do i=1, ntasks-1
            write (9, 9100) i, iwork(i)
9100        format ('task ', i3, ' received ', i9, ' replicas')
         end do
      end if
      
c --- tell the children we're all done.
         
      do itask=1, ntasks-1

         imsg(1)=0

         call MPI_SEND(imsg,
     +                 1,
     +                 MPI_INTEGER, 
     +                 itask,
     +                 0204,
     +                 MPI_COMM_WORLD,
     +                 ierr)

         call errchk(0, ierr, 100204)
         
      end do

      write (6, 6900) ztacc
6900  format ('total number of accepted moves = ', f20.1)

      write (6, 6901) ztrej
6901  format ('total number of rejected moves = ', f20.1/)

      if (idebug.gt.0) write (9, *) ''
      if (idebug.gt.0) write (9, *) 'end parent PART TWO'

      return

901   write (6, *) 'error opening lattice file'
      goto 999

902   write (6, *) 'error reading number of atoms from lattice file'
      goto 999 

903   write (6, *) 'error reading (unscaled) supercell edge lengths'
      goto 999
         
904   write (6, *) 'error reading atom number ', i
      goto 999
      
999   call quit

      return
      end
