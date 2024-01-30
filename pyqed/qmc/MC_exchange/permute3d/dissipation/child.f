c     QSATS version 1.0 (3 March 2011)

c     file name: child.f

c ----------------------------------------------------------------------

c     this is the child process that runs on all nodes except node 0
c     (which is running the parent process).

c ----------------------------------------------------------------------

      subroutine child(MPI_R)

      implicit double precision (a-h, o-z)

      include 'mpif.h'

      include 'sizes.h'

      common /rancm1/ rscale
 
      dimension replic(NATOM6), npair(NATOMS), rv(NATOM3)

      dimension istat(MPI_STATUS_SIZE)

      dimension vpvec(3, NPAIRS)
      dimension ivpair(2, NPAIRS)
      dimension ipairs(NIP, NATOMS)

      dimension xx(NATOMS), yy(NATOMS), zz(NATOMS)

      dimension r2old(NATOMS), r2new(NATOMS), v1(NATOMS), v2(NATOMS)

      dimension v(2, NVBINS)

      dimension imsg(9), fmsg(6)

      dimension rstate(8)

      parameter (half=0.5d0)
      parameter (two=2.0d0)
      parameter (one=1.0d0)

c ======================================================================
c     PART ONE: INITIALIZATION
c ======================================================================

c --- numerical factor for random number generator.

      rscale=1.0d0/4294967088.0d0

c --- determine which process this is and store it in myid.

      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)

c --- receive all of the information that is broadcast by the parent
c     process.

c --- first receive some integer constants.  these are primarily used to
c     check that the arrays are properly dimensioned.

      call MPI_RECV(imsg,
     +              9,
     +              MPI_INTEGER,
     +              0,
     +              0101,
     +              MPI_COMM_WORLD,
     +              istat,
     +              ierr)

      call errchk(myid, ierr, 200101)

      istop=0

      if (imsg(1).ne.NATOMS) then
         write (6, *) 'size mismatch 1: ', imsg(1)
         istop=1
      end if

      if (imsg(2).ne.NATOM3) then
         write (6, *) 'size mismatch 2: ', imsg(2)
         istop=1
      end if

      if (imsg(3).ne.NATOM6) then
         write (6, *) 'size mismatch 3: ', imsg(3)
         istop=1
      end if

      if (imsg(4).ne.NATOM7) then
         write (6, *) 'size mismatch 4: ', imsg(4)
         istop=1
      end if

      if (imsg(5).ne.NREPS) then
         write (6, *) 'size mismatch 5: ', imsg(5)
         istop=1
      end if

      if (imsg(6).ne.NIP) then
         write (6, *) 'size mismatch 6: ', imsg(6)
         istop=1
      end if

      if (imsg(7).ne.NPAIRS) then
         write (6, *) 'size mismatch 7: ', imsg(7)
         istop=1
      end if

      if (imsg(8).ne.NVBINS) then
         write (6, *) 'size mismatch 8: ', imsg(8)
         istop=1
      end if

      if (istop.eq.1) call quit

      idebug=imsg(9)

c --- debugging output.

      if (idebug.eq.4) write (30+myid, *) 'idebug = ', idebug

c --- next receive some floating-point constants.

      call MPI_RECV(fmsg,
     +              6,
     +              MPI_R,
     +              0,
     +              0102,
     +              MPI_COMM_WORLD,
     +              istat,
     +              ierr)

      call errchk(myid, ierr, 200102)

      tau=fmsg(1)
      bin=fmsg(2)
      r2min=fmsg(3)
      amass=fmsg(4)
      aa=fmsg(5)
      bb=fmsg(6)

      if (idebug.eq.4) then
         write (30+myid, *) 'tau = ', tau
         write (30+myid, *) 'bin = ', bin
         write (30+myid, *) 'r2min = ', r2min
         write (30+myid, *) 'amass = ', amass
         write (30+myid, *) 'aa = ', aa
         write (30+myid, *) 'bb = ', bb
      end if

c --- compute the inverse of the potential energy V(R) bin width, to
c     avoid unnecessary divisions.

      binvrs=one/bin

c --- compute gaussian scaling parameters.

      gscale=sqrt(half*tau/amass)
      gscal2=sqrt(tau/amass)

c --- next receive the vectors that connect pairs of atoms in a stencil.

      call MPI_RECV(vpvec,
     +              3*NPAIRS,
     +              MPI_R,
     +              0,
     +              0103,
     +              MPI_COMM_WORLD,
     +              istat,
     +              ierr)

      call errchk(myid, ierr, 200103)

c --- next receive the list of pairs of atoms.

      call MPI_RECV(ivpair,
     +              2*NPAIRS,
     +              MPI_INTEGER,
     +              0,
     +              0104,
     +              MPI_COMM_WORLD,
     +              istat,
     +              ierr)

      call errchk(myid, ierr, 200104)

c --- next receive the number of atoms that belong to each atom's stencil.
c     this should really be the same for every atom for a regular crystal
c     lattice, but we treat it as a variable.

      call MPI_RECV(npair,
     +              NATOMS,
     +              MPI_INTEGER,
     +              0,
     +              0105,
     +              MPI_COMM_WORLD,
     +              istat,
     +              ierr)

      call errchk(myid, ierr, 200105)

c --- next receive the pairs that constitute each atom's stencil.

      call MPI_RECV(ipairs,
     +              NIP*NATOMS,
     +              MPI_INTEGER,
     +              0,
     +              0106,
     +              MPI_COMM_WORLD,
     +              istat,
     +              ierr)

      call errchk(myid, ierr, 200106)

c --- next receive the potential energy curve V(R) for interpolation.

      call MPI_RECV(v,
     +              2*NVBINS,
     +              MPI_R,
     +              0,
     +              0107,
     +              MPI_COMM_WORLD,
     +              istat,
     +              ierr)

      call errchk(myid, ierr, 200107)

      if (idebug.eq.4) then
         write (30+myid, *) 'child moving to PART TWO'
         call flush(30+myid)
      end if

c ======================================================================
c     PART TWO: PERFORMING THE SIMULATION
c ======================================================================

100   idrep=0

      nacc=0
      nrej=0

c --- send request for data (message type 1201) to parent.  the first
c     time through, or if we are waiting for all children to sync up,
c     there are no results to send back to the parent, so we indicate
c     this by setting idrep=0 just above, and then sending this to
c     the parent in imsg(1).

200   imsg(1)=idrep

      imsg(2)=nacc
      imsg(3)=nrej

      call MPI_SEND(imsg,
     +              3,
     +              MPI_INTEGER,
     +              0,
     +              1201,
     +              MPI_COMM_WORLD,
     +              ierr)

      call errchk(myid, ierr, 201201)

c --- on the other hand, if there are results to send back, then we
c     do so here.

      if (idrep.gt.0) then

c ------ first we send a message of type 1202 that contains the atoms'
c        new positions.

         call MPI_SEND(replic,
     +                 NATOM3,
     +                 MPI_R,
     +                 0,
     +                 1202,
     +                 MPI_COMM_WORLD,
     +                 ierr)

         call errchk(myid, ierr, 201202)

c ------ then we send a message of type 1203 that contains the updated
c        random number generator state vector.

         call MPI_SEND(rstate,
     +                 8,
     +                 MPI_DOUBLE_PRECISION,
     +                 0,
     +                 1203,
     +                 MPI_COMM_WORLD,
     +                 ierr)

         call errchk(myid, ierr, 201203)

      end if

c --- wait for acknowledgement (message type 0204) from parent.  the
c     parent also uses this to signal the child that more input will
c     be sent.

c     if imsg(1) is positive, it is a replica number that represents the
c     next replica that this child should process.

c     if imsg(1) is negative, then this child needs to wait for the
c     other children to sync up, and so the child goes back to the top
c     of PART TWO.

c     if imsg(1) is zero, there is no more work to be done.

      call MPI_RECV(imsg,
     +              1,
     +              MPI_INTEGER,
     +              0,
     +              0204,
     +              MPI_COMM_WORLD,
     +              istat,
     +              ierr)

      call errchk(myid, ierr, 200204)

c --- loop back and wait for more input if instructed by parent.

      if (imsg(1).lt.0) goto 100

c --- terminate if the simulation is complete.

      if (imsg(1).eq.0) then
         if (idebug.eq.4) write (30+myid, *) 'child is done!'
         return
      end if

c --- if there is a new replica to process, then receive data from
c     the parent.

c --- we need to save the replica number that we are about to work on.

      idrep=imsg(1)

c --- first receive the loop number, in a message of type 0207.

      call MPI_RECV(loop,
     +              1,
     +              MPI_INTEGER,
     +              0,
     +              0207,
     +              MPI_COMM_WORLD,
     +              istat,
     +              ierr)

      call errchk(myid, ierr, 200207)

c --- next receive the old atomic coordinates and the means of the
c     neighboring replicas' coordinates, in a message of type 0205.

      call MPI_RECV(replic,
     +              NATOM6,
     +              MPI_R,
     +              0,
     +              0205,
     +              MPI_COMM_WORLD,
     +              istat,
     +              ierr)

      call errchk(myid, ierr, 200205)

c --- next receive the random number generator state vector, in
c     a message of type 0206.

      call MPI_RECV(rstate,
     +              8,
     +              MPI_DOUBLE_PRECISION,
     +              0,
     +              0206,
     +              MPI_COMM_WORLD,
     +              istat,
     +              ierr)

      call errchk(myid, ierr, 200206)

c --- generate provisional new atomic positions by adding gaussian
c     displacements.

c --- first choose the appropriate gaussian scaling factor.

      if (idrep.eq.1.or.idrep.eq.NREPS) then
         gsc=gscal2
      else
         gsc=gscale
      end if

c --- then add the gaussian displacements.

      do nn=1, NATOM3
         call gstep(rstate, gg, rscale)
         replic(NATOM3+nn)=replic(NATOM3+nn)+gg*gsc
      end do

c --- attempt to move each atom in turn.

      nacc=0
      nrej=0

      do nn=1, NATOMS

c ------ debugging output.

         if (nn.eq.1) then
            if (idebug.eq.4) then
               write (30+myid, *) 'moving atom 1'
               call flush(30+myid)
            end if
         end if

c ------ set up the coordinates of the atoms that are in this atom's
c        stencil.

         do i=1, npair(nn)

            ip=ipairs(i, nn)
            j=ivpair(2, ip)

            xx(i)=replic(3*j-2)+vpvec(1, ip)
            yy(i)=replic(3*j-1)+vpvec(2, ip)
            zz(i)=replic(3*j-0)+vpvec(3, ip)

         end do

c ------ debugging output.

         if (nn.eq.1) then
            if (idebug.eq.4) then
               write (30+myid, *) 'after do loop, xx(1) = ', xx(1)
               call flush(30+myid)
            end if
         end if

c ------ get the old and new coordinates of the atom that we're about
c        to try to move.

         xold=replic(3*nn-2)
         yold=replic(3*nn-1)
         zold=replic(3*nn-0)

         xnew=replic(3*nn-2+NATOM3)
         ynew=replic(3*nn-1+NATOM3)
         znew=replic(3*nn-0+NATOM3)

c ------ debugging output.

         if (nn.eq.1) then
            if (idebug.eq.4) then
               write (30+myid, *) 'xold, xnew = ', xold, xnew
               call flush(30+myid)
            end if
         end if

c ------ compute the old and new distances between this atom and all
c        of the atoms in the stencil.

c ------ the do loops are split up to promote vectorization, although
c        i'm not sure this is necessary.

         do i=1, npair(nn)
            r2old(i)=(xx(i)-xold)**2
         end do

         do i=1, npair(nn)
            r2old(i)=r2old(i)+(yy(i)-yold)**2
         end do

         do i=1, npair(nn)
            r2old(i)=r2old(i)+(zz(i)-zold)**2
         end do

         do i=1, npair(nn)
            r2new(i)=(xx(i)-xnew)**2
         end do

         do i=1, npair(nn)
            r2new(i)=r2new(i)+(yy(i)-ynew)**2
         end do

         do i=1, npair(nn)
            r2new(i)=r2new(i)+(zz(i)-znew)**2
         end do

c ------ compute the change in potential energy.

         do i=1, npair(nn)

c --------- use linear interpolation.

            ibin1=int((r2old(i)-r2min)*binvrs)+1
            ibin2=int((r2new(i)-r2min)*binvrs)+1

            if (ibin1.gt.0) then
               dr1=(r2old(i)-r2min)-bin*dble(ibin1-1)
               v1(i)=v(1, ibin1)+v(2, ibin1)*dr1
            else
               v1(i)=v(1, 1)
            end if

            if (ibin2.gt.0) then
               dr2=(r2new(i)-r2min)-bin*dble(ibin2-1)
               v2(i)=v(1, ibin2)+v(2, ibin2)*dr2
            else
               v2(i)=v(1, 1)
            end if

         end do

         dv=0.0

         do i=1, npair(nn)
            dv=dv+v1(i)-v2(i)
         end do

c ------ debugging output.

         if (nn.eq.1) then
            if (idebug.eq.4) then
               write (30+myid, *) 'dv = ', dv
               call flush(30+myid)
            end if
         end if

         dv=dv*tau

c ------ deal with trial function for first and last replicas.

         if (idrep.eq.1.or.idrep.eq.NREPS) then

            dpsi=0.0

            do i=1, npair(nn)
               dpsi=dpsi+ 
     +              (1.0d0/sqrt(r2old(i)))**5-
     -              (1.0d0/sqrt(r2new(i)))**5
            end do

            sold=xold**2+yold**2+zold**2
            snew=xnew**2+ynew**2+znew**2

            dpsi=0.5d0*bb**5*dpsi+aa*(sold-snew)

c --------- debugging output.

            if (nn.eq.1) then
               if (idebug.eq.4) then
                  write (30+myid, *) 'evaluating trial function'
                  write (30+myid, *) 'dpsi = ', dpsi
                  call flush(30+myid)
               end if
            end if

c --------- also remember to scale the change in potential energy by
c           one-half for the end replicas.

            dv=half*dv+dpsi

         end if

c ------ choose whether to accept the new position.

         call rstep(rstate, zran, rscale)

         if (dv.ge.0.0) then

c --------- accept this move.

            replic(3*nn-2)=xnew
            replic(3*nn-1)=ynew
            replic(3*nn-0)=znew

            nacc=nacc+1

         else if (zran.lt.exp(dv)) then

c --------- accept this move.

            replic(3*nn-2)=xnew
            replic(3*nn-1)=ynew
            replic(3*nn-0)=znew

            nacc=nacc+1

         else

c --------- reject this move.

            nrej=nrej+1

         end if

c --- end of loop over atoms.

      end do

c --- go back to send these results back to the parent.

      goto 200

      end

