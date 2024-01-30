c     QSATS version 1.0 (3 March 2011)

c     file name: even.f

c ----------------------------------------------------------------------

c     this subroutine distributes even-numbered replicas to the child
c     processes, waits for them to be processed, and then returns
c     control to the main parent subroutine.

c     errchk is a subroutine called after every MPI subroutine that
c     checks the MPI error code and reports any errors.

c ----------------------------------------------------------------------

      subroutine evnrep(loop, nsent, nrcvd, MPI_R)

      implicit double precision (a-h, o-z)

      include 'sizes.h'

      include 'qsats.h'

      include 'mpif.h'

      dimension istat(MPI_STATUS_SIZE)

      dimension imsg(3)

      dimension isent(NREPS), ikeep(NATOMS), replic(NATOM7)

      dimension rstate(8)

c --- loop over all even replicas.
     
      do nrep=2, NREPS, 2

         if (idebug.eq.4)
     +      write (9, *) 'finding child who can receive nrep = ', nrep

c ------ wait for data request from a child.

         call MPI_PROBE(MPI_ANY_SOURCE,
     +                  1201,
     +                  MPI_COMM_WORLD,
     +                  istat,
     +                  ierr)

         call errchk(0, ierr, 131201)

         nchild=istat(MPI_SOURCE)

         call MPI_RECV(imsg,
     +                 3,
     +                 MPI_INTEGER,
     +                 nchild,
     +                 1201,
     +                 MPI_COMM_WORLD,
     +                 istat,
     +                 ierr)

         call errchk(0, ierr, 161201)

         if (idebug.eq.4)
     +      write (9, *) 'sending nrep = ', nrep, ' to ', nchild

c ------ check whether the child is returning results.  if so, then
c        receive the results.

         if (imsg(1).gt.0) then

            idrep=imsg(1)

            if (idebug.eq.4)
     +         write (9, *) 'child ', nchild, ' returning replica ',
     +                      idrep

c --------- keep track of acceptances and rejections.

            ztacc=ztacc+imsg(2)
            ztrej=ztrej+imsg(3)

            call MPI_RECV(replic,
     +                    NATOM3,
     +                    MPI_R,
     +                    nchild,
     +                    1202,
     +                    MPI_COMM_WORLD,
     +                    istat,
     +                    ierr)

            call errchk(0, ierr, 131202)

            call MPI_RECV(rstate,
     +                    8,
     +                    MPI_DOUBLE_PRECISION,
     +                    nchild,
     +                    1203,
     +                    MPI_COMM_WORLD,
     +                    istat,
     +                    ierr)

            call errchk(0, ierr, 131203)

c --------- update the random number generator state vector for this
c           replica.

            do i=1, 8
               rstatv(i, idrep)=rstate(i)
            end do

c --------- update the atom positions in this replica.

            do i=1, NATOM3
               path(i, idrep)=replic(i)
            end do

c --------- update the number of received replicas.

            nrcvd=nrcvd+1

c --------- indicate that this replica has been processed and returned.

            isent(idrep)=-nchild

         end if

c ------ send a new replica to child.  first tell the child which replica
c        it is going to receive.

         imsg(1)=nrep

         call MPI_SEND(imsg,
     +                 1,
     +                 MPI_INTEGER,
     +                 nchild,
     +                 0204,
     +                 MPI_COMM_WORLD,
     +                 ierr)

         call errchk(0, ierr, 130204)

c ------ send the replica.

         if (idebug.eq.4)
     +      write (9, *) 'calling rpsend for child ', nchild

         call rpsend(loop, nrep, nchild, MPI_R)

         if (idebug.eq.4)
     +      write (9, *) 'replica ', nrep, ' sent to child ', nchild

c ------ update how many replicas have been sent.

         nsent=nsent+1

c ------ leave the trail of crumbs!

         isent(nrep)=nchild

c ------ update how much work has been sent to this child.

         iwork(nchild)=iwork(nchild)+1

      end do

c --- at this point we don't have any more odd replicas to send to the
c     children, but we need to retrieve any processed replicas that the
c     children are still holding to send back to the parent.  this
c     flushes out all of those replicas.
            
      do i=2, NREPS, 2

         if (isent(i).gt.0) then

            nchild=isent(i)

            call MPI_RECV(imsg,
     +                    3,
     +                    MPI_INTEGER,
     +                    nchild,
     +                    1201,
     +                    MPI_COMM_WORLD,
     +                    istat,
     +                    ierr)

            call errchk(0, ierr, 141201)

c --------- check whether the child is returning results.  if so, get
c           the results and update the atomic positions.

            if (imsg(1).gt.0) then

               idrep=imsg(1)

               if (idebug.eq.4)
     +            write (9, *) 'child ', nchild,
     +                         ' returning replica ', idrep

c ------------ keep track of acceptances and rejections.

               ztacc=ztacc+imsg(2)
               ztrej=ztrej+imsg(3)

               call MPI_RECV(replic,
     +                       NATOM3,
     +                       MPI_R,
     +                       nchild,
     +                       1202,
     +                       MPI_COMM_WORLD,
     +                       istat,
     +                       ierr)

               call errchk(0, ierr, 141202)

               call MPI_RECV(rstate,
     +                       8,
     +                       MPI_DOUBLE_PRECISION,
     +                       nchild,
     +                       1203,
     +                       MPI_COMM_WORLD,
     +                       istat,
     +                       ierr)

               call errchk(0, ierr, 141203)

c ------------ update the random number generator state vector for this
c              replica.

               do k=1, 8
                  rstatv(k, idrep)=rstate(k)
               end do

c ------------ update the atom positions in this replica.

               do n=1, NATOM3
                  path(n, idrep)=replic(n)
               end do

c ------------ update the number of received replicas.

               nrcvd=nrcvd+1

c ------------ indicate that this replica has been processed and returned.

               isent(idrep)=-nchild

            end if

c --------- now tell the child to wait until all of the children are done
c           and more work is available.

            imsg(1)=-1

            call MPI_SEND(imsg,
     +                    1,
     +                    MPI_INTEGER,
     +                    nchild,
     +                    0204,
     +                    MPI_COMM_WORLD,
     +                    ierr)

            call errchk(0, ierr, 141204)

         end if

      end do

      return
      end
