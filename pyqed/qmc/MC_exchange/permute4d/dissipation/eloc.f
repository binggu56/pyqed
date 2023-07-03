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

c --- this common block is used to enable interpolation in the potential
c     energy lookup table in the subroutine local below.

      common /bincom/ bin, binvrs, r2min

      dimension q(NATOM3), vtavg(NREPS), vtavg2(NREPS),
     +          etavg(NREPS), etavg2(NREPS)

      parameter (half=0.5d0)
      parameter (one=1.0d0)

c --- initialization.

      call tstamp

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

      call input

      call vinit(r2min, bin)

      binvrs=one/bin

c --- read crystal lattice points.

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
      do k=1, NREPS
         vtavg(k)=0.0d0
         etavg(k)=0.0d0
         vtavg2(k)=0.0d0
         etavg2(k)=0.0d0
      end do

      open (10, file=spfile, form='unformatted')

c --- this loops reads the snapshots saved by QSATS.

300   loop=loop+1

      do k=1, NREPS, 11

         read (10, end=600) (path(i, k), i=1, NATOM3)

c ------ compute the local energy and the potential energy.

         do i=1, NATOM3
            q(i)=path(i, k)
         end do

         call local(q, tloc, vloc)

c ------ convert to kelvin per atom.

         tloc=tloc/(3.1668513d-6*dble(NATOMS))
         vloc=vloc/(3.1668513d-6*dble(NATOMS))

c ------ accumulate the results.

         vtavg(k)=vtavg(k)+vloc
         vtavg2(k)=vtavg2(k)+(vloc)**2
         etavg(k)=etavg(k)+tloc+vloc
         etavg2(k)=etavg2(k)+(tloc+vloc)**2

350      continue

      end do

      goto 300

c --- account for overshooting.

600   loop=loop-1

      write (6, 6600) loop
6600  format ('number of snapshots = ', i6/)

c --- compute the averages and standard deviations.

      do k=1, NREPS, 11

      vtavg(k)=vtavg(k)/dble(loop)
      vtavg2(k)=vtavg2(k)/dble(loop)
      etavg(k)=etavg(k)/dble(loop)
      etavg2(k)=etavg2(k)/dble(loop)

      vsd=sqrt(vtavg2(k)-vtavg(k)**2)
      esd=sqrt(etavg2(k)-etavg(k)**2)

      write (6, 6610) k, 'VAVG = ', vtavg(k)
6610  format ('replica ', i3, 1x, a7, f10.5, ' Kelvin')

      write (6, 6610) k, 'V SD = ', vsd

      write (6, 6610) k, 'EAVG = ', etavg(k)

      write (6, 6610) k, 'E SD = ', esd

      end do

      stop
      end

c ----------------------------------------------------------------------

c     this subroutine computes the local energy and potential energy
c     of a configuration.

c ----------------------------------------------------------------------

      subroutine local(q, tloc, vloc)

      implicit double precision (a-h, o-z)

      include 'sizes.h'

      include 'qsats.h'

      common /bincom/ bin, binvrs, r2min

c --- alpha is the exponential parameter in psi:

c     psi = N * exp(-alpha*(r-r0)**2) * Jastrow

c --- bb is the exponential parameter in Jastrow:

c     ln Jastrow(ij) = -0.5 * (bb/rij)**5

      dimension q(NATOM3), dlng(NATOM3), d2lng(NATOM3)

      do i=1, NATOM3
         dlng(i)=0.0d0
         d2lng(i)=0.0d0
      end do

      do i=1, NATOMS

         xx=q(3*i-2)
         yy=q(3*i-1)
         zz=q(3*i)

         dlng(3*i-2)=dlng(3*i-2)-2.0d0*aa*xx
         dlng(3*i-1)=dlng(3*i-1)-2.0d0*aa*yy
         dlng(3*i)  =dlng(3*i)  -2.0d0*aa*zz

         d2lng(3*i-2)=d2lng(3*i-2)-2.0d0*aa
         d2lng(3*i-1)=d2lng(3*i-1)-2.0d0*aa
         d2lng(3*i)  =d2lng(3*i)  -2.0d0*aa

      end do

c --- loop over all interacting pairs.

      vloc=0.0d0
      tloc=0.0d0

      do n=1, nvpair

         i=ivpair(1, n)
         j=ivpair(2, n)

         dx=-((q(3*j-2))+vpvec(1, n)+(-q(3*i-2)))
         dy=-((q(3*j-1))+vpvec(2, n)+(-q(3*i-1)))
         dz=-((q(3*j))  +vpvec(3, n)+(-q(3*i))  )

         r2=dx*dx+dy*dy+dz*dz

         ibin=int((r2-r2min)*binvrs)+1

         if (ibin.gt.0) then
            dr=(r2-r2min)-bin*dble(ibin-1)
            vloc=vloc+v(1, ibin)+v(2, ibin)*dr
         else
            vloc=vloc+v(1, 1)
         end if

         br2=bb*bb/r2

         br5=br2*br2*sqrt(br2)

         br52=br5/r2

         dlng(3*i-2)=dlng(3*i-2)+2.5d0*br52*dx
         dlng(3*i-1)=dlng(3*i-1)+2.5d0*br52*dy
         dlng(3*i)  =dlng(3*i)  +2.5d0*br52*dz

         d2lng(3*i-2)=d2lng(3*i-2)+2.5d0*br52*
     *                             (1.0d0-7.0d0*dx**2/r2)
         d2lng(3*i-1)=d2lng(3*i-1)+2.5d0*br52*
     *                             (1.0d0-7.0d0*dy**2/r2)
         d2lng(3*i)  =d2lng(3*i)  +2.5d0*br52*
     *                             (1.0d0-7.0d0*dz**2/r2)

      end do

c --- now sum up the kinetic energy components.

      do i=1, NATOM3
         tloc=tloc+d2lng(i)+dlng(i)**2
      end do

c --- account for mass factor and for double-counting of pairs.

      tloc=-0.5d0*tloc/amass
      vloc=0.5d0*vloc

      return
      end

c ----------------------------------------------------------------------

c     quit is a subroutine used to terminate execution if there is
c     an error.

c     it is needed here because the subroutine that reads the parameters
c     (subroutine input) may call it.

c ----------------------------------------------------------------------

      subroutine quit

      write (6, *) 'termination via subroutine quit'

      stop

      return
      end
