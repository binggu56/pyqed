!     polynomial fitting for {p,r}
!     two steps: 1st global linear fitting; 2nd local fitting with basis up to third order 
      subroutine fit(nt,ndim,ntraj,am,x,p,r,w,u,du,ap,fr)

      USE AQP, only : kmax 

      implicit real*8(a-h, o-z)

      integer*4,intent(in)    :: ntraj,ndim
      real*8, intent(in), dimension(ntraj)  :: w
      real*8, intent(in), dimension(ndim) :: am
      real*8, intent(in), dimension(ndim,ntraj) :: x,p,r

      real*8, intent(out), dimension(ndim,ntraj) :: du,fr,ap
      real*8, intent(out), dimension(ntraj)  :: u

      real*8, dimension(ndim,ndim) :: dp,dr
      real*8, dimension(ndim)      :: ddp,ddr


      real*8 :: f(ndim+1),f2(4),df2(4),ddf2(4),s1(ndim+1,ndim+1),s2(4,4),cp(ndim+1,ndim), & 
                mat(ndim+1,ndim+1),cr(ndim+1,ndim),cpr(4,2),ar(ndim,ntraj),               &
                cp2(4,ndim),cr2(4,ndim)

      integer info

!---------matrix S1 & c for linear fitting------------------

      nb = ndim+1
      cp = 0d0
      cr = 0d0
      s1 = 0d0
!      err = 0d0

!      call basis(Ndim,ntraj,x,f)
      
      do i = 1,ntraj
        call basis(ndim,ntraj,i,x,f)
        do j = 1,ndim+1
          do k = 1,ndim
            cp(j,k) = cp(j,k)+f(j)*p(k,i)*w(i) 
            cr(j,k) = cr(j,k)+f(j)*r(k,i)*w(i)
          enddo
        enddo
      enddo 

! -----matrix S=f*f---------------------------
      do i=1,ntraj
        call basis(ndim,ntraj,i,x,f)
        do k1=1,ndim+1
          do k2=k1,ndim+1
            s1(k1,k2) = s1(k1,k2)+f(k1)*f(k2)*w(i)
          enddo
        enddo
      enddo

      do k1=1,ndim+1
        do k2=k1+1,ndim+1
          s1(k2,k1) = s1(k1,k2)
        enddo
      enddo

!---------------solve matrix M*c = f*p-------------
      mat = s1
      call dposv('U',nb,ndim,mat,nb,cp,nb,INFO)
      if(info/=0) then
        write(*,*) 'linear fitting of p failed.'
        stop
      endif
!-------------same for r--------------------------
!      mat = s1
      call dposv('U',nb,ndim,s1,nb,cr,nb,INFO)
      if(info/=0) then
        write(*,*) 'linear fitting of r failed.'
        stop
      endif

!-------------output for {ap,ar}--------------------
      dp = 0d0
      ap = 0d0
      ar = 0d0
      dr = 0d0
      ddp = 0.d0
      ddr = 0.d0

      do i=1,ntraj
        call basis(ndim,ntraj,i,x,f)
        do j=1,ndim
          do k=1,nb
            ap(j,i) = ap(j,i)+cp(k,j)*f(k)
            ar(j,i) = ar(j,i)+cr(k,j)*f(k)
          enddo
        enddo
      enddo
!------first-order derivative of p,r, second order is 0 for linear fitting---------------
!      do i=1,ntraj
        do j=1,ndim
          do k=1,ndim
            dp(k,j) = cp(k,j)
            dr(k,j) = cr(k,j)
          enddo
        enddo
!      enddo

!---------------------------------------
!     2nd fitting for each DoF      
!---------------------------------------
      cp2 = 0d0
      cr2 = 0d0

      dimloop:do j=1,ndim

        s2 = 0d0
        cpr = 0d0

!        do i=1,ntraj
!          f2 = (/x(j,i),x(j,i)**2,x(j,i)**3,1d0/)
!          df2 = (/1d0,2d0*x(j,i),3d0*x(j,i)**2,0d0/)
!        enddo

        do i=1,ntraj
          f2 = (/x(j,i),x(j,i)**2,x(j,i)**3,1d0/)
          do k=1,4
            cpr(k,1) = cpr(k,1)+(p(j,i)-ap(j,i))*f2(k)*w(i)
            cpr(k,2) = cpr(k,2)+(r(j,i)-ar(j,i))*f2(k)*w(i)
          enddo
        enddo
          
        do i=1,ntraj
          f2 = (/x(j,i),x(j,i)**2,x(j,i)**3,1d0/)
          do m=1,4
            do n=1,4
              s2(m,n) = s2(m,n)+f2(m)*f2(n)*w(i) 
            enddo
          enddo
        enddo 
        
        call dposv('U',4,2,s2,4,cpr,4,info)
        if(info/=0) then
          write(*,*) 'cubic fitting of r failed.'
          stop
        endif


!-------store coefficients---------------------------

        do i=1,4
          cp2(i,j) = cpr(i,1)
          cr2(i,j) = cpr(i,2)
        enddo

      
!------second order derivative only contain diagonal elements--------------------
!-------4-dim array ddp(ndim,ndim,ndim,ntraj) contract to ddp(ndim,ndim,ntraj)----
        do i=1,ntraj
          f2 = (/x(j,i),x(j,i)**2,x(j,i)**3,1d0/)
          df2 = (/1d0,2d0*x(j,i),3d0*x(j,i)**2,0d0/)
          ddf2 = (/0d0,2d0,6d0*x(j,i),0d0/)

          do k=1,4
            ap(j,i) = ap(j,i)+cpr(k,1)*f2(k)
            ar(j,i) = ar(j,i)+cpr(k,2)*f2(k)
          enddo

        enddo

      enddo dimloop

! --- save fitting coefficients 
   
      if(nt == kmax) then 
        open(100, file = 'linearFit.data') 
        open(101, file = 'cubicFit.data') 
        
        do j=1,ndim+1
          write(100,1000) (cr(j,i), i=1,ndim)
        enddo
        do j=1,4
          write(101,1000) (cr2(j,i),i=1,ndim)
        enddo
      
        close(100)
        close(101) 

      endif 


      u = 0.d0
      du = 0.d0
      fr = 0.d0
      ddr = 0d0

      
        do j=1,ndim
          do k=1,ndim
            dr(k,j) = cr(k,j)
            dp(k,j) = cp(k,j)
          enddo
        enddo

      traj: do i=1,ntraj



        do j=1,ndim
            dr(j,j) = cr(j,j)+cr2(1,j)+2d0*cr2(2,j)*x(j,i)+3d0*cr2(3,j)*x(j,i)**2
            dp(j,j) = cp(j,j)+cp2(1,j)+2d0*cp2(2,j)*x(j,i)+3d0*cp2(3,j)*x(j,i)**2
            ddr(j) = 2d0*cr2(2,j)+6d0*cr2(3,j)*x(j,i)
            ddp(j) = 2d0*cp2(2,j)+6d0*cp2(3,j)*x(j,i)
        enddo

        do j=1,ndim
          u(i) = u(i)-(ar(j,i)**2+dr(j,j))/2d0/am(j)
        enddo
        
        do j=1,ndim
          do k=1,ndim
            du(j,i) = du(j,i)-(ar(k,i)*dr(j,k))/am(k)
            fr(j,i) = fr(j,i)-dp(j,k)*ar(k,i)/am(k)
          enddo
      
          du(j,i) = du(j,i)-ddr(j)/2d0/am(j)
          fr(j,i) = fr(j,i)-ddp(j)/2d0/am(j)
        enddo

      enddo traj

1000  format(10(e14.7,1x)) 
      return
      end subroutine
!----------------------------------------
!     linear basis 
!---------------------------------------
      subroutine basis(Ndim,Ntr,i,x,f)

      implicit real*8(a-h,o-z)

      integer*4,intent(in)  :: Ndim,Ntr,i

      real*8      :: x(Ndim,Ntr)
      real*8,intent(OUT) :: f(Ndim+1)

      f = 0d0
!---basis vector f = ((x(i),i=1,Ndim),1) for each trajectory-------------------
      do j=1,Ndim
        f(j)=x(j,i)
      enddo
      f(Ndim+1)=1d0
     
      return
      end subroutine
!--------------------------------------------------------------
