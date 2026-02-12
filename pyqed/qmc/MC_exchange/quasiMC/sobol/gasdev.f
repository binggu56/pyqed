      FUNCTION gasdev(idum)
C-----------------------------------------------------
C    DOUBLE PRECISION VERSION OF NUMREC ROUTINE
C    FRANK GROSSMANN, 12.9.1994
C-----------------------------------------------------
      INTEGER*4 idum
      REAL*8 gasdev
CU    USES ran1
      INTEGER iset
      REAL*8 fac,gset,rsq,v1,v2,v(2) 
      SAVE iset,gset
      DATA iset/0/
      if (iset.eq.0) then
        
1       call i8_sobol(2, idum, v)
!        call i8_sobol(1, idum, v2) 

       v1=2.*v(1)-1.
        v2=2.*v(2)-1.

        rsq=v1**2+v2**2
        if(rsq.ge.1..or.rsq.eq.0.)goto 1
        fac=sqrt(-2.*log(rsq)/rsq)
        gset=v1*fac
        gasdev=v2*fac
        iset=1
      else
        gasdev=gset
        iset=0
      endif
      return
      END
