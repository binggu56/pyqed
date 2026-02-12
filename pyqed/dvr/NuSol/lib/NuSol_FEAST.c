/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! NuSol FEAST solver based on !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! FEAST Driver sparse example !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!! by Eric Polizzi- 2009-2012!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

#include <stdio.h> 
#include <stdlib.h> 
#include <sys/time.h>
#include "feast.h"
#include "feast_sparse.h"
int main(int argc, char *argv[]) {
  if ( argc != 7 ) /* argc should be 2 for correct execution */
  {
    /* We print argv[0] assuming it is the program name */
    printf( "usage: %s Emin Emax M0 matrixFile Eval.dat Evec.dat \n", argv[0] );
  }
  else 
  {
    /*!!!!!!!!!!!!!!!!! Feast declaration variable */
    int  feastparam[64]; 
    double epsout;
    int loop;
    char UPLO='F'; 

    /*!!!!!!!!!!!!!!!!! Matrix declaration variable */
    FILE *fp;
    int  N,nnz;
    double *sa,*sb;
    int *isa,*jsa;
    /*!!!!!!!!!!!!!!!!! Others */
    struct timeval t1, t2;
    int  n,i,k,err;
    int  M0,M,info;
    int  Nx,Ny,Nz;
    double xmin,xmax,ymin,ymax,zmin,zmax;
    double dx,dy,dz;
    double Emin,Emax,trace;
    double *X; //! eigenvectors
    double *E,*res; //! eigenvalue+residual

    FILE *file, *fileVec;
    file = fopen(argv[5],"w"); /* write eigenvalues */
    fileVec = fopen(argv[6],"w"); /* write eigenvectors */

    /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!! read input file in csr format!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

    // !!!!!!!!!! form CSR arrays isa,jsa,sa 
    fp = fopen (argv[4], "r");
    err=fscanf (fp, "%d%d%d%d%d%lf%lf%lf%lf%lf%lf%lf%lf%lf\n",&N,&nnz,&Nx,&Ny,&Nz,&xmin,&xmax,&ymin,&ymax,&zmin,&zmax,&dx,&dy,&dz);
    printf("%d %d %d %d %d %f %f %f %f %f %f %f %f %f\n",N,nnz,Nx,Ny,Nz,xmin,xmax,ymin,ymax,zmin,zmax,dx,dy,dz);
    sa=calloc(nnz,sizeof(double));
    sb=calloc(nnz,sizeof(double));
    isa=calloc(N+1,sizeof(int));
    jsa=calloc(nnz,sizeof(int));

    for (i=0;i<=N;i++){
      *(isa+i)=0;
    };
    *(isa)=1;
    for (k=0;k<=nnz-1;k++){
      err=fscanf(fp,"%d%d%lf%lf\n",&i,jsa+k,sa+k,sb+k);
      *(isa+i)=*(isa+i)+1;
    };
    fclose(fp);
    for (i=1;i<=N;i++){
      *(isa+i)=*(isa+i)+*(isa+i-1);
    };

    /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!! INFORMATION ABOUT MATRIX !!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
    printf("sparse matrix -system1- size %.d\n",N);
    printf("nnz %d \n",nnz);

    gettimeofday(&t1,NULL);
    /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!! FEAST in sparse format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

    /*!!! search interval [Emin,Emax] including M eigenpairs*/
    sscanf(argv[1],"%lf",&Emin);
    sscanf(argv[2],"%lf",&Emax);
    sscanf(argv[3],"%d",&M0);// !! M0>=M
    printf("argv read successfully\n");
    /*!!!!!!!!!!!!! ALLOCATE VARIABLE */
    E=calloc(M0,sizeof(double));  // eigenvalues
    res=calloc(M0,sizeof(double));// eigenvectors 
    X=calloc(N*M0,sizeof(double));// residual


    /*!!!!!!!!!!!!  FEAST */
    feastinit(feastparam);
    feastparam[0]=1;  /*1 Print runtime info*/
    feastparam[1]=32; /*2 contour points*/
    feastparam[2]=14;  /*3 double digit convergence*/
    feastparam[3]=30;  /*4 NITER*/
    // feastparam[3]=14; /*change from default value */
    dfeast_scsrgv(&UPLO,&N,sa,isa,jsa,sb,isa,jsa,feastparam,&epsout,&loop,&Emin,&Emax,&M0,E,X,&M,res,&info);
    gettimeofday(&t2,NULL);
    /*!!!!!!!!!! REPORT !!!!!!!!!*/
    printf("FEAST OUTPUT INFO %d\n",info);
    if (info==0) {
      printf("*************************************************\n");
      printf("************** REPORT ***************************\n");
      printf("*************************************************\n");
      printf("SIMULATION TIME %f\n",(t2.tv_sec-t1.tv_sec)*1.0+(t2.tv_usec-t1.tv_usec)*0.000001);
      printf("# Search interval [Emin,Emax] %.15e %.15e\n",Emin,Emax);
      printf("# mode found/subspace %d %d \n",M,M0);
      printf("# iterations %d \n",loop);
      trace=(double) 0.0;
      for (i=0;i<=M-1;i=i+1){
        trace=trace+*(E+i);
      }	  
      printf("TRACE %.15e\n", trace);
      printf("Relative error on the Trace %.15e\n",epsout );
      printf("Eigenvalues/Residuals\n");
      printf("%d %d %d %f %f %f %f %f %f\n",Nx,Ny,Nz,xmin,xmax,ymin,ymax,zmin,zmax); /*add info about the grid in each dimension, can be 1,2 or 3 dimensional*/
      fprintf(fileVec,"%d %d %d %f %f %f %f %f %f %f %f %f\n",Nx,Ny,Nz,xmin,xmax,ymin,ymax,zmin,zmax,dx,dy,dz); /*add info about the grid in each dimension, can be 1,2 or 3 dimensional*/
      for (i=0;i<=M-1;i=i+1){
        printf("   %d %.15e %.15e\n",i,*(E+i),*(res+i));
        fprintf(file,"%.15e %.15e\n",*(E+i),*(res+i)); /*writes EV*/
        for (n=0;n<=N-1;n=n+1){
          fprintf(fileVec,"%.15e ",*(X+N*i+n)); /*writes Vectors*/
        }
        fprintf(fileVec,"\n"); /*writes Vectors*/
      }
    }
    return 0;
  }
}
