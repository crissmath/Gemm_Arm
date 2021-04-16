#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "blis.h"
#include "gemm_blis.h"
#include "gemm_blis_neon.h"

#define Aref(a1,a2)  A[ (a2-1)*(Alda)+(a1-1) ]
#define Bref(a1,a2)  B[ (a2-1)*(Blda)+(a1-1) ]
#define Cref(a1,a2)  C[ (a2-1)*(Clda)+(a1-1) ]
#define Cgref(a1,a2) Cg[ (a2-1)*(Clda)+(a1-1) ]
#define dabs(a)      ( (a) > 0.0 ? (a) : -(a) )
#define min(a,b)     ( (a) > (b) ? (b) : (a) )

extern int    print_matrix( char *, int, int, float *, int );
extern int    generate_matrix( int, int, float *, int );
extern double dclock();

void gemm( char, char, int, int, int, float, float *, int, float *, int, float, float *, int );

int main(int argc, char *argv[])
{
  char  transa, transb, test;
  float *A, *B, *C, *Cg, *Ac, *Bc, flops, GFLOPS, tmp, error, nrm, alpha, beta;
  double t1, t2, time, tmin;
  int    i, j, nreps, info, 
         mmin, mmax, mstep,
         nmin, nmax, nstep,
         kmin, kmax, kstep,
         mcmin, mcmax, mcstep,
         ncmin, ncmax, ncstep,
         kcmin, kcmax, kcstep,
         m, n, k, mc, nc, kc, visual, Alda, Blda, Clda;

  printf("# =======================================================================================\n");
  printf("# Driver for the evaluation of GEMM\n");
  printf("# =======================================================================================\n");
  printf("# Program starts...\n");
  printf("#        transa transb     m     n     k     mc     nc     kc     Time   GFLOPS     ERROR\n");
  printf("# ---------------------------------------------------------------------------------------\n");
  
  // printf("-->Read data\n"); fflush(stdout);
  transa  = argv[1][0];
  transb  = argv[2][0];

  alpha   = atof(argv[3]);
  beta    = atof(argv[4]);

  mmin  = atoi(argv[5]);
  mmax  = atoi(argv[6]);
  mstep = atoi(argv[7]);

  nmin  = atoi(argv[8]);
  nmax  = atoi(argv[9]);
  nstep = atoi(argv[10]);

  kmin  = atoi(argv[11]);
  kmax  = atoi(argv[12]);
  kstep = atoi(argv[13]);

  mcmin   = atoi(argv[14]);
  mcmax   = atoi(argv[15]);
  mcstep  = atoi(argv[16]);

  ncmin  = atoi(argv[17]);
  ncmax  = atoi(argv[18]);
  ncstep = atoi(argv[19]);

  kcmin  = atoi(argv[20]);
  kcmax  = atoi(argv[21]);
  kcstep = atoi(argv[22]);

  visual = atoi(argv[23]);
  tmin   = atof(argv[24]);
  test   = argv[25][0];


  // Allocate space for data 
  // printf("-->Allocate data\n"); fflush(stdout);
  A = (float *) malloc(mmax*kmax*sizeof(float));   
  B = (float *) malloc(kmax*nmax*sizeof(float));   
  C = (float *) malloc(mmax*nmax*sizeof(float));   
  Ac = (float *) aligned_alloc(32,mcmax*kcmax*sizeof(float));   
  Bc = (float *) aligned_alloc(32,kcmax*ncmax*sizeof(float));   

  if (test=='T')
    Cg = (float *) malloc(mmax*nmax*sizeof(float));   

  for (m = mmin; m <= mmax; m+=mstep ){
  for (n = nmin; n <= nmax; n+=nstep ){
  for (k = kmin; k <= kmax; k+=kstep ){
  for (mc = mcmin; mc <= mcmax; mc+=mcstep ){
  for (nc = ncmin; nc <= ncmax; nc+=ncstep ){
  for (kc = kcmin; kc <= kcmax; kc+=kcstep ){

    // Generate random data
    // printf("-->Generate data\n"); fflush(stdout);
    if (transa=='N') {
      Alda = m;
      generate_matrix( m, k, A, Alda );
    }
    else {
      Alda = k;
      generate_matrix( k, m, A, Alda );
    }

    if (transb=='N') {
      Blda = k;
      generate_matrix( k, n, B, Blda );
    }
    else {
      Blda = n;
      generate_matrix( n, k, B, Blda );
    }

    Clda = m;
    generate_matrix( m, n, C, Clda );
    if (test=='T') {
      for ( i=1; i<=m; i++ ) 
         for ( j=1; j<=n; j++ )
           Cgref(i,j)=Cref(i,j);
    }

    // Print data
    if ( visual == 1 ){
      if (transb=='N') 
        print_matrix( "Ai", n, k, A, Alda );
      else
        print_matrix( "Ai", k, n, A, Alda );
      if (transb=='N') 
        print_matrix( "Bi", k, n, B, Blda );
      else
        print_matrix( "Bi", n, k, B, Blda );
      print_matrix( "Ci", m, n, C, Clda );
    }

    // printf("-->Solve problem\n"); fflush(stdout);

    time  = 0.0; 
    t1    = dclock();
    nreps = 0;
    while ( time <= tmin ) {
      nreps++;

      // GEMM
      gemm_blis( transa, transb, m, n, k, alpha, A, Alda, B, Blda, beta, C, Clda, 
                 Ac, Bc, mc, nc, kc );
       
      t2   = dclock();
      time = ( t2 > t1 ? t2 - t1 : 0.0 );

      //
    }
    time = time/nreps;

    // Test result
    if (test=='T') {
      gemm( transa, transb, m, n, k, alpha, A, Alda, B, Blda, beta, Cg, Clda );
      error = 0.0;
      nrm   = 0.0;
      for ( i=1; i<=m; i++ ) 
         for ( j=1; j<=n; j++ ) {
           tmp = Cgref(i,j)*Cgref(i,j);
	   nrm += tmp*tmp;
           error += dabs(Cgref(i,j)-Cref(i,j)); 
	   // printf("D(%2d,%2d) = %8.2e\n", i, j, dabs(Cgref(i,j)-Cref(i,j)));
         }
      error = sqrt(error) / sqrt(nrm);
    }
    else
      error = -1.0;

    // Print results
    if (visual == 1) {
      print_matrix( "Cf", m, n, C, Clda );
      print_matrix( "Crf", m, n, Cg, Clda );
    }

    //printf("-->Results\n");
    //printf("   Time         = %12.6e seg.\n", time  );
    flops   = 2.0 * m * n * k;
    GFLOPS  = flops / (1.0e+9 * time );
    //printf("   GFLOPs       = %12.6e     \n", GFLOPS  );
    printf("         %6c %6c %5d %5d %5d %6d %6d %6d %8.2e %8.2e %9.2e\n", transa, transb, m, n, k, mc, nc, kc, time, GFLOPS, error );
  }
  }
  }
  }
  }
  }

  /* Free data */
  free(A);
  free(B);
  free(C);
  free(Ac);
  free(Bc);

  if (test=='T')
    free(Cg);
  printf("# End of program...\n");
  printf("# ==================================================================\n");

  return 0;
}

void gemm( char transa, char transb, int m, int n, int k, 
           float alpha, float *A, int Alda, 
	                float *B, int Blda, 
           float beta,  float *C, int Clda ){
   int    i, j, p;
   float  zero = 0.0, one = 1.0, tmp;

   // Quick return if possible
  if ((m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)))
    return;

  // Quick gemm if alpha==0.0
  if (alpha==zero) {
    if (beta==zero)
      for ( i=1; i<=m; i++ )
        for ( j=1; j<=n; j++ )
          Cref(i,j) = 0.0;
    else
      for ( i=1; i<=m; i++ )
        for ( j=1; j<=n; j++ )
          Cref(i,j) = beta*Cref(i,j);
    return;
  }

  if ((transa=='N')&&(transb=='N'))
    for ( i=1; i<=m; i++ )
      for ( j=1; j<=n; j++ ) {
        tmp = 0.0; 
        for ( p=1; p<=k; p++ )
          tmp += Aref(i,p) * Bref(p,j);

	if (beta==zero)
          Cref(i,j) = alpha*tmp;
	else
          Cref(i,j) = alpha*tmp + beta*Cref(i,j);
      }
   else if ((transa=='N')&&(transb=='T'))
    for ( i=1; i<=m; i++ )
      for ( j=1; j<=n; j++ ) {
        tmp = 0.0; 
        for ( p=1; p<=k; p++ )
          tmp += Aref(i,p) * Bref(j,p);

	if (beta==zero)
          Cref(i,j) = alpha*tmp;
	else
          Cref(i,j) = alpha*tmp + beta*Cref(i,j);
      }
   else if ((transa=='T')&&(transb=='N'))
    for ( i=1; i<=m; i++ )
      for ( j=1; j<=n; j++ ) {
        tmp = 0.0; 
        for ( p=1; p<=k; p++ )
          tmp += Aref(p,i) * Bref(p,j);

	if (beta==zero)
          Cref(i,j) = alpha*tmp;
	else
          Cref(i,j) = alpha*tmp + beta*Cref(i,j);
      }
   else if ((transa=='T')&&(transb=='T'))
    for ( i=1; i<=m; i++ )
      for ( j=1; j<=n; j++ ) {
        tmp = 0.0; 
        for ( p=1; p<=k; p++ )
          tmp += Aref(p,i) * Bref(j,p);

	if (beta==zero)
          Cref(i,j) = alpha*tmp;
	else
          Cref(i,j) = alpha*tmp + beta*Cref(i,j);
      }
   else {
     printf("Error: Invalid options for transa, transb: %c %c\n", transa, transb);
     exit(-1);
   }
}


