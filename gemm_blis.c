#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>
#include "blis.h"
#include "gemm_blis.h"
#include "gemm_blis_neon.h"

#define Aref(a1,a2)  A[ (a2-1)*(Alda)+(a1-1) ]
#define Bref(a1,a2)  B[ (a2-1)*(Blda)+(a1-1) ]
#define Cref(a1,a2)  C[ (a2-1)*(Clda)+(a1-1) ]
#define Ctref(a1,a2) Ctmp[ (a2-1)*(Clda)+(a1-1) ]
#define min(a,b) (((a)<(b))?(a):(b))

void gemm_blis( char transa, char transb, int m, int n, int k, 
                float alpha, float *A, int Alda, 
		float *B, int Blda, 
		float beta,  float *C, int Clda, 
		float *Ac, float *Bc, 
                int MC, int NC, int KC ){
  int    ic, jc, pc, mc, nc, kc,
         ir, jr, mr, nr;
  float  zero = 0.0, one = 1.0, betaI;
/* 
  Computes the GEMM C := beta * C + alpha * A * B
  following the BLIS approach
*/

  // Quick return if possible
  if ((m==0)||(n==0)||(((alpha==zero)||(k==0))&&(beta==one)))
    return;

  // Quick gemm if alpha==0.0
  if (alpha==zero) {
    if (beta==zero)
      #pragma omp parallel for private(jc)
      for ( ic=1; ic<=m; ic++ )
        for ( jc=1; jc<=n; jc++ )
          Cref(ic,jc) = 0.0;
    else
      #pragma omp parallel for private(jc)
      for ( ic=1; ic<=m; ic++ )
        for ( jc=1; jc<=n; jc++ )
          Cref(ic,jc) = beta*Cref(ic,jc);
    return;
  }

  for ( jc=1; jc<=n; jc+=NC ) {
    nc = min(n-jc+1, NC); 

    for ( pc=1; pc<=k; pc+=KC ) {
      kc = min(k-pc+1, KC); 
      if (transb=='N')
        pack_B( transb, kc, nc, &Bref(pc,jc), Blda, Bc);
      else
        pack_B( transb, kc, nc, &Bref(jc,pc), Blda, Bc);
      if (pc==1)
        betaI = beta;
      else
        betaI = one;

      for ( ic=1; ic<=m; ic+=MC ) {
        mc = min(m-ic+1, MC); 
        if (transa=='N')
          pack_A( transa, mc, kc, &Aref(ic,pc), Alda, Ac);
        else
          pack_A( transa, mc, kc, &Aref(pc,ic), Alda, Ac);

        for ( jr=1; jr<=nc; jr+=NR ) {
          nr = min(nc-jr+1, NR); 

          for ( ir=1; ir<=mc; ir+=MR ) {
            mr = min(mc-ir+1, MR);
	    //if( (mr == MR) && (nr == NR))
	    gemm_microkernel_neon_4x8( mr, nr, kc, alpha, &Ac[(ir-1)*kc], &Bc[(jr-1)*kc], betaI, &Cref((ic-1)+ir,(jc-1)+jr), Clda);
	    //else
	    //gemm_base( mr, nr, kc, alpha, &Ac[(ir-1)*kc], MR, &Bc[(jr-1)*kc], NR, betaI, &Cref((ic-1)+ir,(jc-1)+jr), Clda );

	  }
        }
      }
    }
  }
}//  end void

void pack_A( char transa, int mc, int kc, float *A, int Alda, float *Ac ){
/*
  BLIS pack for A-->Ac
*/
  int    i, j, ii, k, mr;

  if (transa=='N')
    #pragma omp parallel for private(j, ii, mr, k)
    for ( i=1; i<=mc; i+=MR ) { 
      k = (i-1)*kc;
      mr = min( mc-i+1, MR );
      for ( j=1; j<=kc; j++ ) {
        for ( ii=1; ii<=mr; ii++ ) {
          Ac[k] = Aref((i-1)+ii,j);
          k++;
        }
        k += (MR-mr);
      }
    }
  else
    #pragma omp parallel for private(j, ii, mr, k)
    for ( i=1; i<=mc; i+=MR ) { 
      k = (i-1)*kc;
      mr = min( mc-i+1, MR );
      for ( j=1; j<=kc; j++ ) {
        for ( ii=1; ii<=mr; ii++ ) {
           Ac[k] = Aref(j,(i-1)+ii);
          k++;
        }
        k += (MR-mr);
      }
    }
}

void pack_B( char transb, int kc, int nc, float *B, int Blda, float *Bc ){
/*
  BLIS pack for B-->Bc
*/
  int    i, j, jj, k, nr;

  k = 0;
  if (transb=='N')
    #pragma omp parallel for private(i, jj, nr, k)
    for ( j=1; j<=nc; j+=NR ) { 
      k = (j-1)*kc;
      nr = min( nc-j+1, NR );
      for ( i=1; i<=kc; i++ ) {
        for ( jj=1; jj<=nr; jj++ ) {
          Bc[k] = Bref(i,(j-1)+jj);
          k++;
        }
        k += (NR-nr);
      }
    }
  else
    #pragma omp parallel for private(i, jj, nr, k)
    for ( j=1; j<=nc; j+=NR ) { 
      k = (j-1)*kc;
      nr = min( nc-j+1, NR );
      for ( i=1; i<=kc; i++ ) {
        for ( jj=1; jj<=nr; jj++ ) {
          Bc[k] = Bref((j-1)+jj,i);
           k++;
        }
        k += (NR-nr);
      }
    }
}

void gemm_base( int m, int n, int k, float alpha,
	       	float *A, int Alda, float *B, int Blda,
	       	float beta, float *C, int Clda ){
/*
  Baseline micro-kernel, for cases where the dimension does not match MR x NR
*/
  int    i, j, p;
  float  zero = 0.0, one = 1.0, tmp;

  for ( j=1; j<=n; j++ )
    for ( i=1; i<=m; i++ ) {
      tmp = 0.0; 
      for ( p=1; p<=k; p++ ) 
        tmp += Aref(i,p) * Bref(j,p);

      if (beta==zero)
        Cref(i,j) = alpha*tmp;
      else
        Cref(i,j) = alpha*tmp + beta*Cref(i,j);
    }
}

