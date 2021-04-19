#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>
#include "blis.h"
#include "gemm_blis.h"
#include "gemm_blis_neon.h"

#define Aref(a1,a2)  A[ (a2-1)*(Alda)+(a1-1) ]
#define Bref(a1,a2)  B[ (a2-1)*(Blda)+(a1-1) ]
#define Cref(a1,a2)  C[ (a2-1)*(Clda)+(a1-1) ]
#define Ctref(a1,a2) Ctmp[ (a2-1)*(MR)+(a1-1) ]
#define min(a,b) (((a)<(b))?(a):(b))


/*---------------NEON MICROKERNEL 4X4---------------------------------- */	
void gemm_microkernel_neon_4x4( int m, int n, int k, float alpha,
	       			float *A, float *B,
			       	float beta, float *C, int Clda ){
/*
  BLIS GEMM microkernel, computes the product Cr := beta * Cr + alpha * Ar * Br
  Specific: only for MRxNR = 4x4
  Cr --> m x n
  Ar --> m x k
  Br --> k x n
 
*/

  int           i, j, kc, baseA, baseB, Ctlda = MR;
  float32x4_t 	C1, C2, C3, C4,
     		A1, A2, A3, A4, B1;
  float         zero = 0.0, one = 1.0, Ctmp[MR*NR];

  if (k == 0)
    return;
 
  // Set to zeros
  C1 = vmovq_n_f32(0);
  C2 = vmovq_n_f32(0);
  C3 = vmovq_n_f32(0);
  C4 = vmovq_n_f32(0);

  // Iterate from 1 to kc-1 (last iteration outside loop)
  for ( kc=1; kc<=k; kc++ ) {

    // Prefect colum/row of A/B for next iteration
    // This is done to overlap loading Ar/Br with the computations in the present iteration
    baseA = (kc-1)*MR;
    baseB = (kc-1)*NR;
    A1 = vld1q_f32(&A[baseA]);
    B1 = vld1q_f32(&B[baseB]);
     
    // Accumulate in each column of Cr, the product of one column of Ar (in A1) and one element of Br (in B1)
    C1 = vfmaq_laneq_f32(C1, A1, B1, 0);
    C2 = vfmaq_laneq_f32(C2, A1, B1, 1);
    C3 = vfmaq_laneq_f32(C3, A1, B1, 2);
    C4 = vfmaq_laneq_f32(C4, A1, B1, 3);
  }

  if (alpha==-one) {
    // If alpha==-1, change sign of result
    C1 = -C1; C2 = -C2; C3 = -C3; C4 = -C4;
  }
  else if (alpha!=one) {
    // If alpha!=one, multiply the result in the vector registers by that value
    C1 = alpha*C1; C2 = alpha*C2; C3 = alpha*C3; C4 = alpha*C4;
  }

  if ((m<MR)||(n<NR)) {
    // If the result is smaller than m x n, we cannot use vector instructions to store it in memory. Do it "manually", by first storing it into Ctmp
    // and then from there to the appropriate positions in memory
    vst1q_f32(&Ctref(1,1), C1);
    vst1q_f32(&Ctref(1,2), C2);
    vst1q_f32(&Ctref(1,3), C3);
    vst1q_f32(&Ctref(1,4), C4);
    if (beta!=zero) {
      // If beta!=0.0, C must be multiplied by that value
      for (j=1; j<=n; j++)
        for (i=1; i<=m; i++)
          Cref(i,j) = beta*Cref(i,j) + Ctref(i,j);
    }
    else {
      // If beta==0.0, no need to retrieve C from memory. Overwrite the value in memory with the result.
      for (j=1; j<=n; j++)
        for (i=1; i<=m; i++)
          Cref(i,j) = Ctref(i,j);
    }
  }
  else if ((m==MR)||(n==NR)) {
    // If the result is exactly MR x NR, utilize vector instructions to store it in memory
    if (beta!=zero) {
      // If beta!=0.0, C must be multiplied by that value. Use A1,A2,A3,A4 as temporary values to retrieve the values of C from memory
      A1 = vld1q_f32(&Cref(1,1));
      A2 = vld1q_f32(&Cref(1,2));
      A3 = vld1q_f32(&Cref(1,3));
      A4 = vld1q_f32(&Cref(1,4));
      C1 = beta*A1 + C1;
      C2 = beta*A2 + C2;
      C3 = beta*A3 + C3;
      C4 = beta*A4 + C4;
    }
    vst1q_f32(&Cref(1,1), C1);
    vst1q_f32(&Cref(1,2), C2);
    vst1q_f32(&Cref(1,3), C3);
    vst1q_f32(&Cref(1,4), C4);
  }
  else {
    printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", m, n);
    exit(-1);
  }
}

/*------------------------------------------------------------------------------*/
/*-------------------NEON MICROKERNEL 8X8---------------------------------------*/
void gemm_microkernel_neon_8x8( int m, int n, int k, float alpha,
	       			float *A, float *B,
			       	float beta, float *C, int Clda ){
/*
  BLIS GEMM microkernel, computes the product Cr := beta * Cr + alpha * Ar * Br
  Specific: only for MRxNR = 4x8
  Cr --> m x n
  Ar --> m x k
  Br --> k x n
*/
  int         i, j, kc, baseA, baseB, Ctlda = MR;

  // columns of C 
  float32x4_t 	C00, C01, C02, C03, C04, C05, C06, C07,
	       	C10, C11, C12, C13, C14, C15, C16, C17;
  // Columns of A
  float32x4_t   A0, A1, A2 , A3 , A4 , A5 , A6 , A7;

  // Bij		
  float32x4_t   B0, B1;

 float       zero = 0.0, one = 1.0, Ctmp[MR*NR];

  if (k == 0)
    return;

  // Set to zeros columns of C
  C00  = vmovq_n_f32(0); C10 = vmovq_n_f32(0);
  C01  = vmovq_n_f32(0); C11 = vmovq_n_f32(0);
  C02  = vmovq_n_f32(0); C12 = vmovq_n_f32(0);
  C03  = vmovq_n_f32(0); C13 = vmovq_n_f32(0);
  C04  = vmovq_n_f32(0); C14 = vmovq_n_f32(0);
  C05  = vmovq_n_f32(0); C15 = vmovq_n_f32(0);
  C06  = vmovq_n_f32(0); C16 = vmovq_n_f32(0);
  C07  = vmovq_n_f32(0); C17 = vmovq_n_f32(0); 

  // Iterate from 1 to kc-1 (last iteration outside loop)
  for ( kc=1; kc<=k; kc++ ) {

    // Prefect colum/row of A/B for next iteration
    // This is done to overlap loading Ar/Br with the computations in the present iteration
    baseA = (kc-1)*MR; 
    baseB = (kc-1)*NR;  
        
    A0 = vld1q_f32(&A[baseA]);
    A1 = vld1q_f32(&A[baseA+4]); 
	
    B0 = vld1q_f32(&B[baseB]);	    
    B1 = vld1q_f32(&B[baseB+4]);
     
    // Accumulate in each column of Cr, the product of one column of Ar (in A1) and one element of Br (in B1)
    // Cij = Cij + Ai * Bj(lane)
    //
    C00   = vfmaq_laneq_f32(C00,  A0, B0, 0); 
    C01   = vfmaq_laneq_f32(C01,  A0, B0, 1);
    C02   = vfmaq_laneq_f32(C02,  A0, B0, 2);
    C03   = vfmaq_laneq_f32(C03,  A0, B0, 3);

    C04   = vfmaq_laneq_f32(C04,  A0, B1, 0);
    C05   = vfmaq_laneq_f32(C05,  A0, B1, 1);
    C06   = vfmaq_laneq_f32(C06,  A0, B1, 2);
    C07   = vfmaq_laneq_f32(C07,  A0, B1, 3);

    C10   = vfmaq_laneq_f32(C10,  A1, B0, 0);
    C11   = vfmaq_laneq_f32(C11,  A1, B0, 1);
    C12   = vfmaq_laneq_f32(C12,  A1, B0, 2);
    C13   = vfmaq_laneq_f32(C13,  A1, B0, 3);

    C14   = vfmaq_laneq_f32(C14,  A1, B1, 0);
    C15   = vfmaq_laneq_f32(C15,  A1, B1, 1);
    C16   = vfmaq_laneq_f32(C16,  A1, B1, 2);
    C17   = vfmaq_laneq_f32(C17,  A1, B1, 3);
  }

  if (alpha==-one) {
    // If alpha==-1, change sign of result
    C00 = -C00;
    C01 = -C01;
    C02 = -C02;
    C03 = -C03;
    C04 = -C04;
    C05 = -C05;
    C06 = -C06;
    C07 = -C07;

    C10 = -C10;
    C11 = -C11;
    C12 = -C12;
    C13 = -C13;
    C14 = -C14;
    C15 = -C15;
    C16 = -C16;
    C17 = -C17;

  }
  else if (alpha!=one) {
    // If alpha!=one, multiply the result in the vector registers by that value
    C00 = alpha*C00;
    C01 = alpha*C01;
    C02 = alpha*C02;
    C03 = alpha*C03;
    C04 = alpha*C04;
    C05 = alpha*C05;
    C06 = alpha*C06;
    C07 = alpha*C07;

    C10 = alpha*C10;
    C11 = alpha*C11;
    C12 = alpha*C12;
    C13 = alpha*C13;
    C14 = alpha*C14;
    C15 = alpha*C15;
    C16 = alpha*C16;
    C17 = alpha*C17;
  }

  if ((m<MR)||(n<NR)) {
    // If the result is smaller than m x n, we cannot use vector instructions to store it in memory. Do it "manually", by first storing it into Ctmp
    // and then from there to the appropriate positions in memory
    
    vst1q_f32(&Ctref(1,1), C00);
    vst1q_f32(&Ctref(1,2), C01);
    vst1q_f32(&Ctref(1,3), C02);
    vst1q_f32(&Ctref(1,4), C03);
    vst1q_f32(&Ctref(1,5), C04);
    vst1q_f32(&Ctref(1,6), C05);
    vst1q_f32(&Ctref(1,7), C06);
    vst1q_f32(&Ctref(1,8), C07);

    vst1q_f32(&Ctref(5,1), C10);
    vst1q_f32(&Ctref(5,2), C11);
    vst1q_f32(&Ctref(5,3), C12);
    vst1q_f32(&Ctref(5,4), C13);
    vst1q_f32(&Ctref(5,5), C14);
    vst1q_f32(&Ctref(5,6), C15);
    vst1q_f32(&Ctref(5,7), C16);
    vst1q_f32(&Ctref(5,8), C17);

    if (beta!=zero) {
      // If beta!=0.0, C must be multiplied by that value
      for (j=1; j<=n; j++)
        for (i=1; i<=m; i++)
          Cref(i,j) = beta*Cref(i,j) + Ctref(i,j);
    }
    else {
      // If beta==0.0, no need to retrieve C from memory. Overwrite the value in memory with the result.
      for (j=1; j<=n; j++)
        for (i=1; i<=m; i++)
          Cref(i,j) = Ctref(i,j);
    }
  }
  else if ((m==MR)||(n==NR)) {
    // If the result is exactly MR x NR, utilize vector instructions to store it in memory


	  
    if (beta!=zero) {

// If beta!=0.0, C must be multiplied by that value. Use A1,A2,A3,A4 as temporary values to retrieve the values of C from memory


      A0  = vld1q_f32(&Cref(1,1));
      A1  = vld1q_f32(&Cref(1,2));
      A2  = vld1q_f32(&Cref(1,3));
      A3  = vld1q_f32(&Cref(1,4));
      A4  = vld1q_f32(&Cref(1,5));
      A5  = vld1q_f32(&Cref(1,6));
      A6  = vld1q_f32(&Cref(1,7));
      A7  = vld1q_f32(&Cref(1,8));

      C00   = beta*A0  + C00;
      C01   = beta*A1  + C01;
      C02   = beta*A2  + C02;
      C03   = beta*A3  + C03;
      C04   = beta*A4  + C04;
      C05   = beta*A5  + C05;
      C06   = beta*A6  + C06;
      C07   = beta*A7  + C07;
    }
    // Reuse the vectorial register
    if (beta!= zero){

      A0  = vld1q_f32(&Cref(5,1));
      A1  = vld1q_f32(&Cref(5,2));
      A2  = vld1q_f32(&Cref(5,3));
      A3  = vld1q_f32(&Cref(5,4));
      A4  = vld1q_f32(&Cref(5,5));
      A5  = vld1q_f32(&Cref(5,6));
      A6  = vld1q_f32(&Cref(5,7));
      A7  = vld1q_f32(&Cref(5,8));

      C10   = beta*A0 + C10;
      C11   = beta*A1 + C11;
      C12   = beta*A2 + C12;
      C13   = beta*A3 + C13;
      C14   = beta*A4 + C14;
      C15   = beta*A5 + C15;
      C16   = beta*A6 + C16;
      C17   = beta*A7 + C17;
    }
    // store the values in register 
    vst1q_f32(&Cref(1,1),  C00);
    vst1q_f32(&Cref(1,2),  C01);
    vst1q_f32(&Cref(1,3),  C02);
    vst1q_f32(&Cref(1,4),  C03);
    vst1q_f32(&Cref(1,5),  C04);
    vst1q_f32(&Cref(1,6),  C05);
    vst1q_f32(&Cref(1,7),  C06);
    vst1q_f32(&Cref(1,8),  C07);

    vst1q_f32(&Cref(5,1), C10);
    vst1q_f32(&Cref(5,2), C11);
    vst1q_f32(&Cref(5,3), C12);
    vst1q_f32(&Cref(5,4), C13);
    vst1q_f32(&Cref(5,5), C14);
    vst1q_f32(&Cref(5,6), C15);
    vst1q_f32(&Cref(5,7), C16);
    vst1q_f32(&Cref(5,8), C17);

  }
  else {
    printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", m, n);
    exit(-1);
  }
}




/*---------------NEON MICROKERNEL 4X8---------------------------------- */	
void gemm_microkernel_neon_4x8( int m, int n, int k, float alpha,
	       			float *A, float *B,
			       	float beta, float *C, int Clda ){
/*
  BLIS GEMM microkernel, computes the product Cr := beta * Cr + alpha * Ar * Br
  Specific: only for MRxNR = 4x8
  Cr --> m x n
  Ar --> m x k
  Br --> k x n
 
*/

  int           i, j, kc, baseA, baseB, Ctlda = MR;
  float32x4_t 	C1, C2, C3, C4,
		C5, C6, C7, C8;

  float32x4_t	A1, A2, A3, A4,
		A5, A6, A7, A8;

  float32x4_t	B0, B1;

  float         zero = 0.0, one = 1.0, Ctmp[MR*NR];

  if (k == 0)
    return;
 
  // Set to zeros
  C1 = vmovq_n_f32(0);
  C2 = vmovq_n_f32(0);
  C3 = vmovq_n_f32(0);
  C4 = vmovq_n_f32(0);

  C5 = vmovq_n_f32(0);
  C6 = vmovq_n_f32(0);
  C7 = vmovq_n_f32(0);
  C8 = vmovq_n_f32(0);


  // Iterate from 1 to kc-1 (last iteration outside loop)
  for ( kc=1; kc<=k; kc++ ) {

    // Prefect colum/row of A/B for next iteration
    // This is done to overlap loading Ar/Br with the computations in the present iteration
    baseA = (kc-1)*MR;
    baseB = (kc-1)*NR;
    A1 = vld1q_f32(&A[baseA]);

    B0 = vld1q_f32(&B[baseB]);
    B1 = vld1q_f32(&B[baseB+4]);
     
    // Accumulate in each column of Cr, the product of one column of Ar (in A1) and one element of Br (in B1)
    C1 = vfmaq_laneq_f32(C1, A1, B0, 0);
    C2 = vfmaq_laneq_f32(C2, A1, B0, 1);
    C3 = vfmaq_laneq_f32(C3, A1, B0, 2);
    C4 = vfmaq_laneq_f32(C4, A1, B0, 3);

    C5 = vfmaq_laneq_f32(C5, A1, B1, 0);
    C6 = vfmaq_laneq_f32(C6, A1, B1, 1);
    C7 = vfmaq_laneq_f32(C7, A1, B1, 2);
    C8 = vfmaq_laneq_f32(C8, A1, B1, 3);

  }

  if (alpha==-one) {
    // If alpha==-1, change sign of result
    C1 = -C1; C2 = -C2; C3 = -C3; C4 = -C4,
    C5 = -C5; C6 = -C6; C7 = -C7; C8 = -C8;
  }
  else if (alpha!=one) {
    // If alpha!=one, multiply the result in the vector registers by that value
    C1 = alpha*C1; C2 = alpha*C2; C3 = alpha*C3; C4 = alpha*C4,
    C5 = alpha*C5; C6 = alpha*C6; C7 = alpha*C7; C8 = alpha*C8;

  }

  if ((m<MR)||(n<NR)) {
    // If the result is smaller than m x n, we cannot use vector instructions to store it in memory. Do it "manually", by first storing it into Ctmp
    // and then from there to the appropriate positions in memory
    
    vst1q_f32(&Ctref(1,1), C1);
    vst1q_f32(&Ctref(1,2), C2);
    vst1q_f32(&Ctref(1,3), C3);
    vst1q_f32(&Ctref(1,4), C4);
    vst1q_f32(&Ctref(1,5), C5);
    vst1q_f32(&Ctref(1,6), C6);
    vst1q_f32(&Ctref(1,7), C7);
    vst1q_f32(&Ctref(1,8), C8);

    if (beta!=zero) {
      // If beta!=0.0, C must be multiplied by that value
      for (j=1; j<=n; j++)
        for (i=1; i<=m; i++)
          Cref(i,j) = beta*Cref(i,j) + Ctref(i,j);
    }
    else {
      // If beta==0.0, no need to retrieve C from memory. Overwrite the value in memory with the result.
      for (j=1; j<=n; j++)
        for (i=1; i<=m; i++)
          Cref(i,j) = Ctref(i,j);
    }
  }
  else if ((m==MR)||(n==NR)) {
    // If the result is exactly MR x NR, utilize vector instructions to store it in memory
    if (beta!=zero) {
      // If beta!=0.0, C must be multiplied by that value. Use A1,A2,A3,A4 as temporary values to retrieve the values of C from memory
      
      A1 = vld1q_f32(&Cref(1,1));
      A2 = vld1q_f32(&Cref(1,2));
      A3 = vld1q_f32(&Cref(1,3));
      A4 = vld1q_f32(&Cref(1,4));
      A5 = vld1q_f32(&Cref(1,5));
      A6 = vld1q_f32(&Cref(1,6));
      A7 = vld1q_f32(&Cref(1,7));
      A8 = vld1q_f32(&Cref(1,8));
      
      C1 = beta*A1 + C1;
      C2 = beta*A2 + C2;
      C3 = beta*A3 + C3;
      C4 = beta*A4 + C4;
      C5 = beta*A5 + C5;
      C6 = beta*A6 + C6;
      C7 = beta*A7 + C7;
      C8 = beta*A8 + C8;

    }

    vst1q_f32(&Cref(1,1), C1);
    vst1q_f32(&Cref(1,2), C2);
    vst1q_f32(&Cref(1,3), C3);
    vst1q_f32(&Cref(1,4), C4);
    vst1q_f32(&Cref(1,5), C5);
    vst1q_f32(&Cref(1,6), C6);
    vst1q_f32(&Cref(1,7), C7);
    vst1q_f32(&Cref(1,8), C8);

  }
  else {
    printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", m, n);
    exit(-1);
  }
}


/*---------------NEON MICROKERNEL 4X4  PREFETCH---------------------------------- */	
void gemm_microkernel_neon_4x4_prefetch( int m, int n, int k, float alpha,
	       				float *A, float *B,
			       		float beta, float *C, int Clda ){
/*
  BLIS GEMM microkernel, computes the product Cr := beta * Cr + alpha * Ar * Br
  Specific: only for MRxNR = 4x4
 
  Use prefectch for address.

  Cr --> m x n
  Ar --> m x k
  Br --> k x n
 
*/

  int           i, j, kc, baseA, baseB, Ctlda = MR;
  float32x4_t 	C1, C2, C3, C4,
     		A1, A2, A3, A4,
	        A1n, B1n,	
		B1;
  float         zero = 0.0, one = 1.0, Ctmp[MR*NR];

  if (k == 0)
    return;
 
  // Set to zero register
  // => v = { 0, 0, 0, 0 }
  C1 = vmovq_n_f32(0);
  C2 = vmovq_n_f32(0);
  C3 = vmovq_n_f32(0);
  C4 = vmovq_n_f32(0);


  // pre-load first column/row of A/B 
  A1 = vld1q_f32(&A[0]);
  B1 = vld1q_f32(&B[0]);
 

  // Iterate from 1 to kc-1 (last iteration outside loop) for prefetch
  for ( kc=1; kc<=k-1; kc++ ) {

    // Prefect colum/row of A/B for next iteration
    // This is done to overlap loading Ar/Br with the computations in the present iteration
    baseA = (kc-1)*MR;
    baseB = (kc-1)*NR;
    A1n = vld1q_f32(&A[baseA]);
    B1n = vld1q_f32(&B[baseB]);
     
    // Accumulate in each column of Cr, the product of one column of Ar (in A1) and one element of Br (in B1)
    C1 = vfmaq_laneq_f32(C1, A1, B1, 0);
    C2 = vfmaq_laneq_f32(C2, A1, B1, 1);
    C3 = vfmaq_laneq_f32(C3, A1, B1, 2);
    C4 = vfmaq_laneq_f32(C4, A1, B1, 3);

    // Load the new values for next iteration
    A1 = A1n;
    B1 = B1n;
  }

 //Last iteration k = 4 
   C1 = vfmaq_laneq_f32(C1, A1, B1, 0);
   C2 = vfmaq_laneq_f32(C2, A1, B1, 1);
   C3 = vfmaq_laneq_f32(C3, A1, B1, 2);
   C4 = vfmaq_laneq_f32(C4, A1, B1, 3);

  if (alpha==-one) {
    // If alpha==-1, change sign of result
    C1 = -C1; C2 = -C2; C3 = -C3; C4 = -C4;
  }
  else if (alpha!=one) {
    // If alpha!=one, multiply the result in the vector registers by that value
    C1 = alpha*C1; C2 = alpha*C2; C3 = alpha*C3; C4 = alpha*C4;
  }

  if ((m<MR)||(n<NR)) {
    // If the result is smaller than m x n, we cannot use vector instructions to store it in memory. Do it "manually", by first storing it into Ctmp
    // and then from there to the appropriate positions in memory
    vst1q_f32(&Ctref(1,1), C1);
    vst1q_f32(&Ctref(1,2), C2);
    vst1q_f32(&Ctref(1,3), C3);
    vst1q_f32(&Ctref(1,4), C4);
    if (beta!=zero) {
      // If beta!=0.0, C must be multiplied by that value
      for (j=1; j<=n; j++)
        for (i=1; i<=m; i++)
          Cref(i,j) = beta*Cref(i,j) + Ctref(i,j);
    }
    else {
      // If beta==0.0, no need to retrieve C from memory. Overwrite the value in memory with the result.
      for (j=1; j<=n; j++)
        for (i=1; i<=m; i++)
          Cref(i,j) = Ctref(i,j);
    }
  }
  else if ((m==MR)||(n==NR)) {
    // If the result is exactly MR x NR, utilize vector instructions to store it in memory
    if (beta!=zero) {
      // If beta!=0.0, C must be multiplied by that value. Use A1,A2,A3,A4 as temporary values to retrieve the values of C from memory
      A1 = vld1q_f32(&Cref(1,1));
      A2 = vld1q_f32(&Cref(1,2));
      A3 = vld1q_f32(&Cref(1,3));
      A4 = vld1q_f32(&Cref(1,4));
      C1 = beta*A1 + C1;
      C2 = beta*A2 + C2;
      C3 = beta*A3 + C3;
      C4 = beta*A4 + C4;
    }
    vst1q_f32(&Cref(1,1), C1);
    vst1q_f32(&Cref(1,2), C2);
    vst1q_f32(&Cref(1,3), C3);
    vst1q_f32(&Cref(1,4), C4);
  }
  else {
    printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", m, n);
    exit(-1);
  }
}


/*------------------------------------------------------------------------------*/
/*-------------------NEON MICROKERNEL 8X8 PREFETCH---------------------------------------*/
void gemm_microkernel_neon_8x8_prefetch( int m, int n, int k, float alpha,
	       			float *A, float *B,
			       	float beta, float *C, int Clda ){
/*
  BLIS GEMM microkernel, computes the product Cr := beta * Cr + alpha * Ar * Br
  Specific: only for MRxNR = 8x8
  Cr --> m x n
  Ar --> m x k
  Br --> k x n
*/
  int         i, j, kc, baseA, baseB, Ctlda = MR;

  // columns of C 
  float32x4_t 	C00, C01, C02, C03, C04, C05, C06, C07,
	       	C10, C11, C12, C13, C14, C15, C16, C17;
  // Columns of A
  float32x4_t   A0, A1, A2 , A3 , A4 , A5 , A6 , A7;

  // Bij		
  float32x4_t   B0, B1;

  //Pre-load data 
  float32x4_t   A0n, A1n, B0n, B1n;

 float       zero = 0.0, one = 1.0, Ctmp[MR*NR];

  if (k == 0)
    return;

  // Set to zeros columns of C
  C00  = vmovq_n_f32(0); C10 = vmovq_n_f32(0);
  C01  = vmovq_n_f32(0); C11 = vmovq_n_f32(0);
  C02  = vmovq_n_f32(0); C12 = vmovq_n_f32(0);
  C03  = vmovq_n_f32(0); C13 = vmovq_n_f32(0);
  C04  = vmovq_n_f32(0); C14 = vmovq_n_f32(0);
  C05  = vmovq_n_f32(0); C15 = vmovq_n_f32(0);
  C06  = vmovq_n_f32(0); C16 = vmovq_n_f32(0);
  C07  = vmovq_n_f32(0); C17 = vmovq_n_f32(0); 

 // Pre-load a first data row/column
 // |----|      
 // | A0 |     |---------|
 // |----|  x  | B0 | B1 |
 // | A1 |     |---------|
 // |----|    
  A0 = vld1q_f32(&A[0]); 	// A0 
  A1 = vld1q_f32(&A[4]); 	// A1

  B0 = vld1q_f32(&B[0]);	// B0    
  B1 = vld1q_f32(&B[4]);  	// B1


  // Iterate from 1 to kc-1 (last iteration outside loop)
  for ( kc=1; kc<=k-1; kc++ ) {

    // Prefect colum/row of A/B for next iteration
    // This is done to overlap loading Ar/Br with the computations in the present iteration
    baseA = (kc-1)*MR; 
    baseB = (kc-1)*NR;  
        
    A0n = vld1q_f32(&A[baseA]);
    A1n = vld1q_f32(&A[baseA+4]); 
	
    B0n = vld1q_f32(&B[baseB]);	    
    B1n = vld1q_f32(&B[baseB+4]);
     
    // Accumulate in each column of Cr, the product of one column of Ar (in A1) and one element of Br (in B1)
    // Cij = Cij + Ai * Bj(lane)
    //
    C00   = vfmaq_laneq_f32(C00,  A0, B0, 0); 
    C01   = vfmaq_laneq_f32(C01,  A0, B0, 1);
    C02   = vfmaq_laneq_f32(C02,  A0, B0, 2);
    C03   = vfmaq_laneq_f32(C03,  A0, B0, 3);

    C04   = vfmaq_laneq_f32(C04,  A0, B1, 0);
    C05   = vfmaq_laneq_f32(C05,  A0, B1, 1);
    C06   = vfmaq_laneq_f32(C06,  A0, B1, 2);
    C07   = vfmaq_laneq_f32(C07,  A0, B1, 3);

    C10   = vfmaq_laneq_f32(C10,  A1, B0, 0);
    C11   = vfmaq_laneq_f32(C11,  A1, B0, 1);
    C12   = vfmaq_laneq_f32(C12,  A1, B0, 2);
    C13   = vfmaq_laneq_f32(C13,  A1, B0, 3);

    C14   = vfmaq_laneq_f32(C14,  A1, B1, 0);
    C15   = vfmaq_laneq_f32(C15,  A1, B1, 1);
    C16   = vfmaq_laneq_f32(C16,  A1, B1, 2);
    C17   = vfmaq_laneq_f32(C17,  A1, B1, 3);

    // Update de data for next iteration
    A0 = A0n;  
    A1 = A1n;
    
    B0 = B0n;
    B1 = B0n;

  }

  if (alpha==-one) {
    // If alpha==-1, change sign of result
    C00 = -C00;
    C01 = -C01;
    C02 = -C02;
    C03 = -C03;
    C04 = -C04;
    C05 = -C05;
    C06 = -C06;
    C07 = -C07;

    C10 = -C10;
    C11 = -C11;
    C12 = -C12;
    C13 = -C13;
    C14 = -C14;
    C15 = -C15;
    C16 = -C16;
    C17 = -C17;

  }
  else if (alpha!=one) {
    // If alpha!=one, multiply the result in the vector registers by that value
    C00 = alpha*C00;
    C01 = alpha*C01;
    C02 = alpha*C02;
    C03 = alpha*C03;
    C04 = alpha*C04;
    C05 = alpha*C05;
    C06 = alpha*C06;
    C07 = alpha*C07;

    C10 = alpha*C10;
    C11 = alpha*C11;
    C12 = alpha*C12;
    C13 = alpha*C13;
    C14 = alpha*C14;
    C15 = alpha*C15;
    C16 = alpha*C16;
    C17 = alpha*C17;
  }

  if ((m<MR)||(n<NR)) {
    // If the result is smaller than m x n, we cannot use vector instructions to store it in memory. Do it "manually", by first storing it into Ctmp
    // and then from there to the appropriate positions in memory
    
    vst1q_f32(&Ctref(1,1), C00);
    vst1q_f32(&Ctref(1,2), C01);
    vst1q_f32(&Ctref(1,3), C02);
    vst1q_f32(&Ctref(1,4), C03);
    vst1q_f32(&Ctref(1,5), C04);
    vst1q_f32(&Ctref(1,6), C05);
    vst1q_f32(&Ctref(1,7), C06);
    vst1q_f32(&Ctref(1,8), C07);

    vst1q_f32(&Ctref(5,1), C10);
    vst1q_f32(&Ctref(5,2), C11);
    vst1q_f32(&Ctref(5,3), C12);
    vst1q_f32(&Ctref(5,4), C13);
    vst1q_f32(&Ctref(5,5), C14);
    vst1q_f32(&Ctref(5,6), C15);
    vst1q_f32(&Ctref(5,7), C16);
    vst1q_f32(&Ctref(5,8), C17);

    if (beta!=zero) {
      // If beta!=0.0, C must be multiplied by that value
      for (j=1; j<=n; j++)
        for (i=1; i<=m; i++)
          Cref(i,j) = beta*Cref(i,j) + Ctref(i,j);
    }
    else {
      // If beta==0.0, no need to retrieve C from memory. Overwrite the value in memory with the result.
      for (j=1; j<=n; j++)
        for (i=1; i<=m; i++)
          Cref(i,j) = Ctref(i,j);
    }
  }
  else if ((m==MR)||(n==NR)) {
    // If the result is exactly MR x NR, utilize vector instructions to store it in memory


	  
    if (beta!=zero) {

// If beta!=0.0, C must be multiplied by that value. Use A1,A2,A3,A4 as temporary values to retrieve the values of C from memory


      A0  = vld1q_f32(&Cref(1,1));
      A1  = vld1q_f32(&Cref(1,2));
      A2  = vld1q_f32(&Cref(1,3));
      A3  = vld1q_f32(&Cref(1,4));
      A4  = vld1q_f32(&Cref(1,5));
      A5  = vld1q_f32(&Cref(1,6));
      A6  = vld1q_f32(&Cref(1,7));
      A7  = vld1q_f32(&Cref(1,8));

      C00   = beta*A0  + C00;
      C01   = beta*A1  + C01;
      C02   = beta*A2  + C02;
      C03   = beta*A3  + C03;
      C04   = beta*A4  + C04;
      C05   = beta*A5  + C05;
      C06   = beta*A6  + C06;
      C07   = beta*A7  + C07;
    }
    // Reuse the vectorial register
    if (beta!= zero){

      A0  = vld1q_f32(&Cref(5,1));
      A1  = vld1q_f32(&Cref(5,2));
      A2  = vld1q_f32(&Cref(5,3));
      A3  = vld1q_f32(&Cref(5,4));
      A4  = vld1q_f32(&Cref(5,5));
      A5  = vld1q_f32(&Cref(5,6));
      A6  = vld1q_f32(&Cref(5,7));
      A7  = vld1q_f32(&Cref(5,8));

      C10   = beta*A0 + C10;
      C11   = beta*A1 + C11;
      C12   = beta*A2 + C12;
      C13   = beta*A3 + C13;
      C14   = beta*A4 + C14;
      C15   = beta*A5 + C15;
      C16   = beta*A6 + C16;
      C17   = beta*A7 + C17;
    }
    // store the values in register 
    vst1q_f32(&Cref(1,1),  C00);
    vst1q_f32(&Cref(1,2),  C01);
    vst1q_f32(&Cref(1,3),  C02);
    vst1q_f32(&Cref(1,4),  C03);
    vst1q_f32(&Cref(1,5),  C04);
    vst1q_f32(&Cref(1,6),  C05);
    vst1q_f32(&Cref(1,7),  C06);
    vst1q_f32(&Cref(1,8),  C07);

    vst1q_f32(&Cref(5,1), C10);
    vst1q_f32(&Cref(5,2), C11);
    vst1q_f32(&Cref(5,3), C12);
    vst1q_f32(&Cref(5,4), C13);
    vst1q_f32(&Cref(5,5), C14);
    vst1q_f32(&Cref(5,6), C15);
    vst1q_f32(&Cref(5,7), C16);
    vst1q_f32(&Cref(5,8), C17);

  }
  else {
    printf("Error: Incorrect use of 4x4 micro-kernel with %d x %d block\n", m, n);
    exit(-1);
  }
}

