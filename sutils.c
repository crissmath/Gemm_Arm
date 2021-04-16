#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_LENGTH_LINE  256

#define Aref(a1,a2)  A[ (a2-1)*(Alda)+(a1-1) ]
#define xref(a1)     x[ (a1-1) ]

/*===========================================================================*/
double dclock()
{
/* 
 * Timer
 *
 */
  struct timeval  tv;
  // struct timezone tz;

  gettimeofday( &tv, NULL );   

  return (double) (tv.tv_sec + tv.tv_usec*1.0e-6);
}
/*===========================================================================*/
int generate_matrix( int m, int n, float *A, int Alda )
{
/*
 * Generate a matrix with random entries
 * m      : Row dimension
 * n      : Column dimension
 * A      : Matrix
 * Alda   : Leading dimension
 *
 */
  int i, j;

  for ( j=1; j<=n; j++ )
    for ( i=1; i<=m; i++ )
      Aref(i,j) = ((float) rand())/RAND_MAX;
      // Aref(i,j) = (float) (j-1)*m+i;

  return 0;
}
/*===========================================================================*/
int print_matrix( char *name, int m, int n, float *A, int Alda )
{
/*
 * Print a matrix to standard output
 * name   : Label for matrix name
 * m      : Row dimension
 * n      : Column dimension
 * A      : Matrix
 * Alda   : Leading dimension
 *
 */
  int i, j;

  for ( j=1; j<=n; j++ )
    for ( i=1; i<=m; i++ )
      printf( "   %s(%d,%d) = %14.8e;\n", name, i, j, Aref(i,j) );

  return 0;
}
/*===========================================================================*/
