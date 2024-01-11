/* gauss_seidel.h - Poisson problem
 *
 */
#ifndef _GAUSS_SEIDEL_H
#define _GAUSS_SEIDEL_H

// define your function prototype here
int gauss_seidel_seq(double ***, double ***, int , int , double );

int gauss_seidel_omp_wrong(double ***, double ***, int , int , double );

int gauss_seidel_omp(double ***, double ***, int , int , double );

#endif
