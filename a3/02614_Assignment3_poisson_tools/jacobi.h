/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

double norm(double ***, double ***, int);
int jacobi(double ***, double ***, double ***, int, int, double);
int jacobi_baseline(double ***, double ***, double ***, int, int, double);
int jacobi_improved(double ***, double ***, double ***, int, int, double);

#endif
