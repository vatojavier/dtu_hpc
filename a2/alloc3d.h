#ifndef __ALLOC_3D
#define __ALLOC_3D

double ***malloc_3d(int m, int n, int k);

void init_jacobi(double ***, double ***, double***, int);

void init_seidel(double ***, double ***, int);

#define HAS_FREE_3D
void free_3d(double ***array3D);

#endif /* __ALLOC_3D */
