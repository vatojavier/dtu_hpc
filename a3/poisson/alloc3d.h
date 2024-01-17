#ifndef __ALLOC_3D
#define __ALLOC_3D

double ***malloc_3d(int m, int n, int k);

double ***d_malloc_3d(int m, int n, int k, double **data);

void init_jacobi(double ***, double ***, double***, int, double);

void init_seidel(double ***, double ***, int, double);

#define HAS_FREE_3D
void free_3d(double ***array3D);
void d_free_3d(double ***p, double *data);

#endif /* __ALLOC_3D */
