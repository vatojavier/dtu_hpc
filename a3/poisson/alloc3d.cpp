#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

double ***
malloc_3d(int m, int n, int k)
{

    if (m <= 0 || n <= 0 || k <= 0)
        return NULL;

    double ***p = (double ***)malloc(m * sizeof(double **) +
                                     m * n * sizeof(double *));
    if (p == NULL)
    {
        return NULL;
    }

    for (int i = 0; i < m; i++)
    {
        p[i] = (double **)p + m + i * n;
    }

    double *a = (double *)malloc(m * n * k * sizeof(double));
    if (a == NULL)
    {
        free(p);
        return NULL;
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            p[i][j] = a + (i * n * k) + (j * k);
        }
    }

    return p;
}

double ***d_malloc_3d(int m, int n, int k, double **data){
    // Abort if dimension are wrong
    if (m <= 0 || n <= 0 || k <= 0){
        return NULL;
    }

    // Allocate an array of pointers which contains array of pointers
    double ***p = (double ***) omp_target_alloc(m*sizeof(double **) + m*n*sizeof(double *), omp_get_default_device());
    if(p == NULL){
        return NULL;
    }

    // printf("testtest 0 \n");
    // Set the outer array of pointers
    #pragma omp target is_device_ptr(p)
    {
    for (int i = 0; i < m; i++){
        p[i] = (double **)p + m + i * n;
    }
    } // End target
    // printf("testtest 1 \n");
    // Allocate data vector
    double *a = (double *) omp_target_alloc(m*n*k*sizeof(double), omp_get_default_device());
    if(a == NULL){
        omp_target_free(p, omp_get_default_device());
        return NULL;
    }
    // printf("testtest 2 \n");
    // Set the inner array of pointers to match data vector
    #pragma omp target is_device_ptr(p, a)
    {
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            p[i][j] = a + (i * n * k) + (j * k);
        }
    }
    } // End target
    // printf("testtest 3 \n");
    *data = a;
    return p;
}

void free_3d(double ***p)
{
    free(p[0][0]);
    free(p);
}

void d_free_3d(double ***p, double *data){
    omp_target_free(data, omp_get_default_device());
    omp_target_free(p, omp_get_default_device());
}
