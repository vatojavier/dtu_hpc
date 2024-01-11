#include <stdlib.h>

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


void init_jacobi(double ***old, double ***new, double ***f, int N2, double T0){
    // Set boundary conditions
    #pragma omp parallel
    {
    #pragma omp for schedule(static, 20)
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            for(int k = 0; k < N2; k++){
                old[i][j][k] = T0;
                new[i][j][k] = T0;
            }
        }
    }
    #pragma omp for schedule(static, 20)
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            old[0][i][j] = 20.0;
            old[N2-1][i][j] = 20.0;
            old[i][0][j] = 0.0;
            old[i][N2-1][j] = 20.0;
            old[i][j][0] = 20.0;
            old[i][j][N2-1] = 20.0;
            new[0][i][j] = 20.0;
            new[N2-1][i][j] = 20.0;
            new[i][0][j] = 0.0;
            new[i][N2-1][j] = 20.0;
            new[i][j][0] = 20.0;
            new[i][j][N2-1] = 20.0;
        }
    }
    // } // END PARALLEL
    // #pragma omp parallel
    // {
    #pragma omp for schedule(static, 20)
    // Set source function (radiator)
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            for(int k = 0; k < N2; k++){
                f[i][j][k] = 0.0;
            }
        }
    }
    #pragma omp for schedule(static, 20)
    for(int i = (N2-1)/6; i <= (N2-1)/2; i++){
        for(int j = 0; j <= (N2-1)/4; j++){
            for(int k = 0; k <= (5*(N2-1))/16; k++){
                f[i][j][k] = 200.0;
            }
        }
    }
    } // END PARALLEL
    return;
}

void init_seidel(double ***u, double ***f, int N2, double T0){
    // Set boundary conditions
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            for(int k = 0; k < N2; k++){
                u[i][j][k] = T0;
            }
        }
    }
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            u[0][i][j] = 20.0;
            u[N2-1][i][j] = 20.0;
            u[i][0][j] = 0.0;
            u[i][N2-1][j] = 20.0;
            u[i][j][0] = 20.0;
            u[i][j][N2-1] = 20.0;
        }
    }
    // Set source function (radiator)
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            for(int k = 0; k < N2; k++){
                f[i][j][k] = 0.0;
            }
        }
    }
    for(int i = (N2-1)/6; i <= (N2-1)/2; i++){
        for(int j = 0; j <= (N2-1)/4; j++){
            for(int k = 0; k <= (5*(N2-1))/16; k++){
                f[i][j][k] = 200.0;
            }
        }
    }
    return;
}

void free_3d(double ***p)
{
    free(p[0][0]);
    free(p);
}
