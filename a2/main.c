/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"

#ifdef _JACOBI
#include "jacobi.h"
#endif

#ifdef _GAUSS_SEIDEL
#include "gauss_seidel.h"
#endif

#define N_DEFAULT 100

int main(int argc, char *argv[])
{

    int N = N_DEFAULT;
    int iter_max = 1000;
    double tolerance;
    double start_T;
    int output_type = 0;
    char *output_prefix = "poisson_res";
    char *output_ext = "";
    char output_filename[FILENAME_MAX];
    double ***u = NULL;
    double ***u2 = NULL;
    double ***f = NULL;
    double x,y,z;

    /* get the paramters from the command line */
    N = atoi(argv[1]);         // grid size
    iter_max = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]); // tolerance
    start_T = atof(argv[4]);   // start T for all inner grid points
    if (argc == 6)
    {
        output_type = atoi(argv[5]); // ouput type
    }

    // Increment N by two
    int N2 = N + 2;

    // allocate memory
    if ((u = malloc_3d(N2, N2, N2)) == NULL)
    {
        perror("array u: allocation failed");
        exit(-1);
    }
    if ((f = malloc_3d(N2, N2, N2)) == NULL){
        perror("array u: allocation failed");
        exit(-1);
    }

    // Set boundary conditions
    #ifdef _JACOBI
    if ((u2 = malloc_3d(N2, N2, N2)) == NULL)
    {
        perror("array u: allocation failed");
        exit(-1);
    }
    init_jacobi(u, u2, f, N2);
    #endif

    #ifdef _GAUSS_SEIDEL
    init_seidel(u, f, N2);
    #endif
    /*
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            for(int k = 0; k < N2; k++){
                u[i][j][k] = start_T;
                u2[i][j][k] = start_T;
            }
        }
    }
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            u[0][i][j] = 20.0;
            u[N+1][i][j] = 20.0;
            u[i][0][j] = 0.0;
            u[i][N+1][j] = 20.0;
            u[i][j][0] = 20.0;
            u[i][j][N+1] = 20.0;
            u2[0][i][j] = 20.0;
            u2[N+1][i][j] = 20.0;
            u2[i][0][j] = 0.0;
            u2[i][N+1][j] = 20.0;
            u2[i][j][0] = 20.0;
            u2[i][j][N+1] = 20.0;
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
    for(int i = (N+1)/6; i <= (N+1)/2; i++){
        for(int j = 0; j <= (N+1)/4; j++){
            for(int k = 0; k <= 5*(N+1)/16; k++){
                f[i][j][k] = 200.0;
            }
        }
    }
    */



    /*
     *
     * fill in your code here
     *
     *
     */

    #ifdef _JACOBI
    jacobi(u, u2, f, iter_max, N, tolerance);
    #endif
    
    // dump  results if wanted
    switch (output_type)
    {
    case 0:
        // no output at all
        break;
    case 3:
        output_ext = ".bin";
        sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
        fprintf(stderr, "Write binary dump to %s: ", output_filename);
        print_binary(output_filename, N2, u);
        break;
    case 4:
        output_ext = ".vtk";
        sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
        fprintf(stderr, "Write VTK file to %s: ", output_filename);
        print_vtk(output_filename, N2, u);
        break;
    default:
        fprintf(stderr, "Non-supported output type!\n");
        break;
    }

    // de-allocate memory
    free_3d(u);
    free_3d(u2);
    free_3d(f);

    return (0);
}
