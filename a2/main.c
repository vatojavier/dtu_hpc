/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"
#include <omp.h>
#include <string.h>

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
    double start_T;     // Starting temperature
    int used_iter;      // Number of iterations used by function
    double time_start, time_end;    // Wall clock
    int output_type = 0;    // store as binary or vtk
    int exp_type = 0;       // Experiment type
    char *output_prefix = "poisson_res";
    char *output_ext = "";
    char output_filename[FILENAME_MAX];
    double ***u = NULL;
    double ***u2 = NULL;
    double ***f = NULL;

    char method_name[3];

    /* get the paramters from the command line */
    N = atoi(argv[1]);         // grid size
    iter_max = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]); // tolerance
    start_T = atof(argv[4]);   // start T for all inner grid points
    output_type = atoi(argv[5]); // ouput type

    exp_type = atoi(argv[6]); // Experiment type

    // char exp_type_str = malloc(strlen(argv[6]) + 1);
    // if(exp_type_str == NULL){
    //     perror("exp_type_str: allocation failed");
    //     exit(-1);
    // }
    // strcpy(exp_type_str, argv[6]);
    
    
    // Increment N by two
    int N2 = N + 2;
    // get the number of threads available to the program
    int n_threads = omp_get_max_threads();
    // printf("Number of threads: %d\n", n_threads);

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
    init_jacobi(u, u2, f, N2, start_T);
    #endif

    #ifdef _GAUSS_SEIDEL
    init_seidel(u, f, N2, start_T);
    #endif

    // Call to Jacobi or Gauss-Seidel
    #ifdef _JACOBI
    strcpy(method_name, "ja");
    time_start = omp_get_wtime();

    switch (exp_type){
        case 1:
            used_iter = jacobi(u, u2, f, iter_max, N, tolerance);
            break;
        case 2:
            used_iter = jacobi_baseline(u, u2, f, iter_max, N, tolerance);
            break;
        case 3:
            used_iter = jacobi_improved(u, u2, f, iter_max, N, tolerance);
            break;
    }
    time_end = omp_get_wtime();
    printf("%lf %d %d %d %lf %lf %d \n", time_end - time_start, used_iter, iter_max, N, tolerance, start_T, n_threads);
    #endif

    #ifdef _GAUSS_SEIDEL
    strcpy(method_name, "gs");
    char seq_par[4];

    time_start = omp_get_wtime();
    switch (exp_type){
        case 1:
            used_iter = gauss_seidel_seq(u, f, iter_max, N, tolerance);
            strcpy(seq_par, "SEQ");
            break;
        case 2:
            used_iter = gauss_seidel_omp_wrong(u, f, iter_max, N, tolerance);
            strcpy(seq_par, "WRG");
            break;
        case 3:
            used_iter = gauss_seidel_omp(u, f, iter_max, N, tolerance);
            strcpy(seq_par, "PAR");
            break;
    }
    time_end = omp_get_wtime();
    printf("%lf %d %d %d %lf %lf %d %s%s \n", time_end - time_start, used_iter, iter_max, N, tolerance, start_T, n_threads, method_name, seq_par);
    // printf("Time took: %lf\nIterations: %d\nMax iterations: %d\nGrid size: %d\nTolerance: %lf\nStart T: %lf\n", time_end - time_start, used_iter, iter_max, N, tolerance, start_T);
    #endif

    // dump  results if wanted
    switch (output_type)
    {
    case 0:
        // no output at all
        break;
    case 3:
        output_ext = ".bin";
        sprintf(output_filename, "%s_%s_%d_%d%s", output_prefix, method_name , exp_type , N, output_ext);
        fprintf(stderr, "Write binary dump to %s: ", output_filename);
        print_binary(output_filename, N2, u);
        break;
    case 4:
        output_ext = ".vtk";
        sprintf(output_filename, "%s_%s_%d_%d%s", output_prefix, method_name , exp_type , N, output_ext);
        fprintf(stderr, "Write VTK file to %s: ", output_filename);
        print_vtk(output_filename, N2, u);
        break;
    default:
        fprintf(stderr, "Non-supported output type!\n");
        break;
    }

    // de-allocate memory
    free_3d(u);
    #ifdef _JACOBI
    free_3d(u2);
    #endif
    free_3d(f);

    return (0);
}
