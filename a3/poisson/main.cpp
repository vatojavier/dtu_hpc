/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"
#include "init.h"
#include <omp.h>
#include <string.h>
#include "jacobi.h"


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

    // exp_type = atoi(argv[6]); // Experiment type

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
    if ((u2 = malloc_3d(N2, N2, N2)) == NULL)
    {
        perror("array u: allocation failed");
        exit(-1);
    }
    if ((f = malloc_3d(N2, N2, N2)) == NULL){
        perror("array u: allocation failed");
        exit(-1);
    }

    // Set boundary conditions
    init_jacobi(u, u2, f, N2, start_T);

    // Here we test d_malloc_3d()

    //double *data;
    //double ***old_dev = d_malloc_3d(N2, N2, N2, &data);
    //omp_target_memcpy(data, u[0][0], N2*N2*N2*sizeof(double), 0, 0, omp_get_default_device(), omp_get_initial_device());
    //#pragma omp target
    //{
    //    printf("Hello from device. Value of old_dev[0][0][0] is %lf \n", old_dev[0][0][0]);
    //    printf("Also the value of old_dev[2][0][2] is %lf \n", old_dev[2][0][2]);
    //}
    
    /*
    printf("Now it is the host speaking. We will run the three jacobi functions. \n");    
    time_start = omp_get_wtime();
    used_iter = jacobi_improved(u, u2, f, iter_max, N, tolerance);
    time_end = omp_get_wtime();
    printf("%d %lf CPU \n", N, time_end - time_start);

    // GPU MAP
    init_jacobi(u, u2, f, N2, start_T);
    time_start = omp_get_wtime();
    used_iter = jacobi_offload_map(u, u2, f, iter_max, N, tolerance);
    time_end = omp_get_wtime();
    printf("%d %lf GPUMAP \n", N, time_end - time_start);
    //printf("Sanity check: %lf \n", u[50][50][50]);

    // GPU MEMCPY
    init_jacobi(u, u2, f, N2, start_T);

    */
    time_start = omp_get_wtime();
    used_iter = jacobi_offload_memcopy(u, u2, f, iter_max, N, tolerance);
    time_end = omp_get_wtime();
    printf("%d %lf GPUCPY \n", N, time_end - time_start);
    printf("Sanity check: %lf \n", u[50][50][50]);

    // Exercise 8
    /*
    
    init_jacobi(u, u2, f, N2, start_T);
    time_start = omp_get_wtime();
    used_iter = jacobi_offload_norm(u, u2, f, iter_max, N, tolerance);
    time_end = omp_get_wtime();
    printf("%d %lf GPUCPY \n", N, time_end - time_start);
    printf("Sanity check: %lf \n", u[50][50][50]);

    */

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
    free_3d(u2);
    free_3d(f);
    //d_free_3d(old_dev, data);

    return (0);
}
