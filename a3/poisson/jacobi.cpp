/* jacobi.c - Poisson problem in 3d
 * 
 */

#include <math.h>
#include "alloc3d.h"
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

int
jacobi_improved(double ***old, double ***newVol, double ***f, int max_iter, int N, double tol) {
    
    // Variables we will use
    double ***temp;
    double h = 1.0/6.0;
    double delta_sq = 4.0/((double) N*N+2*N+1);
    int n = 0;
    int i,j,k = 0;

    // Main loop of jacobi
    while(n < max_iter)
    {
        #pragma omp parallel shared(old, newVol, f, N, h, delta_sq) private(i, j, k)
        {
        
        // Compute newVol 3d matrix
        #pragma omp for collapse(2)
        for(i = 1; i < N+1; i++){
            for(j = 1; j < N+1; j++){
                for(k = 1; k < N+1; k++){
                    newVol[i][j][k] = h*(old[i-1][j][k] + old[i+1][j][k] + old[i][j-1][k] + old[i][j+1][k] + old[i][j][k-1] + old[i][j][k+1] + delta_sq*f[i][j][k]);
                }
            }
        } 

        } // End of parallel region

        // Switch pointers
        temp = old;
        old = newVol;
        newVol = temp;

        // Increment iteration counter
        n += 1;
    }
    
    return n;
}

int
jacobi_offload_map(double ***old, double ***newVol, double ***f, int max_iter, int N, double tol){
   
    // Variables we will use
    int N2 = N+2;
    double ***temp;
    double h = 1.0/6.0;
    double delta_sq = 4.0/((double) N*N+2*N+1);
    int n = 0;
    int i,j,k = 0;

    // Data transfer using map clause
    #pragma omp target enter data map(to: old[:N2][:N2][:N2]) map(to: f[:N2][:N2][:N2]) map(to: newVol[:N2][:N2][:N2])

    // Main loop of jacobi
    while(n < max_iter){
        #pragma omp target teams distribute parallel for num_teams(114) thread_limit(1000) shared(h, N, delta_sq) collapse(2)
        for(i = 1; i < N+1; i++){
            for(j = 1; j < N+1; j++){
                for(k = 1; k < N+1; k++){
                    newVol[i][j][k] = h*(old[i-1][j][k] + old[i+1][j][k] + old[i][j-1][k] + old[i][j+1][k] + old[i][j][k-1] + old[i][j][k+1] + delta_sq*f[i][j][k]);
                }
            }
        }

        // Switch pointers
        temp = old;
        old = newVol;
        newVol = temp;

        // Increment iteration counter
        n += 1;

    }

    // Data transfer to host
    #pragma omp target exit data map(from: old[:N2][:N2][:N2]) map(release: f[:N2][:N2][:N2]) map(release: newVol[:N2][:N2][:N2])
    #pragma omp target exit data map(release: old[:N2][:N2][:N2])
    
    return n;
}


int
jacobi_offload_memcopy(double ***old, double ***newVol, double ***f, int max_iter, int N, double tol){
   
    // Variables we will use
    double ***temp;
    double *temp2;
    double h = 1.0/6.0;
    double delta_sq = 4.0/((double) N*N+2*N+1);
    int n = 0;
    int i,j,k = 0;

    // Allocate memory on device
    double *data;
    double *data_f;
    double *data_new;
    double ***old_dev = d_malloc_3d(N+2, N+2, N+2, &data);
    double ***f_dev = d_malloc_3d(N+2, N+2, N+2, &data_f);
    double ***new_dev = d_malloc_3d(N+2, N+2, N+2, &data_new);

    // Data transfer using memcopy
    omp_target_memcpy(data, old[0][0], (N+2)*(N+2)*(N+2)*sizeof(double), 0, 0, omp_get_default_device(), omp_get_initial_device());
    omp_target_memcpy(data_f, f[0][0], (N+2)*(N+2)*(N+2)*sizeof(double), 0, 0, omp_get_default_device(), omp_get_initial_device());
    omp_target_memcpy(data_new, newVol[0][0], (N+2)*(N+2)*(N+2)*sizeof(double), 0, 0, omp_get_default_device(), omp_get_initial_device());

    // Main loop of jacobi
    while(n < max_iter){
        #pragma omp target teams distribute parallel for num_teams(114) thread_limit(1000) shared(h, delta_sq, N) collapse(2)
        for(i = 1; i < N+1; i++){
            for(j = 1; j < N+1; j++){
                for(k = 1; k < N+1; k++){
                    new_dev[i][j][k] = h*(old_dev[i-1][j][k] + old_dev[i+1][j][k] + old_dev[i][j-1][k] + old_dev[i][j+1][k] + old_dev[i][j][k-1] + old_dev[i][j][k+1] + delta_sq*f_dev[i][j][k]);
                }
            }
        }

        // Switch device pointers (the host does this)
        temp = old_dev;
        old_dev = new_dev;
        new_dev = temp;

        temp2 = data;
        data = data_new;
        data_new = temp2;

        // Increment iteration counter
        n += 1;

    }

    // Data transfer to host
    omp_target_memcpy(old[0][0], data, (N+2)*(N+2)*(N+2)*sizeof(double), 0, 0, omp_get_initial_device(), omp_get_default_device());

    // Free data on device
    d_free_3d(old_dev, data);
    d_free_3d(new_dev, data_new);
    d_free_3d(f_dev, data_f);
    
    return n;
}


int
jacobi_offload_multi(double ***old, double ***newVol, double ***f, int max_iter, int N, double tol){
   
    // Variables we will use
    int N2 = N+2;
    double ***temp;
    double *temp2;
    double h = 1.0/6.0;
    double delta_sq = 4.0/((double) N*N+2*N+1);
    int n = 0;
    int i,j,k = 0;

    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1,0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0,0);
    cudaSetDevice(0);

    // Allocate memory on device 0
    omp_set_default_device(0);
    double *data_d0;
    double *data_f_d0;
    double *data_new_d0;
    double ***old_dev_d0 = d_malloc_3d(N2/2, N2, N2, &data_d0);
    double ***f_dev_d0 = d_malloc_3d(N2/2, N2, N2, &data_f_d0);
    double ***new_dev_d0 = d_malloc_3d(N2/2, N2, N2, &data_new_d0);

    // Allocate memory on device 1
    omp_set_default_device(1);
    double *data_d1;
    double *data_f_d1;
    double *data_new_d1;
    double ***old_dev_d1 = d_malloc_3d(N2/2, N2, N2, &data_d1);
    double ***f_dev_d1 = d_malloc_3d(N2/2, N2, N2, &data_f_d1);
    double ***new_dev_d1 = d_malloc_3d(N2/2, N2, N2, &data_new_d1);

    // Data transfer using memcopy
    // Device 0
    omp_set_default_device(0);
    omp_target_memcpy(data_d0, old[0][0], (N2/2)*N2*N2*sizeof(double), 0, 0, omp_get_default_device(), omp_get_initial_device());
    omp_target_memcpy(data_f_d0, f[0][0], (N2/2)*N2*N2*sizeof(double), 0, 0, omp_get_default_device(), omp_get_initial_device());
    omp_target_memcpy(data_new_d0, newVol[0][0], (N2/2)*N2*N2*sizeof(double), 0, 0, omp_get_default_device(), omp_get_initial_device());

    // Device 1
    omp_set_default_device(1);
    omp_target_memcpy(data_d1, old[N2/2][0], (N2/2)*N2*N2*sizeof(double), 0, 0, omp_get_default_device(), omp_get_initial_device());
    omp_target_memcpy(data_f_d1, f[N2/2][0], (N2/2)*N2*N2*sizeof(double), 0, 0, omp_get_default_device(), omp_get_initial_device());
    omp_target_memcpy(data_new_d1, newVol[N2/2][0], (N2/2)*N2*N2*sizeof(double), 0, 0, omp_get_default_device(), omp_get_initial_device());


    // Main loop of jacobi
    while(n < max_iter){
        // Computations for device 0
        #pragma omp target teams distribute parallel for nowait shared(h, delta_sq, N) collapse(2) device(0)
        for(i = 1; i < N2/2; i++){
            for(j = 1; j < N+1; j++){
                if(i < N2/2 - 1){
                    for(k = 1; k < N+1; k++){
                        new_dev_d0[i][j][k] = h*(old_dev_d0[i-1][j][k] + old_dev_d0[i+1][j][k] + old_dev_d0[i][j-1][k] + old_dev_d0[i][j+1][k] + old_dev_d0[i][j][k-1] + old_dev_d0[i][j][k+1] + delta_sq*f_dev_d0[i][j][k]);
                    }
                }
                else{
                    for(k = 1; k < N+1; k++){
                        new_dev_d0[i][j][k] = h*(old_dev_d0[i-1][j][k] + old_dev_d1[0][j][k] + old_dev_d0[i][j-1][k] + old_dev_d0[i][j+1][k] + old_dev_d0[i][j][k-1] + old_dev_d0[i][j][k+1] + delta_sq*f_dev_d0[i][j][k]);
                    }
                }
            }
        }

        // Computations for device 1
        #pragma omp target teams distribute parallel for nowait shared(h, delta_sq, N) collapse(2) device(1)
        for(i = 0; i < N2/2-1; i++){
            for(j = 1; j < N+1; j++){
                if(i > 0){
                    for(k = 1; k < N+1; k++){
                            new_dev_d1[i][j][k] = h*(old_dev_d1[i-1][j][k] + old_dev_d1[i+1][j][k] + old_dev_d1[i][j-1][k] + old_dev_d1[i][j+1][k] + old_dev_d1[i][j][k-1] + old_dev_d1[i][j][k+1] + delta_sq*f_dev_d1[i][j][k]);
                    }
                }
                else{
                    for(k = 1; k < N+1; k++){
                        new_dev_d1[i][j][k] = h*(old_dev_d0[N2/2-1][j][k] + old_dev_d1[i+1][j][k] + old_dev_d1[i][j-1][k] + old_dev_d1[i][j+1][k] + old_dev_d1[i][j][k-1] + old_dev_d1[i][j][k+1] + delta_sq*f_dev_d1[i][j][k]);
                    }
                }
            }
        }

        #pragma omp taskwait

        // Switch device pointers (the host does this)
        temp = old_dev_d0;
        old_dev_d0 = new_dev_d0;
        new_dev_d0 = temp;

        temp2 = data_d0;
        data_d0 = data_new_d0;
        data_new_d0 = temp2;

        temp = old_dev_d1;
        old_dev_d1 = new_dev_d1;
        new_dev_d1 = temp;

        temp2 = data_d1;
        data_d1 = data_new_d1;
        data_new_d1 = temp2;

        // Increment iteration counter
        n += 1;

    }

    // Data transfer to host
    // Device 0
    omp_set_default_device(0);
    omp_target_memcpy(old[0][0], data_d0, (N2/2)*(N2)*(N2)*sizeof(double), 0, 0, omp_get_initial_device(), omp_get_default_device());

    // Device 1
    omp_set_default_device(1);
    omp_target_memcpy(old[N2/2][0], data_d1, (N2/2)*(N2)*(N2)*sizeof(double), 0, 0, omp_get_initial_device(), omp_get_default_device());

    // Free data on device
    omp_set_default_device(0);
    d_free_3d(old_dev_d0, data_d0);
    d_free_3d(new_dev_d0, data_new_d0);
    d_free_3d(f_dev_d0, data_f_d0);

    omp_set_default_device(1);
    d_free_3d(old_dev_d1, data_d1);
    d_free_3d(new_dev_d1, data_new_d1);
    d_free_3d(f_dev_d1, data_f_d1);
    
    return n;
}



int
jacobi_offload_norm(double ***old, double ***newVol, double ***f, int max_iter, int N, double tol){
   
    // Variables we will use
    int N2 = N+2;
    double ***temp;
    double h = 1.0/6.0;
    double delta_sq = 4.0/((double) N*N+2*N+1);
    double d = 10000.0;
    int n = 0;
    int i,j,k = 0;

    // Data transfer using map clause
    #pragma omp target enter data map(to: old[:N2][:N2][:N2]) map(to: f[:N2][:N2][:N2]) map(to: newVol[:N2][:N2][:N2])

    // Main loop of jacobi
    while(d > tol && n < max_iter){
        d = 0.0;
        #pragma omp target teams distribute parallel for shared(h, N, delta_sq) reduction(+: d) collapse(2)
        for(i = 1; i < N+1; i++){
            for(j = 1; j < N+1; j++){
                for(k = 1; k < N+1; k++){
                    newVol[i][j][k] = h*(old[i-1][j][k] + old[i+1][j][k] + old[i][j-1][k] + old[i][j+1][k] + old[i][j][k-1] + old[i][j][k+1] + delta_sq*f[i][j][k]);
                    //Norm
                    d+=(old[i][j][k] - newVol[i][j][k])*(old[i][j][k] - newVol[i][j][k]);
                }
            }
        }

        // Switch pointers
        temp = old;
        old = newVol;
        newVol = temp;

        // Increment iteration counter
        n += 1;

    }

    // Data transfer to host
    #pragma omp target exit data map(from: old[:N2][:N2][:N2]) map(release: f[:N2][:N2][:N2]) map(release: newVol[:N2][:N2][:N2])
    #pragma omp target exit data map(release: old[:N2][:N2][:N2])
    
    return n;
}
