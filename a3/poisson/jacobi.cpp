/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>
#include "alloc3d.h"


// // Frobenius norm
// double norm(double ***old, double ***newVol, int N){
// double res = 0.0;
// int i,j,k = 0;
// #pragma omp parallel shared(old, newVol, N) private(i, j, k) reduction(+:res)
// {
// #pragma omp for
// for(int i = 1; i < N+1; i++){
//     for(int j = 1; j < N+1; j++){
//         for(int k = 1; k < N+1; k++){
//             res += (old[i][j][k] - newVol[i][j][k])*(old[i][j][k] - newVol[i][j][k]);
//         }
//     }
// }
// } // End of parallel region
// return res;
// }

// Frobenius norm
// double norm(double ***old, double ***newVol, int N)
// {
// double res = 0.0;
// int i,j,k = 0;
// #pragma omp parallel for reduction(+:res)
// for(int i = 1; i < N+1; i++){
//     for(int j = 1; j < N+1; j++){
//         for(int k = 1; k < N+1; k++){
//             res += (old[i][j][k] - newVol[i][j][k])*(old[i][j][k] - newVol[i][j][k]);
//         }
//     }
// }
// // } // End of parallel region
// return res;
// }

/*
int
jacobi(double ***old, double ***newVol, double ***f, int max_iter, int N, double tol) {
    // Variables we will use
    double ***temp;
    double h = 1.0/6.0;
    double delta_sq = 4.0/((double) N*N+2*N+1);
    double d = INFINITY;
    int n = 0;

    // Main loop of jacobi
    while(d > tol && n < max_iter){
        d = 0.0;
        // Compute newVol 3d matrix
        for(int i = 1; i < N+1; i++){
            for(int j = 1; j < N+1; j++){
                for(int k = 1; k < N+1; k++){
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
    return n;
}


int
jacobi_baseline(double ***old, double ***newVol, double ***f, int max_iter, int N, double tol) {
    // Variables we will use
    double ***temp;
    double h = 1.0/6.0;
    double delta_sq = 4.0/((double) N*N+2*N+1);
    double d = 10000.0;
    int n = 0;
    
    int i,j,k = 0;
    // Main loop of jacobi
    while(d > tol && n < max_iter){
        d = 0.0;
        #pragma omp parallel shared(old, newVol, f, N, h, delta_sq) private(i, j, k) reduction(+:d)
        {
        // Compute newVol 3d matrix
            #pragma omp for
            for(i = 1; i < N+1; i++){
                for(j = 1; j < N+1; j++){
                    for(k = 1; k < N+1; k++){
                        newVol[i][j][k] = h*(old[i-1][j][k] + old[i+1][j][k] + old[i][j-1][k] + old[i][j+1][k] + old[i][j][k-1] + old[i][j][k+1] + delta_sq*f[i][j][k]);
                        //Norm
                        d+=(old[i][j][k] - newVol[i][j][k])*(old[i][j][k] - newVol[i][j][k]);

                    }
                }
            }
        } // end of parallel region
    

        // Switch pointers
        temp = old;
        old = newVol;
        newVol = temp;

        // Increment iteration counter
        n += 1;

    }
    return n;
}
*/


int
jacobi_improved(double ***old, double ***newVol, double ***f, int max_iter, int N, double tol) {
    
    
    // Variables we will use
    double ***temp;
    double h = 1.0/6.0;
    double delta_sq = 4.0/((double) N*N+2*N+1);
    // double d = 10000.0;
    int n = 0;
    
    int i,j,k = 0;
    // Main loop of jacobi
    while(n < max_iter)
    {
        // d = 0.0;
        #pragma omp parallel shared(old, newVol, f, N, h, delta_sq) private(i, j, k)
        {
        
        // Compute newVol 3d matrix
        #pragma omp for
        for(i = 1; i < N+1; i++){
            for(j = 1; j < N+1; j++){
                for(k = 1; k < N+1; k++){
                    newVol[i][j][k] = h*(old[i-1][j][k] + old[i+1][j][k] + old[i][j-1][k] + old[i][j+1][k] + old[i][j][k-1] + old[i][j][k+1] + delta_sq*f[i][j][k]);
                    //Norm
                    // d+=(old[i][j][k] - newVol[i][j][k])*(old[i][j][k] - newVol[i][j][k]);
                }
            }
        } 
        } // End of parallel region

        // Switch pointers
        temp = old;
        old = newVol;
        newVol = temp;

        // // Update convergence
        // d = norm(old, newVol, N);

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
    // double d = 10000.0;
    int n = 0;
    int i,j,k = 0;

    // Data transfer using map clause
    #pragma omp target enter data map(to: old[:N2][:N2][:N2]) map(to: f[:N2][:N2][:N2]) map(alloc: newVol[:N2][:N2][:N2])

    // Main loop of jacobi
    while(n < max_iter){
        // d = 0.0;
        #pragma omp target teams distribute parallel for
        for(i = 1; i < N+1; i++){
            for(j = 1; j < N+1; j++){
                for(k = 1; k < N+1; k++){
                    newVol[i][j][k] = h*(old[i-1][j][k] + old[i+1][j][k] + old[i][j-1][k] + old[i][j+1][k] + old[i][j][k-1] + old[i][j][k+1] + delta_sq*f[i][j][k]);
                    //Norm
                    // d+=(old[i][j][k] - newVol[i][j][k])*(old[i][j][k] - newVol[i][j][k]);
                }
            }
        }

        // Switch pointers
        temp = old;
        old = newVol;
        newVol = temp;

        // // Update convergence
        // d = norm(old, newVol, N);

        // Increment iteration counter
        n += 1;

    }

    // Data transfer to host
    #pragma omp target exit data map(from: old[:N2][:N2][:N2]) map(release: f[:N2][:N2][:N2]) map(release: newVol[:N2][:N2][:N2])
    
    return n;
}


int
jacobi_offload_memcopy(double ***old, double ***newVol, double ***f, int max_iter, int N, double tol){
   
    // Variables we will use
    double ***temp;
    double *temp2;
    double h = 1.0/6.0;
    double delta_sq = 4.0/((double) N*N+2*N+1);
    // double d = 10000.0;
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

    // Main loop of jacobi
    while(n < max_iter){
        // d = 0.0;
        #pragma omp target teams distribute parallel for
        for(i = 1; i < N+1; i++){
            for(j = 1; j < N+1; j++){
                for(k = 1; k < N+1; k++){
                    newVol[i][j][k] = h*(old[i-1][j][k] + old[i+1][j][k] + old[i][j-1][k] + old[i][j+1][k] + old[i][j][k-1] + old[i][j][k+1] + delta_sq*f[i][j][k]);
                    //Norm
                    // d+=(old[i][j][k] - newVol[i][j][k])*(old[i][j][k] - newVol[i][j][k]);
                }
            }
        }

        // Switch device pointers (the host does this)
        temp = old_dev;
        old = new_dev;
        new_dev = temp;

        temp2 = data;
        data = data_new;
        data_new = temp2;

        // // Update convergence
        // d = norm(old, newVol, N);

        // Increment iteration counter
        n += 1;

    }

    // Data transfer to host
    omp_target_memcpy(old[0][0], data, (N+2)*(N+2)*(N+2)*sizeof(double), 0, 0, omp_get_initial_device(), omp_get_default_device());
    
    return n;
}