/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>


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
        // #pragma omp for collapse(2)
        // #pragma omp for schedule(static) 
        // #pragma omp for schedule(dynamic, 10)
        // #pragma omp for schedule(guided) 
        // #pragma omp for schedule(runtime)
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
    double ***temp;
    double h = 1.0/6.0;
    double delta_sq = 4.0/((double) N*N+2*N+1);
    // double d = 10000.0;
    int n = 0;
    int i,j,k = 0;

    // Data transfer using map clause
    #pragma omp target enter data map(to: old[:N][:N][:N]) map(to: f[:N][:N][:N]) map(alloc: newVol[:N][:N][:N])

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
    #pragma omp target exit data map(from: old[:N][:N][:N]) map(release: f[:N][:N][:N]) map(release: newVol[:N][:N][:N])
    
    return n;
}