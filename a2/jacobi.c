/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <math.h>


// // Frobenius norm
// double norm(double ***old, double ***new, int N){
// double res = 0.0;
// int i,j,k = 0;
// #pragma omp parallel shared(old, new, N) private(i, j, k) reduction(+:res)
// {
// #pragma omp for
// for(int i = 1; i < N+1; i++){
//     for(int j = 1; j < N+1; j++){
//         for(int k = 1; k < N+1; k++){
//             res += (old[i][j][k] - new[i][j][k])*(old[i][j][k] - new[i][j][k]);
//         }
//     }
// }
// } // End of parallel region
// return res;
// }

// Frobenius norm
// double norm(double ***old, double ***new, int N)
// {
// double res = 0.0;
// int i,j,k = 0;
// #pragma omp parallel for reduction(+:res)
// for(int i = 1; i < N+1; i++){
//     for(int j = 1; j < N+1; j++){
//         for(int k = 1; k < N+1; k++){
//             res += (old[i][j][k] - new[i][j][k])*(old[i][j][k] - new[i][j][k]);
//         }
//     }
// }
// return res;
// }


int
jacobi(double ***old, double ***new, double ***f, int max_iter, int N, double tol) {
    // Variables we will use
    double ***temp;
    double h = 1.0/6.0;
    double delta_sq = 4.0/((double) N*N+2*N+1);
    double d = INFINITY;
    int n = 0;

    // Main loop of jacobi
    while(d > tol && n < max_iter){
        d = 0.0;
        // Compute new 3d matrix
        for(int i = 1; i < N+1; i++){
            for(int j = 1; j < N+1; j++){
                for(int k = 1; k < N+1; k++){
                    new[i][j][k] = h*(old[i-1][j][k] + old[i+1][j][k] + old[i][j-1][k] + old[i][j+1][k] + old[i][j][k-1] + old[i][j][k+1] + delta_sq*f[i][j][k]);
                    //Norm
                    d+=(old[i][j][k] - new[i][j][k])*(old[i][j][k] - new[i][j][k]);
                }
            }
        }

        // Switch pointers
        temp = old;
        old = new;
        new = temp;

        // Increment iteration counter
        n += 1;

    }
    return n;
}


int
jacobi_baseline(double ***old, double ***new, double ***f, int max_iter, int N, double tol) {
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
        #pragma omp parallel shared(old, new, f, N, h, delta_sq) private(i, j, k) reduction(+:d)
        {
        // Compute new 3d matrix
            #pragma omp for
            for(i = 1; i < N+1; i++){
                for(j = 1; j < N+1; j++){
                    for(k = 1; k < N+1; k++){
                        new[i][j][k] = h*(old[i-1][j][k] + old[i+1][j][k] + old[i][j-1][k] + old[i][j+1][k] + old[i][j][k-1] + old[i][j][k+1] + delta_sq*f[i][j][k]);
                        //Norm
                        d+=(old[i][j][k] - new[i][j][k])*(old[i][j][k] - new[i][j][k]);

                    }
                }
            }
        } // end of parallel region
    

        // Switch pointers
        temp = old;
        old = new;
        new = temp;

        // Increment iteration counter
        n += 1;

    }
    return n;
}

int
jacobi_improved(double ***old, double ***new, double ***f, int max_iter, int N, double tol) {
    // Variables we will use
    double ***temp;
    double h = 1.0/6.0;
    double delta_sq = 4.0/((double) N*N+2*N+1);
    double d = 10000.0;
    int n = 0;
    
    int i,j,k = 0;
    // Main loop of jacobi
    while(d > tol && n < max_iter)
    {
        d = 0.0;
        #pragma omp parallel shared(old, new, f, N, h, delta_sq) private(i, j, k) reduction(+:d) 
        {
        
        // Compute new 3d matrix
        #pragma omp for 
        // #pragma omp for collapse(2)
        // #pragma omp for schedule(static) 
        // #pragma omp for schedule(dynamic, 10)
        // #pragma omp for schedule(guided) 
        // #pragma omp for schedule(runtime)
        for(i = 1; i < N+1; i++){
            for(j = 1; j < N+1; j++){
                for(k = 1; k < N+1; k++){
                    new[i][j][k] = h*(old[i-1][j][k] + old[i+1][j][k] + old[i][j-1][k] + old[i][j+1][k] + old[i][j][k-1] + old[i][j][k+1] + delta_sq*f[i][j][k]);
                    //Norm
                    d+=(old[i][j][k] - new[i][j][k])*(old[i][j][k] - new[i][j][k]);
                }
            }
        } 
        } // End of parallel region

        // Switch pointers
        temp = old;
        old = new;
        new = temp;

        // // Update convergence
        // d = norm(old, new, N);

        // Increment iteration counter
        n += 1;

    }
    
    return n;
}