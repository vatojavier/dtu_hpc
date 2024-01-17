#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>

extern "C" {
    #include <cblas.h>
    #define MIN(a,b) (((a)<(b))?(a):(b))

    void matmult_lib(int m, int n, int k, double **A, double **B, double **C) {
        double alpha = 1.0;
        double beta = 0.0;
        int lda = k;
        int ldb = n;
        int ldc = n;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A[0], lda, B[0], ldb, beta, C[0], ldc);
        }

    void zeroC(int m, int n, double **C) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;
            }
        }
    }

    void matmult_mkn(int m, int n, int k, double **A, double **B, double **C) {
        zeroC(m, n, C);
        #pragma omp parallel for 
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int l = 0; l < k; l++) {
                for (int j = 0; j < n; j++){
                    sum += A[i][l] * B[l][j];
                }
                C[i][l] = sum;
            }
        }
    }

    void matmult_lib_offload(int m, int n, int k, double **A, double **B, double **C){
        cublasHandle_t handle; 
        cublasCreate(&handle);

        double alpha = 1.0;
        double beta = 0.0;
        int lda = k;
        int ldb = n;
        int ldc = n;

        int cublas_error = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A[0], lda, B[0], ldb, &beta, C[0], ldc);

        // #pragma omp target enter data map(to: A[0:m][0:n], B[0:m][0:n], C[0:m][0:n])
        // {
        //     #pragma omp target data use_device_ptr(A, B, C)
        //     {
            
        //     }
        // }
        // #pragma omp target exit data map(from: C[0:m][0:n])

        cublasDestroy(handle);
        }

    void matmult_blk_offload(int m, int n, int k, double **A, double **B, double **C) {
        // Seems like the block size should be lower than 4, otherwise we cannot completely unroll the loops
        #define BLK 4

        zeroC(m, n, C);

        #pragma omp target teams loop num_teams(m/16) thread_limit(16) \
        map(tofrom: C[0:m][0:n]) map(to: A[0:m][0:k], B[0:k][0:n]) collapse(2)
        for (int i = 0; i < m; i += BLK) { 
            for (int j = 0; j < n; ++j) { 
                if (i + BLK - 1 < m) { 
                    double sum[BLK] = {0}; 
                    // Do BLK elements of C here
                    for (int l = 0; l < k; l++) { 
                        for (int ii = 0; ii < BLK; ii++) { 
                            sum[ii] += A[i + ii][l] * B[l][j]; 
                        } 
                    }

                    for (int ii = 0; ii < BLK; ii++) { 
                        C[i + ii][j] = sum[ii]; 
                    }
                    
                } else { 
                    // Do the remainder part here 
                    double sum[BLK] = {0}; 
                    for (int l = 0; l < k; l++) { 
                        for (int ii = i; ii < m; ii++) { 
                            sum[ii-i] += A[ii][l] * B[l][j]; 
                        }
                    }
                    for (int ii = i; ii < m; ii++) { 
                        C[ii][j] += sum[ii-i]; 
                    }
                }
            }
        }
    }

        void matmult_mkn(int m, int n, int k, double **A, double **B, double **C) {
            zeroC(m, n, C);
            #pragma omp parallel shared(A, B, C) num_threads(24)
            {
            #pragma omp for 
            for (int i = 0; i < m; i++) {
                for (int l = 0; l < k; l++) {
                    for (int j = 0; j < n; j++){
                        C[i][j] += A[i][l] * B[l][j];
                    }
                }
            }

            } // end of parallel region
        }

        void matmult_mkn_offload(int m, int n, int k, double **A, double **B, double **C) 
        {
            zeroC(m, n, C);
            int num_teams = 4;
            int num_threads = 64;
            #pragma omp target teams distribute parallel for map(to: A[0:m][0:k], B[0:k][0:n]) map(tofrom: C[0:m][0:n]) num_teams(num_teams) thread_limit(num_threads)
            for (int i = 0; i < m; i++) {
            for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
                }
            }
        }
    }

    void matmult_mnk(int m, int n, int k, double **A, double **B, double **C) {
        zeroC(m, n, C);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++){
                for (int l = 0; l < k; l++) {
                    C[i][j] += A[i][l] * B[l][j];
                }
            }
        }
    }

    void matmult_mnk_offload(int m, int n, int k, double **A, double **B, double **C) {
        zeroC(m, n, C);

        // int num_teams = 4; // 4 to 500: 4, 16, 64, 200, 500
        // int num_threads = 256; // 64 to 512: 64, 128, 256, 512

        int num_teams = omp_get_num_teams(); 
        int num_threads = omp_get_num_threads();

        #pragma omp target teams distribute parallel for map(to: A[0:m][0:k], B[0:k][0:n]) map(from: C[0:m][0:n]) \
        num_teams(num_teams) thread_limit(num_threads)
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += A[i][l] * B[l][j];
                }
                C[i][j] = sum;
            }
            // printf("Executed with %d teams and %d threads\n", omp_get_num_teams(), omp_get_num_threads());
        }

        // #pragma omp single
        //     {
        // printf("Executed with %d teams and %d threads\n", num_teams, num_threads);
        //     }
    }

       
} // end of extern "C"

// KEEP EVERYTHING YOU WANT TO RUN IN THE BRACKETS ABOVE


