#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>

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

    // I give up on this version I can only find examples online where they have double * pointers 
    // see ex https://github.com/colleeneb/openmp_offload_and_blas/blob/master/cublas/c/dgemm_cublas.c
    // 2
    void matmult_lib_offload(int m, int n, int k, double *A, double *B, double *C) {
        cublasHandle_t handle;
        cublasCreate(&handle);

        double alpha = 1.0;
        double beta = 0.0;
        int lda = m; // Assuming A is m x k in row-major
        int ldb = k; // Assuming B is k x n in row-major
        int ldc = m; // Assuming C is m x n in row-major

        // Ensure A, B, and C are allocated and initialized correctly before this point

        // Transfer data to the GPU
        #pragma omp target enter data map(to: A[0:m*k], B[0:k*n]) map(to: C[0:m*n])

        #pragma omp target data use_device_ptr(A, B, C)
        {
            cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, B, ldb, A, lda, &beta, C, ldc);
        }

        cublasDestroy(handle);

        // Transfer results back to the host
        #pragma omp target exit data map(from: C[0:m*n])
    }

    void matmult_lib_offload2(int m, int n, int k, double **A, double **B, double **C) {
        cublasHandle_t handle;
        cublasCreate(&handle);

        // device pointers
        double *d_A, *d_B, *d_C;
        cudaMalloc((void **)&d_A, m * k * sizeof(double));
        cudaMalloc((void **)&d_B, k * n * sizeof(double));
        cudaMalloc((void **)&d_C, m * n * sizeof(double));
        // move data to device
        cudaMemcpy(d_A, *A, m * k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, *B, k * n * sizeof(double), cudaMemcpyHostToDevice);

        double alpha = 1.0;
        double beta = 0.0;

        // why does it also work if we have the transposed leading dimensions?
        // cublas is fortran-order
        int lda = m;
        int ldb = k;
        int ldc = m;

        int cublas_error = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
        if (cublas_error != CUBLAS_STATUS_SUCCESS) {
            printf("Error in cublasDgemm!\n");
        }

        cudaMemcpy(*C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
}

    void matmult_blk_offload(int m, int n, int k, double **A, double **B, double **C) {
    #define BLK 4
    zeroC(m, n, C);
    double start_time, end_time, data_transfer_time;

    start_time = omp_get_wtime();
    #pragma omp target enter data map(to: A[0:m][0:k], B[0:k][0:n]) map(to: C[0:m][0:n])

    #pragma omp target teams loop num_teams(m) thread_limit(16) \
    map(tofrom: C[0:m][0:n]) map(to: A[0:m][0:k], B[0:k][0:n]) collapse(2)
    for (int i = 0; i < m; i += BLK) { 
        for (int j = 0; j < n; ++j) {
            if (i + BLK - 1 < m) {
                double sum[BLK] = {0}; 
                for (int l = 0; l < k; l++) { 
                    for (int ii = 0; ii < BLK; ii++) { 
                        sum[ii] += A[i + ii][l] * B[l][j]; 
                    } 
                }

                for (int ii = 0; ii < BLK; ii++) { 
                    C[i + ii][j] = sum[ii]; 
                }
            } else {
                // Remainder part
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

    #pragma omp target exit data map(from: C[0:m][0:n])
    end_time = omp_get_wtime();
    data_transfer_time = end_time - start_time;

    printf("Data Transfer Time in matmult_blk_offload: %f seconds\n", data_transfer_time);
    }

    void matmult_asy_offload(int m, int n, int k, double **A, double **B, double **C) {
        zeroC(m, n, C);

        #define SLAPS 2

        if (m % SLAPS != 0) {
            printf("ERROR; will not give correct results, m must be divisible by SLAPS, but was m=%d, SLAPS=%d\n", m, SLAPS);
            return;
        }

        double start_time, end_time, total_data_transfer_time = 0.0;
        start_time = omp_get_wtime();

        #pragma omp target enter data map(to: A[0:m][0:k], B[0:k][0:n], C[0:m][0:n])
        end_time = omp_get_wtime();
        total_data_transfer_time += (end_time - start_time);

        #pragma omp parallel for
        for (int s = 0; s < SLAPS; ++s) {
            int length = m / SLAPS;
            int start = s * length;

            start_time = omp_get_wtime();
            #pragma target data update device(A[start:length][0:k]) depend(out: A) nowait
            
            end_time = omp_get_wtime();
            total_data_transfer_time += (end_time - start_time);

            #pragma omp target teams distribute parallel for map(to: A[start:length][0:k]) num_teams(m) thread_limit(16) depend(in: A) depend(out: C) collapse(2)
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
            #pragma omp target update from(C[start:length][:n]) depend(in:C) nowait
        }
        // is taskwait this necessary
        #pragma omp taskwait 
        start_time = omp_get_wtime();
        #pragma omp target exit data map(from: C[0:m][0:n]) map(delete: A[0:m][0:k], B[0:k][0:n])
        end_time = omp_get_wtime();
        total_data_transfer_time += (end_time - start_time);
        // Output or use the total_data_transfer_time for analysis
        printf("Total Data Transfer Time: %f seconds\n", total_data_transfer_time);
    }

    void matmult_mkn_omp(int m, int n, int k, double **A, double **B, double **C) {
        zeroC(m, n, C);
        #pragma omp parallel shared(A, B, C)
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

    void matmult_mkn_offload(int m, int n, int k, double **A, double **B, double **C) {
    zeroC(m, n, C);
    double start_time, end_time, data_transfer_time;

    start_time = omp_get_wtime();
    #pragma omp target enter data map(to: A[0:m][0:k], B[0:k][0:n]) map(to: C[0:m][0:n])

    #pragma omp target teams distribute parallel for map(to: A[0:m][0:k], B[0:k][0:n]) map(tofrom: C[0:m][0:n]) num_teams(m) thread_limit(16)
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }

    #pragma omp target exit data map(from: C[0:m][0:n])
    end_time = omp_get_wtime();
    data_transfer_time = end_time - start_time;

    printf("Data Transfer Time in matmult_mkn_offload: %f seconds\n", data_transfer_time);
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
    double start_time, end_time, data_transfer_time;

    start_time = omp_get_wtime();
    #pragma omp target enter data map(to: A[0:m][0:k], B[0:k][0:n]) map(to: C[0:m][0:n])

    #pragma omp target teams distribute parallel for map(to: A[0:m][0:k], B[0:k][0:n]) map(tofrom: C[0:m][0:n]) \
    num_teams(m) thread_limit(16)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int l = 0; l < k; l++) {
                sum += A[i][l] * B[l][j];
            }
            C[i][j] = sum;
        }
    }

    #pragma omp target exit data map(from: C[0:m][0:n])
    end_time = omp_get_wtime();
    data_transfer_time = end_time - start_time;

    printf("Data Transfer Time in matmult_mnk_offload: %f seconds\n", data_transfer_time);
    }

       
} // end of extern "C"

// KEEP EVERYTHING YOU WANT TO RUN IN THE BRACKETS ABOVE


