#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h> // For strcmp

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

    void matmult_lib_offload(int m, int n, int k, double **A, double **B, double **C) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // device pointers
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, m * k * sizeof(double));
    cudaMalloc((void **)&d_B, k * n * sizeof(double));
    cudaMalloc((void **)&d_C, m * n * sizeof(double));
    
    double start_time, end_time;
    double data_in_time = 0.0, computation_time = 0.0, data_out_time = 0.0;

    // Data Transfer In
    start_time = omp_get_wtime();
    cudaMemcpy(d_A, *A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, *B, k * n * sizeof(double), cudaMemcpyHostToDevice);
    end_time = omp_get_wtime();
    data_in_time = end_time - start_time;

    double alpha = 1.0;
    double beta = 0.0;

    // Computation
    start_time = omp_get_wtime();
    int lda = m;
    int ldb = k;
    int ldc = m;
    int cublas_error = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
    if (cublas_error != CUBLAS_STATUS_SUCCESS) {
        printf("Error in cublasDgemm!\n");
    }
    end_time = omp_get_wtime();
    computation_time = end_time - start_time;

    // Data Transfer Out
    start_time = omp_get_wtime();
    cudaMemcpy(*C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    end_time = omp_get_wtime();
    data_out_time = end_time - start_time;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    // Output or use the timing information
    char *printFlag = getenv("PRINT_DATA_TRANSFER_TIME");
    if (printFlag != NULL && strcmp(printFlag, "1") == 0) {
        printf("Data Transfer In Time: %f seconds\n", data_in_time);
        printf("Computation Time: %f seconds\n", computation_time);
        printf("Data Transfer Out Time: %f seconds\n", data_out_time);
    }
}

    void matmult_blk_offload(int m, int n, int k, double **A, double **B, double **C) {
    #define BLK 8
    zeroC(m, n, C);
    double start_time, end_time, data_in_time, computation_time, data_out_time;

    // Data transfer into the device
    start_time = omp_get_wtime();
    #pragma omp target enter data map(to: A[0:m][0:k], B[0:k][0:n]) map(to: C[0:m][0:n])
    end_time = omp_get_wtime();
    data_in_time = end_time - start_time;

    // Computation
    start_time = omp_get_wtime();
    #pragma omp target teams loop \
    map(to: C[0:m][0:n]) map(to: A[0:m][0:k], B[0:k][0:n]) collapse(2)
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
                // Do the remainder part here 
                        for (int ii = 0; ii < (m-i); ii++) { 
                            double sum = 0.0; 
                            for (int l = 0; l < k; l++) {
                                sum += A[i+ii][l] * B[l][j]; 
                            }
                            C[i+ii][j] = sum; 
                    }
            }
        }
    }
    end_time = omp_get_wtime();
    computation_time = end_time - start_time;

    // Data transfer out of the device
    start_time = omp_get_wtime();
    #pragma omp target exit data map(from: C[0:m][0:n]) map(release: A[0:m][0:k], B[0:k][0:n])
    end_time = omp_get_wtime();
    data_out_time = end_time - start_time;

    // Output timing information
    // char expected with the introduction of different time measurements
    char *printFlag = getenv("PRINT_DATA_TRANSFER_TIME");
    if (printFlag != NULL && strcmp(printFlag, "1") == 0) {
        printf("Data Transfer In Time: %f seconds\n", data_in_time);
        printf("Computation Time: %f seconds\n", computation_time);
        printf("Data Transfer Out Time: %f seconds\n", data_out_time);
    }
    }

    void matmult_asy_offload(int m, int n, int k, double **A, double **B, double **C) {
        zeroC(m, n, C);

        #define SLAPS 4

        if (m % SLAPS != 0) {
            printf("ERROR; will not give correct results, m must be divisible by SLAPS, but was m=%d, SLAPS=%d\n", m, SLAPS);
            return;
        }

        double start_time, end_time, data_in_time, computation_time = 0.0, data_out_time, total_data_transfer_time = 0.0;
        
        // Data transfer into the device
        start_time = omp_get_wtime();
        #pragma omp target enter data map(alloc: A[0:m][0:k], C[0:m][0:n]) 
        #pragma omp target enter data map(to: B[0:k][0:n])
        end_time = omp_get_wtime();
        data_in_time = end_time - start_time;

        // Parallel loop for each slab
        double start_compute_time = omp_get_wtime();  // Start timing computation
        #pragma omp parallel for reduction(+:computation_time)
        for (int s = 0; s < SLAPS; ++s) {
            int length = m / SLAPS;
            int start = s * length;

            #pragma omp target update to(A[start:length][0:k], C[start:length][0:n]) nowait
            #pragma omp target teams distribute parallel for \
            num_teams(length) thread_limit(32) collapse(2) nowait\
            depend(in: A[start:length][0:k], B[0:k][0:n]) \
            depend(out: C[start:length][0:n])
            
            for (int i = start; i < start+length; i += BLK) { 
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
                        for (int ii = 0; ii < (m-i); ii++) { 
                            double sum = 0.0; 
                            for (int l = 0; l < k; l++) {
                                sum += A[i+ii][l] * B[l][j]; 
                            }
                            C[i+ii][j] = sum; 
                        }
                    }
                }
            } 
        #pragma omp target update from(C[start:length][0:n]) nowait
        }
        #pragma omp taskwait
        
        double end_compute_time = omp_get_wtime();  // End timing computation
        computation_time = end_compute_time - start_compute_time;

        start_time = omp_get_wtime();
        #pragma omp target exit data map(from: C[0:m][0:n]) map(release: A[0:m][0:k], B[0:k][0:n])
        end_time = omp_get_wtime();
        data_out_time = end_time - start_time;

        // Output timing information
        total_data_transfer_time += data_in_time + data_out_time;
        char *printFlag = getenv("PRINT_DATA_TRANSFER_TIME");
        if (printFlag != NULL && strcmp(printFlag, "1") == 0) {
            printf("Data Transfer In Time: %f seconds\n", data_in_time);
            printf("Computation Time: %f seconds\n", computation_time);
            printf("Data Transfer Out Time: %f seconds\n", data_out_time);
            // printf("Total Data Transfer Time (including async): %f seconds\n", total_data_transfer_time);
        }
    }

    void matmult_mkn_offload(int m, int n, int k, double **A, double **B, double **C) {
    zeroC(m, n, C);
    double start_time, end_time, data_in_time, computation_time, data_out_time;

    // Data transfer to the GPU
    start_time = omp_get_wtime();
    #pragma omp target enter data map(to: A[0:m][0:k], B[0:k][0:n]) map(to: C[0:m][0:n])
    end_time = omp_get_wtime();
    data_in_time = end_time - start_time;

    // Computation
    start_time = omp_get_wtime();
    #pragma omp target teams distribute parallel for map(to: A[0:m][0:k], B[0:k][0:n]) map(tofrom: C[0:m][0:n]) num_teams(m) thread_limit(16)
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
    end_time = omp_get_wtime();
    computation_time = end_time - start_time;

    // Data transfer from the GPU
    start_time = omp_get_wtime();
    #pragma omp target exit data map(from: C[0:m][0:n]) map(release: A[0:m][0:k], B[0:k][0:n])
    end_time = omp_get_wtime();
    data_out_time = end_time - start_time;

    // Check if the environment variable is set
    char *printFlag = getenv("PRINT_DATA_TRANSFER_TIME");
    if (printFlag != NULL && strcmp(printFlag, "1") == 0) {
        printf("Data Transfer In Time: %f seconds\n", data_in_time);
        printf("Computation Time: %f seconds\n", computation_time);
        printf("Data Transfer Out Time: %f seconds\n", data_out_time);
    }
    }


    void matmult_mnk_offload(int m, int n, int k, double **A, double **B, double **C) {
    zeroC(m, n, C);
    double start_time, end_time, data_in_time, computation_time, data_out_time;

    // Data transfer to the GPU
    start_time = omp_get_wtime();
    #pragma omp target enter data map(to: A[0:m][0:k], B[0:k][0:n]) map(to: C[0:m][0:n])
    end_time = omp_get_wtime();
    data_in_time = end_time - start_time;

    // Computation
    start_time = omp_get_wtime();
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
    end_time = omp_get_wtime();
    computation_time = end_time - start_time;

    // Data transfer from the GPU
    start_time = omp_get_wtime();
    #pragma omp target exit data map(from: C[0:m][0:n]) map(release: A[0:m][0:k], B[0:k][0:n])
    end_time = omp_get_wtime();
    data_out_time = end_time - start_time;

    // Check if the environment variable is set
    char *printFlag = getenv("PRINT_DATA_TRANSFER_TIME");
    if (printFlag != NULL && strcmp(printFlag, "1") == 0) {
        printf("Data Transfer In Time: %f seconds\n", data_in_time);
        printf("Computation Time: %f seconds\n", computation_time);
        printf("Data Transfer Out Time: %f seconds\n", data_out_time);
    }
}

       
} // end of extern "C"

// KEEP EVERYTHING YOU WANT TO RUN IN THE BRACKETS ABOVE


