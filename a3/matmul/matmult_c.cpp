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

    void matmult_mkn_omp(int m, int n, int k, double **A, double **B, double **C) {
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


void matmult_mkn(int m, int n, int k, double **A, double **B, double **C) {
    zeroC(m, n, C);
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++){
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

void matmult_nkm(int m, int n, int k, double **A, double **B, double **C) {
    zeroC(m, n, C);
    for (int j = 0; j < n; j++){
        for (int l = 0; l < k; l++) {
            for (int i = 0; i < m; i++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}