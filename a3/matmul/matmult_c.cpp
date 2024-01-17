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
            // #pragma omp target
            // #pragma omp parallel shared(C) num_threads(24)
            // {
            // #pragma omp for
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    C[i][j] = 0.0;
                }
            }

            // } // end of parallel region
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

        void matmult_mkn_offload(int m, int n, int k, double **A, double **B, double **C) {
            zeroC(m, n, C);
            #pragma omp target teams num_teams(200) //distribute parallel for map(to:A[0:m][0:n], B[0:m][0:n],C[0:m][0:n]) map(from:C[0:m][0:n])
            {
            for (int i = 0; i < m; i++) {
                for (int l = 0; l < k; l++) {
                    for (int j = 0; j < n; j++) {
                        C[i][j] += A[i][l] * B[l][j];
                    }
                }
            }
            } // end of parallel region
        }

        
}

// KEEP EVERYTHING YOU WANT TO RUN IN THE BRACKETS ABOVE


