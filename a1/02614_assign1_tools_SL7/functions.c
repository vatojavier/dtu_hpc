#include <stdio.h>
#include <cblas.h>

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

void matmult_nat(int m, int n, int k, double **A, double **B, double **C) {
    zeroC(m, n, C);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++){
            for (int l = 0; l < k; l++) {
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

void matmult_nmk(int m, int n, int k, double **A, double **B, double **C) {
    zeroC(m, n, C);
    for (int j = 0; j < n; j++){
        for (int i = 0; i < m; i++) {
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

void matmult_kmn(int m, int n, int k, double **A, double **B, double **C) {
    zeroC(m, n, C);
    for (int l = 0; l < k; l++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++){
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

void matmult_knm(int m, int n, int k, double **A, double **B, double **C) {
    zeroC(m, n, C);
    for (int l = 0; l < k; l++) {
        for (int j = 0; j < n; j++){
            for (int i = 0; i < m; i++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}


