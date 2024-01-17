#include <stdio.h>
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

// block matrix multiplication
void matmult_blk(int m, int n, int k, double **A, double **B, double **C, int bs) {
    zeroC(m, n, C);
    for (int bm = 0; bm<m; bm+=bs){
        for (int bn = 0; bn<n; bn+=bs){
            for (int bk = 0; bk<k; bk+=bs){
                for (int i = 0; i < MIN(m-bm,bs); i++){
                    for (int l = 0; l < MIN(k-bk,bs); l++){
                        for (int j = 0; j < MIN(n-bn,bs); j++) {
                            C[i+bm][j+bn] += A[i+bm][l+bk] * B[l+bk][j+bn];
                        }
                    }
                }
            }
        }
    }
}


void matmult_blk2(int m, int n, int k, double **A, double **B, double **C, int bs) {
    zeroC(m, n, C);
    int nbm = m/bs;
    int nbn = n/bs;
    // main whole blocks 
    for (int bm = 0; bm<nbm; bm++){
        for (int bn = 0; bn<nbn; bn++){
            for (int i = 0; i < bs; i++){
                for (int j = 0; j < bs; j++){
                    for (int l = 0; l < k; l++) {
                        C[i+(bm*bs)][j+bn*bs] += A[i+bm*bs][l] * B[l][j+bn*bs];
                    }
                }
            }   
        }
    }

    // now there are some remaining blocks left, which are handled separately
    // bottom left
    for (int bn = 0; bn<nbn; bn++){
        for (int i = nbm*bs; i < m; i++){
            for (int j = 0; j < bs; j++){
                for (int l = 0; l < k; l++) {
                    C[i][j+bn*bs] += A[i][l] * B[l][j+bn*bs];
                }
            }
        }   
    }
    // right block
    for (int bm = 0; bm<nbm; bm++){
        for (int i = 0; i < bs; i++){
            for (int j = nbn*bs; j < n; j++){
                for (int l = 0; l < k; l++) {
                    C[i+bm*bs][j] += A[i+bm*bs][l] * B[l][j];
                }
            }
        }   
    }
    // lower right block
    for (int i = nbm*bs; i < m; i++){
        for (int j = nbn*bs; j < n; j++){
            for (int l = 0; l < k; l++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }  
}
