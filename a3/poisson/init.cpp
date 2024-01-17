void init_jacobi(double ***old, double ***newVol, double ***f, int N2, double T0){
    // Set boundary conditions
    #pragma omp parallel
    {
    #pragma omp for 
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            for(int k = 0; k < N2; k++){
                old[i][j][k] = T0;
                newVol[i][j][k] = T0;
            }
        }
    }
    #pragma omp for 
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            old[0][i][j] = 20.0;
            old[N2-1][i][j] = 20.0;
            old[i][0][j] = 0.0;
            old[i][N2-1][j] = 20.0;
            old[i][j][0] = 20.0;
            old[i][j][N2-1] = 20.0;
            newVol[0][i][j] = 20.0;
            newVol[N2-1][i][j] = 20.0;
            newVol[i][0][j] = 0.0;
            newVol[i][N2-1][j] = 20.0;
            newVol[i][j][0] = 20.0;
            newVol[i][j][N2-1] = 20.0;
        }
    }
    #pragma omp for 
    // Set source function (radiator)
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            for(int k = 0; k < N2; k++){
                f[i][j][k] = 0.0;
            }
        }
    }
    #pragma omp for 
    for(int i = (N2-1)/6; i <= (N2-1)/2; i++){
        for(int j = 0; j <= (N2-1)/4; j++){
            for(int k = 0; k <= (5*(N2-1))/16; k++){
                f[i][j][k] = 200.0;
            }
        }
    }
    } // END PARALLEL
    return;
}

void init_seidel(double ***u, double ***f, int N2, double T0){
    // Set boundary conditions
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            for(int k = 0; k < N2; k++){
                u[i][j][k] = T0;
            }
        }
    }
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            u[0][i][j] = 20.0;
            u[N2-1][i][j] = 20.0;
            u[i][0][j] = 0.0;
            u[i][N2-1][j] = 20.0;
            u[i][j][0] = 20.0;
            u[i][j][N2-1] = 20.0;
        }
    }
    // Set source function (radiator)
    for(int i = 0; i < N2; i++){
        for(int j = 0; j < N2; j++){
            for(int k = 0; k < N2; k++){
                f[i][j][k] = 0.0;
            }
        }
    }
    for(int i = (N2-1)/6; i <= (N2-1)/2; i++){
        for(int j = 0; j <= (N2-1)/4; j++){
            for(int k = 0; k <= (5*(N2-1))/16; k++){
                f[i][j][k] = 200.0;
            }
        }
    }
    return;
}