/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <math.h>
#include <stdio.h>

int gauss_seidel_seq(double ***u, double ***f, int max_iter, int N, double tol)
{
    // Variables we will use
    double h = 1.0 / 6.0;
    double delta_sq = 4.0 / ((double)N * N + 2 * N + 1);
    double d = INFINITY;
    int n = 0;

    // Main loop of Gauss-Seidel
    while (d > tol && n < max_iter)
    {
        double d_max = 0.0; // For tracking maximum change in this iteration
        double sum_of_squares = 0.0;

        // Compute updated values directly in the old matrix
        for (int i = 1; i < N + 1; i++)
        {
            for (int j = 1; j < N + 1; j++)
            {
                for (int k = 1; k < N + 1; k++)
                {
                    // double oldValue = u[i][j][k];

                    double old_value = u[i][j][k];
                    u[i][j][k] = h * (u[i - 1][j][k] +      // Value from the "west"
                                      u[i + 1][j][k] +      // Value from the "east"
                                      u[i][j - 1][k] +      // Value from the "south"
                                      u[i][j + 1][k] +      // Value from the "north"
                                      u[i][j][k - 1] +      // Value from the "bottom"
                                      u[i][j][k + 1] +      // Value from the "top"
                                      delta_sq * f[i][j][k] // Source term
                                     );

                    // Accumulate the squares of the changes
                    double change = u[i][j][k] - old_value;
                    // Compute the squared Frobenius norm of the changes
                    d += change * change;
                }
            }
        }


        // Increment iteration counter
        n += 1;
    }

    return n;
}


int gauss_seidel_omp(double ***u, double ***f, int max_iter, int N, double tol){
     // Variables we will use
    double h = 1.0 / 6.0;
    double delta_sq = 4.0 / ((double)N * N + 2 * N + 1);
    double d = INFINITY;
    int n = 0;

    printf("Gauss in parallel\n");

    // Main loop of Gauss-Seidel
    while (d > tol && n < max_iter)
    {
        double d_max = 0.0; // For tracking maximum change in this iteration
        double sum_of_squares = 0.0;

        #pragma omp parallel for reduction(+:sum_of_squares)
        for (int i = 1; i < N + 1; i++)
        {
            for (int j = 1; j < N + 1; j++)
            {
                for (int k = 1; k < N + 1; k++)
                {
                    // double oldValue = u[i][j][k];

                    double old_value = u[i][j][k];
                    u[i][j][k] = h * (u[i - 1][j][k] +      // Value from the "west"
                                      u[i + 1][j][k] +      // Value from the "east"
                                      u[i][j - 1][k] +      // Value from the "south"
                                      u[i][j + 1][k] +      // Value from the "north"
                                      u[i][j][k - 1] +      // Value from the "bottom"
                                      u[i][j][k + 1] +      // Value from the "top"
                                      delta_sq * f[i][j][k] // Source term
                                     );

                    // Accumulate the squares of the changes
                    double change = u[i][j][k] - old_value;
                    sum_of_squares += change * change;
                }
            }
        }

        // Compute the Frobenius norm of the changes
        d = sqrt(sum_of_squares);

        // Increment iteration counter
        n += 1;
    }

    return n;
}