This file describes additional specifications and tools for the 3. assignment in 
the DTU course 02614.

Please re-read the README for Assignment 1 for more details.

New files:
----------

Makefile.nvc++   - Makefile template for OpenMP offload code compiled with nvc++

a) matrices represented by double pointers:

matmult_c.nvc++  - driver for libraries built with nvc++, linked with
                   multithreaded CBLAS from Intel MKL and CUBLAS.

b) matrices represented by single pointers, i.e. as a vector:

matmult_f.nvc++  - driver for libraries built with nvc++, linked with
                   multithreaded CBLAS from Intel MKL and CUBLAS.

Changes and additions
---------------------

The drivers still take the same command line arguments:

matmult_f.nvcc type m n k [bs]

where m, n, k are the parameters defining the matrix sizes, bs is the
optional blocksize for the block version, and type can be one of:

nat           - the native/naïve version
lib           - the library version (note that CBLAS is now linked to a
                multithreaded library that supports OMP_NUM_THREADS)
mkn_omp       - the OpenMP mkn version
mkn_offload   - the OpenMP offload mkn version
mnk_offload   - the OpenMP offload mnk version
blk_offload   - the OpenMP offload blocked version (takes bs as extra argument)
asy_offload   - the OpenMP offload asynchronous version
lib_offload   - the OpenMP offload CUBLAS library version
exp_offload   - the OpenMP offload experimental version (if needed)

as well as blk, mnk, nmk, ... (the permutations).

Changes:
 * The timer in the drivers has been changed to a wall clock timer, which 
   makes the MFLOPS calculation very sensitive to other user activity. 
 * The drivers are now linked to a multithreaded CBLAS library, which launches 
   more threads to do the matrix multiplication on the CPU (if estimated to be 
   worth while by the MKL implementation). It also supports the user setting
   OMP_NUM_THREADS to a fixed number of threads.
 * When using the drivers for type 'offload', the driver will wake up the
   GPU (takes approx. 0.3 sec) and will delete all storage pointers on the
   device corresponding to host pointers A, B, and C.

Note that MATMULT_RESULTS, MATMULT_COMPARE, MFLOPS_MIN_T, and MFLOPS_MAX_IT are
still supported - please apply them appropriately to avoid excessive use of
resources during the development, testing, and benchmarking of your code!

With the help of this driver program, you should be able to run all matrix 
multiplication experiments needed for Assignment 3 in the same manner as for
Assignment 1. 
