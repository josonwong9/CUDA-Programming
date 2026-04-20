#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void add(const double *x, const double *y, double *z, int N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        z[n] = x[n] + y[n];
    }
}

void check(const double *z, int N)
{
    for (int n = 0; n < N; ++n) {
        if (fabs(z[n] - c) > EPSILON) {
            printf("Has errors at index %d: %.17f\n", n, z[n]);
            return;
        }
    }
    printf("No errors\n");
}

int main(void)
{
    const int N = 100000000;
    const size_t M = sizeof(double) * (size_t)N;

    double *h_x = (double*)malloc(M);
    double *h_y = (double*)malloc(M);
    double *h_z = (double*)malloc(M);

    if (!h_x || !h_y || !h_z) {
        printf("Host malloc failed\n");
        return 1;
    }

    for (int n = 0; n < N; ++n) {
        h_x[n] = a;
        h_y[n] = b;
    }

    double *d_x = NULL, *d_y = NULL, *d_z = NULL;

    cudaError_t err;
    err = cudaMalloc((void**)&d_x, M);
    if (err != cudaSuccess) { printf("cudaMalloc d_x failed\n"); return 1; }

    err = cudaMalloc((void**)&d_y, M);
    if (err != cudaSuccess) { printf("cudaMalloc d_y failed\n"); return 1; }

    err = cudaMalloc((void**)&d_z, M);
    if (err != cudaSuccess) { printf("cudaMalloc d_z failed\n"); return 1; }

    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = (N + block_size - 1) / block_size;

    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}
