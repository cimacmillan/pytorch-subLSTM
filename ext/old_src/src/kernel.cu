#include <iostream>

#include "kernel.hpp"

using namespace std;

__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; //What thread in block
    int stride = blockDim.x * gridDim.x; //How many threads in block

    
    for (int i = index; i < n; i += stride) {
        y[i] += x[i];
    }
}

void call_global() {
    int N = 1 << 21; //1 Million

    float *x, *y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Cuda runs in "blocks of threads" where threads in each block are a multiple of 32
    //add<<<num of blocks, num of threads in block>>
    dim3 blockSize(256);
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x);
    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    cout << "Max Error: " << maxError << endl;

    cudaFree(x);
    cudaFree(y);

}
