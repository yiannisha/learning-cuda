#include <stdlib.h>
#include <stdio.h>

__global__
void vecAddKernel(int *a, int *b, int *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // int i = (blockDim.x * blockIdx.x + threadIdx.x)*2;
    if (i >= n) return;
    c[i] = a[i] + b[i];
    // if (i+1 >= n) return;
    // c[i+1] = a[i+1] + b[i+1];
}

void handle_err(cudaError_t err) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
}

int main() {
    int n = 1000;
    int size = n * sizeof(int);

    // allocate memory for the vectors
    
    // this compiles and is best practise in C
    // but does NOT compile when using nvcc (or any C++ compiler)
    // int *a_h = malloc(size);
    // int *b_h = malloc(size);
    // int *c_h = malloc(size);
    
    int *a_h = (int*)malloc(size);
    int *b_h = (int*)malloc(size);
    int *c_h = (int*)malloc(size);

    // read the 2 vectors to add
    for (int i=0; i<n; i++) {
        a_h[i] = i;
        b_h[n-i-1] = i;
    }

    // pointers to device memory
    int *a_d, *b_d, *c_d;
    // allocate memory on device for the 2 vectors and the result vector
    cudaError_t err;
    if ((err = cudaMalloc((void**)&a_d, size)) != cudaSuccess) handle_err(err);
    if ((err = cudaMalloc((void**)&b_d, size)) != cudaSuccess) handle_err(err);
    if ((err = cudaMalloc((void**)&c_d, size)) != cudaSuccess) handle_err(err);

    // copy vectors on device
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    // run kernel function
    vecAddKernel<<<ceil(n/256.0), 256>>>(a_d, b_d, c_d, n);

    // copy result vector from device to host
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    // print output
    // for (int i=0; i<n; i++) {
    //     printf("a[%d] + b[%d] = %d\n", i, i, c_h[i]);
    // }

    // free host memory
    free(a_h);
    free(b_h);
    free(c_h);

    // free device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return EXIT_SUCCESS;
}
