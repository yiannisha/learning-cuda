#include <stdio.h>
#include <stdlib.h>

#define THREAD_COUNT 1024

// MxN * NxK matrix multiplication

void handle_err(cudaError_t err) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
}

__global__
void matMulKernel(
    int* left,
    int* right,
    int* out,
    unsigned int M,
    unsigned int N,
    unsigned int K
)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= K) return;

    int outVal = 0;
    for (int i=0; i<N; i++) {
        outVal += left[row * N + i] * right[i * K + col];
    }
    out[row * K + col] = outVal;
}

int main()
{
    int length = 3;

    // MxN
    int A[12] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };
    unsigned int M = 4;

    // NxK
    int B[6] = {
        1, 2,
        3, 4,
        5, 6
    };
    unsigned int K = 2;

    int *out_h, *out_d, *A_d, *B_d;
    unsigned int size = (M*K) * sizeof(int);
    out_h = (int *)malloc(size);

    cudaError_t err;
    if ((err = cudaMalloc((void**)&out_d, size)) != cudaSuccess) handle_err(err);
    if ((err = cudaMalloc((void**)&A_d, 12*sizeof(int))) != cudaSuccess) handle_err(err);
    if ((err = cudaMalloc((void**)&B_d, 6*sizeof(int))) != cudaSuccess) handle_err(err);

    cudaMemcpy(A_d, A, 12*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, 6*sizeof(int), cudaMemcpyHostToDevice);

    // it is convenient to map 2D data to a 2D grid consisting of
    // 2D blocks
    dim3 dimGrid = dim3(ceil(K/(float)32), ceil(M/(float)32), 1);
    dim3 dimBlocks = dim3(32, 32, 1);

    matMulKernel<<<dimGrid, dimBlocks>>>(A_d, B_d, out_d, M, length, K);

    cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<M; i++) {
        for (int j=0; j<K; j++) {
            printf("%d ", out_h[i*K + j]);
        }
        printf("\n");
    }

    cudaFree(out_d);
    cudaFree(A_d);
    cudaFree(B_d);

    free(out_h);

    return EXIT_SUCCESS;
}