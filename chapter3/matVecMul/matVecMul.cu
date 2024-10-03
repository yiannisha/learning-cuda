#include <stdlib.h>
#include <stdio.h>

void handle_err(cudaError_t err) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
}

__global__
void matVecMul (
    double* out,
    double* mat,
    double* vec,
    uint length
)
{
    uint row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= length) return;

    double val = 0.0;
    for (uint i=0; i<length; i++) {
        val += mat[row * length + i] * vec[i];
    }

    out[row] = val;
}

int main()
{
    uint length = 10;

    double B_h[100] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };
    double C_h[10] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    };

    uint size = length * sizeof(double);
    double *A_h, *A_d, *B_d, *C_d;
    A_h = (double *)malloc(size);

    cudaError_t err;
    if ((err = cudaMalloc((void**)&A_d, size)) != cudaSuccess) handle_err(err);
    if ((err = cudaMalloc((void**)&B_d, size*size)) != cudaSuccess) handle_err(err);
    if ((err = cudaMalloc((void**)&C_d, size)) != cudaSuccess) handle_err(err);

    cudaMemcpy(C_d, C_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size*size, cudaMemcpyHostToDevice);

    dim3 dimGrid = dim3(ceil(length/1024.0), 1, 1);
    dim3 dimBlock = dim3(1024, 1, 1);

    matVecMul<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, length);

    cudaMemcpy(A_h, A_d, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<length; i++) printf("%f\n", A_h[i]);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A_h);

    return EXIT_SUCCESS;
}