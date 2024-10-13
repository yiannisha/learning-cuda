#include <stdlib.h>
#include <stdio.h>

int main()
{
    uint devCount;
    cudaGetDeviceCount((int *)&devCount);
    printf("Detected %d %s.\n", devCount, devCount == 1 ? "device":"devices");

    cudaDeviceProp devProp;
    for (uint i=0; i<devCount; ++i) {
        cudaGetDeviceProperties(&devProp, i);
        printf("\n");
        printf("Device %d:\n", i);

        printf("Max number of threads per block: %d\n", devProp.maxThreadsPerBlock);
        printf("Streaming Multiprocessor (SM) count: %d\n", devProp.multiProcessorCount);
        printf("Clock Rate: %d\n", devProp.clockRate);
        printf("Max throughput: %d\n", devProp.clockRate*devProp.multiProcessorCount);
        printf("Max block dimensions: (%d %d %d)\n",
            devProp.maxThreadsDim[0],
            devProp.maxThreadsDim[1],
            devProp.maxThreadsDim[2]
        );
        printf("Max Grid size: (%d %d %d)\n",
            devProp.maxGridSize[0],
            devProp.maxGridSize[1],
            devProp.maxGridSize[2]
        );
        printf("Registers in each SM: %d\n", devProp.regsPerBlock);
        printf("Warp size: %d\n", devProp.warpSize);
    }

    return EXIT_SUCCESS;
}