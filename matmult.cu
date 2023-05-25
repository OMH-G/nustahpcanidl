#include <stdio.h>

__global__ void matrixMul(int *a, int *b, int *c, int size) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < size && row < size) {
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sum += a[row * size + i] * b[i * size + col];
        }
        c[row * size + col] = sum;
    }
}

int main() {
    int size = 3;
    int *a, *b, *c; // host matrices
    int *d_a, *d_b, *d_c; // device matrices
    int memSize = size * size * sizeof(int);

    // allocate memory on host
    a = (int *)malloc(memSize);
    b = (int *)malloc(memSize);
    c = (int *)malloc(memSize);

    // get input values from user
    printf("Enter values for matrix A (3x3):\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            scanf("%d", &a[i * size + j]);
        }
    }

    printf("Enter values for matrix B (3x3):\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            scanf("%d", &b[i * size + j]);
        }
    }

    // allocate memory on device
    cudaMalloc((void **)&d_a, memSize);
    cudaMalloc((void **)&d_b, memSize);
    cudaMalloc((void **)&d_c, memSize);

    // copy input data from host to device
    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 dimBlock(32, 32);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x, (size + dimBlock.y - 1) / dimBlock.y);
    matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, size);

    // copy output data from device to host
    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);

    // print result
    printf("Result matrix C (3x3):\n");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", c[i * size + j]);
        }
        printf("\n");
    }

    // free memory
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
