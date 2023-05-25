#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel to add two matrices
__global__ void add_matrix(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        int index = i * n + j;
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n;
    cout << "Enter the size of matrices: ";
    cin >> n;

    int size = n * n * sizeof(int);
    int *h_a, *h_b, *h_c; // host arrays
    h_a = new int[n*n];
    h_b = new int[n*n];
    h_c = new int[n*n];

    // Take input of matrix elements
    cout << "Enter elements of matrix A:" << endl;
    for (int i = 0; i < n*n; i++)
        cin >> h_a[i];

    cout << "Enter elements of matrix B:" << endl;
    for (int i = 0; i < n*n; i++)
        cin >> h_b[i];

    // Device arrays
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy input matrices from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Call CUDA kernel to add matrices
    add_matrix<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy result matrix from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print result matrix
    cout << "Result matrix C:" << endl;
    for (int i = 0; i < n*n; i++) {
        cout << h_c[i] << " ";
        if ((i+1) % n == 0) cout << endl;
    }

    // Free memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
