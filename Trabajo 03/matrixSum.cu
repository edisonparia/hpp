#include <iostream>
#include <cmath>
#include <cuda.h>
#include <time.h>

using namespace std;

void initializeMatrix(float *matrix, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            matrix[i * n + j] = 1;
        }
    }
}

void printMatrix(float *matrix, int n)
{
    for (int i = 0; i < n; i++){
      for (int j = 0; j < n; j++){
        cout << matrix[i * n + j] << " \n"[j == n - 1];
      }
            
    }
        
}

__global__ 
void matrixAddKernel(float *h_A, float *h_B, float *h_C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        int index = row * n + col;
        h_C[index] = h_A[index] + h_B[index];
    }
}

__global__ 
void matrixAddKernelRows(float *h_A, float *h_B, float *h_C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        for (int j = 0; j < n; j++) {
            int index = row * n + j;
            h_C[index] = h_A[index] + h_B[index];
        }
    }
}

__global__ 
void matrixAddKernelColumns(float *h_A, float *h_B, float *h_C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n) {
        for (int i = 0; i < n; i++) {
            int index = i * n + col;
            h_C[index] = h_A[index] + h_B[index];
        }
    }
}


void matrixAdd(float *h_A, float *h_B, float *h_C, int n)
{
    size_t size = n * n * sizeof(float);

    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_C, size);

    dim3 dimBlock(16, 16);
    
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);

    matrixAddKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char *argv[])
{
    const int n = 1000;
    size_t size = n * n * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    initializeMatrix(h_A, n);
    initializeMatrix(h_B, n);
    matrixAdd(h_A, h_B, h_C, n);

    printMatrix(h_C, n);

    return 0;
}
