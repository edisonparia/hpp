
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

void initializeVector(float *vector, int n) {
  for(int i = 0; i < n; i++){
    vector[i] = 1 + rand() % 100;
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
void printVector(float *vector, int n) {
  cout << "[ ";
  for(int i = 0; i < n; i++)
    cout << vector[i] << " ";
  cout << "]" << endl;
}

__global__ 
void matrixVectorMultiplicationKernel(float *h_A, float *h_B, float *h_C, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += h_B[row * n + j] * h_C[j];
        }
        h_A[row] = sum;
    }
}

void matrixVectorMultiplication(float *h_A, float *h_B, float *h_C, int n)
{
    size_t matrixSize = n * n * sizeof(float);
    size_t vectorSize = n * sizeof(float);

    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_B, matrixSize);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_C, vectorSize);
    cudaMemcpy(d_C, h_C, vectorSize, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_A, vectorSize);

    dim3 dimBlock(256);
    
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

    matrixVectorMultiplicationKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_A, d_A, vectorSize, cudaMemcpyDeviceToHost);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A);
}

int main(int argc, char *argv[])
{
    const int n = 1000;
    size_t size = n * n * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    initializeMatrix(h_B, n);
    initializeVector(h_C, n);
    matrixVectorMultiplication(h_A, h_B, h_C, n);

    printVector(h_C, n);

    return 0;
}
