#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

__global__ void hist(uint8_t* input, int* hist, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  for (int idx = i; idx < n; idx += gridDim.x * blockDim.x) {
    int val = input[idx];  // 自动转成 int
    atomicAdd(&hist[val], 1);
  }
}

int main() {
  int M = 4096;
  int N = 4096;
  int size = M * N;

  // ✅ 用 uint8_t
  uint8_t* input = new uint8_t[size];

  for (int i = 0; i < size; ++i) {
    input[i] = rand() % 256;  // 0~255
  }

  uint8_t* d_input;
  int* d_hist;

  cudaMalloc(&d_input, size * sizeof(uint8_t));
  cudaMalloc(&d_hist, 256 * sizeof(int));

  cudaMemset(d_hist, 0, 256 * sizeof(int));

  dim3 block_size(256);
  dim3 grid_size(256);

  cudaMemcpy(d_input, input, size * sizeof(uint8_t), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  hist<<<grid_size, block_size>>>(d_input, d_hist, size);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("cuda error: %s\n", cudaGetErrorString(err));
  }

  printf("Kernel execution time: %f ms\n", milliseconds);

  int h_hist[256];
  cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

  // 打印前10个桶
  for (int i = 0; i < 10; ++i) {
    printf("%d : %d\n", i, h_hist[i]);
  }

  // 释放资源
  cudaFree(d_input);
  cudaFree(d_hist);
  delete[] input;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}