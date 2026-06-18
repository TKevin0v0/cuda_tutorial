#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

__global__ void hist(uint8_t* input, int* hist, int n) 
{
  __shared__ int histo_private[256]; //在每个线程块（Block）中开辟一个长度为 256 的数组。

  // 初始化 shared memory
  for (int j = threadIdx.x; j < 256; j += blockDim.x) 
    histo_private[j] = 0;
  
  __syncthreads();

  // 构建 block 内局部直方图
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  for (int idx = i; idx < n; idx += gridDim.x * blockDim.x)  // 这里idx跳着相加就是允许核函数处理比总线程数更多的元素
  {
    int val = input[idx];  // uint8_t → int
    atomicAdd(&histo_private[val], 1);
  }
  __syncthreads();

  // 合并到全局直方图
  for (int j = threadIdx.x; j < 256; j += blockDim.x) 
    atomicAdd(&hist[j], histo_private[j]);
  
}

int main() 
{ 
  //定义图像的行和列
  int M = 4096;
  int N = 4096;
  int size = M * N;
  uint8_t* input = new uint8_t[size];
  for (int i = 0; i < size; ++i) //生成输入数据
    input[i] = rand() % 256;
  uint8_t* d_input;
  int* d_hist;

  cudaMalloc(&d_input, size * sizeof(uint8_t));
  cudaMalloc(&d_hist, 256 * sizeof(int));
  cudaMemset(d_hist, 0, 256 * sizeof(int)); //将显存中的直方图数组初始化为 0

  dim3 block_size(256);
  dim3 grid_size(256);

  cudaMemcpy(d_input, input, size * sizeof(uint8_t), cudaMemcpyHostToDevice); //把数据放到gpu中
  //设置计时器并启动kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  hist<<<grid_size, block_size>>>(d_input, d_hist, size);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop); //阻塞Host端（CPU）线程，直到指定的Event（即stop）被GPU真正执行完毕。

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop); //计算 Kernel 运行的实际毫秒数。

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)  //检查 GPU 运行过程中是否有报错
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  printf("Kernel execution time: %f ms\n", milliseconds);

  int h_hist[256];
  cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

  // 打印前10个桶
  for (int i = 0; i < 10; ++i) 
    printf("%d : %d\n", i, h_hist[i]);

  // 验证总和（非常重要）
  int sum = 0;
  for (int i = 0; i < 256; i++) 
    sum += h_hist[i];
  printf("sum = %d (expected %d)\n", sum, size);

  // 释放资源
  cudaFree(d_input);
  cudaFree(d_hist);
  delete[] input;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}