#include <cuda_runtime.h>
#include <iostream>

#define BDIMX 32
#define BDIMY 16

/*
BDIMX == blockDim.x
BDIMY == blockDim.y
threadIdx.y是block里面的行坐标
threadIdx.x是block里面的列坐标
nx = 列数
ny = 行数
*/
__global__ void transposeSmemUnrollPad(float *out, float *in, int nx, int ny) 
{
  const int IPAD = 1; // 每行增加一个元素的padding，避免bank conflict
  __shared__ float tile[BDIMY * (BDIMX * 2 + IPAD)];
  //计算全局坐标
  unsigned int ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
  //全局内存索引：in[iy][ix]
  unsigned int ti = iy * nx + ix;

  unsigned int bidx = blockDim.x * threadIdx.y + threadIdx.x; //二维线程坐标 → 一维线程编号（表示当前线程是block中的第几个线程）
  //转置后的行和列坐标
  unsigned int irow = bidx / blockDim.y;
  unsigned int icol = bidx % blockDim.y;
  //转置后的位置：out[ix][iy]
  unsigned int ix2 = blockIdx.y * blockDim.y + icol;
  unsigned int iy2 = 2 * blockIdx.x * blockDim.x + irow;
  //输出地址：out[iy2][ix2]
  unsigned int to = iy2 * ny + ix2;

  if ((ix + blockDim.x) < nx && iy < ny) //防止越界访问
  {
    unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) + threadIdx.x;
    //把数据位置不变放到shared memory中，行优先存储
    tile[row_idx] = in[ti]; 
    tile[row_idx + blockDim.x] = in[ti + blockDim.x];
    __syncthreads();
    unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;
    out[to] = tile[col_idx];
    out[to + ny * blockDim.x] = tile[col_idx + blockDim.x];
  }
}

void call_transposeSmemUnrollUnpad(float *d_out, float *d_in, const int nx,
                                   const int ny) 
{
  dim3 blockSize(BDIMX, BDIMY);
  auto grid = (nx + BDIMX - 1) / BDIMX;
  dim3 gridSize(int(grid / 2), (ny + BDIMY - 1) / BDIMY);
  transposeSmemUnrollPad<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
}

void naiveSmemWrapperUnrollUnpad() 、
{
  //矩阵的大小
  int nx = 4096;
  int ny = 4096;
  //矩阵字节数
  size_t size = nx * ny * sizeof(float);

  // 主机内存分配
  float *h_in = (float *)malloc(size);
  float *h_out = (float *)malloc(size);

  // 初始化输入矩阵
  for (int i = 0; i < nx * ny; i++) 
    h_in[i] = float(int(i) % 11);
  
  // 设备内存分配
  float *d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // 将数据从主机复制到设备
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //给gpu先热身
  int warp_up_iter = 5;
  for (int i = 0; i < warp_up_iter; ++i) 
    call_transposeSmemUnrollUnpad(d_out, d_in, nx, ny);
  

  int bench_iter = 5;
  // 开始计时
  cudaEventRecord(start);
  // 调用核函数
  for (int i = 0; i < bench_iter; ++i) 
    call_transposeSmemUnrollUnpad(d_out, d_in, nx, ny);
  

  // 结束计时
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
  {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    return;
  }

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Smem transpose unroll unpad kernel execution time: "
            << milliseconds / float(bench_iter) << " ms" << std::endl;

  // 将结果从设备复制回主机
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
  // 释放内存
  free(h_in);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);

  std::cout << "Matrix transposition completed successfully." << std::endl;
}

int main() {
  naiveSmemWrapperUnrollUnpad();
  return 0;
}