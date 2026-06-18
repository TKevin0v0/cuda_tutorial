#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>   // fabsf的库
#include <fstream> // CSV输入输出
#include <iostream>
#include <vector>
#define TOL 1e-5f

void checkCudaError(cudaError_t err, const char *msg)
{
  if (err != cudaSuccess)
  {
    std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void checkCublasError(cublasStatus_t status, const char *msg)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

/*
优化3:
1.一个线程计算多个 C 元素
2.为了计算这C的某个tile需要A 的BM×K（因为C 的行来自 A 的行。），B 的K×BN（因为：C 的列来自
B 的列）。但K 太大，不能一次放进 shared memory。所以K 维度继续切块，即每次只处理BK长度。于是
每轮shared memory只存A的BM×BK（就是当前 C tile 相关的A 的所有需要行+K方向的一小段列），B的
BK×BN（当前 C tile 相关的：B 的所有需要列+K方向的一小段行）

注意：
1.一个 block 对应一个 tile
2.BM表示C tile有多少行，同时A tile 有多少行。
BN表示C tile有多少列，同时B tile有多少列。
BK表示每次在 K 维度切多少长度，也就是最终结果相关的A取多少列，B取多少行
3.在初始阶段，block的坐标就已经决定了自己要处理结果矩阵c的某一部分
4.搬运阶段:一个线程搬一个数据。计算阶段：一个线程算一整块。
*/
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C)
{
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int block_row_thread = BN / TN;                       // 一个tile的一行，需要多少个线程才能覆盖所有列
  int block_col_thread = BM / TM;                       // 一个tile的一列，需要多少个线程才能覆盖所有行
  int thread_num = block_row_thread * block_col_thread; // 一个block对应一个tile，所以这里本质是一个block需要的总线程数
  // tx和ty是当前线程在当前 block tile 里负责的小tile左上角坐标
  int ty = (threadIdx.x / block_row_thread) * TM; // 当前线程负责的一小块的起始行
  // threadIdx.x % block_row_thread算的是是横向第几个线程，
  int tx = (threadIdx.x % block_row_thread) * TN; // 当前线程负责的一小块的起始列
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  // 把指针移动到当前block要处理的tile起始位置
  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];
  int a_tile_row = threadIdx.x / BK;   // 当前线程在“搬运 A tile 时”负责的初始行
  int a_tile_col = threadIdx.x % BK;   // 当前线程在“搬运 A tile 时”负责的初始列
  int a_tile_stride = thread_num / BK; // 线程下一次搬运时，跳多少行执行下一次搬运。也就是这里因为每一个线程搬运一个数据，所以线程数/BK，就是所以线程一起覆盖了多少行
  int b_tile_row = threadIdx.x / BN;
  int b_tile_col = threadIdx.x % BN;
  int b_tile_stride = thread_num / BN;
  float tmp[TM][TN] = {0.}; // 小矩阵累加器,每个线程私有的
  // 计算一个结果的每个部分和，最后全部加起来就是最终和
  for (int k = 0; k < K; k += BK)
  {
    // 当前block需要的A tile的所有行
    for (int i = 0; i < BM; i += a_tile_stride)
      As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
    // B的相关所有列和部分行
    for (int i = 0; i < BK; i += b_tile_stride)
      Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
    __syncthreads();
    A += BK;
    B += BK * N;
    for (int i = 0; i < BK; i++) // 每个结果的每一项进行计算，最后相加起来
      // 把线程负责的区域的某一项计算出来加到tmp中
      // 把最终结果BM×BN分成很多TM×TN让每个线程计算,然后每个线程用共享内存里的数据去算自己负责的结果
      // 这里在遍历当前线程负责的结果区域，不是遍历共享内存里的数据区域
      for (int j = 0; j < TM; j++)
        for (int l = 0; l < TN; l++)
          tmp[j][l] += As[(ty + j) * BK + i] * Bs[(tx + l) + i * BN]; // i(也就是BK)相对于AS就是列，相对于BS就是行
    __syncthreads();
  }
  for (int j = 0; j < TM; j++)
    for (int l = 0; l < TN; l++)
      C[(ty + j) * N + tx + l] = alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
std::vector<int> generateSizes()
{
  return {4096};
}
int main()
{
  int device_id = 0;
  checkCudaError(cudaSetDevice(device_id), "cudaSetDevice failed");
  std::vector<int> sizes = generateSizes();
  // 打开CSV文件
  std::ofstream csv_file("sgemm_benchmark_v3.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;
  for (int N : sizes)
  {
    std::cout << "Testing size: " << N << std::endl;
    size_t size = N * N * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C_cublas = (float *)malloc(size);
    float *C_v1 = (float *)malloc(size);
    float *d_A, *d_B, *d_C_v1;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");
    bool out_of_memory = false;
    try
    {
      // 初始化矩阵 A 和 B
      for (int i = 0; i < N * N; ++i)
      {
        A[i] = 1.0f;
        B[i] = 2.0f;
      }
      // 拷贝到设备
      checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy A to device failed");
      checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy B to device failed");
      cublasHandle_t handle;
      checkCublasError(cublasCreate(&handle), "cublasCreate failed");
      float alpha = 1.0f;
      float beta = 0.0f;
      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
      checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");
      // 给GPU热身
      int warpup_time = 10; // 热身次数
      for (int i = 0; i < warpup_time; ++i)
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      cudaDeviceSynchronize();
      // cuBLAS SGEMM
      int repeat_time = 5;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i)
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      checkCudaError(cudaEventRecord(stop),
                     "cudaEventRecord(stop cublas) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize cublas failed");
      float cublas_time = 0;
      checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                     "cudaEventElapsedTime cublas failed");
      // 拷贝 cuBLAS 结果
      checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");
      // mysgemm_v4
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");
      dim3 blockDim(256);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));
      for (int i = 0; i < warpup_time; ++i)
        mysgemm_v4<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      cudaDeviceSynchronize();
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");
      for (int i = 0; i < repeat_time; ++i)
        mysgemm_v4<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v1 failed");
      float v1_time = 0;
      checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                     "cudaEventElapsedTime v1 failed");
      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v1 failed");
      // 结果比较
      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i)
        if (fabsf(C_cublas[i] - C_v1[i]) > TOL)
          error_count++;
      float cublas_gflops =
          repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f); // GFlops
      float v1_gflops =
          repeat_time * 2.0f * N * N * N / (v1_time * 1e6f); // GFlops
      // 写入CSV
      csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
               << (error_count == 0 ? "1" : "0") << std::endl;
      // 释放资源
      cublasDestroy(handle);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C_v1);
      free(A);
      free(B);
      free(C_cublas);
      free(C_v1);
    }
    catch (...)
    {
      std::cerr << "Out of memory or error during testing size: " << N
                << std::endl;
      out_of_memory = true;
    }
    if (!out_of_memory)
      std::cout << "Finished size: " << N << std::endl;
    else
      csv_file << N << ",OOM,OOM,0" << std::endl;
  }
  csv_file.close();
  std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
            << std::endl;
  return 0;
}
