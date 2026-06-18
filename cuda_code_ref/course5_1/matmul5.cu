#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // for fabsf
#include <fstream> // for CSV output
#include <iostream>
#include <vector>

#define BLOCK_SIZE 128
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

// 将数据从Global Memory搬运到Shared Memory中。
template <const int BM, const int BN, const int BK, const int row_stride_a,
          const int row_stride_b>
__device__ void load_from_gmem(int N, int K, const float *A, const float *B,
                               float *As, float *Bs, int inner_row_a,
                               int inner_col_a, int inner_row_b,
                               int inner_col_b)
{
  for (uint off_set = 0; off_set + row_stride_a <= BM; off_set += row_stride_a)
  {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(inner_row_a + off_set) * K + inner_col_a * 4])[0];
    // 这里tmp放入AS的时候已经进行了转置
    As[(inner_col_a * 4 + 0) * BM + inner_row_a + off_set] = tmp.x;
    As[(inner_col_a * 4 + 1) * BM + inner_row_a + off_set] = tmp.y;
    As[(inner_col_a * 4 + 2) * BM + inner_row_a + off_set] = tmp.z;
    As[(inner_col_a * 4 + 3) * BM + inner_row_a + off_set] = tmp.w;
  }

  for (uint off_set = 0; off_set + row_stride_b <= BK; off_set += row_stride_b)
  {
    reinterpret_cast<float4 *>(
        &Bs[(inner_row_b + off_set) * BN + inner_col_b * 4])[0] =
        reinterpret_cast<const float4 *>(&B[(inner_row_b + off_set) * N + inner_col_b * 4])[0];
  }
}

/*
thread_row_in_warp：线程在WSUB里的“线程行号”
thread_col_in_warp：线程在WSUB里的“线程列号”
*/
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void process_from_smem(float *reg_m, float *reg_n,
                                  float *thread_results, const float *As,
                                  const float *Bs, const uint warp_row,
                                  const uint warp_col,
                                  const uint thread_row_in_warp,
                                  const uint thread_col_in_warp)
{
  for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
  {
    for (uint w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx)
    {
      for (uint i = 0; i < TM; ++i)
      {
        // 此时As已经是A的转置了，所以dot_idx * BM是行
        reg_m[w_sub_row_idx * TM + i] =
            As[(dot_idx * BM) + warp_row * WM + w_sub_row_idx * WSUBM +
               thread_row_in_warp * TM + i];
      }
    }
    for (uint w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx)
    {
      for (uint i = 0; i < TN; ++i)
      {
        reg_n[w_sub_col_idx * TN + i] =
            Bs[(dot_idx * BN) + warp_col * WN + w_sub_col_idx * WSUBN +
               thread_col_in_warp * TN + i];
      }
    }

    for (uint w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx)
    {
      for (uint w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx)
      {
        for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
        {
          for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
          {
            thread_results[(w_sub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                           (w_sub_col_idx * TN) + res_idx_n] +=
                reg_m[w_sub_row_idx * TM + res_idx_m] *
                reg_n[w_sub_col_idx * TN + res_idx_n];
          }
        }
      }
    }
  }
}

constexpr int WARP_SIZE = 32;

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    mysgemm_warptiling(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C)
{
  const uint c_row = blockIdx.y;
  const uint c_col = blockIdx.x;

  const uint warp_idx = threadIdx.x / WARP_SIZE; // 第几个warp
  const uint warp_col = warp_idx % (BN / WN);    // warp在block内负责计算的C矩阵的列块索引
  const uint warp_row = warp_idx / (BN / WN);    // warp在block内负责计算的C矩阵的行块索引
  // WNITER和WMITER：线程阵列为了覆盖整个Warp区域，需要在N和M方向分别跳跃迭代的次数。
  constexpr uint WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);

  /*
  1.WSUBN和WSUBM的原因：Warp负责的区域还是太大，32个线程一次算不完。于是把Warp Tile进一步切成更小的Sub-tile。
  2.WSUB（Warp Sub-tile）区域本质上就是 32 个线程的“任务块”合并拼凑出来的总面积。
  3.每个线程处理的大小 = TM (行)*TN(列)。
    整个子块的大小 = WSUBM (行)*WSUBN (列)。
    Warp处理的总大小 = WM (行)*WN (列)。
  */
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;

  const uint thread_idx_in_warp = threadIdx.x % WARP_SIZE;
  const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
  const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += c_row * BM * K; // 移动到当前 Block 负责的 A 行
  B += c_col * BN;     // 移动到当前 Block 负责的 B 列
  // C 直接定位到该线程所属 Warp 的起始位置
  C += (c_row * BM + warp_row * WM) * N + c_col * BN + warp_col * WN;

  const uint inner_row_a = threadIdx.x / (BK / 4);      // 线程在M维度（行方向）上初始站在哪一行。
  const uint inner_col_a = threadIdx.x % (BK / 4);      // 线程在K维度（列方向）上的起始位置索引。
  constexpr uint row_stride_a = NUM_THREADS / (BK / 4); // row_stride_a是block的线程处理他分配的矩阵要多少次才能处理完
  const uint inner_row_b = threadIdx.x / (BN / 4);
  const uint inner_col_b = threadIdx.x % (BN / 4);
  constexpr uint row_stride_b = NUM_THREADS / (BN / 4);

  float thread_results[WMITER * TM * WNITER * TN] = {0.0}; // thread_results 是一个存在寄存器里的一维数组，存了该线程负责的所有结果。
  float reg_m[WMITER * TM] = {0.0};
  float reg_n[WNITER * TN] = {0.0};

  for (uint bk_idx = 0; bk_idx < K; bk_idx += BK)
  {
    // 1. 从显存搬运数据到 Shared Memory (As, Bs)
    load_from_gmem<BM, BN, BK, row_stride_a, row_stride_b>(
        N, K, A, B, As, Bs, inner_row_a, inner_col_a, inner_row_b, inner_col_b);
    __syncthreads();
    // 2. 从 Shared Memory 读取到寄存器并完成乘加计算
    process_from_smem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        reg_m, reg_n, thread_results, As, Bs, warp_row, warp_col,
        thread_row_in_warp, thread_col_in_warp);
    A += BK;     // A 矩阵右移 BK 列
    B += BK * N; // B 矩阵下移 BK 行
    __syncthreads();
  }

  for (uint w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) // Warp 内部M方向的子块迭代。
  {
    for (uint w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) // Warp 内部N方向的子块迭代。
    {
      /*
      1.C 在进入这部分循环前，已经指向了该 Warp 负责区域64*64的左上角。
      2.由于 Warp 内部可能分成了多个子块（Sub-tile），这里根据当前的迭代索引 w_sub_row_idx 和 w_sub_col_idx 进一步偏移指针。
      WSUBM/N 是每个子块的宽高。
      */
      float *C_interim = C + (w_sub_row_idx * WSUBM) * N + w_sub_col_idx * WSUBN; // 定位子块起始地址
      for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m += 1)                    // 线程在子块内负责的行。
      {
        for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n += 4) // 线程在子块内负责的列
        {
          // 代码第一行先读出了tmp。这是因为公式里有 +beta*tmp.x。如果beta不为0，我们需要先知道C原来的值是多少，再加上我们算出的新结果。
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(thread_row_in_warp * TM + res_idx_m) * N +
                         thread_col_in_warp * TN + res_idx_n])[0];

          // 由于线程把所有的计算结果都存在一个一维数组 thread_results 里，我们需要一个索引 i 来取出正确的值。
          const int i = (w_sub_row_idx * TM + res_idx_m) * (WNITER * TN) + w_sub_col_idx * TN + res_idx_n;
          tmp.x = alpha * thread_results[i + 0] + beta * tmp.x;
          tmp.y = alpha * thread_results[i + 1] + beta * tmp.y;
          tmp.z = alpha * thread_results[i + 2] + beta * tmp.z;
          tmp.w = alpha * thread_results[i + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(
              &C_interim[(thread_row_in_warp * TM + res_idx_m) * N +
                         thread_col_in_warp * TN + res_idx_n])[0] = tmp;
        }
      }
    }
  }
}

std::vector<int> generateSizes()
{
  std::vector<int> sizes;
  for (int i = 256; i <= 8192; i += 256)
  {
    sizes.push_back(i);
  }
  return sizes;
}

/*
这个整体的意思就是把整个矩阵用很多block处理其中的一个小矩阵，然后block处理的矩阵部分又分成了多个warp来处理，然后warp处理的矩阵中，
每个线程又可以处理一个小矩阵。
*/
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
int main()
{
  std::vector<int> sizes = generateSizes();
  std::ofstream csv_file("sgemm_benchmark_warp_tiling.csv"); // 打开sgemm_benchmark_v7.csv文件，记录测试结果。
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched,Ratio" << std::endl;

  for (int N : sizes)
  {
    std::cout << "Testing size: " << N << std::endl;

    size_t size = N * N * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C_cublas = (float *)malloc(size); // 由NVIDIA的库cuBLAS计算出的结果。
    float *C_v1 = (float *)malloc(size);     // 自己编写的优化Kernel计算出的结果

    // d_C_v1是显存中的结果。它是 GPU 上的缓冲区，不论是 cuBLAS 还是手写 Kernel，计算结果都会先存到这里，再拷贝回 CPU。
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
      checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice), "cudaMemcpy A to device failed");
      checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice), "cudaMemcpy B to device failed");

      cublasHandle_t handle;
      checkCublasError(cublasCreate(&handle), "cublasCreate failed");

      float alpha = 1.0f;
      float beta = 0.0f;

      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
      checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

      // 对GPU进行热身
      int warpup_time = 10; // 热身次数
      for (int i = 0; i < warpup_time; ++i)
      {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 5;                                                            // 跑五次
      checkCudaError(cudaEventRecord(start), "cudaEventRecord(start cublas) failed"); // 计时开始
      for (int i = 0; i < repeat_time; ++i)                                           // 连续调用 5 次官方的矩阵乘法函数。
      {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }

      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop cublas) failed");     // 停止计时
      checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize cublas failed"); // CPU 会在这里停下等待，直到 GPU 把前面所有的计算和“停止打卡”动作全部干完。
      // 计算 start 到 stop 这两个打卡点之间经过的毫秒数。
      float cublas_time = 0; // cublas_time 就是 5 次运算的总耗时。
      checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                     "cudaEventElapsedTime cublas failed");

      // 将 GPU 算好的结果（在 d_C_v1 里）拷贝到 CPU 的 C_cublas 数组里。
      checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");

      // 将显存中的结果缓冲区 d_C_v1 全部抹成 0。
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      const uint K10_NUM_THREADS = 128;
      // 每个 Block 负责计算128*128的C区域。
      const uint K10_BN = 128;
      const uint K10_BM = 128;
      const uint K10_BK = 16; // 每次往 Shared Memory 搬 16 个的 A 和 B。
      // 一个 Warp（32线程）负责计算64*64的区域。
      const uint K10_WN = 64;
      const uint K10_WM = 64;
      const uint K10_WNITER = 4; // Warp 在N方向上需要分 4 次迭代来完成计算。
      // 每个线程最终负责8*4的寄存器小块。
      const uint K10_TN = 4;
      const uint K10_TM = 8;
      dim3 blockDim(K10_NUM_THREADS);

      constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

      // warptile in threadblocktile
      static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
      static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);
      // threads in warpsubtile
      static_assert(
          (K10_WM * K10_WN) % (WARP_SIZE * K10_TM * K10_TN * K10_WNITER) == 0);
      constexpr uint K10_WMITER =
          (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
      // warpsubtile in warptile
      static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

      static_assert(
          (K10_NUM_THREADS * 4) % K10_BK == 0,
          "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
          "issues during GMEM->SMEM tiling (loading only parts of the "
          "final row of Bs during each iteraion)");
      static_assert(
          (K10_NUM_THREADS * 4) % K10_BN == 0,
          "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
          "issues during GMEM->SMEM tiling (loading only parts of the "
          "final row of As during each iteration)");
      static_assert(
          K10_BN % (16 * K10_TN) == 0,
          "BN must be a multiple of 16*TN to avoid quantization effects");
      static_assert(
          K10_BM % (16 * K10_TM) == 0,
          "BM must be a multiple of 16*TM to avoid quantization effects");
      static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                    "BM*BK must be a multiple of 4*256 to vectorize loads");
      static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                    "BN*BK must be a multiple of 4*256 to vectorize loads");

      dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(N, K10_BM));
      // 热身
      for (int i = 0; i < warpup_time; ++i)
      {
        mysgemm_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER,
                           K10_TM, K10_TN, K10_NUM_THREADS>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      cudaDeviceSynchronize();

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");
      // 连续运行repeat_time次自己编写的矩阵乘法 kernel。
      for (int i = 0; i < repeat_time; ++i)
      {
        mysgemm_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER,
                           K10_TM, K10_TN, K10_NUM_THREADS>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v1 failed");
      checkCudaError(cudaGetLastError(), "cuda get last error failed");
      float v1_time = 0;
      checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                     "cudaEventElapsedTime v1 failed");

      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v1 failed");
      // 结果比较
      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i)
      {
        if (fabsf(C_cublas[i] - C_v1[i]) > TOL)
        {
          error_count++;
        }
      }

      float cublas_gflops =
          repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f); // GFlops
      float v1_gflops =
          repeat_time * 2.0f * N * N * N / (v1_time * 1e6f); // GFlops

      float ratio = v1_gflops / cublas_gflops;
      // 写入CSV
      csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
               << (error_count == 0 ? "1" : "0") << "," << ratio << std::endl;

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
      cudaDeviceSynchronize();
    }
    catch (...)
    {
      std::cerr << "Out of memory or error during testing size: " << N
                << std::endl;
      out_of_memory = true;
    }

    if (!out_of_memory)
    {
      std::cout << "Finished size: " << N << std::endl;
    }
    else
    {
      csv_file << N << ",OOM,OOM,0" << std::endl;
    }
  }

  csv_file.close();

  std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
            << std::endl;
  return 0;
}
