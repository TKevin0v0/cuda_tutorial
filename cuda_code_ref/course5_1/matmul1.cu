#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void checkCublasError(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

//K是A的列数，也是B的行数
//M是C的行数，N是C的列数（同时M 是矩阵 A 的行数，N 是矩阵 B 的列数）
/*
一个线程 = 算 C 里一个数
一个 block  = 算 C 里一个 32x32 的小块
整个 grid   = 算完整的 C 矩阵
*/
template <const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B,
                            float beta, float *C)
{
   int bx = blockIdx.x; //blockIdx.x 是当前 block 在 grid 中的列坐标
   int by = blockIdx.y; //blockIdx.y 是当前 block 在 grid 中的行坐标
                            
   const int BM = BLOCK_SIZE; //一个 block 负责BM 行
   const int BN = BLOCK_SIZE; // 一个 block 负责BN 列
   const int BK = BLOCK_SIZE; //A的列或者B的行，每次切多少长度来算
         
   //把一维的线程编号，拆成二维坐标（行、列）
   int tx = threadIdx.x % BN; //当前线程在“当前 block 对应的 tile 中”的列坐标
   int ty = threadIdx.x / BN; //当前线程在“当前 block 对应的 tile 中”的行坐标
                            
   __shared__ float As[BM * BK];
   __shared__ float Bs[BK * BN];
                          
   /*
   by就是行坐标，然后每个block是BM行，所以by*BM就是当前block负责的行块的起始行号，乘以K就是A矩阵中对应的元素个数（因为A是按行存储的），
   所以A[by * BM * K]就是当前block负责的行块在A中的起始位置。同理，bx是列坐标，每个block是BN列，所以bx*BN就是当前block负责的列块的起
   始列号，乘以N就是B矩阵中对应的元素个数（因为B是按行存储的），所以B[bx * BN]就是当前block负责的列块在B中的起始位置。对于C矩阵，by*BM*N是
   当前block负责的行块在C中的起始位置，bx*BN是当前block负责的列块在C中的起始位置，所以C[by * BM * N + bx * BN]就是当前block负责的
   tile在C中的左上角位置。
   */
   A = &A[by * BM * K];                 //让 A 指针移动到：当前 block 对应的“行块”的起始位置
   B = &B[bx * BN];                     //让 B 指针移动到：当前 block 对应的“列块”的起始列
   C = &C[by * BM * N + bx * BN];       //当前 block 负责写入的 C tile 的左上角位置。等价于二维数组写法：C[by*BM][bx*BN];
                            
   float tmp = 0.;
   for (int k = 0; k < K; k += BK) //按照K维度切块，然后分别计算后相加就是C的某个位置的数值
   {
      /*
      每个线程只负责计算 C 中一个元素的值，所以每个线程需要加载A的一行和B的一列到共享内存中，来计算C中对应位置的数值。
      因为block里面的线程一起运行，每个线程加载的是一个元素，所以下面放入两个语句就是把tile中的数据全部放到共享内存中。不会多是因为ty和tx的范围大小就是0-31
      */
      As[ty * BK + tx] = A[ty * K + tx]; //等价于As[ty][tx] = A[ty][tx]，把当前 block 对应的 A 的行块切成 BK 列，每个线程负责一个元素，加载到共享内存 As 中。注意这里的 A[ty * K + tx] 是全局内存中的元素访问方式，因为 A 是按行存储的，所以每行有 K 个元素。
      Bs[ty * BN + tx] = B[ty * N + tx]; //等价于Bs[ty][tx] = B[ty][tx]，把当前 block 对应的 B 的列块切成 BK 行，每个线程负责一个元素，加载到共享内存 Bs 中。注意这里的 B[ty * N + tx] 是全局内存中的元素访问方式，因为 B 是按行存储的，所以每行有 N 个元素。
      __syncthreads();
      A += BK; //A向右跳BK列
      B += BK * N;
      for (int i = 0; i < BK; i++) {
        tmp += As[ty * BK + i] * Bs[i * BN + tx]; //tmp += As[ty][i] * Bs[i][tx]，计算当前线程负责的 C 中元素的值，As 的第 ty 行和 Bs 的第 tx 列进行点积。注意这里的 As[ty * BK + i] 是共享内存中的元素访问方式，因为 As 是按行存储的，所以每行有 BK 个元素；Bs[i * BN + tx] 是共享内存中的元素访问方式，因为 Bs 是按行存储的，所以每行有 BN 个元素。
      }
      __syncthreads();
   }
   C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx]; //C[ty][tx] = alpha * tmp + beta * C[ty][tx]，把计算得到的结果写回全局内存中的 C 矩阵。注意这里的 C[ty * N + tx] 是全局内存中的元素访问方式，因为 C 是按行存储的，所以每行有 N 个元素。
}
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
int main() {
  std::vector<int> sizes = {1024};

  // 打开CSV文件
  std::ofstream csv_file("sgemm_benchmark_v2.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

  for (int N : sizes) {
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

    try {
      // 初始化矩阵 A 和 B
      for (int i = 0; i < N * N; ++i) {
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

      // warmup
      int warpup_time = 10;  // 热身次数
      for (int i = 0; i < warpup_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 5;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }

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

      // mysgemm_v1
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      dim3 blockDim(1024);
      dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(N, 32));

      for (int i = 0; i < warpup_time; ++i) {
        mysgemm_v2<32>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }

      cudaDeviceSynchronize();
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");

      for (int i = 0; i < repeat_time; ++i) {
        mysgemm_v2<32>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
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
      for (int i = 0; i < N * N && error_count < 10; ++i) {
        if (fabsf(C_cublas[i] - C_v1[i]) > TOL) {
          error_count++;
        }
      }

      float cublas_gflops =
          repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // GFlops
      float v1_gflops =
          repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // GFlops
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

    } catch (...) {
      std::cerr << "Out of memory or error during testing size: " << N
                << std::endl;
      out_of_memory = true;
    }

    if (!out_of_memory) {
      std::cout << "Finished size: " << N << std::endl;
    } else {
      csv_file << N << ",OOM,OOM,0" << std::endl;
    }
  }

  csv_file.close();

  std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
            << std::endl;
  return 0;
}
