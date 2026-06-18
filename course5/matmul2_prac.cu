#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>   // for fabsf
#include <fstream> // for CSV output
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


注意：一个 block 对应一个 tile
BM表示C tile有多少行，同时A tile 有多少行。
BN表示C tile有多少列，同时B tile有多少列。
BK表示每次在 K 维度切多少长度，也就是最终结果相关的A取多少列，B取多少行
*/
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v4(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // 线程相关部分
    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread;
    int ty = (threadIdx.x / block_row_thread) * TM;
    int tx = (threadIdx.x % block_row_thread) * TN;
    // 共享内存
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    // 移动指针到block要处理的起始位置
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];
    // 求搬运数据到共享内存的初始值
    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;
    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;
    float tmp[TM][TN] = {0.f};
    for (int i = 0; i < K; i += BK)
    {
        for (int j = 0; j < BM; j += a_tile_stride)
            As[(a_tile_row + j) * BK + a_tile_col] = A[(a_tile_row + j) * K + a_tile_col];
        for (int j = 0; j < BK; j += b_tile_stride)
            Bs[(b_tile_row + j) * BN + b_tile_col] = B[(b_tile_row + j) * N + b_tile_col];
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int j = 0; j < BK; j++)
            for (int k = 0; k < TM; k++)
                for (int l = 0; l < TN; l++)
                    tmp[k][l] += As[(ty + k) * BK + j] * Bs[tx + l + j * BN];
        __syncthreads();
    }
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            C[(ty + i) * N + tx + j] = alpha * tmp[i][j] + beta * C[(ty + i) * N + tx + j];
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
std::vector<int> generateSizes() { return {4096}; }
int main()
{
    int device_id = 7;
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

            // warmup
            int warpup_time = 10; // 热身次数
            for (int i = 0; i < warpup_time; ++i)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                             &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                                 "cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            // cuBLAS SGEMM
            int repeat_time = 5;
            checkCudaError(cudaEventRecord(start),
                           "cudaEventRecord(start cublas) failed");
            for (int i = 0; i < repeat_time; ++i)
            {
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

            // mysgemm_v4
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            dim3 blockDim(256);
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            for (int i = 0; i < warpup_time; ++i)
            {
                mysgemm_v4<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }

            cudaDeviceSynchronize();
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            checkCudaError(cudaEventRecord(start),
                           "cudaEventRecord(start v1) failed");

            for (int i = 0; i < repeat_time; ++i)
            {
                mysgemm_v4<128, 128, 8, 8, 8>
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
