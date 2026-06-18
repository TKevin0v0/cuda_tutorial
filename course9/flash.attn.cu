#include <cuda_runtime.h>
#include <stdio.h>
#include <cassert>
#include <cmath>
#include "helper.h"

#define CUDA_CHECK(condition)                                                \
    do                                                                       \
    {                                                                        \
        cudaError_t error = condition;                                       \
        if (error != cudaSuccess)                                            \
        {                                                                    \
            printf(                                                          \
                "CUDA_CHECK error in line %d of file %s \
              : %s \n",                                                      \
                __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError())); \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#ifdef DEBUG
#define DEBUG_BLOCK(expr) \
    do                    \
    {                     \
        expr              \
    } while (0)
#else
#define DEBUG_BLOCK(...) \
    do                   \
    {                    \
    } while (0)
#endif

using FP = float;
const int Br = 2;
const int Bc = 2;
const int input_seq = 4;
const int dim = 4; // 每个token的特征维度

__global__ void naive_nrow_gemm(float *A, float *B, float *C, float a, float b,
                                int M, int N, int K, int mBlock);
__global__ void row_softmax(float *input, float *output, int n);
__global__ void naive_pv(float *P, float *V, float *O, int M, int N,
                         int mBlock);

__global__ void flash_attention_v2_kernel(FP *Q, FP *K, FP *V, FP *O,
                                          int seqlen, FP smScale);

void flash_attention_v2_cuda(FP *Q, FP *K, FP *V, FP *O, int m, int n)
{
    FP sm_scale = 1.f / sqrtf(static_cast<FP>(n));
    int BS = 1;
    int HEAD = 1;
    int SEQLEN = m;
    int DIM = n;
    int Gc = 1;
    int Gr = (SEQLEN + Br - 1) / Br;
    dim3 grid = dim3(Gc, Gr);
    dim3 block = dim3(Bc, Br);
    /*
    一个ty对应一个Q
    */
    flash_attention_v2_kernel<<<grid, block>>>(Q, K, V, O, SEQLEN, sm_scale);
    DEBUG_BLOCK(printf("== v2: O ==\n"); print_device_matrix(O, SEQLEN, DIM););
}

/*
1.在 kernel 中线程布局是dim3 block(Bc, Br)，sQK的形状是(Br,Bc)，因为是Q*K^T，
所以 ty 负责 tile 内的 Q/O 行，tx 负责 tile 内的 K/V 行
2.block = dim3(Bc, Br);
3.一个block一起处理一个Q tile × 一个K/V tile。然后：每个线程负责这个attention tile里的一个元素。
4.一个Block同时处理 Br 个 Query（Q Tile），然后把整个 Key/Value 矩阵沿 sequence 方向切成多个 Bc
行的 KV Tile，逐块计算 Attention，并在线更新 softmax，最终得到这 Br 个 Query 的输出
5.
       tx=0  tx=1
ty=0   q0k0  q0k1
ty=1   q1k0  q1k1
想着这个图来理解ty和tx对q/k的含义
*/
__global__ void flash_attention_v2_kernel(FP *Q, FP *K, FP *V, FP *O, int seqlen, FP smScale)
{
    int groupSeq = (seqlen + Bc - 1) / Bc;
    int groupTx = (dim + Bc - 1) / Bc;
    int groupTy = (dim + Br - 1) / Br;
    __shared__ FP sQ[Br][dim]; // 可以看到分割后q的大小是Br,dim
    __shared__ FP sK[Bc][dim]; // k/v的大小是Bc,dim
    __shared__ FP sV[Bc][dim];
    __shared__ FP sO[Br][dim];
    __shared__ FP sQK[Br][Bc];    // Qi和Kj的点积结果，
    __shared__ FP sSafeE[Br][Bc]; // exp(score-max)的结果
    __shared__ FP sDenom[Br];     // 分母累计值
    __shared__ FP sMax[Br];       // 当前行的最大值
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = ty + blockIdx.y * blockDim.y;
    if (row >= seqlen)
        return;
    // 当前 Block 内负责的 Q 行的所有维度（dim）搬到了 sQ
    for (int i = 0; i < groupTx; i++)
    {
        sQ[ty][i * blockDim.x + tx] = Q[row * dim + i * blockDim.x + tx];
        sO[ty][i * blockDim.x + tx] = 0;
    }
    sMax[ty] = -INFINITY; // 当前Block负责的Br个Query行,每一行对应一个最大值
    sDenom[ty] = 0;       // 当前的分母
    for (int j = 0; j < groupSeq; j++)
    {
        if ((j * Bc + tx) < seqlen)
            for (int i = 0; i < groupTy; i++)
            {
                sK[tx][i * Br + ty] = K[j * Bc * dim + tx * dim + i * Br + ty];
                sV[tx][i * Br + ty] = V[j * Bc * dim + tx * dim + i * Br + ty];
            }
        __syncthreads();
        FP sum = 0.f;
        for (int i = 0; i < dim; i++)
            sum += sQ[ty][i] * sK[tx][i];
        sQK[ty][tx] = sum * smScale; // 除根号下dk
        __syncthreads();
        FP localMax = -INFINITY;
        for (int i = 0; i < Bc; i++)
            localMax = max(localMax, sQK[ty][i]);
        __syncthreads();
        FP newMax = max(sMax[ty], localMax);
        sSafeE[ty][tx] = exp(sQK[ty][tx] - newMax); // 新值直接减去当前最大值的exp
        __syncthreads();
        FP localDenom = 0.f;
        for (int i = 0; i < Bc; i++)
            localDenom += sSafeE[ty][i];
        __syncthreads();
        FP rescaleOld = exp(sMax[ty] - newMax);
        FP newDenom = sDenom[ty] * rescaleOld + localDenom;
        for (int i = 0; i < groupTx; i++)
        {
            sO[ty][i * Bc + tx] = sO[ty][i * Bc + tx] * rescaleOld;
            for (int k = 0; k < Bc; k++)
                sO[ty][i * Bc + tx] += sSafeE[ty][k] * sV[k][i * Bc + tx];
        }
        sMax[ty] = newMax;
        sDenom[ty] = newDenom;
        __syncthreads();
    }
    for (int i = 0; i < groupTx; i++)
        O[row * dim + i * Bc + tx] = sO[ty][i * Bc + tx] / sDenom[ty];
}
/*
m = seq_len
n = hidden_dim
这里是朴素实现
*/
void self_attention_cuda(float *Q, float *K, float *V, float *O, int m, int n)
{
    int mBlock = 2;
    assert(m % mBlock == 0 && "mBlock should align");
    float sm_scale = 1.f / sqrtf(static_cast<float>(n));
    float *sm_o;
    cudaMalloc((void **)&sm_o, sizeof(float) * m * m);
    dim3 qk_block(m / mBlock, 1, 1);
    naive_nrow_gemm<<<1, qk_block>>>(Q, K, sm_o, sm_scale, 0, m, m, n, mBlock);
    cudaDeviceSynchronize();
    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError()); printf("== naive QK ==\n");
                print_device_matrix(sm_o, m, m);); // 打印QK的结果
    // 一个线程处理一整行 softmax
    dim3 sm_block(m, 1, 1);
    row_softmax<<<1, sm_block>>>(sm_o, sm_o, m);
    cudaDeviceSynchronize();
    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
                printf("== naive softmax(QK) ==\n");
                print_device_matrix(sm_o, m, m););
    dim3 qkv_block(m / mBlock, 1, 1);
    naive_pv<<<1, qkv_block>>>(sm_o, V, O, m, n, mBlock);
    cudaDeviceSynchronize();
    DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
                printf("== naive softmax(QK)V ==\n");
                print_device_matrix(O, m, n););
    cudaFree(sm_o);
}

/*
对Q@K^T的朴素实现，没有加优化
A = Q
B = K
C = S = Q@K^T
a = 1
b = 0
*/
__global__ void naive_nrow_gemm(float *A, float *B, float *C, float a, float b,
                                int M, int N, int K, int mBlock)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    idx *= mBlock;                           // 每个线程处理mBlock行，所以idx要乘以mBlock，才能得到这个线程处理的行的起始位置
    for (int i = idx; i < idx + mBlock; i++) // i = C 的行号
        for (int j = 0; j < N; j++)          // j = C 的列号
        {
            float sum = 0.f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[j * K + k];
            C[i * N + j] = a * sum + b * C[i * N + j];
        }
}

// 计算 QK[M, M] @ V[M, N]
__global__ void naive_pv(float *P, float *V, float *O, int M, int N,
                         int mBlock)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // 每个线程处理mBlock行，所以idx要乘以mBlock，才能得到这个线程处理的行的起始位置
    idx *= mBlock;

    int K = M;
    // P[mBlock, M] x V[M, N] = O[mBlock, N]
    for (int i = idx; i < idx + mBlock; i++)
        for (int j = 0; j < N; j++)
        {
            float sum = 0.f;
            for (int k = 0; k < K; k++)
                sum += P[i * K + k] * V[k * N + j];
            // C[M, N]
            O[i * N + j] = sum;
        }
}

// each thread process one row of softmax
__global__ void row_softmax(float *input, float *output, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float max = -INFINITY;
    float sum = 0.f;

    // 求一行中的最大值
    for (int i = 0; i < n; i++)
        if (input[idx * n + i] > max)
            max = input[idx * n + i];
    // 计算分母的值
    for (int i = 0; i < n; i++)
    {
        output[idx * n + i] = exp(input[idx * n + i] - max);
        sum += output[idx * n + i];
    }
    // 计算最终结果
    for (int i = 0; i < n; i++)
        output[idx * n + i] /= sum;
}

void test_attention()
{
    // Q/K/V shape = [m, n]
    int m = input_seq; // token数量
    int n = dim;       // 每个token的维度
    float *h_K = new float[m * n];
    float *h_Q = new float[m * n];
    float *h_V = new float[m * n];
    float *h_O = new float[m * n];  // self_attention输出
    float *h_O2 = new float[m * n]; // flash attention v2输出
    // 遍历整个矩阵。Q/K/V 全部随机初始化成0~1随机数
    for (int i = 0; i < m * n; ++i)
    {
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;

        DEBUG_BLOCK(h_K[i] = static_cast<float>(i); h_Q[i] = static_cast<float>(i);
                    h_V[i] = static_cast<float>(i););
    }

    DEBUG_BLOCK(printf("== K ==\n"); print_host_matrix(h_K, m, n););

    float *d_K, *d_Q, *d_V, *d_O, *d_O2;
    // 分配显存
    cudaMalloc((void **)&d_K, sizeof(float) * m * n);
    cudaMalloc((void **)&d_Q, sizeof(float) * m * n);
    cudaMalloc((void **)&d_V, sizeof(float) * m * n);
    cudaMalloc((void **)&d_O, sizeof(float) * m * n);
    cudaMalloc((void **)&d_O2, sizeof(float) * m * n);

    // 把host的数据转移到device上
    cudaMemcpy(d_K, h_K, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    // 创建gpu计时器
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 开始计时
    cudaEventRecord(start, 0);
    // 运行self_attention_cuda 10次
    for (int i = 0; i < 1; i++)
    {
        self_attention_cuda(d_Q, d_K, d_V, d_O, m, n);
        CUDA_CHECK(cudaGetLastError());
    }

    // 运行flash_attention_v2_cuda 10次
    for (int i = 0; i < 1; i++)
    {
        flash_attention_v2_cuda(d_Q, d_K, d_V, d_O2, m, n);
        CUDA_CHECK(cudaGetLastError());
    }
    // 停止计时
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // 计算时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 把结果取回CPU
    cudaMemcpy(h_O, d_O, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O2, d_O2, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    // 检查selfattention和FlashAttention的结果是否接近。因为float计算存在误差，不能直接==，而是误差 < epsilon
    bool res = all_close(h_O, h_O2, m, n);
    if (res)
    {
        printf("is equal\n");
    }
    else
    {
        printf("is not equal\n");
    }
    cudaFree(d_K);
    cudaFree(d_Q);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_O2);
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    free(h_O2);
}

int main()
{
    /*
    就是循环调用10次test_attention()
    每次都会随机生成 Q/K/V，调用两种 attention来对比结果
    */
    int epoch = 10;
    for (int i = 0; i < epoch; i++)
        test_attention();
    return 0;
}