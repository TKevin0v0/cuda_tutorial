#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

// cpu版本的实现
void softmax_forward_cpu(float *out, const float *inp, int N, int C)
{
    for (int i = 0; i < N; i++)
    {
        float maxval = -INFINITY;
        for (int j = 0; j < C; j++)
            maxval = std::max(maxval, inp[i * C + j]);
        float sum = 0.f;
        for (int j = 0; j < C; j++)
            sum += expf(inp[i * C + j] - maxval);
        float norm = 1.f / sum;
        for (int j = 0; j < C; j++)
            out[i * C + j] = expf(inp[i * C + j] - maxval) * norm;
    }
}

// 最普通的gpu实现
// 每一行用block的一个线程来处理
__global__ void softmax_forward_kernel1(float *out, const float *inp, int N,
                                        int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float maxval = -INFINITY;
        for (int j = 0; j < C; j++)
            maxval = std::max(maxval, inp[i * C + j]);
        float sum = 0.f;

        for (int j = 0; j < C; j++)
        {
            float val = expf(inp[i * C + j] - maxval);
            out[i * C + j] = val;
            sum += val;
        }
        float norm = 1.f / sum;
        for (int j = 0; j < C; j++)
            out[i * C + j] *= norm;
    }
}

/*
优化1：用block中的多个线程处理一行
求最大值和求和都用共享显存和树形归约的思想
*/
__global__ void softmax_forward_kernel2(float *out, const float *inp, int N, int C)
{
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    float maxval = -INFINITY;
    int block_size = blockDim.x;
    for (int i = tid; i < C; i += block_size)
        maxval = fmaxf(maxval, inp[blockIdx.x * C + i]);
    shared[tid] = maxval;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1)
    {
        __syncthreads();
        if (tid < stride)
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
    float offset = shared[0];
    for (int i = tid; i < C; i += block_size)
        out[blockIdx.x * C + i] = expf(inp[blockIdx.x * C + i] - offset);
    __syncthreads();
    float sumval = 0.f;
    for (int i = tid; i < C; i += block_size)
        sumval += out[blockIdx.x * C + i];
    shared[tid] = sumval;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1)
    {
        __syncthreads();
        if (tid < strid)
            shared[tid] += shared[tid + stride];
    }
    __syncthreads();
    float sum = shared[0];
    for (int i = tid; i < C; i += block_size)
        out[blockIdx.x * C + i] = out[blockIdx.x * C + i] / sum;
}

/*
优化2：利用warp进行归约。这里的代码只能适用于block中线程有32个，因为没有比较各个warp之间的最大值
*/
__global__ void warpReduceMax(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void softmax_forward_kernel3(float *out, const float *inp, int N,
                                        int C)
{
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = inp + idx * C;
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x)
        maxval = fmaxf(maxval, x[i]);
    maxval = warpReduceMax(maxval);
    float offset = __shfl_sync(0xFFFFFFFF, maxval, 0); // 把warp中lane0的maxval广播给整个 warp
    for (int i = tid; i < C; i += blockDim.x)
        out[idx * C + i] = expf(x[i] - offset);
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x)
        sumval += x[i];
    sumval = warpReduceSum(sumval);
    float sum = __shfl_sync(0xFFFFFFFF, sumval, 0);
    for (int i = tid; i < C; i += blockDim.x)
        out[idx * C + i] = x[i] / sum;
}

/*
优化3：因为优化2只有32个线程，每个线程处理的数据多。所以这里会增加多个线程，从而有
多个warp。于是实现warp内用shuffle归约，warp之间用shared memory归约
*/
__global__ void softmax_forward_kernel4(float *out, const float *inp, int N,
                                        int C)
{
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpsPerBlock = blockDim.x / 32;
    const float *x = inp + idx * C;
    float *maxvals = shared;
    float *sumvals = &shared[warpsPerBlock]; // 紧紧挨着maxvals后面的地址用来存储每个warp的sum
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x)
        maxval = fmaxf(maxval, x[i]);
    maxval = warpReduceMax(maxval);
    // 把每个warp的最大值放入shared memory 中
    if (laneId == 0)
        maxvals[warpId] = maxval;
    __syncthreads();
    // 让 block 中线程0做最后一次归约
    if (tid == 0)
    {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++)
            val = fmaxf(val, maxvals[i]);
        maxvals[0] = val;
    }
    __syncthreads();
    float offset = maxvals[0];
    for (int i = tid; i < C; i += blockDim.x)
        out[idx * C + i] = expf(x[i] - offset);
    float sumval = 0.f;
    for (int i = tid; i < C; i += blockDim.x)
        sumval += x[i];
    sumval = warpReduceSum(sumval);
    if (laneId == 0)
        sumvals[warpId] = sumval;
    __syncthreads();
    if (tid == 0)
    {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; i++)
            val += sumvals[i];
        sumvals[0] = val;
    }
    __syncthreads();
    for (int i = tid; i < C; i += blockDim.x)
        out[idx * C + i] = out[idx * C + i] / sumvals[0];
}

int main()
{
    // 输入是一个N*C的矩阵，输出也是一个N*C的矩阵
    int N = 32;
    int C = 4096;
    float *inp = (float *)malloc(N * C * sizeof(float));
    float *out_cpu = (float *)malloc(N * C * sizeof(float));
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            inp[n * C + c] = static_cast<float>(c);
    // cpu版本的实现
    auto start_cpu = std::chrono::high_resolution_clock::now();
    softmax_forward_cpu(out_cpu, inp, N, C);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // gpu版本的实现
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float *d_out, *d_inp;
    cudaMalloc((void **)&d_out, N * C * sizeof(float));
    cudaMalloc((void **)&d_inp, N * C * sizeof(float));
    cudaMemcpy(d_inp, inp, N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    int blockSize = 128;
    int numBlocks = N;
    softmax_forward_kernel2<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    cudaMemcpy(out_gpu, d_out, N * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_inp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    bool success = compare_results(out_cpu, out_gpu, N, C);
    std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time.count() / (gpu_time_ms)) << "x"
              << std::endl;
    free(inp);
    free(out_cpu);
    free(out_gpu);

    return 0;
}
