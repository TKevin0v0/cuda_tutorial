#include <cuda_runtime.h>
#include <chrono>  // 用于 CPU 计时
#include <iostream>
#include <numeric>
#include <vector>

const int BLOCK_SIZE = 1024;
const int N = 1024 * 1024;  // 1M elements
#define FULL_MASK 0XFFFFFFFF
//向上取整最近的2的幂数
unsigned int nextPow2(unsigned int x)
{
    x--;

    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;

    return x + 1;
}

// CPU验证函数
float reduce_cpu(const std::vector<float> &data) 
{
  float sum = 0.0f;
  for (float val : data) 
    sum += val;
  return sum;
}

__device__ __forceinline__ float warpReduce(float val)
{
    for(int i = 16; i >= 1; i >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, i);
    return val;
}

/*
优化3：
1.避免线程分化，避免让warp里面的线程执行不同的指令
2.可以看到这里的归约stride是从大到小，之前是从小到大
3.只要 s ≥ 32，整个 warp 里的 32 个线程要么全部满足 tid < s，要么全部不满足，完全消除 warp divergence；
只有当 s ≤ 31 时，才可能出现同一 warp 内部分线程执行、部分线程空闲的情况。
*/
__global__ void reduce_v3(float *g_idata, float *g_odata, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;
    float sum = 0.0f;
    if(i < n)
        sum += g_idata[i];
    if(i + blockDim.x < n)
        sum += g_idata[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    for(int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if(tid < 32)
    {
        float val = sdata[tid];
        // 只有blockDim >= 64时,才能读取tid+32
        if(blockDim.x >= 64)
            val += sdata[tid + 32];
        val = warpReduce(val);
        if(tid == 0)
            g_odata[blockIdx.x] = val;
    }
    if(tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

int main()
{
    int max_blocks = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    std::vector<float> h_data(N);
    for (int i = 0; i < N; i++) 
      h_data[i] = 1.0f;  // 简单起见，全部初始化为1.0
    // -------------------------------
    // CPU 计时开始
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_result = reduce_cpu(h_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    // CPU 计时结束
    // -------------------------------
    std::cout << "CPU result: " << cpu_result << std::endl;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;
    float *d_data, *d_result;
    float gpu_result;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, max_blocks * sizeof(float));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    // -------------------------------
    // GPU 计时开始 (CUDA Events)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int cur_n = N;
    float *input = d_data;
    float *output = d_result;
    while(cur_n > 1)
    {
        //计算“需要多少线程”并且向上取到最近的 2 的幂, 动态计算需要的线程数，防止大量闲置线程
        int threadsnum = min(nextPow2((cur_n+1)/2), BLOCK_SIZE); 
        int num_blocks = (cur_n + threadsnum*2 - 1) / (threadsnum * 2);
        reduce_v3<<<num_blocks, threadsnum, threadsnum * sizeof(float)>>>(input, output, cur_n);
        cur_n = num_blocks;
        float *tmp = input;
        input = output;
        output = tmp;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // GPU 计时结束
    // -------------------------------
    std::cout << "GPU kernel time: " << milliseconds << " ms" << std::endl;
    cudaMemcpy(&gpu_result, input, sizeof(float),
               cudaMemcpyDeviceToHost);
    std::cout << "GPU result: " << gpu_result << std::endl;
    if (abs(cpu_result - gpu_result) < 1e-5) 
      std::cout << "Result verified successfully!" << std::endl;
    else 
      std::cout << "Result verification failed!" << std::endl;
    // 清理资源
    cudaFree(d_data);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
