#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>

const int BLOCK_SIZE = 1024;
const int N = 1024 * 1024;  // 1M elements

/*
优化点：
1. 一个线程处理两个元素(预归约)
2. 使用 shared memory 做 block reduction
3. 支持任意M
4. kernel 可复用多轮 reduction
5.不再依赖：固定 N,固定 reduction 轮数,固定 block 数
*/
__global__ void reduce_v1(float *g_idata, float *g_odata, int n)
{
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    // 每个 block 处理 2 * blockDim.x 个元素
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    // 寄存器局部累加
    float sum = 0.0f;
    if(i < n)
        sum += g_idata[i];
    if(i + blockDim.x < n)
        sum += g_idata[i + blockDim.x];
    // 写入 shared memory
    sdata[tid] = sum;
    __syncthreads();
    // block 内 reduction
    for(unsigned int s = 1;s < blockDim.x; s *= 2)
    {
        if(tid % (2 * s) == 0)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if(tid == 0)
        g_odata[blockIdx.x] = sdata[0];
    
}

// CPU 验证
float reduce_cpu(const std::vector<float> &data)
{
    float sum = 0.0f;
    for(float val : data)
        sum += val;
    return sum;
}

int main()
{
    std::vector<float> h_data(N);
    for(int i = 0; i < N; i++)
        h_data[i] = 1.0f;
    // CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_result = reduce_cpu(h_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU result: "<< cpu_result << std::endl;
    std::cout << "CPU time: "<< cpu_duration.count()<< " ms" << std::endl;
    float *d_data;
    float *d_result;
    float *d_temp;
    cudaMalloc(&d_data, N * sizeof(float));
    // 最多第一轮需要 N/(2*BLOCK_SIZE)
    int max_blocks =(N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    cudaMalloc(&d_result, max_blocks * sizeof(float));
    cudaMalloc(&d_temp, max_blocks * sizeof(float));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    // GPU 运行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int cur_n = N;
    float *input = d_data;
    float *output = d_result;
    while(cur_n > 1)
    {
        int num_blocks =(cur_n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
        reduce_v1<<<num_blocks, BLOCK_SIZE>>>(input, output, cur_n);
        cur_n = num_blocks;
        float *tmp = input;
        input = output;
        output = tmp;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "GPU kernel time: "<< milliseconds<< " ms" << std::endl;
    // 最终结果在 input[0]
    float gpu_result = 0.0f;
    cudaMemcpy(&gpu_result,input, sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "GPU result: "<< gpu_result << std::endl;
    if(abs(cpu_result - gpu_result) < 1e-5)
        std::cout<< "Result verified successfully!"<< std::endl;
    else
        std::cout<< "Result verification failed!"<< std::endl;
    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}