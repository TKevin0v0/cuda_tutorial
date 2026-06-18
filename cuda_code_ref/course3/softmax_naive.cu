#include <chrono>   // for timing
#include <cmath>    // for INFINITY
#include <cstdlib>  // for malloc/free
#include <iostream>

// CPU implementation
void softmax_forward_cpu(float *out, const float *inp, int N, int C) 
{
  for (int i = 0; i < N; i++) 
  {
    const float *inp_row = inp + i * C;
    float *out_row = out + i * C;

    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) 
      if (inp_row[j] > maxval) 
        maxval = inp_row[j];
    float sum = 0.f;
    for (int j = 0; j < C; j++) 
    {
      out_row[j] = expf(inp_row[j] - maxval);
      sum += out_row[j];
    }
    float norm = 1.f / sum;
    for (int j = 0; j < C; j++) 
      out_row[j] *= norm;
    
  }
}


__global__ void softmax_forward_kernel1(float *out, const float *inp, int N,
                                        int C) 
 {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    const float *inp_row = inp + i * C;
    float *out_row = out + i * C;

    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
      if (inp_row[j] > maxval) {
        maxval = inp_row[j];
      }
    }
    float sum = 0.f;
    for (int j = 0; j < C; j++) {
      out_row[j] = expf(inp_row[j] - maxval);
      sum += out_row[j];
    }
    for (int j = 0; j < C; j++) {
      out_row[j] /= (float)sum;
    }
  }
}

bool compare_results(const float *cpu, const float *gpu, int N, int C,
                     float epsilon = 1e-3f) 
 {
  for (int i = 0; i < N * C; ++i) 
  {
    if (fabs(cpu[i] - gpu[i]) > epsilon) 
    {
      std::cout << "Difference at index " << i << ": CPU=" << cpu[i]
                << ", GPU=" << gpu[i] << ", diff=" << fabs(cpu[i] - gpu[i])
                << std::endl;
      return false;
    }
  }
  return true;
}

__global__ void softmax_forward_kernel2(float *out, const float *inp, int N,
                                        int C) 
{
  extern __shared__ float shared[];
  int idx = blockIdx.x;  
  int tid = threadIdx.x;  
  int block_size = blockDim.x;
  const float *x = inp + idx * C; 
  float maxval = -INFINITY;
  for (int i = tid; i < C; i += block_size) 
    maxval = fmaxf(maxval, x[i]);
  shared[tid] = maxval;
  __syncthreads();
  for (int stride = block_size / 2; stride >= 1; stride /= 2) 
  {
    __syncthreads();
    if (tid < stride) 
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
  }
  __syncthreads();
  float offset = shared[0];
  for (int i = tid; i < C; i += block_size) 
    out[idx * C + i] = expf(x[i] - offset);
  __syncthreads();

  x = out + idx * C; 
  float sumval = 0.0f;
  for (int i = tid; i < C; i += block_size) 
    sumval += x[i];
  shared[tid] = sumval;
  __syncthreads();
  for (int stride = block_size / 2; stride >= 1; stride /= 2)
  {
    __syncthreads();
    if (tid < stride) 
      shared[tid] += shared[tid + stride];
    
  }
 
  __syncthreads();
  float sum = shared[0];
  for (int i = tid; i < C; i += block_size) 
    out[idx * C + i] = x[i] / sum;
  
}

__device__ float warpReduceMax(float val) 
{
  for (int offset = 16; offset > 0; offset /= 2) 
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  return val;
}

__device__ float warpReduceSum(float val) 
{
  for (int offset = 16; offset > 0; offset /= 2) 
  {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
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
  float offset = __shfl_sync(0xFFFFFFFF, maxval, 0);
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

__global__ void softmax_forward_kernel4(float *out, const float *inp, int N,
                                        int C) 
{

  extern __shared__ float shared[];
  int idx = blockIdx.x;
  int tid = threadIdx.x;
  int warpId = threadIdx.x / 32; 
  int laneId = threadIdx.x % 32; 
  int warpsPerBlock = blockDim.x / 32;
  float *maxvals = shared;
  float *sumvals = &shared[warpsPerBlock];
  const float *x = inp + idx * C;
  float maxval = -INFINITY;
  for (int i = tid; i < C; i += blockDim.x) 
    maxval = fmaxf(maxval, x[i]);
  maxval = warpReduceMax(maxval);

  if (laneId == 0) maxvals[warpId] = maxval;
  __syncthreads();
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
  x = out + idx * C;
  float sumval = 0.0f;
  for (int i = tid; i < C; i += blockDim.x) 
    sumval += x[i];
  sumval = warpReduceSum(sumval);
  if (laneId == 0) 
    sumvals[warpId] = sumval;
  __syncthreads();
  if (tid == 0) 
  {
    float val = sumvals[tid];
    for (int i = 1; i < warpsPerBlock; ++i) 
      val += sumvals[i];
    sumvals[0] = val;
  }
  __syncthreads();
  float sum = sumvals[0];
  for (int i = tid; i < C; i += blockDim.x) 
    out[idx * C + i] = x[i] / sum;
  
}

int main() 
{
  int N = 32;
  int C = 4096;

  size_t num_elements = N * C;
  float *inp = (float *)malloc(num_elements * sizeof(float));
  float *out_cpu = (float *)malloc(num_elements * sizeof(float));
  float *out_gpu = (float *)malloc(num_elements * sizeof(float));

  for (int n = 0; n < N; ++n) 
    for (int c = 0; c < C; ++c) 
      inp[n * C + c] = float(c);
    
  auto start_cpu = std::chrono::high_resolution_clock::now();
  softmax_forward_cpu(out_cpu, inp, N, C);
  auto end_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float *d_out, *d_inp;
  cudaMalloc((void **)&d_out, N * C * sizeof(float));
  cudaMalloc((void **)&d_inp, N * C * sizeof(float));
  cudaMemcpy(d_inp, inp, N * C * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start);
  // Launch kernel
  int blockSize = 128;
  int numBlocks = N;
  softmax_forward_kernel2<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);
  cudaEventRecord(stop);

  // Wait for the event to complete
  cudaEventSynchronize(stop);

  // Calculate milliseconds
  float gpu_time_ms = 0;
  cudaEventElapsedTime(&gpu_time_ms, start, stop);

  // Copy result back to host
  cudaMemcpy(out_gpu, d_out, N * C * sizeof(float), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_out);
  cudaFree(d_inp);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Compare results
  bool success = compare_results(out_cpu, out_gpu, N, C);
  std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;

  // Print performance comparison
  std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
  std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
  std::cout << "Speedup: " << (cpu_time.count() / (gpu_time_ms)) << "x"
            << std::endl;

  // Cleanup
  free(inp);
  free(out_cpu);
  free(out_gpu);

  return 0;
}
