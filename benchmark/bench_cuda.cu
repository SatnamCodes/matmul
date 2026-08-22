// Sweeps the naive kernel (naive.cu) and cuBLAS SGEMM across sizes and
// prints one CSV row per (impl, size) to stdout:
//   impl,M,N,K,ms,gflops,correct
//
// Build:
//   nvcc -O3 -arch=sm_89 benchmark/bench_cuda.cu -lcublas -o benchmark/bench_cuda
// Run:
//   ./benchmark/bench_cuda > benchmark/results_cuda.csv

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line "  \
                << __LINE__ << std::endl;                                    \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t st = call;                                                \
    if (st != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLAS Error: " << st << " at line " << __LINE__         \
                << std::endl;                                                \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// Same mapping as naive.cu: row-major A (M x K), B (K x N), C (M x N).
__global__ void matmul_naive(int M, int N, int K, float alpha,
                              const float *A, const float *B, float beta,
                              float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < M && y < N) {
    float temp = 0.0f;
    for (int i = 0; i < K; ++i) {
      temp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * temp + beta * C[x * N + y];
  }
}

struct Result {
  double ms;
  double gflops;
  bool correct;
};

// Times `iters` launches of `launch`, after `warmup` untimed launches.
// Verifies C == K everywhere (inputs are all-ones, alpha=1, beta=0).
template <typename LaunchFn>
Result time_gemm(LaunchFn launch, int M, int N, int K, float *d_C,
                  std::vector<float> &h_C, int warmup = 3, int iters = 10) {
  for (int i = 0; i < warmup; i++) launch();
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) launch();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double ms = total_ms / iters;
  double gflops = (2.0 * M * N * K) / (ms * 1e6);

  CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
  bool correct = true;
  for (size_t i = 0; i < h_C.size(); i++) {
    if (h_C[i] != static_cast<float>(K)) {
      correct = false;
      break;
    }
  }
  return {ms, gflops, correct};
}

int main() {
  std::vector<int> sizes = {256, 512, 1024, 2048, 4096};

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  for (int n : sizes) {
    int M = n, N = n, K = n;
    float alpha = 1.0f, beta = 0.0f;

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    std::vector<float> h_A(M * K, 1.0f), h_B(K * N, 1.0f), h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice));

    // ---- naive kernel ----
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    auto naive_launch = [&]() {
      matmul_naive<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    };
    Result r_naive = time_gemm(naive_launch, M, N, K, d_C, h_C);
    std::cout << "naive," << M << "," << N << "," << K << "," << r_naive.ms
               << "," << r_naive.gflops << "," << (r_naive.correct ? 1 : 0)
               << std::endl;

    // ---- cuBLAS SGEMM ----
    // cuBLAS is column-major. Computing C'(N x M, col-major) = B' * A'
    // is equivalent to row-major C(M x N) = A * B.
    auto cublas_launch = [&]() {
      CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                &alpha, d_B, N, d_A, K, &beta, d_C, N));
    };
    Result r_cublas = time_gemm(cublas_launch, M, N, K, d_C, h_C);
    std::cout << "cublas," << M << "," << N << "," << K << "," << r_cublas.ms
               << "," << r_cublas.gflops << "," << (r_cublas.correct ? 1 : 0)
               << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  }

  cublasDestroy(handle);
  return 0;
}
