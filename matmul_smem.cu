#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define BLOCKSIZE 32
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define CHECK_CUDA(call)                                        \
do {                                                            \
    cudaError_t err = call;                                     \
    if(err != cudaSuccess){                                     \
        std::cerr << "CUDA Error: "                             \
                  << cudaGetErrorString(err)                    \
                  << " at line " << __LINE__ << std::endl;      \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

__global__ void matmul_smem(int M, int N, int K,
                             float alpha, const float *A,
                             const float *B, float beta, float *C)
{
    // which tile of C does this block own?
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // where inside that tile does this thread sit?
    const uint threadRow = threadIdx.x / BLOCKSIZE;
    const uint threadCol = threadIdx.x % BLOCKSIZE;

    // shared memory tiles for A and B
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // slide pointers to the starting position of this block's region
    A += cRow * BLOCKSIZE * K;       // move down to our row-strip of A
    B += cCol * BLOCKSIZE;           // move right to our col-strip of B
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;  // our tile in C

    float tmp = 0.0f;

    // march across K in steps of BLOCKSIZE
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {

        // each thread loads one element of A and one of B into SMEM
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // wait until every thread has finished loading
        __syncthreads();

        // advance pointers to next tile
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // dot product on the cached tile — all reads from fast SMEM
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; dotIdx++) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }

        // wait until every thread is done computing before
        // the next iteration overwrites the shared memory
        __syncthreads();
    }

    // write final result — only once to global memory
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

int main()
{
    int M = 1024;
    int N = 1024;
    int K = 1024;
    float alpha = 1.0f;
    float beta  = 0.0f;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    std::vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), sizeC, cudaMemcpyHostToDevice));

    // 1024 threads per block in a flat 1D layout
    // threadRow = threadIdx.x / 32 → which row inside the tile
    // threadCol = threadIdx.x % 32 → which col inside the tile
    dim3 block(BLOCKSIZE * BLOCKSIZE);
    dim3 grid(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));    
    matmul_smem<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop));    
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));

    // verify: every element of C should equal K (1024)
    // because A and B are all 1s, so each dot product = K*1*1 = K
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (std::fabs(h_C[i] - float(K)) > 1e-3f) {
            correct = false;
            std::cerr << "Mismatch at i=" << i
                      << " got=" << h_C[i]
                      << " expected=" << float(K) << std::endl;
            break;
        }
    }

    std::cout << (correct ? "Result correct." : "Result WRONG.") << std::endl;
    std::cout << "Kernel time : " << ms << " ms" << std::endl;

    // 2 flops per element (1 mul + 1 add), M*N*K elements
    float gflops = (2.0f * M * N * K) / (ms * 1e6f);
    std::cout << "GFLOPS      : " << gflops << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}