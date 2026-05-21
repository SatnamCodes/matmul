# CUDA GEMM Journey

A beginner-friendly CUDA matrix multiplication project, starting from a naive GEMM kernel and moving toward memory-aware GPU programming.

This project was built to understand:

- CUDA kernel execution
- thread, block, and grid hierarchy
- GPU memory management
- kernel timing and benchmarking
- FLOPS measurement
- CUDA profiling basics
- global memory access patterns

## GEMM Equation

The kernel computes:

$$
C = \alpha AB + \beta C
$$

where:

- $A$ is an $M \times K$ matrix
- $B$ is a $K \times N$ matrix
- $C$ is an $M \times N$ matrix

The current programs use square matrices of size `1024 x 1024` and launch one CUDA thread per output element.

## Implementations

### 1. Naive GEMM

File: [naive.cu](naive.cu)

The naive version uses a `16 x 16` thread block. Each thread computes one element of `C`.

```cpp
const uint x = blockIdx.x * blockDim.x + threadIdx.x;
const uint y = blockIdx.y * blockDim.y + threadIdx.y;
```

That mapping is easy to understand, which is why it is a good first CUDA kernel. Every thread loops across `K`, reads one row from `A`, one column from `B`, accumulates the dot product, and writes one output value.

### 2. Memory-Coalesced GEMM

File: [mem_coal_#2.cu](mem_coal_#2.cu)

This version keeps the same mathematical work, but changes how threads inside a block map to output elements.

```cpp
#define BLOCKSIZE 32

const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / 32);
const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % 32);
```

The block has `32 * 32 = 1024` threads. Instead of using a 2D thread block, it uses a 1D block and manually converts `threadIdx.x` into a row and column inside a `32 x 32` output tile.

The important part is `threadIdx.x % 32`. Neighboring threads in a warp get neighboring `y` values, so when they read from matrix `B`:

```cpp
B[i * N + y]
```

they read consecutive memory locations. That is the memory-coalescing idea: make nearby threads access nearby addresses so the GPU can combine memory transactions more efficiently.

This is not the final fast GEMM yet. It still does not use shared memory tiling, register blocking, warp tiling, or Tensor Cores. But it is a useful next step because it shows that performance is not only about "more threads"; it is also about how those threads touch memory.

## Features

- CUDA-based matrix multiplication kernel
- 2D thread and block indexing
- 1D thread block mapped into a 2D output tile
- coalesced global reads from matrix `B`
- Host-to-device and device-to-host memory transfers
- CUDA event timing
- GFLOPS calculation
- Result correctness verification
- Ready for Nsight profiling

## CUDA Concepts Used

- `__global__` kernels
- `cudaMalloc`
- `cudaMemcpy`
- CUDA events
- thread indexing
- block/grid configuration
- GPU memory management
- naive GEMM computation

## How The Kernel Works

The kernel in [naive.cu](naive.cu) computes one output element per thread. Each thread:

1. Finds its row and column in the output matrix.
2. Loops over the shared dimension `K`.
3. Accumulates the dot product for that output element.
4. Writes the final value back to device memory.

The matrices are initialized with ones, so the expected result is easy to check: every element of `C` should become `K` when `alpha = 1` and `beta = 0`.

The kernel in [mem_coal_#2.cu](mem_coal_#2.cu) does the same computation, but changes thread layout:

1. Each block covers a `32 x 32` tile of the output matrix.
2. `threadIdx.x / 32` chooses the row inside that tile.
3. `threadIdx.x % 32` chooses the column inside that tile.
4. Threads in the same warp read consecutive values from `B`.
5. Each thread still computes one complete dot product for one output cell.

## Build

Naive version:

```bash
nvcc -O3 naive.cu -o naive
```

Memory-coalesced version:

```bash
nvcc -O3 'mem_coal_#2.cu' -o mem_coal
```

## Run

Naive version:

```bash
./naive
```

Memory-coalesced version:

```bash
./mem_coal
```

## Example Output

```text
Result is correct.
Kernel Time : 12.8 ms
GFLOPS: 167.6
```

## What I Measured For Naive GEMM

I ran the program 6 times and each run produced the correct result.

| Run | Kernel Time (ms) | GFLOPS |
| --- | ---: | ---: |
| 1 | 14.4626 | 148.485 |
| 2 | 12.8091 | 167.654 |
| 3 | 11.9716 | 179.381 |
| 4 | 10.8335 | 198.226 |
| 5 | 11.2482 | 190.918 |
| 6 | 10.8786 | 197.404 |

Average kernel time: 12.0339 ms

Average throughput reported by the program: 180.3447 GFLOPS

Fastest run: 10.8335 ms

Slowest run: 14.4626 ms

## Memory-Coalesced Run Status And Profiling

I compiled the memory-coalesced implementation successfully with:

```bash
nvcc -O3 'mem_coal_#2.cu' -o mem_coal
```

I also compiled with ptxas verbosity:

```bash
nvcc -O3 -Xptxas=-v 'mem_coal_#2.cu' -o mem_coal
```

Compiler resource output:

```text
ptxas info    : Compiling entry function '_Z16matmul_coalescediiifPKfS0_fPf' for 'sm_52'
ptxas info    : Function properties for _Z16matmul_coalescediiifPKfS0_fPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 31 registers, 368 bytes cmem[0]
```

That means this version is not spilling registers to local memory. The likely bottleneck is memory traffic and missing data reuse, not register pressure.

Benchmarks were collected on:

```text
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Driver: 580.159.03
CUDA runtime reported by nvidia-smi: 13.0
```

Both kernels produced correct results. I ran each implementation 6 times and used the repeated runs for comparison.

### Naive vs Memory-Coalesced Benchmark

| Implementation | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 | Avg Time | Avg GFLOPS | Fastest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `naive.cu` | 19.9474 ms | 13.0193 ms | 13.2733 ms | 13.1747 ms | 10.2245 ms | 10.2201 ms | 13.3099 ms | 169.592 | 10.2201 ms |
| `mem_coal_#2.cu` | 3.6273 ms | 3.0635 ms | 2.6418 ms | 2.6412 ms | 2.6358 ms | 2.5423 ms | 2.8586 ms | 763.074 | 2.5423 ms |

Average speedup:

```text
13.3099 / 2.8586 = 4.66x faster
```

Best-run speedup:

```text
10.2201 / 2.5423 = 4.02x faster
```

This indexing change made a real difference. The kernel is still not fully optimized GEMM, but memory access order alone gave a clear improvement.

### Profiling Commands

```bash
nsys profile --trace=cuda -o mem_coal_nsys ./mem_coal
ncu --target-processes all --set full ./mem_coal
```

Nsight Systems generated a CUDA trace:

```text
mem_coal_nsys.qdstrm
```

For deeper Nsight Compute metrics, run:

```bash
sudo ncu --target-processes all --set full ./mem_coal
sudo ncu --target-processes all --set full ./naive
```

The main metrics to compare are memory throughput, L1/TEX cache behavior, achieved occupancy, eligible warps per scheduler, and stall reasons.

## Bottleneck Analysis For `mem_coal_#2.cu`

The current memory-coalesced kernel improves one important thing: neighboring threads in a warp read neighboring elements from `B`.

```cpp
const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / 32);
const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % 32);
```

For `threadIdx.x = 0..31`, all threads have the same `x` and consecutive `y` values. Inside the loop:

```cpp
temp += A[x * K + i] * B[i * N + y];
```

that gives:

- `B[i * N + y]`: coalesced across the warp
- `A[x * K + i]`: same address across the warp, so it can be broadcast/cached
- `C[x * N + y]`: coalesced store across the warp

That is a good step, but the kernel is still bottlenecked by these issues:

1. No shared memory tiling

   Every output element loops through all `K` values. Data from `A` and `B` is reused mathematically, but the kernel does not stage tiles in shared memory, so it keeps going back to global memory.

2. Repeated `B` loads across warps

   A `32 x 32` block has 32 warps. For a fixed `i`, each warp works on a different row but the same 32 columns, so many warps need the same `B[i * N + y]` values. Without shared memory, those values are loaded repeatedly.

3. Very large block size

   `dim3 block(32 * 32)` launches 1024 threads per block, which is the maximum block size on many NVIDIA GPUs. This can reduce scheduling flexibility because each SM can keep fewer blocks resident.

4. Low arithmetic intensity compared with tiled GEMM

   The kernel does `2 * M * N * K` FLOPs, but it does not reuse global memory aggressively. A tiled GEMM increases arithmetic intensity by loading a tile of `A` and `B` once, then using it for many multiply-adds.

5. No Tensor Core path

   This is FP32 scalar math. It does not use WMMA/Tensor Cores, so it will be far from cuBLAS-level throughput on modern NVIDIA GPUs.

What to inspect in Nsight Compute on a CUDA-enabled machine:

- `Memory Workload Analysis`: global load efficiency, L1/TEX hit rate, L2 hit rate
- `Speed Of Light`: whether the kernel is memory-bound or compute-bound
- `Launch Statistics`: block size, waves per SM, occupancy
- `Scheduler Statistics`: eligible warps per cycle and stall reasons
- `Source Counters`: which source lines cause the most stalls

Expected result: this kernel should look better than the naive version for coalescing, but it should still be dominated by global memory traffic until shared memory tiling is added.

## Profiling

### Nsight Systems

```bash
nsys profile ./naive
```

For the memory-coalesced version:

```bash
nsys profile ./mem_coal
```

I also ran `nsys profile --trace=cuda -o nsys_naive ./naive`, which generated a `.qdstrm` trace file.

### Nsight Compute

```bash
sudo ncu ./naive
```

For the memory-coalesced version:

```bash
sudo ncu ./mem_coal
```

Exact terminal output from the Nsight Compute run:

```text
==PROF== Connected to process 84845 (/home/totallynotsatnam/CS/GPU/matmul/naive)
==PROF== Profiling "matmul_naive" - 0: 0%....50%....100% - 9 passes
Result is correct.
Kernel Time : 484.525 ms
GFLOPS: 4.43214
==PROF== Disconnected from process 84845
[84845] naive@127.0.0.1
  matmul_naive(int, int, int, float, const float *, const float *, float, float *) (64, 64, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- -------------
    Metric Name               Metric Unit  Metric Value
    ----------------------- ------------- -------------
    DRAM Frequency          cycle/nsecond          7.99
    SM Frequency            cycle/nsecond          1.54
    Elapsed Cycles                  cycle    24,150,253
    Memory Throughput                   %         98.62
    DRAM Throughput                     %          0.42
    Duration                      msecond         15.64
    L1/TEX Cache Throughput             %         98.87
    L2 Cache Throughput                 %          4.88
    SM Active Cycles                cycle 24,091,127.96
    Compute (SM) Throughput             %         23.20

    INF   The kernel is utilizing greater than 80.0% of the available compute or memory performance of the device. To   
          further improve performance, work will likely need to be shifted from the most utilized to another unit.      
          Start by analyzing L1 in the Memory Workload Analysis section.                                                

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit        Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                  4,096
    Registers Per Thread             register/thread              36
    Shared Memory Configuration Size           Kbyte           16.38
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    Threads                                   thread       1,048,576
    Waves Per SM                                               28.44

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           24
    Block Limit Registers                 block            6
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        98.81
    Achieved Active Warps Per SM           warp        47.43

		INF   This kernel's theoretical occupancy is not impacted by any block limit.
```

## Performance Notes

The naive implementation is intentionally simple:

- no shared memory tiling
- repeated global memory accesses
- limited memory reuse

As a result, the kernel is memory-bandwidth limited and achieves only a fraction of the GPU's theoretical peak FLOPS.

The memory-coalesced implementation improves the thread-to-data mapping for reads from `B`, but it is still not a fully optimized GEMM. It is an intermediate learning step before shared memory tiling.

This project serves as a foundation for future optimizations such as:

- shared memory tiling
- memory coalescing
- warp-level optimization
- Tensor Core acceleration
- CUTLASS or Triton implementations

## Future Improvements

- Tiled GEMM using shared memory
- Register blocking
- Warp tiling
- WMMA / Tensor Core kernels
- Benchmark against cuBLAS
- Occupancy optimization
- Mixed precision support

## Learning Goals

This project was built as part of learning:

- CUDA programming
- GPU architecture
- high-performance computing
- deep learning systems internals
- matrix multiplication optimization

## Beginner Notes

- `cudaMalloc` allocates memory on the GPU.
- `cudaMemcpy(..., cudaMemcpyHostToDevice)` copies data from CPU memory to GPU memory.
- `dim3 block(16, 16)` creates a 16 by 16 thread block.
- `dim3 grid(...)` creates enough blocks to cover the full matrix.
- CUDA events are used here to measure kernel execution time on the GPU.

## Summary

The naive kernel is correct and easy to understand, but it is not optimized for performance. It is a good learning example for:

- GPU memory allocation
- host-to-device and device-to-host copies
- 2D thread indexing
- kernel launch configuration
- basic GPU timing with CUDA events

The memory-coalesced kernel is the next step in the journey. It keeps the computation simple while changing the memory access pattern so adjacent threads access adjacent elements of `B`.

The main limitation is that each thread still computes its result with a simple inner loop. That makes the code readable, but it is not yet tuned for high-throughput GEMM.
