# Naive CUDA GEMM

A beginner-friendly CUDA implementation of naive matrix multiplication using GPU parallelism.

This project was built to understand:

- CUDA kernel execution
- thread, block, and grid hierarchy
- GPU memory management
- kernel timing and benchmarking
- FLOPS measurement
- CUDA profiling basics

## GEMM Equation

The kernel computes:

$$
C = \alpha AB + \beta C
$$

where:

- $A$ is an $M \times K$ matrix
- $B$ is a $K \times N$ matrix
- $C$ is an $M \times N$ matrix

The current program uses square matrices of size `1024 x 1024` and launches one CUDA thread per output element.

## Features

- CUDA-based matrix multiplication kernel
- 2D thread and block indexing
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

## How It Works

The kernel in [naive.cu](naive.cu) computes one output element per thread. Each thread:

1. Finds its row and column in the output matrix.
2. Loops over the shared dimension `K`.
3. Accumulates the dot product for that output element.
4. Writes the final value back to device memory.

The matrices are initialized with ones, so the expected result is easy to check: every element of `C` should become `K` when `alpha = 1` and `beta = 0`.

## Build

```bash
nvcc -O3 naive.cu -o naive
```

## Run

```bash
./naive
```

## Example Output

```text
Result is correct.
Kernel Time : 12.8 ms
GFLOPS: 167.6
```

## What I Measured

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

## Profiling

### Nsight Systems

```bash
nsys profile ./naive
```

I also ran `nsys profile --trace=cuda -o nsys_naive ./naive`, which generated a `.qdstrm` trace file.

### Nsight Compute

```bash
sudo ncu ./naive
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

This implementation is intentionally naive:

- no shared memory tiling
- repeated global memory accesses
- limited memory reuse

As a result, the kernel is memory-bandwidth limited and achieves only a fraction of the GPU's theoretical peak FLOPS.

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

This kernel is correct and easy to understand, but it is not optimized for performance. It is a good learning example for:

- GPU memory allocation
- host-to-device and device-to-host copies
- 2D thread indexing
- kernel launch configuration
- basic GPU timing with CUDA events

The main limitation is that each thread computes its result with a simple inner loop, so this version is straightforward but not tuned for high throughput.