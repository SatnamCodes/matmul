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

Run results:

- Run 1: 14.4626 ms, 148.485 GFLOPS
- Run 2: 12.8091 ms, 167.654 GFLOPS
- Run 3: 11.9716 ms, 179.381 GFLOPS
- Run 4: 10.8335 ms, 198.226 GFLOPS
- Run 5: 11.2482 ms, 190.918 GFLOPS
- Run 6: 10.8786 ms, 197.404 GFLOPS

Average kernel time: 12.0339 ms

Average throughput reported by the program: 180.3447 GFLOPS

Fastest run: 10.8335 ms

Slowest run: 14.4626 ms

## Profiling

### Nsight Systems

```bash
nsys profile ./naive
```

I also ran `nsys profile --trace=cuda -o nsys_naive ./naive`, which generated a `.qdstrm` trace file. This environment could not complete the importer step needed to turn that trace into a summary report, so a full Nsight Systems statistics report was not available here.

### Nsight Compute

```bash
sudo ncu ./naive
```

I profiled the kernel with `sudo ncu ./naive`. Nsight Compute completed 9 passes for the kernel and reported:

- Kernel launch configuration: `(64, 64, 1)` blocks of `(16, 16, 1)` threads
- Block size: 256 threads
- Grid size: 4,096 blocks
- Theoretical occupancy: 100%
- Achieved occupancy: 98.81%
- Memory throughput: 98.62%
- DRAM throughput: 0.42%
- L1/TEX cache throughput: 98.87%
- Compute throughput: 23.20%
- Kernel duration reported by Nsight Compute: 15.64 ms

During profiling, the runtime printed by the program increased to 484.525 ms because Nsight Compute adds significant profiling overhead. That is expected and does not represent the normal execution time.

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