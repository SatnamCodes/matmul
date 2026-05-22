# CUDA GEMM Journey

A small CUDA matrix multiplication project built step by step: first a readable naive GEMM kernel, then a memory-coalesced version that changes how threads map to output elements.

The goal is not to beat cuBLAS yet. The goal is to understand how GPU performance changes when the math stays the same but the memory access pattern changes.

## GEMM Equation

Both kernels compute:

$$
C = \alpha AB + \beta C
$$

Current matrix shape:

- `A`: `1024 x 1024`
- `B`: `1024 x 1024`
- `C`: `1024 x 1024`
- `alpha = 1.0`
- `beta = 0.0`

The input matrices are initialized with `1.0f`, so every output element should equal `K = 1024`.

## Benchmark Setup

Benchmarks were collected on:

```text
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
Driver: 580.159.03
CUDA runtime reported by nvidia-smi: 13.0
```

Each kernel was run 6 times. All runs produced the correct result.

## Kernel 1: Naive GEMM

File: [naive.cu](naive.cu)

### Idea

The naive kernel uses a `16 x 16` 2D thread block. Each CUDA thread computes one element of `C`.

```cpp
const uint x = blockIdx.x * blockDim.x + threadIdx.x;
const uint y = blockIdx.y * blockDim.y + threadIdx.y;
```

Each thread:

1. Finds its output row and column.
2. Loops over `K`.
3. Accumulates one dot product.
4. Writes one value to `C`.

This is easy to understand, but it does not try to optimize global memory access or data reuse.

### Build And Run

```bash
nvcc -O3 naive.cu -o naive
./naive
```

Example output:

```text
Result is correct.
Kernel Time : 12.8 ms
GFLOPS: 167.6
```

### Benchmark Results

| Run | Kernel Time (ms) | GFLOPS |
| --- | ---: | ---: |
| 1 | 19.9474 | 107.658 |
| 2 | 13.0193 | 164.946 |
| 3 | 13.2733 | 161.790 |
| 4 | 13.1747 | 163.001 |
| 5 | 10.2245 | 210.033 |
| 6 | 10.2201 | 210.123 |

Summary:

| Metric | Value |
| --- | ---: |
| Average kernel time | 13.3099 ms |
| Average throughput | 169.592 GFLOPS |
| Fastest run | 10.2201 ms |
| Slowest run | 19.9474 ms |

### Profiling

Nsight Systems command:

```bash
nsys profile --trace=cuda -o nsys_naive ./naive
```

Generated trace:

```text
nsys_naive.qdstrm
```

Nsight Compute command:

```bash
sudo ncu --target-processes all --set full ./naive
```

Key Nsight Compute metrics captured earlier:

| Metric | Value |
| --- | ---: |
| Kernel duration | 15.64 ms |
| Memory throughput | 98.62% |
| DRAM throughput | 0.42% |
| L1/TEX cache throughput | 98.87% |
| L2 cache throughput | 4.88% |
| Compute throughput | 23.20% |
| Registers per thread | 36 |
| Block size | 256 |
| Grid size | 4096 |
| Theoretical occupancy | 100% |
| Achieved occupancy | 98.81% |
| Achieved active warps per SM | 47.43 |

### Bottleneck Notes

- No shared memory tiling.
- Repeated global memory reads.
- Very limited reuse of values loaded from `A` and `B`.
- High L1/TEX pressure in the profiled run.
- Good learning baseline, but not a high-throughput GEMM design.

## Kernel 2: Memory-Coalesced GEMM

File: [mem_coal_#2.cu](mem_coal_#2.cu)

### Idea

This kernel keeps one thread per output element, but changes the thread mapping inside each block.

```cpp
#define BLOCKSIZE 32

const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / 32);
const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % 32);
```

The block has `32 * 32 = 1024` threads. It is launched as a 1D block, then manually mapped into a `32 x 32` output tile.

For threads `0..31` in a warp:

- `x` stays the same
- `y` changes consecutively
- `B[i * N + y]` becomes a coalesced read
- `C[x * N + y]` becomes a coalesced write

The math is still simple, but the memory access pattern is better.

## Short Manim Explainer

File: [side_by_side_kernels_manim.py](side_by_side_kernels_manim.py)

This is a compact, under-one-minute beginner animation that puts both kernels
side by side:

- naive mapping: warp lanes touch strided rows of `C`
- coalesced mapping: warp lanes touch neighboring columns of `C`
- takeaway: same GEMM math, better memory access pattern

Render it with:

```bash
manim -pql side_by_side_kernels_manim.py SideBySideKernels
```

### Build And Run

```bash
nvcc -O3 'mem_coal_#2.cu' -o mem_coal
./mem_coal
```

Example output:

```text
Result is correct.
Kernel Time : 3.6273 ms
GFLOPS: 592.034
```

### Benchmark Results

| Run | Kernel Time (ms) | GFLOPS |
| --- | ---: | ---: |
| 1 | 3.6273 | 592.034 |
| 2 | 3.0635 | 700.993 |
| 3 | 2.6418 | 812.889 |
| 4 | 2.6412 | 813.086 |
| 5 | 2.6358 | 814.744 |
| 6 | 2.5423 | 844.700 |

Summary:

| Metric | Value |
| --- | ---: |
| Average kernel time | 2.8586 ms |
| Average throughput | 763.074 GFLOPS |
| Fastest run | 2.5423 ms |
| Slowest run | 3.6273 ms |

### Profiling

Compiler resource command:

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

Nsight Systems command:

```bash
nsys profile --trace=cuda --force-overwrite=true -o mem_coal_nsys_exact ./mem_coal
```

Relevant Nsight Systems terminal output:

```text
Result is correct.
Kernel Time : 4.75203 ms
GFLOPS: 451.909
Generating '/tmp/nsys-report-d4cd.qdstrm'
Generated:
    /home/totallynotsatnam/CS/GPU/matmul/mem_coal_nsys_exact.qdstrm
```

Generated trace:

```text
mem_coal_nsys_exact.qdstrm
```

Nsight Compute command:

```bash
sudo ncu --target-processes all --set full ./mem_coal
```

Nsight Compute metrics to collect next:

| Metric Group | Why It Matters |
| --- | --- |
| Memory Workload Analysis | Confirms whether global loads/stores are coalesced efficiently. |
| L1/TEX cache metrics | Shows pressure from repeated memory reads. |
| L2 cache metrics | Shows reuse across blocks and warps. |
| Launch Statistics | Shows effect of the 1024-thread block size. |
| Occupancy | Shows how many warps are active per SM. |
| Scheduler Statistics | Shows whether warps are stalled on memory or instruction scheduling. |
| Source Counters | Points to the exact source lines causing stalls. |

### Bottleneck Notes

- Coalesced reads from `B` are better than the naive mapping.
- Stores to `C` are also coalesced for consecutive `y` values.
- No shared memory tiling yet, so global memory reuse is still weak.
- `B` values are reused mathematically across warps, but not staged in shared memory.
- The `1024` thread block size is large and may reduce scheduling flexibility.
- No Tensor Core path yet.

## Kernel Comparison

| Metric | Naive GEMM | Memory-Coalesced GEMM |
| --- | --- | --- |
| File | `naive.cu` | `mem_coal_#2.cu` |
| Block shape | `16 x 16` | `1024 x 1`, mapped to `32 x 32` |
| Threads per block | 256 | 1024 |
| Average kernel time | 13.3099 ms | 2.8586 ms |
| Average throughput | 169.592 GFLOPS | 763.074 GFLOPS |
| Fastest run | 10.2201 ms | 2.5423 ms |
| Slowest run | 19.9474 ms | 3.6273 ms |
| Correctness | Passed | Passed |
| Nsight Systems trace | `nsys_naive.qdstrm` | `mem_coal_nsys_exact.qdstrm` |

Speedup:

| Comparison | Value |
| --- | ---: |
| Average speedup | 4.66x |
| Best-run speedup | 4.02x |

The memory-coalesced version is faster because adjacent threads access adjacent values of `B`, which makes global memory reads more GPU-friendly. It is still not a fully optimized GEMM because it does not reuse tiles through shared memory.

## What This Teaches

- Correctness is only the first step in CUDA.
- Thread mapping changes memory behavior.
- Coalescing can give a large speedup even before shared memory tiling.
- Profiling needs to be structured per kernel, otherwise it becomes hard to compare results.
- The next serious optimization is tiled GEMM using shared memory.

## Future Improvements

- Shared memory tiled GEMM
- Register blocking
- Warp tiling
- WMMA / Tensor Core kernels
- Benchmark against cuBLAS
- Mixed precision support
- Cleaner Nsight Compute side-by-side table after collecting full counter output for both kernels
