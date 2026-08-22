"""
Sweeps torch.matmul (cuBLAS-backed) and the CUTLASS Python interface across
the same sizes used by bench_cuda.cu, and prints CSV rows to stdout:
    impl,M,N,K,ms,gflops,correct

Run:
    .venv/bin/python benchmark/bench_python.py > benchmark/results_python.csv
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import torch

SIZES = [256, 512, 1024, 2048, 4096]
WARMUP = 3
ITERS = 10


def time_op(op, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        op()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def gflops(m, n, k, ms):
    return (2.0 * m * n * k) / (ms * 1e6)


def bench_torch(m, n, k):
    A = torch.ones((m, k), dtype=torch.float32, device="cuda")
    B = torch.ones((k, n), dtype=torch.float32, device="cuda")

    ms = time_op(lambda: torch.matmul(A, B))
    C = torch.matmul(A, B)
    correct = torch.allclose(C, torch.full_like(C, float(k)), rtol=1e-3)
    return ms, gflops(m, n, k, ms), correct


def bench_cutlass(plan, m, n, k):
    A = torch.ones((m, k), dtype=torch.float32, device="cuda")
    B = torch.ones((k, n), dtype=torch.float32, device="cuda")
    C = torch.zeros((m, n), dtype=torch.float32, device="cuda")

    ms = time_op(lambda: plan.run(A, B, C, C, print_module=False))
    correct = torch.allclose(C, torch.full_like(C, float(k)), rtol=1e-3)
    return ms, gflops(m, n, k, ms), correct


def main():
    if not torch.cuda.is_available():
        print("no CUDA device available", file=sys.stderr)
        sys.exit(1)

    try:
        import cutlass_cppgen as cutlass

        cutlass_plan = cutlass.Gemm(
            element=torch.float32, layout=cutlass.LayoutType.RowMajor
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"cutlass unavailable: {exc}", file=sys.stderr)
        cutlass_plan = None

    for n in SIZES:
        ms, gf, correct = bench_torch(n, n, n)
        print(f"torch,{n},{n},{n},{ms},{gf},{int(correct)}")
        sys.stdout.flush()

        if cutlass_plan is not None:
            try:
                ms, gf, correct = bench_cutlass(cutlass_plan, n, n, n)
                print(f"cutlass,{n},{n},{n},{ms},{gf},{int(correct)}")
            except Exception as exc:  # pragma: no cover
                print(f"cutlass failed at size {n}: {exc}", file=sys.stderr)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
