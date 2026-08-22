"""
Reads benchmark/results_cuda.csv (naive, cublas) and
benchmark/results_python.csv (torch, cutlass), and plots GFLOPS vs matrix
size for all four implementations on one log-log chart.

Run:
    .venv/bin/python benchmark/plot_results.py
"""

import csv
import pathlib

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

HERE = pathlib.Path(__file__).parent

# Chart chrome (light mode), from the project's dataviz palette.
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots 1-4, assigned in fixed order (not cycled).
SERIES = {
    "cublas": {"label": "cuBLAS (raw C++)", "color": "#2a78d6"},
    "cutlass": {"label": "CUTLASS (Python interface)", "color": "#eb6834"},
    "torch": {"label": "PyTorch (torch.matmul)", "color": "#1baf7a"},
    "naive": {"label": "Naive CUDA kernel", "color": "#eda100"},
}


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("impl unavailable") or "," not in line:
                continue
            parts = line.split(",")
            if len(parts) != 7:
                continue
            impl, m, n, k, ms, gf, correct = parts
            rows.append(
                {
                    "impl": impl,
                    "n": int(n),
                    "ms": float(ms),
                    "gflops": float(gf),
                    "correct": correct == "1",
                }
            )
    return rows


def main():
    rows = load_rows(HERE / "results_cuda.csv") + load_rows(HERE / "results_python.csv")

    data = {impl: {"n": [], "gflops": []} for impl in SERIES}
    for r in sorted(rows, key=lambda r: r["n"]):
        if r["impl"] in data:
            data[r["impl"]]["n"].append(r["n"])
            data[r["impl"]]["gflops"].append(r["gflops"])
        if not r["correct"]:
            print(f"WARNING: {r['impl']} at N={r['n']} failed the correctness check")

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for impl, spec in SERIES.items():
        d = data[impl]
        if not d["n"]:
            continue
        ax.plot(
            d["n"],
            d["gflops"],
            marker="o",
            markersize=6,
            linewidth=2,
            color=spec["color"],
            label=spec["label"],
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    all_sizes = sorted({n for d in data.values() for n in d["n"]})
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([str(n) for n in all_sizes], color=MUTED_INK)
    ax.set_xlabel("matrix size (N, square M=N=K, fp32)", color=SECONDARY_INK, fontsize=11)
    ax.set_ylabel("GFLOPS (higher is better)", color=SECONDARY_INK, fontsize=11)

    ax.set_title(
        "SGEMM throughput: naive kernel vs cuBLAS vs PyTorch vs CUTLASS",
        color=PRIMARY_INK,
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=28,
    )
    fig.text(
        0.125,
        0.925,
        "NVIDIA GeForce RTX 4060 Laptop GPU · CUDA 12.0 · fp32",
        color=MUTED_INK,
        fontsize=10,
    )

    ax.grid(True, which="major", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRIDLINE, linewidth=0.4, alpha=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.tick_params(colors=MUTED_INK)

    legend = ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=10,
        labelcolor=SECONDARY_INK,
    )

    fig.text(
        0.125,
        0.01,
        "CUTLASS uses the stock Python interface with no per-size kernel tuning;\n"
        "its Python dispatch overhead dominates at small sizes.",
        color=MUTED_INK,
        fontsize=8.5,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    out_path = HERE / "results.png"
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
