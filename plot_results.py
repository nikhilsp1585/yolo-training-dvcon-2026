"""
plot_results.py
───────────────
Generates all figures needed for the Stage 2 report from simulation_results.json.

Figures produced:
  Fig 1 — Per-layer activation sparsity (top 20 layers, bar chart)
  Fig 2 — Overall sparsity + MAC reduction summary (horizontal bar)
  Fig 3 — DistilBERT encoding latency per task (bar chart)
  Fig 4 — Task-filtered detection relevance: Precision@1 per task
  Fig 5 — Suppression rate per task (objects filtered out %)
  Fig 6 — Latency comparison: sequential vs pipelined (grouped bar)
  Fig 7 — Speedup summary (single highlighted bar)

All figures saved as high-res PNGs in the results/ directory.

Usage:
    python plot_results.py --results results/simulation_results.json
"""

import argparse
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

# Colour palette consistent with paper diagrams
C_TEAL   = "#1D9E75"
C_PURPLE = "#7F77DD"
C_CORAL  = "#D85A30"
C_AMBER  = "#BA7517"
C_BLUE   = "#378ADD"
C_GRAY   = "#888780"


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Per-layer sparsity (top 20 Conv/ReLU layers)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_layer_sparsity(results: dict, out_dir: Path):
    layers = results["sparsity"]["per_layer"]
    # Sort by sparsity, take top 20
    top20 = sorted(layers, key=lambda x: x["sparsity_pct"], reverse=True)[:20]

    names = [l["layer_name"].split(".")[-2] + "." + l["layer_name"].split(".")[-1]
             for l in top20]
    values = [l["sparsity_pct"] for l in top20]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(names)), values, color=C_TEAL, alpha=0.85, height=0.7)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, color=C_GRAY)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Activation sparsity (%)")
    ax.set_title("Fig 1 — Per-layer activation sparsity (top 20 layers, YOLOv5-nano)")
    ax.set_xlim(0, 105)
    ax.axvline(x=results["sparsity"]["overall_sparsity_pct"],
               color=C_CORAL, linestyle="--", linewidth=1.2, label="Overall mean")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_per_layer_sparsity.png")
    plt.close(fig)
    print("  Saved fig1_per_layer_sparsity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Sparsity & MAC reduction summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_sparsity_mac_summary(results: dict, out_dir: Path):
    sp = results["sparsity"]
    labels = ["Activation sparsity", "MAC reduction (zero-skip)"]
    values = [sp["overall_sparsity_pct"], sp["mac_reduction_pct"]]
    colors = [C_TEAL, C_PURPLE]

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(labels, values, color=colors, height=0.45)

    for bar, val in zip(bars, values):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=12, fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Fig 2 — Sparsity and MAC reduction summary (YOLOv5-nano on COCO)")

    # Reference lines for paper claims
    ax.axvline(x=30, color=C_GRAY, linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(x=50, color=C_GRAY, linestyle=":", linewidth=1, alpha=0.6)
    ax.text(30, -0.55, "30%", fontsize=8, color=C_GRAY, ha="center")
    ax.text(50, -0.55, "50%", fontsize=8, color=C_GRAY, ha="center")
    ax.text(40, -0.7, "← Proposed claim range →", fontsize=8, color=C_GRAY, ha="center")

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_sparsity_mac_summary.png")
    plt.close(fig)
    print("  Saved fig2_sparsity_mac_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — DistilBERT encoding latency per task
# ─────────────────────────────────────────────────────────────────────────────

def plot_encoding_latency(results: dict, out_dir: Path):
    lats = results["encoding_latencies"]
    labels = [l["task_label"] for l in lats]
    means  = [l["mean_ms"] for l in lats]
    stds   = [l["std_ms"] for l in lats]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=C_PURPLE, alpha=0.85,
           error_kw={"ecolor": C_GRAY, "linewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Encoding latency (ms)")
    ax.set_title("Fig 3 — DistilBERT task encoding latency (mean ± std, 10 runs)")

    mean_all = np.mean(means)
    ax.axhline(y=mean_all, color=C_CORAL, linestyle="--", linewidth=1.2,
               label=f"Mean = {mean_all:.1f} ms")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_encoding_latency.png")
    plt.close(fig)
    print("  Saved fig3_encoding_latency.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Task relevance: Precision@1 per task (with vs without filtering)
# ─────────────────────────────────────────────────────────────────────────────

def plot_precision_per_task(results: dict, out_dir: Path):
    metrics = results["task_metrics"]
    labels  = [v["label"] for v in metrics.values()]
    p1      = [v["mean_precision_at_1"] for v in metrics.values()]
    p3      = [v["mean_precision_at_3"] for v in metrics.values()]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - width/2, p1, width, label="Precision@1", color=C_TEAL,   alpha=0.85)
    ax.bar(x + width/2, p3, width, label="Precision@3", color=C_PURPLE,  alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.15)
    ax.set_title("Fig 4 — Task-filtered detection precision per task (task-aware pipeline)")
    ax.legend(fontsize=9)
    ax.axhline(y=np.mean(p1), color=C_TEAL, linestyle="--", alpha=0.5, linewidth=1)

    for xi, val in zip(x - width/2, p1):
        ax.text(xi, val + 0.02, f"{val:.2f}", ha="center", fontsize=7.5, color=C_TEAL)

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_precision_per_task.png")
    plt.close(fig)
    print("  Saved fig4_precision_per_task.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Suppression rate per task
# ─────────────────────────────────────────────────────────────────────────────

def plot_suppression_rate(results: dict, out_dir: Path):
    metrics = results["task_metrics"]
    labels  = [v["label"] for v in metrics.values()]
    supps   = [v["mean_suppression_pct"] for v in metrics.values()]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors  = [C_CORAL if s > 60 else C_AMBER if s > 40 else C_TEAL for s in supps]
    ax.bar(range(len(labels)), supps, color=colors, alpha=0.85)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Suppression rate (%)")
    ax.set_title("Fig 5 — Task-irrelevant object suppression rate per task")
    ax.set_ylim(0, 110)
    ax.axhline(y=np.mean(supps), color=C_GRAY, linestyle="--", linewidth=1.2,
               label=f"Mean = {np.mean(supps):.1f}%")

    # Legend patches
    high_p = mpatches.Patch(color=C_CORAL,  label=">60% suppression")
    mid_p  = mpatches.Patch(color=C_AMBER,  label="40–60%")
    low_p  = mpatches.Patch(color=C_TEAL,   label="<40%")
    ax.legend(handles=[high_p, mid_p, low_p], fontsize=9)

    for i, val in enumerate(supps):
        ax.text(i, val + 1.5, f"{val:.0f}%", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_suppression_rate.png")
    plt.close(fig)
    print("  Saved fig5_suppression_rate.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Latency: sequential vs pipelined
# ─────────────────────────────────────────────────────────────────────────────

def plot_latency_comparison(results: dict, out_dir: Path):
    lat = results["latency"]
    seq_ms  = lat["sequential"]["mean_ms_per_image"]
    pipe_ms = lat["pipelined"]["ms_per_image"]
    speedup = lat["speedup_x"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left — per-image latency bar
    ax = axes[0]
    bars = ax.bar(["Sequential\n(CPU-style)", "Pipelined\n(proposed)"],
                  [seq_ms, pipe_ms],
                  color=[C_GRAY, C_TEAL], alpha=0.85, width=0.5)
    for bar, val in zip(bars, [seq_ms, pipe_ms]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2,
                f"{val:.1f} ms", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latency per image (ms)")
    ax.set_title("Fig 6a — Inference latency comparison")
    ax.set_ylim(0, max(seq_ms, pipe_ms) * 1.3)

    # Right — speedup gauge
    ax2 = axes[1]
    ax2.barh(["Speedup"], [speedup], color=C_PURPLE, alpha=0.85, height=0.4)
    ax2.axvline(x=1.0, color=C_GRAY, linestyle="--", linewidth=1)
    ax2.text(speedup + 0.05, 0, f"{speedup}×", va="center",
             fontsize=14, fontweight="bold", color=C_PURPLE)
    ax2.set_xlim(0, max(speedup * 1.3, 2))
    ax2.set_xlabel("Speedup factor (×)")
    ax2.set_title("Fig 6b — Effective speedup")
    ax2.text(1.0, -0.35, "1× (baseline)", ha="center", fontsize=8, color=C_GRAY)

    fig.tight_layout()
    fig.savefig(out_dir / "fig6_latency_comparison.png")
    plt.close(fig)
    print("  Saved fig6_latency_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Combined summary dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_dashboard(results: dict, out_dir: Path):
    sp  = results["sparsity"]
    lat = results["latency"]
    metrics = results["task_metrics"]

    mean_p1  = np.mean([v["mean_precision_at_1"] for v in metrics.values()])
    mean_sup = np.mean([v["mean_suppression_pct"] for v in metrics.values()])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Stage 2 Results Dashboard — Task-Aware Sparse YOLO",
                 fontsize=14, fontweight="bold", y=1.01)

    # Top-left: sparsity
    ax = axes[0, 0]
    ax.barh(["Act. sparsity", "MAC reduction"],
            [sp["overall_sparsity_pct"], sp["mac_reduction_pct"]],
            color=[C_TEAL, C_PURPLE], height=0.45)
    ax.set_xlim(0, 100)
    for i, v in enumerate([sp["overall_sparsity_pct"], sp["mac_reduction_pct"]]):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontweight="bold")
    ax.set_title("Activation Sparsity & MAC Savings")
    ax.axvspan(30, 50, alpha=0.08, color=C_TEAL, label="Target range")
    ax.legend(fontsize=8)

    # Top-right: latency
    ax = axes[0, 1]
    seq_ms  = lat["sequential"]["mean_ms_per_image"]
    pipe_ms = lat["pipelined"]["ms_per_image"]
    bars = ax.bar(["Sequential", "Pipelined"], [seq_ms, pipe_ms],
                  color=[C_GRAY, C_TEAL], alpha=0.85, width=0.45)
    for bar, val in zip(bars, [seq_ms, pipe_ms]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                f"{val:.1f} ms", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("ms / image")
    ax.set_title(f"Latency Comparison  ({lat['speedup_x']}× speedup)")
    ax.set_ylim(0, max(seq_ms, pipe_ms) * 1.35)

    # Bottom-left: precision@1 per task (mini)
    ax = axes[1, 0]
    labels = [v["label"][:10] for v in metrics.values()]
    p1vals = [v["mean_precision_at_1"] for v in metrics.values()]
    ax.bar(range(len(labels)), p1vals, color=C_BLUE, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Precision@1")
    ax.set_title(f"Task Precision@1  (mean={mean_p1:.2f})")
    ax.axhline(mean_p1, color=C_CORAL, linestyle="--", linewidth=1)

    # Bottom-right: suppression rate
    ax = axes[1, 1]
    supps = [v["mean_suppression_pct"] for v in metrics.values()]
    ax.bar(range(len(labels)), supps, color=C_CORAL, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Suppression rate (%)")
    ax.set_title(f"Irrelevant Object Suppression  (mean={mean_sup:.1f}%)")
    ax.axhline(mean_sup, color=C_PURPLE, linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(out_dir / "fig7_summary_dashboard.png")
    plt.close(fig)
    print("  Saved fig7_summary_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot Stage 2 simulation results")
    parser.add_argument("--results", type=str,
                        default="results/simulation_results.json",
                        help="Path to simulation_results.json")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (defaults to same dir as results JSON)")
    args = parser.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent

    print(f"\n[PlotResults] Loading {results_path}...")
    results = load_results(results_path)

    print("[PlotResults] Generating figures...\n")
    plot_per_layer_sparsity(results, out_dir)
    plot_sparsity_mac_summary(results, out_dir)
    plot_encoding_latency(results, out_dir)
    plot_precision_per_task(results, out_dir)
    plot_suppression_rate(results, out_dir)
    plot_latency_comparison(results, out_dir)
    plot_summary_dashboard(results, out_dir)

    print(f"\n[PlotResults] All 7 figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
