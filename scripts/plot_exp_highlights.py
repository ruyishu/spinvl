"""生成正文第 4 章 Fig. 3（矢量 PDF + 300 DPI PNG 预览）以及附录纯数据缩放图（避免与正文 Fig. 3 重复）：
1. paper/figures/fig_exp_highlights.pdf (.png):
   (a) Compute Scaling vs. Visual Patches (TFLOPs)
   (b) End-to-End Latency Crossover (ms)
   (c) Step 2 VL-OPD Free-Generation Gains (0.6B & 1.7B)
2. paper/figures/fig_data_scaling.pdf (.png):
   CE Arm vs. KD+Rel Arm from 1k to 21k samples
"""
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#1E293B"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

PAPER_FIG_DIR = r"D:\Users\kun.z\Documents\paper\spinvl\paper\figures"
ASSETS_DIR = r"D:\Users\kun.z\Documents\paper\spinvl\assets"
os.makedirs(PAPER_FIG_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

C_LLAVA = "#64748B"
C_ROSE = "#94A3B8"
C_LIGHT = "#CBD5E1"
C_TOWER = "#0F766E"
C_FREE = "#B91C1C"
C_AMBER = "#D97706"


def plot_fig3():
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(14.4, 3.15), gridspec_kw={"width_ratios": [1.0, 1.0, 1.18]}, dpi=300
    )

    patches = [256, 576, 1024, 2304, 4096]
    tflops_ours = [0.30, 0.45, 0.67, 1.37, 2.54]
    tflops_rose = [0.47, 0.86, 1.45, 3.40, 6.82]
    tflops_llava = [0.66, 1.27, 2.18, 5.07, 9.85]

    ax1.plot(patches, tflops_llava, "o--", color=C_LLAVA, linewidth=2.0, markersize=5.2, label="LLaVA-Proj (Full Tok)")
    ax1.plot(patches, tflops_rose, "s-.", color=C_TOWER, linewidth=2.0, markersize=5.0, label="ROSE / SpinVL-Tower")
    ax1.plot(patches, tflops_ours, "D-", color=C_FREE, linewidth=2.5, markersize=5.4, label="SpinVL-Free (Ours)")
    ax1.fill_between(patches, tflops_ours, tflops_llava, color=C_FREE, alpha=0.08)
    ax1.annotate(
        "-74.2% FLOPs\n(2.54 vs. 9.85)",
        xy=(4096, 2.54),
        xytext=(920, 6.2),
        arrowprops=dict(arrowstyle="->", color=C_FREE, lw=1.5),
        fontsize=8.1,
        fontweight="bold",
        color=C_FREE,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_FREE, alpha=0.95, lw=0.8),
    )
    ax1.set_xlabel("Visual Patches", fontsize=9.6, fontweight="bold")
    ax1.set_ylabel("Forward TFLOPs", fontsize=9.6, fontweight="bold")
    ax1.set_ylim(-0.2, 10.7)
    ax1.set_title("(a) Prefill Compute Scaling", fontsize=10.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=7.7, loc="upper left", frameon=True)

    ms_ours = [65.8, 66.6, 69.8, 80.4, 99.4]
    ms_rose = [58.2, 57.8, 63.8, 86.0, 126.1]
    ms_llava = [25.7, 25.6, 33.7, 66.9, 123.4]
    ax2.plot(patches, ms_llava, "o--", color=C_LLAVA, linewidth=2.0, markersize=5.2, label="LLaVA-Proj (Full Tok)")
    ax2.plot(patches, ms_rose, "s-.", color=C_TOWER, linewidth=2.0, markersize=5.0, label="ROSE (Ext. Tower)")
    ax2.plot(patches, ms_ours, "D-", color=C_FREE, linewidth=2.5, markersize=5.4, label="SpinVL-Free (Ours)")
    ax2.annotate(
        "-19.5% Latency\n(99.4 vs. 123.4 ms)",
        xy=(4096, 99.4),
        xytext=(2100, 32),
        arrowprops=dict(arrowstyle="->", color=C_FREE, lw=1.5),
        fontsize=8.1,
        fontweight="bold",
        color=C_FREE,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_FREE, alpha=0.95, lw=0.8),
    )
    ax2.set_xlabel("Visual Patches", fontsize=9.6, fontweight="bold")
    ax2.set_ylabel("Exclusive-GPU Latency (ms)", fontsize=9.6, fontweight="bold")
    ax2.set_ylim(18, 136)
    ax2.set_title("(b) End-to-End Latency Crossover", fontsize=10.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=7.7, loc="upper left", frameon=True)

    tasks = ["MMStar\n(0.6B)", "Hallu.\n(0.6B)", "MathV.\n(0.6B)", "MC2\n(0.6B)", "MMStar\n(1.7B)", "MC2\n(1.7B)"]
    before = [14.53, 23.55, 9.70, 19.04, 14.07, 19.23]
    after = [22.20, 33.44, 14.40, 27.82, 18.13, 20.53]
    x3 = np.arange(len(tasks))
    w = 0.36
    b_pre = ax3.bar(x3 - w / 2, before, width=w, label="Before VL-OPD (Offline CE)", color=C_LIGHT, edgecolor="#1E293B", linewidth=0.7)
    b_post = ax3.bar(x3 + w / 2, after, width=w, label="After VL-OPD (On-Policy KL)", color=C_FREE, edgecolor="#1E293B", linewidth=0.7)
    for b0, b1 in zip(b_pre, b_post):
        gain = b1.get_height() - b0.get_height()
        ax3.text(
            b1.get_x() + b1.get_width() / 2 + 0.02,
            b1.get_height() + 0.65,
            f"+{gain:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color=C_FREE,
        )
    ax3.set_xticks(x3)
    ax3.set_xticklabels(tasks, fontsize=8.0, fontweight="bold")
    ax3.set_ylim(0, 42.5)
    ax3.set_ylabel("Free-Gen Accuracy (%)", fontsize=9.6, fontweight="bold")
    ax3.set_title("(c) VL-OPD Free-Generation Gains", fontsize=10.5, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.35)
    ax3.legend(fontsize=7.7, loc="upper right", frameon=True)

    plt.tight_layout()
    for base in (os.path.join(PAPER_FIG_DIR, "fig_exp_highlights"), os.path.join(ASSETS_DIR, "fig_exp_highlights")):
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_data_scaling():
    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=300)
    Ns = [1, 3, 6, 12, 21]
    ce_acc = [19.7, 21.1, 20.5, 21.6, 22.4]
    rel_acc = [18.6, 18.7, 23.3, 19.4, 19.3]
    ax.plot(Ns, ce_acc, "o-", color=C_TOWER, linewidth=2.3, markersize=6.0, label="CE Arm (Scales Best at Large N)")
    ax.plot(Ns, rel_acc, "s--", color=C_AMBER, linewidth=2.1, markersize=5.8, label="KD+Rel Arm (Peaks @ N=6k)")
    ax.axhline(22.4, color=C_LLAVA, linestyle=":", linewidth=1.3, label="CE @ N=21k Target (22.4%)")
    ax.set_xlabel("Training Samples N (in thousands)", fontsize=9.8, fontweight="bold")
    ax.set_ylabel("Val Full-Answer Accuracy (%)", fontsize=9.8, fontweight="bold")
    ax.set_ylim(18.0, 24.3)
    ax.set_title("Data Scaling & Initialisation Arm Comparison", fontsize=10.6, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.2, loc="upper right", frameon=True)
    plt.tight_layout()
    for base in (os.path.join(PAPER_FIG_DIR, "fig_data_scaling"), os.path.join(ASSETS_DIR, "fig_data_scaling")):
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_fig3()
    plot_data_scaling()
    print("Saved fig_exp_highlights.{pdf,png} and fig_data_scaling.{pdf,png}")
