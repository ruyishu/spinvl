"""绘制 SpinVL 论文核心结果图表（CVPR 学术发表级排版，修正遮挡、分组与字号）：
1. assets/fig_main_results.png：主实验对比 + 纯文本保持率
2. assets/fig_efficiency.png：TFLOPs 与推理延迟随视觉 Patch 数变化曲线 + 数据规模效应
3. assets/fig_ablations.png：内置前端同臂分组消融（rel vs ce）、编码深度 K=0..6 双轴消融、VL-OPD 自由生成增益
"""
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#1E293B"
plt.rcParams["axes.linewidth"] = 1.0

OUT_DIR = r"D:\Users\kun.z\Documents\paper\spinvl\assets"
PAPER_FIG_DIR = r"D:\Users\kun.z\Documents\paper\spinvl\paper\figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

C_LLAVA = "#64748B"
C_ROSE = "#94A3B8"
C_VORA = "#CBD5E1"
C_TOWER = "#0F766E"
C_FREE = "#B91C1C"
C_FREE06 = "#D97706"


def save_both(fig, filename):
    for d in (OUT_DIR, PAPER_FIG_DIR):
        fig.savefig(os.path.join(d, filename), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_main_results():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.4), gridspec_kw={"width_ratios": [2.25, 1.0]}, dpi=240)

    metrics = ["MMStar\n(Perception)", "HallusionBench\n(Anti-Hallu.)", "MathVista\n(Visual Math)",
               "MC2 Avg\n(Perc.+Hallu.)", "Core3 Avg\n(3-Bench Mean)"]
    data = {
        "LLaVA-Proj (Full Tokens)":     [29.60, 53.10, 24.90, 41.35, 35.87],
        "ROSE (Single-Freq)":           [25.60, 54.36, 19.70, 39.98, 33.22],
        "VoRA (LoRA K=4)":              [23.73, 49.95, 18.50, 36.84, 30.73],
        "SpinVL-Tower (Ours, 0-Tok)":   [30.60, 58.15, 23.50, 44.37, 35.90],
        "SpinVL-Free+OPD (Ours, 1.7B)": [29.20, 56.99, 19.80, 42.68, 34.95],
        "SpinVL-Free+OPD (Ours, 0.6B)": [27.53, 57.83, 20.40, 42.68, 34.66],
    }
    colors = [C_LLAVA, C_ROSE, C_VORA, C_TOWER, C_FREE, C_FREE06]

    x = np.arange(len(metrics))
    w = 0.132
    for idx, (label, vals) in enumerate(data.items()):
        offset = (idx - 2.5) * w
        bars = ax1.bar(x + offset, vals, width=w * 0.92, label=label, color=colors[idx],
                       edgecolor="#1E293B", linewidth=0.65)
        if "Ours" in label:
            for b in bars:
                ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.65, f"{b.get_height():.1f}",
                         ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=colors[idx])

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=9.8, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(12, 66)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.set_title("(a) Public Multimodal Benchmark Comparison", fontsize=11.5, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", fontsize=8.4, frameon=True, ncol=2)

    variants = ["SpinVL\n(Decoupled)", "Merged\nrot_K4", "Merged\nrots_K4", "Merged\nLoRA (VoRA)"]
    gsm = [70.4, 0.0, 0.8, 0.0]
    arc = [71.4, 24.2, 23.4, 27.4]
    x2 = np.arange(len(variants))
    w2 = 0.35
    b1 = ax2.bar(x2 - w2 / 2, gsm, width=w2, label="GSM8K (Math)", color=C_FREE, edgecolor="#1E293B", linewidth=0.7)
    b2 = ax2.bar(x2 + w2 / 2, arc, width=w2, label="ARC-Challenge", color=C_TOWER, edgecolor="#1E293B", linewidth=0.7)
    for b in list(b1) + list(b2):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(variants, fontsize=9.0, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, 86)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.set_title("(b) Pure-Text Preservation (100% Lossless)", fontsize=11.5, fontweight="bold", pad=10)
    ax2.legend(loc="upper right", fontsize=8.8, frameon=True)

    plt.tight_layout()
    save_both(fig, "fig_main_results.png")


def plot_efficiency():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.8, 4.1), dpi=240)

    patches = [256, 576, 1024, 2304, 4096]
    tflops_ours = [0.30, 0.45, 0.67, 1.37, 2.54]
    tflops_rose = [0.47, 0.86, 1.45, 3.40, 6.82]
    tflops_llava = [0.66, 1.27, 2.18, 5.07, 9.85]

    ax1.plot(patches, tflops_llava, "o--", color=C_LLAVA, linewidth=2.1, label="LLaVA (Ext. Tower + Full Tokens)")
    ax1.plot(patches, tflops_rose, "s-.", color=C_TOWER, linewidth=2.1, label="ROSE / SpinVL-Tower (Ext. Tower)")
    ax1.plot(patches, tflops_ours, "D-", color=C_FREE, linewidth=2.6, label="SpinVL-Free (Ours, -74.2% FLOPs)")
    ax1.fill_between(patches, tflops_ours, tflops_llava, color=C_FREE, alpha=0.08)
    ax1.annotate("-74.2% FLOPs\n(2.54 vs 9.85 TFLOPs)", xy=(4096, 2.54), xytext=(1950, 5.7),
                 arrowprops=dict(arrowstyle="->", color=C_FREE, lw=1.6),
                 fontsize=8.8, fontweight="bold", color=C_FREE)
    ax1.set_xlabel("Number of Visual Patches", fontsize=9.8, fontweight="bold")
    ax1.set_ylabel("Forward TFLOPs", fontsize=9.8, fontweight="bold")
    ax1.set_title("(a) Compute Scaling vs. Image Resolution", fontsize=10.8, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8.0, loc="upper left")

    ms_ours = [65.8, 66.6, 69.8, 80.4, 99.4]
    ms_rose = [58.2, 57.8, 63.8, 86.0, 126.1]
    ms_llava = [25.7, 25.6, 33.7, 66.9, 123.4]
    ax2.plot(patches, ms_llava, "o--", color=C_LLAVA, linewidth=2.1, label="LLaVA (Full Tokens)")
    ax2.plot(patches, ms_rose, "s-.", color=C_TOWER, linewidth=2.1, label="ROSE (Ext. Tower)")
    ax2.plot(patches, ms_ours, "D-", color=C_FREE, linewidth=2.6, label="SpinVL-Free (Ours)")
    ax2.annotate("Faster at High-Res\n99.4ms vs 123.4ms", xy=(4096, 99.4), xytext=(1850, 111),
                 arrowprops=dict(arrowstyle="->", color=C_FREE, lw=1.6),
                 fontsize=8.8, fontweight="bold", color=C_FREE)
    ax2.set_xlabel("Number of Visual Patches", fontsize=9.8, fontweight="bold")
    ax2.set_ylabel("Exclusive-GPU Latency (ms)", fontsize=9.8, fontweight="bold")
    ax2.set_title("(b) End-to-End Latency Crossover", fontsize=10.8, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=8.2, loc="upper left")

    Ns = [1, 3, 6, 12, 21]
    ce_acc = [19.7, 21.1, 20.5, 21.6, 22.4]
    rel_acc = [18.6, 18.7, 23.3, 19.4, 19.3]
    ax3.plot(Ns, ce_acc, "o-", color=C_TOWER, linewidth=2.3, label="CE Arm (Scales Best at Large N)")
    ax3.plot(Ns, rel_acc, "s--", color=C_FREE06, linewidth=2.1, label="KD+Rel Arm (3.5x Eff. @ N=6k)")
    ax3.axhline(22.4, color=C_LLAVA, linestyle=":", linewidth=1.3, label="CE @ N=21k Target (22.4%)")
    ax3.set_xlabel("Training Samples N (in thousands)", fontsize=9.8, fontweight="bold")
    ax3.set_ylabel("Val Full-Answer Accuracy (%)", fontsize=9.8, fontweight="bold")
    ax3.set_title("(c) Data Scaling & Sample Efficiency", fontsize=10.8, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.35)
    ax3.legend(fontsize=8.0, loc="lower right")

    plt.tight_layout()
    save_both(fig, "fig_efficiency.png")


def plot_ablations():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.8, 4.1), dpi=240)

    # Panel A: Controlled Frontend Mode Ablation grouped by arm (rel vs ce)
    modes = ["none\n(K=0)", "full\n(Copy)", "rot\n(SO(64))", "LoRA\n(VoRA)", "rots\n(Ours)", "rots+hyb\n(+OPD)"]
    mc2 = [35.92, 38.82, 40.73, 36.84, 38.14, 42.68]
    core3 = [31.15, 32.08, 33.89, 30.73, 31.32, 34.95]
    x1 = np.arange(len(modes))
    w = 0.36
    b1 = ax1.bar(x1 - w / 2, mc2, width=w, label="MC2 (MMStar+Hallu)", color=C_FREE, edgecolor="#1E293B", linewidth=0.65)
    b2 = ax1.bar(x1 + w / 2, core3, width=w, label="Core3 Avg", color=C_TOWER, edgecolor="#1E293B", linewidth=0.65)
    for b in list(b1) + list(b2):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.45, f"{b.get_height():.1f}",
                 ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    ax1.axvline(2.5, color="#475569", linestyle="--", linewidth=1.2)
    ax1.text(1.0, 43.2, "rel Init Arm", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#334155")
    ax1.text(4.0, 47.0, "ce Init Arm", ha="center", va="center", fontsize=8.4, fontweight="bold", color="#334155")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(modes, fontsize=8.2, fontweight="bold")
    ax1.set_ylim(25, 49.5)
    ax1.set_ylabel("Accuracy (%)", fontsize=9.8, fontweight="bold")
    ax1.set_title("(a) Built-in Frontend Parameterisation", fontsize=10.8, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(fontsize=7.8, loc="upper left", frameon=True)

    # Panel B: Frontend Depth K in {0, 2, 4, 6} with clear dual-axis legend
    Ks = ["K=0", "K=2", "K=4\n(Ours)", "K=6"]
    tf_first = [30.7, 36.7, 37.3, 38.0]
    gap = [0.39, 0.81, 1.07, 0.98]
    x2 = np.arange(len(Ks))
    ax2_r = ax2.twinx()
    bars_k = ax2.bar(x2 - 0.14, tf_first, width=0.36, color=C_TOWER, alpha=0.88, label="Val 1st-Token Acc (%)", edgecolor="#1E293B")
    line_k = ax2_r.plot(x2 + 0.14, gap, "D-", color=C_FREE, linewidth=2.4, markersize=7.0, label="Grounding Gap (CE)")
    for b in bars_k:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.55, f"{b.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=C_TOWER)
    for xi, gi in zip(x2 + 0.14, gap):
        ax2_r.text(xi + 0.06, gi - 0.08, f"+{gi:.2f}", ha="left", va="top", fontsize=7.8, fontweight="bold", color=C_FREE)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(Ks, fontsize=9.2, fontweight="bold")
    ax2.set_ylim(20, 45.0)
    ax2_r.set_ylim(0.1, 1.32)
    ax2.set_ylabel("1st-Token Accuracy (%)", fontsize=9.8, fontweight="bold", color=C_TOWER)
    ax2_r.set_ylabel("Visual Grounding Gap (CE)", fontsize=9.2, fontweight="bold", color=C_FREE)
    ax2.set_title("(b) Built-in Encoder Depth K Ablation", fontsize=10.8, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.8, frameon=True)

    # Panel C: Step 2 VL-OPD Free-Generation Gain (with generous y-headroom so legend never collides)
    tasks = ["MMStar\n(0.6B)", "Hallusion\n(0.6B)", "MathVista\n(0.6B)",
             "MathVista\n(1.7B)", "Mix EM\n(1.7B)"]
    before = [14.53, 23.55, 9.70, 12.20, 16.67]
    after = [22.20, 33.44, 14.40, 16.30, 20.67]
    x3 = np.arange(len(tasks))
    b_pre = ax3.bar(x3 - w / 2, before, width=w, label="Before VL-OPD (Offline CE)", color=C_VORA, edgecolor="#1E293B")
    b_post = ax3.bar(x3 + w / 2, after, width=w, label="After VL-OPD (On-Policy KL)", color=C_FREE, edgecolor="#1E293B")
    for b0, b1_bar in zip(b_pre, b_post):
        gain = b1_bar.get_height() - b0.get_height()
        ax3.text(b1_bar.get_x() + b1_bar.get_width() / 2, b1_bar.get_height() + 0.65, f"+{gain:.1f}%",
                 ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=C_FREE)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(tasks, fontsize=8.0, fontweight="bold")
    ax3.set_ylim(0, 43.5)
    ax3.set_ylabel("Free-Generation Accuracy (%)", fontsize=9.8, fontweight="bold")
    ax3.set_title("(c) Step 2 VL-OPD Cures Exposure Bias", fontsize=10.8, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.35)
    ax3.legend(fontsize=8.0, loc="upper right", frameon=True)

    plt.tight_layout()
    save_both(fig, "fig_ablations.png")


if __name__ == "__main__":
    plot_main_results()
    plot_efficiency()
    plot_ablations()
    print("Polished 3 figures saved to assets/ and paper/figures/")
