"""绘制 SpinVL 论文核心结果图表（CVPR 凛冽学术风格）：
1. assets/fig_main_results.png：主实验对比（1.7B/0.6B/8B 在 MMStar、HallusionBench、MathVista、MC2、Core3 上的表现 + 纯文本 GSM8K 保持率）
2. assets/fig_efficiency.png：TFLOPs 与推理延迟随视觉 Patch 数（256~4096）的变化曲线 + 数据规模曲线（N=1k~21k）
3. assets/fig_ablations.png：内置前端模式消融（none/lora/full/rot/rots）、层数 K=0..6 消融、以及 VL-OPD 自由生成增益
"""
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#222222"
plt.rcParams["axes.linewidth"] = 1.0

OUT_DIR = r"D:\Users\kun.z\Documents\paper\spinvl\assets"
os.makedirs(OUT_DIR, exist_ok=True)

# Color palette (sharp academic slate / crimson / teal / amber)
C_LLAVA = "#64748B"
C_ROSE = "#94A3B8"
C_VORA = "#CBD5E1"
C_TOWER = "#0F766E"
C_FREE = "#B91C1C"
C_FREE06 = "#D97706"


def plot_main_results():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [2.2, 1.0]}, dpi=220)

    metrics = ["MMStar\n(Perception)", "HallusionBench\n(Anti-Hallu.)", "MathVista\n(Visual Math)", "MC2 Avg\n(Perc.+Hallu.)", "Core3 Avg\n(3-Bench Mean)"]
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
    w = 0.13
    for idx, (label, vals) in enumerate(data.items()):
        offset = (idx - 2.5) * w
        bars = ax1.bar(x + offset, vals, width=w * 0.92, label=label, color=colors[idx], edgecolor="#1E293B", linewidth=0.6)
        if "Ours" in label:
            for b in bars:
                ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}",
                         ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=colors[idx])

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(12, 65)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.set_title("(a) Public Multimodal Benchmark Comparison (bench_eval_v4)", fontsize=11, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", fontsize=8, frameon=True, ncol=2)

    # Panel B: Pure-text preservation (GSM8K & ARC-C)
    variants = ["SpinVL\n(Decoupled)", "Merged\nrot_K4", "Merged\nrots_K4", "Merged\nLoRA (VoRA)"]
    gsm = [70.4, 0.0, 0.8, 0.0]
    arc = [71.4, 24.2, 23.4, 27.4]
    x2 = np.arange(len(variants))
    w2 = 0.34
    b1 = ax2.bar(x2 - w2 / 2, gsm, width=w2, label="GSM8K (Math)", color=C_FREE, edgecolor="#1E293B", linewidth=0.7)
    b2 = ax2.bar(x2 + w2 / 2, arc, width=w2, label="ARC-Challenge", color=C_TOWER, edgecolor="#1E293B", linewidth=0.7)
    for b in list(b1) + list(b2):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(variants, fontsize=8.8, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 84)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.set_title("(b) Pure-Text Preservation (100% Lossless)", fontsize=11, fontweight="bold", pad=10)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_main_results.png"), dpi=220)
    plt.close()


def plot_efficiency():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.0, 4.2), dpi=220)

    patches = [256, 576, 1024, 2304, 4096]
    tflops_ours = [0.30, 0.45, 0.67, 1.37, 2.54]
    tflops_rose = [0.47, 0.86, 1.45, 3.40, 6.82]
    tflops_llava = [0.66, 1.27, 2.18, 5.07, 9.85]

    ax1.plot(patches, tflops_llava, "o--", color=C_LLAVA, linewidth=2, label="LLaVA (Ext. Tower + Full Tokens)")
    ax1.plot(patches, tflops_rose, "s-.", color=C_TOWER, linewidth=2, label="ROSE / SpinVL-Tower (Ext. Tower)")
    ax1.plot(patches, tflops_ours, "D-", color=C_FREE, linewidth=2.5, label="SpinVL-Free (Ours, -74.2% FLOPs)")
    ax1.fill_between(patches, tflops_ours, tflops_llava, color=C_FREE, alpha=0.08)
    ax1.annotate("-74.2% FLOPs\n(2.54 vs 9.85 TFLOPs)", xy=(4096, 2.54), xytext=(2100, 5.8),
                 arrowprops=dict(arrowstyle="->", color=C_FREE, lw=1.5),
                 fontsize=8.5, fontweight="bold", color=C_FREE)
    ax1.set_xlabel("Number of Visual Patches", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Forward TFLOPs", fontsize=9.5, fontweight="bold")
    ax1.set_title("(a) Compute Scaling vs. Image Resolution", fontsize=10.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=7.8, loc="upper left")

    # Panel B: Exclusive-GPU Latency at high resolution
    ms_ours = [65.8, 66.6, 69.8, 80.4, 99.4]
    ms_rose = [58.2, 57.8, 63.8, 86.0, 126.1]
    ms_llava = [25.7, 25.6, 33.7, 66.9, 123.4]
    ax2.plot(patches, ms_llava, "o--", color=C_LLAVA, linewidth=2, label="LLaVA (Full Tokens)")
    ax2.plot(patches, ms_rose, "s-.", color=C_TOWER, linewidth=2, label="ROSE (Ext. Tower)")
    ax2.plot(patches, ms_ours, "D-", color=C_FREE, linewidth=2.5, label="SpinVL-Free (Ours)")
    ax2.annotate("Faster at High-Res\n99.4ms vs 123.4ms", xy=(4096, 99.4), xytext=(2000, 112),
                 arrowprops=dict(arrowstyle="->", color=C_FREE, lw=1.5),
                 fontsize=8.5, fontweight="bold", color=C_FREE)
    ax2.set_xlabel("Number of Visual Patches", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Exclusive-GPU Latency (ms)", fontsize=9.5, fontweight="bold")
    ax2.set_title("(b) End-to-End Latency Crossover", fontsize=10.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=8, loc="upper left")

    # Panel C: Data Scaling Curve (P2, N=1k..21k)
    Ns = [1, 3, 6, 12, 21]
    ce_acc = [19.7, 21.1, 20.5, 21.6, 22.4]
    rel_acc = [18.6, 18.7, 23.3, 19.4, 19.3]
    ax3.plot(Ns, ce_acc, "o-", color=C_TOWER, linewidth=2.2, label="CE Arm (Scales Best at Large N)")
    ax3.plot(Ns, rel_acc, "s--", color=C_FREE06, linewidth=2.0, label="KD+Rel Arm (3.5x Sample Eff. @ N=6k)")
    ax3.axhline(22.4, color=C_LLAVA, linestyle=":", linewidth=1.2, label="CE @ N=21k Target (22.4%)")
    ax3.set_xlabel("Training Samples N (in thousands)", fontsize=9.5, fontweight="bold")
    ax3.set_ylabel("Val Full-Answer Accuracy (%)", fontsize=9.5, fontweight="bold")
    ax3.set_title("(c) Data Scaling & Sample Efficiency", fontsize=10.5, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.35)
    ax3.legend(fontsize=7.8, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_efficiency.png"), dpi=220)
    plt.close()


def plot_ablations():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.0, 4.2), dpi=220)

    # Panel A: Frontend Mode Ablation (MC2 & Core3)
    modes = ["none\n(K=0)", "LoRA\n(VoRA)", "full\n(Copy K=4)", "rot\n(SO(64))", "rots+hyb\n(Ours)"]
    mc2 = [35.92, 36.84, 38.82, 40.73, 42.68]
    core3 = [31.15, 30.73, 32.08, 33.89, 34.95]
    x1 = np.arange(len(modes))
    w = 0.35
    b1 = ax1.bar(x1 - w / 2, mc2, width=w, label="MC2 (MMStar+Hallu)", color=C_FREE, edgecolor="#1E293B")
    b2 = ax1.bar(x1 + w / 2, core3, width=w, label="Core3 Avg", color=C_TOWER, edgecolor="#1E293B")
    for b in list(b1) + list(b2):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{b.get_height():.1f}",
                 ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(modes, fontsize=8.5, fontweight="bold")
    ax1.set_ylim(25, 47)
    ax1.set_ylabel("Accuracy (%)", fontsize=9.5, fontweight="bold")
    ax1.set_title("(a) Built-in Frontend Adaptation Mode", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8, loc="upper left")

    # Panel B: Frontend Depth K in {0, 2, 4, 6}
    Ks = ["K=0", "K=2", "K=4\n(Ours)", "K=6"]
    tf_first = [30.7, 36.7, 37.3, 38.0]
    gap = [0.39, 0.81, 1.07, 0.98]
    x2 = np.arange(len(Ks))
    ax2_r = ax2.twinx()
    p1 = ax2.bar(x2 - 0.18, tf_first, width=0.34, color=C_TOWER, label="Val 1st-Token Acc (%)", edgecolor="#1E293B")
    p2 = ax2_r.plot(x2 + 0.18, gap, "D-", color=C_FREE, linewidth=2.2, markersize=7, label="Visual Grounding Gap (CE)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(Ks, fontsize=9, fontweight="bold")
    ax2.set_ylim(20, 44)
    ax2_r.set_ylim(0.0, 1.35)
    ax2.set_ylabel("1st-Token Accuracy (%)", fontsize=9.5, fontweight="bold", color=C_TOWER)
    ax2_r.set_ylabel("Visual Grounding Gap (Shuffled - Paired CE)", fontsize=8.8, fontweight="bold", color=C_FREE)
    ax2.set_title("(b) Built-in Encoder Depth K Ablation", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel C: Step 2 VL-OPD Free-Generation Gain (--force-gen)
    tasks = ["MMStar\n(0.6B Free)", "Hallusion\n(0.6B Free)", "MathVista\n(0.6B Free)", "MathVista\n(1.7B Free)", "Mix Val EM\n(1.7B Tower)"]
    before = [14.53, 23.55, 9.70, 12.20, 16.67]
    after = [22.20, 33.44, 14.40, 16.30, 20.67]
    x3 = np.arange(len(tasks))
    b_pre = ax3.bar(x3 - w / 2, before, width=w, label="Before VL-OPD (Offline CE)", color=C_VORA, edgecolor="#1E293B")
    b_post = ax3.bar(x3 + w / 2, after, width=w, label="After VL-OPD (On-Policy KL)", color=C_FREE, edgecolor="#1E293B")
    for b0, b1_bar in zip(b_pre, b_post):
        gain = b1_bar.get_height() - b0.get_height()
        ax3.text(b1_bar.get_x() + b1_bar.get_width() / 2, b1_bar.get_height() + 0.6, f"+{gain:.1f}%",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=C_FREE)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(tasks, fontsize=8, fontweight="bold")
    ax3.set_ylim(0, 39)
    ax3.set_ylabel("Free-Generation Accuracy (%)", fontsize=9.5, fontweight="bold")
    ax3.set_title("(c) Step 2 VL-OPD Cures Exposure Bias", fontsize=10.5, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.35)
    ax3.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_ablations.png"), dpi=220)
    plt.close()


if __name__ == "__main__":
    plot_main_results()
    plot_efficiency()
    plot_ablations()
    print("ALL 3 FIGURES GENERATED IN", OUT_DIR)
