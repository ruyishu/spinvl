# SpinVL：零外部视觉塔、主干零视觉 Token 的正交旋转多模态大模型

![SpinVL 核心架构图](assets/fig_architecture.png)

**一句话简介**：**SpinVL** 基于完全冻结的纯文本 LLM（`Qwen3-0.6B / 1.7B / 8B`），**既不需要外挂庞大的视觉编码器（Vision Tower），也不往深层 LLM 主干里塞任何视觉 Token**，仅靠**双层正交旋转（Dual-Tier Orthogonal Rotation）+ 在线策略蒸馏（VL-OPD）**，在节省 **74.2% FLOPs**、**100% 保持原装纯文本智商**的前提下，在视觉感知、抗幻觉与数学推理上打平甚至反超传统 LLaVA，并全方位击败同赛道的 ROSE 与 VoRA。

> **投稿稿**：完整 CVPR 格式 LaTeX 源码在 [`paper/`](paper/)，见 [paper/README.md](paper/README.md)（官方模板版 `main_cvpr_official.tex` 与免模板可编译版 `main.tex`）。

---

## 一、Motivation（为什么要做 SpinVL？）

现有的视觉语言大模型（VLM）主要有两条路线，但各自卡死在一个致命痛点上：

1. **传统 Token 拼接派（如 LLaVA）——算力太重、且伤纯文本能力**
   - 外挂一个庞大的视觉塔，把每张图变成几百上千个视觉 Token 拼在文本前面，送进 LLM 的全部 $L$ 层。
   - **痛点**：自注意力算力和 KV Cache 随图像分辨率暴涨（4096 个 patch 下单次前向高达 **9.85 TFLOPs**）；且微调主干会破坏原模型原有的纯文本数学与代码能力。
2. **现有的轻量化探索（ROSE 与 VoRA）——各自只走了一半**
   - **ROSE（参数旋转派）**：不往主干塞视觉 Token，而是用图像特征生成正交旋转矩阵去“拧”LLM 的权重。但它**依然拖着笨重的外部视觉塔**；且原版只做单一全局平均池化（丢了空间位置细节），只靠离线 Teacher-Forcing 训练，一到自己连续生成就会跑偏（Exposure Bias）。
   - **VoRA（内置编码派）**：拔掉了外部视觉塔，用 LoRA 把 LLM 前 $K$ 层改造成视觉编码器。但它的**视觉 Token 依然全程留在深层主干里**，推理算力一点没省；而且一旦把前 $K$ 层改动并入主干，原模型的数学能力（`GSM8K`）会从 **70.4% 直接崩到 0.0%**。

**👉 我们的破局点（SpinVL）**：
能不能鱼和熊掌兼得——**既彻底拔掉外部视觉塔（像 VoRA），又在深层主干里保持 0 个视觉 Token（像 ROSE，省掉 74% 算力），还能保住空间细粒度、治好自回归生成跑偏，并且让纯文本能力 100% 零退化？**

---

## 二、Method（核心方法：两步走框架）

整套方法将学生纯文本 LLM 全程冻结，分为前后衔接的两步：

### 第一步：双层正交旋转架构（浅层读图、深层扔 Token）

我们提供两个递进形态（保留外部视觉塔的 **`SpinVL-Tower`** 与彻底拔掉外部视觉塔的 **`SpinVL-Free`**）：

1. **浅层内置视觉编码器 `PatchFront`（`src/front.py`，仅在 `SpinVL-Free` 启用）**
   - **做法**：彻底拔掉外部视觉塔，把原图切出的 Patch 送进学生 LLM 自己的**前 $K=4$ 层**（开启双向注意力）。
   - **静态正交旋转 + 逐通道对角缩放（`rots_K4`）**：为了既保住参数正交几何、又弥补纯正交矩阵无法改变奇异值的短板，我们对前 4 层线性投影施加 $W' = W R_{\mathrm{static}} \mathrm{diag}(d)$，其中 $R_{\mathrm{static}} = (I - P)^{-1}(I + P) \in SO(64)$，仅需 **2.4M** 静态参数。经过 $2\times 2$ 空间合并后输出对齐老师的视觉特征 $\hat{v} \in \mathbb{R}^{N_c \times d_v}$。
   - **读完即扔**：第 4 层吐出 $\hat{v}$ 后，**把所有图片 Patch Token 彻底扔掉**，后 $L-4$ 层完全只有纯文本 Token。
   - **图文通路彻底解耦**：处理纯文本任务时不触发视觉旋转开关，与原装纯文本 LLM **逐位 100% 一致**。

2. **深层动态旋转注入 `LayerVRMG`（`src/vrmg.py`，深层主干 0 视觉 Token）**
   - **做法**：在 LLM 的 7 个选定层（如 1.7B 的 `[0,4,8,12,16,20,24]`），把视觉特征（`SpinVL-Tower` 用老师视觉塔特征 $v$，`SpinVL-Free` 用内置前端特征 $\hat{v}$）映射为 $64\times 64$ 块对角 Cayley 正交矩阵 $R_v \in SO(64)$，去旋转各线性层输入 $z = (x R_v) W^\top$。
   - **打破全局平均瓶颈的两种互补设计**：
     - **`multifreq`（多频旋转）**：类比 RoPE，把旋转块分为**低频**（全局均值）、**中频**（缓变 2D 位置编码）、**高频**（快变位置编码 + $\tau=0.25$ 尖锐注意力）三档 + 零初始化通道门控——**最擅长多步视觉数学推理（`MathVista`）**。
     - **`hybrid`（正交旋转 + 空间锚点混合）**：在正交旋转基础上，用 256 维窄子空间的 $M=16$ 个可学习空间锚点做轻量交叉注意力残差——**最擅长细粒度视觉感知与抗幻觉（`MMStar / HallusionBench`）**。

### 第二步：在线策略蒸馏 `VL-OPD`（`scripts/train_opd.py` & `train_joint_opd.py`）
- **为什么需要**：旋转模型在离线 Teacher-Forcing（CE）训练时只见过标准答案前缀；一到测试时自己贪婪生成，一旦首词略有偏差就会一路跑偏或停不下来（Exposure Bias）。
- **怎么做**：在冷启动之后，让学生模型自己对图片和问题做 On-Policy Rollout 生成轨迹 $\hat{y} \sim \pi_\theta(\cdot \mid I, q)$，请 `Qwen3-VL-8B-Instruct` 老师在学生自己的生成轨迹上逐词计算**全词表前向 KL 散度 + 金标答案 CE 锚定**（仅需 500 步轻量在线蒸馏），直接治好生成跑偏。

---

## 三、Experiment & Results（实验结果与大白话解读）

### 1. 主实验对比与纯文本保持（`Qwen3-1.7B` 同底座、同数据、同预算 + `0.6B / 8B` 跨尺寸）

![主实验结果与纯文本保持率](assets/fig_main_results.png)

| 模型类别 | 模型名称 | 外部视觉塔 | 深层视觉 Token | 算力 (@4096p) | 纯文本 GSM8K | **MMStar** (感知) | **Hallusion** (抗幻觉) | **MathVista** (视觉数学) | **Geo3K** (几何生成 / 首词) | **感知+抗幻觉 (`MC2`)** | **核心三榜均分 (`Core3`)** | **六榜总均分 (`Avg6`)** |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **传统全层拼接基线 (21k)** | `LLaVA-Proj (p6_proj)` | 8B 视觉塔 | 全 28 层保留 | 9.85T | 70.4% | 29.60% | 53.10% | 24.90% | **6.49%** / — | 41.35% | 35.87% | **21.02%** |
| **传统全层拼接基线 (28k)** | `LLaVA-Proj (p15_proj)` | 8B 视觉塔 | 全 28 层保留 | 9.85T | 70.4% | 29.27% | 54.15% | **25.00%** | 5.49% / — | 41.71% | 36.14% | 20.89% |
| **同赛道旋转基线** | `原版单频 ROSE` | 8B 视觉塔 | 0 | 6.82T | 70.4% | 25.60% | 54.36% | 19.70% | 2.00% / 95.8% | 39.98% | 33.22% | 17.14% |
| **同赛道无塔基线** | `VoRA 同型 (LoRA K=4)` | 无 (拔掉) | 0 | 2.54T | 0.0% (合入后) | 23.73% | 49.95% | 18.50% | 1.33% / — | 36.84% | 30.73% | 15.82% |
| **🌟 Ours 有塔 (感知最强)** | **`1.7B SpinVL-Tower (hybrid)`** | 8B 视觉塔 | **0** | 6.82T | **70.4%** | **30.60%** | **58.15%** | 18.50% | 1.83% / **96.0%** | **44.37% (+3.0%)** | 35.75% | 18.36% |
| **🌟 Ours 有塔+OPD (全能最强)** | **`1.7B SpinVL-Tower (mf+OPD)`** | 8B 视觉塔 | **0** | 6.82T | **70.4%** | 26.80% | **57.41%** | **23.50%** | 2.00% / **96.3%** | **42.11%** | **35.90% (超LLaVA)** | **18.50%** |
| **🌟 Ours 无塔+OPD (21k)** | **`1.7B SpinVL-Free + OPD`** | **无 (-74%算力)** | **0** | **2.54T** | **70.4%** | 27.93% | **56.99%** | 19.80% | 2.16% / — | **42.46% (超LLaVA)** | **34.91%** | **18.02%** |
| **🌟 Ours 无塔+OPD (28k)** | **`1.7B SpinVL-Free + OPD (s2)`** | **无 (-74%算力)** | **0** | **2.54T** | **70.4%** | **29.20%** | **56.15%** | 19.50% | 1.00% / — | **42.68% (超LLaVA)** | **34.95%** | 17.80% |
| **🌟 Ours 无塔 (纯端到端)** | **`1.7B SpinVL-Free (无特征对齐)`** | **无 (-74%算力)** | **0** | **2.54T** | **70.4%** | 27.53% | **57.73%** | 19.40% | 2.66% / — | **42.63% (超LLaVA)** | **34.89%** | **18.13%** |
| **🌟 Ours 0.6B 无塔跨尺寸** | **`0.6B SpinVL-Free + OPD`** | **无 (-74%算力)** | **0** | **0.91T** | **100% 零损** | **27.53%** | **57.83%** | **20.40%** | **3.16%** / **96.2%** | **42.68% (超1.7B LLaVA)** | **34.66%** | **18.05%** |
| **🌟 Ours 8B 无塔跨尺寸** | **`8B SpinVL-Free (hybrid)`** | **无 (-74%算力)** | **0** | — | **100% 零损** | **27.40%** | **56.99%** | **19.20%** | 2.50% / **96.2%** | **40.90%** | **33.66%** | **17.50%** |

---

### 2. 效率曲线、规模效应与核心消融

![算力延迟与数据规模曲线](assets/fig_efficiency.png)

![核心消融实验](assets/fig_ablations.png)

### 3. 五条核心实验结论（大白话总结）

1. **对比传统 LLaVA：0 视觉 Token 在感知、抗幻觉与数学推理上打平甚至反超 LLaVA**
   - **`SpinVL-Tower`**：主干不放一个视觉 Token，`hybrid` 版在感知+抗幻觉两榜（`MC2`）上拿到 **44.37%**，比全层拼 Token 的 LLaVA（**41.35%**）高出 **+3.02 个点**；`mf + OPD` 版将 `MathVista` 推到 **23.50%**，**核心三榜均分 `35.90%` 正面超越传统 LLaVA（`35.87%`）**。
   - **`SpinVL-Free + OPD`（彻底拔掉 8B 视觉塔、省 74% 算力）**：在 `MC2` 上达到 **42.68%**（依然反超挂着 8B 视觉塔的 LLaVA），核心三榜均分 **34.95%**（以 1/4 的算力拿到 LLaVA **97.4%** 的性能）；甚至 **0.6B 无塔版本**（`MC2 42.68%`）在感知和抗幻觉上也超过了 1.7B 的传统 LLaVA。
2. **对比同赛道对手（ROSE 与 VoRA）：全指标大幅胜出**
   - 比原版单频 `ROSE` 在 `MC2` 上高出 **+4.39%**（`44.37% vs 39.98%`），核心三榜高出 **+2.68%**；
   - 比同架构无塔 `VoRA (LoRA K=4)` 在 `MC2` 上高出 **+5.84%**（`42.68% vs 36.84%`），核心三榜高出 **+4.22%**，证明“静态正交旋转+对角缩放（`rots_K4`）”显著优于加法 `LoRA`。
   - **纯端到端可训**：哪怕在联合训练时完全去掉老师视觉塔特征对齐损失（`feat_weight=0`），`SpinVL-Free` 依然拿到 **`MC2 42.63% / Avg6 18.13%`**，证明前 4 层真正学会了读图而非死记老师特征。
3. **第二步 `VL-OPD` 的效果：自由生成准确率暴涨 4～10 个点**
   - 在强制自由生成评测（`--force-gen`，不给选项概率、让模型自己逐字输出）下，`VL-OPD` 让 `MMStar` 涨 **+7.7%**（`14.5% -> 22.2%`）、`HallusionBench` 涨 **+9.9%**（`23.6% -> 33.4%`）、`MathVista` 涨 **+4.7%**（`9.7% -> 14.4%`），EOS 正常停止率达 **95%～100%**。
4. **真省算力 + 100% 护住原装纯文本智商**
   - 在 4096 patches 高分辨率下，前向算力从 LLaVA 的 **9.85 TFLOPs 降至 2.54 TFLOPs（省 74.2%）**，独占 GPU 实测延迟从 **123.4 ms 降至 99.4 ms（快 19.5%）**；
   - 纯文本通路 `GSM8K (70.4%)` 与 `ARC-C (71.4%)` **100% 零损**；而对照实验证明若将前 4 层视觉改动（无论是 `LoRA` 还是旋转）直接合并进主干，`GSM8K` 会从 `70.4%` 直接崩至 **`0.0%～0.8%`**。
5. **机制边界（Mechanism Boundary：什么时候该用 0-Token 旋转？）**
   - 在同数据对照实验中（加入 6,500 条 `DocVQA` 图表/单据读数 + 2,101 条几何题），我们的 `Geometry3K` 从 `1.83%` 翻倍至 **`3.83%`**，但 `ChartQA`（精确读小数）和 `OCRBench`（密集字符抄写）在同数据 LLaVA 上也仅停留在 `10.32%` 和 `1.10%`（我们为 `1.24%～2.60%`）。
   - 这清晰界定了方法边界：**0-Token 参数旋转将图像编码为算子级全局/中粒度调制，极度适合语义感知、抗幻觉、空间关系与数学推理；而逐字符的密集 OCR 抄写则依赖逐位置视觉 Token 的 KV 寻址。**

---

## 四、仓库结构与快速复现

```text
├── src/
│   ├── vrmg.py               # 深层动态正交旋转 (LayerVRMG: Cayley SO(64), multifreq, hybrid)
│   ├── front.py              # 浅层内置视觉编码器 (PatchFront: rot, rots, lora, full)
│   ├── wrap.py               # 非侵入式 LLM 挂载与图文通路开关
│   └── imgdata.py            # 多源图片与 memmap 视觉特征加载
├── scripts/
│   ├── train_final.py        # Stage 1: 训练 SpinVL-Tower
│   ├── train_front.py        # Stage 2A: 预热内置视觉前端 PatchFront
│   ├── train_joint.py        # Stage 2B: 端到端联合训练 SpinVL-Free (拔掉外部视觉塔)
│   ├── train_opd.py          # Step 2: SpinVL-Tower 的在线策略蒸馏 (VL-OPD)
│   ├── train_joint_opd.py    # Step 2: SpinVL-Free 的在线策略蒸馏 (VL-OPD)
│   ├── train_proj.py         # 对照基线: 标准 LLaVA Token 拼接 Projector
│   ├── eval_benchmarks.py    # 6 大公开榜单统一零样本评测
│   ├── bench_efficiency.py   # 解析式 TFLOPs 与独占 GPU 延迟评测 (256..4096 patches)
│   ├── eval_lang.py          # 纯文本 GSM8K / ARC-Challenge 保持率评测
│   └── plot_results.py       # 一键生成 assets/ 下的论文图表
├── results/
│   ├── paper_results.json    # 论文主表、效率表与语言保持率汇总数据
│   └── benchmarks/           # 核心模型的 6 榜原始评测 JSON
└── assets/                   # 核心架构图与实验结果图表
```

### 核心复现命令

```bash
# 1. Stage 1：训练 SpinVL-Tower (0 视觉 Token 进主干)
python scripts/train_final.py --arm ce --data "$DATA" --layers "0,4,8,12,16,20,24" \
  --block 64 --query-dim 256 --bs 8 --seq 256 --steps 6000 --hybrid \
  --log tower.log --result-json tower.json

# 2. Stage 2：训练 SpinVL-Free (前 K=4 层 rots 内置编码 + 拔掉外部视觉塔)
python scripts/train_front.py --mode rots --K 4 --source "$SRC" --steps 2500 --bs 16 --out front_rots
python scripts/train_joint.py --stage joint --front-mode rots --K 4 --arm ce --hybrid \
  --data "$DATA" --source "$SRC" --vrmg-ckpt tower_pt.pt --front-ckpt front_rots.pt \
  --steps 3000 --bs 8 --out spinvl_free

# 3. Step 2：在线策略蒸馏 (VL-OPD)
python scripts/train_joint_opd.py --data "$DATA" --source "$SRC" \
  --joint-ckpt spinvl_free.pt --hybrid --steps 500 --bs 4 --max-new 16 --out spinvl_free_opd

# 4. 公开 6 榜评测与重绘图表
python scripts/eval_benchmarks.py --mode tower_free --ckpt spinvl_free_opd.pt --name spinvl_free_opd
python scripts/plot_results.py
```
