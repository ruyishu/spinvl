# SpinVL: Vision-Tower-Free & Zero-Visual-Token VLM

![SpinVL Architecture](assets/fig_architecture.png)

**SpinVL** turns frozen pure-text LLMs (`Qwen3-0.6B / 1.7B / 8B`) into vision-language models **without an external vision tower** and **without feeding visual tokens into the deeper LLM backbone**:

- **Zero External Vision Tower & -74.2% FLOPs**: Repurposes the student LLM's first $K=4$ layers via static orthogonal rotation + diagonal scaling (`PatchFront`), then **drops all image tokens**.
- **Zero Backbone Visual Tokens**: Modulates deeper frozen layers via multi-frequency (`mf`) or spatial-anchor (`hybrid`) block-diagonal Cayley rotations $R_v \in SO(64)$ (`LayerVRMG`).
- **100% Lossless Pure-Text**: Vision rotations are decoupled from the text pathway (`GSM8K 70.4%`, `ARC-C 71.4%` bit-exact).
- **Online Policy Distillation (`VL-OPD`)**: 500-step on-policy token-level KL distillation from `Qwen3-VL-8B-Instruct` eliminates autoregressive exposure bias.

---

## 1. Method

### Step 1: Dual-Tier Orthogonal Rotation
1. **Shallow Built-in Encoder (`src/front.py`, $K=4$)**:
   Image patches pass through the student's first $K=4$ layers under bidirectional attention with static Cayley rotation and channel-wise scaling:
   $$W' = W \, R_{\text{static}} \, \operatorname{diag}(d), \quad R_{\text{static}} = (I - P)^{-1}(I + P) \in SO(64)$$
   A $2\times 2$ merger outputs cell features $\hat{v} \in \mathbb{R}^{N_c \times d_v}$. **All image tokens are discarded after layer $K$.**
2. **Deep Dynamic Rotation (`src/vrmg.py`, Layers $[0,4,8,12,16,20,24]$)**:
   `LayerVRMG` maps visual features ($v$ in `SpinVL-Tower`, or $\hat{v}$ in `SpinVL-Free`) to block-diagonal orthogonal matrices $R_v \in SO(64)$ rotating linear inputs $z = (x R_v) W^\top$, enhanced by:
   - **`multifreq` (`mf`)**: Low/Mid/High-frequency 2D positional bands + zero-init channel gate $2\sigma(g(x))$.
   - **`hybrid`**: Orthogonal rotation + 256-dim cross-attention over $M=16$ learnable spatial anchors.

### Step 2: Online Policy Distillation (`scripts/train_opd.py`, `scripts/train_joint_opd.py`)
The student generates on-policy rollouts $\hat{y} \sim \pi_\theta(\cdot \mid I, q)$; the teacher supervises the student's exact trajectory via full-vocabulary forward KL divergence + gold-answer CE anchoring.

---

## 2. Main Results

![Main Results and Pure-Text Preservation](assets/fig_main_results.png)

| Model | Ext. Tower | Deep Vis Tokens | FLOPs (@4096p) | GSM8K (Text) | **MMStar** | **Hallu.** | **MathV.** | **Geo3K** (Gen / 1st) | **MC2 Avg** | **Core3 Avg** | **6-Bench Avg** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `LLaVA-Proj (21k)` | 8B Vis | All 28L | 9.85T | 70.4% | 29.60% | 53.10% | 24.90% | **6.49%** / — | 41.35% | 35.87% | **21.02%** |
| `LLaVA-Proj (28k)` | 8B Vis | All 28L | 9.85T | 70.4% | 29.27% | 54.15% | **25.00%** | 5.49% / — | 41.71% | 36.14% | 20.89% |
| `ROSE (Single-Freq)` | 8B Vis | 0 | 6.82T | 70.4% | 25.60% | 54.36% | 19.70% | 2.00% / 95.8% | 39.98% | 33.22% | 17.14% |
| `VoRA (LoRA K=4)` | None | 0 | 2.54T | 0.0% (merged) | 23.73% | 49.95% | 18.50% | 1.33% / — | 36.84% | 30.73% | 15.82% |
| **SpinVL-Tower (`hybrid`)** | 8B Vis | **0** | 6.82T | **70.4%** | **30.60%** | **58.15%** | 18.50% | 1.83% / **96.0%** | **44.37%** | 35.75% | 18.36% |
| **SpinVL-Tower (`mf+OPD`)** | 8B Vis | **0** | 6.82T | **70.4%** | 26.80% | **57.41%** | **23.50%** | 2.00% / **96.3%** | 42.11% | **35.90%** | **18.50%** |
| **SpinVL-Free+OPD (`1.7B`)** | **None** | **0** | **2.54T** | **70.4%** | **29.20%** | **56.99%** | 19.80% | 2.16% / — | **42.68%** | **34.95%** | **18.02%** |
| **SpinVL-Free (`1.7B No-Align`)** | **None** | **0** | **2.54T** | **70.4%** | 27.53% | **57.73%** | 19.40% | 2.66% / — | **42.63%** | **34.89%** | **18.13%** |
| **SpinVL-Free+OPD (`0.6B`)** | **None** | **0** | **0.91T** | **100% Kept** | **27.53%** | **57.83%** | 20.40% | **3.16%** / **96.2%** | **42.68%** | **34.66%** | **18.05%** |
| **SpinVL-Free (`8B`)** | **None** | **0** | — | **100% Kept** | **27.40%** | **56.99%** | 19.20% | 2.50% / **96.2%** | **40.90%** | **33.66%** | **17.50%** |

---

## 3. Efficiency, Scaling & Ablations

![Efficiency and Data Scaling](assets/fig_efficiency.png)

![Ablation Studies](assets/fig_ablations.png)

- **Compute & Latency (`results/paper_results.json`)**: At 4,096 visual patches, `SpinVL-Free` reduces forward FLOPs from **9.85 TFLOPs (`LLaVA`) to 2.54 TFLOPs (-74.2%)** and cuts exclusive-GPU latency from **123.4 ms to 99.4 ms (-19.5%)**.
- **Built-in Frontend (`rots_K4` vs. `LoRA` / `K=0..6`)**: `rots_K4` outperforms `VoRA (LoRA K=4)` by **+5.84%** on `MC2` (`42.68%` vs `36.84%`). Increasing $K$ from $0 \to 2 \to 4$ improves visual grounding `gap` (`+0.39 -> +0.81 -> +1.07`) and 1st-token accuracy (`30.7% -> 36.7% -> 37.3%`).
- **`VL-OPD` Free-Generation Gain**: Under forced free-form greedy generation (`--force-gen`), `VL-OPD` boosts `MMStar` by **+7.7%**, `HallusionBench` by **+9.9%**, and `MathVista` by **+4.7%**, with **95%–100%** EOS termination.
- **Mechanism Boundary (`ChartQA / OCRBench`)**: Zero-token orthogonal rotation acts as a global/mid-grained operator modulation (excelling at perception, anti-hallucination, and math reasoning), whereas dense character-level transcription requires token-level KV addressing.

---

## 4. Code Structure

```text
├── src/
│   ├── vrmg.py               # LayerVRMG (Cayley SO(64), multifreq, hybrid) & RotatedLinear
│   ├── front.py              # PatchFront (built-in K-layer vision encoder: rot, rots, lora, full)
│   ├── wrap.py               # Non-invasive LLM rotation hook & context switch
│   └── imgdata.py            # Multi-source image & memmap feature loader
├── scripts/
│   ├── train_final.py        # Stage 1: SpinVL-Tower training
│   ├── train_front.py        # Stage 2A: PatchFront warm-up
│   ├── train_joint.py        # Stage 2B: SpinVL-Free end-to-end joint training
│   ├── train_opd.py          # Step 2: VL-OPD for SpinVL-Tower
│   ├── train_joint_opd.py    # Step 2: VL-OPD for SpinVL-Free
│   ├── train_proj.py         # Baseline: LLaVA-Proj token concatenation
│   ├── eval_benchmarks.py    # Unified 6-benchmark zero-shot evaluation
│   ├── bench_efficiency.py   # FLOPs & latency profiler (256..4096 patches)
│   ├── eval_lang.py          # GSM8K & ARC-Challenge pure-text retention
│   └── plot_results.py       # Generate paper figures in assets/
├── results/
│   ├── paper_results.json    # Consolidated main, efficiency, and language tables
│   └── benchmarks/           # Raw evaluation JSONs for core models
└── assets/                   # Architecture & result figures
```

---

## 5. Quick Start

```bash
# 1. Stage 1: Train SpinVL-Tower
python scripts/train_final.py --arm ce --data "$DATA" --layers "0,4,8,12,16,20,24" \
  --block 64 --query-dim 256 --bs 8 --seq 256 --steps 6000 --hybrid \
  --log tower.log --result-json tower.json

# 2. Stage 2: Train SpinVL-Free (PatchFront rots K=4 + LayerVRMG)
python scripts/train_front.py --mode rots --K 4 --source "$SRC" --steps 2500 --bs 16 --out front_rots
python scripts/train_joint.py --stage joint --front-mode rots --K 4 --arm ce --hybrid \
  --data "$DATA" --source "$SRC" --vrmg-ckpt tower_pt.pt --front-ckpt front_rots.pt \
  --steps 3000 --bs 8 --out spinvl_free

# 3. Step 2: Online Policy Distillation (VL-OPD)
python scripts/train_joint_opd.py --data "$DATA" --source "$SRC" \
  --joint-ckpt spinvl_free.pt --hybrid --steps 500 --bs 4 --max-new 16 --out spinvl_free_opd

# 4. Evaluate on 6 Public Benchmarks & Regenerate Figures
python scripts/eval_benchmarks.py --mode tower_free --ckpt spinvl_free_opd.pt --name spinvl_free_opd
python scripts/plot_results.py
```
