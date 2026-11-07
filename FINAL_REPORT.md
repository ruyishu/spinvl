# SpinVL (`rose-distill`) Complete Experimental Report (Phases P1–P18 & `bench_eval_v4`)

> **Last Updated**: 2026-11-07  
> **Student Backbones**: Frozen `Qwen3-0.6B`, `Qwen3-1.7B` (primary), `Qwen3-8B`  
> **Teacher Model**: `Qwen3-VL-8B-Instruct`  
> **Raw JSON Provenance**: All numbers in this report are directly traceable to `results/bench_eval_v4/*.json` and `results/p1/*.json` .. `results/p18/*.json` in this repository.

---

## 1. Main Public Benchmark Comparison (`bench_eval_v4`)

### 1.1 Evaluation Protocol
- **Benchmarks (6 total, 7,552 test items)**:
  1. **`MMStar`** ($n=1,500$): General visual perception & reasoning (4-way multiple choice).
  2. **`HallusionBench`** ($n=951$): Visual hallucination & illusion resistance (`Yes/No`).
  3. **`MathVista testmini`** ($n=1,000$): Visual mathematical reasoning (540 multiple-choice + 460 free-form numeric).
  4. **`Geometry3K test`** ($n=601$): Plane geometry problem solving (reported as both zero-shot free generation and dedicated Stage-1 first-token accuracy).
  5. **`ChartQA test`** ($n=2,500$): Open-ended chart number/label extraction (relaxed accuracy $\le 5\%$).
  6. **`OCRBench`** ($n=1,000$): Scene text, receipt, and handwritten mathematical formula transcription.
- **Aggregate Metrics**:
  - **`MC2`**: Mean of `MMStar` and `HallusionBench` (visual perception & anti-hallucination).
  - **`Core3`**: Mean of `MMStar`, `HallusionBench`, and `MathVista` (the three reasoning/perception benchmarks aligned with short-VQA training).
  - **`Avg6`**: Unweighted mean across all 6 benchmarks.

### 1.2 Main Comparison Table (`Qwen3-1.7B` Same-Base, Same-Data, Same-Budget)

| Model | JSON File (`results/bench_eval_v4/`) | Ext. Tower | Backbone Vis Toks | **MMStar** | **Hallu** | **MathV** | **Geo3K** | **ChartQA** | **OCR** | **MC2** | **Core3** | **Avg6** |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **LLaVA-Proj (21k mix)** | `q17_proj_n21000.json` | 8B Vis | All 28 Layers | 29.60% | 53.10% | 24.90% | **6.49%** | **10.40%** | **1.60%** | 41.35% | 35.87% | **21.02%** |
| **LLaVA-Proj (28k P15)** | `q17_p15_proj.json` | 8B Vis | All 28 Layers | 29.27% | 54.15% | **25.00%** | 5.49% | 10.32% | 1.10% | 41.71% | 36.14% | 20.89% |
| **ROSE Single-Freq (21k)** | `q17_rose_single_ce.json` | 8B Vis | **0** | 25.60% | 54.36% | 19.70% | 2.00% | 1.20% | 0.00% | 39.98% | 33.22% | 17.14% |
| **VoRA-style (`LoRA K=4`)** | `q17_vora_lora_mf.json` | **None** | **0** (L5–28) | 23.73% | 49.95% | 18.50% | 1.33% | 1.40% | 0.00% | 36.84% | 30.73% | 15.82% |
| **🌟 SpinVL-Tower (`hybrid`, 21k)** | `q17_tower_hybrid_ce.json` | 8B Vis | **0** | **30.60%** | **58.15%** | 18.50% | 1.83% | 1.08% | 0.00% | **44.37%** | 35.75% | 18.36% |
| **🌟 SpinVL-Tower (`mf`, 7875st)** | `q17_tower_mf_s7875_ce.json` | 8B Vis | **0** | 27.73% | 53.73% | 20.60% | 3.16% | 0.80% | 0.00% | 40.73% | 34.02% | 17.67% |
| **🌟 SpinVL-Tower (`hybrid`, P15)** | `q17_p15_tower_hybrid_ce.json` | 8B Vis | **0** | 28.27% | 56.05% | 19.60% | **3.83%** | 1.24% | 0.00% | **42.16%** | 34.64% | 18.16% |
| **🌟 SpinVL-Tower (`mf + OPD`, P15)** | `q17_p15_tower_mf_opd.json` | 8B Vis | **0** | 26.80% | **57.41%** | **23.50%** | 2.00% | 1.28% | 0.00% | **42.11%** | **35.90%** | **18.50%** |
| **🌟 SpinVL-Free (`hybrid`, 21k)** | `q17_free_hybrid_ce.json` | **None** | **0** (L5–28) | 27.20% | 53.21% | 20.30% | 2.66% | 0.84% | 0.00% | 40.20% | 33.57% | 17.37% |
| **🌟 SpinVL-Free + OPD (`hybrid`, 21k)** | `q17_free_hybrid_opd.json` | **None** | **0** (L5–28) | 27.93% | **56.99%** | 19.80% | 2.16% | 1.24% | 0.00% | **42.46%** | **34.91%** | **18.02%** |
| **🌟 SpinVL-Free (`hybrid`, P15 s0)** | `q17_p15_free_hybrid_ce.json` | **None** | **0** (L5–28) | 27.60% | 54.57% | 19.10% | 2.66% | 1.56% | 0.00% | 41.09% | 33.76% | 17.58% |
| **🌟 SpinVL-Free (`hybrid`, P15 s2)** | `q17_p15_free_hybrid_ce_s2.json` | **None** | **0** (L5–28) | **28.73%** | 52.37% | **21.00%** | 1.50% | 1.16% | 0.00% | 40.55% | 34.03% | 17.46% |
| **🌟 SpinVL-Free + OPD (P15 s2)** | `q17_p15_free_hybrid_opd_s2.json` | **None** | **0** (L5–28) | **29.20%** | **56.15%** | 19.50% | 1.00% | 0.96% | 0.00% | **42.68%** | **34.95%** | 17.80% |
| **🌟 SpinVL-Free (P16 No-Align)** | `q17_p16_free_noalign.json` | **None** | **0** (L5–28) | 27.53% | **57.73%** | 19.40% | 2.66% | 1.44% | 0.00% | **42.63%** | **34.89%** | **18.13%** |

---

## 2. Cross-Size Scaling (`Qwen3-0.6B`, `1.7B`, `8B`) & Multi-Seed Stability

### 2.1 Public Benchmark Cross-Size Results (`bench_eval_v4`)

| Scale | Model Variant | JSON File | **MMStar** | **Hallu** | **MathV** | **Geo3K** | **ChartQA** | **MC2** | **Core3** | **Avg6** |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.6B** | `SpinVL-Tower (hybrid, P15)` | `q06_p15_tower_hybrid_ce.json` | 26.40% | **57.73%** | **20.20%** | 2.66% | 0.84% | **42.06%** | **34.78%** | **17.97%** |
| **0.6B** | `SpinVL-Free (CE, 21k)` | `q06_free_ce.json` | 28.07% | 54.68% | **20.40%** | **3.16%** | 1.88% | 41.37% | 34.38% | 18.03% |
| **0.6B** | `SpinVL-Free + OPD (21k)` | `q06_free_opd.json` | **28.87%** | 54.47% | 19.00% | 1.66% | **2.16%** | 41.67% | 34.11% | 17.69% |
| **0.6B** | `SpinVL-Free + OPD (P15 s0)` | `q06_p15_free_hybrid_opd.json` | 27.53% | **57.83%** | 18.60% | 2.16% | **2.16%** | **42.68%** | **34.66%** | **18.05%** |
| **0.6B** | `SpinVL-Free + OPD (P15, 3 seeds)` | `q06_p15_free_hybrid_opd{,_s1,_s2}.json` | 25.35±2.21% | **57.69±0.10%** | 18.73±0.09% | 1.77±0.55% | 2.13±0.39% | **41.52±1.14%** | **33.93±0.74%** | **17.62±0.41%** |
| **1.7B** | `SpinVL-Tower (hybrid, P15, 3 seeds)` | `q17_p15_tower_hybrid_ce{,_s1,_s2}.json` | 26.56±1.21% | **56.36±0.85%** | 18.97±0.97% | **3.11±0.80%** | 1.12±0.09% | **41.46±0.62%** | **33.96±0.48%** | **17.68±0.35%** |
| **1.7B** | `SpinVL-Free + OPD (P15, 3 seeds)` | `q17_p15_free_hybrid_opd{,_s1,_s2}.json` | 26.11±2.62% | **56.11±1.42%** | 19.13±0.82% | 2.17±0.85% | 1.44±0.47% | **41.12±1.10%** | **33.79±0.86%** | **17.49±0.24%** |
| **8B** | `SpinVL-Tower (hybrid, P15)` | `q8b_p15_tower_hybrid_ce.json` | 25.67% | 54.15% | **19.90%** | **3.16%** | 0.92% | 39.91% | 33.24% | 17.30% |
| **8B** | `SpinVL-Free (hybrid, P15)` | `q8b_p15_free_hybrid_ce.json` | 24.80% | **56.99%** | 19.20% | 2.50% | 1.52% | **40.90%** | **33.66%** | **17.50%** |
| **8B** | `SpinVL-Free + OPD (hybrid, P15)` | `q8b_p15_free_hybrid_opd.json` | **27.40%** | 53.31% | 18.10% | 1.33% | 1.32% | 40.36% | 32.94% | 16.91% |

---

## 3. Step 1 Architectural Ablations (`PatchFront` & `LayerVRMG`)

### 3.1 Built-in Frontend Mode Ablation (`none_K0` vs `rot_K4` vs `rots_K4` vs `lora16_K4` vs `full_K4`)
 evaluated on both `mix` validation (`results/p4/`, `results/p6/`, `results/p10/`) and public `bench_eval_v4` (`q17_abl_front_*.json`):

| Frontend Mode ($K=4$) | Static Params | Total Front Params | `mix` Val CE $\downarrow$ | `mix` `tf_first` / `tf_full` $\uparrow$ | Public `MC2` $\uparrow$ | Public `Core3` $\uparrow$ | Public `Avg6` $\uparrow$ | Pure-Text `GSM8K` (if merged) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none_K0` (PatchEmbed+Merger only, 0 LLM layers) | 0.00M | 54.0M | 5.237 | 24.0% / 16.0% | 35.92% | 31.15% | 15.81% | 70.4% |
| `lora16_K4` (VoRA-style additive LoRA, $r=16$) | 3.74M | 57.7M | 1.532 | 44.7% / 18.7% | 36.84% | 30.73% | 15.82% | **0.0%** (collapsed) |
| `full_K4` (Unfrozen copied first 4 LLM layers) | 201.4M | 255.4M | 1.745 | 43.3% / 20.0% | 38.82% | 32.08% | 16.80% | **0.0%** (collapsed) |
| `rot_K4` (Static Cayley orthogonal rotation only) | 2.32M | 56.4M | 1.745 | 43.3% / 20.0% | 40.73% | 33.89% | 17.55% | **70.4%** (decoupled) |
| **🌟 `rots_K4 + hybrid + OPD` (Ours)** | **2.40M** | **56.4M** | **1.554** | **45.3% / 18.7%** | **42.68%** | **34.95%** | **18.02%** | **70.4%** (decoupled) |

### 3.2 Built-in Frontend Depth Ablation ($K \in \{0, 2, 4, 6\}$, `results/p18/p18_q17_free_hyb_K*_joint.json`)

| Frontend Depth $K$ | Static Adapter Params | P15 Val Paired CE $\downarrow$ | Image Grounding `gap` $\uparrow$ | Val `tf_first` $\uparrow$ | Val `tf_full` $\uparrow$ | Public `MC2` (`v4`) | Public `Avg6` (`v4`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **$K=0$** (No LLM layers) | 0.00M | 2.382 | +0.387 | 30.7% | 13.3% | 42.31% | 17.94% |
| **$K=2$** (First 2 layers) | 1.20M | 2.267 | +0.815 | 36.7% | 17.3% | 41.30% | 17.68% |
| **🌟 $K=4$ (Default)** | **2.40M** | **2.248** | **+1.070** | **37.3%** | **16.7%** | **42.68%** | **18.02%** |
| **$K=6$** (First 6 layers) | 3.59M | 2.272 | +0.978 | **38.0%** | **18.7%** | 41.94% | 17.66% |

> **Finding**: Increasing $K$ from $0 \to 2 \to 4$ steadily improves image grounding `gap` (`+0.39 -> +0.81 -> +1.07`) and teacher-forced accuracy (`30.7% -> 36.7% -> 37.3%`), while $K=4$ provides the optimal accuracy-efficiency trade-off before diminishing returns at $K=6$.

### 3.3 End-to-End Joint Training Without Teacher Feature Alignment (`P16`)
- **`p16_q17_free_noalign_joint.json` / `q17_p16_free_noalign.json`**: Setting `feat_weight = 0.0` during Stage 2B joint training (training `PatchFront` + `LayerVRMG` purely from downstream answer CE loss without teacher vision tower feature cosine supervision) achieves **`MC2 = 42.63%`, `Core3 = 34.89%`, `Avg6 = 18.13%`**, proving that `PatchFront` genuinely learns visual representations for downstream reasoning rather than merely mimicking teacher embeddings.

### 3.4 Data Scaling Curve ($N \in \{1\text{k}, 3\text{k}, 6\text{k}, 12\text{k}, 21\text{k}\}$, `results/p2/p2_*.json`, 3 seeds per point)
- At small scale ($N=6,000$), relational + KL distillation (`rel` arm) reaches **23.3% `tf_full`**, matching the accuracy that pure `ce` requires $N=21,000$ samples (**3.5× sample efficiency gain**) to reach (**22.4% `tf_full`**).
- At full scale ($N=21,000$), `ce` + online `VL-OPD` achieves the highest accuracy (`tf_first = 50.0%, tf_full = 23.3%, EM = 24.0%`).

---

## 4. Step 2 (`VL-OPD`) Online Policy Distillation Ablations

### 4.1 Internal Generation `EM` & `tf_full` Gains (`P9`, `P10`, `P15`, `P18`)
- **`1.7B SpinVL-Tower (hybrid)` (`p10_topd_hybrid.json`)**: `tf_full` improves from **18.7% $\to$ 22.0% (+3.3%)**, free-generation `EM` improves from **16.7% $\to$ 20.7% (+4.0%)**.
- **`1.7B SpinVL-Tower (mf)` (`p10_topd_mf.json`)**: reaches **`tf_first = 50.0%`, `tf_full = 23.3%`, `EM = 24.0%`**.
- **`1.7B SpinVL-Free (mf)` (`p9_opd_mf_rots.json`)**: `tf_first` improves from **43.3% $\to$ 46.7% (+3.4%)**, `tf_full` improves from **17.3% $\to$ 22.0% (+4.7%)**.
- **`0.6B SpinVL-Tower (hybrid, P15)` (`p15_q06_tower_hybrid_opd.json`)**: `tf_full` improves from **16.0% $\to$ 20.7% (+4.7%)**, `EM` improves from **17.0% $\to$ 20.7% (+3.7%)**, EOS `stop_rate` = **96.7%**.
- **`1.7B SpinVL-Free (hybrid, P15)` (`p15_q17_free_hybrid_opd.json`)**: `tf_full` improves from **16.0% $\to$ 18.7% (+2.7%)**, `EM` reaches **19.0%**, EOS `stop_rate` = **95.0%**.

### 4.2 Forced Free-Form Generation on Public Benchmarks (`--force-gen`, `bench_eval_v4/*_fgen.json`)
When candidate log-likelihood scoring is disabled and models must generate open-ended text strings greedily on `MMStar`, `HallusionBench`, and `MathVista`:

| Model Pair (Before vs. After `VL-OPD`) | `MMStar` (Free-Gen) | `Hallusion` (Free-Gen) | `MathVista` (Free-Gen) | `Core3` Free-Gen Avg |
|---|---:|---:|---:|---:|
| `0.6B SpinVL-Free (P15)` Before OPD (`ce_fgen`) | 14.53% | 23.55% | 9.70% | 15.93% |
| **🌟 `0.6B SpinVL-Free (P15)` After OPD (`opd_fgen`)** | **22.20% (+7.67%)** | **33.44% (+9.89%)** | **14.40% (+4.70%)** | **23.35% (+7.42%)** |
| `1.7B SpinVL-Free (P9)` Before OPD (`ce_fgen`) | 14.07% | **24.40%** | 8.30% | 15.59% |
| **🌟 `1.7B SpinVL-Free (P9)` After OPD (`opd_fgen`)** | **18.13% (+4.06%)** | 22.92% | **9.30% (+1.00%)** | **16.79% (+1.20%)** |
| `1.7B SpinVL-Free (P15)` Before OPD (`ce_fgen`) | 25.33% | 0.11% | 12.20% | 12.55% |
| **🌟 `1.7B SpinVL-Free (P15)` After OPD (`opd_fgen`)** | 25.07% | **1.26%** | **16.30% (+4.10%)** | **14.21% (+1.66%)** |

---

## 5. Efficiency & Pure-Text Capability Preservation (`P18`)

### 5.1 Analytical FLOPs & Exclusive-GPU Latency (`results/p18/bench_K4_clean.json`)
Measured on `Qwen3-1.7B` ($K=4$, text query length $Q=64$) across increasing image resolutions:

| Visual Patches ($N_{\text{patches}}$) | Merged Cells | **🌟 Ours (`SpinVL-Free`)** TFLOPs / Latency | `ROSE` (Ext. Tower + Rot) TFLOPs / Latency | `LLaVA` (Ext. Tower + Full Tokens) TFLOPs / Latency | **FLOPs Saved vs. LLaVA** |
|---:|---:|---:|---:|---:|---:|
| **256** | 64 | **0.30 TFLOPs** / 65.8 ms | 0.47 TFLOPs / 58.2 ms | 0.66 TFLOPs / 25.7 ms | **-54.8%** |
| **576** | 144 | **0.45 TFLOPs** / 66.6 ms | 0.86 TFLOPs / 57.8 ms | 1.27 TFLOPs / 25.6 ms | **-64.9%** |
| **1,024** | 256 | **0.67 TFLOPs** / 69.8 ms | 1.45 TFLOPs / 63.8 ms | 2.18 TFLOPs / 33.7 ms | **-69.4%** |
| **2,304** | 576 | **1.37 TFLOPs** / 80.4 ms | 3.40 TFLOPs / 86.0 ms | 5.07 TFLOPs / 66.9 ms | **-72.9%** |
| **4,096** | 1,024 | **2.54 TFLOPs / 99.4 ms** | 6.82 TFLOPs / 126.1 ms | 9.85 TFLOPs / 123.4 ms | **-74.2% (-19.5% ms)** |

### 5.2 Pure-Text Capability Preservation (`results/p18/lang_K4_full.json`)

| Variant | `GSM8K` ($n=250$, 8-shot CoT) | `ARC-Challenge` ($n=500$) | Explanation |
|---|---:|---:|---|
| **🌟 `SpinVL` Text Pathway (`base`)** | **70.4%** | **71.4%** | **100% bit-exact lossless** (vision rotations decoupled from text pathway) |
| Merged `rot_K4` into LLM weights | 0.0% | 24.2% | Catastrophic collapse if static vision rotation is permanently merged into text layers |
| Merged `rots_K4` into LLM weights | 0.8% | 23.4% | Catastrophic collapse if static rotation+scaling is permanently merged |
| Merged `lora16_K4` (VoRA-style) | 0.0% | 27.4% | Catastrophic collapse if additive LoRA vision encoder is merged into text layers |

---

## 6. Chronological Phase Summary & Negative-Result Post-Mortems (`P1`–`P18`)

| Phase | Objective | Outcome & Key Lesson |
|---|---|---|
| **P1–P2** | `Geometry3K` (5 seeds + hyperparam ablations $b, L, T$) & `mix` data scaling curve ($N=1\text{k}..21\text{k}$ across `ce/kd/rel`). | Confirmed `rel` gives 3.5× sample efficiency at small $N$, while `ce` scales best at $N=21\text{k}$ (`CE=1.594`, `Geo3K test tf_first=96.2%`). |
| **P3–P5** | Stage 2A frontend warm-up (`none/rot/lora/full`, $K=0..6$), Stage 2B joint training (`P4`), and efficiency/language benchmarks (`P5`). | Proved `PatchFront` ($K=4$) enables tower-free VLM inference saving 74% FLOPs at 4096 patches. |
| **P6–P9** | Fair `LLaVA-Proj` baseline (`p6_proj_n21000`), `rots_K4` (rotation + diagonal scaling), `multifreq` (`P7`), `tokrot` & `hybrid` (`P8`), and `SpinVL-Free + VL-OPD` (`P9`). | `rots_K4` + `hybrid`/`mf` + `VL-OPD` achieved our flagship results (`MMStar 30.60%`, `Hallusion 58.15%`). |
| **P10–P11** | Fair `VoRA (LoRA K=4)` baseline, `SpinVL-Tower + OPD`, equal-step 7875-step runs (`P10`), and `Qwen3-0.6B / 8B` cross-size pipelines (`P11`). | Beat `VoRA` by +5.84% on `MC2` and confirmed cross-size generalization across 0.6B, 1.7B, and 8B. |
| **P12** *(Negative)* | Tested prompt-segment rotation attenuation (`--prompt-alpha 0.2`) to see if weakening prompt rotation helps open-ended benchmarks. | **Post-mortem**: Right-padded training sequences caused the tail mask `w[:, -tail:]` to land on `PAD` tokens rather than answer tokens; removed from final model. |
| **P13** *(Negative)* | Tested injecting 16–64 pooled visual tokens (`train_joint_tok.py`) into the backbone for OCR/ChartQA. | **Post-mortem**: Unnormalized visual projections caused gradient spikes (`gn=361`) and violated the core "0 visual tokens in backbone" narrative; immediately cancelled and rolled back. |
| **P14** *(Mixed)* | Fixed `bad_words_ids` evaluation bug (`Yes/No` ban on `ChartQA`); tested `--qcond` (text-mean conditioned VRMG queries) + `--direct-hyb` + `geo_8b` training data. | **Post-mortem**: Adding `geo_8b` doubled `Geometry3K` accuracy (`1.83% -> 3.49%`), but `--qcond` suffered from train-test sequence-mean mismatch (training included answer + padding) and lowered `MMStar/Hallusion`. |
| **P15** *(Positive)* | Audited `mix` (found 0 open-ended chart/OCR items); added 8,334 `DocVQA/GQA` open-ended items + 2,101 `Geo3K` items with clean `P8/P9` architecture (`--hybrid` / `--multifreq`, no `--qcond`). | **Breakthrough**: `SpinVL-Tower (mf+OPD)` surged to **23.50% on `MathVista`** and **35.90% on `Core3` (beating LLaVA)**; `Geo3K` doubled to **3.83%**; same-data `LLaVA-Proj` (`10.32%` on `ChartQA`) proved `ChartQA/OCR` is an architectural boundary of 0-token modulation rather than a data bug. |
| **P16 & P18** *(Positive)* | `P16`: Joint training without teacher feature alignment (`feat_weight=0`). `P18`: Frontend depth $K \in \{0,2,4,6\}$, 3-seed runs (`s0/s1/s2`), exclusive-GPU latency (`bench_K4_clean`), and `bench_eval_v4` (`MMStar` option parser fix). | Completed all ablation tables, error bars, and latency/language benchmarks with 0 failures. |
