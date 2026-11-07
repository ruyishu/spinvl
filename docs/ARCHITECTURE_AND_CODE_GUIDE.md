# SpinVL Architecture, Mathematical Formulation & Code Guide

This document provides a technical walkthrough of the `spinvl` codebase (`src/`, `scripts/`, `queue/`) for collaborators.

---

## 1. Core Modules (`src/`)

### 1.1 `src/vrmg.py` — Dynamic Orthogonal Rotation Generator (`LayerVRMG`) & `RotatedLinear`

#### Mathematical Formulation
For a frozen linear layer $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with input $x \in \mathbb{R}^{B \times T \times d_{\text{in}}}$, we partition the input channels into $N_b = d_{\text{in}} / b$ blocks of size $b = 64$.
`LayerVRMG` generates $N_b$ skew-symmetric matrices $P_r = -P_r^\top \in \mathbb{R}^{b \times b}$ ($r = 1, \dots, N_b$), each parameterized by $b(b-1)/2 = 2016$ lower-triangular entries, and maps them to the special orthogonal group $SO(b)$ via the **Cayley transform**:
$$R_r = (I - P_r)^{-1}(I + P_r) \in SO(b), \quad R_r^\top R_r = I$$

The rotated linear forward pass (`RotatedLinear.forward`) computes:
1. **Block-diagonal orthogonal rotation**:
   $$x_{\text{rot}}^{(r)} = x^{(r)} R_r, \quad r = 1, \dots, N_b$$
2. **Optional Multi-Frequency Channel Gating (`--multifreq`)**:
   - Queries in `LayerVRMG` are split into 3 frequency bands:
     - **Low-freq (25%)**: Attends to global mean-pooled visual feature $\bar{v}$.
     - **Mid-freq (37.5%)**: Attends to all visual cells with smooth 2D sinusoidal positional encoding ($\text{base}=10000, \tau=1.0$).
     - **High-freq (37.5%)**: Attends to all visual cells with high-frequency positional encoding ($\text{base}=50, \tau=0.25$).
   - A zero-initialized token-wise gate $g(x) = 2\sigma(W_g x) \in (0, 2)^{N_b}$ interpolates between identity and rotation:
     $$\tilde{x}^{(r)} = x^{(r)} + g_r(x)\big(x_{\text{rot}}^{(r)} - x^{(r)}\big)$$
3. **Optional Spatial-Anchor Hybrid Residual (`--hybrid`)**:
   - Pools $M = 16$ spatial anchors $A \in \mathbb{R}^{B \times 16 \times d_q}$ ($d_q = 256$) from the visual cells (with 2D PE, $\text{base}=500$).
   - Projects text hidden states $x$ into the 256-dim subspace ($q_t = W_q \operatorname{LN}(x)$), cross-attends to the 16 anchors, and projects back via a zero-initialized linear layer $W_{\text{out}}$:
     $$x_{\text{in}} = \operatorname{concat}(\tilde{x}^{(1)}, \dots, \tilde{x}^{(N_b)}) + W_{\text{out}}\operatorname{Attn}(q_t, A, A)$$
   - Notice that **0 visual tokens enter the LLM's self-attention or KV cache**.

---

### 1.2 `src/front.py` — Built-in Vision Frontend (`PatchFront`)

Used in **`SpinVL-Free`** to eliminate the external vision tower:
1. **Input**: Raw image patches extracted by the Qwen3-VL image processor (`patch_dim = 1536`, ordered in $2\times 2$ merge windows) plus 2D row/column embeddings (`row_emb`, `col_emb`).
2. **Shallow $K$-Layer Encoding ($K=4$ default)**:
   - Runs through the student LLM's own first $K$ layers (`model.model.layers[0..K-1]`) using **bidirectional self-attention** (all valid patches attend to each other).
   - Supports 4 adaptation modes via `wrap.set_static(self.static)`:
     - `rots` (`StaticRotScale`, **best**): $W' = W R_{\text{static}} \operatorname{diag}(d)$, combining static Cayley orthogonal rotation with channel-wise diagonal scaling ($d$ initialized to $\mathbf{1}$).
     - `rot` (`StaticRot`): Pure static Cayley orthogonal rotation $W' = W R_{\text{static}}$.
     - `lora` (`StaticLoRA`): VoRA-style rank-$r$ additive LoRA adapter.
     - `full`: Unfrozen deep copy of the first $K$ layers.
3. **2×2 Spatial Merger**:
   - Reshapes every 4 adjacent patch hidden states $(B, 4 C, d_{\text{model}}) \to (B, C, 4 d_{\text{model}})$ and applies `LayerNorm -> Linear -> GELU -> Linear` to produce $\hat{v} \in \mathbb{R}^{B \times C \times d_v}$.
   - **All image patch tokens are discarded immediately after `PatchFront.forward` returns $\hat{v}$.**

---

### 1.3 `src/wrap.py` — Non-Invasive LLM Hooking (`RotatedProj`, `attach_rotation`)

- Wraps `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` in selected LLM layers with `RotatedProj`.
- Uses two thread/context-local switches:
  1. `_STATIC`: Active only inside `PatchFront.forward` when encoding image patches through the first $K$ layers.
  2. `_CTX` (`with rotation_context_with(registry, visual, visual_mask):`): Active during text forward/generation; lazily computes `LayerVRMG(visual, visual_mask)` **once per layer** and caches the rotation matrices in `_CTX["cache"][layer_id]` across all 7 projections of that layer (and across autoregressive decoding steps).
- When both `_STATIC` and `_CTX` are `None` (pure-text inference), `RotatedProj.forward(x)` directly executes `self.rot.base(x)`, guaranteeing **bit-exact identity** with the original LLM.

---

## 2. Training & Evaluation Pipeline (`scripts/`)

| Script | Role | Key Arguments & Checkpoint Schema |
|---|---|---|
| `scripts/train_final.py` | **Stage 1 (`SpinVL-Tower`)**: Train `LayerVRMG` with frozen teacher vision features. | `--arm {ce,kd,rel}`, `--multifreq`, `--hybrid`, `--layers`, `--block 64`. Saves `*_pt.pt` with `{"state": [m.state_dict()...], "args": vars(args)}`. |
| `scripts/train_front.py` | **Stage 2A**: Warm up `PatchFront` by aligning $\hat{v}$ to teacher vision tower features (`feature_loss`: cosine + normalized smooth-$L_1$). | `--mode {rot,rots,lora,full,none}`, `--K 4`. Saves `*.pt` with `{"front": front.state_dict(), "args": vars(args)}`. |
| `scripts/train_joint.py` | **Stage 2B (`SpinVL-Free`)**: End-to-end joint training of `PatchFront` + `LayerVRMG` from raw images without any external vision tower at inference. | `--stage {oracle,front_frozen,joint}`, `--vrmg-ckpt`, `--front-ckpt`, `--feat-weight` (`0.0` for `P16` no-align). Saves `*.pt` with `{"front": ..., "vrmg": [...], "args": ...}`. |
| `scripts/train_opd.py` | **Step 2 (`VL-OPD` for Tower)**: On-policy student rollout + teacher token-level KL + gold CE anchoring. | `--vrmg-ckpt`, `--steps 500`, `--max-new 16`, `--lr 2e-5`, `--ce-weight 0.5`. Saves `*.pt` in Tower format (`{"state": [...], "args": a1}`). |
| `scripts/train_joint_opd.py` | **Step 2 (`VL-OPD` for Free)**: Joint on-policy distillation updating both `PatchFront` and `LayerVRMG`. | `--joint-ckpt`, `--steps 500`, `--lr-vrmg 2e-5`, `--lr-front 1e-5`, `--ce-weight 0.5`, `--feat-weight 0.2`. Saves `*.pt` in Joint format. |
| `scripts/train_proj.py` | **Baseline (`LLaVA-Proj`)**: Standard 2-layer MLP projector prepending all teacher visual tokens into all 28 LLM layers. | `--data`, `--source`, `--steps`, `--lr 5e-4`. Saves `*.pt` with `{"proj": proj.state_dict(), "args": ...}`. |
| `scripts/eval_benchmarks.py` | **Unified 6-Benchmark Evaluator**: Zero-shot evaluation on `MMStar`, `HallusionBench`, `ChartQA`, `OCRBench`, `MathVista`, `Geometry3K`. | `--mode {tower,tower_free,proj}`, `--ckpt`, `--force-gen` (ablation for free generation on MC), `--image-ctrl {none,shuffle,blank}`. |

---

## 3. Key Engineering Lessons & Bug Fixes (For Collaborators)

1. **`LayerVRMG.apply_hybrid` Cache Key Under `torch.no_grad()` (`src/vrmg.py`)**:
   - *Pitfall*: Caching `dx` by `x_raw.data_ptr()` caused false cache hits during `torch.no_grad()` evaluation because PyTorch's caching allocator immediately reuses memory addresses of freed intermediate tensors across layers.
   - *Fix*: Key `hcache` by `id(x_raw)` and store the tuple `(x_raw, dx)` inside `hcache` so `x_raw` stays alive for the duration of the forward pass (`if hit is None or hit[0] is not x_raw:`).
2. **Open-Ended Generation First-Step Option-Letter Bias (`scripts/bench_models.py`)**:
   - *Pitfall*: Because 70% of `mix` training samples are `A/B/C/D` or `Yes/No` questions, models occasionally emit a bare option letter on open-ended numeric questions; conversely, banning `Yes/No` or `A..D` globally via `bad_words_ids` broke `ChartQA` `Yes/No` items and `OCRBench` word spelling.
   - *Fix*: `FirstStepBanLetterProcessor` masks standalone `' A'..' E'` **only at decoding step 0** on `chartqa/mathvista/geometry3k` open-ended items, leaving `Yes/No` and all steps $\ge 1$ completely unmasked.
3. **`MMStar` Multi-Format Option Parser (`scripts/bench_data.py`)**:
   - *Pitfall*: `MMStar.tsv` contains both inline `, B: ` options and newline `A. ... \nB. ...` options.
   - *Fix*: Updated `_parse_opts` / `load_mmstar` in `bench_eval_v4`, recovering the missing multi-choice options and raising `MMStar` scores by +6%–8% uniformly across all baselines and our models.
