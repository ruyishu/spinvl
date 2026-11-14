# SpinVL 论文终稿核对与修改记录（verify.md）

> 更新时间：2026-11-15  
> Overleaf 工程：`https://www.overleaf.com/project/6aeeb5d00b27321f7314e61c`  
> 本地工作库：`D:\Users\kun.z\Documents\paper\spinvl`

---

## 一、已定全局策略与硬约束

### 1. CVPR 页数红线（已达标）
- **CVPR 规则**：正文（`References` 之前的所有章节与图表）**严格 $\le 8.0$ 页**，`References` 页数不限，`X_suppl.tex` 作为补充材料单独提交。
- **Overleaf 最新编译实测（13 页合编 PDF，`spinvl (2).pdf` / `paper/main.pdf`）**：
  - `Abstract + 1. Intro + 2. Related`：Pages 1–2（`~2.0` 页）
  - `3. Method`：Pages 3–5 上（`~2.3` 页）
  - `4. Experiments`：Pages 5 中–8 上（`~3.0` 页，保留 `Tab. 1–5`）
  - `5. Limitations and Conclusion`：Page 8 中下（`~0.4` 页，正文在 **Page 8 结束**，严格 $\le 8.0$ 页！）
  - `References`：Page 8 底–Page 10（`[1]–[45]` 全部无误）
  - `Supplementary Material`：Pages 11–13（`S1–S4`，`Tab. S1–S7`，`Fig. S1–S3`）
  - **交稿前拆页提示**：下载编译 PDF 后，将 Pages 1–10 拆为 `main.pdf` 提交主稿，Pages 11–13 拆为 `supp.pdf` 提交补充材料（或注释 `main.tex` 第 47 行 `\input{sec/X_suppl}` 后分别导出）。

### 2. 效率口径策略：方案 A（已落地）
- **摘要、`1. Intro`、`Tab. 4`（原 `Tab. 6`）、`Fig. S3(a,b)`、`5. Conclusion`**：保留 `74.2% fewer FLOPs`（`2.54 vs. 9.85 TFLOPs`）和 `19.5%` 延迟提速（`99.4 vs. 123.4 ms`），`Tab. 3` 的 `9.9%` 保留不动。
- **`sec/4_experiments.tex` 第 4.6 节 & `X_suppl.tex` Sec. S1**：已移除 `further lowers` 自相矛盾措辞，将 `20.0%` 修正为 **`19.9%`**，准确说明在 `clean_v6` 完整 `qcond + direct_hyb` 配置下跳过 `o/down` 旋转将延迟从 `108.9 ms` 降至 `101.9 ms`（vs. 同口径 `LLaVA-Proj` `127.2 ms`），对应 **`2.67 TFLOPs`（`-72.5%`** vs. `9.70 TFLOPs`）。

---

## 二、逐章节修改验收清单

### `sec/0_abstract.tex`、`sec/1_intro.tex`、`sec/2_related.tex`
- [x] **`0_abstract.tex`**：保留精修版与方案 A 口径（`74.2%` 与 `9.9%`）。
- [x] **`1_intro.tex`**：SAVM 描述已同步为 *"bounded, image- and question-conditioned multi-frequency rotations with spatial retrieval"*；全章 `\cite{}` 已统一为 `\citep{}`。
- [x] **`2_related.tex`**：核对 `VoRA`、`VLoRA`、`LLaVA-KD`、`MoVE-KD`、`ROSE`、`EVE` 原始文献无误，保持不动。

### `sec/3_method.tex`
- [x] **补齐李代数切空间有界旋转 & 仅旋 5 投影**：明确给出 $\tau_{\mathrm{rot}}=0.12$ 的 $\tanh$ 切空间缩放公式（`Eq. (6)` 两行对齐防越界），说明跳过 `o`/`down`（仅旋 `{q,k,v,gate,up}`），修正门控公式与偏置初始化为：无界 $2\sigma(\cdot)$ 偏置 `0`（$g=1$）、有界 $\sigma(\cdot)$ 偏置 `-2.0`（$g\approx 0.12$，严格对齐 `src/vrmg.py`）。
- [x] **补齐 `qcond` 与 `direct_hyb` 机制**：在 SAVM 补入提示条件化 `qcond`（按 *image--question prefill* 缓存复用）与 `direct_hyb` / `hyb` 兼容说明。
- [x] **补齐 `Modality Decoupling` 与 `Complexity`**：给出 `logits_diff = 0` 与 $O(KN^2+LT^2)$ vs. $O(L(T+N)^2)$，整章精简压至 `~2.3` 页。

### `sec/4_experiments.tex`
- [x] **修正 `4.2 (iii)` `40.21%` $\to$ `40.20%`**（与 `Tab. 1` 和 `Tab. 3` 完全对齐）。
- [x] **修正 `4.1` `Qwen3-VL-8B-Instruct` 错引 `[37]` $\to$ `[4]`（`bai2025qwen3vl`）**。
- [x] **微调 `Token-Pool` 命名 $\to$ `Spatial-Pool`**，正文注明为受~\citep{cha2024honeybee} 启发的 32-token 空间平均池化基线。
- [x] **澄清 `4.1` 评测协议 & 补充 `44k` 步数**：说明选择题使用候选串似然打分、开放题使用无首词禁词的统一贪心解码、并附 `--force-gen` 审计表；补充 `44k` 训练步数。
- [x] **修正 `4.5` 符号与消融限定**：`W\mathrm{diag}(d)R_{\mathrm{s}}^{\top}` $\to$ `W\mathrm{diag}(s)R_{\mathrm{s}}^{\top}`，注明 `GSM8K=0.0%/0.8%` 为无纯文本回放受控消融下永久合并前 $K$ 层的结果。
- [x] **修正 `4.6` `R3` 3-seed 均值对比、`M0/M1/M2` 分工、`--force-gen` 区间端点（`40.89%--41.33%` 严格对齐 `Tab. S6`）、`direct_hyb` 图表能力桥接及 `19.9%` 效率句**；并在 `4.3` 点明冻结接口带宽使 `0.6B`（`0.11x` FLOPs）成为高性价比部署点。
- [x] **AC 瘦身迁出**：原 `Tab. 2`、`Tab. 4`、`Tab. 7` 与 `Fig. 3–5` 已迁至 `X_suppl.tex`（`Tab. S2–S4`，`Fig. S1–S3`），正文精简至 `~3.0` 页。

### `sec/5_conclusion.tex`
- [x] **修正英文冠词语法**：`a $8$B student` $\to$ **`an $8$B student`**。
- [x] **压缩篇幅至 `~0.4` 页**：保留全部 4 条诚实 Limitations，删除 `Conclusion.` 中对 `Tab. 1` 的重复背数字，并将措辞对齐 *"bounded, question-conditioned multi-frequency rotations and spatial retrieval"* 与方案 A。

### `main.bib`（本地同步至 `paper/references.bib`）
- [x] **修复 `[30]` OCRBench 错标题**：已更正为 `{OCRBench}: On the Hidden Mystery of {OCR} in Large Multimodal Models`（`arXiv:2305.07895`）。
- [x] **专有名词大括号保护与作者重音修复**：已保护 `{LoRA}`、`{OCR}`、`{Inter-GPS}`、`{ChartQA}`、`{DocVQA}`、`{VQA}`、`Try {ARC}, the {AI2}`，并修复 Flamingo 作者 `{Bi{\'n}kowski}` 与 `Karen Simonyan`。

### `sec/X_suppl.tex` & `main.tex`
- [x] **重构 `X_suppl.tex`（S1–S4）与附图重绘**：包含 S1 架构复杂度与无损旁路细节、S2 数据去漏审计与超参精确对照表（`Tab. S1`，区分 `44k` 与 `21k/28k` 损失项）、S3 重绘并去重遮挡/去除 `v5` 内部标签的 `Fig. S1–S3`、迁出的 `Tab. S2–S4`（修正 `Tab. S4` `5.49` 加粗）、`44k` 多 seed 汇总 `Tab. S5`（含 `R0` vs `R1–R3` / `S2` vs `F2` 积极定位说明）与 `--force-gen` 全量表（`Tab. S6`），以及 S4 细分能力与 OCR 错误边界诊断表（`Tab. S7`）。
- [x] **编译与校验**：Overleaf 编译 0 错误、0 警告、全篇 0 处 `??`，第 4 页 `Eq. (6)` 与第 8 页收尾排版已逐页渲染核验；`python scripts/verify_paper_numbers.py` 545 项校验通过，`Tab. S5–S6` 150 项校验通过。

### 待人工确认小项（非阻塞）
- [ ] 请合作者按对话提供的精简修改清单更新 `figures/spinvl_method.pdf`（补充 `qcond`、`direct_hyb`、$\tau_{\mathrm{rot}}\tanh$ 与门控符号 $\tilde{x}$）。
- [ ] 投稿前拆出 Pages 1–10 为 `main.pdf`，Pages 11–13 为 `supp.pdf`。
