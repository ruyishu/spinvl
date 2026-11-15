# SpinVL 论文终稿核对与修改记录（verify.md）

> 更新时间：2026-11-15  
> Overleaf 工程：`https://www.overleaf.com/project/6aeeb5d00b27321f7314e61c`  
> 本地工作库：`D:\Users\kun.z\Documents\paper\spinvl`

---

## 一、已定全局策略与硬约束

### 1. CVPR 页数红线（已达标）
- **CVPR 规则**：正文（`References` 之前的所有章节与图表）**严格 $\le 8.0$ 页**，`References` 页数不限，`X_suppl.tex` 作为补充材料单独提交。
- **Overleaf 最新编译实测（13 页合编 PDF，`spinvl (14).pdf`）**：
  - `Abstract + 1. Intro + 2. Related`：Pages 1–3 上（已合并 `1. Intro` 原孤立的 `a unified question:` 断行与短问句段，修正 `2. Related` 尾行并让 `3. Method / 3.1. Overview` 在第 3 页左下顺畅起段）
  - `3. Method`：Pages 3 上–6 上（含合作者拆分扩展的 `Eq. (1)–(17)` 与新版 `Fig. 2`；已将 `Inference and complexity` 迁至附录 `Sec. S1`）
  - `4. Experiments`：Pages 6 上–8 中（保留 `Tab. 1–3` 与新增跨双栏主图 **`Fig. 3`**）
  - `5. Conclusion` + `LLM Usage.`：Page 8 中下（正文与 `LLM Usage` 均在 **Page 8 结束**，严格 $\le 8.0$ 页！）
  - `References`：Page 8 底–Page 10（`[1]–[45]` 全部无误）
  - `Supplementary Material`：Pages 11–13（`S1–S4`，`Tab. S1–S9`，`Fig. S1–S3`；第 11 页 `Tab. S2` 下方留白与第 12 页跨页孤行已消除，`Tab. S4` 已去方差下标，`Fig. S3` 移至第 13 页左栏与 `Tab. S9` 左右对称收尾）
  - **交稿前拆页提示**：下载编译 PDF 后，将 Pages 1–10 拆为 `main.pdf` 提交主稿，Pages 11–13 拆为 `supp.pdf` 提交补充材料（或注释 `main.tex` 第 47 行 `\input{sec/X_suppl}` 后分别导出）。

### 2. 效率口径策略：方案 A（已落地）
- **摘要、`1. Intro`、`Fig. 3(a,b)`、`Tab. S1`、`5. Conclusion`**：保留 `74.2% fewer FLOPs`（`2.54 vs. 9.85 TFLOPs`）和 `19.5%` 延迟提速（`99.4 vs. 123.4 ms`），`Tab. 2` 的 `9.9%` 保留不动。
- **`sec/4_experiments.tex` 第 4.6 节 & `X_suppl.tex` Sec. S1**：已移除 `further lowers` 自相矛盾措辞，将 `20.0%` 修正为 **`19.9%`**，准确说明在 `clean_v6` 完整 `qcond + direct_hyb` 配置下跳过 `o/down` 旋转将延迟从 `108.9 ms` 降至 `101.9 ms`（vs. 同口径 `LLaVA-Proj` `127.2 ms`），对应 **`2.67 TFLOPs`（`-72.5%`** vs. `9.70 TFLOPs`）。

---

## 二、逐章节修改验收清单

### `sec/0_abstract.tex`、`sec/1_intro.tex`、`sec/2_related.tex`
- [x] **`0_abstract.tex`**：保留精修版与方案 A 口径（`74.2%` 与 `9.9%`）。
- [x] **`1_intro.tex`**：SAVM 描述已同步为 *"bounded, image- and question-conditioned multi-frequency rotations with spatial retrieval"*；全章 `\cite{}` 统一为 `\citep{}`，并优化段落尾行宽度。
- [x] **`2_related.tex`**：核对 `VoRA`、`VLoRA`、`LLaVA-KD`、`MoVE-KD`、`ROSE`、`EVE` 原始文献无误，优化段落尾行。

### `sec/3_method.tex` & `figures/spinvl_method.pdf`
- [x] **新版 `Fig. 2`（`spinvl_method.pdf`）核验通过**：已补齐 `Question-conditioned multi-frequency context`、有界旋转 $\tau_{\mathrm{rot}}\tanh$、`Spatial Retrieval (K_\ell^a / U_\ell^a)` 与门控公式 $y_{\ell,p}=(\tilde{x}+\Delta_\ell(x))W_{\ell,p}^\top+\beta_{\ell,p}$。
- [x] **保留合作者 `Eq. (1)–(17)` 拆分版**：公式推导清晰无越界。
- [x] **将 `Inference and complexity` 迁入附录 `Sec. S1`**：在 `3.4` 末尾保留精简 `Modality decoupling.`（2–3 行）指向 `Sec. S1`，使 `Method` 更纯粹且释放正文篇幅。

### `sec/4_experiments.tex` & `figures/fig_exp_highlights.pdf`
- [x] **`Tab. 1` 瘦身**：移除 `44k Blank-Image Control`、`21k hyb` 重复行及与 `Tab. 2` 重复的 `GKD/MiniLLM/SeqKD` 三行，主表精炼聚焦为 21 行三段主对比。
- [x] **新增正文跨双栏主图 `Fig. 3`（`fig_exp_highlights.pdf`）**：沿用论文一致配色与字体，展示 `(a) Prefill Compute Scaling`、`(b) End-to-End Latency Crossover` 与 `(c) VL-OPD Free-Generation Gains`，替换纯数字堆砌的原效率表。
- [x] **将效率明细表与 `Frontend parameterisation` 表迁至附录（`Tab. S1` 与 `Tab. S3`）**：配合精简 `4.1–4.6` 行文并优化短尾行，使正文保留核心 `Tab. 1–3` 与 `Fig. 3` 且整体严控在 8 页内。
- [x] **口径与引用核验**：`VoRA-style LoRA`、`Spatial-Pool`、`Qwen3-VL-8B-Instruct` `[4]`、`ce` vs. `rel` 定义及 `44k` 审计全部对齐。

### `sec/5_conclusion.tex`
- [x] **删除 `Limitations` 段落**，章标题改为 `5. Conclusion`，在正文第 8 页末尾新增 **`LLM Usage.`** 声明并控制整页严格结束于 Page 8。

### `main.bib`（本地同步至 `paper/references.bib`）
- [x] **修复 `[30]` OCRBench 错标题**、保护专有名词大括号并修复作者重音。

### `sec/X_suppl.tex` & `figures/fig_data_scaling.pdf`
- [x] **完整吸收主文迁入内容**：`Sec. S1` 整合 `Inference and complexity` 推导与原效率表 `Tab. S1`；`Sec. S2` 保留 `Tab. S2` 与 `ce` vs. `rel` 定义并移除重复 `\citep` 消除主文 `[23]/[35]` 的 `, 1` 回指污染；`Sec. S3` 收录 `Tab. S3`（Frontend 参数化）与 `Tab. S4–S8`，`Fig. S3` 更新为独立单栏 `fig_data_scaling.pdf` 以避免与正文 `Fig. 3` 重复；`Sec. S4` 保持 `Tab. S9` 与正文紧密贴合（Page 13 右栏无孤立大白块）并移除失效的 `Limitation (1)` 回指。
- [x] **编译与校验**：Overleaf 编译 0 错误、0 警告、全篇 0 处 `??`；[verify_paper_numbers.py](D:/Users/kun.z/Documents/paper/spinvl/scripts/verify_paper_numbers.py) `515` 项校验通过，`Tab. S7–S8` 150 项校验通过。

### 投稿前操作提示（非阻塞）
- [ ] 投稿前拆出 Pages 1–10 为 `main.pdf`，Pages 11–13 为 `supp.pdf`（或在 Overleaf 注释 `main.tex` 第 47 行 `\input{sec/X_suppl}` 后分别下载）。
