# SpinVL — CVPR submission draft

LaTeX source for the paper. Two entry points share the same body (`sections/`):

| File | Use |
|---|---|
| `main_cvpr_official.tex` | **Submission build.** Requires `cvpr.sty`, `cvpr_eso.sty`, `eso-pic.sty` from the official CVPR author kit (drop them in this folder). Uses `\usepackage[pagenumbers]{cvpr}` for the anonymised, page-numbered review copy. |
| `main.tex` | **Fallback build.** No kit required; reproduces the CVPR two-column geometry (10 pt Times, 6.875 × 8.875 in text block) with plain packages. Useful for local preview and for collaborators without the kit. |

```bash
# official build
pdflatex main_cvpr_official && bibtex main_cvpr_official && pdflatex main_cvpr_official && pdflatex main_cvpr_official

# fallback build (works with any TeX Live / MiKTeX)
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Structure

| File | Content |
|---|---|
| `sections/0_abstract.tex` | Abstract (official build) |
| `sections/0_abstract_wide.tex` | Abstract, full-width variant (fallback build) |
| `sections/1_intro.tex` | Motivation, problem statement, contributions, Fig. 1 |
| `sections/2_related.tex` | Token concatenation, PEFT, rotation injection, distillation |
| `sections/3_method.tex` | Dual-tier rotation (Eqs. 1–7), modality decoupling, VL-OPD, complexity |
| `sections/4_experiments.tex` | Setup, main table (Tab. 1), scaling (Tab. 2), ablations (Tabs. 3–5), efficiency (Tab. 6), boundary study (Tab. 7) |
| `sections/5_conclusion.tex` | Limitations and conclusion |
| `references.bib` | Bibliography |
| `figures/` | Architecture, main results, efficiency, ablations (mirrored from `../assets/`) |

## Numbers

Every number in the tables is taken from `../results/paper_results.json` and
`../results/benchmarks/*.json`; the mapping is:

- Tab. 1, Tab. 2 → `results/benchmarks/*.json`
- Tab. 3 → `results/benchmarks/q17_abl_front_*.json` (frontend cells)
- Tab. 6 (compute/latency) and Fig. 1 right (text retention) → `results/paper_results.json`
- Internal validation columns (Val CE, 1st/Full, gap) → same files, `val` blocks

Aggregates are defined in Sec. 4.1: **MC2** = (MMStar + HallusionBench)/2,
**Core3** = (MMStar + HallusionBench + MathVista)/3, **Avg6** = mean of all six.

## Before submission — TODO

1. **Citations to verify.** `rose2025` and `vora2025` are placeholder entries
   in `references.bib` (author/venue marked `TODO`). Replace them with the
   official records of the ROSE and VoRA papers.
2. **Author block.** `main_cvpr_official.tex` carries the anonymous placeholder;
   fill in the camera-ready block only after acceptance.
3. **Qualitative figure.** A side-by-side prediction figure (ours correct /
   baseline hallucinating, plus an OPD before-after pair) would strengthen
   Sec. 4.2 and is referenced in the outline of the rebuttal plan but not yet
   included.
4. **Page limit.** With 7 tables and 3 figures the draft is close to the 8-page
   limit; Tabs. 3–5 can move to the supplementary material if needed.
5. **Optional ablation.** The `rots` frontend was trained under two rotation
   initialisations (Tab. 3 groups). Re-running the missing cells under a single
   initialisation would let the table collapse into one controlled block.
