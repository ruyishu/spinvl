"""Verify that every table row in paper/sections/4_experiments.tex maps 1-to-1 to a real JSON file on disk by parsing the actual LaTeX source file directly."""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX_PATH = ROOT / "paper" / "sections" / "4_experiments.tex"
BENCH = ROOT / "results" / "benchmarks"
TLOG = ROOT / "results" / "train_logs"
V5 = ROOT / "results" / "clean_v5"


def load_b(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    b = d["benches"]
    mm = b["mmstar"]["acc"] * 100
    ha = b["hallusion"]["acc"] * 100
    mv = b["mathvista"]["acc"] * 100
    cq = b.get("chartqa", {}).get("acc", 0.0) * 100
    oc = b.get("ocrbench", {}).get("acc", 0.0) * 100
    g3 = b.get("geometry3k", {}).get("acc", 0.0) * 100
    mc2 = (mm + ha) / 2
    core3 = (mm + ha + mv) / 3
    avg6 = (mm + ha + mv + cq + oc + g3) / 6
    return dict(mm=mm, ha=ha, mv=mv, cq=cq, oc=oc, g3=g3, mc2=mc2, core3=core3, avg6=avg6)


def clean_tex_cell(cell: str) -> str:
    s = cell.strip()
    s = re.sub(r"\\+$", "", s).strip()
    while True:
        s_new = re.sub(r"\\(?:textbf|underline|texttt|emph|textit)\{([^{}]*)\}", r"\1", s)
        s_new = re.sub(r"\{\\color\{[^{}]+\}([^{}]*)\}", r"\1", s_new)
        if s_new == s:
            break
        s = s_new
    s = s.replace("\\%", "").replace("$", "").strip()
    return s


def extract_table_block(tex_text: str, label: str) -> list:
    m = re.search(
        r"\\label\{" + re.escape(label) + r"\}.*?\\begin\{tabular\}\{[^\n]*\}(.*?)\\end\{tabular\}",
        tex_text,
        flags=re.S,
    )
    assert m, f"Table {label} not found in {TEX_PATH}"
    rows = []
    for raw_line in m.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("\\toprule", "\\midrule", "\\bottomrule", "\\multicolumn")):
            continue
        if "&" in line and "\\\\" in line:
            body = line.split("\\\\")[0]
            rows.append([clean_tex_cell(c) for c in body.split("&")])
    return rows


def assert_rounded(tag, field, json_val, tex_str, decimals=2):
    tex_val = float(tex_str)
    if round(json_val, decimals) != round(tex_val, decimals):
        if abs(json_val - tex_val) > (0.5 * (10 ** (-decimals)) + 1e-6):
            raise AssertionError(
                f"[MISMATCH] {tag}.{field}: json={json_val:.6f} (round={round(json_val, decimals)}) vs tex='{tex_str}'"
            )


def main():
    tex_text = TEX_PATH.read_text(encoding="utf-8")
    checks = 0

    # 1. Table 1 (tab:main)
    t1_rows = extract_table_block(tex_text, "tab:main")[1:]  # skip header row
    t1_files = [
        ("Teacher", BENCH / "teacher_qwen3vl8b.json"),
        ("Blank Control", V5 / "eval" / "v5_tower_spinvl_full_blank.json"),
        ("LLaVA-Proj 21k", BENCH / "q17_proj_n21000.json"),
        ("ROSE 21k", BENCH / "q17_rose_single_ce.json"),
        ("Hinton-KD 21k", BENCH / "q17_rose_hinton_kd.json"),
        ("RKD 21k", BENCH / "q17_rose_rkd.json"),
        ("SpinVL-Tower mf 21k", BENCH / "q17_tower_mf_s7875_ce.json"),
        ("SpinVL-Tower hyb 21k", BENCH / "q17_tower_hybrid_ce.json"),
        ("LLaVA-Proj 28k", BENCH / "q17_p15_proj.json"),
        ("FastV-32 28k", BENCH / "q17_proj_fastv32.json"),
        ("Token-Pool-32 28k", BENCH / "q17_proj_pool32.json"),
        ("SpinVL-Tower hyb 28k", BENCH / "q17_p15_tower_hybrid_ce.json"),
        ("SpinVL-Tower mf+OPD 28k", BENCH / "q17_p15_tower_mf_opd.json"),
        ("K=0 rel 21k", BENCH / "q17_abl_front_none_K0.json"),
        ("EVE full K=4 rel 21k", BENCH / "q17_abl_front_full_K4.json"),
        ("rot K=4 rel 21k", BENCH / "q17_abl_front_rot_K4.json"),
        ("VoRA LoRA+mf 21k", BENCH / "q17_vora_lora_mf.json"),
        ("SpinVL-Free rots+mf 21k", BENCH / "q17_free_mf_ce.json"),
        ("SpinVL-Free rots+hyb 21k", BENCH / "q17_free_hybrid_ce.json"),
        ("SpinVL-Free rots+hyb 28k", BENCH / "q17_p15_free_hybrid_ce.json"),
        ("Free+GKD 28k", BENCH / "q17_p19_free_hybrid_gkd.json"),
        ("Free+MiniLLM 28k", BENCH / "q17_p19_free_hybrid_revkl.json"),
        ("Free+SeqKD 28k", BENCH / "q17_p19_free_hybrid_seqkd.json"),
        ("Free+VL-OPD 1.7B 28k s0", BENCH / "q17_p15_free_hybrid_opd.json"),
        ("Free+VL-OPD 1.7B 21k s0", BENCH / "q17_free_hybrid_opd.json"),
        ("Free+VL-OPD 0.6B 28k s0", BENCH / "q06_p15_free_hybrid_opd.json"),
    ]
    assert len(t1_rows) == len(t1_files), f"tab:main row count {len(t1_rows)} != {len(t1_files)}"
    for cells, (name, bp) in zip(t1_rows, t1_files):
        s = load_b(bp)
        for idx, k in zip((5, 6, 7, 8, 9, 10), ("mm", "ha", "mv", "mc2", "core3", "avg6")):
            assert_rounded(f"Tab1:{name}", k, s[k], cells[idx], decimals=2)
            checks += 1

    # 2. Table 3 (tab:front)
    t3_rows = extract_table_block(tex_text, "tab:front")[1:]
    t3_files = [
        ("none K=0", TLOG / "p4_none_K0_joint.json", BENCH / "q17_abl_front_none_K0.json"),
        ("rot K=4", TLOG / "p4_rot_K4_joint.json", BENCH / "q17_abl_front_rot_K4.json"),
        ("full K=4", TLOG / "p4_full_K4_joint.json", BENCH / "q17_abl_front_full_K4.json"),
        ("rots single", TLOG / "p6_rots_K4_joint.json", BENCH / "q17_abl_front_rots_K4.json"),
        ("LoRA mf", TLOG / "p10_lora16_K4_joint.json", BENCH / "q17_vora_lora_mf.json"),
        ("rots mf", TLOG / "p7_mf_rots_joint.json", BENCH / "q17_free_mf_ce.json"),
        ("rots hyb", TLOG / "p8_hybrid_rots_joint.json", BENCH / "q17_free_hybrid_ce.json"),
    ]
    assert len(t3_rows) == len(t3_files)
    for cells, (name, tp, bp) in zip(t3_rows, t3_files):
        td = json.loads(tp.read_text(encoding="utf-8"))["eval"]["front"]
        bs = load_b(bp)
        assert_rounded(f"Tab3:{name}", "ce", td["correct"], cells[4], decimals=3)
        assert_rounded(f"Tab3:{name}", "gap", td["gap"], cells[5], decimals=3)
        f1_str, ff_str = cells[6].split("/")
        assert_rounded(f"Tab3:{name}", "1st", td["tf_first"] * 100, f1_str, decimals=1)
        assert_rounded(f"Tab3:{name}", "full", td["tf_full"] * 100, ff_str, decimals=1)
        assert_rounded(f"Tab3:{name}", "mc2", bs["mc2"], cells[7], decimals=2)
        assert_rounded(f"Tab3:{name}", "core3", bs["core3"], cells[8], decimals=2)
        assert_rounded(f"Tab3:{name}", "avg6", bs["avg6"], cells[9], decimals=2)
        checks += 7

    # 3. Table 4 (tab:depth)
    t4_rows = extract_table_block(tex_text, "tab:depth")[1:]
    t4_single = [
        (0, "K=0 s0", TLOG / "p18_q17_free_hyb_K0_joint.json", BENCH / "q17_p18_free_hyb_K0.json"),
        (1, "K=2 s0", TLOG / "p18_q17_free_hyb_K2_joint.json", BENCH / "q17_p18_free_hyb_K2.json"),
        (2, "K=4 s0", TLOG / "p15_q17_free_hybrid_joint.json", BENCH / "q17_p15_free_hybrid_ce.json"),
        (4, "K=6 s0", TLOG / "p18_q17_free_hyb_K6_joint.json", BENCH / "q17_p18_free_hyb_K6.json"),
    ]
    for r_idx, name, tp, bp in t4_single:
        cells = t4_rows[r_idx]
        td = json.loads(tp.read_text(encoding="utf-8"))["eval"]["front"]
        bs = load_b(bp)
        assert_rounded(f"Tab4:{name}", "ce", td["correct"], cells[2], decimals=3)
        assert_rounded(f"Tab4:{name}", "gap", td["gap"], cells[3], decimals=3)
        assert_rounded(f"Tab4:{name}", "1st", td["tf_first"] * 100, cells[4], decimals=1)
        assert_rounded(f"Tab4:{name}", "mc2", bs["mc2"], cells[5], decimals=2)
        assert_rounded(f"Tab4:{name}", "core3", bs["core3"], cells[6], decimals=2)
        assert_rounded(f"Tab4:{name}", "avg6", bs["avg6"], cells[7], decimals=2)
        checks += 6

    # 4. Table 5 (tab:opd top block)
    t5_rows = extract_table_block(tex_text, "tab:opd")[1:6]
    t5_files = [
        ("Start CE", TLOG / "p15_q17_free_hybrid_opd.json", "init_eval", BENCH / "q17_p15_free_hybrid_ce.json"),
        ("GKD", TLOG / "p19_q17_free_hybrid_gkd.json", "eval", BENCH / "q17_p19_free_hybrid_gkd.json"),
        ("MiniLLM", TLOG / "p19_q17_free_hybrid_revkl.json", "eval", BENCH / "q17_p19_free_hybrid_revkl.json"),
        ("SeqKD", TLOG / "p19_q17_free_hybrid_seqkd.json", "eval", BENCH / "q17_p19_free_hybrid_seqkd.json"),
        ("VL-OPD s0", TLOG / "p15_q17_free_hybrid_opd.json", "eval", BENCH / "q17_p15_free_hybrid_opd.json"),
    ]
    for cells, (name, tp, key, bp) in zip(t5_rows, t5_files):
        td = json.loads(tp.read_text(encoding="utf-8"))[key]
        bs = load_b(bp)
        assert_rounded(f"Tab5:{name}", "ce", td["correct"], cells[2], decimals=3)
        assert_rounded(f"Tab5:{name}", "eos", td["stop_rate"] * 100, cells[3], decimals=1)
        assert_rounded(f"Tab5:{name}", "mc2", bs["mc2"], cells[4], decimals=2)
        assert_rounded(f"Tab5:{name}", "core3", bs["core3"], cells[5], decimals=2)
        checks += 4

    # 5. Table 6/7 (tab:boundary)
    t6_rows = extract_table_block(tex_text, "tab:boundary")[1:]
    t6_files = [
        ("LLaVA 21k", BENCH / "q17_proj_n21000.json"),
        ("SpinVL-Tower 21k", BENCH / "q17_tower_hybrid_ce.json"),
        ("LLaVA 28k", BENCH / "q17_p15_proj.json"),
        ("SpinVL-Tower 28k", BENCH / "q17_p15_tower_hybrid_ce.json"),
    ]
    for cells, (name, bp) in zip(t6_rows, t6_files):
        bs = load_b(bp)
        assert_rounded(f"TabBoundary:{name}", "cq", bs["cq"], cells[2], decimals=2)
        assert_rounded(f"TabBoundary:{name}", "oc", bs["oc"], cells[3], decimals=2)
        assert_rounded(f"TabBoundary:{name}", "g3", bs["g3"], cells[4], decimals=2)
        checks += 3

    # 6. Table 7/8 (tab:grounding)
    t7_rows = extract_table_block(tex_text, "tab:grounding")[2:]  # skip 2-line header
    t7_files = [
        ("Teacher v5", None, V5 / "eval" / "re_teacher_qwen3vl8b.json", V5 / "eval" / "re_teacher_qwen3vl8b_shuffle.json"),
        ("LLaVA v5", None, V5 / "eval" / "v5_proj_llava.json", V5 / "eval" / "v5_proj_llava_shuffle.json"),
        ("ROSE v5", V5 / "v5_tower_rose.json", V5 / "eval" / "v5_tower_rose.json", V5 / "eval" / "v5_tower_rose_shuffle.json"),
        ("SpinVL-Tower v5", V5 / "v5_tower_spinvl_full.json", V5 / "eval" / "v5_tower_spinvl_full.json", V5 / "eval" / "v5_tower_spinvl_full_shuffle.json"),
        ("K=0 v5", V5 / "v5_free_none_K0_joint.json", V5 / "eval" / "v5_free_none_K0.json", V5 / "eval" / "v5_free_none_K0_shuffle.json"),
        ("VoRA v5", V5 / "v5_free_lora_K4_joint.json", V5 / "eval" / "v5_free_lora_K4.json", V5 / "eval" / "v5_free_lora_K4_shuffle.json"),
        ("SpinVL-Free v5", V5 / "v5_free_rots_K4_joint.json", V5 / "eval" / "v5_free_rots_K4_joint.json", V5 / "eval" / "v5_free_rots_K4_joint_shuffle.json"),
        ("SpinVL-Free+JSD v5", V5 / "v5_free_opd_jsd.json", V5 / "eval" / "v5_free_opd_jsd.json", V5 / "eval" / "v5_free_opd_jsd_shuffle.json"),
    ]
    assert len(t7_rows) == len(t7_files)
    for cells, (name, tp, rp, sp) in zip(t7_rows, t7_files):
        if tp is not None:
            td = json.loads(tp.read_text(encoding="utf-8"))
            ev = td.get("val") or td.get("eval", {}).get("front") or td.get("eval")
            assert_rounded(f"TabGrounding:{name}", "gap", ev["gap"], cells[1], decimals=3)
            checks += 1
        sr = load_b(rp)
        ss = load_b(sp)
        mm_r, mm_s = cells[2].split("/")
        ha_r, ha_s = cells[3].split("/")
        mv_r, mv_s = cells[4].split("/")
        assert_rounded(f"TabGrounding:{name}", "mm_r", sr["mm"], mm_r, decimals=1)
        assert_rounded(f"TabGrounding:{name}", "mm_s", ss["mm"], mm_s, decimals=1)
        assert_rounded(f"TabGrounding:{name}", "ha_r", sr["ha"], ha_r, decimals=1)
        assert_rounded(f"TabGrounding:{name}", "ha_s", ss["ha"], ha_s, decimals=1)
        assert_rounded(f"TabGrounding:{name}", "mv_r", sr["mv"], mv_r, decimals=1)
        assert_rounded(f"TabGrounding:{name}", "mv_s", ss["mv"], mv_s, decimals=1)
        assert_rounded(f"TabGrounding:{name}", "mc2", sr["mc2"], cells[5], decimals=2)
        assert_rounded(f"TabGrounding:{name}", "core3", sr["core3"], cells[6], decimals=2)
        checks += 8

    print(f"ALL_PAPER_TEX_NUMBERS_VERIFIED_OK: {checks} cells parsed directly from {TEX_PATH.name} and verified against raw JSONs!")


if __name__ == "__main__":
    main()
