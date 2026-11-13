"""Verify that every table row across all 8 tables in paper/sections/4_experiments.tex maps 1-to-1 to real JSON files on disk by parsing the actual LaTeX source file directly."""

import argparse
import json
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX_PATH = ROOT / "paper" / "sections" / "4_experiments.tex"
BENCH = ROOT / "results" / "benchmarks"
TLOG = ROOT / "results" / "train_logs"
V5 = ROOT / "results" / "clean_v5"
V6 = ROOT / "results" / "clean_v6"
PRES = ROOT / "results" / "paper_results.json"


def load_b(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    b = d["benches"]
    mm = b["mmstar"]["acc"] * 100
    ha = b["hallusion"]["acc"] * 100
    mv = b["mathvista"]["acc"] * 100
    cq = b["chartqa"]["acc"] * 100 if "chartqa" in b else None
    oc = b["ocrbench"]["acc"] * 100 if "ocrbench" in b else None
    g3 = b["geometry3k"]["acc"] * 100 if "geometry3k" in b else None
    mc2 = (mm + ha) / 2
    core3 = (mm + ha + mv) / 3
    avg6 = (mm + ha + mv + cq + oc + g3) / 6 if all(x is not None for x in (cq, oc, g3)) else None
    return dict(mm=mm, ha=ha, mv=mv, cq=cq, oc=oc, g3=g3, mc2=mc2, core3=core3, avg6=avg6)


def clean_tex_cell(cell: str) -> str:
    s = cell.strip()
    s = re.sub(r"\\+$", "", s).strip()
    while True:
        s_new = re.sub(r"\\(?:textbf|underline|texttt|emph|textit|mathbf)\{([^{}]*)\}", r"\1", s)
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


def parse_mean_std(cell_str: str):
    m = re.match(r"([0-9.]+)_\{?\\pm([0-9.]+)\}?", cell_str)
    assert m, f"Cannot parse mean+-std from '{cell_str}'"
    return float(m.group(1)), float(m.group(2))


def main():
    global ROOT, TEX_PATH, BENCH, TLOG, V5, V6, PRES
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=ROOT)
    ap.add_argument("--tex", type=pathlib.Path, help="Verify a separate TeX copy against --root results")
    args = ap.parse_args()
    ROOT = args.root.resolve()
    TEX_PATH = args.tex.resolve() if args.tex else ROOT / "paper/sections/4_experiments.tex"
    BENCH, TLOG, V5, V6 = ROOT / "results/benchmarks", ROOT / "results/train_logs", ROOT / "results/clean_v5", ROOT / "results/clean_v6"
    PRES = ROOT / "results/paper_results.json"
    tex_text = TEX_PATH.read_text(encoding="utf-8")
    checks = 0

    # 1. Table 1 (tab:main)
    t1_rows = extract_table_block(tex_text, "tab:main")[1:]
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
    assert len(t1_rows) == len(t1_files)
    for cells, (name, bp) in zip(t1_rows, t1_files):
        s = load_b(bp)
        for idx, k in zip((5, 6, 7, 8, 9, 10), ("mm", "ha", "mv", "mc2", "core3", "avg6")):
            assert_rounded(f"Tab1:{name}", k, s[k], cells[idx], decimals=2)
            checks += 1

    # 2. Table 2 (tab:scale: single runs + 3-seed mean/std rows)
    t2_rows = extract_table_block(tex_text, "tab:scale")[1:]
    t2_spec = [
        ("0.6B Tower 28k s0", [BENCH / "q06_p15_tower_hybrid_ce.json"]),
        ("0.6B Free 21k s0", [BENCH / "q06_free_ce.json"]),
        ("0.6B Free+OPD 21k s0", [BENCH / "q06_free_opd.json"]),
        ("0.6B Free+OPD 28k s0", [BENCH / "q06_p15_free_hybrid_opd.json"]),
        ("0.6B Free+OPD 28k 3seeds", [
            BENCH / "q06_p15_free_hybrid_opd.json",
            BENCH / "q06_p15_free_hybrid_opd_s1.json",
            BENCH / "q06_p15_free_hybrid_opd_s2.json",
        ]),
        ("1.7B Tower 28k 3seeds", [
            BENCH / "q17_p15_tower_hybrid_ce.json",
            BENCH / "q17_p15_tower_hybrid_ce_s1.json",
            BENCH / "q17_p15_tower_hybrid_ce_s2.json",
        ]),
        ("1.7B Free 28k 3seeds", [
            BENCH / "q17_p15_free_hybrid_ce.json",
            BENCH / "q17_p15_free_hybrid_ce_s1.json",
            BENCH / "q17_p15_free_hybrid_ce_s2.json",
        ]),
        ("1.7B Free+OPD 28k 3seeds", [
            BENCH / "q17_p15_free_hybrid_opd.json",
            BENCH / "q17_p15_free_hybrid_opd_s1.json",
            BENCH / "q17_p15_free_hybrid_opd_s2.json",
        ]),
        ("8B Tower 28k s0", [BENCH / "q8b_p15_tower_hybrid_ce.json"]),
        ("8B Free 28k s0", [BENCH / "q8b_p15_free_hybrid_ce.json"]),
        ("8B Free+OPD 28k s0", [BENCH / "q8b_p15_free_hybrid_opd.json"]),
    ]
    assert len(t2_rows) == len(t2_spec)
    for cells, (name, paths) in zip(t2_rows, t2_spec):
        loaded = [load_b(p) for p in paths]
        for idx, k in zip((2, 3, 4, 5, 6, 7), ("mm", "ha", "mv", "mc2", "core3", "avg6")):
            if len(loaded) == 1:
                assert_rounded(f"Tab2:{name}", k, loaded[0][k], cells[idx], decimals=2)
                checks += 1
            else:
                vals = [x[k] for x in loaded]
                mean_v = statistics.mean(vals)
                std_s = statistics.stdev(vals)
                m_tex, s_tex = parse_mean_std(cells[idx])
                assert_rounded(f"Tab2:{name}", f"{k}_mean", mean_v, str(m_tex), decimals=2)
                assert_rounded(f"Tab2:{name}", f"{k}_sample_std_ddof1", std_s, str(s_tex), decimals=2)
                checks += 2

    # 3. Table 3 (tab:front)
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

    # 4. Table 4 (tab:depth: all 5 rows including 3-seed mean row)
    t4_rows = extract_table_block(tex_text, "tab:depth")[1:]
    t4_single = [
        (0, "K=0 s0", [TLOG / "p18_q17_free_hyb_K0_joint.json"], [BENCH / "q17_p18_free_hyb_K0.json"]),
        (1, "K=2 s0", [TLOG / "p18_q17_free_hyb_K2_joint.json"], [BENCH / "q17_p18_free_hyb_K2.json"]),
        (2, "K=4 s0", [TLOG / "p15_q17_free_hybrid_joint.json"], [BENCH / "q17_p15_free_hybrid_ce.json"]),
        (3, "K=4 3seeds", [
            TLOG / "p15_q17_free_hybrid_joint.json",
            TLOG / "p18_q17_free_hybrid_joint_s1.json",
            TLOG / "p18_q17_free_hybrid_joint_s2.json",
        ], [
            BENCH / "q17_p15_free_hybrid_ce.json",
            BENCH / "q17_p15_free_hybrid_ce_s1.json",
            BENCH / "q17_p15_free_hybrid_ce_s2.json",
        ]),
        (4, "K=6 s0", [TLOG / "p18_q17_free_hyb_K6_joint.json"], [BENCH / "q17_p18_free_hyb_K6.json"]),
    ]
    for r_idx, name, tps, bps in t4_single:
        cells = t4_rows[r_idx]
        tds = [json.loads(tp.read_text(encoding="utf-8"))["eval"]["front"] for tp in tps]
        bss = [load_b(bp) for bp in bps]
        assert_rounded(f"Tab4:{name}", "ce", statistics.mean(x["correct"] for x in tds), cells[2], decimals=3)
        assert_rounded(f"Tab4:{name}", "gap", statistics.mean(x["gap"] for x in tds), cells[3], decimals=3)
        assert_rounded(f"Tab4:{name}", "1st", statistics.mean(x["tf_first"] * 100 for x in tds), cells[4], decimals=1)
        assert_rounded(f"Tab4:{name}", "mc2", statistics.mean(x["mc2"] for x in bss), cells[5], decimals=2)
        assert_rounded(f"Tab4:{name}", "core3", statistics.mean(x["core3"] for x in bss), cells[6], decimals=2)
        assert_rounded(f"Tab4:{name}", "avg6", statistics.mean(x["avg6"] for x in bss), cells[7], decimals=2)
        checks += 6

    # 5. Table 5 (tab:opd: top block + bottom free-form generation block)
    t5_all = extract_table_block(tex_text, "tab:opd")
    t5_rows = t5_all[1:6]
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

    t5_fgen_rows = t5_all[6:]
    t5_fgen_files = [
        ("0.6B before", BENCH / "q06_p15_free_hyb_ce_fgen.json"),
        ("0.6B after", BENCH / "q06_p15_free_hyb_opd_fgen.json"),
        ("1.7B before", BENCH / "q17_p9_free_hyb_ce_fgen.json"),
        ("1.7B after", BENCH / "q17_p9_free_hyb_opd_fgen.json"),
    ]
    assert len(t5_fgen_rows) == len(t5_fgen_files)
    for cells, (name, bp) in zip(t5_fgen_rows, t5_fgen_files):
        bs = load_b(bp)
        assert_rounded(f"Tab5Fgen:{name}", "mm", bs["mm"], cells[2], decimals=2)
        assert_rounded(f"Tab5Fgen:{name}", "ha", bs["ha"], cells[3], decimals=2)
        assert_rounded(f"Tab5Fgen:{name}", "mv", bs["mv"], cells[4], decimals=2)
        assert_rounded(f"Tab5Fgen:{name}", "mc2", bs["mc2"], cells[5], decimals=2)
        checks += 4

    # 6. Table 6 (tab:eff)
    t_eff_rows = extract_table_block(tex_text, "tab:eff")[2:]  # skip 2-line header
    eff_json = json.loads(PRES.read_text(encoding="utf-8"))["efficiency_K4"]["rows"]
    assert len(t_eff_rows) == len(eff_json)
    for cells, erow in zip(t_eff_rows, eff_json):
        assert int(cells[0]) == erow["N_patches"]
        for c_idx, key in ((1, "ours"), (2, "rose"), (3, "llava")):
            tf_s, ms_s = cells[c_idx].split("/")
            assert_rounded(f"TabEff:{erow['N_patches']}.{key}", "tflops", erow[key]["tflops"], tf_s, decimals=2)
            assert_rounded(f"TabEff:{erow['N_patches']}.{key}", "ms", erow[key]["ms"], ms_s, decimals=1)
            checks += 2
        sav_expected = (erow["ours"]["tflops"] - erow["llava"]["tflops"]) / erow["llava"]["tflops"] * 100
        sav_str = cells[4].replace("\\mathbf{", "").replace("}", "").replace("−", "-")
        assert_rounded(f"TabEff:{erow['N_patches']}", "saving", sav_expected, sav_str, decimals=1)
        checks += 1

    # 7. Table 7 (tab:boundary)
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

    # 8. Table 8 (tab:grounding, clean_v6 44k strict-audit)
    t7_rows = extract_table_block(tex_text, "tab:grounding")[1:]
    t7_files = [
        ("LLaVA-Proj v6", None, V6 / "eval" / "v6_proj_llava_s0.json"),
        ("ROSE v6", V6 / "v6_tower_A0_rose_s0.json", V6 / "eval" / "v6_tower_A0_rose_s0.json"),
        ("CA-Only v6", V6 / "v6_tower_A1_caonly_s0.json", V6 / "eval" / "v6_tower_A1_caonly_s0.json"),
        ("R1 srot v6", V6 / "v6_tower_R1_srot_s0.json", V6 / "eval" / "v6_tower_R1_srot_s0.json"),
        ("R2 mf v6", V6 / "v6_tower_R2_mf_noqc_s0.json", V6 / "eval" / "v6_tower_R2_mf_noqc_s0.json"),
        ("R3 mf+qc v6", V6 / "v6_tower_R3_mf_qc_rot_s0.json", V6 / "eval" / "v6_tower_R3_mf_qc_rot_s0.json"),
        ("S0 K=0 v6", V6 / "v6_free_S0_none_mf_joint_s0.json", V6 / "eval" / "v6_free_S0_none_mf_s0.json"),
        ("S1 LoRA v6", V6 / "v6_free_S1_lora_mf_joint_s0.json", V6 / "eval" / "v6_free_S1_lora_mf_s0.json"),
        ("F2 rots+CA v6", V6 / "v6_free_F2_rots_ca_joint_s0.json", V6 / "eval" / "v6_free_F2_rots_ca_s0.json"),
        ("S2 rots+mf v6", V6 / "v6_free_S2_rots_mf_joint_s0.json", V6 / "eval" / "v6_free_S2_rots_mf_s0.json"),
        ("M0 cont CE v6", V6 / "v6_opd_M0_cont_ce_s0.json", V6 / "eval" / "v6_opd_M0_cont_ce_s0.json"),
        ("M1 OffKD v6", V6 / "v6_opd_M1_offkd_s0.json", V6 / "eval" / "v6_opd_M1_offkd_s0.json"),
        ("M2 Fwd-KL v6", V6 / "v6_opd_M2_fwdkl_s0.json", V6 / "eval" / "v6_opd_M2_fwdkl_s0.json"),
        ("M3 JSD v6", V6 / "v6_opd_M3_jsd_s0.json", V6 / "eval" / "v6_opd_M3_jsd_s0.json"),
    ]
    assert len(t7_rows) == len(t7_files)
    for cells, (name, tp, bp) in zip(t7_rows, t7_files):
        if tp is not None:
            td = json.loads(tp.read_text(encoding="utf-8"))
            if "val" in td:
                ev, em_v = td["val"], td["val"]["em"] * 100
            elif "eval" in td and "front" in td["eval"]:
                ev = td["eval"]["front"]
                em_v = td.get("em_front", {}).get("em", ev.get("em", 0)) * 100
            else:
                ev, em_v = td["eval"], td["eval"]["em"] * 100
            ce_s, em_s = cells[1].split("/")
            assert_rounded(f"TabGrounding:{name}", "ce", ev["correct"], ce_s, decimals=3)
            assert_rounded(f"TabGrounding:{name}", "em", em_v, em_s, decimals=1)
            assert_rounded(f"TabGrounding:{name}", "gap", ev["gap"], cells[2], decimals=3)
            checks += 3
        b = load_b(bp)
        for idx, k in zip((3, 4, 5, 6, 7, 8, 9), ("mm", "ha", "mv", "cq", "mc2", "core3", "avg6")):
            assert_rounded(f"TabGrounding:{name}", k, b[k], cells[idx], decimals=2)
            checks += 1

    print(f"ALL_8_TABLES_TEX_VERIFIED_OK: {checks} cells/stats across all 8 tables in {TEX_PATH.name} parsed and verified against JSONs!")


if __name__ == "__main__":
    main()
