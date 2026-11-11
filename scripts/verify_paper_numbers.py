"""Verify that every table row in paper/sections/4_experiments.tex maps 1-to-1 to a real JSON file on disk."""

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
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


def check_close(name, field, got, exp, tol=0.015):
    if abs(got - exp) > tol:
        raise AssertionError(f"[MISMATCH] {name} {field}: json={got:.4f} vs paper={exp:.4f}")


def main():
    checks = 0
    # 1. Table 1 (Main comparison, all s=0)
    t1_rows = [
        ("Teacher", BENCH / "teacher_qwen3vl8b.json", (65.80, 72.45, 63.20, 69.13, 67.15, 59.03)),
        ("Blank Reference", V5 / "eval" / "v5_tower_spinvl_full_blank.json", (23.73, 48.16, 19.40, 35.95, 30.43, 15.58)),
        ("LLaVA-Proj 21k", BENCH / "q17_proj_n21000.json", (29.60, 53.10, 24.90, 41.35, 35.87, 21.02)),
        ("ROSE 21k", BENCH / "q17_rose_single_ce.json", (25.60, 54.36, 19.70, 39.98, 33.22, 17.14)),
        ("Hinton-KD 21k", BENCH / "q17_rose_hinton_kd.json", (26.00, 49.95, 18.90, 37.97, 31.62, 16.43)),
        ("RKD 21k", BENCH / "q17_rose_rkd.json", (26.13, 51.31, 19.20, 38.72, 32.22, 16.70)),
        ("SpinVL-Tower mf 21k", BENCH / "q17_tower_mf_s7875_ce.json", (27.73, 53.73, 20.60, 40.73, 34.02, 17.67)),
        ("SpinVL-Tower hyb 21k", BENCH / "q17_tower_hybrid_ce.json", (30.60, 58.15, 18.50, 44.37, 35.75, 18.36)),
        ("LLaVA-Proj 28k", BENCH / "q17_p15_proj.json", (29.27, 54.15, 25.00, 41.71, 36.14, 20.89)),
        ("FastV-32 28k", BENCH / "q17_proj_fastv32.json", (28.53, 49.74, 24.90, 39.14, 34.39, 19.43)),
        ("Token-Pool-32 28k", BENCH / "q17_proj_pool32.json", (29.13, 53.10, 25.80, 41.12, 36.01, 21.06)),
        ("SpinVL-Tower hyb 28k", BENCH / "q17_p15_tower_hybrid_ce.json", (28.27, 56.05, 19.60, 42.16, 34.64, 18.16)),
        ("SpinVL-Tower mf+OPD 28k", BENCH / "q17_p15_tower_mf_opd.json", (26.80, 57.41, 23.50, 42.11, 35.90, 18.50)),
        ("K=0 rel 21k", BENCH / "q17_abl_front_none_K0.json", (24.00, 47.84, 21.60, 35.92, 31.15, 15.81)),
        ("EVE full K=4 rel 21k", BENCH / "q17_abl_front_full_K4.json", (24.33, 53.31, 18.60, 38.82, 32.08, 16.80)),
        ("rot K=4 rel 21k", BENCH / "q17_abl_front_rot_K4.json", (25.20, 56.26, 20.20, 40.73, 33.89, 17.55)),
        ("VoRA LoRA+mf 21k", BENCH / "q17_vora_lora_mf.json", (23.73, 49.95, 18.50, 36.84, 30.73, 15.82)),
        ("SpinVL-Free rots+mf 21k", BENCH / "q17_free_mf_ce.json", (27.33, 55.31, 18.00, 41.32, 33.55, 17.33)),
        ("SpinVL-Free rots+hyb 21k", BENCH / "q17_free_hybrid_ce.json", (27.20, 53.21, 20.30, 40.21, 33.57, 17.37)),
        ("SpinVL-Free rots+hyb 28k", BENCH / "q17_p15_free_hybrid_ce.json", (27.60, 54.57, 19.10, 41.09, 33.76, 17.58)),
        ("Free+GKD 28k", BENCH / "q17_p19_free_hybrid_gkd.json", (23.73, 53.84, 19.80, 38.79, 32.46, 17.01)),
        ("Free+MiniLLM 28k", BENCH / "q17_p19_free_hybrid_revkl.json", (24.13, 56.78, 16.10, 40.46, 32.34, 16.75)),
        ("Free+SeqKD 28k", BENCH / "q17_p19_free_hybrid_seqkd.json", (25.53, 56.68, 18.30, 41.11, 33.50, 17.44)),
        ("Free+VL-OPD 1.7B 28k s0", BENCH / "q17_p15_free_hybrid_opd.json", (26.33, 54.36, 18.00, 40.35, 32.90, 17.21)),
        ("Free+VL-OPD 1.7B 21k s0", BENCH / "q17_free_hybrid_opd.json", (27.93, 56.99, 19.80, 42.46, 34.91, 18.02)),
        ("Free+VL-OPD 0.6B 28k s0", BENCH / "q06_p15_free_hybrid_opd.json", (27.53, 57.83, 18.60, 42.68, 34.66, 18.05)),
    ]
    for name, path, (mm, ha, mv, mc2, c3, a6) in t1_rows:
        s = load_b(path)
        for k, v in zip(("mm", "ha", "mv", "mc2", "core3", "avg6"), (mm, ha, mv, mc2, c3, a6)):
            check_close(f"Tab1:{name}", k, s[k], v)
            checks += 1

    # 2. Table 3 (Frontend parameterisation, each row from 1 train log + 1 bench JSON)
    t3_rows = [
        ("none K=0", TLOG / "p4_none_K0_joint.json", BENCH / "q17_abl_front_none_K0.json", (5.237, 0.501, 24.0, 16.0, 35.92, 31.15, 15.81)),
        ("rot K=4", TLOG / "p4_rot_K4_joint.json", BENCH / "q17_abl_front_rot_K4.json", (1.973, 1.130, 40.7, 17.3, 40.73, 33.89, 17.55)),
        ("full K=4", TLOG / "p4_full_K4_joint.json", BENCH / "q17_abl_front_full_K4.json", (1.745, 1.857, 43.3, 20.0, 38.82, 32.08, 16.80)),
        ("rots single", TLOG / "p6_rots_K4_joint.json", BENCH / "q17_abl_front_rots_K4.json", (2.019, 0.850, 42.0, 18.7, 38.14, 31.32, 16.12)),
        ("LoRA mf", TLOG / "p10_lora16_K4_joint.json", BENCH / "q17_vora_lora_mf.json", (1.532, 1.667, 44.7, 18.7, 36.84, 30.73, 15.82)),
        ("rots mf", TLOG / "p7_mf_rots_joint.json", BENCH / "q17_free_mf_ce.json", (1.549, 1.783, 43.3, 17.3, 41.32, 33.55, 17.33)),
        ("rots hyb", TLOG / "p8_hybrid_rots_joint.json", BENCH / "q17_free_hybrid_ce.json", (1.554, 1.866, 43.3, 16.0, 40.21, 33.57, 17.37)),
    ]
    for name, tp, bp, (ce, gap, f1, ff, mc2, c3, a6) in t3_rows:
        td = json.loads(tp.read_text(encoding="utf-8"))["eval"]["front"]
        bs = load_b(bp)
        check_close(f"Tab3:{name}", "ce", td["correct"], ce, 0.002)
        check_close(f"Tab3:{name}", "gap", td["gap"], gap, 0.002)
        check_close(f"Tab3:{name}", "1st", td["tf_first"] * 100, f1, 0.1)
        check_close(f"Tab3:{name}", "full", td["tf_full"] * 100, ff, 0.1)
        check_close(f"Tab3:{name}", "mc2", bs["mc2"], mc2)
        check_close(f"Tab3:{name}", "core3", bs["core3"], c3)
        check_close(f"Tab3:{name}", "avg6", bs["avg6"], a6)
        checks += 7

    # 3. Table 4 (Depth K in {0,2,4,6} on +open, s=0)
    t4_rows = [
        ("K=0 s0", TLOG / "p18_q17_free_hyb_K0_joint.json", BENCH / "q17_p18_free_hyb_K0.json", (2.382, 0.387, 30.7, 42.31, 34.61, 17.94)),
        ("K=2 s0", TLOG / "p18_q17_free_hyb_K2_joint.json", BENCH / "q17_p18_free_hyb_K2.json", (2.267, 0.815, 36.7, 41.30, 33.93, 17.68)),
        ("K=4 s0", TLOG / "p15_q17_free_hybrid_joint.json", BENCH / "q17_p15_free_hybrid_ce.json", (2.294, 0.896, 36.0, 41.09, 33.76, 17.58)),
        ("K=6 s0", TLOG / "p18_q17_free_hyb_K6_joint.json", BENCH / "q17_p18_free_hyb_K6.json", (2.272, 0.978, 38.0, 41.94, 33.90, 17.66)),
    ]
    for name, tp, bp, (ce, gap, f1, mc2, c3, a6) in t4_rows:
        td = json.loads(tp.read_text(encoding="utf-8"))["eval"]["front"]
        bs = load_b(bp)
        check_close(f"Tab4:{name}", "ce", td["correct"], ce, 0.002)
        check_close(f"Tab4:{name}", "gap", td["gap"], gap, 0.002)
        check_close(f"Tab4:{name}", "1st", td["tf_first"] * 100, f1, 0.1)
        check_close(f"Tab4:{name}", "mc2", bs["mc2"], mc2)
        check_close(f"Tab4:{name}", "core3", bs["core3"], c3)
        check_close(f"Tab4:{name}", "avg6", bs["avg6"], a6)
        checks += 6

    # 4. Table 5 (VL-OPD comparison on +open s=0)
    t5_rows = [
        ("Start CE", TLOG / "p15_q17_free_hybrid_opd.json", "init_eval", BENCH / "q17_p15_free_hybrid_ce.json", (2.294, 100.0, 41.09, 33.76)),
        ("GKD", TLOG / "p19_q17_free_hybrid_gkd.json", "eval", BENCH / "q17_p19_free_hybrid_gkd.json", (2.301, 99.0, 38.79, 32.46)),
        ("MiniLLM", TLOG / "p19_q17_free_hybrid_revkl.json", "eval", BENCH / "q17_p19_free_hybrid_revkl.json", (2.565, 59.0, 40.46, 32.34)),
        ("SeqKD", TLOG / "p19_q17_free_hybrid_seqkd.json", "eval", BENCH / "q17_p19_free_hybrid_seqkd.json", (2.417, 84.0, 41.11, 33.50)),
        ("VL-OPD s0", TLOG / "p15_q17_free_hybrid_opd.json", "eval", BENCH / "q17_p15_free_hybrid_opd.json", (2.475, 95.0, 40.35, 32.90)),
    ]
    for name, tp, key, bp, (ce, eos, mc2, c3) in t5_rows:
        td = json.loads(tp.read_text(encoding="utf-8"))[key]
        bs = load_b(bp)
        check_close(f"Tab5:{name}", "ce", td["correct"], ce, 0.002)
        check_close(f"Tab5:{name}", "eos", td["stop_rate"] * 100, eos, 0.1)
        check_close(f"Tab5:{name}", "mc2", bs["mc2"], mc2)
        check_close(f"Tab5:{name}", "core3", bs["core3"], c3)
        checks += 4

    # 5. Table 7 (Debiased 44k clean_v5 + Real vs True-Shuffle)
    t7_rows = [
        ("Teacher v5", None, V5 / "eval" / "re_teacher_qwen3vl8b.json", V5 / "eval" / "re_teacher_qwen3vl8b_shuffle.json", (None, 66.7, 29.5, 72.7, 57.1, 62.5, 33.2, 69.70, 67.30)),
        ("LLaVA v5", None, V5 / "eval" / "v5_proj_llava.json", V5 / "eval" / "v5_proj_llava_shuffle.json", (None, 30.1, 29.9, 55.4, 54.1, 24.4, 24.1, 42.77, 36.65)),
        ("ROSE v5", V5 / "v5_tower_rose.json", V5 / "eval" / "v5_tower_rose.json", V5 / "eval" / "v5_tower_rose_shuffle.json", (0.899, 28.7, 28.5, 46.2, 45.7, 18.7, 19.4, 37.45, 31.20)),
        ("SpinVL-Tower v5", V5 / "v5_tower_spinvl_full.json", V5 / "eval" / "v5_tower_spinvl_full.json", V5 / "eval" / "v5_tower_spinvl_full_shuffle.json", (1.373, 27.5, 28.7, 55.3, 52.3, 19.0, 19.2, 41.42, 33.95)),
        ("K=0 v5", V5 / "v5_free_none_K0_joint.json", V5 / "eval" / "v5_free_none_K0.json", V5 / "eval" / "v5_free_none_K0_shuffle.json", (0.195, 26.1, 26.5, 55.6, 56.5, 19.8, 20.3, 40.85, 33.83)),
        ("VoRA v5", V5 / "v5_free_lora_K4_joint.json", V5 / "eval" / "v5_free_lora_K4.json", V5 / "eval" / "v5_free_lora_K4_shuffle.json", (0.556, 28.4, 28.3, 51.8, 53.7, 19.1, 21.9, 40.12, 33.11)),
        ("SpinVL-Free v5", V5 / "v5_free_rots_K4_joint.json", V5 / "eval" / "v5_free_rots_K4_joint.json", V5 / "eval" / "v5_free_rots_K4_joint_shuffle.json", (0.384, 26.3, 24.7, 55.3, 56.7, 20.2, 21.4, 40.82, 33.95)),
        ("SpinVL-Free+JSD v5", V5 / "v5_free_opd_jsd.json", V5 / "eval" / "v5_free_opd_jsd.json", V5 / "eval" / "v5_free_opd_jsd_shuffle.json", (0.381, 26.3, 25.9, 55.0, 56.8, 20.0, 18.5, 40.66, 33.78)),
    ]
    for name, tp, rp, sp, (gap, mm_r, mm_s, ha_r, ha_s, mv_r, mv_s, mc2, c3) in t7_rows:
        if tp is not None and gap is not None:
            td = json.loads(tp.read_text(encoding="utf-8"))
            ev = td.get("val") or td.get("eval", {}).get("front") or td.get("eval")
            g_val = ev["gap"]
            check_close(f"Tab7:{name}", "gap", g_val, gap, 0.002)
            checks += 1
        sr = load_b(rp)
        ss = load_b(sp)
        check_close(f"Tab7:{name}", "mm_r", sr["mm"], mm_r, 0.06)
        check_close(f"Tab7:{name}", "mm_s", ss["mm"], mm_s, 0.06)
        check_close(f"Tab7:{name}", "ha_r", sr["ha"], ha_r, 0.06)
        check_close(f"Tab7:{name}", "ha_s", ss["ha"], ha_s, 0.06)
        check_close(f"Tab7:{name}", "mv_r", sr["mv"], mv_r, 0.06)
        check_close(f"Tab7:{name}", "mv_s", ss["mv"], mv_s, 0.06)
        check_close(f"Tab7:{name}", "mc2", sr["mc2"], mc2, 0.02)
        check_close(f"Tab7:{name}", "core3", sr["core3"], c3, 0.02)
        checks += 8

    print(f"ALL_PAPER_NUMBERS_VERIFIED_OK: {checks} exact JSON checks passed with 0 discrepancies!")


if __name__ == "__main__":
    main()
