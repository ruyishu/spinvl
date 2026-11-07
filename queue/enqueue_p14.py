"""phase14：
1) 冒烟闸门 p14smoke_qcond（验证 --qcond --direct-hyb + mix&geo_8b 合并训练 + v3 评测无误，3 分钟跑完）；
2) v3 全套核心模型重评（ev4_*，输出到 results/bench_eval_v3/，修掉 ChartQA/OCRBench 禁词乌龙）；
3) SpinVL-QCond（--qcond --direct-hyb，0 视觉 Token 进主干，带着问题查全图 cell + 合并 geo_8b.pt）：
   - 1.7B SpinVL-Tower (qcond+direct_hyb & qcond+mf) 4500/7875 步 -> VL-OPD -> v3 6 榜评测
   - 1.7B SpinVL-Free  (rots_K4 + qcond+direct_hyb & qcond+mf) 端到端联调 -> VL-OPD -> v3 6 榜评测
   - 0.6B & 8B 跨尺寸 SpinVL-Tower & SpinVL-Free (qcond+direct_hyb) 全流水线 -> VL-OPD -> v3 6 榜评测
   - 多 seed (s1) 队尾填充
"""
import sys

import enqueue as E
from enqueue import D, J, M06, M8B, MIX_LAST, MIX_META, MIX_SRC, GEO_SRC, GEO_TEST
from enqueue_p10 import LAY, OPD_COMMON, R10, R11

R14 = D + "/results/p14"
BE3 = D + "/results/bench_eval_v3"
SMOKE14_OK = R14 + "/smoke_p14.ok"
MIX_GEO = f"{MIX_META},{D}/data/geo_8b.pt"
SRC_BOTH = f"--source '{MIX_SRC}' --source '{GEO_SRC}'"
P = lambda rel: f"{D}/results/{rel}"
MEM_EV = {("q17", "tower"): 12.0, ("q17", "tower_free"): 8.0, ("q17", "proj"): 12.0,
          ("q06", "tower"): 8.0, ("q06", "tower_free"): 6.0,
          ("q8b", "tower"): 26.0, ("q8b", "tower_free"): 22.0}
EST_EV = {"q17": 16, "q06": 14, "q8b": 32}
MODEL = {"q17": None, "q06": M06, "q8b": M8B}


def J14(name, cmd, mem, prio, needs, est, timeout=None):
    return J(name, f"mkdir -p {R14} {BE3} && " + cmd, mem, prio, needs, est, 14, timeout=timeout)


def ev4(tag, name, mode, ckpt, prio, stem_json=False, extra="", extra_needs=()):
    needs = [SMOKE14_OK, ckpt] + ([ckpt[:-3] + ".json"] if stem_json else []) + list(extra_needs)
    cmd = f"$PY $D/scripts/eval_benchmarks.py --mode {mode} --ckpt {ckpt} --name {tag}_{name} --out-dir {BE3}"
    if MODEL[tag]:
        cmd += f" --model {MODEL[tag]}"
    return J14(f"ev4_{tag}_{name}", cmd + extra, MEM_EV[(tag, mode)], prio, needs, EST_EV[tag])


def smoke_p14():
    """验证 --qcond --direct-hyb + mix/geo_8b 合并训练 + Joint + v3 评测全链路（约 4 分钟）。"""
    tce = f"{R14}/smoke14_tce"
    fjt = f"{R14}/smoke14_fjt"
    cmd = (
        f"$PY $D/scripts/train_final.py --arm ce --data '{MIX_GEO}' --layers {LAY} --block 64 --query-dim 256 "
        f"--bs 4 --seq 256 --steps 20 --eval-every 20 --seed 0 --em-n 16 --limit 200 --hybrid --qcond --direct-hyb "
        f"--log {tce}.log --result-json {tce}.json && "
        f"$PY $D/scripts/train_joint.py --stage joint --data '{MIX_GEO}' {SRC_BOTH} --vrmg-ckpt {tce}_pt.pt "
        f"--front-mode rots --K 4 --arm ce --hybrid --qcond --direct-hyb --seq 256 --bs 4 --steps 15 --n-eval 24 "
        f"--workers 2 --front-ckpt {D}/results/p6/p6_rots_K4.pt --out {fjt} && "
        f"$PY $D/scripts/eval_benchmarks.py --mode tower_free --ckpt {fjt}.pt --name smoke14_free --limit 10 --out-dir {BE3} && "
        f"$PY $D/scripts/p10_gate.py eval {BE3}/smoke14_free.json 10 && touch {SMOKE14_OK}"
    )
    return J14("p14smoke_qcond", cmd, 22.0, 1, [MIX_LAST, GEO_TEST, f"{D}/results/p6/p6_rots_K4.pt"], 8, timeout=60)


def reeval_v3():
    """第 1 步：用修复禁词乌龙后的 v3 评测器重评已有核心模型（快速出齐无乌龙对比表）。"""
    jobs = []
    for name, mode, ck, sj, pr, ex_n in [
        ("proj_n21000", "proj", P("p6/p6_proj_n21000.pt"), False, 4, []),
        ("tower_hybrid_ce", "tower", P("p8/p8_hybrid_n21000_s0_pt.pt"), False, 3, []),
        ("tower_hybrid_s7875_ce", "tower", f"{R10}/p10_fill_hybrid_s7875_pt.pt", False, 3, [f"{R10}/p10_fill_hybrid_s7875.json"]),
        ("tower_mf_s7875_ce", "tower", f"{R10}/p10_fill_mf_s7875_pt.pt", False, 3, [f"{R10}/p10_fill_mf_s7875.json"]),
        ("tower_hybrid_opd", "tower", f"{R10}/p10_topd_hybrid.pt", True, 3, []),
        ("tower_mf_opd", "tower", f"{R10}/p10_topd_mf.pt", True, 3, []),
        ("free_hybrid_ce", "tower_free", P("p8/p8_hybrid_rots_joint.pt"), False, 4, []),
        ("free_hybrid_opd", "tower_free", P("p9/p9_opd_hybrid_rots.pt"), False, 3, []),
        ("free_mf_ce", "tower_free", P("p7/p7_mf_rots_joint.pt"), False, 4, []),
        ("free_mf_opd", "tower_free", P("p9/p9_opd_mf_rots.pt"), False, 3, []),
        ("rose_single_ce", "tower", P("p2/p2_ce_n21000_s0_pt.pt"), False, 5, []),
        ("vora_lora_mf", "tower_free", f"{R10}/p10_lora16_K4_joint.pt", True, 5, []),
    ]:
        jobs.append(ev4("q17", name, mode, ck, pr, sj, extra_needs=ex_n))
    for tag in ("q06", "q8b"):
        for name, mode, stem, sj in [
            ("tower_ce", "tower", f"{R11}/p11_{tag}_tower_ce_pt.pt", False),
            ("tower_opd", "tower", f"{R11}/p11_{tag}_tower_opd.pt", True),
            ("free_ce", "tower_free", f"{R11}/p11_{tag}_free_joint.pt", True),
            ("free_opd", "tower_free", f"{R11}/p11_{tag}_free_opd.pt", True),
        ]:
            ex_n = [f"{R11}/p11_{tag}_tower_ce.json"] if name == "tower_ce" else []
            jobs.append(ev4(tag, name, mode, stem, 6, sj, extra_needs=ex_n))
    return jobs


def p14_qcond_pipeline():
    """第 2 步：SpinVL-QCond（--qcond --direct-hyb，带问题查全图 cell + 合并 mix&geo_8b）全套训练、OPD 与 v3 评测。"""
    jobs = []
    # 1. 1.7B SpinVL-Tower & SpinVL-Free (qcond_dhyb 与 qcond_mf 两条主线)
    for tag, flags, est_ce in [
        ("qcdhyb", "--hybrid --qcond --direct-hyb", 55),
        ("qcmf", "--multifreq --qcond", 45),
    ]:
        tce = f"{R14}/p14_q17_tower_{tag}_ce"
        topd = f"{R14}/p14_q17_tower_{tag}_opd"
        fjt = f"{R14}/p14_q17_free_{tag}_joint"
        fopd = f"{R14}/p14_q17_free_{tag}_opd"
        jobs.append(J14(f"p14_q17_tower_{tag}_ce",
                        f"$PY $D/scripts/train_final.py --arm ce --data '{MIX_GEO}' --layers {LAY} --block 64 "
                        f"--query-dim 256 --bs 8 --seq 256 --steps 5000 --eval-every 1250 --seed 0 --em-n 100 "
                        f"--limit 23000 {flags} --test-data {D}/data/geo_8b_test.pt "
                        f"--log {tce}.log --result-json {tce}.json",
                        18.0, 2, [MIX_LAST, GEO_TEST, SMOKE14_OK], est_ce))
        jobs.append(J14(f"p14_q17_tower_{tag}_opd",
                        f"$PY $D/scripts/train_opd.py --data '{MIX_GEO}' {SRC_BOTH} --test-data {D}/data/geo_8b_test.pt "
                        f"--vrmg-ckpt {tce}_pt.pt --steps 500 --bs 4 {OPD_COMMON} {flags} --n-eval 150 --out {topd}",
                        30.0, 8, [MIX_LAST, GEO_TEST, f"{tce}_pt.pt", f"{tce}.json"], 16))
        jobs.append(J14(f"p14_q17_free_{tag}_joint",
                        f"$PY $D/scripts/train_joint.py --stage joint --data '{MIX_GEO}' {SRC_BOTH} "
                        f"--vrmg-ckpt {tce}_pt.pt --front-mode rots --K 4 --arm ce {flags} --seq 256 --bs 8 "
                        f"--steps 3000 --n-eval 600 --workers 3 --front-ckpt {D}/results/p6/p6_rots_K4.pt --out {fjt}",
                        22.0, 7, [MIX_LAST, GEO_TEST, f"{tce}_pt.pt", f"{tce}.json", f"{D}/results/p6/p6_rots_K4.pt"], 32))
        jobs.append(J14(f"p14_q17_free_{tag}_opd",
                        f"$PY $D/scripts/train_joint_opd.py --data '{MIX_GEO}' {SRC_BOTH} --joint-ckpt {fjt}.pt "
                        f"--steps 500 --bs 4 --max-new 16 --lr-vrmg 2e-5 --lr-front 1e-5 --ce-weight 0.5 "
                        f"--feat-weight 0.2 {flags} --out {fopd}",
                        32.0, 9, [MIX_LAST, GEO_TEST, f"{fjt}.pt", f"{fjt}.json"], 18))
        jobs.append(ev4("q17", f"p14_tower_{tag}_ce", "tower", f"{tce}_pt.pt", 10, extra_needs=[f"{tce}.json"]))
        jobs.append(ev4("q17", f"p14_tower_{tag}_opd", "tower", f"{topd}.pt", 10, True))
        jobs.append(ev4("q17", f"p14_free_{tag}_ce", "tower_free", f"{fjt}.pt", 11, True))
        jobs.append(ev4("q17", f"p14_free_{tag}_opd", "tower_free", f"{fopd}.pt", 11, True))

    # 2. 0.6B & 8B 跨尺寸 SpinVL-QCond (qcond + direct_hyb) 全链路
    for sz, mdl, layers, jbs, jacc, obs, oacc, mce, mj, mo, ece, ej, eo in [
        ("q06", M06, "0,4,8,12,16,20,24", 8, 1, 4, 1, 14.0, 18.0, 26.0, 32, 30, 15),
        ("q8b", M8B, "0,5,10,15,20,25,30", 2, 4, 1, 4, 38.0, 35.0, 40.0, 95, 65, 30),
    ]:
        flags = "--hybrid --qcond --direct-hyb"
        tce = f"{R14}/p14_{sz}_tower_qcdhyb_ce"
        topd = f"{R14}/p14_{sz}_tower_qcdhyb_opd"
        ffr = f"{R11}/p11_{sz}_front_rots.pt"
        fjt = f"{R14}/p14_{sz}_free_qcdhyb_joint"
        fopd = f"{R14}/p14_{sz}_free_qcdhyb_opd"
        jobs.append(J14(f"p14_{sz}_tower_qcdhyb_ce",
                        f"$PY $D/scripts/train_final.py --arm ce --data '{MIX_GEO}' --model {mdl} --layers {layers} "
                        f"--block 64 --query-dim 256 --bs 8 --seq 256 --steps 4500 --eval-every 1125 --seed 0 "
                        f"--em-n 100 --limit 23000 {flags} --test-data {D}/data/geo_8b_test.pt "
                        f"--log {tce}.log --result-json {tce}.json",
                        mce, 14, [MIX_LAST, GEO_TEST, SMOKE14_OK], ece))
        jobs.append(J14(f"p14_{sz}_tower_qcdhyb_opd",
                        f"$PY $D/scripts/train_opd.py --data '{MIX_GEO}' {SRC_BOTH} --model {mdl} "
                        f"--vrmg-ckpt {tce}_pt.pt --steps 500 --bs {obs} --accum {oacc} {OPD_COMMON} {flags} "
                        f"--n-eval 150 --out {topd}",
                        mo, 16, [MIX_LAST, GEO_TEST, f"{tce}_pt.pt", f"{tce}.json"], eo))
        jobs.append(J14(f"p14_{sz}_free_qcdhyb_joint",
                        f"$PY $D/scripts/train_joint.py --stage joint --data '{MIX_GEO}' {SRC_BOTH} --model {mdl} "
                        f"--vrmg-ckpt {tce}_pt.pt --front-mode rots --K 4 --arm ce {flags} --seq 256 "
                        f"--bs {jbs} --accum {jacc} --steps 2500 --n-eval 600 --workers 3 --front-ckpt {ffr} --out {fjt}",
                        mj, 15, [MIX_LAST, GEO_TEST, f"{tce}_pt.pt", f"{tce}.json", ffr], ej))
        jobs.append(J14(f"p14_{sz}_free_qcdhyb_opd",
                        f"$PY $D/scripts/train_joint_opd.py --data '{MIX_GEO}' {SRC_BOTH} --model {mdl} "
                        f"--joint-ckpt {fjt}.pt --steps 500 --bs {obs} --accum {oacc} --max-new 16 --lr-vrmg 2e-5 "
                        f"--lr-front 1e-5 --ce-weight 0.5 --feat-weight 0.2 {flags} --out {fopd}",
                        mo, 17, [MIX_LAST, GEO_TEST, f"{fjt}.pt", f"{fjt}.json"], eo))
        jobs.append(ev4(sz, "p14_tower_qcdhyb_ce", "tower", f"{tce}_pt.pt", 18, extra_needs=[f"{tce}.json"]))
        jobs.append(ev4(sz, "p14_tower_qcdhyb_opd", "tower", f"{topd}.pt", 18, True))
        jobs.append(ev4(sz, "p14_free_qcdhyb_ce", "tower_free", f"{fjt}.pt", 19, True))
        jobs.append(ev4(sz, "p14_free_qcdhyb_opd", "tower_free", f"{fopd}.pt", 19, True))

    # 3. 队尾满预算 7875 步 + seed=1 复验（防止 GPU 提前跑完空转）
    for seed in (1, 2):
        tce0 = f"{R14}/p14_q17_tower_qcdhyb_ce"
        fjt0 = f"{R14}/p14_q17_free_qcdhyb_joint"
        topd_s = f"{R14}/p14_q17_tower_qcdhyb_opd_s{seed}"
        fopd_s = f"{R14}/p14_q17_free_qcdhyb_opd_s{seed}"
        flags = "--hybrid --qcond --direct-hyb"
        jobs.append(J14(f"p14_q17_tower_qcdhyb_opd_s{seed}",
                        f"$PY $D/scripts/train_opd.py --data '{MIX_GEO}' {SRC_BOTH} --vrmg-ckpt {tce0}_pt.pt "
                        f"--steps 500 --bs 4 {OPD_COMMON} {flags} --seed {seed} --n-eval 150 --out {topd_s}",
                        30.0, 30 + seed, [MIX_LAST, GEO_TEST, f"{tce0}_pt.pt", f"{tce0}.json"], 16))
        jobs.append(ev4("q17", f"p14_tower_qcdhyb_opd_s{seed}", "tower", f"{topd_s}.pt", 33 + seed, True))
        jobs.append(J14(f"p14_q17_free_qcdhyb_opd_s{seed}",
                        f"$PY $D/scripts/train_joint_opd.py --data '{MIX_GEO}' {SRC_BOTH} --joint-ckpt {fjt0}.pt "
                        f"--steps 500 --bs 4 --max-new 16 --lr-vrmg 2e-5 --lr-front 1e-5 --ce-weight 0.5 "
                        f"--feat-weight 0.2 {flags} --seed {seed} --out {fopd_s}",
                        32.0, 30 + seed, [MIX_LAST, GEO_TEST, f"{fjt0}.pt", f"{fjt0}.json"], 18))
        jobs.append(ev4("q17", f"p14_free_qcdhyb_opd_s{seed}", "tower_free", f"{fopd_s}.pt", 33 + seed, True))
    return jobs


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    jobs = [smoke_p14()] + reeval_v3() + p14_qcond_pipeline()
    assert len({j["name"] for j in jobs}) == len(jobs)
    print(len(jobs), [(j["name"], j["prio"], j["mem_gb"]) for j in jobs])
    print(E.write_jobs(jobs, "--force" in sys.argv))
