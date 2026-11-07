"""phase12：三步升级全量入队
1) 零重训 v2 公开榜重评（输出到 results/bench_eval_v2/，覆盖 1.7B/0.6B/8B 全套模型 + 推理端 prompt_alpha/hyb_bound 对照）；
2) 第 2+3 步联合训练（--prompt-alpha 0.2 保读题 + --hyb-bound 0.5 防长序列激活爆炸）：
   - 1.7B SpinVL-Tower (mf & hybrid) CE -> VL-OPD -> v2 6 榜评测
   - 1.7B SpinVL-Free  (mf & hybrid) Joint CE -> VL-OPD -> v2 6 榜评测
   - 0.6B & 8B SpinVL-Free (mf) Joint CE -> VL-OPD -> v2 6 榜评测
用法：python enqueue_p12.py [--force]
"""
import sys

import enqueue as E
from enqueue import D, J, M06, M8B, MIX_LAST, MIX_META, MIX_SRC
from enqueue_p10 import CE1, DATA, LAY, OPD_COMMON, R10, R11

R12 = D + "/results/p12"
BE2 = D + "/results/bench_eval_v2"
SMOKE2_OK = BE2 + "/smoke_v2.ok"
P = lambda rel: f"{D}/results/{rel}"
MEM_EV = {("q17", "tower"): 12.0, ("q17", "tower_free"): 8.0, ("q17", "proj"): 12.0,
          ("q06", "tower"): 8.0, ("q06", "tower_free"): 6.0,
          ("q8b", "tower"): 26.0, ("q8b", "tower_free"): 22.0}
EST_EV = {"q17": 18, "q06": 15, "q8b": 35}
MODEL = {"q17": None, "q06": M06, "q8b": M8B}


def J12(name, cmd, mem, prio, needs, est, timeout=None):
    j = J(name, f"mkdir -p {R12} {BE2} && " + cmd, mem, prio, needs, est, 12, timeout=timeout)
    return j


def ev2(tag, name, mode, ckpt, prio, stem_json=False, extra="", extra_needs=()):
    needs = [SMOKE2_OK, ckpt] + ([ckpt[:-3] + ".json"] if stem_json else []) + list(extra_needs)
    cmd = f"$PY $D/scripts/eval_benchmarks.py --mode {mode} --ckpt {ckpt} --name {tag}_{name} --out-dir {BE2}"
    if MODEL[tag]:
        cmd += f" --model {MODEL[tag]}"
    return J12(f"ev2_{tag}_{name}", cmd + extra, MEM_EV[(tag, mode)], prio, needs, EST_EV[tag])


def smoke_v2():
    """v2 评测 + 新训练参数冒烟闸门（3 分钟跑完）。"""
    runs = [
        ("smoke2_tower", "tower", f"{R10}/p10_topd_hybrid.pt", "--prompt-alpha 0.2 --hyb-bound 0.5"),
        ("smoke2_free", "tower_free", P("p9/p9_opd_hybrid_rots.pt"), "--prompt-alpha 0.2 --hyb-bound 0.5"),
        ("smoke2_proj", "proj", P("p6/p6_proj_n21000.pt"), ""),
    ]
    cmd = " && ".join(
        f"$PY $D/scripts/eval_benchmarks.py --mode {m} --ckpt {c} --name {n} --limit 10 --out-dir {BE2} {ex} && "
        f"$PY $D/scripts/p10_gate.py eval {BE2}/{n}.json 10"
        for n, m, c, ex in runs
    ) + f" && touch {SMOKE2_OK}"
    return J12("p12smoke_v2", cmd, 14.0, 1, [c for _, _, c, _ in runs], 10, timeout=60)


def reeval_existing():
    """第 1 步 + 第 2/3 步推理端门控：把全部已有核心模型用 v2 协议重评一遍（高优先级，快速出齐对比表）。"""
    jobs = []
    # 1.7B 主表全套（原配置 v2 重评）
    for name, mode, ck, sj, pr in [
        ("proj_n21000", "proj", P("p6/p6_proj_n21000.pt"), False, 3),
        ("tower_hybrid_opd", "tower", f"{R10}/p10_topd_hybrid.pt", True, 3),
        ("tower_mf_opd", "tower", f"{R10}/p10_topd_mf.pt", True, 3),
        ("tower_single_opd", "tower", f"{R10}/p10_topd_single.pt", True, 4),
        ("free_hybrid_opd", "tower_free", P("p9/p9_opd_hybrid_rots.pt"), False, 3),
        ("free_mf_opd", "tower_free", P("p9/p9_opd_mf_rots.pt"), False, 3),
        ("free_single_opd_s0", "tower_free", f"{R10}/p10_fopd_single_s0.pt", True, 4),
        ("rose_single_ce", "tower", P("p2/p2_ce_n21000_s0_pt.pt"), False, 5),
        ("tower_mf_ce", "tower", P("p7/p7_mf_n21000_s0_pt.pt"), False, 5),
        ("tower_hybrid_ce", "tower", P("p8/p8_hybrid_n21000_s0_pt.pt"), False, 5),
        ("free_single_ce", "tower_free", P("p6/p6_rots_K4_joint.pt"), False, 5),
        ("free_mf_ce", "tower_free", P("p7/p7_mf_rots_joint.pt"), False, 5),
        ("free_hybrid_ce", "tower_free", P("p8/p8_hybrid_rots_joint.pt"), False, 5),
        ("vora_lora_mf", "tower_free", f"{R10}/p10_lora16_K4_joint.pt", True, 5),
    ]:
        jobs.append(ev2("q17", name, mode, ck, pr, sj))

    # 1.7B 推理端直接开 prompt_alpha + hyb_bound 的消融对照
    for name, mode, ck, ex in [
        ("tower_hybrid_opd_pa03_hb05", "tower", f"{R10}/p10_topd_hybrid.pt", " --prompt-alpha 0.3 --hyb-bound 0.5"),
        ("free_hybrid_opd_pa03_hb05", "tower_free", P("p9/p9_opd_hybrid_rots.pt"), " --prompt-alpha 0.3 --hyb-bound 0.5"),
        ("tower_mf_opd_pa03", "tower", f"{R10}/p10_topd_mf.pt", " --prompt-alpha 0.3"),
        ("free_mf_opd_pa03", "tower_free", P("p9/p9_opd_mf_rots.pt"), " --prompt-alpha 0.3"),
    ]:
        jobs.append(ev2("q17", name, mode, ck, 6, False, ex))

    # 0.6B & 8B 跨尺寸 v2 重评
    for tag in ("q06", "q8b"):
        for name, mode, stem, sj, pr in [
            ("tower_ce", "tower", f"{R11}/p11_{tag}_tower_ce_pt.pt", False, 7),
            ("tower_opd", "tower", f"{R11}/p11_{tag}_tower_opd.pt", True, 7),
            ("free_ce", "tower_free", f"{R11}/p11_{tag}_free_joint.pt", True, 8),
            ("free_opd", "tower_free", f"{R11}/p11_{tag}_free_opd.pt", True, 8),
        ]:
            ex_n = [f"{R11}/p11_{tag}_tower_ce.json"] if name == "tower_ce" else []
            jobs.append(ev2(tag, name, mode, stem, pr, sj, extra_needs=ex_n))
    return jobs


def p12_train_and_eval():
    """第 2+3 步端到端重训（--prompt-alpha 0.2 保读题 + --hyb-bound 0.5 防长序列爆炸）+ VL-OPD + v2 评测。"""
    jobs = []
    # 1. 1.7B SpinVL-Tower (mf & hybrid) 带 prompt_alpha=0.2 + hyb_bound=0.5 训练 -> OPD -> 评测
    for tag, flag, hb, est_ce in (("mf", "--multifreq", "0.0", 40), ("hybrid", "--hybrid", "0.5", 55)):
        tce = f"{R12}/p12_tower_{tag}_pa02_ce"
        topd = f"{R12}/p12_tower_{tag}_pa02_opd"
        jobs.append(J12(f"p12_tower_{tag}_pa02_ce",
                        f"$PY $D/scripts/train_final.py --arm ce --data '{MIX_META}' --layers {LAY} --block 64 "
                        f"--query-dim 256 --bs 8 --seq 256 --steps 4500 --eval-every 1125 --seed 0 --em-n 100 "
                        f"--limit 21000 {flag} --prompt-alpha 0.2 --hyb-bound {hb} "
                        f"--log {tce}.log --result-json {tce}.json",
                        18.0, 12, [MIX_LAST, SMOKE2_OK], est_ce))
        jobs.append(J12(f"p12_tower_{tag}_pa02_opd",
                        f"$PY $D/scripts/train_opd.py {DATA} --vrmg-ckpt {tce}_pt.pt --steps 500 --bs 4 "
                        f"{OPD_COMMON} --prompt-alpha 0.2 --hyb-bound {hb} --n-eval 150 --out {topd}",
                        30.0, 14, [MIX_LAST, f"{tce}_pt.pt", f"{tce}.json"], 15))
        jobs.append(ev2("q17", f"p12_tower_{tag}_pa02_ce", "tower", f"{tce}_pt.pt", 15, extra_needs=[f"{tce}.json"]))
        jobs.append(ev2("q17", f"p12_tower_{tag}_pa02_opd", "tower", f"{topd}.pt", 15, True))

        # 2. 1.7B SpinVL-Free (rots_K4 + mf/hybrid + prompt_alpha=0.2 + hyb_bound) 端到端联调 -> 无塔 VL-OPD -> 评测
        fjt = f"{R12}/p12_free_{tag}_pa02_joint"
        fopd = f"{R12}/p12_free_{tag}_pa02_opd"
        jobs.append(J12(f"p12_free_{tag}_pa02_joint",
                        f"$PY $D/scripts/train_joint.py --stage joint {DATA} --vrmg-ckpt {CE1[tag]} "
                        f"--front-mode rots --K 4 --arm ce {flag} --prompt-alpha 0.2 --hyb-bound {hb} "
                        f"--seq 256 --bs 8 --steps 2500 --n-eval 600 --workers 3 "
                        f"--front-ckpt {D}/results/p6/p6_rots_K4.pt --out {fjt}",
                        22.0, 13, [MIX_LAST, SMOKE2_OK, CE1[tag], f"{D}/results/p6/p6_rots_K4.pt"], 25))
        jobs.append(J12(f"p12_free_{tag}_pa02_opd",
                        f"$PY $D/scripts/train_joint_opd.py {DATA} --joint-ckpt {fjt}.pt --steps 500 --bs 4 "
                        f"--max-new 16 --lr-vrmg 2e-5 --lr-front 1e-5 --ce-weight 0.5 --feat-weight 0.2 "
                        f"--prompt-alpha 0.2 --hyb-bound {hb} --out {fopd}",
                        32.0, 16, [MIX_LAST, f"{fjt}.pt", f"{fjt}.json"], 18))
        jobs.append(ev2("q17", f"p12_free_{tag}_pa02_ce", "tower_free", f"{fjt}.pt", 17, True))
        jobs.append(ev2("q17", f"p12_free_{tag}_pa02_opd", "tower_free", f"{fopd}.pt", 17, True))

    # 3. 0.6B & 8B 跨尺寸 pa02 无塔联调 + VL-OPD + 评测
    for tag, mdl, jbs, jacc, obs, oacc, mj, mo, ej, eo in [
        ("q06", M06, 8, 1, 4, 1, 18.0, 26.0, 28, 15),
        ("q8b", M8B, 2, 4, 1, 4, 34.0, 40.0, 65, 30),
    ]:
        tce = f"{R11}/p11_{tag}_tower_ce_pt.pt"
        ffr = f"{R11}/p11_{tag}_front_rots.pt"
        fjt = f"{R12}/p12_{tag}_free_pa02_joint"
        fopd = f"{R12}/p12_{tag}_free_pa02_opd"
        jobs.append(J12(f"p12_{tag}_free_pa02_joint",
                        f"$PY $D/scripts/train_joint.py --stage joint {DATA} --model {mdl} --vrmg-ckpt {tce} "
                        f"--front-mode rots --K 4 --arm ce --multifreq --prompt-alpha 0.2 --seq 256 "
                        f"--bs {jbs} --accum {jacc} --steps 2500 --n-eval 600 --workers 3 --front-ckpt {ffr} --out {fjt}",
                        mj, 20, [MIX_LAST, SMOKE2_OK, tce, ffr], ej))
        jobs.append(J12(f"p12_{tag}_free_pa02_opd",
                        f"$PY $D/scripts/train_joint_opd.py {DATA} --model {mdl} --joint-ckpt {fjt}.pt --steps 500 "
                        f"--bs {obs} --accum {oacc} --max-new 16 --lr-vrmg 2e-5 --lr-front 1e-5 "
                        f"--ce-weight 0.5 --feat-weight 0.2 --prompt-alpha 0.2 --out {fopd}",
                        mo, 22, [MIX_LAST, f"{fjt}.pt", f"{fjt}.json"], eo))
        jobs.append(ev2(tag, "p12_free_pa02_ce", "tower_free", f"{fjt}.pt", 23, True))
        jobs.append(ev2(tag, "p12_free_pa02_opd", "tower_free", f"{fopd}.pt", 23, True))
    return jobs


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    jobs = [smoke_v2()] + reeval_existing() + p12_train_and_eval()
    assert len({j["name"] for j in jobs}) == len(jobs)
    print(len(jobs), [(j["name"], j["prio"], j["mem_gb"]) for j in jobs])
    print(E.write_jobs(jobs, "--force" in sys.argv))
