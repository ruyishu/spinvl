"""phase15（选择 B：补齐同类型训练数据 + 回归 P8/P9 最干净的 SpinVL 架构）：
1) p15_prep_open_supp：从共享盘 Visual-CoT 中解出 9000 条同类型开放短答题（docvqa + infographicsvqa + textvqa）并抽好 8B 视觉塔特征分片；
2) p15smoke_supp：验证 mix + open_supp + geo_8b 三合一训练、Joint 与 6 榜评测全链路无误；
3) 在三合一数据集上用原版干净架构（--hybrid 与 --multifreq，不开 --qcond）训练：
   - 1.7B LLaVA-Proj 同数据公平基线 + 6 榜评测
   - 1.7B SpinVL-Tower (hybrid & mf) CE -> VL-OPD + 6 榜评测
   - 1.7B SpinVL-Free  (rots_K4 + hybrid & mf) Joint -> VL-OPD + 6 榜评测
   - 0.6B & 8B 跨尺寸 SpinVL-Tower & SpinVL-Free (hybrid) 全流水线 + 6 榜评测
"""
import sys

import enqueue as E
from enqueue import D, J, M06, M8B, MIX_LAST, MIX_META, MIX_SRC, GEO_SRC, GEO_TEST
from enqueue_p10 import LAY, OPD_COMMON, R11

R15 = D + "/results/p15"
BE3 = D + "/results/bench_eval_v3"
SUPP_DIR = D + "/data/open_supp"
SUPP_OK = SUPP_DIR + "/open_supp.ok"
SMOKE15_OK = R15 + "/smoke_p15.ok"
SUPP_META = SUPP_DIR + "/shard*.meta.pt"
SUPP_SRC = f"{SUPP_META}|{SUPP_DIR}/manifest.jsonl|{SUPP_DIR}"
DATA_ALL = f"{MIX_META},{SUPP_META},{D}/data/geo_8b.pt"
SRC_ALL = f"--source '{MIX_SRC}' --source '{SUPP_SRC}' --source '{GEO_SRC}'"

MEM_EV = {("q17", "tower"): 12.0, ("q17", "tower_free"): 8.0, ("q17", "proj"): 12.0,
          ("q06", "tower"): 8.0, ("q06", "tower_free"): 6.0,
          ("q8b", "tower"): 26.0, ("q8b", "tower_free"): 22.0}
EST_EV = {"q17": 16, "q06": 14, "q8b": 32}
MODEL = {"q17": None, "q06": M06, "q8b": M8B}


def J15(name, cmd, mem, prio, needs, est, timeout=None):
    return J(name, f"mkdir -p {R15} {BE3} {SUPP_DIR} && " + cmd, mem, prio, needs, est, 15, timeout=timeout)


def ev5(tag, name, mode, ckpt, prio, stem_json=False, extra_needs=()):
    needs = [SMOKE15_OK, ckpt] + ([ckpt[:-3] + ".json"] if stem_json else []) + list(extra_needs)
    cmd = f"$PY $D/scripts/eval_benchmarks.py --mode {mode} --ckpt {ckpt} --name {tag}_p15_{name} --out-dir {BE3}"
    if MODEL[tag]:
        cmd += f" --model {MODEL[tag]}"
    return J15(f"ev5_{tag}_{name}", cmd, MEM_EV[(tag, mode)], prio, needs, EST_EV[tag])


def prep_and_smoke():
    prep = J15("p15_prep_open_supp",
               f"$PY $D/scripts/build_open_supp.py --out-dir {SUPP_DIR} --per-ds 3000",
               10.0, 1, [MIX_LAST], 15, timeout=90)
    tce = f"{R15}/smoke15_tce"
    fjt = f"{R15}/smoke15_fjt"
    smoke_cmd = (
        f"$PY $D/scripts/train_final.py --arm ce --data '{DATA_ALL}' --layers {LAY} --block 64 --query-dim 256 "
        f"--bs 4 --seq 256 --steps 20 --eval-every 20 --seed 0 --em-n 16 --limit 300 --hybrid "
        f"--log {tce}.log --result-json {tce}.json && "
        f"$PY $D/scripts/train_joint.py --stage joint --data '{DATA_ALL}' {SRC_ALL} --vrmg-ckpt {tce}_pt.pt "
        f"--front-mode rots --K 4 --arm ce --hybrid --seq 256 --bs 4 --steps 15 --n-eval 24 "
        f"--workers 2 --front-ckpt {D}/results/p6/p6_rots_K4.pt --out {fjt} && "
        f"$PY $D/scripts/eval_benchmarks.py --mode tower_free --ckpt {fjt}.pt --name smoke15_free --limit 10 --out-dir {BE3} && "
        f"$PY $D/scripts/p10_gate.py eval {BE3}/smoke15_free.json 10 && touch {SMOKE15_OK}"
    )
    smoke = J15("p15smoke_supp", smoke_cmd, 22.0, 2, [SUPP_OK, MIX_LAST, GEO_TEST, f"{D}/results/p6/p6_rots_K4.pt"], 8, timeout=60)
    return [prep, smoke]


def p15_pipeline():
    jobs = []
    # 0. 同数据 LLaVA-Proj 公平对照基线
    p_proj = f"{R15}/p15_q17_proj"
    jobs.append(J15("p15_q17_proj",
                    f"$PY $D/scripts/train_proj.py --data '{DATA_ALL}' {SRC_ALL} --limit 28000 --steps 6000 "
                    f"--bs 8 --lr 5e-4 --out {p_proj}",
                    32.0, 6, [SMOKE15_OK, SUPP_OK, MIX_LAST, GEO_TEST], 75))
    jobs.append(ev5("q17", "proj", "proj", f"{p_proj}.pt", 8, True))

    # 1. 1.7B SpinVL-Tower & SpinVL-Free（原版干净架构：hybrid 与 mf）
    for tag, flag, est_ce in [("hybrid", "--hybrid", 55), ("mf", "--multifreq", 48)]:
        tce = f"{R15}/p15_q17_tower_{tag}_ce"
        topd = f"{R15}/p15_q17_tower_{tag}_opd"
        fjt = f"{R15}/p15_q17_free_{tag}_joint"
        fopd = f"{R15}/p15_q17_free_{tag}_opd"
        jobs.append(J15(f"p15_q17_tower_{tag}_ce",
                        f"$PY $D/scripts/train_final.py --arm ce --data '{DATA_ALL}' --layers {LAY} --block 64 "
                        f"--query-dim 256 --bs 8 --seq 256 --steps 6000 --eval-every 1500 --seed 0 --em-n 100 "
                        f"--limit 28000 {flag} --test-data {D}/data/geo_8b_test.pt "
                        f"--log {tce}.log --result-json {tce}.json",
                        18.0, 3, [SMOKE15_OK, SUPP_OK, MIX_LAST, GEO_TEST], est_ce))
        jobs.append(J15(f"p15_q17_tower_{tag}_opd",
                        f"$PY $D/scripts/train_opd.py --data '{DATA_ALL}' {SRC_ALL} --test-data {D}/data/geo_8b_test.pt "
                        f"--vrmg-ckpt {tce}_pt.pt --steps 500 --bs 4 {OPD_COMMON} {flag} --n-eval 150 --out {topd}",
                        30.0, 5, [SMOKE15_OK, SUPP_OK, f"{tce}_pt.pt", f"{tce}.json"], 16))
        jobs.append(J15(f"p15_q17_free_{tag}_joint",
                        f"$PY $D/scripts/train_joint.py --stage joint --data '{DATA_ALL}' {SRC_ALL} "
                        f"--vrmg-ckpt {tce}_pt.pt --front-mode rots --K 4 --arm ce {flag} --seq 256 --bs 8 "
                        f"--steps 3200 --n-eval 600 --workers 3 --front-ckpt {D}/results/p6/p6_rots_K4.pt --out {fjt}",
                        32.0, 4, [SMOKE15_OK, SUPP_OK, f"{tce}_pt.pt", f"{tce}.json", f"{D}/results/p6/p6_rots_K4.pt"], 35))
        jobs.append(J15(f"p15_q17_free_{tag}_opd",
                        f"$PY $D/scripts/train_joint_opd.py --data '{DATA_ALL}' {SRC_ALL} --joint-ckpt {fjt}.pt "
                        f"--steps 500 --bs 4 --max-new 16 --lr-vrmg 2e-5 --lr-front 1e-5 --ce-weight 0.5 "
                        f"--feat-weight 0.2 {flag} --out {fopd}",
                        32.0, 6, [SMOKE15_OK, SUPP_OK, f"{fjt}.pt", f"{fjt}.json"], 18))
        jobs.append(ev5("q17", f"tower_{tag}_ce", "tower", f"{tce}_pt.pt", 7, extra_needs=[f"{tce}.json"]))
        jobs.append(ev5("q17", f"tower_{tag}_opd", "tower", f"{topd}.pt", 7, True))
        jobs.append(ev5("q17", f"free_{tag}_ce", "tower_free", f"{fjt}.pt", 7, True))
        jobs.append(ev5("q17", f"free_{tag}_opd", "tower_free", f"{fopd}.pt", 7, True))

    # 2. 0.6B & 8B 跨尺寸（原版干净架构 --hybrid）
    for sz, mdl, layers, jbs, jacc, obs, oacc, mce, mj, mo, ece, ej, eo in [
        ("q06", M06, "0,4,8,12,16,20,24", 8, 1, 4, 1, 14.0, 26.0, 26.0, 35, 32, 15),
        ("q8b", M8B, "0,5,10,15,20,25,30", 2, 4, 1, 4, 38.0, 38.0, 40.0, 95, 65, 30),
    ]:
        flag = "--hybrid"
        tce = f"{R15}/p15_{sz}_tower_hybrid_ce"
        topd = f"{R15}/p15_{sz}_tower_hybrid_opd"
        ffr = f"{R11}/p11_{sz}_front_rots.pt"
        fjt = f"{R15}/p15_{sz}_free_hybrid_joint"
        fopd = f"{R15}/p15_{sz}_free_hybrid_opd"
        jobs.append(J15(f"p15_{sz}_tower_hybrid_ce",
                        f"$PY $D/scripts/train_final.py --arm ce --data '{DATA_ALL}' --model {mdl} --layers {layers} "
                        f"--block 64 --query-dim 256 --bs 8 --seq 256 --steps 5000 --eval-every 1250 --seed 0 "
                        f"--em-n 100 --limit 28000 {flag} --test-data {D}/data/geo_8b_test.pt "
                        f"--log {tce}.log --result-json {tce}.json",
                        mce, 9, [SMOKE15_OK, SUPP_OK, MIX_LAST, GEO_TEST], ece))
        jobs.append(J15(f"p15_{sz}_tower_hybrid_opd",
                        f"$PY $D/scripts/train_opd.py --data '{DATA_ALL}' {SRC_ALL} --model {mdl} "
                        f"--vrmg-ckpt {tce}_pt.pt --steps 500 --bs {obs} --accum {oacc} {OPD_COMMON} {flag} "
                        f"--n-eval 150 --out {topd}",
                        mo, 11, [SMOKE15_OK, SUPP_OK, f"{tce}_pt.pt", f"{tce}.json"], eo))
        jobs.append(J15(f"p15_{sz}_free_hybrid_joint",
                        f"$PY $D/scripts/train_joint.py --stage joint --data '{DATA_ALL}' {SRC_ALL} --model {mdl} "
                        f"--vrmg-ckpt {tce}_pt.pt --front-mode rots --K 4 --arm ce {flag} --seq 256 "
                        f"--bs {jbs} --accum {jacc} --steps 2800 --n-eval 600 --workers 3 --front-ckpt {ffr} --out {fjt}",
                        mj, 10, [SMOKE15_OK, SUPP_OK, f"{tce}_pt.pt", f"{tce}.json", ffr], ej))
        jobs.append(J15(f"p15_{sz}_free_hybrid_opd",
                        f"$PY $D/scripts/train_joint_opd.py --data '{DATA_ALL}' {SRC_ALL} --model {mdl} "
                        f"--joint-ckpt {fjt}.pt --steps 500 --bs {obs} --accum {oacc} --max-new 16 --lr-vrmg 2e-5 "
                        f"--lr-front 1e-5 --ce-weight 0.5 --feat-weight 0.2 {flag} --out {fopd}",
                        mo, 12, [SMOKE15_OK, SUPP_OK, f"{fjt}.pt", f"{fjt}.json"], eo))
        jobs.append(ev5(sz, "tower_hybrid_ce", "tower", f"{tce}_pt.pt", 13, extra_needs=[f"{tce}.json"]))
        jobs.append(ev5(sz, "tower_hybrid_opd", "tower", f"{topd}.pt", 13, True))
        jobs.append(ev5(sz, "free_hybrid_ce", "tower_free", f"{fjt}.pt", 13, True))
        jobs.append(ev5(sz, "free_hybrid_opd", "tower_free", f"{fopd}.pt", 13, True))
    return jobs


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    jobs = prep_and_smoke() + p15_pipeline()
    assert len({j["name"] for j in jobs}) == len(jobs)
    for j in jobs:
        assert not any("*" in n for n in j["needs"]), f"wildcard in needs: {j['name']}"
    print(len(jobs), [(j["name"], j["prio"], j["mem_gb"]) for j in jobs])
    print(E.write_jobs(jobs, "--force" in sys.argv))
