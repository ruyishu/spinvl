"""phase10 / phase11 的 wave2：补丁脚本后的训练链（带冒烟闸门）+ 低优先级填充。用法：python enqueue_p10.py [--force]"""
import sys

import enqueue as E
from enqueue import D, J, M06, M8B, MIX_LAST, MIX_META, MIX_SRC, SRC3, GEO_TEST

R10, R11 = D + "/results/p10", D + "/results/p11"
SM_ACC, SM_TOPD = R10 + "/smoke_accum.ok", R10 + "/smoke_topd.ok"
CE1 = {"single": D + "/results/p2/p2_ce_n21000_s0_pt.pt", "mf": D + "/results/p7/p7_mf_n21000_s0_pt.pt",
       "hybrid": D + "/results/p8/p8_hybrid_n21000_s0_pt.pt"}
LAY = "0,4,8,12,16,20,24"
DATA = f"--data '{MIX_META}' --source '{MIX_SRC}'"
OPD_COMMON = "--max-new 16 --lr 2e-5 --ce-weight 0.5 --seq 256 --gen-temp 0.7"
PL = {"q06": dict(model=M06, fbs=16, facc=1, jbs=8, jacc=1, obs=4, oacc=1, m_front=22.0, m_joint=22.0, m_fopd=30.0,
                  m_topd=26.0, e_front=25, e_joint=40, e_fopd=25, e_topd=22, p0=15),
      "q8b": dict(model=M8B, fbs=4, facc=4, jbs=2, jacc=4, obs=1, oacc=4, m_front=36.0, m_joint=38.0, m_fopd=41.0,
                  m_topd=41.0, e_front=90, e_joint=130, e_fopd=90, e_topd=80, p0=9)}


def smokes():
    acc = (f"$PY $D/scripts/train_front.py --mode rots --K 4 {SRC3} --steps 20 --bs 4 --accum 2 --eval-every 1000 "
           f"--workers 2 --out {R10}/smoke_front && "
           f"$PY $D/scripts/train_joint.py --stage joint {DATA} --vrmg-ckpt {CE1['mf']} --front-mode rots --K 4 "
           f"--arm ce --multifreq --seq 256 --bs 4 --accum 2 --steps 20 --n-eval 24 --workers 2 "
           f"--front-ckpt {R10}/smoke_front.pt --out {R10}/smoke_joint && "
           f"$PY $D/scripts/p10_gate.py json {R10}/smoke_joint.json eval.front.correct eval.front.tf_first eval.teacher.correct && "
           f"$PY $D/scripts/train_joint_opd.py {DATA} --joint-ckpt {R10}/smoke_joint.pt --steps 20 --bs 2 --accum 2 "
           f"--max-new 16 --n-eval 24 --n-acc 24 --n-em 16 --out {R10}/smoke_jopd && "
           f"$PY $D/scripts/p10_gate.py json {R10}/smoke_jopd.json init_eval.correct eval.correct eval.tf_first && "
           f"touch {SM_ACC}")
    topd = (f"$PY $D/scripts/train_opd.py {DATA} --vrmg-ckpt {CE1['hybrid']} --steps 20 --bs 4 --accum 2 {OPD_COMMON} "
            f"--n-eval 24 --out {R10}/smoke_topd && $PY $D/scripts/p10_gate.py opd {R10}/smoke_topd.json && touch {SM_TOPD}")
    return [J("p10smoke_accum", acc, 34.0, 1, [MIX_LAST, GEO_TEST, CE1["mf"]], 20, 2, timeout=90),
            J("p10smoke_topd", topd, 34.0, 1, [MIX_LAST, CE1["hybrid"]], 15, 2, timeout=60)]


def p10_train():
    jobs = []
    for tag in ("single", "mf", "hybrid"):
        jobs.append(J(f"p10_topd_{tag}", f"$PY $D/scripts/train_opd.py {DATA} --vrmg-ckpt {CE1[tag]} --steps 500 --bs 4 "
                      f"{OPD_COMMON} --n-eval 150 --out {R10}/p10_topd_{tag}",
                      34.0, 12, [MIX_LAST, SM_TOPD, CE1[tag]], 25, 2))
    jobs.append(J("p10_lora16_K4_joint",
                  f"$PY $D/scripts/train_joint.py --stage joint {DATA} --vrmg-ckpt {CE1['mf']} --front-mode lora --K 4 "
                  f"--lora-rank 16 --arm ce --multifreq --seq 256 --bs 8 --steps 2500 --n-eval 600 --workers 3 "
                  f"--front-ckpt {R10}/p10_lora16_K4.pt --out {R10}/p10_lora16_K4_joint",
                  28.0, 14, [MIX_LAST, SM_ACC, CE1["mf"], R10 + "/p10_lora16_K4.pt", R10 + "/p10_lora16_K4.json"], 40, 2))
    return jobs


def p10_fill():
    """低优先级尾部填充：等步数(7875)的 mf/hybrid CE、OPD 第二个 seed、单频 Free+OPD 消融。"""
    jobs = []
    for tag, flag, est in (("mf", "--multifreq", 60), ("hybrid", "--hybrid", 110)):
        jobs.append(J(f"p10_fill_{tag}_s7875",
                      f"$PY $D/scripts/train_final.py --arm ce --data '{MIX_META}' --layers {LAY} --block 64 --query-dim 256 "
                      f"--bs 8 --seq 256 --steps 7875 --eval-every 1968 --seed 0 --em-n 100 --limit 21000 {flag} "
                      f"--log {R10}/p10_fill_{tag}_s7875.log --result-json {R10}/p10_fill_{tag}_s7875.json",
                      18.0, 60, [MIX_LAST], est, 2))
    for tag in ("mf", "hybrid"):
        jobs.append(J(f"p10_topd_{tag}_s1", f"$PY $D/scripts/train_opd.py {DATA} --vrmg-ckpt {CE1[tag]} --steps 500 --bs 4 "
                      f"{OPD_COMMON} --n-eval 150 --seed 1 --out {R10}/p10_topd_{tag}_s1",
                      34.0, 61, [MIX_LAST, SM_TOPD, CE1[tag]], 25, 2))
    for tag, jc, seed in (("mf", "p7/p7_mf_rots_joint", 1), ("hybrid", "p8/p8_hybrid_rots_joint", 1),
                          ("single", "p6/p6_rots_K4_joint", 0)):
        jobs.append(J(f"p10_fopd_{tag}_s{seed}",
                      f"$PY $D/scripts/train_joint_opd.py {DATA} --joint-ckpt {D}/results/{jc}.pt --steps 500 --bs 4 "
                      f"--max-new 16 --lr-vrmg 2e-5 --lr-front 1e-5 --ce-weight 0.5 --feat-weight 0.2 --seed {seed} "
                      f"--out {R10}/p10_fopd_{tag}_s{seed}",
                      34.0, 62, [MIX_LAST, SM_ACC, f"{D}/results/{jc}.pt"], 25, 2))
    return jobs


def p11_train():
    jobs = []
    for tag, s in PL.items():
        tce, ffr, fjt = f"{R11}/p11_{tag}_tower_ce", f"{R11}/p11_{tag}_front_rots", f"{R11}/p11_{tag}_free_joint"
        p0, mdl, need_t = s["p0"], s["model"], [f"{tce}_pt.pt", f"{tce}.json"]
        if tag == "q8b":  # 0.6B 的前端热身在 wave1 已挂；8B 需要 --accum，等冒烟
            jobs.append(J(f"p11_{tag}_front_rots",
                          f"$PY $D/scripts/train_front.py --mode rots --K 4 --model {mdl} {SRC3} --steps 2000 "
                          f"--bs {s['fbs']} --accum {s['facc']} --eval-every 1000 --workers 3 --out {ffr}",
                          s["m_front"], p0, [MIX_LAST, GEO_TEST, SM_ACC], s["e_front"], 2))
        jobs.append(J(f"p11_{tag}_free_joint",
                      f"$PY $D/scripts/train_joint.py --stage joint {DATA} --model {mdl} --vrmg-ckpt {tce}_pt.pt "
                      f"--front-mode rots --K 4 --arm ce --multifreq --seq 256 --bs {s['jbs']} --accum {s['jacc']} "
                      f"--steps 2500 --n-eval 600 --workers 3 --front-ckpt {ffr}.pt --out {fjt}",
                      s["m_joint"], p0 + 1, [MIX_LAST, SM_ACC] + need_t + [ffr + ".pt", ffr + ".json"], s["e_joint"], 2))
        jobs.append(J(f"p11_{tag}_tower_opd",
                      f"$PY $D/scripts/train_opd.py {DATA} --model {mdl} --vrmg-ckpt {tce}_pt.pt --steps 500 "
                      f"--bs {s['obs']} --accum {s['oacc']} {OPD_COMMON} --n-eval 150 --out {R11}/p11_{tag}_tower_opd",
                      s["m_topd"], p0 + 2, [MIX_LAST, SM_TOPD, SM_ACC] + need_t, s["e_topd"], 2))
        jobs.append(J(f"p11_{tag}_free_opd",
                      f"$PY $D/scripts/train_joint_opd.py {DATA} --model {mdl} --joint-ckpt {fjt}.pt --steps 500 "
                      f"--bs {s['obs']} --accum {s['oacc']} --max-new 16 --lr-vrmg 2e-5 --lr-front 1e-5 "
                      f"--ce-weight 0.5 --feat-weight 0.2 --out {R11}/p11_{tag}_free_opd",
                      s["m_fopd"], p0 + 4, [MIX_LAST, SM_ACC, fjt + ".pt", fjt + ".json"], s["e_fopd"], 2))
    return jobs


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    jobs = smokes() + p10_train() + p10_fill() + p11_train()
    assert len({j["name"] for j in jobs}) == len(jobs)
    print([(j["name"], j["prio"], j["mem_gb"]) for j in jobs])
    print(E.write_jobs(jobs, "--force" in sys.argv))
