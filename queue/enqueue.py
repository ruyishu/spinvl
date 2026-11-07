"""生成任务 json 并写入共享盘队列。用法：python enqueue.py <phase> [--force]"""

import json
import sys

import ops

D = "/data/ibu-bi-ai-share-disk/zk/project/rose-distill"
MEM = {"ce": 13.5, "kd": 15.5, "rel": 16.0}


def train_job(name, arm, data, outdir, seed=0, steps=2000, layers="0,4,8,12,16,20,24", block=64, T=2.0,
              qdim=256, prio=10, est=16, test="", extra="", mem=None, needs=None, kd_w=0.5, rel_w=0.1):
    cmd = (f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $D/results/{outdir} && $PY $D/scripts/train_final.py --arm {arm} --data '{data}' "
           f"--layers {layers} --block {block} --query-dim {qdim} --bs 8 --seq 160 --steps {steps} "
           f"--eval-every {max(steps // 4, 100)} --seed {seed} --kd-temp {T} --kd-weight {kd_w} --rel-weight {rel_w} "
           f"--log $D/results/{outdir}/{name}.log --result-json $D/results/{outdir}/{name}.json")
    if test:
        cmd += f" --test-data {test}"
    cmd += f" {extra}"
    n = list(needs or []) + [data.replace("$D", D)] + ([test.replace("$D", D)] if test else [])
    return {"name": name, "cmd": cmd, "mem_gb": mem or MEM[arm], "prio": prio, "needs": n,
            "est_min": est, "timeout_min": max(est * 4, 60)}


def phase1():
    jobs = []
    data, test = "$D/data/geo_8b.pt", "$D/data/geo_8b_test.pt"
    for seed in (0, 1, 2):
        for arm in ("ce", "kd", "rel"):
            jobs.append(train_job(f"p1_{arm}_s{seed}", arm, data, "p1", seed=seed, test=test, prio=10 + seed))
    # 超参消融（rel 臂，seed 0）
    for T in (1.0, 4.0):
        jobs.append(train_job(f"p1_abl_T{int(T)}", "rel", data, "p1", T=T, test=test, prio=30))
    jobs.append(train_job("p1_abl_L4", "rel", data, "p1", layers="0,7,14,21", test=test, prio=30, est=12))
    jobs.append(train_job("p1_abl_L14", "rel", data, "p1", layers="0,2,4,6,8,10,12,14,16,18,20,22,24,26",
                          test=test, prio=30, est=28, mem=20.0))
    jobs.append(train_job("p1_abl_B32", "rel", data, "p1", block=32, test=test, prio=30))
    jobs.append(train_job("p1_abl_B128", "rel", data, "p1", block=128, test=test, prio=30))
    return jobs


def prep_mix():
    """8B 老师抽取 mix 数据（约 2.4 万题，分片可续跑）。声明 27GB，使老机器上「剩一个小任务时」即可启动。"""
    return [{"name": "prep_mix8b",
             "cmd": "$PY $D/scripts/prepare_big.py --manifest $D/data/mix/manifest.jsonl --img-root $D/data/mix "
                    "--out-prefix $D/data/mix/shard --teacher Qwen3-VL-8B-Instruct --shard-size 2048 --batch 8",
             "mem_gb": 27.0, "prio": 5, "needs": [D + "/data/mix/manifest.jsonl"], "est_min": 60,
             "timeout_min": 240, "machines": ["c"]}]


MIX_META = "$D/data/mix/shard*.meta.pt"
MIX_LAST = D + "/data/mix/shard11.meta.pt"
GEO_SRC = "$D/data/geo_8b.pt|/data/ibu-bi-ai-share-disk/zk/project/mm-opsd/data/processed-mv/geometry3k_train.jsonl|/data/ibu-bi-ai-share-disk/zk/project/mm-opsd/data/processed-mv"
MIX_SRC = "$D/data/mix/shard*.meta.pt|$D/data/mix/manifest.jsonl|$D/data/mix"
GEO_TEST_SRC = "$D/data/geo_8b_test.pt|/data/ibu-bi-ai-share-disk/zk/project/mm-opsd/data/processed-mv/geometry3k_test.jsonl|/data/ibu-bi-ai-share-disk/zk/project/mm-opsd/data/processed-mv"


def phase2():
    """数据规模曲线：N ∈ {1k,3k,6k,12k,21k} × {ce,kd,rel}；小规模补 seed 1/2。"""
    jobs = []
    plan = [(21000, [0]), (12000, [0]), (6000, [0, 1, 2]), (3000, [0, 1, 2]), (1000, [0, 1, 2])]
    prio_of = {21000: 20, 12000: 21, 6000: 22, 3000: 23, 1000: 24}
    for N, seeds in plan:
        steps = max(1000, 3 * N // 8)
        est = max(12, int(steps * 0.0075))
        for seed in seeds:
            for arm in ("ce", "kd", "rel"):
                name = f"p2_{arm}_n{N}_s{seed}"
                j = train_job(name, arm, MIX_META, "p2", seed=seed, steps=steps, prio=prio_of[N], est=est,
                              needs=[MIX_LAST], extra=f"--limit {N} --seq 256 --em-n 100",
                              mem={"ce": 18.0, "kd": 20.0, "rel": 21.0}[arm])
                j["needs"] = [MIX_LAST]
                j["timeout_min"] = max(est * 3, 90)
                jobs.append(j)
    return jobs


def front_job(name, mode, K, steps=5000, rank=16, est=40, mem=17.0, prio=25, bs=16):
    cmd = (f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $D/results/p3 && $PY $D/scripts/train_front.py --mode {mode} --K {K} --lora-rank {rank} "
           f"--source '{MIX_SRC}' --source '{GEO_SRC}' --eval-source '{GEO_TEST_SRC}' --steps {steps} --bs {bs} "
           f"--eval-every 1000 --workers 3 --out $D/results/p3/{name}")
    return {"name": name, "cmd": cmd, "mem_gb": mem, "prio": prio,
            "needs": [MIX_LAST, D + "/data/geo_8b_test.pt"], "est_min": est, "timeout_min": est * 4}


def phase3():
    """Stage 2A：前 K 层当内置视觉编码器，蒸馏 8B 视觉塔。"""
    return [
        front_job("p3_rot_K4", "rot", 4, steps=6000, est=45, prio=18),
        front_job("p3_lora16_K4", "lora", 4, rank=16, est=45, prio=19),
        front_job("p3_none_K0", "none", 0, est=25, mem=12.0, prio=19),
        front_job("p3_rot_K2", "rot", 2, est=35),
        front_job("p3_rot_K6", "rot", 6, est=55, mem=19.0),
        front_job("p3_rot_K8", "rot", 8, est=70, mem=21.0),
        front_job("p3_lora64_K4", "lora", 4, rank=64, est=45),
        front_job("p3_full_K4", "full", 4, est=60, mem=27.0, bs=8),
    ]


def phase2b():
    """补强：大 N 多 seed；跨学生规模（Qwen3-0.6B / Qwen3-8B）。"""
    jobs = []
    for N in (21000, 12000):
        steps = max(1000, 3 * N // 8)
        for seed in (1, 2):
            for arm in ("ce", "kd", "rel"):
                j = train_job(f"p2_{arm}_n{N}_s{seed}", arm, MIX_META, "p2", seed=seed, steps=steps,
                              prio=40, est=int(steps * 0.0075), extra=f"--limit {N} --seq 256 --em-n 100",
                              mem={"ce": 18.0, "kd": 20.0, "rel": 21.0}[arm])
                j["needs"] = [MIX_LAST]; j["timeout_min"] = max(int(steps * 0.0075) * 3, 90)
                jobs.append(j)
    # 跨规模：同一配方换学生底座（N=6000, seed 0）
    for tag, model, mem, layers, sl in (
            ("q06", "/data/ibu-bi-ai-share-disk/zk/model/Qwen3-0.6B", {"ce": 9, "kd": 11, "rel": 12},
             "0,4,8,12,16,20,24", "6,13,20,27"),
            ("q8b", "/data/ibu-bi-ai-share-disk/zk/model/Qwen3-8B", {"ce": 27, "kd": 29, "rel": 31},
             "0,4,8,12,16,20,24,28,32", "8,16,24,32")):
        for arm in ("ce", "kd", "rel"):
            j = train_job(f"p2s_{tag}_{arm}_n6000", arm, MIX_META, "p2s", steps=2250, layers=layers,
                          prio=45, est=40 if tag == "q8b" else 15, mem=float(mem[arm]),
                          extra=f"--limit 6000 --seq 256 --em-n 100 --model {model} --student-layers {sl}")
            j["needs"] = [MIX_LAST]; j["timeout_min"] = 200
            jobs.append(j)
    return jobs


def jt(name, stage, front_mode, K, front_ckpt, vrmg_ckpt, steps=0, arm="rel", prio=50, est=60, mem=20.0,
       needs=None, extra=""):
    cmd = (f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $D/results/p4 && $PY $D/scripts/train_joint.py --stage {stage} --data '{MIX_META}' "
           f"--source '{MIX_SRC}' --vrmg-ckpt {vrmg_ckpt} --front-mode {front_mode} --K {K} --arm {arm} "
           f"--seq 256 --bs 8 --steps {steps} --n-eval 600 --workers 3 --out $D/results/p4/{name}")
    if front_ckpt:
        cmd += f" --front-ckpt {front_ckpt}"
    cmd += f" {extra}"
    n = [MIX_LAST] + ([front_ckpt.replace("$D", D)] if front_ckpt else []) + [vrmg_ckpt.replace("$D", D)] + (needs or [])
    return {"name": name, "cmd": cmd, "mem_gb": mem, "prio": prio, "needs": n, "est_min": est,
            "timeout_min": est * 4}


def phase4():
    """合体：拔掉外部视觉塔。依赖 P2(rel,N=21000,s0) 的 VRMG 与 P3 的各前端。"""
    V = "$D/results/p2/p2_rel_n21000_s0_pt.pt"
    P3 = "$D/results/p3"
    return [
        jt("p4_oracle", "oracle", "none", 0, "", V, prio=50, est=10, mem=14.0),
        jt("p4_rot_K4_frozen", "front_frozen", "rot", 4, f"{P3}/p3_rot_K4.pt", V, prio=50, est=12, mem=16.0),
        jt("p4_rot_K4_joint", "joint", "rot", 4, f"{P3}/p3_rot_K4.pt", V, steps=4000, prio=48, est=80, mem=22.0),
        jt("p4_lora16_K4_joint", "joint", "lora", 4, f"{P3}/p3_lora16_K4.pt", V, steps=2500, prio=52, est=55,
           mem=22.0, extra="--lora-rank 16"),
        jt("p4_none_K0_joint", "joint", "none", 0, f"{P3}/p3_none_K0.pt", V, steps=2500, prio=52, est=40, mem=18.0),
        jt("p4_full_K4_joint", "joint", "full", 4, f"{P3}/p3_full_K4.pt", V, steps=2500, prio=53, est=70, mem=30.0),
        jt("p4_rot_K4_joint_s1", "joint", "rot", 4, f"{P3}/p3_rot_K4.pt", V, steps=4000, prio=55, est=80, mem=22.0,
           extra="--seed 1"),
    ]


def phase5():
    P3 = "$D/results/p3"
    return [
        {"name": "p5_lang",
         "cmd": f"mkdir -p $D/results/p5 && $PY $D/scripts/eval_lang.py --K 4 "
                f"--variant rot4=rot:{P3}/p3_rot_K4.pt --variant lora16=lora:{P3}/p3_lora16_K4.pt "
                f"--variant lora64=lora:{P3}/p3_lora64_K4.pt --out $D/results/p5/lang_K4.json",
         "mem_gb": 12.0, "prio": 60,
         "needs": [D + "/results/p3/p3_rot_K4.pt", D + "/results/p3/p3_lora16_K4.pt", D + "/results/p3/p3_lora64_K4.pt"],
         "est_min": 40, "timeout_min": 160},
        {"name": "p5_bench",
         "cmd": f"mkdir -p $D/results/p5 && $PY $D/scripts/bench_efficiency.py --K 4 "
                f"--front-ckpt {P3}/p3_rot_K4.pt --out $D/results/p5/bench_K4.json",
         "mem_gb": 41.5, "prio": 70, "needs": [D + "/results/p3/p3_rot_K4.pt"], "est_min": 15, "timeout_min": 60},
    ]


def phase1b():
    """填充任务（不依赖 mix）：Geometry3K 再补 seed 3、4，把 P1 主表样本量做到 5 seeds。"""
    jobs = []
    data, test = "$D/data/geo_8b.pt", "$D/data/geo_8b_test.pt"
    for seed in (3, 4):
        for arm in ("ce", "kd", "rel"):
            jobs.append(train_job(f"p1_{arm}_s{seed}", arm, data, "p1", seed=seed, test=test, prio=35))
    return jobs


def phase6():
    """后续三步（2026-11-05 追加）：OPD 在线蒸馏、projector 基线、旋转+对角缩放、1.7B 超参修正、补跑效率/语言。

    p6smoke_*_v2 是改完代码后的冒烟闸门：冒烟跑完后 check_smoke.py 检查数值（不只是没崩），通过才写 .ok；
    OPD / projector 的正式任务 needs 这个 .ok。上一轮冒烟「通过」了，正式跑却把首词准确率训到 0。
    """
    jobs = []
    pre = "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $D/results/p6 && "
    opd = (f"$PY $D/scripts/train_opd.py --data $D/data/geo_8b.pt --source '{GEO_SRC}' "
           f"--test-data $D/data/geo_8b_test.pt ")
    jobs.append({"name": "p6smoke_opd_v2",
                 "cmd": pre + opd + "--n-test 24 --vrmg-ckpt $D/results/p1/p1_ce_s0_pt.pt --steps 60 --bs 4 "
                                    "--max-new 32 --n-eval 24 --eval-every 30 --log-every 10 "
                                    "--out $D/results/p6/smoke_opd_v2 && "
                                    "$PY $D/scripts/check_smoke.py opd $D/results/p6/smoke_opd_v2.json",
                 "mem_gb": 34.0, "prio": 4,
                 "needs": [D + "/results/p1/p1_ce_s0_pt.pt", D + "/data/geo_8b.pt", D + "/data/geo_8b_test.pt"],
                 "est_min": 12, "timeout_min": 60})
    jobs.append({"name": "p6smoke_proj_v2",
                 "cmd": pre + f"$PY $D/scripts/train_proj.py --data '{MIX_META}' --source '{MIX_SRC}' --limit 120 "
                              "--steps 50 --bs 2 --n-eval 16 --n-acc 8 --workers 0 --out $D/results/p6/smoke_proj_v2 && "
                              "$PY $D/scripts/check_smoke.py proj $D/results/p6/smoke_proj_v2.json",
                 "mem_gb": 34.0, "prio": 4, "needs": [MIX_LAST], "est_min": 12, "timeout_min": 60})
    # 冒烟里 lr 1e-4 + 金标 CE 权重 0.2 下 tf_first 在 60 步内明显下滑（疑似老师在首 token 上与金标不一致，KL 把学生往老师拉），
    # 所以同时挂一组保守配置（lr 2e-5、CE 权重 1.0）对冲，不把整晚押在一个配置上
    for tag, lr, cew, prio in (("", "1e-4", "0.2", 20), ("safe_", "2e-5", "1.0", 21)):
        for s in (0, 1):
            name = f"p6_opd_geo_{tag}s{s}"
            jobs.append({"name": name,
                         "cmd": pre + opd + f"--vrmg-ckpt $D/results/p1/p1_ce_s{s}_pt.pt --steps 600 --bs 4 "
                                            f"--max-new 32 --n-eval 200 --lr {lr} --ce-weight {cew} --seed {s} "
                                            f"--out $D/results/p6/{name}",
                         "mem_gb": 34.0, "prio": prio,
                         "needs": [D + f"/results/p1/p1_ce_s{s}_pt.pt", D + "/data/geo_8b.pt",
                                   D + "/data/geo_8b_test.pt", D + "/results/p6/smoke_opd_v2.ok"],
                         "est_min": 50, "timeout_min": 200})
    # N=21000 的步数按 P2 的公式 3N/8=7875，与 ce 臂同预算（原 4500 不同预算，对比不公平）
    for tag, lim, steps, est in (("n6000", 6000, 2250, 70), ("n21000", 21000, 7875, 120)):
        jobs.append({"name": f"p6_proj_{tag}",
                     "cmd": pre + f"$PY $D/scripts/train_proj.py --data '{MIX_META}' --source '{MIX_SRC}' "
                                  f"--limit {lim} --steps {steps} --bs 8 --lr 5e-4 --out $D/results/p6/p6_proj_{tag}",
                     "mem_gb": 34.0, "prio": 22, "needs": [MIX_LAST, D + "/results/p6/smoke_proj_v2.ok"],
                     "est_min": est, "timeout_min": est * 4})
    # needs 不能留 train_job 默认的通配符路径（os.path.exists 永远为假，任务会卡在队列里）；
    # --seq 256 --em-n 100 与 P2/p2s 的 N=6000 点同口径，才能直接并排看规模曲线
    for s in (0, 1):
        j = train_job(f"p6_lr_n6000_s{s}", "ce", MIX_META, "p6", seed=s, steps=4000, prio=24, mem=18.0, est=70,
                      extra="--limit 6000 --seq 256 --em-n 100 --lr 1e-3")
        j["needs"] = [MIX_LAST]
        jobs.append(j)
    # 2A 余弦与端到端反相关，这里只当热身初始化，步数 5000 → 2500
    jobs.append({"name": "p6_rots_K4",
                 "cmd": f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $D/results/p6 && "
                        f"$PY $D/scripts/train_front.py --mode rots --K 4 --source '{MIX_SRC}' --source '{GEO_SRC}' "
                        f"--eval-source '{GEO_TEST_SRC}' --steps 2500 --bs 16 --eval-every 1000 --workers 3 "
                        f"--out $D/results/p6/p6_rots_K4",
                 "mem_gb": 32.0, "prio": 26, "needs": [MIX_LAST, D + "/data/geo_8b_test.pt"],  # rots 实测 >27GB，19 会被同卡任务挤 OOM
                 "est_min": 25, "timeout_min": 100})
    # 起点从 rel 换成 ce：规模曲线已证明 ce 全面优于 rel，且 ce 臂更省显存
    jobs.append({"name": "p6_rots_K4_joint",
                 "cmd": f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $D/results/p6 && "
                        f"$PY $D/scripts/train_joint.py --stage joint --data '{MIX_META}' --source '{MIX_SRC}' "
                        f"--vrmg-ckpt $D/results/p2/p2_ce_n21000_s0_pt.pt --front-mode rots --K 4 --arm ce "
                        f"--seq 256 --bs 8 --steps 2500 --n-eval 600 --workers 3 "
                        f"--front-ckpt $D/results/p6/p6_rots_K4.pt --out $D/results/p6/p6_rots_K4_joint",
                 "mem_gb": 30.0, "prio": 54,
                 "needs": [MIX_LAST, D + "/results/p6/p6_rots_K4.pt",
                           D + "/results/p2/p2_ce_n21000_s0_pt.pt"],
                 "est_min": 60, "timeout_min": 240})
    jobs.append({"name": "p6_lang",
                 "cmd": f"mkdir -p $D/results/p6 && $PY $D/scripts/eval_lang.py --K 2 "
                        f"--variant rot2=rot:$D/results/p3/p3_rot_K2.pt --out $D/results/p6/lang_K2.json",
                 "mem_gb": 12.0, "prio": 60, "needs": [D + "/results/p3/p3_rot_K2.pt"],
                 "est_min": 35, "timeout_min": 140})
    jobs.append({"name": "p6_bench",
                 "cmd": f"mkdir -p $D/results/p6 && $PY $D/scripts/bench_efficiency.py --K 2 "
                        f"--front-ckpt $D/results/p3/p3_rot_K2.pt --out $D/results/p6/bench_K2.json",
                 "mem_gb": 41.5, "prio": 70, "needs": [D + "/results/p3/p3_rot_K2.pt"],
                 "est_min": 15, "timeout_min": 60})
    return jobs


def phase7():
    """第七阶段（2026-11-05 下午）：RoPE 式高低频多粒度旋转 + CoT 版 OPD + 解析式效率表。"""
    jobs = []
    # A 组：高低频多粒度旋转（multifreq）
    for s in (0, 1):
        j = train_job(f"p7_mf_n6000_s{s}", "ce", MIX_META, "p7", seed=s, steps=2250, prio=10,
                      mem=18.0, est=50, extra="--limit 6000 --multifreq")
        j["needs"] = [MIX_LAST]
        jobs.append(j)
    j = train_job("p7_mf_n21000_s0", "ce", MIX_META, "p7", seed=0, steps=4500, prio=12,
                  mem=18.0, est=95, extra="--limit 21000 --multifreq")
    j["needs"] = [MIX_LAST]
    jobs.append(j)
    jobs.append({"name": "p7_mf_rots_joint",
                 "cmd": f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $D/results/p7 && "
                        f"$PY $D/scripts/train_joint.py --stage joint --data '{MIX_META}' --source '{MIX_SRC}' "
                        f"--vrmg-ckpt $D/results/p7/p7_mf_n21000_s0_pt.pt --front-mode rots --K 4 --arm ce "
                        f"--multifreq --seq 256 --bs 8 --steps 2500 --n-eval 600 --workers 3 "
                        f"--front-ckpt $D/results/p6/p6_rots_K4.pt --out $D/results/p7/p7_mf_rots_joint",
                 "mem_gb": 30.0, "prio": 30,
                 "needs": [MIX_LAST, D + "/results/p7/p7_mf_n21000_s0_pt.pt", D + "/results/p6/p6_rots_K4.pt"],
                 "est_min": 65, "timeout_min": 260})
    # B 组：CoT 冷启动 + OPD
    jobs.append({"name": "p7_prep_cot",
                 "cmd": f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $D/results/p7 && "
                        f"$PY $D/scripts/prep_geo_cot.py --out $D/data/geo_cot.pt "
                        f"> $D/results/p7/prep_cot.log 2>&1",
                 "mem_gb": 30.0, "prio": 5, "needs": [D + "/data/geo_8b.pt"], "est_min": 25, "timeout_min": 120})
    jobs.append({"name": "p7_cot_ce_s0",
                 "cmd": f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $D/results/p7 && "
                        f"$PY $D/scripts/train_final.py --arm ce --data $D/data/geo_cot.pt "
                        f"--layers 0,4,8,12,16,20,24 --block 64 --query-dim 256 --bs 8 --seq 384 --steps 2000 "
                        f"--eval-every 500 --seed 0 --em-n 0 --log $D/results/p7/p7_cot_ce_s0.log "
                        f"--result-json $D/results/p7/p7_cot_ce_s0.json",
                 "mem_gb": 20.0, "prio": 20, "needs": [D + "/data/geo_cot.pt"], "est_min": 45, "timeout_min": 180})
    for s in (0, 1):
        jobs.append({"name": f"p7_cot_opd_s{s}",
                     "cmd": f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $D/results/p7 && "
                            f"$PY $D/scripts/train_opd.py --data $D/data/geo_cot.pt --source '{GEO_SRC}' "
                            f"--vrmg-ckpt $D/results/p7/p7_cot_ce_s0_pt.pt --boxed --max-new 128 --steps 400 "
                            f"--bs 4 --lr 2e-5 --ce-weight 1.0 --seed {s} --n-eval 60 --n-test 150 "
                            f"--test-data $D/data/geo_8b_test.pt --out $D/results/p7/p7_cot_opd_s{s}",
                     "mem_gb": 34.0, "prio": 40,
                     "needs": [D + "/data/geo_cot.pt", D + "/results/p7/p7_cot_ce_s0_pt.pt",
                               D + "/data/geo_8b_test.pt"],
                     "est_min": 80, "timeout_min": 320})
    # C 组：解析式 FLOPs + 实测延迟
    jobs.append({"name": "p7_bench",
                 "cmd": f"mkdir -p $D/results/p7 && $PY $D/scripts/bench_efficiency.py --K 4 "
                        f"--front-ckpt $D/results/p3/p3_rot_K4.pt --out $D/results/p7/bench_K4.json",
                 "mem_gb": 41.5, "prio": 14, "needs": [D + "/results/p3/p3_rot_K4.pt"],
                 "est_min": 15, "timeout_min": 60})
    return jobs


def phase8():
    """第八阶段（2026-11-05 傍晚）：方案 1 (tokrot 逐词查图正交旋转) vs 方案 2 (hybrid 16 空间锚点混合)。"""
    jobs = []
    # 1. N=6000 同台对比（各 2 seeds）
    for mode, flag in (("tokrot", "--tokrot"), ("hybrid", "--hybrid")):
        for seed in (0, 1):
            j = train_job(f"p8_{mode}_n6000_s{seed}", "ce", MIX_META, "p8", seed=seed, steps=2250, prio=8,
                          mem=18.0, est=22, extra=f"--limit 6000 --seq 256 --em-n 100 {flag}")
            j["needs"] = [MIX_LAST]
            jobs.append(j)
    # 2. N=21000 全量规模对比
    for mode, flag in (("tokrot", "--tokrot"), ("hybrid", "--hybrid")):
        j = train_job(f"p8_{mode}_n21000_s0", "ce", MIX_META, "p8", seed=0, steps=4500, prio=10,
                      mem=18.0, est=40, extra=f"--limit 21000 --seq 256 --em-n 100 {flag}")
        j["needs"] = [MIX_LAST]
        jobs.append(j)
    # 3. 拔掉外部视觉塔的端到端合体版（rots_K4 内置前端 + 方案 1 / 方案 2）
    for mode, flag in (("tokrot", "--tokrot"), ("hybrid", "--hybrid")):
        jobs.append({"name": f"p8_{mode}_rots_joint",
                     "cmd": f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $D/results/p8 && "
                            f"$PY $D/scripts/train_joint.py --stage joint --data '{MIX_META}' --source '{MIX_SRC}' "
                            f"--vrmg-ckpt $D/results/p8/p8_{mode}_n21000_s0_pt.pt --front-mode rots --K 4 --arm ce "
                            f"{flag} --seq 256 --bs 8 --steps 2500 --n-eval 600 --workers 3 "
                            f"--front-ckpt $D/results/p6/p6_rots_K4.pt --out $D/results/p8/p8_{mode}_rots_joint",
                     "mem_gb": 30.0, "prio": 25,
                     "needs": [MIX_LAST, D + f"/results/p8/p8_{mode}_n21000_s0_pt.pt",
                               D + "/results/p6/p6_rots_K4.pt"],
                     "est_min": 35, "timeout_min": 180})
    return jobs


def phase9():
    """第九阶段：对「拔掉外部视觉塔的纯文模型（PatchFront + VRMG）」直接做 VL 大模型 OPD 在线蒸馏。"""
    jobs = []
    targets = [
        ("p9_opd_mf_rots", "$D/results/p7/p7_mf_rots_joint.pt"),
        ("p9_opd_tokrot_rots", "$D/results/p8/p8_tokrot_rots_joint.pt"),
        ("p9_opd_hybrid_rots", "$D/results/p8/p8_hybrid_rots_joint.pt"),
    ]
    for name, ckpt in targets:
        jobs.append({
            "name": name,
            "cmd": f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $D/results/p9 && "
                   f"$PY $D/scripts/train_joint_opd.py --data '{MIX_META}' --source '{MIX_SRC}' "
                   f"--joint-ckpt {ckpt} --steps 500 --bs 4 --max-new 16 --lr-vrmg 2e-5 --lr-front 1e-5 "
                   f"--ce-weight 0.5 --feat-weight 0.2 --out $D/results/p9/{name}",
            "mem_gb": 34.0, "prio": 32,
            "needs": [MIX_LAST, ckpt.replace("$D", D)],
            "est_min": 40, "timeout_min": 180,
        })
    return jobs


M06 = "/data/ibu-bi-ai-share-disk/zk/model/Qwen3-0.6B"
M8B = "/data/ibu-bi-ai-share-disk/zk/model/Qwen3-8B"
PRE = "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $D/results/p10 $D/results/p11 $D/results/bench_eval && "
SRC3 = f"--source '{MIX_SRC}' --source '{GEO_SRC}' --eval-source '{GEO_TEST_SRC}'"
GEO_TEST = D + "/data/geo_8b_test.pt"


def J(name, cmd, mem, prio, needs, est, wave, timeout=None):
    """needs 只允许具体文件路径（worker 用 os.path.exists 检查，带 * 会永远等待）。"""
    needs = [n.replace("$D", D) for n in needs]
    assert all("*" not in n for n in needs), f"{name}: needs 含通配符 {needs}"
    return {"name": name, "cmd": PRE + cmd, "mem_gb": mem, "prio": prio, "needs": needs, "est_min": est,
            "timeout_min": timeout or max(est * 4, 90), "wave": wave}


def phase10():
    """第十阶段（冲刺主表，Qwen3-1.7B）：无塔 LoRA 靶子、Tower+OPD、6 榜评测。wave1=无需补丁的训练；wave2=补丁后的训练；wave3=评测。"""
    jobs = []
    jobs.append(J("p10_lora16_K4",
                  f"$PY $D/scripts/train_front.py --mode lora --K 4 --lora-rank 16 {SRC3} --steps 2500 --bs 8 "
                  f"--eval-every 1000 --workers 3 --out $D/results/p10/p10_lora16_K4",
                  28.0, 6, [MIX_LAST, GEO_TEST], 35, 1))
    # P10_MORE
    return jobs


def phase11():
    """第十一阶段（跨尺寸 Qwen3-0.6B / Qwen3-8B 的完整两步流水线，主线 multifreq）。"""
    jobs = []
    sizes = {"q06": dict(model=M06, layers="0,4,8,12,16,20,24", mem_ce=14.0, est_ce=80),
             "q8b": dict(model=M8B, layers="0,5,10,15,20,25,30", mem_ce=40.0, est_ce=120)}
    for tag, s in sizes.items():
        jobs.append(J(f"p11_{tag}_tower_ce",
                      f"$PY $D/scripts/train_final.py --arm ce --data '{MIX_META}' --model {s['model']} "
                      f"--layers {s['layers']} --block 64 --query-dim 256 --bs 8 --seq 256 --steps 4500 "
                      f"--eval-every 1125 --seed 0 --em-n 100 --limit 21000 --multifreq "
                      f"--log $D/results/p11/p11_{tag}_tower_ce.log "
                      f"--result-json $D/results/p11/p11_{tag}_tower_ce.json",
                      s["mem_ce"], 5 if tag == "q8b" else 7, [MIX_LAST], s["est_ce"], 1))
    jobs.append(J("p11_q06_front_rots",
                  f"$PY $D/scripts/train_front.py --mode rots --K 4 --model {M06} {SRC3} --steps 2000 --bs 16 "
                  f"--eval-every 1000 --workers 3 --out $D/results/p11/p11_q06_front_rots",
                  22.0, 8, [MIX_LAST, GEO_TEST], 25, 1))
    # P11_MORE
    return jobs


PHASES = {"p1": phase1, "prep_mix": prep_mix, "p2": phase2, "p3": phase3, "p2b": phase2b, "p4": phase4,
          "p5": phase5, "p1b": phase1b, "p6": phase6, "p7": phase7, "p8": phase8, "p9": phase9,
          "p10": phase10, "p11": phase11}


def write_jobs(jobs, force=False):
    code = (
        "import json, os\nQ=%r\nos.makedirs(Q+'/pending', exist_ok=True)\njobs=%r\nn=0\n"
        "for j in jobs:\n    p=Q+'/pending/'+j['name']+'.json'\n"
        "    if os.path.exists(p) and not %r:\n        continue\n"
        "    json.dump(j, open(p,'w'))\n    n+=1\nprint('enqueued', n, 'of', len(jobs))\n"
        % (D + "/queue", jobs, force)
    )
    return ops.run_remote(code, "a", 120)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ph = sys.argv[1]
    jobs = PHASES[ph]()
    if "--wave" in sys.argv:
        w = int(sys.argv[sys.argv.index("--wave") + 1])
        jobs = [j for j in jobs if j.get("wave") == w]
    print([(j["name"], j["prio"], j["mem_gb"]) for j in jobs])
    print(write_jobs(jobs, "--force" in sys.argv))
