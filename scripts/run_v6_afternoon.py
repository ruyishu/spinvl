"""2026-11-13 11:30—18:00 三卡（A/B/C）针对性改进与全套验证守护脚本。

三大改进点（全部保持原始 LLM 100% 冻结、无首词禁词器、官方 ChartQA 计分、严格异图置换）：
1. Idea 2（深层切空间有界多频正交旋转 `--rot-bound 0.12`）：
   - 严禁旋转 `o_proj` 与 `down_proj`（SwiGLU 稀疏激活），仅在残差流入口 `{q, k, v, gate, up}` 施加李代数切空间有界 Cayley 旋转 $P = \tau \tanh(Z)/\sqrt{b}$；
   - 多频门控偏置初始化为 `-2.0`（从 CA-Only 流形平滑起步），并对旋转参数使用 $0.25\times$ 温和学习率。
2. Idea 1（无塔内置视觉前端 `rots K=4`）：
   - 修复联合训练中 `StaticRotScale.scale` 被 weight_decay 衰减的问题（`weight_decay=0.0`）并恢复静态旋转独立学习率；
   - 分别与 `CA-Only (A1)` 深层（`F0/F1/F2`）及有界多频旋转深层 `R3`（`S0/S1/S2`）联合训练，公平对比 `K=0` vs `LoRA K=4` vs `rots K=4`。
3. Idea 3（冻结 LLM 下的主梯度在线策略蒸馏 `VL-OPD`）：
   - 修复 `kl_weight=0.25` 被 `ce=1.0` 淹没 60 倍的问题，采用 `kl_weight=2.0, ce_weight=0.2, feat_weight=0.5, gen_temp=0.3, kd_temp=1.5`；
   - 在有效无塔起点上严格对比 `D0/M0 (仅 CE+feat)`、`D1/M1 (Offline KD)`、`D2/M2 (Forward-KL OPD)`、`D3/M3 (JSD OPD)`。
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

B = "/data/ibu-bi-ai-share-disk"
PY = f"{B}/zk/envs/mmopsd/bin/python"
PROJ = Path(f"{B}/zk/project/rose-distill")
SCRIPTS = PROJ / "scripts"
OUT = PROJ / "results" / "clean_v6"
EVAL_OUT = OUT / "eval"

DATA_SPEC = (
    f"{PROJ}/data/mix_v2/shard*.meta.pt,"
    f"{PROJ}/data/open_supp_v2/shard*.meta.pt,"
    f"{PROJ}/data/nat_supp/shard*.meta.pt"
)
SOURCES = [
    f"{PROJ}/data/mix_v2/shard*.meta.pt|{PROJ}/data/mix_v2/manifest.jsonl|{PROJ}/data/mix",
    f"{PROJ}/data/open_supp_v2/shard*.meta.pt|{PROJ}/data/open_supp_v2/manifest.jsonl|{PROJ}/data/open_supp",
    f"{PROJ}/data/nat_supp/shard*.meta.pt|{PROJ}/data/nat_supp/manifest.jsonl|{PROJ}/data/nat_supp",
]
SRC_ARGS = []
for s in SOURCES:
    SRC_ARGS.extend(["--source", s])


def log(gpu: str, msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}][AFT-GPU-{gpu}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"afternoon_gpu{gpu}.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def file_ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 100


def wait_for_file(gpu: str, p: Path, poll_sec: int = 15):
    while not file_ok(p):
        log(gpu, f"Waiting for prerequisite: {p} ...")
        time.sleep(poll_sec)
    log(gpu, f"Prerequisite ready: {p}")


def run_cmd(gpu: str, tag: str, cmd: list, done_files: list):
    if all(file_ok(Path(p)) for p in done_files):
        log(gpu, f"[SKIP] {tag} already done")
        return True
    log(gpu, f"[START] {tag}: {' '.join(str(x) for x in cmd)}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    res = subprocess.run([str(x) for x in cmd], cwd=str(PROJ), env=env)
    dt = time.time() - t0
    if res.returncode != 0:
        log(gpu, f"[FAIL] {tag} rc={res.returncode} ({dt:.1f}s)")
        return False
    log(gpu, f"[DONE] {tag} ({dt:.1f}s)")
    return True


def eval_model(gpu: str, mode: str, ckpt: str, base_name: str, full_ctrl: bool = False, with_fgen: bool = True):
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    ckpt_args = ["--ckpt", ckpt] if ckpt else []
    # 1. 6 榜真图
    run_cmd(
        gpu, f"eval:{base_name}:real",
        [PY, SCRIPTS / "eval_benchmarks.py", "--mode", mode, *ckpt_args, "--name", base_name,
         "--image-ctrl", "none", "--out-dir", EVAL_OUT],
        [EVAL_OUT / f"{base_name}.json"],
    )
    # 2. Core3 + ChartQA 异图置换 seed=101
    s101 = f"{base_name}_shuf101"
    run_cmd(
        gpu, f"eval:{s101}",
        [PY, SCRIPTS / "eval_benchmarks.py", "--mode", mode, *ckpt_args, "--name", s101,
         "--benches", "mmstar,hallusion,mathvista,chartqa", "--image-ctrl", "shuffle", "--shuffle-seed", "101",
         "--out-dir", EVAL_OUT],
        [EVAL_OUT / f"{s101}.json"],
    )
    if full_ctrl:
        for sd in (102, 103):
            sn = f"{base_name}_shuf{sd}"
            run_cmd(
                gpu, f"eval:{sn}",
                [PY, SCRIPTS / "eval_benchmarks.py", "--mode", mode, *ckpt_args, "--name", sn,
                 "--benches", "mmstar,hallusion,mathvista,chartqa", "--image-ctrl", "shuffle", "--shuffle-seed", str(sd),
                 "--out-dir", EVAL_OUT],
                [EVAL_OUT / f"{sn}.json"],
            )
        bn = f"{base_name}_blank"
        run_cmd(
            gpu, f"eval:{bn}",
            [PY, SCRIPTS / "eval_benchmarks.py", "--mode", mode, *ckpt_args, "--name", bn,
             "--image-ctrl", "blank", "--out-dir", EVAL_OUT],
            [EVAL_OUT / f"{bn}.json"],
        )
    if with_fgen:
        fn = f"{base_name}_fgen"
        run_cmd(
            gpu, f"eval:{fn}",
            [PY, SCRIPTS / "eval_benchmarks.py", "--mode", mode, *ckpt_args, "--name", fn,
             "--benches", "mmstar,hallusion,mathvista", "--force-gen", "--out-dir", EVAL_OUT],
            [EVAL_OUT / f"{fn}.json"],
        )


def train_tower_bounded(gpu: str, tag: str, extra_flags: list, seed: int, steps: int = 800, lr: str = "2e-4", init_ckpt: str = ""):
    pt_file = OUT / f"{tag}_s{seed}_pt.pt"
    json_file = OUT / f"{tag}_s{seed}.json"
    log_file = OUT / f"{tag}_s{seed}.log"
    init_args = ["--init-ckpt", init_ckpt] if init_ckpt else []
    cmd = [
        PY, SCRIPTS / "train_final.py",
        "--arm", "ce",
        "--data", DATA_SPEC,
        "--steps", str(steps),
        "--bs", "8",
        "--seq", "256",
        "--lr", lr,
        "--eval-every", str(max(steps // 2, 200)),
        "--em-n", "1000",
        "--gnd-weight", "0.0",
        "--seed", str(seed),
        *init_args,
        *extra_flags,
        "--result-json", str(json_file),
        "--log", str(log_file),
    ]
    run_cmd(gpu, f"train:{tag}_s{seed}", cmd, [pt_file, json_file])
    return pt_file


def pipeline_a():
    gpu = "A"
    # 等待此前补测脚本收尾（若有）
    while True:
        ps = subprocess.run(["pgrep", "-f", "_supp_a.py"], capture_output=True, text=True)
        if ps.returncode == 0 and ps.stdout.strip():
            log(gpu, "Waiting for _supp_a.py to finish...")
            time.sleep(10)
        else:
            break

    a1_s0_ckpt = str(OUT / "v6_tower_A1_caonly_s0_pt.pt")
    wait_for_file(gpu, Path(a1_s0_ckpt))

    # 1. 第一时间跑 R3 (有界多频 + qcond + CA, rot_bound=0.12, seed=0)，仅需 ~3.5 分钟，供 GPU-B 联合训练立刻使用！
    r3_flags = ["--rot-bound", "0.12", "--multifreq", "--qcond", "--hybrid", "--direct-hyb"]
    r2_flags = ["--rot-bound", "0.12", "--multifreq", "--hybrid", "--direct-hyb"]
    r1_flags = ["--rot-bound", "0.12", "--hybrid", "--direct-hyb"]
    r0_flags = ["--no-rot", "--hybrid", "--direct-hyb"]

    pt_r3_s0 = train_tower_bounded(gpu, "v6_tower_R3_mf_qc_rot", r3_flags, seed=0, steps=800, lr="2e-4", init_ckpt=a1_s0_ckpt)
    pt_r0_s0 = train_tower_bounded(gpu, "v6_tower_R0_caonly_cont", r0_flags, seed=0, steps=800, lr="2e-4", init_ckpt=a1_s0_ckpt)
    pt_r1_s0 = train_tower_bounded(gpu, "v6_tower_R1_srot", r1_flags, seed=0, steps=800, lr="2e-4", init_ckpt=a1_s0_ckpt)
    pt_r2_s0 = train_tower_bounded(gpu, "v6_tower_R2_mf_noqc", r2_flags, seed=0, steps=800, lr="2e-4", init_ckpt=a1_s0_ckpt)

    # 紧接着把 R3 的 seed 1, 2 也跑出来（各 ~3.5 分钟），让 GPU-B 的 S 组 seed 1, 2 无需等待！
    for sd in (1, 2):
        a1_sd_ckpt = str(OUT / f"v6_tower_A1_caonly_s{sd}_pt.pt")
        train_tower_bounded(gpu, "v6_tower_R3_mf_qc_rot", r3_flags, seed=sd, steps=800, lr="2e-4", init_ckpt=a1_sd_ckpt)

    # 2. 评测 R3_s0, R0_s0, R1_s0, R2_s0
    eval_model(gpu, "tower", str(pt_r3_s0), "v6_tower_R3_mf_qc_rot_s0", full_ctrl=True, with_fgen=True)
    eval_model(gpu, "tower", str(pt_r0_s0), "v6_tower_R0_caonly_cont_s0", full_ctrl=True, with_fgen=True)
    eval_model(gpu, "tower", str(pt_r1_s0), "v6_tower_R1_srot_s0", full_ctrl=False, with_fgen=True)
    eval_model(gpu, "tower", str(pt_r2_s0), "v6_tower_R2_mf_noqc_s0", full_ctrl=False, with_fgen=True)

    # 3. 跑 R0, R1, R2 的 seed 1, 2 并评测全部 R 组 seed 1, 2
    for sd in (1, 2):
        a1_sd_ckpt = str(OUT / f"v6_tower_A1_caonly_s{sd}_pt.pt")
        pt_r3_sd = OUT / f"v6_tower_R3_mf_qc_rot_s{sd}_pt.pt"
        pt_r0_sd = train_tower_bounded(gpu, "v6_tower_R0_caonly_cont", r0_flags, seed=sd, steps=800, lr="2e-4", init_ckpt=a1_sd_ckpt)
        pt_r1_sd = train_tower_bounded(gpu, "v6_tower_R1_srot", r1_flags, seed=sd, steps=800, lr="2e-4", init_ckpt=a1_sd_ckpt)
        pt_r2_sd = train_tower_bounded(gpu, "v6_tower_R2_mf_noqc", r2_flags, seed=sd, steps=800, lr="2e-4", init_ckpt=a1_sd_ckpt)
        eval_model(gpu, "tower", str(pt_r3_sd), f"v6_tower_R3_mf_qc_rot_s{sd}", full_ctrl=False, with_fgen=True)
        eval_model(gpu, "tower", str(pt_r0_sd), f"v6_tower_R0_caonly_cont_s{sd}", full_ctrl=False, with_fgen=True)
        eval_model(gpu, "tower", str(pt_r1_sd), f"v6_tower_R1_srot_s{sd}", full_ctrl=False, with_fgen=True)
        eval_model(gpu, "tower", str(pt_r2_sd), f"v6_tower_R2_mf_noqc_s{sd}", full_ctrl=False, with_fgen=True)

    # 4. 从零训练 2400 步的有界多频旋转 A3_bounded (seed 0, 1, 2)，直接对照原始 2400 步 A0/A1/A2/A3
    for sd in (0, 1, 2):
        pt_ab = train_tower_bounded(gpu, "v6_tower_A3_bounded", r3_flags, seed=sd, steps=2400, lr="5e-4", init_ckpt="")
        eval_model(gpu, "tower", str(pt_ab), f"v6_tower_A3_bounded_s{sd}", full_ctrl=(sd == 0), with_fgen=True)

    log(gpu, "ALL AFTERNOON GPU-A TASKS COMPLETED SUCCESSFULLY!")


def run_joint_with_deep(gpu: str, out_tag: str, fmode: str, K: int, front_ckpt: Path, vrmg_ckpt: Path,
                        extra_vrmg_flags: list, seed: int, steps: int = 1200):
    wait_for_file(gpu, front_ckpt)
    wait_for_file(gpu, vrmg_ckpt)
    joint_out = OUT / f"{out_tag}_joint_s{seed}"
    joint_pt = OUT / f"{out_tag}_joint_s{seed}.pt"
    joint_json = OUT / f"{out_tag}_joint_s{seed}.json"
    cmd = [
        PY, SCRIPTS / "train_joint.py",
        "--stage", "joint",
        "--data", DATA_SPEC,
        *SRC_ARGS,
        "--vrmg-ckpt", str(vrmg_ckpt),
        "--front-ckpt", str(front_ckpt),
        "--front-mode", fmode,
        "--K", str(K),
        "--lora-rank", "16",
        *extra_vrmg_flags,
        "--arm", "ce",
        "--bs", "8",
        "--seq", "256",
        "--steps", str(steps),
        "--lr-vrmg", "1e-4",
        "--lr-front", "1e-4",
        "--feat-weight", "1.0",
        "--gnd-weight", "0.0",
        "--n-eval", "600",
        "--n-em", "1000",
        "--seed", str(seed),
        "--out", str(joint_out),
    ]
    run_cmd(gpu, f"train_joint:{joint_out.name}", cmd, [joint_pt, joint_json])
    return joint_pt


def pipeline_b():
    gpu = "B"
    f_map = {
        "rots": ("rots", 4, "v6_free_B2_rots_K4_front"),
        "none": ("none", 0, "v6_free_B0_none_K0_front"),
        "lora": ("lora", 4, "v6_free_B1_lora_K4_front"),
    }
    ca_flags = ["--no-rot", "--hybrid", "--direct-hyb"]
    r3_flags = ["--rot-bound", "0.12", "--multifreq", "--qcond", "--hybrid", "--direct-hyb"]

    # 1. 第一优先：跑 F2 (rots K=4 + CA-Only, s0) 与 S2 (rots K=4 + Bounded-MF-Rot R3, s0)
    #    这样 GPU-C 可以在 10 分钟后立刻开始跑 D0~D3，在 20 分钟后立刻开始跑 M0~M3！
    f_rots_s0 = OUT / f"{f_map['rots'][2]}_s0.pt"
    a1_s0 = OUT / "v6_tower_A1_caonly_s0_pt.pt"
    r3_s0 = OUT / "v6_tower_R3_mf_qc_rot_s0_pt.pt"

    pt_f2_s0 = run_joint_with_deep(gpu, "v6_free_F2_rots_ca", "rots", 4, f_rots_s0, a1_s0, ca_flags, seed=0, steps=1200)
    pt_s2_s0 = run_joint_with_deep(gpu, "v6_free_S2_rots_mf", "rots", 4, f_rots_s0, r3_s0, r3_flags, seed=0, steps=1200)

    # 2. 跑 F0 (none K=0 + CA-Only, s0) 与 F1 (lora K=4 + CA-Only, s0)
    f_none_s0 = OUT / f"{f_map['none'][2]}_s0.pt"
    f_lora_s0 = OUT / f"{f_map['lora'][2]}_s0.pt"
    pt_f0_s0 = run_joint_with_deep(gpu, "v6_free_F0_none_ca", "none", 0, f_none_s0, a1_s0, ca_flags, seed=0, steps=1200)
    pt_f1_s0 = run_joint_with_deep(gpu, "v6_free_F1_lora_ca", "lora", 4, f_lora_s0, a1_s0, ca_flags, seed=0, steps=1200)

    # 3. 跑 S0 (none K=0 + R3, s0) 与 S1 (lora K=4 + R3, s0)
    pt_s0_s0 = run_joint_with_deep(gpu, "v6_free_S0_none_mf", "none", 0, f_none_s0, r3_s0, r3_flags, seed=0, steps=1200)
    pt_s1_s0 = run_joint_with_deep(gpu, "v6_free_S1_lora_mf", "lora", 4, f_lora_s0, r3_s0, r3_flags, seed=0, steps=1200)

    # 4. 评测 seed 0 的 F0/F1/F2 与 S0/S1/S2
    eval_model(gpu, "tower_free", str(pt_f2_s0), "v6_free_F2_rots_ca_s0", full_ctrl=True, with_fgen=True)
    eval_model(gpu, "tower_free", str(pt_s2_s0), "v6_free_S2_rots_mf_s0", full_ctrl=True, with_fgen=True)
    eval_model(gpu, "tower_free", str(pt_f0_s0), "v6_free_F0_none_ca_s0", full_ctrl=True, with_fgen=True)
    eval_model(gpu, "tower_free", str(pt_f1_s0), "v6_free_F1_lora_ca_s0", full_ctrl=True, with_fgen=True)
    eval_model(gpu, "tower_free", str(pt_s0_s0), "v6_free_S0_none_mf_s0", full_ctrl=True, with_fgen=True)
    eval_model(gpu, "tower_free", str(pt_s1_s0), "v6_free_S1_lora_mf_s0", full_ctrl=True, with_fgen=True)

    # 5. 补齐 S0/S1/S2 与 F0/F1/F2 的 seed 1, 2
    for sd in (1, 2):
        r3_sd = OUT / f"v6_tower_R3_mf_qc_rot_s{sd}_pt.pt"
        a1_sd = OUT / f"v6_tower_A1_caonly_s{sd}_pt.pt"
        for key, (fmode, K, fprefix) in f_map.items():
            f_ck = OUT / f"{fprefix}_s{sd}.pt"
            idx_k = {"none": "0", "lora": "1", "rots": "2"}[key]
            pt_s = run_joint_with_deep(gpu, f"v6_free_S{idx_k}_{key}_mf", fmode, K, f_ck, r3_sd, r3_flags, seed=sd, steps=1200)
            eval_model(gpu, "tower_free", str(pt_s), f"v6_free_S{idx_k}_{key}_mf_s{sd}", full_ctrl=False, with_fgen=True)
            pt_f = run_joint_with_deep(gpu, f"v6_free_F{idx_k}_{key}_ca", fmode, K, f_ck, a1_sd, ca_flags, seed=sd, steps=1200)
            eval_model(gpu, "tower_free", str(pt_f), f"v6_free_F{idx_k}_{key}_ca_s{sd}", full_ctrl=False, with_fgen=True)

    log(gpu, "ALL AFTERNOON GPU-B TASKS COMPLETED SUCCESSFULLY!")


def run_opd_from_joint(gpu: str, out_tag: str, joint_ckpt: Path, kl_mode: str, kl_w: float,
                       extra_flags: list, seed: int, steps: int = 600):
    wait_for_file(gpu, joint_ckpt)
    opd_out = OUT / f"{out_tag}_s{seed}"
    opd_pt = OUT / f"{out_tag}_s{seed}.pt"
    opd_json = OUT / f"{out_tag}_s{seed}.json"
    cmd = [
        PY, SCRIPTS / "train_joint_opd.py",
        "--data", DATA_SPEC,
        *SRC_ARGS,
        "--joint-ckpt", str(joint_ckpt),
        *extra_flags,
        "--steps", str(steps),
        "--bs", "4",
        "--accum", "2",
        "--max-new", "24",
        "--min-new", "1",
        "--gen-temp", "0.3",
        "--kd-temp", "1.5",
        "--kl-mode", kl_mode,
        "--kl-weight", str(kl_w),
        "--ce-weight", "0.2",
        "--feat-weight", "0.5",
        "--lr-vrmg", "3e-5",
        "--lr-front", "1.5e-5",
        "--n-eval", "600",
        "--n-acc", "150",
        "--n-em", "1000",
        "--seed", str(seed),
        "--out", str(opd_out),
    ]
    run_cmd(gpu, f"train_opd:{opd_out.name}", cmd, [opd_pt, opd_json])
    return opd_pt


def pipeline_c():
    gpu = "C"
    # 等待 _supp_c.py 完成
    while True:
        ps = subprocess.run(["pgrep", "-f", "_supp_c.py"], capture_output=True, text=True)
        if ps.returncode == 0 and ps.stdout.strip():
            log(gpu, "Waiting for _supp_c.py to finish...")
            time.sleep(10)
        else:
            break

    # 0. 在等待 GPU-B 产出 F2_s0 的前 8 分钟里，补齐 Base / LLaVA-Proj / A1_s0 的 ChartQA 异图对照
    for mode, ckpt, name in [
        ("proj", str(OUT / "v6_proj_llava_s0.pt"), "v6_proj_llava_s0_cq_shuf101"),
        ("tower", str(OUT / "v6_tower_A1_caonly_s0_pt.pt"), "v6_tower_A1_caonly_s0_cq_shuf101"),
    ]:
        run_cmd(
            gpu, f"eval:{name}",
            [PY, SCRIPTS / "eval_benchmarks.py", "--mode", mode, "--ckpt", ckpt, "--name", name,
             "--benches", "chartqa", "--image-ctrl", "shuffle", "--shuffle-seed", "101", "--out-dir", EVAL_OUT],
            [EVAL_OUT / f"{name}.json"],
        )

    # 1. 在 F2 (rots K=4 + CA-Only, s0) 上跑主梯度冻结 OPD 四组对照 (D0/D1/D2/D3)
    f2_s0 = OUT / "v6_free_F2_rots_ca_joint_s0.pt"
    ca_flags = ["--no-rot", "--hybrid", "--direct-hyb"]
    d_specs = [
        ("v6_opd_D0_cont_ce", "none", 0.0, False),
        ("v6_opd_D1_offkd", "offkd", 2.0, False),
        ("v6_opd_D2_fwdkl", "fwd", 2.0, True),
        ("v6_opd_D3_jsd", "jsd", 2.0, True),
    ]
    for tag, kl_m, kl_w, fctrl in d_specs:
        pt_d = run_opd_from_joint(gpu, tag, f2_s0, kl_m, kl_w, ca_flags, seed=0, steps=600)
        eval_model(gpu, "tower_free", str(pt_d), f"{tag}_s0", full_ctrl=fctrl, with_fgen=True)

    # 2. 在 S2 (rots K=4 + Bounded-MF-Rot R3, s0) 上跑全三 Idea 合体的主梯度冻结 OPD 四组对照 (M0/M1/M2/M3)
    s2_s0 = OUT / "v6_free_S2_rots_mf_joint_s0.pt"
    r3_flags = ["--rot-bound", "0.12", "--multifreq", "--qcond", "--hybrid", "--direct-hyb"]
    m_specs = [
        ("v6_opd_M0_cont_ce", "none", 0.0, False),
        ("v6_opd_M1_offkd", "offkd", 2.0, False),
        ("v6_opd_M2_fwdkl", "fwd", 2.0, True),
        ("v6_opd_M3_jsd", "jsd", 2.0, True),
    ]
    for tag, kl_m, kl_w, fctrl in m_specs:
        pt_m = run_opd_from_joint(gpu, tag, s2_s0, kl_m, kl_w, r3_flags, seed=0, steps=600)
        eval_model(gpu, "tower_free", str(pt_m), f"{tag}_s0", full_ctrl=fctrl, with_fgen=True)

    # 3. 对新主模型 S2_s0 与 R3_s0 测独立进程效率 v6_efficiency_bounded_K4.json
    r3_s0_pt = OUT / "v6_tower_R3_mf_qc_rot_s0_pt.pt"
    a0_s0_pt = OUT / "v6_tower_A0_rose_s0_pt.pt"
    proj_pt = OUT / "v6_proj_llava_s0.pt"
    eff_bounded = OUT / "v6_efficiency_bounded_K4.json"
    run_cmd(
        gpu, "bench_efficiency_bounded",
        [PY, SCRIPTS / "bench_efficiency.py",
         "--modes", "ours,tower_ours,rose,keep,llava",
         "--joint-ckpt", str(s2_s0),
         "--tower-ckpt", str(r3_s0_pt),
         "--rose-ckpt", str(a0_s0_pt),
         "--proj-ckpt", str(proj_pt),
         "--out", str(eff_bounded)],
        [eff_bounded],
    )

    # 4. 补齐 M0/M2/M3 与 D0/D2/D3 的 seed 1, 2
    for sd in (1, 2):
        s2_sd = OUT / f"v6_free_S2_rots_mf_joint_s{sd}.pt"
        for tag, kl_m, kl_w, _ in m_specs:
            pt_m_sd = run_opd_from_joint(gpu, tag, s2_sd, kl_m, kl_w, r3_flags, seed=sd, steps=600)
            eval_model(gpu, "tower_free", str(pt_m_sd), f"{tag}_s{sd}", full_ctrl=False, with_fgen=True)

    log(gpu, "ALL AFTERNOON GPU-C TASKS COMPLETED SUCCESSFULLY!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", choices=["A", "B", "C"], required=True)
    args = ap.parse_args()
    if args.gpu == "A":
        pipeline_a()
    elif args.gpu == "B":
        pipeline_b()
    else:
        pipeline_c()


if __name__ == "__main__":
    main()
